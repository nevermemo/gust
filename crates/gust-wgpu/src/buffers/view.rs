use crate::{Error, GpuBuffer, GpuBufferAccess, GpuVec};
use gust::{GpuPod, TypeLayout};
use std::marker::PhantomData;

/// Which rule of [`GpuBuffer::view_as`] a refusal broke ([`Error::BufferViewSize`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferViewRule {
    /// The buffer's bytes are not a whole number of view elements: `tail` bytes would be left over, and a view never
    /// rounds down.
    Tail {
        /// The bytes after the last whole element: the buffer's bytes modulo the element's stride.
        tail: u64,
    },
    /// The element type's layout stride differs from its Rust size, so "the same bytes" would mean two things (the
    /// element is a hand-written [`GpuPod`] whose layout is padded, or whose size disagrees with Rust's).
    Stride {
        /// The size of the element type in Rust.
        rust_size: usize,
    },
}

/// A buffer's bytes seen as another element type: the typed lens of [`GpuBuffer::view_as`] and [`GpuVec::view_as`].
///
/// A view borrows its source and is `Copy`. It binds wherever a `GpuBuffer<U>` does (a typed call, a
/// [`BufferBinding`](crate::BufferBinding) through `read_only_view` and `read_write_view`, a staged graph node), and to
/// every check GUST has it is the same buffer as its source: bound twice in one call with a writer it is
/// `Error::AliasedBinding`, and a staged graph orders a writer through a view and a reader of the source by an edge,
/// as it does for one element type. Plain successive dispatches of a raw-tier batch run in queue order with no check,
/// as two kernels over one buffer do today.
///
/// The view carries the source's access (a read-only source gives a read-only view) and is writable otherwise. It
/// has no host read or write: use the source. There is no CPU counterpart:
/// `CpuBuffer` has no `view_as` (see [`GpuBuffer::view_as`]). Uniforms and textures have no view. GPU-side clears (the
/// `clear` list of `submit_batch_read`) go by the source's bytes. `view_as` does not check that `U` is a storage
/// element: a type that is not (a uniform-only layout) is refused at dispatch by the binding-layout check.
///
/// ```
/// use gust::math::UVec4;
/// use gust_wgpu::{GpuBufferAccess, HeadlessDevice};
///
/// let device = HeadlessDevice::new()?;
/// let words = device.create_typed_buffer("words", &[1u32, 2, 3, 4, 5, 6, 7, 8], GpuBufferAccess::ReadWrite);
/// let quads = words.view_as::<UVec4>()?;
/// // Eight words are two `UVec4` elements of the same bytes.
/// assert_eq!((quads.len(), quads.layout().stride), (2, 16));
/// // Nine words are not a whole number of `UVec4`: the view refuses instead of hiding the tail.
/// let nine = device.create_typed_buffer("nine", &[0u32; 9], GpuBufferAccess::ReadWrite);
/// assert!(nine.view_as::<UVec4>().is_err());
/// # Ok::<(), gust_wgpu::Error>(())
/// ```
pub struct GpuBufferView<'a, U: GpuPod> {
    /// The source's allocation; the lifetime is the source's, so a binding built from a temporary view still borrows
    /// the source and not the view.
    pub(crate) raw: &'a wgpu::Buffer,
    pub(crate) len: usize,
    /// The bytes a binding exposes: the source's own (`logical_bytes`), a whole number of `U` for a non-empty view.
    pub(crate) bytes: u64,
    pub(crate) access: GpuBufferAccess,
    pub(crate) device_id: u64,
    pub(crate) indirect: bool,
    marker: PhantomData<fn() -> U>,
}

impl<U: GpuPod> Clone for GpuBufferView<'_, U> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<U: GpuPod> Copy for GpuBufferView<'_, U> {}

impl<U: GpuPod> std::fmt::Debug for GpuBufferView<'_, U> {
    /// Length, element and access; never the raw wgpu handle.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBufferView")
            .field("element", &U::LAYOUT.name)
            .field("len", &self.len)
            .field("access", &self.access)
            .field("device_id", &self.device_id)
            .finish_non_exhaustive()
    }
}

impl<U: GpuPod> GpuBufferView<'_, U> {
    /// The number of `U` elements: the source's bytes divided by `U`'s stride (exactly, or `view_as` refused).
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the view holds no element (its source is empty).
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The access of the source: a view is never wider.
    pub const fn access(&self) -> GpuBufferAccess {
        self.access
    }

    /// The layout of one `U` element.
    pub const fn layout(&self) -> TypeLayout {
        U::LAYOUT
    }
}

impl<T: GpuPod> GpuBuffer<T> {
    /// The same bytes as another element type `U`: a borrowed, `Copy` [`GpuBufferView`], writable when this buffer is,
    /// with the aliasing rules of one buffer and no other.
    ///
    /// Refused with [`Error::BufferViewSize`] when the buffer's bytes are not a whole number of `U` (nothing is rounded
    /// down: [`BufferViewRule::Tail`] says how many bytes are left over), or when `T` or `U` is a type whose layout
    /// stride differs from its Rust size ([`BufferViewRule::Stride`]; the scalars, vectors, matrices and arrays of
    /// `gust` are not, only a hand-written `GpuPod` can be). A buffer of no element gives a view of no element:
    /// a dispatch with work that names it is refused by the length rule, as an empty buffer is.
    /// A pooled buffer ([`GpuVec`]) is viewed over its logical length, never its size class.
    ///
    /// **GPU only, no CPU twin.** A view reinterprets bytes, which the CPU twin's typed storage cannot do without a
    /// mirror, so `CpuBuffer` has no `view_as` and a program that names a view does not compile for `CpuDevice`. It is
    /// a feature of the raw tier (`BufferBinding`, `BufferDispatch`, the staged graph) and of typed calls on
    /// [`HeadlessDevice`](crate::HeadlessDevice); the equality promise of the batch tier does not cover it, and nothing
    /// is added to `gust::batch::View`.
    pub fn view_as<U: GpuPod>(&self) -> Result<GpuBufferView<'_, U>, Error> {
        let len = view_len(
            self.len,
            T::LAYOUT,
            size_of::<T>(),
            U::LAYOUT,
            size_of::<U>(),
        )?;
        Ok(GpuBufferView {
            raw: &self.raw,
            len,
            bytes: self.logical_bytes(),
            access: self.access,
            device_id: self.device_id,
            indirect: self.indirect,
            marker: PhantomData,
        })
    }
}

impl<T: GpuPod> GpuVec<T> {
    /// The same bytes as another element type: [`GpuBuffer::view_as`] on [`as_buffer`](Self::as_buffer), over the
    /// vector's logical length.
    pub fn view_as<U: GpuPod>(&self) -> Result<GpuBufferView<'_, U>, Error> {
        self.as_buffer().view_as()
    }
}

/// The element count of a view of `source_len` elements of `source` as `target`, or the rule it breaks. Pure, so the
/// rules are tested without an adapter. The checks run in this order: the source's stride, the target's stride, the
/// tail.
fn view_len(
    source_len: usize,
    source: TypeLayout,
    source_size: usize,
    target: TypeLayout,
    target_size: usize,
) -> Result<usize, Error> {
    let bytes = source_len as u64 * u64::from(source.stride);
    for (layout, rust_size) in [(source, source_size), (target, target_size)] {
        if layout.stride as usize != rust_size {
            return Err(Error::BufferViewSize {
                buffer_bytes: bytes,
                element: layout.name,
                stride: layout.stride,
                rule: BufferViewRule::Stride { rust_size },
            });
        }
    }
    let stride = u64::from(target.stride);
    if !bytes.is_multiple_of(stride) {
        return Err(Error::BufferViewSize {
            buffer_bytes: bytes,
            element: target.name,
            stride: target.stride,
            rule: BufferViewRule::Tail {
                tail: bytes % stride,
            },
        });
    }
    Ok((bytes / stride) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gust::{ScalarKind, TypeLayoutKind, math::UVec4};

    fn view_size(
        source_len: usize,
        source: TypeLayout,
        target: TypeLayout,
    ) -> Result<usize, Error> {
        let size = |layout: TypeLayout| layout.size as usize;
        view_len(source_len, source, size(source), target, size(target))
    }

    #[test]
    fn a_whole_number_of_elements_gives_the_length_in_the_target() {
        let words = <u32 as GpuPod>::LAYOUT;
        let quad = <UVec4 as GpuPod>::LAYOUT;
        assert_eq!(view_size(8, words, quad).unwrap(), 2);
        assert_eq!(view_size(2, quad, words).unwrap(), 8);
        assert_eq!(view_size(7, words, words).unwrap(), 7);
        assert_eq!(view_size(0, words, quad).unwrap(), 0);
        assert_eq!(view_size(0, quad, words).unwrap(), 0);
    }

    #[test]
    fn a_tail_is_refused_and_named_not_rounded_down() {
        let words = <u32 as GpuPod>::LAYOUT;
        let quad = <UVec4 as GpuPod>::LAYOUT;
        for (len, tail) in [(9, 4), (10, 8), (11, 12), (1, 4)] {
            match view_size(len, words, quad).unwrap_err() {
                Error::BufferViewSize {
                    buffer_bytes,
                    element,
                    stride,
                    rule,
                } => {
                    assert_eq!(buffer_bytes, len as u64 * 4);
                    assert_eq!((element, stride), ("UVec4", 16));
                    assert_eq!(rule, BufferViewRule::Tail { tail });
                }
                other => panic!("expected BufferViewSize, got {other:?}"),
            }
        }
    }

    #[test]
    fn the_messages_say_whether_the_tail_or_the_stride_rule_failed() {
        let words = <u32 as GpuPod>::LAYOUT;
        let quad = <UVec4 as GpuPod>::LAYOUT;
        let tail = view_size(9, words, quad).unwrap_err().to_string();
        assert_eq!(
            tail,
            "a buffer of 36 bytes is not a whole number of `UVec4` elements (16 bytes each): 4 bytes would be left over, and a view does not round down",
        );
        let padded = TypeLayout {
            name: "Padded",
            size: 12,
            alignment: 4,
            stride: 16,
            kind: TypeLayoutKind::Scalar(ScalarKind::U32),
        };
        let stride = view_len(3, words, 4, padded, 12).unwrap_err().to_string();
        assert_eq!(
            stride,
            "`Padded` has a layout stride of 16 bytes but a Rust size of 12: a view needs the same bytes in Rust and on the GPU",
        );
    }

    #[test]
    fn a_source_whose_stride_differs_from_its_rust_size_is_refused_before_the_tail() {
        let words = <u32 as GpuPod>::LAYOUT;
        let padded = TypeLayout {
            name: "Padded",
            size: 12,
            alignment: 4,
            stride: 16,
            kind: TypeLayoutKind::Scalar(ScalarKind::U32),
        };
        // 3 elements of stride 16 are 48 bytes, a whole number of `u32`: only the stride rule refuses it.
        match view_len(3, padded, 12, words, 4).unwrap_err() {
            Error::BufferViewSize {
                buffer_bytes,
                element,
                stride,
                rule,
            } => {
                assert_eq!((buffer_bytes, element, stride), (48, "Padded", 16));
                assert_eq!(rule, BufferViewRule::Stride { rust_size: 12 });
            }
            other => panic!("expected BufferViewSize, got {other:?}"),
        }
    }
}
