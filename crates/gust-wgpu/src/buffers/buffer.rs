use crate::pool;
#[cfg(doc)]
use crate::{GpuVec, HeadlessDevice};
use gust::Access;
use std::marker::PhantomData;

/// Logical shader access assigned to a persistent GPU buffer.
///
/// A buffer remembers the access it was created with. A read-only buffer cannot be read back, and a binding that
/// asks for more access than the buffer has is refused too, at dispatch, with `Error::PersistentBufferAccess`:
///
/// ```
/// use gust_wgpu::{Error, GpuBufferAccess, HeadlessDevice};
///
/// let device = HeadlessDevice::new()?;
/// let input = device.create_typed_buffer("input", &[1u32, 2], GpuBufferAccess::ReadOnly);
/// let output = device.create_typed_buffer("output", &[0u32; 2], GpuBufferAccess::ReadWrite);
/// // Only a read-write buffer can be read back; a read-only one refuses.
/// assert_eq!(device.read_typed_buffer(&output)?, [0, 0]);
/// assert!(matches!(
///     device.read_typed_buffer(&input),
///     Err(Error::PersistentBufferNotReadable)
/// ));
/// assert_eq!(input.access(), GpuBufferAccess::ReadOnly);
/// # Ok::<(), gust_wgpu::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBufferAccess {
    /// Kernels only read the buffer, and the host cannot read it back.
    ReadOnly,
    /// Kernels may write the buffer, and the host can read it back.
    ReadWrite,
}

impl GpuBufferAccess {
    pub(crate) fn allows(self, access: Access) -> bool {
        matches!(
            (self, access),
            (Self::ReadOnly, Access::ReadOnly)
                | (Self::ReadWrite, Access::ReadOnly | Access::ReadWrite)
        )
    }
}

/// A device-owned, element-typed allocation that remains resident across dispatches.
///
/// Create one with [`HeadlessDevice::create_typed_buffer`], update it with `write_typed_buffer`, and read it with
/// `read_typed_buffer`:
///
/// ```
/// use gust_wgpu::{GpuBuffer, GpuBufferAccess, HeadlessDevice};
///
/// let device = HeadlessDevice::new()?;
/// let buffer: GpuBuffer<f32> =
///     device.create_typed_buffer("values", &[1.0, 2.0, 3.0], GpuBufferAccess::ReadWrite);
/// assert_eq!(buffer.len(), 3);
/// assert!(!buffer.is_empty());
/// assert_eq!(buffer.layout().size, 4);
///
/// device.write_typed_buffer(&buffer, &[4.0, 5.0, 6.0])?;
/// assert_eq!(device.read_typed_buffer(&buffer)?, [4.0, 5.0, 6.0]);
/// # Ok::<(), gust_wgpu::Error>(())
/// ```
pub struct GpuBuffer<T> {
    pub(crate) raw: wgpu::Buffer,
    pub(crate) len: usize,
    pub(crate) access: GpuBufferAccess,
    pub(crate) layout: gust::TypeLayout,
    pub(crate) device_id: u64,
    /// Created with `INDIRECT` usage by `create_indirect_buffer`.
    pub(crate) indirect: bool,
    /// The pool lease of a pooled buffer (`raw` may be larger than `len` elements; bindings use the logical
    /// length). Dropping the buffer returns the allocation to its device's pool.
    #[allow(dead_code)] // held for its `Drop`, which returns the allocation
    pub(crate) lease: Option<pool::Lease>,
    pub(crate) marker: PhantomData<T>,
}

impl<T> std::fmt::Debug for GpuBuffer<T> {
    /// Length, access and owning device; never the raw wgpu handle.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("len", &self.len)
            .field("access", &self.access)
            .field("device_id", &self.device_id)
            .field("indirect", &self.indirect)
            .finish_non_exhaustive()
    }
}

impl<T> GpuBuffer<T> {
    /// The number of elements.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds no element.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The access the buffer was created with.
    pub const fn access(&self) -> GpuBufferAccess {
        self.access
    }

    /// The layout of one element.
    pub const fn layout(&self) -> gust::TypeLayout {
        self.layout
    }

    /// Expose the underlying allocation for generated or advanced wgpu code.
    ///
    /// A pooled buffer (a [`GpuVec`]'s) is the full size class, not `len()` elements: its `arrayLength` and the bytes
    /// past the logical length are not the vector's (they may hold the previous owner's data). Bind it with
    /// `wgpu::BufferBinding { offset: 0, size: Some(len_bytes), .. }`; `as_entire_binding()` shows a kernel the whole
    /// class.
    ///
    /// The buffer returns to the pool when its owner drops, and the pool clears it for its next owner in a submission
    /// of its own. Commands you recorded in an encoder of your own and submit after the owner dropped may therefore
    /// run after that clear (or after the next owner's writes). Submit them before dropping the owner.
    pub fn as_wgpu_buffer(&self) -> &wgpu::Buffer {
        &self.raw
    }

    /// The bytes a binding of this buffer exposes: `len` elements (one element for an empty buffer, whose allocation
    /// holds a placeholder), capped by the allocation. For every buffer that is not pooled this is the whole
    /// allocation, which is what `as_entire_binding` bound before; for a pooled buffer it is the logical length, not
    /// the class size. One formula for GUST's own bindings and for [`as_binding`](Self::as_binding).
    pub(crate) fn logical_bytes(&self) -> u64 {
        let wanted = (self.len.max(1) as u64).saturating_mul(u64::from(self.layout.stride));
        wanted.min(self.raw.size())
    }

    /// The binding of the live elements, for a bind group of your own: offset 0 and `len()` elements' bytes (`stride`
    /// each), never the size class, so for a non-empty buffer a shader's `arrayLength` is `len()`. This is what GUST's
    /// own bindings expose.
    ///
    /// A buffer of no element binds one placeholder element (zeroed), as GUST's own bindings do, because wgpu refuses a
    /// zero-size binding: the shader's `arrayLength` is 1 then, so read `len()` for the count. Every element's stride
    /// is a multiple of 4 (the storage ABI of `GpuPod`), so the bound size always is.
    ///
    /// Bind it as `read-only` storage when the buffer belongs to a batch (see `gust::batch::View`).
    pub fn as_binding(&self) -> wgpu::BufferBinding<'_> {
        wgpu::BufferBinding {
            buffer: &self.raw,
            offset: 0,
            size: std::num::NonZeroU64::new(self.logical_bytes()),
        }
    }
}
