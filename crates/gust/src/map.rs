//! The device side of the element-wise map calls (M7, D60): what a backend offers so that one generic algorithm can
//! upload slices, run a kernel and read the result back, and the error such a call returns.
//!
//! [`MapHost`] is implemented by `gust_wgpu::HeadlessDevice` (native and `wasm32`) and by
//! [`crate::cpu::CpuDevice`] (the map slice, m7.a1).
//! It extends [`TypedDevice`] with three things a map call needs and a typed `Run` does not have: allocation
//! and upload, a zeroed output, and readback. Nothing here names a backend, so `gust` depends on no GPU crate and a
//! `GUST_PROFILE=cpu` build keeps no GPU API (D21). The trait is generic, not `dyn`: the generated `map_on` is
//! generic over the device and `map` instantiates it with the build's concrete device type.
//!
//! [`Error`] is the one error type of the map calls. It is the same in every profile; the backend's own error is
//! boxed in [`Error::Device`] and recovered with [`Error::downcast_ref`].

use core::fmt;
use std::error;

use core::borrow::Borrow;

use crate::{GpuPod, Resource, TypedDevice};

/// A device that can hold the buffers of a map call: upload, a zeroed output, readback.
///
/// The backend's error is bounded (`Error + Send + Sync + 'static`) so the generic map calls can box it into
/// [`Error::Device`] without further where clauses.
///
/// The buffer type is the device's own typed resource, so a generated `Run` can bind it without conversion. Lengths
/// are `u32` because a map call's element count is (a longer slice is [`Error::TooLong`] before the device is
/// touched).
pub trait MapHost: TypedDevice<Error: error::Error + Send + Sync + 'static> {
    /// The device's buffer for a vector of `T`: a [`Resource`] of this very device, so a call can take it.
    type Buffer<T: GpuPod>: Resource<Element = T, Device = Self, Kind = crate::kind::Storage>;

    /// A buffer holding a copy of `values`. An empty slice is allowed and gives an empty buffer.
    fn upload<T: GpuPod>(&self, values: &[T]) -> Result<Self::Buffer<T>, Self::Error>;

    /// A buffer of `len` elements, every one zero (an output to be written by a kernel).
    fn zeroed<T: GpuPod>(&self, len: u32) -> Result<Self::Buffer<T>, Self::Error>;

    /// Copy the buffer's elements to the host. Blocks until the device is done with every earlier submission that
    /// wrote the buffer. Not available on `wasm32`, where nothing may block (D69): use
    /// [`readback_async`](Self::readback_async).
    #[cfg(not(target_arch = "wasm32"))]
    fn readback<T: GpuPod>(&self, buffer: &Self::Buffer<T>) -> Result<Vec<T>, Self::Error>;

    /// [`readback`](Self::readback) without blocking, on every target. In the browser a plain `.await` completes it;
    /// on a native GPU device the future completes only while something polls the device (each poll of the future
    /// polls it once), so a native caller drives it in a loop.
    fn readback_async<T: GpuPod>(
        &self,
        buffer: &Self::Buffer<T>,
    ) -> impl core::future::Future<Output = Result<Vec<T>, Self::Error>>;

    /// The map [`Error`] of a failure of this device. The default boxes it as [`Error::Device`]; a backend overrides
    /// it to report the causes that have a variant of their own (the GPU backend maps resident buffers of two devices
    /// to [`Error::MixedDevices`]).
    fn map_error(error: Self::Error) -> Error {
        Error::Device(Box::new(error))
    }
}

/// The error of a map call. Profile-stable: every build reports the same variants, whichever device ran.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A slice-shaped argument has a length different from the others. Nothing ran. `parameter` is the kernel
    /// parameter, `expected` the length of the first slice-shaped argument and `found` this one's.
    LengthMismatch {
        parameter: &'static str,
        expected: usize,
        found: usize,
    },
    /// The element count does not fit `u32`, the type of a dispatch's element count. Nothing ran.
    TooLong { len: usize },
    /// The call has no slice-shaped argument to take the element count from (every argument is a broadcast scalar).
    Empty,
    /// Resident arguments of one call belong to different devices, or to a device other than the one named.
    MixedDevices,
    /// The device failed: allocation, upload, dispatch or readback. Holds the backend's own error; see
    /// [`downcast_ref`](Self::downcast_ref).
    Device(Box<dyn error::Error + Send + Sync + 'static>),
}

impl Error {
    /// Wrap a backend error.
    pub fn device(error: impl error::Error + Send + Sync + 'static) -> Self {
        Self::Device(Box::new(error))
    }

    /// The backend's error as `E`, when this is [`Error::Device`] holding one (for the GPU backend,
    /// `downcast_ref::<gust_wgpu::Error>()`).
    pub fn downcast_ref<E: error::Error + 'static>(&self) -> Option<&E> {
        match self {
            Self::Device(error) => error.downcast_ref::<E>(),
            _ => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                parameter,
                expected,
                found,
            } => write!(
                formatter,
                "argument `{parameter}` has {found} elements, but the other slice arguments have {expected}",
            ),
            Self::TooLong { len } => write!(
                formatter,
                "{len} elements do not fit a dispatch's 32-bit element count",
            ),
            Self::Empty => write!(
                formatter,
                "the call has no slice argument to take its length from",
            ),
            Self::MixedDevices => write!(
                formatter,
                "the resident arguments belong to different devices; every one must live on the device that runs the call",
            ),
            Self::Device(error) => write!(formatter, "the device failed: {error}"),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Device(error) => Some(&**error),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The arguments of a map call (m7.a1).
// ---------------------------------------------------------------------------------------------------------------

/// The workgroup size of every generated map kernel.
pub const WORKGROUP_SIZE: u32 = 256;

/// The number of elements one round of a generated map kernel covers: the 1D dispatch ceiling at workgroup size 256
/// (65 535 workgroups, D36). A longer call runs the same kernel with `CEILING` invocations and `ceil(n / CEILING)`
/// rounds (the grid-stride form of spike m7.s1), so no call has an element limit below `u32::MAX`.
pub const CEILING: u32 = 65_535 * WORKGROUP_SIZE;

/// A scalar that stands for a whole vector argument of a map call: every element of the call sees this one value.
///
/// `saxpy::map(Broadcast(2.0), &xs, &ys)` is `saxpy(2.0, xs[i], ys[i])` for every `i`. It has no length of its own, so
/// a call made of broadcasts alone has no element count ([`Error::Empty`]). On the device it is a one-element buffer
/// that the kernel reads at index 0 (a buffer of length 1 is read at index 0 whatever the element index is; a call
/// with other arguments longer than 1 element cannot mistake a real vector for it, because those must all have the
/// call's length).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Broadcast<T>(pub T);

/// A buffer that is already on the device of a map call and has a length: a `GpuVec` of the GPU backend, a
/// `GpuBuffer`, a `CpuBuffer`. `&R` is a [`MapArg`] for every such `R`, bound without a copy.
pub trait Resident: Resource<Kind = crate::kind::Storage> {
    /// The number of elements.
    fn resident_len(&self) -> usize;
}

impl<T: GpuPod> Resident for crate::cpu::CpuBuffer<T> {
    fn resident_len(&self) -> usize {
        self.len()
    }
}

/// The host type of a map function's parameter or result, and the type of the elements of its buffer.
///
/// Every [`GpuPod`] type stands for itself; `bool` is held as a `u32` word (0 or 1) in a buffer. A [`MapArg`] is keyed
/// on the host type, not on the element, so a `bool` parameter takes `bool`s and a `u32` parameter takes `u32`s (Rust
/// would refuse the other call too). Macro support; sealed, so the set of host types (and the shape of [`MapArg`]'s
/// bound) can change.
#[doc(hidden)]
pub trait MapElement: sealed::Sealed + Copy + 'static {
    /// The element of the buffer that holds values of this type.
    type Element: GpuPod;
}

impl<T: GpuPod> MapElement for T {
    type Element = T;
}

impl MapElement for bool {
    type Element = u32;
}

/// The seal of [`MapElement`]: private, so no other crate can name it and so none can implement the trait.
mod sealed {
    pub trait Sealed {}

    impl<T: super::GpuPod> Sealed for T {}

    impl Sealed for bool {}
}

/// One argument of a generated map call, for a function parameter of host type `H` on device `D`.
///
/// Implemented for host slices (`&[T]`, `&Vec<T>`, `&[T; N]`: uploaded through the device), for [`Broadcast<T>`] (a
/// one-element upload) and for `&R` of every [`Resident`] buffer of the device (bound directly). A `bool` parameter
/// takes `&[bool]`, `&Vec<bool>`, `&[bool; N]` and `Broadcast<bool>` (converted to 0/1 words) and never the `u32`
/// forms of those; a resident 0/1 vector is a `GpuVec<u32>`, the buffer element of `bool` (a resident buffer is typed
/// by its element, so it serves a `bool` and a `u32` parameter alike).
pub trait MapArg<H: MapElement, D: MapHost> {
    /// The typed resource the kernel is called with.
    type Res: Resource<Element = H::Element, Device = D, Kind = crate::kind::Storage>;
    /// What [`bind`](Self::bind) returns and keeps alive for the call: the uploaded buffer, or a borrowed resident one.
    type Bound: Borrow<Self::Res>;

    /// The length of a slice-shaped argument, `None` for a [`Broadcast`].
    fn shape(&self) -> Option<usize>;

    /// Upload (or borrow) the argument on `device`.
    fn bind(self, device: &D) -> Result<Self::Bound, Error>;
}

fn upload_values<T: GpuPod, D: MapHost>(device: &D, values: &[T]) -> Result<D::Buffer<T>, Error> {
    device.upload(values).map_err(D::map_error)
}

fn upload_words<D: MapHost>(
    device: &D,
    values: impl Iterator<Item = bool>,
) -> Result<D::Buffer<u32>, Error> {
    let words: Vec<u32> = values.map(u32::from).collect();
    device.upload(&words).map_err(D::map_error)
}

impl<T: GpuPod, D: MapHost> MapArg<T, D> for &[T] {
    type Res = D::Buffer<T>;
    type Bound = D::Buffer<T>;

    fn shape(&self) -> Option<usize> {
        Some(self.len())
    }

    fn bind(self, device: &D) -> Result<D::Buffer<T>, Error> {
        upload_values(device, self)
    }
}

impl<T: GpuPod, D: MapHost> MapArg<T, D> for &Vec<T> {
    type Res = D::Buffer<T>;
    type Bound = D::Buffer<T>;

    fn shape(&self) -> Option<usize> {
        Some(self.len())
    }

    fn bind(self, device: &D) -> Result<D::Buffer<T>, Error> {
        upload_values(device, self)
    }
}

impl<T: GpuPod, D: MapHost, const N: usize> MapArg<T, D> for &[T; N] {
    type Res = D::Buffer<T>;
    type Bound = D::Buffer<T>;

    fn shape(&self) -> Option<usize> {
        Some(N)
    }

    fn bind(self, device: &D) -> Result<D::Buffer<T>, Error> {
        upload_values(device, self)
    }
}

impl<T: GpuPod, D: MapHost> MapArg<T, D> for Broadcast<T> {
    type Res = D::Buffer<T>;
    type Bound = D::Buffer<T>;

    fn shape(&self) -> Option<usize> {
        None
    }

    fn bind(self, device: &D) -> Result<D::Buffer<T>, Error> {
        upload_values(device, &[self.0])
    }
}

impl<D: MapHost> MapArg<bool, D> for &[bool] {
    type Res = D::Buffer<u32>;
    type Bound = D::Buffer<u32>;

    fn shape(&self) -> Option<usize> {
        Some(self.len())
    }

    fn bind(self, device: &D) -> Result<D::Buffer<u32>, Error> {
        upload_words(device, self.iter().copied())
    }
}

impl<D: MapHost> MapArg<bool, D> for &Vec<bool> {
    type Res = D::Buffer<u32>;
    type Bound = D::Buffer<u32>;

    fn shape(&self) -> Option<usize> {
        Some(self.len())
    }

    fn bind(self, device: &D) -> Result<D::Buffer<u32>, Error> {
        upload_words(device, self.iter().copied())
    }
}

impl<D: MapHost, const N: usize> MapArg<bool, D> for &[bool; N] {
    type Res = D::Buffer<u32>;
    type Bound = D::Buffer<u32>;

    fn shape(&self) -> Option<usize> {
        Some(N)
    }

    fn bind(self, device: &D) -> Result<D::Buffer<u32>, Error> {
        upload_words(device, self.iter().copied())
    }
}

impl<D: MapHost> MapArg<bool, D> for Broadcast<bool> {
    type Res = D::Buffer<u32>;
    type Bound = D::Buffer<u32>;

    fn shape(&self) -> Option<usize> {
        None
    }

    fn bind(self, device: &D) -> Result<D::Buffer<u32>, Error> {
        upload_words(device, core::iter::once(self.0))
    }
}

impl<'a, H: MapElement, D: MapHost, R> MapArg<H, D> for &'a R
where
    R: Resident + Resource<Element = H::Element, Device = D>,
{
    type Res = R;
    type Bound = &'a R;

    fn shape(&self) -> Option<usize> {
        Some(self.resident_len())
    }

    fn bind(self, _device: &D) -> Result<&'a R, Error> {
        Ok(self)
    }
}

/// The element count of a call from its arguments' shapes, in parameter order: the first slice-shaped argument gives
/// the length, every later one must equal it ([`Error::LengthMismatch`], naming the later parameter), no slice-shaped
/// argument at all is [`Error::Empty`], and a length above `u32::MAX` is [`Error::TooLong`]. Nothing has touched a
/// device when this returns. Macro support.
#[doc(hidden)]
pub fn element_count(shapes: &[(&'static str, Option<usize>)]) -> Result<u32, Error> {
    let mut first: Option<usize> = None;
    for &(parameter, shape) in shapes {
        let Some(found) = shape else {
            continue;
        };
        match first {
            None => first = Some(found),
            Some(expected) if expected != found => {
                return Err(Error::LengthMismatch {
                    parameter,
                    expected,
                    found,
                });
            }
            Some(_) => {}
        }
    }
    let len = first.ok_or(Error::Empty)?;
    u32::try_from(len).map_err(|_| Error::TooLong { len })
}

/// The typed resource behind a bound argument. Macro support.
#[doc(hidden)]
pub fn resource<H: MapElement, D: MapHost, A: MapArg<H, D>>(bound: &A::Bound) -> &A::Res {
    bound.borrow()
}

/// Marker, stamped on the kernel of every map function by the generator: the kernel writes only `out[i]` for its own
/// element index `i`, reads every other buffer at `i` or at 0 (a broadcast), and has no side effect besides that
/// write. It is the property that makes a split of the index range across threads race-free; the chunked parallel
/// CPU execution of M7 (m7.c) is defined for stamped kernels only, and no other kernel carries the marker. Macro
/// support.
#[doc(hidden)]
pub trait OwnIndex: crate::Kernel {}

impl MapHost for crate::cpu::CpuDevice {
    type Buffer<T: GpuPod> = crate::cpu::CpuBuffer<T>;

    fn upload<T: GpuPod>(
        &self,
        values: &[T],
    ) -> Result<crate::cpu::CpuBuffer<T>, crate::cpu::Error> {
        Ok(self.buffer(values))
    }

    fn zeroed<T: GpuPod>(&self, len: u32) -> Result<crate::cpu::CpuBuffer<T>, crate::cpu::Error> {
        // SAFETY: `GpuPod` is implemented only for types with no invalid bit pattern and no padding, so the all-zero
        // pattern is a valid `T` (the contract of the trait's safety section).
        #[allow(unsafe_code)]
        let zero: T = unsafe { core::mem::zeroed() };
        Ok(self.buffer(&vec![zero; len as usize]))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn readback<T: GpuPod>(
        &self,
        buffer: &crate::cpu::CpuBuffer<T>,
    ) -> Result<Vec<T>, crate::cpu::Error> {
        Ok(buffer.read())
    }

    async fn readback_async<T: GpuPod>(
        &self,
        buffer: &crate::cpu::CpuBuffer<T>,
    ) -> Result<Vec<T>, crate::cpu::Error> {
        Ok(buffer.read())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A backend error type for the tests.
    #[derive(Debug, PartialEq)]
    struct Backend(&'static str);

    impl fmt::Display for Backend {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "backend says {}", self.0)
        }
    }

    impl error::Error for Backend {}

    #[test]
    fn the_messages_name_what_is_wrong() {
        assert_eq!(
            Error::LengthMismatch {
                parameter: "y",
                expected: 3,
                found: 5,
            }
            .to_string(),
            "argument `y` has 5 elements, but the other slice arguments have 3",
        );
        assert_eq!(
            Error::TooLong { len: 5_000_000_000 }.to_string(),
            "5000000000 elements do not fit a dispatch's 32-bit element count",
        );
        assert_eq!(
            Error::Empty.to_string(),
            "the call has no slice argument to take its length from",
        );
        assert!(
            Error::MixedDevices
                .to_string()
                .starts_with("the resident arguments belong to different devices")
        );
    }

    #[test]
    fn a_device_error_keeps_its_type_and_source() {
        let error = Error::device(Backend("out of memory"));
        assert_eq!(
            error.to_string(),
            "the device failed: backend says out of memory"
        );
        assert_eq!(
            error.downcast_ref::<Backend>(),
            Some(&Backend("out of memory"))
        );
        assert!(error.downcast_ref::<fmt::Error>().is_none());
        let source = error::Error::source(&error).expect("a device error has a source");
        assert_eq!(source.to_string(), "backend says out of memory");
    }

    #[test]
    fn the_other_variants_hold_no_backend_error() {
        for error in [
            Error::Empty,
            Error::MixedDevices,
            Error::TooLong { len: 1 },
            Error::LengthMismatch {
                parameter: "x",
                expected: 1,
                found: 2,
            },
        ] {
            assert!(error.downcast_ref::<Backend>().is_none());
            assert!(error::Error::source(&error).is_none());
        }
    }

    #[test]
    fn the_error_crosses_threads() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<Error>();
    }

    #[test]
    fn the_ceiling_is_65535_workgroups_of_256() {
        // The macro's generated kernel and host functions spell this number themselves
        // (`crates/gust-macros/src/expand/map.rs`); this pins both to one value.
        assert_eq!(WORKGROUP_SIZE, 256);
        assert_eq!(CEILING, 16_776_960);
    }

    #[test]
    fn the_first_shaped_argument_gives_the_count() {
        assert_eq!(
            element_count(&[("a", None), ("x", Some(7)), ("y", Some(7))]).unwrap(),
            7
        );
        // A zero length is a count, not an error: an empty call returns an empty result.
        assert_eq!(element_count(&[("x", Some(0))]).unwrap(), 0);
    }

    #[test]
    fn a_later_length_must_equal_the_first() {
        let Err(Error::LengthMismatch {
            parameter,
            expected,
            found,
        }) = element_count(&[("a", None), ("x", Some(3)), ("y", Some(5)), ("z", Some(9))])
        else {
            panic!("a mismatch is an error");
        };
        // The first mismatch in parameter order, naming the later parameter.
        assert_eq!((parameter, expected, found), ("y", 3, 5));
    }

    #[test]
    fn no_shaped_argument_is_empty() {
        assert!(matches!(element_count(&[]), Err(Error::Empty)));
        assert!(matches!(
            element_count(&[("a", None), ("b", None)]),
            Err(Error::Empty)
        ));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn a_count_above_u32_is_too_long() {
        let len = u32::MAX as usize + 1;
        assert!(matches!(
            element_count(&[("x", Some(len))]),
            Err(Error::TooLong { len: found }) if found == len
        ));
        assert_eq!(
            element_count(&[("x", Some(u32::MAX as usize))]).unwrap(),
            u32::MAX
        );
    }

    #[test]
    fn slices_arrays_and_vecs_are_shaped_and_a_broadcast_is_not() {
        type Cpu = crate::cpu::CpuDevice;
        let device = Cpu::new();
        let values = [1.0f32, 2.0, 3.0];
        let vec = values.to_vec();
        assert_eq!(MapArg::<f32, Cpu>::shape(&&values[..]), Some(3));
        assert_eq!(MapArg::<f32, Cpu>::shape(&&vec), Some(3));
        assert_eq!(MapArg::<f32, Cpu>::shape(&&values), Some(3));
        assert_eq!(MapArg::<f32, Cpu>::shape(&Broadcast(2.0f32)), None);
        // A broadcast binds as a buffer of one element, which the kernel reads at index 0.
        let bound = MapArg::<f32, Cpu>::bind(Broadcast(2.0f32), &device).unwrap();
        assert_eq!(bound.read(), [2.0]);
    }

    #[test]
    fn bool_arguments_are_u32_words() {
        type Cpu = crate::cpu::CpuDevice;
        let device = Cpu::new();
        let flags = [true, false, true];
        assert_eq!(MapArg::<bool, Cpu>::shape(&&flags[..]), Some(3));
        let bound = MapArg::<bool, Cpu>::bind(&flags[..], &device).unwrap();
        assert_eq!(bound.read(), [1, 0, 1]);
        let bound = MapArg::<bool, Cpu>::bind(Broadcast(true), &device).unwrap();
        assert_eq!(bound.read(), [1]);
        let bound = MapArg::<bool, Cpu>::bind(&vec![false, true], &device).unwrap();
        assert_eq!(bound.read(), [0, 1]);
        let bound = MapArg::<bool, Cpu>::bind(&[true; 2], &device).unwrap();
        assert_eq!(bound.read(), [1, 1]);
    }

    #[test]
    fn a_resident_buffer_is_shaped_by_its_length_and_bound_as_it_is() {
        type Cpu = crate::cpu::CpuDevice;
        let device = Cpu::new();
        let buffer = device.buffer(&[4u32, 5, 6, 7]);
        assert_eq!(MapArg::<u32, Cpu>::shape(&&buffer), Some(4));
        let bound = MapArg::<u32, Cpu>::bind(&buffer, &device).unwrap();
        let resource = resource::<u32, Cpu, &crate::cpu::CpuBuffer<u32>>(&bound);
        assert_eq!(resource.read(), [4, 5, 6, 7]);
    }

    #[test]
    fn the_cpu_device_is_a_map_host() {
        let device = crate::cpu::CpuDevice::new();
        let uploaded = device.upload(&[1.5f32, 2.5]).unwrap();
        assert_eq!(device.readback(&uploaded).unwrap(), [1.5, 2.5]);
        let zeroed = device.zeroed::<u32>(5).unwrap();
        assert_eq!(device.readback(&zeroed).unwrap(), [0; 5]);
        let none = device.zeroed::<f32>(0).unwrap();
        assert!(device.readback(&none).unwrap().is_empty());
        // The default `map_error` wraps the backend's error as a device error that keeps its type.
        let wrapped =
            <crate::cpu::CpuDevice as MapHost>::map_error(crate::cpu::Error::ElementType {
                parameter: "x",
            });
        assert!(matches!(wrapped, Error::Device(_)));
        assert!(wrapped.downcast_ref::<crate::cpu::Error>().is_some());
    }
}
