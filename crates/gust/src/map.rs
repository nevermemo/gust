//! The device side of the element-wise map calls (M7, D60): what a backend offers so that one generic algorithm can
//! upload slices, run a kernel and read the result back, and the error such a call returns.
//!
//! [`MapHost`] is implemented by `gust_wgpu::HeadlessDevice` (native) and by [`crate::cpu::CpuDevice`] (the map slice,
//! m7.a1). It extends [`TypedDevice`] with three things a map call needs and a typed `Run` does not have: allocation
//! and upload, a zeroed output, and readback. Nothing here names a backend, so `gust` depends on no GPU crate and a
//! `GUST_PROFILE=cpu` build keeps no GPU API (D21). The trait is generic, not `dyn`: the generated `map_on` is
//! generic over the device and `map` instantiates it with the build's concrete device type.
//!
//! [`Error`] is the one error type of the map calls. It is the same in every profile; the backend's own error is
//! boxed in [`Error::Device`] and recovered with [`Error::downcast_ref`].

use core::fmt;
use std::error;

use crate::{GpuPod, Resource, TypedDevice};

/// A device that can hold the buffers of a map call: upload, a zeroed output, readback.
///
/// The buffer type is the device's own typed resource, so a generated `Run` can bind it without conversion. Lengths
/// are `u32` because a map call's element count is (a longer slice is [`Error::TooLong`] before the device is
/// touched).
pub trait MapHost: TypedDevice {
    /// The device's buffer for a vector of `T`: a [`Resource`] of this very device, so a call can take it.
    type Buffer<T: GpuPod>: Resource<Element = T, Device = Self>;

    /// A buffer holding a copy of `values`. An empty slice is allowed and gives an empty buffer.
    fn upload<T: GpuPod>(&self, values: &[T]) -> Result<Self::Buffer<T>, Self::Error>;

    /// A buffer of `len` elements, every one zero (an output to be written by a kernel).
    fn zeroed<T: GpuPod>(&self, len: u32) -> Result<Self::Buffer<T>, Self::Error>;

    /// Copy the buffer's elements to the host. Blocks until the device is done with every earlier submission that
    /// wrote the buffer.
    fn readback<T: GpuPod>(&self, buffer: &Self::Buffer<T>) -> Result<Vec<T>, Self::Error>;
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
}
