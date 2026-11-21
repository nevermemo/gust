use std::fmt;

/// Why a `gust-algo` call failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// `min` or `max` of an empty input: there is no value to return. (`sum` of an empty input is the identity.)
    Empty,
    /// The count buffer of a gather holds this many elements; it must hold exactly one.
    Count(u32),
    /// The output buffer given to a `_into` call holds `actual` elements; it must hold `expected`, the input's length.
    Length {
        /// The number of elements the output must hold.
        expected: u32,
        /// The number of elements it holds.
        actual: u32,
    },
    /// The device refused: a limit (`Error::DeviceLimit`, reported before any dispatch for an input past the storage
    /// binding limit), an allocation (`Error::OutOfMemory`), mixed devices, or a failed dispatch or readback.
    Device(gust_wgpu::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("min or max of an empty input"),
            Self::Count(len) => write!(
                formatter,
                "the count buffer of a gather holds {len} elements, it must hold exactly one"
            ),
            Self::Length { expected, actual } => write!(
                formatter,
                "the output buffer holds {actual} elements, it must hold {expected}"
            ),
            Self::Device(error) => write!(formatter, "device error: {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Empty | Self::Count(_) | Self::Length { .. } => None,
            Self::Device(error) => Some(error),
        }
    }
}

impl From<gust_wgpu::Error> for Error {
    fn from(error: gust_wgpu::Error) -> Self {
        Self::Device(error)
    }
}
