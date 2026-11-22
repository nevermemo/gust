use std::time::Duration;

/// Host-observed time for an explicit transfer boundary.
///
/// The `_timed` transfer functions return one (they do not exist on `wasm32`):
///
/// ```
/// use gust_wgpu::{GpuBufferAccess, HeadlessDevice};
///
/// let device = HeadlessDevice::new()?;
/// let buffer = device.create_typed_buffer("values", &[0u32; 4], GpuBufferAccess::ReadWrite);
/// let timing = device.write_typed_buffer_timed(&buffer, &[1, 2, 3, 4])?;
/// assert_eq!(timing.bytes, 16);
/// let (values, read) = device.read_typed_buffer_timed(&buffer)?;
/// assert_eq!(values, [1, 2, 3, 4]);
/// assert_eq!(read.bytes, 16);
/// # Ok::<(), gust_wgpu::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct TransferTiming {
    /// Number of application bytes transferred.
    pub bytes: u64,
    /// Wall-clock time including encoding, submission, synchronization, and host copies.
    pub elapsed: Duration,
}

/// Host and device measurements for one synchronous persistent submission.
///
/// ```
/// use gust::gpu_module;
/// use gust_wgpu::{BufferBinding, BufferDispatch, GpuBufferAccess, HeadlessDevice};
///
/// #[gpu_module]
/// mod double {
///     #[kernel(workgroup_size(64, 1, 1))]
///     pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) {
///         let i = id.x;
///         if i < out.len() {
///             out[i] = out[i] * 2.0;
///         }
///     }
/// }
///
/// fn main() -> Result<(), gust_wgpu::Error> {
///     let device = HeadlessDevice::new()?;
///     let out = device.create_typed_buffer("out", &[1.0f32, 2.0], GpuBufferAccess::ReadWrite);
///     let bindings = [BufferBinding::read_write(&out)];
///     let timing = device.dispatch_batch_timed(&[BufferDispatch::new(&double::run::DESCRIPTOR, 2, &bindings)])?;
///     // The device time is there only on an adapter with timestamp queries.
///     assert_eq!(timing.gpu_elapsed.is_some(), device.supports_gpu_timestamps());
///     assert_eq!(device.read_typed_buffer(&out)?, [2.0, 4.0]);
///     Ok(())
/// }
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DispatchTiming {
    /// Wall-clock time including bind-group/command encoding, submission, and synchronization.
    pub host_elapsed: Duration,
    /// Hardware timestamp duration around the submitted compute work, when supported.
    pub gpu_elapsed: Option<Duration>,
}

pub(crate) const TIMESTAMP_RESULT_BYTES: u64 = 2 * size_of::<u64>() as u64;

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) struct TimestampProfiler {
    pub(crate) query_set: wgpu::QuerySet,
    pub(crate) resolve_buffer: wgpu::Buffer,
    pub(crate) readback_buffer: wgpu::Buffer,
}

impl TimestampProfiler {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GUST dispatch timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GUST timestamp resolve"),
            size: TIMESTAMP_RESULT_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GUST timestamp readback"),
            size: TIMESTAMP_RESULT_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            query_set,
            resolve_buffer,
            readback_buffer,
        }
    }
}
