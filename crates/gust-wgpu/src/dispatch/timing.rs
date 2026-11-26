use std::sync::atomic::{AtomicU32, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use crate::HeadlessDevice;

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

#[cfg(not(target_arch = "wasm32"))]
pub(crate) const TIMESTAMP_RESULT_BYTES: u64 = 2 * size_of::<u64>() as u64;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct TimestampProfiler {
    pub(crate) query_set: wgpu::QuerySet,
    pub(crate) resolve_buffer: wgpu::Buffer,
    pub(crate) readback_buffer: wgpu::Buffer,
}

/// The device's timestamp profiler, made at the first timed dispatch, batch or graph run and never at open: an
/// untimed device creates no timestamp objects. Shaped like the `Option` it replaced: `is_some` says the device has
/// timestamp queries, `as_ref` gives the profiler, making it on the first call. The profiler exists on native targets
/// only.
pub(crate) struct LazyTimestampProfiler {
    #[cfg(not(target_arch = "wasm32"))]
    device: wgpu::Device,
    supported: bool,
    #[cfg(not(target_arch = "wasm32"))]
    profiler: OnceLock<Mutex<TimestampProfiler>>,
    /// How many times the profiler was made, counted inside `TimestampProfiler::new`.
    made: AtomicU32,
}

impl LazyTimestampProfiler {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        // From the device's own features, which are what it was granted, not from what the adapter offers.
        let supported = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            device: device.clone(),
            supported,
            #[cfg(not(target_arch = "wasm32"))]
            profiler: OnceLock::new(),
            made: AtomicU32::new(0),
        }
    }

    /// Whether the device has timestamp queries (nothing is made by asking).
    pub(crate) const fn is_some(&self) -> bool {
        self.supported
    }

    /// The profiler, made on the first call; `None` when the device has no timestamp queries.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn as_ref(&self) -> Option<&Mutex<TimestampProfiler>> {
        if !self.supported {
            return None;
        }
        Some(
            self.profiler
                .get_or_init(|| Mutex::new(TimestampProfiler::new(&self.device, &self.made))),
        )
    }
}

impl HeadlessDevice {
    /// How many times this device made its timestamp profiler: 0 until the first timed dispatch batch or graph run on
    /// a device with timestamp queries, then 1. Test support, outside the pre-1.0 promise (D111).
    #[doc(hidden)]
    #[must_use]
    pub fn timestamp_profiler_made(&self) -> u32 {
        self.timestamp_profiler.made.load(Ordering::Relaxed)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl TimestampProfiler {
    /// The one place the profiler's objects are made; `made` counts each making (as `GpuTiming::new` counts its own).
    fn new(device: &wgpu::Device, made: &AtomicU32) -> Self {
        made.fetch_add(1, Ordering::Relaxed);
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
