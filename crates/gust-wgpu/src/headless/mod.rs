use crate::BufferPool;
use crate::ShaderChecks;
mod accessors;
mod adapter;
mod adopt;
mod open;

pub(crate) use adapter::*;
pub use adopt::DeviceRequest;

use crate::LazyTimestampProfiler;
#[cfg(all(feature = "spirv-route", not(target_arch = "wasm32")))]
use crate::RouteEvent;
use crate::cache::PipelineCache;
use gust::capability::Capability;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;

pub(crate) static NEXT_DEVICE_ID: AtomicU64 = AtomicU64::new(1);

/// A logical device selected without creating a window or presentation surface. GUST itself never touches a window,
/// hence the name; a renderer that presents can still share the device with GUST ([`HeadlessDevice::adopt`]).
///
/// Open one, create buffers on it, run a kernel and read the result back:
///
/// ```
/// use gust::gpu_module;
/// use gust_wgpu::{GpuBufferAccess, HeadlessDevice};
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
///     println!("running on {}", device.adapter_info().name);
///     let out = device.create_typed_buffer("out", &[1.0f32, 2.0, 3.0], GpuBufferAccess::ReadWrite);
///     double::run::dispatch(&device, 3, &out)?;
///     assert_eq!(device.read_typed_buffer(&out)?, [2.0, 4.0, 6.0]);
///     Ok(())
/// }
/// ```
pub struct HeadlessDevice {
    pub(crate) id: u64,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) adapter_info: wgpu::AdapterInfo,
    pub(crate) pipeline_cache: Mutex<PipelineCache>,
    pub(crate) timestamp_profiler: LazyTimestampProfiler,
    /// Limits every launch check reads: the opened device's, or lower ones installed
    /// by `with_limits`.
    pub(crate) limits: wgpu::Limits,
    /// Whether the adapter lets a fragment shader write a storage buffer (`DownlevelFlags::FRAGMENT_WRITABLE_STORAGE`),
    /// which every `pixel` kernel does (D120). A `pixel` call on an adapter without it is `Error::ShapeUnsupported`.
    pub(crate) fragment_storage: bool,
    /// The capabilities the device was opened with (`DeviceOptions::capabilities`); a kernel's `requires` must be a
    /// subset (`check_kernel_capabilities`).
    pub(crate) capabilities: &'static [Capability],
    /// The subgroup size range wgpu reports for the adapter, when it offers subgroup operations
    /// (`HeadlessDevice::subgroup_sizes`).
    pub(crate) subgroup_sizes: Option<(u32, u32)>,
    /// Whether persistent buffers are created capture-capable (`DeviceOptions::capture`): read-only storage buffers
    /// get `COPY_SRC` too.
    pub(crate) capture: bool,
    /// Whether the WGSL shader modules are built with wgpu's runtime checks (`DeviceOptions::unchecked_kernels`); a
    /// device that was adopted is always checked. Read once, by `HeadlessDevice::wgsl_module`.
    pub(crate) shader_checks: ShaderChecks,
    /// The runtime checks the last WGSL module of this device was built with (test builds only): what the unit test of
    /// the module call reads.
    #[cfg(test)]
    pub(crate) last_module_checks: Mutex<Option<wgpu::ShaderRuntimeChecks>>,
    /// Recycles the allocations of [`GpuVec`] and other pooled buffers.
    pub(crate) pool: BufferPool,
    /// How many queue submissions GUST itself has made on this device (`HeadlessDevice::gust_submissions`). Only
    /// `HeadlessDevice::submit_counted` adds to it, and every `queue.submit` of this crate goes through that.
    pub(crate) submissions: AtomicU64,
    /// The route each kernel's pipeline was built from, in build order (`HeadlessDevice::route_events`).
    #[cfg(all(feature = "spirv-route", not(target_arch = "wasm32")))]
    pub(crate) route_events: Mutex<Vec<RouteEvent>>,
}

impl std::fmt::Debug for HeadlessDevice {
    /// The device's id, the adapter's name and backend, and the capabilities; never the raw wgpu handles.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeadlessDevice")
            .field("id", &self.id)
            .field("adapter", &self.adapter_info.name)
            .field("backend", &self.adapter_info.backend)
            .field("capabilities", &self.capabilities)
            .finish_non_exhaustive()
    }
}
