use crate::cache::PipelineCache;
#[cfg(doc)]
use crate::default_device;
use crate::{
    ADAPTER_ATTEMPTS, ADAPTER_BACKOFF, DeviceOptions, Error, HeadlessDevice, NEXT_DEVICE_ID,
    TimestampProfiler, pause_between_attempts, retry_no_adapter,
};
use crate::{AdapterChoice, BufferPool, FeaturesPolicy, LimitsPolicy, device};
use gust::capability::Capability;
use std::sync::Mutex;
use std::sync::atomic::Ordering;

impl HeadlessDevice {
    /// Select a high-performance Vulkan adapter and open a logical device.
    ///
    /// A missing adapter is retried a bounded number of times with a short backoff,
    /// because the Vulkan loader can fail transiently when many devices are opened at
    /// once. After the last attempt the `NoAdapter` error is returned; there is no skip and
    /// no runtime fallback to the CPU twin. The adapter is whatever wgpu offers for Vulkan with a
    /// high-performance preference: a software Vulkan adapter (lavapipe, llvmpipe) is
    /// accepted like a hardware one, because the adapter type is not checked.
    ///
    /// This blocks the calling thread (`pollster::block_on` over [`new_async`](Self::new_async)); it is
    /// not available on `wasm32`, where the browser cannot be blocked.
    ///
    /// The device requests wgpu's default limits, the portable ones that a browser always gets, and no optional
    /// features. [`DeviceOptions::default()`](DeviceOptions) differs on native: it raises
    /// `max_buffer_size`, the storage binding size and the workgroups per dimension to the adapter's own
    /// ([`LimitsPolicy::Adapter`]) and asks for `float32-filterable` when the adapter offers it, and so does
    /// [`default_device`]. Use [`with_options`](Self::with_options) to get those on a device of your own.
    ///
    /// It requests the capabilities this build was compiled for ([`gust::capability::enabled`], the ones
    /// `GUST_PROFILE` named: none under `dev` or `native`, subgroups under `dev+subgroups`), as
    /// [`DeviceOptions::default()`](DeviceOptions) does, so that a `+subgroups` binary, which holds no portable body, can
    /// run its kernels here. An adapter that lacks one makes the open fail with `Error::MissingCapability`; use
    /// `with_options(DeviceOptions::default().with_capabilities(&[]))` to open without.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self, Error> {
        pollster::block_on(Self::new_async(wgpu::Backends::VULKAN))
    }

    /// Select a high-performance adapter among `backends` and open a logical device, with the retry of
    /// [`new_async_with`](Self::new_async_with).
    ///
    /// On native the retry pauses by sleeping the calling thread, which blocks an executor thread that awaits
    /// this; call it from a thread that may block, or use `new`. In the browser there is no pause between
    /// attempts. Opening the device completes without anyone polling it: wgpu drives the adapter and device
    /// requests itself, unlike the readbacks.
    ///
    /// Like `new`, it requests the capabilities of this build ([`gust::capability::enabled`]). In a browser a
    /// non-empty set is refused with `Error::MissingCapability`: the pinned wgpu cannot request subgroups there.
    pub async fn new_async(backends: wgpu::Backends) -> Result<Self, Error> {
        Self::new_async_with(DeviceOptions {
            backends,
            adapter: AdapterChoice::HighPerformance,
            limits: LimitsPolicy::Default,
            features: FeaturesPolicy::Default,
            capabilities: gust::capability::enabled(),
        })
        .await
    }

    /// Open the device `options` describe, with the adapter retry of
    #[cfg_attr(not(target_arch = "wasm32"), doc = "[`new`](Self::new).")]
    #[cfg_attr(target_arch = "wasm32", doc = "`new`.")]
    /// Unlike `new`, which is
    /// fixed to Vulkan, the high-performance adapter, wgpu's default limits and no optional feature, the options
    /// choose the backends, the adapter, whether the adapter's own buffer limits are requested and whether
    /// `float32-filterable` is enabled where the adapter offers it, so that a linear sampler can be bound
    /// ([`float32_filterable`](Self::float32_filterable) reports the result; see [`DeviceOptions`]).
    pub async fn new_async_with(options: DeviceOptions) -> Result<Self, Error> {
        let adapter = retry_no_adapter(
            ADAPTER_ATTEMPTS,
            &ADAPTER_BACKOFF,
            pause_between_attempts,
            |line| eprintln!("{line}"),
            || Self::request_adapter(options.backends, &options.adapter),
        )
        .await?;
        Self::open(adapter, &options).await
    }

    /// Request the device `options` describe from `adapter` (`DeviceOptions::device_request` says what is asked) and
    /// wrap it.
    async fn open(adapter: wgpu::Adapter, options: &DeviceOptions) -> Result<Self, Error> {
        // A capability the adapter lacks ends the open here (D23's startup failure), before the device is requested.
        // In a browser nothing is requestable: a non-empty set is refused there with the same error (ruling 5).
        let request = options.device_request(&adapter)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("GUST headless device"),
                required_features: request.features,
                required_limits: request.limits,
                ..Default::default()
            })
            .await
            .map_err(|error| Error::RequestDevice(error.to_string()))?;
        Ok(Self::finish(&adapter, device, queue, options.capabilities))
    }

    /// The one place a `HeadlessDevice` is built, for a device GUST opened and for one the caller opened (`adopt`):
    /// what GUST reads from the device and the adapter, whoever requested the device.
    pub(super) fn finish(
        adapter: &wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        capabilities: &'static [Capability],
    ) -> Self {
        let adapter_info = adapter.get_info();
        let subgroup_sizes = device::adapter_subgroup_sizes(adapter.features(), &adapter_info);
        // From the device's own features, which are what it was granted, not from what the adapter offers.
        let timestamp_profiler = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| Mutex::new(TimestampProfiler::new(&device)));
        let limits = device.limits();
        let fragment_storage = adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::FRAGMENT_WRITABLE_STORAGE);

        Self {
            id: NEXT_DEVICE_ID.fetch_add(1, Ordering::Relaxed),
            device,
            queue,
            adapter_info,
            pipeline_cache: Mutex::new(PipelineCache::default()),
            timestamp_profiler,
            limits,
            fragment_storage,
            capabilities,
            subgroup_sizes,
            pool: BufferPool::new(),
            #[cfg(all(feature = "spirv-route", not(target_arch = "wasm32")))]
            route_events: Mutex::new(Vec::new()),
        }
    }
}
