//! Device options, the default device, and the fallible buffer allocations (m7.a0, `docs` plan section 2.2).
//!
//! [`DeviceOptions`] choose how a [`HeadlessDevice`] is opened: the backends, which adapter, and whether the adapter's
//! own buffer limits are requested. [`default_device`] opens one device per process from the options set by
//! [`set_default_options`], for the APIs that take no device (`GpuVec::from_slice`, the element-wise map calls of
//! M7). The allocation functions at the end of this file check `max_buffer_size` before any wgpu call and complete an
//! out-of-memory error scope, so a request the adapter cannot serve is an [`Error`], never a panic.

use std::future::Future;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

use gust::{GpuPod, capability::Capability, pod_slice_as_bytes};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{
    ADAPTER_ATTEMPTS, ADAPTER_BACKOFF, Error, GpuBuffer, GpuBufferAccess, HeadlessDevice,
    assert_pod_matches_storage_layout, pause_between_attempts, retry_empty_list,
};

/// The handle a resident buffer holds to its device: `Arc<HeadlessDevice>` on native, where the device is `Send` and
/// `Sync`, and `Rc<HeadlessDevice>` on `wasm32`, where wgpu's WebGPU device is neither (checked under `check-web`,
/// m7.a0 S3). `GpuVec` is `Send + Sync` on native only.
///
/// `DeviceHandle::new` wraps a device you opened, and cloning the handle shares it:
///
/// ```
/// use gust_wgpu::{DeviceHandle, HeadlessDevice};
///
/// let handle = DeviceHandle::new(HeadlessDevice::new()?);
/// let shared = DeviceHandle::clone(&handle);
/// assert_eq!(shared.id(), handle.id());
/// # Ok::<(), gust_wgpu::Error>(())
/// ```
#[cfg(not(target_arch = "wasm32"))]
pub type DeviceHandle = Arc<HeadlessDevice>;
/// The handle a resident buffer holds to its device: `Rc<HeadlessDevice>` on `wasm32` (see the native alias).
///
/// `DeviceHandle::new` wraps a device you opened, and cloning the handle shares it (on `wasm32` the device comes from
/// `HeadlessDevice::new_async`, since `new` does not exist there):
///
/// ```
/// # use gust_wgpu::HeadlessDevice;
/// use gust_wgpu::DeviceHandle;
///
/// # fn share(device: HeadlessDevice) {
/// let handle = DeviceHandle::new(device);
/// let shared = DeviceHandle::clone(&handle);
/// assert_eq!(shared.id(), handle.id());
/// # }
/// ```
#[cfg(target_arch = "wasm32")]
pub type DeviceHandle = std::rc::Rc<HeadlessDevice>;

/// Which adapter [`HeadlessDevice::new_async_with`] opens.
///
/// The default is the high-performance adapter; `NameContains` picks by name, ignoring letter case:
///
/// ```
/// use gust_wgpu::{AdapterChoice, DeviceOptions};
///
/// assert_eq!(AdapterChoice::default(), AdapterChoice::HighPerformance);
/// let options =
///     DeviceOptions::default().with_adapter(AdapterChoice::NameContains("nvidia".to_owned()));
/// assert_eq!(options.adapter, AdapterChoice::NameContains("nvidia".to_owned()));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum AdapterChoice {
    /// wgpu's high-performance preference (a discrete GPU when there is one). The default.
    #[default]
    HighPerformance,
    /// wgpu's low-power preference (an integrated GPU when there is one).
    LowPower,
    /// The `n`th adapter of [`HeadlessDevice::enumerate_async`] for the same backends.
    Index(usize),
    /// The first adapter whose name contains this text, ignoring letter case.
    NameContains(String),
}

/// Which limits [`HeadlessDevice::new_async_with`] asks the adapter for.
///
/// `LimitsPolicy::default()` is [`Default`](Self::Default), wgpu's portable limits. It is not what
/// `DeviceOptions::default().limits` holds: on native that is [`Adapter`](Self::Adapter), so the process's
/// [`default_device`] allows storage bindings up to the adapter's size (2 GiB on the adapters measured in S7.6),
/// where the browser allows 128 MiB. A program that must behave the same on both sets
/// [`Default`](Self::Default) through [`set_default_options`].
///
/// ```
/// use gust_wgpu::{DeviceOptions, HeadlessDevice, LimitsPolicy};
///
/// assert_eq!(LimitsPolicy::default(), LimitsPolicy::Default);
/// let options = DeviceOptions::default().with_limits(LimitsPolicy::Default);
/// assert_eq!(options.limits, LimitsPolicy::Default);
/// let device = HeadlessDevice::with_options(options)?;
/// // `Default` asks for wgpu's default storage binding size, 128 MiB.
/// assert_eq!(device.limits().max_storage_buffer_binding_size, 128 << 20);
/// # Ok::<(), gust_wgpu::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum LimitsPolicy {
    /// wgpu's default limits (what
    #[cfg_attr(not(target_arch = "wasm32"), doc = "[`HeadlessDevice::new`]")]
    #[cfg_attr(target_arch = "wasm32", doc = "`HeadlessDevice::new`")]
    /// requests): 256 MiB buffers, 128 MiB storage bindings,
    /// 65 535 workgroups per dimension. Portable; the browser always uses it.
    #[default]
    Default,
    /// The adapter's own `max_buffer_size`, `max_storage_buffer_binding_size` and
    /// `max_compute_workgroups_per_dimension`, every other limit at wgpu's default. [`HeadlessDevice::limits`]
    /// reports what was granted, and `with_limits` can still lower them.
    Adapter,
}

impl LimitsPolicy {
    /// The limits to put in the device descriptor for `adapter`.
    pub(crate) fn required_limits(self, adapter: &wgpu::Adapter) -> wgpu::Limits {
        let mut limits = wgpu::Limits::default();
        if self == Self::Adapter {
            let available = adapter.limits();
            limits.max_buffer_size = available.max_buffer_size;
            limits.max_storage_buffer_binding_size = available.max_storage_buffer_binding_size;
            limits.max_compute_workgroups_per_dimension =
                available.max_compute_workgroups_per_dimension;
        }
        limits
    }
}

/// Which optional device features [`HeadlessDevice::new_async_with`] asks the adapter for (slice mx.t2).
///
/// The only feature GUST asks for is `float32-filterable`, which a linear filter of a 32-bit float texture needs
/// (decision D64). A device that did not enable it still runs every texture kernel, with `Nearest` samplers and
/// `load`; binding a `Linear` sampler there is `Error::TextureFilter`. [`HeadlessDevice::float32_filterable`] reports
/// what the opened device actually has.
///
/// Like [`LimitsPolicy`] (decision D90), the policy is a field of [`DeviceOptions`]:
/// `DeviceOptions::default().features`
/// is [`Adapter`](Self::Adapter) on native and [`Default`](Self::Default) in the browser, so [`default_device`] and
/// `with_options(DeviceOptions::default())` open the same device, and [`set_default_options`] governs the feature
/// of [`default_device`].
#[cfg_attr(not(target_arch = "wasm32"), doc = "[`HeadlessDevice::new`]")]
#[cfg_attr(target_arch = "wasm32", doc = "`HeadlessDevice::new`")]
/// and [`new_async`](HeadlessDevice::new_async) use
/// [`Default`](Self::Default), as they use wgpu's default limits.
///
/// The enum is `#[non_exhaustive]`: a later release may add a policy, so a `match` outside this crate needs a wildcard
/// arm.
///
/// With [`Default`](Self::Default) the device enables no optional feature, so `float32-filterable` is off:
///
/// ```
/// use gust_wgpu::{DeviceOptions, FeaturesPolicy, HeadlessDevice};
///
/// assert_eq!(FeaturesPolicy::default(), FeaturesPolicy::Default);
/// let options = DeviceOptions::default().with_features(FeaturesPolicy::Default);
/// let device = HeadlessDevice::with_options(options)?;
/// assert!(!device.float32_filterable());
/// # Ok::<(), gust_wgpu::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum FeaturesPolicy {
    /// No optional feature: wgpu's default features, what the browser always has.
    #[default]
    Default,
    /// `float32-filterable` when the adapter offers it (measured on both adapters of this project, X-S1), nothing
    /// else. An adapter without it opens as under [`Default`](Self::Default); the device never fails to open for
    /// want of it.
    Adapter,
}

impl FeaturesPolicy {
    /// The features to put in the device descriptor for `adapter`.
    pub(crate) fn required_features(self, adapter: &wgpu::Adapter) -> wgpu::Features {
        match self {
            Self::Default => wgpu::Features::empty(),
            Self::Adapter => adapter.features() & wgpu::Features::FLOAT32_FILTERABLE,
        }
    }
}

/// Whether wgpu's runtime checks stay on in the shader modules of a device (m14.pftrust1).
///
/// Every kernel of a device is built through naga with wgpu's runtime checks: bounds checks on buffer accesses, a bound
/// on every loop, and guards on integer division and remainder. That is [`Checked`](Self::Checked), the default, and
/// the only value a safe program can reach. [`Unchecked`](Self::Unchecked) builds the WGSL route's modules with those
/// checks off, which `unsafe` code asks for through [`DeviceOptions::unchecked_kernels`]; read its `# Safety` section
/// before using it. [`HeadlessDevice::shader_checks`] reports what an open device has.
///
/// The setting belongs to the device, not to a kernel, so the pipeline cache needs no extra key. It applies to the WGSL
/// route only: a SPIR-V passthrough module (the `spirv-route` feature) has no wgpu checks in either case. A device
/// adopted with [`HeadlessDevice::adopt`] is always `Checked`. On `wasm32` the setting is accepted and ignored: the
/// browser validates and bounds-checks every shader itself.
///
/// The enum is `#[non_exhaustive]`, so a `match` outside this crate needs a wildcard arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ShaderChecks {
    /// wgpu's runtime checks are on for every kernel of the device.
    #[default]
    Checked,
    /// wgpu's runtime checks are off for every kernel of the device. Set only by the `unsafe`
    /// [`DeviceOptions::unchecked_kernels`].
    Unchecked,
}

impl ShaderChecks {
    /// The wgpu runtime checks a shader module of a device with this setting is built with: all on for
    /// [`Checked`](Self::Checked), all off for [`Unchecked`](Self::Unchecked). The one place the setting is turned into
    /// wgpu's.
    pub(crate) fn runtime_checks(self) -> wgpu::ShaderRuntimeChecks {
        match self {
            Self::Checked => wgpu::ShaderRuntimeChecks::checked(),
            Self::Unchecked => wgpu::ShaderRuntimeChecks::unchecked(),
        }
    }
}

/// How to open a device: the backends, the adapter, the limits and the optional features. Explicit and overridable;
/// there is no workload policy in it.
///
/// `DeviceOptions::default()` is what [`default_device`] opens: Vulkan (the browser's WebGPU on `wasm32`), the
/// high-performance adapter, and on native the adapter's own limits ([`LimitsPolicy::Adapter`], S7.6 outcome A) and
/// optional features ([`FeaturesPolicy::Adapter`], slice mx.t2). It
/// differs from
#[cfg_attr(not(target_arch = "wasm32"), doc = "[`HeadlessDevice::new`],")]
#[cfg_attr(target_arch = "wasm32", doc = "`HeadlessDevice::new`,")]
/// which keeps requesting wgpu's default limits, and from
/// `LimitsPolicy::default()`, which is [`LimitsPolicy::Default`]: native `default_device()` allows storage bindings
/// of the adapter's size (2 GiB measured), the browser's only 128 MiB.
///
/// The struct is `#[non_exhaustive]`, so a later release can add an option without breaking a caller: build one from
/// [`DeviceOptions::default()`](Default::default) and the `with_*` methods, which set one field each
/// (`DeviceOptions::default().with_adapter(AdapterChoice::LowPower)`). The fields stay public to read and to assign
/// on a value you already hold, except the shader checks, which only the `unsafe`
/// [`unchecked_kernels`](Self::unchecked_kernels) sets.
///
/// ```
/// use gust_wgpu::{AdapterChoice, DeviceOptions, FeaturesPolicy, HeadlessDevice, LimitsPolicy};
///
/// let options = DeviceOptions::default()
///     .with_adapter(AdapterChoice::HighPerformance)
///     .with_limits(LimitsPolicy::Default)
///     .with_features(FeaturesPolicy::Default);
/// assert_eq!(options.adapter, AdapterChoice::HighPerformance);
///
/// // A field of a value you hold can be assigned too.
/// let mut again = options.clone();
/// again.adapter = AdapterChoice::Index(0);
/// assert_ne!(again, options);
///
/// let device = HeadlessDevice::with_options(options)?;
/// assert!(!device.adapter_info().name.is_empty());
/// # Ok::<(), gust_wgpu::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceOptions {
    /// The wgpu backends to search for an adapter.
    pub backends: wgpu::Backends,
    /// Which adapter to open among those the backends offer.
    pub adapter: AdapterChoice,
    /// Which limits to request from the adapter.
    pub limits: LimitsPolicy,
    /// Which optional features to request from the adapter.
    pub features: FeaturesPolicy,
    /// The capabilities (M8, D23) the device is opened with; a kernel's `requires` must be a subset. By default the set
    /// this build was compiled for, [`gust::capability::enabled`] (what `GUST_PROFILE=dev+subgroups` named): empty
    /// unless the profile named a capability, so a `+subgroups` binary, which holds no portable body, can run its
    /// kernels on a device it opens by default. `with_capabilities(&[])` opens a device without.
    /// A capability the adapter does not offer is [`Error::MissingCapability`] at open, and no device exists: it is
    /// never dropped silently (D142), and in a browser (`wasm32`) any non-empty set is refused the same way. The set is
    /// separate from [`features`](Self::features), which holds the optional features GUST asks for on its own.
    pub capabilities: &'static [Capability],
    /// Whether the device allocates its buffers so that a debugging capture can copy them (default `false`). A
    /// read-only storage buffer is created without `COPY_SRC`, so it cannot be read back; with this option it gets
    /// `COPY_SRC`, and a call that binds one can be captured (the generated `capture` of a kernel, which copies a
    /// call's inputs for replay on the CPU twin). Read-write buffers, `GpuVec`s, pool buffers, indirect buffers and
    /// uniforms are readable either way. It costs nothing at run time beyond the extra usage flag, but it asks the
    /// driver for a buffer a read-only binding does not need, so it is off unless you are debugging. It applies to
    /// buffers created after the device is opened, on a device GUST opens (a device adopted with
    /// [`HeadlessDevice::adopt`] has it off).
    pub capture: bool,
    /// Whether wgpu's runtime checks stay on in the device's shader modules. Private: it is set only by the `unsafe`
    /// [`unchecked_kernels`](Self::unchecked_kernels) and read with [`shader_checks`](Self::shader_checks).
    shader_checks: ShaderChecks,
    /// The number of storage buffers per shader stage the device is opened with, or `None` (the default) for wgpu's 8.
    /// Set it with [`with_storage_buffers`](Self::with_storage_buffers); the number is yours to state, valid under
    /// either [`LimitsPolicy`].
    pub storage_buffers: Option<u32>,
}

impl Default for DeviceOptions {
    /// Vulkan (WebGPU on `wasm32`), the high-performance adapter and, on native, the adapter's own `max_buffer_size`,
    /// storage binding size and workgroups per dimension ([`LimitsPolicy::Adapter`]) and `float32-filterable` when
    /// offered ([`FeaturesPolicy::Adapter`]); this is what [`default_device`] opens. The capabilities are the build's
    /// own ([`gust::capability::enabled`]): none unless `GUST_PROFILE` named one.
    #[cfg_attr(not(target_arch = "wasm32"), doc = "[`HeadlessDevice::new`]")]
    #[cfg_attr(target_arch = "wasm32", doc = "`HeadlessDevice::new`")]
    /// differs in the limits and features only: it requests wgpu's default limits and no optional features, so its
    /// buffer sizes and workgroup counts are the portable ones, and the same capabilities. `HeadlessDevice::new` does
    /// not exist on `wasm32`; there these options ask for the default limits and no optional features.
    fn default() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let (backends, limits, features) = (
            wgpu::Backends::VULKAN,
            LimitsPolicy::Adapter,
            FeaturesPolicy::Adapter,
        );
        #[cfg(target_arch = "wasm32")]
        let (backends, limits, features) = (
            wgpu::Backends::BROWSER_WEBGPU,
            LimitsPolicy::Default,
            FeaturesPolicy::Default,
        );
        Self {
            backends,
            adapter: AdapterChoice::HighPerformance,
            limits,
            features,
            capabilities: gust::capability::enabled(),
            capture: false,
            shader_checks: ShaderChecks::Checked,
            storage_buffers: None,
        }
    }
}

/// The `max_storage_buffers_per_shader_stage` to request for `requested` (the option
/// [`DeviceOptions::with_storage_buffers`]), given the adapter's own value `available` and its name `adapter`; `None`
/// leaves wgpu's default. `web` is true on `wasm32`.
///
/// A browser device always opens with 8, so a number there is refused by name, as is one below 8 (`with_limits` is the
/// tool that lowers). An adapter that offers fewer than `n` is [`Error::DeviceLimit`]: the device is never opened with
/// less than the developer stated.
pub(crate) fn required_storage_buffers(
    requested: Option<u32>,
    available: u32,
    adapter: &str,
    web: bool,
) -> Result<Option<u32>, Error> {
    let Some(n) = requested else {
        return Ok(None);
    };
    if web {
        return Err(Error::RequestDevice(format!(
            "`with_storage_buffers({n})` is not available in a browser: browser devices open with 8"
        )));
    }
    if n < 8 {
        return Err(Error::RequestDevice(format!(
            "`with_storage_buffers({n})` must be at least 8; `HeadlessDevice::with_limits` lowers the limits of an open device"
        )));
    }
    if n > available {
        return Err(Error::DeviceLimit {
            // `DeviceLimit` names the operation in `kernel` for a limit met outside a launch.
            kernel: format!("device open ({adapter}, with_storage_buffers({n}))"),
            limit: "max_storage_buffers_per_shader_stage",
            requested: u64::from(n),
            maximum: u64::from(available),
        });
    }
    Ok(Some(n))
}

impl DeviceOptions {
    /// These options with the wgpu backends to search for an adapter replaced by `backends`.
    #[must_use]
    pub fn with_backends(mut self, backends: wgpu::Backends) -> Self {
        self.backends = backends;
        self
    }

    /// These options with the adapter choice replaced by `adapter`.
    #[must_use]
    pub fn with_adapter(mut self, adapter: AdapterChoice) -> Self {
        self.adapter = adapter;
        self
    }

    /// These options with the limits policy replaced by `limits`. Not [`HeadlessDevice::with_limits`], which lowers
    /// the limits of a device that is already open.
    #[must_use]
    pub fn with_limits(mut self, limits: LimitsPolicy) -> Self {
        self.limits = limits;
        self
    }

    /// These options with the optional-features policy replaced by `features`.
    #[must_use]
    pub fn with_features(mut self, features: FeaturesPolicy) -> Self {
        self.features = features;
        self
    }

    /// These options with the capability set replaced by `capabilities`: the device is opened with exactly these, and
    /// opening it fails with [`Error::MissingCapability`] when the adapter lacks one. A kernel whose `requires` names a
    /// capability runs only on a device opened with it.
    #[must_use]
    pub fn with_capabilities(mut self, capabilities: &'static [Capability]) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// These options with the storage buffers per shader stage set to `n` (decision D84, amended by m14.lim): the
    /// device request carries `max_storage_buffers_per_shader_stage = n`, so a kernel that binds up to `n` storage
    /// buffers in one stage launches on it. Off by default: without it every device opens with wgpu's 8, whatever the
    /// [`LimitsPolicy`], and the portable floor of a `limits = "portable"` crate stays 8.
    ///
    /// The number is the developer's, not the adapter's own value, so a kernel validated here says in the source what
    /// it needs. An adapter that offers fewer than `n` is [`Error::DeviceLimit`] at open, and no device exists; `n`
    /// below 8 is refused by name ([`Error::RequestDevice`]: [`HeadlessDevice::with_limits`] lowers); in a browser
    /// every `n` is refused, because browser devices open with 8. [`device_request`](Self::device_request) carries
    /// `n`, so a renderer that opens the device itself gets it, and [`HeadlessDevice::adopt`] still never raises a
    /// limit: a kernel over an adopted device's limit is [`Error::DeviceLimit`] at launch.
    /// [`HeadlessDevice::with_limits`] can lower the limit below `n` and cannot raise it above.
    ///
    /// The kernel is no longer portable: it does not run on a default device or in a browser.
    ///
    /// ```no_run
    /// use gust_wgpu::{DeviceOptions, HeadlessDevice};
    ///
    /// let device = HeadlessDevice::with_options(DeviceOptions::default().with_storage_buffers(16))?;
    /// assert_eq!(device.limits().max_storage_buffers_per_shader_stage, 16);
    /// # Ok::<(), gust_wgpu::Error>(())
    /// ```
    #[must_use]
    pub fn with_storage_buffers(mut self, n: u32) -> Self {
        self.storage_buffers = Some(n);
        self
    }

    /// These options with capture-capable allocation switched to `capture`: see [`capture`](Self::capture).
    ///
    /// ```
    /// use gust_wgpu::DeviceOptions;
    ///
    /// assert!(!DeviceOptions::default().capture);
    /// assert!(DeviceOptions::default().with_capture(true).capture);
    /// ```
    #[must_use]
    pub fn with_capture(mut self, capture: bool) -> Self {
        self.capture = capture;
        self
    }

    /// These options with wgpu's runtime shader checks turned off for every kernel of the device (see
    /// [`ShaderChecks::Unchecked`]). The default is [`ShaderChecks::Checked`]; there is no safe way to ask for this.
    ///
    /// The device applies it to the WGSL route, to kernels compiled at run time and to precompiled WGSL artifacts
    /// alike. A device adopted with [`HeadlessDevice::adopt`] does not take options, so it stays checked. On `wasm32`
    /// the option is accepted and ignored: the browser checks every shader itself. The SPIR-V route is not affected: its
    /// modules have no wgpu checks either way, and a build with the `spirv-route` feature still emits its own guards
    /// into the kernels, which this option neither adds nor removes.
    ///
    /// The setting covers every kernel the device builds, GUST's own library kernels included. Options passed to
    /// [`set_default_options`], or to any other device-wide default, make the process default device unchecked: every
    /// call that names no device, every `gust-algo` operation and every batch stage then runs unchecked, including
    /// those that index with values read from data. The promise in `# Safety` covers all of them.
    ///
    /// Measured on loop-heavy kernels on the RTX 5090 only; there was no gain on the Radeon 890M.
    ///
    /// # Safety
    ///
    /// With unchecked kernels wgpu's runtime checks are off for every kernel of this device. An out-of-range index, and
    /// a loop that does not end, are undefined behaviour on the GPU: other buffers of the device may be read or
    /// written, and the device may hang or be lost. GUST does not add its own guards on this route. The CPU twin is
    /// unchanged and remains the reference; run a kernel checked first.
    ///
    /// Integer division and remainder lose wgpu's guards too (`int_div_checks`): a zero divisor, and for a signed type
    /// `MIN / -1`, are undefined behaviour instead of the results the checked device gives. The caller must ensure that
    /// every divisor of every kernel is non-zero and that no signed division overflows.
    #[must_use]
    pub unsafe fn unchecked_kernels(mut self) -> Self {
        self.shader_checks = ShaderChecks::Unchecked;
        self
    }

    /// Whether the device these options open keeps wgpu's runtime shader checks on: [`ShaderChecks::Checked`] unless
    /// [`unchecked_kernels`](Self::unchecked_kernels) was called.
    pub fn shader_checks(&self) -> ShaderChecks {
        self.shader_checks
    }
}

/// The wgpu features a capability needs.
pub(crate) fn capability_features(capability: Capability) -> wgpu::Features {
    match capability {
        Capability::Subgroups => wgpu::Features::SUBGROUP,
        // `Capability` is `#[non_exhaustive]`: a capability this backend does not know has no feature to ask for, and
        // the empty set is "not offered" (see `required_capability_features`).
        _ => wgpu::Features::empty(),
    }
}

/// The features to add to the device descriptor for the requested `capabilities`, or the first capability that the
/// adapter (offering `available`) does not provide, as [`Error::MissingCapability`] naming `adapter`. A capability this
/// backend has no feature for is refused too: a request is never dropped silently.
///
/// `web` is true on `wasm32`: wgpu 30.0.1's WebGPU backend maps no subgroup feature (spike m8.u0), so no capability can
/// be requested in a browser and every non-empty set is refused, whatever `available` says, with the same typed error
/// and no warning (ruling 5, D142). An empty set opens there as everywhere.
pub(crate) fn required_capability_features(
    capabilities: &[Capability],
    available: wgpu::Features,
    adapter: &str,
    web: bool,
) -> Result<wgpu::Features, Error> {
    let mut required = wgpu::Features::empty();
    for &capability in capabilities {
        let feature = capability_features(capability);
        if web || feature.is_empty() || !available.contains(feature) {
            return Err(Error::MissingCapability {
                capability,
                adapter: adapter.to_owned(),
                kernel: None,
            });
        }
        required |= feature;
    }
    Ok(required)
}

/// The subgroup size range as wgpu reports it for the adapter (a fixed 4..64 on Metal), or `None` when the adapter does
/// not offer subgroup operations. The range is what the adapter can use, not what one dispatch gets: the lane count a
/// shader sees can differ between dispatches of one device (spike S3: an AMD 890M on DX12 reported 32 in one kernel and
/// 64 in the others).
pub(crate) fn adapter_subgroup_sizes(
    features: wgpu::Features,
    info: &wgpu::AdapterInfo,
) -> Option<(u32, u32)> {
    features
        .contains(wgpu::Features::SUBGROUP)
        .then_some((info.subgroup_min_size, info.subgroup_max_size))
}

impl HeadlessDevice {
    /// Open the device `options` describe. Blocks the calling thread (`pollster::block_on` over
    /// [`new_async_with`](Self::new_async_with)); not available on `wasm32`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_options(options: DeviceOptions) -> Result<Self, Error> {
        pollster::block_on(Self::new_async_with(options))
    }

    /// The adapters wgpu offers for `backends`, in the order [`AdapterChoice::Index`] counts them. Blocks like
    /// [`with_options`](Self::with_options); not available on `wasm32`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn enumerate(backends: wgpu::Backends) -> Vec<wgpu::AdapterInfo> {
        pollster::block_on(Self::enumerate_async(backends))
    }

    #[cfg_attr(not(target_arch = "wasm32"), doc = "[`enumerate`](Self::enumerate)")]
    #[cfg_attr(target_arch = "wasm32", doc = "`enumerate`")]
    /// without `pollster`. A browser lists at most the adapter it hands out.
    ///
    /// An empty list is asked for again, as device creation does for a transient `NoAdapter` (three attempts; 100 ms
    /// and 400 ms between them on native, none in a browser), because wgpu's Vulkan enumeration has returned an empty
    /// list a few times in test runs while other threads opened devices. A retry that finds adapters logs one line to
    /// stderr; a machine or backend with no adapter still gets the empty list, after the last attempt.
    ///
    /// On native the pauses sleep the calling thread, as in [`new_async`](Self::new_async), so an empty answer
    /// comes back about half a second later and blocks an executor thread that awaits this.
    pub async fn enumerate_async(backends: wgpu::Backends) -> Vec<wgpu::AdapterInfo> {
        retry_empty_list(
            ADAPTER_ATTEMPTS,
            &ADAPTER_BACKOFF,
            pause_between_attempts,
            |line| eprintln!("{line}"),
            || async {
                new_instance(backends)
                    .enumerate_adapters(backends)
                    .await
                    .iter()
                    .map(wgpu::Adapter::get_info)
                    .collect()
            },
        )
        .await
    }

    /// One adapter request. Used by the retry in `new_async_with`; only `Error::NoAdapter` is retried.
    pub(crate) async fn request_adapter(
        backends: wgpu::Backends,
        choice: &AdapterChoice,
    ) -> Result<wgpu::Adapter, Error> {
        let instance = new_instance(backends);
        let power_preference = match choice {
            AdapterChoice::HighPerformance => Some(wgpu::PowerPreference::HighPerformance),
            AdapterChoice::LowPower => Some(wgpu::PowerPreference::LowPower),
            AdapterChoice::Index(_) | AdapterChoice::NameContains(_) => None,
        };
        if let Some(power_preference) = power_preference {
            return instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference,
                    ..Default::default()
                })
                .await
                .map_err(|error| Error::NoAdapter(error.to_string()));
        }
        let adapters = instance.enumerate_adapters(backends).await;
        let count = adapters.len();
        let chosen = match choice {
            AdapterChoice::Index(index) => adapters.into_iter().nth(*index),
            AdapterChoice::NameContains(text) => {
                let text = text.to_lowercase();
                adapters
                    .into_iter()
                    .find(|adapter| adapter.get_info().name.to_lowercase().contains(&text))
            }
            AdapterChoice::HighPerformance | AdapterChoice::LowPower => None,
        };
        chosen.ok_or_else(|| {
            Error::NoAdapter(format!(
                "no adapter matches {choice:?} among the {count} that the requested backends offer"
            ))
        })
    }

    /// This device's identity: equal for clones of one [`DeviceHandle`], different for every opened device. Resident
    /// buffers carry it, and two of them can be used together only if it is equal ([`Error::MixedDevices`]).
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Fail with [`Error::MixedDevices`] unless `other` is this device's id.
    #[doc(hidden)]
    pub fn check_same_device(&self, other: u64) -> Result<(), Error> {
        if self.id == other {
            Ok(())
        } else {
            Err(Error::MixedDevices {
                first: self.id,
                second: other,
            })
        }
    }

    /// What `bytes` of buffer may be allocated: fail with `Error::DeviceLimit` (`max_buffer_size`) before any wgpu
    /// call when the request is above the limit the device was opened with, or lowered to with `with_limits`.
    pub(crate) fn check_buffer_size(&self, label: &str, bytes: u64) -> Result<(), Error> {
        let maximum = self.limits.max_buffer_size;
        if bytes > maximum {
            return Err(Error::DeviceLimit {
                kernel: label.to_owned(),
                limit: "max_buffer_size",
                requested: bytes,
                maximum,
            });
        }
        Ok(())
    }

    /// Create a buffer inside an out-of-memory error scope. The scope is already popped: the returned future
    /// resolves to what it caught. `bytes` is checked against `max_buffer_size` first.
    ///
    /// The caller completes the future: on native with [`finish_scope`](Self::finish_scope), on `wasm32` by awaiting
    /// it. The synchronous functions of `wasm32` do not call this: they push no scope and apply the pre-check only,
    /// so a browser out-of-memory still reaches the uncaptured-error handler instead of being swallowed.
    pub(crate) fn create_buffer_scoped(
        &self,
        label: &str,
        descriptor: &wgpu::BufferDescriptor<'_>,
    ) -> Result<
        (
            wgpu::Buffer,
            impl Future<Output = Option<wgpu::Error>> + use<>,
        ),
        Error,
    > {
        self.check_buffer_size(label, descriptor.size)?;
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let buffer = self.device.create_buffer(descriptor);
        Ok((buffer, scope.pop()))
    }

    /// Resolve the error scope of [`create_buffer_scoped`](Self::create_buffer_scoped) on native, without polling
    /// the device. On wgpu-core the buffer's error is recorded on the creating thread before `create_buffer`
    /// returns, and `pop_error_scope` hands back an already-resolved future (wgpu 30.0.1 `backend/wgpu_core.rs`:
    /// `handle_error_or_return_handler` 657-668 records into the scope, `pop_error_scope` 1860-1897 returns
    /// `ready(scope.error)`), so one poll with a no-op waker completes it. Waiting for the queue here would make every
    /// allocation, every pool miss included, a full GPU synchronisation (measured by the unit test
    /// `a_popped_scope_resolves_without_a_device_poll`).
    ///
    /// Error scopes are kept per thread inside one device's error sink (`ErrorSinkRaw.scopes`, 634), and a scope only
    /// catches errors raised on its own thread: concurrent native allocations on other threads can neither land in
    /// this scope nor be caught by it.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn finish_scope(
        &self,
        label: &str,
        bytes: u64,
        caught: impl Future<Output = Option<wgpu::Error>>,
    ) -> Result<(), Error> {
        finish_popped_scope(label, bytes, caught)
    }

    /// Allocate one buffer of `size` bytes: checked, and on native inside an out-of-memory scope. On `wasm32` this
    /// applies the pre-check only and pushes no scope, because the scope cannot be awaited in a synchronous function.
    pub(crate) fn allocate(
        &self,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Result<wgpu::Buffer, Error> {
        let descriptor = wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (buffer, caught) = self.create_buffer_scoped(label, &descriptor)?;
            self.finish_scope(label, size, caught)?;
            Ok(buffer)
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.check_buffer_size(label, size)?;
            Ok(self.device.create_buffer(&descriptor))
        }
    }

    /// Fallible [`create_typed_buffer`](Self::create_typed_buffer): the same buffer, but a request above
    /// `max_buffer_size` is `Error::DeviceLimit` (decided before any wgpu call) and an allocation the adapter cannot
    /// serve is `Error::OutOfMemory` (through an error scope), where the original panics in wgpu's uncaptured-error
    /// handler. The original keeps its signature and its panics.
    ///
    /// On `wasm32` this applies the pre-check only and pushes no scope: the scope cannot be awaited in a synchronous
    /// function, and a scope nobody reads would swallow the error the browser's uncaptured-error handler reports.
    pub fn try_create_typed_buffer<T: GpuPod>(
        &self,
        label: &str,
        values: &[T],
        access: GpuBufferAccess,
    ) -> Result<GpuBuffer<T>, Error> {
        assert_pod_matches_storage_layout::<T>();
        let usage = self.storage_usage(access);
        let size = u64::from(T::LAYOUT.stride).saturating_mul(values.len().max(1) as u64);
        self.check_buffer_size(label, size)?;
        #[cfg(not(target_arch = "wasm32"))]
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let raw = if values.is_empty() {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        } else {
            self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some(label),
                contents: pod_slice_as_bytes(values),
                usage,
            })
        };
        #[cfg(not(target_arch = "wasm32"))]
        self.finish_scope(label, size, scope.pop())?;
        Ok(self.unpooled(raw, values.len(), access, T::LAYOUT))
    }

    /// A zero-filled persistent buffer of `len` elements, allocated exactly (not from the pool), with the
    /// `max_buffer_size` pre-check and the out-of-memory scope of
    /// [`try_create_typed_buffer`](Self::try_create_typed_buffer). A fresh wgpu buffer reads as zeros.
    pub fn create_zeroed_buffer<T: GpuPod>(
        &self,
        label: &str,
        len: usize,
        access: GpuBufferAccess,
    ) -> Result<GpuBuffer<T>, Error> {
        assert_pod_matches_storage_layout::<T>();
        let size = zeroed_size(T::LAYOUT.stride, len);
        let raw = self.allocate(label, size, self.storage_usage(access))?;
        Ok(self.unpooled(raw, len, access, T::LAYOUT))
    }

    /// [`create_zeroed_buffer`](Self::create_zeroed_buffer) for the browser, awaiting the error scope, which the
    /// synchronous form cannot.
    #[cfg(target_arch = "wasm32")]
    pub async fn create_zeroed_buffer_async<T: GpuPod>(
        &self,
        label: &str,
        len: usize,
        access: GpuBufferAccess,
    ) -> Result<GpuBuffer<T>, Error> {
        assert_pod_matches_storage_layout::<T>();
        let size = zeroed_size(T::LAYOUT.stride, len);
        let (raw, caught) = self.create_buffer_scoped(
            label,
            &wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: self.storage_usage(access),
                mapped_at_creation: false,
            },
        )?;
        scope_result(label, size, caught.await)?;
        Ok(self.unpooled(raw, len, access, T::LAYOUT))
    }

    /// `STORAGE | COPY_DST` (and `COPY_SRC` for a read-write buffer, or for any buffer when the device was opened with
    /// [`DeviceOptions::capture`]): what every persistent typed buffer is created with.
    pub(crate) fn storage_usage(&self, access: GpuBufferAccess) -> wgpu::BufferUsages {
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if access == GpuBufferAccess::ReadWrite || self.capture {
            usage |= wgpu::BufferUsages::COPY_SRC;
        }
        usage
    }

    fn unpooled<T>(
        &self,
        raw: wgpu::Buffer,
        len: usize,
        access: GpuBufferAccess,
        layout: gust::TypeLayout,
    ) -> GpuBuffer<T> {
        GpuBuffer {
            raw,
            len,
            access,
            layout,
            device_id: self.id,
            indirect: false,
            lease: None,
            marker: std::marker::PhantomData,
        }
    }
}

/// [`HeadlessDevice::finish_scope`] for a caller that holds only a `wgpu::Device` (the `pixel` pass encoder): one poll
/// of a popped scope's future with a no-op waker, which is enough on wgpu-core (see `finish_scope`).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn finish_popped_scope(
    label: &str,
    bytes: u64,
    caught: impl Future<Output = Option<wgpu::Error>>,
) -> Result<(), Error> {
    let mut caught = std::pin::pin!(caught);
    match caught
        .as_mut()
        .poll(&mut std::task::Context::from_waker(std::task::Waker::noop()))
    {
        std::task::Poll::Ready(error) => scope_result(label, bytes, error),
        std::task::Poll::Pending => Err(Error::Poll(format!(
            "the error scope of `{label}` did not resolve"
        ))),
    }
}

/// The error a scope's caught value stands for: out of memory is [`Error::OutOfMemory`], anything else the scope
/// could see is reported as it is (S7.6 outcome B: an adapter that loses the device instead of reporting memory
/// shows up as [`Error::Poll`] from the poll, never as a panic).
pub(crate) fn scope_result(
    label: &str,
    bytes: u64,
    caught: Option<wgpu::Error>,
) -> Result<(), Error> {
    match caught {
        None => Ok(()),
        Some(wgpu::Error::OutOfMemory { .. }) => Err(Error::OutOfMemory {
            label: label.to_owned(),
            bytes,
        }),
        Some(other) => Err(Error::Poll(format!(
            "allocating `{label}` ({bytes} bytes): {other}"
        ))),
    }
}

/// The bytes of a buffer of `len` elements of `stride` bytes: one element for an empty buffer (wgpu has no
/// zero-sized binding), saturating so that an absurd length is refused by the `max_buffer_size` pre-check instead of
/// overflowing.
pub(crate) fn zeroed_size(stride: u32, len: usize) -> u64 {
    u64::from(stride).saturating_mul((len as u64).max(1))
}

/// A wgpu instance restricted to `backends`, without a display handle.
fn new_instance(backends: wgpu::Backends) -> wgpu::Instance {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = backends;
    wgpu::Instance::new(descriptor)
}

// ---------------------------------------------------------------------------------------------------------------
// The default device.
// ---------------------------------------------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
mod native_default {
    use std::sync::{Mutex, MutexGuard, PoisonError};

    use super::{Arc, DeviceHandle, DeviceOptions, Error, HeadlessDevice};

    /// Options for the first `default_device()`; `None` means `DeviceOptions::default()`.
    static OPTIONS: Mutex<Option<DeviceOptions>> = Mutex::new(None);
    /// The one device, once opened. The lock is held while it opens, so concurrent first calls open one device.
    static DEVICE: Mutex<Option<DeviceHandle>> = Mutex::new(None);

    fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
        mutex.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(super) fn set_options(options: DeviceOptions) -> Result<(), Error> {
        let device = lock(&DEVICE);
        if device.is_some() {
            return Err(Error::DefaultDeviceOpened);
        }
        *lock(&OPTIONS) = Some(options);
        Ok(())
    }

    pub(super) fn set_device(handle: DeviceHandle) -> Result<(), Error> {
        let mut device = lock(&DEVICE);
        if device.is_some() {
            return Err(Error::DefaultDeviceOpened);
        }
        super::check_default_capabilities(&handle)?;
        *device = Some(handle);
        Ok(())
    }

    pub(super) fn get() -> Result<DeviceHandle, Error> {
        let mut device = lock(&DEVICE);
        if let Some(device) = &*device {
            return Ok(Arc::clone(device));
        }
        let options = lock(&OPTIONS).clone().unwrap_or_default();
        let opened = Arc::new(HeadlessDevice::with_options(options)?);
        *device = Some(Arc::clone(&opened));
        Ok(opened)
    }
}

#[cfg(target_arch = "wasm32")]
mod web_default {
    use std::{cell::RefCell, rc::Rc};

    use super::{DeviceHandle, DeviceOptions, Error, HeadlessDevice};

    thread_local! {
        static OPTIONS: RefCell<Option<DeviceOptions>> = const { RefCell::new(None) };
        static DEVICE: RefCell<Option<DeviceHandle>> = const { RefCell::new(None) };
    }

    pub(super) fn set_options(options: DeviceOptions) -> Result<(), Error> {
        if DEVICE.with(|device| device.borrow().is_some()) {
            return Err(Error::DefaultDeviceOpened);
        }
        OPTIONS.with(|slot| *slot.borrow_mut() = Some(options));
        Ok(())
    }

    pub(super) fn set_device(handle: DeviceHandle) -> Result<(), Error> {
        if DEVICE.with(|device| device.borrow().is_some()) {
            return Err(Error::DefaultDeviceOpened);
        }
        super::check_default_capabilities(&handle)?;
        DEVICE.with(|device| *device.borrow_mut() = Some(handle));
        Ok(())
    }

    pub(super) fn get() -> Result<DeviceHandle, Error> {
        DEVICE
            .with(|device| device.borrow().clone())
            .ok_or_else(|| {
                Error::NoAdapter(
                    "the default device is not open; await `default_device_async()` once, or call `set_default_device`, first"
                        .to_owned(),
                )
            })
    }

    pub(super) async fn get_async() -> Result<DeviceHandle, Error> {
        if let Ok(device) = get() {
            return Ok(device);
        }
        let options = OPTIONS
            .with(|slot| slot.borrow().clone())
            .unwrap_or_default();
        let opened = Rc::new(HeadlessDevice::new_async_with(options).await?);
        // A second caller may have opened one while this awaited; keep the first.
        Ok(DEVICE.with(|device| {
            Rc::clone(
                device
                    .borrow_mut()
                    .get_or_insert_with(|| Rc::clone(&opened)),
            )
        }))
    }
}

/// Set the options of the default device. They apply to the first [`default_device`] (or
/// [`default_device_async`]) of the process; after the device is open this is `Error::DefaultDeviceOpened`, and so it is
/// after [`set_default_device`]. If [`set_default_device`] follows, the options are never used.
///
/// Options made with [`DeviceOptions::unchecked_kernels`] turn wgpu's runtime checks off for the default device and so
/// for every kernel that runs on it, GUST's own library kernels and data-supplied indices included.
///
/// ```
/// use gust_wgpu::{DeviceOptions, Error, LimitsPolicy, default_device, set_default_options};
///
/// set_default_options(DeviceOptions::default().with_limits(LimitsPolicy::Default))?;
/// let device = default_device()?;
/// assert!(!device.adapter_info().name.is_empty());
/// // Once the device is open the options are fixed.
/// assert!(matches!(
///     set_default_options(DeviceOptions::default()),
///     Err(Error::DefaultDeviceOpened)
/// ));
/// # Ok::<(), Error>(())
/// ```
pub fn set_default_options(options: DeviceOptions) -> Result<(), Error> {
    #[cfg(not(target_arch = "wasm32"))]
    return native_default::set_options(options);
    #[cfg(target_arch = "wasm32")]
    return web_default::set_options(options);
}

/// The first capability of `required` that `held` lacks, as the refusal of [`set_default_device`] naming `adapter`.
fn missing_default_capability(
    held: &[Capability],
    required: &[Capability],
    adapter: &str,
) -> Result<(), Error> {
    match required
        .iter()
        .find(|capability| !held.contains(capability))
    {
        None => Ok(()),
        Some(&capability) => Err(Error::DefaultDeviceCapability {
            capability,
            adapter: adapter.to_owned(),
        }),
    }
}

/// Refuse a default device that lacks a capability this build was compiled for.
fn check_default_capabilities(device: &HeadlessDevice) -> Result<(), Error> {
    missing_default_capability(
        device.capabilities(),
        gust::capability::enabled(),
        &device.adapter_info().name,
    )
}

/// Make `device` the process's default device, the one `default_device()` returns and the element-wise map calls,
/// `par_for!` and `GpuVec::from_slice` run on, instead of opening a second device beside the one your renderer
/// shares with GUST (see [`HeadlessDevice::adopt`]).
///
/// | Case | Result |
/// | --- | --- |
/// | a default device exists (opened or set) | `Error::DefaultDeviceOpened` |
/// | [`set_default_options`], then this | accepted; the options are never used |
/// | this, then [`set_default_options`] | `Error::DefaultDeviceOpened` |
/// | the device lacks a capability of [`gust::capability::enabled`] | `Error::DefaultDeviceCapability` |
///
/// A `+subgroups` build therefore needs a device adopted with `gust::capability::enabled()`: a device adopted with
/// `&[]` is refused here, because the calls that take no device cannot name another one. On `wasm32`, when a
/// [`default_device_async`] open is in flight, whichever lands first is kept: a set that lands first wins and the
/// opened device is dropped, a set that lands second is `Error::DefaultDeviceOpened`.
///
/// ```no_run
/// use gust_wgpu::{DeviceHandle, Error, HeadlessDevice, default_device, set_default_device};
///
/// # fn share(device: HeadlessDevice) -> Result<(), Error> {
/// let handle = DeviceHandle::new(device);
/// set_default_device(DeviceHandle::clone(&handle))?;
/// assert_eq!(default_device()?.id(), handle.id());
/// # Ok(())
/// # }
/// ```
pub fn set_default_device(device: DeviceHandle) -> Result<(), Error> {
    #[cfg(not(target_arch = "wasm32"))]
    return native_default::set_device(device);
    #[cfg(target_arch = "wasm32")]
    return web_default::set_device(device);
}

/// The process's default device, opened on the first call from the options of [`set_default_options`] (or
/// `DeviceOptions::default()`), and the same device after that. Native only opens it here, blocking the calling
/// thread; a failure to open (no adapter) is returned and not remembered, so a later call tries again.
///
/// Its limits are those of `DeviceOptions::default()`, which on native are the adapter's own, not the portable ones of
#[cfg_attr(not(target_arch = "wasm32"), doc = "[`HeadlessDevice::new`];")]
#[cfg_attr(target_arch = "wasm32", doc = "`HeadlessDevice::new`;")]
/// [`set_default_options`] changes them before the first call.
///
/// On `wasm32` nothing may block: this returns the device [`default_device_async`] already opened, or
/// `Error::NoAdapter` before that.
///
/// ```
/// use gust_wgpu::default_device;
///
/// let first = default_device()?;
/// let second = default_device()?;
/// assert_eq!(first.id(), second.id());
/// # Ok::<(), gust_wgpu::Error>(())
/// ```
pub fn default_device() -> Result<DeviceHandle, Error> {
    #[cfg(not(target_arch = "wasm32"))]
    return native_default::get();
    #[cfg(target_arch = "wasm32")]
    return web_default::get();
}

/// [`default_device`] for the browser, where opening the device is asynchronous. On native it opens the device on
/// the calling thread and returns at once (the adapter retry may sleep it, as `HeadlessDevice::new_async` does).
///
/// ```
/// use std::{future::Future, pin::pin, task::{Context, Poll, Waker}};
/// use gust_wgpu::{default_device, default_device_async};
///
/// // A minimal executor: on native this future is ready on its first poll.
/// let mut context = Context::from_waker(Waker::noop());
/// let mut future = pin!(default_device_async());
/// let Poll::Ready(opened) = future.as_mut().poll(&mut context) else {
///     panic!("the default device opens without waiting on native");
/// };
/// let opened = opened?;
/// assert_eq!(opened.id(), default_device()?.id());
/// # Ok::<(), gust_wgpu::Error>(())
/// ```
pub async fn default_device_async() -> Result<DeviceHandle, Error> {
    #[cfg(not(target_arch = "wasm32"))]
    return native_default::get();
    #[cfg(target_arch = "wasm32")]
    return web_default::get_async().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mapping the out-of-memory scope relies on. A real out-of-memory needs an exhausted adapter, which the
    /// tests never create, so the translation is pinned here on the values wgpu would deliver.
    #[test]
    fn a_caught_out_of_memory_error_becomes_error_out_of_memory() {
        assert!(scope_result("buffer", 8, None).is_ok());
        let caught = wgpu::Error::OutOfMemory {
            source: Box::new(std::fmt::Error),
        };
        match scope_result("buffer", 8, Some(caught)) {
            Err(Error::OutOfMemory { label, bytes }) => {
                assert_eq!((label.as_str(), bytes), ("buffer", 8));
            }
            other => panic!("expected OutOfMemory, got {other:?}"),
        }
        let caught = wgpu::Error::Validation {
            source: Box::new(std::fmt::Error),
            description: "bad usage".to_owned(),
        };
        match scope_result("buffer", 8, Some(caught)) {
            Err(Error::Poll(message)) => {
                assert!(
                    message.contains("buffer") && message.contains("bad usage"),
                    "{message}"
                );
            }
            other => panic!("expected Poll for a non-memory error, got {other:?}"),
        }
    }

    /// The storage buffer option against faked adapter limits: no adapter is needed.
    #[test]
    fn a_stated_storage_buffer_count_is_requested_or_refused_by_name() {
        let ask = |requested, available, web| {
            required_storage_buffers(requested, available, "Fake Adapter", web)
        };
        // Off: nothing is requested, whatever the adapter offers.
        assert_eq!(ask(None, 4, false).unwrap(), None);
        assert_eq!(ask(None, 4, true).unwrap(), None);
        // Accepted at the adapter's value and below it, from 8 up.
        assert_eq!(ask(Some(10), 16, false).unwrap(), Some(10));
        assert_eq!(ask(Some(16), 16, false).unwrap(), Some(16));
        assert_eq!(ask(Some(8), 8, false).unwrap(), Some(8));
        // One over the adapter's value is a DeviceLimit naming the limit, n and the adapter's value.
        match ask(Some(17), 16, false) {
            Err(Error::DeviceLimit {
                kernel,
                limit,
                requested,
                maximum,
            }) => {
                assert_eq!(limit, "max_storage_buffers_per_shader_stage");
                assert_eq!((requested, maximum), (17, 16));
                assert!(kernel.contains("Fake Adapter"), "{kernel}");
            }
            other => panic!("expected DeviceLimit, got {other:?}"),
        }
        // Below 8 and in a browser: refused by name, as RequestDevice.
        for (requested, web, text) in [
            (7, false, "at least 8"),
            (0, false, "at least 8"),
            (10, true, "browser devices open with 8"),
            (7, true, "browser devices open with 8"),
        ] {
            match ask(Some(requested), 16, web) {
                Err(Error::RequestDevice(message)) => assert!(
                    message.contains(&format!("with_storage_buffers({requested})"))
                        && message.contains(text),
                    "{message}"
                ),
                other => panic!("expected RequestDevice, got {other:?}"),
            }
        }
    }

    /// The open-refusal path with fake feature sets: it needs no adapter, so it runs where no GPU has subgroups.
    #[test]
    fn a_requested_capability_the_adapter_lacks_is_missing_capability() {
        let adapter = "Fake Adapter";
        let with = wgpu::Features::SUBGROUP | wgpu::Features::TIMESTAMP_QUERY;
        let without = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::FLOAT32_FILTERABLE;

        // Nothing requested asks for nothing, whatever the adapter has.
        assert_eq!(
            required_capability_features(&[], without, adapter, false).unwrap(),
            wgpu::Features::empty()
        );
        assert_eq!(
            required_capability_features(&[], with, adapter, false).unwrap(),
            wgpu::Features::empty()
        );
        // The adapter has it: exactly its feature is asked for, nothing else.
        assert_eq!(
            required_capability_features(&[Capability::Subgroups], with, adapter, false).unwrap(),
            wgpu::Features::SUBGROUP
        );
        // The adapter lacks it: a typed refusal that names the capability and the adapter, and no kernel.
        match required_capability_features(&[Capability::Subgroups], without, adapter, false) {
            Err(Error::MissingCapability {
                capability,
                adapter: name,
                kernel,
            }) => {
                assert_eq!(capability, Capability::Subgroups);
                assert_eq!(name, adapter);
                assert_eq!(kernel, None);
            }
            other => panic!("expected MissingCapability, got {other:?}"),
        }
        let text = required_capability_features(&[Capability::Subgroups], without, adapter, false)
            .unwrap_err()
            .to_string();
        assert_eq!(
            text,
            "the adapter `Fake Adapter` does not offer the `subgroups` capability that the device options request"
        );
        // An empty adapter feature set refuses too.
        assert!(
            required_capability_features(
                &[Capability::Subgroups],
                wgpu::Features::empty(),
                adapter,
                false
            )
            .is_err()
        );
    }

    /// The browser (`web` is `cfg!(target_arch = "wasm32")` in `open`): wgpu 30.0.1's WebGPU backend cannot request
    /// subgroups, so a non-empty set is the typed refusal even when the adapter's feature set claims the feature, with
    /// the same error and text as an adapter that lacks it. An empty set opens. This runs on native; `check-web`
    /// compiles the call with `web` true.
    #[test]
    fn a_non_empty_capability_set_is_refused_in_the_browser() {
        let adapter = "Fake Browser Adapter";
        let claims = wgpu::Features::SUBGROUP | wgpu::Features::TIMESTAMP_QUERY;
        match required_capability_features(&[Capability::Subgroups], claims, adapter, true) {
            Err(Error::MissingCapability {
                capability,
                adapter: name,
                kernel,
            }) => {
                assert_eq!(capability, Capability::Subgroups);
                assert_eq!(name, adapter);
                assert_eq!(kernel, None);
            }
            other => panic!("expected MissingCapability, got {other:?}"),
        }
        assert_eq!(
            required_capability_features(&[Capability::Subgroups], claims, adapter, true)
                .unwrap_err()
                .to_string(),
            "the adapter `Fake Browser Adapter` does not offer the `subgroups` capability that the device options request"
        );
        // An empty set is not a request: it opens in the browser as everywhere.
        assert_eq!(
            required_capability_features(&[], claims, adapter, true).unwrap(),
            wgpu::Features::empty()
        );
        // And `web` changes nothing off the browser: the same claim is accepted there.
        assert_eq!(
            required_capability_features(&[Capability::Subgroups], claims, adapter, false).unwrap(),
            wgpu::Features::SUBGROUP
        );
    }

    /// The refusal of `set_default_device` on fake capability lists: it needs no adapter, so it runs everywhere.
    #[test]
    fn a_default_device_must_hold_every_capability_the_build_requires() {
        let adapter = "Fake Adapter";
        // Nothing required: any device is accepted, with or without capabilities.
        assert!(missing_default_capability(&[], &[], adapter).is_ok());
        assert!(missing_default_capability(&[Capability::Subgroups], &[], adapter).is_ok());
        // The device holds what the build requires.
        assert!(
            missing_default_capability(&[Capability::Subgroups], &[Capability::Subgroups], adapter)
                .is_ok()
        );
        // A device adopted with `&[]` under a `+subgroups` build is refused, naming the capability and the adapter.
        let error = missing_default_capability(&[], &[Capability::Subgroups], adapter)
            .expect_err("a device without the capability");
        match &error {
            Error::DefaultDeviceCapability {
                capability,
                adapter: name,
            } => {
                assert_eq!(*capability, Capability::Subgroups);
                assert_eq!(name, adapter);
            }
            other => panic!("expected DefaultDeviceCapability, got {other:?}"),
        }
        assert_eq!(
            error.to_string(),
            "the device given to `set_default_device` (adapter `Fake Adapter`) was not opened or adopted with the `subgroups` capability this build was compiled for; pass `gust::capability::enabled()`"
        );
    }

    #[test]
    fn the_subgroup_range_is_reported_only_for_an_adapter_that_offers_subgroups() {
        let mut info = wgpu::AdapterInfo::new(wgpu::DeviceType::DiscreteGpu, wgpu::Backend::Vulkan);
        info.subgroup_min_size = 32;
        info.subgroup_max_size = 64;
        assert_eq!(
            adapter_subgroup_sizes(wgpu::Features::SUBGROUP, &info),
            Some((32, 64))
        );
        assert_eq!(
            adapter_subgroup_sizes(wgpu::Features::TIMESTAMP_QUERY, &info),
            None
        );
    }

    #[test]
    fn device_options_hold_the_builds_capabilities_until_asked_otherwise() {
        // The default is the set this build was compiled for (empty under `dev`, `[Subgroups]` under `dev+subgroups`);
        // everything below opens with an explicit set, so it does not depend on the profile.
        assert_eq!(
            DeviceOptions::default().capabilities,
            gust::capability::enabled()
        );
        let none = DeviceOptions::default().with_capabilities(&[]);
        let options = none.clone().with_capabilities(&[Capability::Subgroups]);
        assert_eq!(options.capabilities, &[Capability::Subgroups]);
        assert_ne!(options, none);
        // The setter replaces the set and touches no other field.
        let back = options.clone().with_capabilities(&[]);
        assert_eq!(back, none);
    }

    #[test]
    fn pool_classes_are_exact_up_to_64_kib_then_powers_of_two() {
        use crate::pool::size_class;
        let max = 1 << 32;
        assert_eq!(size_class(0, max), Some(4));
        assert_eq!(size_class(4, max), Some(4));
        assert_eq!(size_class(1000 * 4, max), Some(4000));
        assert_eq!(size_class(65_536, max), Some(65_536));
        assert_eq!(size_class(65_540, max), Some(131_072));
        assert_eq!(size_class(4000 * 4, max), Some(16_000));
        assert_eq!(size_class((1 << 20) + 4, max), Some(1 << 21));
        // Capped at the device's largest buffer, rounded down to a multiple of four.
        assert_eq!(
            size_class((1 << 30) + 4, (1 << 31) - 1),
            Some((1_u64 << 31) - 4)
        );
        assert_eq!(size_class(1 << 31, (1 << 31) - 1), None);
        assert_eq!(size_class(u64::MAX, u64::MAX), None);
    }
    /// SF2 of the m7.a0 review: on wgpu-core a popped error scope is already resolved. The error of a buffer
    /// creation is recorded before `create_buffer` returns, and polling the scope's future once, with no device
    /// poll, delivers it. `finish_scope` relies on this, so a regression fails here instead of hanging an allocation.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_popped_scope_resolves_without_a_device_poll() {
        use std::task::{Context, Poll, Waker};

        let device = HeadlessDevice::new().expect("a Vulkan adapter");
        let descriptor = |usage| wgpu::BufferDescriptor {
            label: Some("scope probe"),
            size: 16,
            usage,
            mapped_at_creation: false,
        };
        let resolve = |filter, usage| {
            let scope = device.device.push_error_scope(filter);
            let _buffer = device.device.create_buffer(&descriptor(usage));
            let mut caught = std::pin::pin!(scope.pop());
            caught
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
        };
        // A good buffer: nothing caught, and the future was ready on its first poll.
        assert!(matches!(
            resolve(wgpu::ErrorFilter::OutOfMemory, wgpu::BufferUsages::STORAGE),
            Poll::Ready(None)
        ));
        // A buffer wgpu refuses (no usage) is caught by a validation scope on the first poll too.
        assert!(matches!(
            resolve(wgpu::ErrorFilter::Validation, wgpu::BufferUsages::empty()),
            Poll::Ready(Some(wgpu::Error::Validation { .. }))
        ));
    }
}
