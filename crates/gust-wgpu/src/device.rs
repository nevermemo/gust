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

use gust::{GpuPod, pod_slice_as_bytes};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{Error, GpuBuffer, GpuBufferAccess, HeadlessDevice, assert_pod_matches_storage_layout};

/// The handle a resident buffer holds to its device: `Arc<HeadlessDevice>` on native, where the device is `Send` and
/// `Sync`, and `Rc<HeadlessDevice>` on `wasm32`, where wgpu's WebGPU device is neither (checked under `check-web`,
/// m7.a0 S3). `GpuVec` is `Send + Sync` on native only.
#[cfg(not(target_arch = "wasm32"))]
pub type DeviceHandle = Arc<HeadlessDevice>;
/// The handle a resident buffer holds to its device: `Rc<HeadlessDevice>` on `wasm32` (see the native alias).
#[cfg(target_arch = "wasm32")]
pub type DeviceHandle = std::rc::Rc<HeadlessDevice>;

/// Which adapter [`HeadlessDevice::with_options`] opens.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum AdapterChoice {
    /// wgpu's high-performance preference (a discrete GPU when there is one). The default.
    #[default]
    HighPerformance,
    /// wgpu's low-power preference (an integrated GPU when there is one).
    LowPower,
    /// The `n`th adapter of [`HeadlessDevice::enumerate`] for the same backends.
    Index(usize),
    /// The first adapter whose name contains this text, ignoring letter case.
    NameContains(String),
}

/// Which limits [`HeadlessDevice::with_options`] asks the adapter for.
///
/// `LimitsPolicy::default()` is [`Default`](Self::Default), wgpu's portable limits. It is not what
/// `DeviceOptions::default().limits` holds: on native that is [`Adapter`](Self::Adapter), so the process's
/// [`default_device`] allows storage bindings up to the adapter's size (2 GiB on the adapters measured in S7.6),
/// where the browser allows 128 MiB. A program that must behave the same on both sets
/// [`Default`](Self::Default) through [`set_default_options`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum LimitsPolicy {
    /// wgpu's default limits (what [`HeadlessDevice::new`] requests): 256 MiB buffers, 128 MiB storage bindings,
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

/// Which optional device features [`HeadlessDevice::with_options`] asks the adapter for (slice mx.t2).
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
/// of [`default_device`]. [`HeadlessDevice::new`] and [`new_async`](HeadlessDevice::new_async) use
/// [`Default`](Self::Default), as they use wgpu's default limits.
///
/// The enum is `#[non_exhaustive]`: a later release may add a policy, so a `match` outside this crate needs a wildcard
/// arm.
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

/// How to open a device: the backends, the adapter, the limits and the optional features. Explicit and overridable;
/// there is no workload policy in it.
///
/// `DeviceOptions::default()` is what [`default_device`] opens: Vulkan (the browser's WebGPU on `wasm32`), the
/// high-performance adapter, and on native the adapter's own limits ([`LimitsPolicy::Adapter`], S7.6 outcome A) and
/// optional features ([`FeaturesPolicy::Adapter`], slice mx.t2). It
/// differs from [`HeadlessDevice::new`], which keeps requesting wgpu's default limits, and from
/// `LimitsPolicy::default()`, which is [`LimitsPolicy::Default`]: native `default_device()` allows storage bindings
/// of the adapter's size (2 GiB measured), the browser's only 128 MiB.
///
/// The struct is `#[non_exhaustive]`, so a later release can add an option without breaking a caller: build one from
/// [`DeviceOptions::default()`](Default::default) and the `with_*` methods, which set one field each
/// (`DeviceOptions::default().with_adapter(AdapterChoice::LowPower)`). The fields stay public to read and to assign
/// on a value you already hold.
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
}

impl Default for DeviceOptions {
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
        }
    }
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

    /// [`enumerate`](Self::enumerate) without blocking. A browser lists at most the adapter it hands out.
    pub async fn enumerate_async(backends: wgpu::Backends) -> Vec<wgpu::AdapterInfo> {
        let instance = new_instance(backends);
        instance
            .enumerate_adapters(backends)
            .await
            .iter()
            .map(wgpu::Adapter::get_info)
            .collect()
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
        let usage = storage_usage(access);
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
        let raw = self.allocate(label, size, storage_usage(access))?;
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
                usage: storage_usage(access),
                mapped_at_creation: false,
            },
        )?;
        scope_result(label, size, caught.await)?;
        Ok(self.unpooled(raw, len, access, T::LAYOUT))
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

/// `STORAGE | COPY_DST` (and `COPY_SRC` for a read-write buffer), as `create_typed_buffer` allocates.
pub(crate) fn storage_usage(access: GpuBufferAccess) -> wgpu::BufferUsages {
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    if access == GpuBufferAccess::ReadWrite {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    usage
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

    pub(super) fn get() -> Result<DeviceHandle, Error> {
        DEVICE
            .with(|device| device.borrow().clone())
            .ok_or_else(|| {
                Error::NoAdapter(
                    "the default device is not open; await `default_device_async()` once first"
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
/// [`default_device_async`]) of the process; after the device is open this is `Error::DefaultDeviceOpened`.
pub fn set_default_options(options: DeviceOptions) -> Result<(), Error> {
    #[cfg(not(target_arch = "wasm32"))]
    return native_default::set_options(options);
    #[cfg(target_arch = "wasm32")]
    return web_default::set_options(options);
}

/// The process's default device, opened on the first call from the options of [`set_default_options`] (or
/// `DeviceOptions::default()`), and the same device after that. Native only opens it here, blocking the calling
/// thread; a failure to open (no adapter) is returned and not remembered, so a later call tries again.
///
/// On `wasm32` nothing may block: this returns the device [`default_device_async`] already opened, or
/// `Error::NoAdapter` before that.
pub fn default_device() -> Result<DeviceHandle, Error> {
    #[cfg(not(target_arch = "wasm32"))]
    return native_default::get();
    #[cfg(target_arch = "wasm32")]
    return web_default::get();
}

/// [`default_device`] for the browser, where opening the device is asynchronous. On native it opens the device on
/// the calling thread and returns at once (the adapter retry may sleep it, as `HeadlessDevice::new_async` does).
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
