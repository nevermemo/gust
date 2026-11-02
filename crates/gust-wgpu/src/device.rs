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

/// How to open a device: the backends, the adapter and the limits. Explicit and overridable; there is no workload
/// policy in it.
///
/// `DeviceOptions::default()` is what [`default_device`] opens: Vulkan (the browser's WebGPU on `wasm32`), the
/// high-performance adapter, and on native the adapter's own limits ([`LimitsPolicy::Adapter`], S7.6 outcome A). It
/// differs from [`HeadlessDevice::new`], which keeps requesting wgpu's default limits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceOptions {
    pub backends: wgpu::Backends,
    pub adapter: AdapterChoice,
    pub limits: LimitsPolicy,
}

impl Default for DeviceOptions {
    fn default() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let (backends, limits) = (wgpu::Backends::VULKAN, LimitsPolicy::Adapter);
        #[cfg(target_arch = "wasm32")]
        let (backends, limits) = (wgpu::Backends::BROWSER_WEBGPU, LimitsPolicy::Default);
        Self {
            backends,
            adapter: AdapterChoice::HighPerformance,
            limits,
        }
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
    /// The caller completes the future: on native after `device.poll(Wait)` with `pollster::block_on`, on `wasm32`
    /// by awaiting it (the sync functions of `wasm32` drop it and apply the pre-check only).
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

    /// Resolve the error scope of [`create_buffer_scoped`](Self::create_buffer_scoped) on native: the poll is what
    /// completes it (S7.6), then the future is driven to the end.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn finish_scope(
        &self,
        label: &str,
        bytes: u64,
        caught: impl Future<Output = Option<wgpu::Error>>,
    ) -> Result<(), Error> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| Error::Poll(error.to_string()))?;
        scope_result(label, bytes, pollster::block_on(caught))
    }

    /// Allocate one buffer of `size` bytes, checked and scoped, on native or (pre-check only) on `wasm32`.
    pub(crate) fn allocate(
        &self,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Result<wgpu::Buffer, Error> {
        let (buffer, caught) = self.create_buffer_scoped(
            label,
            &wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            },
        )?;
        #[cfg(not(target_arch = "wasm32"))]
        self.finish_scope(label, size, caught)?;
        #[cfg(target_arch = "wasm32")]
        drop(caught);
        Ok(buffer)
    }

    /// Fallible [`create_typed_buffer`](Self::create_typed_buffer): the same buffer, but a request above
    /// `max_buffer_size` is `Error::DeviceLimit` (decided before any wgpu call) and an allocation the adapter cannot
    /// serve is `Error::OutOfMemory` (through an error scope), where the original panics in wgpu's uncaptured-error
    /// handler. The original keeps its signature and its panics.
    ///
    /// On `wasm32` this applies the pre-check only: the scope cannot be awaited in a synchronous function.
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
        let caught = scope.pop();
        #[cfg(not(target_arch = "wasm32"))]
        self.finish_scope(label, size, caught)?;
        #[cfg(target_arch = "wasm32")]
        drop(caught);
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
            "allocating buffer `{label}` ({bytes} bytes): {other}"
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
}
