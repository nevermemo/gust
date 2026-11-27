//! Sharing a device with a renderer: what GUST asks an adapter for ([`DeviceRequest`]), and wrapping a device the
//! caller opened ([`HeadlessDevice::adopt`]).

use crate::{DeviceOptions, Error, HeadlessDevice, device, spirv_route_features};
use gust::capability::Capability;

/// What GUST asks an adapter for under some [`DeviceOptions`]: the features and limits to put in the device
/// descriptor. Merge your own needs in (`features | wgpu::Features::FLOAT32_BLENDABLE`,
/// `limits.or_better_values_from(&mine)`), open the device in your own `wgpu::Instance`, then
/// [`HeadlessDevice::adopt`] it. See [`DeviceOptions::device_request`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceRequest {
    /// The features GUST requires of the device.
    pub features: wgpu::Features,
    /// The limits GUST requires of the device.
    pub limits: wgpu::Limits,
}

impl DeviceOptions {
    /// Exactly what [`HeadlessDevice::new_async_with`] would request from `adapter` under these options: the
    /// features (`TIMESTAMP_QUERY`, `TIMESTAMP_QUERY_INSIDE_PASSES` and `MEMORY_DECORATION_COHERENT` where the adapter
    /// offers them, the [`features`](Self::features) policy's, `PASSTHROUGH_SHADERS` for the SPIR-V route on Vulkan,
    /// and the features of the [`capabilities`](Self::capabilities)) and the limits of the
    /// [`limits`](Self::limits) policy, with
    /// `max_storage_buffers_per_shader_stage` set to the number of
    /// [`with_storage_buffers`](Self::with_storage_buffers) when one was stated (refused as at open when the adapter
    /// offers fewer). The open itself calls this, so the request has one source of truth. `backends` and `adapter`
    /// are not read: you chose the adapter.
    ///
    /// A capability the adapter does not offer is [`Error::MissingCapability`], as at open, and in a browser a
    /// non-empty set is refused the same way.
    ///
    /// ```no_run
    /// use gust_wgpu::{DeviceOptions, HeadlessDevice, wgpu};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    /// let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    /// let request = DeviceOptions::default().device_request(&adapter)?;
    /// let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
    ///     required_features: request.features | (adapter.features() & wgpu::Features::FLOAT32_BLENDABLE),
    ///     required_limits: request.limits,
    ///     ..Default::default()
    /// }))?;
    /// let gust = HeadlessDevice::adopt(&adapter, device, queue, gust::capability::enabled())?;
    /// # let _ = gust;
    /// # Ok(())
    /// # }
    /// ```
    pub fn device_request(&self, adapter: &wgpu::Adapter) -> Result<DeviceRequest, Error> {
        let available = adapter.features();
        let capability_features = device::required_capability_features(
            self.capabilities,
            available,
            &adapter.get_info().name,
            cfg!(target_arch = "wasm32"),
        )?;
        let mut limits = self.limits.required_limits(adapter);
        if let Some(n) = device::required_storage_buffers(
            self.storage_buffers,
            adapter.limits().max_storage_buffers_per_shader_stage,
            &adapter.get_info().name,
        )? {
            limits.max_storage_buffers_per_shader_stage = n;
        }
        Ok(DeviceRequest {
            features: (available
                & (wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
                    | wgpu::Features::MEMORY_DECORATION_COHERENT))
                | self.features.required_features(adapter)
                | spirv_route_features(adapter)
                | capability_features,
            limits,
        })
    }
}

impl HeadlessDevice {
    /// Wrap a device the caller opened, so that a renderer and GUST share one device. `adapter` must be the one
    /// `device` was requested from; pass `gust::capability::enabled()` as `capabilities` unless you mean otherwise.
    /// Not async, and available on `wasm32`. Open the device with the features and limits of
    /// [`DeviceOptions::device_request`], ORed with your own.
    ///
    /// Refused: an adapter that is not the device's ([`Error::AdapterMismatch`], by comparing `device.adapter_info()`
    /// with `adapter.get_info()`), and a device that lacks the features a capability needs ([`Error::MissingFeatures`];
    /// on `wasm32` any non-empty capability set, as no capability can be requested there). A capability with no feature
    /// on this backend is [`Error::MissingCapability`] (D142). Nothing else is refused. The limits are the device's own
    /// (`device.limits()`; a kernel beyond them is [`Error::DeviceLimit`] at launch, and
    /// [`with_limits`](Self::with_limits) still lowers them). Optional features the device lacks are not errors, and
    /// each has a reporting accessor: [`supports_gpu_timestamps`](Self::supports_gpu_timestamps),
    /// [`supports_coherent_storage`](Self::supports_coherent_storage) and
    /// [`float32_filterable`](Self::float32_filterable) are false, a linear sampler is [`Error::TextureFilter`], and
    /// the SPIR-V route falls back to WGSL. Extra features and higher limits on the device are accepted and ignored.
    /// The backend is not checked either; adoption has evidence on Vulkan only (the browser path is not claimed yet).
    ///
    /// # Sharing a device
    ///
    /// - **One instance.** The surface, the adapter, the device and the queue must come from one `wgpu::Instance`.
    ///   Configuring a surface with a device from another instance panics inside wgpu-core, or silently binds the
    ///   surface to an unrelated device of the surface's own instance; the second case shows up later, as a validation
    ///   error or a panic far from the cause. GUST cannot check this. A device GUST opened itself lives in a private
    ///   instance and must never be given to a surface: use `adopt`.
    /// - **Adopt once** and share the [`DeviceHandle`](crate::DeviceHandle): two adoptions of one device are two GUST
    ///   devices whose buffers refuse to mix ([`Error::MixedDevices`]). Nothing detects it, because `wgpu::Device`
    ///   equality cannot tell instances apart.
    /// - **Waits are shared.** GUST's blocking calls wait for every submission on the queue, so a GUST readback also
    ///   waits for the renderer's frames in flight, and your callbacks can run inside a GUST call.
    /// - **Errors are shared.** An error scope you have open on the same thread also catches the errors of GUST's
    ///   infallible paths. GUST installs no uncaptured-error handler and no device-lost callback; yours stay in force
    ///   (with a handler that does not panic, `create_typed_buffer` no longer panics: use the `try_` forms).
    /// - GUST never calls `destroy`: dropping the `HeadlessDevice` leaves your clone alive. Your `destroy`, and a
    ///   device loss, end both users.
    /// - **wasm32:** the page creates its instance, canvas surface, adapter and device, then adopts with an empty
    ///   capability set and calls [`set_default_device`](crate::set_default_device) with an `Rc`.
    /// - The name `HeadlessDevice` stays: GUST itself never touches a window.
    pub fn adopt(
        adapter: &wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        capabilities: &'static [Capability],
    ) -> Result<Self, Error> {
        let info = adapter.get_info();
        check_same_adapter(&info, &device.adapter_info())?;
        let granted = device.features();
        for &capability in capabilities {
            let feature = device::capability_features(capability);
            if feature.is_empty() {
                // A capability this backend has no feature for is refused, never dropped silently (D142).
                return Err(Error::MissingCapability {
                    capability,
                    adapter: info.name,
                    kernel: None,
                });
            }
            let missing = if cfg!(target_arch = "wasm32") {
                feature
            } else {
                feature - granted
            };
            if !missing.is_empty() {
                return Err(Error::MissingFeatures {
                    adapter: info.name,
                    features: missing,
                    needed_by: format!("the `{}` capability", capability.name()),
                });
            }
        }
        Ok(Self::finish(adapter, device, queue, capabilities))
    }
}

/// `Error::AdapterMismatch` unless the adapter the caller names (`adapter`) and the one the device reports
/// (`device`) are described by the same `AdapterInfo`.
fn check_same_adapter(
    adapter: &wgpu::AdapterInfo,
    device: &wgpu::AdapterInfo,
) -> Result<(), Error> {
    if adapter == device {
        return Ok(());
    }
    Err(Error::AdapterMismatch {
        adapter: adapter.name.clone(),
        device_adapter: device.name.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(name: &str, device: u32) -> wgpu::AdapterInfo {
        let mut info = wgpu::AdapterInfo::new(wgpu::DeviceType::DiscreteGpu, wgpu::Backend::Vulkan);
        info.name = name.to_owned();
        info.device = device;
        info
    }

    /// The comparison `adopt` makes, on two fake descriptions: no adapter is needed.
    #[test]
    fn two_different_adapter_descriptions_are_an_adapter_mismatch() {
        let one = info("Fake Adapter One", 1);
        let two = info("Fake Adapter Two", 2);
        assert!(check_same_adapter(&one, &one.clone()).is_ok());
        let error = check_same_adapter(&one, &two).expect_err("different adapters");
        match &error {
            Error::AdapterMismatch {
                adapter,
                device_adapter,
            } => {
                assert_eq!(adapter, "Fake Adapter One");
                assert_eq!(device_adapter, "Fake Adapter Two");
            }
            other => panic!("expected AdapterMismatch, got {other:?}"),
        }
        assert_eq!(
            error.to_string(),
            "the adapter given to `adopt` (`Fake Adapter One`) is not the one the device was opened on (`Fake Adapter Two`)"
        );
        // One adapter name on two physical devices (a pair of identical cards) still differs by the rest of the info.
        let twin = info("Fake Adapter One", 3);
        assert!(check_same_adapter(&one, &twin).is_err());
    }
}
