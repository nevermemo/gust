//! `set_default_device` after `default_device()` (m14.devshare.1, test 7, the "opened" half of the first row of the
//! semantics table): the default device was opened by GUST, so a set device is `Error::DefaultDeviceOpened` and the
//! opened one stays. Its own process, because the default device is process-wide.

use gust_wgpu::wgpu;
use gust_wgpu::{
    DeviceHandle, DeviceOptions, Error, HeadlessDevice, default_device, set_default_device,
};

#[test]
fn a_default_device_that_is_open_refuses_a_set_device() {
    let opened = default_device().expect("the default device opens");

    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let adapter = pollster::block_on(wgpu::Instance::new(descriptor).request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        },
    ))
    .expect("a Vulkan adapter is required; this test must not silently skip");
    let options = DeviceOptions::default();
    let request = options.device_request(&adapter).expect("the request");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: request.features,
        required_limits: request.limits,
        ..Default::default()
    }))
    .expect("a device");
    let other = DeviceHandle::new(
        HeadlessDevice::adopt(&adapter, device, queue, options.capabilities).expect("adopts"),
    );

    match set_default_device(other) {
        Err(Error::DefaultDeviceOpened) => {}
        other => panic!("expected DefaultDeviceOpened, got {other:?}"),
    }
    assert!(std::sync::Arc::ptr_eq(
        &default_device().expect("still open"),
        &opened
    ));
}
