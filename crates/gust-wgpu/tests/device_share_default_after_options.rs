//! `set_default_device` after `set_default_options` (m14.devshare.1, test 7, the other order): the options are
//! accepted and never used, and the set device is the default. Its own process, because the default device is
//! process-wide.

use gust::map::Broadcast;
use gust::math::Vec2;
use gust_wgpu::wgpu;
use gust_wgpu::{
    DeviceHandle, DeviceOptions, HeadlessDevice, LimitsPolicy, default_device, set_default_device,
    set_default_options,
};

include!("../../../tests/fixtures/map.rs");

#[test]
fn options_then_a_set_device_the_set_device_is_the_default() {
    set_default_options(DeviceOptions::default().with_limits(LimitsPolicy::Default))
        .expect("options before any device");

    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let adapter = pollster::block_on(wgpu::Instance::new(descriptor).request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        },
    ))
    .expect("a Vulkan adapter is required; this test must not silently skip");
    let options = DeviceOptions::default().with_limits(LimitsPolicy::Adapter);
    let request = options.device_request(&adapter).expect("the request");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: request.features,
        required_limits: request.limits.clone(),
        ..Default::default()
    }))
    .expect("a device");
    let handle = DeviceHandle::new(
        HeadlessDevice::adopt(&adapter, device, queue, options.capabilities).expect("adopts"),
    );
    println!("default adapter: {}", handle.adapter_info().name);
    set_default_device(DeviceHandle::clone(&handle)).expect("accepted after the options");

    let default = default_device().expect("the set device");
    assert!(std::sync::Arc::ptr_eq(&default, &handle));
    // The options (`LimitsPolicy::Default`) were never used: the device holds the limits it was opened with.
    assert_eq!(default.limits(), &request.limits);
    if adapter.limits().max_buffer_size == wgpu::Limits::default().max_buffer_size {
        println!("limits seam: not distinguishing, this adapter's buffer limit is wgpu's default");
    } else {
        assert_ne!(
            default.limits().max_buffer_size,
            wgpu::Limits::default().max_buffer_size,
            "the adapter's own buffer limit, not the options' portable one"
        );
    }
    assert_eq!(
        saxpy::map(Broadcast(2.0), &[1.0f32, 2.0], &[10.0, 20.0]).unwrap(),
        [12.0, 24.0]
    );
}
