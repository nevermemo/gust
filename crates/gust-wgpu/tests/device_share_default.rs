//! `set_default_device` (m14.devshare.1, test 7), the order "set a device, then everything else". One process, one
//! test: the default device is process-wide, so the steps run in sequence. The other order (options first) is
//! `device_share_default_after_options.rs`.

use gust::capability::Capability;
use gust::map::Broadcast;
use gust::math::Vec2;
use gust_wgpu::wgpu;
use gust_wgpu::{
    DeviceHandle, DeviceOptions, Error, GpuVec, HeadlessDevice, default_device, set_default_device,
    set_default_options,
};

include!("../../../tests/fixtures/map.rs");

/// A device opened in an instance of the caller's own and adopted with `capabilities`.
fn adopted(capabilities: &'static [Capability]) -> HeadlessDevice {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let adapter = pollster::block_on(wgpu::Instance::new(descriptor).request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        },
    ))
    .expect("a Vulkan adapter is required; this test must not silently skip");
    let request = DeviceOptions::default()
        .with_capabilities(capabilities)
        .device_request(&adapter)
        .expect("the adapter offers the capabilities");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: request.features,
        required_limits: request.limits,
        ..Default::default()
    }))
    .expect("a device");
    HeadlessDevice::adopt(&adapter, device, queue, capabilities).expect("adopts")
}

#[test]
fn a_set_device_is_the_default_device_and_nothing_replaces_it() {
    // Under a `+subgroups` build a device adopted without the capability is refused, and refusing leaves the slot free.
    if gust::capability::enabled().is_empty() {
        println!(
            "capability refusal: not run, this build holds no capability (run it under dev+subgroups)"
        );
    } else {
        let bare = DeviceHandle::new(adopted(&[]));
        match set_default_device(bare) {
            Err(Error::DefaultDeviceCapability {
                capability,
                adapter,
            }) => {
                println!("capability refusal: {capability:?} on {adapter}");
                assert!(gust::capability::enabled().contains(&capability));
            }
            other => panic!("expected DefaultDeviceCapability, got {other:?}"),
        }
    }

    let handle = DeviceHandle::new(adopted(gust::capability::enabled()));
    println!("default adapter: {}", handle.adapter_info().name);
    set_default_device(DeviceHandle::clone(&handle)).expect("the first set is accepted");

    // `default_device()` is the set device, and the front-door calls run on it.
    let default = default_device().expect("the set device");
    assert!(std::sync::Arc::ptr_eq(&default, &handle));
    let x: Vec<f32> = (0..300).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..300).map(|i| (i % 7) as f32).collect();
    let expected: Vec<f32> = x.iter().zip(&y).map(|(x, y)| 2.0 * x + y).collect();
    assert_eq!(saxpy::map(Broadcast(2.0), &x, &y).unwrap(), expected);
    let resident = GpuVec::from_slice(&x).unwrap();
    assert_eq!(resident.device_id(), handle.id());
    let mut doubled = vec![0.0_f32; x.len()];
    let n = x.len() as u32;
    gust::par_for!(i in 0..n, [x: &[f32], doubled: &mut [f32]] {
        doubled[i] = x[i] * 2.0;
    })
    .unwrap();
    assert_eq!(doubled, x.iter().map(|v| v * 2.0).collect::<Vec<_>>());

    // A second set, even of the same handle, and options after a set are both `DefaultDeviceOpened`, with the new text.
    let text = "the default device already exists; call `set_default_options` or `set_default_device` before the first `default_device()`";
    match set_default_device(DeviceHandle::clone(&handle)) {
        Err(error @ Error::DefaultDeviceOpened) => assert_eq!(error.to_string(), text),
        other => panic!("expected DefaultDeviceOpened for a second set, got {other:?}"),
    }
    match set_default_options(DeviceOptions::default()) {
        Err(error @ Error::DefaultDeviceOpened) => assert_eq!(error.to_string(), text),
        other => panic!("expected DefaultDeviceOpened for options after a set, got {other:?}"),
    }
}
