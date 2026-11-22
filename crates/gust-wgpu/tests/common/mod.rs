//! Helpers shared by the integration tests of this crate (m13.dedup-tests). A test file pulls them in with `mod common;`.
//!
//! Each test file is its own binary and uses only some of these, so the ones it leaves out are dead code there.
#![allow(dead_code)]
#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]

use std::{
    future::Future,
    pin::pin,
    task::{Context, Poll, Waker},
    thread,
    time::{Duration, Instant},
};

use gust_wgpu::{DeviceHandle, HeadlessDevice};

/// A headless device on the default Vulkan adapter. Panics when there is none: a GPU test must not skip silently.
pub fn device() -> HeadlessDevice {
    HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip")
}

/// [`device`] behind the shared handle that `GpuVec`, the map calls and the concurrency tests take.
pub fn device_handle() -> DeviceHandle {
    std::sync::Arc::new(device())
}

/// [`device`], printing the adapter's name to stderr, led by `tag`.
pub fn device_logged(tag: &str) -> HeadlessDevice {
    let device = device();
    eprintln!("{tag} adapter: {}", device.adapter_info().name);
    device
}

/// [`device`], printing the adapter's name and backend to stdout.
pub fn device_reported() -> HeadlessDevice {
    let device = device();
    let info = device.adapter_info();
    println!("adapter: {} backend: {:?}", info.name, info.backend);
    device
}

/// Whether the adapter is a software rasteriser (lavapipe, which the public CI runs the suite on).
pub fn is_cpu_adapter(device: &HeadlessDevice) -> bool {
    device.adapter_info().device_type == wgpu::DeviceType::Cpu
}

/// Assert that `device` has compiled nothing and looked nothing up: `why` (for example "a limit error") must come
/// before compilation and the pipeline cache.
#[track_caller]
pub fn assert_cold(device: &HeadlessDevice, why: &str) {
    let stats = device.pipeline_cache_stats();
    assert_eq!(
        (stats.entries, stats.misses, stats.hits),
        (0, 0, 0),
        "{why} must precede compilation and the cache lookup"
    );
}

/// Poll `future` every millisecond until it is ready, and return its output and how often it was polled. Each poll of
/// a future that waits on the device polls the device once; the 60 s deadline turns a future that never completes
/// into a failure.
pub fn drive_counted<F: Future>(future: F) -> (F::Output, u32) {
    // A waker that does nothing: this executor does not wait for wakes.
    let mut context = Context::from_waker(Waker::noop());
    let mut future = pin!(future);
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut polls = 0;
    loop {
        polls += 1;
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return (output, polls);
        }
        assert!(
            Instant::now() < deadline,
            "the future was still pending after {polls} polls"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

/// [`drive_counted`] without the poll count.
pub fn drive<F: Future>(future: F) -> F::Output {
    drive_counted(future).0
}

/// The unit in the last place of a normal `f32`.
pub fn ulp(x: f32) -> f64 {
    let magnitude = x.abs().max(f32::MIN_POSITIVE);
    let exponent = ((magnitude.to_bits() >> 23) & 0xff) as i32 - 127;
    2f64.powi(exponent - 23)
}

/// Whether every kernel this device built so far took the SPIR-V route (m9.rt2). `false` without the feature.
#[cfg(feature = "spirv-route")]
pub fn spirv_route_taken(device: &HeadlessDevice) -> bool {
    let events = device.route_events();
    !events.is_empty()
        && events
            .iter()
            .all(|event| event.route == gust_wgpu::Route::Spirv)
}

/// Whether every kernel this device built so far took the SPIR-V route (m9.rt2). `false` without the feature.
#[cfg(not(feature = "spirv-route"))]
pub fn spirv_route_taken(_device: &HeadlessDevice) -> bool {
    false
}
