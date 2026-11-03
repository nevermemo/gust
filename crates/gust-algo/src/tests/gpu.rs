//! The reductions on a real adapter: the first Vulkan adapter wgpu finds, through the process's default device. The
//! cases are the twin's (`common`), against the same independent references. Nothing here skips: no adapter fails.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use gust_wgpu::{DeviceHandle, GpuVec, HeadlessDevice, default_device};

use super::common::{
    Backend, GPU_C, LENGTHS, SCAN_LENGTHS, Shape, assert_f32_sum, deep_scan_cases, empty_cases,
    f32_input, float_min_max_cases, float_sum_cases, integer_cases, reference_scan, scan_cases,
    tie_ladder, wrapping_cases,
};
use crate::{
    Error,
    reduce::{self, Element},
    scan,
    schedule::ScanSchedule,
};

/// One GPU test at a time. Once, in a Gpu-suite run, the blocking readback of a result failed with
/// `Error::Map("mapping not complete after poll(Wait)")` while other tests read back on the same device. The mechanism
/// (Opus review, read from wgpu-core): `Device::poll` fires only the callbacks its own `maintain` collected, so a
/// concurrent poll on another thread can take this thread's map callback, and this thread's single poll then sees the
/// mapping still pending. It is `gust-wgpu` code, slice m7.rb; this mutex goes when m7.rb lands (NEXT_TASKS).
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

fn device() -> DeviceHandle {
    let device =
        default_device().expect("a Vulkan adapter is required; these tests must not silently skip");
    eprintln!("gust-algo gpu adapter: {:?}", device.adapter_info());
    device
}

struct Gpu(DeviceHandle);

impl Backend for Gpu {
    const C: f64 = GPU_C;

    fn sum<T: Element>(&self, values: &[T]) -> T {
        let input = GpuVec::from_slice_on(&self.0, values).unwrap();
        reduce::sum(&input).unwrap()
    }

    fn min<T: Element>(&self, values: &[T]) -> Option<T> {
        let input = GpuVec::from_slice_on(&self.0, values).unwrap();
        match reduce::min(&input) {
            Ok(value) => Some(value),
            Err(Error::Empty) => None,
            Err(error) => panic!("{error}"),
        }
    }

    fn max<T: Element>(&self, values: &[T]) -> Option<T> {
        let input = GpuVec::from_slice_on(&self.0, values).unwrap();
        match reduce::max(&input) {
            Ok(value) => Some(value),
            Err(Error::Empty) => None,
            Err(error) => panic!("{error}"),
        }
    }

    fn sum_wrapping(&self, values: &[u32]) -> u32 {
        let input = GpuVec::from_slice_on(&self.0, values).unwrap();
        reduce::sum_wrapping(&input).unwrap()
    }

    fn inclusive(&self, values: &[u32]) -> Vec<u32> {
        let input = GpuVec::from_slice_on(&self.0, values).unwrap();
        let out = scan::inclusive(&input).unwrap();
        assert_eq!(out.len() as usize, values.len());
        out.to_vec().unwrap()
    }

    fn exclusive(&self, values: &[u32]) -> Vec<u32> {
        let input = GpuVec::from_slice_on(&self.0, values).unwrap();
        let out = scan::exclusive(&input).unwrap();
        assert_eq!(out.len() as usize, values.len());
        out.to_vec().unwrap()
    }
}

#[test]
fn gpu_integer_sums_min_and_max_are_exact_at_every_boundary() {
    let _serial = serial();
    integer_cases(&Gpu(device()), &LENGTHS);
}

#[test]
fn gpu_float_min_and_max_are_exact() {
    let _serial = serial();
    float_min_max_cases(&Gpu(device()), &LENGTHS);
}

#[test]
fn gpu_empty_inputs() {
    let _serial = serial();
    empty_cases(&Gpu(device()));
}

#[test]
fn gpu_float_sums_are_within_the_per_level_bound() {
    let _serial = serial();
    let gpu = Gpu(device());
    for n in [1, 255, 257, 65_537, 1_000_000] {
        for (shape, check) in float_sum_cases(&gpu, n) {
            eprintln!(
                "gpu f32 sum n = {n} {shape:?}: distance {:e}, bound {:e}, ratio {:.4}",
                check.distance,
                check.bound,
                check.ratio()
            );
        }
    }
}

/// The tie ladder on the adapter: whatever rounding the hardware does, the distance stays within the GPU bound.
#[test]
fn gpu_float_sum_of_the_tie_ladder_is_within_the_bound() {
    let _serial = serial();
    let gpu = Gpu(device());
    let values = tie_ladder(1 << 24);
    let actual = gpu.sum(&values);
    let check = assert_f32_sum("tie ladder", &values, actual, GPU_C);
    eprintln!(
        "gpu tie ladder: result {actual}, distance {:e}, bound {:e}, ratio {:.4}",
        check.distance,
        check.bound,
        check.ratio()
    );
}

#[test]
fn gpu_runs_the_deepest_schedule() {
    let _serial = serial();
    // 2^24 + 1 elements: three passes and the terminal kernel (four dispatches), as the twin test.
    let gpu = Gpu(device());
    let n = (1usize << 24) + 1;
    assert_eq!(gpu.sum(&vec![1u32; n]) as usize, n);
    let mut values = vec![5_000i32; n];
    values[n - 1] = -7;
    values[12_345_678] = 9_000;
    assert_eq!(gpu.min(&values), Some(-7));
    assert_eq!(gpu.max(&values), Some(9_000));
    // A float sum over four dispatches stays within the bound of the deepest schedule.
    let floats = f32_input(Shape::Uniform, n, 0xF00D);
    let check = assert_f32_sum("2^24 + 1 uniform", &floats, gpu.sum(&floats), GPU_C);
    eprintln!("gpu f32 sum n = {n}: ratio {:.4}", check.ratio());
}

#[test]
fn gpu_reductions_leave_the_input_alone_and_return_their_buffers() {
    let _serial = serial();
    // A device of its own: the pool counters of the process-wide default device move with every parallel test.
    let device = Arc::new(HeadlessDevice::new().expect("a Vulkan adapter is required"));
    let values: Vec<u32> = (0..70_001).map(|index| index % 100).collect();
    let input = GpuVec::from_slice_on(&device, &values).unwrap();
    let before = device.buffer_pool_stats();
    assert_eq!(
        reduce::sum(&input).unwrap() as usize,
        values.iter().map(|v| *v as usize).sum::<usize>()
    );
    assert_eq!(reduce::min(&input).unwrap(), 0);
    assert_eq!(reduce::max(&input).unwrap(), 99);
    assert_eq!(input.to_vec().unwrap(), values);
    let after = device.buffer_pool_stats();
    assert_eq!(
        after.outstanding, before.outstanding,
        "the partials and the result went back to the pool"
    );
    assert!(after.returned > before.returned);
    // A second round is served from the pool: no new allocation.
    let misses = after.misses;
    assert_eq!(
        reduce::sum(&input).unwrap() as usize,
        values.iter().map(|v| *v as usize).sum::<usize>()
    );
    assert_eq!(
        device.buffer_pool_stats().misses,
        misses,
        "the second reduction allocated"
    );
}

#[test]
fn gpu_sum_on_device_keeps_the_result_resident() {
    let _serial = serial();
    let device = device();
    let input = GpuVec::from_slice_on(&device, &[3u32; 1_000]).unwrap();
    let total = reduce::sum_on_device(&input).unwrap();
    assert_eq!(total.len(), 1);
    assert_eq!(total.to_vec().unwrap(), vec![3_000]);
    assert!(Arc::ptr_eq(total.device(), &device));
    // An empty input has the identity, still on the device and still one element.
    let empty = GpuVec::<u32>::zeroed_on(&device, 0).unwrap();
    assert_eq!(
        reduce::sum_on_device(&empty).unwrap().to_vec().unwrap(),
        vec![0]
    );
}

/// The documented wrap of an integer sum past the domain, pinned on the adapter (the debug twin panics instead, as
/// every `+` of the dialect does).
#[test]
fn gpu_integer_sums_past_the_domain_wrap() {
    let _serial = serial();
    let gpu = Gpu(device());
    assert_eq!(gpu.sum(&[u32::MAX, 1u32]), 0);
    assert_eq!(gpu.sum(&[i32::MAX, 1i32]), i32::MIN);
}

#[test]
fn gpu_min_of_an_empty_vector_is_empty_and_max_too() {
    let _serial = serial();
    let device = device();
    let empty = GpuVec::<f32>::zeroed_on(&device, 0).unwrap();
    assert!(matches!(reduce::min(&empty), Err(Error::Empty)));
    assert!(matches!(reduce::max(&empty), Err(Error::Empty)));
    assert_eq!(reduce::sum(&empty).unwrap().to_bits(), 0.0f32.to_bits());
}

/// An input past the storage binding limit is a `DeviceLimit` error before any buffer is taken or kernel compiled: the
/// pipeline cache and the pool are untouched.
#[test]
fn gpu_an_input_past_the_binding_limit_is_a_device_limit_error_before_any_dispatch() {
    let _serial = serial();
    // The portable limits: 128 MiB bindings, 2^25 four-byte elements.
    let device = Arc::new(HeadlessDevice::new().expect("a Vulkan adapter is required"));
    let limit = device.limits().max_storage_buffer_binding_size;
    let len = u32::try_from(limit / 4).unwrap() + 1;
    let input = GpuVec::<u32>::zeroed_on(&device, len).unwrap();
    let cache = device.pipeline_cache_stats();
    let pool = device.buffer_pool_stats();
    match reduce::sum(&input) {
        Err(Error::Device(gust_wgpu::Error::DeviceLimit {
            limit: name,
            requested,
            maximum,
            ..
        })) => {
            assert_eq!(name, "max_storage_buffer_binding_size");
            assert_eq!((requested, maximum), (limit + 4, limit));
        }
        other => panic!("expected a DeviceLimit error, got {other:?}"),
    }
    assert!(reduce::sum_on_device(&input).is_err());
    assert!(matches!(
        reduce::max(&input),
        Err(Error::Device(gust_wgpu::Error::DeviceLimit { .. }))
    ));
    assert_eq!(
        device.pipeline_cache_stats(),
        cache,
        "nothing was compiled or dispatched"
    );
    let after = device.buffer_pool_stats();
    assert_eq!(
        (after.hits, after.misses, after.outstanding),
        (pool.hits, pool.misses, pool.outstanding)
    );
    // One element fewer is inside the limit and reduces.
    let inside = GpuVec::<u32>::zeroed_on(&device, len - 1).unwrap();
    assert_eq!(reduce::sum(&inside).unwrap(), 0);
}

/// The first pass of an input launches `ceil(n / 65 536)` workgroups; a device that allows fewer refuses before any
/// dispatch.
#[test]
fn gpu_a_first_pass_past_the_workgroup_limit_is_a_device_limit_error_before_any_dispatch() {
    let _serial = serial();
    let mut limits = HeadlessDevice::new()
        .expect("a Vulkan adapter is required")
        .limits()
        .clone();
    limits.max_compute_workgroups_per_dimension = 3;
    let device = Arc::new(
        HeadlessDevice::new()
            .expect("a Vulkan adapter is required")
            .with_limits(limits)
            .unwrap(),
    );
    let inside = GpuVec::from_slice_on(&device, &vec![1u32; 3 * 65_536]).unwrap();
    assert_eq!(reduce::sum(&inside).unwrap(), 3 * 65_536);
    let outside = GpuVec::from_slice_on(&device, &vec![1u32; 3 * 65_536 + 1]).unwrap();
    let cache = device.pipeline_cache_stats();
    match reduce::sum(&outside) {
        Err(Error::Device(gust_wgpu::Error::DeviceLimit {
            limit,
            requested,
            maximum,
            ..
        })) => assert_eq!(
            (limit, requested, maximum),
            ("max_compute_workgroups_per_dimension", 4, 3)
        ),
        other => panic!("expected a DeviceLimit error, got {other:?}"),
    }
    assert_eq!(device.pipeline_cache_stats(), cache);
}

/// `sum_wrapping` is defined past `u32::MAX` on the adapter, where the plain `sum` is outside its domain: full range
/// values, all `u32::MAX`, the pair that wraps to zero, and the empty input.
#[test]
fn gpu_wrapping_sums_are_defined_modulo_2_32_for_every_input() {
    let _serial = serial();
    wrapping_cases(&Gpu(device()), &LENGTHS);
}

#[test]
fn gpu_scans_are_exact_at_every_boundary() {
    let _serial = serial();
    scan_cases(&Gpu(device()), &SCAN_LENGTHS);
}

/// 2^24 elements are five dispatches and 2^24 + 1 are seven (a third level of totals, the add-back through it).
#[test]
fn gpu_runs_the_deepest_scans() {
    let _serial = serial();
    deep_scan_cases(&Gpu(device()));
}

/// The dispatch count of a scan, measured: every dispatch of a batch takes one pipeline from the device's cache (a hit
/// or a miss), so the counters of a device of its own move by the number of dispatches. 3 up to 65 536 elements, 5 up
/// to 2^24, 7 above (the level-3 path); `exclusive` costs the same as `inclusive`, and an empty input dispatches
/// nothing.
#[test]
fn gpu_a_scan_is_one_batch_of_three_five_or_seven_dispatches() {
    let _serial = serial();
    let device = Arc::new(HeadlessDevice::new().expect("a Vulkan adapter is required"));
    let touched = |device: &HeadlessDevice| {
        let stats = device.pipeline_cache_stats();
        stats.hits + stats.misses
    };
    for (n, dispatches) in [
        (1usize, 3usize),
        (256, 3),
        (257, 3),
        (65_536, 3),
        (65_537, 5),
        (1 << 24, 5),
        ((1 << 24) + 1, 7),
    ] {
        assert_eq!(
            ScanSchedule::new(n as u32).unwrap().dispatches(),
            dispatches,
            "n = {n}"
        );
        let input = GpuVec::<u32>::from_slice_on(&device, &vec![1u32; n]).unwrap();
        for exclusive in [false, true] {
            let before = touched(&device);
            let out = if exclusive {
                scan::exclusive(&input)
            } else {
                scan::inclusive(&input)
            }
            .unwrap();
            assert_eq!(
                touched(&device) - before,
                dispatches as u64,
                "dispatches of the {} scan, n = {n}",
                if exclusive { "exclusive" } else { "inclusive" }
            );
            let last = out.to_vec().unwrap()[n - 1];
            assert_eq!(last as usize, if exclusive { n - 1 } else { n }, "n = {n}");
        }
    }
    let empty = GpuVec::<u32>::zeroed_on(&device, 0).unwrap();
    let before = touched(&device);
    assert!(scan::inclusive(&empty).unwrap().is_empty());
    assert!(scan::exclusive(&empty).unwrap().is_empty());
    assert_eq!(touched(&device), before, "an empty scan dispatches nothing");
}

#[test]
fn gpu_scans_leave_the_input_alone_and_return_their_buffers() {
    let _serial = serial();
    // A device of its own: the pool counters of the process-wide default device move with every parallel test.
    let device = Arc::new(HeadlessDevice::new().expect("a Vulkan adapter is required"));
    let values: Vec<u32> = (0..70_001).map(|index| index % 100).collect();
    let input = GpuVec::from_slice_on(&device, &values).unwrap();
    let before = device.buffer_pool_stats();
    let inclusive = scan::inclusive(&input).unwrap();
    let exclusive = scan::exclusive(&input).unwrap();
    assert_eq!(input.to_vec().unwrap(), values);
    assert!(Arc::ptr_eq(inclusive.device(), &device));
    assert_eq!(inclusive.to_vec().unwrap(), reference_scan(&values, false));
    assert_eq!(exclusive.to_vec().unwrap(), reference_scan(&values, true));
    drop((inclusive, exclusive));
    let after = device.buffer_pool_stats();
    assert_eq!(
        after.outstanding, before.outstanding,
        "the totals and the results went back to the pool"
    );
    // A second round is served from the pool: no new allocation.
    let misses = device.buffer_pool_stats().misses;
    drop(scan::inclusive(&input).unwrap());
    assert_eq!(
        device.buffer_pool_stats().misses,
        misses,
        "the second scan allocated"
    );
}

/// The documented wrap of a scan past the domain, pinned on the adapter (the debug twin panics instead, as every `+`
/// of the dialect does).
#[test]
fn gpu_scans_past_the_domain_wrap() {
    let _serial = serial();
    let device = device();
    let input = GpuVec::from_slice_on(&device, &[u32::MAX, 2u32, 3]).unwrap();
    assert_eq!(
        scan::inclusive(&input).unwrap().to_vec().unwrap(),
        vec![u32::MAX, 1, 4]
    );
}

/// An input past the storage binding limit is a `DeviceLimit` error before any buffer is taken or kernel compiled: the
/// pipeline cache and the pool are untouched.
#[test]
fn gpu_a_scan_past_the_binding_limit_is_a_device_limit_error_before_any_dispatch() {
    let _serial = serial();
    // The portable limits: 128 MiB bindings, 2^25 four-byte elements.
    let device = Arc::new(HeadlessDevice::new().expect("a Vulkan adapter is required"));
    let limit = device.limits().max_storage_buffer_binding_size;
    let len = u32::try_from(limit / 4).unwrap() + 1;
    let input = GpuVec::<u32>::zeroed_on(&device, len).unwrap();
    let cache = device.pipeline_cache_stats();
    let pool = device.buffer_pool_stats();
    for result in [scan::inclusive(&input), scan::exclusive(&input)] {
        match result {
            Err(Error::Device(gust_wgpu::Error::DeviceLimit {
                limit: name,
                requested,
                maximum,
                ..
            })) => {
                assert_eq!(name, "max_storage_buffer_binding_size");
                assert_eq!((requested, maximum), (limit + 4, limit));
            }
            other => panic!("expected a DeviceLimit error, got {other:?}"),
        }
    }
    assert_eq!(
        device.pipeline_cache_stats(),
        cache,
        "nothing was compiled or dispatched"
    );
    let after = device.buffer_pool_stats();
    assert_eq!(
        (after.hits, after.misses, after.outstanding),
        (pool.hits, pool.misses, pool.outstanding)
    );
    // One element fewer is inside the limit and scans.
    let inside = GpuVec::<u32>::zeroed_on(&device, len - 1).unwrap();
    let out = scan::inclusive(&inside).unwrap();
    assert_eq!(out.len(), len - 1);
}

/// Level 0 of a scan launches `ceil(ceil(n / 256) / 256)` workgroups; a device that allows fewer refuses before any
/// dispatch.
#[test]
fn gpu_a_scan_past_the_workgroup_limit_is_a_device_limit_error_before_any_dispatch() {
    let _serial = serial();
    let mut limits = HeadlessDevice::new()
        .expect("a Vulkan adapter is required")
        .limits()
        .clone();
    limits.max_compute_workgroups_per_dimension = 3;
    let device = Arc::new(
        HeadlessDevice::new()
            .expect("a Vulkan adapter is required")
            .with_limits(limits)
            .unwrap(),
    );
    // 3 workgroups of 256 invocations, one invocation per chunk of 256: 196 608 elements.
    let inside = GpuVec::from_slice_on(&device, &vec![1u32; 3 * 256 * 256]).unwrap();
    assert_eq!(
        scan::inclusive(&inside).unwrap().to_vec().unwrap()[3 * 256 * 256 - 1],
        3 * 256 * 256
    );
    let outside = GpuVec::from_slice_on(&device, &vec![1u32; 3 * 256 * 256 + 1]).unwrap();
    let cache = device.pipeline_cache_stats();
    match scan::exclusive(&outside) {
        Err(Error::Device(gust_wgpu::Error::DeviceLimit {
            limit,
            requested,
            maximum,
            ..
        })) => assert_eq!(
            (limit, requested, maximum),
            ("max_compute_workgroups_per_dimension", 4, 3)
        ),
        other => panic!("expected a DeviceLimit error, got {other:?}"),
    }
    assert_eq!(device.pipeline_cache_stats(), cache);
}
