//! The algorithms on a real adapter: the first Vulkan adapter wgpu finds, through the process's default device. The
//! cases are the twin's (`common`), against the same independent references. Nothing here skips: no adapter fails.
//!
//! The tests run in parallel, on the default test threads. They used to share a mutex because one blocking readback
//! once failed with `Error::Map("mapping not complete after poll(Wait)")` while other threads polled the same device;
//! D97 (m7.rb) made blocking readbacks safe under concurrent polls, and the mutex went with m7.r3.

use std::sync::Arc;

use gust_wgpu::{DeviceHandle, GpuVec, HeadlessDevice, default_device};

use super::common::{
    Backend, COMPACT_LENGTHS, GPU_C, LENGTHS, SCAN_LENGTHS, Shape, assert_f32_sum, compact_cases,
    deep_compact_cases, deep_scan_cases, empty_cases, f32_input, float_min_max_cases,
    float_sum_cases, gather_edge_cases, integer_cases, out_of_range_index_cases, reference_indices,
    reference_scan, scan_cases, tie_ladder, wrapping_cases,
};
use super::struct_gather::{particles, population, reference};
use crate::{
    Error, compact,
    reduce::{self, Element},
    scan,
    schedule::{ScanSchedule, Schedule},
};

fn device() -> DeviceHandle {
    let device =
        default_device().expect("a Vulkan adapter is required; these tests must not silently skip");
    eprintln!("gust-algo gpu adapter: {:?}", device.adapter_info());
    device
}

struct Gpu(DeviceHandle);

impl Gpu {
    /// The three inputs of a gather on this device.
    fn upload<T: gust::GpuPod>(
        &self,
        values: &[T],
        indices: &[u32],
        count: u32,
    ) -> (GpuVec<T>, GpuVec<u32>, GpuVec<u32>) {
        (
            GpuVec::from_slice_on(&self.0, values).unwrap(),
            GpuVec::from_slice_on(&self.0, indices).unwrap(),
            GpuVec::from_slice_on(&self.0, &[count]).unwrap(),
        )
    }
}

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

    fn compact_count(&self, flags: &[u32]) -> u32 {
        let flags = GpuVec::from_slice_on(&self.0, flags).unwrap();
        let count = compact::count(&flags).unwrap();
        assert_eq!(count.len(), 1);
        count.to_vec().unwrap()[0]
    }

    fn compact_indices(&self, flags: &[u32]) -> (Vec<u32>, u32) {
        let flags = GpuVec::from_slice_on(&self.0, flags).unwrap();
        let (indices, count) = compact::indices(&flags).unwrap();
        assert_eq!(indices.len(), flags.len());
        assert_eq!(count.len(), 1);
        (indices.to_vec().unwrap(), count.to_vec().unwrap()[0])
    }

    fn gather_u32(&self, values: &[u32], indices: &[u32], count: u32) -> Vec<u32> {
        let (values, indices, count) = self.upload(values, indices, count);
        let out = compact::gather_u32(&values, &indices, &count).unwrap();
        assert_eq!(out.len(), indices.len());
        out.to_vec().unwrap()
    }

    fn gather_i32(&self, values: &[i32], indices: &[u32], count: u32) -> Vec<i32> {
        let (values, indices, count) = self.upload(values, indices, count);
        let out = compact::gather_i32(&values, &indices, &count).unwrap();
        assert_eq!(out.len(), indices.len());
        out.to_vec().unwrap()
    }

    fn gather_f32(&self, values: &[f32], indices: &[u32], count: u32) -> Vec<f32> {
        let (values, indices, count) = self.upload(values, indices, count);
        let out = compact::gather_f32(&values, &indices, &count).unwrap();
        assert_eq!(out.len(), indices.len());
        out.to_vec().unwrap()
    }
}

#[test]
fn gpu_integer_sums_min_and_max_are_exact_at_every_boundary() {
    integer_cases(&Gpu(device()), &LENGTHS);
}

#[test]
fn gpu_float_min_and_max_are_exact() {
    float_min_max_cases(&Gpu(device()), &LENGTHS);
}

#[test]
fn gpu_empty_inputs() {
    empty_cases(&Gpu(device()));
}

#[test]
fn gpu_float_sums_are_within_the_per_level_bound() {
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
    let gpu = Gpu(device());
    assert_eq!(gpu.sum(&[u32::MAX, 1u32]), 0);
    assert_eq!(gpu.sum(&[i32::MAX, 1i32]), i32::MIN);
}

#[test]
fn gpu_min_of_an_empty_vector_is_empty_and_max_too() {
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
    wrapping_cases(&Gpu(device()), &LENGTHS);
}

#[test]
fn gpu_scans_are_exact_at_every_boundary() {
    scan_cases(&Gpu(device()), &SCAN_LENGTHS);
}

/// 2^24 elements are five dispatches and 2^24 + 1 are seven (a third level of totals, the add-back through it).
#[test]
fn gpu_runs_the_deepest_scans() {
    deep_scan_cases(&Gpu(device()));
}

/// The dispatch count of a scan, measured: every dispatch of a batch takes one pipeline from the device's cache (a hit
/// or a miss), so the counters of a device of its own move by the number of dispatches. 3 up to 65 536 elements, 5 up
/// to 2^24, 7 above (the level-3 path); `exclusive` costs the same as `inclusive`, and an empty input dispatches
/// nothing.
#[test]
fn gpu_a_scan_is_one_batch_of_three_five_or_seven_dispatches() {
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
    // One element fewer is inside the limit and scans: ones, so the values at the top of the range (the seven-dispatch
    // path at 2^25 elements) are checked too.
    drop(input);
    let inside = GpuVec::from_slice_on(&device, &vec![1u32; (len - 1) as usize]).unwrap();
    let out = scan::inclusive(&inside).unwrap();
    assert_eq!(out.len(), len - 1);
    let values = out.to_vec().unwrap();
    assert!(
        values
            .iter()
            .enumerate()
            .all(|(index, value)| *value as usize == index + 1),
        "the inclusive scan of ones at the top of the range"
    );
}

/// Level 0 of a scan launches `ceil(ceil(n / 256) / 256)` workgroups; a device that allows fewer refuses before any
/// dispatch.
#[test]
fn gpu_a_scan_past_the_workgroup_limit_is_a_device_limit_error_before_any_dispatch() {
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

#[test]
fn gpu_compaction_is_exact_at_every_boundary() {
    compact_cases(&Gpu(device()), &COMPACT_LENGTHS);
}

/// `2^24 + 1` flags: 257 workgroups of the block-strided mask and scatter launches (an invocation per element would be
/// 65 537 workgroups, past the 65 535 ceiling), and a scan of seven dispatches.
#[test]
fn gpu_runs_the_deepest_compaction() {
    deep_compact_cases(&Gpu(device()));
}

/// An index that is not below `values.len()` leaves its slot zero, and a count caps what is gathered.
#[test]
fn gpu_a_gather_guards_the_index_bound_and_the_count() {
    let gpu = Gpu(device());
    out_of_range_index_cases(&gpu);
    gather_edge_cases(&gpu);
}

/// The flag rule on the adapter: any non-zero flag is set. `2`, `0x8000_0000` and `u32::MAX` are one flag each; the
/// count is 4, not their sum, and the scan sees ones.
#[test]
fn gpu_non_zero_flags_count_as_set() {
    let gpu = Gpu(device());
    let flags = [0u32, 2, 0, 0x8000_0000, u32::MAX, 1, 0];
    assert_eq!(gpu.compact_count(&flags), 4);
    assert_eq!(gpu.compact_indices(&flags), (vec![1, 3, 4, 5, 0, 0, 0], 4));
}

/// The empty compaction: the count is a resident one-element zero, the indices are empty, a gather of nothing is
/// nothing.
#[test]
fn gpu_an_empty_compaction_has_a_resident_zero_count() {
    let device = device();
    let flags = GpuVec::<u32>::zeroed_on(&device, 0).unwrap();
    let count = compact::count(&flags).unwrap();
    assert_eq!((count.len(), count.to_vec().unwrap()), (1, vec![0]));
    assert!(Arc::ptr_eq(count.device(), &device));
    let (indices, total) = compact::indices(&flags).unwrap();
    assert!(indices.is_empty());
    assert_eq!(total.to_vec().unwrap(), vec![0]);
    let gathered = compact::gather_u32(&flags, &indices, &total).unwrap();
    assert!(gathered.is_empty());
}

/// `flags -> indices -> gather` with the count produced on the device and never read: the gathers take the `count`
/// buffer `indices` returned. Only the final results are read, for the comparison.
#[test]
fn gpu_a_compaction_chain_runs_on_the_device_with_the_device_count() {
    let device = device();
    let n = 300_001usize;
    let flags: Vec<u32> = (0..n)
        .map(|i| u32::from(i % 7 == 0 || i % 11 == 3))
        .collect();
    let values: Vec<u32> = (0..n as u32).map(|i| i ^ 0x5555).collect();
    let floats: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 + 0.25).collect();
    let reals: Vec<i32> = (0..n as i32).map(|i| i - 150_000).collect();
    let on_device = GpuVec::from_slice_on(&device, &flags).unwrap();
    let (indices, count) = compact::indices(&on_device).unwrap();
    let source_u32 = GpuVec::from_slice_on(&device, &values).unwrap();
    let source_f32 = GpuVec::from_slice_on(&device, &floats).unwrap();
    let source_i32 = GpuVec::from_slice_on(&device, &reals).unwrap();
    let dense_u32 = compact::gather_u32(&source_u32, &indices, &count).unwrap();
    let dense_f32 = compact::gather_f32(&source_f32, &indices, &count).unwrap();
    let dense_i32 = compact::gather_i32(&source_i32, &indices, &count).unwrap();
    let expected = reference_indices(&flags);
    let live = expected.len();
    assert_eq!(count.to_vec().unwrap(), vec![live as u32]);
    let (dense_u32, dense_f32, dense_i32) = (
        dense_u32.to_vec().unwrap(),
        dense_f32.to_vec().unwrap(),
        dense_i32.to_vec().unwrap(),
    );
    assert_eq!(dense_u32.len(), n, "the gather is as long as the indices");
    for (j, index) in expected.iter().enumerate() {
        let index = *index as usize;
        assert_eq!(dense_u32[j], values[index], "u32, slot {j}");
        assert_eq!(
            dense_f32[j].to_bits(),
            floats[index].to_bits(),
            "f32, slot {j}"
        );
        assert_eq!(dense_i32[j], reals[index], "i32, slot {j}");
    }
    assert!(dense_u32[live..].iter().all(|value| *value == 0));
    assert!(dense_f32[live..].iter().all(|value| value.to_bits() == 0));
    assert!(dense_i32[live..].iter().all(|value| *value == 0));
}

/// The dispatch count of each call, measured on a device of its own (every dispatch of a batch takes one pipeline from
/// the cache, a hit or a miss): `count` is the mask and the reduction, `indices` the mask, the scan, the reduction and
/// the scatter, a gather is one. The numbers come from the schedules the other calls are pinned to.
#[test]
fn gpu_compaction_dispatch_counts_are_the_mask_the_reduction_the_scan_and_the_scatter() {
    let device = Arc::new(HeadlessDevice::new().expect("a Vulkan adapter is required"));
    let touched = |device: &HeadlessDevice| {
        let stats = device.pipeline_cache_stats();
        stats.hits + stats.misses
    };
    for n in [1usize, 256, 257, 65_537] {
        let reduction = Schedule::new(n as u32).unwrap().dispatches() as u64;
        let scan = ScanSchedule::new(n as u32).unwrap().dispatches() as u64;
        let flags = GpuVec::<u32>::from_slice_on(&device, &vec![1u32; n]).unwrap();
        let before = touched(&device);
        let count = compact::count(&flags).unwrap();
        assert_eq!(touched(&device) - before, 1 + reduction, "count, n = {n}");
        let before = touched(&device);
        let (indices, total) = compact::indices(&flags).unwrap();
        assert_eq!(
            touched(&device) - before,
            1 + scan + reduction + 1,
            "indices, n = {n}"
        );
        let values = GpuVec::<u32>::from_slice_on(&device, &vec![5u32; n]).unwrap();
        let before = touched(&device);
        let out = compact::gather_u32(&values, &indices, &total).unwrap();
        assert_eq!(touched(&device) - before, 1, "gather, n = {n}");
        assert_eq!(count.to_vec().unwrap(), vec![n as u32]);
        assert_eq!(out.to_vec().unwrap()[n - 1], 5);
    }
    // Nothing to do, nothing dispatched.
    let empty = GpuVec::<u32>::zeroed_on(&device, 0).unwrap();
    let before = touched(&device);
    drop(compact::count(&empty).unwrap());
    drop(compact::indices(&empty).unwrap());
    assert_eq!(
        touched(&device),
        before,
        "an empty compaction dispatches nothing"
    );
}

#[test]
fn gpu_compaction_leaves_the_flags_alone_and_returns_its_buffers() {
    // A device of its own: the pool counters of the process-wide default device move with every parallel test.
    let device = Arc::new(HeadlessDevice::new().expect("a Vulkan adapter is required"));
    let flags: Vec<u32> = (0..70_001).map(|index| index % 3).collect();
    let values: Vec<u32> = (0..70_001).collect();
    let on_device = GpuVec::from_slice_on(&device, &flags).unwrap();
    let source = GpuVec::from_slice_on(&device, &values).unwrap();
    let before = device.buffer_pool_stats();
    let expected = reference_indices(&flags);
    let live = expected.len() as u32;
    let round = || {
        let count = compact::count(&on_device).unwrap();
        let (indices, total) = compact::indices(&on_device).unwrap();
        let gathered = compact::gather_u32(&source, &indices, &total).unwrap();
        assert_eq!(count.to_vec().unwrap(), vec![live]);
        assert_eq!(total.to_vec().unwrap(), vec![live]);
        assert_eq!(gathered.to_vec().unwrap()[..live as usize], expected[..]);
    };
    round();
    assert_eq!(on_device.to_vec().unwrap(), flags);
    assert_eq!(source.to_vec().unwrap(), values);
    let after = device.buffer_pool_stats();
    assert_eq!(
        after.outstanding, before.outstanding,
        "the mask, the positions, the totals and the results went back to the pool"
    );
    // A second round is served from the pool: no new allocation.
    let misses = after.misses;
    round();
    assert_eq!(
        device.buffer_pool_stats().misses,
        misses,
        "the second compaction allocated"
    );
}

/// A buffer past the storage binding limit is a `DeviceLimit` error before any buffer is taken or kernel compiled: the
/// pipeline cache and the pool are untouched. The flags of `count` and `indices`, the values and the indices of a
/// gather are each the buffer in turn.
#[test]
fn gpu_a_compaction_past_the_binding_limit_is_a_device_limit_error_before_any_dispatch() {
    // The portable limits: 128 MiB bindings, 2^25 four-byte elements.
    let device = Arc::new(HeadlessDevice::new().expect("a Vulkan adapter is required"));
    let limit = device.limits().max_storage_buffer_binding_size;
    let len = u32::try_from(limit / 4).unwrap() + 1;
    let small = GpuVec::from_slice_on(&device, &[0u32]).unwrap();
    let small_signed = GpuVec::from_slice_on(&device, &[0i32]).unwrap();
    let one = GpuVec::from_slice_on(&device, &[1u32]).unwrap();
    let big = GpuVec::<u32>::zeroed_on(&device, len).unwrap();
    let cache = device.pipeline_cache_stats();
    let pool = device.buffer_pool_stats();
    let expect_limit = |which: &str, result: Result<(), Error>| match result {
        Err(Error::Device(gust_wgpu::Error::DeviceLimit {
            limit: name,
            requested,
            maximum,
            ..
        })) => {
            assert_eq!(name, "max_storage_buffer_binding_size", "call {which}");
            assert_eq!((requested, maximum), (limit + 4, limit), "call {which}");
        }
        other => panic!("call {which}: expected a DeviceLimit error, got {other:?}"),
    };
    expect_limit("count", compact::count(&big).map(drop));
    expect_limit("indices", compact::indices(&big).map(drop));
    expect_limit(
        "gather_u32 values",
        compact::gather_u32(&big, &small, &one).map(drop),
    );
    expect_limit(
        "gather_u32 indices",
        compact::gather_u32(&small, &big, &one).map(drop),
    );
    expect_limit(
        "gather_i32 indices",
        compact::gather_i32(&small_signed, &big, &one).map(drop),
    );
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
    // The f32 gather checks its values the same way; one big buffer at a time keeps the footprint down.
    drop(big);
    let big_float = GpuVec::<f32>::zeroed_on(&device, len).unwrap();
    let cache = device.pipeline_cache_stats();
    expect_limit(
        "gather_f32 values",
        compact::gather_f32(&big_float, &small, &one).map(drop),
    );
    assert_eq!(device.pipeline_cache_stats(), cache);
    drop(big_float);
    // One element fewer is inside the limit and compacts: the top of the range, the last slot set.
    let mut flags = vec![0u32; (len - 1) as usize];
    *flags.last_mut().unwrap() = 1;
    let inside = GpuVec::from_slice_on(&device, &flags).unwrap();
    let (indices, count) = compact::indices(&inside).unwrap();
    assert_eq!(count.to_vec().unwrap(), vec![1]);
    assert_eq!(indices.to_vec().unwrap()[0], len - 2);
}

/// The element-wise launches are `ceil(n / 65 536)` workgroups; a device that allows fewer refuses before any dispatch,
/// for the flags of `count` and `indices` and for the indices of a gather.
#[test]
fn gpu_a_compaction_past_the_workgroup_limit_is_a_device_limit_error_before_any_dispatch() {
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
    assert_eq!(
        compact::count(&inside).unwrap().to_vec().unwrap(),
        vec![3 * 65_536]
    );
    let (indices, count) = compact::indices(&inside).unwrap();
    assert_eq!(indices.to_vec().unwrap()[3 * 65_536 - 1], 3 * 65_536 - 1);
    assert_eq!(
        compact::gather_u32(&inside, &indices, &count)
            .unwrap()
            .len(),
        3 * 65_536
    );
    let outside = GpuVec::from_slice_on(&device, &vec![1u32; 3 * 65_536 + 1]).unwrap();
    let one = GpuVec::from_slice_on(&device, &[1u32]).unwrap();
    let cache = device.pipeline_cache_stats();
    for (which, result) in [
        compact::count(&outside).map(drop),
        compact::indices(&outside).map(drop),
        compact::gather_u32(&one, &outside, &one).map(drop),
    ]
    .into_iter()
    .enumerate()
    {
        match result {
            Err(Error::Device(gust_wgpu::Error::DeviceLimit {
                limit,
                requested,
                maximum,
                ..
            })) => assert_eq!(
                (limit, requested, maximum),
                ("max_compute_workgroups_per_dimension", 4, 3),
                "call {which}"
            ),
            other => panic!("call {which}: expected a DeviceLimit error, got {other:?}"),
        }
    }
    assert_eq!(device.pipeline_cache_stats(), cache);
}

/// The count buffer of a gather holds exactly one element, and every buffer is on one device.
#[test]
fn gpu_a_gather_refuses_a_count_that_is_not_one_element_and_buffers_of_another_device() {
    let device = device();
    let values = GpuVec::from_slice_on(&device, &[1u32, 2, 3]).unwrap();
    let indices = GpuVec::from_slice_on(&device, &[2u32, 1, 0]).unwrap();
    for len in [0u32, 2] {
        let count = GpuVec::<u32>::zeroed_on(&device, len).unwrap();
        assert!(
            matches!(
                compact::gather_u32(&values, &indices, &count),
                Err(Error::Count(found)) if found == len
            ),
            "a count of {len} elements"
        );
    }
    let other = Arc::new(HeadlessDevice::new().expect("a Vulkan adapter is required"));
    let count = GpuVec::from_slice_on(&other, &[3u32]).unwrap();
    assert!(matches!(
        compact::gather_u32(&values, &indices, &count),
        Err(Error::Device(gust_wgpu::Error::MixedDevices { .. }))
    ));
    let foreign = GpuVec::from_slice_on(&other, &[1u32, 2, 3]).unwrap();
    let count = GpuVec::from_slice_on(&device, &[3u32]).unwrap();
    assert!(matches!(
        compact::gather_u32(&foreign, &indices, &count),
        Err(Error::Device(gust_wgpu::Error::MixedDevices { .. }))
    ));
}

/// The user-kernel gather for a struct (the pattern the documentation of `compact` points at): a one-line kernel over
/// the `indices` and `count` of a real compaction, compared with the reference.
#[test]
fn gpu_a_user_kernel_gathers_structs_by_the_compaction_indices() {
    let device = device();
    for n in [1usize, 255, 257, 1_000, 70_001] {
        let (flags, population) = population(n);
        let on_device = GpuVec::from_slice_on(&device, &flags).unwrap();
        let values = GpuVec::from_slice_on(&device, &population).unwrap();
        let (indices, count) = compact::indices(&on_device).unwrap();
        let length = u32::try_from(n).unwrap();
        let out = GpuVec::<particles::Particle>::zeroed_on(&device, length).unwrap();
        let call = particles::gather_particles::call(
            length,
            gust::Independent(&values),
            gust::Independent(&indices),
            gust::Independent(&count),
            gust::Independent(&out),
        );
        device
            .dispatch_batch(&[gust_wgpu::BufferDispatch::from(&call)])
            .unwrap();
        assert_eq!(
            out.to_vec().unwrap(),
            reference(&flags, &population),
            "n = {n}"
        );
    }
}
