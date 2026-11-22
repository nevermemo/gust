//! The algorithms on a real adapter: the first Vulkan adapter wgpu finds, through the process's default device. The
//! cases are the twin's (`common`), against the same independent references. Nothing here skips: no adapter fails.
//!
//! The tests run in parallel, on the default test threads. They used to share a mutex because one blocking readback
//! once failed with `Error::Map("mapping not complete after poll(Wait)")` while other threads polled the same device;
//! D97 (m7.rb) made blocking readbacks safe under concurrent polls, and the mutex went with m7.r3.
//!
//! Devices are shared, not opened per test (m7.av2): the default device for the tests that count nothing, and two
//! process-wide devices behind a lock each ([`Lane`]) for the heavy tests and for the tests that read the counters of
//! their device. Only the tests that lower a limit (`with_limits`) still open a device of their own.

use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use gust_wgpu::{DeviceHandle, DeviceOptions, GpuVec, HeadlessDevice, default_device};

use super::common::{
    Backend, COMPACT_LENGTHS, GPU_C, LENGTHS, Rng, SCAN_LENGTHS, Shape, assert_f32_sum,
    assert_f32_sum_by, assert_same, compact_cases, deep_compact_cases, deep_scan_cases,
    empty_cases, exact_sum, f32_input, float_min_max_cases, float_sum_cases, gather_edge_cases,
    integer_cases, out_of_range_index_cases, reference_indices, reference_scan, scan_cases,
    tie_ladder, wrapping_cases,
};
use super::struct_gather::{particles, population, reference};
use crate::{
    Error, Variant, bound, compact, evidence,
    reduce::{self, Element},
    scan,
    schedule::{BlockSchedule, Pass, ScanSchedule, Schedule, TileSchedule},
    variant::{Operation, shipped},
};

/// A lock and the devices only its holder uses: the two groups of tests that must be alone on a device (the heavy
/// ones, and those that read the counters of a device) each get one, and a device is opened once for the whole run.
/// m7.av saw the whole binary die intermittently with an access violation (3 runs in 17), never with one test per
/// process or with one thread. The cause is not known; one candidate is a device created and dropped per test while
/// other tests run, so m7.av2 gives these tests process-wide devices instead.
struct Lane {
    lock: Mutex<()>,
    /// The portable limits of `HeadlessDevice::new()` (128 MiB bindings), for the tests that pin them.
    portable: OnceLock<DeviceHandle>,
    /// The adapter's own limits, for the one test that needs buffers past the portable ones.
    adapter: OnceLock<DeviceHandle>,
}

impl Lane {
    const fn new() -> Self {
        Self {
            lock: Mutex::new(()),
            portable: OnceLock::new(),
            adapter: OnceLock::new(),
        }
    }

    /// Wait for the lane (poison-tolerant: a failed test must not fail the next one with a poisoned lock). The idle
    /// buffers of the devices are dropped on the way in and on the way out: a test starts on an empty pool, as on a
    /// device of its own, so a pool counter or the pool's idle cap cannot depend on the tests that ran before, and
    /// the hundreds of MiB a test took go back to the driver when it ends instead of staying in the pool for the
    /// rest of the run.
    fn enter(&'static self) -> Turn {
        let guard = self.lock.lock().unwrap_or_else(PoisonError::into_inner);
        let turn = Turn {
            lane: self,
            _guard: guard,
        };
        turn.clear_pools();
        turn
    }
}

/// A test's turn on a [`Lane`]; the lane is free again when it drops.
struct Turn {
    lane: &'static Lane,
    _guard: MutexGuard<'static, ()>,
}

impl Turn {
    /// The lane's portable-limits device.
    fn device(&self) -> DeviceHandle {
        self.lane
            .portable
            .get_or_init(|| {
                Arc::new(
                    HeadlessDevice::new()
                        .expect("a Vulkan adapter is required; these tests must not silently skip"),
                )
            })
            .clone()
    }

    /// The lane's device with the adapter's own limits (`DeviceOptions::default()`, as `default_device` opens).
    fn adapter_device(&self) -> DeviceHandle {
        self.lane
            .adapter
            .get_or_init(|| {
                Arc::new(
                    HeadlessDevice::with_options(DeviceOptions::default())
                        .expect("a Vulkan adapter is required; these tests must not silently skip"),
                )
            })
            .clone()
    }

    fn clear_pools(&self) {
        for device in [self.lane.portable.get(), self.lane.adapter.get()]
            .into_iter()
            .flatten()
        {
            device.buffer_pool().clear();
        }
    }
}

impl Drop for Turn {
    fn drop(&mut self) {
        self.clear_pools();
    }
}

/// The turn of the heavy tests: those with 2^24 or more elements, or a buffer at the binding limit (128 MiB), each
/// hold a set of buffers of hundreds of MiB. Every other test runs in parallel; these run one at a time, so at most one
/// heavy set is live at once and the VRAM a run adds to a GPU it shares with model services stays bounded (plan
/// m7.r3 review S1). They also read the counters of the heavy devices, which no other test touches.
fn heavy() -> Turn {
    static LANE: Lane = Lane::new();
    LANE.enter()
}

/// The turn of the light tests that read the counters of a device (pool and pipeline cache) and so must be alone on
/// it. One shared device instead of one per test; the tests run one at a time among themselves and in parallel with
/// all the others, which use `default_device()`.
fn counted() -> Turn {
    static LANE: Lane = Lane::new();
    LANE.enter()
}

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
    let turn = heavy();
    let gpu = Gpu(turn.device());
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
    let turn = heavy();
    // 2^24 + 1 elements on the shipped v1: two block passes (2049 workgroups, then one). The four-dispatch v0 form of
    // this input is `gpu_v0_runs_the_deepest_schedules` and the twin test.
    let gpu = Gpu(turn.device());
    let n = (1usize << 24) + 1;
    assert_eq!(gpu.sum(&vec![1u32; n]) as usize, n);
    let mut values = vec![5_000i32; n];
    values[n - 1] = -7;
    values[12_345_678] = 9_000;
    assert_eq!(gpu.min(&values), Some(-7));
    assert_eq!(gpu.max(&values), Some(9_000));
    // A float sum over the two v1 passes stays within the v1 bound (the v0 four-dispatch sum is in the v0 test).
    let floats = f32_input(Shape::Uniform, n, 0xF00D);
    let check = assert_f32_sum("2^24 + 1 uniform", &floats, gpu.sum(&floats), GPU_C);
    eprintln!("gpu f32 sum n = {n}: ratio {:.4}", check.ratio());
}

#[test]
fn gpu_reductions_leave_the_input_alone_and_return_their_buffers() {
    // The counted tests' shared device, not the default one: the pool counters of the default device move with every
    // parallel test.
    let turn = counted();
    let device = turn.device();
    let values: Vec<u32> = (0..70_001).map(|index| index % 100).collect();
    let input = GpuVec::from_slice_on(&device, &values).unwrap();
    let before = device.buffer_pool_stats();
    assert_eq!(
        reduce::sum(&input).unwrap() as usize,
        values.iter().map(|v| *v as usize).sum::<usize>()
    );
    assert_eq!(reduce::min(&input).unwrap(), 0);
    assert_eq!(reduce::max(&input).unwrap(), 99);
    assert_same(
        &input.to_vec().unwrap(),
        &values,
        "the input after the call",
    );
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
    let turn = heavy();
    // The portable limits: 128 MiB bindings, 2^25 four-byte elements.
    let device = turn.device();
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

/// A device of its own with the workgroup limit lowered to 3 (the limits belong to the device), for the tests of a
/// launch the host has to split (m11.algolift, D211, D208): every dispatch of more than 3 workgroups runs as pieces.
fn lowered_workgroup_device() -> Arc<HeadlessDevice> {
    let adapter = HeadlessDevice::new().expect("a Vulkan adapter is required");
    let mut limits = adapter.limits().clone();
    limits.max_compute_workgroups_per_dimension = 3;
    Arc::new(adapter.with_limits(limits).unwrap())
}

/// Equal to the host reference, or a panic naming the operation, the size and the first differing index.
fn assert_equals_reference(what: &str, got: &[u32], expected: &[u32]) {
    assert_eq!(got.len(), expected.len(), "{what}: length");
    if let Some(index) = got.iter().zip(expected).position(|(a, b)| a != b) {
        panic!(
            "{what}, size {}: first difference at index {index}: got {}, expected {}",
            expected.len(),
            got[index],
            expected[index]
        );
    }
}

/// Wrapping sum, minimum and maximum of `values` on `device`, in both variants, against the host.
fn check_reductions_equal_the_host(device: &Arc<HeadlessDevice>, n: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let values: Vec<u32> = (0..n).map(|_| rng.next_u64() as u32).collect();
    let input = GpuVec::from_slice_on(device, &values).unwrap();
    let sum = values
        .iter()
        .fold(0u32, |sum, value| sum.wrapping_add(*value));
    let (min, max) = (*values.iter().min().unwrap(), *values.iter().max().unwrap());
    for variant in [Variant::V0, Variant::V1] {
        assert_eq!(
            reduce::sum_wrapping_with(variant, &input).unwrap(),
            sum,
            "sum_wrapping {variant:?}, size {n}"
        );
        assert_eq!(
            reduce::min_with(variant, &input).unwrap(),
            min,
            "min {variant:?}, size {n}"
        );
        assert_eq!(
            reduce::max_with(variant, &input).unwrap(),
            max,
            "max {variant:?}, size {n}"
        );
    }
}

/// Inclusive and exclusive scans, in both variants, against the host reference (values 0 to 3, inside the domain).
fn check_scans_equal_the_host(device: &Arc<HeadlessDevice>, n: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let values: Vec<u32> = (0..n).map(|_| (rng.next_u64() % 4) as u32).collect();
    let input = GpuVec::from_slice_on(device, &values).unwrap();
    for variant in [Variant::V0, Variant::V1] {
        for exclusive in [false, true] {
            let out = if exclusive {
                scan::exclusive_with(variant, &input)
            } else {
                scan::inclusive_with(variant, &input)
            }
            .unwrap();
            assert_equals_reference(
                &format!("scan {variant:?} exclusive={exclusive}"),
                &out.to_vec().unwrap(),
                &reference_scan(&values, exclusive),
            );
        }
    }
}

/// `count`, `indices` and `gather_u32` on `device` against the host reference, for random flags of length `n`.
fn check_compaction_equals_the_host(device: &Arc<HeadlessDevice>, n: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let flags: Vec<u32> = (0..n).map(|_| (rng.next_u64() & 1) as u32).collect();
    let values: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(7).wrapping_add(1))
        .collect();
    let expected = reference_indices(&flags);
    let live = u32::try_from(expected.len()).unwrap();
    let gpu_flags = GpuVec::from_slice_on(device, &flags).unwrap();
    assert_eq!(
        compact::count(&gpu_flags).unwrap().to_vec().unwrap(),
        vec![live],
        "count, size {n}"
    );
    let (indices, count) = compact::indices(&gpu_flags).unwrap();
    assert_eq!(
        count.to_vec().unwrap(),
        vec![live],
        "indices count, size {n}"
    );
    let mut padded = expected.clone();
    padded.resize(n, 0);
    assert_equals_reference("indices", &indices.to_vec().unwrap(), &padded);
    let gpu_values = GpuVec::from_slice_on(device, &values).unwrap();
    let gathered = compact::gather_u32(&gpu_values, &indices, &count).unwrap();
    let mut gathered_expected: Vec<u32> = expected.iter().map(|i| values[*i as usize]).collect();
    gathered_expected.resize(n, 0);
    assert_equals_reference(
        "gather_u32",
        &gathered.to_vec().unwrap(),
        &gathered_expected,
    );
}

/// The first pass of an input launches `ceil(n / 65 536)` workgroups (v0) or `ceil(n / 8192)` (v1); a device that
/// allows fewer no longer refuses (D211): the host splits the launch, and the result equals the host reference. The
/// sizes are one pass inside the limit of 3 workgroups of v0, and one element past it.
#[test]
fn gpu_a_first_pass_past_a_lowered_workgroup_limit_is_split_and_equals_the_host_reference() {
    let device = lowered_workgroup_device();
    check_reductions_equal_the_host(&device, 3 * 65_536, 0x5A11);
    check_reductions_equal_the_host(&device, 3 * 65_536 + 1, 0x5A12);
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

/// On the shipped v1, 2^24 elements are three dispatches and 2^24 + 1 are five (a second level of totals, the add-back
/// through it); the v0 counts are five and seven (`gpu_v0_runs_the_deepest_schedules`).
#[test]
fn gpu_runs_the_deepest_scans() {
    let turn = heavy();
    deep_scan_cases(&Gpu(turn.device()));
}

/// The dispatch count of a v0 scan, measured: every dispatch of a batch takes one pipeline from the device's cache (a
/// hit or a miss), so the counters of a device of its own move by the number of dispatches. 3 up to 65 536 elements, 5
/// up to 2^24, 7 above (the level-3 path); `exclusive` costs the same as `inclusive`, and an empty input dispatches
/// nothing. v0 is pinned with `Variant::V0` because it is the fallback and the form under `GUST_PROFILE=cpu`; the v1
/// counts are in `gpu_v1_dispatch_counts_follow_the_schedules`.
#[test]
fn gpu_a_scan_is_one_batch_of_three_five_or_seven_dispatches() {
    let turn = counted();
    let device = turn.device();
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
                scan::exclusive_with(Variant::V0, &input)
            } else {
                scan::inclusive_with(Variant::V0, &input)
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
    assert!(
        scan::inclusive_with(Variant::V0, &empty)
            .unwrap()
            .is_empty()
    );
    assert!(
        scan::exclusive_with(Variant::V0, &empty)
            .unwrap()
            .is_empty()
    );
    assert_eq!(touched(&device), before, "an empty scan dispatches nothing");
}

#[test]
fn gpu_scans_leave_the_input_alone_and_return_their_buffers() {
    // The counted tests' shared device, not the default one: the pool counters of the default device move with every
    // parallel test.
    let turn = counted();
    let device = turn.device();
    let values: Vec<u32> = (0..70_001).map(|index| index % 100).collect();
    let input = GpuVec::from_slice_on(&device, &values).unwrap();
    let before = device.buffer_pool_stats();
    let inclusive = scan::inclusive(&input).unwrap();
    let exclusive = scan::exclusive(&input).unwrap();
    assert_same(
        &input.to_vec().unwrap(),
        &values,
        "the input after the call",
    );
    assert!(Arc::ptr_eq(inclusive.device(), &device));
    assert_same(
        &inclusive.to_vec().unwrap(),
        &reference_scan(&values, false),
        "inclusive scan",
    );
    assert_same(
        &exclusive.to_vec().unwrap(),
        &reference_scan(&values, true),
        "exclusive scan",
    );
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
    let turn = heavy();
    // The portable limits: 128 MiB bindings, 2^25 four-byte elements.
    let device = turn.device();
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
    // One element fewer is inside the limit and scans: ones, so the values at the top of the range (the five-dispatch
    // v1 path at 2^25 elements; seven dispatches in v0) are checked too.
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

/// Level 0 of a v0 scan launches `ceil(ceil(n / 256) / 256)` workgroups (v1: one per span of 4096 elements); a device
/// that allows fewer no longer refuses (D211): the host splits the launch, and both variants equal the host reference.
#[test]
fn gpu_a_scan_past_a_lowered_workgroup_limit_is_split_and_equals_the_host_reference() {
    let device = lowered_workgroup_device();
    check_scans_equal_the_host(&device, 3 * 256 * 256, 0x5CA1);
    check_scans_equal_the_host(&device, 3 * 256 * 256 + 1, 0x5CA2);
}

/// `inclusive_into` and `exclusive_into`, in both variants, into a buffer of the caller's that begins non-zero (so a
/// slot the scan does not write shows), against the host reference (values 0 to 3, inside the domain).
fn check_scans_into_equal_the_host(device: &Arc<HeadlessDevice>, n: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let values: Vec<u32> = (0..n).map(|_| (rng.next_u64() % 4) as u32).collect();
    let input = GpuVec::from_slice_on(device, &values).unwrap();
    for variant in [Variant::V0, Variant::V1] {
        for exclusive in [false, true] {
            let out = GpuVec::from_slice_on(device, &vec![0xDEAD_BEEF; n]).unwrap();
            if exclusive {
                scan::exclusive_into_with(variant, &input, &out)
            } else {
                scan::inclusive_into_with(variant, &input, &out)
            }
            .unwrap();
            assert_equals_reference(
                &format!("scan into {variant:?} exclusive={exclusive}"),
                &out.to_vec().unwrap(),
                &reference_scan(&values, exclusive),
            );
        }
    }
}

/// The scans into a provided buffer equal the host reference at sizes around the tile and chunk boundaries, in both
/// variants (m11.fric1b); the shipped forms (`inclusive_into`, `exclusive_into`) are checked at each size too.
#[test]
fn gpu_a_scan_into_a_callers_buffer_equals_the_host_reference() {
    let device = device();
    for (index, n) in [1usize, 255, 256, 257, 4096, 4097, 70_001, 300_000]
        .into_iter()
        .enumerate()
    {
        check_scans_into_equal_the_host(&device, n, 0x1A70 + index as u64);
        let values: Vec<u32> = (0..n).map(|i| (i % 4) as u32).collect();
        let input = GpuVec::from_slice_on(&device, &values).unwrap();
        let out = GpuVec::<u32>::zeroed_on(&device, u32::try_from(n).unwrap()).unwrap();
        scan::inclusive_into(&input, &out).unwrap();
        assert_equals_reference(
            "inclusive_into",
            &out.to_vec().unwrap(),
            &reference_scan(&values, false),
        );
        scan::exclusive_into(&input, &out).unwrap();
        assert_equals_reference(
            "exclusive_into",
            &out.to_vec().unwrap(),
            &reference_scan(&values, true),
        );
    }
    // The empty input into an empty buffer is nothing to do.
    let empty = GpuVec::<u32>::zeroed_on(&device, 0).unwrap();
    let nothing = GpuVec::<u32>::zeroed_on(&device, 0).unwrap();
    scan::inclusive_into(&empty, &nothing).unwrap();
    scan::exclusive_into(&empty, &nothing).unwrap();
    // The scan's temporaries come from the pool: a second `exclusive_into` of the same size takes no new device buffer
    // (read on the counted tests' shared device, whose pool counters no other test moves).
    let turn = counted();
    let counted_device = turn.device();
    let values: Vec<u32> = (0..70_001).map(|index| index % 4).collect();
    let input = GpuVec::from_slice_on(&counted_device, &values).unwrap();
    let out = GpuVec::<u32>::zeroed_on(&counted_device, 70_001).unwrap();
    scan::exclusive_into(&input, &out).unwrap();
    let misses = counted_device.buffer_pool_stats().misses;
    scan::exclusive_into(&input, &out).unwrap();
    assert_eq!(
        counted_device.buffer_pool_stats().misses,
        misses,
        "the second exclusive_into allocated"
    );
    assert_equals_reference(
        "exclusive_into again",
        &out.to_vec().unwrap(),
        &reference_scan(&values, true),
    );
}

/// A level 0 past a lowered workgroup limit is split by the host (D211) for the `_into` forms too: one pass inside the
/// limit of 3 workgroups of v0, and one element past it.
#[test]
fn gpu_a_scan_into_past_a_lowered_workgroup_limit_is_split_and_equals_the_host_reference() {
    let device = lowered_workgroup_device();
    check_scans_into_equal_the_host(&device, 3 * 256 * 256, 0x5CB1);
    check_scans_into_equal_the_host(&device, 3 * 256 * 256 + 1, 0x5CB2);
}

/// An output of another length than the input is an `Error::Length`, before anything is dispatched (no batch in the
/// timing scope), and the output is left as it was.
#[test]
fn gpu_a_scan_into_a_buffer_of_the_wrong_length_is_refused() {
    let device = device();
    let input = GpuVec::from_slice_on(&device, &[1u32, 2, 3, 4]).unwrap();
    for (len, name) in [(3usize, "shorter"), (5, "longer")] {
        let out = GpuVec::from_slice_on(&device, &vec![7u32; len]).unwrap();
        for variant in [Variant::V0, Variant::V1] {
            for exclusive in [false, true] {
                let timed = crate::timed(|| {
                    Ok(if exclusive {
                        scan::exclusive_into_with(variant, &input, &out)
                    } else {
                        scan::inclusive_into_with(variant, &input, &out)
                    })
                })
                .unwrap();
                assert_eq!(
                    timed.time.batches, 0,
                    "{name} {variant:?} exclusive={exclusive}"
                );
                match timed.value {
                    Err(Error::Length { expected, actual }) => {
                        assert_eq!((expected, actual), (4, len as u32), "{name}");
                    }
                    other => panic!("{name} {variant:?} exclusive={exclusive}: {other:?}"),
                }
            }
        }
        assert_eq!(out.to_vec().unwrap(), vec![7u32; len], "{name}: untouched");
    }
    // An empty input into a non-empty buffer is refused the same way.
    let empty = GpuVec::<u32>::zeroed_on(&device, 0).unwrap();
    let one = GpuVec::<u32>::zeroed_on(&device, 1).unwrap();
    assert!(matches!(
        scan::inclusive_into(&empty, &one),
        Err(Error::Length {
            expected: 0,
            actual: 1
        })
    ));
}

/// The same buffer as input and output is refused as `Error::Device(AliasedBinding)` (it would be bound to a read and a
/// write parameter), in both variants and both scans, and the buffer is left as it was.
#[test]
fn gpu_a_scan_into_the_input_buffer_itself_is_refused() {
    let device = device();
    let values: Vec<u32> = (0..5000).map(|i| (i % 4) as u32 + 1).collect();
    let buffer = GpuVec::from_slice_on(&device, &values).unwrap();
    for variant in [Variant::V0, Variant::V1] {
        for exclusive in [false, true] {
            let result = if exclusive {
                scan::exclusive_into_with(variant, &buffer, &buffer)
            } else {
                scan::inclusive_into_with(variant, &buffer, &buffer)
            };
            assert!(
                matches!(
                    result,
                    Err(Error::Device(gust_wgpu::Error::AliasedBinding { .. }))
                ),
                "{variant:?} exclusive={exclusive}: {result:?}"
            );
            assert_eq!(buffer.to_vec().unwrap(), values, "{variant:?}: unchanged");
        }
    }
}

/// One output buffer reused for different inputs, sizes spanning one and several dispatches, and for both scans in turn:
/// every call gives its own right answer, with nothing of the call before it left in the buffer.
#[test]
fn gpu_a_reused_scan_output_holds_no_stale_data() {
    let device = device();
    for n in [100usize, 70_001] {
        let first: Vec<u32> = (0..n).map(|i| (i % 7) as u32 + 1).collect();
        let second: Vec<u32> = (0..n).map(|i| u32::from(i % 5 == 1)).collect();
        let out = GpuVec::<u32>::zeroed_on(&device, u32::try_from(n).unwrap()).unwrap();
        for variant in [Variant::V0, Variant::V1] {
            for (values, exclusive) in [
                (&first, false),
                (&second, false),
                (&first, true),
                (&second, true),
                (&second, false),
            ] {
                let input = GpuVec::from_slice_on(&device, values).unwrap();
                if exclusive {
                    scan::exclusive_into_with(variant, &input, &out)
                } else {
                    scan::inclusive_into_with(variant, &input, &out)
                }
                .unwrap();
                assert_equals_reference(
                    &format!("reused output {variant:?} exclusive={exclusive}"),
                    &out.to_vec().unwrap(),
                    &reference_scan(values, exclusive),
                );
            }
        }
    }
}

#[test]
fn gpu_compaction_is_exact_at_every_boundary() {
    compact_cases(&Gpu(device()), &COMPACT_LENGTHS);
}

/// `2^24 + 1` flags: 257 workgroups of the block-strided mask and scatter launches (an invocation per element would be
/// 65 537 workgroups, past the 65 535 ceiling), and a scan of five dispatches (v1; v0 takes seven).
#[test]
fn gpu_runs_the_deepest_compaction() {
    let turn = heavy();
    deep_compact_cases(&Gpu(turn.device()));
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
    let turn = counted();
    let device = turn.device();
    let touched = |device: &HeadlessDevice| {
        let stats = device.pipeline_cache_stats();
        stats.hits + stats.misses
    };
    for n in [1usize, 256, 257, 65_537] {
        // The schedules of the variants the public functions ship (`variant::shipped`).
        let reduction = match shipped(Operation::Sum) {
            Variant::V0 => Schedule::new(n as u32).unwrap().dispatches(),
            Variant::V1 => BlockSchedule::new(n as u32).unwrap().dispatches(),
        } as u64;
        let scan = match shipped(Operation::Exclusive) {
            Variant::V0 => ScanSchedule::new(n as u32).unwrap().dispatches(),
            Variant::V1 => TileSchedule::new(n as u32).unwrap().dispatches(),
        } as u64;
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
    // The counted tests' shared device, not the default one: the pool counters of the default device move with every
    // parallel test.
    let turn = counted();
    let device = turn.device();
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
        assert_same(
            &gathered.to_vec().unwrap()[..live as usize],
            &expected,
            "gather",
        );
    };
    round();
    assert_same(
        &on_device.to_vec().unwrap(),
        &flags,
        "the flags after the call",
    );
    assert_same(
        &source.to_vec().unwrap(),
        &values,
        "the values after the call",
    );
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
    let turn = heavy();
    // The portable limits: 128 MiB bindings, 2^25 four-byte elements.
    let device = turn.device();
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
    // The first index, read through a one-element gather on the device (the 128 MiB indices stay there).
    let first = GpuVec::from_slice_on(&device, &[0u32]).unwrap();
    let slot = compact::gather_u32(&indices, &first, &count).unwrap();
    assert_eq!(slot.to_vec().unwrap(), vec![len - 2]);
}

/// The element-wise launches are `ceil(n / 65 536)` workgroups; a device that allows fewer no longer refuses (D211):
/// the host splits the launches of the flags of `count` and `indices` and of the indices of a gather, and the results
/// equal the host reference.
#[test]
fn gpu_a_compaction_past_a_lowered_workgroup_limit_is_split_and_equals_the_host_reference() {
    let device = lowered_workgroup_device();
    check_compaction_equals_the_host(&device, 3 * 65_536, 0xC0A1);
    check_compaction_equals_the_host(&device, 3 * 65_536 + 1, 0xC0A2);
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
    // Another device than the default one: the counted tests' shared device, held for the rest of the test so that no
    // counting test touches it meanwhile.
    let turn = counted();
    let other = turn.device();
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
            Pass::over(length).partials(),
            gust::Independent(&values),
            gust::Independent(&indices),
            gust::Independent(&count),
            gust::Independent(&out),
        );
        device
            .dispatch_batch(&[gust_wgpu::BufferDispatch::from(&call)])
            .unwrap();
        assert_same(
            &out.to_vec().unwrap(),
            &reference(&flags, &population),
            &format!("n = {n}"),
        );
    }
}

// ---- m7.r4: the v1 kernels (shared memory and barriers) against the same references ----

/// Lengths around the boundaries of the v1 block reduce (a workgroup owns 8192 elements; 256 lanes add up to 32 terms
/// each) and of the v1 scan (a workgroup owns 16 tiles of 256 elements, a span of 4096), besides the v0 ones.
const V1_LENGTHS: [usize; 16] = [
    1, 2, 255, 256, 257, 4_095, 4_096, 4_097, 8_191, 8_192, 8_193, 16_385, 65_537, 131_073,
    999_983, 1_000_003,
];

/// [`SCAN_LENGTHS`] and the span boundaries of the v1 scan.
const V1_SCAN_LENGTHS: [usize; 25] = [
    0, 1, 2, 3, 255, 256, 257, 511, 512, 513, 4_095, 4_096, 4_097, 8_191, 8_192, 8_193, 16_385,
    65_535, 65_536, 65_537, 65_793, 131_073, 999_983, 1_000_003, 1_048_576,
];

/// The reductions and scans of one variant, whatever the public functions ship: the case tables run on this to hold
/// v0 and v1 to the same references. Compaction is the shipped path.
struct Pinned(DeviceHandle, Variant);

impl Pinned {
    fn v1() -> Self {
        Self(device(), Variant::V1)
    }
}

impl Backend for Pinned {
    const C: f64 = GPU_C;

    fn sum_bound(&self, n: u32, c: f64, s: f64) -> f64 {
        match self.1 {
            Variant::V0 => bound::f32_sum_bound(n, c, s),
            Variant::V1 => bound::f32_sum_bound_v1(n, c, s),
        }
    }

    fn sum<T: Element>(&self, values: &[T]) -> T {
        let input = GpuVec::from_slice_on(&self.0, values).unwrap();
        reduce::sum_with(self.1, &input).unwrap()
    }

    fn min<T: Element>(&self, values: &[T]) -> Option<T> {
        let input = GpuVec::from_slice_on(&self.0, values).unwrap();
        match reduce::min_with(self.1, &input) {
            Ok(value) => Some(value),
            Err(Error::Empty) => None,
            Err(error) => panic!("{error}"),
        }
    }

    fn max<T: Element>(&self, values: &[T]) -> Option<T> {
        let input = GpuVec::from_slice_on(&self.0, values).unwrap();
        match reduce::max_with(self.1, &input) {
            Ok(value) => Some(value),
            Err(Error::Empty) => None,
            Err(error) => panic!("{error}"),
        }
    }

    fn sum_wrapping(&self, values: &[u32]) -> u32 {
        let input = GpuVec::from_slice_on(&self.0, values).unwrap();
        reduce::sum_wrapping_with(self.1, &input).unwrap()
    }

    fn inclusive(&self, values: &[u32]) -> Vec<u32> {
        let input = GpuVec::from_slice_on(&self.0, values).unwrap();
        let out = scan::inclusive_with(self.1, &input).unwrap();
        assert_eq!(out.len() as usize, values.len());
        out.to_vec().unwrap()
    }

    fn exclusive(&self, values: &[u32]) -> Vec<u32> {
        let input = GpuVec::from_slice_on(&self.0, values).unwrap();
        let out = scan::exclusive_with(self.1, &input).unwrap();
        assert_eq!(out.len() as usize, values.len());
        out.to_vec().unwrap()
    }

    fn compact_count(&self, flags: &[u32]) -> u32 {
        Gpu(self.0.clone()).compact_count(flags)
    }

    fn compact_indices(&self, flags: &[u32]) -> (Vec<u32>, u32) {
        Gpu(self.0.clone()).compact_indices(flags)
    }

    fn gather_u32(&self, values: &[u32], indices: &[u32], count: u32) -> Vec<u32> {
        Gpu(self.0.clone()).gather_u32(values, indices, count)
    }

    fn gather_i32(&self, values: &[i32], indices: &[u32], count: u32) -> Vec<i32> {
        Gpu(self.0.clone()).gather_i32(values, indices, count)
    }

    fn gather_f32(&self, values: &[f32], indices: &[u32], count: u32) -> Vec<f32> {
        Gpu(self.0.clone()).gather_f32(values, indices, count)
    }
}

#[test]
fn gpu_v1_integer_sums_min_and_max_are_exact_at_every_boundary() {
    integer_cases(&Pinned::v1(), &V1_LENGTHS);
}

#[test]
fn gpu_v1_float_min_and_max_are_exact() {
    float_min_max_cases(&Pinned::v1(), &V1_LENGTHS);
}

#[test]
fn gpu_v1_empty_inputs() {
    empty_cases(&Pinned::v1());
}

#[test]
fn gpu_v1_float_sums_are_within_the_per_level_bound() {
    let gpu = Pinned::v1();
    for n in [1, 255, 257, 8_193, 65_537, 1_000_000] {
        for (shape, check) in float_sum_cases(&gpu, n) {
            eprintln!(
                "gpu v1 f32 sum n = {n} {shape:?}: distance {:e}, bound {:e}, ratio {:.4}",
                check.distance,
                check.bound,
                check.ratio()
            );
        }
    }
}

/// The tie ladder on the adapter against the (tighter) bound of the v1 schedule.
#[test]
fn gpu_v1_float_sum_of_the_tie_ladder_is_within_the_bound() {
    let turn = heavy();
    let gpu = Pinned(turn.device(), Variant::V1);
    let values = tie_ladder(1 << 24);
    let actual = gpu.sum(&values);
    let check = assert_f32_sum_by(
        "v1 tie ladder",
        &values,
        actual,
        GPU_C,
        bound::f32_sum_bound_v1,
    );
    eprintln!(
        "gpu v1 tie ladder: result {actual}, distance {:e}, bound {:e}, ratio {:.4}",
        check.distance,
        check.bound,
        check.ratio()
    );
}

#[test]
fn gpu_v1_wrapping_sums_are_defined_modulo_2_32_for_every_input() {
    wrapping_cases(&Pinned::v1(), &V1_LENGTHS);
}

#[test]
fn gpu_v1_scans_are_exact_at_every_boundary() {
    scan_cases(&Pinned::v1(), &V1_SCAN_LENGTHS);
}

/// The deepest v1 schedules: a reduction of 2^26 + 1 elements takes three block passes (the second level holds 8193
/// values, one more than a workgroup owns), and the scans of 2^24 (one level of totals, 4096 of them) and 2^24 + 1 (a
/// second level, the add-back through it).
///
/// The test opens the adapter's own limits, and what runs depends on the adapter's `max_storage_buffer_binding_size`:
/// - 2^28 + 4 bytes or more (2^26 + 1 four-byte elements fit; the RTX 5090 reports 2 GiB): everything above.
/// - Less (lavapipe reports 128 MiB): the three-pass reduction cannot bind. The test prints one line saying so, asserts
///   that the pinned v1 sum of the smallest input past the limit (allocated on the device only, instead of 256 MiB of
///   2^26 + 1 elements) is a `DeviceLimit` naming `max_storage_buffer_binding_size`, the documented error before any
///   dispatch, then runs the deepest reduction that does fit (2^25 elements at 128 MiB: two block passes) and, when
///   2^24 + 1 elements fit, the deep scans.
#[test]
fn gpu_v1_runs_the_deepest_schedules() {
    let turn = heavy();
    // The adapter's own limits: 2^26 + 1 four-byte elements are past the 256 MiB buffers of the portable ones.
    let device = turn.adapter_device();
    let limit = device.limits().max_storage_buffer_binding_size;
    let deepest = (1usize << 26) + 1;
    let mut n = deepest;
    let mut passes = 3;
    if limit < 4 * deepest as u64 {
        n = usize::try_from(limit / 4).unwrap().min(1 << 26);
        passes = 2;
        eprintln!(
            "gpu v1 deepest schedules: this adapter's max_storage_buffer_binding_size is {limit} bytes, so the \
             three-pass reduction (2^26 + 1 elements, {} bytes) was NOT run here; running {n} elements in {passes} \
             passes instead",
            4 * deepest
        );
        // The smallest input past the limit, device memory only: the pinned v1 sum is refused before any dispatch.
        let past = u32::try_from(limit / 4).unwrap() + 1;
        let input = GpuVec::<u32>::zeroed_on(&device, past).unwrap();
        match reduce::sum_with(Variant::V1, &input) {
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
        drop(input);
    }
    let gpu = Pinned(device, Variant::V1);
    assert_eq!(BlockSchedule::new(n as u32).unwrap().dispatches(), passes);
    assert_eq!(gpu.sum(&vec![1u32; n]) as usize, n);
    let mut values = vec![5_000i32; n];
    values[n - 1] = -7;
    values[12_345_678.min(n - 2)] = 9_000;
    assert_eq!(gpu.min(&values), Some(-7));
    assert_eq!(gpu.max(&values), Some(9_000));
    drop(values);
    let floats = f32_input(Shape::Uniform, n, 0xF00D);
    let check = assert_f32_sum_by(
        "v1 deepest uniform",
        &floats,
        gpu.sum(&floats),
        GPU_C,
        bound::f32_sum_bound_v1,
    );
    eprintln!("gpu v1 f32 sum n = {n}: ratio {:.4}", check.ratio());
    drop(floats);
    if limit >= 4 * ((1u64 << 24) + 1) {
        deep_scan_cases(&gpu);
    } else {
        eprintln!(
            "gpu v1 deepest schedules: the 2^24 + 1 scans (64 MiB) do not fit {limit} bytes and were NOT run here"
        );
    }
}

/// The dispatch counts of the v1 schedules, measured as in the v0 test: one dispatch while the input fits a workgroup
/// (a block reduce) or a span (a scan), then one block pass per level for a reduction and `2k + 1` for a scan.
#[test]
fn gpu_v1_dispatch_counts_follow_the_schedules() {
    let turn = heavy();
    let device = turn.device();
    let touched = |device: &HeadlessDevice| {
        let stats = device.pipeline_cache_stats();
        stats.hits + stats.misses
    };
    for (n, dispatches) in [(1usize, 1usize), (8_192, 1), (8_193, 2), (1 << 20, 2)] {
        assert_eq!(
            BlockSchedule::new(n as u32).unwrap().dispatches(),
            dispatches
        );
        let input = GpuVec::<u32>::from_slice_on(&device, &vec![1u32; n]).unwrap();
        let before = touched(&device);
        assert_eq!(reduce::sum_with(Variant::V1, &input).unwrap() as usize, n);
        assert_eq!(touched(&device) - before, dispatches as u64, "sum, n = {n}");
        let before = touched(&device);
        assert_eq!(reduce::max_with(Variant::V1, &input).unwrap(), 1);
        assert_eq!(touched(&device) - before, dispatches as u64, "max, n = {n}");
        let before = touched(&device);
        assert_eq!(
            reduce::sum_wrapping_with(Variant::V1, &input).unwrap() as usize,
            n
        );
        assert_eq!(
            touched(&device) - before,
            dispatches as u64,
            "sum_wrapping, n = {n}"
        );
    }
    for (n, dispatches) in [
        (1usize, 1usize),
        (4_096, 1),
        (4_097, 3),
        (1 << 24, 3),
        ((1 << 24) + 1, 5),
    ] {
        assert_eq!(
            TileSchedule::new(n as u32).unwrap().dispatches(),
            dispatches
        );
        let input = GpuVec::<u32>::from_slice_on(&device, &vec![1u32; n]).unwrap();
        for exclusive in [false, true] {
            let before = touched(&device);
            let out = if exclusive {
                scan::exclusive_with(Variant::V1, &input)
            } else {
                scan::inclusive_with(Variant::V1, &input)
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
    assert!(
        scan::inclusive_with(Variant::V1, &empty)
            .unwrap()
            .is_empty()
    );
    assert_eq!(reduce::sum_with(Variant::V1, &empty).unwrap(), 0);
    assert_eq!(
        touched(&device),
        before,
        "an empty input dispatches nothing"
    );
}

#[test]
fn gpu_variants_leave_the_input_alone_and_return_their_buffers() {
    // The counted tests' shared device, not the default one: the pool counters of the process-wide default device move
    // with every parallel test. Each variant starts on an empty pool, as on a device of its own.
    let turn = counted();
    let device = turn.device();
    for variant in [Variant::V0, Variant::V1] {
        turn.clear_pools();
        let values: Vec<u32> = (0..70_001).map(|index| index % 100).collect();
        let total = values.iter().map(|v| *v as usize).sum::<usize>();
        let input = GpuVec::from_slice_on(&device, &values).unwrap();
        let before = device.buffer_pool_stats();
        let round = || {
            assert_eq!(reduce::sum_with(variant, &input).unwrap() as usize, total);
            assert_eq!(reduce::min_with(variant, &input).unwrap(), 0);
            assert_eq!(reduce::max_with(variant, &input).unwrap(), 99);
            let inclusive = scan::inclusive_with(variant, &input).unwrap();
            let exclusive = scan::exclusive_with(variant, &input).unwrap();
            assert_same(
                &inclusive.to_vec().unwrap(),
                &reference_scan(&values, false),
                &format!("{variant:?} inclusive"),
            );
            assert_same(
                &exclusive.to_vec().unwrap(),
                &reference_scan(&values, true),
                &format!("{variant:?} exclusive"),
            );
        };
        round();
        assert_same(
            &input.to_vec().unwrap(),
            &values,
            "the input after the calls",
        );
        let after = device.buffer_pool_stats();
        assert_eq!(
            after.outstanding, before.outstanding,
            "{variant:?}: the partials, the totals and the results went back to the pool"
        );
        // A second round is served from the pool: no new allocation.
        round();
        assert_eq!(
            device.buffer_pool_stats().misses,
            after.misses,
            "{variant:?}: the second round allocated"
        );
    }
}

/// A v1 request runs v1 whatever the workgroup limit (D211: the host splits a launch past it), and runs v0 instead only
/// where the device has less workgroup memory than the v1 kernels declare (3072 bytes). With 3 workgroups allowed, a
/// reduction of 100 000 elements takes 13 v1 workgroups and a scan of the same 25 v1 spans; both run, split, and equal
/// the host reference. The storage half of the pre-check still refuses (the binding-limit tests above).
#[test]
fn gpu_v1_runs_past_a_lowered_workgroup_limit_and_falls_back_to_v0_only_for_workgroup_memory() {
    let device = lowered_workgroup_device();
    let touched = |device: &HeadlessDevice| {
        let stats = device.pipeline_cache_stats();
        stats.hits + stats.misses
    };
    // The pipeline cache is asked once per dispatch (a split dispatch asks once for its variant), so hits + misses
    // count dispatches, which tells v1 from v0: with 3 workgroups allowed, 100 000 elements are 13 v1 workgroups (reduce: the
    // block pass and the final block, 2 dispatches; v0 takes 3) and 25 v1 spans (scan: 3 dispatches; v0 takes 5).
    let n = 100_000usize;
    let input = GpuVec::from_slice_on(&device, &vec![1u32; n]).unwrap();
    let before = touched(&device);
    assert_eq!(reduce::sum_with(Variant::V1, &input).unwrap() as usize, n);
    assert_eq!(touched(&device) - before, 2, "v1 reduce of {n}");
    let before = touched(&device);
    let inclusive = scan::inclusive_with(Variant::V1, &input).unwrap();
    assert_eq!(inclusive.to_vec().unwrap()[n - 1] as usize, n);
    assert_eq!(touched(&device) - before, 3, "v1 scan of {n}");
    // Inside the limit (3 workgroups of 4096 elements) the v1 kernels run whole, in the dispatches of their schedule.
    let fits = GpuVec::from_slice_on(&device, &vec![1u32; 3 * 4_096]).unwrap();
    let before = touched(&device);
    assert_eq!(reduce::sum_with(Variant::V1, &fits).unwrap(), 3 * 4_096);
    assert_eq!(touched(&device) - before, 2);
    let before = touched(&device);
    let _ = scan::inclusive_with(Variant::V1, &fits).unwrap();
    assert_eq!(touched(&device) - before, 3);
    // Past what v0 can launch whole as well: split, not refused, and still the v1 dispatches (49 spans, 25 blocks).
    let outside = vec![1u32; 3 * 65_536 + 1];
    let outside = GpuVec::from_slice_on(&device, &outside).unwrap();
    let before = touched(&device);
    assert_eq!(
        reduce::sum_with(Variant::V1, &outside).unwrap() as usize,
        3 * 65_536 + 1
    );
    assert_eq!(touched(&device) - before, 2);
    let before = touched(&device);
    let exclusive = scan::exclusive_with(Variant::V1, &outside).unwrap();
    assert_eq!(
        exclusive.to_vec().unwrap()[3 * 65_536],
        u32::try_from(3 * 65_536).unwrap()
    );
    assert_eq!(touched(&device) - before, 3);
    // A device with less workgroup memory than the v1 kernels declare (3072 bytes) runs v0 for any size, however small.
    let adapter = HeadlessDevice::new().expect("a Vulkan adapter is required");
    let mut limits = adapter.limits().clone();
    limits.max_compute_workgroup_storage_size = 2_048;
    let small = Arc::new(adapter.with_limits(limits).unwrap());
    let input = GpuVec::from_slice_on(&small, &vec![1u32; 5_000]).unwrap();
    let before = touched(&small);
    assert_eq!(reduce::sum_with(Variant::V1, &input).unwrap(), 5_000);
    // v0: one pass and the terminal kernel (v1 would be one dispatch).
    assert_eq!(touched(&small) - before, 2);
    let before = touched(&small);
    let inclusive = scan::inclusive_with(Variant::V1, &input).unwrap();
    assert_eq!(inclusive.to_vec().unwrap()[4_999], 5_000);
    // v0: 20 chunks and the top scan, so three dispatches (v1 would be one).
    assert_eq!(touched(&small) - before, 3);
}

// ---- m11.algolift: past 2^24 elements, on the adapter's own limits (nothing lowered). Each is equal to the host
// reference. These are size tests, not split tests: at 2^24 + 1000 elements the widest launch is about 4097
// workgroups (v1 scan level 0, spans of 4096; v1 reduce 2049 blocks of 8192; v0 and compaction 257 blocks of 65 536),
// so nothing is split at any workgroup limit of 65 535 or more. The lowered-limit tests above are the split tests. ----

/// `2^24 + 1000` elements.
const BIG: usize = (1 << 24) + 1_000;

#[test]
fn gpu_big_2_pow_24_plus_1000_sum_wrapping_equals_the_host_reference() {
    let turn = heavy();
    let device = turn.adapter_device();
    let mut rng = Rng::new(0x0B16_5A5A);
    let values: Vec<u32> = (0..BIG).map(|_| rng.next_u64() as u32).collect();
    let expected = values
        .iter()
        .fold(0u32, |sum, value| sum.wrapping_add(*value));
    let input = GpuVec::from_slice_on(&device, &values).unwrap();
    assert_eq!(
        reduce::sum_wrapping(&input).unwrap(),
        expected,
        "sum_wrapping, size {BIG}"
    );
    assert_eq!(
        reduce::sum_wrapping_with(Variant::V0, &input).unwrap(),
        expected,
        "sum_wrapping v0, size {BIG}"
    );
}

#[test]
fn gpu_big_2_pow_24_plus_1000_scans_equal_the_host_reference() {
    let turn = heavy();
    let device = turn.adapter_device();
    let mut rng = Rng::new(0x0B16_5C5C);
    let values: Vec<u32> = (0..BIG).map(|_| (rng.next_u64() % 4) as u32).collect();
    let input = GpuVec::from_slice_on(&device, &values).unwrap();
    for variant in [Variant::V1, Variant::V0] {
        for exclusive in [false, true] {
            let out = if exclusive {
                scan::exclusive_with(variant, &input)
            } else {
                scan::inclusive_with(variant, &input)
            }
            .unwrap();
            assert_equals_reference(
                &format!("scan {variant:?} exclusive={exclusive}"),
                &out.to_vec().unwrap(),
                &reference_scan(&values, exclusive),
            );
        }
    }
}

#[test]
fn gpu_big_2_pow_24_plus_1000_compaction_equals_the_host_reference() {
    let turn = heavy();
    let device = turn.adapter_device();
    check_compaction_equals_the_host(&device, BIG, 0x0B16_C0C0);
}

// ---- m7.r4: v0 stays pinned. The public functions ship v1 (`variant::shipped`), and v0 is the fallback for an input
// the v1 launch cannot take and the only form without barriers, so its tables run on `Pinned(.., Variant::V0)` ----

#[test]
fn gpu_v0_integer_sums_min_and_max_are_exact_at_every_boundary() {
    integer_cases(&Pinned(device(), Variant::V0), &LENGTHS);
}

#[test]
fn gpu_v0_float_min_and_max_are_exact() {
    float_min_max_cases(&Pinned(device(), Variant::V0), &LENGTHS);
}

#[test]
fn gpu_v0_float_sums_are_within_the_per_level_bound() {
    let gpu = Pinned(device(), Variant::V0);
    for n in [1, 255, 257, 65_537, 1_000_000] {
        float_sum_cases(&gpu, n);
    }
}

#[test]
fn gpu_v0_wrapping_sums_are_defined_modulo_2_32_for_every_input() {
    wrapping_cases(&Pinned(device(), Variant::V0), &LENGTHS);
}

#[test]
fn gpu_v0_scans_are_exact_at_every_boundary() {
    scan_cases(&Pinned(device(), Variant::V0), &SCAN_LENGTHS);
}

/// The v0 schedules at their deepest: four dispatches for a reduction of 2^24 + 1 elements (the tie ladder of 2^24 too,
/// against the v0 bound) and seven for the scan of 2^24 + 1.
#[test]
fn gpu_v0_runs_the_deepest_schedules() {
    let turn = heavy();
    let gpu = Pinned(turn.device(), Variant::V0);
    let n = (1usize << 24) + 1;
    assert_eq!(Schedule::new(n as u32).unwrap().dispatches(), 4);
    assert_eq!(gpu.sum(&vec![1u32; n]) as usize, n);
    let mut values = vec![5_000i32; n];
    values[n - 1] = -7;
    values[12_345_678] = 9_000;
    assert_eq!(gpu.min(&values), Some(-7));
    assert_eq!(gpu.max(&values), Some(9_000));
    drop(values);
    let floats = f32_input(Shape::Uniform, n, 0xF00D);
    assert_f32_sum("v0 2^24 + 1 uniform", &floats, gpu.sum(&floats), GPU_C);
    drop(floats);
    let ladder = tie_ladder(1 << 24);
    assert_f32_sum("v0 tie ladder", &ladder, gpu.sum(&ladder), GPU_C);
    drop(ladder);
    deep_scan_cases(&gpu);
}

/// The lengths of the evidence sum: around the lane chain (32 terms of 256 lanes), the 8192-element block, a length that
/// is no multiple of anything and two passes' worth of workgroups (1e6 is 123 workgroups, 1e7 is 1221).
const EVIDENCE_LENGTHS: [usize; 15] = [
    1, 2, 31, 32, 33, 255, 256, 257, 8_191, 8_192, 8_193, 65_537, 1_000_003, 2_000_029, 9_999_991,
];

/// Slice m8.e (D146, Opus 14): the per-block step of the `u32` sum as the pair of bodies of
/// [`evidence::block_sum_kernels`], whichever body the build selects (the portable tree under `dev`, the subgroup one
/// under `dev+subgroups`, which is where the subgroup body is judged), against the independent `i128` reference and
/// against the shipped v1 on the same input, in one process: the cross-variant test. The profile decides which body
/// ran, and the test says so by asserting the descriptor and the device agree with `gust::capability::enabled()`.
#[test]
fn gpu_evidence_block_sum_equals_the_reference_and_v1_whichever_body_the_build_selects() {
    let device = device();
    let enabled = gust::capability::enabled();
    let selected = &evidence::block_sum_kernels::block_sum::DESCRIPTOR;
    assert_eq!(selected.requires, enabled, "the selected body's requires");
    assert_eq!(device.capabilities(), enabled, "the device's capabilities");
    eprintln!(
        "gust-algo evidence body: requires {:?}, subgroup sizes {:?}",
        selected.requires,
        device.subgroup_sizes()
    );
    let empty = GpuVec::<u32>::from_slice_on(&device, &[]).unwrap();
    assert_eq!(evidence::sum_u32(&empty).unwrap(), 0);
    for n in EVIDENCE_LENGTHS {
        let mut rng = Rng::new(0xE71D + n as u64);
        // No partial of any grouping overflows: every element is below 2^32 / n.
        let cap = (u64::from(u32::MAX) / n as u64).max(1);
        let random: Vec<u32> = (0..n).map(|_| rng.below(cap) as u32).collect();
        let mut last = vec![0_u32; n];
        last[n - 1] = u32::MAX;
        let mut middle = vec![0_u32; n];
        middle[n / 2] = u32::MAX;
        for (what, values) in [
            ("random", random),
            ("ones", vec![1_u32; n]),
            ("zeros", vec![0_u32; n]),
            ("only the last element", last),
            ("only the middle element", middle),
        ] {
            let input = GpuVec::from_slice_on(&device, &values).unwrap();
            let want = exact_sum(&values);
            assert_eq!(
                i128::from(evidence::sum_u32(&input).unwrap()),
                want,
                "evidence sum, {what}, n = {n}"
            );
            assert_eq!(
                i128::from(reduce::sum_with(Variant::V1, &input).unwrap()),
                want,
                "v1 sum, {what}, n = {n}"
            );
        }
    }
}

/// The GPU-time rule of `timed` on this adapter: a time exactly when it can report one.
fn assert_time_rule(device: &HeadlessDevice, time: crate::GpuTime, batches: usize) {
    assert_eq!(time.batches, batches);
    assert_eq!(
        time.gpu_elapsed.is_some(),
        device.supports_gpu_timestamps() && batches > 0,
        "gpu_elapsed must be Some exactly when the adapter reports timestamps and a batch ran"
    );
}

/// D194: a timed call returns what the untimed call returns, and reports one batch and a time per the adapter.
#[test]
fn gpu_timed_calls_return_the_untimed_values_and_report_one_batch() {
    let device = device();
    let words: Vec<u32> = (0..20_000u32)
        .map(|i| i.wrapping_mul(2_654_435_761) % 1000)
        .collect();
    let floats: Vec<f32> = (0..9_000u32).map(|i| (i % 64) as f32 - 31.0).collect();
    let ints = GpuVec::from_slice_on(&device, &words).unwrap();
    let reals = GpuVec::from_slice_on(&device, &floats).unwrap();

    let t = crate::timed(|| reduce::sum(&ints)).unwrap();
    assert_eq!(t.value, reduce::sum(&ints).unwrap());
    assert_time_rule(&device, t.time, 1);
    let t = crate::timed(|| reduce::min(&ints)).unwrap();
    assert_eq!(t.value, reduce::min(&ints).unwrap());
    assert_time_rule(&device, t.time, 1);
    let t = crate::timed(|| reduce::max(&ints)).unwrap();
    assert_eq!(t.value, reduce::max(&ints).unwrap());
    assert_time_rule(&device, t.time, 1);
    let t = crate::timed(|| reduce::sum_wrapping(&ints)).unwrap();
    assert_eq!(t.value, reduce::sum_wrapping(&ints).unwrap());
    assert_time_rule(&device, t.time, 1);
    // The inputs are small integers and every partial sum is below 2^24, so the f32 sum is exact in any grouping:
    // the two runs agree bit for bit whatever order the device adds in.
    let t = crate::timed(|| reduce::sum(&reals)).unwrap();
    assert_eq!(t.value.to_bits(), reduce::sum(&reals).unwrap().to_bits());
    assert_time_rule(&device, t.time, 1);
    // The operation the benchmark's W7 times: the f32 sum that stays on the device.
    let t = crate::timed(|| reduce::sum_on_device(&reals)).unwrap();
    let untimed = reduce::sum_on_device(&reals).unwrap();
    assert_eq!(
        t.value.to_vec().unwrap()[0].to_bits(),
        untimed.to_vec().unwrap()[0].to_bits()
    );
    assert_time_rule(&device, t.time, 1);
    // The evidence block sum is wrapped too. On an adapter or build that refuses it the refusal is the same timed and
    // untimed, and the output says so.
    match (
        evidence::sum_u32(&ints),
        crate::timed(|| evidence::sum_u32(&ints)),
    ) {
        (Ok(plain), Ok(t)) => {
            assert_eq!(t.value, plain);
            assert_time_rule(&device, t.time, 1);
        }
        (Err(plain), Err(timed)) => {
            eprintln!("evidence::sum_u32 NOT timed on this adapter or build: {plain}");
            // Only a build that selected the subgroup body may be refused (a missing capability or a subgroup size
            // below the minimum); the portable body runs on any adapter, so a refusal there is a failure.
            assert!(
                !evidence::block_sum_kernels::block_sum::DESCRIPTOR
                    .requires
                    .is_empty(),
                "the portable evidence body was refused: {plain}"
            );
            assert_eq!(plain.to_string(), timed.to_string());
        }
        (plain, timed) => {
            panic!("timed and untimed evidence::sum_u32 disagree: {plain:?} / {timed:?}")
        }
    }
    // The v0 fallback form is one batch too.
    let t = crate::timed(|| reduce::sum_with(Variant::V0, &ints)).unwrap();
    assert_eq!(t.value, reduce::sum_with(Variant::V0, &ints).unwrap());
    assert_time_rule(&device, t.time, 1);

    for exclusive in [false, true] {
        let run = |input: &GpuVec<u32>| {
            if exclusive {
                scan::exclusive(input)
            } else {
                scan::inclusive(input)
            }
        };
        let t = crate::timed(|| run(&ints)).unwrap();
        assert_eq!(
            t.value.to_vec().unwrap(),
            run(&ints).unwrap().to_vec().unwrap()
        );
        // One batch however many dispatches the scan has.
        assert_time_rule(&device, t.time, 1);
    }

    let values = GpuVec::from_slice_on(&device, &[10u32, 20, 30, 40]).unwrap();
    let indices = GpuVec::from_slice_on(&device, &[3u32, 1, 0, 0]).unwrap();
    let count = GpuVec::from_slice_on(&device, &[2u32]).unwrap();
    let t = crate::timed(|| compact::gather_u32(&values, &indices, &count)).unwrap();
    assert_eq!(t.value.to_vec().unwrap(), [40, 20, 0, 0]);
    assert_time_rule(&device, t.time, 1);
}

/// D194: a multi-batch operation sums its batches (and says how many), and so does a closure of several calls.
#[test]
fn gpu_timed_multi_batch_operations_sum_their_batches() {
    let device = device();
    let flags: Vec<u32> = (0..5_000u32).map(|i| u32::from(i % 3 == 0)).collect();
    let flags = GpuVec::from_slice_on(&device, &flags).unwrap();

    let t = crate::timed(|| compact::count(&flags)).unwrap();
    assert_eq!(t.value.to_vec().unwrap(), [1667]);
    assert_time_rule(&device, t.time, 2);

    let t = crate::timed(|| compact::indices(&flags)).unwrap();
    let (indices, count) = &t.value;
    assert_eq!(count.to_vec().unwrap(), [1667]);
    assert_eq!(&indices.to_vec().unwrap()[..4], [0, 3, 6, 9]);
    assert_time_rule(&device, t.time, 4);

    let ints = GpuVec::from_slice_on(&device, &[1u32, 2, 3]).unwrap();
    let t = crate::timed(|| {
        let a = reduce::sum(&ints)?;
        let b = reduce::max(&ints)?;
        Ok((a, b))
    })
    .unwrap();
    assert_eq!(t.value, (6, 3));
    assert_time_rule(&device, t.time, 2);

    // Nested: the inner scope has its own total, the outer one includes it.
    let t = crate::timed(|| {
        let inner = crate::timed(|| reduce::sum(&ints))?;
        assert_time_rule(&device, inner.time, 1);
        reduce::min(&ints)
    })
    .unwrap();
    assert_eq!(t.value, 1);
    assert_time_rule(&device, t.time, 2);
}

/// D194: an empty input dispatches no batch, an error comes back as the error, and a thread other than the caller is
/// not counted.
#[test]
fn gpu_timed_empty_error_and_other_threads() {
    let device = device();
    let empty = GpuVec::<u32>::from_slice_on(&device, &[]).unwrap();
    let t = crate::timed(|| reduce::sum(&empty)).unwrap();
    assert_eq!(t.value, 0);
    assert_eq!(t.time.batches, 0);
    assert_eq!(t.time.gpu_elapsed, None);
    assert!(matches!(
        crate::timed(|| reduce::min(&empty)),
        Err(Error::Empty)
    ));

    let ints = GpuVec::from_slice_on(&device, &[4u32, 5, 6]).unwrap();
    let t = crate::timed(|| {
        // The sum on another thread is not part of this scope; the one on this thread is.
        let elsewhere =
            std::thread::scope(|scope| scope.spawn(|| reduce::sum(&ints)).join().unwrap())?;
        assert_eq!(elsewhere, 15);
        reduce::max(&ints)
    })
    .unwrap();
    assert_eq!(t.value, 6);
    assert_time_rule(&device, t.time, 1);
}

/// D194: after timed scopes (one that failed, one that panicked) an untimed call is unchanged and a new scope starts
/// from zero.
#[test]
fn gpu_untimed_calls_are_unaffected_by_earlier_timed_scopes() {
    let device = device();
    let ints = GpuVec::from_slice_on(&device, &[1u32, 2, 3, 4]).unwrap();
    assert!(crate::timed(|| reduce::sum(&ints).and(Err::<(), _>(Error::Empty))).is_err());
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = crate::timed(|| -> Result<(), Error> {
            reduce::sum(&ints)?;
            panic!("inside the scope");
        });
    }));
    assert!(panicked.is_err());
    assert!(
        !crate::timing::scope_is_open(),
        "a failed or panicked scope left a timed scope open on this thread"
    );
    assert_eq!(reduce::sum(&ints).unwrap(), 10);
    let t = crate::timed(|| reduce::sum(&ints)).unwrap();
    assert_time_rule(&device, t.time, 1);
}

// ---- m11.scope: a caller's own batch in a `timed` scope ----

#[gust::gpu_module]
pub mod doubling {
    /// `out[i] = values[i] * 2` for `i < out.len()`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        values: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = values[i] * 2u32;
        }
    }
}

#[test]
fn gpu_timed_own_batch_and_reduce_share_one_scope() {
    let device = device();
    let info = device.adapter_info();
    eprintln!("adapter: {} backend: {:?}", info.name, info.backend);

    let words: Vec<u32> = (0..5_000u32).map(|i| i % 97).collect();
    let length = u32::try_from(words.len()).unwrap();
    let values = GpuVec::from_slice_on(&device, &words).unwrap();
    let doubled_words: Vec<u32> = words.iter().map(|word| word * 2).collect();
    let expected: u32 = doubled_words.iter().sum();
    let timestamps = device.supports_gpu_timestamps();

    // A fresh zeroed output for every dispatch, so a value read back proves that this dispatch ran.
    let fresh = || GpuVec::<u32>::zeroed_on(&device, length).unwrap();
    let own = |device: &gust_wgpu::HeadlessDevice, out: &GpuVec<u32>| {
        let call = doubling::run::call(length, gust::Independent(&values), gust::Independent(out));
        crate::dispatch_batch_in_scope(device, &[gust_wgpu::BufferDispatch::from(&call)])
    };

    // Outside a scope the batch runs, and a scope opened afterwards starts from zero.
    let unscoped = fresh();
    own(&device, &unscoped).unwrap();
    assert_eq!(unscoped.to_vec().unwrap(), doubled_words);
    assert_eq!(
        crate::timed(|| Ok(())).unwrap().time,
        crate::GpuTime::default()
    );

    // An empty slice is not a batch: it counts nothing, in a scope.
    let empty =
        crate::timed(|| crate::dispatch_batch_in_scope(&device, &[]).map_err(Error::from)).unwrap();
    assert_eq!(empty.time, crate::GpuTime::default());

    // The parts, each in a scope of its own, then both in one.
    let out = fresh();
    let part_own = crate::timed(|| own(&device, &out).map_err(Error::from))
        .unwrap()
        .time;
    assert_eq!(out.to_vec().unwrap(), doubled_words);
    let part_sum = crate::timed(|| reduce::sum(&out)).unwrap();
    assert_eq!(part_sum.value, expected);
    assert_eq!(part_own.batches, 1);
    assert_eq!(part_sum.time.batches, 1);

    let out = fresh();
    let both = crate::timed(|| {
        own(&device, &out)?;
        reduce::sum(&out)
    })
    .unwrap();
    assert_eq!(both.value, expected, "the in-scope own batch ran");
    assert_eq!(both.time.batches, 2, "one own batch and one algo batch");
    assert_eq!(both.time.gpu_elapsed.is_some(), timestamps);

    if timestamps {
        // The exact rule is pinned with nested scopes: the outer total is the sum of the inner ones, and the own batch
        // adds a real, non-zero GPU time.
        let out = fresh();
        let nested = crate::timed(|| {
            let inner_own = crate::timed(|| own(&device, &out).map_err(Error::from))?;
            let inner_sum = crate::timed(|| reduce::sum(&out))?;
            Ok((inner_own.time, inner_sum.time))
        })
        .unwrap();
        let (inner_own, inner_sum) = nested.value;
        let (own_time, sum_time) = (
            inner_own.gpu_elapsed.unwrap(),
            inner_sum.gpu_elapsed.unwrap(),
        );
        assert_eq!(nested.time.batches, 2);
        assert_eq!(nested.time.gpu_elapsed, Some(own_time + sum_time));
        assert!(own_time > std::time::Duration::ZERO);
        assert!(nested.time.gpu_elapsed.unwrap() >= own_time);
        assert!(nested.time.gpu_elapsed.unwrap() >= sum_time);
    }
}
