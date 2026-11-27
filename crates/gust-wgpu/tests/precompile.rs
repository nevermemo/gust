//! `HeadlessDevice::precompile`: kernels created ahead of use (slice m14.bzpre).
//!
//! The counters of `pipeline_cache_stats` are the evidence that a kernel was created by the call and found by the
//! dispatch; the results of the dispatches (against a host loop) show that what was created is the kernel that runs.
//! No time is asserted: `elapsed` is only checked for being there on a native build and absent on wasm32.

#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]

mod common;

use common::{assert_cold, device, device_reported};

use gust::{KernelDescriptor, target::Target};
use gust_wgpu::{
    BufferBinding, BufferDispatch, Error, GpuBuffer, GpuBufferAccess, HeadlessDevice,
    PrecompileOutcome,
};

#[gust::gpu_module]
mod kernels {
    /// `dst[i] = src[i] + 1`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn inc(
        id: SV_DispatchThreadID,
        src: StructuredBuffer<u32>,
        mut dst: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < dst.len() && i < src.len() {
            dst[i] = src[i] + 1u32;
        }
    }

    /// `dst[i] = src[i] * 2`.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn dbl(
        id: SV_DispatchThreadID,
        src: StructuredBuffer<u32>,
        mut dst: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        if i < dst.len() && i < src.len() {
            dst[i] = src[i] * 2u32;
        }
    }

    /// A `pixel` kernel: `dst[i] = src[i] + 3`.
    #[kernel(shape = pixel)]
    pub fn shade(
        id: SV_DispatchThreadID,
        src: StructuredBuffer<u32>,
        mut dst: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        dst[i] = src[i] + 3u32;
    }
}

/// Whether `elapsed` is what the build promises: a time on a native build, nothing on wasm32.
fn elapsed_is_as_promised(report: &gust_wgpu::Precompiled) -> bool {
    report.elapsed.is_some() == cfg!(not(target_arch = "wasm32"))
}

fn input(n: u32) -> Vec<u32> {
    (0..n).map(|i| i * 7 + 1).collect()
}

fn expected(n: u32, f: impl Fn(u32) -> u32) -> Vec<u32> {
    input(n).into_iter().map(f).collect()
}

fn buffers(device: &HeadlessDevice, n: u32) -> (GpuBuffer<u32>, GpuBuffer<u32>) {
    (
        device.create_typed_buffer("src", &input(n), GpuBufferAccess::ReadWrite),
        device.create_typed_buffer("dst", &vec![0u32; n as usize], GpuBufferAccess::ReadWrite),
    )
}

/// One dispatch of `kernel` over `n` elements through the batch path, and what it wrote.
fn run(device: &HeadlessDevice, kernel: &'static KernelDescriptor, n: u32) -> Vec<u32> {
    let (src, dst) = buffers(device, n);
    let bindings = [
        BufferBinding::read_only(&src),
        BufferBinding::read_write(&dst),
    ];
    device
        .dispatch_batch(&[BufferDispatch::new(kernel, n, &bindings)])
        .unwrap();
    device.read_typed_buffer(&dst).unwrap()
}

fn counters(device: &HeadlessDevice) -> (usize, u64, u64) {
    let stats = device.pipeline_cache_stats();
    (stats.entries, stats.misses, stats.hits)
}

/// (1) The call creates, in order; the dispatches find. The report says which, and names each kernel.
#[test]
fn precompile_creates_the_kernels_and_the_dispatches_find_them() {
    let device = device_reported();
    assert_cold(&device, "a new device");
    let (inc, dbl) = (&kernels::inc::DESCRIPTOR, &kernels::dbl::DESCRIPTOR);
    let report = device.precompile(&[inc, dbl]).unwrap();
    assert_eq!(report.len(), 2);
    assert_eq!(report[0].kernel, inc.qualified_name().to_string());
    assert_eq!(report[1].kernel, dbl.qualified_name().to_string());
    for entry in &report {
        assert_eq!(entry.outcome, PrecompileOutcome::Compiled, "{entry:?}");
        assert!(elapsed_is_as_promised(entry), "{entry:?}");
    }
    assert_eq!(counters(&device), (2, 2, 0));

    let n = 100;
    assert_eq!(run(&device, inc, n), expected(n, |v| v + 1));
    assert_eq!(run(&device, dbl, n), expected(n, |v| v * 2));
    assert_eq!(
        counters(&device),
        (2, 2, 2),
        "both dispatches found their kernel"
    );
}

/// (2) A repeat is a hit, counted like any lookup; a descriptor twice in one call is a miss and then a hit.
#[test]
fn a_repeat_is_a_hit_and_adds_no_miss() {
    let device = device();
    let (inc, dbl) = (&kernels::inc::DESCRIPTOR, &kernels::dbl::DESCRIPTOR);
    device.precompile(&[inc, dbl]).unwrap();
    let again = device.precompile(&[inc, dbl]).unwrap();
    assert_eq!(again.len(), 2);
    for entry in &again {
        assert_eq!(entry.outcome, PrecompileOutcome::Hit, "{entry:?}");
        assert!(elapsed_is_as_promised(entry), "{entry:?}");
    }
    assert_eq!(counters(&device), (2, 2, 2));

    let fresh = common::device();
    let report = fresh.precompile(&[inc, inc]).unwrap();
    let outcomes: Vec<_> = report.iter().map(|entry| entry.outcome).collect();
    assert_eq!(
        outcomes,
        [PrecompileOutcome::Compiled, PrecompileOutcome::Hit]
    );
    assert_eq!(counters(&fresh), (1, 1, 1));
}

/// An empty list is a call that does nothing.
#[test]
fn an_empty_list_reports_nothing() {
    let device = device();
    assert!(device.precompile(&[]).unwrap().is_empty());
    assert_cold(&device, "an empty list");
}

/// (3) `clear_pipeline_cache` undoes it: the next dispatch is a miss again.
#[test]
fn after_clear_pipeline_cache_a_dispatch_is_a_miss_again() {
    let device = device();
    let inc = &kernels::inc::DESCRIPTOR;
    device.precompile(&[inc]).unwrap();
    device.clear_pipeline_cache();
    assert_cold(&device, "a cleared cache");
    let n = 10;
    assert_eq!(run(&device, inc, n), expected(n, |v| v + 1));
    assert_eq!(counters(&device), (1, 1, 0), "the dispatch compiled it");
}

/// (4) A `pixel` kernel is created too, and a later pixel dispatch finds it. An adapter that cannot write storage from
/// a fragment stage gives the dispatch's own error instead.
#[test]
fn a_pixel_kernel_is_precompiled_or_refused_as_the_dispatch_refuses_it() {
    let device = device();
    let shade = &kernels::shade::DESCRIPTOR;
    let n = 64;
    match device.precompile(&[shade]) {
        Ok(report) => {
            assert_eq!(report[0].outcome, PrecompileOutcome::Compiled);
            assert_eq!(counters(&device), (1, 1, 0));
            assert_eq!(run(&device, shade, n), expected(n, |v| v + 3));
            assert_eq!(counters(&device), (1, 1, 1), "the pixel dispatch found it");
        }
        Err(refused) => {
            assert!(
                matches!(refused, Error::ShapeUnsupported { .. }),
                "{refused:?}"
            );
            let (src, dst) = buffers(&device, n);
            let bindings = [
                BufferBinding::read_only(&src),
                BufferBinding::read_write(&dst),
            ];
            let dispatched = device
                .dispatch_batch(&[BufferDispatch::new(shade, n, &bindings)])
                .unwrap_err();
            assert_eq!(format!("{refused:?}"), format!("{dispatched:?}"));
            assert_eq!(device.pipeline_cache_stats().entries, 0);
        }
    }
}

/// (6) The split variant is its own entry: both are created, both are found on a repeat, and a launch past a lowered
/// workgroup limit (which runs as the variant) adds no miss.
#[test]
fn the_split_variant_is_created_when_named_and_the_oversized_launch_adds_no_miss() {
    let mut limits = device().limits().clone();
    limits.max_compute_workgroups_per_dimension = 4;
    let device = device()
        .with_limits(limits)
        .expect("lowering limits is allowed");
    let base = &kernels::inc::DESCRIPTOR;
    let variant = base.split.expect("a grid kernel carries its split variant");

    let report = device.precompile(&[base, variant]).unwrap();
    let outcomes: Vec<_> = report.iter().map(|entry| entry.outcome).collect();
    assert_eq!(
        outcomes,
        [PrecompileOutcome::Compiled, PrecompileOutcome::Compiled]
    );
    assert_eq!(counters(&device), (2, 2, 0));
    let repeat = device.precompile(&[base, variant]).unwrap();
    assert!(
        repeat
            .iter()
            .all(|entry| entry.outcome == PrecompileOutcome::Hit),
        "{repeat:?}"
    );
    assert_eq!(counters(&device), (2, 2, 2));

    // 4 workgroups of 64 hold 256 elements: this launch needs 10 workgroups and runs as pieces of the variant.
    let n = 640;
    assert_eq!(run(&device, base, n), expected(n, |v| v + 1));
    assert_eq!(
        counters(&device),
        (2, 2, 3),
        "the oversized launch created nothing and found the variant"
    );
}

/// (7) The list stops at the first error: the entries before it stay created, and the error is the one a dispatch of
/// the refused descriptor gives.
#[test]
fn the_list_stops_at_the_first_error_and_earlier_entries_stay_created() {
    let device = device();
    let inc = &kernels::inc::DESCRIPTOR;
    let cpu = kernels::dbl::DESCRIPTOR.with_target(Target::Cpu);
    let error = device
        .precompile(&[inc, &cpu, &kernels::shade::DESCRIPTOR])
        .unwrap_err();
    assert!(
        matches!(
            error,
            Error::KernelTarget {
                target: Target::Cpu,
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(
        counters(&device),
        (1, 1, 0),
        "the first descriptor is created and nothing after the refused one"
    );
    let n = 10;
    assert_eq!(run(&device, inc, n), expected(n, |v| v + 1));
    assert_eq!(counters(&device), (1, 1, 1), "the first stayed cached");
}

/// A kernel the device could never launch is refused at `precompile` with the dispatch's own error, before anything is
/// created.
#[test]
fn a_kernel_past_a_device_limit_is_refused_as_the_dispatch_refuses_it() {
    let mut limits = device().limits().clone();
    limits.max_compute_workgroup_size_x = 32;
    let device = device()
        .with_limits(limits)
        .expect("lowering limits is allowed");
    let inc = &kernels::inc::DESCRIPTOR;
    let refused = device.precompile(&[inc]).unwrap_err();
    assert!(
        matches!(&refused, Error::DeviceLimit { limit, .. } if *limit == "max_compute_workgroup_size_x"),
        "{refused:?}"
    );
    assert_cold(&device, "a limit error");
    let n = 10;
    let (src, dst) = buffers(&device, n);
    let bindings = [
        BufferBinding::read_only(&src),
        BufferBinding::read_write(&dst),
    ];
    let dispatched = device
        .dispatch_batch(&[BufferDispatch::new(inc, n, &bindings)])
        .unwrap_err();
    assert_eq!(format!("{refused:?}"), format!("{dispatched:?}"));
}

/// (8) The ruling: `precompile` then the typed `dispatch` is one entry and a hit, wherever the typed call keeps its
/// descriptor. Typed calls read the descriptor through `Kernel`, at other addresses than `DESCRIPTOR` here.
#[test]
fn precompile_then_a_typed_dispatch_is_one_entry_and_a_hit() {
    let device = device_reported();
    device.precompile(&[&kernels::inc::DESCRIPTOR]).unwrap();
    assert_eq!(counters(&device), (1, 1, 0));
    let n = 100;
    let (src, dst) = buffers(&device, n);
    kernels::inc::dispatch(&device, n, &src, &dst).unwrap();
    assert_eq!(
        device.read_typed_buffer(&dst).unwrap(),
        expected(n, |v| v + 1)
    );
    assert_eq!(counters(&device), (1, 1, 1), "the typed call found it");
    // And the other way round: a typed call first, then the call is a hit and still one entry.
    kernels::dbl::dispatch(&device, n, &src, &dst).unwrap();
    assert_eq!(counters(&device), (2, 2, 1));
    let report = device.precompile(&[&kernels::dbl::DESCRIPTOR]).unwrap();
    assert_eq!(report[0].outcome, PrecompileOutcome::Hit);
    assert_eq!(counters(&device), (2, 2, 2));
}

/// (9) A kernel is the same by content, not by address: two copies of one descriptor, at two addresses, are one entry,
/// and the first lookup of the second address is a hit (content that differs is covered by the key unit tests).
#[test]
fn two_copies_of_one_descriptor_at_different_addresses_are_one_entry() {
    let device = device();
    let inc = &kernels::inc::DESCRIPTOR;
    device.precompile(&[inc]).unwrap();

    let mut copy = kernels::inc::DESCRIPTOR;
    copy.slang_source = Box::leak(copy.slang_source.to_owned().into_boxed_str());
    assert_ne!(copy.slang_source.as_ptr(), inc.slang_source.as_ptr());
    let report = device.precompile(&[&copy]).unwrap();
    assert_eq!(report[0].outcome, PrecompileOutcome::Hit);
    assert_eq!(counters(&device), (1, 1, 1));
    // The alias now serves the copy, and a repeat is a hit too.
    let report = device.precompile(&[&copy, inc]).unwrap();
    assert!(
        report
            .iter()
            .all(|entry| entry.outcome == PrecompileOutcome::Hit),
        "{report:?}"
    );
    assert_eq!(counters(&device), (1, 1, 3));
}

/// (10) The split variant precompiled, then hit by an oversized typed launch: the typed call reaches the variant
/// through `split` and finds it by content.
#[test]
fn the_split_variant_precompiled_is_hit_by_an_oversized_typed_launch() {
    let mut limits = device().limits().clone();
    limits.max_compute_workgroups_per_dimension = 4;
    let device = device()
        .with_limits(limits)
        .expect("lowering limits is allowed");
    let base = &kernels::inc::DESCRIPTOR;
    let variant = base.split.expect("a grid kernel carries its split variant");
    device.precompile(&[base, variant]).unwrap();
    assert_eq!(counters(&device), (2, 2, 0));
    let n = 640;
    let (src, dst) = buffers(&device, n);
    kernels::inc::dispatch(&device, n, &src, &dst).unwrap();
    assert_eq!(
        device.read_typed_buffer(&dst).unwrap(),
        expected(n, |v| v + 1)
    );
    assert_eq!(
        counters(&device),
        (2, 2, 1),
        "the typed launch created nothing and found the variant (as the batch launch of test 6 does)"
    );
}
