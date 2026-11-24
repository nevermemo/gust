//! The recording form on a real adapter (slice m14.s2f1): a list of recorded operations, run as one `dispatch_batch`,
//! gives the results of the immediate functions (and of the independent host references), at a power of two, at a
//! size that is not one, and at a size with more than one pass of both variants. Each test names the adapter in its
//! log line through `gpu::device`.

use std::any::Any;

use gust_wgpu::{GpuBufferAccess, GpuVec};

use super::{
    common::{Rng, assert_same, reference_indices, reference_scan},
    gpu::device,
};
use crate::{
    Error, Variant,
    compact::{self, CompactPlan, CountPlan},
    record::{Recording, Sink},
    reduce::{self, Op, ReducePlan, SumWrappingPlan},
    scan::{self, ScanPlan},
    schedule::{BlockSchedule, Schedule, TileSchedule},
    timed,
};

/// 4096 is a power of two inside one block, 1000 is not one, and 100 003 is not one and takes two passes of v0 and
/// of v1 (a pass covers 65 536 and 8192 elements).
const SIZES: [usize; 3] = [4096, 1000, 100_003];

fn random(n: usize, seed: u64, below: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.below(below) as u32).collect()
}

#[test]
fn gpu_a_recorded_reduction_list_equals_the_immediate_results() {
    let device = device();
    for n in SIZES {
        let values = random(n, 11, 1 << 15);
        let signed: Vec<i32> = values.iter().map(|v| *v as i32 - (1 << 14)).collect();
        let floats: Vec<f32> = values.iter().map(|v| *v as f32 / 1024.0).collect();
        let input = GpuVec::from_slice_on(&device, &values).unwrap();
        let signed_input = GpuVec::from_slice_on(&device, &signed).unwrap();
        let float_input = GpuVec::from_slice_on(&device, &floats).unwrap();
        for variant in [Variant::V0, Variant::V1] {
            // Every output starts at a sentinel: a stage that never wrote its result, or wrote another stage's, fails
            // (the `min` of the 100 003-element input is probably 0, which a zeroed buffer would hide).
            let one = || GpuVec::from_slice_on(&device, &[0xDEAD_BEEF_u32]).unwrap();
            let (sum, min, max, wrapping) = (one(), one(), one(), one());
            let signed_min = GpuVec::from_slice_on(&device, &[0x5EED_5EED_i32]).unwrap();
            let float_sum = GpuVec::from_slice_on(&device, &[f32::NAN]).unwrap();
            let mut recording = Recording::new(&device).with_variant(variant);
            recording.sum(&input, &sum).unwrap();
            recording.min(&input, &min).unwrap();
            recording.max(&input, &max).unwrap();
            recording.sum_wrapping(&input, &wrapping).unwrap();
            recording.min(&signed_input, &signed_min).unwrap();
            recording.sum(&float_input, &float_sum).unwrap();

            let per_reduction = match variant {
                Variant::V0 => Schedule::new(n as u32).unwrap().dispatches(),
                Variant::V1 => BlockSchedule::new(n as u32).unwrap().dispatches(),
            };
            let label = format!("{variant:?}, size {n}");
            assert_eq!(recording.dispatch_count(), 6 * per_reduction, "{label}");
            // One list, one batch.
            let timed_run = timed(|| recording.dispatch()).unwrap();
            assert_eq!(timed_run.time.batches, 1, "{label}");

            assert_eq!(
                sum.to_vec().unwrap(),
                [reduce::sum_with(variant, &input).unwrap()],
                "sum, {label}"
            );
            assert_eq!(
                min.to_vec().unwrap(),
                [reduce::min_with(variant, &input).unwrap()],
                "min, {label}"
            );
            assert_eq!(
                max.to_vec().unwrap(),
                [reduce::max_with(variant, &input).unwrap()],
                "max, {label}"
            );
            assert_eq!(
                wrapping.to_vec().unwrap(),
                [reduce::sum_wrapping_with(variant, &input).unwrap()],
                "sum_wrapping, {label}"
            );
            assert_eq!(
                signed_min.to_vec().unwrap(),
                [reduce::min_with(variant, &signed_input).unwrap()],
                "i32 min, {label}"
            );
            // The same kernels in the same order, no atomics: the float sums are bit for bit the immediate ones.
            assert_eq!(
                float_sum.to_vec().unwrap()[0].to_bits(),
                reduce::sum_with(variant, &float_input).unwrap().to_bits(),
                "f32 sum, {label}"
            );
            // And the independent host references.
            let host_sum: u32 = values.iter().sum();
            assert_eq!(sum.to_vec().unwrap(), [host_sum], "host sum, {label}");
            assert_eq!(
                min.to_vec().unwrap(),
                [*values.iter().min().unwrap()],
                "host min, {label}"
            );
            assert_eq!(
                max.to_vec().unwrap(),
                [*values.iter().max().unwrap()],
                "host max, {label}"
            );
        }
    }
}

#[test]
fn gpu_a_recorded_scan_list_equals_the_immediate_results() {
    let device = device();
    for n in SIZES {
        let values = random(n, 23, 4);
        let input = GpuVec::from_slice_on(&device, &values).unwrap();
        for variant in [Variant::V0, Variant::V1] {
            let inclusive = GpuVec::<u32>::zeroed_on(&device, n as u32).unwrap();
            let exclusive = GpuVec::<u32>::zeroed_on(&device, n as u32).unwrap();
            let mut recording = Recording::new(&device).with_variant(variant);
            recording.inclusive(&input, &inclusive).unwrap();
            recording.exclusive(&input, &exclusive).unwrap();

            let per_scan = match variant {
                Variant::V0 => crate::schedule::ScanSchedule::new(n as u32)
                    .unwrap()
                    .dispatches(),
                Variant::V1 => TileSchedule::new(n as u32).unwrap().dispatches(),
            };
            let label = format!("{variant:?}, size {n}");
            assert_eq!(recording.dispatch_count(), 2 * per_scan, "{label}");
            recording.dispatch().unwrap();

            let immediate_inclusive = scan::inclusive_with(variant, &input).unwrap();
            let immediate_exclusive = scan::exclusive_with(variant, &input).unwrap();
            assert_same(
                &inclusive.to_vec().unwrap(),
                &immediate_inclusive.to_vec().unwrap(),
                &format!("inclusive against immediate, {label}"),
            );
            assert_same(
                &exclusive.to_vec().unwrap(),
                &immediate_exclusive.to_vec().unwrap(),
                &format!("exclusive against immediate, {label}"),
            );
            assert_same(
                &inclusive.to_vec().unwrap(),
                &reference_scan(&values, false),
                &format!("inclusive against the host, {label}"),
            );
            assert_same(
                &exclusive.to_vec().unwrap(),
                &reference_scan(&values, true),
                &format!("exclusive against the host, {label}"),
            );
        }
    }
}

#[test]
fn gpu_a_recorded_compaction_list_equals_the_immediate_results() {
    let device = device();
    for n in SIZES {
        // Flags that are not only 0 and 1: every non-zero flag is set.
        let flags: Vec<u32> = random(n, 37, 3).iter().map(|f| f * 5).collect();
        let input = GpuVec::from_slice_on(&device, &flags).unwrap();
        let expected = reference_indices(&flags);
        let live = u32::try_from(expected.len()).unwrap();
        let live_len = expected.len();

        // The counts and the indices start full of a sentinel. A run writes the counts whole and only the first
        // `count` elements of `indices`: the tail is unspecified by the recording form's contract (the zero tail is
        // the immediate function's promise), and what this run leaves there is the sentinel.
        let counted = GpuVec::from_slice_on(&device, &[0xDEAD_BEEF_u32]).unwrap();
        let indices = GpuVec::from_slice_on(&device, &vec![0xDEAD_BEEF_u32; n]).unwrap();
        let total = GpuVec::from_slice_on(&device, &[0xDEAD_BEEF_u32]).unwrap();

        let mut recording = Recording::new(&device);
        recording.count(&input, &counted).unwrap();
        recording.indices(&input, &indices, &total).unwrap();
        let timed_run = timed(|| recording.dispatch()).unwrap();
        assert_eq!(timed_run.time.batches, 1, "size {n}");

        assert_eq!(counted.to_vec().unwrap(), [live], "count, size {n}");
        assert_eq!(total.to_vec().unwrap(), [live], "indices count, size {n}");
        let recorded = indices.to_vec().unwrap();
        assert_same(
            &recorded[..live_len],
            &expected,
            &format!("indices prefix against the host, size {n}"),
        );
        let (immediate_indices, immediate_count) = compact::indices(&input).unwrap();
        assert_same(
            &recorded[..live_len],
            &immediate_indices.to_vec().unwrap()[..live_len],
            &format!("indices prefix against immediate, size {n}"),
        );
        assert!(
            recorded[live_len..].iter().all(|word| *word == 0xDEAD_BEEF),
            "the tail past the count holds what the buffer held, size {n}"
        );
        assert_eq!(
            total.to_vec().unwrap(),
            immediate_count.to_vec().unwrap(),
            "count against immediate, size {n}"
        );
        assert_eq!(
            counted.to_vec().unwrap(),
            compact::count(&input).unwrap().to_vec().unwrap(),
            "count against immediate count, size {n}"
        );
    }
}

/// A later call reads the `out` of an earlier one: the dispatches run in the order recorded.
#[test]
fn gpu_a_recorded_list_runs_in_order_and_runs_again() {
    let device = device();
    for n in SIZES {
        let values = random(n, 41, 2);
        let input = GpuVec::from_slice_on(&device, &values).unwrap();
        let prefix = GpuVec::from_slice_on(&device, &vec![0xFFFF_u32; n]).unwrap();
        let total = GpuVec::from_slice_on(&device, &[0xFFFF_u32]).unwrap();
        let mut recording = Recording::new(&device);
        recording.inclusive(&input, &prefix).unwrap();
        recording.sum(&prefix, &total).unwrap();
        let expected_prefix = reference_scan(&values, false);
        let expected_total: u64 = expected_prefix.iter().map(|v| u64::from(*v)).sum();
        // Flags of 0 and 1 keep the sum of the prefixes inside u32 at every size (about n^2 / 4).
        assert!(
            expected_total <= u64::from(u32::MAX),
            "size {n} leaves the domain of the sum"
        );
        for run in 0..2 {
            recording.dispatch().unwrap();
            assert_same(
                &prefix.to_vec().unwrap(),
                &expected_prefix,
                &format!("prefix, run {run}, size {n}"),
            );
            assert_eq!(
                total.to_vec().unwrap(),
                [expected_total as u32],
                "total of the prefix, run {run}, size {n}"
            );
        }
    }
}

#[test]
fn gpu_recording_refuses_what_it_cannot_record() {
    let device = device();
    let input = GpuVec::from_slice_on(&device, &[3u32, 1, 4, 1, 5]).unwrap();
    let empty = GpuVec::<u32>::zeroed_on(&device, 0).unwrap();
    let one = GpuVec::<u32>::zeroed_on(&device, 1).unwrap();
    let two = GpuVec::<u32>::zeroed_on(&device, 2).unwrap();
    let five = GpuVec::<u32>::zeroed_on(&device, 5).unwrap();
    let mut recording = Recording::new(&device);

    assert!(matches!(
        recording.sum(&input, &two),
        Err(Error::Length {
            expected: 1,
            actual: 2
        })
    ));
    assert!(matches!(recording.sum(&empty, &one), Err(Error::Empty)));
    assert!(matches!(recording.count(&empty, &one), Err(Error::Empty)));
    assert!(matches!(
        recording.inclusive(&input, &two),
        Err(Error::Length {
            expected: 5,
            actual: 2
        })
    ));
    assert!(matches!(
        recording.indices(&input, &two, &one),
        Err(Error::Length {
            expected: 5,
            actual: 2
        })
    ));
    assert!(matches!(
        recording.indices(&input, &five, &two),
        Err(Error::Length {
            expected: 1,
            actual: 2
        })
    ));
    // An empty scan has nothing to do and nothing to leave stale: it records nothing.
    recording.exclusive(&empty, &empty).unwrap();
    assert_eq!(recording.dispatch_count(), 0);
    // Nothing was recorded by the refusals, and an empty list dispatches nothing.
    let batches = timed(|| recording.dispatch()).unwrap().time.batches;
    assert_eq!(batches, 0);
}

/// A plan holds no borrow (D259): it is kept in a `Box<dyn Any>`, which needs `'static`, and appended over `GpuBuffer`
/// operands that are not `GpuVec`s, then again over `GpuVec`s. Each run gives the immediate results.
#[test]
fn gpu_a_plan_held_without_a_borrow_appends_over_any_operand() {
    let device = device();
    for n in SIZES {
        let len = n as u32;
        let values = random(n, 53, 2);
        let plans: Vec<Box<dyn Any>> = vec![
            Box::new(ReducePlan::<u32>::new(&device, Variant::V1, Op::Sum, len).unwrap()),
            Box::new(ScanPlan::new(&device, Variant::V1, len, false).unwrap()),
            Box::new(CompactPlan::new(&device, None, len).unwrap()),
            Box::new(SumWrappingPlan::new(&device, Variant::V0, len).unwrap()),
            Box::new(CountPlan::new(&device, None, len).unwrap()),
        ];
        let reduce_plan = plans[0].downcast_ref::<ReducePlan<u32>>().unwrap();
        let scan_plan = plans[1].downcast_ref::<ScanPlan>().unwrap();
        let compact_plan = plans[2].downcast_ref::<CompactPlan>().unwrap();
        let wrapping_plan = plans[3].downcast_ref::<SumWrappingPlan>().unwrap();
        let count_plan = plans[4].downcast_ref::<CountPlan>().unwrap();

        let expected_prefix = reference_scan(&values, false);
        let expected_indices = reference_indices(&values);
        let live = u32::try_from(expected_indices.len()).unwrap();
        let host_sum: u32 = values.iter().sum();

        // Run 1: every operand is a `GpuBuffer`, the outputs start at a sentinel.
        let make = |label: &str, data: &[u32]| {
            device.create_typed_buffer(label, data, GpuBufferAccess::ReadWrite)
        };
        let input = make("plan input", &values);
        let total = make("plan total", &[0xDEAD_BEEF]);
        let wrapped = make("plan wrapped", &[0xDEAD_BEEF]);
        let prefix = make("plan prefix", &vec![0xDEAD_BEEF; n]);
        let indices = make("plan indices", &vec![0xDEAD_BEEF; n]);
        let live_count = make("plan count", &[0xDEAD_BEEF]);
        let counted = make("plan counted", &[0xDEAD_BEEF]);
        let mut sink = Sink::new();
        reduce_plan.append(&mut sink, &input, &total).unwrap();
        wrapping_plan.append(&mut sink, &input, &wrapped).unwrap();
        scan_plan.append(&mut sink, &input, &prefix).unwrap();
        compact_plan
            .append(&mut sink, &input, &indices, &live_count)
            .unwrap();
        count_plan.append(&mut sink, &input, &counted).unwrap();
        sink.dispatch(&device).unwrap();
        let read = |buffer| device.read_typed_buffer(buffer).unwrap();
        assert_eq!(read(&total), [host_sum], "sum over buffers, size {n}");
        assert_eq!(
            read(&wrapped),
            [host_sum],
            "wrapping sum over buffers, size {n}"
        );
        assert_same(
            &read(&prefix),
            &expected_prefix,
            &format!("scan over buffers, size {n}"),
        );
        assert_eq!(read(&live_count), [live], "indices count over buffers");
        assert_eq!(read(&counted), [live], "count over buffers, size {n}");
        assert_same(
            &read(&indices)[..expected_indices.len()],
            &expected_indices,
            &format!("indices prefix over buffers, size {n}"),
        );

        // Run 2: the same plans again, over `GpuVec`s.
        let input = GpuVec::from_slice_on(&device, &values).unwrap();
        let total = GpuVec::from_slice_on(&device, &[0xDEAD_BEEF_u32]).unwrap();
        let prefix = GpuVec::from_slice_on(&device, &vec![0xDEAD_BEEF_u32; n]).unwrap();
        let mut sink = Sink::new();
        reduce_plan.append(&mut sink, &input, &total).unwrap();
        scan_plan.append(&mut sink, &input, &prefix).unwrap();
        sink.dispatch(&device).unwrap();
        assert_eq!(total.to_vec().unwrap(), [host_sum], "sum again, size {n}");
        assert_same(
            &prefix.to_vec().unwrap(),
            &expected_prefix,
            &format!("scan again, size {n}"),
        );
    }
}

/// An operand whose length is not the plan's is `Error::Length` at the append, and nothing is appended.
#[test]
fn gpu_plans_refuse_operands_of_the_wrong_length_and_empty_shapes() {
    let device = device();
    let make = |len: u32| GpuVec::<u32>::zeroed_on(&device, len).unwrap();
    let (five, four, two, one) = (make(5), make(4), make(2), make(1));
    let buffer_of_four = device.create_typed_buffer("four", &[0u32; 4], GpuBufferAccess::ReadWrite);
    let mut sink = Sink::new();

    let reduce_plan = ReducePlan::<u32>::new(&device, Variant::V1, Op::Sum, 5).unwrap();
    assert_length(reduce_plan.append(&mut sink, &four, &one), 5, 4);
    assert_length(reduce_plan.append(&mut sink, &buffer_of_four, &one), 5, 4);
    assert_length(reduce_plan.append(&mut sink, &five, &two), 1, 2);
    let wrapping = SumWrappingPlan::new(&device, Variant::V1, 5).unwrap();
    assert_length(wrapping.append(&mut sink, &four, &one), 5, 4);
    assert_length(wrapping.append(&mut sink, &five, &two), 1, 2);
    let scan_plan = ScanPlan::new(&device, Variant::V1, 5, true).unwrap();
    assert_length(scan_plan.append(&mut sink, &four, &five), 5, 4);
    assert_length(scan_plan.append(&mut sink, &five, &four), 5, 4);
    let count_plan = CountPlan::new(&device, None, 5).unwrap();
    assert_length(count_plan.append(&mut sink, &four, &one), 5, 4);
    assert_length(count_plan.append(&mut sink, &five, &two), 1, 2);
    let compact_plan = CompactPlan::new(&device, None, 5).unwrap();
    assert_length(compact_plan.append(&mut sink, &four, &five, &one), 5, 4);
    assert_length(compact_plan.append(&mut sink, &five, &four, &one), 5, 4);
    assert_length(compact_plan.append(&mut sink, &five, &five, &two), 1, 2);
    assert_eq!(sink.len(), 0, "nothing is appended on an error");

    // A length of zero has no plan.
    assert!(matches!(
        ReducePlan::<u32>::new(&device, Variant::V1, Op::Min, 0),
        Err(Error::Empty)
    ));
    assert!(matches!(
        SumWrappingPlan::new(&device, Variant::V1, 0),
        Err(Error::Empty)
    ));
    assert!(matches!(
        ScanPlan::new(&device, Variant::V1, 0, false),
        Err(Error::Empty)
    ));
    assert!(matches!(
        CountPlan::new(&device, None, 0),
        Err(Error::Empty)
    ));
    assert!(matches!(
        CompactPlan::new(&device, None, 0),
        Err(Error::Empty)
    ));
}

/// The value of a reduction over nothing: the identity of the sum, nothing for `min` and `max`.
#[test]
fn op_identity_is_the_empty_sum_and_nothing_for_min_and_max() {
    assert_eq!(Op::Sum.identity::<u32>(), Some(0));
    assert_eq!(Op::Sum.identity::<i32>(), Some(0));
    assert_eq!(Op::Sum.identity::<f32>(), Some(0.0));
    for op in [Op::Min, Op::Max] {
        assert_eq!(op.identity::<u32>(), None);
        assert_eq!(op.identity::<f32>(), None);
    }
}

/// `result` is `Error::Length` with these two numbers.
fn assert_length(result: Result<(), Error>, expected: u32, actual: u32) {
    match result {
        Err(Error::Length {
            expected: got_expected,
            actual: got_actual,
        }) => assert_eq!((got_expected, got_actual), (expected, actual)),
        other => panic!("expected a Length error, got {other:?}"),
    }
}
