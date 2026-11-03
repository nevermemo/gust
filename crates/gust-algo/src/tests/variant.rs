//! The v1 schedules (pure arithmetic) and the shipping record of slice m7.r4.

use crate::{
    bound,
    schedule::{
        BlockSchedule, SPAN, SPAN_TILES, Schedule, TREE_DEPTH, TileSchedule, V1_BLOCK, V1_CHAIN,
    },
    variant::{OPERATIONS, Operation, Variant, shipped},
};

fn block_shape(n: u32) -> Vec<(u32, u32)> {
    BlockSchedule::new(n)
        .expect("a non-empty input")
        .passes()
        .iter()
        .map(|pass| (pass.input_len, pass.workgroups))
        .collect()
}

#[test]
fn the_v1_constants_are_the_documented_ones() {
    assert_eq!((V1_BLOCK, V1_CHAIN, TREE_DEPTH), (8_192, 32, 8));
    assert_eq!((SPAN, SPAN_TILES), (4_096, 16));
}

#[test]
fn an_empty_input_has_no_v1_schedule() {
    assert!(BlockSchedule::new(0).is_none());
    assert!(TileSchedule::new(0).is_none());
}

#[test]
fn a_block_reduce_ends_with_one_workgroup_that_writes_the_result() {
    assert_eq!(block_shape(1), [(1, 1)]);
    assert_eq!(block_shape(8_192), [(8_192, 1)]);
    assert_eq!(block_shape(8_193), [(8_193, 2), (2, 1)]);
    assert_eq!(block_shape(1 << 26), [(1 << 26, 8_192), (8_192, 1)]);
    assert_eq!(
        block_shape((1 << 26) + 1),
        [((1 << 26) + 1, 8_193), (8_193, 2), (2, 1)]
    );
    assert_eq!(
        block_shape(1 << 29),
        [(1 << 29, 65_536), (65_536, 8), (8, 1)]
    );
    // 1e6, 1e7 and 1e8, the measured sizes: two, two and two passes' worth of workgroups.
    assert_eq!(block_shape(1_000_000), [(1_000_000, 123), (123, 1)]);
    assert_eq!(block_shape(10_000_000), [(10_000_000, 1_221), (1_221, 1)]);
    assert_eq!(
        block_shape(100_000_000),
        [(100_000_000, 12_208), (12_208, 2), (2, 1)]
    );
    assert_eq!(BlockSchedule::new(1).unwrap().dispatches(), 1);
    assert_eq!(BlockSchedule::new(100_000_000).unwrap().dispatches(), 3);
}

#[test]
fn a_tile_scan_takes_one_dispatch_for_a_span_and_two_k_plus_one_above() {
    let shape = |n: u32| {
        let schedule = TileSchedule::new(n).expect("a non-empty input");
        (
            schedule.sums().to_vec(),
            schedule.dispatches(),
            schedule.workgroups(),
        )
    };
    assert_eq!(shape(1), (vec![1], 1, 1));
    assert_eq!(shape(4_096), (vec![1], 1, 1));
    assert_eq!(shape(4_097), (vec![2], 3, 2));
    assert_eq!(shape(1 << 24), (vec![4_096], 3, 4_096));
    assert_eq!(shape((1 << 24) + 1), (vec![4_097, 2], 5, 4_097));
    assert_eq!(shape(1_000_000), (vec![245], 3, 245));
    assert_eq!(shape(100_000_000), (vec![24_415, 6], 5, 24_415));
    // 65 535 spans are the most a 1D launch of 65 535 workgroups takes: 268 431 360 elements.
    assert_eq!(shape(65_535 * 4_096).2, 65_535);
    assert_eq!(shape(65_535 * 4_096 + 1).2, 65_536);
}

/// The v1 bound is the same form with fewer additions on the longest path: at most 39 per level (31 for a lane's 32
/// terms, 8 for the tree) and 86 for the three passes of 2^28 elements, against 796 for v0 at 2^29. It is not below
/// v0's for a tiny input: v1's depth is 8 from one element up, v0's is `n - 1` (at most 8 for `n <= 9`), so v1 is
/// larger for `n` of 2 to 8, equal at 9 and smaller from 10 on.
#[test]
fn the_v1_depth_is_below_the_v0_depth_from_ten_elements_up() {
    let v0 = |n: u32| bound::depth(&Schedule::new(n).unwrap());
    let v1 = |n: u32| bound::depth_v1(&BlockSchedule::new(n).unwrap());
    assert_eq!(v1(1), 8);
    assert_eq!(v1(256), 8);
    assert_eq!(v1(8_192), 31 + 8);
    assert_eq!(v1(1 << 28), 39 + 39 + 8);
    assert_eq!(v0(1 << 29), 796);
    assert_eq!(v0(1), 0);
    for n in 2..=8 {
        assert_eq!(v0(n), u64::from(n - 1));
        assert!(v1(n) > v0(n), "n = {n}: {} against {}", v1(n), v0(n));
    }
    assert_eq!(v1(9), v0(9));
    for n in [
        10,
        100,
        256,
        257,
        1_000,
        65_537,
        1_000_000,
        1 << 24,
        1 << 28,
    ] {
        assert!(v1(n) < v0(n), "n = {n}: {} against {}", v1(n), v0(n));
        assert!(v1(n) <= 3 * (31 + 8), "n = {n}");
    }
    // The bound is the form gamma_K * S with the K above: tighter than v0's for a large input.
    let (n, c, s) = (100_000_000, bound::C_GPU, 1.0);
    assert!(bound::f32_sum_bound_v1(n, c, s) < bound::f32_sum_bound(n, c, s) / 4.0);
    assert_eq!(bound::f32_sum_bound_v1(0, c, s), 0.0);
}

/// The record of the measurement of m7.r4 (plan section 5.5: v1 ships per operation only where it is at least 10 %
/// faster than v0 at 1e6, 1e7 and 1e8 elements on the RTX 5090; the log of the slice, `plans/m7/probe/r4/`, has the
/// table). `WORST` is the lowest median ratio v0 / v1 over the sizes and element types of each operation, in
/// thousandths (the medians of the per-run medians; single runs went lower, see the module docs of [`crate::variant`]).
/// `shipped` must be v1 exactly where that ratio is at least 1.10. It is a two-place tripwire, not a proof: it fails
/// when only one of `WORST` and `shipped` is edited, and a changed `shipped` needs the new measurement written here
/// beside it. The Radeon 890M half of the measurement is owed.
#[test]
fn the_shipped_variants_are_the_measured_ones() {
    const WORST: [(Operation, u32); 6] = [
        (Operation::Sum, 1_188),
        (Operation::Min, 1_180),
        (Operation::Max, 1_158),
        (Operation::SumWrapping, 1_222),
        (Operation::Inclusive, 1_516),
        (Operation::Exclusive, 1_565),
    ];
    assert_eq!(WORST.len(), OPERATIONS.len());
    for (operation, worst) in WORST {
        let expected = if worst >= 1_100 {
            Variant::V1
        } else {
            Variant::V0
        };
        assert_eq!(
            shipped(operation),
            expected,
            "{operation:?} at {worst} thousandths"
        );
    }
    for operation in OPERATIONS {
        assert!(
            WORST.iter().any(|(measured, _)| *measured == operation),
            "{operation:?} has no measurement"
        );
    }
}
