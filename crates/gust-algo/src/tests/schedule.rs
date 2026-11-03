//! The pass schedule: pure arithmetic, pinned at every boundary the plan names (section 5.2).

use crate::schedule::{BLOCK, CHAIN, CHUNK, LANES, Pass, ScanSchedule, Schedule};

fn shape(n: u32) -> (Vec<(u32, u32)>, u32) {
    let schedule = Schedule::new(n).expect("a non-empty input");
    (
        schedule
            .passes()
            .iter()
            .map(|pass| (pass.input_len, pass.partials()))
            .collect(),
        schedule.final_len(),
    )
}

#[test]
fn constants_are_the_documented_ones() {
    assert_eq!((LANES, BLOCK, CHAIN), (256, 65_536, 256));
}

#[test]
fn an_empty_input_has_no_schedule() {
    assert!(Schedule::new(0).is_none());
}

#[test]
fn up_to_256_elements_are_one_terminal_dispatch() {
    for n in [1, 2, 255, 256] {
        let (passes, final_len) = shape(n);
        assert!(passes.is_empty(), "n = {n}");
        assert_eq!(final_len, n);
        assert_eq!(Schedule::new(n).unwrap().dispatches(), 1);
    }
}

#[test]
fn one_block_is_one_pass_of_256_partials() {
    for n in [257, 65_535, 65_536] {
        assert_eq!(shape(n), (vec![(n, 256)], 256), "n = {n}");
        assert_eq!(Schedule::new(n).unwrap().dispatches(), 2);
    }
}

#[test]
fn the_second_block_adds_a_pass() {
    assert_eq!(shape(65_537), (vec![(65_537, 512), (512, 256)], 256));
    assert_eq!(Schedule::new(65_537).unwrap().dispatches(), 3);
}

#[test]
fn the_range_of_a_default_device_is_four_dispatches() {
    // 2^25 elements: the 128 MiB binding of the portable limits.
    assert_eq!(
        shape(1 << 25),
        (vec![(1 << 25, 1 << 17), (1 << 17, 512), (512, 256)], 256)
    );
    assert_eq!(Schedule::new(1 << 25).unwrap().dispatches(), 4);
}

#[test]
fn the_range_of_a_two_gib_binding_is_four_dispatches() {
    // 2^29 elements: 8 192 workgroups in the first pass, 2^21 partials.
    let schedule = Schedule::new(1 << 29).unwrap();
    assert_eq!(schedule.passes()[0].workgroups, 8_192);
    assert_eq!(schedule.passes()[0].partials(), 1 << 21);
    assert_eq!(
        shape(1 << 29),
        (
            vec![(1 << 29, 1 << 21), (1 << 21, 8_192), (8_192, 256)],
            256
        )
    );
    assert_eq!(schedule.dispatches(), 4);
}

#[test]
fn the_u32_range_needs_at_most_four_dispatches() {
    for n in [1u32 << 24, (1 << 24) + 1, 1 << 30, u32::MAX] {
        let schedule = Schedule::new(n).unwrap();
        assert!(schedule.dispatches() <= 4, "n = {n}");
        // The last pass leaves at most the terminal kernel's 256 values.
        assert!(schedule.final_len() <= LANES);
    }
    // Only the very top of the u32 range needs 65 536 workgroups in the first pass; a device refuses it by its limit.
    assert_eq!(
        Schedule::new(u32::MAX).unwrap().passes()[0].workgroups,
        65_536
    );
    assert_eq!(
        Schedule::new(65_535 * BLOCK).unwrap().passes()[0].workgroups,
        65_535
    );
}

#[test]
fn chains_are_at_most_256_terms() {
    let schedule = Schedule::new(1 << 29).unwrap();
    // The third pass reads 8 192 values with 256 lanes: 32 each.
    assert_eq!(schedule.chains(), vec![256, 256, 32, 256]);
    // A short input: lane 0 reads indices 0, 256, ... below n.
    assert_eq!(
        Pass {
            input_len: 257,
            workgroups: 1
        }
        .longest_chain(),
        2
    );
    assert_eq!(
        Pass {
            input_len: 513,
            workgroups: 1
        }
        .longest_chain(),
        3
    );
    assert_eq!(Schedule::new(257).unwrap().chains(), vec![2, 256]);
    assert_eq!(Schedule::new(100).unwrap().chains(), vec![100]);
}

fn scan_shape(n: u32) -> (Vec<u32>, usize, u32) {
    let schedule = ScanSchedule::new(n).expect("a non-empty input");
    (
        schedule.sums().to_vec(),
        schedule.dispatches(),
        schedule.workgroups(),
    )
}

#[test]
fn an_empty_scan_has_no_schedule() {
    assert!(ScanSchedule::new(0).is_none());
}

#[test]
fn up_to_65_536_elements_scan_in_three_dispatches() {
    // Level 0, the top scan over the totals, the final add-back.
    assert_eq!(scan_shape(1), (vec![1], 3, 1));
    assert_eq!(scan_shape(256), (vec![1], 3, 1));
    assert_eq!(scan_shape(257), (vec![2], 3, 1));
    assert_eq!(scan_shape(65_536), (vec![256], 3, 1));
}

#[test]
fn up_to_2_pow_24_elements_scan_in_five_dispatches() {
    // The third level appears with the 257th total; level 3 is skipped while level 2 holds at most 256.
    assert_eq!(scan_shape(65_537), (vec![257, 2], 5, 2));
    assert_eq!(scan_shape(1 << 24), (vec![65_536, 256], 5, 256));
}

#[test]
fn past_2_pow_24_elements_scan_in_seven_dispatches() {
    assert_eq!(scan_shape((1 << 24) + 1), (vec![65_537, 257, 2], 7, 257));
    // 2^25 elements: the 128 MiB binding of the portable limits.
    assert_eq!(scan_shape(1 << 25), (vec![131_072, 512, 2], 7, 512));
    // 2^29 elements: the 2 GiB binding the adapters offer; level 0 has 2^21 invocations, 8 192 workgroups.
    assert_eq!(scan_shape(1 << 29), (vec![1 << 21, 8_192, 32], 7, 8_192));
}

#[test]
fn the_u32_range_needs_at_most_seven_dispatches() {
    assert_eq!(CHUNK, 256);
    for n in [1u32 << 30, (1 << 31) + 1, u32::MAX] {
        assert_eq!(ScanSchedule::new(n).unwrap().dispatches(), 7, "n = {n}");
    }
    // The very top of the range: 2^24 level-0 invocations are 65 536 workgroups, one more than a device allows by
    // default; the device refuses it by its limit (see the `gpu::` test of the limit).
    let top = ScanSchedule::new(u32::MAX).unwrap();
    assert_eq!(top.sums(), [1 << 24, 1 << 16, 256]);
    assert_eq!(top.workgroups(), 65_536);
    // The last level never exceeds the top scan's 256 values.
    for n in [
        1u32,
        256,
        257,
        65_536,
        65_537,
        1 << 24,
        (1 << 24) + 1,
        u32::MAX,
    ] {
        let schedule = ScanSchedule::new(n).unwrap();
        assert!(schedule.sums().last().copied().unwrap() <= CHUNK, "n = {n}");
        assert_eq!(schedule.dispatches(), 2 * schedule.sums().len() + 1);
    }
}
