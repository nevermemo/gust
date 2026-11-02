//! The pass schedule: pure arithmetic, pinned at every boundary the plan names (section 5.2).

use crate::schedule::{BLOCK, CHAIN, LANES, Pass, Schedule};

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
