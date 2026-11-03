//! The reductions on the CPU twins of the kernels (`gust::cpu::CpuDevice`), against the independent references. A twin
//! result is not GPU evidence; `gpu::` runs the same cases on an adapter.

use super::common::{
    Backend, COMPACT_LENGTHS, CPU_C, LENGTHS, SCAN_LENGTHS, Shape, assert_f32_sum, compact_cases,
    deep_compact_cases, deep_scan_cases, empty_cases, f32_input, float_min_max_cases,
    float_sum_cases, gather_edge_cases, integer_cases, out_of_range_index_cases, scan_cases,
    tie_ladder, wrapping_cases,
};
use crate::{bound, cpu, reduce::Element};

struct Twin;

impl Backend for Twin {
    const C: f64 = CPU_C;

    fn sum<T: Element>(&self, values: &[T]) -> T {
        cpu::sum(values).expect("the twin runs")
    }

    fn min<T: Element>(&self, values: &[T]) -> Option<T> {
        cpu::min(values).expect("the twin runs")
    }

    fn max<T: Element>(&self, values: &[T]) -> Option<T> {
        cpu::max(values).expect("the twin runs")
    }

    fn sum_wrapping(&self, values: &[u32]) -> u32 {
        cpu::sum_wrapping(values).expect("the twin runs")
    }

    fn inclusive(&self, values: &[u32]) -> Vec<u32> {
        cpu::inclusive(values).expect("the twin runs")
    }

    fn exclusive(&self, values: &[u32]) -> Vec<u32> {
        cpu::exclusive(values).expect("the twin runs")
    }

    fn compact_count(&self, flags: &[u32]) -> u32 {
        cpu::compact_count(flags).expect("the twin runs")
    }

    fn compact_indices(&self, flags: &[u32]) -> (Vec<u32>, u32) {
        cpu::compact_indices(flags).expect("the twin runs")
    }

    fn gather_u32(&self, values: &[u32], indices: &[u32], count: u32) -> Vec<u32> {
        cpu::gather_u32(values, indices, count).expect("the twin runs")
    }

    fn gather_i32(&self, values: &[i32], indices: &[u32], count: u32) -> Vec<i32> {
        cpu::gather_i32(values, indices, count).expect("the twin runs")
    }

    fn gather_f32(&self, values: &[f32], indices: &[u32], count: u32) -> Vec<f32> {
        cpu::gather_f32(values, indices, count).expect("the twin runs")
    }
}

#[test]
fn twin_integer_sums_min_and_max_are_exact_at_every_boundary() {
    integer_cases(&Twin, &LENGTHS);
}

#[test]
fn twin_float_min_and_max_are_exact() {
    float_min_max_cases(&Twin, &LENGTHS);
}

#[test]
fn twin_empty_inputs() {
    empty_cases(&Twin);
}

#[test]
fn twin_float_sums_are_within_the_per_level_bound() {
    for n in [1, 255, 257, 65_537, 1_000_000] {
        for (shape, check) in float_sum_cases(&Twin, n) {
            eprintln!(
                "twin f32 sum n = {n} {shape:?}: distance {:e}, bound {:e}, ratio {:.4}",
                check.distance,
                check.bound,
                check.ratio()
            );
        }
    }
}

#[test]
fn twin_float_sum_equals_the_grouped_fold_bit_for_bit() {
    // The twin evaluates the kernels' statements in Rust's IEEE order, which is what `sum_grouped` is.
    for (shape, n) in [
        (Shape::Uniform, 300_001),
        (Shape::Alternating, 70_001),
        (Shape::Uniform, 100),
    ] {
        let values = f32_input(shape, n, 7);
        assert_eq!(
            cpu::sum(&values).unwrap().to_bits(),
            bound::sum_grouped(&values).to_bits(),
            "{shape:?}, n = {n}"
        );
    }
}

/// The input that makes every level lose its small terms: the distance reaches two thirds of the CPU bound, so a bound
/// of half that size (the mutation of the slice) fails here and nowhere else in the unit run.
#[test]
fn twin_float_sum_of_the_tie_ladder_loses_what_the_bound_allows() {
    let values = tie_ladder(1 << 24);
    let actual = cpu::sum(&values).unwrap();
    assert_eq!(
        actual, 256.0,
        "every addition of a tiny term to 1.0 is an exact tie that rounds to even"
    );
    let check = assert_f32_sum("tie ladder", &values, actual, CPU_C);
    eprintln!(
        "twin tie ladder: distance {:e}, bound {:e}, ratio {:.4}",
        check.distance,
        check.bound,
        check.ratio()
    );
    assert!(
        check.ratio() > 0.6,
        "the shape must approach the bound, ratio {}",
        check.ratio()
    );
    assert_eq!(bound::sum_grouped(&values), 256.0);
}

#[test]
fn twin_runs_the_deepest_schedule() {
    // 2^24 + 1 elements: three passes and the terminal kernel (four dispatches).
    let n = (1usize << 24) + 1;
    let ones = vec![1u32; n];
    assert_eq!(cpu::sum(&ones).unwrap() as usize, n);
    let mut values = vec![5_000u32; n];
    values[n - 1] = 7;
    values[12_345_678] = 9_000;
    assert_eq!(cpu::min(&values).unwrap(), Some(7));
    assert_eq!(cpu::max(&values).unwrap(), Some(9_000));
    // A float sum over four dispatches stays within the bound of the deepest schedule.
    let floats = f32_input(Shape::Uniform, n, 0xF00D);
    let check = assert_f32_sum(
        "2^24 + 1 uniform",
        &floats,
        cpu::sum(&floats).unwrap(),
        CPU_C,
    );
    eprintln!("twin f32 sum n = {n}: ratio {:.4}", check.ratio());
}

/// Mutation (a) of the slice, as a test: a pass whose last partial is dropped, or a batch whose passes run out of
/// order, changes an integer sum. This pins that the sum of a length with a partly filled last block sees every
/// element once.
#[test]
fn twin_sum_of_a_partly_filled_last_block_counts_each_element_once() {
    for n in [65_537usize, 65_536 + 255, 131_071, 200_000] {
        let values: Vec<u32> = (0..n as u32).map(|index| index % 1_000 + 1).collect();
        let expected: u64 = values.iter().map(|value| u64::from(*value)).sum();
        assert_eq!(u64::from(cpu::sum(&values).unwrap()), expected, "n = {n}");
    }
}

#[test]
fn twin_wrapping_sums_are_defined_modulo_2_32_for_every_input() {
    wrapping_cases(&Twin, &LENGTHS);
}

#[test]
fn twin_scans_are_exact_at_every_boundary() {
    scan_cases(&Twin, &SCAN_LENGTHS);
}

/// 2^24 elements are five dispatches and 2^24 + 1 are seven (a third level of totals).
#[test]
fn twin_runs_the_deepest_scans() {
    deep_scan_cases(&Twin);
}

/// The exclusive scan is the inclusive one shifted by the element, inside the final add-back: the two differ on every
/// element that is not zero, so an exclusive scan that returned the inclusive result fails here.
#[test]
fn twin_exclusive_is_not_inclusive() {
    let values = [3u32, 1, 4, 1, 5];
    assert_eq!(cpu::inclusive(&values).unwrap(), vec![3, 4, 8, 9, 14]);
    assert_eq!(cpu::exclusive(&values).unwrap(), vec![0, 3, 4, 8, 9]);
    assert!(cpu::inclusive(&[]).unwrap().is_empty());
    assert!(cpu::exclusive(&[]).unwrap().is_empty());
}

#[test]
fn twin_compaction_is_exact_at_every_boundary() {
    compact_cases(&Twin, &COMPACT_LENGTHS);
}

/// `2^24 + 1` flags: a mask launch of 257 workgroups, a seven-dispatch scan, a scatter and the gathers.
#[test]
fn twin_runs_the_deepest_compaction() {
    deep_compact_cases(&Twin);
}

#[test]
fn twin_a_gather_guards_the_index_bound_and_the_count() {
    out_of_range_index_cases(&Twin);
    gather_edge_cases(&Twin);
}

/// The flag rule on the twin: any non-zero flag is set, so `2`, `0x8000_0000` and `u32::MAX` count as one each, and the
/// indices are the same as for flags of 0 and 1 (a mask that kept the flag would give a count of 2 + 2^31 + ...).
#[test]
fn twin_non_zero_flags_count_as_set() {
    let flags = [0u32, 2, 0, 0x8000_0000, u32::MAX, 1, 0];
    assert_eq!(cpu::compact_count(&flags).unwrap(), 4);
    let (indices, count) = cpu::compact_indices(&flags).unwrap();
    assert_eq!(count, 4);
    assert_eq!(indices, vec![1, 3, 4, 5, 0, 0, 0]);
}

/// Nothing in `compact` reads a value back to the host: the T13 rule, enforced on the source (a readback has to name
/// `to_vec`, `read` or an async variant of them).
#[test]
fn compact_never_reads_back() {
    let source = include_str!("../compact.rs").replace("\r\n", "\n");
    let code: String = source
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    for readback in [".to_vec", "to_vec_async", ".read(", "map_async", "Readback"] {
        assert!(!code.contains(readback), "compact.rs mentions `{readback}`");
    }
}
