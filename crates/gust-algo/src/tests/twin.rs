//! The reductions on the CPU twins of the kernels (`gust::cpu::CpuDevice`), against the independent references. A twin
//! result is not GPU evidence; `gpu::` runs the same cases on an adapter.

use super::common::{
    Backend, CPU_C, LENGTHS, Shape, assert_f32_sum, empty_cases, f32_input, float_min_max_cases,
    float_sum_cases, integer_cases, tie_ladder,
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
