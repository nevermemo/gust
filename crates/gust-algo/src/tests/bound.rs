//! The float bound and the grouped host fold: the arithmetic the float tests lean on, checked on its own.

use crate::{
    bound::{self, C_CPU, C_GPU},
    schedule::Schedule,
};

#[test]
fn the_constants_are_the_per_operation_ones_of_semantics_section_4() {
    // Half an ULP of the result on the twin (u = 2^-24), one ULP on the GPU (2u = 2^-23).
    assert_eq!(C_CPU, 2.0_f64.powi(-24));
    assert_eq!(C_GPU, 2.0_f64.powi(-23));
}

#[test]
fn the_depth_is_the_sum_of_the_chains_less_one() {
    assert_eq!(bound::depth(&Schedule::new(1).unwrap()), 0);
    assert_eq!(bound::depth(&Schedule::new(100).unwrap()), 99);
    assert_eq!(bound::depth(&Schedule::new(257).unwrap()), 1 + 255);
    // 65 537: a 256-term pass, a 2-term pass (512 partials, two per lane), the 256-term terminal chain.
    assert_eq!(bound::depth(&Schedule::new(65_537).unwrap()), 255 + 1 + 255);
    // The plan's 1 020 is the ceiling of four 256-term chains; the real schedules are below it because the third pass
    // reads 32 (2^29) or 2 (2^25) values per lane.
    assert_eq!(
        bound::depth(&Schedule::new(1 << 29).unwrap()),
        255 + 255 + 31 + 255
    );
    assert_eq!(
        bound::depth(&Schedule::new(1 << 25).unwrap()),
        255 + 255 + 1 + 255
    );
    assert_eq!(
        bound::depth(&Schedule::new(100_000_000).unwrap()),
        255 + 255 + 5 + 255
    );
    for n in [1u32 << 20, 1 << 24, (1 << 24) + 1, 1 << 29, u32::MAX] {
        assert!(bound::depth(&Schedule::new(n).unwrap()) <= 1_020, "n = {n}");
    }
}

#[test]
fn the_bound_is_gamma_of_the_depth_times_the_absolute_sum() {
    let steps = 796.0 * C_GPU;
    let expected = 1.0e6 * steps / (1.0 - steps);
    assert_eq!(bound::f32_sum_bound(1 << 29, C_GPU, 1.0e6), expected);
    // About 796 * c * S, a little more by the second-order factor.
    assert!(expected > 796.0 * C_GPU * 1.0e6);
    assert!(expected < 797.0 * C_GPU * 1.0e6);
    // Far inside the order-free (n - 1) * c * S, which exceeds S at 1e8 (about 12 S).
    let order_free = 1.0e8 * C_GPU * 1.0;
    assert!(order_free > 11.0);
    assert!(bound::f32_sum_bound(100_000_000, C_GPU, 1.0) < 1.3e-4);
}

#[test]
fn the_bound_of_the_twin_is_half_the_bound_of_the_gpu_and_empty_has_none() {
    let gpu = bound::f32_sum_bound(1_000_000, C_GPU, 5.0e5);
    let cpu = bound::f32_sum_bound(1_000_000, C_CPU, 5.0e5);
    assert!(cpu < gpu && cpu > gpu / 2.0 * 0.999);
    assert_eq!(bound::f32_sum_bound(0, C_GPU, 1.0), 0.0);
    // One element: a single exact addition to the identity.
    assert_eq!(bound::f32_sum_bound(1, C_GPU, 1.0), 0.0);
}

#[test]
fn the_reference_error_is_the_f64_accumulation_bound() {
    assert_eq!(bound::f64_reference_error(1, 3.0), 0.0);
    assert_eq!(bound::f64_reference_error(0, 3.0), 0.0);
    assert_eq!(
        bound::f64_reference_error(1_000_001, 2.0),
        1_000_000.0 * 2.0_f64.powi(-53) * 2.0
    );
}

#[test]
fn the_grouped_fold_follows_the_lane_order_of_the_kernels() {
    assert_eq!(bound::sum_grouped(&[]).to_bits(), 0.0_f32.to_bits());
    assert_eq!(bound::sum_grouped(&[1.5, 2.0, 3.25]), 6.75);
    // 257 elements, 2^-24 at x[0] and x[256] and 1.0 at x[1]. A left fold adds 2^-24 + 1.0 = 1.0 (an exact tie
    // rounds to even) and then loses the second 2^-24 the same way. The kernels' lane order adds lane 0 first (x[0] +
    // x[256] = 2^-23, exact), lane 1 holds 1.0, and the terminal chain adds 2^-23 + 0 + 1.0 = 1.0 + 2^-23, exactly.
    let mut values = vec![0.0_f32; 257];
    values[0] = 1.0 / 16_777_216.0;
    values[256] = 1.0 / 16_777_216.0;
    values[1] = 1.0;
    let left_fold = values.iter().fold(0.0_f32, |sum, value| sum + value);
    assert_eq!(left_fold, 1.0);
    assert_eq!(bound::sum_grouped(&values), 1.0 + 1.0 / 8_388_608.0);
}

#[test]
fn the_grouped_fold_is_close_to_the_reference_on_a_long_input() {
    let values: Vec<f32> = (0..200_000)
        .map(|index| ((index % 97) as f32) / 97.0)
        .collect();
    let n = u32::try_from(values.len()).unwrap();
    let reference = bound::sum_f64(&values);
    let distance = (f64::from(bound::sum_grouped(&values)) - reference).abs();
    let limit = bound::f32_sum_bound(n, C_CPU, bound::abs_sum(&values))
        + bound::f64_reference_error(n, bound::abs_sum(&values));
    assert!(distance <= limit, "{distance:e} > {limit:e}");
}
