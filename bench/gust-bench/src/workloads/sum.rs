//! W7: the sum of `n` values, `u32` (`W7-u32`) and `f32` (`W7-f32`), 2^20 to 2^27 values. One output word.
//!
//! GUST runs `gust_algo::reduce` (the public sum: v1 block kernels with a tree in workgroup memory, built with the
//! `algo` feature, which `gust-bench-pre` does not have: gust-algo ships no precompiled artifacts); the hand-WGSL
//! backends run a workgroup-tree reduction of their own (workgroup size 256, two dispatches, a grid that scales with
//! `n`, see [`hand_blocks`]); the CPU backends sum on the host. The kernels differ, so no GUST/hand pairing of W7 is a
//! same-kernel comparison (`Workload::same_kernel`), and the kill rule and the report's overhead rows leave W7 out
//! (listed, not silent: `group::kill_rule`).
//!
//! # The `u32` check
//!
//! Exact, and the sum is the wrapping one: `gust_algo::reduce::sum_wrapping` is defined modulo 2^32 for every input
//! (the plain `sum` is exact only while no partial sum overflows), and every other backend adds with `wrapping_add` (WGSL
//! integer addition wraps by definition). The inputs are 32-bit words from a hash, so the true sum passes 2^32 by far and
//! a lost or doubled element changes the word (a test pins both).
//!
//! # The `f32` bounds (derived here, plan 2.1)
//!
//! Every f32 addition is modelled as `fl(a + b) = (a + b)(1 + delta) + eta`, `|delta| <= u`, `|eta| <= 2^-126` (an
//! absolute error only where a flushing device turns a subnormal result to zero; IEEE addition itself never needs it).
//! `S = sum |x_i|` is the sum of absolute values; the exact sum is `s`.
//!
//! The unit roundoff is the backend's ([`Rounding`](super::Rounding)): `u = 2^-24` for the CPU backends (host IEEE round
//! to nearest) and `c = 2^-23` for every GPU backend (the GPU may round either neighbour, within 1 ULP,
//! `docs/SEMANTICS.md` section 4; `gust_algo::bound::C_GPU`).
//!
//! - *GUST's tree* (`gust-gpu`, `gust-gpu-spirv`): the library's own bound, `|s^ - s| <= gamma_K S` with
//!   `gamma_K = K c / (1 - K c)` and `c = 2^-23`, `K` the most additions any value passes through, for the schedule that
//!   ran: `gust_algo::bound::f32_sum_bound_v1` for the v1 kernels the public `sum` runs, `f32_sum_bound` for the v0
//!   fallback (see `gust_gpu::sum_bound_from` for how the schedule is told, and what is assumed).
//! - *The hand-WGSL tree* (`wgsl-prebuilt`, `wgsl-percall`): its summation order is known, so it has its own bound
//!   (`Backend::sum_tree_bound`), `gamma_K S` with `c = 2^-23` and `K = chain_1 + 8 + chain_2 + 8` ([`hand_depth`]): `chain_1`
//!   is the number of elements one lane of the first pass adds sequentially (`ceil(n / (blocks * 256))`), 8 the rounds of the
//!   workgroup tree (workgroup size 256), `chain_2` the partials one lane of the second pass adds (`ceil(blocks / 256)`) and
//!   8 the rounds of its tree. The first add of a chain, into the zero accumulator, is counted although it is exact on IEEE
//!   hardware (the conservative reading of "either neighbour"). Reassociation a compiler may do (WGSL section 15.7.5) does
//!   not make a chain deeper. This is tighter than the plan's serial worst case, which still bounds this backend (a
//!   test pins that the tree bound is inside it).
//! - *Every other backend* (`serial`, `rayon-1`, `rayon-N`: host, `u = 2^-24`): the serial worst case, which bounds every
//!   summation order (a left-to-right loop and rayon's unknown splits alike): `|s^ - s| <= ((1+u)^{n-1} - 1) S`, the product
//!   form of Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., section 4.2 (every value is multiplied by
//!   at most `n - 1` factors `1 + delta_j`, in whatever order the adds are grouped; `|prod(1+delta_j) - 1| <= (1+u)^k - 1`
//!   for every `k`). It replaces `gamma_{n-1}` everywhere: it is never larger than `gamma_{n-1}`, exists for every `n`
//!   (`gamma_k` needs `k u < 1`, that is `n <= 2^24`) and is monotone in `n`. The flush allowance
//!   `(n - 1) 2^-126 (1 + factor)` is added for a device that flushes. **It is vacuous from `n = 2^24`**: there the
//!   factor passes 1, so the bound exceeds `S >= |s|` and a result of `0.0` passes (a test pins the size). From `2^22`
//!   it is weak (over a quarter of `S`). The order of `serial` is known and sequential, of `rayon` unknown, so no tighter
//!   bound is derived for them (a bound the order does not justify would be a guess); the cells above `2^23` rest on the
//!   same code passing at the small sizes and on W7-u32 being exact (see `docs/BENCHMARKS.md`).
//!
//! The reference is a sequential sum in f64, whose own error is at most `(n - 1) 2^-53 S`
//! (`gust_algo::bound::f64_reference_error`, a test pins the two to the same formula; the rigorous recursive-sum bound is
//! `gamma64_{n-1} S`, and the first-order form is no looser here because every partial f64 sum is exact: the inputs are
//! multiples of `2^-23`, `|partial| < 2^28`, so at most 51 bits). `S` is computed in f64 as well,
//! with relative error at most `gamma64_{n-1}`, so both bounds use the upper value `S_f64 / (1 - gamma64_{n-1})`. The
//! check is therefore `|s^ - s_f64| <= bound(S_up) + (n - 1) 2^-53 S_up`.
//!
//! A backend that fails the bound fails the cell; the bound is never widened to pass. The domain: finite inputs, no
//! partial sum overflowing (every input is in `[-0.5, 1.5)`, so every partial sum is below 2^28 in magnitude), normal or
//! zero inputs (the library's domain; the inputs are multiples of 2^-23 with magnitude below 2).

use super::{
    Rounding, mix,
    polynomial::{self, gamma},
};

/// Unit roundoff of f64, `2^-53`.
pub const U64: f64 = polynomial::U64;

/// Elements one workgroup of the hand tree's first pass takes (256 lanes, 8 each): the grid is `ceil(n / 2048)`
/// workgroups, capped by the device's `max_compute_workgroups_per_dimension`, a standard grid-stride reduction.
pub const HAND_ELEMENTS_PER_GROUP: usize = 2048;

/// Workgroups of the hand tree's first pass for `n` values on a device that allows `max_groups` per dimension.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub fn hand_blocks(n: usize, max_groups: u32) -> u32 {
    let wanted = n.div_ceil(HAND_ELEMENTS_PER_GROUP).max(1);
    u32::try_from(wanted)
        .map_or(max_groups, |w| w.min(max_groups))
        .max(1)
}

/// The most additions any value passes through in the hand tree with `blocks` workgroups in its first pass (see the
/// module notes): `chain_1 + 8 + chain_2 + 8`.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub fn hand_depth(n: usize, blocks: u32) -> usize {
    let lanes = blocks as usize * 256;
    n.div_ceil(lanes) + 8 + (blocks as usize).div_ceil(256) + 8
}

/// The worst-case error of the hand tree: `gamma_K S` with `c = 2^-23` (`K` from [`hand_depth`]).
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub fn hand_tree_bound(n: usize, blocks: u32, abs_sum: f64) -> f64 {
    gamma(hand_depth(n, blocks), Rounding::Either.unit()) * abs_sum
}

/// The `u32` addends: 32-bit words from the hash, so the true sum is far above 2^32 and the wrapping is exercised.
pub fn u32_values(n: usize) -> Vec<u32> {
    (0..n as u64)
        .map(|i| (mix(i ^ 0x5bd1_e995_0000_0000) >> 32) as u32)
        .collect()
}

/// The `f32` addends: 24-bit integers over `2^23` minus one half, so every value is exact, lies in `[-0.5, 1.5)` and
/// carries a full mantissa (the sums round, unlike a sum of short fractions that would stay exact).
pub fn f32_values(n: usize) -> Vec<f32> {
    (0..n as u64)
        .map(|i| {
            let r24 = (mix(i ^ 0x2545_f491_0000_0000) >> 40) as u32;
            (r24 as f32 - 4_194_304.0) / 8_388_608.0
        })
        .collect()
}

/// The `u32` sum modulo 2^32.
pub fn wrapping_sum(values: &[u32]) -> u32 {
    values.iter().fold(0, |sum, v| sum.wrapping_add(*v))
}

/// The sequential f32 sum, left to right: the `serial` backend's computation.
pub fn serial_f32(values: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for v in values {
        sum += *v;
    }
    sum
}

/// The sequential sum and the sum of absolute values, both in f64 (the reference of a case).
pub fn f64_sums(values: &[f32]) -> (f64, f64) {
    let (mut sum, mut abs) = (0.0f64, 0.0f64);
    for v in values {
        let v = f64::from(*v);
        sum += v;
        abs += v.abs();
    }
    (sum, abs)
}

/// An upper bound on the true `S = sum |x_i|` from its f64 computation `abs_f64` (relative error at most `gamma64_{n-1}`).
pub fn abs_sum_upper(n: usize, abs_f64: f64) -> f64 {
    abs_f64 / (1.0 - gamma(n.saturating_sub(1), U64))
}

/// The most the f64 reference can be off: `(n - 1) 2^-53 S` (the formula of `gust_algo::bound::f64_reference_error`).
pub fn reference_error(n: usize, abs_sum: f64) -> f64 {
    n.saturating_sub(1) as f64 * U64 * abs_sum
}

/// The factor of the serial worst case, the product form `(1 + u)^{n-1} - 1` for every `n` (see the module notes), `u`
/// the unit roundoff of `rounding`. `0` for `n < 2`.
pub fn serial_factor(n: usize, rounding: Rounding) -> f64 {
    if n < 2 {
        return 0.0;
    }
    ((n - 1) as f64 * rounding.unit().ln_1p()).exp_m1()
}

/// The serial worst case `((1+u)^{n-1} - 1) S` plus the flush allowance, for the f32 sum of `n` values whose absolute
/// values add up to at most `abs_sum`. It bounds every summation order, so it is the bound of every backend without a
/// bound of its own for the order it runs.
pub fn serial_bound(n: usize, abs_sum: f64, rounding: Rounding) -> f64 {
    let factor = serial_factor(n, rounding);
    factor * abs_sum + n.saturating_sub(1) as f64 * f64::from(f32::MIN_POSITIVE) * (1.0 + factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workloads::{
        Judging, SUM_SIZES, Workload, check, check_with, gen_inputs, reference,
    };

    /// A pairwise (tree) f32 sum, one of the orders the serial bound covers.
    fn pairwise(values: &[f32]) -> f32 {
        match values.len() {
            0 => 0.0,
            1 => values[0],
            len => {
                let (left, right) = values.split_at(len / 2);
                pairwise(left) + pairwise(right)
            }
        }
    }

    #[test]
    fn the_factor_is_the_product_form_for_every_n_and_never_above_gamma() {
        for rounding in [Rounding::Nearest, Rounding::Either] {
            let u = rounding.unit();
            assert_eq!(serial_factor(0, rounding), 0.0);
            assert_eq!(serial_factor(1, rounding), 0.0);
            // Monotone in n, finite and positive for every size the workload has (gamma is not: it blows up near k u = 1).
            let factors: Vec<f64> = SUM_SIZES
                .iter()
                .map(|n| serial_factor(*n, rounding))
                .collect();
            assert!(factors.windows(2).all(|w| 0.0 < w[0] && w[0] < w[1]));
            assert!(factors.iter().all(|f| f.is_finite()));
            // Where gamma exists the product form is the smaller one.
            for n in [2usize, 1000, 1 << 16, 1 << 20] {
                let k = n - 1;
                if (k as f64) * u < 1.0 {
                    assert!(
                        serial_factor(n, rounding) <= gamma(k, u) * (1.0 + 1e-12),
                        "n = {n}"
                    );
                }
            }
        }
        // u = 2^-24 at n = 2^20: exp(2^-4) - 1 = 0.0645 (gamma would be 0.0667).
        assert!((serial_factor(1 << 20, Rounding::Nearest) - 0.0645).abs() < 1e-4);
        // The size where the host bound stops saying anything: the factor passes 1 at 2^24 (so `bound > S >= |s|`
        // and a result of 0.0 passes), is still a weak 0.65 at 2^23 and about 0.28 at 2^22. See docs/BENCHMARKS.md.
        assert!(serial_factor(1 << 24, Rounding::Nearest) > 1.0);
        assert!(serial_factor(1 << 23, Rounding::Nearest) < 1.0);
        assert!((serial_factor(1 << 23, Rounding::Nearest) - 0.65).abs() < 0.01);
        assert!((serial_factor(1 << 22, Rounding::Nearest) - 0.28).abs() < 0.01);
    }

    /// The vacuity claimed in the docs, on real data: from 2^24 values the host bound exceeds `|s|`, so a result of zero
    /// would pass; at 2^20 it does not. (A pure computation of the bound on the stated size: no input of that size is built;
    /// `|s| <= S` holds for every input.)
    #[test]
    fn the_host_bound_is_vacuous_from_2_pow_24_and_meaningful_at_2_pow_20() {
        let small = gen_inputs(Workload::SumF32, 1 << 20);
        let reference = reference(Workload::SumF32, &small);
        assert!(reference.exact[0].abs() > reference.bound[0] + reference.ref_error);
        assert!(check(Workload::SumF32, &small, &reference, &[0.0]).is_err());
        // At 2^24 the bound is above S for any S, so a zero result is within it whatever the data.
        let s = 1.0f64;
        assert!(serial_bound(1 << 24, s, Rounding::Nearest) > s);
        assert!(serial_bound(1 << 22, s, Rounding::Nearest) < s);
    }

    #[test]
    fn the_hand_grid_scales_with_n_and_its_bound_is_inside_the_serial_one() {
        assert_eq!(hand_blocks(1 << 20, 65535), 512);
        assert_eq!(hand_blocks(1 << 24, 65535), 8192);
        // Capped by the device limit; never zero.
        assert_eq!(hand_blocks(1 << 27, 65535), 65535);
        assert_eq!(hand_blocks(1 << 27, 1024), 1024);
        assert_eq!(hand_blocks(1, 65535), 1);
        // The depth: 2^20 values on 512 workgroups: a lane adds 8, 8 rounds, 2 partials, 8 rounds.
        assert_eq!(hand_depth(1 << 20, 512), 8 + 8 + 2 + 8);
        assert_eq!(hand_depth(1 << 27, 65535), 9 + 8 + 256 + 8);
        for n in SUM_SIZES {
            let blocks = hand_blocks(n, 65535);
            let depth = hand_depth(n, blocks);
            // gamma_K S with c = 2^-23: tiny against S, and far inside the serial worst case of the same unit.
            let bound = hand_tree_bound(n, blocks, 1.0);
            assert!(bound > 0.0 && bound < 1e-3, "n = {n}: {bound}");
            assert!(bound < serial_bound(n, 1.0, Rounding::Either), "n = {n}");
            assert!(depth < n);
        }
    }

    #[test]
    fn the_reference_error_is_the_formula_of_the_library() {
        assert_eq!(reference_error(1, 7.0), 0.0);
        assert_eq!(reference_error(1001, 3.0), 1000.0 * U64 * 3.0);
        #[cfg(feature = "algo")]
        for (n, s) in [(2usize, 1.0f64), (4096, 2048.0), (1 << 24, 1e7)] {
            assert_eq!(
                reference_error(n, s),
                gust_algo::bound::f64_reference_error(n as u32, s)
            );
        }
    }

    #[test]
    fn the_domain_the_derivation_needs_holds_for_the_inputs() {
        let x = f32_values(1 << 16);
        assert!(x.iter().all(|v| v.is_finite() && (-0.5..1.5).contains(v)));
        // Normal or zero, and a multiple of 2^-23: exact in f32 and in f64.
        assert!(x.iter().all(|v| *v == 0.0 || v.is_normal()));
        assert!(
            x.iter()
                .all(|v| (f64::from(*v) * 8_388_608.0).fract() == 0.0)
        );
        // 2^27 values below 2 in magnitude keep every partial sum below 2^28 (the f32 range reaches 3e38).
        // The values carry full mantissas: the f32 sums round (otherwise the bound would never be exercised).
        let sum = serial_f32(&x);
        assert_ne!(
            f64::from(sum),
            f64_sums(&x).0,
            "a left-to-right f32 sum rounds"
        );
        // The u32 sum passes 2^32 by far, so the wrapping is exercised.
        let words = u32_values(1 << 16);
        assert!(words.iter().map(|w| u64::from(*w)).sum::<u64>() > 1 << 40);
    }

    /// The f64 reference sum is exact: every input is `k / 2^23` with `|k| < 2^23 + 2^22`, so a sequential f64 sum (53 bits)
    /// equals the exact integer sum over `2^23` while the partial sums stay below `2^53` in units of `2^-23`, that is for
    /// every `n <= 2^27` (`2^27 * 1.5 * 2^23 < 2^53`). It is why the first-order `(n - 1) 2^-53 S` is no looser than the
    /// rigorous recursive-sum bound here (review N1).
    #[test]
    fn the_f64_reference_sum_is_exact() {
        let x = f32_values(1 << 16);
        let exact_units: i64 = x.iter().map(|v| (f64::from(*v) * 8_388_608.0) as i64).sum();
        let (sum, _) = f64_sums(&x);
        assert_eq!(sum * 8_388_608.0, exact_units as f64);
    }

    /// Every order of summation of the f32 inputs is inside the serial bound: left to right, a pairwise tree, the
    /// reverse, and a strided order like a GPU lane layout.
    #[test]
    fn orders_that_round_differently_stay_inside_the_serial_bound() {
        let w = Workload::SumF32;
        let inputs = gen_inputs(w, 1 << 14);
        let reference = reference(w, &inputs);
        let reverse = inputs.x.iter().rev().fold(0.0f32, |s, v| s + v);
        let mut lanes = [0.0f32; 256];
        for (i, v) in inputs.x.iter().enumerate() {
            lanes[i % 256] += v;
        }
        let strided = lanes.iter().fold(0.0f32, |s, v| s + v);
        for sum in [serial_f32(&inputs.x), pairwise(&inputs.x), reverse, strided] {
            check(w, &inputs, &reference, &[sum]).unwrap();
        }
    }

    /// The f32 just inside `exact + limit` passes and the next one fails, for the serial bound and for a given tree
    /// bound; NaN and the infinities never pass; the poison alone fails.
    #[test]
    fn a_result_just_past_the_bound_fails_and_the_one_just_inside_passes() {
        let w = Workload::SumF32;
        let inputs = gen_inputs(w, 4096);
        let reference = reference(w, &inputs);
        let exact = reference.exact[0];
        for (tree, rounding) in [
            (None, Rounding::Nearest),
            (None, Rounding::Either),
            (Some(0.01), Rounding::Either),
        ] {
            let judging = Judging { rounding, tree };
            let bound = match rounding {
                Rounding::Nearest => reference.bound[0],
                Rounding::Either => reference.bound_gpu[0],
            };
            let limit = tree.unwrap_or(bound) + reference.ref_error;
            assert!(limit > 0.0 && limit.is_finite());
            // The largest f32 inside, and the next one above it (outside).
            let mut inside = (exact + limit) as f32;
            while f64::from(inside) - exact > limit {
                inside = inside.next_down();
            }
            while f64::from(inside.next_up()) - exact <= limit {
                inside = inside.next_up();
            }
            let outside = inside.next_up();
            assert!(f64::from(outside) - exact > limit, "the step is outside");
            check_with(w, &inputs, &reference, &[inside], judging).unwrap();
            let error = check_with(w, &inputs, &reference, &[outside], judging).unwrap_err();
            assert!(error.contains("first at 0"), "{error}");
            // And the same on the low side.
            let mut inside = (exact - limit) as f32;
            while exact - f64::from(inside) > limit {
                inside = inside.next_up();
            }
            while exact - f64::from(inside.next_down()) <= limit {
                inside = inside.next_down();
            }
            check_with(w, &inputs, &reference, &[inside], judging).unwrap();
            assert!(check_with(w, &inputs, &reference, &[inside.next_down()], judging).is_err());
        }
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(check(w, &inputs, &reference, &[bad]).is_err());
        }
        assert!(check(w, &inputs, &reference, &[w.poison_word()]).is_err());
        assert!(
            check(w, &inputs, &reference, &[]).is_err(),
            "a missing word fails"
        );
        // A tree bound tighter than the serial one is the one that judges a GUST result: the serial bound alone would
        // let this through.
        let loose = (exact + 0.5 * reference.bound[0]) as f32;
        check(w, &inputs, &reference, &[loose]).unwrap();
        let tight = Judging {
            rounding: Rounding::Either,
            tree: Some(0.01),
        };
        assert!(check_with(w, &inputs, &reference, &[loose], tight).is_err());
    }

    #[test]
    fn the_u32_sum_is_exact_modulo_2_32() {
        let w = Workload::SumU32;
        let inputs = gen_inputs(w, 5000);
        let reference = reference(w, &inputs);
        let sum = wrapping_sum(&inputs.values);
        assert_eq!(reference.want[0].to_bits(), sum);
        check(w, &inputs, &reference, &[f32::from_bits(sum)]).unwrap();
        for wrong in [sum.wrapping_add(1), sum.wrapping_sub(1), 0, u32::MAX] {
            assert!(
                check(w, &inputs, &reference, &[f32::from_bits(wrong)]).is_err(),
                "{wrong}"
            );
        }
        // A lost element and a doubled one both change the word.
        let lost = sum.wrapping_sub(inputs.values[17]);
        let doubled = sum.wrapping_add(inputs.values[17]);
        assert!(inputs.values[17] != 0);
        for wrong in [lost, doubled] {
            assert!(check(w, &inputs, &reference, &[f32::from_bits(wrong)]).is_err());
        }
        // The poison is never the answer of a case (it is the check's job to fail an undispatched output).
        assert_ne!(w.poison_word().to_bits(), sum);
        assert!(check(w, &inputs, &reference, &[w.poison_word()]).is_err());
    }
}
