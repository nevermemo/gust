//! W3: a polynomial of degree `d` (4, 16 or 64) by Horner's rule, `out[i] = c_0 + x (c_1 + x (c_2 + ...))`, f32.
//! Moves the benchmark from bandwidth to arithmetic: 2 * d flops per 8 bytes moved. The degree is the runtime length
//! of the coefficient buffer, so one kernel serves all three degrees (the loop is bounded by 64 and leaves at `d`).
//!
//! # The bound (derived here, plan 2.1)
//!
//! Model every f32 add and multiply of a backend, fused or not, as `fl(a op b) = (a op b)(1 + delta) + eta` with
//! `|delta| <= u = 2^-24` (round to nearest) and `|eta| <= 2^-126` (an absolute error only where a result underflows;
//! gradual underflow makes it `2^-149`, flushing makes it `2^-126`, so the larger is taken). A fused multiply-add rounds
//! once and so has only less error than its two parts. Horner's rule does `2 d` such operations, and Higham, *Accuracy
//! and Stability of Numerical Algorithms*, 2nd ed., section 5.1 (eq. 5.3), gives for the computed value `p^`:
//!
//! ```text
//! |p^ - p| <= gamma_{2d} * S,   S = sum_k |c_k| |x|^k,   gamma_k = k u / (1 - k u)
//! ```
//!
//! for the exact polynomial `p` of the f32 coefficients and the f32 `x` (both exact in f64). The reference is Horner's
//! rule in f64, which obeys the same theorem with `u64 = 2^-53`, so `|p_f64 - p| <= gamma64_{2d} * S` (`S` itself is
//! evaluated in f64 by Horner on `|c_k|` and `|x|`, all non-negative, so no cancellation and a relative error of at
//! most `gamma64_{2d}`: the computed `S_f64 <= (1 + gamma64_{2d}) S`, and the code scales the bound by that factor,
//! so the bound holds for the true `S` and not only for the computed one). The check compares `p^` with `p_f64`, so
//!
//! ```text
//! |p^ - p_f64| <= (gamma_{2d} + gamma64_{2d}) * S + 2 d * 2^-126
//! ```
//!
//! The last term is the underflow allowance: each of the `2 d` operations adds at most `2^-126`, and every later step
//! multiplies by `|x| <= 1`. A backend that fails the bound fails the cell; the bound is never widened to pass.
//! Domain the derivation needs (pinned by a test): finite inputs, no intermediate overflow (`|c_k| <= 1` and `|x| <= 1`
//! keep every partial result below `sum |c_k| < d + 1`), and a backend whose multiply and add are correctly rounded
//! (WGSL and Vulkan state that for `+` and `*`; a backend that fuses is inside the bound as shown).

use gust::gpu;

/// The GUST kernel. Workgroup size 256 (see `workloads::WORKGROUP`). The loop runs `degree` steps, at most 64.
#[gpu]
pub mod kernel {
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn horner(
        id: SV_DispatchThreadID,
        coeffs: StructuredBuffer<float>,
        x: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() && i < x.len() && 1u32 < coeffs.len() {
            let degree = coeffs.len() - 1u32;
            let v = x[i];
            let mut acc = coeffs[degree];
            for k in 0u32..64u32 {
                if k >= degree {
                    break;
                }
                acc = acc * v + coeffs[degree - 1u32 - k];
            }
            out[i] = acc;
        }
    }
}

/// Deterministic coefficients in [-1, 1), `degree + 1` of them, lowest degree first.
pub fn coefficients(degree: usize) -> Vec<f32> {
    (0..=degree)
        .map(|k| ((k * 37 + 11) % 97) as f32 / 48.5 - 1.0)
        .collect()
}

const U32: f64 = 1.0 / 16_777_216.0; // 2^-24
const U64: f64 = 1.0 / 9_007_199_254_740_992.0; // 2^-53

/// `gamma_k = k u / (1 - k u)`.
pub fn gamma(k: usize, u: f64) -> f64 {
    let ku = k as f64 * u;
    ku / (1.0 - ku)
}

/// The exact value (Horner in f64) and the derived absolute bound of the f32 Horner result at each `x`.
pub fn exact_and_bound(coeffs: &[f32], xs: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let degree = coeffs.len() - 1;
    let factor = gamma(2 * degree, U32) + gamma(2 * degree, U64);
    let eta = 2.0 * degree as f64 * f64::from(f32::MIN_POSITIVE);
    // S is itself computed in f64: `S_f64 <= (1 + gamma64) S`.
    let s_slack = 1.0 + gamma(2 * degree, U64);
    let mut exact = Vec::with_capacity(xs.len());
    let mut bound = Vec::with_capacity(xs.len());
    for x in xs {
        let x = f64::from(*x);
        let (mut p, mut s) = (f64::from(coeffs[degree]), f64::from(coeffs[degree]).abs());
        for c in coeffs[..degree].iter().rev() {
            p = p * x + f64::from(*c);
            s = s * x.abs() + f64::from(*c).abs();
        }
        exact.push(p);
        bound.push(factor * s * s_slack + eta);
    }
    (exact, bound)
}

/// The element function of the CPU backends: Horner in f32, one multiply and one add per step, never fused.
#[inline(always)]
pub fn element(coeffs: &[f32], x: f32) -> f32 {
    let degree = coeffs.len() - 1;
    let mut acc = coeffs[degree];
    for c in coeffs[..degree].iter().rev() {
        acc = acc * x + *c;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workloads::{Workload, check, gen_inputs, reference};

    #[test]
    fn gamma_matches_its_definition() {
        let g = gamma(8, U32);
        assert!((g - 8.0 * U32 / (1.0 - 8.0 * U32)).abs() < 1e-30);
        assert!(gamma(128, U32) > gamma(8, U32));
        // For d = 64 the factor is about 128 * 2^-24 = 7.6e-6.
        assert!((gamma(128, U32) - 7.63e-6).abs() < 2e-8);
    }

    #[test]
    fn the_domain_the_derivation_needs_holds_for_the_inputs() {
        for degree in [4, 16, 64] {
            let c = coefficients(degree);
            assert_eq!(c.len(), degree + 1);
            assert!(c.iter().all(|c| c.is_finite() && c.abs() <= 1.0));
            let inputs = gen_inputs(Workload::Poly64, 1 << 14);
            assert!(inputs.x.iter().all(|x| x.abs() <= 1.0));
            let sum: f64 = c.iter().map(|c| f64::from(c.abs())).sum();
            assert!(sum < degree as f64 + 1.0);
        }
    }

    /// The f32 results of three orders of evaluation all satisfy the bound: separate multiply and add, fused
    /// multiply-add, and a powers-then-sum evaluation (not Horner at all, but its error is also within gamma_{2d}).
    #[test]
    fn backends_that_round_differently_stay_inside_the_bound() {
        for w in [Workload::Poly4, Workload::Poly16, Workload::Poly64] {
            let inputs = gen_inputs(w, 1 << 13);
            let reference = reference(w, &inputs);
            let plain = inputs
                .x
                .iter()
                .map(|x| element(&inputs.coeffs, *x))
                .collect::<Vec<_>>();
            check(w, &inputs, &reference, &plain).unwrap();
            let fused = inputs
                .x
                .iter()
                .map(|x| {
                    let d = inputs.coeffs.len() - 1;
                    inputs.coeffs[..d]
                        .iter()
                        .rev()
                        .fold(inputs.coeffs[d], |acc, c| acc.mul_add(*x, *c))
                })
                .collect::<Vec<_>>();
            check(w, &inputs, &reference, &fused).unwrap();
            let powers = inputs
                .x
                .iter()
                .map(|x| {
                    let (mut p, mut sum) = (1.0f32, 0.0f32);
                    for c in &inputs.coeffs {
                        sum += c * p;
                        p *= x;
                    }
                    sum
                })
                .collect::<Vec<_>>();
            check(w, &inputs, &reference, &powers).unwrap();
        }
    }

    #[test]
    fn a_result_off_by_more_than_the_bound_fails_and_one_within_it_passes() {
        let w = Workload::Poly16;
        let inputs = gen_inputs(w, 4096);
        let reference = reference(w, &inputs);
        let result = inputs
            .x
            .iter()
            .map(|x| element(&inputs.coeffs, *x))
            .collect::<Vec<_>>();
        let i = 1234;
        let bound = reference.bound[i];
        assert!(
            bound > 0.0 && bound < 1e-3,
            "the bound is tight, not vacuous: {bound}"
        );
        // The bound is far above one ulp of the value only where the polynomial is small relative to S; the point is
        // that a perturbation of 4 bounds fails and half a bound passes, at the element where it is applied.
        let mut inside = result.clone();
        inside[i] = (reference.exact[i] + 0.5 * bound) as f32;
        let mut outside = result.clone();
        outside[i] = (reference.exact[i] + 4.0 * bound) as f32;
        let ulp = f64::from(f32::EPSILON) * reference.exact[i].abs().max(1.0);
        assert!(
            4.0 * bound > ulp,
            "a 4-bound step is not lost in the f32 rounding"
        );
        check(w, &inputs, &reference, &inside).unwrap();
        let error = check(w, &inputs, &reference, &outside).unwrap_err();
        assert!(error.contains("first at 1234"), "{error}");
        // A NaN and an infinity never pass.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut poisoned = result.clone();
            poisoned[i] = bad;
            assert!(check(w, &inputs, &reference, &poisoned).is_err());
        }
        // The wrong degree (the cubic of examples/polynomial) fails everywhere it differs.
        let wrong = inputs.x.iter().map(|x| x * x * x).collect::<Vec<_>>();
        assert!(check(w, &inputs, &reference, &wrong).is_err());
    }
}
