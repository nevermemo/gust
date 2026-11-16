//! W2: saxpy, `out[i] = scale * x[i] + y[i]`, f32, out of place. Memory-bandwidth bound: 12 bytes per element.
//!
//! The scalar travels as a one-element read-only storage buffer on both GPU paths (research section 2, D40).
//!
//! # The bound (derived here, b3b; it replaces the b1 two-epsilon allowance)
//!
//! Model every f32 add and multiply as `fl(a op b) = (a op b)(1 + delta) + eta`, `|delta| <= u`, `|eta| <= 2^-126`, with
//! the backend's unit roundoff `u` (`2^-24` for the CPU backends, `2^-23` for every GPU backend, as for W3, W7 and W9; see
//! [`Rounding`](super::Rounding)). The kernel does one multiply and one add. With `p = a x` (exact in f64: two 24-bit
//! significands) the computed value is `(p (1 + d1) + y)(1 + d2) + eta`, so against the exact `p + y`
//!
//! ```text
//! |out^ - (p + y)| <= u (1 + u) |p| + u |p + y| + (the eta terms) <= ((1 + u)^2 - 1) (|p| + |y|) + 2 * 2^-126
//! ```
//!
//! (the product form of Higham, section 3.1: two operations, `(1 + u)^2 - 1 = u (2 + u)`). A fused multiply-add rounds
//! once and has only less error, so it is inside the same bound. The reference is `a x + y` in f64, one rounding of
//! `2^-53 |p + y| <= 2^-53 (|p| + |y|)`, so the check is
//!
//! ```text
//! |out^ - ref_f64| <= (((1 + u)^2 - 1) + 2^-53) (|a x| + |y|) + 2 * 2^-126
//! ```
//!
//! The old allowance, two f32 epsilons (`2^-22`) of the same magnitude, assumed round to nearest (`u = 2^-24`, factor
//! about `2^-23`); this form follows the rule W3, W7 and W9 follow, so a GPU gets the factor of `u = 2^-23`.

use gust::gpu;

use super::Rounding;

#[gpu]
pub mod kernel {
    #[allow(clippy::len_zero)]
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn saxpy(
        id: SV_DispatchThreadID,
        scale: StructuredBuffer<float>,
        x: StructuredBuffer<float>,
        y: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() && 0u32 < scale.len() {
            out[i] = scale[0u32] * x[i] + y[i];
        }
    }
}

/// The rayon / serial / reference element function: a multiply and an add, never fused.
#[inline(always)]
pub fn element(scale: f32, x: f32, y: f32) -> f32 {
    scale * x + y
}

/// The exact value `scale * x + y` in f64 (the product is exact there; one rounding for the add).
pub fn exact(scale: f32, x: f32, y: f32) -> f64 {
    f64::from(scale) * f64::from(x) + f64::from(y)
}

/// The derived absolute bound of one element for a backend of the given rounding (see the module notes).
pub fn bound(rounding: Rounding, scale: f32, x: f32, y: f32) -> f64 {
    let u = rounding.unit();
    let magnitude = (f64::from(scale) * f64::from(x)).abs() + f64::from(y).abs();
    (((1.0 + u) * (1.0 + u) - 1.0) + super::polynomial::U64) * magnitude
        + 2.0 * f64::from(f32::MIN_POSITIVE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workloads::{Workload, check, check_on, gen_inputs, reference};

    /// The unit-roundoff factor of two operations: about `2^-23` for a CPU, `2^-22` for a GPU.
    #[test]
    fn the_bound_is_two_roundings_of_the_backends_unit() {
        let (x, y) = (1.0f32, 0.5f32);
        let magnitude = 2.5 * 1.0 + 0.5;
        let near = bound(Rounding::Nearest, 2.5, x, y);
        let far = bound(Rounding::Either, 2.5, x, y);
        assert!((near / magnitude - 2.0 * super::super::polynomial::U32).abs() < 1e-9);
        assert!((far / magnitude - 4.0 * super::super::polynomial::U32).abs() < 1e-8);
        assert!(far > near && far < 2.1 * near);
    }

    /// Inputs that really round (the benchmark's own W2 inputs are exact by construction, so a run of it cannot observe a
    /// rounding or a contraction): full 24-bit significands from a hash and a scale that is not a power of two.
    fn rounding_inputs(n: usize) -> crate::workloads::Inputs {
        let mut inputs = gen_inputs(Workload::Saxpy, n);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u32 << 23) as f32) - 1.0
        };
        inputs.x = (0..n).map(|_| next()).collect();
        inputs.y = (0..n).map(|_| next()).collect();
        inputs.scale = 1.1;
        inputs
    }

    #[test]
    fn a_plain_and_a_fused_result_are_inside_the_bound_under_both_units() {
        let w = Workload::Saxpy;
        let inputs = rounding_inputs(4096);
        let reference = reference(w, &inputs);
        let plain = reference.want.clone();
        let fused = inputs
            .x
            .iter()
            .zip(&inputs.y)
            .map(|(x, y)| inputs.scale.mul_add(*x, *y))
            .collect::<Vec<_>>();
        // The test means something only if fusing changes the result and the rounding is visible: a stated share of
        // elements (at least a tenth) differ, and the worst plain error is a real fraction of the bound.
        let differ = plain.iter().zip(&fused).filter(|(a, b)| a != b).count();
        assert!(
            differ * 10 >= inputs.n,
            "{differ} of {} elements differ",
            inputs.n
        );
        let worst = (0..inputs.n)
            .map(|i| {
                (f64::from(plain[i]) - exact(inputs.scale, inputs.x[i], inputs.y[i])).abs()
                    / bound(Rounding::Nearest, inputs.scale, inputs.x[i], inputs.y[i])
            })
            .fold(0.0, f64::max);
        assert!(
            worst > 0.1 && worst <= 1.0,
            "worst plain error is {worst} of the bound"
        );
        for result in [&plain, &fused] {
            check(w, &inputs, &reference, result).unwrap();
            for rounding in [Rounding::Nearest, Rounding::Either] {
                check_on(w, &inputs, &reference, result, rounding).unwrap();
            }
        }
    }

    /// The largest f32 within `limit` of `exact` on the high side, and its successor (just outside).
    fn edge(exact: f64, limit: f64) -> (f32, f32) {
        let mut inside = (exact + limit) as f32;
        while f64::from(inside) - exact > limit {
            inside = inside.next_down();
        }
        while f64::from(inside.next_up()) - exact <= limit {
            inside = inside.next_up();
        }
        (inside, inside.next_up())
    }

    /// A value between the two bounds is a GPU's to give and not a CPU's; a value just past the GPU bound fails on both.
    #[test]
    fn a_value_between_the_two_bounds_passes_on_a_gpu_and_fails_on_a_cpu() {
        let w = Workload::Saxpy;
        let inputs = gen_inputs(w, 4096);
        let reference = reference(w, &inputs);
        // An element where the largest f32 inside the GPU bound is outside the CPU bound (the f32 grid is fine enough).
        let (i, between) = (0..inputs.n)
            .find_map(|i| {
                let (scale, x, y) = (inputs.scale, inputs.x[i], inputs.y[i]);
                let (exact, near, far) = (
                    super::exact(scale, x, y),
                    bound(Rounding::Nearest, scale, x, y),
                    bound(Rounding::Either, scale, x, y),
                );
                let (inside, _) = edge(exact, far);
                (f64::from(inside) - exact > near).then_some((i, inside))
            })
            .expect("some element has an f32 between the two bounds");
        let mut got = reference.want.clone();
        got[i] = between;
        check_on(w, &inputs, &reference, &got, Rounding::Either).unwrap();
        assert!(check_on(w, &inputs, &reference, &got, Rounding::Nearest).is_err());
        assert!(check(w, &inputs, &reference, &got).is_err());
    }

    #[test]
    fn a_result_one_f32_step_past_the_bound_fails_and_poison_never_passes() {
        let w = Workload::Saxpy;
        let inputs = gen_inputs(w, 4096);
        let reference = reference(w, &inputs);
        let i = 1234;
        let (scale, x, y) = (inputs.scale, inputs.x[i], inputs.y[i]);
        let far = bound(Rounding::Either, scale, x, y);
        assert!(
            far > 0.0 && far < 1e-5,
            "tight, not vacuous: {far} (an f32 step of the result is about 1e-7)"
        );
        let (inside, outside) = edge(exact(scale, x, y), far);
        let mut got = reference.want.clone();
        got[i] = inside;
        check_on(w, &inputs, &reference, &got, Rounding::Either).unwrap();
        got[i] = outside;
        let error = check_on(w, &inputs, &reference, &got, Rounding::Either).unwrap_err();
        assert!(
            error.starts_with("1 of 4096 elements are wrong; first at 1234"),
            "{error}"
        );
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            got[i] = poison;
            assert!(check_on(w, &inputs, &reference, &got, Rounding::Either).is_err());
        }
    }
}
