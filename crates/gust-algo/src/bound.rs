//! The error bound of a float `sum` (plan section 5.1, R1.1 M2), the one number the float tests assert and the
//! derivation behind it.
//!
//! Model. Every addition the kernels evaluate rounds with a relative error of at most `c`: `c = 2^-24` for Rust's
//! round-to-nearest (the CPU twin), `c = 2^-23` for the GPU, which `docs/SEMANTICS.md` section 4 allows to round to
//! either neighbour ("correctly rounded", within 1 ULP). A lane adds at most
//! [`CHAIN`](crate::schedule::CHAIN) terms. The spec lets an implementation regroup a sum freely (`docs/SEMANTICS.md`
//! section 4), but any regrouping of the `m` terms of one lane is a tree of depth at most `m - 1`, so it costs a term
//! no more roundings than the lane's own left-to-right chain; the levels are separate dispatches, so no grouping spans
//! two of them; and the fixture has no product, so no fused multiply-add can form. Each value passes through at
//! most `K = sum over the levels of (longest chain - 1)` additions (a lane starts from `0.0`, whose first addition is
//! exact). The classical bound for such a depth is `|computed - exact| <= gamma_K * S` with
//! `gamma_K = K * c / (1 - K * c)` and `S` the sum of the absolute values of the inputs (Higham, "Accuracy and
//! Stability of Numerical Algorithms", section 4.2, applied to the tree of chains). For the four-level schedule of an
//! input up to 2^29 elements `K <= 1 020` (the plan's ceiling; the real value is 796 at 2^29, see
//! [`depth`]), so the bound is about `K * c * S`, where the order-free `(n - 1) * c * S` would be 12 times `S`
//! at 1e8.
//!
//! The bound is absolute in `S`, not relative to the result: for inputs of alternating sign the result can be far
//! smaller than `S`, and the bound says nothing about its relative error.
//!
//! The comparison is made against an independent reference, an `f64` accumulation on the host, whose own error is at
//! most `(n - 1) * 2^-53 * S` ([`f64_reference_error`]); a test adds it to the bound. Domain: finite inputs, normal or
//! zero (a subnormal operand may be flushed on the GPU), and no partial overflowing. The grouped CPU form
//! ([`sum_grouped`]) is supplementary: recorded, never asserted equal.

use crate::schedule::{BlockSchedule, CHAIN, LANES, Schedule};

/// The per-addition relative error of the GPU: 1 ULP of the result, `2^-23`.
pub const C_GPU: f64 = 1.0 / 8_388_608.0;
/// The per-addition relative error of the CPU twin: half an ULP, `2^-24`.
pub const C_CPU: f64 = 1.0 / 16_777_216.0;

/// `K`: the most additions any input passes through, `sum(chain - 1)` over the levels of `schedule`.
pub fn depth(schedule: &Schedule) -> u64 {
    schedule
        .chains()
        .iter()
        .map(|chain| u64::from(chain.saturating_sub(1)))
        .sum()
}

/// The bound on `|computed - exact|` of a float sum of inputs whose absolute values add up to `abs_sum`, with the
/// per-addition error `c` ([`C_GPU`] or [`C_CPU`]): `gamma_K * abs_sum`. `n == 0` has no error.
///
/// # Panics
///
/// If `K * c >= 1`, outside any schedule this crate produces (`K <= 1 020`, `c <= 2^-23`).
pub fn f32_sum_bound(n: u32, c: f64, abs_sum: f64) -> f64 {
    let Some(schedule) = Schedule::new(n) else {
        return 0.0;
    };
    gamma_bound(depth(&schedule), c, abs_sum)
}

/// `K` of the v1 block reduce: the additions on the longest path, summed over the levels of `schedule` (each level: the
/// lane chain of at most 32 terms, one addition fewer than terms, and the 8 rounds of the tree).
pub fn depth_v1(schedule: &BlockSchedule) -> u64 {
    schedule
        .chains()
        .iter()
        .map(|steps| u64::from(*steps))
        .sum()
}

/// [`f32_sum_bound`] for the v1 block reduce (plan section 5.5): the same form `gamma_K * S` with the `K` of
/// [`depth_v1`], at most 39 per level and 86 for the levels of an input of 2^29 elements, against the 796 of v0.
/// The tree adds 256 values of one level in a balanced shape, so a value passes through 8 additions in it, not 255.
///
/// # Panics
///
/// If `K * c >= 1`, outside any schedule this crate produces.
pub fn f32_sum_bound_v1(n: u32, c: f64, abs_sum: f64) -> f64 {
    let Some(schedule) = BlockSchedule::new(n) else {
        return 0.0;
    };
    gamma_bound(depth_v1(&schedule), c, abs_sum)
}

fn gamma_bound(depth: u64, c: f64, abs_sum: f64) -> f64 {
    let steps = depth as f64 * c;
    assert!(
        steps < 1.0,
        "the depth of the schedule is outside the bound"
    );
    abs_sum * steps / (1.0 - steps)
}

/// The most the `f64` reference of a sum of `n` `f32` values (each exact in `f64`) can be off: `(n - 1) * 2^-53 * S`.
pub fn f64_reference_error(n: u32, abs_sum: f64) -> f64 {
    f64::from(n.saturating_sub(1)) * (1.0 / 9_007_199_254_740_992.0) * abs_sum
}

/// The sum of the absolute values, accumulated in `f64`.
pub fn abs_sum(values: &[f32]) -> f64 {
    values.iter().map(|value| f64::from(value.abs())).sum()
}

/// The independent reference of a float sum: a sequential `f64` accumulation.
pub fn sum_f64(values: &[f32]) -> f64 {
    values.iter().map(|value| f64::from(*value)).sum()
}

/// The sum in the order of the kernels, on the host and in `f32`: the block-strided lane chains of each pass, then the
/// terminal chain. It is the CPU form of the schedule without the CPU device (supplementary: a result to compare with,
/// not an oracle, since the GPU may regroup within an expression). `0.0` for an empty input.
pub fn sum_grouped(values: &[f32]) -> f32 {
    let Some(schedule) =
        Schedule::new(u32::try_from(values.len()).expect("at most u32::MAX elements"))
    else {
        return 0.0;
    };
    let mut level: Vec<f32> = values.to_vec();
    for pass in schedule.passes() {
        let mut partials = vec![0.0_f32; pass.partials() as usize];
        for (slot, partial) in partials.iter_mut().enumerate() {
            let slot = slot as u64;
            let first = (slot / u64::from(LANES)) * u64::from(crate::schedule::BLOCK)
                + slot % u64::from(LANES);
            let mut accumulator = 0.0_f32;
            for k in 0..u64::from(CHAIN) {
                let index = first + k * u64::from(LANES);
                if index >= level.len() as u64 {
                    break;
                }
                accumulator += level[index as usize];
            }
            *partial = accumulator;
        }
        level = partials;
    }
    let mut accumulator = 0.0_f32;
    for value in &level {
        accumulator += *value;
    }
    accumulator
}
