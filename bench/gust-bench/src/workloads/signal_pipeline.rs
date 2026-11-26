//! W6: a staged pipeline, the six steps of `examples/signal-pipeline`'s `transform` as six kernels of their own, `f32`,
//! one invocation per element, a buffer between every two stages. The example runs them as one kernel (six statements, one
//! dispatch); here each statement is a kernel, so a run is six dispatches in a chain and the workload is about what
//! sequencing them costs: GUST's `StagedGraph` (the graph derives the edges from the buffers, checks them at execution and runs the nodes in the order added) against the same six
//! dispatches in hand order (GUST's `dispatch_batch`, the hand WGSL's six passes).
//!
//! ```text
//! centered   = signal - baseline          // stage 0: signal, baseline          -> centered
//! amplified  = centered * gain            // stage 1: centered, gain            -> amplified
//! biased     = amplified + bias           // stage 2: amplified, bias           -> biased
//! normalized = biased / scale             // stage 3: biased, scale             -> normalized
//! energy     = normalized * normalized    // stage 4: normalized                -> energy
//! out        = energy + 0.25 * normalized // stage 5: energy, normalized        -> out
//! ```
//!
//! The inputs and their ranges are the example's (`signal` in `[-4, 4)`, `baseline` in `[-0.15, 0.15]`, `gain` in
//! `[0.75, 1.15]`, `bias` in `[-0.12, 0.12]`, `scale` in `[1, 1.66]`; every one periodic, so a few thousand elements reach
//! every extreme). Only the final stage is read back in a timed sample; every stage is checked in a unit test.
//!
//! # The bound (derived here, b3c)
//!
//! The example derives a twin-versus-GPU tolerance by first-order propagation (`TWIN_VS_GPU_BOUND`, a table of
//! magnitudes). This workload checks against the f64 value instead, as W3, W5 and W9 do, so each backend is judged by the
//! bound of its own rounding ([`Rounding`](super::Rounding)): `u = 2^-24` for the CPU backends and `u = 2^-23` for every
//! GPU backend. Every add and multiply is `fl(a op b) = (a op b)(1 + delta) + eta`, `|delta| <= u`, `|eta| <= 2^-126`,
//! and the division is the same with `|delta| <= u_div`: `u_div = u = 2^-24` (0.5 ULP, round to nearest) on a CPU and
//! `u_div = 2.5 * 2^-23` on a GPU (`docs/SEMANTICS.md` section 4: "within 2.5 ULP", one ULP being at most `2^-23` of the
//! magnitude; the example's `5u`). The forward error of the expression tree is carried numerically, in f64, along the
//! f64 trajectory: a value is a pair `(v, e)`, the computed value within `e` of the exact one, and
//!
//! ```text
//! add:  v = a.v + b.v    e = (a.e + b.e)(1 + u) + u |v| + eta                    (a subtraction is the same)
//! mul:  v = a.v * b.v    e = (|a.v| b.e + |b.v| a.e + a.e b.e)(1 + u) + u |v| + eta
//! div:  v = a.v / b.v    e = ((a.e + |v| b.e) / (|b.v| - b.e)) (1 + u_div) + u_div |v| + eta     (b.e < |b.v|)
//! ```
//!
//! (the quotient of operands within `e_a` and `e_b` of the exact ones is within `(e_a + |v| e_b) / (|b| - e_b)` of the
//! exact quotient: `|a' b - a b'| / (|b'| |b|)`). The five inputs are exact f32 values. `0.25 * normalized` is a multiply by
//! a power of two, exact (no rounding, error `0.25 e`), so the last stage is one rounded add whether or not the compiler
//! fuses it. No stage has a chain of additions longer than two, and each stage is a dispatch of its own that stores its
//! result as an `f32`, so a GPU has nothing to regroup or to fuse across a stage (a fused multiply-add rounds once and so
//! has only less error than the multiply and add it replaces, the argument of W2 and W5), and the bound needs no
//! order-free form: each stage's computed value is its operation applied to the previous stage's computed values, which is
//! exactly what the recurrence propagates. The reference is the same six operations in f64, which has its own error,
//! carried with `u = 2^-53` (the division also correctly rounded), no `eta`. A stage's bound is the sum of the f32 model's
//! `e` and the f64 reference's, times `1 + 1e-9` ([`SLACK`], the f64 evaluation of the bound itself and the difference
//! between the exact trajectory and the f64 one inside its terms). The check of a timed sample compares the final
//! stage with its bound (the composed bound of the six stages); a unit test compares every stage separately with its own.
//! A backend that fails the bound fails the cell; the bound is never widened to pass.

// The check loops index several parallel slices by element and stage; an iterator form would only hide that.
#![allow(clippy::needless_range_loop)]

use gust::gpu_module;

use super::{Rounding, polynomial::U64};

/// The six GUST kernels, one per stage (workgroup size 256, see `workloads::WORKGROUP`). Every kernel guards on its
/// output's length, as the other workloads' do.
#[gpu_module]
pub mod kernel {
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn center(
        id: SV_DispatchThreadID,
        signal: StructuredBuffer<float>,
        baseline: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = signal[i] - baseline[i];
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn amplify(
        id: SV_DispatchThreadID,
        centered: StructuredBuffer<float>,
        gain: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = centered[i] * gain[i];
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn shift(
        id: SV_DispatchThreadID,
        amplified: StructuredBuffer<float>,
        bias: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = amplified[i] + bias[i];
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn rescale(
        id: SV_DispatchThreadID,
        biased: StructuredBuffer<float>,
        scale: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = biased[i] / scale[i];
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn energy(
        id: SV_DispatchThreadID,
        normalized: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = normalized[i] * normalized[i];
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn combine(
        id: SV_DispatchThreadID,
        squared: StructuredBuffer<float>,
        normalized: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = squared[i] + (0.25 * normalized[i]);
        }
    }
}

/// The hand-fused form of the six stages (spike m14.pffuse): `examples/signal-pipeline`'s `transform` at the stages'
/// workgroup size, one kernel and one dispatch that keeps the five intermediate values in registers and writes only the
/// final output. A module of its own, so the six stage kernels' module is unchanged.
#[gpu_module]
pub mod fused {
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn transform(
        id: SV_DispatchThreadID,
        signal: StructuredBuffer<float>,
        baseline: StructuredBuffer<float>,
        gain: StructuredBuffer<float>,
        bias: StructuredBuffer<float>,
        scale: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            let centered = signal[i] - baseline[i];
            let amplified = centered * gain[i];
            let biased = amplified + bias[i];
            let normalized = biased / scale[i];
            let energy = normalized * normalized;
            out[i] = energy + (0.25 * normalized);
        }
    }
}

/// Stages of the pipeline.
pub const STAGES: usize = 6;
/// Host input streams: `signal`, `baseline`, `gain`, `bias`, `scale`.
#[cfg(test)]
pub const INPUTS: usize = 5;

/// The five input streams of `n` elements, the example's `Inputs::new`.
pub fn inputs(n: usize) -> Vec<Vec<f32>> {
    let stream = |f: fn(usize) -> f32| (0..n).map(f).collect::<Vec<_>>();
    vec![
        stream(|i| (i % 2048) as f32 / 256.0 - 4.0),
        stream(|i| (i % 31) as f32 * 0.01 - 0.15),
        stream(|i| 0.75 + (i % 17) as f32 * 0.025),
        stream(|i| (i % 13) as f32 * 0.02 - 0.12),
        stream(|i| 1.0 + (i % 23) as f32 * 0.03),
    ]
}

/// The six stage values of one element, in `f32` round to nearest, every operation rounded (the host arithmetic of the
/// CPU backends; the kernels' arithmetic).
#[inline(always)]
pub fn stages(signal: f32, baseline: f32, gain: f32, bias: f32, scale: f32) -> [f32; STAGES] {
    let centered = signal - baseline;
    let amplified = centered * gain;
    let biased = amplified + bias;
    let normalized = biased / scale;
    let energy = normalized * normalized;
    [
        centered,
        amplified,
        biased,
        normalized,
        energy,
        energy + 0.25 * normalized,
    ]
}

/// The final stage of one element: the rayon and serial rows' fused loop body (no stage buffers) and the reference's.
#[inline(always)]
pub fn element(signal: f32, baseline: f32, gain: f32, bias: f32, scale: f32) -> f32 {
    stages(signal, baseline, gain, bias, scale)[STAGES - 1]
}

/// A computed value and a bound on its distance from the exact one (see the module notes).
#[derive(Clone, Copy)]
struct Tracked {
    val: f64,
    err: f64,
}

/// One backend model: the unit roundoff of an add or a multiply, of a division, and the underflow term.
#[derive(Clone, Copy)]
struct Model {
    u: f64,
    u_div: f64,
    eta: f64,
}

impl Model {
    fn add(self, a: Tracked, b: Tracked) -> Tracked {
        let val = a.val + b.val;
        Tracked {
            val,
            err: (a.err + b.err) * (1.0 + self.u) + self.u * val.abs() + self.eta,
        }
    }

    fn mul(self, a: Tracked, b: Tracked) -> Tracked {
        let val = a.val * b.val;
        let input = a.val.abs() * b.err + b.val.abs() * a.err + a.err * b.err;
        Tracked {
            val,
            err: input * (1.0 + self.u) + self.u * val.abs() + self.eta,
        }
    }

    fn div(self, a: Tracked, b: Tracked) -> Tracked {
        let val = a.val / b.val;
        let room = b.val.abs() - b.err;
        // A divisor whose error reaches its magnitude has no bound: infinity makes every check against it pass, so it
        // is refused loudly instead (the domain test shows it does not happen).
        assert!(room > 0.0, "the divisor's error reaches its magnitude");
        let input = (a.err + val.abs() * b.err) / room;
        Tracked {
            val,
            err: input * (1.0 + self.u_div) + self.u_div * val.abs() + self.eta,
        }
    }

    /// The six stages of one element: the value and error of each stage's output.
    fn run(self, s: f32, b: f32, g: f32, bi: f32, sc: f32) -> [Tracked; STAGES] {
        let exact = |v: f32| Tracked {
            val: f64::from(v),
            err: 0.0,
        };
        let centered = self.add(
            exact(s),
            Tracked {
                val: -f64::from(b),
                err: 0.0,
            },
        );
        let amplified = self.mul(centered, exact(g));
        let biased = self.add(amplified, exact(bi));
        let normalized = self.div(biased, exact(sc));
        let energy = self.mul(normalized, normalized);
        // A multiply by a power of two: exact, so the error scales and nothing is rounded.
        let quarter = Tracked {
            val: 0.25 * normalized.val,
            err: 0.25 * normalized.err,
        };
        let out = self.add(energy, quarter);
        [centered, amplified, biased, normalized, energy, out]
    }
}

/// The model of a backend of the given rounding: its unit roundoff, the division's (0.5 ULP on a CPU, 2.5 ULP of
/// `u = 2^-23` on a GPU) and `eta = 2^-126`.
fn model_of(rounding: Rounding) -> Model {
    let u = rounding.unit();
    Model {
        u,
        u_div: match rounding {
            Rounding::Nearest => u,
            Rounding::Either => 2.5 * u,
        },
        eta: f64::from(f32::MIN_POSITIVE),
    }
}

/// The f64 reference's own model: `u = 2^-53` (the division correctly rounded too), no underflow term.
const REFERENCE: Model = Model {
    u: U64,
    u_div: U64,
    eta: 0.0,
};

/// Covers the f64 evaluation of the bound itself and the exact-versus-f64 trajectory inside its terms.
pub const SLACK: f64 = 1.0 + 1e-9;

/// The f64 value of stage `stage` (0 to 5) of every element and the derived absolute bounds for a CPU backend
/// (`u = 2^-24`) and for a GPU (`u = 2^-23`, division 2.5 ULP): three vectors of `n` values. `stage = 5` is the final
/// output a sample reads back.
pub fn stage_exact_and_bound(inputs: &[Vec<f32>], stage: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = inputs[0].len();
    let (near, gpu) = (model_of(Rounding::Nearest), model_of(Rounding::Either));
    let mut exact = Vec::with_capacity(n);
    let mut bound = Vec::with_capacity(n);
    let mut bound_gpu = Vec::with_capacity(n);
    for i in 0..n {
        let at = |k: usize| inputs[k][i];
        let reference = REFERENCE.run(at(0), at(1), at(2), at(3), at(4))[stage];
        let a = near.run(at(0), at(1), at(2), at(3), at(4))[stage];
        let b = gpu.run(at(0), at(1), at(2), at(3), at(4))[stage];
        exact.push(reference.val);
        bound.push((a.err + reference.err) * SLACK);
        bound_gpu.push((b.err + reference.err) * SLACK);
    }
    (exact, bound, bound_gpu)
}

/// The final stage: the f64 value and the two bounds of the check (`stage_exact_and_bound` of stage 5).
pub fn exact_and_bound(inputs: &[Vec<f32>]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    stage_exact_and_bound(inputs, STAGES - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workloads::{Workload, check, check_on, gen_inputs, reference};

    const W: Workload = Workload::PipelineHand;

    /// Every stage of every element of `inputs`, written in `f32` round to nearest.
    fn host_stages(inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut out = vec![Vec::new(); STAGES];
        for i in 0..inputs[0].len() {
            let s = stages(
                inputs[0][i],
                inputs[1][i],
                inputs[2][i],
                inputs[3][i],
                inputs[4][i],
            );
            for (stage, value) in s.into_iter().enumerate() {
                out[stage].push(value);
            }
        }
        out
    }

    /// `x` rounded to `f32` in a chosen direction (what a GPU's "either neighbour" allows an add or a multiply).
    fn directed(x: f64, up: bool) -> f32 {
        let near = x as f32;
        if up && f64::from(near) < x {
            near.next_up()
        } else if !up && f64::from(near) > x {
            near.next_down()
        } else {
            near
        }
    }

    /// The pipeline as a worst-case GPU may run it: every add and multiply rounded in the direction `direction(index)`
    /// says (a pseudo-random one per operation), every division pushed two ULP from the nearest result (2.5 ULP in all).
    /// Each stage is stored as an `f32`, as the kernels store theirs.
    fn gpu_stages(inputs: &[Vec<f32>], direction: impl Fn(usize) -> bool) -> Vec<Vec<f32>> {
        let mut out = vec![Vec::new(); STAGES];
        for i in 0..inputs[0].len() {
            let at = |k: usize| f64::from(inputs[k][i]);
            let d = |op: usize| direction(i * 8 + op);
            let centered = directed(at(0) - at(1), d(0));
            let amplified = directed(f64::from(centered) * at(2), d(1));
            let biased = directed(f64::from(amplified) + at(3), d(2));
            let mut normalized = ((f64::from(biased) / at(4)) as f32).next_up().next_up();
            if d(3) {
                normalized = ((f64::from(biased) / at(4)) as f32).next_down().next_down();
            }
            let energy = directed(f64::from(normalized) * f64::from(normalized), d(4));
            let last = directed(f64::from(energy) + 0.25 * f64::from(normalized), d(5));
            for (stage, value) in [centered, amplified, biased, normalized, energy, last]
                .into_iter()
                .enumerate()
            {
                out[stage].push(value);
            }
        }
        out
    }

    fn direction(seed: u64) -> impl Fn(usize) -> bool {
        move |index| {
            let z = (index as u64 ^ seed).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            (z >> 40) & 1 == 1
        }
    }

    /// The domain the derivation needs: the example's ranges, finite values nowhere near the subnormals, and a divisor
    /// whose (zero) error is below its magnitude. `inputs(n)` is periodic, so 4096 elements reach every extreme.
    #[test]
    fn the_domain_holds_and_the_stages_round() {
        let inputs = inputs(4103);
        assert_eq!(inputs.len(), INPUTS);
        let largest = |values: &[f32]| values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(largest(&inputs[0]) <= 4.0, "signal");
        assert!(largest(&inputs[1]) <= 0.16, "baseline");
        assert!(largest(&inputs[2]) <= 1.16, "gain");
        assert!(largest(&inputs[3]) <= 0.13, "bias");
        assert!(inputs[4].iter().all(|s| *s >= 1.0), "the divisor is >= 1");
        let host = host_stages(&inputs);
        for stage in 0..STAGES {
            let (exact, ..) = stage_exact_and_bound(&inputs, stage);
            assert!(exact.iter().all(|v| v.abs() < 100.0), "stage {stage}");
            let differs = host[stage]
                .iter()
                .zip(&exact)
                .filter(|(h, e)| f64::from(**h) != **e)
                .count();
            // Stage 0 is the exact difference of two nearby values for most elements; every later stage rounds.
            if stage > 0 {
                assert!(
                    differs > inputs[0].len() / 10,
                    "stage {stage} rounds in most elements: {differs}"
                );
            }
        }
        // The last stage of the fused loop body is the stage value.
        for i in 0..inputs[0].len() {
            assert_eq!(
                element(
                    inputs[0][i],
                    inputs[1][i],
                    inputs[2][i],
                    inputs[3][i],
                    inputs[4][i]
                )
                .to_bits(),
                host[STAGES - 1][i].to_bits()
            );
        }
    }

    /// The bound is tight, not vacuous: a few ulps of the value at each stage, not a fraction of it. The composed GPU bound
    /// is measured in epsilons (`2^-23`) of the value's scale (at least 1) at most 20, the CPU's at most 12; the example's
    /// first-order table gives about 5 (CPU) and 13 (GPU) at the last stage.
    #[test]
    fn the_bound_is_a_small_multiple_of_the_roundings_it_covers() {
        let inputs = inputs(4103);
        for stage in 0..STAGES {
            let (exact, bound, bound_gpu) = stage_exact_and_bound(&inputs, stage);
            let ratio = |b: &[f64]| {
                (0..exact.len())
                    .map(|i| b[i] / (f64::from(f32::EPSILON) * exact[i].abs().max(1.0)))
                    .fold(0.0, f64::max)
            };
            for (near, far) in bound.iter().zip(&bound_gpu) {
                assert!(*far > *near && *near > 0.0, "stage {stage}");
            }
            assert!(ratio(&bound) < 12.0, "stage {stage}: {}", ratio(&bound));
            assert!(
                ratio(&bound_gpu) < 20.0,
                "stage {stage}: {}",
                ratio(&bound_gpu)
            );
        }
    }

    /// A correct result passes: the written-order `f32` pipeline under both units, and every stage separately against its
    /// own bound (the composed bound of the stages before it).
    #[test]
    fn a_correct_result_passes_and_every_stage_is_inside_its_own_bound() {
        let n = 4103;
        let case = gen_inputs(W, n);
        let want = reference(W, &case);
        let host = host_stages(&case.pipeline);
        check(W, &case, &want, &host[STAGES - 1]).unwrap();
        for rounding in [Rounding::Nearest, Rounding::Either] {
            check_on(W, &case, &want, &host[STAGES - 1], rounding).unwrap();
        }
        for stage in 0..STAGES {
            let (exact, bound, bound_gpu) = stage_exact_and_bound(&case.pipeline, stage);
            for i in 0..n {
                let err = (f64::from(host[stage][i]) - exact[i]).abs();
                assert!(err <= bound[i], "stage {stage} element {i}: {err}");
                assert!(err <= bound_gpu[i], "stage {stage} element {i}: {err}");
            }
        }
    }

    /// The model of a GPU is not a guess: pipelines run with every operation rounded in a chosen direction (all up, all
    /// down, two pseudo-random mixes) and the division two ULP off are inside the GPU bound at every stage, differ from
    /// the round-to-nearest result, and are outside the CPU bound somewhere (so the two bounds really differ).
    #[test]
    fn a_gpu_that_rounds_either_way_is_inside_the_gpu_bound_at_every_stage() {
        let n = 4103;
        let case = gen_inputs(W, n);
        let want = reference(W, &case);
        let host = host_stages(&case.pipeline);
        let mut outside_cpu = 0usize;
        for pattern in 0..4u64 {
            let run = match pattern {
                0 => gpu_stages(&case.pipeline, |_| true),
                1 => gpu_stages(&case.pipeline, |_| false),
                p => gpu_stages(&case.pipeline, direction(p)),
            };
            assert_ne!(run, host, "pattern {pattern}: the directed run differs");
            for stage in 0..STAGES {
                let (exact, bound, bound_gpu) = stage_exact_and_bound(&case.pipeline, stage);
                for i in 0..n {
                    let err = (f64::from(run[stage][i]) - exact[i]).abs();
                    assert!(
                        err <= bound_gpu[i],
                        "pattern {pattern} stage {stage} element {i}: {err} against {}",
                        bound_gpu[i]
                    );
                    outside_cpu += usize::from(err > bound[i]);
                }
            }
            check_on(W, &case, &want, &run[STAGES - 1], Rounding::Either).unwrap();
        }
        assert!(
            outside_cpu > 0,
            "some directed result is outside the CPU bound, so the units differ"
        );
    }

    /// The largest f32 within `limit` above `exact`, and the next one (just outside).
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

    #[test]
    fn a_value_between_the_two_bounds_passes_on_a_gpu_fails_on_a_cpu_and_one_past_fails_on_both() {
        let n = 300;
        let case = gen_inputs(W, n);
        let want = reference(W, &case);
        let at = 77;
        let (exact, near, far) = (want.exact[at], want.bound[at], want.bound_gpu[at]);
        assert!(far > near && near > 0.0);
        let (inside_gpu, past_gpu) = edge(exact, far);
        assert!(
            f64::from(inside_gpu) - exact > near,
            "an f32 lies between the bounds"
        );
        let mut got = want.want.clone();
        got[at] = inside_gpu;
        check_on(W, &case, &want, &got, Rounding::Either).unwrap();
        assert!(check_on(W, &case, &want, &got, Rounding::Nearest).is_err());
        // One f32 step past the GPU bound fails on a GPU too, and so does the same distance on the low side.
        got[at] = past_gpu;
        let error = check_on(W, &case, &want, &got, Rounding::Either).unwrap_err();
        assert!(
            error.starts_with(&format!("1 of {n} elements are wrong; first at {at}")),
            "{error}"
        );
        got[at] = (exact - 1.5 * far) as f32;
        assert!(check_on(W, &case, &want, &got, Rounding::Either).is_err());
        // Just past the CPU bound fails on a CPU and just inside it passes (the bound is the line, not a hole).
        let (inside_cpu, past_cpu) = edge(exact, near);
        got[at] = inside_cpu;
        check_on(W, &case, &want, &got, Rounding::Nearest).unwrap();
        got[at] = past_cpu;
        assert!(check_on(W, &case, &want, &got, Rounding::Nearest).is_err());
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            got[at] = poison;
            assert!(check_on(W, &case, &want, &got, Rounding::Either).is_err());
        }
    }

    /// The poison, a short output, the unstaged input and an earlier stage's values all fail: a run that skipped a stage
    /// cannot pass.
    #[test]
    fn a_pipeline_that_skipped_a_stage_fails() {
        let n = 1000;
        let case = gen_inputs(W, n);
        let want = reference(W, &case);
        assert!(check(W, &case, &want, &vec![f32::NAN; n]).is_err());
        assert!(check(W, &case, &want, &want.want[1..]).is_err());
        let host = host_stages(&case.pipeline);
        for stage in 0..STAGES - 1 {
            assert!(
                check_on(W, &case, &want, &host[stage], Rounding::Either).is_err(),
                "stage {stage}'s values are not the output"
            );
        }
    }

    /// The hand WGSL has the six entry points the kernels here are, in order, over the three-binding layout its host side
    /// builds, and no hidden extra stage.
    #[test]
    fn the_hand_shader_has_the_six_stages_in_order() {
        let shader = include_str!("../../../shaders/signal_pipeline.wgsl");
        let entries = shader
            .lines()
            .filter_map(|line| line.strip_prefix("fn "))
            .map(|rest| rest.split('(').next().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            entries,
            ["center", "amplify", "shift", "rescale", "energy", "combine"]
        );
        assert_eq!(shader.matches("@compute").count(), STAGES);
    }
}
