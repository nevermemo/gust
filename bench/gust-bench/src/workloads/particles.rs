//! W5: a particle step, 16 explicit-Euler steps of every particle of a struct buffer, `f32`, one invocation per particle.
//! The struct has a nested struct (`position`, `velocity`: pairs of `f32`), a float (`mass`) and two integers (`tag`,
//! `charge`) that the step carries through untouched, so the buffer is a 28-byte std430 element (7 words, alignment 4).
//! The starting point is `examples/particle-step`; the kernel here runs the step 16 times in registers, so one dispatch is
//! the whole case and the bench's one-dispatch model (the batch of 16 passes, the repeated dispatches of a cell) applies
//! unchanged: every dispatch writes the same output from the same input.
//!
//! The step, per particle, with `force` its acceleration and `dt = 0.016`:
//!
//! ```text
//! vx = vx + force.x * dt;  vy = vy + force.y * dt;      // the velocity first,
//! px = px + vx * dt;       py = py + vy * dt;           // then the position from the NEW velocity
//! ```
//!
//! Sixteen of these, written as compound assignments (`vx += force.x * 0.016`). `mass`, `tag` and `charge` are copied by
//! the whole-struct assignment `out[i] = particle`.
//!
//! The example's `energy` output is left out: it is a second, derived output with its own bound, and the step the
//! workload is about is the state update.
//!
//! # Words
//!
//! The harness moves `Vec<f32>` words (see `workloads`). A particle is 7 words, in field order: `px`, `py`, `vx`, `vy`,
//! `mass`, `tag` and `charge` (the two integers as their bit patterns). The acceleration is 2 words per particle in
//! [`Inputs::y`](super::Inputs). Words 0 to 3 are the float fields the bound judges, word 4 (`mass`) and words 5 and 6
//! (the integers) are compared as bits.
//!
//! # The bound (derived here, b3b)
//!
//! Every f32 add and multiply is modelled as `fl(a op b) = (a op b)(1 + delta) + eta`, `|delta| <= u`, `|eta| <= 2^-126`,
//! with the backend's unit roundoff `u` ([`Rounding`](super::Rounding): `2^-24` for the CPU backends, `2^-23` for every
//! GPU backend). A fused multiply-add rounds once and so has only less error than the multiply and add it replaces
//! (the same argument as W2's), so the bound covers a compiler that contracts. The forward error of an expression
//! tree is carried through the 16 steps numerically, in f64, along the f64 trajectory (the exact one differs from it by about
//! `1e-14` relative, which the factor below covers): a value is a pair `(v, e)` with the computed value within `e` of the exact one, and
//!
//! ```text
//! add:  v = a.v + b.v        e = (a.e + b.e)(1 + u) + u |v| + eta
//! mul:  v = a.v * b.v        e = (|a.v| b.e + |b.v| a.e + a.e b.e)(1 + u) + u |v| + eta
//! ```
//!
//! (the computed result of an operation on operands within `e_a` and `e_b` is within `e_in = |a| e_b + |b| e_a + e_a e_b`,
//! respectively `e_a + e_b`, of the exact result, and its own rounding adds at most `u` times its magnitude, which is at
//! most `|v| + e_in`). The inputs are exact. The constant `dt = 0.016` is the nearest f32 on a CPU backend (a Rust
//! literal) and is the same f32 value in the reference; on a GPU backend it is taken as `dt (1 + 2^-23)` away at most,
//! because the compiler of the shader (Slang, then naga or the driver) is only required to give the same f32 or its
//! neighbour; the model lets it be either neighbour (the "either neighbour" reading of `docs/SEMANTICS.md` section 4),
//! as an error `|dt| 2^-23` on the constant's operand.
//!
//! **Grouping.** The CPU adds in the written order (Rust and the twin never reassociate floats), so the recurrence above
//! is its bound. A GPU may regroup a chain of additions with no accuracy condition (`docs/SEMANTICS.md` section 4, WGSL
//! 15.7.5), so for `Rounding::Either` every chain is bounded over *every* grouping: `vx_k = vx_0 + c + ... + c` (`k + 1`
//! terms, `c = fx dt`), likewise `vy_k`, and the final `px = px_0 + vx_1 dt + ... + vx_16 dt` (17 terms), likewise `py`. For
//! a chain of `k + 1` terms `t_i`, each computed within `e_i`, Higham's order-free product bound (section 4.2: a sum
//! of `k` additions in any order or grouping) is
//!
//! ```text
//! e <= (1 + u)^k sum e_i + ((1 + u)^k - 1) sum |t_i| + k (1 + u)^k eta
//! ```
//!
//! (`Chain`). Each `vx_k` is itself such a chain, and its bound enters the product `vx_k dt`, a term of the position chain.
//! The product-form bound is loose, not a guess: it is rigorous for every grouping and never smaller than the bound of the
//! written order. Tests evaluate the step with the additions reversed, in a pairwise tree and largest first, and show all three inside it.
//!
//! The reference is the same recurrence in f64 (`dt` as the exact f64 of the f32 constant), which has its own error,
//! carried with `u = 2^-53`, no `eta` and an exact `dt`. A field's bound is the sum of the f32 model's final `e` and the
//! f64 reference's, and a factor `1 + 1e-9` that covers the f64 evaluation of the bound itself and the difference between
//! the exact trajectory and the f64 one inside its terms (both are of order `1e-14` relative). The check compares each
//! float field of every particle with the f64 trajectory; the integers and the mass must be bit-exact. A backend that
//! fails the bound fails the cell; the bound is never widened to pass.

use gust::gpu;

use super::{Rounding, polynomial::U64};

/// The GUST kernel (workgroup size 256, see `workloads::WORKGROUP`), and the struct types it takes.
#[gpu]
pub mod kernel {
    #[derive(Debug, PartialEq)]
    pub struct Pair {
        pub x: f32,
        pub y: f32,
    }

    #[derive(Debug, PartialEq)]
    pub struct Particle {
        pub position: Pair,
        pub velocity: Pair,
        pub mass: f32,
        pub tag: u32,
        pub charge: i32,
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn step(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<Particle>,
        acceleration: StructuredBuffer<Pair>,
        mut out: RWStructuredBuffer<Particle>,
    ) {
        let i = id.x;
        if i < out.len() && i < input.len() && i < acceleration.len() {
            let particle = input[i];
            let force = acceleration[i];
            let mut vx = particle.velocity.x;
            let mut vy = particle.velocity.y;
            let mut px = particle.position.x;
            let mut py = particle.position.y;
            for _step in 0u32..16u32 {
                vx += force.x * 0.016;
                vy += force.y * 0.016;
                px += vx * 0.016;
                py += vy * 0.016;
            }
            // Preserve mass and the integer metadata with a whole-struct copy.
            out[i] = particle;
            out[i].velocity.x = vx;
            out[i].velocity.y = vy;
            out[i].position.x = px;
            out[i].position.y = py;
        }
    }
}

/// Steps per case (the kernel's loop count).
pub const STEPS: usize = 16;
/// The time step, the f32 the kernels' literal `0.016` is.
pub const DT: f32 = 0.016;
/// Words of one particle.
pub const WORDS: usize = 7;
/// Words of one acceleration.
pub const ACCEL_WORDS: usize = 2;
/// The float fields the bound judges: `px`, `py`, `vx`, `vy`.
pub const FIELDS: usize = 4;

/// Particles `0..n`: 7 words each, then their accelerations, 2 words each. Deterministic and varied, the example's data:
/// the starting position and velocity cycle with the index, the mass takes 7 values, the tag is a hash-like word
/// (`i ^ 0xdeadbeef`), the charge is in `-2..=2`; the acceleration is `(i % 13) / 4` in x and `-9.81` (not an f32) in y.
pub fn inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut particles = Vec::with_capacity(n * WORDS);
    let mut accelerations = Vec::with_capacity(n * ACCEL_WORDS);
    for i in 0..n {
        particles.extend([
            (i % 257) as f32 * 0.125,
            -2.0,
            0.5,
            (i % 31) as f32 * -0.0625,
            1.0 + (i % 7) as f32 * 0.25,
            f32::from_bits(i as u32 ^ 0xdead_beef),
            f32::from_bits(((i % 5) as i32 - 2) as u32),
        ]);
        accelerations.extend([(i % 13) as f32 * 0.25, -9.81]);
    }
    (particles, accelerations)
}

/// The typed particles of 7-word groups (`mass` is the f32 word, the integers are the words' bit patterns).
pub fn to_particles(words: &[f32]) -> Vec<kernel::Particle> {
    words
        .as_chunks::<WORDS>()
        .0
        .iter()
        .map(|w| kernel::Particle {
            position: kernel::Pair { x: w[0], y: w[1] },
            velocity: kernel::Pair { x: w[2], y: w[3] },
            mass: w[4],
            tag: w[5].to_bits(),
            charge: w[6].to_bits() as i32,
        })
        .collect()
}

/// The typed accelerations of 2-word groups.
pub fn to_pairs(words: &[f32]) -> Vec<kernel::Pair> {
    words
        .as_chunks::<ACCEL_WORDS>()
        .0
        .iter()
        .map(|w| kernel::Pair { x: w[0], y: w[1] })
        .collect()
}

/// The 7-word groups of typed particles.
pub fn from_particles(particles: &[kernel::Particle]) -> Vec<f32> {
    let mut words = Vec::with_capacity(particles.len() * WORDS);
    for p in particles {
        words.extend([
            p.position.x,
            p.position.y,
            p.velocity.x,
            p.velocity.y,
            p.mass,
            f32::from_bits(p.tag),
            f32::from_bits(p.charge as u32),
        ]);
    }
    words
}

/// The rayon / serial / reference step of one particle: `particle` (7 words) and `force` (2 words) in, the stepped
/// particle (7 words) in `out`. The arithmetic of the kernel, a multiply and an add per update, never fused.
#[inline(always)]
pub fn step_words(particle: &[f32], force: &[f32], out: &mut [f32]) {
    let (mut px, mut py, mut vx, mut vy) = (particle[0], particle[1], particle[2], particle[3]);
    for _ in 0..STEPS {
        vx += force[0] * DT;
        vy += force[1] * DT;
        px += vx * DT;
        py += vy * DT;
    }
    out[..WORDS].copy_from_slice(&particle[..WORDS]);
    out[0] = px;
    out[1] = py;
    out[2] = vx;
    out[3] = vy;
}

/// A computed value and a bound on its distance from the exact one (see the module notes).
#[derive(Clone, Copy)]
struct Tracked {
    val: f64,
    err: f64,
}

/// One backend model: the unit roundoff, the underflow term and the error of the constant `dt` (`dt_err`, absolute).
#[derive(Clone, Copy)]
struct Model {
    u: f64,
    eta: f64,
    dt_err: f64,
    /// Bound every addition chain over any grouping (the GPU: `docs/SEMANTICS.md` section 4 lets it regroup), not in the
    /// written order (the CPU).
    regroup: bool,
}

/// An addition chain `t_0 + ... + t_k` judged for every grouping: Higham's order-free product bound
/// `(1+u)^k sum e_i + ((1+u)^k - 1) sum |t_i|`, plus `k` underflow terms. Terms are pushed one at a time.
#[derive(Clone, Copy, Default)]
struct Chain {
    val: f64,
    err_sum: f64,
    abs_sum: f64,
    terms: i32,
}

impl Chain {
    fn push(&mut self, t: Tracked) {
        self.val += t.val;
        self.err_sum += t.err;
        self.abs_sum += t.val.abs();
        self.terms += 1;
    }

    fn tracked(self, model: Model) -> Tracked {
        let k = self.terms - 1;
        let growth = (1.0 + model.u).powi(k);
        Tracked {
            val: self.val,
            err: growth * self.err_sum
                + (growth - 1.0) * self.abs_sum
                + f64::from(k) * growth * model.eta,
        }
    }
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

    /// The 16 steps of one particle; the four float fields `[px, py, vx, vy]` with their forward errors.
    fn run(self, particle: &[f32], force: &[f32]) -> [Tracked; FIELDS] {
        let exact = |v: f32| Tracked {
            val: f64::from(v),
            err: 0.0,
        };
        let dt = Tracked {
            val: f64::from(DT),
            err: self.dt_err,
        };
        let (mut px, mut py) = (exact(particle[0]), exact(particle[1]));
        let (mut vx, mut vy) = (exact(particle[2]), exact(particle[3]));
        let (fx, fy) = (exact(force[0]), exact(force[1]));
        if self.regroup {
            let chain = |first: Tracked| {
                let mut chain = Chain::default();
                chain.push(first);
                chain
            };
            let (mut cpx, mut cpy, mut cvx, mut cvy) = (chain(px), chain(py), chain(vx), chain(vy));
            for _ in 0..STEPS {
                cvx.push(self.mul(fx, dt));
                cvy.push(self.mul(fy, dt));
                vx = cvx.tracked(self);
                vy = cvy.tracked(self);
                cpx.push(self.mul(vx, dt));
                cpy.push(self.mul(vy, dt));
            }
            px = cpx.tracked(self);
            py = cpy.tracked(self);
        } else {
            for _ in 0..STEPS {
                vx = self.add(vx, self.mul(fx, dt));
                vy = self.add(vy, self.mul(fy, dt));
                px = self.add(px, self.mul(vx, dt));
                py = self.add(py, self.mul(vy, dt));
            }
        }
        [px, py, vx, vy]
    }
}

/// The model of a backend of the given rounding: its unit roundoff, `eta = 2^-126`, and for a GPU the either-neighbour
/// error of the constant.
fn model_of(rounding: Rounding) -> Model {
    let u = rounding.unit();
    Model {
        u,
        eta: f64::from(f32::MIN_POSITIVE),
        dt_err: match rounding {
            Rounding::Nearest => 0.0,
            Rounding::Either => f64::from(DT) * u,
        },
        regroup: rounding == Rounding::Either,
    }
}

/// The f64 reference's own model: `u = 2^-53`, no underflow term, an exact `dt`.
const REFERENCE: Model = Model {
    u: U64,
    eta: 0.0,
    dt_err: 0.0,
    regroup: false,
};

/// Covers the f64 evaluation of the bound itself and the exact-versus-f64 trajectory inside its terms (see the notes).
pub const SLACK: f64 = 1.0 + 1e-9;

/// The f64 trajectory of every float field and the derived absolute bounds, for a CPU backend (`u = 2^-24`) and for a GPU
/// (`u = 2^-23`): three vectors of `4 n` values, particle by particle in the order `px, py, vx, vy`.
pub fn exact_and_bound(particles: &[f32], forces: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = forces.len() / ACCEL_WORDS;
    let (near, gpu) = (model_of(Rounding::Nearest), model_of(Rounding::Either));
    let mut exact = Vec::with_capacity(n * FIELDS);
    let mut bound = Vec::with_capacity(n * FIELDS);
    let mut bound_gpu = Vec::with_capacity(n * FIELDS);
    for (particle, force) in particles
        .as_chunks::<WORDS>()
        .0
        .iter()
        .zip(forces.as_chunks::<ACCEL_WORDS>().0.iter())
    {
        let reference = REFERENCE.run(particle, force);
        let (a, b) = (near.run(particle, force), gpu.run(particle, force));
        for field in 0..FIELDS {
            exact.push(reference[field].val);
            bound.push((a[field].err + reference[field].err) * SLACK);
            bound_gpu.push((b[field].err + reference[field].err) * SLACK);
        }
    }
    (exact, bound, bound_gpu)
}

/// The reference words: the f64 trajectory rounded to f32 in the float fields, the input words everywhere else.
pub fn expected_words(particles: &[f32], exact: &[f64]) -> Vec<f32> {
    let mut want = particles.to_vec();
    for (i, word) in want.as_chunks_mut::<WORDS>().0.iter_mut().enumerate() {
        for field in 0..FIELDS {
            word[field] = exact[i * FIELDS + field] as f32;
        }
    }
    want
}

#[cfg(test)]
mod tests {
    use gust::GpuPod;

    use super::*;
    use crate::workloads::{Workload, check, check_on, gen_inputs, reference};

    /// The hand WGSL struct is the same std430 element as GUST's: 28 bytes, alignment 4, stride 28, fields in the same
    /// order at the same offsets. GUST's side is read from the `GpuPod` layout the macro derives, the hand side from the
    /// shader's text.
    #[test]
    fn the_hand_wgsl_layout_matches_gusts_derived_layout() {
        use gust::TypeLayoutKind;
        let layout = <kernel::Particle as GpuPod>::LAYOUT;
        assert_eq!(
            (layout.size, layout.alignment, layout.stride),
            (28, 4, 28),
            "the WORDS * 4 bytes the hand shader and the word vectors assume"
        );
        let TypeLayoutKind::Struct(fields) = layout.kind else {
            panic!("a struct");
        };
        let gust = fields
            .iter()
            .map(|f| (f.name, f.offset, f.ty.size))
            .collect::<Vec<_>>();
        assert_eq!(
            gust,
            [
                ("position", 0, 8),
                ("velocity", 8, 8),
                ("mass", 16, 4),
                ("tag", 20, 4),
                ("charge", 24, 4)
            ]
        );
        let pair = <kernel::Pair as GpuPod>::LAYOUT;
        assert_eq!((pair.size, pair.alignment, pair.stride), (8, 4, 8));
        // The hand shader declares the same structs in the same field order, with the same scalar types.
        let shader = include_str!("../../../shaders/particles.wgsl");
        let declared = |text: &str| {
            let at = shader
                .find(text)
                .unwrap_or_else(|| panic!("`{text}` in the shader"));
            shader[at..]
                .split('}')
                .next()
                .unwrap()
                .split_whitespace()
                .collect::<String>()
        };
        assert_eq!(declared("struct Pair"), "structPair{x:f32,y:f32,");
        assert_eq!(
            declared("struct Particle"),
            "structParticle{position:Pair,velocity:Pair,mass:f32,tag:u32,charge:i32,"
        );
        assert_eq!(layout.size as usize, WORDS * 4);
        assert_eq!(pair.size as usize, ACCEL_WORDS * 4);
    }

    /// N6: the committed Slang reflection of the step kernel (`gust-bench-pre/gust-artifacts`, the layout the shader compiler
    /// itself produced) says what the macro-derived layout and the hand shader assume: stride 28 and these offsets.
    #[test]
    fn the_committed_reflection_of_the_step_kernel_has_the_layout_the_hand_shader_assumes() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../gust-bench-pre/gust-artifacts");
        let mut found = 0;
        for entry in std::fs::read_dir(&dir).expect("the committed artifacts") {
            let path = entry.unwrap().path();
            if !path.to_string_lossy().ends_with(".reflection.json") {
                continue;
            }
            let json: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            if json["entryPoint"] != "gpu_kernel_step" {
                continue;
            }
            found += 1;
            let input = &json["parameters"][0]["type"]["resultType"];
            assert_eq!(
                (
                    input["size"].as_u64(),
                    input["alignment"].as_u64(),
                    input["stride"].as_u64()
                ),
                (Some(28), Some(4), Some(28))
            );
            let fields = input["fields"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| {
                    (
                        f["name"].as_str().unwrap().to_owned(),
                        f["binding"]["offset"].as_u64().unwrap(),
                        f["binding"]["size"].as_u64().unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            let want = [
                ("position", 0, 8),
                ("velocity", 8, 8),
                ("mass", 16, 4),
                ("tag", 20, 4),
                ("charge", 24, 4),
            ]
            .map(|(n, o, s)| (n.to_owned(), o, s));
            assert_eq!(fields, want);
        }
        assert_eq!(found, 1, "one committed reflection for the step kernel");
    }

    #[test]
    fn the_words_round_trip_through_the_typed_structs() {
        let (x, y) = inputs(1000);
        assert_eq!((x.len(), y.len()), (1000 * WORDS, 1000 * ACCEL_WORDS));
        let typed = to_particles(&x);
        assert_eq!(typed.len(), 1000);
        let back = from_particles(&typed);
        assert_eq!(
            back.iter().map(|w| w.to_bits()).collect::<Vec<_>>(),
            x.iter().map(|w| w.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(to_pairs(&y).len(), 1000);
        assert_eq!(typed[3].tag, 3 ^ 0xdead_beef);
        assert_eq!(typed[3].charge, 3 - 2);
        assert_eq!(typed[1].mass, 1.25);
    }

    /// The domain the derivation needs: finite values nowhere near overflow or the subnormals, and an f32 step that does
    /// round (the f32 trajectory is not the f64 one).
    #[test]
    fn the_domain_holds_and_the_f32_step_rounds() {
        let (x, y) = inputs(4103);
        let (exact, bound, bound_gpu) = exact_and_bound(&x, &y);
        assert_eq!(exact.len(), 4103 * FIELDS);
        assert!(exact.iter().all(|v| v.abs() < 1000.0));
        for (near, far) in bound.iter().zip(&bound_gpu) {
            assert!(*far > *near && *far < 4.0 * *near && *near > 0.0);
        }
        let mut out = vec![0.0f32; WORDS];
        let mut differs = 0;
        for i in 0..4103 {
            step_words(&x[i * WORDS..], &y[i * ACCEL_WORDS..], &mut out);
            for field in 0..FIELDS {
                differs += usize::from(f64::from(out[field]) != exact[i * FIELDS + field]);
            }
        }
        assert!(
            differs > 4103,
            "the f32 step rounds in most fields: {differs}"
        );
    }

    /// The bound is tight, not vacuous: a few ulps of the value, not a fraction of it.
    #[test]
    fn the_bound_is_a_small_multiple_of_the_roundings_it_covers() {
        let (x, y) = inputs(4103);
        let (exact, bound, bound_gpu) = exact_and_bound(&x, &y);
        let ratio = |b: &[f64]| {
            (0..exact.len())
                .map(|i| b[i] / (f64::from(f32::EPSILON) * exact[i].abs().max(1.0)))
                .fold(0.0, f64::max)
        };
        // 16 steps of 4 operations each: measured at about 8 (CPU) and 20 (GPU, every grouping) epsilons of the scale at
        // most, so 12 and 28 mean a few rounding errors per step and not a fraction of the value.
        assert!(ratio(&bound) < 12.0, "{}", ratio(&bound));
        assert!(ratio(&bound_gpu) < 28.0, "{}", ratio(&bound_gpu));
    }

    /// The written-order f32 step of every particle of `inputs` into `out`.
    fn step_all(inputs: &crate::workloads::Inputs, out: &mut [f32]) {
        for i in 0..out.len() / WORDS {
            step_words(
                &inputs.x[i * WORDS..],
                &inputs.y[i * ACCEL_WORDS..],
                &mut out[i * WORDS..(i + 1) * WORDS],
            );
        }
    }

    /// Sums `terms` in a reversed order, or as a pairwise tree, in f32 (a grouping the written order does not use).
    fn grouped_sum(terms: &[f32], tree: bool) -> f32 {
        if !tree {
            return terms.iter().rev().fold(0.0, |sum, t| sum + t);
        }
        match terms.len() {
            1 => terms[0],
            len => grouped_sum(&terms[..len / 2], true) + grouped_sum(&terms[len / 2..], true),
        }
    }

    /// The step with every addition chain regrouped: `reversed` (last term first), `tree` (pairwise), or `largest` first.
    fn regrouped_step(p: &[f32], a: &[f32], tree: bool, largest_first: bool) -> [f32; FIELDS] {
        let order = |mut terms: Vec<f32>| {
            if largest_first {
                terms.sort_by(|x, y| y.abs().total_cmp(&x.abs()));
                terms.iter().fold(0.0, |sum, t| sum + t)
            } else {
                grouped_sum(&terms, tree)
            }
        };
        let (cx, cy) = (a[0] * DT, a[1] * DT);
        let (mut vxs, mut vys) = (vec![], vec![]);
        for k in 1..=STEPS {
            let mut tx = vec![p[2]];
            tx.extend(std::iter::repeat_n(cx, k));
            let mut ty = vec![p[3]];
            ty.extend(std::iter::repeat_n(cy, k));
            vxs.push(order(tx));
            vys.push(order(ty));
        }
        let mut tx = vec![p[0]];
        tx.extend(vxs.iter().map(|v| v * DT));
        let mut ty = vec![p[1]];
        ty.extend(vys.iter().map(|v| v * DT));
        [order(tx), order(ty), vxs[STEPS - 1], vys[STEPS - 1]]
    }

    /// M1: a GPU may regroup the chains (`docs/SEMANTICS.md` section 4); reversed, pairwise and largest-first evaluations
    /// are inside the GPU bound (they really do differ from the written order), and the GPU bound is never below the
    /// written-order one.
    #[test]
    fn a_regrouped_step_is_inside_the_gpu_bound() {
        let w = Workload::Particles;
        let n = 4103;
        let inputs = gen_inputs(w, n);
        let reference = reference(w, &inputs);
        let mut plain = vec![0.0f32; n * WORDS];
        step_all(&inputs, &mut plain);
        for (tree, largest) in [(false, false), (true, false), (false, true)] {
            let mut got = inputs.x.clone();
            for i in 0..n {
                let r = regrouped_step(
                    &inputs.x[i * WORDS..],
                    &inputs.y[i * ACCEL_WORDS..],
                    tree,
                    largest,
                );
                got[i * WORDS..i * WORDS + FIELDS].copy_from_slice(&r);
            }
            assert_ne!(
                got, plain,
                "tree={tree} largest={largest}: the grouping changes some rounding"
            );
            check_on(w, &inputs, &reference, &got, Rounding::Either).unwrap();
        }
        // The written-order model of the GPU (the bound before the grouping fix) is never above the grouped one.
        let written = Model {
            regroup: false,
            ..model_of(Rounding::Either)
        };
        for i in 0..n {
            let (p, f) = (&inputs.x[i * WORDS..], &inputs.y[i * ACCEL_WORDS..]);
            let (g, w2) = (model_of(Rounding::Either).run(p, f), written.run(p, f));
            for field in 0..FIELDS {
                assert!(
                    g[field].err >= w2[field].err * (1.0 - 1e-12),
                    "{i} {field}: {} against {}",
                    g[field].err,
                    w2[field].err
                );
            }
        }
    }

    #[test]
    fn the_unfused_and_the_fused_step_are_inside_the_bound_under_both_units() {
        let w = Workload::Particles;
        let n = 4103;
        let inputs = gen_inputs(w, n);
        let reference = reference(w, &inputs);
        let mut plain = vec![0.0f32; n * WORDS];
        let mut fused = vec![0.0f32; n * WORDS];
        for i in 0..n {
            let (p, a) = (&inputs.x[i * WORDS..], &inputs.y[i * ACCEL_WORDS..]);
            step_words(p, a, &mut plain[i * WORDS..(i + 1) * WORDS]);
            let (mut px, mut py, mut vx, mut vy) = (p[0], p[1], p[2], p[3]);
            for _ in 0..STEPS {
                vx = a[0].mul_add(DT, vx);
                vy = a[1].mul_add(DT, vy);
                px = vx.mul_add(DT, px);
                py = vy.mul_add(DT, py);
            }
            fused[i * WORDS..(i + 1) * WORDS].copy_from_slice(&p[..WORDS]);
            fused[i * WORDS..i * WORDS + FIELDS].copy_from_slice(&[px, py, vx, vy]);
        }
        assert_ne!(plain, fused, "fusing changes some rounding");
        check(w, &inputs, &reference, &plain).unwrap();
        check(w, &inputs, &reference, &fused).unwrap();
        for rounding in [Rounding::Nearest, Rounding::Either] {
            check_on(w, &inputs, &reference, &plain, rounding).unwrap();
            check_on(w, &inputs, &reference, &fused, rounding).unwrap();
        }
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
        let w = Workload::Particles;
        let n = 300;
        let inputs = gen_inputs(w, n);
        let reference = reference(w, &inputs);
        let (particle, field) = (77, 3);
        let word = particle * WORDS + field;
        let at = particle * FIELDS + field;
        let (exact, near, far) = (
            reference.exact[at],
            reference.bound[at],
            reference.bound_gpu[at],
        );
        assert!(far > near && near > 0.0);
        let (inside_gpu, past_gpu) = edge(exact, far);
        assert!(
            f64::from(inside_gpu) - exact > near,
            "an f32 lies between the bounds"
        );
        let mut got = reference.want.clone();
        got[word] = inside_gpu;
        check_on(w, &inputs, &reference, &got, Rounding::Either).unwrap();
        assert!(check_on(w, &inputs, &reference, &got, Rounding::Nearest).is_err());
        // One f32 step past the GPU bound fails on a GPU too, on the low side as well.
        got[word] = past_gpu;
        let error = check_on(w, &inputs, &reference, &got, Rounding::Either).unwrap_err();
        assert!(
            error.starts_with(&format!(
                "1 of {} elements are wrong; first at {word}",
                n * WORDS
            )),
            "{error}"
        );
        got[word] = (exact - 1.5 * far) as f32;
        assert!(check_on(w, &inputs, &reference, &got, Rounding::Either).is_err());
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            got[word] = poison;
            assert!(check_on(w, &inputs, &reference, &got, Rounding::Either).is_err());
        }
    }

    /// The integers and the mass are exact: one bit off, in either direction, fails; so does a swapped field.
    #[test]
    fn the_integer_fields_and_the_mass_must_be_bit_exact() {
        let w = Workload::Particles;
        let n = 100;
        let inputs = gen_inputs(w, n);
        let reference = reference(w, &inputs);
        check(w, &inputs, &reference, &reference.want).unwrap();
        for field in [4usize, 5, 6] {
            let word = 42 * WORDS + field;
            for delta in [1u32, u32::MAX] {
                let mut got = reference.want.clone();
                got[word] = f32::from_bits(got[word].to_bits().wrapping_add(delta));
                assert!(
                    check(w, &inputs, &reference, &got).is_err(),
                    "field {field}"
                );
            }
        }
        // Poison (every word NaN) fails, and a short output fails.
        assert!(check(w, &inputs, &reference, &vec![f32::NAN; n * WORDS]).is_err());
        assert!(check(w, &inputs, &reference, &reference.want[1..]).is_err());
        // The step really moved the particles: the unstepped input fails.
        assert!(check(w, &inputs, &reference, &inputs.x).is_err());
    }
}
