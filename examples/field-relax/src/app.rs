//! Three scalar fields and an exact integer channel relaxing on one wrapping grid, each step in one kernel.
//!
//! The program is a numerical workload of the kind of heat or concentration models: every field spreads to its four
//! neighbours at its own rate and fades at its own rate, on a periodic `SIDE x SIDE` grid, for `STEPS` steps. One
//! `grid` kernel, `relax`, updates the three `f32` fields and the integer channel of a cell together. It uses module
//! level `const` items (m11.g3), `rem_euclid` and `mul_add` (m11.math3), and one `Uniform<T>` struct for the
//! parameters of all three fields (no parameter buffer, no launch per field).
//!
//! 1. The fields live in one `f32` buffer of `FIELDS * SIDE * SIDE` (field-major) and the integer channel in one `u32`
//!    buffer of `SIDE * SIDE`. Each has two copies, `a` and `b`; step 1 reads `a` and writes `b`, step 2 reads `b` and
//!    writes `a`, and so on.
//! 2. The float update of field `f` (spread `d_f`, keep `k_f` = 1 minus its decay), with `n s e w` the four wrapped
//!    neighbours: `next = k_f * mul_add(d_f * 0.25, n + s + e + w, (1 - d_f) * c)`.
//! 3. The integer update: a cell keeps `c - 4 * (c >> 3)` and receives `x >> 3` from each of its four neighbours `x`.
//!    Every cell gives away exactly what the others receive, so the sum of the channel never changes.
//!
//! # The parameter struct
//!
//! A uniform is `Std140V1` (`docs/SEMANTICS.md`, "Uniforms"): a struct is rounded up to 16 bytes and may have no
//! padding, so six `f32` (24 bytes) are written with two explicit `f32` pad fields (32 bytes). A pair of `Vec4`
//! lanes (`d` and `k`) would avoid the pad fields; the scalar form keeps each name next to its field index.
//!
//! # The exact channel
//!
//! Integer arithmetic has no rounding, so the channel must be equal bit for bit on the host reference, the CPU twin and
//! the GPU after every step, and its sum (a `u64` on the host) must equal the starting sum after every step. The
//! inputs are below `2^16` plus one spike of `2^24`; a step raises the largest cell by at most 3 (a cell keeps at most
//! `c / 2 + 7 / 2` and receives at most half of the largest cell), so no `u32` overflows in 16 steps.
//!
//! # The float bound
//!
//! Domain: finite, non-negative inputs, no overflow, no subnormal. SEMANTICS keeps subnormals on the twin and allows
//! the GPU to flush them (KM-08; flushing was observed), so the derivation does not cover them and the inputs keep
//! away: every input is at least `2^-8` and below 4, and each step keeps at least `k (1 - d) >= 0.49` of the smallest
//! cell, so after 16 steps every value is above `2^-8 * k^16` (the real factor per step is `k`; `0.49^16`, about
//! `5e-8`, is a loose floor), far above the smallest normal `f32` (a test asserts it on the reference). The parameters `d_f` and `k_f` are exact in `f32` (dyadic), so `1 - d_f` is
//! exact and `d_f * 0.25` is exact (a power of two).
//!
//! The model is that of `docs/SEMANTICS.md` ("Deriving a tolerance"): `u = 2^-24`; on the GPU `+` and `*` land on
//! either neighbour of the exact result, an error of at most `2u` of the value (one ULP is at most `2^-23` of a
//! normal value). All operands are non-negative, so there is no cancellation, and the relative error of a sum of
//! non-negative terms is at most the largest relative error among its terms. Let `1 + r` be the factor the cell and
//! its four neighbours carry from the previous step (each within `1 +- r` of the exact value). One step is:
//!
//! - `n + s + e + w`: three additions in any grouping (all terms are non-negative, so any grouping is at most three
//!   additions deep): `(1 + r)(1 + 2u)^3`.
//! - `d * 0.25` is exact; the product with the sum is one more rounding: `(1 + r)(1 + 2u)^4`.
//! - `(1 - d) * c` is one rounding of a product by an exact constant: `(1 + r)(1 + 2u)`. The larger of the two addends
//!   is the first.
//! - `mul_add` is held to the unfused form SEMANTICS states for it (a fused one rounds once and does less): the
//!   product, then the sum. The sum adds one ULP "at the larger of the exact and the actual result", which is at most
//!   `2u (1 + r_out)` of the exact result; `r_out` stays far below `2^-12`, so `2u (1 + 2^-12)` is charged.
//! - the multiplication by `k`: one more rounding, `(1 + 2u)`.
//!
//! So `1 + r' = (1 + r) * f` with `f = (1 + 2u)^5 * (1 + 2u (1 + 2^-12))`, about `12u = 7.2e-7` per step. The
//! relative error is carried, not amplified, whatever the weight sum: the stencil is a non-negative combination of
//! the inputs, so a relative error `r` in every input is a relative error `r` in the exact result of the combination
//! (the weights sum to `k_f ((1 - d_f) + 4 * d_f / 4) = k_f`, and `k_f <= 1` matters only for an absolute bound, which
//! this derivation does not use). After `S` steps from exact inputs `r_S = f^S - 1`; [`float_bound`] is
//! `f^S - 1 + 2^-40`, the last term for the `f64` reference (its own error over about 100 roundings is below `2^-47`) and the arithmetic of the bound.
//! A correctly rounded twin has half the error charged on the GPU, a fused `mul_add` less; one bound covers every
//! device. Two checks use it:
//!
//! - the per-step bound `float_bound(1)`, asserted for every step against one `f64` step applied to the *same
//!   device's output of the step before* (this judges step 1 and step 2, which read different buffers, separately);
//! - the `S`-step bound `float_bound(S)` for every step `S`, against the `f64` reference evolved from the inputs.
//!
//! Both are asserted for every cell of every field on the twin and the GPU. The worst observed error against the
//! bound is printed, and is not asserted.
//!
//! # Why two modules
//!
//! `SIDE` is a module constant, so one `#[gpu_module]` module has one grid size (`MASK = SIDE - 1` and `CELLS = SIDE * SIDE`
//! are constant expressions of it, D210). The macro below takes the side and writes the module and the
//! host code that drives it once per size: `big` (512, the run of `main` and the GPU tests) and, under `cfg(test)`,
//! `small` (32, the unit tests on the twin).

#[cfg(any(test, gust_dual))]
use gust::cpu::CpuDevice;
use gust::{Independent, gpu_module};
use gust_wgpu::{DeviceHandle, GpuVec, HeadlessDevice, default_device};

/// The number of fields. The kernel and the parameter struct spell out three.
pub const FIELDS: usize = 3;
/// Steps of every run: even, so the result ends in buffer `a`.
pub const STEPS: usize = 16;
/// Spread rate `d` and keep factor `k` (1 minus decay) of each field. All are exact in `f32`.
pub const RATES: [(f32, f32); FIELDS] = [
    (0.5, 1.0 - 1.0 / 1024.0),
    (0.375, 1.0 - 1.0 / 256.0),
    (0.1875, 1.0 - 1.0 / 64.0),
];
/// The seed of each field's inputs, and of the integer channel.
const SEEDS: [u32; FIELDS + 1] = [0x1234_5678, 0x9abc_def0, 0x0bad_cafe, 0xfeed_f00d];

// ---- Host side shared by every size: inputs and references, plain Rust, not the `#[gpu_module]` bodies. ----

/// A multiply-xorshift finaliser: "lowbias32" from Chris Wellons' hash-prospector
/// (https://github.com/skeeto/hash-prospector; public domain, Unlicense), as in `examples/hash-rng`.
fn mix(x: u32) -> u32 {
    let mut h = x ^ (x >> 16);
    h = h.wrapping_mul(2146121005);
    h ^= h >> 15;
    h = h.wrapping_mul(2221713035);
    h ^ (h >> 16)
}

/// A hash of a seed and an index (the seed is mixed first, as in `examples/hash-rng`).
fn hash(seed: u32, index: u32) -> u32 {
    mix(mix(seed) ^ index)
}

/// The starting state of a `side x side` grid.
pub struct Inputs {
    pub side: usize,
    /// `FIELDS * side * side` values, field-major.
    pub fields: Vec<f32>,
    /// `side * side` counts.
    pub counts: Vec<u32>,
}

/// Inputs from the hash: field values in `[2^-8, 4)` (not symmetric: each field has its own seed, and the value
/// depends on the cell through the hash), counts below `2^16` plus one spike of `2^24` in the last column of row 0,
/// so the wrap at the edge is exercised.
pub fn make_inputs(side: usize) -> Inputs {
    let cells = side * side;
    let mut fields = Vec::with_capacity(FIELDS * cells);
    for seed in SEEDS.iter().take(FIELDS) {
        for i in 0..cells {
            let m = f64::from(hash(*seed, i as u32) >> 8);
            fields.push((1.0 / 256.0 + m / 16_777_216.0 * 3.5) as f32);
        }
    }
    let mut counts: Vec<u32> = (0..cells)
        .map(|i| hash(SEEDS[FIELDS], i as u32) >> 16)
        .collect();
    counts[side - 1] += 1 << 24;
    Inputs {
        side,
        fields,
        counts,
    }
}

/// One step of the three fields in `f64`, written from the formula with `%` for the wrap.
fn reference_step_fields(side: usize, now: &[f64]) -> Vec<f64> {
    let cells = side * side;
    let mut next = vec![0.0f64; now.len()];
    for (f, (d, k)) in RATES.iter().enumerate() {
        let (d, k) = (f64::from(*d), f64::from(*k));
        for y in 0..side {
            for x in 0..side {
                let at = |x: usize, y: usize| now[f * cells + y * side + x];
                let sum = at(x, (y + side - 1) % side)
                    + at(x, (y + 1) % side)
                    + at((x + 1) % side, y)
                    + at((x + side - 1) % side, y);
                next[f * cells + y * side + x] = k * ((1.0 - d) * at(x, y) + d / 4.0 * sum);
            }
        }
    }
    next
}

/// One step of the integer channel: keep `c - 4 * (c >> 3)`, receive `x >> 3` from each neighbour.
pub fn reference_step_counts(side: usize, now: &[u32]) -> Vec<u32> {
    let mut next = vec![0u32; now.len()];
    for y in 0..side {
        for x in 0..side {
            let at = |x: usize, y: usize| now[y * side + x];
            let c = at(x, y);
            next[y * side + x] = c - 4 * (c >> 3)
                + (at(x, (y + side - 1) % side) >> 3)
                + (at(x, (y + 1) % side) >> 3)
                + (at((x + 1) % side, y) >> 3)
                + (at((x + side - 1) % side, y) >> 3);
        }
    }
    next
}

/// The relative error bound after `steps` steps (derivation in the module documentation).
pub fn float_bound(steps: usize) -> f64 {
    let u = 1.0 / 16_777_216.0f64;
    let step = (1.0 + 2.0 * u).powi(5) * (1.0 + 2.0 * u * (1.0 + 1.0 / 4096.0));
    step.powi(steps as i32) - 1.0 + 1.0 / 1_099_511_627_776.0
}

/// The largest relative error of `got` against `want`, or the first element outside `bound` as a message.
pub fn check_against(want: &[f64], got: &[f32], bound: f64) -> Result<f64, String> {
    if want.len() != got.len() {
        return Err(format!("{} results for {} cells", got.len(), want.len()));
    }
    let mut worst = 0.0f64;
    for (i, (want, got)) in want.iter().zip(got).enumerate() {
        let got = f64::from(*got);
        let error = (got - want).abs() / want;
        // A NaN is outside the bound too.
        if error.is_nan() || error > bound {
            return Err(format!(
                "index {i}: got {got}, reference {want}, relative error {error} over the bound {bound}"
            ));
        }
        worst = worst.max(error);
    }
    Ok(worst)
}

/// The values after every step: index 0 is the inputs, index `s` the state after `s` steps.
pub struct Trace {
    pub fields: Vec<Vec<f32>>,
    pub counts: Vec<Vec<u32>>,
}

/// What [`verify`] measured.
pub struct Report {
    /// The largest relative error against the `f64` reference evolved from the inputs, over every step.
    pub worst_cumulative: f64,
    /// The largest relative error of one step against an `f64` step from the device's own previous state.
    pub worst_single_step: f64,
    /// The largest relative error of step 1 and of step 2 against their own `f64` step (the two buffer directions).
    pub worst_step_1: f64,
    pub worst_step_2: f64,
}

/// Checks a whole trace: the integer channel exactly and the fields against both bounds, every step, every cell.
pub fn verify(name: &str, inputs: &Inputs, trace: &Trace) -> Result<Report, String> {
    let side = inputs.side;
    if trace.fields.len() != STEPS + 1 || trace.counts.len() != STEPS + 1 {
        return Err(format!("{name}: a trace of {} states", trace.fields.len()));
    }
    if trace.fields[0] != inputs.fields || trace.counts[0] != inputs.counts {
        return Err(format!("{name}: the trace does not start at the inputs"));
    }
    let total: u64 = inputs.counts.iter().map(|&c| u64::from(c)).sum();
    let mut want_counts = inputs.counts.clone();
    let mut want_fields: Vec<f64> = inputs.fields.iter().map(|&v| f64::from(v)).collect();
    let mut report = Report {
        worst_cumulative: 0.0,
        worst_single_step: 0.0,
        worst_step_1: 0.0,
        worst_step_2: 0.0,
    };
    for step in 1..=STEPS {
        // Exact channel: equal to the host reference, and the sum unchanged.
        want_counts = reference_step_counts(side, &want_counts);
        let got = &trace.counts[step];
        if let Some(i) = (0..got.len()).find(|&i| got[i] != want_counts[i]) {
            return Err(format!(
                "{name}: step {step}: count at cell {i} (x {}, y {}): got {}, reference {}",
                i % side,
                i / side,
                got[i],
                want_counts[i]
            ));
        }
        let sum: u64 = got.iter().map(|&c| u64::from(c)).sum();
        if sum != total {
            return Err(format!(
                "{name}: step {step}: sum of the counts {sum}, started at {total}"
            ));
        }
        // Float, one step: from the device's own previous state.
        let previous: Vec<f64> = trace.fields[step - 1]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let single = reference_step_fields(side, &previous);
        let local = check_against(&single, &trace.fields[step], float_bound(1))
            .map_err(|message| format!("{name}: step {step}, one-step bound: {message}"))?;
        report.worst_single_step = report.worst_single_step.max(local);
        match step {
            1 => report.worst_step_1 = local,
            2 => report.worst_step_2 = local,
            _ => {}
        }
        // Float, from the inputs.
        want_fields = reference_step_fields(side, &want_fields);
        let cumulative = check_against(&want_fields, &trace.fields[step], float_bound(step))
            .map_err(|message| format!("{name}: step {step}, {step}-step bound: {message}"))?;
        report.worst_cumulative = report.worst_cumulative.max(cumulative);
    }
    Ok(report)
}

/// Writes, per size, the `#[gpu_module]` module and the host code that drives it.
macro_rules! field_program {
    ($name:ident, $side:tt) => {
        pub mod $name {
            use super::*;

            #[gpu_module]
            pub mod k {
                /// Cells per side: a power of two.
                pub const SIDE: u32 = $side;
                /// Mask of a coordinate: `SIDE - 1`.
                pub const MASK: u32 = SIDE - 1;
                /// Cells of one field.
                pub const CELLS: u32 = SIDE * SIDE;

                /// Per field a spread rate `d` and a keep factor `k`, then two pad fields: six `f32` are 24
                /// bytes and a uniform struct is rounded up to 16 bytes with no gap.
                pub struct Params {
                    pub d0: f32,
                    pub k0: f32,
                    pub d1: f32,
                    pub k1: f32,
                    pub d2: f32,
                    pub k2: f32,
                    pub pad0: f32,
                    pub pad1: f32,
                }

                /// One field of one cell: `k * mul_add(d * 0.25, n + s + e + w, (1 - d) * c)`.
                pub fn blend(c: f32, n: f32, s: f32, e: f32, w: f32, d: f32, k: f32) -> f32 {
                    k * (d * 0.25f32).mul_add(n + s + e + w, (1.0f32 - d) * c)
                }

                /// One step of the three fields and the integer channel of every cell, reading `src` and
                /// `src_count`, writing `dst` and `dst_count`.
                #[kernel(workgroup_size(256, 1, 1))]
                pub fn relax(
                    id: SV_DispatchThreadID,
                    p: Uniform<Params>,
                    src: StructuredBuffer<float>,
                    mut dst: RWStructuredBuffer<float>,
                    src_count: StructuredBuffer<uint>,
                    mut dst_count: RWStructuredBuffer<uint>,
                ) {
                    let i = id.x;
                    if i < CELLS {
                        let x = i & MASK;
                        let y = i / SIDE;
                        let side = SIDE as i32;
                        let row = y * SIDE;
                        // East and west wrap with the mask, north and south with `rem_euclid`.
                        let e = row + ((x + 1u32) & MASK);
                        let w = row + ((x + MASK) & MASK);
                        let n = ((y as i32) - 1i32).rem_euclid(side) as u32 * SIDE + x;
                        let s = ((y as i32) + 1i32).rem_euclid(side) as u32 * SIDE + x;
                        dst[i] = blend(src[i], src[n], src[s], src[e], src[w], p.d0, p.k0);
                        dst[CELLS + i] = blend(
                            src[CELLS + i],
                            src[CELLS + n],
                            src[CELLS + s],
                            src[CELLS + e],
                            src[CELLS + w],
                            p.d1,
                            p.k1,
                        );
                        dst[2u32 * CELLS + i] = blend(
                            src[2u32 * CELLS + i],
                            src[2u32 * CELLS + n],
                            src[2u32 * CELLS + s],
                            src[2u32 * CELLS + e],
                            src[2u32 * CELLS + w],
                            p.d2,
                            p.k2,
                        );
                        let c = src_count[i];
                        dst_count[i] = c - 4u32 * (c >> 3u32)
                            + (src_count[n] >> 3u32)
                            + (src_count[s] >> 3u32)
                            + (src_count[e] >> 3u32)
                            + (src_count[w] >> 3u32);
                    }
                }
            }

            pub const SIDE: usize = k::SIDE as usize;

            /// The parameter struct of the module, from [`RATES`].
            pub fn params() -> k::Params {
                k::Params {
                    d0: RATES[0].0,
                    k0: RATES[0].1,
                    d1: RATES[1].0,
                    k1: RATES[1].1,
                    d2: RATES[2].0,
                    k2: RATES[2].1,
                    pad0: 0.0,
                    pad1: 0.0,
                }
            }

            /// `STEPS` steps on the device, ping-pong between `a` and `b`, reading back after every step.
            pub fn run_gpu(
                handle: &DeviceHandle,
                inputs: &Inputs,
            ) -> Result<Trace, Box<dyn std::error::Error>> {
                let device: &HeadlessDevice = handle;
                let uniform = device.create_uniform("field-relax params", &params())?;
                let fields_a = GpuVec::from_slice_on(handle, &inputs.fields)?;
                let fields_b = GpuVec::<f32>::zeroed_on(handle, inputs.fields.len() as u32)?;
                let counts_a = GpuVec::from_slice_on(handle, &inputs.counts)?;
                let counts_b = GpuVec::<u32>::zeroed_on(handle, inputs.counts.len() as u32)?;
                let mut trace = Trace {
                    fields: vec![inputs.fields.clone()],
                    counts: vec![inputs.counts.clone()],
                };
                // Only the field buffers are `Independent`: their length is `FIELDS * CELLS`, not the dispatch count
                // `CELLS`, so the default one-element-per-invocation access does not describe them.
                for step in 1..=STEPS {
                    if step % 2 == 1 {
                        k::relax::dispatch(
                            device,
                            k::CELLS,
                            &uniform,
                            Independent(&fields_a),
                            Independent(&fields_b),
                            &counts_a,
                            &counts_b,
                        )?;
                        trace.fields.push(fields_b.to_vec()?);
                        trace.counts.push(counts_b.to_vec()?);
                    } else {
                        k::relax::dispatch(
                            device,
                            k::CELLS,
                            &uniform,
                            Independent(&fields_b),
                            Independent(&fields_a),
                            &counts_b,
                            &counts_a,
                        )?;
                        trace.fields.push(fields_a.to_vec()?);
                        trace.counts.push(counts_a.to_vec()?);
                    }
                }
                Ok(trace)
            }

            /// The same steps on the CPU twin (the kernel's own Rust body).
            #[cfg(any(test, gust_dual))]
            pub fn run_twin(inputs: &Inputs) -> Result<Trace, Box<dyn std::error::Error>> {
                let device = CpuDevice::new();
                let uniform = device.uniform(params())?;
                let fields_a = device.buffer(&inputs.fields);
                let fields_b = device.buffer(&vec![0.0f32; inputs.fields.len()]);
                let counts_a = device.buffer(&inputs.counts);
                let counts_b = device.buffer(&vec![0u32; inputs.counts.len()]);
                let mut trace = Trace {
                    fields: vec![inputs.fields.clone()],
                    counts: vec![inputs.counts.clone()],
                };
                for step in 1..=STEPS {
                    if step % 2 == 1 {
                        k::relax::dispatch(
                            &device,
                            k::CELLS,
                            &uniform,
                            Independent(&fields_a),
                            Independent(&fields_b),
                            &counts_a,
                            &counts_b,
                        )?;
                        trace.fields.push(fields_b.read());
                        trace.counts.push(counts_b.read());
                    } else {
                        k::relax::dispatch(
                            &device,
                            k::CELLS,
                            &uniform,
                            Independent(&fields_b),
                            Independent(&fields_a),
                            &counts_b,
                            &counts_a,
                        )?;
                        trace.fields.push(fields_a.read());
                        trace.counts.push(counts_a.read());
                    }
                }
                Ok(trace)
            }
        }
    };
}

field_program!(big, 512u32);
#[cfg(test)]
field_program!(small, 32u32);

fn describe(name: &str, report: &Report) {
    println!(
        "{name}: counts exact on every step and sum conserved; floats within the bounds (one step {:.3e}, {STEPS} steps {:.3e}); \
         worst one-step error {:.3e} (step 1 {:.3e}, step 2 {:.3e}), worst error against the reference {:.3e}",
        float_bound(1),
        float_bound(STEPS),
        report.worst_single_step,
        report.worst_step_1,
        report.worst_step_2,
        report.worst_cumulative
    );
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let handle = default_device()?;
    let info = handle.adapter_info();
    println!("adapter: {} backend: {:?}", info.name, info.backend);
    let inputs = make_inputs(big::SIDE);
    println!(
        "grid {0} x {0}, {FIELDS} fields, {STEPS} steps, {1} cells per field",
        big::SIDE,
        big::SIDE * big::SIDE
    );
    let trace = big::run_gpu(&handle, &inputs)?;
    let report = verify("gpu", &inputs, &trace)?;
    describe("gpu", &report);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_host_reads_the_modules_constants() {
        assert_eq!(big::SIDE, 512);
        assert_eq!(small::SIDE, 32);
        assert!(big::SIDE.is_power_of_two() && small::SIDE.is_power_of_two());
        assert_eq!(FIELDS, 3);
        assert_eq!(big::k::MASK as usize, big::SIDE - 1);
        assert_eq!(small::k::CELLS as usize, small::SIDE * small::SIDE);
        // The struct is 32 bytes, rounded up to 16 and without a gap.
        assert_eq!(std::mem::size_of::<small::k::Params>(), 32);
    }

    #[test]
    fn the_inputs_stay_inside_the_derivation_domain() {
        // Both sizes: the unit tests run the small grid, the GPU claims and `main` the big one.
        for side in [small::SIDE, big::SIDE] {
            let inputs = make_inputs(side);
            assert!(inputs.fields.iter().all(|v| (1.0 / 256.0..4.0).contains(v)));
            // Not symmetric: no field equals another, and a field is not its own transpose.
            let cells = side * side;
            assert_ne!(inputs.fields[..cells], inputs.fields[cells..2 * cells]);
            assert!((0..side).any(|y| {
                (0..y).any(|x| inputs.fields[y * side + x] != inputs.fields[x * side + y])
            }));
            // Counts stay far inside `u32` (the largest cell, with the spike; the sum is a `u64` on the host).
            let largest = inputs.counts.iter().copied().max().unwrap();
            assert!(
                (1 << 24..1 << 25).contains(&largest),
                "side {side}: {largest}"
            );
            // The smallest value after every step is a normal `f32` (SEMANTICS: subnormals may be flushed on a GPU).
            let mut now: Vec<f64> = inputs.fields.iter().map(|&v| f64::from(v)).collect();
            for _ in 0..STEPS {
                now = reference_step_fields(side, &now);
                let smallest = now.iter().copied().fold(f64::INFINITY, f64::min);
                assert!(smallest > 1.0e-8, "side {side}: {smallest}");
                let largest = now.iter().copied().fold(0.0, f64::max);
                assert!(largest < 4.0, "side {side}: {largest}");
            }
        }
    }

    #[test]
    fn the_bound_is_about_twelve_roundings_a_step() {
        let u = 1.0 / 16_777_216.0f64;
        let one = float_bound(1);
        assert!(one > 12.0 * u && one < 12.0 * u * 1.001, "{one}");
        // The derivation needs the carried error far below 2^-12.
        assert!(float_bound(STEPS) < 1.0 / 4096.0 / 8.0);
    }

    #[test]
    fn the_integer_reference_conserves_the_sum_and_moves_mass_over_the_edge() {
        let inputs = make_inputs(small::SIDE);
        let side = small::SIDE;
        let total: u64 = inputs.counts.iter().map(|&c| u64::from(c)).sum();
        let next = reference_step_counts(side, &inputs.counts);
        assert_eq!(next.iter().map(|&c| u64::from(c)).sum::<u64>(), total);
        // The spike at (side - 1, 0) gives to (0, 0) and (side - 1, side - 1): wrapped neighbours.
        let spike = inputs.counts[side - 1];
        assert!(next[0] >= spike >> 3);
        assert!(next[side * (side - 1) + side - 1] >= spike >> 3);
    }

    #[test]
    fn the_bound_check_rejects_an_error_just_over_it() {
        let want = vec![1.0f64, 2.0];
        let bound = float_bound(1);
        assert_eq!(check_against(&want, &[1.0, 2.0], bound).unwrap(), 0.0);
        // One ULP at 1.0 is 2^-23 (1.2e-7), under the bound of 7.2e-7; 8 ULP (9.5e-7) is over it.
        let near = [f32::from_bits(1.0f32.to_bits() + 1), 2.0];
        assert!(check_against(&want, &near, bound).is_ok());
        let far = [f32::from_bits(1.0f32.to_bits() + 8), 2.0];
        assert!(check_against(&want, &far, bound).is_err());
        assert!(check_against(&want, &[f32::NAN, 2.0], bound).is_err());
    }

    #[test]
    fn a_wrong_count_is_reported_with_its_step_and_cell() {
        let inputs = make_inputs(small::SIDE);
        let mut trace = small::run_twin(&inputs).unwrap();
        trace.counts[2][100] += 1;
        let message = verify("test", &inputs, &trace).err().unwrap();
        assert!(
            message.contains("step 2: count at cell 100 (x 4, y 3)"),
            "{message}"
        );
    }

    #[test]
    fn twin_matches_the_references_on_every_step() {
        let inputs = make_inputs(small::SIDE);
        let trace = small::run_twin(&inputs).unwrap();
        let report = verify("twin", &inputs, &trace).unwrap();
        describe("twin, 32 x 32", &report);
        // The twin's mix of rounding is at most the bound derived for the GPU.
        assert!(report.worst_cumulative <= float_bound(STEPS));
        // Alternation: the first two steps read different buffers and each is checked on its own.
        assert!(report.worst_step_1 <= float_bound(1) && report.worst_step_2 <= float_bound(1));
        assert_ne!(trace.fields[1], trace.fields[2]);
        assert_ne!(trace.counts[1], trace.counts[2]);
    }

    #[test]
    fn twin_matches_the_references_at_the_large_size() {
        let inputs = make_inputs(big::SIDE);
        let trace = big::run_twin(&inputs).unwrap();
        let report = verify("twin", &inputs, &trace).unwrap();
        describe("twin, 512 x 512", &report);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_matches_the_twin_and_the_references() {
        let handle = default_device().unwrap_or_else(|error| {
            panic!("Vulkan adapter required; this test must not silently skip: {error}")
        });
        let info = handle.adapter_info();
        println!("adapter: {} backend: {:?}", info.name, info.backend);
        let inputs = make_inputs(big::SIDE);
        let twin = big::run_twin(&inputs).unwrap();
        verify("twin", &inputs, &twin).unwrap();
        let gpu = big::run_gpu(&handle, &inputs).unwrap();
        let report = verify("gpu", &inputs, &gpu).unwrap();
        // The integer channel: bit-equal to the twin after every step.
        for step in 0..=STEPS {
            assert!(gpu.counts[step] == twin.counts[step], "step {step}");
        }
        describe(&format!("gpu on {}", info.name), &report);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_matches_the_references_at_the_small_size() {
        let handle = default_device().unwrap_or_else(|error| {
            panic!("Vulkan adapter required; this test must not silently skip: {error}")
        });
        let inputs = make_inputs(small::SIDE);
        let gpu = small::run_gpu(&handle, &inputs).unwrap();
        verify("gpu", &inputs, &gpu).unwrap();
    }
}
