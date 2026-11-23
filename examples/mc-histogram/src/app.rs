//! A sampled histogram smoothed by diffusion: Monte Carlo sampling with integer atomics, then eight diffusion steps.
//!
//! One program uses the additions of milestone 11 together: the hash generator of `examples/hash-rng`, `atomic_add`,
//! module-level `const` items (m11.g3), `rem_euclid` (m11.math2), `mul_add` (m11.math3) and the host's split of a
//! launch past the per-dimension workgroup limit (m11.g4b).
//!
//! 1. `deposit`: sample `i` is `hash(seed, i)`, turned into a cell of a `SIDE x SIDE` field, and counted with
//!    `atomic_add(.., UNIT)`, a fixed-point weight. The two coordinates are the averages of two 8-bit fields of the
//!    hash (a triangular density, so the histogram is not flat), moved by a small signed drift and wrapped into the
//!    field with `rem_euclid`.
//! 2. `density`: the fixed-point cells as `f32`, times `1 / UNIT`.
//! 3. `diffuse`: one step `c * KEEP + (l + r + u + d) * SHARE` on the wrapped field, written with `mul_add`. Eight
//!    steps ping-pong between two buffers.
//!
//! # Why the path to the cell uses no float
//!
//! The histogram must be equal, bit for bit, on the host reference, the CPU twin and every GPU. Integer atomics are
//! exact and do not depend on the order of the invocations, and the cell is reached with integer arithmetic only. A
//! float product on a GPU may round to either neighbour (D39), which can move a sample across a cell edge and change
//! two cells by one sample each; an integer path has no such rounding.
//!
//! # The float bound of the diffusion
//!
//! The fixed-point cells are below 2^24, so the `u32` to `f32` conversion is exact, and `UNIT` is a power of two, so
//! the density `cell * (1 / UNIT)` is exact. Every operand of the diffusion is then non-negative: no cancellation, and
//! a relative error of a sum of non-negative values is at most the largest relative error among its terms. Model the
//! primitives with the GPU model of `docs/SEMANTICS.md` (the method under "Deriving a tolerance"): `u = 2^-24`, and
//! `e_k` is the relative error of the field after step `k` (`e_0 = 0`).
//!
//! - `+` and `*` land on either neighbour of the exact result, within one ULP, which is at most `2^-23 = 2u` of the
//!   value. A multiplication by a power of two is exact, and so is an addition of 0 (the result is representable).
//!   Every value here is a multiple of `2^-28` or more, so normal.
//! - `(l + r + u + d) * SHARE` is three additions and an exact multiplication: `(1 + e_k)(1 + 2u)^3`. The grouping does
//!   not matter: all four terms are non-negative, so any grouping of four terms is at most three additions deep, and
//!   each addition raises the larger relative error of its two terms by at most `2u`.
//! - `c.mul_add(KEEP, s)` is held to the unfused form that `docs/SEMANTICS.md` states for it: the product `c * KEEP`,
//!   then the sum. Here the product is exact (a power of two), so the fused and the unfused value differ only by the
//!   one final rounding, and the row's domain ("normal operands") is not needed for the product: where `c` or `s` is
//!   exactly 0 while the result is not, the product is 0 or exact and `0 + s` is exact, under the rows for `*` and `+`.
//!   The addends are `c * KEEP` with `(1 + e_k)` and `s` with `(1 + e_k)(1 + 2u)^3`; the larger is the second, and the
//!   final addition adds `2u * (1 + 2^-16)` (the extra factor covers "one ULP at the larger of the exact and the
//!   actual result", since `e_8` is far below `2^-16`).
//!
//! So `1 + e_{k+1} = (1 + e_k) * f` with `f = (1 + 2u)^3 * (1 + 2u * (1 + 2^-16))`, about `8u` per step, and `e_8 =
//! f^8 - 1`, about `64u = 3.8e-6`. The reference runs in `f64`, whose own error over 40 roundings is below `2^-47`;
//! [`float_bound`] adds `2^-40` for it and for the arithmetic of the bound itself. A fused `mul_add` rounds once and
//! has less error than this, and a correctly rounded `+` (the twin) has half the error charged here, so one bound
//! covers every device. Where the exact value is zero the result must be exactly zero (all operands are then zero).
//!
//! The bound is asserted. The largest observed error is printed per adapter and is not asserted.

#[cfg(any(test, gust_dual))]
use gust::cpu::CpuDevice;
use gust::{Independent, gpu_fn, gpu_module, map::Broadcast};
use gust_wgpu::{DeviceHandle, GpuVec, HeadlessDevice, default_device};

#[gpu_module]
mod mc {
    /// Cells per side of the field: a power of two.
    pub const SIDE: u32 = 256;
    /// The fixed-point weight of one sample (a power of two, so the density is exact).
    pub const UNIT: u32 = 16;
    /// The weight of a cell's own value in a diffusion step.
    pub const KEEP: f32 = 0.5;
    /// The weight of each of the four neighbours; `KEEP + 4 * SHARE` is 1.
    pub const SHARE: f32 = 0.125;

    /// A multiply-xorshift finaliser. Every shift amount is a literal below 32 (D39).
    ///
    /// The constants and shifts are those of "lowbias32" from Chris Wellons' hash-prospector
    /// (https://github.com/skeeto/hash-prospector; public domain, Unlicense): multipliers 0x7feb352d and
    /// 0x846ca68b, shifts 16, 15, 16.
    pub fn mix(x: u32) -> u32 {
        let mut h = x ^ (x >> 16u32);
        h = h.wrapping_mul(2146121005u32);
        h = h ^ (h >> 15u32);
        h = h.wrapping_mul(2221713035u32);
        h ^ (h >> 16u32)
    }

    /// A hash of a seed and an index. The seed is mixed before the index joins it, so different seeds give
    /// unrelated streams rather than shifted copies of one.
    pub fn hash(seed: u32, index: u32) -> u32 {
        mix(mix(seed) ^ index)
    }

    /// The index of the cell of sample `i`, integers only. `x` and `y` are averages of two 8-bit fields of the hash
    /// (triangular on 0..=255); a second hash gives a drift in -8..=7 on each axis; the offsets push mass over both
    /// edges, so the wrap is used. `rem_euclid` brings a negative or large coordinate into `0..SIDE`.
    pub fn cell(seed: u32, i: u32) -> u32 {
        let h = hash(seed, i);
        let g = mix(h ^ 2654435769u32);
        let x = ((h & 255u32) + ((h >> 8u32) & 255u32)) >> 1u32;
        let y = (((h >> 16u32) & 255u32) + (h >> 24u32)) >> 1u32;
        let dx = (g & 15u32) as i32 - 8i32;
        let dy = ((g >> 8u32) & 15u32) as i32 - 8i32;
        let n = SIDE as i32;
        let cx = (x as i32 + dx - 96i32).rem_euclid(n);
        let cy = (y as i32 + dy + 80i32).rem_euclid(n);
        (cy as u32) * SIDE + (cx as u32)
    }

    /// Counts sample `i` of `params[1]` samples (`params[0]` is the seed) into its cell, with weight `UNIT`.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn deposit(
        id: SV_DispatchThreadID,
        params: StructuredBuffer<uint>, // seed, sample count; bound with `Independent`
        mut cells: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < params[1u32] {
            atomic_add(&mut cells[cell(params[0u32], i)], UNIT);
        }
    }

    /// One diffusion step on the `SIDE x SIDE` field, wrapping at the edges.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn diffuse(
        id: SV_DispatchThreadID,
        src: StructuredBuffer<float>,
        mut dst: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < dst.len() {
            let x = i % SIDE;
            let y = i / SIDE;
            let n = SIDE as i32;
            let xl = ((x as i32) - 1i32).rem_euclid(n) as u32;
            let xr = ((x as i32) + 1i32).rem_euclid(n) as u32;
            let yu = ((y as i32) - 1i32).rem_euclid(n) as u32;
            let yd = ((y as i32) + 1i32).rem_euclid(n) as u32;
            let c = src[i];
            let l = src[y * SIDE + xl];
            let r = src[y * SIDE + xr];
            let u = src[yu * SIDE + x];
            let d = src[yd * SIDE + x];
            dst[i] = c.mul_add(KEEP, (l + r + u + d) * SHARE);
        }
    }
}

/// The fixed-point cells as `f32` (the map stage): `cell * scale`, with `scale = 1 / UNIT` exact.
// The plain Rust function behind each `#[gpu_fn]` stays callable (D60), but this program calls only the generated
// maps, so the lint is silenced here.
#[allow(dead_code)]
#[gpu_fn]
pub fn density(cell: u32, scale: f32) -> f32 {
    cell as f32 * scale
}

/// The seeds of the tests and the demo.
pub const SEEDS: [u32; 3] = [0, 1, 0xdead_beef];
/// Diffusion steps (even: the result ends in the buffer the density started in).
pub const STEPS: usize = 8;
/// Samples of `main`: 2^24 + 1000, which at workgroup 256 needs 65,540 workgroups, past the 65,535 limit of most
/// devices.
pub const MAIN_SAMPLES: u32 = (1 << 24) + 1000;

const FIELD: u32 = mc::SIDE * mc::SIDE;

/// The scale of the density map: `1 / UNIT`, a power of two.
fn density_scale() -> f32 {
    1.0 / mc::UNIT as f32
}

// ---- Host references: plain Rust, with their own copy of the arithmetic, not the `#[gpu_module]` bodies. ----

fn reference_mix(x: u32) -> u32 {
    let mut h = x ^ (x >> 16);
    h = h.wrapping_mul(2146121005);
    h ^= h >> 15;
    h = h.wrapping_mul(2221713035);
    h ^ (h >> 16)
}

/// The cell of sample `i`, in `i64` arithmetic with `rem_euclid` of the host.
fn reference_cell(seed: u32, i: u32) -> usize {
    let h = reference_mix(reference_mix(seed) ^ i);
    let g = reference_mix(h ^ 0x9e37_79b9);
    let x = i64::from(((h & 255) + ((h >> 8) & 255)) / 2);
    let y = i64::from((((h >> 16) & 255) + (h >> 24)) / 2);
    let dx = i64::from(g & 15) - 8;
    let dy = i64::from((g >> 8) & 15) - 8;
    let side = i64::from(mc::SIDE);
    let cx = (x + dx - 96).rem_euclid(side);
    let cy = (y + dy + 80).rem_euclid(side);
    (cy * side + cx) as usize
}

/// The histogram in fixed point: `UNIT` per sample.
pub fn reference_histogram(seed: u32, samples: u32) -> Vec<u32> {
    let mut cells = vec![0u32; FIELD as usize];
    for i in 0..samples {
        cells[reference_cell(seed, i)] += mc::UNIT;
    }
    cells
}

/// `STEPS` diffusion steps in `f64` from the exact density of `cells`.
pub fn reference_field(cells: &[u32]) -> Vec<f64> {
    let side = mc::SIDE as usize;
    let mut now: Vec<f64> = cells
        .iter()
        .map(|&cell| f64::from(cell) / f64::from(mc::UNIT))
        .collect();
    let (keep, share) = (f64::from(mc::KEEP), f64::from(mc::SHARE));
    for _ in 0..STEPS {
        let mut next = vec![0.0f64; now.len()];
        for y in 0..side {
            for x in 0..side {
                let at = |x: usize, y: usize| now[y * side + x];
                let (xl, xr) = ((x + side - 1) % side, (x + 1) % side);
                let (yu, yd) = ((y + side - 1) % side, (y + 1) % side);
                next[y * side + x] =
                    at(x, y) * keep + (at(xl, y) + at(xr, y) + at(x, yu) + at(x, yd)) * share;
            }
        }
        now = next;
    }
    now
}

/// The relative error bound of the derivation in the module documentation.
pub fn float_bound() -> f64 {
    let u = 1.0 / 16_777_216.0f64;
    let step = (1.0 + 2.0 * u).powi(3) * (1.0 + 2.0 * u * (1.0 + 1.0 / 65_536.0));
    step.powi(STEPS as i32) - 1.0 + 1.0 / 1_099_511_627_776.0
}

/// The largest relative error of `got` against `want`, or the first element outside [`float_bound`] as a message.
pub fn check_field(want: &[f64], got: &[f32]) -> Result<f64, String> {
    if want.len() != got.len() {
        return Err(format!("{} results for {} cells", got.len(), want.len()));
    }
    let bound = float_bound();
    let mut worst = 0.0f64;
    for (i, (want, got)) in want.iter().zip(got).enumerate() {
        let got = f64::from(*got);
        if *want == 0.0 {
            if got != 0.0 {
                return Err(format!("cell {i}: exact value 0, got {got}"));
            }
            continue;
        }
        let error = (got - want).abs() / want;
        // A NaN is outside the bound too.
        if error.is_nan() || error > bound {
            return Err(format!(
                "cell {i}: got {got}, reference {want}, relative error {error} over the bound {bound}"
            ));
        }
        worst = worst.max(error);
    }
    Ok(worst)
}

/// The first cell at which two histograms differ, as a message.
pub fn first_difference(name: &str, left: &[u32], right: &[u32]) -> Result<(), String> {
    if left.len() != right.len() {
        return Err(format!(
            "{name}: {} cells versus {}",
            left.len(),
            right.len()
        ));
    }
    match (0..left.len()).find(|&i| left[i] != right[i]) {
        Some(i) => Err(format!(
            "{name}: first differing cell {i} (x {}, y {}): {} versus {}",
            i % mc::SIDE as usize,
            i / mc::SIDE as usize,
            left[i],
            right[i]
        )),
        None => Ok(()),
    }
}

/// What one run of the three stages produced.
pub struct Outcome {
    pub cells: Vec<u32>,
    /// `reduce::sum_wrapping` of the cells (the host's wrapping sum on the twin).
    pub sum: u32,
    pub density: Vec<f32>,
    pub field: Vec<f32>,
}

/// The three stages on the device: the cells stay on the device from `deposit` through the map and the diffusion.
pub fn run_gpu(
    handle: &DeviceHandle,
    seed: u32,
    samples: u32,
) -> Result<Outcome, Box<dyn std::error::Error>> {
    let device: &HeadlessDevice = handle;
    let params = GpuVec::from_slice_on(handle, &[seed, samples])?;
    let cells = GpuVec::<u32>::zeroed_on(handle, FIELD)?;
    mc::deposit::dispatch(device, samples, Independent(&params), Independent(&cells))?;
    let sum = gust_algo::reduce::sum_wrapping(&cells)?;
    let host_cells = cells.to_vec()?;

    let mut now = density::map_gpu_on(handle, &cells, Broadcast(density_scale()))?;
    let density_values = now.to_vec()?;
    let mut next = GpuVec::<f32>::zeroed_on(handle, FIELD)?;
    for _ in 0..STEPS {
        mc::diffuse::dispatch(device, FIELD, &now, &next)?;
        std::mem::swap(&mut now, &mut next);
    }
    Ok(Outcome {
        cells: host_cells,
        sum,
        density: density_values,
        field: now.to_vec()?,
    })
}

/// The same stages on the CPU twin (the functions' and kernels' own Rust bodies).
#[cfg(any(test, gust_dual))]
pub fn run_twin(seed: u32, samples: u32) -> Result<Outcome, Box<dyn std::error::Error>> {
    let device = CpuDevice::new();
    let params = device.buffer(&[seed, samples]);
    let cells = device.buffer(&vec![0u32; FIELD as usize]);
    mc::deposit::dispatch(&device, samples, Independent(&params), Independent(&cells))?;
    let host_cells = cells.read();
    let sum = host_cells.iter().fold(0u32, |a, &c| a.wrapping_add(c));

    let density_values = density::map_on(&device, &host_cells, Broadcast(density_scale()))?;
    let mut now = device.buffer(&density_values);
    let mut next = device.buffer(&vec![0.0f32; FIELD as usize]);
    for _ in 0..STEPS {
        mc::diffuse::dispatch(&device, FIELD, &now, &next)?;
        std::mem::swap(&mut now, &mut next);
    }
    Ok(Outcome {
        cells: host_cells,
        sum,
        density: density_values,
        field: now.read(),
    })
}

/// Checks one outcome against the host references; returns the largest relative error of the diffusion.
pub fn check(
    name: &str,
    seed: u32,
    samples: u32,
    outcome: &Outcome,
    want_cells: &[u32],
) -> Result<f64, String> {
    first_difference(
        &format!("{name} histogram, seed {seed}"),
        &outcome.cells,
        want_cells,
    )?;
    let want_sum = samples.wrapping_mul(mc::UNIT);
    if outcome.sum != want_sum {
        return Err(format!(
            "{name}: sum of the cells {} for {samples} samples of weight {}, expected {want_sum}",
            outcome.sum,
            mc::UNIT
        ));
    }
    let scale = density_scale();
    let want_density: Vec<f32> = want_cells.iter().map(|&c| c as f32 * scale).collect();
    let got_bits: Vec<u32> = outcome.density.iter().map(|x| x.to_bits()).collect();
    let want_bits: Vec<u32> = want_density.iter().map(|x| x.to_bits()).collect();
    first_difference(
        &format!("{name} density, seed {seed}"),
        &got_bits,
        &want_bits,
    )?;
    check_field(&reference_field(want_cells), &outcome.field)
        .map_err(|message| format!("{name} diffusion, seed {seed}: {message}"))
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let handle = default_device()?;
    let device: &HeadlessDevice = &handle;
    let info = device.adapter_info();
    let limit = device.limits().max_compute_workgroups_per_dimension;
    let workgroups = MAIN_SAMPLES.div_ceil(mc::deposit::DESCRIPTOR.workgroup_size[0]);
    println!("adapter: {} backend: {:?}", info.name, info.backend);
    println!(
        "samples {MAIN_SAMPLES}: max_compute_workgroups_per_dimension {limit}, workgroups {workgroups}, launch {}",
        if workgroups > limit {
            "workgroup count past the limit (the host splits such a launch)"
        } else {
            "workgroup count within the limit (not split)"
        }
    );

    let seed = SEEDS[1];
    let want_cells = reference_histogram(seed, MAIN_SAMPLES);
    let outcome = run_gpu(&handle, seed, MAIN_SAMPLES)?;
    // The histogram (against the host reference), its sum, the density and the diffusion.
    let worst = check("gpu", seed, MAIN_SAMPLES, &outcome, &want_cells)?;
    println!(
        "deposit: {} cells equal to the host reference, sum {} = {MAIN_SAMPLES} samples x {}",
        outcome.cells.len(),
        outcome.sum,
        mc::UNIT
    );
    println!(
        "density: {} cells exact against the host",
        outcome.density.len()
    );
    println!(
        "diffuse: {STEPS} steps within the bound {:.3e}, largest relative error {worst:.3e}",
        float_bound()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The number of samples of the CPU test: not a multiple of the workgroup width.
    const SMALL: u32 = 5_000 + 37;

    #[test]
    fn twin_matches_host_reference() {
        for seed in SEEDS {
            let want = reference_histogram(seed, SMALL);
            let outcome = run_twin(seed, SMALL).unwrap();
            let worst = check("twin", seed, SMALL, &outcome, &want).unwrap();
            println!("seed {seed}: twin diffusion largest relative error {worst:.3e}");
        }
    }

    #[test]
    fn the_host_reads_the_modules_constants() {
        assert_eq!(mc::SIDE, 256);
        assert!(mc::SIDE.is_power_of_two());
        assert!(mc::UNIT.is_power_of_two());
        assert_eq!(mc::KEEP + 4.0 * mc::SHARE, 1.0);
        assert_eq!(density_scale(), 0.0625);
    }

    #[test]
    fn the_histogram_is_not_flat_and_reaches_the_wrapped_edges() {
        let cells = reference_histogram(1, 1 << 16);
        let side = mc::SIDE as usize;
        assert_eq!(cells.iter().sum::<u32>(), (1 << 16) * mc::UNIT);
        let busiest = *cells.iter().max().unwrap();
        let used = cells.iter().filter(|&&c| c > 0).count();
        assert!(busiest > 8 * mc::UNIT, "busiest cell {busiest}");
        assert!(used > side * side / 8, "{used} cells used");
        // Mass lands in the first and last column and row: the drift and the offsets reach over the edges.
        assert!((0..side).any(|y| cells[y * side] > 0));
        assert!((0..side).any(|y| cells[y * side + side - 1] > 0));
        assert!(cells[..side].iter().any(|&c| c > 0));
        assert!(cells[side * (side - 1)..].iter().any(|&c| c > 0));
    }

    #[test]
    fn the_inputs_stay_inside_the_derivation_domain() {
        // Cells stay below 2^24 (the conversion to `f32` is exact) at the largest sample count of `main`.
        let cells = reference_histogram(1, MAIN_SAMPLES);
        assert!(cells.iter().all(|&c| c < 1 << 24));
        // The smallest positive value after `STEPS` steps is a normal `f32` (the derivation assumes no subnormal).
        let smallest = f64::from(mc::SHARE).powi(STEPS as i32) / f64::from(mc::UNIT);
        assert!(smallest > f64::from(f32::MIN_POSITIVE));
    }

    #[test]
    fn the_bound_is_about_sixty_four_roundings_of_a_single_precision_float() {
        let u = 1.0 / 16_777_216.0f64;
        let bound = float_bound();
        assert!(bound > 64.0 * u, "{bound}");
        assert!(bound < 64.0 * u * 1.001, "{bound}");
    }

    #[test]
    fn the_bound_check_rejects_an_error_just_over_it() {
        let want = vec![1.0f64, 2.0, 0.0];
        let ok = [1.0f32, 2.0, 0.0];
        assert_eq!(check_field(&want, &ok).unwrap(), 0.0);
        // One ULP at 1.0 is 2^-23 (1.2e-7), far under the bound of about 3.8e-6; 40 ULP (4.8e-6) is over it.
        let near = [f32::from_bits(1.0f32.to_bits() + 1), 2.0, 0.0];
        assert!(check_field(&want, &near).is_ok());
        let far = [f32::from_bits(1.0f32.to_bits() + 40), 2.0, 0.0];
        assert!(check_field(&want, &far).is_err());
        let nonzero = [1.0f32, 2.0, 1e-30];
        assert!(check_field(&want, &nonzero).is_err());
        assert!(check_field(&want, &[f32::NAN, 2.0, 0.0]).is_err());
    }

    #[test]
    fn a_wrong_cell_is_reported_with_its_position() {
        let mut cells = reference_histogram(0, 100);
        let want = cells.clone();
        cells[300] += 1;
        let message = first_difference("test", &cells, &want).unwrap_err();
        assert!(message.contains("cell 300 (x 44, y 1)"), "{message}");
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_matches_the_twin_and_the_host_reference() {
        let handle = default_device().unwrap_or_else(|error| {
            panic!("Vulkan adapter required; this test must not silently skip: {error}")
        });
        let info = handle.adapter_info();
        println!("adapter: {} backend: {:?}", info.name, info.backend);
        let samples = (1u32 << 16) + 37;
        let mut worst = 0.0f64;
        for seed in SEEDS {
            let want = reference_histogram(seed, samples);
            let twin = run_twin(seed, samples).unwrap();
            check("twin", seed, samples, &twin, &want).unwrap();
            let gpu = run_gpu(&handle, seed, samples).unwrap();
            let error = check("gpu", seed, samples, &gpu, &want).unwrap();
            first_difference("gpu against twin", &gpu.cells, &twin.cells).unwrap();
            println!(
                "seed {seed}: histogram equal on reference, twin and GPU; diffusion error {error:.3e}"
            );
            worst = worst.max(error);
        }
        println!(
            "diffusion: largest relative error {worst:.3e} on {} (bound {:.3e})",
            info.name,
            float_bound()
        );
    }
}
