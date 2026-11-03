//! Three element-wise maps (`#[gpu] fn`, D60) chained two ways.
//!
//! - **Resident**: the input goes to the device once as a `GpuVec`, `map_gpu_on` runs the three steps on device
//!   buffers (each step's result is the next step's argument, never read back), and one `to_vec` reads the answer.
//! - **Materialised**: `map_on` over host slices three times; every step uploads its argument and reads its result
//!   back into a `Vec`, which the next step uploads again.
//!
//! Both are checked against an independent host reference (`reference`, plain `f64` arithmetic written out in this
//! file, not the `#[gpu]` bodies) within a bound derived below, and against each other. On the CPU twin
//! (`CpuDevice`, test and dual builds) the same chain runs the functions' own Rust bodies.
//!
//! # The float bound
//!
//! The steps use only `+` and `*` on positive operands, with no cancellation and no branch whose outcome depends on a
//! rounded value. Model each GPU or CPU primitive, fused or not, as correct to a relative error `u = 2^-24`
//! (round to nearest, no flush, no overflow; a fused multiply-add rounds once, so it only has less error than its two
//! parts). For positive operands the relative error of a product is at most the sum of its operands' errors plus `u`,
//! and that of a sum at most the larger of its operands' errors plus `u`. The inputs and constants are exact in `f32`.
//!
//! - `rescale = x * gain + offset`: product `u`, sum `u + u = 2u`.
//! - `curve = x * x * k + x` (`x` has error `2u`): `x * x` is `4u + u = 5u`, `* k` is `6u`, `+ x` is `max(6u, 2u) + u
//!   = 7u`.
//! - `damp = (x + shift) * scale` (`x` has error `7u`): the sum is `7u + u = 8u`, the product `9u`.
//!
//! So whatever order or fusion a backend picks, each result is within `9u` of the exact value, relative. The exact
//! value is computed in `f64`, whose own error is below `2^-49` relative here (nine roundings of `2^-53`), so
//! [`FLOAT_BOUND`] is `9 * 2^-24 * (1 + 2^-20)`. `inputs_stay_inside_the_derivation_domain` pins the domain the
//! argument needs (all operands positive and finite, every intermediate normal, nothing near overflow).
//!
//! The same bound holds against the exact value for the CPU twin, the GPU resident chain and the GPU materialised
//! chain, one by one. The two GPU chains also agree bit for bit with each other: they run the same kernels over the
//! same bits, and a buffer that stays on the device is not rounded again.

use std::time::Instant;

use std::path::PathBuf;

#[cfg(any(test, gust_dual))]
use gust::cpu::CpuDevice;
use gust::{
    KernelDescriptor, gpu,
    map::{Broadcast, Error as MapError},
};
use gust_wgpu::{DeviceHandle, GpuVec, HeadlessDevice, default_device};

// The plain Rust function behind each `#[gpu]` stays callable (D60), but this program calls only the generated maps,
// so the lint is silenced here.
// Step 1: an affine map. Both constants are broadcast scalars (a one-element buffer, read at index 0).
#[allow(dead_code)]
#[gpu]
pub fn rescale(x: f32, gain: f32, offset: f32) -> f32 {
    x * gain + offset
}

// Step 2: a quadratic.
#[allow(dead_code)]
#[gpu]
pub fn curve(x: f32, k: f32) -> f32 {
    x * x * k + x
}

// Step 3: a shift and a scale.
#[allow(dead_code)]
#[gpu]
pub fn damp(x: f32, shift: f32, scale: f32) -> f32 {
    (x + shift) * scale
}

pub const GAIN: f32 = 0.75;
pub const OFFSET: f32 = 1.5;
pub const CURVE: f32 = 0.25;
pub const SHIFT: f32 = 0.5;
pub const SCALE: f32 = 0.125;

/// The relative error bound of the derivation in the module documentation: `9u` for `u = 2^-24`, with `2^-20`
/// relative slack for the `f64` reference.
pub const FLOAT_BOUND: f64 = 9.0 / 16_777_216.0 * (1.0 + 1.0 / 1_048_576.0);

/// The pool hand-outs of one resident chain: the input upload (1), the two broadcasts and the result of `rescale`
/// (3), the broadcast and the result of `curve` (2), the two broadcasts and the result of `damp` (3), and the one
/// read-back's staging buffer (1). Every upload, result buffer and read-back takes exactly one buffer from the
/// device's pool, so a step that read back and uploaded again would add two.
pub const RESIDENT_HAND_OUTS: u64 = 1 + 3 + 2 + 3 + 1;

/// Inputs in `[0, 4)`, multiples of 1/1024 (exact in `f32`), different in every position.
pub fn input(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) >> 20) as f32 / 1024.0)
        .collect()
}

/// The chain on one value in `f64`, written out here and not taken from the `#[gpu]` bodies.
pub fn reference(x: f32) -> f64 {
    let y1 = f64::from(x) * f64::from(GAIN) + f64::from(OFFSET);
    let y2 = y1 * y1 * f64::from(CURVE) + y1;
    (y2 + f64::from(SHIFT)) * f64::from(SCALE)
}

/// The first element at which `got` leaves [`FLOAT_BOUND`] of the reference of `x`, as a message; `None` when all
/// are inside. Also reports a length difference.
pub fn first_outside_bound(x: &[f32], got: &[f32]) -> Option<String> {
    if x.len() != got.len() {
        return Some(format!("{} results for {} inputs", got.len(), x.len()));
    }
    x.iter().zip(got).enumerate().find_map(|(i, (x, got))| {
        let want = reference(*x);
        let error = (f64::from(*got) - want).abs();
        // A NaN fails the comparison, so it is outside.
        let inside = error <= FLOAT_BOUND * want.abs();
        (!inside).then(|| {
            format!(
                "element {i}: x = {x}, got {got}, reference {want}, relative error {}",
                error / want.abs()
            )
        })
    })
}

/// The chain on the device: the input goes up once, three `map_gpu_on` calls pass `GpuVec`s, one read-back.
pub fn resident(handle: &DeviceHandle, x: &[f32]) -> Result<Vec<f32>, MapError> {
    let input = GpuVec::from_slice_on(handle, x).map_err(MapError::device)?;
    let stepped = rescale::map_gpu_on(handle, &input, Broadcast(GAIN), Broadcast(OFFSET))?;
    let curved = curve::map_gpu_on(handle, &stepped, Broadcast(CURVE))?;
    let damped = damp::map_gpu_on(handle, &curved, Broadcast(SHIFT), Broadcast(SCALE))?;
    damped.to_vec().map_err(MapError::device)
}

/// The chain over host slices: every step uploads its argument and reads its result into a `Vec`.
pub fn materialised(device: &HeadlessDevice, x: &[f32]) -> Result<Vec<f32>, MapError> {
    let stepped = rescale::map_on(device, x, Broadcast(GAIN), Broadcast(OFFSET))?;
    let curved = curve::map_on(device, &stepped, Broadcast(CURVE))?;
    damp::map_on(device, &curved, Broadcast(SHIFT), Broadcast(SCALE))
}

/// The materialised chain on the CPU twin (test and dual builds): the functions' own Rust bodies, no GPU.
#[cfg(any(test, gust_dual))]
pub fn twin(device: &CpuDevice, x: &[f32]) -> Result<Vec<f32>, MapError> {
    let stepped = rescale::map_on(device, x, Broadcast(GAIN), Broadcast(OFFSET))?;
    let curved = curve::map_on(device, &stepped, Broadcast(CURVE))?;
    damp::map_on(device, &curved, Broadcast(SHIFT), Broadcast(SCALE))
}

fn descriptors() -> [&'static KernelDescriptor; 3] {
    [&rescale::DESCRIPTOR, &curve::DESCRIPTOR, &damp::DESCRIPTOR]
}

/// Slang, WGSL, SPIR-V and readable host code of the three kernels, `<module>__<kernel>.*` in `generated-wgpu/`.
fn export_artifacts() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../generated-wgpu");
    std::fs::create_dir_all(&directory)?;
    for descriptor in descriptors() {
        let stem = format!("{}__{}", descriptor.module, descriptor.name);
        std::fs::write(
            directory.join(format!("{stem}.slang")),
            descriptor.slang_source,
        )?;
        std::fs::write(
            directory.join(format!("{stem}.wgsl")),
            gust::slang::compile_wgsl(descriptor)?,
        )?;
        std::fs::write(
            directory.join(format!("{stem}.spv")),
            gust::spirv::words_as_le_bytes(&gust::slang::compile_spirv(descriptor)?),
        )?;
        std::fs::write(
            directory.join(format!("{stem}.rs")),
            gust_wgpu::render_wgpu_source(descriptor)?,
        )?;
    }
    Ok(directory.canonicalize().unwrap_or(directory))
}

/// The input length of `main`: 2^20 elements, above the size where a host-slice map is only a convenience.
const MAIN_LEN: usize = 1 << 20;
/// Calls before the timed ones (the first call builds the pipelines).
const WARM_UP: usize = 3;
/// Timed calls; the median is printed.
const REPEATS: usize = 10;

/// Milliseconds of `call`, `WARM_UP` untimed calls first and then `REPEATS` timed ones: the median and the minimum.
fn time_ms(mut call: impl FnMut()) -> (f64, f64) {
    for _ in 0..WARM_UP {
        call();
    }
    let mut samples: Vec<f64> = (0..REPEATS)
        .map(|_| {
            let started = Instant::now();
            call();
            started.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    (samples[REPEATS / 2], samples[0])
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let handle = default_device()?;
    let x = input(MAIN_LEN);

    let resident_result = resident(&handle, &x)?;
    let materialised_result = materialised(&handle, &x)?;
    if let Some(problem) = first_outside_bound(&x, &resident_result) {
        return Err(format!("resident chain: {problem}").into());
    }
    if let Some(problem) = first_outside_bound(&x, &materialised_result) {
        return Err(format!("materialised chain: {problem}").into());
    }
    assert!(
        resident_result
            .iter()
            .zip(&materialised_result)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the two GPU chains must agree bit for bit"
    );
    let worst = x
        .iter()
        .zip(&resident_result)
        .map(|(x, got)| {
            let want = reference(*x);
            (f64::from(*got) - want).abs() / want.abs()
        })
        .fold(0.0, f64::max);

    let before = handle.buffer_pool_stats();
    resident(&handle, &x)?;
    let after = handle.buffer_pool_stats();
    let hand_outs = after.hits + after.misses - before.hits - before.misses;

    let (resident_ms, resident_min) = time_ms(|| {
        resident(&handle, &x).expect("resident chain");
    });
    let (materialised_ms, materialised_min) = time_ms(|| {
        materialised(&handle, &x).expect("materialised chain");
    });

    println!("map-chain: three #[gpu] fn maps, resident versus materialised");
    println!("adapter: {}", handle.adapter_info().name);
    println!(
        "conditions: n = {MAIN_LEN}, {WARM_UP} warm-up calls, {REPEATS} timed calls, host slice in and host Vec out"
    );
    println!("resident     (1 upload, 3 x map_gpu_on, 1 read-back):");
    println!("  median {resident_ms:.3} ms, min {resident_min:.3} ms");
    println!("materialised (3 x map_on over host slices):");
    println!("  median {materialised_ms:.3} ms, min {materialised_min:.3} ms");
    println!(
        "materialised / resident: {:.2} (median), {:.2} (min)",
        materialised_ms / resident_ms,
        materialised_min / resident_min
    );
    println!(
        "resident chain pool hand-outs: {hand_outs} (expected {RESIDENT_HAND_OUTS}: one upload and one read-back)"
    );
    println!(
        "worst relative error against the f64 reference: {worst:.3e} (bound {FLOAT_BOUND:.3e})"
    );
    assert_eq!(hand_outs, RESIDENT_HAND_OUTS);

    for descriptor in descriptors() {
        println!(
            "kernel: {} ({} artifact(s) embedded)",
            descriptor.qualified_name(),
            descriptor.artifacts.len()
        );
    }

    println!("generated artifacts: {}", export_artifacts()?.display());

    #[cfg(gust_dual)]
    report_twin(&x);
    Ok(())
}

/// The same chain on the CPU twin, inside the bound too.
#[cfg(gust_dual)]
fn report_twin(x: &[f32]) {
    let twin = twin(&CpuDevice::new(), x).expect("the CPU twin should run");
    assert_eq!(
        first_outside_bound(x, &twin),
        None,
        "CPU twin versus the reference"
    );
    println!(
        "CPU twin: matches ({} elements, within the derived bound)",
        twin.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Either side of a workgroup (256) and of 256 workgroups, and the empty and one-element calls.
    const SIZES: [usize; 6] = [0, 1, 255, 256, 257, 65_537];
    /// One large call: 4 Mi elements, 16 MiB per vector.
    const LARGE: usize = 1 << 22;
    /// The pool hand-outs of one materialised chain: per step the argument's upload, the broadcasts, the result and
    /// its read-back (5 + 4 + 5).
    const MATERIALISED_HAND_OUTS: u64 = (1 + 2 + 1 + 1) + (1 + 1 + 1 + 1) + (1 + 2 + 1 + 1);

    /// The chain in plain `f32` arithmetic, step by step as the three functions are written: the CPU twin runs the
    /// functions' own bodies, so it must equal this to the last bit.
    fn plain_f32(x: f32) -> f32 {
        let y1 = x * GAIN + OFFSET;
        let y2 = y1 * y1 * CURVE + y1;
        (y2 + SHIFT) * SCALE
    }

    #[test]
    fn inputs_stay_inside_the_derivation_domain() {
        // The derivation needs positive operands (no cancellation), finite normal intermediates and no overflow.
        let x = input(LARGE);
        let (low, high) = x.iter().fold((f32::MAX, f32::MIN), |(low, high), v| {
            (low.min(*v), high.max(*v))
        });
        assert!(
            (0.0..4.0).contains(&low) && (0.0..4.0).contains(&high),
            "{low}..{high}"
        );
        for constant in [GAIN, OFFSET, CURVE, SHIFT, SCALE] {
            assert!(constant > 0.0 && constant.is_normal());
        }
        // The smallest and the largest input: every intermediate is positive and far from the subnormal and the
        // overflow ranges of `f32`.
        for value in [0.0f32, 4095.0 / 1024.0] {
            let y1 = f64::from(value) * f64::from(GAIN) + f64::from(OFFSET);
            let y2 = y1 * y1 * f64::from(CURVE) + y1;
            let y3 = reference(value);
            for intermediate in [y1, y2, y3] {
                assert!(
                    (0.1..1.0e6).contains(&intermediate),
                    "{intermediate} for x = {value}"
                );
            }
        }
    }

    #[test]
    fn the_bound_is_nine_roundings_of_a_single_precision_float() {
        assert_eq!(FLOAT_BOUND, 9.0 * 2f64.powi(-24) * (1.0 + 2f64.powi(-20)));
        // A one-ulp error on a result is within it, a ten-ulp error is not.
        let x = [1.0f32];
        let right = plain_f32(1.0);
        assert!(first_outside_bound(&x, &[right]).is_none());
        let ulp = f32::EPSILON * right;
        assert!(first_outside_bound(&x, &[right + ulp]).is_none());
        assert!(first_outside_bound(&x, &[right + 10.0 * ulp]).is_some());
        assert!(first_outside_bound(&x, &[f32::NAN]).is_some());
        assert!(first_outside_bound(&x, &[]).is_some());
    }

    #[test]
    fn twin_chain_matches_host_reference() {
        let device = CpuDevice::new();
        for len in SIZES {
            let x = input(len);
            let got = twin(&device, &x).unwrap();
            assert_eq!(first_outside_bound(&x, &got), None, "{len} elements");
            let plain: Vec<f32> = x.iter().map(|x| plain_f32(*x)).collect();
            assert_eq!(
                got, plain,
                "{len} elements: the twin runs the functions' bodies"
            );
        }
    }

    #[test]
    fn twin_steps_run_in_order() {
        // Each step alone, then the chain: swapping two steps changes the answer (the steps do not commute).
        let device = CpuDevice::new();
        let x = input(300);
        let swapped = {
            let curved = curve::map_on(&device, &x, Broadcast(CURVE)).unwrap();
            let stepped =
                rescale::map_on(&device, &curved, Broadcast(GAIN), Broadcast(OFFSET)).unwrap();
            damp::map_on(&device, &stepped, Broadcast(SHIFT), Broadcast(SCALE)).unwrap()
        };
        let chain = twin(&device, &x).unwrap();
        assert_ne!(swapped, chain);
        assert!(first_outside_bound(&x, &swapped).is_some());
        assert_eq!(first_outside_bound(&x, &chain), None);
    }

    #[test]
    #[cfg(not(gust_profile = "cpu"))]
    fn the_kernels_carry_their_committed_artifacts() {
        // `gust-artifacts/` is made by `cargo gust build -p map-chain`; a changed kernel or a deleted file leaves a
        // kernel without its pair, and the strict profiles refuse to build it.
        for descriptor in descriptors() {
            assert_eq!(
                descriptor.artifacts.len(),
                1,
                "{} has no embedded artifact: run `cargo gust build -p map-chain`",
                descriptor.qualified_name()
            );
        }
    }

    #[test]
    fn an_empty_input_is_an_empty_result_and_a_call_of_broadcasts_has_no_length() {
        let device = CpuDevice::new();
        assert_eq!(twin(&device, &[]).unwrap(), Vec::<f32>::new());
        let error = rescale::map_on(&device, Broadcast(1.0), Broadcast(GAIN), Broadcast(OFFSET))
            .unwrap_err();
        assert!(matches!(error, MapError::Empty), "{error:?}");
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_chains_match_the_host_reference_on_every_size() {
        let handle = default_device().expect("Vulkan adapter");
        for len in SIZES {
            let x = input(len);
            let resident = resident(&handle, &x).unwrap();
            let materialised = materialised(&handle, &x).unwrap();
            assert_eq!(first_outside_bound(&x, &resident), None, "resident, {len}");
            assert_eq!(
                first_outside_bound(&x, &materialised),
                None,
                "materialised, {len}"
            );
            let same = resident
                .iter()
                .zip(&materialised)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "{len}: the two GPU chains agree bit for bit");
            let twin = twin(&CpuDevice::new(), &x).unwrap();
            assert_eq!(first_outside_bound(&x, &twin), None, "twin, {len}");
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_chains_match_the_host_reference_on_a_large_input() {
        let handle = default_device().expect("Vulkan adapter");
        let x = input(LARGE);
        let resident = resident(&handle, &x).unwrap();
        let materialised = materialised(&handle, &x).unwrap();
        assert_eq!(first_outside_bound(&x, &resident), None);
        assert_eq!(first_outside_bound(&x, &materialised), None);
        assert!(
            resident
                .iter()
                .zip(&materialised)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn the_resident_chain_uploads_once_and_reads_back_once() {
        // A device of its own, so no other test's buffers are counted. Every upload, result and read-back takes one
        // buffer from the device's pool (hits plus misses), and gust-wgpu has no counter of submissions.
        let handle: DeviceHandle = Arc::new(HeadlessDevice::new().expect("Vulkan adapter"));
        let x = input(1000);
        resident(&handle, &x).unwrap();
        let stats = handle.buffer_pool_stats();
        let before = stats.hits + stats.misses;
        let got = resident(&handle, &x).unwrap();
        let stats = handle.buffer_pool_stats();
        assert_eq!(stats.hits + stats.misses - before, RESIDENT_HAND_OUTS);
        assert_eq!(first_outside_bound(&x, &got), None);
        // The materialised chain uploads its argument, makes a result and reads it back in every step, and uploads
        // the broadcasts: 5 + 4 + 5.
        let before = stats.hits + stats.misses;
        materialised(&handle, &x).unwrap();
        let stats = handle.buffer_pool_stats();
        assert_eq!(stats.hits + stats.misses - before, MATERIALISED_HAND_OUTS);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_calls_of_broadcasts_have_no_length_and_empty_inputs_are_empty() {
        let handle = default_device().expect("Vulkan adapter");
        let error =
            rescale::map_gpu_on(&handle, Broadcast(1.0), Broadcast(GAIN), Broadcast(OFFSET))
                .unwrap_err();
        assert!(matches!(error, MapError::Empty), "{error:?}");
        assert_eq!(resident(&handle, &[]).unwrap(), Vec::<f32>::new());
        assert_eq!(materialised(&handle, &[]).unwrap(), Vec::<f32>::new());
    }
}
