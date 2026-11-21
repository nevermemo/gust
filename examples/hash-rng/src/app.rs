// The literal in `unit` is 2^-24 written out exactly, as the dialect has no `const`.
#![allow(clippy::excessive_precision)]

use gust::gpu;
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

/// Indices hashed by the demo and by the tests.
const COUNT: usize = 4096;

#[gpu]
mod rng {
    /// A multiply-xorshift hash of a seed and an index. Every shift amount is a literal below 32 (D39).
    pub fn hash(seed: u32, index: u32) -> u32 {
        let mut h = seed
            .wrapping_mul(2654435761u32)
            .wrapping_add(index.wrapping_mul(2246822507u32))
            .wrapping_add(668265263u32);
        h = h ^ (h >> 15u32);
        h = h.wrapping_mul(739982445u32);
        h = h ^ (h >> 12u32);
        h = h.wrapping_mul(695872825u32);
        h ^ (h >> 15u32)
    }

    /// The top 24 bits of `h` as a float in `[0, 1)`. The literal is 2^-24.
    pub fn unit(h: u32) -> f32 {
        (h >> 8u32) as f32 * 0.000000059604644775390625
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn fill(
        id: SV_DispatchThreadID,
        seed: StructuredBuffer<uint>,
        mut words: RWStructuredBuffer<uint>,
        mut floats: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < words.len() {
            let h = hash(seed[i], i);
            words[i] = h;
            floats[i] = unit(h);
        }
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = HeadlessDevice::new()?;
    let info = device.adapter_info();
    println!("adapter: {} backend: {:?}", info.name, info.backend);
    let zeros_u = vec![0u32; COUNT];
    let zeros_f = vec![0.0f32; COUNT];
    for seed in [0u32, 1, u32::MAX] {
        let seed_buffer =
            device.create_typed_buffer("seed", &vec![seed; COUNT], GpuBufferAccess::ReadOnly);
        let words = device.create_typed_buffer("words", &zeros_u, GpuBufferAccess::ReadWrite);
        let floats = device.create_typed_buffer("floats", &zeros_f, GpuBufferAccess::ReadWrite);
        rng::fill::dispatch(&device, COUNT as u32, &seed_buffer, &words, &floats)?;
        let words = device.read_typed_buffer(&words)?;
        let floats = device.read_typed_buffer(&floats)?;
        let mean = floats.iter().map(|&x| f64::from(x)).sum::<f64>() / COUNT as f64;
        println!(
            "seed {seed}: first hash {:#010x}, first unit {}, mean {mean:.4}",
            words[0], floats[0]
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gust::cpu::CpuDevice;

    /// The reference: plain Rust outside the module, with its own copy of the arithmetic.
    fn reference_hash(seed: u32, index: u32) -> u32 {
        let mut h = seed
            .wrapping_mul(2654435761)
            .wrapping_add(index.wrapping_mul(2246822507))
            .wrapping_add(668265263);
        h ^= h >> 15;
        h = h.wrapping_mul(739982445);
        h ^= h >> 12;
        h = h.wrapping_mul(695872825);
        h ^ (h >> 15)
    }

    fn reference_unit(h: u32) -> f32 {
        (h >> 8) as f32 * f32::from_bits(0x3380_0000)
    }

    fn reference(seed: u32) -> (Vec<u32>, Vec<f32>) {
        let words: Vec<u32> = (0..COUNT as u32).map(|i| reference_hash(seed, i)).collect();
        let floats = words.iter().map(|&h| reference_unit(h)).collect();
        (words, floats)
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|x| x.to_bits()).collect()
    }

    fn first_difference(name: &str, seed: u32, left: &[u32], right: &[u32]) {
        if let Some(index) = (0..left.len()).find(|&i| left[i] != right[i]) {
            panic!(
                "{name}: seed {seed} index {index}: {:#010x} versus {:#010x}",
                left[index], right[index]
            );
        }
    }

    #[test]
    fn the_literal_is_two_to_the_minus_24() {
        assert_eq!(0.000000059604644775390625f32, f32::from_bits(0x3380_0000));
        assert_eq!(f32::from_bits(0x3380_0000), 1.0 / 16_777_216.0);
    }

    #[test]
    fn units_are_below_one_with_a_central_mean() {
        let (_, floats) = reference(1);
        assert!(floats.iter().all(|&x| (0.0..1.0).contains(&x)));
        let mean = floats.iter().map(|&x| f64::from(x)).sum::<f64>() / COUNT as f64;
        assert!((0.45..0.55).contains(&mean), "mean {mean}");
        let top = reference_unit(u32::MAX);
        assert_eq!(top.to_bits(), 0x3f7f_ffff);
        assert!(top < 1.0);
    }

    #[test]
    fn reference_twin_and_gpu_are_bit_equal() {
        let device = HeadlessDevice::new().unwrap_or_else(|error| {
            panic!("Vulkan adapter required; this test must not silently skip: {error}")
        });
        let info = device.adapter_info();
        println!("adapter: {} backend: {:?}", info.name, info.backend);
        let zeros_u = vec![0u32; COUNT];
        let zeros_f = vec![0.0f32; COUNT];
        for seed in [0u32, 1, u32::MAX] {
            let (want_words, want_floats) = reference(seed);

            let twin = CpuDevice::new();
            let seed_cpu = twin.buffer(&vec![seed; COUNT]);
            let words_cpu = twin.buffer(&zeros_u);
            let floats_cpu = twin.buffer(&zeros_f);
            rng::fill::dispatch(&twin, COUNT as u32, &seed_cpu, &words_cpu, &floats_cpu).unwrap();
            first_difference("twin words", seed, &words_cpu.read(), &want_words);
            first_difference(
                "twin floats",
                seed,
                &bits(&floats_cpu.read()),
                &bits(&want_floats),
            );

            let seed_gpu =
                device.create_typed_buffer("seed", &vec![seed; COUNT], GpuBufferAccess::ReadOnly);
            let words_gpu =
                device.create_typed_buffer("words", &zeros_u, GpuBufferAccess::ReadWrite);
            let floats_gpu =
                device.create_typed_buffer("floats", &zeros_f, GpuBufferAccess::ReadWrite);
            rng::fill::dispatch(&device, COUNT as u32, &seed_gpu, &words_gpu, &floats_gpu).unwrap();
            let gpu_words = device.read_typed_buffer(&words_gpu).unwrap();
            let gpu_floats = device.read_typed_buffer(&floats_gpu).unwrap();
            first_difference("gpu words", seed, &gpu_words, &want_words);
            first_difference("gpu floats", seed, &bits(&gpu_floats), &bits(&want_floats));
            println!("seed {seed}: {COUNT} indices bit-equal (reference, twin, GPU)");
        }
    }
}
