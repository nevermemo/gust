// The literal in `unit` is 2^-24 written out exactly, as the dialect has no `const`.
#![allow(clippy::excessive_precision)]

use gust::{Independent, gpu_module};
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

/// Indices hashed by the demo and by the tests.
const COUNT: usize = 4096;

#[gpu_module]
mod rng {
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

    /// The top 24 bits of `h` as a float in `[0, 1)`. The literal is 2^-24.
    pub fn unit(h: u32) -> f32 {
        (h >> 8u32) as f32 * 0.000000059604644775390625
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn fill(
        id: SV_DispatchThreadID,
        seed: StructuredBuffer<uint>, // one element, bound with `Independent`
        mut words: RWStructuredBuffer<uint>,
        mut floats: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < words.len() {
            let h = hash(seed[0u32], i);
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
        let seed_buffer = device.create_typed_buffer("seed", &[seed], GpuBufferAccess::ReadOnly);
        let words = device.create_typed_buffer("words", &zeros_u, GpuBufferAccess::ReadWrite);
        let floats = device.create_typed_buffer("floats", &zeros_f, GpuBufferAccess::ReadWrite);
        rng::fill::dispatch(
            &device,
            COUNT as u32,
            Independent(&seed_buffer),
            &words,
            &floats,
        )?;
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
    fn reference_mix(x: u32) -> u32 {
        let mut h = x ^ (x >> 16);
        h = h.wrapping_mul(2146121005);
        h ^= h >> 15;
        h = h.wrapping_mul(2221713035);
        h ^ (h >> 16)
    }

    fn reference_hash(seed: u32, index: u32) -> u32 {
        reference_mix(reference_mix(seed) ^ index)
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
        assert_eq!(left.len(), COUNT, "{name}: seed {seed} left length");
        assert_eq!(right.len(), COUNT, "{name}: seed {seed} right length");
        if let Some(index) = (0..left.len()).find(|&i| left[i] != right[i]) {
            panic!(
                "{name}: seed {seed} index {index}: {:#010x} versus {:#010x}",
                left[index], right[index]
            );
        }
    }

    /// Known answers of lowbias32, computed in a separate Python script from the published constants and shifts
    /// (not from this file's code).
    #[test]
    fn mix_matches_known_answers() {
        for (x, want) in [
            (0u32, 0u32),
            (1, 0x6889_90c0),
            (0xdead_beef, 0xe628_c683),
            (u32::MAX, 0x6768_824a),
        ] {
            assert_eq!(rng::mix(x), want, "mix({x:#x})");
        }
    }

    #[test]
    fn the_literal_is_two_to_the_minus_24() {
        assert_eq!(0.000000059604644775390625f32, f32::from_bits(0x3380_0000));
        assert_eq!(f32::from_bits(0x3380_0000), 1.0 / 16_777_216.0);
    }

    #[test]
    fn different_seeds_are_not_shifted_copies() {
        // The old linear seed gave hash(s + 1, i) == hash(s, i + 0xa046cb53) and large overlaps at small offsets.
        for seed in [0u32, 1, 159_109, u32::MAX - 1] {
            assert_ne!(
                reference_hash(seed.wrapping_add(1), 0),
                reference_hash(seed, 0xa046_cb53)
            );
        }
        let streams: Vec<std::collections::HashSet<u32>> = [0u32, 1, 2, 159_109]
            .iter()
            .map(|&seed| reference(seed).0.into_iter().collect())
            .collect();
        for (a, left) in streams.iter().enumerate() {
            for right in &streams[a + 1..] {
                let shared = left.intersection(right).count();
                assert!(shared <= 4, "{shared} values shared between two seeds");
            }
        }
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
    fn twin_matches_reference() {
        let zeros_u = vec![0u32; COUNT];
        let zeros_f = vec![0.0f32; COUNT];
        for seed in [0u32, 1, u32::MAX] {
            let (want_words, want_floats) = reference(seed);
            let twin = CpuDevice::new();
            let seed_cpu = twin.buffer(&[seed]);
            let words_cpu = twin.buffer(&zeros_u);
            let floats_cpu = twin.buffer(&zeros_f);
            rng::fill::dispatch(
                &twin,
                COUNT as u32,
                Independent(&seed_cpu),
                &words_cpu,
                &floats_cpu,
            )
            .unwrap();
            first_difference("twin words", seed, &words_cpu.read(), &want_words);
            first_difference(
                "twin floats",
                seed,
                &bits(&floats_cpu.read()),
                &bits(&want_floats),
            );
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
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
            let seed_cpu = twin.buffer(&[seed]);
            let words_cpu = twin.buffer(&zeros_u);
            let floats_cpu = twin.buffer(&zeros_f);
            rng::fill::dispatch(
                &twin,
                COUNT as u32,
                Independent(&seed_cpu),
                &words_cpu,
                &floats_cpu,
            )
            .unwrap();
            first_difference("twin words", seed, &words_cpu.read(), &want_words);
            first_difference(
                "twin floats",
                seed,
                &bits(&floats_cpu.read()),
                &bits(&want_floats),
            );

            let seed_gpu = device.create_typed_buffer("seed", &[seed], GpuBufferAccess::ReadOnly);
            let words_gpu =
                device.create_typed_buffer("words", &zeros_u, GpuBufferAccess::ReadWrite);
            let floats_gpu =
                device.create_typed_buffer("floats", &zeros_f, GpuBufferAccess::ReadWrite);
            rng::fill::dispatch(
                &device,
                COUNT as u32,
                Independent(&seed_gpu),
                &words_gpu,
                &floats_gpu,
            )
            .unwrap();
            let gpu_words = device.read_typed_buffer(&words_gpu).unwrap();
            let gpu_floats = device.read_typed_buffer(&floats_gpu).unwrap();
            first_difference("gpu words", seed, &gpu_words, &want_words);
            first_difference("gpu floats", seed, &bits(&gpu_floats), &bits(&want_floats));
            println!("seed {seed}: {COUNT} indices bit-equal (reference, twin, GPU)");
        }
    }
}
