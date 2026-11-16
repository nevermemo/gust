//! The rust-gpu comparator `rustgpu-wgpu` (m10.b4): kernels written in Rust with rust-gpu 0.11.0
//! (`bench/rust-gpu-shaders/shader`, a nightly-only crate outside the bench workspace), built to one committed SPIR-V
//! module (`bench/rust-gpu-shaders/rustgpu.spv`, with `rustgpu.manifest.json` beside it) and loaded here through wgpu's
//! passthrough route (`Features::PASSTHROUGH_SHADERS`, `Device::create_shader_module_passthrough`), the call
//! `gust-wgpu` uses for its own SPIR-V route.
//!
//! The backend is `HandWgsl::rust_gpu`: the host side of `wgsl-prebuilt` (second device on the same adapter selection,
//! bind group and staging buffer built once, one submission per end-to-end sample, the same storage-buffer binding order
//! as the hand WGSL because the kernels were written to it), with the shader module this file creates. The module's
//! four kernels are W1 (`f32`), W2, W3 and W4; a fifth entry point, W1 on `u32`, is built and validated with the module
//! but no workload runs it. A workload without a kernel here (W5 to W9) is not run by this backend.
//!
//! The module declares the Vulkan memory model and the module's out-of-range policy is rust-gpu's: an invocation whose
//! slice index is out of range ends early (`docs/BENCHMARKS.md`, "Bounds policy").

use std::{borrow::Cow, sync::OnceLock};

use crate::workloads::Workload;

/// The committed module: little-endian SPIR-V words as bytes.
const MODULE: &[u8] = include_bytes!("../../rust-gpu-shaders/rustgpu.spv");

/// The manifest beside the module: toolchain, crate versions, entry points and the module's sha256.
#[cfg(test)]
const MANIFEST: &str = include_str!("../../rust-gpu-shaders/rustgpu.manifest.json");

/// Every entry point of the module, the names its listing in the passthrough descriptor needs (sorted, as the manifest).
pub const ENTRY_POINTS: [&str; 5] = ["w1_f32", "w1_u32", "w2_saxpy", "w3_horner", "w4_histogram"];

/// The workgroup size every kernel declares (`threads(256)`), the same as the hand WGSL's.
const WORKGROUP_SIZE: (u32, u32, u32) = (256, 1, 1);

/// The entry point that runs `workload`, if the module has one.
pub fn entry_point(workload: Workload) -> Option<&'static str> {
    match workload.family() {
        "W1" => Some("w1_f32"),
        "W2" => Some("w2_saxpy"),
        "W3" => Some("w3_horner"),
        "W4" => Some("w4_histogram"),
        _ => None,
    }
}

/// The module as SPIR-V words (the bytes are little-endian words; converted once).
fn words() -> &'static [u32] {
    static WORDS: OnceLock<Vec<u32>> = OnceLock::new();
    WORDS.get_or_init(|| {
        MODULE
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect()
    })
}

/// The module on `device`, which must have been opened with `PASSTHROUGH_SHADERS` (`WgpuContext::open_passthrough_on`).
pub fn create_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    let entry_points = ENTRY_POINTS
        .iter()
        .map(|name| wgpu::PassthroughShaderEntryPoint {
            name: (*name).into(),
            workgroup_size: WORKGROUP_SIZE,
        })
        .collect::<Vec<_>>();
    // SAFETY: the module is committed, `spirv-val --target-env vulkan1.1` clean (the `rgval` gate) and byte-checked against
    // its manifest by a test; passthrough hands it to the driver unchanged, so wgpu's own validation does not run on it.
    // Every sample of every run is checked against the serial reference by the harness.
    unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some("bench rust-gpu module"),
            entry_points: entry_points.into(),
            spirv: Some(Cow::Borrowed(words())),
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use bytemuck::cast_slice;

    use super::*;
    use crate::{
        backend::Backend,
        wgsl_wgpu::{HandWgsl, WgpuContext},
        workloads::{Judging, Rounding, check_with, gen_inputs, reference, test_size},
    };

    /// SHA-256 of `data`, as lowercase hex (a small FIPS 180-4 implementation: the bench has no hash dependency).
    fn sha256(data: &[u8]) -> String {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        let mut padded = data.to_vec();
        padded.push(0x80);
        while padded.len() % 64 != 56 {
            padded.push(0);
        }
        padded.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
        for block in padded.as_chunks::<64>().0 {
            let mut w = [0u32; 64];
            for (i, word) in block.as_chunks::<4>().0.iter().enumerate() {
                w[i] = u32::from_be_bytes(*word);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }
            let mut v = h;
            for i in 0..64 {
                let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
                let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
                let t1 = v[7]
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
                let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
                let t2 = s0.wrapping_add(maj);
                v = [
                    t1.wrapping_add(t2),
                    v[0],
                    v[1],
                    v[2],
                    v[3].wrapping_add(t1),
                    v[4],
                    v[5],
                    v[6],
                ];
            }
            for (state, value) in h.iter_mut().zip(v) {
                *state = state.wrapping_add(value);
            }
        }
        h.iter().map(|word| format!("{word:08x}")).collect()
    }

    #[test]
    fn the_sha256_helper_matches_the_standard_vectors() {
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// The committed module is the one its manifest describes: the sha256, the size, the entry points (the passthrough
    /// descriptor lists exactly these) and the toolchain pins.
    #[test]
    fn the_committed_module_matches_its_manifest() {
        let manifest: serde_json::Value = serde_json::from_str(MANIFEST).expect("manifest is JSON");
        assert_eq!(manifest["sha256"], sha256(MODULE), "module sha256");
        assert_eq!(manifest["bytes"], MODULE.len());
        let listed: Vec<&str> = manifest["entry_points"]
            .as_array()
            .expect("entry_points")
            .iter()
            .map(|name| name.as_str().expect("name"))
            .collect();
        assert_eq!(listed, ENTRY_POINTS);
        assert_eq!(manifest["toolchain"], "nightly-2026-08-15");
        assert_eq!(manifest["rust_gpu"], "0.11.0");
        assert_eq!(manifest["target"], "spirv-unknown-vulkan1.1");
        // A SPIR-V module: whole words, the magic number first.
        assert_eq!(MODULE.len() % 4, 0);
        assert_eq!(words()[0], 0x0723_0203);
    }

    /// Exactly W1 to W4 have a kernel; every other workload has none (the backend does not run it).
    #[test]
    fn the_module_has_kernels_for_w1_to_w4_only() {
        for workload in Workload::ALL {
            let want = matches!(workload.family(), "W1" | "W2" | "W3" | "W4");
            assert_eq!(entry_point(workload).is_some(), want, "{}", workload.id());
            if let Some(name) = entry_point(workload) {
                assert!(ENTRY_POINTS.contains(&name), "{name}");
                // Each kernel is the algorithm of the hand WGSL of its workload, so no workload of this backend is
                // outside the same-kernel pairing the way W7 is (the kill rule is GUST's and does not see this backend).
                assert!(workload.same_kernel(), "{}", workload.id());
            }
        }
    }

    fn passthrough_context() -> (std::sync::MutexGuard<'static, ()>, Rc<WgpuContext>) {
        static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = WgpuContext::open_passthrough_on(None)
            .expect("an adapter with passthrough shaders is needed for the rust-gpu tests");
        (guard, Rc::new(ctx))
    }

    /// Every workload the module has a kernel for, end to end, as one dispatch and as a 16-pass batch, judged by the
    /// same checker as every other GPU backend; the poison alone fails the check.
    #[test]
    fn every_rust_gpu_workload_passes_its_bound() {
        let (_gpu, ctx) = passthrough_context();
        let mut ran = 0;
        for w in Workload::ALL {
            let mut backend = HandWgsl::rust_gpu(Rc::clone(&ctx));
            assert_eq!(backend.id(), "rustgpu-wgpu");
            if !backend.supports(w) {
                continue;
            }
            ran += 1;
            let inputs = gen_inputs(w, test_size(w));
            let reference = reference(w, &inputs);
            backend.prepare(w, &inputs).unwrap();
            let judging = Judging {
                rounding: Rounding::Either,
                tree: None,
            };
            let mut out = Vec::new();
            backend.poison().unwrap();
            backend.read_back(&mut out).unwrap();
            assert!(
                check_with(w, &inputs, &reference, &out, judging).is_err(),
                "{} poison",
                w.id()
            );
            backend.poison().unwrap();
            backend.end_to_end(&inputs, &mut out).unwrap();
            check_with(w, &inputs, &reference, &out, judging)
                .unwrap_or_else(|e| panic!("{} end to end: {e}", w.id()));
            backend.poison().unwrap();
            backend.dispatch().unwrap();
            backend.read_back(&mut out).unwrap();
            check_with(w, &inputs, &reference, &out, judging)
                .unwrap_or_else(|e| panic!("{} dispatch: {e}", w.id()));
            backend.poison().unwrap();
            if backend.gpu_time(16).unwrap().is_some() {
                backend.read_back(&mut out).unwrap();
                check_with(w, &inputs, &reference, &out, judging)
                    .unwrap_or_else(|e| panic!("{} batch: {e}", w.id()));
            }
            // The cold start recompiles the module and runs once more.
            backend.cold_start().unwrap();
        }
        // W1, W2, W3 (degrees 4, 16, 64) and W4 (four cases).
        assert_eq!(ran, 9);
    }

    /// The module's `w1_u32` entry point (built and validated with the module, run by no workload): wrapping `u32` add on
    /// a size that is not a multiple of the workgroup, so the output-length guard is exercised.
    #[test]
    fn the_u32_vector_add_entry_point_runs() {
        let (_gpu, ctx) = passthrough_context();
        let n = 1000usize;
        let a: Vec<u32> = (0..n as u32)
            .map(|i| i.wrapping_mul(2_654_435_761))
            .collect();
        let b: Vec<u32> = (0..n as u32)
            .map(|i| i.wrapping_mul(40_503).wrapping_add(7))
            .collect();
        let bytes = (n * 4) as u64;
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let device = &ctx.device;
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage,
                mapped_at_creation: false,
            })
        };
        let (ba, bb) = (buffer("a", storage), buffer("b", storage));
        let bout = buffer("out", storage | wgpu::BufferUsages::COPY_SRC);
        let staging = buffer(
            "staging",
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        ctx.queue.write_buffer(&ba, 0, cast_slice(&a));
        ctx.queue.write_buffer(&bb, 0, cast_slice(&b));
        let entry = |binding: u32, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[entry(0, true), entry(1, true), entry(2, false)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = create_module(device);
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("w1_u32"),
            compilation_options: Default::default(),
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ba.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bb.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bout.as_entire_binding(),
                },
            ],
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(n.div_ceil(256) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&bout, 0, &staging, 0, bytes);
        ctx.queue.submit([encoder.finish()]);
        staging.map_async(wgpu::MapMode::Read, .., |result| result.expect("map"));
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let view = staging.get_mapped_range(..).unwrap();
        let got: Vec<u32> = cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        staging.unmap();
        for i in 0..n {
            assert_eq!(got[i], a[i].wrapping_add(b[i]), "element {i}");
        }
    }
}
