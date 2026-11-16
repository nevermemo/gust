//! The rust-gpu comparator `rustgpu-wgpu` (m10.b4): kernels written in Rust with rust-gpu 0.11.0
//! (`bench/rust-gpu-shaders/shader`, a nightly-only crate outside the bench workspace), built to one committed SPIR-V
//! module (`bench/rust-gpu-shaders/rustgpu.spv`, with `rustgpu.manifest.json` beside it) and loaded here through wgpu's
//! passthrough route (`Features::PASSTHROUGH_SHADERS`, `Device::create_shader_module_passthrough`), the call
//! `gust-wgpu` uses for its own SPIR-V route.
//!
//! The backend is `HandWgsl::rust_gpu`: the host side of `wgsl-prebuilt` (its own device, the run's third, on the same
//! adapter selection, bind group and staging buffer built once, one submission per end-to-end sample, the same
//! storage-buffer binding order as the hand WGSL because the kernels were written to it), with the shader module this
//! file creates. The module's four kernels are W1 (`f32`), W2, W3 and W4; a fifth entry point, W1 on `u32`, is built and
//! validated with the module but no workload runs it. A workload without a kernel here (W5 to W9) is not run by this
//! backend.
//!
//! The module declares the Vulkan memory model, which wgpu-hal 30.0.2 does not enable on its devices, so the backend
//! runs only on a device whose Vulkan API version is 1.3 or later, where support is required
//! (`WgpuContext::open_passthrough_on`, `docs/BENCHMARKS.md`), and the module's out-of-range policy is rust-gpu's: an
//! invocation whose slice index is out of range ends early (`docs/BENCHMARKS.md`, "Bounds policy").

use std::{borrow::Cow, sync::OnceLock};

use crate::workloads::Workload;

use crate::rustgpu_module::ENTRY_POINTS;
use crate::rustgpu_module::MODULE;

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

    /// Exactly W1 to W4 have a kernel; every other workload has none (the backend does not run it).
    #[test]
    fn the_module_has_kernels_for_w1_to_w4_only() {
        for workload in Workload::ALL {
            let want = matches!(workload.family(), "W1" | "W2" | "W3" | "W4");
            assert_eq!(entry_point(workload).is_some(), want, "{}", workload.id());
            if let Some(name) = entry_point(workload) {
                assert!(ENTRY_POINTS.contains(&name), "{name}");
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
