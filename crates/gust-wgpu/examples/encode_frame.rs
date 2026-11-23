//! The submission-count measurement of m14.encb: one frame of a two-dispatch batch, three ways, on one device.
//!
//! - `raw`: two compute passes of a plain wgpu pipeline in one encoder, one `queue.submit`, one wait.
//! - `submit_batch`: GUST's batch through `HeadlessDevice::submit_batch`, one wait.
//! - `encode_batch`: the same batch through `HeadlessDevice::encode_batch` into the caller's encoder, one
//!   `queue.submit` of the caller's, `EncodedJob::submitted`, one wait.
//!
//! Each size runs the three variants interleaved for 400 measured frames after 40 warm-up frames and prints the median
//! host time (`Instant`, from recording to the wait returning) of each, then the median GPU time of the GUST batch
//! where the adapter offers timestamp queries (`dispatch_batch_timed`). Output is plain text for a person to read; it
//! is not parsed by anything.

// A native measurement binary: the host clock is the point of it.
#![allow(clippy::disallowed_methods)]

use std::time::{Duration, Instant};

use gust::{
    Access, GpuPod, KernelDescriptor, ParameterDescriptor, ParameterKind, ResourceBinding,
    ResourceLayout,
};
use gust_wgpu::{BufferBinding, BufferDispatch, GpuBufferAccess, HeadlessDevice, wgpu};

const FRAMES: usize = 400;
const WARM_UP: usize = 40;

const fn storage(name: &'static str, access: Access, binding: u32) -> ParameterDescriptor {
    ParameterDescriptor {
        name,
        rust_type: "uint",
        kind: ParameterKind::Storage,
        access,
        resource_layout: Some(ResourceLayout::storage_v1(<u32 as GpuPod>::LAYOUT)),
        binding: Some(ResourceBinding { group: 0, binding }),
    }
}

/// `add_one(src: RO, dst: RW)`: `dst[i] = src[i] + 1`.
const ADD_ONE: KernelDescriptor = KernelDescriptor::new(
    "encode_frame",
    "add_one",
    "encode_frame_add_one",
    [64, 1, 1],
    &[
        storage("src", Access::ReadOnly, 0),
        storage("dst", Access::ReadWrite, 1),
    ],
    r#"
[[vk::binding(0, 0)]] StructuredBuffer<uint> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> dst;
[shader("compute")] [numthreads(64, 1, 1)]
void encode_frame_add_one(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; dst.GetDimensions(n, stride);
    if (id.x < n) { dst[id.x] = src[id.x] + 1u; }
}
"#,
);

/// The same kernel in WGSL, for the raw variant.
const RAW_WGSL: &str = "
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&dst)) { dst[id.x] = src[id.x] + 1u; }
}
";

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn main() {
    let device = HeadlessDevice::new().expect("a Vulkan adapter is required");
    println!("adapter: {}", device.adapter_info().name);
    for exponent in [12u32, 20] {
        measure(&device, 1u32 << exponent, exponent);
    }
}

fn measure(device: &HeadlessDevice, n: u32, exponent: u32) {
    let values: Vec<u32> = (0..n).collect();
    let zeros = vec![0u32; n as usize];
    let a = device.create_typed_buffer("a", &values, GpuBufferAccess::ReadWrite);
    let b = device.create_typed_buffer("b", &zeros, GpuBufferAccess::ReadWrite);
    let c = device.create_typed_buffer("c", &zeros, GpuBufferAccess::ReadWrite);
    let first = [BufferBinding::read_only(&a), BufferBinding::read_write(&b)];
    let second = [BufferBinding::read_only(&b), BufferBinding::read_write(&c)];
    let batch = [
        BufferDispatch::new(&ADD_ONE, n, &first),
        BufferDispatch::new(&ADD_ONE, n, &second),
    ];

    let raw = RawPipeline::new(device);
    let raw_first = raw.bind_group(device, a.as_wgpu_buffer(), b.as_wgpu_buffer());
    let raw_second = raw.bind_group(device, b.as_wgpu_buffer(), c.as_wgpu_buffer());
    let groups = n.div_ceil(64);

    let (mut raw_times, mut submit_times, mut encode_times) = (Vec::new(), Vec::new(), Vec::new());
    let mut gpu_times = Vec::new();
    for frame in 0..WARM_UP + FRAMES {
        let started = Instant::now();
        let mut encoder = device
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for group in [&raw_first, &raw_second] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&raw.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        let index = device.wgpu_queue().submit([encoder.finish()]);
        device
            .wgpu_device()
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            })
            .expect("poll");
        let raw_time = started.elapsed();

        let started = Instant::now();
        device
            .submit_batch(&batch)
            .expect("submit_batch")
            .wait()
            .expect("wait");
        let submit_time = started.elapsed();

        let started = Instant::now();
        let mut encoder = device
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let encoded = device
            .encode_batch(&mut encoder, &batch)
            .expect("encode_batch");
        let index = device.wgpu_queue().submit([encoder.finish()]);
        encoded.submitted(&index).wait().expect("wait");
        let encode_time = started.elapsed();

        if frame >= WARM_UP {
            raw_times.push(raw_time);
            submit_times.push(submit_time);
            encode_times.push(encode_time);
            if device.supports_gpu_timestamps()
                && let Some(gpu) = device
                    .dispatch_batch_timed(&batch)
                    .expect("timed")
                    .gpu_elapsed
            {
                gpu_times.push(gpu);
            }
        }
    }
    let raw_median = median(&mut raw_times);
    let submit_median = median(&mut submit_times);
    let encode_median = median(&mut encode_times);
    println!("2^{exponent} elements, 2 dispatches, {FRAMES} frames, median host time per frame:");
    println!(
        "  raw wgpu, one submit      {:8.4} ms",
        milliseconds(raw_median)
    );
    println!(
        "  submit_batch              {:8.4} ms",
        milliseconds(submit_median)
    );
    println!(
        "  encode_batch, one submit  {:8.4} ms",
        milliseconds(encode_median)
    );
    println!(
        "  encode_batch - raw        {:+8.4} ms",
        milliseconds(encode_median) - milliseconds(raw_median)
    );
    if gpu_times.is_empty() {
        println!("  GPU timestamps: not offered by this adapter");
    } else {
        println!(
            "  GPU time of the GUST batch (dispatch_batch_timed, median of {}): {:.4} ms",
            gpu_times.len(),
            milliseconds(median(&mut gpu_times))
        );
    }
}

/// A plain wgpu compute pipeline of the kernel, bound over the GUST buffers.
struct RawPipeline {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl RawPipeline {
    fn new(device: &HeadlessDevice) -> Self {
        let wgpu_device = device.wgpu_device();
        let module = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raw add_one"),
            source: wgpu::ShaderSource::Wgsl(RAW_WGSL.into()),
        });
        let entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raw add_one layout"),
            entries: &[entry(0, true), entry(1, false)],
        });
        let pipeline_layout = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raw add_one pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raw add_one"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { layout, pipeline }
    }

    fn bind_group(
        &self,
        device: &HeadlessDevice,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device
            .wgpu_device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raw add_one bind group"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                ],
            })
    }
}
