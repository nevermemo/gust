//! The host-side cost of a three-dispatch frame (update, chunk scan, compaction) over 2^20 elements, as GUST runs it
//! and as raw wgpu runs the same WGSL.
//!
//! ```text
//! GUST_PROFILE=dev cargo run --release --manifest-path bench/Cargo.toml -p graph-overhead --bin batch_shape
//! ```
//!
//! One frame is the three kernels in order, in one submission. Four ways to run it:
//!
//! - `raw`: wgpu pipelines built from the WGSL GUST itself compiles for the kernels (`compile_reflected`), the bind
//!   group layout GUST builds, one encoder with three compute passes, one `queue.submit`.
//! - `batch`: `HeadlessDevice::submit_batch` of three `BufferDispatch`es, a bind group built for each every frame.
//! - `graph`: a `StagedGraph` of three `run` nodes, executed again and again (bind groups kept by the graph).
//! - `graph, one-shot`: a new `StagedGraph` and its execution every frame (a node's kept group is built once).
//! - `3 bind groups`: not a way to run the frame but a part of one: three `create_bind_group` calls for the raw layouts
//!   and buffers, which `batch` pays every frame and a re-executed `graph` does not.
//!
//! Each row is the median of three runs of the mean over the frames of the wall time and of the CPU time of the
//! calling thread (Windows only), per frame. `serial` waits for each frame before the next; `pipelined` waits for the
//! frame before the previous one (two frames in flight), which is how the renderer spike that found the gap ran. The
//! graph has no non-blocking call, so it has the serial rows only. The program asserts no time; every way's result is
//! compared with a host reference once. A one-machine measurement, never a speed claim.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::time::Instant;

use graph_overhead::{median, thread_cpu_ns};
use gust::{Access, Independent, KernelDescriptor};
use gust_wgpu::wgpu;
use gust_wgpu::{
    BufferBinding, BufferDispatch, GpuBuffer, GpuBufferAccess, HeadlessDevice, StagedGraph, WgpuJob,
};

#[gust::gpu_module]
mod frame {
    /// val = x ^ (x >> 5); flag = 1 when the low two bits of val are zero.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn update(
        id: SV_DispatchThreadID,
        x: StructuredBuffer<uint>,
        mut val: RWStructuredBuffer<uint>,
        mut flag: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < x.len() {
            let v = x[i] ^ (x[i] >> 5u32);
            val[i] = v;
            let mut f = 0u32;
            if (v & 3u32) == 0u32 {
                f = 1u32;
            }
            flag[i] = f;
        }
    }

    /// One invocation per chunk of 256: the exclusive scan of the flags of its chunk and the chunk's sum.
    #[allow(clippy::assign_op_pattern)]
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn scan_chunks(
        id: SV_DispatchThreadID,
        flag: StructuredBuffer<uint>,
        mut starts: RWStructuredBuffer<uint>,
        mut sums: RWStructuredBuffer<uint>,
    ) {
        let chunk = id.x;
        if chunk < sums.len() {
            let mut acc = 0u32;
            for k in 0u32..256u32 {
                let i = chunk * 256u32 + k;
                if i < flag.len() {
                    starts[i] = acc;
                    acc = acc + flag[i];
                }
            }
            sums[chunk] = acc;
        }
    }

    /// A flagged element goes to slot (chunk start + its exclusive scan within the chunk).
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn compact(
        id: SV_DispatchThreadID,
        val: StructuredBuffer<uint>,
        flag: StructuredBuffer<uint>,
        starts: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < val.len() && flag[i] == 1u32 {
            out[(i / 256u32) * 256u32 + starts[i]] = val[i];
        }
    }
}

const ELEMENTS: usize = 1 << 20;
const CHUNKS: usize = ELEMENTS / 256;
const WARM_UP: usize = 100;
const RUNS: usize = 3;
const FRAMES_PER_RUN: usize = 400;

fn input() -> Vec<u32> {
    (0..ELEMENTS as u32)
        .map(|i| i.wrapping_mul(2_654_435_761).rotate_left(7) ^ 0x9E37_79B9)
        .collect()
}

/// The host reference of the compaction's output (`out`); slots a chunk does not fill stay zero.
fn reference(x: &[u32]) -> Vec<u32> {
    let mut out = vec![0u32; ELEMENTS];
    for chunk in 0..CHUNKS {
        let mut acc = 0usize;
        for i in chunk * 256..(chunk + 1) * 256 {
            let v = x[i] ^ (x[i] >> 5);
            if v & 3 == 0 {
                out[chunk * 256 + acc] = v;
                acc += 1;
            }
        }
    }
    out
}

struct Rig<'d> {
    device: &'d HeadlessDevice,
    x: GpuBuffer<u32>,
    val: GpuBuffer<u32>,
    flag: GpuBuffer<u32>,
    starts: GpuBuffer<u32>,
    sums: GpuBuffer<u32>,
    out: GpuBuffer<u32>,
}

impl<'d> Rig<'d> {
    fn new(device: &'d HeadlessDevice) -> Self {
        let zeros = |label: &str, len: usize| {
            device.create_typed_buffer(label, &vec![0u32; len], GpuBufferAccess::ReadWrite)
        };
        Self {
            device,
            x: device.create_typed_buffer("x", &input(), GpuBufferAccess::ReadWrite),
            val: zeros("val", ELEMENTS),
            flag: zeros("flag", ELEMENTS),
            starts: zeros("starts", ELEMENTS),
            sums: zeros("sums", CHUNKS),
            out: zeros("out", ELEMENTS),
        }
    }

    fn clear_out(&self) {
        self.device
            .write_typed_buffer(&self.out, &vec![0u32; ELEMENTS])
            .expect("clear out");
    }

    fn check(&self, what: &str) {
        let got = self.device.read_typed_buffer(&self.out).expect("read out");
        assert!(got == reference(&input()), "{what}: out differs from the host reference");
    }
}

/// A raw wgpu pipeline and bind group of one kernel, built from the WGSL GUST compiles for it.
struct RawKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    groups: u32,
    /// What `bind_group` was made from, to make another one.
    layout: wgpu::BindGroupLayout,
    bound: Vec<(u32, wgpu::Buffer)>,
}

impl RawKernel {
    /// Another bind group of the same layout and buffers.
    fn new_bind_group(&self, device: &HeadlessDevice) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = self
            .bound
            .iter()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        device
            .wgpu_device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raw bind group"),
                layout: &self.layout,
                entries: &entries,
            })
    }
}

fn raw_kernel(
    device: &HeadlessDevice,
    kernel: &KernelDescriptor,
    buffers: &[&GpuBuffer<u32>],
    invocations: u32,
) -> RawKernel {
    let compiled =
        gust::reflect::compile_reflected(kernel, gust::slang::Target::Wgsl).expect("compile");
    let wgsl = std::str::from_utf8(&compiled.artifact)
        .expect("utf-8 WGSL")
        .to_owned();
    let gpu = device.wgpu_device();
    let params: Vec<_> = kernel
        .parameters
        .iter()
        .filter(|parameter| parameter.binding.is_some())
        .collect();
    assert_eq!(params.len(), buffers.len());
    let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = params
        .iter()
        .map(|parameter| wgpu::BindGroupLayoutEntry {
            binding: parameter.binding.unwrap().binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: parameter.access == Access::ReadOnly,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let bind_group_layout = gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("raw layout"),
        entries: &layout_entries,
    });
    let pipeline_layout = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("raw pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let module = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raw shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(wgsl)),
    });
    let pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("raw pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(kernel.entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    let entries: Vec<wgpu::BindGroupEntry> = params
        .iter()
        .zip(buffers)
        .map(|(parameter, buffer)| wgpu::BindGroupEntry {
            binding: parameter.binding.unwrap().binding,
            resource: buffer.as_wgpu_buffer().as_entire_binding(),
        })
        .collect();
    let bind_group = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raw bind group"),
        layout: &bind_group_layout,
        entries: &entries,
    });
    RawKernel {
        pipeline,
        bind_group,
        groups: invocations.div_ceil(kernel.workgroup_size[0]),
        layout: bind_group_layout,
        bound: params
            .iter()
            .zip(buffers)
            .map(|(parameter, buffer)| {
                (
                    parameter.binding.unwrap().binding,
                    buffer.as_wgpu_buffer().clone(),
                )
            })
            .collect(),
    }
}

fn raw_frame(device: &HeadlessDevice, kernels: &[RawKernel]) -> wgpu::SubmissionIndex {
    let mut encoder = device
        .wgpu_device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    for kernel in kernels {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raw pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &kernel.bind_group, &[]);
        pass.dispatch_workgroups(kernel.groups, 1, 1);
    }
    device.wgpu_queue().submit([encoder.finish()])
}

fn wait_raw(device: &HeadlessDevice, submission: wgpu::SubmissionIndex) {
    device
        .wgpu_device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("poll");
}

/// Median over the runs of (mean wall us, mean thread CPU us) of one call of `frame`. `frame` runs one whole frame,
/// including the waiting its mode needs.
fn measure(mut frame: impl FnMut()) -> (f64, f64) {
    for _ in 0..WARM_UP {
        frame();
    }
    let mut wall = [0.0; RUNS];
    let mut cpu = [0.0; RUNS];
    for run in 0..RUNS {
        let cpu_before = thread_cpu_ns();
        let started = Instant::now();
        for _ in 0..FRAMES_PER_RUN {
            frame();
        }
        wall[run] = started.elapsed().as_secs_f64() * 1e6 / FRAMES_PER_RUN as f64;
        cpu[run] = (thread_cpu_ns() - cpu_before) as f64 / 1e3 / FRAMES_PER_RUN as f64;
    }
    (median(wall), median(cpu))
}

fn row(name: &str, (wall, cpu): (f64, f64)) {
    println!("{name:<22} wall {wall:>8.1} us   thread cpu {cpu:>8.1} us");
}

fn main() {
    let device = HeadlessDevice::new().expect("an adapter");
    let info = device.adapter_info();
    println!(
        "adapter: {} ({:?}); {ELEMENTS} elements; {WARM_UP} warm-up frames, median of {RUNS} runs of {FRAMES_PER_RUN} frames",
        info.name, info.backend
    );
    let rig = Rig::new(&device);

    let raw = [
        raw_kernel(
            &device,
            &frame::update::DESCRIPTOR,
            &[&rig.x, &rig.val, &rig.flag],
            ELEMENTS as u32,
        ),
        raw_kernel(
            &device,
            &frame::scan_chunks::DESCRIPTOR,
            &[&rig.flag, &rig.starts, &rig.sums],
            CHUNKS as u32,
        ),
        raw_kernel(
            &device,
            &frame::compact::DESCRIPTOR,
            &[&rig.val, &rig.flag, &rig.starts, &rig.out],
            ELEMENTS as u32,
        ),
    ];
    let bindings = [
        vec![
            BufferBinding::read_only(&rig.x),
            BufferBinding::read_write(&rig.val),
            BufferBinding::read_write(&rig.flag),
        ],
        vec![
            BufferBinding::read_only(&rig.flag).independent_length(),
            BufferBinding::read_write(&rig.starts).independent_length(),
            BufferBinding::read_write(&rig.sums),
        ],
        vec![
            BufferBinding::read_only(&rig.val),
            BufferBinding::read_only(&rig.flag),
            BufferBinding::read_only(&rig.starts),
            BufferBinding::read_write(&rig.out),
        ],
    ];
    let dispatches = [
        BufferDispatch::new(&frame::update::DESCRIPTOR, ELEMENTS as u32, &bindings[0]),
        BufferDispatch::new(&frame::scan_chunks::DESCRIPTOR, CHUNKS as u32, &bindings[1]),
        BufferDispatch::new(&frame::compact::DESCRIPTOR, ELEMENTS as u32, &bindings[2]),
    ];
    let update = frame::update::call(ELEMENTS as u32, &rig.x, &rig.val, &rig.flag);
    let scan = frame::scan_chunks::call(
        CHUNKS as u32,
        Independent(&rig.flag),
        Independent(&rig.starts),
        &rig.sums,
    );
    let compact = frame::compact::call(ELEMENTS as u32, &rig.val, &rig.flag, &rig.starts, &rig.out);
    let mut graph = StagedGraph::new();
    graph.run(&update);
    graph.run(&scan);
    graph.run(&compact);

    // Exactness, once per way.
    rig.clear_out();
    wait_raw(&device, raw_frame(&device, &raw));
    rig.check("raw");
    rig.clear_out();
    device.dispatch_batch(&dispatches).expect("batch");
    rig.check("batch");
    rig.clear_out();
    device.execute_graph(&graph).expect("graph");
    rig.check("graph");

    row(
        "raw serial",
        measure(|| wait_raw(&device, raw_frame(&device, &raw))),
    );
    row(
        "batch serial",
        measure(|| {
            device
                .submit_batch(&dispatches)
                .expect("batch")
                .wait()
                .expect("wait");
        }),
    );
    row(
        "graph serial",
        measure(|| {
            device.execute_graph(&graph).expect("graph");
        }),
    );
    row(
        "graph, one-shot serial",
        measure(|| {
            let mut once = StagedGraph::new();
            once.run(&update);
            once.run(&scan);
            once.run(&compact);
            device.execute_graph(&once).expect("graph");
        }),
    );

    // What one frame of `batch` pays for bind groups that the graph keeps: three `create_bind_group` calls (of the raw
    // layouts and buffers), a hundred rounds per call of the closure so that the coarse CPU clock can resolve them.
    const ROUNDS: usize = 100;
    let (wall, cpu) = measure(|| {
        for _ in 0..ROUNDS {
            for kernel in &raw {
                std::hint::black_box(kernel.new_bind_group(&device));
            }
        }
    });
    row(
        "3 bind groups",
        (wall / ROUNDS as f64, cpu / ROUNDS as f64),
    );

    let mut in_flight = VecDeque::new();
    row(
        "raw pipelined",
        measure(|| {
            if in_flight.len() == 2 {
                wait_raw(&device, in_flight.pop_front().expect("two in flight"));
            }
            in_flight.push_back(raw_frame(&device, &raw));
        }),
    );
    while let Some(submission) = in_flight.pop_front() {
        wait_raw(&device, submission);
    }
    let mut jobs: VecDeque<WgpuJob<'_>> = VecDeque::new();
    row(
        "batch pipelined",
        measure(|| {
            if jobs.len() == 2 {
                jobs.pop_front()
                    .expect("two in flight")
                    .wait()
                    .expect("wait");
            }
            jobs.push_back(device.submit_batch(&dispatches).expect("batch"));
        }),
    );
    while let Some(job) = jobs.pop_front() {
        job.wait().expect("wait");
    }
}
