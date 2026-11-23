//! SB1 (plan m14.s2, slice s2a2): what does a `gpu_batch!` frame cost on the host, against the same ten dispatches
//! written by hand in raw wgpu?
//!
//! The frame is S1 Q4's ten stages over `u32` data: seven per-element stages, a block stage of 256, an add of the block
//! value, and one partial per 1024 elements with one ticket on the partials. The raw version has the same ten compute
//! passes (one per stage, as the batch records them), written in WGSL by hand with the same arithmetic (the batch's
//! WGSL comes from Slang, so it is not byte-identical), its pipelines and bind groups made once, one encoder, one copy
//! of the partials and one submit.
//!
//! The span measured is the host time from `begin` through `submit` of an unchanged frame (no writes, no `set`); the
//! wait for the GPU is outside it. Protocol: 20 warm-up frames, then the median over 5 runs of the mean of 200 frames.
//! The two loops alternate their order from run to run. This is the host span only: no GPU pass timestamps are taken,
//! and the figures are those of a release build under `GUST_PROFILE=dev` (a release build infers the `native` profile,
//! which refuses the run-time Slang bridge the batch needs).
//! Run with `GUST_PROFILE=dev cargo run --release -p batch-basics --bin sb1 -- <part of the adapter name>`.

use std::sync::Arc;
use std::time::Instant;

use batch_basics::app::block_on;
use gust::batch::{Batch, Error, Stage};
use gust::{gpu_batch, gpu_task};
use gust_wgpu::{AdapterChoice, DeviceOptions, HeadlessDevice, wgpu};

const WARMUP: usize = 20;
const RUNS: usize = 5;
const FRAMES: usize = 200;

struct Handles {
    x: Stage<u32, gust::batch::D1>,
    part: Stage<u32, gust::batch::D1>,
}

fn build(device: &Arc<HeadlessDevice>, n: u32) -> Result<(Batch<HeadlessDevice>, Handles), Error> {
    gpu_batch!("sb1", on(device); in_flight(1); |b| {
        let x = b.input::<u32>(n);
        let s1 = gpu_task!(n, |i| -> u32 { (x[i] * 3u32 + 1u32) % 65521u32 });
        let s2 = gpu_task!(n, |i| -> u32 { (s1[i] * 5u32 + 2u32) % 65521u32 });
        let s3 = gpu_task!(n, |i| -> u32 { (s2[i] * 7u32 + 3u32) % 65521u32 });
        let s4 = gpu_task!(n, |i| -> u32 { (s3[i] * 9u32 + 4u32) % 65521u32 });
        let s5 = gpu_task!(n, |i| -> u32 { (s4[i] * 11u32 + 5u32) % 65521u32 });
        let s6 = gpu_task!(n, |i| -> u32 { (s5[i] * 13u32 + 6u32) % 65521u32 });
        let s7 = gpu_task!(n, |i| -> u32 { (s6[i] * 15u32 + 7u32) % 65521u32 });
        let blk = gpu_task!(n, |i| -> u32 { s7[i] ^ s7[(i / 256u32) * 256u32] });
        let sum = gpu_task!(n, |i| -> u32 { blk[i] + s7[(i / 256u32) * 256u32] });
        let part = gpu_task!(n / 1024u32, |j| -> u32 {
            sum[j * 1024u32] + sum[j * 1024u32 + 256u32] + sum[j * 1024u32 + 512u32] + sum[j * 1024u32 + 768u32]
        });
        Handles { x, part }
    })
}

/// The host reference of the partials.
fn reference(input: &[u32]) -> Vec<u32> {
    let step =
        |v: &[u32], m: u32, c: u32| -> Vec<u32> { v.iter().map(|v| (v * m + c) % 65521).collect() };
    let mut v = input.to_vec();
    for (m, c) in [(3, 1), (5, 2), (7, 3), (9, 4), (11, 5), (13, 6), (15, 7)] {
        v = step(&v, m, c);
    }
    let blk: Vec<u32> = (0..v.len()).map(|i| v[i] ^ v[i / 256 * 256]).collect();
    let sum: Vec<u32> = (0..v.len()).map(|i| blk[i] + v[i / 256 * 256]).collect();
    (0..v.len() / 1024)
        .map(|j| sum[j * 1024] + sum[j * 1024 + 256] + sum[j * 1024 + 512] + sum[j * 1024 + 768])
        .collect()
}

fn wgsl(body: &str) -> String {
    format!(
        "struct P {{ n: u32, a: u32, b: u32, c: u32 }}\n\
         @group(0) @binding(0) var<uniform> p: P;\n\
         @group(0) @binding(1) var<storage, read> src: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> dst: array<u32>;\n\
         @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {{\n\
         let i = g.x; if (i >= p.n) {{ return; }}\n{body}\n}}\n"
    )
}

struct Raw {
    device: wgpu::Device,
    queue: wgpu::Queue,
    passes: Vec<(wgpu::ComputePipeline, wgpu::BindGroup, u32)>,
    part: wgpu::Buffer,
    staging: wgpu::Buffer,
    input: wgpu::Buffer,
    part_bytes: u64,
}

fn raw(gust_device: &HeadlessDevice, n: u32) -> Raw {
    let device = gust_device.wgpu_device().clone();
    let queue = gust_device.wgpu_queue().clone();
    let storage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let make = |len: u64, usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: len * 4,
            usage,
            mapped_at_creation: false,
        })
    };
    let n64 = u64::from(n);
    let input = make(n64, storage);
    let chain: Vec<wgpu::Buffer> = (0..9).map(|_| make(n64, storage)).collect();
    let part = make(n64 / 1024, storage);
    let part_bytes = n64 / 1024 * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: part_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut bodies: Vec<(String, u32, usize, usize)> = Vec::new();
    // (body, element count, source index, destination index) over [input, chain 0..9, part]; index 0 is the input.
    for (k, (m, c)) in [(3, 1), (5, 2), (7, 3), (9, 4), (11, 5), (13, 6), (15, 7)]
        .iter()
        .enumerate()
    {
        bodies.push((
            format!("dst[i] = (src[i] * {m}u + {c}u) % 65521u;"),
            n,
            k,
            k + 1,
        ));
    }
    // 7 is s7; the block stage reads s7 and writes chain 7, the add reads that and s7 through the same source, so the
    // add stage binds `blk` as its source and recomputes the block value from `blk`'s own block head: that would
    // change the arithmetic, so the add reads s7 through a second source below.
    bodies.push((
        "dst[i] = src[i] ^ src[(i / 256u) * 256u];".to_owned(),
        n,
        7,
        8,
    ));
    let buffer_of = |index: usize| -> &wgpu::Buffer {
        match index {
            0 => &input,
            k if k <= 9 => &chain[k - 1],
            _ => &part,
        }
    };
    let uniform = |count: u32| {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytes(&[count, 0, 0, 0]));
        buffer
    };
    let mut passes = Vec::new();
    let mut add = |source: &wgpu::Buffer,
                   destination: &wgpu::Buffer,
                   body: &str,
                   count: u32,
                   extra: Option<&wgpu::Buffer>| {
        let source_text = if extra.is_some() {
            format!(
                "struct P {{ n: u32, a: u32, b: u32, c: u32 }}\n\
                 @group(0) @binding(0) var<uniform> p: P;\n\
                 @group(0) @binding(1) var<storage, read> src: array<u32>;\n\
                 @group(0) @binding(2) var<storage, read_write> dst: array<u32>;\n\
                 @group(0) @binding(3) var<storage, read> head: array<u32>;\n\
                 @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {{\n\
                 let i = g.x; if (i >= p.n) {{ return; }}\n{body}\n}}\n"
            )
        } else {
            wgsl(body)
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source_text.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let counts = uniform(count);
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: counts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: source.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: destination.as_entire_binding(),
            },
        ];
        if let Some(head) = extra {
            entries.push(wgpu::BindGroupEntry {
                binding: 3,
                resource: head.as_entire_binding(),
            });
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        passes.push((pipeline, group, count.div_ceil(256)));
        std::mem::forget(counts);
    };
    for (body, count, source, destination) in &bodies {
        add(
            buffer_of(*source),
            buffer_of(*destination),
            body,
            *count,
            None,
        );
    }
    // add: dst = blk[i] + s7[block head]; source blk (chain 7 = index 8), head s7 (index 7).
    add(
        buffer_of(8),
        buffer_of(9),
        "dst[i] = src[i] + head[(i / 256u) * 256u];",
        n,
        Some(buffer_of(7)),
    );
    // partials: dst[j] = sum[j * 1024] + sum[j * 1024 + 256] + ...; n / 1024 invocations.
    add(
        buffer_of(9),
        buffer_of(10),
        "dst[i] = src[i * 1024u] + src[i * 1024u + 256u] + src[i * 1024u + 512u] + src[i * 1024u + 768u];",
        n / 1024,
        None,
    );
    Raw {
        device,
        queue,
        passes,
        part,
        staging,
        input,
        part_bytes,
    }
}

fn bytes(words: &[u32]) -> &[u8] {
    // SAFETY: `u32` has no padding and any alignment of `u8` is fine.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), std::mem::size_of_val(words)) }
}

impl Raw {
    /// One frame: ten passes, one copy, one submit. Returns the submission index (to wait on outside the span).
    fn frame(&self) -> wgpu::SubmissionIndex {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for (pipeline, group, groups) in &self.passes {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(*groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.part, 0, &self.staging, 0, self.part_bytes);
        self.queue.submit([encoder.finish()])
    }

    fn wait(&self, index: wgpu::SubmissionIndex) {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            })
            .expect("the device polls");
    }

    fn read(&self) -> Vec<u32> {
        let slice = self.staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| {
            result.expect("the staging buffer maps")
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the device polls");
        let data = slice
            .get_mapped_range()
            .expect("the staging buffer is mapped");
        let words = data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        drop(data);
        self.staging.unmap();
        words
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let wanted = std::env::args().nth(1).unwrap_or_default();
    let options = DeviceOptions::default().with_adapter(AdapterChoice::NameContains(wanted));
    let device = Arc::new(HeadlessDevice::with_options(options)?);
    let info = device.adapter_info();
    println!("adapter: {} backend: {:?}", info.name, info.backend);
    for n in [1_u32 << 16, 1 << 20] {
        let input: Vec<u32> = (0..n).map(|i| (i * 7 + 3) % 65521).collect();
        let want = reference(&input);

        let (mut batch, handles) = build(&device, n)?;
        let raw = raw(&device, n);
        raw.queue.write_buffer(&raw.input, 0, bytes(&input));

        // One run of `frames` frames of GUST, and one of raw wgpu: the mean host span of a frame, in microseconds.
        let mut gust_part = Vec::new();
        let mut gust_run = |frames: usize, first: bool| -> Result<f64, Error> {
            let mut span = 0.0;
            for frame_number in 0..frames {
                let started = Instant::now();
                let mut frame = block_on(batch.begin())?;
                if first && frame_number == 0 {
                    frame.write(&handles.x, &input)?;
                }
                let ticket = frame.ticket(&handles.part)?;
                let submitted = frame.submit()?;
                span += started.elapsed().as_secs_f64();
                let values = block_on(ticket)?;
                submitted.wait()?;
                gust_part = values;
            }
            Ok(span / frames as f64 * 1e6)
        };
        let raw_run = |frames: usize| -> f64 {
            let mut span = 0.0;
            for _ in 0..frames {
                let started = Instant::now();
                let index = raw.frame();
                span += started.elapsed().as_secs_f64();
                raw.wait(index);
            }
            span / frames as f64 * 1e6
        };

        // Warm-up of both, then the runs with the order alternating (gust first, raw first, ..), so that neither
        // always runs on the device state the other left.
        gust_run(WARMUP, true)?;
        raw_run(WARMUP);
        let (mut gust_us, mut raw_us) = (Vec::new(), Vec::new());
        for run in 0..RUNS {
            if run % 2 == 0 {
                gust_us.push(gust_run(FRAMES, false)?);
                raw_us.push(raw_run(FRAMES));
            } else {
                raw_us.push(raw_run(FRAMES));
                gust_us.push(gust_run(FRAMES, false)?);
            }
        }
        assert_eq!(
            gust_part, want,
            "the batch's partials equal the host reference (n = {n})"
        );
        assert_eq!(
            raw.read(),
            want,
            "the raw partials equal the host reference (n = {n})"
        );

        let (gust, raw) = (median(gust_us.clone()), median(raw_us.clone()));
        println!(
            "n = {n}: gust {gust:.1} us, raw {raw:.1} us, ratio {:.2} (runs gust {gust_us:.1?}, raw {raw_us:.1?})",
            gust / raw
        );
    }
    Ok(())
}
