//! The probe of the frame timing hook (m14.pfft, checks C5 and C6): tests that print numbers and assert only what the
//! contract says must hold. They are not part of full verification (they are not `#[ignore]`d, so the example's GPU
//! table does not list them) and do nothing unless `GUST_PFFT_PROBE=1` is set, which is how to run them:
//!
//! `GUST_PFFT_PROBE=1 cargo test -p batch-basics --test frame_time_probe -- --nocapture --test-threads=1`
//!
//! on the adapter the environment selects. Their numbers are not a promise.
//!
//! C5: the same four-stage chain through the frame path (level `Frame`) and through `dispatch_batch_timed`, in one
//! process on one device, interleaved, 50 iterations each, three repetitions; the ratio of the medians must be within
//! 0.85 and 1.15 in every repetition. The batch's stages are `gpu_task!` kernels and the bench path's are one hand-written
//! kernel with the same arithmetic, so the two paths run the same work written twice, not the same shader.
//!
//! C6: wall clock per frame (200 frames, interleaved) for `Off`, `Frame` and `Passes`, and the spread of the span of 200
//! identical frames (p10, median, p90). The creation order of the variants rotates with the repetition (the first
//! variant made on the Radeon 890M can be slower).

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use batch_basics::app::block_on;
use gust::batch::{Batch, D1, Error, FrameTime, FrameTiming, Stage};
use gust::{
    Access, GpuPod, KernelDescriptor, ParameterDescriptor, ParameterKind, ResourceBinding,
    ResourceLayout,
};
use gust::{gpu_batch, gpu_task};
use gust_wgpu::{BufferBinding, BufferDispatch, GpuBufferAccess, HeadlessDevice};

const STAGES: usize = 4;

/// `chain(out: RW)`: the work of one stage of the batch, in place.
const CHAIN: KernelDescriptor = KernelDescriptor::new(
    "frame_time_probe",
    "chain",
    "frame_time_probe_chain",
    [64, 1, 1],
    &[ParameterDescriptor {
        name: "out",
        rust_type: "uint",
        kind: ParameterKind::Storage,
        access: Access::ReadWrite,
        resource_layout: Some(ResourceLayout::storage_v1(<u32 as GpuPod>::LAYOUT)),
        binding: Some(ResourceBinding {
            group: 0,
            binding: 0,
        }),
    }],
    r#"
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> out;
[shader("compute")] [numthreads(64, 1, 1)]
void frame_time_probe_chain(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; out.GetDimensions(n, stride);
    if (id.x < n) {
        uint v = out[id.x] % 65521u;
        for (uint k = 0u; k < 2000u; k++) { v = (v * 3u + 1u) % 65521u; }
        out[id.x] = v;
    }
}
"#,
);

struct Handles {
    x: Stage<u32, D1>,
}

fn build(
    device: &Arc<HeadlessDevice>,
    n: u32,
    level: FrameTiming,
) -> Result<(Batch<HeadlessDevice>, Handles), Error> {
    gpu_batch!("probe", on(device); in_flight(2); |b| {
        b.frame_timing(level);
        let x = b.input::<u32>(n);
        let y = gpu_task!(n, |i| -> u32 {
            let mut v = x[i] % 65521u32;
            for _k in 0u32..2000u32 {
                v = (v * 3u32 + 1u32) % 65521u32;
            }
            v
        });
        let _s2 = gpu_task!(&mut y, |i| {
            let mut v = y[i] % 65521u32;
            for _k in 0u32..2000u32 {
                v = (v * 3u32 + 1u32) % 65521u32;
            }
            y[i] = v;
        });
        let _s3 = gpu_task!(&mut y, |i| {
            let mut v = y[i] % 65521u32;
            for _k in 0u32..2000u32 {
                v = (v * 3u32 + 1u32) % 65521u32;
            }
            y[i] = v;
        });
        let _s4 = gpu_task!(&mut y, |i| {
            let mut v = y[i] % 65521u32;
            for _k in 0u32..2000u32 {
                v = (v * 3u32 + 1u32) % 65521u32;
            }
            y[i] = v;
        });
        Handles { x }
    })
}

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
}

fn private_device() -> Arc<HeadlessDevice> {
    let device = HeadlessDevice::new().unwrap_or_else(|error| {
        panic!("a GPU adapter is required; this test must not silently skip: {error}")
    });
    println!("probe adapter: {}", device.adapter_info().name);
    Arc::new(device)
}

fn data(n: u32) -> Vec<u32> {
    (0..n).map(|i| (i * 7 + 1) % 100_000).collect()
}

fn percentile(sorted: &[Duration], per_cent: usize) -> Duration {
    sorted[(sorted.len() - 1) * per_cent / 100]
}

fn median(values: &[Duration]) -> Duration {
    let mut sorted = values.to_vec();
    sorted.sort();
    percentile(&sorted, 50)
}

/// One frame of `batch` that asks for its time when `ask`; the first frame writes `x`.
fn run_frame(
    batch: &mut Batch<HeadlessDevice>,
    h: &Handles,
    x: Option<&[u32]>,
    ask: bool,
) -> (Option<FrameTime>, gust::batch::Submitted<HeadlessDevice>) {
    let mut run = block_on(batch.begin()).unwrap();
    if let Some(x) = x {
        run.write(&h.x, x).unwrap();
    }
    let time = ask.then(|| run.time());
    (time, run.submit().unwrap())
}

#[test]
fn probe_c5_frame_span_against_the_bench_quantity() {
    if !probe_requested() {
        return;
    }
    let _serial = serial();
    let device = private_device();
    // A size whose frame span is at least 1.5 ms (four stages of the counted loop).
    let mut n = 1_u32 << 14;
    loop {
        let (mut batch, h) = build(&device, n, FrameTiming::Frame).unwrap();
        let x = data(n);
        let (_t, _s) = run_frame(&mut batch, &h, Some(&x), true);
        let (time, _s) = run_frame(&mut batch, &h, None, true);
        let span = block_on(time.unwrap()).unwrap().unwrap().span;
        if span >= Duration::from_micros(1500) || n >= 1 << 22 {
            println!("C5 size: {n} elements, one frame {span:?}");
            break;
        }
        n *= 2;
    }
    let mut ratios = Vec::new();
    for repetition in 0..3 {
        // The creation order alternates: the batch first, or the bench's buffer first.
        let make_buffer = |device: &HeadlessDevice| {
            device.create_typed_buffer("probe", &data(n), GpuBufferAccess::ReadWrite)
        };
        let ((mut batch, h), buffer) = if repetition % 2 == 0 {
            let batch = build(&device, n, FrameTiming::Frame).unwrap();
            (batch, make_buffer(&device))
        } else {
            let buffer = make_buffer(&device);
            (build(&device, n, FrameTiming::Frame).unwrap(), buffer)
        };
        let bindings = [BufferBinding::read_write(&buffer)];
        let chain: [BufferDispatch<'_>; STAGES] =
            std::array::from_fn(|_| BufferDispatch::new(&CHAIN, n, &bindings));
        let x = data(n);
        // Warm both paths.
        let (time, _s) = run_frame(&mut batch, &h, Some(&x), true);
        block_on(time.unwrap()).unwrap().unwrap();
        device.dispatch_batch_timed(&chain).unwrap();
        let (mut frame_spans, mut bench_spans) = (Vec::new(), Vec::new());
        // 50 iterations of each path, interleaved.
        for k in 0..100 {
            if k % 2 == 0 {
                let (time, _s) = run_frame(&mut batch, &h, None, true);
                frame_spans.push(block_on(time.unwrap()).unwrap().unwrap().span);
            } else {
                let timing = device.dispatch_batch_timed(&chain).unwrap();
                bench_spans.push(timing.gpu_elapsed.expect("this adapter has timestamps"));
            }
        }
        let (frame, bench) = (median(&frame_spans), median(&bench_spans));
        let ratio = frame.as_secs_f64() / bench.as_secs_f64();
        println!(
            "C5 repetition {repetition} (n = {n}, {} + {} iterations): frame {frame:?}, bench {bench:?}, ratio {ratio:.3}",
            frame_spans.len(),
            bench_spans.len()
        );
        assert!(frame >= Duration::from_millis(1), "the span is below 1 ms");
        ratios.push(ratio);
    }
    println!("C5 ratios: {ratios:.3?}");
    for ratio in ratios {
        assert!(
            (0.85..=1.15).contains(&ratio),
            "ratio {ratio:.3} is outside 0.85 to 1.15"
        );
    }
}

#[test]
fn probe_c6_cost_and_spread() {
    if !probe_requested() {
        return;
    }
    let _serial = serial();
    let device = private_device();
    let n = 1_u32 << 18;
    let x = data(n);
    let levels = [FrameTiming::Off, FrameTiming::Frame, FrameTiming::Passes];
    for repetition in 0..3_usize {
        // Rotate the creation order of the three variants.
        let mut made: Vec<(FrameTiming, Batch<HeadlessDevice>, Handles)> = Vec::new();
        for k in 0..3 {
            let level = levels[(repetition + k) % 3];
            let (batch, h) = build(&device, n, level).unwrap();
            made.push((level, batch, h));
        }
        made.sort_by_key(|(level, _, _)| levels.iter().position(|l| l == level).unwrap());
        for (_, batch, h) in &mut made {
            let (_t, s) = run_frame(batch, h, Some(&x), false);
            s.wait().unwrap();
        }
        let mut walls: Vec<Vec<Duration>> = vec![Vec::new(); 3];
        let mut spans = Vec::new();
        for _ in 0..200 {
            for (index, (level, batch, h)) in made.iter_mut().enumerate() {
                let started = Instant::now();
                let (time, submitted) = run_frame(batch, h, None, *level != FrameTiming::Off);
                submitted.wait().unwrap();
                walls[index].push(started.elapsed());
                if *level == FrameTiming::Frame {
                    spans.push(block_on(time.unwrap()).unwrap().unwrap().span);
                }
            }
        }
        let medians: Vec<Duration> = walls.iter().map(|w| median(w)).collect();
        let means: Vec<f64> = walls
            .iter()
            .map(|w| w.iter().map(Duration::as_secs_f64).sum::<f64>() / w.len() as f64)
            .collect();
        spans.sort();
        println!(
            "C6 repetition {repetition}: wall median Off {:?} Frame {:?} Passes {:?}; ratios Frame/Off {:.3} Passes/Off {:.3} (medians), {:.3} {:.3} (means)",
            medians[0],
            medians[1],
            medians[2],
            medians[1].as_secs_f64() / medians[0].as_secs_f64(),
            medians[2].as_secs_f64() / medians[0].as_secs_f64(),
            means[1] / means[0],
            means[2] / means[0],
        );
        println!(
            "C6 repetition {repetition}: span of {} identical frames: p10 {:?}, median {:?}, p90 {:?}",
            spans.len(),
            percentile(&spans, 10),
            percentile(&spans, 50),
            percentile(&spans, 90)
        );
    }
}

/// The probes run only on request: `GUST_PFFT_PROBE` set to anything.
fn probe_requested() -> bool {
    let asked = std::env::var_os("GUST_PFFT_PROBE").is_some();
    if !asked {
        println!("probe skipped: set GUST_PFFT_PROBE=1 to run it");
    }
    asked
}
