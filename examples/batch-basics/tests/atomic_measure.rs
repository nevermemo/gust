//! Registered check s2sa.m (m14.s2sa): an atomic stage of a batch against the same histogram written as a
//! `#[gpu_module]` kernel. A probe, skipped unless `GUST_S2SA_PROBE=1`: it prints a table and asserts only that the two
//! counts equal the host's before it times anything (a measurement of a wrong answer is worth nothing). Run it on one
//! adapter at a time, outside a quiet window:
//!
//! `GUST_S2SA_PROBE=1 cargo test -p batch-basics --release --test atomic_measure -- --nocapture`
//!
//! The cells are the key counts `K` of 4 (every row fights for four cells), 4096 and 2^20 (almost no contention), over
//! 2^20 and 2^22 rows. The stage side is an atomic stage of a batch frame (`begin`, `submit`, wait; the keys are written
//! once before timing, as the frames of a batch keep their buffers, and the counts are not cleared between frames: a
//! frame adds to what the frames before left, on both sides, so the timed work is the atomic traffic alone); the kernel
//! side is the typed `dispatch` of a module kernel with the same body, which returns when the work is done. Each round
//! builds the two variants in the other order (the first variant created on the Radeon 890M can be up to 3x slower than
//! the second), and the median is the middle frame time of [`FRAMES`] frames after [`WARMUP`] warm-up frames, over
//! [`ROUNDS`] rounds.
//!
//! What each timed span includes. Stage side: `batch.begin()`, `submit()` of the frame the batch built once, and
//! `wait()`: the batch's own per-frame host work (not itemised here) and the kernel's run. Kernel side: one typed
//! `dispatch` call of the module kernel: that call's own per-call host work (not itemised here) and the kernel's run.
//! The probe does not time a pre-built submission on the kernel side, so the two spans are not the same set of host
//! steps; a ratio therefore says nothing about the kernels' own cost until the difference is accounted for.
//!
//! Pre-registered rule, written before the first run: the ratio stage over kernel inside 0.90 to 1.10 on every cell
//! means there is nothing to say; outside it, the cause is explained before any sentence about cost goes into a
//! document, and no speed claim is made either way.

#![allow(clippy::let_unit_value)]

use std::sync::Arc;
use std::time::Instant;

use batch_basics::app::block_on;
use gust::batch::{Batch, D1, Error, Stage};
use gust::{Independent, gpu_batch, gpu_module, gpu_task};
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

const WARMUP: usize = 5;
const FRAMES: usize = 21;
const ROUNDS: usize = 3;
const SIZES: [u32; 2] = [1 << 20, 1 << 22];
const KEYS: [u32; 3] = [4, 4096, 1 << 20];

#[gpu_module]
mod kernels {
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn count(
        id: SV_DispatchThreadID,
        key: StructuredBuffer<uint>,
        mut counts: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < key.len() {
            atomic_add(&mut counts[key[i]], 1u32);
        }
    }
}

#[derive(Clone, Copy)]
struct Handles {
    key: Stage<u32, D1>,
    counts: Stage<u32, D1>,
}

fn build(
    device: &Arc<HeadlessDevice>,
    n: u32,
    k: u32,
) -> Result<(Batch<HeadlessDevice>, Handles), Error> {
    gpu_batch!("atomic_measure", on(device); in_flight(1); |b| {
        let key = b.input::<u32>(n);
        let counts = b.input::<u32>(k);
        let _count = gpu_task!(n, atomic(&mut counts), |i| {
            atomic_add(&mut counts[key[i]], 1u32);
        });
        Handles { key, counts }
    })
}

/// Keys spread over `k` cells (`k` is a power of two).
fn keys_of(n: u32, k: u32) -> Vec<u32> {
    (0..n)
        .map(|i| i.wrapping_mul(2_654_435_761) >> 7 & (k - 1))
        .collect()
}

fn histogram(keys: &[u32], k: u32) -> Vec<u32> {
    let mut counts = vec![0u32; k as usize];
    for &key in keys {
        counts[key as usize] += 1;
    }
    counts
}

/// The median of the two middle values.
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Writes the keys once, reads the counts back after one frame and compares them with the host's; returns the median
/// frame time in microseconds of the frames after it.
fn stage_time(batch: &mut Batch<HeadlessDevice>, h: &Handles, keys: &[u32], want: &[u32]) -> f64 {
    let counts = block_on(async {
        let mut frame = batch.begin().await?;
        frame.write(&h.key, keys)?;
        let ticket = frame.ticket(&h.counts)?;
        let _submitted = frame.submit()?;
        ticket.await
    })
    .unwrap();
    assert_eq!(counts, want, "the stage's counts differ from the host's");
    let mut times = Vec::new();
    for round in 0..WARMUP + FRAMES {
        let started = Instant::now();
        let frame = block_on(batch.begin()).unwrap();
        let submitted = frame.submit().unwrap();
        submitted.wait().unwrap();
        if round >= WARMUP {
            times.push(started.elapsed().as_secs_f64() * 1e6);
        }
    }
    median(times)
}

/// The same for the module kernel.
fn kernel_time(device: &HeadlessDevice, keys: &[u32], k: u32, want: &[u32]) -> f64 {
    let n = keys.len() as u32;
    let key = device.create_typed_buffer("atomic key", keys, GpuBufferAccess::ReadOnly);
    let counts = device.create_typed_buffer(
        "atomic counts",
        &vec![0u32; k as usize],
        GpuBufferAccess::ReadWrite,
    );
    let run = || kernels::count::dispatch(device, n, &key, Independent(&counts)).unwrap();
    run();
    assert_eq!(
        device.read_typed_buffer(&counts).unwrap(),
        want,
        "the kernel's counts differ from the host's"
    );
    let mut times = Vec::new();
    for round in 0..WARMUP + FRAMES {
        let started = Instant::now();
        run();
        if round >= WARMUP {
            times.push(started.elapsed().as_secs_f64() * 1e6);
        }
    }
    median(times)
}

#[test]
fn gpu_s2sa_m_atomic_stage_against_module_kernel() {
    // A probe, not a gate test: it runs only when `GUST_S2SA_PROBE=1` and is no `#[ignore]` for the pin scan to list.
    if std::env::var("GUST_S2SA_PROBE").as_deref() != Ok("1") {
        return;
    }
    let device = Arc::new(HeadlessDevice::new().expect("a GPU adapter is required"));
    println!("adapter: {}", device.adapter_info().name);
    println!("keys,rows,stage_us,kernel_us,ratio_stage_over_kernel");
    for k in KEYS {
        for n in SIZES {
            let keys = keys_of(n, k);
            let want = histogram(&keys, k);
            let mut stage = Vec::new();
            let mut kernel = Vec::new();
            for round in 0..ROUNDS {
                let mut run_stage = || {
                    let (mut batch, h) = build(&device, n, k).unwrap();
                    stage.push(stage_time(&mut batch, &h, &keys, &want));
                };
                let mut run_kernel = || kernel.push(kernel_time(&device, &keys, k, &want));
                if round % 2 == 0 {
                    run_stage();
                    run_kernel();
                } else {
                    run_kernel();
                    run_stage();
                }
            }
            let (stage, kernel) = (median(stage), median(kernel));
            println!("{k},{n},{stage:.1},{kernel:.1},{:.3}", stage / kernel);
        }
    }
}
