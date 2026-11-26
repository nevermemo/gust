//! Measurement s2i.m (m14.s2i, D278): the indirect launch against the bounded one on one light counted stage, over
//! capacities of 65536, 1M, 4M and 16M rows and counts of 1 %, 10 %, 50 % and 100 % of the capacity, wgpu's indirect
//! validation on. A probe, skipped unless `GUST_S2I_PROBE=1`: it prints a table and asserts only that every result is the model's (a
//! measurement of a wrong answer is worth nothing). Run it on one adapter at a time, outside a quiet window:
//!
//! `GUST_S2I_PROBE=1 cargo test -p batch-basics --release --test counted_measure -- --nocapture`
//!
//! A frame is `begin`, a write of the one-element count, `submit`, and the wait for the submission to finish; its time
//! is the wall-clock of the three. The medians are of [`FRAMES`] frames after [`WARMUP`] warm-up frames, taken over
//! [`ROUNDS`] rounds; each round builds the two batches in the other order (the first batch created on the Radeon 890M
//! can be up to 3x slower than the second, so the creation order is rotated).

use std::sync::Arc;
use std::time::Instant;

use batch_basics::app::block_on;
use gust::batch::{Batch, CountedLaunch, D1, Error, Stage, set_counted_launch};
use gust::{gpu_batch, gpu_task};
use gust_wgpu::HeadlessDevice;

const WARMUP: usize = 5;
const FRAMES: usize = 21;
const ROUNDS: usize = 3;
const CAPACITIES: [u32; 4] = [65_536, 1 << 20, 4 << 20, 16 << 20];
const PERCENT: [u32; 4] = [1, 10, 50, 100];

#[derive(Clone, Copy)]
struct Light {
    n: Stage<u32, D1>,
    src: Stage<u32, D1>,
    out: Stage<u32, D1>,
}

fn build(
    device: &Arc<HeadlessDevice>,
    capacity: u32,
) -> Result<(Batch<HeadlessDevice>, Light), Error> {
    gpu_batch!("measure", on(device); in_flight(1); |b| {
        let n = b.input::<u32>(1u32);
        let src = b.input::<u32>(capacity);
        let out = gpu_task!(count(n); capacity, |i| -> u32 {
            let v = src[i];
            (v * 3u32 + 1u32) ^ (v >> 3u32)
        });
        Light { n, src, out }
    })
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// The median frame time of one batch at one count, in microseconds.
fn frame_micros(batch: &mut Batch<HeadlessDevice>, h: &Light, n: u32) -> f64 {
    let mut times = Vec::new();
    for round in 0..WARMUP + FRAMES {
        let started = Instant::now();
        let mut frame = block_on(batch.begin()).unwrap();
        frame.write(&h.n, &[n]).unwrap();
        let submitted = frame.submit().unwrap();
        submitted.wait().unwrap();
        if round >= WARMUP {
            times.push(started.elapsed().as_secs_f64() * 1e6);
        }
    }
    median(times)
}

/// One checked frame: the first `n` rows of the result are the model's, the others are zero.
fn check(
    batch: &mut Batch<HeadlessDevice>,
    h: &Light,
    n: u32,
    capacity: u32,
    src: &[u32],
    write: bool,
) {
    let out = block_on(async {
        let mut frame = batch.begin().await?;
        frame.write(&h.n, &[n])?;
        if write {
            frame.write(&h.src, src)?;
        }
        let out = frame.ticket(&h.out)?;
        let _submitted = frame.submit()?;
        let result: Result<Vec<u32>, Error> = out.await;
        result
    })
    .unwrap();
    let rows = n.min(capacity) as usize;
    for i in (0..rows).step_by(4099).chain([rows.saturating_sub(1)]) {
        let v = src[i];
        assert_eq!(out[i], (v * 3 + 1) ^ (v >> 3), "row {i}");
    }
}

#[test]
fn gpu_s2i_m_indirect_against_bounded() {
    // A probe, not a gate test: it runs only when `GUST_S2I_PROBE=1` and is no `#[ignore]` for the pin scan to list.
    if std::env::var("GUST_S2I_PROBE").as_deref() != Ok("1") {
        return;
    }
    let device = Arc::new(HeadlessDevice::new().expect("a GPU adapter is required"));
    println!("adapter: {}", device.adapter_info().name);
    println!("capacity,percent,indirect_us,bounded_us,ratio_bounded_over_indirect");
    for capacity in CAPACITIES {
        let src: Vec<u32> = (0..capacity).map(|i| i % 1000).collect();
        let mut indirect = vec![vec![]; PERCENT.len()];
        let mut bounded = vec![vec![]; PERCENT.len()];
        for round in 0..ROUNDS {
            let order = if round % 2 == 0 {
                [CountedLaunch::Indirect, CountedLaunch::Bounded]
            } else {
                [CountedLaunch::Bounded, CountedLaunch::Indirect]
            };
            let mut batches = Vec::new();
            for launch in order {
                set_counted_launch(Some(launch));
                batches.push((launch, build(&device, capacity).unwrap()));
                set_counted_launch(None);
            }
            for (launch, (batch, h)) in &mut batches {
                for (slot, percent) in PERCENT.into_iter().enumerate() {
                    let n = (u64::from(capacity) * u64::from(percent) / 100) as u32;
                    check(batch, h, n, capacity, &src, slot == 0);
                    let micros = frame_micros(batch, h, n);
                    match launch {
                        CountedLaunch::Indirect => indirect[slot].push(micros),
                        _ => bounded[slot].push(micros),
                    }
                }
            }
        }
        for (slot, percent) in PERCENT.into_iter().enumerate() {
            let (i, b) = (
                median(indirect[slot].clone()),
                median(bounded[slot].clone()),
            );
            println!("{capacity},{percent},{i:.1},{b:.1},{:.3}", b / i);
        }
    }
}
