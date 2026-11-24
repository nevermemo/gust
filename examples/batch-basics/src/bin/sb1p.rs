//! SB1p (plan m14.s2, slice s2c): frames in flight without a thread. Frames per second of `in_flight(2)` against
//! `in_flight(1)`, with a ticket on every frame, at n in {2^10, 2^16, 2^20}.
//!
//! The frame is six chained per-element stages over `u32` data (the first reads a parameter that every frame sets) and
//! one ticket on the last stage. The loop is the same for both depths: `begin`, `set`, `ticket`, `submit`, then await
//! the ticket of the frame before, so that with `in_flight(2)` the GPU can run frame N while the host records N+1, and
//! with `in_flight(1)` `begin` waits for frame N first. The last frame of every run is checked against the host.
//!
//! Protocol: 20 warm-up frames, then the median over 5 runs of frames per second; the two depths alternate their order
//! from run to run. Run it once per adapter, with a part of the adapter name:
//! `GUST_PROFILE=dev cargo run --release -p batch-basics --bin sb1p -- <part of the adapter name>` (a release build
//! infers the `native` profile, which refuses the run-time Slang bridge the batch needs). Under the GPU lock of both
//! adapters, one adapter after the other.

use std::sync::Arc;
use std::time::Instant;

use batch_basics::app::block_on;
use gust::batch::{Batch, D1, Error, Param, Stage, Ticket};
use gust::{gpu_batch, gpu_task};
use gust_wgpu::{AdapterChoice, DeviceOptions, HeadlessDevice};

const WARMUP: usize = 20;
const RUNS: usize = 5;

struct Handles {
    gain: Param<u32>,
    x: Stage<u32, D1>,
    last: Stage<u32, D1>,
}

macro_rules! define_build {
    ($build:ident, $depth:tt) => {
        fn $build(
            device: &Arc<HeadlessDevice>,
            n: u32,
        ) -> Result<(Batch<HeadlessDevice>, Handles), Error> {
            gpu_batch!("sb1p", on(device); in_flight($depth); |b| {
                let gain = b.param::<u32>(1);
                let x = b.input::<u32>(n);
                let s1 = gpu_task!(n, |i| -> u32 { (x[i] * 3u32 + gain.get()) % 65521u32 });
                let s2 = gpu_task!(n, |i| -> u32 { (s1[i] * 5u32 + 2u32) % 65521u32 });
                let s3 = gpu_task!(n, |i| -> u32 { (s2[i] * 7u32 + 3u32) % 65521u32 });
                let s4 = gpu_task!(n, |i| -> u32 { (s3[i] * 9u32 + 4u32) % 65521u32 });
                let s5 = gpu_task!(n, |i| -> u32 { (s4[i] * 11u32 + 5u32) % 65521u32 });
                let last = gpu_task!(n, |i| -> u32 { (s5[i] * 13u32 + 6u32) % 65521u32 });
                Handles { gain, x, last }
            })
        }
    };
}

define_build!(build_one, 1);
define_build!(build_two, 2);

fn reference(input: &[u32], gain: u32) -> Vec<u32> {
    input
        .iter()
        .map(|v| {
            let mut v = (v * 3 + gain) % 65521;
            for (m, c) in [(5, 2), (7, 3), (9, 4), (11, 5), (13, 6)] {
                v = (v * m + c) % 65521;
            }
            v
        })
        .collect()
}

/// `frames` frames, each awaiting the ticket of the frame before it; returns frames per second. The values of the last
/// frame are checked against `want`.
fn run(
    batch: &mut Batch<HeadlessDevice>,
    h: &Handles,
    frames: usize,
    gain_base: u32,
    want: &dyn Fn(u32) -> Vec<u32>,
) -> Result<f64, Error> {
    let started = Instant::now();
    let mut before: Option<Ticket<u32>> = None;
    let mut last_gain = gain_base;
    for k in 0..frames {
        last_gain = gain_base + u32::try_from(k % 1000).expect("small");
        let mut frame = block_on(batch.begin())?;
        frame.set(&h.gain, last_gain)?;
        let ticket = frame.ticket(&h.last)?;
        let _submitted = frame.submit()?;
        if let Some(before) = before.replace(ticket) {
            block_on(before)?;
        }
    }
    let last = block_on(before.take().expect("at least one frame"))?;
    let seconds = started.elapsed().as_secs_f64();
    assert_eq!(
        last,
        want(last_gain),
        "the last frame differs from the host"
    );
    Ok(f64::from(u32::try_from(frames).expect("small")) / seconds)
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
    let mut best = 0.0_f64;
    for (n, frames) in [(1_u32 << 10, 2000_usize), (1 << 16, 500), (1 << 20, 200)] {
        let input: Vec<u32> = (0..n).map(|i| (i * 7 + 3) % 65521).collect();
        let want = |gain: u32| reference(&input, gain);
        let (mut one, h_one) = build_one(&device, n)?;
        let (mut two, h_two) = build_two(&device, n)?;
        // The input is written once, by a frame of its own.
        for (batch, h) in [(&mut one, &h_one), (&mut two, &h_two)] {
            let mut frame = block_on(batch.begin())?;
            frame.write(&h.x, &input)?;
            frame.submit()?.wait()?;
        }
        run(&mut one, &h_one, WARMUP, 1, &want)?;
        run(&mut two, &h_two, WARMUP, 1, &want)?;
        let (mut fps_one, mut fps_two) = (Vec::new(), Vec::new());
        for round in 0..RUNS {
            if round % 2 == 0 {
                fps_one.push(run(&mut one, &h_one, frames, 1, &want)?);
                fps_two.push(run(&mut two, &h_two, frames, 1, &want)?);
            } else {
                fps_two.push(run(&mut two, &h_two, frames, 1, &want)?);
                fps_one.push(run(&mut one, &h_one, frames, 1, &want)?);
            }
        }
        let (one_fps, two_fps) = (median(fps_one), median(fps_two));
        let ratio = two_fps / one_fps;
        best = best.max(ratio);
        println!(
            "n = 2^{}: in_flight(1) {one_fps:.0} frames/s ({:.3} ms/frame), in_flight(2) {two_fps:.0} frames/s ({:.3} ms/frame), ratio {ratio:.3}",
            n.trailing_zeros(),
            1000.0 / one_fps,
            1000.0 / two_fps,
        );
    }
    println!("best ratio on this adapter: {best:.3}");
    Ok(())
}
