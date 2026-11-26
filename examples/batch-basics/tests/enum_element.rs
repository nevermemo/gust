//! An enum stored as the element of a batch's buffers (m14.s2e2c, D275): `b.input::<Phase>(..)`, a stage that reads
//! `phase[i]` (and `grid[(x, y)]`) through a `match` and a stage whose result type is `-> Phase`.
//!
//! The host writes and reads `Phase` values (`write`, `ticket`), the device holds one `u32` word each, a load in a
//! stage body decodes the word through the enum's one decoder and the result is stored as its word. Every value is
//! small, so a result equals the host reference bit for bit. The `twin_` tests run on the CPU twin; the `gpu_` ones
//! are `#[ignore]`d like the example's other GPU tests and run in full verification on the adapter the environment
//! selects.

use std::sync::Arc;

use batch_basics::app::block_on;
use gust::batch::{Batch, BatchDevice, D1, D2, Error, Stage};
use gust::cpu::CpuDevice;
use gust::{gpu_batch, gpu_task};
use gust_wgpu::HeadlessDevice;

mod phases {
    /// Words with gaps and a zero, so that a copy whose words were counted instead of copied would read the wrong arm.
    #[derive(Clone, Copy, Debug, PartialEq, gust::GpuEnum)]
    #[repr(u32)]
    pub enum Phase {
        Idle = 0,
        Run = 3,
        Done = 7,
    }
}

use phases::Phase;

const N: u32 = 48;
const W: u32 = 7;
const H: u32 = 5;

const PHASES: [Phase; 3] = [Phase::Idle, Phase::Run, Phase::Done];

#[derive(Clone, Copy)]
struct Handles {
    phase: Stage<Phase, D1>,
    next: Stage<Phase, D1>,
    weight: Stage<u32, D1>,
    grid: Stage<Phase, D2>,
    cell: Stage<u32, D2>,
}

macro_rules! define_build {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles), Error> {
            gpu_batch!("enum_element", on(device); in_flight(1); |b| {
                let phase = b.input::<Phase>(N);
                let next = gpu_task!(N, |i| -> Phase {
                    match phase[i] {
                        Phase::Idle => Phase::Run,
                        Phase::Run => Phase::Done,
                        Phase::Done => Phase::Idle,
                    }
                });
                let weight = gpu_task!(N, |i| -> u32 {
                    let lo: u32 = match phase[i] {
                        Phase::Idle => 1u32,
                        Phase::Run => 10u32,
                        Phase::Done => 100u32,
                    };
                    let hi: u32 = match next[i] {
                        Phase::Idle => 1000u32,
                        Phase::Run => 2000u32,
                        Phase::Done => 3000u32,
                    };
                    lo + hi
                });
                let grid = b.input::<Phase>((W, H));
                let cell = gpu_task!((W, H), |x, y| -> u32 {
                    let picked: u32 = match grid[(x, y)] {
                        Phase::Idle => 1u32,
                        Phase::Run => 10u32 + x,
                        Phase::Done => 100u32 + y * 10u32,
                    };
                    picked
                });
                Handles { phase, next, weight, grid, cell }
            })
        }
    };
}

define_build!(build_cpu, CpuDevice);
define_build!(build_gpu, HeadlessDevice);

fn private_device() -> Arc<HeadlessDevice> {
    Arc::new(HeadlessDevice::new().unwrap_or_else(|error| {
        panic!("a GPU adapter is required; this test must not silently skip: {error}")
    }))
}

fn phases_of(len: u32, seed: u32) -> Vec<Phase> {
    (0..len)
        .map(|i| PHASES[((i * 5 + seed) % 3) as usize])
        .collect()
}

fn successor(phase: Phase) -> Phase {
    match phase {
        Phase::Idle => Phase::Run,
        Phase::Run => Phase::Done,
        Phase::Done => Phase::Idle,
    }
}

fn weight_of(phase: Phase) -> u32 {
    let lo = match phase {
        Phase::Idle => 1,
        Phase::Run => 10,
        Phase::Done => 100,
    };
    let hi = match successor(phase) {
        Phase::Idle => 1000,
        Phase::Run => 2000,
        Phase::Done => 3000,
    };
    lo + hi
}

fn cell_of(phase: Phase, x: u32, y: u32) -> u32 {
    match phase {
        Phase::Idle => 1,
        Phase::Run => 10 + x,
        Phase::Done => 100 + y * 10,
    }
}

type Results = (Vec<Phase>, Vec<u32>, Vec<u32>);

/// One frame: both inputs written, the three results read.
fn frame<D: BatchDevice>(
    batch: &mut Batch<D>,
    h: &Handles,
    line: &[Phase],
    grid: &[Phase],
) -> Results
where
    D::Buffer<u32>: 'static,
{
    block_on(async {
        let mut frame = batch.begin().await?;
        frame.write(&h.phase, line)?;
        frame.write(&h.grid, grid)?;
        let next = frame.ticket(&h.next)?;
        let weight = frame.ticket(&h.weight)?;
        let cell = frame.ticket(&h.cell)?;
        let _submitted = frame.submit()?;
        Ok::<_, Error>((next.await?, weight.await?, cell.await?))
    })
    .unwrap()
}

/// Every phase is loaded and stored, in a 1-D and a 2-D view, and each result equals the host reference.
fn results_equal_the_host<D: BatchDevice>(mut batch: Batch<D>, h: &Handles)
where
    D::Buffer<u32>: 'static,
{
    for seed in 0..3 {
        let line = phases_of(N, seed);
        let grid = phases_of(W * H, seed + 1);
        let (next, weight, cell) = frame(&mut batch, h, &line, &grid);
        assert_eq!(
            next,
            line.iter().map(|&p| successor(p)).collect::<Vec<_>>(),
            "the stored enum result, seed {seed}"
        );
        assert_eq!(
            weight,
            line.iter().map(|&p| weight_of(p)).collect::<Vec<_>>(),
            "1-D loads of an input and of an enum result, seed {seed}"
        );
        let expected: Vec<u32> = grid
            .iter()
            .enumerate()
            .map(|(index, &p)| cell_of(p, index as u32 % W, index as u32 / W))
            .collect();
        assert_eq!(cell, expected, "2-D loads, seed {seed}");
    }
}

/// A stage that writes a word which is no variant of `Phase` (9), viewed as `Phase` by the macro-support `retype`: the
/// way to put such a word in a stored enum without a public raw write (D275).
macro_rules! define_fallback {
    ($build:ident, $device:ty) => {
        fn $build(
            device: &Arc<$device>,
        ) -> Result<(Batch<$device>, (Stage<Phase, D1>, Stage<u32, D1>)), Error> {
            gpu_batch!("fallback", on(device); in_flight(1); |b| {
                let raw = gpu_task!(N, |i| -> u32 { 9u32 + i * 0u32 });
                let bad: Stage<Phase, D1> = raw.retype::<Phase>();
                let seen = gpu_task!(N, |i| -> u32 {
                    let picked: u32 = match bad[i] {
                        Phase::Idle => 1u32,
                        Phase::Run => 2u32,
                        Phase::Done => 3u32,
                    };
                    picked
                });
                (bad, seen)
            })
        }
    };
}

define_fallback!(fallback_cpu, CpuDevice);
define_fallback!(fallback_gpu, HeadlessDevice);

/// In a body a word that is no variant is the last variant (`Done`, 3), and the host read of the same buffer is
/// `InvalidEnum` naming the batch, the stage, the first index and the word.
fn a_bad_word_reads_as_the_last_variant_in_a_body<D: BatchDevice>(
    mut batch: Batch<D>,
    handles: (Stage<Phase, D1>, Stage<u32, D1>),
) where
    D::Buffer<u32>: 'static,
{
    let (bad, seen) = handles;
    let (seen, bad_read) = block_on(async {
        let mut frame = batch.begin().await?;
        let bad_ticket = frame.ticket(&bad)?;
        let seen_ticket = frame.ticket(&seen)?;
        let _submitted = frame.submit()?;
        Ok::<_, Error>((seen_ticket.await?, bad_ticket.await))
    })
    .unwrap();
    assert_eq!(seen, vec![3; N as usize], "every bad word reads as Done");
    match bad_read {
        Err(Error::InvalidEnum {
            batch,
            stage,
            index,
            word,
        }) => assert_eq!(
            (batch.as_str(), stage.as_str(), index, word),
            ("fallback", "raw", 0, 9)
        ),
        other => panic!("expected InvalidEnum, got {other:?}"),
    }
}

#[test]
fn twin_a_word_that_is_no_variant_reads_as_the_last_variant_in_a_body() {
    let device = Arc::new(CpuDevice::new());
    let (batch, handles) = fallback_cpu(&device).unwrap();
    a_bad_word_reads_as_the_last_variant_in_a_body(batch, handles);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_word_that_is_no_variant_reads_as_the_last_variant_in_a_body() {
    let device = private_device();
    println!("adapter: {}", device.adapter_info().name);
    let (batch, handles) = fallback_gpu(&device).unwrap();
    a_bad_word_reads_as_the_last_variant_in_a_body(batch, handles);
}

#[test]
fn twin_an_enum_input_mapped_through_a_match_to_an_enum_result_equals_the_host() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu(&device).unwrap();
    results_equal_the_host(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_an_enum_input_mapped_through_a_match_to_an_enum_result_equals_the_host() {
    let device = private_device();
    println!("adapter: {}", device.adapter_info().name);
    let (batch, h) = build_gpu(&device).unwrap();
    results_equal_the_host(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_an_enum_element_equals_the_cpu_twin() {
    let gpu = private_device();
    let cpu = Arc::new(CpuDevice::new());
    println!("adapter: {}", gpu.adapter_info().name);
    let (mut on_gpu, gpu_h) = build_gpu(&gpu).unwrap();
    let (mut on_cpu, cpu_h) = build_cpu(&cpu).unwrap();
    for seed in 0..3 {
        let line = phases_of(N, seed);
        let grid = phases_of(W * H, seed);
        let from_gpu = frame(&mut on_gpu, &gpu_h, &line, &grid);
        let from_cpu = frame(&mut on_cpu, &cpu_h, &line, &grid);
        assert_eq!(from_gpu, from_cpu, "seed {seed}");
    }
}
