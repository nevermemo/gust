//! An enum element written in place and a 2-D enum result (m14.s2e2d, D275): `gpu_task!(&mut phase, ..)` over an
//! input of `Phase` loads through the enum's one decoder and stores the value as its word (`e as u32`), the input keeps
//! what the stage stored from frame to frame (D258), a resize makes zeroed buffers, and a `(w, h)` stage with
//! `-> Phase` stores one word per cell.
//!
//! A batch has no `state` or host-slice form yet, so the persistence of an enum across frames is shown through the
//! in-place stage of an input. Every value is small, so a result equals the host reference bit for bit. The `twin_`
//! tests run on the CPU twin, the `gpu_` ones are `#[ignore]`d like the example's other GPU tests and run in full
//! verification on the adapter the environment selects.

use std::sync::Arc;

use batch_basics::app::block_on;
use gust::batch::{Batch, BatchDevice, D1, D2, Error, Param, Stage, View};
use gust::cpu::CpuDevice;
use gust::{gpu_batch, gpu_task};
use gust_wgpu::HeadlessDevice;

mod phases {
    /// Words with gaps and a zero, as in `enum_element.rs`.
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
const SMALL: u32 = 20;
const W: u32 = 7;
const H: u32 = 5;

const PHASES: [Phase; 3] = [Phase::Idle, Phase::Run, Phase::Done];

#[derive(Clone, Copy)]
struct Handles {
    len: Param<u32>,
    phase: Stage<Phase, D1>,
    grid: Stage<Phase, D2>,
    lifted: Stage<Phase, D2>,
}

macro_rules! define_build {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles), Error> {
            gpu_batch!("enum_store", on(device); in_flight(1); |b| {
                let len = b.param::<u32>(N);
                let phase = b.input::<Phase>(&len);
                let _step = gpu_task!(&mut phase, |i| {
                    phase[i] = match phase[i] {
                        Phase::Idle => Phase::Run,
                        Phase::Run => Phase::Done,
                        Phase::Done => Phase::Idle,
                    };
                });
                let grid = b.input::<Phase>((W, H));
                let lifted = gpu_task!((W, H), |x, y| -> Phase {
                    match grid[(x, y)] {
                        Phase::Idle => Phase::Run,
                        Phase::Run => Phase::Done,
                        Phase::Done => Phase::Idle,
                    }
                });
                Handles { len, phase, grid, lifted }
            })
        }
    };
}

define_build!(build_cpu, CpuDevice);
define_build!(build_gpu, HeadlessDevice);

/// A stage that stores a word which is no variant (9) as `raw`, viewed as `Phase` by the macro-support `retype`, and an
/// in-place stage over it: the in-place load decodes the word (the last variant, `Done`) and the store encodes the
/// value it chose, so the buffer holds a valid word afterwards (D275: a derived store launders it).
macro_rules! define_laundered {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Stage<Phase, D1>), Error> {
            gpu_batch!("laundered", on(device); in_flight(1); |b| {
                let raw = gpu_task!(N, |i| -> u32 { 9u32 + i * 0u32 });
                let bad: Stage<Phase, D1> = raw.retype::<Phase>();
                let _launder = gpu_task!(&mut bad, |i| {
                    bad[i] = match bad[i] {
                        Phase::Idle => Phase::Done,
                        Phase::Run => Phase::Idle,
                        Phase::Done => Phase::Run,
                    };
                });
                bad
            })
        }
    };
}

define_laundered!(laundered_cpu, CpuDevice);
define_laundered!(laundered_gpu, HeadlessDevice);

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

/// One frame: an optional resize, an optional write of the line and of the grid, and the reads of the line (after the
/// in-place stage) and of the 2-D result.
fn frame<D: BatchDevice>(
    batch: &mut Batch<D>,
    h: &Handles,
    size: Option<u32>,
    line: Option<&[Phase]>,
    grid: Option<&[Phase]>,
) -> (Vec<Phase>, Vec<Phase>)
where
    D::Buffer<u32>: 'static,
{
    block_on(async {
        let mut frame = batch.begin().await?;
        if let Some(size) = size {
            frame.set(&h.len, size)?;
        }
        if let Some(line) = line {
            frame.write(&h.phase, line)?;
        }
        if let Some(grid) = grid {
            frame.write(&h.grid, grid)?;
        }
        let phase = frame.ticket(&h.phase)?;
        let lifted = frame.ticket(&h.lifted)?;
        let _submitted = frame.submit()?;
        Ok::<_, Error>((phase.await?, lifted.await?))
    })
    .unwrap()
}

/// The in-place stage stores its encoded values, the input keeps them to the next frame, a resize makes zeroed buffers
/// (all `Idle`, so a step makes them `Run`), and the 2-D result stores one word per cell.
fn the_target_keeps_its_stored_enums_across_frames_and_a_resize<D: BatchDevice>(
    mut batch: Batch<D>,
    h: &Handles,
) where
    D::Buffer<u32>: 'static,
{
    let line = phases_of(N, 1);
    let grid = phases_of(W * H, 2);
    let stepped: Vec<Phase> = line.iter().map(|&p| successor(p)).collect();
    let lifted: Vec<Phase> = grid.iter().map(|&p| successor(p)).collect();

    let (first, cells) = frame(&mut batch, h, None, Some(&line), Some(&grid));
    assert_eq!(first, stepped, "the in-place store of a written input");
    assert_eq!(cells, lifted, "the 2-D `-> Phase` result");
    // `view` of an enum handle is the view of its words.
    let words: View<D, u32> = batch.view(&h.phase).unwrap();
    assert_eq!(words.len(), N as usize);

    // Nothing written: the input holds what the last frame stored, and the stage steps it again.
    let (second, _) = frame(&mut batch, h, None, None, None);
    let twice: Vec<Phase> = stepped.iter().map(|&p| successor(p)).collect();
    assert_eq!(second, twice, "the stored enums persist to the next frame");

    // A resize makes zeroed buffers: every element is `Idle`, and one step makes it `Run`.
    let (small, _) = frame(&mut batch, h, Some(SMALL), None, None);
    assert_eq!(small, vec![Phase::Run; SMALL as usize], "after a resize");
    assert_eq!(batch.view(&h.phase).unwrap().len(), SMALL as usize);
    let (small, _) = frame(&mut batch, h, None, None, None);
    assert_eq!(
        small,
        vec![Phase::Done; SMALL as usize],
        "and the next frame"
    );

    // Back to the first size: zeroed again, then the written values step as before.
    let (back, _) = frame(&mut batch, h, Some(N), None, None);
    assert_eq!(
        back,
        vec![Phase::Run; N as usize],
        "after the second resize"
    );
    let (rewritten, _) = frame(&mut batch, h, None, Some(&line), None);
    assert_eq!(rewritten, stepped, "a write after the resizes");
}

/// A word that is no variant, stored by `raw` and viewed as `Phase`, is read by the in-place stage as the last variant
/// and stored as the variant the body chose: the ticket then holds valid words only.
fn a_bad_word_decoded_in_place_is_stored_as_a_valid_word<D: BatchDevice>(
    mut batch: Batch<D>,
    bad: Stage<Phase, D1>,
) where
    D::Buffer<u32>: 'static,
{
    let words = block_on(async {
        let mut frame = batch.begin().await?;
        let ticket = frame.ticket(&bad)?;
        let _submitted = frame.submit()?;
        ticket.await
    })
    .unwrap();
    // 9 reads as `Done`, and the body maps `Done` to `Run` (word 3): a skipped store leaves 9 (`InvalidEnum`), a
    // wrong encode or a fallback to another variant gives another word.
    assert_eq!(words, vec![Phase::Run; N as usize]);
}

#[test]
fn twin_an_enum_target_keeps_its_stored_enums_across_frames_and_a_resize() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu(&device).unwrap();
    the_target_keeps_its_stored_enums_across_frames_and_a_resize(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_an_enum_target_keeps_its_stored_enums_across_frames_and_a_resize() {
    let device = private_device();
    println!("adapter: {}", device.adapter_info().name);
    let (batch, h) = build_gpu(&device).unwrap();
    the_target_keeps_its_stored_enums_across_frames_and_a_resize(batch, &h);
}

#[test]
fn twin_a_bad_word_decoded_in_place_is_stored_as_a_valid_word() {
    let device = Arc::new(CpuDevice::new());
    let (batch, bad) = laundered_cpu(&device).unwrap();
    a_bad_word_decoded_in_place_is_stored_as_a_valid_word(batch, bad);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_bad_word_decoded_in_place_is_stored_as_a_valid_word() {
    let device = private_device();
    println!("adapter: {}", device.adapter_info().name);
    let (batch, bad) = laundered_gpu(&device).unwrap();
    a_bad_word_decoded_in_place_is_stored_as_a_valid_word(batch, bad);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_an_enum_target_equals_the_cpu_twin() {
    let gpu = private_device();
    let cpu = Arc::new(CpuDevice::new());
    println!("adapter: {}", gpu.adapter_info().name);
    let (mut on_gpu, gpu_h) = build_gpu(&gpu).unwrap();
    let (mut on_cpu, cpu_h) = build_cpu(&cpu).unwrap();
    for seed in 0..3 {
        let line = phases_of(N, seed);
        let grid = phases_of(W * H, seed + 1);
        let from_gpu = frame(&mut on_gpu, &gpu_h, None, Some(&line), Some(&grid));
        let from_cpu = frame(&mut on_cpu, &cpu_h, None, Some(&line), Some(&grid));
        assert_eq!(from_gpu, from_cpu, "seed {seed}");
    }
}
