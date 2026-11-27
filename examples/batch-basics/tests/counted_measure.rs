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
//!
//! Measurement s2i.m2 (the second test, `GUST_S2I_M2_PROBE=1`, `GUST_S2I_M2_CELLS=small|all`, default `all`): the same
//! two launches timed by the frame's GPU span (`FrameTiming::Frame`, D277), beside the wall-clock of the same frames,
//! over the cells of the ruling, Breeze's cells and one frame of six counted stages. A counted stage's args dispatch
//! and indirect dispatch are one entry at `FrameTiming::Passes` and join the frame's one pass, so the span at `Frame`
//! is the quantity. wgpu's indirect validation is recorded before the indirect dispatch, so the span includes it. On a
//! device without timestamp queries the probe prints a different quantity (16 frames submitted, one wait, time over
//! 16) and says so. The six-stage cell is labelled in its output: the difference of six stages is not one launch's
//! fixed cost. Before timing, every batch's launch is checked (a batch asked for `Indirect` that fell back to
//! `Bounded` fails), and the median over the rounds is the mean of the two middle ones.
//!
//! `GUST_S2I_M2_PROBE=1 cargo test -p batch-basics --release --test counted_measure gpu_s2i_m2 -- --nocapture`
//!
//! Measurement s2i.pass (`gpu_s2i_pass_span`, `GUST_S2I_PASS_PROBE=1`, `GUST_S2I_PASS_ROUNDS`, default 4): the frame's
//! GPU span of the indirect launch recorded in the frame's one pass (the default), recorded as the old form
//! (`force_pass_per_dispatch`, two passes a counted stage) and of the bounded launch, over a one-stage cell at 65536
//! and at 2^20 rows and the six-stage cell, then the `Passes` reading of the six-stage cell of each form (the entry
//! of each stage and the frame's span).
//!
//! `GUST_S2I_PASS_PROBE=1 cargo test -p batch-basics --release --test counted_measure gpu_s2i_pass -- --nocapture`

use std::sync::Arc;
use std::time::Instant;

use batch_basics::app::block_on;
use gust::batch::{Batch, CountedLaunch, D1, Error, FrameTiming, Stage, set_counted_launch};
use gust::{gpu_batch, gpu_task};
use gust_wgpu::{HeadlessDevice, force_pass_per_dispatch};

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
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
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

// ---- measurement s2i.m2: the GPU span of the indirect launch against the bounded one ----

const M2_WARMUP: usize = 10;
const M2_FRAMES: usize = 101;
const M2_ROUNDS: usize = 4;
/// The fallback's batch: frames encoded and submitted before the one wait.
const M2_BATCH: usize = 16;
const M2_VARIANTS: [CountedLaunch; 2] = [CountedLaunch::Indirect, CountedLaunch::Bounded];

/// One cell: the capacity and, per counted stage of the frame, its fill in hundredths of a percent.
struct Cell {
    capacity: u32,
    fills: Vec<u32>,
}

impl Cell {
    fn single(capacity: u32, fill: u32) -> Self {
        Self {
            capacity,
            fills: vec![fill],
        }
    }

    /// The count of each stage, in rows.
    fn counts(&self) -> Vec<u32> {
        self.fills
            .iter()
            .map(|&fill| (u64::from(self.capacity) * u64::from(fill) / 10_000) as u32)
            .collect()
    }

    fn label(&self) -> String {
        let fills: Vec<String> = self
            .fills
            .iter()
            .map(|&fill| format!("{}", f64::from(fill) / 100.0))
            .collect();
        let label = format!("{}@{}", self.capacity, fills.join("/"));
        if self.fills.len() > 1 {
            format!("{label} [six stages: the difference is not one launch's fixed cost]")
        } else {
            label
        }
    }
}

/// The cells: the proxy ruling's 8, Breeze's single-stage ones not already among them, and the 6-stage frame.
fn m2_cells() -> Vec<Cell> {
    let mut cells = Vec::new();
    let mut add = |cell: Cell| {
        let has = |other: &Cell| other.capacity == cell.capacity && other.fills == cell.fills;
        if !cells.iter().any(has) {
            cells.push(cell);
        }
    };
    for capacity in [65_536, 1 << 20, 4 << 20, 16 << 20] {
        for fill in [100, 10_000] {
            add(Cell::single(capacity, fill));
        }
    }
    for capacity in [1u32 << 20, 1 << 22] {
        for fill in [10, 100, 1_000, 900, 1_750, 2_600, 3_500] {
            add(Cell::single(capacity, fill));
        }
    }
    add(Cell {
        capacity: 1 << 20,
        fills: vec![3_500, 2_600, 1_750, 900, 100, 10],
    });
    cells
}

/// The handles of one rig: a count input per counted stage, the shared source and each stage's result.
struct Rig {
    counts: Vec<Stage<u32, D1>>,
    src: Stage<u32, D1>,
    outs: Vec<Stage<u32, D1>>,
}

/// What the block returns: the count inputs, the source and the results.
type Returned<const N: usize> = ([Stage<u32, D1>; N], Stage<u32, D1>, [Stage<u32, D1>; N]);

/// The block returns the handles as arrays (the macro finds the handles it returns by name); a rig holds them as vectors.
fn into_rig<const N: usize>(
    (batch, (counts, src, outs)): (Batch<HeadlessDevice>, Returned<N>),
) -> (Batch<HeadlessDevice>, Rig) {
    let rig = Rig {
        counts: counts.to_vec(),
        src,
        outs: outs.to_vec(),
    };
    (batch, rig)
}

macro_rules! define_m2_builds {
    ($one:ident, $six:ident, $depth:tt) => {
        /// A batch of one counted stage over the light kernel of the first probe, timed at level `Frame`.
        fn $one(device: &Arc<HeadlessDevice>, capacity: u32, level: FrameTiming) -> Result<(Batch<HeadlessDevice>, Rig), Error> {
            gpu_batch!("measure-m2", on(device); in_flight($depth); |b| {
                b.frame_timing(level);
                let n = b.input::<u32>(1u32);
                let src = b.input::<u32>(capacity);
                let out = gpu_task!(count(n); capacity, |i| -> u32 {
                    let v = src[i];
                    (v * 3u32 + 1u32) ^ (v >> 3u32)
                });
                ([n], src, [out])
            }).map(into_rig)

        }

        /// A batch of six counted stages over the same kernel, each with a count of its own.
        fn $six(device: &Arc<HeadlessDevice>, capacity: u32, level: FrameTiming) -> Result<(Batch<HeadlessDevice>, Rig), Error> {
            gpu_batch!("measure-m2-six", on(device); in_flight($depth); |b| {
                b.frame_timing(level);
                let src = b.input::<u32>(capacity);
                let n0 = b.input::<u32>(1u32);
                let o0 = gpu_task!(count(n0); capacity, |i| -> u32 {
                    let v = src[i];
                    (v * 3u32 + 1u32) ^ (v >> 3u32)
                });
                let n1 = b.input::<u32>(1u32);
                let o1 = gpu_task!(count(n1); capacity, |i| -> u32 {
                    let v = src[i];
                    (v * 3u32 + 1u32) ^ (v >> 3u32)
                });
                let n2 = b.input::<u32>(1u32);
                let o2 = gpu_task!(count(n2); capacity, |i| -> u32 {
                    let v = src[i];
                    (v * 3u32 + 1u32) ^ (v >> 3u32)
                });
                let n3 = b.input::<u32>(1u32);
                let o3 = gpu_task!(count(n3); capacity, |i| -> u32 {
                    let v = src[i];
                    (v * 3u32 + 1u32) ^ (v >> 3u32)
                });
                let n4 = b.input::<u32>(1u32);
                let o4 = gpu_task!(count(n4); capacity, |i| -> u32 {
                    let v = src[i];
                    (v * 3u32 + 1u32) ^ (v >> 3u32)
                });
                let n5 = b.input::<u32>(1u32);
                let o5 = gpu_task!(count(n5); capacity, |i| -> u32 {
                    let v = src[i];
                    (v * 3u32 + 1u32) ^ (v >> 3u32)
                });
                (
                    [n0, n1, n2, n3, n4, n5],
                    src,
                    [o0, o1, o2, o3, o4, o5],
                )
            }).map(into_rig)

        }
    };
}

define_m2_builds!(build_one, build_six, 1);
// The fallback keeps four frames in flight (the most a batch takes) so that its frames queue before the one wait.
define_m2_builds!(build_one_deep, build_six_deep, 4);

/// The batch of `stages` counted stages (1 or 6) at `capacity`, made with `launch`, timed at level `Frame`.
fn m2_build(
    device: &Arc<HeadlessDevice>,
    capacity: u32,
    stages: usize,
    deep: bool,
    launch: CountedLaunch,
) -> (Batch<HeadlessDevice>, Rig) {
    m2_build_at(device, capacity, stages, deep, launch, FrameTiming::Frame)
}

/// [`m2_build`] timed at `level`.
fn m2_build_at(
    device: &Arc<HeadlessDevice>,
    capacity: u32,
    stages: usize,
    deep: bool,
    launch: CountedLaunch,
    level: FrameTiming,
) -> (Batch<HeadlessDevice>, Rig) {
    HeadlessDevice::counted_probe(true);
    set_counted_launch(Some(launch));
    let built = match (stages, deep) {
        (1, false) => build_one(device, capacity, level),
        (1, true) => build_one_deep(device, capacity, level),
        (6, false) => build_six(device, capacity, level),
        (6, true) => build_six_deep(device, capacity, level),
        _ => unreachable!("the cells have one or six stages"),
    };
    set_counted_launch(None);
    let sealed_indirect = HeadlessDevice::counted_probe_args().len();
    HeadlessDevice::counted_probe(false);
    let expected = if launch == CountedLaunch::Indirect {
        stages
    } else {
        0
    };
    assert_eq!(
        sealed_indirect, expected,
        "the batch asked for {launch:?} sealed {sealed_indirect} indirect stages, expected {expected}"
    );
    built.unwrap()
}

/// One checked frame: every stage's result is the model's over its first rows.
fn m2_check(
    batch: &mut Batch<HeadlessDevice>,
    rig: &Rig,
    counts: &[u32],
    capacity: u32,
    src: &[u32],
    write: bool,
) {
    let outs = block_on(async {
        let mut frame = batch.begin().await?;
        for (stage, &n) in rig.counts.iter().zip(counts) {
            frame.write(stage, &[n])?;
        }
        if write {
            frame.write(&rig.src, src)?;
        }
        let mut tickets = Vec::new();
        for out in &rig.outs {
            tickets.push(frame.ticket(out)?);
        }
        let _submitted = frame.submit()?;
        let mut results = Vec::new();
        for ticket in tickets {
            let result: Result<Vec<u32>, Error> = ticket.await;
            results.push(result?);
        }
        Ok::<_, Error>(results)
    })
    .unwrap();
    for (stage, (out, &n)) in outs.iter().zip(counts).enumerate() {
        let rows = n.min(capacity) as usize;
        for i in (0..rows).step_by(4099).chain([rows.saturating_sub(1)]) {
            let v = src[i];
            assert_eq!(out[i], (v * 3 + 1) ^ (v >> 3), "stage {stage}, row {i}");
        }
    }
}

/// One timed frame of the span's quantity: the wall-clock of `begin`, the writes, `submit` and the wait (the time is
/// asked for before the submit, and read after the clock stopped), and the frame's GPU span in microseconds.
fn m2_span_frame(
    batch: &mut Batch<HeadlessDevice>,
    rig: &Rig,
    counts: &[u32],
) -> (f64, Option<f64>) {
    let started = Instant::now();
    let mut frame = block_on(batch.begin()).unwrap();
    for (stage, &n) in rig.counts.iter().zip(counts) {
        frame.write(stage, &[n]).unwrap();
    }
    let time = frame.time();
    let submitted = frame.submit().unwrap();
    submitted.wait().unwrap();
    let wall = started.elapsed().as_secs_f64() * 1e6;
    let span = block_on(time)
        .unwrap()
        .map(|times| times.span.as_secs_f64() * 1e6);
    (wall, span)
}

/// The fallback's quantity (a different one, never mixed with the span): [`M2_BATCH`] frames encoded and submitted,
/// one wait for the last, the wall-clock divided by the number of frames, in microseconds.
fn m2_batched_frame(batch: &mut Batch<HeadlessDevice>, rig: &Rig, counts: &[u32]) -> f64 {
    let started = Instant::now();
    let mut last = None;
    for _ in 0..M2_BATCH {
        let mut frame = block_on(batch.begin()).unwrap();
        for (stage, &n) in rig.counts.iter().zip(counts) {
            frame.write(stage, &[n]).unwrap();
        }
        last = Some(frame.submit().unwrap());
    }
    last.expect("a batch has frames").wait().unwrap();
    started.elapsed().as_secs_f64() * 1e6 / M2_BATCH as f64
}

/// What one variant of one cell measured in one round: the median of the quantity and the median wall-clock.
#[derive(Clone, Copy, Default)]
struct Measured {
    quantity: f64,
    wall: f64,
}

fn m2_measure(
    batch: &mut Batch<HeadlessDevice>,
    rig: &Rig,
    counts: &[u32],
    span_mode: bool,
) -> Measured {
    let (mut quantities, mut walls) = (Vec::new(), Vec::new());
    for round in 0..M2_WARMUP + M2_FRAMES {
        let (quantity, wall) = if span_mode {
            let (wall, span) = m2_span_frame(batch, rig, counts);
            (
                span.expect("the batch is timed and the device has timestamp queries"),
                wall,
            )
        } else {
            let batched = m2_batched_frame(batch, rig, counts);
            (batched, batched)
        };
        if round >= M2_WARMUP {
            quantities.push(quantity);
            walls.push(wall);
        }
    }
    Measured {
        quantity: median(quantities),
        wall: median(walls),
    }
}

/// Every round's median in the order of the rounds (round 0 and 2 created Indirect first), as text.
fn m2_rounds_text(values: &[f64]) -> String {
    let texts: Vec<String> = values.iter().map(|value| format!("{value:.1}")).collect();
    texts.join(";")
}

/// Measurement s2i.m2 (see the header). Skipped unless `GUST_S2I_M2_PROBE=1`; `GUST_S2I_M2_CELLS=small` keeps the
/// cells up to 2^20 rows (a smoke), `all` (the default) runs every cell.
#[test]
fn gpu_s2i_m2_span_indirect_against_bounded() {
    // A probe, not a gate test: no `#[ignore]` for the pin scan to list.
    if std::env::var("GUST_S2I_M2_PROBE").as_deref() != Ok("1") {
        return;
    }
    let small_only = match std::env::var("GUST_S2I_M2_CELLS").as_deref() {
        Ok("small") => true,
        Ok("all") | Err(_) => false,
        Ok(other) => panic!("GUST_S2I_M2_CELLS is `small` or `all`, not `{other}`"),
    };
    let cells: Vec<Cell> = m2_cells()
        .into_iter()
        .filter(|cell| !small_only || cell.capacity <= 1 << 20)
        .collect();
    let device = Arc::new(HeadlessDevice::new().expect("a GPU adapter is required"));
    let adapter = device.adapter_info();
    println!("adapter: {} (backend {:?})", adapter.name, adapter.backend);
    let span_mode = m2_build(&device, 1024, 1, false, CountedLaunch::Indirect)
        .0
        .frame_timing()
        .is_some();
    if span_mode {
        println!(
            "quantity: the GPU span of the frame at FrameTiming::Frame, microseconds; the Instant columns are the \
             wall-clock of begin, writes, submit and wait of the same frames (the time is asked for in them)"
        );
    } else {
        println!(
            "quantity: FALLBACK, not the span: {M2_BATCH} frames submitted, one wait, wall-clock over {M2_BATCH}, \
             microseconds; both pairs of columns carry it"
        );
    }
    println!(
        "{M2_ROUNDS} rounds, creation order rotated, {M2_FRAMES} frames after {M2_WARMUP} warm-up per variant and round; \
         cell = capacity@fill percent of each counted stage"
    );
    println!(
        "cell,indirect,bounded,ratio_bounded_over_indirect,indirect_minus_bounded_us,indirect_rounds,bounded_rounds,\
         indirect_instant,bounded_instant,instant_ratio"
    );
    // The cells that share a batch (a capacity and a number of counted stages), in the order of the cells.
    let mut groups: Vec<(u32, usize)> = Vec::new();
    for cell in &cells {
        let key = (cell.capacity, cell.fills.len());
        if !groups.contains(&key) {
            groups.push(key);
        }
    }
    for (capacity, stages) in groups {
        let group: Vec<&Cell> = cells
            .iter()
            .filter(|cell| cell.capacity == capacity && cell.fills.len() == stages)
            .collect();
        let src: Vec<u32> = (0..capacity).map(|i| i % 1000).collect();
        // [variant][cell] -> the round's measurements
        let mut rounds = vec![vec![Vec::<Measured>::new(); group.len()]; M2_VARIANTS.len()];
        for round in 0..M2_ROUNDS {
            let order = if round % 2 == 0 { [0, 1] } else { [1, 0] };
            let mut batches = Vec::new();
            for variant in order {
                let rig = m2_build(&device, capacity, stages, !span_mode, M2_VARIANTS[variant]);
                batches.push((variant, rig));
            }
            for (variant, (batch, rig)) in &mut batches {
                for (slot, cell) in group.iter().enumerate() {
                    let counts = cell.counts();
                    m2_check(batch, rig, &counts, capacity, &src, slot == 0);
                    rounds[*variant][slot].push(m2_measure(batch, rig, &counts, span_mode));
                }
            }
        }
        for (slot, cell) in group.iter().enumerate() {
            let column = |variant: usize, wall: bool| -> Vec<f64> {
                rounds[variant][slot]
                    .iter()
                    .map(|m| if wall { m.wall } else { m.quantity })
                    .collect()
            };
            let (i, b) = (median(column(0, false)), median(column(1, false)));
            let (iw, bw) = (median(column(0, true)), median(column(1, true)));
            println!(
                "{},{i:.1},{b:.1},{:.3},{:.1},{},{},{iw:.1},{bw:.1},{:.3}",
                cell.label(),
                b / i,
                i - b,
                m2_rounds_text(&column(0, false)),
                m2_rounds_text(&column(1, false)),
                bw / iw,
            );
        }
    }
}

// ---- measurement s2i.pass: the passes of an indirect counted stage (m14.s2imerge) ----

/// The layouts one Indirect batch is timed in (the setting flips between the first two) and the Bounded reference.
const PASS_VARIANTS: [&str; 3] = ["default", "forced", "bounded"];

/// The cells: the two 100 % one-stage cells and the six-stage cell.
fn pass_cells() -> Vec<Cell> {
    vec![
        Cell::single(65_536, 10_000),
        Cell::single(1 << 20, 10_000),
        Cell {
            capacity: 1 << 20,
            fills: vec![3_500, 2_600, 1_750, 900, 100, 10],
        },
    ]
}

/// A cell's label without the m2 note.
fn pass_label(cell: &Cell) -> String {
    let fills: Vec<String> = cell
        .fills
        .iter()
        .map(|&fill| format!("{}", f64::from(fill) / 100.0))
        .collect();
    format!("{}@{}", cell.capacity, fills.join("/"))
}

/// The passes a frame records per variant for `stages` counted stages: one in the default form and for Bounded, two a
/// stage under `force_pass_per_dispatch`.
fn expected_passes(variant: usize, stages: usize) -> u64 {
    if variant == 1 { 2 * stages as u64 } else { 1 }
}

/// Switch the pass form of the next frames: variant 1 is the forced (old) one, the others the default.
fn set_layout(variant: usize) {
    force_pass_per_dispatch(variant == 1);
}

/// One frame with the given counts: every stage's whole result buffer is read, and the passes the frame recorded.
fn pass_frame(
    batch: &mut Batch<HeadlessDevice>,
    rig: &Rig,
    counts: &[u32],
) -> (Vec<Vec<u32>>, u64) {
    let before = gust_wgpu::batch_passes_recorded();
    block_on(async {
        let mut frame = batch.begin().await?;
        for (stage, &n) in rig.counts.iter().zip(counts) {
            frame.write(stage, &[n])?;
        }
        let mut tickets = Vec::new();
        for out in &rig.outs {
            tickets.push(frame.ticket(out)?);
        }
        let _submitted = frame.submit()?;
        let passes = gust_wgpu::batch_passes_recorded() - before;
        let mut results = Vec::new();
        for ticket in tickets {
            let result: Result<Vec<u32>, Error> = ticket.await;
            results.push(result?);
        }
        Ok::<_, Error>((results, passes))
    })
    .unwrap()
}

/// Measurement s2i.pass, timing (`GUST_S2I_PASS_PROBE=1`; `GUST_S2I_PASS_ROUNDS`, default 4): the frame's GPU span at
/// `Frame` of the default form, the forced form and Bounded over [`pass_cells`], in the shape of s2i.m2 (two batches a
/// round, Indirect and Bounded, whose creation order alternates; within the Indirect batch the order default, forced
/// alternates). Per cell it prints the spans, `d` = span minus Bounded for the default and forced forms, and the ratio of
/// the default over the forced span. The pass count of every variant is asserted before timing, and the results are
/// held to the model. Then, for the six-stage cell, the `Passes` reading of each form: the entry of every stage and the
/// frame's span, medians over the same number of frames.
#[test]
fn gpu_s2i_pass_span() {
    // A probe, not a gate test: no `#[ignore]` for the pin scan to list.
    if std::env::var("GUST_S2I_PASS_PROBE").as_deref() != Ok("1") {
        return;
    }
    let rounds_wanted: usize = std::env::var("GUST_S2I_PASS_ROUNDS")
        .ok()
        .map_or(M2_ROUNDS, |text| {
            text.parse().expect("GUST_S2I_PASS_ROUNDS is a number")
        });
    let device = Arc::new(HeadlessDevice::new().expect("a GPU adapter is required"));
    let adapter = device.adapter_info();
    println!("adapter: {} (backend {:?})", adapter.name, adapter.backend);
    assert!(
        m2_build(&device, 1024, 1, false, CountedLaunch::Indirect)
            .0
            .frame_timing()
            .is_some(),
        "the span needs timestamp queries"
    );
    println!(
        "quantity: the GPU span of the frame at FrameTiming::Frame, microseconds; {rounds_wanted} rounds, \
         {M2_FRAMES} frames after {M2_WARMUP} warm-up per variant and round; d = variant minus bounded"
    );
    println!(
        "cell,default,forced,bounded,d_default,d_forced,ratio_default_over_forced,default_rounds,forced_rounds,\
         bounded_rounds,default_wall,forced_wall,bounded_wall"
    );
    for cell in pass_cells() {
        let (capacity, stages) = (cell.capacity, cell.fills.len());
        let src: Vec<u32> = (0..capacity).map(|i| i % 1000).collect();
        let counts = cell.counts();
        // [variant] -> the round's measurements
        let mut rounds = vec![Vec::<Measured>::new(); PASS_VARIANTS.len()];
        for round in 0..rounds_wanted {
            let order = if round % 2 == 0 { [0, 1] } else { [1, 0] };
            let mut batches = Vec::new();
            for batch_kind in order {
                // batch_kind 0: the Indirect batch, 1: the Bounded one
                let launch = if batch_kind == 0 {
                    CountedLaunch::Indirect
                } else {
                    CountedLaunch::Bounded
                };
                batches.push((
                    batch_kind,
                    m2_build(&device, capacity, stages, false, launch),
                ));
            }
            for (batch_kind, (batch, rig)) in &mut batches {
                m2_check(batch, rig, &counts, capacity, &src, true);
                let variants: Vec<usize> = match (*batch_kind, round % 2) {
                    (0, 0) => vec![0, 1],
                    (0, _) => vec![1, 0],
                    _ => vec![2],
                };
                for variant in variants {
                    set_layout(variant);
                    let (outs, passes) = pass_frame(batch, rig, &counts);
                    assert_eq!(
                        passes,
                        expected_passes(variant, stages),
                        "{}: passes recorded",
                        PASS_VARIANTS[variant]
                    );
                    assert_eq!(outs.len(), stages);
                    rounds[variant].push(m2_measure(batch, rig, &counts, true));
                }
                set_layout(0);
            }
        }
        let column = |variant: usize, wall: bool| -> Vec<f64> {
            rounds[variant]
                .iter()
                .map(|m| if wall { m.wall } else { m.quantity })
                .collect()
        };
        let [d, f, b] = [0, 1, 2].map(|variant| median(column(variant, false)));
        let [dw, fw, bw] = [0, 1, 2].map(|variant| median(column(variant, true)));
        println!(
            "{},{d:.1},{f:.1},{b:.1},{:.1},{:.1},{:.3},{},{},{},{dw:.1},{fw:.1},{bw:.1}",
            pass_label(&cell),
            d - b,
            f - b,
            d / f,
            m2_rounds_text(&column(0, false)),
            m2_rounds_text(&column(1, false)),
            m2_rounds_text(&column(2, false)),
        );
    }
    passes_reading(&device);
}

/// The `Passes` reading of the six-stage cell for each form: one line a form, `passes,<form>,<frame span>,<entry of
/// stage 0>;..;<entry of stage 5>` in microseconds, medians over [`M2_FRAMES`] frames after [`M2_WARMUP`], taken in
/// the order default, forced, bounded on a batch built just before.
fn passes_reading(device: &Arc<HeadlessDevice>) {
    let cell = pass_cells().pop().expect("the six-stage cell is last");
    let (capacity, stages) = (cell.capacity, cell.fills.len());
    let src: Vec<u32> = (0..capacity).map(|i| i % 1000).collect();
    let counts = cell.counts();
    println!("passes,form,frame_span_us,stage_entries_us");
    for (variant, name) in PASS_VARIANTS.into_iter().enumerate() {
        let launch = if variant == 2 {
            CountedLaunch::Bounded
        } else {
            CountedLaunch::Indirect
        };
        let (mut batch, rig) =
            m2_build_at(device, capacity, stages, false, launch, FrameTiming::Passes);
        m2_check(&mut batch, &rig, &counts, capacity, &src, true);
        set_layout(variant);
        let (mut spans, mut entries) = (Vec::new(), vec![Vec::new(); stages]);
        for round in 0..M2_WARMUP + M2_FRAMES {
            let mut frame = block_on(batch.begin()).unwrap();
            for (stage, &n) in rig.counts.iter().zip(&counts) {
                frame.write(stage, &[n]).unwrap();
            }
            let time = frame.time();
            let submitted = frame.submit().unwrap();
            submitted.wait().unwrap();
            let times = block_on(time)
                .unwrap()
                .expect("the batch is timed and the device has timestamp queries");
            if round >= M2_WARMUP {
                assert_eq!(times.passes.len(), stages, "one entry a stage");
                spans.push(times.span.as_secs_f64() * 1e6);
                for (slot, entry) in entries.iter_mut().zip(&times.passes) {
                    slot.push(entry.duration.as_secs_f64() * 1e6);
                }
            }
        }
        set_layout(0);
        let per_stage: Vec<String> = entries
            .into_iter()
            .map(|values| format!("{:.1}", median(values)))
            .collect();
        println!("passes,{name},{:.1},{}", median(spans), per_stage.join(";"));
    }
}
