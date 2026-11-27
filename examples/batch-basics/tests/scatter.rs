//! The scatter stage (m14.s2sc): `gpu_task!(count(n); CAP, scatter(&mut out), |i| { out[place] = value; })` runs over
//! the rows of its extent and stores each row's value at a place the row computes, in one or several handles of the
//! block (the targets). The stage declares that the places of its rows are pairwise distinct within one run of the stage;
//! the GPU checks nothing, and the CPU twin panics on a repeated place, naming the stage, the target and the slot. A
//! path through the body may store any subset of the targets, or none, and a slot no row names keeps every byte.
//!
//! Every test compares the whole target buffers with a host model. Every value is small and exactly representable, so a
//! result equals the model bit for bit. The `twin_` tests run on the CPU twin; the `gpu_` ones are `#[ignore]`d like the
//! example's other GPU tests, print the adapter and run in full verification on the adapter the environment selects.
//! Where a stage is counted, the `gpu_` test runs under both counted launches (D277).

// A scatter stage has the value `()`, and its `let` names the stage for the twin's panic message.
#![allow(clippy::let_unit_value)]

use std::sync::Arc;

use batch_basics::app::block_on;
use gust::batch::{Batch, BatchDevice, CountedLaunch, D1, Error, Param, Stage, set_counted_launch};
use gust::cpu::CpuDevice;
use gust::{gpu_batch, gpu_task};
use gust_wgpu::HeadlessDevice;

mod rows {
    /// Three words of two kinds, so that a field stored at the wrong offset is seen.
    #[derive(Clone, Copy, Debug, PartialEq, gust::GpuStruct)]
    #[repr(C)]
    pub struct Row {
        pub life: f32,
        pub flags: u32,
        pub tag: u32,
    }
}

mod phases {
    /// Words with gaps and a zero.
    #[derive(Clone, Copy, Debug, PartialEq, gust::GpuEnum)]
    #[repr(u32)]
    pub enum Phase {
        Idle = 0,
        Run = 3,
        Done = 7,
    }
}

use phases::Phase;
use rows::Row;

const N: u32 = 64;
const LAUNCHES: [CountedLaunch; 2] = [CountedLaunch::Indirect, CountedLaunch::Bounded];

/// What the places a stage does not name hold before the first frame's stage runs: a value no row writes.
const SENTINEL: Row = Row {
    life: -1.0,
    flags: 99,
    tag: 9999,
};

/// Forces a launch for the batches this thread builds until it is dropped.
struct Launch;

impl Launch {
    fn set(launch: CountedLaunch) -> Self {
        set_counted_launch(Some(launch));
        Self
    }
}

impl Drop for Launch {
    fn drop(&mut self) {
        set_counted_launch(None);
    }
}

fn private_device() -> Arc<HeadlessDevice> {
    let device = Arc::new(HeadlessDevice::new().unwrap_or_else(|error| {
        panic!("a GPU adapter is required; this test must not silently skip: {error}")
    }));
    println!("adapter: {}", device.adapter_info().name);
    device
}

fn rows_for(k: u32, len: u32) -> Vec<Row> {
    (0..len)
        .map(|i| Row {
            life: i as f32 * 0.25 + k as f32,
            flags: i + k,
            tag: 1000 + i,
        })
        .collect()
}

/// `(i * 7 + 3) % N`: a permutation of `0..N`, since 7 and 64 are coprime.
fn permutation() -> Vec<u32> {
    (0..N).map(|i| (i * 7 + 3) % N).collect()
}

// ---- one target, the place a handle read ----

#[derive(Clone, Copy)]
struct Plain {
    rows: Stage<Row, D1>,
    dst: Stage<u32, D1>,
    out: Stage<Row, D1>,
}

macro_rules! define_plain {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Plain), Error> {
            gpu_batch!("scatter_plain", on(device); in_flight(1); |b| {
                let rows = b.input::<Row>(N);
                let dst = b.input::<u32>(N);
                let out = b.input::<Row>(N);
                let _scatter = gpu_task!(N, scatter(&mut out), |i| {
                    out[dst[i]] = rows[i];
                });
                Plain { rows, dst, out }
            })
        }
    };
}

define_plain!(plain_cpu, CpuDevice);
define_plain!(plain_gpu, HeadlessDevice);

/// A frame: the rows, the places and the target's contents before the stage written, then the target read back.
fn plain_frame<D: BatchDevice>(
    batch: &mut Batch<D>,
    h: &Plain,
    rows: &[Row],
    dst: &[u32],
    before: &[Row],
) -> Result<Vec<Row>, Error>
where
    D::Buffer<u32>: 'static,
    D::Buffer<Row>: 'static,
{
    block_on(async {
        let mut frame = batch.begin().await?;
        frame.write(&h.rows, rows)?;
        frame.write(&h.dst, dst)?;
        frame.write(&h.out, before)?;
        let out = frame.ticket(&h.out)?;
        let _submitted = frame.submit()?;
        out.await
    })
}

/// The host model of a scatter: each row's value at its place, every other slot as it was.
fn scattered(
    rows: &[Row],
    dst: &[u32],
    before: &[Row],
    stores: impl Fn(usize) -> bool,
) -> Vec<Row> {
    let mut out = before.to_vec();
    for (i, row) in rows.iter().enumerate() {
        if stores(i) {
            out[dst[i] as usize] = *row;
        }
    }
    out
}

fn permutation_frames<D: BatchDevice>(mut batch: Batch<D>, h: &Plain, what: &str) -> Vec<Vec<Row>>
where
    D::Buffer<u32>: 'static,
    D::Buffer<Row>: 'static,
{
    let dst = permutation();
    (0..3)
        .map(|k| {
            let rows = rows_for(k, N);
            let before = vec![SENTINEL; N as usize];
            let got = plain_frame(&mut batch, h, &rows, &dst, &before).unwrap();
            assert_eq!(
                got,
                scattered(&rows, &dst, &before, |_| true),
                "{what}: frame {k}"
            );
            got
        })
        .collect()
}

fn twin_permutation() -> Vec<Vec<Row>> {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = plain_cpu(&device).unwrap();
    permutation_frames(batch, &h, "twin")
}

#[test]
fn twin_scatter_permutes_struct_rows() {
    let frames = twin_permutation();
    // A permutation names every slot: no sentinel is left, and the row of place 3 is row 0.
    assert!(frames[0].iter().all(|row| *row != SENTINEL));
    assert_eq!(frames[0][3].tag, 1000);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_scatter_permutes_struct_rows() {
    let device = private_device();
    let twin = twin_permutation();
    let (batch, h) = plain_gpu(&device).unwrap();
    assert_eq!(permutation_frames(batch, &h, "gpu"), twin);
}

// ---- counted: a reversal, the count clamped ----

#[derive(Clone, Copy)]
struct Counted {
    rows: Stage<Row, D1>,
    live: Stage<u32, D1>,
    dst: Stage<u32, D1>,
    out: Stage<Row, D1>,
}

macro_rules! define_counted {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Counted), Error> {
            gpu_batch!("scatter_counted", on(device); in_flight(1); |b| {
                let rows = b.input::<Row>(N);
                let live = b.input::<u32>(1u32);
                let dst = b.input::<u32>(N);
                let out = b.input::<Row>(N);
                let _scatter = gpu_task!(count(live); N, scatter(&mut out), |i| {
                    out[dst[i]] = rows[i];
                });
                Counted { rows, live, dst, out }
            })
        }
    };
}

define_counted!(counted_cpu, CpuDevice);
define_counted!(counted_gpu, HeadlessDevice);

fn counted_frames<D: BatchDevice>(mut batch: Batch<D>, h: &Counted, what: &str) -> Vec<Vec<Row>>
where
    D::Buffer<u32>: 'static,
    D::Buffer<Row>: 'static,
{
    // A reversal of the first `n` rows (and of all of them past the capacity), places written for every row.
    let dst: Vec<u32> = (0..N).map(|i| N - 1 - i).collect();
    [0, 1, 10, N / 2, N, N + 25, 3]
        .into_iter()
        .enumerate()
        .map(|(k, live)| {
            let rows = rows_for(k as u32, N);
            let before = vec![SENTINEL; N as usize];
            let got = block_on(async {
                let mut frame = batch.begin().await?;
                frame.write(&h.rows, &rows)?;
                frame.write(&h.live, &[live])?;
                frame.write(&h.dst, &dst)?;
                frame.write(&h.out, &before)?;
                let out = frame.ticket(&h.out)?;
                let _submitted = frame.submit()?;
                out.await
            })
            .unwrap();
            let rows_run = live.min(N) as usize;
            let want = scattered(&rows, &dst, &before, |i| i < rows_run);
            assert_eq!(got, want, "{what}: count {live}, frame {k}");
            got
        })
        .collect()
}

fn twin_counted() -> Vec<Vec<Row>> {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = counted_cpu(&device).unwrap();
    counted_frames(batch, &h, "twin")
}

#[test]
fn twin_counted_scatter_runs_the_live_rows_only() {
    let frames = twin_counted();
    // Count 0: nothing stored. Count 1: row 0 at the last place. Count above the capacity: the whole reversal.
    assert!(frames[0].iter().all(|row| *row == SENTINEL));
    assert_eq!(frames[1][N as usize - 1].tag, 1000);
    assert_eq!(frames[1].iter().filter(|row| **row != SENTINEL).count(), 1);
    assert!(frames[5].iter().all(|row| *row != SENTINEL));
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_counted_scatter_runs_the_live_rows_only() {
    let device = private_device();
    let twin = twin_counted();
    for launch in LAUNCHES {
        let _launch = Launch::set(launch);
        let (batch, h) = counted_gpu(&device).unwrap();
        assert_eq!(
            counted_frames(batch, &h, &format!("{launch:?}")),
            twin,
            "{launch:?}"
        );
    }
}

// ---- the place is an expression of the rule: a counting sort's slots ----

#[derive(Clone, Copy)]
struct Sorted {
    rows: Stage<Row, D1>,
    start: Stage<u32, D1>,
    out: Stage<Row, D1>,
}

macro_rules! define_sorted {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Sorted), Error> {
            gpu_batch!("scatter_sorted", on(device); in_flight(1); |b| {
                let rows = b.input::<Row>(N);
                let start = b.input::<u32>(4u32);
                let out = b.input::<Row>(N);
                let _scatter = gpu_task!(N, scatter(&mut out), |i| {
                    let kind = rows[i].flags % 4u32;
                    let mut rank = 0u32;
                    for j in 0u32..i {
                        if rows[j].flags % 4u32 == kind {
                            rank += 1u32;
                        }
                    }
                    out[start[kind] + rank] = rows[i];
                });
                Sorted { rows, start, out }
            })
        }
    };
}

define_sorted!(sorted_cpu, CpuDevice);
define_sorted!(sorted_gpu, HeadlessDevice);

/// Rows of four kinds, `flags % 4`, in an order that mixes them.
fn kinds_for(k: u32) -> Vec<Row> {
    (0..N)
        .map(|i| Row {
            life: i as f32,
            flags: (i * 5 + k * 3 + i / 7) % 4 + 4 * (i % 3),
            tag: 1000 + i,
        })
        .collect()
}

fn sorted_frames<D: BatchDevice>(mut batch: Batch<D>, h: &Sorted, what: &str) -> Vec<Vec<Row>>
where
    D::Buffer<u32>: 'static,
    D::Buffer<Row>: 'static,
{
    (0..3)
        .map(|k| {
            let rows = kinds_for(k);
            // The first slot of each kind, from the counts a host computes.
            let mut counts = [0u32; 4];
            for row in &rows {
                counts[(row.flags % 4) as usize] += 1;
            }
            let start: Vec<u32> = (0..4).map(|kind| counts[..kind].iter().sum()).collect();
            let got = block_on(async {
                let mut frame = batch.begin().await?;
                frame.write(&h.rows, &rows)?;
                frame.write(&h.start, &start)?;
                frame.write(&h.out, &[SENTINEL; N as usize])?;
                let out = frame.ticket(&h.out)?;
                let _submitted = frame.submit()?;
                out.await
            })
            .unwrap();
            // The host's stable sort by kind.
            let mut want = rows.clone();
            want.sort_by_key(|row| row.flags % 4);
            assert_eq!(got, want, "{what}: frame {k}");
            got
        })
        .collect()
}

#[test]
fn twin_scatter_place_is_a_computed_expression() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = sorted_cpu(&device).unwrap();
    let frames = sorted_frames(batch, &h, "twin");
    assert!(
        frames[0]
            .windows(2)
            .all(|pair| pair[0].flags % 4 <= pair[1].flags % 4)
    );
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_scatter_place_is_a_computed_expression() {
    let device = private_device();
    let twin = {
        let cpu = Arc::new(CpuDevice::new());
        let (batch, h) = sorted_cpu(&cpu).unwrap();
        sorted_frames(batch, &h, "twin")
    };
    let (batch, h) = sorted_gpu(&device).unwrap();
    assert_eq!(sorted_frames(batch, &h, "gpu"), twin);
}

// ---- a path may store nothing: rows with a flag store no row ----

#[derive(Clone, Copy)]
struct Dead {
    rows: Stage<Row, D1>,
    dst: Stage<u32, D1>,
    out: Stage<Row, D1>,
}

macro_rules! define_dead {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Dead), Error> {
            gpu_batch!("scatter_dead", on(device); in_flight(1); |b| {
                let rows = b.input::<Row>(N);
                let dst = b.input::<u32>(N);
                let out = b.input::<Row>(N);
                let _scatter = gpu_task!(N, scatter(&mut out), |i| {
                    if rows[i].flags % 2u32 == 0u32 {
                        out[dst[i]] = rows[i];
                    }
                });
                Dead { rows, dst, out }
            })
        }
    };
}

define_dead!(dead_cpu, CpuDevice);
define_dead!(dead_gpu, HeadlessDevice);

fn dead_frames<D: BatchDevice>(mut batch: Batch<D>, h: &Dead, what: &str) -> Vec<Vec<Row>>
where
    D::Buffer<u32>: 'static,
    D::Buffer<Row>: 'static,
{
    // Every row names a place, but the odd ones store nothing: their places keep the sentinel, which a row that stores
    // nothing must not turn into anything else. The places are a permutation, so no two stores share one.
    let dst = permutation();
    (0..3)
        .map(|k| {
            let rows = rows_for(k, N);
            let before = vec![SENTINEL; N as usize];
            let got = plain_dead(&mut batch, h, &rows, &dst, &before);
            let want = scattered(&rows, &dst, &before, |i| rows[i].flags.is_multiple_of(2));
            assert_eq!(got, want, "{what}: frame {k}");
            got
        })
        .collect()
}

fn plain_dead<D: BatchDevice>(
    batch: &mut Batch<D>,
    h: &Dead,
    rows: &[Row],
    dst: &[u32],
    before: &[Row],
) -> Vec<Row>
where
    D::Buffer<u32>: 'static,
    D::Buffer<Row>: 'static,
{
    block_on(async {
        let mut frame = batch.begin().await?;
        frame.write(&h.rows, rows)?;
        frame.write(&h.dst, dst)?;
        frame.write(&h.out, before)?;
        let out = frame.ticket(&h.out)?;
        let _submitted = frame.submit()?;
        out.await
    })
    .unwrap()
}

#[test]
fn twin_scatter_path_that_stores_nothing_keeps_the_slot() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = dead_cpu(&device).unwrap();
    let frames = dead_frames(batch, &h, "twin");
    let kept = frames[0].iter().filter(|row| **row == SENTINEL).count();
    assert_eq!(
        kept,
        N as usize / 2,
        "the odd rows' places hold the sentinel"
    );
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_scatter_path_that_stores_nothing_keeps_the_slot() {
    let device = private_device();
    let twin = {
        let cpu = Arc::new(CpuDevice::new());
        let (batch, h) = dead_cpu(&cpu).unwrap();
        dead_frames(batch, &h, "twin")
    };
    let (batch, h) = dead_gpu(&device).unwrap();
    assert_eq!(dead_frames(batch, &h, "gpu"), twin);
}

// ---- several targets: a set per path and a name table with its own place ----

#[derive(Clone, Copy)]
struct Many {
    rows: Stage<Row, D1>,
    dst: Stage<u32, D1>,
    hot: Stage<Row, D1>,
    cold: Stage<u32, D1>,
    names: Stage<u32, D1>,
}

macro_rules! define_many {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Many), Error> {
            gpu_batch!("scatter_many", on(device); in_flight(1); |b| {
                let rows = b.input::<Row>(N);
                let dst = b.input::<u32>(N);
                let hot = b.input::<Row>(N);
                let cold = b.input::<u32>(N);
                let names = b.input::<u32>(N);
                let _scatter = gpu_task!(N, scatter(&mut hot, &mut cold, &mut names), |i| {
                    let r = rows[i];
                    let slot = dst[i];
                    hot[slot] = r;
                    if r.flags % 2u32 == 0u32 {
                        cold[slot] = r.tag;
                    }
                    names[63u32 - i] = slot;
                });
                Many { rows, dst, hot, cold, names }
            })
        }
    };
}

define_many!(many_cpu, CpuDevice);
define_many!(many_gpu, HeadlessDevice);

type ManyOut = (Vec<Row>, Vec<u32>, Vec<u32>);

fn many_frames<D: BatchDevice>(mut batch: Batch<D>, h: &Many, what: &str) -> Vec<ManyOut>
where
    D::Buffer<u32>: 'static,
    D::Buffer<Row>: 'static,
{
    let dst = permutation();
    (0..3)
        .map(|k| {
            let rows = rows_for(k, N);
            let got = block_on(async {
                let mut frame = batch.begin().await?;
                frame.write(&h.rows, &rows)?;
                frame.write(&h.dst, &dst)?;
                frame.write(&h.hot, &[SENTINEL; N as usize])?;
                frame.write(&h.cold, &[7777u32; N as usize])?;
                frame.write(&h.names, &[8888u32; N as usize])?;
                let hot = frame.ticket(&h.hot)?;
                let cold = frame.ticket(&h.cold)?;
                let names = frame.ticket(&h.names)?;
                let _submitted = frame.submit()?;
                Ok::<_, Error>((hot.await?, cold.await?, names.await?))
            })
            .unwrap();
            let mut hot = vec![SENTINEL; N as usize];
            let mut cold = vec![7777u32; N as usize];
            let mut names = vec![8888u32; N as usize];
            for (i, row) in rows.iter().enumerate() {
                let slot = dst[i] as usize;
                hot[slot] = *row;
                if row.flags % 2 == 0 {
                    cold[slot] = row.tag;
                }
                names[N as usize - 1 - i] = slot as u32;
            }
            assert_eq!(got, (hot, cold, names), "{what}: frame {k}");
            got
        })
        .collect()
}

#[test]
fn twin_scatter_stores_several_targets_with_a_set_per_path() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = many_cpu(&device).unwrap();
    let frames = many_frames(batch, &h, "twin");
    let (_, cold, _) = &frames[0];
    assert_eq!(
        cold.iter().filter(|word| **word == 7777).count(),
        N as usize / 2
    );
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_scatter_stores_several_targets_with_a_set_per_path() {
    let device = private_device();
    let twin = {
        let cpu = Arc::new(CpuDevice::new());
        let (batch, h) = many_cpu(&cpu).unwrap();
        many_frames(batch, &h, "twin")
    };
    let (batch, h) = many_gpu(&device).unwrap();
    assert_eq!(many_frames(batch, &h, "gpu"), twin);
}

// ---- the twin's checks: a repeated place and a place out of range ----

fn run_plain_twin(dst: &[u32]) {
    let device = Arc::new(CpuDevice::new());
    let (mut batch, h) = plain_cpu(&device).unwrap();
    let rows = rows_for(0, N);
    plain_frame(&mut batch, &h, &rows, dst, &[SENTINEL; N as usize]).unwrap();
}

/// The panic message of the twin run over `dst`, or a failure when it did not panic. (A `should_panic` test would be
/// reported as `- should panic ... ok`, which the check-cpu guard does not read as a pinned test that passed.)
fn twin_panic_message(dst: &[u32]) -> String {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_plain_twin(dst)));
    let payload = outcome.expect_err("the twin run must panic");
    if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else {
        panic!("a panic payload that is no text")
    }
}

#[test]
fn twin_scatter_panics_on_a_repeated_place() {
    // Rows 2 and 40 share the place 5 (row 46 of the permutation had it too, so the second store to it is row 40's).
    let mut dst = permutation();
    dst[2] = 5;
    dst[40] = 5;
    let message = twin_panic_message(&dst);
    assert!(
        message.contains("scatter stage `_scatter`: slot 5 of target `out` is written twice"),
        "{message}"
    );
}

#[test]
fn twin_scatter_panics_on_a_place_out_of_range() {
    let mut dst = permutation();
    dst[9] = N + 5;
    let message = twin_panic_message(&dst);
    assert!(message.contains("index out of bounds"), "{message}");
}

#[test]
fn twin_scatter_repeating_a_place_on_two_stage_runs_is_fine() {
    // A place repeated in two runs of the stage (two frames) is no repeat: the check is per run.
    let device = Arc::new(CpuDevice::new());
    let (mut batch, h) = plain_cpu(&device).unwrap();
    let dst = permutation();
    for k in 0..2 {
        let rows = rows_for(k, N);
        plain_frame(&mut batch, &h, &rows, &dst, &[SENTINEL; N as usize]).unwrap();
    }
}

// ---- created rows: the count covers rows past the live ones ----

#[derive(Clone, Copy)]
struct Created {
    rows: Stage<Row, D1>,
    live: Stage<u32, D1>,
    made: Stage<u32, D1>,
    out: Stage<Row, D1>,
}

macro_rules! define_created {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Created), Error> {
            gpu_batch!("scatter_created", on(device); in_flight(1); |b| {
                let rows = b.input::<Row>(N);
                let live = b.input::<u32>(1u32);
                let made = b.input::<u32>(1u32);
                let total = gpu_task!(1u32, |i| -> u32 { live[i] + made[i] });
                let out = b.input::<Row>(N);
                let _scatter = gpu_task!(count(total); N, scatter(&mut out), |i| {
                    let mut v: Row = rows[i];
                    if i >= live[0u32] {
                        v = Row { life: 0.5, flags: 7u32, tag: 5000u32 + i };
                    }
                    out[i] = v;
                });
                Created { rows, live, made, out }
            })
        }
    };
}

define_created!(created_cpu, CpuDevice);
define_created!(created_gpu, HeadlessDevice);

fn created_frames<D: BatchDevice>(mut batch: Batch<D>, h: &Created, what: &str) -> Vec<Vec<Row>>
where
    D::Buffer<u32>: 'static,
    D::Buffer<Row>: 'static,
{
    [(10u32, 5u32), (0, 4), (N - 3, 3), (N - 1, 9), (20, 0)]
        .into_iter()
        .enumerate()
        .map(|(k, (live, made))| {
            let rows = rows_for(k as u32, N);
            let got = block_on(async {
                let mut frame = batch.begin().await?;
                frame.write(&h.rows, &rows)?;
                frame.write(&h.live, &[live])?;
                frame.write(&h.made, &[made])?;
                frame.write(&h.out, &[SENTINEL; N as usize])?;
                let out = frame.ticket(&h.out)?;
                let _submitted = frame.submit()?;
                out.await
            })
            .unwrap();
            let run = (live + made).min(N) as usize;
            let want: Vec<Row> = (0..N as usize)
                .map(|i| {
                    if i >= run {
                        SENTINEL
                    } else if i < live as usize {
                        rows[i]
                    } else {
                        Row {
                            life: 0.5,
                            flags: 7,
                            tag: 5000 + i as u32,
                        }
                    }
                })
                .collect();
            assert_eq!(got, want, "{what}: live {live}, made {made}, frame {k}");
            got
        })
        .collect()
}

#[test]
fn twin_scatter_over_live_and_created_rows() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = created_cpu(&device).unwrap();
    let frames = created_frames(batch, &h, "twin");
    assert_eq!(frames[0][12].tag, 5012, "a created row, past the live ones");
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_scatter_over_live_and_created_rows() {
    let device = private_device();
    let twin = {
        let cpu = Arc::new(CpuDevice::new());
        let (batch, h) = created_cpu(&cpu).unwrap();
        created_frames(batch, &h, "twin")
    };
    for launch in LAUNCHES {
        let _launch = Launch::set(launch);
        let (batch, h) = created_gpu(&device).unwrap();
        assert_eq!(
            created_frames(batch, &h, &format!("{launch:?}")),
            twin,
            "{launch:?}"
        );
    }
}

// ---- a resize: the rows' extent follows a parameter, the targets are made again zeroed ----

#[derive(Clone, Copy)]
struct Resized {
    size: Param<u32>,
    rows: Stage<Row, D1>,
    dst: Stage<u32, D1>,
    out: Stage<Row, D1>,
}

macro_rules! define_resized {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Resized), Error> {
            gpu_batch!("scatter_resized", on(device); in_flight(1); |b| {
                let size = b.param::<u32>(8u32);
                let rows = b.input::<Row>(&size);
                let dst = b.input::<u32>(&size);
                let out = b.input::<Row>(&size);
                let _scatter = gpu_task!(&size, scatter(&mut out), |i| {
                    out[dst[i]] = rows[i];
                });
                Resized { size, rows, dst, out }
            })
        }
    };
}

define_resized!(resized_cpu, CpuDevice);
define_resized!(resized_gpu, HeadlessDevice);

fn resized_frames<D: BatchDevice>(mut batch: Batch<D>, h: &Resized, what: &str) -> Vec<Vec<Row>>
where
    D::Buffer<u32>: 'static,
    D::Buffer<Row>: 'static,
{
    [8u32, 20, 5, 8, 33]
        .into_iter()
        .enumerate()
        .map(|(k, size)| {
            let rows = rows_for(k as u32, size);
            // A rotation by one: a permutation for every size.
            let dst: Vec<u32> = (0..size).map(|i| (i + 1) % size).collect();
            let got = block_on(async {
                let mut frame = batch.begin().await?;
                frame.set(&h.size, size)?;
                frame.write(&h.rows, &rows)?;
                frame.write(&h.dst, &dst)?;
                let out = frame.ticket(&h.out)?;
                let _submitted = frame.submit()?;
                out.await
            })
            .unwrap();
            let before = vec![
                Row {
                    life: 0.0,
                    flags: 0,
                    tag: 0
                };
                size as usize
            ];
            // The rotation names every slot, so what the target held before (zeros after a resize, an earlier frame's
            // rows where the size is back to the sealed one) does not show.
            let want = scattered(&rows, &dst, &before, |_| true);
            assert_eq!(
                got.len(),
                size as usize,
                "{what}: the target follows the size, frame {k}"
            );
            assert_eq!(got, want, "{what}: size {size}, frame {k}");
            got
        })
        .collect()
}

#[test]
fn twin_scatter_extent_follows_a_parameter() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = resized_cpu(&device).unwrap();
    resized_frames(batch, &h, "twin");
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_scatter_extent_follows_a_parameter() {
    let device = private_device();
    let twin = {
        let cpu = Arc::new(CpuDevice::new());
        let (batch, h) = resized_cpu(&cpu).unwrap();
        resized_frames(batch, &h, "twin")
    };
    let (batch, h) = resized_gpu(&device).unwrap();
    assert_eq!(resized_frames(batch, &h, "gpu"), twin);
}

// ---- an extent that moves while every buffer keeps its size ----

const CAP: u32 = 16;

#[derive(Clone, Copy)]
struct Narrowed {
    live: Param<u32>,
    rows: Stage<Row, D1>,
    dst: Stage<u32, D1>,
    out: Stage<Row, D1>,
}

macro_rules! define_narrowed {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Narrowed), Error> {
            gpu_batch!("scatter_narrowed", on(device); in_flight(1); |b| {
                let live = b.param::<u32>(CAP);
                let rows = b.input::<Row>(CAP);
                let dst = b.input::<u32>(CAP);
                let out = b.input::<Row>(CAP);
                let _scatter = gpu_task!(&live, scatter(&mut out), |i| {
                    out[dst[i]] = rows[i];
                });
                Narrowed { live, rows, dst, out }
            })
        }
    };
}

define_narrowed!(narrowed_cpu, CpuDevice);
define_narrowed!(narrowed_gpu, HeadlessDevice);

/// The extent is the parameter and no buffer changes size, so only the stage's extent moves between frames: the
/// stage must run over the first `live` rows, and the slots it does not reach keep what the frame wrote.
fn narrowed_frames<D: BatchDevice>(mut batch: Batch<D>, h: &Narrowed, what: &str) -> Vec<Vec<Row>>
where
    D::Buffer<u32>: 'static,
    D::Buffer<Row>: 'static,
{
    [CAP, 5, CAP, 11, 5]
        .into_iter()
        .enumerate()
        .map(|(k, live)| {
            let rows = rows_for(k as u32, CAP);
            let dst: Vec<u32> = (0..CAP).map(|i| (i + 1) % live.max(1)).collect();
            let before = vec![SENTINEL; CAP as usize];
            let got = block_on(async {
                let mut frame = batch.begin().await?;
                frame.set(&h.live, live)?;
                frame.write(&h.rows, &rows)?;
                frame.write(&h.dst, &dst)?;
                frame.write(&h.out, &before)?;
                let out = frame.ticket(&h.out)?;
                let _submitted = frame.submit()?;
                out.await
            })
            .unwrap();
            let want = scattered(&rows, &dst, &before, |i| (i as u32) < live);
            assert_eq!(got, want, "{what}: live {live}, frame {k}");
            got
        })
        .collect()
}

#[test]
fn twin_scatter_extent_moves_while_every_buffer_keeps_its_size() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = narrowed_cpu(&device).unwrap();
    narrowed_frames(batch, &h, "twin");
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_scatter_extent_moves_while_every_buffer_keeps_its_size() {
    let device = private_device();
    let twin = {
        let cpu = Arc::new(CpuDevice::new());
        let (batch, h) = narrowed_cpu(&cpu).unwrap();
        narrowed_frames(batch, &h, "twin")
    };
    let (batch, h) = narrowed_gpu(&device).unwrap();
    assert_eq!(narrowed_frames(batch, &h, "gpu"), twin);
}

// ---- an enum element and a scalar target ----

#[derive(Clone, Copy)]
struct Enumerated {
    src: Stage<Phase, D1>,
    dst: Stage<u32, D1>,
    out: Stage<Phase, D1>,
}

macro_rules! define_enumerated {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Enumerated), Error> {
            gpu_batch!("scatter_enum", on(device); in_flight(1); |b| {
                let src = b.input::<Phase>(N);
                let dst = b.input::<u32>(N);
                let out = b.input::<Phase>(N);
                let _scatter = gpu_task!(N, scatter(&mut out), |i| {
                    out[dst[i]] = match src[i] {
                        Phase::Idle => Phase::Run,
                        Phase::Run => Phase::Done,
                        Phase::Done => Phase::Idle,
                    };
                });
                Enumerated { src, dst, out }
            })
        }
    };
}

define_enumerated!(enumerated_cpu, CpuDevice);
define_enumerated!(enumerated_gpu, HeadlessDevice);

fn enumerated_frames<D: BatchDevice>(mut batch: Batch<D>, h: &Enumerated, what: &str) -> Vec<Phase>
where
    D::Buffer<u32>: 'static,
{
    let all = [Phase::Idle, Phase::Run, Phase::Done];
    let src: Vec<Phase> = (0..N).map(|i| all[((i * 5 + 1) % 3) as usize]).collect();
    let dst = permutation();
    let got = block_on(async {
        let mut frame = batch.begin().await?;
        frame.write(&h.src, &src)?;
        frame.write(&h.dst, &dst)?;
        let out = frame.ticket(&h.out)?;
        let _submitted = frame.submit()?;
        out.await
    })
    .unwrap();
    let next = |phase: Phase| match phase {
        Phase::Idle => Phase::Run,
        Phase::Run => Phase::Done,
        Phase::Done => Phase::Idle,
    };
    let mut want = vec![Phase::Idle; N as usize];
    for (i, phase) in src.iter().enumerate() {
        want[dst[i] as usize] = next(*phase);
    }
    assert_eq!(got, want, "{what}");
    got
}

#[test]
fn twin_scatter_stores_an_enum_element() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = enumerated_cpu(&device).unwrap();
    enumerated_frames(batch, &h, "twin");
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_scatter_stores_an_enum_element() {
    let device = private_device();
    let (batch, h) = enumerated_gpu(&device).unwrap();
    let got = enumerated_frames(batch, &h, "gpu");
    let (cpu_batch, cpu_h) = enumerated_cpu(&Arc::new(CpuDevice::new())).unwrap();
    assert_eq!(got, enumerated_frames(cpu_batch, &cpu_h, "twin"));
}
