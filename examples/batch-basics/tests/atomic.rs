//! The atomic stage (m14.s2sa): `gpu_task!(count(n); ROWS, atomic(&mut counts), |i| { atomic_add(&mut counts[key[i]], 1u32); })`
//! runs over the rows of its extent and accumulates into one or several handles of the block (the targets) at places the
//! rows compute: `atomic_add` on `u32` and `i32` targets, `atomic_min` and `atomic_max` on `u32` targets, at any `u32`
//! index, any number of times on any path. A target persists from frame to frame like any buffer, so the stage adds to
//! what the buffer holds and the author clears it (an in-place stage, or a write of the host).
//!
//! The twin equals the GPU over the whole target when every cell receives one kind of operation and no index, operand or
//! condition of an atomic depends on a bound old value; every test below stays inside that rule except the one that binds
//! the old value, which asserts totals only. Every value is an integer, so a result equals the host model bit for bit.
//! The `twin_` tests run on the CPU twin; the `gpu_` ones are `#[ignore]`d like the example's other GPU tests, print the
//! adapter and run in full verification on the adapter the environment selects. Where a stage is counted, the `gpu_` test
//! runs under both counted launches (D277).

// An atomic stage has the value `()`, and its `let` names the stage.
#![allow(
    clippy::let_unit_value,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use batch_basics::app::block_on;
use gust::batch::{Batch, BatchDevice, CountedLaunch, D1, Error, Param, Stage, set_counted_launch};
use gust::cpu::CpuDevice;
use gust::{gpu_batch, gpu_task};
use gust_wgpu::HeadlessDevice;

const LAUNCHES: [CountedLaunch; 2] = [CountedLaunch::Indirect, CountedLaunch::Bounded];
/// The workgroup of every stage here (the default of a 1-D stage).
const WG: u32 = 256;

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

/// A key below `keys` for row `i`: a hash of the row, so that the keys repeat in no pattern.
fn key_of(i: u32, keys: u32) -> u32 {
    (i.wrapping_mul(2_654_435_761) ^ (i >> 5)) % keys
}

fn keys_for(rows: u32, keys: u32, seed: u32) -> Vec<u32> {
    (0..rows).map(|i| key_of(i + seed * 7919, keys)).collect()
}

/// The host histogram of `keys` over `size` cells.
fn histogram(keys: &[u32], size: u32) -> Vec<u32> {
    let mut counts = vec![0_u32; size as usize];
    for &key in keys {
        counts[key as usize] += 1;
    }
    counts
}

// ---- the histogram: clear, then count ----

#[derive(Clone, Copy)]
struct Hist {
    key: Stage<u32, D1>,
    counts: Stage<u32, D1>,
}

macro_rules! define_hist {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>, rows: u32, keys: u32) -> Result<(Batch<$device>, Hist), Error> {
            gpu_batch!("atomic_hist", on(device); in_flight(1); |b| {
                let key = b.input::<u32>(rows);
                let counts = b.input::<u32>(keys);
                let _clear = gpu_task!(&mut counts, |k| { counts[k] = 0u32; });
                let _count = gpu_task!(rows, atomic(&mut counts), |i| {
                    atomic_add(&mut counts[key[i]], 1u32);
                });
                Hist { key, counts }
            })
        }
    };
}

/// The same count with no clear: the counts of every frame are added to those of the frames before.
macro_rules! define_hist_keep {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>, rows: u32, keys: u32) -> Result<(Batch<$device>, Hist), Error> {
            gpu_batch!("atomic_hist_keep", on(device); in_flight(1); |b| {
                let key = b.input::<u32>(rows);
                let counts = b.input::<u32>(keys);
                let _count = gpu_task!(rows, atomic(&mut counts), |i| {
                    atomic_add(&mut counts[key[i]], 1u32);
                });
                Hist { key, counts }
            })
        }
    };
}

define_hist!(hist_cpu, CpuDevice);
define_hist!(hist_gpu, HeadlessDevice);
define_hist_keep!(hist_keep_cpu, CpuDevice);
define_hist_keep!(hist_keep_gpu, HeadlessDevice);

/// A frame: the keys written, the counts read back.
fn hist_frame<D: BatchDevice>(batch: &mut Batch<D>, h: &Hist, keys: &[u32]) -> Vec<u32>
where
    D::Buffer<u32>: 'static,
{
    block_on(async {
        let mut frame = batch.begin().await?;
        frame.write(&h.key, keys)?;
        let counts = frame.ticket(&h.counts)?;
        let _submitted = frame.submit()?;
        counts.await
    })
    .unwrap()
}

/// Three frames over `rows` rows and `keys` cells: each frame's counts are its own histogram (the stage clears first).
fn hist_frames<D: BatchDevice>(
    mut batch: Batch<D>,
    h: &Hist,
    rows: u32,
    keys: u32,
    what: &str,
) -> Vec<Vec<u32>>
where
    D::Buffer<u32>: 'static,
{
    (0..3)
        .map(|seed| {
            let data = keys_for(rows, keys, seed);
            let got = hist_frame(&mut batch, h, &data);
            assert_eq!(
                got,
                histogram(&data, keys),
                "{what}: keys {keys}, frame {seed}"
            );
            assert_eq!(
                got.iter().map(|&c| u64::from(c)).sum::<u64>(),
                u64::from(rows),
                "{what}: the counts add up to the rows"
            );
            got
        })
        .collect()
}

/// K = 1 (every row contends for one cell), 7, 4096 and a K near the number of rows (most cells see one row).
const HIST_CASES: [(u32, u32); 4] = [(8192, 1), (8192, 7), (8192, 4096), (4096, 4000)];

#[test]
fn twin_atomic_histogram_equals_the_host() {
    for (rows, keys) in HIST_CASES {
        let device = Arc::new(CpuDevice::new());
        let (batch, h) = hist_cpu(&device, rows, keys).unwrap();
        hist_frames(batch, &h, rows, keys, "twin");
    }
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_atomic_histogram_equals_the_host_and_the_twin() {
    let device = private_device();
    for (rows, keys) in HIST_CASES {
        let twin = Arc::new(CpuDevice::new());
        let (batch, h) = hist_cpu(&twin, rows, keys).unwrap();
        let on_twin = hist_frames(batch, &h, rows, keys, "twin");
        let (batch, h) = hist_gpu(&device, rows, keys).unwrap();
        assert_eq!(hist_frames(batch, &h, rows, keys, "gpu"), on_twin);
    }
}

#[test]
fn twin_atomic_over_many_rows_is_whole() {
    // More rows than any chunk a parallel twin could cut: the twin of an atomic stage stays equal to the host.
    let (rows, keys) = (300_000, 7);
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = hist_cpu(&device, rows, keys).unwrap();
    hist_frames(batch, &h, rows, keys, "twin");
}

// ---- counts, scan, rank, scatter: the stages of a counting sort ----

const SORT_ROWS: u32 = 4096;
const SORT_KEYS: u32 = 64;

#[derive(Clone, Copy)]
struct Sort {
    key: Stage<u32, D1>,
    val: Stage<u32, D1>,
    counts: Stage<u32, D1>,
    starts: Stage<u32, D1>,
    dst: Stage<u32, D1>,
    sorted: Stage<u32, D1>,
    twice: Stage<u32, D1>,
}

macro_rules! define_sort {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Sort), Error> {
            gpu_batch!("atomic_sort", on(device); in_flight(1); |b| {
                let key = b.input::<u32>(SORT_ROWS);
                let val = b.input::<u32>(SORT_ROWS);
                let counts = b.input::<u32>(SORT_KEYS);
                let sorted = b.input::<u32>(SORT_ROWS);
                let _clear = gpu_task!(&mut counts, |k| { counts[k] = 0u32; });
                let _count = gpu_task!(SORT_ROWS, atomic(&mut counts), |i| {
                    atomic_add(&mut counts[key[i]], 1u32);
                });
                // A stage of the user's reads the counts as plain words after the atomic stage wrote them.
                let twice = gpu_task!(SORT_KEYS, |k| -> u32 { counts[k] * 2u32 });
                let starts = gust::scan_exclusive(&counts);
                let dst = gpu_task!(SORT_ROWS, |i| -> u32 {
                    let k = key[i];
                    let mut rank = 0u32;
                    for j in 0u32..i {
                        if key[j] == k {
                            rank += 1u32;
                        }
                    }
                    starts[k] + rank
                });
                let _place = gpu_task!(SORT_ROWS, scatter(&mut sorted), |i| {
                    sorted[dst[i]] = val[i];
                });
                Sort { key, val, counts, starts, dst, sorted, twice }
            })
        }
    };
}

define_sort!(sort_cpu, CpuDevice);
define_sort!(sort_gpu, HeadlessDevice);

/// Counts, starts, destinations and the sorted copy of one frame.
type Sorted = (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>);

fn sort_frame<D: BatchDevice>(batch: &mut Batch<D>, h: &Sort, key: &[u32], val: &[u32]) -> Sorted
where
    D::Buffer<u32>: 'static,
{
    block_on(async {
        let mut frame = batch.begin().await?;
        frame.write(&h.key, key)?;
        frame.write(&h.val, val)?;
        let counts = frame.ticket(&h.counts)?;
        let starts = frame.ticket(&h.starts)?;
        let dst = frame.ticket(&h.dst)?;
        let sorted = frame.ticket(&h.sorted)?;
        let twice = frame.ticket(&h.twice)?;
        let _submitted = frame.submit()?;
        Ok::<_, Error>((
            counts.await?,
            starts.await?,
            dst.await?,
            sorted.await?,
            twice.await?,
        ))
    })
    .unwrap()
}

/// The host stable counting sort of `val` by `key`.
fn host_sort(key: &[u32], val: &[u32]) -> Sorted {
    let counts = histogram(key, SORT_KEYS);
    let mut starts = vec![0_u32; SORT_KEYS as usize];
    for k in 1..SORT_KEYS as usize {
        starts[k] = starts[k - 1] + counts[k - 1];
    }
    let mut next = starts.clone();
    let mut dst = vec![0_u32; key.len()];
    let mut sorted = vec![0_u32; key.len()];
    for (i, &k) in key.iter().enumerate() {
        dst[i] = next[k as usize];
        sorted[dst[i] as usize] = val[i];
        next[k as usize] += 1;
    }
    let twice = counts.iter().map(|c| c * 2).collect();
    (counts, starts, dst, sorted, twice)
}

fn sort_frames<D: BatchDevice>(mut batch: Batch<D>, h: &Sort, what: &str) -> Vec<Sorted>
where
    D::Buffer<u32>: 'static,
{
    (0..3)
        .map(|seed| {
            let key = keys_for(SORT_ROWS, SORT_KEYS, seed);
            let val: Vec<u32> = (0..SORT_ROWS).map(|i| i * 3 + seed).collect();
            let got = sort_frame(&mut batch, h, &key, &val);
            assert_eq!(got, host_sort(&key, &val), "{what}: frame {seed}");
            got
        })
        .collect()
}

#[test]
fn twin_counts_scan_rank_and_scatter_sort_like_the_host() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = sort_cpu(&device).unwrap();
    sort_frames(batch, &h, "twin");
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_counts_scan_rank_and_scatter_sort_like_the_host() {
    let device = private_device();
    let twin = Arc::new(CpuDevice::new());
    let (batch, h) = sort_cpu(&twin).unwrap();
    let on_twin = sort_frames(batch, &h, "twin");
    let (batch, h) = sort_gpu(&device).unwrap();
    assert_eq!(sort_frames(batch, &h, "gpu"), on_twin);
}

// ---- a counted range: the rows past the count add nothing ----

const CAP: u32 = 1000;

#[derive(Clone, Copy)]
struct Counted {
    live: Stage<u32, D1>,
    key: Stage<u32, D1>,
    counts: Stage<u32, D1>,
}

macro_rules! define_counted {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Counted), Error> {
            gpu_batch!("atomic_counted", on(device); in_flight(1); |b| {
                let live = b.input::<u32>(1u32);
                let key = b.input::<u32>(CAP);
                let counts = b.input::<u32>(16u32);
                let _clear = gpu_task!(&mut counts, |k| { counts[k] = 0u32; });
                let _count = gpu_task!(count(live); CAP, atomic(&mut counts), |i| {
                    atomic_add(&mut counts[key[i]], 1u32);
                });
                Counted { live, key, counts }
            })
        }
    };
}

define_counted!(counted_cpu, CpuDevice);
define_counted!(counted_gpu, HeadlessDevice);

/// 0, 1, wg-1, wg, wg+1, cap-1, cap, cap+1, `u32::MAX`, then 0 again.
fn live_counts() -> [u32; 10] {
    [0, 1, WG - 1, WG, WG + 1, CAP - 1, CAP, CAP + 1, u32::MAX, 0]
}

fn counted_frames<D: BatchDevice>(mut batch: Batch<D>, h: &Counted, what: &str) -> Vec<Vec<u32>>
where
    D::Buffer<u32>: 'static,
{
    live_counts()
        .into_iter()
        .enumerate()
        .map(|(k, live)| {
            let keys = keys_for(CAP, 16, k as u32);
            let got = block_on(async {
                let mut frame = batch.begin().await?;
                frame.write(&h.live, &[live])?;
                frame.write(&h.key, &keys)?;
                let counts = frame.ticket(&h.counts)?;
                let _submitted = frame.submit()?;
                counts.await
            })
            .unwrap();
            let rows = live.min(CAP) as usize;
            assert_eq!(got, histogram(&keys[..rows], 16), "{what}: count {live}");
            got
        })
        .collect()
}

#[test]
fn twin_counted_atomic_runs_the_live_rows_only() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = counted_cpu(&device).unwrap();
    counted_frames(batch, &h, "twin");
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_counted_atomic_runs_the_live_rows_only() {
    let device = private_device();
    let twin = Arc::new(CpuDevice::new());
    let (batch, h) = counted_cpu(&twin).unwrap();
    let on_twin = counted_frames(batch, &h, "twin");
    for launch in LAUNCHES {
        let _launch = Launch::set(launch);
        let (batch, h) = counted_gpu(&device).unwrap();
        assert_eq!(
            counted_frames(batch, &h, &format!("gpu {launch:?}")),
            on_twin
        );
    }
}

// ---- the clear: a target persists, and the author clears it ----

fn clear_frames<D: BatchDevice>(
    mut cleared: Batch<D>,
    ch: &Hist,
    mut kept: Batch<D>,
    kh: &Hist,
    what: &str,
) where
    D::Buffer<u32>: 'static,
{
    let (rows, keys) = (2048, 9);
    let first = keys_for(rows, keys, 11);
    let second = keys_for(rows, keys, 12);
    let each = [histogram(&first, keys), histogram(&second, keys)];
    let both: Vec<u32> = each[0].iter().zip(&each[1]).map(|(a, b)| a + b).collect();
    // With the zero stage the second frame's counts are its own histogram.
    assert_eq!(
        hist_frame(&mut cleared, ch, &first),
        each[0],
        "{what}: cleared, frame 0"
    );
    assert_eq!(
        hist_frame(&mut cleared, ch, &second),
        each[1],
        "{what}: cleared, frame 1"
    );
    // Without it they are both frames' sum.
    assert_eq!(
        hist_frame(&mut kept, kh, &first),
        each[0],
        "{what}: kept, frame 0"
    );
    assert_eq!(
        hist_frame(&mut kept, kh, &second),
        both,
        "{what}: kept, frame 1"
    );
}

#[test]
fn twin_a_target_persists_across_frames_and_the_zero_stage_clears_it() {
    let device = Arc::new(CpuDevice::new());
    let (cleared, ch) = hist_cpu(&device, 2048, 9).unwrap();
    let (kept, kh) = hist_keep_cpu(&device, 2048, 9).unwrap();
    clear_frames(cleared, &ch, kept, &kh, "twin");
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_target_persists_across_frames_and_the_zero_stage_clears_it() {
    let device = private_device();
    let (cleared, ch) = hist_gpu(&device, 2048, 9).unwrap();
    let (kept, kh) = hist_keep_gpu(&device, 2048, 9).unwrap();
    clear_frames(cleared, &ch, kept, &kh, "gpu");
}

// ---- min, max, wrapping adds and a signed target ----

#[derive(Clone, Copy)]
struct Extremes {
    key: Stage<u32, D1>,
    val: Stage<u32, D1>,
    delta: Stage<i32, D1>,
    hi: Stage<u32, D1>,
    lo: Stage<u32, D1>,
    bal: Stage<i32, D1>,
    wrap: Stage<u32, D1>,
}

const EXTREME_ROWS: u32 = 6000;
const EXTREME_KEYS: u32 = 33;

macro_rules! define_extremes {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Extremes), Error> {
            gpu_batch!("atomic_extremes", on(device); in_flight(1); |b| {
                let key = b.input::<u32>(EXTREME_ROWS);
                let val = b.input::<u32>(EXTREME_ROWS);
                let delta = b.input::<i32>(EXTREME_ROWS);
                let hi = b.input::<u32>(EXTREME_KEYS);
                let lo = b.input::<u32>(EXTREME_KEYS);
                let bal = b.input::<i32>(EXTREME_KEYS);
                let wrap = b.input::<u32>(EXTREME_KEYS);
                let _all = gpu_task!(EXTREME_ROWS, atomic(&mut hi, &mut lo, &mut bal, &mut wrap), |i| {
                    atomic_max(&mut hi[key[i]], val[i]);
                    atomic_min(&mut lo[key[i]], val[i]);
                    atomic_add(&mut bal[key[i]], delta[i]);
                    atomic_add(&mut wrap[key[i]], val[i]);
                });
                Extremes { key, val, delta, hi, lo, bal, wrap }
            })
        }
    };
}

define_extremes!(extremes_cpu, CpuDevice);
define_extremes!(extremes_gpu, HeadlessDevice);

type ExtremeBuffers = (Vec<u32>, Vec<u32>, Vec<i32>, Vec<u32>);

fn extremes_frames<D: BatchDevice>(
    mut batch: Batch<D>,
    h: &Extremes,
    what: &str,
) -> Vec<ExtremeBuffers>
where
    D::Buffer<u32>: 'static,
    D::Buffer<i32>: 'static,
{
    (0..3)
        .map(|seed| {
            let key = keys_for(EXTREME_ROWS, EXTREME_KEYS, seed);
            // Large values: the sums of `val` wrap modulo 2^32, and the deltas wrap as `i32`.
            let val: Vec<u32> = (0..EXTREME_ROWS)
                .map(|i| {
                    i.wrapping_mul(1_000_003)
                        .wrapping_add(seed.wrapping_mul(0x9E37_79B9))
                        ^ 0x8000_0000
                })
                .collect();
            let delta: Vec<i32> = (0..EXTREME_ROWS)
                .map(|i| {
                    (i as i32)
                        .wrapping_mul(1_000_003)
                        .wrapping_add(i32::MAX / 2 + seed as i32)
                })
                .collect();
            let (mut hi, mut lo) = (
                vec![0_u32; EXTREME_KEYS as usize],
                vec![u32::MAX; EXTREME_KEYS as usize],
            );
            let (mut bal, mut wrap) = (
                vec![0_i32; EXTREME_KEYS as usize],
                vec![0_u32; EXTREME_KEYS as usize],
            );
            for i in 0..EXTREME_ROWS as usize {
                let k = key[i] as usize;
                hi[k] = hi[k].max(val[i]);
                lo[k] = lo[k].min(val[i]);
                bal[k] = bal[k].wrapping_add(delta[i]);
                wrap[k] = wrap[k].wrapping_add(val[i]);
            }
            let seed_lo = vec![u32::MAX; EXTREME_KEYS as usize];
            let zeros = vec![0_u32; EXTREME_KEYS as usize];
            let signed_zeros = vec![0_i32; EXTREME_KEYS as usize];
            let got = block_on(async {
                let mut frame = batch.begin().await?;
                frame.write(&h.key, &key)?;
                frame.write(&h.val, &val)?;
                frame.write(&h.delta, &delta)?;
                // The host seeds the targets: the neutral element of each operation.
                frame.write(&h.hi, &zeros)?;
                frame.write(&h.lo, &seed_lo)?;
                frame.write(&h.bal, &signed_zeros)?;
                frame.write(&h.wrap, &zeros)?;
                let hi = frame.ticket(&h.hi)?;
                let lo = frame.ticket(&h.lo)?;
                let bal = frame.ticket(&h.bal)?;
                let wrap = frame.ticket(&h.wrap)?;
                let _submitted = frame.submit()?;
                Ok::<_, Error>((hi.await?, lo.await?, bal.await?, wrap.await?))
            })
            .unwrap();
            assert_eq!(got, (hi, lo, bal, wrap), "{what}: frame {seed}");
            got
        })
        .collect()
}

#[test]
fn twin_min_max_and_wrapping_adds_equal_the_host() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = extremes_cpu(&device).unwrap();
    extremes_frames(batch, &h, "twin");
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_min_max_and_wrapping_adds_equal_the_host_and_the_twin() {
    let device = private_device();
    let twin = Arc::new(CpuDevice::new());
    let (batch, h) = extremes_cpu(&twin).unwrap();
    let on_twin = extremes_frames(batch, &h, "twin");
    let (batch, h) = extremes_gpu(&device).unwrap();
    assert_eq!(extremes_frames(batch, &h, "gpu"), on_twin);
}

// ---- two targets and the old value: totals only ----

#[derive(Clone, Copy)]
struct Pair {
    key: Stage<u32, D1>,
    val: Stage<u32, D1>,
    counts: Stage<u32, D1>,
    sums: Stage<u32, D1>,
    seen: Stage<u32, D1>,
    distinct: Stage<u32, D1>,
}

const PAIR_ROWS: u32 = 5000;
const PAIR_KEYS: u32 = 211;

macro_rules! define_pair {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Pair), Error> {
            gpu_batch!("atomic_pair", on(device); in_flight(1); |b| {
                let key = b.input::<u32>(PAIR_ROWS);
                let val = b.input::<u32>(PAIR_ROWS);
                let counts = b.input::<u32>(PAIR_KEYS);
                let sums = b.input::<u32>(PAIR_KEYS);
                let seen = b.input::<u32>(PAIR_KEYS);
                let distinct = b.input::<u32>(1u32);
                let _clear = gpu_task!(&mut counts, |k| { counts[k] = 0u32; });
                let _clear_sums = gpu_task!(&mut sums, |k| { sums[k] = 0u32; });
                let _clear_seen = gpu_task!(&mut seen, |k| { seen[k] = 0u32; });
                let _clear_distinct = gpu_task!(&mut distinct, |k| { distinct[k] = 0u32; });
                let _both = gpu_task!(PAIR_ROWS, atomic(&mut counts, &mut sums, &mut seen, &mut distinct), |i| {
                    atomic_add(&mut counts[key[i]], 1u32);
                    atomic_add(&mut sums[key[i]], val[i]);
                    let old = atomic_add(&mut seen[key[i]], 1u32);
                    if old == 0u32 {
                        atomic_add(&mut distinct[0u32], 1u32);
                    }
                });
                Pair { key, val, counts, sums, seen, distinct }
            })
        }
    };
}

define_pair!(pair_cpu, CpuDevice);
define_pair!(pair_gpu, HeadlessDevice);

fn pair_frames<D: BatchDevice>(mut batch: Batch<D>, h: &Pair, what: &str)
where
    D::Buffer<u32>: 'static,
{
    for seed in 0..3 {
        // Not every key is present: the keys are drawn from the first 150 of the cells in frame 1.
        let range = if seed == 1 { 150 } else { PAIR_KEYS };
        let key = keys_for(PAIR_ROWS, range, seed);
        let val: Vec<u32> = (0..PAIR_ROWS).map(|i| 0x4000_0000 + i * 977).collect();
        let counts = histogram(&key, PAIR_KEYS);
        let mut sums = vec![0_u32; PAIR_KEYS as usize];
        for (i, &k) in key.iter().enumerate() {
            sums[k as usize] = sums[k as usize].wrapping_add(val[i]);
        }
        let distinct = counts.iter().filter(|&&c| c > 0).count() as u32;
        let (got_counts, got_sums, got_seen, got_distinct) = block_on(async {
            let mut frame = batch.begin().await?;
            frame.write(&h.key, &key)?;
            frame.write(&h.val, &val)?;
            let counts = frame.ticket(&h.counts)?;
            let sums = frame.ticket(&h.sums)?;
            let seen = frame.ticket(&h.seen)?;
            let distinct = frame.ticket(&h.distinct)?;
            let _submitted = frame.submit()?;
            Ok::<_, Error>((counts.await?, sums.await?, seen.await?, distinct.await?))
        })
        .unwrap();
        // Totals only: which row saw `old == 0` is the schedule's, so the per-cell counts, the sums and the distinct
        // total are asserted, and no value that `old` steered per row.
        assert_eq!(got_counts, counts, "{what}: counts, frame {seed}");
        assert_eq!(got_sums, sums, "{what}: sums, frame {seed}");
        assert_eq!(
            got_seen, counts,
            "{what}: the first add's per-cell total, frame {seed}"
        );
        assert_eq!(
            got_distinct,
            vec![distinct],
            "{what}: distinct keys, frame {seed}"
        );
    }
}

#[test]
fn twin_two_targets_and_the_old_value_give_the_host_totals() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = pair_cpu(&device).unwrap();
    pair_frames(batch, &h, "twin");
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_two_targets_and_the_old_value_give_the_host_totals() {
    let device = private_device();
    let (batch, h) = pair_gpu(&device).unwrap();
    pair_frames(batch, &h, "gpu");
}

// ---- a resize: the stage follows its rows, the targets are made again zeroed ----

#[derive(Clone, Copy)]
struct Resized {
    rows: Param<u32>,
    keys: Param<u32>,
    key: Stage<u32, D1>,
    counts: Stage<u32, D1>,
}

macro_rules! define_resized {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Resized), Error> {
            gpu_batch!("atomic_resized", on(device); in_flight(1); |b| {
                let rows = b.param::<u32>(64u32);
                let keys = b.param::<u32>(8u32);
                let key = b.input::<u32>(&rows);
                let counts = b.input::<u32>(&keys);
                let _count = gpu_task!(&rows, atomic(&mut counts), |i| {
                    atomic_add(&mut counts[key[i]], 1u32);
                });
                Resized { rows, keys, key, counts }
            })
        }
    };
}

define_resized!(resized_cpu, CpuDevice);
define_resized!(resized_gpu, HeadlessDevice);

/// The sizes of each frame: the rows and the cells. The stage runs over the rows, grown and shrunk around the cells, and
/// once over no row at all (no dispatch).
const RESIZES: [(u32, u32); 8] = [
    (64, 8),
    (300, 20),
    (5, 20),
    (700, 5),
    (0, 5),
    (40, 5),
    (64, 8),
    (1, 1),
];

fn resized_frames<D: BatchDevice>(mut batch: Batch<D>, h: &Resized, what: &str) -> Vec<Vec<u32>>
where
    D::Buffer<u32>: 'static,
{
    let mut held: Vec<u32> = vec![0; 8];
    let mut size = 8;
    RESIZES
        .into_iter()
        .enumerate()
        .map(|(k, (rows, keys))| {
            let key = keys_for(rows, keys, k as u32);
            let got = block_on(async {
                let mut frame = batch.begin().await?;
                frame.set(&h.rows, rows)?;
                frame.set(&h.keys, keys)?;
                frame.write(&h.key, &key)?;
                let counts = frame.ticket(&h.counts)?;
                let _submitted = frame.submit()?;
                counts.await
            })
            .unwrap();
            // A resize makes a new zeroed target; the same size keeps what the buffer held and adds to it.
            if keys != size {
                held = vec![0; keys as usize];
                size = keys;
            }
            for (cell, add) in held.iter_mut().zip(histogram(&key, keys)) {
                *cell += add;
            }
            assert_eq!(got, held, "{what}: rows {rows}, keys {keys}, frame {k}");
            got
        })
        .collect()
}

#[test]
fn twin_the_atomic_extent_follows_the_rows_not_the_target() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = resized_cpu(&device).unwrap();
    resized_frames(batch, &h, "twin");
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_the_atomic_extent_follows_the_rows_not_the_target() {
    let device = private_device();
    let twin = Arc::new(CpuDevice::new());
    let (batch, h) = resized_cpu(&twin).unwrap();
    let on_twin = resized_frames(batch, &h, "twin");
    let (batch, h) = resized_gpu(&device).unwrap();
    assert_eq!(resized_frames(batch, &h, "gpu"), on_twin);
}

// ---- a key out of range: the twin panics, the GPU is open (KM-11) ----

fn twin_panic_message(rows: u32, keys: u32, bad: u32) -> String {
    let outcome = std::panic::catch_unwind(|| {
        let device = Arc::new(CpuDevice::new());
        let (mut batch, h) = hist_cpu(&device, rows, keys).unwrap();
        let mut data = keys_for(rows, keys, 0);
        data[(rows / 2) as usize] = bad;
        hist_frame(&mut batch, &h, &data);
    });
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
fn twin_panics_on_a_key_out_of_range() {
    // A cell past the end of the target: the twin's slice index panics. (A `should_panic` test would be reported as
    // `- should panic ... ok`, which the check-cpu guard does not read as a pinned test that passed.)
    let message = twin_panic_message(64, 4, 4);
    assert!(message.contains("index out of bounds"), "{message}");
}

// ---- a launch above the 65535-workgroup limit: the stage kernel splits, and the counts stay whole ----

#[test]
fn twin_atomic_over_more_rows_than_the_workgroup_limit() {
    let rows = 65_535 * WG + 1000;
    let keys = 7;
    let device = Arc::new(CpuDevice::new());
    let (mut batch, h) = hist_cpu(&device, rows, keys).unwrap();
    let data = keys_for(rows, keys, 0);
    assert_eq!(hist_frame(&mut batch, &h, &data), histogram(&data, keys));
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_atomic_over_more_rows_than_the_workgroup_limit() {
    // 65535 workgroups of WG rows is the most one dispatch dimension holds: the stage's kernel is launched in pieces, and
    // the pieces add into one counts buffer, so the whole buffer equals the host's.
    let rows = 65_535 * WG + 1000;
    let keys = 7;
    let device = private_device();
    let (mut batch, h) = hist_gpu(&device, rows, keys).unwrap();
    let data = keys_for(rows, keys, 0);
    let want = histogram(&data, keys);
    assert_eq!(hist_frame(&mut batch, &h, &data), want);
    // A second frame: the same counts again (the stage clears first), through the same pieces.
    assert_eq!(hist_frame(&mut batch, &h, &data), want);
}

/// A target made by an op is refused when the block seals, as a stage that writes an op's output is (the scatter suite has
/// the same test for a scatter target).
#[test]
fn twin_atomic_refuses_an_op_output_as_a_target() {
    let device = Arc::new(CpuDevice::new());
    let result = gpu_batch!("atomic_overwrite", on(&device); in_flight(1); |b| {
        let key = b.input::<u32>(SORT_ROWS);
        let counts = b.input::<u32>(SORT_KEYS);
        let starts = gust::scan_exclusive(&counts);
        let _count = gpu_task!(SORT_ROWS, atomic(&mut starts), |i| {
            atomic_add(&mut starts[key[i]], 1u32);
        });
        key
    });
    match result {
        Err(Error::Overwrite {
            stage, op, buffer, ..
        }) => {
            assert_eq!(
                (stage.as_str(), op.as_str(), buffer.as_str()),
                ("_count", "starts", "starts")
            )
        }
        Err(other) => panic!("another error: {other}"),
        Ok(_) => panic!("a stage accumulated into an op's output and the batch was built"),
    }
}
