//! Counted batch stages (m14.s2i, D278): a stage that names a GPU count `n` (`gpu_task!(count(n); extent, ..)`) runs over
//! the rows `0 .. min(n, capacity)`; the rows from the count on keep what they held, and a count above the capacity is
//! clamped, not reported. The CPU twin reads `n` from its own buffer and agrees over the whole buffer.
//!
//! Every GPU result is held to the host model and to the CPU twin over the WHOLE output buffer: the free stage's result
//! persists across frames (rows past the count keep the row of an earlier frame), and the in-place stage's target is
//! rewritten with a sentinel before every frame, so rows it did not touch are checked too. The `twin_` tests run on the
//! CPU twin; the `gpu_` tests are `#[ignore]`d like the example's other GPU tests and run in full verification on the
//! adapter the environment selects, once with the indirect launch (the default) and once with the launch forced to the
//! bounded one, which must give the same bytes.

#![allow(
    clippy::needless_question_mark,
    clippy::needless_range_loop,
    clippy::assign_op_pattern
)]

use std::sync::Arc;

use batch_basics::app::block_on;
use gust::batch::{Batch, BatchDevice, CountedLaunch, D1, Error, Param, Stage, set_counted_launch};
use gust::cpu::CpuDevice;
use gust::map::MapHost;
use gust::{gpu_batch, gpu_task};
use gust_wgpu::HeadlessDevice;

/// The workgroup of every counted stage here (the default of a 1-D stage).
const WG: u32 = 256;
/// What the in-place stage's target holds before a frame.
const SENTINEL: u32 = 999_999;
/// The capacity of the exactness batches.
const CAP: u32 = 1000;

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
    Arc::new(HeadlessDevice::new().unwrap_or_else(|error| {
        panic!("a GPU adapter is required; this test must not silently skip: {error}")
    }))
}

// ---- the batch: a free counted stage and an in-place counted stage over one count ----

#[derive(Clone, Copy)]
struct Handles {
    n: Stage<u32, D1>,
    src: Stage<u32, D1>,
    dst: Stage<u32, D1>,
    out: Stage<u32, D1>,
}

macro_rules! define_build {
    ($build:ident, $device:ty, $depth:tt) => {
        fn $build(device: &Arc<$device>, capacity: u32) -> Result<(Batch<$device>, Handles), Error> {
            gpu_batch!("counted", on(device); in_flight($depth); |b| {
                let n = b.input::<u32>(1u32);
                let src = b.input::<u32>(capacity);
                let dst = b.input::<u32>(capacity);
                let out = gpu_task!(count(n); capacity, |i| -> u32 { src[i] * 3u32 + 1u32 });
                let _fill = gpu_task!(count(n); &mut dst, |i| { dst[i] = src[i] + 7u32; });
                Handles { n, src, dst, out }
            })
        }
    };
}

define_build!(build_cpu, CpuDevice, 1);
define_build!(build_gpu, HeadlessDevice, 1);
define_build!(build_gpu2, HeadlessDevice, 2);

fn source(capacity: u32, seed: u32) -> Vec<u32> {
    (0..capacity).map(|i| (i * 5 + seed) % 1000).collect()
}

/// The host model of the batch: the free stage's result persists, the in-place target starts each frame as the sentinel.
struct Model {
    capacity: u32,
    out: Vec<u32>,
}

impl Model {
    fn new(capacity: u32) -> Self {
        Self {
            capacity,
            out: vec![0; capacity as usize],
        }
    }

    /// One frame at count `n` over `src`: the result and the target after it.
    fn step(&mut self, src: &[u32], n: u32) -> (Vec<u32>, Vec<u32>) {
        let rows = n.min(self.capacity) as usize;
        for i in 0..rows {
            self.out[i] = src[i] * 3 + 1;
        }
        let dst = (0..self.capacity as usize)
            .map(|i| if i < rows { src[i] + 7 } else { SENTINEL })
            .collect();
        (self.out.clone(), dst)
    }
}

/// One frame: the count, the source (when given) and the sentinel are written, both results are read back.
fn frame<D: BatchDevice>(
    batch: &mut Batch<D>,
    h: &Handles,
    n: u32,
    src: Option<&[u32]>,
    sentinel: &[u32],
) -> (Vec<u32>, Vec<u32>)
where
    D::Buffer<u32>: 'static,
{
    block_on(async {
        let mut frame = batch.begin().await?;
        frame.write(&h.n, &[n])?;
        if let Some(src) = src {
            frame.write(&h.src, src)?;
        }
        frame.write(&h.dst, sentinel)?;
        let out = frame.ticket(&h.out)?;
        let dst = frame.ticket(&h.dst)?;
        let _submitted = frame.submit()?;
        Ok::<_, Error>((out.await?, dst.await?))
    })
    .unwrap()
}

/// The counts of check 1 for a capacity: 0, 1, wg-1, wg, wg+1, capacity-1, capacity, capacity+1, `u32::MAX`, then 0 again
/// (the result keeps its rows).
fn counts(capacity: u32) -> Vec<u32> {
    vec![
        0,
        1,
        WG - 1,
        WG,
        WG + 1,
        capacity - 1,
        capacity,
        capacity + 1,
        u32::MAX,
        0,
    ]
}

// ---- the twin ----

#[test]
fn twin_counts_match_the_model_over_the_whole_buffer() {
    let device = Arc::new(CpuDevice::new());
    let (mut batch, h) = build_cpu(&device, CAP).unwrap();
    let mut model = Model::new(CAP);
    let sentinel = vec![SENTINEL; CAP as usize];
    let mut src = source(CAP, 3);
    for (frame_number, n) in counts(CAP).into_iter().enumerate() {
        // A new source every third frame: the rows past the count keep an older frame's values.
        let write = (frame_number % 3 == 0).then(|| {
            src = source(CAP, frame_number as u32 + 1);
            src.clone()
        });
        let got = frame(&mut batch, &h, n, write.as_deref(), &sentinel);
        assert_eq!(got, model.step(&src, n), "count {n}");
    }
}

#[test]
fn twin_the_count_above_the_capacity_equals_the_capacity() {
    let device = Arc::new(CpuDevice::new());
    let sentinel = vec![SENTINEL; CAP as usize];
    let src = source(CAP, 9);
    let run = |n: u32| {
        let (mut batch, h) = build_cpu(&device, CAP).unwrap();
        frame(&mut batch, &h, n, Some(&src), &sentinel)
    };
    let full = run(CAP);
    assert_eq!(run(CAP + 1), full);
    assert_eq!(run(u32::MAX), full);
}

// ---- the GPU: check 1 (counts), check 3 (the args buffer and the pass), check 7 (the bounded launch) ----

/// The args of a frame: `x * y * z` is 0 for no row, otherwise `g <= x * y < g + 65535`, `x <= 65535` and `z == 1`
/// (check 3), for the groups `g = ceil(min(n, capacity) / WG)` the stage launches.
fn check_args(device: &HeadlessDevice, stages: usize, n: u32, capacity: u32) {
    let args = HeadlessDevice::counted_probe_args();
    assert_eq!(args.len(), stages, "one args buffer per counted stage");
    let rows = u64::from(n.min(capacity));
    let g = rows.div_ceil(u64::from(WG));
    for buffer in args {
        let words = device.readback(&*buffer).unwrap();
        let [x, y, z] = [words[0], words[1], words[2]].map(u64::from);
        assert!(x <= 65_535, "n = {n}: x = {x}");
        if rows == 0 {
            assert_eq!(x * y * z, 0, "n = {n}: no row, no group");
        } else {
            assert_eq!(z, 1, "n = {n}");
            assert!(
                g <= x * y && x * y < g + 65_535,
                "n = {n}: g = {g}, grid {x} x {y}"
            );
        }
    }
}

fn gpu_counts(launch: CountedLaunch) {
    let device = private_device();
    let _launch = Launch::set(launch);
    HeadlessDevice::counted_probe(true);
    let (mut batch, h) = build_gpu(&device, CAP).unwrap();
    let twin = Arc::new(CpuDevice::new());
    let (mut twin_batch, th) = build_cpu(&twin, CAP).unwrap();
    let mut model = Model::new(CAP);
    let sentinel = vec![SENTINEL; CAP as usize];
    let mut src = source(CAP, 3);
    let frames = counts(CAP);
    for (frame_number, n) in frames.iter().copied().enumerate() {
        let write = (frame_number % 3 == 0).then(|| {
            src = source(CAP, frame_number as u32 + 1);
            src.clone()
        });
        let before = device.gust_submissions();
        let got = frame(&mut batch, &h, n, write.as_deref(), &sentinel);
        assert_eq!(
            device.gust_submissions() - before,
            1,
            "count {n}: one submission per frame"
        );
        let expected = model.step(&src, n);
        assert_eq!(got, expected, "count {n}: GPU against the model");
        let twin_got = frame(&mut twin_batch, &th, n, write.as_deref(), &sentinel);
        assert_eq!(got, twin_got, "count {n}: GPU against the twin");
        if launch == CountedLaunch::Indirect {
            check_args(&device, 2, n, CAP);
        }
    }
    let passes = HeadlessDevice::counted_probe_passes();
    match launch {
        CountedLaunch::Indirect => assert_eq!(
            passes,
            2 * frames.len() as u64,
            "each frame records one indirect pass per counted stage"
        ),
        _ => assert_eq!(passes, 0, "the bounded launch records no indirect pass"),
    }
    HeadlessDevice::counted_probe(false);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_counts_indirect() {
    gpu_counts(CountedLaunch::Indirect);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_counts_bounded() {
    gpu_counts(CountedLaunch::Bounded);
}

// ---- the GPU: check 2 (the per-dimension limit of 65535 groups) ----

#[derive(Clone, Copy)]
struct Big {
    n: Stage<u32, D1>,
    dst: Stage<u32, D1>,
}

fn build_big(
    device: &Arc<HeadlessDevice>,
    capacity: u32,
) -> Result<(Batch<HeadlessDevice>, Big), Error> {
    gpu_batch!("counted-big", on(device); in_flight(1); |b| {
        let n = b.input::<u32>(1u32);
        let dst = b.input::<u32>(capacity);
        let _bump = gpu_task!(count(n); &mut dst, |i| { dst[i] = dst[i] + 1u32; });
        Big { n, dst }
    })
}

/// Capacities giving `groups` groups, with the counts around the rollover of the first dimension: every row below the
/// count is written exactly once (the body adds one to a zeroed target), none above.
fn gpu_group_limit(launch: CountedLaunch) {
    let device = private_device();
    let _launch = Launch::set(launch);
    HeadlessDevice::counted_probe(true);
    let limit = 65_535 * WG;
    let cases: [(u32, u32); 6] = [
        (65_535, 65_535 * WG),
        (65_536, limit),
        (65_536, limit + 1),
        (65_536, 65_536 * WG),
        (65_537, 65_537 * WG),
        (65_537, u32::MAX),
    ];
    for (groups, n) in cases {
        let capacity = groups * WG;
        let (mut batch, h) = build_big(&device, capacity).unwrap();
        let zeros = vec![0_u32; capacity as usize];
        let got = block_on(async {
            let mut frame = batch.begin().await?;
            frame.write(&h.n, &[n])?;
            frame.write(&h.dst, &zeros)?;
            let dst = frame.ticket(&h.dst)?;
            let _submitted = frame.submit()?;
            Ok::<_, Error>(dst.await?)
        })
        .unwrap();
        let rows = n.min(capacity) as usize;
        assert!(
            got[..rows].iter().all(|&v| v == 1),
            "{groups} groups, n = {n}: a row below the count was not written exactly once"
        );
        assert!(
            got[rows..].iter().all(|&v| v == 0),
            "{groups} groups, n = {n}: a row at or past the count was written"
        );
        if launch == CountedLaunch::Indirect {
            check_args(&device, 1, n, capacity);
        }
        HeadlessDevice::counted_probe(true);
    }
    HeadlessDevice::counted_probe(false);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_group_limit_indirect() {
    gpu_group_limit(CountedLaunch::Indirect);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_group_limit_bounded() {
    gpu_group_limit(CountedLaunch::Bounded);
}

// ---- check 4: a count that changes every frame, produced on the GPU, with no read of it ----

const CYCLE: u32 = 24;
const STEP: u32 = 100;
const COUNTER_CAP: u32 = 12 * STEP;
const FRAMES: u32 = 48;

#[derive(Clone, Copy)]
struct Counter {
    k: Param<u32>,
    cnt: Stage<u32, D1>,
    src: Stage<u32, D1>,
    out: Stage<u32, D1>,
}

macro_rules! define_counter {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Counter), Error> {
            gpu_batch!("counter", on(device); in_flight(1); |b| {
                let k = b.param::<u32>(0);
                let cnt = b.input::<u32>(1u32);
                let src = b.input::<u32>(COUNTER_CAP);
                // The count is produced by a stage on the GPU, going up, down and through zero, and past the capacity.
                let _tick = gpu_task!(&mut cnt, |i| {
                    let t = k.get() % 24u32;
                    let m = t.min(24u32 - t);
                    cnt[i] = m * 100u32 + k.get() / 24u32;
                });
                let out = gpu_task!(count(cnt); COUNTER_CAP, |i| -> u32 { src[i] + 1u32 });
                Counter { k, cnt, src, out }
            })
        }
    };
}

define_counter!(build_counter_cpu, CpuDevice);
define_counter!(build_counter_gpu, HeadlessDevice);

/// The count the counter stage writes for frame `k`.
fn counter_at(k: u32) -> u32 {
    let t = k % CYCLE;
    t.min(CYCLE - t) * STEP + k / CYCLE
}

/// 48 frames: only the result is read each frame (a ticket on `out`), never the count or the args buffer; the last frame
/// reads the count too. Each frame equals the model and the twin.
fn counter_frames<D: BatchDevice>(batch: &mut Batch<D>, h: &Counter) -> Vec<Vec<u32>>
where
    D::Buffer<u32>: 'static,
{
    let src: Vec<u32> = (0..COUNTER_CAP).map(|i| i * 3 % 1000).collect();
    let mut all = Vec::new();
    for k in 0..FRAMES {
        let results = block_on(async {
            let mut frame = batch.begin().await?;
            frame.set(&h.k, k)?;
            if k == 0 {
                frame.write(&h.src, &src)?;
            }
            let out = frame.ticket(&h.out)?;
            let cnt = (k == FRAMES - 1)
                .then(|| frame.ticket(&h.cnt))
                .transpose()?;
            let _submitted = frame.submit()?;
            let out = out.await?;
            if let Some(cnt) = cnt {
                assert_eq!(cnt.await?, [counter_at(k)], "the count of the last frame");
            }
            Ok::<_, Error>(out)
        })
        .unwrap();
        all.push(results);
    }
    all
}

fn counter_model() -> Vec<Vec<u32>> {
    let src: Vec<u32> = (0..COUNTER_CAP).map(|i| i * 3 % 1000).collect();
    let mut out = vec![0_u32; COUNTER_CAP as usize];
    (0..FRAMES)
        .map(|k| {
            let rows = counter_at(k).min(COUNTER_CAP) as usize;
            for i in 0..rows {
                out[i] = src[i] + 1;
            }
            out.clone()
        })
        .collect()
}

#[test]
fn twin_a_count_that_changes_every_frame() {
    let counts: Vec<u32> = (0..FRAMES).map(counter_at).collect();
    assert!(counts.contains(&0), "the sequence goes through zero");
    assert!(
        counts.iter().any(|&n| n > COUNTER_CAP),
        "the sequence goes above the capacity"
    );
    assert!(counts.windows(2).any(|w| w[1] < w[0]), "and down");
    let device = Arc::new(CpuDevice::new());
    let (mut batch, h) = build_counter_cpu(&device).unwrap();
    assert_eq!(counter_frames(&mut batch, &h), counter_model());
}

fn gpu_counter(launch: CountedLaunch) {
    let device = private_device();
    let _launch = Launch::set(launch);
    let (mut batch, h) = build_counter_gpu(&device).unwrap();
    let before = device.gust_submissions();
    let got = counter_frames(&mut batch, &h);
    assert_eq!(
        device.gust_submissions() - before,
        u64::from(FRAMES),
        "one submission per frame, no read of the count in between"
    );
    assert_eq!(got, counter_model());
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_count_that_changes_every_frame_indirect() {
    gpu_counter(CountedLaunch::Indirect);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_count_that_changes_every_frame_bounded() {
    gpu_counter(CountedLaunch::Bounded);
}

// ---- check 4 again, with the count of `compact` ----

#[derive(Clone, Copy)]
struct Compacted {
    k: Param<u32>,
    flags: Stage<u32, D1>,
    values: Stage<u32, D1>,
    gathered: Stage<u32, D1>,
}

macro_rules! define_compacted {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Compacted), Error> {
            gpu_batch!("compacted", on(device); in_flight(1); |b| {
                let k = b.param::<u32>(0);
                let flags = b.input::<u32>(COUNTER_CAP);
                let values = b.input::<u32>(COUNTER_CAP);
                let (idx, n_set) = gust::compact(&flags);
                let gathered = gpu_task!(count(n_set); COUNTER_CAP, |i| -> u32 { values[idx[i]] });
                Compacted { k, flags, values, gathered }
            })
        }
    };
}

define_compacted!(build_compacted_cpu, CpuDevice);
define_compacted!(build_compacted_gpu, HeadlessDevice);

/// The flags of frame `k`: a share of the rows, going 0, 1/12, .. 12/12 and back, so the count changes every frame and
/// passes through 0.
fn flags_at(k: u32) -> Vec<u32> {
    let t = k % CYCLE;
    let thresh = t.min(CYCLE - t);
    (0..COUNTER_CAP)
        .map(|i| u32::from((i + k * 3) % 12 < thresh))
        .collect()
}

fn compacted_frames<D: BatchDevice>(batch: &mut Batch<D>, h: &Compacted) -> Vec<Vec<u32>>
where
    D::Buffer<u32>: 'static,
{
    let values: Vec<u32> = (0..COUNTER_CAP).map(|i| i * 7 % 1000).collect();
    (0..FRAMES)
        .map(|k| {
            block_on(async {
                let mut frame = batch.begin().await?;
                frame.set(&h.k, k)?;
                if k == 0 {
                    frame.write(&h.values, &values)?;
                }
                frame.write(&h.flags, &flags_at(k))?;
                let gathered = frame.ticket(&h.gathered)?;
                let _submitted = frame.submit()?;
                Ok::<_, Error>(gathered.await?)
            })
            .unwrap()
        })
        .collect()
}

fn compacted_model() -> Vec<Vec<u32>> {
    let values: Vec<u32> = (0..COUNTER_CAP).map(|i| i * 7 % 1000).collect();
    let mut out = vec![0_u32; COUNTER_CAP as usize];
    (0..FRAMES)
        .map(|k| {
            let selected: Vec<u32> = flags_at(k)
                .iter()
                .enumerate()
                .filter(|(_, flag)| **flag != 0)
                .map(|(i, _)| values[i])
                .collect();
            out[..selected.len()].copy_from_slice(&selected);
            out.clone()
        })
        .collect()
}

#[test]
fn twin_the_count_of_compact_counts_a_stage() {
    let device = Arc::new(CpuDevice::new());
    let (mut batch, h) = build_compacted_cpu(&device).unwrap();
    assert_eq!(compacted_frames(&mut batch, &h), compacted_model());
}

fn gpu_compacted(launch: CountedLaunch) {
    let device = private_device();
    let _launch = Launch::set(launch);
    let (mut batch, h) = build_compacted_gpu(&device).unwrap();
    assert_eq!(compacted_frames(&mut batch, &h), compacted_model());
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_the_count_of_compact_counts_a_stage_indirect() {
    gpu_compacted(CountedLaunch::Indirect);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_the_count_of_compact_counts_a_stage_bounded() {
    gpu_compacted(CountedLaunch::Bounded);
}

// ---- check 5: frames in flight ----

/// Two frames are submitted before either is read, with different counts and sources: each equals the frame run alone.
#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_frames_in_flight_keep_their_counts() {
    let device = private_device();
    let (mut batch, h) = build_gpu2(&device, CAP).unwrap();
    let sentinel = vec![SENTINEL; CAP as usize];
    let mut model = Model::new(CAP);
    let sources = [source(CAP, 2), source(CAP, 11)];
    let counts = [700_u32, 300];
    let expected: Vec<_> = counts
        .iter()
        .zip(&sources)
        .map(|(&n, src)| model.step(src, n))
        .collect();
    let mut tickets = Vec::new();
    let mut held = Vec::new();
    for (n, src) in counts.iter().zip(&sources) {
        let mut frame = block_on(batch.begin()).unwrap();
        frame.write(&h.n, &[*n]).unwrap();
        frame.write(&h.src, src).unwrap();
        frame.write(&h.dst, &sentinel).unwrap();
        let out = frame.ticket(&h.out).unwrap();
        let dst = frame.ticket(&h.dst).unwrap();
        held.push(frame.submit().unwrap());
        tickets.push((out, dst));
    }
    for ((out, dst), expected) in tickets.into_iter().zip(expected).rev() {
        assert_eq!((block_on(out).unwrap(), block_on(dst).unwrap()), expected);
    }
}

// ---- check 6: resize with a counted stage present ----

#[derive(Clone, Copy)]
struct Resizable {
    len: Param<u32>,
    n: Stage<u32, D1>,
    src: Stage<u32, D1>,
    out: Stage<u32, D1>,
}

macro_rules! define_sized {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Resizable), Error> {
            gpu_batch!("counted-resize", on(device); in_flight(1); |b| {
                let len = b.param::<u32>(8);
                let n = b.input::<u32>(1u32);
                let src = b.input::<u32>(&len);
                let out = gpu_task!(count(n); &len, |i| -> u32 { src[i] + 100u32 });
                Resizable { len, n, src, out }
            })
        }
    };
}

define_sized!(build_sized_cpu, CpuDevice);
define_sized!(build_sized_gpu, HeadlessDevice);

fn sized_frame<D: BatchDevice>(
    batch: &mut Batch<D>,
    h: &Resizable,
    len: Option<u32>,
    n: u32,
    src: Option<&[u32]>,
) -> Vec<u32>
where
    D::Buffer<u32>: 'static,
{
    block_on(async {
        let mut frame = batch.begin().await?;
        if let Some(len) = len {
            frame.set(&h.len, len)?;
        }
        frame.write(&h.n, &[n])?;
        if let Some(src) = src {
            frame.write(&h.src, src)?;
        }
        let out = frame.ticket(&h.out)?;
        let _submitted = frame.submit()?;
        Ok::<_, Error>(out.await?)
    })
    .unwrap()
}

/// Grow, shrink and grow again with a counted stage present: the capacity follows the shape, a new buffer holds zeros
/// (rows the count does not reach stay zero), the count is clamped against the new capacity, and a wrong-sized write is
/// refused as before.
fn resize_follows_the_capacity<D: BatchDevice>(mut batch: Batch<D>, h: &Resizable)
where
    D::Buffer<u32>: 'static,
{
    let at = |len: u32, seed: u32| -> Vec<u32> { (0..len).map(|i| (i * 3 + seed) % 50).collect() };
    let a = at(8, 1);
    let first = sized_frame(&mut batch, h, None, 5, Some(&a));
    let mut expected: Vec<u32> = (0..8).map(|i| if i < 5 { a[i] + 100 } else { 0 }).collect();
    assert_eq!(first, expected, "capacity 8, count 5");
    let b = at(20, 2);
    let grown = sized_frame(&mut batch, h, Some(20), 17, Some(&b));
    expected = (0..20)
        .map(|i| if i < 17 { b[i] + 100 } else { 0 })
        .collect();
    assert_eq!(grown, expected, "capacity 20, count 17");
    let c = at(5, 3);
    let shrunk = sized_frame(&mut batch, h, Some(5), 99, Some(&c));
    assert_eq!(
        shrunk,
        c.iter().map(|v| v + 100).collect::<Vec<_>>(),
        "capacity 5, a count of 99 clamps to the capacity"
    );
    let fresh = sized_frame(&mut batch, h, Some(11), 4, None);
    let mut zeros_then_hundreds = vec![0_u32; 11];
    zeros_then_hundreds[..4].fill(100);
    assert_eq!(
        fresh, zeros_then_hundreds,
        "new buffers hold zeros: the source reads zero, the four counted rows hold 100, the others stay zero"
    );
}

#[test]
fn twin_resize_follows_the_capacity() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_sized_cpu(&device).unwrap();
    resize_follows_the_capacity(batch, &h);
}

fn gpu_resize(launch: CountedLaunch) {
    let device = private_device();
    let _launch = Launch::set(launch);
    let (batch, h) = build_sized_gpu(&device).unwrap();
    resize_follows_the_capacity(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_resize_follows_the_capacity_indirect() {
    gpu_resize(CountedLaunch::Indirect);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_resize_follows_the_capacity_bounded() {
    gpu_resize(CountedLaunch::Bounded);
}

// ---- a count of the wrong length ----

/// A count with more than one element is refused when the stage is recorded, by name.
#[test]
fn twin_a_count_of_many_elements_is_refused() {
    let device = Arc::new(CpuDevice::new());
    let result = gpu_batch!("not-a-count", on(&device); in_flight(1); |b| {
        let n = b.input::<u32>(4u32);
        let src = b.input::<u32>(8u32);
        let _out = gpu_task!(count(n); 8u32, |i| -> u32 { src[i] });
    });
    match result {
        Err(Error::NotCount { len, .. }) => assert_eq!(len, 4),
        Err(other) => panic!("another error: {other}"),
        Ok(_) => panic!("a count of four elements was accepted"),
    }
}
