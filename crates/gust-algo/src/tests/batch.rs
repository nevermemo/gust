//! The reductions, the scans and the compaction as ops of a `gust::batch` (slice m14.s2f2a, D259), through
//! `gust::batch::build` and `Builder::op`: the GPU device against gust-algo's immediate functions and the independent
//! host references, and the CPU device against the references and the twin chain of `cpu.rs`. The `gpu_` tests need an
//! adapter and name it in the log; a twin result is never GPU evidence.

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use gust::batch::{Batch, BatchDevice, Builder, D1, Error, Stage, Ticket};
use gust::cpu::CpuDevice;
use gust_wgpu::{GpuVec, HeadlessDevice};

use super::{
    common::{
        CPU_C, GPU_C, Rng, assert_f32_sum_by, assert_same, exact_sum, reference_indices,
        reference_scan,
    },
    gpu::device,
};
use crate::{
    batch::{Compact, Count, Max, Min, Scan, Sum, SumWrapping},
    bound, compact, cpu, reduce, scan,
};

/// A length that is not a power of two, and one that takes two passes of every reduction.
const SIZES: [usize; 2] = [1000, 100_003];

const DEADLINE: Duration = Duration::from_secs(120);

/// Drive `future` on this thread until it completes (a native ticket completes only while something polls the device).
fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    let started = Instant::now();
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => {
                assert!(started.elapsed() < DEADLINE, "a future stayed pending");
                std::thread::yield_now();
            }
        }
    }
}

/// The inputs of the batch, on the host. Small values: no sum or prefix leaves `u32` or `i32` at either size.
#[derive(Clone)]
struct Data {
    u: Vec<u32>,
    i: Vec<i32>,
    f: Vec<f32>,
    flags: Vec<u32>,
}

impl Data {
    fn new(n: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let u: Vec<u32> = (0..n).map(|_| rng.below(1 << 15) as u32).collect();
        let i = (0..n)
            .map(|_| rng.below(1 << 15) as i32 - (1 << 14))
            .collect();
        let f = (0..n).map(|_| rng.below(1 << 15) as f32 / 1024.0).collect();
        // The flag rule: any non-zero value is set.
        let flags = (0..n)
            .map(|_| {
                if rng.below(3) == 0 {
                    1 + rng.below(5) as u32
                } else {
                    0
                }
            })
            .collect();
        Self { u, i, f, flags }
    }
}

/// What one frame writes: an input that is `None` keeps the value of the frame before (D258).
#[derive(Clone, Default)]
struct Frame {
    u: Option<Vec<u32>>,
    i: Option<Vec<i32>>,
    f: Option<Vec<f32>>,
    flags: Option<Vec<u32>>,
}

impl Frame {
    /// A frame that writes every input, from `seed`.
    fn all(n: usize, seed: u64) -> Self {
        let data = Data::new(n, seed);
        Self {
            u: Some(data.u),
            i: Some(data.i),
            f: Some(data.f),
            flags: Some(data.flags),
        }
    }

    /// A frame that writes `u` and `flags` only, from `seed`.
    fn partial(n: usize, seed: u64) -> Self {
        let data = Data::new(n, seed);
        Self {
            u: Some(data.u),
            flags: Some(data.flags),
            ..Self::default()
        }
    }

    /// The inputs after this frame, given the inputs before it.
    fn apply(&self, data: &mut Data) {
        if let Some(u) = &self.u {
            data.u.clone_from(u);
        }
        if let Some(i) = &self.i {
            data.i.clone_from(i);
        }
        if let Some(f) = &self.f {
            data.f.clone_from(f);
        }
        if let Some(flags) = &self.flags {
            data.flags.clone_from(flags);
        }
    }
}

/// The three frames of the exactness tests: every input, then `u` and `flags` alone (the others hold), then every
/// input.
fn three_frames(n: usize, seed: u64) -> [Frame; 3] {
    [
        Frame::all(n, seed),
        Frame::partial(n, seed + 1),
        Frame::all(n, seed + 2),
    ]
}

/// The handles of the batch: the inputs and the outputs of every op.
struct Handles {
    u: Stage<u32, D1>,
    i: Stage<i32, D1>,
    f: Stage<f32, D1>,
    flags: Stage<u32, D1>,
    sum_u: Stage<u32, D1>,
    min_u: Stage<u32, D1>,
    max_u: Stage<u32, D1>,
    wrapping: Stage<u32, D1>,
    sum_i: Stage<i32, D1>,
    min_i: Stage<i32, D1>,
    max_i: Stage<i32, D1>,
    sum_f: Stage<f32, D1>,
    min_f: Stage<f32, D1>,
    max_f: Stage<f32, D1>,
    count: Stage<u32, D1>,
    inclusive: Stage<u32, D1>,
    exclusive: Stage<u32, D1>,
    indices: Stage<u32, D1>,
    n: Stage<u32, D1>,
}

fn expose_all<D: BatchDevice>(b: &Builder<D>, h: &Handles) {
    b.expose(&h.u);
    b.expose(&h.i);
    b.expose(&h.f);
    b.expose(&h.flags);
    b.expose(&h.sum_u);
    b.expose(&h.min_u);
    b.expose(&h.max_u);
    b.expose(&h.wrapping);
    b.expose(&h.sum_i);
    b.expose(&h.min_i);
    b.expose(&h.max_i);
    b.expose(&h.sum_f);
    b.expose(&h.min_f);
    b.expose(&h.max_f);
    b.expose(&h.count);
    b.expose(&h.inclusive);
    b.expose(&h.exclusive);
    b.expose(&h.indices);
    b.expose(&h.n);
}

/// The two batches of a device: every op over four inputs, and a compaction alone.
macro_rules! define_builders {
    ($all:ident, $compaction:ident, $device:ty) => {
        fn $all(
            device: &Arc<$device>,
            n: usize,
            in_flight: u32,
        ) -> Result<(Batch<$device>, Handles), Error> {
            let n = u32::try_from(n).unwrap();
            gust::batch::build("ops", device, in_flight, |b| {
                let u = b.input::<u32, _>(n);
                let i = b.input::<i32, _>(n);
                let f = b.input::<f32, _>(n);
                let flags = b.input::<u32, _>(n);
                let sum_u = b.op(&["sum_u"], Sum::of(&u));
                let min_u = b.op(&["min_u"], Min::of(&u));
                let max_u = b.op(&["max_u"], Max::of(&u));
                let wrapping = b.op(&["wrapping"], SumWrapping::of(&u));
                let sum_i = b.op(&["sum_i"], Sum::of(&i));
                let min_i = b.op(&["min_i"], Min::of(&i));
                let max_i = b.op(&["max_i"], Max::of(&i));
                let sum_f = b.op(&["sum_f"], Sum::of(&f));
                let min_f = b.op(&["min_f"], Min::of(&f));
                let max_f = b.op(&["max_f"], Max::of(&f));
                let count = b.op(&["count"], Count::of(&flags));
                let inclusive = b.op(&["inclusive"], Scan::of(&u));
                let exclusive = b.op(&["exclusive"], Scan::exclusive(&u));
                let (indices, n_set) = b.op(&["indices", "n"], Compact::of(&flags));
                let h = Handles {
                    u,
                    i,
                    f,
                    flags,
                    sum_u,
                    min_u,
                    max_u,
                    wrapping,
                    sum_i,
                    min_i,
                    max_i,
                    sum_f,
                    min_f,
                    max_f,
                    count,
                    inclusive,
                    exclusive,
                    indices,
                    n: n_set,
                };
                expose_all(b, &h);
                h
            })
        }

        fn $compaction(
            device: &Arc<$device>,
            n: usize,
        ) -> Result<(Batch<$device>, [Stage<u32, D1>; 3]), Error> {
            let n = u32::try_from(n).unwrap();
            gust::batch::build("compaction", device, 1, |b| {
                let flags = b.input::<u32, _>(n);
                let (indices, count) = b.op(&["indices", "count"], Compact::of(&flags));
                let handles = [flags, indices, count];
                for handle in &handles {
                    b.expose(handle);
                }
                handles
            })
        }
    };
}

define_builders!(build_gpu, build_compaction_gpu, HeadlessDevice);
define_builders!(build_cpu, build_compaction_cpu, CpuDevice);

/// The tickets of one frame.
struct Pending {
    sum_u: Ticket<u32>,
    min_u: Ticket<u32>,
    max_u: Ticket<u32>,
    wrapping: Ticket<u32>,
    sum_i: Ticket<i32>,
    min_i: Ticket<i32>,
    max_i: Ticket<i32>,
    sum_f: Ticket<f32>,
    min_f: Ticket<f32>,
    max_f: Ticket<f32>,
    count: Ticket<u32>,
    inclusive: Ticket<u32>,
    exclusive: Ticket<u32>,
    indices: Ticket<u32>,
    n: Ticket<u32>,
}

/// What the ops gave (or what a reference says they must give) for one frame's data. `indices` holds the first `n`
/// positions only: the tail of a recorded compaction is not read (D259).
#[derive(Debug)]
struct Results {
    sum_u: u32,
    min_u: u32,
    max_u: u32,
    wrapping: u32,
    sum_i: i32,
    min_i: i32,
    max_i: i32,
    sum_f: f32,
    min_f: f32,
    max_f: f32,
    count: u32,
    inclusive: Vec<u32>,
    exclusive: Vec<u32>,
    indices: Vec<u32>,
    n: u32,
}

fn one<T: Copy>(values: Result<Vec<T>, Error>) -> T {
    let values = values.expect("the ticket resolves");
    assert_eq!(values.len(), 1);
    values[0]
}

impl Pending {
    fn resolve(self) -> Results {
        let n = one(block_on(self.n));
        let mut indices = block_on(self.indices).expect("the ticket resolves");
        assert!(n as usize <= indices.len(), "a count past the flags");
        indices.truncate(n as usize);
        Results {
            sum_u: one(block_on(self.sum_u)),
            min_u: one(block_on(self.min_u)),
            max_u: one(block_on(self.max_u)),
            wrapping: one(block_on(self.wrapping)),
            sum_i: one(block_on(self.sum_i)),
            min_i: one(block_on(self.min_i)),
            max_i: one(block_on(self.max_i)),
            sum_f: one(block_on(self.sum_f)),
            min_f: one(block_on(self.min_f)),
            max_f: one(block_on(self.max_f)),
            count: one(block_on(self.count)),
            inclusive: block_on(self.inclusive).expect("the ticket resolves"),
            exclusive: block_on(self.exclusive).expect("the ticket resolves"),
            indices,
            n,
        }
    }
}

/// Record one frame of `frame`'s writes, reserve every ticket and submit. The frame is one submission.
fn submit<D: BatchDevice>(batch: &mut Batch<D>, h: &Handles, frame: &Frame) -> Pending
where
    D::Buffer<u32>: 'static,
    D::Buffer<i32>: 'static,
    D::Buffer<f32>: 'static,
{
    let mut run = block_on(batch.begin()).expect("a frame begins");
    if let Some(u) = &frame.u {
        run.write(&h.u, u).unwrap();
    }
    if let Some(i) = &frame.i {
        run.write(&h.i, i).unwrap();
    }
    if let Some(f) = &frame.f {
        run.write(&h.f, f).unwrap();
    }
    if let Some(flags) = &frame.flags {
        run.write(&h.flags, flags).unwrap();
    }
    let pending = Pending {
        sum_u: run.ticket(&h.sum_u).unwrap(),
        min_u: run.ticket(&h.min_u).unwrap(),
        max_u: run.ticket(&h.max_u).unwrap(),
        wrapping: run.ticket(&h.wrapping).unwrap(),
        sum_i: run.ticket(&h.sum_i).unwrap(),
        min_i: run.ticket(&h.min_i).unwrap(),
        max_i: run.ticket(&h.max_i).unwrap(),
        sum_f: run.ticket(&h.sum_f).unwrap(),
        min_f: run.ticket(&h.min_f).unwrap(),
        max_f: run.ticket(&h.max_f).unwrap(),
        count: run.ticket(&h.count).unwrap(),
        inclusive: run.ticket(&h.inclusive).unwrap(),
        exclusive: run.ticket(&h.exclusive).unwrap(),
        indices: run.ticket(&h.indices).unwrap(),
        n: run.ticket(&h.n).unwrap(),
    };
    let _submitted = run.submit().expect("the frame is submitted");
    pending
}

/// The independent host references of every op over `data`: `i128` sums, `Iterator::min`, a sequential prefix sum, the
/// positions of the set flags, and an `f64` accumulation for the float sum.
fn reference(data: &Data) -> Results {
    let indices = reference_indices(&data.flags);
    let float_min = data.f.iter().copied().fold(f32::INFINITY, f32::min);
    let float_max = data.f.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    Results {
        sum_u: u32::try_from(exact_sum(&data.u)).unwrap(),
        min_u: *data.u.iter().min().unwrap(),
        max_u: *data.u.iter().max().unwrap(),
        wrapping: data
            .u
            .iter()
            .fold(0u32, |sum, value| sum.wrapping_add(*value)),
        sum_i: i32::try_from(exact_sum(&data.i)).unwrap(),
        min_i: *data.i.iter().min().unwrap(),
        max_i: *data.i.iter().max().unwrap(),
        sum_f: bound::sum_f64(&data.f) as f32,
        min_f: float_min,
        max_f: float_max,
        count: u32::try_from(indices.len()).unwrap(),
        inclusive: reference_scan(&data.u, false),
        exclusive: reference_scan(&data.u, true),
        n: u32::try_from(indices.len()).unwrap(),
        indices,
    }
}

/// gust-algo's immediate functions over `data`, on `device`.
fn immediate(device: &gust_wgpu::DeviceHandle, data: &Data) -> Results {
    let u = GpuVec::from_slice_on(device, &data.u).unwrap();
    let i = GpuVec::from_slice_on(device, &data.i).unwrap();
    let f = GpuVec::from_slice_on(device, &data.f).unwrap();
    let flags = GpuVec::from_slice_on(device, &data.flags).unwrap();
    let (indices, n_set) = compact::indices(&flags).unwrap();
    let n = n_set.to_vec().unwrap()[0];
    let mut indices = indices.to_vec().unwrap();
    indices.truncate(n as usize);
    Results {
        sum_u: reduce::sum(&u).unwrap(),
        min_u: reduce::min(&u).unwrap(),
        max_u: reduce::max(&u).unwrap(),
        wrapping: reduce::sum_wrapping(&u).unwrap(),
        sum_i: reduce::sum(&i).unwrap(),
        min_i: reduce::min(&i).unwrap(),
        max_i: reduce::max(&i).unwrap(),
        sum_f: reduce::sum(&f).unwrap(),
        min_f: reduce::min(&f).unwrap(),
        max_f: reduce::max(&f).unwrap(),
        count: compact::count(&flags).unwrap().to_vec().unwrap()[0],
        inclusive: scan::inclusive(&u).unwrap().to_vec().unwrap(),
        exclusive: scan::exclusive(&u).unwrap().to_vec().unwrap(),
        indices,
        n,
    }
}

/// The twin chains of `cpu.rs` over `data` (the v0 kernels' twins).
fn twin(data: &Data) -> Results {
    let (indices, n) = cpu::compact_indices(&data.flags).unwrap();
    let mut indices = indices;
    indices.truncate(n as usize);
    Results {
        sum_u: cpu::sum(&data.u).unwrap(),
        min_u: cpu::min(&data.u).unwrap().unwrap(),
        max_u: cpu::max(&data.u).unwrap().unwrap(),
        wrapping: cpu::sum_wrapping(&data.u).unwrap(),
        sum_i: cpu::sum(&data.i).unwrap(),
        min_i: cpu::min(&data.i).unwrap().unwrap(),
        max_i: cpu::max(&data.i).unwrap().unwrap(),
        sum_f: cpu::sum(&data.f).unwrap(),
        min_f: cpu::min(&data.f).unwrap().unwrap(),
        max_f: cpu::max(&data.f).unwrap().unwrap(),
        count: cpu::compact_count(&data.flags).unwrap(),
        inclusive: cpu::inclusive(&data.u).unwrap(),
        exclusive: cpu::exclusive(&data.u).unwrap(),
        indices,
        n,
    }
}

/// `got` equals `want`: integers, `min`, `max`, scans, the count and the first `n` indices exactly, the float sum
/// within `float_tolerance` (0 for bit equality).
#[track_caller]
fn assert_results(label: &str, got: &Results, want: &Results, float_tolerance: f64) {
    assert_eq!(got.sum_u, want.sum_u, "sum_u, {label}");
    assert_eq!(got.min_u, want.min_u, "min_u, {label}");
    assert_eq!(got.max_u, want.max_u, "max_u, {label}");
    assert_eq!(got.wrapping, want.wrapping, "wrapping, {label}");
    assert_eq!(got.sum_i, want.sum_i, "sum_i, {label}");
    assert_eq!(got.min_i, want.min_i, "min_i, {label}");
    assert_eq!(got.max_i, want.max_i, "max_i, {label}");
    assert_eq!(got.min_f, want.min_f, "min_f, {label}");
    assert_eq!(got.max_f, want.max_f, "max_f, {label}");
    assert_eq!(got.count, want.count, "count, {label}");
    assert_eq!(got.n, want.n, "n, {label}");
    assert_same(
        &got.inclusive,
        &want.inclusive,
        &format!("inclusive, {label}"),
    );
    assert_same(
        &got.exclusive,
        &want.exclusive,
        &format!("exclusive, {label}"),
    );
    assert_same(&got.indices, &want.indices, &format!("indices, {label}"));
    if float_tolerance == 0.0 {
        assert_eq!(got.sum_f.to_bits(), want.sum_f.to_bits(), "sum_f, {label}");
    } else {
        let distance = (f64::from(got.sum_f) - f64::from(want.sum_f)).abs();
        assert!(
            distance <= float_tolerance,
            "sum_f, {label}: {} against {} is {distance:e}, past {float_tolerance:e}",
            got.sum_f,
            want.sum_f
        );
    }
}

/// The larger of the bounds of the two schedules a GPU reduction may have run (v1, or the v0 fallback).
fn gpu_bound(n: u32, c: f64, abs_sum: f64) -> f64 {
    bound::f32_sum_bound(n, c, abs_sum).max(bound::f32_sum_bound_v1(n, c, abs_sum))
}

/// Run `frames` on a batch of `D` one after the other, resolving each frame before the next, and give each frame's
/// results with the inputs the frame ran on.
fn run_frames<D: BatchDevice>(
    batch: &mut Batch<D>,
    h: &Handles,
    n: usize,
    frames: &[Frame],
) -> Vec<(Data, Results)>
where
    D::Buffer<u32>: 'static,
    D::Buffer<i32>: 'static,
    D::Buffer<f32>: 'static,
{
    // The inputs start as zeros, as the batch's buffers do.
    let mut data = Data {
        u: vec![0; n],
        i: vec![0; n],
        f: vec![0.0; n],
        flags: vec![0; n],
    };
    frames
        .iter()
        .map(|frame| {
            frame.apply(&mut data);
            let results = submit(batch, h, frame).resolve();
            (data.clone(), results)
        })
        .collect()
}

#[test]
fn gpu_every_op_of_a_batch_equals_the_immediate_functions_over_three_frames() {
    let device = device();
    for n in SIZES {
        let (mut batch, h) = build_gpu(&device, n, 1).unwrap();
        for (k, (data, got)) in run_frames(&mut batch, &h, n, &three_frames(n, 40))
            .iter()
            .enumerate()
        {
            let label = format!("size {n}, frame {k}");
            // The same plans as the immediate functions: bit for bit, even the float sum.
            assert_results(&label, got, &immediate(&device, data), 0.0);
            // And the independent host references: the float sum inside the bound of the exact sum.
            assert_results(&label, got, &reference(data), f64::from(f32::MAX));
            assert_f32_sum_by(&label, &data.f, got.sum_f, GPU_C, gpu_bound);
        }
    }
}

#[test]
fn twin_every_op_of_a_batch_equals_the_references_and_the_twin_chain_over_three_frames() {
    #[allow(clippy::arc_with_non_send_sync)]
    let device = Arc::new(CpuDevice::new());
    for n in SIZES {
        let (mut batch, h) = build_cpu(&device, n, 1).unwrap();
        for (k, (data, got)) in run_frames(&mut batch, &h, n, &three_frames(n, 40))
            .iter()
            .enumerate()
        {
            let label = format!("size {n}, frame {k}");
            // The independent reference: the host form is an `f64` accumulation rounded once, so it is that value.
            assert_results(&label, got, &reference(data), 0.0);
            let host_check =
                assert_f32_sum_by(&label, &data.f, got.sum_f, CPU_C, bound::f32_sum_bound);
            // The twin chain of `cpu.rs`: integers exact, the float sums each inside the bound of the exact sum.
            let twin = twin(data);
            let twin_check =
                assert_f32_sum_by(&label, &data.f, twin.sum_f, CPU_C, bound::f32_sum_bound);
            assert_results(&label, got, &twin, host_check.bound + twin_check.bound);
        }
    }
}

/// Four frames of different inputs in a batch of depth 2: two are in flight before either ticket is read, the third and
/// the fourth wait for the oldest, and the tickets are read from the last frame to the first. Each frame's tickets give
/// that frame's own results.
fn frames_in_flight<D: BatchDevice>(
    batch: &mut Batch<D>,
    h: &Handles,
    n: usize,
    check: &dyn Fn(&str, &Data, &Results),
) where
    D::Buffer<u32>: 'static,
    D::Buffer<i32>: 'static,
    D::Buffer<f32>: 'static,
{
    let frames: Vec<Frame> = (0..4).map(|k| Frame::all(n, 70 + k)).collect();
    let pending: Vec<Pending> = frames.iter().map(|frame| submit(batch, h, frame)).collect();
    for (k, (frame, pending)) in frames.iter().zip(pending).enumerate().rev() {
        let mut data = Data::new(0, 0);
        frame.apply(&mut data);
        check(&format!("size {n}, frame {k}"), &data, &pending.resolve());
    }
}

#[test]
fn gpu_two_frames_in_flight_each_equal_their_own_immediate_results() {
    let device = device();
    for n in SIZES {
        let (mut batch, h) = build_gpu(&device, n, 2).unwrap();
        frames_in_flight(&mut batch, &h, n, &|label, data, got| {
            // Kill rule: a mismatch here means a plan reads scratch it did not write in this run (the fix is in the
            // plan, never a ring of buffers, D258).
            assert_results(label, got, &immediate(&device, data), 0.0);
            assert_results(label, got, &reference(data), f64::from(f32::MAX));
        });
    }
}

#[test]
fn twin_two_frames_in_flight_each_equal_their_own_results() {
    #[allow(clippy::arc_with_non_send_sync)]
    let device = Arc::new(CpuDevice::new());
    for n in SIZES {
        let (mut batch, h) = build_cpu(&device, n, 2).unwrap();
        frames_in_flight(&mut batch, &h, n, &|label, data, got| {
            assert_results(label, got, &reference(data), 0.0);
        });
    }
}

#[test]
fn gpu_n_frames_make_n_submissions_after_the_seal() {
    // A device of its own: no other test's submissions are counted.
    let device = Arc::new(HeadlessDevice::new().expect("a Vulkan adapter is required"));
    eprintln!("gust-algo gpu adapter: {:?}", device.adapter_info());
    for n in SIZES {
        // Every plan's scratch is cleared before the seal: the build may submit, the frames may not add to it.
        let (mut batch, h) = build_gpu(&device, n, 1).unwrap();
        let frames = three_frames(n, 90);
        let before = device.gust_submissions();
        for (k, frame) in frames.iter().enumerate() {
            let pending = submit(&mut batch, &h, frame);
            assert_eq!(
                device.gust_submissions() - before,
                k as u64 + 1,
                "size {n}: after frame {k} was submitted"
            );
            pending.resolve();
        }
        assert_eq!(
            device.gust_submissions() - before,
            frames.len() as u64,
            "size {n}: {} frames after the seal",
            frames.len()
        );
    }
}

/// Flags with `set` of every `stride`th position set (`stride == 0`: none).
fn strided(n: usize, stride: usize, offset: usize) -> Vec<u32> {
    (0..n)
        .map(|index| u32::from(stride != 0 && index % stride == offset % stride))
        .collect()
}

/// `compact` over frames whose count shrinks to nothing: the count is whole and the indices are right up to it; the
/// tail of the indices is not read (D259: it holds what the earlier frames left).
fn shrinking_counts<D: BatchDevice>(
    batch: &mut Batch<D>,
    [flags, indices, count]: [Stage<u32, D1>; 3],
    n: usize,
) where
    D::Buffer<u32>: 'static,
{
    let frames = [
        strided(n, 2, 0),
        strided(n, 50, 7),
        strided(n, n, n - 1),
        strided(n, 0, 0),
        strided(n, 3, 1),
    ];
    for (k, written) in frames.iter().enumerate() {
        let mut run = block_on(batch.begin()).unwrap();
        run.write(&flags, written).unwrap();
        let indices_ticket = run.ticket(&indices).unwrap();
        let count_ticket = run.ticket(&count).unwrap();
        let _submitted = run.submit().unwrap();
        let want = reference_indices(written);
        let got_count = one(block_on(count_ticket));
        let got = block_on(indices_ticket).unwrap();
        assert_eq!(got_count as usize, want.len(), "size {n}, frame {k}: count");
        assert_same(
            &got[..got_count as usize],
            &want,
            &format!("size {n}, frame {k}: indices up to the count"),
        );
    }
}

#[test]
fn gpu_compact_over_frames_with_a_shrinking_count() {
    let device = device();
    for n in SIZES {
        let (mut batch, handles) = build_compaction_gpu(&device, n).unwrap();
        shrinking_counts(&mut batch, handles, n);
    }
}

#[test]
fn twin_compact_over_frames_with_a_shrinking_count() {
    #[allow(clippy::arc_with_non_send_sync)]
    let device = Arc::new(CpuDevice::new());
    for n in SIZES {
        let (mut batch, handles) = build_compaction_cpu(&device, n).unwrap();
        shrinking_counts(&mut batch, handles, n);
    }
}

/// Small batches of a device: a wrapping sum, a chain of ops that read each other's outputs, and ops over nothing.
macro_rules! define_small_builders {
    ($wrapping:ident, $chain:ident, $empty:ident, $device:ty) => {
        fn $wrapping(
            device: &Arc<$device>,
            n: usize,
        ) -> Result<(Batch<$device>, [Stage<u32, D1>; 2]), Error> {
            let n = u32::try_from(n).unwrap();
            gust::batch::build("wrapping", device, 1, |b| {
                let u = b.input::<u32, _>(n);
                let wrapping = b.op(&["wrapping"], SumWrapping::of(&u));
                b.expose(&u);
                b.expose(&wrapping);
                [u, wrapping]
            })
        }

        /// `inclusive` is a scan of `u`, `largest` the maximum of that scan and `total` the sum of `u`.
        fn $chain(
            device: &Arc<$device>,
            n: usize,
        ) -> Result<(Batch<$device>, [Stage<u32, D1>; 4]), Error> {
            let n = u32::try_from(n).unwrap();
            gust::batch::build("chain", device, 1, |b| {
                let u = b.input::<u32, _>(n);
                let inclusive = b.op(&["inclusive"], Scan::of(&u));
                let largest = b.op(&["largest"], Max::of(&inclusive));
                let total = b.op(&["total"], Sum::of(&u));
                let handles = [u, inclusive, largest, total];
                for handle in &handles {
                    b.expose(handle);
                }
                handles
            })
        }

        /// One op over an input of zero elements: a sum (0), a minimum (1), a scan (2) or a compaction (3).
        fn $empty(device: &Arc<$device>, kind: usize) -> Result<(), Error> {
            gust::batch::build("nothing", device, 1, |b| {
                let input = b.input::<u32, _>(0);
                match kind {
                    0 => b.expose(&b.op(&["total"], Sum::of(&input))),
                    1 => b.expose(&b.op(&["least"], Min::of(&input))),
                    2 => b.expose(&b.op(&["scanned"], Scan::of(&input))),
                    _ => b.expose(&b.op(&["positions", "n"], Compact::of(&input)).0),
                }
            })
            .map(|_| ())
        }
    };
}

define_small_builders!(
    build_wrapping_gpu,
    build_chain_gpu,
    build_empty_gpu,
    HeadlessDevice
);
define_small_builders!(
    build_wrapping_cpu,
    build_chain_cpu,
    build_empty_cpu,
    CpuDevice
);

/// Write `values` into `input` for one frame and read the `outputs` back.
fn frame_of<D: BatchDevice>(
    batch: &mut Batch<D>,
    input: &Stage<u32, D1>,
    outputs: &[&Stage<u32, D1>],
    values: &[u32],
) -> Vec<Vec<u32>>
where
    D::Buffer<u32>: 'static,
{
    let mut run = block_on(batch.begin()).expect("a frame begins");
    run.write(input, values).unwrap();
    let tickets: Vec<Ticket<u32>> = outputs
        .iter()
        .map(|output| run.ticket(output).unwrap())
        .collect();
    let _submitted = run.submit().expect("the frame is submitted");
    tickets
        .into_iter()
        .map(|ticket| block_on(ticket).expect("the ticket resolves"))
        .collect()
}

/// Values near `u32::MAX`: the sum of a few of them passes 2^32.
fn wide_values(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| u32::MAX - rng.below(1000) as u32).collect()
}

/// `SumWrapping` over data whose exact sum is past 2^32, against the wrapping sum of the host: a saturating sum fails
/// here, and so does a plain one on the CPU twin (the GPU kernels wrap anyway).
fn wrapping_past_the_limit<D: BatchDevice>(
    batch: &mut Batch<D>,
    [u, wrapping]: [Stage<u32, D1>; 2],
    n: usize,
) where
    D::Buffer<u32>: 'static,
{
    for k in 0..3 {
        let values = wide_values(n, 200 + k);
        let exact: u64 = values.iter().map(|value| u64::from(*value)).sum();
        assert!(exact > u64::from(u32::MAX), "the data must overflow 2^32");
        let want = values
            .iter()
            .fold(0u32, |sum, value| sum.wrapping_add(*value));
        let got = frame_of(batch, &u, &[&wrapping], &values);
        assert_eq!(got[0], [want], "size {n}, frame {k}: the wrapping sum");
    }
}

#[test]
fn gpu_the_wrapping_sum_wraps_past_two_to_the_32() {
    let device = device();
    for n in SIZES {
        let (mut batch, handles) = build_wrapping_gpu(&device, n).unwrap();
        wrapping_past_the_limit(&mut batch, handles, n);
    }
}

#[test]
fn twin_the_wrapping_sum_wraps_past_two_to_the_32() {
    #[allow(clippy::arc_with_non_send_sync)]
    let device = Arc::new(CpuDevice::new());
    for n in SIZES {
        let (mut batch, handles) = build_wrapping_cpu(&device, n).unwrap();
        wrapping_past_the_limit(&mut batch, handles, n);
    }
}

/// An op reads another op's output in the same frame: the maximum of an inclusive scan is the sum of the input of this
/// frame, not of the frame before, and the scan's last element is the same number.
fn ops_that_read_ops<D: BatchDevice>(
    batch: &mut Batch<D>,
    [u, inclusive, largest, total]: [Stage<u32, D1>; 4],
    n: usize,
) where
    D::Buffer<u32>: 'static,
{
    for k in 0..3 {
        let values = Data::new(n, 300 + k).u;
        let want = u32::try_from(exact_sum(&values)).unwrap();
        let got = frame_of(batch, &u, &[&largest, &total, &inclusive], &values);
        assert_eq!(
            got[0],
            [want],
            "size {n}, frame {k}: the maximum of the scan"
        );
        assert_eq!(got[1], [want], "size {n}, frame {k}: the sum");
        assert_eq!(got[2].last(), Some(&want), "size {n}, frame {k}: the scan");
    }
}

#[test]
fn gpu_an_op_reads_the_output_of_another_op_in_the_same_frame() {
    let device = device();
    for n in SIZES {
        let (mut batch, handles) = build_chain_gpu(&device, n).unwrap();
        ops_that_read_ops(&mut batch, handles, n);
    }
}

#[test]
fn twin_an_op_reads_the_output_of_another_op_in_the_same_frame() {
    #[allow(clippy::arc_with_non_send_sync)]
    let device = Arc::new(CpuDevice::new());
    for n in SIZES {
        let (mut batch, handles) = build_chain_cpu(&device, n).unwrap();
        ops_that_read_ops(&mut batch, handles, n);
    }
}

/// The interim rule until s2f2b: an op over zero elements is refused when the batch is built, as a device error that
/// names the stage, on every op (no panic).
fn refuses_zero_elements(build: &dyn Fn(usize) -> Result<(), Error>) {
    for (kind, stage) in [(0, "total"), (1, "least"), (2, "scanned"), (3, "positions")] {
        match build(kind) {
            Err(Error::Device(error)) => assert!(
                error.to_string().contains(stage),
                "kind {kind}: the refusal names the stage `{stage}`: {error}"
            ),
            Err(other) => panic!("kind {kind}: expected a device error, got {other}"),
            Ok(()) => panic!("kind {kind}: a batch over zero elements was built"),
        }
    }
}

#[test]
fn gpu_an_op_over_zero_elements_is_refused_with_an_error() {
    let device = device();
    refuses_zero_elements(&|kind| build_empty_gpu(&device, kind));
}

#[test]
fn twin_an_op_over_zero_elements_is_refused_with_an_error() {
    #[allow(clippy::arc_with_non_send_sync)]
    let device = Arc::new(CpuDevice::new());
    refuses_zero_elements(&|kind| build_empty_cpu(&device, kind));
}
