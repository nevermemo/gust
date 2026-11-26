//! A batch with library op stages follows a resize (m14.s2b2): `sum`, `scan` and `compact` over inputs whose length is a
//! shape parameter, resized up, down, to zero elements and back. The outputs that follow the operand (those of `scan`
//! and `compact`) are replaced and move the generation, the one-element outputs are not, the plan of each length is
//! built inside the `set` (never in the frame) and the batch keeps the plans of the last two lengths.
//!
//! The inputs are small integers, so every sum and prefix is exact and a result equals the host reference bit for bit.
//! Past its count the indices of a compaction are unspecified (D259), so they are compared up to the count. The `cpu_`
//! tests run on the CPU device, the `gpu_` ones are `#[ignore]`d like the example's other GPU tests, print the adapter
//! they ran on and run in full verification.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use batch_basics::app::block_on;
use gust::batch::{
    Batch, BatchDevice, D1, Empty, Error, FrameRun, Op, OpDraft, Param, Stage, View,
    build as build_batch,
};
use gust::cpu::CpuDevice;
use gust::{gpu_batch, gpu_task};
use gust_wgpu::{HeadlessDevice, wgpu};

#[derive(Clone, Copy)]
struct Handles {
    n: Param<u32>,
    values: Stage<u32, D1>,
    flags: Stage<u32, D1>,
    total: Stage<u32, D1>,
    inclusive: Stage<u32, D1>,
    exclusive: Stage<u32, D1>,
    idx: Stage<u32, D1>,
    n_set: Stage<u32, D1>,
    doubled: Stage<u32, D1>,
}

/// The buffers of `Handles` that a resize replaces: both inputs, both scans, the indices and the kernel stage over the
/// scan. `total` and `n_set` have one element whatever the length.
const REPLACED: u64 = 6;

macro_rules! define_build {
    ($build:ident, $device:ty, $depth:tt) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles), Error> {
            gpu_batch!("resize-ops", on(device); in_flight($depth); |b| {
                let n = b.param::<u32>(8);
                let values = b.input::<u32>(&n);
                let flags = b.input::<u32>(&n);
                let total = gust::sum(&values);
                let inclusive = gust::scan(&values);
                let exclusive = gust::scan_exclusive(&values);
                let (idx, n_set) = gust::compact(&flags);
                let doubled = gpu_task!(&n, |i| -> u32 { inclusive[i] * 2u32 });
                Handles { n, values, flags, total, inclusive, exclusive, idx, n_set, doubled }
            })
        }
    };
}

define_build!(build_cpu, CpuDevice, 1);
define_build!(build_cpu2, CpuDevice, 2);
define_build!(build_gpu, HeadlessDevice, 1);
define_build!(build_gpu2, HeadlessDevice, 2);

fn private_device() -> Arc<HeadlessDevice> {
    Arc::new(HeadlessDevice::new().unwrap_or_else(|error| {
        panic!("a GPU adapter is required; this test must not silently skip: {error}")
    }))
}

/// A length that is not a power of two, one that takes two passes of every reduction, a short one, and zero.
const LENGTHS: [u32; 9] = [1000, 100_003, 37, 0, 1000, 0, 8, 100_003, 0];

/// The inputs of a frame at this length.
fn inputs(len: u32, seed: u32) -> (Vec<u32>, Vec<u32>) {
    (
        (0..len).map(|i| (i * 7 + seed) % 1000).collect(),
        (0..len)
            .map(|i| u32::from((i * 5 + seed).is_multiple_of(3)) * 2)
            .collect(),
    )
}

/// What a frame reads back.
#[derive(Debug, PartialEq)]
struct Out {
    total: u32,
    inclusive: Vec<u32>,
    exclusive: Vec<u32>,
    idx: Vec<u32>,
    n_set: u32,
    doubled: Vec<u32>,
}

/// The host reference, written from the rules of the ops.
fn reference(values: &[u32], flags: &[u32]) -> Out {
    let mut sum = 0u32;
    let (mut inclusive, mut exclusive) = (Vec::new(), Vec::new());
    for value in values {
        exclusive.push(sum);
        sum = sum.wrapping_add(*value);
        inclusive.push(sum);
    }
    let idx: Vec<u32> = (0..flags.len() as u32)
        .filter(|&i| flags[i as usize] != 0)
        .collect();
    Out {
        total: sum,
        doubled: inclusive.iter().map(|v| v * 2).collect(),
        inclusive,
        exclusive,
        n_set: idx.len() as u32,
        idx,
    }
}

fn one(values: Result<Vec<u32>, Error>) -> u32 {
    let values = values.expect("the ticket resolves");
    assert_eq!(values.len(), 1, "a one-element output");
    values[0]
}

/// One frame: `len` is set first (when given), the inputs are written (when given), every output is reserved, the frame
/// is submitted and read.
fn frame<D: BatchDevice>(
    batch: &mut Batch<D>,
    h: &Handles,
    len: Option<u32>,
    input: Option<&(Vec<u32>, Vec<u32>)>,
) -> Out
where
    D::Buffer<u32>: 'static,
{
    let mut run = block_on(batch.begin()).unwrap();
    if let Some(len) = len {
        run.set(&h.n, len).unwrap();
    }
    if let Some((values, flags)) = input {
        run.write(&h.values, values).unwrap();
        run.write(&h.flags, flags).unwrap();
    }
    let total = run.ticket(&h.total).unwrap();
    let inclusive = run.ticket(&h.inclusive).unwrap();
    let exclusive = run.ticket(&h.exclusive).unwrap();
    let idx = run.ticket(&h.idx).unwrap();
    let n_set = run.ticket(&h.n_set).unwrap();
    let doubled = run.ticket(&h.doubled).unwrap();
    let _submitted = run.submit().unwrap();
    let n_set = one(block_on(n_set));
    let mut idx = block_on(idx).unwrap();
    idx.truncate(n_set as usize);
    Out {
        total: one(block_on(total)),
        inclusive: block_on(inclusive).unwrap(),
        exclusive: block_on(exclusive).unwrap(),
        idx,
        n_set,
        doubled: block_on(doubled).unwrap(),
    }
}

/// The frames of `LENGTHS`, each at its own length with its own inputs, and the frame after each one that writes
/// nothing: both equal the host reference (a resize holds zeros until the frame writes, and the ops of a zero length
/// write their identities).
fn the_ops_follow_every_resize<D: BatchDevice>(mut batch: Batch<D>, h: &Handles) -> Vec<Out>
where
    D::Buffer<u32>: 'static,
{
    let mut seen = Vec::new();
    for (k, len) in LENGTHS.into_iter().enumerate() {
        let data = inputs(len, k as u32 + 3);
        let got = frame(&mut batch, h, Some(len), Some(&data));
        assert_eq!(got, reference(&data.0, &data.1), "length {len}, frame {k}");
        let again = frame(&mut batch, h, None, None);
        assert_eq!(
            again,
            reference(&data.0, &data.1),
            "length {len}: the frame after, which writes nothing"
        );
        seen.push(got);
    }
    // A resize after a frame that wrote nothing sees zeros, not what the length before left.
    let zeros = frame(&mut batch, h, Some(50), None);
    assert_eq!(zeros, reference(&[0; 50], &[0; 50]), "zeros after a resize");
    seen
}

/// A resize replaces the buffers that follow the operand and moves the generation by their number, not by the
/// one-element outputs; the batch's views follow.
fn a_resize_replaces_the_outputs_that_follow_the_operand<D: BatchDevice>(
    mut batch: Batch<D>,
    h: &Handles,
) where
    D::Buffer<u32>: 'static,
{
    let data = inputs(8, 1);
    frame(&mut batch, h, None, Some(&data));
    let old: View<D, u32> = batch.view(&h.inclusive).unwrap();
    let before = batch.generation();
    assert_eq!(old.len(), 8);
    let data = inputs(40, 2);
    let got = frame(&mut batch, h, Some(40), Some(&data));
    assert_eq!(got, reference(&data.0, &data.1));
    assert_eq!(batch.generation() - before, REPLACED);
    let new = batch.view(&h.inclusive).unwrap();
    assert_eq!(new.len(), 40);
    assert_eq!(new.generation(), batch.generation());
    assert_eq!(old.len(), 8, "the old view keeps what it was taken with");
    assert_eq!(batch.view(&h.idx).unwrap().len(), 40);
    assert_eq!(batch.view(&h.total).unwrap().len(), 1);
    assert_eq!(batch.view(&h.n_set).unwrap().len(), 1);
}

/// A frame that resizes while the frame before it is still in flight: the earlier frame's tickets read the buffers and
/// the plan of the old length, the new frame's the new ones, and both are exact.
fn an_op_resize_with_a_frame_in_flight<D: BatchDevice>(mut batch: Batch<D>, h: &Handles)
where
    D::Buffer<u32>: 'static,
{
    let (a, b) = (inputs(8, 4), inputs(1000, 5));
    let (early, late) = block_on(async {
        let mut first = batch.begin().await.unwrap();
        first.write(&h.values, &a.0).unwrap();
        first.write(&h.flags, &a.1).unwrap();
        let early = (
            first.ticket(&h.total).unwrap(),
            first.ticket(&h.inclusive).unwrap(),
            first.ticket(&h.idx).unwrap(),
            first.ticket(&h.n_set).unwrap(),
        );
        let _first = first.submit().unwrap();
        let mut second = batch.begin().await.unwrap();
        second.set(&h.n, 1000).unwrap();
        second.write(&h.values, &b.0).unwrap();
        second.write(&h.flags, &b.1).unwrap();
        let late = (
            second.ticket(&h.total).unwrap(),
            second.ticket(&h.inclusive).unwrap(),
            second.ticket(&h.idx).unwrap(),
            second.ticket(&h.n_set).unwrap(),
        );
        let _second = second.submit().unwrap();
        let early = (
            one(early.0.await),
            early.1.await.unwrap(),
            early.2.await.unwrap(),
            one(early.3.await),
        );
        let late = (
            one(late.0.await),
            late.1.await.unwrap(),
            late.2.await.unwrap(),
            one(late.3.await),
        );
        (early, late)
    });
    for (got, data) in [(early, &a), (late, &b)] {
        let want = reference(&data.0, &data.1);
        let (total, inclusive, mut idx, n_set) = got;
        idx.truncate(n_set as usize);
        assert_eq!(
            (total, inclusive, idx, n_set),
            (want.total, want.inclusive, want.idx, want.n_set)
        );
    }
}

/// The frame of a resize recorded into the caller's encoder: the plan is built in the `set`, the new buffers are the
/// batch's when `encode` returns, and the results are exact.
trait Caller: BatchDevice {
    fn run_encoder(&self, encoder: Self::Encoder) -> Self::Token;
}

impl Caller for HeadlessDevice {
    fn run_encoder(&self, encoder: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
        self.wgpu_queue().submit([encoder.finish()])
    }
}

impl Caller for CpuDevice {
    fn run_encoder(&self, encoder: gust::cpu::CpuEncoder) -> gust::cpu::CpuJob {
        encoder.run(self).unwrap()
    }
}

fn an_encoded_op_resize_is_exact<D: Caller>(device: &D, mut batch: Batch<D>, h: &Handles)
where
    D::Buffer<u32>: 'static,
{
    let a = inputs(8, 6);
    frame(&mut batch, h, None, Some(&a));
    for (len, seed) in [(1000u32, 7u32), (0, 8), (37, 9)] {
        let data = inputs(len, seed);
        let before = batch.generation();
        let mut encoder = BatchDevice::new_encoder(device);
        let mut run: FrameRun<D> = block_on(batch.begin()).unwrap();
        run.set(&h.n, len).unwrap();
        run.write(&h.values, &data.0).unwrap();
        run.write(&h.flags, &data.1).unwrap();
        let total = run.ticket(&h.total).unwrap();
        let inclusive = run.ticket(&h.inclusive).unwrap();
        let idx = run.ticket(&h.idx).unwrap();
        let n_set = run.ticket(&h.n_set).unwrap();
        let encoded = run.encode(&mut encoder).unwrap();
        assert!(batch.generation() > before, "moved when encode returned");
        assert_eq!(batch.view(&h.inclusive).unwrap().len(), len as usize);
        let token = device.run_encoder(encoder);
        let _submitted = encoded.submitted(&token);
        let want = reference(&data.0, &data.1);
        let n_set = one(block_on(n_set));
        let mut idx = block_on(idx).unwrap();
        idx.truncate(n_set as usize);
        assert_eq!(one(block_on(total)), want.total, "length {len}");
        assert_eq!(block_on(inclusive).unwrap(), want.inclusive, "length {len}");
        assert_eq!((idx, n_set), (want.idx, want.n_set), "length {len}");
    }
}

// ---- the rules of `min` and `max` over nothing, through a resize ----

#[derive(Clone, Copy)]
struct Bounds {
    n: Param<u32>,
    values: Stage<u32, D1>,
    least: Stage<u32, D1>,
    most: Stage<u32, D1>,
}

macro_rules! define_bounds {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Bounds), Error> {
            gpu_batch!("resize-bounds", on(device); in_flight(1); |b| {
                let n = b.param::<u32>(5);
                let values = b.input::<u32>(&n);
                let least = gust::min(&values);
                let most = gust::max(&values);
                Bounds { n, values, least, most }
            })
        }
    };
}

define_bounds!(bounds_cpu, CpuDevice);
define_bounds!(bounds_gpu, HeadlessDevice);

/// `min` and `max` have no value over nothing: a frame resized to zero elements is refused with `Error::Empty` before
/// anything is encoded, the refused frame changes nothing, and a resize back to some elements is exact again.
fn a_min_over_nothing_refuses_the_frame_and_the_batch_recovers<D: BatchDevice>(
    mut batch: Batch<D>,
    h: &Bounds,
) where
    D::Buffer<u32>: 'static,
{
    let read = |batch: &mut Batch<D>, len: Option<u32>, data: Option<&[u32]>| {
        let mut run = block_on(batch.begin()).unwrap();
        if let Some(len) = len {
            run.set(&h.n, len).unwrap();
        }
        if let Some(data) = data {
            run.write(&h.values, data).unwrap();
        }
        let (least, most) = (run.ticket(&h.least).unwrap(), run.ticket(&h.most).unwrap());
        run.submit().map(|_submitted| (least, most))
    };
    let first = [9, 4, 7, 4, 8];
    let (least, most) = read(&mut batch, None, Some(&first)).unwrap();
    assert_eq!((one(block_on(least)), one(block_on(most))), (4, 9));
    let generation = batch.generation();
    let Err(Error::Empty { stage, .. }) = read(&mut batch, Some(0), Some(&[])) else {
        panic!("a frame resized to nothing under a `min` is refused");
    };
    assert!(["least", "most"].contains(&stage.as_str()), "{stage}");
    assert_eq!(
        batch.generation(),
        generation,
        "the refused frame changed nothing"
    );
    let (least, most) = read(&mut batch, None, None).unwrap();
    assert_eq!(
        (one(block_on(least)), one(block_on(most))),
        (4, 9),
        "the batch runs at the old length with its data"
    );
    let second = [30, 12, 55];
    let (least, most) = read(&mut batch, Some(3), Some(&second)).unwrap();
    assert_eq!((one(block_on(least)), one(block_on(most))), (12, 55));
}

// ---- the plans of an op stage, counted ----

/// An op of one result as long as its operand that adds the length it was planned for to every element, and counts its
/// plans: a stage prepared over a plan of another length gives other numbers, and `builds` says when a plan was made.
struct Counted<'a> {
    input: &'a Stage<u32, D1>,
    builds: Rc<Cell<usize>>,
    fail: Rc<Cell<bool>>,
}

impl Op<CpuDevice> for Counted<'_> {
    type Out = Stage<u32, D1>;

    fn record(self, mut draft: OpDraft<'_, CpuDevice>) -> Self::Out {
        if draft.read(self.input).is_none() || draft.result_like::<u32>().is_none() {
            return draft.abandon();
        }
        let (builds, fail) = (self.builds, self.fail);
        draft.seal_planned(
            move |_, len| {
                builds.set(builds.get() + 1);
                Ok(u32::try_from(len).expect("a short test length"))
            },
            move |device, plan, operands| {
                let (input, out, tag) =
                    (operands.read::<u32>(0), operands.result::<u32>(0), **plan);
                let fail = Rc::clone(&fail);
                Ok(device.prepare_fn(move |_| {
                    if fail.get() {
                        return Err(gust::cpu::Error::WrongDevice {
                            parameter: "refused",
                        });
                    }
                    let values: Vec<u32> = input.read().iter().map(|v| v + tag).collect();
                    out.write(&values)
                }))
            },
            Empty::Nothing,
        )
    }
}

struct CountedHandles {
    n: Param<u32>,
    x: Stage<u32, D1>,
    y: Stage<u32, D1>,
    builds: Rc<Cell<usize>>,
    fail: Rc<Cell<bool>>,
}

fn build_counted(device: &Arc<CpuDevice>) -> Result<(Batch<CpuDevice>, CountedHandles), Error> {
    let (builds, fail) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(false)));
    let (counted, refused) = (Rc::clone(&builds), Rc::clone(&fail));
    build_batch("resize-counted", device, 1, |b| {
        let n = b.param::<u32>(4);
        let x = b.input::<u32, _>(&n);
        let y = b.op(
            &["y"],
            Counted {
                input: &x,
                builds: counted,
                fail: refused,
            },
        );
        b.expose(&x);
        b.expose(&y);
        CountedHandles {
            n,
            x,
            y,
            builds,
            fail,
        }
    })
}

/// One frame of the counted batch at `len` (when given): `x` is written with `1..=len` and `y` is read.
fn counted_frame(
    batch: &mut Batch<CpuDevice>,
    h: &CountedHandles,
    len: Option<u32>,
) -> Result<Vec<u32>, Error> {
    let mut run = block_on(batch.begin())?;
    if let Some(len) = len {
        run.set(&h.n, len)?;
        run.write(&h.x, &(1..=len).collect::<Vec<_>>())?;
    }
    let y = run.ticket(&h.y)?;
    let _submitted = run.submit()?;
    block_on(y)
}

/// The plan of a length is built once when the batch alternates between two lengths, a third length drops the oldest
/// one, and each result is that of the plan of its own length.
#[test]
fn cpu_a_plan_is_kept_for_the_last_two_lengths() {
    let device = Arc::new(CpuDevice::new());
    let (mut batch, h) = build_counted(&device).unwrap();
    let expect = |len: u32| -> Vec<u32> { (1..=len).map(|v| v + len).collect() };
    assert_eq!(h.builds.get(), 1, "the seal builds the first plan");
    // 4 (sealed), 6, 4, 6, 4: two builds in all after the seal's.
    for (len, builds) in [(4, 1), (6, 2), (4, 2), (6, 2), (4, 2)] {
        let got = counted_frame(&mut batch, &h, Some(len)).unwrap();
        assert_eq!(got, expect(len), "length {len}");
        assert_eq!(h.builds.get(), builds, "after length {len}");
    }
    // A third length drops the plan of the oldest: 9 is built and 6 goes, 4 is still kept, and 6 is built again.
    for (len, builds) in [(9, 3), (4, 3), (6, 4)] {
        let got = counted_frame(&mut batch, &h, Some(len)).unwrap();
        assert_eq!(got, expect(len), "length {len}");
        assert_eq!(h.builds.get(), builds, "after length {len}");
    }
    // A length of zero builds nothing, runs the `Empty` rule (nothing) and comes back to the kept plan.
    let got = counted_frame(&mut batch, &h, Some(0)).unwrap();
    assert!(got.is_empty());
    assert_eq!(h.builds.get(), 4, "a length of zero builds no plan");
    let got = counted_frame(&mut batch, &h, Some(6)).unwrap();
    assert_eq!(got, expect(6));
    assert_eq!(
        h.builds.get(),
        4,
        "the plan of 6 was kept through the empty frame"
    );
}

/// A frame the device refuses after the `set` built a plan for the new length leaves the batch with the old plan: the
/// next frame runs at the old length with the old plan, and the plan the refused frame built was not kept.
#[test]
fn cpu_a_failed_frame_after_a_staged_op_resize_leaves_the_old_plan() {
    let device = Arc::new(CpuDevice::new());
    let (mut batch, h) = build_counted(&device).unwrap();
    let ok = counted_frame(&mut batch, &h, Some(4)).unwrap();
    assert_eq!(ok, vec![5, 6, 7, 8]);
    let generation = batch.generation();
    h.fail.set(true);
    let error = counted_frame(&mut batch, &h, Some(6)).unwrap_err();
    assert!(matches!(error, Error::Device(_)), "{error:?}");
    assert_eq!(h.builds.get(), 2, "the set built the plan of 6");
    assert_eq!(
        batch.generation(),
        generation,
        "the refused frame changed nothing"
    );
    h.fail.set(false);
    let got = counted_frame(&mut batch, &h, None).unwrap();
    assert_eq!(
        got,
        vec![5, 6, 7, 8],
        "the old length, the old plan, the data kept"
    );
    assert_eq!(batch.view(&h.y).unwrap().len(), 4);
    let got = counted_frame(&mut batch, &h, Some(6)).unwrap();
    assert_eq!(got, vec![7, 8, 9, 10, 11, 12]);
    assert_eq!(
        h.builds.get(),
        3,
        "the plan of the refused frame was not kept"
    );
}

// ---- the same scenarios on the CPU device and on the GPU ----

#[test]
fn cpu_the_ops_follow_every_resize() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu(&device).unwrap();
    the_ops_follow_every_resize(batch, &h);
}

#[test]
fn cpu_a_resize_replaces_the_outputs_that_follow_the_operand() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu(&device).unwrap();
    a_resize_replaces_the_outputs_that_follow_the_operand(batch, &h);
}

#[test]
fn cpu_an_op_resize_with_a_frame_in_flight() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu2(&device).unwrap();
    an_op_resize_with_a_frame_in_flight(batch, &h);
}

#[test]
fn cpu_an_encoded_op_resize_is_exact() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu(&device).unwrap();
    an_encoded_op_resize_is_exact(&*device, batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_an_op_resize_with_a_frame_in_flight() {
    let device = private_device();
    println!("adapter: {}", device.adapter_info().name);
    let (batch, h) = build_gpu2(&device).unwrap();
    an_op_resize_with_a_frame_in_flight(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_an_encoded_op_resize_is_exact() {
    let device = private_device();
    println!("adapter: {}", device.adapter_info().name);
    let (batch, h) = build_gpu(&device).unwrap();
    an_encoded_op_resize_is_exact(&*device, batch, &h);
}

#[test]
fn cpu_a_min_over_nothing_refuses_the_frame_and_the_batch_recovers() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = bounds_cpu(&device).unwrap();
    a_min_over_nothing_refuses_the_frame_and_the_batch_recovers(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_the_ops_follow_every_resize_and_equal_the_cpu_twin() {
    let device = private_device();
    println!("adapter: {}", device.adapter_info().name);
    let (batch, h) = build_gpu(&device).unwrap();
    let on_gpu = the_ops_follow_every_resize(batch, &h);
    let twin_device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu(&twin_device).unwrap();
    let on_twin = the_ops_follow_every_resize(batch, &h);
    assert_eq!(
        on_gpu, on_twin,
        "the GPU equals the CPU twin at every length"
    );
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_resize_replaces_the_outputs_that_follow_the_operand() {
    let device = private_device();
    println!("adapter: {}", device.adapter_info().name);
    let (batch, h) = build_gpu(&device).unwrap();
    a_resize_replaces_the_outputs_that_follow_the_operand(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_min_over_nothing_refuses_the_frame_and_the_batch_recovers() {
    let device = private_device();
    println!("adapter: {}", device.adapter_info().name);
    let (batch, h) = bounds_gpu(&device).unwrap();
    a_min_over_nothing_refuses_the_frame_and_the_batch_recovers(batch, &h);
}

/// The frame of a resize, counted. The `set` may take submissions of its own (the new zeroed buffers, and the scratch of
/// a plan that was not kept), the `submit` of the frame is one submission, and so is the frame after it: a resize leaves
/// no work behind. Going back to a length whose plan was kept builds no plan.
#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_the_frame_after_an_op_resize_is_one_submission() {
    let device = private_device();
    println!("adapter: {}", device.adapter_info().name);
    let (mut batch, h) = build_gpu(&device).unwrap();
    frame(&mut batch, &h, None, None);
    for (k, len) in [100_003u32, 1000, 100_003, 37, 1000]
        .into_iter()
        .enumerate()
    {
        let data = inputs(len, k as u32 + 9);
        let mut run = block_on(batch.begin()).unwrap();
        let before_set = device.gust_submissions();
        run.set(&h.n, len).unwrap();
        let set = device.gust_submissions() - before_set;
        if k == 2 {
            // 100003 again, after 100003 and 1000: its plan was kept, so the `set` builds none and takes only the
            // clears of the buffers it replaced.
            assert_eq!(set, REPLACED, "a set back to a kept length");
        }
        run.write(&h.values, &data.0).unwrap();
        run.write(&h.flags, &data.1).unwrap();
        let total = run.ticket(&h.total).unwrap();
        let inclusive = run.ticket(&h.inclusive).unwrap();
        let before_submit = device.gust_submissions();
        let _submitted = run.submit().unwrap();
        assert_eq!(
            device.gust_submissions() - before_submit,
            1,
            "the frame that resized to {len}, after a set of {set} submissions"
        );
        let want = reference(&data.0, &data.1);
        assert_eq!(one(block_on(total)), want.total);
        assert_eq!(block_on(inclusive).unwrap(), want.inclusive);
        let started = device.gust_submissions();
        let next = frame(&mut batch, &h, None, None);
        assert_eq!(device.gust_submissions() - started, 1, "the next frame");
        assert_eq!(next, want);
    }
}

/// A back-and-forth resize, A, B, A, B, ...: every frame is exact and one submission, and the `set` costs are printed
/// (the plan-cache measurement of slice m14.s2b2: run with `--nocapture`).
#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_back_and_forth_resize_is_exact() {
    let device = private_device();
    println!("adapter: {}", device.adapter_info().name);
    let (mut batch, h) = build_gpu(&device).unwrap();
    let lengths = [100_003u32, 1000];
    let (mut set_time, mut set_submissions) = ([0.0f64; 2], [0u64; 2]);
    let rounds = 12;
    for round in 0..rounds {
        for (which, len) in lengths.into_iter().enumerate() {
            let data = inputs(len, round);
            let mut run = block_on(batch.begin()).unwrap();
            let (before, started) = (device.gust_submissions(), Instant::now());
            run.set(&h.n, len).unwrap();
            if round >= 1 {
                // Both lengths were met in round 0 and their plans kept: a `set` takes the clears of the replaced
                // buffers and nothing more (a plan's scratch would add its own).
                assert_eq!(
                    device.gust_submissions() - before,
                    REPLACED,
                    "a set to {len} over a kept plan"
                );
            }
            if round >= 2 {
                set_time[which] += started.elapsed().as_secs_f64();
                set_submissions[which] += device.gust_submissions() - before;
            }
            run.write(&h.values, &data.0).unwrap();
            run.write(&h.flags, &data.1).unwrap();
            let total = run.ticket(&h.total).unwrap();
            let idx = run.ticket(&h.idx).unwrap();
            let n_set = run.ticket(&h.n_set).unwrap();
            let before_submit = device.gust_submissions();
            let _submitted = run.submit().unwrap();
            assert_eq!(device.gust_submissions() - before_submit, 1, "the frame");
            let want = reference(&data.0, &data.1);
            assert_eq!(one(block_on(total)), want.total);
            let count = one(block_on(n_set));
            assert_eq!(count, want.n_set);
            let mut got = block_on(idx).unwrap();
            got.truncate(count as usize);
            assert_eq!(got, want.idx);
        }
    }
    let counted = f64::from(rounds - 2);
    for (which, len) in lengths.into_iter().enumerate() {
        println!(
            "resize to {len}: {:.3} ms per set, {:.1} submissions per set (rounds 2..{rounds})",
            1000.0 * set_time[which] / counted,
            set_submissions[which] as f64 / counted
        );
    }
}
