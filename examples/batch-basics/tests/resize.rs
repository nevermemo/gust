//! Resize and derived shapes (m14.s2b): a `set` of a shape parameter between frames resizes what depends on it, with no
//! count node (nothing on the GPU computes a size: the CPU recomputes the derived values at the `set`).
//!
//! The batch has a 2-D input and stage over a `(u32, u32)` parameter `size`, a 2-D stage over a shape derived from it
//! (`size.map(..)`, which reads the bigger 2-D stage and so needs its current width), and a 1-D input, stage and
//! in-place stage over `size.count()`. Every value is small, so a result equals the host reference bit for bit. The
//! `cpu_` tests run on the CPU twin, the `gpu_` ones are `#[ignore]`d like the example's other GPU tests and run in full
//! verification on the adapter the environment selects.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use batch_basics::app::block_on;
use gust::batch::{
    Batch, BatchDevice, D1, D2, Empty, Error, FrameRun, Op, OpDraft, Param, Stage, View,
};
use gust::cpu::CpuDevice;
use gust::map::MapHost;
use gust::{gpu_batch, gpu_task};
use gust_wgpu::{HeadlessDevice, wgpu};

#[derive(Clone, Copy)]
struct Handles {
    size: Param<(u32, u32)>,
    pixels: Stage<u32, D2>,
    flat: Stage<u32, D1>,
    doubled: Stage<u32, D2>,
    small: Stage<u32, D2>,
    scaled: Stage<u32, D1>,
}

macro_rules! define_build {
    ($build:ident, $device:ty, $depth:tt) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles), Error> {
            gpu_batch!("resize", on(device); in_flight($depth); |b| {
                let size = b.param::<(u32, u32)>((4, 2));
                let half = size.map(|(w, h)| -> (u32, u32) { (w / 2, h / 2) });
                let count = size.count();
                let pixels = b.input::<u32>(&size);
                let flat = b.input::<u32>(&count);
                let doubled = gpu_task!(&size, |x, y| -> u32 { pixels[(x, y)] * 2u32 + y * 100u32 + x });
                let small = gpu_task!(&half, |x, y| -> u32 { doubled[(x, y)] + 1u32 });
                let scaled = gpu_task!(&count, |i| -> u32 { flat[i] * 3u32 });
                let _bump = gpu_task!(&mut scaled, |i| { scaled[i] += 5u32; });
                Handles { size, pixels, flat, doubled, small, scaled }
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

/// The input of a frame at this size: the 2-D pixels and the 1-D values.
fn inputs(width: u32, height: u32, seed: u32) -> (Vec<u32>, Vec<u32>) {
    let len = width * height;
    (
        (0..len).map(|i| (i * 7 + seed) % 1000).collect(),
        (0..len).map(|i| (i * 3 + seed) % 1000).collect(),
    )
}

/// The three results of a frame.
#[derive(Debug, PartialEq)]
struct Out {
    doubled: Vec<u32>,
    small: Vec<u32>,
    scaled: Vec<u32>,
}

/// The host reference, written from the rules of the stages (row-major, the width of the bigger stage for `small`).
fn reference(width: u32, height: u32, pixels: &[u32], flat: &[u32]) -> Out {
    let at = |x: u32, y: u32| (y * width + x) as usize;
    let mut doubled = Vec::new();
    for y in 0..height {
        for x in 0..width {
            doubled.push(pixels[at(x, y)] * 2 + y * 100 + x);
        }
    }
    let mut small = Vec::new();
    for y in 0..height / 2 {
        for x in 0..width / 2 {
            small.push(doubled[at(x, y)] + 1);
        }
    }
    Out {
        doubled,
        small,
        scaled: flat.iter().map(|v| v * 3 + 5).collect(),
    }
}

/// One frame: `size` is set first (when given), then the inputs are written (when given), the three results are
/// reserved and the frame is submitted.
fn frame<D: BatchDevice>(
    batch: &mut Batch<D>,
    h: &Handles,
    size: Option<(u32, u32)>,
    input: Option<&(Vec<u32>, Vec<u32>)>,
) -> Out
where
    D::Buffer<u32>: 'static,
{
    block_on(async {
        let mut frame = batch.begin().await?;
        if let Some(size) = size {
            frame.set(&h.size, size)?;
        }
        if let Some((pixels, flat)) = input {
            frame.write(&h.pixels, pixels)?;
            frame.write(&h.flat, flat)?;
        }
        let doubled = frame.ticket(&h.doubled)?;
        let small = frame.ticket(&h.small)?;
        let scaled = frame.ticket(&h.scaled)?;
        let _submitted = frame.submit()?;
        Ok::<_, Error>(Out {
            doubled: doubled.await?,
            small: small.await?,
            scaled: scaled.await?,
        })
    })
    .unwrap()
}

fn dims_of<D: BatchDevice>(batch: &Batch<D>) -> Vec<[u32; 2]> {
    batch.stages().iter().map(|stage| stage.dims).collect()
}

/// Two frames at one size, two at another, a third size, and back to the first: every output equals the host reference
/// whatever the size, a frame after a resize that writes nothing sees zeros (nothing is carried over), and the stages
/// and the memory report follow.
fn resize_across_frames_is_exact<D: BatchDevice>(mut batch: Batch<D>, h: &Handles)
where
    D::Buffer<u32>: 'static,
{
    let a = inputs(4, 2, 1);
    let first = frame(&mut batch, h, None, Some(&a));
    assert_eq!(first, reference(4, 2, &a.0, &a.1), "first frame at 4 x 2");
    let second = frame(&mut batch, h, None, None);
    assert_eq!(second, first, "second frame at 4 x 2 holds the data");
    assert_eq!(batch.generation(), 0, "no resize yet");
    assert_eq!(
        dims_of(&batch),
        [[4, 2], [2, 1], [8, 1], [8, 1]],
        "the stages at 4 x 2"
    );

    let b = inputs(8, 3, 2);
    let third = frame(&mut batch, h, Some((8, 3)), Some(&b));
    assert_eq!(third, reference(8, 3, &b.0, &b.1), "first frame at 8 x 3");
    let moved = batch.generation();
    assert_eq!(moved, 5, "the five buffers were replaced");
    let fourth = frame(&mut batch, h, None, None);
    assert_eq!(fourth, third, "second frame at 8 x 3 holds the data");
    assert_eq!(batch.generation(), moved, "no further resize");
    assert_eq!(
        dims_of(&batch),
        [[8, 3], [4, 1], [24, 1], [24, 1]],
        "the stages at 8 x 3"
    );
    let words = 24 + 24 + 24 + 4 + 24;
    assert_eq!(batch.report().bytes, words * 4, "the memory at 8 x 3");

    let c = inputs(6, 5, 3);
    let fifth = frame(&mut batch, h, Some((6, 5)), Some(&c));
    assert_eq!(fifth, reference(6, 5, &c.0, &c.1), "a third size, 6 x 5");

    let zeros = (vec![0; 8], vec![0; 8]);
    let back = frame(&mut batch, h, Some((4, 2)), None);
    assert_eq!(
        back,
        reference(4, 2, &zeros.0, &zeros.1),
        "back at 4 x 2 the new buffers hold zeros"
    );
    let again = frame(&mut batch, h, None, Some(&a));
    assert_eq!(again, first, "and write what they are given");
}

/// A view taken before a resize keeps the old buffer (old data, old extents, old generation); a view taken after it is
/// the new buffer.
fn a_view_taken_before_a_resize_is_stale_after_it<D: BatchDevice + MapHost>(
    device: &D,
    mut batch: Batch<D>,
    h: &Handles,
) where
    D::Buffer<u32>: 'static,
{
    let a = inputs(4, 2, 5);
    let before = frame(&mut batch, h, None, Some(&a));
    let old: View<D, u32> = batch.view(&h.doubled).unwrap();
    let old_generation = batch.generation();
    assert_eq!((old.width(), old.height(), old.len()), (4, 2, 8));
    assert_eq!(old.generation(), old_generation);

    let b = inputs(8, 3, 6);
    let after = frame(&mut batch, h, Some((8, 3)), Some(&b));
    assert_eq!(after, reference(8, 3, &b.0, &b.1));
    assert_ne!(
        batch.generation(),
        old_generation,
        "a resize moves the generation"
    );
    let new = batch.view(&h.doubled).unwrap();
    assert_eq!((new.width(), new.height(), new.len()), (8, 3, 24));
    assert_eq!(new.generation(), batch.generation());
    assert_eq!(
        (old.width(), old.height(), old.generation()),
        (4, 2, old_generation),
        "the old view keeps what it was taken with"
    );
    assert_eq!(
        device.readback(old.buffer()).unwrap(),
        before.doubled,
        "the old buffer holds the old frame's data"
    );
    assert_eq!(device.readback(new.buffer()).unwrap(), after.doubled);
}

/// A frame that resizes and is dropped changes nothing: the batch has the size, the data and the generation it had.
fn a_dropped_resize_changes_nothing<D: BatchDevice>(mut batch: Batch<D>, h: &Handles)
where
    D::Buffer<u32>: 'static,
{
    let a = inputs(4, 2, 7);
    let first = frame(&mut batch, h, None, Some(&a));
    let before = (dims_of(&batch), batch.report(), batch.generation());
    {
        let mut dropped = block_on(batch.begin()).unwrap();
        dropped.set(&h.size, (8, 3)).unwrap();
        let (pixels, flat) = inputs(8, 3, 9);
        dropped.write(&h.pixels, &pixels).unwrap();
        dropped.write(&h.flat, &flat).unwrap();
        let ticket = dropped.ticket(&h.doubled).unwrap();
        drop(dropped);
        assert!(matches!(block_on(ticket), Err(Error::NotSubmitted { .. })));
    }
    assert_eq!(
        (dims_of(&batch), batch.report(), batch.generation()),
        before
    );
    assert_eq!(
        frame(&mut batch, h, None, None),
        first,
        "the data of 4 x 2 is intact"
    );
}

/// A `set` that would change a buffer the frame already wrote or read is refused and changes nothing; the same shape is
/// not a change; a shape set before the writes is fine.
fn a_shape_set_after_a_write_is_refused<D: BatchDevice>(mut batch: Batch<D>, h: &Handles)
where
    D::Buffer<u32>: 'static,
{
    let a = inputs(4, 2, 11);
    let mut frame_run = block_on(batch.begin()).unwrap();
    frame_run.write(&h.pixels, &a.0).unwrap();
    let error = frame_run.set(&h.size, (8, 3)).unwrap_err();
    assert!(matches!(error, Error::ShapeMismatch { .. }), "{error:?}");
    assert!(
        error.to_string().contains("already written"),
        "the message says why: {error}"
    );
    frame_run.set(&h.size, (4, 2)).unwrap();
    frame_run.write(&h.flat, &a.1).unwrap();
    let doubled = frame_run.ticket(&h.doubled).unwrap();
    let small = frame_run.ticket(&h.small).unwrap();
    let scaled = frame_run.ticket(&h.scaled).unwrap();
    let _submitted = frame_run.submit().unwrap();
    let got = Out {
        doubled: block_on(doubled).unwrap(),
        small: block_on(small).unwrap(),
        scaled: block_on(scaled).unwrap(),
    };
    assert_eq!(got, reference(4, 2, &a.0, &a.1));
    assert_eq!(batch.generation(), 0, "the refused resize changed nothing");
    // After a resize in the same frame the writes follow the new extents, and a wrong length is the old mismatch.
    let mut resized = block_on(batch.begin()).unwrap();
    resized.set(&h.size, (8, 3)).unwrap();
    assert!(matches!(
        resized.write(&h.pixels, &a.0),
        Err(Error::ShapeMismatch { .. })
    ));
    let b = inputs(8, 3, 12);
    resized.write(&h.pixels, &b.0).unwrap();
    resized.write(&h.flat, &b.1).unwrap();
    let doubled = resized.ticket(&h.doubled).unwrap();
    let _submitted = resized.submit().unwrap();
    assert_eq!(
        block_on(doubled).unwrap(),
        reference(8, 3, &b.0, &b.1).doubled
    );
}

/// Setting a shape twice in one frame, the second time back to the first, resizes nothing.
fn a_shape_set_and_set_back_resizes_nothing<D: BatchDevice>(mut batch: Batch<D>, h: &Handles)
where
    D::Buffer<u32>: 'static,
{
    let a = inputs(4, 2, 13);
    let first = frame(&mut batch, h, None, Some(&a));
    let mut frame_run = block_on(batch.begin()).unwrap();
    frame_run.set(&h.size, (8, 3)).unwrap();
    frame_run.set(&h.size, (4, 2)).unwrap();
    let doubled = frame_run.ticket(&h.doubled).unwrap();
    let _submitted = frame_run.submit().unwrap();
    assert_eq!(block_on(doubled).unwrap(), first.doubled);
    assert_eq!(batch.generation(), 0, "back at the shape it had: no resize");
}

/// A frame that resizes while the frame before it is still in flight: the earlier frame's tickets read the old buffers,
/// the new frame's the new ones, and both are exact.
fn a_resize_with_a_frame_in_flight<D: BatchDevice>(mut batch: Batch<D>, h: &Handles)
where
    D::Buffer<u32>: 'static,
{
    let a = inputs(4, 2, 15);
    let b = inputs(8, 3, 16);
    let (early, late) = block_on(async {
        let mut one = batch.begin().await.unwrap();
        one.write(&h.pixels, &a.0).unwrap();
        one.write(&h.flat, &a.1).unwrap();
        let early = one.ticket(&h.doubled).unwrap();
        let _one = one.submit().unwrap();
        let mut two = batch.begin().await.unwrap();
        two.set(&h.size, (8, 3)).unwrap();
        two.write(&h.pixels, &b.0).unwrap();
        two.write(&h.flat, &b.1).unwrap();
        let late = two.ticket(&h.doubled).unwrap();
        let _two = two.submit().unwrap();
        (early.await.unwrap(), late.await.unwrap())
    });
    assert_eq!(early, reference(4, 2, &a.0, &a.1).doubled);
    assert_eq!(late, reference(8, 3, &b.0, &b.1).doubled);
}

/// A frame recorded into the caller's encoder makes the new buffers the batch's before `encode` returns, so a pass the
/// caller records after it fetches the new views and compares the generation.
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

fn an_encoded_resize_moves_the_generation_before_encode_returns<D: Caller>(
    device: &D,
    mut batch: Batch<D>,
    h: &Handles,
) where
    D::Buffer<u32>: 'static,
{
    let a = inputs(4, 2, 17);
    frame(&mut batch, h, None, Some(&a));
    let before = batch.generation();
    let b = inputs(8, 3, 18);
    let mut encoder = BatchDevice::new_encoder(device);
    let mut run: FrameRun<D> = block_on(batch.begin()).unwrap();
    run.set(&h.size, (8, 3)).unwrap();
    run.write(&h.pixels, &b.0).unwrap();
    run.write(&h.flat, &b.1).unwrap();
    let doubled = run.ticket(&h.doubled).unwrap();
    let encoded = run.encode(&mut encoder).unwrap();
    assert!(batch.generation() > before, "moved when encode returned");
    assert_eq!(batch.view(&h.doubled).unwrap().len(), 24);
    let token = device.run_encoder(encoder);
    let _submitted = encoded.submitted(&token);
    assert_eq!(
        block_on(doubled).unwrap(),
        reference(8, 3, &b.0, &b.1).doubled
    );
}

// ---- more rules of a resize ----

/// A `set` of a shape after a `ticket` of a buffer it would change is refused like one after a `write`: the reserved
/// read-back is of the buffer the frame has, and it stays what it was.
fn a_shape_set_after_a_ticket_is_refused<D: BatchDevice>(mut batch: Batch<D>, h: &Handles)
where
    D::Buffer<u32>: 'static,
{
    let a = inputs(4, 2, 19);
    let mut run = block_on(batch.begin()).unwrap();
    let doubled = run.ticket(&h.doubled).unwrap();
    let error = run.set(&h.size, (8, 3)).unwrap_err();
    assert!(matches!(error, Error::ShapeMismatch { .. }), "{error:?}");
    assert!(
        error.to_string().contains("already written or read"),
        "the message says why: {error}"
    );
    run.write(&h.pixels, &a.0).unwrap();
    let _submitted = run.submit().unwrap();
    assert_eq!(
        block_on(doubled).unwrap(),
        reference(4, 2, &a.0, &[0; 8]).doubled
    );
    assert_eq!(batch.generation(), 0, "the refused resize changed nothing");
}

/// A resize to an extent of zero elements: a kernel stage over it has no elements, the others follow their own extents,
/// and a resize back to a size with elements is exact again.
fn a_resize_to_a_zero_extent_is_exact<D: BatchDevice>(mut batch: Batch<D>, h: &Handles)
where
    D::Buffer<u32>: 'static,
{
    let a = inputs(4, 2, 21);
    frame(&mut batch, h, None, Some(&a));
    let one = inputs(1, 1, 22);
    let tiny = frame(&mut batch, h, Some((1, 1)), Some(&one));
    assert_eq!(
        tiny,
        reference(1, 1, &one.0, &one.1),
        "1 x 1 halves to 0 x 0"
    );
    assert!(tiny.small.is_empty());
    assert_eq!(batch.view(&h.small).unwrap().len(), 0);
    let b = inputs(6, 4, 23);
    let back = frame(&mut batch, h, Some((6, 4)), Some(&b));
    assert_eq!(
        back,
        reference(6, 4, &b.0, &b.1),
        "back to a size with elements"
    );
}

/// A frame whose device refuses it after the resize was made leaves the batch as it was, like a dropped resize: the
/// extents, the memory report and the generation, and the next frame runs at the old size.
fn a_frame_the_device_refuses_leaves_the_resize_undone(
    batch: &mut Batch<CpuDevice>,
    h: &RefuseHandles,
) {
    let a = inputs(4, 2, 24);
    let before = (dims_of(batch), batch.report(), batch.generation());
    h.fail.set(true);
    let (pixels, _) = inputs(8, 3, 25);
    let error = block_on(async {
        let mut run = batch.begin().await?;
        run.set(&h.size, (8, 3))?;
        run.write(&h.pixels, &pixels)?;
        run.submit().map(|_| ())
    })
    .unwrap_err();
    assert!(matches!(error, Error::Device(_)), "{error:?}");
    assert_eq!(
        (dims_of(batch), batch.report(), batch.generation()),
        before,
        "the refused frame changed nothing"
    );
    h.fail.set(false);
    let doubled = block_on(async {
        let mut run = batch.begin().await?;
        run.write(&h.pixels, &a.0)?;
        let doubled = run.ticket(&h.doubled)?;
        let _submitted = run.submit()?;
        doubled.await
    })
    .unwrap();
    assert_eq!(doubled, reference(4, 2, &a.0, &a.1).doubled);
}

/// The frame after a frame that resized is one counted submission, like any frame: a resize leaves no work behind. The
/// frame that resizes is at most one submission and one clear per buffer it replaced (a recycled allocation is cleared
/// in a submission of its own).
fn the_frame_after_a_resize_is_one_submission(
    device: &HeadlessDevice,
    mut batch: Batch<HeadlessDevice>,
    h: &Handles,
) {
    let a = inputs(4, 2, 26);
    frame(&mut batch, h, None, Some(&a));
    let b = inputs(8, 3, 27);
    let (started, generation) = (device.gust_submissions(), batch.generation());
    frame(&mut batch, h, Some((8, 3)), Some(&b));
    let resizing = device.gust_submissions() - started;
    let replaced = batch.generation() - generation;
    assert!(
        (1..=1 + replaced).contains(&resizing),
        "the resizing frame took {resizing} submissions, it replaced {replaced} buffers"
    );
    let started = device.gust_submissions();
    let next = frame(&mut batch, h, None, None);
    assert_eq!(device.gust_submissions() - started, 1, "the next frame");
    assert_eq!(next, reference(8, 3, &b.0, &b.1));
}

/// An op that fails on the CPU device when `fail` is set, over a buffer no resize reaches.
struct Refuse<'a>(&'a Stage<u32, D1>, Rc<Cell<bool>>);

impl Op<CpuDevice> for Refuse<'_> {
    type Out = Stage<u32, D1>;

    fn record(self, mut draft: OpDraft<'_, CpuDevice>) -> Self::Out {
        let Some((_, len)) = draft.read(self.0) else {
            return draft.abandon();
        };
        if draft.result::<u32>(len).is_none() {
            return draft.abandon();
        }
        let fail = self.1;
        draft.seal(
            move |device, _| {
                let fail = Rc::clone(&fail);
                Ok(device.prepare_fn(move |_| {
                    if fail.get() {
                        Err(gust::cpu::Error::WrongDevice {
                            parameter: "refused",
                        })
                    } else {
                        Ok(())
                    }
                }))
            },
            Empty::Nothing,
        )
    }
}

struct RefuseHandles {
    size: Param<(u32, u32)>,
    pixels: Stage<u32, D2>,
    doubled: Stage<u32, D2>,
    fail: Rc<Cell<bool>>,
}

fn build_refuse(device: &Arc<CpuDevice>) -> Result<(Batch<CpuDevice>, RefuseHandles), Error> {
    let fail = Rc::new(Cell::new(false));
    gpu_batch!("resize_refuse", on(device); |b| {
        let size = b.param::<(u32, u32)>((4, 2));
        let other = b.param::<u32>(6);
        let guard = b.input::<u32>(&other);
        let _made = b.op(&["refused"], Refuse(&guard, Rc::clone(&fail)));
        let pixels = b.input::<u32>(&size);
        let doubled = gpu_task!(&size, |x, y| -> u32 { pixels[(x, y)] * 2u32 + y * 100u32 + x });
        RefuseHandles { size, pixels, doubled, fail }
    })
}

// ---- the same scenarios on the CPU twin and on the GPU ----

#[test]
fn cpu_a_frame_the_device_refuses_leaves_the_resize_undone() {
    let device = Arc::new(CpuDevice::new());
    let (mut batch, h) = build_refuse(&device).unwrap();
    a_frame_the_device_refuses_leaves_the_resize_undone(&mut batch, &h);
}

#[test]
fn cpu_a_resize_to_a_zero_extent_is_exact() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu(&device).unwrap();
    a_resize_to_a_zero_extent_is_exact(batch, &h);
}

#[test]
fn cpu_a_shape_set_after_a_ticket_is_refused() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu(&device).unwrap();
    a_shape_set_after_a_ticket_is_refused(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_resize_to_a_zero_extent_is_exact() {
    let device = private_device();
    let (batch, h) = build_gpu(&device).unwrap();
    println!("adapter: {}", device.adapter_info().name);
    a_resize_to_a_zero_extent_is_exact(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_shape_set_after_a_ticket_is_refused() {
    let device = private_device();
    let (batch, h) = build_gpu(&device).unwrap();
    println!("adapter: {}", device.adapter_info().name);
    a_shape_set_after_a_ticket_is_refused(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_the_frame_after_a_resize_is_one_submission() {
    let device = private_device();
    let (batch, h) = build_gpu(&device).unwrap();
    println!("adapter: {}", device.adapter_info().name);
    the_frame_after_a_resize_is_one_submission(&device, batch, &h);
}

#[test]
fn cpu_resize_across_frames_is_exact() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu(&device).unwrap();
    resize_across_frames_is_exact(batch, &h);
}

#[test]
fn cpu_a_view_taken_before_a_resize_is_stale_after_it() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu(&device).unwrap();
    a_view_taken_before_a_resize_is_stale_after_it(&*device, batch, &h);
}

#[test]
fn cpu_a_dropped_resize_changes_nothing() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu(&device).unwrap();
    a_dropped_resize_changes_nothing(batch, &h);
}

#[test]
fn cpu_a_shape_set_after_a_write_is_refused() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu(&device).unwrap();
    a_shape_set_after_a_write_is_refused(batch, &h);
}

#[test]
fn cpu_a_shape_set_and_set_back_resizes_nothing() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu(&device).unwrap();
    a_shape_set_and_set_back_resizes_nothing(batch, &h);
}

#[test]
fn cpu_a_resize_with_a_frame_in_flight() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu2(&device).unwrap();
    a_resize_with_a_frame_in_flight(batch, &h);
}

#[test]
fn cpu_an_encoded_resize_moves_the_generation_before_encode_returns() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu(&device).unwrap();
    an_encoded_resize_moves_the_generation_before_encode_returns(&*device, batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_resize_across_frames_is_exact() {
    let device = private_device();
    let (batch, h) = build_gpu(&device).unwrap();
    resize_across_frames_is_exact(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_resize_equals_the_cpu_twin_at_every_size() {
    let gpu = private_device();
    let cpu = Arc::new(CpuDevice::new());
    let (mut on_gpu, gpu_h) = build_gpu(&gpu).unwrap();
    let (mut on_cpu, cpu_h) = build_cpu(&cpu).unwrap();
    println!("adapter: {}", gpu.adapter_info().name);
    for (round, (size, seed)) in [
        (None, 1),
        (Some((8, 3)), 2),
        (Some((6, 5)), 3),
        (Some((2, 2)), 4),
    ]
    .into_iter()
    .enumerate()
    {
        let (width, height) = size.unwrap_or((4, 2));
        let input = inputs(width, height, seed);
        let from_gpu = frame(&mut on_gpu, &gpu_h, size, Some(&input));
        let from_cpu = frame(&mut on_cpu, &cpu_h, size, Some(&input));
        assert_eq!(from_gpu, from_cpu, "round {round}");
        assert_eq!(
            from_gpu,
            reference(width, height, &input.0, &input.1),
            "round {round}"
        );
        assert_eq!(on_gpu.stages(), on_cpu.stages(), "round {round}");
        assert_eq!(on_gpu.report(), on_cpu.report(), "round {round}");
    }
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_view_taken_before_a_resize_is_stale_after_it() {
    let device = private_device();
    let (batch, h) = build_gpu(&device).unwrap();
    a_view_taken_before_a_resize_is_stale_after_it(&*device, batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_dropped_resize_changes_nothing() {
    let device = private_device();
    let (batch, h) = build_gpu(&device).unwrap();
    a_dropped_resize_changes_nothing(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_shape_set_after_a_write_is_refused() {
    let device = private_device();
    let (batch, h) = build_gpu(&device).unwrap();
    a_shape_set_after_a_write_is_refused(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_shape_set_and_set_back_resizes_nothing() {
    let device = private_device();
    let (batch, h) = build_gpu(&device).unwrap();
    a_shape_set_and_set_back_resizes_nothing(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_resize_with_a_frame_in_flight() {
    let device = private_device();
    let (batch, h) = build_gpu2(&device).unwrap();
    a_resize_with_a_frame_in_flight(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_an_encoded_resize_moves_the_generation_before_encode_returns() {
    let device = private_device();
    let (batch, h) = build_gpu(&device).unwrap();
    an_encoded_resize_moves_the_generation_before_encode_returns(&*device, batch, &h);
}
