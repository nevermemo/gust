//! A sealed frame recorded into a caller's encoder (m14.s2a2e): `FrameRun::encode`, the caller's own submission, and
//! `Encoded::submitted`.
//!
//! The batch is `y = x * 3 + gain`, then `y = y * 2 + 1` in place, over `u32` values small enough never to overflow
//! (the CPU twin checks overflow), so a result equals the host reference bit for bit. The GPU tests are `#[ignore]`d like the example's other GPU tests and run in full
//! verification, on the adapter the environment selects. The twin tests run the same refusals on the CPU device, whose
//! "encoder" is a list of steps that the caller runs.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use batch_basics::app::{block_on, ready_now};
use gust::batch::{Batch, BatchDevice, D1, Error, FrameRun, Param, Stage};
use gust::cpu::CpuDevice;
use gust::{gpu_batch, gpu_task};
use gust_wgpu::{HeadlessDevice, wgpu};

const N: u32 = 1 << 12;

struct Handles {
    gain: Param<u32>,
    x: Stage<u32, D1>,
    y: Stage<u32, D1>,
}

macro_rules! define_build {
    ($build:ident, $device:ty, $depth:tt) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles), Error> {
            gpu_batch!("encoded", on(device); in_flight($depth); |b| {
                let gain = b.param::<u32>(1);
                let x = b.input::<u32>(N);
                let y = gpu_task!(N, |i| -> u32 { x[i] * 3u32 + gain.get() });
                let _double = gpu_task!(&mut y, |i| { y[i] = y[i] * 2u32 + 1u32; });
                Handles { gain, x, y }
            })
        }
    };
}

define_build!(build_cpu, CpuDevice, 1);
define_build!(build_cpu2, CpuDevice, 2);
define_build!(build_gpu, HeadlessDevice, 1);
define_build!(build_gpu2, HeadlessDevice, 2);

fn data(seed: u32) -> Vec<u32> {
    (0..N).map(|i| (i * 7 + seed) % 100_000).collect()
}

fn reference(x: &[u32], gain: u32) -> Vec<u32> {
    x.iter().map(|v| (v * 3 + gain) * 2 + 1).collect()
}

fn private_device() -> Arc<HeadlessDevice> {
    Arc::new(HeadlessDevice::new().unwrap_or_else(|error| {
        panic!("a GPU adapter is required; this test must not silently skip: {error}")
    }))
}

fn gpu_encoder(device: &HeadlessDevice) -> wgpu::CommandEncoder {
    device
        .wgpu_device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
}

/// The caller's own submission, which GUST does not count.
fn submit_caller(device: &HeadlessDevice, encoder: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
    device.wgpu_queue().submit([encoder.finish()])
}

// ---- the encode form equals `submit()` and the CPU twin ----

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_encode_then_submitted_equals_submit_and_the_twin() {
    let device = private_device();
    let (mut batch, h) = build_gpu(&device).unwrap();
    let x = data(1);

    // The frame as `submit()` runs it.
    let mut frame = block_on(batch.begin()).unwrap();
    frame.write(&h.x, &x).unwrap();
    frame.set(&h.gain, 5).unwrap();
    let by_submit = frame.ticket(&h.y).unwrap();
    let _submitted = frame.submit().unwrap();
    let by_submit = block_on(by_submit).unwrap();
    assert_eq!(by_submit, reference(&x, 5));

    // The same frame recorded into the caller's encoder, after a pass of the caller's own: one submission carries both.
    let scratch = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu_encoder(&device);
    encoder.clear_buffer(&scratch, 0, None);
    let mut frame = block_on(batch.begin()).unwrap();
    frame.write(&h.x, &x).unwrap();
    frame.set(&h.gain, 5).unwrap();
    let by_encode = frame.ticket(&h.y).unwrap();
    let before = device.gust_submissions();
    let encoded = frame.encode(&mut encoder).unwrap();
    let index = submit_caller(&device, encoder);
    let _submitted = encoded.submitted(&index);
    assert_eq!(
        device.gust_submissions(),
        before,
        "the encode form makes no GUST submission"
    );
    let by_encode = block_on(by_encode).unwrap();
    assert_eq!(by_encode, by_submit, "encode equals submit");

    // And the CPU twin, through its own encoder.
    let cpu = Arc::new(CpuDevice::new());
    let (mut twin, hc) = build_cpu(&cpu).unwrap();
    let mut encoder = BatchDevice::new_encoder(&*cpu);
    let mut frame = block_on(twin.begin()).unwrap();
    frame.write(&hc.x, &x).unwrap();
    frame.set(&hc.gain, 5).unwrap();
    let by_twin = frame.ticket(&hc.y).unwrap();
    let encoded = frame.encode(&mut encoder).unwrap();
    let job = encoder.run(&cpu).unwrap();
    let _submitted = encoded.submitted(&job);
    assert_eq!(block_on(by_twin).unwrap(), by_encode, "twin equals GPU");
}

/// With two frames in flight, each is encoded and reported in turn; the frames run in the order of the submissions.
#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_encode_works_for_each_frame_in_flight() {
    let device = private_device();
    let (mut batch, h) = build_gpu2(&device).unwrap();
    let x = data(2);
    let mut tickets = Vec::new();
    for k in 0..5_u32 {
        let mut encoder = gpu_encoder(&device);
        let mut frame = block_on(batch.begin()).unwrap();
        if k == 0 {
            frame.write(&h.x, &x).unwrap();
        }
        frame.set(&h.gain, 10 + k).unwrap();
        tickets.push((frame.ticket(&h.y).unwrap(), 10 + k));
        let encoded = frame.encode(&mut encoder).unwrap();
        let index = submit_caller(&device, encoder);
        let _submitted = encoded.submitted(&index);
    }
    for (k, (ticket, gain)) in tickets.into_iter().enumerate().rev() {
        assert_eq!(block_on(ticket).unwrap(), reference(&x, gain), "frame {k}");
    }
}

// ---- refusals: NotSubmitted, never stale bytes, with no later submission ----

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_the_three_refusals_are_not_submitted() {
    let device = private_device();
    let (mut batch, h) = build_gpu2(&device).unwrap();
    let x = data(3);
    // Earlier frames leave other bytes in the staging buffers the pool hands out again.
    for gain in [1, 2] {
        let mut frame = block_on(batch.begin()).unwrap();
        frame.write(&h.x, &x).unwrap();
        frame.set(&h.gain, gain).unwrap();
        let ticket = frame.ticket(&h.y).unwrap();
        let _s = frame.submit().unwrap();
        assert_eq!(block_on(ticket).unwrap(), reference(&x, gain));
    }
    let before = device.gust_submissions();
    let retired = device.buffer_pool_stats().retired;

    // 1. A dropped FrameRun.
    let ticket = {
        let mut frame = block_on(batch.begin()).unwrap();
        frame.set(&h.gain, 7).unwrap();
        frame.ticket(&h.y).unwrap()
    };
    assert!(matches!(block_on(ticket), Err(Error::NotSubmitted { .. })));

    // 2. A dropped Encoded, and its encoder with it: nothing of the frame runs, and the staging buffer of its ticket
    // is retired, never given back to the pool.
    let mut encoder = gpu_encoder(&device);
    let mut frame = block_on(batch.begin()).unwrap();
    frame.set(&h.gain, 8).unwrap();
    let ticket = frame.ticket(&h.y).unwrap();
    drop(frame.encode(&mut encoder).unwrap());
    drop(encoder);
    assert!(matches!(block_on(ticket), Err(Error::NotSubmitted { .. })));
    assert_eq!(
        device.buffer_pool_stats().retired,
        retired + 1,
        "the dropped Encoded retired its ticket's staging buffer, the dropped FrameRun none"
    );

    // 3. An Encoded marked submitted whose encoder was dropped: the index is of another submission.
    let unrelated = submit_caller(&device, gpu_encoder(&device));
    let mut encoder = gpu_encoder(&device);
    let mut frame = block_on(batch.begin()).unwrap();
    frame.set(&h.gain, 9).unwrap();
    let ticket = frame.ticket(&h.y).unwrap();
    let encoded = frame.encode(&mut encoder).unwrap();
    drop(encoder);
    let _submitted = encoded.submitted(&unrelated);
    assert!(
        matches!(block_on(ticket), Err(Error::NotSubmitted { .. })),
        "a ticket of an encoder that never ran must not return the bytes of an earlier frame"
    );
    assert_eq!(
        device.buffer_pool_stats().retired,
        retired + 2,
        "a reported frame whose encoder never ran retired its ticket's staging buffer"
    );
    assert_eq!(
        device.gust_submissions(),
        before,
        "none of the refusals made a GUST submission"
    );

    // The next frame starts from the batch as the last reported frame left it, and is exact.
    let mut frame = block_on(batch.begin()).unwrap();
    frame.set(&h.gain, 11).unwrap();
    let ticket = frame.ticket(&h.y).unwrap();
    let _s = frame.submit().unwrap();
    assert_eq!(block_on(ticket).unwrap(), reference(&x, 11));
}

#[test]
fn twin_the_three_refusals_are_not_submitted() {
    let cpu = Arc::new(CpuDevice::new());
    let (mut batch, h) = build_cpu2(&cpu).unwrap();
    let x = data(4);
    let mut frame = block_on(batch.begin()).unwrap();
    frame.write(&h.x, &x).unwrap();
    frame.set(&h.gain, 1).unwrap();
    let ticket = frame.ticket(&h.y).unwrap();
    let _s = frame.submit().unwrap();
    assert_eq!(block_on(ticket).unwrap(), reference(&x, 1));

    // 1. A dropped FrameRun.
    let ticket = {
        let mut frame = block_on(batch.begin()).unwrap();
        frame.set(&h.gain, 7).unwrap();
        frame.ticket(&h.y).unwrap()
    };
    assert!(matches!(block_on(ticket), Err(Error::NotSubmitted { .. })));

    // 2. A dropped Encoded.
    let mut encoder = BatchDevice::new_encoder(&*cpu);
    let mut frame = block_on(batch.begin()).unwrap();
    frame.set(&h.gain, 8).unwrap();
    let ticket = frame.ticket(&h.y).unwrap();
    drop(frame.encode(&mut encoder).unwrap());
    assert!(matches!(block_on(ticket), Err(Error::NotSubmitted { .. })));

    // 3. Marked submitted, the encoder never run: the job is of an empty encoder.
    let job = BatchDevice::new_encoder(&*cpu).run(&cpu).unwrap();
    let mut encoder = BatchDevice::new_encoder(&*cpu);
    let mut frame = block_on(batch.begin()).unwrap();
    frame.set(&h.gain, 9).unwrap();
    let ticket = frame.ticket(&h.y).unwrap();
    let encoded = frame.encode(&mut encoder).unwrap();
    drop(encoder);
    let _submitted = encoded.submitted(&job);
    assert!(matches!(block_on(ticket), Err(Error::NotSubmitted { .. })));

    // The next frame is exact, and a frame left live by an Encoded refuses a second `begin` until it is dropped.
    let mut encoder = BatchDevice::new_encoder(&*cpu);
    let mut frame = block_on(batch.begin()).unwrap();
    frame.set(&h.gain, 12).unwrap();
    let ticket = frame.ticket(&h.y).unwrap();
    let encoded = frame.encode(&mut encoder).unwrap();
    assert!(matches!(
        ready_now(batch.begin()),
        Some(Err(Error::FrameLive { .. }))
    ));
    let job = encoder.run(&cpu).unwrap();
    let _submitted = encoded.submitted(&job);
    assert_eq!(block_on(ticket).unwrap(), reference(&x, 12));
}

#[test]
fn twin_encode_equals_submit() {
    let cpu = Arc::new(CpuDevice::new());
    let (mut batch, h) = build_cpu(&cpu).unwrap();
    let x = data(5);
    let mut frame = block_on(batch.begin()).unwrap();
    frame.write(&h.x, &x).unwrap();
    frame.set(&h.gain, 3).unwrap();
    let by_submit = frame.ticket(&h.y).unwrap();
    let _s = frame.submit().unwrap();
    let by_submit = block_on(by_submit).unwrap();

    let mut encoder = BatchDevice::new_encoder(&*cpu);
    let mut frame = block_on(batch.begin()).unwrap();
    frame.write(&h.x, &x).unwrap();
    frame.set(&h.gain, 3).unwrap();
    let by_encode = frame.ticket(&h.y).unwrap();
    let encoded = frame.encode(&mut encoder).unwrap();
    let job = encoder.run(&cpu).unwrap();
    let _s = encoded.submitted(&job);
    assert_eq!(block_on(by_encode).unwrap(), by_submit);
    assert_eq!(by_submit, reference(&x, 3));
}

// ---- what a refused or dropped frame leaves behind (decision D258, M1) ----
//
// A batch of two parameters: `y = (x * 3 + gain + offset) * 2 + 1`.

struct Handles2 {
    gain: Param<u32>,
    offset: Param<u32>,
    x: Stage<u32, D1>,
    y: Stage<u32, D1>,
}

macro_rules! define_build2 {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles2), Error> {
            gpu_batch!("encoded2", on(device); in_flight(2); |b| {
                let gain = b.param::<u32>(1);
                let offset = b.param::<u32>(1);
                let x = b.input::<u32>(N);
                let y = gpu_task!(N, |i| -> u32 { x[i] * 3u32 + gain.get() + offset.get() });
                let _double = gpu_task!(&mut y, |i| { y[i] = y[i] * 2u32 + 1u32; });
                Handles2 { gain, offset, x, y }
            })
        }
    };
}

define_build2!(build2_cpu, CpuDevice);
define_build2!(build2_gpu, HeadlessDevice);

fn reference2(x: &[u32], gain: u32, offset: u32) -> Vec<u32> {
    x.iter().map(|v| (v * 3 + gain + offset) * 2 + 1).collect()
}

/// What the scenarios need of a device beyond `BatchDevice`: the caller's own submission of an encoder, and the token of
/// a submission that carries nothing of the frame.
trait Caller: BatchDevice {
    fn run_encoder(&self, encoder: Self::Encoder) -> Self::Token;
    fn unrelated(&self) -> Self::Token;
}

impl Caller for HeadlessDevice {
    fn run_encoder(&self, encoder: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
        submit_caller(self, encoder)
    }
    fn unrelated(&self) -> wgpu::SubmissionIndex {
        submit_caller(self, gpu_encoder(self))
    }
}

impl Caller for CpuDevice {
    fn run_encoder(&self, encoder: gust::cpu::CpuEncoder) -> gust::cpu::CpuJob {
        encoder.run(self).unwrap()
    }
    fn unrelated(&self) -> gust::cpu::CpuJob {
        BatchDevice::new_encoder(self).run(self).unwrap()
    }
}

/// A frame that is submitted as usual: `record` writes and sets, the values come back.
fn next<D: BatchDevice>(
    batch: &mut Batch<D>,
    h: &Handles2,
    record: impl FnOnce(&mut FrameRun<D>),
) -> Vec<u32>
where
    D::Buffer<u32>: 'static,
{
    let mut frame = block_on(batch.begin()).unwrap();
    record(&mut frame);
    let ticket = frame.ticket(&h.y).unwrap();
    let _submitted = frame.submit().unwrap();
    block_on(ticket).unwrap()
}

/// Refusal 3: the frame is encoded, its encoder dropped, and the caller reports a submission that did not carry it.
fn refused<D: Caller>(
    device: &D,
    batch: &mut Batch<D>,
    h: &Handles2,
    record: impl FnOnce(&mut FrameRun<D>),
) where
    D::Buffer<u32>: 'static,
{
    let mut encoder = BatchDevice::new_encoder(device);
    let mut frame = block_on(batch.begin()).unwrap();
    record(&mut frame);
    let ticket = frame.ticket(&h.y).unwrap();
    let unrelated = device.unrelated();
    let encoded = frame.encode(&mut encoder).unwrap();
    drop(encoder);
    let _submitted = encoded.submitted(&unrelated);
    assert!(matches!(block_on(ticket), Err(Error::NotSubmitted { .. })));
}

/// The first frame of every scenario: `x`, gain 1, offset 2.
fn first<D: BatchDevice>(batch: &mut Batch<D>, h: &Handles2, x: &[u32])
where
    D::Buffer<u32>: 'static,
{
    let values = next(batch, h, |frame| {
        frame.write(&h.x, x).unwrap();
        frame.set(&h.gain, 1).unwrap();
        frame.set(&h.offset, 2).unwrap();
    });
    assert_eq!(values, reference2(x, 1, 2));
}

/// The `set` of a reported frame holds for the frames after it, whether or not its encoder ran.
fn a_refused_set_holds<D: Caller>(device: &D, mut batch: Batch<D>, h: &Handles2)
where
    D::Buffer<u32>: 'static,
{
    let x = data(6);
    first(&mut batch, h, &x);
    refused(device, &mut batch, h, |frame| {
        frame.set(&h.gain, 9).unwrap()
    });
    // A frame that sets nothing: the device must hold what the host holds, gain 9.
    let values = next(&mut batch, h, |_| {});
    assert_eq!(values, reference2(&x, 9, 2));
}

/// The next frame sets only the other parameter of the stage: the refused `set` is not lost, nor taken from the device.
fn a_refused_set_holds_when_the_next_frame_sets_the_other_parameter<D: Caller>(
    device: &D,
    mut batch: Batch<D>,
    h: &Handles2,
) where
    D::Buffer<u32>: 'static,
{
    let x = data(7);
    first(&mut batch, h, &x);
    refused(device, &mut batch, h, |frame| {
        frame.set(&h.gain, 9).unwrap()
    });
    let values = next(&mut batch, h, |frame| frame.set(&h.offset, 5).unwrap());
    assert_eq!(values, reference2(&x, 9, 5));
}

/// The write of a frame that never reached the queue is lost: the next frame sees the input of the frame before.
fn a_refused_write_is_lost<D: Caller>(device: &D, mut batch: Batch<D>, h: &Handles2)
where
    D::Buffer<u32>: 'static,
{
    let x = data(8);
    first(&mut batch, h, &x);
    let other = data(9);
    refused(device, &mut batch, h, |frame| {
        frame.write(&h.x, &other).unwrap();
    });
    let values = next(&mut batch, h, |_| {});
    assert_eq!(values, reference2(&x, 1, 2));
}

/// A dropped `Encoded` whose encoder the caller still submits: its tickets are `NotSubmitted`, its write does land (the
/// device does what the encoder says), its `set` is discarded and the next frame records the parameters again.
fn a_dropped_encoded_with_a_submitted_encoder<D: Caller>(
    device: &D,
    mut batch: Batch<D>,
    h: &Handles2,
) where
    D::Buffer<u32>: 'static,
{
    let x = data(10);
    first(&mut batch, h, &x);
    let other = data(11);
    let mut encoder = BatchDevice::new_encoder(device);
    let mut frame = block_on(batch.begin()).unwrap();
    frame.write(&h.x, &other).unwrap();
    frame.set(&h.gain, 8).unwrap();
    let ticket = frame.ticket(&h.y).unwrap();
    drop(frame.encode(&mut encoder).unwrap());
    assert!(matches!(block_on(ticket), Err(Error::NotSubmitted { .. })));
    let _token = device.run_encoder(encoder);
    let values = next(&mut batch, h, |_| {});
    assert_eq!(values, reference2(&other, 1, 2));
}

#[test]
fn twin_a_refused_set_holds() {
    let cpu = Arc::new(CpuDevice::new());
    let (batch, h) = build2_cpu(&cpu).unwrap();
    a_refused_set_holds(&*cpu, batch, &h);
}

#[test]
fn twin_a_refused_set_holds_when_the_next_frame_sets_the_other_parameter() {
    let cpu = Arc::new(CpuDevice::new());
    let (batch, h) = build2_cpu(&cpu).unwrap();
    a_refused_set_holds_when_the_next_frame_sets_the_other_parameter(&*cpu, batch, &h);
}

#[test]
fn twin_a_refused_write_is_lost() {
    let cpu = Arc::new(CpuDevice::new());
    let (batch, h) = build2_cpu(&cpu).unwrap();
    a_refused_write_is_lost(&*cpu, batch, &h);
}

#[test]
fn twin_a_dropped_encoded_with_a_submitted_encoder() {
    let cpu = Arc::new(CpuDevice::new());
    let (batch, h) = build2_cpu(&cpu).unwrap();
    a_dropped_encoded_with_a_submitted_encoder(&*cpu, batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_refused_set_holds() {
    let device = private_device();
    let (batch, h) = build2_gpu(&device).unwrap();
    a_refused_set_holds(&*device, batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_refused_set_holds_when_the_next_frame_sets_the_other_parameter() {
    let device = private_device();
    let (batch, h) = build2_gpu(&device).unwrap();
    a_refused_set_holds_when_the_next_frame_sets_the_other_parameter(&*device, batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_refused_write_is_lost() {
    let device = private_device();
    let (batch, h) = build2_gpu(&device).unwrap();
    a_refused_write_is_lost(&*device, batch, &h);
}

/// Retire-on-drop: the staging buffer of a ticket whose copy is recorded in an encoder that the caller still holds is
/// retired when the `Encoded` is dropped, not given back to the pool.
#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_dropped_encoded_retires_its_staging_and_the_next_frame_is_exact() {
    let device = private_device();
    let (batch, h) = build2_gpu(&device).unwrap();
    let retired = device.buffer_pool_stats().retired;
    a_dropped_encoded_with_a_submitted_encoder(&*device, batch, &h);
    assert_eq!(
        device.buffer_pool_stats().retired,
        retired + 1,
        "the one ticket of the dropped Encoded retired its staging buffer"
    );
}

// ---- the encode form re-records only what changed (m14.s2cf) ----
//
// Two stages with one parameter each: `y = x * 3 + gain`, then `z = y * 2 + offset`. `Batch::extent_records` counts the
// extent records the frames recorded, so a frame's cost shows as a difference.

struct Handles3 {
    gain: Param<u32>,
    offset: Param<u32>,
    x: Stage<u32, D1>,
    z: Stage<u32, D1>,
}

macro_rules! define_build3 {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles3), Error> {
            gpu_batch!("encoded3", on(device); in_flight(2); |b| {
                let offset = b.param::<u32>(1);
                let gain = b.param::<u32>(1);
                let x = b.input::<u32>(N);
                let y = gpu_task!(N, |i| -> u32 { x[i] * 3u32 + gain.get() });
                let z = gpu_task!(N, |i| -> u32 { y[i] * 2u32 + offset.get() });
                Handles3 { gain, offset, x, z }
            })
        }
    };
}

define_build3!(build3_cpu, CpuDevice);
define_build3!(build3_gpu, HeadlessDevice);

fn reference3(x: &[u32], gain: u32, offset: u32) -> Vec<u32> {
    x.iter().map(|v| (v * 3 + gain) * 2 + offset).collect()
}

/// One frame, recorded into the caller's encoder and reported: its values, and how many extent records it recorded.
fn encoded_frame<D: Caller>(
    device: &D,
    batch: &mut Batch<D>,
    h: &Handles3,
    record: impl FnOnce(&mut FrameRun<D>),
) -> (Vec<u32>, u64)
where
    D::Buffer<u32>: 'static,
{
    let before = batch.extent_records();
    let mut encoder = BatchDevice::new_encoder(device);
    let mut frame = block_on(batch.begin()).unwrap();
    record(&mut frame);
    let ticket = frame.ticket(&h.z).unwrap();
    let encoded = frame.encode(&mut encoder).unwrap();
    let token = device.run_encoder(encoder);
    let _submitted = encoded.submitted(&token);
    let recorded = batch.extent_records() - before;
    (block_on(ticket).unwrap(), recorded)
}

/// The same frame made by `submit`.
fn submitted_frame<D: BatchDevice>(
    batch: &mut Batch<D>,
    h: &Handles3,
    record: impl FnOnce(&mut FrameRun<D>),
) -> (Vec<u32>, u64)
where
    D::Buffer<u32>: 'static,
{
    let before = batch.extent_records();
    let mut frame = block_on(batch.begin()).unwrap();
    record(&mut frame);
    let ticket = frame.ticket(&h.z).unwrap();
    let _submitted = frame.submit().unwrap();
    let recorded = batch.extent_records() - before;
    (block_on(ticket).unwrap(), recorded)
}

/// After an encoded frame sets `gain`, only the stage of `gain` is recorded again, by the later encoded frames and by
/// the first frame `submit` makes (which clears the list); the stage of `offset` is not, and every value is exact.
fn an_unchanged_stage_is_not_re_recorded<D: Caller>(device: &D, mut batch: Batch<D>, h: &Handles3)
where
    D::Buffer<u32>: 'static,
{
    let x = data(12);
    let (values, recorded) = submitted_frame(&mut batch, h, |frame| {
        frame.write(&h.x, &x).unwrap();
        frame.set(&h.gain, 1).unwrap();
        frame.set(&h.offset, 2).unwrap();
    });
    assert_eq!(values, reference3(&x, 1, 2));
    assert_eq!(recorded, 2, "a first frame records both stages");

    let (values, recorded) = encoded_frame(device, &mut batch, h, |frame| {
        frame.set(&h.gain, 3).unwrap()
    });
    assert_eq!(values, reference3(&x, 3, 2));
    assert_eq!(recorded, 1, "only the stage of `gain` changed");

    let (values, recorded) = encoded_frame(device, &mut batch, h, |_| {});
    assert_eq!(values, reference3(&x, 3, 2));
    assert_eq!(
        recorded, 1,
        "`gain` is still stale (the host cannot tell whether the encoder ran); the stage of `offset` is left alone"
    );

    let (values, recorded) = encoded_frame(device, &mut batch, h, |frame| {
        frame.set(&h.offset, 5).unwrap()
    });
    assert_eq!(values, reference3(&x, 3, 5));
    assert_eq!(recorded, 2, "the stale `gain` and the `offset` set now");

    let (values, recorded) = submitted_frame(&mut batch, h, |_| {});
    assert_eq!(values, reference3(&x, 3, 5));
    assert_eq!(
        recorded, 2,
        "a submitted frame records what is stale, then clears it"
    );

    let (values, recorded) = encoded_frame(device, &mut batch, h, |_| {});
    assert_eq!(values, reference3(&x, 3, 5));
    assert_eq!(recorded, 0, "nothing is stale and nothing is set");
}

#[test]
fn twin_an_unchanged_stage_is_not_re_recorded() {
    let cpu = Arc::new(CpuDevice::new());
    let (batch, h) = build3_cpu(&cpu).unwrap();
    an_unchanged_stage_is_not_re_recorded(&*cpu, batch, &h);
}

/// The CPU twin checks overflow (a panic in a debug build): a `submit` whose stage fails after the extent record of `gain` ran leaves that value on
/// the device. The host discards the `set`, so the next frame, which sets only `offset`, must record `gain` again.
#[test]
fn twin_a_failed_submission_leaves_no_discarded_set() {
    let cpu = Arc::new(CpuDevice::new());
    let (mut batch, h) = build3_cpu(&cpu).unwrap();
    let x = data(13);
    let (values, _) = submitted_frame(&mut batch, &h, |frame| {
        frame.write(&h.x, &x).unwrap();
        frame.set(&h.gain, 1).unwrap();
        frame.set(&h.offset, 2).unwrap();
    });
    assert_eq!(values, reference3(&x, 1, 2));

    let mut frame = block_on(batch.begin()).unwrap();
    frame.set(&h.gain, u32::MAX).unwrap();
    let outcome = catch_unwind(AssertUnwindSafe(|| frame.submit().map(|_| ())));
    assert!(!matches!(outcome, Ok(Ok(()))), "x * 3 + u32::MAX overflows");

    let (values, _) = submitted_frame(&mut batch, &h, |frame| frame.set(&h.offset, 5).unwrap());
    assert_eq!(
        values,
        reference3(&x, 1, 5),
        "`gain` is the host's 1, not the failed frame's"
    );
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_an_unchanged_stage_is_not_re_recorded() {
    let device = private_device();
    let (batch, h) = build3_gpu(&device).unwrap();
    an_unchanged_stage_is_not_re_recorded(&*device, batch, &h);
}
