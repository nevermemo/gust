//! Frames in flight above one (m14.s2c): `in_flight(2)` and `in_flight(3)` batches, on the CPU twin and the GPU.
//!
//! Frame N stays pending while frame N+1 is recorded and submitted. Every frame writes the same input and the same
//! stage buffers, and every frame's ticket resolves to that frame's values, whatever the order of the frames and of the
//! reads. The values stay small enough never to overflow (the CPU twin checks overflow), so a result equals the host
//! reference bit for bit.
//!
//! The GPU tests are `#[ignore]`d like the example's other GPU tests: they run in full verification, on the adapter the
//! environment selects. The pending-state test depends on timing by design: frame N is compute-bound (a counted loop
//! per element, tens of milliseconds on the Radeon 890M), and the check runs microseconds after frame N+1's submission.

use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use batch_basics::app::{block_on, ready_now};
use gust::batch::{Batch, BatchDevice, D1, Error, Param, Stage, Submitted, Ticket};
use gust::cpu::CpuDevice;
use gust::{gpu_batch, gpu_task};
use gust_wgpu::HeadlessDevice;

/// Elements of the exactness tests.
const SMALL: u32 = 1 << 14;
/// Elements of the pending-state test, whose one stage loops [`LOOPS`] times per element.
const HEAVY: u32 = 1 << 20;
/// The counted loop of the pending-state stage.
const LOOPS: u32 = 4000;

struct Handles {
    gain: Param<u32>,
    x: Stage<u32, D1>,
    y: Stage<u32, D1>,
}

/// The batch: `y = x * 3 + gain`, then `y = y * 2 + 1` in place. `u32` arithmetic on small values.
macro_rules! define_build {
    ($build:ident, $device:ty, $depth:tt) => {
        fn $build(
            device: &Arc<$device>,
            n: u32,
        ) -> Result<(Batch<$device>, Handles), Error> {
            gpu_batch!("flight", on(device); in_flight($depth); |b| {
                let gain = b.param::<u32>(1);
                let x = b.input::<u32>(n);
                let y = gpu_task!(n, |i| -> u32 { x[i] * 3u32 + gain.get() });
                let _double = gpu_task!(&mut y, |i| { y[i] = y[i] * 2u32 + 1u32; });
                Handles { gain, x, y }
            })
        }
    };
}

define_build!(build_cpu2, CpuDevice, 2);
define_build!(build_cpu3, CpuDevice, 3);
define_build!(build_gpu2, HeadlessDevice, 2);
define_build!(build_gpu3, HeadlessDevice, 3);

/// A batch of one compute-bound stage: `y = f(x)`, with `f(v) = (v * 3 + gain) % 65521` applied 4000 times.
fn build_gpu2_heavy(
    device: &Arc<HeadlessDevice>,
    n: u32,
) -> Result<(Batch<HeadlessDevice>, Handles), Error> {
    gpu_batch!("heavy", on(device); in_flight(2); |b| {
        let gain = b.param::<u32>(1);
        let x = b.input::<u32>(n);
        let y = gpu_task!(n, |i| -> u32 {
            let mut v = x[i] % 65521u32;
            for _k in 0u32..4000u32 {
                v = (v * 3u32 + gain.get()) % 65521u32;
            }
            v
        });
        Handles { gain, x, y }
    })
}

/// The host check of the heavy stage covers every `SAMPLE`th element: the loop is long, and the host runs it unoptimised.
const SAMPLE: usize = 4096;

fn sample(values: &[u32]) -> Vec<u32> {
    values.iter().step_by(SAMPLE).copied().collect()
}

fn heavy_reference(x: &[u32], gain: u32) -> Vec<u32> {
    x.iter()
        .step_by(SAMPLE)
        .map(|v| {
            let mut v = v % 65521;
            for _ in 0..LOOPS {
                v = (v * 3 + gain) % 65521;
            }
            v
        })
        .collect()
}

fn data(n: u32, seed: u32) -> Vec<u32> {
    (0..n).map(|i| (i * 7 + seed) % 100_000).collect()
}

fn reference(x: &[u32], gain: u32) -> Vec<u32> {
    x.iter().map(|v| (v * 3 + gain) * 2 + 1).collect()
}

/// One frame: the write of `x` when given, the gain, one ticket on `y`, one submission.
fn submit<D: BatchDevice>(
    batch: &mut Batch<D>,
    h: &Handles,
    x: Option<&[u32]>,
    gain: u32,
) -> (Ticket<u32>, Submitted<D>)
where
    D::Buffer<u32>: 'static,
{
    let mut frame = ready_now(batch.begin())
        .expect("begin returns at once while fewer frames than `in_flight` are in flight")
        .unwrap();
    if let Some(x) = x {
        frame.write(&h.x, x).unwrap();
    }
    frame.set(&h.gain, gain).unwrap();
    let ticket = frame.ticket(&h.y).unwrap();
    (ticket, frame.submit().unwrap())
}

/// Like [`submit`], for a frame that may have to wait for the oldest frame in flight first.
fn submit_waiting<D: BatchDevice>(
    batch: &mut Batch<D>,
    h: &Handles,
    x: Option<&[u32]>,
    gain: u32,
) -> (Ticket<u32>, Submitted<D>)
where
    D::Buffer<u32>: 'static,
{
    let mut frame = block_on(batch.begin()).unwrap();
    if let Some(x) = x {
        frame.write(&h.x, x).unwrap();
    }
    frame.set(&h.gain, gain).unwrap();
    let ticket = frame.ticket(&h.y).unwrap();
    (ticket, frame.submit().unwrap())
}

/// Six frames that write the same input and the same stage buffers; odd frames write no input (it stays from the frame
/// before, in queue order: the persistence rule, D258). All tickets are kept and resolved in reverse order, after every frame was submitted.
fn overlapping_frames_are_exact<D: BatchDevice>(mut batch: Batch<D>, h: &Handles)
where
    D::Buffer<u32>: 'static,
{
    let mut sent = Vec::new();
    let mut current = Vec::new();
    for k in 0..6_u32 {
        let gain = 10 * k + 1;
        let x = if k % 2 == 0 {
            current = data(SMALL, k);
            Some(current.as_slice())
        } else {
            None
        };
        let (ticket, _submitted) = submit_waiting(&mut batch, h, x, gain);
        sent.push((ticket, reference(&current, gain)));
    }
    for (k, (ticket, expected)) in sent.into_iter().enumerate().rev() {
        assert_eq!(block_on(ticket).unwrap(), expected, "frame {k}");
    }
}

/// A dropped frame fails its tickets and costs the queue nothing; a `begin` dropped while it waits for the oldest frame
/// gives that frame back to the queue.
fn dropped_frames_and_claims<D: BatchDevice>(mut batch: Batch<D>, h: &Handles, depth: u32)
where
    D::Buffer<u32>: 'static,
{
    let x = data(SMALL, 7);
    let (first, _s) = submit(&mut batch, h, Some(&x[..]), 3);
    // A frame dropped before `submit`: its ticket is `NotSubmitted`, and it holds no place in the queue.
    let doomed = {
        let mut frame = ready_now(batch.begin()).unwrap().unwrap();
        // A second `begin` while a frame is being recorded is refused, whatever the depth.
        assert!(matches!(
            ready_now(batch.begin()),
            Some(Err(Error::FrameLive { .. }))
        ));
        frame.set(&h.gain, 99).unwrap();
        frame.ticket(&h.y).unwrap()
    };
    assert!(matches!(block_on(doomed), Err(Error::NotSubmitted { .. })));
    // Fill the queue, then drop a `begin` future that was polled once (it waits for the oldest frame, or it was ready).
    let mut tickets = vec![(first, reference(&x, 3))];
    for gain in 4..(3 + depth) {
        let (ticket, _s) = submit(&mut batch, h, None, gain);
        tickets.push((ticket, reference(&x, gain)));
    }
    drop(ready_now(batch.begin()));
    // The queue is intact: the next frame waits as it should, and every frame is exact.
    let (ticket, _s) = submit_waiting(&mut batch, h, None, 50);
    tickets.push((ticket, reference(&x, 50)));
    for (k, (ticket, expected)) in tickets.into_iter().enumerate() {
        assert_eq!(block_on(ticket).unwrap(), expected, "frame {k}");
    }
}

#[test]
fn twin_overlapping_frames_are_exact() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu2(&device, SMALL).unwrap();
    overlapping_frames_are_exact(batch, &h);
    let (batch, h) = build_cpu3(&device, SMALL).unwrap();
    overlapping_frames_are_exact(batch, &h);
}

#[test]
fn twin_dropped_frames_and_claims() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu2(&device, SMALL).unwrap();
    dropped_frames_and_claims(batch, &h, 2);
    let (batch, h) = build_cpu3(&device, SMALL).unwrap();
    dropped_frames_and_claims(batch, &h, 3);
}

fn private_device() -> Arc<HeadlessDevice> {
    Arc::new(HeadlessDevice::new().unwrap_or_else(|error| {
        panic!("a GPU adapter is required; this test must not silently skip: {error}")
    }))
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_overlapping_frames_are_exact() {
    let device = private_device();
    let (batch, h) = build_gpu2(&device, SMALL).unwrap();
    overlapping_frames_are_exact(batch, &h);
    let (batch, h) = build_gpu3(&device, SMALL).unwrap();
    overlapping_frames_are_exact(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_dropped_frames_and_claims() {
    let device = private_device();
    let (batch, h) = build_gpu2(&device, SMALL).unwrap();
    dropped_frames_and_claims(batch, &h, 2);
    let (batch, h) = build_gpu3(&device, SMALL).unwrap();
    dropped_frames_and_claims(batch, &h, 3);
}

/// Frame N is pending when frame N+1 is submitted, and `begin` of a third frame waits for N, also after a `begin` that
/// waited was dropped (the rollback of the claim); both frames are exact.
///
/// Whether N has finished when the check runs depends on the timing of the GPU and of this thread (N is compute-bound,
/// which takes tens of milliseconds on the Radeon 890M), so each attempt uses a fresh batch and the pending state has to
/// be observed in at least one of [`ATTEMPTS`]. What does not depend on timing is asserted in every attempt: frame N+1's
/// `begin` returns at once, both frames are exact, and each is one counted submission.
#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_frame_n_is_pending_when_n_plus_1_is_submitted() {
    const ATTEMPTS: usize = 10;
    let device = private_device();
    let x = data(HEAVY, 1);
    let (mut pending_seen, mut third_waited, mut rolled_back) = (0, 0, 0);
    for attempt in 0..ATTEMPTS {
        let (mut batch, h) = build_gpu2_heavy(&device, HEAVY).unwrap();
        let before = device.gust_submissions();
        let (mut first, first_done) = submit(&mut batch, &h, Some(&x[..]), 5);
        // `begin` of frame N+1 returns at once (`submit` asserts it): one frame is in flight, the depth is two.
        let (second, _second_done) = submit(&mut batch, &h, None, 9);
        assert_eq!(
            device.gust_submissions() - before,
            2,
            "attempt {attempt}: one submission per frame"
        );
        // A ticket that resolved here must not be polled again, so its values are kept.
        let first_early = ready_now(&mut first);
        if first_early.is_none() {
            pending_seen += 1;
            // Two frames are in flight, so a third `begin` has to wait for the first.
            if ready_now(batch.begin()).is_none() {
                third_waited += 1;
                // That `begin` was dropped while it waited: the claim gave frame N back to the queue, so the next
                // `begin` waits for it too. Without the rollback only frame N+1 would be in flight, and it would
                // return at once.
                if ready_now(batch.begin()).is_none() {
                    rolled_back += 1;
                }
            }
        }
        let first_values = match first_early {
            Some(values) => values,
            None => block_on(first),
        };
        assert_eq!(
            sample(&first_values.unwrap()),
            heavy_reference(&x, 5),
            "frame N"
        );
        assert_eq!(
            sample(&block_on(second).unwrap()),
            heavy_reference(&x, 9),
            "frame N+1"
        );
        first_done.wait().unwrap();
        // The batch is usable after the dropped `begin`s.
        let (third, _t) = submit_waiting(&mut batch, &h, None, 11);
        assert_eq!(
            sample(&block_on(third).unwrap()),
            heavy_reference(&x, 11),
            "frame N+2"
        );
    }
    println!(
        "frame N pending at N+1's submission in {pending_seen} of {ATTEMPTS} attempts, a third begin waited in {third_waited}, a second one after the drop in {rolled_back}"
    );
    assert!(
        pending_seen > 0,
        "frame N was already resolved when frame N+1 was submitted in every attempt: the pending state was never observed"
    );
    assert!(
        third_waited > 0,
        "a third begin returned at once with two frames in flight in every attempt that saw frame N pending"
    );
    assert!(
        rolled_back > 0,
        "a `begin` after a dropped `begin` returned at once: the claim did not give the oldest frame back"
    );
}

/// The time `count_polls` waits for a ticket before it fails.
const POLL_DEADLINE: Duration = Duration::from_secs(20);

/// Polls `ticket` until it resolves, on this thread, and returns the values and the number of polls. Fails after
/// [`POLL_DEADLINE`].
fn count_polls(ticket: &mut Ticket<u32>) -> (Vec<u32>, u64) {
    let mut context = Context::from_waker(Waker::noop());
    let started = Instant::now();
    let mut polls = 0_u64;
    loop {
        polls += 1;
        if let Poll::Ready(values) = std::pin::Pin::new(&mut *ticket).poll(&mut context) {
            return (values.unwrap(), polls);
        }
        assert!(
            started.elapsed() < POLL_DEADLINE,
            "a ticket was still pending after {polls} polls"
        );
        std::thread::yield_now();
    }
}

/// Polls that resolve a ticket whose frame has run: the first poll polls the device and delivers the mapping, so
/// tickets of frames that were waited for resolve at once, with two frames in flight as with one.
const POLLS_AFTER_WAIT: u64 = 2;

/// Tickets of frames that were waited for resolve within [`POLLS_AFTER_WAIT`] polls. A ticket nobody waited for
/// resolves by polls alone, before [`POLL_DEADLINE`]; its poll count is printed and not bounded,
/// because it depends on how long the frame runs against the speed of the polling loop.
#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_ticket_resolves_in_a_bounded_number_of_polls() {
    let device = private_device();
    let (mut batch, h) = build_gpu2(&device, SMALL).unwrap();
    let x = data(SMALL, 3);
    for round in 0..3 {
        // Two frames in flight, both waited for: each ticket resolves within the bound.
        let (mut first, first_done) = submit_waiting(&mut batch, &h, Some(&x[..]), 20 + round);
        let (mut second, second_done) = submit_waiting(&mut batch, &h, None, 30 + round);
        first_done.wait().unwrap();
        second_done.wait().unwrap();
        for (ticket, gain, name) in [(&mut second, 30, "second"), (&mut first, 20, "first")] {
            let (values, polls) = count_polls(ticket);
            assert_eq!(values, reference(&x, gain + round), "{name}");
            println!("round {round}: the {name} ticket resolved in {polls} polls after the wait");
            assert!(
                polls <= POLLS_AFTER_WAIT,
                "the {name} ticket needed {polls} polls after its frame was waited for (bound {POLLS_AFTER_WAIT})"
            );
        }
        // The same without waiting: the polls themselves drive the device; the count is printed, not bounded.
        let (mut third, _t) = submit_waiting(&mut batch, &h, None, 40 + round);
        let (values, polls) = count_polls(&mut third);
        assert_eq!(values, reference(&x, 40 + round));
        println!("round {round}: an unwaited ticket resolved in {polls} polls");
    }
}

/// A ticket is `Send`: awaited on another thread it resolves to the same values. The scheduling thread of decision D257
/// awaits tickets on the thread that built the batch; this records what happens elsewhere.
#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_ticket_polled_on_another_thread_resolves_too() {
    let device = private_device();
    let (mut batch, h) = build_gpu2(&device, SMALL).unwrap();
    let x = data(SMALL, 5);
    let (first, _a) = submit(&mut batch, &h, Some(&x[..]), 1);
    let (second, _b) = submit(&mut batch, &h, None, 2);
    let worker = std::thread::spawn(move || {
        let mut second = second;
        let (values, polls) = count_polls(&mut second);
        (values, polls)
    });
    let mut first = first;
    let (first_values, first_polls) = count_polls(&mut first);
    let (second_values, second_polls) = worker.join().unwrap();
    println!("own thread: {first_polls} polls; other thread: {second_polls} polls");
    assert_eq!(first_values, reference(&x, 1));
    assert_eq!(second_values, reference(&x, 2));
}
