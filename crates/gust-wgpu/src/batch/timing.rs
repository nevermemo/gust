//! Frame timing on the GPU device: one query set, one resolve buffer and a few read-back buffers per timed batch, made
//! in [`GpuTiming::new`] when the batch is sealed. They belong to the batch, never to the device's timestamp profiler
//! (which is locked across a wait and so cannot serve frames in flight).
//!
//! A timed frame (one that called `FrameRun::time`) writes timestamps in passes that exist anyway (the first pass of a
//! stage writes its begin, the last its end), then records one resolve and one copy into its encoder; a frame that did
//! not ask records none of it.
//!
//! **Stamps inside passes (level `Passes`).** Where the device has `TIMESTAMP_QUERY_INSIDE_PASSES` (read from the
//! device when the frame is recorded, so a device the caller opened counts too) the frame has the passes of the untimed
//! one: the plain compute dispatches of neighbouring stages share one compute pass, which the clock keeps open from one
//! `frame_stage_timed` call to the next, and only a `pixel` dispatch keeps a pass of its own. An indirect counted stage
//! (slice m14.s2imerge) is stamped inside the shared pass like a plain one: its begin before its args dispatch and its
//! end after its indirect dispatch, one entry for the stage, and wgpu's own validation dispatch for the indirect launch
//! falls inside that entry. The slot layout is the one of the pass-per-stage form, `2 * stage` for a stage's begin and
//! `2 * stage + 1` for its end, so the plan, the read-back and the resolve do not change. Inside a shared pass a
//! stage's begin is a `write_timestamp` before its first dispatch and its end one after its last, so two neighbouring
//! stages write the same point twice and a stage's time runs from its begin to its end as before; a stage that has a
//! pass of its own writes them at the pass boundary (`timestamp_writes`: begin on the first pass, end on the last).
//! Without the feature (a browser device, or a caller's device opened without it) every stage is recorded in a pass of
//! its own, as before, a counted stage as one pass that holds both its dispatches. While `force_pass_per_dispatch` is
//! on every dispatch is a pass of its own, a counted stage's two passes carrying its begin and its end. A timed and an
//! untimed frame record the same passes for a counted stage in the cases with the feature (one shared pass) and under
//! the force (two passes a counted stage); without the feature the timed frame has a pass per stage where the untimed
//! one has the one. A stamp between two stages that depend on nothing from each other (no barrier between them) only
//! splits work that may overlap on the GPU loosely: such a stage's time is the span between its two stamps, not an
//! exact cost.
//!
//! Frames in flight share the query set and the resolve buffer, because the queue runs
//! them in order and each frame copies its slots out before the next one writes them (decision D258); only the
//! read-back buffer is the frame's own, taken from the pool when the frame is recorded and given back by the mapping's
//! callback, so a caller that never reads its frame times still gets its buffers back as the device is polled. When the
//! pool is empty and the batch already holds its cap of buffers the frame is not timed (`Ok(None)`), it never waits. A
//! buffer of a frame whose caller's encoder never ran, or may still run (the `Encoded` was dropped after it recorded the
//! copy), is retired, not reused, as a ticket's staging buffer is; a retired buffer, a buffer lost to a failed mapping
//! and one dropped with an unread frame time are counted out, so the pool makes a replacement on demand and never
//! shrinks for good.

use core::task::Poll;
#[cfg(target_arch = "wasm32")]
use std::rc::Rc as Shared;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc as Shared;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use gust::batch::{BoxFuture, DeviceTiming, FrameClock, FrameTiming, OpenedTiming};

use super::counted::record_indirect;
use super::frame::{Done, FRAME_PASS_LABEL, GpuSubmission};
use super::{GpuStage, map_error};
use crate::{Error, HeadlessDevice, PassTimestamps};

/// Timing objects made by [`GpuTiming::new`], timestamp slots written by passes and resolves recorded, in this process
/// (test support: `HeadlessDevice::frame_timing_counters`).
static OBJECTS: AtomicU64 = AtomicU64::new(0);
static STAMPS: AtomicU64 = AtomicU64::new(0);
static RESOLVES: AtomicU64 = AtomicU64::new(0);
static RETIRED: AtomicU64 = AtomicU64::new(0);

/// Bytes of one timestamp.
const TICK_BYTES: u64 = size_of::<u64>() as u64;

/// What a timed batch keeps on the device.
pub(super) struct GpuTiming {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    /// The read-back buffers nobody is using.
    free: Mutex<Vec<wgpu::Buffer>>,
    /// How many read-back buffers exist (free or in use), and the most the batch keeps.
    existing: AtomicUsize,
    cap: usize,
    device: wgpu::Device,
    bytes: u64,
}

impl GpuTiming {
    /// The one place a batch's timing objects are made: a query set and a resolve buffer of `slots` timestamps, and
    /// `readbacks` read-back buffers of the same size.
    fn new(device: &wgpu::Device, slots: u32, readbacks: usize) -> Self {
        OBJECTS.fetch_add(1, Ordering::Relaxed);
        let bytes = u64::from(slots) * TICK_BYTES;
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GUST batch frame timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: slots,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GUST batch frame timestamp resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let this = Self {
            query_set,
            resolve,
            free: Mutex::new(Vec::new()),
            existing: AtomicUsize::new(0),
            cap: readbacks,
            device: device.clone(),
            bytes,
        };
        // A few buffers ready for the first frames; the rest are made when a frame needs one.
        let ready = (readbacks / 2).max(1);
        let buffers: Vec<wgpu::Buffer> = (0..ready).map(|_| this.make()).collect();
        this.existing.store(buffers.len(), Ordering::Relaxed);
        *this.free.lock().unwrap_or_else(PoisonError::into_inner) = buffers;
        this
    }

    fn make(&self) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GUST batch frame timestamp readback"),
            size: self.bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    /// A free buffer, or a new one while the batch holds fewer than its cap; `None` when all of them are in use.
    fn take(&self) -> Option<wgpu::Buffer> {
        if let Some(buffer) = self
            .free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop()
        {
            return Some(buffer);
        }
        self.existing
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |held| {
                (held < self.cap).then_some(held + 1)
            })
            .ok()
            .map(|_| self.make())
    }

    /// A buffer that must not be reused (a copy into it may still run, or it was lost): counted out, so a later frame
    /// makes a replacement.
    fn retire(&self, buffer: wgpu::Buffer) {
        drop(buffer);
        self.forget();
    }

    /// One buffer is gone without being given back.
    fn forget(&self) {
        self.existing.fetch_sub(1, Ordering::AcqRel);
        RETIRED.fetch_add(1, Ordering::Relaxed);
    }

    fn give(&self, buffer: wgpu::Buffer) {
        self.free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(buffer);
    }
}

/// What one timed frame holds: the batch's objects and its own read-back buffer, given back if the frame is dropped
/// before it was submitted.
struct GpuClock {
    timing: Shared<GpuTiming>,
    readback: Option<wgpu::Buffer>,
    slots: u32,
    /// The copy into the buffer was recorded into an encoder, which a caller may still submit.
    recorded: bool,
    /// The compute pass of the frame that is still open, for the next stage to join (stamps inside passes only). The
    /// encoder must not be used while it is held: every other recording first drops it, which ends the pass.
    run: Option<wgpu::ComputePass<'static>>,
}

impl Drop for GpuClock {
    fn drop(&mut self) {
        if let Some(buffer) = self.readback.take() {
            if self.recorded {
                // The ticket rule: a recorded copy may still run, so the buffer is not reused.
                self.timing.retire(buffer);
            } else {
                self.timing.give(buffer);
            }
        }
    }
}

/// The words a read-back mapping landed, and the buffer when the frame's own future gives it back.
struct Landing {
    done: Done,
    outcome: Mutex<Option<Result<Vec<u64>, String>>>,
    buffer: Mutex<Option<wgpu::Buffer>>,
    timing: Shared<GpuTiming>,
}

impl Drop for Landing {
    fn drop(&mut self) {
        // A buffer nobody decided about (the frame's time was dropped unread): retired, as when the encoder never ran.
        let buffer = self
            .buffer
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(buffer) = buffer {
            self.timing.retire(buffer);
        }
    }
}

impl HeadlessDevice {
    /// The counters of the frame timing hook, in this process: timing objects made, timestamp slots written by passes,
    /// resolves recorded, read-back buffers retired. Test support, outside the pre-1.0 promise (D111).
    #[doc(hidden)]
    #[must_use]
    pub fn frame_timing_counters(&self) -> (u64, u64, u64, u64) {
        (
            OBJECTS.load(Ordering::Relaxed),
            STAMPS.load(Ordering::Relaxed),
            RESOLVES.load(Ordering::Relaxed),
            RETIRED.load(Ordering::Relaxed),
        )
    }

    /// The timing objects of a batch, when this device has timestamp queries.
    pub(super) fn open_timing(
        &self,
        level: FrameTiming,
        stages: usize,
        in_flight: usize,
    ) -> Option<OpenedTiming> {
        if !self
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return None;
        }
        let period = self.queue.get_timestamp_period();
        if !(period.is_finite() && period > 0.0) {
            return None;
        }
        let slots = match level {
            FrameTiming::Passes => 2 * stages.max(1),
            _ => 2,
        };
        let slots = u32::try_from(slots).ok()?;
        let timing = GpuTiming::new(&self.device, slots, 2 * in_flight + 4);
        Some(OpenedTiming {
            handle: Shared::new(timing),
            period_ns: f64::from(period),
        })
    }

    /// A clock for one frame, or `None` when every read-back buffer is in use.
    pub(super) fn take_clock(timing: &DeviceTiming) -> Option<FrameClock> {
        let timing = Shared::clone(timing).downcast::<GpuTiming>().ok()?;
        let readback = timing.take()?;
        Some(Box::new(GpuClock {
            timing,
            readback: Some(readback),
            slots: 0,
            recorded: false,
            run: None,
        }))
    }

    /// Record the passes of a stage with the begin and end timestamps given.
    pub(super) fn encode_stage_timed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stage: &GpuStage,
        clock: &mut FrameClock,
        begin: Option<u32>,
        end: Option<u32>,
    ) -> Result<(), Error> {
        let inside = self.stamps_inside_passes();
        let clock = clock
            .downcast_mut::<GpuClock>()
            .expect("a GPU batch's clock is its own");
        if inside {
            return self.encode_stage_in_run(encoder, stage, clock, begin, end);
        }
        // A pass left open by an earlier stage ends before anything else goes into the encoder.
        clock.run = None;
        let stamps = Some((&clock.timing.query_set, begin, end));
        match &stage.indirect {
            // A counted stage without stamps inside passes is one pass of its own holding both dispatches, with `begin`
            // and `end` on the pass; under `force_pass_per_dispatch` two passes, `begin` on the args one, `end` on the
            // counted one.
            Some(args) => self.record_counted(encoder, &stage.dispatches, args, stamps)?,
            None => self.record_prepared_stamped(encoder, &stage.dispatches, stamps)?,
        }
        let written = u64::from(begin.is_some()) + u64::from(end.is_some());
        STAMPS.fetch_add(written, Ordering::Relaxed);
        Ok(())
    }

    /// Whether a timed frame at level `Passes` records the passes of the untimed one, with stamps inside them: the
    /// device has `TIMESTAMP_QUERY_INSIDE_PASSES` and the one-pass form is not switched off.
    fn stamps_inside_passes(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
            && !crate::pass_per_dispatch_forced()
    }

    /// Record a stage as `encode_stages` records it: its compute dispatches join the pass the clock holds open (or open
    /// one), the counted dispatch of an indirect counted stage launched from its args buffer, a `pixel` dispatch ends
    /// the pass and takes a render pass of its own. `begin` is written before the stage's first dispatch and `end`
    /// after its last: inside the shared pass as `write_timestamp`, on the pass boundary where the first or last
    /// dispatch has a pass of its own. The stage has at least one dispatch.
    fn encode_stage_in_run(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stage: &GpuStage,
        clock: &mut GpuClock,
        begin: Option<u32>,
        end: Option<u32>,
    ) -> Result<(), Error> {
        let timing = Shared::clone(&clock.timing);
        let query_set = &timing.query_set;
        let last = stage.dispatches.len() - 1;
        for (index, dispatch) in stage.dispatches.iter().enumerate() {
            let (begin, end) = (
                (index == 0).then_some(begin).flatten(),
                (index == last).then_some(end).flatten(),
            );
            // The counted dispatch of an indirect counted stage is the last dispatch of its stage.
            let indirect = stage.indirect.as_ref().filter(|_| index == last);
            if dispatch.is_compute() {
                let pass = clock.run.get_or_insert_with(|| {
                    crate::count_passes(1);
                    encoder
                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(FRAME_PASS_LABEL),
                            timestamp_writes: None,
                        })
                        .forget_lifetime()
                });
                if let Some(slot) = begin {
                    pass.write_timestamp(query_set, slot);
                }
                match indirect {
                    Some(args) => record_indirect(pass, dispatch, args),
                    None => dispatch.record_compute(pass),
                }
                if let Some(slot) = end {
                    pass.write_timestamp(query_set, slot);
                }
            } else {
                clock.run = None;
                crate::count_passes(1);
                let stamps = (begin.is_some() || end.is_some()).then_some(PassTimestamps {
                    query_set,
                    begin,
                    end,
                });
                dispatch.encode(&self.device, encoder, FRAME_PASS_LABEL, stamps)?;
            }
            STAMPS.fetch_add(
                u64::from(begin.is_some()) + u64::from(end.is_some()),
                Ordering::Relaxed,
            );
        }
        Ok(())
    }

    /// Record the stages of a frame as `encode_stages` does, with `begin` stamped at the start of the first pass and
    /// `end` at the end of the last.
    pub(super) fn encode_stages_timed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stages: &[&GpuStage],
        clock: &mut FrameClock,
        begin: Option<u32>,
        end: Option<u32>,
    ) -> Result<(), Error> {
        let clock = clock
            .downcast_mut::<GpuClock>()
            .expect("a GPU batch's clock is its own");
        self.encode_stages(encoder, stages, Some((&clock.timing.query_set, begin, end)))?;
        let written = u64::from(begin.is_some()) + u64::from(end.is_some());
        STAMPS.fetch_add(written, Ordering::Relaxed);
        Ok(())
    }

    /// Resolve the first `slots` timestamps and copy them to the frame's read-back buffer.
    pub(super) fn end_clock(
        encoder: &mut wgpu::CommandEncoder,
        clock: &mut FrameClock,
        slots: u32,
    ) {
        let clock = clock
            .downcast_mut::<GpuClock>()
            .expect("a GPU batch's clock is its own");
        // The pass of the last stages, still open, ends before the resolve.
        clock.run = None;
        let readback = clock
            .readback
            .as_ref()
            .expect("a clock keeps its buffer until the frame is submitted");
        let bytes = u64::from(slots) * TICK_BYTES;
        encoder.resolve_query_set(&clock.timing.query_set, 0..slots, &clock.timing.resolve, 0);
        encoder.copy_buffer_to_buffer(&clock.timing.resolve, 0, readback, 0, bytes);
        clock.slots = slots;
        clock.recorded = true;
        RESOLVES.fetch_add(1, Ordering::Relaxed);
    }

    /// Request the mapping, after the submission that carried the copy, and return the future of the raw timestamps:
    /// `None` when the submission was a caller's and its epoch word says the encoder never ran.
    pub(super) fn start_time(
        &self,
        clock: FrameClock,
        submission: &GpuSubmission,
    ) -> BoxFuture<Result<Option<Vec<u64>>, gust::map::Error>> {
        let mut clock = clock
            .downcast::<GpuClock>()
            .expect("a GPU batch's clock is its own");
        let buffer = clock
            .readback
            .take()
            .expect("a clock keeps its buffer until the frame is submitted");
        let bytes = u64::from(clock.slots) * TICK_BYTES;
        let timing = Shared::clone(&clock.timing);
        let epoch = submission.epoch.clone();
        let landing = Shared::new(Landing {
            done: Done::default(),
            outcome: Mutex::new(None),
            buffer: Mutex::new(None),
            timing: Shared::clone(&timing),
        });
        let landed = Shared::clone(&landing);
        let mapped = buffer.clone();
        // Without an epoch word (the batch's own submission) the callback gives the buffer back as soon as it has read
        // it, whether or not anyone reads the frame's time; with one, the future decides after the word is read.
        let recycle = epoch.is_none();
        buffer.map_async(wgpu::MapMode::Read, 0..bytes, move |result| {
            let ticks = result.map_err(|error| error.to_string()).and_then(|()| {
                let view = mapped
                    .get_mapped_range(0..bytes)
                    .map_err(|error| error.to_string())?;
                let ticks: Vec<u64> = view[..]
                    .as_chunks::<{ size_of::<u64>() }>()
                    .0
                    .iter()
                    .map(|word| u64::from_le_bytes(*word))
                    .collect();
                drop(view);
                mapped.unmap();
                Ok(ticks)
            });
            if ticks.is_ok() {
                if recycle {
                    timing.give(mapped);
                } else {
                    *landed.buffer.lock().unwrap_or_else(PoisonError::into_inner) = Some(mapped);
                }
            } else {
                // A failed mapping loses the buffer: counted out, so a later frame makes another.
                drop(mapped);
                timing.forget();
            }
            *landed
                .outcome
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = Some(ticks);
            landed.done.finish();
        });
        let device = self.device.clone();
        let timing = Shared::clone(&clock.timing);
        Box::pin(async move {
            let polling = device.clone();
            let landing_of = Shared::clone(&landing);
            core::future::poll_fn(move |cx| {
                if let Err(error) = polling.poll(wgpu::PollType::Poll) {
                    return Poll::Ready(Err(Error::Poll(error.to_string())));
                }
                landing_of.done.poll(cx.waker()).map(Ok)
            })
            .await
            .map_err(map_error)?;
            let ticks = landing
                .outcome
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take()
                .expect("the callback stores the outcome before it finishes")
                .map_err(|message| map_error(Error::Map(message)))?;
            let Some(epoch) = epoch else {
                return Ok(Some(ticks));
            };
            let ran = core::future::poll_fn(move |cx| {
                if let Err(error) = device.poll(wgpu::PollType::Poll) {
                    return Poll::Ready(Err(Error::Poll(error.to_string())));
                }
                epoch.poll(cx.waker()).map_err(Error::Map)
            })
            .await
            .map_err(map_error)?;
            let buffer = landing
                .buffer
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take();
            if !ran {
                // Retired: the caller may still submit that encoder, which would write into a buffer reused by then.
                if let Some(buffer) = buffer {
                    timing.retire(buffer);
                }
                return Ok(None);
            }
            if let Some(buffer) = buffer {
                timing.give(buffer);
            }
            Ok(Some(ticks))
        })
    }
}
