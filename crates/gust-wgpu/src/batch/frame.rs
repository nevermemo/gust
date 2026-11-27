//! One frame: the copies it collects while it is written, the encoder it is recorded into (its own, or a caller's), and
//! its one submission.

use core::task::{Poll, Waker};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use gust::batch::BoxFuture;
use gust::{Access, GpuPod, LengthRule, Resource, pod_slice_as_bytes};

use super::counted::record_indirect;
use super::{GpuStage, map_error};
use crate::encode::{record_upload_copy, stage_upload};
use crate::pool::EPOCH;
use crate::{
    Error, GpuBuffer, HeadlessDevice, PassTimestamps, PreparedDispatch, begin_compute,
    encode_batch_passes, pass_per_dispatch_forced,
};

/// The label of the passes `encode_stages` records for a frame.
pub(super) const FRAME_PASS_LABEL: &str = "GUST batch frame pass";

/// A dispatch of a run: a plain compute dispatch, or the counted dispatch of an indirect counted stage with the args
/// buffer it launches from (the stage's args dispatch, a plain compute dispatch, comes before it in the run).
#[derive(Clone, Copy)]
enum Item<'a> {
    Plain(&'a PreparedDispatch),
    Indirect(&'a PreparedDispatch, &'a wgpu::Buffer),
}

impl<'a> Item<'a> {
    fn dispatch(&self) -> &'a PreparedDispatch {
        match *self {
            Item::Plain(dispatch) | Item::Indirect(dispatch, _) => dispatch,
        }
    }
}

/// Record a run that holds the dispatches of indirect counted stages in one compute pass, in order: the plain
/// dispatches (an args dispatch among them) as dispatches, the counted ones launched from their args buffers. A
/// dispatch of the pass sees the writes of the ones before it (wgpu tracks the usage of each dispatch and inserts the
/// barrier), so the args dispatch's write reaches the indirect launch after it.
fn record_merged_run(
    encoder: &mut wgpu::CommandEncoder,
    run: &[Item<'_>],
    stamps: Option<PassTimestamps<'_>>,
) {
    let mut pass = begin_compute(encoder, FRAME_PASS_LABEL, stamps);
    for item in run {
        match *item {
            Item::Plain(dispatch) => dispatch.record_compute(&mut pass),
            Item::Indirect(dispatch, args) => record_indirect(&mut pass, dispatch, args),
        }
    }
    crate::count_passes(1);
}

/// What the stages of a frame record as: a run of compute dispatches in one pass, a dispatch that is not a compute one
/// (a render pass of its own), or, under `force_pass_per_dispatch` only, an indirect counted stage (its two passes).
enum Segment<'a> {
    Run(Vec<Item<'a>>),
    Single(&'a PreparedDispatch),
    Counted(&'a GpuStage, &'a wgpu::Buffer),
}

/// The segments of `stages`, in order. A stage without a dispatch adds nothing and cuts no run. An indirect counted
/// stage joins the run, its args dispatch then its counted dispatch, unless `per_dispatch` (`force_pass_per_dispatch`
/// is on): then it is a segment of its own, recorded as two passes.
fn segments_of<'a>(stages: &[&'a GpuStage], per_dispatch: bool) -> Vec<Segment<'a>> {
    let mut segments = Vec::new();
    let mut run: Vec<Item<'a>> = Vec::new();
    for &stage in stages {
        if let Some(args) = &stage.indirect {
            if per_dispatch {
                if !run.is_empty() {
                    segments.push(Segment::Run(std::mem::take(&mut run)));
                }
                segments.push(Segment::Counted(stage, args));
            } else {
                let [grid, counted] = &stage.dispatches[..] else {
                    unreachable!(
                        "an indirect counted stage holds the args dispatch and the counted one"
                    )
                };
                run.push(Item::Plain(grid));
                run.push(Item::Indirect(counted, args));
            }
            continue;
        }
        for dispatch in &stage.dispatches {
            if dispatch.is_compute() {
                run.push(Item::Plain(dispatch));
            } else {
                if !run.is_empty() {
                    segments.push(Segment::Run(std::mem::take(&mut run)));
                }
                segments.push(Segment::Single(dispatch));
            }
        }
    }
    if !run.is_empty() {
        segments.push(Segment::Run(run));
    }
    segments
}

/// A copy that a frame will record: the staging buffer that holds the bytes (made when the frame is written), and the
/// buffer it goes into.
struct Upload {
    staging: wgpu::Buffer,
    target: wgpu::Buffer,
    bytes: u64,
}

/// A frame being written: its copies in the order of the calls, and the staging buffers of the ones already recorded,
/// which stay alive until the frame has run. The copies are recorded when the frame is, into its own encoder or a
/// caller's.
#[doc(hidden)]
#[derive(Default)]
pub struct GpuFrame {
    uploads: Vec<Upload>,
    staging: Vec<wgpu::Buffer>,
}

/// A frame recorded into a caller's encoder: what it keeps alive until the caller reports its submission, and the
/// buffer that holds the epoch word the encoder copies when it runs.
#[doc(hidden)]
pub struct GpuEncoded {
    staging: Vec<wgpu::Buffer>,
    epoch: wgpu::Buffer,
}

/// A frame that was submitted: the queue's index of it, whether the queue has finished it, what it must keep alive until
/// it has run, and, for a frame the caller submitted, the check that its encoder ran.
#[doc(hidden)]
#[derive(Clone)]
pub struct GpuSubmission {
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(super) index: wgpu::SubmissionIndex,
    done: Arc<Done>,
    pub(super) epoch: Option<Arc<Epoch>>,
    _staging: Arc<Vec<wgpu::Buffer>>,
}

impl HeadlessDevice {
    pub(super) fn new_frame() -> GpuFrame {
        GpuFrame::default()
    }

    pub(super) fn new_batch_encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GUST batch frame encoder"),
            })
    }

    /// Collect the copy of `bytes` into `target`: the staging buffer is made now, the copy is recorded with the frame.
    fn collect_upload(
        &self,
        frame: &mut GpuFrame,
        target: &wgpu::Buffer,
        bytes: &[u8],
        label: &str,
    ) {
        let Some(staging) = stage_upload(&self.device, bytes, label) else {
            return;
        };
        frame.uploads.push(Upload {
            staging,
            target: target.clone(),
            bytes: bytes.len() as u64,
        });
    }

    /// Collect the copy of `data` into `target`, in the order of the frame's calls.
    pub(super) fn encode_write<T: GpuPod>(
        &self,
        frame: &mut GpuFrame,
        target: &Arc<GpuBuffer<T>>,
        data: &[T],
    ) -> Result<(), Error> {
        self.validate_buffer_device(&**target)?;
        let binding = target.bind(Access::ReadWrite, LengthRule::Shared);
        self.collect_upload(
            frame,
            binding.buffer(),
            pod_slice_as_bytes(data),
            "GUST batch write",
        );
        Ok(())
    }

    /// Collect the copy of `value` into the uniform `target`.
    pub(super) fn encode_uniform<T: GpuPod, U>(
        &self,
        frame: &mut GpuFrame,
        target: &U,
        value: &T,
    ) -> Result<(), Error>
    where
        U: Resource<Element = T, Device = Self, Kind = gust::kind::Uniform>,
    {
        let binding = target.bind(Access::ReadOnly, LengthRule::Independent);
        if binding.device_id != self.id {
            return Err(Error::ForeignPersistentBuffer);
        }
        self.collect_upload(
            frame,
            binding.buffer(),
            pod_slice_as_bytes(std::slice::from_ref(value)),
            "GUST batch extent record",
        );
        Ok(())
    }

    /// Record the collected copies into `encoder`, in order.
    pub(super) fn record_uploads(frame: &mut GpuFrame, encoder: &mut wgpu::CommandEncoder) {
        for Upload {
            staging,
            target,
            bytes,
        } in frame.uploads.drain(..)
        {
            record_upload_copy(encoder, &staging, &target, bytes);
            frame.staging.push(staging);
        }
    }

    /// Record the pass of a stage.
    pub(super) fn encode_stage(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stage: &GpuStage,
    ) -> Result<(), Error> {
        match &stage.indirect {
            Some(args) => self.record_counted(encoder, &stage.dispatches, args, None),
            None => self.record_prepared(encoder, &stage.dispatches),
        }
    }

    /// Record the stages of a frame in order, the compute dispatches of neighbouring stages (and a library op's
    /// dispatches, a bounded counted stage's, and both dispatches of an indirect counted stage) in one compute pass. A
    /// `pixel` dispatch keeps its render pass and ends the run before it; under `force_pass_per_dispatch` every
    /// dispatch has a pass of its own. `stamps` write `begin` at the start of the first pass recorded and `end` at the
    /// end of the last.
    pub(super) fn encode_stages(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stages: &[&GpuStage],
        stamps: Option<(&wgpu::QuerySet, Option<u32>, Option<u32>)>,
    ) -> Result<(), Error> {
        let segments = segments_of(stages, pass_per_dispatch_forced());
        let last = segments.len().saturating_sub(1);
        for (index, segment) in segments.iter().enumerate() {
            let (begin, end) = stamps.map_or((None, None), |(_, begin, end)| {
                (
                    (index == 0).then_some(begin).flatten(),
                    (index == last).then_some(end).flatten(),
                )
            });
            let pass_stamps = stamps.and_then(|(query_set, _, _)| {
                (begin.is_some() || end.is_some()).then_some(PassTimestamps {
                    query_set,
                    begin,
                    end,
                })
            });
            match segment {
                Segment::Run(run) if run.iter().any(|item| matches!(item, Item::Indirect(..))) => {
                    record_merged_run(encoder, run, pass_stamps);
                }
                Segment::Run(run) => {
                    let plain: Vec<&PreparedDispatch> = run.iter().map(Item::dispatch).collect();
                    encode_batch_passes(
                        &self.device,
                        encoder,
                        FRAME_PASS_LABEL,
                        &plain,
                        pass_stamps,
                    )?;
                }
                Segment::Single(dispatch) => {
                    crate::count_passes(1);
                    dispatch.encode(&self.device, encoder, FRAME_PASS_LABEL, pass_stamps)?;
                }
                Segment::Counted(stage, args) => self.record_counted(
                    encoder,
                    &stage.dispatches,
                    args,
                    stamps.map(|(query_set, _, _)| (query_set, begin, end)),
                )?,
            }
        }
        Ok(())
    }

    /// Finish the encoder and make the frame's one counted queue submission.
    pub(super) fn submit_one(
        &self,
        frame: GpuFrame,
        encoder: wgpu::CommandEncoder,
    ) -> GpuSubmission {
        let index = self.submit_counted([encoder.finish()]);
        self.submission_of(index, frame.staging, None)
    }

    /// The caller's encoder holds the frame: record the epoch copy, and keep the staging buffers alive.
    pub(super) fn encoded_one(
        &self,
        frame: GpuFrame,
        encoder: &mut wgpu::CommandEncoder,
    ) -> GpuEncoded {
        // One source buffer per device; only the buffer the encoder copies it into is the frame's own.
        let source = self.pool.epoch_source(&self.device);
        let epoch = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GUST batch epoch readback"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        record_upload_copy(encoder, &source, &epoch, 4);
        GpuEncoded {
            staging: frame.staging,
            epoch,
        }
    }

    /// The caller submitted the encoder (`index` is what its `queue.submit` returned): start reading the epoch word.
    pub(super) fn encoded_submission(
        &self,
        encoded: GpuEncoded,
        index: &wgpu::SubmissionIndex,
    ) -> GpuSubmission {
        let GpuEncoded { staging, epoch } = encoded;
        self.submission_of(index.clone(), staging, Some(Epoch::start(epoch)))
    }

    fn submission_of(
        &self,
        index: wgpu::SubmissionIndex,
        staging: Vec<wgpu::Buffer>,
        epoch: Option<Arc<Epoch>>,
    ) -> GpuSubmission {
        // The callback is registered right after the submission, so it fires when this frame has run, not when a later
        // one has: a batch with frames in flight waits for the oldest of them.
        let done = Arc::new(Done::default());
        let callback = Arc::clone(&done);
        self.queue.on_submitted_work_done(move || callback.finish());
        // The staging buffers are shared with the later clones of the submission; on `wasm32` wgpu's handles are not
        // `Send`, and the submission stays on its thread.
        #[allow(clippy::arc_with_non_send_sync)]
        let staging = Arc::new(staging);
        GpuSubmission {
            index,
            done,
            epoch,
            _staging: staging,
        }
    }

    /// A future that completes when `submission` has run: the queue's callback of that submission, completed while the
    /// future's polls poll the device (as a read-back's mapping is). It never waits.
    pub(super) fn completion_of(
        &self,
        submission: &GpuSubmission,
    ) -> BoxFuture<Result<(), gust::map::Error>> {
        let device = self.device.clone();
        let done = Arc::clone(&submission.done);
        Box::pin(core::future::poll_fn(move |cx| {
            if let Err(error) = device.poll(wgpu::PollType::Poll) {
                return Poll::Ready(Err(map_error(Error::Poll(error.to_string()))));
            }
            done.poll(cx.waker()).map(Ok)
        }))
    }
}

/// Whether the queue has finished a submission, and the tasks waiting for it.
#[derive(Default)]
pub(super) struct Done {
    finished: AtomicBool,
    wakers: Mutex<Vec<Waker>>,
}

impl Done {
    pub(super) fn finish(&self) {
        let wakers = {
            let mut wakers = self.wakers.lock().unwrap_or_else(PoisonError::into_inner);
            self.finished.store(true, Ordering::Release);
            core::mem::take(&mut *wakers)
        };
        for waker in wakers {
            waker.wake();
        }
    }

    pub(super) fn poll(&self, waker: &Waker) -> Poll<()> {
        if self.finished.load(Ordering::Acquire) {
            return Poll::Ready(());
        }
        let mut wakers = self.wakers.lock().unwrap_or_else(PoisonError::into_inner);
        if self.finished.load(Ordering::Acquire) {
            return Poll::Ready(());
        }
        if !wakers.iter().any(|stored| stored.will_wake(waker)) {
            wakers.push(waker.clone());
        }
        Poll::Pending
    }
}

/// The epoch word of a frame the caller submitted: read once, shared by the frame's tickets. A word that is not
/// [`EPOCH`] means the copy never ran, so the frame's encoder was not submitted and the tickets' staging buffers hold
/// what an earlier frame left in them.
pub(super) struct Epoch {
    done: Done,
    outcome: Mutex<Option<Result<bool, String>>>,
}

impl Epoch {
    /// Request the mapping of the epoch buffer; the callback reads the word.
    fn start(buffer: wgpu::Buffer) -> Arc<Self> {
        let epoch = Arc::new(Self {
            done: Done::default(),
            outcome: Mutex::new(None),
        });
        let landed = Arc::clone(&epoch);
        let mapped = buffer.clone();
        buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            let outcome = result.map_err(|error| error.to_string()).and_then(|()| {
                let view = mapped
                    .get_mapped_range(..)
                    .map_err(|error| error.to_string())?;
                let ran = view[..] == EPOCH.to_le_bytes();
                drop(view);
                mapped.unmap();
                Ok(ran)
            });
            *landed
                .outcome
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = Some(outcome);
            landed.done.finish();
        });
        epoch
    }

    /// Whether the encoder ran, once the word has been read.
    pub(super) fn poll(&self, waker: &Waker) -> Poll<Result<bool, String>> {
        self.done.poll(waker).map(|()| {
            self.outcome
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
                .expect("the callback stores the outcome before it finishes")
        })
    }
}
