//! One frame: the copies it collects while it is written, the encoder it is recorded into (its own, or a caller's), and
//! its one submission.

use core::task::{Poll, Waker};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use gust::batch::BoxFuture;
use gust::{Access, GpuPod, LengthRule, Resource, pod_slice_as_bytes};

use super::{GpuStage, map_error};
use crate::encode::{record_upload_copy, stage_upload};
use crate::pool::EPOCH;
use crate::{Error, GpuBuffer, HeadlessDevice};

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
