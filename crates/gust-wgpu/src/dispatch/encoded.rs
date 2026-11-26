//! Batches recorded into a caller's command encoder (m14.encb): [`HeadlessDevice::encode_batch`] and the
//! [`EncodedJob`] it returns. [`HeadlessDevice::submit_batch`] is built from the same two steps.

use crate::{BufferDispatch, Error, HeadlessDevice, PassTimestamps, PreparedDispatch, WgpuJob};
#[cfg(target_arch = "wasm32")]
use std::sync::{Arc, atomic::Ordering};

/// The label of every pass `encode_batch` records.
const BATCH_PASS_LABEL: &str = "GUST typed batched pass";

impl HeadlessDevice {
    /// Record a batch into `encoder`, as one compute pass when every dispatch is a `grid` dispatch and one pass per
    /// dispatch otherwise, and never submit it. This is the form for a caller that
    /// records its own passes on a shared device ([`adopt`](Self::adopt)) and wants GUST work and its own in one
    /// command buffer and one queue submission: encode GUST's batch, record your own commands, encode another batch,
    /// call `finish` and `queue.submit` once, then hand the [`SubmissionIndex`](wgpu::SubmissionIndex) to
    /// [`EncodedJob::submitted`], which returns the same [`WgpuJob`] that [`submit_batch`](Self::submit_batch) does.
    ///
    /// ```no_run
    /// # use gust_wgpu::{BufferDispatch, HeadlessDevice};
    /// # fn frame(
    /// #     device: &HeadlessDevice,
    /// #     first: &[BufferDispatch<'_>],
    /// #     second: &[BufferDispatch<'_>],
    /// # ) -> Result<(), gust_wgpu::Error> {
    /// let mut encoder = device
    ///     .wgpu_device()
    ///     .create_command_encoder(&gust_wgpu::wgpu::CommandEncoderDescriptor::default());
    /// let a = device.encode_batch(&mut encoder, first)?;
    /// // ... your own copies and passes, recorded into `encoder` ...
    /// let b = device.encode_batch(&mut encoder, second)?;
    /// let index = device.wgpu_queue().submit([encoder.finish()]);
    /// let (a, b) = (a.submitted(&index), b.submitted(&index));
    /// a.wait()?;
    /// b.wait()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// The batch means what [`submit_batch`](Self::submit_batch) means: the same validation, the same order (every
    /// dispatch sees the writes of the one before it, in one pass or across passes), and an empty request is valid and
    /// records nothing.
    ///
    /// # Errors leave the encoder alone, with one exception
    ///
    /// Every fallible step that does not need the encoder runs before the first pass is recorded: the validation of
    /// every dispatch against the kernel and the device's limits, the pipeline lookup or compilation, and the bind
    /// groups. An `Err` from any of those leaves `encoder` exactly as it was, so you can go on recording into it.
    /// The one step that can only fail while recording is the attachment of a `pixel` call
    /// ([`Error::OutOfMemory`] named `GUST pixel attachment`): earlier dispatches of the same batch may already be in
    /// the encoder, so on that error **discard the encoder** (and with it the batch and whatever else it holds).
    ///
    /// # Ordering
    ///
    /// All of this is the single-queue order of wgpu, which GUST does not add to or change.
    ///
    /// - **Submission order, not completion.** The submissions of one queue behave as if they ran in the order they
    ///   were submitted. A later submission sees the writes of an earlier one, with no barrier of yours. That is an
    ///   order, not a promise that the earlier work has finished before the later work starts. Inside one command
    ///   buffer the order is the order of recording: your pass between two GUST batches sees the first batch's writes
    ///   and the second batch sees yours.
    /// - **`queue.write_buffer` lands before the next submission.** A write you queue after this call and before your
    ///   `queue.submit` is applied before the encoder's commands run, so the batch recorded here sees it, although it
    ///   was recorded earlier. To order a write after a batch, submit the batch first.
    /// - **Nothing runs until you submit.** This call records; [`EncodedJob::submitted`] after the submission is
    ///   what makes a job. A job made with a wrong index (another submission's, or one from before your
    ///   `queue.submit`) is not detected: it waits for that other submission, and a wait on it proves nothing about
    ///   this batch. An encoding that is never submitted does nothing and its job is never made.
    /// - **Callbacks and mappings.** A `map_async` or `on_submitted_work_done` callback, and the completion of a
    ///   GUST readback or job, fire when the device is polled: by a blocking GUST call, by a poll of yours, or by
    ///   wgpu's own maintenance, which may run inside `queue.submit`. A mapping you request on a buffer this
    ///   encoding writes is valid only after your submission. Do not hold a lock across `queue.submit` that a callback
    ///   of yours takes.
    /// - **GUST submits on its own account** in these calls, each counted by
    ///   [`gust_submissions`](Self::gust_submissions), which names them: [`submit_batch`](Self::submit_batch) and the
    ///   typed `submit` and `dispatch` built on it, [`submit_batch_read`](Self::submit_batch_read),
    ///   `dispatch_batch_timed`, every read-back (the inspect read-back included),
    ///   `write_typed_buffer_timed`, the deprecated `dispatch_f32` and `dispatch_u32`, a recycled zeroed buffer (the
    ///   clear), a `GpuPool` growth and `execute_graph`. Each lands after what you submitted before
    ///   it. `encode_batch` and `EncodedJob::submitted` submit nothing.
    /// - **Keep the buffers alive until your `queue.submit`.** The encoding does not hold the buffers it binds, nor
    ///   the `read` and `clear` buffers of [`encode_batch_read`](Self::encode_batch_read). A pooled buffer you drop
    ///   before your submission can be handed out again and written (`queue.write_buffer` lands before the
    ///   submission), and the batch then overwrites that data.
    /// - **GUST cannot check that `encoder` belongs to this device.** Like the one-instance rule of
    ///   [`adopt`](Self::adopt), an encoder made on another device, or on another instance, is the caller's to
    ///   avoid: wgpu reports it at `finish` or `submit`, far from this call, or panics. The resources must be this
    ///   device's too (that part is checked: [`Error::ForeignPersistentBuffer`]).
    ///
    /// There is no timed form of this call, on purpose: the timestamp profiler's lock is held from the recording to the
    /// read of the result, which cannot span a submission that you make. Use
    /// `dispatch_batch_timed` when you need GPU time of a batch; it submits itself.
    pub fn encode_batch<'device>(
        &'device self,
        encoder: &mut wgpu::CommandEncoder,
        dispatches: &[BufferDispatch<'_>],
    ) -> Result<EncodedJob<'device>, Error> {
        let prepared = self.prepare_extent_batch(dispatches)?;
        self.record_prepared(encoder, &prepared)?;
        Ok(self.encoded_job(!prepared.is_empty()))
    }

    /// The one place a batch's passes are recorded: `encode_batch`, `submit_batch` and the read-back forms all end
    /// here.
    pub(crate) fn record_prepared(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        prepared: &[PreparedDispatch],
    ) -> Result<(), Error> {
        self.record_prepared_stamped(encoder, prepared, None)
    }

    /// [`record_prepared`](Self::record_prepared) with timestamp writes: the first pass writes slot `begin` of the
    /// query set and the last pass slot `end` (the same pass, when the batch is one), where given (a frame's timing
    /// hook, which times the passes that exist and
    /// never changes them). `None` writes none.
    pub(crate) fn record_prepared_stamped(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        prepared: &[PreparedDispatch],
        stamps: Option<(&wgpu::QuerySet, Option<u32>, Option<u32>)>,
    ) -> Result<(), Error> {
        let timestamps = stamps.and_then(|(query_set, begin, end)| {
            (begin.is_some() || end.is_some()).then_some(PassTimestamps {
                query_set,
                begin,
                end,
            })
        });
        crate::encode_batch_passes(
            &self.device,
            encoder,
            BATCH_PASS_LABEL,
            prepared,
            timestamps,
        )
    }

    /// An [`EncodedJob`] of this device; `recorded` says whether any pass went into the encoder.
    pub(crate) const fn encoded_job(&self, recorded: bool) -> EncodedJob<'_> {
        EncodedJob {
            device: self,
            recorded,
        }
    }
}

/// A batch recorded into a caller's encoder by [`HeadlessDevice::encode_batch`], waiting for the caller's
/// submission. Call [`submitted`](Self::submitted) with that submission's index to get the job.
///
/// Dropping it, or the encoder, without a submission records nothing anywhere else: no job exists and nothing runs.
#[must_use = "call `submitted` with the index of the submission that carries the encoder"]
pub struct EncodedJob<'device> {
    device: &'device HeadlessDevice,
    /// Whether any pass was recorded: an empty batch recorded none, and its job is complete at once.
    recorded: bool,
}

impl<'device> EncodedJob<'device> {
    /// The job of this batch, after `index` of the `queue.submit` that carried the encoder it was recorded into.
    ///
    /// Like [`HeadlessDevice::submit_batch`]'s, the job waits for exactly that submission on native
    /// ([`WgpuJob::wait`], [`WgpuJob::status`]); on `wasm32` it registers `on_submitted_work_done` now, after the
    /// submission, and reports `Complete` once the callback has run (see [`WgpuJob`]). The index is not checked: pass
    /// the one `queue.submit` returned for the command buffer that holds the batch, and call this after it, not
    /// before. A batch that recorded nothing (an empty request) has a job that is complete at once, whatever the index.
    #[must_use = "GPU work is asynchronous; poll, await a readback or wait on the returned job when completion matters"]
    pub fn submitted(self, index: &wgpu::SubmissionIndex) -> WgpuJob<'device> {
        if !self.recorded {
            return self.nothing_submitted();
        }
        #[cfg(target_arch = "wasm32")]
        let done = self.completion_flag();
        WgpuJob {
            device: &self.device.device,
            submission: Some(index.clone()),
            #[cfg(target_arch = "wasm32")]
            done,
        }
    }

    /// The job of a batch that recorded nothing: no submission, complete at once.
    pub(crate) fn nothing_submitted(self) -> WgpuJob<'device> {
        WgpuJob {
            device: &self.device.device,
            submission: None,
            #[cfg(target_arch = "wasm32")]
            done: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }

    /// The browser cannot poll, so the job learns of completion from the queue (`WgpuJob::status`).
    #[cfg(target_arch = "wasm32")]
    fn completion_flag(&self) -> Arc<std::sync::atomic::AtomicBool> {
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&done);
        self.device
            .queue
            .on_submitted_work_done(move || flag.store(true, Ordering::Release));
        done
    }
}

impl std::fmt::Debug for EncodedJob<'_> {
    /// Whether any pass was recorded; never the raw wgpu handles.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodedJob")
            .field("recorded", &self.recorded)
            .finish_non_exhaustive()
    }
}
