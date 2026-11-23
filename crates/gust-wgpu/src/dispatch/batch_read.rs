//! A batch and the read-back of one of its buffers in one command buffer (m14.encb): one queue submission where a
//! submit plus a read-back makes two, and three when a recycled output is zeroed first.

#[cfg(not(target_arch = "wasm32"))]
use crate::block_on_mapping;
use crate::pool::{PooledReadback, StagedReadback};
use crate::{
    BufferBinding, BufferDispatch, Error, GpuBuffer, GpuBufferAccess, HeadlessDevice,
    PreparedDispatch,
};
use gust::{Access, BindingKind, GpuPod};

impl HeadlessDevice {
    /// A pooled read-write buffer of `len` elements that is **not** zeroed here: a fresh allocation reads as zero, a
    /// recycled one holds what its previous user left. Pass it in the `clear` list of
    /// [`submit_batch_read`](Self::submit_batch_read) or [`encode_batch_read`](Self::encode_batch_read), which zero it
    /// inside the one command buffer they record, and so make no submission for it. This call itself submits nothing
    /// ([`GpuVec::zeroed_on`](crate::GpuVec::zeroed_on) and the map calls' zeroed outputs submit the clear of a
    /// recycled buffer at once, as a submission of its own).
    ///
    /// A buffer that is read or bound before it is cleared this way shows stale contents: nothing detects it.
    pub fn uncleared_buffer<T: GpuPod>(&self, len: usize) -> Result<GpuBuffer<T>, Error> {
        self.pooled_buffer("batch output", len, crate::pool::Fill::Uncleared)
    }

    /// Zero the `clear` buffers, run `dispatches` as one batch and copy `read` into a pooled staging buffer, all in one
    /// command buffer and **one** counted submission ([`gust_submissions`](Self::gust_submissions) rises by 1; by 0
    /// when there is nothing to record: no work, no clear and an empty `read`). The returned [`PendingRead`] is the
    /// read-back, already requested; await [`finish`](PendingRead::finish) or, on native, [`wait`](PendingRead::wait).
    ///
    /// Compared with `submit_batch` and a read-back this saves a submission, and a third when a recycled output has to
    /// be zeroed first. The batch means what [`submit_batch`](Self::submit_batch) means (validation, one pass per
    /// dispatch, each seeing the writes of the one before it); the clears come before the first pass and the copy
    /// after the last, in the order of recording.
    ///
    /// `clear` holds buffer bindings of this device whose buffers were created read-write (the access of the binding
    /// is not read): each is zeroed over its logical length. A read-only buffer or a buffer of another device is
    /// refused ([`Error::PersistentBufferAccess`], [`Error::ForeignPersistentBuffer`]) before anything is recorded.
    /// `read` must be readable ([`Error::PersistentBufferNotReadable`] otherwise) and of this device; an empty `read`
    /// has an empty result and no copy.
    ///
    /// Every fallible step happens before the first command is recorded, so an `Err` submits nothing and leaves
    /// [`gust_submissions`](Self::gust_submissions) as it was, including the attachment of a `pixel` call
    /// ([`Error::OutOfMemory`]), which is made while recording: the encoder is this call's own and is dropped.
    pub fn submit_batch_read<T: GpuPod>(
        &self,
        dispatches: &[BufferDispatch<'_>],
        clear: &[BufferBinding<'_>],
        read: &GpuBuffer<T>,
    ) -> Result<PendingRead<T>, Error> {
        self.check_clear(clear)?;
        let prepared = self.prepare_extent_batch(dispatches)?;
        let mut staged = self.stage_pooled_readback(read)?;
        if prepared.is_empty() && clear.is_empty() && staged.is_empty() {
            return Ok(PendingRead {
                readback: staged.start(),
                submission: None,
                device_id: self.id,
            });
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GUST batch read encoder"),
            });
        self.record_read(&mut encoder, &prepared, clear, &mut staged, read)?;
        let submission = self.submit_counted([encoder.finish()]);
        Ok(PendingRead {
            readback: staged.start(),
            submission: Some(submission),
            device_id: self.id,
        })
    }

    /// [`submit_batch_read`](Self::submit_batch_read) into `encoder`, which you submit: the clears, the batch and the
    /// copy of `read` are recorded in that order and nothing is submitted (this call makes 0 counted submissions).
    /// After your `queue.submit`, call [`EncodedRead::submitted`] with its index to request the read-back.
    ///
    /// The fallible steps, the exception for a `pixel` attachment (discard the encoder on that error) and the ordering
    /// rules (including keeping `read`, `clear` and every bound buffer alive until your submission) are those of
    /// [`encode_batch`](Self::encode_batch). Dropping the [`EncodedRead`] without `submitted` makes no read-back; the
    /// commands you submit still run, and the staging buffer, which your encoder still copies into, is retired from
    /// the pool instead of being returned to it ([`PoolStats::retired`](crate::PoolStats::retired)).
    pub fn encode_batch_read<T: GpuPod>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        dispatches: &[BufferDispatch<'_>],
        clear: &[BufferBinding<'_>],
        read: &GpuBuffer<T>,
    ) -> Result<EncodedRead<T>, Error> {
        self.check_clear(clear)?;
        let prepared = self.prepare_extent_batch(dispatches)?;
        let mut staged = self.stage_pooled_readback(read)?;
        self.record_read(encoder, &prepared, clear, &mut staged, read)?;
        Ok(EncodedRead {
            staged,
            device_id: self.id,
        })
    }

    /// Refuse a `clear` binding that is not a read-write buffer of this device, before anything is recorded.
    fn check_clear(&self, clear: &[BufferBinding<'_>]) -> Result<(), Error> {
        for binding in clear {
            if binding.device_id != self.id {
                return Err(Error::ForeignPersistentBuffer);
            }
            if binding.texture.is_some()
                || binding.kind != BindingKind::Storage
                || binding.buffer_access != GpuBufferAccess::ReadWrite
            {
                return Err(Error::PersistentBufferAccess {
                    parameter: "clear".to_owned(),
                    buffer: binding.buffer_access,
                    binding: Access::ReadWrite,
                });
            }
        }
        Ok(())
    }

    /// Record the clears, the passes and the copy, in that order.
    fn record_read<T: GpuPod>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        prepared: &[PreparedDispatch],
        clear: &[BufferBinding<'_>],
        staged: &mut StagedReadback<T>,
        read: &GpuBuffer<T>,
    ) -> Result<(), Error> {
        for binding in clear {
            encoder.clear_buffer(binding.buffer(), 0, Some(binding.bytes));
        }
        self.record_prepared(encoder, prepared)?;
        staged.encode_copy(encoder, read);
        Ok(())
    }
}

/// A read-back whose copy was recorded and submitted and whose mapping was requested, by
/// [`HeadlessDevice::submit_batch_read`] or [`EncodedRead::submitted`]. It owns its staging buffer until the mapping's
/// callback has fired; dropping it early hands the buffer back to the pool then, and discards the values.
///
/// `finish` and `wait` take the device that made it and refuse another ([`Error::ForeignPersistentBuffer`]), as every
/// call that takes a buffer does; the device is a parameter, not a borrow, so a `PendingRead` can outlive a frame.
#[must_use = "dropping a `PendingRead` discards the values; await `finish` or call `wait`"]
pub struct PendingRead<T> {
    readback: PooledReadback<T>,
    /// The submission that carries the copy, when one was made: what [`wait`](Self::wait) waits for.
    submission: Option<wgpu::SubmissionIndex>,
    /// The id of the device that made it.
    device_id: u64,
}

impl<T: GpuPod> PendingRead<T> {
    fn check_device(&self, device: &HeadlessDevice) -> Result<(), Error> {
        if self.device_id == device.id {
            Ok(())
        } else {
            Err(Error::ForeignPersistentBuffer)
        }
    }

    /// Await the copied values: each poll of the future polls the device once, so a caller that cannot block drives it
    /// from its own loop, and on `wasm32` the browser's event loop drives it.
    ///
    /// # Errors
    ///
    /// [`Error::ForeignPersistentBuffer`] when `device` is not the device that made this read-back.
    pub async fn finish(self, device: &HeadlessDevice) -> Result<Vec<T>, Error> {
        self.check_device(device)?;
        self.readback.finish(&device.device).await
    }

    /// Block until the copied values are ready. Native only. It waits for the submission that carries the copy, not
    /// for later work of yours on the queue, then for the mapping.
    #[cfg(not(target_arch = "wasm32"))]
    ///
    /// # Errors
    ///
    /// [`Error::ForeignPersistentBuffer`] when `device` is not the device that made this read-back.
    pub fn wait(self, device: &HeadlessDevice) -> Result<Vec<T>, Error> {
        self.check_device(device)?;
        if let Some(submission) = self.submission.clone() {
            device.wait_for_submission(submission)?;
        }
        block_on_mapping(self.readback.finish(&device.device))
    }
}

impl<T> std::fmt::Debug for PendingRead<T> {
    /// Whether a submission carries the copy; never the raw wgpu handles.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingRead")
            .field("submitted", &self.submission.is_some())
            .finish_non_exhaustive()
    }
}

/// The clears, the batch and the read-back copy that [`HeadlessDevice::encode_batch_read`] recorded into a caller's
/// encoder, waiting for the caller's submission. Call [`submitted`](Self::submitted) with that submission's index.
#[must_use = "call `submitted` with the index of the submission that carries the encoder"]
pub struct EncodedRead<T> {
    staged: StagedReadback<T>,
    device_id: u64,
}

impl<T: GpuPod> EncodedRead<T> {
    /// Request the read-back, after `index` of the `queue.submit` that carried the encoder. The mapping is requested
    /// at once and the index is not checked, which has two consequences, neither detected:
    ///
    /// - Called **before** your `queue.submit`, the staging buffer has a mapping pending when your submission uses it,
    ///   and wgpu rejects your submission (on native, a panic from the queue's error handler).
    /// - Called when the encoder is **never submitted**, the mapping resolves at the next poll and the read-back
    ///   returns the stale bytes an earlier read-back left in that pooled staging buffer, as if they were the values
    ///   of `read`.
    ///
    /// A wrong index (another submission's) only changes what [`PendingRead::wait`] waits for.
    pub fn submitted(self, index: &wgpu::SubmissionIndex) -> PendingRead<T> {
        PendingRead {
            readback: self.staged.start(),
            submission: Some(index.clone()),
            device_id: self.device_id,
        }
    }
}
