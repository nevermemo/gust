//! A ticket's read-back: the staging buffer taken from the pool when the ticket is reserved, the copy recorded into the
//! frame's encoder after the stages, and the mapping requested once the frame is submitted.

use std::sync::Arc;

use core::task::Poll;
use gust::GpuPod;
use gust::batch::BoxFuture;

use super::frame::GpuSubmission;
use super::map_error;
use crate::pool::StagedReadback;
use crate::{Error, GpuBuffer, HeadlessDevice};

/// A reserved read-back: the buffer it copies and the staging buffer it copies into.
#[doc(hidden)]
pub struct GpuRead<T> {
    source: Arc<GpuBuffer<T>>,
    staged: StagedReadback<T>,
}

impl<T: GpuPod> GpuRead<T> {
    /// Record the copy of the source into the staging buffer.
    pub(super) fn encode_copy(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.staged.encode_copy(encoder, &self.source);
    }
}

impl HeadlessDevice {
    /// Take the staging buffer of a read-back of `source` (nothing is recorded).
    pub(super) fn reserve<T: GpuPod>(
        &self,
        source: &Arc<GpuBuffer<T>>,
    ) -> Result<GpuRead<T>, Error> {
        Ok(GpuRead {
            staged: self.stage_pooled_readback(source)?,
            source: Arc::clone(source),
        })
    }

    /// Request the mapping, after the submission that carried the copy, and return the future of the values: `None`
    /// when the submission was a caller's and its epoch word says the encoder never ran.
    pub(super) fn start_ticket<T: GpuPod + Send>(
        &self,
        read: GpuRead<T>,
        submission: &GpuSubmission,
    ) -> BoxFuture<Result<Option<Vec<T>>, gust::map::Error>> {
        let GpuRead { staged, source } = read;
        let readback = staged.start();
        // The source stays alive until its copy was submitted; the mapping needs only the staging buffer.
        drop(source);
        let device = self.device.clone();
        let epoch = submission.epoch.clone();
        Box::pin(async move {
            // The mapping is awaited whether or not the copy ran: a staging buffer is not given back with a mapping
            // still requested. When the encoder's epoch word says the copy never ran, the buffer is retired, not
            // recycled: the caller may still submit that encoder (the doc of `Encoded` says it must not).
            let Some(epoch) = epoch else {
                return Ok(Some(readback.finish(&device).await.map_err(map_error)?));
            };
            let polling = device.clone();
            let ran = core::future::poll_fn(move |cx| {
                if let Err(error) = polling.poll(wgpu::PollType::Poll) {
                    return Poll::Ready(Err(Error::Poll(error.to_string())));
                }
                epoch.poll(cx.waker()).map_err(Error::Map)
            });
            readback
                .finish_if_ran(&device, ran)
                .await
                .map_err(map_error)
        })
    }
}
