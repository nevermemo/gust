//! A ticket's read-back: the staging buffer taken from the pool when the ticket is reserved, the copy recorded into the
//! frame's encoder after the stages, and the mapping requested once the frame is submitted.

use std::sync::Arc;

use gust::GpuPod;
use gust::batch::BoxFuture;

use super::{GpuFrame, map_error};
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
    pub(super) fn encode_copy(&mut self, frame: &mut GpuFrame) {
        self.staged.encode_copy(&mut frame.encoder, &self.source);
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

    /// Request the mapping, after the submission that carried the copy, and return the future of the values.
    pub(super) fn start_ticket<T: GpuPod + Send>(
        &self,
        read: GpuRead<T>,
    ) -> BoxFuture<Result<Vec<T>, gust::map::Error>> {
        let GpuRead { staged, source } = read;
        let readback = staged.start();
        // The source stays alive until its copy was submitted; the mapping needs only the staging buffer.
        drop(source);
        let device = self.device.clone();
        Box::pin(async move { readback.finish(&device).await.map_err(map_error) })
    }
}
