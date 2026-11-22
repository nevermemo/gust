use crate::completion::Completion;
use crate::{Error, pod_vec_from_bytes};
use gust::GpuPod;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};

/// A staging buffer whose `map_async` has been requested, with the completion its callback fills.
pub(crate) struct PendingMapping {
    pub(crate) readback: wgpu::Buffer,
    pub(crate) mapped: Completion<Result<(), String>>,
}

/// A typed readback after its copy was submitted and its mapping requested. An empty buffer has no mapping.
pub(crate) struct TypedReadback<T> {
    pub(crate) mapping: Option<PendingMapping>,
    pub(crate) element: PhantomData<fn() -> T>,
}

impl<T: GpuPod> TypedReadback<T> {
    /// Block until the device has finished the submitted work. This does not mean this mapping's callback has run:
    /// another thread's poll may have collected it and not fired it yet. The native blocking wrappers call it before
    /// they wait for the completion in `finish` (`block_on_mapping`), so that the wait is short; the completion is
    /// what they rely on.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn wait(&self, device: &wgpu::Device) -> Result<(), Error> {
        if self.mapping.is_some() {
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|error| Error::Poll(error.to_string()))?;
        }
        Ok(())
    }

    /// Await the mapping (`MappingReady`), copy the bytes out and unmap.
    pub(crate) async fn finish(self, device: &wgpu::Device) -> Result<Vec<T>, Error> {
        let Some(PendingMapping { readback, mapped }) = self.mapping else {
            return Ok(Vec::new());
        };
        MappingReady { device, mapped }.await?;
        let view = readback
            .get_mapped_range(..)
            .map_err(|error| Error::Map(error.to_string()))?;
        let values = pod_vec_from_bytes(&view)?;
        drop(view);
        readback.unmap();
        Ok(values)
    }
}

/// Resolves when the `map_async` callback has delivered its result. Every poll polls the device once without
/// waiting (`PollType::Poll`), which is what runs the callback on native; it never waits, because the browser must not
/// block. It stores the task's waker in the completion and never wakes itself.
pub(crate) struct MappingReady<'device> {
    pub(crate) device: &'device wgpu::Device,
    pub(crate) mapped: Completion<Result<(), String>>,
}

impl Future for MappingReady<'_> {
    type Output = Result<(), Error>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            return Poll::Ready(Err(Error::Poll(error.to_string())));
        }
        Pin::new(&mut self.mapped)
            .poll(cx)
            .map(|result| result.map_err(Error::Map))
    }
}
