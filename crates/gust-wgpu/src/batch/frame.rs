//! One frame: a command encoder that collects the copies and the passes, and its one submission.

use std::sync::Arc;

use gust::batch::BoxFuture;
use gust::{Access, GpuPod, LengthRule, Resource, pod_slice_as_bytes};

use super::{GpuStage, map_error};
use crate::encode::encode_upload;
use crate::{Completion, Error, GpuBuffer, HeadlessDevice, MappingReady};

/// A frame being recorded: the encoder, and the staging buffers of its encoded copies, which stay alive until the frame
/// has run.
#[doc(hidden)]
pub struct GpuFrame {
    pub(super) encoder: wgpu::CommandEncoder,
    staging: Vec<wgpu::Buffer>,
}

/// A frame that was submitted: the queue's index of it, and what it must keep alive until it has run.
#[doc(hidden)]
#[derive(Clone)]
pub struct GpuSubmission {
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(super) index: wgpu::SubmissionIndex,
    _staging: Arc<Vec<wgpu::Buffer>>,
}

impl HeadlessDevice {
    pub(super) fn new_frame(&self) -> GpuFrame {
        GpuFrame {
            encoder: self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("GUST batch frame encoder"),
                }),
            staging: Vec::new(),
        }
    }

    /// Record the copy of `data` into `target`, in the order of the frame's calls.
    pub(super) fn encode_write<T: GpuPod>(
        &self,
        frame: &mut GpuFrame,
        target: &Arc<GpuBuffer<T>>,
        data: &[T],
    ) -> Result<(), Error> {
        self.validate_buffer_device(&**target)?;
        let binding = target.bind(Access::ReadWrite, LengthRule::Shared);
        frame.staging.extend(encode_upload(
            &self.device,
            &mut frame.encoder,
            &binding,
            pod_slice_as_bytes(data),
            "GUST batch write",
        ));
        Ok(())
    }

    /// Record the copy of `value` into the uniform `target`.
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
        frame.staging.extend(encode_upload(
            &self.device,
            &mut frame.encoder,
            &binding,
            pod_slice_as_bytes(std::slice::from_ref(value)),
            "GUST batch extent record",
        ));
        Ok(())
    }

    /// Record the pass of a stage.
    pub(super) fn encode_stage(&self, frame: &mut GpuFrame, stage: &GpuStage) -> Result<(), Error> {
        self.record_prepared(&mut frame.encoder, &stage.dispatches)
    }

    /// Finish the encoder and make the frame's one counted queue submission.
    pub(super) fn submit_one(&self, frame: GpuFrame) -> GpuSubmission {
        let GpuFrame { encoder, staging } = frame;
        let index = self.submit_counted([encoder.finish()]);
        // The staging buffers are shared with the later clones of the submission; on `wasm32` wgpu's handles are not
        // `Send`, and the submission stays on its thread.
        #[allow(clippy::arc_with_non_send_sync)]
        let staging = Arc::new(staging);
        GpuSubmission {
            index,
            _staging: staging,
        }
    }

    /// A future that completes when the work submitted so far has run: the queue's callback, completed while the
    /// future's polls poll the device (as a read-back's mapping is).
    pub(super) fn completion_of(
        &self,
        _submission: &GpuSubmission,
    ) -> BoxFuture<Result<(), gust::map::Error>> {
        let device = self.device.clone();
        let mapped = Completion::new();
        let callback = mapped.clone();
        self.queue.on_submitted_work_done(move || {
            let _ = callback.complete(Ok(()));
        });
        Box::pin(async move {
            MappingReady {
                device: &device,
                mapped,
            }
            .await
            .map_err(map_error)
        })
    }
}
