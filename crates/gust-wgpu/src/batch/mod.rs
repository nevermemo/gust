//! The GPU device's batch (plan m14.s2, slice s2a2): [`HeadlessDevice`] as a `gust::batch::BatchDevice`.
//!
//! A frame is one command encoder. The writes of the frame are encoded copies, the extent records of the stages whose
//! parameters changed are encoded copies, then come the passes of the stages in order, then the read-back copies of the
//! tickets; `submit` finishes the encoder and makes **one** counted queue submission
//! ([`HeadlessDevice::gust_submissions`] rises by 1 whatever the number of stages and tickets). Decision D251.
//!
//! The pieces are by concern: [`build`] seals a stage, [`frame`] records a frame and submits it, [`ticket`] reserves
//! and starts the read-backs. `shape` and `ops` hold what later slices add.

use std::sync::Arc;

use gust::batch::{BatchDevice, BoxFuture};
use gust::map::{self, MapHost};
use gust::{Call, GpuPod, Kernel, Resource, Run, kind};

use crate::{Error, GpuBuffer, HeadlessDevice};

mod build;
mod frame;
mod ops;
mod shape;
mod slot;
mod ticket;

pub use build::GpuStage;
pub use frame::{GpuFrame, GpuSubmission};
pub use ticket::GpuRead;

impl BatchDevice for HeadlessDevice {
    type Prepared = GpuStage;
    type Frame = GpuFrame;
    type Read<T: GpuPod> = GpuRead<T>;
    type Submission = GpuSubmission;

    fn prepare<K, R, const N: usize>(
        &self,
        keep: R,
        build: impl for<'o> Fn(&'o R) -> Call<'o, Self, K, N> + 'static,
    ) -> Result<GpuStage, Error>
    where
        Self: Run<K, N>,
        K: Kernel,
        R: 'static,
    {
        self.prepare_stage(keep, build)
    }

    fn begin_frame(&self) -> Result<GpuFrame, Error> {
        Ok(self.new_frame())
    }

    fn frame_write<T: GpuPod>(
        &self,
        frame: &mut GpuFrame,
        target: &Arc<GpuBuffer<T>>,
        data: &[T],
    ) -> Result<(), Error> {
        self.encode_write(frame, target, data)
    }

    fn frame_uniform<T: GpuPod, U>(
        &self,
        frame: &mut GpuFrame,
        target: &Arc<U>,
        value: T,
    ) -> Result<(), Error>
    where
        U: Resource<Element = T, Device = Self, Kind = kind::Uniform> + 'static,
    {
        self.encode_uniform(frame, &**target, &value)
    }

    fn frame_stage(&self, frame: &mut GpuFrame, stage: &GpuStage) -> Result<(), Error> {
        self.encode_stage(frame, stage)
    }

    fn reserve_read<T: GpuPod>(&self, source: &Arc<GpuBuffer<T>>) -> Result<GpuRead<T>, Error> {
        self.reserve(source)
    }

    fn frame_read<T: GpuPod>(&self, frame: &mut GpuFrame, read: &mut GpuRead<T>) {
        read.encode_copy(frame);
    }

    fn submit_frame(&self, frame: GpuFrame) -> Result<GpuSubmission, Error> {
        Ok(self.submit_one(frame))
    }

    fn begin_read<T: GpuPod + Send>(
        &self,
        read: GpuRead<T>,
        _submission: &GpuSubmission,
    ) -> BoxFuture<Result<Vec<T>, map::Error>> {
        self.start_ticket(read)
    }

    fn completion(&self, submission: &GpuSubmission) -> BoxFuture<Result<(), map::Error>> {
        self.completion_of(submission)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self, submission: &GpuSubmission) -> Result<(), Error> {
        self.wait_for_submission(submission.index.clone())
    }
}

/// The map error of a failure of this device, for the futures of a frame.
fn map_error(error: Error) -> map::Error {
    <HeadlessDevice as MapHost>::map_error(error)
}
