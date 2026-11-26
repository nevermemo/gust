//! The GPU device's batch (plan m14.s2, slice s2a2): [`HeadlessDevice`] as a `gust::batch::BatchDevice`.
//!
//! A frame is recorded into one command encoder: its own at `submit`, or the caller's at `encode`. The writes of the
//! frame are encoded copies, the extent records of the stages with a parameter set in the frame (or left stale by an
//! earlier frame recorded into a caller's encoder, see `gust::batch::Encoded`) are encoded copies, then come
//! the passes of the stages in order, then the read-back copies of the tickets; `submit` finishes the encoder and makes
//! **one** counted queue submission ([`HeadlessDevice::gust_submissions`] rises by 1 whatever the number of stages and
//! tickets). One encoder per frame and ticket-before-submit are decision D251.
//!
//! Frames in flight (`in_flight(n)`) share the batch's one set of buffers and are kept apart by queue order
//! (decision D258): only a ticket owns per-frame storage, its staging buffer. So an input or parameter that a frame
//! does not write holds the value of the last frame that reached the queue, for every `n`. A frame recorded into a
//! caller's encoder follows `gust::batch::Encoded`: a reported frame's `set`s hold, a dropped frame's are discarded.
//!
//! A caller reads a batch's buffers on the GPU through `gust::batch::View` (`Batch::view`, decision D261): the batch's
//! own [`GpuBuffer`], bound with [`GpuBuffer::as_binding`] (the live elements, not the size class), as read-only
//! storage. A reader sees what the last frame recorded before its place in queue order left there: a pass recorded
//! into the same encoder after `FrameRun::encode` returned reads that frame, one recorded before `encode` reads the
//! frame before it, and one in a later submission reads the last frame submitted before it. A view carries no check
//! that a frame ran, so an encoder of a failed, dropped or misreported frame is dropped with its passes.
//!
//! The pieces are by concern: [`build`] seals a stage, [`frame`] records a frame and submits it, [`ticket`] reserves
//! and starts the read-backs. `shape` and `ops` hold what later slices add.

use std::sync::Arc;

use gust::batch::{BatchDevice, BoxFuture};
#[cfg(not(target_arch = "wasm32"))]
use gust::batch::{DeviceTiming, FrameClock, FrameTiming, OpenedTiming};
use gust::map::{self, MapHost};
use gust::{Call, GpuPod, Kernel, Resource, Run, kind};

use crate::{Error, GpuBuffer, HeadlessDevice};

mod build;
mod counted;
mod frame;
mod ops;
mod shape;
mod slot;
mod ticket;
#[cfg(not(target_arch = "wasm32"))]
mod timing;

pub use build::GpuStage;
pub use frame::{GpuEncoded, GpuFrame, GpuSubmission};
pub use ticket::GpuRead;

impl BatchDevice for HeadlessDevice {
    type Prepared = GpuStage;
    type Frame = GpuFrame;
    type Encoder = wgpu::CommandEncoder;
    type Token = wgpu::SubmissionIndex;
    type Encoded = GpuEncoded;
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
        Ok(Self::new_frame())
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

    fn new_encoder(&self) -> wgpu::CommandEncoder {
        self.new_batch_encoder()
    }

    fn frame_uploads(
        &self,
        frame: &mut GpuFrame,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), Error> {
        Self::record_uploads(frame, encoder);
        Ok(())
    }

    fn frame_stage(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stage: &GpuStage,
    ) -> Result<(), Error> {
        self.encode_stage(encoder, stage)
    }

    fn reserve_read<T: GpuPod>(&self, source: &Arc<GpuBuffer<T>>) -> Result<GpuRead<T>, Error> {
        self.reserve(source)
    }

    fn frame_read<T: GpuPod>(&self, encoder: &mut wgpu::CommandEncoder, read: &mut GpuRead<T>) {
        read.encode_copy(encoder);
    }

    fn submit_frame(
        &self,
        frame: GpuFrame,
        encoder: wgpu::CommandEncoder,
    ) -> Result<GpuSubmission, Error> {
        Ok(self.submit_one(frame, encoder))
    }

    fn encoded_frame(
        &self,
        frame: GpuFrame,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<GpuEncoded, Error> {
        Ok(self.encoded_one(frame, encoder))
    }

    fn encoded_submitted(
        &self,
        encoded: GpuEncoded,
        token: &wgpu::SubmissionIndex,
    ) -> GpuSubmission {
        self.encoded_submission(encoded, token)
    }

    fn begin_read<T: GpuPod + Send>(
        &self,
        read: GpuRead<T>,
        submission: &GpuSubmission,
    ) -> BoxFuture<Result<Option<Vec<T>>, map::Error>> {
        self.start_ticket(read, submission)
    }

    fn completion(&self, submission: &GpuSubmission) -> BoxFuture<Result<(), map::Error>> {
        self.completion_of(submission)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self, submission: &GpuSubmission) -> Result<(), Error> {
        self.wait_for_submission(submission.index.clone())
    }

    // Frame timing reads timestamps through blocking-free mappings, which this slice builds for native targets only:
    // on `wasm32` the defaults answer "no timing" (the browser's availability is a probe of its own).
    #[cfg(not(target_arch = "wasm32"))]
    fn timing_open(
        &self,
        level: FrameTiming,
        stages: usize,
        in_flight: usize,
    ) -> Option<OpenedTiming> {
        self.open_timing(level, stages, in_flight)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn stage_passes(&self, stage: &GpuStage) -> usize {
        stage.dispatches.len()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn frame_clock(&self, timing: &DeviceTiming) -> Option<FrameClock> {
        Self::take_clock(timing)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn frame_stage_timed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stage: &GpuStage,
        clock: &mut FrameClock,
        begin: Option<u32>,
        end: Option<u32>,
    ) -> Result<(), Error> {
        self.encode_stage_timed(encoder, stage, clock, begin, end)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn frame_clock_end(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        clock: &mut FrameClock,
        slots: u32,
    ) {
        Self::end_clock(encoder, clock, slots);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn begin_time(
        &self,
        clock: FrameClock,
        submission: &GpuSubmission,
    ) -> BoxFuture<Result<Option<Vec<u64>>, map::Error>> {
        self.start_time(clock, submission)
    }
}

/// The map error of a failure of this device, for the futures of a frame.
fn map_error(error: Error) -> map::Error {
    <HeadlessDevice as MapHost>::map_error(error)
}
