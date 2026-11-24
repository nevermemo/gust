//! The CPU device's batch: the stages run in order at `submit`, and the tickets are ready then (plan m14.s2, slice
//! s2a2). There is no queue, so nothing overlaps: a frame is a list of steps.

use core::any::Any;
use std::rc::Rc;
use std::sync::{Arc, Mutex, PoisonError};

use super::{CpuBuffer, CpuDevice, CpuJob, CpuUniform, Error};
use crate::batch::{BatchDevice, BoxFuture};
use crate::map;
use crate::{Call, GpuPod, Kernel, Resource, Run, kind};

/// The call of a CPU stage.
type StageCall = dyn Fn(&CpuDevice) -> Result<(), Error>;

/// One step of a CPU frame.
type Step = Box<dyn FnOnce(&CpuDevice) -> Result<(), Error>>;

/// A CPU stage: the call, made again and run at each frame.
#[doc(hidden)]
pub struct CpuStage(Rc<StageCall>);

/// A CPU frame: the writes and parameter records, in the order of the calls.
#[doc(hidden)]
#[derive(Default)]
pub struct CpuFrame {
    steps: Vec<Step>,
}

/// What a CPU frame is recorded into: the steps, run in order by [`run`](Self::run). The caller of
/// [`FrameRun::encode`](crate::batch::FrameRun::encode) runs it, as a GPU caller submits its encoder, and reports the
/// job it returns; an encoder that is dropped, or never run, makes the frame's tickets resolve to
/// [`Error::NotSubmitted`](crate::batch::Error::NotSubmitted). Made by `BatchDevice::new_encoder`.
#[derive(Default)]
pub struct CpuEncoder {
    steps: Vec<Step>,
}

impl core::fmt::Debug for CpuEncoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CpuEncoder")
            .field("steps", &self.steps.len())
            .finish()
    }
}

impl CpuEncoder {
    /// Run the recorded steps in order on `device` (the CPU's queue submission) and return the job to report with
    /// [`Encoded::submitted`](crate::batch::Encoded::submitted).
    ///
    /// # Errors
    ///
    /// The first step that fails; the steps after it do not run.
    pub fn run(self, device: &CpuDevice) -> Result<CpuJob, Error> {
        for step in self.steps {
            step(device)?;
        }
        Ok(CpuJob::complete())
    }
}

/// A reserved read-back on the CPU: the buffer, and where the copy made at `submit` is left.
#[doc(hidden)]
pub struct CpuRead<T: GpuPod> {
    source: Arc<CpuBuffer<T>>,
    values: Arc<Mutex<Option<Vec<T>>>>,
}

impl BatchDevice for CpuDevice {
    type Prepared = CpuStage;
    type Frame = CpuFrame;
    type Encoder = CpuEncoder;
    type Token = CpuJob;
    type Encoded = ();
    type Read<T: GpuPod> = CpuRead<T>;
    type Submission = CpuJob;

    fn prepare<K, R, const N: usize>(
        &self,
        keep: R,
        build: impl for<'o> Fn(&'o R) -> Call<'o, Self, K, N> + 'static,
    ) -> Result<CpuStage, Error>
    where
        Self: Run<K, N>,
        K: Kernel,
        R: 'static,
    {
        Ok(CpuStage(Rc::new(move |device: &CpuDevice| {
            let call = build(&keep);
            <CpuDevice as Run<K, N>>::dispatch_call(device, &call)
        })))
    }

    fn begin_frame(&self) -> Result<CpuFrame, Error> {
        Ok(CpuFrame::default())
    }

    fn frame_write<T: GpuPod>(
        &self,
        frame: &mut CpuFrame,
        target: &Arc<CpuBuffer<T>>,
        data: &[T],
    ) -> Result<(), Error> {
        let target = Arc::clone(target);
        let data = data.to_vec();
        frame.steps.push(Box::new(move |_| target.write(&data)));
        Ok(())
    }

    fn frame_uniform<T: GpuPod, U>(
        &self,
        frame: &mut CpuFrame,
        target: &Arc<U>,
        value: T,
    ) -> Result<(), Error>
    where
        U: Resource<Element = T, Device = Self, Kind = kind::Uniform> + 'static,
    {
        let target = Arc::clone(target);
        frame.steps.push(Box::new(move |_| {
            (&*target as &dyn Any)
                .downcast_ref::<CpuUniform<T>>()
                .expect("the uniform resource of the CPU device is a `CpuUniform`")
                .write(value)
        }));
        Ok(())
    }

    fn new_encoder(&self) -> CpuEncoder {
        CpuEncoder::default()
    }

    fn frame_uploads(&self, frame: &mut CpuFrame, encoder: &mut CpuEncoder) -> Result<(), Error> {
        encoder.steps.append(&mut frame.steps);
        Ok(())
    }

    fn frame_stage(&self, encoder: &mut CpuEncoder, stage: &CpuStage) -> Result<(), Error> {
        let run = Rc::clone(&stage.0);
        encoder.steps.push(Box::new(move |device| run(device)));
        Ok(())
    }

    fn reserve_read<T: GpuPod>(&self, source: &Arc<CpuBuffer<T>>) -> Result<CpuRead<T>, Error> {
        Ok(CpuRead {
            source: Arc::clone(source),
            values: Arc::new(Mutex::new(None)),
        })
    }

    fn frame_read<T: GpuPod>(&self, encoder: &mut CpuEncoder, read: &mut CpuRead<T>) {
        let source = Arc::clone(&read.source);
        let values = Arc::clone(&read.values);
        encoder.steps.push(Box::new(move |_| {
            *values.lock().unwrap_or_else(PoisonError::into_inner) = Some(source.read());
            Ok(())
        }));
    }

    fn submit_frame(&self, _frame: CpuFrame, encoder: CpuEncoder) -> Result<CpuJob, Error> {
        encoder.run(self)
    }

    fn encoded_frame(&self, _frame: CpuFrame, _encoder: &mut CpuEncoder) -> Result<(), Error> {
        Ok(())
    }

    fn encoded_submitted(&self, (): (), token: &CpuJob) -> CpuJob {
        *token
    }

    fn begin_read<T: GpuPod + Send>(
        &self,
        read: CpuRead<T>,
        _submission: &CpuJob,
    ) -> BoxFuture<Result<Option<Vec<T>>, map::Error>> {
        let CpuRead { values, .. } = read;
        // `None` when the copy step never ran: the caller's encoder was dropped, or never run.
        Box::pin(async move { Ok(values.lock().unwrap_or_else(PoisonError::into_inner).take()) })
    }

    fn completion(&self, _submission: &CpuJob) -> BoxFuture<Result<(), map::Error>> {
        Box::pin(async { Ok(()) })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self, _submission: &CpuJob) -> Result<(), Error> {
        Ok(())
    }
}
