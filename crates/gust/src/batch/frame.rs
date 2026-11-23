//! Frames: one trip of a batch to the device ([`FrameRun`]), the submitted frame ([`Submitted`]) and the read-back
//! reserved in a frame ([`Ticket`]).

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use super::{BatchDevice, BoxFuture, Core, Error, Param, ParamType, Stage, Value, Values, map};
use crate::GpuPod;

/// One read-back of a frame, reserved by [`FrameRun::ticket`]: a future of the values (row-major for a 2-D handle).
///
/// It owns what it needs and borrows nothing: it is `'static`, and `Send` on native targets (not on `wasm32`, whose
/// handles are not). A frame that is never submitted resolves it to [`Error::NotSubmitted`]. Polling it on a native
/// GPU device polls the device, so a caller without a runtime polls it in a loop (`gust::task::Task::dispatch` does
/// the same for a task).
#[must_use = "a ticket is a future of the values: await it"]
pub struct Ticket<T> {
    cell: Arc<TicketCell<T>>,
}

impl<T> core::fmt::Debug for Ticket<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Ticket").finish_non_exhaustive()
    }
}

enum TicketState<T> {
    /// The frame has not been submitted; holds the waker of the last poll.
    Reserved(Option<Waker>),
    /// The frame was submitted: the read-back.
    Running(BoxFuture<Result<Vec<T>, map::Error>>),
    /// The frame was never submitted.
    Failed,
    /// The result was handed out.
    Taken,
}

struct TicketCell<T> {
    batch: String,
    state: Mutex<TicketState<T>>,
}

impl<T> TicketCell<T> {
    fn lock(&self) -> MutexGuard<'_, TicketState<T>> {
        // No code runs under the lock that can leave the state half-written.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn start(&self, read: BoxFuture<Result<Vec<T>, map::Error>>) {
        let waker = {
            let mut state = self.lock();
            let TicketState::Reserved(waker) = &mut *state else {
                return;
            };
            let waker = waker.take();
            *state = TicketState::Running(read);
            waker
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn fail(&self) {
        let waker = {
            let mut state = self.lock();
            let TicketState::Reserved(waker) = &mut *state else {
                return;
            };
            let waker = waker.take();
            *state = TicketState::Failed;
            waker
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

impl<T> Future for Ticket<T> {
    type Output = Result<Vec<T>, Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.cell.lock();
        match &mut *state {
            TicketState::Reserved(waker) => {
                if !waker
                    .as_ref()
                    .is_some_and(|stored| stored.will_wake(cx.waker()))
                {
                    *waker = Some(cx.waker().clone());
                }
                Poll::Pending
            }
            TicketState::Running(read) => match read.as_mut().poll(cx) {
                Poll::Ready(result) => {
                    *state = TicketState::Taken;
                    Poll::Ready(result.map_err(Error::Device))
                }
                Poll::Pending => Poll::Pending,
            },
            TicketState::Failed => {
                *state = TicketState::Taken;
                Poll::Ready(Err(Error::NotSubmitted {
                    batch: self.cell.batch.clone(),
                }))
            }
            TicketState::Taken => panic!("a Ticket was polled after it returned its values"),
        }
    }
}

// A ticket outlives its batch and moves between threads on native targets (D244): checked at compile time.
#[cfg(not(target_arch = "wasm32"))]
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<Ticket<f32>>();
};

/// A reserved read-back, with its element type erased so that a frame holds a list of them.
trait ReadSlot<D: BatchDevice> {
    /// Record the copy into the frame, after the stages.
    fn record(&mut self, device: &D, frame: &mut D::Frame);
    /// The frame was submitted: start the read-back.
    fn start(self: Box<Self>, device: &D, submission: &D::Submission);
    /// The frame was not submitted.
    fn abandon(self: Box<Self>);
}

struct Reserved<D: BatchDevice, T: GpuPod> {
    read: D::Read<T>,
    cell: Arc<TicketCell<T>>,
}

impl<D: BatchDevice, T: GpuPod + Send> ReadSlot<D> for Reserved<D, T> {
    fn record(&mut self, device: &D, frame: &mut D::Frame) {
        device.frame_read(frame, &mut self.read);
    }

    fn start(self: Box<Self>, device: &D, submission: &D::Submission) {
        let Self { read, cell } = *self;
        cell.start(device.begin_read(read, submission));
    }

    fn abandon(self: Box<Self>) {
        self.cell.fail();
    }
}

/// One frame of a batch, between [`Batch::begin`](super::Batch::begin) and `submit`: `write` and `set` change the
/// data, `ticket` reserves a read-back, and `submit` makes one queue submission of all of it. It borrows nothing.
///
/// Dropping it before `submit` changes nothing on the device: its tickets resolve to [`Error::NotSubmitted`] and the
/// next `begin` runs on the batch as it was.
#[must_use = "a frame does nothing until it is submitted"]
pub struct FrameRun<D: BatchDevice> {
    core: Arc<Core<D>>,
    frame: Option<D::Frame>,
    pending: Vec<(usize, Value)>,
    reads: Vec<Box<dyn ReadSlot<D>>>,
    /// The frame was handed to `submit`'s last step: nothing is left for `Drop` to undo.
    done: bool,
}

impl<D: BatchDevice> core::fmt::Debug for FrameRun<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FrameRun")
            .field("batch", &self.core.name)
            .field("tickets", &self.reads.len())
            .finish_non_exhaustive()
    }
}

fn device_error<D: BatchDevice>(error: D::Error) -> Error {
    Error::Device(D::map_error(error))
}

impl<D: BatchDevice> super::Batch<D> {
    /// Start a frame. With one frame in flight this waits until the frame before it has run (it never waits for a
    /// frame that was dropped).
    ///
    /// # Errors
    ///
    /// [`Error::FrameLive`] when the frame before is neither submitted nor dropped; the device's failures.
    pub async fn begin(&mut self) -> Result<FrameRun<D>, Error> {
        let core = &self.core;
        let last = {
            let mut run = core.run.lock().unwrap_or_else(PoisonError::into_inner);
            if run.live {
                return Err(Error::FrameLive {
                    batch: core.name.clone(),
                });
            }
            run.live = true;
            run.last.take()
        };
        let mut claim = Claim {
            core,
            armed: true,
            last,
        };
        if let Some(last) = claim.last.as_ref() {
            core.device.completion(last).await.map_err(Error::Device)?;
        }
        claim.last = None;
        let frame = core.device.begin_frame().map_err(device_error::<D>)?;
        claim.armed = false;
        Ok(FrameRun {
            core: Arc::clone(core),
            frame: Some(frame),
            pending: Vec::new(),
            reads: Vec::new(),
            done: false,
        })
    }
}

/// Gives the claim on the frame back when `begin` ends without making a frame (an error, or its future dropped), and
/// the submission it was about to wait for, so that the next `begin` still waits for it.
struct Claim<'a, D: BatchDevice> {
    core: &'a Core<D>,
    armed: bool,
    last: Option<D::Submission>,
}

impl<D: BatchDevice> Drop for Claim<'_, D> {
    fn drop(&mut self) {
        if self.armed {
            if let Some(last) = self.last.take() {
                self.core
                    .run
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .last = Some(last);
            }
            release(self.core);
        }
    }
}

fn release<D: BatchDevice>(core: &Core<D>) {
    core.run.lock().unwrap_or_else(PoisonError::into_inner).live = false;
}

impl<D: BatchDevice> FrameRun<D> {
    /// Copy `data` into an input. The copy is recorded into this frame's command encoder, so it runs before the
    /// frame's stages and never before the frame before it.
    ///
    /// # Errors
    ///
    /// [`Error::Foreign`] for a handle of another batch, [`Error::NotOutput`] for one the block did not return,
    /// [`Error::NotInput`] for a stage, [`Error::ShapeMismatch`] for a slice of another length, and the device's
    /// failures.
    pub fn write<T: GpuPod, R>(&mut self, handle: &Stage<T, R>, data: &[T]) -> Result<(), Error>
    where
        D::Buffer<T>: 'static,
    {
        let (id, buffer) = self.core.output(handle)?;
        let entry = &self.core.buffers[id];
        if !entry.input {
            return Err(Error::NotInput {
                batch: self.core.name.clone(),
                stage: entry.name.clone(),
            });
        }
        if data.len() != entry.len {
            return Err(Error::ShapeMismatch {
                batch: self.core.name.clone(),
                stage: entry.name.clone(),
                expected: format!("{} elements", entry.len),
                found: format!("{} elements", data.len()),
            });
        }
        let frame = self
            .frame
            .as_mut()
            .expect("a live frame holds its recorder");
        self.core
            .device
            .frame_write(frame, &buffer, data)
            .map_err(device_error::<D>)
    }

    /// Give a parameter this value from this frame on. A parameter that sizes a stage cannot change after the seal:
    /// setting it to another value is an error (a later slice lifts that).
    ///
    /// # Errors
    ///
    /// [`Error::Foreign`] for a parameter of another batch; [`Error::ShapeMismatch`] for a new shape.
    pub fn set<T: ParamType>(&mut self, param: &Param<T>, value: T) -> Result<(), Error> {
        let core = &self.core;
        if param.batch() != core.id || param.id() >= core.params.len() {
            return Err(Error::Foreign {
                batch: core.name.clone(),
                stage: String::from("a parameter"),
            });
        }
        let entry = &core.params[param.id()];
        let packed = value.pack();
        if entry.shape && entry.value != packed {
            return Err(Error::ShapeMismatch {
                batch: core.name.clone(),
                stage: entry.name.clone(),
                expected: format!("{:?}", entry.value),
                found: format!("{packed:?}"),
            });
        }
        self.pending.retain(|(id, _)| *id != param.id());
        self.pending.push((param.id(), packed));
        Ok(())
    }

    /// Reserve the read-back of a handle: its values after this frame's stages, copied inside the frame's one
    /// submission. The ticket resolves when the frame has run.
    ///
    /// # Errors
    ///
    /// [`Error::Foreign`], [`Error::NotOutput`] and the device's failures.
    pub fn ticket<T: GpuPod + Send, R>(&mut self, handle: &Stage<T, R>) -> Result<Ticket<T>, Error>
    where
        D::Buffer<T>: 'static,
    {
        let (_, buffer) = self.core.output(handle)?;
        let read = self
            .core
            .device
            .reserve_read::<T>(&buffer)
            .map_err(device_error::<D>)?;
        let cell = Arc::new(TicketCell {
            batch: self.core.name.clone(),
            state: Mutex::new(TicketState::Reserved(None)),
        });
        self.reads.push(Box::new(Reserved::<D, T> {
            read,
            cell: Arc::clone(&cell),
        }));
        Ok(Ticket { cell })
    }

    /// Run the frame: one queue submission on the GPU whatever the number of tickets. Returns at once.
    ///
    /// # Errors
    ///
    /// The device's failures. The frame's tickets then resolve to [`Error::NotSubmitted`].
    pub fn submit(mut self) -> Result<Submitted<D>, Error> {
        let frame = self.encode()?;
        let submission = self
            .core
            .device
            .submit_frame(frame)
            .map_err(device_error::<D>)?;
        Ok(self.mark_submitted(submission))
    }

    /// First step of `submit`: the extent records that a `set` changed, the stages in order, then the copies of the
    /// tickets.
    fn encode(&mut self) -> Result<D::Frame, Error> {
        let core = Arc::clone(&self.core);
        let device = &*core.device;
        let mut frame = self.frame.take().expect("a live frame holds its recorder");
        let mut values = core
            .run
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values
            .clone();
        for (id, value) in &self.pending {
            values[*id] = *value;
        }
        let values = Values(values);
        for stage in &core.stages {
            let changed = stage
                .params
                .iter()
                .any(|id| self.pending.iter().any(|(set, _)| set == id));
            if let (true, Some(update)) = (changed, &stage.update) {
                update(device, &mut frame, &values).map_err(device_error::<D>)?;
            }
        }
        for stage in &core.stages {
            device
                .frame_stage(&mut frame, &stage.prepared)
                .map_err(device_error::<D>)?;
        }
        for slot in &mut self.reads {
            slot.record(device, &mut frame);
        }
        Ok(frame)
    }

    /// Last step of `submit`: the frame is on the device; start its read-backs, keep its parameters and let the next
    /// `begin` wait for it.
    fn mark_submitted(mut self, submission: D::Submission) -> Submitted<D> {
        let core = Arc::clone(&self.core);
        for slot in self.reads.drain(..) {
            slot.start(&core.device, &submission);
        }
        {
            let mut run = core.run.lock().unwrap_or_else(PoisonError::into_inner);
            for (id, value) in self.pending.drain(..) {
                run.values[id] = value;
            }
            run.live = false;
            run.last = Some(submission.clone());
        }
        self.done = true;
        Submitted { core, submission }
    }
}

impl<D: BatchDevice> Drop for FrameRun<D> {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        for slot in self.reads.drain(..) {
            slot.abandon();
        }
        release(&self.core);
    }
}

/// A frame that was submitted. Its tickets are the way to its values; this waits for the frame itself.
#[must_use = "a submitted frame is waited for through its tickets, or with `wait`"]
pub struct Submitted<D: BatchDevice> {
    core: Arc<Core<D>>,
    submission: D::Submission,
}

impl<D: BatchDevice> core::fmt::Debug for Submitted<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Submitted")
            .field("batch", &self.core.name)
            .finish_non_exhaustive()
    }
}

impl<D: BatchDevice> Submitted<D> {
    /// Wait until the frame has run. Not available on `wasm32`, where nothing may block (use
    /// [`finished`](Self::finished)).
    ///
    /// # Errors
    ///
    /// The device's failure to wait.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(&self) -> Result<(), Error> {
        self.core
            .device
            .wait(&self.submission)
            .map_err(device_error::<D>)
    }

    /// A future that completes when the frame has run.
    ///
    /// # Errors
    ///
    /// The device's failure to wait.
    pub async fn finished(&self) -> Result<(), Error> {
        self.core
            .device
            .completion(&self.submission)
            .await
            .map_err(Error::Device)
    }
}

/// The blocking form of a batch, native targets only: [`Batch::dispatch`](super::Batch::dispatch) ran a frame of the
/// data as it stands and waited for it; `read` blocks for a result.
#[cfg(not(target_arch = "wasm32"))]
pub struct Blocking<'b, D: BatchDevice> {
    batch: &'b mut super::Batch<D>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<D: BatchDevice> super::Batch<D> {
    /// Run one frame of the data as it stands (no `write`, no `set`), wait for it, and return the handle to read the
    /// results with. Not available on `wasm32`.
    ///
    /// The frame is one submission on the GPU device, but every [`Blocking::read`] after it is a submission of its own
    /// (a blocking read-back), so `dispatch()` and two reads are three submissions. Use tickets to count one.
    ///
    /// # Errors
    ///
    /// Those of [`begin`](Self::begin) and [`FrameRun::submit`], and the device's failure to wait.
    pub fn dispatch(&mut self) -> Result<Blocking<'_, D>, Error> {
        let frame = block_on(self.begin())?;
        frame.submit()?.wait()?;
        Ok(Blocking { batch: self })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<D: BatchDevice> Blocking<'_, D> {
    /// The values of a handle, after the frame that [`Batch::dispatch`](super::Batch::dispatch) ran (row-major for
    /// 2-D). On the GPU device each call is a submission of its own, apart from the frame's.
    ///
    /// # Errors
    ///
    /// [`Error::Foreign`], [`Error::NotOutput`] and the device's failures.
    pub fn read<T: GpuPod, R>(&self, handle: &Stage<T, R>) -> Result<Vec<T>, Error>
    where
        D::Buffer<T>: 'static,
    {
        let (_, buffer) = self.batch.core.output(handle)?;
        self.batch
            .core
            .device
            .readback(&buffer)
            .map_err(device_error::<D>)
    }
}

/// Drive `future` on this thread until it completes.
#[cfg(not(target_arch = "wasm32"))]
fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = core::pin::pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}
