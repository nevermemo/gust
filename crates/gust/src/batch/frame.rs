//! Frames: one trip of a batch to the device ([`FrameRun`]), the submitted frame ([`Submitted`]) and the read-back
//! reserved in a frame ([`Ticket`]).

use core::any::Any;
use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::Ordering;
use core::task::{Context, Poll, Waker};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use super::FrameTime;
use super::timing::{Recorded, TimeCell, TimePlan, TimedFrame};
use super::{
    BatchDevice, BoxFuture, Core, Element, Empty, Error, Layout, OpPlan, Operands, PLANS_KEPT,
    Param, ParamType, Stage, Value, Values, map, typed,
};
use crate::GpuPod;

/// One read-back of a frame, reserved by [`FrameRun::ticket`]: a future of the values (row-major for a 2-D handle). A
/// handle of an enum resolves to [`Error::InvalidEnum`] when a word it read is no variant (D275).
///
/// It owns what it needs and borrows nothing: it is `'static`, and `Send` on native targets (not on `wasm32`, whose
/// handles are not). A frame that is never submitted resolves it to [`Error::NotSubmitted`]. Polling it on a native
/// GPU device polls the device, so a caller without a runtime polls it in a loop (`gust::task::Task::dispatch` does
/// the same for a task).
#[must_use = "a ticket is a future of the values: await it"]
pub struct Ticket<T: Element> {
    cell: Arc<TicketCell<T>>,
}

impl<T: Element> core::fmt::Debug for Ticket<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Ticket").finish_non_exhaustive()
    }
}

enum TicketState<T: Element> {
    /// The frame has not been submitted; holds the waker of the last poll.
    Reserved(Option<Waker>),
    /// The frame was submitted: the read-back, in the words the device holds.
    Running(BoxFuture<Result<Option<Vec<T::Word>>, map::Error>>),
    /// The frame was never submitted.
    Failed,
    /// The result was handed out.
    Taken,
}

struct TicketCell<T: Element> {
    batch: String,
    /// The stage or input the words are read from, for [`Error::InvalidEnum`].
    stage: String,
    state: Mutex<TicketState<T>>,
}

impl<T: Element> TicketCell<T> {
    fn lock(&self) -> MutexGuard<'_, TicketState<T>> {
        // No code runs under the lock that can leave the state half-written.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn start(&self, read: BoxFuture<Result<Option<Vec<T::Word>>, map::Error>>) {
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

impl<T: Element> Future for Ticket<T> {
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
                    Poll::Ready(match result {
                        // The checked unpack: an enum element is a value only if its word is a variant.
                        Ok(Some(words)) => T::unpack(words).map_err(|bad| Error::InvalidEnum {
                            batch: self.cell.batch.clone(),
                            stage: self.cell.stage.clone(),
                            index: bad.index(),
                            word: bad.word(),
                        }),
                        // The copy never ran: the encoder it was recorded into was not submitted.
                        Ok(None) => Err(Error::NotSubmitted {
                            batch: self.cell.batch.clone(),
                        }),
                        Err(error) => Err(Error::Device(error)),
                    })
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
    /// Record the copy into the encoder, after the stages.
    fn record(&mut self, device: &D, encoder: &mut D::Encoder);
    /// The frame was submitted: start the read-back.
    fn start(self: Box<Self>, device: &D, submission: &D::Submission);
    /// The frame was not submitted.
    fn abandon(self: Box<Self>);
}

struct Reserved<D: BatchDevice, T: Element> {
    read: D::Read<T::Word>,
    cell: Arc<TicketCell<T>>,
}

impl<D: BatchDevice, T: Element + Send> ReadSlot<D> for Reserved<D, T>
where
    T::Word: Send,
{
    fn record(&mut self, device: &D, encoder: &mut D::Encoder) {
        device.frame_read(encoder, &mut self.read);
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
    /// The parameters this frame set, derived values included, with their new values.
    pending: Vec<(usize, Value)>,
    /// The buffers this frame was resized to, until the frame is submitted or encoded (see [`set`](Self::set)).
    staged: Option<Staged<D>>,
    /// The buffers this frame has written or reserved a read-back of: a later `set` may not change their extents.
    used: Vec<usize>,
    reads: Vec<Box<dyn ReadSlot<D>>>,
    /// The frame's time, once [`time`](Self::time) asked for it.
    timed: Option<TimedFrame>,
    /// The frame was recorded into a caller's encoder (`encode`): the host cannot tell whether that encoder ran.
    encoded: bool,
    /// The frame was handed to `submit`'s last step: nothing is left for `Drop` to undo.
    done: bool,
}

/// What a frame that resizes the batch holds until it is submitted or encoded: the buffers (new ones where the extents
/// changed), the extents, the values of the shape parameters and the stages prepared over the new buffers. A dropped
/// frame drops all of it and the batch is as it was.
struct Staged<D: BatchDevice> {
    values: Vec<Value>,
    dims: Vec<[u32; 2]>,
    buffers: Vec<Arc<dyn Any>>,
    /// How many of `buffers` are not the batch's own.
    replaced: usize,
    /// The stages whose buffers are not the batch's own, prepared over these ones.
    prepared: Vec<StagedStage<D>>,
}

/// A stage prepared again by a resize.
struct StagedStage<D: BatchDevice> {
    /// The stage's number.
    stage: usize,
    /// What it dispatches now: `None` for an op over zero elements.
    prepared: Option<Arc<D::Prepared>>,
    /// For an op over some elements: its length and the plan it was prepared with, which the batch keeps when the frame
    /// is committed.
    plan: Option<(usize, Arc<dyn Any>)>,
}

impl<D: BatchDevice> Clone for StagedStage<D> {
    fn clone(&self) -> Self {
        Self {
            stage: self.stage,
            prepared: self.prepared.clone(),
            plan: self.plan.clone(),
        }
    }
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

/// Make `plan`, the plan of a length an op stage now runs at, the first of the stage's kept plans, and drop the oldest
/// when more than `PLANS_KEPT` remain.
fn keep_plan(kept: &mut VecDeque<(usize, Arc<dyn Any>)>, plan: (usize, Arc<dyn Any>)) {
    kept.retain(|(len, _)| *len != plan.0);
    kept.push_front(plan);
    kept.truncate(PLANS_KEPT);
}

impl<D: BatchDevice> super::Batch<D> {
    /// Start a frame. It waits only when the batch already has `in_flight(n)` frames in flight: then it waits until
    /// the oldest of them has run (it never waits for a frame that was dropped).
    ///
    /// # Errors
    ///
    /// [`Error::FrameLive`] when the frame before is neither submitted nor dropped; the device's failures.
    pub async fn begin(&mut self) -> Result<FrameRun<D>, Error> {
        let core = &self.core;
        let oldest = {
            let mut run = core.run.lock().unwrap_or_else(PoisonError::into_inner);
            if run.live {
                return Err(Error::FrameLive {
                    batch: core.name.clone(),
                });
            }
            run.live = true;
            if run.pending.len() >= run.depth {
                run.pending.pop_front()
            } else {
                None
            }
        };
        let mut claim = Claim {
            core,
            armed: true,
            oldest,
        };
        if let Some(oldest) = claim.oldest.as_ref() {
            core.device
                .completion(oldest)
                .await
                .map_err(Error::Device)?;
        }
        claim.oldest = None;
        let frame = core.device.begin_frame().map_err(device_error::<D>)?;
        claim.armed = false;
        Ok(FrameRun {
            core: Arc::clone(core),
            frame: Some(frame),
            pending: Vec::new(),
            staged: None,
            used: Vec::new(),
            reads: Vec::new(),
            timed: None,
            encoded: false,
            done: false,
        })
    }
}

/// Gives the claim on the frame back when `begin` ends without making a frame (an error, or its future dropped), and
/// the submission it was about to wait for, so that the next `begin` still waits for it.
struct Claim<'a, D: BatchDevice> {
    core: &'a Core<D>,
    armed: bool,
    oldest: Option<D::Submission>,
}

impl<D: BatchDevice> Drop for Claim<'_, D> {
    fn drop(&mut self) {
        if self.armed {
            if let Some(oldest) = self.oldest.take() {
                self.core
                    .run
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .pending
                    .push_front(oldest);
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
    pub fn write<T: Element, R>(&mut self, handle: &Stage<T, R>, data: &[T]) -> Result<(), Error>
    where
        D::Buffer<T::Word>: 'static,
    {
        let id = self.core.locate(handle)?;
        if handle.is_raw() {
            return Err(Error::NotInput {
                batch: self.core.name.clone(),
                stage: format!(
                    "{} (the words of an enum, write the enum)",
                    self.core.buffers[id].name
                ),
            });
        }
        let entry = &self.core.buffers[id];
        if !entry.input {
            return Err(Error::NotInput {
                batch: self.core.name.clone(),
                stage: entry.name.clone(),
            });
        }
        let [width, height] = self.dims_of(id);
        let len = u64::from(width) * u64::from(height);
        if data.len() as u64 != len {
            return Err(Error::ShapeMismatch {
                batch: self.core.name.clone(),
                stage: entry.name.clone(),
                expected: format!("{len} elements"),
                found: format!("{} elements", data.len()),
            });
        }
        let buffer = self.buffer_of::<T::Word>(id);
        let frame = self
            .frame
            .as_mut()
            .expect("a live frame holds its recorder");
        self.core
            .device
            .frame_write(frame, &buffer, &T::pack(data))
            .map_err(device_error::<D>)?;
        self.used.push(id);
        Ok(())
    }

    /// Give a parameter this value from this frame on.
    ///
    /// A parameter that sizes an input or a stage (directly, or through a [`Derived`](super::Derived) value) can change
    /// between frames: what depends on it is **resized** here. The buffers whose extents change are made again (zeroed:
    /// an input holds zeros until the frame writes it, and nothing is carried over), the stages that bind them are
    /// prepared again over the new buffers (their workgroup counts follow the extents), and the extent records are
    /// written again at the frame's submission; nothing is counted on the GPU. The batch itself changes only when the
    /// frame is submitted (or encoded) and the device took it, and then its [`generation`](super::Batch::generation)
    /// moves; a frame that is dropped, or fails before that, leaves the batch as it was. Set a shape **before** writing
    /// or reading what it resizes: a `set` that would change a buffer this frame already wrote or reserved a read-back
    /// of is an error. A library op stage follows too: its plan for the new length is built here (or found among the
    /// plans the batch kept) and its outputs that follow the operand are made again. The device may clear recycled
    /// buffers and the scratch of a new plan in submissions of its own, made inside this call, so a `set` that resizes can
    /// cost submissions; the frame that follows it is one submission like any frame (the allocation rule is in the `batch` module doc).
    ///
    /// # Errors
    ///
    /// [`Error::Foreign`] for a parameter of another batch; [`Error::ShapeMismatch`] for a shape that would change a
    /// buffer the frame has used; the device's
    /// failures (a buffer too long, or a limit of the new extents). The frame is unchanged after an error.
    pub fn set<T: ParamType>(&mut self, param: &Param<T>, value: T) -> Result<(), Error> {
        let core = Arc::clone(&self.core);
        if param.batch() != core.id || param.id() >= core.params.len() {
            return Err(Error::Foreign {
                batch: core.name.clone(),
                stage: String::from("a parameter"),
            });
        }
        let packed = value.pack();
        let mut set = vec![(param.id(), packed)];
        if core.params[param.id()].shape {
            self.resize(param.id(), &mut set)?;
        }
        for (id, value) in set {
            self.pending.retain(|(pending, _)| *pending != id);
            self.pending.push((id, value));
        }
        Ok(())
    }

    /// The values of the parameters in this frame: the batch's, with what the frame has set.
    fn values(&self) -> Vec<Value> {
        let mut values = self
            .core
            .run
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values
            .clone();
        for (id, value) in &self.pending {
            values[*id] = *value;
        }
        values
    }

    /// The extents of every buffer in this frame.
    fn dims(&self) -> Vec<[u32; 2]> {
        match &self.staged {
            Some(staged) => staged.dims.clone(),
            None => self
                .core
                .run
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .dims
                .clone(),
        }
    }

    /// The extents of the buffer numbered `id` in this frame.
    fn dims_of(&self, id: usize) -> [u32; 2] {
        match &self.staged {
            Some(staged) => staged.dims[id],
            None => {
                self.core
                    .run
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .dims[id]
            }
        }
    }

    /// The buffers of this frame: the resized ones when it resized, the batch's otherwise.
    fn buffers(&self) -> Vec<Arc<dyn Any>> {
        match &self.staged {
            Some(staged) => staged.buffers.clone(),
            None => self
                .core
                .run
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .buffers
                .clone(),
        }
    }

    /// The buffer numbered `id` in this frame, as `T`.
    fn buffer_of<T: GpuPod>(&self, id: usize) -> Arc<D::Buffer<T>>
    where
        D::Buffer<T>: 'static,
    {
        match &self.staged {
            Some(staged) => typed::<D, T>(&staged.buffers[id]),
            None => typed::<D, T>(
                &self
                    .core
                    .run
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .buffers[id],
            ),
        }
    }

    /// The step of `set` for a parameter that gives extents: the new extents of every buffer, the buffers and stages
    /// that follow them, kept in `staged`. `set` gets the derived values that changed with it. Nothing changes in the
    /// frame when it fails.
    fn resize(&mut self, id: usize, set: &mut Vec<(usize, Value)>) -> Result<(), Error> {
        let core = Arc::clone(&self.core);
        let mut values = self.values();
        values[id] = set[0].1;
        for (derived, entry) in core.params.iter().enumerate() {
            if let Some((source, derive)) = &entry.derived
                && *source == id
            {
                values[derived] = derive.apply(values[id]);
                set.push((derived, values[derived]));
            }
        }
        let dims: Vec<[u32; 2]> = core
            .buffers
            .iter()
            .map(|buffer| buffer.rule.dims(&values))
            .collect();
        let (sealed_dims, sealed) = {
            let run = core.run.lock().unwrap_or_else(PoisonError::into_inner);
            (run.dims.clone(), run.buffers.clone())
        };
        let (here_dims, here) = (self.dims(), self.buffers());
        if let Some(&used) = self
            .used
            .iter()
            .find(|&&buffer| dims[buffer] != here_dims[buffer])
        {
            let [width, height] = here_dims[used];
            return Err(Error::ShapeMismatch {
                batch: core.name.clone(),
                stage: core.params[id].name.clone(),
                expected: format!(
                    "`{}` at {width} x {height}, which the frame has already written or read",
                    core.buffers[used].name
                ),
                found: format!(
                    "{:?}, which makes it {} x {}",
                    set[0].1, dims[used][0], dims[used][1]
                ),
            });
        }
        // A buffer keeps the frame's own where its extents are as they are here, goes back to the batch's where they are
        // as the batch has them, and is made again otherwise.
        let mut buffers = Vec::with_capacity(dims.len());
        for (buffer, meta) in core.buffers.iter().enumerate() {
            buffers.push(if dims[buffer] == here_dims[buffer] {
                Arc::clone(&here[buffer])
            } else if dims[buffer] == sealed_dims[buffer] {
                Arc::clone(&sealed[buffer])
            } else {
                let [width, height] = dims[buffer];
                let len = u64::from(width) * u64::from(height);
                let Ok(count) = u32::try_from(len) else {
                    return Err(Error::Device(map::Error::TooLong {
                        len: usize::try_from(len).unwrap_or(usize::MAX),
                    }));
                };
                (meta.alloc)(&*core.device, count).map_err(device_error::<D>)?
            });
        }
        let same = |buffer: usize, other: &[Arc<dyn Any>], other_dims: &[[u32; 2]]| {
            dims[buffer] == other_dims[buffer] && Arc::ptr_eq(&buffers[buffer], &other[buffer])
        };
        let replaced = (0..buffers.len())
            .filter(|&buffer| !same(buffer, &sealed, &sealed_dims))
            .count();
        if replaced == 0 {
            self.staged = None;
            return Ok(());
        }
        let layout = Layout {
            buffers: &buffers,
            dims: &dims,
        };
        let mut prepared = Vec::new();
        for (index, stage) in core.stages.iter().enumerate() {
            let touched = || stage.reads.iter().copied().chain(stage.written());
            if touched().all(|buffer| same(buffer, &sealed, &sealed_dims)) {
                continue;
            }
            if let Some(staged) = &self.staged
                && touched().all(|buffer| same(buffer, &here, &here_dims))
                && let Some(kept) = staged.prepared.iter().find(|kept| kept.stage == index)
            {
                prepared.push(kept.clone());
                continue;
            }
            prepared.push(match (&stage.remake, &stage.plan) {
                (Some(remake), _) => StagedStage {
                    stage: index,
                    prepared: Some(Arc::new(remake(&*core.device, &layout)?)),
                    plan: None,
                },
                (None, Some(plan)) => Self::replan(&core, index, plan, &layout)?,
                (None, None) => {
                    unreachable!("a stage is a kernel with a remake or an op with a plan")
                }
            });
        }
        self.staged = Some(Staged {
            values,
            dims,
            buffers,
            replaced,
            prepared,
        });
        Ok(())
    }

    /// The op stage `index` prepared again over the buffers of `layout`, inside the `set`: over the plan the batch kept
    /// for the new length when it has one, over a new plan (built here, which may take submissions of the device's own:
    /// see the allocation rule in the `batch` module doc) otherwise. A length of zero prepares nothing, the stage's `empty` rule runs in the frame.
    fn replan(
        core: &Core<D>,
        index: usize,
        plan: &OpPlan<D>,
        layout: &Layout<'_>,
    ) -> Result<StagedStage<D>, Error> {
        let stage = &core.stages[index];
        let operands = Operands::new(
            stage
                .reads
                .iter()
                .map(|&id| Arc::clone(&layout.buffers[id]))
                .collect(),
            stage
                .written()
                .map(|id| Arc::clone(&layout.buffers[id]))
                .collect(),
            stage.len(layout.dims),
        );
        let len = operands.len();
        if len == 0 {
            return Ok(StagedStage {
                stage: index,
                prepared: None,
                plan: None,
            });
        }
        let kept = core
            .run
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .plans[index]
            .iter()
            .find(|(kept, _)| *kept == len)
            .map(|(_, plan)| Arc::clone(plan));
        let device = &*core.device;
        let plan_of_len = match kept {
            Some(kept) => kept,
            None => plan.build(device, len).map_err(device_error::<D>)?,
        };
        let prepared = plan
            .bind(device, &plan_of_len, &operands)
            .map_err(device_error::<D>)?;
        Ok(StagedStage {
            stage: index,
            prepared: Some(Arc::new(prepared)),
            plan: Some((len, plan_of_len)),
        })
    }

    /// Make the resize of this frame the batch's: its buffers, extents and prepared stages, and the values of the
    /// parameters that give extents. The generation moves by the number of buffers it replaced.
    fn commit(&mut self) {
        let Some(staged) = self.staged.take() else {
            return;
        };
        let mut run = self.core.run.lock().unwrap_or_else(PoisonError::into_inner);
        for (id, entry) in self.core.params.iter().enumerate() {
            if entry.shape {
                run.values[id] = staged.values[id];
            }
        }
        run.dims = staged.dims;
        run.buffers = staged.buffers;
        for entry in staged.prepared {
            run.prepared[entry.stage] = entry.prepared;
            if let Some(plan) = entry.plan {
                keep_plan(&mut run.plans[entry.stage], plan);
            }
        }
        drop(run);
        self.core
            .generation
            .fetch_add(staged.replaced as u64, Ordering::Relaxed);
    }

    /// Reserve the read-back of a handle: its values after this frame's stages, copied inside the frame's one
    /// submission. The ticket resolves when the frame has run.
    ///
    /// # Errors
    ///
    /// [`Error::Foreign`], [`Error::NotOutput`] and the device's failures.
    pub fn ticket<T: Element + Send, R>(&mut self, handle: &Stage<T, R>) -> Result<Ticket<T>, Error>
    where
        T::Word: Send,
        D::Buffer<T::Word>: 'static,
    {
        let id = self.core.locate(handle)?;
        let buffer = self.buffer_of::<T::Word>(id);
        let read = self
            .core
            .device
            .reserve_read::<T::Word>(&buffer)
            .map_err(device_error::<D>)?;
        self.used.push(id);
        let cell = Arc::new(TicketCell {
            batch: self.core.name.clone(),
            stage: self.core.buffers[id].name.clone(),
            state: Mutex::new(TicketState::Reserved(None)),
        });
        self.reads.push(Box::new(Reserved::<D, T> {
            read,
            cell: Arc::clone(&cell),
        }));
        Ok(Ticket { cell })
    }

    /// Ask for the GPU time of this frame: a [`FrameTime`], read after the submission like a ticket, with the span from
    /// the begin of the frame's first pass to the end of its last one (see [`FrameTimes`]). Only a frame that asks
    /// records timestamps, so a frame of a timed batch that does not call this costs nothing; calling it again returns
    /// the same measurement.
    ///
    /// A batch built without a timing level, a device without timestamp queries (the CPU twin among them) and a frame
    /// with no pass give `Ok(None)` when the frame is submitted; [`Batch::frame_timing`](super::Batch::frame_timing)
    /// tells which of these a batch is, up front. On the CPU twin there is never a time:
    ///
    /// ```
    /// use std::sync::Arc;
    /// use std::task::{Context, Poll, Waker};
    /// use gust::batch::FrameTiming;
    /// use gust::cpu::CpuDevice;
    ///
    /// fn block_on<F: std::future::Future>(future: F) -> F::Output {
    ///     let mut future = std::pin::pin!(future);
    ///     let mut context = Context::from_waker(Waker::noop());
    ///     loop {
    ///         if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
    ///             return value;
    ///         }
    ///     }
    /// }
    ///
    /// let device = Arc::new(CpuDevice::new());
    /// let (mut batch, x) = gust::gpu_batch!("timed", on(&device); in_flight(1); |b| {
    ///     b.frame_timing(FrameTiming::Frame);
    ///     let x = b.input::<u32>(4u32);
    ///     x
    /// })?;
    /// assert!(batch.frame_timing().is_none());
    /// let mut frame = block_on(batch.begin())?;
    /// frame.write(&x, &[1, 2, 3, 4])?;
    /// let time = frame.time();
    /// let values = frame.ticket(&x)?;
    /// let _submitted = frame.submit()?;
    /// assert_eq!(block_on(values)?, [1, 2, 3, 4]);
    /// assert!(block_on(time)?.is_none());
    /// # Ok::<(), gust::batch::Error>(())
    /// ```
    pub fn time(&mut self) -> FrameTime {
        if self.timed.is_none() {
            // On `wasm32` the device's futures are not `Send`: the cell stays on its thread there.
            #[allow(clippy::arc_with_non_send_sync)]
            let cell = Arc::new(TimeCell::new(&self.core.name));
            self.timed = Some(TimedFrame {
                cell,
                recorded: None,
            });
        }
        let timed = self.timed.as_ref().expect("set above");
        FrameTime {
            cell: Arc::clone(&timed.cell),
        }
    }

    /// Run the frame: one queue submission on the GPU whatever the number of tickets. Returns at once.
    ///
    /// # Errors
    ///
    /// [`Error::Empty`] when a `min` or `max` stage is over zero elements, and the device's failures. The frame's
    /// tickets then resolve to [`Error::NotSubmitted`].
    pub fn submit(mut self) -> Result<Submitted<D>, Error> {
        let core = Arc::clone(&self.core);
        let mut encoder = core.device.new_encoder();
        let frame = self.record(&mut encoder)?;
        let submission = core
            .device
            .submit_frame(frame, encoder)
            .map_err(device_error::<D>)?;
        self.commit();
        Ok(self.mark_submitted(submission))
    }

    /// Record the frame into the caller's encoder instead of making a submission: the writes, the extent records of
    /// the stages whose parameters were set in this frame or are stale (the host cannot tell whether an earlier encoded
    /// frame's encoder ran, so what it set is recorded again until a frame made by `submit` clears it), the stages in
    /// order and the copies of the tickets. Nothing runs until the caller submits the encoder and reports it with
    /// [`Encoded::submitted`]; one queue submission can then carry the frame and the caller's own passes. Returns at
    /// once.
    ///
    /// A pass the caller records into `encoder` after this call reads this frame's results through a
    /// [`View`](super::View), and a pass recorded before it reads the frame before (see
    /// [`Batch::view`](super::Batch::view) for the rule).
    ///
    /// The frame stays live until it is reported as submitted (`Encoded::submitted`) or dropped, `begin` refuses a
    /// new frame while one is live, and its tickets resolve only after the report. Frames run in the order of the
    /// caller's submissions, so with `in_flight(n)` up to `n` reported frames may still be running on the GPU while the
    /// next one is encoded, as long as the caller submits them in the order it began them (decision D258; GUST does
    /// not check the order, and results are wrong if it is broken).
    ///
    /// # Errors
    ///
    /// [`Error::Empty`] when a `min` or `max` stage is over zero elements; it is raised before anything is recorded,
    /// so `encoder` is left untouched.
    ///
    /// The device's failures. The frame is then gone (its tickets resolve to [`Error::NotSubmitted`], the batch can
    /// begin a new frame), and `encoder` holds whatever was recorded before the failure: on the GPU the copies and the
    /// passes of the stages that came before the one that failed, and nothing of the stage that failed. Discard the
    /// encoder; do not submit it.
    pub fn encode(mut self, encoder: &mut D::Encoder) -> Result<Encoded<D>, Error> {
        let core = Arc::clone(&self.core);
        // Set before recording: a failure inside `record` leaves the frame's copies in the caller's encoder; discard it
        // (see the errors above).
        self.encoded = true;
        let frame = self.record(encoder)?;
        let encoded = core
            .device
            .encoded_frame(frame, encoder)
            .map_err(device_error::<D>)?;
        self.commit();
        Ok(Encoded { run: self, encoded })
    }

    /// First step of `submit` and of `encode`: the writes, the extent records of the stages that have a parameter set
    /// in this frame or left stale by a frame recorded into a caller's encoder, the stages in order, then the copies of
    /// the tickets, all into `encoder`.
    fn record(&mut self, encoder: &mut D::Encoder) -> Result<D::Frame, Error> {
        let core = Arc::clone(&self.core);
        let device = &*core.device;
        let mut frame = self.frame.take().expect("a live frame holds its recorder");
        // An op over zero elements: its rule runs here, before anything is encoded and before a resize is made the
        // batch's, so a refusal leaves the encoder and the batch as they were and `submit` and `encode` both return it.
        let sizes = self.dims();
        core.stages.iter().try_for_each(|stage| {
            if let (0, Some(empty)) = (stage.len(&sizes), &stage.empty) {
                match empty {
                    Empty::Identity(write) => {
                        write(device, &mut frame).map_err(device_error::<D>)?;
                    }
                    Empty::Nothing => {}
                    Empty::Refuse => {
                        return Err(Error::Empty {
                            batch: core.name.clone(),
                            stage: stage.name.clone(),
                        });
                    }
                }
            }
            Ok::<(), Error>(())
        })?;
        // The frame's resize stays in `staged` here: `submit` and `encode` make it the batch's once the device has
        // taken the frame, so a frame that fails before that is dropped and leaves the batch as it was.
        let (stale, mut prepared) = {
            let run = core.run.lock().unwrap_or_else(PoisonError::into_inner);
            (run.stale.clone(), run.prepared.clone())
        };
        if let Some(staged) = &self.staged {
            for kept in &staged.prepared {
                prepared[kept.stage].clone_from(&kept.prepared);
            }
        }
        let values = Values {
            params: self.values(),
            dims: sizes,
        };
        let mut recorded = 0;
        let result = core.stages.iter().try_for_each(|stage| {
            let changed = stage
                .params
                .iter()
                .any(|id| stale.contains(id) || self.pending.iter().any(|(set, _)| set == id));
            if let (true, Some(update)) = (changed, &stage.update) {
                update(device, &mut frame, &values).map_err(device_error::<D>)?;
                recorded += 1;
            }
            Ok::<(), Error>(())
        });
        core.run
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extent_records += recorded;
        result?;
        device
            .frame_uploads(&mut frame, encoder)
            .map_err(device_error::<D>)?;
        self.record_stages(encoder, &prepared)?;
        for slot in &mut self.reads {
            slot.record(device, encoder);
        }
        Ok(frame)
    }

    /// The passes of the stages, in order: when the frame asked for its time and the batch can give it, the first pass
    /// of the frame writes the first timestamp and the last pass the last one (and at level `Passes` each stage's
    /// first and last pass write that stage's pair); a resolve and a copy follow the last stage. A frame that did not
    /// ask records none of it.
    fn record_stages(
        &mut self,
        encoder: &mut D::Encoder,
        prepared: &[Option<Arc<D::Prepared>>],
    ) -> Result<(), Error> {
        let core = Arc::clone(&self.core);
        let device = &*core.device;
        let mut timing = None;
        if let (Some(timed), Some(state)) = (self.timed.as_mut(), core.timing.as_ref()) {
            // The timed stages: the ones that have a pass (a stage over zero elements has none).
            let groups: Vec<usize> = prepared
                .iter()
                .enumerate()
                .filter(|(_, stage)| {
                    stage
                        .as_deref()
                        .is_some_and(|stage| device.stage_passes(stage) > 0)
                })
                .map(|(index, _)| index)
                .collect();
            if !groups.is_empty()
                && let Some(clock) = device.frame_clock(&state.handle)
            {
                let plan = TimePlan {
                    level: state.level,
                    names: groups
                        .iter()
                        .map(|&index| core.stages[index].name.clone())
                        .collect(),
                    period_ns: state.period_ns,
                };
                timed.recorded = None;
                timing = Some((groups, plan, clock));
            }
        }
        let Some((groups, plan, mut clock)) = timing else {
            for stage in prepared.iter().flatten() {
                device
                    .frame_stage(encoder, stage)
                    .map_err(device_error::<D>)?;
            }
            return Ok(());
        };
        for (index, stage) in prepared.iter().enumerate() {
            let Some(stage) = stage else { continue };
            match groups.iter().position(|&group| group == index) {
                Some(group) => {
                    let (begin, end) = plan.slots(group);
                    device
                        .frame_stage_timed(encoder, stage, &mut clock, begin, end)
                        .map_err(device_error::<D>)?;
                }
                None => device
                    .frame_stage(encoder, stage)
                    .map_err(device_error::<D>)?,
            }
        }
        device.frame_clock_end(encoder, &mut clock, plan.used());
        self.timed
            .as_mut()
            .expect("a timed frame keeps its cell")
            .recorded = Some(Recorded { clock, plan });
        Ok(())
    }

    /// Last step of `submit`: the frame is on the device; start its read-backs, keep its parameters and let the next
    /// `begin` wait for it.
    fn mark_submitted(mut self, submission: D::Submission) -> Submitted<D> {
        let core = Arc::clone(&self.core);
        for slot in self.reads.drain(..) {
            slot.start(&core.device, &submission);
        }
        let time = self.timed.take().map(|timed| {
            match timed.recorded {
                Some(Recorded { clock, plan }) => timed
                    .cell
                    .start(core.device.begin_time(clock, &submission), plan),
                None => timed.cell.none(),
            }
            FrameTime { cell: timed.cell }
        });
        {
            let mut run = core.run.lock().unwrap_or_else(PoisonError::into_inner);
            if self.encoded {
                // The caller's encoder may not have run: what this frame set may not be on the device.
                run.mark_stale(self.pending.iter().map(|(id, _)| *id));
            } else {
                // The frame recorded every stale parameter and the batch itself submitted it.
                run.stale.clear();
            }
            for (id, value) in self.pending.drain(..) {
                run.values[id] = value;
            }
            run.live = false;
            run.pending.push_back(submission.clone());
        }
        self.done = true;
        Submitted {
            core,
            submission,
            time,
        }
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
        if let Some(timed) = self.timed.take() {
            timed.cell.fail();
        }
        if self.encoded || self.frame.is_none() {
            // The frame was recorded: the caller may still submit its encoder, or a CPU submission failed after its
            // extent records ran. Either way the device can hold this frame's parameters and writes.
            self.core
                .run
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .mark_stale(self.pending.iter().map(|(id, _)| *id));
        }
        release(&self.core);
    }
}

/// A frame recorded into a caller's encoder by [`FrameRun::encode`], waiting for the caller's submission. Submit the
/// encoder, then report it with [`submitted`](Self::submitted).
///
/// Dropping it without that report, or reporting a submission that did not carry the encoder (the encoder was dropped
/// or never submitted), makes no frame: its tickets resolve to [`Error::NotSubmitted`], never to bytes an earlier frame
/// left behind. What the next frame sees follows the rule of the batch (decision D258: data holds the value of the last
/// frame that reached the queue), with these cases, which the host cannot tell apart from the outside:
///
/// * An input: the `write`s of a frame that never reached the queue are lost (the host keeps no bytes), so the next
///   frame sees the input as the last frame that did reach the queue left it. If the caller still submits the encoder
///   of a dropped `Encoded`, its writes do land in the device.
/// * A parameter: the `set`s of a reported frame hold for the later frames whether or not its encoder ran (the next
///   frames record those parameters again, so the device follows the host); the `set`s of a dropped `Encoded` are
///   discarded, and the next frame records the values from before it, even if the caller submitted its encoder.
/// * A resize: a frame that resized the batch ([`FrameRun::set`]) makes the new buffers, the new extents and the values
///   of the parameters that give them the batch's when `encode` returns (so that a pass the caller records after it
///   can fetch the new views, see [`Batch::generation`](super::Batch::generation)), whether or not the encoder runs or
///   the `Encoded` is reported. The next frame records the stages' extents again if the frame was dropped.
///
/// An encoder whose frame was reported with a submission that did not carry it, or was dropped, must be dropped, never
/// submitted after a later frame was begun: it would write the batch's buffers out of order with that frame.
#[must_use = "submit the encoder, then report it with `submitted`"]
pub struct Encoded<D: BatchDevice> {
    run: FrameRun<D>,
    encoded: D::Encoded,
}

impl<D: BatchDevice> core::fmt::Debug for Encoded<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Encoded")
            .field("batch", &self.run.core.name)
            .finish_non_exhaustive()
    }
}

impl<D: BatchDevice> Encoded<D> {
    /// The caller submitted the encoder: `token` is what the submission returned (on the GPU, the index of the queue
    /// submission). Call it right after the submission and before any later one, so that waiting for the frame does not
    /// also wait for work submitted after it. The tickets are live from here, and the batch keeps the frame as one of
    /// the frames in flight. Submit the encoder first: reporting a submission that has not been made is an error of the
    /// device (a validation error on the GPU).
    pub fn submitted(self, token: &D::Token) -> Submitted<D> {
        let Self { run, encoded } = self;
        let submission = run.core.device.encoded_submitted(encoded, token);
        run.mark_submitted(submission)
    }
}

/// A frame that was submitted. Its tickets are the way to its values; this waits for the frame itself.
#[must_use = "a submitted frame is waited for through its tickets, or with `wait`"]
pub struct Submitted<D: BatchDevice> {
    core: Arc<Core<D>>,
    submission: D::Submission,
    /// The time of the frame, when it asked for one.
    time: Option<FrameTime>,
}

impl<D: BatchDevice> core::fmt::Debug for Submitted<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Submitted")
            .field("batch", &self.core.name)
            .finish_non_exhaustive()
    }
}

impl<D: BatchDevice> Submitted<D> {
    /// The time of the frame, when it asked for one with [`FrameRun::time`]: the same measurement that call returned.
    #[must_use]
    pub fn frame_time(&self) -> Option<FrameTime> {
        self.time.clone()
    }

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
    pub fn read<T: Element, R>(&self, handle: &Stage<T, R>) -> Result<Vec<T>, Error>
    where
        D::Buffer<T::Word>: 'static,
    {
        let core = &self.batch.core;
        let (id, buffer, _) = core.output(handle)?;
        let words = core.device.readback(&buffer).map_err(device_error::<D>)?;
        T::unpack(words).map_err(|bad| Error::InvalidEnum {
            batch: core.name.clone(),
            stage: core.buffers[id].name.clone(),
            index: bad.index(),
            word: bad.word(),
        })
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
