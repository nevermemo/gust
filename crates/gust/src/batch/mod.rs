//! A sealed batch of stages: `gpu_batch!` (plan m14.s2, slice s2a2).
//!
//! ```text
//! let (mut batch, h) = gust::gpu_batch!("exposure", on(&device); in_flight(1); |b| {
//!     let size = b.param::<(u32, u32)>((16, 8));
//!     let pixels = b.input::<Vec4>(&size);
//!     let lum = gust::gpu_task!(&size, |x, y| -> f32 { pixels[(x, y)].x * 2.0 });
//!     Handles { size, pixels, lum }
//! })?;
//! let mut f = batch.begin().await?;
//! f.write(&h.pixels, &image)?;
//! let out = f.ticket(&h.lum)?;
//! f.submit()?;
//! let values = out.await?;
//! ```
//!
//! The block is the build scope: the inputs, parameters and stages it records belong to the batch, and when it ends the
//! batch is validated once and sealed (the structure is frozen, the data stays writable). A frame
//! ([`FrameRun`]) is one trip to the device: `write` and `set` change the data, `ticket` reserves a read-back, `submit`
//! makes one queue submission for all of it and returns a [`Submitted`]. The batch is generic over a [`BatchDevice`]
//! with two implementations: the GPU device (`gust_wgpu::HeadlessDevice`) and [`CpuDevice`](crate::cpu::CpuDevice),
//! which runs the stages in order at `submit` and has its tickets ready then.
//!
//! A batch keeps up to `in_flight(n)` frames in flight (1 to 4; `in_flight(1)` when the option is left out): a frame
//! is submitted while the ones before it may still be running, and [`Batch::begin`] waits only when `n` frames are
//! already in flight, for the oldest. The frames share the batch's one set of buffers and run in the order they were
//! submitted (decision D258), so the batch's data persists from frame to frame: an input or parameter that a frame
//! does not `write` or `set` holds the value of the last frame that reached the queue, whatever `n` is, on both
//! devices. A frame's tickets read that frame's results; a later frame's writes never reach an earlier frame. A
//! shape cannot change after the seal (a `set` that would is [`Error::ShapeMismatch`]).

use core::any::Any;
use core::cell::{Cell, RefCell};
use core::future::Future;
use core::pin::Pin;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use crate::map::{self, MapHost};
use crate::{Call, GpuPod, Kernel, Resource, Run, kind};

mod frame;
mod ops;
mod shape;
mod slot;

#[cfg(not(target_arch = "wasm32"))]
pub use frame::Blocking;
pub use frame::{Encoded, FrameRun, Submitted, Ticket};
pub use shape::{D1, D2, IntoShape, ParamType, Rank};
#[doc(hidden)]
pub use shape::{ShapeSource, Value, Values};

/// A boxed future: `Send` on native targets, where a [`Ticket`] must be, and not on `wasm32`, whose handles are not.
#[cfg(not(target_arch = "wasm32"))]
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// A boxed future: `Send` on native targets, where a [`Ticket`] must be, and not on `wasm32`, whose handles are not.
#[cfg(target_arch = "wasm32")]
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T>>>;

/// Why a batch call failed. Each variant names the batch and, where one applies, the stage by the name of its `let`.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A recording call came after the seal (a [`Builder`] kept beyond its block).
    Sealed {
        /// The batch.
        batch: String,
    },
    /// A `gpu_batch!` was started while another one was being built on this thread.
    NestedBuild {
        /// The batch that was being started.
        batch: String,
    },
    /// A handle of another batch was used.
    Foreign {
        /// The batch that was used wrongly.
        batch: String,
        /// The stage, input or parameter that took the handle (or the data call's target).
        stage: String,
    },
    /// A stage binds one buffer both read-only and read-write.
    Hazard {
        /// The batch.
        batch: String,
        /// The stage.
        stage: String,
        /// The buffer, by the name of its `let`.
        buffer: String,
    },
    /// A data call does not fit the sealed shape: a `write` of another length, or a `set` that would change a shape.
    ShapeMismatch {
        /// The batch.
        batch: String,
        /// The input or parameter.
        stage: String,
        /// What the batch was sealed with.
        expected: String,
        /// What the call brought.
        found: String,
    },
    /// A ticket's frame was never submitted: it was dropped, or its submission failed.
    NotSubmitted {
        /// The batch.
        batch: String,
    },
    /// A handle the block did not return was used.
    NotOutput {
        /// The batch.
        batch: String,
        /// The stage, input or parameter.
        stage: String,
    },
    /// A `write` named a stage that is not an input.
    NotInput {
        /// The batch.
        batch: String,
        /// The stage.
        stage: String,
    },
    /// `begin` while the frame before it was neither submitted nor dropped.
    FrameLive {
        /// The batch.
        batch: String,
    },
    /// The device failed.
    Device(map::Error),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Sealed { batch } => {
                write!(f, "batch `{batch}` is sealed: its structure cannot change")
            }
            Self::NestedBuild { batch } => write!(
                f,
                "batch `{batch}` was started inside the build block of another batch"
            ),
            Self::Foreign { batch, stage } => write!(
                f,
                "`{stage}` of batch `{batch}` took a handle that belongs to another batch"
            ),
            Self::Hazard {
                batch,
                stage,
                buffer,
            } => write!(
                f,
                "stage `{stage}` of batch `{batch}` binds `{buffer}` read-only and read-write"
            ),
            Self::ShapeMismatch {
                batch,
                stage,
                expected,
                found,
            } => write!(
                f,
                "`{stage}` of batch `{batch}` was sealed with {expected} and the call brought {found}"
            ),
            Self::NotSubmitted { batch } => {
                write!(f, "the frame of batch `{batch}` was never submitted")
            }
            Self::NotOutput { batch, stage } => write!(
                f,
                "`{stage}` of batch `{batch}` is not among the handles the build block returned"
            ),
            Self::NotInput { batch, stage } => {
                write!(
                    f,
                    "`{stage}` of batch `{batch}` is not an input, so it cannot be written"
                )
            }
            Self::FrameLive { batch } => write!(
                f,
                "batch `{batch}` has a frame that is neither submitted nor dropped (one frame is in flight)"
            ),
            Self::Device(error) => write!(f, "device: {error}"),
        }
    }
}

impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Device(error) => Some(error),
            _ => None,
        }
    }
}

impl From<map::Error> for Error {
    fn from(error: map::Error) -> Self {
        Self::Device(error)
    }
}

/// What a device must offer to run a batch. Implemented by `gust_wgpu::HeadlessDevice` and by
/// [`CpuDevice`](crate::cpu::CpuDevice). Macro support: the surface is the batch's, not a stable one for other
/// devices.
#[doc(hidden)]
pub trait BatchDevice: MapHost + Sized + 'static {
    /// A stage that was sealed with the batch.
    type Prepared;
    /// What one frame collects before it is recorded: its writes and parameter records, in the order of the calls.
    type Frame;
    /// What a frame is recorded into: on the GPU the command encoder of the queue submission (the frame's own, or the
    /// caller's for [`FrameRun::encode`]), on the CPU the list of steps that run at the submission.
    type Encoder;
    /// What the caller of [`FrameRun::encode`] reports after its own submission: on the GPU the index of the queue
    /// submission, on the CPU the job of the encoder it ran.
    type Token;
    /// What a frame recorded into a caller's encoder keeps until the caller reports its submission.
    type Encoded;
    /// A reserved read-back of a buffer.
    type Read<T: GpuPod>: 'static;
    /// What a submitted frame waits for.
    type Submission: Clone;

    /// Seal the kernel call that `build` makes over `keep`.
    ///
    /// # Errors
    ///
    /// The device's refusal of the call.
    fn prepare<K, R, const N: usize>(
        &self,
        keep: R,
        build: impl for<'o> Fn(&'o R) -> Call<'o, Self, K, N> + 'static,
    ) -> Result<Self::Prepared, Self::Error>
    where
        Self: Run<K, N>,
        K: Kernel,
        R: 'static;

    /// Start a frame.
    ///
    /// # Errors
    ///
    /// The device's failure to make one.
    fn begin_frame(&self) -> Result<Self::Frame, Self::Error>;

    /// Record a copy of `data` into `target`, in the order of the frame's calls.
    ///
    /// # Errors
    ///
    /// The device's failure to record it.
    fn frame_write<T: GpuPod>(
        &self,
        frame: &mut Self::Frame,
        target: &Arc<Self::Buffer<T>>,
        data: &[T],
    ) -> Result<(), Self::Error>
    where
        Self::Buffer<T>: 'static;

    /// Record a copy of `value` into the uniform `target`.
    ///
    /// # Errors
    ///
    /// The device's failure to record it.
    fn frame_uniform<T: GpuPod, U>(
        &self,
        frame: &mut Self::Frame,
        target: &Arc<U>,
        value: T,
    ) -> Result<(), Self::Error>
    where
        U: Resource<Element = T, Device = Self, Kind = kind::Uniform> + 'static;

    /// An encoder of the device's own, for a frame that is submitted with [`FrameRun::submit`].
    fn new_encoder(&self) -> Self::Encoder;

    /// Record the frame's writes and parameter records into `encoder`, in the order of the calls.
    ///
    /// # Errors
    ///
    /// The device's failure to record them.
    fn frame_uploads(
        &self,
        frame: &mut Self::Frame,
        encoder: &mut Self::Encoder,
    ) -> Result<(), Self::Error>;

    /// Record the dispatch of a stage.
    ///
    /// # Errors
    ///
    /// The device's failure to record it. What the encoder holds then is the device's to document.
    fn frame_stage(
        &self,
        encoder: &mut Self::Encoder,
        stage: &Self::Prepared,
    ) -> Result<(), Self::Error>;

    /// Reserve the read-back of `source`.
    ///
    /// # Errors
    ///
    /// The device's failure to reserve it.
    fn reserve_read<T: GpuPod>(
        &self,
        source: &Arc<Self::Buffer<T>>,
    ) -> Result<Self::Read<T>, Self::Error>
    where
        Self::Buffer<T>: 'static;

    /// Record the copy of a reserved read-back, after the stages.
    fn frame_read<T: GpuPod>(&self, encoder: &mut Self::Encoder, read: &mut Self::Read<T>);

    /// Submit the frame with its own encoder: one queue submission on the GPU, the recorded steps in order on the CPU.
    ///
    /// # Errors
    ///
    /// The device's failure to submit it.
    fn submit_frame(
        &self,
        frame: Self::Frame,
        encoder: Self::Encoder,
    ) -> Result<Self::Submission, Self::Error>;

    /// The frame was recorded into a caller's encoder: what it keeps until the caller reports its submission. On the
    /// GPU this also records the epoch word that tells a ticket whether the encoder ran.
    ///
    /// # Errors
    ///
    /// The device's failure to record it.
    fn encoded_frame(
        &self,
        frame: Self::Frame,
        encoder: &mut Self::Encoder,
    ) -> Result<Self::Encoded, Self::Error>;

    /// The caller submitted the encoder: the submission of the frame.
    fn encoded_submitted(&self, encoded: Self::Encoded, token: &Self::Token) -> Self::Submission;

    /// The values of a reserved read-back, after the submission that carried its copy: `None` when the copy never ran
    /// (the caller's encoder was dropped, or never submitted).
    fn begin_read<T: GpuPod + Send>(
        &self,
        read: Self::Read<T>,
        submission: &Self::Submission,
    ) -> BoxFuture<Result<Option<Vec<T>>, map::Error>>;

    /// A future that completes when the submission has run.
    fn completion(&self, submission: &Self::Submission) -> BoxFuture<Result<(), map::Error>>;

    /// Block until the submission has run. Not on `wasm32`, where nothing may block.
    ///
    /// # Errors
    ///
    /// The device's failure to wait.
    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self, submission: &Self::Submission) -> Result<(), Self::Error>;
}

/// What names a device for `gpu_batch!`'s `on(..)`: an `Arc` of it, or a reference to one. Macro support.
#[doc(hidden)]
pub trait DeviceRef {
    /// The device.
    type Device: BatchDevice;

    /// A shared handle to it.
    fn handle(&self) -> Arc<Self::Device>;
}

impl<D: BatchDevice> DeviceRef for Arc<D> {
    type Device = D;

    fn handle(&self) -> Arc<D> {
        Arc::clone(self)
    }
}

impl<D: BatchDevice> DeviceRef for &Arc<D> {
    type Device = D;

    fn handle(&self) -> Arc<D> {
        Arc::clone(self)
    }
}

/// A buffer of a batch (an input or the result of a stage), seen from outside: the handle a frame writes or reads.
/// `T` is the element, `R` the rank ([`D1`] or [`D2`]); a 2-D handle is row-major.
pub struct Stage<T, R> {
    batch: u64,
    id: usize,
    dims: [u32; 2],
    marker: shape::Marker<(T, R)>,
}

impl<T, R> Clone for Stage<T, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, R> Copy for Stage<T, R> {}

impl<T, R> core::fmt::Debug for Stage<T, R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Stage")
            .field("batch", &self.batch)
            .field("id", &self.id)
            .field("dims", &self.dims)
            .finish()
    }
}

impl<T, R> Stage<T, R> {
    fn new(batch: u64, id: usize, dims: [u32; 2]) -> Self {
        Self {
            batch,
            id,
            dims,
            marker: core::marker::PhantomData,
        }
    }

    /// The width: the extent along x (the whole length of a 1-D handle).
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.dims[0]
    }

    /// The height: the extent along y (1 for a 1-D handle).
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.dims[1]
    }

    /// The number of elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.dims[0] as usize * self.dims[1] as usize
    }

    /// Whether there is no element.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Stage<T, D1> {
    /// The shape of this handle as a 1-D shape (the extent of an in-place stage). Macro support.
    #[doc(hidden)]
    #[must_use]
    pub const fn shape(&self) -> u32 {
        self.dims[0]
    }
}

impl<T> Stage<T, D2> {
    /// The shape of this handle as a 2-D shape. Macro support.
    #[doc(hidden)]
    #[must_use]
    pub const fn shape(&self) -> (u32, u32) {
        (self.dims[0], self.dims[1])
    }
}

/// A scalar of a batch: set per frame with `FrameRun::set`, read in a stage's body as `p.get()`; a `(u32, u32)`
/// parameter is a shape.
pub struct Param<T> {
    batch: u64,
    id: usize,
    marker: shape::Marker<T>,
}

impl<T> Clone for Param<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Param<T> {}

impl<T> core::fmt::Debug for Param<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Param")
            .field("batch", &self.batch)
            .field("id", &self.id)
            .finish()
    }
}

impl<T> Param<T> {
    pub(crate) const fn batch(&self) -> u64 {
        self.batch
    }

    pub(crate) const fn id(&self) -> usize {
        self.id
    }
}

/// What a batch holds, by name: for a test that requires a check per stage.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct StageInfo {
    /// The name of the stage's `let`.
    pub name: String,
    /// The extents of its result: width and height (1 for a 1-D stage).
    pub dims: [u32; 2],
    /// The names of the buffers it reads.
    pub reads: Vec<String>,
}

/// What a batch holds in memory: the logical bytes of its buffers.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Report {
    /// The number of stages.
    pub stages: usize,
    /// The number of buffers (inputs and results of stages).
    pub buffers: usize,
    /// The logical bytes of every buffer (`len * size of the element`), the extent records not counted.
    pub bytes: u64,
    /// The bytes of each buffer, by the name of its `let`.
    pub per_buffer: Vec<(String, u64)>,
}

pub(crate) struct BufEntry {
    pub(crate) name: String,
    pub(crate) input: bool,
    pub(crate) len: usize,
    pub(crate) dims: [u32; 2],
    pub(crate) bytes: u64,
    /// An `Arc<D::Buffer<T>>`.
    pub(crate) buffer: Box<dyn Any>,
}

pub(crate) struct ParamEntry {
    pub(crate) name: String,
    pub(crate) value: Value,
    pub(crate) shape: bool,
}

type Update<D> = Box<
    dyn Fn(
        &D,
        &mut <D as BatchDevice>::Frame,
        &Values,
    ) -> Result<(), <D as crate::job::Device>::Error>,
>;

pub(crate) struct StageRec<D: BatchDevice> {
    pub(crate) name: String,
    pub(crate) prepared: D::Prepared,
    pub(crate) update: Option<Update<D>>,
    pub(crate) params: Vec<usize>,
    pub(crate) reads: Vec<usize>,
    pub(crate) write: usize,
}

pub(crate) struct RunState<D: BatchDevice> {
    pub(crate) values: Vec<Value>,
    /// A frame is being recorded.
    pub(crate) live: bool,
    /// The most frames the batch has submitted and not yet waited for (`in_flight(n)`).
    pub(crate) depth: usize,
    /// The submitted frames that `begin` has not yet waited for, the oldest first; never more than `depth`.
    pub(crate) pending: VecDeque<D::Submission>,
    /// The device may not hold the parameter values of `values`: a frame recorded into a caller's encoder was reported
    /// or dropped, and the host cannot tell whether that encoder ran. The next frame records every parameter again.
    pub(crate) resync: bool,
}

pub(crate) struct Core<D: BatchDevice> {
    pub(crate) id: u64,
    pub(crate) name: String,
    pub(crate) device: Arc<D>,
    pub(crate) buffers: Vec<BufEntry>,
    pub(crate) stages: Vec<StageRec<D>>,
    pub(crate) params: Vec<ParamEntry>,
    pub(crate) exposed: Vec<bool>,
    pub(crate) run: Mutex<RunState<D>>,
}

impl<D: BatchDevice> Core<D> {
    /// The buffer behind a handle of this batch that the block returned, as `T`.
    pub(crate) fn output<T: GpuPod, R>(
        &self,
        handle: &Stage<T, R>,
    ) -> Result<(usize, Arc<D::Buffer<T>>), Error>
    where
        D::Buffer<T>: 'static,
    {
        if handle.batch != self.id || handle.id >= self.buffers.len() {
            return Err(Error::Foreign {
                batch: self.name.clone(),
                stage: String::from("a handle"),
            });
        }
        let entry = &self.buffers[handle.id];
        if !self.exposed[handle.id] {
            return Err(Error::NotOutput {
                batch: self.name.clone(),
                stage: entry.name.clone(),
            });
        }
        let buffer = entry
            .buffer
            .downcast_ref::<Arc<D::Buffer<T>>>()
            .expect("a handle's element type is its buffer's");
        Ok((handle.id, Arc::clone(buffer)))
    }
}

/// A sealed batch: stages, buffers and parameters that were recorded in a `gpu_batch!` block. A shared handle (cloning
/// shares the batch); frames and tickets borrow nothing from it. A `Batch`, its `FrameRun` and the future of `begin` are
/// not `Send` (only a [`Ticket`] is); drive them on the thread that built the batch.
pub struct Batch<D: BatchDevice> {
    pub(crate) core: Arc<Core<D>>,
}

impl<D: BatchDevice> Clone for Batch<D> {
    fn clone(&self) -> Self {
        Self {
            core: Arc::clone(&self.core),
        }
    }
}

impl<D: BatchDevice> core::fmt::Debug for Batch<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Batch")
            .field("name", &self.core.name)
            .field("stages", &self.core.stages.len())
            .finish_non_exhaustive()
    }
}

impl<D: BatchDevice> Batch<D> {
    /// The batch's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.core.name
    }

    /// The stages in the order they run, by the name of their `let`.
    #[must_use]
    pub fn stages(&self) -> Vec<StageInfo> {
        self.core
            .stages
            .iter()
            .map(|stage| StageInfo {
                name: stage.name.clone(),
                dims: self.core.buffers[stage.write].dims,
                reads: stage
                    .reads
                    .iter()
                    .map(|&id| self.core.buffers[id].name.clone())
                    .collect(),
            })
            .collect()
    }

    /// What the batch holds in memory.
    #[must_use]
    pub fn report(&self) -> Report {
        let per_buffer: Vec<(String, u64)> = self
            .core
            .buffers
            .iter()
            .map(|entry| (entry.name.clone(), entry.bytes))
            .collect();
        Report {
            stages: self.core.stages.len(),
            buffers: per_buffer.len(),
            bytes: per_buffer.iter().map(|(_, bytes)| bytes).sum(),
            per_buffer,
        }
    }
}

static NEXT_BATCH: AtomicU64 = AtomicU64::new(1);

std::thread_local! {
    /// Whether a build block is running on this thread.
    static BUILDING: Cell<bool> = const { Cell::new(false) };
}

struct BuildingGuard;

impl BuildingGuard {
    fn enter() -> Self {
        BUILDING.with(|building| building.set(true));
        Self
    }
}

impl Drop for BuildingGuard {
    fn drop(&mut self) {
        BUILDING.with(|building| building.set(false));
    }
}

struct BuildState<D: BatchDevice> {
    id: u64,
    name: String,
    device: Arc<D>,
    sealed: bool,
    error: Option<Error>,
    next_name: Option<String>,
    anonymous: usize,
    buffers: Vec<BufEntry>,
    stages: Vec<StageRec<D>>,
    params: Vec<ParamEntry>,
    exposed: Vec<usize>,
}

impl<D: BatchDevice> BuildState<D> {
    fn fail(&mut self, error: Error) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    fn take_name(&mut self, kind: &str) -> String {
        self.next_name.take().unwrap_or_else(|| {
            self.anonymous += 1;
            format!("{kind}#{}", self.anonymous)
        })
    }

    /// The extents a shape has now, or `None` after recording why it has none.
    fn resolve(&mut self, source: ShapeSource, who: &str) -> Option<[u32; 2]> {
        match source {
            ShapeSource::Fixed(dims) => Some(dims),
            ShapeSource::Param { batch, id } => {
                if batch != self.id || id >= self.params.len() {
                    self.fail(Error::Foreign {
                        batch: self.name.clone(),
                        stage: who.to_owned(),
                    });
                    return None;
                }
                self.params[id].shape = true;
                match self.params[id].value {
                    Value::U32(n) => Some([n, 1]),
                    Value::Pair(width, height) => Some([width, height]),
                    other => unreachable!("a shape parameter holds {other:?}"),
                }
            }
        }
    }

    fn add_buffer<T: GpuPod>(
        &mut self,
        name: String,
        input: bool,
        dims: [u32; 2],
    ) -> Option<(usize, Arc<D::Buffer<T>>)>
    where
        D::Buffer<T>: 'static,
    {
        let len = u64::from(dims[0]) * u64::from(dims[1]);
        let Ok(count) = u32::try_from(len) else {
            self.fail(Error::Device(map::Error::TooLong {
                len: usize::try_from(len).unwrap_or(usize::MAX),
            }));
            return None;
        };
        match self.device.zeroed::<T>(count) {
            Ok(buffer) => {
                let buffer = Arc::new(buffer);
                self.buffers.push(BufEntry {
                    name,
                    input,
                    len: count as usize,
                    dims,
                    bytes: len * size_of::<T>() as u64,
                    buffer: Box::new(Arc::clone(&buffer)),
                });
                Some((self.buffers.len() - 1, buffer))
            }
            Err(error) => {
                self.fail(Error::Device(D::map_error(error)));
                None
            }
        }
    }
}

/// The recorder of a build block: the `b` of `gpu_batch!(.., |b| { .. })`. A shared handle: a clone kept beyond the
/// block finds the batch sealed (see [`check`](Self::check)).
pub struct Builder<D: BatchDevice> {
    state: Rc<RefCell<BuildState<D>>>,
}

impl<D: BatchDevice> Clone for Builder<D> {
    fn clone(&self) -> Self {
        Self {
            state: Rc::clone(&self.state),
        }
    }
}

impl<D: BatchDevice> Builder<D> {
    /// A buffer the frames write: `T` elements, of the shape's extents (a `u32`, a `(u32, u32)` or a reference to a
    /// [`Param`] of one of them). Its value starts as zeros. A call on a sealed builder, or with a handle of another
    /// batch, records the error that [`check`](Self::check) and the end of the block report.
    pub fn input<T: GpuPod, S: IntoShape>(&self, shape: S) -> Stage<T, S::Rank>
    where
        D::Buffer<T>: 'static,
    {
        let mut state = self.state.borrow_mut();
        if state.sealed {
            let batch = state.name.clone();
            state.fail(Error::Sealed { batch });
            return Stage::new(state.id, usize::MAX, [0, 0]);
        }
        let name = state.take_name("input");
        let dims = state.resolve(shape.source(), &name).unwrap_or([0, 0]);
        let id = state.id;
        match state.add_buffer::<T>(name, true, dims) {
            Some((buffer, _)) => Stage::new(id, buffer, dims),
            None => Stage::new(id, usize::MAX, dims),
        }
    }

    /// A scalar with this starting value (`u32`, `i32`, `f32` or `bool`, or a `(u32, u32)` shape).
    pub fn param<T: ParamType>(&self, value: T) -> Param<T> {
        let mut state = self.state.borrow_mut();
        if state.sealed {
            let batch = state.name.clone();
            state.fail(Error::Sealed { batch });
            return Param {
                batch: state.id,
                id: usize::MAX,
                marker: core::marker::PhantomData,
            };
        }
        let name = state.take_name("param");
        state.params.push(ParamEntry {
            name,
            value: value.pack(),
            shape: false,
        });
        Param {
            batch: state.id,
            id: state.params.len() - 1,
            marker: core::marker::PhantomData,
        }
    }

    /// The first error this builder recorded, or [`Error::Sealed`] once the block has ended.
    ///
    /// # Errors
    ///
    /// As described.
    pub fn check(&self) -> Result<(), Error> {
        let state = self.state.borrow();
        if state.sealed {
            return Err(Error::Sealed {
                batch: state.name.clone(),
            });
        }
        match &state.error {
            Some(error) => Err(error.clone_shallow()),
            None => Ok(()),
        }
    }

    /// Name the next input, parameter or stage this block records. Macro support.
    #[doc(hidden)]
    pub fn next_name(&self, name: &str) {
        self.state.borrow_mut().next_name = Some(name.to_owned());
    }

    /// Mark a handle the block returns, so frames may use it. Macro support.
    #[doc(hidden)]
    pub fn expose<T, R>(&self, handle: &Stage<T, R>) {
        let mut state = self.state.borrow_mut();
        if handle.batch == state.id && handle.id < state.buffers.len() {
            state.exposed.push(handle.id);
        }
    }

    /// Begin recording a stage of this shape. Macro support.
    #[doc(hidden)]
    pub fn draft<S: IntoShape<Rank = R>, R: Rank>(&self, shape: &S) -> StageDraft<'_, D> {
        let mut state = self.state.borrow_mut();
        if state.sealed {
            let batch = state.name.clone();
            state.fail(Error::Sealed { batch });
            return StageDraft::dud(self, String::from("stage"));
        }
        let name = state.take_name("stage");
        let Some(dims) = state.resolve(shape.source(), &name) else {
            return StageDraft::dud(self, name);
        };
        StageDraft {
            builder: self,
            name,
            dims,
            reads: Vec::new(),
            params: Vec::new(),
            write: None,
            ok: true,
        }
    }
}

impl Error {
    /// An equal error for reporting twice (the device's error is reported by its message).
    fn clone_shallow(&self) -> Self {
        match self {
            Self::Sealed { batch } => Self::Sealed {
                batch: batch.clone(),
            },
            Self::NestedBuild { batch } => Self::NestedBuild {
                batch: batch.clone(),
            },
            Self::Foreign { batch, stage } => Self::Foreign {
                batch: batch.clone(),
                stage: stage.clone(),
            },
            Self::Hazard {
                batch,
                stage,
                buffer,
            } => Self::Hazard {
                batch: batch.clone(),
                stage: stage.clone(),
                buffer: buffer.clone(),
            },
            Self::ShapeMismatch {
                batch,
                stage,
                expected,
                found,
            } => Self::ShapeMismatch {
                batch: batch.clone(),
                stage: stage.clone(),
                expected: expected.clone(),
                found: found.clone(),
            },
            Self::NotSubmitted { batch } => Self::NotSubmitted {
                batch: batch.clone(),
            },
            Self::NotOutput { batch, stage } => Self::NotOutput {
                batch: batch.clone(),
                stage: stage.clone(),
            },
            Self::NotInput { batch, stage } => Self::NotInput {
                batch: batch.clone(),
                stage: stage.clone(),
            },
            Self::FrameLive { batch } => Self::FrameLive {
                batch: batch.clone(),
            },
            Self::Device(error) => {
                Self::Device(map::Error::device(DeviceMessage(error.to_string())))
            }
        }
    }
}

#[derive(Debug)]
struct DeviceMessage(String);

impl core::fmt::Display for DeviceMessage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::error::Error for DeviceMessage {}

/// A stage being recorded: its buffers, parameters and kernel call, until `finish` seals it into the batch. Macro
/// support.
#[doc(hidden)]
pub struct StageDraft<'b, D: BatchDevice> {
    builder: &'b Builder<D>,
    name: String,
    dims: [u32; 2],
    reads: Vec<usize>,
    params: Vec<usize>,
    write: Option<usize>,
    ok: bool,
}

impl<'b, D: BatchDevice> StageDraft<'b, D> {
    fn dud(builder: &'b Builder<D>, name: String) -> Self {
        Self {
            builder,
            name,
            dims: [0, 0],
            reads: Vec::new(),
            params: Vec::new(),
            write: None,
            ok: false,
        }
    }

    /// The extents of the stage.
    #[must_use]
    pub const fn dims(&self) -> [u32; 2] {
        self.dims
    }

    /// Whether every step so far worked (an error is recorded in the builder otherwise).
    #[must_use]
    pub const fn ok(&self) -> bool {
        self.ok
    }

    /// The device.
    #[must_use]
    pub fn device(&self) -> Arc<D> {
        Arc::clone(&self.builder.state.borrow().device)
    }

    /// The parameters' values now, for the extent record's first contents.
    #[must_use]
    pub fn values(&self) -> Values {
        Values(
            self.builder
                .state
                .borrow()
                .params
                .iter()
                .map(|entry| entry.value)
                .collect(),
        )
    }

    fn foreign(&mut self) {
        let mut state = self.builder.state.borrow_mut();
        let (batch, stage) = (state.name.clone(), self.name.clone());
        state.fail(Error::Foreign { batch, stage });
        self.ok = false;
    }

    /// The buffer behind a handle this stage reads.
    pub fn view<T: GpuPod, R>(&mut self, handle: &Stage<T, R>) -> Option<Arc<D::Buffer<T>>>
    where
        D::Buffer<T>: 'static,
    {
        let found = {
            let state = self.builder.state.borrow();
            (handle.batch == state.id && handle.id < state.buffers.len()).then(|| {
                state.buffers[handle.id]
                    .buffer
                    .downcast_ref::<Arc<D::Buffer<T>>>()
                    .cloned()
            })
        };
        match found.flatten() {
            Some(buffer) => {
                self.reads.push(handle.id);
                Some(buffer)
            }
            None => {
                self.foreign();
                None
            }
        }
    }

    /// The buffer an in-place stage writes: the handle's own, which the stage does not read as a view.
    pub fn target<T: GpuPod>(&mut self, handle: &Stage<T, D1>) -> Option<Arc<D::Buffer<T>>>
    where
        D::Buffer<T>: 'static,
    {
        let before = self.reads.len();
        let buffer = self.view(handle)?;
        self.reads.truncate(before);
        self.write = Some(handle.id);
        Some(buffer)
    }

    /// The number of the parameter, after checking that it belongs to this batch.
    pub fn param<T: ParamType>(&mut self, param: &Param<T>) -> usize {
        let valid = {
            let state = self.builder.state.borrow();
            param.batch == state.id && param.id < state.params.len()
        };
        if valid {
            self.params.push(param.id);
        } else {
            self.foreign();
        }
        param.id
    }

    /// The stage's result buffer: new, zeroed, of the stage's extents.
    pub fn output<T: GpuPod>(&mut self) -> Option<Arc<D::Buffer<T>>>
    where
        D::Buffer<T>: 'static,
    {
        let mut state = self.builder.state.borrow_mut();
        let name = self.name.clone();
        let Some((id, buffer)) = state.add_buffer::<T>(name, false, self.dims) else {
            self.ok = false;
            return None;
        };
        self.write = Some(id);
        Some(buffer)
    }

    /// The stage's handle when it could not be recorded (an error is in the builder): a handle no frame accepts.
    #[must_use]
    pub fn abandon<T, Rk>(self) -> Stage<T, Rk> {
        let state = self.builder.state.borrow();
        Stage::new(state.id, usize::MAX, self.dims)
    }

    /// Record an error of the map layer (a limit of the device, a uniform that would not be made).
    pub fn fail_map(&mut self, error: map::Error) {
        self.builder.state.borrow_mut().fail(Error::Device(error));
        self.ok = false;
    }

    /// Record a device error of this stage's preparation.
    pub fn fail(&mut self, error: D::Error) {
        self.builder
            .state
            .borrow_mut()
            .fail(Error::Device(D::map_error(error)));
        self.ok = false;
    }

    /// Seal the stage: the kernel call `build` makes over `keep`, and `update`, which rewrites the stage's extent
    /// record from the parameters' values when one of them was set. Returns the handle of its result.
    pub fn finish<T, Rk, K, R, const N: usize>(
        self,
        keep: R,
        build: impl for<'o> Fn(&'o R) -> Call<'o, D, K, N> + 'static,
        update: impl Fn(&D, &mut D::Frame, &Values) -> Result<(), D::Error> + 'static,
    ) -> Stage<T, Rk>
    where
        D: Run<K, N>,
        K: Kernel,
        R: 'static,
    {
        let Self {
            builder,
            name,
            dims,
            reads,
            params,
            write,
            ok,
        } = self;
        let mut state = builder.state.borrow_mut();
        let dud = Stage::new(state.id, usize::MAX, dims);
        let (true, Some(write)) = (ok, write) else {
            return dud;
        };
        match state.device.prepare(keep, build) {
            Ok(prepared) => {
                let rec = StageRec {
                    name,
                    prepared,
                    update: Some(Box::new(update)),
                    params,
                    reads,
                    write,
                };
                state.stages.push(rec);
                Stage::new(state.id, write, dims)
            }
            Err(error) => {
                state.fail(Error::Device(D::map_error(error)));
                dud
            }
        }
    }
}

/// The value of a `gpu_batch!` call that did not compile: it names the device type, so that the code after the call
/// type-checks and one mistake stays one error. Never runs. Macro support.
#[doc(hidden)]
pub fn failed<D: BatchDevice, H: DeviceRef<Device = D>, O>(
    _device: &H,
) -> Result<(Batch<D>, O), Error> {
    unreachable!("a `gpu_batch!` that failed to compile is never run")
}

/// The batch of a `gpu_batch!`: run `block` as the build scope and seal what it recorded. Macro support; the macro
/// is the form to write. `in_flight` is the `in_flight(n)` of the macro (1 to 4, checked there); the function clamps
/// 0 to 1.
///
/// # Errors
///
/// [`Error::NestedBuild`] when a build is already running on this thread; the first error the block recorded; and
/// [`Error::Hazard`] for a stage that binds one buffer read-only and read-write.
#[doc(hidden)]
pub fn build<D: BatchDevice, H: DeviceRef<Device = D>, O>(
    name: &str,
    device: &H,
    in_flight: u32,
    block: impl FnOnce(&Builder<D>) -> O,
) -> Result<(Batch<D>, O), Error> {
    if BUILDING.with(Cell::get) {
        return Err(Error::NestedBuild {
            batch: name.to_owned(),
        });
    }
    let builder = Builder {
        state: Rc::new(RefCell::new(BuildState {
            id: NEXT_BATCH.fetch_add(1, Ordering::Relaxed),
            name: name.to_owned(),
            device: device.handle(),
            sealed: false,
            error: None,
            next_name: None,
            anonymous: 0,
            buffers: Vec::new(),
            stages: Vec::new(),
            params: Vec::new(),
            exposed: Vec::new(),
        })),
    };
    let output = {
        let _guard = BuildingGuard::enter();
        block(&builder)
    };
    let mut state = builder.state.borrow_mut();
    state.sealed = true;
    if let Some(error) = state.error.take() {
        return Err(error);
    }
    for stage in &state.stages {
        if stage.reads.contains(&stage.write) {
            return Err(Error::Hazard {
                batch: state.name.clone(),
                stage: stage.name.clone(),
                buffer: state.buffers[stage.write].name.clone(),
            });
        }
    }
    let mut exposed = vec![false; state.buffers.len()];
    for &id in &state.exposed {
        exposed[id] = true;
    }
    let params = core::mem::take(&mut state.params);
    let values = params.iter().map(|entry| entry.value).collect();
    let core = Core {
        id: state.id,
        name: state.name.clone(),
        device: Arc::clone(&state.device),
        buffers: core::mem::take(&mut state.buffers),
        stages: core::mem::take(&mut state.stages),
        params,
        exposed,
        run: Mutex::new(RunState {
            values,
            live: false,
            depth: usize::try_from(in_flight.max(1)).unwrap_or(1),
            pending: VecDeque::new(),
            resync: false,
        }),
    };
    // The cell of `core` is not `Sync` (the CPU device's buffers are single-threaded); the batch is a shared handle for
    // its frames and tickets, which run on the thread that owns the device.
    #[allow(clippy::arc_with_non_send_sync)]
    let core = Arc::new(core);
    Ok((Batch { core }, output))
}
