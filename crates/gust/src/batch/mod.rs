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
//! devices. A frame recorded into a caller's encoder follows [`Encoded`]: a reported frame's `set`s hold, a dropped
//! frame's are discarded. A frame records the extent record of a stage only when a parameter of the stage was set in
//! it; after a frame recorded into a caller's encoder, the parameters that frame set stay on the list of the ones a
//! later frame records again, until a frame made by `submit` clears it. A frame's tickets read that frame's results; a
//! later frame's writes never reach an earlier frame.
//!
//! **Resize (slice s2b).** A parameter that sizes an input or a stage can be `set` to a new shape between frames, and
//! what depends on it follows: [`FrameRun::set`] makes new, zeroed buffers (an input holds zeros until the frame writes
//! it: nothing is carried over), prepares the stages that bind them again (their workgroup counts follow the new
//! extents) and, when the frame is submitted or encoded, moves the batch's [`generation`](Batch::generation). A shape
//! can also be a [`Derived`] value (`p.map(..)`, `p.count()`), which the CPU recomputes at each `set` of its parameter:
//! no stage or GPU pass counts anything. A frame `set`s the shape first and `write`s after: a `set` that would change a
//! buffer the frame has already written or read is [`Error::ShapeMismatch`]. A library op stage follows a resize too
//! (slice s2b2): the op's plan for the new length is built inside the `set`, and the outputs that follow the operand
//! (those of `scan` and `compact`) are replaced and move the generation. The batch keeps the plans of the last two
//! lengths of each op stage, so a `set` back to a kept length builds no plan; a new plan is kept only with a frame the
//! device took. An op stage's length is that of its first operand (a resize that changes only a later operand is a
//! device error from the `set`, and the frame is unchanged). The allocation rule: a `set` that resizes may take
//! submissions of its own (the zeroed new buffers and the scratch of a new plan are cleared inside the call), a `set`
//! that finds a kept plan takes one per replaced buffer and no more, and the frame that follows is exactly one
//! submission, like any frame (D251).
//!
//! A caller reads a batch's buffers on the GPU through a [`View`] ([`Batch::view`], decision D261): the batch's own
//! buffer, kept alive by the view, with the batch's [`generation`](Batch::generation). What a reader sees follows the
//! queue order of D258: a pass recorded into the same encoder after [`FrameRun::encode`] returned reads that frame, a
//! pass recorded before `encode` reads the frame before it, and a pass in a later submission reads the last frame
//! submitted before it; a view carries no check that a frame ran. Bind a view as read-only storage (a documented rule,
//! not a check). A zero-length buffer keeps one placeholder element, as GUST's own bindings of an empty buffer do, so a
//! reader uses [`View::len`] and not the shader's `arrayLength`.

use core::any::Any;
use core::cell::{Cell, RefCell};
use core::future::Future;
use core::pin::Pin;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::{
    Arc, Mutex, PoisonError,
    atomic::{AtomicU64, Ordering},
};

use crate::map::{self, MapHost};
use crate::{Call, GpuPod, Kernel, Resource, Run, kind};

mod counted;
mod element;
mod frame;
mod ops;
mod shape;
mod slot;
mod view;

#[doc(hidden)]
pub use counted::{CountedDevice, GROUP_LIMIT, grid_of, set_counted_launch};
pub use counted::{CountedLaunch, counted_launch};
pub use element::{Element, InvalidWord};
#[doc(hidden)]
pub use element::{pack_enum, unpack_enum};
#[cfg(not(target_arch = "wasm32"))]
pub use frame::Blocking;
pub use frame::{Encoded, FrameRun, Submitted, Ticket};
pub use ops::{Empty, Op, OpDraft, Operands, Outputs};
use ops::{OpPlan, PLANS_KEPT};
use shape::Rule;
pub use shape::{D1, D2, Derived, IntoShape, ParamType, Rank};
#[doc(hidden)]
pub use shape::{Derive, Layout, ShapeSource, Value, Values};
pub use view::View;

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
    /// A later stage writes a buffer that a library op made: the op's outputs are its alone to write.
    Overwrite {
        /// The batch.
        batch: String,
        /// The later stage that writes the buffer.
        stage: String,
        /// The stage of the op that made it.
        op: String,
        /// The buffer, by the name of its `let`.
        buffer: String,
    },
    /// A library op that has no value over zero elements (`min`, `max`) ran over none: the frame was refused before
    /// anything was encoded.
    Empty {
        /// The batch.
        batch: String,
        /// The stage of the op.
        stage: String,
    },
    /// A data call does not fit the shape the frame has: a `write` of another length, or a `set` that would change the
    /// shape of a buffer the frame has already written or read.
    ShapeMismatch {
        /// The batch.
        batch: String,
        /// The input or parameter.
        stage: String,
        /// What the batch (or the frame) has.
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
    /// A read-back of an enum handle held a word that is no variant of the enum (D275): a ticket or a `read` reports
    /// the lowest such index, row-major for a 2-D handle, and the word. The ticket is taken; the frame's other tickets
    /// resolve as they would, and the batch stays usable.
    InvalidEnum {
        /// The batch.
        batch: String,
        /// The stage or input the words were read from.
        stage: String,
        /// The index of the first word that is no variant.
        index: usize,
        /// That word.
        word: u32,
    },
    /// The count of a counted stage is not one element: it names a buffer of another length.
    NotCount {
        /// The batch.
        batch: String,
        /// The counted stage.
        stage: String,
        /// The buffer named as its count, by the name of its `let`.
        buffer: String,
        /// The number of elements it has.
        len: usize,
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
            Self::Overwrite {
                batch,
                stage,
                op,
                buffer,
            } => write!(
                f,
                "stage `{stage}` of batch `{batch}` writes `{buffer}`, which is an output of the op stage `{op}`"
            ),
            Self::Empty { batch, stage } => write!(
                f,
                "stage `{stage}` of batch `{batch}` is an operation over zero elements and has no value there"
            ),
            Self::ShapeMismatch {
                batch,
                stage,
                expected,
                found,
            } => write!(
                f,
                "`{stage}` of batch `{batch}` has {expected} and the call brought {found}"
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
            Self::InvalidEnum {
                batch,
                stage,
                index,
                word,
            } => write!(
                f,
                "`{stage}` of batch `{batch}` holds the word {word} at index {index}, which is no variant of its enum"
            ),
            Self::NotCount {
                batch,
                stage,
                buffer,
                len,
            } => write!(
                f,
                "stage `{stage}` of batch `{batch}` is counted by `{buffer}`, which has {len} elements: a count is one `u32`"
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
/// `T` is the element, `R` the rank ([`D1`] or [`D2`]); a 2-D handle is row-major. The extents a handle reports
/// ([`width`](Self::width), [`height`](Self::height), [`len`](Self::len)) are the ones the batch was sealed with: a
/// resize does not change a handle, so read the current extents from a [`View`].
pub struct Stage<T, R> {
    batch: u64,
    id: usize,
    dims: [u32; 2],
    /// Made by `words()`, the handle a kernel binds: it is no handle to write (D275, no public raw write of an enum).
    raw: bool,
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
    const fn new(batch: u64, id: usize, dims: [u32; 2]) -> Self {
        Self {
            batch,
            id,
            dims,
            raw: false,
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

    /// The number of the buffer in its batch. Macro support.
    #[doc(hidden)]
    #[must_use]
    pub const fn index(&self) -> usize {
        self.id
    }
}

impl<T: Element, R> Stage<T, R> {
    /// The same buffer as the handle of its words (the `u32` words of an enum element), which is what a kernel binds.
    /// It is no handle to write: `FrameRun::write` refuses it (D275). Macro support.
    #[doc(hidden)]
    #[must_use]
    pub const fn words(&self) -> Stage<T::Word, R> {
        let mut handle = Stage::new(self.batch, self.id, self.dims);
        handle.raw = true;
        handle
    }

    /// Whether this is a kernel's handle of an enum's words (made by `words()`).
    pub(crate) const fn is_raw(&self) -> bool {
        self.raw
    }
}

impl<R> Stage<u32, R> {
    /// The handle of a stage's result as the enum element the stage's body wrote as words. Macro support: a stage's
    /// result handle is the one it is meant for. A handle made by `words()` stays raw, so `write` refuses it after a
    /// `retype` too; a plain `Stage<u32>` (an input of words) is writable with any word, which is why this is not API.
    #[doc(hidden)]
    #[must_use]
    pub const fn retype<E: Element<Word = u32>>(self) -> Stage<E, R> {
        let mut handle = Stage::new(self.batch, self.id, self.dims);
        handle.raw = self.raw;
        handle
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

/// A located output: its buffer id, the buffer and its extents.
type Located<D, T> = (usize, Arc<<D as crate::map::MapHost>::Buffer<T>>, [u32; 2]);

/// The function that makes a new, zeroed buffer of one element type, for a resize.
type Alloc<D> = Box<dyn Fn(&D, u32) -> Result<Arc<dyn Any>, <D as crate::job::Device>::Error>>;

/// A buffer while the batch is being built. At the seal it splits into a [`BufMeta`] and the live state of the run.
pub(crate) struct BufEntry {
    pub(crate) name: String,
    pub(crate) input: bool,
    pub(crate) len: usize,
    pub(crate) dims: [u32; 2],
    /// The bytes of one element.
    pub(crate) stride: u64,
    pub(crate) rule: Rule,
    /// An `Arc<D::Buffer<T>>`, erased.
    pub(crate) buffer: Arc<dyn Any>,
}

/// What a sealed buffer keeps for good: the live buffer and its extents are in [`RunState`].
pub(crate) struct BufMeta<D: BatchDevice> {
    pub(crate) name: Arc<str>,
    pub(crate) input: bool,
    pub(crate) stride: u64,
    pub(crate) rule: Rule,
    pub(crate) alloc: Alloc<D>,
}

pub(crate) struct ParamEntry {
    pub(crate) name: String,
    pub(crate) value: Value,
    /// Some buffer takes its extents from this parameter, or from a value derived from it.
    pub(crate) shape: bool,
    /// For a derived value: the parameter it comes from and the function.
    pub(crate) derived: Option<(usize, Derive)>,
}

type Update<D> = Box<
    dyn Fn(
        &D,
        &mut <D as BatchDevice>::Frame,
        &Values,
    ) -> Result<(), <D as crate::job::Device>::Error>,
>;

/// The function that prepares a kernel stage again over the buffers of a resized frame.
pub(crate) type Remake<D> =
    Box<dyn Fn(&D, &Layout<'_>) -> Result<<D as BatchDevice>::Prepared, Error>>;

pub(crate) struct StageRec<D: BatchDevice> {
    pub(crate) name: String,
    /// What the stage dispatches as it was sealed: `None` for an op sealed over zero elements, where nothing is
    /// prepared and the stage's `empty` rule runs instead. The seal moves it into the run's list of prepared stages,
    /// which a resize replaces.
    pub(crate) prepared: Option<D::Prepared>,
    pub(crate) update: Option<Update<D>>,
    /// The parameters the stage depends on: the ones its body reads, and (added at the seal) the ones that give the
    /// extents of its buffers.
    pub(crate) params: Vec<usize>,
    pub(crate) reads: Vec<usize>,
    /// The buffer the stage writes (an op's first output).
    pub(crate) write: usize,
    /// The other buffers an op writes, in order.
    pub(crate) also: Vec<usize>,
    /// How to prepare a kernel stage again over resized buffers; `None` for an op stage (see `plan`).
    pub(crate) remake: Option<Remake<D>>,
    /// An op's functions that prepare it again at another length; `Some` marks an op's stage, `None` a kernel stage.
    pub(crate) plan: Option<OpPlan<D>>,
    /// The plan the op was sealed with and its length, which the seal moves into the run's list of kept plans.
    pub(crate) first: Option<(usize, Arc<dyn Any>)>,
    /// An op's rule for a length of zero (`None` for a kernel stage).
    pub(crate) empty: Option<Empty<D>>,
}

impl<D: BatchDevice> StageRec<D> {
    /// A kernel stage.
    fn kernel(
        name: String,
        prepared: D::Prepared,
        update: Update<D>,
        params: Vec<usize>,
        reads: Vec<usize>,
        write: usize,
    ) -> Self {
        Self {
            name,
            prepared: Some(prepared),
            update: Some(update),
            params,
            reads,
            write,
            also: Vec::new(),
            remake: None,
            plan: None,
            first: None,
            empty: None,
        }
    }

    /// The stage can be prepared again over resized buffers.
    fn with_remake(mut self, remake: Remake<D>) -> Self {
        self.remake = Some(remake);
        self
    }

    /// An op's stage: `prepared` and `first` are `None` at length 0.
    pub(crate) fn op(
        name: String,
        prepared: Option<D::Prepared>,
        reads: Vec<usize>,
        written: &[(usize, [u32; 2])],
        plan: OpPlan<D>,
        first: Option<(usize, Arc<dyn Any>)>,
        empty: Empty<D>,
    ) -> Self {
        Self {
            name,
            prepared,
            update: None,
            params: Vec::new(),
            reads,
            write: written[0].0,
            also: written[1..].iter().map(|(id, _)| *id).collect(),
            remake: None,
            plan: Some(plan),
            first,
            empty: Some(empty),
        }
    }

    /// The number of elements the stage runs over in a frame whose buffers have these extents (one entry per buffer).
    /// The one place the core reads a stage's length: a kernel stage runs over the buffer it writes, an op stage over its
    /// first operand.
    pub(crate) fn len(&self, dims: &[[u32; 2]]) -> usize {
        let [width, height] = dims[if self.plan.is_some() {
            self.reads[0]
        } else {
            self.write
        }];
        usize::try_from(u64::from(width) * u64::from(height)).unwrap_or(usize::MAX)
    }

    /// Every buffer the stage writes: `write`, then `also`.
    pub(crate) fn written(&self) -> impl Iterator<Item = usize> + '_ {
        core::iter::once(self.write).chain(self.also.iter().copied())
    }
}

pub(crate) struct RunState<D: BatchDevice> {
    /// The parameters' values (a derived value is one of them).
    pub(crate) values: Vec<Value>,
    /// The extents of each buffer now.
    pub(crate) dims: Vec<[u32; 2]>,
    /// Each buffer now: an `Arc<D::Buffer<T>>`, erased.
    pub(crate) buffers: Vec<Arc<dyn Any>>,
    /// What each stage dispatches now (`None` for an op sealed over zero elements).
    pub(crate) prepared: Vec<Option<Arc<D::Prepared>>>,
    /// The plans each op stage keeps, the plan of its last non-zero length first (a resize to zero leaves them as
    /// they were) and then the lengths before it (at most `PLANS_KEPT`): a stage that is not an op has none. An erased `Arc<P>` of the op's plan type.
    pub(crate) plans: Vec<VecDeque<(usize, Arc<dyn Any>)>>,
    /// A frame is being recorded.
    pub(crate) live: bool,
    /// The most frames the batch has submitted and not yet waited for (`in_flight(n)`).
    pub(crate) depth: usize,
    /// The submitted frames that `begin` has not yet waited for, the oldest first; never more than `depth`.
    pub(crate) pending: VecDeque<D::Submission>,
    /// The parameters whose values the device may not hold: the ones set in a frame recorded into a caller's encoder
    /// that was reported or dropped (the host cannot tell whether that encoder ran), and the ones that were stale when
    /// it was recorded. A frame records the extent record of every stage that has one of them, and a frame that the
    /// batch submitted itself clears the set.
    pub(crate) stale: Vec<usize>,
    /// How many extent records the frames of the batch have recorded (see [`Batch::extent_records`]).
    pub(crate) extent_records: u64,
}

impl<D: BatchDevice> RunState<D> {
    /// Remember that the device may not hold these parameters' values.
    pub(crate) fn mark_stale(&mut self, ids: impl Iterator<Item = usize>) {
        for id in ids {
            if !self.stale.contains(&id) {
                self.stale.push(id);
            }
        }
    }
}

pub(crate) struct Core<D: BatchDevice> {
    pub(crate) id: u64,
    pub(crate) name: Arc<str>,
    pub(crate) device: Arc<D>,
    pub(crate) buffers: Vec<BufMeta<D>>,
    pub(crate) stages: Vec<StageRec<D>>,
    pub(crate) params: Vec<ParamEntry>,
    pub(crate) exposed: Vec<bool>,
    /// One counter per batch: moves when a buffer behind a handle is replaced (see [`Batch::generation`]).
    pub(crate) generation: AtomicU64,
    pub(crate) run: Mutex<RunState<D>>,
}

/// The buffer `any` holds, as `T`.
pub(crate) fn typed<D: BatchDevice, T: GpuPod>(any: &Arc<dyn Any>) -> Arc<D::Buffer<T>>
where
    D::Buffer<T>: 'static,
{
    Arc::clone(
        any.downcast_ref::<Arc<D::Buffer<T>>>()
            .expect("a handle's element type is its buffer's"),
    )
}

impl<D: BatchDevice> Core<D> {
    /// The number of the buffer behind a handle of this batch that the block returned.
    pub(crate) fn locate<T, R>(&self, handle: &Stage<T, R>) -> Result<usize, Error> {
        if handle.batch != self.id || handle.id >= self.buffers.len() {
            return Err(Error::Foreign {
                batch: self.name.to_string(),
                stage: String::from("a handle"),
            });
        }
        if !self.exposed[handle.id] {
            return Err(Error::NotOutput {
                batch: self.name.to_string(),
                stage: self.buffers[handle.id].name.to_string(),
            });
        }
        Ok(handle.id)
    }

    /// The buffer behind a handle of this batch that the block returned, as `T`, and its extents, as the last frame
    /// submitted or encoded left them.
    pub(crate) fn output<T: Element, R>(
        &self,
        handle: &Stage<T, R>,
    ) -> Result<Located<D, T::Word>, Error>
    where
        D::Buffer<T::Word>: 'static,
    {
        let id = self.locate(handle)?;
        let run = self.run.lock().unwrap_or_else(PoisonError::into_inner);
        Ok((id, typed::<D, T::Word>(&run.buffers[id]), run.dims[id]))
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

    /// The stages in the order they run, by the name of their `let`: one entry per buffer a stage writes (an op with
    /// several outputs, such as `compact`, lists each by its own name, every one with the op's reads). The extents are
    /// the ones the last frame submitted or encoded left (a resize changes them).
    #[must_use]
    pub fn stages(&self) -> Vec<StageInfo> {
        let buffers = &self.core.buffers;
        let dims = self
            .core
            .run
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .dims
            .clone();
        self.core
            .stages
            .iter()
            .flat_map(|stage| {
                let reads: Vec<String> = stage
                    .reads
                    .iter()
                    .map(|&id| buffers[id].name.to_string())
                    .collect();
                let dims = &dims;
                stage.written().enumerate().map(move |(k, id)| StageInfo {
                    name: if k == 0 {
                        stage.name.clone()
                    } else {
                        buffers[id].name.to_string()
                    },
                    dims: dims[id],
                    reads: reads.clone(),
                })
            })
            .collect()
    }

    /// Test support, outside the pre-1.0 promise (D111): how many extent records (the per-stage records that carry the
    /// parameters' values) the batch's frames have recorded so far. A frame records the record of each stage that has
    /// a parameter set in that frame or left stale by a frame recorded into a caller's encoder.
    #[doc(hidden)]
    #[must_use]
    pub fn extent_records(&self) -> u64 {
        self.core
            .run
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extent_records
    }

    /// What the batch holds in memory, at the extents the last frame submitted or encoded left.
    #[must_use]
    pub fn report(&self) -> Report {
        let run = self.core.run.lock().unwrap_or_else(PoisonError::into_inner);
        let per_buffer: Vec<(String, u64)> = self
            .core
            .buffers
            .iter()
            .zip(&run.dims)
            .map(|(entry, dims)| {
                (
                    entry.name.to_string(),
                    u64::from(dims[0]) * u64::from(dims[1]) * entry.stride,
                )
            })
            .collect();
        drop(run);
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
    /// The allocator of each buffer, in the order of `buffers`.
    allocs: Vec<Alloc<D>>,
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

    /// The extents a shape has now and the rule that keeps them current, or `None` after recording why it has none.
    fn resolve(&mut self, source: ShapeSource, who: &str) -> Option<([u32; 2], Rule)> {
        let id = match source {
            ShapeSource::Fixed(dims) => return Some((dims, Rule::Fixed(dims))),
            ShapeSource::Param { batch, id } if batch == self.id && id < self.params.len() => id,
            ShapeSource::Derived {
                batch,
                source,
                derive,
            } if batch == self.id && source < self.params.len() => self.derived(source, &derive),
            ShapeSource::Buffer { batch, id } if batch == self.id && id < self.buffers.len() => {
                let entry = &self.buffers[id];
                return Some((entry.dims, entry.rule));
            }
            ShapeSource::Param { .. }
            | ShapeSource::Derived { .. }
            | ShapeSource::Buffer { .. } => {
                self.fail(Error::Foreign {
                    batch: self.name.clone(),
                    stage: who.to_owned(),
                });
                return None;
            }
        };
        self.params[id].shape = true;
        let rule = Rule::Param(id);
        Some((rule.dims(&self.values()), rule))
    }

    /// The values of the parameters now.
    fn values(&self) -> Vec<Value> {
        self.params.iter().map(|entry| entry.value).collect()
    }

    /// The number of the parameter that holds `derive` of the parameter `source`: the one made before when this value was
    /// used already, a new one otherwise.
    fn derived(&mut self, source: usize, derive: &Derive) -> usize {
        let known = self.params.iter().position(|entry| {
            entry
                .derived
                .as_ref()
                .is_some_and(|(from, known)| *from == source && known.same(derive))
        });
        if let Some(id) = known {
            return id;
        }
        let name = format!("{} (derived)", self.params[source].name);
        self.params[source].shape = true;
        self.params.push(ParamEntry {
            name,
            value: derive.apply(self.params[source].value),
            shape: false,
            derived: Some((source, derive.clone())),
        });
        self.params.len() - 1
    }

    /// A buffer of these extents, whose extents follow `rule` after a resize.
    fn add_buffer_ruled<T: GpuPod>(
        &mut self,
        name: String,
        input: bool,
        dims: [u32; 2],
        rule: Rule,
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
                    stride: size_of::<T>() as u64,
                    rule,
                    buffer: Arc::new(Arc::clone(&buffer)),
                });
                self.allocs.push(Box::new(|device, count| {
                    device
                        .zeroed::<T>(count)
                        .map(|buffer| Arc::new(Arc::new(buffer)) as Arc<dyn Any>)
                }));
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
    /// [`Param`] of one of them). `T` is a [`GpuPod`] type or an enum deriving `GpuEnum` (an [`Element`]: the device holds
    /// an enum as one `u32` word, and it needs a variant whose word is 0). Its value starts as zeros. A call on a sealed
    /// builder, or with a handle of another batch, records the error that [`check`](Self::check) and the end of the block
    /// report.
    pub fn input<T: Element, S: IntoShape>(&self, shape: S) -> Stage<T, S::Rank>
    where
        D::Buffer<T::Word>: 'static,
    {
        let mut state = self.state.borrow_mut();
        if state.sealed {
            let batch = state.name.clone();
            state.fail(Error::Sealed { batch });
            return Stage::new(state.id, usize::MAX, [0, 0]);
        }
        let name = state.take_name("input");
        let (dims, rule) = state
            .resolve(shape.source(), &name)
            .unwrap_or(([0, 0], Rule::Fixed([0, 0])));
        let id = state.id;
        match state.add_buffer_ruled::<T::Word>(name, true, dims, rule) {
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
            derived: None,
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
        let Some((dims, rule)) = state.resolve(shape.source(), &name) else {
            return StageDraft::dud(self, name);
        };
        StageDraft {
            builder: self,
            name,
            dims,
            rule,
            reads: Vec::new(),
            bound: Vec::new(),
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
            Self::Overwrite {
                batch,
                stage,
                op,
                buffer,
            } => Self::Overwrite {
                batch: batch.clone(),
                stage: stage.clone(),
                op: op.clone(),
                buffer: buffer.clone(),
            },
            Self::Empty { batch, stage } => Self::Empty {
                batch: batch.clone(),
                stage: stage.clone(),
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
            Self::InvalidEnum {
                batch,
                stage,
                index,
                word,
            } => Self::InvalidEnum {
                batch: batch.clone(),
                stage: stage.clone(),
                index: *index,
                word: *word,
            },
            Self::NotCount {
                batch,
                stage,
                buffer,
                len,
            } => Self::NotCount {
                batch: batch.clone(),
                stage: stage.clone(),
                buffer: buffer.clone(),
                len: *len,
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
    /// Where the extents come from after a resize.
    rule: Rule,
    reads: Vec<usize>,
    /// Every buffer the stage binds (read, written in place or made), in the order the stage took them.
    bound: Vec<usize>,
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
            rule: Rule::Fixed([0, 0]),
            reads: Vec::new(),
            bound: Vec::new(),
            params: Vec::new(),
            write: None,
            ok: false,
        }
    }

    /// The numbers of the buffers the stage binds, in the order it took them. Macro support.
    #[must_use]
    pub fn bound(&self) -> Vec<usize> {
        self.bound.clone()
    }

    /// The number of the buffer the stage writes: the stage's extents are its extents. Call it after the stage has
    /// taken its buffers and `ok` is true. Macro support.
    ///
    /// # Panics
    ///
    /// When the stage has no result buffer yet.
    #[must_use]
    pub fn written(&self) -> usize {
        self.write.expect("a stage takes its result buffer first")
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

    /// The parameters' values and the buffers' extents now, for the extent record's first contents.
    #[must_use]
    pub fn values(&self) -> Values {
        let state = self.builder.state.borrow();
        Values {
            params: state.values(),
            dims: state.buffers.iter().map(|entry| entry.dims).collect(),
        }
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
                self.bound.push(handle.id);
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
        let Some((id, buffer)) = state.add_buffer_ruled::<T>(name, false, self.dims, self.rule)
        else {
            self.ok = false;
            return None;
        };
        self.write = Some(id);
        self.bound.push(id);
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

    /// Seal the stage: the kernel call `build` makes over `keep`, `update`, which rewrites the stage's extent record from
    /// the parameters' values and the buffers' extents when one of them changed, and `keep_with`, which makes the
    /// stage's `keep` again over the buffers of a resized frame (the stage is then prepared again with the same `build`,
    /// so `build` holds nothing of its own). Returns the handle of its result.
    pub fn finish<T, Rk, K, R, const N: usize>(
        self,
        keep: R,
        build: impl for<'o> Fn(&'o R) -> Call<'o, D, K, N> + Copy + 'static,
        update: impl Fn(&D, &mut D::Frame, &Values) -> Result<(), D::Error> + 'static,
        keep_with: impl Fn(&Layout<'_>) -> Result<R, Error> + 'static,
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
            ..
        } = self;
        let mut state = builder.state.borrow_mut();
        let dud = Stage::new(state.id, usize::MAX, dims);
        let (true, Some(write)) = (ok, write) else {
            return dud;
        };
        match state.device.prepare(keep, build) {
            Ok(prepared) => {
                let remake: Remake<D> = Box::new(move |device, layout| {
                    device
                        .prepare(keep_with(layout)?, build)
                        .map_err(|error| Error::Device(D::map_error(error)))
                });
                let rec = StageRec::kernel(name, prepared, Box::new(update), params, reads, write)
                    .with_remake(remake);
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
            allocs: Vec::new(),
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
    // The stage of the op that made each buffer: nothing later writes it.
    let mut made_by: Vec<Option<usize>> = vec![None; state.buffers.len()];
    for (index, stage) in state.stages.iter().enumerate() {
        for id in stage.written() {
            if stage.reads.contains(&id) {
                return Err(Error::Hazard {
                    batch: state.name.clone(),
                    stage: stage.name.clone(),
                    buffer: state.buffers[id].name.clone(),
                });
            }
            if let Some(op) = made_by[id] {
                return Err(Error::Overwrite {
                    batch: state.name.clone(),
                    stage: stage.name.clone(),
                    op: state.stages[op].name.clone(),
                    buffer: state.buffers[id].name.clone(),
                });
            }
        }
        if stage.plan.is_some() {
            for id in stage.written() {
                made_by[id] = Some(index);
            }
        }
    }
    let mut exposed = vec![false; state.buffers.len()];
    for &id in &state.exposed {
        exposed[id] = true;
    }
    let params = core::mem::take(&mut state.params);
    let values = params.iter().map(|entry| entry.value).collect();
    let mut stages = core::mem::take(&mut state.stages);
    // A stage depends on the parameters that give the extents of the buffers it binds, as on the ones its body reads: a
    // `set` of one of them records its extent record again.
    for stage in &mut stages {
        let bound = stage.reads.iter().copied().chain(stage.written());
        let ruled: Vec<usize> = bound
            .filter_map(|id| match state.buffers[id].rule {
                Rule::Param(param) => Some(param),
                Rule::Fixed(_) => None,
            })
            .collect();
        for param in ruled {
            if !stage.params.contains(&param) {
                stage.params.push(param);
            }
        }
    }
    let prepared = stages
        .iter_mut()
        .map(|stage| stage.prepared.take().map(Arc::new))
        .collect();
    let plans = stages
        .iter_mut()
        .map(|stage| stage.first.take().into_iter().collect())
        .collect();
    let mut metas = Vec::new();
    let mut dims = Vec::new();
    let mut live = Vec::new();
    for (entry, alloc) in core::mem::take(&mut state.buffers)
        .into_iter()
        .zip(core::mem::take(&mut state.allocs))
    {
        metas.push(BufMeta {
            name: Arc::from(entry.name),
            input: entry.input,
            stride: entry.stride,
            rule: entry.rule,
            alloc,
        });
        dims.push(entry.dims);
        live.push(entry.buffer);
    }
    let core = Core {
        id: state.id,
        name: Arc::from(state.name.as_str()),
        device: Arc::clone(&state.device),
        buffers: metas,
        stages,
        params,
        exposed,
        generation: AtomicU64::new(0),
        run: Mutex::new(RunState {
            values,
            dims,
            buffers: live,
            prepared,
            plans,
            live: false,
            depth: usize::try_from(in_flight.max(1)).unwrap_or(1),
            pending: VecDeque::new(),
            stale: Vec::new(),
            extent_records: 0,
        }),
    };
    // The cell of `core` is not `Sync` (the CPU device's buffers are single-threaded); the batch is a shared handle for
    // its frames and tickets, which run on the thread that owns the device.
    #[allow(clippy::arc_with_non_send_sync)]
    let core = Arc::new(core);
    Ok((Batch { core }, output))
}
