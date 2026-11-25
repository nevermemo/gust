//! Library operations in a batch (plan m14.s2, slice s2f2a, D259 and the s2f2 ruling): an op is one stage with one
//! prepared list of dispatches (or one step, on the CPU) and one or more new buffers that it alone writes. gust-algo
//! implements [`Op`] on its own types for each device (the orphan rule forbids an impl on a device), and a build block
//! records one with [`Builder::op`].
//!
//! Everything here is macro support, outside the pre-1.0 promise (D111): the forms users write arrive with the macro
//! (slice s2f2c).

use std::sync::Arc;

use super::{BatchDevice, BufEntry, Builder, D1, DeviceMessage, Error, Stage, StageRec, map};
use crate::GpuPod;

/// A library operation that records itself as one stage of a batch. Macro support, outside the pre-1.0 promise (D111).
#[doc(hidden)]
pub trait Op<D: BatchDevice> {
    /// The handles of the op's outputs.
    type Out;

    /// Record the op into `draft`: read its operands, make its result buffers and seal the stage.
    fn record(self, draft: OpDraft<'_, D>) -> Self::Out;
}

/// The write of an operation's identity into its result, through the frame's writes.
type Identity<D> =
    Box<dyn Fn(&D, &mut <D as BatchDevice>::Frame) -> Result<(), <D as crate::job::Device>::Error>>;

/// What a stage does at a length of zero. Macro support, outside the pre-1.0 promise (D111).
///
/// Not applied yet: until slice s2f2b, [`OpDraft::seal`] refuses a length of zero as a device error whatever the op
/// asks for, and the variants only record what the op will want then.
#[doc(hidden)]
pub enum Empty<D: BatchDevice> {
    /// Write the identity of the operation (a sum or a count of nothing) into the result, through the frame's writes.
    Identity(Identity<D>),
    /// Record nothing (a scan of nothing has nothing to write).
    Nothing,
    /// The op has no value at length zero (`min`, `max`): refuse the frame.
    Refuse,
}

/// The handles an op returns: one [`Stage`], or a tuple of them. Macro support, outside the pre-1.0 promise (D111).
#[doc(hidden)]
pub trait Outputs {
    /// How many buffers the op makes.
    const COUNT: usize;

    /// The handles of buffers (id and extents), in the order the op made them.
    fn make(batch: u64, outputs: &[(usize, [u32; 2])]) -> Self;

    /// The handles of an op that could not be recorded (an error is in the builder): handles no frame accepts.
    fn dud(batch: u64) -> Self;
}

impl<T> Outputs for Stage<T, D1> {
    const COUNT: usize = 1;

    fn make(batch: u64, outputs: &[(usize, [u32; 2])]) -> Self {
        Self::new(batch, outputs[0].0, outputs[0].1)
    }

    fn dud(batch: u64) -> Self {
        Self::new(batch, usize::MAX, [0, 0])
    }
}

impl<T, U> Outputs for (Stage<T, D1>, Stage<U, D1>) {
    const COUNT: usize = 2;

    fn make(batch: u64, outputs: &[(usize, [u32; 2])]) -> Self {
        (
            Stage::new(batch, outputs[0].0, outputs[0].1),
            Stage::new(batch, outputs[1].0, outputs[1].1),
        )
    }

    fn dud(batch: u64) -> Self {
        (
            Stage::new(batch, usize::MAX, [0, 0]),
            Stage::new(batch, usize::MAX, [0, 0]),
        )
    }
}

impl<D: BatchDevice> Builder<D> {
    /// Record a library op as one stage. `names` are the names of its outputs, in order (the names of the `let`s that
    /// bind them); an output with no name is `op#n`. Returns the op's handles. A call on a sealed builder, or with a
    /// handle of another batch, records the error that [`check`](Self::check) and the end of the block report.
    /// Macro support, outside the pre-1.0 promise (D111).
    #[doc(hidden)]
    pub fn op<O: Op<D>>(&self, names: &[&str], op: O) -> O::Out {
        let ok = {
            let mut state = self.state.borrow_mut();
            if state.sealed {
                let batch = state.name.clone();
                state.fail(Error::Sealed { batch });
                false
            } else {
                true
            }
        };
        op.record(OpDraft {
            builder: self,
            names: names.iter().map(|name| (*name).to_owned()).collect(),
            named: Vec::new(),
            len: None,
            reads: Vec::new(),
            written: Vec::new(),
            ok,
        })
    }
}

/// An op being recorded: a stage draft without extents or a kernel. Macro support, outside the pre-1.0 promise (D111).
#[doc(hidden)]
pub struct OpDraft<'b, D: BatchDevice> {
    builder: &'b Builder<D>,
    names: Vec<String>,
    /// The names the results got, in order: `names`, then `op#n` for the results past them.
    named: Vec<String>,
    /// The length of the first operand: the length the stage is sealed for.
    len: Option<usize>,
    reads: Vec<usize>,
    /// The buffers the op made, with their extents, in order.
    written: Vec<(usize, [u32; 2])>,
    ok: bool,
}

impl<D: BatchDevice> OpDraft<'_, D> {
    /// The name of the stage: the name of its first output (the one its buffer got, `op#n` when the op has no name).
    fn stage_name(&self) -> String {
        self.named
            .first()
            .or_else(|| self.names.first())
            .cloned()
            .unwrap_or_else(|| String::from("op"))
    }

    fn foreign(&mut self) {
        let mut state = self.builder.state.borrow_mut();
        let (batch, stage) = (state.name.clone(), self.stage_name());
        state.fail(Error::Foreign { batch, stage });
        self.ok = false;
    }

    /// The device.
    #[must_use]
    pub fn device(&self) -> Arc<D> {
        Arc::clone(&self.builder.state.borrow().device)
    }

    /// The buffer behind a handle the op reads, and its length. The first operand read sets the length the stage is
    /// sealed for. `None` after recording why (a handle of another batch, or one that could not be recorded).
    pub fn read<T: GpuPod, R>(&mut self, handle: &Stage<T, R>) -> Option<(Arc<D::Buffer<T>>, usize)>
    where
        D::Buffer<T>: 'static,
    {
        if !self.ok {
            return None;
        }
        let found = {
            let state = self.builder.state.borrow();
            (handle.batch == state.id && handle.id < state.buffers.len()).then(|| {
                let entry: &BufEntry = &state.buffers[handle.id];
                entry
                    .buffer
                    .downcast_ref::<Arc<D::Buffer<T>>>()
                    .cloned()
                    .map(|buffer| (buffer, entry.len))
            })
        };
        if let Some((buffer, len)) = found.flatten() {
            self.reads.push(handle.id);
            self.len.get_or_insert(len);
            Some((buffer, len))
        } else {
            self.foreign();
            None
        }
    }

    /// The next result of the op: a new buffer of `len` elements, zeroed, named by the next name of the op. `None`
    /// after recording why.
    pub fn result<T: GpuPod>(&mut self, len: usize) -> Option<Arc<D::Buffer<T>>>
    where
        D::Buffer<T>: 'static,
    {
        if !self.ok {
            return None;
        }
        let mut state = self.builder.state.borrow_mut();
        let Ok(width) = u32::try_from(len) else {
            state.fail(Error::Device(map::Error::TooLong { len }));
            self.ok = false;
            return None;
        };
        let name = match self.names.get(self.written.len()) {
            Some(name) => name.clone(),
            None => state.take_name("op"),
        };
        let Some((id, buffer)) = state.add_buffer::<T>(name.clone(), false, [width, 1]) else {
            self.ok = false;
            return None;
        };
        self.named.push(name);
        self.written.push((id, [width, 1]));
        Some(buffer)
    }

    /// The handles of an op that could not be recorded (an error is in the builder).
    #[must_use]
    pub fn abandon<H: Outputs>(self) -> H {
        H::dud(self.builder.state.borrow().id)
    }

    /// Seal the stage: `plan` makes the prepared stage for a length, and is called now for the length of the first
    /// operand. Returns the handles of the results.
    ///
    /// The plan is built here, in the build block, so whatever it allocates and clears is submitted before the seal and
    /// a frame stays one submission (D251).
    ///
    /// Interim rule, until s2f2b: a stage sealed at length 0 is refused here as a device error, and `empty` is not
    /// applied. s2f2b replaces both with `Error::Empty` and the three rules of [`Empty`], raised by the frame.
    ///
    /// The core keeps nothing of `plan` yet: a stage record has no field for it, and adding one is s2f2b's edit of
    /// `mod.rs`. What `plan` makes is valid for the sealed length only: the stage holds the operands (and, for scan and
    /// compact, the outputs) of the sealed buffers, so a later resize cannot call it again to replace those outputs. On
    /// the GPU device a call at another length returns a length error, never a panic; the CPU device's host form
    /// ignores the length and runs over the sealed buffers. A resize will need the new buffers handed to the plan along
    /// with the length.
    ///
    /// `plan` runs with the builder released, so it may use the device freely.
    ///
    /// # Panics
    ///
    /// When the op made another number of results than `H` has handles, or read nothing: both are the op's bug.
    pub fn seal<H: Outputs>(
        self,
        plan: impl Fn(&D, usize) -> Result<D::Prepared, D::Error> + 'static,
        empty: Empty<D>,
    ) -> H {
        let name = self.stage_name();
        let Self {
            builder,
            len,
            reads,
            written,
            ok,
            ..
        } = self;
        let mut state = builder.state.borrow_mut();
        let batch = state.id;
        if !ok {
            return H::dud(batch);
        }
        assert_eq!(
            written.len(),
            H::COUNT,
            "an op makes as many results as it has handles"
        );
        let len = len.expect("an op reads an operand before it seals");
        if len == 0 {
            // Interim, s2f2b: the frame's `Empty` rules replace this refusal.
            drop(empty);
            state.fail(Error::Device(map::Error::device(DeviceMessage(format!(
                "stage `{name}` is an operation over zero elements, which a batch cannot run yet"
            )))));
            return H::dud(batch);
        }
        let device = Arc::clone(&state.device);
        drop(state);
        let planned = plan(&device, len);
        let mut state = builder.state.borrow_mut();
        match planned {
            Ok(prepared) => {
                state.stages.push(StageRec {
                    name,
                    prepared,
                    update: None,
                    params: Vec::new(),
                    reads,
                    write: written[0].0,
                });
                H::make(batch, &written)
            }
            Err(error) => {
                state.fail(Error::Device(D::map_error(error)));
                H::dud(batch)
            }
        }
    }
}
