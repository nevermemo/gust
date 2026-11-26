//! Library operations in a batch (plan m14.s2, slices s2f2a and s2b2, D259 and the s2f2 ruling): an op is one stage with
//! one prepared list of dispatches (or one step, on the CPU) and one or more new buffers that it alone writes. gust-algo
//! implements [`Op`] on its own types for each device (the orphan rule forbids an impl on a device), and a build block
//! records one with [`Builder::op`].
//!
//! **Resize.** An op stage follows a resize like a kernel stage: its operands and the outputs that follow them (the
//! outputs made with [`OpDraft::result_like`]) are replaced when a shape parameter is set, and the stage is prepared
//! again over the new buffers by the op's own functions: a plan for the new length, built once per length the batch
//! visited recently (the core keeps [`PLANS_KEPT`] of them per stage) and bound to the operands of the frame
//! ([`Operands`]). That happens inside [`FrameRun::set`](super::FrameRun::set), never in the encoder step, and
//! commits with the rest of the resize when the device took the frame. A length of zero prepares nothing: the stage's
//! [`Empty`] rule runs in each frame.
//!
//! **Allocation submissions.** The frame that follows a `set` is one queue submission, as any frame (D251). What the
//! `set` itself allocates, the new zeroed buffers and the scratch a new plan clears, may take submissions of the device's
//! own, made inside the `set` call and so outside the frame; a `set` that finds the length's plan kept builds nothing.
//! Every op inherits that rule (the reductions do not widen it): a plan function may allocate and clear, at its own cost,
//! but only in the call that makes the plan.
//!
//! Everything here is macro support, outside the pre-1.0 promise (D111): the forms users write arrive with the macro
//! (slice s2f2c).

use core::any::Any;
use core::marker::PhantomData;
use std::sync::Arc;

use super::{BatchDevice, BufEntry, Builder, D1, Error, Rule, Stage, StageRec, map};
use crate::GpuPod;

/// How many plans an op stage keeps: the plan of the length it runs at and the one of the length before it, so a batch
/// that alternates between two lengths builds each plan once.
pub(crate) const PLANS_KEPT: usize = 2;

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

/// The buffers an op's stage binds in one frame, in the order the op took them: the operands it read (the first one
/// gives the stage's length), then the results it made. A resize replaces some of them, so an op's functions take them
/// from here at each preparation and keep none of their own. The stage's length, and so the kept plan it looks up, is
/// that of the first operand read: an op that reads several must resize them together. Macro support, outside the pre-1.0 promise (D111).
#[doc(hidden)]
pub struct Operands<D: BatchDevice> {
    reads: Vec<Arc<dyn Any>>,
    results: Vec<Arc<dyn Any>>,
    len: usize,
    marker: PhantomData<fn() -> D>,
}

impl<D: BatchDevice> Operands<D> {
    pub(crate) const fn new(
        reads: Vec<Arc<dyn Any>>,
        results: Vec<Arc<dyn Any>>,
        len: usize,
    ) -> Self {
        Self {
            reads,
            results,
            len,
            marker: PhantomData,
        }
    }

    /// The number of elements of the first operand: the length the stage runs over.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the stage runs over no element (such a stage is never prepared; see [`Empty`]).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The operand read `k`-th (from 0), as `T`.
    ///
    /// # Panics
    ///
    /// When the op read fewer operands, or `T` is not the element of that one: both are the op's bug.
    #[must_use]
    pub fn read<T: GpuPod>(&self, k: usize) -> Arc<D::Buffer<T>>
    where
        D::Buffer<T>: 'static,
    {
        typed::<D, T>(&self.reads[k])
    }

    /// The result made `k`-th (from 0), as `T`.
    ///
    /// # Panics
    ///
    /// When the op made fewer results, or `T` is not the element of that one: both are the op's bug.
    #[must_use]
    pub fn result<T: GpuPod>(&self, k: usize) -> Arc<D::Buffer<T>>
    where
        D::Buffer<T>: 'static,
    {
        typed::<D, T>(&self.results[k])
    }
}

fn typed<D: BatchDevice, T: GpuPod>(any: &Arc<dyn Any>) -> Arc<D::Buffer<T>>
where
    D::Buffer<T>: 'static,
{
    Arc::clone(
        any.downcast_ref::<Arc<D::Buffer<T>>>()
            .expect("an operand is of the element type the op took it with"),
    )
}

/// The erased function that makes the plan of a length.
type Build<D> = Box<dyn Fn(&D, usize) -> Result<Arc<dyn Any>, <D as crate::job::Device>::Error>>;

/// The erased function that prepares an op stage over a plan and the operands of a frame.
type Bind<D> = Box<
    dyn Fn(
        &D,
        &Arc<dyn Any>,
        &Operands<D>,
    ) -> Result<<D as BatchDevice>::Prepared, <D as crate::job::Device>::Error>,
>;

/// What an op's stage keeps to be prepared again at another length: `build` makes the plan of a length (the part that
/// costs: scratch, clears, the choice of kernels) and `bind` prepares the stage over a plan and the operands of a frame.
/// The plan is erased here; the core keeps the plans of the lengths the batch visited last.
pub(crate) struct OpPlan<D: BatchDevice> {
    build: Build<D>,
    bind: Bind<D>,
}

impl<D: BatchDevice> OpPlan<D> {
    fn new<P: 'static>(
        build: impl Fn(&D, usize) -> Result<P, D::Error> + 'static,
        bind: impl Fn(&D, &Arc<P>, &Operands<D>) -> Result<D::Prepared, D::Error> + 'static,
    ) -> Self {
        Self {
            build: Box::new(move |device, len| {
                build(device, len).map(|plan| Arc::new(Arc::new(plan)) as Arc<dyn Any>)
            }),
            bind: Box::new(move |device, plan, operands| {
                bind(
                    device,
                    plan.downcast_ref::<Arc<P>>()
                        .expect("a plan is of the type its op built"),
                    operands,
                )
            }),
        }
    }

    /// The plan for a length.
    pub(crate) fn build(&self, device: &D, len: usize) -> Result<Arc<dyn Any>, D::Error> {
        (self.build)(device, len)
    }

    /// The stage prepared over `plan` and the operands.
    pub(crate) fn bind(
        &self,
        device: &D,
        plan: &Arc<dyn Any>,
        operands: &Operands<D>,
    ) -> Result<D::Prepared, D::Error> {
        (self.bind)(device, plan, operands)
    }
}

/// What a stage does at a length of zero, applied by every frame in `FrameRun::submit` and `encode` before anything
/// is encoded. Macro support, outside the pre-1.0 promise (D111).
#[doc(hidden)]
pub enum Empty<D: BatchDevice> {
    /// Write the identity of the operation (a sum or a count of nothing) into the result, through the frame's writes.
    /// The results it writes must be buffers no resize replaces (`result`, not `result_like`): the closure holds its
    /// buffers from the seal. Tested by the batch-basics resize tests (the count of a compaction at a length of zero).
    Identity(Identity<D>),
    /// Record nothing (a scan of nothing has nothing to write).
    Nothing,
    /// The op has no value at length zero (`min`, `max`): refuse the frame with [`Error::Empty`].
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

    /// The next result of the op: a new buffer of `len` elements, zeroed, named by the next name of the op. Its length
    /// stays `len` in every frame. `None` after recording why.
    pub fn result<T: GpuPod>(&mut self, len: usize) -> Option<Arc<D::Buffer<T>>>
    where
        D::Buffer<T>: 'static,
    {
        if !self.ok {
            return None;
        }
        let Ok(width) = u32::try_from(len) else {
            self.builder
                .state
                .borrow_mut()
                .fail(Error::Device(map::Error::TooLong { len }));
            self.ok = false;
            return None;
        };
        self.make_result([width, 1], Rule::Fixed([width, 1]))
    }

    /// The next result of the op, as long as the first operand in every frame: a new buffer, zeroed, whose extents follow
    /// the operand's through a resize (a scan's output, a compaction's indices). `None` after recording why.
    ///
    /// # Panics
    ///
    /// When the op has read no operand yet: it is the op's bug.
    pub fn result_like<T: GpuPod>(&mut self) -> Option<Arc<D::Buffer<T>>>
    where
        D::Buffer<T>: 'static,
    {
        if !self.ok {
            return None;
        }
        let first = *self
            .reads
            .first()
            .expect("an op reads an operand before it makes a result that follows it");
        let (dims, rule) = {
            let state = self.builder.state.borrow();
            let entry = &state.buffers[first];
            (entry.dims, entry.rule)
        };
        self.make_result(dims, rule)
    }

    fn make_result<T: GpuPod>(&mut self, dims: [u32; 2], rule: Rule) -> Option<Arc<D::Buffer<T>>>
    where
        D::Buffer<T>: 'static,
    {
        let mut state = self.builder.state.borrow_mut();
        let name = match self.names.get(self.written.len()) {
            Some(name) => name.clone(),
            None => state.take_name("op"),
        };
        let Some((id, buffer)) = state.add_buffer_ruled::<T>(name.clone(), false, dims, rule)
        else {
            self.ok = false;
            return None;
        };
        self.named.push(name);
        self.written.push((id, dims));
        Some(buffer)
    }

    /// The handles of an op that could not be recorded (an error is in the builder).
    #[must_use]
    pub fn abandon<H: Outputs>(self) -> H {
        H::dud(self.builder.state.borrow().id)
    }

    /// Seal the stage of an op that has no plan to keep: `prepare` makes the prepared stage over the operands of a frame,
    /// and is called now for the operands as sealed, and again at each resize that replaces one of them. See
    /// [`seal_planned`](Self::seal_planned) for the rest.
    pub fn seal<H: Outputs>(
        self,
        prepare: impl Fn(&D, &Operands<D>) -> Result<D::Prepared, D::Error> + 'static,
        empty: Empty<D>,
    ) -> H {
        self.seal_planned(
            |_, _| Ok(()),
            move |device, _, operands| prepare(device, operands),
            empty,
        )
    }

    /// Seal the stage: `plan` makes the plan of a length (whatever costs to make: scratch, clears) and `bind` prepares the
    /// stage over a plan and the operands of a frame. Both are called now, for the length of the first operand, and again
    /// by [`FrameRun::set`](super::FrameRun::set) when a resize changes it. Returns the handles of the results.
    ///
    /// The plan is built here, in the build block, so whatever it allocates and clears is submitted before the seal and
    /// a frame stays one submission (D251). A resize builds one in the `set`, outside the frame (see the module
    /// documentation for the rule), and the core keeps the plans of the last [`PLANS_KEPT`] lengths, so `bind` may be
    /// called again over a plan it was called with before, with other operands: a plan holds nothing of its operands.
    ///
    /// A stage over zero elements calls neither: it keeps no prepared stage, and each frame applies `empty` instead,
    /// before anything is encoded (see [`Empty`]).
    ///
    /// Both run with the builder released, so they may use the device freely.
    ///
    /// # Panics
    ///
    /// When the op made another number of results than `H` has handles, or read nothing: both are the op's bug.
    pub fn seal_planned<H: Outputs, P: 'static>(
        self,
        plan: impl Fn(&D, usize) -> Result<P, D::Error> + 'static,
        bind: impl Fn(&D, &Arc<P>, &Operands<D>) -> Result<D::Prepared, D::Error> + 'static,
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
        let state = builder.state.borrow_mut();
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
        let device = Arc::clone(&state.device);
        let buffer_of = |id: &usize| Arc::clone(&state.buffers[*id].buffer);
        let operands = Operands::new(
            reads.iter().map(buffer_of).collect(),
            written.iter().map(|(id, _)| id).map(buffer_of).collect(),
            len,
        );
        drop(state);
        let op_plan = OpPlan::new(plan, bind);
        let planned = if len == 0 {
            Ok((None, None))
        } else {
            op_plan.build(&device, len).and_then(|plan| {
                op_plan
                    .bind(&device, &plan, &operands)
                    .map(|prepared| (Some(prepared), Some((len, plan))))
            })
        };
        let mut state = builder.state.borrow_mut();
        match planned {
            Ok((prepared, first)) => {
                state.stages.push(StageRec::op(
                    name, prepared, reads, &written, op_plan, first, empty,
                ));
                H::make(batch, &written)
            }
            Err(error) => {
                let batch_name = state.name.clone();
                state.fail(Error::StageDevice {
                    batch: batch_name,
                    stage: name,
                    source: D::map_error(error),
                });
                H::dud(batch)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuDevice;

    /// An op of one result as long as its operand, that does nothing.
    struct Fill<'a>(&'a Stage<u32, D1>);

    impl Op<CpuDevice> for Fill<'_> {
        type Out = Stage<u32, D1>;

        fn record(self, mut draft: OpDraft<'_, CpuDevice>) -> Self::Out {
            if draft.read(self.0).is_none() || draft.result_like::<u32>().is_none() {
                return draft.abandon();
            }
            draft.seal(
                |device, _| Ok(device.prepare_fn(|_| Ok(()))),
                Empty::Nothing,
            )
        }
    }

    /// The rule is the core's and holds for any device; a kernel stage that writes an op's output cannot be written
    /// through `gpu_task!` yet (the macro does not see an op's type, slice s2f2c), so the `StageRec` is built by hand
    /// until then, and `StageDraft::target` (the macro's way to name a written buffer) is not exercised here.
    #[test]
    fn a_later_stage_may_not_write_an_op_output() {
        #[allow(clippy::arc_with_non_send_sync)]
        let device = Arc::new(CpuDevice::new());
        let result = crate::batch::build("overwrite", &device, 1, |b| {
            let input = b.input::<u32, _>(4);
            let made = b.op(&["made"], Fill(&input));
            let rec = StageRec::kernel(
                String::from("later"),
                device.prepare_fn(|_| Ok(())),
                Box::new(|_, _, _| Ok(())),
                Vec::new(),
                Vec::new(),
                made.id,
            );
            b.state.borrow_mut().stages.push(rec);
            made
        });
        let Err(Error::Overwrite {
            batch,
            stage,
            op,
            buffer,
        }) = result
        else {
            panic!("a later stage wrote an op's output and the batch was built");
        };
        assert_eq!(
            (batch.as_str(), stage.as_str(), op.as_str(), buffer.as_str()),
            ("overwrite", "later", "made", "made")
        );
    }
}
