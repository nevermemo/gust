//! The recording form of the reductions, the scans and the compaction (slices m14.s2f1 and its fix round, D259).
//!
//! The immediate functions of [`reduce`](crate::reduce), [`scan`](crate::scan) and [`compact`](crate::compact) take
//! their buffers, dispatch at once and read or return the result. This module holds the same operations as a list of
//! dispatches, in two layers.
//!
//! **Plans** ([`ReducePlan`](crate::reduce::ReducePlan), [`SumWrappingPlan`](crate::reduce::SumWrappingPlan),
//! [`ScanPlan`](crate::scan::ScanPlan), [`CountPlan`](crate::compact::CountPlan) and
//! [`CompactPlan`](crate::compact::CompactPlan)) are owned and have no
//! lifetime: each is made from a device, a variant and a length, owns the schedule and the pooled scratch buffers of
//! one shape (partials, totals, masks, positions) and holds no operand. A plan's `append` takes borrowed operands, any
//! [`Operand`] (a `GpuVec`, a `GpuBuffer`, a type of a caller's), checks their lengths against the plan's
//! (`Error::Length`, nothing appended) and appends the operation's dispatches to a [`Sink`]. The same plan can be
//! appended again, to the same buffers or to others of its length. A batch that keeps its stages `'static` keeps plans.
//!
//! **A [`Recording`]** is the convenience form for a caller with `GpuVec`s and a device of its own: each `sum`,
//! `min`, `max`, `sum_wrapping`, `inclusive`, `exclusive`, `count` and `indices` call of it makes the plan, checks its
//! operands and keeps both in a list. Nothing runs until the list does: [`Recording::dispatch`] sends every recorded
//! dispatch in one `dispatch_batch` (inside a [`timed`](crate::timed) scope it is counted as one batch), and
//! [`Recording::with_dispatches`] hands the same dispatches, as `BufferDispatch` values, to a caller that runs them
//! itself.
//!
//! The results go to buffers of the caller's, as the `_into` forms of [`scan`](crate::scan) do: a reduction writes
//! its one-element `out`, a scan its `out` of the input's length, `indices` its buffer (the input's length) and
//! `count` (one element). Every element of a reduction's, a scan's and a count's `out` is written by the run, so those
//! buffers can be reused from run to run. **Past `count`, the contents of a recorded `indices` are unspecified once a
//! list has run more than once**: the scatter writes only the first `count` elements, so the tail keeps what the
//! buffer held. The zero tail is the promise of the immediate [`compact::indices`](crate::compact::indices) alone (a
//! buffer made with `GpuVec::zeroed_on` shows it on the first run). A consumer uses the count. The recorded
//! dispatches run in the order the calls were made, so a later call may read the `out` of an earlier one.
//!
//! The immediate functions are this code too: each builds the same dispatches through [`Sink`] and sends them at
//! once, so a recorded list and an immediate call run the same kernels with the same schedules. An immediate
//! `compact::indices` stays four batches (as its timing documentation says); the recorded `indices` is one list.
//!
//! The module compiles for `wasm32`, where the immediate functions and [`Recording::dispatch`], which block, are
//! compiled out. The kernels need artifacts this crate does not ship, so nothing runs there yet.

use gust::{Call, Extent, GpuPod, Kernel, KernelDescriptor, Resource, kind::Storage};
use gust_wgpu::{BufferBinding, BufferDispatch, DeviceHandle, GpuBuffer, GpuVec, HeadlessDevice};

use crate::{
    Error, Variant,
    compact::{Compact, Count},
    reduce::{Element, Op, Reduce, SumWrapping},
    scan::Scan,
    variant::{Operation, shipped},
};

/// A buffer a plan can be appended over: a storage [`Resource`] of the GPU device that reports its length. `GpuVec`
/// and `GpuBuffer` are operands; a type of a caller's that is a `Resource<Element = T, Device = HeadlessDevice,
/// Kind = Storage>` becomes one by implementing the one method (the `Resource` trait has no length to read).
pub trait Operand<T: GpuPod>:
    Resource<Element = T, Device = HeadlessDevice, Kind = Storage>
{
    /// The number of elements the buffer holds.
    fn operand_len(&self) -> u32;
}

impl<T: GpuPod> Operand<T> for GpuVec<T> {
    fn operand_len(&self) -> u32 {
        self.len()
    }
}

impl<T: GpuPod> Operand<T> for GpuBuffer<T> {
    fn operand_len(&self) -> u32 {
        // A buffer past `u32::MAX` elements matches no plan's length.
        u32::try_from(self.len()).unwrap_or(u32::MAX)
    }
}

/// One dispatch ready for a batch: the kernel, the extent and the bindings, owned apart from the buffers.
struct Prepared<'s> {
    kernel: &'static KernelDescriptor,
    extent: Extent,
    bindings: Vec<BufferBinding<'s>>,
}

/// The dispatches of one or more operations, in order: what a plan's `append` fills. `'s` is the borrow of the plans
/// and operands the dispatches name, so a sink cannot outlive them.
///
/// ```
/// use gust_algo::{Variant, record::Sink, reduce::{Op, ReducePlan}};
/// use gust_wgpu::{GpuVec, default_device};
///
/// let device = default_device()?;
/// let input = GpuVec::from_slice_on(&device, &[3u32, 1, 4, 1, 5])?;
/// let total = GpuVec::<u32>::zeroed_on(&device, 1)?;
/// let plan = ReducePlan::<u32>::new(&device, Variant::V1, Op::Sum, 5)?;
///
/// let mut sink = Sink::new();
/// plan.append(&mut sink, &input, &total)?;
/// sink.dispatch(&device)?;
/// assert_eq!(total.to_vec()?, [14]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Sink<'s> {
    items: Vec<Prepared<'s>>,
}

impl Default for Sink<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'s> Sink<'s> {
    /// An empty list.
    #[must_use]
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// The number of dispatches in the list.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the list holds no dispatch.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Append the dispatch of `call`.
    pub fn push<K: Kernel, const N: usize>(&mut self, call: &Call<'s, HeadlessDevice, K, N>) {
        self.items.push(Prepared {
            kernel: call.kernel(),
            extent: call.extent(),
            bindings: call.bindings().to_vec(),
        });
    }

    /// Run `run` on the dispatches, as one slice of `BufferDispatch` values, for a caller that submits them its own
    /// way. The dispatches name the buffers of the plans and operands appended; those must outlive the GPU work of
    /// everything `run` encodes or submits.
    pub fn with<R>(&self, run: impl FnOnce(&[BufferDispatch<'_>]) -> R) -> R {
        let dispatches: Vec<BufferDispatch<'_>> = self
            .items
            .iter()
            .map(|item| BufferDispatch::new_extent(item.kernel, item.extent, &item.bindings))
            .collect();
        run(&dispatches)
    }

    /// Send the dispatches as one batch to `device`, counted in the enclosing [`timed`](crate::timed) scope. An empty
    /// list dispatches nothing. Not available on `wasm32`.
    ///
    /// # Errors
    ///
    /// `Error::Device` when the device refuses the batch (a buffer of another device, an `out` that is also an input,
    /// a failed submission).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch(&self, device: &HeadlessDevice) -> Result<(), Error> {
        self.with(|batch| crate::timing::dispatch_batch_in_scope(device, batch))?;
        Ok(())
    }
}

/// One recorded operation with its plan and operands.
pub(crate) trait Stage {
    /// Append the dispatches of the operation to `sink`.
    fn prepare<'s>(&'s self, sink: &mut Sink<'s>);
}

/// A list of recorded operations that run as one batch (see the module documentation). It is built on the plans: each
/// call makes the plan of its operation and keeps it with the operands.
///
/// **The recording must outlive the GPU work of everything it hands out or sends**: its plans own the scratch buffers
/// of the dispatches, and dropping it returns them to the pool.
///
/// ```
/// use gust_algo::record::Recording;
/// use gust_wgpu::{GpuVec, default_device};
///
/// let device = default_device()?;
/// let input = GpuVec::from_slice_on(&device, &[3u32, 1, 4, 1, 5])?;
/// let total = GpuVec::<u32>::zeroed_on(&device, 1)?;
/// let prefix = GpuVec::<u32>::zeroed_on(&device, 5)?;
///
/// let mut recording = Recording::new(&device);
/// recording.sum(&input, &total)?;
/// recording.inclusive(&input, &prefix)?;
/// assert_eq!(recording.dispatch_count(), 2);
/// recording.dispatch()?;
///
/// assert_eq!(total.to_vec()?, [14]);
/// assert_eq!(prefix.to_vec()?, [3, 4, 8, 9, 14]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Recording<'a> {
    device: &'a DeviceHandle,
    variant: Option<Variant>,
    stages: Vec<Box<dyn Stage + 'a>>,
}

impl<'a> Recording<'a> {
    /// An empty list for `device`. [`dispatch`](Self::dispatch) sends it there; every buffer recorded must belong to
    /// the same device (the dispatch refuses a buffer of another one, as `Error::Device`).
    #[must_use]
    pub fn new(device: &'a DeviceHandle) -> Self {
        Self {
            device,
            variant: None,
            stages: Vec::new(),
        }
    }

    /// Record the operations called from now on with the kernels of `variant` instead of the ones
    /// [`shipped`](crate::variant::shipped) names (v1 falls back to v0 for an input its launch cannot take, as for
    /// the `*_with` functions).
    #[must_use]
    pub fn with_variant(mut self, variant: Variant) -> Self {
        self.variant = Some(variant);
        self
    }

    /// The device the list was made for.
    #[must_use]
    pub fn device(&self) -> &'a DeviceHandle {
        self.device
    }

    fn variant_of(&self, operation: Operation) -> Variant {
        self.variant.unwrap_or_else(|| shipped(operation))
    }

    /// Record the reduction `op` of `input` into the one-element `out`.
    ///
    /// # Errors
    ///
    /// `Error::Empty` for an empty `input` (a recorded reduction must write `out`, and an empty input has no
    /// dispatch to do it; [`Op::identity`] is the value to write), `Error::Length` when `out` does not hold exactly
    /// one element, `Error::Device` for an input past the device's limits or a failed allocation. Nothing is
    /// recorded on an error.
    pub fn reduce<T: Element>(
        &mut self,
        op: Op,
        input: &'a GpuVec<T>,
        out: &'a GpuVec<T>,
    ) -> Result<(), Error> {
        let variant = self.variant_of(op.operation());
        let stage = Reduce::new(self.device, variant, op, input, out)?;
        self.stages.push(Box::new(stage));
        Ok(())
    }

    /// Record the sum of `input` into `out`; see [`reduce`](Self::reduce).
    ///
    /// # Errors
    ///
    /// As for [`reduce`](Self::reduce).
    pub fn sum<T: Element>(
        &mut self,
        input: &'a GpuVec<T>,
        out: &'a GpuVec<T>,
    ) -> Result<(), Error> {
        self.reduce(Op::Sum, input, out)
    }

    /// Record the minimum of `input` into `out`; see [`reduce`](Self::reduce).
    ///
    /// # Errors
    ///
    /// As for [`reduce`](Self::reduce).
    pub fn min<T: Element>(
        &mut self,
        input: &'a GpuVec<T>,
        out: &'a GpuVec<T>,
    ) -> Result<(), Error> {
        self.reduce(Op::Min, input, out)
    }

    /// Record the maximum of `input` into `out`; see [`reduce`](Self::reduce).
    ///
    /// # Errors
    ///
    /// As for [`reduce`](Self::reduce).
    pub fn max<T: Element>(
        &mut self,
        input: &'a GpuVec<T>,
        out: &'a GpuVec<T>,
    ) -> Result<(), Error> {
        self.reduce(Op::Max, input, out)
    }

    /// Record the sum of `input` modulo 2^32 into `out` (the recorded `crate::reduce::sum_wrapping`).
    ///
    /// # Errors
    ///
    /// As for [`reduce`](Self::reduce).
    pub fn sum_wrapping(
        &mut self,
        input: &'a GpuVec<u32>,
        out: &'a GpuVec<u32>,
    ) -> Result<(), Error> {
        let variant = self.variant_of(Operation::SumWrapping);
        let stage = SumWrapping::new(self.device, variant, input, out)?;
        self.stages.push(Box::new(stage));
        Ok(())
    }

    /// Record the inclusive prefix sums of `input` into `out` (the recorded [`crate::scan::inclusive_into`]).
    ///
    /// # Errors
    ///
    /// `Error::Length` when `out` holds another number of elements than `input`, `Error::Device` for an input past the
    /// device's limits or a failed allocation. An empty `input` records nothing and is `Ok`. Nothing is recorded on
    /// an error.
    pub fn inclusive(&mut self, input: &'a GpuVec<u32>, out: &'a GpuVec<u32>) -> Result<(), Error> {
        let variant = self.variant_of(Operation::Inclusive);
        self.scan(variant, input, out, false)
    }

    /// Record the exclusive prefix sums of `input` into `out` (the recorded [`crate::scan::exclusive_into`]).
    ///
    /// # Errors
    ///
    /// As for [`inclusive`](Self::inclusive).
    pub fn exclusive(&mut self, input: &'a GpuVec<u32>, out: &'a GpuVec<u32>) -> Result<(), Error> {
        let variant = self.variant_of(Operation::Exclusive);
        self.scan(variant, input, out, true)
    }

    fn scan(
        &mut self,
        variant: Variant,
        input: &'a GpuVec<u32>,
        out: &'a GpuVec<u32>,
        exclusive: bool,
    ) -> Result<(), Error> {
        if let Some(stage) = Scan::new(self.device, variant, input, out, exclusive)? {
            self.stages.push(Box::new(stage));
        }
        Ok(())
    }

    /// Record the number of set (non-zero) flags of `flags` into the one-element `count` (the recorded
    /// [`crate::compact::count`]).
    ///
    /// # Errors
    ///
    /// `Error::Empty` for empty `flags` (see [`reduce`](Self::reduce)), `Error::Length` when `count` does not hold
    /// exactly one element, `Error::Device` for an input past the device's limits or a failed allocation. Nothing is
    /// recorded on an error.
    pub fn count(&mut self, flags: &'a GpuVec<u32>, count: &'a GpuVec<u32>) -> Result<(), Error> {
        let stage = Count::new(self.device, self.variant, flags, count)?;
        self.stages.push(Box::new(stage));
        Ok(())
    }

    /// Record the positions of the set flags and their number (the recorded [`crate::compact::indices`]): `indices` and
    /// `count` are the two results the immediate function returns, `indices` as long as `flags` and `count` one
    /// element. Only the first `count` elements of `indices` are written (the scatter has one writer per set flag).
    /// **Past `count` the contents are unspecified once a list has run more than once**: the tail keeps what the buffer
    /// held, which is zero on the first run of a buffer made with `GpuVec::zeroed_on` and the earlier results
    /// after that. Only the immediate function promises a zero tail; a consumer uses the count. `count` is always
    /// written.
    ///
    /// # Errors
    ///
    /// `Error::Empty` for empty `flags`, `Error::Length` when `indices` does not hold as many elements as `flags` or
    /// `count` does not hold exactly one, `Error::Device` as for [`count`](Self::count). Nothing is recorded on an
    /// error.
    pub fn indices(
        &mut self,
        flags: &'a GpuVec<u32>,
        indices: &'a GpuVec<u32>,
        count: &'a GpuVec<u32>,
    ) -> Result<(), Error> {
        let stage = Compact::new(self.device, self.variant, flags, indices, count)?;
        self.stages.push(Box::new(stage));
        Ok(())
    }

    /// The number of dispatches the list holds.
    #[must_use]
    pub fn dispatch_count(&self) -> usize {
        self.with_dispatches(|dispatches| dispatches.len())
    }

    /// Run `run` on the recorded dispatches, in order, as one slice: what [`dispatch`](Self::dispatch) sends to
    /// `dispatch_batch`, for a caller that submits them its own way (`encode_batch`, a batch of the browser build).
    ///
    /// **The `Recording` must outlive the GPU work of anything `run` encodes or submits**: the dispatches write the
    /// plans' scratch buffers, which go back to the device's pool when the recording drops, and a new owner of one
    /// would race the caller's dispatches.
    pub fn with_dispatches<R>(&self, run: impl FnOnce(&[BufferDispatch<'_>]) -> R) -> R {
        let mut sink = Sink::new();
        for stage in &self.stages {
            stage.prepare(&mut sink);
        }
        sink.with(run)
    }

    /// Send every recorded dispatch in one `dispatch_batch` (one batch of a [`timed`](crate::timed) scope; an empty
    /// list dispatches nothing). The list is not consumed: it can run again, and the second run writes the same
    /// results from the buffers' current contents. Not available on `wasm32`.
    ///
    /// # Errors
    ///
    /// `Error::Device` when the device refuses the batch (a buffer of another device, an `out` that is also an
    /// input, a failed submission).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch(&self) -> Result<(), Error> {
        self.with_dispatches(|batch| crate::timing::dispatch_batch_in_scope(self.device, batch))?;
        Ok(())
    }
}
