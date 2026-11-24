//! `sum`, `min`, `max` of a [`GpuVec`] of `u32`, `i32` or `f32` and `sum_wrapping` of `u32` (plan section 5.2).
//!
//! One reduction is one `dispatch_batch`, with one pooled `GpuVec` per pass for the partials and one element for the
//! result, all taken from the device's buffer pool and returned to it on drop. Which kernels it runs is
//! [`crate::variant::shipped`] (v1 for every operation, plan section 5.5):
//!
//! - v1, the shipped form: block passes of [`BlockSchedule`], one workgroup of 256 lanes per 8192 elements, each lane
//!   adding up to 32 terms and a tree in workgroup memory (barriers) leaving one value per workgroup, until one
//!   workgroup is left and writes the result: one dispatch up to 8192 elements, two up to 67 million, three above.
//! - v0, the fallback and the CPU form: the block-strided passes of [`Schedule`], then the terminal kernel, at most
//!   four dispatches. A v1 call on a device with too little workgroup memory runs v0 (a binding past the storage limit
//!   is refused by v0 as well; a first dispatch past the workgroup limit is split by the host, D211).
//!
//! Every input binding is `Independent` (its length differs from the invocation count), and in v0 every output binding
//! shares the invocation count (in v1 the outputs hold one value per workgroup, so they are `Independent` too).
//!
//! The contract (section 5.1): the sum of an empty input is the identity (`0`, `0.0`); `min` and `max` of an empty
//! input are [`Error::Empty`]. Integer sums use plain `+` and are exact and in the domain only when no partial sum
//! overflows (the kernels wrap on the GPU, the debug twin panics, as every `+` of the dialect). `min` and `max` are
//! exact. Float sums are within the bound of the variant that ran: [`crate::bound::f32_sum_bound_v1`] when v1 ran,
//! [`crate::bound::f32_sum_bound`] when v0 ran as the fallback. The crate has no public call that says which one ran
//! for a given input and device, so the bound a caller may assert is the looser (larger) of the two; v1's is the
//! smaller one except for `n` of 2 to 8 elements, where its depth of 8 exceeds v0's `n - 1`. An input larger than
//! the device's storage binding limit is `Error::Device(DeviceLimit)` before any buffer is taken or dispatch is made.

use gust::{GpuPod, Independent};
use gust_wgpu::{DeviceHandle, GpuVec};

#[cfg(not(target_arch = "wasm32"))]
use crate::variant::shipped;
use crate::{
    Error, kernels, limits,
    record::{Operand, Sink, Stage},
    schedule::{BlockSchedule, Schedule},
    variant::{Operation, Variant},
};

/// The reduction a call performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Sum,
    Min,
    Max,
}

impl Op {
    /// The value of the reduction over an empty input: `Some(0)` for the sum (and, as a sum of flags, for the count
    /// of [`compact`](crate::compact)), `None` for `min` and `max`, which have no value there. The plans refuse an
    /// empty input with [`Error::Empty`]; a caller that must still write a result for one (a batch whose length can
    /// reach zero) writes this value, and refuses the operation when it is `None`. The wrapping sum of
    /// `sum_wrapping` has the identity of [`Op::Sum`] for `u32`.
    #[must_use]
    pub fn identity<T: Element>(self) -> Option<T> {
        match self {
            Self::Sum => Some(T::ZERO),
            Self::Min | Self::Max => None,
        }
    }

    /// The [`Operation`] whose shipped variant the reduction runs.
    pub(crate) const fn operation(self) -> Operation {
        match self {
            Self::Sum => Operation::Sum,
            Self::Min => Operation::Min,
            Self::Max => Operation::Max,
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// The element types of [`sum`], [`min`] and [`max`]: `u32`, `i32` and `f32`.
pub trait Element: GpuPod + sealed::Sealed + Copy + PartialOrd + 'static {
    /// The additive identity, the sum of an empty input.
    const ZERO: Self;

    /// Append the passes of `schedule` for `op` over `input` to `sink`: pass `j` reads the previous level and writes
    /// `partials[j]`, the terminal kernel writes `out[0]`.
    #[doc(hidden)]
    fn plan<'s, I: Operand<Self>, O: Operand<Self>>(
        op: Op,
        sink: &mut Sink<'s>,
        schedule: &Schedule,
        input: &'s I,
        partials: &'s [GpuVec<Self>],
        out: &'s O,
    );

    /// The v1 chain (plan section 5.5): pass `j` of `schedule` reads the previous level and writes `partials[j]`, the
    /// last pass (one workgroup) writes `out[0]`. There is no CPU form: the block kernels are `cpu = "unsupported"`.
    #[doc(hidden)]
    fn plan_v1<'s, I: Operand<Self>, O: Operand<Self>>(
        op: Op,
        sink: &mut Sink<'s>,
        schedule: &BlockSchedule,
        input: &'s I,
        partials: &'s [GpuVec<Self>],
        out: &'s O,
    );

    /// The same chain on the CPU twins of the kernels (`gust::cpu::CpuDevice`), in test and dual builds.
    #[cfg(any(test, gust_dual))]
    #[doc(hidden)]
    fn run_cpu(
        op: Op,
        device: &gust::cpu::CpuDevice,
        schedule: &Schedule,
        input: &gust::cpu::CpuBuffer<Self>,
        partials: &[gust::cpu::CpuBuffer<Self>],
        out: &gust::cpu::CpuBuffer<Self>,
    ) -> Result<(), gust::cpu::Error>;
}

/// The passes of one `(op, type)` pair as typed calls appended to a sink. The input and the result are any operands and
/// the partials are the plan's own `GpuVec`s, so the first pass and the terminal kernel are written out for their types.
macro_rules! gpu_chain {
    ($pass:ident, $terminal:ident, $sink:expr, $schedule:expr, $input:expr, $partials:expr, $out:expr) => {{
        let passes = $schedule.passes();
        for (index, (pass, destination)) in passes.iter().zip($partials).enumerate() {
            if index == 0 {
                $sink.push(&kernels::$pass::call(
                    pass.partials(),
                    Independent($input),
                    destination,
                ));
            } else {
                $sink.push(&kernels::$pass::call(
                    pass.partials(),
                    Independent(&$partials[index - 1]),
                    destination,
                ));
            }
        }
        if let Some(last) = passes.len().checked_sub(1) {
            $sink.push(&kernels::$terminal::call(
                1,
                Independent(&$partials[last]),
                $out,
            ));
        } else {
            $sink.push(&kernels::$terminal::call(1, Independent($input), $out));
        }
    }};
}

/// The block passes of one `(op, type)` pair as typed calls appended to a sink: every pass but the last writes its own
/// partials, the last (one workgroup) writes the result. The outputs hold one value per workgroup, not per invocation,
/// so they are `Independent`.
macro_rules! gpu_block_chain {
    ($block:ident, $sink:expr, $schedule:expr, $input:expr, $partials:expr, $out:expr) => {{
        let passes = $schedule.passes();
        for (index, pass) in passes.iter().enumerate() {
            let first = index == 0;
            let last = index + 1 == passes.len();
            let invocations = pass.invocations();
            if first && last {
                $sink.push(&kernels::$block::call(
                    invocations,
                    Independent($input),
                    Independent($out),
                ));
            } else if first {
                $sink.push(&kernels::$block::call(
                    invocations,
                    Independent($input),
                    Independent(&$partials[index]),
                ));
            } else if last {
                $sink.push(&kernels::$block::call(
                    invocations,
                    Independent(&$partials[index - 1]),
                    Independent($out),
                ));
            } else {
                $sink.push(&kernels::$block::call(
                    invocations,
                    Independent(&$partials[index - 1]),
                    Independent(&$partials[index]),
                ));
            }
        }
    }};
}

#[cfg(any(test, gust_dual))]
macro_rules! cpu_chain {
    ($pass:ident, $terminal:ident, $device:expr, $schedule:expr, $input:expr, $partials:expr, $out:expr) => {{
        let mut source = $input;
        for (pass, destination) in $schedule.passes().iter().zip($partials) {
            kernels::$pass::dispatch($device, pass.partials(), Independent(source), destination)?;
            source = destination;
        }
        kernels::$terminal::dispatch($device, 1, Independent(source), $out)
    }};
}

macro_rules! element {
    (
        $ty:ty, $zero:expr,
        sum: ($sum_pass:ident, $sum_final:ident),
        min: ($min_pass:ident, $min_final:ident),
        max: ($max_pass:ident, $max_final:ident),
        block: ($sum_block:ident, $min_block:ident, $max_block:ident)
    ) => {
        impl sealed::Sealed for $ty {}

        impl Element for $ty {
            const ZERO: Self = $zero;

            fn plan<'s, I: Operand<Self>, O: Operand<Self>>(
                op: Op,
                sink: &mut Sink<'s>,
                schedule: &Schedule,
                input: &'s I,
                partials: &'s [GpuVec<Self>],
                out: &'s O,
            ) {
                match op {
                    Op::Sum => {
                        gpu_chain!($sum_pass, $sum_final, sink, schedule, input, partials, out)
                    }
                    Op::Min => {
                        gpu_chain!($min_pass, $min_final, sink, schedule, input, partials, out)
                    }
                    Op::Max => {
                        gpu_chain!($max_pass, $max_final, sink, schedule, input, partials, out)
                    }
                }
            }

            fn plan_v1<'s, I: Operand<Self>, O: Operand<Self>>(
                op: Op,
                sink: &mut Sink<'s>,
                schedule: &BlockSchedule,
                input: &'s I,
                partials: &'s [GpuVec<Self>],
                out: &'s O,
            ) {
                match op {
                    Op::Sum => gpu_block_chain!($sum_block, sink, schedule, input, partials, out),
                    Op::Min => gpu_block_chain!($min_block, sink, schedule, input, partials, out),
                    Op::Max => gpu_block_chain!($max_block, sink, schedule, input, partials, out),
                }
            }

            #[cfg(any(test, gust_dual))]
            fn run_cpu(
                op: Op,
                device: &gust::cpu::CpuDevice,
                schedule: &Schedule,
                input: &gust::cpu::CpuBuffer<Self>,
                partials: &[gust::cpu::CpuBuffer<Self>],
                out: &gust::cpu::CpuBuffer<Self>,
            ) -> Result<(), gust::cpu::Error> {
                match op {
                    Op::Sum => cpu_chain!(
                        $sum_pass, $sum_final, device, schedule, input, partials, out
                    ),
                    Op::Min => cpu_chain!(
                        $min_pass, $min_final, device, schedule, input, partials, out
                    ),
                    Op::Max => cpu_chain!(
                        $max_pass, $max_final, device, schedule, input, partials, out
                    ),
                }
            }
        }
    };
}

element!(
    u32, 0,
    sum: (sum_u32_pass, sum_u32_final),
    min: (min_u32_pass, min_u32_final),
    max: (max_u32_pass, max_u32_final),
    block: (sum_u32_block, min_u32_block, max_u32_block)
);
element!(
    i32, 0,
    sum: (sum_i32_pass, sum_i32_final),
    min: (min_i32_pass, min_i32_final),
    max: (max_i32_pass, max_i32_final),
    block: (sum_i32_block, min_i32_block, max_i32_block)
);
element!(
    f32, 0.0,
    sum: (sum_f32_pass, sum_f32_final),
    min: (min_f32_pass, min_f32_final),
    max: (max_f32_pass, max_f32_final),
    block: (sum_f32_block, min_f32_block, max_f32_block)
);

/// The sum of all elements, read back to the host. The sum of an empty input is `0`. Not available on `wasm32`
/// (it blocks on the readback; the recorded form is [`Recording::sum`](crate::record::Recording::sum)).
///
/// # Errors
///
/// `Error::Device` for a device failure or an input past the storage binding limit.
#[cfg(not(target_arch = "wasm32"))]
pub fn sum<T: Element>(input: &GpuVec<T>) -> Result<T, Error> {
    sum_with(shipped(Operation::Sum), input)
}

/// [`sum`] with the kernels of `variant` (v1 falls back to v0 for an input its launch cannot take).
///
/// # Errors
///
/// As for [`sum`].
#[cfg(not(target_arch = "wasm32"))]
pub fn sum_with<T: Element>(variant: Variant, input: &GpuVec<T>) -> Result<T, Error> {
    if input.is_empty() {
        return Ok(T::ZERO);
    }
    read_one(&reduce_on_device(variant, Op::Sum, input)?)
}

/// The smallest element. [`Error::Empty`] for an empty input.
///
/// # Errors
///
/// `Error::Empty`, or `Error::Device` as for [`sum`].
#[cfg(not(target_arch = "wasm32"))]
pub fn min<T: Element>(input: &GpuVec<T>) -> Result<T, Error> {
    min_with(shipped(Operation::Min), input)
}

/// [`min`] with the kernels of `variant`.
///
/// # Errors
///
/// As for [`min`].
#[cfg(not(target_arch = "wasm32"))]
pub fn min_with<T: Element>(variant: Variant, input: &GpuVec<T>) -> Result<T, Error> {
    if input.is_empty() {
        return Err(Error::Empty);
    }
    read_one(&reduce_on_device(variant, Op::Min, input)?)
}

/// The largest element. [`Error::Empty`] for an empty input.
///
/// # Errors
///
/// `Error::Empty`, or `Error::Device` as for [`sum`].
#[cfg(not(target_arch = "wasm32"))]
pub fn max<T: Element>(input: &GpuVec<T>) -> Result<T, Error> {
    max_with(shipped(Operation::Max), input)
}

/// [`max`] with the kernels of `variant`.
///
/// # Errors
///
/// As for [`max`].
#[cfg(not(target_arch = "wasm32"))]
pub fn max_with<T: Element>(variant: Variant, input: &GpuVec<T>) -> Result<T, Error> {
    if input.is_empty() {
        return Err(Error::Empty);
    }
    read_one(&reduce_on_device(variant, Op::Max, input)?)
}

/// The sum as a one-element `GpuVec` on the device of `input`, with no readback (for `compact::count` of m7.r3). The
/// sum of an empty input is a vector holding the identity.
///
/// # Errors
///
/// As for [`sum`].
#[cfg(not(target_arch = "wasm32"))]
pub fn sum_on_device<T: Element>(input: &GpuVec<T>) -> Result<GpuVec<T>, Error> {
    sum_on_device_with(shipped(Operation::Sum), input)
}

/// [`sum_on_device`] with the kernels of `variant`.
///
/// # Errors
///
/// As for [`sum`].
#[cfg(not(target_arch = "wasm32"))]
pub fn sum_on_device_with<T: Element>(
    variant: Variant,
    input: &GpuVec<T>,
) -> Result<GpuVec<T>, Error> {
    if input.is_empty() {
        return Ok(GpuVec::zeroed_on(input.device(), 1)?);
    }
    reduce_on_device(variant, Op::Sum, input)
}

#[cfg(not(target_arch = "wasm32"))]
fn read_one<T: Element>(result: &GpuVec<T>) -> Result<T, Error> {
    let values = result.to_vec()?;
    Ok(values[0])
}

/// One reduction of a non-empty input: the limit checks, the pooled buffers, one `dispatch_batch`.
#[cfg(not(target_arch = "wasm32"))]
fn reduce_on_device<T: Element>(
    variant: Variant,
    op: Op,
    input: &GpuVec<T>,
) -> Result<GpuVec<T>, Error> {
    let device = input.device();
    let parts = Parts::<T>::new(device, variant, input.len())?;
    let out = GpuVec::<T>::zeroed_on(device, 1)?;
    let mut sink = Sink::new();
    parts.plan(op, &mut sink, input, &out);
    sink.dispatch(device)?;
    Ok(out)
}

/// The sum of all elements of a `u32` vector, defined modulo 2^32 for every input: the lane chains and the partials
/// wrap (`wrapping_add`, SEMANTICS section 3 item 1, KM-10). The sum of an empty input is `0`.
///
/// This is `sum` for inputs whose total may pass `u32::MAX`; for inputs that stay inside, both give the same value.
/// The twin is `cpu::sum_wrapping` and calls Rust's `wrapping_add`, so the value is the same on the device, on the twin
/// and in plain Rust, with overflow checks on and off.
///
/// # Errors
///
/// As for [`sum`].
#[cfg(not(target_arch = "wasm32"))]
pub fn sum_wrapping(input: &GpuVec<u32>) -> Result<u32, Error> {
    sum_wrapping_with(shipped(Operation::SumWrapping), input)
}

/// [`sum_wrapping`] with the kernels of `variant`.
///
/// # Errors
///
/// As for [`sum`].
#[cfg(not(target_arch = "wasm32"))]
pub fn sum_wrapping_with(variant: Variant, input: &GpuVec<u32>) -> Result<u32, Error> {
    if input.is_empty() {
        return Ok(0);
    }
    let device = input.device();
    let parts = Parts::<u32>::new(device, variant, input.len())?;
    let out = GpuVec::<u32>::zeroed_on(device, 1)?;
    let mut sink = Sink::new();
    parts.plan_wrapping(&mut sink, input, &out);
    sink.dispatch(device)?;
    read_one(&out)
}

/// The schedule of one reduction and the pooled buffers it needs besides its input and result.
pub(crate) struct Parts<T: Element> {
    plan: Plan,
    partials: Vec<GpuVec<T>>,
}

/// The v1 block passes, or the v0 block-strided passes and the terminal kernel.
enum Plan {
    Blocks(BlockSchedule),
    Strided(Schedule),
}

impl<T: Element> Parts<T> {
    /// The schedule for a non-empty input of `len` elements and the pooled partials: v1 when `variant` asks for it and
    /// the device admits it (storage binding, workgroup memory; a first dispatch past the workgroup limit is split by
    /// the host, D211), else v0, which checks the storage binding and reports whatever limit it hits.
    pub(crate) fn new(device: &DeviceHandle, variant: Variant, len: u32) -> Result<Self, Error> {
        let v1 = (variant == Variant::V1)
            .then(|| BlockSchedule::new(len))
            .flatten()
            .filter(|_| limits::admits(device, len, size_of::<T>()));
        if let Some(schedule) = v1 {
            // Every pass but the last writes its own partials; the last (one workgroup) writes the result.
            let passes = schedule.passes();
            let partials = passes[..passes.len() - 1]
                .iter()
                .map(|pass| GpuVec::<T>::zeroed_on(device, pass.workgroups))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(Self {
                plan: Plan::Blocks(schedule),
                partials,
            });
        }
        let schedule = Schedule::new(len).expect("the input is not empty");
        // Only the storage binding is checked: a first pass past the workgroup limit is split by the host (D211, D208).
        limits::check(device, "gust_algo::reduce", len, size_of::<T>())?;
        // Every lane writes its slot, so the zero fill is not needed for correctness; `GpuVec` has no public
        // constructor without it (a recycled buffer costs one clearing submission), and `gust-wgpu` is not this
        // slice's to change.
        let partials = schedule
            .passes()
            .iter()
            .map(|pass| GpuVec::<T>::zeroed_on(device, pass.partials()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            plan: Plan::Strided(schedule),
            partials,
        })
    }

    /// Append the passes of reduction `op` of `input` into the one-element `out`.
    pub(crate) fn plan<'s, I: Operand<T>, O: Operand<T>>(
        &'s self,
        op: Op,
        sink: &mut Sink<'s>,
        input: &'s I,
        out: &'s O,
    ) {
        match &self.plan {
            Plan::Blocks(schedule) => T::plan_v1(op, sink, schedule, input, &self.partials, out),
            Plan::Strided(schedule) => T::plan(op, sink, schedule, input, &self.partials, out),
        }
    }
}

impl Parts<u32> {
    /// Append the passes of the wrapping sum of `input` into the one-element `out`.
    pub(crate) fn plan_wrapping<'s, I: Operand<u32>, O: Operand<u32>>(
        &'s self,
        sink: &mut Sink<'s>,
        input: &'s I,
        out: &'s O,
    ) {
        match &self.plan {
            Plan::Blocks(schedule) => {
                gpu_block_chain!(
                    sum_wrapping_u32_block,
                    sink,
                    schedule,
                    input,
                    self.partials,
                    out
                );
            }
            Plan::Strided(schedule) => gpu_chain!(
                sum_wrapping_u32_pass,
                sum_wrapping_u32_final,
                sink,
                schedule,
                input,
                &self.partials,
                out
            ),
        }
    }
}

/// The dispatches of one reduction of `len` elements, made once and appended over any operands of that length (see
/// [`Sink`]): the schedule and the pooled partial buffers, owned, with no borrow of an input or a result and no
/// lifetime, so a plan can be kept for as long as the shape it was made for lasts (in a `Box<dyn Any>`, say).
///
/// **A plan must outlive the GPU work of every dispatch appended from it**: dropping it returns its partial buffers
/// to the device's pool, and a new owner may then write them while those dispatches still run. The plan may be
/// appended more than once (to other buffers of the same length, or the same ones again); the appended runs share the
/// partials, so two of them must not be in flight at the same time.
///
/// The result is one element and every run writes it whole, so it can be reused from run to run. An empty input has
/// no plan ([`Error::Empty`]); the value of the reduction over it is [`Op::identity`]. The wrapping sum has its own
/// plan, [`SumWrappingPlan`].
pub struct ReducePlan<T: Element> {
    parts: Parts<T>,
    op: Op,
    len: u32,
}

impl<T: Element> ReducePlan<T> {
    /// The plan of reduction `op` over `len` elements on `device`, with the kernels of `variant` (v1 falls back to
    /// v0 for an input its launch cannot take).
    ///
    /// # Errors
    ///
    /// `Error::Empty` for `len == 0`, `Error::Device` for a length past the device's storage binding limit or a
    /// failed allocation.
    pub fn new(device: &DeviceHandle, variant: Variant, op: Op, len: u32) -> Result<Self, Error> {
        non_empty(len)?;
        Ok(Self {
            parts: Parts::new(device, variant, len)?,
            op,
            len,
        })
    }

    /// The number of elements of the input the plan was made for.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// A plan is never for an empty input, so this is `false`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Append the dispatches of the reduction of `input` into the one-element `out` to `sink`.
    ///
    /// # Errors
    ///
    /// `Error::Length` when `input` does not hold the plan's length or `out` does not hold one element; nothing is
    /// appended then. An operand of another device is refused later, by the dispatch.
    pub fn append<'s, I: Operand<T>, O: Operand<T>>(
        &'s self,
        sink: &mut Sink<'s>,
        input: &'s I,
        out: &'s O,
    ) -> Result<(), Error> {
        check_len(self.len, input.operand_len())?;
        check_len(1, out.operand_len())?;
        self.parts.plan(self.op, sink, input, out);
        Ok(())
    }
}

/// The plan of the wrapping sum of a `u32` input (modulo 2^32, the `sum_wrapping` function): as [`ReducePlan`], with
/// the same lifetime rule, length checks and `Error::Empty` for an empty input. Its empty-input value is that of
/// [`Op::Sum`], `Some(0)`.
pub struct SumWrappingPlan {
    parts: Parts<u32>,
    len: u32,
}

impl SumWrappingPlan {
    /// The plan of the wrapping sum over `len` elements on `device`, with the kernels of `variant`.
    ///
    /// # Errors
    ///
    /// As for [`ReducePlan::new`].
    pub fn new(device: &DeviceHandle, variant: Variant, len: u32) -> Result<Self, Error> {
        non_empty(len)?;
        Ok(Self {
            parts: Parts::new(device, variant, len)?,
            len,
        })
    }

    /// The number of elements of the input the plan was made for.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// A plan is never for an empty input, so this is `false`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Append the dispatches of the wrapping sum of `input` into the one-element `out` to `sink`.
    ///
    /// # Errors
    ///
    /// As for [`ReducePlan::append`].
    pub fn append<'s, I: Operand<u32>, O: Operand<u32>>(
        &'s self,
        sink: &mut Sink<'s>,
        input: &'s I,
        out: &'s O,
    ) -> Result<(), Error> {
        check_len(self.len, input.operand_len())?;
        check_len(1, out.operand_len())?;
        self.parts.plan_wrapping(sink, input, out);
        Ok(())
    }
}

/// `Error::Empty` for a length of zero.
pub(crate) fn non_empty(len: u32) -> Result<(), Error> {
    if len == 0 { Err(Error::Empty) } else { Ok(()) }
}

/// `actual` is the length `expected`, as a plan or a recording needs.
pub(crate) fn check_len(expected: u32, actual: u32) -> Result<(), Error> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::Length { expected, actual })
    }
}

/// A recorded reduction of a non-empty `input` into the one-element `out`.
pub(crate) struct Reduce<'a, T: Element> {
    plan: ReducePlan<T>,
    input: &'a GpuVec<T>,
    out: &'a GpuVec<T>,
}

impl<'a, T: Element> Reduce<'a, T> {
    /// `Error::Empty` for an empty `input`, `Error::Length` for an `out` that is not one element.
    pub(crate) fn new(
        device: &DeviceHandle,
        variant: Variant,
        op: Op,
        input: &'a GpuVec<T>,
        out: &'a GpuVec<T>,
    ) -> Result<Self, Error> {
        let plan = ReducePlan::new(device, variant, op, input.len())?;
        // The same check the append makes: a list never holds an operand pair its plan would refuse.
        check_len(1, out.len())?;
        Ok(Self { plan, input, out })
    }
}

impl<T: Element> Stage for Reduce<'_, T> {
    fn prepare<'s>(&'s self, sink: &mut Sink<'s>) {
        self.plan
            .append(sink, self.input, self.out)
            .expect("the operands were checked when the plan was made");
    }
}

/// A recorded wrapping sum of a non-empty `u32` input into the one-element `out`.
pub(crate) struct SumWrapping<'a> {
    plan: SumWrappingPlan,
    input: &'a GpuVec<u32>,
    out: &'a GpuVec<u32>,
}

impl<'a> SumWrapping<'a> {
    /// `Error::Empty` for an empty `input`, `Error::Length` for an `out` that is not one element.
    pub(crate) fn new(
        device: &DeviceHandle,
        variant: Variant,
        input: &'a GpuVec<u32>,
        out: &'a GpuVec<u32>,
    ) -> Result<Self, Error> {
        let plan = SumWrappingPlan::new(device, variant, input.len())?;
        check_len(1, out.len())?;
        Ok(Self { plan, input, out })
    }
}

impl Stage for SumWrapping<'_> {
    fn prepare<'s>(&'s self, sink: &mut Sink<'s>) {
        self.plan
            .append(sink, self.input, self.out)
            .expect("the operands were checked when the plan was made");
    }
}
