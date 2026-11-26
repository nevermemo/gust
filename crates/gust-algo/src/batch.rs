//! The reductions, the scans and the compaction as ops of a `gust::batch` (slice m14.s2f2a, D259 and the s2f2 ruling).
//!
//! Each op type here implements `gust::batch::Op` for the GPU device and for the CPU device (the orphan rule forbids
//! an impl on a device the other way round). A build block records one with `b.op(&["total"], Sum::of(&blocks))`; the
//! forms users write (`let total = gust::sum(&blocks);`) arrive with the macro (slice s2f2c). Everything here is macro
//! support, outside the pre-1.0 promise (D111).
//!
//! **The GPU form** builds an owned plan of this crate (`ReducePlan`, `SumWrappingPlan`, `ScanPlan`, `CountPlan`,
//! `CompactPlan`) for the sealed length inside `Op::record`, in the build block, and prepares the stage over it and
//! the operands (`HeadlessDevice::prepare_list`); the batch keeps the plan, not the operands. Every clearing submission
//! of the plan's pooled scratch happens before the seal, or inside the `set` that resizes, so a frame is one
//! submission (D251). Each op hands `OpDraft::seal_planned` two functions: `plan(&device,
//! len)`, which makes the plan of a length (the part that costs), and `bind(&device, &plan, &operands)`, which
//! prepares the stage over that plan and the buffers of one frame. A resize (slice s2b2) calls them again inside
//! `FrameRun::set`: the plan only for a length the batch has not kept a plan for, `bind` always, over the new buffers.
//! A plan holds none of its operands, so the plan of a length serves any buffers of that length, and the outputs of a
//! scan or a compaction (made with `OpDraft::result_like`) can be replaced by a resize.
//!
//! **The CPU form** is plain Rust over the buffers, compiled in every build: wrapping integer folds, a float sum
//! accumulated in `f64` and rounded once, prefix sums, and the positions of the set flags with the tail left alone. It
//! is not the kernels' twin (those exist only in test and dual builds, D26); a test holds it equal to the twin chain
//! of the `cpu` module (test and dual builds) and to an independent reference (integers exact, `f32` sums inside the
//! bound).
//!
//! **Over zero elements** the core applies the `Empty` rule each op hands over, in every frame and before anything is
//! encoded: `sum`, `sum_wrapping` and `count` write their identity (0), `scan` records nothing, `compact` writes a
//! count of 0 (its indices are empty), and `min` and `max` have no value there, so `submit` and `encode` return
//! `Error::Empty` naming the stage. The outputs of an op are new buffers that no frame can `write`
//! (`Error::NotInput`), and a later stage that names one as its in-place target is refused when the batch is built
//! (`Error::Overwrite`).

use std::sync::Arc;

use gust::GpuPod;
use gust::batch::{BatchDevice, D1, Empty, Op as BatchOp, OpDraft, Stage};
use gust::cpu::{CpuBuffer, CpuDevice};

use crate::reduce::{Element, Op};

/// The element types of the reductions of a batch: `u32`, `i32` and `f32` (those of [`Element`]), with the host form of
/// their sum. Macro support.
pub trait Host: Element {
    /// The sum of `values` on the host: modulo 2^32 for the integers, and for `f32` the `f64` accumulation rounded once
    /// (inside every bound of [`bound`](crate::bound), whatever the length).
    fn sum(values: &[Self]) -> Self;
}

impl Host for u32 {
    fn sum(values: &[Self]) -> Self {
        values.iter().fold(0, |sum, value| sum.wrapping_add(*value))
    }
}

impl Host for i32 {
    fn sum(values: &[Self]) -> Self {
        values.iter().fold(0, |sum, value| sum.wrapping_add(*value))
    }
}

impl Host for f32 {
    fn sum(values: &[Self]) -> Self {
        values.iter().map(|value| f64::from(*value)).sum::<f64>() as Self
    }
}

/// The reduction `op` of a handle: a stage of length 1.
pub struct Reduce<'a, T: Host> {
    op: Op,
    input: &'a Stage<T, D1>,
}

impl<'a, T: Host> Reduce<'a, T> {
    /// The reduction `op` of `input`.
    #[must_use]
    pub fn of(op: Op, input: &'a Stage<T, D1>) -> Self {
        Self { op, input }
    }
}

macro_rules! reduction {
    ($(#[$meta:meta])* $name:ident, $op:expr) => {
        $(#[$meta])*
        pub struct $name<'a, T: Host>(Reduce<'a, T>);

        impl<'a, T: Host> $name<'a, T> {
            /// The reduction of `input`.
            #[must_use]
            pub fn of(input: &'a Stage<T, D1>) -> Self {
                Self(Reduce::of($op, input))
            }
        }

        impl<'a, D: BatchDevice, T: Host> BatchOp<D> for $name<'a, T>
        where
            Reduce<'a, T>: BatchOp<D>,
        {
            type Out = <Reduce<'a, T> as BatchOp<D>>::Out;

            fn record(self, draft: OpDraft<'_, D>) -> Self::Out {
                self.0.record(draft)
            }
        }
    };
}

reduction!(
    /// The sum of a handle (`gust::sum`).
    Sum,
    Op::Sum
);
reduction!(
    /// The minimum of a handle (`gust::min`).
    Min,
    Op::Min
);
reduction!(
    /// The maximum of a handle (`gust::max`).
    Max,
    Op::Max
);

/// The sum of a `u32` handle modulo 2^32 (`gust::sum_wrapping`).
pub struct SumWrapping<'a> {
    input: &'a Stage<u32, D1>,
}

impl<'a> SumWrapping<'a> {
    /// The wrapping sum of `input`.
    #[must_use]
    pub fn of(input: &'a Stage<u32, D1>) -> Self {
        Self { input }
    }
}

/// The number of set (non-zero) flags of a `u32` handle: a stage of length 1 (`gust::count`).
pub struct Count<'a> {
    flags: &'a Stage<u32, D1>,
}

impl<'a> Count<'a> {
    /// The count of the set flags of `flags`.
    #[must_use]
    pub fn of(flags: &'a Stage<u32, D1>) -> Self {
        Self { flags }
    }
}

/// The prefix sums of a `u32` handle, inclusive or exclusive: a stage of the operand's length (`gust::scan` and
/// `gust::scan_exclusive`).
pub struct Scan<'a> {
    input: &'a Stage<u32, D1>,
    exclusive: bool,
}

impl<'a> Scan<'a> {
    /// The inclusive scan of `input`.
    #[must_use]
    pub fn of(input: &'a Stage<u32, D1>) -> Self {
        Self {
            input,
            exclusive: false,
        }
    }

    /// The exclusive scan of `input`.
    #[must_use]
    pub fn exclusive(input: &'a Stage<u32, D1>) -> Self {
        Self {
            input,
            exclusive: true,
        }
    }
}

/// The compaction of a `u32` handle of flags: `(indices, count)`, the positions of the set flags (as long as the flags;
/// only the first `count` are meaningful, D259) and their number (`gust::compact`).
pub struct Compact<'a> {
    flags: &'a Stage<u32, D1>,
}

impl<'a> Compact<'a> {
    /// The compaction of `flags`.
    #[must_use]
    pub fn of(flags: &'a Stage<u32, D1>) -> Self {
        Self { flags }
    }
}

/// What a stage over zero elements writes into `out`: `value`, through the frame's writes.
fn identity<D: BatchDevice, T: GpuPod>(out: &Arc<D::Buffer<T>>, value: T) -> Empty<D>
where
    D::Buffer<T>: 'static,
{
    let out = Arc::clone(out);
    Empty::Identity(Box::new(move |device, frame| {
        device.frame_write(frame, &out, &[value])
    }))
}

#[cfg(not(target_arch = "wasm32"))]
mod gpu {
    use gust_wgpu::{DeviceHandle, HeadlessDevice};

    use super::*;
    use crate::{
        Error,
        compact::{CompactPlan, CountPlan},
        record::Sink,
        reduce::{ReducePlan, SumWrappingPlan},
        scan::ScanPlan,
        variant::{Operation, shipped},
    };

    // ---------------------------------------------------------------------------------------------------------------
    // The GPU form.

    /// The stage a GPU op prepares: the dispatches `append` puts in a sink over the plan and the operands, prepared once.
    /// The plan (shared with the batch's kept plans, which run in the order of the queue) and the operands move into
    /// the stage, which holds them for as long as it lives (the plan's pooled scratch goes back to the pool when its
    /// last owner drops).
    fn gpu_stage<P: 'static, O: 'static>(
        device: &DeviceHandle,
        plan: &Arc<P>,
        operands: O,
        append: impl for<'s> FnOnce(&'s P, &'s O, &mut Sink<'s>) -> Result<(), Error>,
    ) -> Result<<HeadlessDevice as BatchDevice>::Prepared, gust_wgpu::Error> {
        let held = Arc::new((Arc::clone(plan), operands));
        let mut sink = Sink::new();
        let (plan, operands) = &*held;
        append(&**plan, operands, &mut sink).map_err(device_error)?;
        sink.with(|dispatches| device.prepare_list(Arc::clone(&held), dispatches))
    }

    /// The device's error of a plan or an append. The batch finds or builds a plan by the length of the op's first
    /// operand, so the plan and its operands agree by construction; a length mismatch of an operand (an op that reads
    /// several, resized apart) is still returned as `BufferLength` rather than a panic.
    fn device_error(error: Error) -> gust_wgpu::Error {
        let mismatch = |expected: usize, actual: usize| gust_wgpu::Error::BufferLength {
            parameter: String::from("operand of a batch op"),
            expected,
            actual,
        };
        match error {
            Error::Device(error) => error,
            Error::Length { expected, actual } => mismatch(expected as usize, actual as usize),
            Error::Count(actual) => mismatch(1, actual as usize),
            Error::Empty => mismatch(1, 0),
        }
    }

    fn length(len: usize) -> u32 {
        u32::try_from(len).expect("a batch buffer holds at most u32::MAX elements")
    }

    impl<T: Host> BatchOp<HeadlessDevice> for Reduce<'_, T> {
        type Out = Stage<T, D1>;

        fn record(self, mut draft: OpDraft<'_, HeadlessDevice>) -> Self::Out {
            if draft.read(self.input).is_none() {
                return draft.abandon();
            }
            let Some(out) = draft.result::<T>(1) else {
                return draft.abandon();
            };
            let op = self.op;
            let empty = match op.identity::<T>() {
                Some(zero) => identity(&out, zero),
                None => Empty::Refuse,
            };
            let device = draft.device();
            let bound = Arc::clone(&device);
            draft.seal_planned(
                move |_, len| {
                    ReducePlan::<T>::new(&device, shipped(op.operation()), op, length(len))
                        .map_err(device_error)
                },
                move |_, plan, operands| {
                    let operands = (operands.read::<T>(0), operands.result::<T>(0));
                    gpu_stage(&bound, plan, operands, |plan, (input, out), sink| {
                        plan.append(sink, &**input, &**out)
                    })
                },
                empty,
            )
        }
    }

    impl BatchOp<HeadlessDevice> for SumWrapping<'_> {
        type Out = Stage<u32, D1>;

        fn record(self, mut draft: OpDraft<'_, HeadlessDevice>) -> Self::Out {
            if draft.read(self.input).is_none() {
                return draft.abandon();
            }
            let Some(out) = draft.result::<u32>(1) else {
                return draft.abandon();
            };
            let empty = identity(&out, 0);
            let device = draft.device();
            let bound = Arc::clone(&device);
            draft.seal_planned(
                move |_, len| {
                    SumWrappingPlan::new(&device, shipped(Operation::SumWrapping), length(len))
                        .map_err(device_error)
                },
                move |_, plan, operands| {
                    let operands = (operands.read::<u32>(0), operands.result::<u32>(0));
                    gpu_stage(&bound, plan, operands, |plan, (input, out), sink| {
                        plan.append(sink, &**input, &**out)
                    })
                },
                empty,
            )
        }
    }

    impl BatchOp<HeadlessDevice> for Count<'_> {
        type Out = Stage<u32, D1>;

        fn record(self, mut draft: OpDraft<'_, HeadlessDevice>) -> Self::Out {
            if draft.read(self.flags).is_none() {
                return draft.abandon();
            }
            let Some(count) = draft.result::<u32>(1) else {
                return draft.abandon();
            };
            let empty = identity(&count, 0);
            let device = draft.device();
            let bound = Arc::clone(&device);
            draft.seal_planned(
                move |_, len| CountPlan::new(&device, None, length(len)).map_err(device_error),
                move |_, plan, operands| {
                    let operands = (operands.read::<u32>(0), operands.result::<u32>(0));
                    gpu_stage(&bound, plan, operands, |plan, (flags, count), sink| {
                        plan.append(sink, &**flags, &**count)
                    })
                },
                empty,
            )
        }
    }

    impl BatchOp<HeadlessDevice> for Scan<'_> {
        type Out = Stage<u32, D1>;

        fn record(self, mut draft: OpDraft<'_, HeadlessDevice>) -> Self::Out {
            if draft.read(self.input).is_none() || draft.result_like::<u32>().is_none() {
                return draft.abandon();
            }
            let exclusive = self.exclusive;
            let operation = if exclusive {
                Operation::Exclusive
            } else {
                Operation::Inclusive
            };
            let device = draft.device();
            let bound = Arc::clone(&device);
            draft.seal_planned(
                move |_, len| {
                    ScanPlan::new(&device, shipped(operation), length(len), exclusive)
                        .map_err(device_error)
                },
                move |_, plan, operands| {
                    let operands = (operands.read::<u32>(0), operands.result::<u32>(0));
                    gpu_stage(&bound, plan, operands, |plan, (input, out), sink| {
                        plan.append(sink, &**input, &**out)
                    })
                },
                Empty::Nothing,
            )
        }
    }

    impl BatchOp<HeadlessDevice> for Compact<'_> {
        type Out = (Stage<u32, D1>, Stage<u32, D1>);

        fn record(self, mut draft: OpDraft<'_, HeadlessDevice>) -> Self::Out {
            if draft.read(self.flags).is_none() || draft.result_like::<u32>().is_none() {
                return draft.abandon();
            }
            let Some(count) = draft.result::<u32>(1) else {
                return draft.abandon();
            };
            // At length 0 the indices are empty and the count is 0: the count is written, the indices are not.
            let empty = identity(&count, 0);
            let device = draft.device();
            let bound = Arc::clone(&device);
            draft.seal_planned(
                move |_, len| CompactPlan::new(&device, None, length(len)).map_err(device_error),
                move |_, plan, operands| {
                    let operands = (
                        operands.read::<u32>(0),
                        operands.result::<u32>(0),
                        operands.result::<u32>(1),
                    );
                    gpu_stage(
                        &bound,
                        plan,
                        operands,
                        |plan, (flags, indices, count), sink| {
                            plan.append(sink, &**flags, &**indices, &**count)
                        },
                    )
                },
                empty,
            )
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The CPU form.

/// The reduction `op` of `values` on the host: not empty, because a stage over zero elements never runs.
fn host_reduce<T: Host>(op: Op, values: &[T]) -> T {
    match op {
        Op::Sum => T::sum(values),
        Op::Min => values
            .iter()
            .copied()
            .reduce(|least, value| if value < least { value } else { least })
            .expect("a reduction is sealed over at least one element"),
        Op::Max => values
            .iter()
            .copied()
            .reduce(|most, value| if value > most { value } else { most })
            .expect("a reduction is sealed over at least one element"),
    }
}

/// The prefix sums of `values` modulo 2^32.
fn host_scan(values: &[u32], exclusive: bool) -> Vec<u32> {
    let mut sum = 0u32;
    values
        .iter()
        .map(|value| {
            let before = sum;
            sum = sum.wrapping_add(*value);
            if exclusive { before } else { sum }
        })
        .collect()
}

/// The CPU stage that writes `f(input)` into the one-element `out`.
fn cpu_fold<T: GpuPod, U: GpuPod>(
    device: &CpuDevice,
    input: &Arc<CpuBuffer<T>>,
    out: &Arc<CpuBuffer<U>>,
    f: impl Fn(&[T]) -> U + 'static,
) -> <CpuDevice as BatchDevice>::Prepared {
    let (input, out) = (Arc::clone(input), Arc::clone(out));
    device.prepare_fn(move |_| out.write(&[f(&input.read())]))
}

impl<T: Host> BatchOp<CpuDevice> for Reduce<'_, T> {
    type Out = Stage<T, D1>;

    fn record(self, mut draft: OpDraft<'_, CpuDevice>) -> Self::Out {
        if draft.read(self.input).is_none() {
            return draft.abandon();
        }
        let Some(out) = draft.result::<T>(1) else {
            return draft.abandon();
        };
        let op = self.op;
        let empty = match op.identity::<T>() {
            Some(zero) => identity(&out, zero),
            None => Empty::Refuse,
        };
        draft.seal(
            move |device, operands| {
                Ok(cpu_fold(
                    device,
                    &operands.read::<T>(0),
                    &operands.result::<T>(0),
                    move |values| host_reduce(op, values),
                ))
            },
            empty,
        )
    }
}

impl BatchOp<CpuDevice> for SumWrapping<'_> {
    type Out = Stage<u32, D1>;

    fn record(self, mut draft: OpDraft<'_, CpuDevice>) -> Self::Out {
        if draft.read(self.input).is_none() {
            return draft.abandon();
        }
        let Some(out) = draft.result::<u32>(1) else {
            return draft.abandon();
        };
        let empty = identity(&out, 0);
        draft.seal(
            |device, operands| {
                Ok(cpu_fold(
                    device,
                    &operands.read::<u32>(0),
                    &operands.result::<u32>(0),
                    <u32 as Host>::sum,
                ))
            },
            empty,
        )
    }
}

impl BatchOp<CpuDevice> for Count<'_> {
    type Out = Stage<u32, D1>;

    fn record(self, mut draft: OpDraft<'_, CpuDevice>) -> Self::Out {
        if draft.read(self.flags).is_none() {
            return draft.abandon();
        }
        let Some(count) = draft.result::<u32>(1) else {
            return draft.abandon();
        };
        let empty = identity(&count, 0);
        draft.seal(
            |device, operands| {
                Ok(cpu_fold(
                    device,
                    &operands.read::<u32>(0),
                    &operands.result::<u32>(0),
                    |flags| {
                        u32::try_from(flags.iter().filter(|flag| **flag != 0).count())
                            .expect("a batch buffer holds at most u32::MAX elements")
                    },
                ))
            },
            empty,
        )
    }
}

impl BatchOp<CpuDevice> for Scan<'_> {
    type Out = Stage<u32, D1>;

    fn record(self, mut draft: OpDraft<'_, CpuDevice>) -> Self::Out {
        if draft.read(self.input).is_none() || draft.result_like::<u32>().is_none() {
            return draft.abandon();
        }
        let exclusive = self.exclusive;
        draft.seal(
            move |device, operands| {
                let (input, out) = (operands.read::<u32>(0), operands.result::<u32>(0));
                Ok(device.prepare_fn(move |_| out.write(&host_scan(&input.read(), exclusive))))
            },
            Empty::Nothing,
        )
    }
}

impl BatchOp<CpuDevice> for Compact<'_> {
    type Out = (Stage<u32, D1>, Stage<u32, D1>);

    fn record(self, mut draft: OpDraft<'_, CpuDevice>) -> Self::Out {
        if draft.read(self.flags).is_none() || draft.result_like::<u32>().is_none() {
            return draft.abandon();
        }
        let Some(count) = draft.result::<u32>(1) else {
            return draft.abandon();
        };
        let empty = identity(&count, 0);
        draft.seal(
            |device, operands| {
                let (flags, indices, count) = (
                    operands.read::<u32>(0),
                    operands.result::<u32>(0),
                    operands.result::<u32>(1),
                );
                Ok(device.prepare_fn(move |_| {
                    // The tail past the count keeps what the buffer held, as the kernels' scatter leaves it (D259).
                    let mut positions = indices.read();
                    let mut found = 0;
                    for (position, flag) in flags.read().iter().enumerate() {
                        if *flag != 0 {
                            positions[found] =
                                u32::try_from(position).expect("a position fits the u32 length");
                            found += 1;
                        }
                    }
                    indices.write(&positions)?;
                    count.write(&[u32::try_from(found).expect("a count fits the u32 length")])
                }))
            },
            empty,
        )
    }
}
