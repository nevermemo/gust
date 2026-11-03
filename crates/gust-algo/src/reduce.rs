//! `sum`, `min`, `max` of a [`GpuVec`] of `u32`, `i32` or `f32` and `sum_wrapping` of `u32` (plan section 5.2).
//!
//! One reduction is one `dispatch_batch` of at most four dispatches (the block-strided passes of
//! [`Schedule`], then the terminal kernel), with one pooled `GpuVec` per pass for the partials and one element for the
//! result, all taken from the device's buffer pool and returned to it on drop. Every input binding is `Independent`
//! (its length differs from the invocation count), every output binding shares the invocation count.
//!
//! The contract (section 5.1): the sum of an empty input is the identity (`0`, `0.0`); `min` and `max` of an empty
//! input are [`Error::Empty`]. Integer sums use plain `+` and are exact and in the domain only when no partial sum
//! overflows (the kernels wrap on the GPU, the debug twin panics, as every `+` of the dialect). `min` and `max` are
//! exact. Float sums are within [`crate::bound::f32_sum_bound`] of the exact sum. An input larger than the device's
//! storage binding limit is `Error::Device(DeviceLimit)` before any buffer is taken or dispatch is made.

use gust::{GpuPod, Independent};
use gust_wgpu::{BufferDispatch, GpuVec, HeadlessDevice};

use crate::{Error, kernels, limits, schedule::Schedule};

/// The reduction a call performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Sum,
    Min,
    Max,
}

mod sealed {
    pub trait Sealed {}
}

/// The element types of [`sum`], [`min`] and [`max`]: `u32`, `i32` and `f32`.
pub trait Element: GpuPod + sealed::Sealed + Copy + PartialOrd + 'static {
    /// The additive identity, the sum of an empty input.
    const ZERO: Self;

    /// Run the passes of `schedule` for `op` over `input` as one batch: pass `j` reads the previous level and writes
    /// `partials[j]`, the terminal kernel writes `out[0]`.
    #[doc(hidden)]
    fn run(
        op: Op,
        device: &HeadlessDevice,
        schedule: &Schedule,
        input: &GpuVec<Self>,
        partials: &[GpuVec<Self>],
        out: &GpuVec<Self>,
    ) -> Result<(), gust_wgpu::Error>;

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

/// The passes of one `(op, type)` pair as typed calls in one batch.
macro_rules! gpu_chain {
    ($pass:ident, $terminal:ident, $device:expr, $schedule:expr, $input:expr, $partials:expr, $out:expr) => {{
        let mut calls = Vec::with_capacity($schedule.passes().len());
        let mut source = $input;
        for (pass, destination) in $schedule.passes().iter().zip($partials) {
            calls.push(kernels::$pass::call(
                pass.partials(),
                Independent(source),
                destination,
            ));
            source = destination;
        }
        let last = kernels::$terminal::call(1, Independent(source), $out);
        let mut batch: Vec<BufferDispatch<'_>> = calls.iter().map(BufferDispatch::from).collect();
        batch.push(BufferDispatch::from(&last));
        $device.dispatch_batch(&batch)
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
        max: ($max_pass:ident, $max_final:ident)
    ) => {
        impl sealed::Sealed for $ty {}

        impl Element for $ty {
            const ZERO: Self = $zero;

            fn run(
                op: Op,
                device: &HeadlessDevice,
                schedule: &Schedule,
                input: &GpuVec<Self>,
                partials: &[GpuVec<Self>],
                out: &GpuVec<Self>,
            ) -> Result<(), gust_wgpu::Error> {
                match op {
                    Op::Sum => gpu_chain!(
                        $sum_pass, $sum_final, device, schedule, input, partials, out
                    ),
                    Op::Min => gpu_chain!(
                        $min_pass, $min_final, device, schedule, input, partials, out
                    ),
                    Op::Max => gpu_chain!(
                        $max_pass, $max_final, device, schedule, input, partials, out
                    ),
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
    max: (max_u32_pass, max_u32_final)
);
element!(
    i32, 0,
    sum: (sum_i32_pass, sum_i32_final),
    min: (min_i32_pass, min_i32_final),
    max: (max_i32_pass, max_i32_final)
);
element!(
    f32, 0.0,
    sum: (sum_f32_pass, sum_f32_final),
    min: (min_f32_pass, min_f32_final),
    max: (max_f32_pass, max_f32_final)
);

/// The sum of all elements, read back to the host. The sum of an empty input is `0`.
///
/// # Errors
///
/// `Error::Device` for a device failure or an input past the storage binding limit.
pub fn sum<T: Element>(input: &GpuVec<T>) -> Result<T, Error> {
    if input.is_empty() {
        return Ok(T::ZERO);
    }
    read_one(&reduce_on_device(Op::Sum, input)?)
}

/// The smallest element. [`Error::Empty`] for an empty input.
///
/// # Errors
///
/// `Error::Empty`, or `Error::Device` as for [`sum`].
pub fn min<T: Element>(input: &GpuVec<T>) -> Result<T, Error> {
    if input.is_empty() {
        return Err(Error::Empty);
    }
    read_one(&reduce_on_device(Op::Min, input)?)
}

/// The largest element. [`Error::Empty`] for an empty input.
///
/// # Errors
///
/// `Error::Empty`, or `Error::Device` as for [`sum`].
pub fn max<T: Element>(input: &GpuVec<T>) -> Result<T, Error> {
    if input.is_empty() {
        return Err(Error::Empty);
    }
    read_one(&reduce_on_device(Op::Max, input)?)
}

/// The sum as a one-element `GpuVec` on the device of `input`, with no readback (for `compact::count` of m7.r3). The
/// sum of an empty input is a vector holding the identity.
///
/// # Errors
///
/// As for [`sum`].
pub fn sum_on_device<T: Element>(input: &GpuVec<T>) -> Result<GpuVec<T>, Error> {
    if input.is_empty() {
        return Ok(GpuVec::zeroed_on(input.device(), 1)?);
    }
    reduce_on_device(Op::Sum, input)
}

fn read_one<T: Element>(result: &GpuVec<T>) -> Result<T, Error> {
    let values = result.to_vec()?;
    Ok(values[0])
}

/// One reduction of a non-empty input: the limit checks, the pooled buffers, one `dispatch_batch`.
fn reduce_on_device<T: Element>(op: Op, input: &GpuVec<T>) -> Result<GpuVec<T>, Error> {
    reduce_with(input, |device, schedule, partials, out| {
        T::run(op, device, schedule, input, partials, out)
    })
}

/// The shared frame of a reduction of a non-empty `input`: the limit checks, then one pooled `GpuVec` per pass and one
/// for the result, which `run` fills with its batch.
fn reduce_with<T: Element>(
    input: &GpuVec<T>,
    run: impl FnOnce(
        &HeadlessDevice,
        &Schedule,
        &[GpuVec<T>],
        &GpuVec<T>,
    ) -> Result<(), gust_wgpu::Error>,
) -> Result<GpuVec<T>, Error> {
    let schedule = Schedule::new(input.len()).expect("the input is not empty");
    let device = input.device();
    // A first pass past the workgroup limit is refused here too: it is the largest dispatch of the batch.
    let workgroups = schedule.passes().first().map_or(1, |pass| pass.workgroups);
    limits::check(
        device,
        "gust_algo::reduce",
        workgroups,
        input.len(),
        size_of::<T>(),
    )?;
    // Every lane writes its slot, so the zero fill is not needed for correctness; `GpuVec` has no public constructor
    // without it (a recycled buffer costs one clearing submission), and `gust-wgpu` is not this slice's to change.
    let partials = schedule
        .passes()
        .iter()
        .map(|pass| GpuVec::<T>::zeroed_on(device, pass.partials()))
        .collect::<Result<Vec<_>, _>>()?;
    let out = GpuVec::<T>::zeroed_on(device, 1)?;
    run(device, &schedule, &partials, &out)?;
    Ok(out)
}

/// The sum of all elements of a `u32` vector, defined modulo 2^32 for every input: the lane chains and the partials
/// wrap (`wrapping_add`, SEMANTICS section 6.2). The sum of an empty input is `0`.
///
/// This is `sum` for inputs whose total may pass `u32::MAX`; for inputs that stay inside, both give the same value.
/// The twin is `cpu::sum_wrapping` and calls Rust's `wrapping_add`, so the value is the same on the device, on the twin
/// and in plain Rust, with overflow checks on and off.
///
/// # Errors
///
/// As for [`sum`].
pub fn sum_wrapping(input: &GpuVec<u32>) -> Result<u32, Error> {
    if input.is_empty() {
        return Ok(0);
    }
    let result = reduce_with(input, |device, schedule, partials, out| {
        gpu_chain!(
            sum_wrapping_u32_pass,
            sum_wrapping_u32_final,
            device,
            schedule,
            input,
            partials,
            out
        )
    })?;
    read_one(&result)
}
