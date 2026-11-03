//! The reductions and scans on `gust::cpu::CpuDevice`: the same schedules and the same kernels, run through their CPU
//! twins (D22, `docs/SEMANTICS.md`). Present in test and `--cfg gust_dual` builds only (D26): the twin of a kernel
//! exists there and in `GUST_PROFILE=cpu` builds, never in a GPU production build.
//!
//! A twin result is never GPU evidence. These functions are the twin assertion that sits beside the GPU assertion of
//! the same case, against the same independent reference.

use gust::{
    Independent,
    cpu::{CpuBuffer, CpuDevice, Error},
};

use crate::{
    compact_kernels, kernels,
    reduce::{Element, Op},
    scan_kernels,
    schedule::{Pass, ScanSchedule, Schedule},
};

/// The result of `op` over `values`, or `None` for an empty input (the identity applies to `sum`, which has its own
/// function below).
fn reduce<T: Element>(op: Op, values: &[T]) -> Result<Option<T>, Error> {
    reduce_with(values, |device, schedule, input, partials, out| {
        T::run_cpu(op, device, schedule, input, partials, out)
    })
}

/// The shared frame of a reduction on the twins: the buffers of the schedule, which `run` then fills.
fn reduce_with<T: Element>(
    values: &[T],
    run: impl FnOnce(
        &CpuDevice,
        &Schedule,
        &CpuBuffer<T>,
        &[CpuBuffer<T>],
        &CpuBuffer<T>,
    ) -> Result<(), Error>,
) -> Result<Option<T>, Error> {
    let Some(schedule) =
        Schedule::new(u32::try_from(values.len()).expect("at most u32::MAX elements"))
    else {
        return Ok(None);
    };
    let device = CpuDevice::new();
    let input = device.buffer(values);
    let partials: Vec<_> = schedule
        .passes()
        .iter()
        .map(|pass| device.buffer(&vec![T::ZERO; pass.partials() as usize]))
        .collect();
    let out = device.buffer(&[T::ZERO]);
    run(&device, &schedule, &input, &partials, &out)?;
    Ok(Some(out.read()[0]))
}

/// The sum of `values` on the CPU twins; the sum of an empty slice is `T::ZERO`.
pub fn sum<T: Element>(values: &[T]) -> Result<T, Error> {
    Ok(reduce(Op::Sum, values)?.unwrap_or(T::ZERO))
}

/// The smallest of `values` on the CPU twins; `None` for an empty slice.
pub fn min<T: Element>(values: &[T]) -> Result<Option<T>, Error> {
    reduce(Op::Min, values)
}

/// The largest of `values` on the CPU twins; `None` for an empty slice.
pub fn max<T: Element>(values: &[T]) -> Result<Option<T>, Error> {
    reduce(Op::Max, values)
}

/// The sum of `values` modulo 2^32 on the CPU twins of the `wrapping_add` kernels (Rust's own `wrapping_add`, so the
/// result is the same with overflow checks on and off); `0` for an empty slice.
pub fn sum_wrapping(values: &[u32]) -> Result<u32, Error> {
    let total = reduce_with(values, |device, schedule, input, partials, out| {
        let mut source = input;
        for (pass, destination) in schedule.passes().iter().zip(partials) {
            kernels::sum_wrapping_u32_pass::dispatch(
                device,
                pass.partials(),
                Independent(source),
                destination,
            )?;
            source = destination;
        }
        kernels::sum_wrapping_u32_final::dispatch(device, 1, Independent(source), out)
    })?;
    Ok(total.unwrap_or(0))
}

/// The inclusive prefix sums of `values` on the CPU twins of the scan kernels (the same chunks, levels and add-backs as
/// the device); empty for an empty slice. Plain `+`: a prefix past `u32::MAX` panics in a debug build.
pub fn inclusive(values: &[u32]) -> Result<Vec<u32>, Error> {
    scan(values, false)
}

/// The exclusive prefix sums of `values` on the CPU twins; empty for an empty slice.
pub fn exclusive(values: &[u32]) -> Result<Vec<u32>, Error> {
    scan(values, true)
}

fn scan(values: &[u32], exclusive: bool) -> Result<Vec<u32>, Error> {
    let Some(schedule) =
        ScanSchedule::new(u32::try_from(values.len()).expect("at most u32::MAX elements"))
    else {
        return Ok(Vec::new());
    };
    let device = CpuDevice::new();
    let input = device.buffer(values);
    let out = device.buffer(&vec![0u32; values.len()]);
    let lengths = schedule.sums();
    let sums: Vec<_> = lengths
        .iter()
        .map(|len| device.buffer(&vec![0u32; *len as usize]))
        .collect();
    let levels = sums.len();
    scan_kernels::scan_chunks_u32::dispatch(
        &device,
        lengths[0],
        Independent(&input),
        Independent(&out),
        &sums[0],
    )?;
    for level in 1..levels {
        scan_kernels::scan_sums_u32::dispatch(
            &device,
            lengths[level],
            Independent(&sums[level - 1]),
            &sums[level],
        )?;
    }
    scan_kernels::scan_top_u32::dispatch(&device, 1, Independent(&sums[levels - 1]))?;
    for level in (1..levels).rev() {
        scan_kernels::add_back_sums_u32::dispatch(
            &device,
            lengths[level - 1],
            Independent(&sums[level]),
            &sums[level - 1],
        )?;
    }
    if exclusive {
        scan_kernels::add_back_exclusive_u32::dispatch(
            &device,
            lengths[0],
            &sums[0],
            Independent(&input),
            Independent(&out),
        )?;
    } else {
        scan_kernels::add_back_inclusive_u32::dispatch(
            &device,
            lengths[0],
            &sums[0],
            Independent(&out),
        )?;
    }
    Ok(out.read())
}

/// The 0/1 mask of `flags` on the twin of `flags_to_mask_u32`: 1 where the flag is not zero.
fn mask(device: &CpuDevice, flags: &[u32]) -> Result<CpuBuffer<u32>, Error> {
    let input = device.buffer(flags);
    let mask = device.buffer(&vec![0u32; flags.len()]);
    compact_kernels::flags_to_mask_u32::dispatch(
        device,
        Pass::over(u32::try_from(flags.len()).expect("at most u32::MAX elements")).partials(),
        Independent(&input),
        Independent(&mask),
    )?;
    Ok(mask)
}

/// The number of non-zero `flags` on the twins: the mask, then the twin of the reduction.
pub fn compact_count(flags: &[u32]) -> Result<u32, Error> {
    if flags.is_empty() {
        return Ok(0);
    }
    sum(&mask(&CpuDevice::new(), flags)?.read())
}

/// `(indices, count)` of the non-zero `flags` on the twins, the same chain as `compact::indices`: the mask, the
/// exclusive scan, the count and the scatter. `indices` is `flags.len()` long, zero from `count` on.
pub fn compact_indices(flags: &[u32]) -> Result<(Vec<u32>, u32), Error> {
    if flags.is_empty() {
        return Ok((Vec::new(), 0));
    }
    let device = CpuDevice::new();
    let mask = mask(&device, flags)?;
    let ones = mask.read();
    let positions = device.buffer(&exclusive(&ones)?);
    let total = sum(&ones)?;
    let indices = device.buffer(&vec![0u32; flags.len()]);
    compact_kernels::scatter_indices_u32::dispatch(
        &device,
        Pass::over(u32::try_from(flags.len()).expect("at most u32::MAX elements")).partials(),
        Independent(&mask),
        Independent(&positions),
        Independent(&indices),
    )?;
    Ok((indices.read(), total))
}

macro_rules! twin_gather {
    ($(#[$meta:meta])* $name:ident, $ty:ty, $zero:expr) => {
        $(#[$meta])*
        pub fn $name(values: &[$ty], indices: &[u32], count: u32) -> Result<Vec<$ty>, Error> {
            let device = CpuDevice::new();
            let out = device.buffer(&vec![$zero; indices.len()]);
            if !indices.is_empty() && !values.is_empty() {
                let len = u32::try_from(indices.len()).expect("at most u32::MAX elements");
                let invocations = Pass::over(len).partials();
                let (values, indices, count) = (
                    device.buffer(values),
                    device.buffer(indices),
                    device.buffer(&[count]),
                );
                compact_kernels::$name::dispatch(
                    &device,
                    invocations,
                    Independent(&values),
                    Independent(&indices),
                    Independent(&count),
                    Independent(&out),
                )?;
            }
            Ok(out.read())
        }
    };
}

twin_gather!(
    /// `out[j] = values[indices[j]]` for `j < count` on the twin of `gather_u32`; `out` is `indices.len()` long, zero
    /// elsewhere.
    gather_u32,
    u32,
    0u32
);
twin_gather!(
    /// As [`gather_u32`], for `i32`.
    gather_i32,
    i32,
    0i32
);
twin_gather!(
    /// As [`gather_u32`], for `f32`.
    gather_f32,
    f32,
    0.0f32
);
