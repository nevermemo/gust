//! The reductions on `gust::cpu::CpuDevice`: the same schedule and the same kernels, run through their CPU twins (D22,
//! `docs/SEMANTICS.md`). Present in test and `--cfg gust_dual` builds only (D26): the twin of a kernel exists there
//! and in `GUST_PROFILE=cpu` builds, never in a GPU production build.
//!
//! A twin result is never GPU evidence. These functions are the twin assertion that sits beside the GPU assertion of
//! the same case, against the same independent reference.

use gust::cpu::{CpuDevice, Error};

use crate::{
    reduce::{Element, Op},
    schedule::Schedule,
};

/// The result of `op` over `values`, or `None` for an empty input (the identity applies to `sum`, which has its own
/// function below).
fn reduce<T: Element>(op: Op, values: &[T]) -> Result<Option<T>, Error> {
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
    T::run_cpu(op, &device, &schedule, &input, &partials, &out)?;
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
