//! Which kernels a public function runs (plan section 5.5, slice m7.r4).
//!
//! v0 is the block-strided form of m7.r1 and m7.r2: it has CPU twins and runs wherever the dialect runs. v1 is the
//! shared-memory form (`Workgroup<[T; 256]>` and barriers), whose kernels are `cpu = "unsupported"` (D101). Both give
//! the same results, integers bit for bit and floats within the bound of [`crate::bound`]; they differ in the
//! dispatches they take. The public functions of [`reduce`](crate::reduce), [`scan`](crate::scan) and
//! [`compact`](crate::compact) run the variant [`shipped`] names for their operation: v1 only where m7.r4 measured it
//! faster than v0 on the RTX 5090 by at least 10 % at 1e6, 1e7 and 1e8 elements, v0 everywhere else. The `*_with`
//! functions of `reduce` and `scan` run either on request (the measurement, the tests).
//!
//! A v1 request falls back to v0 for an input its launch cannot take: a first dispatch past the device's workgroup
//! limit (more than 65 535 spans of 4096 scan elements, or of 8192 reduce elements, on the usual device) or a
//! binding past the storage limit. v0 then answers, with its own `DeviceLimit` error where it also cannot run.

/// The kernels a call runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// The block-strided kernels with CPU twins.
    V0,
    /// The shared-memory kernels with barriers (no CPU twin).
    V1,
}

/// The operations that have both a v0 and a v1 form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    /// `reduce::sum` and `reduce::sum_on_device` (and with them `compact::count`), for `u32`, `i32` and `f32`.
    Sum,
    /// `reduce::min`.
    Min,
    /// `reduce::max`.
    Max,
    /// `reduce::sum_wrapping`.
    SumWrapping,
    /// `scan::inclusive` (and with it `compact::indices`).
    Inclusive,
    /// `scan::exclusive` (and with it `compact::indices`).
    Exclusive,
}

/// Every [`Operation`], in the order of the measurement table.
pub const OPERATIONS: [Operation; 6] = [
    Operation::Sum,
    Operation::Min,
    Operation::Max,
    Operation::SumWrapping,
    Operation::Inclusive,
    Operation::Exclusive,
];

/// The variant the public function of `operation` runs: the outcome of the m7.r4 measurement on the RTX 5090
/// (the log is `plans/m7/probe/r4` of the work area, the summary is in the slice record). The rule is plan section 5.5:
/// v1 ships for an operation only where it is at least 10 % faster at every measured size.
pub const fn shipped(operation: Operation) -> Variant {
    match operation {
        Operation::Sum
        | Operation::Min
        | Operation::Max
        | Operation::SumWrapping
        | Operation::Inclusive
        | Operation::Exclusive => Variant::V1,
    }
}
