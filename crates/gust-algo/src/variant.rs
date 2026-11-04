//! Which kernels a public function runs (plan section 5.5, slice m7.r4).
//!
//! v0 is the block-strided form of m7.r1 and m7.r2: it has CPU twins and runs wherever the dialect runs. v1 is the
//! shared-memory form (`Workgroup<[T; 256]>` and barriers), whose kernels are `cpu = "unsupported"` (D101). Both give
//! the same results, integers bit for bit and floats within the bound of [`crate::bound`]; they differ in the
//! dispatches they take. The public functions of [`reduce`](crate::reduce), [`scan`](crate::scan) and
//! [`compact`](crate::compact) run the variant [`shipped`] names for their operation. The rule is the plan's (section
//! 5.5): v1 ships for an operation only where it is faster than v0 on both adapters at 1e6, 1e7 and 1e8 elements, and
//! v0 stays everywhere else. The `*_with` functions of `reduce` and `scan` run either on request (the measurement, the
//! tests).
//!
//! The choice is made (D121); both halves of the measurement ran. On the RTX 5090 v1 was at least 10 % ahead for all
//! six operations; that margin was the slice brief's and the 5090 half used it. On the Radeon 890M v1 was faster for
//! every operation and size, but the reductions are only 1.01 to 1.03 ahead at 1e8, where both forms are bound by the
//! shared system memory, so the 10 % margin is not met there. D121 does not apply it there, because v0 is faster
//! nowhere, and keeps v1. The statistic is the median over runs of the per-run median ratio (v0 / v1, interleaved, end
//! to end), chosen after the runs and not pre-registered. On the 5090 the lowest single run was `min_f32` at 1e8 with
//! 1.085, and the next ones were `sum_i32` (1.120) and `max_f32` (1.137) at 1e8. At 1e6 and 1e7 a reduction costs
//! about 600 microseconds a call on the 5090 and the memory traffic only tens, so v1's win there is mostly per-call
//! overhead (one dispatch fewer and one pooled zeroing fewer); re-measure if the pool's zeroing changes. The logs and
//! notes are in `plans/m7/probe/r4` of the work area.
//!
//! A v1 request falls back to v0 for an input or a device its launch cannot take: a device with less workgroup
//! memory than the v1 kernels declare (3072 bytes), a first dispatch past the device's workgroup
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

/// The variant the public function of `operation` runs: the outcome of the m7.r4 measurement on the RTX 5090 and the
/// Radeon 890M (the logs are `plans/m7/probe/r4` of the work area; D121). The rule is plan section 5.5: v1 ships for
/// an operation only where it is faster than v0 on both adapters at 1e6, 1e7 and 1e8 elements. It is faster on both
/// for every operation, so all six run v1; the reductions are only 1.01 to 1.03 ahead at 1e8 on the 890M, and D121
/// keeps v1 there.
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
