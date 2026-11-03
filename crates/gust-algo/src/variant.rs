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
//! The choice is provisional. Plan section 5.5 asks for both adapters; the RTX 5090 half selected v1 for all six
//! operations, and the Radeon 890M half is owed and may change [`shipped`]. The statistic is the median over runs of
//! the per-run median ratio (v0 / v1, interleaved, end to end), chosen after the runs and not pre-registered; the
//! lowest single run was `min_f32` at 1e8 with 1.085, under the 1.10 line, and the next ones were `sum_i32` (1.120) and
//! `max_f32` (1.137) at 1e8. At 1e6 and 1e7 a reduction costs about 600 microseconds a call and the memory traffic only
//! tens, so v1's win there is mostly per-call overhead (one dispatch fewer and one pooled zeroing fewer); re-measure if
//! the pool's zeroing changes. The log and notes are in `plans/m7/probe/r4` of the work area.
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

/// The variant the public function of `operation` runs: the outcome of the m7.r4 measurement on the RTX 5090 alone
/// (the log is `plans/m7/probe/r4` of the work area). The rule is plan section 5.5: v1 ships for an operation only
/// where it is at least 10 % faster at every measured size. Provisional: the Radeon 890M half of the measurement is
/// owed and may change this.
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
