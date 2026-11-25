#![forbid(unsafe_code)]
//! Device-resident algorithms on GUST kernels (roadmap M7, slices m7.r1: `reduce`, m7.r2: `scan`, m7.r3: `compact`).
//!
//! The functions take a [`GpuVec`](gust_wgpu::GpuVec) and run on the device that holds it: no host round trip except
//! the one-element result of [`reduce::sum`], [`reduce::sum_wrapping`], [`reduce::min`] and [`reduce::max`]; a
//! [`scan`] returns a new vector on the same device (`scan::inclusive_into` and `scan::exclusive_into` write into the
//! caller's own buffer instead), and so does everything in [`compact`] (a count that stays on the device, the indices
//! of the set flags, gathers by those indices). Every kernel is a GUST kernel
//! (`tests/fixtures/reduce.rs`, included as [`kernels`], `tests/fixtures/scan.rs`, included as [`scan_kernels`], and
//! `tests/fixtures/compact.rs`, included as [`compact_kernels`]); the host code only chooses the schedule
//! ([`schedule::Schedule`] for a reduction, [`schedule::ScanSchedule`] for a scan), takes one pooled `GpuVec` per level
//! and submits all dispatches in one `dispatch_batch`. These are the v0 kernels, block-strided and with CPU twins.
//!
//! Slice m7.r4 adds the v1 kernels (`Workgroup<[T; 256]>` and barriers, `cpu = "unsupported"`, D101) of the reduce
//! and the scan beside them ([`variant`]): the same results, fewer and wider dispatches. **The public functions run
//! v1** (`variant::shipped` is `V1` for every operation). The plan's rule (section 5.5) is faster than v0 on both
//! adapters at 1e6, 1e7 and 1e8 elements, and it holds on the RTX 5090 and on the Radeon 890M; the reductions are only
//! 1.01 to 1.03 ahead at 1e8 on the 890M, where both forms are bound by shared system memory, and D121 keeps v1.
//! v0 stays the fallback for a device with too little workgroup memory for v1 (a binding past the storage limit is
//! refused by v0 as well, and a launch past the workgroup limit is split by the host, D211), and the only form with CPU
//! twins (the `cpu` module, `GUST_PROFILE=cpu`). The `*_with` functions run either on request.
//!
//! What is promised (plan section 5.1): integer sums and scans are exact and inside the domain only when no partial sum
//! (or prefix) overflows, and [`reduce::sum_wrapping`] is defined modulo 2^32 for every input; `min` and `max` are
//! exact; float sums are within the per-level bound of the variant that ran
//! ([`bound::f32_sum_bound_v1`] for v1, [`bound::f32_sum_bound`] for the v0 fallback; a caller who cannot tell which
//! ran may assert the larger of the two) of the exact sum, a bound over every grouping WGSL permits, not a bit
//! pattern. The v0 kernels run on `gust::cpu::CpuDevice` through their twins in test and dual builds (the `cpu` module); a twin
//! result is never GPU evidence.
//!
//! Opt-in GPU timestamps: [`timed`] runs any calls of this crate in a scope and reports the GPU time of the batches
//! they dispatched ([`Timed`], [`GpuTime`]); calls outside a scope pay one thread-local read per batch (D194).
//! A caller's own batch counts in the same scope when it goes through [`dispatch_batch_in_scope`] instead of the
//! device's own `dispatch_batch`.
//!
//! [`evidence`] is the subgroup form of the reduction's block step, kept as measured evidence: not for users, and not part
//! of what this crate promises (see `docs/development/VALIDATION.md`, "Subgroup evidence").
//!
//! [`record`] is the recording form (slice m14.s2f1, D259): owned plans of the reductions, scans and compactions that
//! append their dispatches over borrowed operands to a [`record::Sink`], and [`record::Recording`], a list of them for a
//! caller with `GpuVec`s, with results in buffers of the caller's; the list runs as one `dispatch_batch` (or its
//! dispatches go to a batch of the caller's). The crate compiles for `wasm32` (what blocks there, the immediate
//! functions, `timed`, `evidence` and `Recording::dispatch`, is compiled out), but its kernels need artifacts the crate
//! does not ship, so nothing runs there yet.
//!
//! `batch` (slice m14.s2f2a, hidden macro support, outside the pre-1.0 promise) is the route from a `gust::batch` to
//! the same plans: op types that a build block records (`b.op(&["total"], Sum::of(&blocks))`), for the GPU device
//! and, as a host form, for the CPU device. The forms users write arrive with the macro.
//!
//! Kernels are compiled at run time through gust-wgpu's `slang-runtime` feature (enable this crate's feature of the
//! same name); the crate ships no precompiled artifacts.

#[doc(hidden)]
pub mod batch;
pub mod bound;
pub mod compact;
pub mod compact_kernels;
#[cfg(any(test, gust_dual))]
pub mod cpu;
mod error;
#[cfg(not(target_arch = "wasm32"))]
pub mod evidence;
pub mod kernels;
mod limits;
pub mod record;
pub mod reduce;
pub mod scan;
pub mod scan_kernels;
pub mod schedule;
#[cfg(not(target_arch = "wasm32"))]
mod timing;
pub mod variant;

pub use error::Error;
#[cfg(not(target_arch = "wasm32"))]
pub use timing::{GpuTime, Timed, dispatch_batch_in_scope, timed};
pub use variant::Variant;

#[cfg(test)]
mod tests;
