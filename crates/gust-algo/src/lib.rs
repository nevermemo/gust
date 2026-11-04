#![forbid(unsafe_code)]
//! Device-resident algorithms on GUST kernels (roadmap M7, slices m7.r1: `reduce`, m7.r2: `scan`, m7.r3: `compact`).
//!
//! The functions take a [`GpuVec`](gust_wgpu::GpuVec) and run on the device that holds it: no host round trip except
//! the one-element result of [`reduce::sum`], [`reduce::sum_wrapping`], [`reduce::min`] and [`reduce::max`]; a
//! [`scan`] returns a new vector on the same device, and so does everything in [`compact`] (a count that stays on the
//! device, the indices of the set flags, gathers by those indices). Every kernel is a GUST kernel
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
//! v0 stays the fallback, for an input whose v1 launch exceeds the
//! device's workgroup limit, and the only form with CPU twins (the `cpu` module, `GUST_PROFILE=cpu`). The `*_with`
//! functions run either on request.
//!
//! What is promised (plan section 5.1): integer sums and scans are exact and inside the domain only when no partial sum
//! (or prefix) overflows, and [`reduce::sum_wrapping`] is defined modulo 2^32 for every input; `min` and `max` are
//! exact; float sums are within the per-level bound of the variant that ran
//! ([`bound::f32_sum_bound_v1`] for v1, [`bound::f32_sum_bound`] for the v0 fallback; a caller who cannot tell which
//! ran may assert the larger of the two) of the exact sum, a bound over every grouping WGSL permits, not a bit
//! pattern. The v0 kernels run on `gust::cpu::CpuDevice` through their twins in test and dual builds ([`cpu`]); a twin
//! result is never GPU evidence.
//!
//! Kernels are compiled at run time through gust-wgpu's `slang-runtime` feature (enable this crate's feature of the
//! same name); the crate ships no precompiled artifacts.

pub mod bound;
pub mod compact;
pub mod compact_kernels;
#[cfg(any(test, gust_dual))]
pub mod cpu;
mod error;
pub mod kernels;
mod limits;
pub mod reduce;
pub mod scan;
pub mod scan_kernels;
pub mod schedule;
pub mod variant;

pub use error::Error;
pub use variant::Variant;

#[cfg(test)]
mod tests;
