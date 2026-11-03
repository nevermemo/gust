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
//! and submits all dispatches in one `dispatch_batch`.
//!
//! What is promised (plan section 5.1): integer sums and scans are exact and inside the domain only when no partial sum
//! (or prefix) overflows, and [`reduce::sum_wrapping`] is defined modulo 2^32 for every input; `min` and `max` are
//! exact; float sums are within the per-level bound of [`bound::f32_sum_bound`] of the exact sum, a bound over every
//! grouping WGSL permits, not a bit pattern. The kernels run on `gust::cpu::CpuDevice` through their twins in test and
//! dual builds ([`cpu`]); a twin result is never GPU evidence.
//!
//! Kernels are compiled at run time through gust-wgpu's `slang-runtime` feature (enable this crate's feature of the
//! same name); the crate ships no precompiled artifacts in v0.

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

pub use error::Error;

#[cfg(test)]
mod tests;
