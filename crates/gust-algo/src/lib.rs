//! Device-resident algorithms on GUST kernels (roadmap M7, slice m7.r1: `reduce`).
//!
//! The functions take a [`GpuVec`](gust_wgpu::GpuVec) and run on the device that holds it: no host round trip except
//! the one-element result of [`reduce::sum`], [`reduce::min`] and [`reduce::max`]. Every kernel is a GUST kernel
//! (`tests/fixtures/reduce.rs`, included as [`kernels`]); the host code only chooses the pass schedule
//! ([`schedule::Schedule`]), takes one pooled `GpuVec` per pass and submits all passes in one `dispatch_batch`.
//!
//! What is promised (plan section 5.1): integer sums are exact and inside the domain only when no partial sum
//! overflows; `min` and `max` are exact; float sums are within the per-level bound of
//! [`bound::f32_sum_bound`] of the exact sum, a bound over every grouping WGSL permits, not a bit pattern. The
//! kernels run on `gust::cpu::CpuDevice` through their twins in test and dual builds ([`cpu`]); a twin result is never
//! GPU evidence.
//!
//! Kernels are compiled at run time through gust-wgpu's `slang-runtime` feature (enable this crate's feature of the
//! same name); the crate ships no precompiled artifacts in v0.

pub mod bound;
#[cfg(any(test, gust_dual))]
pub mod cpu;
mod error;
pub mod kernels;
pub mod reduce;
pub mod schedule;

pub use error::Error;

#[cfg(test)]
mod tests;
