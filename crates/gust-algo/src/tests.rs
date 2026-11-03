//! The tests of `gust-algo`. `twin::`, `schedule::`, `bound::` and `golden::` need neither an adapter nor Slang
//! (the Unit suite runs everything but `gpu::`); `gpu::` runs the same cases on the first Vulkan adapter (the Gpu
//! suite).
//!
//! Every case has an independent reference (a sequential `f64` accumulation, an `i128` sum, `Iterator::min`), and the
//! twin and the GPU are each compared with it, never with each other.

mod bound;
mod common;
mod golden;
mod gpu;
mod schedule;
mod struct_gather;
mod twin;
mod variant;
