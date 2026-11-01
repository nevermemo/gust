//! Differential program generator for the GUST dialect, its batch runner and its reducer (M5, slices m5.c1a to m5.c1c).
//!
//! `generator` turns a seed into scalar programs (`ast`), `interp` evaluates them with domain tracking (the independent
//! host reference), `emit` writes them as a `#[gust::gpu]` kernel crate, and `corpus` pins what the committed seeds
//! decode to. `runsrc` adds the test module that runs a batch on the CPU twin and the GPU, `harness` runs the batch
//! with cargo, and the report is written by `harness`. `reduce` shrinks a failing program and `shrink` drives it with
//! cargo (`difftest-run --shrink`). See `docs/development/DIFFTEST.md`.

pub mod ast;
pub mod classify;
pub mod corpus;
pub mod emit;
pub mod float;
pub mod generator;
pub mod harness;
pub mod interp;
pub mod json;
pub mod prng;
pub mod reduce;
#[cfg(test)]
mod reparse;
pub mod runsrc;
pub mod shrink;
#[cfg(test)]
mod soundness;

pub use generator::DIFFTEST_FORMAT;
