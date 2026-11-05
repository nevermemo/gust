//! Differential program generator for the GUST dialect, its batch runner and its reducer (M5, slices m5.c1a to m5.c1c;
//! m6.d added the vector and matrix programs).
//!
//! `generator` turns a seed into scalar and vector programs (`ast`), `interp` evaluates them with domain tracking (the
//! independent host reference; `agg` holds its vector and matrix arithmetic), `emit` writes them as a `#[gust::gpu]`
//! kernel crate, and `corpus` pins what the committed seeds
//! decode to. `runsrc` adds the test module that runs a batch on the CPU twin and the GPU, `harness` runs the batch
//! with cargo, and the report is written by `harness`. `reduce` shrinks a failing program and `shrink` drives it with
//! cargo (`difftest-run --shrink`). See `docs/development/DIFFTEST.md`.

pub mod agg;
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
pub mod shapes;
pub mod shrink;
#[cfg(test)]
mod soundness;
#[cfg(test)]
mod vector_tests;

pub use generator::DIFFTEST_FORMAT;
