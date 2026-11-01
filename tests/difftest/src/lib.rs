//! Differential program generator for the GUST dialect and its batch runner (M5, slices m5.c1a, m5.c1a2 and m5.c1b).
//!
//! `generator` turns a seed into scalar programs (`ast`), `interp` evaluates them with domain tracking (the independent
//! host reference), `emit` writes them as a `#[gust::gpu]` kernel crate, and `corpus` pins what the committed seeds
//! decode to. `runsrc` adds the test module that runs a batch on the CPU twin and the GPU, `harness` runs the batch
//! with cargo, and the report is written by `harness`. See `docs/development/DIFFTEST.md`.

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
#[cfg(test)]
mod reparse;
pub mod runsrc;
#[cfg(test)]
mod soundness;

pub use generator::DIFFTEST_FORMAT;
