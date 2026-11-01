//! Differential program generator for the GUST dialect, integer core (M5, slice m5.c1a).
//!
//! `generator` turns a seed into scalar programs (`ast`), `interp` evaluates them with domain tracking (the independent
//! host reference), `emit` writes them as a `#[gust::gpu]` kernel crate, and `corpus` pins what the committed seeds
//! decode to. There is no GPU and no runner here; those are m5.c1b. See `docs/development/DIFFTEST.md`.

pub mod ast;
pub mod corpus;
pub mod emit;
pub mod generator;
pub mod interp;
pub mod prng;
#[cfg(test)]
mod reparse;

pub use generator::DIFFTEST_FORMAT;
