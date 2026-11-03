//! The compaction kernels: the module of `tests/fixtures/compact.rs`, whose Slang is the golden
//! `tests/fixtures/compact.slang` (the block-strided shape and the output rules are documented there).
//!
//! Public so a program can run a step on buffers of its own; the typed `call`, `submit` and `dispatch` of each kernel
//! are the generated ones. `compact::count`, `compact::indices` and the `compact::gather_*` functions are the supported
//! surface.

include!("../../../tests/fixtures/compact.rs");

pub use self::compact::*;
