//! The reduce kernels: the module of `tests/fixtures/reduce.rs`, whose Slang is the golden
//! `tests/fixtures/reduce.slang` (the lane layout is documented there).
//!
//! Public so a program can run a pass on buffers of its own; the typed `call`, `submit` and `dispatch` of each kernel
//! are the generated ones. `reduce::sum` and friends are the supported surface. The `*_pass` and `*_final`
//! kernels are v0 (block-strided, with CPU twins); the `*_block` kernels are v1 (a tree in workgroup memory, no CPU
//! twin).

include!("../../../tests/fixtures/reduce.rs");

pub use self::reduce::*;
