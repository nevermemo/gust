//! The reduce kernels: the module of `tests/fixtures/reduce.rs`, whose Slang is the golden
//! `tests/fixtures/reduce.slang` (the lane layout is documented there).
//!
//! Public so a program can run a pass on buffers of its own; the typed `call`, `submit` and `dispatch` of each kernel
//! are the generated ones. `reduce::sum` and friends are the supported surface.

include!("../../../tests/fixtures/reduce.rs");

pub use self::reduce::*;
