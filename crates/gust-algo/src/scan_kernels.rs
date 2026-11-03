//! The scan kernels: the module of `tests/fixtures/scan.rs`, whose Slang is the golden `tests/fixtures/scan.slang` (the
//! chunk layout and the add-back order are documented there).
//!
//! Public so a program can run a step on buffers of its own; the typed `call`, `submit` and `dispatch` of each kernel
//! are the generated ones. `scan::inclusive` and `scan::exclusive` are the supported surface. The chunk kernels are v0
//! (with CPU twins); the `scan_tiles_*` kernels are v1 (tiles scanned in workgroup memory, no CPU twin) and
//! `add_carry_u32` is their element-wise add-back.

include!("../../../tests/fixtures/scan.rs");

pub use self::scan::*;
