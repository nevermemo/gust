// Measured: a crate that forbids unsafe code compiles a `GpuStruct` derive. (check-ui runs `cargo check`, so this pins
// that the derive's `unsafe impl` raises no `unsafe_code` lint; the outcome is the fixture's passing.)
#![forbid(unsafe_code)]

use gust::GpuStruct;

#[derive(Clone, Copy, GpuStruct)]
#[repr(C)]
struct Row {
    a: f32,
    b: u32,
}

fn main() {
    let _ = Row { a: 0.0, b: 0 };
}
