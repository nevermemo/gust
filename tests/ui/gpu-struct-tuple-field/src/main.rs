// A tuple is no field of a `GpuStruct` struct (GUST-E0212, the code a module gives it): Rust gives it no layout.

use gust::GpuStruct;

#[derive(Clone, Copy, GpuStruct)]
#[repr(C)]
struct Row {
    a: u32,
    b: (u32, u32), // error here
}

fn main() {}
