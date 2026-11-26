// A `GpuStruct` struct has no generics (GUST-E1020), at the parameter list. The parameter is a const one, which rustc
// accepts unused, so there is no error of its own.

use gust::GpuStruct;

#[derive(Clone, Copy, GpuStruct)]
#[repr(C)]
struct Row<const N: u32> { // error here
    a: f32,
    b: u32,
}

fn main() {}
