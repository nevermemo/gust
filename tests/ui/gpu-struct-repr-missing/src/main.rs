// A `GpuStruct` struct is written `#[repr(C)]` (GUST-E1020), reported at the struct's name: the derive checks it and
// does not add it. expected.txt pins the code and the line.

use gust::GpuStruct;

#[derive(Clone, Copy, GpuStruct)]
struct Row { // error here
    a: f32,
    b: u32,
}

fn main() {}
