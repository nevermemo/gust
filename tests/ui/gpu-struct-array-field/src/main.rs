// An array is no field of a `GpuStruct` struct yet (GUST-E1021): write its elements as fields of their own.

use gust::GpuStruct;

#[derive(Clone, Copy, GpuStruct)]
#[repr(C)]
struct Row {
    a: u32,
    b: [f32; 4], // error here
}

fn main() {}
