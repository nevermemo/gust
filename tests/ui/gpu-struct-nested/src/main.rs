// A struct is no field of a `GpuStruct` struct yet (GUST-E1021): the derive sees only a name, and a name that is not a
// scalar, a vector or a matrix is refused, whatever it names.

use gust::GpuStruct;

#[derive(Clone, Copy, GpuStruct)]
#[repr(C)]
struct Inner {
    x: f32,
}

#[derive(Clone, Copy, GpuStruct)]
#[repr(C)]
struct Row {
    a: u32,
    b: Inner, // error here
}

fn main() {}
