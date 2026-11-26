// An enum is no field of a `GpuStruct` struct yet (GUST-E1021): a stored enum word could be no variant. A `u32` field
// is the stand-in.

use gust::{GpuEnum, GpuStruct};

#[derive(Clone, Copy, PartialEq, GpuEnum)]
#[repr(u32)]
enum Mode {
    Linear,
    Curve,
}

#[derive(Clone, Copy, GpuStruct)]
#[repr(C)]
struct Row {
    a: u32,
    b: Mode, // error here
}

fn main() {}
