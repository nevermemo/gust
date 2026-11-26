// A bare `Vec3` is no field of a `GpuStruct` struct (GUST-E0204, the code a module gives it): Rust's 12-byte,
// 4-aligned layout has no std430 match. Use `Vec4` or three `f32` fields.

use gust::GpuStruct;
use gust::prelude::*;

#[derive(Clone, Copy, GpuStruct)]
#[repr(C)]
struct Row {
    a: f32,
    b: Vec3, // error here
}

fn main() {}
