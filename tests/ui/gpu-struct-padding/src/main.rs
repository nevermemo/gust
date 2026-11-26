// A `GpuStruct` struct has no implicit padding (GUST-E0208): `b` needs 8-byte alignment, so four bytes would sit
// unused after `a`. The message names the field to add and where; reported at the field that sits late.

use gust::GpuStruct;
use gust::prelude::*;

#[derive(Clone, Copy, GpuStruct)]
#[repr(C)]
struct Row {
    a: f32,
    b: Vec2, // error here
}

fn main() {}
