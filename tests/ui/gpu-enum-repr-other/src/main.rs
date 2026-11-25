// A `GpuEnum` enum has `#[repr(u32)]` and no other representation (GUST-E1011), at the attribute. expected.txt pins
// the code and the line.

use gust::GpuEnum;

#[derive(Clone, Copy, PartialEq, GpuEnum)]
#[repr(i32)]
enum Mode {
    Linear,
    Curve,
}

fn main() {}
