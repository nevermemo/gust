// A `GpuEnum` enum is written `#[repr(u32)]`: the derive checks it and does not add it (GUST-E1011), at the enum's
// name. expected.txt pins the code and the line.

use gust::GpuEnum;

#[derive(Clone, Copy, PartialEq, GpuEnum)]
enum Mode {
    Linear,
    Curve,
}

fn main() {}
