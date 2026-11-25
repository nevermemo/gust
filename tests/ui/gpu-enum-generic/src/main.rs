// A `GpuEnum` enum has no generics (GUST-E1011), at the parameter list. expected.txt pins the code and the line. The
// parameter is a const one, which rustc accepts unused, so there is no error of its own.

use gust::GpuEnum;

#[derive(Clone, Copy, PartialEq, GpuEnum)]
#[repr(u32)]
enum Mode<const N: u32> {
    Linear,
    Curve,
}

fn main() {}
