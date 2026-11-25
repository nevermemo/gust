// A variant of a `GpuEnum` enum carries no data: GUST-E1011, once, at the variant. expected.txt pins the code and the
// line.

use gust::GpuEnum;

#[derive(Clone, Copy, PartialEq, GpuEnum)]
#[repr(u32)]
enum Shape {
    Dot,
    Circle(u32),
}

fn main() {}
