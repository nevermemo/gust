// A discriminant of a `GpuEnum` enum is an integer literal (GUST-E1011), at the expression. expected.txt pins the code
// and the line.

use gust::GpuEnum;

const BASE: u32 = 4;

#[derive(Clone, Copy, PartialEq, GpuEnum)]
#[repr(u32)]
enum Mode {
    Linear = 0,
    Curve = BASE,
}

fn main() {}
