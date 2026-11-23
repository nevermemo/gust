// A `gpu_task!` closure of a free stage names its result type (GUST-E1001, docs/DIAGNOSTICS.md): the type is the element
// type of the result buffer. expected.txt pins the code and the line of the call.

use gust::cpu::CpuDevice;

fn main() {
    let device = CpuDevice::new();
    let n = 4_u32;
    let _ = gust::gpu_task!(on(&device); n, |i| { i });
}
