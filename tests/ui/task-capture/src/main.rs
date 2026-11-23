// The target of an in-place `gpu_task!` needs an entry in the capture list (GUST-E1002, docs/DIAGNOSTICS.md): the macro
// cannot see the element type of a variable. expected.txt pins the code and the line of the call.

use gust::cpu::CpuDevice;

fn main() {
    let device = CpuDevice::new();
    let mut data = vec![0_u32; 4];
    let _ = gust::gpu_task!(on(&device); &mut data, [], |i| { data[i] = 1u32; });
}
