// An in-place `gpu_task!` reads its target only at the invocation own index (GUST-E1008, docs/DIAGNOSTICS.md): a read of
// another element races with the write of its invocation. expected.txt pins the code and the line of the read.

use gust::cpu::CpuDevice;

fn main() {
    let device = CpuDevice::new();
    let mut data = vec![0_u32; 4];
    let _ = gust::gpu_task!(on(&device); &mut data, [data: &mut [u32]], |i| { data[i] = data[i + 1u32]; });
}
