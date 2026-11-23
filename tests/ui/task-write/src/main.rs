// A `gpu_task!` stage does not write a capture (GUST-E1003, docs/DIAGNOSTICS.md): only the result buffer, or the target
// of an in-place stage, is written. expected.txt pins the code and the line of the write.

use gust::cpu::CpuDevice;

fn main() {
    let device = CpuDevice::new();
    let a = vec![0_u32; 4];
    let n = a.len();
    let _ = gust::gpu_task!(on(&device); n, [a: &[u32]], |i| -> u32 {
        a[i] = 1u32;
        2u32
    });
}
