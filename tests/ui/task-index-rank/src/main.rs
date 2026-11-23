// A host slice of a `gpu_task!` is flat: indexing it with a tuple is refused (GUST-E1007, docs/DIAGNOSTICS.md).
// expected.txt pins the code and the line of the index.

use gust::cpu::CpuDevice;

fn main() {
    let device = CpuDevice::new();
    let a = vec![0_u32; 4];
    let n = a.len();
    let _ = gust::gpu_task!(on(&device); n, [a: &[u32]], |i| -> u32 { a[(i, 0u32)] });
}
