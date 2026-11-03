// A `par_for!` range starts at the literal 0 (GUST-E0719, docs/DIAGNOSTICS.md): a loop that starts later has no
// meaning on the device, where the invocation index is the loop variable. expected.txt pins the code and the line of
// the range.

use gust::cpu::CpuDevice;

fn run(out: &mut [u32], device: &CpuDevice) -> Result<(), gust::map::Error> {
    let n = out.len() as u32;
    gust::par_for!(on(device); i in 1..n, [out: &mut [u32]] {
        out[i] = 1u32;
    })
}

fn main() {
    let _ = run(&mut [0_u32; 2], &CpuDevice::new());
}
