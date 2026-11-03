// A `bool` is a scalar binding of `par_for!`, never a buffer element: a buffer holds `u32` words (GUST-E0720,
// docs/DIAGNOSTICS.md). The map generator takes `bool` by value, so this pins the `par_for!` side of the shared
// element lookup. expected.txt pins the code, the line of the binding and the start of the message.

use gust::cpu::CpuDevice;

fn run(flags: &mut [bool], device: &CpuDevice) -> Result<(), gust::map::Error> {
    let n = flags.len() as u32;
    gust::par_for!(on(device); i in 0..n, [flags: &mut [bool]] {
        flags[i] = true;
    })
}

fn main() {
    let _ = run(&mut [false; 2], &CpuDevice::new());
}
