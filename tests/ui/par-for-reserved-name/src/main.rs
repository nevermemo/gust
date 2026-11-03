// A `par_for!` binding reads the variable of its own name, and the expansion defines `run`, `launch`, `launch_default`
// and `DESCRIPTOR` in the block of the call, so a binding of one of those names would read the expansion's item
// instead of the variable (GUST-E0720, docs/DIAGNOSTICS.md). expected.txt pins the code and the line of the binding.

use gust::cpu::CpuDevice;

fn run_loop(launch: &mut [f32], device: &CpuDevice) -> Result<(), gust::map::Error> {
    let n = launch.len() as u32;
    gust::par_for!(on(device); i in 0..n, [launch: &mut [f32]] {
        launch[i] = 1.0;
    })
}

fn main() {
    let _ = run_loop(&mut [0.0_f32; 2], &CpuDevice::new());
}
