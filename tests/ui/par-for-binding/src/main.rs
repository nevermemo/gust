// A `par_for!` binding is a slice, a `GpuVec` or one of four scalars (GUST-E0720, docs/DIAGNOSTICS.md): a `Vec` is
// none of them, because the macro cannot tell whether the loop reads it or writes it back. expected.txt pins the code
// and the line of the binding.

use gust::cpu::CpuDevice;

fn run(out: &mut Vec<f32>, device: &CpuDevice) -> Result<(), gust::map::Error> {
    let n = out.len() as u32;
    gust::par_for!(on(device); i in 0..n, [out: Vec<f32>] {
        out[i] = 1.0;
    })
}

fn main() {
    let _ = run(&mut vec![0.0_f32; 2], &CpuDevice::new());
}
