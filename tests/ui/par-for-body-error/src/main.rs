// A dialect error inside the body of a `par_for!` is reported at the body, with the code of the same error in a
// `#[gpu]` kernel (D33): a `while` loop is GUST-E0402, and the error points at the `while`, not at the call or at the
// code the macro generates around the body. expected.txt pins the code and the line of the `while`.

use gust::cpu::CpuDevice;

fn run(out: &mut [f32], device: &CpuDevice) -> Result<(), gust::map::Error> {
    let n = out.len() as u32;
    gust::par_for!(on(device); i in 0..n, [out: &mut [f32]] {
        let mut v = 1.0;
        while v < 4.0 {
            v = v * 2.0;
        }
        out[i] = v;
    })
}

fn main() {
    let _ = run(&mut [0.0_f32; 2], &CpuDevice::new());
}
