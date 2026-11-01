// One dispatch on `gust::cpu::CpuDevice` in a non-test build, and nothing else. Under `GUST_PROFILE=cpu` it compiles
// (the macro emitted the impl in `kernels`). Under any other profile `cargo build` sets neither `cfg(test)` nor
// `--cfg gust_dual`, so the kernel has no `Run` impl for `CpuDevice` and this is the one E0277 that e-d-expected.txt
// pins (step E-d): the production-exclusion proof of the twin, and the flip detector of step E-h.

use gust::cpu::CpuDevice;
use kernels::vector_add;

fn main() {
    let device = CpuDevice::new();
    let a = device.buffer(&[1.0f32, 2.0, 3.0]);
    let b = device.buffer(&[10.0f32, 20.0, 30.0]);
    let out = device.buffer(&[0.0f32; 3]);
    vector_add::add::dispatch(&device, 3, &a, &b, &out).unwrap();
    println!("{:?}", out.read());
}
