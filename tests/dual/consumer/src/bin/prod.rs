// A production-style binary of the consumer: it dispatches on `gust::cpu::CpuDevice` unconditionally. A normal
// build of it has no twin to run (E3 pins the error); with `--cfg gust_dual` it compiles, which is what a dual
// build means and is stated, not tested.

use gust::cpu::CpuDevice;
use kernel_lib::kernels;

fn main() {
    let device = CpuDevice::new();
    let a = device.buffer(&[1.0f32, 2.0, 3.0]);
    let b = device.buffer(&[10.0f32, 20.0, 30.0]);
    let out = device.buffer(&[0.0f32; 3]);
    kernels::add::dispatch(&device, 3, &a, &b, &out).unwrap();
    println!("{:?}", out.read());
}
