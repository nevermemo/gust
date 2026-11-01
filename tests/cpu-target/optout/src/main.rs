// Step E-g: the kernel that opted out of the CPU (`#[kernel(cpu = "unsupported")]`) has no `Run` impl for `CpuDevice`
// under any profile, `GUST_PROFILE=cpu` included, so this dispatch is the one E0277 that e-g-expected.txt pins; its
// note names the opt-out.

use gust::cpu::CpuDevice;
use kernels::opt_out;

fn main() {
    let device = CpuDevice::new();
    let input = device.buffer(&[1.0f32, 2.0, 3.0]);
    let out = device.buffer(&[0.0f32; 3]);
    opt_out::negate::dispatch(&device, 3, &input, &out).unwrap();
    println!("{:?}", out.read());
}
