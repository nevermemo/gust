// A production-style binary that dispatches the routed module of `kernel-lib` on `gust::cpu::CpuDevice`. It builds
// when the tag `physics` is routed to the CPU (check-dual E4 sets `GUST_ROUTE__kernel_lib__physics=cpu`), with no
// `--cfg gust_dual`: a CPU-routed module is a CPU body in any build. Without the route the module is a GPU kernel and
// the call has no `Run` impl (E4's control pins that error, `e4-expected.txt`).

use gust::cpu::CpuDevice;
use kernel_lib::routed;

fn main() {
    let device = CpuDevice::new();
    let input = device.buffer(&[1.0f32, 2.0, 3.0]);
    let out = device.buffer(&[0.0f32; 3]);
    routed::scale::dispatch(&device, 3, &input, &out).unwrap();
    println!("routed: {:?}", out.read());
}
