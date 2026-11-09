// Runs `physics::scale` through a `RoutedDevice` and prints the target the build gave the kernel and the result;
// `cargo xtask check-routing` reads both. The call site names no target: routing `physics` to the CPU or to the GPU in
// `gust.toml` is the only difference between the two builds the check runs.

use gust::routed::RoutedDevice;
use gust_wgpu::HeadlessDevice;
use route_kernels::physics;

fn main() {
    // A handle with the GPU when the machine has an adapter, else the CPU only (a GPU-routed kernel then fails below).
    let device = match HeadlessDevice::new() {
        Ok(gpu) => {
            println!("adapter: {}", gpu.adapter_info().name);
            RoutedDevice::with_gpu(gpu)
        }
        Err(_) => RoutedDevice::cpu_only(),
    };
    let input = device
        .buffer(&[1.0f32, 2.0, 3.0, 4.0])
        .expect("input buffer");
    let out = device.buffer(&[0.0f32; 4]).expect("output buffer");
    physics::scale::dispatch(&device, 4, &input, &out).expect("dispatch through the handle");
    let result = out.read(&device).expect("readback");
    assert_eq!(result, [2.0, 4.0, 6.0, 8.0], "scale doubles its input");
    println!(
        "routing: handle ran physics::scale on {:?} {result:?}",
        physics::scale::DESCRIPTOR.target
    );
}
