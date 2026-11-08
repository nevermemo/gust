// Runs the GPU-routed `physics::scale` on a GPU and prints the adapter and the result; `cargo xtask check-routing` reads
// both. The call names `HeadlessDevice`, so it compiles only while `physics` is routed to the GPU.

use gust_wgpu::{GpuBufferAccess, HeadlessDevice};
use route_kernels::physics;

fn main() {
    let device = HeadlessDevice::new().expect("a Vulkan compute adapter should be available");
    let input = device.create_typed_buffer(
        "input",
        &[1.0f32, 2.0, 3.0, 4.0],
        GpuBufferAccess::ReadWrite,
    );
    let out = device.create_typed_buffer("out", &[0.0f32; 4], GpuBufferAccess::ReadWrite);
    physics::scale::dispatch(&device, 4, &input, &out).expect("dispatch on the GPU");
    let result = device
        .read_typed_buffer(&out)
        .expect("readback should succeed");
    println!("adapter: {}", device.adapter_info().name);
    assert_eq!(result, [2.0, 4.0, 6.0, 8.0], "scale doubles its input");
    println!("routing: physics::scale ran on the GPU {result:?}");
}
