fn main() {
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let y = [10.0f32, 20.0, 30.0, 40.0];
    let device = HeadlessDevice::new().expect("a Vulkan compute adapter");
    let gx = device.create_typed_buffer("x", &x, GpuBufferAccess::ReadOnly);
    let gy = device.create_typed_buffer("y", &y, GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("out", &[0.0f32; 4], GpuBufferAccess::ReadWrite);
    let uniform = device
        .create_uniform("params", &params(2.0))
        .expect("a padding-free uniform");

    // The typed call: the kernel's own signature, checked by the compiler (a buffer where the uniform goes is an error).
    saxpy::apply::dispatch(&device, 4, &uniform, &gx, &gy, &out).expect("dispatch");
    let first = device.read_typed_buffer(&out).expect("read back");

    // The generated `call` is a value: it names the bound resources and can be built once and submitted again.
    // A later `write_uniform` is seen by the next dispatch.
    let call = saxpy::apply::call(4, &uniform, &gx, &gy, &out);
    device
        .write_uniform(&uniform, &params(3.0))
        .expect("write the uniform");
    device.dispatch_call(&call).expect("dispatch the call");
    let second = device.read_typed_buffer(&out).expect("read back");
    assert_eq!(first, [12.0, 24.0, 36.0, 48.0]);
    assert_eq!(second, [13.0, 26.0, 39.0, 52.0]);

    println!("saxpy-uniform: a=2 {first:?}, a=3 {second:?}");
    println!("adapter: {}", device.adapter_info().name);
}
