fn main() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [10.0f32, 20.0, 30.0, 40.0];
    let device = HeadlessDevice::new().expect("a Vulkan compute adapter");
    let ga = device.create_typed_buffer("a", &a, GpuBufferAccess::ReadOnly);
    let gb = device.create_typed_buffer("b", &b, GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("out", &[0.0; 4], GpuBufferAccess::ReadWrite);
    vector_add::add::dispatch(&device, 4, &ga, &gb, &out).expect("dispatch");
    let result = device.read_typed_buffer(&out).expect("read back");
    println!("hello-gust: {result:?}");
    println!("adapter: {}", device.adapter_info().name);
}
