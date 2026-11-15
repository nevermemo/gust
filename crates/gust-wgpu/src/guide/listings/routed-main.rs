fn main() {
    let n = 1000;
    let raw = samples(n);
    let gpu = HeadlessDevice::new().expect("a Vulkan compute adapter");
    println!("adapter: {}", gpu.adapter_info().name);
    // One handle for both stages: each call runs on the target its kernel was built for, and the buffers move between
    // the CPU and the GPU as the calls need. Nothing below names a target, so moving a stage in `gust.toml` changes no
    // line.
    let device = RoutedDevice::with_gpu(gpu);
    let raw_buffer = device.buffer(&raw).expect("input buffer");
    let tidy_buffer = device.buffer(&vec![0u32; n]).expect("middle buffer");
    let out_buffer = device.buffer(&vec![0u32; n]).expect("output buffer");
    let count = u32::try_from(n).expect("fewer than 2^32 samples");
    tidy::saturate::dispatch(&device, count, &raw_buffer, &tidy_buffer).expect("stage one");
    smooth::blur::dispatch(&device, count, &tidy_buffer, &out_buffer).expect("stage two");
    let result = out_buffer.read(&device).expect("read back");
    assert_eq!(result, host_reference(&raw), "both stages are exact");
    let checksum: u64 = result.iter().map(|&v| u64::from(v)).sum();
    println!(
        "routed: {n} samples, tidy on {:?}, smooth on {:?}, checksum {checksum}",
        tidy::saturate::DESCRIPTOR.target,
        smooth::blur::DESCRIPTOR.target
    );
}
