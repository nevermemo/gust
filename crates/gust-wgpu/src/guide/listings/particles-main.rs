/// The program; it runs on the GPU, so it is not part of the test build (which has the CPU twin for `Dev`).
#[cfg(not(test))]
fn main() {
    let n: usize = 1000;
    let count = u32::try_from(n).expect("fewer than 2^32 particles");
    let initial: Vec<_> = (0..n).map(start).collect();
    let device = HeadlessDevice::new().expect("a Vulkan compute adapter");
    let buffer = device.create_typed_buffer("particles", &initial, GpuBufferAccess::ReadWrite);

    // The same call sixteen times in one batch: one command buffer, one pass per step, each pass sees the last one's
    // writes. The call is built once and borrowed by every entry.
    let call = particles::step::call(count, &buffer);
    let batch: Vec<BufferDispatch<'_>> = (0..STEPS).map(|_| (&call).into()).collect();
    device.dispatch_batch(&batch).expect("dispatch the batch");

    let after = device.read_typed_buffer(&buffer).expect("read back");
    for (i, p) in after.iter().enumerate() {
        let want = host_steps(host_start(i), STEPS);
        assert_eq!(p.pos.map(f64::from), want.pos, "pos of particle {i}");
        assert_eq!(p.vel.map(f64::from), want.vel, "vel of particle {i}");
        assert_eq!(p.bounces, want.bounces, "bounces of particle {i}");
    }
    let bounces: u32 = after.iter().map(|p| p.bounces).sum();
    let energy = total_energy(&after, &device);
    println!("particles: {n} particles, {STEPS} steps, {bounces} bounces, energy {energy}");
    println!("adapter: {}", device.adapter_info().name);
}
