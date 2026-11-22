// ANCHOR: imports
use gust::gpu_module;
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};
// ANCHOR_END: imports

// ANCHOR: kernel
#[gpu_module]
mod vector_add {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn add(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<float>,
        b: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = a[i] + b[i];
        }
    }
}
// ANCHOR_END: kernel

// ANCHOR: main
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
// ANCHOR_END: main

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use gust::cpu::CpuDevice;

    use super::vector_add;

    /// The inputs of `n` elements: small integers, so that every sum is exact in `f32` on any target.
    fn inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
        let a = (0..n).map(|i| (i % 97) as f32).collect();
        let b = (0..n).map(|i| (i % 13) as f32 * 8.0).collect();
        (a, b)
    }

    /// The independent host reference: plain Rust over slices, which never goes through the kernel (D26, D42).
    fn host_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    /// The kernel on the CPU twin (the macro emits the same typed `dispatch` for `CpuDevice` under `cfg(test)`).
    fn on_twin(a: &[f32], b: &[f32], threads: u32) -> Vec<f32> {
        let device = CpuDevice::new();
        let ga = device.buffer(a);
        let gb = device.buffer(b);
        let out = device.buffer(&vec![0.0; a.len()]);
        vector_add::add::dispatch(&device, threads, &ga, &gb, &out).expect("the twin dispatches");
        out.read()
    }

    #[test]
    fn the_twin_matches_the_program_values() {
        let result = on_twin(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0], 4);
        assert_eq!(result, [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn the_twin_matches_the_host_reference() {
        // Sizes below, at and above the workgroup size of 64, and one that is not a multiple of it.
        for n in [1, 63, 64, 65, 1000] {
            let (a, b) = inputs(n);
            let threads = u32::try_from(n).expect("small");
            assert_eq!(on_twin(&a, &b, threads), host_reference(&a, &b), "n = {n}");
        }
    }
}
// ANCHOR_END: tests
