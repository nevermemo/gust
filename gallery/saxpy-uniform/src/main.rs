// ANCHOR: imports
use gust::{Run, gpu_module};
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};
// ANCHOR_END: imports

// ANCHOR: kernel
#[gpu_module]
mod saxpy {
    /// The uniform: one value in a constant buffer, read by every invocation. A uniform is a struct of the module
    /// whose layout has no implicit padding and whose size is a multiple of 16 bytes, so the three `pad` fields are
    /// written out to round the 4-byte `a` up to 16.
    pub struct Params {
        pub a: f32,
        pub pad0: f32,
        pub pad1: f32,
        pub pad2: f32,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn apply(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        x: StructuredBuffer<float>,
        y: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = p.a * x[i] + y[i];
        }
    }
}
// ANCHOR_END: kernel

// ANCHOR: params
/// The uniform value for `a`; the padding is zero.
fn params(a: f32) -> saxpy::Params {
    saxpy::Params {
        a,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}
// ANCHOR_END: params

// ANCHOR: main
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
// ANCHOR_END: main

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use gust::cpu::CpuDevice;

    use super::{params, saxpy};

    /// The independent host reference: the sum in `f64`, then rounded once to `f32`. It never goes through the kernel
    /// (D26, D42).
    fn host_reference(a: f32, x: &[f32], y: &[f32]) -> Vec<f32> {
        x.iter()
            .zip(y)
            .map(|(x, y)| (f64::from(a) * f64::from(*x) + f64::from(*y)) as f32)
            .collect()
    }

    /// The kernel on the CPU twin through the typed `dispatch` (the macro emits it for `CpuDevice` under `cfg(test)`).
    fn on_twin(a: f32, x: &[f32], y: &[f32], threads: u32) -> Vec<f32> {
        let device = CpuDevice::new();
        let uniform = device.uniform(params(a)).expect("a padding-free uniform");
        let out = device.buffer(&vec![f32::NAN; x.len()]);
        saxpy::apply::dispatch(
            &device,
            threads,
            &uniform,
            &device.buffer(x),
            &device.buffer(y),
            &out,
        )
        .expect("the twin dispatches");
        out.read()
    }

    fn threads(n: usize) -> u32 {
        u32::try_from(n).expect("small")
    }

    #[test]
    fn the_twin_matches_the_program_values() {
        let result = on_twin(2.0, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0], 4);
        assert_eq!(result, [12.0, 24.0, 36.0, 48.0]);
    }

    /// Small integers and a power-of-two `a`: every product and sum is exact in `f32`, so the comparison is exact.
    #[test]
    fn the_twin_matches_the_host_reference_exactly_on_exact_data() {
        // Sizes below, at and above the workgroup size of 64, and ones that are not a multiple of it.
        for n in [1, 63, 64, 65, 129, 1000] {
            let x: Vec<f32> = (0..n).map(|i| (i % 97) as f32).collect();
            let y: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 8.0).collect();
            for a in [2.0f32, -0.5, 0.0] {
                assert_eq!(
                    on_twin(a, &x, &y, threads(n)),
                    host_reference(a, &x, &y),
                    "n = {n}, a = {a}"
                );
            }
        }
    }

    /// Data whose products are not exact. The tolerance is 2 ulp of `|a x| + |y|` (`f32::EPSILON` is one ulp at 1.0):
    /// a device may fuse the multiply and the add, and the reference rounds once where the twin rounds twice.
    #[test]
    fn the_twin_agrees_with_the_host_reference_within_two_ulp_on_inexact_data() {
        for n in [1, 63, 64, 65, 1000] {
            let x: Vec<f32> = (0..n).map(|i| (i % 101) as f32 * 0.37 - 11.0).collect();
            let y: Vec<f32> = (0..n).map(|i| (i % 59) as f32 * -0.91 + 7.5).collect();
            let a = 1.7f32;
            let got = on_twin(a, &x, &y, threads(n));
            let want = host_reference(a, &x, &y);
            for i in 0..n {
                let bound = 2.0 * f32::EPSILON * ((a * x[i]).abs() + y[i].abs());
                assert!(
                    (got[i] - want[i]).abs() <= bound,
                    "n = {n}, i = {i}: {} versus {} (bound {bound})",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// The uniform is one value for the whole dispatch, and a written value is the next dispatch's.
    #[test]
    fn a_written_uniform_is_seen_by_the_next_dispatch() {
        let device = CpuDevice::new();
        let uniform = device.uniform(params(2.0)).expect("a uniform");
        let (x, y) = (
            device.buffer(&[1.0f32, 2.0, 3.0]),
            device.buffer(&[1.0f32; 3]),
        );
        let out = device.buffer(&[0.0f32; 3]);
        saxpy::apply::dispatch(&device, 3, &uniform, &x, &y, &out).expect("first");
        assert_eq!(out.read(), [3.0, 5.0, 7.0]);
        uniform.write(params(-1.0)).expect("write");
        saxpy::apply::dispatch(&device, 3, &uniform, &x, &y, &out).expect("second");
        assert_eq!(out.read(), [0.0, -1.0, -2.0]);
    }
}
// ANCHOR_END: tests
