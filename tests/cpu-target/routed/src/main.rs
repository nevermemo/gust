// Step E-j (m8.r1): runs the CPU-routed `scale` of the library on `CpuDevice` and compares it with a host loop, then
// calls the CPU-routed map function `triple` of this binary (review r1, S6). Both are routed by the one variable
// `GUST_ROUTE__routed__physics=cpu` in a `native` build: `triple` is a `#[gpu(route = ..)] fn`, whose host functions are
// the `cpu` family only because its tag resolved to the CPU, a combination that no other gate compiles. The binary is
// not compiled when the library fails (the unrouted control), so the control keeps its one error.

use gust::cpu::CpuDevice;
use routed::kernels;

const ELEMENTS: usize = 1024;

/// The map function: routed with the module of the library (the same tag), so the same variable decides both.
#[gust::gpu(route = "physics")]
pub fn triple(x: f32) -> f32 {
    x * 3.0
}

fn main() {
    let device = CpuDevice::new();
    let host: Vec<f32> = (0..ELEMENTS).map(|i| i as f32 * 0.5 - 100.0).collect();
    let input = device.buffer(&host);
    let output = device.buffer(&vec![0.0f32; ELEMENTS]);
    kernels::scale::dispatch(&device, ELEMENTS as u32, &input, &output)
        .expect("dispatch on CpuDevice");
    let result = output.read();
    let worst = host
        .iter()
        .zip(&result)
        .map(|(x, got)| (x * 2.0).to_bits().abs_diff(got.to_bits()))
        .max()
        .unwrap_or(0);
    assert_eq!(result.len(), ELEMENTS, "the output length");
    assert_eq!(
        worst, 0,
        "the routed CPU kernel must equal the host loop bit for bit"
    );
    println!("CPU target: matches (routed::scale, {ELEMENTS} elements, max ULP {worst})");

    routed_map(&device, &host);
}

/// The routed map function, through `map` (the process-wide CPU device) and `map_on` (the one given).
#[cfg(not(target_arch = "wasm32"))]
fn routed_map(device: &CpuDevice, host: &[f32]) {
    let mapped = triple::map(host).expect("the routed map on the CPU");
    let mapped_on = triple::map_on(device, host).expect("the routed map_on on CpuDevice");
    let wanted: Vec<f32> = host.iter().map(|x| x * 3.0).collect();
    assert_eq!(mapped, wanted, "the routed map must equal the host loop");
    assert_eq!(
        mapped_on, wanted,
        "the routed map_on must equal the host loop"
    );
    println!("CPU target: routed map matches (routed::triple, {ELEMENTS} elements)");
}

/// The blocking `map` and `map_on` are not emitted on `wasm32` (nothing may block there, D69), so the wasm32
/// measurement of the routed crate builds this binary without calling them.
#[cfg(target_arch = "wasm32")]
fn routed_map(_device: &CpuDevice, _host: &[f32]) {}
