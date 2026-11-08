// Step E-j (m8.r1): runs the CPU-routed `scale` of the library on `CpuDevice` and compares it with a host loop.

use gust::cpu::CpuDevice;
use routed::kernels;

const ELEMENTS: usize = 1024;

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
}
