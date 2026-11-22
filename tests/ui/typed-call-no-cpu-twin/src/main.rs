// The CPU twin exists only in test builds (`cfg(test)`) and under `--cfg gust_dual` (D26). `cargo check` sets
// neither, so a kernel dispatched on `gust::cpu::CpuDevice` has no `Run` impl and rustc reports the
// `#[diagnostic::on_unimplemented]` text of `Run`. This is the production-exclusion proof: a normal build of a
// user's crate contains no twin. The kernel is the one of `typed-call-ok`.

use gust::{Independent, cpu::CpuDevice};

#[gust::gpu_module]
mod kernels {
    // User parameters named `element_count` and `device`, a settings buffer and a read-write output.
    #[kernel]
    pub fn scale(
        id: SV_DispatchThreadID,
        element_count: StructuredBuffer<float>,
        settings: StructuredBuffer<float>,
        mut device: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < device.len() && i < element_count.len() && 0u32 < settings.len() {
            device[i] = element_count[i] * settings[0u32];
        }
    }
}

fn main() {
    let device = CpuDevice::new();
    let input = device.buffer(&[1.0f32, 2.0, 3.0, 4.0]);
    let settings = device.buffer(&[2.0f32]);
    let output = device.buffer(&[0.0f32; 4]);
    kernels::scale::dispatch(&device, 4, &input, Independent(&settings), &output).unwrap();
}
