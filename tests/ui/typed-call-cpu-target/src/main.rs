// The positive twin of `typed-call-no-cpu-twin`: the same kernel, the same non-test `dispatch` on
// `gust::cpu::CpuDevice`, but built under `GUST_PROFILE=cpu` (this fixture's env.txt). `cargo check` sets neither
// `cfg(test)` nor `--cfg gust_dual`, so the impl exists only because the macro read the profile at expansion time and
// emitted it ungated (D21: the CPU target is a build-time choice). Without that, this crate fails with the E0277 that
// `typed-call-no-cpu-twin` expects. This must compile cleanly.

use gust::{Independent, cpu::CpuDevice};

#[gust::gpu]
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
