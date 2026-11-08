// The tag's `default = "cpu"` in Cargo.toml routes the module when no variable says otherwise, so plain `cargo` builds
// the CPU form (m8.r1, D66): the dispatch on `gust::cpu::CpuDevice` compiles with no variable, no `cfg(test)` and no
// `--cfg gust_dual`. This must compile cleanly.

use gust::cpu::CpuDevice;

#[gust::gpu(route = "physics")]
mod kernels {
    #[kernel]
    pub fn scale(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() && i < input.len() {
            out[i] = input[i] * 2.0;
        }
    }
}

fn main() {
    let device = CpuDevice::new();
    let input = device.buffer(&[1.0f32, 2.0, 3.0, 4.0]);
    let output = device.buffer(&[0.0f32; 4]);
    kernels::scale::dispatch(&device, 4, &input, &output).unwrap();
}
