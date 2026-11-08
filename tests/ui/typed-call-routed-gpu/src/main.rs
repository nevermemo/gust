// The control of `typed-call-routed-cpu`: the same source, with the tag routed to the GPU by the variable of env.txt, which
// beats the `default = "cpu"` of this crate's Cargo.toml (a build that ignored the variable would compile). The kernel is a
// GPU kernel, its CPU twin exists only in test builds and under `--cfg gust_dual` (D26), and the non-test `dispatch` on
// `gust::cpu::CpuDevice` is the E0277 of `typed-call-no-cpu-twin`: a route flip changes which device the call sites compile against.

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
