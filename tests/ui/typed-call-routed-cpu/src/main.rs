// A module routed to the CPU is a CPU body in any build (m8.r1, D65 to D67, D137): the tag `physics` is declared in this
// crate's Cargo.toml and routed by the variable of this fixture's env.txt, so the non-test `dispatch` on `gust::cpu::CpuDevice`
// has its `Run` impl although `cargo check` sets neither `cfg(test)` nor `--cfg gust_dual`. The same source with the
// variable unset (or `gpu`) is the E0277 of `typed-call-routed-gpu`. This must compile cleanly.

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
