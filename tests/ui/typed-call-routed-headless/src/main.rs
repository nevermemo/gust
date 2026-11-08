// The call-site consequence of a CPU route (m8.r1, ruling 5, D137): the tag is routed to the CPU by its `default`, so the
// macro emits `Target::Cpu` and no `impl gust::target::GpuKernel` for `scale`, and the typed `Run` impl of
// `HeadlessDevice` (bounded on the marker) is not there. Naming `HeadlessDevice` for the kernel does not compile; the error
// is at the call, with the wording of `#[diagnostic::on_unimplemented]` on `GpuKernel`, which names the kernel. The
// same call on `CpuDevice` is `typed-call-routed-default`, which compiles; with the tag routed to the GPU it is this
// call that compiles instead.

use gust_wgpu::{GpuBufferAccess, HeadlessDevice};

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

#[allow(dead_code)]
fn run(device: &HeadlessDevice) {
    let input = device.create_typed_buffer(
        "input",
        &[1.0f32, 2.0, 3.0, 4.0],
        GpuBufferAccess::ReadWrite,
    );
    let output = device.create_typed_buffer("output", &[0.0f32; 4], GpuBufferAccess::ReadWrite);
    kernels::scale::dispatch(device, 4, &input, &output).unwrap();
}

fn main() {}
