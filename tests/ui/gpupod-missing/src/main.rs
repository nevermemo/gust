// A buffer element type declared outside the `#[gust::gpu_module]` module is not a `GpuPod`: the
// macro accepts the path, so rustc reports the missing trait implementation, and
// `#[diagnostic::on_unimplemented]` on `GpuPod` supplies the wording checked by expected.txt.

#[derive(Clone, Copy)]
struct Outside {
    value: u32,
}

#[gust::gpu_module]
mod kernels {
    #[kernel]
    pub fn run(id: SV_DispatchThreadID, out: RWStructuredBuffer<super::Outside>) {
        let i = id.x;
        if i < out.len() {}
    }
}

fn main() {
    let _ = Outside { value: 0 }.value;
}
