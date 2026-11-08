// The tag lists `cpu` in `targets`, but the kernel has no CPU form (`cpu = "unsupported"`): the declaration is false for
// this kernel in every build, whatever the tag is routed to (m8.r1, GUST-E0744). A macro error (`code: none`), reported at
// the kernel's name (line 8).

#[gust::gpu(route = "physics")]
mod kernels {
    #[kernel(cpu = "unsupported")]
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

fn main() {}
