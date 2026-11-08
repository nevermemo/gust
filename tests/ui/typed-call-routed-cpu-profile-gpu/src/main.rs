// A build under GUST_PROFILE=cpu holds the CPU target only (D21), so a variable that routes a tag to the GPU cannot be
// honoured (m8.r1, D65): GUST-E0749, a macro error (`code: none`) reported at the string literal of the route (line 4).

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

fn main() {}
