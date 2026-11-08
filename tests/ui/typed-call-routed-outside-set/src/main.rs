// The tag declares `targets = ["gpu"]` and the variable of env.txt routes it to the CPU: a target outside the tag's set
// (m8.r1, D65). GUST-E0749 is a macro error, so `code: none`; it is reported at the kernel (line 8), and the message
// names the variable that decided the route and the set.

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
