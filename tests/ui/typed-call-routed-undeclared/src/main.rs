// A route tag must be declared in `[package.metadata.gust.routes]` of the crate that uses it (m8.r1, D65 to D67): this
// crate declares `physics` and routes the module to `render`. GUST-E0748 is a macro error, so `code: none`; it is
// reported at the tag's string literal (line 5), and names the tags the crate does declare.

#[gust::gpu(route = "render")]
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
