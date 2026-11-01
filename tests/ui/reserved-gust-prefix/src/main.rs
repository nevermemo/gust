// Names that begin `__gust_`, `__gpu_` or `__Gust` belong to GUST: the first and the last are the prefixes of the
// macro's own emitted text and of its rename of colliding names (`__gust_r_`), so a user name there could capture or
// be mistaken for it. The macro refuses them with E0901, on the identifier itself.

#[gust::gpu]
mod kernels {
    #[kernel]
    pub fn copy(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut output: RWStructuredBuffer<float>,
    ) {
        let __GustCount = input.len();
        if id.x < input.len() && id.x < output.len() {
            output[id.x] = input[id.x];
        }
    }
}

fn main() {}
