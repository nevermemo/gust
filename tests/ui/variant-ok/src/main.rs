// A kernel and a variant of it (GUST-E0740 to GUST-E0744 and GUST-E0750 are the ways this goes wrong). Both are
// plain bodies, so this crate compiles under every profile: the variant has the signature of its kernel, and rustc
// proves that with the const the macro emits (a mismatch the token rule missed would be E0308 at the variant).

#[gust::gpu]
mod reduce {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn sum(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }

    #[variant(of = sum, requires(subgroups))]
    pub fn sum_subgroups(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }
}

fn main() {}
