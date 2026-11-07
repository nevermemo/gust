// A subgroup operation on the right of `&&` is GUST-E0747: the operator short-circuits, so not every invocation
// reaches it.

#[gust::gpu]
mod scan {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn votes(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }

    #[variant(of = votes, requires(subgroups))]
    pub fn votes_subgroups(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let p = data[id.x] > 3u32;
        let both = p && subgroup_all(p);
        if both {
            out[id.x] = 1u32;
        }
    }
}

fn main() {}
