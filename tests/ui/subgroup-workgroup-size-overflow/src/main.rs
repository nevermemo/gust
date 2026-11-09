// A workgroup_size has no upper bound. The invocation count of this one, 4294967295 x 4294967295 x 4, is over what a
// u64 holds, and GUST-E0751 still reports it exactly (the macro once overflowed there and panicked, m9.0).

#[gust::gpu]
mod scan {
    #[kernel(workgroup_size(4294967295, 4294967295, 4), requires(subgroups))]
    pub fn totals(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = subgroup_add(data[id.x]);
    }
}

fn main() {}
