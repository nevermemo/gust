// A subgroup operation in a helper function is GUST-E0745: the helper is called under conditions it cannot see, so
// the operation belongs in the kernel body.

#[gust::gpu]
mod scan {
    fn total(value: u32) -> u32 {
        subgroup_add(value)
    }

    #[kernel(workgroup_size(64, 1, 1), requires(subgroups))]
    pub fn totals(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = total(data[id.x]);
    }
}

fn main() {}
