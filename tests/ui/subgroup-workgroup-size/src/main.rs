// A kernel that calls a subgroup operation with a workgroup of 32 invocations is GUST-E0751: the invocation count
// must be a multiple of 64 so that a workgroup is whole subgroups on every device.

#[gust::gpu]
mod scan {
    #[kernel(workgroup_size(32, 1, 1), requires(subgroups))]
    pub fn totals(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = subgroup_add(data[id.x]);
    }
}

fn main() {}
