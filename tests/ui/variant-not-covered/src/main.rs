// A kernel that requires `subgroups` and has no body for the default profiles (GUST-E0744): `native` and `web` have
// no capability. A variant that requires less, or a `profiles` list with the capability, would cover them.

#[gust::gpu]
mod reduce {
    #[kernel(workgroup_size(64, 1, 1), requires(subgroups))]
    pub fn sum(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }
}

fn main() {}
