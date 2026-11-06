// A kernel that requires `subgroups`, declared only for a profile that has it, built under one that does not
// (GUST-E0750). The workspace builds the fixtures under `GUST_PROFILE=dev`, which has no capability.

#[gust::gpu]
mod reduce {
    #[kernel(workgroup_size(64, 1, 1), requires(subgroups))]
    pub fn sum(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }
}

fn main() {}
