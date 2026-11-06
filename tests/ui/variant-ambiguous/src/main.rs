// Two variants with one set of capabilities (GUST-E0743): a device that has the set cannot choose. `vulkan` is an
// alias for `subgroups`.

#[gust::gpu]
mod reduce {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn sum(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }

    #[variant(of = sum, requires(subgroups))]
    pub fn first(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }

    #[variant(of = sum, requires(vulkan))]
    pub fn second(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }
}

fn main() {}
