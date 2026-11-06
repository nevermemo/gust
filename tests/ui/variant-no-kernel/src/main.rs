// A variant of a function that is no kernel of the module (GUST-E0742).

#[gust::gpu]
mod reduce {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn sum(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }

    #[variant(of = prod, requires(subgroups))]
    pub fn sum_subgroups(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }
}

fn main() {}
