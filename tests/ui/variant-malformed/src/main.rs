// A `#[variant]` with an option it does not have (GUST-E0740): the workgroup size, the shape and the CPU form are
// the kernel's. expected.txt pins the code and the line of the attribute.

#[gust::gpu]
mod reduce {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn sum(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }

    #[variant(of = sum, workgroup_size(8, 1, 1))]
    pub fn sum_subgroups(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }
}

fn main() {}
