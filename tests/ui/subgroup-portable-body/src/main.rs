// A subgroup operation in a portable body (a `#[kernel]` with no `requires(subgroups)`) is GUST-E0745: the body
// also runs on the CPU target and on devices without subgroup operations. Write a `#[variant]` instead.

#[gust::gpu]
mod scan {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn totals(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = subgroup_add(data[id.x]);
    }
}

fn main() {}
