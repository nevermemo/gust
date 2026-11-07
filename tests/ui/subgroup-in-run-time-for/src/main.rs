// A subgroup operation in a `for` loop with a run-time bound is GUST-E0747: the trip count would differ per
// invocation, so some invocations would not reach the call.

#[gust::gpu]
mod scan {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn totals(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }

    #[variant(of = totals, requires(subgroups))]
    pub fn totals_subgroups(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let mut x = 0u32;
        for step in 0u32..data[id.x] {
            x += subgroup_add(step);
        }
        out[id.x] = x;
    }
}

fn main() {}
