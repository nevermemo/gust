// A subgroup operation under an `if` is GUST-E0747: only the invocations that take the branch would combine.

#[gust::gpu]
mod scan {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn totals(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }

    #[variant(of = totals, requires(subgroups))]
    pub fn totals_subgroups(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        if id.x < data.len() {
            out[id.x] = subgroup_add(data[id.x]);
        }
    }
}

fn main() {}
