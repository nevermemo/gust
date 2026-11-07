// A `return` in a kernel that calls a subgroup operation is GUST-E0415, as it is for a barrier: an invocation that
// leaves early never reaches the operation the others combine at.

#[gust::gpu]
mod scan {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn totals(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }

    #[variant(of = totals, requires(subgroups))]
    pub fn totals_subgroups(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        if id.x >= data.len() {
            return;
        }
        let total = subgroup_add(data[id.x]);
        out[id.x] = total;
    }
}

fn main() {}
