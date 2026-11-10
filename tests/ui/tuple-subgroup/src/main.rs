// A subgroup operation inside a tuple element is GUST-E0747: it is allowed only as the whole value of a `let` or an
// assignment. expected.txt pins the code and the line of the call.

#[gust::gpu]
mod scan {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn totals(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }

    #[variant(of = totals, requires(subgroups))]
    pub fn totals_subgroups(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let t = (subgroup_add(data[id.x]), 1u32);
        out[id.x] = t.0 + t.1;
    }
}

fn main() {}
