// A subgroup operation in a `while` loop is GUST-E0402: the loop is refused in every kernel, and the iteration count
// would differ per invocation. This pins the `while` rule (E0402), not a subgroup rule; the run-time-bound `for`, which
// is a subgroup rule (E0747), is the fixture subgroup-in-run-time-for.

#[gust::gpu]
mod scan {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn totals(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = data[id.x];
    }

    #[variant(of = totals, requires(subgroups))]
    pub fn totals_subgroups(id: SV_DispatchThreadID, data: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let mut x = data[id.x];
        while x < 100u32 {
            x += subgroup_add(x);
        }
        out[id.x] = x;
    }
}

fn main() {}
