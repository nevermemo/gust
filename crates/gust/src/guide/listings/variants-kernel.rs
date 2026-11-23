#[gpu_module]
mod stats {
    /// Sum, count over a limit and maximum of a stream, into three cells. The portable body: every invocation does its
    /// own atomic operations, so the three cells see one operation per element.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn tally(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<uint>,
        mut cells: RWStructuredBuffer<uint>,
    ) {
        if id.x < data.len() {
            let x = data[id.x];
            atomic_add(&mut cells[0u32], x);
            if x > 100u32 {
                atomic_add(&mut cells[1u32], 1u32);
            }
            atomic_max(&mut cells[2u32], x);
        }
    }

    /// The same kernel for a device with subgroup operations: the lanes of a subgroup combine their values first, and
    /// one lane of the subgroup does the atomic operations. The signature is the portable body's, token for token. An
    /// invocation past the end of the data holds 0 and still takes part in the subgroup operations (a 0 changes none of
    /// the three results), and every subgroup operation is the whole value of a `let` at the top level of the body.
    #[variant(of = tally, requires(subgroups))]
    pub fn tally_subgroups(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<uint>,
        mut cells: RWStructuredBuffer<uint>,
    ) {
        let mut x = 0u32;
        if id.x < data.len() {
            x = data[id.x];
        }
        let sum = subgroup_add(x);
        let over = subgroup_count(x > 100u32);
        let top = subgroup_max(x);
        let lane = subgroup_lane();
        if lane == 0u32 {
            atomic_add(&mut cells[0u32], sum);
            atomic_add(&mut cells[1u32], over);
            atomic_max(&mut cells[2u32], top);
        }
    }
}
