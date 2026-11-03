#[gust::gpu]
mod workgroup {
    // One workgroup sums up to 256 elements in shared memory: a tree of eight rounds, a barrier between the rounds.
    // The kernel has no CPU twin (`cpu = "unsupported"`): a barrier needs the lockstep twin of a later slice.
    // Every invocation writes its own cell before the first barrier (an element of the input, or 0.0 past its end),
    // and the barrier sits at the top level of the body and of the top-level literal-range loop.
    #[kernel(workgroup_size(256, 1, 1), cpu = "unsupported")]
    pub fn block_sum(
        id: SV_DispatchThreadID,
        lid: LocalId,
        gid: GroupId,
        input: StructuredBuffer<f32>,
        mut partial: RWStructuredBuffer<f32>,
        mut scratch: Workgroup<[f32; 256]>,
    ) {
        let i = id.x;
        scratch[lid.x] = 0.0f32;
        if i < input.len() {
            scratch[lid.x] = input[i];
        }
        workgroup_barrier();
        for round in 0u32..8u32 {
            let s = 128u32 >> round;
            if lid.x < s {
                scratch[lid.x] += scratch[lid.x + s];
            }
            workgroup_barrier();
        }
        if lid.x == 0u32 {
            partial[gid.x] = scratch[0u32];
        }
    }

    // The local and the group id, derived from the dispatch-thread id and the workgroup size, written out per
    // invocation: `locals[i] == i % 256` and `groups[i] == i / 256`. No barrier and no workgroup memory, so this
    // kernel keeps its CPU twin.
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn local_ids(
        id: SV_DispatchThreadID,
        lid: SV_GroupThreadID,
        gid: SV_GroupID,
        mut locals: RWStructuredBuffer<u32>,
        mut groups: RWStructuredBuffer<u32>,
    ) {
        if id.x < locals.len() {
            locals[id.x] = lid.x;
            groups[id.x] = gid.x;
        }
    }
}
