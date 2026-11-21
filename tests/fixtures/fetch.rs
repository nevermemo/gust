#[gust::gpu]
mod fetch {
    // m11.fetch: the old value of `atomic_add`, bound by a `let`. `workgroup_size(1, 1, 1)` makes the dispatch
    // produce exactly one invocation per item, as in the `atomics` fixture.

    /// Each item takes the next free slot of its cell: `slot = atomic_add(&mut counts[cell], 1)`, then writes its own
    /// id at `out[cell * 1024 + slot]`. Which item gets which slot is schedule-dependent; the set of ids of a cell,
    /// the slots `0..count` and the count are not.
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn scatter(
        id: SV_DispatchThreadID,
        keys: StructuredBuffer<uint>,
        mut counts: RWStructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let cell = keys[id.x];
        let slot = atomic_add(&mut counts[cell], 1u32);
        out[cell * 1024u32 + slot] = id.x;
    }

    /// One signed cell, every invocation adds 3 and keeps the old value: the values are a permutation of
    /// `start, start + 3, ..`.
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn ticket_signed(
        id: SV_DispatchThreadID,
        mut cell: RWStructuredBuffer<int>,
        mut out: RWStructuredBuffer<int>,
    ) {
        let old: i32 = atomic_add(&mut cell[0u32], 3i32);
        out[id.x] = old;
    }
}
