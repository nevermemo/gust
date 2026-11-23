#[gpu_module]
mod blocks {
    /// The sum of every tile of 64 elements, one tile per workgroup: each lane loads one element into workgroup memory,
    /// then six rounds of a tree (a barrier after the load and after every round) leave the tile's sum in cell 0, which
    /// lane 0 writes to `sums[gid]`. A lane past the end of the input holds 0, so the last tile may be partial.
    ///
    /// Workgroup memory and barriers have no CPU twin (SEMANTICS, D101): the kernel says `cpu = "unsupported"`, and its
    /// result is checked against a host reference on the GPU instead (`main`).
    #[kernel(workgroup_size(64, 1, 1), cpu = "unsupported")]
    pub fn tile_sum(
        _id: SV_DispatchThreadID,
        lid: LocalId,
        gid: GroupId,
        input: StructuredBuffer<uint>,
        mut sums: RWStructuredBuffer<uint>,
        mut scratch: Workgroup<[u32; 64]>,
    ) {
        let i = gid.x * 64u32 + lid.x;
        let mut value = 0u32;
        if i < input.len() {
            value = input[i];
        }
        scratch[lid.x] = value;
        workgroup_barrier();
        for round in 0u32..6u32 {
            let stride = 32u32 >> round;
            if lid.x < stride {
                scratch[lid.x] += scratch[lid.x + stride];
            }
            workgroup_barrier();
        }
        if lid.x == 0u32 {
            sums[gid.x] = scratch[0u32];
        }
    }

    /// The inclusive scan of every tile of 64 elements (Hillis and Steele): round `r` adds the value `2^r` lanes to the
    /// left. Every lane reads its neighbour, a barrier follows, every lane writes its own cell, a barrier follows: the
    /// barriers sit at the top level of the loop body, where every lane reaches them. The tile's total is the last
    /// lane's value (lanes past the input hold 0), written to `totals[gid]`.
    #[kernel(workgroup_size(64, 1, 1), cpu = "unsupported")]
    pub fn tile_scan(
        _id: SV_DispatchThreadID,
        lid: LocalId,
        gid: GroupId,
        input: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
        mut totals: RWStructuredBuffer<uint>,
        mut scratch: Workgroup<[u32; 64]>,
    ) {
        let i = gid.x * 64u32 + lid.x;
        let mut value = 0u32;
        if i < input.len() {
            value = input[i];
        }
        scratch[lid.x] = value;
        workgroup_barrier();
        for round in 0u32..6u32 {
            let offset = 1u32 << round;
            let mut left = 0u32;
            if lid.x >= offset {
                left = scratch[lid.x - offset];
            }
            workgroup_barrier();
            scratch[lid.x] += left;
            workgroup_barrier();
        }
        if i < out.len() {
            out[i] = scratch[lid.x];
        }
        if lid.x == 63u32 {
            totals[gid.x] = scratch[63u32];
        }
    }

    /// Adds to every element of tile `t > 0` the inclusive scan of the tile totals before it (`offsets[t - 1]`). No
    /// workgroup memory, no barrier: this kernel has a CPU twin like the ones of the other entries.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn add_offsets(
        id: SV_DispatchThreadID,
        offsets: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < out.len() {
            let tile = i / 64u32;
            if tile > 0u32 {
                out[i] += offsets[tile - 1u32];
            }
        }
    }
}
