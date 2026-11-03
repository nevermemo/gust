#[gust::gpu]
mod compact {
    // m7.r3: the kernels of `gust_algo::compact` (plan section 5.4). The sums and the scan of a compaction are the
    // `reduce` and `scan` kernels; these are the three element-wise steps around them.
    //
    // All of them use the block-strided shape of the reduce passes: the host launches `256 * ceil(n / 65536)`
    // invocations, so even the longest input stays inside the 1D workgroup ceiling (an invocation per element would
    // not at 2^24 + 1 elements). Workgroup w owns the block [w * 65536, (w + 1) * 65536) and lane t visits
    // `block + t + 256 * k` for k in 0..256 while the index is below n: coalesced across the lanes, at most 256
    // elements per lane. Every write goes to the element the lane is visiting (or, for the scatter, to the slot that
    // element owns), so there is no race and no atomic.
    //
    //   flags_to_mask_u32   mask[i] = 1 when flags[i] is not zero, else 0: any non-zero flag counts as set, and the
    //                       count and the scan that follow see only 0 and 1;
    //   scatter_indices_u32 if mask[i] == 1 { indices[positions[i]] = i }, positions being the exclusive scan of the
    //                       mask: the scan is injective on the flagged set, so each output slot has one writer;
    //   gather_*            out[j] = values[indices[j]] for j < count[0]; slots at and past the count, and an index
    //                       that is not below values.len(), leave the zero the output was created with.

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn flags_to_mask_u32(
        id: SV_DispatchThreadID,
        flags: StructuredBuffer<uint>,
        mut mask: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        let n = mask.len();
        let first = (i / 256u32) * 65536u32 + i % 256u32;
        for k in 0u32..256u32 {
            let index = first + k * 256u32;
            if index >= n {
                break;
            }
            let mut bit = 0u32;
            if flags[index] != 0u32 {
                bit = 1u32;
            }
            mask[index] = bit;
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn scatter_indices_u32(
        id: SV_DispatchThreadID,
        mask: StructuredBuffer<uint>,
        positions: StructuredBuffer<uint>,
        mut indices: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        let n = mask.len();
        let slots = indices.len();
        let first = (i / 256u32) * 65536u32 + i % 256u32;
        for k in 0u32..256u32 {
            let index = first + k * 256u32;
            if index >= n {
                break;
            }
            if mask[index] == 1u32 {
                let slot = positions[index];
                if slot < slots {
                    indices[slot] = index;
                }
            }
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn gather_u32(
        id: SV_DispatchThreadID,
        values: StructuredBuffer<uint>,
        indices: StructuredBuffer<uint>,
        count: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        let n = out.len();
        let live = count[0u32];
        let first = (i / 256u32) * 65536u32 + i % 256u32;
        for k in 0u32..256u32 {
            let j = first + k * 256u32;
            if j >= n || j >= live {
                break;
            }
            let source = indices[j];
            if source < values.len() {
                out[j] = values[source];
            }
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn gather_i32(
        id: SV_DispatchThreadID,
        values: StructuredBuffer<int>,
        indices: StructuredBuffer<uint>,
        count: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<int>,
    ) {
        let i = id.x;
        let n = out.len();
        let live = count[0u32];
        let first = (i / 256u32) * 65536u32 + i % 256u32;
        for k in 0u32..256u32 {
            let j = first + k * 256u32;
            if j >= n || j >= live {
                break;
            }
            let source = indices[j];
            if source < values.len() {
                out[j] = values[source];
            }
        }
    }

    #[kernel(workgroup_size(256, 1, 1))]
    pub fn gather_f32(
        id: SV_DispatchThreadID,
        values: StructuredBuffer<float>,
        indices: StructuredBuffer<uint>,
        count: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        let n = out.len();
        let live = count[0u32];
        let first = (i / 256u32) * 65536u32 + i % 256u32;
        for k in 0u32..256u32 {
            let j = first + k * 256u32;
            if j >= n || j >= live {
                break;
            }
            let source = indices[j];
            if source < values.len() {
                out[j] = values[source];
            }
        }
    }
}
