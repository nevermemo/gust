#[gust::gpu]
mod grid {
    // Multi-dimensional dispatch (m7.d). The extent of a call is `Extent { x, y, z }`; every kernel here has a
    // workgroup of 16 x 16 (or 4 x 4 x 4), so an extent that is not a multiple of it runs whole workgroups and the
    // invocations past the extent are guarded by the kernel, exactly as in one dimension.
    //
    // `size` holds the extent: it is an independent buffer, so its length (2 or 3) is not the number of invocations.
    //
    // One row-major cell per invocation, `i == id.y * width + id.x`: `coords[i] == id.y * 65536 + id.x`,
    // `locals[i] == lid.y * 16 + lid.x` and `groups[i] == gid.y * 65536 + gid.x`.
    #[kernel(workgroup_size(16, 16, 1))]
    pub fn cells(
        id: SV_DispatchThreadID,
        lid: LocalId,
        gid: GroupId,
        size: StructuredBuffer<u32>,
        mut coords: RWStructuredBuffer<u32>,
        mut locals: RWStructuredBuffer<u32>,
        mut groups: RWStructuredBuffer<u32>,
    ) {
        let width = size[0u32];
        let height = size[1u32];
        if id.x < width && id.y < height {
            let i = id.y * width + id.x;
            coords[i] = id.y * 65536u32 + id.x;
            locals[i] = lid.y * 16u32 + lid.x;
            groups[i] = gid.y * 65536u32 + gid.x;
        }
    }

    // Three dimensions: one packed coordinate per cell, `z * 1048576 + y * 1024 + x` (each below 1024), at the
    // row-major index `(id.z * height + id.y) * width + id.x`. The write is guarded by the length of `packed`, so
    // a test can launch the full 2^32 invocations of a 2048 x 2048 x 1024 extent over a buffer of a few elements.
    #[kernel(workgroup_size(4, 4, 4))]
    pub fn volume(id: SV_DispatchThreadID, size: StructuredBuffer<u32>, mut packed: RWStructuredBuffer<u32>) {
        let width = size[0u32];
        let height = size[1u32];
        let depth = size[2u32];
        if id.x < width && id.y < height && id.z < depth {
            let i = (id.z * height + id.y) * width + id.x;
            if i < packed.len() {
                packed[i] = id.z * 1048576u32 + id.y * 1024u32 + id.x;
            }
        }
    }

    // The last cell of the extent writes its own coordinate and nothing else, so a launch of 2^32 invocations needs
    // no buffer of that size: both buffers are independent.
    #[kernel(workgroup_size(16, 16, 1))]
    pub fn corner(id: SV_DispatchThreadID, size: StructuredBuffer<u32>, mut last: RWStructuredBuffer<u32>) {
        if id.x + 1u32 == size[0u32] && id.y + 1u32 == size[1u32] {
            last[0u32] = id.x;
            last[1u32] = id.y;
        }
    }
}
