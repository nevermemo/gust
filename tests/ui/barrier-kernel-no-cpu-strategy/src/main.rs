// A kernel with workgroup memory or a barrier must say `cpu = "unsupported"`: the CPU twin of this release runs the
// invocations one after the other and cannot honour a barrier. This is GUST-E0717; expected.txt pins the code and the
// line of the signature.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn reduce(
        id: SV_DispatchThreadID,
        lid: LocalId,
        gid: GroupId,
        input: StructuredBuffer<f32>,
        mut partial: RWStructuredBuffer<f32>,
        mut scratch: Workgroup<[f32; 256]>,
    ) {
        scratch[lid.x] = 1.0f32;
        workgroup_barrier();
        partial[gid.x] = scratch[0u32] + input[id.x];
    }
}

fn main() {}
