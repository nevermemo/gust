// The same kernel as workgroup-over-16kib, which is GUST-E0718 in a crate without the setting: in a crate with
// `limits = "portable"` the workgroup memory limit is the floor's, and GUST-E0746 stands where E0718 would.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(256, 1, 1), cpu = "unsupported")]
    pub fn reduce(
        id: SV_DispatchThreadID,
        lid: LocalId,
        gid: GroupId,
        input: StructuredBuffer<f32>,
        mut partial: RWStructuredBuffer<f32>,
        mut scratch: Workgroup<[f32; 4097]>,
    ) {
        scratch[lid.x] = 1.0f32;
        workgroup_barrier();
        partial[gid.x] = scratch[0u32] + input[id.x];
    }
}

fn main() {}
