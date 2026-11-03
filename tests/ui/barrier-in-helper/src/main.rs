// A barrier in a helper is GUST-E0414: a helper may be called under a condition, so not every invocation would reach
// it. expected.txt pins the code and the line of the barrier.

#[gust::gpu]
mod block {
    fn sync() {
        workgroup_barrier();
    }

    #[kernel(workgroup_size(256, 1, 1), cpu = "unsupported")]
    pub fn reduce(
        id: SV_DispatchThreadID,
        lid: LocalId,
        gid: GroupId,
        input: StructuredBuffer<f32>,
        mut partial: RWStructuredBuffer<f32>,
        mut scratch: Workgroup<[f32; 256]>,
    ) {
        scratch[lid.x] = 1.0f32;
        sync();
        partial[gid.x] = scratch[0u32] + input[id.x];
    }
}

fn main() {}
