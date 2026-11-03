// A barrier under an `if` is GUST-E0414: the invocations for which the condition is false never reach it, so the
// others would wait for them. expected.txt pins the code and the line of the barrier.

#[gust::gpu]
mod block {
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
        if lid.x < 8u32 {
            workgroup_barrier();
        }
        partial[gid.x] = scratch[0u32] + input[id.x];
    }
}

fn main() {}
