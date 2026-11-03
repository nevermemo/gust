// A conditional `continue` in a loop that holds a barrier is GUST-E0416: some invocations would skip the barrier of
// an iteration that the others wait at. expected.txt pins the code and the line of the `continue`.

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
        for round in 0u32..4u32 {
            if lid.x + round > 100u32 {
                continue;
            }
            workgroup_barrier();
        }
        partial[gid.x] = scratch[0u32] + input[id.x];
    }
}

fn main() {}
