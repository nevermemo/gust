// A conditional `break` in a loop that holds a barrier is GUST-E0416: some invocations would leave the loop while
// the others wait at the barrier. expected.txt pins the code and the line of the `break`.

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
                break;
            }
            workgroup_barrier();
        }
        partial[gid.x] = scratch[0u32] + input[id.x];
    }
}

fn main() {}
