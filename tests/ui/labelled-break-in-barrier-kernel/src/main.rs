// A labelled loop is rejected in every kernel, GUST-E0403, so a barrier kernel adds no second error for the labelled
// `break` that leaves it. expected.txt pins the code and the line of the label.

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
        'rounds: for round in 0u32..4u32 {
            for step in 0u32..2u32 {
                if step + round > 100u32 {
                    break 'rounds;
                }
            }
        }
        workgroup_barrier();
        partial[gid.x] = scratch[0u32] + input[id.x];
    }
}

fn main() {}
