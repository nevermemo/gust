// A barrier in a `while` loop: the loop is rejected for its own sake, GUST-E0402 (a `while` has no trip count fixed
// at entry), and a barrier kernel adds no second error. expected.txt pins the code and the line of the `while`.

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
        while lid.x < 8u32 {
            workgroup_barrier();
        }
        partial[gid.x] = scratch[0u32] + input[id.x];
    }
}

fn main() {}
