// A barrier in a `for` nested in another `for` is GUST-E0414: only the statements of a top-level literal-range loop
// are accepted positions. expected.txt pins the code and the line of the barrier.

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
            for step in 0u32..4u32 {
                workgroup_barrier();
            }
        }
        partial[gid.x] = scratch[0u32] + input[id.x];
    }
}

fn main() {}
