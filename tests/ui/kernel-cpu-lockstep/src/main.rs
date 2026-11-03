// `cpu = "lockstep"` is the name of the CPU twin that a later slice adds for barrier kernels. Until then it is
// GUST-E0709, and the message says the twin is not available yet. expected.txt pins the code and the line of the
// option.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(256, 1, 1), cpu = "lockstep")]
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
