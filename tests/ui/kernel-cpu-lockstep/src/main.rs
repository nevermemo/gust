// `cpu = "lockstep"` would name a CPU twin for barrier kernels, and GUST defines none (D118): a barrier kernel says
// `cpu = "unsupported"` (GUST-E0717), and `"lockstep"` is GUST-E0709. expected.txt pins the code, the message and the
// line of the option.

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
