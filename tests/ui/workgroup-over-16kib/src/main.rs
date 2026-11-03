// All workgroup arrays of a kernel together may take 16384 bytes. Here 4097 `f32` take 16388: GUST-E0718. expected.txt
// pins the code and the line of the signature.

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
