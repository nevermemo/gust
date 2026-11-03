// Control case for the workgroup rules (m7.g1): a block reduce with workgroup memory and barriers at the two accepted
// positions, a statement of the kernel body and a statement of a top-level `for` with two literal bounds. This crate
// must compile, so the validator, the lowering and the preserved Rust all accept it.

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
        let i = id.x;
        scratch[lid.x] = 0.0f32;
        if i < input.len() {
            scratch[lid.x] = input[i];
        }
        workgroup_barrier();
        for round in 0u32..8u32 {
            let s = 128u32 >> round;
            if lid.x < s {
                scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s];
            }
            workgroup_barrier();
        }
        if lid.x == 0u32 {
            partial[gid.x] = scratch[0u32];
        }
    }
}

fn main() {}
