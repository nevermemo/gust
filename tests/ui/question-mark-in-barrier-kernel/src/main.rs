// The `?` operator is rejected everywhere, GUST-E0311, so a barrier kernel adds no second error for the early exit
// that a `?` in a helper it calls would be. expected.txt pins the code and the line of the `?`.

#[gust::gpu]
mod block {
    fn half(x: u32) -> Option<u32> {
        Some(x / 2u32)
    }

    fn quarter(x: u32) -> Option<u32> {
        let h = half(x)?;
        half(h)
    }

    #[kernel(workgroup_size(256, 1, 1), cpu = "unsupported")]
    pub fn reduce(
        id: SV_DispatchThreadID,
        lid: LocalId,
        gid: GroupId,
        input: StructuredBuffer<f32>,
        mut partial: RWStructuredBuffer<f32>,
        mut scratch: Workgroup<[f32; 256]>,
    ) {
        scratch[lid.x] = quarter(lid.x).unwrap_or(0u32) as f32;
        workgroup_barrier();
        partial[gid.x] = scratch[0u32] + input[id.x];
    }
}

fn main() {}
