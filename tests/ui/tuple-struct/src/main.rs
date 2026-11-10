// A tuple struct is GUST-E0104: a struct has named fields. expected.txt pins the code and the line of the struct.

#[gust::gpu]
mod block {
    pub struct Pair(u32, f32);

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = input[id.x];
    }
}

fn main() {}
