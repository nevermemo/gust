// A tuple element is a scalar, vector, matrix, struct, enum or tuple, GUST-E0213: not an array. expected.txt pins the
// code and the line of the element type.

#[gust::gpu]
mod block {
    fn pick(v: u32) -> ([u32; 2], u32) {
        ([v, v], v)
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = pick(input[id.x]).1;
    }
}

fn main() {}
