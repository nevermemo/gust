// A field array of arrays is not stored (the element of a stored array is a scalar, vector, matrix or struct), GUST-E0217.

#[gust::gpu]
mod block {
    pub struct Row {
        pub cells: [[u32; 2]; 2],
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = id.x;
    }
}

fn main() {}
