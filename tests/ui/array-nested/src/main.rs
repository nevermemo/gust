// An array of arrays is GUST-E0217 (nested arrays are not in this slice). expected.txt pins the code and the line.

#[gust::gpu]
mod block {
    fn first(a: [[u32; 2]; 2]) -> u32 {
        1u32
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = id.x;
    }
}

fn main() {}
