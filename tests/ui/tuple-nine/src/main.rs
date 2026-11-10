// A tuple has at most eight elements, GUST-E0213. expected.txt pins the code and the line of the nine-element type.

#[gust::gpu]
mod block {
    fn wide(t: (u32, u32, u32, u32, u32, u32, u32, u32, u32)) -> u32 {
        t.0
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = input[id.x];
    }
}

fn main() {}
