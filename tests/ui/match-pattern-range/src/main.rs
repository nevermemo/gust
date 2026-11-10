// An integer `match` lists literals; a range is not a pattern here, GUST-E0521. expected.txt pins the code and the
// line of the range.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let x: u32 = match input[id.x] {
            0u32..=3u32 => 1u32,
            _ => 2u32,
        };
        out[id.x] = x;
    }
}

fn main() {}
