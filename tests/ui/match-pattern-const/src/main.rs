// A constant is not a pattern of an integer `match`, GUST-E0521: list the literals. expected.txt pins the code and the line
// of the path.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let x: u32 = match input[id.x] {
            0 => 1u32,
            u32::MAX => 2u32,
            _ => 3u32,
        };
        out[id.x] = x;
    }
}

fn main() {}
