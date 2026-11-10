// A GPU enum derives PartialEq, because `==`, `!=` and `match` compare enum values and the CPU twin must do the same,
// GUST-E0122. expected.txt pins the code and the line of the enum name.

#[gust::gpu]
mod block {
    pub enum Level {
        Low,
        High,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = input[id.x];
    }
}

fn main() {}
