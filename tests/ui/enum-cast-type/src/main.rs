// An enum casts with `as` to its repr type only (`u32` here), GUST-E0122. expected.txt pins the code and the line of
// the wrong target type.

#[gust::gpu]
mod block {
    #[derive(PartialEq, Eq)]
    pub enum Level {
        Low,
        High,
    }

    fn index(level: Level) -> i32 {
        level as i32
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = input[id.x];
    }
}

fn main() {}
