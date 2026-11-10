// An enum is never stored: it has no `GpuPod` layout, so a struct field of an enum type is GUST-E0211. expected.txt pins
// the code and the line of the field type.

#[gust::gpu]
mod block {
    #[derive(PartialEq, Eq)]
    pub enum Level {
        Low,
        High,
    }

    pub struct Cell {
        pub level: Level,
        pub count: u32,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = input[id.x];
    }
}

fn main() {}
