// An enum is never stored: a buffer whose element is an enum is GUST-E0211. expected.txt pins the code and the line
// of the parameter type.

#[gust::gpu]
mod block {
    #[derive(PartialEq, Eq)]
    pub enum Level {
        Low,
        High,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, levels: StructuredBuffer<Level>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = id.x;
    }
}

fn main() {}
