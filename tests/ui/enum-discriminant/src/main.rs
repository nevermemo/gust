// A discriminant of a GPU enum is an integer literal: a computed one would have to be evaluated by both Rust and Slang,
// GUST-E0122. expected.txt pins the code and the line of the discriminant.

#[gust::gpu]
mod block {
    #[derive(PartialEq, Eq)]
    pub enum Level {
        Low = 1,
        High = 2 + 3,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = input[id.x];
    }
}

fn main() {}
