// A value matched twice, in two spellings, is GUST-E0521: the second arm can never run, and in Slang two equal case
// labels would not compile. expected.txt pins the code and the line of the second arm.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let x: u32 = match input[id.x] {
            1 => 1u32,
            1u32 => 2u32,
            _ => 3u32,
        };
        out[id.x] = x;
    }
}

fn main() {}
