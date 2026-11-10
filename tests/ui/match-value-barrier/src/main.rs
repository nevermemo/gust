// A barrier in an arm of a value `match` is GUST-E0414 like one in an `if`: an arm runs for some invocations only.
// expected.txt pins the code and the line of the barrier.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1), cpu = "unsupported")]
    pub fn run(
        id: SV_DispatchThreadID,
        lid: LocalId,
        input: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
        mut scratch: Workgroup<[u32; 64]>,
    ) {
        scratch[lid.x] = input[id.x];
        let x: u32 = match input[id.x] {
            0 => {
                workgroup_barrier();
                1u32
            }
            _ => 2u32,
        };
        out[id.x] = x + scratch[0u32];
    }
}

fn main() {}
