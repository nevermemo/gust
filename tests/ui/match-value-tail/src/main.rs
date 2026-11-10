// A `match` that gives a value as the tail of a function is GUST-E0419: bind it to a typed `let` and return the local.
// expected.txt pins the code and the line of the `match`.

#[gust::gpu]
mod block {
    fn pick(v: u32) -> u32 {
        match v {
            0 => 1u32,
            _ => 2u32,
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = pick(input[id.x]);
    }
}

fn main() {}
