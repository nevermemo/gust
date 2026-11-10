// A `match` that gives a value is a typed `let`, GUST-E0419: the Slang local is declared before its `switch` assigns
// it, so it needs its type. expected.txt pins the code and the line of the `let`.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let x = match input[id.x] {
            0 => 1u32,
            _ => 2u32,
        };
        out[id.x] = x;
    }
}

fn main() {}
