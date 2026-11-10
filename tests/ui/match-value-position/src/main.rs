// A `match` that gives a value as an operand is GUST-E0419: only a typed `let` takes the value form. expected.txt pins the
// code and the line of the `match`.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let x: u32 = 1u32
            + match input[id.x] {
                0 => 1u32,
                _ => 2u32,
            };
        out[id.x] = x;
    }
}

fn main() {}
