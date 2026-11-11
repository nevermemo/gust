// The array length is a literal, not an expression, GUST-E0216 (a `const` item is GUST-E0102 in a GPU module).
// expected.txt pins the code and the line of the type.

#[gust::gpu]
mod block {
    fn first(a: [u32; 2 + 2]) -> u32 {
        1u32
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = id.x;
    }
}

fn main() {}
