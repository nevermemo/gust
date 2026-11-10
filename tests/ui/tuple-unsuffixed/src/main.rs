// An integer literal without a suffix in a tuple is GUST-E0326: Slang types it `int`, rustc infers the type from the later
// uses. expected.txt pins the code and the line of the literal.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let v = input[id.x];
        let t = (1, v);
        out[id.x] = t.1 + t.0;
    }
}

fn main() {}
