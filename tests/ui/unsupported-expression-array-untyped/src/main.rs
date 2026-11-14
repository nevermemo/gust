// An array literal in an untyped `let` is GUST-E0811: slangc cannot infer the type of `{ .. }`. expected.txt pins the
// code, the whole message and the line of the literal (line 9).

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let i = id.x;
        let a = [1u32, 2u32];
        out[i] = a[0];
    }
}

fn main() {}
