// `..` in a tuple destructuring is GUST-E0319. expected.txt pins the code and the line of the `let`.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let v = input[id.x];
        let (a, ..) = (v, v, v);
        out[id.x] = a;
    }
}

fn main() {}
