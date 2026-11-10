// A tuple has two or more elements, GUST-E0213: `(v,)` is a one-element tuple. expected.txt pins the code and the line.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let v = input[id.x];
        let t = (v,);
        out[id.x] = t.0;
    }
}

fn main() {}
