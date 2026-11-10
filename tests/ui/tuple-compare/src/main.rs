// Comparing tuples is not supported yet, GUST-E0213 (a scope choice of slice m9.t). expected.txt pins the code and the
// line of the comparison.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let v = input[id.x];
        let t = (v, v + 1u32);
        let u = (v, v + 2u32);
        let mut r = 0u32;
        if t == u {
            r = 1u32;
        }
        out[id.x] = r;
    }
}

fn main() {}
