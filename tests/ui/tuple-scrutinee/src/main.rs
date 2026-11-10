// A tuple is not a `match` scrutinee, GUST-E0515: there are no tuple patterns. expected.txt pins the code and the line of
// the scrutinee.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let v = input[id.x];
        let t = (v, v + 1u32);
        let mut r = 0u32;
        match t {
            (0, _) => {
                r = 1u32;
            }
            _ => {
                r = 2u32;
            }
        }
        out[id.x] = r;
    }
}

fn main() {}
