// A destructured name takes the per-binding checks of a plain `let`, GUST-E0901: `__gust_r_new` would capture the
// local `new`, which is renamed to it. expected.txt pins the code and the line of the binding.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let new = input[id.x] + 9u32;
        {
            let (__gust_r_new, z) = (5u32, 6u32);
            out[id.x] = new + z;
        }
    }
}

fn main() {}
