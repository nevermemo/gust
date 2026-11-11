// A closure with no types (m9.c, GUST-E0331): every parameter and the result are written out.
// Expected: one error at the parameter x, line 8.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>) {
        let f = |x| x * 2.0f32;
        out[id.x] = f(1.0f32);
    }
}

fn main() {}
