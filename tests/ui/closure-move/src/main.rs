// A move closure (m9.c, GUST-E0329): move decides how Rust captures, and a closure here captures nothing.
// Expected: one error at the let of f, line 8.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>) {
        let f = move |x: f32| -> f32 { x * 2.0f32 };
        out[id.x] = f(1.0f32);
    }
}

fn main() {}
