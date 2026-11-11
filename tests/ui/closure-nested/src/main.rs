// A closure inside a closure (m9.c, GUST-E0332): each closure is a function of its own.
// Expected: one error at the inner closure, line 8.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>) {
        let f = |x: f32| -> f32 { let g = |y: f32| -> f32 { y }; g(x) };
        out[id.x] = f(1.0f32);
    }
}

fn main() {}
