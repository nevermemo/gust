// A barrier in a closure body (m9.c, GUST-E0414): a closure is judged as a helper, so no barrier.
// Expected: one error naming the closure, line 7 (the closure body is checked with its function).

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1), cpu = "unsupported")]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>) {
        let f = |x: f32| -> f32 {
            workgroup_barrier();
            x
        };
        out[id.x] = f(1.0f32);
    }
}

fn main() {}
