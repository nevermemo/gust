// A mutable closure binding (m9.c, GUST-E0330): a closure is called by its name and is no value.
// Expected: one error at the let of f, line 8.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>) {
        let mut f = |x: f32| -> f32 { x * 2.0f32 };
        out[id.x] = f(1.0f32);
    }
}

fn main() {}
