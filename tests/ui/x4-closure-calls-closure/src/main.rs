// Spike M9.s0 X4 (pre-registered, slice m9.c): a closure that calls an earlier closure captures it (GUST-E0329).
// Expected: one error at the let of quad, line 9, naming double.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>) {
        let double = |x: f32| -> f32 { x * 2.0f32 };
        let quad = |x: f32| -> f32 { double(double(x)) };
        out[id.x] = quad(1.0f32);
    }
}

fn main() {}
