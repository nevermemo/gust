// A closure named with the reserved prefix (m9.c, GUST-E0901): hoisted closures use __gust_cl_.
// Expected: one error at the name, line 8.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>) {
        let __gust_cl_run_0 = |x: f32| -> f32 { x * 2.0f32 };
        out[id.x] = __gust_cl_run_0(1.0f32);
    }
}

fn main() {}
