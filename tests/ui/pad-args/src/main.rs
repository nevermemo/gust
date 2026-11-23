// `#[pad]` takes no arguments, GUST-E0219.

#[gust::gpu_module]
mod block {
    #[pad(4)]
    pub struct Params {
        pub a: f32,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, params: Uniform<Params>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = params.a;
    }
}

fn main() {}
