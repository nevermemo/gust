// `#[pad]` on a struct without named fields, GUST-E0219.

#[gust::gpu]
mod block {
    #[pad]
    pub struct Params(pub f32);

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, params: Uniform<Params>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = 1.0;
    }
}

fn main() {}
