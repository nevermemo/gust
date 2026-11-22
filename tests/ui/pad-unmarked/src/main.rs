// A struct a uniform reaches, without `#[pad]`, keeps GUST-E0209; the message names the attribute.

#[gust::gpu]
mod block {
    pub struct Params {
        pub a: f32,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, params: Uniform<Params>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = params.a;
    }
}

fn main() {}
