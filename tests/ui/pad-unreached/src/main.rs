// `#[pad]` on a struct no uniform reaches, GUST-E0219.

#[gust::gpu]
mod block {
    pub struct Params {
        pub a: f32,
        pub b: f32,
        pub c: f32,
        pub d: f32,
    }

    #[pad]
    pub struct Unused {
        pub a: f32,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, params: Uniform<Params>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = params.a;
    }
}

fn main() {}
