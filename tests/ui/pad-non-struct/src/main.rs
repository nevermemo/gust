// `#[pad]` on an item that is no struct, GUST-E0219.

#[gust::gpu]
mod block {
    pub struct Params {
        pub a: f32,
        pub b: f32,
        pub c: f32,
        pub d: f32,
    }

    #[pad]
    fn helper(x: f32) -> f32 {
        x
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, params: Uniform<Params>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = helper(params.a);
    }
}

fn main() {}
