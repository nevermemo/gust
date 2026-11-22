// A `new` of the user's in the module clashes with the constructor `#[pad]` generates, GUST-E0219.

#[gust::gpu]
mod block {
    #[pad]
    pub struct Params {
        pub a: f32,
    }

    impl Params {
        fn new() -> Params {
            Params { a: 0.0 }
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, params: Uniform<Params>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = params.a;
    }
}

fn main() {}
