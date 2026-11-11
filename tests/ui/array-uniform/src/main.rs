// An array in a uniform: the std140 stride of an element is rounded up to 16, so no layout is promised, GUST-E0215.

#[gust::gpu]
mod block {
    pub struct Params {
        pub weights: [f32; 4],
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, params: Uniform<Params>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = params.weights[1];
    }
}

fn main() {}
