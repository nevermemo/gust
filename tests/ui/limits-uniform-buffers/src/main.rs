// A crate with `limits = "portable"` in its [package.metadata.gust] table is checked against the portable floor
// (WebGPU defaults, D143): GUST-E0746 names the limit and both values. expected.txt pins the code and the signature line.

#[gust::gpu]
mod kernels {
    pub struct Params {
        pub gain: f32,
        pub bias: f32,
        pub pad0: f32,
        pub pad1: f32,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn k(id: SV_DispatchThreadID, u0: Uniform<Params>, u1: Uniform<Params>, u2: Uniform<Params>, u3: Uniform<Params>, u4: Uniform<Params>, u5: Uniform<Params>, u6: Uniform<Params>, u7: Uniform<Params>, u8: Uniform<Params>, u9: Uniform<Params>, u10: Uniform<Params>, u11: Uniform<Params>, u12: Uniform<Params>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = id.x;
    }
}

fn main() {}
