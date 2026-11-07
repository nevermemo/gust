// A crate with `limits = "portable"` in its [package.metadata.gust] table is checked against the portable floor
// (WebGPU defaults, D143): GUST-E0746 names the limit and both values. expected.txt pins the code and the signature line.

#[gust::gpu]
mod kernels {
    #[kernel(workgroup_size(8, 8, 1))]
    pub fn k(id: SV_DispatchThreadID, m0: Sampler, m1: Sampler, m2: Sampler, m3: Sampler, m4: Sampler, m5: Sampler, m6: Sampler, m7: Sampler, m8: Sampler, m9: Sampler, m10: Sampler, m11: Sampler, m12: Sampler, m13: Sampler, m14: Sampler, m15: Sampler, m16: Sampler, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = id.x;
    }
}

fn main() {}
