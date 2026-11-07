// A crate with `limits = "portable"` in its [package.metadata.gust] table is checked against the portable floor
// (WebGPU defaults, D143): GUST-E0746 names the limit and both values. expected.txt pins the code and the signature line.

#[gust::gpu]
mod kernels {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn k(id: SV_DispatchThreadID, s0: StructuredBuffer<u32>, s1: StructuredBuffer<u32>, s2: StructuredBuffer<u32>, s3: StructuredBuffer<u32>, s4: StructuredBuffer<u32>, s5: StructuredBuffer<u32>, s6: StructuredBuffer<u32>, s7: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = id.x;
    }
}

fn main() {}
