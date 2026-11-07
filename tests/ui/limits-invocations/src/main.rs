// A crate with `limits = "portable"` in its [package.metadata.gust] table is checked against the portable floor
// (WebGPU defaults, D143): GUST-E0746 names the limit and both values. expected.txt pins the code and the signature line.

#[gust::gpu]
mod kernels {
    #[kernel(workgroup_size(17, 16, 1))]
    pub fn k(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = id.x;
    }
}

fn main() {}
