// A crate without `limits = "portable"` is not checked against the floor (D143): a 512-wide workgroup and 9 storage
// buffers compile here, as they did before the setting existed. This crate must compile.

#[gust::gpu]
mod kernels {
    #[kernel(workgroup_size(512, 1, 1))]
    pub fn k(id: SV_DispatchThreadID, s0: StructuredBuffer<u32>, s1: StructuredBuffer<u32>, s2: StructuredBuffer<u32>, s3: StructuredBuffer<u32>, s4: StructuredBuffer<u32>, s5: StructuredBuffer<u32>, s6: StructuredBuffer<u32>, s7: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = id.x;
    }
}

fn main() {}
