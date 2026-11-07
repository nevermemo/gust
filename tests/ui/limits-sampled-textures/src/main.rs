// A crate with `limits = "portable"` in its [package.metadata.gust] table is checked against the portable floor
// (WebGPU defaults, D143): GUST-E0746 names the limit and both values. expected.txt pins the code and the signature line.

#[gust::gpu]
mod kernels {
    #[kernel(workgroup_size(8, 8, 1))]
    pub fn k(id: SV_DispatchThreadID, t0: Texture2D<f32>, t1: Texture2D<f32>, t2: Texture2D<f32>, t3: Texture2D<f32>, t4: Texture2D<f32>, t5: Texture2D<f32>, t6: Texture2D<f32>, t7: Texture2D<f32>, t8: Texture2D<f32>, t9: Texture2D<f32>, t10: Texture2D<f32>, t11: Texture2D<f32>, t12: Texture2D<f32>, t13: Texture2D<f32>, t14: Texture2D<f32>, t15: Texture2D<f32>, t16: Texture2D<f32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = id.x;
    }
}

fn main() {}
