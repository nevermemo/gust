// Control case for the portable floor (D143): every limit of the table is reached exactly and nothing is over. The
// workgroup axes and their product cannot all sit at their maxima in one kernel, so three kernels take x, y and z; the
// first also holds 8 storage buffers and 16384 bytes of workgroup memory, the last 12 uniform buffers, 16 sampled
// textures, 16 samplers and 4 storage textures. This crate must compile.

#[gust::gpu]
mod kernels {
    pub struct Params {
        pub gain: f32,
        pub bias: f32,
        pub pad0: f32,
        pub pad1: f32,
    }

    #[kernel(workgroup_size(256, 1, 1), cpu = "unsupported")]
    pub fn wide(id: SV_DispatchThreadID, lid: LocalId, gid: GroupId, b0: StructuredBuffer<f32>, b1: StructuredBuffer<f32>, b2: StructuredBuffer<f32>, b3: StructuredBuffer<f32>, b4: StructuredBuffer<f32>, b5: StructuredBuffer<f32>, b6: StructuredBuffer<f32>,  mut out: RWStructuredBuffer<f32>, mut scratch: Workgroup<[f32; 4096]>) {
        scratch[lid.x] = b0[id.x] + b6[id.x];
        workgroup_barrier();
        out[gid.x] = scratch[0u32];
    }

    #[kernel(workgroup_size(1, 256, 1))]
    pub fn tall(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.y] = id.y;
    }

    #[kernel(workgroup_size(1, 1, 64))]
    pub fn deep(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.z] = id.z;
    }

    #[kernel(workgroup_size(8, 8, 1))]
    pub fn resources(id: SV_DispatchThreadID, u0: Uniform<Params>, u1: Uniform<Params>, u2: Uniform<Params>, u3: Uniform<Params>, u4: Uniform<Params>, u5: Uniform<Params>, u6: Uniform<Params>, u7: Uniform<Params>, u8: Uniform<Params>, u9: Uniform<Params>, u10: Uniform<Params>, u11: Uniform<Params>, t0: Texture2D<f32>, t1: Texture2D<f32>, t2: Texture2D<f32>, t3: Texture2D<f32>, t4: Texture2D<f32>, t5: Texture2D<f32>, t6: Texture2D<f32>, t7: Texture2D<f32>, t8: Texture2D<f32>, t9: Texture2D<f32>, t10: Texture2D<f32>, t11: Texture2D<f32>, t12: Texture2D<f32>, t13: Texture2D<f32>, t14: Texture2D<f32>, t15: Texture2D<f32>, m0: Sampler, m1: Sampler, m2: Sampler, m3: Sampler, m4: Sampler, m5: Sampler, m6: Sampler, m7: Sampler, m8: Sampler, m9: Sampler, m10: Sampler, m11: Sampler, m12: Sampler, m13: Sampler, m14: Sampler, m15: Sampler, mut w0: WriteTexture2D<R32Float>, mut w1: WriteTexture2D<R32Float>, mut w2: WriteTexture2D<R32Float>, mut w3: WriteTexture2D<R32Float>) {
        w0.store(id.x, id.y, t0.sample(m0, 0.5f32, 0.5f32) + u0.gain + u11.bias + t15.sample(m15, 0.25f32, 0.75f32));
    }
}

fn main() {}
