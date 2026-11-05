//! Control case for the `missing_docs` lint (slice mr.b2): a crate with `#![deny(missing_docs)]` compiles over a
//! `#[gust::gpu]` module. Every item below that the user writes has a doc comment; every item that the macro generates
//! (the kernel handles, `DESCRIPTOR`, `MODULE_DESCRIPTOR`, the typed calls) is documented or `#[doc(hidden)]` itself,
//! so the lint asks the user for nothing else. The module holds a grid kernel with a uniform, a pixel kernel, a texture
//! kernel with a sampler and a workgroup kernel. This crate must compile.
#![deny(missing_docs)]

/// Kernels of every shape and parameter kind.
#[gust::gpu]
pub mod kernels {
    /// The uniform of `scale`.
    pub struct Params {
        /// The factor.
        pub gain: f32,
        /// The offset.
        pub bias: f32,
        /// Padding to 16 bytes.
        pub pad0: f32,
        /// Padding to 16 bytes.
        pub pad1: f32,
    }

    /// A grid kernel with a uniform.
    #[kernel]
    pub fn scale(id: SV_DispatchThreadID, params: Uniform<Params>, mut output: RWStructuredBuffer<float>) {
        let i = id.x;
        if i < output.len() {
            output[i] = params.gain * output[i] + params.bias;
        }
    }

    /// A pixel kernel: one fragment per element.
    #[kernel(shape = pixel)]
    pub fn paint(id: SV_DispatchThreadID, input: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = input[id.x];
    }

    /// A kernel with a sampled texture, a sampler and a write-only storage texture.
    #[kernel(workgroup_size(8, 8, 1))]
    pub fn blur(id: SV_DispatchThreadID, image: Texture2D<Vec4>, bilinear: Sampler, mut out: WriteTexture2D<Rgba32Float>) {
        let u = (id.x as f32 + 0.5f32) / (out.width() as f32);
        let v = (id.y as f32 + 0.5f32) / (out.height() as f32);
        out.store(id.x, id.y, image.sample(bilinear, u, v));
    }

    /// A kernel with workgroup memory and a barrier, which has no CPU twin.
    #[kernel(workgroup_size(64, 1, 1), cpu = "unsupported")]
    pub fn reduce(
        id: SV_DispatchThreadID,
        lid: LocalId,
        gid: GroupId,
        input: StructuredBuffer<f32>,
        mut partial: RWStructuredBuffer<f32>,
        mut scratch: Workgroup<[f32; 64]>,
    ) {
        scratch[lid.x] = 0.0f32;
        if id.x < input.len() {
            scratch[lid.x] = input[id.x];
        }
        workgroup_barrier();
        if lid.x == 0u32 {
            partial[gid.x] = scratch[0u32];
        }
    }
}

fn main() {}
