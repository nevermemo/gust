// The accepted forms: every access of a texture, a sampler, the sizes, a kernel without a sampler and one with two.
// This crate must compile, so the expansion (descriptor, typed call, CPU twin) is checked by rustc.

#[gust::gpu]
mod shade {
    #[kernel(workgroup_size(8, 8, 1))]
    pub fn blur(id: SV_DispatchThreadID, image: Texture2D<Vec4>, bilinear: Sampler, mut out: WriteTexture2D<Rgba32Float>) {
        let u = (id.x as f32 + 0.5f32) / (out.width() as f32);
        let v = (id.y as f32 + 0.5f32) / (out.height() as f32);
        out.store(id.x, id.y, image.sample(bilinear, u, v));
    }

    #[kernel(workgroup_size(8, 8, 1))]
    pub fn invert(id: SV_DispatchThreadID, mut image: StorageTexture2D<R32Float>) {
        if id.x < image.width() && id.y < image.height() {
            let value = image.load(id.x, id.y);
            image.store(id.x, id.y, 1.0f32 - value);
        }
    }

    #[kernel(workgroup_size(8, 8, 1))]
    pub fn mix(id: SV_DispatchThreadID, a: Texture2D<f32>, b: Texture2D<UVec2>, near: Sampler, far: Sampler, mut out: WriteTexture2D<Rg32Float>) {
        let first = a.sample(near, 0.25f32, 0.25f32);
        let second = a.sample(far, 0.75f32, 0.75f32);
        let counts = b.load(id.x, id.y);
        out.store(id.x, id.y, Vec2::new(first + counts.x as f32, second + counts.y as f32));
    }
}

fn main() {}
