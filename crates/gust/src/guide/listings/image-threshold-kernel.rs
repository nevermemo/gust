#[gpu_module]
mod threshold {
    /// A `grid` kernel on a storage texture: every texel brighter than one half becomes 1.0 and every other texel 0.0,
    /// in place. The dispatch is a whole number of 8 by 8 workgroups and so may be larger than the texture; the kernel
    /// guards against the texture's own size, which it reads from the texture like a buffer's length.
    #[kernel(workgroup_size(8, 8, 1))]
    pub fn binarize(id: SV_DispatchThreadID, mut image: StorageTexture2D<R32Float>) {
        if id.x < image.width() && id.y < image.height() {
            let value = image.load(id.x, id.y);
            if value > 0.5f32 {
                image.store(id.x, id.y, 1.0f32);
            } else {
                image.store(id.x, id.y, 0.0f32);
            }
        }
    }

    /// The same decision as a `pixel` kernel: the body runs once for each element `0..n` of a one-dimensional call
    /// (as a fragment shader; there is no workgroup size, no guard and no padding invocation). The element `i` is the
    /// texel `(i % width, i / width)`, read through a sampler at the centre of that texel, and the answer is the one
    /// output `flags[i]`, written on every path.
    #[kernel(shape = pixel)]
    pub fn flag(
        id: SV_DispatchThreadID,
        image: Texture2D<f32>,
        nearest: Sampler,
        mut flags: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        let width = image.width();
        let u = ((i % width) as f32 + 0.5f32) / (width as f32);
        let v = ((i / width) as f32 + 0.5f32) / (image.height() as f32);
        if image.sample(nearest, u, v) > 0.5f32 {
            flags[i] = 1u32;
        } else {
            flags[i] = 0u32;
        }
    }
}
