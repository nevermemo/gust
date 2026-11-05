#[gust::gpu]
mod textures {
    // Textures and samplers (slice mx.t3, D62, D64). A texture is a kernel parameter and a resource: it is bound with
    // a view, and its texel type or format is part of its type. The methods are `load(x, y)` (a texel by integer
    // address), `store(x, y, v)`, `sample(sampler, u, v)` (level 0, `u` and `v` in 0..1) and `width()` / `height()`.
    //
    //   Texture2D<T>           sampled, read-only: `load`, `sample` (`sample` needs f32, Vec2 or Vec4 texels)
    //   StorageTexture2D<F>    read and write: `load`, `store`; only R32Float, R32Uint and R32Sint
    //   WriteTexture2D<F>      write only: `store`; all nine formats
    //   Sampler                the filter state `sample` reads; it has no methods
    //
    // No kernel here is pinned to a shape. The texel address is the invocation's own id, and a dispatch that is a whole
    // number of workgroups may be larger than the texture, so every kernel guards against the texture's size itself.

    // A cross blur of a sampled texture into a write-only one: four bilinear samples around the texel centre.
    #[kernel(workgroup_size(8, 8, 1))]
    pub fn blur(
        id: SV_DispatchThreadID,
        image: Texture2D<Vec4>,
        bilinear: Sampler,
        mut blurred: WriteTexture2D<Rgba32Float>,
    ) {
        let width = blurred.width();
        let height = blurred.height();
        if id.x < width && id.y < height {
            let u = (id.x as f32 + 0.5f32) / (width as f32);
            let v = (id.y as f32 + 0.5f32) / (height as f32);
            let du = 1.0f32 / (width as f32);
            let dv = 1.0f32 / (height as f32);
            let left = image.sample(bilinear, u - du, v);
            let right = image.sample(bilinear, u + du, v);
            let above = image.sample(bilinear, u, v - dv);
            let below = image.sample(bilinear, u, v + dv);
            blurred.store(id.x, id.y, (left + right + above + below) * 0.25f32);
        }
    }

    // A storage texture read and written in place; the size is read from the texture itself.
    #[kernel(workgroup_size(8, 8, 1))]
    pub fn invert(id: SV_DispatchThreadID, mut image: StorageTexture2D<R32Float>) {
        // `(image).width()` is the same call as `image.width()`: the receiver is the parameter either way.
        if id.x < (image).width() && id.y < image.height() {
            let value = image.load(id.x, id.y);
            image.store(id.x, id.y, 1.0f32 - value);
        }
    }

    // Integer texels: a sampled `Texture2D<u32>` is read with `load` (an integer texture is never filtered), and a
    // `StorageTexture2D<R32Uint>` accumulates it.
    #[kernel(workgroup_size(8, 8, 1))]
    pub fn accumulate(
        id: SV_DispatchThreadID,
        ids: Texture2D<u32>,
        mut totals: StorageTexture2D<R32Uint>,
    ) {
        if id.x < totals.width() && id.y < totals.height() {
            let total = totals.load(id.x, id.y) + ids.load(id.x, id.y);
            totals.store(id.x, id.y, total);
        }
    }

    // The signed storage format, with a texel address that is an expression. The neighbour is read from a separate
    // sampled texture and the result is stored in another, so no invocation reads a texel that another writes (the
    // race-free precondition of docs/SEMANTICS.md section 5): `field[x] = source[x + 1] - 1`.
    #[kernel(workgroup_size(8, 8, 1))]
    pub fn shift(
        id: SV_DispatchThreadID,
        source: Texture2D<i32>,
        mut field: StorageTexture2D<R32Sint>,
    ) {
        if id.x + 1u32 < field.width() && id.y < field.height() {
            let next = source.load(id.x + 1u32, id.y);
            field.store(id.x, id.y, next - 1);
        }
    }

    // Every texel type a sampled texture has: the float ones are sampled, the integer ones loaded. Two samplers, one
    // per filter state, and a write-only target.
    #[allow(clippy::too_many_arguments)]
    #[kernel(workgroup_size(8, 8, 1))]
    pub fn gather(
        id: SV_DispatchThreadID,
        scalar: Texture2D<f32>,
        pair: Texture2D<Vec2>,
        counts: Texture2D<UVec4>,
        offsets: Texture2D<IVec2>,
        nearest: Sampler,
        bilinear: Sampler,
        mut gathered: WriteTexture2D<Rgba32Float>,
    ) {
        let width = gathered.width();
        let height = gathered.height();
        if id.x < width && id.y < height {
            let u = (id.x as f32 + 0.5f32) / (width as f32);
            let v = (id.y as f32 + 0.5f32) / (height as f32);
            let a = scalar.sample(nearest, u, v);
            let b = pair.sample(bilinear, u, v);
            let c = counts.load(id.x, id.y);
            let d = offsets.load(id.x, id.y);
            gathered.store(
                id.x,
                id.y,
                Vec4::new(a + b.x, b.y, c.x as f32 + c.w as f32, d.x as f32 - d.y as f32),
            );
        }
    }

    // All nine write-only formats, three to a kernel.
    #[kernel(workgroup_size(8, 8, 1))]
    pub fn write_floats(
        id: SV_DispatchThreadID,
        mut one: WriteTexture2D<R32Float>,
        mut two: WriteTexture2D<Rg32Float>,
        mut four: WriteTexture2D<Rgba32Float>,
    ) {
        if id.x < one.width() && id.y < one.height() {
            one.store(id.x, id.y, 1.0f32);
            two.store(id.x, id.y, Vec2::new(1.0f32, 2.0f32));
            four.store(id.x, id.y, Vec4::new(1.0f32, 2.0f32, 3.0f32, 4.0f32));
        }
    }

    #[kernel(workgroup_size(8, 8, 1))]
    pub fn write_uints(
        id: SV_DispatchThreadID,
        mut one: WriteTexture2D<R32Uint>,
        mut two: WriteTexture2D<Rg32Uint>,
        mut four: WriteTexture2D<Rgba32Uint>,
    ) {
        if id.x < one.width() && id.y < one.height() {
            one.store(id.x, id.y, id.x);
            two.store(id.x, id.y, UVec2::new(id.x, id.y));
            four.store(id.x, id.y, UVec4::new(id.x, id.y, 3u32, 4u32));
        }
    }

    #[kernel(workgroup_size(8, 8, 1))]
    pub fn write_ints(
        id: SV_DispatchThreadID,
        mut one: WriteTexture2D<R32Sint>,
        mut two: WriteTexture2D<Rg32Sint>,
        mut four: WriteTexture2D<Rgba32Sint>,
    ) {
        if id.x < one.width() && id.y < one.height() {
            one.store(id.x, id.y, -1);
            two.store(id.x, id.y, IVec2::new(-1, 2));
            four.store(id.x, id.y, IVec4::new(-1, 2, -3, 4));
        }
    }
}
