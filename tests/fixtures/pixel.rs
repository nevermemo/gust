#[gust::gpu]
mod pixel {
    // The `pixel` shape (slice mx.p1a, decisions D61, D63, D120). A `pixel` kernel is a fragment shader: the host
    // draws one fragment per element and each fragment writes its own element of every output, which stays a storage
    // buffer (`RWStructuredBuffer`) bound like a `grid` output. The fragment entry returns nothing; the pass carries one
    // never-written 8-bit attachment with a write mask of 0, so no render target holds a value. The element index is
    // the fragment's `y * 8192 + x`, and the host scissors the draw to the live pixels, so there is no padding pixel
    // and no guard. The invocation id is a local `uint3(index, 0, 0)`; only `id.x` is meaningful.
    //
    // What the shape takes (the classifier of slice mx.c decides, `#[kernel(shape = pixel)]` is checked against it):
    // the global id and no other built-in, no workgroup size, memory, barrier or atomic, at least one output, every
    // output a `u32`, `i32`, `f32` or a 2- or 4-wide vector of them, written once at `out[id.x]` on every path, no
    // `return`. Sampled textures and samplers are allowed.

    // Two outputs of different element types and one input, the whole contract: both outputs are written on both
    // arms of the `if`, so every pixel carries a value.
    #[kernel(shape = pixel)]
    pub fn shade(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<f32>,
        mut scaled: RWStructuredBuffer<f32>,
        mut flags: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        let v = input[i];
        scaled[i] = v * 2.0f32 + 1.0f32;
        if v > 0.5f32 {
            flags[i] = 1u32;
        } else {
            flags[i] = 0u32;
        }
    }

    // Vector and signed outputs: a 2-wide, a 4-wide and an `i32` output, from two inputs.
    #[kernel(shape = pixel)]
    pub fn pack(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<Vec2>,
        b: StructuredBuffer<i32>,
        mut sums: RWStructuredBuffer<Vec2>,
        mut wide: RWStructuredBuffer<Vec4>,
        mut steps: RWStructuredBuffer<i32>,
    ) {
        let v = a[id.x];
        sums[id.x] = v + Vec2::splat(1.0f32);
        wide[id.x] = Vec4::new(v.x, v.y, 0.0f32, 1.0f32);
        steps[id.x] = b[id.x] + 1;
    }

    // A sampled texture and a sampler in a fragment: one bilinear sample at the pixel's own column of a 16-column strip,
    // plus the texture's width (its size is read in the fragment like a buffer's length).
    #[kernel(shape = pixel)]
    pub fn row(
        id: SV_DispatchThreadID,
        image: Texture2D<Vec4>,
        bilinear: Sampler,
        mut colours: RWStructuredBuffer<Vec4>,
    ) {
        let u = (id.x as f32 + 0.5f32) * 0.0625f32;
        colours[id.x] = image.sample(bilinear, u, 0.5f32) + Vec4::splat(image.width() as f32);
    }
}
