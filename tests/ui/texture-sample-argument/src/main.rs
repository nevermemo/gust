// A sample whose first argument is not a sampler parameter by name (GUST-E0733). The expression is a valid Rust
// `Sampler`, so rustc reports nothing of its own, but the shader needs the parameter.
// expected.txt pins the code, the line of the offending node and the message.

#[gust::gpu]
mod shade {
    #[kernel]
    pub fn paint(id: SV_DispatchThreadID, image: Texture2D<Vec4>, filter: Sampler, mut out: WriteTexture2D<Rgba32Float>) {
        let value = image.sample(*&filter, 0.5f32, 0.5f32);
        out.store(id.x, id.y, value);
    }
}

fn main() {}
