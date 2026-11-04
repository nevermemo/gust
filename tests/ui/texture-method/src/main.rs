// A method no texture has (GUST-E0735). `Texture2D` is `Clone`, so rustc accepts the call and only the macro refuses it.
// expected.txt pins the code, the line of the offending node and the message.

#[gust::gpu]
mod shade {
    #[kernel]
    pub fn paint(id: SV_DispatchThreadID, image: Texture2D<Vec4>, mut out: WriteTexture2D<Rgba32Float>) {
        let copy = image.clone();
        out.store(id.x, id.y, Vec4::splat(1.0f32));
    }
}

fn main() {}
