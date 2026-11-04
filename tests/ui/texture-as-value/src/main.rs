// A texture used as a value instead of as the receiver of a method (GUST-E0736).
// expected.txt pins the code, the line of the offending node and the message.

#[gust::gpu]
mod shade {
    #[kernel]
    pub fn paint(id: SV_DispatchThreadID, image: Texture2D<Vec4>, mut out: WriteTexture2D<Rgba32Float>) {
        let alias = image;
        out.store(id.x, id.y, Vec4::splat(1.0f32));
    }
}

fn main() {}
