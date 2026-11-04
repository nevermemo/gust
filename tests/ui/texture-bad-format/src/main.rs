// A sampled texture of a 3-wide texel, which has no texture format (GUST-E0731).
// expected.txt pins the code, the line of the offending node and the message.

#[gust::gpu]
mod shade {
    #[kernel]
    pub fn paint(id: SV_DispatchThreadID, image: Texture2D<Vec3>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = image.width();
    }
}

fn main() {}
