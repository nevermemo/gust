// A pixel pin on a kernel with a storage texture (GUST-E0723).
// expected.txt pins the code, the line of the offending node and the message.

#[gust::gpu]
mod shade {
    #[kernel(shape = pixel)]
    pub fn paint(id: SV_DispatchThreadID, mut image: StorageTexture2D<R32Float>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = image.load(id.x, id.y);
    }
}

fn main() {}
