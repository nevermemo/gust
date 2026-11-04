// A pixel pin on a kernel whose output element is a struct, which has no colour-target format (GUST-E0723).
// expected.txt pins the code, the line of the pin and the reason.

#[gust::gpu]
mod shade {
    pub struct Pair {
        pub x: f32,
        pub y: f32,
    }

    #[kernel(shape = pixel)]
    pub fn paint(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<Pair>) {
        out[id.x] = Pair { x: 1.0f32, y: 2.0f32 };
    }
}

fn main() {}
