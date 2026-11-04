// A pixel pin on a kernel whose invocation parameter is mut (GUST-E0724).
// expected.txt pins the code and the line of the pin.

#[gust::gpu]
mod shade {
    #[kernel(shape = pixel)]
    pub fn paint(mut id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = 1.0f32;
    }
}

fn main() {}
