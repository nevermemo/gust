// A pixel pin on a kernel that reads its output (GUST-E0728).
// expected.txt pins the code and the line of the pin.

#[gust::gpu]
mod shade {
    #[kernel(shape = pixel)]
    pub fn paint(id: SV_DispatchThreadID, input: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = out[id.x] + input[id.x];
    }
}

fn main() {}
