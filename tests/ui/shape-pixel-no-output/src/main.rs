// A pixel pin on a kernel that writes no output buffer (GUST-E0723).
// expected.txt pins the code, the line of the pin and the reason.

#[gust::gpu]
mod shade {
    #[kernel(shape = pixel)]
    pub fn paint(id: SV_DispatchThreadID, input: StructuredBuffer<f32>) {
        let x = input[id.x] + 1.0f32;
    }
}

fn main() {}
