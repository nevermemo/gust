// A pixel pin on a kernel that writes its output on one path only (GUST-E0726).
// expected.txt pins the code and the line of the pin.

#[gust::gpu]
mod shade {
    #[kernel(shape = pixel)]
    pub fn paint(id: SV_DispatchThreadID, input: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>) {
        if input[id.x] > 0.0f32 {
            out[id.x] = 1.0f32;
        }
    }
}

fn main() {}
