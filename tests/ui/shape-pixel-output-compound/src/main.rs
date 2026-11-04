// A pixel pin on a kernel that updates its output with a compound assignment (GUST-E0728).
// expected.txt pins the code, the line of the pin and the reason.

#[gust::gpu]
mod shade {
    #[kernel(shape = pixel)]
    pub fn paint(id: SV_DispatchThreadID, input: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] += input[id.x];
    }
}

fn main() {}
