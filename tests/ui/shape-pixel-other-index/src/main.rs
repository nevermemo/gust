// A pixel pin on a kernel that writes at an index other than its own (GUST-E0725).
// expected.txt pins the code and the line of the pin.

#[gust::gpu]
mod shade {
    #[kernel(shape = pixel)]
    pub fn paint(id: SV_DispatchThreadID, input: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>) {
        out[id.x + 1u32] = input[id.x];
    }
}

fn main() {}
