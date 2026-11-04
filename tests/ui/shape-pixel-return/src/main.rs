// A pixel pin on a kernel with a return (GUST-E0727).
// expected.txt pins the code and the line of the pin.

#[gust::gpu]
mod shade {
    #[kernel(shape = pixel)]
    pub fn paint(id: SV_DispatchThreadID, input: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>) {
        if id.x >= out.len() {
            return;
        }
        out[id.x] = input[id.x];
    }
}

fn main() {}
