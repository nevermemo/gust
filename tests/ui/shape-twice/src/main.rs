// The shape option given twice (GUST-E0722).
// expected.txt pins the code and the line of the pin.

#[gust::gpu]
mod shade {
    #[kernel(shape = grid, shape = pixel)]
    pub fn paint(id: SV_DispatchThreadID, input: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = input[id.x];
    }
}

fn main() {}
