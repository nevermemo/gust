// cpu is a target, not a shape (GUST-E0721): the hint names GUST_PROFILE=cpu.
// expected.txt pins the code and the line of the pin.

#[gust::gpu]
mod shade {
    #[kernel(shape = cpu)]
    pub fn paint(id: SV_DispatchThreadID, input: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = input[id.x];
    }
}

fn main() {}
