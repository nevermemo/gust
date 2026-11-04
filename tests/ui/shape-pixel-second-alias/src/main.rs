// A pixel pin on a kernel with a second alias of the own index (GUST-E0725).
// expected.txt pins the code, the line of the pin and the reason.

#[gust::gpu]
mod shade {
    #[kernel(shape = pixel)]
    pub fn paint(id: SV_DispatchThreadID, input: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>) {
        let i = id.x;
        let j = id.x;
        out[i] = input[j];
    }
}

fn main() {}
