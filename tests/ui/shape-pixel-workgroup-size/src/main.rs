// A pixel pin on a kernel with a workgroup size (GUST-E0723): the error is at the pin and names the size.
// expected.txt pins the code and the line of the pin.

#[gust::gpu]
mod shade {
    #[kernel(shape = pixel, workgroup_size(64, 1, 1))]
    pub fn paint(id: SV_DispatchThreadID, input: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = input[id.x];
    }
}

fn main() {}
