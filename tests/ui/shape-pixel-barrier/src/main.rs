// A pixel pin on a kernel with a workgroup barrier and no workgroup memory (GUST-E0723).
// expected.txt pins the code, the line of the pin and the reason.

#[gust::gpu]
mod shade {
    #[kernel(shape = pixel, cpu = "unsupported")]
    pub fn paint(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = 1.0f32;
        workgroup_barrier();
    }
}

fn main() {}
