// A pixel pin on a kernel with an atomic operation (GUST-E0723).
// expected.txt pins the code, the line of the pin and the reason.

#[gust::gpu]
mod shade {
    #[kernel(shape = pixel)]
    pub fn paint(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<u32>, mut out: RWStructuredBuffer<f32>) {
        atomic_add(&mut counter[0], 1u32);
        out[id.x] = 1.0f32;
    }
}

fn main() {}
