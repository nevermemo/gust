// A pixel pin on a kernel with workgroup memory and a barrier (GUST-E0723).
// expected.txt pins the code and the line of the pin.

#[gust::gpu]
mod shade {
    #[kernel(shape = pixel, cpu = "unsupported")]
    pub fn paint(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>, mut scratch: Workgroup<[f32; 64]>) {
        scratch[0u32] = 1.0f32;
        workgroup_barrier();
        out[id.x] = scratch[0u32];
    }
}

fn main() {}
