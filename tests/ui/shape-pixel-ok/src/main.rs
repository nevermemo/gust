// The accepted form of the pixel pin (slice mx.p1a): a kernel that writes every output once at its own index, and
// one whose write sits behind the one guard a live pixel always passes. This crate must compile.

#[gust::gpu]
mod shade {
    #[kernel(shape = pixel)]
    pub fn paint(id: SV_DispatchThreadID, input: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = input[id.x];
    }

    #[kernel(shape = pixel)]
    pub fn guarded(id: SV_DispatchThreadID, input: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>) {
        let i = id.x;
        if i < out.len() {
            out[i] = input[i] * 2.0f32;
        }
    }
}

fn main() {}
