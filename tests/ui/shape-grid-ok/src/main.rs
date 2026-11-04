// The accepted forms of the shape pin: grid on a kernel that could also be a pixel, grid next to a workgroup size,
// and grid on a kernel that no pixel could be. This crate must compile.

#[gust::gpu]
mod shade {
    #[kernel(shape = grid)]
    pub fn plain(id: SV_DispatchThreadID, input: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>) {
        out[id.x] = input[id.x];
    }

    #[kernel(shape = grid, workgroup_size(64, 1, 1))]
    pub fn sized(id: SV_DispatchThreadID, input: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>) {
        if id.x < out.len() {
            out[id.x] = input[id.x];
        }
    }

    #[kernel(workgroup_size(8, 1, 1), shape = grid)]
    pub fn guarded(id: SV_DispatchThreadID, input: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>) {
        if input[id.x] > 0.0f32 {
            out[id.x] = 1.0f32;
        }
    }
}

fn main() {}
