// A stored array has a length of the GpuPod set (1 to 32, then the powers of two from 64 to 4096), GUST-E0216.

#[gust::gpu]
mod block {
    pub struct Row {
        pub cells: [u32; 33],
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut rows: RWStructuredBuffer<Row>) {
        rows[id.x].cells[0] = id.x;
    }
}

fn main() {}
