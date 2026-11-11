// The same length rule for the element of a buffer, GUST-E0216.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut cells: RWStructuredBuffer<[u32; 100]>) {
        cells[id.x][0] = id.x;
    }
}

fn main() {}
