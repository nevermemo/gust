// An array is a value of the invocation and is not a struct field, GUST-E0813 (as on main).

#[gust::gpu]
mod block {
    pub struct Row {
        pub cells: [u32; 4],
    }


    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = id.x;
    }
}

fn main() {}
