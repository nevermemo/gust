// A buffer of tuples is GUST-E0212: a tuple is a value of the body, not an element of a resource. expected.txt pins the
// code and the line of the element type.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<(u32, u32)>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = input[id.x].0;
    }
}

fn main() {}
