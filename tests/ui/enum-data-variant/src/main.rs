// A variant of a GPU enum carries no data: a data-carrying variant has no Slang lowering yet, GUST-E0122. expected.txt
// pins the code and the line of the variant.

#[gust::gpu]
mod block {
    #[derive(PartialEq, Eq)]
    pub enum Shape {
        Dot,
        Circle(u32),
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = input[id.x];
    }
}

fn main() {}
