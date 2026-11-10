// A buffer is no element of a tuple, GUST-E0212: a helper takes values only (D154), and a tuple would carry the
// buffer in. expected.txt pins the code and the line of the element type.

#[gust::gpu]
mod block {
    fn read(t: (StructuredBuffer<u32>, u32)) -> u32 {
        t.0[t.1]
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = read((input, id.x));
    }
}

fn main() {}
