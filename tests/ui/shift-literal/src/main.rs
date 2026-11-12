// A constant shift amount of 32 or more is GUST-E0335, as rustc's deny-by-default `arithmetic_overflow` refuses it
// (a run-time amount is accepted and masks on the GPU, see shift-ok). expected.txt pins the code, the line and the
// amount named in the message.

#[gust::gpu]
mod block {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        out[id.x] = id.x << 34u32;
    }
}

fn main() {}
