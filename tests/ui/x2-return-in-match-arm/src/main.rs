// X2 (m9.e1): a `return` in an arm of an enum `match` is legal in a helper (`pick`) and stays GUST-E0415 in a kernel
// with workgroup memory or a barrier. expected.txt pins the code and the line of the `return` in the arm.

#[gust::gpu]
mod block {
    #[derive(PartialEq, Eq)]
    pub enum Mode {
        Skip,
        Run,
    }

    fn pick(v: uint) -> Mode {
        if v == 0u32 {
            return Mode::Skip;
        }
        Mode::Run
    }

    #[kernel(workgroup_size(64, 1, 1), cpu = "unsupported")]
    pub fn reduce(
        id: SV_DispatchThreadID,
        lid: LocalId,
        input: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
        mut scratch: Workgroup<[uint; 64]>,
    ) {
        scratch[lid.x] = 0u32;
        match pick(input[id.x]) {
            Mode::Skip => {
                return;
            }
            Mode::Run => {
                scratch[lid.x] = 1u32;
            }
        }
        workgroup_barrier();
        out[id.x] = scratch[0u32];
    }
}

fn main() {}
