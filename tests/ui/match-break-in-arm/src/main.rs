// A `break` in an arm of an enum `match` would end the Slang `switch`, not the loop around it, GUST-E0417. expected.txt
// pins the code and the line of the `break`.

#[gust::gpu]
mod block {
    #[derive(PartialEq, Eq)]
    pub enum Level {
        Low,
        High,
    }

    fn pick(v: u32) -> Level {
        if v == 0u32 {
            return Level::Low;
        }
        Level::High
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        for k in 0u32..4u32 {
            match pick(input[id.x] + k) {
                Level::Low => {
                    break;
                }
                Level::High => {
                    out[id.x] = k;
                }
            }
        }
    }
}

fn main() {}
