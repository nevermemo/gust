// An arm of an enum `match` takes no `if` guard: the Slang `switch` has none, GUST-E0418. expected.txt pins the code and
// the line of the guard.

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
        match pick(input[id.x]) {
            Level::Low if id.x > 3u32 => {
                out[id.x] = 1u32;
            }
            _ => {
                out[id.x] = 2u32;
            }
        }
    }
}

fn main() {}
