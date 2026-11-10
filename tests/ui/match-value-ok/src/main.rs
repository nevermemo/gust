// Control case for the value form of `match` (m9.e2): a typed `let` on an enum and on `u32` and `i32` scrutinees with
// integer-literal patterns, and the statement form with literals. This crate must compile.

#[gust::gpu]
mod block {
    #[derive(PartialEq, Eq)]
    pub enum Mode {
        Off,
        Low = 3,
    }

    fn mode_from(v: u32) -> Mode {
        if v == 0u32 {
            return Mode::Off;
        }
        Mode::Low
    }

    fn sgn(s: i32) -> i32 {
        let r: i32 = match s {
            -1 => -10i32,
            0 => 0i32,
            1 | 2 => 10i32,
            _ => 99i32,
        };
        r + 1i32
    }

    fn weight(m: Mode) -> u32 {
        let w: u32 = match m {
            Mode::Off => 1u32,
            Mode::Low => 5u32,
        };
        w + 1u32
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        let v = input[id.x];
        let b: u32 = match v {
            0 => 100u32,
            1 | 2 => 200u32,
            3000000000 => 300u32,
            _ => 400u32,
        };
        match v {
            7 => {
                out[id.x] = b;
            }
            _ => {
                out[id.x] = b + weight(mode_from(v)) + (sgn((v as i32) - 1i32) as u32);
            }
        }
    }
}

fn main() {}
