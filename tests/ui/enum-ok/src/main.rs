// Control case for enums (m9.e1): a field-less enum with an explicit discriminant, a signed one, a helper that takes and
// returns an enum, `==`, `as u32` and a statement `match` with a grouped arm and `_`. This crate must compile.

#[gust::gpu]
mod block {
    #[derive(PartialEq, Eq)]
    pub enum Mode {
        Off,
        Low = 3,
        Mid,
        High,
    }

    #[derive(PartialEq, Eq)]
    #[repr(i32)]
    pub enum Sign {
        Neg = -1,
        Pos = 1,
    }

    fn mode_from(v: u32) -> Mode {
        if v == 0u32 {
            return Mode::Off;
        }
        if v == 1u32 {
            return Mode::Low;
        }
        if v == 2u32 {
            return Mode::Mid;
        }
        Mode::High
    }

    fn sign_of(m: Mode) -> Sign {
        if m == Mode::Off {
            return Sign::Neg;
        }
        Sign::Pos
    }

    fn weight(m: Mode) -> u32 {
        let mut w = 1u32;
        match m {
            Mode::Off => {
                w += 1u32;
            }
            Mode::Low | Mode::Mid => {
                w += 2u32;
            }
            _ => {
                w += 3u32;
            }
        }
        w
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<u32>, mut out: RWStructuredBuffer<u32>) {
        if id.x < input.len() {
            let m = mode_from(input[id.x]);
            let s = sign_of(m);
            out[id.x] = weight(m) + (m as u32) + ((s as i32) + 1i32) as u32;
        }
    }
}

fn main() {}
