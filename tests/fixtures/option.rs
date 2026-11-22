#[gust::gpu_module]
mod optional {
    pub struct Pair {
        pub first: uint,
        pub second: uint,
    }

    // Even inputs yield half the value; odd inputs yield nothing.
    fn half_if_even(x: uint) -> Option<uint> {
        if x & 1u32 == 0u32 {
            Some(x / 2u32)
        } else {
            None
        }
    }

    // `Some(Pair { .. })` is a struct literal in expression position, which stays
    // rejected; bind the struct first.
    fn pair_if_small(x: uint) -> Option<Pair> {
        let p = Pair { first: x, second: x + 1u32 };
        if x < 8u32 { Some(p) } else { None }
    }

    fn score(value: Option<uint>, fallback: uint) -> uint {
        value.unwrap_or(fallback) * 10u32
    }

    // `explicit` deliberately exercises `unwrap_or` on a typed `Some` local.
    #[allow(clippy::assign_op_pattern, clippy::unnecessary_literal_unwrap)]
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut flags: RWStructuredBuffer<uint>,
        mut values: RWStructuredBuffer<uint>,
        mut pairs: RWStructuredBuffer<Pair>,
    ) {
        let i = id.x;
        if i < input.len() {
            let x = input[i];
            let half = half_if_even(x);
            let mut flag = 0u32;
            if half.is_some() {
                flag = flag + 1u32;
            }
            if half.is_none() {
                flag = flag + 2u32;
            }
            let mut missing: Option<uint> = None;
            if x > 1000u32 {
                missing = Some(x);
            }
            flags[i] = flag + score(missing, 3u32);
            let explicit: Option<uint> = Some(x + 100u32);
            let mut total = half.unwrap_or(7u32) + explicit.unwrap_or(0u32);
            if let Some(h) = half {
                total = total + h;
            }
            values[i] = total;
            let mut out_pair = Pair { first: 0u32, second: 0u32 };
            if let Some(p) = pair_if_small(x) {
                out_pair = p;
            } else {
                out_pair.first = 99u32;
            }
            pairs[i] = out_pair;
        }
    }
}
