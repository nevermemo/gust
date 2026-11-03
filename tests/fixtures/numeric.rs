#[gust::gpu]
mod numeric {
    pub struct Pair {
        pub first: uint,
        pub second: uint,
    }

    pub struct Wrapped {
        pub hash: uint,
        pub step: int,
    }

    pub struct Typed {
        pub big: uint,
        pub small: uint,
        pub above: uint,
        pub shifted: uint,
        pub negative: int,
        pub arithmetic: int,
        pub quotient: int,
    }

    // Wrapping arithmetic (slice m7.w): a multiplicative hash and a linear congruential step, defined modulo 2^32 for
    // every input. Plain `*` and `+` would be outside the domain (KM-10) wherever the product does not fit.
    fn scramble(x: uint) -> uint {
        x.wrapping_mul(2654435761u32).wrapping_add(2654435769u32)
    }

    fn step(x: int) -> int {
        x.wrapping_neg().wrapping_mul(1103515245i32).wrapping_sub(12345i32)
    }

    // `+=` is not in the supported subset (only plain `=` lowers), so the
    // `x = x + y` form is required; the fixture intentionally drives several
    // buffer types, hence more than seven parameters.
    #[allow(clippy::too_many_arguments, clippy::assign_op_pattern)]
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn run(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<int>,
        b: StructuredBuffer<uint>,
        f: StructuredBuffer<float>,
        mut signed_out: RWStructuredBuffer<int>,
        mut unsigned_out: RWStructuredBuffer<uint>,
        mut cast_out: RWStructuredBuffer<int>,
        mut pair_out: RWStructuredBuffer<Pair>,
        mut wrap_out: RWStructuredBuffer<Wrapped>,
    ) {
        let i: uint = id.x;
        if i < signed_out.len() {
            // Signed division truncates toward zero; signed modulo keeps the
            // dividend's sign. Both match Rust for non-zero divisors.
            signed_out[i] = a[i] / 2 * 100 + a[i] % 3;
            // Unsigned division and modulo.
            unsigned_out[i] = b[i] / 2 * 100 + b[i] % 3;
            // In-range float -> int cast truncates toward zero.
            cast_out[i] = f[i] as int;
            // Field-aware struct construction: a let binding (fields written in
            // source order, identity by name), a reassignment, and a
            // buffer-element assignment.
            let mut p = Pair { second: b[i] + 1u32, first: b[i] + 2u32 };
            p.first = p.first + 1u32;
            p.second = p.second + 1u32;
            pair_out[i] = Pair { first: p.first, second: p.second };
            // Wrapping methods through helpers: the same value on the GPU, the twin and Rust's own `wrapping_*`.
            wrap_out[i] = Wrapped { hash: scramble(b[i]), step: step(a[i]) };
        }
    }

    // Typed locals at the edge the untyped path got wrong (slice m6.z2, D86). An unannotated `let mut d = 7;` would
    // be Slang `int`, and `d /= y` with `y >= 2^31` divides signed there and unsigned in Rust; GUST-E0326 now refuses
    // that `let`, so every local here carries a suffix or an annotation. The `uint` fields are the unsigned division
    // of a large and of a small dividend, an unsigned comparison and a logical shift right of a value with the top bit
    // set; the `int` fields are a signed comparison, an arithmetic shift right and a signed division. Every case is
    // inside the defined domain (no zero divisor, no overflow, shifts below 32).
    #[allow(clippy::redundant_field_names)]
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn typed_locals(
        id: SV_DispatchThreadID,
        y: StructuredBuffer<uint>,
        s: StructuredBuffer<int>,
        mut out: RWStructuredBuffer<Typed>,
    ) {
        let i: uint = id.x;
        if i < y.len() && i < s.len() && i < out.len() {
            let mut big = 4_000_000_000u32;
            big /= y[i];
            let mut small: u32 = 7;
            small /= y[i];
            let word: u32 = y[i];
            let limit: u32 = 5;
            let mut above: u32 = 0;
            if word > limit {
                above = 1;
            }
            let shifted: u32 = word >> 4;
            let value: i32 = s[i];
            let mut negative: i32 = 0;
            if value < 0 {
                negative = 1;
            }
            let mut quotient = -100i32;
            quotient /= value;
            out[i] = Typed {
                big: big,
                small: small,
                above: above,
                shifted: shifted,
                negative: negative,
                arithmetic: value >> 3,
                quotient: quotient,
            };
        }
    }
}
