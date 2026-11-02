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
}
