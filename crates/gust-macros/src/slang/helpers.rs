//! The text of the helper functions a kernel defines when it uses them, and the names that call them.

use std::fmt::Write as _;

/// The vector types `__gust_dot` has an overload for: Slang spelling, result type, component count.
const DOT_OVERLOADS: &[(&str, &str, usize)] = &[
    ("float2", "float", 2),
    ("float3", "float", 3),
    ("float4", "float", 4),
    ("int2", "int", 2),
    ("int4", "int", 4),
    ("uint2", "uint", 2),
    ("uint3", "uint", 3),
    ("uint4", "uint", 4),
];

/// The overloads of `__gust_dot`: Slang's `dot` for the float vectors, and the sum of the component products, grouped
/// from the left like the twin's `x * x + y * y + ..`, for the integer ones. Slang's `dot` on an integer vector is
/// `OpSDot` or `OpUDot` in SPIR-V, which needs `Capability DotProduct` (naga's SPIR-V reader refuses it); the sum
/// is plain multiplications and additions, which pass naga's reader with empty capabilities (measured, slangc 2026.20).
pub(super) fn dot_overload_text() -> String {
    let mut text = String::new();
    for &(vector, result, width) in DOT_OVERLOADS {
        let body = if result == "float" {
            "dot(a, b)".to_owned()
        } else {
            let mut sum = "a.x * b.x".to_owned();
            for field in &["y", "z", "w"][..width - 1] {
                sum = format!("({sum} + a.{field} * b.{field})");
            }
            sum
        };
        writeln!(
            text,
            "{result} __gust_dot({vector} a, {vector} b) {{ return {body}; }}"
        )
        .unwrap();
    }
    text.push('\n');
    text
}

/// Rust's `fract` is `x - x.trunc()`; Slang's `frac` is `x - floor(x)`, which differs for a negative `x`. The helper
/// evaluates the receiver once.
pub(super) const FRACT_HELPER: &str = "__gust_fract";

/// `x.round()`: half-way cases away from zero, which WGSL and Slang `round` (half to even) does not do. `x - trunc(x)`
/// is exact, so the test `>= 0.5` is exact (`0.49999997` stays down), and `t +- 1` is exact below `2^23`, where the
/// test can hold. The sign comes from the sign bit, so `-0.5` rounds to `-1.0`. Every result is exact except the sign
/// of a zero result: it comes from `trunc`, and WGSL may not keep zero signs. Overloads are not needed (`f32` only).
pub(super) const ROUND_HELPER_TEXT: &str = "float __gust_round(float x) { float t = trunc(x); return (abs(x - t) >= 0.5) ? (t + (((asuint(x) >> 31u) != 0u) ? -1.0 : 1.0)) : t; }\n\n";

/// `x.signum()`: `f32` is `1.0` for `+0.0` and positives and `-1.0` for `-0.0` and negatives (the sign bit; Slang's
/// `sign` returns 0 for a zero), `i32` is a comparison chain (not `sign`). The macro sees no types, so the two are
/// overloads.
pub(super) const SIGNUM_HELPER_TEXT: &str = "float __gust_signum(float x) { return ((asuint(x) >> 31u) != 0u) ? -1.0 : 1.0; }\nint __gust_signum(int x) { return (x > 0) ? 1 : ((x < 0) ? -1 : 0); }\n\n";

/// `a.rem_euclid(b)`: Rust's `r = a % b; if r < 0 { r + b.abs() } else { r }`. The integer overloads sit on `__gust_rem`,
/// the helper that stands for `%` (zero divisors and the SPIR-V guards included); for `u32` it is `%` itself. The
/// float overload cannot use the float `%` of WGSL (inherited from `x - y * trunc(x / y)`): on the Radeon 890M it
/// returned a wrong whole divisor at exact multiples (slice m11.math2 measured 79 of 5000 rows, `7.0 % 7.0` among
/// them). `__gust_fmod` is `fmod` by binary long division instead: `x - s` for `s = y * 2^k <= x` is exact (the
/// result is below `s`), doubling and halving are exact, so `__gust_fmod` is exact for normal operands, as Rust's.
/// The doubling test is `s <= (r - s)`, not `(s * 2.0) <= r`, which overflows for `abs(a)` near `f32::MAX` (WGSL: an
/// overflowing intermediate is indeterminate). Equivalent and overflow-free: if `s <= r/2` the real `r - s >= s`, and
/// rounding to a neighbour cannot go below the representable `s`; if `r/2 < s <= r`, `r - s` is exact (Sterbenz) and
/// below `s`; if `s > r` it is negative. The final `r + abs(b)` of `__gust_rem_euclid` is a rounded addition: WGSL's
/// "correctly rounded" allows either neighbour, so the portable result is within 1 ULP of Rust's (observed exact on
/// the Radeon 890M).
/// The loops run at most 254 and 255 times, so a zero divisor or an infinity ends with some value, not a hang.
/// The halving loop is deliberately the flag form (`done`, no `break`): the same loop with `if (s == y) { break; }` was
/// miscompiled by NVIDIA driver 610.88 (Vulkan, RTX 5090) once it needed 128 or more iterations (observed on `f32::MAX %
/// 0.7` and five more rows of slice m11.fm; the cause is inferred, not measured): the `break` never fired and the result
/// was wrong. Do not simplify it back.
pub(super) const REM_EUCLID_HELPER_TEXT: &str = "float __gust_fmod(float a, float b) { float x = abs(a); float y = abs(b); float r = x; if (x >= y) { float s = y; for (int i = 0; i < 254 && s <= (r - s); i++) { s = s * 2.0; } bool done = false; for (int j = 0; j < 255; j++) { if (!done) { if (r >= s) { r = r - s; } if (s == y) { done = true; } else { s = s * 0.5; } } } } return ((asuint(a) >> 31u) != 0u) ? -r : r; }\nfloat __gust_rem_euclid(float a, float b) { float r = __gust_fmod(a, b); return (r < 0.0) ? (r + abs(b)) : r; }\nint __gust_rem_euclid(int a, int b) { int r = __gust_rem(a, b); return (r < 0) ? (r + abs(b)) : r; }\nuint __gust_rem_euclid(uint a, uint b) { return __gust_rem(a, b); }\n\n";

/// `x.hypot(y)` (slice m11.math3): Rust's `hypot` has no intermediate overflow, so the helper scales by the larger
/// magnitude: `m * sqrt(1 + (n / m)^2)` for `m = max(|x|, |y|)` and `n = min(|x|, |y|)`, and `0.0` when `m` is zero (the
/// quotient would be `0 / 0`). `length(float2(x, y))` is not used: WGSL inherits its accuracy from `sqrt(dot(v, v))`,
/// whose square overflows near `1e30`. The accuracy is the sum of the WGSL rules of the five operations, for operands
/// with `max(|x|, |y|)` in `[2^-126, 2^126]` and a finite result: the division is 2.5 ULP, the product `q * q` and the
/// sum `1 + q * q` are "correctly rounded" (either neighbour), `sqrt` is the 6.5 ULP the float table of SEMANTICS.md
/// states, and the final product is either neighbour again. That bounds the relative error by `11 * 2^-23`, so the
/// result is within 22 ULP of the exact value (spacing taken at the larger of the exact and the actual result); the
/// observed error is far smaller and is printed per adapter by `crates/gust-wgpu/tests/math3.rs`. It is not
/// bit-equal to Rust's `hypot` in general (D39: documented, never chased). Outside that domain the helper's own text
/// decides, even on an IEEE-exact GPU: `hypot(inf, inf)` has `q = inf / inf = NaN`, so it returns NaN where Rust gives
/// infinity. Inferred, not measured: with a `max` and `min` that ignore a NaN, `hypot(NaN, 1.0)` may be about 1.414; and
/// with flush-to-zero, two subnormal operands may reach `0 / 0`. No loop, so none of the loop-exit trouble of `__gust_fmod`.
pub(super) const HYPOT_HELPER_TEXT: &str = "float __gust_hypot(float x, float y) { float m = max(abs(x), abs(y)); float n = min(abs(x), abs(y)); if (m == 0.0) { return 0.0; } float q = n / m; return m * sqrt(1.0 + q * q); }\n\n";

/// `b.pow(e)` on `u32` and `i32` (slice m11.math3): square and multiply over the bits of the exponent, at most 32
/// trips, and the base is not squared after the last bit, so for a result that fits the type no intermediate overflows
/// either. The loop exit is an integer compare in the loop condition (`e != 0u`), not a float compare and not a
/// `break`: NVIDIA driver 610.88 miscompiled a float-compare `break` (see `REM_EUCLID_HELPER_TEXT`). Every result is
/// exact when it fits the type. A result that does not fit wraps modulo 2^32 on the GPU (WGSL section 6.2.3, as `*`
/// does) and panics in a debug build of the twin; `pow(0)` is `1` for every base, `0.pow(0)` included, as in Rust. The
/// overloads are selected by the base (the exponent is always `uint(e)`).
pub(super) const POW_HELPER_TEXT: &str = "uint __gust_pow(uint b, uint e) { uint r = 1u; for (int i = 0; i < 32 && e != 0u; i++) { if ((e & 1u) != 0u) { r = r * b; } e = e >> 1u; if (e != 0u) { b = b * b; } } return r; }\nint __gust_pow(int b, uint e) { int r = 1; for (int i = 0; i < 32 && e != 0u; i++) { if ((e & 1u) != 0u) { r = r * b; } e = e >> 1u; if (e != 0u) { b = b * b; } } return r; }\n\n";

/// The name of the identity helper that keeps a `<` comparison out of a WGSL template list.
///
/// Still needed, measured in slice m4.z (2026-11-02, `plans/m4/probe/m4z/keep`): 30 comparison-in-list shapes, each
/// compiled with and without the wrapper and the WGSL parsed and validated by naga 30.0.1. slangc 2026.20 puts
/// parentheses around every comparison argument itself (all 60 files validate, and the wrapper only adds the identity
/// function and its call), but slangc 2026.13.1 does not (13 of the 30 fail to parse without it). `gust::slang` runs
/// whatever `slangc` is first on `PATH` and checks no version; only `cargo gust` and `cargo xtask` enforce the 2026.22
/// pin. So the wrapper stays until the library enforces the pin. Removing it changes the WGSL of every kernel that
/// uses it, which also wants a read-back on the GPU.
pub(super) const KEEP_HELPER: &str = "__gust_keep";

pub(super) const SUBGROUP_AND: &str = "__gust_subgroup_and";
pub(super) const SUBGROUP_OR: &str = "__gust_subgroup_or";
pub(super) const SUBGROUP_XOR: &str = "__gust_subgroup_xor";
pub(super) const SUBGROUP_INCLUSIVE_ADD: &str = "__gust_subgroup_inclusive_add";
pub(super) const SUBGROUP_INCLUSIVE_MUL: &str = "__gust_subgroup_inclusive_mul";
pub(super) const SUBGROUP_COUNT: &str = "__gust_subgroup_count";

/// The overloaded Slang helpers the subgroup operations of a kernel call, in a fixed order, and only those it calls.
pub(super) fn subgroup_helper_text(used: &std::collections::BTreeSet<&'static str>) -> String {
    let mut text = String::new();
    for (operation, helper, intrinsic) in [
        ("subgroup_and", SUBGROUP_AND, "WaveActiveBitAnd"),
        ("subgroup_or", SUBGROUP_OR, "WaveActiveBitOr"),
        ("subgroup_xor", SUBGROUP_XOR, "WaveActiveBitXor"),
    ] {
        if used.contains(operation) {
            // The signed overload reduces the bits of the unsigned value: DX12's compiler has no signed bit
            // reduction (S3), and the result is the same bits.
            writeln!(
                text,
                "uint {helper}(uint value) {{ return {intrinsic}(value); }}"
            )
            .unwrap();
            writeln!(
                text,
                "int {helper}(int value) {{ return asint({intrinsic}(asuint(value))); }}"
            )
            .unwrap();
        }
    }
    for (operation, helper, scan, combine) in [
        (
            "subgroup_inclusive_add",
            SUBGROUP_INCLUSIVE_ADD,
            "WavePrefixSum",
            '+',
        ),
        (
            "subgroup_inclusive_mul",
            SUBGROUP_INCLUSIVE_MUL,
            "WavePrefixProduct",
            '*',
        ),
    ] {
        if used.contains(operation) {
            // The exclusive scan combined with the operand, which the helper names once.
            for ty in ["uint", "int", "float"] {
                writeln!(
                    text,
                    "{ty} {helper}({ty} value) {{ return {scan}(value) {combine} value; }}"
                )
                .unwrap();
            }
        }
    }
    if used.contains("subgroup_count") {
        writeln!(
            text,
            "uint {SUBGROUP_COUNT}(bool value) {{ uint4 ballot = WaveActiveBallot(value); return countbits(ballot.x) + countbits(ballot.y) + countbits(ballot.z) + countbits(ballot.w); }}"
        )
        .unwrap();
    }
    if !text.is_empty() {
        text.push('\n');
    }
    text
}
