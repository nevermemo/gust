//! The division table behind `/`, `%`, `/=` and `%=` (slice m6.dv, D74's cure; spike m6.s1).
//!
//! slangc 2026.20 folds integer `/` and `%` and refuses a divisor that folds to zero (E30002), even in code that no
//! invocation runs (a dead branch, `7 / (i * 0)`, a wrapping `/=`). The same operation behind a call compiles in every
//! shape the spike probed, on WGSL and on SPIR-V. So every `/` and `%` whose divisor is not a literal lowers to the
//! overloaded call `__gust_div(a, b)` or `__gust_rem(a, b)`, and `/=` and `%=` to `__gust_div_assign(place, b)` and
//! `__gust_rem_assign(place, b)` with the place `inout` (as `__gust_xor_assign`), so the place is evaluated once. The
//! macro sees syntax and not types, so the float and vector operands take the overloads too.
//!
//! The body is the plain operator, with no `[noinline]`: slangc inlines the helper in SPIR-V (the same `OpSDiv`,
//! `OpUDiv`, `OpSRem`, `OpUMod`; instruction counts equal or within a few) and WGSL keeps the call. The GPU and twin
//! semantics do not change, and the cost is noise (spike m6.s1: a median of 20 dispatches over 1M elements, within
//! 10 % in both directions). A literal divisor, negated or not, stays a plain operator: the validator rejects a literal
//! zero (E0317), so slangc has nothing to fold into a trap.
//!
//! The rows are the operand pairs `gust::math` has: `Div` and `DivAssign` for the eight vector types with themselves
//! and with their scalar, and Rust's own operators on the three scalars. `%` is scalar only: `gust::math` has no `Rem`
//! on a vector.

use super::mul_table::slang;

/// One operand pair of `lhs / rhs`, by Rust spelling.
pub const DIV_ROWS: &[(&str, &str)] = &[
    ("f32", "f32"),
    ("i32", "i32"),
    ("u32", "u32"),
    ("Vec2", "Vec2"),
    ("Vec2", "f32"),
    ("Vec3", "Vec3"),
    ("Vec3", "f32"),
    ("Vec4", "Vec4"),
    ("Vec4", "f32"),
    ("IVec2", "IVec2"),
    ("IVec2", "i32"),
    ("IVec4", "IVec4"),
    ("IVec4", "i32"),
    ("UVec2", "UVec2"),
    ("UVec2", "u32"),
    ("UVec3", "UVec3"),
    ("UVec3", "u32"),
    ("UVec4", "UVec4"),
    ("UVec4", "u32"),
];

/// One operand pair of `lhs % rhs`: the scalars only.
pub const REM_ROWS: &[(&str, &str)] = &[("f32", "f32"), ("i32", "i32"), ("u32", "u32")];

/// The overloads of one operator family (`name` is `div` or `rem`, `op` is `/` or `%`): the value forms, then the
/// `_assign` forms, then a blank line.
pub fn overload_text(name: &str, op: &str, rows: &[(&str, &str)]) -> String {
    let mut text = String::new();
    for &(left, right) in rows {
        let (l, r) = (slang(left), slang(right));
        text.push_str(&format!(
            "{l} __gust_{name}({l} a, {r} b) {{ return a {op} b; }}\n"
        ));
    }
    for &(left, right) in rows {
        let (l, r) = (slang(left), slang(right));
        text.push_str(&format!(
            "void __gust_{name}_assign(inout {l} a, {r} b) {{ a = a {op} b; }}\n"
        ));
    }
    text.push('\n');
    text
}
