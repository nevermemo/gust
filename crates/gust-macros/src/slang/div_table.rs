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

/// The same overloads as [`overload_text`] with the integer rules of naga's SPIR-V writer as explicit code (the SPIR-V
/// route, D172; WGSL section 8.8): a zero divisor, and for `i32` also `MIN / -1`, divides by one, so `x / 0` is `x` and
/// `x % 0` is 0. A signed remainder is `a - d * (a / d)`, naga's own form: `OpSRem` is poison for negative operands on
/// Vulkan without maintenance8 (wgpu issue 8191). An unsigned remainder is `OpUMod`, which needs only the divisor
/// guard. The float rows keep the plain operator. Vector rows apply the scalar helper to each lane.
pub fn guarded_overload_text(name: &str, op: &str, rows: &[(&str, &str)]) -> String {
    let remainder = name == "rem";
    let integer = |kind: &str| kind.starts_with("int") || kind.starts_with("uint");
    let scalar_body = |kind: &str| -> String {
        match (kind, remainder) {
            ("uint", false) => "uint d = (b == 0u) ? 1u : b; return a / d;".to_owned(),
            ("uint", true) => "uint d = (b == 0u) ? 1u : b; return a % d;".to_owned(),
            ("int", false) => {
                "int d = (b == 0 || (a == (-2147483647 - 1) && b == -1)) ? 1 : b; return a / d;"
                    .to_owned()
            }
            ("int", true) => {
                "int d = (b == 0 || (a == (-2147483647 - 1) && b == -1)) ? 1 : b; return a - d * (a / d);"
                    .to_owned()
            }
            _ => format!("return a {op} b;"),
        }
    };
    let mut text = String::new();
    for &(left, right) in rows {
        let (l, r) = (slang(left), slang(right));
        let body = if l == "int" || l == "uint" {
            scalar_body(l)
        } else if integer(l) {
            let width = l
                .chars()
                .last()
                .and_then(|digit| digit.to_digit(10))
                .expect("a vector type ends in its width") as usize;
            let lanes = ["x", "y", "z", "w"];
            let parts: Vec<String> = lanes[..width]
                .iter()
                .map(|lane| {
                    if r == l {
                        format!("__gust_{name}(a.{lane}, b.{lane})")
                    } else {
                        format!("__gust_{name}(a.{lane}, b)")
                    }
                })
                .collect();
            format!("return {l}({});", parts.join(", "))
        } else {
            format!("return a {op} b;")
        };
        text.push_str(&format!("{l} __gust_{name}({l} a, {r} b) {{ {body} }}\n"));
    }
    for &(left, right) in rows {
        let (l, r) = (slang(left), slang(right));
        if integer(l) {
            text.push_str(&format!(
                "void __gust_{name}_assign(inout {l} a, {r} b) {{ a = __gust_{name}(a, b); }}\n"
            ));
        } else {
            text.push_str(&format!(
                "void __gust_{name}_assign(inout {l} a, {r} b) {{ a = a {op} b; }}\n"
            ));
        }
    }
    text.push('\n');
    text
}

/// The overloads a module that divides gets: the plain ones, or the guarded ones when the guards of the SPIR-V route
/// are on (see `guards`).
pub fn module_overload_text(name: &str, op: &str, rows: &[(&str, &str)]) -> String {
    if super::guards::enabled() {
        guarded_overload_text(name, op, rows)
    } else {
        overload_text(name, op, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plain_text_is_unchanged_and_the_guarded_text_has_the_integer_rules() {
        let plain = overload_text("div", "/", DIV_ROWS);
        assert!(plain.contains("int __gust_div(int a, int b) { return a / b; }\n"));
        let guarded = guarded_overload_text("div", "/", DIV_ROWS);
        assert!(guarded.contains(
            "int __gust_div(int a, int b) { int d = (b == 0 || (a == (-2147483647 - 1) && b == -1)) ? 1 : b; return a / d; }\n"
        ));
        assert!(guarded.contains(
            "uint __gust_div(uint a, uint b) { uint d = (b == 0u) ? 1u : b; return a / d; }\n"
        ));
        assert!(guarded.contains("float __gust_div(float a, float b) { return a / b; }\n"));
        assert!(guarded.contains(
            "uint2 __gust_div(uint2 a, uint b) { return uint2(__gust_div(a.x, b), __gust_div(a.y, b)); }\n"
        ));
        assert!(guarded.contains(
            "int4 __gust_div(int4 a, int4 b) { return int4(__gust_div(a.x, b.x), __gust_div(a.y, b.y), __gust_div(a.z, b.z), __gust_div(a.w, b.w)); }\n"
        ));
        assert!(
            guarded.contains(
                "void __gust_div_assign(inout int2 a, int b) { a = __gust_div(a, b); }\n"
            )
        );
        assert!(
            guarded.contains("void __gust_div_assign(inout float3 a, float b) { a = a / b; }\n")
        );
        let rem = guarded_overload_text("rem", "%", REM_ROWS);
        assert!(rem.contains("return a - d * (a / d); }\n"));
        assert!(rem.contains("uint d = (b == 0u) ? 1u : b; return a % d; }\n"));
        assert!(rem.contains("float __gust_rem(float a, float b) { return a % b; }\n"));
        assert_eq!(
            guarded.lines().count(),
            plain.lines().count(),
            "one line per row in both"
        );
    }

    #[test]
    fn the_route_picks_the_text() {
        {
            let _off = super::super::guards::Forced::new(false);
            assert_eq!(
                module_overload_text("div", "/", DIV_ROWS),
                overload_text("div", "/", DIV_ROWS)
            );
        }
        let _on = super::super::guards::Forced::new(true);
        assert_eq!(
            module_overload_text("rem", "%", REM_ROWS),
            guarded_overload_text("rem", "%", REM_ROWS)
        );
    }
}
