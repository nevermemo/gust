//! Name and spelling tables of the validator: banned names, accepted methods and attributes, scalar-math arities.

use super::*;

pub(super) const BANNED_NAMES: &[&str] = &[
    "std",
    "core",
    "alloc",
    "String",
    "Vec",
    "Box",
    "Rc",
    "Arc",
    "HashMap",
    "HashSet",
    "println",
    "print",
    "eprintln",
    "dbg",
    "panic",
    "thread",
    "spawn",
    "File",
    "TcpStream",
];

/// Numeric conversions with a proven Slang lowering and GPU-verified semantics.
pub(super) const CAST_TARGETS: &[&str] = &["f32", "float", "i32", "int", "u32", "uint"];

/// The wrapping integer methods (slice m7.w) with their argument counts: `a.wrapping_add(b)` is the plain Slang `+`,
/// which wraps modulo 2^32 on the GPU (WGSL section 6.2.3), and the twin calls Rust's own method, so the result is
/// inside the domain for every input. `wrapping_div` and `wrapping_rem` are deliberately absent: a divisor of zero
/// has no wrapped result.
pub(crate) fn wrapping_arity(name: &str) -> Option<usize> {
    match name {
        "wrapping_add" | "wrapping_sub" | "wrapping_mul" => Some(1),
        "wrapping_neg" => Some(0),
        _ => None,
    }
}

/// The scalar math methods (slice m11.math) with their argument counts. The macro sees no types, so one emitter arm per
/// name serves every receiver type rustc allows: the float names lower to the Slang intrinsic (or to a helper for
/// `fract`), and `abs`, `min`, `max` and `clamp` are the same intrinsic on `i32` and `u32`. The twin calls the `std`
/// method unchanged. Edge inputs (a NaN in `min`, `clamp` with `min > max`, `i32::MIN.abs()`) differ between the twin
/// and the GPU and are documented, not chased (D39).
pub(crate) fn scalar_math_arity(name: &str) -> Option<usize> {
    match name {
        "abs" | "floor" | "ceil" | "trunc" | "fract" | "sqrt" | "sin" | "cos" | "tan" | "atan"
        | "exp" | "exp2" | "log2" | "ln" | "round" | "signum" | "to_radians" | "to_degrees"
        | "asin" | "acos" => Some(0),
        "min" | "max" | "atan2" | "powf" | "rem_euclid" | "hypot" | "pow" => Some(1),
        "clamp" | "mul_add" => Some(2),
        _ => None,
    }
}

/// The value of `std::f32::consts::{PI, TAU, E}` (or the `core::` path), the three constants of slice m11.math.
pub(crate) fn math_constant(path: &syn::Path) -> Option<f32> {
    if path.leading_colon.is_some() || path.segments.len() != 4 {
        return None;
    }
    if path
        .segments
        .iter()
        .any(|segment| !segment.arguments.is_empty())
    {
        return None;
    }
    let names: Vec<String> = path
        .segments
        .iter()
        .map(|segment| crate::names::name_of(&segment.ident))
        .collect();
    if !matches!(names[0].as_str(), "std" | "core") || names[1] != "f32" || names[2] != "consts" {
        return None;
    }
    match names[3].as_str() {
        "PI" => Some(std::f32::consts::PI),
        "TAU" => Some(std::f32::consts::TAU),
        "E" => Some(std::f32::consts::E),
        _ => None,
    }
}

/// The accepted methods, spelled out in the text of E0305 and E0325.
pub(super) const ACCEPTED_METHODS: &str = "the accepted methods are `.global_id()` and `.len()`, the Option/Result methods \
`.is_some()`, `.is_none()`, `.is_ok()`, `.is_err()` and `.unwrap_or(default)`, the wrapping integer methods \
`.wrapping_add(b)`, `.wrapping_sub(b)`, `.wrapping_mul(b)` and `.wrapping_neg()`, the scalar math methods \
`.abs()`, `.min(b)`, `.max(b)`, `.clamp(lo, hi)`, `.floor()`, `.ceil()`, `.trunc()`, `.fract()`, `.sqrt()`, `.sin()`, \
`.cos()`, `.tan()`, `.atan()`, `.atan2(x)`, `.exp()`, `.exp2()`, `.ln()`, `.log2()`, `.powf(b)`, `.round()`, \
`.signum()`, `.to_radians()`, `.to_degrees()`, `.asin()`, `.acos()`, `.rem_euclid(b)`, `.hypot(y)`, \
`.mul_add(b, c)` and the integer `.pow(e)`, and the vector and \
matrix methods (swizzles such as `.xy()`, `.dot(v)`, `.mul_vec4(v)`, `.mul_mat4(m)`, `.transpose()`)";

/// Rust primitives outside the four-byte scalar subset (`usize` is not here: `visit_type_path` gives it GUST-E0334
/// first). They are not shadowed by
/// the prelude, so rustc accepts them, and `emit_type` would pass the Rust
/// spelling through to Slang unchanged.
pub(super) const UNSUPPORTED_PRIMITIVES: &[&str] = &[
    "i8", "i16", "i64", "i128", "isize", "u8", "u16", "u64", "u128", "f16", "f64", "f128", "char",
    "str",
];

/// The attributes a function (and an `impl` block) may carry: documentation and lints, `derive` and `kernel` (judged
/// elsewhere). Any other has no proven Rust-to-Slang meaning and cannot be dropped.
pub(super) const ALLOWED_ATTRIBUTES: [&str; 7] =
    ["doc", "allow", "warn", "deny", "forbid", "derive", "kernel"];

/// Why `attribute` is refused, or `None` when it is on the allowlist.
pub(super) fn attribute_refusal(attribute: &syn::Attribute) -> Option<&'static str> {
    (!ALLOWED_ATTRIBUTES
        .iter()
        .any(|name| attribute.path().is_ident(name)))
    .then_some("this attribute has no proven Rust-to-Slang meaning and cannot be dropped")
}

/// Why `attribute` is refused on an `impl` block: only documentation and lint levels are (`derive` and `kernel` are
/// errors in rustc there, and `cfg` would let the Slang and the twin disagree about the methods).
pub(super) fn impl_attribute_refusal(attribute: &syn::Attribute) -> Option<&'static str> {
    (!["doc", "allow", "warn", "deny", "forbid"]
        .iter()
        .any(|name| attribute.path().is_ident(name)))
    .then_some("only documentation and lint attributes (`doc`, `allow`, `warn`, `deny`, `forbid`) are accepted on an `impl` block: any other has no proven Rust-to-Slang meaning and cannot be dropped")
}

pub(super) const QUALIFIED_MATH_PATH: &str = "math types are the prelude's: write `Vec4`, `Mat4` and their constructors without a module path (`Vec4::new`, `Mat4::IDENTITY`)";

/// Whether `path` reaches a math type through a module path (`super::Mat4`, `gust::math::Vec4::new`, `::Mat4`) instead
/// of by the prelude name. `Vec4::new` and `Mat4::IDENTITY` are the prelude form and are not qualified.
pub(super) fn is_qualified_math_path(path: &syn::Path) -> bool {
    let names_math_type = path
        .segments
        .iter()
        .any(|segment| math_types::math_type(&segment.ident.unraw().to_string()).is_some());
    if !names_math_type {
        return false;
    }
    if path.leading_colon.is_some() {
        return true;
    }
    if path.segments.len() == 1 {
        return false;
    }
    // Two segments whose first is the math type are the prelude form, whatever the second is (a member the type does
    // not have is the caller's error, not a qualification).
    !(path.segments.len() == 2
        && math_types::math_type(&path.segments[0].ident.unraw().to_string()).is_some())
}

/// Names the dialect gives a meaning to as a callee or a value: a closure cannot take one (GUST-E0328), and a body that
/// uses one has not captured anything (slice m9.c).
pub(super) fn is_dialect_callee(name: &str) -> bool {
    matches!(
        name,
        "Some"
            | "Ok"
            | "Err"
            | "None"
            | "workgroup_barrier"
            | "atomic_add"
            | "atomic_min"
            | "atomic_max"
            | "atomic_exchange"
            | "atomic_compare_exchange"
    ) || barrier::subgroup_arity(name).is_some()
}
