//! Calls and methods: math, subgroup and atomic operations, constructors and argument lists.

use syn::ext::IdentExt as _;

use super::expressions::{emit_expression, emit_value_expression};
use super::helpers::{
    FRACT_HELPER, KEEP_HELPER, SUBGROUP_AND, SUBGROUP_COUNT, SUBGROUP_INCLUSIVE_ADD,
    SUBGROUP_INCLUSIVE_MUL, SUBGROUP_OR, SUBGROUP_XOR,
};
use super::lowering::lowering;
use super::operators::{is_integer_literal, needs_keep};
use crate::math_types;

/// The two names of a call `Owner::member(..)` of two plain segments (`S::f`, `Self::f`), unraw. A math constructor is the
/// same shape but has its own arm before this one is asked.
pub(super) fn associated_call_path(call: &syn::ExprCall) -> Option<(String, String)> {
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return None;
    };
    if path.qself.is_some()
        || path.path.leading_colon.is_some()
        || path.path.segments.len() != 2
        || path
            .path
            .segments
            .iter()
            .any(|segment| !segment.arguments.is_none())
    {
        return None;
    }
    Some((
        path.path.segments[0].ident.unraw().to_string(),
        path.path.segments[1].ident.unraw().to_string(),
    ))
}

/// The math type and member of a call to a constructor of `gust::math` (`Vec4::new`, `Mat4::from_cols`, ..).
pub(super) fn math_constructor(
    call: &syn::ExprCall,
) -> Option<(&'static math_types::MathType, &'static str)> {
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    math_types::associated_item(&path.path).filter(|(_, member)| *member != "IDENTITY")
}

/// `Vec4::new(a, b, c, d)` is `float4(a, b, c, d)` and `Vec4::splat(s)` is `float4(s)`. Slang's matrix constructor
/// takes rows, so `Mat4::from_rows(r0, r1, r2, r3)` is `float4x4(r0, r1, r2, r3)` and
/// `Mat4::from_cols(c0, c1, c2, c3)`, with the matrices column-major, is the transpose of the same constructor
/// (measured: plan S6.1 (a)).
pub(super) fn emit_math_constructor(call: &syn::ExprCall) -> syn::Result<String> {
    let (math, member) = math_constructor(call).expect("the caller checked the call");
    let arguments = emit_list(call.args.iter())?.join(", ");
    Ok(match member {
        "from_cols" => format!("transpose({}({arguments}))", math.slang),
        _ => format!("{}({arguments})", math.slang),
    })
}

/// `Mat4::IDENTITY` as a constructor call: Slang has no such constant.
pub(super) fn identity_matrix(math: &math_types::MathType) -> String {
    let width = math_types::width(math);
    let row_type = format!("float{width}");
    let rows: Vec<String> = (0..width)
        .map(|row| {
            let components: Vec<&str> = (0..width)
                .map(|column| if column == row { "1.0" } else { "0.0" })
                .collect();
            format!("{row_type}({})", components.join(", "))
        })
        .collect();
    format!("{}({})", math.slang, rows.join(", "))
}

/// Whether a method call is one of the vector and matrix methods: a swizzle, `dot`, `mul_vecN`, `mul_matN` or
/// `transpose`.
pub(super) fn is_math_method(call: &syn::ExprMethodCall) -> bool {
    let name = crate::names::name_of(&call.method);
    math_types::is_swizzle(&name)
        || math_types::method_arity(&name).is_some_and(|arity| arity == call.args.len())
}

/// The Slang spelling of a scalar math method (slice m11.math). `fract` is a helper (`FRACT_HELPER`), not an intrinsic.
fn scalar_math_function(name: &str) -> &'static str {
    match name {
        "abs" => "abs",
        "min" => "min",
        "max" => "max",
        "clamp" => "clamp",
        "floor" => "floor",
        "ceil" => "ceil",
        "trunc" => "trunc",
        "sqrt" => "sqrt",
        "sin" => "sin",
        "cos" => "cos",
        "tan" => "tan",
        "atan" => "atan",
        "exp" => "exp",
        "exp2" => "exp2",
        "log2" => "log2",
        "atan2" => "atan2",
        "ln" => "log",
        "powf" => "pow",
        "asin" => "asin",
        "acos" => "acos",
        "round" => "__gust_round",
        "signum" => "__gust_signum",
        "rem_euclid" => "__gust_rem_euclid",
        "mul_add" => "fma",
        "hypot" => "__gust_hypot",
        "pow" => "__gust_pow",
        "fract" => FRACT_HELPER,
        other => {
            unreachable!("`{other}` is accepted by `scalar_math_arity` but has no emitter arm")
        }
    }
}

/// `x.sqrt()` is `sqrt(x)`, `y.atan2(x)` is `atan2(y, x)` (the receiver is y), `a.clamp(lo, hi)` is
/// `clamp(a, lo, hi)`, `x.ln()` is `log(x)`, `x.powf(e)` is `pow(x, e)` and `x.fract()` is `__gust_fract(x)`.
pub(super) fn emit_scalar_math(call: &syn::ExprMethodCall) -> syn::Result<String> {
    let name = crate::names::name_of(&call.method);
    // `to_radians` and `to_degrees` (slice m11.math2) are `std`'s own multiplication by an `f32` constant.
    if let Some(factor) = angle_factor(&name) {
        return Ok(format!(
            "({} * float({factor:?}))",
            emit_expression(&call.receiver)?
        ));
    }
    let function = scalar_math_function(&name);
    let mut operands = vec![emit_expression(&call.receiver)?];
    operands.extend(emit_list(call.args.iter())?);
    // The integer `pow` (slice m11.math3): the exponent is a `u32` in Rust, and the cast fixes it so the base alone
    // picks the `uint` or `int` overload of the helper (the macro sees no types).
    if name == "pow" {
        operands[1] = format!("uint({})", operands[1]);
    }
    Ok(format!("{function}({})", operands.join(", ")))
}

/// The `f32` factor of `to_radians` and `to_degrees`, the constants of `core::f32::<impl f32>` (read from the
/// toolchain's `rust-src`, `library/core/src/num/f32.rs`): `consts::PI / 180.0` and
/// `57.2957795130823208767981548141051703_f32`. `std` multiplies the receiver by the constant, and so do we.
fn angle_factor(name: &str) -> Option<f32> {
    match name {
        "to_radians" => Some(std::f32::consts::PI / 180.0),
        "to_degrees" => Some(
            "57.2957795130823208767981548141051703"
                .parse::<f32>()
                .expect("a decimal"),
        ),
        _ => None,
    }
}

/// `v.xy()` is `v.xy`, `a.dot(b)` is `dot(a, b)`, `m.mul_vec4(v)` and `a.mul_mat4(b)` are `mul(m, v)` and `mul(a, b)`,
/// and `m.transpose()` is `transpose(m)`.
pub(super) fn emit_math_method(call: &syn::ExprMethodCall) -> syn::Result<String> {
    let name = crate::names::name_of(&call.method);
    let receiver = emit_expression(&call.receiver)?;
    Ok(if math_types::is_swizzle(&name) {
        format!("{receiver}.{name}")
    } else if name == "transpose" {
        format!("transpose({receiver})")
    } else {
        let function = match name.as_str() {
            "dot" if lowering().dot => "__gust_dot",
            "dot" => "dot",
            _ => "mul",
        };
        let arguments = emit_list(call.args.iter())?.join(", ");
        format!("{function}({receiver}, {arguments})")
    })
}

thread_local! {
    /// Whether the kernel being emitted wrapped an element in `__gust_keep` (set by [`emit_list`], the one place that
    /// writes the call). `emit_kernel` defines the helper exactly when this is set, so no scan of the source has to
    /// agree with the emitter about which lists wrap.
    pub(super) static KEEP_USED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Emit the elements of a comma-separated list, wrapping those that need `__gust_keep`.
pub(super) fn emit_list<'a>(
    elements: impl ExactSizeIterator<Item = &'a syn::Expr>,
) -> syn::Result<Vec<String>> {
    emit_list_of(elements, false)
}

/// [`emit_list`] for the arguments of a call, where an array literal has a lowering (`f({ a, b })`, slice m9.r1).
pub(super) fn emit_arguments<'a>(
    elements: impl ExactSizeIterator<Item = &'a syn::Expr>,
) -> syn::Result<Vec<String>> {
    emit_list_of(elements, true)
}

fn emit_list_of<'a>(
    elements: impl ExactSizeIterator<Item = &'a syn::Expr>,
    arrays: bool,
) -> syn::Result<Vec<String>> {
    let len = elements.len();
    elements
        .enumerate()
        .map(|(index, element)| {
            let text = if arrays {
                emit_value_expression(element)?
            } else {
                emit_expression(element)?
            };
            Ok(if needs_keep(element, index, len) {
                KEEP_USED.with(|used| used.set(true));
                format!("{KEEP_HELPER}({text})")
            } else {
                text
            })
        })
        .collect()
}

/// The Slang of a subgroup operation (M8, D140). The macro sees no types, so an operation that Slang's own intrinsics
/// overload by operand type is the intrinsic (`WaveActiveSum(x)` is `uint`, `int` or `float` by `x`), and an operation
/// that needs a different form per type, or an operand twice, is a call of a helper that
/// [`subgroup_helper_text`] defines in overloads (the `__gust_xor` pattern): the bitwise reductions (an `i32` goes
/// through `asuint` and `asint`, which the DX12 compiler needs, S3), the inclusive scans and the count.
///
/// `subgroup_broadcast(x, lane)` is `WaveBroadcastLaneAt` for an integer literal `lane`, which slangc requires to be
/// a compile-time constant (E40012), and `WaveReadLaneAt`, WGSL's `subgroupShuffle`, for any other expression (S1).
/// `subgroup_elect()` is the first active lane comparing its own index, since naga 30.0.1 has no `subgroupElect`.
pub(super) fn emit_subgroup_call(call: &syn::ExprCall) -> syn::Result<String> {
    let name =
        crate::validate::subgroup_operation(call).expect("the caller matched a subgroup operation");
    let arguments = emit_list(call.args.iter())?;
    let list = arguments.join(", ");
    let lane_is_literal = call.args.get(1).is_some_and(is_integer_literal);
    Ok(match name {
        "subgroup_add" => format!("WaveActiveSum({list})"),
        "subgroup_mul" => format!("WaveActiveProduct({list})"),
        "subgroup_min" => format!("WaveActiveMin({list})"),
        "subgroup_max" => format!("WaveActiveMax({list})"),
        "subgroup_and" => format!("{SUBGROUP_AND}({list})"),
        "subgroup_or" => format!("{SUBGROUP_OR}({list})"),
        "subgroup_xor" => format!("{SUBGROUP_XOR}({list})"),
        "subgroup_all" => format!("WaveActiveAllTrue({list})"),
        "subgroup_any" => format!("WaveActiveAnyTrue({list})"),
        "subgroup_ballot" => format!("WaveActiveBallot({list})"),
        "subgroup_broadcast_first" => format!("WaveReadLaneFirst({list})"),
        "subgroup_broadcast" if lane_is_literal => format!("WaveBroadcastLaneAt({list})"),
        "subgroup_broadcast" => format!("WaveReadLaneAt({list})"),
        "subgroup_shuffle" => format!("WaveShuffle({list})"),
        "subgroup_exclusive_add" => format!("WavePrefixSum({list})"),
        "subgroup_exclusive_mul" => format!("WavePrefixProduct({list})"),
        "subgroup_inclusive_add" => format!("{SUBGROUP_INCLUSIVE_ADD}({list})"),
        "subgroup_inclusive_mul" => format!("{SUBGROUP_INCLUSIVE_MUL}({list})"),
        "subgroup_count" => format!("{SUBGROUP_COUNT}({list})"),
        "subgroup_elect" => {
            "(WaveGetLaneIndex() == WaveReadLaneFirst(WaveGetLaneIndex()))".to_owned()
        }
        "subgroup_size" => "WaveGetLaneCount()".to_owned(),
        "subgroup_lane" => "WaveGetLaneIndex()".to_owned(),
        other => unreachable!("`{other}` is a subgroup operation but has no emitter arm"),
    })
}

/// Whether a call is a recognized atomic operation with the expected argument count.
pub(super) fn is_atomic_call(call: &syn::ExprCall) -> bool {
    let Some(name) = atomic_op_name(call) else {
        return false;
    };
    let expected = if name == "atomic_compare_exchange" {
        3
    } else {
        2
    };
    call.args.len() == expected
}

/// The atomic operation name from the call, if recognized.
pub(super) fn atomic_op_name(call: &syn::ExprCall) -> Option<&'static str> {
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return None;
    };
    let ident = path.path.get_ident()?;
    match ident.unraw().to_string().as_str() {
        "atomic_add" => Some("atomic_add"),
        "atomic_min" => Some("atomic_min"),
        "atomic_max" => Some("atomic_max"),
        "atomic_exchange" => Some("atomic_exchange"),
        "atomic_compare_exchange" => Some("atomic_compare_exchange"),
        _ => None,
    }
}

/// The Slang method name for an atomic operation.
pub(super) fn atomic_slang_method(op_name: &str) -> Option<&'static str> {
    match op_name {
        "atomic_add" => Some("add"),
        "atomic_min" => Some("min"),
        "atomic_max" => Some("max"),
        "atomic_exchange" => Some("exchange"),
        "atomic_compare_exchange" => Some("compareExchange"),
        _ => None,
    }
}

pub(super) fn is_some_constructor(call: &syn::ExprCall) -> bool {
    matches!(call.func.as_ref(), syn::Expr::Path(path) if crate::names::path_is(&path.path, "Some"))
        && call.args.len() == 1
}

pub(super) fn is_ok_constructor(call: &syn::ExprCall) -> bool {
    matches!(call.func.as_ref(), syn::Expr::Path(path) if crate::names::path_is(&path.path, "Ok"))
        && call.args.len() == 1
}

pub(super) fn is_result_constructor(call: &syn::ExprCall) -> bool {
    is_ok_constructor(call)
        || (matches!(call.func.as_ref(), syn::Expr::Path(path) if crate::names::path_is(&path.path, "Err"))
            && call.args.len() == 1)
}
