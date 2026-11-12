use std::collections::{HashMap, HashSet};

use quote::ToTokens;
use syn::ext::IdentExt as _;
use syn::visit::{self, Visit};

use crate::diagnostics::{self, Code};
use crate::math_types;
use crate::profile::{BodyChecks, StageProfile};
use crate::slang::texture;

mod barrier;
mod callgraph;
pub(crate) mod classify;
pub(crate) mod limits;
pub(crate) mod traits;

pub(crate) use barrier::{is_barrier_call, subgroup_operation, workgroup_array};

const BANNED_NAMES: &[&str] = &[
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
const CAST_TARGETS: &[&str] = &["f32", "float", "i32", "int", "u32", "uint"];

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

/// Why `expression` is visibly not a `u32` texel address (GUST-E0732), or `None` when it may be one: the macro sees no
/// types, so it refuses what is plainly another type and leaves the rest to rustc, which checks the shadow body.
fn address_problem(expression: &syn::Expr) -> Option<&'static str> {
    match expression {
        syn::Expr::Paren(paren) => address_problem(&paren.expr),
        syn::Expr::Group(group) => address_problem(&group.expr),
        syn::Expr::Lit(literal) => match &literal.lit {
            syn::Lit::Float(_) => Some("a float literal"),
            syn::Lit::Int(int) if !matches!(int.suffix(), "" | "u32") => {
                Some("an integer literal that is not a `u32`")
            }
            _ => None,
        },
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            Some("a negated value (a texel address cannot be negative)")
        }
        syn::Expr::Cast(cast) => match cast.ty.as_ref() {
            syn::Type::Path(path) if path.path.is_ident("u32") || path.path.is_ident("uint") => {
                None
            }
            _ => Some("a cast to a type other than `u32`"),
        },
        _ => None,
    }
}

/// Why `expression` is visibly not an `f32` sample coordinate (GUST-E0732), or `None`: the counterpart of
/// [`address_problem`].
fn coordinate_problem(expression: &syn::Expr) -> Option<&'static str> {
    match expression {
        syn::Expr::Paren(paren) => coordinate_problem(&paren.expr),
        syn::Expr::Group(group) => coordinate_problem(&group.expr),
        syn::Expr::Lit(literal) => match &literal.lit {
            syn::Lit::Int(_) => {
                Some("an integer literal (write `0.5`, or `1.0` for a whole number)")
            }
            syn::Lit::Float(float) if !matches!(float.suffix(), "" | "f32") => {
                Some("a float literal that is not an `f32`")
            }
            _ => None,
        },
        syn::Expr::Cast(cast) => match cast.ty.as_ref() {
            syn::Type::Path(path) if path.path.is_ident("f32") || path.path.is_ident("float") => {
                None
            }
            _ => Some("a cast to a type other than `f32`"),
        },
        _ => None,
    }
}

/// The accepted methods, spelled out in the text of E0305 and E0325.
const ACCEPTED_METHODS: &str = "the accepted methods are `.global_id()` and `.len()`, the Option/Result methods \
`.is_some()`, `.is_none()`, `.is_ok()`, `.is_err()` and `.unwrap_or(default)`, the wrapping integer methods \
`.wrapping_add(b)`, `.wrapping_sub(b)`, `.wrapping_mul(b)` and `.wrapping_neg()`, and the vector and matrix methods \
(swizzles such as `.xy()`, `.dot(v)`, `.mul_vec4(v)`, `.mul_mat4(m)`, `.transpose()`)";

/// Rust primitives outside the four-byte scalar subset (`usize` is not here: `visit_type_path` gives it GUST-E0334
/// first). They are not shadowed by
/// the prelude, so rustc accepts them, and `emit_type` would pass the Rust
/// spelling through to Slang unchanged.
const UNSUPPORTED_PRIMITIVES: &[&str] = &[
    "i8", "i16", "i64", "i128", "isize", "u8", "u16", "u64", "u128", "f16", "f64", "f128", "char",
    "str",
];

/// The `Option<T>` shape this dialect lowers to Slang `Optional<T>`: a single
/// segment with exactly one type argument.
fn option_payload(ty: &syn::TypePath) -> Option<Option<&syn::Type>> {
    if ty.qself.is_some() || ty.path.segments.len() != 1 {
        return None;
    }
    let segment = &ty.path.segments[0];
    if segment.ident != "Option" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return Some(None);
    };
    let mut types = arguments.args.iter().filter_map(|argument| match argument {
        syn::GenericArgument::Type(ty) => Some(ty),
        _ => None,
    });
    match (types.next(), types.next(), arguments.args.len()) {
        (Some(payload), None, 1) => Some(Some(payload)),
        _ => Some(None),
    }
}

/// The deterministic `Result<T, T>` subset lowered to `__GustResult<T>`.
fn result_payload(ty: &syn::TypePath) -> Option<Option<(&syn::Type, &syn::Type)>> {
    if ty.qself.is_some() || ty.path.segments.len() != 1 {
        return None;
    }
    let segment = &ty.path.segments[0];
    if segment.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return Some(None);
    };
    let types = arguments
        .args
        .iter()
        .filter_map(|argument| match argument {
            syn::GenericArgument::Type(ty) => Some(ty),
            _ => None,
        })
        .collect::<Vec<_>>();
    match types.as_slice() {
        [ok, err] if arguments.args.len() == 2 => Some(Some((*ok, *err))),
        _ => Some(None),
    }
}

/// The element type of an `RWStructuredBuffer<T>` parameter, when the type is a
/// single-segment path with exactly one type argument.
fn rw_buffer_element_type(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    if type_path.qself.is_some() || type_path.path.segments.len() != 1 {
        return None;
    }
    let segment = &type_path.path.segments[0];
    if segment.ident != "RWStructuredBuffer" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    let syn::GenericArgument::Type(element) = arguments.args.first()? else {
        return None;
    };
    let syn::Type::Path(element_path) = element else {
        return None;
    };
    element_path.path.get_ident().map(ToString::to_string)
}

/// `Some(identifier)` in an `if let`, the only pattern with a direct Slang lowering.
pub fn option_pattern(pattern: &syn::Pat) -> Option<&syn::PatIdent> {
    let syn::Pat::TupleStruct(tuple) = pattern else {
        return None;
    };
    if !tuple.path.is_ident("Some") || tuple.elems.len() != 1 {
        return None;
    }
    match &tuple.elems[0] {
        syn::Pat::Ident(ident) => Some(ident),
        _ => None,
    }
}

pub fn result_pattern(pattern: &syn::Pat) -> Option<&syn::PatIdent> {
    let syn::Pat::TupleStruct(tuple) = pattern else {
        return None;
    };
    if !tuple.path.is_ident("Ok") || tuple.elems.len() != 1 {
        return None;
    }
    match &tuple.elems[0] {
        syn::Pat::Ident(ident) => Some(ident),
        _ => None,
    }
}

fn err_pattern(pattern: &syn::Pat) -> Option<&syn::PatIdent> {
    let syn::Pat::TupleStruct(tuple) = pattern else {
        return None;
    };
    if !tuple.path.is_ident("Err") || tuple.elems.len() != 1 {
        return None;
    }
    match &tuple.elems[0] {
        syn::Pat::Ident(ident) => Some(ident),
        _ => None,
    }
}

pub fn result_match_arms(
    expression: &syn::ExprMatch,
) -> syn::Result<(&syn::PatIdent, &syn::Block, &syn::PatIdent, &syn::Block)> {
    let mut ok = None;
    let mut err = None;
    for arm in &expression.arms {
        if !arm.attrs.is_empty() {
            return Err(diagnostics::error(
                Code::MatchArmAttribute,
                &arm.attrs[0],
                "match arm attributes are not supported",
            ));
        }
        if arm.guard.is_some() {
            return Err(diagnostics::error(
                Code::MatchGuard,
                arm,
                "match guards are not supported in the Rust-to-Slang subset",
            ));
        }
        let syn::Expr::Block(body) = arm.body.as_ref() else {
            return Err(diagnostics::error(
                Code::MatchArmBlock,
                &arm.body,
                "Result matches require block arms",
            ));
        };
        if let Some(binding) = result_pattern(&arm.pat) {
            if binding.mutability.is_some() || binding.by_ref.is_some() || binding.subpat.is_some()
            {
                return Err(diagnostics::error(
                    Code::MatchBinding,
                    binding,
                    "Result match bindings must be plain identifiers; `mut`, `ref`, and @ bindings are not supported",
                ));
            }
            if ok.replace((binding, &body.block)).is_some() {
                return Err(diagnostics::error(
                    Code::MatchArmCount,
                    &arm.pat,
                    "Result matches require exactly one `Ok` arm and one `Err` arm",
                ));
            }
        } else if let Some(binding) = err_pattern(&arm.pat) {
            if binding.mutability.is_some() || binding.by_ref.is_some() || binding.subpat.is_some()
            {
                return Err(diagnostics::error(
                    Code::MatchBinding,
                    binding,
                    "Result match bindings must be plain identifiers; `mut`, `ref`, and @ bindings are not supported",
                ));
            }
            if err.replace((binding, &body.block)).is_some() {
                return Err(diagnostics::error(
                    Code::MatchArmCount,
                    &arm.pat,
                    "Result matches require exactly one `Ok` arm and one `Err` arm",
                ));
            }
        } else {
            return Err(diagnostics::error(
                Code::MatchPattern,
                &arm.pat,
                "Result matches support only `Ok(identifier)` and `Err(identifier)` patterns",
            ));
        }
    }
    match (ok, err) {
        (Some((ok_name, ok_body)), Some((err_name, err_body))) => {
            Ok((ok_name, ok_body, err_name, err_body))
        }
        _ => Err(diagnostics::error(
            Code::MatchArmCount,
            expression,
            "Result matches require exactly one `Ok` arm and one `Err` arm",
        )),
    }
}

/// The atomic operation name if the path is a recognized atomic intrinsic.
fn atomic_operation_name(path: &syn::ExprPath) -> Option<&'static str> {
    match path.path.get_ident()?.to_string().as_str() {
        "atomic_add" => Some("atomic_add"),
        "atomic_min" => Some("atomic_min"),
        "atomic_max" => Some("atomic_max"),
        "atomic_exchange" => Some("atomic_exchange"),
        "atomic_compare_exchange" => Some("atomic_compare_exchange"),
        _ => None,
    }
}

/// Whether an expression is an integer literal with value zero, including a
/// unary negation, parentheses and a macro-expanded group around one (the spellings the emitter keeps as a plain
/// operator, so a zero divisor never reaches slangc). Float zeros are excluded: floating-point division by
/// zero is defined (it yields infinity/NaN), so only integer div/rem by zero is
/// a portable-semantic hazard.
pub(crate) fn is_literal_zero(expression: &syn::Expr) -> bool {
    match expression {
        syn::Expr::Lit(literal) => matches!(
            &literal.lit,
            syn::Lit::Int(value) if value.base10_digits().bytes().all(|digit| digit == b'0')
        ),
        syn::Expr::Paren(paren) => is_literal_zero(&paren.expr),
        syn::Expr::Group(group) => is_literal_zero(&group.expr),
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            is_literal_zero(&unary.expr)
        }
        _ => false,
    }
}

/// The value of a constant shift amount: an integer literal (any suffix or none), through parentheses, groups, unary
/// `-` and `as` casts to `u32`/`uint` or `i32`/`int` (the cast wraps as in Rust), or `None` for anything else. A literal
/// too large for `i128` saturates, so it is refused. No arithmetic is folded: a sum of literals, a local bound to a
/// literal or a float literal cast passes the macro (a known gap): Slang folds it to a constant, naga rejects the
/// WGSL, and rustc refuses it when building (`arithmetic_overflow`, not under `cargo check`). Only an amount Slang
/// cannot fold (from a buffer, a parameter, a loop variable) is run-time, and only that kind masks on the GPU (KM-04).
fn constant_shift_amount(expression: &syn::Expr) -> Option<i128> {
    match expression {
        syn::Expr::Lit(literal) => match &literal.lit {
            syn::Lit::Int(value) => Some(value.base10_parse::<i128>().unwrap_or(i128::MAX)),
            _ => None,
        },
        syn::Expr::Paren(paren) => constant_shift_amount(&paren.expr),
        syn::Expr::Group(group) => constant_shift_amount(&group.expr),
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            constant_shift_amount(&unary.expr).map(|value| value.saturating_neg())
        }
        syn::Expr::Cast(cast) => {
            let value = constant_shift_amount(&cast.expr)?;
            let syn::Type::Path(target) = strip_type_parens(&cast.ty) else {
                return None;
            };
            if target.path.is_ident("u32") || target.path.is_ident("uint") {
                Some(i128::from(value as u32))
            } else if target.path.is_ident("i32") || target.path.is_ident("int") {
                Some(i128::from(value as i32))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// A loop bound that is a bare integer literal (through parentheses and unary
/// minus) without a type suffix. Slang would type it `int` while rustc may infer
/// `u32` from the other bound, so the counter types would silently diverge.
fn is_unsuffixed_integer_literal(expression: &syn::Expr) -> bool {
    match expression {
        syn::Expr::Lit(literal) => {
            matches!(&literal.lit, syn::Lit::Int(value) if value.suffix().is_empty())
        }
        syn::Expr::Paren(paren) => is_unsuffixed_integer_literal(&paren.expr),
        syn::Expr::Group(group) => is_unsuffixed_integer_literal(&group.expr),
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            is_unsuffixed_integer_literal(&unary.expr)
        }
        _ => false,
    }
}

/// The first unsuffixed integer literal of an initializer whose type nothing in the initializer itself fixes, or `None`
/// when some operand is typed (a suffix, a cast, a name, a call, a field, an index) or the expression is something
/// else. Covered: a bare literal, through parentheses, groups, unary `-` and `!`; arithmetic and bitwise binary
/// operators with a literal-only operand on each side; a shift with a literal-only left operand (the result takes the
/// left operand's type, whatever the right one is); `Some(..)`, `Ok(..)` and `Err(..)` of a literal-only payload;
/// `X.unwrap_or(Y)` with both literal-only (`Some(7).unwrap_or(0)`). Also applied to an `if let` scrutinee.
/// Comparisons and `&&`/`||` give `bool`, so they are not covered. Slang types such an initializer `int`, while rustc
/// types the local from its later uses (`u32` for `d /= y` with `y: u32`), so the two would silently diverge (D86).
fn unsuffixed_literal_only(expression: &syn::Expr) -> Option<&syn::Expr> {
    match expression {
        syn::Expr::Lit(literal) => {
            matches!(&literal.lit, syn::Lit::Int(value) if value.suffix().is_empty())
                .then_some(expression)
        }
        syn::Expr::Paren(paren) => unsuffixed_literal_only(&paren.expr),
        syn::Expr::Group(group) => unsuffixed_literal_only(&group.expr),
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_) | syn::UnOp::Not(_)) => {
            unsuffixed_literal_only(&unary.expr)
        }
        syn::Expr::Binary(binary)
            if matches!(binary.op, syn::BinOp::Shl(_) | syn::BinOp::Shr(_)) =>
        {
            unsuffixed_literal_only(&binary.left)
        }
        syn::Expr::Binary(binary)
            if matches!(
                binary.op,
                syn::BinOp::Add(_)
                    | syn::BinOp::Sub(_)
                    | syn::BinOp::Mul(_)
                    | syn::BinOp::Div(_)
                    | syn::BinOp::Rem(_)
                    | syn::BinOp::BitAnd(_)
                    | syn::BinOp::BitOr(_)
                    | syn::BinOp::BitXor(_)
            ) =>
        {
            let left = unsuffixed_literal_only(&binary.left)?;
            unsuffixed_literal_only(&binary.right)?;
            Some(left)
        }
        syn::Expr::Call(call)
            if call.args.len() == 1
                && matches!(
                    call.func.as_ref(),
                    syn::Expr::Path(path)
                        if path.path.is_ident("Some") || path.path.is_ident("Ok") || path.path.is_ident("Err")
                ) =>
        {
            unsuffixed_literal_only(&call.args[0])
        }
        // `Some(7).unwrap_or(0)`: the receiver is an `Option<{integer}>`, which rustc accepts, and the default is
        // the same unconstrained integer. (On a bare integer receiver rustc itself refuses the method call.)
        syn::Expr::MethodCall(call) if call.method == "unwrap_or" && call.args.len() == 1 => {
            let receiver = unsuffixed_literal_only(&call.receiver)?;
            unsuffixed_literal_only(&call.args[0])?;
            Some(receiver)
        }
        _ => None,
    }
}

/// A rejection reason without a span: the code and message the caller attaches to
/// the node it was checking (see [`diagnostics::error`]).
pub type Reason = (Code, &'static str);

/// The loop-variable binding of a `for` loop: `_` or a plain immutable identifier.
pub fn loop_variable(pattern: &syn::Pat) -> Result<Option<&syn::PatIdent>, Reason> {
    match pattern {
        syn::Pat::Wild(_) => Ok(None),
        syn::Pat::Ident(ident) if ident.mutability.is_some() => Err((
            Code::MutableLoopVariable,
            "the loop variable is immutable in the Rust-to-Slang subset: Slang's counter would diverge from Rust's fresh per-iteration binding; copy it into a `let mut` local",
        )),
        syn::Pat::Ident(ident) if ident.by_ref.is_some() || ident.subpat.is_some() => Err((
            Code::BindingMode,
            "reference and @ binding modes are not supported in GPU code",
        )),
        syn::Pat::Ident(ident) => Ok(Some(ident)),
        _ => Err((
            Code::LoopVariablePattern,
            "the `for` loop variable must be a single identifier or `_`",
        )),
    }
}

/// The `start..end` bounds of a `for` loop, or the diagnostic explaining why the
/// iterable has no bounded Slang lowering.
pub fn range_bounds(iterable: &syn::Expr) -> Result<(&syn::Expr, &syn::Expr), Reason> {
    let syn::Expr::Range(range) = iterable else {
        return Err((
            Code::ForIterable,
            "`for` iterates only over an exclusive integer range `start..end`; buffers, iterators, and adaptors are not iterable in the Rust-to-Slang subset",
        ));
    };
    if matches!(range.limits, syn::RangeLimits::Closed(_)) {
        return Err((
            Code::InclusiveRange,
            "inclusive ranges (`..=`) are not supported: when `end` is the type maximum the Slang counter never exceeds it and the loop would not terminate; use `start..end + 1`",
        ));
    }
    match (range.start.as_deref(), range.end.as_deref()) {
        (Some(start), Some(end)) => Ok((start, end)),
        _ => Err((
            Code::OpenRange,
            "`for` ranges need both bounds (`start..end`) so the trip count is fixed at loop entry",
        )),
    }
}

/// `#[gpu]`, `#[gust::gpu]` or `#[::gust::gpu]`: a path that ends in `gpu`.
pub(crate) fn is_gpu_attribute(attribute: &syn::Attribute) -> bool {
    attribute
        .path()
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "gpu")
}

/// [`validate_module_in`] for the `grid` profile, the entry point of the tests; the expander passes its profile.
#[cfg(test)]
pub fn validate_module(module: &syn::ItemMod) -> syn::Result<()> {
    validate_module_in(&crate::profile::GRID, module)
}

/// The names of the helper functions of `module`: every function that is not a `#[kernel]`. These are the free
/// functions a kernel body may call (`visit_expr_call`); the shape classifier reads the same set (D94). The set also
/// holds one marker per method of an `impl` block, `.name`, which is not an identifier and so never clashes with a
/// function: it is how the classifier knows a method call as a call of the module's own method (slice m9.a1). An
/// associated function (no receiver) has the marker `Struct::name` instead (slice m9.a2).
pub(crate) fn helper_names(module: &syn::ItemMod) -> HashSet<String> {
    let items = module.content.iter().flat_map(|(_, items)| items);
    let methods = items.clone().flat_map(|item| match item {
        syn::Item::Impl(block) => {
            let owner = impl_self_name(block).unwrap_or_default();
            block
                .items
                .iter()
                .filter_map(|inner| match inner {
                    syn::ImplItem::Fn(method) if receiver_of(&method.sig).is_some() => {
                        Some(format!(".{}", method.sig.ident.unraw()))
                    }
                    syn::ImplItem::Fn(method) => {
                        Some(format!("{owner}::{}", method.sig.ident.unraw()))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
        }
        _ => Vec::new(),
    });
    items
        .filter_map(|item| match item {
            syn::Item::Fn(function)
                if !function
                    .attrs
                    .iter()
                    .any(|attribute| attribute.path().is_ident("kernel")) =>
            {
                Some(function.sig.ident.to_string())
            }
            _ => None,
        })
        .chain(methods)
        .collect()
}

/// Validates a module whose kernels have the shape of the profile `stage`: the profile decides which parameter is
/// the invocation, which options a kernel may carry and which constructs it may declare.
pub(crate) fn validate_module_in(
    stage: &dyn StageProfile,
    module: &syn::ItemMod,
) -> syn::Result<()> {
    if module.content.is_none() {
        return Err(diagnostics::error(
            Code::InlineModuleRequired,
            module,
            "#[gpu] requires an inline module; use `#[gpu] mod name { ... }`",
        ));
    }
    // A trait path call (`T::m(&s, a)`, `<S as T>::m(&s, a)`) is judged as the method call it lowers to (slice m9.g1);
    // a module without a trait is judged as written.
    let desugared = traits::desugar(module)?;
    let module = desugared.as_ref().unwrap_or(module);
    let (_, items) = module.content.as_ref().expect("checked above");

    let function_names = helper_names(module);
    let trait_table = traits::trait_table(items);
    // `dyn`, `impl Trait` and a trait as a type come before every other check of a type (GUST-E0218).
    traits::check_type_uses(items, &trait_table)?;
    let methods = method_table(items);
    let struct_fields: HashMap<String, HashSet<String>> = items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Struct(item) => Some((
                item.ident.unraw().to_string(),
                item.fields
                    .iter()
                    .filter_map(|field| field.ident.as_ref().map(|name| name.unraw().to_string()))
                    .collect(),
            )),
            _ => None,
        })
        .collect();

    let result_helpers = items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Fn(function)
                if matches!(&function.sig.output, syn::ReturnType::Type(_, ty) if matches!(ty.as_ref(), syn::Type::Path(path) if result_payload(path).is_some())) =>
            {
                Some(function.sig.ident.to_string())
            }
            _ => None,
        })
        .collect();
    // Before the item loop: a `Workgroup<[Mode; 4]>` parameter would otherwise stop at the workgroup shape check (E0210).
    check_enum_layout(stage, items, &methods.enums)?;
    check_tuple_layout(stage, items)?;
    check_array_layout(stage, items)?;

    let context = ImplContext {
        function_names: &function_names,
        result_helpers: &result_helpers,
        methods: &methods,
        struct_fields: &struct_fields,
        traits: &trait_table,
        items,
    };
    for item in items {
        match item {
            syn::Item::Fn(function) => {
                if let Some(attribute) = function
                    .attrs
                    .iter()
                    .find(|attribute| is_gpu_attribute(attribute))
                {
                    return Err(diagnostics::error(
                        Code::NestedGpuFunction,
                        attribute,
                        "`#[gpu]` on a function goes on a free function outside any `#[gpu]` module: a module \
                         already is GPU code, and its functions are helpers or `#[kernel]`s",
                    ));
                }
                validate_function(
                    stage,
                    function,
                    &function_names,
                    &result_helpers,
                    &methods,
                    None,
                    false,
                )?
            }
            syn::Item::Struct(item) => validate_struct(item)?,
            syn::Item::Impl(item) => validate_impl(stage, item, &context)?,
            syn::Item::Trait(item) => validate_trait(stage, item, &context)?,
            syn::Item::Enum(item) => validate_enum(item)?,
            _ => {
                return Err(diagnostics::error(
                    Code::UnsupportedModuleItem,
                    item,
                    "GPU modules may currently contain only functions, plain structs, field-less enums, traits with methods only and `impl` blocks of those structs and enums; constants, `use` items and type aliases are not accepted",
                ));
            }
        }
    }
    let structs: HashMap<String, &syn::ItemStruct> = items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Struct(item) => Some((item.ident.unraw().to_string(), item)),
            _ => None,
        })
        .collect();
    // The shape of the uniform parameters and of the structs they reach comes before the padding checks: a `Mat2` in
    // a uniform struct is E0205 whatever else is wrong with its layout.
    crate::uniform::check_parameters(stage, items)?;
    for item in items {
        if let syn::Item::Struct(item) = item {
            check_padding(item, &structs)?;
        }
    }
    crate::uniform::check_padding(stage, items)?;
    // The call graph comes last, so a module that has another fault is reported for that fault alone (D162, E0120).
    callgraph::check(items)
}

/// The size and alignment of a field type the macro can compute from names alone: a scalar, a vector, a matrix or a
/// struct of the module whose fields are all such types. `None` when any part is unknown (a type the macro cannot see
/// into), so the `const` assertions the expansion emits stay the check for it.
fn field_layout(
    ty: &syn::Type,
    structs: &HashMap<String, &syn::ItemStruct>,
    depth: usize,
) -> Option<(u32, u32)> {
    // `[T; N]` is `N` elements of `T` end to end with the alignment of `T` (a Rust array has no padding between
    // elements, and `GpuPod for [T; N]` admits only elements whose stride is their size; slice m9.r2a).
    if let syn::Type::Array(array) = strip_type_parens(ty) {
        let length = crate::slang::array_length_literal(&array.len)?;
        if depth > 16 {
            return None;
        }
        let (size, alignment) = field_layout(&array.elem, structs, depth + 1)?;
        return Some((size.checked_mul(length)?, alignment));
    }
    let syn::Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() || path.path.leading_colon.is_some() {
        return None;
    }
    let ident = path.path.get_ident()?;
    let name = ident.unraw().to_string();
    if let Some(layout) = math_types::known_layout(&name) {
        return Some(layout);
    }
    // A struct cannot contain itself, but the guard keeps a malformed module from recursing.
    if depth > 16 {
        return None;
    }
    struct_layout(structs.get(&name)?, structs, depth + 1)
        .map(|(size, alignment, _)| (size, alignment))
}

/// A struct's `#[repr(C)]` size, alignment and the offsets of its fields, when every field has a known layout.
fn struct_layout(
    item: &syn::ItemStruct,
    structs: &HashMap<String, &syn::ItemStruct>,
    depth: usize,
) -> Option<(u32, u32, Vec<u32>)> {
    let syn::Fields::Named(fields) = &item.fields else {
        return None;
    };
    let mut offset = 0_u32;
    let mut alignment = 1_u32;
    let mut offsets = Vec::new();
    for field in &fields.named {
        let (size, field_alignment) = field_layout(&field.ty, structs, depth)?;
        offset = offset.checked_next_multiple_of(field_alignment)?;
        offsets.push(offset);
        offset = offset.checked_add(size)?;
        alignment = alignment.max(field_alignment);
    }
    Some((
        offset.checked_next_multiple_of(alignment)?,
        alignment,
        offsets,
    ))
}

/// E0208: a struct whose layout has implicit padding is not a `GpuPod` type (`pod_slice_as_bytes` would read the
/// padding bytes, which are uninitialised). The model covers scalars, vectors, matrices and the module's structs;
/// anything else is left to the expansion's `const` assertions.
fn check_padding(
    item: &syn::ItemStruct,
    structs: &HashMap<String, &syn::ItemStruct>,
) -> syn::Result<()> {
    let syn::Fields::Named(fields) = &item.fields else {
        return Ok(());
    };
    let Some((size, _, offsets)) = struct_layout(item, structs, 0) else {
        return Ok(());
    };
    let mut end = 0_u32;
    for (field, offset) in fields.named.iter().zip(offsets) {
        let Some((field_size, _)) = field_layout(&field.ty, structs, 0) else {
            return Ok(());
        };
        let name = field
            .ident
            .as_ref()
            .map_or_else(String::new, ToString::to_string);
        if offset != end {
            return Err(diagnostics::error(
                Code::ImplicitPadding,
                field,
                format!(
                    "field `{name}` of `{}` would sit at offset {offset}, {} bytes after the previous field ends at {end}: GPU structs may not contain implicit padding; order the fields by decreasing alignment, or fill the gap with an explicit field",
                    item.ident,
                    offset - end,
                ),
            ));
        }
        end = offset + field_size;
    }
    if size != end {
        return Err(diagnostics::error(
            Code::ImplicitPadding,
            &item.ident,
            format!(
                "`{}` ends at byte {end} but its size is {size}: GPU structs may not contain implicit tail padding; add explicit fields up to a multiple of the largest field alignment",
                item.ident,
            ),
        ));
    }
    Ok(())
}

/// What the methods of one name have in common (slices m9.a1 and m9.a2). The macro sees no types, so a call
/// `recv.m(..)` is judged by the name alone, as the conservative call graph of D162 judges it.
#[derive(Clone, Copy, Default)]
struct MethodFacts {
    /// Some method of the name takes `&mut self`.
    ref_mut: bool,
    /// Some method of the name takes `self` by value.
    by_value: bool,
}

impl MethodFacts {
    /// Whether a call of the name may write through its receiver or consume it: a storage, uniform or workgroup place
    /// must not be the receiver (GUST-E0333).
    fn mutating(self) -> bool {
        self.ref_mut || self.by_value
    }
}

/// The members of the module's `impl` blocks: the methods by name, and every function of an impl by `Struct::name`.
#[derive(Clone, Default)]
struct MethodTable {
    /// Methods (functions with a receiver) by name.
    by_name: HashMap<String, MethodFacts>,
    /// Every function of an `impl` block, by `Struct::name`, with whether it has a receiver.
    members: HashMap<String, bool>,
    /// Whether the module names `Mat2`, `Mat3` or `Mat4`: the emitter then lowers `*=` to `__gust_mul` (D79).
    matrix: bool,
    /// The module's enums by name (slice m9.e1). They live in this table because the visitor already holds it.
    enums: HashMap<String, EnumFacts>,
    /// The helper functions whose result type is an enum of the module, by name, with the enum's name.
    enum_returns: HashMap<String, String>,
    /// The helpers and methods whose result type is a tuple, by name (slice m9.t). A method is known by its name alone,
    /// so a name that some impl gives a tuple result counts for every call of that name.
    tuple_returns: HashSet<String>,
    /// The array fields of the module's structs by field name (slice m9.r2a, see `slang::array_field_lengths`): the
    /// length, or `None` when two structs disagree about the name. For `s.items.len()`.
    array_fields: HashMap<String, Option<u32>>,
}

/// What the validator knows of a field-less enum of the module (slice m9.e1, D157).
#[derive(Clone, Debug, Default)]
struct EnumFacts {
    /// Whether the repr is `i32` (else `u32`).
    signed: bool,
    /// The variant names, unraw.
    variants: Vec<String>,
}

impl EnumFacts {
    /// The integer type the enum casts to, as Rust spells it.
    fn repr_name(&self) -> &'static str {
        if self.signed { "i32" } else { "u32" }
    }
}

/// Whether the attributes of an enum carry `#[repr(i32)]` (the strict reading of `repr` is `validate_enum`'s).
fn enum_is_signed(attributes: &[syn::Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        attribute.path().is_ident("repr")
            && matches!(&attribute.meta, syn::Meta::List(list)
                if list.tokens.clone().into_iter().any(|tree| matches!(tree,
                    proc_macro2::TokenTree::Ident(ident) if ident == "i32")))
    })
}

/// The name of the enum of `module` that `ty` is, when it is a plain name of one.
fn enum_type_name(ty: &syn::Type, enums: &HashMap<String, EnumFacts>) -> Option<String> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    let name = path.path.get_ident()?.unraw().to_string();
    enums.contains_key(&name).then_some(name)
}

/// Whether any identifier in `stream` is one of the matrix type names (as `slang::module_lowering` counts them).
fn names_matrix(stream: proc_macro2::TokenStream) -> bool {
    stream.into_iter().any(|tree| match tree {
        proc_macro2::TokenTree::Ident(ident) => {
            matches!(ident.unraw().to_string().as_str(), "Mat2" | "Mat3" | "Mat4")
        }
        proc_macro2::TokenTree::Group(group) => names_matrix(group.stream()),
        _ => false,
    })
}

/// The receiver of `signature` when it is the first parameter.
fn receiver_of(signature: &syn::Signature) -> Option<&syn::Receiver> {
    match signature.inputs.first() {
        Some(syn::FnArg::Receiver(receiver)) => Some(receiver),
        _ => None,
    }
}

/// Whether a receiver is `&mut self`.
fn receiver_is_ref_mut(receiver: &syn::Receiver) -> bool {
    receiver.reference.is_some() && receiver.mutability.is_some()
}

/// The name of the struct an inherent `impl` is for, when its type is a plain name of the module: `S` or `self::S`
/// (slice m9.a2, review m9.a1 N2). `None` for anything else (a foreign path, a generic type, a tuple).
pub(crate) fn impl_self_name(item: &syn::ItemImpl) -> Option<String> {
    let syn::Type::Path(own) = item.self_ty.as_ref() else {
        return None;
    };
    if own.qself.is_some() || own.path.leading_colon.is_some() {
        return None;
    }
    let segments = &own.path.segments;
    let last = segments.last()?;
    if !last.arguments.is_none() {
        return None;
    }
    match segments.len() {
        1 => {}
        2 if segments[0].ident == "self" && segments[0].arguments.is_none() => {}
        _ => return None,
    }
    Some(last.ident.unraw().to_string())
}

/// The members of every `impl` block of `items`.
fn method_table(items: &[syn::Item]) -> MethodTable {
    let mut table = MethodTable {
        matrix: items
            .iter()
            .any(|item| names_matrix(item.to_token_stream())),
        array_fields: crate::slang::array_field_lengths(items),
        ..MethodTable::default()
    };
    for item in items {
        if let syn::Item::Enum(item) = item {
            table.enums.insert(
                item.ident.unraw().to_string(),
                EnumFacts {
                    signed: enum_is_signed(&item.attrs),
                    variants: item
                        .variants
                        .iter()
                        .map(|variant| variant.ident.unraw().to_string())
                        .collect(),
                },
            );
        }
    }
    for item in items {
        if let syn::Item::Fn(function) = item
            && let syn::ReturnType::Type(_, ty) = &function.sig.output
        {
            if let Some(name) = enum_type_name(ty, &table.enums) {
                table
                    .enum_returns
                    .insert(function.sig.ident.to_string(), name);
            }
            if is_tuple_type(ty) {
                table
                    .tuple_returns
                    .insert(function.sig.ident.unraw().to_string());
            }
        }
    }
    for item in items {
        let syn::Item::Impl(block) = item else {
            continue;
        };
        let owner = impl_self_name(block).unwrap_or_default();
        for inner in &block.items {
            let syn::ImplItem::Fn(method) = inner else {
                continue;
            };
            let name = method.sig.ident.unraw().to_string();
            let receiver = receiver_of(&method.sig);
            table
                .members
                .insert(format!("{owner}::{name}"), receiver.is_some());
            if let syn::ReturnType::Type(_, ty) = &method.sig.output
                && is_tuple_type(ty)
            {
                table.tuple_returns.insert(name.clone());
            }
            if let Some(receiver) = receiver {
                let facts = table.by_name.entry(name).or_default();
                if receiver_is_ref_mut(receiver) {
                    facts.ref_mut = true;
                } else if receiver.reference.is_none() {
                    facts.by_value = true;
                }
            }
        }
    }
    table
}

/// Slang words that start a member declaration of a struct or name the enclosing type: a method of that name would be
/// a constructor, a subscript, a property or the type itself (GUST-E0328).
pub(crate) const SLANG_MEMBER_KEYWORDS: &[&str] =
    &["This", "init", "property", "subscript", "this"];

/// Why a method called `name` cannot be one (GUST-E0328), or `None`: the name is a method the dialect gives a meaning
/// to (the emitter and the validator read the name alone), a Slang member keyword, or a field of the same struct
/// (Slang has one namespace for the members of a struct).
pub(crate) fn method_name_clash(name: &str, fields: &HashSet<String>) -> Option<String> {
    let dialect = matches!(
        name,
        "global_id"
            | "len"
            | "is_some"
            | "is_none"
            | "is_ok"
            | "is_err"
            | "unwrap_or"
            | "unwrap"
            | "expect"
    ) || wrapping_arity(name).is_some()
        || math_types::is_swizzle(name)
        || math_types::method_arity(name).is_some()
        || texture::Method::from_name(name).is_some();
    if dialect {
        return Some(format!(
            "`{name}` is a method of the dialect, and a call of the name means that method: choose another name for this one"
        ));
    }
    if SLANG_MEMBER_KEYWORDS.contains(&name) {
        return Some(format!(
            "`{name}` is reserved by GUST as a method name (it starts a member declaration of a Slang struct, or names the struct itself): choose another name"
        ));
    }
    if fields.contains(name) {
        return Some(format!(
            "`{name}` is also a field of this struct, and a Slang struct has one namespace for the members: rename the method or the field"
        ));
    }
    None
}

/// What `validate_impl` reads of the module.
struct ImplContext<'a> {
    function_names: &'a HashSet<String>,
    result_helpers: &'a HashSet<String>,
    methods: &'a MethodTable,
    struct_fields: &'a HashMap<String, HashSet<String>>,
    /// The traits of the module with their method names (slice m9.g1).
    traits: &'a traits::Traits,
    /// The items of the module, for the checks of a trait impl that look at the other `impl` blocks.
    items: &'a [syn::Item],
}

/// The struct an `impl` block is for, while its methods are validated: `Self` is its name, and its fields are the
/// names that would capture an unqualified call inside the Slang struct.
struct Owner<'a> {
    name: &'a str,
    fields: &'a HashSet<String>,
}

/// Checks one `impl` block (slices m9.a1 and m9.a2): an inherent `impl S` (or `impl self::S`) of a struct of the module,
/// any number of blocks per struct, holding methods (`&self`, `&mut self`, `self`) and associated functions.
fn validate_impl(
    stage: &dyn StageProfile,
    item: &syn::ItemImpl,
    context: &ImplContext<'_>,
) -> syn::Result<()> {
    if item.unsafety.is_some() || item.defaultness.is_some() {
        return Err(diagnostics::error(
            Code::UnsupportedModuleItem,
            item,
            "an `impl` of a GPU module is a plain inherent `impl S { .. }`: not `unsafe` and not `default`",
        ));
    }
    // The attributes of the block: documentation and lint levels only, with the code of an attribute on a function
    // (review m9.a1 N4, m9.a2 N3): `#[cfg(..)] impl S` would let the Slang and the CPU twin disagree about which methods exist.
    for attribute in &item.attrs {
        if let Some(message) = impl_attribute_refusal(attribute) {
            return Err(diagnostics::error(
                Code::UnsupportedAttribute,
                attribute,
                message,
            ));
        }
    }
    // `impl T for S` of a trait the module declares (slice m9.g1); any other trait impl is refused here.
    let mut implemented = None;
    if let Some((negation, path, _)) = &item.trait_ {
        let blanket = item.generics.type_params().any(|parameter| {
            matches!(item.self_ty.as_ref(), syn::Type::Path(own) if own.qself.is_none() && own.path.is_ident(&parameter.ident))
        });
        if blanket {
            return Err(diagnostics::error(
                Code::ImplForeignType,
                item,
                "a blanket `impl` is not part of the dialect: implement a trait for a struct or a unit enum of this module",
            ));
        }
        let declared =
            traits::trait_path_name(path).filter(|name| context.traits.contains_key(name));
        match declared {
            Some(_) if negation.is_some() => {
                return Err(diagnostics::error(
                    Code::UnsupportedModuleItem,
                    item,
                    "a negative impl (`impl !T for S`) is not part of the dialect",
                ));
            }
            Some(name) => implemented = Some(name),
            None => {
                return Err(diagnostics::error(
                    Code::UnsupportedModuleItem,
                    path,
                    "a trait impl is for a trait declared in this `#[gpu]` module (`trait T { fn m(&self); }`); other traits (`Clone`, `Default`, operators) have no GPU lowering: write the methods in an inherent `impl S { .. }`",
                ));
            }
        }
    }
    if !item.generics.params.is_empty() || item.generics.where_clause.is_some() {
        return Err(diagnostics::error(
            Code::ExternOrGenericFunction,
            &item.generics,
            "generic and lifetime parameters on an `impl` are not supported yet",
        ));
    }
    let name = impl_self_name(item).unwrap_or_default();
    let no_fields = HashSet::new();
    let fields = match context.struct_fields.get(&name) {
        Some(fields) => fields,
        // A unit enum has no fields, and a trait is implemented for it (g0 P15); an inherent `impl` of an enum is not.
        None if implemented.is_some() && context.methods.enums.contains_key(&name) => &no_fields,
        None => {
            let message = if implemented.is_some() {
                "a trait impl is for a struct or a unit enum defined in this `#[gpu]` module; a type from elsewhere has no Slang definition to hold the methods"
            } else if context.methods.enums.contains_key(&name) {
                "methods on an enum are not supported: an inherent `impl` is for a struct defined in this `#[gpu]` module; write a free function that takes the enum, or implement a trait of the module for it"
            } else {
                "an `impl` is for a struct defined in this `#[gpu]` module; a type from elsewhere has no Slang definition to hold the methods"
            };
            return Err(diagnostics::error(
                Code::ImplForeignType,
                &item.self_ty,
                message,
            ));
        }
    };
    let owner = Owner {
        name: &name,
        fields,
    };
    if let Some(trait_name) = &implemented {
        check_trait_impl(item, trait_name, &name, context)?;
    }
    for inner in &item.items {
        let syn::ImplItem::Fn(method) = inner else {
            return Err(diagnostics::error(
                if implemented.is_some() {
                    Code::TraitFeature
                } else {
                    Code::UnsupportedModuleItem
                },
                inner,
                "an `impl` of a GPU module holds methods only: associated constants, types and macros are not accepted",
            ));
        };
        validate_method(stage, method, &owner, context)?;
    }
    Ok(())
}

/// The checks of an `impl Trait for S` that look at the whole module (slice m9.g1): the shape of its functions, the
/// P18 rule and the shared names. `owner_type` is the struct or enum `S`.
fn check_trait_impl(
    item: &syn::ItemImpl,
    trait_name: &str,
    owner_type: &str,
    context: &ImplContext<'_>,
) -> syn::Result<()> {
    let is_enum = context.methods.enums.contains_key(owner_type);
    // Every member of an inherent `impl` of the type, method or associated function: Slang has one namespace for them
    // (g0 P18: the same name in the struct and in an `extension` is accepted silently, and `s.m()` then reaches the
    // inherent one where Rust's `Trait::m(&s)` names the trait's; the same signature inside the struct is slangc E30201).
    let inherent: HashSet<String> = context
        .items
        .iter()
        .filter_map(|other| match other {
            syn::Item::Impl(block)
                if block.trait_.is_none()
                    && impl_self_name(block).is_some_and(|name| name == owner_type) =>
            {
                Some(block)
            }
            _ => None,
        })
        .flat_map(|block| &block.items)
        .filter_map(|inner| match inner {
            syn::ImplItem::Fn(method) => Some(method.sig.ident.unraw().to_string()),
            _ => None,
        })
        .collect();
    for inner in &item.items {
        let syn::ImplItem::Fn(method) = inner else {
            continue;
        };
        let name = method.sig.ident.unraw().to_string();
        let Some(receiver) = receiver_of(&method.sig) else {
            return Err(diagnostics::error(
                Code::TraitFeature,
                &method.sig,
                format!(
                    "`{name}` has no receiver: the functions of a trait impl are methods (`&self`, `&mut self` or `self`); an associated function belongs in an inherent `impl`"
                ),
            ));
        };
        if is_enum && receiver_is_ref_mut(receiver) {
            return Err(diagnostics::error(
                Code::TraitFeature,
                receiver,
                format!(
                    "`{name}` takes `&mut self` and `{owner_type}` is an enum: a method cannot assign its receiver (`*self = ..` has no lowering), so a `&mut self` method is for a struct; take `self` and return the new value"
                ),
            ));
        }
        if inherent.contains(&name) {
            return Err(diagnostics::error(
                Code::MethodNameClash,
                &method.sig.ident,
                format!(
                    "`{name}` is a method of the trait `{trait_name}` and also a member of an inherent `impl {owner_type}`: Rust's `{trait_name}::{name}(&value)` names the trait's method, but the lowering to `value.{name}()` would reach the inherent one in Slang; rename one of them"
                ),
            ));
        }
    }
    // Two traits that declare one method name, implemented by one type: the same reason.
    for other in context.items {
        let syn::Item::Impl(block) = other else {
            continue;
        };
        let Some((_, path, _)) = &block.trait_ else {
            continue;
        };
        if std::ptr::eq(block, item) || impl_self_name(block).is_none_or(|name| name != owner_type)
        {
            continue;
        }
        let Some(other_trait) = traits::trait_path_name(path) else {
            continue;
        };
        if other_trait == trait_name {
            continue;
        }
        if let Some((method, other)) =
            traits::shared_name_problem(context.traits, trait_name, &other_trait)
        {
            return Err(diagnostics::error(
                Code::MethodNameClash,
                path,
                format!(
                    "`{owner_type}` implements the traits `{trait_name}` and `{other}`, which both declare a method `{method}`: a call `{trait_name}::{method}(&value)` names one of them in Rust, and `value.{method}()` in Slang could not; rename the method in one trait"
                ),
            ));
        }
    }
    Ok(())
}

/// Checks one `trait` item (slice m9.g1): methods only, without a body, no supertraits, no generics, no associated
/// items; every method has a receiver, a name the dialect does not use, and a signature judged as a helper's.
fn validate_trait(
    stage: &dyn StageProfile,
    item: &syn::ItemTrait,
    context: &ImplContext<'_>,
) -> syn::Result<()> {
    let name = item.ident.unraw().to_string();
    if item.unsafety.is_some() || item.auto_token.is_some() || item.restriction.is_some() {
        return Err(diagnostics::error(
            Code::TraitFeature,
            item,
            "a trait of a GPU module is a plain `trait T { .. }`: not `unsafe`, not `auto`",
        ));
    }
    for attribute in &item.attrs {
        if let Some(message) = impl_attribute_refusal(attribute) {
            return Err(diagnostics::error(
                Code::UnsupportedAttribute,
                attribute,
                message,
            ));
        }
    }
    if math_types::is_reserved_type_name(&name) {
        return Err(diagnostics::error(
            Code::ReservedTypeName,
            &item.ident,
            format!(
                "`{name}` is a type the dialect maps to a Slang type itself, so a trait cannot be named that; choose another name"
            ),
        ));
    }
    if !item.generics.params.is_empty() || item.generics.where_clause.is_some() {
        return Err(diagnostics::error(
            Code::TraitFeature,
            &item.generics,
            "a generic trait (`trait T<X>`, a lifetime, a `where` clause) is not part of the dialect: a trait is a plain list of methods",
        ));
    }
    if !item.supertraits.is_empty() {
        return Err(diagnostics::error(
            Code::TraitFeature,
            &item.supertraits,
            "a supertrait (`trait T: U`) is not part of the dialect: declare the methods of both in one trait, or keep two traits",
        ));
    }
    for inner in &item.items {
        let syn::TraitItem::Fn(method) = inner else {
            return Err(diagnostics::error(
                Code::TraitFeature,
                inner,
                "a trait of a GPU module declares methods only: an associated type or constant is not part of the dialect",
            ));
        };
        if let Some(default) = &method.default {
            return Err(diagnostics::error(
                Code::TraitFeature,
                default,
                "a default method body is not part of the dialect: write the body in each `impl Trait for S`",
            ));
        }
        if let Some(attribute) = method.attrs.iter().find(|attribute| {
            attribute.path().is_ident("kernel")
                || attribute.path().is_ident("variant")
                || is_gpu_attribute(attribute)
        }) {
            return Err(diagnostics::error(
                Code::UnsupportedAttribute,
                attribute,
                "a method cannot be a kernel or a variant: kernels are free functions of the module",
            ));
        }
        let signature = &method.sig;
        let method_name = signature.ident.unraw().to_string();
        if !signature.generics.params.is_empty() || signature.generics.where_clause.is_some() {
            return Err(diagnostics::error(
                Code::TraitFeature,
                &signature.generics,
                "a generic trait method is not part of the dialect yet (generic helpers and methods are a later slice)",
            ));
        }
        if signature.asyncness.is_some()
            || signature.constness.is_some()
            || signature.unsafety.is_some()
            || signature.abi.is_some()
            || signature.variadic.is_some()
        {
            return Err(diagnostics::error(
                Code::TraitFeature,
                signature,
                "a trait method is a plain `fn m(&self, ..)`: not `async`, `const`, `unsafe` or `extern`",
            ));
        }
        let Some(receiver) = receiver_of(signature) else {
            return Err(diagnostics::error(
                Code::TraitFeature,
                signature,
                format!(
                    "`{method_name}` has no receiver: a trait of a GPU module declares methods (`&self`, `&mut self` or `self`); an associated function belongs in an inherent `impl`"
                ),
            ));
        };
        if receiver.colon_token.is_some() {
            return Err(diagnostics::error(
                Code::SelfParameter,
                receiver,
                "write the receiver as `self`, `&self` or `&mut self`: a typed `self: Type` receiver is not accepted",
            ));
        }
        if receiver.reference.is_none() && receiver.mutability.is_some() {
            return Err(diagnostics::error(
                Code::SelfParameter,
                receiver,
                "`mut self` is not accepted: a method that changes its receiver takes `&mut self`, and a method that takes `self` by value reads its copy only",
            ));
        }
        if let Some(message) = method_name_clash(&method_name, &HashSet::new()) {
            return Err(diagnostics::error(
                Code::MethodNameClash,
                &signature.ident,
                message,
            ));
        }
        // The signature as a helper's, without a body: the parameter and result types of the dialect.
        let stub = syn::ItemFn {
            attrs: method.attrs.clone(),
            vis: syn::Visibility::Inherited,
            sig: method_as_function_signature(signature),
            block: Box::new(syn::Block {
                brace_token: syn::token::Brace::default(),
                stmts: Vec::new(),
            }),
        };
        validate_function(
            stage,
            &stub,
            context.function_names,
            context.result_helpers,
            context.methods,
            Some(&Owner {
                name: "Self",
                fields: &HashSet::new(),
            }),
            false,
        )?;
    }
    Ok(())
}

/// The signature of a method without its receiver.
fn method_as_function_signature(signature: &syn::Signature) -> syn::Signature {
    let mut signature = signature.clone();
    signature.inputs = signature
        .inputs
        .into_iter()
        .filter(|argument| matches!(argument, syn::FnArg::Typed(_)))
        .collect();
    signature
}

/// Checks one method or associated function: the name (E0328), the receiver, the attributes, then the body and
/// signature as a helper's, with the receiver taken off (it is `this`).
fn validate_method(
    stage: &dyn StageProfile,
    method: &syn::ImplItemFn,
    owner: &Owner<'_>,
    context: &ImplContext<'_>,
) -> syn::Result<()> {
    if let Some(attribute) = method.attrs.iter().find(|attribute| {
        attribute.path().is_ident("kernel")
            || attribute.path().is_ident("variant")
            || is_gpu_attribute(attribute)
    }) {
        return Err(diagnostics::error(
            Code::UnsupportedAttribute,
            attribute,
            "a method cannot be a kernel or a variant: kernels are free functions of the module",
        ));
    }
    let name = method.sig.ident.unraw().to_string();
    if let Some(message) = method_name_clash(&name, owner.fields) {
        return Err(diagnostics::error(
            Code::MethodNameClash,
            &method.sig.ident,
            message,
        ));
    }
    // Inside a Slang struct an unqualified call finds the member before the free function of the same name (measured
    // on slangc 2026.21: the member captured the call), and a method may call a helper since slice m9.a2.
    if context
        .function_names
        .iter()
        .any(|helper| helper.trim_start_matches("r#") == name)
    {
        return Err(diagnostics::error(
            Code::MethodNameClash,
            &method.sig.ident,
            format!(
                "`{name}` is also the name of a helper function of this module, and inside the Slang struct a call of the name would reach the member instead of the helper: rename the method or the helper"
            ),
        ));
    }
    if let Some(receiver) = receiver_of(&method.sig) {
        if receiver.colon_token.is_some() {
            return Err(diagnostics::error(
                Code::SelfParameter,
                receiver,
                "write the receiver as `self`, `&self` or `&mut self`: a typed `self: Type` receiver is not accepted",
            ));
        }
        if receiver.reference.is_none() && receiver.mutability.is_some() {
            return Err(diagnostics::error(
                Code::SelfParameter,
                receiver,
                "`mut self` is not accepted: a method that changes its receiver takes `&mut self`, and a method that takes `self` by value reads its copy only",
            ));
        }
    }
    let function = method_as_function(method);
    validate_function(
        stage,
        &function,
        context.function_names,
        context.result_helpers,
        context.methods,
        Some(owner),
        false,
    )?;
    // In Rust `let c = self;` in a `&mut self` method binds the reference, and a write through `c` reaches the receiver; in Slang
    // it copies `this`, and the write is lost (review m9.a2 M2). A `self` or `&self` method copies in both languages.
    if let Some(receiver) = receiver_of(&method.sig)
        && receiver_is_ref_mut(receiver)
        && let Some(path) = bare_self_use(&method.block)
    {
        return Err(diagnostics::error(
            Code::SelfParameter,
            path,
            "`self` is used as a value in a `&mut self` method (`let c = self;`, `f(self)`, `return self`): Rust binds the reference there, and Slang copies the struct, so a write through the copy would be lost; write through `self.field` directly, or call a method on `self`",
        ));
    }
    Ok(())
}

/// The method as a free function without its receiver, which the validation of a helper then judges: value
/// parameters only, no `Workgroup`, no barrier or subgroup operation. Spans are the method's own.
fn method_as_function(method: &syn::ImplItemFn) -> syn::ItemFn {
    let mut signature = method.sig.clone();
    signature.inputs = signature
        .inputs
        .into_iter()
        .filter(|argument| matches!(argument, syn::FnArg::Typed(_)))
        .collect();
    syn::ItemFn {
        attrs: method.attrs.clone(),
        vis: method.vis.clone(),
        sig: signature,
        block: Box::new(method.block.clone()),
    }
}

/// The attributes a function (and an `impl` block) may carry: documentation and lints, `derive` and `kernel` (judged
/// elsewhere). Any other has no proven Rust-to-Slang meaning and cannot be dropped.
const ALLOWED_ATTRIBUTES: [&str; 7] =
    ["doc", "allow", "warn", "deny", "forbid", "derive", "kernel"];

/// Why `attribute` is refused, or `None` when it is on the allowlist.
fn attribute_refusal(attribute: &syn::Attribute) -> Option<&'static str> {
    (!ALLOWED_ATTRIBUTES
        .iter()
        .any(|name| attribute.path().is_ident(name)))
    .then_some("this attribute has no proven Rust-to-Slang meaning and cannot be dropped")
}

/// Why `attribute` is refused on an `impl` block: only documentation and lint levels are (`derive` and `kernel` are
/// errors in rustc there, and `cfg` would let the Slang and the twin disagree about the methods).
fn impl_attribute_refusal(attribute: &syn::Attribute) -> Option<&'static str> {
    (!["doc", "allow", "warn", "deny", "forbid"]
        .iter()
        .any(|name| attribute.path().is_ident(name)))
    .then_some("only documentation and lint attributes (`doc`, `allow`, `warn`, `deny`, `forbid`) are accepted on an `impl` block: any other has no proven Rust-to-Slang meaning and cannot be dropped")
}

/// Whether `place` holds an index expression anywhere (`a[i]`, `a[i].f`, `a.f[i]`).
fn place_has_index(place: &syn::Expr) -> bool {
    struct Scan(bool);
    impl<'a> Visit<'a> for Scan {
        fn visit_expr_index(&mut self, _: &'a syn::ExprIndex) {
            self.0 = true;
        }
    }
    let mut scan = Scan(false);
    scan.visit_expr(place);
    scan.0
}

/// The first use of `self` as a value in `block`: any `self` that is not the base of a field access (`self.n`) or the
/// receiver of a method call (`self.bump()`).
fn bare_self_use(block: &syn::Block) -> Option<&syn::ExprPath> {
    struct Scan<'a> {
        found: Option<&'a syn::ExprPath>,
    }
    fn is_self(expression: &syn::Expr) -> bool {
        match expression {
            syn::Expr::Paren(inner) => is_self(&inner.expr),
            syn::Expr::Path(path) => path.qself.is_none() && path.path.is_ident("self"),
            _ => false,
        }
    }
    impl<'a> Visit<'a> for Scan<'a> {
        fn visit_expr_field(&mut self, field: &'a syn::ExprField) {
            if !is_self(&field.base) {
                visit::visit_expr_field(self, field);
            }
        }
        fn visit_expr_method_call(&mut self, call: &'a syn::ExprMethodCall) {
            if !is_self(&call.receiver) {
                self.visit_expr(&call.receiver);
            }
            for argument in &call.args {
                self.visit_expr(argument);
            }
        }
        fn visit_expr_path(&mut self, path: &'a syn::ExprPath) {
            if self.found.is_none() && path.qself.is_none() && path.path.is_ident("self") {
                self.found = Some(path);
            }
        }
    }
    let mut scan = Scan { found: None };
    scan.visit_block(block);
    scan.found
}

fn validate_struct(item: &syn::ItemStruct) -> syn::Result<()> {
    let struct_name = item.ident.unraw().to_string();
    if math_types::is_reserved_type_name(&struct_name) {
        return Err(diagnostics::error(
            Code::ReservedTypeName,
            &item.ident,
            format!(
                "`{struct_name}` is a type the dialect maps to a Slang type itself, so a struct cannot be named that; choose another name"
            ),
        ));
    }
    if !item.generics.params.is_empty() || item.generics.where_clause.is_some() {
        return Err(diagnostics::error(
            Code::GenericStruct,
            &item.generics,
            "generic GPU structs are not supported yet",
        ));
    }
    let syn::Fields::Named(fields) = &item.fields else {
        return Err(diagnostics::error(
            Code::StructNeedsNamedFields,
            item,
            "GPU structs must have named fields",
        ));
    };
    if fields.named.is_empty() {
        return Err(diagnostics::error(
            Code::EmptyStruct,
            item,
            "GPU structs must contain at least one field",
        ));
    }
    if let Some(attribute) = item
        .attrs
        .iter()
        .find(|attribute| attribute.path().is_ident("repr"))
    {
        return Err(diagnostics::error(
            Code::ManualRepr,
            attribute,
            "GPU structs receive repr(C) automatically",
        ));
    }
    for attribute in item
        .attrs
        .iter()
        .filter(|attribute| attribute.path().is_ident("derive"))
    {
        let derives = diagnostics::parse_args_with(
            attribute,
            Code::MalformedDerive,
            syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
        )?;
        if derives.iter().any(|path| {
            path.segments
                .last()
                .is_some_and(|segment| segment.ident == "Clone" || segment.ident == "Copy")
        }) {
            return Err(diagnostics::error(
                Code::ManualCloneCopy,
                attribute,
                "GPU structs receive Clone and Copy automatically",
            ));
        }
    }
    let mut visitor = RestrictedVisitor::new(HashSet::new(), HashSet::new());
    visitor.option_forbidden = Some("GPU struct fields");
    visitor.result_forbidden = Some("GPU struct fields");
    visitor.visit_item_struct(item);
    visitor.finish()
}

/// The integer literal of an enum discriminant (`3`, `3u32`, `-2` with a signed repr), and whether it is negated.
fn enum_discriminant(expression: &syn::Expr, signed: bool) -> Option<()> {
    match expression {
        syn::Expr::Lit(literal) if matches!(literal.lit, syn::Lit::Int(_)) => Some(()),
        syn::Expr::Unary(unary)
            if signed
                && matches!(unary.op, syn::UnOp::Neg(_))
                && matches!(unary.expr.as_ref(), syn::Expr::Lit(literal) if matches!(literal.lit, syn::Lit::Int(_))) =>
        {
            Some(())
        }
        _ => None,
    }
}

/// Checks one `enum` of a GPU module (slice m9.e1, D157, GUST-E0122): field-less variants, integer-literal
/// discriminants, `repr(u32)` (the default) or `repr(i32)`, `PartialEq` derived by the user. The macro adds `Clone`,
/// `Copy` and the default `repr` itself, as it does for structs, so writing them is refused.
fn validate_enum(item: &syn::ItemEnum) -> syn::Result<()> {
    let name = item.ident.unraw().to_string();
    if math_types::is_reserved_type_name(&name) {
        return Err(diagnostics::error(
            Code::ReservedTypeName,
            &item.ident,
            format!(
                "`{name}` is a type the dialect maps to a Slang type itself, so an enum cannot be named that; choose another name"
            ),
        ));
    }
    if !item.generics.params.is_empty() || item.generics.where_clause.is_some() {
        return Err(diagnostics::error(
            Code::EnumShape,
            &item.generics,
            "generic enums are not supported: a GPU enum is a plain list of named integer values",
        ));
    }
    if item.variants.is_empty() {
        return Err(diagnostics::error(
            Code::EnumShape,
            item,
            "a GPU enum needs at least one variant",
        ));
    }
    let mut signed = false;
    let mut partial_eq = false;
    for attribute in &item.attrs {
        if attribute.path().is_ident("repr") {
            let reprs = diagnostics::parse_args_with(
                attribute,
                Code::EnumShape,
                syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
            )?;
            match reprs.iter().collect::<Vec<_>>().as_slice() {
                [only] if only.is_ident("u32") => {}
                [only] if only.is_ident("i32") => signed = true,
                _ => {
                    return Err(diagnostics::error(
                        Code::EnumShape,
                        attribute,
                        "a GPU enum is `#[repr(u32)]` (the default, written by the macro) or `#[repr(i32)]`: the Slang enum has a 32-bit integer type",
                    ));
                }
            }
        } else if attribute.path().is_ident("derive") {
            let derives = diagnostics::parse_args_with(
                attribute,
                Code::MalformedDerive,
                syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
            )?;
            for path in &derives {
                let last = path
                    .segments
                    .last()
                    .map(|segment| segment.ident.to_string());
                match last.as_deref() {
                    Some("Clone" | "Copy") => {
                        return Err(diagnostics::error(
                            Code::EnumShape,
                            attribute,
                            "GPU enums receive Clone and Copy automatically",
                        ));
                    }
                    Some("PartialEq") => partial_eq = true,
                    _ => {}
                }
            }
        } else if let Some(message) = attribute_refusal(attribute) {
            return Err(diagnostics::error(
                Code::UnsupportedAttribute,
                attribute,
                message,
            ));
        }
    }
    if !partial_eq {
        return Err(diagnostics::error(
            Code::EnumShape,
            &item.ident,
            format!(
                "`{name}` needs `#[derive(PartialEq, Eq)]`: `==`, `!=` and `match` compare enum values, and the CPU twin must be able to do the same"
            ),
        ));
    }
    for variant in &item.variants {
        if !matches!(variant.fields, syn::Fields::Unit) {
            return Err(diagnostics::error(
                Code::EnumShape,
                variant,
                "a GPU enum variant carries no data: data-carrying variants have no Slang lowering yet",
            ));
        }
        if let Some(attribute) = variant.attrs.iter().find(|attribute| {
            !["doc", "allow", "warn", "deny", "forbid"]
                .iter()
                .any(|name| attribute.path().is_ident(name))
        }) {
            return Err(diagnostics::error(
                Code::EnumShape,
                attribute,
                "a variant of a GPU enum takes documentation and lint attributes only",
            ));
        }
        if let Some((_, discriminant)) = &variant.discriminant
            && enum_discriminant(discriminant, signed).is_none()
        {
            return Err(diagnostics::error(
                Code::EnumShape,
                discriminant,
                "a discriminant is an integer literal (`B = 3`; a negative one needs `#[repr(i32)]`): a computed value would have to be evaluated by both Rust and Slang",
            ));
        }
    }
    // The names and the discriminant expressions go through the generic checks (reserved prefixes); `repr` is judged
    // above and is not on the attribute allowlist of the visitor.
    let mut clean = item.clone();
    clean
        .attrs
        .retain(|attribute| !attribute.path().is_ident("repr"));
    let mut visitor = RestrictedVisitor::new(HashSet::new(), HashSet::new());
    visitor.visit_item_enum(&clean);
    visitor.finish()
}

/// GUST-E0211: an enum is a value of the invocation, never a stored one. No `GpuPod` exists for a type with invalid bit
/// patterns, so a struct field, a buffer or uniform element (or a `Workgroup` element) naming an enum is refused.
fn check_enum_layout(
    stage: &dyn StageProfile,
    items: &[syn::Item],
    enums: &HashMap<String, EnumFacts>,
) -> syn::Result<()> {
    use crate::slang::{ParameterFlavor, parameter_flavor_in};

    if enums.is_empty() {
        return Ok(());
    }
    /// The first plain path to an enum anywhere inside a type.
    struct Mention<'e> {
        enums: &'e HashMap<String, EnumFacts>,
        found: Option<String>,
    }
    impl<'ast> Visit<'ast> for Mention<'_> {
        fn visit_type_path(&mut self, ty: &'ast syn::TypePath) {
            if let Some(name) = ty.path.get_ident().map(|ident| ident.unraw().to_string())
                && self.enums.contains_key(&name)
            {
                self.found.get_or_insert(name);
            }
            visit::visit_type_path(self, ty);
        }
    }
    let mention = |ty: &syn::Type| {
        let mut scan = Mention { enums, found: None };
        scan.visit_type(ty);
        scan.found
    };
    for item in items {
        match item {
            syn::Item::Struct(item) => {
                for field in &item.fields {
                    if let Some(name) = mention(&field.ty) {
                        return Err(diagnostics::error(
                            Code::EnumInResource,
                            &field.ty,
                            format!(
                                "`{name}` is an enum, and an enum is not a field of a struct: it has no `GpuPod` layout (a stored value could be none of its variants); keep the enum in a local or a parameter and store its integer (`mode as u32`)"
                            ),
                        ));
                    }
                }
            }
            syn::Item::Fn(function)
                if function
                    .attrs
                    .iter()
                    .any(|attribute| attribute.path().is_ident("kernel")) =>
            {
                for argument in &function.sig.inputs {
                    let syn::FnArg::Typed(argument) = argument else {
                        continue;
                    };
                    if matches!(
                        parameter_flavor_in(stage, &argument.ty),
                        ParameterFlavor::StorageRead
                            | ParameterFlavor::StorageReadWrite
                            | ParameterFlavor::Uniform
                            | ParameterFlavor::Workgroup
                    ) && let Some(name) = mention(&argument.ty)
                    {
                        return Err(diagnostics::error(
                            Code::EnumInResource,
                            &argument.ty,
                            format!(
                                "`{name}` is an enum, and an enum is not the element of a buffer, a uniform or workgroup memory: it has no `GpuPod` layout; store its integer (`mode as u32`) and match on it where it is used"
                            ),
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// GUST-E0212: a tuple is a value of the invocation, never a stored one. Rust gives a tuple no layout (the order and
/// the padding of its elements are unspecified), so no `GpuPod` can describe one: a struct field, a buffer or uniform
/// element and a `Workgroup` element naming a tuple anywhere inside the type are refused. A struct without named fields
/// is left to GUST-E0104.
fn check_tuple_layout(stage: &dyn StageProfile, items: &[syn::Item]) -> syn::Result<()> {
    use crate::slang::{ParameterFlavor, parameter_flavor_in};

    /// Whether a type holds a tuple with elements anywhere inside it.
    struct Mention(bool);
    impl<'ast> Visit<'ast> for Mention {
        fn visit_type_tuple(&mut self, ty: &'ast syn::TypeTuple) {
            self.0 |= !ty.elems.is_empty();
            visit::visit_type_tuple(self, ty);
        }
    }
    let mentions = |ty: &syn::Type| {
        let mut scan = Mention(false);
        scan.visit_type(ty);
        scan.0
    };
    for item in items {
        match item {
            syn::Item::Struct(item) if matches!(item.fields, syn::Fields::Named(_)) => {
                for field in &item.fields {
                    if mentions(&field.ty) {
                        return Err(diagnostics::error(
                            Code::TupleInResource,
                            &field.ty,
                            "a tuple is not a field of a struct: Rust gives a tuple no layout, so it has no `GpuPod` description; keep the tuple in a local, a parameter or a result and store its elements as fields of their own",
                        ));
                    }
                }
            }
            syn::Item::Fn(function)
                if function
                    .attrs
                    .iter()
                    .any(|attribute| attribute.path().is_ident("kernel")) =>
            {
                for argument in &function.sig.inputs {
                    let syn::FnArg::Typed(argument) = argument else {
                        continue;
                    };
                    if matches!(
                        parameter_flavor_in(stage, &argument.ty),
                        ParameterFlavor::StorageRead
                            | ParameterFlavor::StorageReadWrite
                            | ParameterFlavor::Uniform
                            | ParameterFlavor::Workgroup
                    ) && mentions(&argument.ty)
                    {
                        return Err(diagnostics::error(
                            Code::TupleInResource,
                            &argument.ty,
                            "a tuple is not the element of a buffer, a uniform or workgroup memory: Rust gives a tuple no layout, so it has no `GpuPod` description; use a struct with named fields",
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// The lengths a stored array may have: the ones `gust::GpuPod` is implemented for (D159).
const STORED_ARRAY_LENGTHS: &str = "1 to 32 and 64, 128, 256, 512, 1024, 2048 and 4096";

/// Whether `[T; length]` has a `GpuPod` impl (D159): 1 to 32, and the powers of two from 64 to 4096.
fn stored_array_length_ok(length: u32) -> bool {
    (1..=32).contains(&length) || ((64..=4096).contains(&length) && length.is_power_of_two())
}

/// The outermost array type inside `ty`, if there is one.
fn first_array(ty: &syn::Type) -> Option<&syn::TypeArray> {
    struct First<'ast>(Option<&'ast syn::TypeArray>);
    impl<'ast> Visit<'ast> for First<'ast> {
        fn visit_type_array(&mut self, ty: &'ast syn::TypeArray) {
            self.0.get_or_insert(ty);
        }
    }
    let mut scan = First(None);
    scan.visit_type(ty);
    scan.0
}

/// GUST-E0214, GUST-E0216 and GUST-E0217 for an array that is stored: a struct field or the element of a buffer. A
/// stored array has the layout of `GpuPod for [T; N]` (D159), so `N` is one of [`STORED_ARRAY_LENGTHS`] and the element
/// is one whose std430 stride is its Rust size. `place` says where the array stands, for the message.
fn check_stored_array(array: &syn::TypeArray, place: &str) -> syn::Result<()> {
    let Some(length) = crate::slang::array_length_literal(&array.len) else {
        return Err(diagnostics::error(
            Code::ArrayLength,
            &array.len,
            "an array length is an integer literal greater than zero (`[f32; 4]`): not `0`, a constant or an expression",
        ));
    };
    if !stored_array_length_ok(length) {
        return Err(diagnostics::error(
            Code::ArrayLength,
            &array.len,
            format!(
                "an array that is {place} has one of the lengths {STORED_ARRAY_LENGTHS}, the ones `GpuPod` is implemented for, and `{length}` is none of them: use one of those lengths, or keep the longer array in a buffer of its own elements (a local array may have any length above zero)"
            ),
        ));
    }
    if let syn::Type::Path(path) = strip_type_parens(&array.elem)
        && let Some(ident) = path.path.get_ident()
        && let Some(math) = math_types::math_type(&ident.unraw().to_string())
        && !math.buffer
    {
        return Err(diagnostics::error(
            Code::ArrayStride,
            &array.elem,
            format!(
                "`{}` cannot be the element of an array that is {place}: its stride in a buffer differs from its size in Rust, so no Rust array matches the buffer; use the 4-wide type, or one array per component",
                math.name
            ),
        ));
    }
    if let Some(problem) = array_element_problem(&array.elem) {
        return Err(diagnostics::error(Code::ArrayElement, &array.elem, problem));
    }
    Ok(())
}

/// Arrays that are stored (slice m9.r2a, D159): a struct field, a buffer element, and never a uniform. GUST-E0215 for a
/// uniform element that is or holds an array and for a field array of a struct a uniform reaches (std140 would round
/// the stride of the array up to 16; the scope choice of D159, not a limit of Slang); GUST-E0216, GUST-E0217 and
/// GUST-E0214 through [`check_stored_array`] for the rest. `Workgroup<[T; N]>` is its own shape (GUST-E0210), so it is
/// left alone: a nested array or a struct element stays refused there.
fn check_array_layout(stage: &dyn StageProfile, items: &[syn::Item]) -> syn::Result<()> {
    use crate::slang::{ParameterFlavor, parameter_flavor_in};

    let kernels = items.iter().filter_map(|item| match item {
        syn::Item::Fn(function) => Some(function),
        _ => None,
    });
    let mut resources: Vec<(ParameterFlavor, &syn::Type)> = Vec::new();
    for function in kernels {
        if crate::expand::kernel_attribute_in(stage, function)?.is_none() {
            continue;
        }
        for argument in &function.sig.inputs {
            let syn::FnArg::Typed(argument) = argument else {
                continue;
            };
            let flavor = parameter_flavor_in(stage, &argument.ty);
            if matches!(
                flavor,
                ParameterFlavor::StorageRead
                    | ParameterFlavor::StorageReadWrite
                    | ParameterFlavor::Uniform
            ) && let Ok(element) = crate::slang::wrapper_element_type(&argument.ty)
            {
                resources.push((flavor, element));
            }
        }
    }
    for &(flavor, element) in &resources {
        if flavor == ParameterFlavor::Uniform
            && let Some(array) = first_array(element)
        {
            return Err(diagnostics::error(
                Code::ArrayInUniform,
                array,
                "an array is not part of a uniform: a constant buffer lays an array out with std140 rules, which no `GpuPod` array promises, and arrays are storage-buffer types in this release; put the array in a storage buffer (`StructuredBuffer<[T; N]>`), or pass its elements as fields",
            ));
        }
    }
    let uniform_structs = crate::uniform::reach(stage, items)?.uniform;
    for item in items {
        let syn::Item::Struct(item) = item else {
            continue;
        };
        if !matches!(item.fields, syn::Fields::Named(_)) {
            continue;
        }
        let uniform = uniform_structs.contains(&item.ident.unraw().to_string());
        for field in &item.fields {
            let Some(array) = first_array(&field.ty) else {
                continue;
            };
            if uniform {
                return Err(diagnostics::error(
                    Code::ArrayInUniform,
                    array,
                    format!(
                        "`{}` is reached from a uniform parameter, and an array is not part of a uniform: a constant buffer lays an array out with std140 rules, which no `GpuPod` array promises; keep the array in a struct that only storage buffers use, or give the uniform one field per element",
                        item.ident
                    ),
                ));
            }
            check_stored_array(array, "a struct field")?;
        }
    }
    for &(flavor, element) in &resources {
        if flavor != ParameterFlavor::Uniform
            && let Some(array) = first_array(element)
        {
            check_stored_array(array, "the element of a buffer")?;
        }
    }
    Ok(())
}

fn validate_function(
    stage: &dyn StageProfile,
    function: &syn::ItemFn,
    function_names: &HashSet<String>,
    result_helpers: &HashSet<String>,
    methods: &MethodTable,
    owner: Option<&Owner<'_>>,
    closure: bool,
) -> syn::Result<()> {
    use crate::slang::{ParameterFlavor, parameter_flavor_in};

    // The word the messages use for a function that is not a kernel: a closure is judged as a helper (slice m9.c).
    let noun = if closure { "closure" } else { "helper" };
    // What the profile makes of each parameter (a receiver is none; `validate_function` rejects it later).
    let flavor_of = |argument: &syn::FnArg| match argument {
        syn::FnArg::Typed(argument) => Some(parameter_flavor_in(stage, &argument.ty)),
        syn::FnArg::Receiver(_) => None,
    };
    let is_kernel = crate::expand::kernel_attribute_in(stage, function)?.is_some();
    // The names of the subgroup operations are the dialect's: a call of one is that operation (the emitter and the
    // classifier read the name), so a function of the module cannot take the name (m8.s, D140).
    if barrier::subgroup_arity(&function.sig.ident.to_string()).is_some() {
        return Err(diagnostics::error(
            Code::SubgroupNotAllowed,
            &function.sig.ident,
            format!(
                "`{}` is the name of a subgroup operation; a function of a GPU module cannot take it, because a call \
                 of the name is that operation: choose another name",
                function.sig.ident
            ),
        ));
    }
    if !is_kernel
        && let Some(argument) = function
            .sig
            .inputs
            .iter()
            .find(|argument| flavor_of(argument) == Some(ParameterFlavor::Workgroup))
    {
        return Err(diagnostics::error(
            Code::WorkgroupParameterOnHelper,
            argument,
            format!(
                "`Workgroup<[T; N]>` memory belongs to a kernel: a {noun} cannot take it as a parameter; declare the \
                 array on the kernel and pass the {noun} the values it needs"
            ),
        ));
    }
    let invocations = function
        .sig
        .inputs
        .iter()
        .filter(|argument| flavor_of(argument) == Some(ParameterFlavor::Invocation))
        .count();
    if is_kernel {
        if invocations != 1 {
            return Err(diagnostics::error(
                Code::KernelDispatchParameter,
                &function.sig,
                stage.binding_rules().missing_dispatch_validate,
            ));
        }
        if !matches!(&function.sig.output, syn::ReturnType::Default)
            && !matches!(&function.sig.output, syn::ReturnType::Type(_, ty) if matches!(ty.as_ref(), syn::Type::Tuple(tuple) if tuple.elems.is_empty()))
        {
            return Err(diagnostics::error(
                Code::KernelReturnsUnit,
                &function.sig.output,
                "GPU kernels must return unit",
            ));
        }
    } else if function.sig.inputs.iter().any(|argument| {
        matches!(
            flavor_of(argument),
            Some(
                ParameterFlavor::StorageRead
                    | ParameterFlavor::StorageReadWrite
                    | ParameterFlavor::Uniform
                    | ParameterFlavor::SampledTexture
                    | ParameterFlavor::StorageTexture
                    | ParameterFlavor::WriteTexture
                    | ParameterFlavor::Sampler
            )
        )
    }) {
        return Err(diagnostics::error(
            Code::HelperResourceParameter,
            &function.sig,
            if closure {
                "a closure takes value parameters only (a buffer, uniform, texture or sampler is a name of the kernel, \
                 which a closure cannot capture); pass the values it needs"
            } else {
                "GPU helper functions currently accept value parameters only; access resources in the kernel"
            },
        ));
    }
    if is_kernel {
        // The constructs the shape refuses, judged from what the kernel declares (the profile never reads the body).
        let mut checks = BodyChecks {
            workgroup_memory: function
                .sig
                .inputs
                .iter()
                .any(|argument| flavor_of(argument) == Some(ParameterFlavor::Workgroup)),
            barrier: barrier::has_barrier(function),
            atomics: !atomic_buffer_names(function).is_empty(),
            forbidden: Vec::new(),
        };
        stage.validate_body(&mut checks);
        if let Some(sentence) = checks.forbidden.into_iter().next() {
            // A construct the kernel's shape does not take (GUST-E0729); the profile supplies the sentence.
            return Err(diagnostics::error(
                Code::ProfileRefusal,
                &function.sig,
                sentence,
            ));
        }
    }
    if function.sig.asyncness.is_some() {
        return Err(diagnostics::error(
            Code::AsyncFunction,
            &function.sig,
            "async is CPU-only in GPU modules",
        ));
    }
    if function.sig.unsafety.is_some() {
        return Err(diagnostics::error(
            Code::UnsafeFunction,
            &function.sig,
            "unsafe is not part of the Rust-to-Slang subset",
        ));
    }
    if function.sig.abi.is_some() || !function.sig.generics.params.is_empty() {
        return Err(diagnostics::error(
            Code::ExternOrGenericFunction,
            &function.sig,
            "extern and generic GPU functions are not supported yet",
        ));
    }

    let mut visitor = RestrictedVisitor::new(function_names.clone(), result_helpers.clone());
    visitor.methods = methods.clone();
    if let Some(owner) = owner {
        visitor.self_struct = Some(owner.name.to_owned());
        visitor.owner_fields = owner.fields.clone();
    }
    visitor.atomic_buffers = atomic_buffer_names(function);
    visitor.array_lens = crate::slang::array_lengths(&function.sig.inputs, &function.block);
    for argument in &function.sig.inputs {
        if let syn::FnArg::Typed(argument) = argument
            && let syn::Pat::Ident(name) = argument.pat.as_ref()
        {
            if parameter_flavor_in(stage, &argument.ty) == ParameterFlavor::Value {
                visitor.value_params.insert(name.ident.unraw().to_string());
                if let Some(enum_name) = enum_type_name(&argument.ty, &methods.enums) {
                    visitor
                        .enum_values
                        .insert(name.ident.unraw().to_string(), enum_name);
                }
                if is_tuple_type(&argument.ty) {
                    visitor.tuple_values.insert(name.ident.unraw().to_string());
                }
            }
            match parameter_flavor_in(stage, &argument.ty) {
                ParameterFlavor::Invocation => {
                    visitor.invocations.insert(name.ident.to_string());
                }
                ParameterFlavor::LocalId | ParameterFlavor::GroupId => {
                    visitor.derived_ids.insert(name.ident.to_string());
                }
                ParameterFlavor::Workgroup => {
                    visitor.derived_ids.insert(name.ident.to_string());
                    if let syn::Type::Path(path) = argument.ty.as_ref() {
                        visitor
                            .workgroup_types
                            .insert(std::ptr::from_ref(path) as usize);
                    }
                    if let Err(message) = workgroup_array(&argument.ty) {
                        visitor.reject(Code::WorkgroupShape, &argument.ty, &message);
                    }
                }
                ParameterFlavor::StorageRead => {
                    visitor.resources.insert(name.ident.to_string());
                }
                ParameterFlavor::Uniform => {
                    visitor.uniforms.insert(name.ident.to_string());
                    // A constant buffer is read-only: a `mut` binding would let the twin's copy be written.
                    if name.mutability.is_some() {
                        visitor.reject(
                            Code::UniformWholeValue,
                            name,
                            "a uniform parameter is read-only: declare it without `mut`",
                        );
                    }
                }
                ParameterFlavor::StorageReadWrite => {
                    visitor.resources.insert(name.ident.to_string());
                    if let Some(element) = rw_buffer_element_type(&argument.ty) {
                        visitor
                            .read_write_buffers
                            .insert(name.ident.to_string(), element);
                    }
                }
                flavor @ (ParameterFlavor::SampledTexture
                | ParameterFlavor::StorageTexture
                | ParameterFlavor::WriteTexture) => {
                    let name = name.ident.to_string();
                    match texture::texture_spec(flavor, &argument.ty) {
                        Ok(spec) => {
                            visitor.textures.insert(
                                name,
                                TextureFacts {
                                    access: spec.access,
                                    filterable: spec.filterable(),
                                },
                            );
                        }
                        Err(message) => {
                            visitor.textures.insert(
                                name,
                                TextureFacts {
                                    access: texture::Access::Sampled,
                                    filterable: false,
                                },
                            );
                            visitor.reject(Code::TextureType, &argument.ty, &message);
                        }
                    }
                    if let syn::Type::Path(path) = argument.ty.as_ref() {
                        visitor
                            .texture_types
                            .insert(std::ptr::from_ref(path) as usize);
                    }
                }
                ParameterFlavor::Sampler => {
                    visitor.samplers.insert(name.ident.to_string());
                    if !texture::sampler_has_no_arguments(&argument.ty) {
                        visitor.reject(
                            Code::TextureType,
                            &argument.ty,
                            "`Sampler` takes no type arguments",
                        );
                    }
                    if let syn::Type::Path(path) = argument.ty.as_ref() {
                        visitor
                            .texture_types
                            .insert(std::ptr::from_ref(path) as usize);
                    }
                }
                ParameterFlavor::Value if matches!(argument.ty.as_ref(), syn::Type::Path(path) if result_payload(path).is_some()) =>
                {
                    visitor.result_values.insert(name.ident.to_string());
                }
                _ => {}
            }
        }
    }
    if !matches!(function.sig.output, syn::ReturnType::Default)
        && let Some(syn::Stmt::Expr(syn::Expr::Match(tail), None)) = function.block.stmts.last()
    {
        visitor.body_tail_match = Some(std::ptr::from_ref(tail) as usize);
    }
    visitor.visit_item_fn(function);
    visitor.finish()?;
    // The closures bound in the body, once nothing else in it was found (slice m9.c).
    validate_closures(
        stage,
        function,
        function_names,
        result_helpers,
        methods,
        owner,
    )?;
    check_workgroup_rules(stage, function, is_kernel, closure)
}

/// The workgroup rules of a function the generic validation accepted (see `barrier`): the barrier positions (E0414),
/// no `return` (E0415) and no `break` or `continue` in a barrier loop (E0416), then the CPU strategy (E0717) and the
/// memory limit (E0718) of a kernel that uses workgroup memory or a barrier.
fn check_workgroup_rules(
    stage: &dyn StageProfile,
    function: &syn::ItemFn,
    is_kernel: bool,
    closure: bool,
) -> syn::Result<()> {
    use crate::slang::{ParameterFlavor, parameter_flavor_in};

    let has_memory = function.sig.inputs.iter().any(|argument| {
        matches!(argument, syn::FnArg::Typed(argument)
            if parameter_flavor_in(stage, &argument.ty) == ParameterFlavor::Workgroup)
    });
    // A kernel that calls a subgroup operation is a barrier kernel for the early exits and the jumps (E0415, E0416): an
    // invocation that skipped the call would leave the others of its subgroup with a wrong value.
    let uses_subgroups = is_kernel && barrier::has_subgroup_call(function);
    let barrier_kernel =
        is_kernel && (has_memory || barrier::has_barrier(function) || uses_subgroups);
    barrier::check(
        function,
        is_kernel,
        barrier_kernel,
        barrier::requires_subgroups(function),
        closure,
    )?;
    // Full subgroups only (E0751): the invocation count is a multiple of 64, which every subgroup size of the
    // certified adapters (4, 8, 32, 64) divides. `Error::SubgroupSize` is the run-time check of the same rule.
    if uses_subgroups
        && let Some((options, _)) = crate::expand::kernel_attribute_in(stage, function)?
    {
        let [x, y, z] = options.workgroup_size;
        // In `u128`: three `u32` dimensions multiply to under 2^96, whereas `workgroup_size` has no upper bound and the
        // product in `u64` overflowed (and panicked the macro) for dimensions near `u32::MAX`.
        let invocations = u128::from(x) * u128::from(y) * u128::from(z);
        if !invocations.is_multiple_of(64) {
            return Err(diagnostics::error(
                Code::SubgroupInvocations,
                &function.sig,
                format!(
                    "kernel `{}` calls subgroup operations, so its workgroup runs whole subgroups only: the invocation \
                     count must be a multiple of 64, and `workgroup_size({x}, {y}, {z})` has {invocations}",
                    function.sig.ident
                ),
            ));
        }
    }
    if !(has_memory || barrier::has_barrier(function)) {
        return Ok(());
    }
    if !barrier_kernel {
        return Ok(());
    }
    let bytes = barrier::workgroup_bytes(function);
    if bytes > barrier::WORKGROUP_MEMORY_LIMIT {
        return Err(diagnostics::error(
            Code::WorkgroupMemoryLimit,
            &function.sig,
            format!(
                "the `Workgroup` arrays of this kernel take {bytes} bytes, over the limit of {} bytes (16 KiB, the \
                 WebGPU default): a scalar or `bool` element takes 4 bytes, a 2-wide vector 8 and a 3-wide or \
                 4-wide one 16; shorten the arrays",
                barrier::WORKGROUP_MEMORY_LIMIT
            ),
        ));
    }
    let twin = crate::expand::kernel_attribute_in(stage, function)?.map(|(_, twin)| twin);
    if !matches!(twin, Some(crate::expand::CpuTwin::Unsupported)) {
        let construct = if has_memory {
            "`Workgroup<[T; N]>` memory"
        } else {
            "`workgroup_barrier()`"
        };
        return Err(diagnostics::error(
            Code::BarrierKernelCpu,
            &function.sig,
            format!(
                "kernel `{}` uses {construct}, which has no CPU twin yet: say so with `cpu = \"unsupported\"`, as in \
                 `#[kernel(workgroup_size(256, 1, 1), cpu = \"unsupported\")]`",
                function.sig.ident
            ),
        ));
    }
    Ok(())
}

/// Names the dialect gives a meaning to as a callee or a value: a closure cannot take one (GUST-E0328), and a body that
/// uses one has not captured anything (slice m9.c).
fn is_dialect_callee(name: &str) -> bool {
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

/// The identifiers a closure body uses that nothing inside it binds (slice m9.c): its captures, in first-use order.
///
/// A walk with a stack of scopes. The parameters bind in the outermost one; a `let` binds after its initializer, a
/// `for` variable, an `if let` and a `match` arm bind inside their body, and a block ends its scope. A single identifier
/// in expression position (a callee, a receiver, a field shorthand `P { x }`) that is not bound is free, and free is a
/// capture, except as the callee of a call, where a helper of the module and a name of the dialect (`Some`, `Ok`, `Err`, the
/// barrier, the subgroup and atomic operations, judged by their own rules) are not captures, and `None` as a value. A
/// local that shadows a helper name is a capture when it is used as a value. A module holds functions, structs, enums and `impl`s only, so
/// every other free identifier names a local, a parameter, `self` or a closure of the surrounding function. Paths of two
/// segments (`Vec2::new`, `E::A`, `S::f`, `Mat4::IDENTITY`) and the paths of struct literals and types are items.
struct CaptureScan<'a> {
    scopes: Vec<HashSet<String>>,
    helpers: &'a HashSet<String>,
    free: Vec<String>,
}

impl CaptureScan<'_> {
    fn bind(&mut self, pattern: &syn::Pat) {
        struct Names(Vec<String>);
        impl<'p> Visit<'p> for Names {
            fn visit_pat_ident(&mut self, pattern: &'p syn::PatIdent) {
                self.0.push(pattern.ident.unraw().to_string());
                visit::visit_pat_ident(self, pattern);
            }
        }
        let mut names = Names(Vec::new());
        names.visit_pat(pattern);
        if let Some(scope) = self.scopes.last_mut() {
            scope.extend(names.0);
        }
    }

    /// A use of the single identifier `name`: a callee when it is the function of a call. A helper and a name of the dialect
    /// are exempt only as callees (a local may shadow a helper name, and then the value use is a capture); `None` is the one
    /// dialect name that is a value, and no `let` can bind it.
    fn mention(&mut self, name: String, callee: bool) {
        let exempt = if callee {
            is_dialect_callee(&name)
                || self.helpers.contains(&name)
                || self.helpers.contains(&format!("r#{name}"))
        } else {
            name == "None"
        };
        if exempt
            || self.scopes.iter().any(|scope| scope.contains(&name))
            || self.free.contains(&name)
        {
            return;
        }
        self.free.push(name);
    }
}

impl<'ast> Visit<'ast> for CaptureScan<'_> {
    fn visit_type(&mut self, _: &'ast syn::Type) {}

    fn visit_expr_macro(&mut self, _: &'ast syn::ExprMacro) {}

    fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {}

    fn visit_block(&mut self, block: &'ast syn::Block) {
        self.scopes.push(HashSet::new());
        visit::visit_block(self, block);
        self.scopes.pop();
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let Some(initial) = &local.init {
            self.visit_expr(&initial.expr);
            if let Some((_, diverge)) = &initial.diverge {
                self.visit_expr(diverge);
            }
        }
        self.bind(&local.pat);
    }

    fn visit_expr_for_loop(&mut self, expression: &'ast syn::ExprForLoop) {
        self.visit_expr(&expression.expr);
        self.scopes.push(HashSet::new());
        self.bind(&expression.pat);
        self.visit_block(&expression.body);
        self.scopes.pop();
    }

    fn visit_expr_if(&mut self, expression: &'ast syn::ExprIf) {
        let syn::Expr::Let(binding) = expression.cond.as_ref() else {
            visit::visit_expr_if(self, expression);
            return;
        };
        self.visit_expr(&binding.expr);
        self.scopes.push(HashSet::new());
        self.bind(&binding.pat);
        self.visit_block(&expression.then_branch);
        self.scopes.pop();
        if let Some((_, alternative)) = &expression.else_branch {
            self.visit_expr(alternative);
        }
    }

    fn visit_expr_match(&mut self, expression: &'ast syn::ExprMatch) {
        self.visit_expr(&expression.expr);
        for arm in &expression.arms {
            self.scopes.push(HashSet::new());
            self.bind(&arm.pat);
            if let Some((_, guard)) = &arm.guard {
                self.visit_expr(guard);
            }
            self.visit_expr(&arm.body);
            self.scopes.pop();
        }
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        match call.func.as_ref() {
            syn::Expr::Path(path) if path.qself.is_none() => {
                if let Some(name) = path.path.get_ident() {
                    self.mention(name.unraw().to_string(), true);
                }
            }
            other => self.visit_expr(other),
        }
        for argument in &call.args {
            self.visit_expr(argument);
        }
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path.qself.is_none()
            && let Some(name) = path.path.get_ident()
        {
            self.mention(name.unraw().to_string(), false);
        }
    }
}

/// The captures of `closure`: the identifiers its body uses that it does not bind (see [`CaptureScan`]).
fn closure_captures(closure: &syn::ExprClosure, helpers: &HashSet<String>) -> Vec<String> {
    let mut scan = CaptureScan {
        scopes: vec![HashSet::new()],
        helpers,
        free: Vec::new(),
    };
    for input in &closure.inputs {
        scan.bind(input);
    }
    scan.visit_expr(&closure.body);
    scan.free
}

/// The closures bound in the body of `function` (slice m9.c). Each is judged on its own and reported once, at its `let`:
/// the form (GUST-E0314, `move` E0329), the binding (E0330), the signature (E0331), a closure inside it (E0332), the
/// name (E0901, E0328), the captures (E0329); then the body is validated as the body of a helper whose parameters are
/// the closure's (D154: the helper rule covers a closure body, so no closure holds a barrier, a subgroup operation,
/// workgroup memory, an atomic or a resource, and the call graph treats its calls as calls of the function that holds it).
fn validate_closures(
    stage: &dyn StageProfile,
    function: &syn::ItemFn,
    function_names: &HashSet<String>,
    result_helpers: &HashSet<String>,
    methods: &MethodTable,
    owner: Option<&Owner<'_>>,
) -> syn::Result<()> {
    let sites = crate::slang::closure_sites(&function.block);
    if sites.is_empty() {
        return Ok(());
    }
    // How many times each name is bound in the function (parameters, locals, loop variables, patterns, the bindings and
    // the parameters of the closures): a closure name that is bound twice shadows or is shadowed, and the validator
    // would have to track which meaning a call has.
    struct Bindings(HashMap<String, usize>);
    impl<'ast> Visit<'ast> for Bindings {
        fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
            *self.0.entry(pattern.ident.unraw().to_string()).or_default() += 1;
            visit::visit_pat_ident(self, pattern);
        }
    }
    let mut bindings = Bindings(HashMap::new());
    bindings.visit_item_fn(function);
    let closure_names: HashSet<String> = sites
        .iter()
        .map(|(_, pattern, _)| pattern.ident.unraw().to_string())
        .collect();
    // The owner of a closure body has no fields to clash with: the hoisted function is a free function.
    let no_fields = HashSet::new();
    let free_owner = owner.map(|owner| Owner {
        name: owner.name,
        fields: &no_fields,
    });

    let mut errors: Option<syn::Error> = None;
    for (local, pattern, closure) in sites {
        let result = validate_closure(
            stage,
            local,
            pattern,
            closure,
            &bindings.0,
            &closure_names,
            function_names,
            result_helpers,
            methods,
            free_owner.as_ref(),
        );
        if let Err(error) = result {
            match &mut errors {
                Some(existing) => existing.combine(error),
                None => errors = Some(error),
            }
        }
    }
    errors.map_or(Ok(()), Err)
}

#[allow(clippy::too_many_arguments)]
fn validate_closure(
    stage: &dyn StageProfile,
    local: &syn::Local,
    pattern: &syn::PatIdent,
    closure: &syn::ExprClosure,
    bindings: &HashMap<String, usize>,
    closure_names: &HashSet<String>,
    function_names: &HashSet<String>,
    result_helpers: &HashSet<String>,
    methods: &MethodTable,
    owner: Option<&Owner<'_>>,
) -> syn::Result<()> {
    let name = pattern.ident.unraw().to_string();
    // The form: what has no lowering at all.
    if closure.asyncness.is_some()
        || closure.constness.is_some()
        || closure.movability.is_some()
        || closure.lifetimes.is_some()
        || !closure.attrs.is_empty()
    {
        return Err(diagnostics::error(
            Code::Closure,
            local,
            "this closure form has no lowering: write `let f = |x: f32| -> f32 { .. };` without `async`, `static`, \
             `const`, a `for<..>` binder or an attribute",
        ));
    }
    if closure.capture.is_some() {
        return Err(diagnostics::error(
            Code::ClosureCapture,
            local,
            format!(
                "`move` makes the closure `{name}` own what it captures; a closure here captures nothing, so it takes \
                 everything it uses as a parameter: remove `move`"
            ),
        ));
    }
    // The binding: immutable, no annotation, a plain name.
    if pattern.mutability.is_some() {
        return Err(diagnostics::error(
            Code::ClosureValue,
            local,
            format!(
                "a closure binding is immutable: write `let {name} = |..| ..;`, not `let mut {name}`"
            ),
        ));
    }
    if matches!(local.pat, syn::Pat::Type(_))
        || pattern.by_ref.is_some()
        || pattern.subpat.is_some()
    {
        return Err(diagnostics::error(
            Code::ClosureValue,
            local,
            format!(
                "the closure `{name}` takes no type annotation on its binding: its parameters and result carry the types"
            ),
        ));
    }
    // The signature: typed parameters, an explicit non-unit result, a block body.
    for input in &closure.inputs {
        let typed = matches!(input, syn::Pat::Type(typed)
            if matches!(typed.pat.as_ref(), syn::Pat::Ident(parameter)
                if parameter.by_ref.is_none() && parameter.subpat.is_none()));
        if !typed {
            return Err(diagnostics::error(
                Code::ClosureSignature,
                input,
                "every parameter of a closure is `name: Type` (`|x: f32, y: f32| -> f32 { .. }`): the macro sees no types, \
                 so none is inferred",
            ));
        }
    }
    match &closure.output {
        syn::ReturnType::Default => {
            return Err(diagnostics::error(
                Code::ClosureSignature,
                closure,
                format!(
                    "the closure `{name}` needs an explicit result type: `|x: f32| -> f32 {{ .. }}`"
                ),
            ));
        }
        syn::ReturnType::Type(_, ty) if matches!(strip_type_parens(ty), syn::Type::Tuple(tuple) if tuple.elems.is_empty()) =>
        {
            return Err(diagnostics::error(
                Code::ClosureSignature,
                ty,
                "a closure gives a value, and a closure cannot change anything outside itself: a unit result has no use",
            ));
        }
        syn::ReturnType::Type(..) => {}
    }
    if !matches!(closure.body.as_ref(), syn::Expr::Block(body) if body.label.is_none()) {
        return Err(diagnostics::error(
            Code::ClosureSignature,
            &closure.body,
            "the body of a closure is a block: `|x: f32| -> f32 { x * 2.0f32 }`",
        ));
    }
    // A closure inside the body.
    struct Nested<'a>(Option<&'a syn::ExprClosure>);
    impl<'a> Visit<'a> for Nested<'a> {
        fn visit_expr_closure(&mut self, inner: &'a syn::ExprClosure) {
            self.0.get_or_insert(inner);
        }
    }
    let mut nested = Nested(None);
    nested.visit_expr(&closure.body);
    if let Some(inner) = nested.0 {
        return Err(diagnostics::error(
            Code::NestedClosure,
            inner,
            format!(
                "a closure inside the closure `{name}`: each closure is a function of its own, and a closure cannot hold \
                 another; bind both in the surrounding function"
            ),
        ));
    }
    // The name.
    let raw = pattern.ident.unraw().to_string();
    if raw.starts_with("__gust_") || raw.starts_with("__gpu_") || raw.starts_with("__Gust") {
        return Err(diagnostics::error(
            Code::ReservedPrefix,
            &pattern.ident,
            "the __gust_, __gpu_ and __Gust prefixes are reserved for generated shader symbols",
        ));
    }
    let clash = if function_names.contains(&name) || methods.by_name.contains_key(&name) {
        Some("a helper function or a method of the module")
    } else if is_dialect_callee(&name) {
        Some("a name the dialect gives a meaning to")
    } else if bindings.get(&name).copied().unwrap_or(0) > 1 {
        Some(
            "another binding of the function (a parameter, a local, a loop variable, a pattern or another closure)",
        )
    } else {
        None
    };
    if let Some(what) = clash {
        return Err(diagnostics::error(
            Code::MethodNameClash,
            &pattern.ident,
            format!(
                "the closure `{name}` is named like {what}: a closure name is unique in its function, so that a call \
                 `{name}(..)` has one meaning; rename the closure"
            ),
        ));
    }
    // The captures.
    let captures = closure_captures(closure, function_names);
    if !captures.is_empty() {
        let earlier: Vec<&String> = captures
            .iter()
            .filter(|capture| closure_names.contains(*capture))
            .collect();
        let listed = captures
            .iter()
            .map(|capture| format!("`{capture}`"))
            .collect::<Vec<_>>()
            .join(", ");
        let note = if earlier.is_empty() {
            String::new()
        } else {
            format!(
                " ({} {} a closure, and a closure cannot call a closure)",
                earlier
                    .iter()
                    .map(|capture| format!("`{capture}`"))
                    .collect::<Vec<_>>()
                    .join(", "),
                if earlier.len() == 1 { "is" } else { "are" }
            )
        };
        return Err(diagnostics::error(
            Code::ClosureCapture,
            local,
            format!(
                "the closure `{name}` uses {listed}, which belong to the surrounding function{note}: a closure captures \
                 nothing, so pass every value it needs as a parameter"
            ),
        ));
    }
    // The body, as the body of a helper (D154).
    let function = crate::slang::closure_function(&pattern.ident, closure)
        .expect("the form, the signature and the body were checked above");
    validate_function(
        stage,
        &function,
        function_names,
        result_helpers,
        methods,
        owner,
        true,
    )
}

/// The first call of a closure of `closures` inside `expression` (a `for` bound), in source order (slice m9.c).
fn first_closure_call<'a>(
    closures: &HashSet<String>,
    expression: &'a syn::Expr,
) -> Option<&'a syn::ExprCall> {
    struct Scan<'m, 'a> {
        closures: &'m HashSet<String>,
        found: Option<&'a syn::ExprCall>,
    }
    impl<'a> Visit<'a> for Scan<'_, 'a> {
        fn visit_expr_call(&mut self, call: &'a syn::ExprCall) {
            if self.found.is_some() {
                return;
            }
            if let syn::Expr::Path(path) = call.func.as_ref()
                && path.qself.is_none()
                && path
                    .path
                    .get_ident()
                    .is_some_and(|name| self.closures.contains(&name.unraw().to_string()))
            {
                self.found = Some(call);
                return;
            }
            visit::visit_expr_call(self, call);
        }
    }
    let mut scan = Scan {
        closures,
        found: None,
    };
    scan.visit_expr(expression);
    scan.found
}

/// The first call of a `&mut self` method of the module inside `expression` (a `for` bound), in source order. The macro
/// sees no types, so a name that some `&mut self` method has counts, as in the call graph (D162).
fn first_mutating_call<'a>(
    methods: &MethodTable,
    expression: &'a syn::Expr,
) -> Option<&'a syn::ExprMethodCall> {
    struct Scan<'m, 'a> {
        methods: &'m MethodTable,
        found: Option<&'a syn::ExprMethodCall>,
    }
    impl<'a> Visit<'a> for Scan<'_, 'a> {
        fn visit_expr_method_call(&mut self, call: &'a syn::ExprMethodCall) {
            if self.found.is_some() {
                return;
            }
            if self
                .methods
                .by_name
                .get(&call.method.unraw().to_string())
                .is_some_and(|facts| facts.ref_mut)
            {
                self.found = Some(call);
                return;
            }
            visit::visit_expr_method_call(self, call);
        }
    }
    let mut scan = Scan {
        methods,
        found: None,
    };
    scan.visit_expr(expression);
    scan.found
}

fn atomic_buffer_names(function: &syn::ItemFn) -> HashSet<String> {
    let mut usage = AtomicBufferUsage(HashSet::new());
    usage.visit_item_fn(function);
    usage.0
}

struct AtomicBufferUsage(HashSet<String>);

impl<'ast> Visit<'ast> for AtomicBufferUsage {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        let Some(_) = (match call.func.as_ref() {
            syn::Expr::Path(path) => atomic_operation_name(path),
            _ => None,
        }) else {
            visit::visit_expr_call(self, call);
            return;
        };
        if let Some(syn::Expr::Reference(reference)) = call.args.first()
            && let syn::Expr::Index(index) = reference.expr.as_ref()
            && let syn::Expr::Path(path) = index.expr.as_ref()
            && let Some(name) = path.path.get_ident()
        {
            self.0.insert(name.to_string());
        }
        visit::visit_expr_call(self, call);
    }
}

const QUALIFIED_MATH_PATH: &str = "math types are the prelude's: write `Vec4`, `Mat4` and their constructors without a module path (`Vec4::new`, `Mat4::IDENTITY`)";

/// Whether `path` reaches a math type through a module path (`super::Mat4`, `gust::math::Vec4::new`, `::Mat4`) instead
/// of by the prelude name. `Vec4::new` and `Mat4::IDENTITY` are the prelude form and are not qualified.
fn is_qualified_math_path(path: &syn::Path) -> bool {
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

/// What the visitor needs to know about a texture parameter.
#[derive(Clone, Copy)]
struct TextureFacts {
    access: texture::Access,
    /// Whether `sample` exists: a sampled texture of float texels.
    filterable: bool,
}

/// Whether `ty` is a tuple type with at least one element (`()` is the unit type, which is no tuple here).
fn is_tuple_type(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Tuple(tuple) => !tuple.elems.is_empty(),
        syn::Type::Paren(inner) => is_tuple_type(&inner.elem),
        syn::Type::Group(inner) => is_tuple_type(&inner.elem),
        _ => false,
    }
}

/// The most elements a tuple has in the dialect (GUST-E0213). Slang has no such limit (spike M9.s0, P3: a 9-tuple
/// compiles and validates); the bound is the dialect's own, to keep the surface that is tested small.
const MAX_TUPLE_ELEMENTS: usize = 8;

/// Why `element` cannot be an element type of a tuple, or `None`. An element is a path type other than `Option` and
/// `Result` (scalars, vectors, matrices, module structs and enums) or a tuple; arrays, references and the rest are
/// refused (GUST-E0213).
fn tuple_element_problem(element: &syn::Type) -> Option<&'static str> {
    match element {
        syn::Type::Paren(inner) => tuple_element_problem(&inner.elem),
        syn::Type::Group(inner) => tuple_element_problem(&inner.elem),
        syn::Type::Tuple(_) => None,
        syn::Type::Path(path) => {
            let last = path.path.segments.last()?.ident.to_string();
            matches!(last.as_str(), "Option" | "Result").then_some(
                "a tuple element cannot be an `Option` or a `Result`: keep the optional value in its own local",
            )
        }
        _ => Some(
            "a tuple element is a scalar, vector, matrix, struct, enum or tuple type (no array, reference or other type)",
        ),
    }
}

/// Why `element` cannot be the element type of an array (GUST-E0217): `None` for a scalar, vector, matrix or struct or
/// enum of the module (slice m9.r1).
fn array_element_problem(element: &syn::Type) -> Option<&'static str> {
    match strip_type_parens(element) {
        syn::Type::Array(_) => Some(
            "an array element cannot be an array: nested arrays are not supported, use one flat array and index `i * N + j`",
        ),
        syn::Type::Path(path) => {
            let last = path.path.segments.last()?.ident.to_string();
            if matches!(last.as_str(), "Option" | "Result") {
                Some(
                    "an array element cannot be an `Option` or a `Result`: keep the optional value in its own local",
                )
            } else if crate::slang::parameter_flavor(&syn::Type::Path(path.clone()))
                != crate::slang::ParameterFlavor::Value
            {
                Some(
                    "an array element must be a value type: a buffer, uniform, workgroup array, texture, sampler or invocation id is a resource of the kernel",
                )
            } else {
                None
            }
        }
        _ => Some(
            "an array element is a scalar, vector, matrix, struct or enum type of the module (no tuple, reference or other type)",
        ),
    }
}

/// `ty` without the parentheses and invisible groups around it.
fn strip_type_parens(ty: &syn::Type) -> &syn::Type {
    match ty {
        syn::Type::Paren(inner) => strip_type_parens(&inner.elem),
        syn::Type::Group(inner) => strip_type_parens(&inner.elem),
        other => other,
    }
}

/// `expression` without the parentheses and invisible groups around it.
fn strip_parens(expression: &syn::Expr) -> &syn::Expr {
    match expression {
        syn::Expr::Paren(inner) => strip_parens(&inner.expr),
        syn::Expr::Group(inner) => strip_parens(&inner.expr),
        other => other,
    }
}

/// The alternatives of a pattern: `A | B` flattened (parentheses looked through), any other pattern as itself.
fn flatten_or<'p>(pattern: &'p syn::Pat, out: &mut Vec<&'p syn::Pat>) {
    match pattern {
        syn::Pat::Or(or) => {
            for case in &or.cases {
                flatten_or(case, out);
            }
        }
        syn::Pat::Paren(inner) => flatten_or(&inner.pat, out),
        other => out.push(other),
    }
}

/// The first path pattern of an arm pattern (`E::A | E::B` gives `E::A`).
fn first_variant_path(pattern: &syn::Pat) -> Option<&syn::Path> {
    let mut alternatives = Vec::new();
    flatten_or(pattern, &mut alternatives);
    alternatives.into_iter().find_map(|pattern| match pattern {
        syn::Pat::Path(path) if path.qself.is_none() => Some(&path.path),
        _ => None,
    })
}

/// Whether some alternative of an arm pattern is a literal (or a range, which is refused with the literals' message).
fn has_literal_alternative(pattern: &syn::Pat) -> bool {
    let mut alternatives = Vec::new();
    flatten_or(pattern, &mut alternatives);
    alternatives
        .into_iter()
        .any(|pattern| matches!(pattern, syn::Pat::Lit(_) | syn::Pat::Range(_)))
}

/// Whether `expression` is a `match` that lowers to a Slang `switch`: a path pattern (an enum, slice m9.e1), a literal
/// or range pattern (an integer, slice m9.e2) in some arm, or only `_` arms. A `Result` match has `Ok(..)`/`Err(..)`
/// arms and neither.
pub(crate) fn is_switch_match(expression: &syn::ExprMatch) -> bool {
    expression
        .arms
        .iter()
        .any(|arm| first_variant_path(&arm.pat).is_some() || has_literal_alternative(&arm.pat))
        // A `match` of `_` arms alone is no `Result` match either (that needs `Ok`/`Err`): it is the enum form.
        || (!expression.arms.is_empty()
            && expression
                .arms
                .iter()
                .all(|arm| matches!(arm.pat, syn::Pat::Wild(_))))
}

/// The `match` that `expression` is, through parentheses, when it lowers to a `switch`.
fn switch_match_of(expression: &syn::Expr) -> Option<&syn::ExprMatch> {
    match strip_parens(expression) {
        syn::Expr::Match(expression) if is_switch_match(expression) => Some(expression),
        _ => None,
    }
}

/// The value and the suffix of an integer-literal pattern, or why it is refused.
fn integer_pattern(literal: &syn::ExprLit) -> Result<(i128, String), &'static str> {
    let syn::Lit::Int(value) = &literal.lit else {
        return Err(
            "only integer literals (`0`, `1u32`, `-1i32`) are patterns of an integer `match`: a `bool` is tested with `if`, and floats, characters and strings have no lowering",
        );
    };
    let suffix = value.suffix().to_owned();
    if !matches!(suffix.as_str(), "" | "u32" | "i32") {
        return Err(
            "an integer pattern is unsuffixed or `u32`/`i32`: the dialect has no other integer type",
        );
    }
    let number: i128 = value
        .base10_parse()
        .map_err(|_| "this integer literal does not parse")?;
    let (low, high) = match suffix.as_str() {
        "u32" => (0, i128::from(u32::MAX)),
        "i32" => (i128::from(i32::MIN), i128::from(i32::MAX)),
        _ => (i128::from(i32::MIN), i128::from(u32::MAX)),
    };
    if number < low || number > high {
        return Err("this integer literal is outside the range of a 32-bit integer");
    }
    Ok((number, suffix))
}

/// The tail expression of a block when it is a value (no `;`, not a statement-like expression): the value form of a
/// `match` arm, which this slice does not lower.
fn value_tail(block: &syn::Block) -> Option<&syn::Expr> {
    match block.stmts.last()? {
        syn::Stmt::Expr(expression, None)
            if !matches!(
                expression,
                syn::Expr::If(_)
                    | syn::Expr::Match(_)
                    | syn::Expr::ForLoop(_)
                    | syn::Expr::Block(_)
                    | syn::Expr::Return(_)
                    | syn::Expr::Break(_)
                    | syn::Expr::Continue(_)
                    | syn::Expr::Assign(_)
            ) && !is_compound_assignment(expression) =>
        {
            Some(expression)
        }
        _ => None,
    }
}

/// Whether `expression` is `a += b` and its kin (syn 2 spells them as binary expressions).
fn is_compound_assignment(expression: &syn::Expr) -> bool {
    matches!(
        expression,
        syn::Expr::Binary(binary) if matches!(binary.op,
            syn::BinOp::AddAssign(_) | syn::BinOp::SubAssign(_) | syn::BinOp::MulAssign(_)
            | syn::BinOp::DivAssign(_) | syn::BinOp::RemAssign(_) | syn::BinOp::BitXorAssign(_)
            | syn::BinOp::BitAndAssign(_) | syn::BinOp::BitOrAssign(_) | syn::BinOp::ShlAssign(_)
            | syn::BinOp::ShrAssign(_))
    )
}

/// How a `match` that lowers to a `switch` is used (slice m9.e2).
#[derive(Clone, Copy, PartialEq, Eq)]
enum MatchForm {
    /// `match e { .. }` as a statement: every arm is a block that ends in a statement.
    Statement,
    /// `let x: T = match e { .. };`, and an arm or a block tail inside one: every arm gives a value.
    Value,
}

/// What a `switch`-`match` is on.
enum SwitchOn {
    /// An enum of the module, by name.
    Enum(String),
    /// A `u32` or `i32`: literal patterns.
    Integer,
}

struct RestrictedVisitor {
    function_names: HashSet<String>,
    result_helpers: HashSet<String>,
    result_values: HashSet<String>,
    resources: HashSet<String>,
    invocations: HashSet<String>,
    /// `LocalId`, `GroupId` and `Workgroup` parameters: locals cannot shadow them, and `.global_id()` is not theirs.
    derived_ids: HashSet<String>,
    /// The `Workgroup<..>` parameter types of the kernel, by address: the only place the type may stand (E0210).
    workgroup_types: HashSet<usize>,
    /// Uniform parameters. They are read field by field only (E0321), and are not in `resources` because `.len()` of
    /// one is E0306.
    uniforms: HashSet<String>,
    /// Texture parameters (mx.t3). They are used as the receiver of a texture method only (GUST-E0736).
    textures: HashMap<String, TextureFacts>,
    /// `Sampler` parameters: the first argument of `sample` and nothing else.
    samplers: HashSet<String>,
    /// The texture and sampler parameter types of the kernel, by address: the only place the types may stand.
    texture_types: HashSet<usize>,
    /// Read-write buffer parameters and their element type, for atomic receivers.
    read_write_buffers: HashMap<String, String>,
    /// Buffer parameters used by an atomic operation in this function.
    atomic_buffers: HashSet<String>,
    /// Set while inside a type position where `Option` has no proven layout.
    option_forbidden: Option<&'static str>,
    /// Set while inside a type position where `Result` has no proven layout.
    result_forbidden: Option<&'static str>,
    /// Number of enclosing `for` loops; `break`/`continue` need at least one.
    loop_depth: usize,
    /// The module's method names, each with whether some impl declares it with `&mut self` or `self` (m9.a1).
    methods: MethodTable,
    /// The struct whose `impl` holds the function being visited: `Self` is its name there (slice m9.a2).
    self_struct: Option<String>,
    /// The fields of that struct: a helper of the same name would be captured by the field inside the Slang struct.
    owner_fields: HashSet<String>,
    /// The value parameters of the function (not its receiver): a `&mut self` method is not called on one of them.
    value_params: HashSet<String>,
    /// The `loop_depth` at the start of each enclosing enum `match` arm, innermost last: a `break` or `continue` at that
    /// depth would bind to the Slang `switch`, not to the Rust loop (GUST-E0417).
    switch_depths: Vec<usize>,
    /// The `match` that ends the body of a function with a result type, by address (m9.e2 review S1): when its arms
    /// give values it is a value in tail position, GUST-E0419.
    body_tail_match: Option<usize>,
    /// The parameters and locals whose type is an enum of the module, with the enum's name (by name, as `value_params`).
    enum_values: HashMap<String, String>,
    /// The parameters and locals that hold a tuple, by name (slice m9.t; scoped like `enum_values`): a `match` on one
    /// and a comparison of one have no lowering.
    tuple_values: HashSet<String>,
    /// The names of the function that hold an array and its length, `None` for a name bound otherwise or with another
    /// length (slice m9.r1; `crate::slang::array_lengths`, which the emitter reads as well).
    array_lens: HashMap<String, Option<u32>>,
    /// How many `[..]` index operands enclose the node being visited, counting only the operators that carry a `usize`
    /// value on (see `visit_expr`): `as usize` and an array `.len()` stand there and nowhere else (GUST-E0334).
    index_depth: usize,
    /// The `a.len()` calls of arrays that are the operand of a cast to a 32-bit scalar (`a.len() as u32`), by address.
    len_casts: HashSet<usize>,
    /// The closures bound by a `let` in scope (slice m9.c), by binding name: `f(..)` calls one, anything else that names
    /// one is GUST-E0330. The names are unique in a function and apart from every other binding (GUST-E0328), so the
    /// set needs no shadowing rules; it is restored at the end of a block.
    closures: HashSet<String>,
    error: Option<syn::Error>,
}

impl RestrictedVisitor {
    fn new(function_names: HashSet<String>, result_helpers: HashSet<String>) -> Self {
        Self {
            function_names,
            result_helpers,
            result_values: HashSet::new(),
            resources: HashSet::new(),
            invocations: HashSet::new(),
            derived_ids: HashSet::new(),
            workgroup_types: HashSet::new(),
            uniforms: HashSet::new(),
            textures: HashMap::new(),
            samplers: HashSet::new(),
            texture_types: HashSet::new(),
            read_write_buffers: HashMap::new(),
            atomic_buffers: HashSet::new(),
            option_forbidden: None,
            result_forbidden: None,
            loop_depth: 0,
            methods: MethodTable::default(),
            self_struct: None,
            owner_fields: HashSet::new(),
            value_params: HashSet::new(),
            switch_depths: Vec::new(),
            body_tail_match: None,
            enum_values: HashMap::new(),
            tuple_values: HashSet::new(),
            array_lens: HashMap::new(),
            index_depth: 0,
            len_casts: HashSet::new(),
            closures: HashSet::new(),
            error: None,
        }
    }

    /// Whether `expression` is a tuple, when the macro can tell without types: a tuple expression, a parameter or local
    /// that holds one, or a call of a helper or method that returns one (a method by its name alone).
    fn is_tuple_expr(&self, expression: &syn::Expr) -> bool {
        match strip_parens(expression) {
            syn::Expr::Tuple(tuple) => !tuple.elems.is_empty(),
            syn::Expr::Path(path) if path.qself.is_none() => path
                .path
                .get_ident()
                .is_some_and(|name| self.tuple_values.contains(&name.unraw().to_string())),
            syn::Expr::Call(call) => matches!(
                call.func.as_ref(),
                syn::Expr::Path(path) if path.qself.is_none()
                    && path.path.segments.last().is_some_and(|last| {
                        self.methods.tuple_returns.contains(&last.ident.unraw().to_string())
                    })
            ),
            syn::Expr::MethodCall(call) => self
                .methods
                .tuple_returns
                .contains(&call.method.unraw().to_string()),
            _ => false,
        }
    }

    /// `let (a, b) = value;` and `let (a, b): (A, B) = value;` (slice m9.t): 2 to 8 plain identifiers, nothing nested.
    fn visit_tuple_local<'ast>(
        &mut self,
        local: &'ast syn::Local,
        tuple: &'ast syn::PatTuple,
        annotation: Option<&'ast syn::Type>,
    ) {
        use syn::visit::Visit;
        if tuple.elems.len() < 2 || tuple.elems.len() > MAX_TUPLE_ELEMENTS {
            self.reject(
                Code::TupleShape,
                tuple,
                &format!(
                    "a tuple pattern has 2 to {MAX_TUPLE_ELEMENTS} elements, this one has {}",
                    tuple.elems.len()
                ),
            );
            return;
        }
        for element in &tuple.elems {
            match element {
                syn::Pat::Ident(ident) if ident.by_ref.is_some() || ident.subpat.is_some() => {
                    self.reject(
                        Code::BindingMode,
                        element,
                        "a tuple element binds by value: no `ref` and no `@`",
                    );
                    return;
                }
                syn::Pat::Ident(_) => {}
                _ => {
                    self.reject(
                        Code::LetNeedsIdentifier,
                        element,
                        "a tuple destructuring binds one plain identifier per element: no nested pattern, `_` or `..`; \
                         bind the element to a name and ignore it",
                    );
                    return;
                }
            }
        }
        // The names and the attributes go through the per-binding checks of a plain `let` (GUST-E0901 on a reserved
        // prefix, which would let `__gust_r_new` capture a renamed local; review m9.t M1).
        for attribute in &local.attrs {
            self.visit_attribute(attribute);
        }
        for element in &tuple.elems {
            self.visit_pat(element);
        }
        if let Some(ty) = annotation {
            let same_length = matches!(
                strip_type_parens(ty),
                syn::Type::Tuple(types) if types.elems.len() == tuple.elems.len()
            );
            if !same_length {
                self.reject(
                    Code::TupleShape,
                    ty,
                    "a tuple destructuring is annotated with a tuple type of the same length",
                );
                return;
            }
        }
        if local
            .init
            .as_ref()
            .is_some_and(|initial| initial.diverge.is_some())
        {
            self.reject(
                Code::LetElse,
                local,
                "let-else requires control-flow lowering and is not supported yet",
            );
            return;
        }
        let element_types: Vec<Option<&syn::Type>> = match annotation.map(strip_type_parens) {
            Some(syn::Type::Tuple(types)) => types.elems.iter().map(Some).collect(),
            _ => vec![None; tuple.elems.len()],
        };
        // The initializer is evaluated before the names exist (`let (a, b) = f(a);` reads the outer `a`).
        if let Some(ty) = annotation {
            self.visit_type(ty);
        }
        if let Some(initial) = &local.init {
            if matches!(strip_parens(&initial.expr), syn::Expr::Match(_)) {
                self.reject(
                    Code::ValueMatchForm,
                    &initial.expr,
                    "a tuple destructuring takes no `match` initializer: bind the `match` to a typed tuple `let` first, then destructure it",
                );
                return;
            }
            self.visit_expr(&initial.expr);
        }
        for (element, ty) in tuple.elems.iter().zip(element_types) {
            let syn::Pat::Ident(ident) = element else {
                continue;
            };
            let name = ident.ident.to_string();
            if self.is_parameter_name(&name) {
                self.reject(
                    Code::ShadowedParameter,
                    element,
                    "GPU locals cannot shadow resource or dispatch-thread parameters",
                );
                return;
            }
            let key = ident.ident.unraw().to_string();
            self.value_params.remove(&key);
            self.result_values.remove(&ident.ident.to_string());
            // What is not visible stays unknown: an enum held in a tuple is not tracked (documented limitation).
            match ty.and_then(|ty| enum_type_name(ty, &self.methods.enums)) {
                Some(enum_name) => {
                    self.enum_values.insert(key.clone(), enum_name);
                }
                None => {
                    self.enum_values.remove(&key);
                }
            }
            if ty.is_some_and(is_tuple_type) {
                self.tuple_values.insert(key);
            } else {
                self.tuple_values.remove(&key);
            }
        }
    }

    /// The enum of the module that `expression` is a value of, when the macro can tell without types: a variant path, a
    /// parameter or local declared with the enum, or a call of a helper that returns one.
    fn enum_of(&self, expression: &syn::Expr) -> Option<String> {
        match expression {
            syn::Expr::Paren(inner) => self.enum_of(&inner.expr),
            syn::Expr::Group(inner) => self.enum_of(&inner.expr),
            syn::Expr::Path(path) if path.qself.is_none() => {
                if let Some(name) = path.path.get_ident() {
                    return self.enum_values.get(&name.unraw().to_string()).cloned();
                }
                let (enum_name, _) = self.variant_path(&path.path)?;
                Some(enum_name)
            }
            syn::Expr::Call(call) => {
                let syn::Expr::Path(path) = call.func.as_ref() else {
                    return None;
                };
                let name = path.path.get_ident()?.to_string();
                self.methods.enum_returns.get(&name).cloned()
            }
            _ => None,
        }
    }

    /// `(enum, variant)` when `path` is `E::V` with `E` an enum of the module and `V` one of its variants.
    fn variant_path(&self, path: &syn::Path) -> Option<(String, String)> {
        if path.leading_colon.is_some() || path.segments.len() != 2 {
            return None;
        }
        let first = &path.segments[0];
        let last = &path.segments[1];
        if !first.arguments.is_none() || !last.arguments.is_none() {
            return None;
        }
        let enum_name = first.ident.unraw().to_string();
        let variant = last.ident.unraw().to_string();
        self.methods
            .enums
            .get(&enum_name)?
            .variants
            .contains(&variant)
            .then_some((enum_name, variant))
    }

    /// Whether `tokens` name an enum of the module anywhere.
    fn mentions_enum(&self, tokens: impl quote::ToTokens) -> bool {
        fn scan(stream: proc_macro2::TokenStream, enums: &HashMap<String, EnumFacts>) -> bool {
            stream.into_iter().any(|tree| match tree {
                proc_macro2::TokenTree::Ident(ident) => {
                    enums.contains_key(&ident.unraw().to_string())
                }
                proc_macro2::TokenTree::Group(group) => scan(group.stream(), enums),
                _ => false,
            })
        }
        !self.methods.enums.is_empty() && scan(tokens.to_token_stream(), &self.methods.enums)
    }

    /// Judges a `match` that lowers to a Slang `switch` (slice m9.e1, D157, and e2 for integers and the value form) and
    /// visits its parts: on an enum (`E::A | E::B`, `_`) or on an integer (`0`, `1 | 2`, `-1i32`, `_`), as a statement or
    /// as the initializer of a typed `let`.
    /// Refuses a `match` on a tuple (GUST-E0515; tuple patterns are not part of the dialect). Returns whether it did.
    fn reject_tuple_scrutinee(&mut self, expression: &syn::ExprMatch) -> bool {
        if !self.is_tuple_expr(&expression.expr) {
            return false;
        }
        self.reject(
            Code::MatchScrutinee,
            &expression.expr,
            "a tuple is not a `match` scrutinee (there are no tuple patterns): destructure it with `let (a, b) = t;` and match on a part",
        );
        true
    }

    fn check_switch_match(&mut self, expression: &syn::ExprMatch, form: MatchForm) {
        if self.reject_tuple_scrutinee(expression) {
            return;
        }
        let integer = expression
            .arms
            .iter()
            .any(|arm| has_literal_alternative(&arm.pat));
        let on = if integer {
            // An integer scrutinee is any expression: `switch (e)` evaluates it once, and the patterns decide the form.
            SwitchOn::Integer
        } else {
            let scrutinee_ok = matches!(
                strip_parens(&expression.expr),
                syn::Expr::Path(_) | syn::Expr::Call(_) | syn::Expr::MethodCall(_)
            );
            // A method call's enum is not tracked; the first arm names it.
            let first_path = expression
                .arms
                .iter()
                .find_map(|arm| first_variant_path(&arm.pat));
            let named_by_arm = first_path.and_then(|path| {
                let owner = path.segments.first()?.ident.unraw().to_string();
                self.methods.enums.contains_key(&owner).then_some(owner)
            });
            let Some(enum_name) = scrutinee_ok
                .then(|| self.enum_of(&expression.expr).or(named_by_arm))
                .flatten()
            else {
                // A path that names no enum of the module (`u32::MAX`, `Other::A`) is a constant used as a pattern.
                let constant = first_path.filter(|path| {
                    path.segments.first().is_none_or(|first| {
                        !self
                            .methods
                            .enums
                            .contains_key(&first.ident.unraw().to_string())
                    })
                });
                if let Some(path) = constant {
                    self.reject(
                        Code::MatchPattern,
                        path,
                        "a constant is not a pattern: an integer `match` takes literals (`0`, `1u32`, `-1i32`) and `_`, an enum `match` takes `E::A` of an enum of the module",
                    );
                    return;
                }
                self.reject(
                    Code::MatchScrutinee,
                    &expression.expr,
                    "a `match` on an enum takes a local, a parameter, a call or a variant as its scrutinee: bind a larger expression to a local first (`let m = ..; match m { .. }`); a `match` on a `u32` or `i32` takes any expression but needs an integer literal arm (`0`, `1 | 2`) before its `_`",
                );
                return;
            };
            SwitchOn::Enum(enum_name)
        };
        let mut seen: HashSet<String> = HashSet::new();
        let mut seen_numbers: HashSet<i128> = HashSet::new();
        let mut wildcard = false;
        for arm in &expression.arms {
            if let Some(attribute) = arm.attrs.first() {
                self.reject(
                    Code::MatchArmAttribute,
                    attribute,
                    "attributes on a `match` arm are not supported",
                );
                return;
            }
            if let Some((_, guard)) = &arm.guard {
                self.reject(
                    Code::EnumMatchGuard,
                    guard,
                    "a `match` arm takes no `if` guard: the Slang `switch` has no guard, and a guard would make the arm's coverage depend on a run-time value; test the condition inside the arm",
                );
                return;
            }
            let mut alternatives: Vec<&syn::Pat> = Vec::new();
            flatten_or(&arm.pat, &mut alternatives);
            let lone = alternatives.len() == 1;
            for pattern in alternatives {
                match (pattern, &on) {
                    (syn::Pat::Wild(_), _) if lone => {
                        if wildcard {
                            self.reject(Code::MatchPattern, pattern, "a `match` has one `_` arm");
                            return;
                        }
                        wildcard = true;
                    }
                    (syn::Pat::Path(path), SwitchOn::Enum(enum_name)) if path.qself.is_none() => {
                        match self.variant_path(&path.path) {
                            Some((owner, variant)) if &owner == enum_name => {
                                if !seen.insert(variant.clone()) || wildcard {
                                    self.reject(
                                        Code::MatchPattern,
                                        pattern,
                                        &format!("`{owner}::{variant}` is matched twice, or after `_`: the second arm can never run"),
                                    );
                                    return;
                                }
                            }
                            _ => {
                                self.reject(
                                    Code::MatchPattern,
                                    pattern,
                                    &format!("a `match` arm on `{enum_name}` names one of its variants as `{enum_name}::Variant`"),
                                );
                                return;
                            }
                        }
                    }
                    (syn::Pat::Lit(literal), SwitchOn::Integer) => match integer_pattern(literal) {
                        Ok((number, _)) => {
                            if !seen_numbers.insert(number) || wildcard {
                                self.reject(
                                    Code::MatchPattern,
                                    pattern,
                                    &format!("`{number}` is matched twice, or after `_`: the second arm can never run"),
                                );
                                return;
                            }
                        }
                        Err(reason) => {
                            self.reject(Code::MatchPattern, pattern, reason);
                            return;
                        }
                    },
                    (syn::Pat::Range(_), _) => {
                        self.reject(
                            Code::MatchPattern,
                            pattern,
                            "a range is not a pattern of a `match` here: list the integer literals (`0 | 1 | 2`) or test the range with `if`",
                        );
                        return;
                    }
                    (_, SwitchOn::Enum(_)) => {
                        self.reject(
                            Code::MatchPattern,
                            pattern,
                            "an enum `match` arm is `E::A`, `E::A | E::B` or `_`: bindings, literals, tuples and ranges have no lowering (an `_` joined by `|` is refused as well)",
                        );
                        return;
                    }
                    (_, SwitchOn::Integer) => {
                        self.reject(
                            Code::MatchPattern,
                            pattern,
                            "an integer `match` arm is a literal (`0`, `1 | 2`, `-1i32`) or `_`: bindings, constants, enum variants (an enum `match` does not mix with literals), tuples and ranges have no lowering (an `_` joined by `|` is refused as well)",
                        );
                        return;
                    }
                }
            }
            if form == MatchForm::Statement && !matches!(arm.body.as_ref(), syn::Expr::Block(_)) {
                self.reject(
                    Code::MatchArmBlock,
                    &arm.body,
                    "each arm of a statement `match` is a block: `E::A => { .. }`; an arm that gives a value belongs to `let x: T = match e { .. };` with a typed `let`",
                );
                return;
            }
        }
        if matches!(on, SwitchOn::Integer) && !wildcard {
            self.reject(
                Code::MatchPattern,
                expression,
                "an integer `match` ends in a `_` arm: no list of literals covers every `u32` or `i32`",
            );
            return;
        }
        self.visit_expr(&expression.expr);
        for arm in &expression.arms {
            self.switch_depths.push(self.loop_depth);
            match form {
                MatchForm::Statement => {
                    let syn::Expr::Block(body) = arm.body.as_ref() else {
                        unreachable!("a statement `match` arm was checked to be a block")
                    };
                    if let Some(tail) = value_tail(&body.block) {
                        self.switch_depths.pop();
                        self.reject(
                            Code::MatchAsValue,
                            tail,
                            "the arms of a statement `match` end in a statement; a tail expression makes the `match` a value, which is `let x: T = match e { .. };` with a typed `let`: bind it there, or end the arm's last statement with `;`",
                        );
                        return;
                    }
                    self.visit_block(&body.block);
                }
                MatchForm::Value => self.visit_value_arm(&arm.body),
            }
            self.switch_depths.pop();
        }
    }

    /// An arm of a value `match`: an expression, a block that ends in an expression (or in a `return`), or a nested
    /// value `match`. The value is assigned to the `let` it initializes.
    fn visit_value_arm(&mut self, body: &syn::Expr) {
        match strip_parens(body) {
            syn::Expr::Block(block) if block.label.is_none() => {
                self.visit_value_block(&block.block);
            }
            syn::Expr::Match(inner) if is_switch_match(inner) => {
                self.check_switch_match(inner, MatchForm::Value);
            }
            other => self.visit_expr(other),
        }
    }

    /// The statements of a block whose last statement is its value; the block is a scope of its own.
    fn visit_value_block(&mut self, block: &syn::Block) {
        let outer = self.enum_values.clone();
        let outer_tuples = self.tuple_values.clone();
        let (head, tail) = match block.stmts.split_last() {
            Some((syn::Stmt::Expr(tail, None), head)) => (head, Some(tail)),
            Some((syn::Stmt::Expr(syn::Expr::Return(_), Some(_)), _)) => {
                (block.stmts.as_slice(), None)
            }
            _ => {
                self.reject(
                    Code::MatchArmBlock,
                    block,
                    "each arm of a value `match` gives a value: an expression, or a block whose last line is an expression without `;` (or a `return`)",
                );
                return;
            }
        };
        for statement in head {
            self.visit_stmt(statement);
        }
        if let Some(tail) = tail {
            self.visit_value_arm(tail);
        }
        self.enum_values = outer;
        self.tuple_values = outer_tuples;
    }

    /// Whether `name` is a kernel parameter: a local cannot shadow one.
    fn is_parameter_name(&self, name: &str) -> bool {
        self.resources.contains(name)
            || self.invocations.contains(name)
            || self.derived_ids.contains(name)
            || self.uniforms.contains(name)
            || self.textures.contains_key(name)
            || self.samplers.contains(name)
    }

    /// Whether `expression` is a bare use of a uniform parameter, possibly in parentheses.
    fn is_uniform_parameter(&self, expression: &syn::Expr) -> bool {
        match expression {
            syn::Expr::Path(path) => {
                path.qself.is_none()
                    && path
                        .path
                        .get_ident()
                        .is_some_and(|name| self.uniforms.contains(&name.to_string()))
            }
            syn::Expr::Paren(paren) => self.is_uniform_parameter(&paren.expr),
            syn::Expr::Group(group) => self.is_uniform_parameter(&group.expr),
            _ => false,
        }
    }

    /// The name of a bare parameter use: `img` or `(img)`.
    fn direct_name(expression: &syn::Expr) -> Option<String> {
        match expression {
            syn::Expr::Path(path) if path.qself.is_none() => {
                path.path.get_ident().map(ToString::to_string)
            }
            syn::Expr::Paren(paren) => Self::direct_name(&paren.expr),
            syn::Expr::Group(group) => Self::direct_name(&group.expr),
            _ => None,
        }
    }

    /// A method call whose receiver is a texture or a sampler parameter (slice mx.t3, D64): `load`, `store`,
    /// `sample`, `width` and `height` on the textures that have them. Returns whether the receiver was one; the call
    /// is then checked here and its arguments visited, and the receiver is not visited (a texture is used as a
    /// receiver only, GUST-E0736).
    fn check_texture_call(&mut self, call: &syn::ExprMethodCall) -> bool {
        let Some(receiver) = Self::direct_name(&call.receiver) else {
            return false;
        };
        if self.samplers.contains(&receiver) {
            self.reject(
                Code::TextureUse,
                call,
                &format!(
                    "`{receiver}` is a `Sampler`: it has no methods; pass it as the first argument of a texture's \
                     `.sample({receiver}, u, v)`"
                ),
            );
            return true;
        }
        let Some(facts) = self.textures.get(&receiver).copied() else {
            return false;
        };
        let method = call.method.to_string();
        if call.turbofish.is_some() {
            self.reject(
                Code::BuiltinTypeArguments,
                call,
                "GPU built-ins do not accept type arguments",
            );
            return true;
        }
        let Some(kind) = texture::Method::from_name(&method) else {
            self.reject(
                Code::TextureMethod,
                call,
                &format!(
                    "a texture has the methods `.load(x, y)`, `.store(x, y, v)`, `.sample(sampler, u, v)`, \
                     `.width()` and `.height()`, not `.{method}(..)`"
                ),
            );
            return true;
        };
        if !kind.exists_on(facts.access) {
            let reason = match (kind, facts.access) {
                (texture::Method::Load, _) => {
                    "a write-only texture cannot be read: use `StorageTexture2D` (three formats) or bind the texture \
                     as a sampled `Texture2D` in a second kernel"
                }
                (texture::Method::Store, _) => {
                    "a sampled texture is read-only: use `StorageTexture2D` or `WriteTexture2D` to write"
                }
                (_, texture::Access::ReadWrite) => {
                    "`.sample(..)` filters a sampled `Texture2D`; a storage texture is read with `.load(x, y)`"
                }
                _ => {
                    "`.sample(..)` filters a sampled `Texture2D`; a write-only texture has `.store(x, y, v)`, \
                     `.width()` and `.height()` only"
                }
            };
            self.reject(
                Code::TextureMethod,
                call,
                &format!("`{receiver}` has no `.{method}(..)`: {reason}"),
            );
            return true;
        }
        if call.args.len() != kind.arity() {
            self.reject(
                Code::TextureCoordinate,
                call,
                &format!(
                    "`.{method}(..)` takes {} argument(s): {}",
                    kind.arity(),
                    match kind {
                        texture::Method::Load => "the texel address `x, y` (both `u32`)",
                        texture::Method::Store =>
                            "the texel address `x, y` (both `u32`) and the value",
                        texture::Method::Sample =>
                            "a sampler and the coordinates `u, v` (both `f32`)",
                        _ => "none",
                    }
                ),
            );
            return true;
        }
        let mut skipped = 0;
        match kind {
            texture::Method::Sample => {
                if !facts.filterable {
                    self.reject(
                        Code::SampleInteger,
                        call,
                        &format!(
                            "`{receiver}` has integer texels, which are never filtered: read it with \
                             `.load(x, y)`, or sample a texture of `f32`, `Vec2` or `Vec4` texels"
                        ),
                    );
                    return true;
                }
                let sampler = Self::direct_name(&call.args[0]);
                if !sampler.is_some_and(|name| self.samplers.contains(&name)) {
                    self.reject(
                        Code::SamplerArgument,
                        &call.args[0],
                        "the first argument of `.sample(..)` is a `Sampler` parameter of the kernel, passed by name",
                    );
                }
                skipped = 1;
                for coordinate in call.args.iter().skip(1) {
                    if let Some(problem) = coordinate_problem(coordinate) {
                        self.reject(
                            Code::TextureCoordinate,
                            coordinate,
                            &format!("a sample coordinate is an `f32`, and this is {problem}"),
                        );
                    }
                }
            }
            texture::Method::Load | texture::Method::Store => {
                for coordinate in call.args.iter().take(2) {
                    if let Some(problem) = address_problem(coordinate) {
                        self.reject(
                            Code::TextureCoordinate,
                            coordinate,
                            &format!("a texel address is a `u32`, and this is {problem}"),
                        );
                    }
                }
            }
            texture::Method::Width | texture::Method::Height => {}
        }
        for argument in call.args.iter().skip(skipped) {
            self.visit_expr(argument);
        }
        true
    }

    /// The identifier a place expression is rooted in: `out` for `out`, `out[i]`, `out[i].a.b` and parenthesised forms.
    fn place_root(expression: &syn::Expr) -> Option<String> {
        match expression {
            syn::Expr::Path(path) if path.qself.is_none() => {
                path.path.get_ident().map(ToString::to_string)
            }
            syn::Expr::Field(field) => Self::place_root(&field.base),
            syn::Expr::Index(index) => Self::place_root(&index.expr),
            syn::Expr::Paren(paren) => Self::place_root(&paren.expr),
            syn::Expr::Group(group) => Self::place_root(&group.expr),
            _ => None,
        }
    }

    /// A call `recv.name(args)` of a method of the module (slice m9.a1). A `&self` method copies its receiver, so any
    /// place may be one. A method that takes `&mut self` or `self` would write through a storage, uniform, workgroup
    /// or texture place, a shared write that the kernel statements do not show, so it is refused there (GUST-E0333,
    /// D154); the validator decides from the root identifier of the receiver, no type is needed.
    ///
    /// Slice m9.a2: a `&mut self` method is called on a local place only, a `let mut` binding or a field or index of one
    /// (or on `self` inside a `&mut self` method). A receiver that is no place (the result of a call, a literal) has no
    /// Slang lvalue to mutate, and a value parameter is an `in` parameter there (slangc warns that the change is lost),
    /// so both are GUST-E0333 as well: copy the value to a `let mut` local first.
    /// GUST-E0421 for the place of an assignment: Rust evaluates the value before the place, and Slang evaluates a call
    /// inside the place first (review m9.a2 M1), so a `&mut self` call there would run in the other order.
    fn reject_mutating_in_place(&mut self, place: &syn::Expr) -> bool {
        if let Some(call) = first_mutating_call(&self.methods, place) {
            self.reject(
                Code::ForBoundSideEffect,
                call,
                &format!(
                    "the place of an assignment calls `.{}(..)`, which takes `&mut self`: Rust evaluates the assigned value before the place and Slang evaluates the call in the place first, so the order of the effects would differ; call the method before the statement and use the value (`let i = c.next(); out[i] = value;`)",
                    call.method.unraw()
                ),
            );
            return true;
        }
        false
    }

    /// GUST-E0421 for an indexed place of a local whose index reads something the value of the assignment changes
    /// (slice m9.r1): Rust evaluates the value of `a[s.n as usize] = s.bump()` first and the index after it, and Slang
    /// computes the index before the value (measured on slangc 2026.21 for `=` and `op=`, on a local array), so the
    /// element written would differ. The roots are the receivers of the `&mut self` calls in the value; the index
    /// "reads" a root when it names it. Every indexed place is covered, a storage or workgroup element included (fix
    /// round 1 of m9.r1): slangc 2026.21 reads the index of `out[c.n] = c.next()` first, where 2026.20 called first,
    /// so a rule that relied on its order would not be robust.
    fn reject_stale_index(&mut self, place: &syn::Expr, value: &syn::Expr) -> bool {
        struct Roots<'m> {
            methods: &'m MethodTable,
            roots: Vec<(String, String)>,
        }
        impl<'a> Visit<'a> for Roots<'_> {
            fn visit_expr_method_call(&mut self, call: &'a syn::ExprMethodCall) {
                if self
                    .methods
                    .by_name
                    .get(&call.method.unraw().to_string())
                    .is_some_and(|facts| facts.ref_mut)
                    && let Some(root) = RestrictedVisitor::place_root(&call.receiver)
                {
                    self.roots.push((root, call.method.unraw().to_string()));
                }
                visit::visit_expr_method_call(self, call);
            }
        }
        struct Names(HashSet<String>);
        impl<'a> Visit<'a> for Names {
            fn visit_expr_path(&mut self, path: &'a syn::ExprPath) {
                if let Some(name) = path.path.get_ident() {
                    self.0.insert(name.unraw().to_string());
                }
            }
        }
        struct Indexes<'a>(Vec<&'a syn::Expr>);
        impl<'a> Visit<'a> for Indexes<'a> {
            fn visit_expr_index(&mut self, index: &'a syn::ExprIndex) {
                self.0.push(&index.index);
                visit::visit_expr_index(self, index);
            }
        }
        if Self::place_root(place).is_none() {
            return false;
        }
        let mut roots = Roots {
            methods: &self.methods,
            roots: Vec::new(),
        };
        roots.visit_expr(value);
        if roots.roots.is_empty() {
            return false;
        }
        let roots = roots.roots;
        let mut indexes = Indexes(Vec::new());
        indexes.visit_expr(place);
        for index in indexes.0 {
            let mut names = Names(HashSet::new());
            names.visit_expr(index);
            if let Some((root, method)) = roots.iter().find(|(root, _)| names.0.contains(root)) {
                self.reject(
                    Code::ForBoundSideEffect,
                    index,
                    &format!(
                        "the index of the assigned place reads `{root}`, and the value calls `.{method}(..)` on it with `&mut self`: Rust evaluates the value first and the index after it, Slang computes the index first, so another element than Rust's would be written; call the method first (`let v = {root}.{method}(..); a[i] = v;`)"
                    ),
                );
                return true;
            }
        }
        false
    }

    fn check_module_method_call(&mut self, call: &syn::ExprMethodCall, facts: MethodFacts) {
        let method = call.method.to_string();
        let root = Self::place_root(&call.receiver);
        if facts.mutating()
            && let Some(root) = &root
            && (self.resources.contains(root)
                || self.uniforms.contains(root)
                || self.derived_ids.contains(root))
        {
            self.reject(
                Code::MutatingMethodOnSharedPlace,
                call,
                &format!(
                    "`.{method}(..)` takes `&mut self` or `self`, and `{root}` is a storage buffer, a uniform, a workgroup array or an invocation id of the kernel: a method would write through it where no statement shows the write; copy the element to a local, call the method on the local and store the result (`out[i] = value;`)"
                ),
            );
            return;
        }
        if facts.ref_mut {
            let problem = match &root {
                None => Some("the receiver is not a place (it is the result of a call, a literal or a block)".to_owned()),
                Some(root) if self.value_params.contains(root) => Some(format!(
                    "`{root}` is a parameter, which Slang passes as a read-only value (`in`); the dialect asks for an explicit local copy before a mutating call"
                )),
                Some(_) => None,
            };
            if let Some(problem) = problem {
                self.reject(
                    Code::MutatingMethodOnSharedPlace,
                    call,
                    &format!(
                        "`.{method}(..)` takes `&mut self` and is called on a `let mut` local only: {problem}; bind the value with `let mut copy = ..;`, call the method on `copy` and use `copy`"
                    ),
                );
                return;
            }
        }
        self.visit_expr(&call.receiver);
        for argument in &call.args {
            self.visit_expr(argument);
        }
    }

    fn reject(&mut self, code: Code, node: impl quote::ToTokens, message: &str) {
        self.push(diagnostics::error(code, node, message));
    }

    /// Adds `error` after the ones already collected.
    fn push(&mut self, error: syn::Error) {
        if let Some(existing) = &mut self.error {
            existing.combine(error);
        } else {
            self.error = Some(error);
        }
    }

    fn finish(self) -> syn::Result<()> {
        self.error.map_or(Ok(()), Err)
    }

    fn is_known_result(&self, expression: &syn::Expr) -> bool {
        match expression {
            syn::Expr::Path(path) => path
                .path
                .get_ident()
                .is_some_and(|name| self.result_values.contains(&name.to_string())),
            syn::Expr::Call(call) => match call.func.as_ref() {
                syn::Expr::Path(path) => path
                    .path
                    .get_ident()
                    .is_some_and(|name| self.result_helpers.contains(&name.to_string())),
                _ => false,
            },
            syn::Expr::Paren(paren) => self.is_known_result(&paren.expr),
            syn::Expr::Group(group) => self.is_known_result(&group.expr),
            _ => false,
        }
    }

    /// Shared checks for `break` and `continue` in statement position.
    fn check_jump(
        &mut self,
        node: impl quote::ToTokens,
        label: Option<&syn::Lifetime>,
        name: &str,
    ) {
        if let Some(label) = label {
            self.reject(
                Code::LoopLabel,
                label,
                "loop labels are not supported in the Rust-to-Slang subset; restructure with a flag or a helper return",
            );
            return;
        }
        if self.loop_depth == 0 {
            self.reject(
                Code::JumpOutsideLoop,
                node,
                &format!("`{name}` outside a loop has no meaning"),
            );
        } else if self.switch_depths.last() == Some(&self.loop_depth) {
            let reason = if name == "break" {
                "would end the Slang `switch`, not the loop around it"
            } else {
                "is refused for symmetry with `break` (Slang's `continue` inside a `switch` does continue the loop)"
            };
            self.reject(
                Code::JumpInMatchArm,
                node,
                &format!(
                    "`{name}` in an arm of an enum `match` {reason}; set a flag the loop tests, or move the `match` into a helper that returns"
                ),
            );
        }
    }

    /// Validates an atomic operation call on a read-write buffer element.
    /// Accepts `atomic_add`, `atomic_min`, `atomic_max`, `atomic_exchange`,
    /// and `atomic_compare_exchange` on `u32`/`uint` and `i32`/`int` elements.
    /// `atomic_min` and `atomic_max` are unsigned-only (Slang/WGSL constraint).
    fn validate_atomic(&mut self, op_name: &str, call: &syn::ExprCall) {
        let expected_args = match op_name {
            "atomic_compare_exchange" => 3,
            _ => 2,
        };
        if call.args.len() != expected_args {
            self.reject(
                Code::AtomicArgumentCount,
                call,
                &format!("`{op_name}` takes exactly {expected_args} arguments"),
            );
            return;
        }
        let syn::Expr::Reference(ref reference) = call.args[0] else {
            self.reject(
                Code::AtomicNotReference,
                &call.args[0],
                &format!("`{op_name}` requires a mutable reference to a buffer element:`&mut buffer[index]`"),
            );
            return;
        };
        if reference.mutability.is_none() {
            self.reject(
                Code::AtomicReferenceNotMutable,
                &call.args[0],
                &format!("`{op_name}` requires a mutable reference:`&mut buffer[index]`"),
            );
            return;
        }
        let syn::Expr::Index(index) = reference.expr.as_ref() else {
            self.reject(
                Code::AtomicOperandNotElement,
                &reference.expr,
                &format!("`{op_name}` requires a read-write buffer element: `&mut buffer[index]`"),
            );
            return;
        };
        let syn::Expr::Path(buffer_path) = index.expr.as_ref() else {
            self.reject(
                Code::AtomicReceiverNotBuffer,
                &index.expr,
                &format!("`{op_name}` requires a direct buffer receiver: `&mut buffer[index]`"),
            );
            return;
        };
        let buffer_name = buffer_path.path.get_ident().map(ToString::to_string);
        let element = match buffer_name
            .as_ref()
            .and_then(|name| self.read_write_buffers.get(name))
        {
            Some(element) => element,
            None => {
                self.reject(
                    Code::AtomicBufferParameter,
                    &index.expr,
                    &format!("`{op_name}` requires a read-write buffer parameter: `RWStructuredBuffer<u32>` or `RWStructuredBuffer<i32>`"),
                );
                return;
            }
        };
        let is_unsigned = element == "u32" || element == "uint";
        let is_signed = element == "i32" || element == "int";
        if !is_unsigned && !is_signed {
            self.reject(
                Code::AtomicElementType,
                &index.expr,
                &format!("`{op_name}` requires a `u32`/`uint` or `i32`/`int` element; other element types are not supported for atomics"),
            );
            return;
        }
        if (op_name == "atomic_min" || op_name == "atomic_max") && !is_unsigned {
            self.reject(
                Code::AtomicSignedMinMax,
                &index.expr,
                &format!("`{op_name}` requires an unsigned element (`u32`/`uint`); signed atomics for min/max are not supported"),
            );
            return;
        }
        self.visit_expr(&index.index);
        for arg in call.args.iter().skip(1) {
            self.visit_expr(arg);
        }
    }
}

impl<'ast> Visit<'ast> for RestrictedVisitor {
    fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
        if pattern.by_ref.is_some() || pattern.subpat.is_some() {
            self.reject(
                Code::BindingMode,
                pattern,
                "reference and @ binding modes are not supported in GPU code",
            );
            return;
        }
        visit::visit_pat_ident(self, pattern);
    }

    fn visit_attribute(&mut self, attribute: &'ast syn::Attribute) {
        if let Some(message) = attribute_refusal(attribute) {
            self.reject(Code::UnsupportedAttribute, attribute, message);
        }
    }

    fn visit_ident(&mut self, ident: &'ast syn::Ident) {
        // The unrawed text, because emission strips a raw prefix: `r#__gust_r_new` would otherwise print the name that
        // renaming gives `new`.
        let name = ident.unraw().to_string();
        if name.starts_with("__gust_") || name.starts_with("__gpu_") || name.starts_with("__Gust") {
            self.reject(
                Code::ReservedPrefix,
                ident,
                "the __gust_, __gpu_ and __Gust prefixes are reserved for generated shader symbols",
            );
        }
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        // `Self` is the name of the struct of the enclosing `impl` (slice m9.a2): alone, as a type or a struct literal.
        // `Self::f(..)` is judged by `visit_expr_call`; any other path through `Self` has no lowering.
        if path
            .segments
            .first()
            .is_some_and(|segment| segment.ident == "Self")
            && (self.self_struct.is_none() || path.segments.len() != 1)
        {
            let message = if self.self_struct.is_none() {
                "`Self` is the type of an `impl` block: outside one, write the name of the struct"
            } else {
                "`Self` is accepted as a type, in a struct literal and in the call `Self::f(..)` of an associated function: other paths through it have no lowering"
            };
            self.reject(Code::UnsupportedModuleItem, path, message);
            return;
        }
        for segment in &path.segments {
            if BANNED_NAMES.contains(&segment.ident.to_string().as_str()) {
                self.reject(
                    Code::CpuOnlyName,
                    path,
                    "CPU-only library/type is not available inside a #[gpu] module",
                );
                return;
            }
        }
        if is_qualified_math_path(path) {
            self.reject(Code::QualifiedMathPath, path, QUALIFIED_MATH_PATH);
            return;
        }
        visit::visit_path(self, path);
    }

    fn visit_type_path(&mut self, ty: &'ast syn::TypePath) {
        use crate::slang::{ParameterFlavor, parameter_flavor};

        // `Workgroup<[T; N]>` stands only as a kernel parameter type, and `validate_function` checked its shape.
        if ty.qself.is_none()
            && ty
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == "Workgroup")
        {
            if !self
                .workgroup_types
                .contains(&(std::ptr::from_ref(ty) as usize))
            {
                self.reject(
                    Code::WorkgroupShape,
                    ty,
                    "`Workgroup<[T; N]>` is the type of a kernel parameter only: not a local, a field, a helper \
                     parameter or a result",
                );
            }
            return;
        }
        // A texture or sampler type stands as a kernel parameter type only (`validate_function` registered those).
        if ty.qself.is_none()
            && parameter_flavor(&syn::Type::Path(ty.clone())).is_texture_or_sampler()
        {
            if !self
                .texture_types
                .contains(&(std::ptr::from_ref(ty) as usize))
            {
                self.reject(
                    Code::TextureUse,
                    ty,
                    "a texture or sampler type is the type of a kernel parameter only: not a local, a field, a helper \
                     parameter or a result",
                );
            }
            return;
        }
        if (option_payload(ty).is_some() || result_payload(ty).is_some()) && self.mentions_enum(ty)
        {
            self.reject(
                Code::EnumInResource,
                ty,
                "an enum is not the payload of an `Option` or a `Result`: it has no `GpuPod` layout and no proven `Option` lowering; return the enum from a helper, or its integer",
            );
            return;
        }
        if let Some(payload) = option_payload(ty) {
            if let Some(context) = self.option_forbidden {
                self.reject(
                    Code::OptionLayout,
                    ty,
                    &format!(
                        "`Option` is not supported in {context}: it has no proven storage layout; keep Option values in locals and helper functions"
                    ),
                );
                return;
            }
            let supported = payload.is_some_and(|payload| match payload {
                syn::Type::Path(inner) => {
                    inner.path.get_ident().is_some()
                        && option_payload(inner).is_none()
                        && parameter_flavor(payload) == ParameterFlavor::Value
                }
                _ => false,
            });
            if !supported {
                self.reject(
                    Code::OptionPayload,
                    ty,
                    "`Option` payload must be a supported scalar or a module struct; nested Options and resources are not lowered",
                );
                return;
            }
            visit::visit_type_path(self, ty);
            return;
        }
        if let Some(payloads) = result_payload(ty) {
            if let Some(context) = self.result_forbidden {
                self.reject(
                    Code::ResultLayout,
                    ty,
                    &format!(
                        "`Result` is not supported in {context}: it has no proven storage layout; keep Result values in locals and helper functions"
                    ),
                );
                return;
            }
            let supported = payloads.is_some_and(|(ok, err)| {
                ok.to_token_stream().to_string() == err.to_token_stream().to_string()
                    && matches!(ok, syn::Type::Path(inner) if inner.path.get_ident().is_some()
                        && option_payload(inner).is_none()
                        && result_payload(inner).is_none()
                        && parameter_flavor(ok) == ParameterFlavor::Value)
            });
            if !supported {
                self.reject(
                    Code::ResultShape,
                    ty,
                    "`Result` is supported only as `Result<T, T>` with a supported scalar or module-struct payload; mixed, nested, and resource payloads are not lowered",
                );
                return;
            }
            visit::visit_type_path(self, ty);
            return;
        }
        if self.option_forbidden.is_some()
            && let Some(ident) = ty.path.get_ident()
            && let Some(math) = math_types::math_type(&ident.unraw().to_string())
            && !math.buffer
        {
            self.reject(
                Code::ThreeWideInBuffer,
                ty,
                &format!(
                    "`{}` cannot be a struct field or a resource element: 3-wide types have stride 16 in a buffer and no Rust layout that matches in every position; use the 4-wide type or scalars in buffers (`{}` stays fine as a local, a helper parameter or a helper result)",
                    math.name, math.name
                ),
            );
            return;
        }
        if let Some(ident) = ty.path.get_ident() {
            let name = ident.to_string();
            if name == "usize" {
                self.reject(
                    Code::UsizeOutsideIndex,
                    ty,
                    "`usize` is the type of an array index in Rust and has no GPU type: it appears only as `i as usize` inside the brackets of an index (`a[i as usize]`); keep the value in a `u32`",
                );
                return;
            }
            if UNSUPPORTED_PRIMITIVES.contains(&name.as_str()) {
                self.reject(
                    Code::UnsupportedPrimitive,
                    ty,
                    &format!(
                        "`{name}` is not a supported GPU type; use the 32-bit scalars `f32`/`float`, `i32`/`int`, `u32`/`uint`, or `bool`"
                    ),
                );
                return;
            }
        }
        let is_resource = matches!(
            parameter_flavor(&syn::Type::Path(ty.clone())),
            ParameterFlavor::StorageRead
                | ParameterFlavor::StorageReadWrite
                | ParameterFlavor::Uniform
        );
        if is_resource {
            let previous = self.option_forbidden.replace("resource element types");
            let previous_result = self.result_forbidden.replace("resource element types");
            visit::visit_type_path(self, ty);
            self.option_forbidden = previous;
            self.result_forbidden = previous_result;
            return;
        }
        visit::visit_type_path(self, ty);
    }

    fn visit_expr_cast(&mut self, cast: &'ast syn::ExprCast) {
        // `i as usize` is the index of an array (slice m9.r1) and nothing else: the dialect has no `usize` value.
        if matches!(strip_type_parens(&cast.ty), syn::Type::Path(path)
            if path.qself.is_none() && path.path.is_ident("usize"))
        {
            if self.index_depth == 0 {
                self.reject(
                    Code::UsizeOutsideIndex,
                    cast,
                    "`as usize` is accepted inside the brackets of an array index only (`a[i as usize]`): the dialect has no `usize` value, so keep the number in a `u32`",
                );
                return;
            }
            if self.enum_of(&cast.expr).is_some() {
                self.reject(
                    Code::EnumShape,
                    cast,
                    "an enum casts to its repr type only: write `e as u32 as usize` in an index",
                );
                return;
            }
            self.visit_expr(&cast.expr);
            return;
        }
        // `a.len() as u32`: the one place the `usize` length of an array may stand outside an index.
        if let syn::Expr::MethodCall(call) = strip_parens(&cast.expr)
            && call.method == "len"
        {
            self.len_casts.insert(std::ptr::from_ref(call) as usize);
        }
        if let Some(enum_name) = self.enum_of(&cast.expr) {
            let repr = self.methods.enums[&enum_name].repr_name();
            let target = match cast.ty.as_ref() {
                syn::Type::Path(path) => path.path.get_ident().map(|ident| ident.to_string()),
                _ => None,
            };
            let matches_repr = target.as_deref().is_some_and(|target| {
                target == repr
                    || (repr == "u32" && target == "uint")
                    || (repr == "i32" && target == "int")
            });
            if !matches_repr {
                self.reject(
                    Code::EnumShape,
                    &cast.ty,
                    &format!(
                        "an enum casts to its repr type only: `{enum_name}` is `#[repr({repr})]`, so write `as {repr}`"
                    ),
                );
                return;
            }
            self.visit_expr(&cast.expr);
            return;
        }
        let supported = match cast.ty.as_ref() {
            syn::Type::Path(path) => path
                .path
                .get_ident()
                .is_some_and(|ident| CAST_TARGETS.contains(&ident.to_string().as_str())),
            _ => false,
        };
        if !supported {
            self.reject(
                Code::UnsupportedCastTarget,
                &cast.ty,
                "casts are supported only to the 32-bit scalar types `f32`/`float`, `i32`/`int`, and `u32`/`uint`",
            );
            return;
        }
        visit::visit_expr_cast(self, cast);
    }

    fn visit_expr_const(&mut self, expression: &'ast syn::ExprConst) {
        self.reject(
            Code::ConstBlock,
            expression,
            "const blocks are not in the Rust-to-Slang subset; write the value directly",
        );
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if matches!(strip_parens(&call.func), syn::Expr::Closure(_)) {
            self.reject(
                Code::ClosureValue,
                call,
                "a closure is accepted only as the initializer of a `let` and is then called as `f(a)`: bind it first (`let f = |x: f32| -> f32 { x };` and `f(1.0f32)`); it cannot be called where it is written",
            );
            return;
        }
        // `f(a, b)` where `f` is a closure bound by an earlier `let` (slice m9.c): the arguments are ordinary expressions,
        // evaluated left to right like a helper's. A closure that is named like a dialect callee or a helper is refused
        // by the closure pass (GUST-E0328), so the name decides here.
        if let syn::Expr::Path(path) = call.func.as_ref()
            && path.qself.is_none()
            && let Some(ident) = path.path.get_ident()
            && self.closures.contains(&ident.unraw().to_string())
        {
            for argument in &call.args {
                self.visit_expr(argument);
            }
            return;
        }
        if let syn::Expr::Path(path) = call.func.as_ref() {
            let contains_banned_name = path
                .path
                .segments
                .iter()
                .any(|segment| BANNED_NAMES.contains(&segment.ident.to_string().as_str()));
            if contains_banned_name {
                self.reject(
                    Code::CpuOnlyName,
                    path,
                    "CPU-only library/type is not available inside a #[gpu] module",
                );
                return;
            }
            if is_qualified_math_path(&path.path) {
                self.reject(Code::QualifiedMathPath, path, QUALIFIED_MATH_PATH);
                return;
            }
            if path.path.is_ident("Some") {
                if call.args.len() != 1 {
                    self.reject(
                        Code::SomeArguments,
                        call,
                        "`Some` takes exactly one argument",
                    );
                    return;
                }
                if self.is_tuple_expr(&call.args[0]) {
                    self.reject(
                        Code::TupleShape,
                        &call.args[0],
                        "a tuple is not the payload of a `Some`: keep the optional value in a struct with named fields",
                    );
                    return;
                }
                // The callee is the prelude constructor, not a bare path value.
                self.visit_expr(&call.args[0]);
                return;
            }
            if path.path.is_ident("Ok") || path.path.is_ident("Err") {
                if call.args.len() != 1 {
                    self.reject(
                        Code::OkErrArguments,
                        call,
                        "`Ok` and `Err` take exactly one argument",
                    );
                    return;
                }
                if self.is_tuple_expr(&call.args[0]) {
                    self.reject(
                        Code::TupleShape,
                        &call.args[0],
                        "a tuple is not the payload of an `Ok` or `Err`: return a struct with named fields",
                    );
                    return;
                }
                self.visit_expr(&call.args[0]);
                return;
            }
            if is_barrier_call(call) {
                // The statement form is taken in `visit_stmt`; a call that reaches here is part of an expression.
                self.reject(
                    Code::BarrierForm,
                    call,
                    "`workgroup_barrier()` is a statement of its own (`workgroup_barrier();`), not a value or part of \
                     an expression",
                );
                return;
            }
            if let Some(operation) = barrier::subgroup_operation(call) {
                // The position and the capability are judged by `barrier::check`, once the generic visitor passed.
                let arity = barrier::subgroup_arity(operation).unwrap_or(0);
                if call.args.len() != arity {
                    self.reject(
                        Code::BuiltinArguments,
                        call,
                        &format!("`{operation}` takes exactly {arity} argument(s) here",),
                    );
                    return;
                }
                for argument in &call.args {
                    self.visit_expr(argument);
                }
                return;
            }
            if let Some(op) = atomic_operation_name(path) {
                self.reject(
                    Code::AtomicAsValue,
                    call,
                    &format!("`{op}` is supported only as a statement, not as a value"),
                );
                return;
            }
            if let Some((math, member)) = math_types::associated_item(&path.path)
                && path.qself.is_none()
                && member != "IDENTITY"
            {
                let expected = match member {
                    "splat" => 1,
                    _ => math_types::width(math),
                };
                if call.args.len() != expected {
                    self.reject(
                        Code::BuiltinArguments,
                        call,
                        &format!(
                            "`{}::{member}` takes exactly {expected} argument(s) here",
                            math.name
                        ),
                    );
                    return;
                }
                // The callee is the prelude constructor, not a bare path value.
                for argument in &call.args {
                    self.visit_expr(argument);
                }
                return;
            }
        }
        // `S::f(..)` and `Self::f(..)`: an associated function of an `impl` of the module (slice m9.a2). A function with
        // a receiver is called with the receiver written before the dot, so `S::method(x)` is refused.
        if let syn::Expr::Path(path) = call.func.as_ref()
            && path.qself.is_none()
            && path.path.leading_colon.is_none()
            && path.path.segments.len() == 2
            && path
                .path
                .segments
                .iter()
                .all(|segment| segment.arguments.is_none())
        {
            let first = path.path.segments[0].ident.unraw().to_string();
            let member = path.path.segments[1].ident.unraw().to_string();
            let owner = if first == "Self" {
                self.self_struct.clone()
            } else {
                Some(first)
            };
            if let Some(has_receiver) = owner.and_then(|owner| {
                self.methods
                    .members
                    .get(&format!("{owner}::{member}"))
                    .copied()
            }) {
                if has_receiver {
                    self.reject(
                        Code::UnknownCall,
                        call,
                        &format!(
                            "`{member}` takes a receiver: call it as `value.{member}(..)`, the path form is for associated functions without one"
                        ),
                    );
                    return;
                }
                for argument in &call.args {
                    self.visit_expr(argument);
                }
                return;
            }
        }
        let allowed = match call.func.as_ref() {
            syn::Expr::Path(path) => path
                .path
                .get_ident()
                .is_some_and(|ident| self.function_names.contains(&ident.to_string())),
            _ => false,
        };
        if allowed
            && let syn::Expr::Path(path) = call.func.as_ref()
            && let Some(ident) = path.path.get_ident()
            && self.owner_fields.contains(&ident.unraw().to_string())
        {
            self.reject(
                Code::MethodNameClash,
                call,
                &format!(
                    "`{ident}` is a helper function and also a field of `{}`: inside the Slang struct the field would capture the call; rename the helper or the field",
                    self.self_struct.as_deref().unwrap_or_default()
                ),
            );
            return;
        }
        if !allowed {
            self.reject(
                Code::UnknownCall,
                call,
                "only calls to helper functions declared in the same GPU module are allowed, and `S::f(..)` or `Self::f(..)` for an associated function of a struct of the module (the path form `self::S::f(..)` is not accepted: write `S::f(..)`)",
            );
            return;
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_expr_index(&mut self, index: &'ast syn::ExprIndex) {
        if let syn::Expr::Path(path) = index.expr.as_ref()
            && let Some(name) = path.path.get_ident()
            && self.atomic_buffers.contains(&name.to_string())
        {
            self.reject(
                Code::AtomicDirectAccess,
                index,
                "atomic buffer elements cannot be read or written directly; use an atomic operation",
            );
            return;
        }
        // The index operand is where `as usize` and an array `.len()` stand (Rust indexes an array with a `usize`; the
        // dialect has no `usize` value elsewhere, GUST-E0334). The base is visited outside it.
        for attribute in &index.attrs {
            self.visit_attribute(attribute);
        }
        self.visit_expr(&index.expr);
        self.index_depth += 1;
        self.visit_expr(&index.index);
        self.index_depth -= 1;
    }

    /// Only the operators that carry a `usize` value on keep the index operand open: a call, a method call (but the
    /// `.len()` of an array), a field or a block starts a new value, so `a[f(x as usize)]` is E0334.
    fn visit_expr(&mut self, expression: &'ast syn::Expr) {
        if self.index_depth > 0
            && !matches!(
                expression,
                syn::Expr::Paren(_)
                    | syn::Expr::Group(_)
                    | syn::Expr::Binary(_)
                    | syn::Expr::Unary(_)
                    | syn::Expr::Cast(_)
                    | syn::Expr::Lit(_)
                    | syn::Expr::Path(_)
                    | syn::Expr::Index(_)
            )
            && !matches!(expression, syn::Expr::MethodCall(call) if call.method == "len")
        {
            let outer = std::mem::replace(&mut self.index_depth, 0);
            visit::visit_expr(self, expression);
            self.index_depth = outer;
            return;
        }
        visit::visit_expr(self, expression);
    }

    fn visit_type_array(&mut self, ty: &'ast syn::TypeArray) {
        if crate::slang::array_length_literal(&ty.len).is_none() {
            self.reject(
                Code::ArrayLength,
                &ty.len,
                "an array length is an integer literal greater than zero (`[f32; 4]`): not `0`, a constant or an expression",
            );
            return;
        }
        if let Some(problem) = array_element_problem(&ty.elem) {
            self.reject(Code::ArrayElement, &ty.elem, problem);
            return;
        }
        visit::visit_type_array(self, ty);
    }

    fn visit_type_slice(&mut self, ty: &'ast syn::TypeSlice) {
        self.reject(
            Code::HelperResourceParameter,
            ty,
            "slices are not supported: pass a fixed-size array `[T; N]` by value, or the values it holds",
        );
    }

    fn visit_expr_array(&mut self, array: &'ast syn::ExprArray) {
        if array.elems.is_empty() {
            self.reject(
                Code::ArrayLength,
                array,
                "an array has at least one element: `[]` has the length zero",
            );
            return;
        }
        if let Some(nested) = array.elems.iter().find(|element| {
            matches!(
                strip_parens(element),
                syn::Expr::Array(_) | syn::Expr::Repeat(_)
            )
        }) {
            self.reject(
                Code::ArrayElement,
                nested,
                "an array element cannot be an array: nested arrays are not supported, use one flat array and index `i * N + j`",
            );
            return;
        }
        visit::visit_expr_array(self, array);
    }

    fn visit_expr_repeat(&mut self, repeat: &'ast syn::ExprRepeat) {
        if crate::slang::array_length_literal(&repeat.len).is_none() {
            self.reject(
                Code::ArrayLength,
                &repeat.len,
                "the length of `[e; N]` is an integer literal greater than zero: not `0`, a constant or an expression",
            );
            return;
        }
        if matches!(
            strip_parens(&repeat.expr),
            syn::Expr::Array(_) | syn::Expr::Repeat(_)
        ) {
            self.reject(
                Code::ArrayElement,
                &repeat.expr,
                "an array element cannot be an array: nested arrays are not supported, use one flat array and index `i * N + j`",
            );
            return;
        }
        // The length is not an expression of the kernel: only the repeated operand is visited.
        self.visit_expr(&repeat.expr);
    }

    fn visit_expr_assign(&mut self, assign: &'ast syn::ExprAssign) {
        if self.reject_mutating_in_place(&assign.left) {
            return;
        }
        if self.reject_stale_index(&assign.left, &assign.right) {
            return;
        }
        visit::visit_expr_assign(self, assign);
    }

    fn visit_expr_binary(&mut self, binary: &'ast syn::ExprBinary) {
        // A constant shift amount of the bit width (32, every integer of the dialect is 32 bits) or more: rustc refuses
        // it (deny-by-default `arithmetic_overflow`), and naga rejects the WGSL, so the macro refuses it. A run-time
        // amount stays and masks on the GPU (KM-04).
        if matches!(
            binary.op,
            syn::BinOp::Shl(_)
                | syn::BinOp::Shr(_)
                | syn::BinOp::ShlAssign(_)
                | syn::BinOp::ShrAssign(_)
        ) && let Some(amount) = constant_shift_amount(&binary.right)
            && !(0..32).contains(&amount)
        {
            let operator = match binary.op {
                syn::BinOp::Shl(_) | syn::BinOp::ShlAssign(_) => "<<",
                _ => ">>",
            };
            let shown = if amount == i128::MAX {
                "larger than any integer".to_string()
            } else {
                amount.to_string()
            };
            self.reject(
                Code::ShiftAmountOutOfRange,
                &binary.right,
                &format!(
                    "the constant shift amount {shown} is outside 0..32, the width of a 32-bit integer: rustc refuses it (`arithmetic_overflow`) and WGSL refuses it (a constant shift amount must be below the width); use an amount in 0..32, or write the mask yourself (`x {operator} (amount & 31)`) if wrapping is wanted"
                ),
            );
            return;
        }
        // Comparing tuples is a scope choice of slice m9.t, not a limit of Slang (review m9.t S1: slangc lowers `==` and
        // the orderings of a `Tuple` with Rust's semantics, NaN included): a comparison of a visible tuple is refused
        // until a later slice admits it with GPU cases. A tuple behind an unseen name is judged by slangc.
        if matches!(
            binary.op,
            syn::BinOp::Eq(_)
                | syn::BinOp::Ne(_)
                | syn::BinOp::Lt(_)
                | syn::BinOp::Le(_)
                | syn::BinOp::Gt(_)
                | syn::BinOp::Ge(_)
        ) && (self.is_tuple_expr(&binary.left) || self.is_tuple_expr(&binary.right))
        {
            self.reject(
                Code::TupleShape,
                binary,
                "comparisons on tuples are not supported yet: compare the elements (`a.0 == b.0 && a.1 == b.1`)",
            );
            return;
        }
        // The place of an assignment or compound assignment, and the value of `*=` in a module that names a matrix
        // (lowered to `place = __gust_mul(place, value)`), evaluate a `&mut self` call in another order than Rust
        // (GUST-E0421, review m9.a2 M1). A field place with no index, `c.n += c.bump()`, matches Rust and stays (an indexed place does not: see `reject_stale_index`).
        let is_assign_op = matches!(
            binary.op,
            syn::BinOp::AddAssign(_)
                | syn::BinOp::SubAssign(_)
                | syn::BinOp::MulAssign(_)
                | syn::BinOp::DivAssign(_)
                | syn::BinOp::RemAssign(_)
                | syn::BinOp::BitXorAssign(_)
                | syn::BinOp::BitAndAssign(_)
                | syn::BinOp::BitOrAssign(_)
                | syn::BinOp::ShlAssign(_)
                | syn::BinOp::ShrAssign(_)
        );
        if is_assign_op && self.reject_mutating_in_place(&binary.left) {
            return;
        }
        if is_assign_op && self.reject_stale_index(&binary.left, &binary.right) {
            return;
        }
        // `/=`, `%=` and `^=` lower to the `inout` helpers `__gust_div_assign`, `__gust_rem_assign` and
        // `__gust_xor_assign` (no other operator does), and slangc computes the index of an indexed place before the
        // value: `out[c.n] /= c.next()` would change another element than Rust (re-check m9.a2 R1). A place with no
        // index (`e.n /= e.bump()`) has no address to compute early and stays accepted.
        if matches!(
            binary.op,
            syn::BinOp::DivAssign(_) | syn::BinOp::RemAssign(_) | syn::BinOp::BitXorAssign(_)
        ) && place_has_index(&binary.left)
            && let Some(call) = first_mutating_call(&self.methods, &binary.right)
        {
            self.reject(
                Code::ForBoundSideEffect,
                call,
                &format!(
                    "the value of `/=`, `%=` or `^=` on an indexed place calls `.{}(..)`, which takes `&mut self`: these operators lower to an `inout` helper and Slang computes the index before the call, so another element than Rust's would change; call the method first (`let v = c.next(); out[i] /= v;`)",
                    call.method.unraw()
                ),
            );
            return;
        }
        if self.methods.matrix
            && matches!(binary.op, syn::BinOp::MulAssign(_))
            && let Some(call) = first_mutating_call(&self.methods, &binary.right)
        {
            self.reject(
                Code::ForBoundSideEffect,
                call,
                &format!(
                    "the value of `*=` calls `.{}(..)`, which takes `&mut self`: in a module that names a matrix `*=` is lowered to `place = __gust_mul(place, value)`, which reads the place before the call; write `let v = ..; x = x * v;`",
                    call.method.unraw()
                ),
            );
            return;
        }
        let is_div_or_rem = matches!(
            binary.op,
            syn::BinOp::Div(_)
                | syn::BinOp::Rem(_)
                | syn::BinOp::DivAssign(_)
                | syn::BinOp::RemAssign(_)
        );
        if is_div_or_rem && is_literal_zero(&binary.right) {
            self.reject(
                Code::DivisionByLiteralZero,
                binary,
                "integer division or modulo by a literal zero has no portable GPU result; guard the divisor before dividing",
            );
            return;
        }
        visit::visit_expr_binary(self, binary);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if self.check_texture_call(call) {
            return;
        }
        let method = call.method.to_string();
        let in_dialect = matches!(
            method.as_str(),
            "global_id"
                | "len"
                | "is_some"
                | "is_none"
                | "is_ok"
                | "is_err"
                | "unwrap_or"
                | "unwrap"
                | "expect"
        ) || wrapping_arity(&method).is_some()
            || math_types::is_swizzle(&method)
            || math_types::method_arity(&method).is_some();
        if !in_dialect
            && let Some(&facts) = self.methods.by_name.get(&call.method.unraw().to_string())
        {
            self.check_module_method_call(call, facts);
            return;
        }
        // A type argument on a method outside the dialect is that method's problem (E0305, E0325), not a built-in's.
        if call.turbofish.is_some() && in_dialect {
            self.reject(
                Code::BuiltinTypeArguments,
                call,
                "GPU built-ins do not accept type arguments",
            );
            return;
        }
        match method.as_str() {
            "is_some" | "is_none" | "is_ok" | "is_err" if call.args.is_empty() => {
                visit::visit_expr_method_call(self, call);
                return;
            }
            "unwrap_or" if call.args.len() == 1 => {
                visit::visit_expr_method_call(self, call);
                return;
            }
            "is_some" | "is_none" | "is_ok" | "is_err" => {
                self.reject(
                    Code::PredicateArguments,
                    call,
                    "`is_some`, `is_none`, `is_ok`, and `is_err` take no arguments",
                );
                return;
            }
            "unwrap_or" => {
                self.reject(
                    Code::UnwrapOrArguments,
                    call,
                    "`unwrap_or` takes exactly one argument",
                );
                return;
            }
            "unwrap" | "expect" => {
                self.reject(
                    Code::UnwrapPanics,
                    call,
                    "`Option::unwrap`/`expect` panics on `None` in Rust and GPU code cannot panic; use `unwrap_or` or `if let Some(x) = ..`",
                );
                return;
            }
            _ => {}
        }
        let math_arity = if math_types::is_swizzle(&method) {
            Some(0)
        } else {
            math_types::method_arity(&method)
        };
        if let Some(arity) = math_arity {
            if call.args.len() != arity {
                self.reject(
                    Code::BuiltinArguments,
                    call,
                    &format!("`.{method}()` takes exactly {arity} argument(s)"),
                );
                return;
            }
            visit::visit_expr_method_call(self, call);
            return;
        }
        if let Some(arity) = wrapping_arity(&method) {
            if call.args.len() != arity {
                self.reject(
                    Code::BuiltinArguments,
                    call,
                    &format!("`.{method}()` takes exactly {arity} argument(s)"),
                );
                return;
            }
            visit::visit_expr_method_call(self, call);
            return;
        }
        let compatibility_builtin = matches!(method.as_str(), "global_id" | "len");
        if !call.args.is_empty() {
            if compatibility_builtin {
                self.reject(
                    Code::BuiltinArguments,
                    call,
                    &format!("`.{method}()` takes no arguments"),
                );
            } else {
                self.reject(
                    Code::MethodNotInDialect,
                    call,
                    &format!(
                        "the method `.{method}(..)` is not in the dialect; {ACCEPTED_METHODS}"
                    ),
                );
            }
            return;
        }
        if !compatibility_builtin {
            self.reject(
                Code::UnsupportedMethod,
                call,
                &format!("`.{method}()` has no lowering; {ACCEPTED_METHODS}"),
            );
            return;
        }
        let receiver = match call.receiver.as_ref() {
            syn::Expr::Path(path) => path.path.get_ident().map(ToString::to_string),
            _ => None,
        };
        // The `.len()` of an array is its `N` (slice m9.r1): a `usize` in Rust, so it stands inside an index or directly
        // under a cast to a 32-bit scalar.
        // Slice m9.r2a: the receiver may also be an array field (`s.items`, `els[i].items`, judged by the field name) or
        // an element of a buffer of arrays (`out[i]`).
        if method == "len"
            && let Some(length) = crate::slang::array_place_length(
                &call.receiver,
                &self.array_lens,
                &self.methods.array_fields,
            )
        {
            let name = call
                .receiver
                .to_token_stream()
                .to_string()
                .replace(" [", "[");
            // The emitter writes the constant `N` and drops the receiver, so a receiver that runs code (a call or a
            // method call, anywhere in its index or base) would run on the CPU twin and not on the GPU: refuse it
            // (fix round 1 of m9.r2a, M1). E0306 because the receiver is not one the length can be read from.
            if receiver.is_none() && len_receiver_runs_code(&call.receiver) {
                self.reject(
                    Code::LenReceiver,
                    call,
                    &format!(
                        "`{name}.len()` is the constant length of the array, so the call inside `{name}` would not run on the GPU: bind the element to a local first (`let e = ..; e.len()`)"
                    ),
                );
                return;
            }
            if receiver.is_none() {
                self.visit_expr(&call.receiver);
            }
            if length.is_none() {
                self.reject(
                    Code::LenReceiver,
                    call,
                    &format!(
                        "`{name}` is bound more than once in this function with another length or as something else (for a field: two structs give the field name another length, or one gives it no array), so `.len()` cannot say which length it means: give each array its own name"
                    ),
                );
            } else if self.index_depth == 0
                && !self
                    .len_casts
                    .contains(&(std::ptr::from_ref(call) as usize))
            {
                self.reject(
                    Code::UsizeOutsideIndex,
                    call,
                    &format!(
                        "`{name}.len()` is a `usize` in Rust, which the dialect has no value for: use it inside an index (`a[a.len() - 1]`) or cast it (`{name}.len() as u32`)"
                    ),
                );
            }
            return;
        }
        let valid = receiver.as_ref().is_some_and(|name| {
            if method == "len" {
                self.resources.contains(name)
            } else {
                self.invocations.contains(name)
            }
        });
        if !valid {
            if method == "len" {
                self.reject(
                    Code::LenReceiver,
                    call,
                    "`.len()` requires a direct storage parameter receiver, or an array local or parameter with a written type (`let g: [f32; 4] = ..;`)",
                );
            } else {
                self.reject(
                    Code::GlobalIdReceiver,
                    call,
                    "`.global_id()` requires a direct dispatch-thread parameter receiver",
                );
            }
            return;
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_field(&mut self, field: &'ast syn::ExprField) {
        // `params.scale` and `params.view.mul_vec4(p)` read one field of a uniform: the base is not a whole-value use.
        if self.is_uniform_parameter(&field.base) {
            // `u.0` is the wrapper's own field: it reaches the whole struct (and the twin could write through it).
            if matches!(field.member, syn::Member::Unnamed(_)) {
                self.reject(
                    Code::UniformWholeValue,
                    field,
                    "a uniform parameter is read one field at a time by name (`params.scale`); `.0` reaches the \
                     whole struct",
                );
            }
            return;
        }
        // `t.0` .. `t.7`: a tuple has at most eight elements (a tuple struct is refused as GUST-E0104, so an unnamed
        // field is a tuple element).
        if let syn::Member::Unnamed(index) = &field.member
            && index.index as usize >= MAX_TUPLE_ELEMENTS
        {
            self.reject(
                Code::TupleShape,
                field,
                &format!(
                    "a tuple has at most {MAX_TUPLE_ELEMENTS} elements, so `.{}` names none",
                    index.index
                ),
            );
            return;
        }
        visit::visit_expr_field(self, field);
    }

    fn visit_type_tuple(&mut self, ty: &'ast syn::TypeTuple) {
        // `()` is the unit type: the result of a function without a value.
        if ty.elems.is_empty() {
            return;
        }
        if ty.elems.len() < 2 || ty.elems.len() > MAX_TUPLE_ELEMENTS {
            self.reject(
                Code::TupleShape,
                ty,
                &format!(
                    "a tuple type has 2 to {MAX_TUPLE_ELEMENTS} elements, this one has {}: group the values in a struct with named fields",
                    ty.elems.len()
                ),
            );
            return;
        }
        for element in &ty.elems {
            if crate::slang::parameter_flavor(strip_type_parens(element))
                != crate::slang::ParameterFlavor::Value
            {
                self.reject(
                    Code::TupleInResource,
                    element,
                    "a tuple element must be a value type: a buffer, uniform, workgroup array, texture, sampler or invocation id is a resource of the kernel, not a value to carry in a tuple",
                );
                return;
            }
            if let Some(problem) = tuple_element_problem(element) {
                self.reject(Code::TupleShape, element, problem);
                return;
            }
        }
        visit::visit_type_tuple(self, ty);
    }

    fn visit_expr_tuple(&mut self, tuple: &'ast syn::ExprTuple) {
        if tuple.elems.is_empty() {
            visit::visit_expr_tuple(self, tuple);
            return;
        }
        if tuple.elems.len() < 2 || tuple.elems.len() > MAX_TUPLE_ELEMENTS {
            self.reject(
                Code::TupleShape,
                tuple,
                &format!(
                    "a tuple has 2 to {MAX_TUPLE_ELEMENTS} elements, this one has {}: group the values in a struct with named fields",
                    tuple.elems.len()
                ),
            );
            return;
        }
        // A resource parameter by name is no value either (the values of a buffer element are read through it).
        if let Some(resource) = tuple.elems.iter().find(|element| {
            matches!(strip_parens(element), syn::Expr::Path(path)
            if path.qself.is_none()
                && path.path.get_ident().is_some_and(|name| {
                    let name = name.unraw().to_string();
                    self.resources.contains(&name)
                        || self.uniforms.contains(&name)
                        || self.textures.contains_key(&name)
                        || self.samplers.contains(&name)
                }))
        }) {
            self.reject(
                Code::TupleInResource,
                resource,
                "a tuple element must be a value type: a buffer, uniform, texture or sampler parameter is a resource of the kernel, not a value to carry in a tuple",
            );
            return;
        }
        // Slang types `makeTuple(1, x)` with an `int` element where rustc may infer `u32` from the use of the tuple.
        if let Some(literal) = tuple.elems.iter().find_map(unsuffixed_literal_only) {
            self.reject(
                Code::UnsuffixedLiteralLet,
                literal,
                "an integer literal without a suffix in a tuple gives the element no type of its own: Slang types it `int` while rustc infers the type from the later uses, so the two can differ; add a suffix (`1u32`, `-1i32`)",
            );
            return;
        }
        visit::visit_expr_tuple(self, tuple);
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        // A texture or a sampler is the receiver of a texture method (or the sampler argument of `sample`) and
        // nothing else; those uses never reach this function (`check_texture_call`).
        if path.qself.is_none()
            && let Some(name) = path.path.get_ident().map(ToString::to_string)
            && (self.textures.contains_key(&name) || self.samplers.contains(&name))
        {
            let uses = if self.samplers.contains(&name) {
                format!(
                    "`{name}` is a sampler parameter: it is used as `texture.sample({name}, u, v)` only"
                )
            } else {
                format!(
                    "`{name}` is a texture parameter: it is used as `{name}.load(x, y)`, \
                     `{name}.store(x, y, v)`, `{name}.sample(sampler, u, v)`, `{name}.width()` or \
                     `{name}.height()` only"
                )
            };
            self.reject(
                Code::TextureUse,
                path,
                &format!("{uses}; it cannot be copied, passed or returned"),
            );
            return;
        }
        // Any other use of a uniform parameter by name copies or passes the whole struct: slangc crashes on a whole
        // struct copy out of a constant buffer on WGSL, and the twin would accept what the GPU cannot compile.
        if path.qself.is_none()
            && path
                .path
                .get_ident()
                .is_some_and(|name| self.uniforms.contains(&name.to_string()))
        {
            self.reject(
                Code::UniformWholeValue,
                path,
                "a uniform parameter is read one field at a time (`params.scale`, `params.view.mul_vec4(p)`); \
                 copying or passing the whole struct (`*params`, `let p = params;`) is not supported",
            );
            return;
        }
        // A closure is called and is nothing else (slice m9.c); a call reaches `visit_expr_call`, not this function.
        if path.qself.is_none()
            && let Some(name) = path.path.get_ident().map(|name| name.unraw().to_string())
            && self.closures.contains(&name)
        {
            self.reject(
                Code::ClosureValue,
                path,
                &format!(
                    "`{name}` is a closure: it can only be called, `{name}(a, b)`; it cannot be passed, returned, stored, \
                     copied or compared"
                ),
            );
            return;
        }
        // `E::V` names a variant of an enum of the module (slice m9.e1).
        if path.qself.is_none() && self.variant_path(&path.path).is_some() {
            return;
        }
        // A bare path value must be a single identifier (a local variable or
        // parameter). Multi-segment paths are associated constants or foreign
        // items (e.g. `f32::INFINITY`, `u32::MAX`), which have no Slang lowering.
        // Callee paths are handled by `visit_expr_call`, which only descends for
        // single-identifier helpers, so this fires only for bare values.
        if path.qself.is_none()
            && let Some((_, "IDENTITY")) = math_types::associated_item(&path.path)
        {
            return;
        }
        if path.qself.is_none() && path.path.is_ident("workgroup_barrier") {
            self.reject(
                Code::BarrierForm,
                path,
                "`workgroup_barrier` is called as a statement (`workgroup_barrier();`), not used as a value",
            );
        } else if is_qualified_math_path(&path.path) {
            self.reject(Code::QualifiedMathPath, path, QUALIFIED_MATH_PATH);
        } else if path.path.get_ident().is_none() {
            self.reject(
                Code::ForeignPath,
                path,
                "associated constants and foreign paths are not in the Rust-to-Slang subset; reference local variables, parameters, or same-module helpers",
            );
        } else if path.path.is_ident("Some")
            || path.path.is_ident("Ok")
            || path.path.is_ident("Err")
        {
            self.reject(
                Code::VariantAsValue,
                path,
                "`Some`, `Ok`, and `Err` must be called with one argument",
            );
        }
    }

    fn visit_expr_if(&mut self, expression: &'ast syn::ExprIf) {
        let syn::Expr::Let(binding) = expression.cond.as_ref() else {
            visit::visit_expr_if(self, expression);
            return;
        };
        let name = option_pattern(&binding.pat).or_else(|| result_pattern(&binding.pat));
        let Some(name) = name else {
            self.reject(
                Code::IfLetPattern,
                &binding.pat,
                "`if let` supports only the `Some(identifier)` or `Ok(identifier)` pattern",
            );
            return;
        };
        let text = name.ident.to_string();
        if self.is_parameter_name(&text) {
            self.reject(
                Code::ShadowedParameter,
                name,
                "GPU locals cannot shadow resource or dispatch-thread parameters",
            );
            return;
        }
        // `if let Some(mut v) = Some(7)` binds a local of the scrutinee's payload type, which is as untyped as the
        // initializer of a `let` (GUST-E0326).
        if let Some(literal) = unsuffixed_literal_only(&binding.expr) {
            self.reject(
                Code::UnsuffixedLiteralLet,
                literal,
                "an integer literal without a suffix or a type annotation gives the bound value no type of its own: \
                 Slang types it `int` while rustc infers the type from the later uses, so the two can differ; add a \
                 suffix (`Some(7u32)`)",
            );
            return;
        }
        self.visit_pat_ident(name);
        self.visit_expr(&binding.expr);
        // The bound name is not an enum value inside the block (it may shadow one), and the outer meaning returns after.
        let outer = self.enum_values.clone();
        let outer_tuples = self.tuple_values.clone();
        self.enum_values.remove(&text);
        self.tuple_values.remove(&text);
        self.visit_block(&expression.then_branch);
        self.enum_values = outer;
        self.tuple_values = outer_tuples;
        if let Some((_, alternative)) = &expression.else_branch {
            self.visit_expr(alternative);
        }
    }

    fn visit_expr_match(&mut self, expression: &'ast syn::ExprMatch) {
        // Reached for a `match` anywhere but a statement or the initializer of a `let` (`visit_local`): an argument, an
        // operand, an assigned value, a `return`, a bound.
        if self.reject_tuple_scrutinee(expression) {
            return;
        }
        if is_switch_match(expression) {
            self.reject(
                Code::ValueMatchForm,
                expression,
                "a `match` that gives a value is accepted only as the initializer of a typed `let`: `let x: u32 = match e { E::A => 1u32, _ => 2u32 };`; bind it to a typed local first, then use the local",
            );
            return;
        }
        self.reject(
            Code::MatchAsValue,
            expression,
            "a `match` on a `Result` is supported only as a statement, not as a value",
        );
    }

    fn visit_expr_let(&mut self, expression: &'ast syn::ExprLet) {
        // Reached only outside the direct `if let` condition position.
        self.reject(
            Code::IfLetPosition,
            expression,
            "`if let` is supported only as `if let Some(x) = value { .. }` or `if let Ok(x) = value { .. }`; let chains and other let positions are not",
        );
    }

    fn visit_expr_struct(&mut self, expression: &'ast syn::ExprStruct) {
        if expression.rest.is_some() {
            self.reject(
                Code::StructUpdateSyntax,
                expression,
                "struct update syntax (`..base`) is not supported; list every field explicitly",
            );
            return;
        }
        visit::visit_expr_struct(self, expression);
    }

    fn visit_expr_try(&mut self, expression: &'ast syn::ExprTry) {
        self.reject(
            Code::TryOperator,
            expression,
            "the `?` operator is not in the Rust-to-Slang subset",
        );
    }

    fn visit_expr_unsafe(&mut self, expression: &'ast syn::ExprUnsafe) {
        self.reject(
            Code::UnsafeBlock,
            expression,
            "unsafe blocks are not in the Rust-to-Slang subset",
        );
    }

    fn visit_expr_macro(&mut self, expression: &'ast syn::ExprMacro) {
        self.reject(
            Code::MacroInModule,
            expression,
            "macros cannot run inside a Rust-to-Slang module",
        );
    }

    fn visit_expr_closure(&mut self, expression: &'ast syn::ExprClosure) {
        // Reached for a closure that is not the initializer of a plain `let` (that one is `visit_local`'s): an argument,
        // a result, a field, an operand, a parenthesized initializer (slice m9.c).
        self.reject(
            Code::ClosureValue,
            expression,
            "a closure is accepted only as the initializer of a `let` (`let f = |x: f32| -> f32 { x * 2.0f32 };`) \
             and is then called as `f(a)`: it cannot be passed, returned, stored or written where a value is expected",
        );
    }

    fn visit_expr_loop(&mut self, expression: &'ast syn::ExprLoop) {
        self.reject(
            Code::LoopNotSupported,
            expression,
            "`loop` is not supported: its trip count is not fixed at entry, so a GPU thread could never be proven to terminate; use `for i in start..end`",
        );
    }

    fn visit_expr_for_loop(&mut self, expression: &'ast syn::ExprForLoop) {
        if let Some(label) = &expression.label {
            self.reject(
                Code::LoopLabel,
                label,
                "loop labels are not supported in the Rust-to-Slang subset; restructure with a flag or a helper return",
            );
            return;
        }
        let variable = match loop_variable(&expression.pat) {
            Ok(variable) => variable,
            Err((code, message)) => {
                self.reject(code, &expression.pat, message);
                return;
            }
        };
        if let Some(variable) = variable {
            let name = variable.ident.to_string();
            if self.is_parameter_name(&name) {
                self.reject(
                    Code::ShadowedParameter,
                    variable,
                    "GPU locals cannot shadow resource or dispatch-thread parameters",
                );
                return;
            }
            self.visit_pat_ident(variable);
        }
        let (start, end) = match range_bounds(&expression.expr) {
            Ok(bounds) => bounds,
            Err((code, message)) => {
                self.reject(code, &expression.expr, message);
                return;
            }
        };
        for bound in [start, end] {
            if is_unsuffixed_integer_literal(bound) {
                self.reject(
                    Code::UnsuffixedLoopBound,
                    bound,
                    "integer-literal loop bounds need an explicit `u32`/`i32` suffix so the Slang counter type matches rustc's inference",
                );
                return;
            }
        }
        // The emitted loop evaluates the end bound before the start bound (spike M9.s0 P9, measured on both targets),
        // where Rust evaluates the start first: a bound that changes a value is refused, not reordered (GUST-E0421).
        for bound in [start, end] {
            if let Some(call) = first_mutating_call(&self.methods, bound) {
                self.reject(
                    Code::ForBoundSideEffect,
                    call,
                    &format!(
                        "the bound of a `for` range calls `.{}(..)`, which takes `&mut self`: the Slang loop evaluates its end bound before its start bound, so the order of the effects would differ from Rust's; call the method before the loop and use the value (`let n = c.next(); for i in 0u32..n {{ .. }}`)",
                        call.method.unraw()
                    ),
                );
                return;
            }
            if let Some(call) = first_closure_call(&self.closures, bound) {
                self.reject(
                    Code::ForBoundSideEffect,
                    call,
                    &format!(
                        "the bound of a `for` range calls the closure `{}`: the Slang loop evaluates its end bound before its start bound, and the dialect keeps the bounds of a loop to plain values; compute the value before the loop (`let n = {}(..); for i in 0u32..n {{ .. }}`)",
                        call.func.to_token_stream(),
                        call.func.to_token_stream()
                    ),
                );
                return;
            }
        }
        self.visit_expr(start);
        self.visit_expr(end);
        self.loop_depth += 1;
        let outer = self.enum_values.clone();
        let outer_tuples = self.tuple_values.clone();
        if let Some(variable) = variable {
            self.enum_values.remove(&variable.ident.unraw().to_string());
            self.tuple_values
                .remove(&variable.ident.unraw().to_string());
        }
        self.visit_block(&expression.body);
        self.enum_values = outer;
        self.tuple_values = outer_tuples;
        self.loop_depth -= 1;
    }

    fn visit_block(&mut self, block: &'ast syn::Block) {
        // An enum local lives to the end of its block: a `let` inside it that shadows an outer name is forgotten after.
        let outer = self.enum_values.clone();
        let outer_tuples = self.tuple_values.clone();
        let outer_closures = self.closures.clone();
        visit::visit_block(self, block);
        self.enum_values = outer;
        self.tuple_values = outer_tuples;
        self.closures = outer_closures;
    }

    fn visit_expr_while(&mut self, expression: &'ast syn::ExprWhile) {
        self.reject(
            Code::WhileNotSupported,
            expression,
            "`while` loops are not supported: the trip count is not fixed at entry, so a GPU thread could never be proven to terminate; use `for i in start..end` with a `break`",
        );
    }

    fn visit_expr_range(&mut self, expression: &'ast syn::ExprRange) {
        // `visit_expr_for_loop` reads its bounds directly, so this is reached only
        // for a range used as a value.
        self.reject(
            Code::RangeAsValue,
            expression,
            "ranges are supported only as the iterable of a `for` loop",
        );
    }

    fn visit_stmt(&mut self, statement: &'ast syn::Stmt) {
        match statement {
            syn::Stmt::Expr(syn::Expr::Match(expression), None)
                if is_switch_match(expression)
                    && self.body_tail_match == Some(std::ptr::from_ref(expression) as usize)
                    && expression.arms.iter().any(|arm| match arm.body.as_ref() {
                        syn::Expr::Block(body) => value_tail(&body.block).is_some(),
                        _ => true,
                    }) =>
            {
                self.reject(
                    Code::ValueMatchForm,
                    expression,
                    "a `match` that gives a value is accepted only as the initializer of a typed `let`, not as the tail of a function: `let x: u32 = match e { E::A => 1u32, _ => 2u32 }; x`; bind it to a typed local first, then return the local",
                );
            }
            syn::Stmt::Expr(syn::Expr::Match(expression), _) if is_switch_match(expression) => {
                self.check_switch_match(expression, MatchForm::Statement);
            }
            syn::Stmt::Expr(syn::Expr::Match(expression), semicolon) => {
                if self.reject_tuple_scrutinee(expression) {
                    return;
                }
                let (ok_name, _, err_name, _) = match result_match_arms(expression) {
                    Ok(arms) => arms,
                    Err(error) => {
                        self.push(error);
                        return;
                    }
                };
                if semicolon.is_none() {
                    self.reject(
                        Code::MatchNeedsSemicolon,
                        expression,
                        "Result match is supported only as a statement; terminate it with `;`",
                    );
                    return;
                }
                if !self.is_known_result(&expression.expr) {
                    self.reject(
                        Code::MatchScrutinee,
                        &expression.expr,
                        "statement match is supported only for a known `Result<T, T>` local, parameter, or same-module helper call",
                    );
                    return;
                }
                for name in [ok_name, err_name] {
                    let text = name.ident.to_string();
                    if self.is_parameter_name(&text) {
                        self.reject(
                            Code::ShadowedParameter,
                            name,
                            "GPU locals cannot shadow resource or dispatch-thread parameters",
                        );
                        return;
                    }
                    self.visit_pat_ident(name);
                }
                self.visit_expr(&expression.expr);
                for arm in &expression.arms {
                    let syn::Expr::Block(body) = arm.body.as_ref() else {
                        unreachable!("result_match_arms checked block bodies");
                    };
                    self.visit_block(&body.block);
                }
            }
            syn::Stmt::Expr(syn::Expr::Call(call), semicolon) if is_barrier_call(call) => {
                if !call.args.is_empty() || semicolon.is_none() {
                    self.reject(
                        Code::BarrierForm,
                        call,
                        "`workgroup_barrier()` takes no arguments and is a statement ending in `;`",
                    );
                }
            }
            syn::Stmt::Expr(syn::Expr::Call(call), _) => {
                if let syn::Expr::Path(path) = call.func.as_ref()
                    && let Some(op) = atomic_operation_name(path)
                {
                    self.validate_atomic(op, call);
                    return;
                }
                visit::visit_stmt(self, statement);
            }
            syn::Stmt::Expr(syn::Expr::Break(jump), _) => {
                if let Some(value) = &jump.expr {
                    self.reject(
                        Code::BreakWithValue,
                        value,
                        "`break` with a value is not supported",
                    );
                    return;
                }
                self.check_jump(jump, jump.label.as_ref(), "break");
            }
            syn::Stmt::Expr(syn::Expr::Continue(jump), _) => {
                self.check_jump(jump, jump.label.as_ref(), "continue");
            }
            _ => visit::visit_stmt(self, statement),
        }
    }

    fn visit_expr_break(&mut self, expression: &'ast syn::ExprBreak) {
        // Statement-position jumps are handled in `visit_stmt`.
        self.reject(
            Code::JumpAsValue,
            expression,
            "`break` and `continue` are supported only as statements, not as values",
        );
    }

    fn visit_expr_continue(&mut self, expression: &'ast syn::ExprContinue) {
        self.reject(
            Code::JumpAsValue,
            expression,
            "`break` and `continue` are supported only as statements, not as values",
        );
    }

    fn visit_expr_await(&mut self, expression: &'ast syn::ExprAwait) {
        self.reject(
            Code::Await,
            expression,
            "await is CPU-only and cannot appear in GPU code",
        );
    }

    fn visit_expr_unary(&mut self, expression: &'ast syn::ExprUnary) {
        // `*x` dereferences a reference, and no value of the dialect is one (`&self` is lowered to `this`, which is not
        // a pointer), so Slang would be handed `*this`: refused here with the code of an unsupported expression.
        let before = self
            .error
            .as_ref()
            .map_or(0, |error| error.into_iter().count());
        visit::visit_expr_unary(self, expression);
        // The operand is judged first (`*u` on a whole uniform is GUST-E0321), and the dereference only when nothing
        // in the operand was refused, so one expression gives one report.
        let after = self
            .error
            .as_ref()
            .map_or(0, |error| error.into_iter().count());
        if matches!(expression.op, syn::UnOp::Deref(_)) && before == after {
            self.reject(
                Code::UnsupportedExpression,
                expression,
                "the dereference `*` is not part of the Rust-to-Slang value model: use the value (`self.field`, a local copy) instead of `*self`",
            );
        }
    }

    // The `Self` type of a shorthand receiver is not written in the source: the receiver is judged by its own rules.
    fn visit_receiver(&mut self, _receiver: &'ast syn::Receiver) {}

    fn visit_expr_reference(&mut self, expression: &'ast syn::ExprReference) {
        self.reject(
            Code::Reference,
            expression,
            "references are not part of the Rust-to-Slang value model",
        );
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let Some((tuple, annotation)) = crate::slang::tuple_pattern(&local.pat) {
            self.visit_tuple_local(local, tuple, annotation);
            return;
        }
        if let Ok((pattern, _)) = crate::slang::local_binding(&local.pat) {
            let name = pattern.ident.to_string();
            if self.is_parameter_name(&name) {
                self.reject(
                    Code::ShadowedParameter,
                    local,
                    "GPU locals cannot shadow resource or dispatch-thread parameters",
                );
                return;
            }
            // A local of the name of a value parameter is a binding of its own: `let mut p = p;` is how a `&mut self`
            // method gets a place to work on (the rule is by name, so a scope that ends is not tracked).
            self.value_params.remove(&pattern.ident.unraw().to_string());
        }
        if crate::slang::local_binding(&local.pat).is_err() {
            self.reject(
                Code::LetNeedsIdentifier,
                local,
                "GPU let bindings require a single identifier",
            );
            return;
        }
        if local
            .init
            .as_ref()
            .is_some_and(|initial| initial.diverge.is_some())
        {
            self.reject(
                Code::LetElse,
                local,
                "let-else requires control-flow lowering and is not supported yet",
            );
            return;
        }
        // `let f = |x: T| -> R { .. };` (slice m9.c): the name is bound and the body is not visited here. The closure
        // pass of `validate_function` judges the form, the names, the captures and the body as the body of a helper,
        // once this visit found nothing else.
        if let Some((pattern, closure)) = crate::slang::closure_binding(local) {
            let key = pattern.ident.unraw().to_string();
            if let syn::ReturnType::Type(_, ty) = &closure.output {
                if let Some(name) = enum_type_name(ty, &self.methods.enums) {
                    self.methods.enum_returns.insert(key.clone(), name);
                }
                if is_tuple_type(ty) {
                    self.methods.tuple_returns.insert(key.clone());
                }
                if matches!(ty.as_ref(), syn::Type::Path(path) if result_payload(path).is_some()) {
                    self.result_helpers.insert(key.clone());
                }
            }
            self.enum_values.remove(&key);
            self.tuple_values.remove(&key);
            self.closures.insert(key);
            return;
        }
        if matches!(local.pat, syn::Pat::Ident(_))
            && let Some(literal) = local
                .init
                .as_ref()
                .and_then(|initial| unsuffixed_literal_only(&initial.expr))
        {
            self.reject(
                Code::UnsuffixedLiteralLet,
                literal,
                "an integer literal without a suffix or a type annotation gives the local no type of its own: Slang \
                 types it `int` while rustc infers the type from the later uses, so the two can differ; add a suffix \
                 (`0u32`, `0i32`) or a type annotation (`let x: u32 = 0;`)",
            );
            return;
        }
        if let Some(value_match) = local
            .init
            .as_ref()
            .and_then(|initial| switch_match_of(&initial.expr))
        {
            // The value form (slice m9.e2): the initializer of a typed `let`, and of nothing else.
            if !matches!(local.pat, syn::Pat::Type(_)) {
                self.reject(
                    Code::ValueMatchForm,
                    value_match,
                    "a `match` that gives a value needs a typed `let`, because the Slang local is declared before its `switch` assigns it: `let x: u32 = match e { E::A => 1u32, _ => 2u32 };`",
                );
                if let Ok((binding, _)) = crate::slang::local_binding(&local.pat) {
                    self.enum_values.remove(&binding.ident.unraw().to_string());
                }
                return;
            }
            self.visit_pat(&local.pat);
            self.check_switch_match(value_match, MatchForm::Value);
        } else {
            visit::visit_local(self, local);
        }
        // The enum a local holds (by name; a rebinding to anything else forgets it).
        if let Ok((binding, _)) = crate::slang::local_binding(&local.pat) {
            let annotated = match &local.pat {
                syn::Pat::Type(typed) => enum_type_name(&typed.ty, &self.methods.enums),
                _ => None,
            };
            let held = annotated.or_else(|| {
                local
                    .init
                    .as_ref()
                    .and_then(|initial| self.enum_of(&initial.expr))
            });
            let key = binding.ident.unraw().to_string();
            match held {
                Some(enum_name) => {
                    self.enum_values.insert(key.clone(), enum_name);
                }
                None => {
                    self.enum_values.remove(&key);
                }
            }
            // The tuple a local holds (by name, like the enum): an annotation or a visible tuple initializer.
            let tuple_held = matches!(&local.pat, syn::Pat::Type(typed) if is_tuple_type(&typed.ty))
                || local
                    .init
                    .as_ref()
                    .is_some_and(|initial| self.is_tuple_expr(&initial.expr));
            if tuple_held {
                self.tuple_values.insert(key);
            } else {
                self.tuple_values.remove(&key);
            }
        }
        let (pattern, annotated_result) = match &local.pat {
            syn::Pat::Ident(pattern) => (pattern, false),
            syn::Pat::Type(typed) => {
                let syn::Pat::Ident(pattern) = typed.pat.as_ref() else {
                    return;
                };
                let result = matches!(typed.ty.as_ref(), syn::Type::Path(path) if result_payload(path).is_some());
                (pattern, result)
            }
            _ => return,
        };
        let initialized_result = local
            .init
            .as_ref()
            .is_some_and(|initial| self.is_known_result(&initial.expr));
        if annotated_result || initialized_result {
            self.result_values.insert(pattern.ident.to_string());
        }
    }
}

/// Whether evaluating `expr` can run code (a call, a method call, or any form this check does not know): the receiver
/// of an array `.len()` that does so is refused, because the emitter replaces `.len()` by a constant (M1, fix round 1 of
/// m9.r2a). Names, literals, fields, indexes and plain arithmetic do not.
fn len_receiver_runs_code(expr: &syn::Expr) -> bool {
    match expr {
        syn::Expr::Path(_) | syn::Expr::Lit(_) => false,
        syn::Expr::Paren(inner) => len_receiver_runs_code(&inner.expr),
        syn::Expr::Group(inner) => len_receiver_runs_code(&inner.expr),
        syn::Expr::Field(field) => len_receiver_runs_code(&field.base),
        syn::Expr::Index(index) => {
            len_receiver_runs_code(&index.expr) || len_receiver_runs_code(&index.index)
        }
        syn::Expr::Cast(cast) => len_receiver_runs_code(&cast.expr),
        syn::Expr::Unary(unary) => len_receiver_runs_code(&unary.expr),
        syn::Expr::Binary(binary) => {
            len_receiver_runs_code(&binary.left) || len_receiver_runs_code(&binary.right)
        }
        _ => true,
    }
}
#[cfg(test)]
mod tests {
    use crate::diagnostics::testing::{assert_code, assert_codes, codes_of_result};

    use super::*;

    #[test]
    fn rejects_cpu_allocation() {
        let module: syn::ItemMod = syn::parse_quote! {
            mod bad {
                #[kernel]
                pub fn bad(ctx: Invocation) {
                    let values = Vec::new();
                }
            }
        };
        let error = validate_module(&module).expect_err("Vec must be rejected");
        assert_code(&error, Code::CpuOnlyName);
    }

    #[test]
    fn independent_causes_report_every_code_in_source_order() {
        let module: syn::ItemMod = syn::parse_quote! {
            mod bad {
                #[kernel]
                pub fn bad(ctx: Invocation) {
                    loop { }
                    while true { }
                    let values = Vec::new();
                }
            }
        };
        let error = validate_module(&module).expect_err("three causes must be rejected");
        assert_codes(
            &error,
            &[
                Code::LoopNotSupported,
                Code::WhileNotSupported,
                Code::CpuOnlyName,
            ],
        );
    }

    #[test]
    fn an_accepted_module_reports_no_code() {
        let module: syn::ItemMod = syn::parse_quote! {
            mod good {
                #[kernel]
                pub fn add(ctx: Invocation, mut out: StorageMut<f32>) {
                    let i = ctx.global_id().x;
                    if i < out.len() { out[i] = 1.0; }
                }
            }
        };
        assert!(codes_of_result(&validate_module(&module)).is_empty());
    }

    #[test]
    fn a_module_without_inline_content_and_stray_items_are_coded() {
        let outline: syn::ItemMod = syn::parse_quote! { mod outline; };
        assert_code(
            &validate_module(&outline).expect_err("outline module"),
            Code::InlineModuleRequired,
        );
        let stray: syn::ItemMod = syn::parse_quote! { mod bad { const LIMIT: u32 = 1; } };
        assert_code(
            &validate_module(&stray).expect_err("const item"),
            Code::UnsupportedModuleItem,
        );
    }

    #[test]
    fn signature_restrictions_are_coded() {
        for (module, code) in [
            (
                syn::parse_quote! { mod bad { async fn f() {} } },
                Code::AsyncFunction,
            ),
            (
                syn::parse_quote! { mod bad { unsafe fn f() {} } },
                Code::UnsafeFunction,
            ),
            (
                syn::parse_quote! { mod bad { fn f<T>(x: T) {} } },
                Code::ExternOrGenericFunction,
            ),
            (
                syn::parse_quote! { mod bad { extern "C" fn f() {} } },
                Code::ExternOrGenericFunction,
            ),
            (
                syn::parse_quote! { mod bad { struct Pair(uint, uint); } },
                Code::StructNeedsNamedFields,
            ),
        ] {
            let module: syn::ItemMod = module;
            assert_code(&validate_module(&module).expect_err("rejected"), code);
        }
    }

    #[test]
    fn a_derive_that_does_not_parse_is_wrapped_with_a_code() {
        let module: syn::ItemMod = syn::parse_quote! {
            mod bad {
                #[derive = "Debug"]
                struct Params { value: f32 }
            }
        };
        // `syn` words this failure itself; the code is added in front of it.
        let error = validate_module(&module).expect_err("malformed derive");
        assert_code(&error, Code::MalformedDerive);
    }

    #[test]
    fn accepts_vector_add_subset() {
        let module: syn::ItemMod = syn::parse_quote! {
            mod good {
                #[kernel]
                pub fn add(ctx: Invocation, a: Storage<f32>, b: Storage<f32>, mut out: StorageMut<f32>) {
                    let i = ctx.global_id().x;
                    if i < out.len() {
                        out[i] = a[i] + b[i];
                    }
                }
            }
        };
        validate_module(&module).expect("vector add should validate");
    }

    #[test]
    fn rejects_empty_or_manually_represented_gpu_structs() {
        let empty: syn::ItemMod = syn::parse_quote! {
            mod bad {
                struct Empty {}
            }
        };
        assert_code(
            &validate_module(&empty).expect_err("empty GPU struct must fail"),
            Code::EmptyStruct,
        );

        let represented: syn::ItemMod = syn::parse_quote! {
            mod bad {
                #[repr(C)]
                struct Params { value: f32 }
            }
        };
        assert_code(
            &validate_module(&represented).expect_err("manual representation must fail"),
            Code::ManualRepr,
        );

        let copied: syn::ItemMod = syn::parse_quote! {
            mod bad {
                #[derive(Debug, Copy)]
                struct Params { value: f32 }
            }
        };
        assert_code(
            &validate_module(&copied).expect_err("manual Copy derive must fail"),
            Code::ManualCloneCopy,
        );
    }

    #[test]
    fn accepts_non_abi_derives_on_gpu_structs() {
        let module: syn::ItemMod = syn::parse_quote! {
            mod good {
                #[derive(Debug, PartialEq)]
                struct Params { scale: f32, bias: f32 }
            }
        };
        validate_module(&module).expect("non-ABI derives should remain available");
    }
}
