//! Shapes the validator recognises: `Option` and `Result` payloads and patterns, loop headers and `match` forms.

use super::*;

/// The `Option<T>` shape this dialect lowers to Slang `Optional<T>`: a single
/// segment with exactly one type argument.
pub(super) fn option_payload(ty: &syn::TypePath) -> Option<Option<&syn::Type>> {
    if ty.qself.is_some() || ty.path.segments.len() != 1 {
        return None;
    }
    let segment = &ty.path.segments[0];
    if !crate::names::ident_is(&segment.ident, "Option") {
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
pub(super) fn result_payload(ty: &syn::TypePath) -> Option<Option<(&syn::Type, &syn::Type)>> {
    if ty.qself.is_some() || ty.path.segments.len() != 1 {
        return None;
    }
    let segment = &ty.path.segments[0];
    if !crate::names::ident_is(&segment.ident, "Result") {
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
pub(super) fn rw_buffer_element_type(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    if type_path.qself.is_some() || type_path.path.segments.len() != 1 {
        return None;
    }
    let segment = &type_path.path.segments[0];
    if !crate::names::ident_is(&segment.ident, "RWStructuredBuffer") {
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
    element_path
        .path
        .get_ident()
        .map(|ident| ident.unraw().to_string())
}

/// `Some(identifier)` in an `if let`, the only pattern with a direct Slang lowering.
pub fn option_pattern(pattern: &syn::Pat) -> Option<&syn::PatIdent> {
    let syn::Pat::TupleStruct(tuple) = pattern else {
        return None;
    };
    if !crate::names::path_is(&tuple.path, "Some") || tuple.elems.len() != 1 {
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
    if !crate::names::path_is(&tuple.path, "Ok") || tuple.elems.len() != 1 {
        return None;
    }
    match &tuple.elems[0] {
        syn::Pat::Ident(ident) => Some(ident),
        _ => None,
    }
}

pub(super) fn err_pattern(pattern: &syn::Pat) -> Option<&syn::PatIdent> {
    let syn::Pat::TupleStruct(tuple) = pattern else {
        return None;
    };
    if !crate::names::path_is(&tuple.path, "Err") || tuple.elems.len() != 1 {
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

/// The alternatives of a pattern: `A | B` flattened (parentheses looked through), any other pattern as itself.
pub(super) fn flatten_or<'p>(pattern: &'p syn::Pat, out: &mut Vec<&'p syn::Pat>) {
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
pub(super) fn first_variant_path(pattern: &syn::Pat) -> Option<&syn::Path> {
    let mut alternatives = Vec::new();
    flatten_or(pattern, &mut alternatives);
    alternatives.into_iter().find_map(|pattern| match pattern {
        syn::Pat::Path(path) if path.qself.is_none() => Some(&path.path),
        _ => None,
    })
}

/// Whether some alternative of an arm pattern is a literal (or a range, which is refused with the literals' message).
pub(super) fn has_literal_alternative(pattern: &syn::Pat) -> bool {
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
pub(super) fn switch_match_of(expression: &syn::Expr) -> Option<&syn::ExprMatch> {
    match peel_expr(expression) {
        syn::Expr::Match(expression) if is_switch_match(expression) => Some(expression),
        _ => None,
    }
}

/// The value and the suffix of an integer-literal pattern, or why it is refused.
pub(super) fn integer_pattern(literal: &syn::ExprLit) -> Result<(i128, String), &'static str> {
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
pub(super) fn value_tail(block: &syn::Block) -> Option<&syn::Expr> {
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
pub(super) fn is_compound_assignment(expression: &syn::Expr) -> bool {
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
pub(super) enum MatchForm {
    /// `match e { .. }` as a statement: every arm is a block that ends in a statement.
    Statement,
    /// `let x: T = match e { .. };`, and an arm or a block tail inside one: every arm gives a value.
    Value,
}

/// What a `switch`-`match` is on.
pub(super) enum SwitchOn {
    /// An enum of the module, by name.
    Enum(String),
    /// A `u32` or `i32`: literal patterns.
    Integer,
}
