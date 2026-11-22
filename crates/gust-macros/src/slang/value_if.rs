//! `if` and `match` in value position, and the type inference their branches need.

use std::fmt::Write as _;

use quote::ToTokens;
use syn::{ext::IdentExt as _, spanned::Spanned};

use super::control::emit_switch;
use super::expressions::emit_expression;
use super::records::LineTable;
use super::statements::{emit_statements, emit_struct_value};
use super::types::{emit_ident, emit_type};
use crate::{
    diagnostics::{self, Code},
    math_types,
    names::{peel_expr, peel_type},
};

/// `T x;` and the `switch` that assigns it, for `let x: T = match e { .. };` (slice m9.e2). WGSL zero-initialises a `var`
/// declared without a value, and every path of the `switch` assigns it (measured: slangc 2026.21, WGSL and SPIR-V).
/// When the name of the local occurs in the `match` (the `match` reads an outer local of that name, or an arm declares
/// one), the switch assigns a temporary and the local is declared from it after, so the `match` never reads the new
/// local. A shadow in the same Slang scope as the outer local still fails in slangc (E30200, as for every `let`); the
/// shadowing `let` belongs in a block of its own (SEMANTICS.md, section 9).
pub(super) fn emit_value_match(
    output: &mut String,
    table: &mut LineTable,
    name: &syn::Ident,
    annotation: Option<&syn::Type>,
    expression: &syn::ExprMatch,
    indent: usize,
    index: usize,
) -> syn::Result<()> {
    let padding = "    ".repeat(indent);
    let declared = emit_ident(name);
    let ty = match annotation {
        Some(ty) => {
            if matches!(peel_type(ty), syn::Type::Array(_)) {
                return Err(diagnostics::error(
                    Code::UnsupportedExpression,
                    expression,
                    "a `match` does not give an array value (arrays are built by a literal or a repeat, or returned by a helper): use `if`/`else` statements that assign the elements, or a helper that returns the array",
                ));
            }
            emit_type(ty)?
        }
        None => type_of_match_arms(expression).ok_or_else(|| {
            diagnostics::error(
                Code::ValueMatchForm,
                expression,
                "a `match` that gives a value needs a typed `let` when no arm shows its type (`let x: u32 = match e { .. };`); an arm shows it by ending in a suffixed literal (`1u32`, `2.0f32`), a `bool`, a comparison, an `as` cast or a vector constructor",
            )
        })?,
    };
    let target = if mentions_ident(expression.to_token_stream(), &name.unraw().to_string()) {
        format!("__gust_match_{indent}_{index}")
    } else {
        declared.clone()
    };
    writeln!(output, "{padding}{ty} {target};").unwrap();
    emit_switch(output, table, expression, indent, index, Some(&target))?;
    if target != declared {
        writeln!(output, "{padding}{ty} {declared} = {target};").unwrap();
    }
    Ok(())
}

/// `T x;` and the `if` that assigns it, for `let x: T = if c { a } else { b };` (slice m12.ifx). The condition is
/// evaluated once and exactly one branch runs, as in Rust; each branch assigns the local as its last step. The type is
/// the annotation, or the one the tails show (`type_of_if_tails`). When the name of the local occurs in the `if` (the
/// condition or a branch reads an outer local of that name), the branches assign a temporary and the local is declared
/// from it after, so the `if` never reads the new local (as for a value `match`).
pub(super) fn emit_value_if(
    output: &mut String,
    table: &mut LineTable,
    name: &syn::Ident,
    annotation: Option<&syn::Type>,
    expression: &syn::ExprIf,
    indent: usize,
    index: usize,
) -> syn::Result<()> {
    let padding = "    ".repeat(indent);
    check_value_if(expression)?;
    let ty = match annotation {
        Some(ty) => value_if_type(ty)?,
        None => type_of_if_tails(expression).ok_or_else(|| {
            diagnostics::error(
                Code::UnsupportedExpression,
                expression,
                "the type of an `if` value is not visible in its branches: annotate the `let` (`let x: u32 = if c { a } else { b };`), or end a branch in a suffixed literal (`1u32`, `2.0f32`), a `bool`, a comparison, an `as` cast or a vector constructor",
            )
        })?,
    };
    let declared = emit_ident(name);
    let target = if mentions_ident(expression.to_token_stream(), &name.unraw().to_string()) {
        format!("__gust_if_{indent}_{index}")
    } else {
        declared.clone()
    };
    writeln!(output, "{padding}{ty} {target};").unwrap();
    emit_value_if_chain(output, table, expression, indent, &target)?;
    if target != declared {
        writeln!(output, "{padding}{ty} {declared} = {target};").unwrap();
    }
    Ok(())
}

/// The refusals of an `if` value that need no type: every `if` of the chain has an `else`, and none tests a pattern.
fn check_value_if(expression: &syn::ExprIf) -> syn::Result<()> {
    if matches!(expression.cond.as_ref(), syn::Expr::Let(_)) {
        return Err(diagnostics::error(
            Code::UnsupportedExpression,
            &expression.cond,
            "an `if let` does not give a value: write the `if let` as a statement that assigns a local declared before it",
        ));
    }
    match expression
        .else_branch
        .as_ref()
        .map(|(_, other)| other.as_ref())
    {
        None => Err(diagnostics::error(
            Code::UnsupportedExpression,
            expression,
            "an `if` that gives a value needs an `else`: every path assigns the local",
        )),
        Some(syn::Expr::If(nested)) => check_value_if(nested),
        Some(_) => Ok(()),
    }
}

/// The Slang type of an annotated `if` value: a scalar or a vector, nothing else.
fn value_if_type(ty: &syn::Type) -> syn::Result<String> {
    let plain = match peel_type(ty) {
        syn::Type::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => path
            .path
            .segments
            .last()
            .filter(|segment| segment.arguments.is_empty())
            .map(|segment| segment.ident.unraw().to_string()),
        _ => None,
    };
    let accepted = plain.as_deref().is_some_and(|name| {
        matches!(
            name,
            "bool" | "i32" | "u32" | "f32" | "int" | "uint" | "float"
        ) || math_types::math_type(name).is_some_and(|math| !math.matrix)
    });
    if !accepted {
        return Err(diagnostics::error(
            Code::UnsupportedExpression,
            ty,
            "an `if` that gives a value is a `bool`, `i32`, `u32`, `f32` or a vector (`Vec2`, `Vec3`, `Vec4`, `IVec2`, `IVec4`, `UVec2`, `UVec3`, `UVec4`): an array, a struct, a matrix, a tuple or an `Option` is assigned by `if`/`else` statements",
        ));
    }
    emit_type(ty)
}

/// The Slang type the tails of an `if` chain show, when one of them shows it: a suffixed integer or `f32` literal, a
/// `bool`, a comparison, an `as` cast to a scalar, or the constructor of a vector. A `match` value (slice m13.matchu)
/// takes its type from its arms by the same rule ([`type_of_match_arms`]); a branch or arm that ends in `return`,
/// `break` or `continue` shows nothing.
fn type_of_if_tails(expression: &syn::ExprIf) -> Option<String> {
    type_of_block_tail(&expression.then_branch).or_else(|| match &expression.else_branch {
        Some((_, other)) => type_of_tail(other),
        None => None,
    })
}

/// The type the arms of a value `match` show: the first arm that shows one.
fn type_of_match_arms(expression: &syn::ExprMatch) -> Option<String> {
    expression
        .arms
        .iter()
        .find_map(|arm| type_of_tail(&arm.body))
}

/// The type the value of a block shows: its last statement, an expression without `;`.
fn type_of_block_tail(block: &syn::Block) -> Option<String> {
    match block.stmts.last() {
        Some(syn::Stmt::Expr(tail, None)) => type_of_tail(tail),
        _ => None,
    }
}

/// The Slang type a tail expression shows by itself, if it does (see [`type_of_if_tails`]).
fn type_of_tail(tail: &syn::Expr) -> Option<String> {
    match peel_expr(tail) {
        syn::Expr::If(nested) => type_of_if_tails(nested),
        syn::Expr::Match(nested) if crate::validate::is_switch_match(nested) => {
            type_of_match_arms(nested)
        }
        syn::Expr::Block(block) if block.label.is_none() => type_of_block_tail(&block.block),
        syn::Expr::Lit(literal) => match &literal.lit {
            syn::Lit::Int(value) => match value.suffix() {
                "u32" => Some("uint".to_owned()),
                "i32" => Some("int".to_owned()),
                _ => None,
            },
            syn::Lit::Float(value) if value.suffix() == "f32" => Some("float".to_owned()),
            syn::Lit::Bool(_) => Some("bool".to_owned()),
            _ => None,
        },
        syn::Expr::Cast(cast) => value_if_type(&cast.ty).ok(),
        syn::Expr::Call(call) => match call.func.as_ref() {
            syn::Expr::Path(path) => math_types::associated_item(&path.path)
                .filter(|(math, _)| !math.matrix)
                .map(|(math, _)| math.slang.to_owned()),
            _ => None,
        },
        syn::Expr::Unary(unary) => type_of_tail(&unary.expr),
        syn::Expr::Binary(binary) => match binary.op {
            syn::BinOp::Eq(_)
            | syn::BinOp::Ne(_)
            | syn::BinOp::Lt(_)
            | syn::BinOp::Le(_)
            | syn::BinOp::Gt(_)
            | syn::BinOp::Ge(_)
            | syn::BinOp::And(_)
            | syn::BinOp::Or(_) => Some("bool".to_owned()),
            syn::BinOp::Shl(_) | syn::BinOp::Shr(_) => type_of_tail(&binary.left),
            // A scalar beside a vector or a matrix gives the vector or matrix, so the first operand is no guide: the
            // operands must show the same type, or one of them a vector (then that is the type). Anything else gives
            // none, and the `let` is annotated (a matrix result is refused there).
            _ => {
                let left = type_of_tail(&binary.left);
                let right = type_of_tail(&binary.right);
                match (left, right) {
                    (Some(left), Some(right)) if left == right => Some(left),
                    (left, right) => [left, right]
                        .into_iter()
                        .flatten()
                        .find(|ty| is_vector_slang_type(ty)),
                }
            }
        },
        _ => None,
    }
}

/// Whether `slang` is the Slang spelling of a vector type of `gust::math`.
fn is_vector_slang_type(slang: &str) -> bool {
    math_types::MATH_TYPES
        .iter()
        .any(|math| !math.matrix && math.slang == slang)
}

/// The `if`/`else` chain of a value `if`: each branch ends by assigning `target`. An `else if` is an `if` in the
/// `else` block, so each level keeps its own braces.
fn emit_value_if_chain(
    output: &mut String,
    table: &mut LineTable,
    expression: &syn::ExprIf,
    indent: usize,
    target: &str,
) -> syn::Result<()> {
    let padding = "    ".repeat(indent);
    writeln!(
        output,
        "{padding}if ({})",
        emit_expression(&expression.cond)?
    )
    .unwrap();
    writeln!(output, "{padding}{{").unwrap();
    emit_value_if_block(output, table, &expression.then_branch, indent + 1, target)?;
    writeln!(output, "{padding}}}").unwrap();
    let Some((_, alternative)) = &expression.else_branch else {
        unreachable!("an `if` value has an `else` (check_value_if)")
    };
    writeln!(output, "{padding}else").unwrap();
    writeln!(output, "{padding}{{").unwrap();
    match alternative.as_ref() {
        syn::Expr::If(nested) => emit_value_if_chain(output, table, nested, indent + 1, target)?,
        syn::Expr::Block(block) => {
            emit_value_if_block(output, table, &block.block, indent + 1, target)?;
        }
        _ => unreachable!("the `else` of an `if` is a block or an `if`"),
    }
    writeln!(output, "{padding}}}").unwrap();
    Ok(())
}

/// A branch of a value `if`: its statements, then its value assigned to `target`. A tail that is itself an `if` is a
/// chain of its own that assigns the same target.
fn emit_value_if_block(
    output: &mut String,
    table: &mut LineTable,
    block: &syn::Block,
    indent: usize,
    target: &str,
) -> syn::Result<()> {
    if let Some((syn::Stmt::Expr(syn::Expr::If(inner), None), head)) = block.stmts.split_last() {
        check_value_if(inner)?;
        emit_statements(output, table, head, indent, false)?;
        let record = table.begin(output, inner.span());
        emit_value_if_chain(output, table, inner, indent, target)?;
        table.finish_written(output, record);
        return Ok(());
    }
    emit_value_block(output, table, block, indent, target).map(|_| ())
}

/// Whether the identifier `name` occurs anywhere in `tokens`.
pub(super) fn mentions_ident(tokens: proc_macro2::TokenStream, name: &str) -> bool {
    tokens.into_iter().any(|tree| match tree {
        proc_macro2::TokenTree::Ident(ident) => ident.unraw() == name,
        proc_macro2::TokenTree::Group(group) => mentions_ident(group.stream(), name),
        _ => false,
    })
}

/// The body of an arm of a value `match` at `indent`: assigns its value to `target`. Returns whether the arm ends in a
/// `return`, `break` or `continue`, so that it needs no `break`.
pub(super) fn emit_value_arm(
    output: &mut String,
    table: &mut LineTable,
    body: &syn::Expr,
    indent: usize,
    target: &str,
) -> syn::Result<bool> {
    match body {
        syn::Expr::Paren(inner) => emit_value_arm(output, table, &inner.expr, indent, target),
        syn::Expr::Group(inner) => emit_value_arm(output, table, &inner.expr, indent, target),
        syn::Expr::Block(block) if block.label.is_none() => {
            emit_value_block(output, table, &block.block, indent, target)
        }
        other => emit_value_tail(output, table, other, indent, 0, target),
    }
}

/// The statements of a block, then its value (the last statement, an expression without `;`) assigned to `target`.
fn emit_value_block(
    output: &mut String,
    table: &mut LineTable,
    block: &syn::Block,
    indent: usize,
    target: &str,
) -> syn::Result<bool> {
    match block.stmts.split_last() {
        Some((syn::Stmt::Expr(tail, None), head))
            if !matches!(
                tail,
                syn::Expr::Return(_) | syn::Expr::Break(_) | syn::Expr::Continue(_)
            ) =>
        {
            emit_statements(output, table, head, indent, false)?;
            let record = table.begin(output, tail.span());
            let diverges = emit_value_tail(output, table, tail, indent, head.len(), target)?;
            table.finish_written(output, record);
            Ok(diverges)
        }
        _ => {
            emit_statements(output, table, &block.stmts, indent, false)?;
            Ok(matches!(
                block.stmts.last(),
                Some(syn::Stmt::Expr(
                    syn::Expr::Return(_) | syn::Expr::Break(_) | syn::Expr::Continue(_),
                    _
                ))
            ))
        }
    }
}

/// One value assigned to `target`: a nested value `match` is a `switch` of its own that assigns the same target, a
/// struct literal is built in a temporary first (as for a `let`), anything else is `target = value;`.
fn emit_value_tail(
    output: &mut String,
    table: &mut LineTable,
    value: &syn::Expr,
    indent: usize,
    index: usize,
    target: &str,
) -> syn::Result<bool> {
    let padding = "    ".repeat(indent);
    match value {
        syn::Expr::Paren(inner) => {
            emit_value_tail(output, table, &inner.expr, indent, index, target)
        }
        syn::Expr::Group(inner) => {
            emit_value_tail(output, table, &inner.expr, indent, index, target)
        }
        syn::Expr::Match(inner) if crate::validate::is_switch_match(inner) => {
            emit_switch(output, table, inner, indent, index, Some(target))?;
            Ok(false)
        }
        syn::Expr::Block(block) if block.label.is_none() => {
            writeln!(output, "{padding}{{").unwrap();
            let diverges = emit_value_block(output, table, &block.block, indent + 1, target)?;
            writeln!(output, "{padding}}}").unwrap();
            Ok(diverges)
        }
        syn::Expr::Return(_) => {
            writeln!(output, "{padding}{};", emit_expression(value)?).unwrap();
            Ok(true)
        }
        syn::Expr::Struct(literal) => {
            let temporary = emit_struct_value(output, literal, indent, index)?;
            writeln!(output, "{padding}{target} = {temporary};").unwrap();
            Ok(false)
        }
        other => {
            writeln!(output, "{padding}{target} = {};", emit_expression(other)?).unwrap();
            Ok(false)
        }
    }
}
