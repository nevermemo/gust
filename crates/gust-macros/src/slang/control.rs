//! Control flow: `if`, `match`, `for`, and the case labels and variant paths they use.

use std::fmt::Write as _;

use super::expressions::emit_expression;
use super::records::LineTable;
use super::statements::emit_statements;
use super::types::emit_ident;
use super::value_if::emit_value_arm;
use crate::diagnostics::{self, Code};

/// `switch (scrutinee) { case E::A: case E::B: { .. break; } default: { .. break; } }` for a `match` on an enum or on an
/// integer (slices m9.e1 and m9.e2, D157). The scrutinee is one expression in the `switch`, so it is evaluated once.
/// An arm that ends in `return` gets no `break` (slangc warns about unreachable code). An enum `match` without a `_`
/// arm makes its last arm the `default` (in any position: rustc proved the arms exhaustive, and WGSL needs the function
/// to return in every case of a `switch`; measured: slangc error 41009 otherwise). An integer `match` always has `_`.
///
/// With `target` (the value form, `let x: T = match e { .. };`) every arm assigns its value to `target`, which the
/// caller declared before the `switch`; the `default` makes that assignment total.
pub(super) fn emit_switch(
    output: &mut String,
    table: &mut LineTable,
    expression: &syn::ExprMatch,
    indent: usize,
    index: usize,
    target: Option<&str>,
) -> syn::Result<()> {
    let padding = "    ".repeat(indent);
    let inner = "    ".repeat(indent + 1);
    let has_wildcard = expression
        .arms
        .iter()
        .any(|arm| matches!(arm.pat, syn::Pat::Wild(_)));
    writeln!(output, "{padding}// @gust construct: match #{index}").unwrap();
    writeln!(
        output,
        "{padding}switch ({})",
        emit_expression(&expression.expr)?
    )
    .unwrap();
    writeln!(output, "{padding}{{").unwrap();
    for (position, arm) in expression.arms.iter().enumerate() {
        let last = position + 1 == expression.arms.len();
        if matches!(arm.pat, syn::Pat::Wild(_)) || (!has_wildcard && last) {
            writeln!(output, "{inner}default:").unwrap();
        } else {
            let mut alternatives = Vec::new();
            flatten_alternatives(&arm.pat, &mut alternatives);
            for pattern in alternatives {
                let label = match pattern {
                    syn::Pat::Path(path) => emit_variant_path(&path.path),
                    syn::Pat::Lit(literal) => emit_case_literal(literal)?,
                    other => {
                        return Err(diagnostics::error(
                            Code::MatchPattern,
                            other,
                            "a `match` arm is `E::A`, an integer literal, `A | B` or `_`",
                        ));
                    }
                };
                writeln!(output, "{inner}case {label}:").unwrap();
            }
        }
        writeln!(output, "{inner}{{").unwrap();
        let diverges = if let Some(target) = target {
            emit_value_arm(output, table, &arm.body, indent + 2, target)?
        } else {
            let syn::Expr::Block(body) = arm.body.as_ref() else {
                return Err(diagnostics::error(
                    Code::MatchArmBlock,
                    &arm.body,
                    "each arm of a statement `match` is a block",
                ));
            };
            emit_statements(output, table, &body.block.stmts, indent + 2, false)?;
            matches!(
                body.block.stmts.last(),
                Some(syn::Stmt::Expr(syn::Expr::Return(_), _))
            )
        };
        if !diverges {
            writeln!(output, "{}break;", "    ".repeat(indent + 2)).unwrap();
        }
        writeln!(output, "{inner}}}").unwrap();
    }
    writeln!(output, "{padding}}}").unwrap();
    Ok(())
}

/// The label of an integer pattern: `0`, `uint(3000000000)`, `int(-2)`. The text follows `integer_literal`'s rules for
/// the type of the literal; a negative value is written as `int(-n)`, which slangc accepts as a case label (measured).
fn emit_case_literal(literal: &syn::ExprLit) -> syn::Result<String> {
    let syn::Lit::Int(value) = &literal.lit else {
        return Err(diagnostics::error(
            Code::MatchPattern,
            literal,
            "only integer literals are patterns",
        ));
    };
    let number: i128 = value.base10_parse().map_err(|_| {
        diagnostics::error(
            Code::MatchPattern,
            literal,
            "this integer literal does not parse",
        )
    })?;
    Ok(match value.suffix() {
        "u32" => format!("uint({number})"),
        "i32" => format!("int({number})"),
        "" if number < 0 => format!("int({number})"),
        "" if number > i128::from(i32::MAX) => format!("uint({number})"),
        "" => number.to_string(),
        _ => {
            return Err(diagnostics::error(
                Code::UnsupportedLiteral,
                literal,
                "GPU literals support 32-bit numbers (i32, u32) only",
            ));
        }
    })
}

/// The alternatives of an arm pattern: `A | B` flattened, parentheses looked through.
fn flatten_alternatives<'p>(pattern: &'p syn::Pat, out: &mut Vec<&'p syn::Pat>) {
    match pattern {
        syn::Pat::Or(or) => {
            for case in &or.cases {
                flatten_alternatives(case, out);
            }
        }
        syn::Pat::Paren(inner) => flatten_alternatives(&inner.pat, out),
        other => out.push(other),
    }
}

/// `E::V` as Slang spells it: both names renamed as their declarations are.
pub(super) fn emit_variant_path(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|segment| emit_ident(&segment.ident))
        .collect::<Vec<_>>()
        .join("::")
}

pub(super) fn emit_if(
    output: &mut String,
    table: &mut LineTable,
    expression: &syn::ExprIf,
    indent: usize,
    return_tail: bool,
    index: usize,
) -> syn::Result<()> {
    let padding = "    ".repeat(indent);
    if let syn::Expr::Let(binding) = expression.cond.as_ref() {
        // `if let Some(x) = value`: evaluate the scrutinee once into a reserved
        // temporary, then bind the payload inside the taken branch.
        let (name, temporary, condition) =
            if let Some(name) = crate::validate::option_pattern(&binding.pat) {
                (name, format!("__gust_opt_{index}"), "hasValue")
            } else if let Some(name) = crate::validate::result_pattern(&binding.pat) {
                (name, format!("__gust_result_{index}"), "isOk")
            } else {
                return Err(diagnostics::error(
                    Code::IfLetPattern,
                    &binding.pat,
                    "`if let` supports only the `Some(identifier)` or `Ok(identifier)` pattern",
                ));
            };
        writeln!(
            output,
            "{padding}var {temporary} = {};",
            emit_expression(&binding.expr)?
        )
        .unwrap();
        writeln!(output, "{padding}if ({temporary}.{condition})").unwrap();
        writeln!(output, "{padding}{{").unwrap();
        writeln!(
            output,
            "{}var {} = {temporary}.value;",
            "    ".repeat(indent + 1),
            emit_ident(&name.ident)
        )
        .unwrap();
    } else {
        writeln!(
            output,
            "{padding}if ({})",
            emit_expression(&expression.cond)?
        )
        .unwrap();
        writeln!(output, "{padding}{{").unwrap();
    }
    emit_statements(
        output,
        table,
        &expression.then_branch.stmts,
        indent + 1,
        return_tail,
    )?;
    writeln!(output, "{padding}}}").unwrap();
    if let Some((_, alternative)) = &expression.else_branch {
        match alternative.as_ref() {
            syn::Expr::If(nested) if !matches!(nested.cond.as_ref(), syn::Expr::Let(_)) => {
                write!(output, "{padding}else ").unwrap();
                emit_if(output, table, nested, indent, return_tail, index)?;
            }
            syn::Expr::If(nested) => {
                // `else if let` needs a scope of its own for the scrutinee temporary.
                writeln!(output, "{padding}else").unwrap();
                writeln!(output, "{padding}{{").unwrap();
                emit_if(output, table, nested, indent + 1, return_tail, index)?;
                writeln!(output, "{padding}}}").unwrap();
            }
            other => {
                writeln!(output, "{padding}else").unwrap();
                writeln!(output, "{padding}{{").unwrap();
                match other {
                    syn::Expr::Block(block) => {
                        emit_statements(output, table, &block.block.stmts, indent + 1, return_tail)?
                    }
                    other => writeln!(
                        output,
                        "{}{};",
                        "    ".repeat(indent + 1),
                        emit_expression(other)?
                    )
                    .unwrap(),
                }
                writeln!(output, "{padding}}}").unwrap();
            }
        }
    }
    Ok(())
}

pub(super) fn emit_result_match(
    output: &mut String,
    table: &mut LineTable,
    expression: &syn::ExprMatch,
    indent: usize,
    index: usize,
) -> syn::Result<()> {
    let (ok_name, ok_body, err_name, err_body) = crate::validate::result_match_arms(expression)?;
    let padding = "    ".repeat(indent);
    let temporary = format!("__gust_match_{index}");
    writeln!(output, "{padding}// @gust construct: match #{index}").unwrap();
    writeln!(
        output,
        "{padding}var {temporary} = {};",
        emit_expression(&expression.expr)?
    )
    .unwrap();
    writeln!(output, "{padding}if ({temporary}.isOk)").unwrap();
    writeln!(output, "{padding}{{").unwrap();
    writeln!(
        output,
        "{}var {} = {temporary}.value;",
        "    ".repeat(indent + 1),
        emit_ident(&ok_name.ident)
    )
    .unwrap();
    emit_statements(output, table, &ok_body.stmts, indent + 1, false)?;
    writeln!(output, "{padding}}}").unwrap();
    writeln!(output, "{padding}else").unwrap();
    writeln!(output, "{padding}{{").unwrap();
    writeln!(
        output,
        "{}var {} = {temporary}.value;",
        "    ".repeat(indent + 1),
        emit_ident(&err_name.ident)
    )
    .unwrap();
    emit_statements(output, table, &err_body.stmts, indent + 1, false)?;
    writeln!(output, "{padding}}}").unwrap();
    Ok(())
}

/// Lower `for v in start..end { .. }` (D18). The end bound is evaluated once into
/// a reserved temporary so the trip count is fixed at entry, as with Rust's
/// `Range`; the counter takes its Slang type from the start expression. The
/// statement index keeps the temporary unique within its block, and a `_`
/// pattern gets a reserved counter name.
pub(super) fn emit_for(
    output: &mut String,
    table: &mut LineTable,
    expression: &syn::ExprForLoop,
    indent: usize,
    index: usize,
) -> syn::Result<()> {
    let padding = "    ".repeat(indent);
    let variable = match crate::validate::loop_variable(&expression.pat) {
        Ok(Some(variable)) => emit_ident(&variable.ident),
        Ok(None) => format!("__gust_iter_{index}"),
        Err((code, message)) => {
            return Err(diagnostics::error(code, &expression.pat, message));
        }
    };
    let (start, end) = crate::validate::range_bounds(&expression.expr)
        .map_err(|(code, message)| diagnostics::error(code, &expression.expr, message))?;
    let end_temporary = format!("__gust_end_{index}");
    writeln!(
        output,
        "{padding}var {end_temporary} = {};",
        emit_expression(end)?
    )
    .unwrap();
    writeln!(
        output,
        "{padding}for (var {variable} = {}; {variable} < {end_temporary}; {variable}++)",
        emit_expression(start)?
    )
    .unwrap();
    writeln!(output, "{padding}{{").unwrap();
    emit_statements(output, table, &expression.body.stmts, indent + 1, false)?;
    writeln!(output, "{padding}}}").unwrap();
    Ok(())
}
