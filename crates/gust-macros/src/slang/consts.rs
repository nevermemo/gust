//! The module-level `const` items of a `#[gpu]` module (slice m11.g3, D205).
//!
//! The validator (`validate_const`) has already checked the form: `const NAME: T = <literal>;` with `T` one of the four
//! scalars and one literal (`-` and one literal for `i32` and `f32`). Nothing is evaluated here: a constant is one
//! `static const` line with the same literal the Rust item holds, so the host, the twin and the shader read one value.

use std::fmt::Write as _;

use super::{emit_ident, emit_literal, emit_type, integer_literal};
use crate::diagnostics::{self, Code};

/// Writes one `static const <type> <name> = <literal>;` line per constant of `items`, in source order, followed by a
/// blank line when there was at least one. A module without a constant gets no byte.
pub(crate) fn emit_consts(output: &mut String, items: &[syn::Item]) -> syn::Result<()> {
    let mut any = false;
    for item in items {
        let syn::Item::Const(item) = item else {
            continue;
        };
        let (negated, literal) = match item.expr.as_ref() {
            syn::Expr::Unary(unary) => match unary.expr.as_ref() {
                syn::Expr::Lit(literal) => (true, literal),
                _ => return Err(not_a_literal(item)),
            },
            syn::Expr::Lit(literal) => (false, literal),
            _ => return Err(not_a_literal(item)),
        };
        let value = match (&literal.lit, negated) {
            // The sign is written outside the literal, so `-2147483648` keeps its width (`integer_literal` is told).
            (syn::Lit::Int(value), true) if value.suffix() != "f32" => {
                format!("-{}", integer_literal(value, true)?)
            }
            (lit, true) => format!("-{}", emit_literal(lit)?),
            (lit, false) => emit_literal(lit)?,
        };
        writeln!(
            output,
            "static const {} {} = {value};",
            emit_type(&item.ty)?,
            emit_ident(&item.ident)
        )
        .unwrap();
        any = true;
    }
    if any {
        writeln!(output).unwrap();
    }
    Ok(())
}

/// The validator never lets this through; an emitter reached without it says so as a diagnostic, not a panic.
fn not_a_literal(item: &syn::ItemConst) -> syn::Error {
    diagnostics::error(
        Code::ConstForm,
        &item.expr,
        "a module `const` is initialized with one literal",
    )
}
