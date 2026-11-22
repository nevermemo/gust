//! The module-level `const` items of a `#[gpu]` module (slice m11.g3, D205; expressions: slice m11.cexpr, D210).
//!
//! The validator (`validate_const`) has already checked the form: `const NAME: T = <expression>;` with `T` one of the
//! four scalars and the expression made of literals of `T`, earlier constants of `T`, parentheses and the operators
//! `check_const_expr` lists. Nothing is evaluated here: a constant is one `static const` line that holds the same
//! expression the Rust item holds, operator for operator, so the host, the twin and the shader each compute it with
//! their own compiler and read one value. An item that is one literal (or `-` and one literal) is written exactly as
//! in slice m11.g3; a larger expression is fully parenthesised, so no precedence rule of Slang is relied on.

use std::collections::HashMap;
use std::fmt::Write as _;

use syn::ext::IdentExt as _;

use super::{array_length_literal, emit_ident, emit_literal, emit_type, integer_literal};
use crate::diagnostics::{self, Code};
use crate::validate::{ConstScalar, const_binary_operator, const_scalar};

/// Writes one `static const <type> <name> = <expression>;` line per constant of `items`, in source order, followed by
/// a blank line when there was at least one. A module without a constant gets no byte.
pub(crate) fn emit_consts(output: &mut String, items: &[syn::Item]) -> syn::Result<()> {
    let mut any = false;
    for item in items {
        let syn::Item::Const(item) = item else {
            continue;
        };
        if let syn::Type::Array(array) = item.ty.as_ref() {
            emit_array(output, item, array)?;
            any = true;
            continue;
        }
        let Some(scalar) = const_scalar(&item.ty) else {
            return Err(not_a_constant(item));
        };
        let value = emit_const_expr(&item.expr, scalar, true, item)?;
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

/// One `static const <type> <name>[N] = { e0, e1, .. };` line for an array constant (slice m12.carr). Each element is
/// written as the scalar form writes the whole initializer of an item; a repeat `[e; N]` writes its element `N` times.
fn emit_array(
    output: &mut String,
    item: &syn::ItemConst,
    array: &syn::TypeArray,
) -> syn::Result<()> {
    let (Some(scalar), Some(length)) =
        (const_scalar(&array.elem), array_length_literal(&array.len))
    else {
        return Err(not_a_constant(item));
    };
    let elements = match peel(&item.expr) {
        syn::Expr::Array(list) => list
            .elems
            .iter()
            .map(|element| emit_const_expr(element, scalar, true, item))
            .collect::<syn::Result<Vec<_>>>()?,
        syn::Expr::Repeat(repeat) => {
            vec![emit_const_expr(&repeat.expr, scalar, true, item)?; length as usize]
        }
        _ => return Err(not_a_constant(item)),
    };
    writeln!(
        output,
        "static const {} {}[{length}] = {{ {} }};",
        emit_type(&array.elem)?,
        emit_ident(&item.ident),
        elements.join(", ")
    )
    .unwrap();
    Ok(())
}

thread_local! {
    /// The array constants of the module being emitted and their lengths, for `NAME.len()` and the index guard.
    static CONST_ARRAYS: std::cell::RefCell<HashMap<String, u32>> = std::cell::RefCell::new(HashMap::new());
}

/// Sets the array constants of the module for the emission of its kernels and restores the previous table when dropped.
pub(super) struct ConstArrayScope(HashMap<String, u32>);

impl ConstArrayScope {
    pub(super) fn enter(items: &[syn::Item]) -> Self {
        let next = crate::validate::const_arrays::array_constants(items);
        Self(CONST_ARRAYS.with(|cell| cell.replace(next)))
    }
}

impl Drop for ConstArrayScope {
    fn drop(&mut self) {
        CONST_ARRAYS.with(|cell| *cell.borrow_mut() = std::mem::take(&mut self.0));
    }
}

/// The length of `place` when it is the bare name of an array constant (it stands where a local array's table says the
/// same: the index guard of the SPIR-V route and `.len()`).
pub(super) fn array_length_of(place: &syn::Expr) -> Option<u32> {
    let syn::Expr::Path(path) = peel(place) else {
        return None;
    };
    let name = path.qself.is_none().then(|| path.path.get_ident())??;
    CONST_ARRAYS.with(|cell| cell.borrow().get(&name.unraw().to_string()).copied())
}

/// The Slang text of the expression. `top`: the whole initializer, where one literal and `-` with one literal are
/// written bare (as in slice m11.g3); every other operator application is wrapped in parentheses.
fn emit_const_expr(
    expr: &syn::Expr,
    scalar: ConstScalar,
    top: bool,
    item: &syn::ItemConst,
) -> syn::Result<String> {
    match expr {
        syn::Expr::Paren(inner) => emit_const_expr(&inner.expr, scalar, top, item),
        syn::Expr::Group(inner) => emit_const_expr(&inner.expr, scalar, top, item),
        syn::Expr::Lit(literal) => emit_operand_literal(&literal.lit, scalar, false, top),
        syn::Expr::Path(path) => match path.path.get_ident() {
            Some(ident) if path.qself.is_none() => Ok(emit_ident(ident)),
            _ => Err(not_a_constant(item)),
        },
        syn::Expr::Unary(unary) => match unary.op {
            syn::UnOp::Neg(_) => {
                if let syn::Expr::Lit(literal) = peel(&unary.expr) {
                    // The bare `-literal` of slice m11.g3 stays bare; inside an expression it is one parenthesised term.
                    let text = format!(
                        "-{}",
                        emit_operand_literal(&literal.lit, scalar, true, true)?
                    );
                    return Ok(if top { text } else { format!("({text})") });
                }
                let operand = emit_const_expr(&unary.expr, scalar, false, item)?;
                Ok(format!("(-{operand})"))
            }
            syn::UnOp::Not(_) => {
                let operand = emit_const_expr(&unary.expr, scalar, false, item)?;
                let sigil = if scalar == ConstScalar::Bool {
                    "!"
                } else {
                    "~"
                };
                Ok(format!("({sigil}{operand})"))
            }
            _ => Err(not_a_constant(item)),
        },
        syn::Expr::Binary(binary) => {
            let Some(operator) = const_binary_operator(scalar, &binary.op) else {
                return Err(not_a_constant(item));
            };
            let left = emit_const_expr(&binary.left, scalar, false, item)?;
            // The amount of a shift is a `u32` whatever the type of the item.
            let right_scalar = if matches!(operator, "<<" | ">>") {
                ConstScalar::U32
            } else {
                scalar
            };
            let right = emit_const_expr(&binary.right, right_scalar, false, item)?;
            Ok(format!("({left} {operator} {right})"))
        }
        _ => Err(not_a_constant(item)),
    }
}

/// The expression under any parentheses.
pub(crate) fn peel(expr: &syn::Expr) -> &syn::Expr {
    match expr {
        syn::Expr::Paren(inner) => peel(&inner.expr),
        syn::Expr::Group(inner) => peel(&inner.expr),
        other => other,
    }
}

/// One literal operand. As the whole initializer (`top`) it has the spelling of slice m11.g3 (`emit_literal`; below a
/// minus, `integer_literal` is told of the sign so `-2147483648` keeps its width). Inside an expression a `u32` literal
/// is always `uint(..)`: a bare literal is an `int` in Slang, and `(1 << 31) / 2` computed in `int` is not the `u32`
/// Rust computes.
fn emit_operand_literal(
    literal: &syn::Lit,
    scalar: ConstScalar,
    negated: bool,
    top: bool,
) -> syn::Result<String> {
    if !top
        && scalar == ConstScalar::U32
        && let syn::Lit::Int(value) = literal
    {
        return Ok(format!("uint({})", value.base10_digits()));
    }
    match (literal, negated) {
        (syn::Lit::Int(value), true) if value.suffix() != "f32" => integer_literal(value, true),
        (literal, _) => emit_literal(literal),
    }
}

/// The validator never lets this through; an emitter reached without it says so as a diagnostic, not a panic.
fn not_a_constant(item: &syn::ItemConst) -> syn::Error {
    diagnostics::error(
        Code::ConstForm,
        &item.expr,
        "a module `const` is initialized with an expression of literals and earlier constants of its type",
    )
}
