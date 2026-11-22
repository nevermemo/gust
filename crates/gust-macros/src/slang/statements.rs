//! Statements and `let` bindings, struct values and repeat initialisers.

use std::fmt::Write as _;

use quote::ToTokens;
use syn::{ext::IdentExt as _, spanned::Spanned};

use super::arrays::array_length_literal;
use super::closures::closure_binding;
use super::control::{emit_for, emit_if, emit_result_match, emit_switch};
use super::expressions::{emit_expression, emit_value_expression};
use super::records::LineTable;
use super::types::{emit_ident, emit_type, emit_value_type};
use super::value_if::{emit_value_if, emit_value_match, mentions_ident};
use crate::{
    diagnostics::{self, Code},
    names::{peel_expr, peel_type},
};

pub(super) fn emit_statements(
    output: &mut String,
    table: &mut LineTable,
    statements: &[syn::Stmt],
    indent: usize,
    return_tail: bool,
) -> syn::Result<()> {
    for (index, statement) in statements.iter().enumerate() {
        // A closure binding emits nothing: its body was hoisted before the owner, and a call names the hoisted function.
        if matches!(statement, syn::Stmt::Local(local) if closure_binding(local).is_some()) {
            continue;
        }
        let is_tail = return_tail && index + 1 == statements.len();
        // One record per statement, opened before and closed after the text it
        // lowers to, so nested statements land inside their parent's range.
        let record = table.begin(output, statement.span());
        emit_statement(output, table, statement, indent, index, is_tail)?;
        table.finish_written(output, record);
    }
    Ok(())
}

fn emit_statement(
    output: &mut String,
    table: &mut LineTable,
    statement: &syn::Stmt,
    indent: usize,
    index: usize,
    is_tail: bool,
) -> syn::Result<()> {
    let padding = "    ".repeat(indent);
    match statement {
        syn::Stmt::Local(local) if tuple_pattern(&local.pat).is_some() => {
            let (tuple, annotation) = tuple_pattern(&local.pat).expect("guarded by the match arm");
            // The validator looks through parentheses around the annotation (`((u32, f32))`), so does the emission.
            let annotation = annotation.map(peel_type);
            let initial = local.init.as_ref().ok_or_else(|| {
                diagnostics::error(
                    Code::LetNeedsInitializer,
                    local,
                    "Slang let bindings require an initializer",
                )
            })?;
            if matches!(peel_expr(&initial.expr), syn::Expr::If(_)) {
                return Err(diagnostics::error(
                    Code::UnsupportedExpression,
                    &initial.expr,
                    "an `if` that gives a value is not destructured by a tuple pattern: bind it to one local (`let w: u32 = if c { 1u32 } else { 2u32 };`), then use the local",
                ));
            }
            // One temporary holds the whole value (evaluated once, in Rust's order), then one local per element.
            let temporary = format!("__gust_tuple_{index}");
            let element_types: Vec<Option<String>> = match annotation {
                Some(syn::Type::Tuple(types)) if types.elems.len() == tuple.elems.len() => types
                    .elems
                    .iter()
                    .map(|ty| emit_type(ty).map(Some))
                    .collect::<syn::Result<_>>()?,
                Some(other) => {
                    return Err(diagnostics::error(
                        Code::TupleShape,
                        other,
                        "a tuple destructuring is annotated with a tuple type of the same length",
                    ));
                }
                None => vec![None; tuple.elems.len()],
            };
            let declaration = annotation
                .map(emit_type)
                .transpose()?
                .unwrap_or_else(|| "var".to_owned());
            writeln!(
                output,
                "{padding}{declaration} {temporary} = {};",
                emit_expression(&initial.expr)?
            )
            .unwrap();
            for (position, (element, ty)) in tuple.elems.iter().zip(&element_types).enumerate() {
                let syn::Pat::Ident(name) = element else {
                    return Err(diagnostics::error(
                        Code::LetNeedsIdentifier,
                        element,
                        "a tuple destructuring binds plain identifiers",
                    ));
                };
                writeln!(
                    output,
                    "{padding}{} {} = {temporary}._{position};",
                    ty.as_deref().unwrap_or("var"),
                    emit_ident(&name.ident)
                )
                .unwrap();
            }
        }
        syn::Stmt::Local(local) => {
            let (pattern, annotation) = local_binding(&local.pat)?;
            let initial = local.init.as_ref().ok_or_else(|| {
                diagnostics::error(
                    Code::LetNeedsInitializer,
                    local,
                    "Slang let bindings require an initializer",
                )
            })?;
            // The value form (slice m9.e2): `T x;` and a `switch` that assigns it.
            let value = peel_expr(&initial.expr);
            // The value form of `if`/`else` (slice m12.ifx): `T x;` and an `if` that assigns it in each branch.
            if let syn::Expr::If(expression) = value {
                return emit_value_if(
                    output,
                    table,
                    &pattern.ident,
                    annotation,
                    expression,
                    indent,
                    index,
                );
            }
            // Arrays (slice m9.r1). Slang cannot infer the type of `{ .. }`, so the `let` is typed.
            let array_annotation = annotation
                .map(peel_type)
                .filter(|ty| matches!(ty, syn::Type::Array(_)));
            if matches!(value, syn::Expr::Array(_) | syn::Expr::Repeat(_)) {
                let Some(array_type) = array_annotation else {
                    return Err(diagnostics::error(
                        Code::UnsupportedExpression,
                        value,
                        "an array literal or repeat initializes a typed `let`, because Slang cannot infer the type of `{ .. }`: write `let a: [u32; 3] = [1, 2, 3];`, with the element type and the length in the binding (typed elements such as `1u32` do not replace it)",
                    ));
                };
                if let syn::Expr::Repeat(repeat) = value {
                    return emit_repeat(
                        output,
                        indent,
                        index,
                        repeat,
                        &RepeatTarget::Let {
                            array_type,
                            name: &pattern.ident,
                        },
                    );
                }
            }
            if let (Some(array_type), false) = (
                array_annotation,
                matches!(value, syn::Expr::Match(_) | syn::Expr::Struct(_)),
            ) {
                // An initializer that names the binding reads the previous one (`let a: [u32; 2] = [a[1], a[0]];`),
                // which Slang would resolve to the new local: build the value in a temporary first.
                let declaration = emit_value_type(array_type)?;
                let name = emit_ident(&pattern.ident);
                let text = emit_value_expression(&initial.expr)?;
                if mentions_ident(
                    initial.expr.to_token_stream(),
                    &pattern.ident.unraw().to_string(),
                ) {
                    let temporary = format!("__gust_arr_{index}");
                    writeln!(output, "{padding}{declaration} {temporary} = {text};").unwrap();
                    writeln!(output, "{padding}{declaration} {name} = {temporary};").unwrap();
                } else {
                    writeln!(output, "{padding}{declaration} {name} = {text};").unwrap();
                }
                return Ok(());
            }
            if let syn::Expr::Match(expression) = value
                && crate::validate::is_switch_match(expression)
            {
                return emit_value_match(
                    output,
                    table,
                    &pattern.ident,
                    annotation,
                    expression,
                    indent,
                    index,
                );
            }
            // Finish the initializer before introducing its binding: an inner
            // `let p = Pair { first: p.second, ... }` reads the outer `p`.
            if let syn::Expr::Struct(value) = initial.expr.as_ref() {
                let temporary = emit_struct_value(output, value, indent, index)?;
                let ty = match annotation {
                    Some(ty) => emit_type(ty)?,
                    None => struct_literal_type(value)?,
                };
                writeln!(
                    output,
                    "{padding}{ty} {} = {temporary};",
                    emit_ident(&pattern.ident)
                )
                .unwrap();
                return Ok(());
            }
            let declaration = annotation
                .map(emit_type)
                .transpose()?
                .unwrap_or_else(|| "var".to_owned());
            writeln!(
                output,
                "{padding}{declaration} {} = {};",
                emit_ident(&pattern.ident),
                emit_expression(&initial.expr)?
            )
            .unwrap();
        }
        syn::Stmt::Expr(expression, semicolon) => match expression {
            syn::Expr::If(expression) => emit_if(
                output,
                table,
                expression,
                indent,
                is_tail && semicolon.is_none(),
                index,
            )?,
            syn::Expr::Match(expression) if crate::validate::is_switch_match(expression) => {
                emit_switch(output, table, expression, indent, index, None)?
            }
            syn::Expr::Match(expression) => {
                emit_result_match(output, table, expression, indent, index)?
            }
            syn::Expr::ForLoop(expression) => emit_for(output, table, expression, indent, index)?,
            syn::Expr::Break(jump) if jump.label.is_none() && jump.expr.is_none() => {
                writeln!(output, "{padding}break;").unwrap();
            }
            syn::Expr::Continue(jump) if jump.label.is_none() => {
                writeln!(output, "{padding}continue;").unwrap();
            }
            syn::Expr::Block(expression) => {
                writeln!(output, "{padding}{{").unwrap();
                emit_statements(
                    output,
                    table,
                    &expression.block.stmts,
                    indent + 1,
                    is_tail && semicolon.is_none(),
                )?;
                writeln!(output, "{padding}}}").unwrap();
            }
            // `a = [e; N];` (slice m9.r1): the operand is evaluated once, before the place is written.
            syn::Expr::Assign(assign)
                if matches!(peel_expr(&assign.right), syn::Expr::Repeat(_)) =>
            {
                let syn::Expr::Repeat(repeat) = peel_expr(&assign.right) else {
                    unreachable!("guarded by the match arm");
                };
                emit_repeat(
                    output,
                    indent,
                    index,
                    repeat,
                    &RepeatTarget::Assign {
                        place: emit_expression(&assign.left)?,
                    },
                )?;
            }
            // `return [e; N];` and a repeat as the tail value of a function.
            syn::Expr::Return(ret)
                if ret
                    .expr
                    .as_deref()
                    .is_some_and(|value| matches!(peel_expr(value), syn::Expr::Repeat(_))) =>
            {
                let Some(syn::Expr::Repeat(repeat)) = ret.expr.as_deref().map(peel_expr) else {
                    unreachable!("guarded by the match arm");
                };
                emit_repeat(output, indent, index, repeat, &RepeatTarget::Return)?;
            }
            syn::Expr::Repeat(repeat) if is_tail && semicolon.is_none() => {
                emit_repeat(output, indent, index, repeat, &RepeatTarget::Return)?;
            }
            syn::Expr::Assign(assign) if matches!(assign.right.as_ref(), syn::Expr::Struct(_)) => {
                // Rust evaluates the complete RHS before evaluating/writing the
                // destination. Direct field writes break swaps and aliasing.
                let target = emit_expression(&assign.left)?;
                let value = match assign.right.as_ref() {
                    syn::Expr::Struct(value) => value,
                    _ => unreachable!("guarded by the match arm"),
                };
                let temporary = emit_struct_value(output, value, indent, index)?;
                writeln!(output, "{padding}{target} = {temporary};").unwrap();
            }
            _ => {
                let prefix = if is_tail
                    && semicolon.is_none()
                    && !matches!(expression, syn::Expr::Return(_))
                {
                    "return "
                } else {
                    ""
                };
                // A tail value may be an array literal (the returned array); a statement may not.
                let text = if prefix.is_empty() {
                    emit_expression(expression)?
                } else {
                    emit_value_expression(expression)?
                };
                writeln!(output, "{padding}{prefix}{text};").unwrap();
            }
        },
        syn::Stmt::Item(item) => {
            return Err(diagnostics::error(
                Code::NestedItem,
                item,
                "items inside Slang functions are not supported",
            ));
        }
        syn::Stmt::Macro(item) => {
            return Err(diagnostics::error(
                Code::MacroStatement,
                item,
                "macros cannot run in Slang",
            ));
        }
    }
    Ok(())
}

/// Build a value before assigning its destination. Statement indices are unique
/// within each emitted lexical block, and the validator reserves `__gust_` names.
pub(super) fn emit_struct_value(
    output: &mut String,
    value: &syn::ExprStruct,
    indent: usize,
    index: usize,
) -> syn::Result<String> {
    let temporary = format!("__gust_struct_{index}");
    writeln!(
        output,
        "{}{} {temporary};",
        "    ".repeat(indent),
        struct_literal_type(value)?
    )
    .unwrap();
    emit_struct_assign(output, &temporary, value, indent)?;
    Ok(temporary)
}

/// Lower a struct literal to per-field assignments on a fresh temporary.
///
/// Slang has no field-name or designated initializers, so `Pair { first = 1 }`
/// is not valid; fields are initialized by name in source order. The caller must
/// not pass the final destination, which may be referenced by the initializer.
fn emit_struct_assign(
    output: &mut String,
    target: &str,
    value: &syn::ExprStruct,
    indent: usize,
) -> syn::Result<()> {
    let padding = "    ".repeat(indent);
    for field in &value.fields {
        let name = match &field.member {
            syn::Member::Named(ident) => emit_ident(ident),
            syn::Member::Unnamed(_) => {
                return Err(diagnostics::error(
                    Code::StructLiteralFieldName,
                    field,
                    "GPU struct literal fields need names",
                ));
            }
        };
        writeln!(
            output,
            "{padding}{target}.{name} = {};",
            // An array literal is a field value with a lowering (`tmp.items = { .. }`, slice m9.r2a); any other
            // expression goes through `emit_expression` as before.
            emit_value_expression(&field.expr)?
        )
        .unwrap();
    }
    Ok(())
}

/// The declared type of a struct literal, rendered as a Slang type name.
fn struct_literal_type(value: &syn::ExprStruct) -> syn::Result<String> {
    emit_type(&syn::Type::Path(syn::TypePath {
        qself: None,
        path: value.path.clone(),
    }))
}

pub fn local_binding(pattern: &syn::Pat) -> syn::Result<(&syn::PatIdent, Option<&syn::Type>)> {
    match pattern {
        syn::Pat::Ident(name) => Ok((name, None)),
        syn::Pat::Type(typed) => match typed.pat.as_ref() {
            syn::Pat::Ident(name) => Ok((name, Some(&typed.ty))),
            _ => Err(diagnostics::error(
                Code::LetBindingForm,
                pattern,
                "GPU let bindings require a single identifier, optionally with a type",
            )),
        },
        _ => Err(diagnostics::error(
            Code::LetBindingForm,
            pattern,
            "GPU let bindings require a single identifier, optionally with a type",
        )),
    }
}

/// A tuple destructuring in a `let`: `(a, b)`, with the annotation of `(a, b): (A, B)` when there is one.
pub(crate) fn tuple_pattern(pattern: &syn::Pat) -> Option<(&syn::PatTuple, Option<&syn::Type>)> {
    match pattern {
        syn::Pat::Tuple(tuple) => Some((tuple, None)),
        syn::Pat::Type(typed) => match typed.pat.as_ref() {
            syn::Pat::Tuple(tuple) => Some((tuple, Some(&typed.ty))),
            _ => None,
        },
        _ => None,
    }
}

/// What a `[e; N]` is assigned to.
enum RepeatTarget<'a> {
    /// `let name: [T; N] = [e; N];`
    Let {
        array_type: &'a syn::Type,
        name: &'a syn::Ident,
    },
    /// `place = [e; N];`
    Assign { place: String },
    /// A returned value.
    Return,
}

/// The longest repeat written as one initializer list; a longer one is filled by a loop.
const REPEAT_LIST_LIMIT: u32 = 16;

/// `[e; N]` (slice m9.r1): `e` is evaluated once, into `__gust_rep_<k>`, before the array is built, as Rust does
/// (`[s.bump(); 4]` calls `bump` once). Up to [`REPEAT_LIST_LIMIT`] elements the array is `{ t, t, .. }`, above it a
/// declaration and a `for` that fills it.
fn emit_repeat(
    output: &mut String,
    indent: usize,
    index: usize,
    repeat: &syn::ExprRepeat,
    target: &RepeatTarget<'_>,
) -> syn::Result<()> {
    let padding = "    ".repeat(indent);
    let length = array_length_literal(&repeat.len).ok_or_else(|| {
        diagnostics::error(
            Code::ArrayLength,
            &repeat.len,
            "an array length is an integer literal greater than zero",
        )
    })?;
    let temporary = format!("__gust_rep_{index}");
    let counter = format!("__gust_fill_{index}");
    // The element type is written when the `let` names it; `var` otherwise (Slang converts on the assignment).
    let element_type = match target {
        RepeatTarget::Let { array_type, .. } => match peel_type(array_type) {
            syn::Type::Array(array) => emit_type(&array.elem)?,
            _ => "var".to_owned(),
        },
        _ => "var".to_owned(),
    };
    writeln!(
        output,
        "{padding}{element_type} {temporary} = {};",
        emit_expression(&repeat.expr)?
    )
    .unwrap();
    let list = || {
        format!(
            "{{ {} }}",
            vec![temporary.as_str(); length as usize].join(", ")
        )
    };
    let fill = |output: &mut String, place: &str| {
        writeln!(
            output,
            "{padding}for (var {counter} = 0; {counter} < {length}; {counter}++)"
        )
        .unwrap();
        writeln!(output, "{padding}{{").unwrap();
        writeln!(output, "{padding}    {place}[{counter}] = {temporary};").unwrap();
        writeln!(output, "{padding}}}").unwrap();
    };
    match target {
        RepeatTarget::Let { array_type, name } => {
            let declaration = emit_value_type(array_type)?;
            let name = emit_ident(name);
            if length <= REPEAT_LIST_LIMIT {
                writeln!(output, "{padding}{declaration} {name} = {};", list()).unwrap();
            } else {
                writeln!(output, "{padding}{declaration} {name};").unwrap();
                fill(output, &name);
            }
        }
        RepeatTarget::Assign { place } => {
            if length <= REPEAT_LIST_LIMIT {
                writeln!(output, "{padding}{place} = {};", list()).unwrap();
            } else {
                fill(output, place);
            }
        }
        RepeatTarget::Return => {
            if length > REPEAT_LIST_LIMIT {
                return Err(diagnostics::error(
                    Code::UnsupportedExpression,
                    repeat,
                    format!(
                        "a returned `[e; N]` has at most {REPEAT_LIST_LIMIT} elements: build a longer one in a typed local (`let a: [T; N] = [e; N]; a`)"
                    ),
                ));
            }
            writeln!(output, "{padding}return {};", list()).unwrap();
        }
    }
    Ok(())
}
