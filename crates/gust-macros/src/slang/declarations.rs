//! Slang declarations: enums, structs, interfaces, extensions, helper functions and functions.

use std::fmt::Write as _;

use syn::{ext::IdentExt as _, spanned::Spanned};

use super::arrays::{ArrayScope, array_length_literal, array_lengths};
use super::closures::{CLOSURE_NAMES, ClosureScope, closure_function, closure_key, closure_sites};
use super::collisions;
use super::expressions::integer_literal;
use super::lowering::SelfScope;
use super::records::LineTable;
use super::statements::emit_statements;
use super::types::{emit_ident, emit_type, emit_value_type, typed_argument};
use crate::{
    diagnostics::{self, Code},
    names::peel_type,
};

/// `enum E : uint { A, B = 3, C };` (slice m9.e1, D157): a field-less enum of the module, with the integer type of its
/// `repr` (`uint`, or `int` for `#[repr(i32)]`). The discriminants are the integer literals the validator accepted, so
/// the values count up in Slang exactly as they do in Rust.
pub(super) fn emit_enum(
    output: &mut String,
    table: &mut LineTable,
    item: &syn::ItemEnum,
) -> syn::Result<()> {
    let signed = item.attrs.iter().any(|attribute| {
        attribute.path().is_ident("repr")
            && matches!(&attribute.meta, syn::Meta::List(list)
                if list.tokens.clone().into_iter().any(|tree| matches!(tree,
                    proc_macro2::TokenTree::Ident(ident) if ident == "i32")))
    });
    let record = table.begin(output, item.ident.span());
    let mut members = Vec::new();
    for variant in &item.variants {
        let name = emit_ident(&variant.ident);
        members.push(match &variant.discriminant {
            None => name,
            Some((_, discriminant)) => format!("{name} = {}", emit_discriminant(discriminant)?),
        });
    }
    writeln!(
        output,
        "enum {} : {} {{ {} }};\n",
        emit_ident(&item.ident),
        if signed { "int" } else { "uint" },
        members.join(", ")
    )
    .unwrap();
    table.finish_written(output, record);
    Ok(())
}

/// The Slang text of an enum discriminant: an integer literal, or `-` and one (the validator took nothing else).
fn emit_discriminant(expression: &syn::Expr) -> syn::Result<String> {
    match expression {
        syn::Expr::Lit(literal) => match &literal.lit {
            syn::Lit::Int(value) => integer_literal(value, false),
            other => Err(diagnostics::error(
                Code::EnumShape,
                other,
                "a discriminant is an integer literal",
            )),
        },
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            match unary.expr.as_ref() {
                syn::Expr::Lit(literal) => match &literal.lit {
                    syn::Lit::Int(value) => Ok(format!("-{}", integer_literal(value, true)?)),
                    other => Err(diagnostics::error(
                        Code::EnumShape,
                        other,
                        "a discriminant is an integer literal",
                    )),
                },
                other => Err(diagnostics::error(
                    Code::EnumShape,
                    other,
                    "a discriminant is an integer literal",
                )),
            }
        }
        other => Err(diagnostics::error(
            Code::EnumShape,
            other,
            "a discriminant is an integer literal",
        )),
    }
}

pub(super) fn emit_struct(
    output: &mut String,
    table: &mut LineTable,
    item: &syn::ItemStruct,
    methods: &[&syn::ImplItemFn],
) -> syn::Result<()> {
    let record = table.begin(output, item.ident.span());
    writeln!(output, "struct {}", emit_ident(&item.ident)).unwrap();
    writeln!(output, "{{").unwrap();
    let syn::Fields::Named(fields) = &item.fields else {
        unreachable!("validated named fields")
    };
    for field in &fields.named {
        let name = emit_ident(field.ident.as_ref().expect("named field"));
        // A stored array is `T name[N];` in Slang (slice m9.r2a); the validator checked the length and the element.
        if let syn::Type::Array(array) = peel_type(&field.ty) {
            let length = array_length_literal(&array.len).ok_or_else(|| {
                diagnostics::error(
                    Code::ArrayLength,
                    &array.len,
                    "an array length is an integer literal greater than zero",
                )
            })?;
            writeln!(output, "    {} {name}[{length}];", emit_type(&array.elem)?).unwrap();
            continue;
        }
        writeln!(output, "    {} {name};", emit_type(&field.ty)?).unwrap();
    }
    // `Self` in the body of a method is this struct (slice m9.a2).
    let _self_scope = SelfScope::enter(&item.ident.unraw().to_string());
    for method in methods {
        writeln!(output).unwrap();
        emit_function(output, table, 1, &method.sig, &method.block)?;
    }
    table.finish_at_next(output, record);
    writeln!(output, "}};\n").unwrap();
    Ok(())
}

thread_local! {
    /// Whether the interface of a trait is being emitted: `Self` is then Slang's `This` (slice m9.g1), the type of the
    /// implementing struct that the interface cannot name.
    pub(super) static IN_INTERFACE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Sets [`IN_INTERFACE`] for a scope and restores the previous value when dropped.
struct InterfaceScope(bool);

impl InterfaceScope {
    pub(super) fn enter() -> Self {
        Self(IN_INTERFACE.with(|flag| flag.replace(true)))
    }
}

impl Drop for InterfaceScope {
    fn drop(&mut self) {
        IN_INTERFACE.with(|flag| flag.set(self.0));
    }
}

/// `trait T { fn m(&self, a: A) -> R; fn n(self) -> Self; fn k(&mut self); }` as a Slang interface (slice m9.g1):
/// `interface T { R m(A a); This n(); [mutating] void k(); }`. `self` and `&self` are the non-mutating requirement,
/// `&mut self` is `[mutating]`; `Self` is `This`. The conformance is `extension S : T` ([`emit_extension`]).
pub(super) fn emit_interface(
    output: &mut String,
    table: &mut LineTable,
    item: &syn::ItemTrait,
) -> syn::Result<()> {
    let record = table.begin(output, item.ident.span());
    writeln!(output, "interface {}", emit_ident(&item.ident)).unwrap();
    writeln!(output, "{{").unwrap();
    let _interface = InterfaceScope::enter();
    for inner in &item.items {
        let syn::TraitItem::Fn(method) = inner else {
            continue;
        };
        let signature = &method.sig;
        let qualifier = match signature.inputs.first() {
            Some(syn::FnArg::Receiver(receiver))
                if receiver.reference.is_some() && receiver.mutability.is_some() =>
            {
                "[mutating] "
            }
            _ => "",
        };
        let return_type = match &signature.output {
            syn::ReturnType::Default => "void".to_owned(),
            syn::ReturnType::Type(_, ty) => emit_value_type(ty)?,
        };
        let mut parameters = Vec::new();
        for argument in &signature.inputs {
            if matches!(argument, syn::FnArg::Receiver(_)) {
                continue;
            }
            let (name, ty) = typed_argument(argument)?;
            parameters.push(format!("{} {}", emit_value_type(ty)?, emit_ident(name)));
        }
        writeln!(
            output,
            "    {qualifier}{return_type} {}({});",
            emit_ident(&signature.ident),
            parameters.join(", ")
        )
        .unwrap();
    }
    table.finish_at_next(output, record);
    writeln!(output, "}}\n").unwrap();
    Ok(())
}

/// `impl T for S { .. }` as `extension S : T { .. }` (slice m9.g1, spike m9.g0 P5 and P15): the methods are written as the
/// members of an inherent `impl` are (`this`, `[mutating]`), the conformance is a separate declaration, and it serves a
/// unit enum as well as a struct. The closures of the methods are hoisted before it, as for a struct.
pub(super) fn emit_extension(
    output: &mut String,
    table: &mut LineTable,
    block: &syn::ItemImpl,
) -> syn::Result<()> {
    let owner = crate::validate::impl_self_name(block).expect("validated trait impl");
    let (_, path, _) = block.trait_.as_ref().expect("a trait impl");
    let trait_name = crate::validate::traits::trait_path_name(path).expect("validated trait impl");
    let _self_scope = SelfScope::enter(&owner);
    let methods: Vec<&syn::ImplItemFn> = block
        .items
        .iter()
        .filter_map(|inner| match inner {
            syn::ImplItem::Fn(method) => Some(method),
            _ => None,
        })
        .collect();
    for method in &methods {
        emit_hoisted_closures(output, table, &method.block)?;
    }
    let record = table.begin(output, block.impl_token.span);
    writeln!(
        output,
        "extension {} : {}",
        collisions::emit_name(&owner),
        collisions::emit_name(&trait_name)
    )
    .unwrap();
    writeln!(output, "{{").unwrap();
    for (position, method) in methods.iter().enumerate() {
        if position > 0 {
            writeln!(output).unwrap();
        }
        emit_function(output, table, 1, &method.sig, &method.block)?;
    }
    table.finish_at_next(output, record);
    writeln!(output, "}}\n").unwrap();
    Ok(())
}

/// The functions the closures bound in `block` hoist to (slice m9.c, D160): `__gust_cl_<owner>_<n>(float x, ..)`, one
/// per closure in source order, by-value parameters, emitted before the function that holds the closure.
pub(super) fn emit_hoisted_closures(
    output: &mut String,
    table: &mut LineTable,
    block: &syn::Block,
) -> syn::Result<()> {
    for (_, _, closure) in closure_sites(block) {
        let name = CLOSURE_NAMES
            .with(|cell| cell.borrow().get(&closure_key(closure)).cloned())
            .expect("every closure of the module is numbered");
        let ident = syn::Ident::new(&name, closure.span());
        let function = closure_function(&ident, closure).ok_or_else(|| {
            diagnostics::error(
                Code::ClosureSignature,
                closure,
                "a closure is `|x: T| -> R { .. }`: typed parameters, a result type and a block body",
            )
        })?;
        emit_function(output, table, 0, &function.sig, &function.block)?;
        writeln!(output).unwrap();
    }
    Ok(())
}

pub(super) fn emit_helper(
    output: &mut String,
    table: &mut LineTable,
    function: &syn::ItemFn,
) -> syn::Result<()> {
    emit_function(output, table, 0, &function.sig, &function.block)?;
    writeln!(output).unwrap();
    Ok(())
}

/// The Slang generic list of a generic helper or method (slice m9.g2): `<T : Shape & Named, U>`. Each type parameter
/// carries the traits it is bound to, in the order the validator's bound map gives them (inline, then `where`, `Copy`
/// dropped); a parameter with no user bound is a bare name. Names go through the collision table like every type name
/// does, and the bounds are interfaces, renamed the same way. Slang specialises the function for the argument types.
fn emit_generic_list(generics: &syn::Generics) -> String {
    let bounds = crate::validate::generics::bounds(generics);
    if bounds.is_empty() {
        return String::new();
    }
    let parameters: Vec<String> = bounds
        .iter()
        .map(|(name, traits)| {
            let name = collisions::emit_name(name);
            if traits.is_empty() {
                name.into_owned()
            } else {
                let traits: Vec<String> = traits
                    .iter()
                    .map(|bound| collisions::emit_name(bound).into_owned())
                    .collect();
                format!("{name} : {}", traits.join(" & "))
            }
        })
        .collect();
    format!("<{}>", parameters.join(", "))
}

/// A function at `depth` levels of indentation: a helper at 0, a method inside its struct at 1. A receiver is `this` in
/// Slang and is not a parameter; a `&self` or `self` method is a plain member function, which Slang does not let
/// mutate, a `&mut self` method is `[mutating]`, and a function without a receiver is `static` (slice m9.a2).
fn emit_function(
    output: &mut String,
    table: &mut LineTable,
    depth: usize,
    signature: &syn::Signature,
    block: &syn::Block,
) -> syn::Result<()> {
    let padding = "    ".repeat(depth);
    let record = table.begin(output, signature.ident.span());
    let _arrays = ArrayScope::enter(array_lengths(&signature.inputs, block));
    // The closures bound in the body are called by their binding names (slice m9.c).
    let _closures = ClosureScope::enter(block);
    let return_type = match &signature.output {
        syn::ReturnType::Default => "void".to_owned(),
        syn::ReturnType::Type(_, ty) => emit_value_type(ty)?,
    };
    let qualifier = match signature.inputs.first() {
        _ if depth == 0 => "",
        Some(syn::FnArg::Receiver(receiver))
            if receiver.reference.is_some() && receiver.mutability.is_some() =>
        {
            "[mutating] "
        }
        Some(syn::FnArg::Receiver(_)) => "",
        _ => "static ",
    };
    write!(
        output,
        "{padding}{qualifier}{return_type} {}{}(",
        emit_ident(&signature.ident),
        emit_generic_list(&signature.generics)
    )
    .unwrap();
    let mut first = true;
    for argument in &signature.inputs {
        if depth > 0 && matches!(argument, syn::FnArg::Receiver(_)) {
            continue;
        }
        if !first {
            write!(output, ", ").unwrap();
        }
        first = false;
        let (name, ty) = typed_argument(argument)?;
        write!(output, "{} {}", emit_value_type(ty)?, emit_ident(name)).unwrap();
    }
    writeln!(output, ")").unwrap();
    writeln!(output, "{padding}{{").unwrap();
    emit_statements(
        output,
        table,
        &block.stmts,
        depth + 1,
        return_type != "void",
    )?;
    table.finish_at_next(output, record);
    writeln!(output, "{padding}}}").unwrap();
    Ok(())
}
