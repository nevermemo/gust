//! Struct and enum declarations.

use super::*;

pub(super) fn validate_struct(item: &syn::ItemStruct) -> syn::Result<()> {
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
            path.segments.last().is_some_and(|segment| {
                crate::names::ident_is(&segment.ident, "Clone")
                    || crate::names::ident_is(&segment.ident, "Copy")
            })
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
    visitor.errors.finish()
}

/// The integer literal of an enum discriminant (`3`, `3u32`, `-2` with a signed repr), and whether it is negated.
pub(super) fn enum_discriminant(expression: &syn::Expr, signed: bool) -> Option<()> {
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
pub(super) fn validate_enum(item: &syn::ItemEnum) -> syn::Result<()> {
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
                    .map(|segment| segment.ident.unraw().to_string());
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
    visitor.errors.finish()
}
