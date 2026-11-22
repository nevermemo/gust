//! `#[pad]` (slice m14.pad, D233 and D239): the padding of a uniform struct, inserted by `#[gpu_module]` itself.
//!
//! A struct of a `#[gpu_module]` module that a uniform reaches may carry the inert attribute `#[pad]`. `#[gpu_module]` reads it in
//! one pass, after [`super::variant::lower`] and before validation, and rewrites the struct: every gap and the tail
//! of its constant-buffer layout ([`crate::uniform::std140_struct`]: a nested struct starts at a multiple of 16, and
//! the struct is rounded up to 16) becomes private `f32` fields `_pad0`, `_pad1`, ... in field order (the first
//! numbers the struct does not use already). The validator, the layout assertions, the `GpuPod` impl and the Slang
//! emitter then read the padded struct, so all of them agree by construction; an unmarked struct is untouched and
//! keeps GUST-E0209.
//!
//! A marked struct also gets the Rust-only constructor `pub const fn new(<its fields in declaration order>) -> Self`
//! that sets the padding to `0.0`. It is returned as tokens for the `extra` items of the expansion and never lowered
//! to Slang. The macro sees only its own module, so an `impl` with a `new` elsewhere is invisible to it: rustc then
//! reports E0592 (documented under GUST-E0219).

use std::collections::{HashMap, HashSet};

use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{ext::IdentExt as _, spanned::Spanned as _};

use crate::{
    diagnostics::{self, Code},
    profile::StageProfile,
    uniform,
};

/// The name of the attribute.
const ATTRIBUTE: &str = "pad";

/// The name of the generated constructor.
const CONSTRUCTOR: &str = "new";

/// Whether `attribute` is `#[pad]` (or `#[pad(..)]`, which is an error where it is read).
fn is_pad(attribute: &syn::Attribute) -> bool {
    attribute.path().is_ident(ATTRIBUTE)
}

/// The attribute lists of a top-level item, for the check that `#[pad]` stands on structs only. An item with no
/// attribute list the macro can read (a verbatim item) has none.
fn attributes_of(item: &syn::Item) -> &[syn::Attribute] {
    match item {
        syn::Item::Const(item) => &item.attrs,
        syn::Item::Enum(item) => &item.attrs,
        syn::Item::ExternCrate(item) => &item.attrs,
        syn::Item::Fn(item) => &item.attrs,
        syn::Item::ForeignMod(item) => &item.attrs,
        syn::Item::Impl(item) => &item.attrs,
        syn::Item::Macro(item) => &item.attrs,
        syn::Item::Mod(item) => &item.attrs,
        syn::Item::Static(item) => &item.attrs,
        syn::Item::Struct(item) => &item.attrs,
        syn::Item::Trait(item) => &item.attrs,
        syn::Item::TraitAlias(item) => &item.attrs,
        syn::Item::Type(item) => &item.attrs,
        syn::Item::Union(item) => &item.attrs,
        syn::Item::Use(item) => &item.attrs,
        _ => &[],
    }
}

/// Rewrites the structs of `module` marked `#[pad]` and returns the module with the attribute gone, and the
/// constructors to emit beside it (`impl` blocks, empty when nothing is marked). A module with no `#[pad]` comes back
/// untouched, so no expansion without the attribute changes. Every misuse is GUST-E0219.
pub(crate) fn apply(
    mut module: syn::ItemMod,
    stage: &dyn StageProfile,
) -> syn::Result<(syn::ItemMod, TokenStream)> {
    let Some((_, items)) = module.content.as_mut() else {
        return Ok((module, TokenStream::new()));
    };
    let marked = marked_structs(items)?;
    if marked.is_empty() {
        return Ok((module, TokenStream::new()));
    }
    check_reach(stage, items, &marked)?;
    check_no_user_constructor(items, &marked)?;
    let mut constructors = TokenStream::new();
    for name in dependency_order(items, &marked) {
        let (index, span) = marked[&name];
        let structs = uniform::module_structs(items);
        let padded = padded_fields(structs[&name], &structs, span);
        let syn::Item::Struct(item) = &mut items[index] else {
            unreachable!("a marked item is a struct");
        };
        item.attrs.retain(|attribute| !is_pad(attribute));
        let pads = padded.pads;
        if let syn::Fields::Named(fields) = &mut item.fields {
            fields.named = padded.fields;
        }
        constructors.extend(constructor(item, &pads, span));
    }
    Ok((module, constructors))
}

/// The marked structs by name: the index of the struct among the items and the span of its first `#[pad]`. A `#[pad]`
/// with arguments, on a struct without named fields or on an item that is no struct is GUST-E0219.
fn marked_structs(items: &[syn::Item]) -> syn::Result<HashMap<String, (usize, Span)>> {
    let mut marked = HashMap::new();
    for (index, item) in items.iter().enumerate() {
        let Some(attribute) = attributes_of(item).iter().find(|a| is_pad(a)) else {
            continue;
        };
        let syn::Item::Struct(item) = item else {
            return Err(diagnostics::error(
                Code::PadMisuse,
                attribute,
                "`#[pad]` goes on a struct of the module that a uniform reaches: it inserts the padding of a \
                 constant-buffer layout, which only a struct has",
            ));
        };
        for attribute in item.attrs.iter().filter(|a| is_pad(a)) {
            if !matches!(attribute.meta, syn::Meta::Path(_)) {
                return Err(diagnostics::error(
                    Code::PadMisuse,
                    attribute,
                    "`#[pad]` takes no arguments: write `#[pad]` alone; GUST inserts `f32` fields named `_pad0`, \
                     `_pad1`, ... for every gap and for the tail of the constant-buffer layout",
                ));
            }
        }
        if !matches!(item.fields, syn::Fields::Named(_)) {
            return Err(diagnostics::error(
                Code::PadMisuse,
                &item.ident,
                "`#[pad]` goes on a struct with named fields: the padding is inserted as named `f32` fields",
            ));
        }
        marked.insert(item.ident.unraw().to_string(), (index, attribute.span()));
    }
    Ok(marked)
}

/// A marked struct must be one a uniform reaches: a struct no uniform reaches has no constant-buffer layout to pad,
/// and a silent no-op would hide a typo. A struct only storage reaches is refused too (D239).
fn check_reach(
    stage: &dyn StageProfile,
    items: &[syn::Item],
    marked: &HashMap<String, (usize, Span)>,
) -> syn::Result<()> {
    let reach = uniform::reach(stage, items)?;
    let mut names: Vec<&String> = marked.keys().collect();
    names.sort_by_key(|name| marked[*name].0);
    for name in names {
        if reach.uniform.contains(name) {
            continue;
        }
        let syn::Item::Struct(item) = &items[marked[name].0] else {
            unreachable!("a marked item is a struct");
        };
        let why = if reach.storage.contains(name) {
            "only a storage buffer reaches it, and a storage struct has no padding insertion: order its fields \
             (GUST-E0208) or fill the gap with an explicit field"
        } else {
            "no uniform parameter of a kernel of the module reaches it, as `Uniform<Params>` or as a field of a struct \
             that is"
        };
        return Err(diagnostics::error(
            Code::PadMisuse,
            &item.ident,
            format!("`#[pad]` is on `{name}`, but {why}"),
        ));
    }
    Ok(())
}

/// The constructor is named `new`: an inherent `impl` of a marked struct in the module that has a `new` would clash
/// with it (E0592 on generated tokens), so it is refused here instead.
fn check_no_user_constructor(
    items: &[syn::Item],
    marked: &HashMap<String, (usize, Span)>,
) -> syn::Result<()> {
    for item in items {
        let syn::Item::Impl(item) = item else {
            continue;
        };
        if item.trait_.is_some() {
            continue;
        }
        let Some(name) = uniform::field_type_name(&item.self_ty) else {
            continue;
        };
        if !marked.contains_key(&name) {
            continue;
        }
        for member in &item.items {
            if let syn::ImplItem::Fn(function) = member
                && function.sig.ident == CONSTRUCTOR
            {
                return Err(diagnostics::error(
                    Code::PadMisuse,
                    &function.sig.ident,
                    format!(
                        "`#[pad]` on `{name}` generates `{name}::new` (its fields in declaration order, the padding \
                         zero), and this `new` would clash with it: rename yours, or drop `#[pad]` and write the \
                         padding fields yourself"
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// The marked structs, those another marked struct holds as a field first, so that the outer struct sees the final
/// size of the inner one. A struct is visited once; a cycle (a Rust error that comes later) ends the descent.
fn dependency_order(items: &[syn::Item], marked: &HashMap<String, (usize, Span)>) -> Vec<String> {
    fn visit(
        name: &str,
        items: &[syn::Item],
        marked: &HashMap<String, (usize, Span)>,
        seen: &mut HashSet<String>,
        order: &mut Vec<String>,
    ) {
        if !seen.insert(name.to_owned()) {
            return;
        }
        let syn::Item::Struct(item) = &items[marked[name].0] else {
            return;
        };
        if let syn::Fields::Named(fields) = &item.fields {
            for field in &fields.named {
                if let Some(inner) = uniform::field_type_name(&field.ty)
                    && marked.contains_key(&inner)
                {
                    visit(&inner, items, marked, seen, order);
                }
            }
        }
        order.push(name.to_owned());
    }
    let mut names: Vec<&String> = marked.keys().collect();
    names.sort_by_key(|name| marked[*name].0);
    let mut seen = HashSet::new();
    let mut order = Vec::new();
    for name in names {
        visit(name, items, marked, &mut seen, &mut order);
    }
    order
}

/// The fields of a struct with its padding inserted, and the padding fields' names.
struct Padded {
    fields: syn::punctuated::Punctuated<syn::Field, syn::Token![,]>,
    pads: Vec<syn::Ident>,
}

/// `item` with an `f32` field `_padN` before every field that sits beyond the end of the one before it in the
/// constant-buffer layout, and at the end for the tail. A struct whose layout the macro cannot compute (a field type it
/// cannot see into, a `Mat2`) keeps its fields: the validator or the `const` assertions of the expansion report it.
fn padded_fields(
    item: &syn::ItemStruct,
    structs: &HashMap<String, &syn::ItemStruct>,
    span: Span,
) -> Padded {
    let syn::Fields::Named(written) = &item.fields else {
        unreachable!("a marked struct has named fields");
    };
    let Some(layout) = uniform::std140_struct(item, structs, 0) else {
        return Padded {
            fields: written.named.clone(),
            pads: Vec::new(),
        };
    };
    let taken: HashSet<String> = written
        .named
        .iter()
        .filter_map(|field| field.ident.as_ref().map(|ident| ident.unraw().to_string()))
        .collect();
    let mut next = 0_usize;
    let mut pads = Vec::new();
    let mut pad =
        |words: u32, fields: &mut syn::punctuated::Punctuated<syn::Field, syn::Token![,]>| {
            for _ in 0..words {
                let name = loop {
                    let name = format!("_pad{next}");
                    next += 1;
                    if !taken.contains(&name) {
                        break name;
                    }
                };
                let ident = syn::Ident::new(&name, span);
                fields.push(syn::parse_quote_spanned!(span=> #ident: f32));
                pads.push(ident);
            }
        };
    let mut fields = syn::punctuated::Punctuated::new();
    let mut end = 0_u32;
    for ((field, offset), size) in written.named.iter().zip(&layout.offsets).zip(&layout.sizes) {
        pad((offset - end) / 4, &mut fields);
        fields.push(field.clone());
        end = offset + size;
    }
    pad((layout.size - end) / 4, &mut fields);
    Padded { fields, pads }
}

/// `impl S { pub const fn new(<fields in declaration order>) -> Self }`, the padding set to `0.0`. Spanned at the
/// attribute, so a lint on the generated function points at `#[pad]`.
fn constructor(item: &syn::ItemStruct, pads: &[syn::Ident], span: Span) -> TokenStream {
    let syn::Fields::Named(fields) = &item.fields else {
        unreachable!("a marked struct has named fields");
    };
    let ident = &item.ident;
    let (impl_generics, type_generics, where_clause) = item.generics.split_for_impl();
    let (names, types): (Vec<&syn::Ident>, Vec<&syn::Type>) = fields
        .named
        .iter()
        .filter_map(|field| Some((field.ident.as_ref()?, &field.ty)))
        .filter(|(name, _)| !pads.contains(name))
        .unzip();
    let doc = format!(
        "Builds a `{ident}` from its fields, in declaration order; the padding GUST inserted (`#[pad]`) is zero."
    );
    let body = quote_spanned! {span=>
        #[doc = #doc]
        #[allow(clippy::too_many_arguments)]
        pub const fn new(#(#names: #types),*) -> Self {
            Self { #(#names,)* #(#pads: 0.0,)* }
        }
    };
    quote! {
        impl #impl_generics #ident #type_generics #where_clause {
            #body
        }
    }
}

/// A module as the recovery hands it back beside an error: the padding applied when the module has no misuse of
/// `#[pad]` (so host code that names the padded struct and `new` still compiles, and the one real error stays the
/// only one). The constructors are added as items of the module. When `#[pad]` is misused the module comes back as
/// written, attribute included, and `recover.rs` strips it afterwards.
pub(crate) fn recover(module: syn::ItemMod, stage: &dyn StageProfile) -> syn::ItemMod {
    match apply(module.clone(), stage) {
        Ok((mut padded, constructors)) => {
            if let Some((_, items)) = padded.content.as_mut()
                && let Ok(file) = diagnostics::parse2::<syn::File>(Code::PadMisuse, constructors)
            {
                items.extend(file.items);
            }
            padded
        }
        Err(_) => module,
    }
}
