//! Error recovery for `#[gpu]`: the original tokens beside `compile_error!`.
//!
//! When the macro rejects its input it used to return only the error. Returning the
//! user's code as well keeps the module's items present, so an editor (rust-analyzer
//! expands the macro and reads what it returns) can still show types and go-to-definition
//! in code that failed validation. The `compile_error!` is always emitted first, so the
//! build still fails and nothing is accepted silently.
//!
//! Measured with rustc 1.98.0 on `examples/vector-add` with one rejection inside the
//! `#[gpu]` module (whose kernel `app.rs` also references through `DESCRIPTOR`):
//! `cargo check` reports exactly one error both without and with the recovery of the
//! module as written, and no error about the missing items in either case. For a module
//! whose structs host code uses as `GpuPod`, the plain recovery added a second error (the
//! struct is not a GPU POD type), so the structs the expansion would have given `Clone`,
//! `Copy` and `GpuPod` get them here too (tests/ui `refused-keeps-*`).

use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident};

use crate::diagnostics::Code;

/// `error` as a `compile_error!`, followed by the user's code.
///
/// For a parsed inline module the code is the module with the attributes rustc does not
/// know (`#[kernel]`) removed, kernel functions renamed and made private as the
/// successful expansion does, the prelude import that the expansion also adds, and
/// warning noise from the unfinished module silenced. Without a parsed module (a parse
/// failure or an item that is not a module) the raw `item` tokens follow unchanged. A
/// module without inline content (`mod m;`) gets the error alone: re-emitting it would
/// make rustc look for a file. No kernel handle stubs are invented: items the expansion
/// would have generated (kernel structs, `DESCRIPTOR`) stay missing. The one exception is the `Clone`, `Copy` and
/// `GpuPod` impls of the module's structs: host code names them (an upload of the struct), and without them rustc
/// would add an error saying the struct is not a GPU POD type, which hides the one error that names the cause.
pub(crate) fn beside_error(
    error: syn::Error,
    item: &TokenStream,
    module: Option<syn::ItemMod>,
) -> TokenStream {
    let mut tokens = error.into_compile_error();
    match module {
        Some(module) => {
            if let Some(module) = recovered_module(module) {
                tokens.extend(module.into_token_stream());
            }
        }
        None => tokens.extend(item.clone()),
    }
    tokens
}

/// The facade path of a `crate = <path>` argument of a `#[gpu(..)]` whose arguments as a whole are refused: the
/// recovered code names `::gust::` as the expansion would, so it goes through the same rewrite
/// ([`crate::forward::forward_crate`]) when a well-formed `crate = <path>` is there. `None` when none is, and then the
/// code keeps `::gust::`.
pub(crate) fn facade_in_arguments(arguments: &TokenStream) -> Option<syn::Path> {
    struct Facade(syn::Path);
    impl syn::parse::Parse for Facade {
        fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
            crate::forward::parse_facade(input, Code::GpuArguments).map(Self)
        }
    }
    let mut pieces: Vec<Vec<proc_macro2::TokenTree>> = vec![Vec::new()];
    for tree in arguments.clone() {
        match &tree {
            proc_macro2::TokenTree::Punct(punct) if punct.as_char() == ',' => {
                pieces.push(Vec::new())
            }
            _ => pieces.last_mut().expect("pieces is never empty").push(tree),
        }
    }
    pieces.iter().find_map(|piece| {
        let [
            proc_macro2::TokenTree::Ident(key),
            proc_macro2::TokenTree::Punct(equals),
            rest @ ..,
        ] = piece.as_slice()
        else {
            return None;
        };
        if key != "crate" || equals.as_char() != '=' {
            return None;
        }
        crate::diagnostics::parse2::<Facade>(Code::GpuArguments, rest.iter().cloned().collect())
            .ok()
            .map(|facade| facade.0)
    })
}

/// The module as the user wrote it, made safe to hand to rustc without the macro, or
/// `None` when it has no inline content to recover.
fn recovered_module(mut module: syn::ItemMod) -> Option<syn::ItemMod> {
    let (_, items) = module.content.as_mut()?;
    for item in items.iter_mut() {
        if let syn::Item::Fn(function) = item
            && is_kernel(&function.attrs)
        {
            // The same shape the successful expansion gives a kernel (`expand/mod.rs`):
            // a private type-checking shadow, so a callable host `add` is never offered.
            function.sig.ident = format_ident!("__gpu_typecheck_{}", function.sig.ident);
            function.vis = syn::Visibility::Inherited;
        }
    }
    strip_kernel_attributes(items);
    let written = hand_written_impls(items);
    let copyable = copyable_types(items, &written);
    let provided = |name: &syn::Ident, attributes: &[syn::Attribute], trait_name: &str| {
        derives(attributes, trait_name) || written.contains(&(unraw(name), trait_name.to_owned()))
    };
    // A field-less enum is `Clone` and `Copy` as in the expansion, so a struct field of it adds no error.
    for item in items.iter_mut() {
        if let syn::Item::Enum(item) = item
            && copyable.contains(&unraw(&item.ident))
        {
            derive_missing(&mut item.attrs, |attributes, trait_name| {
                provided(&item.ident, attributes, trait_name)
            });
        }
    }
    let stand_ins: Vec<syn::Item> = items
        .iter_mut()
        .filter_map(|item| match item {
            syn::Item::Struct(item) if copyable.contains(&unraw(&item.ident)) => {
                let ident = item.ident.clone();
                derive_missing(&mut item.attrs, |attributes, trait_name| {
                    provided(&ident, attributes, trait_name)
                });
                (!written.contains(&(unraw(&ident), "GpuPod".to_owned())))
                    .then(|| pod_stand_in(&ident, &cfg_only(&item.attrs)))
            }
            _ => None,
        })
        .collect();
    items.extend(stand_ins);
    items.insert(0, syn::parse_quote! { use ::gust::prelude::*; });
    module.attrs.push(syn::parse_quote! {
        #[allow(dead_code, unused_variables, unused_mut, unused_imports)]
    });
    Some(module)
}

/// The `GpuPod` impl a struct of a refused module would have had, so host code that needs `GpuPod` of it adds no
/// error. The layout carries no fields: the build fails anyway, and naming the field types would bring an error of
/// its own for a field type that is no `GpuPod`. `cfg` is the struct's `cfg` conditions: a struct that is compiled
/// out has no impl to name it.
fn pod_stand_in(ident: &syn::Ident, cfg: &[syn::Attribute]) -> syn::Item {
    syn::parse_quote! {
        #(#cfg)*
        unsafe impl ::gust::GpuPod for #ident {
            const LAYOUT: ::gust::TypeLayout = ::gust::TypeLayout::structure(
                stringify!(#ident),
                ::core::mem::size_of::<Self>() as u32,
                ::core::mem::align_of::<Self>() as u32,
                &[],
            );
        }
    }
}

fn unraw(ident: &syn::Ident) -> String {
    use syn::ext::IdentExt as _;
    ident.unraw().to_string()
}

/// The `(type, trait)` pairs of the hand-written `impl Clone for T`, `impl Copy for T`, `unsafe impl GpuPod for T` and
/// `impl Drop for T` of the module (the validator refuses each of them, and the recovery hands them back, so adding the
/// same impl would be a second error; a type with a `Drop` impl cannot be `Copy`, E0184).
fn hand_written_impls(items: &[syn::Item]) -> HashSet<(String, String)> {
    let mut written = HashSet::new();
    for item in items {
        let syn::Item::Impl(item) = item else {
            continue;
        };
        let (Some((_, path, _)), syn::Type::Path(target)) = (&item.trait_, item.self_ty.as_ref())
        else {
            continue;
        };
        let (Some(last), Some(target)) = (path.segments.last(), target.path.segments.last()) else {
            continue;
        };
        for name in ["Clone", "Copy", "GpuPod", "Drop"] {
            if crate::names::ident_is(&last.ident, name) {
                written.insert((unraw(&target.ident), name.to_owned()));
            }
        }
    }
    written
}

/// The names of the module's types that are `Copy` by what the macro can see, so deriving `Copy` for them adds no
/// error: a field-less enum, and a struct with named fields and no generics whose fields are all scalars, vectors,
/// matrices, arrays of those, or such a type of the module. Anything else (a data enum, a tuple or generic struct,
/// `Vec`, a type from outside, a type with a hand-written `Drop` impl in `written`, and so a struct that holds one) is
/// left as the user wrote it.
fn copyable_types(items: &[syn::Item], written: &HashSet<(String, String)>) -> HashSet<String> {
    let drops = |ident: &syn::Ident| written.contains(&(unraw(ident), "Drop".to_owned()));
    let mut copyable: HashSet<String> = items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Enum(item)
                if !drops(&item.ident)
                    && item
                        .variants
                        .iter()
                        .all(|variant| variant.fields.is_empty()) =>
            {
                Some(unraw(&item.ident))
            }
            _ => None,
        })
        .collect();
    let structs: Vec<&syn::ItemStruct> = items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Struct(item)
                if !drops(&item.ident)
                    && item.generics.params.is_empty()
                    && matches!(item.fields, syn::Fields::Named(_)) =>
            {
                Some(item)
            }
            _ => None,
        })
        .collect();
    loop {
        let before = copyable.len();
        for item in &structs {
            if item
                .fields
                .iter()
                .all(|field| known_copy(&field.ty, &copyable))
            {
                copyable.insert(unraw(&item.ident));
            }
        }
        if copyable.len() == before {
            return copyable;
        }
    }
}

fn known_copy(ty: &syn::Type, copyable: &HashSet<String>) -> bool {
    match ty {
        syn::Type::Paren(inner) => known_copy(&inner.elem, copyable),
        syn::Type::Group(inner) => known_copy(&inner.elem, copyable),
        syn::Type::Array(array) => known_copy(&array.elem, copyable),
        syn::Type::Path(path) if path.qself.is_none() => {
            path.path.get_ident().is_some_and(|ident| {
                let name = unraw(ident);
                crate::math_types::known_layout(&name).is_some() || copyable.contains(&name)
            })
        }
        _ => false,
    }
}

/// `#[derive(Clone)]` and `#[derive(Copy)]` for whichever of the two `provided` does not say the type has.
fn derive_missing(
    attributes: &mut Vec<syn::Attribute>,
    provided: impl Fn(&[syn::Attribute], &str) -> bool,
) {
    for name in ["Clone", "Copy"] {
        if !provided(attributes, name) {
            let name = format_ident!("{name}");
            attributes.push(syn::parse_quote!(#[derive(#name)]));
        }
    }
}

/// Whether a `#[derive(..)]` among `attributes` names `name` (by its last path segment), one inside a
/// `#[cfg_attr(.., derive(..))]` included: adding another would be a duplicate derive.
fn derives(attributes: &[syn::Attribute], name: &str) -> bool {
    attributes.iter().any(|attribute| {
        if attribute.path().is_ident("derive") {
            crate::diagnostics::parse_args_with(
                attribute,
                Code::MalformedDerive,
                syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
            )
            .is_ok_and(|paths| {
                paths.iter().any(|path| {
                    path.segments
                        .last()
                        .is_some_and(|segment| crate::names::ident_is(&segment.ident, name))
                })
            })
        } else {
            cfg_attr_parts(attribute).is_some_and(|(_, inner)| derives(&inner, name))
        }
    })
}

/// The condition and the attributes (each as an outer attribute) of a `#[cfg_attr(condition, attributes..)]`, `None`
/// for any other attribute or one that does not parse.
fn cfg_attr_parts(attribute: &syn::Attribute) -> Option<(syn::Meta, Vec<syn::Attribute>)> {
    if !attribute.path().is_ident("cfg_attr") {
        return None;
    }
    let metas = crate::diagnostics::parse_args_with(
        attribute,
        Code::MalformedDerive,
        syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
    )
    .ok()?;
    let mut metas = metas.into_iter();
    let condition = metas.next()?;
    Some((
        condition,
        metas.map(|meta| syn::parse_quote!(#[#meta])).collect(),
    ))
}

/// What decides whether an item with `attributes` exists: its `#[cfg(..)]` attributes, and a `#[cfg_attr(condition,
/// cfg(..))]` cut down to its `cfg` parts, so an impl for the item can carry the same conditions.
fn cfg_only(attributes: &[syn::Attribute]) -> Vec<syn::Attribute> {
    attributes
        .iter()
        .filter_map(|attribute| {
            if attribute.path().is_ident("cfg") {
                return Some(attribute.clone());
            }
            let (condition, inner) = cfg_attr_parts(attribute)?;
            let kept = cfg_only(&inner);
            let metas = kept.iter().map(|attribute| &attribute.meta);
            (!kept.is_empty()).then(|| syn::parse_quote!(#[cfg_attr(#condition, #(#metas),*)]))
        })
        .collect()
}

fn is_kernel(attributes: &[syn::Attribute]) -> bool {
    attributes
        .iter()
        .any(|attribute| attribute.path().is_ident("kernel"))
}

/// Removes `#[kernel]` from the attribute lists of the items reachable from `items`
/// (the macro strips it from the functions of a successful expansion; unprocessed it is
/// an unknown attribute). A walk with `syn::visit_mut` would also reach functions nested
/// in bodies, but it needs a `syn` feature the workspace does not enable.
fn strip_kernel_attributes(items: &mut [syn::Item]) {
    for item in items {
        match item {
            syn::Item::Fn(item) => strip_from(&mut item.attrs),
            syn::Item::Struct(item) => strip_from(&mut item.attrs),
            syn::Item::Enum(item) => strip_from(&mut item.attrs),
            syn::Item::Const(item) => strip_from(&mut item.attrs),
            syn::Item::Static(item) => strip_from(&mut item.attrs),
            syn::Item::Type(item) => strip_from(&mut item.attrs),
            syn::Item::Mod(module) => {
                strip_from(&mut module.attrs);
                if let Some((_, nested)) = module.content.as_mut() {
                    strip_kernel_attributes(nested);
                }
            }
            syn::Item::Impl(item_impl) => {
                strip_from(&mut item_impl.attrs);
                for member in &mut item_impl.items {
                    match member {
                        syn::ImplItem::Fn(item) => strip_from(&mut item.attrs),
                        syn::ImplItem::Const(item) => strip_from(&mut item.attrs),
                        _ => {}
                    }
                }
            }
            syn::Item::Trait(item_trait) => {
                strip_from(&mut item_trait.attrs);
                for member in &mut item_trait.items {
                    match member {
                        syn::TraitItem::Fn(item) => strip_from(&mut item.attrs),
                        syn::TraitItem::Const(item) => strip_from(&mut item.attrs),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}

fn strip_from(attributes: &mut Vec<syn::Attribute>) {
    attributes.retain(|attribute| !attribute.path().is_ident("kernel"));
}
