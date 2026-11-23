//! The validator: checks a `#[gpu_module]` module against the restricted dialect before anything is emitted.
//!
//! The entry points are [`validate_module_in`] (and the test-only `validate_module`); the other files hold the checks
//! by concern, and this one declares them and re-exports what the rest of the crate names.
//!
//! - `tables`: the name and spelling tables (banned names, accepted methods, scalar-math arities)
//! - `patterns`: `Option` and `Result` shapes, loop headers and `match` forms
//! - `expr_rules`: predicates over single expressions (addresses, atomics, literals, shifts, places)
//! - `layout`: struct, enum, tuple and array layout checks
//! - `methods`: the method table, enum facts, receivers and method-name clashes
//! - `impls`: impl blocks, trait impls, trait declarations and their methods
//! - `const_items`: module-level `const` items
//! - `items`: struct and enum declarations
//! - `functions`: function items and the workgroup rules
//! - `closures`: closure checks and capture scans
//! - `visitor`, `visitor_checks` and `visit_impl`: the restricted-body visitor (state and `match` checks; the other
//!   checks; the `Visit` implementation)
//! - `barrier`, `callgraph`, `classify`, `const_arrays`, `generics`, `limits`, `traits`: existing rules and tables

use std::collections::{HashMap, HashSet};

use quote::ToTokens;
use syn::ext::IdentExt as _;
use syn::visit::{self, Visit};

use crate::diagnostics::{self, Code};
use crate::math_types;
use crate::names::{peel_expr, peel_type};
use crate::profile::{BodyChecks, StageProfile};
use crate::slang::texture;

mod barrier;
mod callgraph;
pub(crate) mod classify;
mod closures;
pub(crate) mod const_arrays;
mod const_items;
mod expr_rules;
mod functions;
pub(crate) mod generics;
mod impls;
mod items;
mod layout;
pub(crate) mod limits;
mod methods;
mod patterns;
mod tables;
#[cfg(test)]
mod tests;
pub(crate) mod traits;
mod visit_impl;
mod visitor;
mod visitor_checks;

pub(crate) use barrier::{is_barrier_call, subgroup_operation, workgroup_array};

use closures::*;
pub(crate) use const_items::*;
pub(crate) use expr_rules::*;
use functions::*;
use impls::*;
use items::*;
use layout::*;
pub(crate) use methods::*;
pub use patterns::*;
pub(crate) use tables::*;
use visitor::*;

/// `gpu_module` or `gpu_fn` when `attribute` is that attribute, bare or behind a path (`#[gust::gpu_fn]`,
/// `#[::gust::gpu_module]`), else `None`.
pub(crate) fn gpu_attribute_name(attribute: &syn::Attribute) -> Option<&'static str> {
    let last = attribute.path().segments.last()?;
    ["gpu_module", "gpu_fn"]
        .into_iter()
        .find(|name| last.ident == name)
}

/// Whether `attribute` is `#[gpu_module]` or `#[gpu_fn]` (see [`gpu_attribute_name`]).
pub(crate) fn is_gpu_attribute(attribute: &syn::Attribute) -> bool {
    gpu_attribute_name(attribute).is_some()
}

/// [`validate_module_in`] for the `grid` profile, the entry point of the tests; the expander passes its profile.
#[cfg(test)]
pub fn validate_module(module: &syn::ItemMod) -> syn::Result<()> {
    validate_module_in(&crate::profile::GRID, module)
}

/// The names of the helper functions of `module`: every function that is not a `#[kernel]`. These are the free
/// functions a kernel body may call (`visit_expr_call`); the shape classifier reads the same set (D94). The set also
/// holds one marker per method of an `impl` block, `.name`, which is not an identifier and so never clashes with a
/// function: it is how the classifier knows a method call as a call of the module's own method (slice m9.a1). An
/// associated function (no receiver) has the marker `Struct::name` instead (slice m9.a2).
pub(crate) fn helper_names(module: &syn::ItemMod) -> HashSet<String> {
    let items = module.content.iter().flat_map(|(_, items)| items);
    let methods = items.clone().flat_map(|item| match item {
        syn::Item::Impl(block) => {
            let owner = impl_self_name(block).unwrap_or_default();
            block
                .items
                .iter()
                .filter_map(|inner| match inner {
                    syn::ImplItem::Fn(method) if receiver_of(&method.sig).is_some() => {
                        Some(format!(".{}", method.sig.ident.unraw()))
                    }
                    syn::ImplItem::Fn(method) => {
                        Some(format!("{owner}::{}", method.sig.ident.unraw()))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
        }
        _ => Vec::new(),
    });
    items
        .filter_map(|item| match item {
            syn::Item::Fn(function)
                if !function
                    .attrs
                    .iter()
                    .any(|attribute| attribute.path().is_ident("kernel")) =>
            {
                Some(function.sig.ident.unraw().to_string())
            }
            _ => None,
        })
        .chain(methods)
        // One marker per module `const`, `const:NAME` (not an identifier, so it never clashes): a closure reads a constant
        // without capturing it (slice m11.g3).
        .chain(
            module
                .content
                .iter()
                .flat_map(|(_, items)| items)
                .filter_map(|item| match item {
                    syn::Item::Const(item) => Some(format!("const:{}", item.ident.unraw())),
                    _ => None,
                }),
        )
        .collect()
}

/// Validates a module whose kernels have the shape of the profile `stage`: the profile decides which parameter is
/// the invocation, which options a kernel may carry and which constructs it may declare.
pub(crate) fn validate_module_in(
    stage: &dyn StageProfile,
    module: &syn::ItemMod,
) -> syn::Result<()> {
    if module.content.is_none() {
        return Err(diagnostics::error(
            Code::InlineModuleRequired,
            module,
            "#[gpu_module] requires an inline module; use `#[gpu_module] mod name { ... }`",
        ));
    }
    // A trait path call (`T::m(&s, a)`, `<S as T>::m(&s, a)`) is judged as the method call it lowers to (slice m9.g1);
    // a module without a trait is judged as written.
    let desugared = traits::desugar(module)?;
    let module = desugared.as_ref().unwrap_or(module);
    let (_, items) = module.content.as_ref().expect("checked above");

    let function_names = helper_names(module);
    let trait_table = traits::trait_table(items);
    // `dyn`, `impl Trait` and a trait as a type come before every other check of a type (GUST-E0218).
    traits::check_type_uses(items, &trait_table)?;
    let methods = method_table(items);
    let struct_fields: HashMap<String, HashSet<String>> = items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Struct(item) => Some((
                item.ident.unraw().to_string(),
                item.fields
                    .iter()
                    .filter_map(|field| field.ident.as_ref().map(|name| name.unraw().to_string()))
                    .collect(),
            )),
            _ => None,
        })
        .collect();

    let result_helpers = items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Fn(function)
                if matches!(&function.sig.output, syn::ReturnType::Type(_, ty) if matches!(ty.as_ref(), syn::Type::Path(path) if result_payload(path).is_some())) =>
            {
                Some(function.sig.ident.unraw().to_string())
            }
            _ => None,
        })
        .collect();
    // Before the item loop: a `Workgroup<[Mode; 4]>` parameter would otherwise stop at the workgroup shape check (E0210).
    check_enum_layout(stage, items, &methods.enums)?;
    check_tuple_layout(stage, items)?;
    check_array_layout(stage, items)?;

    let context = ImplContext {
        function_names: &function_names,
        result_helpers: &result_helpers,
        methods: &methods,
        struct_fields: &struct_fields,
        traits: &trait_table,
        items,
    };
    // Before the items, so a `match` arm that names a constant is this code and not the `match` rules' (GUST-E0521).
    check_const_patterns(items)?;
    check_const_names(items)?;
    const_arrays::check_uses(items, &methods.const_arrays)?;
    let const_names: HashSet<String> = items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Const(item) => Some(item.ident.unraw().to_string()),
            _ => None,
        })
        .collect();
    let mut declared_consts: HashMap<String, ConstScalar> = HashMap::new();
    for item in items {
        match item {
            syn::Item::Fn(function) => {
                if let Some((attribute, name)) = function
                    .attrs
                    .iter()
                    .find_map(|attribute| Some((attribute, gpu_attribute_name(attribute)?)))
                {
                    let message = if name == "gpu_module" {
                        "`#[gpu_module]` goes on a module, not on a function: a map function takes \
                         `#[gpu_fn]` and goes outside any `#[gpu_module]` module, because a module already is GPU code \
                         and its functions are helpers or `#[kernel]`s"
                    } else {
                        "`#[gpu_fn]` on a function goes on a free function outside any `#[gpu_module]` module: \
                         a module already is GPU code, and its functions are helpers or `#[kernel]`s"
                    };
                    return Err(diagnostics::error(
                        Code::NestedGpuFunction,
                        attribute,
                        message,
                    ));
                }
                // A generic helper (slice m9.g2); a generic kernel is refused by `validate_function` (GUST-E0114).
                if generics::is_generic(&function.sig)
                    && !function
                        .attrs
                        .iter()
                        .any(|attribute| attribute.path().is_ident("kernel"))
                {
                    generics::check_signature(&function.sig, &function.block, &trait_table, items)?;
                }
                validate_function(
                    stage,
                    function,
                    &function_names,
                    &result_helpers,
                    &methods,
                    None,
                    false,
                )?
            }
            syn::Item::Struct(item) => validate_struct(item)?,
            syn::Item::Impl(item) => validate_impl(stage, item, &context)?,
            syn::Item::Trait(item) => validate_trait(stage, item, &context)?,
            syn::Item::Enum(item) => validate_enum(item)?,
            syn::Item::Const(item) => validate_const(
                item,
                &struct_fields,
                &const_names,
                &methods.const_arrays,
                &mut declared_consts,
            )?,
            // `syn` keeps a generic `const K<T>: u32 = 1;` as verbatim tokens: a `const` after the visibility.
            syn::Item::Verbatim(tokens)
                if tokens.clone().into_iter().take(4).any(
                    |tree| matches!(tree, proc_macro2::TokenTree::Ident(ident) if ident == "const"),
                ) =>
            {
                return Err(const_form(
                    item,
                    "a module `const` is not generic and has no other unusual form: write `const NAME: u32 = 4096;`",
                ));
            }
            _ => {
                return Err(diagnostics::error(
                    Code::UnsupportedModuleItem,
                    item,
                    "GPU modules may currently contain only functions, plain structs, field-less enums, traits with methods only, `impl` blocks of those structs and enums, and `const` items of one scalar literal; `static` and `use` items and type aliases are not accepted",
                ));
            }
        }
    }
    let structs: HashMap<String, &syn::ItemStruct> = items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Struct(item) => Some((item.ident.unraw().to_string(), item)),
            _ => None,
        })
        .collect();
    // The shape of the uniform parameters and of the structs they reach comes before the padding checks: a `Mat2` in
    // a uniform struct is E0205 whatever else is wrong with its layout.
    crate::uniform::check_parameters(stage, items)?;
    let reached = crate::uniform::reach(stage, items)?.uniform;
    for item in items {
        if let syn::Item::Struct(item) = item {
            check_padding(
                item,
                &structs,
                reached.contains(&item.ident.unraw().to_string()),
            )?;
        }
    }
    crate::uniform::check_padding(stage, items)?;
    // The call graph comes last, so a module that has another fault is reported for that fault alone (D162, E0120).
    callgraph::check(items)
}
