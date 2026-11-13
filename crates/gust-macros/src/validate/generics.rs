//! Generic helpers and generic inherent methods with trait bounds (M9 plan 2.7, slice m9.g2): static specialisation only.
//!
//! A free helper or a method of an inherent `impl` (one with a receiver) may declare type parameters, bounded by traits
//! of the module and `Copy`, inline or in a `where` clause: `fn total<T: Shape + Named>(a: T, b: T) -> u32`. The Slang
//! text is `uint total<T : Shape & Named>(T a, T b)`, and slangc specialises it for each instantiation. This module
//! holds the checks of the generic list ([`check_signature`]) and the map of bounds the emitter reads ([`bounds`]).
//!
//! - **Bounds** are traits of the module, or `Copy`, which says nothing on the GPU and is dropped. Any other bound is
//!   GUST-E0125 (a std trait such as `Add` or `PartialEq`, `Sized`, `?Sized`, `Fn(..)`, a trait the module does not
//!   declare, a bound with arguments).
//! - **Type arguments are inferred only.** A call with a turbofish is GUST-E0123 (the call checks live in the parent
//!   module), and so is a type parameter that is not the type of some parameter: nothing in a call would fix it.
//! - **A `T` is typed where it is written** (a parameter, a result, `let x: T`) and only as the whole type: a `T` inside
//!   `[T; N]`, a tuple, `Option<T>` or a reference is refused (GUST-E0114), because slangc would have to infer the
//!   element type the way rustc does. A closure that names a generic is refused too (the hoisted function has no
//!   generic list).
//! - **Method calls on a `T`** are judged by name like every method call (D156); the bound traits' impls are methods of
//!   the module, so the macro's tables already hold them, and the call graph already draws an edge to every method of
//!   the name (D162), a superset of the implementors of the bound traits.

use std::collections::HashSet;

use syn::ext::IdentExt as _;
use syn::visit::{self, Visit};

use super::traits::{self, Traits};
use crate::diagnostics::{self, Code};

/// Whether a signature declares anything generic: type, lifetime or const parameters, or a `where` clause.
pub(crate) fn is_generic(signature: &syn::Signature) -> bool {
    !signature.generics.params.is_empty() || signature.generics.where_clause.is_some()
}

/// The names of the items of a module: structs, enums, traits and functions. A generic parameter must not take one.
pub(crate) fn item_names(items: &[syn::Item]) -> HashSet<String> {
    items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Struct(item) => Some(item.ident.unraw().to_string()),
            syn::Item::Enum(item) => Some(item.ident.unraw().to_string()),
            syn::Item::Trait(item) => Some(item.ident.unraw().to_string()),
            syn::Item::Fn(item) => Some(item.sig.ident.unraw().to_string()),
            _ => None,
        })
        .collect()
}

/// Whether `path` is the marker `Copy`: `Copy`, `core::marker::Copy` or `std::marker::Copy`.
fn is_copy_path(path: &syn::Path) -> bool {
    if path
        .segments
        .iter()
        .any(|segment| !segment.arguments.is_none())
    {
        return false;
    }
    let names: Vec<String> = path
        .segments
        .iter()
        .map(|segment| segment.ident.unraw().to_string())
        .collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    matches!(
        names.as_slice(),
        ["Copy"] | ["core" | "std", "marker", "Copy"]
    )
}

/// The trait a bound names, when the module declares it (`T` or `self::T`).
fn module_trait(path: &syn::Path, traits: &Traits) -> Option<String> {
    traits::trait_path_name(path).filter(|name| traits.contains_key(name))
}

/// The bounds of every type parameter of `generics`, in the order the parameters are declared, as the Slang generic list
/// reads them: the inline bounds first, then the `where` predicates, each trait once, `Copy` and everything the
/// validator refuses left out. The validator ran first, so this never sees a bound it refused.
pub(crate) fn bounds(generics: &syn::Generics) -> Vec<(String, Vec<String>)> {
    let mut map: Vec<(String, Vec<String>)> = generics
        .type_params()
        .map(|parameter| {
            let mut found = Vec::new();
            for bound in &parameter.bounds {
                push_bound(&mut found, bound);
            }
            (parameter.ident.unraw().to_string(), found)
        })
        .collect();
    if let Some(clause) = &generics.where_clause {
        for predicate in &clause.predicates {
            let syn::WherePredicate::Type(predicate) = predicate else {
                continue;
            };
            let syn::Type::Path(path) = &predicate.bounded_ty else {
                continue;
            };
            let Some(ident) = path.path.get_ident() else {
                continue;
            };
            let name = ident.unraw().to_string();
            if let Some((_, found)) = map.iter_mut().find(|(declared, _)| *declared == name) {
                for bound in &predicate.bounds {
                    push_bound(found, bound);
                }
            }
        }
    }
    map
}

fn push_bound(found: &mut Vec<String>, bound: &syn::TypeParamBound) {
    let syn::TypeParamBound::Trait(bound) = bound else {
        return;
    };
    let Some(name) = traits::trait_path_name(&bound.path) else {
        return;
    };
    if name != "Copy" && !found.contains(&name) {
        found.push(name);
    }
}

/// Std traits named in the message of a refused bound, so the text can say what the bound would have meant.
const STD_BOUNDS: &[&str] = &[
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Rem",
    "Neg",
    "Not",
    "BitAnd",
    "BitOr",
    "BitXor",
    "Shl",
    "Shr",
    "AddAssign",
    "SubAssign",
    "MulAssign",
    "DivAssign",
    "PartialEq",
    "Eq",
    "PartialOrd",
    "Ord",
    "Clone",
    "Default",
    "Debug",
    "Display",
    "Hash",
    "Sized",
    "Send",
    "Sync",
    "Into",
    "From",
    "Iterator",
    "Fn",
    "FnMut",
    "FnOnce",
];

/// Checks the generic list of the free helper or inherent method `signature` with the body `block`.
///
/// `traits` are the traits of the module and `items` its items (for the names a parameter must not take). The first
/// problem found is the error.
pub(crate) fn check_signature(
    signature: &syn::Signature,
    block: &syn::Block,
    traits: &Traits,
    items: &[syn::Item],
) -> syn::Result<()> {
    let generics = &signature.generics;
    let taken = item_names(items);
    let mut declared: Vec<String> = Vec::new();
    for parameter in &generics.params {
        match parameter {
            syn::GenericParam::Lifetime(lifetime) => {
                return Err(diagnostics::error(
                    Code::ExternOrGenericFunction,
                    lifetime,
                    "a lifetime parameter is not part of the dialect: a GPU function takes values, not references",
                ));
            }
            syn::GenericParam::Const(constant) => {
                return Err(diagnostics::error(
                    Code::ExternOrGenericFunction,
                    constant,
                    "a const generic parameter is not part of the dialect: a type parameter takes the traits of the module as bounds, and an array length is a literal",
                ));
            }
            syn::GenericParam::Type(parameter) => {
                if parameter.eq_token.is_some() || parameter.default.is_some() {
                    return Err(diagnostics::error(
                        Code::ExternOrGenericFunction,
                        parameter,
                        "a default type for a type parameter is not part of the dialect: type arguments are inferred from the arguments of the call",
                    ));
                }
                let name = parameter.ident.unraw().to_string();
                if let Some(problem) = name_problem(&name, &taken) {
                    return Err(diagnostics::error(
                        Code::MethodNameClash,
                        &parameter.ident,
                        problem,
                    ));
                }
                for bound in &parameter.bounds {
                    check_bound(bound, traits)?;
                }
                declared.push(name);
            }
        }
    }
    if let Some(clause) = &generics.where_clause {
        for predicate in &clause.predicates {
            let syn::WherePredicate::Type(predicate) = predicate else {
                return Err(diagnostics::error(
                    Code::ExternOrGenericFunction,
                    predicate,
                    "a lifetime predicate is not part of the dialect",
                ));
            };
            let bounded = match &predicate.bounded_ty {
                syn::Type::Path(path) if path.qself.is_none() => {
                    path.path.get_ident().map(|ident| ident.unraw().to_string())
                }
                _ => None,
            };
            if predicate.lifetimes.is_some()
                || !bounded.as_ref().is_some_and(|name| declared.contains(name))
            {
                return Err(diagnostics::error(
                    Code::ExternOrGenericFunction,
                    predicate,
                    "a `where` predicate bounds a type parameter of the function (`where T: Shape`): other predicates have no meaning in the dialect",
                ));
            }
            for bound in &predicate.bounds {
                check_bound(bound, traits)?;
            }
        }
    }
    // Every type parameter is the whole type of some parameter, so the call infers it (rustc and slangc agree on that).
    for name in &declared {
        let fixed = signature.inputs.iter().any(|argument| match argument {
            syn::FnArg::Typed(argument) => is_bare(&argument.ty, name),
            syn::FnArg::Receiver(_) => false,
        });
        if !fixed {
            return Err(diagnostics::error(
                Code::TurbofishOnGenericCall,
                generics,
                format!(
                    "the type parameter `{name}` is not the type of any parameter, so a call would have to spell it out (`f::<X>(..)`) or rely on the expected type: type arguments are inferred from the arguments only, so give a parameter the type `{name}`"
                ),
            ));
        }
    }
    let mut scan = Scan {
        declared: &declared,
        nesting: 0,
        error: None,
    };
    for argument in &signature.inputs {
        if let syn::FnArg::Typed(argument) = argument {
            scan.visit_type(&argument.ty);
        }
    }
    scan.visit_return_type(&signature.output);
    scan.visit_block(block);
    scan.error.map_or(Ok(()), Err)
}

/// Why a type parameter cannot be called `name`, or `None`.
fn name_problem(name: &str, taken: &HashSet<String>) -> Option<String> {
    let dialect = matches!(
        name,
        "Self"
            | "f32"
            | "float"
            | "i32"
            | "int"
            | "u32"
            | "uint"
            | "bool"
            | "Option"
            | "Result"
            | "Workgroup"
    ) || super::UNSUPPORTED_PRIMITIVES.contains(&name)
        || crate::math_types::math_type(name).is_some();
    if dialect {
        return Some(format!(
            "`{name}` is a type of the dialect, and a type parameter of that name would hide it in the function: choose another name (`T`, `Item`)"
        ));
    }
    if taken.contains(name) {
        return Some(format!(
            "`{name}` is also the name of an item of this module, and a type parameter of that name would hide it in the function, where the Slang text would read the item: choose another name"
        ));
    }
    None
}

/// Whether `ty` is exactly the name `name` (parentheses aside).
fn is_bare(ty: &syn::Type, name: &str) -> bool {
    match ty {
        syn::Type::Paren(inner) => is_bare(&inner.elem, name),
        syn::Type::Group(inner) => is_bare(&inner.elem, name),
        syn::Type::Path(path) => {
            path.qself.is_none()
                && path
                    .path
                    .get_ident()
                    .is_some_and(|ident| ident.unraw() == name)
        }
        _ => false,
    }
}

fn check_bound(bound: &syn::TypeParamBound, traits: &Traits) -> syn::Result<()> {
    let syn::TypeParamBound::Trait(trait_bound) = bound else {
        return Err(diagnostics::error(
            Code::ExternOrGenericFunction,
            bound,
            "a lifetime bound is not part of the dialect: a GPU function takes values, not references",
        ));
    };
    if trait_bound.lifetimes.is_some() {
        return Err(diagnostics::error(
            Code::StdTraitBound,
            bound,
            "a `for<'a>` bound is not part of the dialect",
        ));
    }
    if !matches!(trait_bound.modifier, syn::TraitBoundModifier::None) {
        return Err(diagnostics::error(
            Code::StdTraitBound,
            bound,
            "`?Sized` is a std bound: the dialect accepts the traits of the module and `Copy` only",
        ));
    }
    if module_trait(&trait_bound.path, traits).is_some() || is_copy_path(&trait_bound.path) {
        return Ok(());
    }
    let written = trait_bound
        .path
        .segments
        .last()
        .map(|segment| segment.ident.unraw().to_string())
        .unwrap_or_default();
    let message = if STD_BOUNDS.contains(&written.as_str()) {
        format!(
            "`{written}` is a std trait: a generic bound is a trait of this module or `Copy`, because the arithmetic and comparison traits have no GPU lowering yet; declare a trait with the methods you need (`trait Combine {{ fn combine(&self, other: Self) -> Self; }}`) and implement it for the types"
        )
    } else {
        format!(
            "`{written}` is not a trait declared in this module: a generic bound is a trait of this `#[gpu]` module or `Copy` (a bound with arguments, a trait from elsewhere and a std trait are refused)"
        )
    };
    Err(diagnostics::error(Code::StdTraitBound, bound, message))
}

/// The scan of a generic function's signature and body: a type parameter stands as a whole type only, and a closure does
/// not name one.
struct Scan<'a> {
    declared: &'a [String],
    /// How many compound types (arrays, tuples, references, paths with arguments) enclose the type being visited.
    nesting: usize,
    error: Option<syn::Error>,
}

impl Scan<'_> {
    fn push(&mut self, error: syn::Error) {
        if let Some(existing) = &mut self.error {
            existing.combine(error);
        } else {
            self.error = Some(error);
        }
    }

    fn names(&self, ident: &syn::Ident) -> bool {
        let name = ident.unraw().to_string();
        self.declared.contains(&name)
    }
}

impl<'ast> Visit<'ast> for Scan<'_> {
    fn visit_type(&mut self, ty: &'ast syn::Type) {
        match ty {
            syn::Type::Paren(inner) => self.visit_type(&inner.elem),
            syn::Type::Group(inner) => self.visit_type(&inner.elem),
            syn::Type::Path(path) if path.qself.is_none() && path.path.get_ident().is_some() => {
                let ident = path.path.get_ident().expect("checked");
                if self.nesting > 0 && self.names(ident) {
                    self.push(diagnostics::error(
                        Code::ExternOrGenericFunction,
                        ty,
                        format!(
                            "the type parameter `{ident}` stands as a whole type (a parameter, a result or a `let` annotation), not inside an array, a tuple, an `Option` or a reference: slangc would have to infer that type the way rustc does"
                        ),
                    ));
                }
            }
            _ => {
                self.nesting += 1;
                visit::visit_type(self, ty);
                self.nesting -= 1;
            }
        }
    }

    fn visit_type_path(&mut self, ty: &'ast syn::TypePath) {
        // `T::Item` and `<T as A>::Item`: no trait of the dialect has an associated type, so rustc refuses these too;
        // refuse them here with the reason.
        if let Some(first) = ty.path.segments.first()
            && self.names(&first.ident)
            && ty.path.segments.len() > 1
        {
            self.push(diagnostics::error(
                Code::ExternOrGenericFunction,
                ty,
                "a path through a type parameter (`T::Item`) is not part of the dialect: a trait declares methods only",
            ));
            return;
        }
        visit::visit_type_path(self, ty);
    }

    fn visit_expr_closure(&mut self, closure: &'ast syn::ExprClosure) {
        let mentions = quote::ToTokens::to_token_stream(closure)
            .into_iter()
            .any(|tree| mentions_declared(&tree, self.declared));
        if mentions {
            self.push(diagnostics::error(
                Code::ExternOrGenericFunction,
                closure,
                "a closure in a generic function cannot name the type parameter: the closure becomes a function of its own in the Slang text, with no generic list; give the closure concrete types or call a generic helper from it",
            ));
        }
    }
}

fn mentions_declared(tree: &proc_macro2::TokenTree, declared: &[String]) -> bool {
    match tree {
        proc_macro2::TokenTree::Ident(ident) => declared.contains(&ident.unraw().to_string()),
        proc_macro2::TokenTree::Group(group) => group
            .stream()
            .into_iter()
            .any(|inner| mentions_declared(&inner, declared)),
        _ => false,
    }
}
