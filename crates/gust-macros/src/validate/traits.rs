//! Traits as Slang interfaces (M9 plan 2.6, slice m9.g1): the pieces of the validator that are about the module's
//! `trait` items and the `impl Trait for S` blocks, apart from the checks of one impl (`validate_trait_impl` in the
//! parent module, which needs the method machinery).
//!
//! - [`trait_table`]: the traits of a module with the names of their methods.
//! - [`desugar`]: the path calls `Trait::m(&s, a)`, `Trait::m(s, a)` and `<S as Trait>::m(&s, a)` become the method call
//!   `s.m(a)` before validation, classification and emission, so every rule of a method call (E0333, E0421, the
//!   ledger, the call graph) applies to them unchanged. A call whose first argument is not a place is GUST-E0126.
//! - [`check_type_uses`]: `dyn Trait`, `impl Trait` and a trait used as a type are GUST-E0218.
//! - [`shared_name_problem`]: two traits that declare one method name and are implemented by one type (GUST-E0328).

use std::collections::HashMap;

use syn::ext::IdentExt as _;
use syn::visit::{self, Visit};
use syn::visit_mut::{self, VisitMut};

use crate::diagnostics::{self, Code};

/// The traits of a module: name to the names of the methods the trait declares.
pub(crate) type Traits = HashMap<String, Vec<String>>;

/// The traits declared by `items`, with their method names (a function with or without a body counts; a trait with a
/// default body is refused by `validate_trait`).
pub(crate) fn trait_table(items: &[syn::Item]) -> Traits {
    items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Trait(item) => Some((
                item.ident.unraw().to_string(),
                item.items
                    .iter()
                    .filter_map(|inner| match inner {
                        syn::TraitItem::Fn(method) => Some(method.sig.ident.unraw().to_string()),
                        _ => None,
                    })
                    .collect(),
            )),
            _ => None,
        })
        .collect()
}

/// The name a trait path stands for in the module: `T` or `self::T`.
pub(crate) fn trait_path_name(path: &syn::Path) -> Option<String> {
    if path.leading_colon.is_some() {
        return None;
    }
    let segments = &path.segments;
    if segments.iter().any(|segment| !segment.arguments.is_none()) {
        return None;
    }
    match segments.len() {
        1 => Some(segments[0].ident.unraw().to_string()),
        2 if segments[0].ident == "self" => Some(segments[1].ident.unraw().to_string()),
        _ => None,
    }
}

/// Whether `expression` is a place, syntactically: a name (`self` included), a field of a place, an element of a place,
/// or parentheses around one. A call, a literal, a path of two segments and a block are not.
pub(crate) fn is_place(expression: &syn::Expr) -> bool {
    match expression {
        syn::Expr::Paren(inner) => is_place(&inner.expr),
        syn::Expr::Group(inner) => is_place(&inner.expr),
        syn::Expr::Path(path) => {
            path.qself.is_none()
                && path.path.leading_colon.is_none()
                && path.path.segments.len() == 1
                && path.path.segments[0].arguments.is_none()
        }
        syn::Expr::Field(field) => is_place(&field.base),
        syn::Expr::Index(index) => is_place(&index.expr),
        _ => false,
    }
}

/// The module with every trait path call written as the method call it stands for, or `None` when the module declares
/// no trait (nothing to rewrite, nothing is cloned). The copy keeps the spans of the user's tokens.
pub(crate) fn desugar(module: &syn::ItemMod) -> syn::Result<Option<syn::ItemMod>> {
    let Some((_, items)) = &module.content else {
        return Ok(None);
    };
    let traits = trait_table(items);
    if traits.is_empty() {
        return Ok(None);
    }
    let mut copy = module.clone();
    let mut rewrite = PathCalls {
        traits: &traits,
        error: None,
    };
    rewrite.visit_item_mod_mut(&mut copy);
    match rewrite.error {
        Some(error) => Err(error),
        None => Ok(Some(copy)),
    }
}

/// Rewrites `Trait::m(place, args..)` into `place.m(args..)`.
struct PathCalls<'a> {
    traits: &'a Traits,
    error: Option<syn::Error>,
}

impl PathCalls<'_> {
    /// The method a call of the shape `Trait::m(..)` or `<S as Trait>::m(..)` names, when `Trait` is a trait of the
    /// module that declares `m`.
    fn method_of(&self, call: &syn::ExprCall) -> Option<syn::Ident> {
        let syn::Expr::Path(path) = call.func.as_ref() else {
            return None;
        };
        if path.path.leading_colon.is_some()
            || path.path.segments.iter().any(|s| !s.arguments.is_none())
        {
            return None;
        }
        let segments = &path.path.segments;
        let (trait_name, method) = match (&path.qself, segments.len()) {
            // `Trait::m`
            (None, 2) => (segments[0].ident.unraw().to_string(), &segments[1].ident),
            // `<S as Trait>::m`: the qualified self is the type, the path is `Trait::m`.
            (Some(qself), 2) if qself.position == 1 => {
                (segments[0].ident.unraw().to_string(), &segments[1].ident)
            }
            // `<S as self::Trait>::m`
            (Some(qself), 3) if qself.position == 2 && segments[0].ident == "self" => {
                (segments[1].ident.unraw().to_string(), &segments[2].ident)
            }
            _ => return None,
        };
        self.traits
            .get(&trait_name)
            .filter(|methods| methods.contains(&method.unraw().to_string()))?;
        Some(method.clone())
    }
}

impl VisitMut for PathCalls<'_> {
    fn visit_expr_mut(&mut self, expression: &mut syn::Expr) {
        // The arguments first: a trait path call can stand in the argument of another.
        visit_mut::visit_expr_mut(self, expression);
        if let Some(replacement) = self.method_call_of(expression) {
            *expression = replacement;
        }
    }
}

impl PathCalls<'_> {
    /// The method call that replaces `expression`, when it is a trait path call with a place for its receiver; an
    /// error is recorded (and `None` returned) when the receiver is not a place.
    fn method_call_of(&mut self, expression: &syn::Expr) -> Option<syn::Expr> {
        let syn::Expr::Call(call) = expression else {
            return None;
        };
        let method = self.method_of(call)?;
        let Some(first) = call.args.first() else {
            self.push(diagnostics::error(
                Code::TraitPathReceiver,
                call,
                format!(
                    "`{method}` is a method: its first argument is the receiver, a place (`T::{method}(&value, ..)`), and this call has none"
                ),
            ));
            return None;
        };
        let place = match first {
            syn::Expr::Reference(reference) => reference.expr.as_ref(),
            other => other,
        };
        if !is_place(place) {
            self.push(diagnostics::error(
                Code::TraitPathReceiver,
                first,
                format!(
                    "the receiver of a trait path call must be a place (`value`, `value.field`, `values[i]`, optionally under `&` or `&mut`), because the call lowers to `receiver.{method}(..)`; this one is a call, a literal or another expression: bind it to a local first"
                ),
            ));
            return None;
        }
        let arguments = call.args.iter().skip(1).cloned().collect();
        Some(syn::Expr::MethodCall(syn::ExprMethodCall {
            attrs: call.attrs.clone(),
            receiver: Box::new(place.clone()),
            dot_token: syn::token::Dot::default(),
            method,
            turbofish: None,
            paren_token: call.paren_token,
            args: arguments,
        }))
    }
}

impl PathCalls<'_> {
    fn push(&mut self, error: syn::Error) {
        if let Some(existing) = &mut self.error {
            existing.combine(error);
        } else {
            self.error = Some(error);
        }
    }
}

/// Refuses `dyn Trait`, `impl Trait` and a trait of the module used as a type anywhere in `items` (GUST-E0218): a
/// local, a field, a parameter, a result, a cast, a generic argument. The `impl Trait for S` header names the trait as
/// a path, not as a type, and is not a use.
pub(crate) fn check_type_uses(items: &[syn::Item], traits: &Traits) -> syn::Result<()> {
    let mut scan = TypeUses {
        traits,
        error: None,
    };
    for item in items {
        scan.visit_item(item);
    }
    scan.error.map_or(Ok(()), Err)
}

struct TypeUses<'a> {
    traits: &'a Traits,
    error: Option<syn::Error>,
}

impl TypeUses<'_> {
    fn push(&mut self, error: syn::Error) {
        if let Some(existing) = &mut self.error {
            existing.combine(error);
        } else {
            self.error = Some(error);
        }
    }
}

impl<'ast> Visit<'ast> for TypeUses<'_> {
    fn visit_type_trait_object(&mut self, ty: &'ast syn::TypeTraitObject) {
        self.push(diagnostics::error(
            Code::TraitTypeUse,
            ty,
            "`dyn Trait` has no GPU lowering: traits are static dispatch only (the receiver's concrete type picks the method); call the method on a value of a struct or enum of the module",
        ));
    }

    fn visit_type_impl_trait(&mut self, ty: &'ast syn::TypeImplTrait) {
        self.push(diagnostics::error(
            Code::TraitTypeUse,
            ty,
            "`impl Trait` as a type is not part of the dialect: name the struct or enum of the module (generic helpers with trait bounds arrive in a later slice)",
        ));
    }

    fn visit_type_path(&mut self, ty: &'ast syn::TypePath) {
        if ty.qself.is_none()
            && let Some(name) = trait_path_name(&ty.path)
            && self.traits.contains_key(&name)
        {
            self.push(diagnostics::error(
                Code::TraitTypeUse,
                ty,
                format!(
                    "`{name}` is a trait, not a type: a local, a field, a parameter or a result has the type of a struct or an enum of the module, and a trait is static dispatch only"
                ),
            ));
            return;
        }
        visit::visit_type_path(self, ty);
    }
}

/// Why a type that implements `current` and `other` cannot: they declare a method of one name (GUST-E0328). Rust names
/// the trait in `Trait::m(&s)`, and the lowering to `s.m()` must not meet two members of the name in one Slang type.
pub(crate) fn shared_name_problem(
    traits: &Traits,
    current: &str,
    other: &str,
) -> Option<(String, String)> {
    let first = traits.get(current)?;
    let second = traits.get(other)?;
    first
        .iter()
        .find(|name| second.contains(name))
        .map(|name| (name.clone(), other.to_owned()))
}
