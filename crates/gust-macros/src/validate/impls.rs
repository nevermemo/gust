//! Impl blocks: inherent impls, trait impls, trait declarations and the methods they hold.

use super::*;

/// Checks one `impl` block (slices m9.a1 and m9.a2): an inherent `impl S` (or `impl self::S`) of a struct of the module,
/// any number of blocks per struct, holding methods (`&self`, `&mut self`, `self`) and associated functions.
pub(super) fn validate_impl(
    stage: &dyn StageProfile,
    item: &syn::ItemImpl,
    context: &ImplContext<'_>,
) -> syn::Result<()> {
    if item.unsafety.is_some() || item.defaultness.is_some() {
        return Err(diagnostics::error(
            Code::UnsupportedModuleItem,
            item,
            "an `impl` of a GPU module is a plain inherent `impl S { .. }`: not `unsafe` and not `default`",
        ));
    }
    // The attributes of the block: documentation and lint levels only, with the code of an attribute on a function
    // (review m9.a1 N4, m9.a2 N3): `#[cfg(..)] impl S` would let the Slang and the CPU twin disagree about which methods exist.
    for attribute in &item.attrs {
        if let Some(message) = impl_attribute_refusal(attribute) {
            return Err(diagnostics::error(
                Code::UnsupportedAttribute,
                attribute,
                message,
            ));
        }
    }
    // `impl T for S` of a trait the module declares (slice m9.g1); any other trait impl is refused here.
    let mut implemented = None;
    if let Some((negation, path, _)) = &item.trait_ {
        let blanket = item.generics.type_params().any(|parameter| {
            matches!(item.self_ty.as_ref(), syn::Type::Path(own) if own.qself.is_none() && own.path.get_ident().is_some_and(|ident| crate::names::same_ident(ident, &parameter.ident)))
        });
        if blanket {
            return Err(diagnostics::error(
                Code::ImplForeignType,
                item,
                "a blanket `impl` is not part of the dialect: implement a trait for a struct or a unit enum of this module",
            ));
        }
        let declared =
            traits::trait_path_name(path).filter(|name| context.traits.contains_key(name));
        match declared {
            Some(_) if negation.is_some() => {
                return Err(diagnostics::error(
                    Code::UnsupportedModuleItem,
                    item,
                    "a negative impl (`impl !T for S`) is not part of the dialect",
                ));
            }
            Some(name) => implemented = Some(name),
            None => {
                return Err(diagnostics::error(
                    Code::UnsupportedModuleItem,
                    path,
                    "a trait impl is for a trait declared in this `#[gpu_module]` module (`trait T { fn m(&self); }`); other traits (`Clone`, `Default`, operators) have no GPU lowering: write the methods in an inherent `impl S { .. }`",
                ));
            }
        }
    }
    if !item.generics.params.is_empty() || item.generics.where_clause.is_some() {
        return Err(diagnostics::error(
            Code::ExternOrGenericFunction,
            &item.generics,
            "generic and lifetime parameters on an `impl` are not supported yet",
        ));
    }
    let name = impl_self_name(item).unwrap_or_default();
    let no_fields = HashSet::new();
    let fields = match context.struct_fields.get(&name) {
        Some(fields) => fields,
        // A unit enum has no fields, and a trait is implemented for it (g0 P15); an inherent `impl` of an enum is not.
        None if implemented.is_some() && context.methods.enums.contains_key(&name) => &no_fields,
        None => {
            let message = if implemented.is_some() {
                "a trait impl is for a struct or a unit enum defined in this `#[gpu_module]` module; a type from elsewhere has no Slang definition to hold the methods"
            } else if context.methods.enums.contains_key(&name) {
                "methods on an enum are not supported: an inherent `impl` is for a struct defined in this `#[gpu_module]` module; write a free function that takes the enum, or implement a trait of the module for it"
            } else {
                "an `impl` is for a struct defined in this `#[gpu_module]` module; a type from elsewhere has no Slang definition to hold the methods"
            };
            return Err(diagnostics::error(
                Code::ImplForeignType,
                &item.self_ty,
                message,
            ));
        }
    };
    let owner = Owner {
        name: &name,
        fields,
    };
    if let Some(trait_name) = &implemented {
        check_trait_impl(item, trait_name, &name, context)?;
    }
    for inner in &item.items {
        let syn::ImplItem::Fn(method) = inner else {
            return Err(diagnostics::error(
                if implemented.is_some() {
                    Code::TraitFeature
                } else {
                    Code::UnsupportedModuleItem
                },
                inner,
                "an `impl` of a GPU module holds methods only: associated constants, types and macros are not accepted (a constant is a `const` item of the module, outside the `impl`)",
            ));
        };
        validate_method(stage, method, &owner, context, implemented.is_some())?;
    }
    Ok(())
}

/// The checks of an `impl Trait for S` that look at the whole module (slice m9.g1): the shape of its functions, the
/// P18 rule and the shared names. `owner_type` is the struct or enum `S`.
pub(super) fn check_trait_impl(
    item: &syn::ItemImpl,
    trait_name: &str,
    owner_type: &str,
    context: &ImplContext<'_>,
) -> syn::Result<()> {
    let is_enum = context.methods.enums.contains_key(owner_type);
    // Every member of an inherent `impl` of the type, method or associated function: Slang has one namespace for them
    // (g0 P18: the same name in the struct and in an `extension` is accepted silently, and `s.m()` then reaches the
    // inherent one where Rust's `Trait::m(&s)` names the trait's; the same signature inside the struct is slangc E30201).
    let inherent: HashSet<String> = context
        .items
        .iter()
        .filter_map(|other| match other {
            syn::Item::Impl(block)
                if block.trait_.is_none()
                    && impl_self_name(block).is_some_and(|name| name == owner_type) =>
            {
                Some(block)
            }
            _ => None,
        })
        .flat_map(|block| &block.items)
        .filter_map(|inner| match inner {
            syn::ImplItem::Fn(method) => Some(method.sig.ident.unraw().to_string()),
            _ => None,
        })
        .collect();
    for inner in &item.items {
        let syn::ImplItem::Fn(method) = inner else {
            continue;
        };
        let name = method.sig.ident.unraw().to_string();
        let Some(receiver) = receiver_of(&method.sig) else {
            return Err(diagnostics::error(
                Code::TraitFeature,
                &method.sig,
                format!(
                    "`{name}` has no receiver: the functions of a trait impl are methods (`&self`, `&mut self` or `self`); an associated function belongs in an inherent `impl`"
                ),
            ));
        };
        if is_enum && receiver_is_ref_mut(receiver) {
            return Err(diagnostics::error(
                Code::TraitFeature,
                receiver,
                format!(
                    "`{name}` takes `&mut self` and `{owner_type}` is an enum: a method cannot assign its receiver (`*self = ..` has no lowering), so a `&mut self` method is for a struct; take `self` and return the new value"
                ),
            ));
        }
        if inherent.contains(&name) {
            return Err(diagnostics::error(
                Code::MethodNameClash,
                &method.sig.ident,
                format!(
                    "`{name}` is a method of the trait `{trait_name}` and also a member of an inherent `impl {owner_type}`: Rust's `{trait_name}::{name}(&value)` names the trait's method, but the lowering to `value.{name}()` would reach the inherent one in Slang; rename one of them"
                ),
            ));
        }
    }
    // Two traits that declare one method name, implemented by one type: the same reason.
    for other in context.items {
        let syn::Item::Impl(block) = other else {
            continue;
        };
        let Some((_, path, _)) = &block.trait_ else {
            continue;
        };
        if std::ptr::eq(block, item) || impl_self_name(block).is_none_or(|name| name != owner_type)
        {
            continue;
        }
        let Some(other_trait) = traits::trait_path_name(path) else {
            continue;
        };
        if other_trait == trait_name {
            continue;
        }
        if let Some((method, other)) =
            traits::shared_name_problem(context.traits, trait_name, &other_trait)
        {
            return Err(diagnostics::error(
                Code::MethodNameClash,
                path,
                format!(
                    "`{owner_type}` implements the traits `{trait_name}` and `{other}`, which both declare a method `{method}`: a call `{trait_name}::{method}(&value)` names one of them in Rust, and `value.{method}()` in Slang could not; rename the method in one trait"
                ),
            ));
        }
    }
    Ok(())
}

/// Checks one `trait` item (slice m9.g1): methods only, without a body, no supertraits, no generics, no associated
/// items; every method has a receiver, a name the dialect does not use, and a signature judged as a helper's.
pub(super) fn validate_trait(
    stage: &dyn StageProfile,
    item: &syn::ItemTrait,
    context: &ImplContext<'_>,
) -> syn::Result<()> {
    let name = item.ident.unraw().to_string();
    if item.unsafety.is_some() || item.auto_token.is_some() || item.restriction.is_some() {
        return Err(diagnostics::error(
            Code::TraitFeature,
            item,
            "a trait of a GPU module is a plain `trait T { .. }`: not `unsafe`, not `auto`",
        ));
    }
    for attribute in &item.attrs {
        if let Some(message) = impl_attribute_refusal(attribute) {
            return Err(diagnostics::error(
                Code::UnsupportedAttribute,
                attribute,
                message,
            ));
        }
    }
    if math_types::is_reserved_type_name(&name) {
        return Err(diagnostics::error(
            Code::ReservedTypeName,
            &item.ident,
            format!(
                "`{name}` is a type the dialect maps to a Slang type itself, so a trait cannot be named that; choose another name"
            ),
        ));
    }
    if name == "Copy" {
        return Err(diagnostics::error(
            Code::TraitFeature,
            &item.ident,
            "a trait of a GPU module cannot be named `Copy`: in a generic bound `T: Copy` is the marker, which says nothing on the GPU and is dropped",
        ));
    }
    if !item.generics.params.is_empty() || item.generics.where_clause.is_some() {
        return Err(diagnostics::error(
            Code::TraitFeature,
            &item.generics,
            "a generic trait (`trait T<X>`, a lifetime, a `where` clause) is not part of the dialect: a trait is a plain list of methods",
        ));
    }
    if !item.supertraits.is_empty() {
        return Err(diagnostics::error(
            Code::TraitFeature,
            &item.supertraits,
            "a supertrait (`trait T: U`) is not part of the dialect: declare the methods of both in one trait, or keep two traits",
        ));
    }
    for inner in &item.items {
        let syn::TraitItem::Fn(method) = inner else {
            return Err(diagnostics::error(
                Code::TraitFeature,
                inner,
                "a trait of a GPU module declares methods only: an associated type or constant is not part of the dialect",
            ));
        };
        if let Some(default) = &method.default {
            return Err(diagnostics::error(
                Code::TraitFeature,
                default,
                "a default method body is not part of the dialect: write the body in each `impl Trait for S`",
            ));
        }
        if let Some(attribute) = method.attrs.iter().find(|attribute| {
            attribute.path().is_ident("kernel")
                || attribute.path().is_ident("variant")
                || is_gpu_attribute(attribute)
        }) {
            return Err(diagnostics::error(
                Code::UnsupportedAttribute,
                attribute,
                "a method cannot be a kernel or a variant: kernels are free functions of the module",
            ));
        }
        let signature = &method.sig;
        let method_name = signature.ident.unraw().to_string();
        if !signature.generics.params.is_empty() || signature.generics.where_clause.is_some() {
            return Err(diagnostics::error(
                Code::TraitFeature,
                &signature.generics,
                "a generic trait method is not part of the dialect: a trait method is not generic (a generic helper function with the trait as its bound is)",
            ));
        }
        if signature.asyncness.is_some()
            || signature.constness.is_some()
            || signature.unsafety.is_some()
            || signature.abi.is_some()
            || signature.variadic.is_some()
        {
            return Err(diagnostics::error(
                Code::TraitFeature,
                signature,
                "a trait method is a plain `fn m(&self, ..)`: not `async`, `const`, `unsafe` or `extern`",
            ));
        }
        let Some(receiver) = receiver_of(signature) else {
            return Err(diagnostics::error(
                Code::TraitFeature,
                signature,
                format!(
                    "`{method_name}` has no receiver: a trait of a GPU module declares methods (`&self`, `&mut self` or `self`); an associated function belongs in an inherent `impl`"
                ),
            ));
        };
        if receiver.colon_token.is_some() {
            return Err(diagnostics::error(
                Code::SelfParameter,
                receiver,
                "write the receiver as `self`, `&self` or `&mut self`: a typed `self: Type` receiver is not accepted",
            ));
        }
        if receiver.reference.is_none() && receiver.mutability.is_some() {
            return Err(diagnostics::error(
                Code::SelfParameter,
                receiver,
                "`mut self` is not accepted: a method that changes its receiver takes `&mut self`, and a method that takes `self` by value reads its copy only",
            ));
        }
        if let Some(message) = method_name_clash(&method_name, &HashSet::new()) {
            return Err(diagnostics::error(
                Code::MethodNameClash,
                &signature.ident,
                message,
            ));
        }
        // The signature as a helper's, without a body: the parameter and result types of the dialect.
        let stub = syn::ItemFn {
            attrs: method.attrs.clone(),
            vis: syn::Visibility::Inherited,
            sig: method_as_function_signature(signature),
            block: Box::new(syn::Block {
                brace_token: syn::token::Brace::default(),
                stmts: Vec::new(),
            }),
        };
        validate_function(
            stage,
            &stub,
            context.function_names,
            context.result_helpers,
            context.methods,
            Some(&Owner {
                name: "Self",
                fields: &HashSet::new(),
            }),
            false,
        )?;
    }
    Ok(())
}

/// The signature of a method without its receiver.
pub(super) fn method_as_function_signature(signature: &syn::Signature) -> syn::Signature {
    let mut signature = signature.clone();
    signature.inputs = signature
        .inputs
        .into_iter()
        .filter(|argument| matches!(argument, syn::FnArg::Typed(_)))
        .collect();
    signature
}

/// Checks one method or associated function: the name (E0328), the receiver, the attributes, then the body and
/// signature as a helper's, with the receiver taken off (it is `this`).
pub(super) fn validate_method(
    stage: &dyn StageProfile,
    method: &syn::ImplItemFn,
    owner: &Owner<'_>,
    context: &ImplContext<'_>,
    trait_impl: bool,
) -> syn::Result<()> {
    if let Some(attribute) = method.attrs.iter().find(|attribute| {
        attribute.path().is_ident("kernel")
            || attribute.path().is_ident("variant")
            || is_gpu_attribute(attribute)
    }) {
        return Err(diagnostics::error(
            Code::UnsupportedAttribute,
            attribute,
            "a method cannot be a kernel or a variant: kernels are free functions of the module",
        ));
    }
    let name = method.sig.ident.unraw().to_string();
    if let Some(message) = method_name_clash(&name, owner.fields) {
        return Err(diagnostics::error(
            Code::MethodNameClash,
            &method.sig.ident,
            message,
        ));
    }
    // Inside a Slang struct an unqualified call finds the member before the free function of the same name (measured
    // on slangc 2026.21: the member captured the call), and a method may call a helper since slice m9.a2.
    if context
        .function_names
        .iter()
        .any(|helper| helper.trim_start_matches("r#") == name)
    {
        return Err(diagnostics::error(
            Code::MethodNameClash,
            &method.sig.ident,
            format!(
                "`{name}` is also the name of a helper function of this module, and inside the Slang struct a call of the name would reach the member instead of the helper: rename the method or the helper"
            ),
        ));
    }
    if let Some(receiver) = receiver_of(&method.sig) {
        if receiver.colon_token.is_some() {
            return Err(diagnostics::error(
                Code::SelfParameter,
                receiver,
                "write the receiver as `self`, `&self` or `&mut self`: a typed `self: Type` receiver is not accepted",
            ));
        }
        if receiver.reference.is_none() && receiver.mutability.is_some() {
            return Err(diagnostics::error(
                Code::SelfParameter,
                receiver,
                "`mut self` is not accepted: a method that changes its receiver takes `&mut self`, and a method that takes `self` by value reads its copy only",
            ));
        }
    }
    // Type parameters on a method (slice m9.g2): an inherent method with a receiver may have them; the method of a trait
    // impl may not (a trait method is not generic, GUST-E0124), and nor may an associated function.
    if generics::is_generic(&method.sig) {
        if trait_impl {
            return Err(diagnostics::error(
                Code::TraitFeature,
                &method.sig.generics,
                "a generic method in a trait impl is not part of the dialect: a trait method is not generic (write a generic helper function that takes the bound)",
            ));
        }
        if receiver_of(&method.sig).is_none() {
            return Err(diagnostics::error(
                Code::ExternOrGenericFunction,
                &method.sig.generics,
                "a generic associated function (no receiver) is not part of the dialect: a generic helper function or a generic method with a receiver is",
            ));
        }
        generics::check_signature(&method.sig, &method.block, context.traits, context.items)?;
    }
    let function = method_as_function(method);
    validate_function(
        stage,
        &function,
        context.function_names,
        context.result_helpers,
        context.methods,
        Some(owner),
        false,
    )?;
    // In Rust `let c = self;` in a `&mut self` method binds the reference, and a write through `c` reaches the receiver; in Slang
    // it copies `this`, and the write is lost (review m9.a2 M2). A `self` or `&self` method copies in both languages.
    if let Some(receiver) = receiver_of(&method.sig)
        && receiver_is_ref_mut(receiver)
        && let Some(path) = bare_self_use(&method.block)
    {
        return Err(diagnostics::error(
            Code::SelfParameter,
            path,
            "`self` is used as a value in a `&mut self` method (`let c = self;`, `f(self)`, `return self`): Rust binds the reference there, and Slang copies the struct, so a write through the copy would be lost; write through `self.field` directly, or call a method on `self`",
        ));
    }
    Ok(())
}

/// The method as a free function without its receiver, which the validation of a helper then judges: value
/// parameters only, no `Workgroup`, no barrier or subgroup operation. Spans are the method's own.
pub(super) fn method_as_function(method: &syn::ImplItemFn) -> syn::ItemFn {
    let mut signature = method.sig.clone();
    signature.inputs = signature
        .inputs
        .into_iter()
        .filter(|argument| matches!(argument, syn::FnArg::Typed(_)))
        .collect();
    syn::ItemFn {
        attrs: method.attrs.clone(),
        vis: method.vis.clone(),
        sig: signature,
        block: Box::new(method.block.clone()),
    }
}
