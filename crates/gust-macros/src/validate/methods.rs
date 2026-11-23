//! The module's method and enum facts: method tables, receivers, owners and method-name clashes.

use super::*;

/// What the methods of one name have in common (slices m9.a1 and m9.a2). The macro sees no types, so a call
/// `recv.m(..)` is judged by the name alone, as the conservative call graph of D162 judges it.
#[derive(Clone, Copy, Default)]
pub(super) struct MethodFacts {
    /// Some method of the name takes `&mut self`.
    pub(super) ref_mut: bool,
    /// Some method of the name takes `self` by value.
    pub(super) by_value: bool,
}

impl MethodFacts {
    /// Whether a call of the name may write through its receiver or consume it: a storage, uniform or workgroup place
    /// must not be the receiver (GUST-E0333).
    pub(super) fn mutating(self) -> bool {
        self.ref_mut || self.by_value
    }
}

/// The members of the module's `impl` blocks: the methods by name, and every function of an impl by `Struct::name`.
#[derive(Clone, Default)]
pub(super) struct MethodTable {
    /// Methods (functions with a receiver) by name.
    pub(super) by_name: HashMap<String, MethodFacts>,
    /// Every function of an `impl` block, by `Struct::name`, with whether it has a receiver.
    pub(super) members: HashMap<String, bool>,
    /// Whether the module names `Mat2`, `Mat3` or `Mat4`: the emitter then lowers `*=` to `__gust_mul` (D79).
    pub(super) matrix: bool,
    /// The module's enums by name (slice m9.e1). They live in this table because the visitor already holds it.
    pub(super) enums: HashMap<String, EnumFacts>,
    /// The helper functions whose result type is an enum of the module, by name, with the enum's name.
    pub(super) enum_returns: HashMap<String, String>,
    /// The helpers and methods whose result type is a tuple, by name (slice m9.t). A method is known by its name alone,
    /// so a name that some impl gives a tuple result counts for every call of that name.
    pub(super) tuple_returns: HashSet<String>,
    /// The array fields of the module's structs by field name (slice m9.r2a, see `slang::array_field_lengths`): the
    /// length, or `None` when two structs disagree about the name. For `s.items.len()`.
    pub(super) array_fields: HashMap<String, Option<u32>>,
    /// The array constants of the module with their lengths (slice m12.carr): `NAME.len()` and the index guard read
    /// them as they read a local array.
    pub(super) const_arrays: HashMap<String, u32>,
}

/// What the validator knows of a field-less enum of the module (slice m9.e1, D157).
#[derive(Clone, Debug, Default)]
pub(super) struct EnumFacts {
    /// Whether the repr is `i32` (else `u32`).
    pub(super) signed: bool,
    /// The variant names, unraw.
    pub(super) variants: Vec<String>,
}

impl EnumFacts {
    /// The integer type the enum casts to, as Rust spells it.
    pub(super) fn repr_name(&self) -> &'static str {
        if self.signed { "i32" } else { "u32" }
    }
}

/// Whether the attributes of an enum carry `#[repr(i32)]` (the strict reading of `repr` is `validate_enum`'s).
pub(super) fn enum_is_signed(attributes: &[syn::Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        attribute.path().is_ident("repr")
            && matches!(&attribute.meta, syn::Meta::List(list)
                if list.tokens.clone().into_iter().any(|tree| matches!(tree,
                    proc_macro2::TokenTree::Ident(ident) if ident == "i32")))
    })
}

/// The name of the enum of `module` that `ty` is, when it is a plain name of one.
pub(super) fn enum_type_name(ty: &syn::Type, enums: &HashMap<String, EnumFacts>) -> Option<String> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    let name = path.path.get_ident()?.unraw().to_string();
    enums.contains_key(&name).then_some(name)
}

/// Whether any identifier in `stream` is one of the matrix type names (as `slang::lowering::module_lowering` counts them).
pub(super) fn names_matrix(stream: proc_macro2::TokenStream) -> bool {
    stream.into_iter().any(|tree| match tree {
        proc_macro2::TokenTree::Ident(ident) => {
            matches!(ident.unraw().to_string().as_str(), "Mat2" | "Mat3" | "Mat4")
        }
        proc_macro2::TokenTree::Group(group) => names_matrix(group.stream()),
        _ => false,
    })
}

/// The receiver of `signature` when it is the first parameter.
pub(super) fn receiver_of(signature: &syn::Signature) -> Option<&syn::Receiver> {
    match signature.inputs.first() {
        Some(syn::FnArg::Receiver(receiver)) => Some(receiver),
        _ => None,
    }
}

/// Whether a receiver is `&mut self`.
pub(super) fn receiver_is_ref_mut(receiver: &syn::Receiver) -> bool {
    receiver.reference.is_some() && receiver.mutability.is_some()
}

/// The name of the struct an inherent `impl` is for, when its type is a plain name of the module: `S` or `self::S`
/// (slice m9.a2, review m9.a1 N2). `None` for anything else (a foreign path, a generic type, a tuple).
pub(crate) fn impl_self_name(item: &syn::ItemImpl) -> Option<String> {
    let syn::Type::Path(own) = item.self_ty.as_ref() else {
        return None;
    };
    if own.qself.is_some() || own.path.leading_colon.is_some() {
        return None;
    }
    let segments = &own.path.segments;
    let last = segments.last()?;
    if !last.arguments.is_none() {
        return None;
    }
    match segments.len() {
        1 => {}
        2 if segments[0].ident == "self" && segments[0].arguments.is_none() => {}
        _ => return None,
    }
    Some(last.ident.unraw().to_string())
}

/// The members of every `impl` block of `items`.
pub(super) fn method_table(items: &[syn::Item]) -> MethodTable {
    let mut table = MethodTable {
        matrix: items
            .iter()
            .any(|item| names_matrix(item.to_token_stream())),
        array_fields: crate::slang::array_field_lengths(items),
        const_arrays: const_arrays::array_constants(items),
        ..MethodTable::default()
    };
    for item in items {
        if let syn::Item::Enum(item) = item {
            table.enums.insert(
                item.ident.unraw().to_string(),
                EnumFacts {
                    signed: enum_is_signed(&item.attrs),
                    variants: item
                        .variants
                        .iter()
                        .map(|variant| variant.ident.unraw().to_string())
                        .collect(),
                },
            );
        }
    }
    for item in items {
        if let syn::Item::Fn(function) = item
            && let syn::ReturnType::Type(_, ty) = &function.sig.output
        {
            if let Some(name) = enum_type_name(ty, &table.enums) {
                table
                    .enum_returns
                    .insert(function.sig.ident.unraw().to_string(), name);
            }
            if is_tuple_type(ty) {
                table
                    .tuple_returns
                    .insert(function.sig.ident.unraw().to_string());
            }
        }
    }
    for item in items {
        let syn::Item::Impl(block) = item else {
            continue;
        };
        let owner = impl_self_name(block).unwrap_or_default();
        for inner in &block.items {
            let syn::ImplItem::Fn(method) = inner else {
                continue;
            };
            let name = method.sig.ident.unraw().to_string();
            let receiver = receiver_of(&method.sig);
            table
                .members
                .insert(format!("{owner}::{name}"), receiver.is_some());
            if let syn::ReturnType::Type(_, ty) = &method.sig.output
                && is_tuple_type(ty)
            {
                table.tuple_returns.insert(name.clone());
            }
            if let Some(receiver) = receiver {
                let facts = table.by_name.entry(name).or_default();
                if receiver_is_ref_mut(receiver) {
                    facts.ref_mut = true;
                } else if receiver.reference.is_none() {
                    facts.by_value = true;
                }
            }
        }
    }
    table
}

/// Slang words that start a member declaration of a struct or name the enclosing type: a method of that name would be
/// a constructor, a subscript, a property or the type itself (GUST-E0328).
pub(crate) const SLANG_MEMBER_KEYWORDS: &[&str] =
    &["This", "init", "property", "subscript", "this"];

/// Why a method called `name` cannot be one (GUST-E0328), or `None`: the name is a method the dialect gives a meaning
/// to (the emitter and the validator read the name alone), a Slang member keyword, or a field of the same struct
/// (Slang has one namespace for the members of a struct).
pub(crate) fn method_name_clash(name: &str, fields: &HashSet<String>) -> Option<String> {
    let dialect = matches!(
        name,
        "global_id"
            | "len"
            | "is_some"
            | "is_none"
            | "is_ok"
            | "is_err"
            | "unwrap_or"
            | "unwrap"
            | "expect"
    ) || wrapping_arity(name).is_some()
        || scalar_math_arity(name).is_some()
        || math_types::is_swizzle(name)
        || math_types::method_arity(name).is_some()
        || texture::Method::from_name(name).is_some();
    if dialect {
        return Some(format!(
            "`{name}` is a method of the dialect, and a call of the name means that method: choose another name for this one"
        ));
    }
    if SLANG_MEMBER_KEYWORDS.contains(&name) {
        return Some(format!(
            "`{name}` is reserved by GUST as a method name (it starts a member declaration of a Slang struct, or names the struct itself): choose another name"
        ));
    }
    if fields.contains(name) {
        return Some(format!(
            "`{name}` is also a field of this struct, and a Slang struct has one namespace for the members: rename the method or the field"
        ));
    }
    None
}

/// What `validate_impl` reads of the module.
pub(super) struct ImplContext<'a> {
    pub(super) function_names: &'a HashSet<String>,
    pub(super) result_helpers: &'a HashSet<String>,
    pub(super) methods: &'a MethodTable,
    pub(super) struct_fields: &'a HashMap<String, HashSet<String>>,
    /// The traits of the module with their method names (slice m9.g1).
    pub(super) traits: &'a traits::Traits,
    /// The items of the module, for the checks of a trait impl that look at the other `impl` blocks.
    pub(super) items: &'a [syn::Item],
}

/// The struct an `impl` block is for, while its methods are validated: `Self` is its name, and its fields are the
/// names that would capture an unqualified call inside the Slang struct.
pub(super) struct Owner<'a> {
    pub(super) name: &'a str,
    pub(super) fields: &'a HashSet<String>,
}
