//! Array lengths, the tables that hold them while a function is emitted, and their scopes.

use syn::{ext::IdentExt as _, visit::Visit};

use super::consts;
use super::types::{ParameterFlavor, parameter_flavor, wrapper_element_type};
use crate::names::{peel_expr, peel_type};

/// The length of an array type or repeat expression: an integer literal greater than zero, unsuffixed or `usize`
/// (slice m9.r1, GUST-E0216 for anything else).
pub(crate) fn array_length_literal(len: &syn::Expr) -> Option<u32> {
    let len = peel_expr(len);
    let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Int(value),
        ..
    }) = len
    else {
        return None;
    };
    if !matches!(value.suffix(), "" | "usize") {
        return None;
    }
    value
        .base10_parse::<u32>()
        .ok()
        .filter(|length| *length > 0)
}

/// The names of a function that hold an array, with its length: `Some(n)` for a name every binding of which is
/// `[T; n]` (a parameter or a typed `let`), `None` for a name that is also bound some other way or with another length.
/// The macro sees no types, so `a.len()` is told from a storage `out.len()` by this table, and `N` is read from it
/// (slice m9.r1). The validator and the emitter both ask it, so they agree about every name.
pub(crate) fn array_lengths(
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
    block: &syn::Block,
) -> std::collections::HashMap<String, Option<u32>> {
    #[derive(Default)]
    struct Scan {
        arrays: std::collections::HashMap<String, Option<u32>>,
        others: std::collections::HashSet<String>,
    }
    impl Scan {
        fn bind(&mut self, name: &syn::Ident, length: Option<u32>) {
            let key = name.unraw().to_string();
            let Some(length) = length else {
                self.others.insert(key);
                return;
            };
            match self.arrays.get(&key).copied() {
                None => {
                    self.arrays.insert(key, Some(length));
                }
                Some(Some(known)) if known == length => {}
                Some(_) => {
                    self.arrays.insert(key, None);
                }
            }
        }
        fn array_binding(pattern: &syn::Pat) -> Option<(&syn::Ident, u32)> {
            let syn::Pat::Type(typed) = pattern else {
                return None;
            };
            let syn::Pat::Ident(name) = typed.pat.as_ref() else {
                return None;
            };
            let syn::Type::Array(array) = peel_type(&typed.ty) else {
                return None;
            };
            Some((&name.ident, array_length_literal(&array.len)?))
        }
    }
    impl<'ast> Visit<'ast> for Scan {
        fn visit_local(&mut self, local: &'ast syn::Local) {
            if let Some((name, length)) = Self::array_binding(&local.pat) {
                self.bind(name, Some(length));
                if let Some(initial) = &local.init {
                    self.visit_expr(&initial.expr);
                    if let Some((_, diverge)) = &initial.diverge {
                        self.visit_expr(diverge);
                    }
                }
                return;
            }
            syn::visit::visit_local(self, local);
        }
        fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
            self.bind(&pattern.ident, None);
        }
        // A closure body is a function of its own (slice m9.c): its names are scanned with its own parameters.
        fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {}
    }
    let mut scan = Scan::default();
    for input in inputs {
        if let syn::FnArg::Typed(argument) = input {
            let as_pattern = syn::Pat::Type(syn::PatType {
                attrs: Vec::new(),
                pat: argument.pat.clone(),
                colon_token: argument.colon_token,
                ty: argument.ty.clone(),
            });
            if let Some((name, length)) = Scan::array_binding(&as_pattern) {
                scan.bind(name, Some(length));
            } else {
                scan.visit_pat(&argument.pat);
            }
        }
    }
    scan.visit_block(block);
    let Scan { mut arrays, others } = scan;
    for name in others {
        if let Some(entry) = arrays.get_mut(&name) {
            *entry = None;
        }
    }
    // A storage parameter whose element is an array, `out: RWStructuredBuffer<[T; N]>`: `out[i].len()` is `N`. The
    // key `out[]` cannot be an identifier, so it never meets the length of the buffer itself (`out.len()`) or an array
    // local of the same name (slice m9.r2a).
    for input in inputs {
        let syn::FnArg::Typed(argument) = input else {
            continue;
        };
        let syn::Pat::Ident(name) = argument.pat.as_ref() else {
            continue;
        };
        if matches!(
            parameter_flavor(&argument.ty),
            ParameterFlavor::StorageRead | ParameterFlavor::StorageReadWrite
        ) && let Ok(element) = wrapper_element_type(&argument.ty)
            && let syn::Type::Array(array) = peel_type(element)
            && let Some(length) = array_length_literal(&array.len)
        {
            arrays.insert(format!("{}[]", name.ident.unraw()), Some(length));
        }
    }
    arrays
}

/// The array fields of the module's structs with their length, by field name: `Some(n)` when every field of the name
/// is `[T; n]` in every struct that has one, `None` when two structs give the name another length or some field of the
/// name is no array. A name no struct gives an array to is absent. The macro sees no types, so `s.items.len()` is
/// judged by the field name alone, as a method call is judged by its name (slice m9.r2a). The validator and the emitter
/// both ask it.
pub(crate) fn array_field_lengths(
    items: &[syn::Item],
) -> std::collections::HashMap<String, Option<u32>> {
    let mut lengths: std::collections::HashMap<String, Option<u32>> =
        std::collections::HashMap::new();
    let mut with_array: std::collections::HashSet<String> = std::collections::HashSet::new();
    for item in items {
        let syn::Item::Struct(item) = item else {
            continue;
        };
        let syn::Fields::Named(fields) = &item.fields else {
            continue;
        };
        for field in &fields.named {
            let Some(name) = &field.ident else {
                continue;
            };
            let length = match peel_type(&field.ty) {
                syn::Type::Array(array) => array_length_literal(&array.len),
                _ => None,
            };
            let key = name.unraw().to_string();
            if matches!(peel_type(&field.ty), syn::Type::Array(_)) {
                with_array.insert(key.clone());
            }
            match (lengths.get(&key).copied(), length) {
                (None, length) => {
                    lengths.insert(key, length);
                }
                (Some(Some(known)), Some(length)) if known == length => {}
                (Some(_), _) => {
                    lengths.insert(key, None);
                }
            }
        }
    }
    lengths.retain(|name, _| with_array.contains(name));
    lengths
}

/// What the tables know of the array a place holds, for `place.len()` (slice m9.r2a): the outer `None` says the place
/// is no array the tables can name, `Some(None)` that the name is ambiguous (two bindings or two structs disagree).
/// A name is a local or parameter array (`a`), a field is judged by its name (`s.items`, `self.items`,
/// `els[i].items`), and an element of a storage parameter whose element is an array is `out[i]` (`out[]` in `locals`).
pub(crate) fn array_place_length(
    place: &syn::Expr,
    locals: &std::collections::HashMap<String, Option<u32>>,
    fields: &std::collections::HashMap<String, Option<u32>>,
) -> Option<Option<u32>> {
    match peel_expr(place) {
        syn::Expr::Path(path) if path.qself.is_none() => {
            let name = path.path.get_ident()?.unraw().to_string();
            locals.get(&name).copied()
        }
        syn::Expr::Field(field) => match &field.member {
            syn::Member::Named(name) => fields.get(&name.unraw().to_string()).copied(),
            syn::Member::Unnamed(_) => None,
        },
        syn::Expr::Index(index) => {
            let syn::Expr::Path(path) = peel_expr(&index.expr) else {
                return None;
            };
            let name = path.path.get_ident()?.unraw().to_string();
            locals.get(&format!("{name}[]")).copied()
        }
        _ => None,
    }
}

thread_local! {
    /// The array names of the function being emitted and their lengths (see [`array_lengths`]), for `a.len()`.
    static ARRAY_LENGTHS: std::cell::RefCell<std::collections::HashMap<String, Option<u32>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    /// The array fields of the module being emitted and their lengths (see [`array_field_lengths`]), for `s.a.len()`.
    static ARRAY_FIELDS: std::cell::RefCell<std::collections::HashMap<String, Option<u32>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    /// Whether the next array literal to be emitted stands in a position that has a lowering (see
    /// [`emit_value_expression`]). The literal's own arm consumes it, so it cannot reach another expression.
    pub(super) static ARRAY_LITERAL_OK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Sets [`ARRAY_LENGTHS`] for the body of one function and restores the previous table when dropped.
pub(super) struct ArrayScope(std::collections::HashMap<String, Option<u32>>);

impl ArrayScope {
    pub(super) fn enter(next: std::collections::HashMap<String, Option<u32>>) -> Self {
        Self(ARRAY_LENGTHS.with(|cell| cell.replace(next)))
    }
}

impl Drop for ArrayScope {
    fn drop(&mut self) {
        ARRAY_LENGTHS.with(|cell| *cell.borrow_mut() = std::mem::take(&mut self.0));
    }
}

/// Sets [`ARRAY_FIELDS`] for the emission of one module and restores the previous table when dropped.
pub(super) struct FieldScope(std::collections::HashMap<String, Option<u32>>);

impl FieldScope {
    pub(super) fn enter(next: std::collections::HashMap<String, Option<u32>>) -> Self {
        Self(ARRAY_FIELDS.with(|cell| cell.replace(next)))
    }
}

impl Drop for FieldScope {
    fn drop(&mut self) {
        ARRAY_FIELDS.with(|cell| *cell.borrow_mut() = std::mem::take(&mut self.0));
    }
}

/// The length `place.len()` is, when the tables of the function and of the module name the array `place` holds.
pub(super) fn array_place_length_of(place: &syn::Expr) -> Option<u32> {
    ARRAY_LENGTHS
        .with(|locals| {
            ARRAY_FIELDS.with(|fields| {
                array_place_length(place, &locals.borrow(), &fields.borrow()).flatten()
            })
        })
        // An array constant of the module (slice m12.carr): a local cannot carry its name.
        .or_else(|| consts::array_length_of(place))
}

/// Whether a comparison operand is an array (fix round 2 of m9.r2a, R1). A field access named like a vector component or
/// swizzle (`x`, `y`, `xy`, `rgb`, ..) is never judged an array by its name: `id.x` and `v.x` are scalars or vectors even
/// when a struct of the module has an array field called `x`. Known edge: `a.x == b.x` with two array fields named like
/// components is not refused here and reaches slangc.
pub(super) fn comparison_array_operand(operand: &syn::Expr) -> bool {
    if let syn::Expr::Field(field) = operand
        && let syn::Member::Named(name) = &field.member
    {
        let name = name.unraw().to_string();
        if !name.is_empty() && name.len() <= 4 && name.chars().all(|c| "xyzwrgba".contains(c)) {
            return false;
        }
    }
    array_place_length_of(operand).is_some()
}
