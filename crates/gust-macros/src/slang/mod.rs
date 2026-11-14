use std::fmt::Write as _;

use quote::ToTokens;
use syn::{ext::IdentExt as _, spanned::Spanned, visit::Visit};

use crate::{
    diagnostics::{self, Code},
    expand::KernelOptions,
    math_types,
    profile::{Builtin, DerivedId, GRID, KernelFacts, StageProfile},
};

#[cfg(test)]
mod collision_tests;
pub(crate) mod collisions;
pub(crate) mod div_table;
pub(crate) mod guards;
pub(crate) mod mul_table;
pub(crate) mod neg_table;
pub(crate) mod texture;

/// The lowerings that depend on what the whole module names (the macro sees syntax and not types, so a module that
/// names a matrix or an integer vector sends `*` and `.dot` through overloaded helpers; see [`module_lowering`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Lowering {
    /// Every `*` is `__gust_mul(a, b)` and every `*=` is `(place = __gust_mul(place, b))` (slice m6.m2).
    pub(crate) mul: bool,
    /// Every unary `-` on a non-literal is `__gust_neg(a)` (slice m6.n).
    pub(crate) neg: bool,
    /// Every `.dot(v)` is `__gust_dot(a, b)` (slice m6.m2).
    pub(crate) dot: bool,
}

thread_local! {
    /// The lowering of the module being emitted. The emitter is a set of free functions, so the flags live here
    /// instead of in the signature of each; [`LoweringScope`] sets them for the span of one `emit_kernel`. It is `None`
    /// outside that span, so an emitter entry point that forgot the scope panics instead of emitting ungated text.
    static LOWERING: std::cell::Cell<Option<Lowering>> = const { std::cell::Cell::new(None) };
}

fn lowering() -> Lowering {
    LOWERING
        .with(std::cell::Cell::get)
        .expect("the emitter runs inside emit_kernel, which enters a LoweringScope")
}

/// Sets the lowering for a scope and restores the previous one when dropped.
struct LoweringScope(Option<Lowering>);

impl LoweringScope {
    fn enter(next: Lowering) -> Self {
        Self(LOWERING.with(|cell| cell.replace(Some(next))))
    }
}

impl Drop for LoweringScope {
    fn drop(&mut self) {
        LOWERING.with(|cell| cell.set(self.0));
    }
}

thread_local! {
    /// The Rust name of the struct whose `impl` block is being emitted: `Self` in a method is that struct (slice m9.a2).
    static SELF_STRUCT: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// The struct `Self` stands for, while the methods of its `impl` blocks are emitted.
fn self_struct() -> Option<String> {
    SELF_STRUCT.with(|cell| cell.borrow().clone())
}

/// Sets [`SELF_STRUCT`] for a scope and restores the previous value when dropped.
struct SelfScope(Option<String>);

impl SelfScope {
    fn enter(name: &str) -> Self {
        Self(SELF_STRUCT.with(|cell| cell.replace(Some(name.to_owned()))))
    }
}

impl Drop for SelfScope {
    fn drop(&mut self) {
        SELF_STRUCT.with(|cell| *cell.borrow_mut() = self.0.take());
    }
}

/// `let f = |..| ..;` (slice m9.c): the binding and its closure. The pattern may carry `mut` or a type, which the
/// validator refuses (GUST-E0330); a `let` with another pattern or with an `else` is no closure binding.
pub(crate) fn closure_binding(local: &syn::Local) -> Option<(&syn::PatIdent, &syn::ExprClosure)> {
    let pattern = match &local.pat {
        syn::Pat::Ident(pattern) => pattern,
        syn::Pat::Type(typed) => match typed.pat.as_ref() {
            syn::Pat::Ident(pattern) => pattern,
            _ => return None,
        },
        _ => return None,
    };
    let initial = local.init.as_ref()?;
    if initial.diverge.is_some() {
        return None;
    }
    match initial.expr.as_ref() {
        syn::Expr::Closure(closure) => Some((pattern, closure)),
        _ => None,
    }
}

/// A closure binding of a function body: the `let`, its pattern and its closure.
pub(crate) type ClosureSite<'a> = (&'a syn::Local, &'a syn::PatIdent, &'a syn::ExprClosure);

/// The closure bindings of `block` in source order, without looking inside a closure (a nested closure is
/// GUST-E0332, and a closure body is validated and emitted as a function of its own).
pub(crate) fn closure_sites(block: &syn::Block) -> Vec<ClosureSite<'_>> {
    struct Sites<'a>(Vec<ClosureSite<'a>>);
    impl<'ast> Visit<'ast> for Sites<'ast> {
        fn visit_local(&mut self, local: &'ast syn::Local) {
            if let Some((pattern, closure)) = closure_binding(local) {
                self.0.push((local, pattern, closure));
                return;
            }
            syn::visit::visit_local(self, local);
        }
        fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {}
    }
    let mut sites = Sites(Vec::new());
    sites.visit_block(block);
    sites.0
}

/// The closure as a plain function called `name`, when it has the accepted shape (every parameter `name: Type`, an
/// explicit result type, a block body): `fn name(x: f32) -> f32 { .. }` with the tokens, and so the spans, of the
/// closure. The validator judges the body as the body of a helper, and the emitter hoists it as one (slice m9.c).
pub(crate) fn closure_function(
    name: &syn::Ident,
    closure: &syn::ExprClosure,
) -> Option<syn::ItemFn> {
    let mut inputs = syn::punctuated::Punctuated::<syn::FnArg, syn::Token![,]>::new();
    for input in &closure.inputs {
        let syn::Pat::Type(typed) = input else {
            return None;
        };
        if !matches!(typed.pat.as_ref(), syn::Pat::Ident(_)) {
            return None;
        }
        inputs.push(syn::FnArg::Typed(typed.clone()));
    }
    let output = match &closure.output {
        output @ syn::ReturnType::Type(..) => output.clone(),
        syn::ReturnType::Default => return None,
    };
    let syn::Expr::Block(body) = closure.body.as_ref() else {
        return None;
    };
    let block = &body.block;
    Some(syn::parse_quote! { fn #name(#inputs) #output #block })
}

thread_local! {
    /// The Slang name of every closure of the module being emitted, by the address of its `syn::ExprClosure`:
    /// `__gust_cl_<owner>_<n>`, `n` counting the closures of the module in source order (slice m9.c). The number is
    /// unique, so a helper `S_m` and a method `S::m` cannot give two closures one name.
    static CLOSURE_NAMES: std::cell::RefCell<std::collections::HashMap<usize, String>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    /// The closures callable from the function being emitted: the name of the binding and the Slang name.
    static CLOSURE_SCOPE: std::cell::RefCell<std::collections::HashMap<String, String>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// The address that identifies a closure in [`CLOSURE_NAMES`].
fn closure_key(closure: &syn::ExprClosure) -> usize {
    std::ptr::from_ref(closure) as usize
}

/// The Slang names of the closures of `items`, numbered in source order over the functions, then the methods of each
/// `impl` block as they come (the order of the items).
fn closure_names(items: &[syn::Item]) -> std::collections::HashMap<usize, String> {
    let mut names = std::collections::HashMap::new();
    let mut number = 0_usize;
    let mut take = |owner: &str, block: &syn::Block| {
        for (_, _, closure) in closure_sites(block) {
            names.insert(closure_key(closure), format!("__gust_cl_{owner}_{number}"));
            number += 1;
        }
    };
    for item in items {
        match item {
            syn::Item::Fn(function) => {
                take(&function.sig.ident.unraw().to_string(), &function.block)
            }
            syn::Item::Impl(block) => {
                let owner = crate::validate::impl_self_name(block).unwrap_or_default();
                for inner in &block.items {
                    if let syn::ImplItem::Fn(method) = inner {
                        take(
                            &format!("{owner}_{}", method.sig.ident.unraw()),
                            &method.block,
                        );
                    }
                }
            }
            _ => {}
        }
    }
    names
}

/// Sets [`CLOSURE_NAMES`] for a scope and restores the previous map when dropped.
struct ClosureNamesScope(std::collections::HashMap<usize, String>);

impl ClosureNamesScope {
    fn enter(next: std::collections::HashMap<usize, String>) -> Self {
        Self(CLOSURE_NAMES.with(|cell| cell.replace(next)))
    }
}

impl Drop for ClosureNamesScope {
    fn drop(&mut self) {
        CLOSURE_NAMES.with(|cell| *cell.borrow_mut() = std::mem::take(&mut self.0));
    }
}

/// Makes the closures bound in `block` callable by their binding names, for a scope.
struct ClosureScope(std::collections::HashMap<String, String>);

impl ClosureScope {
    fn enter(block: &syn::Block) -> Self {
        let scope = CLOSURE_NAMES.with(|names| {
            let names = names.borrow();
            closure_sites(block)
                .into_iter()
                .filter_map(|(_, pattern, closure)| {
                    Some((
                        pattern.ident.unraw().to_string(),
                        names.get(&closure_key(closure))?.clone(),
                    ))
                })
                .collect::<std::collections::HashMap<_, _>>()
        });
        Self(CLOSURE_SCOPE.with(|cell| cell.replace(scope)))
    }
}

impl Drop for ClosureScope {
    fn drop(&mut self) {
        CLOSURE_SCOPE.with(|cell| *cell.borrow_mut() = std::mem::take(&mut self.0));
    }
}

/// The Slang name `call` calls, when it calls a closure of the function being emitted.
fn closure_call_target(call: &syn::ExprCall) -> Option<String> {
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    let name = path.path.get_ident()?.unraw().to_string();
    CLOSURE_SCOPE.with(|cell| cell.borrow().get(&name).cloned())
}

/// The length of an array type or repeat expression: an integer literal greater than zero, unsuffixed or `usize`
/// (slice m9.r1, GUST-E0216 for anything else).
pub(crate) fn array_length_literal(len: &syn::Expr) -> Option<u32> {
    let mut len = len;
    while let syn::Expr::Paren(syn::ExprParen { expr, .. })
    | syn::Expr::Group(syn::ExprGroup { expr, .. }) = len
    {
        len = expr;
    }
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

/// `ty` without the parentheses and invisible groups around it.
fn unparenthesized_type(ty: &syn::Type) -> &syn::Type {
    match ty {
        syn::Type::Paren(inner) => unparenthesized_type(&inner.elem),
        syn::Type::Group(inner) => unparenthesized_type(&inner.elem),
        other => other,
    }
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
            let syn::Type::Array(array) = unparenthesized_type(&typed.ty) else {
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
            && let syn::Type::Array(array) = unparenthesized_type(element)
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
            let length = match unparenthesized_type(&field.ty) {
                syn::Type::Array(array) => array_length_literal(&array.len),
                _ => None,
            };
            let key = name.unraw().to_string();
            if matches!(unparenthesized_type(&field.ty), syn::Type::Array(_)) {
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
    match unparenthesized_expr(place) {
        syn::Expr::Path(path) if path.qself.is_none() => {
            let name = path.path.get_ident()?.unraw().to_string();
            locals.get(&name).copied()
        }
        syn::Expr::Field(field) => match &field.member {
            syn::Member::Named(name) => fields.get(&name.unraw().to_string()).copied(),
            syn::Member::Unnamed(_) => None,
        },
        syn::Expr::Index(index) => {
            let syn::Expr::Path(path) = unparenthesized_expr(&index.expr) else {
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
    static ARRAY_LITERAL_OK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Sets [`ARRAY_LENGTHS`] for the body of one function and restores the previous table when dropped.
struct ArrayScope(std::collections::HashMap<String, Option<u32>>);

impl ArrayScope {
    fn enter(next: std::collections::HashMap<String, Option<u32>>) -> Self {
        Self(ARRAY_LENGTHS.with(|cell| cell.replace(next)))
    }
}

impl Drop for ArrayScope {
    fn drop(&mut self) {
        ARRAY_LENGTHS.with(|cell| *cell.borrow_mut() = std::mem::take(&mut self.0));
    }
}

/// Sets [`ARRAY_FIELDS`] for the emission of one module and restores the previous table when dropped.
struct FieldScope(std::collections::HashMap<String, Option<u32>>);

impl FieldScope {
    fn enter(next: std::collections::HashMap<String, Option<u32>>) -> Self {
        Self(ARRAY_FIELDS.with(|cell| cell.replace(next)))
    }
}

impl Drop for FieldScope {
    fn drop(&mut self) {
        ARRAY_FIELDS.with(|cell| *cell.borrow_mut() = std::mem::take(&mut self.0));
    }
}

/// The length `place.len()` is, when the tables of the function and of the module name the array `place` holds.
fn array_place_length_of(place: &syn::Expr) -> Option<u32> {
    ARRAY_LENGTHS.with(|locals| {
        ARRAY_FIELDS
            .with(|fields| array_place_length(place, &locals.borrow(), &fields.borrow()).flatten())
    })
}

/// Whether a comparison operand is an array (fix round 2 of m9.r2a, R1). A field access named like a vector component or
/// swizzle (`x`, `y`, `xy`, `rgb`, ..) is never judged an array by its name: `id.x` and `v.x` are scalars or vectors even
/// when a struct of the module has an array field called `x`. Known edge: `a.x == b.x` with two array fields named like
/// components is not refused here and reaches slangc.
fn comparison_array_operand(operand: &syn::Expr) -> bool {
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

/// How often each identifier occurs in the tokens of a module, after raw stripping (`r#Mat4` counts as `Mat4`).
fn count_identifiers(
    stream: proc_macro2::TokenStream,
    counts: &mut std::collections::HashMap<String, usize>,
) {
    for tree in stream {
        match tree {
            proc_macro2::TokenTree::Ident(ident) => {
                *counts.entry(ident.unraw().to_string()).or_default() += 1;
            }
            proc_macro2::TokenTree::Group(group) => count_identifiers(group.stream(), counts),
            _ => {}
        }
    }
}

/// The lowering of a module, computed once from its tokens (slice m6.m2, M6 plan 2.1).
///
/// `mul` and `neg` are set if any identifier is `Mat2`, `Mat3` or `Mat4`. The gate is exact: E0203 forbids a user
/// type of those names, E0324 a qualified path to one, and E0102 a `use` item, so the prelude is the only way the name
/// resolves and a matrix value exists in a module only if one of those tokens does. The other modules keep the verbatim
/// `*` and `-`.
///
/// `dot` is set if the identifier `dot` occurs. `.dot` is the only route to Slang's `dot` (`emit_math_method`), and the
/// macro cannot tell an integer vector from a float one (the invocation id is a `UVec3` that needs no type token, and
/// `id.xy()` is a `UVec2` with no `UVec2` token), so every module that calls `dot` takes the overloads. The integer
/// `dot` that Slang emits as `OpSDot` or `OpUDot` needs `Capability DotProduct`, which naga's SPIR-V reader refuses; a
/// float `dot` keeps Slang's own (the SPIR-V is byte-identical, measured in m6.m2; the WGSL calls a helper).
fn module_lowering(module: &syn::ItemMod) -> Lowering {
    let mut counts = std::collections::HashMap::new();
    count_identifiers(module.to_token_stream(), &mut counts);
    let count = |name: &str| counts.get(name).copied().unwrap_or_default();

    let matrix = ["Mat2", "Mat3", "Mat4"].iter().any(|name| count(name) > 0);
    Lowering {
        mul: matrix,
        neg: matrix,
        dot: count("dot") > 0,
    }
}

/// The vector types `__gust_dot` has an overload for: Slang spelling, result type, component count.
const DOT_OVERLOADS: &[(&str, &str, usize)] = &[
    ("float2", "float", 2),
    ("float3", "float", 3),
    ("float4", "float", 4),
    ("int2", "int", 2),
    ("int4", "int", 4),
    ("uint2", "uint", 2),
    ("uint3", "uint", 3),
    ("uint4", "uint", 4),
];

/// The overloads of `__gust_dot`: Slang's `dot` for the float vectors, and the sum of the component products, grouped
/// from the left like the twin's `x * x + y * y + ..`, for the integer ones. Slang's `dot` on an integer vector is
/// `OpSDot` or `OpUDot` in SPIR-V, which needs `Capability DotProduct` (naga's SPIR-V reader refuses it); the sum
/// is plain multiplications and additions, which pass naga's reader with empty capabilities (measured, slangc 2026.20).
fn dot_overload_text() -> String {
    let mut text = String::new();
    for &(vector, result, width) in DOT_OVERLOADS {
        let body = if result == "float" {
            "dot(a, b)".to_owned()
        } else {
            let mut sum = "a.x * b.x".to_owned();
            for field in &["y", "z", "w"][..width - 1] {
                sum = format!("({sum} + a.{field} * b.{field})");
            }
            sum
        };
        writeln!(
            text,
            "{result} __gust_dot({vector} a, {vector} b) {{ return {body}; }}"
        )
        .unwrap();
    }
    text.push('\n');
    text
}

/// Whether the place of a `*=` can be written twice: an identifier, a field path, or an index whose expression is
/// built from literals, names, fields, indexing, arithmetic and `.global_id()`/`.len()`. Anything else (a call, an
/// `if`, a block) may have an effect, and Rust evaluates the place of `place *= value` once while
/// `place = __gust_mul(place, value)` evaluates it twice. A `Group` is the invisible parentheses of a `macro_rules!`
/// `$p:expr`.
pub(crate) fn is_pure_place(place: &syn::Expr) -> bool {
    match place {
        syn::Expr::Path(_) | syn::Expr::Lit(_) => true,
        syn::Expr::Paren(paren) => is_pure_place(&paren.expr),
        syn::Expr::Group(group) => is_pure_place(&group.expr),
        syn::Expr::Field(field) => is_pure_place(&field.base),
        syn::Expr::Index(index) => is_pure_place(&index.expr) && is_pure_place(&index.index),
        syn::Expr::Unary(unary) => is_pure_place(&unary.expr),
        syn::Expr::Cast(cast) => is_pure_place(&cast.expr),
        syn::Expr::Binary(binary) => {
            is_pure_place(&binary.left)
                && is_pure_place(&binary.right)
                && !is_assignment_operator(&binary.op)
        }
        syn::Expr::MethodCall(call) => {
            let name = crate::names::name_of(&call.method);
            let builtin = call.args.is_empty() && matches!(name.as_str(), "global_id" | "len");
            // The wrapping methods are arithmetic on their operands, like `+` and `-` above.
            let wrapping = crate::validate::wrapping_arity(&name) == Some(call.args.len())
                && call.args.iter().all(is_pure_place);
            (builtin || wrapping) && is_pure_place(&call.receiver)
        }
        _ => false,
    }
}

fn is_assignment_operator(op: &syn::BinOp) -> bool {
    matches!(
        op,
        syn::BinOp::AddAssign(_)
            | syn::BinOp::SubAssign(_)
            | syn::BinOp::MulAssign(_)
            | syn::BinOp::DivAssign(_)
            | syn::BinOp::RemAssign(_)
            | syn::BinOp::BitXorAssign(_)
            | syn::BinOp::BitAndAssign(_)
            | syn::BinOp::BitOrAssign(_)
            | syn::BinOp::ShlAssign(_)
            | syn::BinOp::ShrAssign(_)
    )
}

/// The Slang spelling of a user identifier (M6 plan 2.5): a raw prefix is stripped, and a name in the collision table
/// becomes `__gust_r_<name>`. Every site that writes a user name into Slang, and every descriptor `name:` string, goes
/// through this one function, so a renamed name is the same text everywhere.
pub(crate) fn emit_ident(ident: &syn::Ident) -> String {
    collisions::emit_name(&ident.unraw().to_string()).into_owned()
}

/// One inclusive, 1-based range of emitted Slang lines generated from one Rust
/// line (DX19). Records are in emission order, outer construct first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LineRecord {
    pub slang_start: u32,
    pub slang_end: u32,
    pub rust_line: u32,
}

/// Emitted Slang text and the line table recorded while writing it.
pub struct EmittedKernel {
    pub source: String,
    pub lines: Vec<LineRecord>,
}

/// Records which Rust line each emitted statement, helper and struct came from.
/// It only observes `output` and never changes what is written. Spans without a
/// location (line 0) are skipped, so a table can be empty.
#[derive(Default)]
struct LineTable {
    records: Vec<LineRecord>,
    /// Bytes of `output` already counted, and the newlines among them.
    scanned: usize,
    newlines: u32,
}

impl LineTable {
    /// The line the next write to `output` lands on.
    fn next_line(&mut self, output: &str) -> u32 {
        let fresh = &output.as_bytes()[self.scanned..];
        self.newlines += fresh.iter().filter(|byte| **byte == b'\n').count() as u32;
        self.scanned = output.len();
        self.newlines + 1
    }

    /// Open a record at the next line to be written.
    fn begin(&mut self, output: &str, span: proc_macro2::Span) -> Option<usize> {
        let start = self.next_line(output);
        let rust_line = u32::try_from(span.start().line)
            .ok()
            .filter(|line| *line > 0)?;
        self.records.push(LineRecord {
            slang_start: start,
            slang_end: start,
            rust_line,
        });
        Some(self.records.len() - 1)
    }

    /// Close a record at the last line written so far.
    fn finish_written(&mut self, output: &str, record: Option<usize>) {
        let end = self.next_line(output) - 1;
        self.set_end(record, end);
    }

    /// Close a record at the line about to be written (a closing brace).
    fn finish_at_next(&mut self, output: &str, record: Option<usize>) {
        let end = self.next_line(output);
        self.set_end(record, end);
    }

    fn set_end(&mut self, record: Option<usize>, end: u32) {
        if let Some(record) = record.and_then(|index| self.records.get_mut(index)) {
            record.slang_end = end.max(record.slang_start);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterFlavor {
    Invocation,
    /// `LocalId` or `SV_GroupThreadID` (slice m7.g1): `uint3` derived from the dispatch-thread id and the workgroup
    /// size, not an entry-point parameter and not in the descriptor.
    LocalId,
    /// `GroupId` or `SV_GroupID` (slice m7.g1): derived like [`ParameterFlavor::LocalId`].
    GroupId,
    /// `Workgroup<[T; N]>` (slice m7.g1): a module-scope `groupshared` array, not a resource and not in the
    /// descriptor.
    Workgroup,
    StorageRead,
    StorageReadWrite,
    Uniform,
    /// `Texture2D<T>` (slice mx.t3): a sampled texture, read with `load` and `sample`.
    SampledTexture,
    /// `StorageTexture2D<F>` (slice mx.t3): a read-write storage texture of one of the three 32-bit single-channel
    /// formats.
    StorageTexture,
    /// `WriteTexture2D<F>` (slice mx.t3): a write-only storage texture of any of the nine formats.
    WriteTexture,
    /// `Sampler` (slice mx.t3): a sampler, passed to `sample`.
    Sampler,
    Value,
}

impl ParameterFlavor {
    /// Whether the parameter is part of the kernel's host contract (the descriptor, the export record and the typed
    /// call): the dispatch-thread id and the resources. The derived ids and the workgroup arrays are the body's own.
    pub(crate) fn in_descriptor(self) -> bool {
        !matches!(self, Self::LocalId | Self::GroupId | Self::Workgroup)
    }

    /// Whether the parameter is a texture (not a sampler).
    pub(crate) fn is_texture(self) -> bool {
        matches!(
            self,
            Self::SampledTexture | Self::StorageTexture | Self::WriteTexture
        )
    }

    /// Whether the parameter is a texture or a sampler: it has a shader binding and the dialect uses it only through
    /// texture methods (GUST-E0736).
    pub(crate) fn is_texture_or_sampler(self) -> bool {
        self.is_texture() || self == Self::Sampler
    }
}

/// [`parameter_flavor_in`] for the `grid` profile, for the callers that classify a type without a kernel (the
/// reachability of uniform structs, the nested checks of the validator): they tell resources from values, and the
/// invocation names do not matter to them.
pub fn parameter_flavor(ty: &syn::Type) -> ParameterFlavor {
    parameter_flavor_in(&GRID, ty)
}

/// What a kernel parameter of type `ty` is: the invocation built-ins are the profile's (D63), the resources are the
/// dialect's.
pub(crate) fn parameter_flavor_in(stage: &dyn StageProfile, ty: &syn::Type) -> ParameterFlavor {
    match stage.invocation(ty) {
        Some(Builtin::Global) => return ParameterFlavor::Invocation,
        Some(Builtin::Local) => return ParameterFlavor::LocalId,
        Some(Builtin::Group) => return ParameterFlavor::GroupId,
        None => {}
    }
    let Some(name) = type_name(ty) else {
        return ParameterFlavor::Value;
    };
    match name.as_str() {
        "Workgroup" => ParameterFlavor::Workgroup,
        "Storage" | "StructuredBuffer" => ParameterFlavor::StorageRead,
        "StorageMut" | "RWStructuredBuffer" => ParameterFlavor::StorageReadWrite,
        "Uniform" | "ConstantBuffer" => ParameterFlavor::Uniform,
        "Texture2D" => ParameterFlavor::SampledTexture,
        "StorageTexture2D" => ParameterFlavor::StorageTexture,
        "WriteTexture2D" => ParameterFlavor::WriteTexture,
        "Sampler" => ParameterFlavor::Sampler,
        _ => ParameterFlavor::Value,
    }
}

/// [`emit_kernel_in`] for the `grid` profile, the entry point of the tests; the expander passes its profile.
#[cfg(test)]
pub fn emit_kernel(
    module: &syn::ItemMod,
    kernel: &syn::ItemFn,
    options: KernelOptions,
) -> syn::Result<EmittedKernel> {
    emit_kernel_in(&GRID, module, kernel, options)
}

/// Emits the Slang of one kernel of `module`. Everything that depends on the execution shape (the stage attribute and
/// the entry signature, which parameters are the invocation and which take a binding slot, the statements before and
/// after the body, the resources the shape adds) comes from `stage`; the rest is the dialect's.
pub(crate) fn emit_kernel_in(
    stage: &dyn StageProfile,
    module: &syn::ItemMod,
    kernel: &syn::ItemFn,
    options: KernelOptions,
) -> syn::Result<EmittedKernel> {
    // A trait path call is emitted as the method call it stands for (slice m9.g1); the copy keeps the kernel's name.
    let desugared = crate::validate::traits::desugar(module)?;
    let module = desugared.as_ref().unwrap_or(module);
    let kernel = desugared
        .as_ref()
        .and_then(|copy| copy.content.as_ref())
        .and_then(|(_, items)| {
            items.iter().find_map(|item| match item {
                syn::Item::Fn(function) if function.sig.ident == kernel.sig.ident => Some(function),
                _ => None,
            })
        })
        .unwrap_or(kernel); // no trait: the caller's kernel is the module's own
    let rules = stage.binding_rules();
    let (_, items) = module.content.as_ref().expect("validated inline module");
    let lowering = module_lowering(module);
    let _scope = LoweringScope::enter(lowering);
    // The array fields of the module's structs, for `s.items.len()` in the kernel, the helpers and the methods.
    let _fields = FieldScope::enter(array_field_lengths(items));
    // The Slang names of the module's closures (slice m9.c).
    let _closure_names = ClosureNamesScope::enter(closure_names(items));
    let mut output = String::new();
    let mut table = LineTable::default();
    writeln!(
        output,
        "// Generated by GUST from Rust syntax. Slang is the compiler backend."
    )
    .unwrap();
    writeln!(output).unwrap();

    // Slang overload resolution distinguishes Rust's boolean and bitwise `!`.
    // Forwarding `!` verbatim would silently turn integer complement into a test.
    let mut not = NotUsage(false);
    not.visit_item_mod(module);
    if not.0 {
        writeln!(output, "bool __gust_not(bool value) {{ return !value; }}").unwrap();
        writeln!(output, "int __gust_not(int value) {{ return ~value; }}").unwrap();
        writeln!(output, "uint __gust_not(uint value) {{ return ~value; }}\n").unwrap();
    }

    // Rust `^` is logical xor on `bool` and bitwise xor on integers, but WGSL defines `^` only on integers, so a
    // `bool ^` forwarded verbatim fails at shader creation. Overload resolution picks the right body for each use.
    // The `_assign` forms take the place by `inout`, so `place ^= value` evaluates the place once.
    let mut xor = XorUsage(false);
    xor.visit_item_mod(module);
    if xor.0 {
        writeln!(
            output,
            "bool __gust_xor(bool a, bool b) {{ return a != b; }}"
        )
        .unwrap();
        writeln!(output, "int __gust_xor(int a, int b) {{ return a ^ b; }}").unwrap();
        writeln!(
            output,
            "uint __gust_xor(uint a, uint b) {{ return a ^ b; }}"
        )
        .unwrap();
        writeln!(
            output,
            "void __gust_xor_assign(inout bool a, bool b) {{ a = a != b; }}"
        )
        .unwrap();
        writeln!(
            output,
            "void __gust_xor_assign(inout int a, int b) {{ a = a ^ b; }}"
        )
        .unwrap();
        writeln!(
            output,
            "void __gust_xor_assign(inout uint a, uint b) {{ a = a ^ b; }}\n"
        )
        .unwrap();
    }

    // WGSL reads `name < .. >` as a template list, and slangc drops the parentheses around a comparison that is an
    // argument, so `f(a < b, c > d)` reaches WGSL with the same text and fails to parse. An identity call keeps
    // the `<` away from the list: the comparison sits inside its own parentheses.
    // Nothing in the source is scanned for it: `emit_list` records that it wrote a `__gust_keep(` call, and the
    // definition is inserted here, once the whole kernel is emitted, exactly when one was written.
    KEEP_USED.with(|used| used.set(false));
    let keep_at = output.len();

    // The subgroup operations that need a helper (M8, D140): overloads by operand type for the bitwise reductions, the
    // inclusive scans and the count. Only the kernel is scanned: a helper function holds none (GUST-E0745).
    let mut subgroups = SubgroupUsage::default();
    subgroups.visit_item_fn(kernel);
    output.push_str(&subgroup_helper_text(&subgroups.0));

    // Rust `Option<T>` lowers to Slang's `Optional<T>`. Generic helpers let the
    // payload type be inferred at each use instead of requiring annotations.
    let mut option = OptionUsage(false);
    option.visit_item_mod(module);
    if option.0 {
        writeln!(
            output,
            "Optional<T> __gust_some<T>(T value) {{ return value; }}"
        )
        .unwrap();
        writeln!(
            output,
            "T __gust_unwrap_or<T>(Optional<T> value, T fallback) {{ if (value.hasValue) {{ return value.value; }} return fallback; }}\n"
        )
        .unwrap();
    }

    let mut result = ResultUsage(false);
    result.visit_item_mod(module);
    if result.0 {
        writeln!(output, "struct __GustResult<T> {{ bool isOk; T value; }}").unwrap();
        writeln!(
            output,
            "__GustResult<T> __gust_ok<T>(T value) {{ __GustResult<T> result; result.isOk = true; result.value = value; return result; }}"
        )
        .unwrap();
        writeln!(
            output,
            "__GustResult<T> __gust_err<T>(T value) {{ __GustResult<T> result; result.isOk = false; result.value = value; return result; }}"
        )
        .unwrap();
        writeln!(
            output,
            "T __gust_unwrap_or<T>(__GustResult<T> result, T fallback) {{ if (result.isOk) {{ return result.value; }} return fallback; }}\n"
        )
        .unwrap();
    }

    // Rust `*` on a matrix is the matrix product, which Slang's `*` is not, and the macro does not know which operand
    // is a matrix. A module that names one sends every `*` and `*=` through the overloads of `__gust_mul`, one per
    // operand pair of the product table, and slangc resolves each call by the operand types.
    if lowering.mul {
        output.push_str(&mul_table::overload_text());
    }

    // Unary `-` on a matrix is not valid WGSL in slangc's own lowering (`mat4x4<f32>(0) - m`), so a module that names
    // a matrix sends every unary `-` through the overloads of `__gust_neg` (see `neg_table`).
    if lowering.neg {
        output.push_str(&neg_table::overload_text());
    }

    // slangc folds integer `/` and `%` and refuses a divisor that folds to zero, even in dead code (D74), so every `/`
    // and `%` with a divisor that is not a literal is a call to `__gust_div` or `__gust_rem` (see `div_table`).
    let mut division = DivisionUsage::default();
    division.visit_item_mod(module);
    if division.div {
        output.push_str(&div_table::module_overload_text(
            "div",
            "/",
            div_table::DIV_ROWS,
        ));
    }
    if division.rem {
        output.push_str(&div_table::module_overload_text(
            "rem",
            "%",
            div_table::REM_ROWS,
        ));
    }

    // The integer `dot` as a sum of products (see `dot_overload_text`); `.dot` is `__gust_dot` in such a module.
    if lowering.dot {
        output.push_str(&dot_overload_text());
    }

    // Buffers used with atomic operations need `Atomic<T>` element types in Slang
    // so WGSL emits `array<atomic<T>>` storage.
    let mut atomic = AtomicUsage(std::collections::HashSet::new());
    atomic.visit_item_fn(kernel);

    // Enums first, in source order: a function and a switch name them, and a struct cannot (GUST-E0211).
    for item in items {
        if let syn::Item::Enum(item) = item {
            emit_enum(&mut output, &mut table, item)?;
        }
    }
    for item in items {
        if let syn::Item::Struct(item) = item {
            let methods = methods_of(items, item);
            // The closures of the methods are free functions before the struct (slice m9.c); `Self` in them is the struct.
            {
                let _self_scope = SelfScope::enter(&item.ident.unraw().to_string());
                for method in &methods {
                    emit_hoisted_closures(&mut output, &mut table, &method.block)?;
                }
            }
            emit_struct(&mut output, &mut table, item, &methods)?;
        }
    }
    // The interfaces of the traits, then the extensions that conform to them (slice m9.g1): after the structs and enums
    // whose names they mention, before the helpers and the kernel that call through them.
    for item in items {
        if let syn::Item::Trait(item) = item {
            emit_interface(&mut output, &mut table, item)?;
        }
    }
    for item in items {
        if let syn::Item::Impl(block) = item
            && block.trait_.is_some()
        {
            emit_extension(&mut output, &mut table, block)?;
        }
    }

    let mut binding = 0_u32;
    let mut resources = Vec::new();
    // The texture parameters, by emitted name, in parameter order.
    let mut textures: Vec<String> = Vec::new();
    let mut invocation = None;
    // The ids derived from the dispatch-thread id (name, whether it is the group id), in parameter order.
    let mut derived_ids: Vec<(String, bool)> = Vec::new();
    // The `Workgroup<[T; N]>` parameters (name, Slang element type, length), for the guards of the SPIR-V route.
    let mut workgroup_arrays: Vec<guards::WorkgroupArray> = Vec::new();
    for argument in &kernel.sig.inputs {
        let (name, ty) = typed_argument(argument)?;
        let name = emit_ident(name);
        let flavor = parameter_flavor_in(stage, ty);
        match flavor {
            ParameterFlavor::Invocation => invocation = Some(name),
            ParameterFlavor::LocalId => derived_ids.push((name, false)),
            ParameterFlavor::GroupId => derived_ids.push((name, true)),
            ParameterFlavor::Workgroup => {
                let (element, length) = crate::validate::workgroup_array(ty)
                    .map_err(|message| diagnostics::error(Code::WorkgroupShape, ty, message))?;
                let element_type = emit_type(element)?;
                writeln!(output, "groupshared {element_type} {name}[{length}];").unwrap();
                workgroup_arrays.push(guards::WorkgroupArray {
                    name: name.clone(),
                    element: element_type,
                    length,
                });
            }
            // A resource the shape numbers (`gets_slot`, the rule of the descriptor and the record) is declared
            // with the next binding; one it leaves out is not declared at all.
            ParameterFlavor::StorageRead | ParameterFlavor::StorageReadWrite
                if rules.gets_slot(flavor) =>
            {
                let element = wrapper_element_type(ty)?;
                let wrapper = if flavor == ParameterFlavor::StorageRead {
                    "StructuredBuffer"
                } else {
                    "RWStructuredBuffer"
                };
                let element_type = if atomic.0.contains(&name) {
                    format!("Atomic<{}>", emit_type(element)?)
                } else {
                    // A buffer of arrays is `StructuredBuffer<T[N]>` (slice m9.r2a); the validator checked the length
                    // and the element before this point.
                    emit_value_type(element)?
                };
                writeln!(
                    output,
                    "[[vk::binding({binding}, 0)]] {wrapper}<{element_type}> {name};",
                )
                .unwrap();
                resources.push(name);
                binding += 1;
            }
            ParameterFlavor::Uniform if rules.gets_slot(flavor) => {
                let element = wrapper_element_type(ty)?;
                writeln!(
                    output,
                    "[[vk::binding({binding}, 0)]] ConstantBuffer<{}> {name};",
                    emit_type(element)?
                )
                .unwrap();
                binding += 1;
            }
            // A texture or a sampler (slice mx.t3) is declared with the next binding, like a buffer. Textures are
            // remembered for the prologue that reads their size.
            ParameterFlavor::SampledTexture
            | ParameterFlavor::StorageTexture
            | ParameterFlavor::WriteTexture
            | ParameterFlavor::Sampler
                if rules.gets_slot(flavor) =>
            {
                writeln!(
                    output,
                    "{}",
                    texture::declaration(flavor, ty, &name, binding)?
                )
                .unwrap();
                if flavor.is_texture() {
                    textures.push(name);
                }
                binding += 1;
            }
            ParameterFlavor::StorageRead
            | ParameterFlavor::StorageReadWrite
            | ParameterFlavor::Uniform
            | ParameterFlavor::SampledTexture
            | ParameterFlavor::StorageTexture
            | ParameterFlavor::WriteTexture
            | ParameterFlavor::Sampler => {}
            ParameterFlavor::Value => {
                return Err(diagnostics::error(
                    Code::EntryPointParameter,
                    ty,
                    rules.value_refusal,
                ));
            }
        }
    }
    writeln!(output).unwrap();

    for item in items {
        let syn::Item::Fn(function) = item else {
            continue;
        };
        // A sibling kernel is recognised by its attribute alone: its options are checked under its own profile (the
        // validator runs every kernel of the module), not under this one, which may refuse them (`workgroup_size`).
        if function.sig.ident == kernel.sig.ident
            || function
                .attrs
                .iter()
                .any(|attribute| attribute.path().is_ident("kernel"))
        {
            continue;
        }
        emit_hoisted_closures(&mut output, &mut table, &function.block)?;
        emit_helper(&mut output, &mut table, function)?;
    }
    emit_hoisted_closures(&mut output, &mut table, &kernel.block)?;

    let invocation = invocation.ok_or_else(|| {
        diagnostics::error(
            Code::EntryPointDispatchParameter,
            &kernel.sig,
            rules.missing_dispatch_emit,
        )
    })?;
    let entry_point = format!("gpu_{}_{}", module.ident, kernel.sig.ident);
    let derived_ids = derived_ids
        .into_iter()
        .map(|(name, group)| DerivedId { name, group })
        .collect::<Vec<_>>();
    let mut facts = KernelFacts {
        entry_point: &entry_point,
        workgroup_size: options.workgroup_size,
        invocation: &invocation,
        resources: &resources,
        derived_ids: &derived_ids,
        next_binding: binding,
    };
    // The resources the shape declares beside the kernel's own take the next binding slots, in order.
    for extra in stage.extra_bindings(&facts) {
        writeln!(
            output,
            "[[vk::binding({}, 0)]] ConstantBuffer<{}> {};",
            facts.next_binding, extra.slang_type, extra.name
        )
        .unwrap();
        facts.next_binding += 1;
    }
    writeln!(
        output,
        "// @rust kernel: {}::{}",
        module.ident, kernel.sig.ident
    )
    .unwrap();
    writeln!(output, "[shader(\"{}\")]", stage.slang_stage()).unwrap();
    writeln!(output, "{}", stage.entry_signature(&facts)).unwrap();
    writeln!(output, "{{").unwrap();
    output.push_str(&stage.prologue(&facts));
    if guards::enabled() && stage.slang_stage() == "compute" {
        output.push_str(&guards::workgroup_zeroing(
            &workgroup_arrays,
            facts.invocation,
            facts.workgroup_size,
        ));
    }
    output.push_str(&texture::dimension_prologue(&textures, kernel));
    let mut kernel_arrays = array_lengths(&kernel.sig.inputs, &kernel.block);
    // A `Workgroup<[T; N]>` parameter is an array of known length: its dynamic indices are clamped like a local array's.
    // Only with the guards on: with them off the tables, and with them every diagnostic and text, are what they were.
    for array in workgroup_arrays.iter().filter(|_| guards::enabled()) {
        kernel_arrays
            .entry(array.name.clone())
            .or_insert(Some(array.length));
    }
    let _arrays = ArrayScope::enter(kernel_arrays);
    let _closures = ClosureScope::enter(&kernel.block);
    emit_statements(&mut output, &mut table, &kernel.block.stmts, 1, false)?;
    output.push_str(&stage.epilogue(&facts));
    writeln!(output, "}}").unwrap();
    let mut records = table.records;
    if KEEP_USED.with(std::cell::Cell::get) {
        // The definition goes where it always went, before every struct, function and kernel that may call it. It is
        // written after them, so the records of every line from there on move down by the lines it adds. The text
        // is inserted, never searched for, so its own `__gust_keep(` cannot make the check true.
        let helper = format!("bool {KEEP_HELPER}(bool value) {{ return value; }}\n\n");
        let at_line = output.as_bytes()[..keep_at]
            .iter()
            .filter(|byte| **byte == b'\n')
            .count() as u32
            + 1;
        let added = helper.matches('\n').count() as u32;
        output.insert_str(keep_at, &helper);
        for record in &mut records {
            if record.slang_start >= at_line {
                record.slang_start += added;
                record.slang_end += added;
            }
        }
    }
    Ok(EmittedKernel {
        source: output,
        lines: records,
    })
}

/// The functions of the `impl` blocks of `item`, in source order (slices m9.a1 and m9.a2): the validator accepts any
/// number of inherent blocks per struct, and a Slang struct holds its member functions.
fn methods_of<'a>(items: &'a [syn::Item], item: &syn::ItemStruct) -> Vec<&'a syn::ImplItemFn> {
    items
        .iter()
        .filter_map(|candidate| match candidate {
            syn::Item::Impl(block)
                if block.trait_.is_none()
                    && crate::validate::impl_self_name(block)
                        .is_some_and(|name| item.ident.unraw() == name) =>
            {
                Some(block)
            }
            _ => None,
        })
        .flat_map(|block| &block.items)
        .filter_map(|inner| match inner {
            syn::ImplItem::Fn(method) => Some(method),
            _ => None,
        })
        .collect()
}

/// `enum E : uint { A, B = 3, C };` (slice m9.e1, D157): a field-less enum of the module, with the integer type of its
/// `repr` (`uint`, or `int` for `#[repr(i32)]`). The discriminants are the integer literals the validator accepted, so
/// the values count up in Slang exactly as they do in Rust.
fn emit_enum(output: &mut String, table: &mut LineTable, item: &syn::ItemEnum) -> syn::Result<()> {
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

/// `switch (scrutinee) { case E::A: case E::B: { .. break; } default: { .. break; } }` for a `match` on an enum or on an
/// integer (slices m9.e1 and m9.e2, D157). The scrutinee is one expression in the `switch`, so it is evaluated once.
/// An arm that ends in `return` gets no `break` (slangc warns about unreachable code). An enum `match` without a `_`
/// arm makes its last arm the `default` (in any position: rustc proved the arms exhaustive, and WGSL needs the function
/// to return in every case of a `switch`; measured: slangc error 41009 otherwise). An integer `match` always has `_`.
///
/// With `target` (the value form, `let x: T = match e { .. };`) every arm assigns its value to `target`, which the
/// caller declared before the `switch`; the `default` makes that assignment total.
fn emit_switch(
    output: &mut String,
    table: &mut LineTable,
    expression: &syn::ExprMatch,
    indent: usize,
    index: usize,
    target: Option<&str>,
) -> syn::Result<()> {
    let padding = "    ".repeat(indent);
    let inner = "    ".repeat(indent + 1);
    let has_wildcard = expression
        .arms
        .iter()
        .any(|arm| matches!(arm.pat, syn::Pat::Wild(_)));
    writeln!(output, "{padding}// @gust construct: match #{index}").unwrap();
    writeln!(
        output,
        "{padding}switch ({})",
        emit_expression(&expression.expr)?
    )
    .unwrap();
    writeln!(output, "{padding}{{").unwrap();
    for (position, arm) in expression.arms.iter().enumerate() {
        let last = position + 1 == expression.arms.len();
        if matches!(arm.pat, syn::Pat::Wild(_)) || (!has_wildcard && last) {
            writeln!(output, "{inner}default:").unwrap();
        } else {
            let mut alternatives = Vec::new();
            flatten_alternatives(&arm.pat, &mut alternatives);
            for pattern in alternatives {
                let label = match pattern {
                    syn::Pat::Path(path) => emit_variant_path(&path.path),
                    syn::Pat::Lit(literal) => emit_case_literal(literal)?,
                    other => {
                        return Err(diagnostics::error(
                            Code::MatchPattern,
                            other,
                            "a `match` arm is `E::A`, an integer literal, `A | B` or `_`",
                        ));
                    }
                };
                writeln!(output, "{inner}case {label}:").unwrap();
            }
        }
        writeln!(output, "{inner}{{").unwrap();
        let diverges = if let Some(target) = target {
            emit_value_arm(output, table, &arm.body, indent + 2, target)?
        } else {
            let syn::Expr::Block(body) = arm.body.as_ref() else {
                return Err(diagnostics::error(
                    Code::MatchArmBlock,
                    &arm.body,
                    "each arm of a statement `match` is a block",
                ));
            };
            emit_statements(output, table, &body.block.stmts, indent + 2, false)?;
            matches!(
                body.block.stmts.last(),
                Some(syn::Stmt::Expr(syn::Expr::Return(_), _))
            )
        };
        if !diverges {
            writeln!(output, "{}break;", "    ".repeat(indent + 2)).unwrap();
        }
        writeln!(output, "{inner}}}").unwrap();
    }
    writeln!(output, "{padding}}}").unwrap();
    Ok(())
}

/// `T x;` and the `switch` that assigns it, for `let x: T = match e { .. };` (slice m9.e2). WGSL zero-initialises a `var`
/// declared without a value, and every path of the `switch` assigns it (measured: slangc 2026.21, WGSL and SPIR-V).
/// When the name of the local occurs in the `match` (the `match` reads an outer local of that name, or an arm declares
/// one), the switch assigns a temporary and the local is declared from it after, so the `match` never reads the new
/// local. A shadow in the same Slang scope as the outer local still fails in slangc (E30200, as for every `let`); the
/// shadowing `let` belongs in a block of its own (SEMANTICS.md, section 9).
fn emit_value_match(
    output: &mut String,
    table: &mut LineTable,
    name: &syn::Ident,
    ty: &syn::Type,
    expression: &syn::ExprMatch,
    indent: usize,
    index: usize,
) -> syn::Result<()> {
    let padding = "    ".repeat(indent);
    let declared = emit_ident(name);
    if matches!(unparenthesized_type(ty), syn::Type::Array(_)) {
        return Err(diagnostics::error(
            Code::UnsupportedExpression,
            expression,
            "a `match` does not give an array value (arrays are built by a literal or a repeat, or returned by a helper): use `if`/`else` statements that assign the elements, or a helper that returns the array",
        ));
    }
    let ty = emit_type(ty)?;
    let target = if mentions_ident(expression.to_token_stream(), &name.unraw().to_string()) {
        format!("__gust_match_{indent}_{index}")
    } else {
        declared.clone()
    };
    writeln!(output, "{padding}{ty} {target};").unwrap();
    emit_switch(output, table, expression, indent, index, Some(&target))?;
    if target != declared {
        writeln!(output, "{padding}{ty} {declared} = {target};").unwrap();
    }
    Ok(())
}

/// Whether the identifier `name` occurs anywhere in `tokens`.
fn mentions_ident(tokens: proc_macro2::TokenStream, name: &str) -> bool {
    tokens.into_iter().any(|tree| match tree {
        proc_macro2::TokenTree::Ident(ident) => ident.unraw() == name,
        proc_macro2::TokenTree::Group(group) => mentions_ident(group.stream(), name),
        _ => false,
    })
}

/// The body of an arm of a value `match` at `indent`: assigns its value to `target`. Returns whether the arm ends in a
/// `return`, so that it needs no `break`.
fn emit_value_arm(
    output: &mut String,
    table: &mut LineTable,
    body: &syn::Expr,
    indent: usize,
    target: &str,
) -> syn::Result<bool> {
    match body {
        syn::Expr::Paren(inner) => emit_value_arm(output, table, &inner.expr, indent, target),
        syn::Expr::Group(inner) => emit_value_arm(output, table, &inner.expr, indent, target),
        syn::Expr::Block(block) if block.label.is_none() => {
            emit_value_block(output, table, &block.block, indent, target)
        }
        other => emit_value_tail(output, table, other, indent, 0, target),
    }
}

/// The statements of a block, then its value (the last statement, an expression without `;`) assigned to `target`.
fn emit_value_block(
    output: &mut String,
    table: &mut LineTable,
    block: &syn::Block,
    indent: usize,
    target: &str,
) -> syn::Result<bool> {
    match block.stmts.split_last() {
        Some((syn::Stmt::Expr(tail, None), head)) if !matches!(tail, syn::Expr::Return(_)) => {
            emit_statements(output, table, head, indent, false)?;
            let record = table.begin(output, tail.span());
            let diverges = emit_value_tail(output, table, tail, indent, head.len(), target)?;
            table.finish_written(output, record);
            Ok(diverges)
        }
        _ => {
            emit_statements(output, table, &block.stmts, indent, false)?;
            Ok(matches!(
                block.stmts.last(),
                Some(syn::Stmt::Expr(syn::Expr::Return(_), _))
            ))
        }
    }
}

/// One value assigned to `target`: a nested value `match` is a `switch` of its own that assigns the same target, a
/// struct literal is built in a temporary first (as for a `let`), anything else is `target = value;`.
fn emit_value_tail(
    output: &mut String,
    table: &mut LineTable,
    value: &syn::Expr,
    indent: usize,
    index: usize,
    target: &str,
) -> syn::Result<bool> {
    let padding = "    ".repeat(indent);
    match value {
        syn::Expr::Paren(inner) => {
            emit_value_tail(output, table, &inner.expr, indent, index, target)
        }
        syn::Expr::Group(inner) => {
            emit_value_tail(output, table, &inner.expr, indent, index, target)
        }
        syn::Expr::Match(inner) if crate::validate::is_switch_match(inner) => {
            emit_switch(output, table, inner, indent, index, Some(target))?;
            Ok(false)
        }
        syn::Expr::Block(block) if block.label.is_none() => {
            writeln!(output, "{padding}{{").unwrap();
            let diverges = emit_value_block(output, table, &block.block, indent + 1, target)?;
            writeln!(output, "{padding}}}").unwrap();
            Ok(diverges)
        }
        syn::Expr::Return(_) => {
            writeln!(output, "{padding}{};", emit_expression(value)?).unwrap();
            Ok(true)
        }
        syn::Expr::Struct(literal) => {
            let temporary = emit_struct_value(output, literal, indent, index)?;
            writeln!(output, "{padding}{target} = {temporary};").unwrap();
            Ok(false)
        }
        other => {
            writeln!(output, "{padding}{target} = {};", emit_expression(other)?).unwrap();
            Ok(false)
        }
    }
}

/// The label of an integer pattern: `0`, `uint(3000000000)`, `int(-2)`. The text follows `integer_literal`'s rules for
/// the type of the literal; a negative value is written as `int(-n)`, which slangc accepts as a case label (measured).
fn emit_case_literal(literal: &syn::ExprLit) -> syn::Result<String> {
    let syn::Lit::Int(value) = &literal.lit else {
        return Err(diagnostics::error(
            Code::MatchPattern,
            literal,
            "only integer literals are patterns",
        ));
    };
    let number: i128 = value.base10_parse().map_err(|_| {
        diagnostics::error(
            Code::MatchPattern,
            literal,
            "this integer literal does not parse",
        )
    })?;
    Ok(match value.suffix() {
        "u32" => format!("uint({number})"),
        "i32" => format!("int({number})"),
        "" if number < 0 => format!("int({number})"),
        "" if number > i128::from(i32::MAX) => format!("uint({number})"),
        "" => number.to_string(),
        _ => {
            return Err(diagnostics::error(
                Code::UnsupportedLiteral,
                literal,
                "GPU literals support 32-bit numbers (i32, u32) only",
            ));
        }
    })
}

/// The alternatives of an arm pattern: `A | B` flattened, parentheses looked through.
fn flatten_alternatives<'p>(pattern: &'p syn::Pat, out: &mut Vec<&'p syn::Pat>) {
    match pattern {
        syn::Pat::Or(or) => {
            for case in &or.cases {
                flatten_alternatives(case, out);
            }
        }
        syn::Pat::Paren(inner) => flatten_alternatives(&inner.pat, out),
        other => out.push(other),
    }
}

/// `E::V` as Slang spells it: both names renamed as their declarations are.
fn emit_variant_path(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|segment| emit_ident(&segment.ident))
        .collect::<Vec<_>>()
        .join("::")
}

fn emit_struct(
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
        if let syn::Type::Array(array) = unparenthesized_type(&field.ty) {
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
    static IN_INTERFACE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Sets [`IN_INTERFACE`] for a scope and restores the previous value when dropped.
struct InterfaceScope(bool);

impl InterfaceScope {
    fn enter() -> Self {
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
fn emit_interface(
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
fn emit_extension(
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
fn emit_hoisted_closures(
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

fn emit_helper(
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

fn emit_statements(
    output: &mut String,
    table: &mut LineTable,
    statements: &[syn::Stmt],
    indent: usize,
    return_tail: bool,
) -> syn::Result<()> {
    for (index, statement) in statements.iter().enumerate() {
        // A closure binding emits nothing: its body was hoisted before the owner, and a call names the hoisted function.
        if matches!(statement, syn::Stmt::Local(local) if closure_binding(local).is_some()) {
            continue;
        }
        let is_tail = return_tail && index + 1 == statements.len();
        // One record per statement, opened before and closed after the text it
        // lowers to, so nested statements land inside their parent's range.
        let record = table.begin(output, statement.span());
        emit_statement(output, table, statement, indent, index, is_tail)?;
        table.finish_written(output, record);
    }
    Ok(())
}

fn emit_statement(
    output: &mut String,
    table: &mut LineTable,
    statement: &syn::Stmt,
    indent: usize,
    index: usize,
    is_tail: bool,
) -> syn::Result<()> {
    let padding = "    ".repeat(indent);
    match statement {
        syn::Stmt::Local(local) if tuple_pattern(&local.pat).is_some() => {
            let (tuple, annotation) = tuple_pattern(&local.pat).expect("guarded by the match arm");
            // The validator looks through parentheses around the annotation (`((u32, f32))`), so does the emission.
            fn unparenthesized(ty: &syn::Type) -> &syn::Type {
                match ty {
                    syn::Type::Paren(inner) => unparenthesized(&inner.elem),
                    syn::Type::Group(inner) => unparenthesized(&inner.elem),
                    other => other,
                }
            }
            let annotation = annotation.map(unparenthesized);
            let initial = local.init.as_ref().ok_or_else(|| {
                diagnostics::error(
                    Code::LetNeedsInitializer,
                    local,
                    "Slang let bindings require an initializer",
                )
            })?;
            // One temporary holds the whole value (evaluated once, in Rust's order), then one local per element.
            let temporary = format!("__gust_tuple_{index}");
            let element_types: Vec<Option<String>> = match annotation {
                Some(syn::Type::Tuple(types)) if types.elems.len() == tuple.elems.len() => types
                    .elems
                    .iter()
                    .map(|ty| emit_type(ty).map(Some))
                    .collect::<syn::Result<_>>()?,
                Some(other) => {
                    return Err(diagnostics::error(
                        Code::TupleShape,
                        other,
                        "a tuple destructuring is annotated with a tuple type of the same length",
                    ));
                }
                None => vec![None; tuple.elems.len()],
            };
            let declaration = annotation
                .map(emit_type)
                .transpose()?
                .unwrap_or_else(|| "var".to_owned());
            writeln!(
                output,
                "{padding}{declaration} {temporary} = {};",
                emit_expression(&initial.expr)?
            )
            .unwrap();
            for (position, (element, ty)) in tuple.elems.iter().zip(&element_types).enumerate() {
                let syn::Pat::Ident(name) = element else {
                    return Err(diagnostics::error(
                        Code::LetNeedsIdentifier,
                        element,
                        "a tuple destructuring binds plain identifiers",
                    ));
                };
                writeln!(
                    output,
                    "{padding}{} {} = {temporary}._{position};",
                    ty.as_deref().unwrap_or("var"),
                    emit_ident(&name.ident)
                )
                .unwrap();
            }
        }
        syn::Stmt::Local(local) => {
            let (pattern, annotation) = local_binding(&local.pat)?;
            let initial = local.init.as_ref().ok_or_else(|| {
                diagnostics::error(
                    Code::LetNeedsInitializer,
                    local,
                    "Slang let bindings require an initializer",
                )
            })?;
            // The value form (slice m9.e2): `T x;` and a `switch` that assigns it.
            let mut value = initial.expr.as_ref();
            while let syn::Expr::Paren(syn::ExprParen { expr, .. })
            | syn::Expr::Group(syn::ExprGroup { expr, .. }) = value
            {
                value = expr;
            }
            // Arrays (slice m9.r1). Slang cannot infer the type of `{ .. }`, so the `let` is typed.
            let array_annotation = annotation
                .map(unparenthesized_type)
                .filter(|ty| matches!(ty, syn::Type::Array(_)));
            if matches!(value, syn::Expr::Array(_) | syn::Expr::Repeat(_)) {
                let Some(array_type) = array_annotation else {
                    return Err(diagnostics::error(
                        Code::UnsupportedExpression,
                        value,
                        "an array literal or repeat initializes a typed `let`, because Slang cannot infer the type of `{ .. }`: write `let a: [T; N] = ..;`",
                    ));
                };
                if let syn::Expr::Repeat(repeat) = value {
                    return emit_repeat(
                        output,
                        indent,
                        index,
                        repeat,
                        &RepeatTarget::Let {
                            array_type,
                            name: &pattern.ident,
                        },
                    );
                }
            }
            if let (Some(array_type), false) = (
                array_annotation,
                matches!(value, syn::Expr::Match(_) | syn::Expr::Struct(_)),
            ) {
                // An initializer that names the binding reads the previous one (`let a: [u32; 2] = [a[1], a[0]];`),
                // which Slang would resolve to the new local: build the value in a temporary first.
                let declaration = emit_value_type(array_type)?;
                let name = emit_ident(&pattern.ident);
                let text = emit_value_expression(&initial.expr)?;
                if mentions_ident(
                    initial.expr.to_token_stream(),
                    &pattern.ident.unraw().to_string(),
                ) {
                    let temporary = format!("__gust_arr_{index}");
                    writeln!(output, "{padding}{declaration} {temporary} = {text};").unwrap();
                    writeln!(output, "{padding}{declaration} {name} = {temporary};").unwrap();
                } else {
                    writeln!(output, "{padding}{declaration} {name} = {text};").unwrap();
                }
                return Ok(());
            }
            if let syn::Expr::Match(expression) = value
                && crate::validate::is_switch_match(expression)
            {
                let ty = annotation.ok_or_else(|| {
                    diagnostics::error(
                        Code::ValueMatchForm,
                        expression,
                        "a `match` that gives a value needs a typed `let`",
                    )
                })?;
                return emit_value_match(
                    output,
                    table,
                    &pattern.ident,
                    ty,
                    expression,
                    indent,
                    index,
                );
            }
            // Finish the initializer before introducing its binding: an inner
            // `let p = Pair { first: p.second, ... }` reads the outer `p`.
            if let syn::Expr::Struct(value) = initial.expr.as_ref() {
                let temporary = emit_struct_value(output, value, indent, index)?;
                let ty = match annotation {
                    Some(ty) => emit_type(ty)?,
                    None => struct_literal_type(value)?,
                };
                writeln!(
                    output,
                    "{padding}{ty} {} = {temporary};",
                    emit_ident(&pattern.ident)
                )
                .unwrap();
                return Ok(());
            }
            let declaration = annotation
                .map(emit_type)
                .transpose()?
                .unwrap_or_else(|| "var".to_owned());
            writeln!(
                output,
                "{padding}{declaration} {} = {};",
                emit_ident(&pattern.ident),
                emit_expression(&initial.expr)?
            )
            .unwrap();
        }
        syn::Stmt::Expr(expression, semicolon) => match expression {
            syn::Expr::If(expression) => emit_if(
                output,
                table,
                expression,
                indent,
                is_tail && semicolon.is_none(),
                index,
            )?,
            syn::Expr::Match(expression) if crate::validate::is_switch_match(expression) => {
                emit_switch(output, table, expression, indent, index, None)?
            }
            syn::Expr::Match(expression) => {
                emit_result_match(output, table, expression, indent, index)?
            }
            syn::Expr::ForLoop(expression) => emit_for(output, table, expression, indent, index)?,
            syn::Expr::Break(jump) if jump.label.is_none() && jump.expr.is_none() => {
                writeln!(output, "{padding}break;").unwrap();
            }
            syn::Expr::Continue(jump) if jump.label.is_none() => {
                writeln!(output, "{padding}continue;").unwrap();
            }
            syn::Expr::Block(expression) => {
                writeln!(output, "{padding}{{").unwrap();
                emit_statements(
                    output,
                    table,
                    &expression.block.stmts,
                    indent + 1,
                    is_tail && semicolon.is_none(),
                )?;
                writeln!(output, "{padding}}}").unwrap();
            }
            // `a = [e; N];` (slice m9.r1): the operand is evaluated once, before the place is written.
            syn::Expr::Assign(assign)
                if matches!(unparenthesized_expr(&assign.right), syn::Expr::Repeat(_)) =>
            {
                let syn::Expr::Repeat(repeat) = unparenthesized_expr(&assign.right) else {
                    unreachable!("guarded by the match arm");
                };
                emit_repeat(
                    output,
                    indent,
                    index,
                    repeat,
                    &RepeatTarget::Assign {
                        place: emit_expression(&assign.left)?,
                    },
                )?;
            }
            // `return [e; N];` and a repeat as the tail value of a function.
            syn::Expr::Return(ret)
                if ret.expr.as_deref().is_some_and(|value| {
                    matches!(unparenthesized_expr(value), syn::Expr::Repeat(_))
                }) =>
            {
                let Some(syn::Expr::Repeat(repeat)) = ret.expr.as_deref().map(unparenthesized_expr)
                else {
                    unreachable!("guarded by the match arm");
                };
                emit_repeat(output, indent, index, repeat, &RepeatTarget::Return)?;
            }
            syn::Expr::Repeat(repeat) if is_tail && semicolon.is_none() => {
                emit_repeat(output, indent, index, repeat, &RepeatTarget::Return)?;
            }
            syn::Expr::Assign(assign) if matches!(assign.right.as_ref(), syn::Expr::Struct(_)) => {
                // Rust evaluates the complete RHS before evaluating/writing the
                // destination. Direct field writes break swaps and aliasing.
                let target = emit_expression(&assign.left)?;
                let value = match assign.right.as_ref() {
                    syn::Expr::Struct(value) => value,
                    _ => unreachable!("guarded by the match arm"),
                };
                let temporary = emit_struct_value(output, value, indent, index)?;
                writeln!(output, "{padding}{target} = {temporary};").unwrap();
            }
            _ => {
                let prefix = if is_tail
                    && semicolon.is_none()
                    && !matches!(expression, syn::Expr::Return(_))
                {
                    "return "
                } else {
                    ""
                };
                // A tail value may be an array literal (the returned array); a statement may not.
                let text = if prefix.is_empty() {
                    emit_expression(expression)?
                } else {
                    emit_value_expression(expression)?
                };
                writeln!(output, "{padding}{prefix}{text};").unwrap();
            }
        },
        syn::Stmt::Item(item) => {
            return Err(diagnostics::error(
                Code::NestedItem,
                item,
                "items inside Slang functions are not supported",
            ));
        }
        syn::Stmt::Macro(item) => {
            return Err(diagnostics::error(
                Code::MacroStatement,
                item,
                "macros cannot run in Slang",
            ));
        }
    }
    Ok(())
}

/// Build a value before assigning its destination. Statement indices are unique
/// within each emitted lexical block, and the validator reserves `__gust_` names.
fn emit_struct_value(
    output: &mut String,
    value: &syn::ExprStruct,
    indent: usize,
    index: usize,
) -> syn::Result<String> {
    let temporary = format!("__gust_struct_{index}");
    writeln!(
        output,
        "{}{} {temporary};",
        "    ".repeat(indent),
        struct_literal_type(value)?
    )
    .unwrap();
    emit_struct_assign(output, &temporary, value, indent)?;
    Ok(temporary)
}

/// Lower a struct literal to per-field assignments on a fresh temporary.
///
/// Slang has no field-name or designated initializers, so `Pair { first = 1 }`
/// is not valid; fields are initialized by name in source order. The caller must
/// not pass the final destination, which may be referenced by the initializer.
fn emit_struct_assign(
    output: &mut String,
    target: &str,
    value: &syn::ExprStruct,
    indent: usize,
) -> syn::Result<()> {
    let padding = "    ".repeat(indent);
    for field in &value.fields {
        let name = match &field.member {
            syn::Member::Named(ident) => emit_ident(ident),
            syn::Member::Unnamed(_) => {
                return Err(diagnostics::error(
                    Code::StructLiteralFieldName,
                    field,
                    "GPU struct literal fields need names",
                ));
            }
        };
        writeln!(
            output,
            "{padding}{target}.{name} = {};",
            // An array literal is a field value with a lowering (`tmp.items = { .. }`, slice m9.r2a); any other
            // expression goes through `emit_expression` as before.
            emit_value_expression(&field.expr)?
        )
        .unwrap();
    }
    Ok(())
}

/// The declared type of a struct literal, rendered as a Slang type name.
fn struct_literal_type(value: &syn::ExprStruct) -> syn::Result<String> {
    emit_type(&syn::Type::Path(syn::TypePath {
        qself: None,
        path: value.path.clone(),
    }))
}

pub fn local_binding(pattern: &syn::Pat) -> syn::Result<(&syn::PatIdent, Option<&syn::Type>)> {
    match pattern {
        syn::Pat::Ident(name) => Ok((name, None)),
        syn::Pat::Type(typed) => match typed.pat.as_ref() {
            syn::Pat::Ident(name) => Ok((name, Some(&typed.ty))),
            _ => Err(diagnostics::error(
                Code::LetBindingForm,
                pattern,
                "GPU let bindings require a single identifier, optionally with a type",
            )),
        },
        _ => Err(diagnostics::error(
            Code::LetBindingForm,
            pattern,
            "GPU let bindings require a single identifier, optionally with a type",
        )),
    }
}

/// A tuple destructuring in a `let`: `(a, b)`, with the annotation of `(a, b): (A, B)` when there is one.
pub(crate) fn tuple_pattern(pattern: &syn::Pat) -> Option<(&syn::PatTuple, Option<&syn::Type>)> {
    match pattern {
        syn::Pat::Tuple(tuple) => Some((tuple, None)),
        syn::Pat::Type(typed) => match typed.pat.as_ref() {
            syn::Pat::Tuple(tuple) => Some((tuple, Some(&typed.ty))),
            _ => None,
        },
        _ => None,
    }
}

fn emit_if(
    output: &mut String,
    table: &mut LineTable,
    expression: &syn::ExprIf,
    indent: usize,
    return_tail: bool,
    index: usize,
) -> syn::Result<()> {
    let padding = "    ".repeat(indent);
    if let syn::Expr::Let(binding) = expression.cond.as_ref() {
        // `if let Some(x) = value`: evaluate the scrutinee once into a reserved
        // temporary, then bind the payload inside the taken branch.
        let (name, temporary, condition) =
            if let Some(name) = crate::validate::option_pattern(&binding.pat) {
                (name, format!("__gust_opt_{index}"), "hasValue")
            } else if let Some(name) = crate::validate::result_pattern(&binding.pat) {
                (name, format!("__gust_result_{index}"), "isOk")
            } else {
                return Err(diagnostics::error(
                    Code::IfLetPattern,
                    &binding.pat,
                    "`if let` supports only the `Some(identifier)` or `Ok(identifier)` pattern",
                ));
            };
        writeln!(
            output,
            "{padding}var {temporary} = {};",
            emit_expression(&binding.expr)?
        )
        .unwrap();
        writeln!(output, "{padding}if ({temporary}.{condition})").unwrap();
        writeln!(output, "{padding}{{").unwrap();
        writeln!(
            output,
            "{}var {} = {temporary}.value;",
            "    ".repeat(indent + 1),
            emit_ident(&name.ident)
        )
        .unwrap();
    } else {
        writeln!(
            output,
            "{padding}if ({})",
            emit_expression(&expression.cond)?
        )
        .unwrap();
        writeln!(output, "{padding}{{").unwrap();
    }
    emit_statements(
        output,
        table,
        &expression.then_branch.stmts,
        indent + 1,
        return_tail,
    )?;
    writeln!(output, "{padding}}}").unwrap();
    if let Some((_, alternative)) = &expression.else_branch {
        match alternative.as_ref() {
            syn::Expr::If(nested) if !matches!(nested.cond.as_ref(), syn::Expr::Let(_)) => {
                write!(output, "{padding}else ").unwrap();
                emit_if(output, table, nested, indent, return_tail, index)?;
            }
            syn::Expr::If(nested) => {
                // `else if let` needs a scope of its own for the scrutinee temporary.
                writeln!(output, "{padding}else").unwrap();
                writeln!(output, "{padding}{{").unwrap();
                emit_if(output, table, nested, indent + 1, return_tail, index)?;
                writeln!(output, "{padding}}}").unwrap();
            }
            other => {
                writeln!(output, "{padding}else").unwrap();
                writeln!(output, "{padding}{{").unwrap();
                match other {
                    syn::Expr::Block(block) => {
                        emit_statements(output, table, &block.block.stmts, indent + 1, return_tail)?
                    }
                    other => writeln!(
                        output,
                        "{}{};",
                        "    ".repeat(indent + 1),
                        emit_expression(other)?
                    )
                    .unwrap(),
                }
                writeln!(output, "{padding}}}").unwrap();
            }
        }
    }
    Ok(())
}

fn emit_result_match(
    output: &mut String,
    table: &mut LineTable,
    expression: &syn::ExprMatch,
    indent: usize,
    index: usize,
) -> syn::Result<()> {
    let (ok_name, ok_body, err_name, err_body) = crate::validate::result_match_arms(expression)?;
    let padding = "    ".repeat(indent);
    let temporary = format!("__gust_match_{index}");
    writeln!(output, "{padding}// @gust construct: match #{index}").unwrap();
    writeln!(
        output,
        "{padding}var {temporary} = {};",
        emit_expression(&expression.expr)?
    )
    .unwrap();
    writeln!(output, "{padding}if ({temporary}.isOk)").unwrap();
    writeln!(output, "{padding}{{").unwrap();
    writeln!(
        output,
        "{}var {} = {temporary}.value;",
        "    ".repeat(indent + 1),
        emit_ident(&ok_name.ident)
    )
    .unwrap();
    emit_statements(output, table, &ok_body.stmts, indent + 1, false)?;
    writeln!(output, "{padding}}}").unwrap();
    writeln!(output, "{padding}else").unwrap();
    writeln!(output, "{padding}{{").unwrap();
    writeln!(
        output,
        "{}var {} = {temporary}.value;",
        "    ".repeat(indent + 1),
        emit_ident(&err_name.ident)
    )
    .unwrap();
    emit_statements(output, table, &err_body.stmts, indent + 1, false)?;
    writeln!(output, "{padding}}}").unwrap();
    Ok(())
}

/// Lower `for v in start..end { .. }` (D18). The end bound is evaluated once into
/// a reserved temporary so the trip count is fixed at entry, as with Rust's
/// `Range`; the counter takes its Slang type from the start expression. The
/// statement index keeps the temporary unique within its block, and a `_`
/// pattern gets a reserved counter name.
fn emit_for(
    output: &mut String,
    table: &mut LineTable,
    expression: &syn::ExprForLoop,
    indent: usize,
    index: usize,
) -> syn::Result<()> {
    let padding = "    ".repeat(indent);
    let variable = match crate::validate::loop_variable(&expression.pat) {
        Ok(Some(variable)) => emit_ident(&variable.ident),
        Ok(None) => format!("__gust_iter_{index}"),
        Err((code, message)) => {
            return Err(diagnostics::error(code, &expression.pat, message));
        }
    };
    let (start, end) = crate::validate::range_bounds(&expression.expr)
        .map_err(|(code, message)| diagnostics::error(code, &expression.expr, message))?;
    let end_temporary = format!("__gust_end_{index}");
    writeln!(
        output,
        "{padding}var {end_temporary} = {};",
        emit_expression(end)?
    )
    .unwrap();
    writeln!(
        output,
        "{padding}for (var {variable} = {}; {variable} < {end_temporary}; {variable}++)",
        emit_expression(start)?
    )
    .unwrap();
    writeln!(output, "{padding}{{").unwrap();
    emit_statements(output, table, &expression.body.stmts, indent + 1, false)?;
    writeln!(output, "{padding}}}").unwrap();
    Ok(())
}

fn emit_expression(expression: &syn::Expr) -> syn::Result<String> {
    Ok(match expression {
        syn::Expr::Array(array) => {
            // Slang takes `{ .. }` where the type is known: a typed `let`, an assignment, a call argument or a returned
            // value (`emit_value_expression` says so); anywhere else it is no array.
            if !ARRAY_LITERAL_OK.with(|flag| flag.replace(false)) {
                return Err(diagnostics::error(
                    Code::UnsupportedExpression,
                    array,
                    "an array literal is the initializer of a typed `let`, the right side of an assignment, a call argument or a returned value: bind it to a typed local first",
                ));
            }
            format!("{{ {} }}", emit_list(array.elems.iter())?.join(", "))
        }
        syn::Expr::Assign(assign) => format!(
            "{} = {}",
            emit_expression(&assign.left)?,
            emit_value_expression(&assign.right)?
        ),
        syn::Expr::Repeat(repeat) => {
            return Err(diagnostics::error(
                Code::UnsupportedExpression,
                repeat,
                "`[e; N]` is the whole initializer of a typed `let`, the right side of an assignment statement or a returned value, so that `e` is evaluated once before the array is built: bind it first (`let a: [T; N] = [e; N];`)",
            ));
        }
        syn::Expr::Binary(binary)
            if matches!(
                binary.op,
                syn::BinOp::Eq(_)
                    | syn::BinOp::Ne(_)
                    | syn::BinOp::Lt(_)
                    | syn::BinOp::Le(_)
                    | syn::BinOp::Gt(_)
                    | syn::BinOp::Ge(_)
            ) && comparison_array_operand(unparenthesized_expr(&binary.left))
                && comparison_array_operand(unparenthesized_expr(&binary.right)) =>
        {
            return Err(diagnostics::error(
                Code::UnsupportedExpression,
                binary,
                "arrays are not compared with `==`, `!=`, `<`, `<=`, `>` or `>=`: compare the elements",
            ));
        }
        // `i as usize` in an index is lowered like `i as u32` (fix round 1 of m9.r1): the index gets the 32-bit
        // unsigned value of the source, so a `bool` source and a signed `%` index as in Rust, and the remaining
        // difference is the 32/64-bit width class of D39. The validator took the cast only in an index (GUST-E0334).
        syn::Expr::Cast(cast) if is_usize_type(&cast.ty) => {
            format!("uint({})", emit_expression(&cast.expr)?)
        }
        syn::Expr::Binary(binary) if matches!(binary.op, syn::BinOp::BitXor(_)) => format!(
            "__gust_xor({})",
            emit_list([binary.left.as_ref(), binary.right.as_ref()].into_iter())?.join(", ")
        ),
        syn::Expr::Binary(binary) if matches!(binary.op, syn::BinOp::BitXorAssign(_)) => format!(
            "__gust_xor_assign({}, {})",
            emit_expression(&binary.left)?,
            emit_expression(&binary.right)?
        ),
        // `/`, `%`, `/=` and `%=` with a divisor that is not a literal go through `__gust_div` and `__gust_rem`.
        syn::Expr::Binary(binary) if division_helper(binary).is_some() => {
            let (name, assign) = division_helper(binary).expect("guarded by the match arm");
            format!(
                "__gust_{name}{}({}, {})",
                if assign { "_assign" } else { "" },
                emit_expression(&binary.left)?,
                emit_expression(&binary.right)?
            )
        }
        // In a module that names a matrix, `*` and `*=` go through the overloads of `__gust_mul`.
        // A product in a future `const` or array length must not take this arm: Slang cannot use a `__gust_mul(..)`
        // call as a size, so the slice that adds constants lowers a product there with the plain `*`.
        syn::Expr::Binary(binary) if lowering().mul && matches!(binary.op, syn::BinOp::Mul(_)) => {
            format!(
                "__gust_mul({}, {})",
                emit_expression(&binary.left)?,
                emit_expression(&binary.right)?
            )
        }
        syn::Expr::Binary(binary)
            if lowering().mul && matches!(binary.op, syn::BinOp::MulAssign(_)) =>
        {
            if !is_pure_place(&binary.left) {
                return Err(diagnostics::error(
                    Code::MulAssignPlace,
                    binary,
                    "`*=` on a place that is more than names, fields, literals and index arithmetic (a call, an `if` \
                     or a block in the place): Rust evaluates the place once and the lowering writes it twice; write \
                     `x = x * y` instead",
                ));
            }
            let place = emit_expression(&binary.left)?;
            format!(
                "({place} = __gust_mul({place}, {}))",
                emit_expression(&binary.right)?
            )
        }
        syn::Expr::Binary(binary) => format!(
            "({} {} {})",
            emit_expression(&binary.left)?,
            binary.op.to_token_stream(),
            emit_expression(&binary.right)?
        ),
        syn::Expr::Call(call) if is_some_constructor(call) => {
            format!("__gust_some({})", emit_expression(&call.args[0])?)
        }
        syn::Expr::Call(call) if is_result_constructor(call) => {
            let helper = if is_ok_constructor(call) {
                "__gust_ok"
            } else {
                "__gust_err"
            };
            format!("{helper}({})", emit_expression(&call.args[0])?)
        }
        syn::Expr::Call(call) if is_atomic_call(call) => {
            let op_name =
                atomic_op_name(call).expect("is_atomic_call guarantees a recognized name");
            let method =
                atomic_slang_method(op_name).expect("recognized atomic op has a Slang method");
            let receiver = call
                .args
                .first()
                .expect("atomic call has at least one argument");
            let syn::Expr::Reference(reference) = receiver else {
                return Err(diagnostics::error(
                    Code::AtomicEmitterReference,
                    receiver,
                    format!("`{op_name}` requires a mutable reference to a buffer element"),
                ));
            };
            let syn::Expr::Index(index) = reference.expr.as_ref() else {
                return Err(diagnostics::error(
                    Code::AtomicEmitterElement,
                    &reference.expr,
                    format!("`{op_name}` requires a buffer element"),
                ));
            };
            let args: Vec<String> = call
                .args
                .iter()
                .skip(1)
                .map(emit_expression)
                .collect::<syn::Result<_>>()?;
            format!(
                "{}[{}].{}({})",
                emit_expression(&index.expr)?,
                emit_expression(&index.index)?,
                method,
                args.join(", ")
            )
        }
        syn::Expr::Call(call) if math_constructor(call).is_some() => emit_math_constructor(call)?,
        // The validator admits the barrier as a statement with no arguments only.
        syn::Expr::Call(call) if crate::validate::is_barrier_call(call) => {
            "GroupMemoryBarrierWithGroupSync()".to_owned()
        }
        // The validator admits a subgroup operation in the body of a `requires(subgroups)` kernel only (GUST-E0745).
        syn::Expr::Call(call) if crate::validate::subgroup_operation(call).is_some() => {
            emit_subgroup_call(call)?
        }
        // An associated function, `S::f(..)` or `Self::f(..)` (slice m9.a2): Slang calls a `static` member as `S.f(..)`.
        // Both names are renamed like their declarations. The validator admits only paths of two plain segments here.
        syn::Expr::Call(call) if associated_call_path(call).is_some() => {
            let (owner, member) = associated_call_path(call).expect("guarded by the match arm");
            let owner = if owner == "Self" {
                collisions::emit_name(&self_struct().ok_or_else(|| {
                    diagnostics::error(
                        Code::UnknownCall,
                        call,
                        "`Self` outside an `impl` block has no lowering",
                    )
                })?)
                .into_owned()
            } else {
                collisions::emit_name(&owner).into_owned()
            };
            format!(
                "{owner}.{}({})",
                collisions::emit_name(&member),
                emit_arguments(call.args.iter())?.join(", ")
            )
        }
        // A closure of the function being emitted (slice m9.c): the hoisted function, arguments left to right as for a helper.
        syn::Expr::Call(call) if closure_call_target(call).is_some() => format!(
            "{}({})",
            closure_call_target(call).expect("guarded by the match arm"),
            emit_arguments(call.args.iter())?.join(", ")
        ),
        syn::Expr::Call(call) => format!(
            "{}({})",
            emit_expression(&call.func)?,
            emit_arguments(call.args.iter())?.join(", ")
        ),
        syn::Expr::Cast(cast) => {
            format!("{}({})", emit_type(&cast.ty)?, emit_expression(&cast.expr)?)
        }
        syn::Expr::Field(field) => format!(
            "{}.{}",
            emit_expression(&field.base)?,
            match &field.member {
                syn::Member::Named(ident) => emit_ident(ident),
                // A tuple element: Slang's `Tuple` names its elements `_0`, `_1`, ...
                syn::Member::Unnamed(index) => format!("_{}", index.index),
            }
        ),
        syn::Expr::Tuple(tuple) if !tuple.elems.is_empty() => {
            format!("makeTuple({})", emit_list(tuple.elems.iter())?.join(", "))
        }
        syn::Expr::Group(group) => emit_expression(&group.expr)?,
        syn::Expr::Index(index) => {
            // The SPIR-V route's guard (D172): naga's `Restrict` policy for arrays, as explicit code. A non-literal
            // index of an array whose length the tables name is clamped to `min(uint(i), N - 1)`. A storage buffer
            // has no length here and is not clamped (D174b).
            let position = emit_expression(&index.index)?;
            let position = match array_place_length_of(&index.expr) {
                Some(length)
                    if guards::enabled() && length > 0 && !is_literal_operand(&index.index) =>
                {
                    format!("min(uint({position}), {}u)", length - 1)
                }
                _ => position,
            };
            format!("{}[{}]", emit_expression(&index.expr)?, position)
        }
        syn::Expr::Lit(literal) => emit_literal(&literal.lit)?,
        syn::Expr::MethodCall(call) if is_math_method(call) => emit_math_method(call)?,
        // The wrapping methods are the plain operators, even in a module that names a matrix (where `*` is
        // `__gust_mul`): WGSL integer arithmetic wraps modulo 2^32 (section 6.2.3). The macro sees no types, so a
        // negation is `0 - a`, which slangc types as the operand's own type and lowers without a unary minus
        // (WGSL has no negation of a `u32`).
        syn::Expr::MethodCall(call)
            if crate::validate::wrapping_arity(&crate::names::name_of(&call.method))
                .is_some_and(|arity| arity == call.args.len()) =>
        {
            let receiver = emit_expression(&call.receiver)?;
            match crate::names::name_of(&call.method).as_str() {
                "wrapping_neg" => format!("(0 - {receiver})"),
                name => {
                    let operator = match name {
                        "wrapping_add" => '+',
                        "wrapping_sub" => '-',
                        "wrapping_mul" => '*',
                        other => unreachable!(
                            "`{other}` is accepted by `wrapping_arity` but has no emitter arm"
                        ),
                    };
                    format!(
                        "({receiver} {operator} {})",
                        emit_expression(&call.args[0])?
                    )
                }
            }
        }
        syn::Expr::MethodCall(call) if crate::names::ident_is(&call.method, "global_id") => {
            emit_expression(&call.receiver)?
        }
        syn::Expr::MethodCall(call) if crate::names::ident_is(&call.method, "len") => {
            // The length of an array is its `N` (slice m9.r1); the length of a storage parameter is a constant of the
            // dispatch. An array field (`s.items.len()`) and an array element of a buffer (`out[i].len()`) are judged by
            // the tables of the module and of the kernel (slice m9.r2a).
            if let Some(length) = array_place_length_of(&call.receiver) {
                return Ok(length.to_string());
            }
            let receiver = emit_expression(&call.receiver)?;
            format!("__gpu_len_{receiver}")
        }
        // `load`, `store`, `sample`, `width` and `height` of a texture parameter (slice mx.t3); the validator has
        // refused them on anything else.
        syn::Expr::MethodCall(call) if texture::is_texture_method(call) => {
            texture::emit_method(call)?
        }
        syn::Expr::MethodCall(call) if crate::names::ident_is(&call.method, "is_some") => {
            format!("{}.hasValue", emit_expression(&call.receiver)?)
        }
        syn::Expr::MethodCall(call) if crate::names::ident_is(&call.method, "is_none") => {
            format!("(!{}.hasValue)", emit_expression(&call.receiver)?)
        }
        syn::Expr::MethodCall(call) if crate::names::ident_is(&call.method, "is_ok") => {
            format!("{}.isOk", emit_expression(&call.receiver)?)
        }
        syn::Expr::MethodCall(call) if crate::names::ident_is(&call.method, "is_err") => {
            format!("(!{}.isOk)", emit_expression(&call.receiver)?)
        }
        syn::Expr::MethodCall(call)
            if crate::names::ident_is(&call.method, "unwrap_or") && call.args.len() == 1 =>
        {
            format!(
                "__gust_unwrap_or({}, {})",
                emit_expression(&call.receiver)?,
                emit_expression(&call.args[0])?
            )
        }
        // Every other method call is a method of the module: the validator refuses a name that is neither a dialect method
        // nor one (GUST-E0305, E0325), and a method cannot take a dialect name (GUST-E0328). Slang resolves the receiver's
        // type, so the call is the member call as written (D156).
        syn::Expr::MethodCall(call) => format!(
            "{}.{}({})",
            emit_expression(&call.receiver)?,
            emit_ident(&call.method),
            emit_arguments(call.args.iter())?.join(", ")
        ),
        syn::Expr::Paren(paren) => format!("({})", emit_expression(&paren.expr)?),
        syn::Expr::Path(path) if crate::names::path_is(&path.path, "None") => "none".to_owned(),
        // The receiver of a method is `this` in Slang (slice m9.a1).
        syn::Expr::Path(path)
            if path.qself.is_none() && crate::names::path_is(&path.path, "self") =>
        {
            "this".to_owned()
        }
        syn::Expr::Path(path)
            if path.qself.is_none()
                && matches!(
                    math_types::associated_item(&path.path),
                    Some((_, "IDENTITY"))
                ) =>
        {
            let (math, _) =
                math_types::associated_item(&path.path).expect("guarded by the match arm");
            identity_matrix(math)
        }
        // A single-segment path is a value name (a local, a parameter or a helper), so it is renamed like its
        // declaration. A longer path names a type or module first and is emitted as written.
        syn::Expr::Path(path) if path.path.get_ident().is_some() && path.qself.is_none() => {
            emit_ident(path.path.get_ident().expect("guarded by the match arm"))
        }
        // `E::V`, a variant of an enum of the module (the validator took no other two-segment path value).
        syn::Expr::Path(path)
            if path.qself.is_none()
                && path.path.leading_colon.is_none()
                && path.path.segments.len() == 2 =>
        {
            emit_variant_path(&path.path)
        }
        syn::Expr::Path(path) => path.path.to_token_stream().to_string().replace(" :: ", "."),
        syn::Expr::Return(expression) => match &expression.expr {
            Some(value) => format!("return {}", emit_value_expression(value)?),
            None => "return".to_owned(),
        },
        syn::Expr::Struct(value) => {
            return Err(diagnostics::error(
                Code::StructLiteralPosition,
                value,
                "GPU struct literals are supported only as a `let` initializer or the right side of an assignment; bind the struct to a local first, or copy a buffer element and update its fields",
            ));
        }
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Not(_)) => {
            format!("__gust_not({})", emit_expression(&unary.expr)?)
        }
        syn::Expr::Unary(unary) if negated_integer_literal(unary).is_some() => {
            let value = negated_integer_literal(unary).expect("guarded by the match arm");
            format!("(-{})", integer_literal(value, true)?)
        }
        // A literal operand is a scalar whatever the module names, so it keeps the plain `-` (the integer literal above
        // has its own arm); every other operand of a unary `-` is `__gust_neg(a)` in a module that names a matrix.
        syn::Expr::Unary(unary)
            if lowering().neg
                && matches!(unary.op, syn::UnOp::Neg(_))
                && !matches!(unary.expr.as_ref(), syn::Expr::Lit(_)) =>
        {
            format!("__gust_neg({})", emit_expression(&unary.expr)?)
        }
        syn::Expr::Unary(unary) => format!(
            "({}{})",
            unary.op.to_token_stream(),
            emit_expression(&unary.expr)?
        ),
        other => {
            return Err(diagnostics::error(
                Code::UnsupportedExpression,
                other,
                "this Rust expression is not in the Rust-to-Slang subset yet",
            ));
        }
    })
}

/// The two names of a call `Owner::member(..)` of two plain segments (`S::f`, `Self::f`), unraw. A math constructor is the
/// same shape but has its own arm before this one is asked.
fn associated_call_path(call: &syn::ExprCall) -> Option<(String, String)> {
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return None;
    };
    if path.qself.is_some()
        || path.path.leading_colon.is_some()
        || path.path.segments.len() != 2
        || path
            .path
            .segments
            .iter()
            .any(|segment| !segment.arguments.is_none())
    {
        return None;
    }
    Some((
        path.path.segments[0].ident.unraw().to_string(),
        path.path.segments[1].ident.unraw().to_string(),
    ))
}

/// The math type and member of a call to a constructor of `gust::math` (`Vec4::new`, `Mat4::from_cols`, ..).
fn math_constructor(call: &syn::ExprCall) -> Option<(&'static math_types::MathType, &'static str)> {
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    math_types::associated_item(&path.path).filter(|(_, member)| *member != "IDENTITY")
}

/// `Vec4::new(a, b, c, d)` is `float4(a, b, c, d)` and `Vec4::splat(s)` is `float4(s)`. Slang's matrix constructor
/// takes rows, so `Mat4::from_rows(r0, r1, r2, r3)` is `float4x4(r0, r1, r2, r3)` and
/// `Mat4::from_cols(c0, c1, c2, c3)`, with the matrices column-major, is the transpose of the same constructor
/// (measured: plan S6.1 (a)).
fn emit_math_constructor(call: &syn::ExprCall) -> syn::Result<String> {
    let (math, member) = math_constructor(call).expect("the caller checked the call");
    let arguments = emit_list(call.args.iter())?.join(", ");
    Ok(match member {
        "from_cols" => format!("transpose({}({arguments}))", math.slang),
        _ => format!("{}({arguments})", math.slang),
    })
}

/// `Mat4::IDENTITY` as a constructor call: Slang has no such constant.
fn identity_matrix(math: &math_types::MathType) -> String {
    let width = math_types::width(math);
    let row_type = format!("float{width}");
    let rows: Vec<String> = (0..width)
        .map(|row| {
            let components: Vec<&str> = (0..width)
                .map(|column| if column == row { "1.0" } else { "0.0" })
                .collect();
            format!("{row_type}({})", components.join(", "))
        })
        .collect();
    format!("{}({})", math.slang, rows.join(", "))
}

/// Whether a method call is one of the vector and matrix methods: a swizzle, `dot`, `mul_vecN`, `mul_matN` or
/// `transpose`.
fn is_math_method(call: &syn::ExprMethodCall) -> bool {
    let name = crate::names::name_of(&call.method);
    math_types::is_swizzle(&name)
        || math_types::method_arity(&name).is_some_and(|arity| arity == call.args.len())
}

/// `v.xy()` is `v.xy`, `a.dot(b)` is `dot(a, b)`, `m.mul_vec4(v)` and `a.mul_mat4(b)` are `mul(m, v)` and `mul(a, b)`,
/// and `m.transpose()` is `transpose(m)`.
fn emit_math_method(call: &syn::ExprMethodCall) -> syn::Result<String> {
    let name = crate::names::name_of(&call.method);
    let receiver = emit_expression(&call.receiver)?;
    Ok(if math_types::is_swizzle(&name) {
        format!("{receiver}.{name}")
    } else if name == "transpose" {
        format!("transpose({receiver})")
    } else {
        let function = match name.as_str() {
            "dot" if lowering().dot => "__gust_dot",
            "dot" => "dot",
            _ => "mul",
        };
        let arguments = emit_list(call.args.iter())?.join(", ");
        format!("{function}({receiver}, {arguments})")
    })
}

/// The literal operand of a unary minus, if it is an integer literal.
fn negated_integer_literal(unary: &syn::ExprUnary) -> Option<&syn::LitInt> {
    match (&unary.op, unary.expr.as_ref()) {
        (syn::UnOp::Neg(_), syn::Expr::Lit(literal)) => match &literal.lit {
            syn::Lit::Int(value) => Some(value),
            _ => None,
        },
        _ => None,
    }
}

/// The name of the identity helper that keeps a `<` comparison out of a WGSL template list.
///
/// Still needed, measured in slice m4.z (2026-11-02, `plans/m4/probe/m4z/keep`): 30 comparison-in-list shapes, each
/// compiled with and without the wrapper and the WGSL parsed and validated by naga 30.0.1. slangc 2026.20 puts
/// parentheses around every comparison argument itself (all 60 files validate, and the wrapper only adds the identity
/// function and its call), but slangc 2026.13.1 does not (13 of the 30 fail to parse without it). `gust::slang` runs
/// whatever `slangc` is first on `PATH` and checks no version; only `cargo gust` and `cargo xtask` enforce the 2026.22
/// pin. So the wrapper stays until the library enforces the pin. Removing it changes the WGSL of every kernel that
/// uses it, which also wants a read-back on the GPU.
const KEEP_HELPER: &str = "__gust_keep";

/// Whether an expression is a `<` comparison, through parentheses and the right operand of `&&` and `||`.
///
/// WGSL's template-list discovery (which naga's lexer follows) starts a candidate at a `<` that follows an
/// identifier and ends it at any later `>` on the same nesting depth, whatever follows the `>`: `>=` and `>>` close
/// one too. A `)` or `]` drops the candidates opened inside its brackets; `=`, `:`, `;` and `{` drop all of them;
/// `&&` and `||` drop only the candidates opened before them, so a `<` after a `&&` stays pending. `<=` and `<<` do
/// not open one. A comparison that is an argument sits at the depth of the list, and slangc keeps no parentheses
/// around it. slangc does parenthesise shifts and turns `&&` and `||` into branches today, so `g(a < b, c >> 1)` and
/// `g(c && a < b, d > e)` come out safe, but by slangc's choice and not by the rule, so the right operand of `&&`
/// and `||` counts as well. Only `<` opens a candidate, so only `<` needs the helper.
fn is_less_than(expression: &syn::Expr) -> bool {
    match expression {
        syn::Expr::Paren(paren) => is_less_than(&paren.expr),
        syn::Expr::Group(group) => is_less_than(&group.expr),
        syn::Expr::Binary(binary) => match binary.op {
            syn::BinOp::Lt(_) => true,
            syn::BinOp::And(_) | syn::BinOp::Or(_) => is_less_than(&binary.right),
            _ => false,
        },
        _ => false,
    }
}

/// Whether the element at `index` of a comma-separated list of `len` elements goes through `__gust_keep`.
///
/// A candidate opened in the last element ends at the `)` that closes the list, so only an element followed by
/// another one can meet a later `>`. The rule is the same for call arguments, array elements and the two operands
/// of `^`, which is a call to `__gust_xor`.
fn needs_keep(expression: &syn::Expr, index: usize, len: usize) -> bool {
    index + 1 < len && is_less_than(expression)
}

thread_local! {
    /// Whether the kernel being emitted wrapped an element in `__gust_keep` (set by [`emit_list`], the one place that
    /// writes the call). `emit_kernel` defines the helper exactly when this is set, so no scan of the source has to
    /// agree with the emitter about which lists wrap.
    static KEEP_USED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Emit the elements of a comma-separated list, wrapping those that need `__gust_keep`.
fn emit_list<'a>(
    elements: impl ExactSizeIterator<Item = &'a syn::Expr>,
) -> syn::Result<Vec<String>> {
    emit_list_of(elements, false)
}

/// [`emit_list`] for the arguments of a call, where an array literal has a lowering (`f({ a, b })`, slice m9.r1).
fn emit_arguments<'a>(
    elements: impl ExactSizeIterator<Item = &'a syn::Expr>,
) -> syn::Result<Vec<String>> {
    emit_list_of(elements, true)
}

fn emit_list_of<'a>(
    elements: impl ExactSizeIterator<Item = &'a syn::Expr>,
    arrays: bool,
) -> syn::Result<Vec<String>> {
    let len = elements.len();
    elements
        .enumerate()
        .map(|(index, element)| {
            let text = if arrays {
                emit_value_expression(element)?
            } else {
                emit_expression(element)?
            };
            Ok(if needs_keep(element, index, len) {
                KEEP_USED.with(|used| used.set(true));
                format!("{KEEP_HELPER}({text})")
            } else {
                text
            })
        })
        .collect()
}

struct NotUsage(bool);

impl<'ast> Visit<'ast> for NotUsage {
    fn visit_expr_unary(&mut self, expression: &'ast syn::ExprUnary) {
        self.0 |= matches!(expression.op, syn::UnOp::Not(_));
        syn::visit::visit_expr_unary(self, expression);
    }
}

/// The Slang of a subgroup operation (M8, D140). The macro sees no types, so an operation that Slang's own intrinsics
/// overload by operand type is the intrinsic (`WaveActiveSum(x)` is `uint`, `int` or `float` by `x`), and an operation
/// that needs a different form per type, or an operand twice, is a call of a helper that
/// [`subgroup_helper_text`] defines in overloads (the `__gust_xor` pattern): the bitwise reductions (an `i32` goes
/// through `asuint` and `asint`, which the DX12 compiler needs, S3), the inclusive scans and the count.
///
/// `subgroup_broadcast(x, lane)` is `WaveBroadcastLaneAt` for an integer literal `lane`, which slangc requires to be
/// a compile-time constant (E40012), and `WaveReadLaneAt`, WGSL's `subgroupShuffle`, for any other expression (S1).
/// `subgroup_elect()` is the first active lane comparing its own index, since naga 30.0.1 has no `subgroupElect`.
fn emit_subgroup_call(call: &syn::ExprCall) -> syn::Result<String> {
    let name =
        crate::validate::subgroup_operation(call).expect("the caller matched a subgroup operation");
    let arguments = emit_list(call.args.iter())?;
    let list = arguments.join(", ");
    let lane_is_literal = call.args.get(1).is_some_and(is_integer_literal);
    Ok(match name {
        "subgroup_add" => format!("WaveActiveSum({list})"),
        "subgroup_mul" => format!("WaveActiveProduct({list})"),
        "subgroup_min" => format!("WaveActiveMin({list})"),
        "subgroup_max" => format!("WaveActiveMax({list})"),
        "subgroup_and" => format!("{SUBGROUP_AND}({list})"),
        "subgroup_or" => format!("{SUBGROUP_OR}({list})"),
        "subgroup_xor" => format!("{SUBGROUP_XOR}({list})"),
        "subgroup_all" => format!("WaveActiveAllTrue({list})"),
        "subgroup_any" => format!("WaveActiveAnyTrue({list})"),
        "subgroup_ballot" => format!("WaveActiveBallot({list})"),
        "subgroup_broadcast_first" => format!("WaveReadLaneFirst({list})"),
        "subgroup_broadcast" if lane_is_literal => format!("WaveBroadcastLaneAt({list})"),
        "subgroup_broadcast" => format!("WaveReadLaneAt({list})"),
        "subgroup_shuffle" => format!("WaveShuffle({list})"),
        "subgroup_exclusive_add" => format!("WavePrefixSum({list})"),
        "subgroup_exclusive_mul" => format!("WavePrefixProduct({list})"),
        "subgroup_inclusive_add" => format!("{SUBGROUP_INCLUSIVE_ADD}({list})"),
        "subgroup_inclusive_mul" => format!("{SUBGROUP_INCLUSIVE_MUL}({list})"),
        "subgroup_count" => format!("{SUBGROUP_COUNT}({list})"),
        "subgroup_elect" => {
            "(WaveGetLaneIndex() == WaveReadLaneFirst(WaveGetLaneIndex()))".to_owned()
        }
        "subgroup_size" => "WaveGetLaneCount()".to_owned(),
        "subgroup_lane" => "WaveGetLaneIndex()".to_owned(),
        other => unreachable!("`{other}` is a subgroup operation but has no emitter arm"),
    })
}

const SUBGROUP_AND: &str = "__gust_subgroup_and";
const SUBGROUP_OR: &str = "__gust_subgroup_or";
const SUBGROUP_XOR: &str = "__gust_subgroup_xor";
const SUBGROUP_INCLUSIVE_ADD: &str = "__gust_subgroup_inclusive_add";
const SUBGROUP_INCLUSIVE_MUL: &str = "__gust_subgroup_inclusive_mul";
const SUBGROUP_COUNT: &str = "__gust_subgroup_count";

/// Whether `expression` is an integer literal, through parentheses.
fn is_integer_literal(expression: &syn::Expr) -> bool {
    match expression {
        syn::Expr::Lit(literal) => matches!(literal.lit, syn::Lit::Int(_)),
        syn::Expr::Paren(paren) => is_integer_literal(&paren.expr),
        syn::Expr::Group(group) => is_integer_literal(&group.expr),
        _ => false,
    }
}

/// The subgroup operations a function calls, for the helpers [`subgroup_helper_text`] defines.
#[derive(Default)]
struct SubgroupUsage(std::collections::BTreeSet<&'static str>);

impl<'ast> Visit<'ast> for SubgroupUsage {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let Some(name) = crate::validate::subgroup_operation(call) {
            self.0.insert(name);
        }
        syn::visit::visit_expr_call(self, call);
    }
}

/// The overloaded Slang helpers the subgroup operations of a kernel call, in a fixed order, and only those it calls.
fn subgroup_helper_text(used: &std::collections::BTreeSet<&'static str>) -> String {
    let mut text = String::new();
    for (operation, helper, intrinsic) in [
        ("subgroup_and", SUBGROUP_AND, "WaveActiveBitAnd"),
        ("subgroup_or", SUBGROUP_OR, "WaveActiveBitOr"),
        ("subgroup_xor", SUBGROUP_XOR, "WaveActiveBitXor"),
    ] {
        if used.contains(operation) {
            // The signed overload reduces the bits of the unsigned value: DX12's compiler has no signed bit
            // reduction (S3), and the result is the same bits.
            writeln!(
                text,
                "uint {helper}(uint value) {{ return {intrinsic}(value); }}"
            )
            .unwrap();
            writeln!(
                text,
                "int {helper}(int value) {{ return asint({intrinsic}(asuint(value))); }}"
            )
            .unwrap();
        }
    }
    for (operation, helper, scan, combine) in [
        (
            "subgroup_inclusive_add",
            SUBGROUP_INCLUSIVE_ADD,
            "WavePrefixSum",
            '+',
        ),
        (
            "subgroup_inclusive_mul",
            SUBGROUP_INCLUSIVE_MUL,
            "WavePrefixProduct",
            '*',
        ),
    ] {
        if used.contains(operation) {
            // The exclusive scan combined with the operand, which the helper names once.
            for ty in ["uint", "int", "float"] {
                writeln!(
                    text,
                    "{ty} {helper}({ty} value) {{ return {scan}(value) {combine} value; }}"
                )
                .unwrap();
            }
        }
    }
    if used.contains("subgroup_count") {
        writeln!(
            text,
            "uint {SUBGROUP_COUNT}(bool value) {{ uint4 ballot = WaveActiveBallot(value); return countbits(ballot.x) + countbits(ballot.y) + countbits(ballot.z) + countbits(ballot.w); }}"
        )
        .unwrap();
    }
    if !text.is_empty() {
        text.push('\n');
    }
    text
}

/// Whether an expression is a literal operand, possibly negated or parenthesised (`2`, `-1`, `(2.0)`).
fn is_literal_operand(expression: &syn::Expr) -> bool {
    match expression {
        syn::Expr::Lit(_) => true,
        syn::Expr::Paren(paren) => is_literal_operand(&paren.expr),
        syn::Expr::Group(group) => is_literal_operand(&group.expr),
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            is_literal_operand(&unary.expr)
        }
        _ => false,
    }
}

/// The helper a `/`, `%`, `/=` or `%=` goes through: its name (`div` or `rem`) and whether it is the `_assign` form.
/// `None` for a literal divisor (the validator rejects a literal zero, so the plain operator is safe) and for every
/// other operator.
fn division_helper(binary: &syn::ExprBinary) -> Option<(&'static str, bool)> {
    let helper = match binary.op {
        syn::BinOp::Div(_) => ("div", false),
        syn::BinOp::Rem(_) => ("rem", false),
        syn::BinOp::DivAssign(_) => ("div", true),
        syn::BinOp::RemAssign(_) => ("rem", true),
        _ => return None,
    };
    // With the guards of the SPIR-V route on (D172), `%` and `%=` take the helper for a literal divisor too: slangc's own
    // `%` becomes `OpSRem`, which is poison for negative operands on Vulkan (naga writes `a - d * (a / d)`).
    (!is_literal_operand(&binary.right) || (guards::enabled() && helper.0 == "rem"))
        .then_some(helper)
}

/// Which division helpers the module needs: `div` (for `/` and `/=`) and `rem` (for `%` and `%=`).
#[derive(Default)]
struct DivisionUsage {
    div: bool,
    rem: bool,
}

impl<'ast> Visit<'ast> for DivisionUsage {
    fn visit_expr_binary(&mut self, expression: &'ast syn::ExprBinary) {
        match division_helper(expression) {
            Some(("div", _)) => self.div = true,
            Some(_) => self.rem = true,
            None => {}
        }
        syn::visit::visit_expr_binary(self, expression);
    }
}

/// Whether the module uses `^` or `^=`, so the `__gust_xor` helpers are emitted only when needed.
struct XorUsage(bool);

impl<'ast> Visit<'ast> for XorUsage {
    fn visit_expr_binary(&mut self, expression: &'ast syn::ExprBinary) {
        self.0 |= matches!(
            expression.op,
            syn::BinOp::BitXor(_) | syn::BinOp::BitXorAssign(_)
        );
        syn::visit::visit_expr_binary(self, expression);
    }
}

fn is_some_constructor(call: &syn::ExprCall) -> bool {
    matches!(call.func.as_ref(), syn::Expr::Path(path) if crate::names::path_is(&path.path, "Some"))
        && call.args.len() == 1
}

fn is_ok_constructor(call: &syn::ExprCall) -> bool {
    matches!(call.func.as_ref(), syn::Expr::Path(path) if crate::names::path_is(&path.path, "Ok"))
        && call.args.len() == 1
}

fn is_result_constructor(call: &syn::ExprCall) -> bool {
    is_ok_constructor(call)
        || (matches!(call.func.as_ref(), syn::Expr::Path(path) if crate::names::path_is(&path.path, "Err"))
            && call.args.len() == 1)
}

/// Whether a call is a recognized atomic operation with the expected argument count.
fn is_atomic_call(call: &syn::ExprCall) -> bool {
    let Some(name) = atomic_op_name(call) else {
        return false;
    };
    let expected = if name == "atomic_compare_exchange" {
        3
    } else {
        2
    };
    call.args.len() == expected
}

/// The atomic operation name from the call, if recognized.
fn atomic_op_name(call: &syn::ExprCall) -> Option<&'static str> {
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return None;
    };
    let ident = path.path.get_ident()?;
    match ident.unraw().to_string().as_str() {
        "atomic_add" => Some("atomic_add"),
        "atomic_min" => Some("atomic_min"),
        "atomic_max" => Some("atomic_max"),
        "atomic_exchange" => Some("atomic_exchange"),
        "atomic_compare_exchange" => Some("atomic_compare_exchange"),
        _ => None,
    }
}

/// The Slang method name for an atomic operation.
fn atomic_slang_method(op_name: &str) -> Option<&'static str> {
    match op_name {
        "atomic_add" => Some("add"),
        "atomic_min" => Some("min"),
        "atomic_max" => Some("max"),
        "atomic_exchange" => Some("exchange"),
        "atomic_compare_exchange" => Some("compareExchange"),
        _ => None,
    }
}

/// Buffers referenced by atomic calls need `Atomic<T>` element types.
struct AtomicUsage(std::collections::HashSet<String>);

impl<'ast> Visit<'ast> for AtomicUsage {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if is_atomic_call(call)
            && let syn::Expr::Reference(reference) = &call.args[0]
            && let syn::Expr::Index(index) = reference.expr.as_ref()
            && let syn::Expr::Path(path) = index.expr.as_ref()
            && let Some(segment) = path.path.segments.first()
        {
            self.0.insert(emit_ident(&segment.ident));
        }
        syn::visit::visit_expr_call(self, call);
    }
}

/// Any Option feature in the module; both generic helpers are then emitted.
struct OptionUsage(bool);

impl<'ast> Visit<'ast> for OptionUsage {
    fn visit_type_path(&mut self, ty: &'ast syn::TypePath) {
        self.0 |= ty
            .path
            .segments
            .last()
            .is_some_and(|segment| crate::names::ident_is(&segment.ident, "Option"));
        syn::visit::visit_type_path(self, ty);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        self.0 |= is_some_constructor(call);
        syn::visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.0 |= matches!(
            crate::names::name_of(&call.method).as_str(),
            "is_some" | "is_none"
        );
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_let(&mut self, expression: &'ast syn::ExprLet) {
        self.0 |= crate::validate::option_pattern(&expression.pat).is_some();
        syn::visit::visit_expr_let(self, expression);
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.0 |= crate::names::path_is(&path.path, "None");
        syn::visit::visit_expr_path(self, path);
    }
}

struct ResultUsage(bool);

impl<'ast> Visit<'ast> for ResultUsage {
    fn visit_type_path(&mut self, ty: &'ast syn::TypePath) {
        self.0 |= ty
            .path
            .segments
            .last()
            .is_some_and(|segment| crate::names::ident_is(&segment.ident, "Result"));
        syn::visit::visit_type_path(self, ty);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        self.0 |= is_result_constructor(call);
        syn::visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.0 |= matches!(
            crate::names::name_of(&call.method).as_str(),
            "is_ok" | "is_err"
        );
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_let(&mut self, expression: &'ast syn::ExprLet) {
        self.0 |= crate::validate::result_pattern(&expression.pat).is_some();
        syn::visit::visit_expr_let(self, expression);
    }
}

/// The Slang spelling of an integer literal; `negated` says a unary minus applies to it directly.
///
/// The dialect has no 64-bit integers, but slangc types a bare decimal above `i32::MAX` as `int64_t`, which WGSL
/// and SPIR-V validation refuse once it meets a 32-bit operand. In Rust such an unsuffixed literal can only be
/// `u32` (an `i32` one is a compile error), so it is spelled `uint(..)`, exactly as `u32`-suffixed ones are. Bare
/// digits stay for everything else: values up to `i32::MAX`, a negated literal (`-2147483648` is `i32::MIN` and
/// slangc types it right), and a value above `u32::MAX`, which no 32-bit type holds and which stays loud (slangc
/// keeps it 64-bit and shader creation refuses it) instead of being truncated by a `uint(..)`. The M6 vector
/// operands call this too.
fn integer_literal(value: &syn::LitInt, negated: bool) -> syn::Result<String> {
    let digits = value.base10_digits();
    match value.suffix() {
        "" => {
            let above_i32 = digits
                .parse::<u64>()
                .is_ok_and(|v| v > i32::MAX as u64 && v <= u64::from(u32::MAX));
            Ok(if above_i32 && !negated {
                format!("uint({digits})")
            } else {
                digits.to_owned()
            })
        }
        "u32" => Ok(format!("uint({digits})")),
        "i32" => Ok(format!("int({digits})")),
        "f32" => Ok(format!("float({digits}.0)")),
        _ => Err(diagnostics::error(
            Code::UnsupportedLiteral,
            value,
            "GPU literals support booleans and 32-bit numbers (i32, u32, f32) only",
        )),
    }
}

fn emit_literal(literal: &syn::Lit) -> syn::Result<String> {
    let unsupported = || {
        diagnostics::error(
            Code::UnsupportedLiteral,
            literal,
            "GPU literals support booleans and 32-bit numbers (i32, u32, f32) only",
        )
    };
    match literal {
        syn::Lit::Bool(value) => Ok(value.value.to_string()),
        syn::Lit::Int(value) => integer_literal(value, false),
        syn::Lit::Float(value) => match value.suffix() {
            "" => Ok(value.base10_digits().to_owned()),
            "f32" => Ok(format!("float({})", value.base10_digits())),
            _ => Err(unsupported()),
        },
        _ => Err(unsupported()),
    }
}

fn emit_type(ty: &syn::Type) -> syn::Result<String> {
    if let syn::Type::Tuple(tuple) = ty {
        if tuple.elems.is_empty() {
            return Ok("void".to_owned());
        }
        // `(A, B)` is Slang's `Tuple<A, B>` (spike M9.s0, P3).
        let elements = tuple
            .elems
            .iter()
            .map(emit_type)
            .collect::<syn::Result<Vec<_>>>()?;
        return Ok(format!("Tuple<{}>", elements.join(", ")));
    }
    let syn::Type::Path(path) = ty else {
        return Err(diagnostics::error(
            Code::TypeNeedsPath,
            ty,
            "Slang types must be named paths",
        ));
    };
    let segment = path.path.segments.last().expect("nonempty type path");
    let source_name = segment.ident.unraw().to_string();
    let name = match source_name.as_str() {
        "f32" | "float" => "float".to_owned(),
        "i32" | "int" => "int".to_owned(),
        "u32" | "uint" => "uint".to_owned(),
        "bool" => "bool".to_owned(),
        "Option" => "Optional".to_owned(),
        "Result" => "__GustResult".to_owned(),
        "SV_DispatchThreadID" | "LocalId" | "GroupId" | "SV_GroupThreadID" | "SV_GroupID" => {
            "uint3".to_owned()
        }
        // `Self` in the declaration of a trait is the implementing type, Slang's `This` (slice m9.g1).
        "Self" if IN_INTERFACE.with(std::cell::Cell::get) => "This".to_owned(),
        // `Self` in an `impl` block is the struct of the block (slice m9.a2).
        "Self" if self_struct().is_some() => {
            collisions::emit_name(&self_struct().expect("guarded by the match arm")).into_owned()
        }
        name if math_types::math_type(name).is_some() => math_types::math_type(name)
            .expect("guarded by the match arm")
            .slang
            .to_owned(),
        // Any other type name is a struct of the module, so it is renamed exactly like its declaration.
        name => collisions::emit_name(name).into_owned(),
    };
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return Ok(name);
    };
    let parameters = arguments
        .args
        .iter()
        .filter_map(|argument| match argument {
            syn::GenericArgument::Type(ty) => Some(emit_type(ty)),
            _ => None,
        })
        .collect::<syn::Result<Vec<_>>>()?;
    if source_name == "Result" {
        let payload = parameters.first().ok_or_else(|| {
            diagnostics::error(
                Code::ResultPayloadMissing,
                ty,
                "`Result` needs a payload type",
            )
        })?;
        return Ok(format!("{name}<{payload}>"));
    }
    Ok(format!("{name}<{}>", parameters.join(", ")))
}

/// `expression` without the parentheses and invisible groups around it.
fn unparenthesized_expr(expression: &syn::Expr) -> &syn::Expr {
    match expression {
        syn::Expr::Paren(inner) => unparenthesized_expr(&inner.expr),
        syn::Expr::Group(inner) => unparenthesized_expr(&inner.expr),
        other => other,
    }
}

/// Whether `ty` is the path `usize`.
fn is_usize_type(ty: &syn::Type) -> bool {
    matches!(unparenthesized_type(ty), syn::Type::Path(path)
        if path.qself.is_none() && crate::names::path_is(&path.path, "usize"))
}

/// The Slang type of a local, a parameter or a result (slice m9.r1): [`emit_type`], and `T[N]` for `[T; N]`. A struct
/// field, a buffer or uniform element and the payload of an `Option` or `Result` still go through [`emit_type`], which
/// has no array arm, so an array there is refused exactly as before (GUST-E0813).
fn emit_value_type(ty: &syn::Type) -> syn::Result<String> {
    let syn::Type::Array(array) = unparenthesized_type(ty) else {
        return emit_type(ty);
    };
    let length = array_length_literal(&array.len).ok_or_else(|| {
        diagnostics::error(
            Code::ArrayLength,
            &array.len,
            "an array length is an integer literal greater than zero",
        )
    })?;
    // A nested array has no arm in `emit_type` either (the validator refused it, GUST-E0217).
    Ok(format!("{}[{length}]", emit_type(&array.elem)?))
}

/// [`emit_expression`] for a position where an array literal has a lowering: the flag lives only for the one expression
/// when it is a literal, so it cannot let a literal nested somewhere else through.
fn emit_value_expression(expression: &syn::Expr) -> syn::Result<String> {
    ARRAY_LITERAL_OK.with(|flag| {
        flag.set(matches!(
            unparenthesized_expr(expression),
            syn::Expr::Array(_)
        ));
    });
    // A parenthesized literal is emitted without the parentheses: `({ .. })` is not an initializer list to slangc.
    let inner = unparenthesized_expr(expression);
    let text = emit_expression(if matches!(inner, syn::Expr::Array(_)) {
        inner
    } else {
        expression
    });
    ARRAY_LITERAL_OK.with(|flag| flag.set(false));
    text
}

/// What a `[e; N]` is assigned to.
enum RepeatTarget<'a> {
    /// `let name: [T; N] = [e; N];`
    Let {
        array_type: &'a syn::Type,
        name: &'a syn::Ident,
    },
    /// `place = [e; N];`
    Assign { place: String },
    /// A returned value.
    Return,
}

/// The longest repeat written as one initializer list; a longer one is filled by a loop.
const REPEAT_LIST_LIMIT: u32 = 16;

/// `[e; N]` (slice m9.r1): `e` is evaluated once, into `__gust_rep_<k>`, before the array is built, as Rust does
/// (`[s.bump(); 4]` calls `bump` once). Up to [`REPEAT_LIST_LIMIT`] elements the array is `{ t, t, .. }`, above it a
/// declaration and a `for` that fills it.
fn emit_repeat(
    output: &mut String,
    indent: usize,
    index: usize,
    repeat: &syn::ExprRepeat,
    target: &RepeatTarget<'_>,
) -> syn::Result<()> {
    let padding = "    ".repeat(indent);
    let length = array_length_literal(&repeat.len).ok_or_else(|| {
        diagnostics::error(
            Code::ArrayLength,
            &repeat.len,
            "an array length is an integer literal greater than zero",
        )
    })?;
    let temporary = format!("__gust_rep_{index}");
    let counter = format!("__gust_fill_{index}");
    // The element type is written when the `let` names it; `var` otherwise (Slang converts on the assignment).
    let element_type = match target {
        RepeatTarget::Let { array_type, .. } => match unparenthesized_type(array_type) {
            syn::Type::Array(array) => emit_type(&array.elem)?,
            _ => "var".to_owned(),
        },
        _ => "var".to_owned(),
    };
    writeln!(
        output,
        "{padding}{element_type} {temporary} = {};",
        emit_expression(&repeat.expr)?
    )
    .unwrap();
    let list = || {
        format!(
            "{{ {} }}",
            vec![temporary.as_str(); length as usize].join(", ")
        )
    };
    let fill = |output: &mut String, place: &str| {
        writeln!(
            output,
            "{padding}for (var {counter} = 0; {counter} < {length}; {counter}++)"
        )
        .unwrap();
        writeln!(output, "{padding}{{").unwrap();
        writeln!(output, "{padding}    {place}[{counter}] = {temporary};").unwrap();
        writeln!(output, "{padding}}}").unwrap();
    };
    match target {
        RepeatTarget::Let { array_type, name } => {
            let declaration = emit_value_type(array_type)?;
            let name = emit_ident(name);
            if length <= REPEAT_LIST_LIMIT {
                writeln!(output, "{padding}{declaration} {name} = {};", list()).unwrap();
            } else {
                writeln!(output, "{padding}{declaration} {name};").unwrap();
                fill(output, &name);
            }
        }
        RepeatTarget::Assign { place } => {
            if length <= REPEAT_LIST_LIMIT {
                writeln!(output, "{padding}{place} = {};", list()).unwrap();
            } else {
                fill(output, place);
            }
        }
        RepeatTarget::Return => {
            if length > REPEAT_LIST_LIMIT {
                return Err(diagnostics::error(
                    Code::UnsupportedExpression,
                    repeat,
                    format!(
                        "a returned `[e; N]` has at most {REPEAT_LIST_LIMIT} elements: build a longer one in a typed local (`let a: [T; N] = [e; N]; a`)"
                    ),
                ));
            }
            writeln!(output, "{padding}return {};", list()).unwrap();
        }
    }
    Ok(())
}

fn typed_argument(argument: &syn::FnArg) -> syn::Result<(&syn::Ident, &syn::Type)> {
    let syn::FnArg::Typed(argument) = argument else {
        return Err(diagnostics::error(
            Code::SelfParameter,
            argument,
            "GPU functions cannot use self",
        ));
    };
    let syn::Pat::Ident(pattern) = argument.pat.as_ref() else {
        return Err(diagnostics::error(
            Code::ParameterNeedsIdentifier,
            &argument.pat,
            "GPU parameters need identifier names",
        ));
    };
    Ok((&pattern.ident, &argument.ty))
}

pub fn wrapper_element_type(ty: &syn::Type) -> syn::Result<&syn::Type> {
    let syn::Type::Path(path) = ty else {
        return Err(diagnostics::error(
            Code::ResourceElementPath,
            ty,
            "GPU resource requires a named element type",
        ));
    };
    let segment = path.path.segments.last().expect("nonempty resource type");
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return Err(diagnostics::error(
            Code::ResourceElementArity,
            segment,
            "GPU resource requires one element type",
        ));
    };
    arguments
        .args
        .iter()
        .find_map(|argument| match argument {
            syn::GenericArgument::Type(ty) => Some(ty),
            _ => None,
        })
        .ok_or_else(|| {
            diagnostics::error(
                Code::ResourceElementMissing,
                arguments,
                "GPU resource requires an element type",
            )
        })
}

pub(crate) fn type_name(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    path.path
        .segments
        .last()
        .map(|segment| segment.ident.unraw().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_slang_resources_semantic_and_body() {
        let module: syn::ItemMod = syn::parse_quote! {
            mod vector_add {
                #[kernel(workgroup_size(64, 1, 1))]
                fn add(id: SV_DispatchThreadID, a: StructuredBuffer<float>, mut out: RWStructuredBuffer<float>) {
                    let i = id.x;
                    if i < out.len() {
                        out[i] = a[i] + 1.0;
                    }
                }
            }
        };
        let (_, items) = module.content.as_ref().unwrap();
        let kernel = items
            .iter()
            .find_map(|item| match item {
                syn::Item::Fn(function) => Some(function),
                _ => None,
            })
            .unwrap();
        let source = emit_kernel(
            &module,
            kernel,
            KernelOptions {
                workgroup_size: [64, 1, 1],
            },
        )
        .unwrap()
        .source;
        assert!(source.contains("StructuredBuffer<float> a;"));
        assert!(source.contains("RWStructuredBuffer<float> out;"));
        assert!(source.contains("void gpu_vector_add_add(uint3 id : SV_DispatchThreadID)"));
        assert!(source.contains("out[i] = (a[i] + 1.0);"));
        assert!(source.contains("i < __gpu_len_out"));
    }

    /// The emitter re-checks shapes the validator already rejects, so these sites are
    /// unreachable through `#[gpu]`; they are exercised on the emitter functions directly
    /// and must still carry their own codes.
    #[test]
    fn emitter_rejections_carry_their_codes() {
        use crate::diagnostics::testing::assert_code;

        let ty = |ty: syn::Type| emit_type(&ty).unwrap_err();
        assert_code(&ty(syn::parse_quote!([u32; 2])), Code::TypeNeedsPath);
        assert_code(
            &ty(syn::parse_quote!(Result<'a>)),
            Code::ResultPayloadMissing,
        );

        let argument: syn::FnArg = syn::parse_quote!(&self);
        assert_code(&typed_argument(&argument).unwrap_err(), Code::SelfParameter);
        let argument: syn::FnArg = syn::parse_quote!((a, b): u32);
        assert_code(
            &typed_argument(&argument).unwrap_err(),
            Code::ParameterNeedsIdentifier,
        );

        let element = |ty: syn::Type| wrapper_element_type(&ty).unwrap_err();
        assert_code(
            &element(syn::parse_quote!(*const u32)),
            Code::ResourceElementPath,
        );
        assert_code(
            &element(syn::parse_quote!(StructuredBuffer)),
            Code::ResourceElementArity,
        );
        assert_code(
            &element(syn::parse_quote!(StructuredBuffer<'a>)),
            Code::ResourceElementMissing,
        );

        let expression = |expression: syn::Expr| emit_expression(&expression).unwrap_err();
        assert_code(
            &expression(syn::parse_quote!(atomic_add(counter, 1u32))),
            Code::AtomicEmitterReference,
        );
        assert_code(
            &expression(syn::parse_quote!(atomic_add(&mut counter, 1u32))),
            Code::AtomicEmitterElement,
        );
        assert_code(
            &expression(syn::parse_quote!(|x| x)),
            Code::UnsupportedExpression,
        );
        assert_code(
            &expression(syn::parse_quote!('a')),
            Code::UnsupportedLiteral,
        );

        let pattern: syn::Pat = syn::parse_quote!((a, b));
        assert_code(&local_binding(&pattern).unwrap_err(), Code::LetBindingForm);
    }
}
