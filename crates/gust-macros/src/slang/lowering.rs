//! The lowerings that depend on what the whole module names, and the per-emit state of `Self`.

use quote::ToTokens;
use syn::ext::IdentExt as _;

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

pub(super) fn lowering() -> Lowering {
    LOWERING
        .with(std::cell::Cell::get)
        .expect("the emitter runs inside emit_kernel, which enters a LoweringScope")
}

/// Sets the lowering for a scope and restores the previous one when dropped.
pub(super) struct LoweringScope(Option<Lowering>);

impl LoweringScope {
    pub(super) fn enter(next: Lowering) -> Self {
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
pub(super) fn self_struct() -> Option<String> {
    SELF_STRUCT.with(|cell| cell.borrow().clone())
}

/// Sets [`SELF_STRUCT`] for a scope and restores the previous value when dropped.
pub(super) struct SelfScope(Option<String>);

impl SelfScope {
    pub(super) fn enter(name: &str) -> Self {
        Self(SELF_STRUCT.with(|cell| cell.replace(Some(name.to_owned()))))
    }
}

impl Drop for SelfScope {
    fn drop(&mut self) {
        SELF_STRUCT.with(|cell| *cell.borrow_mut() = self.0.take());
    }
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
pub(super) fn module_lowering(module: &syn::ItemMod) -> Lowering {
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
