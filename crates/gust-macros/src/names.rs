//! Name comparison for the validator and the emitter (Opus round 3 of m9.z, the class behind M1 and S2).
//!
//! `r#x` and `x` are one name in Rust, and the emitter prints the name without the prefix, so every rule that compares
//! an identifier with a name must compare them without it. `proc_macro2::Ident`'s own `==` and `syn::Path::is_ident`
//! keep the prefix, which is why those two are not used for names of variables, functions and types. They stay in use
//! for attribute paths (`#[kernel]`, `#[repr]`, `#[derive]`, which a raw spelling does not name) and for `self` and
//! `Self`, which have no raw spelling.

use syn::ext::IdentExt as _;

/// Whether `ident` is `name`, whether or not it is written with a raw prefix.
pub(crate) fn ident_is(ident: &syn::Ident, name: &str) -> bool {
    ident.unraw() == name
}

/// Whether `path` is the single identifier `name`, raw or not.
pub(crate) fn path_is(path: &syn::Path, name: &str) -> bool {
    path.get_ident().is_some_and(|ident| ident_is(ident, name))
}

/// Whether two identifiers are the same name.
pub(crate) fn same_ident(a: &syn::Ident, b: &syn::Ident) -> bool {
    a.unraw() == b.unraw()
}

/// `name` as it is written in a suggestion for the user: a keyword needs its `r#` back (`r#type`).
pub(crate) fn spelling(name: &str) -> String {
    let token = proc_macro2::TokenTree::Ident(proc_macro2::Ident::new(
        name,
        proc_macro2::Span::call_site(),
    ));
    if crate::diagnostics::parse2::<syn::Ident>(
        crate::diagnostics::Code::UnsupportedExpression,
        token.into(),
    )
    .is_ok()
    {
        name.to_owned()
    } else {
        format!("r#{name}")
    }
}
