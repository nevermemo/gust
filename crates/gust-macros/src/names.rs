//! Name comparison for the validator and the emitter (Opus round 3 of m9.z, the class behind M1 and S2).
//!
//! `r#x` and `x` are one name in Rust, and the emitter prints the name without the prefix, so every rule that compares
//! an identifier with a name must compare them without it. `proc_macro2::Ident`'s own `==` and `syn::Path::is_ident`
//! keep the prefix, which is why those two are not used for names of variables, functions and types. They stay in use
//! for attribute paths (`#[kernel]`, `#[repr]`, `#[derive]`, which a raw spelling does not name). `self` and `Self`
//! have no raw spelling and go through `path_is` like any other name.

use syn::ext::IdentExt as _;

/// Whether `ident` is `name`, whether or not it is written with a raw prefix.
pub(crate) fn ident_is(ident: &syn::Ident, name: &str) -> bool {
    ident.unraw() == name
}

/// Whether `path` is the single identifier `name`, raw or not.
pub(crate) fn path_is(path: &syn::Path, name: &str) -> bool {
    path.get_ident().is_some_and(|ident| ident_is(ident, name))
}

/// The name of `ident` without a raw prefix, as a `String` (the name of a method call, matched against a table).
pub(crate) fn name_of(ident: &syn::Ident) -> String {
    ident.unraw().to_string()
}

/// Whether two identifiers are the same name.
pub(crate) fn same_ident(a: &syn::Ident, b: &syn::Ident) -> bool {
    a.unraw() == b.unraw()
}

/// `name` as it is written in a suggestion for the user: a keyword needs its `r#` back (`r#type`). `self`, `Self`,
/// `super`, `crate` and `_` cannot be raw, so they print bare (syn refuses them as an `Ident`, which is not the test
/// for a raw prefix). `gen` is reserved in edition 2024, and syn's list does not have it.
pub(crate) fn spelling(name: &str) -> String {
    if matches!(name, "self" | "Self" | "super" | "crate" | "_") {
        return name.to_owned();
    }
    if name == "gen" {
        return format!("r#{name}");
    }
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

#[cfg(test)]
mod tests {
    use super::spelling;

    #[test]
    fn a_name_that_cannot_be_raw_prints_bare_and_a_keyword_keeps_its_prefix() {
        for bare in ["self", "Self", "super", "crate", "_", "x", "Vec3"] {
            assert_eq!(spelling(bare), bare);
        }
        for raw in ["type", "match", "gen", "fn", "async"] {
            assert_eq!(spelling(raw), format!("r#{raw}"));
        }
    }
}
