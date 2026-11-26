//! `#[derive(GpuEnum)]` (plan m14.s2, slice s2e2a, D157).
//!
//! A derive cannot rewrite the enum, so it checks the shape the enum was written in and refuses what a GPU enum may
//! not be (GUST-E1011, one per offending token, in source order). The shape is the one `#[gpu_module]` accepts for an
//! enum (`validate/items.rs`, GUST-E0122): field-less variants, integer-literal discriminants or none, no generics,
//! at least one variant. The representation differs on purpose: it is `#[repr(u32)]` and it is written (the enum keeps
//! its own tokens, nothing adds it); a signed `#[repr(i32)]` is refused, because a word is a `u32`. `Clone`, `Copy` and
//! `PartialEq` are not checked by the derive: the derive list that names `GpuEnum` is not visible to it (measured,
//! m14.s2e2a), so the supertraits of `GpuEnum` (`Copy + PartialEq`) enforce them as rustc's own error. Nor is `cfg` on a
//! variant: rustc has applied it before the derive runs, so the derive only ever sees the variants that stay.
//!
//! The expansion is `impl ::gust::GpuEnum` and `impl TryFrom<u32>`, each a `match` over the discriminants. There is
//! no `transmute` and no `as` from a word, so a word that is no variant can never become a value of the enum.
//!
//! It also hands the enum to a batch stage (slice s2e2b, D270). A proc macro cannot see a type declared elsewhere, so
//! the derive writes a hidden `macro_rules! __gust_enum_<Name>`, re-exported under the enum's own name
//! (`pub(crate) use __gust_enum_<Name> as <Name>;`: the type keeps the type namespace, the macro takes the macro
//! namespace, and both travel through `use` and paths). `gpu_batch!` calls `Mode! { acc = [..]; next = [..]; cont =
//! {..}; orig = {..} }` for each enum a block names: the macro appends the enum's tokens (its name, `#[repr(u32)]` and
//! the variants with their words) to `acc`, then calls the next enum's macro, or, with none left, the continuation
//! `cont` (the `gpu_batch!` itself, in its hidden `@enums` form) with every definition and the untouched original
//! tokens. Same crate only.

use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote};
use syn::ext::IdentExt as _;

use crate::diagnostics::{self, Code};

/// The derive: the impls, or the errors and nothing else (the enum itself is the user's own tokens; a derive does not
/// return them, so it cannot repeat them beside an error).
pub(crate) fn derive(input: TokenStream) -> TokenStream {
    match expand(input) {
        Ok(tokens) => tokens,
        Err(error) => error.into_compile_error(),
    }
}

fn expand(input: TokenStream) -> syn::Result<TokenStream> {
    let item = diagnostics::parse2::<syn::DeriveInput>(Code::GpuEnumShape, input)?;
    let syn::Data::Enum(data) = &item.data else {
        return Err(diagnostics::error(
            Code::GpuEnumShape,
            &item.ident,
            "`#[derive(GpuEnum)]` goes on an enum: a GPU enum is a plain list of named integer values",
        ));
    };
    let words = check(&item, data)?;
    Ok(emit(&item.ident, &data.variants, &words))
}

/// Checks the shape and returns the discriminant of each variant, in order. Every refusal found is reported, each at
/// its own token; a discriminant that is not a literal has no value, so the later implicit ones are not computed.
fn check(item: &syn::DeriveInput, data: &syn::DataEnum) -> syn::Result<Vec<u32>> {
    let mut errors = diagnostics::Collector::default();
    if !item.generics.params.is_empty() || item.generics.where_clause.is_some() {
        errors.reject(
            Code::GpuEnumShape,
            &item.generics,
            "generic enums are not supported: a GPU enum is a plain list of named integer values",
        );
    }
    check_repr(item, &mut errors);
    if data.variants.is_empty() {
        errors.reject(
            Code::GpuEnumShape,
            &item.ident,
            "a GPU enum needs at least one variant",
        );
    }
    let mut words = Vec::new();
    let mut next = Some(0_u32);
    for variant in &data.variants {
        if !matches!(variant.fields, syn::Fields::Unit) {
            errors.reject(
                Code::GpuEnumShape,
                variant,
                "a GPU enum variant carries no data: data-carrying variants have no Slang lowering yet",
            );
        }
        let word = match &variant.discriminant {
            Some((_, expression)) => match literal_word(expression) {
                Ok(word) => Some(word),
                Err(error) => {
                    errors.push(error);
                    None
                }
            },
            None => next,
        };
        next = word.and_then(|word| word.checked_add(1));
        // An implicit discriminant after the largest word cannot exist: rustc refuses the enum itself (E0370).
        words.push(word.unwrap_or_default());
    }
    errors.finish().map(|()| words)
}

/// `#[repr(u32)]` is written, and is the only representation. At the enum's name when it is missing, at the attribute
/// when it is another one.
fn check_repr(item: &syn::DeriveInput, errors: &mut diagnostics::Collector) {
    let mut any = false;
    for attribute in &item.attrs {
        if !attribute.path().is_ident("repr") {
            continue;
        }
        any = true;
        match diagnostics::parse_args_with(
            attribute,
            Code::GpuEnumShape,
            syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
        ) {
            Ok(reprs) if reprs.len() == 1 && reprs[0].is_ident("u32") => {}
            Ok(_) => errors.reject(
                Code::GpuEnumShape,
                attribute,
                "a GPU enum is `#[repr(u32)]` and nothing else: a word is a `u32` (`#[repr(i32)]`, `#[repr(C)]` and a list of representations are refused)",
            ),
            Err(error) => errors.push(error),
        }
    }
    if !any {
        errors.reject(
            Code::GpuEnumShape,
            &item.ident,
            "a GPU enum is written `#[repr(u32)]`: the derive checks it and does not add it",
        );
    }
}

/// The `u32` of a discriminant that is an integer literal, with no suffix or `u32`. A negative literal, a computed
/// value (`1 + 1`, a `const`), any other literal and a value over `u32::MAX` are refused at the expression.
fn literal_word(expression: &syn::Expr) -> syn::Result<u32> {
    let refuse = |message: &str| diagnostics::error(Code::GpuEnumShape, expression, message);
    let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Int(literal),
        ..
    }) = expression
    else {
        return Err(refuse(
            "a discriminant is a non-negative integer literal (`B = 3`): a computed value would have to be evaluated by both Rust and Slang",
        ));
    };
    if !matches!(literal.suffix(), "" | "u32") {
        return Err(refuse(
            "a discriminant is an unsuffixed integer literal or one suffixed `u32`: the representation is `u32`",
        ));
    }
    literal
        .base10_parse::<u32>()
        .map_err(|_| refuse("a discriminant fits in a `u32`"))
}

/// The two impls, each a `match`.
fn emit(
    name: &syn::Ident,
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::Token![,]>,
    words: &[u32],
) -> TokenStream {
    let idents: Vec<_> = variants.iter().map(|variant| &variant.ident).collect();
    let literals: Vec<Literal> = words
        .iter()
        .map(|word| Literal::u32_unsuffixed(*word))
        .collect();
    let callback = format_ident!("__gust_enum_{}", name.unraw());
    // The enum as a stage's module reads it: the name, `#[repr(u32)]` and the variants with their words. The stage adds
    // the comparison the module validator asks for; `Clone` and `Copy` are the module's own.
    let definition = quote! {
        #[repr(u32)]
        enum #name { #(#idents = #literals),* }
    };
    quote! {
        #[automatically_derived]
        impl ::gust::GpuEnum for #name {
            fn from_word(word: u32) -> ::core::option::Option<Self> {
                match word {
                    #(#words => ::core::option::Option::Some(Self::#idents),)*
                    _ => ::core::option::Option::None,
                }
            }

            fn to_word(self) -> u32 {
                match self {
                    #(Self::#idents => #words,)*
                }
            }
        }

        #[automatically_derived]
        impl ::core::convert::TryFrom<u32> for #name {
            type Error = ::gust::EnumWordError;

            fn try_from(word: u32) -> ::core::result::Result<Self, Self::Error> {
                match word {
                    #(#words => ::core::result::Result::Ok(Self::#idents),)*
                    _ => ::core::result::Result::Err(::gust::EnumWordError::new(word)),
                }
            }
        }

        #[allow(unused_macros)]
        macro_rules! #callback {
            (
                acc = [ $($acc:tt)* ] ;
                next = [ { $($head:tt)+ } $($tail:tt)* ] ;
                cont = { $($cont:tt)+ } ;
                orig = { $($orig:tt)* }
            ) => {
                $($head)+ ! {
                    acc = [ $($acc)* { #definition } ] ;
                    next = [ $($tail)* ] ;
                    cont = { $($cont)+ } ;
                    orig = { $($orig)* }
                }
            };
            (
                acc = [ $($acc:tt)* ] ;
                next = [ ] ;
                cont = { $($cont:tt)+ } ;
                orig = { $($orig:tt)* }
            ) => {
                $($cont)+ ! {
                    @enums [ $($acc)* { #definition } ] ;
                    $($orig)*
                }
            };
        }

        #[allow(unused_imports)]
        pub(crate) use #callback as #name;
    }
}

#[cfg(test)]
mod tests {
    use quote::quote;

    use super::{derive, expand};
    use crate::diagnostics::{
        Code,
        testing::{assert_code, assert_codes, parse2},
    };

    const GOOD: &str = "#[repr(u32)] enum Mode { Linear = 0, Curve, Step = 7, Last = 0x10u32 }";
    const HEAD: &str = "#[repr(u32)]";

    fn expansion(source: &str) -> syn::Result<proc_macro2::TokenStream> {
        expand(source.parse().unwrap())
    }

    /// The one refusal of `source`: one error, with the code, and its text.
    fn refused(source: &str) -> String {
        let error = expansion(source).unwrap_err();
        assert_code(&error, Code::GpuEnumShape);
        assert_eq!(error.clone().into_iter().count(), 1, "{error}");
        error.to_string()
    }

    fn assert_refused(source: &str, needle: &str) {
        let text = refused(source);
        assert!(text.contains(needle), "{needle} not in {text}");
    }

    #[test]
    fn the_impls_are_a_match_over_the_discriminants() {
        let output = expansion(GOOD).unwrap();
        parse2::<syn::File>(output.clone()).unwrap();
        let text = output.to_string();
        for expected in [
            quote!(match word {
                0u32 => ::core::option::Option::Some(Self::Linear),
                1u32 => ::core::option::Option::Some(Self::Curve),
                7u32 => ::core::option::Option::Some(Self::Step),
                16u32 => ::core::option::Option::Some(Self::Last),
                _ => ::core::option::Option::None,
            }),
            quote!(match self {
                Self::Linear => 0u32,
                Self::Curve => 1u32,
                Self::Step => 7u32,
                Self::Last => 16u32,
            }),
            quote!(match word {
                0u32 => ::core::result::Result::Ok(Self::Linear),
                1u32 => ::core::result::Result::Ok(Self::Curve),
                7u32 => ::core::result::Result::Ok(Self::Step),
                16u32 => ::core::result::Result::Ok(Self::Last),
                _ => ::core::result::Result::Err(::gust::EnumWordError::new(word)),
            }),
        ] {
            assert!(
                text.contains(&expected.to_string()),
                "{expected} not in {text}"
            );
        }
        assert!(text.contains("impl :: gust :: GpuEnum for Mode"), "{text}");
        assert!(
            text.contains("impl :: core :: convert :: TryFrom < u32 > for Mode"),
            "{text}"
        );
    }

    #[test]
    fn the_derive_adds_the_hidden_callback_macro_and_its_alias_for_a_batch() {
        let text = expansion(GOOD).unwrap().to_string();
        assert!(text.contains("macro_rules ! __gust_enum_Mode"), "{text}");
        assert!(
            text.contains("pub (crate) use __gust_enum_Mode as Mode"),
            "{text}"
        );
        // The definition handed on keeps the words, written out (`Curve` is 1, `Last` is 16).
        for expected in ["Linear = 0", "Curve = 1", "Step = 7", "Last = 16"] {
            assert!(text.contains(expected), "{expected} not in {text}");
        }
        assert!(text.contains("# [repr (u32)] enum Mode"), "{text}");
        // Not the user's derives, and no `Clone` or `Copy` (the stage's module adds its own).
        assert!(!text.contains("# [derive"), "{text}");
    }

    #[test]
    fn nothing_is_transmuted_or_cast_from_a_word() {
        let text = expansion(GOOD).unwrap().to_string();
        assert!(!text.contains("transmute"), "{text}");
        // The one `as` is the rename of the batch alias (`use .. as Mode`), never a cast.
        assert_eq!(
            text.split_whitespace()
                .filter(|token| *token == "as")
                .count(),
            1,
            "{text}"
        );
        assert!(text.contains("__gust_enum_Mode as Mode"), "{text}");
    }

    #[test]
    fn a_refused_enum_gets_the_error_and_no_impl() {
        let text = derive(format!("{HEAD} enum E {{ A(u32) }}").parse().unwrap()).to_string();
        assert!(text.contains("compile_error"), "{text}");
        assert!(!text.contains("impl"), "{text}");
    }

    #[test]
    fn a_data_carrying_variant_is_refused() {
        assert_refused(&format!("{HEAD} enum E {{ A, B(u32) }}"), "carries no data");
        assert_refused(
            &format!("{HEAD} enum E {{ A {{ x: u32 }} }}"),
            "carries no data",
        );
    }

    #[test]
    fn a_missing_or_other_repr_is_refused() {
        assert_refused("enum E { A }", "is written");
        for repr in ["i32", "u8", "u32, C", "C"] {
            assert_refused(
                &format!("#[repr({repr})] enum E {{ A }}"),
                "`#[repr(u32)]` and nothing else",
            );
        }
    }

    #[test]
    fn a_discriminant_that_is_not_a_literal_is_refused() {
        for discriminant in [
            "1 + 2",
            "SOME_CONST",
            "-1",
            "1.5",
            "300u8",
            "4294967296",
            "(1)",
        ] {
            refused(&format!("{HEAD} enum E {{ A = {discriminant} }}"));
        }
    }

    #[test]
    fn an_implicit_discriminant_after_the_largest_word_is_rustcs_to_refuse() {
        // rustc refuses the enum (E0370); the derive only has to give the variant some word and not overflow.
        assert!(expansion(&format!("{HEAD} enum E {{ A = 4294967295, B }}")).is_ok());
    }

    #[test]
    fn generics_are_refused() {
        assert_refused(&format!("{HEAD} enum E<T> {{ A }}"), "generic enums");
        assert_refused(
            &format!("{HEAD} enum E where u32: Copy {{ A }}"),
            "generic enums",
        );
    }

    #[test]
    fn an_empty_enum_and_a_struct_are_refused() {
        assert_refused(&format!("{HEAD} enum E {{}}"), "at least one variant");
        assert_refused(&format!("{HEAD} struct S;"), "goes on an enum");
    }

    #[test]
    fn every_offending_token_is_reported_once_in_source_order() {
        let error = expansion("#[repr(i32)] enum E<T> { A(u32), B = 1 + 1 }").unwrap_err();
        assert_codes(&error, &[Code::GpuEnumShape; 4]);
        let text: Vec<String> = error.into_iter().map(|part| part.to_string()).collect();
        for (part, needle) in text.iter().zip([
            "generic enums",
            "nothing else",
            "carries no data",
            "non-negative integer literal",
        ]) {
            assert!(part.contains(needle), "{needle} not in {part}");
        }
    }
}
