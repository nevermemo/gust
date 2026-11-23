//! The facade path option (`#[gpu_module(crate = <path>)]`, `par_for!`'s `crate(<path>);`; audit S6, D48).
//!
//! The expansions spell the facade as the literal `::gust::` in several hundred places. A crate that depends on
//! `gust` only through another crate (which re-exports it with `pub use gust;`) has no `::gust` of its own, so the
//! option names where the facade is, and a final pass over the emitted tokens rewrites the facade's leading
//! `:: gust` to it. With no option the pass does not run, so the emission is exactly what it was.
//!
//! The pass is syntactic, which has two consequences, documented for users in the README:
//!
//! - It rewrites each `::gust::` in the output that begins a path (the exact rule is on `continues_a_path`: not one
//!   after a path segment, and not one after a `>` that is taken to close a path's arguments, a comparison after an
//!   unmatched `<` operator included), the user's own tokens that spell one included.
//! - It leaves `::gust_wgpu` alone: the function-map form, `#[gpu_fn]`, and `par_for!` still need a direct dependency
//!   on `gust-compute-wgpu`.

use proc_macro2::{Group, Spacing, TokenStream, TokenTree};
use quote::ToTokens;
use syn::parse::ParseStream;

use crate::diagnostics::{self, Code};

/// Parses the path of the option: `::inner::gust` or `crate::inner::gust`, plain identifiers only. A path that starts
/// at the current module (`inner::gust`, `self::..`, `super::..`) would mean something else inside a nested module of
/// the expansion, a generic argument has no place in a module path, and `$crate` is no identifier. Failures carry
/// `code`.
pub(crate) fn parse_facade(input: ParseStream<'_>, code: Code) -> syn::Result<syn::Path> {
    let path = syn::Path::parse_mod_style(input).map_err(|error| diagnostics::wrap(code, error))?;
    let absolute = path.leading_colon.is_some()
        || path
            .segments
            .first()
            .is_some_and(|segment| segment.ident == "crate");
    if !absolute {
        return Err(diagnostics::error(
            code,
            &path,
            "the facade path starts with `::` (another crate) or `crate::` (this one), for example `::inner::gust`",
        ));
    }
    for (index, segment) in path.segments.iter().enumerate() {
        let ident = &segment.ident;
        let keyword = ident == "self" || ident == "super" || ident == "Self";
        if keyword || (ident == "crate" && (index != 0 || path.leading_colon.is_some())) {
            return Err(diagnostics::error(
                code,
                ident,
                format!("`{ident}` has no place in the facade path"),
            ));
        }
    }
    Ok(path)
}

/// `stream` with each `::gust` that begins a path (followed by `::`) replaced by `facade`, groups included.
pub(crate) fn forward_crate(stream: TokenStream, facade: &syn::Path) -> TokenStream {
    rewrite(stream, &facade.to_token_stream())
}

fn rewrite(stream: TokenStream, facade: &TokenStream) -> TokenStream {
    let tokens: Vec<TokenTree> = stream.into_iter().collect();
    let mut output: Vec<TokenTree> = Vec::with_capacity(tokens.len());
    let mut index = 0;
    while index < tokens.len() {
        if leads_the_facade(&tokens, index) {
            output.extend(facade.clone());
            // `::` and `gust`; the `::` after them stays.
            index += 3;
            continue;
        }
        match &tokens[index] {
            TokenTree::Group(group) => {
                let mut inner = Group::new(group.delimiter(), rewrite(group.stream(), facade));
                inner.set_span(group.span());
                output.push(TokenTree::Group(inner));
            }
            other => output.push(other.clone()),
        }
        index += 1;
    }
    output.into_iter().collect()
}

/// Whether `tokens[index..]` starts `:: gust ::` and the first `::` begins a path (it is not the separator after a
/// segment, as in the user's `::inner::gust::`).
fn leads_the_facade(tokens: &[TokenTree], index: usize) -> bool {
    let path_separator = |at: usize| {
        matches!(
            (tokens.get(at), tokens.get(at + 1)),
            (Some(TokenTree::Punct(first)), Some(TokenTree::Punct(second)))
                if first.as_char() == ':'
                    && first.spacing() == Spacing::Joint
                    && second.as_char() == ':'
        )
    };
    path_separator(index)
        && matches!(tokens.get(index + 2), Some(TokenTree::Ident(ident)) if ident == "gust")
        && path_separator(index + 3)
        && !continues_a_path(tokens, index)
}

/// Whether the token in front of `tokens[index]` makes a `::` a separator. The rule, exactly as implemented (spacing
/// is ignored: `S<T> ::gust::x` and `S<T>::gust::x` are the same token stream, and neither is rewritten):
///
/// - an identifier is a path segment unless it is a keyword after which a path may begin (`impl`, `as`, `for`, ..) or
///   the name of a lifetime (`&'a ::gust::X`);
/// - a `>` that is part of `->` or `=>` is not a separator;
/// - any other `>` is matched with the nearest unmatched earlier `<` in the same group, any `<` counting as an opener,
///   the operators `<`, `<=` and `<<` included, in this statement or an earlier one of the block. If there is none, the
///   `>` is a comparison or a shift (`a > ::gust::K`) and a path begins. If that `<` follows `impl` or `for`, the `>`
///   ends a list of generics (`impl<T> ::gust::Tr`) and a path begins. Otherwise the `>` is taken to close the
///   arguments of a path (`Vec<T>::gust::..`, `Vec::<T>::gust::..`) or a qualified path (`<T as X>::gust::..`), and the
///   `::` is a separator: so `a < b && d > ::gust::K` is left as written (rustc then reports E0433 for it, never wrong
///   code); write `d > (::gust::K)` instead, where the path begins a group and is rewritten.
fn continues_a_path(tokens: &[TokenTree], index: usize) -> bool {
    let Some(previous) = index.checked_sub(1).and_then(|at| tokens.get(at)) else {
        return false;
    };
    match previous {
        TokenTree::Ident(ident) => {
            let lifetime = index
                .checked_sub(2)
                .and_then(|at| tokens.get(at))
                .is_some_and(
                    |before| matches!(before, TokenTree::Punct(tick) if tick.as_char() == '\''),
                );
            !lifetime && !is_keyword_before_a_type_or_path(&ident.to_string())
        }
        TokenTree::Punct(punct) if punct.as_char() == '>' => {
            if is_arrow(tokens, index - 1) {
                return false;
            }
            match matching_open(tokens, index - 1) {
                None => false,
                Some(open) => match open.checked_sub(1).and_then(|at| tokens.get(at)) {
                    Some(TokenTree::Ident(keyword)) => {
                        !matches!(keyword.to_string().as_str(), "impl" | "for")
                    }
                    _ => true,
                },
            }
        }
        _ => false,
    }
}

/// Whether the `>` at `tokens[at]` is the end of `->` or `=>`.
fn is_arrow(tokens: &[TokenTree], at: usize) -> bool {
    at.checked_sub(1)
        .and_then(|before| tokens.get(before))
        .is_some_and(|before| {
            matches!(before, TokenTree::Punct(joint)
                if matches!(joint.as_char(), '-' | '=') && joint.spacing() == Spacing::Joint)
        })
}

/// The index of the `<` that the `>` at `tokens[close]` closes, counting back through `<` and `>` in the same group
/// (arrows do not count); `None` when the `>` closes nothing.
fn matching_open(tokens: &[TokenTree], close: usize) -> Option<usize> {
    let mut depth = 0_usize;
    for at in (0..=close).rev() {
        if let TokenTree::Punct(punct) = &tokens[at] {
            match punct.as_char() {
                '>' if !is_arrow(tokens, at) => depth += 1,
                '<' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(at);
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// The keywords after which a path may begin: `impl ::gust::Trait for ::gust::Type`, `x as ::gust::T`,
/// `use ::gust::..`. `crate`, `self`, `super` and `Self` are path segments and are not here.
fn is_keyword_before_a_type_or_path(ident: &str) -> bool {
    matches!(
        ident,
        "as" | "async"
            | "await"
            | "box"
            | "break"
            | "const"
            | "continue"
            | "do"
            | "dyn"
            | "else"
            | "enum"
            | "extern"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "static"
            | "struct"
            | "trait"
            | "type"
            | "unsafe"
            | "use"
            | "where"
            | "while"
            | "yield"
    )
}
