//! The facade path option (audit S6, D48): the forwarder pass, the `#[gpu_module(crate = ..)]` grammar and the `par_for!`
//! clause. The two-crate fixture that compiles the four forms is `tests/reexport` (run by `cargo xtask check-ui`).

use proc_macro2::TokenStream;

use crate::{
    diagnostics::{
        self, Code,
        testing::{assert_code, assert_rendered_code, parse_str},
    },
    expand::{Environment, Profile, routing},
    forward::{forward_crate, parse_facade},
    par_for::par_for_in,
};

const FACADE: &str = ":: inner :: gust";

struct Facade(syn::Path);

impl syn::parse::Parse for Facade {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        parse_facade(input, Code::GpuArguments).map(Self)
    }
}

fn path(text: &str) -> syn::Result<syn::Path> {
    parse_str::<Facade>(text)
        .map(|facade| facade.0)
        .map_err(|error| diagnostics::wrap(Code::GpuArguments, error))
}

/// The text of `source` after the pass, with the facade `::inner::gust`.
fn forwarded(source: &str) -> String {
    let stream: TokenStream = source.parse().unwrap();
    forward_crate(stream, &path("::inner::gust").unwrap()).to_string()
}

/// The text of `source` as tokens, so that the expected text of a test is spaced the way the output is.
fn tokens(source: &str) -> String {
    source.parse::<TokenStream>().unwrap().to_string()
}

#[test]
fn a_path_leading_facade_is_rewritten() {
    assert_eq!(
        forwarded("use ::gust::math::Vec3;"),
        tokens("use ::inner::gust::math::Vec3;")
    );
    assert_eq!(
        forwarded("let x = ::gust::cpu::CpuDevice::new();"),
        tokens("let x = ::inner::gust::cpu::CpuDevice::new();")
    );
    // After a keyword, a punctuation, an arrow and the start of a stream: all begin a path.
    assert_eq!(
        forwarded("impl ::gust::routed::Routed for ::gust::Thing {}"),
        tokens("impl ::inner::gust::routed::Routed for ::inner::gust::Thing {}")
    );
    assert_eq!(
        forwarded("fn f(a: &::gust::A, b: Vec<::gust::B>) -> ::gust::C { x as ::gust::D }"),
        tokens(
            "fn f(a: &::inner::gust::A, b: Vec<::inner::gust::B>) -> ::inner::gust::C { x as ::inner::gust::D }"
        )
    );
    assert_eq!(
        forwarded("match x { 1 => ::gust::a(), _ => ::gust::b() }"),
        tokens("match x { 1 => ::inner::gust::a(), _ => ::inner::gust::b() }")
    );
    assert_eq!(forwarded("::gust::x"), tokens("::inner::gust::x"));
}

#[test]
fn groups_are_searched_at_every_depth() {
    assert_eq!(
        forwarded("a({ [ ::gust::x, (::gust::y) ] })"),
        tokens("a({ [ ::inner::gust::x, (::inner::gust::y) ] })")
    );
}

#[test]
fn a_facade_inside_a_path_is_not_rewritten() {
    // The user's own `::inner::gust::` is not prefixed twice, nor is any other path that has a `gust` segment.
    for source in [
        "::inner::gust::x",
        "inner::gust::x",
        "other::gust::x",
        "<T as Tr>::gust::x",
        "crate::gust::x",
        "self::gust::x",
        "a::<B>::gust::c",
    ] {
        assert_eq!(forwarded(source), tokens(source), "{source}");
    }
}

#[test]
fn other_leading_names_are_left_alone() {
    // Not `gust`, not followed by `::`, or another crate with a longer name.
    for source in [
        "::gust_wgpu::x",
        "::gusty::x",
        "::gust",
        "use ::gust;",
        "::std::gust::x",
        "gust::x",
        ":: other::x",
    ] {
        assert_eq!(forwarded(source), tokens(source), "{source}");
    }
    assert_eq!(
        forwarded("::gust_wgpu::GpuVec<::gust::T>"),
        tokens("::gust_wgpu::GpuVec<::inner::gust::T>")
    );
}

#[test]
fn a_crate_path_is_a_facade_too() {
    let stream: TokenStream = "x(::gust::T) ; ::gust::f()".parse().unwrap();
    let output = forward_crate(stream, &path("crate::reexport::gust").unwrap()).to_string();
    assert_eq!(
        output,
        tokens("x(crate::reexport::gust::T) ; crate::reexport::gust::f()")
    );
}

#[test]
fn only_absolute_plain_paths_are_facades() {
    for good in ["::inner::gust", "crate::inner::gust", "::gust", "::a::b::c"] {
        assert!(path(good).is_ok(), "{good}");
    }
    for bad in [
        "inner::gust",
        "gust",
        "self::gust",
        "super::gust",
        "::inner::self",
        "::inner::super::gust",
        "::inner::Self",
        "::a::crate::b",
        "crate::crate::b",
        "::inner::gust<u8>",
        "::inner::<u8>::gust",
        "$crate::gust",
        "::inner::",
        "",
    ] {
        let error = path(bad).expect_err(bad);
        assert_code(&error, Code::GpuArguments);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// `#[gpu_fn(crate = ..)]`

fn attribute(arguments: &str) -> syn::Result<routing::Attribute> {
    routing::parse_attribute(arguments.parse().unwrap())
}

fn facade_text(attribute: &routing::Attribute) -> String {
    let path = attribute.krate.as_ref().unwrap();
    quote::ToTokens::to_token_stream(path).to_string()
}

#[test]
fn the_attribute_takes_route_and_crate_in_either_order() {
    for arguments in [
        "crate = ::inner::gust, route = \"physics\"",
        "route = \"physics\", crate = ::inner::gust",
        "route = \"physics\", crate = ::inner::gust,",
        "crate=::inner::gust,route=\"physics\"",
    ] {
        let parsed = attribute(arguments).expect(arguments);
        assert_eq!(parsed.tag.as_ref().unwrap().text, "physics", "{arguments}");
        assert_eq!(facade_text(&parsed), FACADE, "{arguments}");
    }
    let alone = attribute("crate = crate::x::gust").unwrap();
    assert!(alone.tag.is_none());
    assert_eq!(facade_text(&alone), tokens("crate::x::gust"));
    assert!(attribute("crate = ::inner::gust,").unwrap().tag.is_none());
    assert!(attribute("route = \"a\"").unwrap().krate.is_none());
}

#[test]
fn a_malformed_or_repeated_crate_is_e0701() {
    for arguments in [
        "crate",
        "crate =",
        "crate = 5",
        "crate = \"::inner::gust\"",
        "crate = inner::gust",
        "crate = ::inner::gust<u8>",
        "crate = $crate",
        "crate = ::inner::gust extra",
        "crate = ::inner::gust route = \"a\"",
        "crate = ::inner::gust, crate = ::other::gust",
        "crate = ::inner::gust, crate = ::inner::gust",
        "route = \"a\", route = \"b\"",
        "route = \"a\", crate = ::inner::gust, route = \"b\"",
        "crate(::inner::gust)",
        ", crate = ::inner::gust",
        "crate = ::inner::gust,, route = \"a\"",
    ] {
        let error = attribute(arguments)
            .err()
            .unwrap_or_else(|| panic!("{arguments} was accepted"));
        assert_code(&error, Code::GpuArguments);
    }
}

const MODULE: &str = "mod k {
    #[kernel]
    pub fn scale(id: SV_DispatchThreadID, input: StructuredBuffer<float>, mut out: RWStructuredBuffer<float>) {
        let i = id.x;
        if i < out.len() && i < input.len() {
            out[i] = input[i] * 2.0;
        }
    }
}";

const FUNCTION: &str = "pub fn saxpy(a: f32, x: f32, y: f32) -> f32 { a * x + y }";

fn expand(arguments: &str, item: &str) -> String {
    crate::gpu_impl_with(
        arguments.parse().unwrap(),
        item.parse().unwrap(),
        Profile::Other,
    )
    .to_string()
}

/// `text` with the facade spelled the way the option spells it; what a forwarded expansion must equal.
fn with_facade(text: &str) -> String {
    text.replace(":: gust ::", ":: inner :: gust ::")
}

#[test]
fn the_module_form_is_the_plain_expansion_with_the_facade_replaced() {
    let plain = expand("", MODULE);
    assert!(
        plain.contains(":: gust ::"),
        "the expansion names the facade: {plain}"
    );
    let forwarded = expand("crate = ::inner::gust", MODULE);
    assert_eq!(forwarded, with_facade(&plain));
    // Every facade is the option's: none is left over.
    assert!(
        !forwarded.replace(FACADE, "").contains(":: gust"),
        "{forwarded}"
    );
    // The order of the arguments does not matter, and a trailing comma is fine.
    assert_eq!(expand("crate = ::inner::gust,", MODULE), forwarded);
}

#[test]
fn without_the_option_the_expansion_is_the_old_one() {
    // A facade that is `::gust` itself rewrites each `::gust` to the same tokens.
    let plain = expand("", MODULE);
    assert!(!plain.contains(FACADE), "{plain}");
    assert_eq!(expand("crate = ::gust", MODULE), plain);
}

#[test]
fn the_function_form_forwards_gust_and_keeps_gust_wgpu_literal() {
    let plain = expand("", FUNCTION);
    assert!(
        plain.contains(":: gust_wgpu"),
        "the map form names gust_wgpu: {plain}"
    );
    let forwarded = expand("crate = ::inner::gust", FUNCTION);
    assert_eq!(forwarded, with_facade(&plain));
    assert!(forwarded.contains(":: gust_wgpu"), "{forwarded}");
    assert!(forwarded.contains(FACADE), "{forwarded}");
}

#[test]
fn a_route_and_the_facade_together() {
    // The route is declared nowhere here, so the expansion refuses it (GUST-E0748); that error is forwarded like the
    // rest and the module still follows it.
    let output = expand("route = \"nowhere\", crate = ::inner::gust", MODULE);
    assert!(output.starts_with(":: core :: compile_error !"), "{output}");
    assert!(output.contains("GUST-E074"), "{output}");
    assert!(output.contains("mod k"), "{output}");
}

#[test]
fn a_malformed_attribute_is_a_coded_error_beside_the_item() {
    let output = expand("crate = inner::gust", MODULE);
    assert!(output.starts_with(":: core :: compile_error !"), "{output}");
    assert_rendered_code(&output, Code::GpuArguments);
    assert!(output.contains("mod k"), "{output}");
}

#[test]
fn user_tokens_that_spell_the_facade_are_rewritten_too() {
    // Documented: the pass is syntactic, and a leading `::gust::` in the user's own code is rewritten.
    let source = "mod k {
        pub fn helper() -> u32 { ::gust::answer() }
        #[kernel]
        pub fn scale(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) { out[id.x] = 1.0; }
    }";
    let forwarded = expand("crate = ::inner::gust", source);
    assert!(
        forwarded.contains(":: inner :: gust :: answer"),
        "{forwarded}"
    );
}

// ---------------------------------------------------------------------------------------------------------------
// `par_for!`

const LOOP: &str = "i in 0..n, [a: &[f32], out: &mut [f32]] { out[i] = a[i] * 2.0; }";

fn par_for(clauses: &str) -> String {
    let environment = Environment {
        manifest_dir: None,
        export_dir: None,
        package: Some("demo".to_owned()),
        reported_file: Some("src/lib.rs".to_owned()),
    };
    let input: TokenStream = format!("{clauses}{LOOP}").parse().unwrap();
    par_for_in(input, &environment, Profile::Other, 1).to_string()
}

#[test]
fn the_par_for_clause_forwards_the_facade() {
    let plain = par_for("");
    assert!(
        plain.contains(":: gust ::") && plain.contains(":: gust_wgpu"),
        "{plain}"
    );
    let forwarded = par_for("crate(::inner::gust); ");
    assert_eq!(forwarded, with_facade(&plain));
    assert!(forwarded.contains(":: gust_wgpu"), "{forwarded}");
    // Among the other clauses, in either place.
    let all = par_for("workgroup_size(8, 4, 1); crate(::inner::gust); ");
    assert!(
        all.contains(FACADE) && !all.replace(FACADE, "").contains(":: gust ::"),
        "{all}"
    );
    let before = par_for("crate(::inner::gust); workgroup_size(8, 4, 1); ");
    assert_eq!(before, all);
}

#[test]
fn a_bad_par_for_clause_is_a_form_error() {
    for clauses in [
        "crate(::inner::gust); crate(::inner::gust); ",
        "crate(inner::gust); ",
        "crate(::inner::gust, ::other::gust); ",
        "crate(); ",
        "crate(::inner::gust) ",
    ] {
        let output = par_for(clauses);
        assert!(
            output.starts_with("{ :: core :: compile_error"),
            "{clauses}: {output}"
        );
        assert_rendered_code(&output, Code::ParForForm);
    }
}

#[test]
fn an_error_after_the_clause_still_names_the_forwarded_facade() {
    // The error's own value (`Result<(), gust::map::Error>`) is spelled with the facade, so it is forwarded too and
    // the call has one error, not an unresolved `::gust` besides.
    let environment = Environment::default();
    let input: TokenStream = "crate(::inner::gust); i in 0..n, [__gust_a: &[f32]] { }"
        .parse()
        .unwrap();
    let output = par_for_in(input, &environment, Profile::Other, 1).to_string();
    assert!(output.starts_with("{ :: core :: compile_error"), "{output}");
    assert!(
        output.contains(":: inner :: gust :: map :: Error"),
        "{output}"
    );
    assert!(
        !output.replace(FACADE, "").contains(":: gust ::"),
        "{output}"
    );
}

// ---------------------------------------------------------------------------------------------------------------
// Where a path begins (review of m10.api2a.3, item 2)

#[test]
fn a_path_after_a_lifetime_is_rewritten() {
    for (source, expected) in [
        (
            "fn f(x: &'a ::gust::X) {}",
            "fn f(x: &'a ::inner::gust::X) {}",
        ),
        (
            "fn f(x: &'static ::gust::X) {}",
            "fn f(x: &'static ::inner::gust::X) {}",
        ),
        (
            "fn f<'a>(x: &'a mut ::gust::X) {}",
            "fn f<'a>(x: &'a mut ::inner::gust::X) {}",
        ),
        (
            "type T = dyn Tr + 'a ::gust::X;",
            "type T = dyn Tr + 'a ::inner::gust::X;",
        ),
    ] {
        assert_eq!(forwarded(source), tokens(expected), "{source}");
    }
}

#[test]
fn a_path_after_the_end_of_generics_is_rewritten() {
    for (source, expected) in [
        (
            "impl<T> ::gust::Tr for X<T> {}",
            "impl<T> ::inner::gust::Tr for X<T> {}",
        ),
        (
            "impl<T: Clone, U> ::gust::Tr for X {}",
            "impl<T: Clone, U> ::inner::gust::Tr for X {}",
        ),
        (
            "fn f(g: impl for<'a> ::gust::F<'a>) {}",
            "fn f(g: impl for<'a> ::inner::gust::F<'a>) {}",
        ),
        (
            "impl<T: Into<u8>> ::gust::Tr for X {}",
            "impl<T: Into<u8>> ::inner::gust::Tr for X {}",
        ),
    ] {
        assert_eq!(forwarded(source), tokens(expected), "{source}");
    }
}

#[test]
fn a_path_after_a_comparison_or_a_shift_is_rewritten() {
    for (source, expected) in [
        ("let c = a > ::gust::K;", "let c = a > ::inner::gust::K;"),
        ("let c = a >> ::gust::K;", "let c = a >> ::inner::gust::K;"),
        (
            "let c = f(x) > ::gust::K;",
            "let c = f(x) > ::inner::gust::K;",
        ),
    ] {
        assert_eq!(forwarded(source), tokens(expected), "{source}");
    }
}

#[test]
fn a_path_after_the_arguments_of_a_path_is_not_rewritten() {
    // The `>` closes the arguments of a path or a qualified path, so the `::` is a separator.
    for source in [
        "Vec<T>::gust::x",
        "Vec::<T>::gust::x",
        "<T as Tr>::gust::x",
        "<Vec<T> as Tr>::gust::x",
        "let v = <Vec<Vec<u8>> as Tr>::gust::x;",
        "a::b<C<D>>::gust::x",
        "x = <T as Tr<U>>::gust::x;",
    ] {
        assert_eq!(forwarded(source), tokens(source), "{source}");
    }
}

#[test]
fn an_error_after_a_clause_names_the_forwarded_facade_when_the_loop_does_not_parse() {
    // A malformed binding list fails the whole parse, after a valid `crate(..);`: the error value still uses the path.
    let environment = Environment::default();
    let input: TokenStream = "crate(::inner::gust); i in 0..n, [a: ] { }"
        .parse()
        .unwrap();
    let output = par_for_in(input, &environment, Profile::Other, 1).to_string();
    assert!(output.starts_with("{ :: core :: compile_error"), "{output}");
    assert!(
        output.contains(":: inner :: gust :: map :: Error"),
        "{output}"
    );
    assert!(
        !output.replace(FACADE, "").contains(":: gust ::"),
        "{output}"
    );
}

#[test]
fn a_comparison_after_an_unmatched_operator_is_left_as_written() {
    // The back-count treats every `<` as an opener, an operator too (the documented rule), so the `>` here matches
    // the `<` of `a < b`, and the path is left literal; `d > (::gust::K)` is the way to write it.
    for source in [
        "let c = a < b && d > ::gust::K;",
        "let x = a < b; let y = c > ::gust::K;",
        "let c = a << b > ::gust::K;",
    ] {
        assert_eq!(forwarded(source), tokens(source), "{source}");
    }
    assert_eq!(
        forwarded("let c = a < b && d > (::gust::K);"),
        tokens("let c = a < b && d > (::inner::gust::K);")
    );
}

#[test]
fn spacing_is_ignored() {
    // `S<T> ::gust::x` and `S<T>::gust::x` differ only in whitespace, which is no token: neither is rewritten.
    for source in ["S<T> ::gust::x", "S<T>::gust::x"] {
        assert_eq!(forwarded(source), tokens(source), "{source}");
    }
}
