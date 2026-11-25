//! The library ops of a `gpu_batch!` (plan m14.s2, slice s2f2c, D262): `gust::sum`, `sum_wrapping`, `min`, `max`,
//! `reduce`, `count`, `scan`, `scan_exclusive` and `compact`, written as the whole init of a `let` in the build block.
//!
//! ```text
//! let total = gust::sum(&blocks);
//! let (idx, n) = gust::compact(&flags);
//! ```
//!
//! The macro lowers each to the builder API of `gust::batch` and gust-algo's op types, naming every output by its
//! `let`:
//!
//! ```text
//! let total = b.op(&["total"], ::gust_algo::batch::Sum::of(&blocks));
//! let (idx, n) = b.op(&["idx", "n"], ::gust_algo::batch::Compact::of(&flags));
//! ```
//!
//! This file recognises the call, reads the pattern and writes the emission; [`crate::batch`] walks the block and
//! gives each output its type. An op that is not the whole init of such a `let` is GUST-E1009, and a crate that does
//! not depend on `gust-algo` is GUST-E1010 (read from the expanding crate's manifest, as `[package.metadata.gust]` is;
//! with `crate(<path>)` or a manifest that cannot be read, rustc speaks).

use std::path::Path;

use proc_macro2::TokenStream;
use quote::quote;
use syn::ext::IdentExt as _;

use crate::{
    diagnostics::{self, Code},
    manifest,
};

/// The crate that implements the ops.
const ALGO: &str = "gust-algo";

/// A library op of a build block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    Sum,
    SumWrapping,
    Min,
    Max,
    Reduce,
    Count,
    Scan,
    ScanExclusive,
    Compact,
}

impl Op {
    fn named(name: &str) -> Option<Self> {
        Some(match name {
            "sum" => Self::Sum,
            "sum_wrapping" => Self::SumWrapping,
            "min" => Self::Min,
            "max" => Self::Max,
            "reduce" => Self::Reduce,
            "count" => Self::Count,
            "scan" => Self::Scan,
            "scan_exclusive" => Self::ScanExclusive,
            "compact" => Self::Compact,
            _ => return None,
        })
    }

    /// The name as written after `gust::`.
    fn written(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::SumWrapping => "sum_wrapping",
            Self::Min => "min",
            Self::Max => "max",
            Self::Reduce => "reduce",
            Self::Count => "count",
            Self::Scan => "scan",
            Self::ScanExclusive => "scan_exclusive",
            Self::Compact => "compact",
        }
    }

    /// How the call is written, for a message.
    fn usage(self) -> &'static str {
        match self {
            Self::Reduce => "let total = gust::reduce(&values, Op::Max);",
            Self::Compact => "let (idx, n) = gust::compact(&flags);",
            Self::Sum => "let total = gust::sum(&values);",
            Self::SumWrapping => "let total = gust::sum_wrapping(&values);",
            Self::Min => "let least = gust::min(&values);",
            Self::Max => "let most = gust::max(&values);",
            Self::Count => "let n = gust::count(&flags);",
            Self::Scan => "let sums = gust::scan(&values);",
            Self::ScanExclusive => "let offsets = gust::scan_exclusive(&values);",
        }
    }

    /// How many arguments the call takes.
    fn arguments(self) -> usize {
        if self == Self::Reduce { 2 } else { 1 }
    }

    /// The op's type in gust-algo, and the function that makes it from its operand.
    fn constructor(self) -> TokenStream {
        match self {
            Self::Sum => quote!(::gust_algo::batch::Sum::of),
            Self::SumWrapping => quote!(::gust_algo::batch::SumWrapping::of),
            Self::Min => quote!(::gust_algo::batch::Min::of),
            Self::Max => quote!(::gust_algo::batch::Max::of),
            Self::Reduce => quote!(::gust_algo::batch::Reduce::of),
            Self::Count => quote!(::gust_algo::batch::Count::of),
            Self::Scan => quote!(::gust_algo::batch::Scan::of),
            Self::ScanExclusive => quote!(::gust_algo::batch::Scan::exclusive),
            Self::Compact => quote!(::gust_algo::batch::Compact::of),
        }
    }

    /// The element type of the output: the operand's for a reduction, `u32` for the rest.
    pub(crate) fn element(self, operand: Option<syn::Type>) -> Option<syn::Type> {
        match self {
            Self::Sum | Self::Min | Self::Max | Self::Reduce => operand,
            _ => Some(syn::parse_quote!(u32)),
        }
    }
}

/// The op a call is, when it is written `gust::<op>(..)` or `::gust::<op>(..)` (or through the `crate(<path>)` facade).
pub(crate) fn recognise(call: &syn::ExprCall, facade: Option<&syn::Path>) -> Option<Op> {
    let syn::Expr::Path(path) = &*call.func else {
        return None;
    };
    if path.qself.is_some()
        || path
            .path
            .segments
            .iter()
            .any(|segment| !segment.arguments.is_none())
    {
        return None;
    }
    let segments: Vec<&syn::Ident> = path
        .path
        .segments
        .iter()
        .map(|segment| &segment.ident)
        .collect();
    let (last, prefix) = segments.split_last()?;
    let op = Op::named(&last.to_string())?;
    let through_gust = prefix.len() == 1 && prefix[0] == "gust";
    let through_facade = facade.is_some_and(|facade| {
        facade.segments.len() == prefix.len()
            && facade
                .segments
                .iter()
                .zip(prefix)
                .all(|(written, seen)| written.ident == **seen)
    });
    (through_gust || through_facade).then_some(op)
}

/// A call of an op that is not the whole init of a `let`.
pub(crate) fn not_a_let(call: &syn::ExprCall, op: Op) -> syn::Error {
    diagnostics::error(
        Code::BatchOpForm,
        &call.func,
        format!(
            "`gust::{}` is the whole init of a `let` in the build block, which names its output: `{}`",
            op.written(),
            op.usage()
        ),
    )
}

/// An op in a place the block does not record once.
pub(crate) fn misplaced(call: &syn::ExprCall, op: Op, place: &str) -> syn::Error {
    diagnostics::error(
        Code::BatchOpForm,
        &call.func,
        format!(
            "an op of a batch is recorded once, when the block runs: this `gust::{}` is inside {place}. Write it \
             directly in the build block",
            op.written()
        ),
    )
}

/// The lowered `let`: the init to put in its place, and the names its outputs got.
pub(crate) struct Lowered {
    pub init: syn::Expr,
    pub names: Vec<syn::Ident>,
}

/// The plain name of a pattern (`total`, `total: Stage<u32>`); not `_`, `mut`, `ref` or a pattern with a part.
fn plain_name(pat: &syn::Pat) -> Option<&syn::Ident> {
    match pat {
        syn::Pat::Ident(ident)
            if ident.subpat.is_none() && ident.by_ref.is_none() && ident.mutability.is_none() =>
        {
            Some(&ident.ident)
        }
        syn::Pat::Type(typed) => plain_name(&typed.pat),
        _ => None,
    }
}

/// The names the pattern gives the outputs of `op`: one plain name, or for `compact` a tuple of two.
fn output_names(op: Op, pat: &syn::Pat) -> syn::Result<Vec<syn::Ident>> {
    let wrong = |what: &str| {
        diagnostics::error(
            Code::BatchOpForm,
            pat,
            format!(
                "{what}; an op names each output by its `let`: `{}`",
                op.usage()
            ),
        )
    };
    if op == Op::Compact {
        let syn::Pat::Tuple(tuple) = pat else {
            return Err(wrong(
                "`gust::compact` gives two handles, the indices and their count, so it takes a tuple of two names",
            ));
        };
        let names: Vec<syn::Ident> = tuple
            .elems
            .iter()
            .filter_map(|element| plain_name(element).cloned())
            .collect();
        if tuple.elems.len() != 2 || names.len() != 2 {
            return Err(wrong(
                "`gust::compact` takes exactly a tuple of two plain names (no `_`, `mut`, `ref`, type or nested pattern)",
            ));
        }
        if names[0] == names[1] {
            return Err(wrong(
                "the two outputs of `gust::compact` need different names",
            ));
        }
        return Ok(names);
    }
    match pat {
        syn::Pat::Type(typed) if matches!(*typed.pat, syn::Pat::Tuple(_)) => {
            Err(wrong(&format!("`gust::{}` gives one handle", op.written())))
        }
        syn::Pat::Tuple(_) => Err(wrong(&format!("`gust::{}` gives one handle", op.written()))),
        _ => match plain_name(pat) {
            Some(name) => Ok(vec![name.clone()]),
            None => Err(wrong("the output of an op is a plain name")),
        },
    }
}

/// Lower `let <pat> = gust::<op>(args)` for the builder `builder`.
pub(crate) fn lower(
    op: Op,
    call: &syn::ExprCall,
    pat: &syn::Pat,
    builder: &syn::Ident,
) -> syn::Result<Lowered> {
    if call.args.len() != op.arguments() {
        return Err(diagnostics::error(
            Code::BatchOpForm,
            call,
            format!(
                "`gust::{}` takes {} argument{}: `{}`",
                op.written(),
                op.arguments(),
                if op.arguments() == 1 { "" } else { "s" },
                op.usage()
            ),
        ));
    }
    let names = output_names(op, pat)?;
    let labels: Vec<String> = names.iter().map(|name| name.unraw().to_string()).collect();
    let constructor = op.constructor();
    let mut arguments = call.args.iter();
    let first = arguments.next();
    let second = arguments.next();
    // `gust::reduce(&s, op)` reads in the order of the sentence; gust-algo's `Reduce::of` takes the operation first.
    let operands = match (op, first, second) {
        (Op::Reduce, Some(values), Some(reduction)) => quote!(#reduction, #values),
        (_, Some(values), _) => quote!(#values),
        _ => unreachable!("the number of arguments was checked"),
    };
    let init = syn::parse_quote!(#builder.op(&[#(#labels),*], #constructor(#operands)));
    Ok(Lowered { init, names })
}

/// Whether the crate whose manifest is in `manifest_dir` may use the ops: GUST-E1010 at `at` when its manifest lists no
/// `gust-algo`; `Ok` when it does or when the manifest cannot say.
pub(crate) fn require_algo(manifest_dir: Option<&Path>, at: &syn::Path) -> syn::Result<()> {
    match manifest::depends_on(manifest_dir, ALGO) {
        Some(false) => Err(diagnostics::error(
            Code::BatchOpCrate,
            at,
            format!(
                "the ops of a batch are implemented by the `{ALGO}` crate, which this crate does not depend on: add \
                 `{ALGO}` to its `[dependencies]` (or `[dev-dependencies]`)"
            ),
        )),
        Some(true) | None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use proc_macro2::TokenStream;

    use crate::{
        batch::gpu_batch_in,
        diagnostics::{Code, testing::assert_rendered_code},
        manifest,
    };

    /// A crate directory with this `Cargo.toml`, removed on drop.
    struct Crate(PathBuf);

    impl Crate {
        fn new(manifest: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "gust batch ops {} {}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            fs::write(path.join("Cargo.toml"), manifest).unwrap();
            Self(path)
        }
    }

    impl Drop for Crate {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// The expansion of a batch whose block is `block`, with no manifest to read, with whitespace removed.
    fn expand(block: &str) -> String {
        let tokens: TokenStream = format!("\"t\", on(&d); |b| {{ {block} }}").parse().unwrap();
        gpu_batch_in(tokens, None)
            .to_string()
            .split_whitespace()
            .collect()
    }

    fn rejected(block: &str, code: Code) {
        let text = expand(block);
        assert!(text.contains("compile_error"), "{block}\n{text}");
        assert_rendered_code(&text, code);
        assert_eq!(text.matches("compile_error").count(), 1, "{text}");
    }

    #[test]
    fn every_op_lowers_to_the_builder_api_with_its_outputs_named_by_the_let() {
        for (written, lowered) in [
            (
                "let total = gust::sum(&s);",
                r#"b.op(&["total"],::gust_algo::batch::Sum::of(&s))"#,
            ),
            (
                "let total = gust::sum_wrapping(&s);",
                r#"b.op(&["total"],::gust_algo::batch::SumWrapping::of(&s))"#,
            ),
            (
                "let total = gust::min(&s);",
                r#"b.op(&["total"],::gust_algo::batch::Min::of(&s))"#,
            ),
            (
                "let total = gust::max(&s);",
                r#"b.op(&["total"],::gust_algo::batch::Max::of(&s))"#,
            ),
            (
                "let total = gust::count(&s);",
                r#"b.op(&["total"],::gust_algo::batch::Count::of(&s))"#,
            ),
            (
                "let total = gust::scan(&s);",
                r#"b.op(&["total"],::gust_algo::batch::Scan::of(&s))"#,
            ),
            (
                "let total = gust::scan_exclusive(&s);",
                r#"b.op(&["total"],::gust_algo::batch::Scan::exclusive(&s))"#,
            ),
            (
                "let (idx, n) = gust::compact(&s);",
                r#"b.op(&["idx","n"],::gust_algo::batch::Compact::of(&s))"#,
            ),
            (
                "let total = ::gust::sum(&s);",
                r#"b.op(&["total"],::gust_algo::batch::Sum::of(&s))"#,
            ),
        ] {
            let text = expand(&format!("let s = b.input::<u32>(4u32); {written}"));
            assert!(text.contains(lowered), "{written}\n{text}");
            assert!(!text.contains("compile_error"), "{text}");
        }
    }

    #[test]
    fn reduce_hands_the_operation_first() {
        let text = expand("let s = b.input::<f32>(4u32); let total = gust::reduce(&s, Op::Max);");
        assert!(
            text.contains(r#"b.op(&["total"],::gust_algo::batch::Reduce::of(Op::Max,&s))"#),
            "{text}"
        );
    }

    #[test]
    fn an_output_is_a_handle_a_later_stage_captures_with_its_type() {
        let text = expand(
            "let s = b.input::<f32>(4u32); let total = gust::sum(&s); let (idx, n) = gust::compact(&f); \
             let g = gust::gpu_task!(4u32, |i| -> f32 { total[0u32] + (idx[i] + n[0u32]) as f32 });",
        );
        assert!(text.contains("total:Stage<f32>"), "{text}");
        assert!(text.contains("idx:Stage<u32>"), "{text}");
        assert!(text.contains("n:Stage<u32>"), "{text}");
    }

    #[test]
    fn an_output_in_the_tail_is_exposed() {
        let text = expand("let s = b.input::<u32>(4u32); let total = gust::sum(&s); total");
        assert!(text.contains("b.expose(&total);"), "{text}");
    }

    #[test]
    fn a_pattern_that_does_not_fit_the_op_is_e1009() {
        rejected(
            "let s = b.input::<u32>(4u32); let t = gust::compact(&s);",
            Code::BatchOpForm,
        );
        rejected(
            "let s = b.input::<u32>(4u32); let (a, b) = gust::sum(&s);",
            Code::BatchOpForm,
        );
        rejected(
            "let s = b.input::<u32>(4u32); let (a, _) = gust::compact(&s);",
            Code::BatchOpForm,
        );
        rejected(
            "let s = b.input::<u32>(4u32); let (a, a) = gust::compact(&s);",
            Code::BatchOpForm,
        );
        rejected(
            "let s = b.input::<u32>(4u32); let (a, b, c) = gust::compact(&s);",
            Code::BatchOpForm,
        );
        rejected(
            "let s = b.input::<u32>(4u32); let (mut a, b) = gust::compact(&s);",
            Code::BatchOpForm,
        );
        rejected(
            "let s = b.input::<u32>(4u32); let _ = gust::sum(&s);",
            Code::BatchOpForm,
        );
    }

    #[test]
    fn an_op_that_is_not_a_let_init_is_e1009() {
        for block in [
            "let s = b.input::<u32>(4u32); gust::sum(&s);",
            "let s = b.input::<u32>(4u32); let t = f(gust::sum(&s));",
            "let s = b.input::<u32>(4u32); let t = gust::sum(&s) + 1;",
            "let s = b.input::<u32>(4u32); gust::sum(&s)",
        ] {
            rejected(block, Code::BatchOpForm);
        }
    }

    #[test]
    fn the_wrong_number_of_arguments_is_e1009() {
        rejected(
            "let s = b.input::<u32>(4u32); let t = gust::sum(&s, &s);",
            Code::BatchOpForm,
        );
        rejected(
            "let s = b.input::<u32>(4u32); let t = gust::sum();",
            Code::BatchOpForm,
        );
        rejected(
            "let s = b.input::<u32>(4u32); let t = gust::reduce(&s);",
            Code::BatchOpForm,
        );
    }

    #[test]
    fn an_op_in_a_closure_or_a_loop_is_e1009() {
        rejected(
            "let s = b.input::<u32>(4u32); let f = || { let t = gust::sum(&s); }; f();",
            Code::BatchOpForm,
        );
        rejected(
            "let s = b.input::<u32>(4u32); for _ in 0..2 { let t = gust::sum(&s); }",
            Code::BatchOpForm,
        );
    }

    #[test]
    fn a_name_that_is_not_an_op_is_left_alone() {
        let text = expand(
            "let s = b.input::<u32>(4u32); let t = gust::other(&s); let u = other::sum(&s);",
        );
        assert!(!text.contains("compile_error"), "{text}");
        assert!(!text.contains("gust_algo"), "{text}");
    }

    #[test]
    fn the_manifest_is_read_for_a_dependency_on_gust_algo() {
        let block = "let s = b.input::<u32>(4u32); let total = gust::sum(&s);";
        let run = |manifest_text: &str| {
            let krate = Crate::new(manifest_text);
            let tokens: TokenStream = format!("\"t\", on(&d); |b| {{ {block} }}").parse().unwrap();
            gpu_batch_in(tokens, Some(&krate.0)).to_string()
        };
        let without = run("[package]\nname = \"x\"\n\n[dependencies]\ngust = \"1\"\n");
        assert!(without.contains("compile_error"), "{without}");
        assert_rendered_code(&without, Code::BatchOpCrate);
        assert_eq!(without.matches("compile_error").count(), 1, "{without}");
        for with in [
            "[dependencies]\ngust-algo = \"1\"\n",
            "[dev-dependencies]\ngust-algo = { path = \"../x\" }\n",
            "[dependencies]\ngust_algo = \"1\"\n",
            "[target.'cfg(unix)'.dependencies]\ngust-algo = \"1\"\n",
        ] {
            let text = run(with);
            assert!(!text.contains("compile_error"), "{with}\n{text}");
        }
        // A crate that uses no op needs no dependency, and a manifest that cannot be read leaves it to rustc.
        let krate = Crate::new("[dependencies]\n");
        let tokens: TokenStream = "\"t\", on(&d); |b| { let s = b.input::<u32>(4u32); }"
            .parse()
            .unwrap();
        assert!(
            !gpu_batch_in(tokens, Some(&krate.0))
                .to_string()
                .contains("compile_error")
        );
        let tokens: TokenStream = format!("\"t\", on(&d); |b| {{ {block} }}").parse().unwrap();
        assert!(
            !gpu_batch_in(tokens, Some(&krate.0.join("missing")))
                .to_string()
                .contains("compile_error")
        );
    }

    #[test]
    fn the_facade_option_leaves_the_dependency_to_rustc() {
        let krate = Crate::new("[dependencies]\n");
        let tokens: TokenStream =
            "\"t\", on(&d); crate(::inner::gust); |b| { let s = b.input::<u32>(4u32); let total = gust::sum(&s); }"
                .parse()
                .unwrap();
        let text = gpu_batch_in(tokens, Some(&krate.0)).to_string();
        assert!(!text.contains("compile_error"), "{text}");
    }

    #[test]
    fn the_dependency_reader_looks_at_the_tables_cargo_reads() {
        for (text, found) in [
            ("[dependencies]\ngust-algo = \"1\"", Some(true)),
            ("[dependencies.gust-algo]\nversion = \"1\"", Some(true)),
            (
                "[dev-dependencies]\ngust_algo = { path = \"p\" }",
                Some(true),
            ),
            ("[build-dependencies]\ngust-algo = \"1\"", Some(false)),
            ("[workspace.dependencies]\ngust-algo = \"1\"", Some(false)),
            ("[dependencies]\ngust = \"1\"", Some(false)),
            ("[package]\nname = \"gust-algo\"", Some(false)),
            ("not toml [", None),
        ] {
            assert_eq!(manifest::dependency_in(text, "gust-algo"), found, "{text}");
        }
    }
}
