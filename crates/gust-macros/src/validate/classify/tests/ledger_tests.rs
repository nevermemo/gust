//! The ledger of accepted constructs (D94: anything unrecognised is `grid`-only).
//!
//! The classifier admits `pixel` when no rule excluded the kernel, which is safe only while the validator and the
//! emitter accept nothing the rules have not been written for. These tests tie the two together: one probe program per
//! kind of `syn`, accepted or refused by the validator and the emitter as the tables below say, and the ledger in
//! `kinds.rs` lists exactly the accepted kinds. A change to what a kernel body may hold fails here, and the fix is a
//! decision about the classifier, written into the ledger.

use super::*;
use crate::validate::classify::kinds::{self, Class, Handling, LEDGER};
use crate::{expand::kernel_options, slang::emit_kernel};

const FIX: &str = "decide what the shape classifier does with it: add or remove its entry in \
                   validate/classify/kinds.rs, and update this probe (D94)";

/// Whether a module is accepted by the validator and lowered by the emitter, which is what a kernel needs to reach
/// the classifier's `pixel`.
fn accepted(source: &str) -> bool {
    let Ok(module) = parse_str::<syn::ItemMod>(source) else {
        return false;
    };
    if validate_module(&module).is_err() {
        return false;
    }
    let kernel = kernel_of(&module);
    let Ok(Some(options)) = kernel_options(kernel) else {
        return false;
    };
    emit_kernel(&module, kernel, options).is_ok()
}

/// A module with a helper `helper(v: f32) -> f32`, a struct `P { x: f32 }` and the kernel `k`.
fn probe_module(parameters: &str, body: &str) -> String {
    format!(
        "mod m {{ pub struct P {{ pub x: f32 }} fn helper(v: f32) -> f32 {{ v }} \
         #[kernel] pub fn k({parameters}) {{ {body} }} }}"
    )
}

/// The name of the variant of a `syn` value, from its `Debug` text (`Expr::Binary(ExprBinary {..})` is `Binary`).
fn variant_of(debug: &str) -> &str {
    let end = debug
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == ':'))
        .unwrap_or(debug.len());
    debug[..end].rsplit("::").next().unwrap_or("")
}

const ATOMIC: &str = "id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<u32>, mut out: RWStructuredBuffer<f32>";

/// Every variant of `syn::Expr` in the `syn` this crate builds with. A variant that `syn` adds later is not matched by
/// `kinds::expression_kind`, so a kernel that holds one is `grid`-only whatever happens here.
const EXPR_VARIANTS: [&str; 40] = [
    "Array",
    "Assign",
    "Async",
    "Await",
    "Binary",
    "Block",
    "Break",
    "Call",
    "Cast",
    "Closure",
    "Const",
    "Continue",
    "Field",
    "ForLoop",
    "Group",
    "If",
    "Index",
    "Infer",
    "Let",
    "Lit",
    "Loop",
    "Macro",
    "Match",
    "MethodCall",
    "Paren",
    "Path",
    "Range",
    "RawAddr",
    "Reference",
    "Repeat",
    "Return",
    "Struct",
    "Try",
    "TryBlock",
    "Tuple",
    "Unary",
    "Unsafe",
    "Verbatim",
    "While",
    "Yield",
];

/// (kind, an expression of that kind, the kernel parameters, a body that holds it, accepted).
#[rustfmt::skip]
const EXPRESSIONS: &[(&str, &str, &str, &str, bool)] = &[
    ("Array", "[1.0f32, 2.0f32]", PARAMETERS, "let t = [1.0f32, 2.0f32]; out[id.x] = t[0u32];", true),
    ("Assign", "out[id.x] = 1.0f32", PARAMETERS, "out[id.x] = 1.0f32;", true),
    ("Binary", "a[id.x] * 2.0f32", PARAMETERS, "out[id.x] = a[id.x] * 2.0f32;", true),
    ("Block", "{ out[id.x] = 1.0f32; }", PARAMETERS, "{ out[id.x] = 1.0f32; }", true),
    ("Break", "break", PARAMETERS, "for i in 0u32..4u32 { break; } out[id.x] = 1.0f32;", true),
    ("Call", "helper(1.0f32)", PARAMETERS, "out[id.x] = helper(1.0f32);", true),
    ("Cast", "id.x as f32", PARAMETERS, "out[id.x] = id.x as f32;", true),
    ("Continue", "continue", PARAMETERS, "for i in 0u32..4u32 { continue; } out[id.x] = 1.0f32;", true),
    ("Field", "id.x", PARAMETERS, "out[id.x] = 1.0f32;", true),
    ("ForLoop", "for i in 0u32..4u32 { }", PARAMETERS, "for i in 0u32..4u32 { } out[id.x] = 1.0f32;", true),
    ("If", "if a[id.x] > 0.0f32 { out[id.x] = 1.0f32; } else { out[id.x] = 2.0f32; }", PARAMETERS,
     "if a[id.x] > 0.0f32 { out[id.x] = 1.0f32; } else { out[id.x] = 2.0f32; }", true),
    ("Index", "a[id.x]", PARAMETERS, "out[id.x] = a[id.x];", true),
    ("Let", "let Some(v) = Some(1.0f32)", PARAMETERS,
     "if let Some(v) = Some(1.0f32) { out[id.x] = v; } else { out[id.x] = 2.0f32; }", true),
    ("Lit", "1.0f32", PARAMETERS, "out[id.x] = 1.0f32;", true),
    ("Match", "match r { Ok(x) => { out[id.x] = x as f32; } Err(e) => { out[id.x] = e as f32; } }", PARAMETERS,
     "let r: Result<u32, u32> = Ok(1u32); \
      match r { Ok(x) => { out[id.x] = x as f32; } Err(e) => { out[id.x] = e as f32; } };", true),
    ("MethodCall", "id.x.wrapping_add(1u32)", PARAMETERS,
     "let n = id.x.wrapping_add(1u32); out[id.x] = n as f32;", true),
    ("Paren", "(a[id.x])", PARAMETERS, "out[id.x] = (a[id.x]);", true),
    ("Path", "id", PARAMETERS, "out[id.x] = 1.0f32;", true),
    ("Range", "0u32..4u32", PARAMETERS, "for i in 0u32..4u32 { } out[id.x] = 1.0f32;", true),
    ("Reference", "&mut counter[0]", ATOMIC, "atomic_add(&mut counter[0], 1u32); out[id.x] = 1.0f32;", true),
    ("Return", "return", PARAMETERS, "out[id.x] = 1.0f32; return;", true),
    ("Struct", "P { x: 1.0f32 }", PARAMETERS, "let p = P { x: 1.0f32 }; out[id.x] = p.x;", true),
    ("Unary", "-a[id.x]", PARAMETERS, "out[id.x] = -a[id.x];", true),
    // ---- refused today ----
    ("Async", "async { }", PARAMETERS, "let t = async { }; out[id.x] = 1.0f32;", false),
    ("Await", "id.x.await", PARAMETERS, "let t = id.x.await; out[id.x] = 1.0f32;", false),
    ("Closure", "|v: f32| v", PARAMETERS, "let f = |v: f32| v; out[id.x] = 1.0f32;", false),
    ("Const", "const { 1u32 }", PARAMETERS, "let t = const { 1u32 }; out[id.x] = 1.0f32;", false),
    ("Infer", "_", PARAMETERS, "let t = _; out[id.x] = 1.0f32;", false),
    ("Loop", "loop { }", PARAMETERS, "loop { } out[id.x] = 1.0f32;", false),
    ("Macro", "foo!()", PARAMETERS, "let t = foo!(); out[id.x] = 1.0f32;", false),
    ("RawAddr", "&raw const id", PARAMETERS, "let t = &raw const id; out[id.x] = 1.0f32;", false),
    ("Repeat", "[0.0f32; 4]", PARAMETERS, "let t = [0.0f32; 4]; out[id.x] = t[0u32];", false),
    ("Try", "id.x?", PARAMETERS, "let t = id.x?; out[id.x] = 1.0f32;", false),
    ("TryBlock", "try { 1u32 }", PARAMETERS, "let t = try { 1u32 }; out[id.x] = 1.0f32;", false),
    ("Tuple", "(1.0f32, 2.0f32)", PARAMETERS, "let t = (1.0f32, 2.0f32); out[id.x] = 1.0f32;", false),
    ("Unsafe", "unsafe { 1u32 }", PARAMETERS, "let t = unsafe { 1u32 }; out[id.x] = 1.0f32;", false),
    ("While", "while true { }", PARAMETERS, "while true { } out[id.x] = 1.0f32;", false),
    ("Yield", "yield 1u32", PARAMETERS, "let t = yield 1u32; out[id.x] = 1.0f32;", false),
];

/// Kinds that have no spelling in source: `Group` comes from a `macro_rules!` expansion (the validator and the emitter
/// both have an arm for it, so the ledger lists it) and `Verbatim` is what `syn` keeps for syntax it cannot represent
/// (nothing accepts it).
const NO_SOURCE_FORM: [(&str, bool); 2] = [("Group", true), ("Verbatim", false)];

#[test]
fn the_ledger_lists_exactly_the_expression_kinds_the_validator_and_the_emitter_accept() {
    let mut covered: Vec<&str> = EXPRESSIONS.iter().map(|probe| probe.0).collect();
    covered.extend(NO_SOURCE_FORM.iter().map(|probe| probe.0));
    covered.sort_unstable();
    let mut variants = EXPR_VARIANTS.to_vec();
    variants.sort_unstable();
    assert_eq!(
        covered, variants,
        "every variant of `syn::Expr` has one probe"
    );

    for &(kind, expression, parameters, body, expected) in EXPRESSIONS {
        let parsed: syn::Expr = parse_str(expression).unwrap_or_else(|e| panic!("{kind}: {e}"));
        assert_eq!(
            variant_of(&format!("{parsed:?}")),
            kind,
            "`{expression}` is not a {kind}"
        );
        assert_eq!(
            kinds::expression_kind(&parsed),
            expected.then_some(kind),
            "{kind}: {FIX}"
        );
        let source = probe_module(parameters, body);
        assert_eq!(
            accepted(&source),
            expected,
            "the validator and the emitter now {} a body with a `{kind}`; {FIX}\n{source}",
            if expected { "refuse" } else { "accept" }
        );
        assert_eq!(
            kinds::lookup(Class::Expression, kind).is_some(),
            expected,
            "{kind}: the ledger and the accepted set differ; {FIX}"
        );
    }
    for (kind, listed) in NO_SOURCE_FORM {
        assert_eq!(
            kinds::lookup(Class::Expression, kind).is_some(),
            listed,
            "{kind}: {FIX}"
        );
    }
}

/// Every variant of `syn::Stmt`.
const STATEMENT_VARIANTS: [&str; 4] = ["Local", "Item", "Expr", "Macro"];

/// (kind, a statement of that kind, accepted).
const STATEMENTS: &[(&str, &str, bool)] = &[
    ("Local", "let t = 1.0f32;", true),
    ("Expr", "out[id.x] = 1.0f32;", true),
    ("Item", "fn inner() {}", false),
    ("Macro", "foo!();", false),
];

#[test]
fn the_ledger_lists_exactly_the_statement_kinds_the_validator_and_the_emitter_accept() {
    let mut covered: Vec<&str> = STATEMENTS.iter().map(|probe| probe.0).collect();
    covered.sort_unstable();
    let mut variants = STATEMENT_VARIANTS.to_vec();
    variants.sort_unstable();
    assert_eq!(covered, variants);
    for &(kind, statement, expected) in STATEMENTS {
        let block: syn::Block = parse_str(&format!("{{ {statement} }}")).unwrap();
        let parsed = &block.stmts[0];
        assert_eq!(
            variant_of(&format!("{parsed:?}")),
            kind,
            "`{statement}` is not a {kind}"
        );
        assert_eq!(
            kinds::statement_kind(parsed),
            expected.then_some(kind),
            "{kind}: {FIX}"
        );
        let body = if kind == "Expr" {
            statement.to_owned()
        } else {
            format!("{statement} out[id.x] = 1.0f32;")
        };
        let source = probe_module(PARAMETERS, &body);
        assert_eq!(
            accepted(&source),
            expected,
            "a body with a `{kind}` statement; {FIX}\n{source}"
        );
        assert_eq!(
            kinds::lookup(Class::Statement, kind).is_some(),
            expected,
            "{kind}: {FIX}"
        );
    }
}

/// Every variant of `syn::Pat`.
const PATTERN_VARIANTS: [&str; 17] = [
    "Const",
    "Ident",
    "Lit",
    "Macro",
    "Or",
    "Paren",
    "Path",
    "Range",
    "Reference",
    "Rest",
    "Slice",
    "Struct",
    "Tuple",
    "TupleStruct",
    "Type",
    "Verbatim",
    "Wild",
];

/// (kind, a body in which the first pattern is of that kind, accepted).
#[rustfmt::skip]
const PATTERNS: &[(&str, &str, bool)] = &[
    ("Ident", "let t = 1.0f32; out[id.x] = t;", true),
    ("Type", "let t: f32 = 1.0f32; out[id.x] = t;", true),
    ("TupleStruct", "if let Some(v) = Some(1.0f32) { out[id.x] = v; } else { out[id.x] = 2.0f32; }", true),
    ("Wild", "for _ in 0u32..4u32 { } out[id.x] = 1.0f32;", true),
    ("Const", "let const { 1u32 } = id.x; out[id.x] = 1.0f32;", false),
    ("Lit", "let 1u32 = id.x; out[id.x] = 1.0f32;", false),
    ("Macro", "let foo!() = id.x; out[id.x] = 1.0f32;", false),
    ("Or", "if let 1u32 | 2u32 = id.x { } out[id.x] = 1.0f32;", false),
    ("Paren", "let (t) = id.x; out[id.x] = 1.0f32;", false),
    ("Path", "let Foo::Bar = id.x; out[id.x] = 1.0f32;", false),
    ("Range", "let 0u32..=1u32 = id.x; out[id.x] = 1.0f32;", false),
    ("Reference", "let &t = id.x; out[id.x] = 1.0f32;", false),
    ("Rest", "let .. = id.x; out[id.x] = 1.0f32;", false),
    ("Slice", "let [t] = id.x; out[id.x] = 1.0f32;", false),
    ("Struct", "let P { x } = id.x; out[id.x] = 1.0f32;", false),
    ("Tuple", "let (s, t) = id.x; out[id.x] = 1.0f32;", false),
];

/// `Verbatim` has no source form, like the expression one.
const PATTERN_NO_SOURCE_FORM: [&str; 1] = ["Verbatim"];

/// The first pattern of `block` in source order.
fn first_pattern(block: &syn::Block) -> syn::Pat {
    struct First(Option<syn::Pat>);
    impl<'ast> syn::visit::Visit<'ast> for First {
        fn visit_pat(&mut self, pattern: &'ast syn::Pat) {
            if self.0.is_none() {
                self.0 = Some(pattern.clone());
            }
        }
    }
    let mut first = First(None);
    syn::visit::Visit::visit_block(&mut first, block);
    first.0.expect("the body holds a pattern")
}

#[test]
fn the_ledger_lists_exactly_the_pattern_kinds_the_validator_and_the_emitter_accept() {
    let mut covered: Vec<&str> = PATTERNS.iter().map(|probe| probe.0).collect();
    covered.extend(PATTERN_NO_SOURCE_FORM);
    covered.sort_unstable();
    let mut variants = PATTERN_VARIANTS.to_vec();
    variants.sort_unstable();
    assert_eq!(
        covered, variants,
        "every variant of `syn::Pat` has one probe"
    );
    let unparsed: Vec<String> = PATTERNS
        .iter()
        .filter_map(|&(kind, body, _)| {
            parse_str::<syn::Block>(&format!("{{ {body} }}"))
                .err()
                .map(|e| format!("{kind}: {e}"))
        })
        .collect();
    assert!(unparsed.is_empty(), "{unparsed:?}");
    for &(kind, body, expected) in PATTERNS {
        let block: syn::Block = parse_str(&format!("{{ {body} }}")).unwrap();
        let parsed = first_pattern(&block);
        assert_eq!(variant_of(&format!("{parsed:?}")), kind, "`{body}`");
        assert_eq!(
            kinds::pattern_kind(&parsed),
            expected.then_some(kind),
            "{kind}: {FIX}"
        );
        let source = probe_module(PARAMETERS, body);
        assert_eq!(
            accepted(&source),
            expected,
            "a body with a `{kind}` pattern; {FIX}\n{source}"
        );
        assert_eq!(
            kinds::lookup(Class::Pattern, kind).is_some(),
            expected,
            "{kind}: {FIX}"
        );
    }
    assert!(kinds::lookup(Class::Pattern, "Verbatim").is_none());
}

/// The quoted names between `anchor` and the first `end` after it in `source`.
fn names_between(source: &str, anchor: &str, end: &str) -> Vec<String> {
    let start = source
        .find(anchor)
        .unwrap_or_else(|| panic!("no `{anchor}`"));
    let rest = &source[start + anchor.len()..];
    let stop = rest
        .find(end)
        .unwrap_or_else(|| panic!("no `{end}` after `{anchor}`"));
    let names: Vec<String> = rest[..stop]
        .split('"')
        .skip(1)
        .step_by(2)
        .map(str::to_owned)
        .collect();
    assert!(!names.is_empty(), "`{anchor}` names nothing");
    names
}

#[test]
fn the_ledger_lists_exactly_the_built_in_methods_of_the_dialect() {
    let validate = include_str!("../../mod.rs");
    let math = include_str!("../../../math_types.rs");
    let texture = include_str!("../../../slang/texture.rs");
    let mut dialect: Vec<String> = Vec::new();
    dialect.extend(names_between(
        validate,
        "let in_dialect = matches!(",
        ") || wrapping_arity",
    ));
    dialect.extend(names_between(
        validate,
        "pub(crate) fn wrapping_arity",
        "_ =>",
    ));
    dialect.extend(names_between(math, "pub(crate) fn method_arity", "_ =>"));
    dialect.extend(names_between(texture, "pub(crate) fn from_name", "_ =>"));
    // `unwrap` and `expect` are in the validator's list so that it refuses them with their own code (E0304).
    dialect.retain(|name| name != "unwrap" && name != "expect");
    dialect.sort_unstable();
    dialect.dedup();
    let mut listed: Vec<String> = LEDGER
        .iter()
        .filter(|entry| entry.class == Class::Method && entry.kind != "swizzle")
        .map(|entry| entry.kind.to_owned())
        .collect();
    listed.sort_unstable();
    assert_eq!(
        dialect, listed,
        "a built-in method was added to or removed from the dialect's tables; {FIX}"
    );
    for refused in ["unwrap", "expect"] {
        assert!(kinds::lookup(Class::Method, refused).is_none(), "{refused}");
    }
    assert_eq!(kinds::method_kind("xyzw"), "swizzle");
    assert_eq!(kinds::method_kind("xy"), "swizzle");
    assert_eq!(kinds::method_kind("abs"), "abs");
}

#[test]
fn the_ledger_has_one_entry_per_kind_and_the_excluded_ones_exclude_pixel() {
    for entry in LEDGER {
        if let Handling::Excluded(why) = entry.handling {
            assert_eq!(
                (entry.class, entry.kind),
                (Class::Expression, "Return"),
                "the only excluded kind"
            );
            assert_excluded(&body("out[id.x] = 1.0f32; return;"), why);
        }
    }
    for (index, entry) in LEDGER.iter().enumerate() {
        assert!(
            LEDGER[index + 1..]
                .iter()
                .all(|other| (other.class, other.kind) != (entry.class, entry.kind)),
            "{entry:?} is listed twice"
        );
    }
}

#[test]
fn the_kinds_the_ledger_calls_neutral_or_inspected_keep_pixel() {
    // One kernel per accepted expression kind that writes the output once on every path: none of them alone costs a
    // kernel `pixel`.
    for &(kind, _, parameters, body, expected) in EXPRESSIONS {
        let Some(entry) = kinds::lookup(Class::Expression, kind) else {
            assert!(!expected);
            continue;
        };
        if matches!(entry.handling, Handling::Excluded(_)) || parameters != PARAMETERS {
            continue;
        }
        let classification = classification(&probe_module(parameters, body));
        assert!(
            classification.allows(Shape::Pixel),
            "{kind}: {:?}\n{body}",
            classification.exclusions
        );
    }
}

// ---- the guard itself ----

/// The classification of a kernel body the validator may not accept, to see what the classifier alone does.
fn classification_without_validator(body: &str) -> Classification {
    let module = module(&probe_module(PARAMETERS, body));
    classify_function(kernel_of(&module))
}

#[test]
fn a_kind_the_ledger_does_not_list_makes_the_kernel_grid_only() {
    for (what, body) in [
        ("a closure", "let f = |v: f32| v; out[id.x] = 1.0f32;"),
        ("a tuple", "let t = (1.0f32, 2.0f32); out[id.x] = 1.0f32;"),
        ("an unlisted method", "out[id.x] = a[id.x].abs();"),
        ("a macro statement", "foo!(); out[id.x] = 1.0f32;"),
        ("a tuple pattern", "let (s, t) = id.x; out[id.x] = 1.0f32;"),
        ("a loop", "loop { } out[id.x] = 1.0f32;"),
    ] {
        let classification = classification_without_validator(body);
        assert_eq!(
            classification.shapes,
            [Shape::Grid],
            "{what}: {:?}",
            classification.exclusions
        );
        let exclusion = classification
            .first_exclusion(Shape::Pixel)
            .unwrap_or_else(|| panic!("{what}: no exclusion"));
        assert_eq!(exclusion.why, Why::Unrecognised, "{what}: {exclusion:?}");
        assert_eq!(exclusion.why.code(), Code::ShapeExcluded);
        assert!(
            exclusion.message("k").contains("D94"),
            "{}",
            exclusion.message("k")
        );
    }
    // Without the stranger the same kernel keeps `pixel`, so the ledger is what cost it the shape.
    assert_both(&body("out[id.x] = a[id.x] * 2.0f32;"));
}
