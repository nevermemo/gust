//! The ledger of accepted constructs (D94: anything unrecognised is `grid`-only).
//!
//! The classifier admits `pixel` when no rule excluded the kernel, which is safe only while the validator and the
//! emitter accept nothing the rules have not been written for. These tests tie the two together: one probe program per
//! kind of `syn`, accepted or refused by the validator and the emitter as the tables below say, and the ledger in
//! `kinds.rs` lists exactly the accepted kinds. A change to what a kernel body may hold fails here, and the fix is a
//! decision about the classifier, written into the ledger.

use super::*;
use crate::diagnostics::testing::codes_of;
use crate::validate::classify::kinds::{self, Class, Handling, LEDGER};
use crate::{expand::kernel_options, slang::emit_kernel};

const FIX: &str = "decide what the shape classifier does with it: add or remove its entry in \
                   validate/classify/kinds.rs, and update this probe (D94)";

/// What the validator and the emitter do with a probe module. A probe that is refused names the code it is refused
/// with, so a probe that stops being refused for its kind (and starts being refused for some other reason) fails too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    /// The validator accepts the module and the emitter lowers its kernel: it can reach the classifier's `pixel`.
    Accepted,
    /// The validator refuses it with this code.
    Refused(Code),
    /// The validator accepts it and the emitter refuses to lower it with this code.
    EmitterRefused(Code),
}
use Verdict::{Accepted, EmitterRefused, Refused};

impl Verdict {
    /// Whether a kernel of this kind reaches the classifier.
    fn is_accepted(self) -> bool {
        self == Accepted
    }
}

/// The verdict on `source`, a module with one kernel. A probe that does not parse is a bug in the probe.
#[track_caller]
fn verdict(source: &str) -> Verdict {
    let module = parse_str::<syn::ItemMod>(source)
        .unwrap_or_else(|error| panic!("the probe does not parse: {error}\n{source}"));
    if let Err(error) = validate_module(&module) {
        return Refused(codes_of(&error)[0]);
    }
    let kernel = kernel_of(&module);
    let options = kernel_options(kernel)
        .expect("the probe's `#[kernel]` attribute is valid")
        .expect("the probe has a `#[kernel]`");
    match emit_kernel(&module, kernel, options) {
        Ok(_) => Accepted,
        Err(error) => EmitterRefused(codes_of(&error)[0]),
    }
}

/// A module with a helper `helper(v: f32) -> f32`, a struct `P { x: f32 }` and the kernel `k`.
fn probe_module(parameters: &str, body: &str) -> String {
    probe_module_with("#[kernel]", parameters, body)
}

/// [`probe_module`] with the kernel attribute `attribute`.
fn probe_module_with(attribute: &str, parameters: &str, body: &str) -> String {
    format!(
        "mod m {{ pub struct P {{ pub x: f32 }} fn helper(v: f32) -> f32 {{ v }} \
         #[derive(PartialEq, Eq)] pub enum Mode {{ A, B }} \
         {attribute} pub fn k({parameters}) {{ {body} }} }}"
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

/// The emitter's refusal of an expression kind it has no arm for.
const UNSUPPORTED: Verdict = EmitterRefused(Code::UnsupportedExpression);

/// (kind, an expression of that kind, the kernel parameters, a body that holds it, the verdict).
#[rustfmt::skip]
const EXPRESSIONS: &[(&str, &str, &str, &str, Verdict)] = &[
    ("Array", "[1.0f32, 2.0f32]", PARAMETERS, "let t = [1.0f32, 2.0f32]; out[id.x] = t[0u32];", Accepted),
    ("Assign", "out[id.x] = 1.0f32", PARAMETERS, "out[id.x] = 1.0f32;", Accepted),
    ("Binary", "a[id.x] * 2.0f32", PARAMETERS, "out[id.x] = a[id.x] * 2.0f32;", Accepted),
    ("Block", "{ out[id.x] = 1.0f32; }", PARAMETERS, "{ out[id.x] = 1.0f32; }", Accepted),
    ("Break", "break", PARAMETERS, "for i in 0u32..4u32 { break; } out[id.x] = 1.0f32;", Accepted),
    ("Call", "helper(1.0f32)", PARAMETERS, "out[id.x] = helper(1.0f32);", Accepted),
    ("Cast", "id.x as f32", PARAMETERS, "out[id.x] = id.x as f32;", Accepted),
    ("Continue", "continue", PARAMETERS, "for i in 0u32..4u32 { continue; } out[id.x] = 1.0f32;", Accepted),
    ("Field", "id.x", PARAMETERS, "out[id.x] = 1.0f32;", Accepted),
    ("ForLoop", "for i in 0u32..4u32 { }", PARAMETERS, "for i in 0u32..4u32 { } out[id.x] = 1.0f32;", Accepted),
    ("If", "if a[id.x] > 0.0f32 { out[id.x] = 1.0f32; } else { out[id.x] = 2.0f32; }", PARAMETERS,
     "if a[id.x] > 0.0f32 { out[id.x] = 1.0f32; } else { out[id.x] = 2.0f32; }", Accepted),
    ("Index", "a[id.x]", PARAMETERS, "out[id.x] = a[id.x];", Accepted),
    ("Let", "let Some(v) = Some(1.0f32)", PARAMETERS,
     "if let Some(v) = Some(1.0f32) { out[id.x] = v; } else { out[id.x] = 2.0f32; }", Accepted),
    ("Lit", "1.0f32", PARAMETERS, "out[id.x] = 1.0f32;", Accepted),
    ("Match", "match r { Ok(x) => { out[id.x] = x as f32; } Err(e) => { out[id.x] = e as f32; } }", PARAMETERS,
     "let r: Result<u32, u32> = Ok(1u32); \
      match r { Ok(x) => { out[id.x] = x as f32; } Err(e) => { out[id.x] = e as f32; } };", Accepted),
    ("MethodCall", "id.x.wrapping_add(1u32)", PARAMETERS,
     "let n = id.x.wrapping_add(1u32); out[id.x] = n as f32;", Accepted),
    ("Paren", "(a[id.x])", PARAMETERS, "out[id.x] = (a[id.x]);", Accepted),
    ("Path", "id", PARAMETERS, "out[id.x] = 1.0f32;", Accepted),
    ("Range", "0u32..4u32", PARAMETERS, "for i in 0u32..4u32 { } out[id.x] = 1.0f32;", Accepted),
    ("Reference", "&mut counter[0]", ATOMIC, "atomic_add(&mut counter[0], 1u32); out[id.x] = 1.0f32;", Accepted),
    ("Return", "return", PARAMETERS, "out[id.x] = 1.0f32; return;", Accepted),
    ("Struct", "P { x: 1.0f32 }", PARAMETERS, "let p = P { x: 1.0f32 }; out[id.x] = p.x;", Accepted),
    ("Unary", "-a[id.x]", PARAMETERS, "out[id.x] = -a[id.x];", Accepted),
    // ---- refused today ----
    ("Async", "async { }", PARAMETERS, "let t = async { }; out[id.x] = 1.0f32;", UNSUPPORTED),
    ("Await", "id.x.await", PARAMETERS, "let t = id.x.await; out[id.x] = 1.0f32;", Refused(Code::Await)),
    ("Closure", "|v: f32| v", PARAMETERS, "let f = |v: f32| v; out[id.x] = 1.0f32;", Refused(Code::Closure)),
    ("Const", "const { 1u32 }", PARAMETERS, "let t = const { 1u32 }; out[id.x] = 1.0f32;", Refused(Code::ConstBlock)),
    ("Infer", "_", PARAMETERS, "let t = _; out[id.x] = 1.0f32;", UNSUPPORTED),
    ("Loop", "loop { }", PARAMETERS, "loop { } out[id.x] = 1.0f32;", Refused(Code::LoopNotSupported)),
    ("Macro", "foo!()", PARAMETERS, "let t = foo!(); out[id.x] = 1.0f32;", Refused(Code::MacroInModule)),
    ("RawAddr", "&raw const id", PARAMETERS, "let t = &raw const id; out[id.x] = 1.0f32;", UNSUPPORTED),
    ("Repeat", "[0.0f32; 4]", PARAMETERS, "let t = [0.0f32; 4]; out[id.x] = t[0u32];", UNSUPPORTED),
    ("Try", "id.x?", PARAMETERS, "let t = id.x?; out[id.x] = 1.0f32;", Refused(Code::TryOperator)),
    ("TryBlock", "try { 1u32 }", PARAMETERS, "let t = try { 1u32 }; out[id.x] = 1.0f32;", UNSUPPORTED),
    ("Tuple", "(1.0f32, 2.0f32)", PARAMETERS, "let t = (1.0f32, 2.0f32); out[id.x] = 1.0f32;", UNSUPPORTED),
    ("Unsafe", "unsafe { 1u32 }", PARAMETERS, "let t = unsafe { 1u32 }; out[id.x] = 1.0f32;",
     Refused(Code::UnsafeBlock)),
    ("While", "while true { }", PARAMETERS, "while true { } out[id.x] = 1.0f32;", Refused(Code::WhileNotSupported)),
    ("Yield", "yield 1u32", PARAMETERS, "let t = yield 1u32; out[id.x] = 1.0f32;", UNSUPPORTED),
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
        let accepted = expected == Accepted;
        let parsed: syn::Expr = parse_str(expression).unwrap_or_else(|e| panic!("{kind}: {e}"));
        assert_eq!(
            variant_of(&format!("{parsed:?}")),
            kind,
            "`{expression}` is not a {kind}"
        );
        assert_eq!(
            kinds::expression_kind(&parsed),
            accepted.then_some(kind),
            "{kind}: {FIX}"
        );
        let source = probe_module(parameters, body);
        assert_eq!(
            verdict(&source),
            expected,
            "the validator and the emitter now treat a body with a `{kind}` differently; {FIX}\n{source}"
        );
        assert_eq!(
            kinds::lookup(Class::Expression, kind).is_some(),
            accepted,
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

/// (kind, a statement of that kind, the verdict).
const STATEMENTS: &[(&str, &str, Verdict)] = &[
    ("Local", "let t = 1.0f32;", Accepted),
    ("Expr", "out[id.x] = 1.0f32;", Accepted),
    ("Item", "fn inner() {}", EmitterRefused(Code::NestedItem)),
    ("Macro", "foo!();", EmitterRefused(Code::MacroStatement)),
];

#[test]
fn the_ledger_lists_exactly_the_statement_kinds_the_validator_and_the_emitter_accept() {
    let mut covered: Vec<&str> = STATEMENTS.iter().map(|probe| probe.0).collect();
    covered.sort_unstable();
    let mut variants = STATEMENT_VARIANTS.to_vec();
    variants.sort_unstable();
    assert_eq!(covered, variants);
    for &(kind, statement, expected) in STATEMENTS {
        let accepted = expected == Accepted;
        let block: syn::Block = parse_str(&format!("{{ {statement} }}")).unwrap();
        let parsed = &block.stmts[0];
        assert_eq!(
            variant_of(&format!("{parsed:?}")),
            kind,
            "`{statement}` is not a {kind}"
        );
        assert_eq!(
            kinds::statement_kind(parsed),
            accepted.then_some(kind),
            "{kind}: {FIX}"
        );
        let body = if kind == "Expr" {
            statement.to_owned()
        } else {
            format!("{statement} out[id.x] = 1.0f32;")
        };
        let source = probe_module(PARAMETERS, &body);
        assert_eq!(
            verdict(&source),
            expected,
            "a body with a `{kind}` statement; {FIX}\n{source}"
        );
        assert_eq!(
            kinds::lookup(Class::Statement, kind).is_some(),
            accepted,
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

/// (kind, a body in which the first pattern is of that kind): the patterns the ledger lists, each in a position that
/// takes it.
#[rustfmt::skip]
const ACCEPTED_PATTERNS: &[(&str, &str)] = &[
    ("Ident", "let t = 1.0f32; out[id.x] = t;"),
    ("Type", "let t: f32 = 1.0f32; out[id.x] = t;"),
    ("TupleStruct", "if let Some(v) = Some(1.0f32) { out[id.x] = v; } else { out[id.x] = 2.0f32; }"),
    ("Wild", "for _ in 0u32..4u32 { } out[id.x] = 1.0f32;"),
];

/// (kind, a body with a `match` whose arms are of that kind): the patterns of an enum or integer `match` arm (m9.e1, m9.e2;
/// the value form `let t: f32 = match ..;` is probed in `a_value_match_is_a_let_initializer_the_ledger_lists`). The ledger
/// lists them, and the `let`, `for` and `if let` positions below still refuse them.
#[rustfmt::skip]
const ARM_ONLY_PATTERNS: &[(&str, &str)] = &[
    ("Path", "match Mode::A { Mode::A => { out[id.x] = 1.0f32; } _ => { out[id.x] = 2.0f32; } }"),
    ("Or", "match Mode::A { Mode::A | Mode::B => { out[id.x] = 1.0f32; } }"),
    ("Lit", "match id.x { 0u32 | 1u32 => { out[id.x] = 1.0f32; } _ => { out[id.x] = 2.0f32; } }"),
];

/// (kind, a pattern of that kind): the kinds the validator refuses. Each is tried in every position where the
/// validator takes a pattern (below), so a kind that one position starts to accept is noticed there. The kinds of
/// [`ARM_ONLY_PATTERNS`] are in the list for the positions that still refuse them.
const REFUSED_PATTERNS: &[(&str, &str)] = &[
    ("Const", "const { 1u32 }"),
    ("Lit", "1u32"),
    ("Macro", "foo!()"),
    ("Or", "1u32 | 2u32"),
    ("Paren", "(t)"),
    ("Path", "Foo::Bar"),
    ("Range", "0u32..=1u32"),
    ("Reference", "&t"),
    ("Rest", ".."),
    ("Slice", "[t]"),
    ("Struct", "P { x }"),
    ("Tuple", "(s, t)"),
];

/// (position, a body with `{}` where the pattern goes, the code the validator refuses any pattern outside the
/// listed kinds with there). A `match` arm is the fourth position, and it takes `Ok(identifier)` and `Err(identifier)`
/// only (`result_match_arms`), which are the `TupleStruct` of the table above.
const PATTERN_POSITIONS: [(&str, &str, Code); 3] = [
    (
        "let",
        "let {} = id.x; out[id.x] = 1.0f32;",
        Code::LetNeedsIdentifier,
    ),
    (
        "for",
        "for {} in 0u32..4u32 { } out[id.x] = 1.0f32;",
        Code::LoopVariablePattern,
    ),
    (
        "if let",
        "if let Some({}) = Some(1.0f32) { out[id.x] = 1.0f32; } else { out[id.x] = 2.0f32; }",
        Code::IfLetPattern,
    ),
];

/// (kind, position) pairs whose source does not parse, so there is nothing to refuse: a `let` takes no top-level
/// `|`, and `..` is not a `for` pattern.
const UNPARSEABLE_PATTERNS: [(&str, &str); 2] = [("Or", "let"), ("Rest", "for")];

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
    let mut covered: Vec<&str> = ACCEPTED_PATTERNS.iter().map(|probe| probe.0).collect();
    covered.extend(REFUSED_PATTERNS.iter().map(|probe| probe.0));
    covered.extend(PATTERN_NO_SOURCE_FORM);
    covered.sort_unstable();
    let mut variants = PATTERN_VARIANTS.to_vec();
    variants.sort_unstable();
    assert_eq!(
        covered, variants,
        "every variant of `syn::Pat` has one probe"
    );

    for &(kind, body) in ACCEPTED_PATTERNS {
        let block: syn::Block =
            parse_str(&format!("{{ {body} }}")).unwrap_or_else(|error| panic!("{kind}: {error}"));
        let parsed = first_pattern(&block);
        assert_eq!(variant_of(&format!("{parsed:?}")), kind, "`{body}`");
        assert_eq!(kinds::pattern_kind(&parsed), Some(kind), "{kind}: {FIX}");
        let source = probe_module(PARAMETERS, body);
        assert_eq!(
            verdict(&source),
            Accepted,
            "a body with a `{kind}` pattern; {FIX}\n{source}"
        );
        assert!(
            kinds::lookup(Class::Pattern, kind).is_some(),
            "{kind}: {FIX}"
        );
    }

    for &(kind, body) in ARM_ONLY_PATTERNS {
        let source = probe_module(PARAMETERS, body);
        assert_eq!(
            verdict(&source),
            Accepted,
            "an enum `match` arm of kind `{kind}`; {FIX}\n{source}"
        );
    }

    for &(kind, pattern) in REFUSED_PATTERNS {
        // A match arm takes every pattern (a top-level `|` and `..` included).
        let arm: syn::Expr = parse_str(&format!("match x {{ {pattern} => () }}"))
            .unwrap_or_else(|error| panic!("{kind}: {error}"));
        let syn::Expr::Match(arm) = arm else {
            unreachable!()
        };
        let parsed = arm.arms[0].pat.clone();
        assert_eq!(variant_of(&format!("{parsed:?}")), kind, "`{pattern}`");
        let arm_only = ARM_ONLY_PATTERNS.iter().any(|probe| probe.0 == kind);
        if arm_only {
            assert_eq!(kinds::pattern_kind(&parsed), Some(kind), "{kind}: {FIX}");
            assert!(
                kinds::lookup(Class::Pattern, kind).is_some(),
                "{kind}: {FIX}"
            );
        } else {
            assert_eq!(kinds::pattern_kind(&parsed), None, "{kind}: {FIX}");
            assert!(
                kinds::lookup(Class::Pattern, kind).is_none(),
                "{kind}: {FIX}"
            );
        }
        for (position, template, code) in PATTERN_POSITIONS {
            let source = probe_module(PARAMETERS, &template.replace("{}", pattern));
            if UNPARSEABLE_PATTERNS.contains(&(kind, position)) {
                assert!(
                    parse_str::<syn::ItemMod>(&source).is_err(),
                    "{kind} in a `{position}` now parses; give it a probe\n{source}"
                );
                continue;
            }
            assert_eq!(
                verdict(&source),
                Refused(code),
                "the validator now treats a `{kind}` pattern in a `{position}` differently; {FIX}\n{source}"
            );
        }
    }
    assert!(kinds::lookup(Class::Pattern, "Verbatim").is_none());
}

/// The value form of `match` (m9.e2) adds no `syn` kind: it is the `Match` expression the ledger lists, as the
/// initializer of a typed `Local`. The validator takes it there and nowhere else, and the classifier's definite-write
/// walk reads its arms like the statement form's.
#[test]
fn a_value_match_is_a_let_initializer_the_ledger_lists() {
    assert!(kinds::lookup(Class::Expression, "Match").is_some());
    assert!(kinds::lookup(Class::Statement, "Local").is_some());
    for body in [
        "let t: f32 = match id.x { 0u32 => 1.0f32, 1u32 | 2u32 => { 3.0f32 } _ => 2.0f32 }; out[id.x] = t;",
        "let t: f32 = match Mode::A { Mode::A => 1.0f32, _ => { let u = 2.0f32; match id.x { 0 => u, _ => 4.0f32 } } }; out[id.x] = t;",
        // Writes in the arms are definite when every arm writes, as for a statement `match`.
        "let _t: u32 = match id.x { 0u32 => { out[id.x] = 1.0f32; 1u32 } _ => { out[id.x] = 2.0f32; 2u32 } };",
    ] {
        let source = probe_module(PARAMETERS, body);
        assert_eq!(verdict(&source), Accepted, "{source}");
    }
    for (body, code) in [
        (
            "let t = match id.x { 0u32 => 1.0f32, _ => 2.0f32 }; out[id.x] = t;",
            Code::ValueMatchForm,
        ),
        (
            "out[id.x] = match id.x { 0u32 => 1.0f32, _ => 2.0f32 };",
            Code::ValueMatchForm,
        ),
        (
            "let t: f32 = helper(match id.x { 0u32 => 1.0f32, _ => 2.0f32 }); out[id.x] = t;",
            Code::ValueMatchForm,
        ),
    ] {
        let source = probe_module(PARAMETERS, body);
        assert_eq!(verdict(&source), Refused(code), "{source}");
    }
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
            assert!(!expected.is_accepted());
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
    classify_function(kernel_of(&module), &crate::validate::helper_names(&module))
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

// ---- the callees ----

/// (kind, the kernel attribute, the parameters, a body with a call of that form): every callee form that
/// `visit_expr_call` in the validator accepts.
#[rustfmt::skip]
const CALLEES: &[(&str, &str, &str, &str)] = &[
    ("helper", "#[kernel]", PARAMETERS, "out[id.x] = helper(1.0f32);"),
    ("Some", "#[kernel]", PARAMETERS,
     "if let Some(v) = Some(1.0f32) { out[id.x] = v; } else { out[id.x] = 2.0f32; }"),
    ("Ok", "#[kernel]", PARAMETERS,
     "let r: Result<u32, u32> = Ok(1u32); \
      match r { Ok(x) => { out[id.x] = x as f32; } Err(e) => { out[id.x] = e as f32; } };"),
    ("Err", "#[kernel]", PARAMETERS,
     "let r: Result<u32, u32> = Err(1u32); \
      match r { Ok(x) => { out[id.x] = x as f32; } Err(e) => { out[id.x] = e as f32; } };"),
    ("workgroup_barrier", "#[kernel(cpu = \"unsupported\")]", PARAMETERS,
     "workgroup_barrier(); out[id.x] = 1.0f32;"),
    ("atomic_add", "#[kernel]", ATOMIC, "atomic_add(&mut counter[0], 1u32); out[id.x] = 1.0f32;"),
    ("atomic_min", "#[kernel]", ATOMIC, "atomic_min(&mut counter[0], 1u32); out[id.x] = 1.0f32;"),
    ("atomic_max", "#[kernel]", ATOMIC, "atomic_max(&mut counter[0], 1u32); out[id.x] = 1.0f32;"),
    ("atomic_exchange", "#[kernel]", ATOMIC,
     "atomic_exchange(&mut counter[0], 1u32); out[id.x] = 1.0f32;"),
    ("atomic_compare_exchange", "#[kernel]", ATOMIC,
     "atomic_compare_exchange(&mut counter[0], 0u32, 1u32); out[id.x] = 1.0f32;"),
    ("math::new", "#[kernel]", PARAMETERS,
     "let v = Vec3::new(1.0f32, 2.0f32, 3.0f32); out[id.x] = v.x;"),
    ("math::splat", "#[kernel]", PARAMETERS, "let v = Vec3::splat(1.0f32); out[id.x] = v.x;"),
    ("math::from_cols", "#[kernel]", PARAMETERS,
     "let m = Mat2::from_cols(Vec2::new(1.0f32, 0.0f32), Vec2::new(0.0f32, 1.0f32)); \
      out[id.x] = m.mul_vec2(Vec2::new(1.0f32, 1.0f32)).x;"),
    ("math::from_rows", "#[kernel]", PARAMETERS,
     "let m = Mat2::from_rows(Vec2::new(1.0f32, 0.0f32), Vec2::new(0.0f32, 1.0f32)); \
      out[id.x] = m.mul_vec2(Vec2::new(1.0f32, 1.0f32)).x;"),
];

/// (what the callee is, a body that calls it, the code the validator refuses it with): callee forms outside the list.
#[rustfmt::skip]
const REFUSED_CALLEES: &[(&str, &str, Code)] = &[
    ("a name that is neither a helper nor a built-in", "out[id.x] = foo();", Code::UnknownCall),
    ("a parenthesised callee", "out[id.x] = (helper)(1.0f32);", Code::UnknownCall),
    ("a qualified helper path", "out[id.x] = m::helper(1.0f32);", Code::UnknownCall),
    ("a function a math type does not have", "let v = Vec3::foo(1.0f32); out[id.x] = 1.0f32;",
     Code::UnknownCall),
    ("a matrix constant called as a function", "let v = Mat4::IDENTITY(); out[id.x] = 1.0f32;",
     Code::UnknownCall),
    ("the kernel itself", "out[id.x] = k(id, a, out);", Code::UnknownCall),
    ("a barrier used as a value", "let t = workgroup_barrier(); out[id.x] = 1.0f32;", Code::BarrierForm),
];

/// Every call expression of `function`, in source order.
fn calls_of(function: &syn::ItemFn) -> Vec<syn::ExprCall> {
    #[derive(Default)]
    struct Calls(Vec<syn::ExprCall>);
    impl<'ast> syn::visit::Visit<'ast> for Calls {
        fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
            self.0.push(call.clone());
            syn::visit::visit_expr_call(self, call);
        }
    }
    let mut calls = Calls::default();
    syn::visit::Visit::visit_item_fn(&mut calls, function);
    calls.0
}

/// The classification of the kernel of `module`, with the helpers it declares.
fn classification_of_module(module: &syn::ItemMod) -> Classification {
    classify_function(kernel_of(module), &crate::validate::helper_names(module))
}

#[test]
fn the_ledger_lists_exactly_the_callee_forms_the_validator_accepts() {
    let mut covered: Vec<&str> = CALLEES.iter().map(|probe| probe.0).collect();
    covered.sort_unstable();
    let mut listed: Vec<&str> = LEDGER
        .iter()
        .filter(|entry| entry.class == Class::Callee)
        .map(|entry| entry.kind)
        .collect();
    listed.sort_unstable();
    assert_eq!(
        covered, listed,
        "every callee entry of the ledger has one probe and the other way round; {FIX}"
    );
    for &(kind, attribute, parameters, body) in CALLEES {
        let source = probe_module_with(attribute, parameters, body);
        assert_eq!(
            verdict(&source),
            Accepted,
            "the validator and the emitter now refuse a call of the form `{kind}`; {FIX}\n{source}"
        );
        let module = module(&source);
        let helpers = crate::validate::helper_names(&module);
        let calls = calls_of(kernel_of(&module));
        assert!(
            calls
                .iter()
                .any(|call| kinds::callee_kind(call, &helpers).as_deref() == Some(kind)),
            "no call of the probe for `{kind}` is classified as `{kind}`: {FIX}\n{source}"
        );
        // The classifier takes the form as listed: it is never `Unrecognised`.
        let classification = classification_of_module(&module);
        assert!(
            !classification
                .exclusions
                .iter()
                .any(|exclusion| exclusion.why == Why::Unrecognised),
            "{kind}: {:?}\n{source}",
            classification.exclusions
        );
    }
}

#[test]
fn neutral_callee_forms_keep_pixel_and_the_inspected_ones_are_excluded_by_their_own_rule() {
    for &(kind, attribute, parameters, body) in CALLEES {
        let module = module(&probe_module_with(attribute, parameters, body));
        let classification = classification_of_module(&module);
        let entry = kinds::lookup(Class::Callee, kind).expect("listed");
        match entry.handling {
            Handling::Neutral => assert!(
                classification.allows(Shape::Pixel),
                "{kind}: {:?}",
                classification.exclusions
            ),
            Handling::Inspected(_) => {
                let why = if kind == "workgroup_barrier" {
                    Why::Barrier
                } else {
                    Why::Atomic
                };
                assert!(
                    classification
                        .exclusions
                        .iter()
                        .any(|exclusion| exclusion.why == why),
                    "{kind}: {:?}",
                    classification.exclusions
                );
            }
            Handling::Excluded(_) => panic!("{kind}: no callee form is excluded by the ledger"),
        }
    }
}

#[test]
fn the_callee_entries_name_the_atomics_and_the_math_constructors_the_validator_has() {
    let validate = include_str!("../../mod.rs");
    let math = include_str!("../../../math_types.rs");
    let mut atomics = names_between(validate, "fn atomic_operation_name", "_ =>");
    atomics.sort_unstable();
    atomics.dedup();
    let mut members = names_between(math, "let members:", ".iter()");
    members.retain(|member| member != "IDENTITY");
    members.sort_unstable();
    members.dedup();
    let named = |prefix: &str| -> Vec<String> {
        let mut names: Vec<String> = LEDGER
            .iter()
            .filter(|entry| entry.class == Class::Callee)
            .filter_map(|entry| entry.kind.strip_prefix(prefix).map(str::to_owned))
            .collect();
        names.sort_unstable();
        names
    };
    let mut listed_atomics = named("");
    listed_atomics.retain(|name| name.starts_with("atomic_"));
    assert_eq!(
        atomics, listed_atomics,
        "an atomic was added to or removed from `atomic_operation_name`; {FIX}"
    );
    assert_eq!(
        members,
        named("math::"),
        "a constructor was added to or removed from `math_types::associated_item`; {FIX}"
    );
}

#[test]
fn a_callee_the_ledger_does_not_know_makes_the_kernel_grid_only() {
    // The classifier alone, without the validator that would refuse these (the codes are in `REFUSED_CALLEES`).
    for (what, body, _) in REFUSED_CALLEES {
        if *what == "a barrier used as a value" {
            continue; // `workgroup_barrier` is a listed callee; its position is the validator's refusal.
        }
        let classification = classification_without_validator(body);
        assert_eq!(classification.shapes, [Shape::Grid], "{what}");
        let exclusion = classification
            .exclusions
            .iter()
            .find(|exclusion| exclusion.why == Why::Unrecognised)
            .unwrap_or_else(|| panic!("{what}: {:?}", classification.exclusions));
        assert!(exclusion.message("k").contains("D94"), "{what}");
    }
    // A call by a name the module does not declare is not a helper call, whatever the validator would say.
    let module = module(&probe_module(PARAMETERS, "out[id.x] = helper(1.0f32);"));
    let classification = classify_function(kernel_of(&module), &HashSet::new());
    assert!(
        classification
            .exclusions
            .iter()
            .any(|exclusion| exclusion.why == Why::Unrecognised),
        "{:?}",
        classification.exclusions
    );
    assert!(classification_of_module(&module).allows(Shape::Pixel));
}

#[test]
fn a_callee_outside_the_list_is_refused_by_the_validator_with_its_code() {
    for &(what, body, code) in REFUSED_CALLEES {
        let source = probe_module(PARAMETERS, body);
        assert_eq!(
            verdict(&source),
            Refused(code),
            "{what}: the validator now treats it differently; {FIX}\n{source}"
        );
    }
}

/// D94: the classifier scans the kernel and never the helpers it calls. That is sound only while a helper reaches
/// no resource and holds no barrier, and the validator guarantees both; loosening either refusal fails here, and the
/// decision is to scan the helpers' bodies too (or to keep the refusal).
#[test]
fn a_helper_reaches_no_resource_and_holds_no_barrier_which_the_classifier_relies_on() {
    let kernel = "#[kernel] pub fn k(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>) \
                  { out[id.x] = h(1.0f32); }";
    for parameter in [
        "b: StructuredBuffer<f32>",
        "mut b: RWStructuredBuffer<f32>",
        "t: Texture2D<f32>",
    ] {
        let source = format!("mod m {{ fn h({parameter}) -> f32 {{ 1.0f32 }} {kernel} }}");
        assert_eq!(
            verdict(&source),
            Refused(Code::HelperResourceParameter),
            "a helper with a resource parameter (D94); {FIX}\n{source}"
        );
    }
    let source = format!("mod m {{ fn h(v: f32) -> f32 {{ workgroup_barrier(); v }} {kernel} }}");
    assert_eq!(
        verdict(&source),
        Refused(Code::BarrierPosition),
        "a barrier in a helper (D94); {FIX}\n{source}"
    );
}

/// A module with a struct, one method of it and a kernel that calls the method on a storage element and on a local.
const METHOD_PROBE: &str = "mod m { pub struct P { pub x: f32 } \
    impl P { pub fn twice(&self) -> f32 { self.x * 2.0f32 } } \
    #[kernel] pub fn k(id: SV_DispatchThreadID, items: StructuredBuffer<P>, mut out: RWStructuredBuffer<f32>) { \
    let local = items[id.x]; out[id.x] = items[id.x].twice() + local.twice(); } }";

#[test]
fn a_call_of_a_method_of_the_module_is_one_listed_neutral_kind_and_keeps_pixel() {
    // D133 (slice m9.a1): the ledger lists the call of a module method by one entry, found through the `.name` marker
    // of `helper_names`; the validator and the emitter accept the probe.
    let listed: Vec<&str> = LEDGER
        .iter()
        .filter(|entry| entry.class == Class::ModuleMethod)
        .map(|entry| entry.kind)
        .collect();
    assert_eq!(listed, ["method", "associated"], "{FIX}");
    assert_eq!(
        verdict(METHOD_PROBE),
        Accepted,
        "the validator and the emitter now refuse a method call; {FIX}"
    );
    let module = module(METHOD_PROBE);
    let helpers = crate::validate::helper_names(&module);
    assert!(helpers.contains(".twice"), "{helpers:?}");
    let classification = classification_of_module(&module);
    assert!(
        !classification
            .exclusions
            .iter()
            .any(|exclusion| exclusion.why == Why::Unrecognised),
        "{:?}",
        classification.exclusions
    );
    assert!(
        classification.allows(Shape::Pixel),
        "{:?}",
        classification.exclusions
    );
    assert!(matches!(
        kinds::lookup(Class::ModuleMethod, "method").map(|entry| entry.handling),
        Some(Handling::Neutral)
    ));
    // Without the marker the call is a method the ledger does not know: `grid` only, the default of D94.
    let classification = classify_function(kernel_of(&module), &HashSet::new());
    assert!(
        classification
            .exclusions
            .iter()
            .any(|exclusion| exclusion.why == Why::Unrecognised),
        "{:?}",
        classification.exclusions
    );
}

/// A module with an associated function (no receiver) of a struct and a kernel that calls it by path (slice m9.a2).
const ASSOCIATED_PROBE: &str = "mod m { pub struct P { pub x: f32 } \
    impl P { pub fn unit() -> P { let p = P { x: 1.0f32 }; p } } \
    #[kernel] pub fn k(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>) { \
    let p = P::unit(); out[id.x] = p.x; } }";

#[test]
fn a_call_of_an_associated_function_is_one_listed_neutral_kind_and_keeps_pixel() {
    // D133 (slice m9.a2): `Struct::name(..)` is found through the `Struct::name` marker of `helper_names`.
    assert!(matches!(
        kinds::lookup(Class::ModuleMethod, "associated").map(|entry| entry.handling),
        Some(Handling::Neutral)
    ));
    assert_eq!(
        verdict(ASSOCIATED_PROBE),
        Accepted,
        "the validator and the emitter now refuse an associated call; {FIX}"
    );
    let module = module(ASSOCIATED_PROBE);
    let helpers = crate::validate::helper_names(&module);
    assert!(helpers.contains("P::unit"), "{helpers:?}");
    let classification = classification_of_module(&module);
    assert!(
        !classification
            .exclusions
            .iter()
            .any(|exclusion| exclusion.why == Why::Unrecognised),
        "{:?}",
        classification.exclusions
    );
    assert!(
        classification.allows(Shape::Pixel),
        "{:?}",
        classification.exclusions
    );
}
