use std::{env, fs, path::Path};

use crate::{
    diagnostics::{
        Code,
        testing::{assert_code, assert_code_for, assert_rendered_code},
    },
    expand::kernel_options,
    slang::{EmittedKernel, LineRecord, emit_kernel},
    validate::validate_module,
};

/// Set to `1` to rewrite a reviewed Slang golden with the emitted text. A bless prints a line
/// diff and still fails the test, so it cannot pass silently; review the diff, then rerun
/// without the variable. `cargo xtask bless` runs the golden tests this way.
const BLESS_VARIABLE: &str = "BLESS_GOLDEN";

/// Compares emitted Slang with the reviewed fixture `tests/fixtures/<fixture>`. Both texts are
/// already normalised to `\n` by the caller. Without `BLESS_GOLDEN=1` this is `assert_eq!`.
pub(crate) fn assert_golden(actual: &str, fixture: &str, expected: &str) {
    if actual == expected {
        return;
    }
    if env::var_os(BLESS_VARIABLE).is_some_and(|value| value == "1") {
        bless_golden(actual, fixture, expected);
    }
    assert_eq!(actual, expected);
}

fn bless_golden(actual: &str, fixture: &str, expected: &str) -> ! {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(fixture);
    // Keep the line endings the working tree already uses, so a bless diffs only real changes.
    let crlf = fs::read_to_string(&path).is_ok_and(|old| old.contains("\r\n"));
    let text = if crlf {
        actual.replace('\n', "\r\n")
    } else {
        actual.to_owned()
    };
    fs::write(&path, text).unwrap_or_else(|error| panic!("could not write {fixture}: {error}"));
    eprintln!("{}", line_diff(fixture, expected, actual));
    panic!("golden blessed; review the diff and rerun without BLESS_GOLDEN ({fixture})");
}

/// A unified line diff (old -> new) with three lines of context.
fn line_diff(name: &str, old: &str, new: &str) -> String {
    const CONTEXT: usize = 3;
    let old: Vec<&str> = old.lines().collect();
    let new: Vec<&str> = new.lines().collect();
    // Longest common subsequence table, filled from the ends.
    let mut common = vec![vec![0usize; new.len() + 1]; old.len() + 1];
    for i in (0..old.len()).rev() {
        for j in (0..new.len()).rev() {
            common[i][j] = if old[i] == new[j] {
                common[i + 1][j + 1] + 1
            } else {
                common[i + 1][j].max(common[i][j + 1])
            };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut edits: Vec<(char, &str)> = Vec::new();
    while i < old.len() || j < new.len() {
        if i < old.len() && j < new.len() && old[i] == new[j] {
            edits.push((' ', old[i]));
            i += 1;
            j += 1;
        } else if i < old.len() && (j == new.len() || common[i + 1][j] >= common[i][j + 1]) {
            // Deletions come before insertions, as in `diff -u`.
            edits.push(('-', old[i]));
            i += 1;
        } else {
            edits.push(('+', new[j]));
            j += 1;
        }
    }
    let changed: Vec<usize> = (0..edits.len()).filter(|&k| edits[k].0 != ' ').collect();
    let mut out = format!("--- a/tests/fixtures/{name}\n+++ b/tests/fixtures/{name}\n");
    let mut next = 0;
    while next < changed.len() {
        let start = changed[next].saturating_sub(CONTEXT);
        let mut end = changed[next];
        while next < changed.len() && changed[next] <= end + 2 * CONTEXT + 1 {
            end = changed[next];
            next += 1;
        }
        let end = (end + CONTEXT + 1).min(edits.len());
        let before = |kind: char| {
            edits[..start]
                .iter()
                .filter(|edit| edit.0 == ' ' || edit.0 == kind)
                .count()
        };
        let span = |kind: char| {
            edits[start..end]
                .iter()
                .filter(|edit| edit.0 == ' ' || edit.0 == kind)
                .count()
        };
        out.push_str(&format!(
            "@@ -{},{} +{},{} @@\n",
            before('-') + 1,
            span('-'),
            before('+') + 1,
            span('+')
        ));
        for (kind, line) in &edits[start..end] {
            out.push_str(&format!("{kind}{line}\n"));
        }
    }
    out
}

#[test]
fn line_diff_reports_changed_lines_with_context() {
    let old = "a\nb\nc\nd\ne\nf\ng\nh\n";
    let new = "a\nb\nc\nX\ne\nf\ng\nh\n";
    assert_eq!(
        line_diff("t.slang", old, new),
        "--- a/tests/fixtures/t.slang\n+++ b/tests/fixtures/t.slang\n@@ -1,7 +1,7 @@\n a\n b\n c\n-d\n+X\n e\n f\n g\n"
    );
    assert_eq!(
        line_diff("t.slang", "a\n", "a\n"),
        "--- a/tests/fixtures/t.slang\n+++ b/tests/fixtures/t.slang\n"
    );
}

fn translate(source: &str) -> syn::Result<String> {
    translate_with_lines(source).map(|emitted| emitted.source)
}

/// Asserts that translating `source` is rejected and that the first diagnostic carries
/// `code`. The code is stable; the message text is not, so tests do not match it.
fn assert_rejected(source: &str, code: Code) {
    let error = translate(source).expect_err(source);
    assert_code_for(&error, code, source);
}

fn translate_with_lines(source: &str) -> syn::Result<EmittedKernel> {
    let module: syn::ItemMod = crate::diagnostics::testing::parse_str(source)?;
    validate_module(&module)?;
    let kernel = module
        .content
        .as_ref()
        .unwrap()
        .1
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(function) if function.sig.ident == "run" => Some(function),
            _ => None,
        })
        .unwrap();
    let options = kernel_options(kernel)?.expect("kernel has #[kernel] attribute");
    emit_kernel(&module, kernel, options)
}

#[test]
fn semantics_golden() {
    let source = translate(include_str!("../../../../tests/fixtures/semantics.rs")).unwrap();
    assert_golden(
        &source.replace("\r\n", "\n"),
        "semantics.slang",
        &include_str!("../../../../tests/fixtures/semantics.slang").replace("\r\n", "\n"),
    );
}

#[test]
fn numeric_golden() {
    let source = translate(include_str!("../../../../tests/fixtures/numeric.rs")).unwrap();
    assert_golden(
        &source.replace("\r\n", "\n"),
        "numeric.slang",
        &include_str!("../../../../tests/fixtures/numeric.slang").replace("\r\n", "\n"),
    );
}

#[test]
fn option_golden() {
    let source = translate(include_str!("../../../../tests/fixtures/option.rs")).unwrap();
    assert_golden(
        &source.replace("\r\n", "\n"),
        "option.slang",
        &include_str!("../../../../tests/fixtures/option.slang").replace("\r\n", "\n"),
    );
}

#[test]
fn result_lowering_emits_tagged_representation() {
    let source = translate(include_str!("../../../../tests/fixtures/result.rs")).unwrap();
    for expected in [
        "struct __GustResult<T> { bool isOk; T value; }",
        "__gust_ok((value / uint(2)))",
        "__gust_err(value)",
        "result.isOk",
        "(!result.isOk)",
        "__gust_unwrap_or(value, fallback)",
        "var __gust_match_",
    ] {
        assert!(source.contains(expected), "missing `{expected}`:\n{source}");
    }
}

#[test]
fn result_golden() {
    let source = translate(include_str!("../../../../tests/fixtures/result.rs")).unwrap();
    assert_golden(
        &source.replace("\r\n", "\n"),
        "result.slang",
        &format!(
            "{}\n",
            include_str!("../../../../tests/fixtures/result.slang")
                .replace("\r\n", "\n")
                .trim_end()
        ),
    );
}

#[test]
fn result_rejects_unproven_forms() {
    for (source, code) in [
        (
            "mod bad { fn f() -> Result<uint, int> { Ok(1u32) } #[kernel] fn run(id: SV_DispatchThreadID) {} }",
            Code::ResultShape,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let value: Result<Option<uint>, Option<uint>> = Ok(Some(1u32)); } }",
            Code::ResultShape,
        ),
        (
            "mod bad { struct State { value: Result<uint, uint> } #[kernel] fn run(id: SV_DispatchThreadID) {} }",
            Code::ResultLayout,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, out: RWStructuredBuffer<Result<uint, uint>>) {} }",
            Code::ResultLayout,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let value: Result<uint, uint> = Ok(1u32); let x = value.unwrap(); } }",
            Code::UnwrapPanics,
        ),
    ] {
        assert_rejected(source, code);
    }
}

#[test]
fn result_match_rejects_unproven_shapes() {
    for (source, code) in [
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let value: Result<uint, uint> = Ok(1u32); match value { Ok(x) => { let y = x; } } } }",
            Code::MatchArmCount,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let value: Result<uint, uint> = Ok(1u32); match value { Ok(x) => { } Ok(y) => { } Err(e) => { } }; } }",
            Code::MatchArmCount,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let value: Result<uint, uint> = Ok(1u32); match value { Ok(x) if x > 0u32 => { } Err(e) => { } } } }",
            Code::MatchGuard,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let value: Result<uint, uint> = Ok(1u32); match value { #[allow(unused)] Ok(x) => { } Err(e) => { } }; } }",
            Code::MatchArmAttribute,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let value: Result<uint, uint> = Ok(1u32); match value { _ => { } Err(e) => { } }; } }",
            Code::MatchPattern,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let value: Result<uint, uint> = Ok(1u32); match value { Ok((x, y)) => { } Err(e) => { } }; } }",
            Code::MatchPattern,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let value: Result<uint, uint> = Ok(1u32); match value { Ok(mut x) => { } Err(e) => { } }; } }",
            Code::MatchBinding,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let value: Result<uint, uint> = Ok(1u32); match value { Ok(ref x) => { } Err(e) => { } }; } }",
            Code::MatchBinding,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let value: Result<uint, uint> = Ok(1u32); match value { Ok(x @ _) => { } Err(e) => { } }; } }",
            Code::MatchBinding,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut output: RWStructuredBuffer<uint>) { let value: Result<uint, uint> = Ok(1u32); match value { Ok(output) => { } Err(error) => { } }; } }",
            Code::ShadowedParameter,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let value: Result<uint, uint> = Ok(1u32); match value { Ok(id) => { } Err(error) => { } }; } }",
            Code::ShadowedParameter,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let value: Result<uint, uint> = Ok(1u32); match value { Ok(x) => x, Err(e) => e } } }",
            Code::MatchArmBlock,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let value: Result<uint, uint> = Ok(1u32); match value { Ok(x) => { let y = x; } Err(e) => { let y = e; } } } }",
            Code::MatchNeedsSemicolon,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let value: Result<uint, uint> = Ok(1u32); let output = match value { Ok(x) => { x } Err(e) => { e } }; } }",
            Code::MatchAsValue,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { match 1u32 { Ok(x) => { } Err(e) => { } }; } }",
            Code::MatchScrutinee,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let value: Option<uint> = Some(1u32); match value { Ok(x) => { } Err(e) => { } }; } }",
            Code::MatchScrutinee,
        ),
    ] {
        assert_rejected(source, code);
    }
}

#[test]
fn result_match_emits_single_scrutinee_evaluation() {
    let source = translate(
        "mod results {
            fn classify(value: uint) -> Result<uint, uint> { if value > 0u32 { Ok(value) } else { Err(value) } }
            #[kernel] fn run(id: SV_DispatchThreadID, mut output: RWStructuredBuffer<uint>) {
                match classify(id.x) {
                    Err(error) => { output[id.x] = error + 1u32; }
                    Ok(value) => { output[id.x] = value + 2u32; }
                };
            }
        }",
    )
    .unwrap();
    assert!(
        source.contains("var __gust_match_0 = classify(id.x);"),
        "{source}"
    );
    assert!(source.contains("if (__gust_match_0.isOk)"), "{source}");
    assert!(
        source.contains("var value = __gust_match_0.value;"),
        "{source}"
    );
    assert!(
        source.contains("var error = __gust_match_0.value;"),
        "{source}"
    );
}

#[test]
fn result_match_accepts_a_result_parameter() {
    let source = translate(
        "mod results {
            fn select(value: Result<uint, uint>) -> uint {
                let mut output = 0u32;
                match value {
                    Ok(ok) => { output = ok; }
                    Err(error) => { output = error + 1u32; }
                };
                output
            }
            #[kernel] fn run(id: SV_DispatchThreadID, mut output: RWStructuredBuffer<uint>) {
                output[id.x] = select(Ok(id.x));
            }
        }",
    )
    .unwrap();
    assert!(
        source.contains("uint select(__GustResult<uint> value)"),
        "{source}"
    );
    assert!(source.contains("var __gust_match_1 = value;"), "{source}");
}

#[test]
fn loops_golden() {
    let source = translate(include_str!("../../../../tests/fixtures/loops.rs")).unwrap();
    assert_golden(
        &source.replace("\r\n", "\n"),
        "loops.slang",
        &include_str!("../../../../tests/fixtures/loops.slang").replace("\r\n", "\n"),
    );
}

#[test]
fn atomics_golden() {
    let source = translate(include_str!("../../../../tests/fixtures/atomics.rs")).unwrap();
    assert_golden(
        &source.replace("\r\n", "\n"),
        "atomics.slang",
        &include_str!("../../../../tests/fixtures/atomics.slang").replace("\r\n", "\n"),
    );
}

#[test]
fn unbounded_or_unproven_loop_forms_are_rejected() {
    for (source, code) in [
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let mut i = 0u32; while i < 4u32 { i = i + 1u32; } } }",
            Code::WhileNotSupported,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { loop { break; } } }",
            Code::LoopNotSupported,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { for i in 0u32..=4u32 { } } }",
            Code::InclusiveRange,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { for i in 0u32.. { } } }",
            Code::OpenRange,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { for i in ..4u32 { } } }",
            Code::OpenRange,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, input: StructuredBuffer<uint>) { for v in input { } } }",
            Code::ForIterable,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { for (a, b) in 0u32..4u32 { } } }",
            Code::LoopVariablePattern,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { for mut i in 0u32..4u32 { i = i + 1u32; } } }",
            Code::MutableLoopVariable,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { 'outer: for i in 0u32..4u32 { break 'outer; } } }",
            Code::LoopLabel,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { for i in 0u32..4u32 { break 'x; } } }",
            Code::LoopLabel,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { if id.x > 1u32 { break; } } }",
            Code::JumpOutsideLoop,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { continue; } }",
            Code::JumpOutsideLoop,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let r = 0u32..4u32; } }",
            Code::RangeAsValue,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { for i in 0..4u32 { } } }",
            Code::UnsuffixedLoopBound,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { for i in 0u32..(-4) { } } }",
            Code::UnsuffixedLoopBound,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, out: RWStructuredBuffer<uint>) { for out in 0u32..4u32 { } } }",
            Code::ShadowedParameter,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { for i in 0u32..4u32 { let v = break; } } }",
            Code::JumpAsValue,
        ),
    ] {
        assert_rejected(source, code);
    }
    // Rejected by the emitter, not the validator.
    assert_rejected(
        "mod bad { fn f() -> uint { for i in 0u32..4u32 { fn g() {} } 0u32 } #[kernel] fn run(id: SV_DispatchThreadID) { } }",
        Code::NestedItem,
    );
}

/// Slice m6.z2 (D86): an unannotated `let` whose initializer is only unsuffixed integer literals is GUST-E0326, because
/// Slang types it `int` while rustc infers the type from the later uses (`let mut d = 7; d /= y` with `y: u32`).
#[test]
fn an_unannotated_unsuffixed_integer_literal_let_is_e0326() {
    for body in [
        "let mut acc = 0; acc += id.x;",
        "let x = 7;",
        "let x = -1;",
        "let x = (7);",
        "let x = -(2147483648);",
        "let x = 0xFF;",
        "let x = 1 + 2;",
        "let x = (1 + 2) * 3 - 4 / 2 % 3;",
        "let x = 1 << 3;",
        "let x = 1 << id.x;",
        "let x = 1 << 3u32;",
        "let x = 6 & 3 | 8 ^ 1;",
        "let x = !0;",
        "let x = Some(7);",
        "let x = Ok(7);",
        "let x = Err(-7);",
        "let mut d = Some(7).unwrap_or(0); d /= x;",
        "let x = Some(-1).unwrap_or(2 + 3);",
    ] {
        let source = kernel_with_body(body);
        assert_rejected(&source, Code::UnsuffixedLiteralLet);
        let text = translate(&source).unwrap_err().to_string();
        assert!(
            text.contains("add a suffix (`0u32`, `0i32`) or a type annotation"),
            "{body}: {text}"
        );
    }
    // An `if let` scrutinee binds an untyped local as well; `while let` and `let .. else` are rejected elsewhere.
    for body in [
        "if let Some(mut v) = Some(7) { v /= x; out[id.x] = v; }",
        "if let Ok(v) = Ok(7) { out[id.x] = v; }",
        "if let Some(v) = Some(7).unwrap_or(0) { }",
    ] {
        assert_rejected(&kernel_with_body(body), Code::UnsuffixedLiteralLet);
    }
    assert_rejected(
        &kernel_with_body("while let Some(v) = Some(7) { }"),
        Code::WhileNotSupported,
    );
    assert_rejected(
        &kernel_with_body("let Some(v) = Some(7) else { return; };"),
        Code::LetNeedsIdentifier,
    );
    // `let x;` has no initializer to check: the emitter refuses it (E0803).
    assert_rejected(&kernel_with_body("let x;"), Code::LetNeedsInitializer);
    // An `if` or a `match` is not a value in the dialect, so literal arms never reach this check (docs/DIAGNOSTICS.md).
    assert_rejected(
        &kernel_with_body("let y = if id.x > 1u32 { 1 } else { 2 };"),
        Code::UnsupportedExpression,
    );
    assert_rejected(
        &kernel_with_body("let y = match id.x { _ => 1 };"),
        Code::MatchAsValue,
    );
    // One error per such `let`, in source order; the annotated one between them is fine.
    let error = translate(&kernel_with_body("let a = 1; let b: u32 = 2; let c = 3;")).unwrap_err();
    crate::diagnostics::testing::assert_codes(
        &error,
        &[Code::UnsuffixedLiteralLet, Code::UnsuffixedLiteralLet],
    );
}

/// What stays accepted: any suffix, any annotation, and every initializer with an operand that has a type of its own.
/// Float literals are not covered: they are documented, not refused (docs/SEMANTICS.md).
#[test]
fn a_suffixed_annotated_or_typed_let_is_not_e0326() {
    for body in [
        "let mut acc = 0u32; acc += id.x;",
        "let x = 7i32;",
        "let x = -1i32;",
        "let x = (7u32);",
        "let x: u32 = 7;",
        "let x: i32 = -(1 + 2);",
        "let x: u32 = 1 << 3;",
        "let x = 1u32 + 2;",
        "let x = 1 + 2u32;",
        "let x = 1u32 << 3;",
        "let x = id.x << 1;",
        "let x = 3 * id.x;",
        "let x = 7 as u32;",
        "let x = id.x + 1;",
        "let x = 2.0;",
        "let x = true;",
        "let x = 1 < 2;",
        "let x: Option<u32> = Some(7);",
        "let x = Some(7u32);",
        "let x = Some(7u32).unwrap_or(0);",
        "let x = Some(id.x).unwrap_or(0);",
        "if let Some(v) = Some(7u32) { out[id.x] = v; }",
        "if let Some(v) = Some(id.x) { out[id.x] = v; }",
    ] {
        let source = kernel_with_body(body);
        let module: syn::ItemMod = crate::diagnostics::testing::parse_str(&source).unwrap();
        if let Err(error) = validate_module(&module) {
            panic!("{body}: {error}");
        }
    }
}

#[test]
fn for_loop_end_bound_is_evaluated_once_and_body_control_flow_maps_directly() {
    let source = translate(
        "mod shapes {
        fn count(limit: uint) -> uint {
            let mut n = 0u32;
            for i in 2u32..limit { if i == 3u32 { continue; } if i > 6u32 { break; } n = n + i; }
            n
        }
        #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
            if id.x < out.len() { for _ in 0u32..2u32 { out[id.x] = count(out[id.x]); } }
        }
    }",
    )
    .unwrap();
    assert!(source.contains("var __gust_end_1 = limit;"), "{source}");
    assert!(
        source.contains("for (var i = uint(2); i < __gust_end_1; i++)"),
        "{source}"
    );
    assert!(source.contains("continue;"), "{source}");
    assert!(source.contains("break;"), "{source}");
    assert!(
        source.contains(
            "for (var __gust_iter_0 = uint(0); __gust_iter_0 < __gust_end_0; __gust_iter_0++)"
        ),
        "{source}"
    );
}

#[test]
fn option_unsafe_forms_are_rejected() {
    for (source, code) in [
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let o: Option<uint> = Some(1u32); let v = o.unwrap(); } }",
            Code::UnwrapPanics,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let o: Option<uint> = Some(1u32); let v = o.expect(\"x\"); } }",
            Code::UnwrapPanics,
        ),
        (
            "mod bad { struct Holder { slot: Option<uint> } #[kernel] fn run(id: SV_DispatchThreadID) {} }",
            Code::OptionLayout,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, out: RWStructuredBuffer<Option<uint>>) {} }",
            Code::OptionLayout,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let o: Option<Option<uint>> = None; } }",
            Code::OptionPayload,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let o: Option<uint> = Some(1u32); if let Some(_) = o { } } }",
            Code::IfLetPattern,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let o: Option<uint> = Some(1u32); if let None = o { } } }",
            Code::IfLetPattern,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let o: Option<uint> = Some(1u32); if let Some(v) = o && v > 0u32 { } } }",
            Code::IfLetPosition,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, out: RWStructuredBuffer<uint>) { let o: Option<uint> = Some(1u32); if let Some(out) = o { } } }",
            Code::ShadowedParameter,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let o: Option<uint> = Some(1u32, 2u32); } }",
            Code::SomeArguments,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let o: Option<uint> = Some(1u32); let v = o.unwrap_or(1u32, 2u32); } }",
            Code::UnwrapOrArguments,
        ),
    ] {
        assert_rejected(source, code);
    }
}

#[test]
fn generated_symbol_prefixes_are_reserved() {
    for source in [
        "mod bad { #[kernel] fn run(__gpu_len_out: SV_DispatchThreadID) {} }",
        "mod bad { fn __gust_not(x: uint) -> uint { x } #[kernel] fn run(id: SV_DispatchThreadID) {} }",
    ] {
        assert_rejected(source, Code::ReservedPrefix);
    }
}

#[test]
fn invalid_kernel_signatures_are_rejected_early() {
    for (source, code) in [
        (
            "mod bad { #[kernel] fn run(a: SV_DispatchThreadID, b: SV_DispatchThreadID) {} }",
            Code::KernelDispatchParameter,
        ),
        (
            "mod bad { #[kernel] fn run() {} }",
            Code::KernelDispatchParameter,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) -> uint { 1u32 } }",
            Code::KernelReturnsUnit,
        ),
    ] {
        assert_rejected(source, code);
    }
}

#[test]
fn invalid_kernel_options_are_rejected_early() {
    // Rejected by `kernel_options` in expand/mod.rs.
    for (source, code) in [
        (
            "mod bad { #[kernel(workgroup_size(0, 1, 1))] fn run(id: SV_DispatchThreadID) {} }",
            Code::WorkgroupDimensionZero,
        ),
        (
            "mod bad { #[kernel] #[kernel] fn run(id: SV_DispatchThreadID) {} }",
            Code::DuplicateKernelAttribute,
        ),
        (
            "mod bad { #[kernel(workgroup_size(64, 1, 1), workgroup_size(32, 1, 1))] fn run(id: SV_DispatchThreadID) {} }",
            Code::DuplicateWorkgroupSize,
        ),
        (
            "mod bad { #[kernel(workgroup_size(1, 2, 3, 4))] fn run(id: SV_DispatchThreadID) {} }",
            Code::WorkgroupDimensionCount,
        ),
        (
            "mod bad { #[kernel(threads(64))] fn run(id: SV_DispatchThreadID) {} }",
            Code::UnknownKernelOption,
        ),
        // Failures `syn` words itself get the malformed-attribute code.
        (
            "mod bad { #[kernel(workgroup_size(64, 1))] fn run(id: SV_DispatchThreadID) {} }",
            Code::MalformedKernelAttribute,
        ),
        (
            "mod bad { #[kernel(workgroup_size(a, 1, 1))] fn run(id: SV_DispatchThreadID) {} }",
            Code::MalformedKernelAttribute,
        ),
        (
            "mod bad { #[kernel(workgroup_size(99999999999, 1, 1))] fn run(id: SV_DispatchThreadID) {} }",
            Code::MalformedKernelAttribute,
        ),
    ] {
        let module: syn::ItemMod = crate::diagnostics::testing::parse_str(source).unwrap();
        let error = crate::expand::expand(module).unwrap_err();
        assert_code(&error, code);
    }
}

#[test]
fn unsupported_attributes_and_builtin_receivers_are_diagnosed() {
    for (source, code) in [
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, out: RWStructuredBuffer<uint>) { let out = 1u32; let n = out.len(); } }",
            Code::ShadowedParameter,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { #[cfg(any())] let ignored = 1u32; } }",
            Code::UnsupportedAttribute,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let n = id.len(); } }",
            Code::LenReceiver,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, out: RWStructuredBuffer<uint>) { let n = out.len(1u32); } }",
            Code::BuiltinArguments,
        ),
        (
            "mod bad { fn helper(b: StructuredBuffer<uint>) -> uint { b.len() } #[kernel] fn run(id: SV_DispatchThreadID) {} }",
            Code::HelperResourceParameter,
        ),
        (
            "mod bad { struct Generic<T> { field: T } #[kernel] fn run(id: SV_DispatchThreadID) {} }",
            Code::GenericStruct,
        ),
    ] {
        assert_rejected(source, code);
    }
}

#[test]
fn helper_tail_branches_return_values() {
    let source = translate(
        "mod tails {
        fn choose(x: float) -> float { if x > 0.0 { x * 2.0 } else { { -x } } }
        #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) {
            if id.x < out.len() { out[id.x] = choose(2.0); }
        }
    }",
    )
    .unwrap();
    assert!(source.contains("return (x * 2.0);"), "{source}");
    assert!(source.contains("return (-x);"), "{source}");
    assert!(!source.contains("return out["));
}

#[test]
fn typed_locals_preserve_numeric_type() {
    let source = translate("mod typed {
        fn quotient(x: uint) -> uint { let divisor: uint = 2; let mut answer: uint = x / divisor; answer += 1u32; answer }
        #[kernel] fn run(id: SV_DispatchThreadID) { let result: uint = quotient(id.x); }
    }").unwrap();
    assert!(source.contains("uint divisor = 2;"), "{source}");
    assert!(
        source.contains("uint answer = __gust_div(x, divisor);"),
        "{source}"
    );
    assert!(source.contains("uint result = quotient(id.x);"), "{source}");
}

/// Slice m6.dv (D74): a divisor that is not a literal goes through `__gust_div` and `__gust_rem`, a literal divisor
/// (negated, parenthesised or not) stays a plain operator, and each helper family is emitted once and only when used.
#[test]
fn division_goes_through_helpers_unless_the_divisor_is_a_literal() {
    let source = translate(
        "mod divide {
        #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<int>, d: StructuredBuffer<int>) {
            let a: int = out[id.x];
            let b: int = d[id.x];
            let mut c: int = a / b;
            c += a % b;
            c /= b;
            c %= b;
            c += a / 2 + a % 3 + a / (-4) + a / (5);
            out[id.x] /= b;
            out[id.x] = c;
        }
    }",
    )
    .unwrap();
    for line in [
        "__gust_div(a, b)",
        "__gust_rem(a, b)",
        "__gust_div_assign(c, b)",
        "__gust_rem_assign(c, b)",
        "__gust_div_assign(out[id.x], b)",
        "(a / 2)",
        "(a % 3)",
        "(a / ((-4)))",
        "(a / (5))",
    ] {
        assert!(source.contains(line), "missing `{line}`:\n{source}");
    }
    assert_eq!(
        source.matches("int __gust_div(int a, int b)").count(),
        1,
        "{source}"
    );
    assert_eq!(
        source.matches("int __gust_rem(int a, int b)").count(),
        1,
        "{source}"
    );
    assert_eq!(
        source
            .matches("float2 __gust_div(float2 a, float b)")
            .count(),
        1,
        "{source}"
    );
    assert!(
        !source.contains("__gust_rem(float2"),
        "`%` has scalar overloads only:\n{source}"
    );

    let literal_only = translate(
        "mod literal {
        #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
            out[id.x] = out[id.x] / 2 + out[id.x] % 3;
            out[id.x] /= 4;
        }
    }",
    )
    .unwrap();
    assert!(!literal_only.contains("__gust_div"), "{literal_only}");
    assert!(!literal_only.contains("__gust_rem"), "{literal_only}");

    let only_div = translate(
        "mod only_div {
        #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
            out[id.x] = id.x / out[id.x];
        }
    }",
    )
    .unwrap();
    assert!(
        only_div.contains("uint __gust_div(uint a, uint b)"),
        "{only_div}"
    );
    assert!(!only_div.contains("__gust_rem"), "{only_div}");
}

#[test]
fn binding_modes_and_typed_resource_shadowing_fail_explicitly() {
    for (source, code) in [
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let ref x: uint = 1u32; } }",
            Code::BindingMode,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let id: uint = 1u32; } }",
            Code::ShadowedParameter,
        ),
        (
            "mod bad { #[kernel] fn run(ref id: SV_DispatchThreadID) {} }",
            Code::BindingMode,
        ),
    ] {
        assert_rejected(source, code);
    }
}

#[test]
fn rust_literals_are_lowered_not_copied() {
    let source = translate(
        "mod literals {
        #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
            if id.x < out.len() { out[id.x] = 0xff_u32 + 0b10u32 + 1_024u32 + (2.5f32 as uint); }
        }
    }",
    )
    .unwrap();
    assert!(source.contains("uint(255)"), "{source}");
    assert!(source.contains("uint(2)"), "{source}");
    assert!(source.contains("uint(1024)"), "{source}");
    assert!(source.contains("float(2.5)"), "{source}");
}

#[test]
fn binary_grouping_preserves_rust_precedence() {
    let source = translate(
        "mod grouping {
        fn masked(x: uint) -> bool { return x & 1u32 == 0u32; }
        #[kernel] fn run(id: SV_DispatchThreadID) { let even = masked(id.x); }
    }",
    )
    .unwrap();
    assert!(
        source.contains("return ((x & uint(1)) == uint(0));"),
        "{source}"
    );
}

#[test]
fn struct_constructor_preserves_field_identity_and_source_order() {
    let source = translate(
        "mod constructors {
        struct Pair { first: uint, second: uint }
        #[kernel] fn run(id: SV_DispatchThreadID) { let p = Pair { second: 1u32, first: 2u32 }; }
    }",
    )
    .unwrap();
    // Field identity is preserved by name even though `second` is written first.
    assert!(source.contains("Pair __gust_struct_0;"), "{source}");
    assert!(
        source.contains("__gust_struct_0.second = uint(1);"),
        "{source}"
    );
    assert!(
        source.contains("__gust_struct_0.first = uint(2);"),
        "{source}"
    );
    assert!(source.contains("Pair p = __gust_struct_0;"), "{source}");
    // Source evaluation order is preserved: `second` is assigned before `first`.
    assert!(
        source.find("__gust_struct_0.second = uint(1);").unwrap()
            < source.find("__gust_struct_0.first = uint(2);").unwrap(),
        "{source}"
    );
}

#[test]
fn struct_literal_in_expression_position_is_diagnosed() {
    assert_rejected(
        "mod constructors {
        struct Pair { first: uint, second: uint }
        fn take(p: Pair) -> uint { p.first }
        #[kernel] fn run(id: SV_DispatchThreadID) { let v = take(Pair { first: 1u32, second: 2u32 }); }
    }",
        Code::StructLiteralPosition,
    );
}

#[test]
fn struct_update_syntax_is_rejected_instead_of_dropping_fields() {
    for statement in [
        "let q = Pair { first: 9u32, ..p };",
        "p = Pair { first: 9u32, ..p };",
    ] {
        let source = format!(
            "mod update {{ struct Pair {{ first: uint, second: uint }} #[kernel] fn run(id: SV_DispatchThreadID) {{ let mut p = Pair {{ first: 1u32, second: 2u32 }}; {statement} }} }}"
        );
        assert_rejected(&source, Code::StructUpdateSyntax);
    }
}

#[test]
fn unsupported_numeric_width_is_diagnosed() {
    assert_rejected(
        "mod widths {
        #[kernel] fn run(id: SV_DispatchThreadID) { let wide = 1u64; }
    }",
        Code::UnsupportedLiteral,
    );
}

/// A kernel `run` whose body is `body`, with a dispatch-thread parameter `id`, a read-write
/// buffer `out` and a local `x`.
fn kernel_with_body(body: &str) -> String {
    format!(
        "mod bad {{ #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {{ let x = 7u32; {body} }} }}"
    )
}

#[test]
fn expression_and_binding_rejections_carry_their_codes() {
    for (body, code) in [
        ("let y = foo(1u32);", Code::UnknownCall),
        ("let y = id.global_id::<u32>();", Code::BuiltinTypeArguments),
        ("let y = x.abs();", Code::UnsupportedMethod),
        ("let y = out.global_id();", Code::GlobalIdReceiver),
        ("let f = |a: u32| a;", Code::Closure),
        ("let y = x.await;", Code::Await),
        ("let y = &x;", Code::Reference),
        ("let (a, b) = x;", Code::LetNeedsIdentifier),
        ("let v = x else { return; };", Code::LetElse),
        ("for i in 0u32..4u32 { break 3u32; }", Code::BreakWithValue),
        ("let r = Ok(1u32, 2u32);", Code::OkErrArguments),
        ("let f = Some;", Code::VariantAsValue),
        (
            "let o = Some(1u32); let b = o.is_some(1u32);",
            Code::PredicateArguments,
        ),
    ] {
        assert_rejected(&kernel_with_body(body), code);
    }
}

#[test]
fn emitter_only_rejections_are_coded() {
    for (source, code) in [
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, x: uint) {} }",
            Code::EntryPointParameter,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let x: uint; } }",
            Code::LetNeedsInitializer,
        ),
        (
            "mod bad { struct Pair { first: uint } #[kernel] fn run(id: SV_DispatchThreadID) { let p = Pair { 0: 1u32 }; } }",
            Code::StructLiteralFieldName,
        ),
        (
            "mod bad { fn f((a, b): uint) -> uint { 1u32 } #[kernel] fn run(id: SV_DispatchThreadID) {} }",
            Code::ParameterNeedsIdentifier,
        ),
        (
            "mod bad { fn f(self) -> uint { 1u32 } #[kernel] fn run(id: SV_DispatchThreadID) {} }",
            Code::SelfParameter,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let t = (1u32, 2u32); } }",
            Code::UnsupportedExpression,
        ),
        (
            "mod bad { fn f(a: (u32, u32)) -> uint { 1u32 } #[kernel] fn run(id: SV_DispatchThreadID) {} }",
            Code::TypeNeedsPath,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, b: StructuredBuffer) {} }",
            Code::ResourceElementArity,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, b: StructuredBuffer<'a>) {} }",
            Code::ResourceElementMissing,
        ),
    ] {
        assert_rejected(source, code);
    }
}

/// Codes the emitter keeps as a second line of defence: the validator rejects the same
/// shape first, so these are reached only when a module skips validation. Each is
/// asserted directly against the emitter or expander, and listed in `UNREACHABLE`.
#[test]
fn unreachable_emitter_rejections_are_coded_when_validation_is_skipped() {
    let module: syn::ItemMod = syn::parse_quote! { mod bad { #[kernel] fn run() {} } };
    let kernel = module
        .content
        .as_ref()
        .unwrap()
        .1
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(function) => Some(function),
            _ => None,
        })
        .unwrap();
    let options = kernel_options(kernel).unwrap().unwrap();
    let error = emit_kernel(&module, kernel, options)
        .err()
        .expect("the emitter rejects a kernel without a dispatch parameter");
    assert_code(&error, Code::EntryPointDispatchParameter);
    // And the validator stops the same module first, with its own code.
    assert_code(
        &validate_module(&module).unwrap_err(),
        Code::KernelDispatchParameter,
    );
}

#[test]
fn casts_to_unproven_targets_are_rejected() {
    // `uint` is a plain `u32` alias, so rustc accepts every numeric target below in
    // the shadow body; only the macro can stop `f64(id.x)` reaching slangc.
    for target in [
        "f64",
        "u64",
        "i64",
        "usize",
        "isize",
        "u8",
        "u16",
        "i8",
        "i16",
        "bool",
        "char",
        "double",
        "Pair",
        "_",
        "*const uint",
        "gust::uint",
    ] {
        let source = format!(
            "mod bad {{ struct Pair {{ first: uint, second: uint }} #[kernel] fn run(id: SV_DispatchThreadID) {{ let v = id.x as {target}; }} }}"
        );
        assert_rejected(&source, Code::UnsupportedCastTarget);
    }
    for (target, expected) in [
        ("f32", "float(id.x)"),
        ("float", "float(id.x)"),
        ("i32", "int(id.x)"),
        ("int", "int(id.x)"),
        ("u32", "uint(id.x)"),
        ("uint", "uint(id.x)"),
    ] {
        let source = format!(
            "mod ok {{ #[kernel] fn run(id: SV_DispatchThreadID) {{ let v = id.x as {target}; }} }}"
        );
        let ok = translate(&source).unwrap();
        assert!(ok.contains(expected), "{ok}");
    }
}

#[test]
fn non_32_bit_primitive_types_are_rejected_before_slang() {
    // rustc accepts each of these in the shadow; without a validator rule they
    // became `f64 v = 1.0;` and failed only inside slangc at pipeline creation.
    for (source, name) in [
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let v: f64 = 1.0; } }",
            "f64",
        ),
        (
            "mod bad { fn helper(x: u64) -> uint { 1u32 } #[kernel] fn run(id: SV_DispatchThreadID) {} }",
            "u64",
        ),
        (
            "mod bad { fn helper(x: uint) -> usize { 1 } #[kernel] fn run(id: SV_DispatchThreadID) {} }",
            "usize",
        ),
        (
            "mod bad { struct Wide { value: f64 } #[kernel] fn run(id: SV_DispatchThreadID) {} }",
            "f64",
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, out: RWStructuredBuffer<i64>) {} }",
            "i64",
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let c: char = 'a'; } }",
            "char",
        ),
    ] {
        assert_rejected(source, Code::UnsupportedPrimitive);
        // The message names the rejected type; that text helps users, so pin it too.
        let message = translate(source).unwrap_err().to_string();
        assert!(message.contains(name), "{message}: {source}");
    }
    let ok = translate(
        "mod ok { struct Mixed { a: float, b: int, c: uint, d: bool, e: f32, f: i32, g: u32 } #[kernel] fn run(id: SV_DispatchThreadID) { let flag: bool = true; } }",
    )
    .unwrap();
    assert!(ok.contains("bool flag = true;"), "{ok}");
}

#[test]
fn const_blocks_are_rejected() {
    for source in [
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let v = const { 1u32 }; } }",
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) { out[id.x] = const { 2u32 } + 1u32; } }",
        "mod bad { fn helper() -> uint { const { 3u32 } } #[kernel] fn run(id: SV_DispatchThreadID) {} }",
    ] {
        assert_rejected(source, Code::ConstBlock);
    }
}

#[test]
fn unsupported_name_paths_are_rejected() {
    for (source, code) in [
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let v = f32::INFINITY; } }",
            Code::ForeignPath,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let v = u32::MAX; } }",
            Code::ForeignPath,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let v = core::f32::consts::PI; } }",
            Code::ForeignPath,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let v: core::f32 = 1.0; } }",
            Code::CpuOnlyName,
        ),
    ] {
        assert_rejected(source, code);
    }
}

#[test]
fn unsupported_effects_are_rejected() {
    for (source, code) in [
        (
            "mod bad { fn maybe() -> Option<uint> { 1u32 } #[kernel] fn run(id: SV_DispatchThreadID) { let v = maybe()?; } }",
            Code::TryOperator,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { unsafe { let v = 1u32; } } }",
            Code::UnsafeBlock,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { let v = asm!(\"nop\"); } }",
            Code::MacroInModule,
        ),
    ] {
        assert_rejected(source, code);
    }
    // A macro in statement position is rejected by the emitter, under its own code.
    assert_rejected(
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID) { asm!(\"nop\"); } }",
        Code::MacroStatement,
    );
}

#[test]
fn integer_division_by_literal_zero_is_diagnosed() {
    for source in [
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) { out[id.x] = out[id.x] / 0u32; } }",
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<int>) { out[id.x] = out[id.x] % 0; } }",
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) { out[id.x] = out[id.x] / -0u32; } }",
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) { out[id.x] = out[id.x] / (0u32); } }",
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<int>) { out[id.x] = out[id.x] % ((-0)); } }",
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) { out[id.x] /= 0u32; } }",
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<int>) { out[id.x] %= 0; } }",
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) { let mut t = out[id.x]; t /= (0); out[id.x] = t; } }",
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) { if false { out[id.x] %= -0u32; } } }",
    ] {
        assert_rejected(source, Code::DivisionByLiteralZero);
    }
    // Float division by zero is defined (it yields inf/NaN), so it must not be
    // rejected even though the divisor literal is zero.
    let ok = translate(
        "mod ok { #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) { out[id.x] = out[id.x] / 0.0; } }",
    )
    .unwrap();
    assert!(ok.contains("(out[id.x] / 0.0)"), "{ok}");
}

#[test]
fn a_literal_zero_through_parentheses_or_a_macro_group_is_still_zero() {
    // `macro_rules!` hands a `$d:expr` over inside an invisible group; the emitter keeps that spelling a plain operator,
    // so the validator must see it as the zero it is. Build the group, as the parser would not.
    let parse = |text: &str| crate::diagnostics::testing::parse_str::<syn::Expr>(text).unwrap();
    let group = |text: &str| {
        syn::Expr::Group(syn::ExprGroup {
            attrs: Vec::new(),
            group_token: syn::token::Group::default(),
            expr: Box::new(parse(text)),
        })
    };
    for text in ["0", "0u32", "-0", "(0)", "((-0u32))"] {
        assert!(crate::validate::is_literal_zero(&parse(text)), "{text}");
        assert!(
            crate::validate::is_literal_zero(&group(text)),
            "group of {text}"
        );
    }
    for text in ["1", "(1)", "x", "(x)", "0.0", "(0.0)", "-1"] {
        assert!(!crate::validate::is_literal_zero(&parse(text)), "{text}");
        assert!(
            !crate::validate::is_literal_zero(&group(text)),
            "group of {text}"
        );
    }
}

// --- T10 atomics ---

#[test]
fn atomic_add_on_rw_u32_buffer_element_is_accepted() {
    let source = translate(
        "mod atomics {
        #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) {
            atomic_add(&mut counter[0], 1u32);
        }
    }",
    )
    .unwrap();
    assert!(
        source.contains("RWStructuredBuffer<Atomic<uint>> counter;"),
        "buffer type must use Atomic<uint>: {source}"
    );
    assert!(
        source.contains("counter[0].add(uint(1));"),
        "atomic_add must lower to .add(): {source}"
    );
}

#[test]
fn atomic_add_with_dynamic_index_is_accepted() {
    let source = translate(
        "mod atomics {
        #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) {
            atomic_add(&mut counter[id.x], 1u32);
        }
    }",
    )
    .unwrap();
    assert!(
        source.contains("RWStructuredBuffer<Atomic<uint>> counter;"),
        "buffer type must use Atomic<uint>: {source}"
    );
    assert!(
        source.contains("counter[id.x].add(uint(1));"),
        "atomic_add must lower to .add(): {source}"
    );
}

#[test]
fn atomic_add_on_rw_i32_buffer_element_is_accepted() {
    let source = translate(
        "mod atomics {
        #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<int>) {
            atomic_add(&mut counter[0], 1i32);
        }
    }",
    )
    .unwrap();
    assert!(
        source.contains("RWStructuredBuffer<Atomic<int>> counter;"),
        "buffer type must use Atomic<int>: {source}"
    );
    assert!(
        source.contains("counter[0].add(int(1));"),
        "atomic_add on i32 must lower to .add(): {source}"
    );
}

#[test]
fn atomic_min_on_rw_u32_is_accepted() {
    let source = translate(
        "mod atomics {
        #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) {
            atomic_min(&mut counter[0], 5u32);
        }
    }",
    )
    .unwrap();
    assert!(
        source.contains("RWStructuredBuffer<Atomic<uint>> counter;"),
        "buffer type must use Atomic<uint>: {source}"
    );
    assert!(
        source.contains("counter[0].min(uint(5));"),
        "atomic_min must lower to .min(): {source}"
    );
}

#[test]
fn atomic_max_on_rw_u32_is_accepted() {
    let source = translate(
        "mod atomics {
        #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) {
            atomic_max(&mut counter[0], 5u32);
        }
    }",
    )
    .unwrap();
    assert!(
        source.contains("RWStructuredBuffer<Atomic<uint>> counter;"),
        "buffer type must use Atomic<uint>: {source}"
    );
    assert!(
        source.contains("counter[0].max(uint(5));"),
        "atomic_max must lower to .max(): {source}"
    );
}

#[test]
fn atomic_exchange_on_rw_u32_is_accepted() {
    let source = translate(
        "mod atomics {
        #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) {
            atomic_exchange(&mut counter[0], 42u32);
        }
    }",
    )
    .unwrap();
    assert!(
        source.contains("RWStructuredBuffer<Atomic<uint>> counter;"),
        "buffer type must use Atomic<uint>: {source}"
    );
    assert!(
        source.contains("counter[0].exchange(uint(42));"),
        "atomic_exchange must lower to .exchange(): {source}"
    );
}

#[test]
fn atomic_exchange_on_rw_i32_is_accepted() {
    let source = translate(
        "mod atomics {
        #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<int>) {
            atomic_exchange(&mut counter[0], 1i32);
        }
    }",
    )
    .unwrap();
    assert!(
        source.contains("RWStructuredBuffer<Atomic<int>> counter;"),
        "buffer type must use Atomic<int>: {source}"
    );
    assert!(
        source.contains("counter[0].exchange(int(1));"),
        "atomic_exchange on i32 must lower to .exchange(): {source}"
    );
}

#[test]
fn atomic_compare_exchange_on_rw_u32_is_accepted() {
    let source = translate(
        "mod atomics {
        #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) {
            atomic_compare_exchange(&mut counter[0], 0u32, 1u32);
        }
    }",
    )
    .unwrap();
    assert!(
        source.contains("RWStructuredBuffer<Atomic<uint>> counter;"),
        "buffer type must use Atomic<uint>: {source}"
    );
    assert!(
        source.contains("counter[0].compareExchange(uint(0), uint(1));"),
        "atomic_compare_exchange must lower to .compareExchange(): {source}"
    );
}

#[test]
fn atomic_compare_exchange_on_rw_i32_is_accepted() {
    let source = translate(
        "mod atomics {
        #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<int>) {
            atomic_compare_exchange(&mut counter[0], 0i32, 1i32);
        }
    }",
    )
    .unwrap();
    assert!(
        source.contains("RWStructuredBuffer<Atomic<int>> counter;"),
        "buffer type must use Atomic<int>: {source}"
    );
    assert!(
        source.contains("counter[0].compareExchange(int(0), int(1));"),
        "atomic_compare_exchange on i32 must lower to .compareExchange(): {source}"
    );
}

#[test]
fn non_atomic_buffer_keeps_plain_element_type() {
    let source = translate(
        "mod plain {
        #[kernel] fn run(id: SV_DispatchThreadID, mut data: RWStructuredBuffer<uint>) {
            data[id.x] = data[id.x] + 1u32;
        }
    }",
    )
    .unwrap();
    assert!(
        source.contains("RWStructuredBuffer<uint> data;"),
        "non-atomic buffer must keep plain element type: {source}"
    );
    assert!(
        !source.contains("Atomic<"),
        "non-atomic buffer must not use Atomic: {source}"
    );
}

#[test]
fn atomic_ops_reject_unsupported_receivers() {
    for (source, code) in [
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, counter: StructuredBuffer<uint>) { atomic_add(&mut counter[0], 1u32); } }",
            Code::AtomicBufferParameter,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<float>) { atomic_add(&mut counter[0], 1.0f32); } }",
            Code::AtomicElementType,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) { let mut local = 0u32; atomic_add(&mut local, 1u32); } }",
            Code::AtomicOperandNotElement,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) { atomic_add(&mut counter, 1u32); } }",
            Code::AtomicOperandNotElement,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) { atomic_add(&counter[0], 1u32); } }",
            Code::AtomicReferenceNotMutable,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) { atomic_add(counter, 1u32); } }",
            Code::AtomicNotReference,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) { atomic_add(&mut counter.items[0], 1u32); } }",
            Code::AtomicReceiverNotBuffer,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) { atomic_add(&mut counter[0]); } }",
            Code::AtomicArgumentCount,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) { atomic_add(&mut counter[0], 1u32, 2u32); } }",
            Code::AtomicArgumentCount,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) { atomic_min(&mut counter[0], 1u32, 2u32); } }",
            Code::AtomicArgumentCount,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) { atomic_compare_exchange(&mut counter[0], 1u32); } }",
            Code::AtomicArgumentCount,
        ),
        (
            "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) { atomic_compare_exchange(&mut counter[0], 1u32, 2u32, 3u32); } }",
            Code::AtomicArgumentCount,
        ),
    ] {
        assert_rejected(source, code);
    }
}

#[test]
fn atomic_min_max_reject_signed_elements() {
    for source in [
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<int>) { atomic_min(&mut counter[0], 1i32); } }",
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<int>) { atomic_max(&mut counter[0], 1i32); } }",
    ] {
        assert_rejected(source, Code::AtomicSignedMinMax);
    }
}

#[test]
fn atomic_ops_reject_unsupported_element_types() {
    for source in [
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<float>) { atomic_add(&mut counter[0], 1.0f32); } }",
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<float>) { atomic_exchange(&mut counter[0], 1.0f32); } }",
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<float>) { atomic_compare_exchange(&mut counter[0], 1.0f32, 2.0f32); } }",
    ] {
        assert_rejected(source, Code::AtomicElementType);
    }
}

#[test]
fn atomic_ops_are_rejected_in_expression_position() {
    for source in [
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) { let old = atomic_add(&mut counter[0], 1u32); } }",
        "mod bad { #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) { if atomic_compare_exchange(&mut counter[0], 0u32, 1u32) { counter[0] = 2u32; } } }",
    ] {
        assert_rejected(source, Code::AtomicAsValue);
    }
}

#[test]
fn atomic_buffers_reject_ordinary_index_access() {
    for source in [
        "mod bad {
            #[kernel] fn run(id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>, mut output: RWStructuredBuffer<uint>) {
                atomic_add(&mut counter[0], 1u32);
                output[id.x] = counter[0];
            }
        }",
        "mod bad {
            #[kernel] fn run(_id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) {
                atomic_add(&mut counter[0], 1u32);
                counter[0] = 0u32;
            }
        }",
    ] {
        assert_rejected(source, Code::AtomicDirectAccess);
    }
}

#[test]
fn atomic_usage_is_isolated_to_the_emitted_kernel() {
    let source = "mod atomics {
        #[kernel] fn run(_id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<uint>) {
            atomic_add(&mut counter[0], 1u32);
        }
        #[kernel] fn reader(id: SV_DispatchThreadID, counter: StructuredBuffer<uint>, mut output: RWStructuredBuffer<uint>) {
            output[id.x] = counter[id.x];
        }
    }";
    let module: syn::ItemMod = crate::diagnostics::testing::parse_str(source).unwrap();
    validate_module(&module).unwrap();
    let reader = module
        .content
        .as_ref()
        .unwrap()
        .1
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(function) if function.sig.ident == "reader" => Some(function),
            _ => None,
        })
        .unwrap();
    let options = kernel_options(reader).unwrap().unwrap();
    let emitted = emit_kernel(&module, reader, options).unwrap().source;
    assert!(
        emitted.contains("StructuredBuffer<uint> counter;"),
        "{emitted}"
    );
    assert!(!emitted.contains("Atomic<uint> counter"), "{emitted}");
}

/// The text of 1-based Slang line `number` of `source`, trimmed.
fn slang_line(source: &str, number: u32) -> &str {
    source.lines().nth(number as usize - 1).unwrap().trim()
}

/// Every range is non-empty and inside the source, records are in emission order,
/// and a record that starts inside an earlier one ends inside it too (outer
/// construct first).
fn assert_table_is_well_formed(emitted: &EmittedKernel, rust_lines: usize) {
    let slang_lines = emitted.source.lines().count() as u32;
    let mut open: Vec<&LineRecord> = Vec::new();
    let mut previous_start = 0;
    for record in &emitted.lines {
        assert!(record.slang_start >= 1, "{record:?}");
        assert!(record.slang_end >= record.slang_start, "{record:?}");
        assert!(record.slang_end <= slang_lines, "{record:?}");
        assert!(record.slang_start >= previous_start, "{record:?}");
        assert!(
            (1..=rust_lines as u32).contains(&record.rust_line),
            "{record:?}"
        );
        previous_start = record.slang_start;
        while open
            .last()
            .is_some_and(|parent| parent.slang_end < record.slang_start)
        {
            open.pop();
        }
        if let Some(parent) = open.last() {
            assert!(
                record.slang_end <= parent.slang_end,
                "{record:?} escapes {parent:?}"
            );
        }
        open.push(record);
    }
}

#[test]
fn line_table_records_statements_helpers_and_structs_at_their_rust_lines() {
    let source = concat!(
        "mod demo {\n",                                                               // 1
        "    struct Pair { first: uint, second: uint }\n",                            // 2
        "    fn twice(x: uint) -> uint {\n",                                          // 3
        "        let y = x + x;\n",                                                   // 4
        "        y\n",                                                                // 5
        "    }\n",                                                                    // 6
        "    #[kernel]\n",                                                            // 7
        "    fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {\n", // 8
        "        let a = twice(id.x);\n",                                             // 9
        "        if a > 1u32 {\n",                                                    // 10
        "            out[a] = 1u32;\n",                                               // 11
        "        } else {\n",                                                         // 12
        "            out[a] = 2u32;\n",                                               // 13
        "        }\n",                                                                // 14
        "        for k in 0u32..4u32 {\n",                                            // 15
        "            out[k] = a;\n",                                                  // 16
        "        }\n",                                                                // 17
        "    }\n",                                                                    // 18
        "}\n",
    );
    let emitted = translate_with_lines(source).unwrap();
    assert_table_is_well_formed(&emitted, 19);
    let described: Vec<(&str, &str, u32)> = emitted
        .lines
        .iter()
        .map(|record| {
            (
                slang_line(&emitted.source, record.slang_start),
                slang_line(&emitted.source, record.slang_end),
                record.rust_line,
            )
        })
        .collect();
    let expected = [
        ("struct Pair", "};", 2),
        ("uint twice(uint x)", "}", 3),
        ("var y = ", "var y = ", 4),
        ("return y;", "return y;", 5),
        ("var a = ", "var a = ", 9),
        // The conditional covers its `else` branch.
        ("if (", "}", 10),
        ("out[a] = ", "out[a] = ", 11),
        ("out[a] = ", "out[a] = ", 13),
        // The loop covers the hoisted end bound and the closing brace.
        ("var __gust_end_", "}", 15),
        ("out[k] = ", "out[k] = ", 16),
    ];
    assert_eq!(
        described.len(),
        expected.len(),
        "one record per statement, helper and struct: {described:#?}\n{}",
        emitted.source
    );
    for ((first, last, rust), (first_prefix, last_prefix, expected_rust)) in
        described.iter().zip(expected)
    {
        assert!(first.starts_with(first_prefix), "{first} vs {first_prefix}");
        assert!(last.starts_with(last_prefix), "{last} vs {last_prefix}");
        assert_eq!(*rust, expected_rust, "{first}");
    }
    // The loop record includes its `for (` header line.
    let for_record = emitted.lines[8];
    assert!(
        (for_record.slang_start..=for_record.slang_end).any(|line| slang_line(
            &emitted.source,
            line
        )
        .starts_with("for ("))
    );
    // The prologue and the marker comment belong to no statement.
    let prologue = emitted
        .source
        .lines()
        .position(|line| line.contains("__gpu_len_out;"))
        .unwrap() as u32
        + 1;
    assert!(
        emitted
            .lines
            .iter()
            .all(|record| !(record.slang_start..=record.slang_end).contains(&prologue))
    );
}

#[test]
fn line_table_records_nested_statements_inside_their_parent() {
    let source = concat!(
        "mod demo {\n",
        "    #[kernel]\n",
        "    fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {\n",
        "        if id.x < out.len() {\n",
        "            let inner = id.x;\n",
        "            if inner > 1u32 {\n",
        "                out[inner] = 1u32;\n",
        "            }\n",
        "        }\n",
        "    }\n",
        "}\n",
    );
    let emitted = translate_with_lines(source).unwrap();
    assert_table_is_well_formed(&emitted, 11);
    let rust: Vec<u32> = emitted
        .lines
        .iter()
        .map(|record| record.rust_line)
        .collect();
    assert_eq!(rust, [4, 5, 6, 7]);
    let outer = emitted.lines[0];
    for nested in &emitted.lines[1..] {
        assert!(outer.slang_start < nested.slang_start && nested.slang_end < outer.slang_end);
    }
}

#[test]
fn line_tables_of_the_reviewed_fixtures_are_well_formed() {
    for (name, source) in [
        (
            "semantics",
            include_str!("../../../../tests/fixtures/semantics.rs"),
        ),
        (
            "numeric",
            include_str!("../../../../tests/fixtures/numeric.rs"),
        ),
        (
            "option",
            include_str!("../../../../tests/fixtures/option.rs"),
        ),
        (
            "result",
            include_str!("../../../../tests/fixtures/result.rs"),
        ),
    ] {
        let emitted = translate_with_lines(source).unwrap();
        assert!(!emitted.lines.is_empty(), "{name}");
        assert_table_is_well_formed(&emitted, source.lines().count());
        // A record's Rust line holds the first token of the statement it came from:
        // never a blank line or a closing brace.
        for record in &emitted.lines {
            let rust = source.lines().nth(record.rust_line as usize - 1).unwrap();
            assert!(
                !rust.trim().is_empty() && !rust.trim().starts_with('}'),
                "{name}: {record:?} -> `{rust}`"
            );
        }
    }
}

/// The items of a macro output, parsed back as a file so assertions look at structure,
/// not at token spacing.
fn output_items(output: proc_macro2::TokenStream) -> Vec<syn::Item> {
    crate::diagnostics::testing::parse2::<syn::File>(output)
        .expect("recovery output must parse as items")
        .items
}

/// Asserts the first item is a `compile_error!`, and, with `Some(code)`, that its message
/// carries `code`.
#[track_caller]
fn assert_leading_compile_error(items: &[syn::Item], code: Option<Code>) {
    let syn::Item::Macro(item) = &items[0] else {
        panic!("first item must be compile_error!, got {:?}", items[0]);
    };
    let name = item.mac.path.segments.last().unwrap();
    assert_eq!(name.ident, "compile_error");
    if let Some(code) = code {
        assert_rendered_code(&item.mac.tokens.to_string(), code);
    }
}

fn recovered_module(items: &[syn::Item]) -> &syn::ItemMod {
    assert_eq!(items.len(), 2, "error plus the module only");
    let syn::Item::Mod(module) = &items[1] else {
        panic!("second item must be the module, got {:?}", items[1]);
    };
    module
}

const REJECTED_MODULE: &str = "mod demo { \
    #[kernel] pub fn run(id: SV_DispatchThreadID) { \
        let value: Result<uint, uint> = Ok(1u32); let x = value.unwrap(); \
    } \
    pub fn helper(x: uint) -> uint { x } \
}";

#[test]
fn rejected_module_is_returned_beside_the_error_without_kernel_attributes() {
    let item: proc_macro2::TokenStream = REJECTED_MODULE.parse().unwrap();
    let items = output_items(crate::gpu_impl(proc_macro2::TokenStream::new(), item));
    assert_leading_compile_error(&items, Some(Code::UnwrapPanics));
    let module = recovered_module(&items);
    assert_eq!(module.ident, "demo");
    let content = &module.content.as_ref().unwrap().1;

    // The prelude import comes first, exactly as in a successful expansion.
    let expected: syn::Item = syn::parse_quote!(
        use ::gust::prelude::*;
    );
    assert_eq!(content[0], expected);
    // The user's functions are all still there, and none keeps #[kernel].
    let functions: Vec<&syn::ItemFn> = content
        .iter()
        .filter_map(|item| match item {
            syn::Item::Fn(function) => Some(function),
            _ => None,
        })
        .collect();
    let names: Vec<String> = functions.iter().map(|f| f.sig.ident.to_string()).collect();
    // The kernel gets the success path's private shadow name; other functions are untouched.
    assert_eq!(names, ["__gpu_typecheck_run", "helper"]);
    assert!(matches!(functions[0].vis, syn::Visibility::Inherited));
    assert!(matches!(functions[1].vis, syn::Visibility::Public(_)));
    for function in functions {
        assert!(
            function
                .attrs
                .iter()
                .all(|attribute| !attribute.path().is_ident("kernel")),
            "{:?}",
            function.attrs
        );
    }
    // Warning noise from the unfinished module is silenced.
    let allow: syn::Attribute = syn::parse_quote! {
        #[allow(dead_code, unused_variables, unused_mut, unused_imports)]
    };
    assert_eq!(module.attrs, [allow]);
}

#[test]
fn recovery_strips_kernel_attributes_from_nested_modules_and_impls() {
    let item: proc_macro2::TokenStream = "mod demo { \
        mod inner { #[kernel] fn deep() {} } \
        struct S; impl S { #[kernel] fn method(&self) {} } \
        #[kernel] struct Marked { x: uint } #[kernel] const LIMIT: uint = 1; \
        #[kernel(workgroup_size(8, 1, 1))] fn run(id: SV_DispatchThreadID) { loop {} } \
    }"
    .parse()
    .unwrap();
    let items = output_items(crate::gpu_impl(proc_macro2::TokenStream::new(), item));
    assert_leading_compile_error(&items, None);
    let text = quote::ToTokens::to_token_stream(recovered_module(&items)).to_string();
    assert!(!text.contains("kernel"), "{text}");
    for name in ["deep", "method", "__gpu_typecheck_run"] {
        assert!(text.contains(&format!("fn {name}")), "{name}: {text}");
    }
    assert!(text.contains("struct Marked"), "{text}");
    assert!(text.contains("const LIMIT"), "{text}");
}

#[test]
fn recovery_keeps_the_users_module_attributes_and_visibility() {
    let item: proc_macro2::TokenStream = "#[doc = \"mine\"] pub(crate) mod demo { \
        #[kernel] fn run(id: SV_DispatchThreadID) { let value: u64 = 1; } \
    }"
    .parse()
    .unwrap();
    let items = output_items(crate::gpu_impl(proc_macro2::TokenStream::new(), item));
    assert_leading_compile_error(&items, None);
    let module = recovered_module(&items);
    let doc: syn::Attribute = syn::parse_quote!(#[doc = "mine"]);
    let allow: syn::Attribute = syn::parse_quote! {
        #[allow(dead_code, unused_variables, unused_mut, unused_imports)]
    };
    assert_eq!(module.attrs, [doc, allow]);
    let visibility: syn::Visibility = syn::parse_quote!(pub(crate));
    assert_eq!(module.vis, visibility);
}

#[test]
fn module_without_inline_content_gets_the_error_alone() {
    // `mod m;` would make rustc look for a file; nothing may be re-emitted.
    for (attribute, code) in [("", Code::InlineModuleRequired), ("x", Code::GpuArguments)] {
        let item: proc_macro2::TokenStream = "mod m;".parse().unwrap();
        let items = output_items(crate::gpu_impl(attribute.parse().unwrap(), item));
        assert_eq!(items.len(), 1, "{items:?}");
        assert_leading_compile_error(&items, Some(code));
    }
}

#[test]
fn attribute_arguments_error_keeps_the_module() {
    let item: proc_macro2::TokenStream = "mod m {}".parse().unwrap();
    let items = output_items(crate::gpu_impl("x".parse().unwrap(), item));
    assert_leading_compile_error(&items, Some(Code::GpuArguments));
    let module = recovered_module(&items);
    assert_eq!(module.ident, "m");
    let expected: syn::Item = syn::parse_quote!(
        use ::gust::prelude::*;
    );
    assert_eq!(module.content.as_ref().unwrap().1, [expected]);
}

#[test]
fn non_module_item_is_returned_unchanged_beside_the_error() {
    let item: proc_macro2::TokenStream = "struct S;".parse().unwrap();
    let items = output_items(crate::gpu_impl(proc_macro2::TokenStream::new(), item));
    assert_eq!(items.len(), 2);
    assert_leading_compile_error(&items, Some(Code::ItemNotModule));
    let expected: syn::Item = syn::parse_quote!(
        struct S;
    );
    assert_eq!(items[1], expected);
}

#[test]
fn unparsable_item_tokens_are_returned_verbatim_after_the_error() {
    // Valid tokens, but not an item: the raw tokens must follow the error untouched.
    let item: proc_macro2::TokenStream = "mod m { fn ( ) } #[kernel] 7".parse().unwrap();
    let output = crate::gpu_impl(proc_macro2::TokenStream::new(), item.clone()).to_string();
    let item = item.to_string();
    assert!(output.starts_with(":: core :: compile_error !"), "{output}");
    assert_rendered_code(&output, Code::ItemNotModule);
    assert!(output.ends_with(&item), "{output}");
}

#[test]
fn accepted_module_is_identical_to_the_expansion_and_has_no_error() {
    let source = "mod demo { \
        #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) { \
            out[id.x] = id.x; \
        } \
    }";
    let item: proc_macro2::TokenStream = source.parse().unwrap();
    let module: syn::ItemMod = crate::diagnostics::testing::parse_str(source).unwrap();
    let expected = crate::expand::expand(module).unwrap().to_string();
    let actual = crate::gpu_impl(proc_macro2::TokenStream::new(), item).to_string();
    assert!(!actual.contains("compile_error"), "{actual}");
    assert_eq!(actual, expected);
}

// ---- typed host API (roadmap M2, slice m2.d1): the generated `call`, `submit` and `dispatch` ----

/// The expansion of an inline module given as source text.
fn expansion(source: &str) -> proc_macro2::TokenStream {
    let module: syn::ItemMod = crate::diagnostics::testing::parse_str(source).unwrap();
    crate::expand::expand(module).unwrap()
}

/// The inherent `impl <kernel>` block that holds the generated typed functions, if any.
fn typed_impl(output: &proc_macro2::TokenStream, kernel: &str) -> Option<syn::ItemImpl> {
    let module: syn::ItemMod =
        crate::diagnostics::parse2(Code::ItemNotModule, output.clone()).unwrap();
    module
        .content
        .unwrap()
        .1
        .into_iter()
        .filter_map(|item| match item {
            syn::Item::Impl(item) if item.trait_.is_none() => Some(item),
            _ => None,
        })
        .find(|item| {
            matches!(item.self_ty.as_ref(), syn::Type::Path(path) if path.path.is_ident(kernel))
                && item
                    .items
                    .iter()
                    .any(|item| matches!(item, syn::ImplItem::Fn(function) if function.sig.ident == "call"))
        })
}

/// One generated function of a kernel's typed impl, as text, and its parameter names.
fn typed_fn(item: &syn::ItemImpl, name: &str) -> (String, Vec<String>) {
    use quote::ToTokens as _;
    let function = item
        .items
        .iter()
        .find_map(|item| match item {
            syn::ImplItem::Fn(function) if function.sig.ident == name => Some(function),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no generated `{name}`"));
    let parameters = function
        .sig
        .inputs
        .iter()
        .map(|input| match input {
            syn::FnArg::Typed(input) => match input.pat.as_ref() {
                syn::Pat::Ident(pattern) => pattern.ident.to_string(),
                other => panic!("unexpected pattern {}", other.to_token_stream()),
            },
            syn::FnArg::Receiver(_) => panic!("generated functions take no receiver"),
        })
        .collect();
    (function.to_token_stream().to_string(), parameters)
}

const VECTOR_ADD: &str = "mod vector_add { \
    #[kernel] \
    pub fn add(id: SV_DispatchThreadID, a: StructuredBuffer<float>, b: StructuredBuffer<float>, \
               mut out: RWStructuredBuffer<float>) { \
        let i = id.x; \
        if i < out.len() { out[i] = a[i] + b[i]; } \
    } \
}";

#[test]
fn a_kernel_gets_call_submit_and_dispatch_with_prefixed_generated_names() {
    let output = expansion(VECTOR_ADD);
    let item = typed_impl(&output, "add").expect("typed impl of `add`");
    let (_, call) = typed_fn(&item, "call");
    let (_, submit) = typed_fn(&item, "submit");
    let (_, dispatch) = typed_fn(&item, "dispatch");
    assert_eq!(call, ["__gust_element_count", "a", "b", "out"]);
    assert_eq!(
        submit,
        ["__gust_device", "__gust_element_count", "a", "b", "out"]
    );
    assert_eq!(
        dispatch,
        ["__gust_device", "__gust_element_count", "a", "b", "out"]
    );
    // The multi-dimensional forms (m7.d): the same three over `gust::Extent` in place of the element count.
    let (_, call_extent) = typed_fn(&item, "call_extent");
    let (_, submit_extent) = typed_fn(&item, "submit_extent");
    let (_, dispatch_extent) = typed_fn(&item, "dispatch_extent");
    assert_eq!(call_extent, ["__gust_extent", "a", "b", "out"]);
    assert_eq!(
        submit_extent,
        ["__gust_device", "__gust_extent", "a", "b", "out"]
    );
    assert_eq!(
        dispatch_extent,
        ["__gust_device", "__gust_extent", "a", "b", "out"]
    );
    assert_eq!(item.items.len(), 6);
}

#[test]
fn the_extent_forms_take_a_gust_extent_and_bind_like_the_one_dimensional_ones() {
    let output = expansion(VECTOR_ADD);
    let item = typed_impl(&output, "add").unwrap();
    let (extent, _) = typed_fn(&item, "call_extent");
    assert!(
        extent.contains("__gust_extent : :: gust :: Extent"),
        "{extent}"
    );
    let (one_dimensional, _) = typed_fn(&item, "call");
    // The same bindings after the count, through `Call::new_extent` instead of `Call::new`.
    let bindings = |function: &str, constructor: &str, first: &str| {
        function
            .split_once(constructor)
            .unwrap_or_else(|| panic!("no {constructor} in {function}"))
            .1
            .replacen(first, "", 1)
    };
    assert_eq!(
        bindings(&extent, "Call :: new_extent", "__gust_extent"),
        bindings(&one_dimensional, "Call :: new", "__gust_element_count")
    );
    for name in ["submit_extent", "dispatch_extent"] {
        let (function, _) = typed_fn(&item, name);
        assert!(
            function.contains("Self :: call_extent (__gust_extent , a , b , out)"),
            "{name}: {function}"
        );
        assert!(function.contains("Run < Self , 3 >"), "{name}: {function}");
    }
}

#[test]
fn call_binds_each_storage_parameter_with_its_access_in_declaration_order() {
    let output = expansion(VECTOR_ADD);
    let item = typed_impl(&output, "add").unwrap();
    let (call, _) = typed_fn(&item, "call");
    let body = call
        .split_once("Call :: new")
        .unwrap_or_else(|| panic!("no Call::new in {call}"))
        .1;
    assert_eq!(
        body,
        " (__gust_element_count , [:: core :: convert :: Into :: into (a) . bind (:: gust :: Access :: ReadOnly) , \
         :: core :: convert :: Into :: into (b) . bind (:: gust :: Access :: ReadOnly) , \
         :: core :: convert :: Into :: into (out) . bind (:: gust :: Access :: ReadWrite)]) }"
    );
    assert!(call.contains(":: gust :: Call < 'r ,"), "{call}");
    assert!(call.contains(", Self , 3 >"), "{call}");
}

#[test]
fn every_bound_names_the_element_type_the_device_and_the_call_lifetime() {
    let output = expansion(VECTOR_ADD);
    let item = typed_impl(&output, "add").unwrap();
    for name in ["call", "submit", "dispatch"] {
        let (function, _) = typed_fn(&item, name);
        // Three resources, three bounds on `float`, and each bound ends in `+ 'r` (S0 decision A).
        assert_eq!(
            function.matches("Resource < Element = float").count(),
            3,
            "{name}"
        );
        assert_eq!(function.matches("> + 'r").count(), 3, "{name}: {function}");
        // The two later arguments must be buffers of the first argument's device type.
        assert_eq!(
            function
                .matches("Device = < __GustR0 as :: gust :: Resource > :: Device")
                .count(),
            2,
            "{name}"
        );
    }
    for name in ["submit", "dispatch"] {
        let (function, _) = typed_fn(&item, name);
        assert!(
            function.contains(":: gust :: Run < Self , 3 >"),
            "{name}: {function}"
        );
    }
}

#[test]
fn submit_and_dispatch_call_the_run_methods_by_path() {
    // Neither `Run` nor `Device` is in the user's module, so a method call would not resolve.
    let output = expansion(VECTOR_ADD);
    let item = typed_impl(&output, "add").unwrap();
    let (submit, _) = typed_fn(&item, "submit");
    let (dispatch, _) = typed_fn(&item, "dispatch");
    assert!(submit.contains(":: submit_call ("), "{submit}");
    assert!(!submit.contains(". submit_call"), "{submit}");
    assert!(dispatch.contains(":: dispatch_call ("), "{dispatch}");
    assert!(!dispatch.contains(". dispatch_call"), "{dispatch}");
}

#[test]
fn the_storage_parameter_count_is_asserted_in_a_non_generic_const() {
    let output = expansion(VECTOR_ADD);
    let module: syn::ItemMod = crate::diagnostics::parse2(Code::ItemNotModule, output).unwrap();
    let constants: Vec<String> = module
        .content
        .unwrap()
        .1
        .into_iter()
        .filter_map(|item| match item {
            syn::Item::Const(item) if item.ident == "_" => {
                assert!(item.generics.params.is_empty());
                Some(quote::ToTokens::to_token_stream(&item).to_string())
            }
            _ => None,
        })
        .collect();
    // `add` has one invocation parameter and three storage parameters. The module also holds one element
    // layout `const _` per storage parameter (m2.d2); those are not this assertion.
    let counts: Vec<&String> = constants
        .iter()
        .filter(|constant| constant.contains("storage_parameter_count"))
        .collect();
    let [constant] = counts.as_slice() else {
        panic!("expected one count assertion in the module: {constants:?}");
    };
    assert!(
        constant.contains(
            "< add as :: gust :: Kernel > :: DESCRIPTOR . storage_parameter_count () + < add as :: gust :: Kernel > :: DESCRIPTOR . uniform_parameter_count () == 3 ,"
        ),
        "{constant}"
    );
}

#[test]
fn a_kernel_without_storage_parameters_takes_the_device_type_as_a_parameter() {
    let output = expansion(
        "mod empty { #[kernel] pub fn nothing(id: SV_DispatchThreadID) { let i = id.x; if i < 1u32 {} } }",
    );
    let item = typed_impl(&output, "nothing").expect("typed impl of `nothing`");
    let (call, parameters) = typed_fn(&item, "call");
    assert_eq!(parameters, ["__gust_element_count"]);
    assert!(
        call.contains(":: gust :: Call < 'r , __GustD , Self , 0 >"),
        "{call}"
    );
    let (dispatch, parameters) = typed_fn(&item, "dispatch");
    assert_eq!(parameters, ["__gust_device", "__gust_element_count"]);
    assert!(
        dispatch.contains(":: gust :: Run < Self , 0 >"),
        "{dispatch}"
    );
    let (call_extent, parameters) = typed_fn(&item, "call_extent");
    assert_eq!(parameters, ["__gust_extent"]);
    assert!(
        call_extent.contains(":: gust :: Call < 'r , __GustD , Self , 0 >"),
        "{call_extent}"
    );
    let (_, parameters) = typed_fn(&item, "submit_extent");
    assert_eq!(parameters, ["__gust_device", "__gust_extent"]);
    let (_, parameters) = typed_fn(&item, "dispatch_extent");
    assert_eq!(parameters, ["__gust_device", "__gust_extent"]);
    assert!(
        output
            .to_string()
            .contains("uniform_parameter_count () == 0 ,")
    );
}

/// A kernel with a uniform struct (four scalars: padding-free in a constant buffer), a storage input and an output.
const UNIFORM_KERNEL: &str = "mod odd { \
    pub struct Settings { pub gain: float, pub bias: float, pub pad0: float, pub pad1: float } \
    #[kernel] \
    pub fn run(id: SV_DispatchThreadID, settings: Uniform<Settings>, input: StructuredBuffer<float>, \
               mut out: RWStructuredBuffer<float>) { \
        let i = id.x; \
        if i < out.len() { out[i] = input[i] * settings.gain + settings.bias; } \
    } \
}";

#[test]
fn a_kernel_with_a_uniform_parameter_gets_a_typed_call_with_one_argument_per_resource() {
    let output = expansion(UNIFORM_KERNEL);
    let item = typed_impl(&output, "run").expect("typed impl of `run`");
    let (call, parameters) = typed_fn(&item, "call");
    // Parameter order, the uniform in the place it was declared.
    assert_eq!(
        parameters,
        ["__gust_element_count", "settings", "input", "out"]
    );
    let (_, parameters) = typed_fn(&item, "submit");
    assert_eq!(
        parameters,
        [
            "__gust_device",
            "__gust_element_count",
            "settings",
            "input",
            "out"
        ]
    );
    // The uniform is bound read-only, with the kind the bound names.
    assert!(
        call.contains(":: gust :: Call :: new (__gust_element_count , [")
            && call.contains("settings) . bind (:: gust :: Access :: ReadOnly)"),
        "{call}"
    );
    let text = output.to_string();
    assert!(text.contains("ParameterKind :: Uniform"));
    // Three resources: the arity of the call and of the descriptor assertion count storage plus uniform.
    assert!(call.contains("Self , 3 >"), "{call}");
}

/// Every generated bound names the kind of its parameter (D57), so a uniform where a buffer is expected, or a buffer
/// where a uniform is, is a compile error. The `tests/ui` case `typed-call-kind-mismatch` is the compile-level half.
#[test]
fn every_generated_bound_names_the_kind_of_its_parameter() {
    let text = expansion(UNIFORM_KERNEL).to_string();
    let bound = |element: &str, kind: &str| {
        format!(
            "Resource < Element = {element} , Device = < __GustR0 as :: gust :: Resource > :: Device , Kind = :: gust :: kind :: {kind} >"
        )
    };
    assert!(
        text.contains("__GustR0 : :: gust :: Resource < Element = Settings , Kind = :: gust :: kind :: Uniform >"),
        "{text}"
    );
    assert!(text.contains(&bound("float", "Storage")), "{text}");
    // Three bounds per function, one function each for `call`, `submit` and `dispatch` and the same three over an
    // extent (`call_extent`, `submit_extent`, `dispatch_extent`): the uniform once, two storage.
    assert_eq!(count(&text, "kind :: Uniform >"), 6, "{text}");
    assert_eq!(count(&text, "kind :: Storage >"), 12, "{text}");
    // No bound without a kind.
    assert_eq!(count(&text, "Resource < Element ="), 18, "{text}");
}

#[test]
fn the_call_arity_assertion_counts_storage_and_uniform_parameters() {
    let text = expansion(UNIFORM_KERNEL).to_string();
    assert!(
        text.contains(
            "< run as :: gust :: Kernel > :: DESCRIPTOR . storage_parameter_count () + < run as :: gust :: Kernel > :: DESCRIPTOR . uniform_parameter_count () == 3 ,"
        ),
        "{text}"
    );
    // A kernel with only a uniform: the count is one, and the first resource is the uniform.
    let only = expansion(
        "mod odd { pub struct Settings { pub gain: float, pub bias: float, pub pad0: float, pub pad1: float } \
         #[kernel] pub fn run(id: SV_DispatchThreadID, settings: Uniform<Settings>) { \
         let i = id.x; if settings.gain > 0.0 && i < 1u32 {} } }",
    )
    .to_string();
    assert!(only.contains("uniform_parameter_count () == 1 ,"), "{only}");
}

#[test]
fn a_plain_value_parameter_never_reaches_the_typed_call() {
    // The emitter rejects it first (GUST-E0801), so `typed_call_items` never sees a `Value` flavour;
    // its guard for that flavour is defensive.
    let module: syn::ItemMod = crate::diagnostics::testing::parse_str(
        "mod odd { #[kernel] pub fn run(id: SV_DispatchThreadID, scale: float, \
         mut out: RWStructuredBuffer<float>) { let i = id.x; if i < out.len() {} } }",
    )
    .unwrap();
    let error = crate::expand::expand(module).unwrap_err();
    assert_code(&error, Code::EntryPointParameter);
}

#[test]
fn user_parameters_named_element_count_and_device_are_kept() {
    let output = expansion(
        "mod clash { #[kernel] pub fn run(id: SV_DispatchThreadID, element_count: StructuredBuffer<float>, \
         mut device: RWStructuredBuffer<float>) { let i = id.x; if i < device.len() {} } }",
    );
    let item = typed_impl(&output, "run").unwrap();
    let (_, call) = typed_fn(&item, "call");
    assert_eq!(call, ["__gust_element_count", "element_count", "device"]);
    let (_, dispatch) = typed_fn(&item, "dispatch");
    assert_eq!(
        dispatch,
        [
            "__gust_device",
            "__gust_element_count",
            "element_count",
            "device"
        ]
    );
}

#[test]
fn the_element_type_is_the_type_written_in_the_module() {
    let output = expansion(
        "mod pool { pub struct Particle { pub x: float } \
         #[kernel] pub fn step(id: SV_DispatchThreadID, input: StructuredBuffer<Particle>, \
         mut out: RWStructuredBuffer<super::Outside>) { let i = id.x; if i < out.len() {} } }",
    );
    let item = typed_impl(&output, "step").unwrap();
    let (call, _) = typed_fn(&item, "call");
    assert!(
        call.contains("Resource < Element = Particle , Kind = :: gust :: kind :: Storage >"),
        "{call}"
    );
    assert!(
        call.contains("Resource < Element = super :: Outside ,"),
        "{call}"
    );
}

#[test]
fn the_handle_and_its_calls_document_the_rules_users_hit() {
    let add = expansion(VECTOR_ADD).to_string();
    assert!(
        add.contains("Handle of the GUST kernel `add`, with the typed host API"),
        "{add}"
    );
    assert!(!add.contains("call::<Device>"), "{add}");
    assert!(!add.contains("(D15)"), "{add}");
    // `u32` is spelled by path, so a user's own `u32` cannot shadow it.
    assert!(
        add.contains("__gust_element_count : :: core :: primitive :: u32"),
        "{add}"
    );

    let nothing = expansion(
        "mod empty { #[kernel] pub fn nothing(id: SV_DispatchThreadID) { let i = id.x; if i < 1u32 {} } }",
    )
    .to_string();
    assert!(nothing.contains("call::<Device>(..)"), "{nothing}");

    let uniform = expansion(UNIFORM_KERNEL).to_string();
    assert!(
        uniform
            .contains("one argument per storage buffer or uniform parameter, in parameter order"),
        "{uniform}"
    );
    assert!(!uniform.contains("has no typed"), "{uniform}");
}

// ---- CPU twin (roadmap M3, slice m3.c): the `impl Run<kernel, N> for gust::cpu::CpuDevice` ----

/// The twin `impl` text opens with this attribute (`proc_macro2` spacing), directly before the impl.
const TWIN_CFG: &str = "# [cfg (any (test , gust_dual))] impl :: gust :: Run <";

/// `text` without its twin `impl`, from the `cfg` attribute to the closing brace of the impl.
fn without_twin(text: &str) -> String {
    let start = text.find(TWIN_CFG).expect("a twin");
    let rest = &text[start..];
    let mut depth = 0_usize;
    for (offset, character) in rest.char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return format!("{}{}", &text[..start], &rest[offset + 1..]);
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced twin in {text}");
}

/// A text with all whitespace removed, for comparing two token streams that differ in spacing only.
fn squeezed(text: &str) -> String {
    text.split_whitespace().collect()
}

#[test]
fn a_storage_kernel_gets_exactly_one_twin_behind_the_dual_cfg() {
    let text = expansion(VECTOR_ADD).to_string();
    assert_eq!(text.matches("impl :: gust :: Run <").count(), 1, "{text}");
    assert_eq!(text.matches(TWIN_CFG).count(), 1, "{text}");
    assert!(
        text.contains(&format!(
            "{TWIN_CFG} add , 3 > for :: gust :: cpu :: CpuDevice"
        )),
        "{text}"
    );
    // The twin is the only place that names the cfg: the host API and the descriptor are not gated.
    assert_eq!(text.matches("any (test , gust_dual)").count(), 1, "{text}");
    assert_eq!(text.matches("cfg (").count(), 1, "{text}");
}

#[test]
fn the_twin_has_the_exact_shape_of_the_hand_written_one_in_gust_cpu() {
    // `crates/gust/src/cpu.rs` (its tests) holds the specification: validate, destructure, acquire, loop,
    // complete; elided lifetimes; the shadow body called as written, once per invocation.
    let text = expansion(VECTOR_ADD).to_string();
    let expected = quote::quote! {
        #[cfg(any(test, gust_dual))]
        impl ::gust::Run<add, 3> for ::gust::cpu::CpuDevice {
            fn submit_call(
                &self,
                __gust_call: &::gust::Call<'_, Self, add, 3>,
            ) -> ::core::result::Result<::gust::cpu::CpuJob, ::gust::cpu::Error> {
                ::gust::cpu::validate(self, __gust_call)?;
                let [a, b, out] = *__gust_call.bindings();
                let a = ::gust::cpu::read::<float>(a, "a")?;
                let b = ::gust::cpu::read::<float>(b, "b")?;
                let mut out = ::gust::cpu::write::<float>(out, "out")?;
                for __gust_id in
                    ::gust::cpu::invocations(__gust_call.kernel(), __gust_call.extent())
                {
                    __gpu_typecheck_add(
                        <SV_DispatchThreadID as ::gust::cpu::FromGlobalId>::from_global_id(__gust_id),
                        ::gust::Storage::new(&a),
                        ::gust::Storage::new(&b),
                        ::gust::StorageMut::new(&mut out)
                    );
                }
                ::core::result::Result::Ok(::gust::cpu::CpuJob::complete())
            }
        }
    }
    .to_string();
    assert!(text.contains(&expected), "{expected}\n---\n{text}");
    // No lifetime parameter of its own (Revision 5), and no downcast in the emitted code.
    assert!(!text.contains("'__gust_d"), "{text}");
    assert!(!text.contains("downcast"), "{text}");
}

#[test]
fn the_twin_acquires_read_only_with_read_and_read_write_with_write() {
    let text = expansion(VECTOR_ADD).to_string();
    assert_eq!(
        text.matches(":: cpu :: read :: < float > (").count(),
        2,
        "{text}"
    );
    assert_eq!(
        text.matches(":: cpu :: write :: < float > (").count(),
        1,
        "{text}"
    );
    assert!(
        text.contains("let mut out = :: gust :: cpu :: write :: < float > (out , \"out\") ?"),
        "{text}"
    );

    let swap = expansion(
        "mod swapping { #[kernel] pub fn swap(id: SV_DispatchThreadID, mut x: RWStructuredBuffer<float>, \
         mut y: RWStructuredBuffer<float>) { let i = id.x; if i < x.len() && i < y.len() {} } }",
    )
    .to_string();
    assert_eq!(swap.matches(":: cpu :: write :: <").count(), 2, "{swap}");
    assert_eq!(swap.matches(":: cpu :: read :: <").count(), 0, "{swap}");
    assert!(swap.contains("impl :: gust :: Run < swap , 2 >"), "{swap}");
}

#[test]
fn the_invocation_argument_is_built_by_from_global_id_for_every_spelling() {
    for spelling in ["SV_DispatchThreadID", "Invocation", "UVec3"] {
        let text = expansion(&format!(
            "mod m {{ #[kernel] pub fn run(id: {spelling}, mut out: RWStructuredBuffer<float>) {{ \
             let i = 0u32; if i < out.len() {{}} }} }}"
        ))
        .to_string();
        let expected = format!(
            "< {spelling} as :: gust :: cpu :: FromGlobalId > :: from_global_id (__gust_id)"
        );
        assert_eq!(text.matches(&expected).count(), 1, "{spelling}: {text}");
    }
}

#[test]
fn twin_arguments_follow_the_parameter_order_and_the_written_element_types() {
    // The invocation parameter is not first, and the element types are the ones written in the module.
    let text = expansion(
        "mod pool { pub struct Particle { pub x: float } \
         #[kernel] pub fn step(input: StructuredBuffer<Particle>, id: SV_DispatchThreadID, \
         mut out: RWStructuredBuffer<super::Outside>) { let i = id.x; if i < out.len() {} } }",
    )
    .to_string();
    assert!(
        text.contains("let input = :: gust :: cpu :: read :: < Particle > (input , \"input\") ?"),
        "{text}"
    );
    assert!(
        text.contains(
            "let mut out = :: gust :: cpu :: write :: < super :: Outside > (out , \"out\") ?"
        ),
        "{text}"
    );
    assert!(
        text.contains(
            "__gpu_typecheck_step (:: gust :: Storage :: new (& input) , \
             < SV_DispatchThreadID as :: gust :: cpu :: FromGlobalId > :: from_global_id (__gust_id) , \
             :: gust :: StorageMut :: new (& mut out))"
        ),
        "{text}"
    );
    assert!(
        text.contains("let [input , out] = * __gust_call . bindings ()"),
        "{text}"
    );
}

#[test]
fn a_kernel_without_storage_parameters_gets_a_twin_over_no_bindings() {
    let text = expansion(
        "mod empty { #[kernel] pub fn nothing(id: SV_DispatchThreadID) { let i = id.x; if i < 1u32 {} } }",
    )
    .to_string();
    assert!(
        text.contains(&format!("{TWIN_CFG} nothing , 0 >")),
        "{text}"
    );
    assert!(
        text.contains("let [] = * __gust_call . bindings ()"),
        "{text}"
    );
}

#[test]
fn a_uniform_kernel_gets_a_twin_that_reads_the_uniform_by_copy() {
    let text = expansion(UNIFORM_KERNEL).to_string();
    // Three bindings, the uniform first; it is read once, by copy, and each invocation gets `Uniform(copy)`.
    assert!(
        text.contains("impl :: gust :: Run < run , 3 > for :: gust :: cpu :: CpuDevice"),
        "{text}"
    );
    assert!(text.contains(TWIN_CFG), "{text}");
    assert!(
        text.contains("let [settings , input , out] = * __gust_call . bindings () ;"),
        "{text}"
    );
    assert!(
        text.contains(
            "let settings = :: gust :: cpu :: read_uniform :: < Settings > (settings , \"settings\") ? ;"
        ),
        "{text}"
    );
    assert!(
        text.contains(
            ":: gust :: cpu :: FromGlobalId > :: from_global_id (__gust_id) , :: gust :: Uniform (settings) , :: gust :: Storage :: new (& input) , :: gust :: StorageMut :: new (& mut out))"
        ),
        "{text}"
    );
    // The opt-out still removes the twin of a kernel with a uniform.
    let opted_out =
        expansion(&UNIFORM_KERNEL.replace("#[kernel]", "#[kernel(cpu = \"unsupported\")]"))
            .to_string();
    assert!(!opted_out.contains("impl :: gust :: Run"), "{opted_out}");
}

// ---- the build profile (roadmap M5, slice m5.e1): `cpu` makes the CPU impl production code ----

/// The expansion of an inline module given as source text, for the build profile `profile`.
fn expansion_for(source: &str, profile: crate::expand::Profile) -> proc_macro2::TokenStream {
    let module: syn::ItemMod = crate::diagnostics::testing::parse_str(source).unwrap();
    crate::expand::expand_with(module, profile).unwrap()
}

/// The impl text without the `cfg` attribute: what the twin looks like when it is ungated.
const UNGATED_IMPL: &str = "impl :: gust :: Run <";

#[test]
fn under_cpu_the_impl_is_ungated_and_no_cfg_copy_exists() {
    let text = expansion_for(VECTOR_ADD, crate::expand::Profile::Cpu).to_string();
    assert_eq!(text.matches(UNGATED_IMPL).count(), 1, "{text}");
    assert!(
        text.contains("impl :: gust :: Run < add , 3 > for :: gust :: cpu :: CpuDevice"),
        "{text}"
    );
    // No gate of any spelling: not on the impl and not on a second copy (the handle doc names the dual cfg in prose).
    assert!(!text.contains("any (test , gust_dual)"), "{text}");
    assert!(!text.contains("any (test"), "{text}");
    assert_eq!(text.matches("cfg (").count(), 0, "{text}");
    assert!(!text.contains(TWIN_CFG), "{text}");
}

#[test]
fn the_ungated_impl_has_the_exact_shape_of_the_gated_one() {
    let text = expansion_for(VECTOR_ADD, crate::expand::Profile::Cpu).to_string();
    let expected = quote::quote! {
        impl ::gust::Run<add, 3> for ::gust::cpu::CpuDevice {
            fn submit_call(
                &self,
                __gust_call: &::gust::Call<'_, Self, add, 3>,
            ) -> ::core::result::Result<::gust::cpu::CpuJob, ::gust::cpu::Error> {
                ::gust::cpu::validate(self, __gust_call)?;
                let [a, b, out] = *__gust_call.bindings();
                let a = ::gust::cpu::read::<float>(a, "a")?;
                let b = ::gust::cpu::read::<float>(b, "b")?;
                let mut out = ::gust::cpu::write::<float>(out, "out")?;
                for __gust_id in
                    ::gust::cpu::invocations(__gust_call.kernel(), __gust_call.extent())
                {
                    __gpu_typecheck_add(
                        <SV_DispatchThreadID as ::gust::cpu::FromGlobalId>::from_global_id(__gust_id),
                        ::gust::Storage::new(&a),
                        ::gust::Storage::new(&b),
                        ::gust::StorageMut::new(&mut out)
                    );
                }
                ::core::result::Result::Ok(::gust::cpu::CpuJob::complete())
            }
        }
    }
    .to_string();
    assert!(text.contains(&expected), "{expected}\n---\n{text}");
}

#[test]
fn the_cpu_expansion_is_the_other_expansion_without_the_gate_and_nothing_else() {
    use crate::expand::Profile;
    let other = expansion_for(VECTOR_ADD, Profile::Other).to_string();
    let cpu = expansion_for(VECTOR_ADD, Profile::Cpu).to_string();
    assert_ne!(other, cpu);
    // Besides the gate, the `cpu` profile is the CPU target (D137, review r1 S1): the descriptor says `Cpu` and there
    // is no `GpuKernel` impl.
    assert_eq!(
        squeezed(&cpu),
        squeezed(&cpu_target_of(&other.replace(TWIN_CFG, UNGATED_IMPL)))
    );
}

#[test]
fn every_value_but_exactly_cpu_keeps_the_gated_twin() {
    use crate::expand::Profile;
    for (value, gated) in [
        (Some("cpu"), false),
        (None, true),
        (Some("dev"), true),
        (Some("native"), true),
        (Some("web"), true),
        (Some("CPU"), true),
        (Some("cpu+simd"), true),
        (Some(""), true),
        (Some("bogus"), true),
    ] {
        let text = expansion_for(VECTOR_ADD, Profile::from_value(value)).to_string();
        assert_eq!(
            text.matches(TWIN_CFG).count(),
            usize::from(gated),
            "{value:?}: {text}"
        );
        assert_eq!(
            text.matches(UNGATED_IMPL).count(),
            1,
            "{value:?}: exactly one impl either way: {text}"
        );
    }
}

#[test]
fn the_entry_point_takes_the_profile_as_an_argument() {
    use crate::expand::Profile;
    let item: proc_macro2::TokenStream = VECTOR_ADD.parse().unwrap();
    let cpu = crate::gpu_impl_with(proc_macro2::TokenStream::new(), item.clone(), Profile::Cpu)
        .to_string();
    let other = crate::gpu_impl(proc_macro2::TokenStream::new(), item.clone()).to_string();
    let explicit =
        crate::gpu_impl_with(proc_macro2::TokenStream::new(), item, Profile::Other).to_string();
    assert!(!cpu.contains("compile_error"), "{cpu}");
    assert_eq!(cpu.matches(TWIN_CFG).count(), 0, "{cpu}");
    assert_eq!(other.matches(TWIN_CFG).count(), 1, "{other}");
    assert_eq!(other, explicit, "`gpu_impl` is the `Other` path");
}

#[test]
fn kernels_without_a_twin_have_none_under_cpu_either() {
    use crate::expand::Profile;
    // `cpu = "unsupported"`: no impl, so a dispatch on `CpuDevice` is the existing E0277 under every profile.
    let opted_out = "mod vector_add { \
        #[kernel(cpu = \"unsupported\")] \
        pub fn add(id: SV_DispatchThreadID, a: StructuredBuffer<float>, b: StructuredBuffer<float>, \
                   mut out: RWStructuredBuffer<float>) { \
            let i = id.x; \
            if i < out.len() { out[i] = a[i] + b[i]; } \
        } \
    }";
    let cpu = expansion_for(opted_out, Profile::Cpu).to_string();
    let other = expansion_for(opted_out, Profile::Other).to_string();
    assert!(!cpu.contains("impl :: gust :: Run"), "{cpu}");
    assert!(!cpu.contains(":: cpu :: CpuDevice"), "{cpu}");
    assert_eq!(squeezed(&cpu), squeezed(&cpu_target_of(&other)));
}

/// `text`, an expansion for a GPU build, as the `cpu` profile has it besides the gate: `Target::Cpu` in the descriptor
/// and no `GpuKernel` impl (the marker is the one empty impl of the expansion).
fn cpu_target_of(text: &str) -> String {
    let text = text.replace("Target :: Gpu", "Target :: Cpu");
    let text = text.replace(":: routed :: OnGpu", ":: routed :: OnCpu");
    let marker = "impl :: gust :: target :: GpuKernel for ";
    let Some(start) = text.find(marker) else {
        return text;
    };
    let end = start + text[start..].find('}').expect("the marker's closing brace") + 1;
    format!("{}{}", &text[..start], &text[end..])
}

/// A uniform parameter is no reason for a kernel to lack a twin since D57: under `cpu` it is the same ungated impl.
#[test]
fn a_uniform_kernel_has_an_ungated_twin_under_cpu() {
    let text = expansion_for(UNIFORM_KERNEL, crate::expand::Profile::Cpu).to_string();
    assert_eq!(text.matches(UNGATED_IMPL).count(), 1, "{text}");
    assert!(
        text.contains("impl :: gust :: Run < run , 3 > for :: gust :: cpu :: CpuDevice"),
        "{text}"
    );
    assert!(
        text.contains(":: gust :: cpu :: read_uniform :: < Settings >"),
        "{text}"
    );
    assert!(!text.contains(TWIN_CFG), "{text}");
}

#[test]
fn a_kernel_without_storage_parameters_gets_an_ungated_impl_over_no_bindings_under_cpu() {
    let text = expansion_for(
        "mod empty { #[kernel] pub fn nothing(id: SV_DispatchThreadID) { let i = id.x; if i < 1u32 {} } }",
        crate::expand::Profile::Cpu,
    )
    .to_string();
    assert!(
        text.contains("impl :: gust :: Run < nothing , 0 >"),
        "{text}"
    );
    assert!(!text.contains("cfg ("), "{text}");
}

#[test]
fn cpu_unsupported_emits_no_twin_and_keeps_the_typed_host_api() {
    let output = expansion(
        "mod vector_add { \
            #[kernel(cpu = \"unsupported\")] \
            pub fn add(id: SV_DispatchThreadID, a: StructuredBuffer<float>, b: StructuredBuffer<float>, \
                       mut out: RWStructuredBuffer<float>) { \
                let i = id.x; \
                if i < out.len() { out[i] = a[i] + b[i]; } \
            } \
        }",
    );
    let text = output.to_string();
    assert!(!text.contains("impl :: gust :: Run"), "{text}");
    assert!(!text.contains("any (test , gust_dual)"), "{text}");
    assert!(!text.contains(":: cpu :: CpuDevice"), "{text}");
    let item = typed_impl(&output, "add").expect("typed impl of `add`");
    for name in ["call", "submit", "dispatch"] {
        let _ = typed_fn(&item, name);
    }
    assert!(text.contains("impl :: gust :: Kernel for add"), "{text}");
    // The option changes nothing else: the same module without it is this one plus the twin.
    assert_eq!(
        squeezed(&text),
        squeezed(&without_twin(&expansion(VECTOR_ADD).to_string()))
    );
}

#[test]
fn cpu_and_workgroup_size_combine_in_either_order() {
    for options in [
        "workgroup_size(8, 1, 1), cpu = \"unsupported\"",
        "cpu = \"unsupported\", workgroup_size(8, 1, 1)",
    ] {
        let text = expansion(&format!(
            "mod m {{ #[kernel({options})] pub fn run(id: SV_DispatchThreadID, \
             mut out: RWStructuredBuffer<float>) {{ let i = id.x; if i < out.len() {{}} }} }}"
        ))
        .to_string();
        assert!(
            text.contains("workgroup_size : [8u32 , 1u32 , 1u32]"),
            "{options}: {text}"
        );
        assert!(!text.contains("impl :: gust :: Run"), "{options}: {text}");
    }
    // `workgroup_size` alone keeps the twin and the written size.
    let text = expansion(
        "mod m { #[kernel(workgroup_size(8, 1, 1))] pub fn run(id: SV_DispatchThreadID, \
         mut out: RWStructuredBuffer<float>) { let i = id.x; if i < out.len() {} } }",
    )
    .to_string();
    assert!(
        text.contains("workgroup_size : [8u32 , 1u32 , 1u32]"),
        "{text}"
    );
    assert!(text.contains("impl :: gust :: Run < run , 1 >"), "{text}");
}

#[test]
fn the_cpu_option_has_one_cause_per_rejection() {
    use crate::diagnostics::testing::assert_codes;
    for (options, code) in [
        ("cpu = \"off\"", Code::UnknownCpuValue),
        ("cpu = \"\"", Code::UnknownCpuValue),
        ("cpu = \"Unsupported\"", Code::UnknownCpuValue),
        (
            "workgroup_size(8, 1, 1), cpu = \"cpu\"",
            Code::UnknownCpuValue,
        ),
        // The first problem is the one reported: a bad value before the repeat.
        (
            "cpu = \"off\", cpu = \"unsupported\"",
            Code::UnknownCpuValue,
        ),
        (
            "cpu = \"unsupported\", cpu = \"unsupported\"",
            Code::DuplicateCpuOption,
        ),
        (
            "cpu = \"unsupported\", cpu = \"off\"",
            Code::DuplicateCpuOption,
        ),
        (
            "cpu = \"unsupported\", workgroup_size(8, 1, 1), cpu = \"unsupported\"",
            Code::DuplicateCpuOption,
        ),
        // A value that is not a string literal, or no value at all, is `syn`'s own failure.
        ("cpu = 1", Code::MalformedKernelAttribute),
        ("cpu", Code::MalformedKernelAttribute),
        // Other options are still unknown.
        (
            "cpu = \"unsupported\", threads(64)",
            Code::UnknownKernelOption,
        ),
    ] {
        let source =
            format!("mod bad {{ #[kernel({options})] fn run(id: SV_DispatchThreadID) {{}} }}");
        let module: syn::ItemMod = crate::diagnostics::testing::parse_str(&source).unwrap();
        let error = crate::expand::expand(module).unwrap_err();
        assert_codes(&error, &[code]);
    }
}

#[test]
fn the_cpu_option_is_checked_by_validation_too() {
    // `validate_module` calls `kernel_options`, so a bad value is reported before the emitter runs.
    let module: syn::ItemMod = crate::diagnostics::testing::parse_str(
        "mod bad { #[kernel(cpu = \"off\")] fn run(id: SV_DispatchThreadID) {} }",
    )
    .unwrap();
    assert_code(
        &validate_module(&module).unwrap_err(),
        Code::UnknownCpuValue,
    );
}

#[test]
fn the_handle_documents_the_twin_and_its_opt_out() {
    let add = expansion(VECTOR_ADD).to_string();
    assert!(
        add.contains(
            "Under `cfg(test)` of the crate that defines the kernel, or `--cfg gust_dual`, the kernel also runs on \
             `gust::cpu::CpuDevice` (its CPU twin), unless `#[kernel(cpu = \\\"unsupported\\\")]`."
        ),
        "{add}"
    );
    // The production CPU target (m5.e1) is named in the same doc.
    assert!(
        add.contains("Under `GUST_PROFILE=cpu` it runs on `gust::cpu::CpuDevice` in every build of the crate."),
        "{add}"
    );
}

#[test]
fn the_module_allows_the_unexpected_cfg_lint_for_the_dual_cfg() {
    let text = expansion(VECTOR_ADD).to_string();
    assert!(
        text.contains("# [allow (dead_code)] # [allow (unexpected_cfgs)] mod vector_add"),
        "{text}"
    );
}

// ---- const layout assertions (roadmap M2, slice m2.d2) ----

/// Two structs with a nested one and a three-field one, so the earlier-field sums have 0, 1 and 2 terms.
const LAYOUT_MODULE: &str = "mod layout { \
    pub struct Pair { pub first: uint, pub second: uint } \
    pub struct Wide { pub pair: Pair, pub extra: float } \
    pub struct Trio { pub a: uint, pub b: float, pub c: int } \
    #[kernel] \
    pub fn run(id: SV_DispatchThreadID, input: StructuredBuffer<Pair>, mut out: RWStructuredBuffer<Wide>) { \
        let i = id.x; \
        if i < out.len() {} \
    } \
}";

/// Occurrences of `needle` in `haystack`.
fn count(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

#[test]
fn every_struct_field_is_asserted_at_its_padding_free_offset() {
    let text = expansion(LAYOUT_MODULE).to_string();
    // Exact token shape: the offset equals the sum of the sizes of the fields before it.
    assert!(text.contains("offset_of ! (Pair , first) == 0 ,"), "{text}");
    assert!(
        text.contains(
            "offset_of ! (Pair , second) == 0 + :: core :: mem :: size_of :: < uint > () ,"
        ),
        "{text}"
    );
    assert!(
        text.contains(
            "offset_of ! (Wide , extra) == 0 + :: core :: mem :: size_of :: < Pair > () ,"
        ),
        "{text}"
    );
    assert!(
        text.contains("offset_of ! (Trio , c) == 0 + :: core :: mem :: size_of :: < uint > () + :: core :: mem :: size_of :: < float > () ,"),
        "{text}"
    );
    // One assert per field (2 + 2 + 3), each naming its field and struct.
    assert_eq!(
        count(&text, "is not at its padding-free offset"),
        7,
        "{text}"
    );
    assert!(
        text.contains(
            "\"GUST StorageV1: field `second` of `Pair` is not at its padding-free offset\""
        ),
        "{text}"
    );
    assert!(
        text.contains("\"GUST StorageV1: field `c` of `Trio` is not at its padding-free offset\""),
        "{text}"
    );
}

#[test]
fn every_struct_is_asserted_to_have_its_largest_field_alignment_and_a_storage_v2_layout() {
    let text = expansion(LAYOUT_MODULE).to_string();
    for name in ["Pair", "Wide", "Trio"] {
        assert!(
            text.contains(&format!(
                "\"GUST StorageV2: `{name}` must have the largest alignment of its fields\""
            )),
            "{name}: {text}"
        );
        let storage = format!(
            "< {name} as :: gust :: GpuPod > :: LAYOUT . is_storage_v2 () , \"GUST StorageV2: `{name}` layout is not StorageV2\""
        );
        assert_eq!(count(&text, &storage), 1, "{name}: {text}");
        // The `is_storage_v2` const reads the `LAYOUT` of the impl, so it follows the impl.
        let implementation = text
            .find(&format!(":: gust :: GpuPod for {name} "))
            .unwrap_or_else(|| panic!("no GpuPod impl for {name}: {text}"));
        assert!(
            text.find(&storage).unwrap() > implementation,
            "{name}: {text}"
        );
    }
    assert_eq!(
        count(&text, "must have the largest alignment of its fields"),
        3,
        "{text}"
    );
    assert_eq!(count(&text, "layout is not StorageV2"), 3, "{text}");
}

#[test]
fn every_storage_parameter_element_is_asserted_against_its_rust_representation() {
    // `add`: three storage parameters of `float`, one assert each (not one per distinct element).
    let text = expansion(VECTOR_ADD).to_string();
    assert_eq!(
        count(&text, "layout does not match its Rust representation"),
        3,
        "{text}"
    );
    // Each reads the recorded layout of its own descriptor entry (`id` is entry 0, so 1, 2 and 3).
    for index in 1..=3 {
        let shape = format!(
            "< add as :: gust :: Kernel > :: DESCRIPTOR . parameters [{index}] . resource_layout {{ \
             :: core :: option :: Option :: Some (__gust_layout) => {{ \
             __gust_layout . element . size as usize == :: core :: mem :: size_of :: < float > () \
             && __gust_layout . element . alignment as usize == :: core :: mem :: align_of :: < float > () \
             && __gust_layout . element_stride as usize == :: core :: mem :: size_of :: < float > () \
             && __gust_layout . element . is_storage_v2 () }} \
             :: core :: option :: Option :: None => false , }} ,"
        );
        assert_eq!(count(&text, &shape), 1, "{index}: {text}");
    }
    // The assertion does not name `GpuPod` again: an element that is no `GpuPod` must report one E0277 only.
    assert!(
        !text.contains("GpuPod > :: LAYOUT . size as usize"),
        "{text}"
    );
    assert!(
        text.contains("\"GUST StorageV1: element `float` layout does not match its Rust representation (parameter `out` of kernel `add`)\""),
        "{text}"
    );

    // Elements keep the path the user wrote, and the two parameters of `run` are told apart.
    let text = expansion(LAYOUT_MODULE).to_string();
    assert!(
        text.contains("parameters [1] . resource_layout { :: core :: option :: Option :: Some (__gust_layout) => { __gust_layout . element . size as usize == :: core :: mem :: size_of :: < Pair > ()"),
        "{text}"
    );
    assert!(
        text.contains("parameters [2] . resource_layout { :: core :: option :: Option :: Some (__gust_layout) => { __gust_layout . element . size as usize == :: core :: mem :: size_of :: < Wide > ()"),
        "{text}"
    );
    assert_eq!(
        count(&text, "layout does not match its Rust representation"),
        2,
        "{text}"
    );
    let outside = expansion(
        "mod pool { #[kernel] pub fn step(id: SV_DispatchThreadID, \
         mut out: RWStructuredBuffer<super::Outside>) { let i = id.x; if i < out.len() {} } }",
    )
    .to_string();
    assert!(
        outside.contains("__gust_layout . element . size as usize == :: core :: mem :: size_of :: < super :: Outside > ()"),
        "{outside}"
    );
    assert!(
        outside.contains("element `super::Outside` layout"),
        "{outside}"
    );
}

#[test]
fn the_element_asserts_are_non_generic_consts_and_kernels_without_storage_have_none() {
    let output = expansion(VECTOR_ADD);
    let module: syn::ItemMod = crate::diagnostics::parse2(Code::ItemNotModule, output).unwrap();
    let elements = module
        .content
        .unwrap()
        .1
        .into_iter()
        .filter(|item| match item {
            syn::Item::Const(item) if item.ident == "_" => {
                assert!(item.generics.params.is_empty());
                quote::ToTokens::to_token_stream(item)
                    .to_string()
                    .contains("Rust representation")
            }
            _ => false,
        })
        .count();
    assert_eq!(elements, 3);

    let none = expansion(
        "mod empty { #[kernel] pub fn nothing(id: SV_DispatchThreadID) { let i = id.x; if i < 1u32 {} } }",
    )
    .to_string();
    assert!(!none.contains("Rust representation"), "{none}");
    assert!(!none.contains("is_storage_v1"), "{none}");
}

#[test]
fn a_kernel_with_a_uniform_parameter_asserts_its_storage_elements_and_its_uniform() {
    let text = expansion(
        "mod odd { pub struct Settings { pub gain: float, pub bias: float, pub pad0: float, pub pad1: float } \
         #[kernel] pub fn run(id: SV_DispatchThreadID, settings: Uniform<Settings>, \
         mut out: RWStructuredBuffer<int>) { let i = id.x; if i < out.len() {} } }",
    )
    .to_string();
    assert_eq!(
        count(&text, "layout does not match its Rust representation"),
        2,
        "{text}"
    );
    // The storage element is parameter 2 and is checked as StorageV2 ...
    assert!(
        text.contains(
            "< run as :: gust :: Kernel > :: DESCRIPTOR . parameters [2] . resource_layout"
        ),
        "{text}"
    );
    assert!(text.contains("size_of :: < int > ()"), "{text}");
    // ... and the uniform, parameter 1, as Std140V1.
    assert!(
        text.contains(
            "< run as :: gust :: Kernel > :: DESCRIPTOR . parameters [1] . resource_layout"
        ),
        "{text}"
    );
    assert!(
        text.contains(
            "matches ! (__gust_layout . rules , :: gust :: ShaderLayoutRules :: Std140V1)"
        ),
        "{text}"
    );
    assert!(text.contains("is_std140_v1 ()"), "{text}");
    assert!(
        text.contains("GUST Std140V1: uniform element `Settings` layout does not match its Rust representation (parameter `settings` of kernel `run`)"),
        "{text}"
    );
    // The descriptor of the uniform is Std140V1 always.
    assert!(
        text.contains("resource_layout : Some (:: gust :: ResourceLayout :: std140_v1 ("),
        "{text}"
    );
}

#[test]
fn an_element_message_with_braces_stays_a_valid_format_string() {
    // `assert!` formats its message; a brace in the element text must be escaped, not interpreted. Such a type
    // fails later (it is no `GpuPod`), but the assertion itself must still parse.
    let text = expansion(
        "mod odd { #[kernel] pub fn run(id: SV_DispatchThreadID, \
         mut out: RWStructuredBuffer<super::Weird<{ 4 }>>) { let i = id.x; if i < out.len() {} } }",
    )
    .to_string();
    assert!(
        text.contains("element `super::Weird<{{ 4 }}>` layout"),
        "{text}"
    );
}

#[test]
fn the_element_assert_covers_everything_the_runtime_checks() {
    // Size, alignment, stride and the whole StorageV2 verdict (an honest element of alignment 8 and size 8 passes the
    // first two and fails the last).
    let text = expansion(VECTOR_ADD).to_string();
    assert_eq!(
        count(
            &text,
            "__gust_layout . element_stride as usize == :: core :: mem :: size_of :: < float > ()"
        ),
        3,
        "{text}"
    );
    assert_eq!(
        count(&text, "__gust_layout . element . is_storage_v2 ()"),
        3,
        "{text}"
    );
}

/// The `mismatch` fixture (known-mismatch ledger probes) has many kernels and no `run`, so its
/// golden holds every kernel's emitted source in declaration order, each under a marker line.
fn translate_every_kernel(source: &str) -> syn::Result<String> {
    let module: syn::ItemMod = crate::diagnostics::testing::parse_str(source)?;
    validate_module(&module)?;
    let mut text = String::new();
    for item in &module.content.as_ref().unwrap().1 {
        let syn::Item::Fn(function) = item else {
            continue;
        };
        let Some(options) = kernel_options(function)? else {
            continue;
        };
        text.push_str(&format!("// ==== kernel {} ====\n", function.sig.ident));
        text.push_str(
            &emit_kernel(&module, function, options)?
                .source
                .replace("\r\n", "\n"),
        );
    }
    Ok(text)
}

#[test]
fn mismatch_golden() {
    let source =
        translate_every_kernel(include_str!("../../../../tests/fixtures/mismatch.rs")).unwrap();
    assert_eq!(
        source.matches("// ==== kernel ").count(),
        15,
        "the mismatch fixture declares 15 kernels"
    );
    assert_golden(
        &source,
        "mismatch.slang",
        &include_str!("../../../../tests/fixtures/mismatch.slang").replace("\r\n", "\n"),
    );
}

/// The build-directory errors (a half artifact pair, an export write, a package name) carry stable codes like every
/// other rejection `#[gpu]` reports (docs/DIAGNOSTICS.md). They are items beside a usable module, not `syn` errors, so
/// the rendered text is asserted.
#[test]
fn build_directory_failures_carry_their_codes() {
    use crate::expand::expand_with_directories;

    const KERNEL: &str = "mod demo { #[kernel] pub fn run(id: SV_DispatchThreadID, \
        mut out: RWStructuredBuffer<float>) { let i = id.x; if i < out.len() {} } }";
    let expand = |manifest: Option<&Path>, export: Option<&Path>, package: Option<&str>| {
        let module: syn::ItemMod = crate::diagnostics::testing::parse_str(KERNEL).unwrap();
        expand_with_directories(module, manifest, export, package)
            .unwrap()
            .to_string()
    };
    let root = env::temp_dir().join(format!("gust-macros-codes-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();

    // A package name that cannot name a directory (E0713).
    let records = root.join("records");
    for package in [None, Some("a/b")] {
        assert_rendered_code(
            &expand(None, Some(&records), package),
            Code::ExportPackageName,
        );
    }
    assert!(!records.exists(), "nothing is written for a refused name");

    // An export directory that is a file (E0712).
    let blocker = root.join("blocker");
    fs::write(&blocker, "a file").unwrap();
    assert_rendered_code(
        &expand(None, Some(&blocker), Some("demo")),
        Code::ExportWriteFailed,
    );

    // Half a pair (E0711), only outside export mode.
    expand(None, Some(&records), Some("demo"));
    let key = fs::read_dir(records.join("demo"))
        .unwrap()
        .filter_map(|entry| {
            let name = entry.unwrap().file_name().into_string().unwrap();
            name.strip_suffix(".slang").map(str::to_owned)
        })
        .next()
        .expect("export mode wrote the key");
    let crate_dir = root.join("crate");
    fs::create_dir_all(crate_dir.join("gust-artifacts")).unwrap();
    fs::write(
        crate_dir.join("gust-artifacts").join(format!("{key}.wgsl")),
        "wgsl",
    )
    .unwrap();
    assert_rendered_code(
        &expand(Some(&crate_dir), None, Some("demo")),
        Code::HalfArtifactPair,
    );
    let repaired = expand(Some(&crate_dir), Some(&records), Some("demo"));
    assert!(!repaired.contains("compile_error"), "{repaired}");
    let _ = fs::remove_dir_all(&root);
}

/// `^` and `^=` lower to the `__gust_xor` overload helpers, because WGSL has no `^` on `bool`. The helpers exist
/// only in a module that uses the operator, and every other bitwise operator on `bool` stays a plain operator.
#[test]
fn bool_xor_lowers_to_overload_helpers() {
    let source = translate(
        "mod m { #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
            let a: bool = out[0] != 0u32;
            let mut b: bool = a ^ true;
            b ^= a;
            let c: bool = a & b | a;
            out[1] = out[2] ^ 3u32;
            out[3] ^= out[4];
        } }",
    )
    .unwrap();
    for helper in [
        "bool __gust_xor(bool a, bool b) { return a != b; }",
        "int __gust_xor(int a, int b) { return a ^ b; }",
        "uint __gust_xor(uint a, uint b) { return a ^ b; }",
        "void __gust_xor_assign(inout bool a, bool b) { a = a != b; }",
        "void __gust_xor_assign(inout int a, int b) { a = a ^ b; }",
        "void __gust_xor_assign(inout uint a, uint b) { a = a ^ b; }",
    ] {
        assert!(source.contains(helper), "missing {helper}\n{source}");
    }
    assert!(source.contains("__gust_xor(a, true)"), "{source}");
    assert!(source.contains("__gust_xor_assign(b, a);"), "{source}");
    assert!(source.contains("__gust_xor(out[2], uint(3))"), "{source}");
    assert!(
        source.contains("__gust_xor_assign(out[3], out[4]);"),
        "{source}"
    );
    assert!(source.contains("((a & b) | a)"), "{source}");
    // A raw `^` must not survive: WGSL rejects it on `bool`.
    let body = source
        .split("void gpu_m_run")
        .nth(1)
        .expect("kernel function emitted");
    assert!(!body.contains(" ^ ") && !body.contains(" ^= "), "{source}");

    let without = translate(
        "mod m { #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
            out[0] = out[1] & out[2] | 4u32;
        } }",
    )
    .unwrap();
    assert!(!without.contains("__gust_xor"), "{without}");

    // The module's only `^` is in a helper, and the helpers must still be emitted.
    let helper_only = translate(
        "mod m {
            fn mix(x: bool, y: bool) -> bool { x ^ y }
            #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
                if mix(true, false) { out[0] = 1u32; }
            }
        }",
    )
    .unwrap();
    assert!(
        helper_only.contains("bool __gust_xor(bool a, bool b) { return a != b; }"),
        "{helper_only}"
    );
    assert!(helper_only.contains("__gust_xor(x, y)"), "{helper_only}");
}

/// A kernel body in a module that has the `both` helper, translated for the `__gust_keep` tests.
fn keep_module(body: &str) -> String {
    translate(&format!(
        "mod m {{
            fn both(x: bool, y: bool) -> bool {{ if x {{ return y; }} return false; }}
            fn all3(x: bool, y: bool, z: bool) -> bool {{ if x {{ if y {{ return z; }} }} return false; }}
            #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {{
                let a: uint = out[0];
                let b: uint = out[1];
                let c: int = out[2] as int;
                let d: int = out[3] as int;
                let t: bool = a == b;
                {body}
            }}
        }}"
    ))
    .unwrap()
}

/// WGSL reads `name < .. >` as a template list and slangc drops the parentheses around a comparison argument, so a
/// `<` comparison that is followed by another element of the same list goes through the gated `__gust_keep` helper
/// (naga's lexer and the fx.wgsl probes). The call arguments, the operands of `^` and the array elements are the lists.
#[test]
fn less_than_list_elements_go_through_keep() {
    let source = keep_module(
        "let r: bool = both(a < b, c > d);
         let s: bool = both((a < b), c >= d);
         let u: bool = all3(t, a < b, c > d);
         let v: bool = (a < b) ^ (c > d);
         let w: bool = both(both(a < b, t), c > d);
         let x: bool = both(a < b && t, c > d);
         let y: bool = both(t && a < b, c > d);
         let z: bool = both(t || (a < b), c > d);
         out[4] = a;",
    );
    assert!(
        source.contains("bool __gust_keep(bool value) { return value; }"),
        "{source}"
    );
    for wrapped in [
        "both(__gust_keep((a < b)), (c > d))",
        "both(__gust_keep(((a < b))), (c >= d))",
        "all3(t, __gust_keep((a < b)), (c > d))",
        "__gust_xor(__gust_keep(((a < b))), ((c > d)))",
        // The inner list is wrapped as well: its `<` is followed by `t`.
        "both(both(__gust_keep((a < b)), t), (c > d))",
    ] {
        assert!(source.contains(wrapped), "missing {wrapped}\n{source}");
    }
    // A `&&` ends a candidate, so it is not a list element that is a `<`.
    assert!(source.contains("both(((a < b) && t), (c > d))"), "{source}");
    // A `<` after a `&&` or `||` stays pending until a later `>`, so the whole element is wrapped.
    for wrapped in [
        "both(__gust_keep((t && (a < b))), (c > d))",
        "both(__gust_keep((t || ((a < b)))), (c > d))",
    ] {
        assert!(source.contains(wrapped), "missing {wrapped}\n{source}");
    }

    // The helper is declared once, before its first use.
    assert_eq!(source.matches("bool __gust_keep(").count(), 1, "{source}");
    assert!(
        source.find("bool __gust_keep(") < source.find("void gpu_m_run"),
        "{source}"
    );
}

/// Elements that cannot open a candidate that a later `>` closes stay as they are.
#[test]
fn keep_is_not_added_where_the_list_cannot_misparse() {
    let source = keep_module(
        "let r1: bool = both(c > d, a < b);
         let r2: bool = both(a <= b, c > d);
         let r3: bool = both(a > b, t);
         let r4: bool = all3(t, c > d, a < b);
         let r5: bool = both(a < b, t);
         let r6: bool = (a < b) & (c > d);
         let r7: bool = (a < b) == (c > d);
         let r8: bool = (a < b) && (c > d);
         let r9: bool = a < b;
         let q: bool = (a >= b) ^ (c <= d);
         if a < b { out[4] = a; }
         out[5] = a;",
    );
    // `r5` has a `<` that another element follows, so it is the only wrapped list.
    assert_eq!(source.matches("__gust_keep(").count(), 2, "{source}");
    assert!(source.contains("both(__gust_keep((a < b)), t)"), "{source}");
    for plain in [
        "both((c > d), (a < b))",
        "both((a <= b), (c > d))",
        "both((a > b), t)",
        "all3(t, (c > d), (a < b))",
        "__gust_xor(((a >= b)), ((c <= d)))",
    ] {
        assert!(source.contains(plain), "missing {plain}\n{source}");
    }
}

/// `__gust_keep` is emitted only in a module that has a list with a wrapped element, like the other `__gust_*`
/// helpers, wherever that list is (here in a helper function, which the kernel calls).
#[test]
fn keep_helper_is_gated_on_use() {
    let without = keep_module(
        "let r: bool = both(c > d, a < b);
         let s: bool = (a < b) == (c > d);
         if both(t, a <= b) { out[4] = a; }
         out[5] = a;",
    );
    assert!(!without.contains("__gust_keep"), "{without}");

    let helper_only = translate(
        "mod m {
            fn wide(a: uint, b: uint, c: int, d: int) -> bool {
                (a < b) ^ (c > d)
            }
            #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
                if wide(out[0], out[1], 1i32, 2i32) { out[2] = 1u32; }
            }
        }",
    )
    .unwrap();
    assert!(
        helper_only.contains("bool __gust_keep(bool value) { return value; }"),
        "{helper_only}"
    );
    assert!(
        helper_only.contains("__gust_xor(__gust_keep(((a < b))), ((c > d)))"),
        "{helper_only}"
    );
    // `^` on two comparisons wants both the xor helpers and the keep helper, in that order.
    assert!(
        helper_only.find("bool __gust_xor(") < helper_only.find("bool __gust_keep("),
        "{helper_only}"
    );

    // `Some`, `Ok`, `Err` and the atomic calls build their own argument lists and never use it.
    let constructors = translate(
        "mod m {
            #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
                let a: uint = out[0];
                let o: Option<bool> = Some(a < 5u32);
                if o.unwrap_or(a > 1u32) { out[1] = 1u32; }
            }
        }",
    )
    .unwrap();
    assert!(!constructors.contains("__gust_keep"), "{constructors}");
}

/// An unsuffixed integer literal above `i32::MAX` is `u32` in the dialect, and slangc types a bare one `int64_t`,
/// which WGSL and SPIR-V validation refuse next to a 32-bit operand. It is spelled `uint(..)`, like a suffixed one.
#[test]
fn large_unsuffixed_integer_literals_are_spelled_uint() {
    let source = translate(
        "mod m { #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
            let x: uint = out[0];
            let n: int = out[1] as int;
            out[0] = x + 3000000000;
            if x == 4294967295 { out[1] = 1u32; }
            out[2] = 2147483648;
            out[3] = 0x8000_0000;
            out[4] = 3000000000u32;
            out[5] = 4294967295;
            if n < 2147483647 { out[6] = 1u32; }
            if n > -2147483648 { out[7] = 1u32; }
            if n < 2147483647i32 { out[8] = 2147483647u32; }
            if x > 0 { out[9] = 5; }
        } }",
    )
    .unwrap();
    for spelled in [
        "(x + uint(3000000000))",
        "(x == uint(4294967295))",
        "out[2] = uint(2147483648);",
        // Hexadecimal digits are normalised by `base10_digits`.
        "out[3] = uint(2147483648);",
        "out[4] = uint(3000000000);",
        "out[5] = uint(4294967295);",
        // The edges: `i32::MAX` and its negation neighbour stay as written.
        "(n < 2147483647)",
        "(n > (-2147483648))",
        "(n < int(2147483647))",
        "out[8] = uint(2147483647);",
        "(x > 0)",
        "out[9] = 5;",
    ] {
        assert!(source.contains(spelled), "missing {spelled}\n{source}");
    }
    assert!(!source.contains("uint(5)"), "{source}");

    // Above `u32::MAX` no 32-bit type fits. The digits are kept, so slangc keeps them 64-bit and shader creation
    // refuses the module (a `uint(..)` would truncate them silently). Reported in the slice notes; no new code.
    let too_large = translate(
        "mod m { #[kernel] fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
            out[0] = out[1] + 5000000000;
        } }",
    )
    .unwrap();
    assert!(too_large.contains("(out[1] + 5000000000)"), "{too_large}");
}

// ---- vector and matrix types (M6, slice m6.m1) ----

/// A module with `items` and one kernel `run` that writes `body` with an output buffer `out` of `float`.
fn vector_module(items: &str, body: &str) -> String {
    format!(
        "mod m {{ {items} #[kernel] pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) {{ \
         let i = id.x; {body} }} }}"
    )
}

#[test]
fn vectors_golden() {
    let source = translate(include_str!("../../../../tests/fixtures/vectors.rs")).unwrap();
    assert_golden(
        &source.replace("\r\n", "\n"),
        "vectors.slang",
        &include_str!("../../../../tests/fixtures/vectors.slang").replace("\r\n", "\n"),
    );
}

#[test]
fn products_golden() {
    let source = translate(include_str!("../../../../tests/fixtures/products.rs")).unwrap();
    assert_golden(
        &source.replace("\r\n", "\n"),
        "products.slang",
        &include_str!("../../../../tests/fixtures/products.slang").replace("\r\n", "\n"),
    );
}

/// The `products` golden holds every row of the table, written in Rust as `a * b` or `a *= b` and lowered by the
/// macro: slangc takes it on both targets (`the_products_golden_compiles_to_wgsl_and_spirv`), so a pair the table lists
/// and Slang cannot resolve fails there. This test reads the golden for one call per row, in table order.
#[test]
fn products_golden_covers_every_row_of_the_table() {
    let golden = include_str!("../../../../tests/fixtures/products.slang").replace("\r\n", "\n");
    let name = |rust: &str| match rust {
        "f32" => "fl",
        "i32" => "sn",
        "u32" => "un",
        "Vec2" => "v2",
        "Vec3" => "v3",
        "Vec4" => "v4",
        "IVec2" => "i2",
        "IVec4" => "i4",
        "UVec2" => "u2",
        "UVec3" => "u3",
        "UVec4" => "u4",
        "Mat2" => "m2",
        "Mat3" => "m3",
        "Mat4" => "m4",
        other => panic!("{other} is not in the table"),
    };
    // The overloads themselves are in the golden, one line per row.
    assert!(golden.contains(&mul_table::overload_text()), "{golden}");
    // Each `a * b` row: a call on the two parameters of the helper, in table order, before the `*=` section.
    let products = &golden[..golden.find("var s0 = ").expect("the `*=` section")];
    let mut after = 0;
    for row in MUL_PAIRS {
        let call = format!("__gust_mul({}, {})", name(row.left), name(row.right));
        let at = products[after..].find(&call).unwrap_or_else(|| {
            panic!(
                "{} * {}: no `{call}` after byte {after}",
                row.left, row.right
            )
        });
        after += at + call.len();
    }
    // Each `a *= b` row: a copy of the left operand, then the assignment through the overload, in table order.
    let mut after = 0;
    let mut assigns = 0;
    for row in MUL_PAIRS.iter().filter(|row| row.assigns()) {
        let copy = format!("var s{assigns} = {};", name(row.left));
        let assignment = format!(
            "(s{assigns} = __gust_mul(s{assigns}, {}));",
            name(row.right)
        );
        for text in [copy, assignment] {
            let at = golden[after..]
                .find(&text)
                .unwrap_or_else(|| panic!("{} *= {}: no `{text}`", row.left, row.right));
            after += at + text.len();
        }
        assigns += 1;
    }
    assert_eq!(assigns, 19);
    assert!(!golden.contains("*="), "{golden}");
}

#[test]
fn every_math_type_maps_to_its_slang_type() {
    for (rust, slang) in [
        ("Vec2", "float2"),
        ("Vec3", "float3"),
        ("Vec4", "float4"),
        ("IVec2", "int2"),
        ("IVec4", "int4"),
        ("UVec2", "uint2"),
        ("UVec3", "uint3"),
        ("UVec4", "uint4"),
        ("Mat2", "float2x2"),
        ("Mat3", "float3x3"),
        ("Mat4", "float4x4"),
        // The raw spelling is the same name.
        ("r#Mat4", "float4x4"),
    ] {
        let source = translate(&vector_module(
            &format!("fn pass(x: {rust}) -> {rust} {{ x }}"),
            "out[i] = 1.0;",
        ))
        .unwrap();
        assert!(
            source.contains(&format!("{slang} pass({slang} x)")),
            "{rust}: {source}"
        );
    }
}

#[test]
fn the_reserved_type_names_are_the_mapped_type_names() {
    use crate::{math_types::is_reserved_type_name, slang::collisions::MAPPED_TYPE_NAMES};
    for name in MAPPED_TYPE_NAMES {
        assert!(is_reserved_type_name(name), "{name}");
    }
    for entry in crate::math_types::MATH_TYPES {
        assert!(MAPPED_TYPE_NAMES.contains(&entry.name), "{}", entry.name);
    }
    assert!(!is_reserved_type_name("Mat5"));
    assert!(!is_reserved_type_name("mat4"));
    assert!(!is_reserved_type_name("Particle"));
}

#[test]
fn a_struct_named_like_a_scalar_vector_or_matrix_is_e0203() {
    for name in [
        "bool", "float", "int", "uint", "Vec2", "Vec3", "Vec4", "IVec2", "IVec4", "UVec2", "UVec3",
        "UVec4", "Mat2", "Mat3", "Mat4", "r#Mat4",
    ] {
        assert_rejected(
            &vector_module(&format!("struct {name} {{ a: f32 }}"), "out[i] = 1.0;"),
            Code::ReservedTypeName,
        );
    }
    // Near misses are ordinary names.
    for name in ["Mat5", "IVec3", "Mat4x4", "vec4"] {
        translate(&vector_module(
            &format!("struct {name} {{ a: f32 }}"),
            "out[i] = 1.0;",
        ))
        .unwrap();
    }
}

#[test]
fn three_wide_types_are_e0204_in_struct_fields_and_resource_elements() {
    for ty in ["Vec3", "UVec3", "Mat3", "r#Vec3"] {
        assert_rejected(
            &vector_module(&format!("struct S {{ a: f32, v: {ty} }}"), "out[i] = 1.0;"),
            Code::ThreeWideInBuffer,
        );
        for resource in [
            "StructuredBuffer",
            "RWStructuredBuffer",
            "Storage",
            "StorageMut",
            "Uniform",
            "ConstantBuffer",
        ] {
            assert_rejected(
                &format!(
                    "mod m {{ #[kernel] pub fn run(id: SV_DispatchThreadID, v: {resource}<{ty}>) {{ \
                     let i = id.x; }} }}"
                ),
                Code::ThreeWideInBuffer,
            );
        }
    }
    // A 3-wide type is fine as a local, a helper parameter and a helper result, and the 4-wide forms are fine
    // in buffers.
    let source = translate(&vector_module(
        "fn lift(v: Vec3, m: Mat3) -> Vec3 { m.mul_vec3(v) } struct S { v: Vec4, w: Vec2, a: f32, b: f32 }",
        "let local = lift(Vec3::splat(1.0), Mat3::IDENTITY); out[i] = local.x;",
    ))
    .unwrap();
    assert!(
        source.contains("float3 lift(float3 v, float3x3 m)"),
        "{source}"
    );
    assert!(source.contains("float4 v;"), "{source}");
}

#[test]
fn implicit_padding_in_a_struct_is_e0208() {
    for (fields, extra) in [
        // Padding before a field.
        ("a: f32, v: Vec2", ""),
        ("a: f32, v: Vec4", ""),
        ("a: f32, b: f32, v: Vec2, c: f32, d: f32, e: f32", ""),
        // Tail padding.
        ("v: Vec2, a: f32", ""),
        ("m: Mat4, a: f32", ""),
        ("m: Mat2, a: f32", ""),
        // Through a nested module struct.
        ("i: Inner, v: Vec2", "struct Inner { a: f32 }"),
        ("v: Vec4, i: Inner", "struct Inner { a: f32 }"),
    ] {
        let source = vector_module(&format!("{extra} struct S {{ {fields} }}"), "out[i] = 1.0;");
        assert_rejected(&source, Code::ImplicitPadding);
    }
    // The message names the struct, the field and both offsets.
    let error = translate(&vector_module(
        "struct S { a: f32, v: Vec2 }",
        "out[i] = 1.0;",
    ))
    .unwrap_err()
    .to_string();
    for needle in ["`v` of `S`", "offset 8", "ends at 4", "4 bytes"] {
        assert!(error.contains(needle), "{needle}: {error}");
    }
    let tail = translate(&vector_module(
        "struct S { v: Vec2, a: f32 }",
        "out[i] = 1.0;",
    ))
    .unwrap_err()
    .to_string();
    assert!(
        tail.contains("ends at byte 12 but its size is 16"),
        "{tail}"
    );

    // Padding-free structs of the same pieces pass, and so does a struct with a field the macro cannot see into
    // (the expansion's const assertions are the check for it).
    for fields in [
        "v: Vec2, a: f32, b: f32",
        "a: f32, b: f32, v: Vec2",
        "v: Vec4, w: Vec4",
        "m: Mat4, v: Vec4",
        "m: Mat2, v: Vec2, a: u32, b: i32",
        "a: f32, b: f32, c: f32, d: f32, v: Vec4",
        "v: Vec2, i: Inner, a: f32",
    ] {
        translate(&vector_module(
            &format!("struct Inner {{ x: f32 }} struct S {{ {fields} }}"),
            "out[i] = 1.0;",
        ))
        .unwrap();
    }
    translate(&vector_module(
        "struct S { a: f32, v: Elsewhere }",
        "out[i] = 1.0;",
    ))
    .unwrap();
}

// ---- uniforms (M6, slice m6.u2) ----

/// A module with `items` and one kernel `run` that reads the uniform `u: Uniform<S>` into `out` of `float`.
fn uniform_module(items: &str, body: &str) -> String {
    format!(
        "mod m {{ {items} #[kernel] pub fn run(id: SV_DispatchThreadID, u: Uniform<S>, \
         mut out: RWStructuredBuffer<float>) {{ let i = id.x; {body} }} }}"
    )
}

#[test]
fn a_uniform_element_that_is_not_a_module_struct_is_e0206() {
    for element in [
        "float",
        "uint",
        "Vec4",
        "Mat4",
        "Elsewhere",
        "super::Outside",
        "r#float",
    ] {
        for wrapper in ["Uniform", "ConstantBuffer"] {
            assert_rejected(
                &format!(
                    "mod m {{ struct S {{ a: f32, b: f32, c: f32, d: f32 }} \
                     #[kernel] pub fn run(id: SV_DispatchThreadID, u: {wrapper}<{element}>, \
                     mut out: RWStructuredBuffer<float>) {{ let i = id.x; }} }}"
                ),
                Code::UniformElement,
            );
        }
    }
    // The message names the fix.
    let message = translate(
        "mod m { #[kernel] pub fn run(id: SV_DispatchThreadID, u: Uniform<float>, \
         mut out: RWStructuredBuffer<float>) { let i = id.x; } }",
    )
    .unwrap_err()
    .to_string();
    assert!(message.contains("`Uniform<Params>`"), "{message}");
    // A struct of the module, in either spelling, is the one thing that works.
    for wrapper in ["Uniform", "ConstantBuffer"] {
        translate(&format!(
            "mod m {{ struct S {{ a: f32, b: f32, c: f32, d: f32 }} \
             #[kernel] pub fn run(id: SV_DispatchThreadID, u: {wrapper}<S>, \
             mut out: RWStructuredBuffer<float>) {{ let i = id.x; out[i] = u.a; }} }}"
        ))
        .unwrap();
    }
}

#[test]
fn a_struct_reachable_from_a_uniform_and_a_storage_element_is_e0207() {
    let four = "a: f32, b: f32, c: f32, d: f32";
    for (items, kernel) in [
        // The same struct as the uniform and as a storage element of one kernel.
        (
            format!("struct S {{ {four} }}"),
            "pub fn run(id: SV_DispatchThreadID, u: Uniform<S>, b: StructuredBuffer<S>, mut out: RWStructuredBuffer<float>) { out[id.x] = u.a; }",
        ),
        // Through a field of the uniform struct.
        (
            format!("struct Inner {{ {four} }} struct S {{ inner: Inner }}"),
            "pub fn run(id: SV_DispatchThreadID, u: Uniform<S>, b: StructuredBuffer<Inner>, mut out: RWStructuredBuffer<float>) { out[id.x] = u.inner.a; }",
        ),
        // Through a field of the storage struct.
        (
            format!(
                "struct Inner {{ {four} }} struct Wrap {{ inner: Inner }} struct S {{ {four} }}"
            ),
            "pub fn run(id: SV_DispatchThreadID, u: Uniform<Inner>, b: RWStructuredBuffer<Wrap>) { }",
        ),
    ] {
        let source = format!("mod m {{ {items} #[kernel] {kernel} }}");
        assert_rejected(&source, Code::UniformAndStorageStruct);
    }
    // Two kernels of one module count: the struct is reachable from a uniform in one and a storage element in another.
    assert_rejected(
        &format!(
            "mod m {{ struct S {{ {four} }} \
             #[kernel] pub fn run(id: SV_DispatchThreadID, u: Uniform<S>, mut out: RWStructuredBuffer<float>) {{ }} \
             #[kernel] pub fn other(id: SV_DispatchThreadID, b: StructuredBuffer<S>, mut out: RWStructuredBuffer<float>) {{ }} }}"
        ),
        Code::UniformAndStorageStruct,
    );
    // The message names the struct and the fix.
    let message = translate(&format!(
        "mod m {{ struct S {{ {four} }} #[kernel] pub fn run(id: SV_DispatchThreadID, u: Uniform<S>, \
         b: StructuredBuffer<S>, mut out: RWStructuredBuffer<float>) {{ }} }}"
    ))
    .unwrap_err()
    .to_string();
    assert!(
        message.contains("`S`") && message.contains("declare a separate struct"),
        "{message}"
    );
    // Separate structs are fine, and so is one struct used by two uniforms.
    translate(&format!(
        "mod m {{ struct S {{ {four} }} struct P {{ {four} }} \
         #[kernel] pub fn run(id: SV_DispatchThreadID, u: Uniform<S>, v: Uniform<S>, b: StructuredBuffer<P>, \
         mut out: RWStructuredBuffer<float>) {{ out[id.x] = u.a + v.b; }} }}"
    ))
    .unwrap();
}

#[test]
fn mat2_and_bool_in_a_struct_a_uniform_reaches_are_e0205() {
    for (items, field) in [
        ("", "m: Mat2, v: Vec2"),
        ("", "v: Vec2, m: Mat2"),
        ("", "a: f32, b: bool, c: f32, d: f32"),
        // Through a nested module struct.
        ("struct Inner { m: Mat2, v: Vec2 }", "i: Inner, v: Vec4"),
        ("struct Inner { m: Mat2, v: Vec2 }", "v: Vec4, i: Inner"),
    ] {
        assert_rejected(
            &uniform_module(&format!("{items} struct S {{ {field} }}"), "out[i] = 1.0;"),
            Code::UniformFieldType,
        );
    }
    let message = translate(&uniform_module(
        "struct S { m: Mat2, v: Vec2 }",
        "out[i] = 1.0;",
    ))
    .unwrap_err()
    .to_string();
    assert!(
        message.contains("`Mat2`") && message.contains("Mat4"),
        "{message}"
    );
    // As a storage element, or in a struct no uniform reaches, a `Mat2` is fine (StorageV2).
    translate(
        "mod m { struct S { m: Mat2, v: Vec2 } #[kernel] pub fn run(id: SV_DispatchThreadID, \
         b: StructuredBuffer<S>, mut out: RWStructuredBuffer<float>) { out[id.x] = b[id.x].v.x; } }",
    )
    .unwrap();
    // A 4x4 matrix is the matrix a uniform may hold.
    translate(&uniform_module(
        "struct S { m: Mat4 }",
        "out[i] = u.m.mul_vec4(Vec4::splat(1.0)).x;",
    ))
    .unwrap();
}

#[test]
fn std140_padding_in_a_struct_a_uniform_reaches_is_e0209() {
    let four = "x: f32, y: f32, z: f32, w: f32";
    for (items, fields) in [
        // A tail: the size rounds up to 16 (a struct with a 16-aligned field is E0208 first: Rust pads it too).
        ("", "a: f32"),
        ("", "a: Vec2"),
        ("", "a: f32, b: f32, c: f32"),
        ("", "a: Vec2, b: Vec2, c: Vec2"),
        // A gap a nested struct causes (it has alignment 16 in a constant buffer, 4 in Rust).
        (
            "struct Inner { x: f32, y: f32, z: f32, w: f32 }",
            "a: f32, inner: Inner",
        ),
        // The nested struct is itself padded.
        (
            "struct Inner { x: f32, y: f32, z: f32 }",
            "inner: Inner, a: f32",
        ),
    ] {
        assert_rejected(
            &uniform_module(&format!("{items} struct S {{ {fields} }}"), "out[i] = 1.0;"),
            Code::UniformPadding,
        );
    }
    // The gap message names the field, the offset and the explicit padding to write.
    let gap = translate(&uniform_module(
        &format!("struct Inner {{ {four} }} struct S {{ a: f32, inner: Inner }}"),
        "out[i] = 1.0;",
    ))
    .unwrap_err()
    .to_string();
    for needle in [
        "`inner` of `S`",
        "byte 16",
        "12 bytes",
        "ends at byte 4",
        "3 `f32` fields",
    ] {
        assert!(gap.contains(needle), "{needle}: {gap}");
    }
    // The tail message names the struct, its rounded size and the padding.
    let tail = translate(&uniform_module("struct S { a: Vec2 }", "out[i] = 1.0;"))
        .unwrap_err()
        .to_string();
    for needle in [
        "`S` ends at byte 8",
        "rounded up to 16 bytes",
        "8 bytes",
        "2 `f32` fields",
    ] {
        assert!(tail.contains(needle), "{needle}: {tail}");
    }
    let one = translate(&uniform_module(
        "struct S { a: f32, b: f32, c: f32 }",
        "out[i] = 1.0;",
    ))
    .unwrap_err()
    .to_string();
    assert!(
        one.contains("4 bytes") && one.contains("one `f32` field"),
        "{one}"
    );

    // Padding-free in a constant buffer: whole 16-byte rows, vectors that pack, a nested struct that fills its rows.
    for (items, fields) in [
        ("", "a: f32, b: f32, c: f32, d: f32"),
        ("", "v: Vec4"),
        ("", "a: Vec2, b: Vec2"),
        ("", "a: Vec2, b: f32, c: f32"),
        ("", "m: Mat4, v: Vec4, a: f32, b: u32, c: i32, d: f32"),
        ("", "a: u32, b: i32, v: IVec2, u: UVec2, c: f32, d: f32"),
        (
            "struct Inner { x: f32, y: f32, z: f32, w: f32 }",
            "inner: Inner, v: Vec4",
        ),
        (
            "struct Inner { x: f32, y: f32, z: f32, w: f32 }",
            "v: Vec4, inner: Inner, a: Vec2, b: Vec2",
        ),
    ] {
        translate(&uniform_module(
            &format!("{items} struct S {{ {fields} }}"),
            "out[i] = 1.0;",
        ))
        .unwrap_or_else(|error| panic!("{items} {fields}: {error}"));
    }
    // The rule is for structs a uniform reaches: the same padded struct is fine as a storage element, and in a
    // struct no kernel parameter reaches.
    translate(
        "mod m { struct P { a: f32 } struct Q { a: f32, b: f32, c: f32 } \
         #[kernel] pub fn run(id: SV_DispatchThreadID, b: StructuredBuffer<P>, mut out: RWStructuredBuffer<float>) { \
         out[id.x] = b[id.x].a; } }",
    )
    .unwrap();
}

#[test]
fn a_uniform_parameter_is_read_one_field_at_a_time_e0321() {
    let ok = "struct S { a: f32, b: f32, c: f32, d: f32 }";
    for body in [
        "let p = u; out[i] = 1.0;",
        "let p = *u; out[i] = 1.0;",
        "out[i] = (*u).a;",
        "out[i] = (u).a + helper(u);",
        "let p = (u); out[i] = 1.0;",
    ] {
        assert_rejected(
            &uniform_module(&format!("{ok} fn helper(s: f32) -> f32 {{ s }}"), body),
            Code::UniformWholeValue,
        );
    }
    // The wrapper's own field `.0` reaches the whole struct (the twin could write through it), and a `mut`
    // parameter would make the copy writable: both are E0321 too.
    for body in [
        "out[i] = u.0.a;",
        "let p = u.0; out[i] = 1.0;",
        "u.0.a = 5.0; out[i] = u.a;",
    ] {
        assert_rejected(&uniform_module(ok, body), Code::UniformWholeValue);
    }
    let mutable = uniform_module(ok, "out[i] = u.a;").replace("u: Uniform<S>", "mut u: Uniform<S>");
    assert!(mutable.contains("mut u: Uniform<S>"), "{mutable}");
    assert_rejected(&mutable, Code::UniformWholeValue);
    // `.len()` of a uniform stays E0306, and a local cannot shadow the parameter.
    assert_rejected(
        &uniform_module(ok, "out[i] = u.len() as f32;"),
        Code::LenReceiver,
    );
    assert_rejected(
        &uniform_module(ok, "let u = 1.0; out[i] = u;"),
        Code::ShadowedParameter,
    );
    // Field reads, nested field reads and method calls on a field are the uses that work.
    let source = translate(&uniform_module(
        "struct Inner { m: Mat4 } struct S { inner: Inner, v: Vec4 }",
        "out[i] = u.inner.m.mul_vec4(u.v).x + (u).v.y;",
    ))
    .unwrap();
    assert!(source.contains("mul(u.inner.m, u.v)"), "{source}");
    assert!(source.contains("(u).v.y"), "{source}");
}

#[test]
fn a_struct_a_uniform_reaches_is_repr_c_align_16_with_std140_asserts() {
    let text = expansion(UNIFORM_KERNEL).to_string();
    assert!(
        text.contains("# [repr (C , align (16))] # [derive (Clone , Copy)] pub struct Settings"),
        "{text}"
    );
    // The model the macro computed is named in each message: offsets 0, 4, 8, 12 and the size 16.
    for (field, offset) in [("gain", 0), ("bias", 4), ("pad0", 8), ("pad1", 12)] {
        assert!(
            text.contains(&format!(
                "GUST Std140V1: field `{field}` of `Settings` is not at byte {offset} of the constant-buffer layout"
            )),
            "{field}: {text}"
        );
    }
    assert!(
        text.contains(
            "GUST Std140V1: `Settings` is not 16 bytes, the size of its constant-buffer layout"
        ),
        "{text}"
    );
    assert!(
        text.contains("GUST Std140V1: `Settings` layout is not Std140V1"),
        "{text}"
    );
    assert!(
        text.contains("GUST Std140V1: `Settings` has padding"),
        "{text}"
    );
    assert!(text.contains("is_std140_v1 ()"), "{text}");
    // The struct is not asserted as StorageV2.
    assert!(
        !text.contains("`Settings` layout is not StorageV2"),
        "{text}"
    );
    // A nested struct is reached through its field and gets the same attribute; a storage struct keeps `repr(C)`.
    let nested = expansion(
        "mod m { struct Inner { x: f32, y: f32, z: f32, w: f32 } struct S { inner: Inner, v: Vec4 } \
         struct P { a: f32, b: f32 } \
         #[kernel] pub fn run(id: SV_DispatchThreadID, u: Uniform<S>, b: StructuredBuffer<P>, \
         mut out: RWStructuredBuffer<float>) { out[id.x] = u.inner.x + b[id.x].a; } }",
    )
    .to_string();
    for name in ["Inner", "S"] {
        assert!(
            nested.contains(&format!(
                "# [repr (C , align (16))] # [derive (Clone , Copy)] struct {name}"
            )),
            "{name}: {nested}"
        );
    }
    assert!(
        nested.contains("# [repr (C)] # [derive (Clone , Copy)] struct P"),
        "{nested}"
    );
    assert!(nested.contains("`P` layout is not StorageV2"), "{nested}");
}

#[test]
fn the_uniforms_golden_expands_with_the_model_offsets() {
    let text = expansion(include_str!("../../../../tests/fixtures/uniforms.rs")).to_string();
    // `Frame`: view 0, tint 64, light 80, offset 112, scale 120, count 124, 128 bytes.
    for (field, offset) in [
        ("view", 0),
        ("tint", 64),
        ("light", 80),
        ("offset", 112),
        ("scale", 120),
        ("count", 124),
    ] {
        assert!(
            text.contains(&format!(
                "field `{field}` of `Frame` is not at byte {offset} of the constant-buffer layout"
            )),
            "{field}: {text}"
        );
    }
    assert!(text.contains("`Frame` is not 128 bytes"), "{text}");
    assert!(text.contains("`Light` is not 32 bytes"), "{text}");
}

#[test]
fn uniforms_golden() {
    let source = translate(include_str!("../../../../tests/fixtures/uniforms.rs")).unwrap();
    assert_golden(
        &source.replace("\r\n", "\n"),
        "uniforms.slang",
        &include_str!("../../../../tests/fixtures/uniforms.slang").replace("\r\n", "\n"),
    );
}

#[test]
fn a_qualified_math_path_is_e0324_and_the_prelude_spellings_are_not() {
    for (items, body) in [
        ("", "let m = super::Mat4::IDENTITY; out[i] = 1.0;"),
        (
            "",
            "let v = gust::math::Vec4::new(1.0, 2.0, 3.0, 4.0); out[i] = v.x;",
        ),
        ("", "let v = crate::Vec4::splat(1.0); out[i] = v.x;"),
        ("", "let v = ::Vec4::splat(1.0); out[i] = v.x;"),
        ("fn f(m: super::Mat4) -> f32 { 1.0 }", "out[i] = 1.0;"),
        ("fn f(m: gust::math::Mat2) -> f32 { 1.0 }", "out[i] = 1.0;"),
        ("struct S { v: super::Vec2, w: Vec2 }", "out[i] = 1.0;"),
        ("", "let c = super::UVec2 { x: 1, y: 2 }; out[i] = 1.0;"),
    ] {
        assert_rejected(&vector_module(items, body), Code::QualifiedMathPath);
    }
    // The prelude form, raw or not, is accepted and emitted as the Slang type.
    let source = translate(&vector_module(
        "fn f(m: r#Mat4) -> Vec4 { r#Mat4::IDENTITY.mul_vec4(r#Vec4::splat(1.0)) }",
        "out[i] = f(Mat4::IDENTITY).x;",
    ))
    .unwrap();
    assert!(source.contains("float4 f(float4x4 m)"), "{source}");
    // `use` inside the module is still E0102: the names come from the prelude only.
    assert_rejected(
        &vector_module("use gust::math::Vec4;", "out[i] = 1.0;"),
        Code::UnsupportedModuleItem,
    );
}

#[test]
fn only_the_listed_associated_items_are_accepted_and_the_rest_is_e0308_or_e0302() {
    // A value path that is not `Mat..::IDENTITY`.
    for body in [
        "let z = Vec4::ZERO; out[i] = 1.0;",
        "let z = Vec2::IDENTITY; out[i] = 1.0;",
        "let z = Mat4::ZERO; out[i] = 1.0;",
        "let z = Vec4::new; out[i] = 1.0;",
        "let z = Mat4::from_cols; out[i] = 1.0;",
        "let z = <Mat4>::IDENTITY; out[i] = 1.0;",
        "let z = f32::MAX; out[i] = 1.0;",
    ] {
        assert_rejected(&vector_module("", body), Code::ForeignPath);
    }
    // A call to an associated function the type does not have.
    for body in [
        "let z = Mat4::new(1.0); out[i] = 1.0;",
        "let z = Vec4::from_cols(Vec4::splat(1.0)); out[i] = 1.0;",
        "let z = Vec4::default(); out[i] = 1.0;",
        "let z = Mat4::IDENTITY(); out[i] = 1.0;",
    ] {
        assert_rejected(&vector_module("", body), Code::UnknownCall);
    }
    // Argument counts are the macro's check as well as rustc's: the lowering needs them.
    for body in [
        "let z = Vec4::new(1.0, 2.0); out[i] = 1.0;",
        "let z = Vec2::new(1.0, 2.0, 3.0); out[i] = 1.0;",
        "let z = Vec4::splat(1.0, 2.0); out[i] = 1.0;",
        "let z = Mat2::from_cols(Vec2::splat(1.0)); out[i] = 1.0;",
        "let z = Mat4::from_rows(Vec4::splat(1.0)); out[i] = 1.0;",
        "let z = Vec3::splat(); out[i] = 1.0;",
    ] {
        assert_rejected(&vector_module("", body), Code::BuiltinArguments);
    }
}

#[test]
fn vector_and_matrix_methods_have_fixed_argument_counts_and_nothing_else_is_a_method() {
    for body in [
        "let v = Vec4::splat(1.0); let w = v.xy(1.0); out[i] = 1.0;",
        "let v = Vec4::splat(1.0); out[i] = v.dot(); ",
        "let v = Vec4::splat(1.0); out[i] = v.dot(v, v);",
        "let m = Mat4::IDENTITY; let t = m.transpose(m); out[i] = 1.0;",
        "let m = Mat4::IDENTITY; let t = m.mul_vec4(); out[i] = 1.0;",
        "let m = Mat4::IDENTITY; let t = m.mul_mat4(m, m); out[i] = 1.0;",
    ] {
        assert_rejected(&vector_module("", body), Code::BuiltinArguments);
    }
    for body in [
        "let v = Vec4::splat(1.0); out[i] = v.x();",
        "let v = Vec4::splat(1.0); let w = v.xyzwx(); out[i] = 1.0;",
        "let v = Vec4::splat(1.0); let w = v.XY(); out[i] = 1.0;",
        "let v = Vec4::splat(1.0); let w = v.xr(); out[i] = 1.0;",
        "let v = Vec4::splat(1.0); out[i] = v.length();",
        "let v = Vec4::splat(1.0); let w = v.normalize(); out[i] = 1.0;",
        "let m = Mat4::IDENTITY; let w = m.inverse(); out[i] = 1.0;",
    ] {
        assert_rejected(&vector_module("", body), Code::UnsupportedMethod);
    }
}

#[test]
fn constructors_swizzles_and_methods_lower_to_their_slang_spelling() {
    let source = translate(&vector_module(
        "",
        "let p = Vec4::new(1.0, 2.0, 3.0, 4.0); \
         let s = Vec2::splat(0.5); \
         let q = p.wzyx(); \
         let r = (p + q).xy(); \
         let d = p.dot(q); \
         let c = Mat2::from_cols(s, r); \
         let w = Mat2::from_rows(s, r); \
         let t = c.transpose(); \
         let v = t.mul_vec2(w.mul_vec2(s)); \
         let m = Mat4::IDENTITY; \
         let n = m.mul_mat4(Mat4::from_cols(p, q, p, q)); \
         let f = n.mul_vec4(p); \
         let g = Mat3::from_rows(Vec3::splat(1.0), Vec3::splat(2.0), Vec3::splat(3.0)); \
         let h = IVec2::new(1, -2); \
         let u = UVec4::splat(7); \
         out[i] = d + v.x + f.w + g.mul_vec3(Vec3::splat(1.0)).y;",
    ))
    .unwrap();
    for expected in [
        "var p = float4(1.0, 2.0, 3.0, 4.0);",
        "var s = float2(0.5);",
        "var q = p.wzyx;",
        "var r = ((p + q)).xy;",
        "var d = __gust_dot(p, q);",
        // Rows build directly; columns build the transpose of the rows constructor.
        "var c = transpose(float2x2(s, r));",
        "var w = float2x2(s, r);",
        "var t = transpose(c);",
        "var v = mul(t, mul(w, s));",
        "var m = float4x4(float4(1.0, 0.0, 0.0, 0.0), float4(0.0, 1.0, 0.0, 0.0), float4(0.0, 0.0, 1.0, 0.0), float4(0.0, 0.0, 0.0, 1.0));",
        "var n = mul(m, transpose(float4x4(p, q, p, q)));",
        "var f = mul(n, p);",
        "var g = float3x3(float3(1.0), float3(2.0), float3(3.0));",
        "var h = int2(1, (-2));",
        "var u = uint4(7);",
        "mul(g, float3(1.0)).y",
    ] {
        assert!(source.contains(expected), "missing `{expected}`:\n{source}");
    }
}

#[test]
fn identity_matrices_are_emitted_at_their_own_width() {
    let source = translate(&vector_module(
        "",
        "let a = Mat2::IDENTITY; let b = Mat3::IDENTITY; out[i] = 1.0;",
    ))
    .unwrap();
    assert!(
        source.contains("var a = float2x2(float2(1.0, 0.0), float2(0.0, 1.0));"),
        "{source}"
    );
    assert!(
        source.contains(
            "var b = float3x3(float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), float3(0.0, 0.0, 1.0));"
        ),
        "{source}"
    );
}

#[test]
fn a_vector_struct_literal_is_a_struct_value_of_the_slang_vector() {
    let source = translate(&vector_module(
        "",
        "let c = UVec2 { x: 3, y: id.x }; let d = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 }; out[i] = d.x + (c.x as f32);",
    ))
    .unwrap();
    for expected in [
        "uint2 __gust_struct_",
        "float4 __gust_struct_",
        ".x = 3;",
        ".w = 4.0;",
    ] {
        assert!(source.contains(expected), "missing `{expected}`:\n{source}");
    }
}

#[test]
fn an_unsuffixed_literal_above_i32_max_next_to_a_vector_is_a_uint() {
    let source = translate(&vector_module(
        "",
        "let a = UVec2::new(3000000000, 1); let b = UVec4::splat(4294967295); out[i] = (a.x + b.y) as f32;",
    ))
    .unwrap();
    assert!(source.contains("uint2(uint(3000000000), 1)"), "{source}");
    assert!(source.contains("uint4(uint(4294967295))"), "{source}");
}

#[test]
fn a_struct_with_vector_fields_gets_storage_v2_asserts_and_its_descriptor_picks_the_rules() {
    let text = expansion(
        "mod m { pub struct Particle { pub position: Vec2, pub velocity: Vec2, pub mass: f32, pub tag: u32 } \
         #[kernel] pub fn run(id: SV_DispatchThreadID, particles: StructuredBuffer<Particle>, \
         points: StructuredBuffer<Vec4>, mut out: RWStructuredBuffer<float>) { let i = id.x; } }",
    )
    .to_string();
    // The struct asserts the largest field alignment, not 4, and the whole V2 verdict.
    assert!(
        text.contains(
            "\"GUST StorageV2: `Particle` must have the largest alignment of its fields\""
        ),
        "{text}"
    );
    assert!(!text.contains("must have alignment 4"), "{text}");
    assert!(
        text.contains("< Particle as :: gust :: GpuPod > :: LAYOUT . is_storage_v2 ()"),
        "{text}"
    );
    // Each element is named once in its descriptor entry, which picks V1 or V2 from the layout itself.
    for element in ["Particle", "Vec4", "float"] {
        let shape = format!(
            "const __GUST_ELEMENT : :: gust :: TypeLayout = < {element} as :: gust :: GpuPod > :: LAYOUT ;"
        );
        assert_eq!(count(&text, &shape), 1, "{element}: {text}");
    }
    assert_eq!(
        count(
            &text,
            "if __GUST_ELEMENT . is_storage_v1 () { :: gust :: ResourceLayout :: storage_v1 (__GUST_ELEMENT) } else { :: gust :: ResourceLayout :: storage_v2 (__GUST_ELEMENT) }"
        ),
        3,
        "{text}"
    );
}

/// The scalar coretests transcription (m5.d3): the fixture the GPU and CPU-twin tests of m5.d1 and m5.d2 run.
#[test]
fn coretests_golden() {
    let source =
        translate_every_kernel(include_str!("../../../../tests/fixtures/coretests.rs")).unwrap();
    assert_eq!(
        source.matches("// ==== kernel ").count(),
        5,
        "the coretests fixture declares 5 kernels"
    );
    assert_golden(
        &source,
        "coretests.slang",
        &include_str!("../../../../tests/fixtures/coretests.slang").replace("\r\n", "\n"),
    );
}

// ---- operator syntax for products (M6, slice m6.m2) ----

use crate::slang::mul_table::{self, Kind, MUL_PAIRS};

/// A module that names a matrix (so `*` is `__gust_mul`) with `body` in the kernel.
fn matrix_module(items: &str, body: &str) -> String {
    vector_module(&format!("fn keep(m: Mat4) -> Mat4 {{ m }} {items}"), body)
}

#[test]
fn the_product_table_has_the_pinned_rows() {
    assert_eq!(MUL_PAIRS.len(), 39);
    let rows_of = |kind: Kind| -> Vec<String> {
        MUL_PAIRS
            .iter()
            .filter(|row| row.kind == kind)
            .map(|row| format!("{} * {} = {}", row.left, row.right, row.output))
            .collect()
    };
    assert_eq!(
        rows_of(Kind::Scalar),
        ["f32 * f32 = f32", "i32 * i32 = i32", "u32 * u32 = u32"]
    );
    assert_eq!(
        rows_of(Kind::MatMul),
        [
            "Mat2 * Mat2 = Mat2",
            "Mat2 * Vec2 = Vec2",
            "Mat3 * Mat3 = Mat3",
            "Mat3 * Vec3 = Vec3",
            "Mat4 * Mat4 = Mat4",
            "Mat4 * Vec4 = Vec4",
        ]
    );
    assert_eq!(rows_of(Kind::Ew).len(), 30);
    // `MulAssign` rows: the output is the left type and the row is neither scalar nor a matrix product.
    let assigns: Vec<String> = MUL_PAIRS
        .iter()
        .filter(|row| row.assigns())
        .map(|row| format!("{} *= {}", row.left, row.right))
        .collect();
    assert_eq!(assigns.len(), 19, "{assigns:?}");
    assert!(assigns.contains(&"Mat4 *= f32".to_owned()));
    assert!(assigns.contains(&"UVec3 *= UVec3".to_owned()));
    assert!(!assigns.iter().any(|row| row.contains("f32 *= ")));
    assert!(
        !assigns
            .iter()
            .any(|row| row.starts_with("Mat") && !row.ends_with("f32"))
    );
}

#[test]
fn the_product_table_is_complete_and_agrees_with_the_math_types() {
    for name in ["f32", "i32", "u32"] {
        assert!(
            MUL_PAIRS
                .iter()
                .any(|row| row.left == name && row.right == name)
        );
    }
    for entry in crate::math_types::MATH_TYPES {
        assert_eq!(mul_table::slang(entry.name), entry.slang, "{}", entry.name);
        assert_eq!(
            mul_table::is_matrix(entry.name),
            entry.matrix,
            "{}",
            entry.name
        );
        assert_eq!(
            mul_table::width(entry.name),
            crate::math_types::width(entry),
            "{}",
            entry.name
        );
        let left = MUL_PAIRS
            .iter()
            .filter(|row| row.left == entry.name)
            .count();
        let right = MUL_PAIRS
            .iter()
            .filter(|row| row.right == entry.name)
            .count();
        // A vector pairs with itself and with its scalar on both sides (2 rows with it on the left, 2 on the right),
        // and a float vector is also the right operand of the matrix of its width; a matrix pairs with itself, its
        // vector and the scalar on the left (3 rows) and with itself and the scalar on the right (2).
        let matrix_partner = usize::from(["Vec2", "Vec3", "Vec4"].contains(&entry.name));
        let (expected_left, expected_right) = if entry.matrix {
            (3, 2)
        } else {
            (2, 2 + matrix_partner)
        };
        assert_eq!(
            (left, right),
            (expected_left, expected_right),
            "{}",
            entry.name
        );
    }
    for row in MUL_PAIRS {
        // The output is the matrix or vector operand; a matrix product has the right operand's type (`m * v` is a
        // vector, `m * m` a matrix).
        let operand = if row.kind == Kind::MatMul || mul_table::is_scalar(row.left) {
            row.right
        } else {
            row.left
        };
        assert_eq!(row.output, operand, "{} * {}", row.left, row.right);
    }
}

#[test]
fn the_overload_text_has_one_line_per_row() {
    let text = mul_table::overload_text();
    assert_eq!(text.lines().filter(|line| !line.is_empty()).count(), 39);
    assert!(text.ends_with("\n\n"));
    for expected in [
        "float __gust_mul(float a, float b) { return a * b; }",
        "uint3 __gust_mul(uint a, uint3 b) { return a * b; }",
        "float4x4 __gust_mul(float4x4 a, float4x4 b) { return mul(a, b); }",
        "float4 __gust_mul(float4x4 a, float4 b) { return mul(a, b); }",
        "float3x3 __gust_mul(float a, float3x3 b) { return a * b; }",
        "float2x2 __gust_mul(float2x2 a, float b) { return a * b; }",
    ] {
        assert!(text.lines().any(|line| line == expected), "{expected}");
    }
    // Only the two matrix products use `mul`.
    assert_eq!(text.matches("mul(a, b)").count(), 6);
}

#[test]
fn a_module_that_names_a_matrix_lowers_every_star_to_the_overload() {
    let source = translate(&matrix_module(
        "",
        "let a = 2.0; let b = a * 3.0; let c = Mat4::IDENTITY * Mat4::IDENTITY; let d = 3 * i; \
         let e = (a * b) * (a * 2.0); out[i * 2] = b * a;",
    ))
    .unwrap();
    for expected in [
        "var b = __gust_mul(a, 3.0);",
        "var c = __gust_mul(float4x4(",
        "var d = __gust_mul(3, i);",
        "var e = __gust_mul((__gust_mul(a, b)), (__gust_mul(a, 2.0)));",
        "out[__gust_mul(i, 2)] = __gust_mul(b, a);",
    ] {
        assert!(source.contains(expected), "missing `{expected}`:\n{source}");
    }
    // The overload table comes once, before the first declaration, and holds every row.
    assert_eq!(
        source.matches("float __gust_mul(float a, float b)").count(),
        1,
        "{source}"
    );
    let table = mul_table::overload_text();
    assert!(source.contains(&table), "{source}");
    assert!(source.find(&table).unwrap() < source.find("[[vk::binding").unwrap());
    // The other operators are untouched.
    let source = translate(&matrix_module("", "out[i] = (1.0 + 2.0) - 3.0 / 4.0;")).unwrap();
    assert!(
        source.contains("(1.0 + 2.0)") && source.contains("(3.0 / 4.0)"),
        "{source}"
    );
    assert!(!source.contains("__gust_mul(1.0"), "{source}");
}

#[test]
fn every_row_of_the_table_lowers_to_its_overload_call() {
    // One parameter per table type, named after it, and one product per non-scalar row.
    let mut names: Vec<&str> = MUL_PAIRS
        .iter()
        .flat_map(|row| [row.left, row.right])
        .collect();
    names.sort_unstable();
    names.dedup();
    let parameters: Vec<String> = names
        .iter()
        .map(|name| format!("p_{name}: {name}"))
        .collect();
    let statements: Vec<String> = MUL_PAIRS
        .iter()
        .enumerate()
        .map(|(index, row)| format!("let r{index} = p_{} * p_{};", row.left, row.right))
        .collect();
    let source = translate(&vector_module(
        &format!(
            "fn all({}) {{ {} }}",
            parameters.join(", "),
            statements.join(" ")
        ),
        "out[i] = 1.0;",
    ))
    .unwrap();
    for (index, row) in MUL_PAIRS.iter().enumerate() {
        let call = format!(
            "var r{index} = __gust_mul(p_{}, p_{});",
            row.left, row.right
        );
        assert!(
            source.contains(&call),
            "{} * {}: missing `{call}`",
            row.left,
            row.right
        );
    }
}

#[test]
fn star_assign_in_a_matrix_module_writes_the_place_through_the_overload() {
    let source = translate(&matrix_module(
        "",
        "let mut v = Vec2::new(1.0, 2.0); v.x *= 2.0; v *= Vec2::splat(2.0); \
         let mut m = Mat4::IDENTITY; m *= 0.5; out[i] *= 2.0; out[i * 2 + 1] *= v.y; \
         let mut a: i32 = 1; a *= 3;",
    ))
    .unwrap();
    for expected in [
        "(v.x = __gust_mul(v.x, 2.0));",
        "(v = __gust_mul(v, float2(2.0)));",
        "(m = __gust_mul(m, 0.5));",
        "(out[i] = __gust_mul(out[i], 2.0));",
        "(out[(__gust_mul(i, 2) + 1)] = __gust_mul(out[(__gust_mul(i, 2) + 1)], v.y));",
        "(a = __gust_mul(a, 3));",
    ] {
        assert!(source.contains(expected), "missing `{expected}`:\n{source}");
    }
    assert!(!source.contains("*="), "{source}");
}

#[test]
fn star_assign_on_a_place_with_a_call_is_e0323_in_a_matrix_module() {
    let items = "fn pick(k: u32) -> u32 { k }";
    for place in ["out[pick(i)]", "out[pick(i) + 1]", "out[(i + pick(1)) * 2]"] {
        let body = format!("{place} *= 2.0;");
        assert_rejected(&matrix_module(items, &body), Code::MulAssignPlace);
        // Without a matrix in the module, the same statement stays verbatim.
        let source = translate(&vector_module(items, &body)).unwrap();
        assert!(source.contains(" *= 2.0);"), "{source}");
    }
    let error = translate(&matrix_module(items, "out[pick(i)] *= 2.0;")).unwrap_err();
    assert!(error.to_string().contains("x = x * y"), "{error}");
    assert!(error.to_string().contains("an `if` or a block"), "{error}");
}

#[test]
fn the_overloads_appear_only_in_modules_that_name_what_they_serve() {
    // The reviewed goldens of modules with no matrix and no integer vector hold neither helper, so their text did not
    // change when the helpers arrived (the golden tests compare them byte for byte).
    for (name, text) in [
        (
            "atomics.slang",
            include_str!("../../../../tests/fixtures/atomics.slang"),
        ),
        (
            "coretests.slang",
            include_str!("../../../../tests/fixtures/coretests.slang"),
        ),
        (
            "loops.slang",
            include_str!("../../../../tests/fixtures/loops.slang"),
        ),
        (
            "mismatch.slang",
            include_str!("../../../../tests/fixtures/mismatch.slang"),
        ),
        (
            "numeric.slang",
            include_str!("../../../../tests/fixtures/numeric.slang"),
        ),
        (
            "option.slang",
            include_str!("../../../../tests/fixtures/option.slang"),
        ),
        (
            "result.slang",
            include_str!("../../../../tests/fixtures/result.slang"),
        ),
        (
            "semantics.slang",
            include_str!("../../../../tests/fixtures/semantics.slang"),
        ),
    ] {
        assert!(!text.contains("__gust_mul"), "{name}");
        assert!(!text.contains("__gust_dot"), "{name}");
        assert!(!text.contains("__gust_neg"), "{name}");
    }
    // The vectors golden names both, and holds both.
    let golden = include_str!("../../../../tests/fixtures/vectors.slang");
    assert!(
        golden
            .replace("\r\n", "\n")
            .contains(&mul_table::overload_text())
    );
    assert!(golden.contains("__gust_dot(") && golden.contains("__gust_mul("));
    assert!(
        golden
            .replace("\r\n", "\n")
            .contains(&neg_table::overload_text())
    );
    assert!(golden.contains("__gust_neg(angle.y)"));
}

#[test]
fn a_matrix_token_anywhere_gates_the_module_and_near_misses_do_not() {
    for items in [
        "fn lift(m: Mat2) -> Mat2 { m }",
        "fn lift(m: Mat3) -> Mat3 { m }",
        "fn lift(m: r#Mat4) -> r#Mat4 { m }",
        "fn lift() -> Vec2 { Mat2::IDENTITY.mul_vec2(Vec2::splat(1.0)) }",
        "struct S { m: Mat4 }",
    ] {
        let source = translate(&vector_module(items, "out[i] = 2.0 * 3.0;")).unwrap();
        assert!(
            source.contains("__gust_mul(2.0, 3.0)"),
            "{items}:\n{source}"
        );
    }
    for items in [
        "struct Mat5 { a: f32 }",
        "struct Mat4x4 { a: f32 }",
        "fn mat4(a: f32) -> f32 { a }",
        "fn lift(v: Vec4) -> Vec4 { v }",
    ] {
        // A near miss is an ordinary name; only the table spellings gate.
        let source = translate(&vector_module(items, "out[i] = 2.0 * 3.0;")).unwrap();
        assert!(!source.contains("__gust_mul"), "{items}:\n{source}");
        assert!(source.contains("(2.0 * 3.0)"), "{items}:\n{source}");
    }
}

#[test]
fn dot_is_the_overload_in_a_module_that_names_an_integer_vector() {
    let body = "let a = IVec2::new(1, 2); let b = UVec4::splat(2); let f = Vec4::splat(1.0); \
                out[i] = (a.dot(a) + b.dot(b) as int) as f32 + f.dot(f);";
    let source = translate(&vector_module("", body)).unwrap();
    for expected in [
        "float __gust_dot(float2 a, float2 b) { return dot(a, b); }",
        "float __gust_dot(float3 a, float3 b) { return dot(a, b); }",
        "float __gust_dot(float4 a, float4 b) { return dot(a, b); }",
        "int __gust_dot(int2 a, int2 b) { return (a.x * b.x + a.y * b.y); }",
        "int __gust_dot(int4 a, int4 b) { return (((a.x * b.x + a.y * b.y) + a.z * b.z) + a.w * b.w); }",
        "uint __gust_dot(uint2 a, uint2 b) { return (a.x * b.x + a.y * b.y); }",
        "uint __gust_dot(uint3 a, uint3 b) { return ((a.x * b.x + a.y * b.y) + a.z * b.z); }",
        "uint __gust_dot(uint4 a, uint4 b) { return (((a.x * b.x + a.y * b.y) + a.z * b.z) + a.w * b.w); }",
        "__gust_dot(a, a)",
        "__gust_dot(b, b)",
        "__gust_dot(f, f)",
    ] {
        assert!(source.contains(expected), "missing `{expected}`:\n{source}");
    }
    // The integer overloads never call Slang's `dot`, which is `OpSDot` or `OpUDot` in SPIR-V.
    for line in source
        .lines()
        .filter(|line| line.contains("__gust_dot(") && line.contains("int"))
    {
        assert!(!line.contains("dot(a, b)"), "{line}");
    }
    // No matrix in the module: `*` stays verbatim.
    assert!(!source.contains("__gust_mul"), "{source}");
    // A module with no `dot` has no helper, whatever vectors it names.
    let source = translate(&vector_module(
        "",
        "let a = IVec2::new(1, 2); out[i] = a.x as f32;",
    ))
    .unwrap();
    assert!(!source.contains("__gust_dot"), "{source}");
}

#[test]
fn every_dot_goes_through_the_overload_even_on_the_invocation_id() {
    let kernel = |parameter: &str, body: &str| {
        format!(
            "mod m {{ #[kernel] pub fn run(id: {parameter}, mut out: RWStructuredBuffer<float>) {{ \
             let i = id.x; {body} }} }}"
        )
    };
    // The kernel's own `UVec3` parameter, and a swizzle of it (a `UVec2` with no `UVec2` token), reach `OpUDot` in
    // SPIR-V unless the module takes the integer overloads.
    for (parameter, body, expected) in [
        ("UVec3", "out[i] = id.dot(id) as f32;", "__gust_dot(id, id)"),
        (
            "SV_DispatchThreadID",
            "out[i] = id.xy().dot(id.yz()) as f32;",
            "__gust_dot(id.xy, id.yz)",
        ),
        (
            "UVec3",
            "let v = Vec2::splat(1.0); out[i] = v.dot(v);",
            "__gust_dot(v, v)",
        ),
    ] {
        let source = translate(&kernel(parameter, body)).unwrap();
        assert!(source.contains(expected), "{body}:\n{source}");
        assert!(
            source.contains("uint __gust_dot(uint3 a, uint3 b)"),
            "{body}:\n{source}"
        );
    }
    // No `dot` token: no helper.
    let source = translate(&kernel("UVec3", "out[i] = 1.0;")).unwrap();
    assert!(!source.contains("__gust_dot"), "{source}");
}

#[test]
fn a_grouped_place_is_as_pure_as_its_contents() {
    // `macro_rules!` hands a `$p:expr` over inside an invisible group. Build one, as the parser would not.
    let group = |text: &str| {
        syn::Expr::Group(syn::ExprGroup {
            attrs: Vec::new(),
            group_token: syn::token::Group::default(),
            expr: Box::new(crate::diagnostics::testing::parse_str(text).unwrap()),
        })
    };
    for text in ["out[i]", "out[i + 1].x", "(out[i])", "v.y"] {
        assert!(crate::slang::is_pure_place(&group(text)), "{text}");
    }
    for text in ["out[pick(i)]", "out[if c { 1 } else { 2 }]", "out[{ 1 }]"] {
        assert!(!crate::slang::is_pure_place(&group(text)), "{text}");
    }
}

#[test]
fn the_generated_impls_are_the_table() {
    let text = crate::mul_impls::mul_impls().to_string();
    // 36 products (the 3 scalar rows are Rust's own) and 19 assignments, every function `#[inline]`.
    assert_eq!(text.matches("impl :: core :: ops :: Mul <").count(), 36);
    assert_eq!(
        text.matches("impl :: core :: ops :: MulAssign <").count(),
        19
    );
    assert_eq!(
        text.matches("fn mul (").count() + text.matches("fn mul_assign (").count(),
        55
    );
    assert_eq!(
        text.matches("# [inline] fn ").count(),
        55,
        "every function is #[inline]"
    );
    for row in MUL_PAIRS {
        let product = format!(
            "impl :: core :: ops :: Mul < {} > for {} {{",
            row.right, row.left
        );
        let assign = format!(
            "impl :: core :: ops :: MulAssign < {} > for {} {{",
            row.right, row.left
        );
        assert_eq!(
            text.contains(&product),
            row.kind != Kind::Scalar,
            "{}",
            product
        );
        assert_eq!(text.contains(&assign), row.assigns(), "{}", assign);
    }
    // Spot checks of the bodies: the products are the methods of the same meaning.
    assert!(text.contains("self . mul_vec4 (rhs)"));
    assert!(text.contains("self . mul_mat2 (rhs)"));
    assert!(text.contains("x : self * rhs . x"));
    assert!(text.contains("x : self . x * rhs . x"));
}

// ---- unary minus on a matrix (M6, slice m6.n) ----

use crate::slang::neg_table::{self, NEG_ROWS, NegKind};

#[test]
fn the_negation_table_has_the_pinned_rows() {
    let operands: Vec<&str> = NEG_ROWS.iter().map(|row| row.operand).collect();
    assert_eq!(
        operands,
        [
            "f32", "i32", "Vec2", "Vec3", "Vec4", "IVec2", "IVec4", "Mat2", "Mat3", "Mat4"
        ]
    );
    for row in NEG_ROWS {
        // Only the matrices are scaled; slangc's own `-a` is the scalar and vector lowering.
        assert_eq!(
            row.kind == NegKind::Scale,
            mul_table::is_matrix(row.operand),
            "{}",
            row.operand
        );
    }
}

/// The rows of the table are the types `gust::math` has a `Neg` for, read from its source (the three float vectors and
/// the two signed integer vectors by `vector_neg!`, every matrix by `matrix_ops!`), plus `f32` and `i32`.
#[test]
fn the_negation_table_covers_every_neg_impl_of_the_math_module() {
    let math = include_str!("../../../gust/src/math.rs").replace("\r\n", "\n");
    let mut negated: Vec<String> = math
        .lines()
        .filter_map(|line| line.trim().strip_prefix("vector_neg!("))
        .map(|rest| rest.split(';').next().unwrap().to_owned())
        .collect();
    assert_eq!(negated, ["Vec2", "Vec3", "Vec4", "IVec2", "IVec4"]);
    assert!(math.contains("impl Neg for $Mat {"));
    negated.extend(
        crate::math_types::MATH_TYPES
            .iter()
            .filter(|entry| entry.matrix)
            .map(|entry| entry.name.to_owned()),
    );
    for name in &negated {
        assert!(
            NEG_ROWS.iter().any(|row| row.operand == name),
            "{name} has a `Neg` impl and no row"
        );
    }
    // The signed integer vectors have an impl and a row; the unsigned ones have neither, because Rust has no unary
    // minus on them.
    assert!(math.contains("vector_neg!(IVec2;") && math.contains("vector_neg!(IVec4;"));
    assert!(!math.contains("vector_neg!(UVec"));
    for row in NEG_ROWS {
        assert!(
            negated.contains(&row.operand.to_owned()) || ["f32", "i32"].contains(&row.operand),
            "{}",
            row.operand
        );
    }
    assert!(
        !NEG_ROWS
            .iter()
            .any(|row| row.operand.starts_with('U') || row.operand == "u32")
    );
}

#[test]
fn the_negation_overloads_are_pinned() {
    let text = neg_table::overload_text();
    assert!(text.ends_with("\n\n"));
    assert_eq!(text.lines().filter(|line| !line.is_empty()).count(), 10);
    let expected = [
        "float __gust_neg(float a) { return -a; }",
        "int __gust_neg(int a) { return -a; }",
        "float2 __gust_neg(float2 a) { return -a; }",
        "float3 __gust_neg(float3 a) { return -a; }",
        "float4 __gust_neg(float4 a) { return -a; }",
        "int2 __gust_neg(int2 a) { return -a; }",
        "int4 __gust_neg(int4 a) { return -a; }",
        "float2x2 __gust_neg(float2x2 a) { return a * -1.0; }",
        "float3x3 __gust_neg(float3x3 a) { return a * -1.0; }",
        "float4x4 __gust_neg(float4x4 a) { return a * -1.0; }",
    ];
    let lines: Vec<&str> = text.lines().filter(|line| !line.is_empty()).collect();
    assert_eq!(lines, expected);
    // `0.0 - m` turns a `+0` lane into `+0` where the twin's negation gives `-0`, and the matrix `-m` is not valid WGSL
    // (naga: "cannot cast a {AbstractInt} to a mat4x4<f32>"); a product with `-1.0` is exact and passes both.
    assert!(!text.contains("0.0 -") && !text.contains("0 -"), "{text}");
    for row in NEG_ROWS.iter().filter(|row| row.kind == NegKind::Scale) {
        assert!(neg_table::overload(row).ends_with("return a * -1.0; }"));
    }
}

#[test]
fn a_module_that_names_a_matrix_lowers_unary_minus_to_the_overload() {
    let source = translate(&matrix_module(
        "",
        "let a = 2.0; let b = -a; let c = -Mat4::IDENTITY; let d = -(-a); let e = -a * 3.0; let f = -2.0; \
         let g: i32 = -3; let h = -(a + 1.0); out[i] = b;",
    ))
    .unwrap();
    for expected in [
        "var b = __gust_neg(a);",
        "var c = __gust_neg(float4x4(",
        "var d = __gust_neg((__gust_neg(a)));",
        "var e = __gust_mul(__gust_neg(a), 3.0);",
        "var h = __gust_neg(((a + 1.0)));",
    ] {
        assert!(source.contains(expected), "missing `{expected}`:\n{source}");
    }
    // A literal operand is a scalar whatever the module names: it keeps the plain `-`. A parenthesised literal is not
    // a literal to the emitter, so it takes the overload, which slangc resolves by the operand type (probed).
    assert!(source.contains("var f = (-2.0);"), "{source}");
    assert!(source.contains("int g = (-3);"), "{source}");
    assert!(!source.contains("__gust_neg(2.0)") && !source.contains("__gust_neg(3)"));
    // The table follows the product table, once, before the first declaration.
    let table = neg_table::overload_text();
    assert_eq!(source.matches(&table).count(), 1, "{source}");
    let products = source.find(&mul_table::overload_text()).unwrap();
    assert!(products + mul_table::overload_text().len() == source.find(&table).unwrap());
    assert!(source.find(&table).unwrap() < source.find("[[vk::binding").unwrap());
    // `!` and the other operators are untouched.
    let source = translate(&matrix_module(
        "",
        "let t = true; let u = !t; out[i] = 2.0 - 3.0;",
    ))
    .unwrap();
    assert!(!source.contains("__gust_neg(t)"), "{source}");
}

/// A parenthesised literal is not a literal to the emitter: it takes the overload, which slangc resolves by the operand
/// type (`-(2147483648)` is the `int` overload, probed in the m6.n review: it wraps to `i32::MIN` as in Rust).
#[test]
fn a_parenthesised_literal_takes_the_overload_in_a_matrix_module() {
    let source = translate(&matrix_module(
        "",
        "let p = -(2.0); let q: i32 = -(2147483648); out[i] = p;",
    ))
    .unwrap();
    assert!(source.contains("var p = __gust_neg((2.0));"), "{source}");
    assert!(
        source.contains("int q = __gust_neg((uint(2147483648)));"),
        "{source}"
    );
}

#[test]
fn unary_minus_is_the_plain_operator_in_a_module_that_names_no_matrix() {
    let source = translate(&vector_module(
        "fn lift(v: Vec4) -> Vec4 { -v }",
        "let a = 2.0; let b = -a; let c = -(a + 1.0); out[i] = b + c;",
    ))
    .unwrap();
    assert!(source.contains("var b = (-a);"), "{source}");
    assert!(source.contains("var c = (-((a + 1.0)));"), "{source}");
    assert!(source.contains("return (-v);"), "{source}");
    assert!(!source.contains("__gust_neg"), "{source}");
    assert!(!source.contains("__gust_mul"), "{source}");
}

#[test]
fn a_matrix_token_gates_unary_minus_and_near_misses_do_not() {
    for items in [
        "fn lift(m: Mat2) -> Mat2 { m }",
        "fn lift(m: r#Mat4) -> r#Mat4 { m }",
        "fn lift(m: Mat3) -> Mat3 { m }",
    ] {
        let source = translate(&vector_module(items, "let a = 2.0; out[i] = -a;")).unwrap();
        assert!(source.contains("= __gust_neg(a);"), "{items}:\n{source}");
    }
    for items in ["struct Mat5 { a: f32 }", "fn lift(v: Vec4) -> Vec4 { v }"] {
        let source = translate(&vector_module(items, "let a = 2.0; out[i] = -a;")).unwrap();
        assert!(source.contains("= (-a);"), "{items}:\n{source}");
        assert!(!source.contains("__gust_neg"), "{items}:\n{source}");
    }
}

#[test]
fn the_products_golden_holds_one_negation_per_row_of_the_table() {
    let golden = include_str!("../../../../tests/fixtures/products.slang").replace("\r\n", "\n");
    assert!(golden.contains(&neg_table::overload_text()), "{golden}");
    let name = |rust: &str| match rust {
        "f32" => "fl",
        "i32" => "sn",
        "Vec2" => "v2",
        "Vec3" => "v3",
        "Vec4" => "v4",
        "IVec2" => "i2",
        "IVec4" => "i4",
        "Mat2" => "m2",
        "Mat3" => "m3",
        "Mat4" => "m4",
        other => panic!("{other} is not in the table"),
    };
    // One `__gust_neg(<parameter>)` per row, in table order, after the table and after the literal products.
    let mut after =
        golden.find(&neg_table::overload_text()).unwrap() + neg_table::overload_text().len();
    for row in NEG_ROWS {
        let call = format!("__gust_neg({})", name(row.operand));
        let at = golden[after..]
            .find(&call)
            .unwrap_or_else(|| panic!("-{}: no `{call}` after byte {after}", row.operand));
        after += at + call.len();
    }
}

#[test]
fn wrapping_methods_lower_to_the_plain_operators_even_in_a_matrix_module() {
    // The macro sees no types, so `wrapping_neg` is `0 - a` (WGSL has no negation of a `u32`; slangc types the literal
    // as the operand's own type). A plain `*` in the same module goes through the overloads, the wrapping form never.
    let source = translate(&matrix_module(
        "",
        "let a = i.wrapping_mul(7u32); let b = i.wrapping_add(a).wrapping_sub(1u32); let c = a.wrapping_neg(); \
         let d = a * b; out[i] = 1.0;",
    ))
    .unwrap();
    for expected in [
        "var a = (i * uint(7));",
        "var b = ((i + a) - uint(1));",
        "var c = (0 - a);",
        "var d = __gust_mul(a, b);",
    ] {
        assert!(source.contains(expected), "missing `{expected}`:\n{source}");
    }
    // Without a matrix in the module, the same methods give the same text.
    let source = translate(&vector_module(
        "",
        "let a = i.wrapping_mul(7u32); let c = a.wrapping_neg().wrapping_add(2u32); out[i] = 1.0;",
    ))
    .unwrap();
    for expected in ["var a = (i * uint(7));", "var c = ((0 - a) + uint(2));"] {
        assert!(source.contains(expected), "missing `{expected}`:\n{source}");
    }
    assert!(!source.contains("__gust_"), "{source}");
}

#[test]
fn wrapping_methods_are_checked_by_name_and_argument_count() {
    for body in [
        "let y = x.wrapping_add(1u32);",
        "let y = x.wrapping_sub(out[0u32]);",
        "let y = x.wrapping_mul(x).wrapping_mul(3u32);",
        "let y = x.wrapping_neg();",
        "let y = 5u32.wrapping_neg().wrapping_add(x);",
    ] {
        translate(&kernel_with_body(body)).unwrap_or_else(|error| panic!("{body}: {error}"));
    }
    // A wrong argument count is E0304, like the vector and matrix methods.
    for body in [
        "let y = x.wrapping_add();",
        "let y = x.wrapping_add(1u32, 2u32);",
        "let y = x.wrapping_neg(1u32);",
        "let y = x.wrapping_mul();",
    ] {
        assert_rejected(&kernel_with_body(body), Code::BuiltinArguments);
    }
    assert_rejected(
        &kernel_with_body("let y = x.wrapping_add::<u32>(1u32);"),
        Code::BuiltinTypeArguments,
    );
    // Division has no wrapped result for a zero divisor, so neither method exists; nor does anything else with
    // arguments. With no arguments the old code stays.
    for body in [
        "let y = x.wrapping_div(1u32);",
        "let y = x.wrapping_rem(1u32);",
        "let y = x.pow(2u32);",
        "let y = x.min(3u32);",
        "let y = x.checked_add(1u32);",
        "let y = x.saturating_sub(1u32);",
    ] {
        assert_rejected(&kernel_with_body(body), Code::MethodNotInDialect);
    }
    for body in [
        "let y = x.abs();",
        "let y = x.wrapping_abs();",
        "let y = x.count_ones();",
    ] {
        assert_rejected(&kernel_with_body(body), Code::UnsupportedMethod);
    }
    // A built-in with arguments is still E0304, not "not in the dialect".
    assert_rejected(
        &kernel_with_body("let y = out.len(1u32);"),
        Code::BuiltinArguments,
    );
    assert_rejected(
        &kernel_with_body("let y = id.global_id(1u32);"),
        Code::BuiltinArguments,
    );
}

#[test]
fn the_not_in_the_dialect_text_lists_the_accepted_methods() {
    let error = translate(&kernel_with_body("let y = x.wrapping_div(2u32);")).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("[GUST-E0325]"), "{text}");
    assert!(
        text.contains("`.wrapping_div(..)` is not in the dialect"),
        "{text}"
    );
    for accepted in [
        ".global_id()",
        ".len()",
        ".is_some()",
        ".unwrap_or(default)",
        ".wrapping_add(b)",
        ".wrapping_sub(b)",
        ".wrapping_mul(b)",
        ".wrapping_neg()",
        ".dot(v)",
        ".transpose()",
    ] {
        assert!(text.contains(accepted), "missing `{accepted}`: {text}");
    }
    // The old text said built-ins "do not accept arguments"; that is no longer the message for a non-built-in.
    assert!(!text.contains("do not accept arguments"), "{text}");
    // E0305 lists the same methods.
    let text = translate(&kernel_with_body("let y = x.abs();"))
        .unwrap_err()
        .to_string();
    assert!(
        text.contains("[GUST-E0305]") && text.contains(".wrapping_mul(b)"),
        "{text}"
    );
}

/// The quoted names in the `match` arms of one function of a source file, found by the function's signature.
fn names_in_fn(source: &str, signature: &str) -> Vec<String> {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("no `{signature}`"));
    let body = &source[start..];
    let end = body.find("\n}").expect("the function ends");
    let mut names = Vec::new();
    for line in body[..end].lines().skip(1) {
        let mut parts = line.split('"');
        while let (Some(_), Some(name)) = (parts.next(), parts.next()) {
            names.push(name.to_string());
        }
    }
    assert!(!names.is_empty(), "`{signature}` names nothing");
    names
}

#[test]
fn every_method_table_name_has_an_emitter_arm_and_is_in_the_accepted_text() {
    let validate = include_str!("../validate/mod.rs");
    let math = include_str!("../math_types.rs");
    let accepted = translate(&kernel_with_body("let y = x.abs();"))
        .unwrap_err()
        .to_string();
    let wrapping = names_in_fn(validate, "pub(crate) fn wrapping_arity");
    assert!(wrapping.len() >= 4, "{wrapping:?}");
    for name in &wrapping {
        // `wrapping_arity` knows the name, so the emitter must lower it: an arm missing there would panic.
        let arity = crate::validate::wrapping_arity(name).expect("a table name has an arity");
        let arguments = if arity == 0 { "" } else { "1u32" };
        let body = format!("let y = x.{name}({arguments});");
        let source =
            translate(&kernel_with_body(&body)).unwrap_or_else(|error| panic!("{body}: {error}"));
        assert!(source.contains("var y = ("), "{name}: {source}");
        assert!(
            accepted.contains(&format!(".{name}(")),
            "`.{name}(` is missing from the accepted text: {accepted}"
        );
    }
    // The vector and matrix methods are listed by example (`.mul_vec4(v)`, `.mul_mat4(m)`), so check the stems.
    for name in names_in_fn(math, "pub(crate) fn method_arity") {
        let stem = name.trim_end_matches(|c: char| c.is_ascii_digit());
        assert!(
            accepted.contains(&format!(".{stem}")),
            "`.{name}` is missing from the accepted text: {accepted}"
        );
    }
}

#[test]
fn a_type_argument_on_a_method_outside_the_dialect_reports_the_method() {
    // A name outside the dialect is E0325 (with arguments) or E0305 (without), never the built-ins' E0303.
    assert_rejected(
        &kernel_with_body("let y = x.checked_add::<u32>(1u32);"),
        Code::MethodNotInDialect,
    );
    assert_rejected(
        &kernel_with_body("let y = x.abs::<u32>();"),
        Code::UnsupportedMethod,
    );
    let text = translate(&kernel_with_body("let y = x.checked_add::<u32>(1u32);"))
        .unwrap_err()
        .to_string();
    assert!(!text.contains("built-ins"), "{text}");
    // A dialect method keeps E0303.
    assert_rejected(
        &kernel_with_body("let y = x.wrapping_neg::<u32>();"),
        Code::BuiltinTypeArguments,
    );
}

#[test]
fn star_assign_on_a_place_with_a_wrapping_method_is_not_a_call_in_a_matrix_module() {
    // `wrapping_*` is arithmetic on its operands, like `+`, so the place is evaluated once either way.
    let source = translate(&matrix_module(
        "",
        "out[i.wrapping_add(1u32)] *= 2.0; out[i.wrapping_neg()] *= 2.0;",
    ))
    .unwrap();
    assert!(
        source.contains("(out[(i + uint(1))] = __gust_mul(out[(i + uint(1))], 2.0));"),
        "{source}"
    );
    assert!(
        source.contains("(out[(0 - i)] = __gust_mul(out[(0 - i)], 2.0));"),
        "{source}"
    );
    // A call to a helper in the same place is still E0323.
    assert_rejected(
        &matrix_module(
            "fn pick(k: u32) -> u32 { k }",
            "out[pick(i).wrapping_add(1u32)] *= 2.0;",
        ),
        Code::MulAssignPlace,
    );
}

// --- Textures and samplers (slice mx.t3, D62, D64): GUST-E0731 to GUST-E0736 and the golden of the fixture ---

const TEXTURES_FIXTURE: &str = include_str!("../../../../tests/fixtures/textures.rs");
const TEXTURES_GOLDEN: &str = include_str!("../../../../tests/fixtures/textures.slang");

/// The textures fixture as `#[gpu]` receives it (the attribute is the macro's own).
fn textures_module() -> String {
    TEXTURES_FIXTURE.replacen("#[gust::gpu]", "", 1)
}

#[test]
fn kernels_of_the_textures_fixture_match_the_golden() {
    let module: syn::ItemMod =
        crate::diagnostics::testing::parse_str(textures_module().trim()).unwrap();
    validate_module(&module).unwrap();
    let mut actual = String::new();
    for item in &module.content.as_ref().unwrap().1 {
        let syn::Item::Fn(function) = item else {
            continue;
        };
        let Some(options) = kernel_options(function).unwrap() else {
            continue;
        };
        actual.push_str(&format!("// ==== kernel {} ====\n", function.sig.ident));
        actual.push_str(
            &emit_kernel(&module, function, options)
                .unwrap()
                .source
                .replace("\r\n", "\n"),
        );
    }
    assert_golden(
        &actual,
        "textures.slang",
        &TEXTURES_GOLDEN.replace("\r\n", "\n"),
    );
}

#[test]
fn the_textures_golden_declares_each_parameter_kind_with_its_binding_and_format() {
    let golden = TEXTURES_GOLDEN.replace("\r\n", "\n");
    for text in [
        "[[vk::binding(0, 0)]] Texture2D<float4> image;",
        "[[vk::binding(1, 0)]] SamplerState bilinear;",
        "[[vk::binding(2, 0)]] [format(\"rgba32f\")] WTexture2D<float4> blurred;",
        "[[vk::binding(0, 0)]] [format(\"r32f\")] RWTexture2D<float> image;",
        "[[vk::binding(1, 0)]] [format(\"r32ui\")] RWTexture2D<uint> totals;",
        "[[vk::binding(0, 0)]] Texture2D<int> source;",
        "[[vk::binding(1, 0)]] [format(\"r32i\")] RWTexture2D<int> field;",
        "[[vk::binding(1, 0)]] [format(\"rg32f\")] WTexture2D<float2> two;",
        "[[vk::binding(1, 0)]] [format(\"rg32ui\")] WTexture2D<uint2> two;",
        "[[vk::binding(1, 0)]] [format(\"rg32i\")] WTexture2D<int2> two;",
        "[[vk::binding(2, 0)]] [format(\"rgba32ui\")] WTexture2D<uint4> four;",
        "[[vk::binding(2, 0)]] [format(\"rgba32i\")] WTexture2D<int4> four;",
        "Texture2D<uint> ids;",
        "Texture2D<uint4> counts;",
        "Texture2D<int2> offsets;",
    ] {
        assert!(golden.contains(text), "{text} in {golden}");
    }
    // The size is read once, in the prologue, whatever the number of `width()` and `height()` calls.
    assert!(golden.contains(".GetDimensions("), "{golden}");
    assert!(golden.contains(".SampleLevel("), "{golden}");
    assert!(golden.contains(".Store("), "{golden}");
}

#[test]
fn the_expansion_of_the_textures_fixture_describes_textures_and_samplers() {
    let item: proc_macro2::TokenStream = textures_module().parse().unwrap();
    let text: String = crate::gpu_impl_with(
        proc_macro2::TokenStream::new(),
        item,
        crate::expand::Profile::Other,
    )
    .to_string()
    .chars()
    .filter(|c| !c.is_whitespace())
    .collect();
    // The descriptor: the texture and sampler kinds, the layout with the texture facts, no layout for a sampler.
    assert!(text.contains("::gust::ParameterKind::Texture"), "{text}");
    assert!(text.contains("::gust::ParameterKind::Sampler"), "{text}");
    assert!(text.contains("texture_v1"), "{text}");
    assert!(text.contains("::gust::TextureAccess::WriteOnly"), "{text}");
    assert!(text.contains("::gust::TextureAccess::ReadWrite"), "{text}");
    assert!(text.contains("::gust::TextureAccess::Sampled"), "{text}");
    // The typed call: a kernel with samplers is built with `new_with_samplers` and runs through the sampled `Run`.
    assert!(text.contains("::gust::Call::new_with_samplers("), "{text}");
    assert!(text.contains("::gust::SamplerDevice"), "{text}");
    // A kernel without a sampler keeps the plain `Call`.
    assert!(text.contains("::gust::Call::new("), "{text}");
}

/// A kernel whose parameters follow the dispatch id, with `body` as its statements.
fn texture_kernel(parameters: &str, body: &str) -> String {
    format!("mod bad {{ #[kernel] fn run(id: SV_DispatchThreadID, {parameters}) {{ {body} }} }}")
}

#[test]
fn texture_and_sampler_types_outside_the_dialect_are_e0731() {
    for parameters in [
        // A 3-wide texel has no texture format.
        "image: Texture2D<Vec3>",
        "image: Texture2D<bool>",
        "image: Texture2D",
        // `StorageTexture2D` is the read-write spelling, which three formats have.
        "mut image: StorageTexture2D<Rgba32Float>",
        "mut image: StorageTexture2D<Rg32Uint>",
        // An unknown format, and a sampled texel type in the place of a format.
        "mut image: WriteTexture2D<Rgba8Unorm>",
        "mut image: WriteTexture2D<f32>",
        "mut image: StorageTexture2D<Vec4>",
        // A sampler takes no type arguments.
        "filter: Sampler<f32>",
    ] {
        assert_rejected(&texture_kernel(parameters, ""), Code::TextureType);
    }
}

#[test]
fn a_texture_method_with_a_wrong_coordinate_is_e0732() {
    for (parameters, body) in [
        (
            "mut image: StorageTexture2D<R32Float>",
            "let v = image.load(0u32);",
        ),
        (
            "mut image: StorageTexture2D<R32Float>",
            "let v = image.load(0u32, 1u32, 2u32);",
        ),
        (
            "mut image: StorageTexture2D<R32Float>",
            "let v = image.load(1.5f32, 0u32);",
        ),
        (
            "mut image: StorageTexture2D<R32Float>",
            "let v = image.load(0u32, 2.0);",
        ),
        (
            "mut image: StorageTexture2D<R32Float>",
            "let v = image.load(-1, 0u32);",
        ),
        (
            "mut image: StorageTexture2D<R32Float>",
            "image.store(0u32, 0u32);",
        ),
        (
            "mut image: StorageTexture2D<R32Float>",
            "image.store(0u32, 1i32, 1.0f32);",
        ),
        (
            "image: Texture2D<Vec4>, filter: Sampler",
            "let v = image.sample(filter, 1u32, 0.5f32);",
        ),
        (
            "image: Texture2D<Vec4>, filter: Sampler",
            "let v = image.sample(filter, 0.5f32);",
        ),
        ("image: Texture2D<Vec4>", "let w = image.width(1u32);"),
    ] {
        assert_rejected(&texture_kernel(parameters, body), Code::TextureCoordinate);
    }
}

#[test]
fn the_first_argument_of_sample_must_be_a_sampler_parameter_e0733() {
    for body in [
        "let v = image.sample(id.x, 0.5f32, 0.5f32);",
        "let v = image.sample(0.5f32, 0.5f32, 0.5f32);",
        "let v = image.sample(image, 0.5f32, 0.5f32);",
        "let v = image.sample(unknown, 0.5f32, 0.5f32);",
    ] {
        assert_rejected(
            &texture_kernel("image: Texture2D<Vec4>, filter: Sampler", body),
            Code::SamplerArgument,
        );
    }
}

#[test]
fn sample_on_integer_texels_is_e0734() {
    for parameters in [
        "ids: Texture2D<u32>, filter: Sampler",
        "ids: Texture2D<UVec4>, filter: Sampler",
        "ids: Texture2D<IVec2>, filter: Sampler",
    ] {
        assert_rejected(
            &texture_kernel(parameters, "let v = ids.sample(filter, 0.5f32, 0.5f32);"),
            Code::SampleInteger,
        );
    }
}

#[test]
fn a_method_the_texture_does_not_have_is_e0735() {
    for (parameters, body) in [
        // A sampled texture is read-only.
        (
            "image: Texture2D<Vec4>",
            "image.store(0u32, 0u32, Vec4::splat(0.0f32));",
        ),
        // A write-only texture cannot be read.
        (
            "mut out: WriteTexture2D<R32Float>",
            "let v = out.load(0u32, 0u32);",
        ),
        // A storage texture is read with `load`, not sampled.
        (
            "mut image: StorageTexture2D<R32Float>, filter: Sampler",
            "let v = image.sample(filter, 0.5f32, 0.5f32);",
        ),
        // Any other name.
        ("image: Texture2D<Vec4>", "let v = image.gather(0u32);"),
        ("image: Texture2D<Vec4>", "let v = image.depth();"),
    ] {
        assert_rejected(&texture_kernel(parameters, body), Code::TextureMethod);
    }
}

#[test]
fn a_texture_or_sampler_used_as_a_value_is_e0736() {
    for (parameters, body) in [
        ("image: Texture2D<Vec4>", "let alias = image;"),
        (
            "image: Texture2D<Vec4>",
            "let size = image.width() + image;",
        ),
        ("filter: Sampler", "let alias = filter;"),
        // A sampler has no methods of its own.
        ("filter: Sampler", "let v = filter.width();"),
        // A texture type anywhere but a kernel parameter.
        (
            "image: Texture2D<Vec4>",
            "let alias: Texture2D<Vec4> = 1u32;",
        ),
        ("filter: Sampler", "let alias: Sampler = 1u32;"),
    ] {
        assert_rejected(&texture_kernel(parameters, body), Code::TextureUse);
    }
}

#[test]
fn a_texture_in_a_helper_function_is_refused_with_the_resource_code() {
    assert_rejected(
        "mod bad { fn helper(image: Texture2D<Vec4>) -> u32 { 0u32 } #[kernel] fn run(id: SV_DispatchThreadID) {} }",
        Code::HelperResourceParameter,
    );
    assert_rejected(
        "mod bad { fn helper(filter: Sampler) -> u32 { 0u32 } #[kernel] fn run(id: SV_DispatchThreadID) {} }",
        Code::HelperResourceParameter,
    );
}

#[test]
fn an_accepted_texture_kernel_emits_the_lowering_of_the_module_comment() {
    let source = translate(&texture_kernel(
        "image: Texture2D<f32>, filter: Sampler, mut out: StorageTexture2D<R32Float>",
        "let w = out.width(); let h = out.height(); \
         let v = image.sample(filter, 0.25f32, 0.75f32); \
         let n = out.load(id.x, id.y); \
         out.store(id.x, id.y, v + n); \
         let t = image.load(w, h);",
    ))
    .unwrap();
    assert!(source.contains("Texture2D<float> image;"), "{source}");
    assert!(source.contains("SamplerState filter;"), "{source}");
    assert!(
        source.contains("[format(\"r32f\")] RWTexture2D<float> out;"),
        "{source}"
    );
    assert!(
        source.contains("image.SampleLevel(filter, float2("),
        "{source}"
    );
    assert!(source.contains("out.Store(uint2("), "{source}");
    assert!(source.contains("out.GetDimensions("), "{source}");
}

#[test]
fn a_parenthesised_receiver_of_width_and_height_lowers_like_a_bare_one() {
    // Review mx.t3 M1: `(out).width()` used to pass validation and print `__gpu_width_(out)`, with no prologue local
    // declared. The size local is named after the stripped identifier, so the two spellings are the same call.
    let bare = translate(&texture_kernel(
        "mut out: StorageTexture2D<R32Float>",
        "if id.x < out.width() && id.y < out.height() { out.store(id.x, id.y, 1.0f32); }",
    ))
    .unwrap();
    let wrapped = translate(&texture_kernel(
        "mut out: StorageTexture2D<R32Float>",
        "if id.x < (out).width() && id.y < ((out)).height() { out.store(id.x, id.y, 1.0f32); }",
    ))
    .unwrap();
    assert_eq!(bare, wrapped);
    assert!(!wrapped.contains("__gpu_width_("), "{wrapped}");
    assert!(
        wrapped.contains("out.GetDimensions(__gpu_width_out, __gpu_height_out);"),
        "{wrapped}"
    );
    assert!(wrapped.contains("id.x < __gpu_width_out"), "{wrapped}");
}

#[test]
fn the_messages_for_a_missing_method_and_a_misused_name_fit_the_type() {
    let message = |parameters: &str, body: &str| {
        translate(&texture_kernel(parameters, body))
            .unwrap_err()
            .to_string()
    };
    // `sample` on a write-only texture does not send it to `load`, which it does not have either.
    let write_only = message(
        "mut out: WriteTexture2D<R32Float>, filter: Sampler",
        "let v = out.sample(filter, 0.5f32, 0.5f32);",
    );
    assert!(
        write_only.contains("write-only texture has"),
        "{write_only}"
    );
    assert!(!write_only.contains("read with `.load"), "{write_only}");
    let storage = message(
        "mut image: StorageTexture2D<R32Float>, filter: Sampler",
        "let v = image.sample(filter, 0.5f32, 0.5f32);",
    );
    assert!(storage.contains("read with `.load(x, y)`"), "{storage}");
    // A texture and a sampler each get the list of its own uses.
    let texture = message("image: Texture2D<Vec4>", "let alias = image;");
    assert!(
        texture.contains("`image` is a texture parameter"),
        "{texture}"
    );
    assert!(texture.contains("image.sample(sampler, u, v)"), "{texture}");
    assert!(!texture.contains("sampler parameter"), "{texture}");
    let sampler = message("filter: Sampler", "let alias = filter;");
    assert!(
        sampler.contains("`filter` is a sampler parameter"),
        "{sampler}"
    );
    assert!(
        sampler.contains("texture.sample(filter, u, v)"),
        "{sampler}"
    );
    assert!(!sampler.contains(".load("), "{sampler}");
}

// ---- Methods in `impl` blocks (slice m9.a1, M9 plan 2.1) ----

#[test]
fn methods_golden() {
    let source = translate(include_str!("../../../../tests/fixtures/methods.rs")).unwrap();
    assert_golden(
        &source.replace("\r\n", "\n"),
        "methods.slang",
        &include_str!("../../../../tests/fixtures/methods.slang").replace("\r\n", "\n"),
    );
}

#[test]
fn methods_mut_golden() {
    let source = translate(include_str!("../../../../tests/fixtures/methods_mut.rs")).unwrap();
    assert_golden(
        &source.replace("\r\n", "\n"),
        "methods_mut.slang",
        &include_str!("../../../../tests/fixtures/methods_mut.slang").replace("\r\n", "\n"),
    );
}

const COUNTER_MODULE: &str = "mod m {
    pub struct Counter { pub n: uint }
    impl Counter {
        pub fn bump(&mut self) { self.n = self.n + 1u32; }
        pub fn read(&self) -> uint { self.n }
    }
    #[kernel]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<Counter>) {
        if id.x > 0u32 { out[0u32].bump(); }
    }
}";

#[test]
fn a_self_method_is_a_struct_member_and_a_call_is_a_member_call() {
    let source = translate(
        "mod m {
            pub struct Pair { pub a: f32, pub b: f32 }
            impl Pair {
                pub fn mix(&self, k: f32) -> f32 { self.a * k + self.b }
            }
            #[kernel]
            pub fn run(id: SV_DispatchThreadID, pairs: StructuredBuffer<Pair>, mut out: RWStructuredBuffer<f32>) {
                let local = pairs[id.x];
                out[id.x] = local.mix(2.0f32) + pairs[id.x].mix(3.0f32);
            }
        }",
    )
    .unwrap();
    let source = source.replace("\r\n", "\n");
    assert!(
        source.contains("struct Pair\n{\n    float a;\n    float b;\n\n    float mix(float k)\n    {\n        return ((this.a * k) + this.b);\n    }\n};"),
        "{source}"
    );
    assert!(source.contains("local.mix(float(2.0))"), "{source}");
    assert!(source.contains("pairs[id.x].mix(float(3.0))"), "{source}");
}

#[test]
fn x1_a_mutating_method_on_a_storage_element_is_one_e0333() {
    // The declaration of `bump` is accepted since m9.a2; the call on the storage element is the one error.
    let error = translate(COUNTER_MODULE).expect_err("X1");
    crate::diagnostics::testing::assert_codes(&error, &[Code::MutatingMethodOnSharedPlace]);
}

#[test]
fn receivers_are_ref_self_ref_mut_self_and_self_and_not_mut_self_or_a_typed_one() {
    let source = "mod m {
        pub struct Counter { pub n: uint }
        impl Counter { pub fn bump(&mut self) { self.n = self.n + 1u32; } }
        #[kernel]
        pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) { out[id.x] = 1u32; }
    }";
    let slang = translate(source).unwrap().replace("\r\n", "\n");
    assert!(
        slang.contains("    [mutating] void bump()\n    {\n        this.n = (this.n + uint(1));"),
        "{slang}"
    );
    // By value: a plain member, which reads its copy.
    let by_value = translate(&source.replace(
        "&mut self) { self.n = self.n + 1u32; }",
        "self) -> uint { self.n }",
    ))
    .unwrap()
    .replace("\r\n", "\n");
    assert!(by_value.contains("    uint bump()\n"), "{by_value}");
    assert!(!by_value.contains("[mutating]"), "{by_value}");
    // `mut self` and a typed receiver stay E0116.
    assert_rejected(
        &source.replace("&mut self", "mut self"),
        Code::SelfParameter,
    );
    assert_rejected(
        &source.replace("&mut self", "self: Box<Self>"),
        Code::SelfParameter,
    );
}

#[test]
fn a_mutating_method_on_a_workgroup_or_uniform_root_is_e0333() {
    for (parameters, receiver) in [
        ("mut scratch: Workgroup<[uint; 4]>", "scratch[0u32]"),
        ("u: Uniform<Counter>", "u.n"),
        ("input: StructuredBuffer<Counter>", "input[0u32]"),
    ] {
        let source = format!(
            "mod m {{
                pub struct Counter {{ pub n: uint }}
                impl Counter {{ pub fn bump(&mut self) {{ self.n = self.n + 1u32; }} }}
                #[kernel(workgroup_size(64, 1, 1), cpu = \"unsupported\")]
                pub fn run(id: SV_DispatchThreadID, {parameters}) {{ {receiver}.bump(); }}
            }}"
        );
        assert_rejected(&source, Code::MutatingMethodOnSharedPlace);
    }
}

#[test]
fn a_shared_place_may_be_the_receiver_of_a_self_method() {
    // `&self` copies the element (spike P1b), on a buffer element and on a field of one.
    translate(
        "mod m {
            pub struct Inner { pub v: f32 }
            pub struct Outer { pub inner: Inner }
            impl Inner { pub fn get_v(&self) -> f32 { self.v } }
            #[kernel]
            pub fn run(id: SV_DispatchThreadID, items: StructuredBuffer<Outer>, mut out: RWStructuredBuffer<f32>) {
                out[id.x] = items[id.x].inner.get_v();
            }
        }",
    )
    .unwrap();
}

#[test]
fn an_impl_for_a_type_the_module_does_not_define_is_e0121() {
    for item in [
        "impl Elsewhere { pub fn f(&self) -> uint { 1u32 } }",
        "impl<T> Anything for T { }",
        "impl Mat4 { pub fn f(&self) -> uint { 1u32 } }",
        "impl (uint, uint) { }",
    ] {
        let source = format!(
            "mod m {{ pub struct A {{ pub x: uint }} {item} #[kernel] pub fn run(id: SV_DispatchThreadID) {{}} }}"
        );
        assert_rejected(&source, Code::ImplForeignType);
    }
}

#[test]
fn impl_shapes_that_are_not_accepted_yet_keep_their_own_codes() {
    for (item, code) in [
        (
            "impl A { const K: uint = 1u32; }",
            Code::UnsupportedModuleItem,
        ),
        (
            "impl Clone for A { fn clone(&self) -> A { *self } }",
            Code::UnsupportedModuleItem,
        ),
        ("impl<T> A { }", Code::ExternOrGenericFunction),
        (
            "impl A { pub fn f<T>(&self) -> uint { 1u32 } }",
            Code::ExternOrGenericFunction,
        ),
        (
            "impl A { pub fn f(self: Box<Self>) -> uint { 1u32 } }",
            Code::SelfParameter,
        ),
        (
            "impl A { #[kernel] pub fn f(&self) {} }",
            Code::UnsupportedAttribute,
        ),
        (
            "impl A { pub fn f(&self, b: StructuredBuffer<uint>) -> uint { 1u32 } }",
            Code::HelperResourceParameter,
        ),
        (
            "impl A { pub fn f(&self) { workgroup_barrier(); } }",
            Code::BarrierPosition,
        ),
        (
            "impl A { pub fn f(&self) -> uint { nowhere() } }",
            Code::UnknownCall,
        ),
        (
            "impl A { pub fn f(&self) -> uint { self.nowhere() } }",
            Code::UnsupportedMethod,
        ),
    ] {
        let source = format!(
            "mod m {{ pub struct A {{ pub x: uint }} fn g() -> uint {{ 1u32 }} {item} #[kernel] pub fn run(id: SV_DispatchThreadID) {{}} }}"
        );
        assert_rejected(&source, code);
    }
}

#[test]
fn a_method_cannot_take_the_name_of_a_dialect_method_a_slang_member_keyword_or_a_field() {
    let mut names: Vec<String> = [
        "global_id",
        "len",
        "is_some",
        "is_none",
        "is_ok",
        "is_err",
        "unwrap_or",
        "unwrap",
        "expect",
        "wrapping_add",
        "wrapping_sub",
        "wrapping_mul",
        "wrapping_neg",
        "dot",
        "mul_vec2",
        "mul_vec3",
        "mul_vec4",
        "mul_mat2",
        "mul_mat3",
        "mul_mat4",
        "transpose",
        "load",
        "store",
        "sample",
        "width",
        "height",
    ]
    .iter()
    .map(|name| (*name).to_owned())
    .collect();
    names.extend(
        crate::validate::SLANG_MEMBER_KEYWORDS
            .iter()
            .map(|name| (*name).to_owned()),
    );
    // Every swizzle: two to four of x, y, z, w.
    let letters = ['x', 'y', 'z', 'w'];
    for length in 2..=4_u32 {
        for index in 0..4_usize.pow(length) {
            let name: String = (0..length)
                .map(|place| letters[(index / 4_usize.pow(place)) % 4])
                .collect();
            names.push(name);
        }
    }
    assert_eq!(names.len(), 26 + 5 + 16 + 64 + 256);
    for name in &names {
        let source = format!(
            "mod m {{ pub struct A {{ pub v: uint }} impl A {{ pub fn {name}(&self) -> uint {{ 1u32 }} }} #[kernel] pub fn run(id: SV_DispatchThreadID) {{}} }}"
        );
        assert_rejected(&source, Code::MethodNameClash);
    }
    // A method named like a field of its own struct, the getter pattern.
    assert_rejected(
        "mod m { pub struct A { pub v: uint } impl A { pub fn v(&self) -> uint { self.v } } #[kernel] pub fn run(id: SV_DispatchThreadID) {} }",
        Code::MethodNameClash,
    );
    // A field of another struct is no clash, and an ordinary name is none.
    translate(
        "mod m { pub struct A { pub v: uint } pub struct B { pub w: uint } impl B { pub fn v(&self) -> uint { self.w } } #[kernel] pub fn run(id: SV_DispatchThreadID) {} }",
    )
    .unwrap();
}

#[test]
fn the_module_item_message_names_impl_blocks() {
    let error = translate(
        "mod m { const K: uint = 1u32; #[kernel] pub fn run(id: SV_DispatchThreadID) {} }",
    )
    .expect_err("const");
    assert_code(&error, Code::UnsupportedModuleItem);
    assert!(error.to_string().contains("`impl` blocks"), "{error}");
}

#[test]
fn a_helper_may_call_a_method() {
    translate(
        "mod m {
            pub struct A { pub x: uint }
            impl A { pub fn twice(&self) -> uint { self.x * 2u32 } }
            fn via(a: A) -> uint { a.twice() }
            #[kernel]
            pub fn run(id: SV_DispatchThreadID, items: StructuredBuffer<A>, mut out: RWStructuredBuffer<uint>) {
                out[id.x] = via(items[id.x]);
            }
        }",
    )
    .unwrap();
}

#[test]
fn a_parameter_or_local_named_this_does_not_shadow_the_receiver() {
    // `this` is how a receiver is written in Slang; a user `this` is renamed like every name of the collision table
    // (review r1, M1): `(this.a - this.a)` would compile and give 0.0 on the GPU.
    let source = translate(
        "mod m {
            pub struct S { pub a: f32 }
            impl S {
                pub fn diff(&self, this: S) -> f32 { self.a - this.a }
                pub fn other(&self, o: S) -> f32 { let this = o; self.a - this.a }
            }
            #[kernel]
            pub fn run(id: SV_DispatchThreadID, items: StructuredBuffer<S>, mut out: RWStructuredBuffer<f32>) {
                out[id.x] = items[id.x].diff(items[0u32]) + items[id.x].other(items[0u32]);
            }
        }",
    )
    .unwrap()
    .replace("\r\n", "\n");
    assert!(source.contains("float diff(S __gust_r_this)"), "{source}");
    assert!(
        source.contains("return (this.a - __gust_r_this.a);"),
        "{source}"
    );
    assert!(source.contains("var __gust_r_this = o;"), "{source}");
}

#[test]
fn self_is_the_struct_of_the_impl_in_types_literals_and_associated_calls_and_nowhere_else() {
    let source = translate(
        "mod m {
            pub struct Circle { pub r: f32 }
            impl Circle {
                pub fn unit() -> Self { let c = Self { r: 1.0f32 }; c }
                pub fn again(&self) -> Self { let c: Self = Self { r: self.r }; c }
                pub fn twin(&self) -> Self { let d = Self::unit(); d }
            }
            #[kernel]
            pub fn run(id: SV_DispatchThreadID, items: StructuredBuffer<Circle>, mut out: RWStructuredBuffer<f32>) {
                out[id.x] = items[id.x].twin().r + Circle::unit().r;
            }
        }",
    )
    .unwrap()
    .replace("\r\n", "\n");
    assert!(source.contains("static Circle unit()"), "{source}");
    assert!(source.contains("Circle again()"), "{source}");
    assert!(source.contains("var d = Circle.unit();"), "{source}");
    assert!(!source.contains("Self"), "{source}");
    // Outside an `impl`, and as any other path, it has no lowering.
    for item in [
        "fn free() -> Self { Circle { r: 1.0f32 } }",
        "impl Circle { pub fn k(&self) -> f32 { Self::K } }",
        "impl Circle { pub fn k(&self) -> f32 { let f = Self::unit; 1.0f32 } pub fn unit() -> Circle { Circle { r: 1.0f32 } } }",
    ] {
        let source = format!(
            "mod m {{ pub struct Circle {{ pub r: f32 }} {item} #[kernel] pub fn run(id: SV_DispatchThreadID) {{}} }}"
        );
        assert!(translate(&source).is_err(), "{source}");
    }
}

#[test]
fn a_dereference_is_an_unsupported_expression() {
    for body in [
        "pub fn copy(&self) -> Circle { *self }",
        "pub fn value(&self) -> f32 { (*self).r }",
    ] {
        let source = format!(
            "mod m {{ pub struct Circle {{ pub r: f32 }} impl Circle {{ {body} }} #[kernel] pub fn run(id: SV_DispatchThreadID) {{}} }}"
        );
        assert_rejected(&source, Code::UnsupportedExpression);
    }
    assert_rejected(
        "mod m { #[kernel] pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) { let x = 1u32; out[id.x] = *x; } }",
        Code::UnsupportedExpression,
    );
}

#[test]
fn a_texture_receiver_is_e0735_and_an_invocation_id_receiver_is_e0333() {
    let source = |parameters: &str, body: &str| {
        format!(
            "mod m {{ pub struct Counter {{ pub n: uint }}
             impl Counter {{ pub fn bump(&mut self) {{ self.n = self.n + 1u32; }} }}
             #[kernel(workgroup_size(64, 1, 1))] pub fn run(id: SV_DispatchThreadID, {parameters}) {{ {body} }} }}"
        )
    };
    assert_rejected(
        &source("image: Texture2D<Vec4>", "image.bump();"),
        Code::TextureMethod,
    );
    assert_rejected(
        &source("lid: LocalId", "lid.bump();"),
        Code::MutatingMethodOnSharedPlace,
    );
    let error = translate(&source("lid: LocalId", "lid.bump();")).expect_err("lid");
    assert!(!error.to_string().contains("texture"), "{error}");
}

#[test]
fn raw_identifiers_name_the_same_struct_and_method() {
    let source = translate(
        "mod m {
            pub struct Circle { pub r: f32 }
            impl r#Circle { pub fn r#area(&self) -> f32 { self.r } }
            #[kernel]
            pub fn run(id: SV_DispatchThreadID, items: StructuredBuffer<Circle>, mut out: RWStructuredBuffer<f32>) {
                out[id.x] = items[id.x].r#area();
            }
        }",
    )
    .unwrap()
    .replace("\r\n", "\n");
    assert!(source.contains("float area()"), "{source}");
    assert!(source.contains("items[id.x].area()"), "{source}");
}

/// A kernel that stores a value computed by the call in `$body` (slice m9.a2 tests).
fn with_kernel(items: &str, body: &str) -> String {
    format!(
        "mod m {{ {items} #[kernel] pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {{ {body} }} }}"
    )
}

#[test]
fn recursion_is_gust_e0120_direct_mutual_through_a_method_and_by_name() {
    // Direct, mutual, through a method, a helper that reaches itself by a method, and by name (D162).
    let cases = [
        "fn f(n: u32) -> u32 { f(n) }",
        "fn a(n: u32) -> u32 { b(n) } fn b(n: u32) -> u32 { a(n) }",
        "pub struct S { pub n: u32 } impl S { fn go(&self) -> u32 { self.go() } }",
        "pub struct S { pub n: u32 } impl S { fn go(&self) -> u32 { Self::again(self.n) } fn again(n: u32) -> u32 { S::again(n) } }",
        "pub struct S { pub n: u32 } impl S { fn go(&self) -> u32 { h(self.n) } } fn h(n: u32) -> u32 { let s = S { n: n }; s.go() }",
        // By name: `x.go()` is an edge to every method named `go`, `S::go` included, so this is a cycle that Rust does not have (D162).
        "pub struct S { pub n: u32 } pub struct T { pub n: u32 } \
         impl S { fn go(&self) -> u32 { let t = T { n: self.n }; t.go() } } \
         impl T { fn go(&self) -> u32 { self.n } }",
    ];
    for items in cases {
        let source = with_kernel(items, "out[id.x] = id.x;");
        let outcome = translate(&source);
        let error = outcome.expect_err(&source);
        assert_code_for(&error, Code::RecursiveCall, &source);
    }
    let source = with_kernel(
        "pub struct S { pub n: u32 } pub struct T { pub n: u32 } \
         impl S { fn go(&self) -> u32 { let t = T { n: self.n }; t.go() } } \
         impl T { fn go(&self) -> u32 { let s = S { n: self.n }; s.go() } }",
        "out[id.x] = id.x;",
    );
    assert_rejected(&source, Code::RecursiveCall);
    // A graph without a cycle is accepted: a diamond.
    let source = with_kernel(
        "fn a(n: u32) -> u32 { b(n) + c(n) } fn b(n: u32) -> u32 { d(n) } fn c(n: u32) -> u32 { d(n) } fn d(n: u32) -> u32 { n }",
        "out[id.x] = a(id.x);",
    );
    assert!(translate(&source).is_ok(), "{source}");
}

#[test]
fn a_mutating_call_in_a_for_bound_is_gust_e0421_and_a_read_is_not() {
    let items = "pub struct C { pub n: u32 } impl C { fn next(&mut self) -> u32 { self.n = self.n + 1u32; self.n } fn peek(&self) -> u32 { self.n } }";
    for body in [
        "let mut c = C { n: 0u32 }; for i in 0u32..c.next() { out[i] = i; }",
        "let mut c = C { n: 0u32 }; for i in c.next()..4u32 { out[i] = i; }",
    ] {
        assert_rejected(&with_kernel(items, body), Code::ForBoundSideEffect);
    }
    let body = "let c = C { n: 3u32 }; for i in 0u32..c.peek() { out[i] = i; }";
    assert!(translate(&with_kernel(items, body)).is_ok());
}

#[test]
fn a_mutating_call_needs_a_local_place_and_a_value_parameter_or_an_rvalue_is_gust_e0333() {
    let items = "pub struct C { pub n: u32 } impl C { fn bump(&mut self) { self.n = self.n + 1u32; } } \
                 fn make() -> C { let c = C { n: 0u32 }; c }";
    assert_rejected(
        &with_kernel(items, "make().bump();"),
        Code::MutatingMethodOnSharedPlace,
    );
    assert_rejected(
        &with_kernel(
            "pub struct C { pub n: u32 } impl C { fn bump(&mut self) { self.n = self.n + 1u32; } } fn by_value(c: C) -> u32 { c.bump(); c.n }",
            "out[id.x] = 0u32;",
        ),
        Code::MutatingMethodOnSharedPlace,
    );
    let ok = "let mut c = C { n: 0u32 }; c.bump(); out[id.x] = c.n;";
    translate(&with_kernel(items, ok)).unwrap();
}

#[test]
fn a_helper_named_like_a_method_or_like_a_field_of_the_owner_is_gust_e0328() {
    let source = with_kernel(
        "pub struct S { pub n: u32 } impl S { fn area(&self) -> u32 { self.n } } fn area(n: u32) -> u32 { n }",
        "out[id.x] = id.x;",
    );
    assert_rejected(&source, Code::MethodNameClash);
    let source = with_kernel(
        "pub struct S { pub total: u32 } impl S { fn go(&self) -> u32 { total(self.total) } } fn total(n: u32) -> u32 { n }",
        "out[id.x] = id.x;",
    );
    assert_rejected(&source, Code::MethodNameClash);
}

#[test]
fn several_impl_blocks_and_the_self_path_form_are_accepted_and_an_impl_attribute_is_not() {
    let source = with_kernel(
        "pub struct S { pub n: u32 } impl S { fn one(&self) -> u32 { self.n } } impl self::S { fn two(&self) -> u32 { self.one() + 1u32 } }",
        "let s = S { n: id.x }; out[id.x] = s.two();",
    );
    let text = translate(&source).unwrap();
    assert!(
        text.contains("uint one()") && text.contains("uint two()"),
        "{text}"
    );
    let source = with_kernel(
        "pub struct S { pub n: u32 } #[inline] impl S { fn one(&self) -> u32 { self.n } }",
        "out[id.x] = id.x;",
    );
    assert_rejected(&source, Code::UnsupportedAttribute);
}

#[test]
fn associated_calls_resolve_by_path_and_a_receiver_function_by_path_is_not_accepted() {
    let items = "pub struct S { pub n: u32 } impl S { fn make(n: u32) -> S { let s = S { n: n }; s } fn get(&self) -> u32 { self.n } }";
    let ok = with_kernel(items, "let s = S::make(id.x); out[id.x] = s.get();");
    let text = translate(&ok).unwrap().replace("\r\n", "\n");
    assert!(text.contains("static S "), "{text}");
    assert_rejected(
        &with_kernel(items, "let s = S::make(1u32); out[id.x] = S::get(s);"),
        Code::UnknownCall,
    );
    assert_rejected(
        &with_kernel(items, "out[id.x] = S::nowhere(1u32);"),
        Code::UnknownCall,
    );
}

#[test]
fn a_method_cannot_open_a_barrier_or_call_a_kernel() {
    let source = with_kernel(
        "pub struct S { pub n: u32 } impl S { fn sync(&self) -> u32 { workgroup_barrier(); self.n } }",
        "out[id.x] = id.x;",
    );
    assert_rejected(&source, Code::BarrierPosition);
    let source = "mod m { pub struct S { pub n: u32 } impl S { fn go(&self) -> u32 { run2(self.n) } } \
        #[kernel] pub fn run2(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) { out[id.x] = id.x; } \
        #[kernel] pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) { out[id.x] = id.x; } }";
    assert!(translate(source).is_err());
}

const ORDER_ITEMS: &str = "pub struct C { pub n: u32 } \
    impl C { fn next(&mut self) -> u32 { self.n = self.n + 1u32; self.n } fn bump(&mut self) -> u32 { self.n = self.n + 1u32; 2u32 } }";

#[test]
fn a_mutating_call_in_an_assignee_or_a_matrix_mul_assign_value_is_gust_e0421() {
    // Review m9.a2 M1: Rust evaluates the value before the place, Slang evaluates a call in the place first.
    for body in [
        "let mut c = C { n: 5u32 }; out[c.next()] = c.next();",
        "let mut d = C { n: 5u32 }; out[d.bump()] = d.n;",
        "let mut c = C { n: 5u32 }; out[c.next()] += c.next();",
        "let mut d = C { n: 5u32 }; out[d.next()] /= d.next();",
        "let mut c = C { n: 1u32 }; out[c.n] /= c.next();",
        "let mut c = C { n: 1u32 }; out[c.n] %= c.next();",
        "let mut c = C { n: 1u32 }; out[c.n] ^= c.next();",
        // The value of `*=` in a module that names a matrix (the `__gust_mul` path).
        "let m = Mat4::IDENTITY; let mut c = C { n: 5u32 }; c.n *= c.bump();",
    ] {
        assert_rejected(&with_kernel(ORDER_ITEMS, body), Code::ForBoundSideEffect);
    }
    // What already matches Rust stays accepted.
    for body in [
        "let mut c = C { n: 5u32 }; out[c.n] = c.next();",
        "let mut c = C { n: 5u32 }; c.n += c.bump();",
        "let mut e = C { n: 5u32 }; e.n /= e.bump();",
        // Without a matrix in the module `*=` is the plain operator.
        "let mut c = C { n: 5u32 }; c.n *= c.bump();",
    ] {
        let source = with_kernel(ORDER_ITEMS, body);
        translate(&source).unwrap_or_else(|error| panic!("{body}: {error}"));
    }
}

#[test]
fn a_mutating_call_in_the_start_bound_that_the_end_call_changes_is_gust_e0421() {
    // Review m9.a2 S1: Rust reads `c.n` first, the emitted loop runs the end bound first, so the start differs.
    let source = with_kernel(
        ORDER_ITEMS,
        "let mut c = C { n: 1u32 }; for i in c.n..c.next() { out[id.x] = i; }",
    );
    assert_rejected(&source, Code::ForBoundSideEffect);
}

#[test]
fn self_as_a_value_in_a_mut_self_method_is_gust_e0116_and_in_the_other_receivers_it_is_not() {
    // Review m9.a2 M2: `let c = self;` binds the reference in Rust and copies in Slang.
    for method in [
        "fn set(&mut self) { let c = self; c.n = 9u32; }",
        "fn set(&mut self) { let c = self; c.bump(); }",
    ] {
        let items = format!(
            "pub struct C {{ pub n: u32 }} impl C {{ fn bump(&mut self) {{ self.n = self.n + 1u32; }} {method} }}"
        );
        assert_rejected(
            &with_kernel(&items, "out[id.x] = id.x;"),
            Code::SelfParameter,
        );
    }
    // A by-value `self` and a `&self` copy in both languages: accepted.
    let items = "pub struct C { pub n: u32 } impl C { \
        fn with(self, n: u32) -> C { let mut c = self; c.n = n; c } \
        fn peek(&self) -> u32 { let c = self; c.n } \
        fn keep(&mut self) -> u32 { (self).n = (self).n + 1u32; (self).n } }";
    let source = with_kernel(
        items,
        "let c = C { n: 1u32 }; let d = c.with(7u32); out[id.x] = d.n + c.peek();",
    );
    translate(&source).unwrap();
}

#[test]
fn an_impl_attribute_other_than_doc_and_lints_is_gust_e0115() {
    for attribute in ["#[cfg(test)]", "#[derive(Clone)]", "#[kernel]", "#[inline]"] {
        let source = with_kernel(
            &format!(
                "pub struct S {{ pub n: u32 }} {attribute} impl S {{ fn one(&self) -> u32 {{ self.n }} }}"
            ),
            "out[id.x] = id.x;",
        );
        assert_rejected(&source, Code::UnsupportedAttribute);
    }
    let source = with_kernel(
        "pub struct S { pub n: u32 } #[allow(dead_code)] impl S { fn one(&self) -> u32 { self.n } }",
        "out[id.x] = id.x;",
    );
    translate(&source).unwrap();
}

#[test]
fn a_raw_helper_name_clashes_with_a_method_of_the_same_name() {
    let source = with_kernel(
        "pub struct S { pub n: u32 } impl S { fn area(&self) -> u32 { self.n } } fn r#area(n: u32) -> u32 { n }",
        "out[id.x] = id.x;",
    );
    assert_rejected(&source, Code::MethodNameClash);
}

// ---- Enums and the statement `match` on them (slice m9.e1, M9 plan 2.2, D157) ----

#[test]
fn enums_golden() {
    let source = translate(include_str!("../../../../tests/fixtures/enums.rs")).unwrap();
    assert_golden(
        &source.replace("\r\n", "\n"),
        "enums.slang",
        &include_str!("../../../../tests/fixtures/enums.slang").replace("\r\n", "\n"),
    );
}

const MODE_ENUM: &str = "#[derive(PartialEq, Eq)] pub enum Mode { A, B = 3, C }";

fn with_mode(items: &str, body: &str) -> String {
    with_kernel(&format!("{MODE_ENUM} {items}"), body)
}

#[test]
fn an_enum_is_a_slang_enum_with_its_discriminants_and_a_match_is_a_switch() {
    let source = translate(&with_mode(
        "fn f(m: Mode) -> u32 { let mut o = 1u32; match m { Mode::A => { o += 1u32; } Mode::B | Mode::C => { o += 2u32; } } o }",
        "out[id.x] = f(Mode::C) + (Mode::B as u32);",
    ))
    .unwrap()
    .replace("\r\n", "\n");
    assert!(
        source.contains("enum Mode : uint { A, B = 3, C };"),
        "{source}"
    );
    // Grouped alternatives are adjacent labels of one body; every arm ends in `break`; the scrutinee is written once.
    assert!(
        source.contains(
            "switch (m)\n    {\n        case Mode::A:\n        {\n            (o += uint(1));\n            break;\n        }\n        case Mode::B:\n        case Mode::C:\n        {\n            (o += uint(2));\n            break;\n        }\n    }"
        ),
        "{source}"
    );
    assert!(source.contains("f(Mode::C)"), "{source}");
    assert!(source.contains("uint(Mode::B)"), "{source}");
}

#[test]
fn a_signed_enum_is_an_int_enum_with_negative_discriminants() {
    let source = translate(&with_kernel(
        "#[derive(PartialEq, Eq)] #[repr(i32)] pub enum Sign { Neg = -2, Mid, Pos = 5 }",
        "let s = Sign::Mid; out[id.x] = (s as i32) as u32;",
    ))
    .unwrap();
    assert!(
        source.contains("enum Sign : int { Neg = -2, Mid, Pos = 5 };"),
        "{source}"
    );
    assert!(source.contains("int(s)"), "{source}");
}

#[test]
fn a_match_in_the_tail_of_a_function_with_a_result_ends_in_default_unless_it_has_one() {
    // Measured: WGSL refuses a `switch` whose arms all return when it has no `default` (slangc 41009), and rustc has
    // proven the arms exhaustive, so the last arm is the `default`.
    let tail = "fn g(m: Mode) -> u32 { match m { Mode::A => { return 1u32; } Mode::B => { return 2u32; } Mode::C => { return 3u32; } } }";
    let source = translate(&with_mode(tail, "out[id.x] = g(Mode::A);"))
        .unwrap()
        .replace("\r\n", "\n");
    assert!(
        source.contains("case Mode::B:\n        {\n            return uint(2);\n        }\n        default:\n        {\n            return uint(3);\n        }"),
        "{source}"
    );
    assert!(!source.contains("case Mode::C"), "{source}");
    // With a `_` arm the labels stay; a `break` is not written after a `return`.
    let wild =
        "fn g(m: Mode) -> u32 { match m { Mode::A => { return 1u32; } _ => { return 2u32; } } }";
    let source = translate(&with_mode(wild, "out[id.x] = g(Mode::A);")).unwrap();
    assert!(source.contains("case Mode::A:"), "{source}");
    assert!(!source.contains("break;"), "{source}");
    // A `match` that is not the tail keeps every label.
    let early = "fn g(m: Mode) -> u32 { match m { Mode::A => { return 1u32; } Mode::B | Mode::C => { } } 7u32 }";
    let source = translate(&with_mode(early, "out[id.x] = g(Mode::A);")).unwrap();
    assert!(!source.contains("default"), "{source}");
    assert!(source.contains("case Mode::C:"), "{source}");
}

#[test]
fn enum_declaration_shapes_outside_the_contract_are_gust_e0122() {
    let cases = [
        // A data-carrying variant, a tuple one and a struct one.
        "#[derive(PartialEq, Eq)] pub enum E { A(u32), B }",
        "#[derive(PartialEq, Eq)] pub enum E { A { x: u32 }, B }",
        // A discriminant that is not an integer literal, and a negative one without a signed repr.
        "#[derive(PartialEq, Eq)] pub enum E { A = 1 + 2, B }",
        "#[derive(PartialEq, Eq)] pub enum E { A = -1, B }",
        "#[derive(PartialEq, Eq)] pub enum E { A = 1.5, B }",
        // `PartialEq` missing, or only another trait derived.
        "pub enum E { A, B }",
        "#[derive(Debug)] pub enum E { A, B }",
        // Generics, no variants, a repr the Slang enum has no type for, manual Clone and Copy, a variant attribute.
        "#[derive(PartialEq, Eq)] pub enum E<T> { A, B }",
        "#[derive(PartialEq, Eq)] pub enum E {}",
        "#[derive(PartialEq, Eq)] #[repr(u8)] pub enum E { A, B }",
        "#[derive(PartialEq, Eq)] #[repr(u32, C)] pub enum E { A, B }",
        "#[derive(PartialEq, Eq, Clone, Copy)] pub enum E { A, B }",
        "#[derive(PartialEq, Eq)] pub enum E { #[default] A, B }",
    ];
    for items in cases {
        assert_rejected(&with_kernel(items, "out[id.x] = id.x;"), Code::EnumShape);
    }
    // Casting to the other integer type, to a float, or a variant to the other integer type.
    for items in [
        "fn f(m: Mode) -> i32 { m as i32 }",
        "fn f(m: Mode) -> f32 { m as f32 }",
        "fn f() -> i32 { Mode::A as i32 }",
        "fn h() -> Mode { Mode::A } fn f() -> i32 { h() as i32 }",
    ] {
        assert_rejected(&with_mode(items, "out[id.x] = id.x;"), Code::EnumShape);
    }
    // The signed enum casts to `i32` only.
    assert_rejected(
        &with_kernel(
            "#[derive(PartialEq, Eq)] #[repr(i32)] pub enum S { A = -1, B } fn f(s: S) -> u32 { s as u32 }",
            "out[id.x] = id.x;",
        ),
        Code::EnumShape,
    );
}

#[test]
fn an_enum_is_never_stored_gust_e0211() {
    // A struct field; an `Option` or `Result` result (a struct field of `Option` type is GUST-E0501 first).
    for items in [
        "pub struct S { pub m: Mode }",
        "fn f() -> Option<Mode> { Some(Mode::A) }",
        "fn f() -> Result<Mode, Mode> { Ok(Mode::A) }",
    ] {
        assert_rejected(&with_mode(items, "out[id.x] = id.x;"), Code::EnumInResource);
    }
    let kernel = |parameter: &str| {
        format!(
            "mod m {{ {MODE_ENUM} #[kernel] pub fn run(id: SV_DispatchThreadID, {parameter}, mut out: RWStructuredBuffer<uint>) {{ out[id.x] = 1u32; }} }}"
        )
    };
    for parameter in [
        "modes: StructuredBuffer<Mode>",
        "mut modes: RWStructuredBuffer<Mode>",
    ] {
        assert_rejected(&kernel(parameter), Code::EnumInResource);
    }
}

#[test]
fn a_match_arm_on_an_enum_takes_no_guard_gust_e0418() {
    assert_rejected(
        &with_mode(
            "",
            "match Mode::A { Mode::A if id.x > 1u32 => { out[id.x] = 1u32; } _ => { out[id.x] = 2u32; } }",
        ),
        Code::EnumMatchGuard,
    );
}

#[test]
fn a_break_or_continue_in_an_arm_aimed_at_the_loop_is_gust_e0417() {
    for jump in ["break;", "continue;"] {
        let body = format!(
            "for k in 0u32..4u32 {{ match Mode::A {{ Mode::A => {{ out[id.x] = k; {jump} }} _ => {{ }} }} }}"
        );
        assert_rejected(&with_mode("", &body), Code::JumpInMatchArm);
    }
    // Nested: the arm of an inner `match` still binds to its own `switch`.
    assert_rejected(
        &with_mode(
            "",
            "for k in 0u32..4u32 { match Mode::A { Mode::A => { match Mode::B { Mode::B => { break; } _ => { } } } _ => { } } }",
        ),
        Code::JumpInMatchArm,
    );
    // A `for` inside the arm is its own target; a jump after the `match` is the loop target.
    translate(&with_mode(
        "",
        "for k in 0u32..4u32 { match Mode::A { Mode::A => { for j in 0u32..2u32 { if j == 1u32 { break; } out[id.x] = j; } } _ => { } } if k == 2u32 { break; } }",
    ))
    .unwrap();
}

#[test]
fn other_arm_patterns_scrutinees_and_bodies_of_an_enum_match_are_refused() {
    // A binding, a literal, a wildcard inside `|`, an unknown variant, a repeated variant, an arm after `_`, another enum.
    let arms = [
        "match Mode::A { Mode::A => { } other => { out[id.x] = 1u32; } }",
        "match Mode::A { Mode::A => { } 1u32 => { } _ => { } }",
        "match Mode::A { Mode::A | _ => { } }",
        "match Mode::A { Mode::A => { } Mode::Z => { } _ => { } }",
        "match Mode::A { Mode::A => { } Mode::A => { } _ => { } }",
        "match Mode::A { _ => { } Mode::A => { } }",
        "match Mode::A { Mode::A => { } Other::A => { } _ => { } }",
    ];
    for arm in arms {
        assert_rejected(&with_mode("", arm), Code::MatchPattern);
    }
    // The scrutinee: a field.
    assert_rejected(
        &with_mode("", "match id.x { Mode::A => { } _ => { } }"),
        Code::MatchScrutinee,
    );
    // An arm that is not a block.
    assert_rejected(
        &with_mode(
            "",
            "match Mode::A { Mode::A => out[id.x] = 1u32, _ => { } }",
        ),
        Code::MatchArmBlock,
    );
    // The value form belongs to a later slice: a tail expression in an arm, or the `match` as a value.
    assert_rejected(
        &with_mode("", "match Mode::A { Mode::A => { 1u32 } _ => { 2u32 } }"),
        Code::MatchAsValue,
    );
    assert_rejected(
        &with_mode(
            "",
            "let x: u32 = match Mode::A { Mode::A => { 1u32 } _ => { 2u32 } }; out[id.x] = x;",
        ),
        Code::MatchAsValue,
    );
}

#[test]
fn an_impl_on_an_enum_is_gust_e0121_and_says_why() {
    let source = with_mode(
        "impl Mode { fn one(&self) -> u32 { 1u32 } }",
        "out[id.x] = 1u32;",
    );
    let error = translate(&source).expect_err(&source);
    assert_code_for(&error, Code::ImplForeignType, &source);
    assert!(error.to_string().contains("methods on an enum"), "{error}");
}

#[test]
fn enum_values_are_locals_parameters_and_results_and_compare_with_eq_and_ne() {
    translate(&with_mode(
        "fn pick(m: Mode, n: Mode) -> Mode { if m == n { return n; } if m != Mode::A { return m; } Mode::C }",
        "let m = pick(Mode::A, Mode::B); let n: Mode = Mode::C; if m != n { out[id.x] = pick(n, m) as u32; }",
    ))
    .unwrap();
}

#[test]
fn a_return_in_an_arm_of_a_helper_is_legal() {
    translate(&with_mode(
        "fn g(m: Mode) -> u32 { match m { Mode::A => { return 1u32; } _ => { } } 0u32 }",
        "out[id.x] = g(Mode::B);",
    ))
    .unwrap();
}

#[test]
fn an_enum_name_that_slang_reserves_is_renamed_in_every_position() {
    // `class` is a Slang keyword: the declaration, the label and the value all go through `emit_name`.
    let source = translate(&with_kernel(
        "#[derive(PartialEq, Eq)] pub enum class { new, other }",
        "let c = class::new; match c { class::new => { out[id.x] = 1u32; } _ => { } }",
    ))
    .unwrap();
    assert!(
        source.contains("case __gust_r_class::__gust_r_new:"),
        "{source}"
    );
    assert!(source.contains("enum __gust_r_class : uint"), "{source}");
}
