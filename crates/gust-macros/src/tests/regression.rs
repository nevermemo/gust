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
fn assert_golden(actual: &str, fixture: &str, expected: &str) {
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
    assert!(source.contains("uint answer = (x / divisor);"), "{source}");
    assert!(source.contains("uint result = quotient(id.x);"), "{source}");
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
    let item: proc_macro2::TokenStream = "fn f() {}".parse().unwrap();
    let items = output_items(crate::gpu_impl(proc_macro2::TokenStream::new(), item));
    assert_eq!(items.len(), 2);
    assert_leading_compile_error(&items, Some(Code::ItemNotModule));
    let expected: syn::Item = syn::parse_quote!(
        fn f() {}
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
    assert_eq!(item.items.len(), 3);
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
            "< add as :: gust :: Kernel > :: DESCRIPTOR . storage_parameter_count () == 3 ,"
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
    assert!(
        output
            .to_string()
            .contains("storage_parameter_count () == 0 ,")
    );
}

#[test]
fn a_kernel_with_a_uniform_parameter_gets_no_typed_call() {
    let output = expansion(
        "mod odd { #[kernel] pub fn run(id: SV_DispatchThreadID, settings: Uniform<float>, \
         mut out: RWStructuredBuffer<float>) { let i = id.x; if i < out.len() {} } }",
    );
    assert!(typed_impl(&output, "run").is_none());
    let text = output.to_string();
    // The handle and its descriptor are unchanged.
    assert!(text.contains("impl :: gust :: Kernel for run"));
    assert!(text.contains("ParameterKind :: Uniform"));
    assert!(!text.contains("fn call"));
    assert!(!text.contains("storage_parameter_count"));
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
    assert!(call.contains("Resource < Element = Particle >"), "{call}");
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

    let uniform = expansion(
        "mod odd { #[kernel] pub fn run(id: SV_DispatchThreadID, settings: Uniform<float>, \
         mut out: RWStructuredBuffer<float>) { let i = id.x; if i < out.len() {} } }",
    )
    .to_string();
    assert!(
        uniform.contains("has a uniform parameter, so it has no typed"),
        "{uniform}"
    );
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
                    ::gust::cpu::invocations(__gust_call.kernel(), __gust_call.element_count())
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
fn a_uniform_kernel_gets_no_twin() {
    let text = expansion(
        "mod odd { #[kernel] pub fn run(id: SV_DispatchThreadID, settings: Uniform<float>, \
         mut out: RWStructuredBuffer<float>) { let i = id.x; if i < out.len() {} } }",
    )
    .to_string();
    assert!(!text.contains("impl :: gust :: Run"), "{text}");
    assert!(!text.contains("any (test , gust_dual)"), "{text}");
    assert!(!text.contains("CpuDevice"), "{text}");
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
                    ::gust::cpu::invocations(__gust_call.kernel(), __gust_call.element_count())
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
    assert_eq!(
        squeezed(&cpu),
        squeezed(&other.replace(TWIN_CFG, UNGATED_IMPL))
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
    // A uniform parameter: no typed call, so no CPU path until M6 (D40).
    let uniform = "mod odd { #[kernel] pub fn run(id: SV_DispatchThreadID, settings: Uniform<float>, \
         mut out: RWStructuredBuffer<float>) { let i = id.x; if i < out.len() {} } }";
    for source in [opted_out, uniform] {
        let cpu = expansion_for(source, Profile::Cpu).to_string();
        let other = expansion_for(source, Profile::Other).to_string();
        assert!(!cpu.contains("impl :: gust :: Run"), "{cpu}");
        assert!(!cpu.contains(":: cpu :: CpuDevice"), "{cpu}");
        assert_eq!(squeezed(&cpu), squeezed(&other));
    }
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
fn a_kernel_with_a_uniform_parameter_still_asserts_its_storage_elements() {
    let text = expansion(
        "mod odd { #[kernel] pub fn run(id: SV_DispatchThreadID, settings: Uniform<float>, \
         mut out: RWStructuredBuffer<int>) { let i = id.x; if i < out.len() {} } }",
    )
    .to_string();
    assert_eq!(
        count(&text, "layout does not match its Rust representation"),
        1,
        "{text}"
    );
    assert!(
        text.contains(
            "< run as :: gust :: Kernel > :: DESCRIPTOR . parameters [2] . resource_layout"
        ),
        "{text}"
    );
    // The uniform parameter is not a storage element.
    assert!(!text.contains("size_of :: < float > ()"), "{text}");
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
        13,
        "the mismatch fixture declares 13 kernels"
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
        "var d = dot(p, q);",
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
