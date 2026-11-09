use super::{
    GPU_SUITE_CALL, GPU_SUITE_PIN, GpuSuitePin, HelperPlan, PinRule, VerifyMode, cargo_calls,
    check_gpu_suite_counts, check_helper_pin, check_helper_tag, check_slangc_pin,
    environment_helper, guard_call, helper_plan, helper_tag, helper_version_output, pin_rule,
    pin_rule_of, pins_gpu_counts, prepare_helper_with, printed_text, slang_pin,
    slang_validated_builds,
};
use crate::process::CheckRecord;
use std::{cell::Cell, ffi::OsString, path::Path};

const ALL_MODES: [VerifyMode; 6] = [
    VerifyMode::Smoke,
    VerifyMode::Fast,
    VerifyMode::Gpu,
    VerifyMode::Examples,
    VerifyMode::Artifacts,
    VerifyMode::Full,
];

/// A pin that does not follow the real one, so these tests do not change when `GPU_SUITE_PIN` is bumped.
const PIN: GpuSuitePin = GpuSuitePin {
    passed: 146,
    result_lines: 20,
};

/// Twenty result lines that sum to 146, the last one a doc-test run with no test, as `cargo test -p gust-compute-wgpu` prints
/// them: the library, each integration test file, then the doc tests.
const PASSED: [u32; 20] = [
    10, 7, 5, 9, 6, 8, 11, 4, 3, 12, 9, 7, 6, 8, 10, 5, 6, 8, 12, 0,
];

fn owned(args: &[&str]) -> Vec<String> {
    args.iter().map(|arg| (*arg).to_owned()).collect()
}

fn call() -> Vec<String> {
    owned(&GPU_SUITE_CALL)
}

fn line(status: &str, passed: u32, failed: u32, ignored: u32) -> String {
    format!(
        "test result: {status}. {passed} passed; {failed} failed; {ignored} ignored; 0 measured; 0 filtered out; finished in 0.31s\n"
    )
}

/// The captured standard output of the call: for each binary a `Running` line, the tests, a blank line and the result.
fn stdout_of(passed: &[u32]) -> String {
    let mut out =
        String::from("   Compiling gust-compute-wgpu v0.1.0\n    Finished `dev` profile\n");
    for (index, count) in passed.iter().enumerate() {
        out.push_str(&format!(
            "     Running tests/binary_{index}.rs\n\nrunning {count} tests\n"
        ));
        for test in 0..*count {
            out.push_str(&format!("test case_{index}_{test} ... ok\n"));
        }
        out.push('\n');
        out.push_str(&line("ok", *count, 0, 0));
        out.push('\n');
    }
    out
}

fn good() -> String {
    stdout_of(&PASSED)
}

#[test]
fn the_pinned_output_passes() {
    assert_eq!(PASSED.iter().sum::<u32>(), 146);
    assert_eq!(check_gpu_suite_counts(&call(), &good(), PIN), Ok(()));
}

/// The `passed` counts of the 20 result lines that `cargo test --locked -p gust-compute-wgpu` printed on Mesa lavapipe (WSL,
/// slice m5.a2 dry run at 672c5ca), in order.
const LAVAPIPE_RUN: [u32; 20] = [
    38, 4, 9, 1, 4, 15, 17, 2, 6, 2, 2, 8, 6, 4, 2, 2, 19, 2, 2, 1,
];

#[test]
fn the_counts_of_the_measured_lavapipe_run_pass_and_one_more_test_does_not() {
    assert_eq!(LAVAPIPE_RUN.iter().sum::<u32>(), 146);
    assert_eq!(
        check_gpu_suite_counts(&call(), &stdout_of(&LAVAPIPE_RUN), PIN),
        Ok(())
    );
    let mut more = LAVAPIPE_RUN;
    more[16] += 1;
    assert!(check_gpu_suite_counts(&call(), &stdout_of(&more), PIN).is_err());
}

/// The passed counts of an output built to `pin`: its passed count spread over its number of lines.
fn passed_for(pin: GpuSuitePin) -> Vec<u32> {
    let lines = usize::try_from(pin.result_lines).unwrap();
    let mut passed = vec![0u32; lines];
    for index in 0..pin.passed {
        passed[usize::try_from(index).unwrap() % lines] += 1;
    }
    passed
}

#[test]
fn the_real_pin_passes_an_output_built_to_it() {
    // Follows `GPU_SUITE_PIN`, so it does not change when the pin is bumped.
    let mut passed = passed_for(GPU_SUITE_PIN);
    assert_eq!(
        check_gpu_suite_counts(&call(), &stdout_of(&passed), GPU_SUITE_PIN),
        Ok(())
    );
    // One line fewer or one test more fails the real pin.
    assert!(check_gpu_suite_counts(&call(), &stdout_of(&passed[1..]), GPU_SUITE_PIN).is_err());
    passed[0] += 1;
    assert!(check_gpu_suite_counts(&call(), &stdout_of(&passed), GPU_SUITE_PIN).is_err());
}

#[test]
fn colored_result_lines_are_read_like_plain_ones() {
    // `CARGO_TERM_COLOR=always`, as CI sets it, puts the colour around `ok`.
    let colored = good().replace("test result: ok.", "test result: \u{1b}[32mok\u{1b}[0m.");
    assert_ne!(colored, good());
    assert_eq!(check_gpu_suite_counts(&call(), &colored, PIN), Ok(()));
}

#[test]
fn mutation_a_one_failed_line_fails() {
    let bad = good().replacen(&line("ok", 9, 0, 0), &line("FAILED", 8, 1, 0), 1);
    assert_ne!(bad, good());
    let error = check_gpu_suite_counts(&call(), &bad, PIN).unwrap_err();
    assert!(error.contains("1 failed, 0 ignored"), "{error}");
    // The passed sum is also off by one here, so both findings are named.
    assert!(error.contains("145 tests passed"), "{error}");
}

#[test]
fn a_failed_line_that_keeps_the_passed_sum_still_fails() {
    // 146 passed over 20 lines, but one line also reports a failure: only the failure check can catch it.
    let bad = good().replacen(&line("ok", 0, 0, 0), &line("FAILED", 0, 1, 0), 1);
    assert_ne!(bad, good());
    let error = check_gpu_suite_counts(&call(), &bad, PIN).unwrap_err();
    assert!(error.contains("1 failed"), "{error}");
    assert!(!error.contains("tests passed, the pinned count"), "{error}");
}

#[test]
fn an_ignored_test_fails() {
    let bad = good().replacen(&line("ok", 4, 0, 0), &line("ok", 4, 0, 1), 1);
    assert_ne!(bad, good());
    let error = check_gpu_suite_counts(&call(), &bad, PIN).unwrap_err();
    assert!(error.contains("0 failed, 1 ignored"), "{error}");
}

#[test]
fn mutation_b_a_passed_count_off_by_one_fails_and_names_both_numbers() {
    let mut more = PASSED;
    more[0] += 1;
    let error = check_gpu_suite_counts(&call(), &stdout_of(&more), PIN).unwrap_err();
    assert!(
        error.contains("147 tests passed, the pinned count is 146"),
        "{error}"
    );
    assert!(!error.contains("test result lines, the pinned"), "{error}");
    let mut fewer = PASSED;
    fewer[0] -= 1;
    let error = check_gpu_suite_counts(&call(), &stdout_of(&fewer), PIN).unwrap_err();
    assert!(
        error.contains("145 tests passed, the pinned count is 146"),
        "{error}"
    );
}

#[test]
fn mutation_c_a_missing_result_line_fails_the_line_count() {
    // The missing line is the doc-test run with no test: the passed sum is unchanged, only the line count catches it.
    let without = good().replacen(&line("ok", 0, 0, 0), "", 1);
    assert_ne!(without, good());
    let error = check_gpu_suite_counts(&call(), &without, PIN).unwrap_err();
    assert!(
        error.contains("19 test result lines, the pinned number is 20"),
        "{error}"
    );
    assert!(!error.contains("tests passed, the pinned count"), "{error}");
}

#[test]
fn drift_that_cancels_out_in_the_sum_fails_the_line_count() {
    // One binary gained two tests and another lost two: the sum holds. A new file with no test changes only the lines.
    let mut moved = PASSED;
    moved[0] += 2;
    moved[1] -= 2;
    assert_eq!(
        check_gpu_suite_counts(&call(), &stdout_of(&moved), PIN),
        Ok(())
    );
    let mut extra = PASSED.to_vec();
    extra.push(0);
    let error = check_gpu_suite_counts(&call(), &stdout_of(&extra), PIN).unwrap_err();
    assert!(
        error.contains("21 test result lines, the pinned number is 20"),
        "{error}"
    );
}

#[test]
fn an_output_without_result_lines_fails() {
    let error = check_gpu_suite_counts(&call(), "running 0 tests\n", PIN).unwrap_err();
    assert!(error.contains("no `test result:` line"), "{error}");
    assert!(check_gpu_suite_counts(&call(), "", PIN).is_err());
}

#[test]
fn an_unreadable_result_line_fails_and_is_named() {
    let bad = good().replacen(
        &line("ok", 9, 0, 0),
        "test result: ok. many passed; 0 failed; 0 ignored\n",
        1,
    );
    let error = check_gpu_suite_counts(&call(), &bad, PIN).unwrap_err();
    assert!(
        error.contains("1 of 20 `test result:` line(s) could not be read"),
        "{error}"
    );
}

#[test]
fn the_error_names_the_call_and_where_the_pin_lives() {
    let mut more = PASSED;
    more[0] += 1;
    let error = check_gpu_suite_counts(&call(), &stdout_of(&more), PIN).unwrap_err();
    assert!(
        error.starts_with("cargo test --locked -p gust-compute-wgpu: "),
        "{error}"
    );
    assert!(
        error.contains("GPU_SUITE_PIN in xtask/src/verify.rs"),
        "{error}"
    );
}

#[test]
fn only_the_gpu_mode_makes_the_pinned_call() {
    for mode in ALL_MODES {
        let pinned = cargo_calls(mode)
            .into_iter()
            .filter(|call| pins_gpu_counts(mode, call))
            .collect::<Vec<_>>();
        if mode == VerifyMode::Gpu {
            assert_eq!(pinned, vec![call()], "{mode:?}");
            // The algorithms call follows (m7.r1); it is judged by name, not by the wgpu pin.
            assert_eq!(
                cargo_calls(mode),
                vec![call(), owned(&["test", "--locked", "-p", "gust-algo"])],
                "{mode:?}"
            );
        } else {
            assert!(pinned.is_empty(), "{mode:?}");
        }
    }
    assert!(!pins_gpu_counts(
        VerifyMode::Gpu,
        &owned(&["test", "--locked", "-p", "gust-compute"])
    ));
}

#[test]
fn the_guard_applies_the_pin_in_the_gpu_mode_only() {
    let mut passed = passed_for(GPU_SUITE_PIN);
    let pinned = stdout_of(&passed);
    passed[0] += 1;
    let drifted = stdout_of(&passed);
    assert_eq!(guard_call(VerifyMode::Gpu, &call(), &pinned), Ok(()));
    assert!(guard_call(VerifyMode::Gpu, &call(), &drifted).is_err());
    assert!(guard_call(VerifyMode::Gpu, &call(), "running 0 tests\n").is_err());
    // The Fast mode makes the same call, but it is not pinned.
    assert_eq!(guard_call(VerifyMode::Fast, &call(), &drifted), Ok(()));
    assert_eq!(
        guard_call(VerifyMode::Fast, &call(), "running 0 tests\n"),
        Ok(())
    );
    // The examples keep their own guard.
    let example = owned(&["test", "-p", "polynomial", "--", "--include-ignored"]);
    assert!(guard_call(VerifyMode::Examples, &example, "running 0 tests\n").is_err());
}

#[test]
fn a_helper_named_by_the_environment_must_exist() {
    assert_eq!(environment_helper(None), None);
    assert_eq!(environment_helper(Some(OsString::new())), None);
    let missing = std::env::temp_dir().join("gust-m5a2-no-such-helper-file");
    assert_eq!(environment_helper(Some(missing.into_os_string())), None);
    // A directory is not a helper.
    assert_eq!(
        environment_helper(Some(std::env::temp_dir().into_os_string())),
        None
    );
    let this_test_binary = std::env::current_exe().unwrap();
    assert_eq!(
        environment_helper(Some(this_test_binary.clone().into_os_string())),
        Some(this_test_binary)
    );
}

fn helper_name() -> String {
    format!("gust-slang-reflect{}", std::env::consts::EXE_SUFFIX)
}

/// A fresh scratch directory under the temp directory, removed by the caller.
fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("gust-m5a2-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn touch(path: &std::path::Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, b"x").unwrap();
}

#[test]
fn the_default_output_is_built_and_any_other_named_helper_is_used() {
    let root = scratch("plan");
    let default = root
        .join("target")
        .join("slang-reflect")
        .join(helper_name());
    let other = root.join("elsewhere").join(helper_name());
    touch(&default);
    touch(&other);
    let named = |path: &std::path::Path| Some(path.as_os_str().to_owned());
    // No variable, or a file that does not exist: build into the default place.
    assert_eq!(helper_plan(&root, None), HelperPlan::Build);
    assert_eq!(
        helper_plan(&root, named(&root.join("missing").join(helper_name()))),
        HelperPlan::Build
    );
    // The default output itself, however it is spelled: still built, so it follows the helper source and Slang.
    assert_eq!(helper_plan(&root, named(&default)), HelperPlan::Build);
    let roundabout = root
        .join("target")
        .join("slang-reflect")
        .join("..")
        .join("slang-reflect")
        .join(helper_name());
    assert_eq!(helper_plan(&root, named(&roundabout)), HelperPlan::Build);
    // Another file is used (after the tag check), and so is the default output of another checkout.
    assert_eq!(helper_plan(&root, named(&other)), HelperPlan::Use(other));
    let second_root = scratch("plan-second");
    assert_eq!(
        helper_plan(&second_root, named(&default)),
        HelperPlan::Use(default)
    );
    let _ = std::fs::remove_dir_all(root);
    let _ = std::fs::remove_dir_all(second_root);
}

#[test]
fn the_helper_tag_is_the_slang_word_of_the_version_line() {
    assert_eq!(
        helper_tag("gust-slang-reflect schema=1 slang=2026.13.1\n"),
        Some("2026.13.1")
    );
    assert_eq!(
        helper_tag("\ngust-slang-reflect schema=1 slang=2026.13.1-1-g84792eb15\n"),
        Some("2026.13.1-1-g84792eb15")
    );
    assert_eq!(helper_tag("gust-slang-reflect schema=1\n"), None);
    assert_eq!(helper_tag("slang= schema=1"), None);
    assert_eq!(helper_tag(""), None);
}

#[test]
fn a_helper_is_used_only_when_its_tag_equals_slangc_version() {
    let helper = |tag: &str| format!("gust-slang-reflect schema=1 slang={tag}\n");
    assert_eq!(
        check_helper_tag(&helper("2026.13.1"), "2026.13.1\n"),
        Ok(())
    );
    assert_eq!(
        check_helper_tag(&helper("2026.13.1"), "\r\n  2026.13.1  \r\n"),
        Ok(())
    );
    // A source build is its own tag: helper and compiler must agree on it.
    let build = "2026.13.1-1-g84792eb15";
    assert_eq!(check_helper_tag(&helper(build), build), Ok(()));
    // A mismatch names both tags.
    let error = check_helper_tag(&helper("2026.12.0"), "2026.13.1\n").unwrap_err();
    assert!(
        error.contains("`2026.12.0`") && error.contains("`2026.13.1`"),
        "{error}"
    );
    assert!(error.contains("GUST_SLANG_REFLECT"), "{error}");
    // The release and the source build of the same version are different compilers.
    assert!(check_helper_tag(&helper("2026.13.1"), build).is_err());
    assert!(check_helper_tag(&helper(build), "2026.13.1").is_err());
    // A prefix is not a match.
    assert!(check_helper_tag(&helper("2026.13"), "2026.13.1").is_err());
    // No tag, or no compiler version, never passes.
    assert!(check_helper_tag("gust-slang-reflect schema=1", "2026.13.1").is_err());
    assert!(check_helper_tag(&helper("2026.13.1"), "").is_err());
    assert!(check_helper_tag(&helper("2026.13.1"), "\n  \n").is_err());
}

#[test]
fn a_helper_that_cannot_run_is_an_error_that_hides_its_path() {
    let missing = std::env::temp_dir()
        .join("gust-m5a2-no-such-dir")
        .join(helper_name());
    let error = helper_version_output(&missing).unwrap_err();
    assert!(error.contains("GUST_SLANG_REFLECT"), "{error}");
    assert!(!error.contains("gust-m5a2-no-such-dir"), "{error}");
}

#[test]
fn slangc_prints_its_version_on_standard_error_and_that_counts() {
    let record = CheckRecord {
        program: "slangc".to_owned(),
        arguments: vec!["-version".to_owned()],
        exit_code: Some(0),
        stdout: String::new(),
        stderr: "2026.13.1\n".to_owned(),
    };
    let helper = "gust-slang-reflect schema=1 slang=2026.13.1\n";
    assert_eq!(check_helper_tag(helper, &printed_text(&record)), Ok(()));
    let on_stdout = CheckRecord {
        stdout: "2026.13.1\n".to_owned(),
        stderr: String::new(),
        ..record
    };
    assert_eq!(check_helper_tag(helper, &printed_text(&on_stdout)), Ok(()));
}

// ---- the helper must be the pin's (slice m4.r) -------------------------------------------------------------------

/// The text of the lines of `gust::toolchain` that hold the pin, with a pin that does not follow the real one.
fn pin_source(pin: &str) -> String {
    format!(
        "/// The release.\npub const SLANG_PIN: &str = \"{pin}\";\n\n/// Other tags.\npub const VALIDATED_BUILDS: &[&str] = &[];\n"
    )
}

#[test]
fn the_pin_is_the_string_of_the_one_slang_pin_constant() {
    assert_eq!(
        slang_pin(&pin_source("2026.99")).as_deref(),
        Some("2026.99")
    );
    // CRLF, as the checkout has it, and indentation.
    let crlf = pin_source("2026.99").replace('\n', "\r\n");
    assert_eq!(slang_pin(&crlf).as_deref(), Some("2026.99"));
    // Not exactly one constant, an empty one, or another type: no pin.
    assert_eq!(slang_pin(""), None);
    assert_eq!(slang_pin(&pin_source("")), None);
    let twice = format!("{}{}", pin_source("2026.99"), pin_source("2026.98"));
    assert_eq!(slang_pin(&twice), None);
    assert_eq!(
        slang_pin("pub const SLANG_PIN: &[&str] = &[\"2026.99\"];\n"),
        None
    );
    assert_eq!(slang_pin("// SLANG_PIN: &str = \"2026.99\";\n"), None);
    // The real file has one, of the shape of a Slang release.
    let real = pin_rule(&crate::workspace::workspace_root().unwrap())
        .unwrap()
        .pin;
    assert!(
        real.split('.').count() >= 2
            && real
                .split('.')
                .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit())),
        "{real}"
    );
}

fn rule(pin: &str, validated: &[&str]) -> PinRule {
    PinRule {
        pin: pin.to_owned(),
        validated: validated.iter().map(|tag| (*tag).to_owned()).collect(),
    }
}

/// No `slangc` on `PATH` to ask.
fn no_slangc() -> Option<String> {
    None
}

#[test]
fn the_validated_builds_are_read_from_the_source_that_holds_the_pin() {
    // Shipped shape: an empty list.
    assert_eq!(
        slang_validated_builds(&pin_source("2026.99")),
        Some(Vec::new())
    );
    // A list over several lines, with comments, CRLF as the checkout has it.
    let listed = "pub const VALIDATED_BUILDS: &[&str] = &[\n    \"2026.99-1-gabc\", // the first\n    // \"2026.99-9-gnot\",\n    \"2026.99-2-gdef\",\n];\n"
        .replace('\n', "\r\n");
    assert_eq!(
        slang_validated_builds(&listed),
        Some(vec![
            "2026.99-1-gabc".to_owned(),
            "2026.99-2-gdef".to_owned()
        ])
    );
    // No constant, or one that is not closed: no rule.
    assert_eq!(slang_validated_builds(""), None);
    assert_eq!(
        slang_validated_builds("pub const VALIDATED_BUILDS: &[&str] = &["),
        None
    );
    // The spelling of the home before mr.t2 is no pin any more.
    assert!(pin_rule_of("pub(crate) const SLANG_VERSION: &str = \"2026.99\";\n").is_err());
    // The rule of `matches_pin`: the pin itself, or a listed build that extends it; nothing else.
    let listed = rule("2026.99", &["2026.99-1-gabc"]);
    for ok in ["2026.99", "2026.99-1-gabc"] {
        assert!(listed.admits(ok), "{ok}");
    }
    for refused in [
        "2026.99-2-gabc",
        "2026.9",
        "2026.13.1",
        "",
        "2026.99-1-gabcd",
    ] {
        assert!(!listed.admits(refused), "{refused}");
    }
    assert!(!rule("2026.99", &[]).admits("2026.99-1-gabc"));
    // The real source: a rule is read, and every listed build extends the pin.
    let root = crate::workspace::workspace_root().unwrap();
    let real =
        pin_rule_of(&std::fs::read_to_string(root.join(super::PIN_SOURCE)).unwrap()).unwrap();
    assert!(real.validated.iter().all(|tag| tag.starts_with(&real.pin)));
}

#[test]
fn a_helper_is_refused_unless_its_tag_is_the_pin_and_its_schema_is_the_current_one() {
    let helper = |tag: &str| format!("gust-slang-reflect schema=3 slang={tag}\n");
    let pin = rule("2026.20", &[]);
    assert_eq!(
        check_helper_pin(&helper("2026.20"), &pin, "the helper"),
        Ok(())
    );
    // Schema 3 is the only one (mx.k): every compile passes the stage, which a schema 2 helper refuses.
    let error = check_helper_pin(
        "gust-slang-reflect schema=2 slang=2026.20\n",
        &pin,
        "the helper",
    )
    .unwrap_err();
    assert!(
        error.contains("prints `schema=2`")
            && error.contains("needs `schema=3`")
            && error.contains("cargo xtask build-slang-reflect --force"),
        "{error}"
    );
    // The 2026.13.1 helper that every lane used before the bump, in both spellings, is refused with both tags named.
    for old in ["2026.13.1", "2026.13.1-1-g84792eb15"] {
        let error = check_helper_pin(&helper(old), &pin, "the helper named by X").unwrap_err();
        assert!(
            error.contains(&format!("built against Slang `{old}`"))
                && error.contains("the pin is `2026.20`")
                && error.contains("cargo gust slang build-helper")
                && error.contains("the helper named by X"),
            "{error}"
        );
    }
    // A prefix, an extension and a missing tag are not the pin.
    assert!(check_helper_pin(&helper("2026.2"), &pin, "h").is_err());
    assert!(check_helper_pin(&helper("2026.20-1-g0a1b2c3d4"), &pin, "h").is_err());
    assert!(check_helper_pin("gust-slang-reflect schema=3\n", &pin, "h").is_err());
    assert!(check_helper_pin("", &pin, "h").is_err());
    // A validated build of the pin is the pin for the helper too (the one rule of `gust::toolchain`).
    let validated = rule("2026.20", &["2026.20-1-g0a1b2c3d4"]);
    assert_eq!(
        check_helper_pin(&helper("2026.20-1-g0a1b2c3d4"), &validated, "h"),
        Ok(())
    );
    // Schema 4 (arrays, m9.r2h) is accepted beside 3.
    assert_eq!(
        check_helper_pin("gust-slang-reflect schema=4 slang=2026.20\n", &pin, "h"),
        Ok(())
    );
    // Stage 1's helper of the right Slang, with schema 1, is refused; so is one that prints no schema or a later one.
    for output in [
        "gust-slang-reflect schema=1 slang=2026.20\n",
        "gust-slang-reflect slang=2026.20\n",
        "gust-slang-reflect schema=5 slang=2026.20\n",
    ] {
        let error = check_helper_pin(output, &pin, "the helper named by X").unwrap_err();
        assert!(
            error.contains("needs `schema=3`")
                && error.contains("cargo gust slang build-helper")
                && error.contains("the helper named by X"),
            "{error}"
        );
    }
    let error =
        check_helper_pin("gust-slang-reflect schema=1 slang=2026.20\n", &pin, "h").unwrap_err();
    assert!(error.contains("prints `schema=1`"), "{error}");
}

#[test]
fn a_slangc_is_refused_unless_it_reports_the_pin() {
    let pin = rule("2026.20", &["2026.20-1-g0a1b2c3d4"]);
    // The release archive prints the bare pin, on standard output or error, with blank lines around it.
    for ok in ["2026.20\n", "\n2026.20\r\n", "2026.20-1-g0a1b2c3d4\n"] {
        assert_eq!(check_slangc_pin(ok, &pin), Ok(()), "{ok:?}");
    }
    // The VulkanSDK compiler that sits first on PATH on this machine, and the near misses.
    for old in [
        "2026.13.1\n",
        "2026.13.1-1-g84792eb15\n",
        "2026.2\n",
        "2026.20-2-gdeadbeef\n",
    ] {
        let error = check_slangc_pin(old, &pin).unwrap_err();
        assert!(
            error.contains(&format!("`slangc -version` is `{}`", old.trim()))
                && error.contains("the pin is `2026.20`")
                && error.contains("SLANG_SDK"),
            "{error}"
        );
    }
    assert!(check_slangc_pin("", &pin).is_err());
    assert!(check_slangc_pin("\n \n", &pin).is_err());
}

/// A checkout with a pin in `crates/gust/src/toolchain.rs`, and the files `helper_plan` looks at.
fn pinned_checkout(name: &str, pin: &str) -> std::path::PathBuf {
    let root = scratch(name);
    let source = root
        .join("crates")
        .join("gust")
        .join("src")
        .join("toolchain.rs");
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    std::fs::write(source, pin_source(pin)).unwrap();
    root
}

#[test]
fn a_named_helper_is_used_when_it_is_the_pin_and_refused_when_it_is_not() {
    let root = pinned_checkout("prepare-use", "2026.99");
    let other = root.join("elsewhere").join(helper_name());
    touch(&other);
    let named = Some(other.as_os_str().to_owned());
    let built = Cell::new(false);
    let build = || {
        built.set(true);
        Ok(())
    };
    let version = |tag: &'static str| {
        move |_: &Path| Ok(format!("gust-slang-reflect schema=3 slang={tag}\n"))
    };

    // The pin's helper is used, and nothing is built.
    let slangc = "2026.99\n";
    prepare_helper_with(
        &root,
        named.clone(),
        Some(slangc),
        &build,
        &version("2026.99"),
        &no_slangc,
    )
    .unwrap();
    prepare_helper_with(
        &root,
        named.clone(),
        None,
        &build,
        &version("2026.99"),
        &no_slangc,
    )
    .unwrap();
    assert!(!built.get(), "a named helper must not trigger a build");

    // A 13.1 helper is refused with the tag message, and nothing is built either.
    for old in ["2026.13.1", "2026.13.1-1-g84792eb15"] {
        let error = prepare_helper_with(
            &root,
            named.clone(),
            Some(slangc),
            &build,
            &version(old),
            &no_slangc,
        )
        .unwrap_err();
        assert!(
            error.contains(&format!("built against Slang `{old}`"))
                && error.contains("the pin is `2026.99`"),
            "{error}"
        );
        let error = prepare_helper_with(
            &root,
            named.clone(),
            None,
            &build,
            &version(old),
            &no_slangc,
        )
        .unwrap_err();
        assert!(error.contains("the pin is `2026.99`"), "{error}");
    }
    assert!(!built.get());

    // The pin's helper next to a compiler of another build is refused (S1 of the m4.r review: now by the pin rule
    // before the helper-to-compiler comparison).
    let error = prepare_helper_with(
        &root,
        named.clone(),
        Some("2026.13.1\n"),
        &build,
        &version("2026.99"),
        &no_slangc,
    )
    .unwrap_err();
    assert!(
        error.contains("`slangc -version` is `2026.13.1`"),
        "{error}"
    );

    // A helper that cannot run is the run's error, not a pass.
    let broken = |_: &Path| Err("could not run".to_owned());
    assert_eq!(
        prepare_helper_with(&root, named, None, &build, &broken, &no_slangc).unwrap_err(),
        "could not run"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn without_a_named_helper_one_is_built_and_it_must_be_the_pin_too() {
    let root = pinned_checkout("prepare-build", "2026.99");
    let default = root
        .join("target")
        .join("slang-reflect")
        .join(helper_name());
    let built = Cell::new(0);
    let build = || {
        built.set(built.get() + 1);
        Ok(())
    };
    let version = |tag: &'static str| {
        move |_: &Path| Ok(format!("gust-slang-reflect schema=3 slang={tag}\n"))
    };

    // No variable: built, and the build is judged. The built helper is the one that is read.
    let seen = std::cell::RefCell::new(Vec::new());
    let reading = |path: &Path| {
        seen.borrow_mut().push(path.to_owned());
        Ok("gust-slang-reflect schema=3 slang=2026.99\n".to_owned())
    };
    prepare_helper_with(&root, None, None, &build, &reading, &no_slangc).unwrap();
    assert_eq!(
        (built.get(), seen.borrow().clone()),
        (1, vec![default.clone()])
    );

    // A build against another Slang (the VulkanSDK one) is refused instead of being used silently.
    let error = prepare_helper_with(
        &root,
        None,
        None,
        &build,
        &version("2026.13.1-1-g84792eb15"),
        &no_slangc,
    )
    .unwrap_err();
    assert!(
        error.contains("the helper built into target/slang-reflect")
            && error.contains("`2026.13.1-1-g84792eb15`")
            && error.contains("SLANG_SDK"),
        "{error}"
    );
    assert_eq!(built.get(), 2);

    // The variable naming this checkout's own default output, or a file that is not there, builds as before.
    touch(&default);
    let own = Some(default.as_os_str().to_owned());
    prepare_helper_with(&root, own, None, &build, &version("2026.99"), &no_slangc).unwrap();
    let missing = Some(root.join("missing").join(helper_name()).into_os_string());
    prepare_helper_with(
        &root,
        missing,
        None,
        &build,
        &version("2026.99"),
        &no_slangc,
    )
    .unwrap();
    assert_eq!(built.get(), 4);

    // A build that fails is its own error.
    let failing = || Err("build failed".to_owned());
    assert_eq!(
        prepare_helper_with(&root, None, None, &failing, &version("2026.99"), &no_slangc)
            .unwrap_err(),
        "build failed"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn the_slangc_on_path_must_be_the_pin_on_every_path_even_when_nobody_ran_it() {
    // S1 of the m4.r review: a managed SDK and a helper of the pin used to run beside a VulkanSDK slangc that came
    // first on PATH, wherever the caller had not run `slangc -version` itself (Build, the Reflect suite, doctor).
    let root = pinned_checkout("prepare-slangc", "2026.99");
    let other = root.join("elsewhere").join(helper_name());
    touch(&other);
    let named = Some(other.as_os_str().to_owned());
    let built = Cell::new(0);
    let build = || {
        built.set(built.get() + 1);
        Ok(())
    };
    let pins_helper = |_: &Path| Ok("gust-slang-reflect schema=3 slang=2026.99\n".to_owned());
    let old_slangc = || Some("2026.13.1\n".to_owned());
    let pin_slangc = || Some("2026.99\n".to_owned());

    // Without a variable (the Build path) and with one (the Use path), with the caller's output absent: refused,
    // with both tags named, and nothing is built.
    for variable in [None, named.clone()] {
        let error = prepare_helper_with(&root, variable, None, &build, &pins_helper, &old_slangc)
            .unwrap_err();
        assert!(
            error.contains("`slangc -version` is `2026.13.1`")
                && error.contains("the pin is `2026.99`"),
            "{error}"
        );
    }
    // A caller that ran `slangc -version` passes that output, and it is judged without asking PATH again.
    let asked = Cell::new(false);
    let ask = || {
        asked.set(true);
        Some("2026.99\n".to_owned())
    };
    let error = prepare_helper_with(&root, None, Some("2026.13.1\n"), &build, &pins_helper, &ask)
        .unwrap_err();
    assert!(
        error.contains("`slangc -version` is `2026.13.1`"),
        "{error}"
    );
    prepare_helper_with(&root, None, Some("2026.99\n"), &build, &pins_helper, &ask).unwrap();
    assert!(!asked.get());
    assert_eq!(built.get(), 1, "only the passing Build path built");

    // The pin's compiler passes on both paths; no `slangc` on PATH is not this check's error.
    for variable in [None, named.clone()] {
        prepare_helper_with(&root, variable, None, &build, &pins_helper, &pin_slangc).unwrap();
    }
    for variable in [None, named] {
        prepare_helper_with(&root, variable, None, &build, &pins_helper, &no_slangc).unwrap();
    }
    let _ = std::fs::remove_dir_all(root);
}
