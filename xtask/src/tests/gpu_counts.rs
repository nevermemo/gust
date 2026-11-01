use super::{
    GPU_SUITE_CALL, GPU_SUITE_PIN, GpuSuitePin, HelperPlan, VerifyMode, cargo_calls,
    check_gpu_suite_counts, check_helper_tag, environment_helper, guard_call, helper_plan,
    helper_tag, helper_version_output, pins_gpu_counts, printed_text,
};
use crate::process::CheckRecord;
use std::ffi::OsString;

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

/// Twenty result lines that sum to 146, the last one a doc-test run with no test, as `cargo test -p gust-wgpu` prints
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
    let mut out = String::from("   Compiling gust-wgpu v0.1.0\n    Finished `dev` profile\n");
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

/// The `passed` counts of the 20 result lines that `cargo test --locked -p gust-wgpu` printed on Mesa lavapipe (WSL,
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
        error.starts_with("cargo test --locked -p gust-wgpu: "),
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
            assert_eq!(cargo_calls(mode), vec![call()], "{mode:?}");
        } else {
            assert!(pinned.is_empty(), "{mode:?}");
        }
    }
    assert!(!pins_gpu_counts(
        VerifyMode::Gpu,
        &owned(&["test", "--locked", "-p", "gust"])
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
