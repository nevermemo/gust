use super::{TestTotals, require_tests_ran, sum_test_results, without_colors};

fn owned(args: &[&str]) -> Vec<String> {
    args.iter().map(|arg| (*arg).to_owned()).collect()
}

fn result(status: &str, passed: u32, failed: u32, ignored: u32) -> String {
    format!(
        "test result: {status}. {passed} passed; {failed} failed; {ignored} ignored; 0 measured; 3 filtered out; finished in 0.01s\n"
    )
}

const PLAIN: &[&str] = &["test", "--locked", "-p", "vector-add"];
const INCLUDE: &[&str] = &["test", "-p", "vector-add", "--", "--include-ignored"];

fn verdict(args: &[&str], output: &str) -> Result<(), String> {
    require_tests_ran(&owned(args), &sum_test_results(output))
}

#[test]
fn zero_tests_in_every_binary_fail_the_guard() {
    // What `cargo test -p x some_filter_that_matches_nothing` prints: one line per binary, all empty.
    let output = format!(
        "running 0 tests\n\n{}running 0 tests\n\n{}   Doc-tests x\nrunning 0 tests\n\n{}",
        result("ok", 0, 0, 0),
        result("ok", 0, 0, 0),
        result("ok", 0, 0, 0)
    );
    let totals = sum_test_results(&output);
    assert_eq!(
        totals,
        TestTotals {
            passed: 0,
            failed: 0,
            ignored: 0,
            summaries: 3
        }
    );
    let error = verdict(PLAIN, &output).unwrap_err();
    assert!(error.contains("ran 0 tests"), "{error}");
    assert!(error.contains("3 test result line(s)"), "{error}");
}

#[test]
fn no_result_line_at_all_fails_the_guard() {
    let error = verdict(PLAIN, "").unwrap_err();
    assert!(error.contains("ran 0 tests"), "{error}");
    assert!(error.contains("0 test result line(s)"), "{error}");
}

#[test]
fn several_binaries_and_doc_tests_are_summed() {
    let output = format!(
        "{}{}{}",
        result("ok", 12, 0, 0),
        result("ok", 5, 0, 36),
        result("ok", 0, 0, 0)
    );
    let totals = sum_test_results(&output);
    assert_eq!((totals.passed, totals.failed, totals.ignored), (17, 0, 36));
    assert_eq!(totals.summaries, 3);
    // A single binary with tests is enough even when the others (doc-tests) ran none.
    assert_eq!(verdict(PLAIN, &output), Ok(()));
    let only_docs = format!("{}{}", result("ok", 0, 0, 0), result("ok", 2, 0, 0));
    assert_eq!(verdict(PLAIN, &only_docs), Ok(()));
}

#[test]
fn ignored_tests_are_fine_without_include_ignored_and_fail_with_it() {
    let output = result("ok", 4, 0, 2);
    assert_eq!(verdict(PLAIN, &output), Ok(()));
    let error = verdict(INCLUDE, &output).unwrap_err();
    assert!(error.contains("2 test(s) reported ignored"), "{error}");
    assert_eq!(verdict(INCLUDE, &result("ok", 6, 0, 0)), Ok(()));
    // All tests ignored and none run: zero passed fails first, with or without the flag.
    assert!(verdict(INCLUDE, &result("ok", 0, 0, 6)).is_err());
    assert!(verdict(PLAIN, &result("ok", 0, 0, 6)).is_err());
}

#[test]
fn a_failed_line_fails_even_beside_passing_binaries() {
    let output = format!("{}{}", result("ok", 9, 0, 0), result("FAILED", 3, 1, 0));
    let totals = sum_test_results(&output);
    assert_eq!((totals.passed, totals.failed), (12, 1));
    let error = verdict(PLAIN, &output).unwrap_err();
    assert!(error.contains("1 test(s) failed"), "{error}");
    // A status that is not `ok` counts as a failure even when no failed test is named.
    assert_eq!(sum_test_results(&result("FAILED", 3, 0, 0)).failed, 1);
}

#[test]
fn colour_codes_and_surrounding_output_do_not_hide_the_counts() {
    let coloured = "test a ... \u{1b}[32mok\u{1b}[0m\ntest result: \u{1b}[32mok\u{1b}[0m. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n";
    assert_eq!(without_colors(coloured).matches('\u{1b}').count(), 0);
    let totals = sum_test_results(coloured);
    assert_eq!((totals.passed, totals.summaries), (2, 1));
    assert_eq!(verdict(PLAIN, coloured), Ok(()));
    // A test whose own output mentions the words is not a result line.
    let noise = "test says: test result: ok. 100 passed\nwarning: 5 passed; 0 failed\n";
    assert_eq!(sum_test_results(noise).summaries, 0);
}

#[test]
fn an_unreadable_result_line_counts_as_nothing() {
    let totals = sum_test_results("test result: ok. many passed; 0 failed; 0 ignored\n");
    assert_eq!(totals, TestTotals::default());
    assert!(verdict(PLAIN, "test result: ok. 5 passed\n").is_err());
}
