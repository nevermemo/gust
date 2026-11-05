use super::{TestExactOptions, check_single_pass, count_listed};

fn summary(status: &str, counts: &str) -> String {
    format!(
        "running 1 test\ntest case ... ok\n\ntest result: {status}. {counts}; 0 measured; 0 filtered out; finished in 0.01s\n"
    )
}

#[test]
fn one_passed_test_is_accepted() {
    assert_eq!(
        check_single_pass(&summary("ok", "1 passed; 0 failed; 0 ignored")),
        Ok(())
    );
}

#[test]
fn zero_executed_tests_are_rejected() {
    let error = check_single_pass(&summary("ok", "0 passed; 0 failed; 0 ignored")).unwrap_err();
    assert!(error.contains("0 passed"), "{error}");
}

#[test]
fn an_ignored_test_is_rejected_even_though_the_result_is_ok() {
    let error = check_single_pass(&summary("ok", "0 passed; 0 failed; 1 ignored")).unwrap_err();
    assert!(error.contains("1 ignored"), "{error}");
}

#[test]
fn a_failed_test_is_rejected() {
    let error = check_single_pass(&summary("FAILED", "0 passed; 1 failed; 0 ignored")).unwrap_err();
    assert!(error.contains("1 failed"), "{error}");
    assert!(error.contains("not ok"), "{error}");
}

#[test]
fn more_than_one_passed_test_is_rejected() {
    assert!(check_single_pass(&summary("ok", "2 passed; 0 failed; 0 ignored")).is_err());
    assert!(check_single_pass(&summary("ok", "1 passed; 0 failed; 1 ignored")).is_err());
}

#[test]
fn a_missing_or_repeated_summary_is_rejected() {
    assert!(check_single_pass("").is_err());
    assert!(check_single_pass("running 1 test\ntest case ... ok\n").is_err());
    let twice = summary("ok", "1 passed; 0 failed; 0 ignored").repeat(2);
    assert!(check_single_pass(&twice).is_err());
}

#[test]
fn cargo_noise_and_test_output_around_the_summary_are_ignored() {
    let output = format!(
        "   Compiling gust-compute-wgpu v0.1.0\n     Running tests\\semantics.rs\nAdapter: NVIDIA\n{}",
        summary("ok", "1 passed; 0 failed; 0 ignored")
    );
    assert_eq!(check_single_pass(&output), Ok(()));
}

#[test]
fn listing_requires_the_exact_name_on_its_own_line() {
    let listing = "alpha: test\nbeta::alpha: test\nalpha_two: test\nalpha: benchmark\n\n3 tests, 0 benchmarks\n";
    assert_eq!(count_listed(listing, "alpha"), 1);
    assert_eq!(count_listed(listing, "beta::alpha"), 1);
    assert_eq!(count_listed(listing, "alph"), 0);
    assert_eq!(count_listed("alpha: test\r\nalpha: test\r\n", "alpha"), 2);
    assert_eq!(count_listed("", "alpha"), 0);
}

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

#[test]
fn options_need_package_target_and_exactly_one_name() {
    let options = TestExactOptions::parse(&args(&[
        "-p",
        "gust-compute-wgpu",
        "--test",
        "semantics",
        "case",
    ]))
    .unwrap();
    assert_eq!(
        options.base_args(),
        args(&[
            "test",
            "--locked",
            "-p",
            "gust-compute-wgpu",
            "--test",
            "semantics"
        ])
    );
    assert_eq!(options.name, "case");

    let featured = TestExactOptions::parse(&args(&[
        "case",
        "--package",
        "p",
        "--test",
        "t",
        "--features",
        "a,b",
    ]))
    .unwrap();
    assert_eq!(featured.base_args()[6..], args(&["--features", "a,b"])[..]);

    assert!(TestExactOptions::parse(&args(&["--test", "t", "case"])).is_err());
    assert!(TestExactOptions::parse(&args(&["-p", "p", "case"])).is_err());
    assert!(TestExactOptions::parse(&args(&["-p", "p", "--test", "t"])).is_err());
    assert!(TestExactOptions::parse(&args(&["-p", "p", "--test", "t", "a", "b"])).is_err());
    assert!(TestExactOptions::parse(&args(&["-p", "p", "--test"])).is_err());
    assert!(TestExactOptions::parse(&args(&["-p", "p", "--test", "t", "--all", "a"])).is_err());
}
