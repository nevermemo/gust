//! E6k to E6o: the profile grammar `<target>[+<capability>]*` as `app` is built under it.

use std::fs;

use super::{PRECOMPILED_DIR, fixture::Fixture};
use crate::{
    process::{PROFILE_VARIABLE, run_capture_in},
    ui::{Expectation, judge, parse_expectation},
};

// ---------------------------------------------------------------------------------------------------------------
// E6k to E6o (m8.a, D138): the profile grammar `<target>[+<capability>]*` as `app` is built under it

/// The expectation of the profile cases that `gust`'s build script refuses, relative to [`PRECOMPILED_DIR`]: lines
/// `<case id> build-error: <text>`, one per refused case.
const E6_PROFILE_EXPECTED: &str = "e6-profile-expected.txt";

/// What an E6 profile case must do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum E6ProfileOutcome {
    /// `cargo check -p app` compiles under the value.
    Compiles,
    /// The build script of `gust` refuses the value with the text of [`E6_PROFILE_EXPECTED`].
    Refused,
}

/// One `cargo check -p app` of E6 under a `GUST_PROFILE` value that carries a capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct E6ProfileCase {
    pub(crate) id: &'static str,
    /// The whole value of `GUST_PROFILE`.
    pub(crate) value: &'static str,
    pub(crate) outcome: E6ProfileOutcome,
}

/// The profile cases of E6, in order. They are not part of [`e6_cases`], which names a profile by its target alone.
pub(crate) fn e6_profile_cases() -> [E6ProfileCase; 5] {
    [
        E6ProfileCase {
            id: "E6k",
            value: "native+subgroups",
            outcome: E6ProfileOutcome::Compiles,
        },
        E6ProfileCase {
            id: "E6l",
            value: "dev+subgroups",
            outcome: E6ProfileOutcome::Compiles,
        },
        E6ProfileCase {
            id: "E6m",
            value: "web+subgroups",
            outcome: E6ProfileOutcome::Refused,
        },
        E6ProfileCase {
            id: "E6n",
            value: "cpu+subgroups",
            outcome: E6ProfileOutcome::Refused,
        },
        E6ProfileCase {
            id: "E6o",
            value: "native+f16",
            outcome: E6ProfileOutcome::Refused,
        },
    ]
}

/// The arguments of the `cargo check` of one E6 profile case (the fixture itself, `--locked`).
pub(crate) fn e6_profile_args(manifest: &str, target_dir: &str) -> Vec<String> {
    [
        "check",
        "--locked",
        "--color",
        "never",
        "--message-format",
        "json",
        "--manifest-path",
        manifest,
        "-p",
        "app",
        "--target-dir",
        target_dir,
    ]
    .iter()
    .map(|arg| (*arg).to_owned())
    .collect()
}

/// The `build-error:` text that `expectation` (the text of [`E6_PROFILE_EXPECTED`]) holds for the case `id`: the
/// lines that start with the id and a space, with that prefix removed. A case without a line is an error, so a new
/// refusal cannot pass on an empty expectation.
pub(crate) fn e6_profile_expectation(expectation: &str, id: &str) -> Result<String, String> {
    let prefix = format!("{id} ");
    let lines: Vec<&str> = expectation
        .trim_start_matches('\u{feff}')
        .lines()
        .map(str::trim)
        .filter_map(|line| line.strip_prefix(prefix.as_str()))
        .collect();
    if lines.is_empty() {
        return Err(format!(
            "{E6_PROFILE_EXPECTED} has no line for {id} (`{id} build-error: <text>`)"
        ));
    }
    Ok(lines.join("\n"))
}

/// The problems of one E6 profile case. A refusal is judged on the exact message of the build script by [`judge`],
/// never on the exit status alone.
pub(crate) fn e6_profile_problems(
    case: &E6ProfileCase,
    expectation: Option<&str>,
    success: bool,
    stdout: &str,
    stderr: &str,
) -> Result<Vec<String>, String> {
    match case.outcome {
        E6ProfileOutcome::Compiles => Ok(judge(
            case.id,
            "app",
            &Expectation::Compiles,
            success,
            stdout,
            stderr,
        )),
        E6ProfileOutcome::Refused => {
            let source = expectation.ok_or_else(|| format!("{}: no expectation text", case.id))?;
            let text = e6_profile_expectation(source, case.id)?;
            let wanted = parse_expectation(&text)
                .map_err(|error| format!("{E6_PROFILE_EXPECTED} ({}): {error}", case.id))?;
            if !matches!(wanted, Expectation::BuildFails(_)) {
                return Err(format!(
                    "{E6_PROFILE_EXPECTED} ({}) must hold `build-error:` lines (the refusal is a build-script failure), not rustc keys",
                    case.id
                ));
            }
            Ok(judge(case.id, "app", &wanted, success, stdout, stderr))
        }
    }
}

/// E6k to E6o: `app` under `native+subgroups` and `dev+subgroups` compiles, and the build script refuses
/// `web+subgroups`, `cpu+subgroups` and `native+f16` with its own text. Pushes the problems onto `problems` and
/// returns the summary.
pub(super) fn e6_profile_values(
    fixture: &Fixture,
    problems: &mut Vec<String>,
) -> Result<String, String> {
    let scratch_target = fixture.scratch_target_dir();
    let target_dir = Fixture::target_dir_arg(&scratch_target)?;
    let manifest = format!("{PRECOMPILED_DIR}/Cargo.toml");
    let path = fixture.root.join(PRECOMPILED_DIR).join(E6_PROFILE_EXPECTED);
    let expectation = fs::read_to_string(&path).map_err(|error| {
        format!("check-precompiled E6: {PRECOMPILED_DIR}/{E6_PROFILE_EXPECTED}: {error}")
    })?;
    let mut lines = Vec::new();
    for case in e6_profile_cases() {
        let args = e6_profile_args(&manifest, target_dir);
        let arguments: Vec<&str> = args.iter().map(String::as_str).collect();
        let output = run_capture_in(
            &mut fixture.cargo_under_profile(&arguments, case.value),
            "check-precompiled E6",
            "cargo",
        )?;
        let found = e6_profile_problems(
            &case,
            Some(&expectation),
            output.status.success(),
            &String::from_utf8_lossy(&output.stdout),
            &String::from_utf8_lossy(&output.stderr),
        )
        .map_err(|error| format!("check-precompiled E6: {error}"))?;
        let shown = match case.outcome {
            E6ProfileOutcome::Compiles => "compiled",
            E6ProfileOutcome::Refused => "refused",
        };
        println!(
            "check-precompiled E6 {}: {} ({PROFILE_VARIABLE}={}): {}",
            case.id,
            args.join(" ").replace(target_dir, "<scratch-target>"),
            case.value,
            if found.is_empty() {
                shown
            } else {
                "NOT AS EXPECTED"
            }
        );
        lines.push(format!("{} {shown}", case.id));
        problems.extend(found);
    }
    Ok(format!(
        "{} ({PRECOMPILED_DIR}/{E6_PROFILE_EXPECTED})",
        lines.join(", ")
    ))
}

/// The tests of E6k to E6o. They sit here and not in `tests/precompiled_refusals.rs`, which pins the cases of
/// `e6_cases`.
#[cfg(test)]
mod e6_profile_tests {
    use super::{super::refusals::e6_cases, *};

    const EXPECTED: &str = include_str!("../../../tests/precompiled/e6-profile-expected.txt");

    fn case(id: &str) -> E6ProfileCase {
        e6_profile_cases()
            .into_iter()
            .find(|case| case.id == id)
            .unwrap()
    }

    fn refusal(id: &str) -> String {
        e6_profile_expectation(EXPECTED, id)
            .unwrap()
            .strip_prefix("build-error: ")
            .unwrap()
            .to_owned()
    }

    #[test]
    fn the_profile_cases_are_the_five_of_the_plan() {
        let cases: Vec<(&str, &str, E6ProfileOutcome)> = e6_profile_cases()
            .iter()
            .map(|case| (case.id, case.value, case.outcome))
            .collect();
        assert_eq!(
            cases,
            [
                ("E6k", "native+subgroups", E6ProfileOutcome::Compiles),
                ("E6l", "dev+subgroups", E6ProfileOutcome::Compiles),
                ("E6m", "web+subgroups", E6ProfileOutcome::Refused),
                ("E6n", "cpu+subgroups", E6ProfileOutcome::Refused),
                ("E6o", "native+f16", E6ProfileOutcome::Refused),
            ]
        );
    }

    #[test]
    fn the_ids_do_not_collide_with_the_cases_of_the_target_alone() {
        let plain: Vec<&str> = e6_cases().iter().map(|case| case.id).collect();
        for case in e6_profile_cases() {
            assert!(!plain.contains(&case.id), "{}", case.id);
        }
    }

    #[test]
    fn every_refusal_has_one_line_and_every_compile_has_none() {
        for case in e6_profile_cases() {
            let found = e6_profile_expectation(EXPECTED, case.id);
            match case.outcome {
                E6ProfileOutcome::Refused => {
                    let text = found.unwrap();
                    assert_eq!(text.lines().count(), 1, "{}", case.id);
                    assert!(text.starts_with("build-error: gust: "), "{}", case.id);
                    // The refusal names the whole value it refused, or the capability in it.
                    let name = case.value.split('+').next_back().unwrap();
                    assert!(
                        text.contains(case.value) || text.contains(name),
                        "{}",
                        case.id
                    );
                }
                E6ProfileOutcome::Compiles => assert!(found.is_err(), "{}", case.id),
            }
        }
    }

    #[test]
    fn a_case_without_a_line_is_an_error_and_a_prefix_of_an_id_is_not_the_id() {
        assert!(e6_profile_expectation("E6m build-error: x\n", "E6n").is_err());
        assert!(e6_profile_expectation("E6mm build-error: x\n", "E6m").is_err());
        assert_eq!(
            e6_profile_expectation("# note\r\nE6m build-error: x\r\n", "E6m").unwrap(),
            "build-error: x"
        );
    }

    #[test]
    fn the_arguments_check_the_locked_fixture_app_and_nothing_else() {
        let args = e6_profile_args("tests/precompiled/Cargo.toml", "T");
        assert_eq!(args[0], "check");
        assert!(args.contains(&"--locked".to_owned()));
        let position = args.iter().position(|arg| arg == "-p").unwrap();
        assert_eq!(args[position + 1], "app");
        assert!(!args.contains(&"--features".to_owned()));
        assert!(!args.contains(&"--release".to_owned()));
        assert_eq!(args.last().unwrap(), "T");
    }

    #[test]
    fn a_refusal_is_judged_on_its_message() {
        let web = case("E6m");
        let text = refusal("E6m");
        let stderr = format!(
            "error: failed to run custom build command for `gust-compute v0.1.0`\n{text}\n"
        );
        let ok = e6_profile_problems(&web, Some(EXPECTED), false, "", &stderr).unwrap();
        assert!(ok.is_empty(), "{ok:?}");
        // A build that compiled is a problem, and so is another failure.
        assert!(
            !e6_profile_problems(&web, Some(EXPECTED), true, "", &stderr)
                .unwrap()
                .is_empty()
        );
        let other = "error: failed to run custom build command\nsomething else\n";
        assert!(
            !e6_profile_problems(&web, Some(EXPECTED), false, "", other)
                .unwrap()
                .is_empty()
        );
        // Another case's message is not this case's.
        let cpu = format!("{}\n", refusal("E6n"));
        assert!(
            !e6_profile_problems(&web, Some(EXPECTED), false, "", &cpu)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn a_refusal_without_an_expectation_is_an_error_not_a_pass() {
        let web = case("E6m");
        assert!(e6_profile_problems(&web, None, false, "", "").is_err());
        assert!(e6_profile_problems(&web, Some("# nothing\n"), false, "", "").is_err());
    }

    #[test]
    fn a_compile_case_fails_when_cargo_fails() {
        let native = case("E6k");
        assert!(
            e6_profile_problems(&native, None, true, "", "")
                .unwrap()
                .is_empty()
        );
        let failed = e6_profile_problems(&native, None, false, "", "error: boom\n").unwrap();
        assert_eq!(failed.len(), 1, "{failed:?}");
    }
}
