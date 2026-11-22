use std::path::Path;

use super::{COMBOS, Combo, Outcome, cargo_args, cargo_program, command, run_combos, summary};

fn names() -> Vec<&'static str> {
    COMBOS.iter().map(|combo| combo.name).collect()
}

/// The feature sets are pinned by name and by arguments: dropping or weakening one fails here, so the removal is a
/// decision made in the open.
#[test]
fn the_runs_are_pinned_by_name_and_arguments() {
    let runs: Vec<(&str, bool, String)> = COMBOS
        .iter()
        .map(|combo| (combo.name, combo.wasm, combo.args.join(" ")))
        .collect();
    let expected: Vec<(&str, bool, &str)> = vec![
        ("gust-compute: no features", false, "-p gust-compute --lib"),
        (
            "gust-compute: spirv-route",
            false,
            "-p gust-compute --lib --features spirv-route",
        ),
        ("gust-algo: no features", false, "-p gust-algo --lib"),
        (
            "gust-compute-wgpu: no features",
            false,
            "-p gust-compute-wgpu --lib",
        ),
        (
            "gust-compute-wgpu: slang-runtime",
            false,
            "-p gust-compute-wgpu --all-targets --features slang-runtime",
        ),
        (
            "gust-compute-wgpu: slang-runtime + spirv-route",
            false,
            "-p gust-compute-wgpu --all-targets --features slang-runtime,spirv-route",
        ),
        (
            "gust-compute-wgpu: spirv-route alone",
            false,
            "-p gust-compute-wgpu --lib --features spirv-route",
        ),
        (
            "wasm32: gust-compute, gust-compute-wgpu, web-vector-add",
            true,
            "--target wasm32-unknown-unknown -p gust-compute -p gust-compute-wgpu -p web-vector-add",
        ),
        (
            "wasm32: gust-compute-wgpu with spirv-route",
            true,
            "--target wasm32-unknown-unknown -p gust-compute-wgpu --features spirv-route",
        ),
    ];
    let expected: Vec<(&str, bool, String)> = expected
        .into_iter()
        .map(|(name, wasm, args)| (name, wasm, args.to_owned()))
        .collect();
    assert_eq!(runs, expected);
}

#[test]
fn every_run_is_clippy_with_warnings_denied_and_a_locked_lockfile() {
    for combo in COMBOS {
        let args = cargo_args(combo);
        assert_eq!(&args[..2], ["clippy", "--locked"], "{}", combo.name);
        assert_eq!(
            &args[args.len() - 3..],
            ["--", "-D", "warnings"],
            "{}",
            combo.name
        );
        assert_eq!(args.len(), combo.args.len() + 5, "{}", combo.name);
    }
}

#[test]
fn a_missing_wasm_target_is_skipped_and_not_counted_as_a_pass() {
    let mut ran = Vec::new();
    let outcomes = run_combos(COMBOS, false, false, &mut |combo: &Combo| {
        ran.push(combo.name);
        Ok(())
    })
    .unwrap();
    let native: Vec<&str> = COMBOS
        .iter()
        .filter(|combo| !combo.wasm)
        .map(|combo| combo.name)
        .collect();
    assert_eq!(ran, native, "no wasm32 run may start without the target");
    assert_eq!(outcomes.len(), COMBOS.len());
    for (name, outcome) in &outcomes {
        let wasm = COMBOS.iter().find(|c| c.name == *name).unwrap().wasm;
        assert_eq!(
            *outcome,
            if wasm {
                Outcome::Skipped
            } else {
                Outcome::Passed
            }
        );
    }
    let line = summary(&outcomes);
    assert!(
        line.starts_with("check-cfg: 7 builds linted, 2 SKIPPED (not passed): "),
        "{line}"
    );
    assert!(
        line.contains("wasm32: gust-compute-wgpu with spirv-route"),
        "{line}"
    );
    assert!(
        !line.contains("passed:"),
        "a run with skips must not say it passed: {line}"
    );
}

#[test]
fn a_run_with_nothing_skipped_says_so() {
    let outcomes = run_combos(COMBOS, true, false, &mut |_: &Combo| Ok(())).unwrap();
    assert!(
        outcomes
            .iter()
            .all(|(_, outcome)| *outcome == Outcome::Passed)
    );
    assert_eq!(summary(&outcomes), "check-cfg passed: 9 of 9 builds linted");
}

#[test]
fn require_wasm_turns_the_skip_into_an_error_before_any_wasm_run() {
    let mut ran = Vec::new();
    let error = run_combos(COMBOS, false, true, &mut |combo: &Combo| {
        ran.push(combo.name);
        Ok(())
    })
    .unwrap_err();
    assert!(error.contains("--require-wasm"), "{error}");
    assert!(error.contains("wasm32-unknown-unknown"), "{error}");
    assert!(ran.is_empty(), "nothing may run before the error: {ran:?}");
}

fn profile_of(combo: &Combo) -> Option<String> {
    let command = command(Path::new("."), combo);
    command
        .get_envs()
        .find(|(name, _)| *name == "GUST_PROFILE")
        .and_then(|(_, value)| value.map(|value| value.to_string_lossy().into_owned()))
}

/// The browser build is linted under the strict `web` profile and the native builds under the workspace profile; a
/// run that lost the override would lint wasm32 under `dev`.
#[test]
fn wasm_runs_set_the_web_profile_and_native_runs_the_dev_profile() {
    for combo in COMBOS {
        let expected = if combo.wasm { "web" } else { "dev" };
        assert_eq!(
            profile_of(combo).as_deref(),
            Some(expected),
            "{}",
            combo.name
        );
    }
}

#[test]
fn every_run_uses_the_same_cargo_program() {
    for combo in COMBOS {
        assert_eq!(
            command(Path::new("."), combo).get_program(),
            cargo_program(),
            "{}",
            combo.name
        );
    }
}

#[test]
fn the_first_failure_stops_the_run_and_names_the_combination() {
    let mut ran = Vec::new();
    let error = run_combos(COMBOS, true, false, &mut |combo: &Combo| {
        ran.push(combo.name);
        if combo.name == "gust-compute-wgpu: slang-runtime" {
            Err("cargo clippy failed (exit 101)".to_owned())
        } else {
            Ok(())
        }
    })
    .unwrap_err();
    assert_eq!(
        error,
        "check-cfg: gust-compute-wgpu: slang-runtime failed: cargo clippy failed (exit 101)"
    );
    assert_eq!(ran, names()[..5], "nothing runs after the failure");
}
