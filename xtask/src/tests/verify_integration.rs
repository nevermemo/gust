use super::{
    Integration, SKIP_WEB_FLAG, VerifyCommand, VerifyMode, integration_checks, web_preflight,
};

const ALL_MODES: [VerifyMode; 6] = [
    VerifyMode::Smoke,
    VerifyMode::Fast,
    VerifyMode::Gpu,
    VerifyMode::Examples,
    VerifyMode::Artifacts,
    VerifyMode::Full,
];

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

#[test]
fn the_full_gate_runs_check_precompiled_check_web_and_check_difftest_and_no_other_mode_does() {
    for mode in ALL_MODES {
        for skip_web in [false, true] {
            let steps = integration_checks(mode, skip_web);
            if mode == VerifyMode::Full && !skip_web {
                assert_eq!(
                    steps,
                    [
                        Integration::Run("check-precompiled"),
                        Integration::Run("check-release"),
                        Integration::Run("check-web"),
                        Integration::Run("check-difftest")
                    ]
                );
            } else if mode == VerifyMode::Full {
                assert_eq!(
                    steps,
                    [
                        Integration::Run("check-precompiled"),
                        Integration::Run("check-release"),
                        Integration::Skip("check-web"),
                        Integration::Run("check-difftest")
                    ]
                );
            } else {
                assert!(steps.is_empty(), "{mode:?}");
            }
        }
    }
}

#[test]
fn check_web_is_skipped_only_when_the_flag_is_given() {
    let without = integration_checks(VerifyMode::Full, false);
    assert!(
        !without
            .iter()
            .any(|step| matches!(step, Integration::Skip(_))),
        "{without:?}"
    );
    let with = integration_checks(VerifyMode::Full, true);
    assert_eq!(
        with.iter()
            .filter(|step| matches!(step, Integration::Skip(_)))
            .count(),
        1
    );
    // The differential batch and the precompiled check are never skippable.
    assert!(with.contains(&Integration::Run("check-precompiled")));
    assert!(with.contains(&Integration::Run("check-release")));
    assert!(with.contains(&Integration::Run("check-difftest")));
}

#[test]
fn skip_web_is_a_full_mode_flag() {
    let full = VerifyCommand::parse(&args(&["--mode", "full", SKIP_WEB_FLAG])).unwrap();
    assert!(full.skip_web && full.mode == VerifyMode::Full && full.record);
    assert!(
        !VerifyCommand::parse(&args(&["--mode", "full"]))
            .unwrap()
            .skip_web
    );
    let error = VerifyCommand::parse(&args(&["--mode", "fast", SKIP_WEB_FLAG]))
        .err()
        .unwrap();
    assert!(error.contains("full mode only"), "{error}");
    assert!(VerifyCommand::parse(&args(&[SKIP_WEB_FLAG])).is_err());
}

/// A source pin: `verify_inner` needs cargo, Slang and a GPU, so a unit test cannot run it. The Full gate calls the
/// whole `check-precompiled` (no `--check-only`) and the whole `check-web` (not `--tree-only`), after its own steps.
#[test]
fn verify_inner_runs_the_whole_integration_checks_last() {
    // A checkout with `core.autocrlf=true` has CRLF line ends; the search below looks for LF.
    let source = include_str!("../verify.rs").replace("\r\n", "\n");
    let body = &source[source.find("fn verify_inner(").expect("verify_inner")..];
    let body = &body[..body.find("\n}\n").expect("end of verify_inner")];
    // `check_difftest_report(&[])` is the default call: all eight seeds on the adapter, and the short_circuit test.
    for needle in [
        "check_difftest_report(&[])",
        "check_precompiled_report(&[])",
        "check_release_report(&[])",
        "check_web_report(&[])",
    ] {
        assert!(
            body.contains(needle),
            "verify_inner no longer calls `{needle}`"
        );
    }
    assert!(!body.contains("--check-only") && !body.contains("--tree-only"));
    let artifacts = body.find("Validated {file}").expect("artifact scan");
    let integration = body
        .find("integration_checks(mode, skip_web)")
        .expect("integration loop");
    assert!(artifacts < integration);
}

/// The preflight is the first thing `verify_inner` does, and it uses check-web's own lookups.
#[test]
fn verify_inner_looks_for_the_web_tools_before_any_other_step() {
    let source = include_str!("../verify.rs").replace("\r\n", "\n");
    let body = &source[source.find("fn verify_inner(").expect("verify_inner")..];
    let preflight = body
        .find("web_preflight(mode, skip_web, preflight_tools)")
        .expect("verify_inner no longer runs the web preflight");
    for later in ["lint(root, checks)", "require_command(", "run_checked("] {
        assert!(
            preflight < body.find(later).expect(later),
            "the preflight comes after `{later}`"
        );
    }
}

const MISSING: &str = "check-web: wasm-bindgen is not on PATH";

#[test]
fn a_full_run_without_a_web_tool_stops_at_once_and_names_the_tool_and_the_flag() {
    let error = web_preflight(VerifyMode::Full, false, || Err(MISSING.to_owned())).unwrap_err();
    assert!(error.contains("wasm-bindgen"), "{error}");
    assert!(error.contains(SKIP_WEB_FLAG), "{error}");
    assert!(web_preflight(VerifyMode::Full, false, || Ok(())).is_ok());
}

#[test]
fn the_preflight_does_not_look_when_check_web_will_not_run() {
    let never = || -> Result<(), String> { panic!("looked for the web tools") };
    assert!(web_preflight(VerifyMode::Full, true, never).is_ok());
    for mode in ALL_MODES
        .into_iter()
        .filter(|mode| *mode != VerifyMode::Full)
    {
        assert!(web_preflight(mode, false, never).is_ok(), "{mode:?}");
    }
}
