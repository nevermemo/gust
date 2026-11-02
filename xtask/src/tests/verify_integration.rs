use super::{Integration, SKIP_WEB_FLAG, VerifyCommand, VerifyMode, integration_checks};

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
fn the_full_gate_runs_check_precompiled_and_check_web_and_no_other_mode_does() {
    for mode in ALL_MODES {
        for skip_web in [false, true] {
            let steps = integration_checks(mode, skip_web);
            if mode == VerifyMode::Full && !skip_web {
                assert_eq!(
                    steps,
                    [
                        Integration::Run("check-precompiled"),
                        Integration::Run("check-web")
                    ]
                );
            } else if mode == VerifyMode::Full {
                assert_eq!(
                    steps,
                    [
                        Integration::Run("check-precompiled"),
                        Integration::Skip("check-web")
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
    // The precompiled check is never skippable.
    assert!(with.contains(&Integration::Run("check-precompiled")));
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
    let source = include_str!("../verify.rs");
    let body = &source[source.find("fn verify_inner(").expect("verify_inner")..];
    let body = &body[..body.find("\n}\n").expect("end of verify_inner")];
    for needle in ["check_precompiled(&[])", "check_web_report(&[])"] {
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
