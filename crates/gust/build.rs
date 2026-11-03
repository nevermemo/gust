//! Build script of `gust`: the effective build profile (D44, plan M4 R2-2).
//!
//! `GUST_PROFILE` (`dev`, `native`, `web` or `cpu`) selects the profile. Without it a release build is strict (`web` on
//! `wasm32`, `native` elsewhere) and any other build is `dev`; `cpu` is never inferred. The result reaches `gust` as
//! `cfg(gust_profile = "..")` (see `gust::profile`), and every profile but `dev` refuses the `slang-runtime` feature,
//! so a production build can never contain the run-time Slang bridge, not even through feature unification.
//!
//! The one exception is the internal feature `cargo-gust-bridge` (D108), which only `cargo-gust` enables so that a
//! plain `cargo install cargo-gust` works: with it on, a release build without `GUST_PROFILE` is `dev` (and the script
//! says so with a `cargo:warning`) instead of `native` or `web`. An explicit `GUST_PROFILE` always wins over it.
//!
//! The decision is a pure function of [`Inputs`] that returns the exact lines this script prints, so the unit tests
//! (`cargo test -p gust`, which includes this file from `src/profile.rs`) pin them.

use std::{env, process::ExitCode};

/// The values `GUST_PROFILE` may take.
const PROFILES: [&str; 4] = ["dev", "native", "web", "cpu"];

/// The line printed whenever `cargo-gust-bridge` changes the profile (a release build without `GUST_PROFILE`).
const BRIDGE_WARNING: &str =
    "cargo:warning=gust: cargo-gust-bridge treats a release build without GUST_PROFILE as dev";

/// What the decision reads from the environment of the build script.
struct Inputs<'a> {
    /// `GUST_PROFILE`; `None` when it is not set (an empty value is `Some("")`).
    profile: Option<&'a str>,
    /// Cargo's `PROFILE` is `release`.
    release: bool,
    /// `CARGO_CFG_TARGET_ARCH` is `wasm32`.
    wasm: bool,
    /// `CARGO_FEATURE_SLANG_RUNTIME` is set.
    slang_runtime: bool,
    /// `CARGO_FEATURE_CARGO_GUST_BRIDGE` is set (the internal feature `cargo-gust-bridge`, D108).
    bridge: bool,
}

/// The effective profile, and whether it was inferred from a release build (no explicit `GUST_PROFILE`). With
/// `cargo-gust-bridge` the inference for a release build is `dev`, whatever the target.
fn effective(inputs: &Inputs<'_>) -> Result<(&'static str, bool), String> {
    match inputs.profile {
        Some("") => Err(format!(
            "gust: GUST_PROFILE is set but empty; use {} (or unset it)",
            allowed()
        )),
        Some(value) => PROFILES
            .iter()
            .find(|known| **known == value)
            .map(|known| (*known, false))
            .ok_or_else(|| format!("gust: unknown GUST_PROFILE `{value}`; use {}", allowed())),
        None if inputs.release && inputs.bridge => Ok(("dev", true)),
        None if inputs.release && inputs.wasm => Ok(("web", true)),
        None if inputs.release => Ok(("native", true)),
        None => Ok(("dev", false)),
    }
}

/// `dev, native, web or cpu`.
fn allowed() -> String {
    format!(
        "{}, {}, {} or {}",
        PROFILES[0], PROFILES[1], PROFILES[2], PROFILES[3]
    )
}

/// The cargo directive that declares the values of `cfg(gust_profile)`.
fn check_cfg() -> String {
    let values: Vec<String> = PROFILES.iter().map(|name| format!("\"{name}\"")).collect();
    format!(
        "cargo:rustc-check-cfg=cfg(gust_profile, values({}))",
        values.join(", ")
    )
}

/// The lines (without the trailing newline) this script prints for cargo, or the reason the build must fail.
fn directives(inputs: &Inputs<'_>) -> Result<Vec<String>, String> {
    let (profile, inferred) = effective(inputs)?;
    if inputs.slang_runtime && profile == "cpu" {
        // `cpu` is never inferred, so there is no origin to name.
        return Err("gust: profile cpu refuses feature slang-runtime; a CPU build contains no compiler, bridge or GPU runtime. Do not enable the feature for this build (look for features = [\"slang-runtime\"] in a dependency, or --features ...slang-runtime; `cargo tree -e features -i gust` shows who enables it), or build with GUST_PROFILE=dev.".to_owned());
    }
    if inputs.slang_runtime && profile != "dev" {
        let origin = if inferred {
            " This profile was inferred from a release build without GUST_PROFILE."
        } else {
            ""
        };
        return Err(format!(
            "gust: profile {profile} refuses feature slang-runtime; a strict profile builds no run-time Slang \
             bridge. Do not enable the feature for this build (look for features = [\"slang-runtime\"] in a \
             dependency, or --features ...slang-runtime; `cargo tree -e features -i gust` shows who enables it), \
             or build with GUST_PROFILE=dev.{origin}"
        ));
    }
    let mut lines = vec![
        "cargo:rerun-if-changed=build.rs".to_owned(),
        "cargo:rerun-if-env-changed=GUST_PROFILE".to_owned(),
        check_cfg(),
    ];
    if inferred && inputs.bridge {
        lines.push(BRIDGE_WARNING.to_owned());
    } else if inferred {
        lines.push(format!(
            "cargo:warning=gust: release build without GUST_PROFILE treated as {profile}"
        ));
    }
    lines.push(format!("cargo:rustc-cfg=gust_profile=\"{profile}\""));
    Ok(lines)
}

fn main() -> ExitCode {
    let value = env::var_os("GUST_PROFILE").map(|value| value.to_string_lossy().into_owned());
    let inputs = Inputs {
        profile: value.as_deref(),
        release: env::var("PROFILE").is_ok_and(|profile| profile == "release"),
        wasm: env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|arch| arch == "wasm32"),
        slang_runtime: env::var_os("CARGO_FEATURE_SLANG_RUNTIME").is_some(),
        bridge: env::var_os("CARGO_FEATURE_CARGO_GUST_BRIDGE").is_some(),
    };
    match directives(&inputs) {
        Ok(lines) => {
            for line in lines {
                println!("{line}");
            }
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Inputs, directives, effective};

    const NONE: Inputs<'static> = Inputs {
        profile: None,
        release: false,
        wasm: false,
        slang_runtime: false,
        bridge: false,
    };

    fn cfg_line(profile: &str) -> String {
        format!("cargo:rustc-cfg=gust_profile=\"{profile}\"")
    }

    fn warning(profile: &str) -> String {
        format!("cargo:warning=gust: release build without GUST_PROFILE treated as {profile}")
    }

    #[test]
    fn without_a_variable_a_debug_build_is_dev_and_says_nothing() {
        let lines = directives(&NONE).unwrap();
        assert!(lines.contains(&cfg_line("dev")), "{lines:?}");
        assert!(
            !lines.iter().any(|line| line.contains("warning")),
            "{lines:?}"
        );
        assert_eq!(effective(&NONE), Ok(("dev", false)));
    }

    #[test]
    fn a_release_build_without_a_variable_is_strict_and_warns_once() {
        let native = Inputs {
            release: true,
            ..NONE
        };
        let lines = directives(&native).unwrap();
        assert!(lines.contains(&cfg_line("native")), "{lines:?}");
        assert!(!lines.contains(&cfg_line("dev")), "{lines:?}");
        assert_eq!(
            lines
                .iter()
                .filter(|line| **line == warning("native"))
                .count(),
            1,
            "{lines:?}"
        );
        let web = Inputs {
            release: true,
            wasm: true,
            ..NONE
        };
        let lines = directives(&web).unwrap();
        assert!(lines.contains(&cfg_line("web")), "{lines:?}");
        assert!(lines.contains(&warning("web")), "{lines:?}");
        // wasm32 alone, without a release build, stays dev.
        let debug_wasm = Inputs { wasm: true, ..NONE };
        assert_eq!(effective(&debug_wasm), Ok(("dev", false)));
    }

    #[test]
    fn an_explicit_value_wins_over_the_release_inference_and_is_never_warned_about() {
        for release in [false, true] {
            for wasm in [false, true] {
                for name in ["dev", "native", "web", "cpu"] {
                    let inputs = Inputs {
                        profile: Some(name),
                        release,
                        wasm,
                        slang_runtime: false,
                        bridge: false,
                    };
                    let lines = directives(&inputs).unwrap();
                    assert!(lines.contains(&cfg_line(name)), "{name}: {lines:?}");
                    assert!(
                        !lines.iter().any(|line| line.contains("warning")),
                        "{name}: {lines:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_value_other_than_the_four_names_is_refused_with_the_names() {
        for (value, message) in [
            (
                "bogus",
                "gust: unknown GUST_PROFILE `bogus`; use dev, native, web or cpu",
            ),
            (
                "Native",
                "gust: unknown GUST_PROFILE `Native`; use dev, native, web or cpu",
            ),
            (
                " native",
                "gust: unknown GUST_PROFILE ` native`; use dev, native, web or cpu",
            ),
            (
                "native+gpu",
                "gust: unknown GUST_PROFILE `native+gpu`; use dev, native, web or cpu",
            ),
            (
                "CPU",
                "gust: unknown GUST_PROFILE `CPU`; use dev, native, web or cpu",
            ),
            (
                "cpu+simd",
                "gust: unknown GUST_PROFILE `cpu+simd`; use dev, native, web or cpu",
            ),
            (
                "",
                "gust: GUST_PROFILE is set but empty; use dev, native, web or cpu (or unset it)",
            ),
        ] {
            let inputs = Inputs {
                profile: Some(value),
                // An invalid value is an error whatever else is true of the build.
                release: true,
                ..NONE
            };
            assert_eq!(directives(&inputs), Err(message.to_owned()), "{value:?}");
        }
    }

    #[test]
    fn cargo_is_told_to_rerun_on_a_changed_variable_and_the_cfg_is_declared() {
        for inputs in [
            NONE,
            Inputs {
                release: true,
                ..NONE
            },
            Inputs {
                profile: Some("web"),
                ..NONE
            },
            Inputs {
                profile: Some("cpu"),
                ..NONE
            },
        ] {
            let lines = directives(&inputs).unwrap();
            assert!(
                lines.contains(&"cargo:rerun-if-env-changed=GUST_PROFILE".to_owned()),
                "{lines:?}"
            );
            assert!(
                lines.contains(
                    &"cargo:rustc-check-cfg=cfg(gust_profile, values(\"dev\", \"native\", \"web\", \"cpu\"))"
                        .to_owned()
                ),
                "{lines:?}"
            );
            assert_eq!(
                lines
                    .iter()
                    .filter(|line| line.starts_with("cargo:rustc-cfg="))
                    .count(),
                1,
                "{lines:?}"
            );
        }
    }

    #[test]
    fn a_strict_profile_refuses_the_slang_runtime_feature() {
        for name in ["native", "web"] {
            let inputs = Inputs {
                profile: Some(name),
                slang_runtime: true,
                ..NONE
            };
            let message = directives(&inputs).unwrap_err();
            let lead = format!("gust: profile {name} refuses feature slang-runtime; ");
            assert!(message.starts_with(&lead), "{message}");
            assert!(
                message.contains("or build with GUST_PROFILE=dev."),
                "{message}"
            );
            assert!(
                message.contains("cargo tree -e features -i gust"),
                "{message}"
            );
            assert!(!message.contains("inferred"), "{message}");
        }
    }

    #[test]
    fn the_explicit_native_refusal_is_pinned_word_for_word() {
        let inputs = Inputs {
            profile: Some("native"),
            slang_runtime: true,
            ..NONE
        };
        assert_eq!(
            directives(&inputs),
            Err("gust: profile native refuses feature slang-runtime; a strict profile builds no run-time Slang bridge. Do not enable the feature for this build (look for features = [\"slang-runtime\"] in a dependency, or --features ...slang-runtime; `cargo tree -e features -i gust` shows who enables it), or build with GUST_PROFILE=dev.".to_owned())
        );
    }

    #[test]
    fn the_inferred_release_profile_refuses_it_too_and_says_where_it_came_from() {
        let inputs = Inputs {
            release: true,
            slang_runtime: true,
            ..NONE
        };
        let message = directives(&inputs).unwrap_err();
        assert!(
            message.starts_with("gust: profile native refuses feature slang-runtime; "),
            "{message}"
        );
        assert!(
            message.ends_with(
                "or build with GUST_PROFILE=dev. This profile was inferred from a release build without GUST_PROFILE."
            ),
            "{message}"
        );
        let wasm = Inputs {
            wasm: true,
            ..inputs
        };
        assert!(
            directives(&wasm)
                .unwrap_err()
                .starts_with("gust: profile web refuses feature slang-runtime; ")
        );
    }

    #[test]
    fn the_cpu_profile_is_never_inferred_and_never_warned_about() {
        // A release build without the variable is native or web, whatever else is true of it.
        for wasm in [false, true] {
            let inferred = Inputs {
                release: true,
                wasm,
                ..NONE
            };
            let (profile, _) = effective(&inferred).unwrap();
            assert_ne!(profile, "cpu");
        }
        // Explicit, it is exactly one cfg and no warning, in a debug and a release build.
        for release in [false, true] {
            let explicit = Inputs {
                profile: Some("cpu"),
                release,
                ..NONE
            };
            assert_eq!(effective(&explicit), Ok(("cpu", false)));
            let lines = directives(&explicit).unwrap();
            assert!(lines.contains(&cfg_line("cpu")), "{lines:?}");
            assert!(
                !lines.iter().any(|line| line.contains("warning")),
                "{lines:?}"
            );
        }
    }

    #[test]
    fn the_cpu_profile_refuses_the_slang_runtime_feature_word_for_word() {
        for release in [false, true] {
            let inputs = Inputs {
                profile: Some("cpu"),
                release,
                slang_runtime: true,
                ..NONE
            };
            assert_eq!(
                directives(&inputs),
                Err("gust: profile cpu refuses feature slang-runtime; a CPU build contains no compiler, bridge or GPU runtime. Do not enable the feature for this build (look for features = [\"slang-runtime\"] in a dependency, or --features ...slang-runtime; `cargo tree -e features -i gust` shows who enables it), or build with GUST_PROFILE=dev.".to_owned())
            );
        }
    }

    #[test]
    fn the_dev_profile_accepts_the_slang_runtime_feature_in_every_build() {
        for release in [false, true] {
            let inputs = Inputs {
                profile: Some("dev"),
                release,
                wasm: false,
                slang_runtime: true,
                bridge: false,
            };
            let lines = directives(&inputs).unwrap();
            assert!(lines.contains(&cfg_line("dev")), "{lines:?}");
        }
        // Without a variable and not in a release build the profile is dev, so the feature is accepted.
        let inputs = Inputs {
            slang_runtime: true,
            ..NONE
        };
        assert!(directives(&inputs).is_ok());
    }

    const BRIDGE_ON: Inputs<'static> = Inputs {
        bridge: true,
        ..NONE
    };

    fn bridge_warning() -> String {
        "cargo:warning=gust: cargo-gust-bridge treats a release build without GUST_PROFILE as dev"
            .to_owned()
    }

    #[test]
    fn the_bridge_feature_makes_a_release_build_without_a_variable_dev_and_says_so_once() {
        for wasm in [false, true] {
            for slang_runtime in [false, true] {
                let inputs = Inputs {
                    release: true,
                    wasm,
                    slang_runtime,
                    ..BRIDGE_ON
                };
                assert_eq!(effective(&inputs), Ok(("dev", true)));
                let lines = directives(&inputs).unwrap();
                assert!(lines.contains(&cfg_line("dev")), "{lines:?}");
                assert_eq!(
                    lines
                        .iter()
                        .filter(|line| **line == bridge_warning())
                        .count(),
                    1,
                    "{lines:?}"
                );
                // It is the only warning: the plain inference text is not printed beside it.
                assert_eq!(
                    lines.iter().filter(|line| line.contains("warning")).count(),
                    1,
                    "{lines:?}"
                );
            }
        }
    }

    #[test]
    fn the_bridge_feature_changes_nothing_else() {
        // A debug build is dev already, so there is nothing to say.
        let debug = Inputs {
            slang_runtime: true,
            ..BRIDGE_ON
        };
        let lines = directives(&debug).unwrap();
        assert!(lines.contains(&cfg_line("dev")), "{lines:?}");
        assert!(
            !lines.iter().any(|line| line.contains("warning")),
            "{lines:?}"
        );
        // An explicit value wins, and is never warned about.
        for name in ["dev", "cpu"] {
            let inputs = Inputs {
                profile: Some(name),
                release: true,
                ..BRIDGE_ON
            };
            let lines = directives(&inputs).unwrap();
            assert!(lines.contains(&cfg_line(name)), "{name}: {lines:?}");
            assert!(
                !lines.iter().any(|line| line.contains("warning")),
                "{name}: {lines:?}"
            );
        }
        // Without the feature the same release build stays strict.
        let plain = Inputs {
            release: true,
            ..NONE
        };
        assert_eq!(effective(&plain), Ok(("native", true)));
    }

    #[test]
    fn an_explicit_strict_value_still_refuses_the_slang_runtime_feature_with_the_bridge_on() {
        for name in ["native", "web"] {
            for release in [false, true] {
                let inputs = Inputs {
                    profile: Some(name),
                    release,
                    slang_runtime: true,
                    ..BRIDGE_ON
                };
                let message = directives(&inputs).unwrap_err();
                assert!(
                    message.starts_with(&format!(
                        "gust: profile {name} refuses feature slang-runtime; "
                    )),
                    "{message}"
                );
                assert!(!message.contains("inferred"), "{message}");
            }
        }
        let cpu = Inputs {
            profile: Some("cpu"),
            release: true,
            slang_runtime: true,
            ..BRIDGE_ON
        };
        assert!(
            directives(&cpu)
                .unwrap_err()
                .starts_with("gust: profile cpu refuses feature slang-runtime; ")
        );
    }

    #[test]
    fn an_unknown_value_is_reported_before_the_feature_refusal() {
        let inputs = Inputs {
            profile: Some("bogus"),
            slang_runtime: true,
            ..NONE
        };
        assert_eq!(
            directives(&inputs),
            Err("gust: unknown GUST_PROFILE `bogus`; use dev, native, web or cpu".to_owned())
        );
    }
}
