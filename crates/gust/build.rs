//! Build script of `gust`: the effective build profile (D44, plan M4 R2-2).
//!
//! `GUST_PROFILE` (`dev`, `native`, `web` or `cpu`) selects the profile. Without it a release build is strict (`web` on
//! `wasm32`, `native` elsewhere) and any other build is `dev`; `cpu` is never inferred. The result reaches `gust` as
//! `cfg(gust_profile = "..")` (see `gust::profile`), and every profile but `dev` refuses the `slang-runtime` feature,
//! so a production build can never contain the run-time Slang bridge, not even through feature unification.
//!
//! The value may end in capabilities, `<target>(+<capability>)*` (M8, D138): `native+subgroups`. The one capability of
//! this release is `subgroups`; each becomes `cfg(gust_capability = "..")` (see `gust::capability::enabled`).
//! `f16` and `atomics64` parse and are refused as not supported, `cpu` takes no capability, and `web+subgroups` is
//! refused on the pinned wgpu. `dev+subgroups` is allowed: it is the profile of the tests of subgroup kernels, and
//! `dev` is the one profile that accepts the `slang-runtime` feature.
//!
//! The one exception is the internal feature `cargo-gust-bridge` (D108), which only `cargo-gust` enables so that a
//! plain `cargo install cargo-gust` works: with it on, a release build without `GUST_PROFILE` is `dev` (and the script
//! says so with a `cargo:warning`) instead of `native`; a `wasm32` build stays `web`. An explicit `GUST_PROFILE`
//! always wins over it.
//!
//! The decision is a pure function of [`Inputs`] that returns the exact lines this script prints, so the unit tests
//! (`cargo test -p gust-compute`, which includes this file from `src/profile.rs`) pin them.

use std::{env, process::ExitCode};

/// The targets `GUST_PROFILE` may start with.
const PROFILES: [&str; 4] = ["dev", "native", "web", "cpu"];

/// The capabilities a `GUST_PROFILE` may name after a `+` (M8, D138): `native+subgroups`. They are the names of
/// `gust::capability::Capability`; each one becomes `cfg(gust_capability = "..")`.
const CAPABILITIES: [&str; 1] = ["subgroups"];

/// Capabilities that are reserved for a later release: they parse, and are refused as not supported.
const RESERVED_CAPABILITIES: [&str; 2] = ["f16", "atomics64"];

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
/// `cargo-gust-bridge` the inference for a release build is `dev`, except on `wasm32`, which stays strict (`cargo
/// install` never targets it).
fn effective(inputs: &Inputs<'_>) -> Result<(&'static str, bool), String> {
    match inputs.profile {
        Some("") => Err(format!(
            "gust: GUST_PROFILE is set but empty; use {} (or unset it)",
            allowed()
        )),
        Some(value) => {
            // The target is the text before the first `+`; the capabilities after it are judged by `capabilities`.
            let target = value.split('+').next().unwrap_or_default();
            PROFILES
                .iter()
                .find(|known| **known == target)
                .map(|known| (*known, false))
                .ok_or_else(|| format!("gust: unknown GUST_PROFILE `{value}`; use {}", allowed()))
        }
        None if inputs.release && inputs.bridge && !inputs.wasm => Ok(("dev", true)),
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

/// The capabilities `GUST_PROFILE` names after the target (`native+subgroups`), in the order of [`CAPABILITIES`], or the
/// reason the value is refused. `profile` is the target [`effective`] found.
///
/// The grammar is `<target>(+<capability>)*`, exact and lower-case, with no trimming. It is mirrored by
/// `Profile::parse` in `gust-macros` (the macro cannot depend on this crate) and the `GRAMMAR` table of both test
/// modules is the same text. The refusals, in the order the pieces are read: an empty piece, an unknown capability, a
/// reserved one (`f16`, `atomics64`), a duplicate; then, for the target, `cpu` takes no capability and `web` cannot
/// have `subgroups` on the pinned wgpu (a browser cannot request the feature).
fn capabilities(inputs: &Inputs<'_>, profile: &str) -> Result<Vec<&'static str>, String> {
    let Some(value) = inputs.profile else {
        return Ok(Vec::new());
    };
    let mut found: Vec<&'static str> = Vec::new();
    for name in value.split('+').skip(1) {
        if name.is_empty() {
            return Err(format!(
                "gust: GUST_PROFILE `{value}` has an empty capability name after `+`; write <target>[+<capability>], with the capability {}",
                CAPABILITIES.join(" or ")
            ));
        }
        if RESERVED_CAPABILITIES.contains(&name) {
            return Err(format!(
                "gust: capability `{name}` in GUST_PROFILE `{value}` is not supported in this release; use {}",
                CAPABILITIES.join(" or ")
            ));
        }
        let Some(known) = CAPABILITIES.iter().find(|known| **known == name) else {
            return Err(format!(
                "gust: unknown capability `{name}` in GUST_PROFILE `{value}`; use {}",
                CAPABILITIES.join(" or ")
            ));
        };
        if found.contains(known) {
            return Err(format!(
                "gust: GUST_PROFILE `{value}` names the capability `{name}` twice"
            ));
        }
        found.push(known);
    }
    if found.is_empty() {
        return Ok(found);
    }
    if profile == "cpu" {
        return Err(format!(
            "gust: GUST_PROFILE `{value}`: the cpu profile takes no capability; a CPU build runs on the host and holds no GPU runtime"
        ));
    }
    if profile == "web" && found.contains(&"subgroups") {
        return Err(format!(
            "gust: GUST_PROFILE `{value}`: the capability subgroups is not available under the web profile in this release (the pinned wgpu cannot request it in a browser); use native or dev"
        ));
    }
    found.sort_by_key(|name| CAPABILITIES.iter().position(|known| known == name));
    Ok(found)
}

/// The cargo directive that declares the values of `cfg(gust_capability)`.
fn check_cfg_capability() -> String {
    let values: Vec<String> = CAPABILITIES
        .iter()
        .map(|name| format!("\"{name}\""))
        .collect();
    format!(
        "cargo:rustc-check-cfg=cfg(gust_capability, values({}))",
        values.join(", ")
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
    let capabilities = capabilities(inputs, profile)?;
    if inputs.slang_runtime && profile == "cpu" {
        // `cpu` is never inferred, so there is no origin to name.
        return Err("gust: profile cpu refuses feature slang-runtime; a CPU build contains no compiler, bridge or GPU runtime. Do not enable the feature for this build (look for features = [\"slang-runtime\"] in a dependency, or --features ...slang-runtime; `cargo tree -e features -i gust-compute` shows who enables it), or build with GUST_PROFILE=dev.".to_owned());
    }
    if inputs.slang_runtime && profile != "dev" {
        let origin = if inferred {
            " This profile was inferred from a release build without GUST_PROFILE."
        } else {
            ""
        };
        // The way out keeps the capabilities: `native+subgroups` is built as `dev+subgroups` instead.
        let suffix: String = capabilities.iter().map(|name| format!("+{name}")).collect();
        return Err(format!(
            "gust: profile {profile} refuses feature slang-runtime; a strict profile builds no run-time Slang \
             bridge. Do not enable the feature for this build (look for features = [\"slang-runtime\"] in a \
             dependency, or --features ...slang-runtime; `cargo tree -e features -i gust-compute` shows who enables it), \
             or build with GUST_PROFILE=dev{suffix}.{origin}"
        ));
    }
    let mut lines = vec![
        "cargo:rerun-if-changed=build.rs".to_owned(),
        "cargo:rerun-if-env-changed=GUST_PROFILE".to_owned(),
        check_cfg(),
        check_cfg_capability(),
    ];
    if inferred && profile == "dev" {
        // The bridge feature is the only way to infer `dev`.
        lines.push(BRIDGE_WARNING.to_owned());
    } else if inferred {
        lines.push(format!(
            "cargo:warning=gust: release build without GUST_PROFILE treated as {profile}"
        ));
    }
    lines.push(format!("cargo:rustc-cfg=gust_profile=\"{profile}\""));
    for name in capabilities {
        lines.push(format!("cargo:rustc-cfg=gust_capability=\"{name}\""));
    }
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
                "CPU",
                "gust: unknown GUST_PROFILE `CPU`; use dev, native, web or cpu",
            ),
            (
                "+subgroups",
                "gust: unknown GUST_PROFILE `+subgroups`; use dev, native, web or cpu",
            ),
            (
                "gpu+subgroups",
                "gust: unknown GUST_PROFILE `gpu+subgroups`; use dev, native, web or cpu",
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
                message.contains("cargo tree -e features -i gust-compute"),
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
            Err("gust: profile native refuses feature slang-runtime; a strict profile builds no run-time Slang bridge. Do not enable the feature for this build (look for features = [\"slang-runtime\"] in a dependency, or --features ...slang-runtime; `cargo tree -e features -i gust-compute` shows who enables it), or build with GUST_PROFILE=dev.".to_owned())
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
                Err("gust: profile cpu refuses feature slang-runtime; a CPU build contains no compiler, bridge or GPU runtime. Do not enable the feature for this build (look for features = [\"slang-runtime\"] in a dependency, or --features ...slang-runtime; `cargo tree -e features -i gust-compute` shows who enables it), or build with GUST_PROFILE=dev.".to_owned())
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
        for slang_runtime in [false, true] {
            let inputs = Inputs {
                release: true,
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

    #[test]
    fn the_bridge_feature_keeps_the_wasm32_inference_strict() {
        // A release build for wasm32 without a variable is still `web`, with the plain warning, and still refuses
        // the bridge: `cargo install` never targets wasm32, so the exception has no reason to reach it.
        let wasm = Inputs {
            release: true,
            wasm: true,
            ..BRIDGE_ON
        };
        assert_eq!(effective(&wasm), Ok(("web", true)));
        let lines = directives(&wasm).unwrap();
        assert!(lines.contains(&cfg_line("web")), "{lines:?}");
        assert!(lines.contains(&warning("web")), "{lines:?}");
        assert!(!lines.contains(&bridge_warning()), "{lines:?}");
        let with_runtime = Inputs {
            slang_runtime: true,
            ..wasm
        };
        assert!(
            directives(&with_runtime)
                .unwrap_err()
                .starts_with("gust: profile web refuses feature slang-runtime; ")
        );
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

    /// A parsed value: its target and capabilities.
    type Parsed<'a> = (&'a str, &'a [&'a str]);

    /// The grammar of `GUST_PROFILE`: every row is (value, the target and the capabilities it parses to, or `None`
    /// when the build script refuses it). `Profile::parse` of `gust-macros` has the same table (its test module
    /// `tests/manifest.rs`), pasted (rows verbatim), so the two copies of the grammar cannot drift apart unnoticed.
    const GRAMMAR: &[(&str, Option<Parsed<'static>>)] = &[
        ("dev", Some(("dev", &[]))),
        ("native", Some(("native", &[]))),
        ("web", Some(("web", &[]))),
        ("cpu", Some(("cpu", &[]))),
        ("dev+subgroups", Some(("dev", &["subgroups"]))),
        ("native+subgroups", Some(("native", &["subgroups"]))),
        ("", None),
        ("bogus", None),
        ("Native", None),
        (" native", None),
        ("native ", None),
        ("+subgroups", None),
        ("native+", None),
        ("native++subgroups", None),
        ("native+subgroups+", None),
        ("native+ subgroups", None),
        ("native+Subgroups", None),
        ("native+subgroup", None),
        ("native+subgroups+subgroups", None),
        ("web+subgroups", None),
        ("cpu+subgroups", None),
        ("cpu+simd", None),
        ("native+gpu", None),
        ("native+f16", None),
        ("dev+f16", None),
        ("native+atomics64", None),
        ("dev+subgroups+f16", None),
    ];

    #[test]
    fn the_grammar_table_is_what_the_build_script_accepts() {
        for (value, parsed) in GRAMMAR {
            let inputs = Inputs {
                profile: Some(value),
                ..NONE
            };
            let result = directives(&inputs);
            match parsed {
                Some((target, names)) => {
                    let lines = result.unwrap_or_else(|error| panic!("{value:?}: {error}"));
                    assert!(lines.contains(&cfg_line(target)), "{value:?}: {lines:?}");
                    let cfgs: Vec<&String> = lines
                        .iter()
                        .filter(|line| line.starts_with("cargo:rustc-cfg=gust_capability="))
                        .collect();
                    let wanted: Vec<String> = names
                        .iter()
                        .map(|name| format!("cargo:rustc-cfg=gust_capability=\"{name}\""))
                        .collect();
                    assert_eq!(cfgs, wanted.iter().collect::<Vec<_>>(), "{value:?}");
                }
                None => assert!(result.is_err(), "{value:?} must be refused: {result:?}"),
            }
        }
    }

    #[test]
    fn a_capability_that_does_not_exist_is_refused_by_name() {
        // These two inputs were `unknown GUST_PROFILE` before capabilities existed; they are still refused.
        for (value, message) in [
            (
                "native+gpu",
                "gust: unknown capability `gpu` in GUST_PROFILE `native+gpu`; use subgroups",
            ),
            (
                "cpu+simd",
                "gust: unknown capability `simd` in GUST_PROFILE `cpu+simd`; use subgroups",
            ),
            (
                "native+Subgroups",
                "gust: unknown capability `Subgroups` in GUST_PROFILE `native+Subgroups`; use subgroups",
            ),
        ] {
            let inputs = Inputs {
                profile: Some(value),
                ..NONE
            };
            assert_eq!(directives(&inputs), Err(message.to_owned()), "{value:?}");
        }
    }

    #[test]
    fn the_reserved_capabilities_parse_and_are_refused_as_not_supported() {
        for name in ["f16", "atomics64"] {
            for target in ["dev", "native", "web", "cpu"] {
                let value = format!("{target}+{name}");
                let inputs = Inputs {
                    profile: Some(&value),
                    ..NONE
                };
                assert_eq!(
                    directives(&inputs),
                    Err(format!(
                        "gust: capability `{name}` in GUST_PROFILE `{value}` is not supported in this release; use subgroups"
                    )),
                    "{value}"
                );
            }
        }
    }

    #[test]
    fn an_empty_or_repeated_capability_is_refused() {
        for value in ["native+", "native++subgroups", "native+subgroups+"] {
            let inputs = Inputs {
                profile: Some(value),
                ..NONE
            };
            assert_eq!(
                directives(&inputs),
                Err(format!(
                    "gust: GUST_PROFILE `{value}` has an empty capability name after `+`; write <target>[+<capability>], with the capability subgroups"
                )),
                "{value}"
            );
        }
        let twice = Inputs {
            profile: Some("native+subgroups+subgroups"),
            ..NONE
        };
        assert_eq!(
            directives(&twice),
            Err("gust: GUST_PROFILE `native+subgroups+subgroups` names the capability `subgroups` twice".to_owned())
        );
    }

    #[test]
    fn the_cpu_profile_takes_no_capability_and_web_cannot_have_subgroups() {
        let cpu = Inputs {
            profile: Some("cpu+subgroups"),
            ..NONE
        };
        assert_eq!(
            directives(&cpu),
            Err("gust: GUST_PROFILE `cpu+subgroups`: the cpu profile takes no capability; a CPU build runs on the host and holds no GPU runtime".to_owned())
        );
        let web = Inputs {
            profile: Some("web+subgroups"),
            ..NONE
        };
        assert_eq!(
            directives(&web),
            Err("gust: GUST_PROFILE `web+subgroups`: the capability subgroups is not available under the web profile in this release (the pinned wgpu cannot request it in a browser); use native or dev".to_owned())
        );
    }

    #[test]
    fn dev_and_native_with_subgroups_set_one_capability_cfg_and_declare_it() {
        for value in ["dev+subgroups", "native+subgroups"] {
            let inputs = Inputs {
                profile: Some(value),
                ..NONE
            };
            let lines = directives(&inputs).unwrap();
            let target = value.split('+').next().unwrap();
            assert!(lines.contains(&cfg_line(target)), "{lines:?}");
            assert_eq!(
                lines
                    .iter()
                    .filter(|line| line.starts_with("cargo:rustc-cfg="))
                    .collect::<Vec<_>>(),
                [
                    &cfg_line(target),
                    &"cargo:rustc-cfg=gust_capability=\"subgroups\"".to_owned()
                ],
                "{lines:?}"
            );
            assert!(
                lines.contains(
                    &"cargo:rustc-check-cfg=cfg(gust_capability, values(\"subgroups\"))".to_owned()
                ),
                "{lines:?}"
            );
            assert!(
                !lines.iter().any(|line| line.contains("warning")),
                "{lines:?}"
            );
        }
        // Without a capability there is no capability cfg, but the cfg is still declared.
        let lines = directives(&NONE).unwrap();
        assert!(
            !lines
                .iter()
                .any(|line| line.starts_with("cargo:rustc-cfg=gust_capability=")),
            "{lines:?}"
        );
        assert!(
            lines.contains(
                &"cargo:rustc-check-cfg=cfg(gust_capability, values(\"subgroups\"))".to_owned()
            ),
            "{lines:?}"
        );
    }

    #[test]
    fn the_slang_runtime_feature_follows_the_target_not_the_capability() {
        // `dev+subgroups` is the profile of the tests of subgroup kernels: it keeps the bridge.
        let dev = Inputs {
            profile: Some("dev+subgroups"),
            slang_runtime: true,
            ..NONE
        };
        assert!(directives(&dev).is_ok());
        // A strict target refuses it, and the way out keeps the capability.
        let native = Inputs {
            profile: Some("native+subgroups"),
            slang_runtime: true,
            ..NONE
        };
        let message = directives(&native).unwrap_err();
        assert!(
            message.starts_with("gust: profile native refuses feature slang-runtime; "),
            "{message}"
        );
        assert!(
            message.ends_with("or build with GUST_PROFILE=dev+subgroups."),
            "{message}"
        );
        // An invalid capability is reported before the feature refusal, as an unknown profile is.
        let bad = Inputs {
            profile: Some("native+f16"),
            slang_runtime: true,
            ..NONE
        };
        assert!(
            directives(&bad)
                .unwrap_err()
                .starts_with("gust: capability `f16`")
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
