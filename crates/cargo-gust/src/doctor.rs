//! `cargo gust doctor`: what this machine and this directory give a build.
//!
//! The facts are gathered by the caller; [`lines`] turns them into text, so the wording and the profile rule are
//! unit tested without touching the environment.
//!
//! The profile rule is a copy of the pure functions in `crates/gust/build.rs` (`effective` and `capabilities`): that
//! file is a build script, not a library, so it cannot be shared; its unit tests and the ones below pin the same
//! table (`GRAMMAR`, compared as text by a test below), and the messages are the build script's without its `gust: `
//! prefix. If the rule changes in `gust`, change it here too.

use std::path::PathBuf;

use crate::{
    slang::{Located, Source, VALIDATED_BUILDS, matches_pin},
    toolchain::check_compatible,
    tools::helper_hint,
};

/// The version of `gust-compute` (the `gust` crate) this tool is built against: the library's own, from
/// `gust::toolchain`.
const GUST_COMPUTE_VERSION: &str = gust::toolchain::GUST_VERSION;

/// The values `GUST_PROFILE` may take.
const PROFILES: [&str; 4] = ["dev", "native", "web", "cpu"];

/// The capabilities `GUST_PROFILE` may name after the target (`dev+subgroups`).
const CAPABILITIES: [&str; 1] = ["subgroups"];

/// Capabilities that parse and are refused as not supported in this release.
const RESERVED_CAPABILITIES: [&str; 2] = ["f16", "atomics64"];

/// `dev, native, web or cpu`.
fn allowed() -> String {
    format!(
        "{}, {}, {} or {}",
        PROFILES[0], PROFILES[1], PROFILES[2], PROFILES[3]
    )
}

/// What ends the refusal of a reserved or unknown capability under `target`: what the target takes.
fn accepted(target: &str) -> String {
    if target == "web" {
        "the web profile takes no capability on the pinned wgpu (a browser cannot request one)"
            .to_owned()
    } else {
        format!("use {}", CAPABILITIES.join(" or "))
    }
}

/// The capabilities `value` (`<target>(+<capability>)*`, whose target is `target`) names, or the reason the build
/// script of `gust` refuses it: the same rule and the same texts, without the `gust: ` prefix (M8, D138).
fn capabilities(value: &str, target: &str) -> Result<Vec<&'static str>, String> {
    let mut found: Vec<&'static str> = Vec::new();
    for name in value.split('+').skip(1) {
        if name.is_empty() {
            return Err(format!(
                "GUST_PROFILE `{value}` has an empty capability name after `+`; write <target>[+<capability>], with the capability {}",
                CAPABILITIES.join(" or ")
            ));
        }
        if target == "cpu" {
            return Err(format!(
                "GUST_PROFILE `{value}`: the cpu profile takes no capability; a CPU build runs on the host and holds no GPU runtime"
            ));
        }
        if RESERVED_CAPABILITIES.contains(&name) {
            return Err(format!(
                "capability `{name}` in GUST_PROFILE `{value}` is not supported in this release; {}",
                accepted(target)
            ));
        }
        let Some(known) = CAPABILITIES.iter().find(|known| **known == name) else {
            return Err(format!(
                "unknown capability `{name}` in GUST_PROFILE `{value}`; {}",
                accepted(target)
            ));
        };
        if found.contains(known) {
            return Err(format!(
                "GUST_PROFILE `{value}` names the capability `{name}` twice"
            ));
        }
        found.push(known);
    }
    if target == "web" && found.contains(&"subgroups") {
        return Err(format!(
            "GUST_PROFILE `{value}`: the capability subgroups is not available under the web profile in this release (the pinned wgpu cannot request it in a browser); use native or dev"
        ));
    }
    Ok(found)
}

/// The effective profile (its target) of a build and whether it was inferred from a release build (no explicit
/// `GUST_PROFILE`). Same rule as `gust`'s build script (D44, R2-2), including the capabilities after the target
/// (`dev+subgroups` is healthy; `web+subgroups` and `cpu+subgroups` are refused, as the build script refuses them).
pub(crate) fn effective_profile(
    profile: Option<&str>,
    release: bool,
    wasm: bool,
) -> Result<(&'static str, bool), String> {
    match profile {
        Some("") => Err(format!(
            "GUST_PROFILE is set but empty; use {} (or unset it)",
            allowed()
        )),
        Some(value) => {
            let target = value.split('+').next().unwrap_or_default();
            let known = PROFILES
                .iter()
                .find(|known| **known == target)
                .ok_or_else(|| format!("unknown GUST_PROFILE `{value}`; use {}", allowed()))?;
            capabilities(value, known)?;
            Ok((*known, false))
        }
        None if release && wasm => Ok(("web", true)),
        None if release => Ok(("native", true)),
        None => Ok(("dev", false)),
    }
}

/// What `doctor` found out.
pub(crate) struct Facts {
    /// This tool's version.
    pub(crate) version: String,
    /// The Slang version this tool is pinned to.
    pub(crate) pin: String,
    /// The pinned `slangc` that `locate` found, or why it found none.
    pub(crate) slangc: Result<Located, String>,
    /// The licence and notice files beside the binaries of a managed install.
    pub(crate) notices: Vec<PathBuf>,
    /// The first `slangc` on `PATH` and what its `-version` gave (or why it gave none); `None` without one.
    pub(crate) path_slangc: Option<(PathBuf, Result<String, String>)>,
    /// The helper path `gust` resolved (`GUST_SLANG_REFLECT`, the repository's `target/slang-reflect`, or a bare name).
    pub(crate) helper_candidate: PathBuf,
    /// Where the helper was found, if it was.
    pub(crate) helper_found: Option<PathBuf>,
    /// The helper's Slang build tag, or why it could not be read; `None` when the helper was not found.
    pub(crate) helper_slang: Option<Result<String, String>>,
    pub(crate) spirv_val: Option<PathBuf>,
    /// `GUST_PROFILE` as the environment holds it.
    pub(crate) gust_profile: Option<String>,
}

/// The warning for a managed install that a different `slangc` on `PATH` shadows, or `None`. `cargo gust` runs the
/// managed `slangc`, but tools that do not know it (such as `cargo xtask` in a GUST checkout) take the first one on
/// `PATH`, so the session would mix two Slang versions. A `slangc` that `locate` took from `PATH` needs no warning (it
/// is the one reported above), and a failed `locate` already names the `PATH` version and the install command in its
/// own message. The fix is the managed `bin` directory first on `PATH`.
fn path_slangc_warning(facts: &Facts) -> Option<String> {
    let located = facts.slangc.as_ref().ok()?;
    if located.source == Source::Path {
        return None;
    }
    let (path, version) = facts.path_slangc.as_ref()?;
    let managed = located.slangc.parent().unwrap_or(&located.slangc);
    let cause = match version {
        Ok(version) if matches_pin(version, VALIDATED_BUILDS) => return None,
        Ok(version) => format!("is Slang `{version}`, not the pin {}", facts.pin),
        Err(error) => format!(
            "cannot report its version ({error}); the pin is {}",
            facts.pin
        ),
    };
    Some(format!(
        "warning: the first slangc on PATH ({}) {cause}; cargo gust uses the managed install, but tools that do \
         not know it (such as `cargo xtask` in a GUST checkout) take PATH and would mix versions: put {} first on PATH",
        path.display(),
        managed.display()
    ))
}

/// The report, one entry per line, and whether everything `cargo gust build` needs is in place.
pub(crate) fn lines(facts: &Facts) -> (Vec<String>, bool) {
    // Both versions: the tool and the library whose run-time bridge it links (`gust-compute`, the `gust` crate).
    let mut out = vec![
        format!("cargo-gust {}", facts.version),
        format!("gust-compute {GUST_COMPUTE_VERSION}"),
    ];
    let mut healthy = true;
    match &facts.slangc {
        Ok(located) => {
            out.push(format!("slangc: found at {}", located.slangc.display()));
            out.push(format!(
                "  Slang {} ({}); the pin is {}",
                located.version,
                located.source.label(),
                facts.pin
            ));
            if !located.pinned {
                healthy = false;
                out.push("  this is not the pinned version".to_owned());
            }
            match located.source {
                Source::Path => out.push(
                    "  licence notices: this slangc is on PATH, so cargo gust does not manage its notices; the release's own LICENSE, LICENSES/ and third-party-notices/ sit beside its binaries (`cargo gust slang install` keeps them beside a cached copy)"
                        .to_owned(),
                ),
                Source::Home | Source::Cache if facts.notices.is_empty() => out.push(
                    "  licence notices: none found beside the binaries of this install".to_owned(),
                ),
                Source::Home | Source::Cache => {
                    for notice in &facts.notices {
                        out.push(format!("  licence notice: {}", notice.display()));
                    }
                }
            }
        }
        Err(error) => {
            healthy = false;
            out.push(format!(
                "slangc: NOT found as the pinned Slang {}: {error}",
                facts.pin
            ));
        }
    }
    if let Some(warning) = path_slangc_warning(facts) {
        out.push(warning);
    }
    match (&facts.helper_found, &facts.helper_slang) {
        (Some(path), slang) => {
            out.push(format!("helper: found at {}", path.display()));
            match slang {
                Some(Ok(tag)) => {
                    out.push(format!("  Slang build: {tag}"));
                    if let Ok(located) = &facts.slangc
                        && let Err(error) = check_compatible(path, tag, located)
                    {
                        healthy = false;
                        out.push(format!("  helper refused: {error}"));
                    }
                }
                Some(Err(error)) => {
                    healthy = false;
                    out.push(format!("  cannot read its version: {error}"));
                }
                None => {}
            }
        }
        (None, _) => {
            healthy = false;
            out.push(format!(
                "helper: NOT found (looked for {}); {}",
                facts.helper_candidate.display(),
                helper_hint()
            ));
        }
    }
    match &facts.spirv_val {
        Some(path) => out.push(format!("spirv-val: found at {}", path.display())),
        None => out.push(
            "spirv-val: not on PATH (`build --format spirv` would write the module with `validated: false`)"
                .to_owned(),
        ),
    }
    out.push(
        "this reads GUST_PROFILE from the process environment only; a value set in .cargo/config.toml [env] reaches cargo's build scripts but not this report"
            .to_owned(),
    );
    let explicit = facts.gust_profile.as_deref();
    match explicit {
        Some(value) => out.push(format!("GUST_PROFILE: {value}")),
        None => out.push("GUST_PROFILE: not set".to_owned()),
    }
    for (label, release, wasm) in [
        ("cargo build", false, false),
        ("cargo build --release", true, false),
        (
            "cargo build --release --target wasm32-unknown-unknown",
            true,
            true,
        ),
    ] {
        match effective_profile(explicit, release, wasm) {
            Ok((profile, true)) => out.push(format!(
                "  profile of `{label}`: {profile} (inferred from the release build; a kernel without an artifact fails the build and the slang-runtime feature is refused)"
            )),
            Ok((profile, false)) => {
                // The capabilities of a valid explicit value (`+subgroups`) are part of what the build is.
                let suffix = explicit
                    .and_then(|value| value.strip_prefix(profile))
                    .unwrap_or_default();
                out.push(format!("  profile of `{label}`: {profile}{suffix}"));
            }
            Err(error) => {
                healthy = false;
                out.push(format!("  profile of `{label}`: {error}"));
            }
        }
    }
    out.push(
        "`cargo gust build` and `check` collect under GUST_PROFILE=dev whatever this says (collecting is not a production build)"
            .to_owned(),
    );
    out.push(
        "install: cargo install cargo-gust --locked --version ~0.1 (this tool enables gust's internal cargo-gust-bridge feature, so a plain release install needs no GUST_PROFILE)"
            .to_owned(),
    );
    (out, healthy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        slang::{self, Env},
        store::tests::Scratch,
        tools::executable_name,
    };
    use std::fs;

    #[test]
    fn the_profile_rule_matches_the_build_script_of_gust() {
        assert_eq!(effective_profile(None, false, false), Ok(("dev", false)));
        assert_eq!(effective_profile(None, false, true), Ok(("dev", false)));
        assert_eq!(effective_profile(None, true, false), Ok(("native", true)));
        assert_eq!(effective_profile(None, true, true), Ok(("web", true)));
        for release in [false, true] {
            for wasm in [false, true] {
                for name in PROFILES {
                    assert_eq!(
                        effective_profile(Some(name), release, wasm),
                        Ok((name, false))
                    );
                }
            }
        }
        assert_eq!(
            effective_profile(Some(""), false, false).unwrap_err(),
            "GUST_PROFILE is set but empty; use dev, native, web or cpu (or unset it)"
        );
        assert_eq!(
            effective_profile(Some("fast"), true, false).unwrap_err(),
            "unknown GUST_PROFILE `fast`; use dev, native, web or cpu"
        );
        assert!(effective_profile(Some("Dev"), false, false).is_err());
    }

    /// The grammar of `GUST_PROFILE`, pasted from the test module of `crates/gust/build.rs` (the macro's test module
    /// has the same rows); `the_grammar_table_is_the_one_in_the_build_script_row_for_row` keeps the copies equal.
    type Parsed<'a> = (&'a str, &'a [&'a str]);

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
    fn the_grammar_is_the_one_of_the_build_script() {
        for (value, parsed) in GRAMMAR {
            let result = effective_profile(Some(value), false, false);
            match parsed {
                Some((target, names)) => {
                    assert_eq!(result, Ok((*target, false)), "{value:?}");
                    let found = capabilities(value, target).unwrap();
                    assert_eq!(&found[..], *names, "{value:?}");
                }
                None => assert!(result.is_err(), "{value:?} must be refused: {result:?}"),
            }
        }
    }

    /// The rows of the `GRAMMAR` table in the Rust `source`, trimmed, from the line of `const GRAMMAR` to `];`.
    fn grammar_rows(source: &str) -> Vec<String> {
        let mut rows = Vec::new();
        let mut inside = false;
        for line in source.lines().map(str::trim) {
            if line.starts_with("const GRAMMAR") {
                inside = true;
            } else if inside && line == "];" {
                return rows;
            } else if inside {
                rows.push(line.to_owned());
            }
        }
        panic!("no complete GRAMMAR table");
    }

    #[test]
    fn the_grammar_table_is_the_one_in_the_build_script_row_for_row() {
        // The build script is a binary of another crate: its test table is compared as text, so a row, a capability or
        // a refusal added there and not here fails this test.
        let build_script = include_str!("../../gust/build.rs");
        let theirs = grammar_rows(build_script);
        assert!(theirs.len() > 20, "{theirs:?}");
        assert_eq!(grammar_rows(include_str!("doctor.rs")), theirs);
    }

    #[test]
    fn the_refusals_are_the_texts_of_the_build_script_without_its_prefix() {
        for (value, message) in [
            (
                "native+",
                "GUST_PROFILE `native+` has an empty capability name after `+`; write <target>[+<capability>], with the capability subgroups",
            ),
            (
                "native+gpu",
                "unknown capability `gpu` in GUST_PROFILE `native+gpu`; use subgroups",
            ),
            (
                "native+f16",
                "capability `f16` in GUST_PROFILE `native+f16` is not supported in this release; use subgroups",
            ),
            (
                "web+f16",
                "capability `f16` in GUST_PROFILE `web+f16` is not supported in this release; the web profile takes no capability on the pinned wgpu (a browser cannot request one)",
            ),
            (
                "native+subgroups+subgroups",
                "GUST_PROFILE `native+subgroups+subgroups` names the capability `subgroups` twice",
            ),
            (
                "cpu+subgroups",
                "GUST_PROFILE `cpu+subgroups`: the cpu profile takes no capability; a CPU build runs on the host and holds no GPU runtime",
            ),
            (
                "cpu+simd",
                "GUST_PROFILE `cpu+simd`: the cpu profile takes no capability; a CPU build runs on the host and holds no GPU runtime",
            ),
            (
                "web+subgroups",
                "GUST_PROFILE `web+subgroups`: the capability subgroups is not available under the web profile in this release (the pinned wgpu cannot request it in a browser); use native or dev",
            ),
            (
                "bogus+subgroups",
                "unknown GUST_PROFILE `bogus+subgroups`; use dev, native, web or cpu",
            ),
        ] {
            assert_eq!(
                effective_profile(Some(value), false, false).unwrap_err(),
                message,
                "{value}"
            );
        }
    }

    #[test]
    fn a_profile_with_a_capability_is_healthy_and_a_refused_one_is_not() {
        let mut healthy_facts = facts();
        healthy_facts.gust_profile = Some("dev+subgroups".to_owned());
        let (text, healthy) = lines(&healthy_facts);
        assert!(healthy, "{text:?}");
        assert!(
            text.contains(&"GUST_PROFILE: dev+subgroups".to_owned()),
            "{text:?}"
        );
        assert!(
            text.contains(&"  profile of `cargo build`: dev+subgroups".to_owned()),
            "{text:?}"
        );
        let mut refused = facts();
        refused.gust_profile = Some("web+subgroups".to_owned());
        let (text, healthy) = lines(&refused);
        assert!(!healthy, "{text:?}");
        assert!(
            text.iter().any(|line| line.contains(
                "profile of `cargo build`: GUST_PROFILE `web+subgroups`: the capability subgroups is not available under the web profile"
            )),
            "{text:?}"
        );
    }

    fn facts() -> Facts {
        Facts {
            version: "0.1.0".to_owned(),
            pin: "2026.20".to_owned(),
            slangc: Ok(Located {
                slangc: PathBuf::from("/slang/2026.20/bin/slangc"),
                version: "2026.20".to_owned(),
                source: Source::Cache,
                pinned: true,
                root: Some(PathBuf::from("/slang")),
            }),
            notices: vec![
                PathBuf::from("/slang/2026.20/LICENSE"),
                PathBuf::from("/slang/2026.20/LICENSES"),
                PathBuf::from("/slang/2026.20/third-party-notices"),
            ],
            path_slangc: None,
            helper_candidate: PathBuf::from("gust-slang-reflect"),
            helper_found: Some(PathBuf::from("/tools/gust-slang-reflect")),
            helper_slang: Some(Ok("2026.20".to_owned())),
            spirv_val: Some(PathBuf::from("/tools/spirv-val")),
            gust_profile: None,
        }
    }

    #[test]
    fn a_complete_setup_is_healthy_and_shows_the_inferred_release_profiles() {
        let (text, healthy) = lines(&facts());
        assert!(healthy);
        assert_eq!(text[0], "cargo-gust 0.1.0");
        // The library's version comes from `gust` itself, so the test follows it.
        assert_eq!(
            text[1],
            format!("gust-compute {}", gust::toolchain::GUST_VERSION)
        );
        assert!(
            text.contains(&"helper: found at /tools/gust-slang-reflect".to_owned()),
            "{text:?}"
        );
        assert!(text.contains(&"  Slang build: 2026.20".to_owned()));
        assert!(text.contains(&"slangc: found at /slang/2026.20/bin/slangc".to_owned()));
        assert!(text.contains(&"  Slang 2026.20 (cargo cache); the pin is 2026.20".to_owned()));
        for notice in ["LICENSE", "LICENSES", "third-party-notices"] {
            assert!(
                text.contains(&format!("  licence notice: /slang/2026.20/{notice}")),
                "{notice}: {text:?}"
            );
        }
        assert!(text.contains(&"GUST_PROFILE: not set".to_owned()));
        assert!(
            text.iter()
                .any(|line| line.contains("process environment only")
                    && line.contains(".cargo/config.toml [env]"))
        );
        assert!(text.contains(&"  profile of `cargo build`: dev".to_owned()));
        assert!(text.iter().any(|line| {
            line.starts_with("  profile of `cargo build --release`: native (inferred")
        }));
        assert!(text.iter().any(|line| line.starts_with(
            "  profile of `cargo build --release --target wasm32-unknown-unknown`: web (inferred"
        )));
        assert!(
            text.last()
                .unwrap()
                .starts_with("install: cargo install cargo-gust --locked --version ~0.1")
        );
    }

    #[test]
    fn a_missing_helper_says_how_to_build_it_and_is_not_healthy() {
        let mut missing = facts();
        missing.helper_found = None;
        missing.helper_slang = None;
        missing.spirv_val = None;
        let (text, healthy) = lines(&missing);
        assert!(!healthy);
        let helper = text
            .iter()
            .find(|line| line.starts_with("helper: NOT found"))
            .unwrap();
        assert!(
            helper.contains("looked for gust-slang-reflect")
                && helper.contains("cargo xtask build-slang-reflect")
                && helper.contains("GUST_SLANG_REFLECT"),
            "{helper}"
        );
        assert!(
            text.iter()
                .any(|line| line.starts_with("spirv-val: not on PATH"))
        );
    }

    #[test]
    fn an_explicit_profile_applies_to_every_build_and_a_bad_one_is_reported() {
        let mut dev = facts();
        dev.gust_profile = Some("dev".to_owned());
        let (text, healthy) = lines(&dev);
        assert!(healthy);
        assert!(
            text.contains(&"  profile of `cargo build --release`: dev".to_owned()),
            "{text:?}"
        );
        let mut bad = facts();
        bad.gust_profile = Some("turbo".to_owned());
        let (text, healthy) = lines(&bad);
        assert!(!healthy);
        assert!(
            text.iter()
                .any(|line| line.contains("unknown GUST_PROFILE `turbo`"))
        );
        let mut unreadable = facts();
        unreadable.helper_slang = Some(Err("exit 3".to_owned()));
        assert!(!lines(&unreadable).1);
    }

    #[test]
    fn a_helper_built_against_another_slang_is_refused_and_build_helper_is_offered() {
        // Mutation (b): the helper tag differs from `slangc -version`.
        let mut mismatched = facts();
        mismatched.helper_slang = Some(Ok("2025.1.0".to_owned()));
        let (text, healthy) = lines(&mismatched);
        assert!(!healthy);
        let refused = text
            .iter()
            .find(|line| line.starts_with("  helper refused:"))
            .unwrap_or_else(|| panic!("{text:?}"));
        assert!(
            refused.contains("built against Slang `2025.1.0`")
                && refused.contains("is Slang `2026.20`")
                && refused.contains("cargo gust slang build-helper"),
            "{refused}"
        );
        // The pin matching is not enough: two builds of the pinned version still differ.
        let mut other_build = facts();
        other_build.helper_slang = Some(Ok("2026.20-1-g0a1b2c3d4".to_owned()));
        assert!(!lines(&other_build).1);
    }

    fn warning_of(facts: &Facts) -> Option<String> {
        lines(facts)
            .0
            .into_iter()
            .find(|line| line.starts_with("warning: the first slangc on PATH"))
    }

    #[test]
    fn a_path_slangc_of_another_version_is_named_with_both_versions_and_the_managed_fix() {
        // Mutation (b): the managed 2026.20 is found, but the first slangc on PATH reports another version.
        let mut shadowed = facts();
        shadowed.path_slangc = Some((
            PathBuf::from("/vulkan/Bin/slangc"),
            Ok("2026.13.1".to_owned()),
        ));
        let (text, healthy) = lines(&shadowed);
        // cargo gust itself runs the managed slangc, so the build is possible; the report still says what is wrong.
        assert!(healthy);
        let warning = warning_of(&shadowed).unwrap_or_else(|| panic!("{text:?}"));
        assert_eq!(
            warning,
            "warning: the first slangc on PATH (/vulkan/Bin/slangc) is Slang `2026.13.1`, not the pin 2026.20; cargo \
             gust uses the managed install, but tools that do not know it (such as `cargo xtask` in a GUST checkout) \
             take PATH and would mix versions: put /slang/2026.20/bin first on PATH"
        );
        // A PATH slangc that cannot report its version is named too.
        shadowed.path_slangc = Some((
            PathBuf::from("/vulkan/Bin/slangc"),
            Err("exit 3".to_owned()),
        ));
        assert!(
            warning_of(&shadowed)
                .unwrap()
                .contains("cannot report its version (exit 3); the pin is 2026.20")
        );
        // The pin on PATH, no slangc on PATH, or the located slangc being the PATH one: nothing to add.
        shadowed.path_slangc = Some((PathBuf::from("/other/slangc"), Ok("2026.20".to_owned())));
        assert_eq!(warning_of(&shadowed), None);
        shadowed.path_slangc = None;
        assert_eq!(warning_of(&shadowed), None);
        let mut on_path = facts();
        if let Ok(located) = &mut on_path.slangc {
            located.source = Source::Path;
            located.root = None;
        }
        on_path.path_slangc = Some((PathBuf::from("/p/slangc"), Ok("2026.20".to_owned())));
        assert_eq!(warning_of(&on_path), None);
    }

    #[test]
    fn without_an_install_a_mismatched_path_slangc_is_refused_naming_both_versions_and_the_install()
    {
        // The same machine without a managed install: `locate` refuses the PATH slangc, and doctor carries its words.
        let scratch = Scratch::new("doctor-path-slangc");
        let bin = scratch.0.join("bin");
        fs::create_dir_all(&bin).unwrap();
        fs::write(bin.join(executable_name("slangc")), "x").unwrap();
        let env = Env {
            slang_home: Some(scratch.0.join("home").into_os_string()),
            path: Some(std::env::join_paths([&bin]).unwrap()),
            ..Env::default()
        };
        let error = slang::locate(&env, false, &|_| Ok("2026.13.1".to_owned())).unwrap_err();
        let mut missing = facts();
        missing.slangc = Err(error);
        let (text, healthy) = lines(&missing);
        assert!(!healthy);
        let line = text
            .iter()
            .find(|line| line.starts_with("slangc: NOT found as the pinned Slang 2026.20"))
            .unwrap_or_else(|| panic!("{text:?}"));
        assert!(
            line.contains("is Slang `2026.13.1`, not 2026.20")
                && line.contains("run `cargo gust slang install`"),
            "{line}"
        );
        // The error already says it all, so no second line is added for it.
        missing.path_slangc = Some((bin.join("slangc"), Ok("2026.13.1".to_owned())));
        assert_eq!(warning_of(&missing), None);
    }

    #[test]
    fn a_missing_pinned_slangc_is_not_healthy_and_carries_the_locate_error() {
        let mut missing = facts();
        missing.slangc =
            Err("Slang 2026.20 was not found: run `cargo gust slang install`".to_owned());
        missing.notices.clear();
        let (text, healthy) = lines(&missing);
        assert!(!healthy);
        assert!(
            text.contains(
                &concat!(
                    "slangc: NOT found as the pinned Slang 2026.20: ",
                    "Slang 2026.20 was not found: run `cargo gust slang install`"
                )
                .to_owned()
            ),
            "{text:?}"
        );
        // With no slangc there is nothing to compare the helper with, so only the slangc line is a problem.
        assert!(!text.iter().any(|line| line.contains("helper refused")));
    }

    #[test]
    fn the_notices_of_an_install_are_listed_and_a_path_slangc_says_they_are_not_managed() {
        let mut none = facts();
        none.notices.clear();
        let (text, _) = lines(&none);
        assert!(text.contains(
            &"  licence notices: none found beside the binaries of this install".to_owned()
        ));
        let mut on_path = facts();
        if let Ok(located) = &mut on_path.slangc {
            located.source = Source::Path;
            located.root = None;
        }
        let (text, healthy) = lines(&on_path);
        assert!(healthy);
        assert!(
            text.iter()
                .any(|line| line.starts_with("  licence notices: this slangc is on PATH")),
            "{text:?}"
        );
        assert!(
            !text
                .iter()
                .any(|line| line.starts_with("  licence notice: "))
        );
        let mut unpinned = facts();
        if let Ok(located) = &mut unpinned.slangc {
            located.pinned = false;
        }
        assert!(!lines(&unpinned).1);
    }
}
