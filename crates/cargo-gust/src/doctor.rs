//! `cargo gust doctor`: what this machine and this directory give a build.
//!
//! The facts are gathered by the caller; [`lines`] turns them into text, so the wording and the profile rule are
//! unit tested without touching the environment.
//!
//! The profile rule is a copy of the pure function in `crates/gust/build.rs` (`effective`): that file is a build
//! script, not a library, so it cannot be shared; its unit tests and the ones below pin the same table. If the rule
//! changes in `gust`, change it here too.

use std::path::PathBuf;

use crate::{
    slang::{Located, Source},
    toolchain::check_compatible,
    tools::helper_hint,
};

/// The values `GUST_PROFILE` may take.
const PROFILES: [&str; 4] = ["dev", "native", "web", "cpu"];

/// `dev, native, web or cpu`.
fn allowed() -> String {
    format!(
        "{}, {}, {} or {}",
        PROFILES[0], PROFILES[1], PROFILES[2], PROFILES[3]
    )
}

/// The effective profile of a build and whether it was inferred from a release build (no explicit `GUST_PROFILE`).
/// Same rule as `gust`'s build script (D44, R2-2).
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
        Some(value) => PROFILES
            .iter()
            .find(|known| **known == value)
            .map(|known| (*known, false))
            .ok_or_else(|| format!("unknown GUST_PROFILE `{value}`; use {}", allowed())),
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

/// The report, one entry per line, and whether everything `cargo gust build` needs is in place.
pub(crate) fn lines(facts: &Facts) -> (Vec<String>, bool) {
    let mut out = vec![format!("cargo-gust {}", facts.version)];
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
            Ok((profile, false)) => out.push(format!("  profile of `{label}`: {profile}")),
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
        "install: GUST_PROFILE=dev cargo install cargo-gust (a plain install builds in release, infers `native` and refuses this tool's slang-runtime feature)"
            .to_owned(),
    );
    (out, healthy)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn facts() -> Facts {
        Facts {
            version: "0.1.0".to_owned(),
            pin: "2026.13.1".to_owned(),
            slangc: Ok(Located {
                slangc: PathBuf::from("/slang/2026.13.1/bin/slangc"),
                version: "2026.13.1-1-g84792eb15".to_owned(),
                source: Source::Cache,
                pinned: true,
                root: Some(PathBuf::from("/slang")),
            }),
            notices: vec![
                PathBuf::from("/slang/2026.13.1/LICENSE"),
                PathBuf::from("/slang/2026.13.1/LICENSES"),
                PathBuf::from("/slang/2026.13.1/third-party-notices"),
            ],
            helper_candidate: PathBuf::from("gust-slang-reflect"),
            helper_found: Some(PathBuf::from("/tools/gust-slang-reflect")),
            helper_slang: Some(Ok("2026.13.1-1-g84792eb15".to_owned())),
            spirv_val: Some(PathBuf::from("/tools/spirv-val")),
            gust_profile: None,
        }
    }

    #[test]
    fn a_complete_setup_is_healthy_and_shows_the_inferred_release_profiles() {
        let (text, healthy) = lines(&facts());
        assert!(healthy);
        assert_eq!(text[0], "cargo-gust 0.1.0");
        assert!(
            text.contains(&"helper: found at /tools/gust-slang-reflect".to_owned()),
            "{text:?}"
        );
        assert!(text.contains(&"  Slang build: 2026.13.1-1-g84792eb15".to_owned()));
        assert!(text.contains(&"slangc: found at /slang/2026.13.1/bin/slangc".to_owned()));
        assert!(text.contains(
            &"  Slang 2026.13.1-1-g84792eb15 (cargo cache); the pin is 2026.13.1".to_owned()
        ));
        for notice in ["LICENSE", "LICENSES", "third-party-notices"] {
            assert!(
                text.contains(&format!("  licence notice: /slang/2026.13.1/{notice}")),
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
                .starts_with("install: GUST_PROFILE=dev cargo install cargo-gust")
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
                && refused.contains("is Slang `2026.13.1-1-g84792eb15`")
                && refused.contains("cargo gust slang build-helper"),
            "{refused}"
        );
        // The pin matching is not enough: two builds of the pinned version still differ.
        let mut other_build = facts();
        other_build.helper_slang = Some(Ok("2026.13.1".to_owned()));
        assert!(!lines(&other_build).1);
    }

    #[test]
    fn a_missing_pinned_slangc_is_not_healthy_and_carries_the_locate_error() {
        let mut missing = facts();
        missing.slangc =
            Err("Slang 2026.13.1 was not found: run `cargo gust slang install`".to_owned());
        missing.notices.clear();
        let (text, healthy) = lines(&missing);
        assert!(!healthy);
        assert!(
            text.contains(
                &"slangc: NOT found as the pinned Slang 2026.13.1: Slang 2026.13.1 was not found: run `cargo gust slang install`"
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
