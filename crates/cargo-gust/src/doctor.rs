//! `cargo gust doctor`: what this machine and this directory give a build.
//!
//! The facts are gathered by the caller; [`lines`] turns them into text, so the wording and the profile rule are
//! unit tested without touching the environment.
//!
//! The profile rule is a copy of the pure function in `crates/gust/build.rs` (`effective`): that file is a build
//! script, not a library, so it cannot be shared; its unit tests and the ones below pin the same table. If the rule
//! changes in `gust`, change it here too.

use std::path::PathBuf;

use crate::tools::helper_hint;

/// The values `GUST_PROFILE` may take.
const PROFILES: [&str; 3] = ["dev", "native", "web"];

/// `dev, native or web`.
fn allowed() -> String {
    format!("{}, {} or {}", PROFILES[0], PROFILES[1], PROFILES[2])
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
    match (&facts.helper_found, &facts.helper_slang) {
        (Some(path), slang) => {
            out.push(format!("helper: found at {}", path.display()));
            match slang {
                Some(Ok(tag)) => out.push(format!("  Slang build: {tag}")),
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
            "GUST_PROFILE is set but empty; use dev, native or web (or unset it)"
        );
        assert_eq!(
            effective_profile(Some("fast"), true, false).unwrap_err(),
            "unknown GUST_PROFILE `fast`; use dev, native or web"
        );
        assert!(effective_profile(Some("Dev"), false, false).is_err());
    }

    fn facts() -> Facts {
        Facts {
            version: "0.1.0".to_owned(),
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
}
