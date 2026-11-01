//! The toolchain a build needs: the pinned `slangc` and a native helper that was compiled against the same Slang.
//!
//! `build` and `check --recompile` call [`require`]; `doctor` gathers the same facts without stopping at the first
//! problem. The helper compatibility rule (R4.8): the helper reports the Slang build tag of the library it loads
//! (`gust-slang-reflect --version`), and that tag must equal what the located `slangc -version` prints. Both come from
//! one Slang build, so the comparison is exact, not the pin rule of [`slang::matches_pin`].

use std::path::{Path, PathBuf};

use crate::{
    slang::{self, Env, Located, SLANG_VERSION},
    tools,
};

/// What to do about a helper that does not match; part of every refusal.
pub(crate) const REBUILD_HINT: &str = "run `cargo gust slang build-helper` (add `--sdk-root <dir>` for a Slang that cargo gust did not install)";

/// The helper to look for: `GUST_SLANG_REFLECT` when set, else the helper that `build-helper` wrote for the located
/// install, else whatever `gust` resolves (the repository's `target/slang-reflect`, or the bare name on `PATH`).
pub(crate) fn helper_candidate(located: Option<&Located>) -> PathBuf {
    if std::env::var_os("GUST_SLANG_REFLECT").is_none()
        && let Some(root) = located.and_then(|located| located.root.as_deref())
    {
        let managed = slang::managed_helper(root, SLANG_VERSION);
        if managed.is_file() {
            return managed;
        }
    }
    gust::reflect::native_compiler_path()
}

/// `Ok` when the helper's Slang build tag is the version of the located `slangc`.
pub(crate) fn check_compatible(
    helper: &Path,
    helper_tag: &str,
    slangc: &Located,
) -> Result<(), String> {
    if helper_tag == slangc.version {
        Ok(())
    } else {
        Err(format!(
            "the helper {} was built against Slang `{helper_tag}`, but {} is Slang `{}`; {REBUILD_HINT}",
            helper.display(),
            slangc.slangc.display(),
            slangc.version
        ))
    }
}

/// A located Slang and a helper that matches it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Toolchain {
    pub(crate) helper: PathBuf,
    pub(crate) slangc: Located,
}

/// Locate Slang and the helper and refuse a helper of another build.
pub(crate) fn require(env: &Env, allow_mismatch: bool) -> Result<Toolchain, String> {
    let slangc = slang::locate(env, allow_mismatch, &slang::slangc_version)?;
    let candidate = helper_candidate(Some(&slangc));
    let helper = tools::locate_executable(&candidate, env.path.as_deref()).ok_or_else(|| {
        format!(
            "the native Slang helper was not found (looked for {}); {}",
            candidate.display(),
            tools::helper_hint()
        )
    })?;
    let tag = tools::helper_slang_version(&helper)?;
    check_compatible(&helper, &tag, &slangc)?;
    Ok(Toolchain { helper, slangc })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slang::Source;

    fn located(version: &str) -> Located {
        Located {
            slangc: PathBuf::from("/slang/bin/slangc"),
            version: version.to_owned(),
            source: Source::Cache,
            pinned: true,
            root: None,
        }
    }

    #[test]
    fn a_helper_is_compatible_only_when_its_tag_is_exactly_the_slangc_version() {
        let helper = Path::new("/h/gust-slang-reflect");
        assert!(check_compatible(helper, "2026.13.1", &located("2026.13.1")).is_ok());
        assert!(
            check_compatible(
                helper,
                "2026.13.1-1-g84792eb15",
                &located("2026.13.1-1-g84792eb15")
            )
            .is_ok()
        );
        // Both match the pin, but they are different builds: the helper is refused.
        let error =
            check_compatible(helper, "2026.13.1-1-g84792eb15", &located("2026.13.1")).unwrap_err();
        assert!(
            error.contains("built against Slang `2026.13.1-1-g84792eb15`")
                && error.contains("is Slang `2026.13.1`")
                && error.contains("cargo gust slang build-helper"),
            "{error}"
        );
        assert!(check_compatible(helper, "2025.1.0", &located("2026.13.1")).is_err());
        assert!(check_compatible(helper, "", &located("2026.13.1")).is_err());
    }
}
