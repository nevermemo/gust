//! The toolchain a build needs: the pinned `slangc` and a native helper that was compiled against the same Slang.
//!
//! `build` and `check --recompile` call [`require`]; `doctor` gathers the same facts without stopping at the first
//! problem. Where things are looked for is `gust::toolchain`'s business (D109): this module asks it, so the run-time
//! bridge and this tool find the same binaries. The helper compatibility rule (R4.8): the helper reports its reflection
//! schema and the Slang build tag of the library it loads (`gust-slang-reflect --version`). The schema must be the one
//! this tool needs (3, see [`tools::REQUIRED_HELPER_SCHEMA`]) and the tag must equal what the located `slangc -version`
//! prints. Both come from one Slang build, so the tag comparison is exact, not the pin rule of [`slang::matches_pin`].

use std::path::{Path, PathBuf};

use gust::toolchain as rules;

use crate::{
    slang::{self, Env, Located},
    tools,
};

/// What to do about a helper that does not match; part of every refusal.
pub(crate) const REBUILD_HINT: &str = "run `cargo gust slang build-helper` (add `--sdk-root <dir>` for a Slang that cargo gust did not install), or point GUST_SLANG_REFLECT at a helper built against that Slang";

/// The helper to look for, by the rules of `gust::toolchain`: `GUST_SLANG_REFLECT` when set, else the helper that
/// `build-helper` wrote under the Slang cache root for the pin, else the repository checkout's `target/slang-reflect`,
/// else the bare name. The managed helper is used whenever a cache root can be named, whichever `slangc` was located
/// (`cargo-gust` used to read it only beside a managed `slangc`).
///
/// An unusable `GUST_SLANG_REFLECT` or `GUST_SLANG_HOME` gives the variable's value as given (the lookup then reports
/// that path as missing); [`require`] is the call that refuses it with the reason. The rule is
/// `gust::toolchain::helper_candidate_in`, the one the run-time bridge uses (`reflect::native_compiler_path`).
pub(crate) fn helper_candidate() -> PathBuf {
    rules::helper_candidate_in(&Env::from_process())
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

/// Locate Slang and the helper and refuse a helper of another schema or build.
pub(crate) fn require(env: &Env, allow_mismatch: bool) -> Result<Toolchain, String> {
    require_with(
        env,
        allow_mismatch,
        &slang::slangc_version,
        &tools::helper_version_output,
    )
}

/// [`require`] with `slangc -version` and `<helper> --version` given, so a test supplies both answers.
fn require_with(
    env: &Env,
    allow_mismatch: bool,
    slangc_version: &dyn Fn(&Path) -> Result<String, String>,
    helper_version_output: &dyn Fn(&Path) -> Result<String, String>,
) -> Result<Toolchain, String> {
    let slangc = slang::locate(env, allow_mismatch, slangc_version)?;
    // An unusable variable is an error here, not a path to look for.
    let candidate = rules::helper_path_with(env)?.unwrap_or_else(rules::bare_helper_name);
    let helper = tools::locate_executable(&candidate, env.path.as_deref()).ok_or_else(|| {
        format!(
            "the native Slang helper was not found (looked for {}); {}",
            candidate.display(),
            tools::helper_hint()
        )
    })?;
    // Schema 3 first: a helper of schema 1 or 2 cannot take the stage argument (mx.k), so it is refused by its schema
    // before its tag is compared.
    let tag = tools::parse_helper_version(&helper_version_output(&helper)?)?;
    check_compatible(&helper, &tag, &slangc)?;
    Ok(Toolchain { helper, slangc })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{slang::Source, store::tests::Scratch};
    use std::{ffi::OsString, fs};

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

    /// A cache root holding a (fake) pinned `slangc` and a managed helper; `named_helper` adds a helper file elsewhere.
    struct Setup {
        scratch: Scratch,
        env: Env,
    }

    impl Setup {
        fn new(name: &str) -> Self {
            let scratch = Scratch::new(name);
            let home = scratch.0.join("home");
            let slangc = rules::managed_slangc(&home);
            fs::create_dir_all(slangc.parent().unwrap()).unwrap();
            fs::write(&slangc, "x").unwrap();
            let helper = rules::managed_helper(&home);
            fs::create_dir_all(helper.parent().unwrap()).unwrap();
            fs::write(&helper, "x").unwrap();
            let env = Env {
                slang_home: Some(home.into_os_string()),
                ..Env::default()
            };
            Self { scratch, env }
        }

        fn named_helper(&mut self) -> PathBuf {
            let helper = self.scratch.0.join("elsewhere").join("helper-exe");
            fs::create_dir_all(helper.parent().unwrap()).unwrap();
            fs::write(&helper, "x").unwrap();
            self.env.helper = Some(helper.clone().into_os_string());
            helper
        }
    }

    fn pinned(_: &Path) -> Result<String, String> {
        Ok(rules::SLANG_PIN.to_owned())
    }

    #[test]
    fn the_helper_is_the_named_one_then_the_managed_one_whatever_slangc_was_found() {
        let mut setup = Setup::new("helper-order");
        let home = PathBuf::from(setup.env.slang_home.clone().unwrap());
        // The managed helper beside the pinned install.
        assert_eq!(
            rules::helper_candidate_in(&setup.env),
            rules::managed_helper(&home)
        );
        // The managed helper is used even when the slangc is not a managed one: cargo-gust used to look for it
        // only beside a managed slangc (mr.t1 review S4). Here the cache has the helper and no slangc.
        fs::remove_file(rules::managed_slangc(&home)).unwrap();
        assert_eq!(
            rules::helper_candidate_in(&setup.env),
            rules::managed_helper(&home)
        );
        // GUST_SLANG_REFLECT beats it.
        let named = setup.named_helper();
        assert_eq!(rules::helper_candidate_in(&setup.env), named);
    }

    #[test]
    fn an_unusable_helper_variable_is_refused_by_require_and_reported_as_given_elsewhere() {
        let mut setup = Setup::new("helper-relative");
        for value in ["gust-slang-reflect", "./helper", "rel/helper"] {
            setup.env.helper = Some(OsString::from(value));
            let error = require_with(&setup.env, false, &pinned, &|_| {
                Ok("gust-slang-reflect schema=3 slang=2026.21".to_owned())
            })
            .unwrap_err();
            assert!(
                error.contains("GUST_SLANG_REFLECT must be an absolute path")
                    && error.contains(value),
                "{value}: {error}"
            );
            assert_eq!(rules::helper_candidate_in(&setup.env), PathBuf::from(value));
        }
        setup.env.helper = Some(OsString::new());
        let error = require_with(&setup.env, false, &pinned, &|_| unreachable!()).unwrap_err();
        assert!(error.contains("set but empty"), "{error}");
    }

    #[test]
    fn require_refuses_a_helper_below_schema_3_and_one_of_another_build() {
        let mut setup = Setup::new("require-schema");
        let named = setup.named_helper();
        let output = |text: &'static str| move |_: &Path| Ok(text.to_owned());
        // Schema 3 and the pin's tag: accepted, and it is the named helper that is returned.
        let accepted = require_with(
            &setup.env,
            false,
            &pinned,
            &output("gust-slang-reflect schema=3 slang=2026.21\n"),
        )
        .unwrap();
        assert_eq!(accepted.helper, named);
        // Plumbing only: the fixture reports the pin, so this checks that the version is carried through.
        assert_eq!(accepted.slangc.version, crate::slang::SLANG_VERSION);
        // A helper of schema 1 or 2 cannot take the stage argument: refused by its schema, with the rebuild command,
        // although its tag is the right one (mx.k review S1).
        for text in [
            "gust-slang-reflect schema=1 slang=2026.21\n",
            "gust-slang-reflect schema=2 slang=2026.21\n",
        ] {
            let error = require_with(&setup.env, false, &pinned, &output(text)).unwrap_err();
            assert!(
                error.contains("needs schema 3")
                    && error.contains("cargo gust slang build-helper --force"),
                "{text}: {error}"
            );
        }
        // Schema 3 of another Slang build: refused by the tag.
        let error = require_with(
            &setup.env,
            false,
            &pinned,
            &output("gust-slang-reflect schema=3 slang=2025.1.0\n"),
        )
        .unwrap_err();
        assert!(error.contains("built against Slang `2025.1.0`"), "{error}");
        // A helper that cannot run is reported with its reason.
        let error = require_with(&setup.env, false, &pinned, &|_| {
            Err("cannot run the helper".to_owned())
        })
        .unwrap_err();
        assert_eq!(error, "cannot run the helper");
    }
}
