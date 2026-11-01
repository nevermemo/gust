//! The build profile of this build (D44).
//!
//! `gust`'s build script reads `GUST_PROFILE` (`dev`, `native` or `web`), infers `native` (`web` on `wasm32`) for a
//! release build without it, and hands the result to this crate as `cfg(gust_profile = "..")`. [`NAME`] is the
//! effective profile and [`STRICT`] is true for `native` and `web`. Code the `#[gpu]` macro generates calls
//! [`require_artifact`], which makes a strict build fail at the kernel when no precompiled artifact is present.
//!
//! See the crate documentation for how to choose the profile and for the release rule.

#[cfg(gust_profile = "dev")]
pub const NAME: &str = "dev";
#[cfg(gust_profile = "native")]
pub const NAME: &str = "native";
#[cfg(gust_profile = "web")]
pub const NAME: &str = "web";

/// Whether the effective profile is strict (`native` or `web`): a kernel needs a precompiled artifact and the
/// `slang-runtime` feature is refused.
pub const STRICT: bool = cfg!(not(gust_profile = "dev"));

/// The text of the compile error that [`require_artifact`] raises.
const MISSING: &str = "gust: a strict profile (GUST_PROFILE native or web) needs a precompiled artifact for this kernel; run `cargo gust build`, or build with GUST_PROFILE=dev";

/// Fails const evaluation, at the `#[gpu]` kernel that called it, when the profile is strict and `present` is false.
///
/// The `#[gpu]` macro emits `const _: () = ::gust::profile::require_artifact(..)` for every kernel; `present` says
/// whether the kernel has a precompiled artifact. Under `dev` this does nothing.
pub const fn require_artifact(present: bool) {
    check_artifact(STRICT, present);
}

/// [`require_artifact`] with the strictness as an argument, so that a unit test can reach both outcomes.
const fn check_artifact(strict: bool, present: bool) {
    assert!(present || !strict, "{}", MISSING);
}

/// The text of the compile error that [`require_manifest`] raises.
const MISSING_MANIFEST: &str = "gust: a strict profile (GUST_PROFILE native or web) needs gust-artifacts/manifest.json in the kernel crate; run `cargo gust init` and `cargo gust build`, or build with GUST_PROFILE=dev";

/// Fails const evaluation, at the `#[gpu]` kernel that called it, when the profile is strict and `present` is false.
///
/// The `#[gpu]` macro emits `const _: () = ::gust::profile::require_manifest(..)` beside `require_artifact(true)` for a
/// kernel that has its artifact pair; `present` says whether the crate has `gust-artifacts/manifest.json`. The
/// manifest is what makes cargo re-expand the module when an artifact changes. Under `dev` this does nothing.
///
/// The macro cannot read the profile itself: the build script of this crate owns the effective profile and its
/// release inference, so the decision stays here, in the same place as [`require_artifact`].
pub const fn require_manifest(present: bool) {
    check_manifest(STRICT, present);
}

/// [`require_manifest`] with the strictness as an argument, so that a unit test can reach both outcomes.
const fn check_manifest(strict: bool, present: bool) {
    assert!(present || !strict, "{}", MISSING_MANIFEST);
}

#[cfg(test)]
mod tests {
    use super::{
        MISSING, MISSING_MANIFEST, NAME, STRICT, check_artifact, check_manifest, require_artifact,
        require_manifest,
    };

    // A satisfied assert is a build-time proof too: these items fail the compilation of the test build otherwise.
    const _: () = require_artifact(true);
    const _: () = check_artifact(false, false);
    const _: () = check_artifact(false, true);
    const _: () = check_artifact(true, true);
    const _: () = require_manifest(true);
    const _: () = check_manifest(false, false);
    const _: () = check_manifest(false, true);
    const _: () = check_manifest(true, true);

    #[test]
    fn the_name_is_one_of_the_three_profiles_and_strict_means_not_dev() {
        assert!(["dev", "native", "web"].contains(&NAME), "{NAME}");
        assert_eq!(STRICT, NAME != "dev");
    }

    #[test]
    fn a_strict_profile_without_an_artifact_panics_with_the_message() {
        let error = std::panic::catch_unwind(|| check_artifact(true, false)).unwrap_err();
        let text = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied())
            .unwrap_or_default()
            .to_owned();
        assert_eq!(text, MISSING);
        assert!(MISSING.starts_with("gust: a strict profile"));
        assert!(MISSING.ends_with("or build with GUST_PROFILE=dev"));
    }

    #[test]
    fn every_other_combination_passes() {
        check_artifact(true, true);
        check_artifact(false, true);
        check_artifact(false, false);
    }

    #[test]
    fn the_public_function_follows_the_effective_profile() {
        require_artifact(true);
        let outcome = std::panic::catch_unwind(|| require_artifact(false));
        assert_eq!(outcome.is_err(), STRICT);
    }

    #[test]
    fn a_strict_profile_without_a_manifest_panics_with_the_message() {
        let error = std::panic::catch_unwind(|| check_manifest(true, false)).unwrap_err();
        let text = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied())
            .unwrap_or_default()
            .to_owned();
        assert_eq!(text, MISSING_MANIFEST);
        assert!(MISSING_MANIFEST.starts_with("gust: a strict profile"));
        assert!(MISSING_MANIFEST.contains("run `cargo gust init` and `cargo gust build`"));
        assert!(MISSING_MANIFEST.ends_with("or build with GUST_PROFILE=dev"));
        assert_ne!(MISSING_MANIFEST, MISSING);
    }

    #[test]
    fn every_other_manifest_combination_passes() {
        check_manifest(true, true);
        check_manifest(false, true);
        check_manifest(false, false);
    }

    #[test]
    fn the_public_manifest_function_follows_the_effective_profile() {
        require_manifest(true);
        let outcome = std::panic::catch_unwind(|| require_manifest(false));
        assert_eq!(outcome.is_err(), STRICT);
    }
}

// The build script holds the profile decision and its unit tests; including it here is what runs them under
// `cargo test -p gust`. Its `main` is never called from the library.
#[cfg(test)]
#[allow(dead_code)]
#[path = "../build.rs"]
mod build_script;
