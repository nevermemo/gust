//! `gust::toolchain` (D109, slice mr.t1): the one owner of the toolchain-location rules.
//!
//! The module is `gust::toolchain` in `lib.rs` (mr.t2, behind the `slang-runtime` feature), and this test compiles the
//! source file itself, so it runs with or without that feature. The file is std-only and has no `crate::` path, so it
//! builds standalone here. Its items are `pub`, which a private module of a test crate reports as unused where this
//! file does not call them, hence the `allow`.
//!
//! The rules are tested on a hand-built [`Env`], so no case reads or writes the process environment and none can race
//! another. The one place that reads the real environment, `Env::from_process`, is tested in child processes whose
//! environment this test sets for the child alone.
//!
//! The root table is the one place that tests the cache-root cases: mr.t1 copied them from `cargo-gust`'s `slang.rs`,
//! and since mr.t2 `cargo-gust` calls this module and keeps no copy of the rules.

#[allow(dead_code)]
#[path = "../src/toolchain.rs"]
mod toolchain;

use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use toolchain::{
    Env, HELPER_VARIABLE, HOME_VARIABLE, SLANG_PIN, VALIDATED_BUILDS, VersionError, cache_root,
    cache_root_in, check_slangc_version, check_slangc_version_in, checkout_helper, executable_name,
    find_on_path, helper_path_in, lookup_root_in, managed_helper, managed_slangc, matches_pin,
    parse_version_output, slangc_path_in,
};

/// An absolute path that exists on no machine: `C:\gust-synthetic\<name>` on Windows, `/gust-synthetic/<name>`
/// elsewhere (`/x` is not absolute on Windows).
fn synthetic(name: &str) -> PathBuf {
    let base = if cfg!(windows) {
        PathBuf::from(r"C:\gust-synthetic")
    } else {
        PathBuf::from("/gust-synthetic")
    };
    base.join(name)
}

fn os(path: &Path) -> Option<OsString> {
    Some(path.as_os_str().to_owned())
}

fn env_with(
    slang_home: Option<&Path>,
    cargo_home: Option<&Path>,
    home: Option<&Path>,
    path: Option<&OsString>,
) -> Env {
    Env {
        slang_home: slang_home.and_then(os),
        cargo_home: cargo_home.and_then(os),
        home: home.and_then(os),
        path: path.cloned(),
        helper: None,
    }
}

/// A scratch directory under the system temp directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let serial = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!(
            "gust-toolchain-test-{}-{serial}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A file whose content is `content`, with its directories.
fn touch(path: &Path, content: &str) -> PathBuf {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
    path.to_path_buf()
}

fn join(dirs: &[&Path]) -> OsString {
    env::join_paths(dirs).unwrap()
}

/// What a root case expects: the root, or an error that contains the text.
#[derive(Debug)]
enum Root {
    Is(PathBuf),
    Error(&'static str),
}

struct RootCase {
    /// The line range of `slang_tests.rs` the row copies.
    source: &'static str,
    name: &'static str,
    env: Env,
    /// `cache_root_in`.
    install: Root,
    /// `lookup_root_in`: `Ok(None)` where nobody can name a root, the same error otherwise.
    lookup: Result<Option<PathBuf>, &'static str>,
}

fn root_cases() -> Vec<RootCase> {
    let home = synthetic("slang-home");
    let cargo = synthetic("cargo");
    let user = synthetic("user");
    let default_of_cargo = cargo.join("gust").join("slang");
    let default_of_user = user.join(".cargo").join("gust").join("slang");
    let empty = OsString::new();
    let with =
        |slang_home: Option<OsString>, cargo_home: Option<OsString>, home: Option<OsString>| Env {
            slang_home,
            cargo_home,
            home,
            path: None,
            helper: None,
        };
    vec![
        RootCase {
            source: "188-194",
            name: "GUST_SLANG_HOME beats CARGO_HOME",
            env: with(os(&home), os(&cargo), None),
            install: Root::Is(home.clone()),
            lookup: Ok(Some(home.clone())),
        },
        RootCase {
            source: "195-199",
            name: "CARGO_HOME/gust/slang without GUST_SLANG_HOME",
            env: with(None, os(&cargo), None),
            install: Root::Is(default_of_cargo.clone()),
            lookup: Ok(Some(default_of_cargo)),
        },
        RootCase {
            source: "200-205",
            name: "~/.cargo/gust/slang without either",
            env: with(None, None, os(&user)),
            install: Root::Is(default_of_user.clone()),
            lookup: Ok(Some(default_of_user.clone())),
        },
        RootCase {
            source: "206-211",
            name: "an empty CARGO_HOME counts as unset",
            env: with(None, Some(empty.clone()), os(&user)),
            install: Root::Is(default_of_user.clone()),
            lookup: Ok(Some(default_of_user)),
        },
        RootCase {
            source: "212-213",
            name: "nothing known: an install has nowhere to go",
            env: with(None, Some(empty.clone()), None),
            install: Root::Error("GUST_SLANG_HOME"),
            lookup: Ok(None),
        },
        RootCase {
            source: "214-217",
            name: "a relative GUST_SLANG_HOME is refused, not resolved against the current directory",
            env: with(
                Some(OsString::from("relative/dir")),
                Some(empty.clone()),
                None,
            ),
            install: Root::Error("absolute"),
            lookup: Err("absolute"),
        },
        RootCase {
            source: "218-219",
            name: "an empty GUST_SLANG_HOME is refused",
            env: with(Some(empty.clone()), Some(empty), None),
            install: Root::Error("empty"),
            lookup: Err("empty"),
        },
        RootCase {
            source: "220-224",
            name: "a relative CARGO_HOME is not used",
            env: with(None, Some(OsString::from("rel")), None),
            install: Root::Error("GUST_SLANG_HOME"),
            lookup: Ok(None),
        },
        RootCase {
            // The same rule from the code (`default_root`): a relative CARGO_HOME is not replaced by `~/.cargo`.
            source: "195-199 (code, not a row of the original table)",
            name: "a relative CARGO_HOME is not replaced by the home directory either",
            env: with(None, Some(OsString::from("rel")), os(&user)),
            install: Root::Error("GUST_SLANG_HOME"),
            lookup: Ok(None),
        },
    ]
}

#[test]
fn the_cache_root_table_equals_the_cargo_gust_rules_case_by_case() {
    for case in root_cases() {
        let label = format!("slang_tests.rs:{} {}", case.source, case.name);
        match (&case.install, cache_root_in(&case.env)) {
            (Root::Is(want), Ok(got)) => assert_eq!(&got, want, "{label}"),
            (Root::Error(text), Err(error)) => assert!(error.contains(text), "{label}: {error}"),
            (want, got) => panic!("{label}: wanted {want:?}, got {got:?}"),
        }
        match (&case.lookup, lookup_root_in(&case.env)) {
            (Ok(want), Ok(got)) => assert_eq!(&got, want, "{label} (lookup)"),
            (Err(text), Err(error)) => assert!(error.contains(text), "{label} (lookup): {error}"),
            (want, got) => panic!("{label} (lookup): wanted {want:?}, got {got:?}"),
        }
    }
}

#[test]
fn the_pin_and_the_managed_paths_are_keyed_by_it() {
    // slang_tests.rs:228-242, with the pin fixed (the copy took the version as an argument).
    assert_eq!(SLANG_PIN, "2026.22");
    let root = synthetic("r");
    assert_eq!(
        managed_slangc(&root),
        root.join("2026.22")
            .join("bin")
            .join(executable_name("slangc"))
    );
    assert_eq!(
        managed_helper(&root),
        root.join("helper")
            .join("2026.22")
            .join(executable_name("gust-slang-reflect"))
    );
    assert_eq!(
        executable_name("x"),
        format!("x{}", std::env::consts::EXE_SUFFIX)
    );
}

#[test]
fn slangc_prefers_gust_slang_home_then_the_cache_then_path() {
    // slang_tests.rs:245-293, selection only: no version is judged by `slangc_path_in`.
    let scratch = Scratch::new("slangc-order");
    let home = scratch.0.join("home");
    let cargo = scratch.0.join("cargo");
    let cache = cargo.join("gust").join("slang");
    let on_path = scratch.0.join("path");
    let path = join(&[&on_path]);
    let path_slangc = touch(&on_path.join(executable_name("slangc")), "");
    let with_home = env_with(Some(&home), Some(&cargo), None, Some(&path));
    let without_home = env_with(None, Some(&cargo), None, Some(&path));

    // Only PATH has it (:256-260).
    assert_eq!(slangc_path_in(&without_home), Ok(Some(path_slangc.clone())));

    // The cache beats PATH (:262-267).
    let cached = touch(&managed_slangc(&cache), "");
    assert_eq!(slangc_path_in(&without_home), Ok(Some(cached)));

    // GUST_SLANG_HOME replaces the cache: a home without the pin goes to PATH, not to the cache (:269-271).
    assert_eq!(slangc_path_in(&with_home), Ok(Some(path_slangc.clone())));
    // A home with the pin beats both (:272-277).
    let homed = touch(&managed_slangc(&home), "");
    assert_eq!(slangc_path_in(&with_home), Ok(Some(homed)));

    // A home with no pin directory does not fall through to the cache, it falls through to PATH (:279-286) ...
    let other_home = scratch.0.join("empty-home");
    fs::create_dir_all(&other_home).unwrap();
    let env = env_with(Some(&other_home), Some(&cargo), None, Some(&path));
    assert_eq!(slangc_path_in(&env), Ok(Some(path_slangc)));
    // ... and with nothing on PATH either, the cache that exists is still not used (:287-292).
    let nowhere = scratch.0.join("nowhere");
    fs::create_dir_all(&nowhere).unwrap();
    let env = env_with(
        Some(&other_home),
        Some(&cargo),
        None,
        Some(&join(&[&nowhere])),
    );
    assert_eq!(slangc_path_in(&env), Ok(None));

    // An unusable GUST_SLANG_HOME is an error, never another root.
    let mut env = env_with(None, Some(&cargo), None, Some(&path));
    env.slang_home = Some(OsString::from("relative/dir"));
    assert!(slangc_path_in(&env).unwrap_err().contains("absolute"));
    // No root can be named: PATH alone.
    let env = env_with(None, None, None, Some(&join(&[&nowhere])));
    assert_eq!(slangc_path_in(&env), Ok(None));
}

#[test]
fn path_lookup_never_uses_the_current_directory() {
    let scratch = Scratch::new("path-lookup");
    let absolute = scratch.0.join("bin");
    let tool = touch(&absolute.join("tool"), "");
    // The test runs in the package directory, which holds a `Cargo.toml`: any lookup relative to the current
    // directory would find it through `.` or an empty entry.
    assert!(Path::new("Cargo.toml").is_file());
    let relative = join(&[Path::new("."), Path::new("")]);
    assert_eq!(find_on_path("Cargo.toml", Some(&relative)), None);
    let mixed = join(&[
        Path::new("."),
        Path::new(""),
        Path::new("rel/bin"),
        &absolute,
    ]);
    assert_eq!(find_on_path("tool", Some(&mixed)), Some(tool));
    assert_eq!(find_on_path("Cargo.toml", Some(&mixed)), None);
    assert_eq!(find_on_path("tool", None), None);
    // A directory of that name is not a file.
    fs::create_dir_all(absolute.join("dir")).unwrap();
    assert_eq!(find_on_path("dir", Some(&join(&[&absolute]))), None);
}

/// A repository checkout in a scratch directory: `<top>/crates/gust` (the "manifest directory"), the helper under
/// `<top>/target/slang-reflect`, and, when `marker` is set, `<top>/xtask/Cargo.toml`.
struct Tree {
    scratch: Scratch,
    manifest: PathBuf,
    helper: PathBuf,
}

fn tree(name: &str, marker: bool, helper: bool) -> Tree {
    let scratch = Scratch::new(name);
    let top = scratch.0.clone();
    let manifest = top.join("crates").join("gust");
    fs::create_dir_all(&manifest).unwrap();
    let built = top
        .join("target")
        .join("slang-reflect")
        .join(executable_name("gust-slang-reflect"));
    if helper {
        touch(&built, "");
    }
    if marker {
        touch(&top.join("xtask").join("Cargo.toml"), "");
    }
    // What `checkout_helper` returns, spelled as the compile-time manifest directory leads to it.
    let helper = manifest
        .join("../../target/slang-reflect")
        .join(executable_name("gust-slang-reflect"));
    Tree {
        scratch,
        manifest,
        helper,
    }
}

#[test]
fn the_checkout_helper_needs_the_xtask_marker_beside_the_manifest_directory() {
    // A checkout: the marker and the helper are there.
    let checkout = tree("checkout", true, true);
    assert_eq!(
        checkout_helper(&checkout.manifest),
        Some(checkout.helper.clone())
    );
    // A registry build has no `xtask/Cargo.toml` two levels up: it never returns a checkout path, even when a helper
    // sits where a checkout would keep it.
    let registry = tree("registry", false, true);
    assert!(
        registry
            .scratch
            .0
            .join("target")
            .join("slang-reflect")
            .join(executable_name("gust-slang-reflect"))
            .is_file()
    );
    assert_eq!(checkout_helper(&registry.manifest), None);
    // The marker alone is not enough: the helper must have been built.
    let unbuilt = tree("unbuilt", true, false);
    assert_eq!(checkout_helper(&unbuilt.manifest), None);
    // The marker is a file: a directory of that name does not count.
    let directory = tree("marker-directory", false, true);
    fs::create_dir_all(directory.scratch.0.join("xtask").join("Cargo.toml")).unwrap();
    assert_eq!(checkout_helper(&directory.manifest), None);
}

/// A manifest directory somewhere that is not `<top>/crates/gust`, with a foreign `xtask/Cargo.toml` and a helper
/// at `<top>` where a checkout would keep them: `<top>` is `<manifest>/../..`.
fn foreign_tree(name: &str, manifest_below_top: &[&str]) -> (Scratch, PathBuf) {
    let scratch = Scratch::new(name);
    let top = scratch.0.clone();
    touch(&top.join("xtask").join("Cargo.toml"), "");
    touch(
        &top.join("target")
            .join("slang-reflect")
            .join(executable_name("gust-slang-reflect")),
        "",
    );
    let mut manifest = top;
    for part in manifest_below_top {
        manifest.push(part);
    }
    fs::create_dir_all(&manifest).unwrap();
    (scratch, manifest)
}

#[test]
fn the_checkout_helper_also_needs_the_manifest_directory_to_be_crates_gust_of_that_tree() {
    // `cargo vendor`: `<project>/vendor/gust`, in a project with its own `xtask` and `target`.
    let (_project, vendored) = foreign_tree("vendor", &["vendor", "gust"]);
    assert_eq!(checkout_helper(&vendored), None);
    // An unpacked crate in a work directory inside a repository checkout (`<repo>/w/gust-0.1.0`), which also has a
    // real `crates/gust`: the marker and the helper are the repository's, the manifest directory is not.
    let (repo, unpacked) = foreign_tree("nested", &["w", "gust-0.1.0"]);
    fs::create_dir_all(repo.0.join("crates").join("gust")).unwrap();
    assert_eq!(checkout_helper(&unpacked), None);
    // The real layout still gives the path.
    let checkout = tree("real-layout", true, true);
    assert_eq!(
        checkout_helper(&checkout.manifest),
        Some(checkout.helper.clone())
    );
    // The same through the whole rule, with nothing before it to answer.
    let env = env_with(None, None, None, None);
    assert_eq!(helper_path_in(&env, &vendored), Ok(None));
    assert_eq!(helper_path_in(&env, &unpacked), Ok(None));
    assert_eq!(
        helper_path_in(&env, &checkout.manifest),
        Ok(Some(checkout.helper.clone()))
    );
}

#[test]
fn the_helper_order_is_the_variable_then_the_managed_cache_then_the_checkout() {
    let checkout = tree("helper-order", true, true);
    let root = checkout.scratch.0.join("cache");
    let nothing = join(&[Path::new("")]);
    let mut env = env_with(Some(&root), None, None, Some(&nothing));
    let ask = |env: &Env| helper_path_in(env, &checkout.manifest);

    // Only the checkout has it.
    assert_eq!(ask(&env), Ok(Some(checkout.helper.clone())));
    // The managed helper of the pin beats the checkout.
    let managed = touch(&managed_helper(&root), "");
    assert_eq!(ask(&env), Ok(Some(managed.clone())));
    // A helper for another version is not the pin's.
    let other = root
        .join("helper")
        .join("2025.1.0")
        .join(executable_name("gust-slang-reflect"));
    touch(&other, "");
    assert_eq!(ask(&env), Ok(Some(managed.clone())));
    // GUST_SLANG_REFLECT beats both, and is returned as given, whether or not the file exists.
    let explicit = synthetic("explicit").join("helper");
    env.helper = os(&explicit);
    assert_eq!(ask(&env), Ok(Some(explicit.clone())));
    // An empty GUST_SLANG_REFLECT is refused, not treated as unset.
    env.helper = Some(OsString::new());
    assert!(ask(&env).unwrap_err().contains(HELPER_VARIABLE));
    // A relative value is refused, like a relative GUST_SLANG_HOME: it would be looked up in the current directory.
    for relative in [
        "given",
        "gust-slang-reflect",
        "./gust-slang-reflect",
        "rel/helper",
    ] {
        env.helper = Some(OsString::from(relative));
        let error = ask(&env).unwrap_err();
        assert!(
            error.contains(HELPER_VARIABLE)
                && error.contains("absolute")
                && error.contains(relative),
            "{relative}: {error}"
        );
    }
    // GUST_SLANG_REFLECT is read before the root, so an unusable GUST_SLANG_HOME does not hide it ...
    env.helper = os(&explicit);
    env.slang_home = Some(OsString::from("relative/dir"));
    assert_eq!(ask(&env), Ok(Some(explicit.clone())));
    // ... but without it that home is an error, not a fall-through to the checkout.
    env.helper = None;
    assert!(ask(&env).unwrap_err().contains(HOME_VARIABLE));
}

#[test]
fn a_registry_build_finds_no_helper_in_a_checkout_path() {
    // The same layout as the checkout above but without `xtask/Cargo.toml`: the whole rule returns nothing, not the
    // checkout's helper, with no variable and no managed helper to answer first.
    let registry = tree("registry-order", false, true);
    let env = env_with(
        Some(&registry.scratch.0.join("cache")),
        None,
        None,
        Some(&join(&[Path::new("")])),
    );
    assert_eq!(helper_path_in(&env, &registry.manifest), Ok(None));
    // With no root known at all the answer is the same.
    let env = env_with(None, None, None, None);
    assert_eq!(helper_path_in(&env, &registry.manifest), Ok(None));
}

#[test]
fn the_version_output_table_of_cargo_gust() {
    // slang_tests.rs:172-185.
    assert_eq!(parse_version_output("2026.22\n").unwrap(), "2026.22");
    assert_eq!(
        parse_version_output("\r\n 2026.22-1-g0a1b2c3d4 \r\n").unwrap(),
        "2026.22-1-g0a1b2c3d4"
    );
    for bad in ["", "\n", "error: no input files", "slangc 2026.22", "v1"] {
        let error = parse_version_output(bad).unwrap_err();
        assert!(
            matches!(error, VersionError::Output(_)),
            "{bad:?}: {error:?}"
        );
        assert!(
            error.to_string().contains("unexpected `-version` output"),
            "{bad:?}: {error}"
        );
    }
}

/// A `git describe` tag that no Slang prints, listed as a validated build only by the tests of the mechanism.
const SYNTHETIC_BUILD: &str = "2026.22-1-g0a1b2c3d4";

#[test]
fn the_pin_rule_is_exactly_the_one_of_cargo_gust() {
    // slang_tests.rs:104-169 (`matches_pin`), with the pin fixed.
    let validated = [SYNTHETIC_BUILD];
    for reported in [SLANG_PIN, SYNTHETIC_BUILD] {
        assert!(matches_pin(reported, &validated), "{reported}");
    }
    assert!(matches_pin(SLANG_PIN, &[]));
    assert!(!matches_pin(SYNTHETIC_BUILD, &[]));
    assert!(!matches_pin(SYNTHETIC_BUILD, VALIDATED_BUILDS));
    for reported in [
        "",
        "2026.2",
        "2026.220",
        "2026.23",
        "2026.22-",
        "2026.22-rc1",
        "2026.22-1",
        "2026.22-g0a1b2c3d4",
        "2026.22-1-g0a1b2c3d4-dirty",
        "2026.22-2-gdeadbee",
        "2026.22-1-g0a1b2c3d5",
        "2026.22 ",
        "v2026.22",
        "2026.19-1-g0a1b2c3d4",
        "12026.22",
    ] {
        assert!(!matches_pin(reported, &validated), "{reported:?}");
    }
    // A listed tag that does not extend the pin is not the pin.
    assert!(!matches_pin(
        "2026.23-1-g0a1b2c3d4",
        &["2026.23-1-g0a1b2c3d4"]
    ));
}

/// Serialises the tests that start a process: a script written just before it runs can otherwise be inherited, still
/// open for writing, by a child another thread forks (`ETXTBSY` on Linux).
static SPAWN: Mutex<()> = Mutex::new(());

/// A fake `slangc` whose `-version` prints `stdout` and `stderr` (single words, no quotes) and exits with `code`.
/// The scripts live under the temp directory and are executed there, so a `noexec` temp directory (a hardened `/tmp`
/// or `TMPDIR`) makes the tests that run them fail with a permission error, not the code under test.
fn fake_slangc(dir: &Path, stdout: &str, stderr: &str, code: i32) -> PathBuf {
    fs::create_dir_all(dir).unwrap();
    let mut script = String::new();
    let path;
    if cfg!(windows) {
        path = dir.join("slangc.cmd");
        script.push_str("@echo off\r\n");
        if !stdout.is_empty() {
            script.push_str(&format!("echo {stdout}\r\n"));
        }
        if !stderr.is_empty() {
            script.push_str(&format!("echo {stderr} 1>&2\r\n"));
        }
        script.push_str(&format!("exit /b {code}\r\n"));
    } else {
        path = dir.join("slangc");
        script.push_str("#!/bin/sh\n");
        if !stdout.is_empty() {
            script.push_str(&format!("echo '{stdout}'\n"));
        }
        if !stderr.is_empty() {
            script.push_str(&format!("echo '{stderr}' 1>&2\n"));
        }
        script.push_str(&format!("exit {code}\n"));
    }
    fs::write(&path, script).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    }
    path
}

#[test]
fn the_version_check_accepts_exactly_the_pin_and_names_both_otherwise() {
    let _spawning = SPAWN
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let scratch = Scratch::new("version-check");
    let case = |name: &str, stdout: &str, stderr: &str, code: i32| {
        check_slangc_version(&fake_slangc(&scratch.0.join(name), stdout, stderr, code))
    };

    assert_eq!(case("pin", "2026.22", "", 0), Ok(()));
    // The version on stderr when stdout is empty.
    assert_eq!(case("stderr", "", "2026.22", 0), Ok(()));

    // Anything but the exact pin is a mismatch that names the executable, what it printed and the pin.
    for found in [
        "2025.1.0",
        "2026.19",
        "2026.220",
        "2026.2",
        "2026.22-1-g0a1b2c3d4",
        "2026.22-rc1",
    ] {
        let slangc = fake_slangc(&scratch.0.join(format!("other-{found}")), found, "", 0);
        let error = check_slangc_version(&slangc).unwrap_err();
        assert_eq!(
            error,
            VersionError::Mismatch {
                slangc: slangc.clone(),
                found: found.to_owned()
            }
        );
        let text = error.to_string();
        assert!(text.contains(&slangc.display().to_string()), "{text}");
        assert!(text.contains(&format!("`{found}`")), "{text}");
        assert!(text.contains("not the pinned 2026.22"), "{text}");
        assert!(text.contains("cargo gust slang install"), "{text}");
    }

    // The listed-build mechanism, with a synthetic list (the shipped one is empty): a listed tag that extends the pin
    // passes; without the listing, with the shipped list, or with another tag listed, it does not.
    assert!(VALIDATED_BUILDS.is_empty(), "{VALIDATED_BUILDS:?}");
    let build = fake_slangc(&scratch.0.join("listed"), SYNTHETIC_BUILD, "", 0);
    assert_eq!(check_slangc_version_in(&build, &[SYNTHETIC_BUILD]), Ok(()));
    assert!(check_slangc_version_in(&build, &[]).is_err());
    assert!(check_slangc_version(&build).is_err());
    assert!(check_slangc_version_in(&build, &["2026.22-2-gdeadbee"]).is_err());

    // Output that is not a version, a failing run and a missing executable are their own errors.
    let error = case("garbage", "error:", "", 0).unwrap_err();
    assert!(
        matches!(&error, VersionError::Output(text) if text.contains("unexpected `-version` output")),
        "{error:?}"
    );
    let error = case("failing", "", "boom", 3).unwrap_err();
    assert!(
        matches!(&error, VersionError::Run(text) if text.contains("-version` failed") && text.contains("boom")),
        "{error:?}"
    );
    let missing = scratch.0.join("missing").join(executable_name("slangc"));
    let error = check_slangc_version(&missing).unwrap_err();
    assert!(
        matches!(&error, VersionError::Run(text) if text.contains("cannot run")),
        "{error:?}"
    );
}

/// The probe the next test starts as a child: it prints what the process environment holds.
#[test]
#[ignore = "a child of the process-environment test, not a test of its own"]
fn child_prints_the_process_environment() {
    println!("CHILD-ENV {:?}", Env::from_process());
    println!("CHILD-ROOT {:?}", cache_root());
}

/// Run the probe in a child whose five variables, `CARGO_HOME` and the home variables are exactly `set`.
fn child_output(set: &[(&str, &OsString)]) -> String {
    let _spawning = SPAWN
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut command = Command::new(env::current_exe().unwrap());
    command.args([
        "child_prints_the_process_environment",
        "--exact",
        "--ignored",
        "--nocapture",
        "--test-threads=1",
    ]);
    for name in [
        HOME_VARIABLE,
        HELPER_VARIABLE,
        "CARGO_HOME",
        "HOME",
        "USERPROFILE",
        "PATH",
    ] {
        command.env_remove(name);
    }
    for (name, value) in set {
        command.env(name, value);
    }
    let output = command.output().unwrap();
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stdout).unwrap()
}

fn child_lines(output: &str) -> (String, String) {
    let find = |prefix: &str| {
        output
            .lines()
            .find_map(|line| line.split_once(prefix).map(|(_, rest)| rest))
            .unwrap_or_else(|| panic!("no `{prefix}` line in {output}"))
            .to_owned()
    };
    (find("CHILD-ENV "), find("CHILD-ROOT "))
}

#[test]
fn from_process_reads_the_five_variables_and_the_right_home() {
    let slang_home = synthetic("home-var").into_os_string();
    let cargo_home = synthetic("cargo-var").into_os_string();
    let userprofile = synthetic("userprofile").into_os_string();
    let home = synthetic("home").into_os_string();
    let path = synthetic("bin").into_os_string();
    let helper = synthetic("helper").into_os_string();

    // Everything set: GUST_SLANG_HOME wins the root; the home is USERPROFILE on Windows and HOME elsewhere.
    let (env_line, root_line) = child_lines(&child_output(&[
        (HOME_VARIABLE, &slang_home),
        (HELPER_VARIABLE, &helper),
        ("CARGO_HOME", &cargo_home),
        ("USERPROFILE", &userprofile),
        ("HOME", &home),
        ("PATH", &path),
    ]));
    let expected = Env {
        slang_home: Some(slang_home.clone()),
        cargo_home: Some(cargo_home.clone()),
        home: Some(if cfg!(windows) {
            userprofile.clone()
        } else {
            home.clone()
        }),
        path: Some(path.clone()),
        helper: Some(helper),
    };
    assert_eq!(env_line, format!("{expected:?}"));
    assert_eq!(
        root_line,
        format!("{:?}", Ok::<_, String>(PathBuf::from(&slang_home)))
    );

    // Windows falls back to HOME without USERPROFILE; the root is then CARGO_HOME's.
    let (env_line, root_line) = child_lines(&child_output(&[
        ("CARGO_HOME", &cargo_home),
        ("HOME", &home),
    ]));
    let expected = Env {
        cargo_home: Some(cargo_home.clone()),
        home: Some(home.clone()),
        ..Env::default()
    };
    assert_eq!(env_line, format!("{expected:?}"));
    let cargo_root = PathBuf::from(&cargo_home).join("gust").join("slang");
    assert_eq!(root_line, format!("{:?}", Ok::<_, String>(cargo_root)));

    // Only a home: `<home>/.cargo/gust/slang`.
    let (env_line, root_line) =
        child_lines(&child_output(&[("USERPROFILE", &home), ("HOME", &home)]));
    let expected = Env {
        home: Some(home.clone()),
        ..Env::default()
    };
    assert_eq!(env_line, format!("{expected:?}"));
    let home_root = PathBuf::from(&home)
        .join(".cargo")
        .join("gust")
        .join("slang");
    assert_eq!(root_line, format!("{:?}", Ok::<_, String>(home_root)));

    // Nothing set: every field is `None` and there is no root.
    let (env_line, root_line) = child_lines(&child_output(&[]));
    assert_eq!(env_line, format!("{:?}", Env::default()));
    assert!(
        root_line.starts_with("Err(") && root_line.contains(HOME_VARIABLE),
        "{root_line}"
    );

    // The in-process wrapper agrees with the pure function on this process's own environment.
    assert_eq!(cache_root(), cache_root_in(&Env::from_process()));
}
