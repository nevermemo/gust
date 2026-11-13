use super::*;
use crate::store::tests::Scratch;
use std::{
    cell::{Cell, RefCell},
    ffi::OsString,
};

fn env_with(slang_home: Option<&Path>, cargo_home: Option<&Path>, path: Option<&OsString>) -> Env {
    Env {
        slang_home: slang_home.map(|path| path.as_os_str().to_owned()),
        cargo_home: cargo_home.map(|path| path.as_os_str().to_owned()),
        home: None,
        path: path.cloned(),
        helper: None,
    }
}

/// A fake `slangc` whose `-version` output is the content of the file.
fn fake_slangc(dir: &Path, version: &str) -> PathBuf {
    fs::create_dir_all(dir).unwrap();
    let path = dir.join(executable_name("slangc"));
    fs::write(&path, version).unwrap();
    path
}

/// `-version` for the fake slangc files above.
fn read_version(slangc: &Path) -> Result<String, String> {
    fs::read_to_string(slangc).map_err(|error| error.to_string())
}

fn join(dirs: &[&Path]) -> OsString {
    env::join_paths(dirs).unwrap()
}

/// A `git describe` tag that no Slang prints. The tests list it as a validated build to exercise the mechanism; the
/// shipped list is empty (2026.22 was validated with the release archive), so no real tag is ever listed here.
const SYNTHETIC_BUILD: &str = "2026.22-1-g0a1b2c3d4";
const SYNTHETIC_VALIDATED: &[&str] = &[SYNTHETIC_BUILD];

fn locate_synthetic(env: &Env, allow_mismatch: bool) -> Result<Located, String> {
    locate_with(env, allow_mismatch, &read_version, SYNTHETIC_VALIDATED)
}

#[test]
fn the_asset_table_has_the_three_pinned_archives_with_their_digests() {
    assert_eq!(SLANG_VERSION, "2026.22");
    type Row = (&'static str, &'static str, &'static str, &'static str, u64);
    let table: Vec<Row> = ASSETS
        .iter()
        .map(|asset| (asset.os, asset.arch, asset.file, asset.sha256, asset.size))
        .collect();
    assert_eq!(
        table,
        [
            (
                "windows",
                "x86_64",
                "slang-2026.22-windows-x86_64.zip",
                "dd0c4a86bd44cdb7e9cf712af0c8ebf2fff1bba19fa5e3c128ff87384e8ff17b",
                64_105_029
            ),
            (
                "linux",
                "x86_64",
                "slang-2026.22-linux-x86_64.tar.gz",
                "a877b75c6ec515f5d8bf66ea4bae4f0a5ba85050b9540c2046670285808347e6",
                83_714_452
            ),
            (
                "macos",
                "aarch64",
                "slang-2026.22-macos-aarch64.tar.gz",
                "885b5bd9a79a1ae81fdb92212f5f7c785001f870d45c39f9bb5df01339db3fea",
                63_098_220
            ),
        ]
    );
    for asset in ASSETS {
        assert_eq!(asset.version, SLANG_VERSION);
        assert_eq!(asset.sha256.len(), 64, "{}", asset.file);
        assert!(
            asset
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "{}",
            asset.file
        );
        assert!(asset.file.starts_with(&format!(
            "slang-{}-{}-{}.",
            asset.version, asset.os, asset.arch
        )));
    }
    assert_eq!(
        asset_for(SLANG_VERSION, "linux", "x86_64").map(|asset| asset.file),
        Some("slang-2026.22-linux-x86_64.tar.gz")
    );
    assert_eq!(asset_for(SLANG_VERSION, "linux", "aarch64"), None);
    assert_eq!(asset_for("2026.19", "linux", "x86_64"), None);
    assert_eq!(asset_for(SLANG_VERSION, "freebsd", "x86_64"), None);
    assert_eq!(
        download_url(asset_for(SLANG_VERSION, "windows", "x86_64").unwrap()),
        "https://github.com/shader-slang/slang/releases/download/v2026.22/slang-2026.22-windows-x86_64.zip"
    );
}

#[test]
fn the_shipped_list_of_validated_builds_is_empty() {
    // 2026.22 was validated with the release archive, which prints the bare pin; no source build is listed. The check
    // below is an identity check (the pin matches itself); the asset table above holds that the release is the pin.
    assert!(VALIDATED_BUILDS.is_empty(), "{VALIDATED_BUILDS:?}");
    assert!(matches_pin(SLANG_VERSION, VALIDATED_BUILDS));
    // So with the shipped list a `git describe` build of the pin is not the pin, whatever it looks like, and neither is
    // the 2026.20 source build that an earlier pin listed.
    for reported in [
        SYNTHETIC_BUILD,
        "2026.22-1-gdeadbee",
        "2026.13.1",
        "2026.13.1-1-g84792eb15",
    ] {
        assert!(!matches_pin(reported, VALIDATED_BUILDS), "{reported}");
    }
}

#[test]
fn the_pin_matches_exactly_or_a_listed_validated_build_and_nothing_else() {
    for reported in [SLANG_VERSION, SYNTHETIC_BUILD] {
        assert!(matches_pin(reported, SYNTHETIC_VALIDATED), "{reported}");
    }
    // The exact pin matches with or without a list.
    assert!(matches_pin(SLANG_VERSION, &[]));
    assert!(!matches_pin(SYNTHETIC_BUILD, &[]));
    for reported in [
        "",
        "2026.2",
        "2026.220",
        "2026.23",
        "2026.22-",
        "2026.22-rc1",
        "2026.22-1",
        "2026.22-g0a1b2c3d4",
        "2026.22--1-g0a1b2c3d4",
        "2026.22-x-g0a1b2c3d4",
        "2026.22-1-g0a1b2c3d",
        "2026.22-1-gzzzzzzzzz",
        "2026.22-1-g0a1b2c3d4-dirty",
        // Well-formed `git describe` tags of builds nobody validated.
        "2026.22-2-gdeadbee",
        "2026.22-12-gabcdef0",
        "2026.22-400-gabc1234",
        "2026.22-1-g0a1b2c3d5",
        "2026.22 ",
        "v2026.22",
        "2026.19-1-g0a1b2c3d4",
        "12026.22",
    ] {
        assert!(!matches_pin(reported, SYNTHETIC_VALIDATED), "{reported:?}");
    }
}

#[test]
fn the_install_root_is_the_cache_root_of_gust_toolchain() {
    // The rules (GUST_SLANG_HOME, CARGO_HOME, the home directory, refusals) are `gust::toolchain`'s and are tested
    // there table by table; this tool must return exactly what that function returns.
    let scratch = Scratch::new("roots");
    let home = scratch.0.join("slang-home");
    let cargo = scratch.0.join("cargo");
    let user = scratch.0.join("user");
    let mut env = env_with(Some(&home), Some(&cargo), None);
    assert_eq!(install_root(&env).unwrap(), home);
    env.slang_home = None;
    assert_eq!(
        install_root(&env).unwrap(),
        cargo.join("gust").join("slang")
    );
    env.cargo_home = None;
    env.home = Some(user.clone().into_os_string());
    assert_eq!(
        install_root(&env).unwrap(),
        user.join(".cargo").join("gust").join("slang")
    );
    for slang_home in [OsString::new(), OsString::from("relative/dir")] {
        env.slang_home = Some(slang_home);
        assert_eq!(
            install_root(&env),
            toolchain::cache_root_in(&env),
            "{:?}",
            env.slang_home
        );
        assert!(install_root(&env).is_err());
    }
}

#[test]
fn managed_paths_are_keyed_by_the_version() {
    let root = Path::new("/r");
    assert_eq!(
        slangc_in(&root.join(SLANG_VERSION)),
        root.join(SLANG_VERSION)
            .join("bin")
            .join(executable_name("slangc"))
    );
    // `build-helper` writes into `helper_dir`, and the bridge reads `gust::toolchain::managed_helper`: one place.
    assert_eq!(
        helper_dir(root, SLANG_VERSION),
        root.join("helper").join(SLANG_VERSION)
    );
    assert_eq!(
        toolchain::managed_helper(root).parent(),
        Some(helper_dir(root, SLANG_VERSION).as_path())
    );
    assert_eq!(
        slangc_in(&root.join(SLANG_VERSION)),
        toolchain::managed_slangc(root)
    );
}

#[test]
fn locate_prefers_gust_slang_home_then_the_cache_then_path() {
    let scratch = Scratch::new("locate-order");
    let home = scratch.0.join("home");
    let cargo = scratch.0.join("cargo");
    let cache = cargo.join("gust").join("slang");
    let on_path = scratch.0.join("path");
    let path = join(&[&on_path]);
    let path_slangc = fake_slangc(&on_path, SLANG_VERSION);
    let env = env_with(Some(&home), Some(&cargo), Some(&path));
    let without_home = env_with(None, Some(&cargo), Some(&path));

    // Only PATH has it.
    let found = locate(&without_home, false, &read_version).unwrap();
    assert_eq!(found.source, Source::Path);
    assert_eq!(found.slangc, path_slangc);
    assert!(found.pinned && found.root.is_none() && found.install_dir().is_none());

    // The cache beats PATH.
    let cached = fake_slangc(&cache.join(SLANG_VERSION).join("bin"), SLANG_VERSION);
    let found = locate(&without_home, false, &read_version).unwrap();
    assert_eq!(found.source, Source::Cache);
    assert_eq!(found.slangc, cached);
    assert_eq!(found.install_dir(), Some(cache.join(SLANG_VERSION)));

    // GUST_SLANG_HOME beats the cache, and replaces it: a home without the pin goes to PATH, not to the cache.
    let found = locate(&env, false, &read_version).unwrap();
    assert_eq!(found.source, Source::Path);
    let homed = fake_slangc(&home.join(SLANG_VERSION).join("bin"), SYNTHETIC_BUILD);
    let found = locate_synthetic(&env, false).unwrap();
    assert_eq!(found.source, Source::Home);
    assert_eq!(found.slangc, homed);
    assert_eq!(found.version, SYNTHETIC_BUILD);
    assert_eq!(found.root, Some(home.clone()));

    // GUST_SLANG_HOME is the only cache root (section 2.5): a home with no pin directory does not fall through to
    // the cache, it falls through to PATH.
    let other_home = scratch.0.join("empty-home");
    fs::create_dir_all(&other_home).unwrap();
    let env = env_with(Some(&other_home), Some(&cargo), Some(&path));
    let found = locate(&env, false, &read_version).unwrap();
    assert_eq!(found.source, Source::Path);
    assert_eq!(found.slangc, path_slangc);
    // And with nothing on PATH either, the cache that exists is still not used.
    let nowhere = scratch.0.join("nowhere");
    fs::create_dir_all(&nowhere).unwrap();
    let env = env_with(Some(&other_home), Some(&cargo), Some(&join(&[&nowhere])));
    let error = locate(&env, false, &read_version).unwrap_err();
    assert!(error.contains("was not found"), "{error}");
}

#[test]
fn a_located_slang_that_is_not_exactly_the_pin_gets_a_one_line_note() {
    let located = |version: &str, pinned: bool| Located {
        slangc: PathBuf::from("/s/bin/slangc"),
        version: version.to_owned(),
        source: Source::Path,
        pinned,
        root: None,
    };
    assert_eq!(version_note(&located(SLANG_VERSION, true)), None);
    let validated = version_note(&located(SYNTHETIC_BUILD, true)).unwrap();
    assert!(
        validated.contains(&format!("`{SYNTHETIC_BUILD}`"))
            && validated.contains("validated build of the pinned 2026.22")
            && !validated.contains('\n'),
        "{validated}"
    );
    let other = version_note(&located("2025.1.0", false)).unwrap();
    assert!(
        other.contains("`2025.1.0`")
            && other.contains("not the pinned 2026.22")
            && other.contains("--allow-slang-mismatch")
            && !other.contains('\n'),
        "{other}"
    );
}

#[test]
fn an_unlisted_build_of_the_pinned_version_is_refused_everywhere() {
    let scratch = Scratch::new("locate-unlisted");
    let home = scratch.0.join("home");
    let on_path = scratch.0.join("path");
    // A tag that looks like a `git describe` build but was not validated.
    let unlisted = "2026.22-2-gdeadbee";
    fake_slangc(&on_path, unlisted);
    let env = env_with(
        None,
        Some(&scratch.0.join("cargo")),
        Some(&join(&[&on_path])),
    );
    let error = locate(&env, false, &read_version).unwrap_err();
    assert!(error.contains(unlisted), "{error}");
    assert!(error.contains("--allow-slang-mismatch"), "{error}");
    let found = locate(&env, true, &read_version).unwrap();
    assert!(!found.pinned);
    // In the cache it is a hard error, never substituted.
    fake_slangc(&home.join(SLANG_VERSION).join("bin"), unlisted);
    let env = env_with(Some(&home), None, Some(&join(&[&on_path])));
    for allow in [false, true] {
        let error = locate(&env, allow, &read_version).unwrap_err();
        assert!(
            error.contains(&format!("reports Slang `{unlisted}`")),
            "{error}"
        );
    }
    // The shipped list is empty, so the synthetic build is just another unlisted build for the public path ...
    fake_slangc(&home.join(SLANG_VERSION).join("bin"), SYNTHETIC_BUILD);
    let error = locate(&env, false, &read_version).unwrap_err();
    assert!(
        error.contains(&format!("reports Slang `{SYNTHETIC_BUILD}`")),
        "{error}"
    );
    // ... and with the synthetic build listed it is accepted from the cache and is pinned, though it is not exactly
    // the pin. The unlisted build is still refused.
    let found = locate_synthetic(&env, false).unwrap();
    assert!(found.pinned);
    assert_ne!(found.version, SLANG_VERSION);
    fake_slangc(&home.join(SLANG_VERSION).join("bin"), unlisted);
    assert!(locate_synthetic(&env, false).is_err());
}

#[test]
fn a_mismatched_slangc_on_path_is_refused_with_the_install_command_unless_allowed() {
    let scratch = Scratch::new("locate-mismatch");
    let first = scratch.0.join("first");
    let second = scratch.0.join("second");
    let wrong = fake_slangc(&first, "2025.1.0");
    fake_slangc(&second, SLANG_VERSION);
    let path = join(&[&first, &second]);
    let env = env_with(None, Some(&scratch.0.join("cargo")), Some(&path));
    let error = locate(&env, false, &read_version).unwrap_err();
    assert!(error.contains("2025.1.0"), "{error}");
    assert!(error.contains(&wrong.display().to_string()), "{error}");
    assert!(error.contains("cargo gust slang install"), "{error}");
    assert!(error.contains("--allow-slang-mismatch"), "{error}");
    assert!(
        error.contains("https://github.com/shader-slang/slang/releases/download/v2026.22/"),
        "the manual download URL: {error}"
    );
    // With the flag the first slangc on PATH is used, and it is marked as not the pin.
    let found = locate(&env, true, &read_version).unwrap();
    assert_eq!(found.slangc, wrong);
    assert_eq!(found.version, "2025.1.0");
    assert!(!found.pinned);
    assert_eq!(found.source, Source::Path);
    // The flag never overrides a pinned install in the cache.
    let cache = scratch.0.join("cargo/gust/slang").join(SLANG_VERSION);
    let cached = fake_slangc(&cache.join("bin"), SLANG_VERSION);
    assert_eq!(locate(&env, true, &read_version).unwrap().slangc, cached);
}

#[test]
fn nothing_found_is_an_error_that_names_the_install_command_and_the_download() {
    let scratch = Scratch::new("locate-none");
    let empty = scratch.0.join("empty");
    fs::create_dir_all(&empty).unwrap();
    let path = join(&[&empty]);
    let env = env_with(None, Some(&scratch.0.join("cargo")), Some(&path));
    for allow in [false, true] {
        let error = locate(&env, allow, &read_version).unwrap_err();
        assert!(error.contains("was not found"), "{error}");
        assert!(error.contains("cargo gust slang install"), "{error}");
    }
    let error = locate(&env_with(None, None, None), false, &read_version).unwrap_err();
    assert!(error.contains("cargo gust slang install"), "{error}");
}

#[test]
fn an_install_directory_with_another_version_is_never_substituted() {
    let scratch = Scratch::new("locate-bad-install");
    let home = scratch.0.join("home");
    fake_slangc(&home.join(SLANG_VERSION).join("bin"), "2026.12.0");
    let on_path = scratch.0.join("path");
    fake_slangc(&on_path, SLANG_VERSION);
    let env = env_with(Some(&home), None, Some(&join(&[&on_path])));
    for allow in [false, true] {
        let error = locate(&env, allow, &read_version).unwrap_err();
        assert!(error.contains("reports Slang `2026.12.0`"), "{error}");
        assert!(error.contains("cargo gust slang install"), "{error}");
    }
    // A slangc that cannot report its version is an error too, not a skip.
    let failing = |_: &Path| -> Result<String, String> { Err("exit 3".to_owned()) };
    let error = locate(&env, false, &failing).unwrap_err();
    assert!(
        error.contains("cannot report its version (exit 3)"),
        "{error}"
    );
    let env = env_with(None, None, Some(&join(&[&on_path])));
    let error = locate(&env, true, &failing).unwrap_err();
    assert!(
        error.contains("cannot report its version (exit 3)"),
        "{error}"
    );
}

#[test]
fn notice_files_are_the_licence_files_and_directories_beside_the_binaries() {
    let scratch = Scratch::new("notices");
    let dir = &scratch.0;
    for name in [
        "LICENSE",
        "README.md",
        "NOTICE.txt",
        "THIRD-PARTY-NOTICES.txt",
        "licence.md",
    ] {
        fs::write(dir.join(name), "x").unwrap();
    }
    // The two directories of a 2026.22 bundle are listed as directories; any other directory is not, whatever its name.
    fs::create_dir(dir.join("LICENSES")).unwrap();
    fs::create_dir(dir.join("third-party-notices")).unwrap();
    fs::create_dir(dir.join("licenses-extra")).unwrap();
    fs::create_dir(dir.join("share")).unwrap();
    fs::write(dir.join("LICENSES").join("MIT.txt"), "x").unwrap();
    let names: Vec<String> = notice_files(dir)
        .iter()
        .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        [
            "LICENSE",
            "LICENSES",
            "NOTICE.txt",
            "THIRD-PARTY-NOTICES.txt",
            "licence.md",
            "third-party-notices",
        ]
    );
    assert!(notice_files(&dir.join("missing")).is_empty());
}

// ---- install -------------------------------------------------------------------------------------------------------

/// The bytes of a pretend archive and an asset that pins them.
fn fake_asset(bytes: &[u8]) -> Asset {
    Asset {
        version: SLANG_VERSION,
        os: "test",
        arch: "test",
        file: "slang-test.tar.gz",
        sha256: Box::leak(sha256_hex(bytes).into_boxed_str()),
        size: bytes.len() as u64,
    }
}

/// A callback run inside the fake extraction.
type Hook = Box<dyn Fn(&Path)>;

/// A `Fetch` that serves fixed bytes and extracts a canned tree, recording what happened.
struct Fake {
    served: Vec<u8>,
    /// What `extract` does: write the tree, then fail midway when true.
    fail_extract_after_writing: bool,
    write_slangc: bool,
    fail_download: bool,
    /// A hook run inside `extract`, after the tree is written (to simulate another process).
    during_extract: Option<Hook>,
    downloads: Cell<u32>,
    extracts: Cell<u32>,
    seen_urls: RefCell<Vec<String>>,
}

impl Fake {
    fn serving(bytes: &[u8]) -> Self {
        Self {
            served: bytes.to_vec(),
            fail_extract_after_writing: false,
            write_slangc: true,
            fail_download: false,
            during_extract: None,
            downloads: Cell::new(0),
            extracts: Cell::new(0),
            seen_urls: RefCell::new(Vec::new()),
        }
    }
}

impl Fetch for Fake {
    fn download(&self, url: &str, dest: &Path) -> Result<(), String> {
        self.downloads.set(self.downloads.get() + 1);
        self.seen_urls.borrow_mut().push(url.to_owned());
        if self.fail_download {
            fs::write(dest, b"half a file").unwrap();
            return Err("`curl -fsSL` failed (exit 22)".to_owned());
        }
        fs::write(dest, &self.served).map_err(|error| error.to_string())
    }

    fn extract(&self, _archive: &Path, dest: &Path) -> Result<(), String> {
        self.extracts.set(self.extracts.get() + 1);
        fs::write(dest.join("LICENSE"), "Apache-2.0 WITH LLVM-exception").unwrap();
        // The two notice directories of a 2026.22 bundle.
        fs::create_dir(dest.join("LICENSES")).unwrap();
        fs::write(dest.join("LICENSES").join("MIT.txt"), "MIT").unwrap();
        fs::create_dir(dest.join("third-party-notices")).unwrap();
        fs::write(
            dest.join("third-party-notices").join("lz4-LICENSE"),
            "BSD-2",
        )
        .unwrap();
        if self.write_slangc {
            fake_slangc(&dest.join("bin"), SLANG_VERSION);
        }
        if let Some(hook) = &self.during_extract {
            hook(dest);
        }
        if self.fail_extract_after_writing {
            return Err("tar: unexpected end of archive".to_owned());
        }
        Ok(())
    }
}

/// Everything under `root`, relative and sorted: the directory state a failed install must leave unchanged.
fn tree(root: &Path) -> Vec<String> {
    fn walk(dir: &Path, base: &Path, out: &mut Vec<String>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            out.push(
                path.strip_prefix(base)
                    .unwrap()
                    .display()
                    .to_string()
                    .replace('\\', "/"),
            );
            if path.is_dir() {
                walk(&path, base, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

const PID: u32 = 4242;

#[test]
fn install_downloads_verifies_extracts_and_renames_keeping_the_notices_beside_the_binaries() {
    let scratch = Scratch::new("install-ok");
    let root = scratch.0.join("cache");
    let asset = fake_asset(b"the archive bytes");
    let tools = Fake::serving(b"the archive bytes");
    let installed = install(&root, &asset, &tools, PID).unwrap();
    let dir = root.join(SLANG_VERSION);
    assert_eq!(installed, Installed::Fresh(dir.clone()));
    assert_eq!(
        tools.seen_urls.borrow().as_slice(),
        ["https://github.com/shader-slang/slang/releases/download/v2026.22/slang-test.tar.gz"]
    );
    assert!(slangc_in(&dir).is_file());
    assert_eq!(
        notice_files(&dir),
        [
            dir.join("LICENSE"),
            dir.join("LICENSES"),
            dir.join("third-party-notices")
        ],
        "the archive layout is kept, so the licence and the notice directories sit beside bin/"
    );
    // No partial directory and no download left over: only the finished tree and the (empty) archives directory.
    let leftovers: Vec<String> = tree(&root)
        .into_iter()
        .filter(|name| name.contains("partial") || name.contains(".part-"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
    assert!(!root.join("archives").join("slang-test.tar.gz").exists());
    // A second install is a no-op that does not even download.
    let again = install(&root, &asset, &tools, PID).unwrap();
    assert_eq!(again, Installed::Already(dir));
    assert_eq!((tools.downloads.get(), tools.extracts.get()), (1, 1));
}

#[test]
fn a_download_that_fails_the_digest_is_refused_before_extraction_and_leaves_nothing() {
    // Mutation (a), download form: one byte of the served archive differs.
    let scratch = Scratch::new("install-corrupt-download");
    let root = scratch.0.join("cache");
    let asset = fake_asset(b"the archive bytes");
    let mut corrupt = b"the archive bytes".to_vec();
    corrupt[3] ^= 1;
    let tools = Fake::serving(&corrupt);
    let error = install(&root, &asset, &tools, PID).unwrap_err();
    assert!(
        error.contains("SHA-256") && error.contains(asset.sha256),
        "{error}"
    );
    assert!(error.contains("discarded"), "{error}");
    assert!(
        error.contains("slang-test.tar.gz"),
        "the manual step names the file: {error}"
    );
    assert_eq!(
        tools.extracts.get(),
        0,
        "nothing is extracted before the check passes"
    );
    assert!(!root.join(SLANG_VERSION).exists());
    assert_eq!(
        tree(&root),
        ["archives"],
        "no partial directory, no download"
    );
}

#[test]
fn a_cached_archive_with_one_changed_byte_is_refused_before_extraction_and_kept() {
    // Mutation (a), cache form: the archive is already in <root>/archives (an earlier run, or put there by hand).
    let scratch = Scratch::new("install-corrupt-cache");
    let root = scratch.0.join("cache");
    let asset = fake_asset(b"the archive bytes");
    fs::create_dir_all(root.join("archives")).unwrap();
    let mut corrupt = b"the archive bytes".to_vec();
    corrupt[0] ^= 0x20;
    let cached = root.join("archives").join("slang-test.tar.gz");
    fs::write(&cached, &corrupt).unwrap();
    let tools = Fake::serving(b"the archive bytes");
    let error = install(&root, &asset, &tools, PID).unwrap_err();
    assert!(error.contains("cached archive failed its check"), "{error}");
    assert!(error.contains("delete it"), "{error}");
    assert_eq!((tools.downloads.get(), tools.extracts.get()), (0, 0));
    assert_eq!(
        fs::read(&cached).unwrap(),
        corrupt,
        "the evidence is left in place"
    );
    assert_eq!(tree(&root), ["archives", "archives/slang-test.tar.gz"]);
    // The same cache with the good bytes installs without a download, and keeps the archive the user supplied.
    fs::write(&cached, b"the archive bytes").unwrap();
    assert_eq!(
        install(&root, &asset, &tools, PID).unwrap(),
        Installed::Fresh(root.join(SLANG_VERSION))
    );
    assert_eq!(tools.downloads.get(), 0);
    assert!(cached.is_file());
}

#[test]
fn a_failed_extraction_removes_the_partial_directory_and_no_version_directory_appears() {
    // Mutation (d): extraction writes part of the tree and fails.
    let scratch = Scratch::new("install-extract-fail");
    let root = scratch.0.join("cache");
    let asset = fake_asset(b"the archive bytes");
    let mut tools = Fake::serving(b"the archive bytes");
    tools.fail_extract_after_writing = true;
    let error = install(&root, &asset, &tools, PID).unwrap_err();
    assert!(error.contains("unexpected end of archive"), "{error}");
    assert!(!root.join(SLANG_VERSION).exists());
    assert!(!root.join(format!("{SLANG_VERSION}.partial-{PID}")).exists());
    // The verified archive stays, so the next try does not download again.
    assert_eq!(tree(&root), ["archives", "archives/slang-test.tar.gz"]);
    tools.fail_extract_after_writing = false;
    install(&root, &asset, &tools, PID).unwrap();
    assert_eq!(tools.downloads.get(), 1);
    assert!(slangc_in(&root.join(SLANG_VERSION)).is_file());
}

#[test]
fn an_archive_without_slangc_and_a_failed_download_leave_nothing_behind() {
    let scratch = Scratch::new("install-bad-shape");
    let root = scratch.0.join("cache");
    let asset = fake_asset(b"the archive bytes");
    let mut tools = Fake::serving(b"the archive bytes");
    tools.write_slangc = false;
    let error = install(&root, &asset, &tools, PID).unwrap_err();
    assert!(error.contains("not a Slang release archive"), "{error}");
    assert!(!root.join(SLANG_VERSION).exists());
    assert!(!root.join(format!("{SLANG_VERSION}.partial-{PID}")).exists());

    let scratch = Scratch::new("install-download-fail");
    let root = scratch.0.join("cache");
    let mut tools = Fake::serving(b"the archive bytes");
    tools.fail_download = true;
    let error = install(&root, &asset, &tools, PID).unwrap_err();
    assert!(error.contains("curl -fsSL"), "{error}");
    assert_eq!(
        tree(&root),
        ["archives"],
        "the half-written download is removed"
    );
}

#[test]
fn an_incomplete_version_directory_is_reported_not_overwritten() {
    let scratch = Scratch::new("install-incomplete");
    let root = scratch.0.join("cache");
    fs::create_dir_all(root.join(SLANG_VERSION)).unwrap();
    fs::write(root.join(SLANG_VERSION).join("stray"), "x").unwrap();
    let asset = fake_asset(b"the archive bytes");
    let tools = Fake::serving(b"the archive bytes");
    let error = install(&root, &asset, &tools, PID).unwrap_err();
    assert!(error.contains("holds no bin/slangc"), "{error}");
    assert_eq!(tools.downloads.get(), 0);
    assert!(root.join(SLANG_VERSION).join("stray").is_file());
}

#[test]
fn an_install_that_loses_the_race_to_another_process_succeeds_and_cleans_its_partial() {
    let scratch = Scratch::new("install-race");
    let root = scratch.0.join("cache");
    let asset = fake_asset(b"the archive bytes");
    let mut tools = Fake::serving(b"the archive bytes");
    let rival = root.join(SLANG_VERSION);
    tools.during_extract = Some(Box::new(move |_| {
        fake_slangc(&rival.join("bin"), SLANG_VERSION);
    }));
    let installed = install(&root, &asset, &tools, PID).unwrap();
    assert_eq!(installed, Installed::Already(root.join(SLANG_VERSION)));
    assert!(!root.join(format!("{SLANG_VERSION}.partial-{PID}")).exists());
}

/// A `Fetch` that serves fixed bytes but extracts with the real `tar`.
struct RealTar(Vec<u8>);

impl Fetch for RealTar {
    fn download(&self, _url: &str, dest: &Path) -> Result<(), String> {
        fs::write(dest, &self.0).map_err(|error| error.to_string())
    }

    fn extract(&self, archive: &Path, dest: &Path) -> Result<(), String> {
        SystemTools.extract(archive, dest)
    }
}

#[test]
fn the_real_tar_extracts_a_good_archive_and_a_bad_one_leaves_no_version_directory() {
    let scratch = Scratch::new("install-real-tar");
    let source = scratch.0.join("src");
    fake_slangc(&source.join("bin"), SLANG_VERSION);
    fs::write(source.join("LICENSE"), "licence text").unwrap();
    let archive = scratch.0.join("made.tar.gz");
    let status = Command::new(SystemTools::tar())
        .arg("-czf")
        .arg(&archive)
        .arg("-C")
        .arg(&source)
        .arg(".")
        .status()
        .expect("tar is available (the manager needs it too)");
    assert!(status.success());
    let bytes = fs::read(&archive).unwrap();
    let asset = fake_asset(&bytes);
    let root = scratch.0.join("cache");
    let installed = install(&root, &asset, &RealTar(bytes), PID).unwrap();
    let dir = root.join(SLANG_VERSION);
    assert_eq!(installed, Installed::Fresh(dir.clone()));
    assert_eq!(
        fs::read_to_string(dir.join("LICENSE")).unwrap(),
        "licence text"
    );
    assert_eq!(read_version(&slangc_in(&dir)).unwrap(), SLANG_VERSION);

    // A bad archive whose digest is right: the real tar fails, and nothing but the archive is left.
    let garbage = b"this is not an archive at all".to_vec();
    let asset = fake_asset(&garbage);
    let root = scratch.0.join("cache2");
    let error = install(&root, &asset, &RealTar(garbage), PID).unwrap_err();
    assert!(error.contains("tar") && error.contains("failed"), "{error}");
    assert!(!root.join(SLANG_VERSION).exists());
    assert!(!root.join(format!("{SLANG_VERSION}.partial-{PID}")).exists());
}
