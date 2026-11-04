use super::*;
use crate::store::tests::Scratch;
use std::ffi::OsString;

/// `PATH` with one directory per name, each holding the given tools.
fn path_with(scratch: &Scratch, dirs: &[&[&str]]) -> OsString {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut entries = Vec::new();
    for (index, tools) in dirs.iter().enumerate() {
        let dir = scratch.0.join(format!("bin{call}-{index}"));
        fs::create_dir_all(&dir).unwrap();
        for tool in *tools {
            fs::write(dir.join(executable_name(tool)), "x").unwrap();
        }
        entries.push(dir);
    }
    env::join_paths(entries).unwrap()
}

#[test]
fn the_packaged_helper_source_is_the_probe_source() {
    // `scripts/probes/slang-layout.cpp` is the source of truth; the package carries a copy so that a published tool
    // can build the helper. Line endings are normalised because a Windows checkout may convert one of the two.
    let original =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/probes/slang-layout.cpp");
    let original = fs::read_to_string(&original).unwrap();
    assert_eq!(
        HELPER_SOURCE.replace("\r\n", "\n"),
        original.replace("\r\n", "\n"),
        "copy scripts/probes/slang-layout.cpp to crates/cargo-gust/assets/slang-layout.cpp"
    );
    assert!(
        HELPER_SOURCE.contains(&format!("gust-slang-reflect schema={HELPER_SCHEMA} slang=")),
        "HELPER_SCHEMA must be the schema the packaged source prints"
    );
}

#[test]
fn schema_3_is_written_only_for_textures_samplers_and_a_non_compute_stage() {
    // The behaviour itself is exercised by running the built helper (the byte-identity gate and the Reflect suite);
    // these pins catch an edit that widens schema 3 to every document or drops the compute default of the stage.
    let source = HELPER_SOURCE.replace("\r\n", "\n");
    assert!(
        source.contains("const int schema = use.shapes ? 3 : (use.newKinds ? 2 : 1);"),
        "schema 3 must be written only when a texture, a sampler or a non-compute stage is present"
    );
    assert!(
        source.contains("use.shapes = !compute;"),
        "a fragment entry is schema 3 even without a texture"
    );
    assert!(
        source.contains("Stage stage = Stage::Compute;") && source.contains("if (argc == 7)"),
        "the stage is the optional seventh argument and defaults to compute"
    );
}

/// The probe inputs of `src/testdata/schema3`, by file stem (the entry point is `gpu_<stem>`), with the stage passed to
/// the helper. `fragment_gap` also runs as `compute` to pin the stage refusal.
const SCHEMA_3_CASES: &[(&str, &str)] = &[
    ("sampled_float4", "compute"),
    ("sampled_uint", "compute"),
    ("storage_declared", "compute"),
    ("storage_inferred", "compute"),
    ("sampler_fragment", "fragment"),
    ("fragment_gap", "fragment"),
    ("fragment_gap", "compute"),
    ("refuse_storage_noformat", "compute"),
    ("refuse_comparison_sampler", "compute"),
    ("refuse_texture3d", "compute"),
    ("refuse_texture_array", "compute"),
    ("refuse_fragment_depth", "fragment"),
];

fn schema_3_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/testdata/schema3")
}

#[test]
fn the_schema_3_expectations_cover_every_probe_input_and_both_targets() {
    let expected = fs::read_to_string(schema_3_dir().join("expected.txt"))
        .unwrap()
        .replace("\r\n", "\n");
    let mut headers = Vec::new();
    for (stem, stage) in SCHEMA_3_CASES {
        assert!(
            schema_3_dir().join(format!("{stem}.slang")).is_file(),
            "{stem}"
        );
        for target in ["wgsl", "spirv"] {
            headers.push(format!("== {stem} {target} {stage}"));
        }
    }
    let found: Vec<&str> = expected
        .lines()
        .filter(|line| line.starts_with("== "))
        .collect();
    assert_eq!(found, headers, "expected.txt must list the cases in order");
}

/// Runs the built helper on every probe input and compares its output (a document, or the refusal) with
/// `expected.txt`, byte for byte once line endings are normalised. It needs the helper, so it is ignored by default and
/// fails, never skips, when there is no helper (the rule of the Reflect tests). The Reflect suite runs it
/// (`REFLECTION_TESTS` in xtask), after it has prepared the helper; by hand:
/// `GUST_SLANG_REFLECT=<helper> cargo test -p cargo-gust -- --ignored schema_3_output`. The helper is the one
/// `gust::reflect::native_compiler_path` names (`GUST_SLANG_REFLECT`, else the managed helper of the Slang cache, else
/// `target/slang-reflect` of a GUST checkout). The expected text
/// holds the Slang build tag and the artifact fingerprints of the pinned Slang, so a pin bump re-blesses it: set
/// `GUST_SCHEMA3_ACTUAL=<file>` to have the test write what the helper printed, review it, and copy it over
/// `expected.txt`.
#[test]
#[ignore = "needs the native helper: GUST_SLANG_REFLECT=<helper> cargo test -p cargo-gust -- --ignored schema_3_output"]
fn schema_3_output_of_the_built_helper_matches_the_committed_expectations() {
    let helper = gust::reflect::native_compiler_path();
    let scratch = Scratch::new("schema3-output");
    let mut actual = String::new();
    for (stem, stage) in SCHEMA_3_CASES {
        // The source hash in the document covers the source bytes, so the copy is LF whatever the checkout did.
        let text = fs::read_to_string(schema_3_dir().join(format!("{stem}.slang")))
            .unwrap()
            .replace("\r\n", "\n");
        let source = scratch.0.join(format!("{stem}.slang"));
        fs::write(&source, text).unwrap();
        for target in ["wgsl", "spirv"] {
            let artifact = scratch.0.join(format!("{stem}.{target}.artifact"));
            let document = scratch.0.join(format!("{stem}.{target}.json"));
            let output = Command::new(&helper)
                .arg(&source)
                .arg(format!("gpu_{stem}"))
                .arg(target)
                .arg(&artifact)
                .arg(&document)
                .arg(stage)
                .output()
                .unwrap();
            actual.push_str(&format!("== {stem} {target} {stage}\n"));
            if output.status.success() {
                actual.push_str(&fs::read_to_string(&document).unwrap());
            } else {
                actual.push_str(&format!(
                    "error: {}",
                    String::from_utf8_lossy(&output.stderr).trim()
                ));
            }
            actual.push('\n');
        }
    }
    if let Some(path) = env::var_os("GUST_SCHEMA3_ACTUAL") {
        fs::write(path, &actual).unwrap();
    }
    let expected = fs::read_to_string(schema_3_dir().join("expected.txt"))
        .unwrap()
        .replace("\r\n", "\n");
    assert_eq!(actual, expected);
}

#[test]
fn the_compiler_is_the_first_of_cl_clang_c_plus_plus_and_g_plus_plus_on_path() {
    assert_eq!(COMPILER_SEARCH_ORDER, ["cl", "clang++", "c++", "g++"]);
    let scratch = Scratch::new("compiler-order");
    let all = path_with(&scratch, &[&["g++", "c++"], &["clang++"], &["cl"]]);
    assert_eq!(
        choose_compiler(None, Some(&all), false),
        Ok(Compiler::Cl("cl".to_owned()))
    );
    let no_cl = path_with(&scratch, &[&["g++", "c++"], &["clang++"]]);
    assert_eq!(
        choose_compiler(None, Some(&no_cl), false),
        Ok(Compiler::Driver("clang++".to_owned()))
    );
    let gnu = path_with(&scratch, &[&["g++"], &["c++"]]);
    assert_eq!(
        choose_compiler(None, Some(&gnu), false),
        Ok(Compiler::Driver("c++".to_owned()))
    );
    let only_gpp = path_with(&scratch, &[&["g++"]]);
    assert_eq!(
        choose_compiler(None, Some(&only_gpp), false),
        Ok(Compiler::Driver("g++".to_owned()))
    );
}

#[test]
fn cxx_overrides_the_search_and_an_empty_one_is_ignored() {
    let scratch = Scratch::new("compiler-cxx");
    let path = path_with(&scratch, &[&["cl"]]);
    assert_eq!(
        choose_compiler(Some("/opt/clang/bin/clang++"), Some(&path), false),
        Ok(Compiler::Driver("/opt/clang/bin/clang++".to_owned()))
    );
    assert_eq!(
        choose_compiler(Some(""), Some(&path), false),
        Ok(Compiler::Cl("cl".to_owned()))
    );
}

#[test]
fn a_cxx_whose_file_stem_is_cl_selects_msvc_and_keeps_its_path() {
    // PATH holds clang++ only, so an MSVC choice can come from CXX alone.
    let scratch = Scratch::new("compiler-cxx-cl");
    let path = path_with(&scratch, &[&["clang++"]]);
    for cxx in [
        "cl",
        "CL",
        "Cl",
        "cl.exe",
        "CL.EXE",
        "cl.Exe",
        "C:\\Program Files\\MSVC\\bin\\Hostx64\\x64\\cl.exe",
        "C:\\msvc\\bin\\CL",
        "C:/msvc/bin/cl.exe",
        "/opt/msvc/bin/cl",
        "bin\\x64/cl.exe",
    ] {
        for windows in [false, true] {
            assert_eq!(
                choose_compiler(Some(cxx), Some(&path), windows),
                Ok(Compiler::Cl(cxx.to_owned())),
                "CXX={cxx} windows={windows}"
            );
        }
    }
}

#[test]
fn clang_cl_and_other_cxx_values_stay_drivers() {
    let scratch = Scratch::new("compiler-cxx-driver");
    let path = path_with(&scratch, &[&["cl"]]);
    for cxx in [
        "clang-cl",
        "clang-cl.exe",
        "C:\\LLVM\\bin\\clang-cl.exe",
        "clang++",
        "C:\\LLVM\\bin\\clang++.exe",
        "/usr/bin/g++",
        "cl.exe.bak",
        "cl.com",
        "mycl",
        "xcl.exe",
        "C:\\cl\\clang++.exe",
        "C:\\tools\\cl\\",
        "cl\\",
        ".exe",
        "x.exe",
    ] {
        assert_eq!(
            choose_compiler(Some(cxx), Some(&path), true),
            Ok(Compiler::Driver(cxx.to_owned())),
            "CXX={cxx}"
        );
    }
}

#[test]
fn no_compiler_is_an_error_on_unix_and_the_vcvars_fallback_on_windows() {
    let scratch = Scratch::new("compiler-none");
    let empty = path_with(&scratch, &[&[]]);
    let error = choose_compiler(None, Some(&empty), false).unwrap_err();
    assert!(
        error.contains("cl, clang++, c++, g++") && error.contains("set CXX"),
        "{error}"
    );
    assert!(choose_compiler(None, None, false).is_err());
    assert_eq!(
        choose_compiler(None, Some(&empty), true),
        Ok(Compiler::ClViaVcvars)
    );
    // A MinGW compiler cannot link the MSVC import library, so Windows ignores `c++` and `g++` on PATH.
    let mingw = path_with(&scratch, &[&["g++", "c++"]]);
    assert_eq!(
        choose_compiler(None, Some(&mingw), true),
        Ok(Compiler::ClViaVcvars)
    );
    let clang = path_with(&scratch, &[&["g++"], &["clang++"]]);
    assert_eq!(
        choose_compiler(None, Some(&clang), true),
        Ok(Compiler::Driver("clang++".to_owned()))
    );
}

#[test]
fn the_compiler_command_lines_are_the_ones_xtask_used() {
    let args = msvc_args(
        Path::new("inc"),
        Path::new("a.cpp"),
        Path::new("a.obj"),
        Path::new("a.exe"),
        Path::new("slang.lib"),
    );
    assert_eq!(
        args,
        [
            "/nologo",
            "/std:c++17",
            "/EHsc",
            "/W4",
            "/WX",
            "/Iinc",
            "a.cpp",
            "/Foa.obj",
            "/Fea.exe",
            "/link",
            "slang.lib"
        ]
    );
    let args = driver_args(
        Path::new("inc"),
        Path::new("a.cpp"),
        Path::new("lib/libslang.so"),
        Some(Path::new("lib")),
        Path::new("out/helper"),
    );
    assert_eq!(
        args,
        [
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-isystem",
            "inc",
            "a.cpp",
            "lib/libslang.so",
            "-Wl,-rpath,lib",
            "-o",
            "out/helper"
        ]
    );
    let without = driver_args(
        Path::new("inc"),
        Path::new("a.cpp"),
        Path::new("slang.lib"),
        None,
        Path::new("h.exe"),
    );
    assert!(without.iter().all(|arg| !arg.starts_with("-Wl,")));
}

#[test]
fn the_header_directory_is_include_slang_or_include() {
    let scratch = Scratch::new("include");
    let sdk = scratch.0.join("sdk");
    fs::create_dir_all(sdk.join("include")).unwrap();
    assert!(
        include_dir(&sdk)
            .unwrap_err()
            .contains("slang.h was not found")
    );
    fs::write(sdk.join("include/slang.h"), "x").unwrap();
    assert_eq!(include_dir(&sdk).unwrap(), sdk.join("include"));
    fs::create_dir_all(sdk.join("include/slang")).unwrap();
    fs::write(sdk.join("include/slang/slang.h"), "x").unwrap();
    assert_eq!(include_dir(&sdk).unwrap(), sdk.join("include/slang"));
}

#[test]
fn the_source_is_rewritten_only_when_its_bytes_differ() {
    let scratch = Scratch::new("source");
    let path = write_source(&scratch.0).unwrap();
    assert_eq!(fs::read(&path).unwrap(), HELPER_SOURCE.as_bytes());
    // A stale copy is replaced.
    fs::write(&path, "stale").unwrap();
    write_source(&scratch.0).unwrap();
    assert_eq!(fs::read(&path).unwrap(), HELPER_SOURCE.as_bytes());
    // A current copy is not written at all: a read-only file is left alone without an error.
    let mut permissions = fs::metadata(&path).unwrap().permissions();
    permissions.set_readonly(true);
    fs::set_permissions(&path, permissions.clone()).unwrap();
    write_source(&scratch.0).unwrap();
    #[allow(clippy::permissions_set_readonly_false)]
    permissions.set_readonly(false);
    fs::set_permissions(&path, permissions).unwrap();
}

#[test]
fn the_source_hash_is_a_sha256_of_the_source_with_unix_line_endings() {
    let hash = source_sha256();
    assert_eq!(hash.len(), 64);
    assert!(hash.bytes().all(|byte| byte.is_ascii_hexdigit()), "{hash}");
    let unix = HELPER_SOURCE.replace("\r\n", "\n");
    assert_eq!(hash, sha256_hex(unix.as_bytes()));
}

/// A scratch directory with a helper file and the build stamp a good build of this tool's probe source would leave.
fn stamped_helper(name: &str, schema: &str, tag: &str) -> (Scratch, PathBuf, PathBuf) {
    let scratch = Scratch::new(name);
    let stamp = scratch.0.join(HELPER_STAMP);
    let exe = scratch.0.join(executable_name(HELPER_NAME));
    fs::write(&exe, "x").unwrap();
    fs::write(&stamp, stamp_text(schema, tag)).unwrap();
    (scratch, exe, stamp)
}

fn reports(output: &'static str) -> impl Fn(&Path) -> Result<String, String> {
    move |_| Ok(output.to_owned())
}

#[test]
fn a_helper_is_current_only_when_its_stamp_schema_and_slang_tag_are_this_tools_and_the_sdks() {
    let (_scratch, exe, stamp) = stamped_helper("staleness", "3", "2026.20");
    let sdk = Ok("2026.20".to_owned());
    let current = reports("gust-slang-reflect schema=3 slang=2026.20\n");
    assert_eq!(staleness(&exe, &stamp, &sdk, &current), Ok(None));
    // A helper built against another Slang is stale whatever the times of its files say; the reason names both tags.
    for old in [
        "2026.13.1",
        "2026.13.1-1-g84792eb15",
        "2026.20-1-g0a1b2c3d4",
        "2026.2",
    ] {
        let output = format!("gust-slang-reflect schema=3 slang={old}\n");
        let older = move |_: &Path| Ok(output.clone());
        let reason = staleness(&exe, &stamp, &sdk, &older).unwrap().unwrap();
        assert_eq!(
            reason,
            format!("it was built against Slang `{old}`, but the SDK is Slang `2026.20`")
        );
    }
}

#[test]
fn a_helper_without_this_sources_stamp_is_stale() {
    let (_scratch, exe, stamp) = stamped_helper("staleness-stamp", "3", "2026.20");
    let sdk = Ok("2026.20".to_owned());
    let current = reports("gust-slang-reflect schema=3 slang=2026.20\n");
    assert_eq!(staleness(&exe, &stamp, &sdk, &current), Ok(None));
    // A stamp of another probe source (a helper built by a tool with a different source) is stale.
    let other = stamp_text("3", "2026.20").replace(&source_sha256(), &"0".repeat(64));
    fs::write(&stamp, other).unwrap();
    let reason = staleness(&exe, &stamp, &sdk, &current).unwrap().unwrap();
    assert!(
        reason.starts_with("it was built from another probe source"),
        "{reason}"
    );
    assert!(reason.contains("0000000000000000"), "{reason}");
    // No stamp, or one that cannot be read, is stale too.
    fs::write(&stamp, "garbage\n").unwrap();
    let reason = staleness(&exe, &stamp, &sdk, &current).unwrap().unwrap();
    assert!(reason.contains("cannot be read"), "{reason}");
    fs::remove_file(&stamp).unwrap();
    let reason = staleness(&exe, &stamp, &sdk, &current).unwrap().unwrap();
    assert!(reason.starts_with("it has no build stamp"), "{reason}");
    // A helper replaced after its build (the stamp names another tag than the helper prints) is stale.
    fs::write(&stamp, stamp_text("3", "2026.19")).unwrap();
    let reason = staleness(&exe, &stamp, &sdk, &current).unwrap().unwrap();
    assert!(reason.contains("replaced after the build"), "{reason}");
}

#[test]
fn a_stamp_does_not_depend_on_which_tool_wrote_it() {
    // Two cargo-gust versions with the same probe source write byte-identical stamps for the same helper, so neither
    // rebuilds the other's helper.
    assert_eq!(
        stamp_text("3", "2026.20"),
        format!(
            "gust-slang-reflect build stamp\nsource-sha256={}\nschema=3\nslang=2026.20\n",
            source_sha256()
        )
    );
    assert_eq!(
        parse_stamp(&stamp_text("3", "2026.20")),
        Some((source_sha256(), "3".to_owned(), "2026.20".to_owned()))
    );
    assert_eq!(parse_stamp("schema=3\nslang=x\n"), None);
    assert_eq!(parse_stamp("source-sha256=\nschema=3\nslang=x\n"), None);
}

#[test]
fn a_helper_with_another_schema_or_no_readable_version_is_stale() {
    let (_scratch, exe, stamp) = stamped_helper("staleness-schema", "3", "2026.20");
    let sdk = Ok("2026.20".to_owned());
    let reason = |output: &'static str| staleness(&exe, &stamp, &sdk, &reports(output)).unwrap();
    // The helpers of the earlier schemas (a schema-2 helper from before mx.t0a) are rebuilt.
    for old in ["1", "2"] {
        let output: &'static str = match old {
            "1" => "gust-slang-reflect schema=1 slang=2026.20\n",
            _ => "gust-slang-reflect schema=2 slang=2026.20\n",
        };
        assert_eq!(
            reason(output),
            Some(format!(
                "it prints schema {old}, but the source prints schema 3"
            ))
        );
    }
    for unreadable in ["", "slangc 2026.20\n", "gust-slang-reflect schema=3\n"] {
        let reason = reason(unreadable).unwrap();
        assert!(
            reason.contains("--version") || reason.contains("unexpected"),
            "{unreadable:?}: {reason}"
        );
    }
    let reason = reason("gust-slang-reflect schema=4 slang=2026.20\n").unwrap();
    assert!(reason.contains("schema 4"), "{reason}");
    let failing = |_: &Path| Err::<String, String>("exit 3".to_owned());
    assert_eq!(
        staleness(&exe, &stamp, &sdk, &failing),
        Ok(Some("it cannot report its version (exit 3)".to_owned()))
    );
    // An SDK that cannot name its version never makes a helper current.
    let unknown = Err("no slangc".to_owned());
    let current = reports("gust-slang-reflect schema=3 slang=2026.20\n");
    assert_eq!(
        staleness(&exe, &stamp, &unknown, &current),
        Ok(Some(
            "the SDK's slangc cannot report its version (no slangc)".to_owned()
        ))
    );
}

#[test]
fn a_missing_helper_is_stale_without_being_run_and_force_skips_the_judgement() {
    let scratch = Scratch::new("staleness-files");
    let stamp = scratch.0.join(HELPER_STAMP);
    let exe = scratch.0.join(executable_name(HELPER_NAME));
    let sdk = Ok("2026.20".to_owned());
    let never = |_: &Path| -> Result<String, String> { panic!("the helper must not be run") };
    assert_eq!(
        staleness(&exe, &stamp, &sdk, &never),
        Ok(Some("it is not built yet".to_owned()))
    );
    // A helper without a stamp is judged before it is run.
    fs::write(&exe, "x").unwrap();
    let reason = staleness(&exe, &stamp, &sdk, &never).unwrap().unwrap();
    assert!(reason.starts_with("it has no build stamp"), "{reason}");
    // `--force` skips the judgement altogether.
    let options = BuildOptions {
        force: true,
        ..BuildOptions::default()
    };
    assert_eq!(up_to_date(&options, &exe, &stamp, &scratch.0), Ok(false));
}

#[test]
fn a_build_without_an_sdk_or_an_output_directory_says_what_is_missing() {
    let scratch = Scratch::new("build-missing");
    let error = build(&BuildOptions::default()).unwrap_err();
    assert!(error.contains("--sdk-root"), "{error}");
    let error = build(&BuildOptions {
        sdk_root: Some(scratch.0.clone()),
        ..BuildOptions::default()
    })
    .unwrap_err();
    assert!(error.contains("output directory"), "{error}");
    let error = build(&BuildOptions {
        force: false,
        sdk_root: Some(scratch.0.join("no-sdk")),
        out_dir: Some(scratch.0.join("out")),
    })
    .unwrap_err();
    assert!(error.contains("slang.h was not found"), "{error}");
}

#[test]
fn a_cl_that_could_not_be_launched_gets_the_vcvars_hint_and_other_errors_stay_as_they_are() {
    let launch = with_cl_launch_hint("cl", "could not launch cl: not found".to_owned());
    assert!(
        launch.starts_with("could not launch cl: not found"),
        "{launch}"
    );
    assert!(launch.contains("Developer prompt"), "{launch}");
    assert!(launch.contains("unset CXX"), "{launch}");
    let failed = "cl exited with exit code: 2".to_owned();
    assert_eq!(with_cl_launch_hint("cl", failed.clone()), failed);
}
