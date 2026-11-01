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
    assert!(HELPER_SOURCE.contains("gust-slang-reflect schema=1 slang="));
}

#[test]
fn the_compiler_is_the_first_of_cl_clang_c_plus_plus_and_g_plus_plus_on_path() {
    assert_eq!(COMPILER_SEARCH_ORDER, ["cl", "clang++", "c++", "g++"]);
    let scratch = Scratch::new("compiler-order");
    let all = path_with(&scratch, &[&["g++", "c++"], &["clang++"], &["cl"]]);
    assert_eq!(choose_compiler(None, Some(&all), false), Ok(Compiler::Cl));
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
        Ok(Compiler::Cl)
    );
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
    // A stale copy is replaced; a current one is not touched (its time stays the time of the last change).
    fs::write(&path, "stale").unwrap();
    write_source(&scratch.0).unwrap();
    assert_eq!(fs::read(&path).unwrap(), HELPER_SOURCE.as_bytes());
    let before = fs::metadata(&path).unwrap().modified().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(30));
    write_source(&scratch.0).unwrap();
    assert_eq!(fs::metadata(&path).unwrap().modified().unwrap(), before);
}

#[test]
fn an_output_is_up_to_date_only_when_it_exists_and_no_input_is_newer() {
    let scratch = Scratch::new("newer");
    let output = scratch.0.join("out");
    let input = scratch.0.join("in");
    fs::write(&input, "x").unwrap();
    assert_eq!(newer_than_all(&output, &[&input]), Ok(false));
    std::thread::sleep(std::time::Duration::from_millis(30));
    fs::write(&output, "x").unwrap();
    assert_eq!(newer_than_all(&output, &[&input]), Ok(true));
    std::thread::sleep(std::time::Duration::from_millis(30));
    fs::write(&input, "y").unwrap();
    assert_eq!(newer_than_all(&output, &[&input]), Ok(false));
    assert!(newer_than_all(&output, &[&scratch.0.join("missing")]).is_err());
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
