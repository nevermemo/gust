//! `cargo gust slang build-helper`: build `gust-slang-reflect`, the native reflection helper, against a Slang install.
//!
//! The helper's C++ source ships inside this package (`assets/slang-layout.cpp`, a byte copy of
//! `scripts/probes/slang-layout.cpp`, which stays the source of truth; a test fails when they differ), so a published
//! tool has it. The logic of `cargo xtask build-slang-reflect` lives here since slice m4.f: xtask runs this command.
//!
//! The C++ compiler is the first of `cl`, `clang++`, `c++`, `g++` found on `PATH` (the `CXX` variable, when set, is
//! used instead). On Windows, when none is on `PATH`, `vcvars64.bat` of the newest Visual Studio with the C++ tools is
//! run once (see `msvc.rs`) and `cl` is taken from the environment it produces.
//!
//! The helper reads the Slang build tag from the library it loads (`--version` prints `slang=<tag>`), so it reports
//! the Slang it runs against. `doctor` and `build` compare that tag with `slangc -version`.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use crate::{
    msvc,
    slang::{slangc_in, slangc_version},
    tools::{self, executable_name, find_on_path},
};

/// The helper's C++ source, from the packaged copy.
pub(crate) const HELPER_SOURCE: &str = include_str!("../assets/slang-layout.cpp");

/// The helper's file name without the platform suffix.
pub(crate) const HELPER_NAME: &str = "gust-slang-reflect";

/// The reflection schema the packaged source prints (`gust-slang-reflect schema=<n> slang=<tag>`); a test ties it to
/// the source.
pub(crate) const HELPER_SCHEMA: &str = "2";

/// The compilers to look for, in order.
pub(crate) const COMPILER_SEARCH_ORDER: [&str; 4] = ["cl", "clang++", "c++", "g++"];

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct BuildOptions {
    /// Rebuild even when the helper is current (see [`staleness`]). It overwrites the helper in the output
    /// directory, which other builds may share, so it is rarely the right fix.
    pub(crate) force: bool,
    /// The Slang SDK (an extracted release or a Vulkan SDK): `include/`, `lib/` and, on Windows, `bin/`.
    pub(crate) sdk_root: Option<PathBuf>,
    pub(crate) out_dir: Option<PathBuf>,
}

/// The compiler a build will use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Compiler {
    /// `cl.exe` found on `PATH`.
    Cl,
    /// `cl.exe` from the environment `vcvars64.bat` produces (Windows, nothing on `PATH`).
    ClViaVcvars,
    /// A GCC or Clang style driver, by program name or path.
    Driver(String),
}

/// Choose the compiler. `cxx` is the `CXX` variable. `windows` selects the Windows fallback; it is a parameter so
/// that the choice is tested on every platform.
pub(crate) fn choose_compiler(
    cxx: Option<&str>,
    path_var: Option<&std::ffi::OsStr>,
    windows: bool,
) -> Result<Compiler, String> {
    if let Some(cxx) = cxx.filter(|cxx| !cxx.is_empty()) {
        return Ok(Compiler::Driver(cxx.to_owned()));
    }
    for name in COMPILER_SEARCH_ORDER {
        // A MinGW `c++` or `g++` cannot link the MSVC import library of the Slang release, so Windows takes only
        // `cl` and `clang++` (clang for the MSVC target) from PATH, then `vcvars64.bat`.
        if windows && !matches!(name, "cl" | "clang++") {
            continue;
        }
        if find_on_path(&executable_name(name), path_var).is_some() {
            return Ok(if name == "cl" {
                Compiler::Cl
            } else {
                Compiler::Driver(name.to_owned())
            });
        }
    }
    if windows {
        Ok(Compiler::ClViaVcvars)
    } else {
        Err(format!(
            "no C++ compiler found (looked for {} on PATH); install one, or set CXX",
            COMPILER_SEARCH_ORDER.join(", ")
        ))
    }
}

/// The `cl` command line (after the program name).
pub(crate) fn msvc_args(
    include: &Path,
    source: &Path,
    obj: &Path,
    exe: &Path,
    library: &Path,
) -> Vec<String> {
    vec![
        "/nologo".into(),
        "/std:c++17".into(),
        "/EHsc".into(),
        "/W4".into(),
        "/WX".into(),
        format!("/I{}", include.display()),
        source.display().to_string(),
        format!("/Fo{}", obj.display()),
        format!("/Fe{}", exe.display()),
        "/link".into(),
        library.display().to_string(),
    ]
}

/// The GCC or Clang style command line (after the program name). The Slang headers are a system include, so that
/// `-Werror` judges only the helper's own source. `lib_dir` becomes the run-time search path of the helper where the
/// platform has one (not on Windows, which finds the DLL beside the executable).
pub(crate) fn driver_args(
    include: &Path,
    source: &Path,
    library: &Path,
    lib_dir: Option<&Path>,
    exe: &Path,
) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "-std=c++17".into(),
        "-Wall".into(),
        "-Wextra".into(),
        "-Werror".into(),
        "-isystem".into(),
        include.display().to_string(),
        source.display().to_string(),
        library.display().to_string(),
    ];
    if let Some(lib_dir) = lib_dir {
        args.push(format!("-Wl,-rpath,{}", lib_dir.display()));
    }
    args.push("-o".into());
    args.push(exe.display().to_string());
    args
}

/// The header directory of an SDK: `include/slang/slang.h` or `include/slang.h`.
pub(crate) fn include_dir(sdk_root: &Path) -> Result<PathBuf, String> {
    [sdk_root.join("include/slang"), sdk_root.join("include")]
        .into_iter()
        .find(|path| path.join("slang.h").is_file())
        .ok_or_else(|| {
            format!(
                "slang.h was not found under {}/include[/slang].",
                sdk_root.display()
            )
        })
}

fn io_error(path: &Path, error: &std::io::Error) -> String {
    format!("{}: {error}", path.display())
}

/// Why the helper at `exe` has to be rebuilt for an SDK whose `slangc -version` is `sdk_tag`, or `None` when it is
/// current. `version_of` runs `<helper> --version` and returns what it printed.
///
/// Freshness is judged by content, not by the times of the SDK's libraries: a release ships its libraries with the
/// times of its own build, so a helper built against an older Slang looked newer than the 2026.20 libraries and was
/// reported "up to date" (m4.r review, S6). The helper reads its Slang build tag from the library it loads (the copy
/// beside it), so the smallest sound rule is that it is current when
///
/// 1. it is not older than the packaged source (the source file keeps its time while its bytes are unchanged);
/// 2. it prints the schema of that source ([`HELPER_SCHEMA`]); and
/// 3. its Slang build tag is the one the SDK's own `slangc -version` prints.
///
/// An SDK whose `slangc` cannot report a version is never judged current. Two builds with the same tag but different
/// library bytes are not told apart; a tag names a build, and `--force` is for that case.
pub(crate) fn staleness(
    exe: &Path,
    source: &Path,
    sdk_tag: &Result<String, String>,
    version_of: &dyn Fn(&Path) -> Result<String, String>,
) -> Result<Option<String>, String> {
    if !exe.is_file() {
        return Ok(Some("it is not built yet".to_owned()));
    }
    if !newer_than_all(exe, &[source])? {
        return Ok(Some("it is older than its source".to_owned()));
    }
    let sdk_tag = match sdk_tag {
        Ok(tag) => tag,
        Err(error) => {
            return Ok(Some(format!(
                "the SDK's slangc cannot report its version ({error})"
            )));
        }
    };
    let output = match version_of(exe) {
        Ok(output) => output,
        Err(error) => return Ok(Some(format!("it cannot report its version ({error})"))),
    };
    let (schema, tag) = match tools::parse_helper_report(&output) {
        Ok(report) => report,
        Err(error) => return Ok(Some(error)),
    };
    if schema != HELPER_SCHEMA {
        return Ok(Some(format!(
            "it prints schema {schema}, but the source prints schema {HELPER_SCHEMA}"
        )));
    }
    if tag != *sdk_tag {
        return Ok(Some(format!(
            "it was built against Slang `{tag}`, but the SDK is Slang `{sdk_tag}`"
        )));
    }
    Ok(None)
}

/// Whether `output` exists and is not older than any of `inputs`.
fn newer_than_all(output: &Path, inputs: &[&Path]) -> Result<bool, String> {
    if !output.is_file() {
        return Ok(false);
    }
    let modified = |path: &Path| {
        fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .map_err(|error| io_error(path, &error))
    };
    let output_time = modified(output)?;
    for input in inputs {
        if output_time < modified(input)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Write the packaged source into `dir` and return its path. An existing file with the same bytes is left alone, so
/// its modification time keeps meaning "when the source last changed".
fn write_source(dir: &Path) -> Result<PathBuf, String> {
    let path = dir.join("gust-slang-reflect.cpp");
    if fs::read(&path).ok().as_deref() != Some(HELPER_SOURCE.as_bytes()) {
        fs::write(&path, HELPER_SOURCE).map_err(|error| io_error(&path, &error))?;
    }
    Ok(path)
}

/// Run `<exe> --version` and return the Slang build tag it reports.
fn helper_version(exe: &Path) -> Result<String, String> {
    tools::helper_slang_version(exe)
}

/// Decide whether the build is skipped: it is, unless `force` is set or [`staleness`] gives a reason. A reason is
/// printed together with the helper it is about, so a rebuild of a shared directory is never silent.
fn up_to_date(
    options: &BuildOptions,
    exe: &Path,
    source: &Path,
    sdk_root: &Path,
) -> Result<bool, String> {
    if options.force {
        return Ok(false);
    }
    let sdk_tag = slangc_version(&slangc_in(sdk_root));
    match staleness(exe, source, &sdk_tag, &tools::helper_version_output)? {
        None => Ok(true),
        Some(reason) => {
            if exe.is_file() {
                println!("slang: rebuilding {}: {reason}", exe.display());
            }
            Ok(false)
        }
    }
}

/// Build the helper. Returns its path and the Slang build tag it reports.
pub(crate) fn build(options: &BuildOptions) -> Result<(PathBuf, String), String> {
    let sdk_root = options
        .sdk_root
        .as_deref()
        .ok_or("build-helper needs a Slang SDK: pass --sdk-root, or install the pin first")?;
    let out_dir = options
        .out_dir
        .as_deref()
        .ok_or("build-helper needs an output directory")?;
    fs::create_dir_all(out_dir).map_err(|error| io_error(out_dir, &error))?;
    let include = include_dir(sdk_root)?;
    let source = write_source(out_dir)?;
    let lib_dir = sdk_root.join("lib");
    let exe = out_dir.join(executable_name(HELPER_NAME));
    let current = up_to_date(options, &exe, &source, sdk_root)?;
    if cfg!(windows) {
        build_windows(
            current,
            &source,
            &include,
            &lib_dir,
            &sdk_root.join("bin"),
            out_dir,
            &exe,
        )?;
    } else {
        build_unix(current, &source, &include, &lib_dir, &exe)?;
    }
    let tag = helper_version(&exe)?;
    Ok((exe, tag))
}

fn build_windows(
    current: bool,
    source: &Path,
    include: &Path,
    lib_dir: &Path,
    bin_dir: &Path,
    out_dir: &Path,
    exe: &Path,
) -> Result<(), String> {
    let library = ["slang-compiler.lib", "slang.lib"]
        .iter()
        .map(|name| lib_dir.join(name))
        .find(|path| path.is_file())
        .ok_or_else(|| format!("No Slang import library in {}.", lib_dir.display()))?;
    let dll = bin_dir.join(format!(
        "{}.dll",
        library
            .file_stem()
            .and_then(|name| name.to_str())
            .ok_or("invalid Slang library name")?
    ));
    if !dll.is_file() {
        return Err(format!(
            "Missing matching compiler library: {}",
            dll.display()
        ));
    }
    if current {
        println!("slang: the helper is up to date");
        return Ok(());
    }
    let compiler = choose_compiler(
        env::var("CXX").ok().as_deref(),
        env::var_os("PATH").as_deref(),
        true,
    )?;
    let obj = out_dir.join("gust-slang-reflect.obj");
    match compiler {
        Compiler::Cl => run("cl", &msvc_args(include, source, &obj, exe, &library))?,
        Compiler::ClViaVcvars => {
            let environment = msvc::capture_vcvars_environment(&msvc::find_vcvars64()?)?;
            msvc::run_cl_in_environment(
                &environment,
                &msvc_args(include, source, &obj, exe, &library),
            )?;
        }
        Compiler::Driver(program) => {
            run(&program, &driver_args(include, source, &library, None, exe))?
        }
    }
    fs::copy(
        &dll,
        out_dir.join(dll.file_name().ok_or("invalid dll name")?),
    )
    .map_err(|error| io_error(&dll, &error))?;
    for name in ["slang-glsl-module.dll", "slang-glslang.dll", "slang-rt.dll"] {
        let dependency = bin_dir.join(name);
        if dependency.is_file() {
            fs::copy(&dependency, out_dir.join(name))
                .map_err(|error| io_error(&dependency, &error))?;
        }
    }
    Ok(())
}

fn build_unix(
    current: bool,
    source: &Path,
    include: &Path,
    lib_dir: &Path,
    exe: &Path,
) -> Result<(), String> {
    let extension = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let library = [
        format!("libslang-compiler.{extension}"),
        format!("libslang.{extension}"),
    ]
    .into_iter()
    .map(|name| lib_dir.join(name))
    .find(|path| path.is_file())
    .ok_or_else(|| format!("No Slang shared library in {}.", lib_dir.display()))?;
    if current {
        println!("slang: the helper is up to date");
        return Ok(());
    }
    let compiler = choose_compiler(
        env::var("CXX").ok().as_deref(),
        env::var_os("PATH").as_deref(),
        false,
    )?;
    let program = match compiler {
        Compiler::Driver(program) => program,
        // `cl` on PATH of a non-Windows host is not something this build can drive with GCC style arguments.
        Compiler::Cl | Compiler::ClViaVcvars => {
            return Err("found `cl` on PATH, which this platform cannot use; set CXX".to_owned());
        }
    };
    run(
        &program,
        &driver_args(include, source, &library, Some(lib_dir), exe),
    )
}

/// Run a compiler with inherited output.
fn run(program: &str, args: &[String]) -> Result<(), String> {
    let status = Command::new(program)
        .args(args)
        .status()
        .map_err(|error| format!("could not launch {program}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "{program} failed (exit {}); the arguments were: {}",
            status.code().unwrap_or(-1),
            args.join(" ")
        ))
    }
}

#[cfg(test)]
#[path = "helper_tests.rs"]
mod tests;
