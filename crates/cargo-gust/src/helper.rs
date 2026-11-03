//! `cargo gust slang build-helper`: build `gust-slang-reflect`, the native reflection helper, against a Slang install.
//!
//! The helper's C++ source ships inside this package (`assets/slang-layout.cpp`, a byte copy of
//! `scripts/probes/slang-layout.cpp`, which stays the source of truth; a test fails when they differ), so a published
//! tool has it. The logic of `cargo xtask build-slang-reflect` lives here since slice m4.f: xtask runs this command.
//!
//! The C++ compiler is the first of `cl`, `clang++`, `c++`, `g++` found on `PATH` (the `CXX` variable, when set, is
//! used instead; a `CXX` whose file name is `cl` or `cl.exe` gets MSVC arguments, any other gets GCC style ones). On
//! Windows, when none is on `PATH`, `vcvars64.bat` of the newest Visual Studio with the C++ tools is
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
    sha256::sha256_hex,
    slang::{slangc_in, slangc_version},
    tools::{self, executable_name, find_on_path},
};

/// The helper's C++ source, from the packaged copy.
pub(crate) const HELPER_SOURCE: &str = include_str!("../assets/slang-layout.cpp");

/// The helper's file name without the platform suffix.
pub(crate) const HELPER_NAME: &str = "gust-slang-reflect";

/// The reflection schema the packaged source prints (`gust-slang-reflect schema=<n> slang=<tag>`); a test ties it to
/// the source.
pub(crate) const HELPER_SCHEMA: &str = "3";

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
    /// MSVC `cl`, as the program to launch: `cl` when it was found on `PATH`, or the value of `CXX` (a bare name or a
    /// path) whose file stem is `cl`.
    Cl(String),
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
        // `cl` takes MSVC style arguments; every other `CXX` is taken for a GCC or Clang style driver (`clang-cl`
        // included: it is a different file stem, and it also accepts the driver arguments).
        return Ok(if is_cl_program(cxx) {
            Compiler::Cl(cxx.to_owned())
        } else {
            Compiler::Driver(cxx.to_owned())
        });
    }
    for name in COMPILER_SEARCH_ORDER {
        // A MinGW `c++` or `g++` cannot link the MSVC import library of the Slang release, so Windows takes only
        // `cl` and `clang++` (clang for the MSVC target) from PATH, then `vcvars64.bat`.
        if windows && !matches!(name, "cl" | "clang++") {
            continue;
        }
        if find_on_path(&executable_name(name), path_var).is_some() {
            return Ok(if name == "cl" {
                Compiler::Cl("cl".to_owned())
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

/// Whether a `CXX` value names MSVC `cl`: its file stem is `cl`, in any case, with or without `.exe`, bare or after a
/// path. Both separators count on every platform, so that a Windows path is judged the same way everywhere.
fn is_cl_program(program: &str) -> bool {
    let name = program.rsplit(['/', '\\']).next().unwrap_or(program);
    let stem = match name.len().checked_sub(4) {
        Some(split)
            if name.is_char_boundary(split) && name[split..].eq_ignore_ascii_case(".exe") =>
        {
            &name[..split]
        }
        _ => name,
    };
    stem.eq_ignore_ascii_case("cl")
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

/// The build stamp, written next to the helper after a good build.
pub(crate) const HELPER_STAMP: &str = "gust-slang-reflect.stamp";

/// SHA-256 of the packaged probe source, with line endings normalised (a Windows checkout may convert the file), so
/// that every cargo-gust built from the same probe source computes the same hash.
pub(crate) fn source_sha256() -> String {
    sha256_hex(HELPER_SOURCE.replace("\r\n", "\n").as_bytes())
}

/// The text of a build stamp: the hash of the probe source, and the schema and Slang tag the helper reported.
pub(crate) fn stamp_text(schema: &str, tag: &str) -> String {
    format!(
        "gust-slang-reflect build stamp\nsource-sha256={}\nschema={schema}\nslang={tag}\n",
        source_sha256()
    )
}

/// The three values of a stamp: the source hash, the schema and the Slang tag.
fn parse_stamp(text: &str) -> Option<(String, String, String)> {
    let value = |key: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(key))
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    };
    Some((
        value("source-sha256=")?,
        value("schema=")?,
        value("slang=")?,
    ))
}

/// Why the helper at `exe` has to be rebuilt for an SDK whose `slangc -version` is `sdk_tag`, or `None` when it is
/// current. `stamp` is the build stamp beside it; `version_of` runs `<helper> --version` and returns what it printed.
///
/// Freshness is judged by content, and no file time plays a part: a release ships its libraries with the times of its
/// own build, so a helper built against an older Slang looked newer than the 2026.20 libraries and was reported "up to
/// date" (m4.r review, S6), and a copied probe source or helper carries a time that says nothing about its bytes. The
/// helper is current when
///
/// 1. the stamp written after its build holds the SHA-256 of the packaged probe source, so it was built from this
///    source (a missing or unreadable stamp means the source it came from is unknown);
/// 2. it prints the schema of that source ([`HELPER_SCHEMA`]);
/// 3. its Slang build tag, which it reads from the library it loads (the copy beside it), is the one the SDK's own
///    `slangc -version` prints; and
/// 4. the stamp and the helper agree on schema and tag, so a replacement that prints another schema or tag is
///    rebuilt (one that prints the same schema and tag, say a copied-in helper or a rebuild by an older
///    cargo-gust that left the stamp in place, is not told apart).
///
/// Two cargo-gust versions with the same probe source therefore leave each other's helper alone. An SDK whose
/// `slangc` cannot report a version is never judged current. Two builds with the same tag but different library bytes
/// are not told apart; a tag names a build, and `--force` is for that case.
pub(crate) fn staleness(
    exe: &Path,
    stamp: &Path,
    sdk_tag: &Result<String, String>,
    version_of: &dyn Fn(&Path) -> Result<String, String>,
) -> Result<Option<String>, String> {
    if !exe.is_file() {
        return Ok(Some("it is not built yet".to_owned()));
    }
    let Ok(stamp_text) = fs::read_to_string(stamp) else {
        return Ok(Some(format!(
            "it has no build stamp ({}), so the probe source it was built from is unknown",
            stamp.display()
        )));
    };
    let Some((stamped_source, stamped_schema, stamped_tag)) = parse_stamp(&stamp_text) else {
        return Ok(Some(format!(
            "its build stamp ({}) cannot be read",
            stamp.display()
        )));
    };
    let source = source_sha256();
    if stamped_source != source {
        return Ok(Some(format!(
            "it was built from another probe source (stamp {}, this tool {})",
            stamped_source.get(..16).unwrap_or(&stamped_source),
            source.get(..16).unwrap_or(&source)
        )));
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
    if stamped_schema != schema || stamped_tag != tag {
        return Ok(Some(format!(
            "its build stamp says schema {stamped_schema} and Slang `{stamped_tag}`, but it prints schema {schema} \
             and Slang `{tag}`: it was replaced after the build"
        )));
    }
    Ok(None)
}

/// Write the packaged source into `dir` and return its path. An existing file with the same bytes is left alone.
fn write_source(dir: &Path) -> Result<PathBuf, String> {
    let path = dir.join("gust-slang-reflect.cpp");
    if fs::read(&path).ok().as_deref() != Some(HELPER_SOURCE.as_bytes()) {
        fs::write(&path, HELPER_SOURCE).map_err(|error| io_error(&path, &error))?;
    }
    Ok(path)
}

/// Run `<exe> --version` and return the schema and the Slang build tag it reports.
fn helper_report(exe: &Path) -> Result<(String, String), String> {
    tools::parse_helper_report(&tools::helper_version_output(exe)?)
}

/// Decide whether the build is skipped: it is, unless `force` is set or [`staleness`] gives a reason. A reason is
/// printed together with the helper it is about, so a rebuild of a shared directory is never silent.
fn up_to_date(
    options: &BuildOptions,
    exe: &Path,
    stamp: &Path,
    sdk_root: &Path,
) -> Result<bool, String> {
    if options.force {
        return Ok(false);
    }
    let sdk_tag = slangc_version(&slangc_in(sdk_root));
    match staleness(exe, stamp, &sdk_tag, &tools::helper_version_output)? {
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
    let stamp = out_dir.join(HELPER_STAMP);
    let current = up_to_date(options, &exe, &stamp, sdk_root)?;
    if !current {
        // A rebuild starts without a stamp, so a build that fails half way never leaves one that vouches for it.
        match fs::remove_file(&stamp) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(io_error(&stamp, &error)),
        }
    }
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
    let (schema, tag) = helper_report(&exe)?;
    if !current {
        fs::write(&stamp, stamp_text(&schema, &tag)).map_err(|error| io_error(&stamp, &error))?;
    }
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
        Compiler::Cl(program) => run(&program, &msvc_args(include, source, &obj, exe, &library))
            .map_err(|error| with_cl_launch_hint(&program, error))?,
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
        // `cl` on a non-Windows host is not something this build can drive with GCC style arguments.
        Compiler::Cl(_) | Compiler::ClViaVcvars => {
            return Err(
                "found `cl` (on PATH or as CXX), which this platform cannot use; set CXX to a GCC or Clang style driver"
                    .to_owned(),
            );
        }
    };
    run(
        &program,
        &driver_args(include, source, &library, Some(lib_dir), exe),
    )
}

/// A `cl` that could not even be started (a `CXX` naming it when it is not on `PATH`) gets the way out: `CXX` is
/// taken as given, so the `vcvars64.bat` fallback only runs when `CXX` is unset.
fn with_cl_launch_hint(program: &str, error: String) -> String {
    if error.starts_with("could not launch") {
        format!(
            "{error}; run from a Developer prompt (so `{program}` is on PATH), or unset CXX to use the vcvars fallback"
        )
    } else {
        error
    }
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
