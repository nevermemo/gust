//! The command line: `cargo gust <command>` (cargo runs `cargo-gust gust <command>`) or `cargo-gust <command>`.

use std::path::PathBuf;

use gust::Format;

pub(crate) const USAGE: &str = "\
cargo gust: export precompiled GUST kernels (D45, D46)

USAGE:
    cargo gust init   [-p <crate>] [--manifest-path <path>]
    cargo gust build  [-p <crate>] [--manifest-path <path>] [--format wgsl|spirv]... [--allow-slang-mismatch]
    cargo gust check  [-p <crate>] [--manifest-path <path>] [--recompile] [--allow-slang-mismatch]
    cargo gust run    [cargo run arguments]
    cargo gust test   [cargo test arguments]
    cargo gust doctor
    cargo gust slang install
    cargo gust slang locate [--allow-slang-mismatch]
    cargo gust slang build-helper [--force] [--sdk-root <dir>] [--out-dir <dir>]
    cargo gust help

COMMANDS:
    init     create gust-artifacts/manifest.json (an empty manifest) in the crate
    build    collect the crate's #[gpu] kernels, compile them with Slang, verify them (D17) and write gust-artifacts/
             WGSL is always built; --format spirv adds SPIR-V (checked with spirv-val when it is on PATH)
    check    fail unless gust-artifacts/ is exactly what a build of the current sources gives
             --recompile also recompiles and compares the bytes; --allow-slang-mismatch accepts another Slang build
             a plain check compares the manifest's Slang build with the helper's and does not run slangc; build and
             check --recompile require the pinned slangc and a helper of the same build
    run      cargo run with the route variables of the package's gust.toml (D65 to D67); test is cargo test likewise
             -p, --manifest-path, --release, --profile and --target are read (the package and the build profile), and
             every argument is passed to cargo unchanged. build and check read gust.toml too. A gust.toml beside the
             package's Cargo.toml routes the tags that the crates of the build declare, and the resolved table is
             written to gust-artifacts/routes.json; without a gust.toml nothing changes
    doctor   show Slang, the native helper, the build profile and how to install this tool
    slang    manage the pinned Slang (the authors who export need it; consumers and production builds do not)
               install       download the pinned release, verify its SHA-256 and unpack it into the cache
               locate        print the slangc a build would use: the cache (GUST_SLANG_HOME if set), then PATH
               build-helper  build the native reflection helper against that Slang (needs a C++ compiler)
             --allow-slang-mismatch uses a slangc of another version, printed and never silent
GUST_SLANG_HOME names the Slang cache directory (default: <CARGO_HOME or ~/.cargo>/gust/slang).

EXIT CODES: 0 success; 1 check found problems; 2 the command could not run (a failed compile, no kernels, bad input).
";

/// Which package a command works on.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Target {
    /// `-p <crate>`; `None` means the package of the current directory (or of `--manifest-path`).
    pub(crate) package: Option<String>,
    pub(crate) manifest_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BuildOptions {
    pub(crate) target: Target,
    /// What `--format` asked for, in order; WGSL is added by the build.
    pub(crate) formats: Vec<Format>,
    pub(crate) allow_slang_mismatch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CheckOptions {
    pub(crate) target: Target,
    pub(crate) recompile: bool,
    pub(crate) allow_slang_mismatch: bool,
}

/// `cargo gust run` and `cargo gust test` (m8.r2): cargo's own arguments, read for the package and the build profile
/// and passed on unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CargoOptions {
    pub(crate) target: Target,
    /// Every argument after the command, as given.
    pub(crate) arguments: Vec<String>,
}

/// `cargo gust slang <command>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SlangCommand {
    Install,
    Locate {
        allow_mismatch: bool,
    },
    BuildHelper {
        force: bool,
        sdk_root: Option<PathBuf>,
        out_dir: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Command {
    Init(Target),
    Build(BuildOptions),
    Check(CheckOptions),
    Run(CargoOptions),
    Test(CargoOptions),
    Doctor,
    Slang(SlangCommand),
    Help,
    Version,
}

/// Drop the program name and the `gust` that cargo inserts when it runs `cargo gust ...` as `cargo-gust gust ...`.
pub(crate) fn strip_shim(args: &[String]) -> Vec<String> {
    let mut rest = args.iter().skip(1).cloned().collect::<Vec<_>>();
    if rest.first().is_some_and(|first| first == "gust") {
        rest.remove(0);
    }
    rest
}

/// Parse the arguments that [`strip_shim`] left.
pub(crate) fn parse(args: &[String]) -> Result<Command, String> {
    let Some((command, rest)) = args.split_first() else {
        return Ok(Command::Help);
    };
    match command.as_str() {
        "help" | "--help" | "-h" => Ok(Command::Help),
        "--version" | "-V" => Ok(Command::Version),
        "init" => {
            let parsed = options(rest, &["-p", "--package", "--manifest-path"], &[])?;
            Ok(Command::Init(parsed.target))
        }
        "build" => {
            let parsed = options(
                rest,
                &["-p", "--package", "--manifest-path", "--format"],
                &["--allow-slang-mismatch"],
            )?;
            Ok(Command::Build(BuildOptions {
                allow_slang_mismatch: parsed.flags.contains(&"--allow-slang-mismatch"),
                target: parsed.target,
                formats: parsed.formats,
            }))
        }
        "check" => {
            let parsed = options(
                rest,
                &["-p", "--package", "--manifest-path"],
                &["--recompile", "--allow-slang-mismatch"],
            )?;
            Ok(Command::Check(CheckOptions {
                recompile: parsed.flags.contains(&"--recompile"),
                allow_slang_mismatch: parsed.flags.contains(&"--allow-slang-mismatch"),
                target: parsed.target,
            }))
        }
        "run" => cargo_options(rest).map(Command::Run),
        "test" => cargo_options(rest).map(Command::Test),
        "doctor" => {
            options(rest, &[], &[])?;
            Ok(Command::Doctor)
        }
        "slang" => slang(rest).map(Command::Slang),
        other => Err(format!("unknown command `{other}`")),
    }
}

/// Parse what follows `run` or `test`: cargo's arguments. The package (`-p`, `--package`) and the manifest
/// (`--manifest-path`) are read before the first `--`, in cargo's spellings (`-p x`, `-px`, `--package x`,
/// `--package=x`); everything is kept for cargo.
fn cargo_options(args: &[String]) -> Result<CargoOptions, String> {
    let mut target = Target::default();
    let mut index = 0;
    while let Some(argument) = args.get(index) {
        index += 1;
        if argument == "--" {
            break;
        }
        let (name, inline) = match argument.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value.to_owned())),
            _ => (argument.as_str(), None),
        };
        let attached = argument
            .strip_prefix("-p")
            .filter(|rest| !rest.is_empty() && !rest.starts_with('-'));
        if !matches!(name, "-p" | "--package" | "--manifest-path") && attached.is_none() {
            continue;
        }
        let value = match (inline, attached) {
            (Some(value), _) => value,
            (None, Some(value)) => value.to_owned(),
            (None, None) => {
                let value = args
                    .get(index)
                    .ok_or_else(|| format!("`{name}` needs a value"))?;
                index += 1;
                value.clone()
            }
        };
        if value.is_empty() {
            return Err(format!("`{name}` needs a non-empty value"));
        }
        if name == "--manifest-path" {
            if target.manifest_path.replace(PathBuf::from(value)).is_some() {
                return Err("`--manifest-path` was given twice".to_owned());
            }
        } else if target.package.replace(value).is_some() {
            return Err(
                "`-p` was given twice; `cargo gust run` and `test` work on one package".to_owned(),
            );
        }
    }
    Ok(CargoOptions {
        target,
        arguments: args.to_vec(),
    })
}

/// Parse what follows `slang`.
fn slang(args: &[String]) -> Result<SlangCommand, String> {
    let Some((command, rest)) = args.split_first() else {
        return Err("`slang` needs a command: install, locate or build-helper".to_owned());
    };
    match command.as_str() {
        "install" => {
            options(rest, &[], &[])?;
            Ok(SlangCommand::Install)
        }
        "locate" => {
            let parsed = options(rest, &[], &["--allow-slang-mismatch"])?;
            Ok(SlangCommand::Locate {
                allow_mismatch: parsed.flags.contains(&"--allow-slang-mismatch"),
            })
        }
        "build-helper" => {
            let parsed = options(rest, &["--sdk-root", "--out-dir"], &["--force"])?;
            Ok(SlangCommand::BuildHelper {
                force: parsed.flags.contains(&"--force"),
                sdk_root: parsed.sdk_root,
                out_dir: parsed.out_dir,
            })
        }
        other => Err(format!(
            "unknown slang command `{other}` (use install, locate or build-helper)"
        )),
    }
}

struct Parsed {
    target: Target,
    formats: Vec<Format>,
    flags: Vec<&'static str>,
    sdk_root: Option<PathBuf>,
    out_dir: Option<PathBuf>,
}

/// Read `--name value`, `--name=value` and the listed bare flags.
fn options(args: &[String], valued: &[&str], bare: &[&'static str]) -> Result<Parsed, String> {
    let mut target = Target::default();
    let mut formats = Vec::new();
    let mut flags = Vec::new();
    let mut sdk_root = None;
    let mut out_dir = None;
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        index += 1;
        let (name, inline) = match argument.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value.to_owned())),
            _ => (argument.as_str(), None),
        };
        if let Some(flag) = bare.iter().find(|flag| **flag == name) {
            if inline.is_some() {
                return Err(format!("`{name}` takes no value"));
            }
            if !flags.contains(flag) {
                flags.push(*flag);
            }
            continue;
        }
        if !valued.contains(&name) {
            return Err(if name.starts_with('-') {
                format!("unknown option `{name}`")
            } else {
                format!("unexpected argument `{name}`")
            });
        }
        let value = match inline {
            Some(value) => value,
            None => {
                let value = args
                    .get(index)
                    .ok_or_else(|| format!("`{name}` needs a value"))?;
                index += 1;
                value.clone()
            }
        };
        if value.is_empty() {
            return Err(format!("`{name}` needs a non-empty value"));
        }
        match name {
            "-p" | "--package" => {
                if target.package.replace(value).is_some() {
                    return Err("`-p` was given twice".to_owned());
                }
            }
            "--manifest-path" => {
                if target.manifest_path.replace(PathBuf::from(value)).is_some() {
                    return Err("`--manifest-path` was given twice".to_owned());
                }
            }
            "--sdk-root" => {
                if sdk_root.replace(PathBuf::from(value)).is_some() {
                    return Err("`--sdk-root` was given twice".to_owned());
                }
            }
            "--out-dir" => {
                if out_dir.replace(PathBuf::from(value)).is_some() {
                    return Err("`--out-dir` was given twice".to_owned());
                }
            }
            _ => {
                for part in value.split(',') {
                    formats.push(match part {
                        "wgsl" => Format::Wgsl,
                        "spirv" => Format::Spirv,
                        other => {
                            return Err(format!("unknown format `{other}` (use wgsl or spirv)"));
                        }
                    });
                }
            }
        }
    }
    Ok(Parsed {
        target,
        formats,
        flags,
        sdk_root,
        out_dir,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(line: &str) -> Vec<String> {
        line.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn the_shim_accepts_cargo_gust_and_cargo_dash_gust() {
        assert_eq!(
            strip_shim(&words("cargo-gust gust build -p kernels")),
            words("build -p kernels")
        );
        assert_eq!(
            strip_shim(&words("cargo-gust build -p kernels")),
            words("build -p kernels")
        );
        assert_eq!(
            strip_shim(&words("C:/bin/cargo-gust.exe gust")),
            Vec::<String>::new()
        );
        assert_eq!(strip_shim(&words("cargo-gust")), Vec::<String>::new());
        // Only one `gust` is the shim; a second one is the (unknown) command.
        assert_eq!(strip_shim(&words("cargo-gust gust gust")), words("gust"));
        assert_eq!(parse(&words("gust")).unwrap_err(), "unknown command `gust`");
        assert_eq!(strip_shim(&[]), Vec::<String>::new());
    }

    #[test]
    fn commands_and_their_options_parse() {
        assert_eq!(parse(&[]).unwrap(), Command::Help);
        assert_eq!(parse(&words("help")).unwrap(), Command::Help);
        assert_eq!(parse(&words("--version")).unwrap(), Command::Version);
        assert_eq!(parse(&words("doctor")).unwrap(), Command::Doctor);
        assert_eq!(
            parse(&words("init")).unwrap(),
            Command::Init(Target::default())
        );
        assert_eq!(
            parse(&words("init --manifest-path k/Cargo.toml")).unwrap(),
            Command::Init(Target {
                package: None,
                manifest_path: Some(PathBuf::from("k/Cargo.toml")),
            })
        );
        assert_eq!(
            parse(&words(
                "build -p kernels --format spirv --format=wgsl,spirv --manifest-path=a/Cargo.toml"
            ))
            .unwrap(),
            Command::Build(BuildOptions {
                target: Target {
                    package: Some("kernels".to_owned()),
                    manifest_path: Some(PathBuf::from("a/Cargo.toml")),
                },
                formats: vec![Format::Spirv, Format::Wgsl, Format::Spirv],
                allow_slang_mismatch: false,
            })
        );
        assert_eq!(
            parse(&words("build --allow-slang-mismatch -p k")).unwrap(),
            Command::Build(BuildOptions {
                target: Target {
                    package: Some("k".to_owned()),
                    manifest_path: None,
                },
                formats: vec![],
                allow_slang_mismatch: true,
            })
        );
        assert_eq!(
            parse(&words(
                "check --package k --recompile --allow-slang-mismatch"
            ))
            .unwrap(),
            Command::Check(CheckOptions {
                target: Target {
                    package: Some("k".to_owned()),
                    manifest_path: None,
                },
                recompile: true,
                allow_slang_mismatch: true,
            })
        );
        assert_eq!(
            parse(&words("check")).unwrap(),
            Command::Check(CheckOptions {
                target: Target::default(),
                recompile: false,
                allow_slang_mismatch: false,
            })
        );
    }

    #[test]
    fn the_slang_commands_and_their_options_parse() {
        assert_eq!(
            parse(&words("slang install")).unwrap(),
            Command::Slang(SlangCommand::Install)
        );
        assert_eq!(
            parse(&words("slang locate")).unwrap(),
            Command::Slang(SlangCommand::Locate {
                allow_mismatch: false
            })
        );
        assert_eq!(
            parse(&words("slang locate --allow-slang-mismatch")).unwrap(),
            Command::Slang(SlangCommand::Locate {
                allow_mismatch: true
            })
        );
        assert_eq!(
            parse(&words("slang build-helper")).unwrap(),
            Command::Slang(SlangCommand::BuildHelper {
                force: false,
                sdk_root: None,
                out_dir: None,
            })
        );
        assert_eq!(
            parse(&words(
                "slang build-helper --force --sdk-root /s --out-dir=/o"
            ))
            .unwrap(),
            Command::Slang(SlangCommand::BuildHelper {
                force: true,
                sdk_root: Some(PathBuf::from("/s")),
                out_dir: Some(PathBuf::from("/o")),
            })
        );
    }

    #[test]
    fn bad_input_is_refused_with_a_reason() {
        for (line, expected) in [
            ("frobnicate", "unknown command `frobnicate`"),
            ("build --format glsl", "unknown format `glsl`"),
            ("build --format", "`--format` needs a value"),
            ("build --format=", "needs a non-empty value"),
            ("build -p", "`-p` needs a value"),
            ("build -p a -p b", "`-p` was given twice"),
            ("build --manifest-path a --manifest-path b", "given twice"),
            ("build --recompile", "unknown option `--recompile`"),
            ("check --format wgsl", "unknown option `--format`"),
            ("check --recompile=yes", "`--recompile` takes no value"),
            ("build extra", "unexpected argument `extra`"),
            ("doctor -p x", "unknown option `-p`"),
            ("init --release", "unknown option `--release`"),
            ("slang", "`slang` needs a command"),
            ("slang frobnicate", "unknown slang command `frobnicate`"),
            ("slang install --force", "unknown option `--force`"),
            ("slang install extra", "unexpected argument `extra`"),
            ("slang locate --sdk-root x", "unknown option `--sdk-root`"),
            (
                "slang build-helper --allow-slang-mismatch",
                "unknown option",
            ),
            (
                "slang build-helper --sdk-root",
                "`--sdk-root` needs a value",
            ),
            (
                "slang build-helper --out-dir a --out-dir b",
                "`--out-dir` was given twice",
            ),
            (
                "slang build-helper --sdk-root a --sdk-root b",
                "`--sdk-root` was given twice",
            ),
            ("slang locate --allow-slang-mismatch=yes", "takes no value"),
            ("build --force", "unknown option `--force`"),
            ("check --sdk-root x", "unknown option `--sdk-root`"),
        ] {
            let error = parse(&words(line)).unwrap_err();
            assert!(error.contains(expected), "{line}: {error}");
        }
    }

    #[test]
    fn the_usage_text_lists_every_command() {
        for command in [
            "init",
            "build",
            "check",
            "doctor",
            "slang install",
            "slang locate",
            "slang build-helper",
            "run",
            "test",
            "help",
        ] {
            assert!(
                USAGE.contains(&format!("cargo gust {command}")),
                "{command}"
            );
        }
        assert!(USAGE.contains("GUST_SLANG_HOME"));
    }

    #[test]
    fn run_and_test_read_the_package_and_pass_every_argument_on() {
        for (line, package, manifest) in [
            ("run", None, None),
            ("run -p app", Some("app"), None),
            ("run -papp --release", Some("app"), None),
            ("test --package app -- --nocapture", Some("app"), None),
            ("test --package=app", Some("app"), None),
            (
                "run --manifest-path a/Cargo.toml",
                None,
                Some("a/Cargo.toml"),
            ),
            // After `--` the words belong to the program or the test harness.
            ("run -p app -- -p other", Some("app"), None),
            ("test -- --package other", None, None),
        ] {
            let args = words(line);
            let parsed = parse(&args).unwrap();
            let (Command::Run(options) | Command::Test(options)) = parsed else {
                panic!("{line}: {parsed:?}");
            };
            assert_eq!(options.target.package.as_deref(), package, "{line}");
            assert_eq!(
                options.target.manifest_path.as_deref(),
                manifest.map(std::path::Path::new),
                "{line}"
            );
            assert_eq!(options.arguments, args[1..], "{line}");
        }
        for (line, expected) in [
            ("run -p", "needs a value"),
            ("run -p a -p b", "was given twice"),
            ("test --package a --package=b", "was given twice"),
            ("run --manifest-path", "needs a value"),
            ("run --manifest-path a --manifest-path b", "was given twice"),
            ("run --package=", "needs a non-empty value"),
        ] {
            let error = parse(&words(line)).unwrap_err();
            assert!(error.contains(expected), "{line}: {error}");
        }
    }
}
