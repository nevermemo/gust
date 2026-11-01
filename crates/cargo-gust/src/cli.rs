//! The command line: `cargo gust <command>` (cargo runs `cargo-gust gust <command>`) or `cargo-gust <command>`.

use std::path::PathBuf;

use gust::Format;

pub(crate) const USAGE: &str = "\
cargo gust: export precompiled GUST kernels (D45, D46)

USAGE:
    cargo gust init   [-p <crate>] [--manifest-path <path>]
    cargo gust build  [-p <crate>] [--manifest-path <path>] [--format wgsl|spirv]...
    cargo gust check  [-p <crate>] [--manifest-path <path>] [--recompile] [--allow-slang-mismatch]
    cargo gust doctor
    cargo gust help

COMMANDS:
    init     create gust-artifacts/manifest.json (an empty manifest) in the crate
    build    collect the crate's #[gpu] kernels, compile them with Slang, verify them (D17) and write gust-artifacts/
             WGSL is always built; --format spirv adds SPIR-V (checked with spirv-val when it is on PATH)
    check    fail unless gust-artifacts/ is exactly what a build of the current sources gives
             --recompile also recompiles and compares the bytes; --allow-slang-mismatch accepts another Slang build
    doctor   show the native helper, the build profile and how to install this tool

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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CheckOptions {
    pub(crate) target: Target,
    pub(crate) recompile: bool,
    pub(crate) allow_slang_mismatch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Command {
    Init(Target),
    Build(BuildOptions),
    Check(CheckOptions),
    Doctor,
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
                &[],
            )?;
            Ok(Command::Build(BuildOptions {
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
        "doctor" => {
            options(rest, &[], &[])?;
            Ok(Command::Doctor)
        }
        other => Err(format!("unknown command `{other}`")),
    }
}

struct Parsed {
    target: Target,
    formats: Vec<Format>,
    flags: Vec<&'static str>,
}

/// Read `--name value`, `--name=value` and the listed bare flags.
fn options(args: &[String], valued: &[&str], bare: &[&'static str]) -> Result<Parsed, String> {
    let mut target = Target::default();
    let mut formats = Vec::new();
    let mut flags = Vec::new();
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
        ] {
            let error = parse(&words(line)).unwrap_err();
            assert!(error.contains(expected), "{line}: {error}");
        }
    }

    #[test]
    fn the_usage_text_lists_every_command() {
        for command in ["init", "build", "check", "doctor", "help"] {
            assert!(
                USAGE.contains(&format!("cargo gust {command}")),
                "{command}"
            );
        }
        assert!(
            !USAGE.contains("slang install"),
            "slang management is slice m4.f"
        );
    }
}
