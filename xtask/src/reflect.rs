//! `cargo xtask build-slang-reflect`: builds the native reflection helper (D17).
//!
//! The build itself lives in `cargo-gust` since slice m4.f (`cargo gust slang build-helper`: the C++ compiler search,
//! the MSVC environment capture and the packaged copy of `scripts/probes/slang-layout.cpp`), so there is one
//! implementation. This command keeps its name and options, finds the SDK the way it always did (`--sdk-root`,
//! `SLANG_SDK`, then `VULKAN_SDK`) and runs `cargo-gust` on it.

use std::{env, path::PathBuf};

use crate::{process::run_direct_args, workspace::workspace_root};

#[derive(Default)]
pub(crate) struct BuildOptions {
    force: bool,
    sdk_root: Option<PathBuf>,
    out_dir: Option<PathBuf>,
}

impl BuildOptions {
    pub(crate) fn parse(args: &[String]) -> Result<Self, String> {
        let mut options = Self::default();
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--force" | "-Force" => options.force = true,
                "--sdk-root" | "-SdkRoot" => {
                    index += 1;
                    options.sdk_root = Some(PathBuf::from(
                        args.get(index).ok_or("--sdk-root needs a path")?,
                    ));
                }
                "--out-dir" | "--output-directory" | "-OutputDirectory" => {
                    index += 1;
                    options.out_dir = Some(PathBuf::from(
                        args.get(index).ok_or("--out-dir needs a path")?,
                    ));
                }
                other => return Err(format!("unknown build-slang-reflect option `{other}`")),
            }
            index += 1;
        }
        Ok(options)
    }
}

/// The `cargo` arguments that run `cargo gust slang build-helper` on an SDK. `cargo run` builds `cargo-gust` first
/// (under `GUST_PROFILE=dev`, which `run_direct_args` sets for cargo children); the helper's own output reaches the
/// terminal unchanged.
fn cargo_gust_args(
    options: &BuildOptions,
    sdk_root: &std::path::Path,
    out_dir: &std::path::Path,
) -> Vec<String> {
    let mut args: Vec<String> = [
        "run",
        "--locked",
        "--quiet",
        "-p",
        "cargo-gust",
        "--",
        "slang",
        "build-helper",
    ]
    .map(str::to_owned)
    .to_vec();
    if options.force {
        args.push("--force".to_owned());
    }
    args.push("--sdk-root".to_owned());
    args.push(sdk_root.display().to_string());
    args.push("--out-dir".to_owned());
    args.push(out_dir.display().to_string());
    args
}

pub(crate) fn build_slang_reflect(options: BuildOptions) -> Result<(), String> {
    let root = workspace_root()?;
    let sdk_root = options
        .sdk_root
        .clone()
        .or_else(|| env::var_os("SLANG_SDK").map(PathBuf::from))
        .or_else(|| env::var_os("VULKAN_SDK").map(PathBuf::from))
        .ok_or("Set SLANG_SDK (standalone Slang SDK) or VULKAN_SDK, or pass --sdk-root.")?;
    let out_dir = options
        .out_dir
        .clone()
        .unwrap_or_else(|| root.join("target/slang-reflect"));
    run_direct_args(
        &root,
        "cargo",
        &cargo_gust_args(&options, &sdk_root, &out_dir),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn words(line: &str) -> Vec<String> {
        line.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn the_options_keep_their_names_and_reject_the_rest() {
        let options = BuildOptions::parse(&words("--force --sdk-root /s --out-dir /o")).unwrap();
        assert!(options.force);
        assert_eq!(options.sdk_root, Some(PathBuf::from("/s")));
        assert_eq!(options.out_dir, Some(PathBuf::from("/o")));
        let legacy = BuildOptions::parse(&words("-Force -SdkRoot /s -OutputDirectory /o")).unwrap();
        assert!(legacy.force && legacy.sdk_root.is_some() && legacy.out_dir.is_some());
        assert!(BuildOptions::parse(&words("--sdk-root")).is_err());
        assert!(BuildOptions::parse(&words("--nope")).is_err());
    }

    #[test]
    fn the_command_runs_cargo_gust_build_helper_on_the_sdk() {
        let args = cargo_gust_args(
            &BuildOptions::default(),
            Path::new("/sdk"),
            Path::new("/out"),
        );
        assert_eq!(
            args,
            [
                "run",
                "--locked",
                "--quiet",
                "-p",
                "cargo-gust",
                "--",
                "slang",
                "build-helper",
                "--sdk-root",
                "/sdk",
                "--out-dir",
                "/out"
            ]
        );
        let forced = cargo_gust_args(
            &BuildOptions {
                force: true,
                ..BuildOptions::default()
            },
            Path::new("/sdk"),
            Path::new("/out"),
        );
        assert!(forced.contains(&"--force".to_owned()));
        assert_eq!(forced.len(), args.len() + 1);
    }
}
