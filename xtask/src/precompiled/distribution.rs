//! E8: distribution. The four packages are packaged, unpacked outside the repository and built into a consumer with
//! registry-style dependencies.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    env::consts::EXE_SUFFIX,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use super::{
    Mode, Options, PRECOMPILED_DIR,
    fixture::{Fixture, require_success},
    reexpand::snapshot,
    stripped::{RunReport, VERIFIED_PREFIX, judge_run},
};
use crate::{
    json::{self, Json},
    process::run_capture_in,
    release::{PackageSpec, WorkDir, package_files, package_problems, untar_crate_command},
};

// ---------------------------------------------------------------------------------------------------------------
// E8

/// The packages of the distribution proof, in the order they are unpacked and patched. What their `cargo package
/// --list` may hold is decided by `package_problems` of `release.rs`, the one home of the allow-lists (mr.r); the
/// `kernels` package is the only one with `gust-artifacts/**`.
pub(crate) const E8_PACKAGES: [PackageSpec<'static>; 4] = [
    PackageSpec::library("gust-compute-macros", false, false),
    PackageSpec::library("gust-compute", true, false),
    PackageSpec::library("gust-compute-wgpu", false, false),
    PackageSpec::library("kernels", false, true),
];

/// The strings that the packaged `kernels` must not contain as text: the CPU twin's types and function names.
const TWIN_TEXT: [&str; 3] = ["fn cpu", "gust::cpu", "CpuDevice"];

/// R4.6 (c): the unpacked kernels tree holds no `.slang` file, nothing from `generated-wgpu/` and no twin text.
/// `files` are relative paths with their text.
pub(crate) fn slang_free_package_problems(files: &[(String, String)]) -> Vec<String> {
    let mut problems = Vec::new();
    for (path, text) in files {
        if path.to_ascii_lowercase().ends_with(".slang") {
            problems.push(format!(
                "the kernels package holds the Slang source `{path}`"
            ));
        }
        if path.contains("generated-wgpu") {
            problems.push(format!(
                "the kernels package holds `{path}` from generated-wgpu/"
            ));
        }
        for marker in TWIN_TEXT {
            if text.contains(marker) {
                problems.push(format!(
                    "`{path}` of the kernels package contains the twin text `{marker}`"
                ));
            }
        }
    }
    problems
}

/// The `version` of the first `version = ".."` line in the table `[section]` of `manifest`.
pub(crate) fn section_version(manifest: &str, section: &str) -> Option<String> {
    let header = format!("[{section}]");
    let mut inside = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            inside = line == header;
            continue;
        }
        if !inside {
            continue;
        }
        let Some(rest) = line.strip_prefix("version") else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix('=') else {
            continue;
        };
        let Some((value, _)) = rest
            .trim()
            .strip_prefix('"')
            .and_then(|rest| rest.split_once('"'))
        else {
            continue;
        };
        return Some(value.to_owned());
    }
    None
}

/// The `Cargo.toml` of the relocated consumer (E8): plain registry-style dependencies on the packages that the
/// workspace calls `gust-compute`, `gust-compute-wgpu` and `kernels` (no `default-features = false`, D45), an empty `[workspace]` table
/// so that it is a root of its own, and a `[patch.crates-io]` for all four packages onto their unpacked trees, which
/// sit beside the consumer as `../unpacked/<name>-<version>`. `versions` has the version of each of
/// [`E8_PACKAGES`], in its order.
pub(crate) fn consumer_manifest(versions: &[String; 4]) -> String {
    let mut text = String::from(
        "# The relocated consumer of E8 in `cargo xtask check-precompiled`: registry-style dependencies, patched onto the\n# unpacked packages.\n[package]\nname = \"dist-consumer\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n[dependencies]\n",
    );
    for (spec, version) in E8_PACKAGES.iter().zip(versions) {
        if spec.name != "gust-compute-macros" {
            text.push_str(&format!("{} = \"={version}\"\n", spec.name));
        }
    }
    text.push_str("\n[workspace]\n\n[patch.crates-io]\n");
    for (spec, version) in E8_PACKAGES.iter().zip(versions) {
        text.push_str(&format!(
            "{} = {{ path = \"../unpacked/{}-{version}\" }}\n",
            spec.name, spec.name
        ));
    }
    text
}

/// One `[[package]]` of a `Cargo.lock`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LockPackage {
    pub(crate) name: String,
    pub(crate) source: Option<String>,
}

/// The packages of a `Cargo.lock`.
pub(crate) fn lock_packages(lock: &str) -> Vec<LockPackage> {
    let mut packages: Vec<LockPackage> = Vec::new();
    let mut current: Option<LockPackage> = None;
    for line in lock.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            packages.extend(current.take());
            current = Some(LockPackage {
                name: String::new(),
                source: None,
            });
        } else if let Some(package) = current.as_mut() {
            let value = |key: &str| {
                line.strip_prefix(key)
                    .and_then(|rest| rest.trim_start().strip_prefix('='))
                    .and_then(|rest| rest.trim().strip_prefix('"'))
                    .and_then(|rest| rest.split_once('"'))
                    .map(|(value, _)| value.to_owned())
            };
            if let Some(name) = value("name") {
                package.name = name;
            } else if let Some(source) = value("source") {
                package.source = Some(source);
            }
        }
    }
    packages.extend(current);
    packages
}

/// The problems of the consumer's `Cargo.lock`: each of the four packages once and with no `source` line (a path source
/// has none; a registry one reads `registry+https://..`), and at least one package from a registry, so that an
/// unreadable lock cannot pass.
pub(crate) fn lock_problems(lock: &str) -> Vec<String> {
    let packages = lock_packages(lock);
    let mut problems = Vec::new();
    for spec in &E8_PACKAGES {
        let entries: Vec<&LockPackage> = packages
            .iter()
            .filter(|package| package.name == spec.name)
            .collect();
        match entries.as_slice() {
            [] => problems.push(format!(
                "the lock has no `[[package]]` named `{}`",
                spec.name
            )),
            [only] => {
                if let Some(source) = &only.source {
                    problems.push(format!(
                        "the lock's `{}` has the source `{source}`: it came from a registry, not from the unpacked package",
                        spec.name
                    ));
                }
            }
            many => problems.push(format!(
                "the lock has {} `[[package]]` entries named `{}`",
                many.len(),
                spec.name
            )),
        }
    }
    if !packages.iter().any(|package| {
        package
            .source
            .as_deref()
            .is_some_and(|source| source.starts_with("registry+"))
    }) {
        problems.push("the lock has no package from a registry (wgpu should be), so it was not read correctly".to_owned());
    }
    problems
}

/// The lines of cargo's output that report a patch that nothing used.
pub(crate) fn unused_patch_lines(output: &str) -> Vec<String> {
    output
        .lines()
        .filter(|line| line.contains("was not used"))
        .map(|line| line.trim().to_owned())
        .collect()
}

/// Whether the feature entry `entry` of a `[features]` table turns on `slang-runtime`: the feature itself, or the
/// feature of a dependency (`dep/slang-runtime`, `dep?/slang-runtime`).
fn names_slang_runtime(entry: &str) -> bool {
    entry == "slang-runtime" || entry.ends_with("/slang-runtime")
}

/// The problems of `cargo metadata` output for the packages `names`: each must be there, come from a path (the
/// unpacked tree under `unpacked`, forward slashes), and no feature reachable from its `default` feature may turn on
/// `slang-runtime`.
pub(crate) fn default_feature_problems(
    metadata: &str,
    names: &[&str],
    unpacked: &str,
) -> Result<Vec<String>, String> {
    let parsed =
        json::parse(metadata).map_err(|error| format!("cargo metadata output: {error}"))?;
    let packages = parsed
        .get("packages")
        .and_then(Json::as_array)
        .ok_or("cargo metadata output has no `packages` array")?;
    let mut problems = Vec::new();
    for name in names {
        let Some(package) = packages
            .iter()
            .find(|package| package.get("name").and_then(Json::as_str) == Some(*name))
        else {
            problems.push(format!("cargo metadata lists no package `{name}`"));
            continue;
        };
        let manifest = package
            .get("manifest_path")
            .and_then(Json::as_str)
            .unwrap_or("")
            .replace('\\', "/");
        if !manifest.starts_with(unpacked) {
            problems.push(format!(
                "the package `{name}` of the consumer is not the unpacked one (its manifest is `{manifest}`)"
            ));
        }
        let Some(Json::Object(features)) = package.get("features") else {
            problems.push(format!("cargo metadata gives `{name}` no `features` table"));
            continue;
        };
        let table: BTreeMap<&str, Vec<&str>> = features
            .iter()
            .map(|(feature, list)| {
                (
                    feature.as_str(),
                    list.as_array()
                        .unwrap_or(&[])
                        .iter()
                        .filter_map(Json::as_str)
                        .collect(),
                )
            })
            .collect();
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        let mut pending: Vec<&str> = vec!["default"];
        while let Some(feature) = pending.pop() {
            if !seen.insert(feature) {
                continue;
            }
            for entry in table.get(feature).into_iter().flatten() {
                if names_slang_runtime(entry) {
                    problems.push(format!(
                        "a default feature of `{name}` turns on `slang-runtime` (`{feature}` lists `{entry}`)"
                    ));
                } else if table.contains_key(entry) {
                    pending.push(entry);
                }
            }
        }
    }
    Ok(problems)
}

/// Whether `child` is `parent` or below it, compared as text with forward slashes (and without case on Windows).
pub(crate) fn is_inside(parent: &Path, child: &Path) -> bool {
    let normal = |path: &Path| {
        let text = path.to_string_lossy().replace('\\', "/");
        let text = text.trim_end_matches('/').to_owned();
        if cfg!(windows) {
            text.to_ascii_lowercase()
        } else {
            text
        }
    };
    let (parent, child) = (normal(parent), normal(child));
    child == parent || child.starts_with(&format!("{parent}/"))
}

/// The relative paths (forward slashes) of the files below `directory`.
fn relative_files(directory: &Path) -> io::Result<Vec<String>> {
    Ok(snapshot(directory)?.into_keys().collect())
}

/// Runs `command` and returns its output when it succeeds.
fn run_step(step: &str, what: &str, mut command: Command) -> Result<Output, String> {
    let output = run_capture_in(&mut command, &format!("check-precompiled {step}"), "cargo")?;
    require_success(step, what, &output)?;
    Ok(output)
}

/// The two `--config` values with which `cargo package -p kernels` resolves `gust-compute` and `gust-compute-macros`
/// onto this checkout (E8 step 2): `patch.crates-io.<package>.path='<root>/crates/<directory>'`, with forward
/// slashes. The path is a TOML literal string, which has no escapes, so a root that holds a quote or a line break
/// is refused with a clear error and not written wrongly.
pub(crate) fn kernels_patch_configs(root: &Path) -> Result<[String; 2], String> {
    let root = root.to_string_lossy().replace('\\', "/");
    if root.contains(['\'', '\n', '\r']) {
        return Err(format!(
            "check-precompiled E8: the repository path holds a quote or a line break, which the `--config` patch of \
             `cargo package -p kernels` cannot carry: {root:?}"
        ));
    }
    Ok([
        format!("patch.crates-io.gust-compute.path='{root}/crates/gust'"),
        format!("patch.crates-io.gust-compute-macros.path='{root}/crates/gust-macros'"),
    ])
}

/// The directory under the cargo target directory where `--prepare` keeps the consumer of E8.
const KEPT_CONSUMER_DIR: &str = "precompiled-dist";

/// Where `--prepare` keeps the built consumer of E8, whose work directory is deleted.
pub(super) fn kept_consumer(fixture: &Fixture) -> PathBuf {
    fixture
        .target
        .join(KEPT_CONSUMER_DIR)
        .join(format!("dist-consumer{EXE_SUFFIX}"))
}

/// The run half of E8: the built consumer on the adapter. Returns what it printed.
pub(super) fn e8_run(fixture: &Fixture, binary: &Path) -> Result<RunReport, String> {
    let run = fixture
        .command(binary)
        .output()
        .map_err(|error| format!("check-precompiled E8: could not run the consumer: {error}"))?;
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&run.stderr).into_owned();
    print!("{stdout}");
    judge_run(run.status.success(), &stdout, &stderr).map_err(|problems| {
        format!(
            "check-precompiled E8: the consumer: {}",
            problems.join("; ")
        )
    })
}

/// E8: the packages and the relocated consumer. Returns the summary. The work directory is deleted at the end, and
/// a failure to delete it fails the check.
pub(super) fn e8_distribution(fixture: &Fixture, options: &Options) -> Result<String, String> {
    let work = WorkDir::create(
        "check-precompiled E8",
        "gust-dist",
        &fixture.root,
        &env::temp_dir(),
        options.work_dir.as_deref(),
    )?;
    let outcome = e8_steps(fixture, options, &work.path);
    work.finish(outcome)
}

/// E8 in the work directory `work`.
fn e8_steps(fixture: &Fixture, options: &Options, work: &Path) -> Result<String, String> {
    let read = |path: PathBuf| {
        fs::read_to_string(&path)
            .map_err(|error| format!("check-precompiled E8: {}: {error}", path.display()))
    };
    let workspace_version =
        section_version(&read(fixture.root.join("Cargo.toml"))?, "workspace.package")
            .ok_or("check-precompiled E8: Cargo.toml has no [workspace.package] version")?;
    let kernels_version = section_version(
        &read(
            fixture
                .root
                .join(PRECOMPILED_DIR)
                .join("kernels")
                .join("Cargo.toml"),
        )?,
        "package",
    )
    .ok_or("check-precompiled E8: kernels/Cargo.toml has no [package] version")?;
    let versions = [
        workspace_version.clone(),
        workspace_version.clone(),
        workspace_version,
        kernels_version,
    ];
    let fixture_manifest = format!("{PRECOMPILED_DIR}/Cargo.toml");
    let committed: Vec<String> = relative_files(
        &fixture
            .root
            .join(PRECOMPILED_DIR)
            .join("kernels")
            .join("gust-artifacts"),
    )
    .map_err(|error| format!("check-precompiled E8: {error}"))?;

    // 1. `cargo package --list`: the allow-list of each package.
    let mut problems = Vec::new();
    for spec in &E8_PACKAGES {
        let mut args = vec!["package", "--list", "--allow-dirty", "--color", "never"];
        if spec.artifacts {
            args.extend(["--manifest-path", &fixture_manifest]);
        }
        args.extend(["-p", spec.name]);
        let output = run_step(
            "E8",
            &format!("cargo package --list -p {}", spec.name),
            fixture.cargo(&args),
        )?;
        let files = package_files(&String::from_utf8_lossy(&output.stdout));
        let required: &[String] = if spec.artifacts { &committed } else { &[] };
        problems.extend(package_problems(spec, &files, required));
    }
    if !problems.is_empty() {
        return Err(format!(
            "check-precompiled E8 (cargo package --list) found {} problem(s):\n{}",
            problems.len(),
            problems.join("\n")
        ));
    }
    println!(
        "check-precompiled E8: the file lists of {} packages are on the allow-list (kernels lists every committed artifact)",
        E8_PACKAGES.len()
    );

    // 2. Package and unpack outside the repository. The three repository crates go in one call: cargo then resolves
    // them against each other, which a single `-p gust-compute` cannot (`gust-compute-macros` is not on crates.io).
    let repo_target = work.join("package-target");
    let kernels_target = work.join("package-kernels-target");
    let repo_target_arg = Fixture::target_dir_arg(&repo_target)?;
    let kernels_target_arg = Fixture::target_dir_arg(&kernels_target)?;
    run_step(
        "E8",
        "cargo package -p gust-compute-macros -p gust-compute -p gust-compute-wgpu",
        fixture.cargo(&[
            "package",
            "--allow-dirty",
            "--no-verify",
            "--color",
            "never",
            "-p",
            "gust-compute-macros",
            "-p",
            "gust-compute",
            "-p",
            "gust-compute-wgpu",
            "--target-dir",
            repo_target_arg,
        ]),
    )?;
    // `cargo package` resolves the dependencies of the packaged manifest against crates.io, where `kernels` asks for
    // `gust-compute = "0.1.0"`. What crates.io holds under a GUST name must not decide this check: before the rename
    // the requirement was for `gust`, which cargo met with the unrelated charting crate of that name (0.1.0 and 0.1.4
    // exist), and once the packages were renamed the placeholder `gust-compute 0.0.0` met nothing. So the two packages
    // that `kernels` reaches are patched onto their trees in the repository (`--config`, which `cargo package`
    // honours and which leaves the packaged manifest as it is): the resolution finds this checkout's `gust-compute`
    // and its `gust-compute-macros`, and `--no-verify` still skips the build of the unpacked package.
    let [patch_compute, patch_macros] = kernels_patch_configs(&fixture.root)?;
    run_step(
        "E8",
        "cargo package -p kernels",
        fixture.cargo(&[
            "package",
            "--allow-dirty",
            "--no-verify",
            "--color",
            "never",
            "--config",
            &patch_compute,
            "--config",
            &patch_macros,
            "--manifest-path",
            &fixture_manifest,
            "-p",
            "kernels",
            "--target-dir",
            kernels_target_arg,
        ]),
    )?;
    let crates = work.join("crates");
    let unpacked = work.join("unpacked");
    fs::create_dir_all(&crates)
        .and_then(|()| fs::create_dir_all(&unpacked))
        .map_err(|error| format!("check-precompiled E8: {error}"))?;
    for (spec, version) in E8_PACKAGES.iter().zip(&versions) {
        let file = format!("{}-{version}.crate", spec.name);
        let from = if spec.artifacts {
            &kernels_target
        } else {
            &repo_target
        }
        .join("package")
        .join(&file);
        fs::copy(&from, crates.join(&file))
            .map_err(|error| format!("check-precompiled E8: {}: {error}", fixture.shown(&from)))?;
        let output = untar_crate_command(&unpacked, &file)
            .output()
            .map_err(|error| format!("check-precompiled E8: could not run tar: {error}"))?;
        require_success("E8", &format!("tar -xzf {file}"), &output)?;
    }

    // 3. The unpacked trees are judged again, and the kernels tree is Slang-free.
    for (spec, version) in E8_PACKAGES.iter().zip(&versions) {
        let tree = unpacked.join(format!("{}-{version}", spec.name));
        let files = relative_files(&tree)
            .map_err(|error| format!("check-precompiled E8: {}: {error}", tree.display()))?;
        let required: &[String] = if spec.artifacts { &committed } else { &[] };
        problems.extend(package_problems(spec, &files, required));
        if spec.artifacts {
            let contents: Vec<(String, String)> = snapshot(&tree)
                .map_err(|error| format!("check-precompiled E8: {error}"))?
                .into_iter()
                .map(|(name, bytes)| (name, String::from_utf8_lossy(&bytes).into_owned()))
                .collect();
            problems.extend(slang_free_package_problems(&contents));
        }
    }
    if !problems.is_empty() {
        return Err(format!(
            "check-precompiled E8 (unpacked packages) found {} problem(s):\n{}",
            problems.len(),
            problems.join("\n")
        ));
    }

    // 4. The consumer.
    let consumer = work.join("consumer");
    fs::create_dir_all(consumer.join("src"))
        .map_err(|error| format!("check-precompiled E8: {error}"))?;
    let write = |path: PathBuf, text: &str| {
        fs::write(&path, text)
            .map_err(|error| format!("check-precompiled E8: {}: {error}", path.display()))
    };
    write(consumer.join("Cargo.toml"), &consumer_manifest(&versions))?;
    // The same `main` as E1's `app`, so the consumer proves the same run; the lock is seeded from the fixture's so that
    // `wgpu` and the rest resolve to the versions the fixture is tested with.
    write(
        consumer.join("src").join("main.rs"),
        &read(
            fixture
                .root
                .join(PRECOMPILED_DIR)
                .join("app")
                .join("src")
                .join("main.rs"),
        )?,
    )?;
    write(
        consumer.join("Cargo.lock"),
        &read(fixture.root.join(PRECOMPILED_DIR).join("Cargo.lock"))?,
    )?;
    let consumer_manifest_path = consumer.join("Cargo.toml");
    let consumer_manifest_arg = consumer_manifest_path
        .to_str()
        .ok_or("check-precompiled E8: the work directory is not valid UTF-8")?;

    // 4a. `cargo metadata` first: the features the registry consumer sees are the packaged ones.
    let output = run_step(
        "E8",
        "cargo metadata (the consumer)",
        fixture.cargo_in(
            &consumer,
            &[
                "metadata",
                "--format-version",
                "1",
                "--color",
                "never",
                "--manifest-path",
                consumer_manifest_arg,
            ],
        ),
    )?;
    let unpacked_prefix = unpacked.to_string_lossy().replace('\\', "/");
    let mut unused = unused_patch_lines(&String::from_utf8_lossy(&output.stderr));
    problems.extend(default_feature_problems(
        &String::from_utf8_lossy(&output.stdout),
        &["gust-compute", "gust-compute-wgpu"],
        &unpacked_prefix,
    )?);
    if !problems.is_empty() {
        return Err(format!(
            "check-precompiled E8 (cargo metadata) found {} problem(s):\n{}",
            problems.len(),
            problems.join("\n")
        ));
    }

    // 4b. The release build under `native` in the stripped environment of E1.
    let target = work.join("target");
    let target_arg = Fixture::target_dir_arg(&target)?;
    let output = run_step(
        "E8",
        "cargo build --release (the consumer)",
        fixture.cargo_in(
            &consumer,
            &[
                "build",
                "--release",
                "--locked",
                "--color",
                "never",
                "--manifest-path",
                consumer_manifest_arg,
                "--target-dir",
                target_arg,
            ],
        ),
    )?;
    unused.extend(unused_patch_lines(&String::from_utf8_lossy(&output.stderr)));
    problems.extend(
        unused
            .into_iter()
            .map(|line| format!("cargo reported an unused patch: {line}")),
    );
    problems.extend(lock_problems(&read(consumer.join("Cargo.lock"))?));
    if !problems.is_empty() {
        return Err(format!(
            "check-precompiled E8 (patch and lock) found {} problem(s):\n{}",
            problems.len(),
            problems.join("\n")
        ));
    }

    // 4c. Run it.
    let built = Fixture::app_binary(&target);
    let built = built.with_file_name(format!("dist-consumer{EXE_SUFFIX}"));
    if options.mode == Mode::Prepare {
        // The work directory is deleted when this step ends; `--adapter-runs` runs the kept copy.
        let kept = kept_consumer(fixture);
        fs::create_dir_all(fixture.target.join(KEPT_CONSUMER_DIR))
            .and_then(|()| fs::copy(&built, &kept))
            .map_err(|error| {
                format!("check-precompiled E8: could not keep the consumer: {error}")
            })?;
    }
    if !options.mode.runs() {
        return Ok("passed (build only: this mode skips the run, which needs an adapter): the four packages are on the allow-list and Slang-free, no default feature of gust-compute or gust-compute-wgpu turns on slang-runtime, and the consumer with registry-style dependencies built in release under native in the stripped environment, with every patch used and no `source` in the lock for the four packages".to_owned());
    }
    let report = e8_run(fixture, &built)?;
    Ok(format!(
        "passed: the four packages are on the allow-list and Slang-free, no default feature of gust-compute or gust-compute-wgpu turns on slang-runtime, and the consumer with registry-style dependencies on the unpacked packages built in release under native in the stripped environment (every patch used, no `source` in the lock for the four) and printed `{VERIFIED_PREFIX}{} elements` on the adapter `{}`",
        report.elements, report.adapter
    ))
}

#[cfg(test)]
#[path = "../tests/precompiled_distribution.rs"]
mod tests;
