use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

pub(crate) const EXAMPLES: &[&str] = &[
    "atomic-counter",
    "vector-add",
    "polynomial",
    "signal-pipeline",
    "particle-step",
    "typed-pipeline",
    "staged-graph",
    "component-pool",
];

pub(crate) const SMOKE_EXAMPLES: &[&str] = &["vector-add", "typed-pipeline"];

pub(crate) const WORKSPACE_EXCLUDES: &[&str] = &[
    "atomic-counter",
    "vector-add",
    "polynomial",
    "signal-pipeline",
    "particle-step",
    "typed-pipeline",
    "staged-graph",
    "component-pool",
];

/// SPIR-V files the examples export to `generated-wgpu/`, one `<module>__<kernel>.spv` per
/// kernel (the examples build each stem from `descriptor.module` and `descriptor.name`).
/// Sorted; `verify` compares the directory against this list by name.
pub(crate) const EXPECTED_SPIRV: &[&str] = &[
    "atomic_counter__increment.spv",
    "particles__snapshot.spv",
    "particles__step.spv",
    "polynomial__evaluate.spv",
    "pool__classify.spv",
    "pool__integrate.spv",
    "pool__prepare_dispatch.spv",
    "sensor_pipeline__calibrate.spv",
    "sensor_pipeline__classify.spv",
    "sensor_pipeline__finalize.spv",
    "signal_pipeline__transform.spv",
    "staged__summarize.spv",
    "staged__transform.spv",
    "vector_add__add.spv",
];

pub(crate) const FEATURE_AREAS: &[&str] = &[
    "macro",
    "core",
    "wgpu",
    "gpu-smoke",
    "gpu-semantics",
    "gpu-runtime",
    "reflection",
    "loops",
    "result",
    "vectors",
    "uniforms",
    "map",
    "workgroup",
    "algo",
    "examples",
    "artifacts",
    "full",
];

pub(crate) fn workspace_root() -> Result<PathBuf, String> {
    let mut path = env::current_dir().map_err(|error| error.to_string())?;
    loop {
        if path.join("Cargo.toml").is_file() && path.join("crates").is_dir() {
            return Ok(path);
        }
        if !path.pop() {
            return Err("could not locate workspace root".into());
        }
    }
}

/// Where cargo puts build output: `CARGO_TARGET_DIR` when set, else `<root>/target`.
pub(crate) fn cargo_target_dir(root: &Path) -> PathBuf {
    env::var_os("CARGO_TARGET_DIR").map_or_else(|| root.join("target"), PathBuf::from)
}

pub(crate) fn workspace_test_args() -> Vec<String> {
    let mut args = vec![
        "test".to_owned(),
        "--locked".to_owned(),
        "--workspace".to_owned(),
    ];
    for package in WORKSPACE_EXCLUDES {
        args.push("--exclude".to_owned());
        args.push((*package).to_owned());
    }
    args
}

pub(crate) fn changed_paths() -> Result<Vec<String>, String> {
    let root = workspace_root()?;
    let mut paths = Vec::new();
    for args in [
        ["diff", "--name-only", "HEAD"].as_slice(),
        ["ls-files", "--others", "--exclude-standard"].as_slice(),
    ] {
        let output = Command::new("git")
            .args(args)
            .current_dir(&root)
            .output()
            .map_err(|error| format!("could not launch git: {error}"))?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).into_owned());
        }
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let path = line.trim().replace('\\', "/");
            if !path.is_empty() && !paths.contains(&path) {
                paths.push(path);
            }
        }
    }
    paths.sort();
    Ok(paths)
}

/// Repository-relative paths (forward slashes) of the files the next commit would contain:
/// tracked files plus untracked files that are not ignored, limited to files that exist.
pub(crate) fn project_files(root: &Path) -> Result<Vec<String>, String> {
    let output = Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .current_dir(root)
        .output()
        .map_err(|error| format!("could not launch git: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git ls-files failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let mut files = String::from_utf8_lossy(&output.stdout)
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(str::to_owned)
        .filter(|path| root.join(path).is_file())
        .collect::<Vec<_>>();
    files.sort();
    files.dedup();
    Ok(files)
}

/// The file's text, or `None` for binary or non-UTF-8 content.
pub(crate) fn read_text(path: &Path) -> Result<Option<String>, std::io::Error> {
    let bytes = std::fs::read(path)?;
    Ok(String::from_utf8(bytes)
        .ok()
        .filter(|text| !text.contains('\0')))
}
