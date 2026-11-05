//! `cargo xtask check-web`: the first browser WebGPU run (plan M4 section 2.6, D45). It builds `examples/web-vector-add`
//! for `wasm32-unknown-unknown` under `GUST_PROFILE=web`, runs `wasm-bindgen --target web` on the result, serves the
//! output from a loopback HTTP server, opens it in headless Chrome and requires the line `GUST_OK <vendor>
//! <architecture>` that the page's Rust code logs to the console.
//!
//! The result is read from Chrome's console (`--enable-logging=stderr`), not from `--dump-dom --virtual-time-budget`,
//! which stays "pending" with WebGPU (measured by the m4.g2 probe). A console line counts only when its source is a
//! file of this check's own server, and only a message that starts with `GUST_OK ` or `GUST_FAIL ` is a result. Chrome
//! never exits on its own in headless mode, so the check kills it (the whole process tree) as soon as it has a result,
//! on a timeout and on every error path, and deletes its fresh profile directory.
//!
//! What this is evidence for: the real `gust-wgpu` vector-add ran on the browser's GPU in both forms (grid and pixel,
//! mx.w), and each equalled the host reference and the other bit for bit.
//! The label printed at the end is `Gpu (WebGPU, Chrome <version>, <vendor> <architecture>)`, or `Gpu (software ...)`
//! when the adapter is a software one. Nothing here runs Slang: the shader is the committed WGSL artifact.
//!
//! The server binds `127.0.0.1` on an ephemeral port only, answers `GET` for the plain files of the output directory
//! only, and is stopped when the check ends.
//!
//! Before any of that the check runs a feature guard (all of `--tree-only`): `cargo tree -e features -i wgpu` of the
//! host target must not list wgpu's `webgpu` feature, and the same for `wasm32-unknown-unknown` must (R4.9: `webgpu`
//! is enabled from `gust-wgpu`'s wasm32 target dependencies only, so a native build never gains it).

use std::{
    env,
    env::consts::EXE_SUFFIX,
    ffi::OsString,
    fs,
    io::{self, BufRead, BufReader, Read, Write},
    net::{IpAddr, Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crate::workspace::{cargo_target_dir, workspace_root};

/// The example, relative to the repository root.
const EXAMPLE_DIR: &str = "examples/web-vector-add";

/// The cargo package of the example, and the stem of its output (`-` becomes `_` in the wasm file).
const PACKAGE: &str = "web-vector-add";
const OUT_NAME: &str = "web_vector_add";

const WASM_TARGET: &str = "wasm32-unknown-unknown";

/// The profile of the build: strict, so the committed artifact is embedded and Slang never runs (R4.2).
const WEB_PROFILE: &str = "web";

/// How long the page may take to log its result, after Chrome is launched.
const DEFAULT_TIMEOUT_SECS: u64 = 90;

/// The messages the page logs.
const OK_PREFIX: &str = "GUST_OK ";
const FAIL_PREFIX: &str = "GUST_FAIL ";

/// What the check was asked to do: `check-web [--wasm-bindgen PATH] [--chrome PATH] [--timeout-secs N] [--chrome-arg ARG]...`
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Options {
    pub(crate) wasm_bindgen: Option<PathBuf>,
    pub(crate) chrome: Option<PathBuf>,
    pub(crate) timeout_secs: Option<u64>,
    /// Extra Chrome command-line flags, in order (for example the software-adapter flags of a CI machine).
    pub(crate) chrome_args: Vec<String>,
    /// Only the dependency-feature guard (`webgpu` is a wasm32-only feature of wgpu); needs no Chrome, no wasm-bindgen
    /// and no build, so CI can run it.
    pub(crate) tree_only: bool,
}

impl Options {
    pub(crate) fn parse(args: &[String]) -> Result<Self, String> {
        let mut options = Self::default();
        let mut index = 0;
        while index < args.len() {
            let flag = args[index].as_str();
            if flag == "--tree-only" {
                if options.tree_only {
                    return Err("check-web: --tree-only was given twice".into());
                }
                options.tree_only = true;
                index += 1;
                continue;
            }
            let value = || {
                args.get(index + 1)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| format!("check-web: {flag} needs a value"))
            };
            match flag {
                "--wasm-bindgen" => {
                    set_once(&mut options.wasm_bindgen, PathBuf::from(value()?), flag)?;
                }
                "--chrome" => set_once(&mut options.chrome, PathBuf::from(value()?), flag)?,
                "--timeout-secs" => {
                    let text = value()?;
                    let seconds = text.parse::<u64>().ok().filter(|seconds| *seconds > 0);
                    let seconds = seconds.ok_or_else(|| {
                        format!(
                            "check-web: --timeout-secs needs a positive whole number, not `{text}`"
                        )
                    })?;
                    set_once(&mut options.timeout_secs, seconds, flag)?;
                }
                "--chrome-arg" => options.chrome_args.push(value()?.clone()),
                other => {
                    return Err(format!(
                        "check-web: unknown argument `{other}`; usage: check-web [--wasm-bindgen PATH] [--chrome PATH] [--timeout-secs N] [--chrome-arg ARG]..."
                    ));
                }
            }
            index += 2;
        }
        if options.tree_only
            && (options.wasm_bindgen.is_some()
                || options.chrome.is_some()
                || options.timeout_secs.is_some()
                || !options.chrome_args.is_empty())
        {
            return Err("check-web: --tree-only takes no other flag".into());
        }
        Ok(options)
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(self.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS))
    }
}

fn set_once<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("check-web: {flag} was given twice"));
    }
    *slot = Some(value);
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// The entry point

pub(crate) fn check_web(args: &[String]) -> Result<(), String> {
    let summary = check_web_report(args)?;
    if !summary.is_empty() {
        println!("{summary}");
    }
    Ok(())
}

/// The tools `check-web` needs, found and checked before any build starts.
struct Tools {
    wanted: String,
    bindgen: PathBuf,
    chrome: PathBuf,
}

/// The lookups of `check-web` that can fail for a missing tool: the locked `wasm-bindgen` version and a CLI of that
/// version, Chrome, and the wasm32 standard library. `path_var` is the `PATH` the CLI is looked up in.
fn find_tools(root: &Path, options: &Options, path_var: Option<OsString>) -> Result<Tools, String> {
    let wanted = locked_wasm_bindgen_version(
        &fs::read_to_string(root.join("Cargo.lock"))
            .map_err(|error| format!("check-web: could not read Cargo.lock: {error}"))?,
    )?;
    let bindgen = locate_wasm_bindgen(options, &wanted, path_var)?;
    let chrome = locate_chrome(options)?;
    require_wasm_target()?;
    Ok(Tools {
        wanted,
        bindgen,
        chrome,
    })
}

/// The tool lookups of a default `check-web` run, and nothing else: no build, no Chrome. `check-full` calls this
/// first, so a missing tool fails in seconds and not after the long steps.
pub(crate) fn preflight_tools() -> Result<(), String> {
    find_tools(&workspace_root()?, &Options::default(), env::var_os("PATH")).map(|_| ())
}

/// [`check_web`] that returns its passing summary (empty for `--tree-only`, which has printed its one line) instead of
/// printing it, so that `check-full` can record it.
pub(crate) fn check_web_report(args: &[String]) -> Result<String, String> {
    let options = Options::parse(args)?;
    let root = workspace_root()?;
    let started = Instant::now();

    // The feature guard first: it is cheap, and a native build that gained `webgpu` is wrong whatever Chrome says.
    println!("check-web: {}", check_feature_scope(&root)?);
    if options.tree_only {
        return Ok(String::new());
    }

    let Tools {
        wanted,
        bindgen,
        chrome,
    } = find_tools(&root, &options, env::var_os("PATH"))?;
    let chrome_version = chrome_version(&chrome);
    println!(
        "check-web: Chrome {} ({}), wasm-bindgen {wanted} ({}), page deadline {} s",
        chrome_version.as_deref().unwrap_or("version unknown"),
        shown_name(&chrome),
        shown_name(&bindgen),
        options.timeout().as_secs(),
    );

    let target_dir = cargo_target_dir(&root);
    let out_dir = target_dir.join("web-check").join("site");

    let build_started = Instant::now();
    let wasm = build_example(&root, &target_dir)?;
    let build_time = build_started.elapsed();

    let bindgen_started = Instant::now();
    prepare_site(&root, &bindgen, &wasm, &out_dir)?;
    let bindgen_time = bindgen_started.elapsed();

    let page_started = Instant::now();
    let outcome = run_in_chrome(&chrome, &options, &target_dir, &out_dir)?;
    let page_time = page_started.elapsed();

    let timings = format!(
        "timings: build {:.1} s, wasm-bindgen {:.1} s, Chrome to result {:.1} s, total {:.1} s",
        build_time.as_secs_f64(),
        bindgen_time.as_secs_f64(),
        page_time.as_secs_f64(),
        started.elapsed().as_secs_f64()
    );
    match outcome {
        Outcome::Result(PageResult::Ok { label }) => Ok(format!(
            "check-web passed: the page logged `{OK_PREFIX}{label}`: the real gust-wgpu vector-add (1000 elements) ran in both forms, grid and pixel, on the browser GPU, and each equalled the host reference and the other bit for bit.\n  evidence: {}\n  {timings}",
            evidence_label(chrome_version.as_deref(), &label)
        )),
        Outcome::Result(PageResult::Fail { reason }) => Err(format!(
            "check-web: the page logged `{FAIL_PREFIX}{reason}`\n  {timings}"
        )),
        Outcome::TimedOut { tail, served } => Err(format!(
            "check-web: no `{OK_PREFIX}` or `{FAIL_PREFIX}` line within {} s ({}); Chrome and the server were stopped\n  {timings}\n{}",
            options.timeout().as_secs(),
            describe_served(&served),
            describe_tail(&tail)
        )),
        Outcome::BrowserGone { tail, served } => Err(format!(
            "check-web: Chrome ended before the page logged a result ({})\n  {timings}\n{}",
            describe_served(&served),
            describe_tail(&tail)
        )),
    }
}

fn describe_served(served: &[String]) -> String {
    if served.is_empty() {
        "the server saw no request".to_owned()
    } else {
        format!("the server was asked for: {}", served.join(", "))
    }
}

fn describe_tail(tail: &[String]) -> String {
    if tail.is_empty() {
        "  Chrome wrote nothing to stderr".to_owned()
    } else {
        format!(
            "  last {} line(s) of Chrome's stderr:\n    {}",
            tail.len(),
            tail.join("\n    ")
        )
    }
}

/// The file name of a path, which is all that is printed of a tool's location.
fn shown_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || "?".to_owned(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// `Gpu (WebGPU, Chrome <version>, <vendor> <architecture>)`; `Gpu (software, WebGPU, ...)` when the adapter says it
/// is a software one (SwiftShader), because VALIDATION.md labels those `Gpu (software)`.
pub(crate) fn evidence_label(version: Option<&str>, adapter: &str) -> String {
    let software = adapter.to_ascii_lowercase().contains("swiftshader");
    let kind = if software {
        "Gpu (software, WebGPU"
    } else {
        "Gpu (WebGPU"
    };
    format!(
        "{kind}, Chrome {}, {adapter})",
        version.unwrap_or("unknown version")
    )
}

// ---------------------------------------------------------------------------------------------------------------
// The feature guard

/// The line `cargo tree -e features -i wgpu` prints for the `webgpu` feature of wgpu, without its tree drawing.
const WEBGPU_FEATURE: &str = "wgpu feature \"webgpu\"";

/// Whether an inverted `cargo tree -e features` of wgpu lists its `webgpu` feature as enabled.
pub(crate) fn tree_enables_webgpu(tree: &str) -> bool {
    tree.lines().any(|line| {
        let line = line.trim_start_matches(['│', '├', '└', '─', ' ', '|', '+', '\\', '`', '-']);
        // Cargo may append an annotation such as ` (command-line)`.
        line.trim_end() == WEBGPU_FEATURE
            || line
                .strip_prefix(WEBGPU_FEATURE)
                .is_some_and(|rest| rest.starts_with(' '))
    })
}

/// What is wrong with the two trees (host target, then `wasm32-unknown-unknown`), one sentence each.
pub(crate) fn feature_scope_problems(host_tree: &str, wasm_tree: &str) -> Vec<String> {
    let mut problems = Vec::new();
    if !host_tree.lines().any(|line| line.starts_with("wgpu v")) {
        problems.push("the host tree does not show wgpu at all, so it proves nothing".to_owned());
    }
    if tree_enables_webgpu(host_tree) {
        problems.push(
            "a host build enables the wgpu `webgpu` feature; it belongs in `[target.'cfg(target_arch = \"wasm32\")'.dependencies]` of gust-wgpu only".to_owned(),
        );
    }
    if !tree_enables_webgpu(wasm_tree) {
        problems.push(
            "the wasm32 build does not enable the wgpu `webgpu` feature, so the browser would find no adapter (and this guard would pass for any host tree)".to_owned(),
        );
    }
    problems
}

/// `cargo tree --locked --workspace -e features -i wgpu`, for the host or for `target`.
fn wgpu_feature_tree(root: &Path, target: Option<&str>) -> Result<String, String> {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut command = Command::new(cargo);
    command
        .args([
            "tree",
            "--locked",
            "--workspace",
            "-e",
            "features",
            "-i",
            "wgpu",
            "--color",
            "never",
        ])
        .current_dir(root)
        .stdin(Stdio::null());
    if let Some(target) = target {
        command.args(["--target", target]);
    }
    let output = command
        .output()
        .map_err(|error| format!("check-web: could not launch cargo tree: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "check-web: cargo tree failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The `webgpu` feature of wgpu is enabled for wasm32 and for nothing else (R4.9: a native build must not gain it).
fn check_feature_scope(root: &Path) -> Result<String, String> {
    let host = wgpu_feature_tree(root, None)?;
    let wasm = wgpu_feature_tree(root, Some(WASM_TARGET))?;
    let problems = feature_scope_problems(&host, &wasm);
    if problems.is_empty() {
        Ok(format!(
            "`cargo tree -e features` shows the wgpu `webgpu` feature for {WASM_TARGET} and not for the host target"
        ))
    } else {
        Err(format!("check-web: {}", problems.join("; ")))
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Tools

/// The version of the `wasm-bindgen` package that Cargo.lock pins; the CLI must be exactly it.
pub(crate) fn locked_wasm_bindgen_version(lock: &str) -> Result<String, String> {
    let mut versions = Vec::new();
    let mut lines = lock.lines().map(str::trim);
    while let Some(line) = lines.next() {
        if line != "name = \"wasm-bindgen\"" {
            continue;
        }
        let version = lines
            .next()
            .and_then(|next| next.strip_prefix("version = \""))
            .and_then(|rest| rest.strip_suffix('"'));
        match version {
            Some(version) if !version.is_empty() => versions.push(version.to_owned()),
            _ => {
                return Err(
                    "check-web: Cargo.lock has a wasm-bindgen entry without a version".into(),
                );
            }
        }
    }
    match versions.as_slice() {
        [version] => Ok(version.clone()),
        [] => Err("check-web: Cargo.lock has no wasm-bindgen package; is examples/web-vector-add in the workspace?".into()),
        many => Err(format!(
            "check-web: Cargo.lock pins several wasm-bindgen versions ({}); the CLI can match only one",
            many.join(", ")
        )),
    }
}

/// The version in the output of `wasm-bindgen --version` (`wasm-bindgen 0.2.127`).
pub(crate) fn parse_cli_version(output: &str) -> Option<String> {
    let mut words = output.split_whitespace();
    (words.next()? == "wasm-bindgen").then_some(())?;
    let version = words.next()?;
    version
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '+')
        .then(|| version.to_owned())
}

/// The first existing file `name` (or `name` plus the platform's executable suffix) in the directories of `path`.
pub(crate) fn find_in_path(name: &str, path: &OsString) -> Option<PathBuf> {
    env::split_paths(path)
        .filter(|directory| !directory.as_os_str().is_empty())
        .flat_map(|directory| {
            [
                directory.join(name),
                directory.join(format!("{name}{EXE_SUFFIX}")),
            ]
        })
        .find(|candidate| candidate.is_file())
}

fn locate_wasm_bindgen(
    options: &Options,
    wanted: &str,
    path_var: Option<OsString>,
) -> Result<PathBuf, String> {
    let path = match &options.wasm_bindgen {
        Some(path) => path.clone(),
        None => path_var
            .and_then(|path| find_in_path("wasm-bindgen", &path))
            .ok_or_else(|| {
                format!(
                    "check-web: wasm-bindgen is not on PATH; pass --wasm-bindgen PATH or install it with `cargo install --locked wasm-bindgen-cli --version {wanted}`"
                )
            })?,
    };
    let output = Command::new(&path)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("check-web: could not run {}: {error}", shown_name(&path)))?;
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    let found = output
        .status
        .success()
        .then(|| parse_cli_version(&text))
        .flatten()
        .ok_or_else(|| {
            format!(
                "check-web: `{} --version` did not print `wasm-bindgen <version>`: {}",
                shown_name(&path),
                text.trim()
            )
        })?;
    if found != wanted {
        return Err(format!(
            "check-web: the wasm-bindgen CLI is {found}, but Cargo.lock pins the library at {wanted}; they must match exactly (`cargo install --locked wasm-bindgen-cli --version {wanted}`)"
        ));
    }
    Ok(path)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Platform {
    Windows,
    Mac,
    Other,
}

impl Platform {
    fn current() -> Self {
        if cfg!(windows) {
            Self::Windows
        } else if cfg!(target_os = "macos") {
            Self::Mac
        } else {
            Self::Other
        }
    }
}

/// Where a Chrome install is looked for, in order. Nothing here checks that the file exists.
pub(crate) fn chrome_candidates(
    platform: Platform,
    variable: &dyn Fn(&str) -> Option<OsString>,
) -> Vec<PathBuf> {
    match platform {
        Platform::Windows => ["ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"]
            .iter()
            .filter_map(|name| variable(name))
            .filter(|base| !base.is_empty())
            .map(|base| {
                PathBuf::from(base)
                    .join("Google")
                    .join("Chrome")
                    .join("Application")
                    .join("chrome.exe")
            })
            .collect(),
        Platform::Mac => vec![PathBuf::from(
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        )],
        Platform::Other => {
            let path = variable("PATH").unwrap_or_default();
            [
                "google-chrome",
                "google-chrome-stable",
                "chromium",
                "chromium-browser",
            ]
            .iter()
            .flat_map(|name| {
                env::split_paths(&path)
                    .filter(|directory| !directory.as_os_str().is_empty())
                    .map(move |directory| directory.join(name))
            })
            .collect()
        }
    }
}

fn locate_chrome(options: &Options) -> Result<PathBuf, String> {
    if let Some(path) = &options.chrome {
        return if path.is_file() {
            Ok(path.clone())
        } else {
            Err(format!(
                "check-web: --chrome {} is not a file",
                shown_name(path)
            ))
        };
    }
    chrome_candidates(Platform::current(), &|name| env::var_os(name))
        .into_iter()
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| {
            "check-web: no Chrome found in the standard install locations; pass --chrome PATH"
                .to_owned()
        })
}

/// `1.2.3.4`-shaped versions only: what `--version` or the file's product version holds, and nothing else.
pub(crate) fn parse_chrome_version(text: &str) -> Option<String> {
    text.split_whitespace()
        .find(|word| {
            let parts: Vec<&str> = word.split('.').collect();
            (2..=4).contains(&parts.len())
                && parts
                    .iter()
                    .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()))
        })
        .map(str::to_owned)
}

/// Chrome's version. `chrome --version` prints nothing on Windows, so there the product version of the executable is
/// read through PowerShell. `None` when it cannot be determined; the check still runs, and says so.
fn chrome_version(chrome: &Path) -> Option<String> {
    let output = if cfg!(windows) {
        Command::new("powershell")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-Item -LiteralPath $env:GUST_CHROME_PATH).VersionInfo.ProductVersion",
            ])
            .env("GUST_CHROME_PATH", chrome)
            .stdin(Stdio::null())
            .output()
    } else {
        Command::new(chrome)
            .arg("--version")
            .stdin(Stdio::null())
            .output()
    }
    .ok()?;
    parse_chrome_version(&String::from_utf8_lossy(&output.stdout))
}

fn require_wasm_target() -> Result<(), String> {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = Command::new(&rustc)
        .args(["--print", "sysroot"])
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("check-web: could not run rustc: {error}"))?;
    let sysroot = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    if sysroot
        .join("lib")
        .join("rustlib")
        .join(WASM_TARGET)
        .is_dir()
    {
        Ok(())
    } else {
        Err(format!(
            "check-web: the {WASM_TARGET} standard library is not installed; run `rustup target add {WASM_TARGET}`"
        ))
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Build

fn build_example(root: &Path, target_dir: &Path) -> Result<PathBuf, String> {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = Command::new(cargo)
        .args([
            "build",
            "--locked",
            "--release",
            "--target",
            WASM_TARGET,
            "-p",
            PACKAGE,
            "--target-dir",
        ])
        .arg(target_dir)
        .env(crate::process::PROFILE_VARIABLE, WEB_PROFILE)
        .current_dir(root)
        .stdin(Stdio::null())
        .status()
        .map_err(|error| format!("check-web: could not launch cargo: {error}"))?;
    if !status.success() {
        return Err(format!(
            "check-web: `cargo build --release --target {WASM_TARGET} -p {PACKAGE}` under {}={WEB_PROFILE} failed ({status})",
            crate::process::PROFILE_VARIABLE
        ));
    }
    let wasm = target_dir
        .join(WASM_TARGET)
        .join("release")
        .join(format!("{OUT_NAME}.wasm"));
    if wasm.is_file() {
        Ok(wasm)
    } else {
        Err(format!(
            "check-web: the build succeeded but produced no {OUT_NAME}.wasm"
        ))
    }
}

/// Recreates `out_dir` with the bindgen output and the page.
fn prepare_site(root: &Path, bindgen: &Path, wasm: &Path, out_dir: &Path) -> Result<(), String> {
    if out_dir.exists() {
        fs::remove_dir_all(out_dir).map_err(|error| {
            format!("check-web: could not clear the previous output directory: {error}")
        })?;
    }
    fs::create_dir_all(out_dir)
        .map_err(|error| format!("check-web: could not create the output directory: {error}"))?;
    let status = Command::new(bindgen)
        .args(["--target", "web", "--no-typescript", "--out-name", OUT_NAME])
        .arg("--out-dir")
        .arg(out_dir)
        .arg(wasm)
        .stdin(Stdio::null())
        .status()
        .map_err(|error| format!("check-web: could not run wasm-bindgen: {error}"))?;
    if !status.success() {
        return Err(format!("check-web: wasm-bindgen failed ({status})"));
    }
    fs::copy(
        root.join(EXAMPLE_DIR).join("index.html"),
        out_dir.join("index.html"),
    )
    .map_err(|error| format!("check-web: could not copy index.html: {error}"))?;
    for name in [format!("{OUT_NAME}.js"), format!("{OUT_NAME}_bg.wasm")] {
        if !out_dir.join(&name).is_file() {
            return Err(format!("check-web: wasm-bindgen produced no {name}"));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// The result line

/// What the page said.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PageResult {
    Ok { label: String },
    Fail { reason: String },
}

/// The result in one line of Chrome's stderr, or `None` for every other line. A console line looks like
/// `[pid:tid:date:INFO:CONSOLE:line] "message", source: <url> (line)`. It counts only when the message starts with
/// `GUST_OK ` or `GUST_FAIL ` and the source starts with `origin`, the address of this check's own server.
pub(crate) fn parse_console_line(line: &str, origin: &str) -> Option<PageResult> {
    let after_marker = &line[line.find(":INFO:CONSOLE:")? + ":INFO:CONSOLE:".len()..];
    let message_and_source = &after_marker[after_marker.find("] \"")? + "] \"".len()..];
    let split = message_and_source.rfind("\", source: ")?;
    let message = &message_and_source[..split];
    let source = &message_and_source[split + "\", source: ".len()..];
    if !source.starts_with(origin) {
        return None;
    }
    if let Some(label) = message.strip_prefix(OK_PREFIX) {
        let label = label.trim();
        return (!label.is_empty()).then(|| PageResult::Ok {
            label: label.to_owned(),
        });
    }
    message
        .strip_prefix(FAIL_PREFIX)
        .map(|reason| PageResult::Fail {
            reason: reason.trim().to_owned(),
        })
}

/// How the wait for the page ended.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    Result(PageResult),
    TimedOut {
        tail: Vec<String>,
        served: Vec<String>,
    },
    BrowserGone {
        tail: Vec<String>,
        served: Vec<String>,
    },
}

const TAIL_LINES: usize = 25;

/// Reads Chrome's stderr lines until one is a result, `deadline` passes, or Chrome is gone (`alive` says so, or the
/// stream ended). `tail` keeps the last [`TAIL_LINES`] lines for the failure message.
pub(crate) fn wait_for_result(
    lines: &Receiver<String>,
    deadline: Instant,
    origin: &str,
    alive: &mut dyn FnMut() -> bool,
) -> (Option<PageResult>, WaitEnd, Vec<String>) {
    let mut tail: Vec<String> = Vec::new();
    let take = |line: String, tail: &mut Vec<String>| {
        let parsed = parse_console_line(&line, origin);
        tail.push(line);
        if tail.len() > TAIL_LINES {
            tail.remove(0);
        }
        parsed
    };
    loop {
        let now = Instant::now();
        if now >= deadline {
            return (None, WaitEnd::TimedOut, tail);
        }
        let step = (deadline - now).min(Duration::from_millis(200));
        match lines.recv_timeout(step) {
            Ok(line) => {
                if let Some(result) = take(line, &mut tail) {
                    return (Some(result), WaitEnd::Result, tail);
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if !alive() {
                    // Whatever Chrome wrote before it ended is already in the channel.
                    while let Ok(line) = lines.try_recv() {
                        if let Some(result) = take(line, &mut tail) {
                            return (Some(result), WaitEnd::Result, tail);
                        }
                    }
                    return (None, WaitEnd::BrowserGone, tail);
                }
            }
            Err(RecvTimeoutError::Disconnected) => return (None, WaitEnd::BrowserGone, tail),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WaitEnd {
    Result,
    TimedOut,
    BrowserGone,
}

// ---------------------------------------------------------------------------------------------------------------
// The loopback server

/// The file name a request target names, or `None` when it names anything else. Only a plain file name of the output
/// directory is served (letters, digits, `.`, `_`, `-`; no directory part, no percent escape, no leading dot); `/` is
/// `index.html`. A query or fragment is ignored.
pub(crate) fn resolve_target(target: &str) -> Option<String> {
    let path = target.split(['?', '#']).next()?;
    let name = path.strip_prefix('/')?;
    if name.is_empty() {
        return Some("index.html".to_owned());
    }
    let plain = name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    (plain && !name.starts_with('.')).then(|| name.to_owned())
}

pub(crate) fn content_type(name: &str) -> &'static str {
    let extension = name.rsplit('.').next().unwrap_or("");
    match extension.to_ascii_lowercase().as_str() {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "wasm" => "application/wasm",
        "json" => "application/json",
        _ => "application/octet-stream",
    }
}

/// A server on `127.0.0.1` and an ephemeral port that serves the plain files of one directory.
pub(crate) struct Server {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    served: Arc<Mutex<Vec<String>>>,
    thread: Option<JoinHandle<()>>,
}

impl Server {
    pub(crate) fn start(directory: PathBuf) -> io::Result<Self> {
        let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0))?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let stop = Arc::new(AtomicBool::new(false));
        let served = Arc::new(Mutex::new(Vec::new()));
        let thread = {
            let (stop, served) = (Arc::clone(&stop), Arc::clone(&served));
            thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let (directory, served) = (directory.clone(), Arc::clone(&served));
                            thread::spawn(move || {
                                let _ = answer(stream, &directory, &served);
                            });
                        }
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(10));
                        }
                        Err(_) => thread::sleep(Duration::from_millis(50)),
                    }
                }
            })
        };
        Ok(Self {
            addr,
            stop,
            served,
            thread: Some(thread),
        })
    }

    pub(crate) fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// `http://127.0.0.1:<port>/`
    pub(crate) fn origin(&self) -> String {
        format!("http://{}/", self.addr)
    }

    /// What was asked for so far, in order.
    pub(crate) fn served(&self) -> Vec<String> {
        self.served
            .lock()
            .map(|served| served.clone())
            .unwrap_or_default()
    }

    /// Stops accepting and joins the accept thread (a request in flight finishes on its own thread).
    pub(crate) fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Reads one request (head only, at most 16 KiB, five seconds) and writes one response.
fn answer(mut stream: TcpStream, directory: &Path, served: &Mutex<Vec<String>>) -> io::Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    let mut head = Vec::new();
    // Reading through `&TcpStream` leaves `stream` whole for the response, with no duplicated handle.
    let mut reader = BufReader::new(&stream);
    loop {
        let mut line = Vec::new();
        if reader
            .by_ref()
            .take(16 * 1024)
            .read_until(b'\n', &mut line)?
            == 0
        {
            return Ok(());
        }
        let blank = line == b"\r\n" || line == b"\n";
        head.extend_from_slice(&line);
        if blank || head.len() > 16 * 1024 {
            break;
        }
    }
    let text = String::from_utf8_lossy(&head);
    let mut first = text.lines().next().unwrap_or("").split_whitespace();
    let (method, target) = (first.next().unwrap_or(""), first.next().unwrap_or(""));
    if let Ok(mut served) = served.lock() {
        served.push(format!(
            "{method} {}",
            target.split(['?', '#']).next().unwrap_or("")
        ));
    }
    let (status, content_type, body) = if method != "GET" {
        (
            "405 Method Not Allowed",
            "text/plain",
            b"method not allowed".to_vec(),
        )
    } else if let Some(name) = resolve_target(target)
        && let Ok(body) = read_plain_file(&directory.join(&name))
    {
        ("200 OK", content_type(&name), body)
    } else {
        ("404 Not Found", "text/plain", b"not found".to_vec())
    };
    let mut response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    response.extend_from_slice(&body);
    stream.write_all(&response)?;
    stream.flush()?;
    // A lingering close: say that nothing more comes, then read until the client has closed too (or a second passes).
    // Closing a socket that still has unread input makes the stack send a reset, which can discard the response the
    // client has not read yet.
    stream.shutdown(Shutdown::Write)?;
    stream.set_read_timeout(Some(Duration::from_secs(1)))?;
    let mut sink = [0u8; 1024];
    let mut drained = 0;
    while drained < 64 * 1024 {
        match stream.read(&mut sink) {
            Ok(0) | Err(_) => break,
            Ok(count) => drained += count,
        }
    }
    Ok(())
}

fn read_plain_file(path: &Path) -> io::Result<Vec<u8>> {
    if !fs::metadata(path)?.is_file() {
        return Err(io::Error::new(io::ErrorKind::NotFound, "not a file"));
    }
    fs::read(path)
}

// ---------------------------------------------------------------------------------------------------------------
// Chrome

/// A running headless Chrome. Dropping it kills the process tree, waits for it and deletes its profile directory.
struct Browser {
    child: Child,
    profile: PathBuf,
    lines: Receiver<String>,
    reader: Option<JoinHandle<()>>,
}

impl Browser {
    fn launch(
        chrome: &Path,
        extra: &[String],
        profile: PathBuf,
        url: &str,
    ) -> Result<Self, String> {
        fs::create_dir_all(&profile).map_err(|error| {
            format!("check-web: could not create Chrome's profile directory: {error}")
        })?;
        let mut command = Command::new(chrome);
        command
            .arg("--headless=new")
            .arg(format!("--user-data-dir={}", profile.display()))
            .args([
                "--no-first-run",
                "--no-default-browser-check",
                "--enable-logging=stderr",
                "--v=0",
            ])
            .args(extra)
            .arg(url)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut command, 0);
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                let _ = fs::remove_dir_all(&profile);
                return Err(format!("check-web: could not launch Chrome: {error}"));
            }
        };
        let stderr = child.stderr.take();
        let (sender, lines) = mpsc::channel();
        let reader = stderr.map(|stderr| {
            thread::spawn(move || {
                let mut reader = BufReader::new(stderr);
                let mut line = Vec::new();
                loop {
                    line.clear();
                    match reader.read_until(b'\n', &mut line) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {
                            let text = String::from_utf8_lossy(&line).trim_end().to_owned();
                            if sender.send(text).is_err() {
                                break;
                            }
                        }
                    }
                }
            })
        });
        Ok(Self {
            child,
            profile,
            lines,
            reader,
        })
    }
}

/// Kills `child` and everything it started, then waits for `child`.
pub(crate) fn kill_tree(child: &mut Child) {
    let pid = child.id();
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(unix)]
    {
        let _ = Command::new("kill")
            .args(["-KILL", "--", &format!("-{pid}")])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// How long `Drop for Browser` waits for the stderr reader after the kill.
const READER_JOIN_LIMIT: Duration = Duration::from_secs(5);

/// Joins `thread` if it ends within `limit`; otherwise leaves it running (detached) and says so with `false`.
fn join_within<T>(thread: thread::JoinHandle<T>, limit: Duration) -> bool {
    let deadline = Instant::now() + limit;
    while !thread.is_finished() {
        if Instant::now() >= deadline {
            return false;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let _ = thread.join();
    true
}

impl Drop for Browser {
    fn drop(&mut self) {
        kill_tree(&mut self.child);
        // The stream ends when the last process that held Chrome's stderr is gone; a process that survived the kill and
        // still holds the pipe must not hang the check, so the wait is bounded and the reader thread is left behind.
        if let Some(reader) = self.reader.take()
            && !join_within(reader, READER_JOIN_LIMIT)
        {
            eprintln!(
                "check-web: warning: Chrome's stderr did not close within {} s (a process may have survived the kill)",
                READER_JOIN_LIMIT.as_secs()
            );
        }
        // The profile can stay locked for a moment after the processes end.
        for _ in 0..40 {
            if fs::remove_dir_all(&self.profile).is_ok() || !self.profile.exists() {
                return;
            }
            thread::sleep(Duration::from_millis(250));
        }
        eprintln!(
            "check-web: warning: could not delete Chrome's profile directory (a process may still hold it)"
        );
    }
}

fn run_in_chrome(
    chrome: &Path,
    options: &Options,
    target_dir: &Path,
    site: &Path,
) -> Result<Outcome, String> {
    let mut server = Server::start(site.to_path_buf())
        .map_err(|error| format!("check-web: could not start the loopback server: {error}"))?;
    if !server.addr().ip().is_loopback() {
        return Err("check-web: the server did not bind a loopback address".into());
    }
    let origin = server.origin();
    println!("check-web: serving the page on {origin} (loopback only)");

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let profile = target_dir
        .join("web-check")
        .join(format!("chrome-{}-{nanos}", std::process::id()));
    let mut browser = Browser::launch(chrome, &options.chrome_args, profile, &origin)?;

    let deadline = Instant::now() + options.timeout();
    let (lines, child) = (&browser.lines, &mut browser.child);
    let (result, end, tail) = wait_for_result(lines, deadline, &origin, &mut || {
        matches!(child.try_wait(), Ok(None))
    });
    let served = server.served();
    // Stop Chrome first (it is the client), then the server; both also stop on drop if anything above had failed.
    drop(browser);
    server.stop();
    Ok(match (result, end) {
        (Some(result), _) => Outcome::Result(result),
        (None, WaitEnd::BrowserGone) => Outcome::BrowserGone { tail, served },
        (None, _) => Outcome::TimedOut { tail, served },
    })
}

#[cfg(test)]
#[path = "tests/web.rs"]
mod web_tests;
