use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    process::{CheckRecord, require_command, run_checked},
    validation::{json_string, sha256_hex},
    workspace::cargo_target_dir,
};

/// Hand-written toolchain fixture used by `doctor --smoke`; not emitted by the Rust compiler.
const SMOKE_SOURCE: &str = ".agents/skills/slang-language/assets/vector-add.slang";
const SMOKE_ENTRY: &str = "computeMain";
const SMOKE_TARGET_ENV: &str = "vulkan1.2";

/// The matrix layout GUST compiles with: column-major, named instead of left to the default, because the command
/// line of slangc defaults to column-major today and the native API to row-major (M6 spike V1). The compile paths of
/// `gust` (`slang::slangc_command`, `reflect::slangc_reflection_command`) pass the same text; the native helper sets
/// `defaultMatrixLayoutMode`. A test compares this constant with their source.
pub(crate) const MATRIX_LAYOUT_FLAG: &str = "-matrix-layout-column-major";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShaderTarget {
    Wgsl,
    Spirv,
}

impl ShaderTarget {
    fn from_name(value: &str) -> Result<Self, String> {
        match value {
            "wgsl" => Ok(Self::Wgsl),
            "spirv" => Ok(Self::Spirv),
            _ => Err(format!(
                "unknown slangc target `{value}`; use wgsl or spirv"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Wgsl => "wgsl",
            Self::Spirv => "spirv",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SlangcOptions {
    source: PathBuf,
    entry: String,
    target: ShaderTarget,
    out: PathBuf,
    extra: Vec<String>,
}

impl SlangcOptions {
    /// `<file.slang> --entry <name> --target wgsl|spirv --out <path> [-- <slangc args>]`
    pub(crate) fn parse(args: &[String]) -> Result<Self, String> {
        let mut source = None;
        let mut entry = None;
        let mut target = None;
        let mut out = None;
        let mut extra = Vec::new();
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--entry" => entry = Some(option_value(args, &mut index, "--entry")?),
                "--target" => {
                    target = Some(ShaderTarget::from_name(&option_value(
                        args, &mut index, "--target",
                    )?)?)
                }
                "--out" => out = Some(PathBuf::from(option_value(args, &mut index, "--out")?)),
                "--" => {
                    extra = args[index + 1..].to_vec();
                    break;
                }
                flag if flag.starts_with('-') => {
                    return Err(format!("unknown slangc option `{flag}`"));
                }
                path => {
                    if source.replace(PathBuf::from(path)).is_some() {
                        return Err("slangc takes exactly one source file".into());
                    }
                }
            }
            index += 1;
        }
        Ok(Self {
            source: source.ok_or("slangc needs a source file")?,
            entry: entry.ok_or("slangc needs --entry <name>")?,
            target: target.ok_or("slangc needs --target wgsl|spirv")?,
            out: out.ok_or("slangc needs --out <path>")?,
            extra,
        })
    }

    /// The exact arguments handed to `slangc`: the entry point and compute stage are explicit, and so is the matrix
    /// layout ([`MATRIX_LAYOUT_FLAG`]), for both targets. Extra arguments come last.
    fn compiler_args(&self) -> Vec<String> {
        let mut args = vec![
            self.source.display().to_string(),
            "-entry".to_owned(),
            self.entry.clone(),
            "-stage".to_owned(),
            "compute".to_owned(),
            "-target".to_owned(),
            self.target.as_str().to_owned(),
            "-o".to_owned(),
            self.out.display().to_string(),
            MATRIX_LAYOUT_FLAG.to_owned(),
        ];
        args.extend(self.extra.iter().cloned());
        args
    }
}

fn option_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

/// `<out>.<suffix>`: records live next to the artifact they describe.
fn sidecar(out: &Path, suffix: &str) -> PathBuf {
    let mut name = out.as_os_str().to_owned();
    name.push(".");
    name.push(suffix);
    PathBuf::from(name)
}

/// Both streams of a finished tool run, trimmed.
fn combined_output(check: &CheckRecord) -> String {
    format!("{}{}", check.stdout, check.stderr)
        .trim()
        .to_owned()
}

/// A tool's version text, read without echoing it (`spirv-val --version` lists every target).
fn tool_version(program: &str, flag: &str) -> Result<String, String> {
    let output = Command::new(program)
        .arg(flag)
        .output()
        .map_err(|error| format!("could not launch {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "could not read the {program} version: `{program} {flag}` failed (exit {})",
            output.status.code().unwrap_or(-1)
        ));
    }
    Ok(format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
    .trim()
    .to_owned())
}

/// `cargo xtask slangc`: compiles one entry point to a fresh output path and leaves a record
/// of the compiler version, arguments and output hash next to it.
pub(crate) fn slangc(options: &SlangcOptions) -> Result<(), String> {
    let record_path = sidecar(&options.out, "command.json");
    let log_path = sidecar(&options.out, "compile.log");
    for path in [&options.out, &record_path, &log_path] {
        if path.exists() {
            return Err(format!(
                "{} already exists; use a fresh output path",
                path.display()
            ));
        }
    }
    if !options.source.is_file() {
        return Err(format!(
            "source file not found: {}",
            options.source.display()
        ));
    }
    require_command("slangc")?;
    if let Some(parent) = options
        .out
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }

    let version = tool_version("slangc", "-version")?;

    let arguments = options.compiler_args();
    let borrowed = arguments.iter().map(String::as_str).collect::<Vec<_>>();
    let mut checks = Vec::new();
    let compiled = run_checked(Path::new("."), &mut checks, "slangc", &borrowed);
    // The record is written even when slangc fails, so the failing command stays inspectable.
    let run = checks.first();
    let log = run.map(combined_output).unwrap_or_default();
    fs::write(&log_path, format!("{log}\n")).map_err(|error| error.to_string())?;
    let artifact = fs::read(&options.out).ok();
    let record = command_record(
        &version,
        &arguments,
        options.target,
        run.and_then(|run| run.exit_code),
        &options.out,
        artifact.as_deref(),
    );
    fs::write(&record_path, record).map_err(|error| error.to_string())?;
    compiled?;

    match artifact {
        None => Err(format!(
            "slangc succeeded but produced no artifact at {}; see {}",
            options.out.display(),
            log_path.display()
        )),
        Some(bytes) if bytes.is_empty() => Err(format!(
            "slangc produced an empty artifact at {}; see {}",
            options.out.display(),
            log_path.display()
        )),
        Some(bytes) => {
            println!(
                "slangc passed: {} target={} {} bytes\n  sha256: {}\n  record: {}",
                options.out.display(),
                options.target.as_str(),
                bytes.len(),
                sha256_hex(&bytes),
                record_path.display()
            );
            Ok(())
        }
    }
}

fn command_record(
    version: &str,
    arguments: &[String],
    target: ShaderTarget,
    exit_code: Option<i32>,
    out: &Path,
    artifact: Option<&[u8]>,
) -> String {
    let arguments = arguments
        .iter()
        .map(|argument| json_string(argument))
        .collect::<Vec<_>>()
        .join(", ");
    let exit_code = exit_code.map_or("null".to_owned(), |code| code.to_string());
    let (bytes, sha256) = match artifact {
        Some(bytes) => (bytes.len().to_string(), json_string(&sha256_hex(bytes))),
        None => ("null".to_owned(), "null".to_owned()),
    };
    let file = out
        .file_name()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
    format!(
        "{{\n  \"schema_version\": 1,\n  \"executable\": \"slangc\",\n  \"version\": {},\n  \"arguments\": [{arguments}],\n  \"exit_code\": {exit_code},\n  \"target\": {},\n  \"output\": {},\n  \"output_bytes\": {bytes},\n  \"output_sha256\": {sha256}\n}}\n",
        json_string(version),
        json_string(target.as_str()),
        json_string(&file),
    )
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SpirvValOptions {
    file: PathBuf,
    target_env: String,
}

impl SpirvValOptions {
    /// `<file.spv> --target-env <env>`
    pub(crate) fn parse(args: &[String]) -> Result<Self, String> {
        let mut file = None;
        let mut target_env = None;
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--target-env" => {
                    target_env = Some(option_value(args, &mut index, "--target-env")?)
                }
                flag if flag.starts_with('-') => {
                    return Err(format!("unknown spirv-val option `{flag}`"));
                }
                path => {
                    if file.replace(PathBuf::from(path)).is_some() {
                        return Err("spirv-val takes exactly one SPIR-V file".into());
                    }
                }
            }
            index += 1;
        }
        Ok(Self {
            file: file.ok_or("spirv-val needs a SPIR-V file")?,
            target_env: target_env.ok_or("spirv-val needs --target-env <env>")?,
        })
    }
}

/// Early format check before the external validator: length, alignment and the
/// little-endian magic number. It cannot establish validity.
pub(crate) fn check_spirv_header(bytes: &[u8]) -> Result<(), String> {
    if bytes.len() < 20 || !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "invalid SPIR-V binary length {} (need at least 20 bytes, a multiple of 4)",
            bytes.len()
        ));
    }
    if bytes[..4] != [0x03, 0x02, 0x23, 0x07] {
        return Err("not a little-endian SPIR-V binary (bad magic number)".into());
    }
    Ok(())
}

/// `cargo xtask spirv-val`: header check, then the external validator for one environment.
pub(crate) fn spirv_val(options: &SpirvValOptions) -> Result<(), String> {
    let bytes = fs::read(&options.file)
        .map_err(|error| format!("could not read {}: {error}", options.file.display()))?;
    check_spirv_header(&bytes)?;
    require_command("spirv-val")?;
    let version = tool_version("spirv-val", "--version")?;
    let file = options.file.display().to_string();
    let mut checks = Vec::new();
    run_checked(
        Path::new("."),
        &mut checks,
        "spirv-val",
        &["--target-env", &options.target_env, &file],
    )?;
    println!(
        "spirv-val passed: {file} ({} bytes)\n  sha256: {}\n  target-env: {}\n  validator: {}",
        bytes.len(),
        sha256_hex(&bytes),
        options.target_env,
        version.lines().next().unwrap_or("unknown")
    );
    println!("  {}", copy_logical_note(&options.file));
    Ok(())
}

/// How many `OpCopyLogical` instructions the text printed by `spirv-dis` holds. The instruction copies between two
/// aggregates of different layouts and is the reason naga's SPIR-V front end refuses some modules (M6 plan, 2.6). A
/// match needs the shape `%id = OpCopyLogical ...`, so a name or string that merely contains the word does not count.
pub(crate) fn count_copy_logical(disassembly: &str) -> usize {
    disassembly
        .lines()
        .filter(|line| {
            let mut words = line.split_whitespace();
            matches!(
                (words.next(), words.next(), words.next()),
                (Some(id), Some("="), Some("OpCopyLogical")) if id.starts_with('%')
            )
        })
        .count()
}

/// The informational line about the `OpCopyLogical` count of one SPIR-V file. It never fails: a missing or failing
/// `spirv-dis` is said in the line instead, and the count decides nothing (GUST's SPIR-V targets Vulkan drivers;
/// feeding it to naga is unsupported).
pub(crate) fn copy_logical_note(file: &Path) -> String {
    let output = Command::new("spirv-dis")
        .arg("--no-color")
        .arg(file)
        .output();
    match output {
        Ok(output) if output.status.success() => format!(
            "OpCopyLogical: {} (informational, never a failure)",
            count_copy_logical(&String::from_utf8_lossy(&output.stdout))
        ),
        Ok(output) => format!(
            "OpCopyLogical: not counted (spirv-dis exited {})",
            output.status.code().unwrap_or(-1)
        ),
        Err(error) => format!("OpCopyLogical: not counted (could not launch spirv-dis: {error})"),
    }
}

/// The compute entry points of a Slang text: the name after each `[shader("compute")]` attribute line, taken from the
/// next line that starts with `void`.
pub(crate) fn compute_entries(slang: &str) -> Vec<String> {
    let mut entries = Vec::new();
    let mut pending = false;
    for line in slang.lines().map(str::trim) {
        if line == "[shader(\"compute\")]" {
            pending = true;
        } else if pending && let Some(rest) = line.strip_prefix("void ") {
            if let Some((name, _)) = rest.split_once('(') {
                entries.push(name.trim().to_owned());
            }
            pending = false;
        }
    }
    entries
}

/// `<stem>__<kernel>.spv` for the entry `gpu_<stem>_<kernel>` of the golden `<stem>.slang`, the spelling of the files
/// the examples export (`<module>__<kernel>.spv`).
pub(crate) fn golden_file_name(stem: &str, entry: &str) -> Result<String, String> {
    let prefix = format!("gpu_{stem}_");
    match entry.strip_prefix(&prefix) {
        Some(kernel) if !kernel.is_empty() => Ok(format!("{stem}__{kernel}.spv")),
        _ => Err(format!(
            "the entry `{entry}` of tests/fixtures/{stem}.slang does not start with `{prefix}`"
        )),
    }
}

/// Compares the SPIR-V files made from the goldens with the expected names and names the differences separately,
/// like the comparison of the exported examples: a renamed kernel is one missing and one unexpected file.
pub(crate) fn compare_golden_names(expected: &[&str], found: &[String]) -> Result<(), String> {
    let mut missing = expected
        .iter()
        .copied()
        .filter(|name| !found.iter().any(|found| found == name))
        .collect::<Vec<_>>();
    let mut unexpected = found
        .iter()
        .map(String::as_str)
        .filter(|name| !expected.contains(name))
        .collect::<Vec<_>>();
    if missing.is_empty() && unexpected.is_empty() {
        return Ok(());
    }
    missing.sort_unstable();
    unexpected.sort_unstable();
    let mut message = format!(
        "The SPIR-V made from the goldens differs from the {} expected names ({} found).",
        expected.len(),
        found.len()
    );
    if !missing.is_empty() {
        message.push_str(&format!(" Missing: {}.", missing.join(", ")));
    }
    if !unexpected.is_empty() {
        message.push_str(&format!(" Unexpected: {}.", unexpected.join(", ")));
    }
    message.push_str(
        " Update the expected names in xtask/src/checks.rs deliberately when a golden gains or loses a kernel.",
    );
    Err(message)
}

/// Compiles every compute entry of `tests/fixtures/<stem>.slang`, for each stem, to SPIR-V in a fresh directory under
/// the cargo target directory (with the column-major matrix layout and Slang's entry-point names, as GUST compiles a
/// kernel) and requires the file names to equal `expected`. Returns the files, sorted by name. The golden lists are
/// kept apart from `EXPECTED_SPIRV`, which names what the examples write to `generated-wgpu`.
pub(crate) fn compile_golden_spirv(
    root: &Path,
    area: &str,
    stems: &[&str],
    expected: &[&str],
) -> Result<Vec<PathBuf>, String> {
    let directory = cargo_target_dir(root).join("golden-spirv").join(area);
    match fs::remove_dir_all(&directory) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("could not clear {}: {error}", directory.display())),
    }
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let fixtures = root.join("tests").join("fixtures");
    let mut found = Vec::new();
    for stem in stems {
        let source = fixtures.join(format!("{stem}.slang"));
        let text = fs::read_to_string(&source)
            .map_err(|error| format!("could not read {}: {error}", source.display()))?;
        let entries = compute_entries(&text);
        if entries.is_empty() {
            return Err(format!(
                "tests/fixtures/{stem}.slang has no compute entry point"
            ));
        }
        for entry in entries {
            let file = golden_file_name(stem, &entry)?;
            slangc(&SlangcOptions {
                source: source.clone(),
                entry,
                target: ShaderTarget::Spirv,
                out: directory.join(&file),
                extra: vec!["-fvk-use-entrypoint-name".to_owned()],
            })?;
            found.push(file);
        }
    }
    compare_golden_names(expected, &found)?;
    found.sort_unstable();
    Ok(found.iter().map(|file| directory.join(file)).collect())
}

/// [`compile_golden_spirv`], then `spirv-val` for Vulkan 1.2 on each file (which prints the informational
/// `OpCopyLogical` count). `check-feature vectors` and `check-feature uniforms` call this; `verify` (`check-artifacts`,
/// the Spirv suite, `check-full`) compiles the same files and validates them as recorded runs.
pub(crate) fn export_golden_spirv(
    root: &Path,
    area: &str,
    stems: &[&str],
    expected: &[&str],
) -> Result<(), String> {
    require_command("spirv-val")?;
    for file in compile_golden_spirv(root, area, stems, expected)? {
        spirv_val(&SpirvValOptions {
            file,
            target_env: SMOKE_TARGET_ENV.to_owned(),
        })?;
    }
    Ok(())
}

/// The smoke half of `doctor --smoke`: compile the hand-written vector-add kernel to both
/// targets in a fresh directory and validate the SPIR-V. Rust frontend and GPU execution are
/// out of scope and said so.
pub(crate) fn doctor_smoke(root: &Path) -> Result<(), String> {
    require_command("spirv-val")?;
    let directory = fresh_smoke_directory(root)?;
    let source = root.join(SMOKE_SOURCE);
    for (target, file) in [
        (ShaderTarget::Wgsl, "vector-add.wgsl"),
        (ShaderTarget::Spirv, "vector-add.spv"),
    ] {
        slangc(&SlangcOptions {
            source: source.clone(),
            entry: SMOKE_ENTRY.to_owned(),
            target,
            out: directory.join(file),
            extra: Vec::new(),
        })?;
    }
    spirv_val(&SpirvValOptions {
        file: directory.join("vector-add.spv"),
        target_env: SMOKE_TARGET_ENV.to_owned(),
    })?;
    println!(
        "Toolchain smoke passed: hand-written Slang compiled to WGSL and SPIR-V and the SPIR-V validated for {SMOKE_TARGET_ENV}.\nThe Rust frontend and GPU execution were NOT run. Evidence: {}",
        directory.display()
    );
    Ok(())
}

/// A new, empty directory under the cargo target directory; never reuses an existing one.
fn fresh_smoke_directory(root: &Path) -> Result<PathBuf, String> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let base = cargo_target_dir(root).join("doctor-smoke");
    fs::create_dir_all(&base).map_err(|error| error.to_string())?;
    let directory = base.join(format!("{stamp}-{}", std::process::id()));
    fs::create_dir(&directory).map_err(|error| error.to_string())?;
    Ok(directory)
}

#[cfg(test)]
#[path = "tests/shader.rs"]
mod shader_tests;
