use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use crate::{
    arch::check_arch,
    checks::{GOLDEN_SPIRV_SETS, check_dual, tests_not_ok},
    difftest::check_difftest_report,
    docs::docs_check,
    lint::lint,
    precompiled::check_precompiled_report,
    process::{
        CheckRecord, report_check, require_command, require_tests_ran, run_cargo_test_captured,
        run_checked, run_checked_owned, sum_test_results, without_colors,
    },
    reflect::{BuildOptions, build_slang_reflect},
    shader::{compile_golden_spirv, copy_logical_note},
    suite::DIFFTEST_CARGO_CALLS,
    time::now_utc,
    ui::check_ui,
    validation::{AdapterRecord, ArtifactRecord, GitStamp, RunInfo, sha256_hex, write_validation},
    web::{check_web_report, preflight_tools},
    workspace::{EXAMPLES, EXPECTED_SPIRV, SMOKE_EXAMPLES, workspace_root, workspace_test_args},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VerifyMode {
    Smoke,
    Fast,
    Gpu,
    Examples,
    Artifacts,
    Full,
}

impl VerifyMode {
    pub(crate) fn from_name(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "smoke" => Ok(Self::Smoke),
            "fast" => Ok(Self::Fast),
            "gpu" => Ok(Self::Gpu),
            "examples" => Ok(Self::Examples),
            "artifacts" => Ok(Self::Artifacts),
            "full" => Ok(Self::Full),
            _ => Err(format!("unknown verify mode `{value}`")),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Fast => "fast",
            Self::Gpu => "gpu",
            Self::Examples => "examples",
            Self::Artifacts => "artifacts",
            Self::Full => "full",
        }
    }
}

pub(crate) struct VerifyCommand {
    pub(crate) mode: VerifyMode,
    pub(crate) record: bool,
    /// `--skip-web`: the Full mode leaves out `check-web` (see [`integration_checks`]).
    pub(crate) skip_web: bool,
}

/// The flag that lets the Full mode go without the browser run, on a machine that has no Chrome or no WebGPU adapter.
pub(crate) const SKIP_WEB_FLAG: &str = "--skip-web";

impl VerifyCommand {
    pub(crate) fn parse(args: &[String]) -> Result<Self, String> {
        let mut mode = VerifyMode::Smoke;
        let mut record = None;
        let mut skip_web = false;
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                SKIP_WEB_FLAG => skip_web = true,
                "--full" | "-Full" | "--Full" => mode = VerifyMode::Full,
                "--mode" | "-Mode" => {
                    index += 1;
                    let Some(value) = args.get(index) else {
                        return Err("--mode needs a value".into());
                    };
                    mode = VerifyMode::from_name(value)?;
                }
                "--record" => record = Some(true),
                "--no-record" => record = Some(false),
                other => mode = VerifyMode::from_name(other)?,
            }
            index += 1;
        }
        if skip_web && mode != VerifyMode::Full {
            return Err(format!(
                "{SKIP_WEB_FLAG} applies to the full mode only (the other modes never run check-web)"
            ));
        }
        Ok(Self {
            mode,
            record: record.unwrap_or(mode == VerifyMode::Full),
            skip_web,
        })
    }
}

/// One of the integration checks that the Full mode runs after its own steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Integration {
    /// Run `cargo xtask <name>` in process and record it.
    Run(&'static str),
    /// Leave it out because the caller asked to (`--skip-web`); the skip is printed and recorded.
    Skip(&'static str),
}

/// The integration checks of a mode, in order (plan M4 section 14, item 3; plan M5 R1-13). The Full mode runs the
/// whole `check-precompiled`, the whole `check-web` and, last, `check-difftest` (all eight seeds on the hardware
/// adapter, plus the `short_circuit` GPU test: m5.c2), with no silent skip: a machine without Chrome or a WebGPU
/// adapter fails the web check, naming the cause, unless `skip_web` turns it into a recorded [`Integration::Skip`].
/// The batch comes last because it is the only one of the three whose verdict depends on generated programs, so a
/// disagreement it finds does not hide whether the other two passed. The artifact export and `spirv-val` checks of
/// `check-artifacts` are part of the Full mode's own steps. The other modes run none of these.
pub(crate) fn integration_checks(mode: VerifyMode, skip_web: bool) -> Vec<Integration> {
    if mode != VerifyMode::Full {
        return Vec::new();
    }
    vec![
        Integration::Run("check-precompiled"),
        if skip_web {
            Integration::Skip("check-web")
        } else {
            Integration::Run("check-web")
        },
        Integration::Run("check-difftest"),
    ]
}

/// `cargo xtask check-full [--skip-web]`: the Full mode, always recorded.
pub(crate) fn check_full(args: &[String]) -> Result<(), String> {
    match args {
        [] => verify(VerifyMode::Full, true),
        [flag] if flag == SKIP_WEB_FLAG => verify_with(VerifyMode::Full, true, true),
        _ => Err(format!(
            "check-full takes no argument except {SKIP_WEB_FLAG}"
        )),
    }
}

pub(crate) fn verify(mode: VerifyMode, record: bool) -> Result<(), String> {
    verify_with(mode, record, false)
}

/// [`verify`] with `--skip-web`. Only the Full mode takes it ([`VerifyCommand::parse`] refuses it elsewhere).
pub(crate) fn verify_with(mode: VerifyMode, record: bool, skip_web: bool) -> Result<(), String> {
    let root = workspace_root()?;
    let started = now_utc();
    // Taken before the checks run: the record names the tree they ran against.
    let git = GitStamp::read(&root);
    let mut checks = Vec::new();
    let mut artifacts = Vec::new();
    let mut adapter = None;
    let mut skipped = Vec::new();
    let result = verify_inner(
        &root,
        mode,
        skip_web,
        &mut checks,
        &mut artifacts,
        &mut adapter,
        &mut skipped,
    );
    if record {
        let run = RunInfo {
            mode,
            started: &started,
            git: &git,
            adapter: adapter.as_ref(),
            skipped: &skipped,
        };
        write_validation(&root, &run, result.as_ref().err(), &checks, &artifacts)?;
    }
    result?;
    if skipped.is_empty() {
        println!("GUST verification passed.");
    } else {
        println!(
            "GUST verification passed, with {} skipped on request: this is not the full release record.",
            skipped.join(", ")
        );
    }
    Ok(())
}

/// The tools of `check-web` are looked up at the start of a Full run that will run it, with `probe` (check-web's own
/// lookups). Every other run, including Full with `--skip-web`, does not look: it never runs `check-web`.
fn web_preflight(
    mode: VerifyMode,
    skip_web: bool,
    probe: impl FnOnce() -> Result<(), String>,
) -> Result<(), String> {
    if mode != VerifyMode::Full || skip_web {
        return Ok(());
    }
    probe().map_err(|error| {
        format!(
            "{error}\n  check-full runs check-web, which needs this tool; install it, or pass {SKIP_WEB_FLAG} \
             (the run is then recorded as not the full release record)"
        )
    })
}

/// Run one integration check in process and record it like `check-dual`. On failure the returned error carries the
/// check's own message (the cause), not the generic "found 1 problem".
fn run_integration(
    checks: &mut Vec<CheckRecord>,
    name: &'static str,
    outcome: Result<String, String>,
) -> Result<(), String> {
    let cause = outcome.as_ref().err().cloned();
    report_check(checks, name, outcome.map_err(|error| vec![error])).map_err(
        |generic| match cause {
            Some(cause) if cause.starts_with(name) => cause,
            Some(cause) => format!("{name} failed: {cause}"),
            None => generic,
        },
    )
}

fn verify_inner(
    root: &Path,
    mode: VerifyMode,
    skip_web: bool,
    checks: &mut Vec<CheckRecord>,
    artifacts: &mut Vec<ArtifactRecord>,
    adapter: &mut Option<AdapterRecord>,
    skipped: &mut Vec<String>,
) -> Result<(), String> {
    // Before anything else: a missing tool of check-web must not show up after the long steps.
    web_preflight(mode, skip_web, preflight_tools)?;
    // Cheapest checks first; they need only git, and check-ui needs cargo (no slangc, no GPU).
    if matches!(
        mode,
        VerifyMode::Smoke | VerifyMode::Fast | VerifyMode::Full
    ) {
        lint(root, checks)?;
        docs_check(root, checks)?;
        check_arch(root, checks)?;
        check_ui(root, checks)?;
    }
    require_command("cargo")?;
    require_command("rustc")?;
    require_command("slangc")?;
    if matches!(mode, VerifyMode::Artifacts | VerifyMode::Full) {
        require_command("spirv-val")?;
    }
    run_checked(root, checks, "rustc", &["--version"])?;
    run_checked(root, checks, "slangc", &["-version"])?;
    let slangc_output = checks.last().map(printed_text).unwrap_or_default();
    prepare_helper(root, Some(&slangc_output))?;
    // Before the first GPU step: no adapter, or one that cannot be named, fails the run here and the record says null.
    *adapter = Some(record_adapter(root, checks)?);
    clear_generated_output(root, mode)?;
    for args in cargo_calls(mode) {
        run_checked_owned(root, checks, "cargo", &args)?;
        if let Some(record) = checks.last_mut()
            && let Err(error) = guard_call(mode, &args, &record.stdout)
        {
            // The process exited 0, so the record must not say success: no exit status, and the reason.
            record.exit_code = None;
            record
                .stderr
                .push_str(&format!("test-count guard: {error}\n"));
            return Err(error);
        }
    }
    if runs_check_dual(mode) {
        // The dual build and the example runs with the CPU twin (D26): its cargo calls are its own, with their own
        // target directory and rustflags, so it runs in process and is recorded like the other xtask checks.
        let outcome = check_dual(&[])
            .map(|()| "check-dual passed (full mode)\n".to_owned())
            .map_err(|error| vec![error]);
        report_check(checks, "check-dual", outcome)?;
    }
    if matches!(mode, VerifyMode::Artifacts | VerifyMode::Full) {
        run_checked(root, checks, "spirv-val", &["--version"])?;
        let mut spv = fs::read_dir(root.join(GENERATED_DIR))
            .map_err(|error| error.to_string())?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "spv"))
            .collect::<Vec<_>>();
        spv.sort();
        let found = spv
            .iter()
            .filter_map(|path| path.file_name())
            .map(|name| name.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        compare_artifact_names(EXPECTED_SPIRV, &found)?;
        // The goldens of tests/fixtures (vectors, products, uniforms) are compiled here and validated like the
        // examples' files: no example uses a vector or a uniform, so nothing else covers their SPIR-V. Their expected
        // names are kept apart from EXPECTED_SPIRV.
        for (area, stems, expected) in GOLDEN_SPIRV_SETS {
            spv.extend(compile_golden_spirv(root, area, stems, expected)?);
        }
        for artifact in spv {
            let path = artifact.to_string_lossy().into_owned();
            run_checked(
                root,
                checks,
                "spirv-val",
                &["--target-env", "vulkan1.2", &path],
            )?;
            let file = artifact
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<invalid>")
                .to_owned();
            let bytes = fs::read(&artifact).map_err(|error| error.to_string())?;
            artifacts.push(ArtifactRecord {
                file: file.clone(),
                sha256: sha256_hex(&bytes),
            });
            println!("Validated {file}");
            println!("  {}", copy_logical_note(&artifact));
        }
    }
    // Last: the slowest checks, and the ones that need the most (a registry download, Chrome with a WebGPU adapter).
    for step in integration_checks(mode, skip_web) {
        match step {
            Integration::Run("check-difftest") => {
                let outcome = check_difftest_report(&[]).map(|summary| format!("{summary}\n"));
                run_integration(checks, "check-difftest", outcome)?;
            }
            Integration::Run("check-precompiled") => {
                let outcome = check_precompiled_report(&[]).map(|summary| format!("{summary}\n"));
                run_integration(checks, "check-precompiled", outcome)?;
            }
            Integration::Run("check-web") => {
                let outcome = check_web_report(&[]).map(|summary| format!("{summary}\n"));
                run_integration(checks, "check-web", outcome)?;
            }
            Integration::Run(other) => return Err(format!("unknown integration check `{other}`")),
            Integration::Skip(name) => {
                println!(
                    "{name}: SKIPPED on request ({SKIP_WEB_FLAG}); this run is not the full release record, \
                     and VALIDATION.json says so"
                );
                skipped.push(name.to_owned());
            }
        }
    }
    Ok(())
}

/// The variable that names the reflection helper the tests use (`gust::reflect::native_compiler_path`).
const HELPER_VARIABLE: &str = "GUST_SLANG_REFLECT";

/// The helper file `GUST_SLANG_REFLECT` names, when it is set and the file exists. Anything else (unset, empty, a
/// missing file) is `None`, and the caller builds the helper into the default place as before.
pub(crate) fn environment_helper(value: Option<OsString>) -> Option<PathBuf> {
    let path = PathBuf::from(value.filter(|value| !value.is_empty())?);
    path.is_file().then_some(path)
}

/// What `verify` does about the reflection helper.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum HelperPlan {
    /// Build it into the default place, `<root>/target/slang-reflect`, refreshing it when it is out of date.
    Build,
    /// Use this file, which `GUST_SLANG_REFLECT` names; it is checked against `slangc` first ([`check_helper_tag`]).
    Use(PathBuf),
}

/// A helper that `GUST_SLANG_REFLECT` names is used as it is, except when it is the default output of this checkout
/// (`<root>/target/slang-reflect/gust-slang-reflect`): that one is still built, so it follows a change of the helper
/// source or of Slang as before (the local wrapper names exactly that file). Anything else, or no variable, or a
/// missing file, builds into the default place.
pub(crate) fn helper_plan(root: &Path, value: Option<OsString>) -> HelperPlan {
    let Some(path) = environment_helper(value) else {
        return HelperPlan::Build;
    };
    let default = default_helper(root);
    match (fs::canonicalize(&path), fs::canonicalize(&default)) {
        (Ok(named), Ok(default)) if named == default => HelperPlan::Build,
        _ => HelperPlan::Use(path),
    }
}

/// The file `cargo xtask build-slang-reflect` writes by default: `<root>/target/slang-reflect/gust-slang-reflect`.
fn default_helper(root: &Path) -> PathBuf {
    root.join("target")
        .join("slang-reflect")
        .join(format!("gust-slang-reflect{}", env::consts::EXE_SUFFIX))
}

/// The file that holds the pin of `cargo gust`, the one place the version lives (the workflows read it there too).
const PIN_SOURCE: &str = "crates/cargo-gust/src/slang.rs";

/// The Slang version `cargo gust` is pinned to: the string of the one `SLANG_VERSION` constant in `source`. `None`
/// when there is not exactly one such line or its value is empty.
pub(crate) fn slang_pin(source: &str) -> Option<String> {
    let mut found = source.lines().filter_map(|line| {
        line.trim()
            .strip_prefix("pub(crate) const SLANG_VERSION: &str = \"")?
            .strip_suffix("\";")
            .map(str::to_owned)
    });
    let pin = found.next()?;
    (found.next().is_none() && !pin.is_empty() && !pin.contains('"')).then_some(pin)
}

/// The other `slangc -version` tags that count as the pin: the strings of the `VALIDATED_BUILDS` list of `source`
/// (empty when the list is empty). Together with [`slang_pin`] this is the rule of `matches_pin` in `slang.rs`,
/// read from the same source so the two cannot drift: adding a validated build there admits it here too. `None`
/// when the constant is missing or not closed.
pub(crate) fn slang_validated_builds(source: &str) -> Option<Vec<String>> {
    let rest = source
        .split_once("pub(crate) const VALIDATED_BUILDS: &[&str] = &[")?
        .1;
    let list = rest.split_once("];")?.0;
    let mut builds = Vec::new();
    for line in list.lines() {
        let line = line.split("//").next().unwrap_or_default();
        for piece in line.split('"').skip(1).step_by(2) {
            builds.push(piece.to_owned());
        }
    }
    Some(builds)
}

/// The pin of `cargo gust` and the extra tags that count as it, from [`PIN_SOURCE`].
pub(crate) struct PinRule {
    pub(crate) pin: String,
    pub(crate) validated: Vec<String>,
}

impl PinRule {
    /// The rule of `matches_pin` in `slang.rs`: the tag is the pin, or it extends the pin and is a validated build.
    pub(crate) fn admits(&self, tag: &str) -> bool {
        tag == self.pin || (tag.starts_with(&self.pin) && self.validated.iter().any(|v| v == tag))
    }
}

pub(crate) fn pin_rule_of(source: &str) -> Result<PinRule, String> {
    Ok(PinRule {
        pin: slang_pin(source)
            .ok_or_else(|| format!("cannot read the one SLANG_VERSION constant of {PIN_SOURCE}"))?,
        validated: slang_validated_builds(source)
            .ok_or_else(|| format!("cannot read the VALIDATED_BUILDS list of {PIN_SOURCE}"))?,
    })
}

fn pin_rule(root: &Path) -> Result<PinRule, String> {
    let source = fs::read_to_string(root.join(PIN_SOURCE))
        .map_err(|error| format!("cannot read {PIN_SOURCE}: {}", error.kind()))?;
    pin_rule_of(&source)
}

/// The reflection schemas xtask accepts from the helper: `schema=2` since m6.r, `schema=3` since mx.t0a (the same
/// documents for every buffer kernel; schema 3 only adds textures, samplers and the fragment stage).
const HELPER_SCHEMAS: [&str; 2] = ["2", "3"];

/// Fails unless the helper writes the schema xtask reads and was built against the Slang the pin names (the pin
/// itself, or a validated build, see [`PinRule`]). `which` says what the helper is, for the message, which names both
/// tags and how to get a helper of the pin.
pub(crate) fn check_helper_pin(
    helper_output: &str,
    rule: &PinRule,
    which: &str,
) -> Result<(), String> {
    let Some(tag) = helper_tag(helper_output) else {
        return Err(format!("{which} printed no `slang=<tag>` for `--version`"));
    };
    let pin = &rule.pin;
    if !rule.admits(tag) {
        return Err(format!(
            "{which} was built against Slang `{tag}`, but the pin is `{pin}`; use the helper of the managed install, \
             `<root>/helper/{pin}/gust-slang-reflect{suffix}` (`cargo gust slang install`, then `cargo gust slang \
             build-helper`; the root is GUST_SLANG_HOME, by default `<CARGO_HOME or ~/.cargo>/gust/slang`), and \
             name it with {HELPER_VARIABLE}, with SLANG_SDK pointing at that install when xtask builds the helper \
             itself; do not overwrite a helper that other builds share",
            suffix = env::consts::EXE_SUFFIX
        ));
    }
    let schema = helper_output
        .split_whitespace()
        .find_map(|word| word.strip_prefix("schema="));
    if !schema.is_some_and(|schema| HELPER_SCHEMAS.contains(&schema)) {
        let accepted = HELPER_SCHEMAS
            .iter()
            .map(|schema| format!("`schema={schema}`"))
            .collect::<Vec<_>>()
            .join(" or ");
        return Err(format!(
            "{which} prints `schema={}`, but xtask needs {accepted}; rebuild it with the current \
             `cargo gust slang build-helper`",
            schema.unwrap_or("<none>")
        ));
    }
    Ok(())
}

/// Fails unless `slangc -version` reports the pin (or a validated build, see [`PinRule`]). It runs on every path of
/// [`prepare_helper`], so a managed `SLANG_SDK` and a helper of the pin do not hide an older `slangc` that comes first
/// on `PATH`: the tools would run as a mixed toolchain.
pub(crate) fn check_slangc_pin(slangc_output: &str, rule: &PinRule) -> Result<(), String> {
    let Some(tag) = slangc_output
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
    else {
        return Err("`slangc -version` printed nothing".to_owned());
    };
    if rule.admits(tag) {
        return Ok(());
    }
    Err(format!(
        "`slangc -version` is `{tag}`, but the pin is `{}`; put the pinned slangc first on PATH (`cargo gust slang \
         install`, then the `bin` directory of `cargo gust slang locate`) and point SLANG_SDK at that install",
        rule.pin
    ))
}

/// Makes the reflection helper ready for the checks that follow (`verify`, `check-feature reflection`, `doctor`).
///
/// A helper that `GUST_SLANG_REFLECT` names (see [`helper_plan`]) is used as it is: it is not rebuilt, and nothing
/// is built into `target/slang-reflect`. It must carry the pin's Slang build tag, and, when `slangc_output` is given
/// (the output of `slangc -version`), that compiler's tag as well; a helper of another build is refused with a
/// message that names the tags. Otherwise the helper is built from `SLANG_SDK` or `VULKAN_SDK` into the default
/// place, and the result must carry the pin too, so a Slang SDK of another version is refused instead of being used
/// silently. On every path the `slangc` on `PATH` must report the pin as well: `slangc_output` when the caller ran
/// `slangc -version` already, else it is run here (a `PATH` without `slangc` is not an error here: the callers that
/// need one require it themselves). The helper's path is not printed: it is a machine path.
pub(crate) fn prepare_helper(root: &Path, slangc_output: Option<&str>) -> Result<(), String> {
    prepare_helper_with(
        root,
        env::var_os(HELPER_VARIABLE),
        slangc_output,
        &|| build_slang_reflect(BuildOptions::default()),
        &helper_version_output,
        &path_slangc_version,
    )
}

/// What `slangc -version` on `PATH` prints (standard output and error), or `None` when it cannot be run.
fn path_slangc_version() -> Option<String> {
    let output = Command::new("slangc").arg("-version").output().ok()?;
    Some(format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

/// [`prepare_helper`] with the variable, the build, the `--version` run of a helper and the `slangc -version` run on
/// `PATH` given, so the tests can judge each path.
pub(crate) fn prepare_helper_with(
    root: &Path,
    variable: Option<OsString>,
    slangc_output: Option<&str>,
    build: &dyn Fn() -> Result<(), String>,
    version_of: &dyn Fn(&Path) -> Result<String, String>,
    slangc_now: &dyn Fn() -> Option<String>,
) -> Result<(), String> {
    let rule = pin_rule(root)?;
    let pin = &rule.pin;
    let slangc = slangc_output.map(str::to_owned).or_else(slangc_now);
    if let Some(slangc) = &slangc {
        check_slangc_pin(slangc, &rule)?;
    }
    match helper_plan(root, variable) {
        HelperPlan::Build => {
            build()?;
            let built = version_of(&default_helper(root))?;
            check_helper_pin(&built, &rule, "the helper built into target/slang-reflect")
        }
        HelperPlan::Use(helper) => {
            let named = version_of(&helper)?;
            check_helper_pin(
                &named,
                &rule,
                &format!("the helper named by {HELPER_VARIABLE}"),
            )?;
            if let Some(slangc) = &slangc {
                check_helper_tag(&named, slangc)?;
            }
            println!(
                "Using the reflection helper named by {HELPER_VARIABLE}: its Slang tag is the pin `{pin}`."
            );
            Ok(())
        }
    }
}

/// The output of `<helper> --version`. A failure names the variable, not the path, and hides the path in the helper's
/// own message.
fn helper_version_output(helper: &Path) -> Result<String, String> {
    let output = Command::new(helper)
        .arg("--version")
        .output()
        .map_err(|error| {
            format!(
                "could not run `--version` of the helper named by {HELPER_VARIABLE}: {}",
                error.kind()
            )
        })?;
    let stderr = String::from_utf8_lossy(&output.stderr)
        .replace(helper.to_string_lossy().as_ref(), "<helper>");
    if !output.status.success() {
        return Err(format!(
            "`--version` of the helper named by {HELPER_VARIABLE} failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            stderr.trim()
        ));
    }
    Ok(format!(
        "{}\n{stderr}",
        String::from_utf8_lossy(&output.stdout)
    ))
}

/// Everything a finished check printed. `slangc -version` prints its version on standard error (Linux, release
/// archive), so the standard output alone can be empty.
pub(crate) fn printed_text(record: &CheckRecord) -> String {
    format!("{}\n{}", record.stdout, record.stderr)
}

/// The Slang build tag a helper reports: the `slang=<tag>` word of `gust-slang-reflect schema=2 slang=2026.20`.
pub(crate) fn helper_tag(output: &str) -> Option<&str> {
    output
        .split_whitespace()
        .find_map(|word| word.strip_prefix("slang="))
        .filter(|tag| !tag.is_empty())
}

/// Fails unless the helper was built against the compiler `slangc -version` names: its tag must equal the first
/// line of that output exactly (a release prints the bare pin, a source build its own tag, and the helper reports
/// the same one for the same install). The message names both tags.
pub(crate) fn check_helper_tag(helper_output: &str, slangc_output: &str) -> Result<(), String> {
    let Some(helper) = helper_tag(helper_output) else {
        return Err(format!(
            "the helper named by {HELPER_VARIABLE} printed no `slang=<tag>` for `--version`"
        ));
    };
    let Some(slangc) = slangc_output
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
    else {
        return Err("`slangc -version` printed nothing".to_owned());
    };
    if helper == slangc {
        return Ok(());
    }
    Err(format!(
        "the helper named by {HELPER_VARIABLE} was built against Slang `{helper}`, but `slangc -version` is `{slangc}`; \
         rebuild it (`cargo gust slang build-helper`) or unset {HELPER_VARIABLE}"
    ))
}

/// The directory the examples export their Slang, WGSL and SPIR-V to (ignored by git, never committed).
const GENERATED_DIR: &str = "generated-wgpu";

/// The cargo call that prints the selected adapter as one JSON line (`crates/gust-wgpu/examples/adapter_info.rs`).
pub(crate) const ADAPTER_INFO_ARGS: [&str; 7] = [
    "run",
    "--locked",
    "--quiet",
    "-p",
    "gust-wgpu",
    "--example",
    "adapter_info",
];

/// Runs [`ADAPTER_INFO_ARGS`] and reads the adapter from its output. A failed run, a run that prints nothing, and
/// output that does not name an adapter all fail the step; the record of the check then carries no exit code.
fn record_adapter(root: &Path, checks: &mut Vec<CheckRecord>) -> Result<AdapterRecord, String> {
    run_checked(root, checks, "cargo", &ADAPTER_INFO_ARGS)?;
    let Some(record) = checks.last_mut() else {
        return Err("the adapter step left no check record".to_owned());
    };
    adapter_from_check(record)
}

/// Reads the adapter from the captured output of the finished adapter check. When the output does not name one, the
/// process still exited 0, so the record must not say success: no exit code, and the reason on stderr.
pub(crate) fn adapter_from_check(record: &mut CheckRecord) -> Result<AdapterRecord, String> {
    AdapterRecord::parse(&record.stdout).inspect_err(|error| {
        record.exit_code = None;
        record.stderr.push_str(&format!("adapter guard: {error}\n"));
    })
}

/// Whether a mode checks the exported artifacts and so starts from an empty [`GENERATED_DIR`].
pub(crate) fn clears_generated_output(mode: VerifyMode) -> bool {
    matches!(mode, VerifyMode::Artifacts | VerifyMode::Full)
}

/// Empties [`GENERATED_DIR`] under `root` before the examples export into it, for the modes that check the
/// artifacts. The directory is ignored by git and survives between runs, so without this a file of a deleted or
/// renamed kernel would still satisfy the name check and pass `spirv-val`. A missing directory is fine (the examples
/// create it); any other failure fails the step, because stale files would then remain.
pub(crate) fn clear_generated_output(root: &Path, mode: VerifyMode) -> Result<(), String> {
    if !clears_generated_output(mode) {
        return Ok(());
    }
    let directory = root.join(GENERATED_DIR);
    // `remove_dir_all` removes a symlink or junction itself, not its target.
    match fs::remove_dir_all(&directory) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("could not clear {GENERATED_DIR}: {error}")),
    }
    fs::create_dir_all(&directory)
        .map_err(|error| format!("could not recreate {GENERATED_DIR}: {error}"))
}

/// The Slang tests of the core library in the release profile (the Full mode and suite).
/// `cfg(not(debug_assertions))` tests, such as the source-map fallback, run only here. The two filters after `--`
/// are ORed by libtest: `slang` (everything under the Slang emitter, as before) and the one release-only `gust::cpu`
/// test, which the `slang` substring does not match.
pub(crate) const RELEASE_SLANG_TESTS: [&str; 9] = [
    "test",
    "--locked",
    "--release",
    "-p",
    "gust",
    "--lib",
    "--",
    "slang",
    "cpu::tests::invocations_are_capped_at_the_id_range_in_release",
];

/// The known-mismatch ledger in the release profile (the Full mode). The ledger asserts the plain-Rust outcome
/// of every row for the `overflow-checks` mode it detects at run time (D38); the debug run (Gpu suite) covers
/// checks on, this run covers checks off, so both branches of every mode-keyed row run on every full gate.
/// `--nocapture` (m5.f1, review) lets the run print, on stderr, the adapter, the detected overflow mode and the `S2 |`
/// rows of the ledger: that is the release evidence in the log. The tests print to stderr only, so the `test .. ok`
/// lines of standard output, which the guard reads ([`check_counted_call`]), stay whole.
pub(crate) const RELEASE_LEDGER_TEST: [&str; 9] = [
    "test",
    "--locked",
    "--release",
    "-p",
    "gust-wgpu",
    "--test",
    "mismatch",
    "--",
    "--nocapture",
];

/// The tests of `crates/gust-wgpu/tests/mismatch.rs`, all of which the release ledger call must report `ok`. A count
/// alone would pass with one of them `#[ignore]`d, renamed or deleted, because the others still run; the guard
/// ([`check_counted_call`]) therefore also requires `0 ignored` and each of these names (m5.f1, sanity checks of M5).
/// Adding, renaming or removing a test of that file changes this list in the same commit.
pub(crate) const RELEASE_LEDGER_TESTS: [&str; 6] = [
    "known_mismatches_pin_the_gpu_the_plain_rust_and_the_cpu_twin_outcome",
    "every_pinned_ledger_id_is_documented_and_every_documented_id_is_pinned",
    "the_emitted_wgsl_uses_plain_operators_and_conversions",
    "the_ledger_mode_is_the_overflow_mode_of_this_crate",
    "the_out_of_bounds_store_twin_stops_at_the_store_after_a_load_that_succeeded",
    "a_kernel_without_storage_parameters_runs_on_the_twin",
];

/// Every cargo call a mode makes after its prerequisite checks, in order, as the arguments
/// after `cargo`. `verify_inner` runs exactly this list, so a test of it is a test of the mode:
/// the Full mode is the release gate and must not lose a step unnoticed.
pub(crate) fn cargo_calls(mode: VerifyMode) -> Vec<Vec<String>> {
    let owned = |args: &[&str]| args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>();
    let mut calls = Vec::new();
    if matches!(
        mode,
        VerifyMode::Smoke | VerifyMode::Fast | VerifyMode::Full
    ) {
        calls.push(owned(&["fmt", "--all", "--", "--check"]));
    }
    if matches!(mode, VerifyMode::Smoke | VerifyMode::Full) {
        calls.push(owned(&[
            "clippy",
            "--locked",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ]));
    }
    match mode {
        VerifyMode::Fast => {
            calls.push(owned(&["test", "--locked", "-p", "gust-macros"]));
            calls.push(owned(&["test", "--locked", "-p", "gust", "--lib"]));
            calls.push(owned(&["test", "--locked", "-p", "gust-wgpu"]));
        }
        VerifyMode::Gpu => {
            calls.push(owned(&GPU_SUITE_CALL));
            calls.push(owned(&ALGO_GPU_CALL));
        }
        VerifyMode::Smoke => calls.push(workspace_test_args()),
        VerifyMode::Full => {
            calls.push(workspace_test_args());
            // The harness crate is a workspace of its own: format, clippy and its tests (m5.c2, fix round 2).
            calls.extend(DIFFTEST_CARGO_CALLS.iter().map(|args| owned(args)));
            calls.push(owned(&RELEASE_SLANG_TESTS));
            calls.push(owned(&RELEASE_LEDGER_TEST));
        }
        VerifyMode::Examples | VerifyMode::Artifacts => {}
    }
    if matches!(mode, VerifyMode::Examples | VerifyMode::Full) {
        for example in EXAMPLES {
            // `--include-ignored`, not `--ignored`: the examples' ordinary tests (the CPU twin against the host
            // reference, the derivation-domain guards) run here too, next to the GPU tests they gate.
            calls.push(owned(&["test", "-p", example, "--", "--include-ignored"]));
        }
    }
    let examples_to_run = match mode {
        VerifyMode::Smoke => SMOKE_EXAMPLES,
        VerifyMode::Examples | VerifyMode::Artifacts | VerifyMode::Full => EXAMPLES,
        VerifyMode::Fast | VerifyMode::Gpu => &[],
    };
    for example in examples_to_run {
        calls.push(owned(&example_run_args(example)));
    }
    calls
}

/// Whether a mode also runs `cargo xtask check-dual` (full mode, GPU): only the release gate does, after its cargo
/// calls. It is not part of [`cargo_calls`] because it is a check of its own, not one cargo call.
pub(crate) fn runs_check_dual(mode: VerifyMode) -> bool {
    mode == VerifyMode::Full
}

/// Whether a cargo call of a mode must run at least one test: the examples' tests (`--include-ignored`, which
/// also must leave nothing ignored), the name-filtered release Slang tests and the release ledger (`0 ignored` and
/// every name of [`RELEASE_LEDGER_TESTS`]). The unfiltered package and the other integration-target calls are not
/// counted here; they have no filter that can match nothing.
fn counts_tests(call: &[String]) -> bool {
    call.iter().any(|arg| arg == "--include-ignored")
        || call.iter().map(String::as_str).eq(RELEASE_SLANG_TESTS)
        || is_release_ledger(call)
}

/// Whether `call` is [`RELEASE_LEDGER_TEST`].
fn is_release_ledger(call: &[String]) -> bool {
    call.iter().map(String::as_str).eq(RELEASE_LEDGER_TEST)
}

/// `cargo xtask check-ledger`: runs exactly [`RELEASE_LEDGER_TEST`] and judges its output with the guard of the Full
/// mode ([`check_counted_call`]), so that the release gate and the nightly list of the `gpu-software` workflow share
/// one guard. Needs `slangc`, the reflection helper (prepared as `verify` does) and a Vulkan adapter, and is
/// evidence only for the adapter the ledger lines name.
pub(crate) fn check_ledger(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("check-ledger takes no arguments".to_owned());
    }
    require_command("slangc")?;
    let root = workspace_root()?;
    // As `verify` does before its first GPU step: the helper named by `GUST_SLANG_REFLECT` must match the Slang on
    // `PATH`, or is built when the variable is unset.
    prepare_helper(&root, None)?;
    let call = owned_args(&RELEASE_LEDGER_TEST);
    let (_, stdout) = run_cargo_test_captured(&root, &call)?;
    check_counted_call(&call, &stdout)?;
    println!(
        "check-ledger passed: the release ledger ran {} named tests, 0 ignored",
        RELEASE_LEDGER_TESTS.len()
    );
    Ok(())
}

fn owned_args(args: &[&str]) -> Vec<String> {
    args.iter().map(|arg| (*arg).to_owned()).collect()
}

/// The cargo call of the Gpu mode: every test of `gust-wgpu`, the unit tests of the library, each integration test
/// file and the doc tests.
pub(crate) const GPU_SUITE_CALL: [&str; 4] = ["test", "--locked", "-p", "gust-wgpu"];

/// The cargo call of the device-resident algorithms (m7.r1): every test of `gust-algo`, the `gpu::` ones on the first
/// adapter. Run after [`GPU_SUITE_CALL`] in the Gpu mode and by the Gpu suite, so the lavapipe job runs it too.
pub(crate) const ALGO_GPU_CALL: [&str; 4] = ["test", "--locked", "-p", "gust-algo"];

/// The `gpu::` tests of `gust-algo`, by name: each must be reported `ok`, and no other `tests::gpu::` test may exist
/// ([`check_algo_gpu_counts`]), so a deleted or an unpinned adapter test fails the gate. The twins would still satisfy
/// a plain "at least one test ran". Adding or removing a `gpu::` test changes this list in the same commit.
pub(crate) const ALGO_GPU_TESTS: [&str; 12] = [
    "tests::gpu::gpu_a_first_pass_past_the_workgroup_limit_is_a_device_limit_error_before_any_dispatch",
    "tests::gpu::gpu_an_input_past_the_binding_limit_is_a_device_limit_error_before_any_dispatch",
    "tests::gpu::gpu_empty_inputs",
    "tests::gpu::gpu_float_min_and_max_are_exact",
    "tests::gpu::gpu_float_sum_of_the_tie_ladder_is_within_the_bound",
    "tests::gpu::gpu_float_sums_are_within_the_per_level_bound",
    "tests::gpu::gpu_integer_sums_min_and_max_are_exact_at_every_boundary",
    "tests::gpu::gpu_integer_sums_past_the_domain_wrap",
    "tests::gpu::gpu_min_of_an_empty_vector_is_empty_and_max_too",
    "tests::gpu::gpu_reductions_leave_the_input_alone_and_return_their_buffers",
    "tests::gpu::gpu_runs_the_deepest_schedule",
    "tests::gpu::gpu_sum_on_device_keeps_the_result_resident",
];

/// The shape of [`GPU_SUITE_CALL`]'s output: `passed` is the sum over every `test result:` line, `result_lines`
/// the number of those lines (one per test binary and doc-test run), which catches drift that cancels out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GpuSuitePin {
    pub(crate) passed: u64,
    pub(crate) result_lines: u64,
}

/// The pinned shape of [`GPU_SUITE_CALL`]'s output (D52, plan M5 R1-6), checked in the Gpu mode on every adapter and by
/// the Gpu suite (`cargo xtask test --suite Gpu`, m5.f1). Every `test result:` line must also read `ok` with
/// `0 failed; 0 ignored`.
///
/// This is the only place the numbers live (the `gpu-software` workflow only calls `verify --mode gpu`). Every
/// slice that adds or removes a Gpu test or test file bumps them in the same commit; drift fails loudly, by
/// design. Measured: 140 passed over 18 lines on the RTX 5090 at the m4.e1 merge (first CI run 123/17); 146 over
/// 20 is the value the workflow pinned at 672c5ca, moved here by slice m5.a2. To recompute them, on any adapter and
/// any OS, at the commit to be merged (a skipped assertion inside a test does not change the count; a failing test
/// does, so fix failures first):
///
/// ```text
/// GUST_PROFILE=dev cargo test --locked -p gust-wgpu --no-fail-fast 2>&1 \
///   | grep '^test result:' | awk '{ sum += $4; n++ } END { print sum, n }'
/// ```
///
/// prints `<passed> <lines>`; the output must also show no line other than `0 failed; 0 ignored`.
pub(crate) const GPU_SUITE_PIN: GpuSuitePin = GpuSuitePin {
    passed: 223,
    result_lines: 26,
};

/// Whether a cargo call of a mode is the one whose output [`GPU_SUITE_PIN`] describes: [`GPU_SUITE_CALL`] in the Gpu
/// mode. The Fast mode makes the same call, but it is not the evidence the pin belongs to; the Gpu suite has its
/// own step ([`crate::suite::Step::GpuSuite`]) that applies the same check.
pub(crate) fn pins_gpu_counts(mode: VerifyMode, call: &[String]) -> bool {
    mode == VerifyMode::Gpu && call.iter().map(String::as_str).eq(GPU_SUITE_CALL)
}

/// Whether a cargo call of a mode is [`ALGO_GPU_CALL`] in the Gpu mode, whose output [`check_algo_gpu_counts`] judges.
pub(crate) fn judges_algo_gpu(mode: VerifyMode, call: &[String]) -> bool {
    mode == VerifyMode::Gpu && call.iter().map(String::as_str).eq(ALGO_GPU_CALL)
}

/// The guards on a finished cargo call, in order: the examples and release Slang tests must run tests
/// ([`check_counted_call`]), the Gpu suite must match [`GPU_SUITE_PIN`] ([`check_gpu_suite_counts`]).
fn guard_call(mode: VerifyMode, call: &[String], stdout: &str) -> Result<(), String> {
    if counts_tests(call) {
        check_counted_call(call, stdout)
    } else if pins_gpu_counts(mode, call) {
        check_gpu_suite_counts(call, stdout, GPU_SUITE_PIN)
    } else if judges_algo_gpu(mode, call) {
        check_algo_gpu_counts(call, stdout)
    } else {
        Ok(())
    }
}

/// The guard for the Gpu suite on its captured `stdout`: at least one `test result:` line; every line readable, `ok`,
/// `0 failed; 0 ignored`; the `passed` sum and the number of lines equal to `pin`. All findings are reported
/// together, and a count mismatch names the number found and the number pinned.
pub(crate) fn check_gpu_suite_counts(
    call: &[String],
    stdout: &str,
    pin: GpuSuitePin,
) -> Result<(), String> {
    let plain = without_colors(stdout);
    let printed = plain
        .lines()
        .filter(|line| line.trim_start().starts_with("test result:"))
        .count() as u64;
    let totals = sum_test_results(&plain);
    let mut problems = Vec::new();
    if printed == 0 {
        problems.push("no `test result:` line in the output".to_owned());
    }
    if totals.summaries != printed {
        problems.push(format!(
            "{} of {printed} `test result:` line(s) could not be read",
            printed - totals.summaries
        ));
    }
    if totals.failed > 0 || totals.ignored > 0 {
        problems.push(format!(
            "the result lines are not all `ok` with `0 failed; 0 ignored` ({} failed, {} ignored)",
            totals.failed, totals.ignored
        ));
    }
    if totals.passed != pin.passed {
        problems.push(format!(
            "{} tests passed, the pinned count is {} (GPU_SUITE_PIN.passed)",
            totals.passed, pin.passed
        ));
    }
    if printed != pin.result_lines {
        problems.push(format!(
            "{printed} test result lines, the pinned number is {} (GPU_SUITE_PIN.result_lines)",
            pin.result_lines
        ));
    }
    if problems.is_empty() {
        println!(
            "{printed} test result lines (pinned {}), {} passed (pinned {})",
            pin.result_lines, totals.passed, pin.passed
        );
        return Ok(());
    }
    Err(format!(
        "cargo {}: {}. If a Gpu test or test file was added or removed on purpose, change GPU_SUITE_PIN in xtask/src/verify.rs in the same commit",
        call.join(" "),
        problems.join("; ")
    ))
}

/// The guard for [`ALGO_GPU_CALL`] on its captured `stdout`: tests ran and none failed or was ignored, every test of
/// [`ALGO_GPU_TESTS`] is reported `ok`, and no `tests::gpu::` test outside the list ran.
pub(crate) fn check_algo_gpu_counts(call: &[String], stdout: &str) -> Result<(), String> {
    let plain = without_colors(stdout);
    let totals = sum_test_results(&plain);
    require_tests_ran(call, &totals)?;
    let mut problems = Vec::new();
    if totals.ignored > 0 {
        problems.push(format!("{} test(s) reported ignored", totals.ignored));
    }
    let missing = tests_not_ok(&plain, &ALGO_GPU_TESTS);
    if !missing.is_empty() {
        problems.push(format!(
            "the adapter test(s) `{}` did not run and pass (renamed, deleted, ignored or failed?)",
            missing.join("`, `")
        ));
    }
    let ran = plain
        .lines()
        .filter(|line| {
            let line = line.trim();
            line.starts_with("test tests::gpu::") && line.ends_with("... ok")
        })
        .count();
    if ran != ALGO_GPU_TESTS.len() {
        problems.push(format!(
            "{ran} adapter test(s) passed, the pinned number is {} (ALGO_GPU_TESTS)",
            ALGO_GPU_TESTS.len()
        ));
    }
    if problems.is_empty() {
        println!(
            "{ran} gust-algo adapter tests passed (pinned {})",
            ALGO_GPU_TESTS.len()
        );
        return Ok(());
    }
    Err(format!(
        "cargo {}: {}. If an adapter test was added or removed on purpose, change ALGO_GPU_TESTS in xtask/src/verify.rs in the same commit",
        call.join(" "),
        problems.join("; ")
    ))
}

/// The release-only tests the release `--lib` step exists for (`cfg(not(debug_assertions))`, so they run nowhere
/// else). A count alone would still pass if one were moved, renamed or dropped from the filters, because dozens of
/// other tests remain; each must be reported `ok`.
pub(crate) const RELEASE_ONLY_SLANG_TEST: &str = "slang::source_map::tests::unparsable_match_marker_number_falls_back_to_the_map_ordinal_in_release";
pub(crate) const RELEASE_ONLY_CPU_TEST: &str =
    "cpu::tests::invocations_are_capped_at_the_id_range_in_release";
pub(crate) const RELEASE_ONLY_TESTS: [&str; 2] = [RELEASE_ONLY_SLANG_TEST, RELEASE_ONLY_CPU_TEST];

/// The guard for a call of [`counts_tests`] on its captured `stdout`: tests ran, and the release call also
/// reported every one of [`RELEASE_ONLY_TESTS`] as `ok`; the ledger call reported `0 ignored` and every one of
/// [`RELEASE_LEDGER_TESTS`] as `ok`.
fn check_counted_call(call: &[String], stdout: &str) -> Result<(), String> {
    let totals = sum_test_results(stdout);
    require_tests_ran(call, &totals)?;
    if is_release_ledger(call) {
        if totals.ignored > 0 {
            return Err(format!(
                "cargo {}: {} test(s) reported ignored; the release ledger runs every test of its file",
                call.join(" "),
                totals.ignored
            ));
        }
        let missing = tests_not_ok(stdout, &RELEASE_LEDGER_TESTS);
        if !missing.is_empty() {
            return Err(format!(
                "cargo {}: the ledger test(s) `{}` did not run and pass (ignored, renamed, removed, or filtered out? \
                 RELEASE_LEDGER_TESTS in xtask/src/verify.rs names them)",
                call.join(" "),
                missing.join("`, `")
            ));
        }
    }
    if call.iter().map(String::as_str).eq(RELEASE_SLANG_TESTS) {
        let missing = tests_not_ok(stdout, &RELEASE_ONLY_TESTS);
        if !missing.is_empty() {
            return Err(format!(
                "cargo {}: the release-only test(s) `{}` did not run and pass (moved, renamed, ignored, or left out of the filters?)",
                call.join(" "),
                missing.join("`, `")
            ));
        }
    }
    Ok(())
}

/// The cargo call that runs one example binary and exports its SPIR-V (the Spirv suite).
pub(crate) fn example_run_args(example: &str) -> [&str; 5] {
    ["run", "--locked", "--quiet", "-p", example]
}

/// Compares the exported SPIR-V file names with the expected kernels and names the
/// differences separately: a renamed kernel is one missing and one unexpected file.
fn compare_artifact_names(expected: &[&str], found: &[String]) -> Result<(), String> {
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
        "Exported SPIR-V kernels differ from the {} expected ({} found).",
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
        " Update EXPECTED_SPIRV in xtask/src/workspace.rs deliberately for new examples.",
    );
    Err(message)
}

#[cfg(test)]
#[path = "tests/artifact_names.rs"]
mod artifact_names_tests;

#[cfg(test)]
#[path = "tests/gpu_counts.rs"]
mod gpu_counts_tests;

#[cfg(test)]
#[path = "tests/verify_integration.rs"]
mod verify_integration_tests;

#[cfg(test)]
#[path = "tests/verify_generated.rs"]
mod verify_generated_tests;

#[cfg(test)]
#[path = "tests/verify_steps.rs"]
mod verify_steps_tests;
