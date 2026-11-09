//! The commands: `init`, `build`, `check`, `run`, `test` and `doctor`, and the pure core of `build`.

use std::{
    ffi::OsString,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

use gust::Format;

use crate::{
    check::{self, Report},
    cli::{BuildOptions, CargoOptions, CheckOptions, SlangCommand, Target},
    collect::{self, CollectPlan, Package, clean_command, collect, fresh_nonce, metadata_command},
    doctor::{self, Facts},
    export::{self, Compiler, HelperCompiler, formats_with_wgsl, verify_stored},
    helper,
    manifest::{MANIFEST_FILE, Manifest, format_name},
    record::Record,
    routes::{self, RoutePlan},
    slang::{self, Env, Located, SLANG_VERSION, Source},
    store::{self, Commit, Event, read_dir_state},
    toolchain, tools,
};

/// The name of the artifact directory in the package directory.
pub(crate) const ARTIFACTS_DIR: &str = "gust-artifacts";

/// What a command resolved about its package.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Context {
    pub(crate) cargo: OsString,
    pub(crate) package: Package,
    pub(crate) manifest_path: Option<PathBuf>,
    pub(crate) target_directory: PathBuf,
    pub(crate) artifacts: PathBuf,
}

/// Run a command with inherited output and say whether it succeeded.
fn run_status(command: &mut Command) -> Result<bool, String> {
    command
        .status()
        .map(|status| status.success())
        .map_err(|error| {
            format!(
                "cannot run `{}`: {error}",
                command.get_program().to_string_lossy()
            )
        })
}

/// The cargo to run and what `cargo metadata --no-deps` says about the workspace of `target`.
fn read_workspace(target: &Target) -> Result<(OsString, collect::Workspace), String> {
    let cargo = collect::cargo_program(std::env::var_os("CARGO"));
    let output = metadata_command(&cargo, target.manifest_path.as_deref())
        .output()
        .map_err(|error| format!("cannot run `cargo metadata`: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "`cargo metadata` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let workspace = collect::parse_metadata(&String::from_utf8_lossy(&output.stdout))?;
    Ok((cargo, workspace))
}

fn resolve(target: &Target) -> Result<Context, String> {
    let (cargo, workspace) = read_workspace(target)?;
    select_context(cargo, workspace, target)
}

/// The manifest cargo works on: `--manifest-path`, else the closest `Cargo.toml` from the current directory.
fn manifest_of(target: &Target) -> Option<PathBuf> {
    match &target.manifest_path {
        Some(path) => Some(fs::canonicalize(path).unwrap_or_else(|_| path.clone())),
        None => std::env::current_dir()
            .ok()
            .and_then(|cwd| collect::find_manifest(&cwd, &|path| path.is_file())),
    }
}

/// The package of `target` in `workspace`, with what a command needs to know about it.
fn select_context(
    cargo: OsString,
    workspace: collect::Workspace,
    target: &Target,
) -> Result<Context, String> {
    let manifest = manifest_of(target);
    let package =
        collect::select_package(&workspace, target.package.as_deref(), manifest.as_deref())?
            .clone();
    let artifacts = package.directory().join(ARTIFACTS_DIR);
    Ok(Context {
        cargo,
        package,
        manifest_path: target.manifest_path.clone(),
        target_directory: workspace.target_directory,
        artifacts,
    })
}

fn collect_plan(ctx: &Context) -> CollectPlan {
    CollectPlan {
        cargo: ctx.cargo.clone(),
        package: ctx.package.name.clone(),
        manifest_path: ctx.manifest_path.clone(),
        target_directory: ctx.target_directory.clone(),
        nonce: fresh_nonce(),
    }
}

/// The records of every body of the package, from one `cargo check` through `run` (M8, D139). `build` and `check` both
/// collect through here and nowhere else: the macro's export mode writes a record for every body whatever the profile,
/// so there is no profile loop and a command never runs cargo a second time to collect.
fn collect_records(
    ctx: &Context,
    run: &dyn Fn(&mut Command) -> Result<bool, String>,
) -> Result<Vec<Record>, String> {
    collect(&collect_plan(ctx), run)
}

/// The routing of the package's `gust.toml` for the collect step (`dev`, the profile it runs under), or `None` when no
/// `gust.toml` is beside the manifest (M8, m8.r2). `build` and `check` have no cargo arguments to forward.
fn collect_routes(ctx: &Context) -> Result<Option<RoutePlan>, String> {
    routes::plan(&ctx.cargo, &ctx.package, collect::COLLECT_PROFILE, &[])
}

/// Sets the route variables of `routes` on `command`, if there are any. `build` and `check` use it through
/// [`routed_runner`]; `run` and `test` build their command with `routes::cargo_command`, which calls
/// `RoutePlan::apply` for the same variables.
fn apply_routes(routes: Option<&RoutePlan>, command: &mut Command) {
    if let Some(plan) = routes {
        plan.apply(command);
    } else {
        routes::without_routed_profile(command);
    }
}

/// The runner of the collect step of `build` and `check`: `run` after the route variables are set on the command.
fn routed_runner<'a>(
    routes: Option<&'a RoutePlan>,
    run: &'a dyn Fn(&mut Command) -> Result<bool, String>,
) -> impl Fn(&mut Command) -> Result<bool, String> + 'a {
    move |command| {
        apply_routes(routes, command);
        run(command)
    }
}

/// What the driver says about a plan, on the error stream (`run` hands its standard output to the program): one line
/// per tag, then a note for each preset `GUST_ROUTE__*` variable that differs from the plan or matches no tag.
fn report_routes(plan: &RoutePlan) {
    for line in plan.summary() {
        eprintln!("{line}");
    }
    let preset: Vec<(String, String)> = std::env::vars_os()
        .map(|(name, value)| {
            (
                name.to_string_lossy().into_owned(),
                value.to_string_lossy().into_owned(),
            )
        })
        .collect();
    for line in plan.notes(&preset) {
        eprintln!("{line}");
    }
}

/// Writes the table and reports the plan, or, when no `gust.toml` applies, removes the table an earlier run left.
fn settle_routes(ctx: &Context, plan: Option<&RoutePlan>) -> Result<(), String> {
    match plan {
        Some(plan) => {
            plan.write()?;
            report_routes(plan);
            Ok(())
        }
        None => routes::forget(&ctx.package),
    }
}

/// Runs `cargo <verb> <arguments>` as given, with the routes of `plan`, and returns cargo's exit code.
fn spawn_cargo(
    cargo: &OsString,
    verb: &str,
    arguments: &[String],
    plan: Option<&RoutePlan>,
) -> Result<i32, String> {
    let status = routes::cargo_command(cargo, verb, arguments, plan)
        .status()
        .map_err(|error| format!("cannot run `cargo {verb}`: {error}"))?;
    Ok(status.code().unwrap_or(2))
}

/// `cargo gust run` and `cargo gust test`: cargo with the route variables of the package's `gust.toml` (m8.r2). The
/// exit status is cargo's.
pub(crate) fn cargo_with_routes(verb: &str, options: &CargoOptions) -> Result<i32, String> {
    let (cargo, workspace) = read_workspace(&options.target)?;
    if options.workspace {
        // `--workspace` and `--all` build every member, which the driver cannot route one by one: refused when a
        // selected member has a `gust.toml`, else cargo runs as asked (also from a virtual root, where no package is
        // selected and none applies).
        routes::refuse_workspace(&workspace.packages, &routes::excluded(&options.arguments))?;
        return spawn_cargo(&cargo, verb, &options.arguments, None);
    }
    // No `-p`, at a workspace root whose default members are several packages (a virtual root, or `default-members`):
    // cargo builds all of them, so the same member check applies.
    if options.target.package.is_none()
        && let Some(selected) = manifest_of(&options.target)
            .and_then(|manifest| routes::default_selection(&workspace, &manifest))
    {
        routes::refuse_workspace(selected, &[])?;
        return spawn_cargo(&cargo, verb, &options.arguments, None);
    }
    let ctx = select_context(cargo, workspace, &options.target)?;
    let gust_profile =
        std::env::var_os("GUST_PROFILE").map(|value| value.to_string_lossy().into_owned());
    let profile = routes::build_profile(gust_profile.as_deref(), &options.arguments)?;
    let plan = routes::plan(&ctx.cargo, &ctx.package, profile, &options.arguments)?;
    settle_routes(&ctx, plan.as_ref())?;
    spawn_cargo(&ctx.cargo, verb, &options.arguments, plan.as_ref())
}

/// The helper executable, or the error that says how to get one. It does not look at Slang itself: `check` without
/// `--recompile` only needs the helper's build tag.
fn require_helper() -> Result<PathBuf, String> {
    let env = Env::from_process();
    let candidate = toolchain::helper_candidate();
    tools::locate_executable(&candidate, env.path.as_deref()).ok_or_else(|| {
        format!(
            "the native Slang helper was not found (looked for {}); {}",
            candidate.display(),
            tools::helper_hint()
        )
    })
}

/// The one-time `cargo clean -p` of the init transition (R2-3).
fn clean_once(ctx: &Context) -> Result<(), String> {
    println!(
        "gust: running `cargo clean -p {}` once: the crate was built before it had a manifest, and under GUST_PROFILE=dev cargo would not notice that artifacts now exist. Other target directories (rust-analyzer's, per-lane ones, a CARGO_TARGET_DIR set elsewhere) stay stale until their next re-expansion of the crate.",
        ctx.package.name
    );
    let mut command = clean_command(&ctx.cargo, &ctx.package.name, ctx.manifest_path.as_deref());
    let name = &ctx.package.name;
    let advice =
        format!("run `cargo clean -p {name}` yourself; until then `dev` builds can be stale");
    match run_status(&mut command) {
        Ok(true) => Ok(()),
        Ok(false) => Err(format!("`cargo clean -p {name}` failed: {advice}")),
        Err(error) => Err(format!("{error}: {advice}")),
    }
}

pub(crate) fn init(target: &Target) -> Result<i32, String> {
    let ctx = resolve(target)?;
    if ctx.artifacts.join(MANIFEST_FILE).is_file() {
        println!(
            "{ARTIFACTS_DIR}/{MANIFEST_FILE} already exists for `{}`; nothing to do",
            ctx.package.name
        );
        return Ok(0);
    }
    // The clean comes first: when it fails there is still no manifest, so running `init` again retries it.
    clean_once(&ctx)?;
    if !store::create_manifest(&ctx.artifacts, &Manifest::empty().render())? {
        println!(
            "{ARTIFACTS_DIR}/{MANIFEST_FILE} already exists for `{}`; nothing to do",
            ctx.package.name
        );
        return Ok(0);
    }
    println!(
        "created {ARTIFACTS_DIR}/{MANIFEST_FILE} for `{}`",
        ctx.package.name
    );
    println!(
        "hint: add `**/gust-artifacts/** -text` to .gitattributes so that git never rewrites the committed artifacts (their hashes are byte-exact)"
    );
    println!("next: run `cargo gust build` to write the artifacts");
    Ok(0)
}

/// What a build writes: WGSL, the requested formats, and every format the stored manifest already lists, so that a
/// plain `build` never drops committed SPIR-V.
pub(crate) fn formats_for_build(requested: &[Format], state: &store::DirState) -> Vec<Format> {
    let mut formats = requested.to_vec();
    formats.extend(stored_formats(state));
    formats_with_wgsl(&formats)
}

pub(crate) fn build(options: &BuildOptions) -> Result<i32, String> {
    let ctx = resolve(&options.target)?;
    // The routing first: an error in `gust.toml` needs no Slang and no build. The table is written below, once the
    // toolchain is known to be there.
    let routes = collect_routes(&ctx)?;
    let state = read_dir_state(&ctx.artifacts)?;
    let requested = formats_with_wgsl(&options.formats);
    let formats = formats_for_build(&options.formats, &state);
    for format in formats.iter().filter(|format| !requested.contains(format)) {
        println!(
            "gust: also building {}: the manifest already lists it",
            format_name(*format)
        );
    }
    // The pinned slangc and a helper of the same Slang build, or no build at all (R4.8).
    let toolchain = toolchain::require(&Env::from_process(), options.allow_slang_mismatch)?;
    if let Some(note) = slang::version_note(&toolchain.slangc) {
        println!("gust: {note}");
    }
    settle_routes(&ctx, routes.as_ref())?;
    let helper = toolchain.helper;
    let spirv_val = if formats.contains(&Format::Spirv) {
        let found = tools::find_spirv_val(std::env::var_os("PATH").as_deref());
        if found.is_none() {
            eprintln!(
                "gust: warning: spirv-val is not on PATH; SPIR-V is written with `validated: false`"
            );
        }
        found
    } else {
        None
    };
    let manifest_missing_before = !ctx.artifacts.join(MANIFEST_FILE).is_file();
    println!("gust: collecting the kernels of `{}`", ctx.package.name);
    let records = collect_records(&ctx, &routed_runner(routes.as_ref(), &run_status))?;
    println!("gust: compiling {} kernel(s)", records.len());
    let compiler = HelperCompiler { helper, spirv_val };
    // Everything is compiled and verified before the clean runs, and the clean comes before the manifest is created,
    // so when it fails the next build (which still sees no manifest) retries it.
    let summary = finish_build(
        &ctx.artifacts,
        records,
        &formats,
        &compiler,
        &mut || {
            if manifest_missing_before {
                clean_once(&ctx)
            } else {
                Ok(())
            }
        },
        &mut |_| Ok(()),
    )?;
    for line in summary.lines() {
        println!("{line}");
    }
    Ok(0)
}

pub(crate) fn check(options: &CheckOptions) -> Result<i32, String> {
    let ctx = resolve(&options.target)?;
    // `check` reads: it resolves the routing (so a bad `gust.toml` fails here too) and sets the variables, but it does
    // not write `routes.json`.
    let routes = collect_routes(&ctx)?;
    let records = export::dedupe(collect_records(
        &ctx,
        &routed_runner(routes.as_ref(), &run_status),
    )?)?;
    let state = read_dir_state(&ctx.artifacts)?;
    let helper = if options.recompile {
        // Recompiling uses the compiler, so it needs the pinned Slang and a helper of the same build (R4.8).
        let toolchain = toolchain::require(&Env::from_process(), options.allow_slang_mismatch)?;
        if let Some(note) = slang::version_note(&toolchain.slangc) {
            println!("gust: {note}");
        }
        Some(Ok(toolchain.helper))
    } else if !options.allow_slang_mismatch {
        Some(require_helper())
    } else {
        None
    };
    let installed_slang = if options.allow_slang_mismatch {
        None
    } else {
        Some(match &helper {
            Some(Ok(path)) => tools::helper_slang_version(path),
            Some(Err(error)) => Err(error.clone()),
            None => Err("no helper lookup".to_owned()),
        })
    };
    let recompiled = if options.recompile {
        let helper = match helper {
            Some(Ok(path)) => path,
            Some(Err(error)) => return Err(error),
            None => return Err("no helper lookup".to_owned()),
        };
        let formats = stored_formats(&state);
        let spirv_val = if formats.contains(&Format::Spirv) {
            tools::find_spirv_val(std::env::var_os("PATH").as_deref())
        } else {
            None
        };
        println!("gust: recompiling {} kernel(s)", records.len());
        Some(export::plan(
            records.clone(),
            &formats,
            &HelperCompiler { helper, spirv_val },
        )?)
    } else {
        None
    };
    let mut report = check::run(&check::Inputs {
        records: &records,
        state: &state,
        installed_slang,
        verify: &verify_stored,
        recompiled: recompiled.as_ref(),
    });
    add_variant_problems(&mut report, &records, &state);
    for line in render_report(&report, records.len()) {
        println!("{line}");
    }
    Ok(i32::from(!report.problems.is_empty()))
}

/// Add what the manifest's `variants` lists get wrong against the collected records (M8, D139) to the report. A
/// manifest that cannot be read is already a problem of `check::run`, so it adds nothing here.
pub(crate) fn add_variant_problems(
    report: &mut Report,
    records: &[Record],
    state: &store::DirState,
) {
    let manifest = state
        .file(MANIFEST_FILE)
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .and_then(|text| Manifest::parse(text).ok());
    if let Some(manifest) = manifest {
        report
            .problems
            .extend(export::variant_problems(records, &manifest));
    }
}

/// The formats the stored manifest lists, WGSL when it lists none (or cannot be read).
fn stored_formats(state: &store::DirState) -> Vec<Format> {
    let listed = state
        .file(MANIFEST_FILE)
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .and_then(|text| Manifest::parse(text).ok())
        .map(|manifest| {
            manifest
                .kernels
                .iter()
                .flat_map(|kernel| kernel.formats.iter().map(|format| format.format))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    formats_with_wgsl(&listed)
}

/// The lines `check` prints: notes first, then one line per problem, then the verdict.
pub(crate) fn render_report(report: &Report, kernels: usize) -> Vec<String> {
    let mut lines: Vec<String> = report
        .notes
        .iter()
        .map(|note| format!("note: {note}"))
        .collect();
    lines.extend(
        report
            .problems
            .iter()
            .map(|problem| format!("problem: {problem}")),
    );
    lines.push(if report.problems.is_empty() {
        format!("ok: {ARTIFACTS_DIR}/ matches the {kernels} collected kernel(s)")
    } else {
        format!(
            "FAILED: {} problem(s) in {ARTIFACTS_DIR}/",
            report.problems.len()
        )
    });
    lines
}

pub(crate) fn doctor() -> Result<i32, String> {
    let env = Env::from_process();
    let slangc = slang::locate(&env, false, &slang::slangc_version);
    let candidate = toolchain::helper_candidate();
    let path_var = env.path.clone();
    let helper_found = tools::locate_executable(&candidate, path_var.as_deref());
    let helper_slang = helper_found.as_deref().map(tools::helper_slang_version);
    let path_slangc = tools::find_on_path(&tools::executable_name("slangc"), path_var.as_deref())
        .map(|path| {
            let version = slang::slangc_version(&path);
            (path, version)
        });
    let notices = slangc
        .as_ref()
        .ok()
        .and_then(Located::install_dir)
        .map(|dir| slang::notice_files(&dir))
        .unwrap_or_default();
    let facts = Facts {
        version: env!("CARGO_PKG_VERSION").to_owned(),
        pin: SLANG_VERSION.to_owned(),
        slangc,
        notices,
        path_slangc,
        helper_candidate: candidate,
        helper_found,
        helper_slang,
        spirv_val: tools::find_spirv_val(path_var.as_deref()),
        gust_profile: std::env::var_os("GUST_PROFILE")
            .map(|value| value.to_string_lossy().into_owned()),
    };
    let (lines, healthy) = doctor::lines(&facts);
    for line in lines {
        println!("{line}");
    }
    Ok(i32::from(!healthy))
}

/// `cargo gust slang <command>`.
pub(crate) fn slang(command: &SlangCommand) -> Result<i32, String> {
    let env = Env::from_process();
    match command {
        SlangCommand::Install => slang_install(&env),
        SlangCommand::Locate { allow_mismatch } => {
            let located = slang::locate(&env, *allow_mismatch, &slang::slangc_version)?;
            println!("slangc: {}", located.slangc.display());
            println!("version: {}", located.version);
            println!("source: {}", located.source.label());
            if let Some(note) = slang::version_note(&located) {
                println!("{note}");
            }
            Ok(0)
        }
        SlangCommand::BuildHelper {
            force,
            sdk_root,
            out_dir,
        } => slang_build_helper(&env, *force, sdk_root.as_deref(), out_dir.as_deref()),
    }
}

fn slang_install(env: &Env) -> Result<i32, String> {
    let asset = slang::asset_for(SLANG_VERSION, std::env::consts::OS, std::env::consts::ARCH)
        .ok_or_else(|| {
            format!(
                "no Slang {SLANG_VERSION} release archive is pinned for {}-{}; install Slang yourself and put its bin directory first on PATH (or pass --allow-slang-mismatch to commands that accept it)",
                std::env::consts::OS,
                std::env::consts::ARCH
            )
        })?;
    let root = slang::install_root(env)?;
    match slang::install(&root, asset, &slang::SystemTools, std::process::id())? {
        slang::Installed::Already(dir) => {
            println!(
                "slang: {SLANG_VERSION} is already installed at {}",
                dir.display()
            );
        }
        slang::Installed::Fresh(dir) => {
            println!("slang: installed {SLANG_VERSION} at {}", dir.display());
        }
    }
    let dir = root.join(SLANG_VERSION);
    for notice in slang::notice_files(&dir) {
        println!("slang: licence notice kept at {}", notice.display());
    }
    println!("next: run `cargo gust slang build-helper` to build the native helper");
    Ok(0)
}

fn slang_build_helper(
    env: &Env,
    force: bool,
    sdk_root: Option<&Path>,
    out_dir: Option<&Path>,
) -> Result<i32, String> {
    let (sdk_root, out_dir) = match (sdk_root, out_dir) {
        (Some(sdk), Some(out)) => (sdk.to_path_buf(), out.to_path_buf()),
        (Some(_), None) => {
            return Err("with --sdk-root, also pass --out-dir (the managed helper directory belongs to the pinned install)".to_owned());
        }
        (None, out) => {
            let located = slang::locate(env, false, &slang::slangc_version)?;
            let Some(root) = located
                .root
                .as_deref()
                .filter(|_| located.source != Source::Path)
            else {
                return Err(format!(
                    "the pinned Slang found at {} is not a cargo gust install, so there is no SDK to build against: pass --sdk-root <dir> and --out-dir <dir>, or run `cargo gust slang install`",
                    located.slangc.display()
                ));
            };
            let sdk = root.join(SLANG_VERSION);
            let out = out.map_or_else(|| slang::helper_dir(root, SLANG_VERSION), Path::to_path_buf);
            (sdk, out)
        }
    };
    let (exe, tag) = helper::build(&helper::BuildOptions {
        force,
        sdk_root: Some(sdk_root),
        out_dir: Some(out_dir),
    })?;
    println!("slang: Slang build tag of the helper: {tag}");
    println!("helper: {}", exe.display());
    Ok(0)
}

/// What a build did, for printing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BuildSummary {
    pub(crate) kernels: usize,
    pub(crate) commit: Commit,
    /// `key (module::name)` of kernels that were in the old manifest and are gone.
    pub(crate) stale: Vec<String>,
    pub(crate) warnings: Vec<String>,
    /// One note per kernel with `export_unchecked` parameters.
    pub(crate) notes: Vec<String>,
    pub(crate) slang: Option<String>,
    pub(crate) formats: Vec<Format>,
}

impl BuildSummary {
    pub(crate) fn lines(&self) -> Vec<String> {
        let formats: Vec<&str> = self
            .formats
            .iter()
            .map(|format| format_name(*format))
            .collect();
        let mut lines = vec![format!(
            "gust: {} kernel(s) ({}), Slang {}: {} file(s) written, {} unchanged",
            self.kernels,
            formats.join(", "),
            self.slang.as_deref().unwrap_or("-"),
            self.commit.written.len(),
            self.commit.unchanged
        )];
        lines.extend(
            self.warnings
                .iter()
                .map(|warning| format!("gust: warning: {warning}")),
        );
        lines.extend(self.notes.iter().map(|note| format!("gust: note: {note}")));
        lines.extend(
            self.stale
                .iter()
                .map(|stale| format!("gust: stale kernel {stale} removed")),
        );
        lines.extend(
            self.commit
                .pruned
                .iter()
                .map(|name| format!("gust: pruned {name}")),
        );
        lines.extend(self.commit.left_alone.iter().map(|name| {
            format!("gust: left the directory {name}/ alone (cargo gust check reports it)")
        }));
        lines
    }
}

/// Plan the export of `records` and write it into `artifacts`: compile everything first, then stage, move, write the
/// manifest last and prune. Nothing is written when any kernel fails. `before_commit` runs once everything
/// compiled and verified, right before the first write; when it fails nothing is written.
/// (The first build uses it for the one-time clean, which must precede the manifest.)
pub(crate) fn finish_build(
    artifacts: &Path,
    records: Vec<Record>,
    formats: &[Format],
    compiler: &dyn Compiler,
    before_commit: &mut dyn FnMut() -> Result<(), String>,
    hook: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
) -> Result<BuildSummary, String> {
    let old = fs::read_to_string(artifacts.join(MANIFEST_FILE))
        .ok()
        .and_then(|text| Manifest::parse(&text).ok());
    let planned = export::plan(records, formats, compiler)?;
    before_commit()?;
    let commit = store::commit(artifacts, &planned.files, &planned.manifest_file(), hook)?;
    let stale = old
        .iter()
        .flat_map(|manifest| manifest.kernels.iter())
        .filter(|kernel| planned.manifest.entry(&kernel.key).is_none())
        .map(|kernel| format!("{} ({})", kernel.key, kernel.qualified_name()))
        .collect();
    let notes = planned
        .manifest
        .kernels
        .iter()
        .filter(|kernel| !kernel.export_unchecked.is_empty())
        .map(|kernel| check::unchecked_note(&kernel.qualified_name(), &kernel.export_unchecked))
        .collect();
    Ok(BuildSummary {
        kernels: planned.manifest.kernels.len(),
        commit,
        stale,
        warnings: planned.warnings,
        notes,
        slang: planned.manifest.slang,
        formats: formats.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        export::tests::{FakeCompiler, other, sample},
        store::tests::Scratch,
        testdata::SAMPLE_KEY,
    };

    #[test]
    fn the_runner_of_the_collect_step_sets_the_routes_before_it_runs_the_command() {
        let row = |tag: &str, target: &'static str| routes::Row {
            package: "kernel-lib".to_owned(),
            tag: tag.to_owned(),
            variable: format!("GUST_ROUTE__kernel_lib__{tag}"),
            target,
            level: routes::Level::Entry,
            from: None,
        };
        let plan = RoutePlan::new(
            "dev",
            vec![row("physics", "cpu"), row("fusion", "gpu")],
            PathBuf::from("/w/app/gust-artifacts"),
        );
        let seen = std::cell::RefCell::new(Vec::new());
        let run = |command: &mut Command| -> Result<bool, String> {
            let envs: Vec<(String, String)> = command
                .get_envs()
                .map(|(name, value)| {
                    (
                        name.to_string_lossy().into_owned(),
                        value
                            .map(|v| v.to_string_lossy().into_owned())
                            .unwrap_or_default(),
                    )
                })
                .collect();
            seen.borrow_mut().push(envs);
            Ok(true)
        };
        // With a plan the variables are on the command when the inner runner sees it.
        let routed = routed_runner(Some(&plan), &run);
        assert_eq!(routed(&mut Command::new("cargo")), Ok(true));
        let mut first = seen.borrow()[0].clone();
        first.sort();
        assert_eq!(
            first,
            [
                // The profile the plan was resolved for (m8.z), which the macro's guard checks.
                ("GUST_ROUTED_PROFILE".to_owned(), "dev".to_owned()),
                (
                    "GUST_ROUTE__kernel_lib__fusion".to_owned(),
                    "gpu".to_owned()
                ),
                (
                    "GUST_ROUTE__kernel_lib__physics".to_owned(),
                    "cpu".to_owned()
                ),
            ]
        );
        // Without one the routes are left alone, a stale `GUST_ROUTED_PROFILE` is removed (an entry without a value) and the
        // inner result comes back.
        let plain = routed_runner(None, &run);
        assert_eq!(plain(&mut Command::new("cargo")), Ok(true));
        assert_eq!(
            seen.borrow()[1],
            [("GUST_ROUTED_PROFILE".to_owned(), String::new())]
        );
        let failing = |_: &mut Command| -> Result<bool, String> { Err("boom".to_owned()) };
        assert_eq!(
            routed_runner(Some(&plan), &failing)(&mut Command::new("cargo")),
            Err("boom".to_owned())
        );
    }

    fn build_into(
        dir: &Path,
        records: Vec<Record>,
        compiler: &dyn Compiler,
    ) -> Result<BuildSummary, String> {
        finish_build(
            dir,
            records,
            &[Format::Wgsl],
            compiler,
            &mut || Ok(()),
            &mut |_| Ok(()),
        )
    }

    /// The names and bytes of the files in `dir`.
    fn snapshot(dir: &Path) -> Vec<(String, Vec<u8>)> {
        read_dir_state(dir).unwrap().files
    }

    #[test]
    fn a_first_build_writes_the_set_and_a_second_unchanged_build_writes_nothing() {
        let scratch = Scratch::new("build-twice");
        let dir = scratch.0.join(ARTIFACTS_DIR);
        let first = build_into(&dir, vec![sample(), other("b")], &FakeCompiler::new()).unwrap();
        assert_eq!(first.kernels, 2);
        assert_eq!(
            first.commit.written.len(),
            5,
            "4 artifact files and the manifest"
        );
        assert_eq!(
            first.commit.written.last().map(String::as_str),
            Some("manifest.json")
        );
        assert!(first.stale.is_empty() && first.commit.pruned.is_empty());
        let before = snapshot(&dir);
        // S3 at unit level: the same inputs give byte-identical output, no write, no prune message.
        let second = build_into(&dir, vec![other("b"), sample()], &FakeCompiler::new()).unwrap();
        assert!(
            second.commit.written.is_empty(),
            "{:?}",
            second.commit.written
        );
        assert!(second.commit.pruned.is_empty() && second.stale.is_empty());
        assert_eq!(second.commit.unchanged, 5);
        assert_eq!(snapshot(&dir), before);
        assert_eq!(
            second.lines(),
            ["gust: 2 kernel(s) (wgsl), Slang fake-slang-1: 0 file(s) written, 5 unchanged"]
        );
    }

    #[test]
    fn a_crate_with_variants_builds_in_one_pass_and_a_second_unchanged_build_writes_nothing() {
        use crate::testdata::variant::{MANIFEST, ReplayCompiler, records};
        let scratch = Scratch::new("variants-twice");
        let dir = scratch.0.join(ARTIFACTS_DIR);
        // One call: the records of all four bodies, compiled together (that collection is one cargo run is pinned by
        // `build_and_check_collect_through_one_cargo_run`).
        let first = build_into(&dir, records(), &ReplayCompiler).unwrap();
        assert_eq!(first.kernels, 4);
        assert_eq!(
            first.commit.written.len(),
            9,
            "8 artifact files and the manifest"
        );
        assert_eq!(
            fs::read_to_string(dir.join(MANIFEST_FILE)).unwrap(),
            MANIFEST
        );
        let before = snapshot(&dir);
        let mut again = records();
        again.reverse();
        let second = build_into(&dir, again, &ReplayCompiler).unwrap();
        assert!(
            second.commit.written.is_empty(),
            "{:?}",
            second.commit.written
        );
        assert!(second.commit.pruned.is_empty() && second.stale.is_empty());
        assert_eq!(second.commit.unchanged, 9);
        assert_eq!(snapshot(&dir), before);
    }

    /// A context for a package `probe` under a scratch directory (nothing is run against it).
    fn context_in(scratch: &Scratch) -> Context {
        Context {
            cargo: OsString::from("cargo"),
            package: Package {
                name: "probe".to_owned(),
                manifest_path: scratch.0.join("Cargo.toml"),
            },
            manifest_path: None,
            target_directory: scratch.0.join("target"),
            artifacts: scratch.0.join(ARTIFACTS_DIR),
        }
    }

    #[test]
    fn build_and_check_collect_through_one_cargo_run() {
        use crate::testdata::variant::BODIES;
        use std::cell::Cell;
        let scratch = Scratch::new("one-run");
        let ctx = context_in(&scratch);
        // This pins one cargo run per `collect_records(&ctx, run)` call, which is what `build` and `check` each make
        // (`check` then dedupes). It calls `collect_records` itself, once per command name, so it does not count the
        // calls that `build` or `check` make: that each of them calls it once is a reading of the code. The counting
        // runner stands for cargo, writing the record of every body as the macro's export mode does under the one
        // `GUST_PROFILE=dev` run. A second cargo run inside `collect_records` (a profile loop) fails this.
        for command_name in ["build", "check"] {
            let runs = Cell::new(0);
            let run = |command: &mut Command| -> Result<bool, String> {
                runs.set(runs.get() + 1);
                let root = command
                    .get_envs()
                    .find(|(key, _)| key.to_string_lossy() == "GUST_EXPORT_DIR")
                    .and_then(|(_, value)| value)
                    .map(PathBuf::from)
                    .expect("the collect command sets GUST_EXPORT_DIR");
                let dir = root.join("probe");
                fs::create_dir_all(&dir).unwrap();
                for body in BODIES {
                    fs::write(
                        dir.join(format!("{}.kernel.json", body.key)),
                        body.record_json,
                    )
                    .unwrap();
                    fs::write(dir.join(format!("{}.slang", body.key)), body.slang).unwrap();
                }
                Ok(true)
            };
            let collected = collect_records(&ctx, &run).unwrap();
            let collected = if command_name == "check" {
                export::dedupe(collected).unwrap()
            } else {
                collected
            };
            assert_eq!(runs.get(), 1, "`{command_name}` ran cargo more than once");
            assert_eq!(collected.len(), 4, "{command_name}");
        }
    }

    #[test]
    fn check_finds_a_stale_variants_list_and_nothing_in_a_fresh_one() {
        use crate::testdata::variant::{ReplayCompiler, records};
        let scratch = Scratch::new("variants-check");
        let dir = scratch.0.join(ARTIFACTS_DIR);
        build_into(&dir, records(), &ReplayCompiler).unwrap();
        let mut report = Report::default();
        add_variant_problems(&mut report, &records(), &read_dir_state(&dir).unwrap());
        assert_eq!(report, Report::default());
        // The manifest of an older build, from before the variants (only the kernels' own bodies, no list anywhere), is
        // stale for the same sources. (Entries of one kernel with no list at all are not a manifest; the reader refuses.)
        let text = fs::read_to_string(dir.join(MANIFEST_FILE)).unwrap();
        let mut manifest = Manifest::parse(&text).unwrap();
        manifest.kernels.retain(|entry| !entry.variants.is_empty());
        for entry in &mut manifest.kernels {
            entry.variants.clear();
        }
        fs::write(dir.join(MANIFEST_FILE), manifest.render()).unwrap();
        let mut report = Report::default();
        add_variant_problems(&mut report, &records(), &read_dir_state(&dir).unwrap());
        assert_eq!(report.problems.len(), 2, "{:?}", report.problems);
        assert!(
            report
                .problems
                .iter()
                .all(|problem| problem.contains("the manifest lists none")),
            "{:?}",
            report.problems
        );
        // An unreadable manifest is `check::run`'s problem, not this one's.
        fs::write(dir.join(MANIFEST_FILE), "not json").unwrap();
        let mut report = Report::default();
        add_variant_problems(&mut report, &records(), &read_dir_state(&dir).unwrap());
        assert!(report.problems.is_empty());
    }

    #[test]
    fn a_kernel_removed_from_the_source_is_reported_stale_and_its_files_are_pruned() {
        let scratch = Scratch::new("stale");
        let dir = scratch.0.join(ARTIFACTS_DIR);
        build_into(&dir, vec![sample(), other("b")], &FakeCompiler::new()).unwrap();
        fs::write(dir.join("stray.txt"), "x").unwrap();
        let gone = other("b").key;
        let summary = build_into(&dir, vec![sample()], &FakeCompiler::new()).unwrap();
        assert_eq!(summary.stale, [format!("{gone} (demo::b)")]);
        assert_eq!(
            summary.commit.pruned,
            [
                format!("{gone}.wgsl"),
                format!("{gone}.wgsl.reflection.json"),
                "stray.txt".to_owned()
            ]
        );
        let names: Vec<String> = snapshot(&dir).into_iter().map(|(name, _)| name).collect();
        assert_eq!(
            names,
            [
                format!("{SAMPLE_KEY}.wgsl"),
                format!("{SAMPLE_KEY}.wgsl.reflection.json"),
                "manifest.json".to_owned()
            ]
        );
        let lines = summary.lines();
        assert!(
            lines.contains(&format!("gust: stale kernel {gone} (demo::b) removed")),
            "{lines:?}"
        );
        assert!(
            lines.contains(&"gust: pruned stray.txt".to_owned()),
            "{lines:?}"
        );
    }

    #[test]
    fn a_failed_compile_writes_nothing_and_leaves_the_old_set_in_place() {
        let scratch = Scratch::new("compile-fail");
        let dir = scratch.0.join(ARTIFACTS_DIR);
        build_into(&dir, vec![sample()], &FakeCompiler::new()).unwrap();
        let before = snapshot(&dir);
        let failing = FakeCompiler {
            tag: "fake-slang-1",
            fail_on: Some(("gpu_demo_b", Format::Wgsl)),
        };
        let error = build_into(&dir, vec![sample(), other("b")], &failing).unwrap_err();
        assert_eq!(error, "demo::b (wgsl): injected compile failure");
        assert_eq!(snapshot(&dir), before);
        // And a first build that fails creates no directory at all.
        let fresh = scratch.0.join("fresh").join(ARTIFACTS_DIR);
        build_into(&fresh, vec![other("b")], &failing).unwrap_err();
        assert!(!fresh.exists());
    }

    #[test]
    fn a_failure_while_staging_leaves_no_new_file_and_no_staging_directory() {
        let scratch = Scratch::new("stage-fail");
        let dir = scratch.0.join(ARTIFACTS_DIR);
        build_into(&dir, vec![sample()], &FakeCompiler::new()).unwrap();
        let before = read_dir_state(&dir).unwrap();
        let mut staged = 0;
        let error = finish_build(
            &dir,
            vec![sample(), other("b")],
            &[Format::Wgsl],
            &FakeCompiler::new(),
            &mut || Ok(()),
            &mut |event| {
                if matches!(event, Event::Staged(_)) {
                    staged += 1;
                    if staged == 2 {
                        return Err(io::Error::other("disk full"));
                    }
                }
                Ok(())
            },
        )
        .unwrap_err();
        assert!(error.contains("disk full"), "{error}");
        assert_eq!(read_dir_state(&dir).unwrap(), before);
    }

    #[test]
    fn a_kernel_with_unchecked_parameters_is_built_listed_and_noted_once_without_refusing() {
        let scratch = Scratch::new("unchecked");
        let dir = scratch.0.join(ARTIFACTS_DIR);
        let mut unchecked = sample();
        unchecked.parameters[1].layout = None;
        let summary = build_into(
            &dir,
            vec![unchecked.clone(), other("b")],
            &FakeCompiler::new(),
        )
        .unwrap();
        assert_eq!(summary.kernels, 2);
        let notes: Vec<String> = summary
            .lines()
            .into_iter()
            .filter(|line| line.starts_with("gust: note: "))
            .collect();
        assert_eq!(
            notes,
            [format!(
                "gust: note: {}",
                check::unchecked_note("demo::run", &["pairs".to_owned()])
            )],
            "one note, for the one kernel"
        );
        let text = fs::read_to_string(dir.join(MANIFEST_FILE)).unwrap();
        let manifest = Manifest::parse(&text).unwrap();
        assert_eq!(
            manifest.entry(SAMPLE_KEY).unwrap().export_unchecked,
            ["pairs"]
        );
        assert!(
            manifest
                .entry(&other("b").key)
                .unwrap()
                .export_unchecked
                .is_empty()
        );
        // The same build again changes nothing, and `check` agrees with the list.
        let again = build_into(&dir, vec![other("b"), unchecked], &FakeCompiler::new()).unwrap();
        assert!(
            again.commit.written.is_empty(),
            "{:?}",
            again.commit.written
        );
    }

    #[test]
    fn a_half_pair_in_the_artifact_directory_is_repaired_by_a_build() {
        // The macro treats a half pair as absent in export mode, so a build is how it is repaired (R4.3).
        let reflection = format!("{SAMPLE_KEY}.wgsl.reflection.json");
        let artifact = format!("{SAMPLE_KEY}.wgsl");
        let scratch = Scratch::new("half-pair");
        let dir = scratch.0.join(ARTIFACTS_DIR);
        build_into(&dir, vec![sample()], &FakeCompiler::new()).unwrap();
        let whole = snapshot(&dir);
        // Each file of the pair missing in turn, and each replaced by other bytes (an interrupted or hand-edited
        // pair), with the manifest in place.
        for (what, remove, overwrite) in [
            ("reflection missing", Some(&reflection), None),
            ("artifact missing", Some(&artifact), None),
            ("artifact stale", None, Some(&artifact)),
            ("reflection stale", None, Some(&reflection)),
        ] {
            if let Some(name) = remove {
                fs::remove_file(dir.join(name)).unwrap();
            }
            if let Some(name) = overwrite {
                fs::write(dir.join(name), b"stale").unwrap();
            }
            let summary = build_into(&dir, vec![sample()], &FakeCompiler::new())
                .unwrap_or_else(|error| panic!("{what}: {error}"));
            assert_eq!(
                summary.commit.written.len(),
                1,
                "{what}: only the broken file is written"
            );
            assert!(summary.commit.pruned.is_empty(), "{what}");
            assert_eq!(snapshot(&dir), whole, "{what}");
        }
        // No manifest and one half of a pair: a build writes both files and the manifest.
        fs::remove_file(dir.join(MANIFEST_FILE)).unwrap();
        fs::remove_file(dir.join(&reflection)).unwrap();
        let summary = build_into(&dir, vec![sample()], &FakeCompiler::new()).unwrap();
        assert_eq!(
            summary.commit.written,
            [reflection.clone(), MANIFEST_FILE.to_owned()]
        );
        assert_eq!(snapshot(&dir), whole);
    }

    #[test]
    fn the_report_lines_end_with_a_verdict() {
        let ok = render_report(&Report::default(), 2);
        assert_eq!(
            ok,
            ["ok: gust-artifacts/ matches the 2 collected kernel(s)"]
        );
        let failed = render_report(
            &Report {
                problems: vec!["a".to_owned(), "b".to_owned()],
                notes: vec!["n".to_owned()],
            },
            2,
        );
        assert_eq!(
            failed,
            [
                "note: n",
                "problem: a",
                "problem: b",
                "FAILED: 2 problem(s) in gust-artifacts/"
            ]
        );
    }

    #[test]
    fn the_formats_a_check_recompiles_are_the_ones_the_manifest_lists() {
        let scratch = Scratch::new("formats");
        let dir = scratch.0.join(ARTIFACTS_DIR);
        finish_build(
            &dir,
            vec![sample()],
            &[Format::Wgsl, Format::Spirv],
            &FakeCompiler::new(),
            &mut || Ok(()),
            &mut |_| Ok(()),
        )
        .unwrap();
        assert_eq!(
            stored_formats(&read_dir_state(&dir).unwrap()),
            [Format::Wgsl, Format::Spirv]
        );
        assert_eq!(stored_formats(&store::DirState::default()), [Format::Wgsl]);
    }

    #[test]
    fn a_plain_build_keeps_the_formats_the_manifest_lists_and_a_fresh_one_builds_only_wgsl() {
        let scratch = Scratch::new("keep-spirv");
        let dir = scratch.0.join(ARTIFACTS_DIR);
        let first = formats_for_build(&[Format::Spirv], &read_dir_state(&dir).unwrap());
        assert_eq!(first, [Format::Wgsl, Format::Spirv]);
        assert_eq!(
            formats_for_build(&[], &read_dir_state(&dir).unwrap()),
            [Format::Wgsl]
        );
        build_into_formats(&dir, &first);
        let before = snapshot(&dir);
        assert!(before.iter().any(|(name, _)| name.ends_with(".spv")));
        // `cargo gust build` with no --format: SPIR-V stays, nothing is rewritten or pruned.
        let formats = formats_for_build(&[], &read_dir_state(&dir).unwrap());
        assert_eq!(formats, [Format::Wgsl, Format::Spirv]);
        let summary = build_into_formats(&dir, &formats);
        assert!(summary.commit.written.is_empty() && summary.commit.pruned.is_empty());
        assert_eq!(snapshot(&dir), before);
        // Asking for what is already there changes nothing either.
        assert_eq!(
            formats_for_build(
                &[Format::Spirv, Format::Wgsl],
                &read_dir_state(&dir).unwrap()
            ),
            formats
        );
    }

    fn build_into_formats(dir: &Path, formats: &[Format]) -> BuildSummary {
        finish_build(
            dir,
            vec![sample()],
            formats,
            &FakeCompiler::new(),
            &mut || Ok(()),
            &mut |_| Ok(()),
        )
        .unwrap()
    }

    #[test]
    fn a_failing_before_commit_step_writes_nothing_and_a_passing_one_runs_after_the_plan() {
        let scratch = Scratch::new("before-commit");
        let dir = scratch.0.join(ARTIFACTS_DIR);
        let mut calls = 0;
        let error = finish_build(
            &dir,
            vec![sample()],
            &[Format::Wgsl],
            &FakeCompiler::new(),
            &mut || {
                calls += 1;
                Err("the clean failed".to_owned())
            },
            &mut |_| Ok(()),
        )
        .unwrap_err();
        assert_eq!(error, "the clean failed");
        assert_eq!(calls, 1);
        assert!(!dir.exists(), "no manifest, so the next run retries");
        // A compile failure never reaches the step (no clean for a build that cannot succeed).
        let failing = FakeCompiler {
            tag: "fake-slang-1",
            fail_on: Some(("gpu_demo_run", Format::Wgsl)),
        };
        let mut reached = false;
        finish_build(
            &dir,
            vec![sample()],
            &[Format::Wgsl],
            &failing,
            &mut || {
                reached = true;
                Ok(())
            },
            &mut |_| Ok(()),
        )
        .unwrap_err();
        assert!(!reached);
    }
}
