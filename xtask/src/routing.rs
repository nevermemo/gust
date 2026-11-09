//! `cargo xtask check-routing`: the routing driver (`cargo gust run`, `test`, `build`, `check` and their `gust.toml`;
//! m8.r2, D65 to D67, D137) against the fixture workspace `tests/routing`.
//!
//! The fixture is copied to a scratch directory under the cargo target directory (its `gust.toml` files are rewritten per
//! state, and the repository stays unchanged), and built with a target directory of its own, warm across the states.
//! Every child starts without `GUST_PROFILE`, `GUST_ROUTE__*` and rustflags variables.
//!
//! - `plain` (R1): plain `cargo`, no driver: the macro's own source defaults.
//! - `resolution` (R2): seven states of `gust.toml` and profile. The probe's observed targets, the driver's `routes.json`
//!   and a golden table must agree: this pins the driver's resolution to the macro's (one resolution, D137).
//! - `errors` (R3): bad `gust.toml` files fail with exit 2, file, line and tag, before cargo starts.
//! - `test-run` (R4): `cargo gust test` and `cargo gust run` pass the variables to the build (the control, plain
//!   `cargo test`, fails with the E0277 "cannot run the kernel": under plain cargo a GPU kernel is called on `CpuDevice`).
//! - `flip` (R5): one warm target directory and one cargo command, the route variable unset, set and unset again, in both
//!   directions twice; each state must rebuild and print the new target. With the macro's route tracking removed this case
//!   must fail (the mutation proof, done by hand: `--case flip`).
//! - `lint` (R7): `cargo fmt --check` on the fixture, and clippy (all targets, warnings denied) per fixture package under
//!   that package's own route. It is part of `--check-only`, so CI covers the fixture's style.
//! - `gpu` (R6, not with `--check-only`): `gpu-app` dispatches the GPU-routed kernel on an adapter (named in the output);
//!   routed to the CPU, the same source does not compile.

use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

use crate::{
    json::{self, Json},
    process::{PROFILE_VARIABLE, exit_code, set_workspace_profile, without_colors},
    workspace::{cargo_target_dir, workspace_root},
};

/// The fixture workspace, relative to the repository root. It is a workspace of its own.
pub(crate) const FIXTURE_DIR: &str = "tests/routing";

/// The prefix of every route variable (`gust::routes::VARIABLE_PREFIX`).
const ROUTE_PREFIX: &str = "GUST_ROUTE__";

/// Variables every child starts without (the route variables are removed by prefix).
const CLEARED: [&str; 4] = [
    PROFILE_VARIABLE,
    "GUST_ROUTED_PROFILE",
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
];

/// What `cpu-app` prints when `physics::scale` ran on `CpuDevice`.
pub(crate) const CPU_APP_LINE: &str =
    "routing: physics::scale ran on CpuDevice [2.0, 4.0, 6.0, 8.0]";

/// What `gpu-app` prints when `physics::scale` ran on the GPU.
pub(crate) const GPU_APP_LINE: &str = "routing: physics::scale ran on the GPU [2.0, 4.0, 6.0, 8.0]";

/// The line of `gpu-app` that names the adapter.
const ADAPTER_PREFIX: &str = "adapter: ";

/// The tags `route-kernels` declares, in the order of `routes.json` (by package, then tag).
pub(crate) const TAGS: [&str; 6] = [
    "fusion",
    "physics",
    "physics.broadphase",
    "render",
    "solver",
    "solver.pinned",
];

/// The deciding levels of `routes.json`.
const DEFAULT: &str = "default";
const ENTRY: &str = "entry";
const FALLBACK: &str = "fallback";
const PROFILE: &str = "profile";

/// The case names, in the order they run.
pub(crate) const CASES: [&str; 7] = [
    "plain",
    "resolution",
    "errors",
    "test-run",
    "flip",
    "lint",
    "gpu",
];

/// One state of the resolution case: a profile, the body of the probe's `gust.toml` after `[routes]`, and the target and
/// level `routes.json` must give each of [`TAGS`]. The probe's observed lines are derived from the rows.
pub(crate) struct State {
    pub(crate) name: &'static str,
    pub(crate) profile: Option<&'static str>,
    pub(crate) entries: &'static str,
    pub(crate) rows: [(&'static str, &'static str); 6],
}

const NONE_ROWS: [(&str, &str); 6] = [
    ("cpu", DEFAULT),
    ("gpu", DEFAULT),
    ("gpu", DEFAULT),
    ("gpu", FALLBACK),
    ("gpu", FALLBACK),
    ("gpu", FALLBACK),
];

const PHYSICS_CPU_ROWS: [(&str, &str); 6] = [
    ("cpu", DEFAULT),
    ("cpu", ENTRY),
    ("cpu", ENTRY),
    ("gpu", FALLBACK),
    ("gpu", FALLBACK),
    ("gpu", FALLBACK),
];

/// The states of the resolution case.
pub(crate) const STATES: [State; 7] = [
    State {
        name: "no entries: the declared defaults, then gpu",
        profile: None,
        entries: "",
        rows: NONE_ROWS,
    },
    State {
        name: "physics to the cpu: its child follows",
        profile: None,
        entries: "\"route-kernels::physics\" = \"cpu\"\n",
        rows: PHYSICS_CPU_ROWS,
    },
    State {
        name: "the child keeps an entry of its own",
        profile: None,
        entries: "\"route-kernels::physics\" = \"cpu\"\n\"route-kernels::physics.broadphase\" = \"gpu\"\n",
        rows: [
            ("cpu", DEFAULT),
            ("cpu", ENTRY),
            ("gpu", ENTRY),
            ("gpu", FALLBACK),
            ("gpu", FALLBACK),
            ("gpu", FALLBACK),
        ],
    },
    State {
        name: "an entry beats a source default (fusion to the gpu)",
        profile: None,
        entries: "\"route-kernels::fusion\" = \"gpu\"\n",
        rows: [
            ("gpu", ENTRY),
            ("gpu", DEFAULT),
            ("gpu", DEFAULT),
            ("gpu", FALLBACK),
            ("gpu", FALLBACK),
            ("gpu", FALLBACK),
        ],
    },
    State {
        name: "a target-keyed entry under native",
        profile: Some("native"),
        entries: "\"route-kernels::physics\" = { native = \"cpu\", web = \"gpu\" }\n",
        rows: PHYSICS_CPU_ROWS,
    },
    State {
        name: "a target-keyed entry under dev uses the native key",
        profile: None,
        entries: "\"route-kernels::physics\" = { native = \"cpu\" }\n",
        rows: PHYSICS_CPU_ROWS,
    },
    State {
        name: "the cpu profile sends every tag to the cpu",
        profile: Some("cpu"),
        entries: "",
        rows: [
            ("cpu", PROFILE),
            ("cpu", PROFILE),
            ("cpu", PROFILE),
            ("cpu", PROFILE),
            ("cpu", PROFILE),
            ("cpu", PROFILE),
        ],
    },
];

/// The lines the probe prints for `state`: `physics scale`, `physics.broadphase pairs` and `fusion blend`.
pub(crate) fn probe_lines(state: &State) -> Vec<String> {
    probe_lines_for(state.rows[1].0, state.rows[2].0, state.rows[0].0)
}

fn probe_lines_for(physics: &str, broadphase: &str, fusion: &str) -> Vec<String> {
    vec![
        format!("physics scale {physics}"),
        format!("physics.broadphase pairs {broadphase}"),
        format!("fusion blend {fusion}"),
    ]
}

/// What a bad `gust.toml` must be refused with: the file and line, the tag or key, and a phrase.
pub(crate) struct ErrorCase {
    pub(crate) name: &'static str,
    pub(crate) profile: Option<&'static str>,
    pub(crate) entries: &'static str,
    pub(crate) needles: &'static [&'static str],
}

/// The error states: each is refused with exit code 2, before cargo starts.
pub(crate) const ERRORS: [ErrorCase; 9] = [
    ErrorCase {
        name: "a syntax error",
        profile: None,
        entries: "\"route-kernels::physics\" = \"cpu\" x\n",
        needles: &["gust.toml:2: ", "after the value"],
    },
    ErrorCase {
        name: "a key that is not <package>::<tag>",
        profile: None,
        entries: "physics = \"cpu\"\n",
        needles: &["gust.toml:2: ", "must be quoted"],
    },
    ErrorCase {
        name: "a tag no package declares",
        profile: None,
        entries: "\"route-kernels::nope\" = \"cpu\"\n",
        needles: &["gust.toml:2: ", "route-kernels::nope", "unknown tag"],
    },
    ErrorCase {
        name: "a package outside the dependency graph",
        profile: None,
        entries: "\"ghost::physics\" = \"cpu\"\n",
        needles: &["gust.toml:2: ", "ghost::physics", "unknown tag"],
    },
    ErrorCase {
        name: "a target outside the tag's set",
        profile: None,
        entries: "\"route-kernels::render\" = \"cpu\"\n",
        needles: &[
            "gust.toml:2: ",
            "route-kernels::render",
            "targets = [\"gpu\"]",
        ],
    },
    ErrorCase {
        name: "an inherited conflict",
        profile: None,
        entries: "\"route-kernels::solver\" = \"cpu\"\n",
        needles: &[
            "gust.toml:2: ",
            "route-kernels::solver",
            "solver.pinned",
            "by inheritance",
        ],
    },
    ErrorCase {
        name: "a key given twice",
        profile: None,
        entries: "\"route-kernels::physics\" = \"cpu\"\n\"route-kernels::physics\" = \"gpu\"\n",
        needles: &["gust.toml:3: ", "repeats the one on line 2"],
    },
    ErrorCase {
        name: "a capability profile",
        profile: None,
        entries: "\"route-kernels::physics\" = { \"native+subgroups\" = \"gpu\" }\n",
        needles: &[
            "gust.toml:2: ",
            "route-kernels::physics",
            "capability profile",
        ],
    },
    ErrorCase {
        name: "a gpu entry under the cpu profile",
        profile: Some("cpu"),
        entries: "\"route-kernels::physics\" = \"gpu\"\n",
        needles: &[
            "gust.toml:2: ",
            "route-kernels::physics",
            "GUST_PROFILE=cpu",
        ],
    },
];

/// How a state of the flip case builds the probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Via {
    /// Plain `cargo run`: no driver, so no route variable is set at all.
    Unset,
    /// The driver with an entry sending `physics` to the cpu: the variable is set to `cpu`.
    Entry,
    /// The driver with no entry: the variable is set to the declared default (`gpu`).
    Default,
}

/// The states of the flip case, in one warm target directory and with one cargo command: `cargo run --locked -q -p
/// probe`, which differs between the states only by the environment the driver gives it. Both directions occur twice:
/// unset to set, set to unset, set to the default and the default to set.
pub(crate) const FLIPS: [Via; 7] = [
    Via::Unset,
    Via::Entry,
    Via::Default,
    Via::Entry,
    Via::Unset,
    Via::Entry,
    Via::Default,
];

/// The target `physics` has after a flip state.
pub(crate) fn flip_target(via: Via) -> &'static str {
    match via {
        Via::Entry => "cpu",
        Via::Unset | Via::Default => "gpu",
    }
}

/// What `check-routing` was asked to run.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Options {
    pub(crate) cases: Vec<&'static str>,
}

impl Options {
    /// `[--check-only] [--case NAME]...`: without `--case`, every case (the `gpu` case unless `--check-only`).
    pub(crate) fn parse(args: &[String]) -> Result<Self, String> {
        let mut check_only = false;
        let mut chosen: Vec<&'static str> = Vec::new();
        let mut index = 0;
        while let Some(argument) = args.get(index) {
            index += 1;
            match argument.as_str() {
                "--check-only" => check_only = true,
                "--case" => {
                    let name = args
                        .get(index)
                        .ok_or_else(|| "`--case` needs a case name".to_owned())?;
                    index += 1;
                    let known = CASES.iter().find(|known| **known == name).ok_or_else(|| {
                        format!("unknown case `{name}`; the cases are {}", CASES.join(", "))
                    })?;
                    if !chosen.contains(known) {
                        chosen.push(known);
                    }
                }
                other => {
                    return Err(format!(
                        "check-routing takes `--check-only` and `--case <name>`, not `{other}`"
                    ));
                }
            }
        }
        if chosen.is_empty() {
            chosen = CASES
                .iter()
                .copied()
                .filter(|case| !(check_only && *case == "gpu"))
                .collect();
        } else if check_only && chosen.contains(&"gpu") {
            return Err("`--check-only` leaves out the gpu case; drop one of the two".to_owned());
        }
        chosen.sort_by_key(|case| CASES.iter().position(|known| known == case));
        Ok(Self { cases: chosen })
    }
}

/// `manifest` with its `path = "../../../crates/` dependencies pointing at `root`: the copy of the fixture does not sit
/// three directories below the repository root. `root` uses forward slashes.
pub(crate) fn rewrite_manifest(manifest: &str, root: &str) -> String {
    manifest.replace(
        "path = \"../../../crates/",
        &format!("path = \"{}/crates/", root.trim_end_matches('/')),
    )
}

/// The last `count` lines of `text`, for a failure message.
pub(crate) fn tail(text: &str, count: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(count)..].join("\n")
}

/// What a child printed and how it ended.
struct Done {
    success: bool,
    code: String,
    stdout: String,
    stderr: String,
}

impl Done {
    fn everything(&self) -> String {
        format!("{}\n{}", self.stdout, self.stderr)
    }

    /// The lines of standard output, trimmed.
    fn lines(&self) -> Vec<String> {
        self.stdout
            .lines()
            .map(|line| line.trim().to_owned())
            .collect()
    }
}

fn exec(mut command: Command, label: &str) -> Result<Done, String> {
    let output = command
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("could not launch {label}: {error}"))?;
    Ok(Done {
        success: output.status.success(),
        code: exit_code(output.status),
        stdout: without_colors(&String::from_utf8_lossy(&output.stdout)),
        stderr: without_colors(&String::from_utf8_lossy(&output.stderr)),
    })
}

/// The rows `(tag, target, level)` of a `routes.json`, in file order.
pub(crate) fn routes_rows(text: &str) -> Result<Vec<(String, String, String)>, String> {
    let document = json::parse(text).map_err(|error| format!("routes.json: {error}"))?;
    let routes = document
        .get("routes")
        .and_then(Json::as_array)
        .ok_or("routes.json has no `routes` array")?;
    routes
        .iter()
        .map(|row| {
            let field = |name: &str| {
                row.get(name)
                    .and_then(Json::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| format!("routes.json: a row has no `{name}`"))
            };
            Ok((field("tag")?, field("target")?, field("level")?))
        })
        .collect()
}

/// The prepared scratch copy of the fixture and the driver.
struct Check {
    cargo: OsString,
    driver: PathBuf,
    workspace: PathBuf,
    target: PathBuf,
}

impl Check {
    /// A command in the scratch workspace with a clean environment and the fixture's target directory.
    fn command(&self, program: impl AsRef<std::ffi::OsStr>) -> Command {
        let mut command = Command::new(program);
        command
            .current_dir(&self.workspace)
            .env("CARGO_TARGET_DIR", &self.target)
            .env("CARGO_TERM_COLOR", "never");
        for name in CLEARED {
            command.env_remove(name);
        }
        for (name, _) in env::vars_os() {
            if name.to_string_lossy().starts_with(ROUTE_PREFIX) {
                command.env_remove(name);
            }
        }
        command
    }

    /// `cargo gust <verb> <args>` with `GUST_PROFILE` when `profile` names one.
    fn driver_command(&self, verb: &str, args: &[&str], profile: Option<&str>) -> Command {
        let mut command = self.command(&self.driver);
        command.arg("gust").arg(verb).args(args);
        if let Some(profile) = profile {
            command.env(PROFILE_VARIABLE, profile);
        }
        command
    }

    /// Plain `cargo <args>`: no driver.
    fn cargo_command(&self, args: &[&str]) -> Command {
        let mut command = self.command(&self.cargo);
        command.args(args);
        command
    }

    fn gust_toml(&self, app: &str) -> PathBuf {
        self.workspace.join(app).join("gust.toml")
    }

    fn write_entries(&self, app: &str, entries: &str) -> Result<(), String> {
        let path = self.gust_toml(app);
        fs::write(&path, format!("[routes]\n{entries}"))
            .map_err(|error| format!("cannot write {}: {error}", path.display()))
    }

    fn routes_json(&self, app: &str) -> PathBuf {
        self.workspace
            .join(app)
            .join("gust-artifacts")
            .join("routes.json")
    }

    /// Runs the probe through the driver with `entries` in its `gust.toml`.
    fn driver_probe(&self, entries: &str, profile: Option<&str>) -> Result<Done, String> {
        self.write_entries("probe", entries)?;
        let _ = fs::remove_file(self.routes_json("probe"));
        exec(
            self.driver_command("run", &["--locked", "-q", "-p", "probe"], profile),
            "cargo gust run -p probe",
        )
    }

    fn plain_probe(&self) -> Result<Done, String> {
        exec(
            self.cargo_command(&["run", "--locked", "-q", "-p", "probe"]),
            "cargo run -p probe",
        )
    }

    fn require_probe(done: &Done, wanted: &[String], label: &str) -> Result<(), String> {
        if !done.success {
            return Err(format!(
                "{label}: exit {}:\n{}",
                done.code,
                tail(&done.everything(), 25)
            ));
        }
        if done.lines() != wanted {
            return Err(format!(
                "{label}: the probe printed {:?}, wanted {wanted:?}",
                done.lines()
            ));
        }
        Ok(())
    }

    /// R1: plain cargo gives the macro's own defaults.
    fn plain(&self) -> Result<String, String> {
        self.write_entries("probe", "\"route-kernels::physics\" = \"cpu\"\n")?;
        let done = self.plain_probe()?;
        Self::require_probe(
            &done,
            &probe_lines_for("gpu", "gpu", "cpu"),
            "plain cargo (a gust.toml it does not read)",
        )?;
        Ok("plain cargo ignores gust.toml: physics gpu, broadphase gpu, fusion cpu (the source defaults)".to_owned())
    }

    /// R2: the driver's table, the macro's result and the golden are one.
    fn resolution(&self) -> Result<String, String> {
        for state in &STATES {
            let label = format!("resolution `{}`", state.name);
            let done = self.driver_probe(state.entries, state.profile)?;
            Self::require_probe(&done, &probe_lines(state), &label)?;
            let text = fs::read_to_string(self.routes_json("probe"))
                .map_err(|error| format!("{label}: no routes.json: {error}"))?;
            let rows = routes_rows(&text).map_err(|error| format!("{label}: {error}"))?;
            let wanted: Vec<(String, String, String)> = TAGS
                .iter()
                .zip(state.rows)
                .map(|(tag, (target, level))| {
                    ((*tag).to_owned(), target.to_owned(), level.to_owned())
                })
                .collect();
            if rows != wanted {
                return Err(format!(
                    "{label}: routes.json says {rows:?}, the golden says {wanted:?}"
                ));
            }
            // A rerun gives the same bytes.
            let again = self.driver_probe(state.entries, state.profile)?;
            Self::require_probe(&again, &probe_lines(state), &label)?;
            let second = fs::read_to_string(self.routes_json("probe")).unwrap_or_default();
            if second != text {
                return Err(format!("{label}: a rerun wrote a different routes.json"));
            }
        }
        Ok(format!(
            "{} states: the probe's targets, routes.json and the golden agree; a rerun gives the same bytes",
            STATES.len()
        ))
    }

    /// R3: bad files are refused before cargo starts.
    fn errors(&self) -> Result<String, String> {
        for case in &ERRORS {
            let label = format!("error `{}`", case.name);
            let done = self.driver_probe(case.entries, case.profile)?;
            let text = done.everything();
            if done.success || done.code != "2" {
                return Err(format!(
                    "{label}: wanted exit 2, got {} (success {}):\n{}",
                    done.code,
                    done.success,
                    tail(&text, 12)
                ));
            }
            if let Some(missing) = case.needles.iter().find(|needle| !text.contains(**needle)) {
                return Err(format!(
                    "{label}: the message lacks `{missing}`:\n{}",
                    tail(&text, 12)
                ));
            }
            if text.contains("Compiling") || text.contains("Running") {
                return Err(format!("{label}: cargo started:\n{}", tail(&text, 12)));
            }
        }
        Ok(format!(
            "{} bad files refused with exit 2, file, line and tag, before cargo started",
            ERRORS.len()
        ))
    }

    /// R4: `cargo gust test` and `cargo gust run` pass the variables.
    fn test_run(&self) -> Result<String, String> {
        let control = exec(
            self.cargo_command(&["test", "--locked", "-q", "-p", "cpu-app"]),
            "cargo test -p cpu-app",
        )?;
        // The control must fail for the reason that matters: under plain cargo `physics::scale` is a GPU kernel and the
        // call on `CpuDevice` is an E0277, "cannot run the kernel". A network error, a lock mismatch or another compile
        // error does not count.
        let control_text = control.everything();
        if control.success {
            return Err(
                "the control, plain `cargo test -p cpu-app`, passed: the test does not depend on the route"
                    .to_owned(),
            );
        }
        if !control_text.contains("E0277") || !control_text.contains("cannot run the kernel") {
            return Err(format!(
                "the control, plain `cargo test -p cpu-app`, failed for another reason than the E0277 \"cannot run the kernel\" of a GPU kernel on `CpuDevice` (exit {}):
{}",
                control.code,
                tail(&control_text, 25)
            ));
        }
        let test = exec(
            self.driver_command("test", &["--locked", "-p", "cpu-app"], None),
            "cargo gust test -p cpu-app",
        )?;
        let text = test.everything();
        if !test.success
            || !text.contains("the_cpu_route_reached_the_build ... ok")
            || !text.contains("1 passed")
        {
            return Err(format!(
                "`cargo gust test -p cpu-app` (exit {}) did not pass its test:\n{}",
                test.code,
                tail(&text, 25)
            ));
        }
        for profile in [None, Some("native")] {
            let run = exec(
                self.driver_command("run", &["--locked", "-q", "-p", "cpu-app"], profile),
                "cargo gust run -p cpu-app",
            )?;
            let found = run
                .lines()
                .iter()
                .filter(|line| *line == CPU_APP_LINE)
                .count();
            if !run.success || found != 1 {
                return Err(format!(
                    "`cargo gust run -p cpu-app` under {profile:?} (exit {}) must print `{CPU_APP_LINE}` once, printed {found}:\n{}",
                    run.code,
                    tail(&run.everything(), 25)
                ));
            }
        }
        Ok("cargo gust test passes where plain cargo test fails with E0277 (cannot run the kernel); cargo gust run runs the CPU-routed kernel under dev and native".to_owned())
    }

    /// R5: the flip, in both directions, in one warm target directory with one cargo command.
    fn flip(&self) -> Result<String, String> {
        let mut steps = Vec::new();
        for (index, via) in FLIPS.into_iter().enumerate() {
            let target = flip_target(via);
            let done = match via {
                Via::Unset => self.plain_probe()?,
                Via::Entry => self.driver_probe("\"route-kernels::physics\" = \"cpu\"\n", None)?,
                Via::Default => self.driver_probe("", None)?,
            };
            Self::require_probe(
                &done,
                &probe_lines_for(target, target, "cpu"),
                &format!("flip state {} ({via:?})", index + 1),
            )?;
            steps.push(format!("{via:?}={target}"));
        }
        Ok(format!(
            "the build followed every change of the route variable ({})",
            steps.join(", ")
        ))
    }

    /// R7: the fixture is formatted, and clippy has nothing to say about a package built under its own route (the CPU
    /// apps with `physics` on the CPU, `gpu-app` on the GPU; the variable is what the driver would set).
    fn lint(&self) -> Result<String, String> {
        let format = exec(
            self.cargo_command(&["fmt", "--all", "--check"]),
            "cargo fmt --all --check",
        )?;
        if !format.success {
            return Err(format!(
                "cargo fmt --check found a difference in tests/routing (exit {}):
{}",
                format.code,
                tail(&format.everything(), 25)
            ));
        }
        let physics = format!("{ROUTE_PREFIX}route_kernels__physics");
        for (packages, route) in [
            (&["route-kernels", "probe", "cpu-app"][..], "cpu"),
            (&["gpu-app"][..], "gpu"),
        ] {
            let mut arguments = vec!["clippy", "--locked", "--all-targets"];
            for package in packages {
                arguments.extend(["-p", package]);
            }
            arguments.extend(["--", "-D", "warnings"]);
            let mut command = self.cargo_command(&arguments);
            command.env(&physics, route);
            let done = exec(command, "cargo clippy")?;
            if !done.success {
                return Err(format!(
                    "clippy on {} with physics on the {route} (exit {}):
{}",
                    packages.join(", "),
                    done.code,
                    tail(&done.everything(), 30)
                ));
            }
        }
        Ok("rustfmt is clean, and clippy (all targets, -D warnings) passes for each package under its own route".to_owned())
    }

    /// R6: the GPU step, on the adapter the machine has.
    fn gpu(&self) -> Result<String, String> {
        self.write_entries("gpu-app", "\"route-kernels::physics\" = \"gpu\"\n")?;
        let run = exec(
            self.driver_command("run", &["--locked", "-q", "-p", "gpu-app"], None),
            "cargo gust run -p gpu-app",
        )?;
        let lines = run.lines();
        let adapter = lines
            .iter()
            .find_map(|line| line.strip_prefix(ADAPTER_PREFIX))
            .map(str::to_owned);
        let found = lines.iter().filter(|line| *line == GPU_APP_LINE).count();
        let Some(adapter) = adapter.filter(|_| run.success && found == 1) else {
            return Err(format!(
                "`cargo gust run -p gpu-app` (exit {}) must name an adapter and print `{GPU_APP_LINE}` once:\n{}",
                run.code,
                tail(&run.everything(), 30)
            ));
        };
        self.write_entries("gpu-app", "\"route-kernels::physics\" = \"cpu\"\n")?;
        let flipped = exec(
            self.driver_command("run", &["--locked", "-q", "-p", "gpu-app"], None),
            "cargo gust run -p gpu-app (physics to the cpu)",
        )?;
        let text = flipped.everything();
        if flipped.success || !text.contains("E0277") {
            return Err(format!(
                "with physics routed to the cpu, `gpu-app` must fail with E0277 (a GPU device refuses a CPU kernel); exit {}:\n{}",
                flipped.code,
                tail(&text, 25)
            ));
        }
        self.write_entries("gpu-app", "\"route-kernels::physics\" = \"gpu\"\n")?;
        Ok(format!(
            "physics::scale ran on the GPU `{adapter}` and matched its host values; routed to the cpu the same source fails with E0277"
        ))
    }
}

/// Copies the fixture to `to`, skipping build output and routing output, with the manifests' repository paths rewritten.
fn copy_fixture(from: &Path, to: &Path, root: &str) -> Result<(), String> {
    fs::create_dir_all(to).map_err(|error| format!("cannot create {}: {error}", to.display()))?;
    let entries =
        fs::read_dir(from).map_err(|error| format!("cannot read {}: {error}", from.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| error.to_string())?;
        let name = entry.file_name();
        let source = entry.path();
        let target = to.join(&name);
        if source.is_dir() {
            if name == "target" || name == "gust-artifacts" {
                continue;
            }
            copy_fixture(&source, &target, root)?;
        } else if name == "Cargo.toml" {
            let text = fs::read_to_string(&source)
                .map_err(|error| format!("cannot read {}: {error}", source.display()))?;
            fs::write(&target, rewrite_manifest(&text, root))
                .map_err(|error| format!("cannot write {}: {error}", target.display()))?;
        } else {
            fs::copy(&source, &target)
                .map_err(|error| format!("cannot copy {}: {error}", source.display()))?;
        }
    }
    Ok(())
}

/// Builds `cargo-gust` in the repository workspace and returns the binary.
fn build_driver(root: &Path, cargo: &OsString, target: &Path) -> Result<PathBuf, String> {
    let mut build = Command::new(cargo);
    build
        .current_dir(root)
        .args(["build", "--locked", "--color", "never", "-p", "cargo-gust"])
        .stdin(Stdio::null());
    set_workspace_profile(&mut build);
    let done = exec(build, "cargo build -p cargo-gust")?;
    if !done.success {
        return Err(format!(
            "cargo build -p cargo-gust failed (exit {}):\n{}",
            done.code,
            tail(&done.everything(), 25)
        ));
    }
    let binary = target
        .join("debug")
        .join(format!("cargo-gust{}", env::consts::EXE_SUFFIX));
    if !binary.is_file() {
        return Err(format!(
            "cargo built cargo-gust, but {} does not exist",
            binary.display()
        ));
    }
    Ok(binary)
}

fn prepare(root: &Path) -> Result<Check, String> {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    let target = root.join(cargo_target_dir(root));
    let driver = build_driver(root, &cargo, &target)?;
    let scratch = target.join("routing-scratch");
    if scratch.exists() {
        fs::remove_dir_all(&scratch)
            .map_err(|error| format!("cannot remove {}: {error}", scratch.display()))?;
    }
    let workspace = scratch.join("workspace");
    let root_text = root.to_string_lossy().replace('\\', "/");
    copy_fixture(&root.join(FIXTURE_DIR), &workspace, &root_text)?;
    Ok(Check {
        cargo,
        driver,
        workspace,
        target: target.join("routing-fixture"),
    })
}

/// `cargo xtask check-routing [--check-only] [--case NAME]...`.
pub(crate) fn check_routing(args: &[String]) -> Result<(), String> {
    let options = Options::parse(args)?;
    let root = workspace_root()?;
    let check = prepare(&root)?;
    let started = Instant::now();
    for name in &options.cases {
        let step_started = Instant::now();
        let summary = match *name {
            "plain" => check.plain(),
            "resolution" => check.resolution(),
            "errors" => check.errors(),
            "test-run" => check.test_run(),
            "flip" => check.flip(),
            "lint" => check.lint(),
            "gpu" => check.gpu(),
            other => Err(format!("no case `{other}`")),
        }
        .map_err(|error| format!("check-routing {name} failed: {error}"))?;
        println!(
            "check-routing {name}: ok ({:.1} s) {summary}",
            step_started.elapsed().as_secs_f64()
        );
    }
    println!(
        "check-routing passed in {:.1} s ({}): the routing driver resolved, flipped and passed routes to cargo as the macro reads them",
        started.elapsed().as_secs_f64(),
        options.cases.join(", ")
    );
    Ok(())
}

#[cfg(test)]
#[path = "tests/routing.rs"]
mod routing_tests;
