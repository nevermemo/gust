//! The ignored tests of the examples, pinned by name (m13.exgpu), and `cargo xtask check-examples --gpu`.
//!
//! Every example keeps its validation in `#[ignore]`d tests (the program on the adapter, the descriptors it
//! exports, the stages it compiles), which only `--include-ignored` runs. A count alone passes when one of them is
//! renamed, deleted or loses its attribute, because the example's ordinary tests still run; so each name below must be
//! reported `ok` by the call that runs the example ([`check_pinned_example_tests`], used by the `check-examples` and
//! `check-full` modes and by `--gpu`). Adding, renaming or removing an ignored test of an example changes its entry in
//! the same commit; a unit test reads the example sources and compares them with the table.

use crate::{
    checks::tests_not_ok,
    gpu_jobs::{
        Job, JobOutput, Package, TestExecutable, judge_named, no_run_build,
        package_of as package_environment, read_metadata, run_executable, run_printed,
        test_executables,
    },
    process::{require_tests_ran, run_cargo_test_captured, sum_test_results},
    verify::{name_adapter, prepare_helper},
    workspace::workspace_root,
};
use std::{cmp::Reverse, path::Path, time::Duration};

/// How many test executables of the examples run at a time when `--gpu-jobs` is not given (a value of its own, apart
/// from the Gpu suite's).
const DEFAULT_EXAMPLE_JOBS: usize = 3;

/// How long one example test executable may run. The slowest takes well under a minute, and the gate's outer time
/// limit is longer, so a hang is killed and named here instead of leaving test processes behind.
const EXAMPLE_JOB_TIME_LIMIT: Duration = Duration::from_secs(300);

/// The examples whose dependency features differ from the others (`subgroup-reduce` leaves out `slang-runtime` on
/// purpose: its strict proof refuses the run-time Slang compiler). Cargo unifies the features of the packages of one
/// call, so each of these is built and listed by a call of its own; a unit test requires the packages of the combined
/// call to declare the same features.
const OWN_BUILD: &[&str] = &["subgroup-reduce"];

/// The ignored tests of each example package, by the name `cargo test` prints (`app::tests::..` for the examples
/// that keep their tests in `src/app.rs`, `tests::..` for the single-file ones).
const EXAMPLE_GPU_TESTS: &[(&str, &[&str])] = &[
    ("atomic-counter", &["tests::twin_matches_reference_and_gpu"]),
    (
        "vector-add",
        &[
            "app::tests::twin_matches_reference_and_gpu",
            "app::tests::descriptor_is_slang_first_and_reflects_bindings",
            "app::tests::slang_compiles_to_readable_wgsl",
            "app::tests::rust_to_slang_executes_on_headless_wgpu",
        ],
    ),
    (
        "polynomial",
        &[
            "app::tests::twin_matches_reference_and_gpu_within_the_derived_bound",
            "app::tests::cpu_reference_evaluates_all_steps",
            "app::tests::headless_gpu_matches_cpu_reference",
            "app::tests::descriptor_and_generated_host_code_are_executable",
        ],
    ),
    (
        "signal-pipeline",
        &[
            "app::tests::twin_matches_reference_and_gpu_within_the_derived_bound",
            "app::tests::cpu_reference_runs_the_full_pipeline",
            "app::tests::headless_gpu_matches_cpu_reference",
            "app::tests::descriptor_and_generated_host_code_are_executable",
        ],
    ),
    (
        "particle-step",
        &[
            "app::tests::twin_matches_reference_and_gpu_for_the_exact_kernels",
            "app::tests::step_twin_matches_reference_and_gpu_within_the_derived_bounds",
            "app::tests::nested_layout_matches_executable_artifacts",
            "app::tests::scalar_read_after_write",
            "app::tests::field_read_after_write",
            "app::tests::cpu_step_has_expected_physics_and_preserves_metadata",
            "app::tests::gpu_nested_structs_and_dependent_batches_match_cpu",
        ],
    ),
    (
        "typed-pipeline",
        &[
            "app::tests::twin_matches_reference_and_gpu_within_the_derived_bounds",
            "app::tests::descriptors_cover_distinct_types_helpers_and_bindings",
            "app::tests::every_stage_compiles_to_wgsl",
            "app::tests::dependent_multi_dispatch_pipeline_matches_cpu",
            "app::tests::struct_layouts_are_distinct_and_storage_compatible",
        ],
    ),
    (
        "staged-graph",
        &[
            "app::tests::twin_matches_reference_and_gpu_within_the_derived_bounds",
            "app::tests::every_stage_compiles_to_wgsl",
            "app::tests::graph_matches_cpu_and_keeps_intermediate_resident",
            "app::tests::second_update_observes_new_settings_in_order",
            "app::tests::undeclared_dependencies_and_invalid_resources_are_rejected",
        ],
    ),
    (
        "component-pool",
        &[
            "app::tests::twin_matches_reference_and_gpu_within_the_derived_bounds",
            "app::tests::every_stage_compiles_to_wgsl",
            "app::tests::gpu_authored_state_survives_growth_without_readback",
            "app::tests::active_count_is_derived_on_the_gpu_and_clamped",
            "app::tests::growth_while_a_frame_is_in_flight_preserves_its_writes",
            "app::tests::indirect_dispatch_rejects_wrong_buffers_and_bindings",
            "app::tests::gpu_active_count_matches_cpu_reference_at_boundaries",
            "app::tests::indirect_arguments_are_clamped_to_budget_and_allocation",
            "app::tests::graph_rejects_undeclared_classification_edges",
            "app::tests::count_bindings_reject_empty_and_mismatched_buffers",
            "app::tests::compaction_keeps_the_prefix_and_never_integrates_retired_particles",
            "app::tests::compaction_does_not_depend_on_the_order_particles_are_released",
            "app::tests::an_empty_pool_has_nothing_to_compact",
            "app::tests::compaction_at_one_past_65536_slots_matches_the_reference",
            "app::tests::the_graph_refuses_a_compaction_pass_without_its_declared_edge",
        ],
    ),
    (
        "map-chain",
        &[
            "app::tests::gpu_chains_match_the_host_reference_on_every_size",
            "app::tests::gpu_chains_match_the_host_reference_on_a_large_input",
            "app::tests::the_resident_chain_uploads_once_and_reads_back_once",
            "app::tests::gpu_calls_of_broadcasts_have_no_length_and_empty_inputs_are_empty",
        ],
    ),
    (
        "texture-threshold",
        &["tests::twin_matches_reference_and_gpu"],
    ),
    (
        "hash-rng",
        &["app::tests::reference_twin_and_gpu_are_bit_equal"],
    ),
    (
        "polar-transform",
        &["app::tests::polar_and_rotation_match_the_host_on_the_gpu_and_the_twin"],
    ),
    (
        "buffer-handoff",
        &[
            "tests::gpu_histogram_consumed_in_place_has_zero_mismatches",
            "tests::one_changed_bin_gives_exactly_one_mismatch",
            "tests::unwaited_submit_then_the_compare_pass_has_zero_mismatches",
            "tests::unwaited_submit_with_one_changed_bin_gives_exactly_one_mismatch",
        ],
    ),
    (
        "mc-histogram",
        &["app::tests::gpu_matches_the_twin_and_the_host_reference"],
    ),
    (
        "field-relax",
        &[
            "app::tests::gpu_matches_the_twin_and_the_references",
            "app::tests::gpu_matches_the_references_at_the_small_size",
        ],
    ),
    ("cell-bins", &["app::tests::gpu_matches_the_host_reference"]),
    (
        "walker-steer",
        &[
            "app::tests::gpu_matches_the_host_on_the_small_grid",
            "app::tests::gpu_matches_the_host_on_the_medium_grid",
        ],
    ),
    (
        "walker-transport",
        &[
            "app::tests::gpu_matches_the_host_on_the_small_grid",
            "app::tests::gpu_matches_the_host_on_the_medium_grid",
            "app::tests::gpu_a_crowd_above_the_walker_count_is_the_rule_off",
        ],
    ),
    (
        "walker-life",
        &[
            "app::tests::gpu_matches_the_host_on_the_small_grid",
            "app::tests::gpu_matches_the_host_on_the_medium_grid",
        ],
    ),
    ("event-drain", &["app::tests::gpu_matches_the_host"]),
    (
        "task-basics",
        &[
            "app::tests::gpu_a_fresh_one_d_task_is_one_submission",
            "app::tests::gpu_a_recycled_output_is_still_one_submission",
            "app::tests::gpu_a_two_d_task_is_one_submission",
            "app::tests::gpu_an_in_place_task_is_one_submission",
            "app::tests::gpu_an_empty_capture_on_a_recycled_buffer_makes_its_own_submission",
            "app::tests::gpu_a_whole_run_is_one_submission_per_stage",
            "app::tests::gpu_matches_the_host",
        ],
    ),
    (
        "batch-basics",
        &[
            "app::tests::gpu_a_frame_is_one_submission_at_submit_and_none_before",
            "app::tests::gpu_a_dropped_frame_is_not_submitted_and_the_next_one_is_exact",
            "app::tests::gpu_and_the_cpu_twin_agree_on_every_output",
            "gpu_frame_n_is_pending_when_n_plus_1_is_submitted",
            "gpu_dropped_frames_and_claims",
            "gpu_overlapping_frames_are_exact",
            "gpu_a_ticket_resolves_in_a_bounded_number_of_polls",
            "gpu_a_ticket_polled_on_another_thread_resolves_too",
            "gpu_encode_then_submitted_equals_submit_and_the_twin",
            "gpu_encode_works_for_each_frame_in_flight",
            "gpu_the_three_refusals_are_not_submitted",
            "gpu_a_refused_set_holds",
            "gpu_a_refused_set_is_not_mixed_with_the_next_one",
            "gpu_a_refused_write_is_lost",
            "gpu_a_dropped_encoded_retires_its_staging_and_the_next_frame_is_exact",
            "app::tests::gpu_matches_the_host",
        ],
    ),
    (
        "subgroup-reduce",
        &["tests::the_selected_body_runs_on_the_adapter"],
    ),
];

/// The ignored tests pinned for `package`, or `None` when it is not an example with any.
fn pinned_tests(package: &str) -> Option<&'static [&'static str]> {
    EXAMPLE_GPU_TESTS
        .iter()
        .find(|(name, _)| *name == package)
        .map(|(_, tests)| *tests)
}

/// The package a `cargo test` call names with `-p`.
fn package_of(call: &[String]) -> Option<&str> {
    let at = call.iter().position(|arg| arg == "-p")?;
    call.get(at + 1).map(String::as_str)
}

/// The guard on the captured output of a `cargo test` call of an example: when the call names an example package
/// that has pinned tests, every one of them must be reported `ok` (a missing name means it was renamed, deleted,
/// filtered out or left ignored, and a run that executed none of them reports all of them missing). Any other call
/// passes.
pub(crate) fn check_pinned_example_tests(call: &[String], stdout: &str) -> Result<(), String> {
    let Some(package) = package_of(call) else {
        return Ok(());
    };
    let Some(pinned) = pinned_tests(package) else {
        return Ok(());
    };
    let missing = tests_not_ok(stdout, pinned);
    if missing.is_empty() {
        return Ok(());
    }
    Err(format!(
        "cargo {}: {} of {} pinned ignored test(s) of `{package}` did not run and pass: `{}` (renamed, deleted, not ignored any more, or the run executed none of them; EXAMPLE_GPU_TESTS in xtask/src/examples_gpu.rs names them)",
        call.join(" "),
        missing.len(),
        pinned.len(),
        missing.join("`, `")
    ))
}

/// The cargo call that runs only the ignored tests of one example package.
fn example_gpu_call(package: &str) -> Vec<String> {
    ["test", "--locked", "-p", package, "--", "--ignored"]
        .iter()
        .map(|arg| (*arg).to_owned())
        .collect()
}

/// Fails unless `adapter` is the adapter line the Gpu suite prints (`adapter: <name> (<type>, <backend>, driver ..)`),
/// with a name. Returns it, so the caller prints exactly what was checked.
fn require_adapter_line(adapter: &str) -> Result<&str, String> {
    let line = adapter
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("adapter: "));
    match line {
        Some(line) if !line["adapter: ".len()..].trim_start().starts_with('(') => Ok(line),
        Some(_) => Err("the adapter line names no adapter".to_owned()),
        None => Err("no `adapter:` line: the run does not say which adapter it used".to_owned()),
    }
}

/// The option of `check-examples --gpu`: how many test executables run at a time.
const GPU_JOBS_OPTION: &str = "--gpu-jobs";

/// The build-only option of `check-examples --gpu`.
const PREBUILD_OPTION: &str = "--prebuild";

/// What `check-examples --gpu` was asked to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GpuMode {
    /// Run the tests, `width` executables at a time.
    Run { width: usize },
    /// Only build (and list) the test executables the run needs, with the run's own calls and environment.
    Prebuild,
}

/// The arguments after `--gpu`: nothing, `--gpu-jobs <n>` (`n` at least 1) or `--prebuild`.
pub(crate) fn parse_gpu_args(args: &[String]) -> Result<GpuMode, String> {
    match args {
        [] => Ok(GpuMode::Run {
            width: DEFAULT_EXAMPLE_JOBS,
        }),
        [option] if option == PREBUILD_OPTION => Ok(GpuMode::Prebuild),
        [option, value] if option == GPU_JOBS_OPTION => match value.parse::<usize>() {
            Ok(width) if width >= 1 => Ok(GpuMode::Run { width }),
            _ => Err(format!(
                "{GPU_JOBS_OPTION} takes a whole number of at least 1, not `{value}`"
            )),
        },
        _ => Err(format!(
            "check-examples --gpu takes no argument, {GPU_JOBS_OPTION} <n> or {PREBUILD_OPTION}"
        )),
    }
}

/// `cargo xtask check-examples --gpu [--gpu-jobs <n> | --prebuild]`: names the adapter (the first Vulkan one, which the
/// examples' tests open), then runs the ignored tests of every example and requires each pinned name to pass
/// ([`check_pinned_example_tests`]). With `--gpu-jobs 1` that is one cargo call per package; otherwise the prebuilt test
/// executables of the examples run `width` at a time ([`run_example_executables`]). `--prebuild` only makes those
/// executables ([`build_example_executables`]), so a gate can do it before it takes the adapter. It runs nothing else of
/// `check-examples` (no example binary, no artifact), so the coordinator can run it once per adapter. Needs slangc, the
/// reflection helper and (but for `--prebuild`) an adapter.
pub(crate) fn check_examples_gpu(mode: GpuMode) -> Result<(), String> {
    let root = workspace_root()?;
    crate::process::require_command("cargo")?;
    crate::process::require_command("slangc")?;
    prepare_helper(&root, None)?;
    let width = match mode {
        GpuMode::Run { width } => width,
        GpuMode::Prebuild => {
            let metadata = read_metadata(&root)?;
            let jobs = example_jobs(&metadata, &build_example_executables(&root, false)?)?;
            println!(
                "check-examples --gpu --prebuild: {} test executable(s) of {} example packages are built",
                jobs.len(),
                packages().count()
            );
            return Ok(());
        }
    };
    let adapter = name_adapter(&root)?;
    let adapter = require_adapter_line(&adapter)?;
    println!("{adapter}");
    let ran = if width == 1 {
        run_example_gpu_tests(&root)?
    } else {
        run_example_executables(&root, width)?
    };
    println!(
        "check-examples --gpu passed: {ran} pinned ignored tests of {} example packages ran and passed on the {adapter}",
        packages().count()
    );
    Ok(())
}

/// The packages `--gpu` runs: every example with pinned tests, in the order of [`EXAMPLE_GPU_TESTS`].
fn packages() -> impl Iterator<Item = &'static str> {
    EXAMPLE_GPU_TESTS.iter().map(|(name, _)| *name)
}

fn run_example_gpu_tests(root: &Path) -> Result<u64, String> {
    let mut ran = 0;
    for package in packages() {
        let call = example_gpu_call(package);
        println!("-- cargo {}", call.join(" "));
        let (totals, stdout) = run_cargo_test_captured(root, &call)?;
        check_ignored_only_run(&call, &stdout, totals.passed)?;
        ran += totals.passed;
    }
    Ok(ran)
}

/// One test executable of an example package, with the package's environment.
#[derive(Debug)]
struct ExampleJob {
    package: &'static str,
    environment: Package,
    executable: TestExecutable,
}

impl ExampleJob {
    /// The job the runner prints it as.
    fn job(&self) -> Job {
        Job::Executable(self.executable.clone())
    }
}

/// The groups of packages that one `cargo test --no-run` call lists and builds: every package whose features agree in
/// one call, then each of [`OWN_BUILD`] alone.
fn build_groups() -> Vec<Vec<&'static str>> {
    let mut groups = vec![
        packages()
            .filter(|package| !OWN_BUILD.contains(package))
            .collect::<Vec<_>>(),
    ];
    groups.extend(
        packages()
            .filter(|package| OWN_BUILD.contains(package))
            .map(|package| vec![package]),
    );
    groups
}

/// The `cargo test --no-run` call that lists (and, run beforehand, builds) the test executables of `group`.
fn example_build_call(group: &[&str]) -> Vec<String> {
    ["test", "--locked", "--no-run", "--message-format=json"]
        .into_iter()
        .map(str::to_owned)
        .chain(
            group
                .iter()
                .flat_map(|package| ["-p".to_owned(), (*package).to_owned()]),
        )
        .collect()
}

/// Runs the call of every group of [`build_groups`] and returns their standard output together, for [`example_jobs`].
/// With `refuse_build` (the run) a compilation is an error; without it (`--prebuild`) the calls build.
fn build_example_executables(root: &Path, refuse_build: bool) -> Result<String, String> {
    let mut output = String::new();
    for group in build_groups() {
        let call = example_build_call(&group);
        println!("-- cargo {}", call.join(" "));
        output.push_str(&no_run_build(root, &call, refuse_build)?);
        output.push('\n');
    }
    Ok(output)
}

/// The jobs of a run in print order (the packages in the table's order, each one's executables in cargo's order), from
/// the output of `cargo metadata` and of [`example_build_call`]. A package without a test executable is an error.
fn example_jobs(metadata: &str, build_output: &str) -> Result<Vec<ExampleJob>, String> {
    let mut jobs = Vec::new();
    for package in packages() {
        let environment = package_environment(metadata, package)?;
        let executables = test_executables(build_output, &environment.manifest_dir)
            .map_err(|error| format!("{package}: {error}"))?;
        jobs.extend(executables.into_iter().map(|executable| ExampleJob {
            package,
            environment: environment.clone(),
            executable,
        }));
    }
    Ok(jobs)
}

/// The order in which the jobs start: the packages with the most pinned tests first (the time of a package follows its
/// tests), so that the long ones do not start last; a permutation of the job indices. Printing stays in job order.
fn example_start_order(jobs: &[ExampleJob]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    order.sort_by_key(|&index| Reverse(pinned_tests(jobs[index].package).map_or(0, <[_]>::len)));
    order
}

/// Judges the finished jobs: none may have failed or been left unstarted, and the standard output of each package's
/// executables together must pass [`check_ignored_only_run`] as the output of the package's one cargo call did. Returns
/// how many pinned tests passed.
fn judge_example_jobs(jobs: &[ExampleJob], results: &[Option<JobOutput>]) -> Result<u64, String> {
    let all: Vec<Job> = jobs.iter().map(ExampleJob::job).collect();
    judge_named("the example packages", &all, results)?;
    let mut ran = 0;
    for package in packages() {
        let stdout: String = jobs
            .iter()
            .zip(results)
            .filter(|(job, _)| job.package == package)
            .filter_map(|(_, result)| result.as_ref())
            .map(|output| format!("{}\n", output.stdout))
            .collect();
        let totals = sum_test_results(&stdout);
        let call = example_gpu_call(package);
        require_tests_ran(&call, &totals)?;
        check_ignored_only_run(&call, &stdout, totals.passed)?;
        ran += totals.passed;
    }
    Ok(ran)
}

/// The ignored tests of every example through its prebuilt test executables, `width` at a time ([`example_jobs`]; the
/// build must exist already, one that happens here is an error), judged by [`judge_example_jobs`].
fn run_example_executables(root: &Path, width: usize) -> Result<u64, String> {
    let metadata = read_metadata(root)?;
    let jobs = example_jobs(&metadata, &build_example_executables(root, true)?)?;
    println!(
        "   {} test executable(s) of {} example packages, {width} at a time",
        jobs.len(),
        packages().count()
    );
    let results = run_printed(
        &jobs,
        &example_start_order(&jobs),
        width,
        &ExampleJob::job,
        &|job, cancel| {
            run_executable(
                &job.executable,
                &job.environment,
                &["--ignored"],
                EXAMPLE_JOB_TIME_LIMIT,
                cancel,
            )
        },
    );
    judge_example_jobs(&jobs, &results)
}

/// The guard on a `--ignored` run of one example: every pinned name passed ([`check_pinned_example_tests`]) and the run
/// passed exactly as many tests as are pinned, so an ignored test that is not in the table fails here too (the count
/// is the one the harness reported, not the table's).
fn check_ignored_only_run(call: &[String], stdout: &str, passed: u64) -> Result<(), String> {
    check_pinned_example_tests(call, stdout)?;
    let pinned = package_of(call)
        .and_then(pinned_tests)
        .map_or(0, <[_]>::len) as u64;
    if passed != pinned {
        return Err(format!(
            "cargo {}: {passed} ignored test(s) passed, {pinned} are pinned (an ignored test was added or removed without changing EXAMPLE_GPU_TESTS in xtask/src/examples_gpu.rs)",
            call.join(" ")
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::EXAMPLES;
    use std::fs;

    /// The examples of `EXAMPLES` with no ignored test: the shape bench measures and checks nothing on its own.
    const NO_GPU_TEST_EXAMPLES: [&str; 1] = ["shape-bench"];

    /// The strict proof of the subgroups capability: not one of `EXAMPLES` (it runs under profiles of its own), but its
    /// ignored test runs on the adapter like the others and is pinned the same way.
    const SUBGROUP_REDUCE: &str = "subgroup-reduce";

    fn owned(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| (*arg).to_owned()).collect()
    }

    /// The `cargo test` output of `package` in which every pinned test passed.
    fn passing_output(package: &str) -> String {
        let mut out = String::from("running tests\n");
        for name in pinned_tests(package).expect("pinned") {
            out.push_str(&format!("test {name} ... ok\n"));
        }
        out.push_str(
            "test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n",
        );
        out
    }

    /// The `cargo metadata` text of every pinned package, one manifest each.
    fn fixture_metadata() -> String {
        let entries: Vec<String> = packages()
            .map(|package| {
                format!(
                    "{{\"name\":\"{package}\",\"version\":\"0.1.0\",\"manifest_path\":\"C:/w/examples/{package}/Cargo.toml\"}}"
                )
            })
            .collect();
        format!("{{\"packages\":[{}]}}", entries.join(","))
    }

    /// One `compiler-artifact` line of a test build of `package`.
    fn test_build(package: &str, kind: &str, name: &str) -> String {
        format!(
            "{{\"reason\":\"compiler-artifact\",\"manifest_path\":\"C:/w/examples/{package}/Cargo.toml\",\"target\":{{\"kind\":[\"{kind}\"],\"name\":\"{name}\",\"src_path\":\"C:/w/examples/{package}/src/{name}.rs\"}},\"profile\":{{\"test\":true}},\"executable\":\"C:/w/target/debug/deps/{name}-{package}.exe\"}}"
        )
    }

    /// The build output in which every package has its unit tests and `batch-basics` also an integration target.
    fn fixture_build() -> String {
        let mut lines: Vec<String> = packages()
            .map(|package| test_build(package, "bin", package))
            .collect();
        lines.push(test_build("batch-basics", "test", "encode"));
        lines.join("\n")
    }

    /// A finished job whose standard output reports `names` ok and `passed` passed tests.
    fn finished_with(names: &[&str], passed: usize) -> Option<JobOutput> {
        let mut stdout = String::new();
        for name in names {
            stdout.push_str(&format!("test {name} ... ok\n"));
        }
        stdout.push_str(&format!(
            "test result: ok. {passed} passed; 0 failed; 1 ignored; 0 measured; 0 filtered out\n"
        ));
        Some(JobOutput {
            success: true,
            status: "exit 0".to_owned(),
            stdout,
            stderr: String::new(),
        })
    }

    /// Results in which every job of `jobs` passed its package's pinned tests (a package's tests split over its jobs).
    fn passing_results(jobs: &[ExampleJob]) -> Vec<Option<JobOutput>> {
        jobs.iter()
            .map(|job| {
                let pinned = pinned_tests(job.package).unwrap();
                let split = jobs.iter().filter(|other| other.package == job.package);
                let (position, count) = (
                    split
                        .clone()
                        .position(|other| other.executable == job.executable)
                        .unwrap(),
                    split.count(),
                );
                let share: Vec<&str> = pinned
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(index, _)| index % count == position)
                    .map(|(_, name)| name)
                    .collect();
                finished_with(&share, share.len())
            })
            .collect()
    }

    #[test]
    fn the_job_list_is_every_package_in_table_order_with_its_executables() {
        let jobs = example_jobs(&fixture_metadata(), &fixture_build()).unwrap();
        let names: Vec<&str> = jobs.iter().map(|job| job.package).collect();
        let mut expected: Vec<&str> = packages().collect();
        let at = expected
            .iter()
            .position(|name| *name == "batch-basics")
            .unwrap();
        expected.insert(at, "batch-basics");
        assert_eq!(
            names, expected,
            "one job per executable, packages in table order"
        );
        let batch: Vec<&str> = jobs
            .iter()
            .filter(|job| job.package == "batch-basics")
            .map(|job| job.executable.name.as_str())
            .collect();
        assert_eq!(
            batch,
            ["batch-basics", "encode"],
            "unit tests, then integration targets"
        );
        assert!(
            jobs[0]
                .environment
                .environment
                .contains(&("CARGO_PKG_NAME", "atomic-counter".to_owned()))
        );
        // A package with no test executable (a stale or missing build) is named.
        let without: String = fixture_build()
            .lines()
            .filter(|line| !line.contains("examples/hash-rng/"))
            .collect::<Vec<_>>()
            .join("\n");
        let error = example_jobs(&fixture_metadata(), &without).unwrap_err();
        assert!(error.starts_with("hash-rng: "), "{error}");
        let calls: Vec<Vec<String>> = build_groups()
            .iter()
            .map(|group| example_build_call(group))
            .collect();
        assert!(
            calls
                .iter()
                .all(|call| call.contains(&"--no-run".to_owned()))
        );
    }

    #[test]
    fn jobs_start_with_the_packages_that_pin_the_most_tests() {
        let jobs = example_jobs(&fixture_metadata(), &fixture_build()).unwrap();
        let order = example_start_order(&jobs);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..jobs.len()).collect::<Vec<_>>(), "a permutation");
        let pinned = |index: usize| pinned_tests(jobs[order[index]].package).unwrap().len();
        assert!((1..order.len()).all(|index| pinned(index - 1) >= pinned(index)));
    }

    #[test]
    fn finished_jobs_are_judged_by_package_with_the_pinned_guards() {
        let jobs = example_jobs(&fixture_metadata(), &fixture_build()).unwrap();
        let total: usize = packages()
            .map(|name| pinned_tests(name).unwrap().len())
            .sum();
        let results = passing_results(&jobs);
        assert_eq!(judge_example_jobs(&jobs, &results), Ok(total as u64));
        // A pinned name that is not reported ok fails and names its package, whichever job should have run it.
        let mut renamed = results.clone();
        let output = renamed[3].as_mut().unwrap();
        output.stdout = output.stdout.replacen("test app::", "test other::", 1);
        let error = judge_example_jobs(&jobs, &renamed).unwrap_err();
        assert!(error.contains("did not run and pass"), "{error}");
        // An extra passing test is a count mismatch.
        let mut extra = results.clone();
        let output = extra[0].as_mut().unwrap();
        output.stdout = output.stdout.replace("1 passed", "2 passed");
        let error = judge_example_jobs(&jobs, &extra).unwrap_err();
        assert!(
            error.contains("2 ignored test(s) passed, 1 are pinned"),
            "{error}"
        );
    }

    #[test]
    fn a_failed_job_fails_the_run_and_names_the_executable_and_the_jobs_never_started() {
        let jobs = example_jobs(&fixture_metadata(), &fixture_build()).unwrap();
        let mut results = passing_results(&jobs);
        let last = results.len() - 1;
        results[1].as_mut().unwrap().success = false;
        results[1].as_mut().unwrap().status = "exit 101".to_owned();
        results[last] = None;
        let error = judge_example_jobs(&jobs, &results).unwrap_err();
        assert!(error.contains("the example packages failed"), "{error}");
        assert!(error.contains("exit 101"), "{error}");
        assert!(error.contains("1 job(s) were not started"), "{error}");
        assert!(
            error.contains(&jobs[1].executable.path.display().to_string()),
            "{error}"
        );
    }

    #[test]
    fn the_gpu_jobs_option_is_a_whole_number_of_at_least_one() {
        let run = |width| Ok(GpuMode::Run { width });
        assert_eq!(parse_gpu_args(&[]), run(DEFAULT_EXAMPLE_JOBS));
        assert_eq!(parse_gpu_args(&owned(&["--gpu-jobs", "1"])), run(1));
        assert_eq!(parse_gpu_args(&owned(&["--gpu-jobs", "5"])), run(5));
        assert_eq!(
            parse_gpu_args(&owned(&["--prebuild"])),
            Ok(GpuMode::Prebuild)
        );
        for bad in [
            &["--gpu-jobs", "0"][..],
            &["--gpu-jobs", "x"],
            &["--gpu-jobs"],
            &["--jobs", "2"],
            &["--gpu-jobs", "2", "3"],
            &["--prebuild", "--gpu-jobs", "2"],
        ] {
            assert!(parse_gpu_args(&owned(bad)).is_err(), "{bad:?}");
        }
    }

    /// The `features = [..]` of the dependency `name` in a manifest (`Some("")` when it has none, `None` when the
    /// manifest does not depend on it).
    fn dependency_features(manifest: &str, name: &str) -> Option<String> {
        let line = manifest.lines().find(|line| {
            line.strip_prefix(name)
                .is_some_and(|rest| rest.starts_with(['.', ' ', '=']))
        })?;
        Some(
            line.split_once("features")
                .and_then(|(_, rest)| rest.split_once('['))
                .and_then(|(_, rest)| rest.split_once(']'))
                .map(|(features, _)| features.split_whitespace().collect::<String>())
                .unwrap_or_default(),
        )
    }

    #[test]
    fn the_packages_of_one_build_call_declare_the_same_features_on_the_gust_crates() {
        let root = workspace_root().unwrap();
        let features = |package: &str, name: &str| {
            let manifest =
                fs::read_to_string(root.join("examples").join(package).join("Cargo.toml")).unwrap();
            dependency_features(&manifest, name)
        };
        for group in build_groups() {
            for name in ["gust", "gust-wgpu", "gust-algo"] {
                let declared: Vec<(&str, String)> = group
                    .iter()
                    .filter_map(|package| Some((*package, features(package, name)?)))
                    .collect();
                assert!(
                    declared.windows(2).all(|pair| pair[0].1 == pair[1].1),
                    "{name}: the packages of one cargo call (features are unified across them) declare different features: {declared:?}; give the odd one an entry of OWN_BUILD"
                );
            }
        }
        // The groups together are every pinned package, once, and an own build is pinned.
        let mut all: Vec<&str> = build_groups().concat();
        all.sort_unstable();
        let mut pinned: Vec<&str> = packages().collect();
        pinned.sort_unstable();
        assert_eq!(all, pinned);
        for own in OWN_BUILD {
            assert!(pinned_tests(own).is_some(), "{own} is not pinned");
        }
    }

    #[test]
    fn the_dependency_features_are_read_from_the_manifest_line() {
        let manifest = "[dependencies]\ngust.workspace = true\ngust-wgpu = { workspace = true, features = [\"slang-runtime\"] }\ngust-algo = { path = \"x\", features = [ \"a\", \"b\" ] }\n";
        assert_eq!(dependency_features(manifest, "gust"), Some(String::new()));
        assert_eq!(
            dependency_features(manifest, "gust-wgpu"),
            Some("\"slang-runtime\"".to_owned())
        );
        assert_eq!(
            dependency_features(manifest, "gust-algo"),
            Some("\"a\",\"b\"".to_owned())
        );
        assert_eq!(dependency_features(manifest, "gust-macros"), None);
    }

    #[test]
    fn no_example_library_has_a_doc_test() {
        // The step runs test executables only, never rustdoc (which would compile under the lock), so a doc test in
        // the library of an example (a fenced block in a doc comment) would not be run by it.
        let root = workspace_root().unwrap();
        let mut libraries = 0;
        for package in packages() {
            let source = root.join("examples").join(package).join("src");
            if !source.join("lib.rs").is_file() {
                continue;
            }
            libraries += 1;
            for entry in fs::read_dir(&source).unwrap() {
                let path = entry.unwrap().path();
                if path.extension().is_none_or(|extension| extension != "rs") {
                    continue;
                }
                let text = fs::read_to_string(&path).unwrap();
                assert!(
                    !text.contains("```"),
                    "{} has a fenced block: the doc tests of an example library are not run by check-examples --gpu",
                    path.display()
                );
            }
        }
        assert!(libraries > 0, "no example library was found to scan");
    }

    #[test]
    fn a_compilation_in_the_listing_is_seen_through_colours() {
        use crate::gpu_jobs::compiling_line;
        assert_eq!(compiling_line("   Finished `test` profile in 0.1s\n"), None);
        assert_eq!(
            compiling_line(
                "\u{1b}[1m\u{1b}[92m   Compiling\u{1b}[0m vector-add v0.1.0\n   Finished x\n"
            ),
            Some("Compiling vector-add v0.1.0".to_owned())
        );
    }

    #[test]
    fn an_extra_unpinned_passing_test_fails_the_ignored_only_run() {
        let call = example_gpu_call("vector-add");
        let stdout = format!(
            "{}test app::tests::an_unpinned_one ... ok
",
            passing_output("vector-add")
        );
        assert_eq!(
            check_ignored_only_run(&call, &passing_output("vector-add"), 4),
            Ok(())
        );
        let error = check_ignored_only_run(&call, &stdout, 5).unwrap_err();
        assert!(
            error.contains("5 ignored test(s) passed, 4 are pinned"),
            "{error}"
        );
    }

    #[test]
    fn a_run_with_every_pinned_test_ok_passes() {
        let call = example_gpu_call("vector-add");
        assert_eq!(
            check_pinned_example_tests(&call, &passing_output("vector-add")),
            Ok(())
        );
    }

    #[test]
    fn a_missing_pinned_name_fails_and_is_named() {
        let call = owned(&["test", "-p", "vector-add", "--", "--include-ignored"]);
        let stdout = passing_output("vector-add")
            .replace("rust_to_slang_executes_on_headless_wgpu", "renamed_test");
        let error = check_pinned_example_tests(&call, &stdout).unwrap_err();
        assert!(
            error.contains("1 of 4 pinned ignored test(s) of `vector-add`"),
            "{error}"
        );
        assert!(
            error.contains("app::tests::rust_to_slang_executes_on_headless_wgpu"),
            "{error}"
        );
    }

    #[test]
    fn a_pinned_test_that_failed_or_stayed_ignored_fails() {
        let call = example_gpu_call("hash-rng");
        let name = pinned_tests("hash-rng").unwrap()[0];
        for outcome in ["FAILED", "ignored"] {
            let stdout = format!("test {name} ... {outcome}\n");
            assert!(
                check_pinned_example_tests(&call, &stdout).is_err(),
                "{outcome}"
            );
        }
    }

    #[test]
    fn a_run_that_executed_none_of_the_pinned_tests_fails() {
        let call = example_gpu_call("walker-life");
        let stdout = "running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 13 filtered out\n";
        let error = check_pinned_example_tests(&call, stdout).unwrap_err();
        assert!(
            error.contains("2 of 2 pinned ignored test(s) of `walker-life`"),
            "{error}"
        );
    }

    #[test]
    fn a_call_that_is_no_pinned_example_passes_unchecked() {
        let stdout = "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n";
        for call in [
            owned(&["test", "-p", "gust-algo"]),
            owned(&["test", "--locked"]),
            owned(&["test", "-p", "shape-bench", "--", "--include-ignored"]),
        ] {
            assert_eq!(
                check_pinned_example_tests(&call, stdout),
                Ok(()),
                "{call:?}"
            );
        }
    }

    #[test]
    fn a_missing_adapter_line_fails() {
        assert!(
            require_adapter_line("")
                .unwrap_err()
                .contains("no `adapter:` line")
        );
        assert!(
            require_adapter_line("running 3 tests\ntest result: ok.\n")
                .unwrap_err()
                .contains("no `adapter:` line")
        );
        assert!(require_adapter_line("adapter: (discrete, vulkan, driver 1)").is_err());
        let line = "adapter: Some GPU (discrete, vulkan, driver 1.2)";
        assert_eq!(require_adapter_line(line), Ok(line));
    }

    #[test]
    fn the_gpu_call_runs_only_the_ignored_tests_of_one_package() {
        assert_eq!(
            example_gpu_call("map-chain"),
            owned(&["test", "--locked", "-p", "map-chain", "--", "--ignored"])
        );
    }

    #[test]
    fn every_example_is_pinned_or_has_no_ignored_test_and_nothing_else_is_pinned() {
        for example in EXAMPLES {
            let pinned = pinned_tests(example).is_some();
            let exempt = NO_GPU_TEST_EXAMPLES.contains(example);
            assert!(
                pinned != exempt,
                "{example}: pinned {pinned}, exempt {exempt}"
            );
        }
        for package in packages() {
            assert!(
                EXAMPLES.contains(&package) || package == SUBGROUP_REDUCE,
                "{package} is pinned but is no example"
            );
        }
        assert!(pinned_tests(SUBGROUP_REDUCE).is_some());
        let mut seen = Vec::new();
        for package in packages() {
            assert!(!seen.contains(&package), "{package} is pinned twice");
            seen.push(package);
            let tests = pinned_tests(package).unwrap();
            assert!(!tests.is_empty(), "{package} has no pinned test");
            let mut sorted = tests.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), tests.len(), "{package} pins a name twice");
        }
    }

    /// What follows `fn ` on a line that declares a function, through `pub`, `async`, `const` and `unsafe`.
    fn fn_after_qualifiers(line: &str) -> Option<&str> {
        let mut rest = line.trim_start();
        while let Some(stripped) = ["pub ", "async ", "const ", "unsafe "]
            .iter()
            .find_map(|word| rest.strip_prefix(word))
        {
            rest = stripped.trim_start();
        }
        rest.strip_prefix("fn ")
    }

    /// The names of the `#[ignore]`d tests of the sources of one example package: the `fn` that follows each attribute.
    fn ignored_functions(root: &Path, package: &str) -> Vec<String> {
        let mut names = Vec::new();
        let mut files = Vec::new();
        // The sources of the binary and, where an example has them, its integration tests.
        for sub_dir in ["src", "tests"] {
            let dir = root.join("examples").join(package).join(sub_dir);
            // `src` must exist, so that a wrong package path cannot pass silently; `tests` is optional.
            let entries = match fs::read_dir(&dir) {
                Ok(entries) => entries,
                Err(_) if sub_dir == "tests" => continue,
                Err(error) => panic!("{}: {error}", dir.display()),
            };
            files.extend(
                entries
                    .map(|entry| entry.unwrap().path())
                    .filter(|path| path.extension().is_some_and(|ext| ext == "rs")),
            );
        }
        files.sort();
        for file in files {
            let text = fs::read_to_string(&file).unwrap();
            let mut lines = text.lines();
            while let Some(line) = lines.next() {
                let attribute = line.trim_start();
                if !(attribute.starts_with("#[ignore")
                    || (attribute.starts_with("#[cfg_attr(") && attribute.contains("ignore")))
                {
                    continue;
                }
                for next in lines.by_ref() {
                    let next = next.trim_start();
                    if let Some(rest) = fn_after_qualifiers(next) {
                        let name = rest.split('(').next().unwrap();
                        names.push(name.to_owned());
                        break;
                    }
                }
            }
        }
        names
    }

    #[test]
    fn the_source_scan_sees_qualified_functions() {
        assert_eq!(
            fn_after_qualifiers("    pub async fn a(x: u8) {"),
            Some("a(x: u8) {")
        );
        assert_eq!(fn_after_qualifiers("fn b() {"), Some("b() {"));
        assert_eq!(fn_after_qualifiers("let c = 1;"), None);
    }

    #[test]
    fn the_table_is_exactly_the_ignored_tests_of_the_example_sources() {
        let root = workspace_root().unwrap();
        for package in EXAMPLES.iter().copied().chain([SUBGROUP_REDUCE]) {
            let mut in_source = ignored_functions(&root, package);
            in_source.sort();
            let mut pinned = pinned_tests(package)
                .unwrap_or(&[])
                .iter()
                .map(|name| name.rsplit("::").next().unwrap().to_owned())
                .collect::<Vec<_>>();
            pinned.sort();
            assert_eq!(
                in_source, pinned,
                "{package}: ignored tests in the source and EXAMPLE_GPU_TESTS differ"
            );
        }
    }
}
