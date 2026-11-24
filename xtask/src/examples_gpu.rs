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
    process::run_cargo_test_captured,
    verify::{name_adapter, prepare_helper},
    workspace::workspace_root,
};
use std::path::Path;

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
            "gpu_a_refused_set_holds_when_the_next_frame_sets_the_other_parameter",
            "gpu_a_refused_write_is_lost",
            "gpu_an_unchanged_stage_is_not_re_recorded",
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

/// `cargo xtask check-examples --gpu`: names the adapter (the first Vulkan one, which the examples' tests open), then
/// runs the ignored tests of every example, one cargo call per package, and requires each pinned name to pass
/// ([`check_pinned_example_tests`]). It runs nothing else of `check-examples` (no example binary, no artifact), so the
/// coordinator can run it once per adapter. Needs slangc, the reflection helper and an adapter.
pub(crate) fn check_examples_gpu() -> Result<(), String> {
    let root = workspace_root()?;
    crate::process::require_command("cargo")?;
    crate::process::require_command("slangc")?;
    prepare_helper(&root, None)?;
    let adapter = name_adapter(&root)?;
    let adapter = require_adapter_line(&adapter)?;
    println!("{adapter}");
    let ran = run_example_gpu_tests(&root)?;
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
