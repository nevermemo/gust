use std::{
    env,
    ffi::OsString,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

use crate::{
    precompiled::PRECOMPILED_DIR,
    process::{
        PROFILE_VARIABLE, command_exists, exit_code, print_command, require_command, run_checked,
        run_counted, run_counted_captured, run_direct, run_direct_args, set_workspace_profile,
        without_colors, workspace_command,
    },
    shader::{
        ExportOptions, clear_exports, compare_manifests, doctor_smoke, export_golden_spirv,
        hash_manifest,
    },
    ui::{Expectation, judge, parse_expected},
    validation::{compact_validation_summary, json_string},
    verify::{VerifyMode, prepare_helper, verify},
    workspace::{
        EXAMPLES, NO_EXPORT_EXAMPLES, cargo_target_dir, changed_paths, workspace_root,
        workspace_test_args,
    },
};

/// The cargo calls of `check-feature reflection` after the native helper is built (the
/// Reflect suite). Kept as data so a test can require `--locked` on each.
pub(crate) const REFLECTION_TESTS: &[&[&str]] = &[
    &[
        "test",
        "--locked",
        "-p",
        "gust-compute",
        "--test",
        "reflection",
    ],
    &[
        "test",
        "--locked",
        "-p",
        "gust-compute-wgpu",
        "--test",
        "reflection",
    ],
    // The schema-3 output of the helper on the committed probe inputs (mx.t0a): an ignored test of cargo-gust.
    &[
        "test",
        "--locked",
        "-p",
        "cargo-gust",
        "--",
        "--ignored",
        "schema_3_output",
    ],
    // The naga layout gate (mx.n): the WGSL of every fixture kernel against the descriptors' layouts. Needs slangc and
    // the native helper, no adapter; the Slangc suite runs it too (the CI job `slang`).
    &[
        "test",
        "--locked",
        "-p",
        "gust-compute-wgpu",
        "--test",
        "naga_layout",
    ],
];

/// The `gust-wgpu` integration tests of `check-feature gpu-semantics`, by file name under crates/gust-wgpu/tests. Kept
/// as data so a test can pin the list (the scalar coretests of m5.d2 are one of them).
pub(crate) const GPU_SEMANTICS_TESTS: [&str; 8] = [
    "semantics",
    "numeric",
    "option",
    "result",
    "loops",
    "struct_assignment",
    "mismatch",
    "coretests",
];

/// The test modules of `gust-macros` that `check-feature dispatch` runs, as name filters (`par_for_tests` is the module
/// of `tests/par_for.rs`: the `par_for!` expansion, its golden, the export records and GUST-E0719/E0720; `grid_tests`
/// that of `tests/grid.rs`: the multi-dimensional kernels and their golden). Each is its own cargo call, counted.
pub(crate) const DISPATCH_MACRO_MODULES: [&str; 2] = ["par_for_tests::", "grid_tests::"];

/// The name filters of the `gust` library tests of `check-feature dispatch`, each its own counted call (a filter that
/// matches nothing fails by itself): the `dispatch` module (`Extent` and the extent of a call), the CPU buffer sized by
/// the product of the extent, and the CPU invocations of a two- and three-dimensional extent (one test each, by full
/// name, so that a rename fails here).
pub(crate) const DISPATCH_CORE_FILTERS: [&str; 3] = [
    "dispatch::",
    "cpu::tests::a_shared_buffer_holds_the_product_of_the_extent",
    "cpu::tests::invocations_cover_whole_workgroups_in_two_and_three_dimensions",
];

/// The `gust-wgpu` integration tests of `check-feature dispatch`, by file name under crates/gust-wgpu/tests: `par_for!`
/// on the GPU, multi-dimensional grids, concurrent calls and the asynchronous readback.
pub(crate) const DISPATCH_GPU_TESTS: [&str; 4] =
    ["par_for", "grid", "concurrency", "async_readback"];

/// The name filters of the `gust-macros` tests of `check-feature vectors` (libtest ORs them): the vector and matrix
/// types, the product and negation overloads, the goldens and their rejections.
pub(crate) const VECTOR_MACRO_FILTERS: [&str; 8] = [
    "vector",
    "matri",
    "product",
    "negation",
    "unary_minus",
    "star_assign",
    "swizzle",
    "dot_is",
];

/// The goldens of `tests/fixtures` that `check-feature vectors` and `verify` (`check-artifacts`, the Spirv suite,
/// `check-full`) compile to SPIR-V, and the names of the files that must come out of them (`<golden>__<kernel>.spv`,
/// sorted). They are kept apart from `EXPECTED_SPIRV`, which names what the examples write to `generated-wgpu`; a
/// golden that gains a kernel changes its list in the same commit.
pub(crate) const VECTOR_GOLDEN_STEMS: [&str; 2] = ["vectors", "products"];
pub(crate) const VECTOR_GOLDEN_SPIRV: [&str; 2] = ["products__run.spv", "vectors__run.spv"];

/// The name filters of the `gust-macros` tests of `check-feature uniforms`: the uniform parameters, the Std140 layout
/// asserts and the `ConstantBuffer` alias.
pub(crate) const UNIFORM_MACRO_FILTERS: [&str; 3] = ["uniform", "std140", "constant_buffer"];

/// The golden of `check-feature uniforms` and its SPIR-V name, as [`VECTOR_GOLDEN_STEMS`].
pub(crate) const UNIFORM_GOLDEN_STEMS: [&str; 1] = ["uniforms"];
pub(crate) const UNIFORM_GOLDEN_SPIRV: [&str; 1] = ["uniforms__run.spv"];

/// The goldens of M7 (`tests/fixtures`) that `verify` compiles to SPIR-V and validates, with the names of the files
/// that must come out of them, sorted. A golden that holds several kernels is compiled one section at a time
/// (`shader::golden_sections`); an entry that does not carry the golden's name keeps its own
/// (`map__<function>_map_kernel`, `par_for__<hashed module>_run`, see `shader::golden_file_name`). A golden that gains
/// or loses a kernel changes its list in the same commit; the `par_for` list also changes when a lowering change
/// re-hashes a body (the module name is the hash of the lowered Slang). The unit tests of xtask catch both.
pub(crate) const MAP_GOLDEN_STEMS: [&str; 1] = ["map"];
pub(crate) const MAP_GOLDEN_SPIRV: [&str; 5] = [
    "map__add_one_map_kernel.spv",
    "map__clamp_signed_map_kernel.spv",
    "map__even_and_map_kernel.spv",
    "map__mirror_map_kernel.spv",
    "map__saxpy_map_kernel.spv",
];

pub(crate) const GRID_GOLDEN_STEMS: [&str; 1] = ["grid"];
pub(crate) const GRID_GOLDEN_SPIRV: [&str; 3] =
    ["grid__cells.spv", "grid__corner.spv", "grid__volume.spv"];

/// The golden of the textures (mx.t3): one file per kernel, "textures__<kernel>.spv".
pub(crate) const TEXTURES_GOLDEN_STEMS: [&str; 1] = ["textures"];
pub(crate) const TEXTURES_GOLDEN_SPIRV: [&str; 8] = [
    "textures__accumulate.spv",
    "textures__blur.spv",
    "textures__gather.spv",
    "textures__invert.spv",
    "textures__shift.spv",
    "textures__write_floats.spv",
    "textures__write_ints.spv",
    "textures__write_uints.spv",
];

/// The golden of the `pixel` profile (mx.p1a): fragment entries, one file per kernel, "pixel__<kernel>.spv".
pub(crate) const PIXEL_GOLDEN_STEMS: [&str; 1] = ["pixel"];
pub(crate) const PIXEL_GOLDEN_SPIRV: [&str; 3] =
    ["pixel__pack.spv", "pixel__row.spv", "pixel__shade.spv"];

/// The golden of the variants fixture (m8.b1): one section per body, "variant__<kernel>.spv".
pub(crate) const VARIANT_GOLDEN_STEMS: [&str; 1] = ["variant"];
pub(crate) const VARIANT_GOLDEN_SPIRV: [&str; 2] =
    ["variant__double.spv", "variant__double_subgroups.spv"];

/// The golden of the subgroup operations fixture (m8.s): one section per body, "subgroup__<kernel>.spv".
pub(crate) const SUBGROUP_GOLDEN_STEMS: [&str; 1] = ["subgroup"];
pub(crate) const SUBGROUP_GOLDEN_SPIRV: [&str; 12] = [
    "subgroup__count_if.spv",
    "subgroup__count_if_subgroups.spv",
    "subgroup__lanes_f32.spv",
    "subgroup__lanes_f32_subgroups.spv",
    "subgroup__lanes_i32.spv",
    "subgroup__lanes_i32_subgroups.spv",
    "subgroup__lanes_u32.spv",
    "subgroup__lanes_u32_subgroups.spv",
    "subgroup__partial_sums.spv",
    "subgroup__partial_sums_subgroups.spv",
    "subgroup__votes.spv",
    "subgroup__votes_subgroups.spv",
];

/// The golden of the methods fixture (m9.a1): one kernel, "methods__run.spv".
pub(crate) const METHODS_GOLDEN_STEMS: [&str; 1] = ["methods"];
pub(crate) const METHODS_GOLDEN_SPIRV: [&str; 1] = ["methods__run.spv"];

/// The golden of the mutating-methods fixture (m9.a2): one kernel, "methods_mut__run.spv".
pub(crate) const METHODS_MUT_GOLDEN_STEMS: [&str; 1] = ["methods_mut"];
pub(crate) const METHODS_MUT_GOLDEN_SPIRV: [&str; 1] = ["methods_mut__run.spv"];

/// The golden of the enums fixture (m9.e1): one kernel, "enums__run.spv".
pub(crate) const ENUMS_GOLDEN_STEMS: [&str; 1] = ["enums"];
pub(crate) const ENUMS_GOLDEN_SPIRV: [&str; 1] = ["enums__run.spv"];

/// The golden of the value-match fixture (m9.e2): one kernel, "match_value__run.spv".
pub(crate) const MATCH_VALUE_GOLDEN_STEMS: [&str; 1] = ["match_value"];
pub(crate) const MATCH_VALUE_GOLDEN_SPIRV: [&str; 1] = ["match_value__run.spv"];

/// The golden of the tuples fixture (m9.t): one kernel, "tuples__run.spv".
pub(crate) const TUPLES_GOLDEN_STEMS: [&str; 1] = ["tuples"];
pub(crate) const TUPLES_GOLDEN_SPIRV: [&str; 1] = ["tuples__run.spv"];

/// The golden of the arrays fixture (m9.r1): one kernel, "arrays__run.spv".
pub(crate) const ARRAYS_GOLDEN_STEMS: [&str; 1] = ["arrays"];
pub(crate) const ARRAYS_GOLDEN_SPIRV: [&str; 1] = ["arrays__run.spv"];

/// The golden of the arrays-in-storage fixture (m9.r2a): four kernels, sorted by name.
pub(crate) const ARRAYS_STORAGE_GOLDEN_STEMS: [&str; 1] = ["arrays_storage"];
pub(crate) const ARRAYS_STORAGE_GOLDEN_SPIRV: [&str; 4] = [
    "arrays_storage__fields.spv",
    "arrays_storage__orders.spv",
    "arrays_storage__roundtrip.spv",
    "arrays_storage__wide.spv",
];

/// The golden of the closures fixture (m9.c): one kernel, "closures__run.spv".
pub(crate) const CLOSURES_GOLDEN_STEMS: [&str; 1] = ["closures"];
pub(crate) const CLOSURES_GOLDEN_SPIRV: [&str; 1] = ["closures__run.spv"];

/// The golden of the traits fixture (m9.g1): one kernel, "traits__run.spv".
pub(crate) const TRAITS_GOLDEN_STEMS: [&str; 1] = ["traits"];
pub(crate) const TRAITS_GOLDEN_SPIRV: [&str; 1] = ["traits__run.spv"];

pub(crate) const WORKGROUP_GOLDEN_STEMS: [&str; 1] = ["workgroup"];
pub(crate) const WORKGROUP_GOLDEN_SPIRV: [&str; 2] =
    ["workgroup__block_sum.spv", "workgroup__local_ids.spv"];

pub(crate) const PAR_FOR_GOLDEN_STEMS: [&str; 1] = ["par_for"];
pub(crate) const PAR_FOR_GOLDEN_SPIRV: [&str; 10] = [
    "par_for____gust_pf_0ed359173a83349f_run.spv",
    "par_for____gust_pf_58ff63fbf2b09015_run.spv",
    "par_for____gust_pf_610d503f9f8cc550_run.spv",
    "par_for____gust_pf_630275d089043ce5_run.spv",
    "par_for____gust_pf_7e03716763028642_run.spv",
    "par_for____gust_pf_93f80c0c6a554f60_run.spv",
    "par_for____gust_pf_e626553f08c28179_run.spv",
    "par_for____gust_pf_f38e4a1a0a82c685_run.spv",
    "par_for____gust_pf_fd9f3f4a5ddb914a_run.spv",
    "par_for____gust_pf_fee81667ae515719_run.spv",
];

pub(crate) const REDUCE_GOLDEN_STEMS: [&str; 1] = ["reduce"];
pub(crate) const REDUCE_GOLDEN_SPIRV: [&str; 30] = [
    "reduce__max_f32_block.spv",
    "reduce__max_f32_final.spv",
    "reduce__max_f32_pass.spv",
    "reduce__max_i32_block.spv",
    "reduce__max_i32_final.spv",
    "reduce__max_i32_pass.spv",
    "reduce__max_u32_block.spv",
    "reduce__max_u32_final.spv",
    "reduce__max_u32_pass.spv",
    "reduce__min_f32_block.spv",
    "reduce__min_f32_final.spv",
    "reduce__min_f32_pass.spv",
    "reduce__min_i32_block.spv",
    "reduce__min_i32_final.spv",
    "reduce__min_i32_pass.spv",
    "reduce__min_u32_block.spv",
    "reduce__min_u32_final.spv",
    "reduce__min_u32_pass.spv",
    "reduce__sum_f32_block.spv",
    "reduce__sum_f32_final.spv",
    "reduce__sum_f32_pass.spv",
    "reduce__sum_i32_block.spv",
    "reduce__sum_i32_final.spv",
    "reduce__sum_i32_pass.spv",
    "reduce__sum_u32_block.spv",
    "reduce__sum_u32_final.spv",
    "reduce__sum_u32_pass.spv",
    "reduce__sum_wrapping_u32_block.spv",
    "reduce__sum_wrapping_u32_final.spv",
    "reduce__sum_wrapping_u32_pass.spv",
];

pub(crate) const SCAN_GOLDEN_STEMS: [&str; 1] = ["scan"];
pub(crate) const SCAN_GOLDEN_SPIRV: [&str; 11] = [
    "scan__add_back_exclusive_u32.spv",
    "scan__add_back_inclusive_u32.spv",
    "scan__add_back_sums_u32.spv",
    "scan__add_carry_u32.spv",
    "scan__scan_chunks_u32.spv",
    "scan__scan_sums_u32.spv",
    "scan__scan_tiles_exclusive_u32.spv",
    "scan__scan_tiles_inclusive_u32.spv",
    "scan__scan_tiles_sums_u32.spv",
    "scan__scan_tiles_top_u32.spv",
    "scan__scan_top_u32.spv",
];

pub(crate) const COMPACT_GOLDEN_STEMS: [&str; 1] = ["compact"];
pub(crate) const COMPACT_GOLDEN_SPIRV: [&str; 5] = [
    "compact__flags_to_mask_u32.spv",
    "compact__gather_f32.spv",
    "compact__gather_i32.spv",
    "compact__gather_u32.spv",
    "compact__scatter_indices_u32.spv",
];

/// Every golden set `verify` exports and validates after the examples' SPIR-V: the area (the directory under the
/// target directory's `golden-spirv`), the fixture stems and the expected file names.
pub(crate) const GOLDEN_SPIRV_SETS: [(&str, &[&str], &[&str]); 22] = [
    ("vectors", &VECTOR_GOLDEN_STEMS, &VECTOR_GOLDEN_SPIRV),
    ("uniforms", &UNIFORM_GOLDEN_STEMS, &UNIFORM_GOLDEN_SPIRV),
    ("map", &MAP_GOLDEN_STEMS, &MAP_GOLDEN_SPIRV),
    ("grid", &GRID_GOLDEN_STEMS, &GRID_GOLDEN_SPIRV),
    (
        "workgroup",
        &WORKGROUP_GOLDEN_STEMS,
        &WORKGROUP_GOLDEN_SPIRV,
    ),
    ("par_for", &PAR_FOR_GOLDEN_STEMS, &PAR_FOR_GOLDEN_SPIRV),
    ("reduce", &REDUCE_GOLDEN_STEMS, &REDUCE_GOLDEN_SPIRV),
    ("scan", &SCAN_GOLDEN_STEMS, &SCAN_GOLDEN_SPIRV),
    ("compact", &COMPACT_GOLDEN_STEMS, &COMPACT_GOLDEN_SPIRV),
    ("textures", &TEXTURES_GOLDEN_STEMS, &TEXTURES_GOLDEN_SPIRV),
    ("pixel", &PIXEL_GOLDEN_STEMS, &PIXEL_GOLDEN_SPIRV),
    ("variant", &VARIANT_GOLDEN_STEMS, &VARIANT_GOLDEN_SPIRV),
    ("subgroup", &SUBGROUP_GOLDEN_STEMS, &SUBGROUP_GOLDEN_SPIRV),
    ("methods", &METHODS_GOLDEN_STEMS, &METHODS_GOLDEN_SPIRV),
    (
        "methods_mut",
        &METHODS_MUT_GOLDEN_STEMS,
        &METHODS_MUT_GOLDEN_SPIRV,
    ),
    ("enums", &ENUMS_GOLDEN_STEMS, &ENUMS_GOLDEN_SPIRV),
    (
        "match_value",
        &MATCH_VALUE_GOLDEN_STEMS,
        &MATCH_VALUE_GOLDEN_SPIRV,
    ),
    ("tuples", &TUPLES_GOLDEN_STEMS, &TUPLES_GOLDEN_SPIRV),
    ("arrays", &ARRAYS_GOLDEN_STEMS, &ARRAYS_GOLDEN_SPIRV),
    (
        "arrays_storage",
        &ARRAYS_STORAGE_GOLDEN_STEMS,
        &ARRAYS_STORAGE_GOLDEN_SPIRV,
    ),
    ("closures", &CLOSURES_GOLDEN_STEMS, &CLOSURES_GOLDEN_SPIRV),
    ("traits", &TRAITS_GOLDEN_STEMS, &TRAITS_GOLDEN_SPIRV),
];

/// The cargo calls of `cargo xtask check-cpu` (a step of the Unit suite): the non-ignored,
/// GPU-free tests of the examples, which run each kernel's CPU twin against its host reference.
/// Later slices append their examples. Kept as data so a test can require `--locked` on each.
pub(crate) const CPU_TESTS: &[&[&str]] = &[
    &["test", "--locked", "-p", "vector-add"],
    &["test", "--locked", "-p", "polynomial"],
    &["test", "--locked", "-p", "signal-pipeline"],
    &["test", "--locked", "-p", "particle-step"],
    &["test", "--locked", "-p", "typed-pipeline"],
    &["test", "--locked", "-p", "atomic-counter"],
    &["test", "--locked", "-p", "staged-graph"],
    &["test", "--locked", "-p", "component-pool"],
    &["test", "--locked", "-p", "map-chain"],
    &["test", "--locked", "-p", "texture-threshold"],
    &["test", "--locked", "-p", "shape-bench"],
];

/// The tests that `cargo xtask check-cpu` requires to be reported `ok` by name, per package of [`CPU_TESTS`] (same
/// order): the non-ignored twin tests (`twin_*`) and, where the example has one, the guard that its inputs stay inside
/// the derivation domain of its tolerances. The count guard alone passes while a sibling test still runs, so a
/// deleted, renamed or `#[ignore]`d twin test would go unnoticed. The pin tests in `tests/cpu_pins.rs` read the
/// example sources, so a new non-ignored `twin_*` or derivation-domain test must be added here.
pub(crate) const CPU_REQUIRED_TESTS: &[(&str, &[&str])] = &[
    (
        "vector-add",
        &[
            "app::tests::twin_matches_host_reference",
            "app::tests::twin_checks_buffer_aliasing_like_the_gpu_backend",
        ],
    ),
    ("polynomial", &["app::tests::twin_matches_host_reference"]),
    (
        "signal-pipeline",
        &[
            "app::tests::inputs_stay_inside_the_derivation_domain",
            "app::tests::twin_matches_host_reference",
        ],
    ),
    (
        "particle-step",
        &[
            "app::tests::inputs_stay_inside_the_derivation_domain",
            "app::tests::twin_matches_host_references_for_every_kernel",
        ],
    ),
    (
        "typed-pipeline",
        &[
            "app::tests::inputs_stay_inside_the_derivation_domain",
            "app::tests::twin_chain_matches_host_reference",
        ],
    ),
    ("atomic-counter", &["tests::twin_counts_every_invocation"]),
    (
        "staged-graph",
        &[
            "app::tests::inputs_stay_inside_the_derivation_domain",
            "app::tests::twin_chain_matches_host_reference",
        ],
    ),
    (
        "component-pool",
        &[
            "app::tests::the_derivation_domain_holds_for_the_settings_and_the_spawned_particles",
            "app::tests::twin_classify_counts_the_spawned_live_slots",
            "app::tests::twin_compacting_frames_match_host_reference",
            "app::tests::twin_frames_match_host_reference",
            "app::tests::twin_gather_and_scatter_follow_the_compaction_indices",
            "app::tests::twin_mark_live_flags_the_spawned_unretired_slots",
        ],
    ),
    (
        "map-chain",
        &[
            "app::tests::inputs_stay_inside_the_derivation_domain",
            "app::tests::twin_chain_matches_host_reference",
            "app::tests::twin_steps_run_in_order",
        ],
    ),
    ("texture-threshold", &["tests::twin_matches_host_reference"]),
    ("shape-bench", &["tests::twin_matches_host_reference"]),
];

/// The package of a `cargo test --locked -p <package>` call of [`CPU_TESTS`].
pub(crate) fn cpu_test_package<'a>(args: &[&'a str]) -> Option<&'a str> {
    let at = args.iter().position(|arg| *arg == "-p")?;
    args.get(at + 1).copied()
}

/// Judges the captured output of one [`CPU_TESTS`] call: every test of [`CPU_REQUIRED_TESTS`] for `package` must be
/// reported `ok`. A package without an entry is an error too, so a new example cannot be added to `CPU_TESTS` unpinned.
pub(crate) fn check_cpu_output(package: &str, stdout: &str) -> Result<(), String> {
    let Some((_, required)) = CPU_REQUIRED_TESTS.iter().find(|(name, _)| *name == package) else {
        return Err(format!(
            "check-cpu: no required tests are listed for `{package}` in CPU_REQUIRED_TESTS (xtask/src/checks.rs)"
        ));
    };
    let missing = tests_not_ok(stdout, required);
    if missing.is_empty() {
        return Ok(());
    }
    Err(format!(
        "check-cpu: cargo test -p {package}: the required test(s) `{}` did not run and pass (renamed, deleted, ignored or failed?)",
        missing.join("`, `")
    ))
}

/// `cargo xtask check-cpu`: the CPU-twin evidence, needing neither slangc nor a GPU.
pub(crate) fn check_cpu() -> Result<(), String> {
    let root = workspace_root()?;
    // Counted: `cargo test` exits 0 with no test at all, so a package whose tests vanished (or were
    // all marked `#[ignore]`) must fail here instead of passing. The twin tests are also required by name:
    // the count passes while a sibling test of the package still runs.
    for args in CPU_TESTS {
        let (_, stdout) = run_counted_captured(&root, args)?;
        let package = cpu_test_package(args)
            .ok_or_else(|| format!("check-cpu: no `-p <package>` in the call {args:?}"))?;
        check_cpu_output(package, &stdout)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/cpu_pins.rs"]
mod cpu_pins_tests;

pub(crate) fn check_feature(area: Option<&str>) -> Result<(), String> {
    let root = workspace_root()?;
    let area = area.ok_or("check-feature needs an area")?;
    match area {
        "macro" => run_direct(&root, "cargo", &["test", "-p", "gust-compute-macros"]),
        "core" => {
            run_direct(&root, "cargo", &["test", "-p", "gust-compute", "--lib"])?;
            run_direct(
                &root,
                "cargo",
                &["test", "-p", "gust-compute", "--test", "diagnostics"],
            )
        }
        "wgpu" => verify(VerifyMode::Gpu, false),
        "gpu-smoke" => run_direct(
            &root,
            "cargo",
            &["test", "-p", "gust-compute-wgpu", "--lib"],
        ),
        "gpu-semantics" => {
            for test in GPU_SEMANTICS_TESTS {
                run_direct(
                    &root,
                    "cargo",
                    &["test", "-p", "gust-compute-wgpu", "--test", test],
                )?;
            }
            Ok(())
        }
        "gpu-runtime" => run_direct(
            &root,
            "cargo",
            &[
                "test",
                "-p",
                "gust-compute-wgpu",
                "--test",
                "reflection",
                "--test",
                "limits",
                "--test",
                "typed",
                "--test",
                "graph",
                "--test",
                "subgroup",
                "--test",
                "capabilities",
            ],
        ),
        "reflection" => {
            prepare_helper(&root, None)?;
            for args in REFLECTION_TESTS {
                run_direct(&root, "cargo", args)?;
            }
            Ok(())
        }
        "loops" => {
            run_counted(&root, &["test", "-p", "gust-compute-macros", "loops"])?;
            run_direct(
                &root,
                "cargo",
                &["test", "-p", "gust-compute-wgpu", "--test", "loops"],
            )
        }
        "result" => {
            run_counted(
                &root,
                &["test", "-p", "gust-compute-macros", "--", "result"],
            )?;
            run_direct(
                &root,
                "cargo",
                &["test", "-p", "gust-compute-wgpu", "--test", "result"],
            )
        }
        "map" => {
            run_direct(
                &root,
                "cargo",
                &["test", "-p", "gust-compute", "--lib", "map::"],
            )?;
            run_counted(
                &root,
                &["test", "-p", "gust-compute-macros", "--lib", "map_tests"],
            )?;
            // The parallel CPU map (m7.c, D119): the runner's unit tests and the integration tests of the thread
            // count, the lowest failing index and the bit equality with one thread. Counted: no adapter is needed.
            run_counted(
                &root,
                &["test", "-p", "gust-compute", "--lib", "cpu::parallel"],
            )?;
            run_counted(&root, &["test", "-p", "gust-compute", "--test", "parallel"])?;
            run_direct(
                &root,
                "cargo",
                &[
                    "test",
                    "-p",
                    "gust-compute-wgpu",
                    "--test",
                    "map",
                    "--test",
                    "gpu_vec",
                    "--test",
                    "device_options",
                    "--test",
                    "limits",
                ],
            )
        }
        "dispatch" => {
            // The macros: each module is its own call, so a module that lost its tests fails by itself.
            for module in DISPATCH_MACRO_MODULES {
                run_counted(
                    &root,
                    &["test", "-p", "gust-compute-macros", "--lib", module],
                )?;
            }
            // Extent and the dispatch types of gust (the recording device, the extent of a call, the CPU
            // invocations of a multi-dimensional extent).
            for filter in DISPATCH_CORE_FILTERS {
                run_counted(&root, &["test", "-p", "gust-compute", "--lib", filter])?;
            }
            // The GPU tests, one binary per call: a binary that ran no test fails by itself.
            for test in DISPATCH_GPU_TESTS {
                run_counted(&root, &["test", "-p", "gust-compute-wgpu", "--test", test])?;
            }
            Ok(())
        }
        "workgroup" => {
            run_counted(
                &root,
                &["test", "-p", "gust-compute-macros", "--lib", "workgroup"],
            )?;
            run_counted(&root, &["test", "-p", "gust-compute", "--lib", "workgroup"])?;
            run_direct(
                &root,
                "cargo",
                &["test", "-p", "gust-compute-wgpu", "--test", "workgroup"],
            )
        }
        "algo" => run_counted(&root, &["test", "-p", "gust-algo"]).map(drop),
        "vectors" => {
            let mut macros = vec!["test", "-p", "gust-compute-macros", "--"];
            macros.extend(VECTOR_MACRO_FILTERS);
            run_counted(&root, &macros)?;
            run_counted(&root, &["test", "-p", "gust-compute", "--lib", "math::"])?;
            run_counted(
                &root,
                &[
                    "test",
                    "-p",
                    "gust-compute",
                    "--test",
                    "diagnostics",
                    "--",
                    "vectors_golden",
                    "products_golden",
                ],
            )?;
            export_golden_spirv(&root, "vectors", &VECTOR_GOLDEN_STEMS, &VECTOR_GOLDEN_SPIRV)?;
            run_counted(
                &root,
                &["test", "-p", "gust-compute-wgpu", "--test", "vectors"],
            )
            .map(drop)
        }
        "uniforms" => {
            let mut macros = vec!["test", "-p", "gust-compute-macros", "--"];
            macros.extend(UNIFORM_MACRO_FILTERS);
            run_counted(&root, &macros)?;
            run_counted(
                &root,
                &["test", "-p", "gust-compute", "--test", "uniforms_twin"],
            )?;
            run_counted(
                &root,
                &[
                    "test",
                    "-p",
                    "gust-compute",
                    "--test",
                    "diagnostics",
                    "--",
                    "uniforms_golden",
                ],
            )?;
            export_golden_spirv(
                &root,
                "uniforms",
                &UNIFORM_GOLDEN_STEMS,
                &UNIFORM_GOLDEN_SPIRV,
            )?;
            run_counted(
                &root,
                &["test", "-p", "gust-compute-wgpu", "--test", "uniforms"],
            )?;
            // The uniform cases of the typed-call and graph tests (m6.u1): their names say `uniform`.
            run_counted(
                &root,
                &[
                    "test",
                    "-p",
                    "gust-compute-wgpu",
                    "--test",
                    "typed",
                    "--test",
                    "graph",
                    "--",
                    "uniform",
                ],
            )
            .map(drop)
        }
        "examples" => verify(VerifyMode::Examples, false),
        "artifacts" => verify(VerifyMode::Artifacts, false),
        "full" => verify(VerifyMode::Full, true),
        _ => Err(format!("unknown feature area `{area}`")),
    }
}

pub(crate) fn check_workspace() -> Result<(), String> {
    let root = workspace_root()?;
    run_direct_args(&root, "cargo", &workspace_test_args())
}

pub(crate) fn check_format() -> Result<(), String> {
    run_direct(
        &workspace_root()?,
        "cargo",
        &["fmt", "--all", "--", "--check"],
    )
}

pub(crate) fn check_lints() -> Result<(), String> {
    run_direct(
        &workspace_root()?,
        "cargo",
        &[
            "clippy",
            "--locked",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
    )
}

pub(crate) fn check_changed() -> Result<(), String> {
    let paths = changed_paths()?;
    if paths.is_empty() {
        println!("No changed files; suggested check: cargo xtask check-fast");
        return Ok(());
    }
    println!("Changed files:");
    for path in &paths {
        println!("  {path}");
    }
    for check in select_changed_checks(&paths) {
        match check {
            ChangedCheck::Fast => {
                println!("Selected check: cargo xtask check-fast");
                verify(VerifyMode::Fast, false)?;
            }
            ChangedCheck::Feature(area) => {
                println!("Selected check: cargo xtask check-feature {area}");
                check_feature(Some(area))?;
            }
            ChangedCheck::Workspace => {
                println!("Selected check: cargo xtask check-workspace");
                check_workspace()?;
            }
        }
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum ChangedCheck {
    Fast,
    Feature(&'static str),
    Workspace,
}

fn select_changed_checks(paths: &[String]) -> Vec<ChangedCheck> {
    if paths.is_empty() {
        return Vec::new();
    }
    let has = |prefix: &str| paths.iter().any(|path| path.starts_with(prefix));
    let any = |needles: &[&str]| {
        paths
            .iter()
            .any(|path| needles.iter().any(|needle| path.contains(needle)))
    };

    if any(&[
        "Cargo.toml",
        "Cargo.lock",
        "xtask/",
        ".cargo/",
        ".github/",
        ".vscode/",
        "AGENTS.md",
    ]) {
        return vec![ChangedCheck::Fast];
    }

    let mut checks = Vec::new();
    if any(&["tests/fixtures/loops", "crates/gust-wgpu/tests/loops.rs"]) {
        checks.push(ChangedCheck::Feature("loops"));
    }
    if any(&[
        "tests/fixtures/workgroup",
        "crates/gust-wgpu/tests/workgroup.rs",
        "crates/gust-macros/src/validate/barrier.rs",
        "crates/gust-macros/src/tests/workgroup.rs",
    ]) {
        checks.push(ChangedCheck::Feature("workgroup"));
    }
    // M7 areas. A specific area is added to the broader check the path already had (macro, wgpu, core), never
    // instead of it; a golden fixture alone selects only its area, as the workgroup and loops fixtures do, while
    // a fixture of another golden still selects `macro` ([`unclaimed_fixture`] below).
    let fixture = |stems: &[&str]| {
        paths.iter().any(|path| {
            path.strip_prefix("tests/fixtures/").is_some_and(|rest| {
                stems.iter().any(|stem| {
                    rest.strip_prefix(stem)
                        .is_some_and(|tail| tail.starts_with('.'))
                })
            })
        })
    };
    if has("crates/gust-algo/") || fixture(&["reduce", "scan", "compact"]) {
        checks.push(ChangedCheck::Feature("algo"));
    }
    if fixture(&["map"])
        || any(&[
            "crates/gust/src/map.rs",
            "crates/gust/src/cpu/parallel.rs",
            "crates/gust/tests/parallel.rs",
            "crates/gust-macros/src/expand/map.rs",
            "crates/gust-macros/src/tests/map.rs",
            "crates/gust-wgpu/src/map_host.rs",
            "crates/gust-wgpu/tests/map.rs",
            "crates/gust-wgpu/tests/gpu_vec.rs",
            "crates/gust-wgpu/tests/device_options.rs",
        ])
    {
        checks.push(ChangedCheck::Feature("map"));
    }
    if fixture(&["par_for", "grid"])
        || any(&[
            "crates/gust/src/dispatch.rs",
            "crates/gust-macros/src/par_for.rs",
            "crates/gust-macros/src/tests/par_for.rs",
            "crates/gust-macros/src/tests/grid.rs",
            "crates/gust-wgpu/tests/par_for.rs",
            "crates/gust-wgpu/tests/grid.rs",
            "crates/gust-wgpu/tests/concurrency.rs",
            "crates/gust-wgpu/tests/async_readback.rs",
        ])
    {
        checks.push(ChangedCheck::Feature("dispatch"));
    }
    // A changed fixture that no fixture rule above claims (loops, workgroup, the M7 stems) is pinned by the macro
    // suite, whatever else changed: it selects `macro` even when another path selected an area first.
    let unclaimed_fixture = paths.iter().any(|path| {
        path.strip_prefix("tests/fixtures/").is_some_and(|rest| {
            !(rest.starts_with("loops")
                || rest.starts_with("workgroup")
                || ["reduce", "scan", "compact", "map", "par_for", "grid"]
                    .iter()
                    .any(|stem| {
                        rest.strip_prefix(stem)
                            .is_some_and(|tail| tail.starts_with('.'))
                    }))
        })
    });
    if has("crates/gust-macros/") || unclaimed_fixture {
        checks.push(ChangedCheck::Feature("macro"));
    }
    if any(&[
        "crates/gust/src/reflect/",
        "reflect.rs",
        "reflection.rs",
        "scripts/probes/",
    ]) {
        checks.push(ChangedCheck::Feature("reflection"));
    }
    if has("crates/gust-wgpu/") {
        checks.push(ChangedCheck::Feature("wgpu"));
    }
    if has("crates/gust/")
        && !has("crates/gust/src/reflect/")
        && !any(&["reflect.rs", "reflection.rs"])
    {
        checks.push(ChangedCheck::Feature("core"));
    }
    // The examples that depend on gust-algo (component-pool) are built and tested beside the crate's own area.
    if has("examples/") || has("generated-wgpu/") || has("crates/gust-algo/") {
        checks.push(ChangedCheck::Feature("examples"));
    }
    if checks.is_empty() {
        vec![ChangedCheck::Workspace]
    } else {
        checks
    }
}

#[cfg(test)]
#[path = "tests/check_routing.rs"]
mod check_routing_tests;

#[cfg(test)]
#[path = "tests/feature_areas.rs"]
mod feature_areas_tests;

pub(crate) fn status() -> Result<(), String> {
    let root = workspace_root()?;
    println!("Repository: {}", root.display());
    print_command(&root, "git", &["status", "--short", "--branch"])?;
    if let Ok(text) = fs::read_to_string(root.join("docs/development/STATUS.md")) {
        let active = text.lines().find(|line| line.starts_with("## Active:"));
        if let Some(active) = active {
            println!("{active}");
        }
        for line in text.lines().skip_while(|line| !line.starts_with("```text")) {
            println!("{line}");
            if line == "```" {
                break;
            }
        }
    }
    if let Ok(text) = fs::read_to_string(root.join(".ai/VALIDATION.json")) {
        println!("Validation: {}", compact_validation_summary(&text));
    }
    println!("Suggested clean-tree check: cargo xtask check-fast");
    Ok(())
}

/// Reports tool readiness and fails when a required tool is missing. `--smoke` also compiles
/// a hand-written Slang kernel to WGSL and SPIR-V and validates the SPIR-V.
pub(crate) fn doctor(args: &[String]) -> Result<(), String> {
    let smoke = match args {
        [] => false,
        [flag] if flag == "--smoke" => true,
        _ => return Err("doctor takes only `--smoke`".into()),
    };
    let mut missing = Vec::new();
    println!("Toolchain doctor:");
    for command in ["cargo", "rustc", "slangc"] {
        if command_exists(command) {
            println!("  {command}: found");
        } else {
            println!("  {command}: missing");
            missing.push(command);
        }
    }
    println!(
        "  spirv-val: {}",
        if command_exists("spirv-val") {
            "found"
        } else {
            "missing (needed for artifacts/full)"
        }
    );
    match workspace_root().and_then(|root| prepare_helper(&root, None)) {
        Ok(()) => println!("  gust-slang-reflect: ready"),
        Err(error) => {
            println!("  gust-slang-reflect: not ready ({error})");
            missing.push("gust-slang-reflect");
        }
    }
    println!(
        "  Vulkan/wgpu adapter: run `cargo xtask check-feature gpu-smoke` for a cheap runtime check"
    );
    if !missing.is_empty() {
        return Err(format!(
            "required tools missing or not ready: {}",
            missing.join(", ")
        ));
    }
    if smoke {
        doctor_smoke(&workspace_root()?)?;
    } else {
        println!("  Slang compile smoke: run `cargo xtask doctor --smoke`");
    }
    Ok(())
}

pub(crate) fn list_tests() -> Result<(), String> {
    let root = workspace_root()?;
    let output = workspace_command("cargo")
        .args(["test", "--workspace", "--", "--list"])
        .current_dir(&root)
        .output()
        .map_err(|error| format!("could not launch cargo test -- --list: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut total_tests = 0usize;
    let mut doctests = 0usize;
    let mut ignored_examples = 0usize;
    for line in stdout.lines() {
        if line.ends_with(": test") {
            total_tests += 1;
            if line.contains("tests::") && line.contains("example validation") {
                ignored_examples += 1;
            }
        }
        if line.contains(" - (line ") {
            doctests += 1;
        }
    }
    println!("cargo test --workspace -- --list");
    println!("  listed Rust tests: {total_tests}");
    println!("  listed doctests: {doctests}");
    println!("  example tests are ignored by attribute and run by `cargo xtask check-examples`");
    println!(
        "  categories: macro golden/rejection, core bridge/reflection, wgpu GPU/runtime, ignored examples, doctests"
    );
    if ignored_examples > 0 {
        println!("  ignored example test-name matches in list: {ignored_examples}");
    }
    Ok(())
}

pub(crate) fn explain_check(area: Option<&str>) -> Result<(), String> {
    println!("{}", explanation(area)?);
    Ok(())
}

/// The text `explain-check` prints for `area`, or the refusal for an unknown one.
pub(crate) fn explanation(area: Option<&str>) -> Result<&'static str, String> {
    let area = area.unwrap_or("all");
    let message = match area {
        "macro" => "macro: validator/emitter/golden changes; fastest compiler-front-end proof",
        "core" => {
            "core: ABI, Slang bridge, reflection parser, diagnostic mapping, SPIR-V structure helpers"
        }
        "wgpu" => "wgpu: all runtime/GPU integration tests; about 10s on this machine",
        "gpu-smoke" => "gpu-smoke: wgpu crate unit tests only; cheap runtime/cache sanity",
        "gpu-semantics" => {
            "gpu-semantics: semantics/numeric/option/loops/struct assignment GPU differential tests, the known-mismatch ledger and the scalar coretests"
        }
        "gpu-runtime" => {
            "gpu-runtime: reflection gate, cache/runtime mismatch tests, launch-limit checks and buffer layout assertions, typed calls and derived graph edges, device capabilities (a kernel's `requires` and `target` against the device, m8.w1; the build's default capability set, m8.w2)"
        }
        "reflection" => {
            "reflection: native Slang helper + core/wgpu reflection contract + the naga layout gate (the WGSL of every golden against the descriptors' layouts)"
        }
        "loops" => "loops: bounded-loop golden/rejection test plus real-GPU loop differential test",
        "result" => {
            "result: Result lowering/match regressions plus WGSL, SPIR-V, and real-GPU Result tests"
        }
        "map" => {
            "map: the map calls of `#[gpu]` on a function and their device side: gust::map unit tests, the gust-macros map tests (expansion, diagnostics and the golden of the generated kernels), the parallel CPU map of m7.c (the gust::cpu::parallel unit tests and crates/gust/tests/parallel.rs: thread count, bit equality with one thread, the lowest failing index; counted, no adapter needed), the map calls on the GPU against their CPU twins (crates/gust-wgpu/tests/map.rs), then GpuVec and the buffer pool, device options and adapter choice, and the launch and buffer limits on the GPU"
        }
        "dispatch" => {
            "dispatch: multi-dimensional dispatch and `par_for!` (m7.p, m7.d), each call counted so that a missing test file fails by itself: the gust-macros tests `par_for_tests::` (expansion, golden, export records, GUST-E0719 and E0720) and `grid_tests::` (the multi-dimensional kernels and their golden), the gust library tests of Extent (the `dispatch::` module, the CPU buffer sized by the extent and the CPU invocations of a two- and three-dimensional extent, each filter its own counted call), then crates/gust-wgpu/tests/par_for.rs, grid.rs, concurrency.rs and async_readback.rs on the first Vulkan adapter"
        }
        "workgroup" => {
            "workgroup: workgroup shared memory, the workgroup barrier and the local and group ids (m7.g1): the gust-macros workgroup tests (validator rules, expansion, the golden of the generated kernels, one rejection per diagnostic), the gust unit tests of the Workgroup runtime type, then crates/gust-wgpu/tests/workgroup.rs on the first Vulkan adapter (a block reduce against an independent host sum, the derived ids, the barrier in the SPIR-V validated with spirv-val and in the WGSL)"
        }
        "algo" => {
            "algo: the device-resident algorithms of gust-algo (m7.r1 reduce, m7.r2 scan and sum_wrapping, m7.r3 compaction): the schedule and bound arithmetic, the goldens of the reduce, scan and compact kernels and the CPU twins against independent references (no adapter needed), then the same cases on the first Vulkan adapter with the per-level float bound asserted, the dispatch counts of a scan and of a compaction, the empty and over-limit inputs and the pooled buffers"
        }
        "vectors" => {
            "vectors: the vector and matrix types (M6): the gust-macros tests of the types, products and negation, the gust::math unit tests, the vectors and products goldens on slangc, their SPIR-V (made here from the goldens, since no example uses a vector) validated with spirv-val for Vulkan 1.2 with an informational OpCopyLogical count, as check-artifacts does too; then the GPU tests of crates/gust-wgpu/tests/vectors.rs"
        }
        "uniforms" => {
            "uniforms: the uniform parameters (M6): the gust-macros tests of Uniform<T>, Std140 and ConstantBuffer, the uniforms_twin test, the uniforms golden on slangc and its SPIR-V validated as for vectors, then the GPU tests of crates/gust-wgpu/tests/uniforms.rs and the uniform cases of the typed and graph tests"
        }
        "examples" => {
            "examples: every example test (ordinary and ignored, `--include-ignored`; a package that runs no test fails) and example binaries; use for major behavior confidence"
        }
        "artifacts" => {
            "artifacts: example binaries plus exported SPIR-V validation (the examples' files and the goldens of vectors, products, uniforms, map, grid, workgroup, par_for, reduce, scan and compact, compiled here one section at a time), each module with an informational OpCopyLogical count (never a failure)"
        }
        "full" => {
            "full: release evidence (the Full mode of verify, as `cargo xtask check-full [--skip-web]`): everything above, check-dual in full mode, the artifact export with spirv-val, then the whole check-precompiled and check-web; writes .ai/VALIDATION.json, with `skipped_checks` naming a check that --skip-web left out"
        }
        "changed" => "changed: route dirty files to the cheapest likely sufficient check",
        "workspace" => {
            "workspace: non-example workspace tests via --exclude; example tests stay in check-examples"
        }
        "format" => "format: rustfmt only",
        "lints" => "lints: strict Clippy over workspace/all targets",
        "lint" => {
            "lint: no tracked shell scripts (D27) and no machine-specific paths in tracked text (home paths include WSL spellings); also part of check-fast"
        }
        "docs-check" => {
            "docs-check: relative Markdown links resolve to project files (anchors and external links are not checked), no merge-conflict markers, DECISIONS ids unique and increasing; also part of check-fast"
        }
        "check-arch" => {
            "check-arch: per-crate dependency allow-list (D02), no IR module or file outside future/, no SPIR-V library in the compiler crates; also part of check-fast"
        }
        "check-ui" => {
            "check-ui: cargo check of each tests/ui fixture crate; a crate with an expected.txt must fail with exactly one error satisfying every key (`code: none` names a macro error without a rustc code and needs a `message:`), or, with `build-error:` lines instead, must fail in the build script of gust with those texts in cargo's stderr; an env.txt gives a fixture its own environment variables (for example GUST_PROFILE=native, set after the workspace's dev); also part of check-fast and the Unit suite"
        }
        "check-cpu" => {
            "check-cpu: `cargo test --locked` of the examples with a CPU twin (vector-add, polynomial, signal-pipeline, particle-step, typed-pipeline, atomic-counter, staged-graph, component-pool, map-chain, texture-threshold, shape-bench), running the non-ignored twin tests against the host references (a package that runs no test fails, and each package's twin tests and derivation-domain guard must be reported `ok` by name); needs neither slangc nor a GPU; a step of the Unit suite"
        }
        "check-dual" => {
            "check-dual: the dual test build (`--cfg gust_dual`, D26) and its cross-crate proof, with its own target directory <target>/dual: tests/dual is formatted (`cargo fmt --check`) and passes clippy with -D warnings (the root fmt and CI clippy do not reach that workspace), as do vector-add and polynomial with the cfg (CI's clippy compiles their dual code out); the workspace builds with the cfg; in tests/dual a consumer's test (E1) and production binary (E3) each fail with exactly one E0277 for the missing CPU twin of the dependency `kernel-lib` without the cfg, and with it (E2) the test passes; a production binary that dispatches a module routed to the CPU (E4: `GUST_ROUTE__kernel_lib__physics=cpu`, no cfg) builds and runs on `gust::cpu::CpuDevice`, and the same source without the route fails with exactly one E0277 (E4-control). Without `--check-only` it then runs `cargo run` of vector-add and polynomial with the cfg, which must print the `CPU twin: matches (N elements, max ULP k)` line (polynomial also the twin's time beside run_cpu's), and of vector-add without the cfg, which must not mention the twin; that part needs a Vulkan GPU, and `check-full` runs it. Refuses to run while RUSTFLAGS, CARGO_ENCODED_RUSTFLAGS or CARGO_BUILD_RUSTFLAGS is set (it delivers the cfg through CARGO_ENCODED_RUSTFLAGS). `--check-only` is the CPU-only mode CI runs; it needs cargo only"
        }
        "check-cpu-target" => {
            "check-cpu-target: the production CPU target (D21, D51) on the fixture workspace tests/cpu-target, with cargo only (no slangc, no GPU), in <target>/cpu-target, cpu-target-dual and cpu-target-flip: E-a builds `app` in release under GUST_PROFILE=cpu and runs it (one `CPU target: matches` line per kernel, the `CPU target: map calls match (..)` line of the map case (which includes the parallel map of m7.c) and the `CPU target: par_for calls match (..)` line of the par_for case, each exactly once, the final line); E-b scans that binary for slangc, gust-slang-reflect, wgpu and vulkan (any letter case, with a control string) and reads `cargo tree -e features -p app`; E-c builds and runs it under cpu with --cfg gust_dual; E-d to E-g judge the four refusals on tests/cpu-target/e-*-expected.txt (E-e is the E0080 of gust-wgpu's require_gpu_target); E-h flips the profile cpu, unset, cpu, unset, cpu, unset in one directory; E-i runs cargo test -p kernels under cpu. Stops at the first failing step and names it. The Cpu suite is this command; CI's unit job runs it"
        }
        "check-routing" => {
            "check-routing: the routing driver (m8.r2, D65 to D67, D137) on the fixture workspace tests/routing, copied to <target>/routing-scratch and built in <target>/routing-fixture: plain cargo gives the macro's source defaults; seven gust.toml and profile states give the same targets in the probe, in routes.json and in a golden table, and a rerun gives the same bytes; nine bad gust.toml files are refused with exit 2, file, line and tag before cargo starts; `cargo gust test` passes where plain `cargo test` fails with the E0277 \"cannot run the kernel\" and `cargo gust run` runs the CPU-routed kernel; the route variable is flipped unset, set and defaulted in one warm target directory, in both directions twice (with the macro's route tracking removed the flip case must fail: run `--case flip` against that macro by hand); the fixture is rustfmt-clean and passes clippy with -D warnings per package under its own route (the lint case, also in `--check-only`); without `--check-only` the GPU-routed kernel also runs on the adapter the machine has (named in the output; needs slangc, its helper and Vulkan) and fails to compile when routed to the CPU, and handle-app (one source that calls the kernel through a RoutedDevice, m9.h1) is built and run under both routes and must print Gpu and then Cpu with the same values (the printed target is the build's own fact, `DESCRIPTOR.target`; the run itself needs the adapter, so `--check-only`, which CI runs, proves the CPU route by compiling and linting it, not by running it). `--check-only` is the mode CI runs; it needs cargo only"
        }
        "bless" => {
            "bless: reruns the Slang golden tests with BLESS_GOLDEN=1, which rewrites changed fixtures and fails with the diff; review before committing"
        }
        "test" => {
            "test: `cargo xtask test --suite Unit|Cpu|Slangc|Spirv|Reflect|Gpu|Full` runs one evidence tier of VALIDATION.md"
        }
        "all" => {
            "known areas: macro, core, wgpu, gpu-smoke, gpu-semantics, gpu-runtime, reflection, loops, result, vectors, uniforms, map, dispatch, workgroup, algo, examples, artifacts, full, changed, workspace, format, lints, lint, docs-check, check-arch, check-ui, check-cpu, check-dual, check-cpu-target, bless, test"
        }
        _ => return Err(format!("unknown check area `{area}`")),
    };
    Ok(message)
}

/// `export-artifacts [--hash-manifest PATH] [--compare PATH]`. The dispatcher in `main.rs` calls this without the
/// arguments, so they are read from the process arguments here (`cargo xtask export-artifacts ...`).
pub(crate) fn export_artifacts() -> Result<(), String> {
    let args = env::args().skip(2).collect::<Vec<_>>();
    export_artifacts_with(&args)
}

/// Runs every example, which exports its Slang, WGSL and SPIR-V to `generated-wgpu/`. With `--hash-manifest PATH`
/// or `--compare PATH` the `.slang`, `.rs`, `.wgsl` and `.spv` files of that directory are deleted first (so a stale
/// file cannot survive), and afterwards the SHA-256 of each is written to PATH (sorted) and/or compared with the manifest at PATH,
/// which fails on any missing, unexpected or changed file. The byte-identical gate of mx.p0 (plan section 2.3).
pub(crate) fn export_artifacts_with(args: &[String]) -> Result<(), String> {
    let options = ExportOptions::parse(args)?;
    let root = workspace_root()?;
    let directory = root.join("generated-wgpu");
    if options.hashes() {
        clear_exports(&directory)?;
    }
    // The shape bench exports nothing, so a GPU smoke of it here would only slow the export down (`verify` still runs it).
    for example in EXAMPLES
        .iter()
        .filter(|example| !NO_EXPORT_EXAMPLES.contains(example))
    {
        println!("Exporting artifacts via {example}");
        run_direct(&root, "cargo", &["run", "--quiet", "-p", example])?;
    }
    if options.hashes() {
        let manifest = hash_manifest(&directory)?;
        let recorded = manifest.lines().count();
        if let Some(path) = &options.hash_manifest {
            fs::write(path, &manifest).map_err(|error| format!("{}: {error}", path.display()))?;
            println!(
                "Wrote the hash manifest of {recorded} exports to {}",
                path.display()
            );
        }
        if let Some(path) = &options.compare {
            let baseline =
                fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
            compare_manifests(&baseline, &manifest)?;
            println!(
                "The {recorded} exports equal the baseline manifest {}",
                path.display()
            );
        }
    }
    Ok(())
}

pub(crate) fn measure_tests(args: &[String]) -> Result<(), String> {
    let root = workspace_root()?;
    let mut json_path = None;
    let mut command_args = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => {
                index += 1;
                json_path = Some(PathBuf::from(args.get(index).ok_or("--json needs a path")?));
            }
            other => command_args.push(other.to_owned()),
        }
        index += 1;
    }
    let commands = if command_args.is_empty() {
        let mut workspace = vec!["cargo".to_owned()];
        workspace.extend(workspace_test_args());
        vec![
            vec![
                "cargo".into(),
                "test".into(),
                "-p".into(),
                "gust-compute-macros".into(),
            ],
            vec![
                "cargo".into(),
                "test".into(),
                "-p".into(),
                "gust-compute".into(),
                "--lib".into(),
            ],
            vec![
                "cargo".into(),
                "test".into(),
                "-p".into(),
                "gust-compute-wgpu".into(),
            ],
            workspace,
        ]
    } else {
        vec![command_args]
    };
    let mut records = Vec::new();
    for command in commands {
        if command.is_empty() {
            return Err("measure-tests command cannot be empty".into());
        }
        println!("== {}", command.join(" "));
        let start = Instant::now();
        let status = workspace_command(&command[0])
            .args(&command[1..])
            .current_dir(&root)
            .status()
            .map_err(|error| format!("could not launch {}: {error}", command[0]))?;
        let elapsed = start.elapsed().as_millis();
        let exit = status.code().unwrap_or(-1);
        println!("elapsed_ms={elapsed} exit={exit}");
        records.push((command.join(" "), elapsed, exit));
        if !status.success() {
            return Err(format!("{} failed", command.join(" ")));
        }
    }
    if let Some(path) = json_path {
        let mut json = String::from("{\n  \"schema_version\": 1,\n  \"measurements\": [\n");
        for (index, (command, elapsed, exit)) in records.iter().enumerate() {
            if index != 0 {
                json.push_str(",\n");
            }
            json.push_str(&format!(
                "    {{ \"command\": {}, \"elapsed_ms\": {elapsed}, \"exit_code\": {exit} }}",
                json_string(command)
            ));
        }
        json.push_str("\n  ]\n}\n");
        fs::write(path, json).map_err(|error| error.to_string())?;
    }
    Ok(())
}

pub(crate) fn hook_format_rust_after_edit() -> Result<(), String> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|error| error.to_string())?;
    if !input.contains("PostToolUse") || !input.contains(".rs") {
        return Ok(());
    }
    let edit_tool = [
        "create_file",
        "replace_string_in_file",
        "multi_replace_string_in_file",
        "insert_edit_into_file",
        "edit_file",
        "apply_patch",
    ]
    .iter()
    .any(|tool| input.contains(tool));
    if edit_tool {
        run_direct(&workspace_root()?, "cargo", &["fmt", "--all"])?;
    }
    Ok(())
}

pub(crate) fn self_test() -> Result<(), String> {
    require_command("cargo")?;
    require_command("rustc")?;
    require_command("slangc")?;
    let root = workspace_root()?;
    let mut checks = Vec::new();
    run_checked(&root, &mut checks, "rustc", &["--version"])?;
    if checks.len() != 1 || !checks[0].stdout.contains("rustc") {
        return Err("command capture self-test failed".into());
    }
    println!("PASS: xtask command capture and tool lookup work.");
    Ok(())
}

// ---- check-dual (m3.f1): the dual test build and its cross-crate fixture ----

/// The fixture workspace of the cross-crate proof, relative to the repository root. It has a `[workspace]` of its
/// own, so the root workspace neither contains it nor builds it with `--workspace`.
pub(crate) const DUAL_DIR: &str = "tests/dual";

/// The value of `CARGO_ENCODED_RUSTFLAGS` that delivers `--cfg gust_dual`: two flags joined by the unit separator
/// (0x1f) that cargo splits that variable at.
pub(crate) const DUAL_RUSTFLAGS: &str = "--cfg\u{1f}gust_dual";

/// The variable that carries the cfg to the child cargo processes.
pub(crate) const DUAL_RUSTFLAGS_VARIABLE: &str = "CARGO_ENCODED_RUSTFLAGS";

/// The environment variables that decide rustflags. `check-dual` refuses to run while any of them is present, even
/// with an empty value, because it sets `CARGO_ENCODED_RUSTFLAGS` itself and that takes precedence over the rest.
pub(crate) const RUSTFLAGS_VARIABLES: [&str; 3] = [
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_BUILD_RUSTFLAGS",
];

/// The tests of `consumer` that E2 requires to run and pass: the twin on a dependency's kernel, the overflow mode
/// that a dependent with a per-package override reports, and what the twins do with an overflowing addition: the one
/// of a dependency's kernel follows the dependency's mode, the one of the dependent's own kernel follows the
/// dependent's.
pub(crate) const DUAL_TESTS: &[&str] = &[
    "tests::twin_runs",
    "tests::overflow_mode_is_the_dependents_own",
    "twin_overflow::a_twin_follows_the_overflow_mode_of_the_package_that_defines_the_kernel",
    "twin_overflow::a_twin_defined_in_the_dependent_follows_the_dependents_own_mode",
    "vector_overflow::an_integer_vector_add_panics_only_where_gust_and_the_kernels_package_both_check",
];

/// What `cargo xtask check-dual` runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DualMode {
    /// `--check-only`: CPU only, no GPU; the mode CI runs.
    CheckOnly,
    /// No argument: the same steps, then the example runs of [`dual_run_steps`], which need the GPU.
    Full,
}

/// How one step of `check-dual` is judged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DualKind {
    /// Must succeed; its output goes to the terminal.
    Build,
    /// Must fail with exactly the one error of the expectation file (relative to [`DUAL_DIR`]) on the cargo target
    /// `target` of the package in the directory `case`, judged by `ui::judge` like a `check-ui` fixture.
    Fails {
        case: &'static str,
        target: &'static str,
        expected: &'static str,
    },
    /// E2, first half: must compile with the cfg. A failure means the cfg did not reach the dependency.
    Compiles,
    /// E2, second half: must succeed, and each named test must be reported `ok` (a run that finds no test passes
    /// vacuously, so the names are required).
    Passes { tests: &'static [&'static str] },
    /// An example's `cargo run` (Full mode; needs the GPU). It must succeed. With `twin` its output must hold exactly
    /// one `CPU twin: matches (N elements, max ULP k)` line with N above zero, and, on lines that start with
    /// `CPU twin`, each substring of `also`; without `twin` (the build without the cfg) no line may mention
    /// `CPU twin` at all.
    Runs {
        twin: bool,
        also: &'static [&'static str],
    },
    /// E5 of `check-precompiled` (`cargo tree -e features`): must succeed, list the crates that carry the feature,
    /// and show no `slang-runtime` ([`crate::precompiled::tree_problems`]). It builds nothing and takes no
    /// `--target-dir`.
    Tree,
    /// E4 (m8.r1, D65 to D67): a production binary that dispatches a routed module on `CpuDevice`, built with no
    /// `--cfg gust_dual` and with the tag routed to the CPU by the variable `variable` set to `value` (the one `cargo
    /// gust` sets from the app's `gust.toml`, m8.r2). It must run, and its standard output must hold the line `line`
    /// exactly. The same source without the variable is the one E0277 of `e4-expected.txt` (a [`DualKind::Fails`]
    /// step).
    Routed {
        variable: &'static str,
        value: &'static str,
        line: &'static str,
    },
}

/// One cargo call of `check-dual`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DualStep {
    /// `fmt`, `build`, `clippy`, `clippy-examples`, `E1`, `E2-compile`, `E2-run`, `E3`, `E4-routed`, `E4-control` or, in Full mode,
    /// `run-vector-add`, `run-polynomial` or `run-vector-add-no-cfg`; the steps of the precompiled fixture
    /// ([`precompiled_dual_steps`]) are `E5-build`, `E5-test` and `E5-tree`.
    pub(crate) id: &'static str,
    /// The arguments after `cargo`. Every step has exactly one `--target-dir <dir>` pair (before any `--`), except
    /// `fmt`, which builds nothing, and `E5-tree`.
    pub(crate) args: Vec<String>,
    /// Whether the child gets `CARGO_ENCODED_RUSTFLAGS` with `--cfg gust_dual`.
    pub(crate) cfg: bool,
    /// The `GUST_PROFILE` this step builds under, set after the `dev` of the workspace steps ([`dual_command`]);
    /// `None` keeps `dev`. Only the steps of the precompiled fixture are strict (R4.2).
    pub(crate) profile: Option<&'static str>,
    pub(crate) kind: DualKind,
}

impl DualStep {
    /// The environment this step adds to the child's: the cfg (E1 and E3 build without it) and, for the fixture, the
    /// strict profile. The workspace steps add no profile and keep the `dev` of [`set_workspace_profile`].
    pub(crate) fn env(&self) -> Vec<(&'static str, &'static str)> {
        let mut env = Vec::new();
        if self.cfg {
            env.push((DUAL_RUSTFLAGS_VARIABLE, DUAL_RUSTFLAGS));
        }
        if let Some(profile) = self.profile {
            env.push((PROFILE_VARIABLE, profile));
        }
        if let DualKind::Routed {
            variable, value, ..
        } = &self.kind
        {
            env.push((*variable, *value));
        }
        env
    }

    /// The command line for the terminal: the arguments with the target directory (a machine-specific path) replaced
    /// by `<target>/dual`.
    pub(crate) fn shown(&self) -> String {
        let mut env = String::new();
        if self.cfg {
            env.push_str("CARGO_ENCODED_RUSTFLAGS=\"--cfg gust_dual\" ");
        }
        if let Some(profile) = self.profile {
            env.push_str(&format!("{PROFILE_VARIABLE}={profile} "));
        }
        if let DualKind::Routed {
            variable, value, ..
        } = &self.kind
        {
            env.push_str(&format!("{variable}={value} "));
        }
        let mut shown: Vec<&str> = Vec::with_capacity(self.args.len());
        let mut args = self.args.iter().map(String::as_str);
        while let Some(arg) = args.next() {
            shown.push(arg);
            if arg == "--target-dir" {
                args.next();
                shown.push("<target>/dual");
            }
        }
        format!("{env}cargo {}", shown.join(" "))
    }
}

/// `check-dual`'s own target directory: `<cargo target dir>/dual`, so that the changed rustflags never invalidate
/// the cache of the normal build. A relative `CARGO_TARGET_DIR` is read against the repository root, where the
/// child cargo processes run.
pub(crate) fn dual_target_dir(root: &Path) -> PathBuf {
    dual_dir_under(root, &cargo_target_dir(root))
}

/// `<target>/dual` for the cargo target directory `target` (absolute, or relative to `root`).
pub(crate) fn dual_dir_under(root: &Path, target: &Path) -> PathBuf {
    root.join(target).join("dual")
}

/// The cargo calls of `check-dual --check-only`, in order: `cargo fmt --check` of the `tests/dual` fixture, the
/// workspace build with the cfg, clippy of the fixture and of the two examples that have a dual `main` (both with
/// the cfg), then E1, E2 and E3 on the fixture. Every call
/// that builds is `--locked`. Kept as data so a test can require the exact lists.
///
/// The fixture is a workspace of its own, so neither the root `cargo fmt --all` nor the clippy of CI (both of the root
/// workspace) look at it; `fmt` and `clippy` here are what cover it. Clippy runs with the cfg because `kernel-lib`'s
/// twin and the `prod` binary only compile with it (E1 and E3 prove exactly that).
pub(crate) fn dual_steps(target_dir: &str) -> Vec<DualStep> {
    let target_dir = target_dir.to_owned();
    let manifest = format!("{DUAL_DIR}/Cargo.toml");
    let strings = |args: &[&str]| -> Vec<String> {
        let mut args: Vec<String> = args.iter().map(|arg| (*arg).to_owned()).collect();
        args.push("--target-dir".to_owned());
        args.push(target_dir.clone());
        args
    };
    vec![
        DualStep {
            id: "fmt",
            args: [
                "fmt",
                "--manifest-path",
                &manifest,
                "--all",
                "--",
                "--check",
            ]
            .iter()
            .map(|arg| (*arg).to_owned())
            .collect(),
            cfg: false,
            profile: None,
            kind: DualKind::Build,
        },
        DualStep {
            id: "build",
            args: strings(&["build", "--locked", "--workspace", "--all-targets"]),
            cfg: true,
            profile: None,
            kind: DualKind::Build,
        },
        DualStep {
            id: "clippy",
            args: [
                "clippy",
                "--locked",
                "--manifest-path",
                &manifest,
                "--workspace",
                "--all-targets",
                "--target-dir",
                &target_dir,
                "--",
                "-D",
                "warnings",
            ]
            .iter()
            .map(|arg| (*arg).to_owned())
            .collect(),
            cfg: true,
            profile: None,
            kind: DualKind::Build,
        },
        // The root workspace's code that only exists with the cfg (the examples' `report_twin`, the macro's twins
        // outside `cfg(test)`) is compiled out of CI's clippy, and the `build` step above allows warnings.
        DualStep {
            id: "clippy-examples",
            args: [
                "clippy",
                "--locked",
                "-p",
                "vector-add",
                "-p",
                "polynomial",
                "--all-targets",
                "--target-dir",
                &target_dir,
                "--",
                "-D",
                "warnings",
            ]
            .iter()
            .map(|arg| (*arg).to_owned())
            .collect(),
            cfg: true,
            profile: None,
            kind: DualKind::Build,
        },
        // E1: without the cfg the consumer's own `cfg(test)` does not give `kernel-lib` its twin.
        DualStep {
            id: "E1",
            args: strings(&[
                "test",
                "--locked",
                "--color",
                "never",
                "--manifest-path",
                &manifest,
                "-p",
                "consumer",
                "--lib",
                "--no-run",
                "--message-format",
                "json",
            ]),
            cfg: false,
            profile: None,
            kind: DualKind::Fails {
                case: "consumer",
                target: "consumer",
                expected: "e1-expected.txt",
            },
        },
        // E2: with the cfg the same test builds, and runs.
        DualStep {
            id: "E2-compile",
            args: strings(&[
                "test",
                "--locked",
                "--manifest-path",
                &manifest,
                "-p",
                "consumer",
                "--lib",
                "--no-run",
            ]),
            cfg: true,
            profile: None,
            kind: DualKind::Compiles,
        },
        DualStep {
            id: "E2-run",
            args: strings(&[
                "test",
                "--locked",
                "--manifest-path",
                &manifest,
                "-p",
                "consumer",
                "--lib",
            ]),
            cfg: true,
            profile: None,
            kind: DualKind::Passes { tests: DUAL_TESTS },
        },
        // E3: without the cfg a production binary gets no twin.
        DualStep {
            id: "E3",
            args: strings(&[
                "check",
                "--locked",
                "--color",
                "never",
                "--manifest-path",
                &manifest,
                "-p",
                "consumer",
                "--bin",
                "prod",
                "--message-format",
                "json",
            ]),
            cfg: false,
            profile: None,
            kind: DualKind::Fails {
                case: "consumer",
                target: "prod",
                expected: "e3-expected.txt",
            },
        },
        // E4 (m8.r1): a module routed to the CPU is a CPU body in a production build. `prod-routed` dispatches
        // `kernel-lib`'s `routed` module on `CpuDevice`; with the tag routed to the CPU it builds without the cfg and
        // runs, and without the route the same source is E3's error for the GPU kernel. E4-control is a `cargo build`
        // (JSON), not a `cargo check`, so that it is a cpu-to-unset flip of the very unit E4-routed built: a macro
        // that did not track the route variable would leave `kernel-lib` stale and the control would compile.
        DualStep {
            id: "E4-routed",
            args: strings(&[
                "run",
                "--locked",
                "--color",
                "never",
                "--manifest-path",
                &manifest,
                "-p",
                "consumer",
                "--bin",
                "prod-routed",
            ]),
            cfg: false,
            profile: None,
            kind: DualKind::Routed {
                variable: ROUTE_VARIABLE,
                value: "cpu",
                line: ROUTED_LINE,
            },
        },
        DualStep {
            id: "E4-control",
            args: strings(&[
                "build",
                "--locked",
                "--color",
                "never",
                "--manifest-path",
                &manifest,
                "-p",
                "consumer",
                "--bin",
                "prod-routed",
                "--message-format",
                "json",
            ]),
            cfg: false,
            profile: None,
            kind: DualKind::Fails {
                case: "consumer",
                target: "prod-routed",
                expected: "e4-expected.txt",
            },
        },
    ]
}

/// The variable E4 sets: the route of the tag `physics` of the package `kernel-lib`
/// (`gust::routes::variable_name("kernel-lib", "physics")`).
pub(crate) const ROUTE_VARIABLE: &str = "GUST_ROUTE__kernel_lib__physics";

/// What `consumer`'s `prod-routed` prints when the routed module ran on `CpuDevice`: the doubled input.
pub(crate) const ROUTED_LINE: &str = "routed: [2.0, 4.0, 6.0]";

/// The problems of a [`DualKind::Routed`] step for the outcome of its `cargo run`: none when it succeeded and printed
/// the line exactly once.
pub(crate) fn routed_run_problems(
    id: &str,
    line: &str,
    success: bool,
    stdout: &str,
    stderr: &str,
) -> Vec<String> {
    let mut problems = Vec::new();
    if !success {
        problems.push(format!(
            "check-dual {id}: `cargo run` of the routed binary failed; with the tag routed to the CPU it must build and run without --cfg gust_dual:\n{stderr}"
        ));
        return problems;
    }
    let found = without_colors(stdout)
        .lines()
        .filter(|printed| printed.trim() == line)
        .count();
    if found != 1 {
        problems.push(format!(
            "check-dual {id}: expected the line `{line}` exactly once on standard output, found it {found} time(s); output was: {}",
            stdout.trim()
        ));
    }
    problems
}

/// The example runs of the full mode, in order (debug profile, so debug sizes): `vector-add` and `polynomial` with the
/// cfg must each print the `CPU twin: matches` line, and `vector-add` without it must not mention the twin (a normal
/// build prints nothing new). They run in the dual target directory beside the fixture steps; the cfg changes the
/// fingerprints, so the two builds of the examples keep separate artifacts there. They need the GPU.
pub(crate) fn dual_run_steps(target_dir: &str) -> Vec<DualStep> {
    let run = |package: &str| -> Vec<String> {
        ["run", "--locked", "-p", package, "--target-dir", target_dir]
            .iter()
            .map(|arg| (*arg).to_owned())
            .collect()
    };
    vec![
        DualStep {
            id: "run-vector-add",
            args: run("vector-add"),
            cfg: true,
            profile: None,
            kind: DualKind::Runs {
                twin: true,
                also: &[],
            },
        },
        DualStep {
            id: "run-polynomial",
            args: run("polynomial"),
            cfg: true,
            profile: None,
            kind: DualKind::Runs {
                twin: true,
                also: &["CPU twin time: ", "debug profile, single-threaded"],
            },
        },
        DualStep {
            id: "run-vector-add-no-cfg",
            args: run("vector-add"),
            cfg: false,
            profile: None,
            kind: DualKind::Runs {
                twin: false,
                also: &[],
            },
        },
    ]
}

/// The test of the fixture `app` that E5 requires to run and pass: the CPU twin of the fixture's own kernel against
/// the host reference. It exists only under `--cfg gust_dual` (`tests/precompiled/app/src/lib.rs`), so a build
/// without the cfg runs nothing, reports no such test, and fails.
pub(crate) const PRECOMPILED_TWIN_TESTS: &[&str] =
    &["tests::the_cpu_twin_matches_the_host_reference"];

/// The profile of E5: the strict one the fixture is built under (R4.2).
pub(crate) const PRECOMPILED_PROFILE: &str = "native";

/// The cargo calls of E5 (`check-precompiled`, strict plus dual, R4.2) on the fixture `tests/precompiled`, built
/// through [`dual_command`] like every other step here: no rustflags variable but the cfg, the dual target
/// directory, and `GUST_PROFILE=native` instead of the `dev` of the workspace steps. Each step has the cfg and the
/// strict profile; the fixture has `default-features = false` and no feature, so the bridge is absent, and its
/// committed artifacts satisfy the strict assert. Kept as data so a test can require the exact lists.
///
/// 1. `E5-build`: the whole fixture builds (all targets), `kernels` with its twin and its embedded artifact.
/// 2. `E5-test`: `app`'s unit test dispatches the twin on `gust::cpu::CpuDevice` and compares it with the host
///    reference; it exists only under the cfg, so dropping the cfg makes this step fail on the missing test.
/// 3. `E5-tree`: `cargo tree -e features` of `app` shows no `slang-runtime`.
pub(crate) fn precompiled_dual_steps(target_dir: &str) -> Vec<DualStep> {
    let manifest = format!("{PRECOMPILED_DIR}/Cargo.toml");
    let owned =
        |args: &[&str]| -> Vec<String> { args.iter().map(|arg| (*arg).to_owned()).collect() };
    let with_target = |args: &[&str]| -> Vec<String> {
        let mut args = owned(args);
        args.push("--target-dir".to_owned());
        args.push(target_dir.to_owned());
        args
    };
    vec![
        DualStep {
            id: "E5-build",
            args: with_target(&[
                "build",
                "--locked",
                "--manifest-path",
                &manifest,
                "--workspace",
                "--all-targets",
            ]),
            cfg: true,
            profile: Some(PRECOMPILED_PROFILE),
            kind: DualKind::Build,
        },
        DualStep {
            id: "E5-test",
            args: with_target(&[
                "test",
                "--locked",
                "--manifest-path",
                &manifest,
                "-p",
                "app",
                "--lib",
            ]),
            cfg: true,
            profile: Some(PRECOMPILED_PROFILE),
            kind: DualKind::Passes {
                tests: PRECOMPILED_TWIN_TESTS,
            },
        },
        DualStep {
            id: "E5-tree",
            args: owned(&[
                "tree",
                "--locked",
                "--color",
                "never",
                "-e",
                "features",
                "--manifest-path",
                &manifest,
                "-p",
                "app",
            ]),
            cfg: true,
            profile: Some(PRECOMPILED_PROFILE),
            kind: DualKind::Tree,
        },
    ]
}

/// Every step of a mode, in order: the `--check-only` steps, then in the full mode the example runs.
pub(crate) fn dual_steps_for(mode: DualMode, target_dir: &str) -> Vec<DualStep> {
    let mut steps = dual_steps(target_dir);
    if mode == DualMode::Full {
        steps.extend(dual_run_steps(target_dir));
    }
    steps
}

/// The `(elements, max ULP)` of a line `CPU twin: matches (N elements, max ULP k)`, or `None` for any other text.
pub(crate) fn parse_twin_match(line: &str) -> Option<(usize, u64)> {
    let inner = line
        .strip_prefix("CPU twin: matches (")?
        .strip_suffix(')')?;
    let (elements, ulps) = inner.split_once(" elements, max ULP ")?;
    let digits = |text: &str| text.bytes().all(|byte| byte.is_ascii_digit()) && !text.is_empty();
    if !digits(elements) || !digits(ulps) {
        return None;
    }
    Some((elements.parse().ok()?, ulps.parse().ok()?))
}

/// The problems of a [`DualKind::Runs`] step for the standard output of its successful `cargo run`; empty when the
/// output is what the step requires.
pub(crate) fn twin_run_problems(id: &str, twin: bool, also: &[&str], stdout: &str) -> Vec<String> {
    let stdout = without_colors(stdout);
    let mentions: Vec<&str> = stdout
        .lines()
        .map(str::trim)
        .filter(|line| line.contains("CPU twin"))
        .collect();
    let mut problems = Vec::new();
    if !twin {
        if !mentions.is_empty() {
            problems.push(format!(
                "check-dual {id}: a build without --cfg gust_dual printed twin output, which only the dual build may print: {}",
                mentions.join(" | ")
            ));
        }
        return problems;
    }
    let matches: Vec<(usize, u64)> = mentions
        .iter()
        .filter_map(|line| parse_twin_match(line))
        .collect();
    match matches.as_slice() {
        [] => problems.push(format!(
            "check-dual {id}: the run printed no `CPU twin: matches (N elements, max ULP k)` line, so the twin did not run (is --cfg gust_dual reaching the example?)"
        )),
        [(0, _)] => problems.push(format!(
            "check-dual {id}: the twin matched 0 elements, which proves nothing"
        )),
        [_] => {}
        _ => problems.push(format!(
            "check-dual {id}: the run printed {} `CPU twin: matches` lines, expected exactly one",
            matches.len()
        )),
    }
    for wanted in also {
        if !mentions.iter().any(|line| line.contains(wanted)) {
            problems.push(format!(
                "check-dual {id}: no line mentioning `CPU twin` contains `{wanted}`"
            ));
        }
    }
    problems
}

/// `--check-only`, or no argument for the full mode.
pub(crate) fn parse_dual_args(args: &[String]) -> Result<DualMode, String> {
    match args {
        [] => Ok(DualMode::Full),
        [flag] if flag == "--check-only" => Ok(DualMode::CheckOnly),
        _ => Err("check-dual takes only `--check-only` (CPU only, what CI runs)".into()),
    }
}

/// Refuses while any of [`RUSTFLAGS_VARIABLES`] is present in the environment, even with an empty value
/// (`lookup` answers `Some(..)` for a variable that is set). The error names every one that is.
pub(crate) fn refuse_preset_rustflags(
    lookup: impl Fn(&str) -> Option<OsString>,
) -> Result<(), String> {
    let set: Vec<&str> = RUSTFLAGS_VARIABLES
        .iter()
        .copied()
        .filter(|name| lookup(name).is_some())
        .collect();
    if set.is_empty() {
        return Ok(());
    }
    Err(format!(
        "check-dual refuses to run because {} {} set in the environment (an empty value counts). It delivers `--cfg gust_dual` through {DUAL_RUSTFLAGS_VARIABLE}, which takes precedence over the other rustflags sources, so flags set there would be dropped from the dual build, and a set {DUAL_RUSTFLAGS_VARIABLE} would be replaced; unset {} and run it again",
        set.join(", "),
        if set.len() == 1 { "is" } else { "are" },
        if set.len() == 1 { "it" } else { "them" },
    ))
}

/// The cargo config files that mention `rustflags`, described without machine paths: the workspace's own
/// `.cargo/config.toml`, those of its parent directories (cargo reads them all) and `config.toml` of `cargo_home`.
/// `CARGO_ENCODED_RUSTFLAGS` replaces such `build.rustflags` and `target.*.rustflags` settings, so on a machine with
/// one the dual build differs from a normal build in those flags.
pub(crate) fn rustflags_config_files(root: &Path, cargo_home: Option<&Path>) -> Vec<String> {
    let mentions = |path: &Path| {
        fs::read_to_string(path).is_ok_and(|text| text.to_ascii_lowercase().contains("rustflags"))
    };
    let mut found = Vec::new();
    for (level, directory) in root.ancestors().enumerate() {
        for name in ["config.toml", "config"] {
            if mentions(&directory.join(".cargo").join(name)) {
                let place = match level {
                    0 => "the workspace".to_owned(),
                    1 => "1 level above the workspace".to_owned(),
                    n => format!("{n} levels above the workspace"),
                };
                found.push(format!("{place}: .cargo/{name}"));
            }
        }
    }
    if let Some(home) = cargo_home {
        for name in ["config.toml", "config"] {
            if mentions(&home.join(name)) {
                found.push(format!("CARGO_HOME: {name}"));
            }
        }
    }
    found
}

/// `$CARGO_HOME`, or `~/.cargo`.
pub(crate) fn cargo_home() -> Option<PathBuf> {
    env::var_os("CARGO_HOME").map(PathBuf::from).or_else(|| {
        env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
            .map(|home| PathBuf::from(home).join(".cargo"))
    })
}

/// The names of `tests` that `stdout` of `cargo test` does not report as `ok`.
pub(crate) fn tests_not_ok<'a>(stdout: &str, tests: &[&'a str]) -> Vec<&'a str> {
    let stdout = without_colors(stdout);
    tests
        .iter()
        .copied()
        .filter(|name| {
            let wanted = format!("test {name} ... ok");
            !stdout.lines().any(|line| line.trim() == wanted)
        })
        .collect()
}

pub(crate) fn dual_command(root: &Path, step: &DualStep) -> Command {
    let mut command = Command::new(env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo")));
    command
        .args(&step.args)
        .current_dir(root)
        .stdin(Stdio::null());
    // Absent, not inherited: whatever the parent has, the child starts without any rustflags variable, then gets
    // the cfg on the steps that build with it.
    for name in RUSTFLAGS_VARIABLES {
        command.env_remove(name);
    }
    // The workspace builds under `dev` whatever the parent has (R4.2); a step's own environment, applied next, may
    // still choose another profile (the strict-plus-dual gate of m4.e2 does).
    set_workspace_profile(&mut command);
    for (name, value) in step.env() {
        command.env(name, value);
    }
    command
}

/// The `CARGO_TARGET_<TRIPLE>_RUSTFLAGS` variables among `names` (upper-cased and sorted): cargo config given
/// through the environment, which `CARGO_ENCODED_RUSTFLAGS` replaces like the config files.
pub(crate) fn target_rustflags_variables(names: impl Iterator<Item = OsString>) -> Vec<String> {
    let mut found: Vec<String> = names
        .filter_map(|name| name.into_string().ok())
        .map(|name| name.to_ascii_uppercase())
        .filter(|name| {
            name.len() > "CARGO_TARGET__RUSTFLAGS".len()
                && name.starts_with("CARGO_TARGET_")
                && name.ends_with("_RUSTFLAGS")
        })
        .collect();
    found.sort();
    found.dedup();
    found
}

/// `cargo xtask check-dual [--check-only]`: the dual test build (`--cfg gust_dual`, D26) and its cross-crate
/// proof. `--check-only` needs cargo only (no slangc, no GPU); the full mode also runs two examples on the GPU.
///
/// 1. `tests/dual` is formatted (`cargo fmt --check`; the root `cargo fmt --all` does not reach that workspace).
/// 2. The workspace builds with the cfg (all targets) in `<target dir>/dual`.
/// 3. The fixture, and the two examples with a dual `main`, pass clippy with `-D warnings`, with the cfg (CI's
///    clippy covers the root workspace without it).
/// 4. E1 (no cfg): `tests/dual`'s `consumer` test fails with exactly one E0277, the missing twin of the dependency.
/// 5. E2 (cfg): the same test compiles and passes, which is the proof that the cfg reached `kernel-lib`.
/// 6. E3 (no cfg): a production binary fails with exactly that one error.
/// 7. E4 (no cfg): a module routed to the CPU runs on `CpuDevice` in a production binary; unrouted it is E3's error.
/// 8. Full mode only: `cargo run -p vector-add` and `-p polynomial` with the cfg print the `CPU twin: matches` line,
///    and `-p vector-add` without it prints nothing about the twin.
pub(crate) fn check_dual(args: &[String]) -> Result<(), String> {
    let mode = parse_dual_args(args)?;
    refuse_preset_rustflags(|name| env::var_os(name))?;
    let root = workspace_root()?;
    let configs = rustflags_config_files(&root, cargo_home().as_deref());
    if !configs.is_empty() {
        println!(
            "note: {DUAL_RUSTFLAGS_VARIABLE} replaces `build.rustflags` and `target.*.rustflags` of cargo config files, and these mention rustflags ({}); the dual build here differs from a normal build in those flags",
            configs.join("; ")
        );
    }
    let target_variables = target_rustflags_variables(env::vars_os().map(|(name, _)| name));
    if !target_variables.is_empty() {
        println!(
            "note: {DUAL_RUSTFLAGS_VARIABLE} also replaces the rustflags of {}, so the dual build differs from a normal build in those flags",
            target_variables.join(", ")
        );
    }
    // Cargo takes the directory as an argument, so a path that is not valid UTF-8 would be altered, not passed.
    let target_dir = dual_target_dir(&root);
    let target_dir = target_dir.to_str().ok_or_else(|| {
        "check-dual: the cargo target directory is not valid UTF-8; set CARGO_TARGET_DIR to a plain path".to_owned()
    })?;
    for step in dual_steps_for(mode, target_dir) {
        println!("check-dual {}: {}", step.id, step.shown());
        run_dual_step(&root, &step)?;
    }
    // The strict-plus-dual build of the precompiled fixture (E5 of `check-precompiled`), last: it is the only part
    // here that builds under another profile than `dev`.
    for step in precompiled_dual_steps(target_dir) {
        println!("check-dual {}: {}", step.id, step.shown());
        run_dual_step(&root, &step)?;
    }
    println!(
        "check-dual passed: tests/dual is formatted and passes clippy with -D warnings; the workspace built with --cfg gust_dual; without it the tests/dual consumer's test (E1) and binary (E3) each failed with exactly one E0277 for the missing CPU twin of the dependency `kernel-lib`, and with it (E2) the test dispatched on `gust::cpu::CpuDevice` and passed, and a production binary with a module routed to the CPU (E4) ran on `CpuDevice` with no cfg while the unrouted source (E4-control) failed with exactly one E0277, together with the overflow-mode test of a package with its own override and the two tests that dispatch an overflowing integer kernel (the dependency's twin followed the dependency's mode, the twin of the dependent's own kernel the dependent's). Last, the precompiled fixture tests/precompiled (E5 of check-precompiled) built with the cfg under GUST_PROFILE=native, its `app` test ran the CPU twin of the fixture's own kernel against the host reference, and `cargo tree -e features` of `app` showed no slang-runtime."
    );
    match mode {
        DualMode::CheckOnly => println!(
            "check-dual --check-only: needs cargo only; no slangc and no GPU was involved, and the example runs of the full mode were skipped"
        ),
        DualMode::Full => println!(
            "check-dual full mode: with --cfg gust_dual `cargo run` of vector-add and polynomial (debug profile, on the GPU) each printed the `CPU twin: matches` line, and without it vector-add printed nothing about the twin"
        ),
    }
    Ok(())
}

/// The problems of a `Fails` step (E1 or E3) for the outcome of its cargo call: none when it failed with exactly the
/// one error of the step's expectation file. Any other kind of step has no expectation.
pub(crate) fn dual_problems(
    root: &Path,
    step: &DualStep,
    success: bool,
    stdout: &str,
    stderr: &str,
) -> Result<Vec<String>, String> {
    let DualKind::Fails {
        case,
        target,
        expected,
    } = &step.kind
    else {
        return Err(format!("check-dual {}: has no expectation file", step.id));
    };
    let id = step.id;
    let text = fs::read_to_string(root.join(DUAL_DIR).join(expected))
        .map_err(|error| format!("check-dual {id}: {DUAL_DIR}/{expected}: {error}"))?;
    let wanted = parse_expected(&text)
        .map_err(|error| format!("check-dual {id}: {DUAL_DIR}/{expected}: {error}"))?;
    Ok(judge(
        case,
        target,
        &Expectation::Fails(wanted),
        success,
        stdout,
        stderr,
    ))
}

pub(crate) fn run_dual_step(root: &Path, step: &DualStep) -> Result<(), String> {
    let id = step.id;
    let launch = |error: io::Error| format!("check-dual {id}: could not launch cargo: {error}");
    match &step.kind {
        DualKind::Build | DualKind::Compiles => {
            let status = dual_command(root, step)
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .status()
                .map_err(launch)?;
            if status.success() {
                return Ok(());
            }
            let code = exit_code(status);
            if step.kind == DualKind::Compiles {
                return Err(format!(
                    "check-dual {id}: cfg did not reach `kernel-lib`, or the fixture failed to compile for another reason; see the cargo output above (the consumer's test did not compile with {DUAL_RUSTFLAGS_VARIABLE} set to `--cfg gust_dual`, cargo exit {code}). Stop: there is no fallback delivery. Run the same cargo command with {DUAL_RUSTFLAGS_VARIABLE} set and `-vv` to see the rustc flags it passed"
                ));
            }
            Err(format!("check-dual {id}: cargo failed (exit {code})"))
        }
        DualKind::Fails { .. } => {
            let output = dual_command(root, step).output().map_err(launch)?;
            let problems = dual_problems(
                root,
                step,
                output.status.success(),
                &String::from_utf8_lossy(&output.stdout),
                &String::from_utf8_lossy(&output.stderr),
            )?;
            if problems.is_empty() {
                return Ok(());
            }
            Err(format!(
                "check-dual {id} found {} problem(s):\n{}",
                problems.len(),
                problems.join("\n")
            ))
        }
        DualKind::Runs { twin, also } => {
            let output = dual_command(root, step).output().map_err(launch)?;
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            if !output.status.success() {
                // What the example printed before it stopped, then its own panic message (an unavailable
                // adapter, a failed twin assertion), which is on stderr.
                print!("{stdout}");
                eprint!("{}", String::from_utf8_lossy(&output.stderr));
                return Err(format!(
                    "check-dual {id}: cargo run failed (exit {}); the example needs a Vulkan adapter and its own assertions must hold",
                    exit_code(output.status)
                ));
            }
            // The whole output of an example is long (the generated Slang); the lines about the twin are the point.
            for line in without_colors(&stdout)
                .lines()
                .filter(|line| line.contains("CPU twin"))
            {
                println!("  {line}");
            }
            let problems = twin_run_problems(id, *twin, also, &stdout);
            if problems.is_empty() {
                return Ok(());
            }
            Err(problems.join("\n"))
        }
        DualKind::Routed { line, .. } => {
            let output = dual_command(root, step).output().map_err(launch)?;
            let problems = routed_run_problems(
                id,
                line,
                output.status.success(),
                &String::from_utf8_lossy(&output.stdout),
                &String::from_utf8_lossy(&output.stderr),
            );
            if problems.is_empty() {
                return Ok(());
            }
            Err(problems.join("\n"))
        }
        DualKind::Tree => {
            let output = dual_command(root, step).output().map_err(launch)?;
            if !output.status.success() {
                return Err(format!(
                    "check-dual {id}: cargo failed (exit {}):\n{}{}",
                    exit_code(output.status),
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
            let tree = String::from_utf8_lossy(&output.stdout).into_owned();
            let problems = crate::precompiled::tree_problems(&tree);
            if problems.is_empty() {
                return Ok(());
            }
            Err(format!(
                "check-dual {id}: cargo tree -e features -p app: {}",
                problems.join("; ")
            ))
        }
        DualKind::Passes { tests } => {
            let output = dual_command(root, step).output().map_err(launch)?;
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            print!("{stdout}");
            eprint!("{}", String::from_utf8_lossy(&output.stderr));
            if !output.status.success() {
                return Err(format!(
                    "check-dual {id}: cargo failed (exit {})",
                    exit_code(output.status)
                ));
            }
            let missing = tests_not_ok(&stdout, tests);
            if missing.is_empty() {
                return Ok(());
            }
            Err(format!(
                "check-dual {id}: cargo succeeded but did not report these tests as ok: {}",
                missing.join(", ")
            ))
        }
    }
}

#[cfg(test)]
#[path = "tests/dual.rs"]
mod dual_tests;
