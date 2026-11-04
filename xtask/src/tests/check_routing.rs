use super::{ChangedCheck, select_changed_checks};

fn select(paths: &[&str]) -> Vec<ChangedCheck> {
    select_changed_checks(
        &paths
            .iter()
            .map(|path| (*path).to_owned())
            .collect::<Vec<_>>(),
    )
}

#[test]
fn reflection_module_keeps_native_contract_checks() {
    assert_eq!(
        select(&["crates/gust/src/reflect/mod.rs"]),
        vec![ChangedCheck::Feature("reflection")]
    );
}

#[test]
fn mixed_loop_and_pool_changes_keep_both_scopes() {
    assert_eq!(
        select(&["tests/fixtures/loops.slang", "crates/gust-wgpu/src/pool.rs"]),
        vec![
            ChangedCheck::Feature("loops"),
            ChangedCheck::Feature("wgpu")
        ]
    );
}

#[test]
fn macro_changes_do_not_mask_ignored_example_tests() {
    assert_eq!(
        select(&[
            "crates/gust-macros/src/slang/mod.rs",
            "examples/component-pool/src/app.rs"
        ]),
        vec![
            ChangedCheck::Feature("macro"),
            ChangedCheck::Feature("examples")
        ]
    );
}

#[test]
fn macro_changes_with_the_workgroup_fixture_select_the_macro_suite_and_workgroup() {
    assert_eq!(
        select(&[
            "crates/gust-macros/src/validate/barrier.rs",
            "tests/fixtures/workgroup.rs"
        ]),
        vec![
            ChangedCheck::Feature("workgroup"),
            ChangedCheck::Feature("macro")
        ]
    );
}

#[test]
fn a_fixture_alone_selects_only_its_feature() {
    assert_eq!(
        select(&["tests/fixtures/workgroup.slang"]),
        vec![ChangedCheck::Feature("workgroup")]
    );
}

// ---- the paths of M7 (m7.x1) ----

#[test]
fn the_algorithm_crate_and_its_goldens_select_the_algo_area() {
    // Before m7.x1 the crate fell through to the workspace tests. component-pool depends on it, so the examples are
    // built and tested beside the area.
    assert_eq!(
        select(&["crates/gust-algo/src/scan.rs"]),
        vec![
            ChangedCheck::Feature("algo"),
            ChangedCheck::Feature("examples")
        ]
    );
    for golden in [
        "tests/fixtures/reduce.slang",
        "tests/fixtures/scan.rs",
        "tests/fixtures/compact.slang",
    ] {
        assert_eq!(
            select(&[golden]),
            vec![ChangedCheck::Feature("algo")],
            "{golden}"
        );
    }
}

#[test]
fn the_map_sources_and_tests_select_the_map_area_beside_the_broader_check() {
    assert_eq!(
        select(&["tests/fixtures/map.slang"]),
        vec![ChangedCheck::Feature("map")]
    );
    // The source keeps the broader check it had: core for gust, macro for gust-macros, wgpu for gust-wgpu.
    for (path, broader) in [
        ("crates/gust/src/map.rs", "core"),
        ("crates/gust/src/cpu/parallel.rs", "core"),
        ("crates/gust/tests/parallel.rs", "core"),
        ("crates/gust-macros/src/expand/map.rs", "macro"),
        ("crates/gust-macros/src/tests/map.rs", "macro"),
        ("crates/gust-wgpu/src/map_host.rs", "wgpu"),
        ("crates/gust-wgpu/tests/map.rs", "wgpu"),
        ("crates/gust-wgpu/tests/gpu_vec.rs", "wgpu"),
        ("crates/gust-wgpu/tests/device_options.rs", "wgpu"),
    ] {
        assert_eq!(
            select(&[path]),
            vec![ChangedCheck::Feature("map"), ChangedCheck::Feature(broader)],
            "{path}"
        );
    }
}

#[test]
fn the_par_for_and_grid_sources_and_tests_select_the_dispatch_area_beside_the_broader_check() {
    for golden in [
        "tests/fixtures/par_for.rs",
        "tests/fixtures/par_for.slang",
        "tests/fixtures/grid.rs",
        "tests/fixtures/grid.slang",
    ] {
        assert_eq!(
            select(&[golden]),
            vec![ChangedCheck::Feature("dispatch")],
            "{golden}"
        );
    }
    for (path, broader) in [
        ("crates/gust/src/dispatch.rs", "core"),
        ("crates/gust-macros/src/par_for.rs", "macro"),
        ("crates/gust-macros/src/tests/par_for.rs", "macro"),
        ("crates/gust-macros/src/tests/grid.rs", "macro"),
        ("crates/gust-wgpu/tests/par_for.rs", "wgpu"),
        ("crates/gust-wgpu/tests/grid.rs", "wgpu"),
        ("crates/gust-wgpu/tests/concurrency.rs", "wgpu"),
        ("crates/gust-wgpu/tests/async_readback.rs", "wgpu"),
    ] {
        assert_eq!(
            select(&[path]),
            vec![
                ChangedCheck::Feature("dispatch"),
                ChangedCheck::Feature(broader)
            ],
            "{path}"
        );
    }
}

#[test]
fn the_workgroup_validator_and_its_tests_select_the_workgroup_area_beside_the_macro_suite() {
    for path in [
        "crates/gust-macros/src/validate/barrier.rs",
        "crates/gust-macros/src/tests/workgroup.rs",
    ] {
        assert_eq!(
            select(&[path]),
            vec![
                ChangedCheck::Feature("workgroup"),
                ChangedCheck::Feature("macro")
            ],
            "{path}"
        );
    }
}

#[test]
fn other_fixtures_and_look_alike_names_keep_the_macro_suite() {
    // A fixture belongs to an area by its stem, not by a shared prefix: `mapping` and `grid_extra` are not `map`
    // and `grid`; the older fixtures route as before.
    for path in [
        "tests/fixtures/vectors.slang",
        "tests/fixtures/mismatch.rs",
        "tests/fixtures/mapping.rs",
        "tests/fixtures/grid_extra.rs",
        "tests/fixtures/scanner.slang",
    ] {
        assert_eq!(
            select(&[path]),
            vec![ChangedCheck::Feature("macro")],
            "{path}"
        );
    }
    // The crate directories of the algorithm crate do not capture its neighbours.
    assert_eq!(
        select(&["crates/gust-wgpu/src/pool.rs"]),
        vec![ChangedCheck::Feature("wgpu")]
    );
}

#[test]
fn mixed_m7_paths_keep_every_scope_in_a_fixed_order() {
    assert_eq!(
        select(&[
            "crates/gust-algo/src/reduce.rs",
            "crates/gust/src/map.rs",
            "crates/gust-wgpu/tests/par_for.rs"
        ]),
        vec![
            ChangedCheck::Feature("algo"),
            ChangedCheck::Feature("map"),
            ChangedCheck::Feature("dispatch"),
            ChangedCheck::Feature("wgpu"),
            ChangedCheck::Feature("core"),
            ChangedCheck::Feature("examples")
        ]
    );
}

#[test]
fn a_fixture_of_another_golden_keeps_the_macro_suite_beside_an_m7_area() {
    // The pair of the review (M1): the vectors golden is pinned by gust-macros tests, and the map source selecting
    // its area must not hide that.
    assert_eq!(
        select(&["crates/gust/src/map.rs", "tests/fixtures/vectors.slang"]),
        vec![
            ChangedCheck::Feature("map"),
            ChangedCheck::Feature("macro"),
            ChangedCheck::Feature("core")
        ]
    );
    // An unclaimed fixture alone, and beside the older fixture rules and the other M7 sources.
    assert_eq!(
        select(&["tests/fixtures/products.rs"]),
        vec![ChangedCheck::Feature("macro")]
    );
    assert_eq!(
        select(&["tests/fixtures/loops.rs", "tests/fixtures/vectors.slang"]),
        vec![
            ChangedCheck::Feature("loops"),
            ChangedCheck::Feature("macro")
        ]
    );
    assert_eq!(
        select(&[
            "crates/gust-wgpu/tests/par_for.rs",
            "tests/fixtures/uniforms.slang"
        ]),
        vec![
            ChangedCheck::Feature("dispatch"),
            ChangedCheck::Feature("macro"),
            ChangedCheck::Feature("wgpu")
        ]
    );
    // Fixtures that rules claim (M7 stems, workgroup) add no macro suite by themselves.
    assert_eq!(
        select(&[
            "tests/fixtures/map.slang",
            "tests/fixtures/grid.rs",
            "tests/fixtures/workgroup.rs"
        ]),
        vec![
            ChangedCheck::Feature("workgroup"),
            ChangedCheck::Feature("map"),
            ChangedCheck::Feature("dispatch")
        ]
    );
}

#[test]
fn xtask_changes_still_select_only_the_fast_check() {
    assert_eq!(
        select(&["xtask/src/checks.rs", "crates/gust-algo/src/scan.rs"]),
        vec![ChangedCheck::Fast]
    );
}
