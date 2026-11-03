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
