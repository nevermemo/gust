use super::compare_artifact_names;
use crate::workspace::EXPECTED_SPIRV;

fn names(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

#[test]
fn an_exact_match_passes_in_any_order() {
    assert_eq!(
        compare_artifact_names(&["a.spv", "b.spv"], &names(&["b.spv", "a.spv"])),
        Ok(())
    );
    assert_eq!(compare_artifact_names(&[], &[]), Ok(()));
}

#[test]
fn a_missing_kernel_is_named() {
    let error =
        compare_artifact_names(&["a.spv", "b.spv", "c.spv"], &names(&["a.spv"])).unwrap_err();
    assert!(error.contains("Missing: b.spv, c.spv."), "{error}");
    assert!(!error.contains("Unexpected"), "{error}");
}

#[test]
fn an_unexpected_kernel_is_named() {
    let error =
        compare_artifact_names(&["a.spv"], &names(&["z.spv", "a.spv", "y.spv"])).unwrap_err();
    assert!(error.contains("Unexpected: y.spv, z.spv."), "{error}");
    assert!(!error.contains("Missing"), "{error}");
}

#[test]
fn a_renamed_kernel_is_one_missing_and_one_unexpected_file() {
    // The count still matches, which is exactly what the old count check could not see.
    let error =
        compare_artifact_names(&["a.spv", "b.spv"], &names(&["a.spv", "c.spv"])).unwrap_err();
    assert!(error.contains("Missing: b.spv."), "{error}");
    assert!(error.contains("Unexpected: c.spv."), "{error}");
    assert!(error.contains("deliberately for new examples"), "{error}");
}

#[test]
fn the_expected_list_is_twenty_sorted_unique_spirv_names() {
    // 14 before m7.t14 and m7.a2; component-pool's compaction adds three, and map-chain adds the three kernels of its
    // `#[gpu] fn` maps (`<function>__map_kernel`).
    assert_eq!(EXPECTED_SPIRV.len(), 20);
    for map_kernel in ["curve", "damp", "rescale"] {
        assert!(
            EXPECTED_SPIRV.contains(&format!("{map_kernel}__map_kernel.spv").as_str()),
            "{map_kernel}"
        );
    }
    assert!(EXPECTED_SPIRV.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(EXPECTED_SPIRV.iter().all(|name| name.ends_with(".spv")));
    assert!(EXPECTED_SPIRV.iter().all(|name| name.contains("__")));
}
