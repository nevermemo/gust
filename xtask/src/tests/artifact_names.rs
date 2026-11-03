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
fn the_expected_list_is_fourteen_sorted_unique_spirv_names() {
    assert_eq!(EXPECTED_SPIRV.len(), 14);
    assert!(EXPECTED_SPIRV.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(EXPECTED_SPIRV.iter().all(|name| name.ends_with(".spv")));
    assert!(EXPECTED_SPIRV.iter().all(|name| name.contains("__")));
}
