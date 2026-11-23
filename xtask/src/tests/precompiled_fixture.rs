use super::*;

#[test]
fn a_scratch_copy_points_its_path_dependencies_at_the_repository() {
    let manifest = "[dependencies]\nkernels = { path = \"../kernels\" }\ngust = { version = \"0.1.0\", path = \"../../../crates/gust\" }\n";
    let retargeted = retarget_paths(manifest, "C:/work space/repo");
    assert!(retargeted.contains("path = \"C:/work space/repo/crates/gust\""));
    assert!(retargeted.contains("kernels = { path = \"../kernels\" }"));
    assert!(!retargeted.contains("../../../"));
    assert_eq!(retarget_paths("[package]\n", "/r"), "[package]\n");
}
