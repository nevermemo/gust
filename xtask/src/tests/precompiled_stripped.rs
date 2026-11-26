use super::*;

#[test]
fn a_run_is_judged_by_its_exit_code_its_adapter_its_artifact_and_the_verified_line() {
    let good = format!(
        "precompiled vector-add\nkernel: vector_add::add\nartifact: Wgsl key=52e85c5f03f72272 bytes=476\n{ADAPTER_PREFIX}NVIDIA GeForce RTX 5090\n{VERIFIED_PREFIX}1000 elements\n{SPLIT_PREFIX}1000 elements\n{SPLIT_REFUSED_PREFIX}max_compute_workgroups_per_dimension\n"
    );
    assert_eq!(
        judge_run(true, &good, ""),
        Ok(RunReport {
            adapter: "NVIDIA GeForce RTX 5090".to_owned(),
            elements: "1000".to_owned()
        })
    );

    let problems = judge_run(false, &good, "error: boom").unwrap_err();
    assert_eq!(problems.len(), 1);
    assert!(problems[0].contains("error: boom"));

    for (broken, wanted) in [
        (good.replace(VERIFIED_PREFIX, "checked: "), "verified:"),
        (good.replace(ADAPTER_PREFIX, "device: "), "adapter:"),
        (
            good.replace("artifact: Wgsl", "artifact: Spirv"),
            "artifact: Wgsl",
        ),
        (good.replace("1000 elements", "0 elements"), "verified:"),
        // D209: a run that never launched past a limit from the split artifact is not the evidence, whatever else it printed.
        (good.replace(SPLIT_PREFIX, "pieces: "), "split:"),
        (
            good.replace(SPLIT_REFUSED_PREFIX, "refused: "),
            "split fallback:",
        ),
        (
            good.replace(&format!("{SPLIT_PREFIX}1000"), &format!("{SPLIT_PREFIX}0")),
            "split:",
        ),
        (
            good.replace("adapter: NVIDIA GeForce RTX 5090", "adapter:   "),
            "adapter:",
        ),
    ] {
        let problems = judge_run(true, &broken, "").unwrap_err();
        assert!(
            problems.iter().any(|problem| problem.contains(wanted)),
            "{wanted}: {problems:?}"
        );
    }
    assert!(judge_run(true, "", "").unwrap_err().len() >= 3);
}

#[test]
fn markers_are_counted_without_overlap_and_an_empty_needle_never_matches() {
    assert_eq!(count_occurrences(b"abcabc", b"abc"), 2);
    assert_eq!(count_occurrences(b"aaaa", b"aa"), 2);
    assert_eq!(count_occurrences(b"ab", b"abc"), 0);
    assert_eq!(count_occurrences(b"", b"a"), 0);
    assert_eq!(count_occurrences(b"abc", b""), 0);
    assert_eq!(count_occurrences(b"xxabcxx", b"abc"), 1);
    assert_eq!(count_occurrences(b"abab\0abab", b"abab"), 2);

    let mut binary = vec![0_u8; 64];
    binary.extend_from_slice(b"..gust-slang-reflect..");
    binary.extend_from_slice(&[0; 8]);
    binary.extend_from_slice(b"gust-slang-reflect");
    let counts = marker_counts(&binary, &BRIDGE_MARKERS);
    assert_eq!(
        counts,
        [
            ("gust-slang-reflect", 2),
            ("kernel.slang", 0),
            ("kernel.artifact", 0)
        ]
    );
    // The clean scan reports what occurs, the control scan what does not.
    assert_eq!(markers_present(&counts), ["`gust-slang-reflect` x2"]);
    assert_eq!(
        markers_missing(&counts),
        ["`kernel.slang`", "`kernel.artifact`"]
    );
    let nothing = marker_counts(b"no markers here", &BRIDGE_MARKERS);
    assert!(markers_present(&nothing).is_empty());
    assert_eq!(markers_missing(&nothing).len(), BRIDGE_MARKERS.len());
}

#[test]
fn the_bridge_markers_are_the_strings_of_the_gated_code_and_not_slangc() {
    assert!(BRIDGE_MARKERS.contains(&"gust-slang-reflect"));
    // The ungated `Display` of `slang::Error::CompilerNotFound` spells `slangc`, and that of
    // `reflect::Error::HelperUnavailable` spells `GUST_SLANG_REFLECT`: both are in a binary without the bridge
    // (measured), so neither can be a marker.
    assert!(!BRIDGE_MARKERS.contains(&"slangc"));
    assert!(!BRIDGE_MARKERS.contains(&"GUST_SLANG_REFLECT"));
    for marker in BRIDGE_MARKERS {
        assert!(marker.len() >= 10, "{marker} is too short to be a marker");
    }
}

const TREE: &str = "app v0.0.0 (/w/tests/precompiled/app)\n├── gust-compute v0.1.0 (/w/crates/gust)\n│   └── gust-compute-macros v0.1.0 (/w/crates/gust-macros)\n├── gust-compute-wgpu v0.1.0 (/w/crates/gust-wgpu)\n│   ├── gust-compute v0.1.0 (/w/crates/gust) (*)\n│   └── wgpu v30.0.1\n└── kernels v0.1.0 (/w/tests/precompiled/kernels)\n    └── gust-compute v0.1.0 (/w/crates/gust) (*)\n";

#[test]
fn the_tree_must_show_the_crates_and_not_the_feature() {
    assert_eq!(tree_problems(TREE), Vec::<String>::new());

    let on = TREE.replace(
        "│   └── wgpu v30.0.1\n",
        "│   ├── gust feature \"slang-runtime\"\n│   └── wgpu v30.0.1\n",
    );
    let problems = tree_problems(&on);
    assert_eq!(problems.len(), 1);
    assert!(problems[0].contains("the feature is on") && problems[0].contains("slang-runtime"));

    assert_eq!(tree_problems("").len(), 3, "an empty tree reads nothing");
    let no_wgpu = TREE.replace("gust-compute-wgpu v0.1.0", "wgpu-gust v0.1.0");
    assert_eq!(tree_problems(&no_wgpu).len(), 1);
}

#[test]
fn a_dependency_disables_its_defaults_on_one_inline_line_without_the_bridge_feature() {
    let manifest = "[dependencies]\nkernels = { path = \"../kernels\" }\ngust = { path = \"../../../crates/gust\", default-features = false }\ngust-wgpu = { path = \"../../../crates/gust-wgpu\", default-features = false }\n";
    assert!(dependency_disables_defaults(manifest, "gust"));
    assert!(dependency_disables_defaults(manifest, "gust-wgpu"));
    assert!(!dependency_disables_defaults(manifest, "kernels"));
    assert!(!dependency_disables_defaults(manifest, "wgpu"));

    // Dropping the setting, spelling the table form, adding the feature, or a comment do not count.
    let dropped = manifest.replace(", default-features = false", "");
    assert!(!dependency_disables_defaults(&dropped, "gust"));
    let table = "[dependencies.gust]\npath = \"x\"\ndefault-features = false\n";
    assert!(!dependency_disables_defaults(table, "gust"));
    let featured = manifest.replace(
        "default-features = false }\ngust-wgpu",
        "default-features = false, features = [\"slang-runtime\"] }\ngust-wgpu",
    );
    assert!(!dependency_disables_defaults(&featured, "gust"));
    assert!(dependency_disables_defaults(&featured, "gust-wgpu"));
    let commented = "# gust = { default-features = false }\n";
    assert!(!dependency_disables_defaults(commented, "gust"));
    let spaced = "gust   =   { path = \"x\" ,  default-features=false }\n";
    assert!(dependency_disables_defaults(spaced, "gust"));
}

#[test]
fn the_counted_stage_binary_is_judged_by_its_exit_code_its_adapter_and_the_counted_line() {
    let good = format!(
        "{ADAPTER_PREFIX}NVIDIA GeForce RTX 5090\n{COUNTED_PREFIX}, three counts (one above the capacity) equal the host model for 1000 rows\n{STRUCT_PREFIX}, three frames of 1000 rows equal the host model\n"
    );
    assert_eq!(
        judge_counted(true, &good, ""),
        Ok("NVIDIA GeForce RTX 5090".to_owned())
    );
    let problems = judge_counted(false, &good, "error: boom").unwrap_err();
    assert_eq!(problems.len(), 1);
    assert!(problems[0].contains("error: boom"));
    // No adapter line, or no counted line (the binary printed nothing, or a stage that did not run), is refused by name.
    let no_adapter =
        judge_counted(true, &format!("{COUNTED_PREFIX}\n{STRUCT_PREFIX}\n"), "").unwrap_err();
    assert!(
        no_adapter
            .iter()
            .any(|problem| problem.contains(ADAPTER_PREFIX))
    );
    let no_line = judge_counted(true, &format!("{ADAPTER_PREFIX}x\n"), "").unwrap_err();
    assert!(
        no_line
            .iter()
            .any(|problem| problem.contains(COUNTED_PREFIX))
    );
    // A binary that ran the counted stage but not the struct stage is refused by the struct line's name.
    let no_struct =
        judge_counted(true, &format!("{ADAPTER_PREFIX}x\n{COUNTED_PREFIX}\n"), "").unwrap_err();
    assert_eq!(no_struct.len(), 1);
    assert!(no_struct[0].contains(STRUCT_PREFIX));
    assert!(judge_counted(true, "", "").unwrap_err().len() >= 3);
    // The binary sits beside `app` in the same release directory.
    let app = std::path::Path::new("target")
        .join("release")
        .join(format!("app{}", std::env::consts::EXE_SUFFIX));
    assert!(counted_binary_of(&app).ends_with(format!("counted{}", std::env::consts::EXE_SUFFIX)));
}
