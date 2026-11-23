use super::*;

#[test]
fn e7_alternates_unset_and_changed_values_and_observes_one_artifact_each_time() {
    let steps = e7_steps();
    assert_eq!(steps[0].expect, Expansion::Any);
    assert_eq!(steps[1].expect, Expansion::Fresh);
    assert_eq!(steps[1].export, None);
    assert_eq!(
        (steps[2].export, steps[2].expect),
        (Some("a"), Expansion::Again)
    );
    assert_eq!(
        (steps[3].export, steps[3].expect),
        (Some("a"), Expansion::Fresh)
    );
    assert_eq!(
        (steps[4].export, steps[4].expect),
        (Some("b"), Expansion::Again)
    );
    assert_eq!((steps[5].export, steps[5].expect), (None, Expansion::Again));
    // A fresh step follows a step with the same value; an expanding step follows one with another value.
    for pair in steps.windows(2) {
        match pair[1].expect {
            Expansion::Fresh => assert_eq!(pair[0].export, pair[1].export),
            Expansion::Again => assert_ne!(pair[0].export, pair[1].export),
            Expansion::Any => {}
        }
    }
}

#[test]
fn the_expansion_is_read_from_the_fresh_flag_of_the_kernels_artifact() {
    let stdout = concat!(
        "{\"reason\":\"compiler-artifact\",\"target\":{\"name\":\"gust\"},\"fresh\":false}\n",
        "{\"reason\":\"compiler-artifact\",\"target\":{\"name\":\"kernels\"},\"fresh\":true}\n",
        "not json\n",
        "{\"reason\":\"build-finished\",\"success\":true}\n"
    );
    assert_eq!(artifact_fresh_flags(stdout, "kernels"), [true]);
    assert_eq!(artifact_fresh_flags(stdout, "gust"), [false]);
    assert!(artifact_fresh_flags(stdout, "app").is_empty());
    let steps = e7_steps();
    assert!(expansion_problem(&steps[1], &[true]).is_none());
    assert!(
        expansion_problem(&steps[1], &[false])
            .unwrap()
            .contains("compiled again")
    );
    assert!(expansion_problem(&steps[2], &[false]).is_none());
    assert!(
        expansion_problem(&steps[2], &[true])
            .unwrap()
            .contains("fresh")
    );
    assert!(expansion_problem(&steps[0], &[true]).is_none());
    assert!(expansion_problem(&steps[0], &[false]).is_none());
    // No message, or two, is a blind observation, not a pass.
    assert!(expansion_problem(&steps[2], &[]).unwrap().contains("blind"));
    assert!(
        expansion_problem(&steps[1], &[true, true])
            .unwrap()
            .contains("blind")
    );
}

#[test]
fn two_builds_differ_when_one_byte_or_one_file_does() {
    use std::collections::BTreeMap;
    let first: BTreeMap<String, Vec<u8>> =
        [("a".to_owned(), vec![1, 2]), ("b".to_owned(), vec![3])].into();
    assert!(snapshot_differences(&first, &first.clone()).is_empty());
    let mut changed = first.clone();
    changed.insert("a".to_owned(), vec![1, 9]);
    assert_eq!(snapshot_differences(&first, &changed).len(), 1);
    let mut extra = first.clone();
    extra.insert("c".to_owned(), Vec::new());
    assert!(snapshot_differences(&first, &extra)[0].contains("only in the second"));
    let mut fewer = first.clone();
    fewer.remove("b");
    assert!(snapshot_differences(&first, &fewer)[0].contains("only in the first"));
}

#[test]
fn a_gust_build_run_that_failed_or_pruned_is_a_problem() {
    assert!(gust_build_problems("run 1", true, "gust: wrote 4 files\n", "").is_empty());
    assert_eq!(gust_build_problems("run 1", false, "", "boom").len(), 1);
    let pruned = gust_build_problems("run 2", true, "gust: pruned 1 stale file\n", "");
    assert_eq!(pruned.len(), 1);
    assert!(pruned[0].contains("pruned"));
}

#[test]
fn the_files_written_by_cargo_gust_are_read_from_its_summary_line() {
    let line = "gust: 1 kernel(s) (wgsl), Slang 2026.13.1: 3 file(s) written, 0 unchanged\n";
    assert_eq!(files_written(line), Some(3));
    assert_eq!(
        files_written("gust: x: 0 file(s) written, 3 unchanged"),
        Some(0)
    );
    assert_eq!(files_written("gust: compiling 1 kernel(s)\n"), None);
}
