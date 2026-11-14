use std::{
    fs,
    path::{Path, PathBuf},
};

use super::{
    ENTRIES, Listing, Parts, anchor_names, anchored_region, check_listings, listing_matches,
    output_has_line, path_without_slang, tail, tree_difference, workspace_members,
};
use crate::workspace::workspace_root;

fn args(words: &[&str]) -> Vec<String> {
    words.iter().map(|word| (*word).to_owned()).collect()
}

/// A file of the repository, with LF endings.
fn repository_file(path: &str) -> String {
    let root = workspace_root().expect("the repository root");
    fs::read_to_string(root.join(path))
        .unwrap_or_else(|error| panic!("{path}: {error}"))
        .replace("\r\n", "\n")
}

#[test]
fn no_flag_runs_all_three_parts_and_each_flag_only_its_own() {
    assert_eq!(Parts::parse(&[]), Ok(Parts::ALL));
    assert_eq!(
        Parts::parse(&args(&["--slang"])),
        Ok(Parts {
            slang: true,
            no_slang: false,
            gpu: false
        })
    );
    assert_eq!(
        Parts::parse(&args(&["--no-slang"])),
        Ok(Parts {
            slang: false,
            no_slang: true,
            gpu: false
        })
    );
    assert_eq!(
        Parts::parse(&args(&["--gpu"])),
        Ok(Parts {
            slang: false,
            no_slang: false,
            gpu: true
        })
    );
    // Order and repetition do not matter; naming all three is the default.
    assert_eq!(
        Parts::parse(&args(&["--gpu", "--slang", "--gpu"])),
        Ok(Parts {
            slang: true,
            no_slang: false,
            gpu: true
        })
    );
    assert_eq!(
        Parts::parse(&args(&["--gpu", "--no-slang", "--slang"])),
        Ok(Parts::ALL)
    );
}

#[test]
fn an_unknown_argument_is_refused_and_named() {
    for bad in ["--skip-gpu", "slang", "--GPU", "", "--slang=1"] {
        let error = Parts::parse(&args(&["--slang", bad])).unwrap_err();
        assert!(error.contains(&format!("`{bad}`")), "{error}");
    }
}

#[test]
fn the_names_of_the_parts_say_which_run_and_which_do_not() {
    let only_gpu = Parts::parse(&args(&["--gpu"])).unwrap();
    assert_eq!(only_gpu.run_names(), ["gpu"]);
    assert_eq!(only_gpu.not_run_names(), ["slang", "no-slang"]);
    assert_eq!(Parts::ALL.run_names(), ["slang", "no-slang", "gpu"]);
    assert!(Parts::ALL.not_run_names().is_empty());
}

const SOURCE: &str = "\
use a;
// ANCHOR: first
fn one() {}

    // ANCHOR: nested
    let x = 1;
    // ANCHOR_END: nested
fn two() {}
// ANCHOR_END: first
// ANCHOR: second
fn three() {}
// ANCHOR_END: second
";

#[test]
fn a_region_is_the_lines_between_its_anchors_and_nested_anchor_lines_are_dropped() {
    assert_eq!(
        anchored_region(SOURCE, "first").unwrap(),
        "fn one() {}\n\n    let x = 1;\nfn two() {}\n"
    );
    assert_eq!(
        anchored_region(SOURCE, "nested").unwrap(),
        "    let x = 1;\n"
    );
    assert_eq!(
        anchored_region(SOURCE, "second").unwrap(),
        "fn three() {}\n"
    );
    assert_eq!(anchor_names(SOURCE).unwrap(), ["first", "nested", "second"]);
}

#[test]
fn a_crlf_source_gives_the_same_region_as_an_lf_source() {
    let crlf = SOURCE.replace('\n', "\r\n");
    for name in ["first", "nested", "second"] {
        assert_eq!(
            anchored_region(&crlf, name).unwrap(),
            anchored_region(SOURCE, name).unwrap(),
            "{name}"
        );
    }
}

#[test]
fn an_empty_region_is_empty_and_an_unknown_name_is_an_error() {
    let text = "// ANCHOR: nothing\n// ANCHOR_END: nothing\n";
    assert_eq!(anchored_region(text, "nothing").unwrap(), "");
    assert_eq!(
        anchored_region(SOURCE, "third").unwrap_err(),
        "no anchor `third`"
    );
    assert_eq!(
        anchor_names("fn main() {}\n").unwrap(),
        Vec::<String>::new()
    );
}

#[test]
fn malformed_anchors_are_errors_with_the_line_or_the_name() {
    assert_eq!(
        anchor_names("// ANCHOR: a\nfn f() {}\n").unwrap_err(),
        "anchor `a` is never closed"
    );
    assert_eq!(
        anchor_names("fn f() {}\n// ANCHOR_END: a\n").unwrap_err(),
        "line 2: ANCHOR_END `a` has no open ANCHOR (missing, closed twice, or before its start)"
    );
    assert_eq!(
        anchor_names("// ANCHOR: a\n// ANCHOR_END: a\n// ANCHOR_END: a\n").unwrap_err(),
        "line 3: ANCHOR_END `a` has no open ANCHOR (missing, closed twice, or before its start)"
    );
    assert_eq!(
        anchor_names("// ANCHOR: a\n// ANCHOR_END: a\n// ANCHOR: a\n// ANCHOR_END: a\n")
            .unwrap_err(),
        "line 3: anchor `a` is declared twice"
    );
    assert_eq!(
        anchor_names("// ANCHOR: a b\n").unwrap_err(),
        "line 1: an anchor name is one word with no spaces (got `a b`)"
    );
    assert_eq!(
        anchor_names("// ANCHOR: \n").unwrap_err(),
        "line 1: an anchor name is one word with no spaces (got ``)"
    );
    assert_eq!(
        anchor_names("// ANCHOR:a\n").unwrap_err(),
        "line 1: write one space after the colon of an anchor line (got `a`)"
    );
    assert_eq!(
        anchor_names("// ANCHOR_END:a\n").unwrap_err(),
        "line 1: write one space after the colon of an anchor line (got `a`)"
    );
    // A misspelt end is not silently accepted as text of the region.
    assert!(anchor_names("// ANCHOR: a\n// ANCHOR_END: b\n").is_err());
    // A region of a malformed file is refused even when its own anchors are fine.
    assert!(anchored_region("// ANCHOR: a\n// ANCHOR_END: a\n// ANCHOR: b\n", "a").is_err());
}

#[test]
fn the_gallery_sources_have_well_formed_anchors_with_the_regions_the_guide_will_show() {
    let source = repository_file("gallery/hello-gust/src/main.rs");
    assert_eq!(
        anchor_names(&source).unwrap(),
        ["imports", "kernel", "main", "tests"]
    );
    let kernel = anchored_region(&source, "kernel").unwrap();
    assert!(kernel.starts_with("#[gpu]\nmod vector_add {\n"), "{kernel}");
    assert!(kernel.ends_with("    }\n}\n"), "{kernel}");
    let main = anchored_region(&source, "main").unwrap();
    assert!(main.starts_with("fn main() {\n"), "{main}");
    assert!(
        main.contains("println!(\"hello-gust: {result:?}\");"),
        "{main}"
    );
}

#[test]
fn a_listing_matches_only_when_every_byte_is_the_region() {
    let region = "fn one() {}\n";
    assert!(listing_matches(b"fn one() {}\n", region));
    assert!(!listing_matches(b"fn one() {}", region), "no final newline");
    assert!(!listing_matches(b"fn one() {}\r\n", region), "CRLF");
    assert!(!listing_matches(b"fn one() { }\n", region));
    assert!(!listing_matches(b"fn two() {}\n", region), "same length");
    assert!(!listing_matches(b"", region));
}

fn scratch(name: &str) -> PathBuf {
    let directory =
        std::env::temp_dir().join(format!("gust-gallery-test-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).unwrap();
    directory
}

fn write(root: &Path, relative: &str, text: &str) {
    fs::write(root.join(relative), text).unwrap();
}

#[test]
fn the_listing_check_counts_matches_and_names_the_file_that_drifted() {
    let root = scratch("listings");
    write(&root, "source.rs", SOURCE);
    write(&root, "good.rs", "fn three() {}\n");
    write(&root, "stale.rs", "fn three() { 1 }\n");
    let good = Listing {
        packaged: "good.rs",
        source: "source.rs",
        anchor: "second",
    };
    let stale = Listing {
        packaged: "stale.rs",
        source: "source.rs",
        anchor: "second",
    };
    let missing_anchor = Listing {
        packaged: "good.rs",
        source: "source.rs",
        anchor: "nope",
    };
    let missing_file = Listing {
        packaged: "absent.rs",
        source: "source.rs",
        anchor: "second",
    };
    assert_eq!(check_listings(&root, &[]), Ok(0));
    assert_eq!(check_listings(&root, &[good]), Ok(1));
    let error = check_listings(&root, &[stale]).unwrap_err();
    assert!(
        error.contains("stale.rs is not byte-identical to region `second` of source.rs"),
        "{error}"
    );
    assert!(
        check_listings(&root, &[missing_anchor])
            .unwrap_err()
            .contains("no anchor `nope`")
    );
    assert!(
        check_listings(&root, &[missing_file])
            .unwrap_err()
            .contains("absent.rs")
    );
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn the_list_of_entries_is_the_members_of_the_gallery_workspace() {
    let manifest = repository_file("gallery/Cargo.toml");
    let mut members = workspace_members(&manifest).unwrap();
    members.sort();
    let mut entries = ENTRIES
        .iter()
        .map(|entry| entry.dir.to_owned())
        .collect::<Vec<_>>();
    entries.sort();
    assert_eq!(members, entries);
    assert_eq!(entries, ["hello-gust"]);
}

#[test]
fn workspace_members_reads_a_multi_line_array_and_ignores_comments() {
    let manifest = "# members = [\"no\"]\n[workspace]\nmembers = [\n    \"a\", # one\n    \"b\",\n]\nresolver = \"3\"\n";
    // The comment line before `[workspace]` is skipped like any line that does not start with `members`.
    assert_eq!(workspace_members(manifest).unwrap(), ["a", "b"]);
    assert!(workspace_members("[workspace]\nresolver = \"3\"\n").is_err());
}

#[test]
fn slang_folders_are_dropped_from_the_path_and_the_others_kept_in_order() {
    let folders = ["/a", "/slang/bin", "/b", "/other/slang/bin"]
        .map(PathBuf::from)
        .to_vec();
    let kept = path_without_slang(&folders, |folder| {
        folder.to_string_lossy().contains("slang")
    });
    assert_eq!(kept, [PathBuf::from("/a"), PathBuf::from("/b")]);
    assert_eq!(path_without_slang(&folders, |_| false), folders);
    assert!(path_without_slang(&folders, |_| true).is_empty());
}

#[test]
fn a_folder_with_a_slangc_is_recognised_in_either_spelling_and_a_neighbour_is_not() {
    let root = scratch("slangc");
    for (folder, file) in [
        ("plain", "slangc"),
        ("exe", "slangc.exe"),
        ("upper", "SLANGC.EXE"),
        ("none", "slangd"),
        ("also", "slangc.txt"),
    ] {
        fs::create_dir_all(root.join(folder)).unwrap();
        write(&root.join(folder), file, "");
    }
    let held = |folder: &str| super::holds_slangc(&root.join(folder));
    assert!(held("plain") && held("exe") && held("upper"));
    assert!(!held("none") && !held("also"));
    assert!(!held("does-not-exist"));
    let _ = fs::remove_dir_all(&root);
}

fn files(items: &[(&str, &str)]) -> Vec<(String, Vec<u8>)> {
    items
        .iter()
        .map(|(name, bytes)| ((*name).to_owned(), bytes.as_bytes().to_vec()))
        .collect()
}

#[test]
fn artifact_trees_differ_by_name_set_or_by_one_byte() {
    let committed = files(&[("a.wgsl", "x"), ("manifest.json", "{}")]);
    assert_eq!(tree_difference(&committed, &committed), None);
    assert!(
        tree_difference(&committed, &files(&[("a.wgsl", "x")]))
            .unwrap()
            .contains("file lists differ")
    );
    assert!(
        tree_difference(
            &committed,
            &files(&[("b.wgsl", "x"), ("manifest.json", "{}")])
        )
        .unwrap()
        .contains("file lists differ")
    );
    assert_eq!(
        tree_difference(
            &committed,
            &files(&[("a.wgsl", "y"), ("manifest.json", "{}")])
        )
        .as_deref(),
        Some("a.wgsl differs from the fresh build")
    );
    assert_eq!(
        tree_difference(
            &committed,
            &files(&[("a.wgsl", "x"), ("manifest.json", "{} ")])
        )
        .as_deref(),
        Some("manifest.json differs from the fresh build")
    );
}

#[test]
fn the_expected_line_must_be_a_whole_line_of_the_output() {
    let line = ENTRIES[0].expected_line;
    assert!(output_has_line(&format!("{line}\n"), line));
    assert!(output_has_line(&format!("noise\n{line}  \r\nmore\n"), line));
    assert!(!output_has_line(&format!("prefix {line}\n"), line));
    assert!(!output_has_line(
        "hello-gust: [11.0, 22.0, 33.0, 45.0]\n",
        line
    ));
    assert!(!output_has_line("", line));
}

#[test]
fn tail_keeps_the_last_lines() {
    assert_eq!(tail("a\nb\nc\nd", 2), "c\nd");
    assert_eq!(tail("a", 5), "a");
    assert_eq!(tail("", 3), "");
}

/// The text of one job of ci.yml: from its header line to the next job.
fn job(ci: &str, name: &str) -> String {
    let start = ci.find(&format!("\n  {name}:\n")).expect(name) + 1;
    let rest = &ci[start..];
    let end = rest
        .match_indices("\n  ")
        .find(|(index, _)| {
            rest[index + 3..]
                .chars()
                .next()
                .is_some_and(|c| c != ' ' && c != '#' && c != '\n')
        })
        .map_or(rest.len(), |(index, _)| index);
    rest[..end].to_owned()
}

#[test]
fn ci_runs_the_slang_part_in_the_slang_job_and_the_no_slang_part_in_the_unit_job_only() {
    let ci = repository_file(".github/workflows/ci.yml");
    let slang = job(&ci, "slang");
    let unit = job(&ci, "unit");
    assert!(
        slang.contains("        run: cargo --locked xtask check-gallery --slang\n"),
        "{slang}"
    );
    assert!(
        unit.contains("        run: cargo --locked xtask check-gallery --no-slang\n"),
        "{unit}"
    );
    assert!(
        !slang.contains("--no-slang") && !slang.contains("--gpu"),
        "{slang}"
    );
    assert!(
        !unit.contains("check-gallery --slang") && !unit.contains("--gpu"),
        "{unit}"
    );
    // Two steps in the whole workflow, no GPU part anywhere (the lavapipe step is a later slice).
    let steps = ci
        .lines()
        .filter(|line| line.trim_start().starts_with("run:") && line.contains("check-gallery"))
        .count();
    assert_eq!(steps, 2, "{ci}");
    assert!(!ci.contains("check-gallery --gpu"));
    // The slang job installs the Slang before the gallery step uses it.
    let install = slang.find("gust slang install").expect("install step");
    let gallery = slang.find("check-gallery --slang").expect("gallery step");
    assert!(install < gallery);
}

#[test]
fn check_full_runs_only_the_gpu_part_of_the_gallery() {
    let source = repository_file("xtask/src/verify.rs");
    assert!(source.contains("check_gallery_gpu_report()"));
    assert!(!source.contains("check_gallery_report"));
    let gallery = repository_file("xtask/src/gallery.rs");
    let gpu = gallery
        .split("pub(crate) fn check_gallery_gpu_report()")
        .nth(1)
        .expect("check_gallery_gpu_report");
    let body = &gpu[..gpu.find("\n}\n").expect("end")];
    assert!(
        body.contains("slang: false")
            && body.contains("no_slang: false")
            && body.contains("gpu: true")
    );
}
