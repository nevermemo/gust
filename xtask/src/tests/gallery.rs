use std::{
    ffi::{OsStr, OsString},
    fs,
    path::{Path, PathBuf},
};

use super::{
    ENTRIES, LISTINGS, Listing, Parts, TWIN_RULES, WEB_HELLO, adapter_name, anchor_names,
    anchored_region, browser_skipped_line, check_entry_tests, check_listings, folder_holds,
    lf_line_ends, listing_matches, no_slang_environment, output_has_line, required_files,
    shared_folder_problems, slang_home_problem, slangc_cannot_be_found, tail, tree_difference,
    tree_has_slang_runtime, workspace_members,
};
use crate::{
    precompiled::{StripReport, child_environment, directory_has_slang_tool, slang_free_problems},
    verify::GALLERY_BROWSER_SKIP,
    workspace::workspace_root,
};

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
    assert_eq!(
        entries,
        [
            "hello-gust",
            "histogram",
            "image-threshold",
            "particles",
            "reduce-scan",
            "routed",
            "saxpy-uniform",
            "state-machine",
            "variants",
            "web-hello"
        ]
    );
}

#[test]
fn only_variants_runs_a_second_time_under_a_profile_and_its_two_lines_name_their_bodies() {
    for entry in &ENTRIES {
        match (&entry.profile_run, entry.dir) {
            (None, "variants") => panic!("variants must run under native+subgroups too"),
            (None, _) => {}
            (Some(run), dir) => {
                assert_eq!(dir, "variants", "only variants has a second run");
                assert_eq!(run.profile, "native+subgroups");
                assert!(
                    entry.expected_line.starts_with("variants: body=portable, "),
                    "{}",
                    entry.expected_line
                );
                assert!(
                    run.expected_line.starts_with("variants: body=subgroups, "),
                    "{}",
                    run.expected_line
                );
                // Both bodies compute the same three cells, so the lines agree after the body name.
                assert_eq!(
                    entry
                        .expected_line
                        .trim_start_matches("variants: body=portable, "),
                    run.expected_line
                        .trim_start_matches("variants: body=subgroups, ")
                );
            }
        }
    }
}

#[test]
fn workspace_members_reads_a_multi_line_array_and_ignores_comments() {
    let manifest = "# members = [\"no\"]\n[workspace]\nmembers = [\n    \"a\", # one\n    \"b\",\n]\nresolver = \"3\"\n";
    // The comment line before `[workspace]` is skipped like any line that does not start with `members`.
    assert_eq!(workspace_members(manifest).unwrap(), ["a", "b"]);
    assert!(workspace_members("[workspace]\nresolver = \"3\"\n").is_err());
}

fn environment(pairs: &[(&str, &OsStr)]) -> Vec<(OsString, OsString)> {
    pairs
        .iter()
        .map(|(name, value)| (OsString::from(*name), (*value).to_owned()))
        .collect()
}

fn never_found(_: &[(OsString, OsString)]) -> Result<(), String> {
    Ok(())
}

/// A scratch tree: `slang/` holds a `slangc`, `helper/` a `gust-slang-reflect`, `shared/` a `slangc` and a `cargo`,
/// `plain/` a `cargo`; `home/` is empty. The path is `plain`, `slang`, `helper`.
fn fake_machine(name: &str) -> (PathBuf, OsString) {
    let root = scratch(name);
    for (folder, files) in [
        ("slang", &["slangc", "slangc.exe"][..]),
        (
            "helper",
            &["gust-slang-reflect", "gust-slang-reflect.exe"][..],
        ),
        (
            "shared",
            &["slangc", "slangc.exe", "cargo", "cargo.exe"][..],
        ),
        ("plain", &["cargo", "cargo.exe"][..]),
        ("home", &[][..]),
    ] {
        fs::create_dir_all(root.join(folder)).unwrap();
        for file in files {
            write(&root.join(folder), file, "");
        }
    }
    let path =
        std::env::join_paths(["plain", "slang", "helper"].map(|folder| root.join(folder))).unwrap();
    (root, path)
}

#[test]
fn the_slang_folders_and_variables_are_taken_out_and_the_rest_is_kept() {
    let (root, path) = fake_machine("strip");
    let parent = environment(&[
        ("PATH", &path),
        ("KEEP", OsStr::new("1")),
        ("GUST_SLANG_HOME", OsStr::new("/cache")),
        ("VULKAN_SDK", OsStr::new("/sdk")),
        ("VK_SDK_PATH", OsStr::new("/sdk")),
        ("GUST_EXPORT_DIR", OsStr::new("/export")),
        ("MY_SLANG_THING", OsStr::new("x")),
        ("GUST_PROFILE", OsStr::new("dev")),
    ]);
    let (child, report) = no_slang_environment(
        &parent,
        &root.join("home"),
        &directory_has_slang_tool,
        &folder_holds,
        &never_found,
    )
    .unwrap();
    let value = |name: &str| {
        child
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.clone())
    };
    assert_eq!(
        std::env::split_paths(&value("PATH").unwrap()).collect::<Vec<_>>(),
        [root.join("plain")]
    );
    assert_eq!(value("KEEP"), Some(OsString::from("1")));
    assert_eq!(value("GUST_PROFILE"), Some(OsString::from("native")));
    for name in [
        "VULKAN_SDK",
        "VK_SDK_PATH",
        "GUST_EXPORT_DIR",
        "MY_SLANG_THING",
    ] {
        assert_eq!(value(name), None, "{name}");
    }
    // The only Slang variable left is the empty scratch home.
    assert_eq!(
        value("GUST_SLANG_HOME"),
        Some(root.join("home").into_os_string())
    );
    assert_eq!(
        report.directories,
        [root.join("slang"), root.join("helper")]
    );
    assert_eq!(report.variables.len(), 5, "{:?}", report.variables);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn the_post_check_can_fail_when_the_stripping_missed_a_slang_folder() {
    let (root, path) = fake_machine("missed");
    let parent = environment(&[("PATH", &path)]);
    // A stripping that lies (it sees no Slang tool anywhere) leaves the folders in; `slang_free_problems` with the
    // honest predicate must then name them.
    let mut report = StripReport::default();
    let child = child_environment(&parent, &|_: &Path| false, &mut report);
    assert!(report.directories.is_empty());
    let problems = slang_free_problems(&child, &directory_has_slang_tool);
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(
        problems.iter().all(|p| p.contains("holds a Slang tool")),
        "{problems:?}"
    );
    // The assembled check fails the same way when the independent lookup finds slangc.
    let finds = |_: &[(OsString, OsString)]| Err("`slangc` is found".to_owned());
    let error = no_slang_environment(
        &parent,
        &root.join("home"),
        &directory_has_slang_tool,
        &folder_holds,
        &finds,
    )
    .unwrap_err();
    assert!(error.contains("`slangc` is found"), "{error}");
    // And when the predicate used for the stripping misses the folders (the post-check uses the same one, so only the
    // lookup can catch this): the real lookup is exercised in the next test.
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn the_real_lookup_finds_a_slangc_that_a_missed_stripping_leaves_on_the_path() {
    let (root, path) = fake_machine("lookup");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for file in ["slang/slangc", "shared/slangc"] {
            fs::set_permissions(root.join(file), fs::Permissions::from_mode(0o755)).unwrap();
        }
    }
    // `cmd` needs SystemRoot and `where` lives in System32, which is on both paths; the folders under test come first.
    let mut with_slang = environment(&[("PATH", &path)]);
    let mut clean = environment(&[("PATH", root.join("plain").as_os_str())]);
    if let Some(system_root) = std::env::var_os("SystemRoot") {
        let system32 = PathBuf::from(&system_root).join("System32");
        for (list, folders) in [
            (
                &mut with_slang,
                std::env::split_paths(&path).collect::<Vec<_>>(),
            ),
            (&mut clean, vec![root.join("plain")]),
        ] {
            let mut folders = folders;
            folders.push(system32.clone());
            list[0].1 = std::env::join_paths(folders).unwrap();
            list.push((OsString::from("SystemRoot"), system_root.clone()));
        }
    }
    // The Slang case must fail because `slangc` was found, not because the lookup itself could not run.
    let error = slangc_cannot_be_found(&with_slang).unwrap_err();
    assert!(
        error.contains("is found with the stripped environment"),
        "{error}"
    );
    assert!(!error.contains("could not run"), "{error}");
    assert_eq!(slangc_cannot_be_found(&clean), Ok(()));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_cache_that_holds_slang_or_is_missing_is_refused() {
    let (root, path) = fake_machine("cache");
    let parent = environment(&[("PATH", &path)]);
    assert_eq!(slang_home_problem(&root.join("home")), None);
    // A fake cache with an installed Slang in it (`<home>/<version>/bin/slangc`).
    fs::create_dir_all(root.join("cache/2026.22/bin")).unwrap();
    write(&root.join("cache/2026.22/bin"), "slangc", "");
    let problem = slang_home_problem(&root.join("cache")).unwrap();
    assert!(problem.contains("is not empty"), "{problem}");
    let error = no_slang_environment(
        &parent,
        &root.join("cache"),
        &directory_has_slang_tool,
        &folder_holds,
        &never_found,
    )
    .unwrap_err();
    assert!(error.contains("is not empty"), "{error}");
    let missing = slang_home_problem(&root.join("absent")).unwrap();
    assert!(missing.contains("cannot be read"), "{missing}");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_slang_folder_that_also_holds_the_cargo_is_refused_not_dropped() {
    let (root, _) = fake_machine("shared");
    let shared = std::env::join_paths([root.join("shared"), root.join("plain")]).unwrap();
    let parent = environment(&[("PATH", &shared)]);
    let error = no_slang_environment(
        &parent,
        &root.join("home"),
        &directory_has_slang_tool,
        &folder_holds,
        &never_found,
    )
    .unwrap_err();
    assert!(error.contains("`cargo`"), "{error}");
    assert!(error.contains("shared"), "{error}");
    assert!(shared_folder_problems(&[root.join("slang")], &folder_holds).is_empty());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_tree_with_the_run_time_bridge_is_recognised() {
    assert!(tree_has_slang_runtime(
        "hello-gust\n└── gust-compute-wgpu feature \"slang-runtime\"\n"
    ));
    assert!(!tree_has_slang_runtime(
        "hello-gust\n└── gust-compute-wgpu feature \"default\"\n"
    ));
}

#[test]
fn the_adapter_is_the_name_after_the_prefix() {
    assert_eq!(
        adapter_name("hello-gust: [1.0]\nadapter: NVIDIA GeForce RTX 5090 \n"),
        Some("NVIDIA GeForce RTX 5090")
    );
    assert_eq!(adapter_name("hello-gust: [1.0]\n"), None);
    assert_eq!(adapter_name("adapter: \n"), None);
    assert_eq!(adapter_name("an adapter: x\n"), None);
}

#[test]
fn a_crlf_packaged_listing_matches_and_any_other_difference_does_not() {
    let region = "fn one() {}\n\nfn two() {}\n";
    assert!(listing_matches(
        b"fn one() {}\r\n\r\nfn two() {}\r\n",
        region
    ));
    assert!(listing_matches(b"fn one() {}\n\nfn two() {}\n", region));
    assert!(!listing_matches(b"fn one() {}\r\n\r\nfn two() {}", region));
    assert!(
        !listing_matches(b"fn one() {}\r\r\nfn two() {}\n", region),
        "a lone CR is text"
    );
    assert_eq!(lf_line_ends(b"a\r\nb\rc\r\n"), b"a\nb\rc\n");
    // The file check: a CRLF copy of the region passes, and the same file with a changed byte fails.
    let root = scratch("crlf");
    write(&root, "source.rs", &SOURCE.replace('\n', "\r\n"));
    write(&root, "good.rs", "fn three() {}\r\n");
    write(&root, "bad.rs", "fn three() {x}\r\n");
    let listing = |packaged| Listing {
        packaged,
        source: "source.rs",
        anchor: "second",
    };
    assert_eq!(check_listings(&root, &[listing("good.rs")]), Ok(1));
    assert!(check_listings(&root, &[listing("bad.rs")]).is_err());
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
fn a_driver_output_is_not_an_artifact() {
    // `routes.json` is written by `cargo gust build` and `run` for the profile of the run: present on one side only, or
    // different, it is no difference; any other file still is.
    let committed = files(&[("a.wgsl", "x"), ("manifest.json", "{}")]);
    let fresh = files(&[
        ("a.wgsl", "x"),
        ("manifest.json", "{}"),
        ("routes.json", "{\"profile\": \"dev\"}"),
    ]);
    assert_eq!(tree_difference(&committed, &fresh), None);
    assert_eq!(tree_difference(&fresh, &committed), None);
    let other = files(&[
        ("a.wgsl", "x"),
        ("manifest.json", "{}"),
        ("routes.json", "{\"profile\": \"native\"}"),
    ]);
    assert_eq!(tree_difference(&fresh, &other), None);
    assert!(
        tree_difference(
            &committed,
            &files(&[("a.wgsl", "x"), ("routes.json.bak", "")])
        )
        .unwrap()
        .contains("file lists differ")
    );
}

#[test]
fn only_routed_runs_through_the_driver() {
    for entry in &ENTRIES {
        assert_eq!(entry.via_driver, entry.dir == "routed", "{}", entry.dir);
    }
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
    assert!(source.contains("check_gallery_gpu_report(!skip_web)"));
    // The call sits directly inside the Full-mode branch, so no other mode runs it.
    let call = source
        .find("check_gallery_gpu_report(!skip_web).map")
        .expect("the call");
    assert!(
        source[..call]
            .trim_end()
            .trim_end_matches("let outcome =")
            .trim_end()
            .ends_with("if mode == VerifyMode::Full {"),
        "the gallery step is not the first statement of `if mode == VerifyMode::Full`"
    );
    let usage = repository_file("xtask/src/main.rs");
    assert!(
        usage.contains("`check-full` runs the GPU part only"),
        "usage"
    );
    assert!(!source.contains("check_gallery_report"));
    let gallery = repository_file("xtask/src/gallery.rs");
    let gpu = gallery
        .split("pub(crate) fn check_gallery_gpu_report(browser: bool)")
        .nth(1)
        .expect("check_gallery_gpu_report");
    let body = &gpu[..gpu.find("\n}\n").expect("end")];
    assert!(
        body.contains("slang: false")
            && body.contains("no_slang: false")
            && body.contains("gpu: true")
    );
}

#[test]
fn the_registered_guide_listings_match_the_real_gallery_and_a_changed_gallery_line_fails_them() {
    let root = workspace_root().expect("the repository root");
    assert_eq!(
        LISTINGS.len(),
        43,
        "the guide's listings: chapter 2 (6), chapters 3, 5 and 8 (6), chapters 4 to 6 (27), chapter 9 (4)"
    );
    assert_eq!(check_listings(&root, &LISTINGS), Ok(LISTINGS.len()));
    // The same listings against copies of the gallery sources, G1's with one changed line of each region: each change must fail the check.
    let source = repository_file("gallery/hello-gust/src/main.rs");
    let scratch_root = scratch("guide-listings");
    for listing in &LISTINGS {
        let target = scratch_root.join(listing.packaged);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::copy(root.join(listing.packaged), &target).unwrap();
        let source_copy = scratch_root.join(listing.source);
        fs::create_dir_all(source_copy.parent().unwrap()).unwrap();
        fs::copy(root.join(listing.source), &source_copy).unwrap();
    }
    let gallery_file = scratch_root.join("gallery/hello-gust/src/main.rs");
    fs::create_dir_all(gallery_file.parent().unwrap()).unwrap();
    fs::write(&gallery_file, &source).unwrap();
    assert_eq!(
        check_listings(&scratch_root, &LISTINGS),
        Ok(LISTINGS.len()),
        "an unchanged copy"
    );
    for (needle, replacement) in [
        ("use gust::gpu;", "use gust::gpu as g;"),
        ("let i = id.x;", "let i = id.y;"),
        (
            "println!(\"hello-gust: {result:?}\");",
            "println!(\"hello: {result:?}\");",
        ),
    ] {
        assert_eq!(source.matches(needle).count(), 1, "{needle}");
        fs::write(&gallery_file, source.replace(needle, replacement)).unwrap();
        let error = check_listings(&scratch_root, &LISTINGS).unwrap_err();
        assert!(
            error.contains("is not byte-identical to region"),
            "{needle}: {error}"
        );
    }
    // Every copy on its own: a changed copy, with the gallery and every other copy intact, fails and names that copy
    // (the check stops at the first mismatch, so a run over all copies at once names only the first one that differs).
    fs::write(&gallery_file, &source).unwrap();
    for listing in &LISTINGS {
        let copy = scratch_root.join(listing.packaged);
        let original = fs::read_to_string(&copy).unwrap();
        fs::write(&copy, format!("{original}// drift\n")).unwrap();
        let error = check_listings(&scratch_root, &LISTINGS).unwrap_err();
        assert!(
            error.contains(&format!(
                "packaged listing {} is not byte-identical",
                listing.packaged
            )),
            "{}: {error}",
            listing.packaged
        );
        fs::write(&copy, original).unwrap();
        assert_eq!(check_listings(&scratch_root, &LISTINGS), Ok(LISTINGS.len()));
    }
    let _ = fs::remove_dir_all(&scratch_root);
}

#[test]
fn every_registered_guide_listing_is_included_by_the_chapter_that_shows_it() {
    for (guide, crate_dir) in [
        ("crates/gust/src/guide/mod.rs", "crates/gust/src/guide/"),
        (
            "crates/gust-wgpu/src/guide/mod.rs",
            "crates/gust-wgpu/src/guide/",
        ),
    ] {
        let text = repository_file(guide);
        for listing in LISTINGS
            .iter()
            .filter(|l| l.packaged.starts_with(crate_dir))
        {
            let relative = listing.packaged.strip_prefix(crate_dir).unwrap();
            assert!(
                text.contains(&format!("include_str!(\"{relative}\")")),
                "{guide} does not include {relative}"
            );
        }
    }
}

#[test]
fn only_web_hello_is_a_browser_entry_and_it_needs_a_library_and_a_page_not_a_main() {
    let root = workspace_root().expect("the repository root");
    for entry in &ENTRIES {
        assert_eq!(entry.browser, entry.dir == "web-hello", "{}", entry.dir);
        assert_eq!(
            required_files(entry),
            if entry.browser {
                &["Cargo.toml", "README.md", "src/lib.rs", "index.html"][..]
            } else {
                &["Cargo.toml", "README.md", "src/main.rs"][..]
            },
            "{}",
            entry.dir
        );
        // Every entry's files are in the repository (the check itself fails on a missing one).
        for file in required_files(entry) {
            assert!(
                root.join("gallery").join(entry.dir).join(file).is_file(),
                "gallery/{}/{file}",
                entry.dir
            );
        }
    }
    // A browser entry is run by the page, never by cargo, and has no second run or driver.
    let hello = ENTRIES.iter().find(|entry| entry.browser).unwrap();
    assert!(hello.profile_run.is_none() && !hello.via_driver);
}

#[test]
fn the_page_of_web_hello_logs_the_closing_part_the_runner_requires() {
    // The runner's `expected_line` of the entry, the constant of its page and the text of the library are one string.
    let hello = ENTRIES.iter().find(|entry| entry.browser).unwrap();
    assert_eq!(hello.expected_line, WEB_HELLO.closing);
    let library = repository_file("gallery/web-hello/src/lib.rs");
    assert!(
        library.contains(&format!("const HELD: &str = \"{}\";", WEB_HELLO.closing)),
        "gallery/web-hello/src/lib.rs does not hold the text the runner requires"
    );
    // The page logs the two prefixes the runner reads, and the OK line is made of the three parts the runner splits.
    assert!(library.contains("format!(\"GUST_OK {label}\")"));
    assert!(library.contains("format!(\"GUST_FAIL {}\", one_line(&reason))"));
    assert!(library.contains("\"{} | selected {} | {HELD}\""));
}

#[test]
fn the_sources_of_web_hello_have_the_anchored_regions_a_guide_chapter_can_show() {
    let source = repository_file("gallery/web-hello/src/lib.rs");
    assert_eq!(
        anchor_names(&source).unwrap(),
        [
            "imports",
            "kernel",
            "reference",
            "browser",
            "map-async",
            "tests"
        ]
    );
    let kernel = anchored_region(&source, "kernel").unwrap();
    assert!(kernel.contains("#[gpu]\npub fn add(a: f32, b: f32) -> f32 {\n    a + b\n}\n"));
    let browser = anchored_region(&source, "browser").unwrap();
    assert!(browser.contains("#[cfg(target_arch = \"wasm32\")]\nmod browser {"));
    assert!(browser.contains("add::map_async(&a, &b)"));
    let tests = anchored_region(&source, "tests").unwrap();
    assert!(tests.contains("add::map_on(&CpuDevice::new()"));
    // The host reference is plain Rust: it does not name the kernel.
    let reference = anchored_region(&source, "reference").unwrap();
    assert!(!reference.contains("add::") && reference.contains("fn host_reference"));
}

#[test]
fn the_wasm_bindgen_pin_of_the_gallery_equals_the_one_of_the_repository() {
    // `check-web` and the gallery's browser run use one `wasm-bindgen` CLI, so the two locks must pin one version.
    let version = |lock: &str| {
        let after = lock
            .split("name = \"wasm-bindgen\"\nversion = \"")
            .nth(1)
            .expect("a wasm-bindgen package in the lock");
        after[..after.find('"').unwrap()].to_owned()
    };
    let root = version(&repository_file("Cargo.lock"));
    assert_eq!(version(&repository_file("gallery/Cargo.lock")), root);
    assert!(
        repository_file("gallery/web-hello/Cargo.toml")
            .contains(&format!("wasm-bindgen = \"={root}\"")),
        "gallery/web-hello/Cargo.toml must pin wasm-bindgen = \"={root}\""
    );
}

#[test]
fn skip_web_leaves_the_browser_entry_out_prints_it_and_records_it() {
    // The skipped line names the entry and the flag; it is not silent.
    let hello = ENTRIES.iter().find(|entry| entry.browser).unwrap();
    assert_eq!(
        browser_skipped_line(hello),
        "web-hello: browser run SKIPPED on request (check-full --skip-web); no evidence for the page in a browser"
    );
    // The gallery step passes the opposite of `skip_web` and records the skip beside `check-web`; without the flag the
    // preflight of the Full run covers the gallery page's tools too.
    let verify = repository_file("xtask/src/verify.rs");
    assert!(verify.contains("check_gallery_gpu_report(!skip_web)"));
    let step = verify
        .split("check_gallery_gpu_report(!skip_web)")
        .nth(1)
        .unwrap();
    assert!(
        step.contains("if skip_web {\n            skipped.push(GALLERY_BROWSER_SKIP.to_owned());")
    );
    assert_eq!(GALLERY_BROWSER_SKIP, "check-gallery web-hello browser run");
    let web = repository_file("xtask/src/web.rs");
    assert!(web.contains("for example in [&WEB_VECTOR_ADD, &crate::gallery::WEB_HELLO]"));
    // Only the gpu part takes the flag, and the flagless check-gallery runs the browser.
    let gallery = repository_file("xtask/src/gallery.rs");
    assert!(gallery.contains("run_parts(&workspace_root()?, parts, true)?;"));
    assert!(gallery.contains("browser_skipped_line(entry)"));
}

#[test]
fn the_browser_build_of_the_gallery_clears_the_flag_variables_and_check_web_clears_none() {
    assert_eq!(
        WEB_HELLO.cleared_env,
        ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"]
    );
    assert!(crate::web::WEB_VECTOR_ADD.cleared_env.is_empty());
}

/// A libtest output with the given `(name, status)` lines.
fn test_output(lines: &[(&str, &str)]) -> String {
    let mut text = format!("running {} tests\n", lines.len());
    for (name, status) in lines {
        text.push_str(&format!("test {name} ... {status}\n"));
    }
    text.push_str("\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n");
    text
}

#[test]
fn every_entry_has_one_twin_rule_in_the_same_order() {
    let entries = ENTRIES.iter().map(|entry| entry.dir).collect::<Vec<_>>();
    let rules = TWIN_RULES.iter().map(|rule| rule.dir).collect::<Vec<_>>();
    assert_eq!(entries, rules);
    assert!(TWIN_RULES.iter().all(|rule| rule.min_twin >= 1));
}

#[test]
fn an_entry_whose_twin_tests_all_pass_is_judged_ok_and_the_other_tests_do_not_count() {
    let output = test_output(&[
        ("tests::the_twin_matches_the_program_values", "ok"),
        ("tests::the_twin_matches_the_host_reference", "ok"),
        ("tests::some_other_test", "ok"),
    ]);
    assert_eq!(check_entry_tests("hello-gust", &output), Ok(2));
}

#[test]
fn an_entry_with_an_ignored_twin_test_is_refused_and_named() {
    let output = test_output(&[
        ("tests::the_twin_matches_the_program_values", "ok"),
        ("tests::the_twin_matches_the_host_reference", "ignored"),
        ("tests::some_other_test", "ok"),
    ]);
    let error = check_entry_tests("hello-gust", &output).unwrap_err();
    assert!(error.contains("`hello-gust`"), "{error}");
    assert!(error.contains("1 ignored"), "{error}");
    assert!(
        error.contains("the_twin_matches_the_host_reference"),
        "{error}"
    );
}

#[test]
fn an_entry_with_no_twin_test_at_all_is_refused_and_named() {
    let output = test_output(&[("tests::some_other_test", "ok")]);
    let error = check_entry_tests("histogram", &output).unwrap_err();
    assert!(error.contains("`histogram`"), "{error}");
    assert!(error.contains("0 CPU-twin"), "{error}");
    let empty = test_output(&[]);
    let error = check_entry_tests("histogram", &empty).unwrap_err();
    assert!(
        error.contains("`histogram`") && error.contains("0 CPU-twin"),
        "{error}"
    );
}

#[test]
fn an_entry_that_lost_one_twin_test_is_refused_even_when_others_pass() {
    // `image-threshold` has three twin tests (text, pixel and grid); two ran.
    let output = test_output(&[
        ("tests::the_twin_matches_hand_computed_values", "ok"),
        (
            "tests::the_pixel_twin_matches_the_host_reference_exactly",
            "ok",
        ),
    ]);
    let error = check_entry_tests("image-threshold", &output).unwrap_err();
    assert!(
        error.contains("2 CPU-twin") && error.contains("at least 3"),
        "{error}"
    );
}

#[test]
fn a_failed_test_and_an_unknown_entry_are_errors() {
    let output = test_output(&[
        ("tests::the_twin_matches_the_program_values", "FAILED"),
        ("tests::the_twin_matches_the_host_reference", "ok"),
    ]);
    let error = check_entry_tests("hello-gust", &output).unwrap_err();
    assert!(
        error.contains("failed") && error.contains("the_twin_matches_the_program_values"),
        "{error}"
    );
    assert!(
        check_entry_tests("no-such-entry", &output)
            .unwrap_err()
            .contains("no test rule")
    );
}

#[test]
fn reduce_scan_may_ignore_exactly_its_gpu_test() {
    let gpu = (
        "tests::the_gpu_matches_the_host_references_exactly",
        "ignored, needs a Vulkan compute adapter",
    );
    let twin = (
        "tests::the_add_offsets_twin_completes_the_host_scan_exactly",
        "ok",
    );
    assert_eq!(
        check_entry_tests("reduce-scan", &test_output(&[twin, gpu])),
        Ok(1)
    );
    // Without its ignored GPU test, or with a second ignored test, the rule is not met.
    assert!(check_entry_tests("reduce-scan", &test_output(&[twin])).is_err());
    let other = ("tests::another", "ignored");
    assert!(check_entry_tests("reduce-scan", &test_output(&[twin, gpu, other])).is_err());
}

#[test]
fn the_rules_match_the_twin_tests_the_gallery_sources_declare() {
    // The source of each entry has at least as many `fn ..twin..` test functions as its rule demands, and exactly the
    // ignored attributes the rule allows (a drift guard between the table and the programs).
    let root = workspace_root().expect("the repository root");
    for rule in &TWIN_RULES {
        let source = if rule.dir == "web-hello" {
            "src/lib.rs"
        } else {
            "src/main.rs"
        };
        let text = fs::read_to_string(root.join("gallery").join(rule.dir).join(source))
            .unwrap_or_else(|error| panic!("{}: {error}", rule.dir))
            .replace("\r\n", "\n");
        let twins = text
            .lines()
            .filter(|line| line.trim_start().starts_with("fn ") && line.contains("twin"))
            .count();
        assert!(
            twins >= rule.min_twin,
            "{}: {twins} twin test function(s), rule {}",
            rule.dir,
            rule.min_twin
        );
        let ignored = text.matches("#[ignore").count();
        assert_eq!(ignored, rule.ignored, "{}: ignored attributes", rule.dir);
    }
}
