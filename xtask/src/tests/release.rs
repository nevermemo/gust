use std::{
    collections::BTreeSet,
    ffi::OsString,
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use super::{
    DependencyKind, DocDebt, DocMessage, HELLO_DIR, HELLO_OUTPUT, Member, Options, PackageSpec,
    RegistryCleanup, Run, WorkDir, caret_satisfied, clean_registry_src, consumer_lock_problems,
    dep_info_entries, doc_messages, doc_problems, docsrs_request, dry_run_problems,
    existing_crate_lines, finish_outcome, fixture_dependency_problems, has_nightly, helper_choice,
    metadata_problems, package_allows, package_files, package_problems, parse_args, parse_metadata,
    patch_config_arguments, patched_manifest, r1_command, r8_command, readme_blocks,
    readme_problems, recipe_ran, registry_entries, registry_target, required_file_problems,
    section_value, size_problems, target_dep_info_entries, unused_library_patches, verify_problems,
};
use crate::workspace::workspace_root;

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| (*item).to_owned()).collect()
}

fn args(items: &[&str]) -> Vec<String> {
    strings(items)
}

fn repository_file(path: &str) -> String {
    fs::read_to_string(workspace_root().unwrap().join(path)).unwrap()
}

// ---------------------------------------------------------------------------------------------------------------
// The allow-list (R2)

fn library(name: &str) -> PackageSpec<'_> {
    PackageSpec::library(name, false, false)
}

#[test]
fn a_test_module_or_test_data_in_src_is_off_the_allow_list() {
    let spec = library("a");
    assert!(package_allows(&spec, "src/lib.rs"));
    assert!(package_allows(&spec, "src/a/b.rs"));
    for file in [
        "src/check_tests.rs",
        "src/tests/grid.rs",
        "src/testdata.rs",
        "src/testdata/uniform/a.wgsl",
        "src/slang/deep/testdata/x.txt",
    ] {
        assert!(!package_allows(&spec, file), "{file}");
    }
    // `testdata` is a path component, not a prefix: a module whose name merely starts with it ships.
    for file in [
        "src/testdata_loader.rs",
        "src/testdatabase/mod.rs",
        "src/contests/a.rs",
    ] {
        assert!(package_allows(&spec, file), "{file}");
    }
    let problems = package_problems(
        &spec,
        &strings(&["Cargo.toml", "src/lib.rs", "src/x_tests.rs"]),
        &[],
    );
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("src/x_tests.rs") && problems[0].contains("test module"),
        "{problems:?}"
    );
}

#[test]
fn assets_and_the_entry_depend_on_the_package() {
    let tool = PackageSpec {
        name: "tool",
        build_script: false,
        artifacts: false,
        assets: true,
        entry: "src/main.rs",
    };
    assert!(package_allows(&tool, "assets/helper.cpp"));
    assert!(!package_allows(&library("a"), "assets/helper.cpp"));
    let problems = package_problems(&tool, &strings(&["Cargo.toml", "src/lib.rs"]), &[]);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("src/main.rs"), "{problems:?}");
    let problems = package_problems(
        &library("a"),
        &strings(&["Cargo.toml", "src/lib.rs", "assets/helper.cpp"]),
        &[],
    );
    assert!(problems[0].contains("assets/helper.cpp"), "{problems:?}");
    assert!(!problems[0].contains("assets/**)"), "{problems:?}");
}

#[test]
fn the_allow_list_text_of_e8_is_unchanged() {
    let problems = package_problems(
        &PackageSpec::library("gust", true, false),
        &strings(&["Cargo.toml", "src/lib.rs", "docs/x.md"]),
        &[],
    );
    assert_eq!(
        problems,
        strings(&[
            "`gust` lists `docs/x.md`, which is not on the allow-list (src/**, Cargo.toml, Cargo.toml.orig, \
             Cargo.lock, .cargo_vcs_info.json, build.rs, README*, LICENSE*)"
        ])
    );
    assert_eq!(
        package_files("Cargo.toml\r\nsrc\\lib.rs\r\n\r\n"),
        strings(&["Cargo.toml", "src/lib.rs"])
    );
}

#[test]
fn both_licences_and_a_readme_are_required() {
    let all = strings(&["LICENSE-MIT", "LICENSE-APACHE", "README.md", "Cargo.toml"]);
    assert!(required_file_problems("a", &all).is_empty());
    let problems = required_file_problems("a", &strings(&["LICENSE-MIT", "README.md"]));
    assert_eq!(problems, strings(&["`a` does not list `LICENSE-APACHE`"]));
}

// ---------------------------------------------------------------------------------------------------------------
// Options

#[test]
fn the_options_are_parsed_and_the_rest_refused() {
    assert_eq!(parse_args(&[]).unwrap(), Options::default());
    let options = parse_args(&args(&["--check-only", "--work-dir", "somewhere"])).unwrap();
    assert!(options.check_only);
    assert_eq!(options.work_dir, Some(PathBuf::from("somewhere")));
    assert_eq!(
        parse_args(&args(&["--work-dir=x"])).unwrap().work_dir,
        Some(PathBuf::from("x"))
    );
    for bad in [
        args(&["--check-only=1"]),
        args(&["--work-dir"]),
        args(&["--work-dir="]),
        args(&["--work-dir", "a", "--work-dir", "b"]),
        args(&["--skip-web"]),
    ] {
        assert!(parse_args(&bad).is_err(), "{bad:?}");
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Members, R3 and R4

const METADATA: &str = r#"{
  "workspace_root": "C:\\work\\repo",
  "target_directory": "C:\\work\\target",
  "packages": [
    {
      "name": "alpha-macros", "version": "1.2.0", "publish": null,
      "manifest_path": "C:\\work\\repo\\crates\\alpha-macros\\Cargo.toml",
      "description": "d", "license": "MIT", "repository": "r", "documentation": "x",
      "keywords": ["k"], "categories": ["c"], "rust_version": "1.90",
      "dependencies": [],
      "targets": [{"kind": ["proc-macro"], "src_path": "C:\\work\\repo\\crates\\alpha-macros\\src\\lib.rs"}],
      "metadata": null
    },
    {
      "name": "alpha", "version": "1.2.0", "publish": null,
      "manifest_path": "C:\\work\\repo\\crates\\alpha\\Cargo.toml",
      "description": "d", "license": "MIT", "repository": "r", "documentation": "x",
      "keywords": ["k"], "categories": ["c"], "rust_version": "1.90",
      "dependencies": [
        {"name": "alpha-macros", "req": "=1.2.0", "kind": null},
        {"name": "alpha", "req": "*", "kind": "dev"},
        {"name": "serde", "req": "^1", "kind": null}
      ],
      "targets": [
        {"kind": ["lib"], "src_path": "C:\\work\\repo\\crates\\alpha\\src\\lib.rs"},
        {"kind": ["custom-build"], "src_path": "C:\\work\\repo\\crates\\alpha\\build.rs"}
      ],
      "metadata": {"docs": {"rs": {"features": ["extra"], "rustdoc-args": ["--cfg", "docsrs"]}}}
    },
    {
      "name": "tool", "version": "1.2.0", "publish": null,
      "manifest_path": "C:\\work\\repo\\crates\\tool\\Cargo.toml",
      "description": "d", "license": "MIT", "repository": "r", "documentation": "x",
      "keywords": ["k"], "categories": ["c"], "rust_version": "1.90",
      "dependencies": [{"name": "alpha", "req": "=1.2.0", "kind": null}],
      "targets": [{"kind": ["bin"], "name": "tool-bin", "src_path": "C:\\work\\repo\\crates\\tool\\src\\main.rs"}],
      "metadata": null
    },
    {
      "name": "example", "version": "0.1.0", "publish": [],
      "manifest_path": "C:\\work\\repo\\examples\\example\\Cargo.toml",
      "dependencies": [{"name": "alpha", "req": "*", "kind": null}],
      "targets": [{"kind": ["bin"], "name": "example", "src_path": "C:\\work\\repo\\examples\\example\\src\\main.rs"}]
    }
  ]
}"#;

fn members() -> Vec<Member> {
    parse_metadata(METADATA).unwrap().members
}

fn member<'a>(members: &'a [Member], name: &str) -> &'a Member {
    members.iter().find(|member| member.name == name).unwrap()
}

#[test]
fn members_are_read_from_cargo_metadata() {
    let metadata = parse_metadata(METADATA).unwrap();
    assert_eq!(metadata.target_directory, PathBuf::from("C:\\work\\target"));
    let members = metadata.members;
    assert_eq!(members.len(), 4);
    let alpha = member(&members, "alpha");
    assert!(alpha.publishable && alpha.has_library && !alpha.has_binary && alpha.build_script);
    assert_eq!(alpha.directory, "crates/alpha");
    assert_eq!(alpha.entry, "src/lib.rs");
    assert_eq!(alpha.docs_features, strings(&["extra"]));
    assert_eq!(alpha.docs_rustdoc_args, strings(&["--cfg", "docsrs"]));
    assert!(member(&members, "alpha-macros").has_library);
    let tool = member(&members, "tool");
    assert!(tool.has_binary && !tool.has_library);
    assert_eq!(tool.binary.as_deref(), Some("tool-bin"));
    assert_eq!(tool.entry, "src/main.rs");
    assert!(tool.spec().assets);
    assert_eq!(tool.spec().entry, "src/main.rs");
    assert!(!alpha.spec().assets);
    assert!(!member(&members, "example").publishable);
    assert_eq!(
        alpha.dependencies[0].kind,
        DependencyKind::Normal,
        "a null kind is a normal dependency"
    );
    assert!(parse_metadata("{}").is_err());
}

#[test]
fn a_consistent_workspace_has_no_metadata_problem() {
    assert_eq!(metadata_problems(&members()), Vec::<String>::new());
}

#[test]
fn r3_names_a_missing_field_a_split_version_and_a_loose_pin() {
    let mut set = members();
    let index = set
        .iter()
        .position(|member| member.name == "alpha")
        .unwrap();
    set[index].description = false;
    set[index].keywords = 0;
    let problems = metadata_problems(&set);
    assert!(
        problems
            .iter()
            .any(|p| p.contains("`alpha` has no `description`"))
    );
    assert!(
        problems
            .iter()
            .any(|p| p.contains("`alpha` has no `keywords`"))
    );

    let mut set = members();
    let index = set.iter().position(|member| member.name == "tool").unwrap();
    set[index].version = "1.3.0".to_owned();
    let problems = metadata_problems(&set);
    assert!(
        problems.iter().any(|p| p.contains("share one version")),
        "{problems:?}"
    );
    // The tool's pin on alpha is `=1.2.0` and alpha is still 1.2.0, so only the version split is reported for it.
    assert!(
        !problems.iter().any(|p| p.contains("exact")),
        "{problems:?}"
    );

    let mut set = members();
    let index = set
        .iter()
        .position(|member| member.name == "alpha")
        .unwrap();
    set[index].dependencies[0].requirement = "^1.2.0".to_owned();
    let problems = metadata_problems(&set);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("not the exact `=1.2.0`"),
        "{problems:?}"
    );

    let mut set = members();
    let index = set.iter().position(|member| member.name == "tool").unwrap();
    set[index].dependencies[0].requirement = "=1.1.0".to_owned();
    assert!(
        metadata_problems(&set)[0].contains("not the exact `=1.2.0`"),
        "a pin on another version is no pin on this release"
    );
}

#[test]
fn r4_refuses_a_publishable_example_and_a_dependency_on_an_unpublished_member() {
    let mut set = members();
    let index = set
        .iter()
        .position(|member| member.name == "example")
        .unwrap();
    set[index].publishable = true;
    let problems = metadata_problems(&set);
    assert!(
        problems
            .iter()
            .any(|p| p.starts_with("R4") && p.contains("outside crates/")),
        "{problems:?}"
    );

    let mut set = members();
    let index = set
        .iter()
        .position(|member| member.name == "alpha-macros")
        .unwrap();
    set[index].publishable = false;
    let problems = metadata_problems(&set);
    assert!(
        problems
            .iter()
            .any(|p| p.starts_with("R4") && p.contains("publish = false")),
        "{problems:?}"
    );

    let mut set = members();
    for member in &mut set {
        member.publishable = false;
    }
    assert!(metadata_problems(&set)[0].contains("no workspace member is publishable"));
}

// ---------------------------------------------------------------------------------------------------------------
// R5

fn message(level: &str, text: &str) -> DocMessage {
    DocMessage {
        level: level.to_owned(),
        text: text.to_owned(),
        location: "src/lib.rs:1".to_owned(),
    }
}

#[test]
fn rustdoc_diagnostics_are_read_from_the_json_lines() {
    let output = concat!(
        "{\"reason\":\"compiler-artifact\",\"package_id\":\"x\"}\n",
        "{\"reason\":\"compiler-message\",\"message\":{\"level\":\"warning\",\"message\":\"redundant explicit link\
         target\",\"spans\":[{\"file_name\":\"crates\\\\a\\\\src\\\\lib.rs\",\"line_start\":108}]}}\n",
        "{\"reason\":\"compiler-message\",\"message\":{\"level\":\"note\",\"message\":\"a note\",\"spans\":[]}}\n",
        "{\"reason\":\"compiler-message\",\"message\":{\"level\":\"error\",\"message\":\"boom\",\"spans\":[]}}\n",
        "not json\n",
        "{\"reason\":\"build-finished\",\"success\":true}\n",
    );
    let found = doc_messages(output);
    assert_eq!(found.len(), 2, "{found:?}");
    assert_eq!(found[0].level, "warning");
    assert_eq!(found[0].location, "crates/a/src/lib.rs:108");
    assert_eq!(found[1].level, "error");
    assert_eq!(found[1].location, "");
}

const DEBT: [DocDebt; 2] = [
    DocDebt {
        needle: "redundant explicit link target",
        file: "src/lib.rs",
        stable: true,
    },
    DocDebt {
        needle: "links to private item",
        file: "src/other.rs",
        stable: false,
    },
];

fn message_at(level: &str, text: &str, location: &str) -> DocMessage {
    DocMessage {
        level: level.to_owned(),
        text: text.to_owned(),
        location: location.to_owned(),
    }
}

#[test]
fn known_doc_debt_is_admitted_and_everything_else_is_a_problem() {
    let known = [
        message("warning", "redundant explicit link target"),
        message_at(
            "warning",
            "public documentation for `x` links to private item `y`",
            "crates/b/src/other.rs:9",
        ),
        message("warning", "`gust` (lib doc) generated 2 warnings"),
    ];
    assert!(doc_problems("nightly", &known, true, &DEBT).is_empty());
    let with_new = [
        message("warning", "redundant explicit link target"),
        message("warning", "unresolved link to `foo`"),
        message("error", "redundant explicit link target"),
    ];
    let problems = doc_problems("stable", &with_new, false, &DEBT);
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(
        problems[0].contains("unresolved link to `foo`") && problems[0].contains("src/lib.rs:1")
    );
    assert!(
        problems[1].contains("error"),
        "an error is never debt: {problems:?}"
    );
}

#[test]
fn a_debt_entry_that_no_longer_occurs_is_a_problem_in_the_run_that_shows_it() {
    // Stable run: only the stable entry must occur.
    let problems = doc_problems("stable", &[], false, &DEBT);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("redundant explicit link target")
            && problems[0].contains("src/lib.rs")
            && problems[0].contains("delete")
    );
    // The docs.rs-shaped run owes both.
    assert_eq!(doc_problems("nightly", &[], true, &DEBT).len(), 2);
    // With no debt at all a clean run is clean.
    assert!(doc_problems("stable", &[], false, &[]).is_empty());
}

#[test]
fn the_docsrs_request_joins_features_and_dedupes_flags() {
    let set = members();
    let alpha = member(&set, "alpha");
    let macros = member(&set, "alpha-macros");
    let (features, flags) = docsrs_request(&[alpha, macros, alpha]);
    assert_eq!(features, strings(&["alpha/extra", "alpha/extra"]));
    assert_eq!(flags, "--cfg docsrs");
}

#[test]
fn a_nightly_toolchain_is_found_in_the_rustup_listing() {
    assert!(has_nightly(
        "stable-x86_64-pc-windows-msvc (default)\nnightly-x86_64-pc-windows-msvc\n"
    ));
    assert!(!has_nightly(
        "stable-x86_64-pc-windows-msvc (default)\n1.90.0-x86_64-pc-windows-msvc\n"
    ));
    assert!(!has_nightly(""));
}

// ---------------------------------------------------------------------------------------------------------------
// R6, R8

#[test]
fn a_crate_over_2_mib_is_named() {
    let ok = [("a-1.crate".to_owned(), 2 * 1024 * 1024)];
    assert!(size_problems(&ok).is_empty());
    let big = [("a-1.crate".to_owned(), 2 * 1024 * 1024 + 1)];
    assert!(size_problems(&big)[0].contains("a-1.crate"));
}

#[test]
fn the_dry_run_must_end_every_package() {
    let stderr = "warning: crate a@1 already exists on crates.io index\n\
                  warning: aborting upload due to dry run\nwarning: aborting upload due to dry run\n";
    assert!(dry_run_problems(stderr, 2).is_empty());
    assert_eq!(dry_run_problems(stderr, 3).len(), 1);
    assert_eq!(existing_crate_lines(stderr).len(), 1);
}

// ---------------------------------------------------------------------------------------------------------------
// R7

#[test]
fn the_readme_blocks_are_the_fixture_word_for_word() {
    let readme = repository_file("README.md");
    let manifest = repository_file(&format!("{HELLO_DIR}/Cargo.toml"));
    let source = repository_file(&format!("{HELLO_DIR}/src/main.rs"));
    let problems = readme_problems(&readme, &manifest, &source);
    assert!(problems.is_empty(), "{problems:?}");
    // The fixture is the recipe, so it names the release it is made for and prints the line that R7 requires.
    assert!(source.contains(&HELLO_OUTPUT.replace("[11.0, 22.0, 33.0, 44.0]", "{result:?}")));
    assert!(manifest.contains("gust-dev = ["));
    assert_eq!(
        section_value(&manifest, "package", "name").as_deref(),
        Some("hello-gust")
    );
}

#[test]
fn a_changed_readme_block_or_fixture_is_named() {
    let readme = "# T\n\n## Install from crates.io\n\n```toml\n[package]\nname = \"x\"\n```\n\n\
                  ```rust\nfn main() {}\n```\n\n## Next\n";
    assert!(readme_problems(readme, "[package]\nname = \"x\"\n", "fn main() {}\n").is_empty());
    let problems = readme_problems(readme, "[package]\nname = \"y\"\n", "fn main() {}\n");
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("Cargo.toml block differs"));
    let problems = readme_problems(readme, "[package]\nname = \"x\"\n", "fn main() { }\n");
    assert!(problems[0].contains("main.rs block differs"));
    // CRLF text (a Windows checkout) is the same text.
    let crlf = readme.replace('\n', "\r\n");
    assert!(readme_problems(&crlf, "[package]\r\nname = \"x\"\r\n", "fn main() {}\r\n").is_empty());
}

#[test]
fn the_readme_section_needs_exactly_one_block_of_each_kind() {
    assert!(
        readme_blocks("# T\n")
            .unwrap_err()
            .contains("no `## Install from crates.io`")
    );
    let two = "## Install from crates.io\n```toml\na\n```\n```toml\nb\n```\n```rust\nc\n```\n";
    assert!(
        readme_blocks(two)
            .unwrap_err()
            .contains("exactly one ```toml")
    );
    // A block after the next heading is not the recipe's.
    let late = "## Install from crates.io\n```toml\na\n```\n## Other\n```rust\nc\n```\n";
    assert!(
        readme_blocks(late)
            .unwrap_err()
            .contains("exactly one ```rust")
    );
    let ok = "## Install from crates.io\ntext\n```toml\na\n```\n```rust\nc\n```\n## Other\n";
    assert_eq!(
        readme_blocks(ok).unwrap(),
        ("a\n".to_owned(), "c\n".to_owned())
    );
}

#[test]
fn the_fixture_depends_on_publishable_members_with_a_caret_requirement_the_release_satisfies() {
    let set = members();
    for good in [
        "\"1.2\"",
        "\"^1.2\"",
        "\"1\"",
        "\"1.2.0\"",
        "\"^1.2.0\"",
        "{ version = \"1.2\", features = [\"x\"] }",
    ] {
        let manifest = format!("[package]\nname = \"h\"\n\n[dependencies]\nalpha = {good}\n");
        assert!(
            fixture_dependency_problems(&manifest, &set).is_empty(),
            "{good}"
        );
    }
    // The release is 1.2.0: a requirement it does not satisfy, or one that is no caret requirement, is refused.
    for bad in [
        "\"1.3\"",
        "\"2\"",
        "\"=1.2.0\"",
        "\"~1.2\"",
        "\">=1.2\"",
        "\"*\"",
        "\"0.1\"",
        "{ path = \"x\" }",
    ] {
        let manifest = format!("[dependencies]\nalpha = {bad}\n");
        let problems = fixture_dependency_problems(&manifest, &set);
        assert_eq!(problems.len(), 1, "{bad}: {problems:?}");
        assert!(
            problems[0].contains("caret requirement"),
            "{bad}: {problems:?}"
        );
    }
    let renamed = "[dependencies]\ngust-compute = \"1.2\"\n";
    assert!(fixture_dependency_problems(renamed, &set)[0].contains("no publishable member"));
    let unpublished = "[dependencies]\nexample = \"0.1\"\n";
    assert!(fixture_dependency_problems(unpublished, &set)[0].contains("no publishable member"));
    assert!(
        fixture_dependency_problems("[package]\nname = \"h\"\n", &set)[0]
            .contains("no [dependencies]")
    );
    // The dependencies of the real fixture are all publishable members of the real workspace in the metadata of
    // any naming: here, only the shape is checked.
    let fixture = repository_file(&format!("{HELLO_DIR}/Cargo.toml"));
    assert!(fixture.contains("[dependencies]"));
}

#[test]
fn a_pre_release_requirement_passes_on_a_pre_release_version_and_fails_once_released() {
    let manifest = "[dependencies]\nalpha = \"1.2.0-dev\"\n";
    // The development line: the version is a pre-release and the requirement names it.
    let mut development = members();
    for member in &mut development {
        member.version = "1.2.0-dev".to_owned();
    }
    assert!(fixture_dependency_problems(manifest, &development).is_empty());
    // The release (1.2.0): the same requirement is the edit that was missed.
    let released = members();
    let problems = fixture_dependency_problems(manifest, &released);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("pre-release requirement"),
        "{problems:?}"
    );
    // The released form passes.
    let plain = "[dependencies]\nalpha = \"1.2\"\n";
    assert!(fixture_dependency_problems(plain, &released).is_empty());
}

#[test]
fn the_consumer_manifest_patches_every_package_onto_its_tree() {
    let trees = vec![
        ("alpha".to_owned(), "C:/w/unpacked/alpha-1.2.0".to_owned()),
        ("tool".to_owned(), "C:/w/unpacked/tool-1.2.0".to_owned()),
    ];
    let manifest = patched_manifest("[package]\r\nname = \"h\"\r\n", &trees);
    assert!(manifest.starts_with("[package]\nname = \"h\"\n\n# Added by"));
    // The consumer is its own workspace, declared before the patch table.
    assert!(manifest.contains("\n[workspace]\n\n[patch.crates-io]\n"));
    assert!(
        manifest.contains("[patch.crates-io]\nalpha = { path = \"C:/w/unpacked/alpha-1.2.0\" }\n")
    );
    assert!(manifest.ends_with("tool = { path = \"C:/w/unpacked/tool-1.2.0\" }\n"));
    let config = patch_config_arguments(&trees, "tool");
    assert_eq!(
        config,
        strings(&[
            "--config",
            "patch.crates-io.alpha.path=\"C:/w/unpacked/alpha-1.2.0\""
        ])
    );
}

#[test]
fn an_unused_patch_of_a_library_or_a_registry_source_in_the_lock_is_a_problem() {
    let stderr = "warning: Patch `alpha v1.2.0 (C:/w/unpacked/alpha-1.2.0)` was not used in the crate graph.\n\
                  warning: Patch `tool v1.2.0 (C:/w/unpacked/tool-1.2.0)` was not used in the crate graph.\n";
    let unused = unused_library_patches(stderr, &["alpha"]);
    assert_eq!(unused.len(), 1);
    assert!(unused[0].contains("alpha"));

    let lock = "[[package]]\nname = \"alpha\"\nversion = \"1.2.0\"\n\n[[package]]\nname = \"wgpu\"\nversion = \"1\"\n\
                source = \"registry+https://github.com/rust-lang/crates.io-index\"\n";
    assert!(consumer_lock_problems(lock, &["alpha"]).is_empty());
    let sourced = lock.replace(
        "name = \"alpha\"\nversion = \"1.2.0\"\n",
        "name = \"alpha\"\nversion = \"1.2.0\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\n",
    );
    assert!(consumer_lock_problems(&sourced, &["alpha"])[0].contains("came from a registry"));
    assert!(consumer_lock_problems(lock, &["beta"])[0].contains("no package `beta`"));
    assert!(
        consumer_lock_problems("", &["alpha"]).len() >= 2,
        "an empty lock must not pass"
    );
}

#[test]
fn the_recipe_result_is_one_exact_line() {
    assert!(recipe_ran(&format!("noise\n{HELLO_OUTPUT}\n")));
    assert!(recipe_ran(&format!("{HELLO_OUTPUT}\r\n")));
    assert!(!recipe_ran("hello-gust: [0.0, 0.0, 0.0, 0.0]\n"));
    assert!(!recipe_ran(&format!("x {HELLO_OUTPUT}\n")));
}

#[test]
fn a_debt_entry_admits_one_warning_in_its_file_and_a_second_match_fails() {
    // The same message in another file is no debt, and the entry then also did not occur.
    let elsewhere = [message_at(
        "warning",
        "redundant explicit link target",
        "crates/c/src/lib2.rs:5",
    )];
    let problems = doc_problems("stable", &elsewhere, false, &DEBT);
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(
        problems[0].contains("crates/c/src/lib2.rs:5"),
        "{problems:?}"
    );
    assert!(problems[1].contains("no longer occurs"), "{problems:?}");
    // Two warnings of the admitted kind: the first is the debt, the second is new (also in another file).
    let twice = [
        message_at(
            "warning",
            "redundant explicit link target",
            "crates/a/src/lib.rs:108",
        ),
        message_at(
            "warning",
            "redundant explicit link target",
            "crates/a/src/lib.rs:300",
        ),
    ];
    let problems = doc_problems("stable", &twice, false, &DEBT);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("lib.rs:300") && problems[0].contains("another"),
        "{problems:?}"
    );
    // The line number plays no part, so an edit above the warning does not break the entry.
    let moved = [message_at(
        "warning",
        "redundant explicit link target",
        "crates/a/src/lib.rs:9999",
    )];
    assert!(doc_problems("stable", &moved, false, &DEBT).is_empty());
}

#[test]
fn caret_requirements_follow_cargos_rules() {
    assert!(caret_satisfied("0.1", "0.1.0") && caret_satisfied("0.1", "0.1.9"));
    assert!(caret_satisfied("^0.1.0", "0.1.4") && !caret_satisfied("0.1", "0.2.0"));
    assert!(!caret_satisfied("0.1.2", "0.1.1") && caret_satisfied("0.1.2", "0.1.2"));
    assert!(caret_satisfied("0", "0.9.9") && !caret_satisfied("0", "1.0.0"));
    assert!(caret_satisfied("1", "1.9.0") && !caret_satisfied("1.3", "1.2.9"));
    assert!(caret_satisfied("0.0.3", "0.0.3") && !caret_satisfied("0.0.3", "0.0.4"));
    assert!(
        caret_satisfied("0.1", "0.1.0-rc.1"),
        "a pre-release suffix is ignored"
    );
    for bad in ["=0.1.0", ">=0.1", "~0.1", "*", "0.1.x", "", "1.2.3.4"] {
        assert!(!caret_satisfied(bad, "0.1.0"), "{bad}");
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The work directory (S4)

/// A fresh name below the temporary directory, for a test to use as a repository root or a work directory.
fn scratch(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    std::env::temp_dir().join(format!(
        "gust-release-test-{label}-{}-{nanos}",
        std::process::id()
    ))
}

#[test]
fn a_work_directory_inside_the_repository_is_refused() {
    let root = scratch("repo");
    let inside = root.join("target").join("work");
    let error = WorkDir::create(
        "check-release",
        "p",
        &root,
        &scratch("parent"),
        Some(&inside),
    )
    .err()
    .expect("inside the repository");
    assert!(error.contains("outside the repository"), "{error}");
    assert!(error.starts_with("check-release:"), "{error}");
    assert!(!inside.exists(), "a refused directory is not created");
}

#[test]
fn a_work_directory_with_content_is_refused_and_never_deleted() {
    let root = scratch("repo");
    let work = scratch("full");
    fs::create_dir_all(&work).unwrap();
    fs::write(work.join("keep.txt"), "mine").unwrap();
    let error = WorkDir::create(
        "check-precompiled E8",
        "p",
        &root,
        &scratch("parent"),
        Some(&work),
    )
    .err()
    .expect("not empty");
    assert!(
        error.starts_with("check-precompiled E8:") && error.contains("not empty"),
        "{error}"
    );
    assert!(
        work.join("keep.txt").is_file(),
        "the content of a refused directory survives"
    );
    fs::remove_dir_all(&work).unwrap();
}

#[test]
fn an_empty_or_new_work_directory_is_used_and_deleted_by_finish() {
    let root = scratch("repo");
    // An existing empty directory.
    let empty = scratch("empty");
    fs::create_dir_all(&empty).unwrap();
    let work = WorkDir::create(
        "check-release",
        "p",
        &root,
        &scratch("parent"),
        Some(&empty),
    )
    .unwrap();
    fs::create_dir_all(work.path.join("a")).unwrap();
    fs::write(work.path.join("a").join("f.txt"), "x").unwrap();
    assert_eq!(work.finish(Ok("done".to_owned())), Ok("done".to_owned()));
    assert!(!empty.exists(), "deleted by finish, with its content");
    // A directory that does not exist yet.
    let new = scratch("new");
    let work =
        WorkDir::create("check-release", "p", &root, &scratch("parent"), Some(&new)).unwrap();
    assert!(new.is_dir());
    assert_eq!(work.finish(Err("boom".to_owned())), Err("boom".to_owned()));
    assert!(!new.exists(), "a failed run deletes it too");
}

#[test]
fn the_default_work_directory_is_below_the_target_directory_or_else_the_temporary_directory() {
    let root = scratch("repo");
    // A cargo target directory outside the repository holds the work directory.
    let outside = scratch("target");
    let work = WorkDir::create("check-release", "gust-x", &root, &outside, None).unwrap();
    assert!(work.path.starts_with(&outside), "{}", work.path.display());
    let name = work
        .path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert!(name.starts_with("gust-x-"), "{name}");
    let first = work.path.clone();
    assert!(work.finish(Ok(String::new())).is_ok());
    assert!(!first.exists());
    fs::remove_dir_all(&outside).unwrap();
    // A target directory inside the repository (the default `<repo>/target`, as on CI) falls back to the temporary
    // directory.
    let inside = root.join("target");
    let work = WorkDir::create("check-release", "gust-x", &root, &inside, None).unwrap();
    assert!(
        work.path.starts_with(std::env::temp_dir()),
        "{}",
        work.path.display()
    );
    assert!(!work.path.starts_with(&root));
    assert!(work.finish(Ok(String::new())).is_ok());
}

#[test]
fn a_failed_deletion_fails_the_run_and_both_failures_are_shown() {
    assert_eq!(
        finish_outcome(Ok("s".to_owned()), Ok(())),
        Ok("s".to_owned())
    );
    assert_eq!(
        finish_outcome(Err("run".to_owned()), Ok(())),
        Err("run".to_owned())
    );
    // A run that passed but left its directory behind did not pass.
    assert_eq!(
        finish_outcome(Ok("s".to_owned()), Err("could not delete".to_owned())),
        Err("could not delete".to_owned())
    );
    assert_eq!(
        finish_outcome(Err("run".to_owned()), Err("could not delete".to_owned())),
        Err("run\ncould not delete".to_owned())
    );
}

// ---------------------------------------------------------------------------------------------------------------
// The reflection helper of the full run (S2) and the verification judgement of R1 (S5)

#[test]
fn the_helper_is_inherited_defaulted_or_missing_by_name() {
    let helper = scratch("helper");
    fs::write(&helper, "x").unwrap();
    let default = scratch("default-helper");
    // A variable that names a file is the caller's: the children inherit it and nothing is added.
    assert_eq!(
        helper_choice(Some(OsString::from(&helper)), &default),
        Ok(None)
    );
    // A variable that names nothing is refused, with the default named as the way out.
    let error = helper_choice(Some(OsString::from("nowhere/helper")), &default).unwrap_err();
    assert!(
        error.contains("nowhere") && error.contains(&default.display().to_string()),
        "{error}"
    );
    // Unset: the default is handed to the children when it exists.
    assert_eq!(helper_choice(None, &helper), Ok(Some(helper.clone())));
    // Unset and no default: the failure names both.
    let error = helper_choice(None, &default).unwrap_err();
    assert!(
        error.contains("GUST_SLANG_REFLECT") && error.contains(&default.display().to_string()),
        "{error}"
    );
    fs::remove_file(&helper).unwrap();
}

#[test]
fn r1_requires_a_verifying_line_for_every_package() {
    let set = members();
    let both: Vec<&Member> = ["alpha", "tool"]
        .iter()
        .map(|name| member(&set, name))
        .collect();
    let stderr = "   Packaging alpha v1.2.0\n   Verifying alpha v1.2.0 (C:\\w)\n   Verifying tool v1.2.0 (C:\\w)\n";
    assert!(verify_problems(stderr, &both).is_empty());
    // A missing line and a line of another version do not count.
    let missing = "   Verifying alpha v1.2.0 (C:\\w)\n";
    let problems = verify_problems(missing, &both);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("`tool`") && problems[0].contains("Verifying"),
        "{problems:?}"
    );
    let wrong = "   Verifying alpha v1.2.1\n   Verifying tool v1.2.0\n";
    assert_eq!(verify_problems(wrong, &both).len(), 1);
    assert_eq!(verify_problems("", &both).len(), 2);
    assert!(verify_problems("", &[]).is_empty());
}

// ---------------------------------------------------------------------------------------------------------------
// The target of R1 and R8 (mr.r2) and the cleanup of cargo's registry/src

/// The value that `command` sets for the environment variable `name`.
fn env_of(command: &std::process::Command, name: &str) -> Option<OsString> {
    command
        .get_envs()
        .find(|(key, _)| *key == std::ffi::OsStr::new(name))
        .and_then(|(_, value)| value.map(OsString::from))
}

fn args_of(command: &std::process::Command) -> Vec<String> {
    command
        .get_args()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect()
}

#[test]
fn r1_and_r8_build_into_a_target_directory_inside_the_work_directory() {
    let set = members();
    let both: Vec<&Member> = ["alpha", "tool"]
        .iter()
        .map(|name| member(&set, name))
        .collect();
    let root = scratch("repo");
    let work = scratch("work");
    let run = Run {
        root: &root,
        cargo: OsString::from("cargo"),
        work: &work,
        check_only: true,
        helper: None,
    };
    let package = r1_command(&run, &both);
    let publish = r8_command(&run, &both);
    for (command, leading) in [(&package, "package"), (&publish, "publish")] {
        let target = PathBuf::from(
            env_of(command, "CARGO_TARGET_DIR").expect("the target is set on the command itself"),
        );
        assert!(
            target.starts_with(&work) && target != work,
            "{} is not inside {}",
            target.display(),
            work.display()
        );
        assert_eq!(target, registry_target(&work));
        let args = args_of(command);
        assert_eq!(args[0], leading, "{args:?}");
        assert!(
            args.iter().any(|argument| argument == "--locked"),
            "{args:?}"
        );
        for name in ["alpha", "tool"] {
            let at = args.iter().position(|argument| argument == name).unwrap();
            assert_eq!(args[at - 1], "-p", "{args:?}");
        }
        assert_eq!(command.get_current_dir(), Some(root.as_path()));
    }
    assert_eq!(args_of(&publish)[1], "--dry-run");
}

fn packages(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|item| (*item).to_owned()).collect()
}

/// Makes `src/<name>/<child>` for each child: a name ending in `/` is a directory with a source file, else a file.
fn fill_registry(src: &std::path::Path, entries: &[(&str, &[&str])]) {
    fs::create_dir_all(src).unwrap();
    for (name, children) in entries {
        let entry = src.join(name);
        fs::create_dir_all(&entry).unwrap();
        for child in *children {
            match child.strip_suffix('/') {
                Some(directory) => {
                    let path = entry.join(directory);
                    fs::create_dir_all(path.join("src")).unwrap();
                    fs::write(path.join("src").join("lib.rs"), "// source").unwrap();
                }
                None => fs::write(entry.join(child), "x").unwrap(),
            }
        }
    }
}

fn names(items: &[&str]) -> BTreeSet<String> {
    packages(items)
}

#[test]
fn a_named_new_entry_of_only_publishable_name_version_directories_is_deleted() {
    let wanted = packages(&["alpha-1.2.0", "tool-1.2.0"]);
    let src = scratch("reg-new");
    fill_registry(&src, &[("index.crates.io-aaaa", &["serde-1.0.0/"])]);
    let before = registry_entries(&src);
    // Cargo makes this one during the run: the hash of the registry's path, with the packages of the run.
    fill_registry(
        &src,
        &[("-dff94c00431122b2", &["alpha-1.2.0/", "tool-1.2.0/"])],
    );
    let named = names(&["-dff94c00431122b2", "index.crates.io-aaaa"]);
    let result = clean_registry_src(&src, &before, &named, &wanted);
    assert_eq!(result.removed.len(), 1, "{result:?}");
    assert_eq!(result.removed[0].0, "-dff94c00431122b2");
    assert!(result.removed[0].1 > 0, "the size is reported");
    assert!(
        result.kept.is_empty() && result.failed.is_empty(),
        "{result:?}"
    );
    assert!(!result.is_failure());
    assert!(!src.join("-dff94c00431122b2").exists());
    assert!(
        src.join("index.crates.io-aaaa")
            .join("serde-1.0.0")
            .is_dir(),
        "the registry's own entry is untouched, though the dep-info names it"
    );
    assert!(result.describe().contains("-dff94c00431122b2"));
    fs::remove_dir_all(&src).unwrap();
}

#[test]
fn a_new_entry_that_no_dep_info_of_this_run_names_is_never_deleted() {
    // Another run's entry: new since our listing and shaped like ours, but our builds never read it.
    let wanted = packages(&["alpha-1.2.0", "tool-1.2.0"]);
    let src = scratch("reg-other-run");
    fill_registry(&src, &[]);
    let before = registry_entries(&src);
    fill_registry(&src, &[("-8888", &["alpha-1.2.0/", "tool-1.2.0/"])]);
    let result = clean_registry_src(&src, &before, &names(&[]), &wanted);
    assert_eq!(result, RegistryCleanup::default());
    assert!(src.join("-8888").join("alpha-1.2.0").is_dir());
    assert!(
        result.describe().contains("nothing deleted"),
        "{}",
        result.describe()
    );
    // Naming an entry that does not exist (another cargo home's, say) is not an error either.
    let result = clean_registry_src(&src, &before, &names(&["-9999"]), &wanted);
    assert_eq!(result, RegistryCleanup::default());
    fs::remove_dir_all(&src).unwrap();
}

#[test]
fn a_named_entry_that_existed_before_the_run_fails_closed_and_is_not_deleted() {
    let wanted = packages(&["alpha-1.2.0", "tool-1.2.0"]);
    // What a killed run left under the same work directory name: shaped like ours, named by our dep-info, not new.
    let src = scratch("reg-old");
    fill_registry(&src, &[("-7777", &["alpha-1.2.0/", "tool-1.2.0/"])]);
    let before = registry_entries(&src);
    let result = clean_registry_src(&src, &before, &names(&["-7777"]), &wanted);
    assert_eq!(result.reused, strings(&["-7777"]), "{result:?}");
    assert!(result.removed.is_empty());
    assert!(result.is_failure());
    assert!(src.join("-7777").join("alpha-1.2.0").is_dir());
    let said = result.describe();
    assert!(
        said.contains("-7777") && said.contains("stale") && said.contains("delete"),
        "{said}"
    );
    // An old entry of another shape that the builds read (an older crates.io layout) is nothing to report.
    fill_registry(&src, &[("github.com-1ecc", &["serde-1.0.0/"])]);
    let before = registry_entries(&src);
    let result = clean_registry_src(&src, &before, &names(&["github.com-1ecc"]), &wanted);
    assert_eq!(result, RegistryCleanup::default());
    fs::remove_dir_all(&src).unwrap();
}

#[test]
fn a_named_new_entry_with_any_other_content_is_kept_and_named() {
    let wanted = packages(&["alpha-1.2.0", "tool-1.2.0"]);
    let src = scratch("reg-other");
    fill_registry(&src, &[]);
    let before = registry_entries(&src);
    fill_registry(
        &src,
        &[
            // another crate next to ours
            ("-1111", &["alpha-1.2.0/", "serde-1.0.0/"]),
            // another version of ours
            ("-2222", &["alpha-1.2.1/"]),
            // a file beside our directory
            ("-3333", &["alpha-1.2.0/", "notes.txt"]),
            // our name as a file
            ("-4444", &["tool-1.2.0"]),
            // nothing at all
            ("-5555", &[]),
            // an index entry, even one that looks like ours
            ("index.crates.io-bbbb", &["alpha-1.2.0/"]),
        ],
    );
    // A new entry that is itself a file.
    fs::write(src.join("-6666"), "x").unwrap();
    let named = names(&[
        "-1111",
        "-2222",
        "-3333",
        "-4444",
        "-5555",
        "-6666",
        "index.crates.io-bbbb",
    ]);
    let result = clean_registry_src(&src, &before, &named, &wanted);
    assert!(
        result.removed.is_empty() && result.failed.is_empty(),
        "{result:?}"
    );
    assert!(!result.is_failure());
    assert_eq!(result.kept.len(), 6, "{result:?}");
    for name in ["-1111", "-2222", "-3333", "-4444", "-5555", "-6666"] {
        assert!(src.join(name).exists(), "{name} survives");
        assert!(
            result.kept.iter().any(|line| line.starts_with(name)),
            "{name} is named: {result:?}"
        );
    }
    assert!(
        src.join("index.crates.io-bbbb")
            .join("alpha-1.2.0")
            .is_dir()
    );
    let said = result.describe();
    assert!(
        said.contains("-1111") && said.contains("serde-1.0.0"),
        "{said}"
    );
    assert!(said.contains("not deleted"), "{said}");
    fs::remove_dir_all(&src).unwrap();
}

#[test]
fn dep_info_names_the_registry_src_entries_whatever_the_spelling() {
    let windows = concat!(
        "C:\\w\\package-target\\debug\\deps\\gust-1a2b.d: ",
        "C:\\h\\.cargo\\registry\\src\\-dff94c\\gust-compute-0.1.0\\src\\lib.rs ",
        "C:\\h\\.cargo\\registry\\src\\-dff94c\\gust-compute-macros-0.1.0\\src\\lib.rs\n",
        "C:\\h\\.cargo\\registry\\src\\index.crates.io-1949\\serde-1.0.0\\src\\lib.rs:\n"
    );
    assert_eq!(
        dep_info_entries(windows),
        names(&["-dff94c", "index.crates.io-1949"])
    );
    let unix = "deps/x.d: /h/.cargo/registry/src/-0abc/gust-0.1.0/src/lib.rs\n";
    assert_eq!(dep_info_entries(unix), names(&["-0abc"]));
    // A path outside registry/src names nothing, and neither does an empty text.
    let other = "deps/x.d: /h/work/crates/gust/src/lib.rs /h/.cargo/registry/cache/x.crate\n";
    assert_eq!(dep_info_entries(other), names(&[]));
    assert_eq!(dep_info_entries(""), names(&[]));
}

#[test]
fn the_dep_info_files_below_a_target_name_the_entries_and_other_files_do_not() {
    let target = scratch("target");
    let deps = target
        .join("package")
        .join("tool-1.2.0")
        .join("debug")
        .join("deps");
    fs::create_dir_all(&deps).unwrap();
    fs::write(
        deps.join("alpha-1a.d"),
        "alpha.d: /h/.cargo/registry/src/-aaaa/alpha-1.2.0/src/lib.rs\n",
    )
    .unwrap();
    fs::write(
        deps.join("tool.d"),
        "tool.d: /h/.cargo/registry/src/-bbbb/alpha-1.2.0/src/lib.rs\n",
    )
    .unwrap();
    // Only `.d` files count: a log or a binary that happens to contain such a path does not.
    fs::write(
        deps.join("note.txt"),
        "/h/.cargo/registry/src/-cccc/alpha-1.2.0/src/lib.rs",
    )
    .unwrap();
    fs::write(
        deps.join("libalpha.rlib"),
        b"\xff\xfe /h/.cargo/registry/src/-dddd/x/",
    )
    .unwrap();
    assert_eq!(target_dep_info_entries(&target), names(&["-aaaa", "-bbbb"]));
    fs::remove_dir_all(&target).unwrap();
    // A target that does not exist (the build failed before it was made) names nothing.
    assert_eq!(target_dep_info_entries(&scratch("no-target")), names(&[]));
}
