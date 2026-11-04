use std::{fs, path::PathBuf};

use super::{
    DependencyKind, DocDebt, DocMessage, HELLO_DIR, HELLO_OUTPUT, Member, Options, PackageSpec,
    consumer_lock_problems, doc_messages, doc_problems, docsrs_request, dry_run_problems,
    existing_crate_lines, fixture_dependency_problems, has_nightly, metadata_problems,
    package_allows, package_files, package_problems, parse_args, parse_metadata,
    patch_config_arguments, patched_manifest, readme_blocks, readme_problems, recipe_ran,
    required_file_problems, section_value, size_problems, unused_library_patches,
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
        stable: true,
    },
    DocDebt {
        needle: "links to private item",
        stable: false,
    },
];

#[test]
fn known_doc_debt_is_admitted_and_everything_else_is_a_problem() {
    let known = [
        message("warning", "redundant explicit link target"),
        message(
            "warning",
            "public documentation for `x` links to private item `y`",
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
        problems[0].contains("redundant explicit link target") && problems[0].contains("delete")
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
fn the_fixture_depends_on_publishable_members_at_their_exact_version() {
    let set = members();
    let good = "[package]\nname = \"h\"\n\n[dependencies]\nalpha = \"=1.2.0\"\n";
    assert!(fixture_dependency_problems(good, &set).is_empty());
    let loose = "[dependencies]\nalpha = \"1.2\"\n";
    assert!(fixture_dependency_problems(loose, &set)[0].contains("must pin \"=1.2.0\""));
    let renamed = "[dependencies]\ngust-compute = \"=1.2.0\"\n";
    assert!(fixture_dependency_problems(renamed, &set)[0].contains("no publishable member"));
    let unpublished = "[dependencies]\nexample = \"=0.1.0\"\n";
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
fn the_consumer_manifest_patches_every_package_onto_its_tree() {
    let trees = vec![
        ("alpha".to_owned(), "C:/w/unpacked/alpha-1.2.0".to_owned()),
        ("tool".to_owned(), "C:/w/unpacked/tool-1.2.0".to_owned()),
    ];
    let manifest = patched_manifest("[package]\r\nname = \"h\"\r\n", &trees);
    assert!(manifest.starts_with("[package]\nname = \"h\"\n\n# Added by"));
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
