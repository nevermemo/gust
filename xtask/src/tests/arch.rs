use super::{
    CRATE_RULES, CrateRule, Dependency, LockedPackage, dependency_problems, ir_module_problems,
    ir_path_problems, lock_problems, package_problems, parse_lock, parse_metadata, relative_to,
    unlisted_crate_problems,
};

/// The workspace root of the sample metadata below.
const ROOT: &str = "/ws";
const CRATES_IO: &str = "registry+https://github.com/rust-lang/crates.io-index";

fn rule(name: &str) -> &'static CrateRule {
    CRATE_RULES.iter().find(|rule| rule.name == name).unwrap()
}

fn quoted(value: Option<&str>) -> String {
    value.map_or("null".to_owned(), |text| format!("\"{text}\""))
}

/// One dependency object in the shape `cargo metadata --no-deps` prints.
fn dep(
    name: &str,
    rename: Option<&str>,
    source: Option<&str>,
    registry: Option<&str>,
    path: Option<&str>,
    kind: Option<&str>,
) -> String {
    let path = path.map_or(String::new(), |path| format!(r#","path":"{path}""#));
    format!(
        r#"{{"name":"{name}","source":{},"req":"^1","kind":{},"rename":{},"optional":false,"uses_default_features":true,"features":["f"],"target":null,"registry":{}{path}}}"#,
        quoted(source),
        quoted(kind),
        quoted(rename),
        quoted(registry),
    )
}

/// A crates.io dependency.
fn from_registry(name: &str) -> String {
    dep(name, None, Some(CRATES_IO), None, None, None)
}

/// A path dependency on a workspace crate in its own directory.
fn workspace_crate(name: &str) -> String {
    dep(
        name,
        None,
        None,
        None,
        Some(&format!("{ROOT}/crates/{name}")),
        None,
    )
}

/// One package object, located at the manifest of the rule of the same name.
fn package_json(name: &str, dependencies: &[String]) -> String {
    let manifest = rule(name).manifest;
    format!(
        r#"{{"name":"{name}","version":"0.1.0","id":"path+file://{ROOT}/{manifest}#0.1.0","source":null,"dependencies":[{}],"targets":[{{"kind":["lib"],"name":"{name}"}}],"features":{{}},"manifest_path":"{ROOT}/{manifest}","metadata":null}}"#,
        dependencies.join(",")
    )
}

fn metadata_json(packages: &[String]) -> String {
    format!(
        r#"{{"packages":[{}],"workspace_members":[],"resolve":null,"target_directory":"{ROOT}/target","version":1,"workspace_root":"{ROOT}","metadata":null}}"#,
        packages.join(",")
    )
}

/// The dependency problems of the crate `name` with the given dependencies, read through the
/// JSON reader like the real input.
fn found_for(name: &str, dependencies: &[String]) -> Vec<String> {
    let json = metadata_json(&[package_json(name, dependencies)]);
    let packages = parse_metadata(&json).unwrap();
    assert_eq!(packages.len(), 1);
    dependency_problems(rule(name), &packages[0], ROOT)
}

/// The dependency sets of the real workspace.
fn current_packages() -> Vec<String> {
    vec![
        package_json(
            "gust-macros",
            &[
                from_registry("proc-macro2"),
                from_registry("quote"),
                from_registry("syn"),
            ],
        ),
        package_json(
            "gust",
            &[workspace_crate("gust-macros"), workspace_crate("gust")],
        ),
        package_json(
            "gust-wgpu",
            &[
                workspace_crate("gust"),
                workspace_crate("gust-wgpu"),
                from_registry("pollster"),
                from_registry("wgpu"),
            ],
        ),
        package_json("xtask", &[]),
    ]
}

#[test]
fn gust_and_gust_wgpu_may_depend_on_themselves_but_nothing_else_may() {
    // The dev-dependency that enables the package's own `slang-runtime` feature for its tests (D44).
    assert!(found_for("gust", &[workspace_crate("gust")]).is_empty());
    assert!(found_for("gust-wgpu", &[workspace_crate("gust-wgpu")]).is_empty());
    // It is not a licence for a package to depend on its siblings, or for another crate to depend on gust.
    let sibling = found_for("gust", &[workspace_crate("gust-wgpu")]);
    assert_eq!(sibling.len(), 1, "{sibling:?}");
    assert!(
        sibling[0].contains("`gust-wgpu`, which is not in its allow-list"),
        "{sibling:?}"
    );
    let other = found_for("gust-macros", &[workspace_crate("gust-macros")]);
    assert_eq!(other.len(), 1, "{other:?}");
    assert!(
        other[0].contains("`gust-macros` has dependency `gust-macros`"),
        "{other:?}"
    );
    let xtask = found_for("xtask", &[workspace_crate("xtask")]);
    assert_eq!(xtask.len(), 1, "{xtask:?}");
}

#[test]
fn metadata_is_read_into_packages_and_dependencies() {
    let json = metadata_json(&[
        package_json(
            "gust-macros",
            &[
                from_registry("syn"),
                dep("quote", None, Some(CRATES_IO), None, None, Some("dev")),
                dep(
                    "friendly",
                    Some("shim"),
                    Some(CRATES_IO),
                    None,
                    None,
                    Some("build"),
                ),
                dep(
                    "gust",
                    None,
                    None,
                    Some("mine"),
                    Some("/ws/crates/gust"),
                    None,
                ),
            ],
        ),
        package_json("xtask", &[]),
    ]);
    let packages = parse_metadata(&json).unwrap();
    assert_eq!(packages.len(), 2);
    assert_eq!(packages[0].name, "gust-macros");
    assert_eq!(
        packages[0].manifest_path,
        "/ws/crates/gust-macros/Cargo.toml"
    );
    assert_eq!(
        packages[0].dependencies,
        [
            Dependency {
                name: "syn".to_owned(),
                rename: None,
                source: Some(CRATES_IO.to_owned()),
                registry: None,
                path: None,
            },
            Dependency {
                name: "quote".to_owned(),
                rename: None,
                source: Some(CRATES_IO.to_owned()),
                registry: None,
                path: None,
            },
            Dependency {
                name: "friendly".to_owned(),
                rename: Some("shim".to_owned()),
                source: Some(CRATES_IO.to_owned()),
                registry: None,
                path: None,
            },
            Dependency {
                name: "gust".to_owned(),
                rename: None,
                source: None,
                registry: Some("mine".to_owned()),
                path: Some("/ws/crates/gust".to_owned()),
            },
        ]
    );
    assert!(packages[1].dependencies.is_empty());
}

#[test]
fn unreadable_metadata_is_an_error_not_a_pass() {
    for (text, expected) in [
        ("", "invalid JSON"),
        ("not json", "invalid JSON"),
        ("{}", "no `packages` array"),
        (r#"{"packages": {}}"#, "no `packages` array"),
        (r#"{"packages": [{"name": "x"}]}"#, "lacks `manifest_path`"),
        (
            r#"{"packages": [{"name": "x", "manifest_path": "/x/Cargo.toml"}]}"#,
            "package `x` has no `dependencies`",
        ),
        (
            r#"{"packages": [{"name": "x", "manifest_path": "/m", "dependencies": [{}]}]}"#,
            "lacks `name`",
        ),
        (
            r#"{"packages": [{"name": null, "manifest_path": "/m", "dependencies": []}]}"#,
            "lacks `name`",
        ),
    ] {
        let error = parse_metadata(text).unwrap_err();
        assert!(error.contains(expected), "{text}: {error}");
        assert!(error.starts_with("cargo metadata"), "{error}");
    }
}

#[test]
fn the_current_dependency_sets_pass() {
    let mut packages = current_packages();
    // Example packages are not checked, whatever they depend on.
    packages.push(
        r#"{"name":"vector-add","manifest_path":"/ws/examples/vector-add/Cargo.toml","dependencies":[{"name":"serde","source":null,"rename":null,"registry":null}]}"#
            .to_owned(),
    );
    let parsed = parse_metadata(&metadata_json(&packages)).unwrap();
    assert_eq!(package_problems(ROOT, &parsed), Vec::<String>::new());
}

#[test]
fn a_new_dependency_names_d02_and_decisions() {
    let mut deps = vec![
        from_registry("proc-macro2"),
        from_registry("quote"),
        from_registry("syn"),
    ];
    deps.push(from_registry("serde"));
    let problems = found_for("gust-macros", &deps);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("`serde`"), "{}", problems[0]);
    assert!(problems[0].contains("D02"), "{}", problems[0]);
    assert!(
        problems[0].contains("explain the change in DECISIONS"),
        "{}",
        problems[0]
    );
    assert!(
        problems[0].starts_with("crates/gust-macros/Cargo.toml: "),
        "{}",
        problems[0]
    );
    assert!(
        problems[0].contains("allow-list (proc-macro2, quote, syn)"),
        "{}",
        problems[0]
    );
}

#[test]
fn xtask_may_not_gain_any_dependency() {
    let problems = found_for("xtask", &[from_registry("sha2")]);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("allow-list (none)"), "{}", problems[0]);
}

#[test]
fn a_dev_or_build_dependency_counts_like_any_other() {
    for kind in ["dev", "build"] {
        let dependency = dep("tempfile", None, Some(CRATES_IO), None, None, Some(kind));
        let problems = found_for("gust", &[dependency]);
        assert_eq!(problems.len(), 1, "{kind}: {problems:?}");
    }
}

#[test]
fn a_dependency_listed_under_several_kinds_is_reported_once() {
    let dependencies = [
        dep("serde", None, Some(CRATES_IO), None, None, None),
        dep("serde", None, Some(CRATES_IO), None, None, Some("dev")),
        dep("serde", None, Some(CRATES_IO), None, None, Some("build")),
    ];
    assert_eq!(found_for("gust", &dependencies).len(), 1);
}

#[test]
fn spirv_libraries_are_rejected_in_the_compiler_crates_by_name_and_by_rename() {
    for dependency in [
        from_registry("rspirv"),
        from_registry("spirv"),
        from_registry("spirv-std"),
        from_registry("rspirv-reflect"),
        // `friendly = { package = "rspirv" }`: cargo reports the real name and the key.
        dep(
            "rspirv",
            Some("friendly"),
            Some(CRATES_IO),
            None,
            None,
            None,
        ),
        dep("spirv", Some("friendly"), Some(CRATES_IO), None, None, None),
    ] {
        for name in ["gust-macros", "gust"] {
            let problems = found_for(name, std::slice::from_ref(&dependency));
            assert_eq!(problems.len(), 1, "{name}: {dependency}");
            assert!(problems[0].contains("no SPIR-V emitter"), "{}", problems[0]);
        }
    }
    // The runtime crate gets the generic message; it is not a compiler crate.
    let problems = found_for("gust-wgpu", &[from_registry("rspirv")]);
    assert_eq!(problems.len(), 1);
    assert!(problems[0].contains("allow-list"), "{}", problems[0]);
}

#[test]
fn a_rename_is_judged_by_the_real_package_and_is_itself_a_problem() {
    // `syn = { package = "evil-syn", .. }`: the key looks allowed, the package is not.
    let renamed_away = dep("evil-syn", Some("syn"), Some(CRATES_IO), None, None, None);
    let problems = found_for("gust-macros", &[renamed_away]);
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(
        problems[0].contains("dependency `evil-syn`"),
        "{}",
        problems[0]
    );
    assert!(problems[0].contains("allow-list"), "{}", problems[0]);
    assert!(
        problems[1].contains("`syn` is renamed with package = \"evil-syn\""),
        "{}",
        problems[1]
    );
    // The reverse: an innocent key that is really an allowed package passes the allow-list
    // but is still a rename.
    let shim = dep("syn", Some("shim"), Some(CRATES_IO), None, None, None);
    let problems = found_for("gust-macros", &[shim]);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("`shim` is renamed with package = \"syn\""),
        "{}",
        problems[0]
    );
    // A rename of something that is allowed nowhere is only an allow-list problem.
    let other = dep("serde", Some("alias"), Some(CRATES_IO), None, None, None);
    assert_eq!(problems_len("gust-macros", other), 1);
}

fn problems_len(name: &str, dependency: String) -> usize {
    found_for(name, &[dependency]).len()
}

#[test]
fn dependencies_with_a_foreign_source_fail() {
    let path = |name: &str, dir: &str| dep(name, None, None, None, Some(dir), None);
    let bad = [
        (
            "gust-macros",
            path("syn", "/elsewhere/syn"),
            "path dependency on `/elsewhere/syn`",
        ),
        (
            "gust-macros",
            path("syn", "/ws/vendor/syn"),
            "path dependency on `vendor/syn`",
        ),
        (
            "gust-macros",
            dep(
                "syn",
                None,
                Some("git+https://example.com/syn"),
                None,
                None,
                None,
            ),
            "git source",
        ),
        (
            "gust-wgpu",
            dep("wgpu", None, Some(CRATES_IO), Some("mine"), None, None),
            "names a registry",
        ),
        (
            "gust-wgpu",
            dep(
                "wgpu",
                None,
                Some("registry+https://example.com/index"),
                None,
                None,
                None,
            ),
            "names a registry",
        ),
        (
            "gust-wgpu",
            dep(
                "pollster",
                None,
                Some("sparse+https://example.com/"),
                None,
                None,
                None,
            ),
            "names a registry",
        ),
        (
            "gust-wgpu",
            path("gust", "/ws/vendor/gust"),
            "must be `path = \"crates/gust\"`, found `vendor/gust`",
        ),
        (
            "gust",
            from_registry("gust-macros"),
            "must be `path = \"crates/gust-macros\"`, found no path",
        ),
        (
            "gust-wgpu",
            dep(
                "gust",
                None,
                None,
                None,
                Some("/ws/crates/gust-macros"),
                None,
            ),
            "found `crates/gust-macros`",
        ),
        (
            "gust-wgpu",
            dep("gust", None, None, None, Some("/ws/crates/gust/sub"), None),
            "found `crates/gust/sub`",
        ),
    ];
    for (name, dependency, expected) in bad {
        let problems = found_for(name, std::slice::from_ref(&dependency));
        assert_eq!(problems.len(), 1, "{dependency}: {problems:?}");
        assert!(
            problems[0].contains(expected),
            "{dependency}: {}",
            problems[0]
        );
        assert!(
            problems[0].contains(&format!("`{name}` dependency `")),
            "{}",
            problems[0]
        );
        assert!(problems[0].contains("D02"), "{}", problems[0]);
    }
}

#[test]
fn a_workspace_crate_may_be_reached_by_any_spelling_of_its_directory() {
    let root = r"D:\ws";
    for (manifest, dir) in [
        (r"D:\ws\crates\gust-wgpu\Cargo.toml", r"D:\ws\crates\gust"),
        (r"D:\ws\crates\gust-wgpu\Cargo.toml", r"D:\ws\crates\gust\\"),
        (
            r"\\?\D:\ws\crates\gust-wgpu\Cargo.toml",
            r"\\?\D:\ws\crates\gust",
        ),
        ("D:/ws/crates/gust-wgpu/Cargo.toml", "D:/ws/crates/gust/"),
    ] {
        let escaped = |text: &str| text.replace('\\', "\\\\");
        let json = format!(
            r#"{{"packages":[{{"name":"gust-wgpu","manifest_path":"{}","dependencies":[{}]}}]}}"#,
            escaped(manifest),
            dep("gust", None, None, None, Some(&escaped(dir)), None)
        );
        let packages = parse_metadata(&json).unwrap();
        let found = dependency_problems(rule("gust-wgpu"), &packages[0], root);
        assert!(found.is_empty(), "{dir}: {found:?}");
    }
    // Windows paths ignore case, others do not.
    assert_eq!(
        relative_to(r"D:\ws", r"d:\WS\crates\gust").is_some(),
        cfg!(windows)
    );
}

#[test]
fn relative_paths_need_a_whole_directory_prefix() {
    assert_eq!(
        relative_to("/ws", "/ws/crates/a").as_deref(),
        Some("crates/a")
    );
    assert_eq!(
        relative_to("/ws/", "/ws/crates/a/").as_deref(),
        Some("crates/a")
    );
    assert_eq!(relative_to("/ws", "/ws"), None);
    assert_eq!(relative_to("/ws", "/ws2/crates/a"), None);
    assert_eq!(relative_to("/ws", "/other/ws/crates"), None);
    assert_eq!(relative_to("/ws", "/w"), None);
    // A multi-byte character at the cut must not panic.
    assert_eq!(relative_to("/ws", "/w\u{e9}/x"), None);
}

#[test]
fn a_rule_crate_that_cargo_does_not_list_is_reported() {
    let packages = parse_metadata(&metadata_json(&current_packages()[..3])).unwrap();
    let problems = package_problems(ROOT, &packages);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0]
            .starts_with("xtask/Cargo.toml: cargo metadata lists no workspace package `xtask`"),
        "{}",
        problems[0]
    );
    let none = package_problems(ROOT, &[]);
    assert_eq!(none.len(), CRATE_RULES.len(), "{none:?}");
}

#[test]
fn a_rule_crate_at_another_path_is_reported_and_not_judged() {
    let moved = r#"{"name":"gust","manifest_path":"/ws/elsewhere/gust/Cargo.toml","dependencies":[{"name":"serde","source":null,"rename":null,"registry":null}]}"#;
    let mut packages = current_packages();
    packages[1] = moved.to_owned();
    let parsed = parse_metadata(&metadata_json(&packages)).unwrap();
    let problems = package_problems(ROOT, &parsed);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("package `gust` is at `elsewhere/gust/Cargo.toml`"),
        "{}",
        problems[0]
    );
}

#[test]
fn a_crate_without_a_rule_is_reported() {
    let files = [
        "crates/gust/Cargo.toml",
        "crates/gust-macros/Cargo.toml",
        "crates/gust-wgpu/Cargo.toml",
        "crates/gust-new/Cargo.toml",
        "crates/gust/tests/Cargo.toml",
        "examples/demo/Cargo.toml",
        "xtask/Cargo.toml",
    ]
    .map(str::to_owned);
    let problems = unlisted_crate_problems(&files);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].starts_with("crates/gust-new/Cargo.toml: "),
        "{}",
        problems[0]
    );
}

#[test]
fn ir_files_and_directories_are_rejected_outside_future() {
    for path in [
        "crates/gust/src/ir.rs",
        "crates/gust/src/ir/mod.rs",
        "crates/gust-macros/src/shader_ir.rs",
        "crates/gust-macros/src/shader_ir/emit.rs",
        "crates/gust/src/slang/IR.rs",
        "crates/gust/src/slang/ir",
        "crates/gust/src/lower_ir.md",
    ] {
        assert_eq!(ir_path_problems(path).len(), 1, "{path}");
    }
}

#[test]
fn lookalike_paths_and_future_paths_are_allowed() {
    for path in [
        "crates/gust/src/air.rs",
        "crates/gust/src/irq.rs",
        "crates/gust/src/ir_notes.rs",
        "crates/gust/src/first/mod.rs",
        "crates/gust/src/future/ir.rs",
        "crates/gust/future/shader_ir/mod.rs",
        "docs/ir.md",
        "xtask/src/ir.rs",
        "crates/gust/src/slang/mod.rs",
    ] {
        assert!(ir_path_problems(path).is_empty(), "{path}");
    }
}

#[test]
fn ir_module_declarations_are_rejected_outside_future() {
    let text = "use x;\nmod ir;\npub mod shader_ir {\n}\npub(crate) mod IR;\n    mod emit_ir { }\n";
    let problems = ir_module_problems("crates/gust/src/lib.rs", text);
    assert_eq!(problems.len(), 4, "{problems:?}");
    assert!(
        problems[0].starts_with("crates/gust/src/lib.rs:2: "),
        "{}",
        problems[0]
    );
    assert!(problems[1].contains("`shader_ir`"), "{}", problems[1]);
    assert!(problems[3].contains("D02"), "{}", problems[3]);
    assert!(ir_module_problems("crates/gust/src/future/lib.rs", text).is_empty());
}

#[test]
fn other_module_names_and_comments_are_allowed() {
    let text = "mod air;\nmod irq;\nmod ir_notes;\n// mod ir;\n/// mod x_ir;\nlet ir = 1;\nfn ir() {}\nmod modir;\n";
    assert!(ir_module_problems("crates/gust/src/lib.rs", text).is_empty());
}

fn locked(name: &str, version: &str, source: Option<&str>) -> LockedPackage {
    LockedPackage {
        name: name.to_owned(),
        version: version.to_owned(),
        source: source.map(str::to_owned),
    }
}

const LOCK_HEAD: &str = "# This file is automatically @generated by Cargo.\n# It is not intended for manual editing.\nversion = 4\n\n";

#[test]
fn a_lockfile_is_read_into_name_version_and_source() {
    let text = format!(
        "{LOCK_HEAD}[[package]]\nname = \"gust\"\nversion = \"0.1.0\"\ndependencies = [\n \"gust-macros\",\n]\n\n[[package]]\nname = \"syn\"\nversion = \"2.0.117\"\nsource = \"{CRATES_IO}\"\nchecksum = \"abc\"\ndependencies = [\n \"quote\",\n]\n\n[[patch.unused]]\nname = \"serde\"\nversion = \"1.0.0\"\nsource = \"git+https://example.invalid/serde\"\n\n[metadata]\nname = \"not-a-package\"\n"
    );
    assert_eq!(
        parse_lock(&text).unwrap(),
        [
            locked("gust", "0.1.0", None),
            locked("syn", "2.0.117", Some(CRATES_IO)),
        ]
    );
}

#[test]
fn an_unreadable_lockfile_is_an_error_not_a_pass() {
    for text in [
        "",
        "version = 4\n",
        "[[package]]\nversion = \"1.0.0\"\n",
        "[[package]]\nname = \"syn\"\n",
        "[[package]]\nname = syn\nversion = \"1.0.0\"\n",
        "[[package]]\nname = \"sy\\\"n\"\nversion = \"1.0.0\"\n",
    ] {
        assert!(parse_lock(text).is_err(), "{text:?}");
    }
}

#[test]
fn the_real_lockfile_passes_and_is_not_empty() {
    let text = std::fs::read_to_string(
        crate::workspace::workspace_root()
            .unwrap()
            .join("Cargo.lock"),
    )
    .unwrap();
    let packages = parse_lock(&text).unwrap();
    for name in [
        "gust",
        "gust-macros",
        "gust-wgpu",
        "xtask",
        "syn",
        "quote",
        "wgpu",
    ] {
        assert!(
            packages.iter().any(|package| package.name == name),
            "{name}"
        );
    }
    assert_eq!(lock_problems(&packages), Vec::<String>::new());
}

#[test]
fn registry_and_path_packages_in_the_right_places_pass() {
    let packages = [
        locked("syn", "2.0.117", Some(CRATES_IO)),
        locked("wgpu", "30.0.1", Some("sparse+https://index.crates.io/")),
        locked("gust", "0.1.0", None),
        locked("xtask", "0.1.0", None),
        // Packages outside the allow-list are not this check's business.
        locked("serde", "1.0.0", Some("git+https://example.invalid/serde")),
        locked("vector-add", "0.1.0", None),
    ];
    assert_eq!(lock_problems(&packages), Vec::<String>::new());
}

#[test]
fn a_patched_or_replaced_allow_listed_package_is_reported() {
    // A root [patch.crates-io] or [replace] entry changes the source the lockfile records;
    // `cargo metadata --no-deps` does not show it.
    let git = "git+https://example.invalid/syn?branch=main#0123456789abcdef";
    let problems = lock_problems(&[
        locked("syn", "2.0.117", Some(git)),
        locked("quote", "1.0.40", None),
        locked(
            "wgpu",
            "30.0.1",
            Some("registry+https://example.invalid/index"),
        ),
        locked("proc-macro2", "1.0.107", Some(CRATES_IO)),
    ]);
    assert_eq!(problems.len(), 3, "{problems:#?}");
    assert!(
        problems[0]
            .contains("package `syn` 2.0.117 resolves from `git+https://example.invalid/syn"),
        "{problems:?}"
    );
    assert!(problems[0].contains("[patch] or [replace]"), "{problems:?}");
    assert!(
        problems[1].contains("package `quote` 1.0.40 resolves from a local path"),
        "{problems:?}"
    );
    assert!(problems[2].contains("package `wgpu`"), "{problems:?}");
    assert!(
        problems
            .iter()
            .all(|p| p.contains("D02 keeps the architecture fixed"))
    );
}

#[test]
fn a_workspace_crate_resolving_from_a_registry_or_git_is_reported() {
    let problems = lock_problems(&[
        locked("gust-macros", "0.1.0", Some(CRATES_IO)),
        locked("gust", "0.1.0", Some("git+https://example.invalid/gust")),
        locked("gust-wgpu", "0.1.0", None),
    ]);
    assert_eq!(problems.len(), 2, "{problems:#?}");
    assert!(problems[0].contains("package `gust-macros`") && problems[0].contains("path package"));
    assert!(problems[1].contains("package `gust`"));
}
