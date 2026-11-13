use super::{
    Member, carries_compiler_tag, compiler_tag_problems, find_machine_path, is_forbidden_script,
    licence_problem, machine_path_problems, member_field_problems, parse_members,
    path_dependency_problems,
};

#[test]
fn shell_script_extensions_are_forbidden_in_any_directory() {
    for path in [
        "build.ps1",
        "tools/Setup.psm1",
        "tools/Setup.psd1",
        "run.bat",
        "scripts/run.cmd",
        "ci/check.sh",
        ".agents/skills/slang-language/scripts/Compile-Slang.ps1",
    ] {
        assert!(is_forbidden_script(path), "{path} should be forbidden");
    }
}

#[test]
fn script_extensions_match_case_insensitively_and_on_hidden_names() {
    assert!(is_forbidden_script("Build.PS1"));
    assert!(is_forbidden_script("scripts/RUN.Sh"));
    assert!(is_forbidden_script(".sh"));
    assert!(is_forbidden_script(r"scripts\run.bat"));
}

#[test]
fn other_files_and_lookalike_names_are_allowed() {
    for path in [
        "build.rs",
        "README.md",
        "scripts/probes/slang-layout.cpp",
        "docs/shell.md",
        "xtask/src/shell.rs",
        "script.sh.md",
        "notes/ps1",
        "tools/cmd",
        "scripts/run.sh/README.md",
        ".github/workflows/ci.yml",
    ] {
        assert!(!is_forbidden_script(path), "{path} should be allowed");
    }
}

#[test]
fn windows_home_paths_match_in_every_slash_style() {
    for (line, expected) in [
        (r"C:\Users\someone\Desktop\gust", r"C:\Users\someone"),
        ("C:/Users/someone/Desktop/gust", "C:/Users/someone"),
        (r"c:\users\someone", r"c:\users\someone"),
        (r"C:\\Users\\someone\\gust", r"C:\\Users\\someone"),
        (r"\\?\D:\Users\some-one\x", r"D:\Users\some-one"),
        (
            r#"error at "C:\Users\bob.smith\x.rs""#,
            r"C:\Users\bob.smith",
        ),
    ] {
        assert_eq!(find_machine_path(line), Some(expected), "{line}");
    }
}

#[test]
fn unix_home_paths_match_with_a_trailing_slash() {
    assert_eq!(
        find_machine_path("cd /home/someone/project"),
        Some("/home/someone")
    );
    assert_eq!(
        find_machine_path("at /Users/someone/project/src"),
        Some("/Users/someone")
    );
    assert_eq!(find_machine_path("(/home/a_b-c/x)"), Some("/home/a_b-c"));
}

#[test]
fn placeholders_and_relative_paths_are_not_machine_paths() {
    for line in [
        r"<workspace>\crates\gust",
        "<workspace>/crates/gust",
        "~/.cargo/bin/slangc",
        r"~\.cargo\bin\slangc.exe",
        r"X:\Users\<name>",
        r"X:/Users/<name>/",
        "/home/<name>/",
        "/Users/<name>/",
        r"%USERPROFILE%\.cargo",
        "$HOME/.cargo/bin",
        "docs/home/foo/bar",
        "./home/foo/bar",
        "https://example.com/home/page/",
        "https://example.com/Users/page/",
        r"PATH:\Users\someone",
        r"C:\Users",
        r"C:\Users\",
        "/home/",
        "no paths here",
        "",
    ] {
        assert_eq!(find_machine_path(line), None, "{line}");
    }
}

#[test]
fn multibyte_text_around_a_path_is_handled() {
    assert_eq!(
        find_machine_path("日本語 /home/someone/x"),
        Some("/home/someone")
    );
    assert_eq!(find_machine_path("日本語/home/someone/x"), None);
    assert_eq!(
        find_machine_path(r"é C:\Users\José\x"),
        Some(r"C:\Users\José")
    );
}

#[test]
fn problems_report_file_and_line() {
    let text = "clean line\nroot = C:\\Users\\someone\\gust\nalso /home/someone/x/y\n";
    let problems = machine_path_problems("docs/notes.md", text);
    assert_eq!(problems.len(), 2);
    assert!(
        problems[0].starts_with(r"docs/notes.md:2: machine-specific path `C:\Users\someone`"),
        "{}",
        problems[0]
    );
    assert!(
        problems[1].starts_with("docs/notes.md:3: machine-specific path `/home/someone`"),
        "{}",
        problems[1]
    );
}

#[test]
fn only_the_listed_fixtures_may_spell_placeholder_machine_paths() {
    let text = r"C:\Users\someone\gust";
    assert!(machine_path_problems("xtask/src/tests/lint.rs", text).is_empty());
    assert!(machine_path_problems("xtask/src/tests/validation_redaction.rs", text).is_empty());
    assert_eq!(machine_path_problems("xtask/src/lint.rs", text).len(), 1);
    assert_eq!(machine_path_problems("docs/other.md", text).len(), 1);
}

#[test]
fn fixture_files_accept_only_the_placeholder_user_names() {
    let fixture = "xtask/src/tests/lint.rs";
    for name in ["someone", "bob.smith", "José", "a_b-c", "some-one"] {
        for text in [
            format!(r"C:\Users\{name}\gust"),
            format!("/home/{name}/gust"),
            format!("/mnt/c/Users/{name}/gust"),
            format!(r"\\wsl.localhost\Ubuntu\home\{name}\gust"),
        ] {
            assert!(machine_path_problems(fixture, &text).is_empty(), "{text}");
        }
    }
    // Case variants of a placeholder are placeholders too (the redaction tests need them).
    assert!(machine_path_problems(fixture, &format!(r"c:\USERS\{}", "SOMEONE")).is_empty());
    // The names below are built at run time so that this file stays clean itself.
    let real = ["alice", "carol-d"];
    for name in real {
        for text in [
            format!(r"C:\Users\{name}\gust"),
            format!("/home/{name}/gust"),
            format!("/mnt/c/Users/{name}"),
            format!(r"\\wsl.localhost\Ubuntu\home\{name}\gust"),
        ] {
            let problems = machine_path_problems(fixture, &text);
            assert_eq!(problems.len(), 1, "{text}");
            assert!(
                problems[0].contains("fixtures may only use"),
                "{}",
                problems[0]
            );
        }
    }
}

#[test]
fn every_path_on_a_line_is_checked_not_only_the_first() {
    let mixed = format!(r"C:\Users\someone\a and /home/{}/x", "alice");
    let problems = machine_path_problems("xtask/src/tests/lint.rs", &mixed);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains(&format!("/home/{}", "alice")),
        "{}",
        problems[0]
    );
    let two = r"C:\Users\someone\a and D:\Users\bob.smith\b";
    assert_eq!(machine_path_problems("docs/x.md", two).len(), 2);
    assert!(machine_path_problems("xtask/src/tests/lint.rs", two).is_empty());
}

#[test]
fn wsl_mount_and_network_home_paths_match() {
    for (line, expected) in [
        ("cd /mnt/c/Users/someone/Desktop", "/mnt/c/Users/someone"),
        ("/mnt/d/users/bob.smith", "/mnt/d/users/bob.smith"),
        ("(/mnt/c/Users/a_b-c/x)", "/mnt/c/Users/a_b-c"),
        (
            r"\\wsl.localhost\Ubuntu\home\someone\gust",
            r"\\wsl.localhost\Ubuntu\home\someone",
        ),
        (
            r"\wsl.localhost\Ubuntu-24.04\home\some-one\",
            r"\wsl.localhost\Ubuntu-24.04\home\some-one",
        ),
        (r"\\WSL$\Ubuntu\home\José\x", r"\\WSL$\Ubuntu\home\José"),
        (
            r"\\wsl.localhost\\Ubuntu\\home\\someone\\x",
            r"\\wsl.localhost\\Ubuntu\\home\\someone",
        ),
        (
            r"//wsl.localhost/Ubuntu/home/someone/x",
            r"//wsl.localhost/Ubuntu/home/someone",
        ),
        (
            r"\\wsl.localhost\<distro>\home\someone\",
            r"\\wsl.localhost\<distro>\home\someone",
        ),
        (
            r#"open "\\wsl$\Debian\home\someone""#,
            r"\\wsl$\Debian\home\someone",
        ),
    ] {
        assert_eq!(find_machine_path(line), Some(expected), "{line}");
    }
}

#[test]
fn wsl_placeholders_and_non_home_paths_are_not_machine_paths() {
    for line in [
        "/mnt/c/Users/<name>",
        "/mnt/c/Users/%USERNAME%/x",
        "/mnt/c/Users",
        "/mnt/c/Users/",
        "/mnt/cc/Users/someone",
        "/mnt/c/Windows/someone",
        "docs/mnt/c/Users/someone",
        r"\\wsl.localhost\<distro>\home\<name>\",
        r"\\wsl.localhost\Ubuntu\home",
        r"\\wsl.localhost\Ubuntu\opt\someone",
        r"\\wsl.localhost\Ubuntu",
        r"\\wsl.localhost",
        r"\\wsl.localhost\\home\someone",
        r"\\wsl.example\Ubuntu\home\someone",
        r"x\wsl.localhost\Ubuntu\home\someone",
    ] {
        assert_eq!(find_machine_path(line), None, "{line}");
    }
}

#[test]
fn a_unix_home_directory_is_flagged_at_the_end_of_a_line_or_before_a_quote() {
    for (line, expected) in [
        ("/home/someone", "/home/someone"),
        ("cd /home/someone", "/home/someone"),
        ("/Users/someone", "/Users/someone"),
        (r#"HOME="/home/someone""#, "/home/someone"),
        ("home = '/home/some-one'", "/home/some-one"),
        ("see `/Users/bob.smith`", "/Users/bob.smith"),
    ] {
        assert_eq!(find_machine_path(line), Some(expected), "{line}");
    }
    // A bare `/home` or `/home/` names no user, and a relative path is not a home path.
    for line in [
        "/home",
        "/home/",
        "docs/home/someone",
        r#""/home/""#,
        "x/Users/someone",
    ] {
        assert_eq!(find_machine_path(line), None, "{line}");
    }
    let problems = machine_path_problems("docs/notes.md", "ends here: /home/someone\n");
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("`/home/someone`"), "{}", problems[0]);
}

// ---- Release rules (mr.a) -----------------------------------------------------------------------------------

/// A publishable member with every field the rules read.
fn complete_member() -> Member {
    Member {
        name: "gust-compute".to_owned(),
        directory: "<workspace>/crates/gust".to_owned(),
        publishable: true,
        description: Some("A crate".to_owned()),
        license: Some("MIT OR Apache-2.0".to_owned()),
        repository: Some("https://example.com/repo".to_owned()),
        documentation: Some("https://docs.rs/gust-compute".to_owned()),
        readme: Some("README.md".to_owned()),
        keywords: vec!["gpu".to_owned()],
        categories: vec!["compilers".to_owned()],
    }
}

#[test]
fn metadata_output_is_read_into_members() {
    let text = r#"{"packages": [
        {"name": "lib", "manifest_path": "C:\\work\\crates\\lib\\Cargo.toml", "publish": null,
         "description": "d", "license": "MIT", "repository": "r", "documentation": "docs", "readme": "README.md",
         "keywords": ["a", "b"], "categories": ["c"]},
        {"name": "example", "manifest_path": "/work/examples/example/Cargo.toml", "publish": [],
         "description": null, "license": null, "repository": null, "documentation": null, "readme": null,
         "keywords": [], "categories": []},
        {"name": "registry", "manifest_path": "/work/registry/Cargo.toml", "publish": ["crates-io"]},
        {"name": "bare", "manifest_path": "/work/bare/Cargo.toml", "repository": ""}
    ]}"#;
    let members = parse_members(text).unwrap();
    assert_eq!(members.len(), 4);
    assert_eq!(members[0].directory, r"C:\work\crates\lib");
    assert!(members[0].publishable);
    assert_eq!(members[0].readme.as_deref(), Some("README.md"));
    assert_eq!(members[0].keywords, ["a", "b"]);
    assert_eq!(members[0].categories, ["c"]);
    assert_eq!(members[1].directory, "/work/examples/example");
    assert!(!members[1].publishable, "publish = false is `[]`");
    assert_eq!(members[1].description, None);
    assert!(
        members[2].publishable,
        "a list of registries allows publishing"
    );
    assert!(members[3].publishable, "no `publish` key allows publishing");
    assert_eq!(members[3].repository, None, "an empty string is missing");
    assert!(parse_members("{}").is_err());
    assert!(parse_members("[").is_err());
    assert!(parse_members(r#"{"packages": [{"name": "x"}]}"#).is_err());
}

#[test]
fn a_complete_publishable_member_and_any_unpublished_member_have_no_field_problems() {
    assert!(member_field_problems(&complete_member()).is_empty());
    let example = Member {
        publishable: false,
        description: None,
        license: None,
        repository: None,
        documentation: None,
        readme: None,
        keywords: Vec::new(),
        categories: Vec::new(),
        ..complete_member()
    };
    assert!(member_field_problems(&example).is_empty());
}

#[test]
fn each_missing_metadata_field_of_a_publishable_member_is_named() {
    let without = [
        (
            "description",
            Member {
                description: None,
                ..complete_member()
            },
        ),
        (
            "license",
            Member {
                license: None,
                ..complete_member()
            },
        ),
        (
            "repository",
            Member {
                repository: None,
                ..complete_member()
            },
        ),
        (
            "documentation",
            Member {
                documentation: None,
                ..complete_member()
            },
        ),
        (
            "readme",
            Member {
                readme: None,
                ..complete_member()
            },
        ),
        (
            "keywords",
            Member {
                keywords: Vec::new(),
                ..complete_member()
            },
        ),
        (
            "categories",
            Member {
                categories: Vec::new(),
                ..complete_member()
            },
        ),
    ];
    for (field, member) in without {
        let problems = member_field_problems(&member);
        assert_eq!(problems.len(), 1, "{field}: {problems:?}");
        assert!(
            problems[0].starts_with(&format!(
                "package `gust-compute` can be published but has no `{field}`; add it"
            )),
            "{}",
            problems[0]
        );
        assert!(problems[0].ends_with("or set `publish = false` if it is not meant for crates.io"));
    }
    // An example that lost its `publish = false` line fails for every field it never had.
    let example = Member {
        name: "vector-add".to_owned(),
        repository: None,
        documentation: None,
        readme: None,
        keywords: Vec::new(),
        categories: Vec::new(),
        ..complete_member()
    };
    assert_eq!(member_field_problems(&example).len(), 5);
}

#[test]
fn a_licence_copy_must_equal_the_root_file_byte_for_byte() {
    let root = b"MIT License\r\n\r\nCopyright (c) 2026\r\n";
    assert_eq!(
        licence_problem("gust", "LICENSE-MIT", root, Some(root)),
        None
    );
    // One byte different: the message names the file, the package and the offset.
    let mut changed = root.to_vec();
    changed[12] = b'X';
    assert_eq!(
        licence_problem("gust", "LICENSE-MIT", root, Some(&changed)).as_deref(),
        Some(
            "package `gust`: LICENSE-MIT differs from the root LICENSE-MIT (first difference at byte 12; \
             35 bytes against 35); copy the root file again"
        )
    );
    // Line endings are bytes too.
    let lf = root
        .iter()
        .copied()
        .filter(|byte| *byte != b'\r')
        .collect::<Vec<_>>();
    let problem = licence_problem("gust", "LICENSE-MIT", root, Some(&lf)).unwrap();
    assert!(
        problem.contains("first difference at byte 11;"),
        "{problem}"
    );
    // A copy that is a prefix or an extension of the original differs too.
    let truncated = &root[..root.len() - 1];
    let problem = licence_problem("gust", "LICENSE-MIT", root, Some(truncated)).unwrap();
    assert!(
        problem.contains("first difference at byte 34;"),
        "{problem}"
    );
    let mut longer = root.to_vec();
    longer.push(b'\n');
    let problem = licence_problem("gust", "LICENSE-MIT", root, Some(&longer)).unwrap();
    assert!(
        problem.contains("first difference at byte 35;"),
        "{problem}"
    );
    // A missing copy.
    assert_eq!(
        licence_problem("gust", "LICENSE-APACHE", root, None).as_deref(),
        Some(
            "package `gust`: LICENSE-APACHE is missing; copy the root LICENSE-APACHE into the package directory \
             (cp LICENSE-APACHE <crate dir>/)"
        )
    );
}

fn manifest(path: &str, text: &str) -> (String, String) {
    (path.to_owned(), text.to_owned())
}

/// The three kinds of manifest the rule reads, with the package `crates/gust` named `name`.
fn manifests_with(name: &str, fixture: &str) -> Vec<(String, String)> {
    vec![
        manifest(
            "crates/gust/Cargo.toml",
            &format!("[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n"),
        ),
        manifest(
            "crates/gust-wgpu/Cargo.toml",
            "[package]\nname = \"gust-wgpu\"\n",
        ),
        manifest("tests/ui/case/Cargo.toml", fixture),
    ]
}

#[test]
fn a_path_dependency_that_names_its_package_passes() {
    let fixture = "[package]\nname = \"case\"\n\n[dependencies]\n\
                   gust = { path = \"../../../crates/gust\" }\n\
                   gust-wgpu = { path = \"../../../crates/gust-wgpu\", features = [\"slang-runtime\"] }\n";
    assert!(path_dependency_problems(&manifests_with("gust", fixture)).is_empty());
    // After a rename the key may stay when `package` carries the new name, or follow the new name.
    for line in [
        "gust = { path = \"../../../crates/gust\", package = \"gust-compute\" }",
        "gust-compute = { path = \"../../../crates/gust\" }",
        "gust = { package = \"gust-compute\", path = \"../../../crates/gust\" }",
        "gust = { path = '../../../crates/gust', package = 'gust-compute' }",
        "\"gust\" = { path = \"../../../crates/gust\", package = \"gust-compute\" }",
    ] {
        let fixture = format!("[package]\nname = \"case\"\n\n[dependencies]\n{line}\n");
        assert!(
            path_dependency_problems(&manifests_with("gust-compute", &fixture)).is_empty(),
            "{line}"
        );
    }
}

#[test]
fn a_path_dependency_on_a_renamed_package_without_the_new_name_fails_with_a_clear_message() {
    let fixture = "[package]\nname = \"case\"\n\n[dependencies]\n\
                   gust = { path = \"../../../crates/gust\" }\n";
    assert_eq!(
        path_dependency_problems(&manifests_with("gust-compute", fixture)),
        [
            "tests/ui/case/Cargo.toml:5: dependency `gust` has path `../../../crates/gust`, the package \
             `gust-compute` (crates/gust), but calls it `gust`; name the dependency `gust-compute` or add \
             package = \"gust-compute\""
        ]
    );
    // A `package` that names something else is wrong too.
    let fixture =
        "[dependencies]\ngust = { path = \"../../../crates/gust\", package = \"gust-wgpu\" }\n";
    let problems = path_dependency_problems(&manifests_with("gust-compute", fixture));
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("but calls it `gust-wgpu`"),
        "{}",
        problems[0]
    );
}

#[test]
fn every_manifest_form_of_a_path_dependency_is_read() {
    let wrong = |text: &str, line: usize| {
        let problems = path_dependency_problems(&manifests_with("gust-compute", text));
        assert_eq!(problems.len(), 1, "{text}: {problems:?}");
        assert!(
            problems[0].starts_with(&format!(
                "tests/ui/case/Cargo.toml:{line}: dependency `gust` "
            )),
            "{}",
            problems[0]
        );
    };
    wrong(
        "[dev-dependencies]\ngust = { path = \"../../../crates/gust\" }\n",
        2,
    );
    wrong(
        "[build-dependencies]\ngust = { path = \"../../../crates/gust\" }\n",
        2,
    );
    wrong(
        "[target.'cfg(unix)'.dependencies]\ngust = { path = \"../../../crates/gust\" }\n",
        2,
    );
    wrong(
        "[target.\"cfg(any(unix, windows))\".dev-dependencies]\n\
         gust = { path = \"../../../crates/gust\" }\n",
        2,
    );
    wrong(
        "[dependencies.gust]\nfeatures = [\"a\"]\npath = \"../../../crates/gust\"\n",
        1,
    );
    wrong(
        "[dev-dependencies.gust]\npath = \"../../../crates/gust\"\n",
        1,
    );
    wrong("[dependencies]\ngust.path = \"../../../crates/gust\"\n", 2);
    wrong(
        "[dependencies]\ngust.version = \"1\"\ngust.path = \"../../../crates/gust\"\n",
        3,
    );
    wrong(
        "[workspace.dependencies]\ngust = { version = \"=0.1.0\", path = \"../../../crates/gust\" }\n",
        2,
    );
    wrong(
        "[dependencies]\ngust = { path = \"../../../crates/gust\" } # trailing comment\n",
        2,
    );
    wrong(
        "[dependencies]\ngust = { features = [\"a\", \"b\"], path = \"..\\\\..\\\\..\\\\crates\\\\gust\" }\n",
        2,
    );
    wrong(
        "[dependencies]\ngust = { path = \"../../../crates/./gust/../gust\" }\n",
        2,
    );
    // The self dev-dependency of a package: `path = "."` from its own directory.
    let own = vec![manifest(
        "crates/gust/Cargo.toml",
        "[package]\nname = \"gust-compute\"\n\n[dev-dependencies]\n\
         gust = { path = \".\", features = [\"slang-runtime\"] }\n",
    )];
    let problems = path_dependency_problems(&own);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].starts_with("crates/gust/Cargo.toml:5: dependency `gust` has path `.`"));
    let own = vec![manifest(
        "crates/gust/Cargo.toml",
        "[package]\nname = \"gust-compute\"\n\n[dev-dependencies]\n\
         gust = { path = \".\", package = \"gust-compute\" }\n",
    )];
    assert!(path_dependency_problems(&own).is_empty());
}

#[test]
fn dependencies_the_rule_does_not_concern_are_left_alone() {
    let fixture = "\
[package]
name = \"case\"
path = \"../../../crates/gust\"

[[bin]]
name = \"x\"
path = \"../../../crates/gust\"

[lib]
path = \"../../../crates/gust\"

[dependencies]
# gust = { path = \"../../../crates/gust\" }
kernels = { path = \"../kernels\" }
gust-wgpu.workspace = true
gust-macros = \"1\"
outside = { path = \"../../../../elsewhere/crates/gust\" }
absolute = { path = \"/abs/crates/gust\" }
drive = { path = \"C:/abs/crates/gust\" }
inside = { path = \"../../../crates/gust/src\" }
note = { version = \"1\", features = [\"path = x\"] }
";
    assert!(path_dependency_problems(&manifests_with("gust-compute", fixture)).is_empty());
    // A fixture directory that merely contains the word `crates` is not a package.
    let fixture = "[dependencies]\ngust = { path = \"../crates/gust\" }\n";
    assert!(path_dependency_problems(&manifests_with("gust-compute", fixture)).is_empty());
}

#[test]
fn a_dependency_table_over_several_lines_is_reported_instead_of_skipped() {
    for first in [
        "gust = { path = \"../../../crates/gust\",",
        "gust = { package = \"gust-compute\",",
        "gust = { version = \"1\", path = \"../../../crates/gust\",",
    ] {
        let fixture = format!("[dependencies]\n{first}\n    features = [\"a\"] }}\n");
        let problems = path_dependency_problems(&manifests_with("gust-compute", &fixture));
        assert_eq!(problems.len(), 1, "{first}: {problems:?}");
        assert!(
            problems[0].starts_with("tests/ui/case/Cargo.toml:2: "),
            "{}",
            problems[0]
        );
    }
    // A table that names neither `path` nor `package` on its first line is not refused.
    for first in [
        "other = {",
        "other = { version = \"1\",",
        "other = { features = [\"path = x\"],",
    ] {
        let fixture = format!("[dependencies]\n{first}\n    default-features = false }}\n");
        let problems = path_dependency_problems(&manifests_with("gust-compute", &fixture));
        assert!(problems.is_empty(), "{first}: {problems:?}");
    }
    let fixture =
        "[dependencies]\ngust = { path = \"../../../crates/gust\",\n    features = [\"a\"] }\n";
    let problems = path_dependency_problems(&manifests_with("gust-compute", fixture));
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].starts_with("tests/ui/case/Cargo.toml:2: "),
        "{}",
        problems[0]
    );
    assert!(
        problems[0].contains("write it on one line"),
        "{}",
        problems[0]
    );
}

// ---- compiler tags (NEXT_TASKS: every committed tag equals SLANG_PIN) ----

const PIN: &str = "2026.22";

#[test]
fn a_current_compiler_tag_passes_in_a_reflection_document_and_a_manifest() {
    let reflection = r#"{"schemaVersion":4,"compiler":"2026.22","target":"wgsl"}"#;
    let manifest = "{\n  \"schema\": 1,\n  \"slang\": \"2026.22\"\n}\n";
    assert_eq!(
        compiler_tag_problems("a.wgsl.reflection.json", reflection, PIN),
        Vec::<String>::new()
    );
    assert_eq!(
        compiler_tag_problems("gust-artifacts/manifest.json", manifest, PIN),
        Vec::<String>::new()
    );
}

#[test]
fn a_stale_compiler_tag_is_named_with_its_file_and_both_tags() {
    // The fixture string of a document that an older Slang made (the m9.p regeneration missed one such example).
    let stale = r#"{"schemaVersion":4,"compiler":"2026.21","target":"wgsl"}"#;
    assert_eq!(
        compiler_tag_problems(
            "examples/x/gust-artifacts/k.wgsl.reflection.json",
            stale,
            PIN
        ),
        [
            "examples/x/gust-artifacts/k.wgsl.reflection.json: `compiler` is `2026.21` but SLANG_PIN is `2026.22`; \
             regenerate the artifact with the pinned compiler"
        ]
    );
    let manifest = r#"{"schema":1,"slang":"2026.21.1"}"#;
    let problems = compiler_tag_problems("examples/x/gust-artifacts/manifest.json", manifest, PIN);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("`slang` is `2026.21.1`"),
        "{problems:?}"
    );
    // A tag that only extends the pin is not the pin.
    let longer = r#"{"compiler":"2026.22.1"}"#;
    assert_eq!(
        compiler_tag_problems("k.reflection.json", longer, PIN).len(),
        1
    );
    // Both fields stale: two problems.
    let both = r#"{"compiler":"old","slang":"old"}"#;
    assert_eq!(compiler_tag_problems("manifest.json", both, PIN).len(), 2);
}

#[test]
fn a_null_slang_a_missing_field_and_odd_values_are_judged_as_documented() {
    assert!(compiler_tag_problems("manifest.json", r#"{"schema":1,"slang":null}"#, PIN).is_empty());
    assert!(compiler_tag_problems("manifest.json", r#"{"artifacts":[]}"#, PIN).is_empty());
    // `compiler` is never null; a number is not a tag.
    assert_eq!(
        compiler_tag_problems("k.reflection.json", r#"{"compiler":null}"#, PIN).len(),
        1
    );
    assert_eq!(
        compiler_tag_problems("manifest.json", r#"{"slang":22}"#, PIN).len(),
        1
    );
    // Only the top level counts: a nested field of another meaning is left alone.
    assert!(
        compiler_tag_problems(
            "k.reflection.json",
            r#"{"compiler":"2026.22","x":{"compiler":"other"}}"#,
            PIN
        )
        .is_empty()
    );
    // A broken document cannot hide a stale tag.
    let problems = compiler_tag_problems("manifest.json", r#"{"slang":"2026.21""#, PIN);
    assert_eq!(problems.len(), 1);
    assert!(
        problems[0].starts_with("manifest.json: not JSON"),
        "{problems:?}"
    );
}

#[test]
fn only_reflection_documents_and_manifests_carry_a_checked_tag() {
    for yes in [
        "examples/map-chain/gust-artifacts/443ab55c412cf3e5.wgsl.reflection.json",
        "tests/precompiled/kernels/gust-artifacts/manifest.json",
        "crates/gust/tests/fixtures/schema3/storage_inferred.wgsl.json",
        "manifest.json",
    ] {
        assert!(carries_compiler_tag(yes), "{yes}");
    }
    for no in [
        "crates/gust-macros/tests/golden/sample.kernel.json",
        "Cargo.json",
        "docs/manifest.json.md",
        "not-a-manifest.json",
    ] {
        assert!(!carries_compiler_tag(no), "{no}");
    }
}
