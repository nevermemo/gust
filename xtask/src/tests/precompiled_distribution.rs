use super::{super::stripped::dependency_disables_defaults, super::test_support::fixture_file, *};
use crate::{release::package_allows, workspace::workspace_root};

#[test]
fn the_committed_manifests_keep_the_bridge_out_and_the_kernels_packageable() {
    let app = fixture_file("app/Cargo.toml");
    assert!(dependency_disables_defaults(&app, "gust"));
    assert!(dependency_disables_defaults(&app, "gust-wgpu"));
    assert!(!app.contains("slang-runtime\"]"));

    let kernels = fixture_file("kernels/Cargo.toml");
    assert!(dependency_disables_defaults(&kernels, "gust"));
    // R2-4: `gust` by version plus path, and no `publish = false` (`cargo package` refuses such a package).
    // The version is the workspace's, read from the root manifest (xtask has its own package version, so
    // `env!("CARGO_PKG_VERSION")` would name the wrong package), so a version bump does not edit this test.
    let version = section_version(&root_manifest(), "workspace.package").unwrap();
    assert!(kernels.contains(&format!(
        "version = \"{version}\", path = \"../../../crates/gust\""
    )));
    assert!(
        !kernels
            .lines()
            .any(|line| !line.trim_start().starts_with('#') && line.contains("publish")),
        "no publish key"
    );

    let workspace = fixture_file("Cargo.toml");
    assert!(workspace.contains("members = [\"kernels\", \"app\"]"));
    assert!(
        !workspace.contains("[env]"),
        "the profile is set by the environment, never by a config file"
    );
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| (*item).to_owned()).collect()
}

fn spec(name: &str) -> PackageSpec<'static> {
    E8_PACKAGES
        .into_iter()
        .find(|spec| spec.name == name)
        .unwrap()
}

#[test]
fn the_package_allow_list_names_a_stray_file() {
    let good = strings(&[
        "Cargo.toml",
        "Cargo.toml.orig",
        ".cargo_vcs_info.json",
        "Cargo.lock",
        "src/lib.rs",
        "src/a/b.rs",
        "README.md",
        "LICENSE-MIT",
    ]);
    assert!(package_problems(&spec("gust-compute-macros"), &good, &[]).is_empty());
    for stray in [
        "tests/ui.rs",
        "examples/a.rs",
        "notes.txt",
        "src/../x",
        "benches/b.rs",
        "gust-artifacts/manifest.json",
        "build.rs",
        "docs/README.md",
    ] {
        let mut files = good.clone();
        files.push(stray.to_owned());
        let problems = package_problems(&spec("gust-compute-macros"), &files, &[]);
        assert_eq!(problems.len(), 1, "{stray}: {problems:?}");
        assert!(problems[0].contains(stray), "{problems:?}");
    }
    // `gust` alone may list a build script.
    let mut with_script = good.clone();
    with_script.push("build.rs".to_owned());
    assert!(package_problems(&spec("gust-compute"), &with_script, &[]).is_empty());
    assert!(!package_allows(
        &spec("gust-compute-wgpu"),
        "gust-artifacts/a.wgsl"
    ));
    assert!(package_allows(&spec("kernels"), "gust-artifacts/a.wgsl"));
}

#[test]
fn a_package_needs_its_manifest_and_library_and_the_kernels_package_every_artifact() {
    let problems = package_problems(&spec("gust-compute"), &strings(&["Cargo.toml"]), &[]);
    assert_eq!(problems.len(), 1);
    assert!(problems[0].contains("src/lib.rs"));
    let committed = strings(&["manifest.json", "k.wgsl", "k.wgsl.reflection.json"]);
    let mut files = strings(&[
        "Cargo.toml",
        "src/lib.rs",
        "gust-artifacts/manifest.json",
        "gust-artifacts/k.wgsl",
    ]);
    let problems = package_problems(&spec("kernels"), &files, &committed);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("k.wgsl.reflection.json"));
    files.push("gust-artifacts/k.wgsl.reflection.json".to_owned());
    assert!(package_problems(&spec("kernels"), &files, &committed).is_empty());
    assert_eq!(
        package_files("Cargo.toml\r\nsrc\\lib.rs\r\n\r\n"),
        ["Cargo.toml", "src/lib.rs"]
    );
}

#[test]
fn a_kernels_package_with_slang_or_twin_text_is_refused() {
    let clean = vec![("src/lib.rs".to_owned(), "pub fn add() {}".to_owned())];
    assert!(slang_free_package_problems(&clean).is_empty());
    for (path, text) in [
        ("gust-artifacts/k.slang", ""),
        ("generated-wgpu/k.wgsl", ""),
        ("src/lib.rs", "use gust::cpu::CpuDevice;"),
        ("src/lib.rs", "fn cpu_add()"),
    ] {
        let problems = slang_free_package_problems(&[(path.to_owned(), text.to_owned())]);
        assert!(!problems.is_empty(), "{path}: {text}");
    }
}

fn root_manifest() -> String {
    let path = workspace_root().unwrap().join("Cargo.toml");
    fs::read_to_string(path).unwrap()
}

#[test]
fn the_versions_come_from_the_manifests_and_the_consumer_patches_all_four_packages() {
    let root = root_manifest();
    // The workspace version is read, not pinned here: three numeric parts before any pre-release suffix.
    let workspace = section_version(&root, "workspace.package").unwrap();
    let core = workspace.split(['-', '+']).next().unwrap();
    let parts: Vec<&str> = core.split('.').collect();
    assert!(
        parts.len() == 3
            && parts
                .iter()
                .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit())),
        "workspace.package version `{workspace}`"
    );
    assert_eq!(
        section_version(&fixture_file("kernels/Cargo.toml"), "package").as_deref(),
        Some("0.1.0")
    );
    assert_eq!(
        section_version("[a]\nversion = \"1\"\n[package]\nname = \"x\"\n", "package"),
        None
    );
    let versions = ["0.1.0", "0.1.0", "0.1.0", "0.0.0"].map(str::to_owned);
    let manifest = consumer_manifest(&versions);
    for name in [
        "gust-compute-macros",
        "gust-compute",
        "gust-compute-wgpu",
        "kernels",
    ] {
        assert!(
            manifest.contains(&format!("{name} = {{ path = \"../unpacked/")),
            "{name}"
        );
    }
    assert!(manifest.contains(
        "gust-compute = \"=0.1.0\"\ngust-compute-wgpu = \"=0.1.0\"\nkernels = \"=0.0.0\"\n"
    ));
    assert!(
        !manifest.contains("default-features"),
        "plain registry-style dependencies (D45)"
    );
    assert!(manifest.contains("[workspace]\n"));
    assert!(manifest.contains("[patch.crates-io]"));
}

#[test]
fn the_packaged_crates_carry_versions_on_their_path_dependencies() {
    let root = root_manifest();
    // The internal dependencies are exact (`=`) so that the four released crates can never be mixed (mr.a).
    let version = section_version(&root, "workspace.package").unwrap();
    for name in ["gust", "gust-macros", "gust-wgpu"] {
        let line = root
            .lines()
            .find(|line| line.trim_start().starts_with(&format!("{name} = ")))
            .unwrap();
        assert!(
            line.contains(&format!("version = \"={version}\"")) && line.contains("path = "),
            "{line}"
        );
        let manifest = fs::read_to_string(
            workspace_root()
                .unwrap()
                .join("crates")
                .join(name)
                .join("Cargo.toml"),
        )
        .unwrap();
        // `gust-wgpu` also keeps its `clippy.toml` out of the package (it only configures this workspace's lint).
        assert!(
            manifest.contains("exclude = [\"tests/\", \"examples/\", \"benches/\""),
            "{name}"
        );
    }
}

#[test]
fn a_lock_entry_with_a_source_or_a_missing_package_is_a_problem() {
    let entry = |name: &str, source: Option<&str>| {
        let mut text = format!("[[package]]\nname = \"{name}\"\nversion = \"0.1.0\"\n");
        if let Some(source) = source {
            text.push_str(&format!("source = \"{source}\"\n"));
        }
        text.push('\n');
        text
    };
    let registry = "registry+https://github.com/rust-lang/crates.io-index";
    let good: String = [
        "gust-compute",
        "gust-compute-macros",
        "gust-compute-wgpu",
        "kernels",
    ]
    .iter()
    .map(|name| entry(name, None))
    .chain([entry("wgpu", Some(registry))])
    .collect();
    assert!(
        lock_problems(&good).is_empty(),
        "{:?}",
        lock_problems(&good)
    );
    let sourced = good.replacen(
        "name = \"gust-compute\"\nversion = \"0.1.0\"\n",
        &format!("name = \"gust-compute\"\nversion = \"0.1.0\"\nsource = \"{registry}\"\n"),
        1,
    );
    let problems = lock_problems(&sourced);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("`gust-compute`") && problems[0].contains("registry"));
    let missing = good.replacen("name = \"kernels\"", "name = \"other\"", 1);
    assert!(
        lock_problems(&missing)
            .iter()
            .any(|problem| problem.contains("`kernels`"))
    );
    assert!(lock_problems("").len() >= 5, "an empty lock must not pass");
}

#[test]
fn an_unused_patch_is_found_in_cargos_warning() {
    let output = "warning: Patch `gust-compute-macros v0.1.0 (C:\\x)` was not used in the crate graph.\nhelp: Check that the patched package version and available features are compatible\n";
    assert_eq!(unused_patch_lines(output).len(), 1);
    assert!(unused_patch_lines("Compiling gust-compute v0.1.0\n").is_empty());
}

fn metadata(gust_features: &str, wgpu_features: &str, manifest: &str) -> String {
    format!(
        "{{\"packages\":[{{\"name\":\"gust-compute\",\"manifest_path\":\"{manifest}/gust-compute-0.1.0/Cargo.toml\",\"features\":{gust_features}}},{{\"name\":\"gust-compute-wgpu\",\"manifest_path\":\"{manifest}/gust-compute-wgpu-0.1.0/Cargo.toml\",\"features\":{wgpu_features}}}]}}"
    )
}

#[test]
fn a_default_feature_that_reaches_slang_runtime_is_found_directly_or_through_another_feature() {
    let ok = "{\"slang-runtime\":[],\"other\":[]}";
    let names = ["gust-compute", "gust-compute-wgpu"];
    assert_eq!(
        default_feature_problems(&metadata(ok, ok, "C:/u"), &names, "C:/u"),
        Ok(Vec::new())
    );
    let direct = "{\"default\":[\"slang-runtime\"],\"slang-runtime\":[]}";
    let problems = default_feature_problems(&metadata(ok, direct, "C:/u"), &names, "C:/u").unwrap();
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("gust-compute-wgpu"));
    let through = "{\"default\":[\"bridge\"],\"bridge\":[\"gust/slang-runtime\"]}";
    let problems =
        default_feature_problems(&metadata(through, ok, "C:/u"), &names, "C:/u").unwrap();
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("`bridge`"));
    // A package that is not the unpacked one (a registry copy) is a problem, and so is a missing package.
    let problems =
        default_feature_problems(&metadata(ok, ok, "C:/registry"), &names, "C:/u").unwrap();
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert_eq!(
        default_feature_problems("{\"packages\":[]}", &names, "C:/u")
            .unwrap()
            .len(),
        2
    );
    assert!(default_feature_problems("not json", &names, "C:/u").is_err());
}

#[test]
fn the_work_directory_must_lie_outside_the_repository() {
    assert!(is_inside(
        Path::new("C:/repo"),
        Path::new("C:/repo/target/x")
    ));
    assert!(is_inside(Path::new("C:/repo/"), Path::new("C:/repo")));
    assert!(!is_inside(Path::new("C:/repo"), Path::new("C:/repository")));
    assert!(!is_inside(Path::new("C:/repo"), Path::new("C:/tmp/repo")));
}

#[test]
fn the_packaging_of_kernels_patches_the_two_packages_it_reaches_onto_the_checkout() {
    // The values are given to `cargo package --config`: the patch keys are the package names (not the directories),
    // the paths are the checkout's own `crates/<directory>` with forward slashes, quoted as TOML literal strings.
    let configs = kernels_patch_configs(Path::new(r"C:\work\gust")).unwrap();
    assert_eq!(
        configs,
        [
            "patch.crates-io.gust-compute.path='C:/work/gust/crates/gust'",
            "patch.crates-io.gust-compute-macros.path='C:/work/gust/crates/gust-macros'",
        ]
    );
    // `gust-compute-wgpu` is not reached by `kernels`: a patch for it would only be reported as unused.
    assert!(configs.iter().all(|config| !config.contains("wgpu")));
    // A literal string has no escape for a quote or a line break, so such a root is refused and not written wrongly.
    for root in ["/it's/gust", "/a\nb/gust", "/a\rb/gust"] {
        let error = kernels_patch_configs(Path::new(root)).unwrap_err();
        assert!(error.contains("check-precompiled E8"), "{error}");
        assert!(error.contains("quote or a line break"), "{error}");
    }
}
