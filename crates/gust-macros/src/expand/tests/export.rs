//! Tests of artifacts and export mode (m4.c).

use super::*;

// ---- artifacts and export mode (m4.c) ----

#[test]
fn the_exported_key_is_the_content_key_of_the_entry_point_and_the_clean_text() {
    let (key, slang) = exported(KERNEL);
    assert_eq!(key, artifact::content_key("gpu_layout_inspect", &slang));
    assert!(
        !slang.contains("#line"),
        "the key hashes the clean text: {slang}"
    );
}

#[test]
fn a_complete_pair_is_embedded_and_required() {
    let (scratch, key) = crate_with_pair("pair-present", true);
    let text = expanded(KERNEL, &environment(Some(&scratch.0), None)).to_string();
    assert!(
        text.contains("artifacts : & [:: gust :: Artifact :: new (:: gust :: Format :: Wgsl"),
        "{text}"
    );
    assert!(text.contains(&format!("Wgsl , \"{key}\" ,")), "{text}");
    assert!(text.contains(&format!(
            ", :: core :: include_bytes ! (:: core :: concat ! (:: core :: env ! (\"CARGO_MANIFEST_DIR\") , \"/gust-artifacts/{key}.wgsl\"))"
        )), "{text}");
    assert!(text.contains(&format!(
            ", :: core :: include_str ! (:: core :: concat ! (:: core :: env ! (\"CARGO_MANIFEST_DIR\") , \"/gust-artifacts/{key}.wgsl.reflection.json\"))"
        )), "{text}");
    assert!(text.contains("require_artifact (true)"), "{text}");
    assert!(!text.contains("require_artifact (false)"), "{text}");
    assert!(text.contains("require_manifest (true)"), "{text}");
    assert_eq!(
        text.matches("manifest.json").count(),
        1,
        "one include: {text}"
    );
}

/// m11.psplit-a (D209): the split is a second artifact of the same kernel under a key of its own.
#[test]
fn a_split_has_its_own_key_over_the_same_entry_point_and_its_own_text() {
    let (body_key, body_slang) = exported(KERNEL);
    let (key, slang, owner) = exported_split(KERNEL);
    assert_ne!(key, body_key);
    assert_eq!(owner, body_key);
    assert_ne!(slang, body_slang);
    assert!(slang.contains("__gust_launch_base"), "{slang}");
    assert!(!body_slang.contains("__gust_launch_base"), "{body_slang}");
    assert_eq!(key, artifact::content_key("gpu_layout_inspect", &slang));
}

#[test]
fn a_present_split_pair_is_embedded_in_the_split_descriptor_alone_and_never_required() {
    let (body_key, _) = exported(KERNEL);
    let (split_key, _, _) = exported_split(KERNEL);
    let scratch = Scratch::new("split-pair-present");
    for key in [&body_key, &split_key] {
        scratch.artifact(&format!("{key}.wgsl"), "wgsl bytes");
        scratch.artifact(&format!("{key}.wgsl.reflection.json"), "{}");
    }
    let text = expanded(KERNEL, &environment(Some(&scratch.0), None)).to_string();
    assert!(
        text.contains(&format!("Wgsl , \"{split_key}\" ,")),
        "{text}"
    );
    assert!(text.contains(&format!("Wgsl , \"{body_key}\" ,")), "{text}");
    assert_eq!(text.matches("include_bytes").count(), 2, "{text}");
    // One requirement, the body's: the split is optional.
    assert_eq!(text.matches("require_artifact").count(), 1, "{text}");
    assert!(text.contains("require_artifact (true)"), "{text}");
    assert!(!text.contains("compile_error"), "{text}");
}

#[test]
fn a_body_pair_without_a_split_pair_leaves_the_split_without_artifacts() {
    let (body_key, _) = exported(KERNEL);
    let (scratch, _) = crate_with_pair("split-pair-absent", false);
    let text = expanded(KERNEL, &environment(Some(&scratch.0), None)).to_string();
    assert!(text.contains(&format!("Wgsl , \"{body_key}\" ,")), "{text}");
    assert_eq!(text.matches("include_bytes").count(), 1, "{text}");
    // The split's own `artifacts` is the empty slice, and no error and no requirement comes of it.
    assert_eq!(text.matches("artifacts : & []").count(), 1, "{text}");
    assert!(!text.contains("compile_error"), "{text}");
    assert_eq!(text.matches("require_artifact").count(), 1, "{text}");
}

#[test]
fn half_a_split_pair_is_the_error_of_half_a_body_pair_and_absent_in_export_mode() {
    let (body_key, _) = exported(KERNEL);
    let (split_key, _, _) = exported_split(KERNEL);
    let half = Scratch::new("split-pair-half");
    half.artifact(&format!("{body_key}.wgsl"), "x");
    half.artifact(&format!("{body_key}.wgsl.reflection.json"), "{}");
    half.artifact(&format!("{split_key}.wgsl"), "x");
    let output = expanded(KERNEL, &environment(Some(&half.0), None));
    let text = output.to_string();
    assert!(text.contains("[GUST-E0711]"), "{text}");
    assert!(
        text.contains(&format!(
            "`gust-artifacts/{split_key}.wgsl.reflection.json` is missing"
        )),
        "{text}"
    );
    let records = half.0.join("target-records");
    let text = expanded(KERNEL, &environment(Some(&half.0), Some(&records))).to_string();
    assert!(!text.contains("compile_error"), "{text}");
    assert_eq!(text.matches("artifacts : & []").count(), 1, "{text}");
    assert!(
        records
            .join("demo-crate")
            .join(format!("{split_key}.kernel.json"))
            .is_file()
    );
}

#[test]
fn export_mode_writes_a_split_record_for_a_grid_kernel_and_none_for_a_pixel_kernel() {
    let all = exported_all(KERNEL);
    assert_eq!(all.len(), 2, "a body and its split");
    let pixel = "mod layout {\n#[kernel(shape = pixel)]\npub fn inspect(id: SV_DispatchThreadID, values: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>) {\nout[id.x] = values[id.x];\n}\n}";
    let scratch = Scratch::new("split-pixel");
    let records = scratch.0.join("records");
    expanded(pixel, &environment(None, Some(&records)));
    let slangs = fs::read_dir(records.join("demo-crate"))
        .unwrap()
        .filter(|entry| {
            entry
                .as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".slang")
        })
        .count();
    assert_eq!(slangs, 1, "a pixel kernel writes its body alone");
}

#[test]
fn without_files_the_slice_is_empty_and_the_artifact_is_required() {
    let scratch = Scratch::new("pair-absent");
    scratch.artifact("0000000000000000.wgsl", "another kernel's file");
    let text = expanded(KERNEL, &environment(Some(&scratch.0), None)).to_string();
    assert!(text.contains("artifacts : & []"), "{text}");
    assert!(text.contains("require_artifact (false)"), "{text}");
    assert!(!text.contains("require_artifact (true)"), "{text}");
    // One cause, one error under a strict profile: the artifact alone is reported.
    assert!(!text.contains("require_manifest"), "{text}");
    assert!(!text.contains("include_bytes"), "{text}");
    assert!(!text.contains("compile_error"), "{text}");
}

#[test]
fn half_a_pair_is_a_compile_error_naming_the_missing_file_never_a_silent_empty_slice() {
    let (key, _) = exported(KERNEL);
    for remove in [format!("{key}.wgsl.reflection.json"), format!("{key}.wgsl")] {
        let missing = &remove;
        let half = Scratch::new(&format!("half {remove}"));
        half.artifact(&format!("{key}.wgsl"), "x");
        half.artifact(&format!("{key}.wgsl.reflection.json"), "{}");
        fs::remove_file(half.0.join("gust-artifacts").join(&remove)).unwrap();
        let output = expanded(KERNEL, &environment(Some(&half.0), None));
        let text = output.to_string();
        assert!(text.contains("compile_error !"), "{text}");
        assert!(text.contains("[GUST-E0711]"), "{text}");
        assert!(
            text.contains(&format!("`gust-artifacts/{missing}` is missing")),
            "{text}"
        );
        // The message names the file to delete (the one that is left) and the command that then works.
        let left = if remove.ends_with(".json") {
            format!("{key}.wgsl")
        } else {
            format!("{key}.wgsl.reflection.json")
        };
        assert!(
            text.contains(&format!(
                "Delete the file that is left, `gust-artifacts/{left}`, then run `cargo gust build`"
            )),
            "{text}"
        );
        // Not a silent `&[]`: the error is the whole story, with no profile check that could add a second one.
        assert!(!text.contains("require_artifact"), "{text}");
        assert!(!text.contains("include_bytes"), "{text}");
        // The error sits at the kernel's identifier (line 3 of KERNEL).
        let mut lines = Vec::new();
        ident_lines(output, "compile_error", &mut lines);
        assert_eq!(lines, [3], "{text}");
    }
}

/// `cargo gust build` expands the kernel crate in export mode to learn what to compile; a half pair on disk must
/// not stop that expansion, or the build tool could never repair the half it finds.
#[test]
fn in_export_mode_half_a_pair_is_absent_so_that_the_build_tool_can_repair_it() {
    let (key, _) = exported(KERNEL);
    for remove in [format!("{key}.wgsl.reflection.json"), format!("{key}.wgsl")] {
        let half = Scratch::new(&format!("half-export {remove}"));
        half.artifact(&format!("{key}.wgsl"), "x");
        half.artifact(&format!("{key}.wgsl.reflection.json"), "{}");
        fs::remove_file(half.0.join("gust-artifacts").join(&remove)).unwrap();
        let records = half.0.join("target-records");
        let text = expanded(KERNEL, &environment(Some(&half.0), Some(&records))).to_string();
        assert!(!text.contains("compile_error"), "{text}");
        assert!(text.contains("artifacts : & []"), "{text}");
        assert!(text.contains("require_artifact (false)"), "{text}");
        assert!(!text.contains("include_bytes"), "{text}");
        // The records were still written: that is what the build tool reads.
        assert!(
            records
                .join("demo-crate")
                .join(format!("{key}.kernel.json"))
                .is_file()
        );
        // The same crate without export mode keeps the hard error.
        let strict = expanded(KERNEL, &environment(Some(&half.0), None)).to_string();
        assert!(strict.contains("[GUST-E0711]"), "{strict}");
    }
}

/// The manifest check is a property of the crate, so a module with several kernels that have a pair gets it once.
#[test]
fn require_manifest_is_emitted_once_per_module_not_once_per_kernel() {
    let two = "mod two {\n#[kernel]\npub fn a(id: SV_DispatchThreadID, v: StructuredBuffer<f32>) { let i = id.x; }\n#[kernel]\npub fn b(id: SV_DispatchThreadID, mut v: RWStructuredBuffer<f32>) { v[id.x] = 1.0; }\n}";
    let scratch = Scratch::new("manifest-once");
    let records = scratch.0.join("records");
    expanded(two, &environment(None, Some(&records)));
    let directory = records.join("demo-crate");
    let mut keys = 0;
    for entry in fs::read_dir(&directory).unwrap() {
        let name = entry.unwrap().file_name().into_string().unwrap();
        if let Some(key) = name.strip_suffix(".slang") {
            scratch.artifact(&format!("{key}.wgsl"), "wgsl");
            scratch.artifact(&format!("{key}.wgsl.reflection.json"), "{}");
            keys += 1;
        }
    }
    // Two bodies and the split of each (D209); only the bodies are required.
    assert_eq!(keys, 4);
    for manifest in [false, true] {
        if manifest {
            scratch.artifact("manifest.json", "{}");
        }
        let output = expanded(two, &environment(Some(&scratch.0), None));
        let text = output.to_string();
        assert_eq!(text.matches("require_artifact (true)").count(), 2, "{text}");
        assert_eq!(
            text.matches(&format!("require_manifest ({manifest})"))
                .count(),
            1,
            "{text}"
        );
        assert_eq!(text.matches("require_manifest").count(), 1, "{text}");
        // The one check sits at the first kernel with a pair (`a`, line 3 of the module).
        let mut lines = Vec::new();
        ident_lines(output, "require_manifest", &mut lines);
        assert_eq!(lines, [3], "{text}");
    }
}

#[test]
fn a_strict_profile_needs_the_manifest_only_beside_a_present_pair() {
    let (without, _) = crate_with_pair("manifest-absent", false);
    let text = expanded(KERNEL, &environment(Some(&without.0), None)).to_string();
    assert!(text.contains("require_artifact (true)"), "{text}");
    assert!(text.contains("require_manifest (false)"), "{text}");
    assert!(!text.contains("manifest.json"), "{text}");
    let mut lines = Vec::new();
    ident_lines(
        expanded(KERNEL, &environment(Some(&without.0), None)),
        "require_manifest",
        &mut lines,
    );
    assert_eq!(lines, [3]);
}

#[test]
fn the_manifest_is_included_once_per_module_even_without_kernels() {
    let scratch = Scratch::new("manifest-modules");
    scratch.artifact("manifest.json", "{}");
    for (source, kernels) in [(KERNEL, 1), ("mod empty {}", 0)] {
        let text = expanded(source, &environment(Some(&scratch.0), None)).to_string();
        assert_eq!(
            text.matches("\"/gust-artifacts/manifest.json\"").count(),
            1,
            "{text}"
        );
        assert_eq!(text.matches("require_artifact").count(), kernels, "{text}");
    }
    let two = "mod two {\n#[kernel]\npub fn a(id: SV_DispatchThreadID, v: StructuredBuffer<f32>) { let i = id.x; }\n#[kernel]\npub fn b(id: SV_DispatchThreadID, v: StructuredBuffer<f32>) { let i = id.x; }\n}";
    let text = expanded(two, &environment(Some(&scratch.0), None)).to_string();
    assert_eq!(text.matches("manifest.json").count(), 1, "{text}");
}

#[test]
fn export_mode_writes_both_records_under_the_package_and_matches_the_golden() {
    let scratch = Scratch::new("export-golden");
    let records = scratch.0.join("records");
    let source = "mod demo {\nstruct Pair { a: uint, b: float }\n#[kernel(workgroup_size(32, 2, 1))]\npub fn run(id: SV_DispatchThreadID, pairs: StructuredBuffer<Pair>, mut out: RWStructuredBuffer<float>) {\nout[id.x] = pairs[id.x].b;\n}\n}";
    expanded(source, &environment(None, Some(&records)));
    let directory = records.join("demo-crate");
    let mut names: Vec<String> = fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    // The body's two files and its split's two (D209).
    assert_eq!(names.len(), 4, "{names:?}");
    let key = exported(source).0;
    let (split_key, _, owner) = exported_split(source);
    assert_eq!(owner, key);
    assert_eq!(
        names
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        [
            format!("{key}.kernel.json"),
            format!("{key}.slang"),
            format!("{split_key}.kernel.json"),
            format!("{split_key}.slang"),
        ]
        .into_iter()
        .collect()
    );
    let split_json =
        fs::read_to_string(directory.join(format!("{split_key}.kernel.json"))).unwrap();
    assert!(split_json.contains("\"schema\": 5"), "{split_json}");
    assert!(
        split_json.contains(&format!("\"split_of\": \"{key}\"")),
        "{split_json}"
    );
    let slang = fs::read(directory.join(format!("{key}.slang"))).unwrap();
    assert_eq!(
        key,
        artifact::content_key("gpu_demo_run", std::str::from_utf8(&slang).unwrap())
    );
    let json = fs::read_to_string(directory.join(format!("{key}.kernel.json"))).unwrap();
    // The key follows the emitter's output, which has its own goldens; the rest of the record is pinned here.
    assert_eq!(json.matches(&key).count(), 1, "{json}");
    artifact::tests::assert_golden(
        &json.replace(&key, "<key>"),
        "demo-run.kernel.json",
        &include_str!("../../../tests/golden/demo-run.kernel.json").replace("\r\n", "\n"),
    );
    // A second expansion rewrites identical bytes.
    expanded(source, &environment(None, Some(&records)));
    assert_eq!(
        fs::read(directory.join(format!("{key}.kernel.json"))).unwrap(),
        json.as_bytes()
    );
    assert_eq!(
        fs::read(directory.join(format!("{key}.slang"))).unwrap(),
        slang
    );
}

#[test]
fn export_mode_keeps_the_strict_assert_for_what_is_present() {
    let (scratch, _) = crate_with_pair("export-present", true);
    let records = scratch.0.join("target-records");
    let text = expanded(KERNEL, &environment(Some(&scratch.0), Some(&records))).to_string();
    assert!(text.contains("require_artifact (true)"), "{text}");
    assert!(
        text.contains("artifacts : & [:: gust :: Artifact"),
        "{text}"
    );
    let empty = Scratch::new("export-absent");
    let text = expanded(KERNEL, &environment(Some(&empty.0), Some(&records))).to_string();
    assert!(text.contains("require_artifact (false)"), "{text}");
    assert!(!text.contains("require_artifact (true)"), "{text}");
    assert!(text.contains("artifacts : & []"), "{text}");
}

#[test]
fn export_mode_without_a_package_name_is_a_compile_error() {
    let scratch = Scratch::new("export-no-package");
    let mut environment = environment(None, Some(&scratch.0));
    environment.package = None;
    let text = expanded(KERNEL, &environment).to_string();
    assert!(text.contains("compile_error !"), "{text}");
    assert!(text.contains("[GUST-E0713]"), "{text}");
    assert!(text.contains("CARGO_PKG_NAME"), "{text}");
    assert!(fs::read_dir(&scratch.0).unwrap().next().is_none());
}

#[test]
fn an_export_write_failure_is_a_compile_error_naming_the_path() {
    let scratch = Scratch::new("export-failure");
    let blocker = scratch.0.join("blocked");
    fs::write(&blocker, "a file where the export directory should be").unwrap();
    let output = expanded(KERNEL, &environment(None, Some(&blocker)));
    let text = output.to_string();
    assert!(text.contains("compile_error !"), "{text}");
    assert!(text.contains("[GUST-E0712]"), "{text}");
    assert!(text.contains("blocked"), "{text}");
    assert!(text.contains("could not create"), "{text}");
    let mut lines = Vec::new();
    ident_lines(output, "compile_error", &mut lines);
    assert_eq!(lines, [3], "{text}");
}

#[test]
fn no_export_without_the_variable() {
    let scratch = Scratch::new("no-export");
    let text = expanded(KERNEL, &environment(Some(&scratch.0), None)).to_string();
    assert!(!text.contains("compile_error"), "{text}");
    assert!(fs::read_dir(&scratch.0).unwrap().next().is_none());
}

/// The sibling of `descriptor_contains_slang_not_custom_ir_or_spirv`'s `fn cpu` check: the records of a kernel
/// that does get a CPU twin hold none of the twin's text, though the expansion itself does.
#[test]
fn the_records_of_a_kernel_with_a_twin_hold_no_twin_text() {
    let scratch = Scratch::new("twin-free");
    let records = scratch.0.join("records");
    let expansion = expanded(KERNEL, &environment(None, Some(&records))).to_string();
    for marker in ["gust_dual", "CpuDevice", "__gpu_typecheck_"] {
        assert!(
            expansion.contains(marker),
            "the expansion must carry the twin, or this test proves nothing: {marker}"
        );
    }
    let directory = records.join("demo-crate");
    let mut checked = 0;
    for entry in fs::read_dir(&directory).unwrap() {
        let record = fs::read_to_string(entry.unwrap().path()).unwrap();
        for marker in artifact::TWIN_TEXT {
            assert!(!record.contains(marker), "{marker} in {record}");
        }
        assert!(!record.contains("cpu"), "no `cpu` at all in {record}");
        checked += 1;
    }
    // The body's two files and its split's two.
    assert_eq!(checked, 4);
}

/// A user identifier is not twin text: a struct called `CpuDevice` and a function called `gust_dual` export like
/// any other names. The guard matches the forms the generated twin takes (see `TWIN_TEXT`), and reserving the two
/// names would reject programs that are valid today.
#[test]
fn user_names_like_the_twin_export_without_a_bug_report() {
    let source = "mod names {\nstruct CpuDevice { a: uint }\nfn gust_dual(x: uint) -> uint { x + 1u32 }\n#[kernel]\npub fn run(id: SV_DispatchThreadID, v: StructuredBuffer<CpuDevice>, mut out: RWStructuredBuffer<uint>) {\nlet gust_dual_value = gust_dual(v[id.x].a);\nout[id.x] = gust_dual_value;\n}\n}";
    let scratch = Scratch::new("twin-names");
    let records = scratch.0.join("records");
    let text = expanded(source, &environment(None, Some(&records))).to_string();
    assert!(!text.contains("compile_error"), "{text}");
    let directory = records.join("demo-crate");
    let mut slang = String::new();
    let mut json = String::new();
    for entry in fs::read_dir(&directory).unwrap() {
        let path = entry.unwrap().path();
        let text = fs::read_to_string(&path).unwrap();
        if path
            .extension()
            .is_some_and(|extension| extension == "slang")
        {
            slang = text;
        } else {
            json = text;
        }
    }
    assert!(slang.contains("gust_dual"), "{slang}");
    assert!(slang.contains("CpuDevice"), "{slang}");
    assert!(json.contains("StructuredBuffer < CpuDevice >"), "{json}");
}

/// The source table of the record is the descriptor's: with a real file and a line table, both are non-empty and
/// equal. (The fallback spans of a unit test have no file, so the file is injected through the `Environment`.)
#[test]
fn export_records_hold_the_source_file_and_the_line_table_the_descriptor_holds() {
    let scratch = Scratch::new("origin");
    let records = scratch.0.join("records");
    let mut environment = environment(Some(&scratch.0), Some(&records));
    environment.reported_file = Some(format!(
        "{}/src/kernels.rs",
        scratch.0.display().to_string().replace('\\', "/")
    ));
    let text = expanded(KERNEL, &environment).to_string();
    // The descriptor: the file relative to the crate, and a non-empty table.
    assert!(
            text.contains("source : :: gust :: SourceOrigin { file : \"src/kernels.rs\" , lines : & [:: gust :: LineSpan"),
            "{text}"
        );
    let directory = records.join("demo-crate");
    let json = fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.to_string_lossy().ends_with(".kernel.json"))
        .map(|path| fs::read_to_string(path).unwrap())
        .find(|json| !json.contains("\"split_of\""))
        .expect("the body's record");
    assert!(json.contains("\"file\": \"src/kernels.rs\""), "{json}");
    assert!(!json.contains("\"lines\": []"), "{json}");
    // The same table, entry for entry: every `[start, end, rust_line]` triple of the descriptor is in the record.
    let mut triples = 0;
    // The kernel's own descriptor: the split variant's (m11.g4a) has a table of its own, shifted by the lines the
    // hidden uniform adds, and a record of its own (D209) that this check does not read.
    let own = text
        .split("impl :: gust :: Kernel for ")
        .nth(1)
        .expect("the kernel's impl");
    for part in own.split("slang_start : ").skip(1) {
        let start = part.split(',').next().unwrap();
        let end = part
            .split("slang_end : ")
            .nth(1)
            .unwrap()
            .split(',')
            .next()
            .unwrap();
        let line = part
            .split("rust_line : ")
            .nth(1)
            .unwrap()
            .split(['}', ','])
            .next()
            .unwrap();
        let number = |text: &str| -> String {
            text.trim()
                .chars()
                .take_while(char::is_ascii_digit)
                .collect()
        };
        let triple = format!("[{}, {}, {}]", number(start), number(end), number(line));
        assert!(json.contains(&triple), "{triple} not in {json}");
        triples += 1;
    }
    assert!(triples > 0, "{text}");
    // Without a real file there is no table in either place.
    let anonymous = Scratch::new("origin-none");
    let records = anonymous.0.join("records");
    let text = expanded(KERNEL, &environment_without_file(&records)).to_string();
    assert!(text.contains("SourceOrigin :: UNKNOWN"), "{text}");
}

#[test]
fn the_record_file_is_the_one_the_descriptor_origin_names() {
    let lines = [crate::slang::LineRecord {
        slang_start: 3,
        slang_end: 4,
        rust_line: 12,
    }];
    assert_eq!(
        known_origin_file("/repo/src/lib.rs", REPO, &lines).as_deref(),
        Some("src/lib.rs")
    );
    assert_eq!(known_origin_file("/repo/src/lib.rs", REPO, &[]), None);
    assert_eq!(known_origin_file("src/my kernel.rs", REPO, &lines), None);
}

#[test]
fn every_expansion_tracks_the_profile_and_the_export_directory() {
    let kernel = "mod layout {\n#[kernel]\npub fn inspect(id: SV_DispatchThreadID, values: StructuredBuffer<f32>) {\nlet i = id.x;\n}\n}";
    for source in [kernel, "mod empty {}"] {
        let module: syn::ItemMod = crate::diagnostics::testing::parse_str(source).unwrap();
        let text = expand(module).unwrap().to_string();
        for variable in ["GUST_PROFILE", "GUST_EXPORT_DIR"] {
            let item = format!(
                "const _ : :: core :: option :: Option < & :: core :: primitive :: str > = :: core :: option_env ! (\"{variable}\") ;"
            );
            assert_eq!(text.matches(&item).count(), 1, "{variable} in {text}");
        }
    }
}
