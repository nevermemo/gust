use std::path::PathBuf;

use super::{
    ExportOptions, MATRIX_LAYOUT_FLAG, ShaderTarget, SlangcOptions, SpirvValOptions,
    check_spirv_header, clear_exports, compare_golden_names, compare_manifests, compute_entries,
    count_copy_logical, golden_file_name, hash_manifest, parse_manifest, sidecar,
};

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

#[test]
fn slangc_options_parse_in_any_order() {
    let options = SlangcOptions::parse(&args(&[
        "--target",
        "spirv",
        "k.slang",
        "--out",
        "out/k.spv",
        "--entry",
        "main",
    ]))
    .unwrap();
    assert_eq!(options.source, PathBuf::from("k.slang"));
    assert_eq!(options.entry, "main");
    assert_eq!(options.target, ShaderTarget::Spirv);
    assert_eq!(options.out, PathBuf::from("out/k.spv"));
    assert!(options.extra.is_empty());
}

#[test]
fn slangc_compiler_args_are_explicit_and_keep_extra_args_last() {
    let options = SlangcOptions::parse(&args(&[
        "k.slang", "--entry", "main", "--target", "wgsl", "--out", "k.wgsl", "--", "-O0", "-DX=1",
    ]))
    .unwrap();
    assert_eq!(
        options.compiler_args(),
        args(&[
            "k.slang",
            "-entry",
            "main",
            "-stage",
            "compute",
            "-target",
            "wgsl",
            "-o",
            "k.wgsl",
            "-matrix-layout-column-major",
            "-O0",
            "-DX=1",
        ])
    );
}

#[test]
fn slangc_options_reject_missing_unknown_and_repeated_input() {
    for partial in [
        // No source, no entry, no target, no output.
        &["--entry", "main", "--target", "wgsl", "--out", "k"][..],
        &["k.slang", "--target", "wgsl", "--out", "k"][..],
        &["k.slang", "--entry", "main", "--out", "k"][..],
        &["k.slang", "--entry", "main", "--target", "wgsl"][..],
    ] {
        assert!(
            SlangcOptions::parse(&args(partial)).is_err(),
            "{partial:?} should be rejected"
        );
    }
    assert!(SlangcOptions::parse(&args(&["--bogus"])).is_err());
    assert!(
        SlangcOptions::parse(&args(&[
            "a.slang", "b.slang", "--entry", "e", "--target", "wgsl", "--out", "o"
        ]))
        .is_err()
    );
    assert!(
        SlangcOptions::parse(&args(&[
            "a.slang", "--entry", "e", "--target", "glsl", "--out", "o"
        ]))
        .is_err()
    );
    assert!(SlangcOptions::parse(&args(&["a.slang", "--entry"])).is_err());
}

#[test]
fn sidecar_appends_to_the_full_file_name() {
    assert_eq!(
        sidecar(&PathBuf::from("out/vector-add.wgsl"), "command.json"),
        PathBuf::from("out/vector-add.wgsl.command.json")
    );
}

#[test]
fn spirv_val_options_need_one_file_and_an_environment() {
    let options = SpirvValOptions::parse(&args(&["k.spv", "--target-env", "vulkan1.2"])).unwrap();
    assert_eq!(options.file, PathBuf::from("k.spv"));
    assert_eq!(options.target_env, "vulkan1.2");
    assert!(SpirvValOptions::parse(&args(&["k.spv"])).is_err());
    assert!(SpirvValOptions::parse(&args(&["--target-env", "vulkan1.2"])).is_err());
    assert!(SpirvValOptions::parse(&args(&["a.spv", "b.spv", "--target-env", "e"])).is_err());
    assert!(SpirvValOptions::parse(&args(&["a.spv", "--target-env"])).is_err());
    assert!(SpirvValOptions::parse(&args(&["a.spv", "--bogus", "--target-env", "e"])).is_err());
}

#[test]
fn spirv_header_accepts_magic_and_rejects_short_misaligned_or_foreign_files() {
    let mut module = vec![0x03, 0x02, 0x23, 0x07];
    module.resize(20, 0);
    assert!(check_spirv_header(&module).is_ok());
    module.resize(24, 0);
    assert!(check_spirv_header(&module).is_ok());

    assert!(check_spirv_header(&[]).is_err());
    assert!(check_spirv_header(&module[..16]).is_err());
    assert!(check_spirv_header(&module[..22]).is_err());

    let mut big_endian = vec![0x07, 0x23, 0x02, 0x03];
    big_endian.resize(20, 0);
    let error = check_spirv_header(&big_endian).unwrap_err();
    assert!(error.contains("magic"), "{error}");
    let mut text = b"struct Params {}".to_vec();
    text.resize(20, b' ');
    assert!(check_spirv_header(&text).is_err());
}

// ---- the matrix layout flag (m6.x) ----

#[test]
fn the_matrix_layout_flag_is_column_major_and_in_the_arguments_of_both_targets() {
    assert_eq!(MATRIX_LAYOUT_FLAG, "-matrix-layout-column-major");
    for (target, name) in [("wgsl", "k.wgsl"), ("spirv", "k.spv")] {
        let options = SlangcOptions::parse(&args(&[
            "k.slang", "--entry", "main", "--target", target, "--out", name, "--", "-O0",
        ]))
        .unwrap();
        let compiler_args = options.compiler_args();
        let flags: Vec<&String> = compiler_args
            .iter()
            .filter(|argument| argument.contains("matrix-layout"))
            .collect();
        assert_eq!(flags, [MATRIX_LAYOUT_FLAG], "{target}");
        // After the output, before the caller's own arguments.
        let position = compiler_args
            .iter()
            .position(|argument| argument == MATRIX_LAYOUT_FLAG)
            .unwrap();
        assert_eq!(compiler_args[position - 2], "-o", "{target}");
        assert_eq!(compiler_args[position + 1], "-O0", "{target}");
    }
}

/// The text of the free function that starts with `signature`: up to the first line that is a closing brace at the
/// start of a line. Panics when the function is gone, so a rename fails the test that reads it.
fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("`{signature}` is gone from the source"));
    let rest = &source[start..];
    let end = rest.find("\n}").unwrap_or(rest.len());
    &rest[..end]
}

#[test]
fn the_flag_is_the_text_the_compile_paths_of_gust_pass() {
    // `gust::slang::slangc_command` and `gust::reflect::slangc_reflection_command` each pass the flag as one string
    // literal; the xtask command must name the same layout. Only the bodies of the two functions are read, so a flag
    // deleted from either one fails here (each file also repeats the literal in its own unit test, which would
    // otherwise satisfy a search of the whole file).
    let literal = format!("\"{MATRIX_LAYOUT_FLAG}\"");
    for (path, source, signature) in [
        (
            "crates/gust/src/slang/mod.rs",
            include_str!("../../../crates/gust/src/slang/mod.rs"),
            "fn slangc_command(",
        ),
        (
            "crates/gust/src/reflect/mod.rs",
            include_str!("../../../crates/gust/src/reflect/mod.rs"),
            "fn slangc_reflection_command(",
        ),
    ] {
        let body = function_body(source, signature);
        assert!(
            body.contains(&literal),
            "{path}: `{signature}` no longer passes {MATRIX_LAYOUT_FLAG}"
        );
    }
}

#[test]
fn function_body_stops_at_the_closing_brace_of_the_function() {
    let source = "fn a() {\n    let x = \"flag\";\n}\n\nfn b() {\n    }\n}\n";
    assert_eq!(
        function_body(source, "fn a("),
        "fn a() {\n    let x = \"flag\";"
    );
    assert!(!function_body(source, "fn a(").contains("fn b"));
    assert!(function_body(source, "fn b(").contains("    }"));
}

// ---- OpCopyLogical (m6.x) ----

#[test]
fn copy_logical_instructions_are_counted_by_shape_not_by_word() {
    let disassembly = "\
; SPIR-V
               OpCapability Shader
               OpName %p \"OpCopyLogical\"
               OpName %q \"%1 = OpCopyLogical\"
               OpString \"%2 = OpCopyLogical %T %x\"
         %10 = OpCopyLogical %T %a
%copy_two = OpCopyLogical %T %b
         %11 = OpCopyLogicalX %T %c
         %12 = OpLoad %T %d
";
    // Two real instructions; the names, the string, the longer opcode and a load do not count.
    assert_eq!(count_copy_logical(disassembly), 2);
    assert_eq!(count_copy_logical(""), 0);
    assert_eq!(
        count_copy_logical("  %1 = OpCopyLogical %T %a\r\n  %2 = OpCopyLogical %T %b\r\n"),
        2
    );
}

// ---- the goldens as SPIR-V (m6.x) ----

#[test]
fn compute_entries_are_the_names_after_the_compute_attribute() {
    let slang = "\
struct S { float a; };
[shader(\"compute\")]
[numthreads(64, 1, 1)]
void gpu_vectors_run(uint3 id : SV_DispatchThreadID)
{
}
void helper(uint x) {}
  [shader(\"compute\")]
void gpu_vectors_other( uint3 id )
{
}
";
    assert_eq!(
        compute_entries(slang),
        ["gpu_vectors_run", "gpu_vectors_other"]
    );
    assert!(compute_entries("void gpu_x_run() {}").is_empty());
    assert!(compute_entries("[shader(\"compute\")]\nstruct NotAKernel {};").is_empty());
}

#[test]
fn a_golden_entry_names_its_spirv_file_like_the_examples_do() {
    assert_eq!(
        golden_file_name("vectors", "gpu_vectors_run").unwrap(),
        "vectors__run.spv"
    );
    assert_eq!(
        golden_file_name("uniforms", "gpu_uniforms_scale_all").unwrap(),
        "uniforms__scale_all.spv"
    );
    for entry in [
        "run",
        "gpu_run",
        "gpu_vectors_",
        "gpu_products_run",
        "vectors_run",
    ] {
        assert!(
            golden_file_name("vectors", entry).is_err(),
            "{entry} should be refused"
        );
    }
}

#[test]
fn the_golden_names_are_compared_and_the_differences_named() {
    let found =
        |values: &[&str]| -> Vec<String> { values.iter().map(|v| (*v).to_owned()).collect() };
    assert_eq!(
        compare_golden_names(
            &["a__run.spv", "b__run.spv"],
            &found(&["b__run.spv", "a__run.spv"])
        ),
        Ok(())
    );
    // A golden dropped from the expected names: its file is unexpected.
    let error =
        compare_golden_names(&["a__run.spv"], &found(&["a__run.spv", "b__run.spv"])).unwrap_err();
    assert!(error.contains("Unexpected: b__run.spv."), "{error}");
    assert!(!error.contains("Missing"), "{error}");
    // A golden that no longer produces its file: it is missing.
    let error =
        compare_golden_names(&["a__run.spv", "b__run.spv"], &found(&["a__run.spv"])).unwrap_err();
    assert!(error.contains("Missing: b__run.spv."), "{error}");
    assert!(error.contains("deliberately"), "{error}");
}

fn export_directory(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("gust-mxp0-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn export_options_take_both_flags_and_refuse_the_rest() {
    assert_eq!(ExportOptions::parse(&[]).unwrap(), ExportOptions::default());
    assert!(!ExportOptions::default().hashes());
    let options =
        ExportOptions::parse(&args(&["--compare", "b.txt", "--hash-manifest", "a.txt"])).unwrap();
    assert_eq!(options.hash_manifest, Some(PathBuf::from("a.txt")));
    assert_eq!(options.compare, Some(PathBuf::from("b.txt")));
    assert!(options.hashes());
    assert!(
        ExportOptions::parse(&args(&["--compare"]))
            .unwrap_err()
            .contains("needs a value")
    );
    let same = ExportOptions::parse(&args(&["--compare", "x.txt", "--hash-manifest", "x.txt"]));
    assert!(same.unwrap_err().contains("same file"));
    assert!(
        ExportOptions::parse(&args(&["--bogus"]))
            .unwrap_err()
            .contains("--bogus")
    );
}

#[test]
fn the_hash_manifest_lists_slang_rs_wgsl_and_spv_files_sorted_and_skips_the_rest() {
    let dir = export_directory("manifest");
    std::fs::write(dir.join("b__run.spv"), b"two").unwrap();
    std::fs::write(dir.join("a__run.wgsl"), b"one").unwrap();
    std::fs::write(dir.join("a__run.slang"), b"three").unwrap();
    std::fs::write(dir.join("a__run.rs"), b"four").unwrap();
    std::fs::write(dir.join("a__run.json"), b"not hashed").unwrap();
    let manifest = hash_manifest(&dir).unwrap();
    let lines = manifest.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 4, "{manifest}");
    assert!(lines[0].ends_with("  a__run.rs"), "{manifest}");
    assert!(lines[1].ends_with("  a__run.slang"), "{manifest}");
    assert_eq!(
        lines[2],
        "7692c3ad3540bb803c020b3aee66cd8887123234ea0c6e7143c0add73ff431ed  a__run.wgsl"
    );
    assert!(lines[3].ends_with("  b__run.spv"), "{manifest}");
    assert_eq!(parse_manifest(&manifest).unwrap().len(), 4);
    clear_exports(&dir).unwrap();
    assert!(dir.join("a__run.json").exists());
    assert!(!dir.join("a__run.slang").exists());
    assert!(!dir.join("b__run.spv").exists());
    assert!(hash_manifest(&dir).unwrap_err().contains("no export file"));
    std::fs::remove_dir_all(&dir).unwrap();
    clear_exports(&dir).unwrap();
}

#[test]
fn the_comparison_names_missing_changed_and_unexpected_exports() {
    let dir = export_directory("compare");
    for (name, bytes) in [("a.spv", "1"), ("b.spv", "2"), ("c.wgsl", "3")] {
        std::fs::write(dir.join(name), bytes).unwrap();
    }
    let baseline = hash_manifest(&dir).unwrap();
    assert_eq!(compare_manifests(&baseline, &baseline), Ok(()));
    std::fs::remove_file(dir.join("a.spv")).unwrap();
    std::fs::write(dir.join("b.spv"), "changed").unwrap();
    std::fs::write(dir.join("d.spv"), "4").unwrap();
    let error = compare_manifests(&baseline, &hash_manifest(&dir).unwrap()).unwrap_err();
    assert!(error.contains("(3 recorded, 3 found)"), "{error}");
    assert!(error.contains("Missing: a.spv."), "{error}");
    assert!(error.contains("Changed: b.spv."), "{error}");
    assert!(error.contains("Unexpected: d.spv."), "{error}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_malformed_or_empty_manifest_is_refused() {
    assert!(parse_manifest("").unwrap_err().contains("no export"));
    assert!(
        parse_manifest("abc  a.spv\n")
            .unwrap_err()
            .contains("line 1")
    );
    let hash = "0".repeat(64);
    let twice = format!("{hash}  a.spv\n{hash}  a.spv\n");
    assert!(
        parse_manifest(&twice)
            .unwrap_err()
            .contains("repeats `a.spv`")
    );
    assert!(
        compare_manifests("", &format!("{hash}  a.spv\n"))
            .unwrap_err()
            .starts_with("baseline:")
    );
}
