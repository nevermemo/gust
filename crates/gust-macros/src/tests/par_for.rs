//! `par_for!` (slice m7.p, plan section 3.1): the Slang of the fixture's calls, the identity of an expansion (R1.1 S5:
//! the module name is the hash of the lowered Slang), the export records (R1.1 M5: schema 2 and one sidecar per call
//! site), the attribution of an error in the body (D33) and the rejections (GUST-E0719, GUST-E0720).

use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use proc_macro2::{TokenStream, TokenTree};

use crate::{
    diagnostics::{Code, testing::assert_rendered_code},
    expand::{Environment, Profile, TWIN_TEXT},
    par_for::{expand_input, par_for_in},
    regression_tests::assert_golden,
};

const FIXTURE: &str = include_str!("../../../../tests/fixtures/par_for.rs");
const GOLDEN: &str = include_str!("../../../../tests/fixtures/par_for.slang");

/// A scratch directory under the system temp directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "gust par_for {name} {} {}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    /// The names (not the paths) of the exported files that end with `suffix`, sorted.
    fn files(&self, suffix: &str) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(self.0.join("demo"))
            .map(|entries| {
                entries
                    .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                    .filter(|name| name.ends_with(suffix))
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    }

    fn read(&self, name: &str) -> String {
        fs::read_to_string(self.0.join("demo").join(name)).unwrap()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// An environment that exports into `scratch` and says the call is in `file`.
fn environment(scratch: &Scratch, file: &str) -> Environment {
    Environment {
        manifest_dir: None,
        export_dir: Some(scratch.0.clone()),
        package: Some("demo".to_owned()),
        reported_file: Some(file.to_owned()),
    }
}

/// The text of a `par_for!` expansion for `input`, with the call on `call_line` of `file`.
fn expand(input: &str, scratch: &Scratch, file: &str, call_line: u32, profile: Profile) -> String {
    let tokens: TokenStream = input.parse().unwrap();
    par_for_in(tokens, &environment(scratch, file), profile, call_line).to_string()
}

/// Whether an expansion is an error (`{ ::core::compile_error!{..} Ok(()) }`) rather than a kernel; a good expansion
/// also
/// holds a `compile_error!` behind `cfg(target_arch = "wasm32")`.
fn failed(expansion: &str) -> bool {
    expansion.starts_with("{ :: core :: compile_error")
}

/// The generated module's name, `__gust_pf_<hash16>`.
fn module_name(expansion: &str) -> String {
    let start = expansion.find("mod __gust_pf_").expect(expansion) + "mod ".len();
    expansion[start..]
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
}

/// The input of every `par_for!` call in `source`, in order.
fn calls(source: &str) -> Vec<String> {
    fn walk(stream: TokenStream, found: &mut Vec<String>) {
        let tokens: Vec<TokenTree> = stream.into_iter().collect();
        let mut index = 0;
        while index < tokens.len() {
            match &tokens[index] {
                TokenTree::Ident(ident)
                    if ident == "par_for"
                        && matches!(tokens.get(index + 1), Some(TokenTree::Punct(p)) if p.as_char() == '!') =>
                {
                    if let Some(TokenTree::Group(group)) = tokens.get(index + 2) {
                        found.push(group.stream().to_string());
                        index += 3;
                        continue;
                    }
                }
                TokenTree::Group(group) => walk(group.stream(), found),
                _ => {}
            }
            index += 1;
        }
    }
    let mut found = Vec::new();
    walk(source.parse().unwrap(), &mut found);
    found
}

/// A one-dimensional loop whose body is on lines 2 and 3 of the input.
const BODY: &str =
    "i in 0..n, [a: &[f32], out: &mut [f32]] {\n    let v = a[i];\n    out[i] = v * 2.0;\n}";

/// `BODY` moved down by `lines`.
fn moved(lines: usize) -> String {
    format!("{}{BODY}", "\n".repeat(lines))
}

// ---- the Slang ----

#[test]
fn the_calls_of_the_fixture_match_the_golden() {
    let inputs = calls(FIXTURE);
    assert_eq!(inputs.len(), 8, "{inputs:#?}");
    let mut actual = String::new();
    for (index, input) in inputs.iter().enumerate() {
        let scratch = Scratch::new("golden");
        let expansion = expand(input, &scratch, "src/demo.rs", 1, Profile::Other);
        assert!(!failed(&expansion), "{expansion}");
        let slang = scratch.files(".slang");
        assert_eq!(slang.len(), 1, "{slang:?}");
        actual.push_str(&format!("// ==== call {} ====\n", index + 1));
        actual.push_str(&scratch.read(&slang[0]).replace("\r\n", "\n"));
    }
    assert_golden(&actual, "par_for.slang", &GOLDEN.replace("\r\n", "\n"));
}

#[test]
fn the_golden_shows_the_default_workgroups_the_guard_and_the_extent_record() {
    let golden = GOLDEN.replace("\r\n", "\n");
    // 256 x 1 x 1 for one dimension (and 64 for the override), 16 x 16 x 1 for two, 8 x 8 x 4 for three.
    for numthreads in [
        "[numthreads(256, 1, 1)]",
        "[numthreads(64, 1, 1)]",
        "[numthreads(16, 16, 1)]",
        "[numthreads(8, 8, 4)]",
    ] {
        assert!(
            golden.contains(numthreads),
            "{numthreads} not in:\n{golden}"
        );
    }
    // The record is a uniform padded with explicit fields; a `bool` is a `u32` word in it.
    assert!(golden.contains("struct __GustExtent"), "{golden}");
    assert!(golden.contains("__gust_pad0"), "{golden}");
    assert!(
        golden.contains("var invert = (__gust_ext.invert != uint(0));"),
        "{golden}"
    );
}

// ---- identity (R1.1 S5) ----

#[test]
fn one_body_in_two_spellings_is_one_module_name() {
    // `float` is the dialect's spelling of `f32`, and layout does not matter: the name is the hash of the lowered
    // Slang, not of the tokens, so these calls are one kernel. (Mutation: naming the module from the token text.)
    let plain = "i in 0..n, [a: &[f32], out: &mut [f32]] { let v = a[i]; out[i] = v * 2.0; }";
    let spelled = "i in 0..n , [ a : &[float] , out : &mut [float] ] {\n let v = a [ i ] ;\n\n out [ i ] = v * 2.0 ;\n }";
    let (first, second) = (Scratch::new("spelling"), Scratch::new("spelling"));
    let one = expand(plain, &first, "src/lib.rs", 1, Profile::Other);
    let two = expand(spelled, &second, "src/lib.rs", 1, Profile::Other);
    assert!(!failed(&one) && !failed(&two), "{one}\n{two}");
    assert_eq!(module_name(&one), module_name(&two));
    assert_eq!(first.files(".slang"), second.files(".slang"));
    assert_eq!(first.files(".slang").len(), 1);
    // Tokens that differ in the body itself and lower alike: a local's type spelled `f32` or `float`, a literal's
    // suffix spelled `2.0f32` or `2.0_f32`.
    for (left, right) in [
        (
            "i in 0..n, [a: &[f32], out: &mut [f32]] { let v: f32 = a[i]; out[i] = v * 2.0; }",
            "i in 0..n, [a: &[f32], out: &mut [f32]] { let v: float = a[i]; out[i] = v * 2.0; }",
        ),
        (
            "i in 0..n, [a: &[f32], out: &mut [f32]] { let v = a[i]; out[i] = v * 2.0f32; }",
            "i in 0..n, [a: &[f32], out: &mut [f32]] { let v = a[i]; out[i] = v * 2.0_f32; }",
        ),
    ] {
        let (first, second) = (Scratch::new("body spelling"), Scratch::new("body spelling"));
        let one = expand(left, &first, "src/lib.rs", 1, Profile::Other);
        let two = expand(right, &second, "src/lib.rs", 1, Profile::Other);
        assert!(!failed(&one) && !failed(&two), "{one}\n{two}");
        assert_eq!(module_name(&one), module_name(&two), "{left} / {right}");
        assert_eq!(first.files(".slang"), second.files(".slang"));
    }
}

#[test]
fn the_same_text_at_two_lines_and_in_two_files_is_one_key_and_one_entry() {
    let (first, second) = (Scratch::new("moved"), Scratch::new("moved"));
    let one = expand(BODY, &first, "src/one.rs", 10, Profile::Other);
    let two = expand(
        &moved(30),
        &second,
        "tests/other/two.rs",
        40,
        Profile::Other,
    );
    let name = module_name(&one);
    assert_eq!(name, module_name(&two));
    assert!(
        name.starts_with("__gust_pf_") && name.len() == "__gust_pf_".len() + 16,
        "{name}"
    );
    // One key: the same artifact files, byte for byte.
    assert_eq!(first.files(".slang"), second.files(".slang"));
    assert_eq!(first.files(".kernel.json"), second.files(".kernel.json"));
    let key = first.files(".slang")[0]
        .trim_end_matches(".slang")
        .to_owned();
    assert_eq!(
        first.read(&format!("{key}.slang")),
        second.read(&format!("{key}.slang"))
    );
    assert_eq!(
        first.read(&format!("{key}.kernel.json")),
        second.read(&format!("{key}.kernel.json"))
    );
    // One entry name, which carries the hash.
    let entry = format!("gpu_{name}_run");
    assert!(first.read(&format!("{key}.slang")).contains(&entry));
    assert!(one.contains(&entry) || one.contains(&name), "{one}");
    // The call site is the only difference: two sidecars with different names and different line tables.
    assert_ne!(first.files(".site.json"), second.files(".site.json"));
    assert_ne!(
        first.read(&first.files(".site.json")[0]),
        second.read(&second.files(".site.json")[0])
    );
}

#[test]
fn bodies_that_lower_differently_get_different_module_names() {
    let scratch = Scratch::new("different");
    let other = BODY.replace("2.0", "3.0");
    let third = "(x, y) in (0..w, 0..h), [out: &mut [u32]] { out[y * 8u32 + x] = 1u32; }";
    let names: Vec<String> = [BODY, other.as_str(), third]
        .iter()
        .map(|input| module_name(&expand(input, &scratch, "src/lib.rs", 1, Profile::Other)))
        .collect();
    assert_ne!(names[0], names[1]);
    assert_ne!(names[0], names[2]);
    assert_ne!(names[1], names[2]);
}

// ---- export (R1.1 M5) ----

#[test]
fn two_equal_bodies_export_one_artifact_and_two_sidecars() {
    // (Mutation: write `source` into the record, or drop a sidecar.)
    let scratch = Scratch::new("two sites");
    expand(BODY, &scratch, "src/lib.rs", 10, Profile::Other);
    expand(&moved(20), &scratch, "src/lib.rs", 30, Profile::Other);
    assert_eq!(scratch.files(".slang").len(), 1, "{:?}", scratch.files(""));
    assert_eq!(scratch.files(".kernel.json").len(), 1);
    let sites = scratch.files(".site.json");
    assert_eq!(sites.len(), 2, "{sites:?}");
    let key = scratch.files(".slang")[0]
        .trim_end_matches(".slang")
        .to_owned();
    for site in &sites {
        assert!(site.starts_with(&format!("{key}.")), "{site}");
        let hash = site
            .trim_start_matches(&format!("{key}."))
            .trim_end_matches(".site.json");
        assert!(
            hash.len() == 16 && hash.chars().all(|c| c.is_ascii_hexdigit()),
            "{site}"
        );
    }
    // The same call expanded again (a library and a test target both expand it) writes the same bytes again.
    expand(BODY, &scratch, "src/lib.rs", 10, Profile::Other);
    assert_eq!(scratch.files(".site.json"), sites);
}

#[test]
fn a_schema_2_record_has_no_source_and_the_sidecar_has_the_call_site() {
    let scratch = Scratch::new("schema");
    let expansion = expand(BODY, &scratch, "src/lib.rs", 10, Profile::Other);
    assert!(!failed(&expansion), "{expansion}");
    let record = scratch.read(&scratch.files(".kernel.json")[0]);
    assert!(record.contains("\"schema\": 2"), "{record}");
    assert!(!record.contains("\"source\""), "{record}");
    assert!(record.contains("\"workgroup_size\""), "{record}");
    assert!(record.contains("\"kind\": \"uniform\""), "{record}");
    let sidecar = scratch.read(&scratch.files(".site.json")[0]);
    assert!(sidecar.contains("\"schema\": 2"), "{sidecar}");
    assert!(sidecar.contains("\"file\": \"src/lib.rs\""), "{sidecar}");
    assert!(sidecar.contains("\"key\""), "{sidecar}");
    assert!(sidecar.contains("\"lines\""), "{sidecar}");
}

/// The Rust lines of a sidecar's line table, sorted and without repeats.
fn sidecar_rust_lines(sidecar: &str) -> Vec<u32> {
    let lines = sidecar.split("\"lines\"").nth(1).expect(sidecar);
    let lines = lines.split("\"schema\"").next().unwrap();
    let numbers: Vec<u32> = lines
        .split(|c: char| !c.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .map(|part| part.parse().unwrap())
        .collect();
    assert_eq!(numbers.len() % 3, 0, "{sidecar}");
    let mut rust: Vec<u32> = numbers.chunks(3).map(|triple| triple[2]).collect();
    rust.sort_unstable();
    rust.dedup();
    rust
}

#[test]
fn each_sidecars_line_table_points_at_its_own_call_site() {
    // D33: the body's statements are on lines 2 and 3 of `BODY`; moved down by 20 lines they are on 22 and 23, and each
    // site's sidecar says so for its own copy, while the shared record says nothing of either.
    let scratch = Scratch::new("lines");
    expand(BODY, &scratch, "src/lib.rs", 1, Profile::Other);
    expand(&moved(20), &scratch, "src/other.rs", 21, Profile::Other);
    let sites = scratch.files(".site.json");
    assert_eq!(sites.len(), 2);
    let mut seen = Vec::new();
    for site in &sites {
        let text = scratch.read(site);
        let rust = sidecar_rust_lines(&text);
        if text.contains("\"src/lib.rs\"") {
            assert!(rust.contains(&2) && rust.contains(&3), "{rust:?}");
            assert!(rust.iter().all(|line| *line <= 3), "{rust:?}");
            seen.push("src/lib.rs");
        } else {
            assert!(text.contains("\"src/other.rs\""), "{text}");
            assert!(rust.contains(&22) && rust.contains(&23), "{rust:?}");
            // (The generated guard and prelude have the call's own span, which a unit test does not have: line 1.)
            assert!(
                rust.iter().all(|line| *line == 1 || *line >= 22),
                "{rust:?}"
            );
            seen.push("src/other.rs");
        }
    }
    seen.sort_unstable();
    assert_eq!(seen, ["src/lib.rs", "src/other.rs"]);
}

#[test]
fn the_exported_files_hold_no_twin_text() {
    // The m7.a1 lesson: test the text that is written, not only the Slang. Every file of every call in the fixture.
    for input in calls(FIXTURE) {
        let scratch = Scratch::new("twin");
        let expansion = expand(&input, &scratch, "src/demo.rs", 1, Profile::Other);
        assert!(!failed(&expansion), "{expansion}");
        let names = scratch.files("");
        assert_eq!(names.len(), 3, "{names:?}");
        for name in names {
            let text = scratch.read(&name);
            for marker in TWIN_TEXT {
                assert!(!text.contains(marker), "{marker} in {name}:\n{text}");
            }
        }
    }
}

// ---- profiles ----

#[test]
fn the_cpu_profile_names_no_gpu_api_and_the_other_profile_names_both_devices() {
    let scratch = Scratch::new("profiles");
    let cpu = expand(BODY, &scratch, "src/lib.rs", 1, Profile::Cpu);
    let other = expand(BODY, &scratch, "src/lib.rs", 1, Profile::Other);
    assert!(!failed(&cpu) && !failed(&other), "{cpu}\n{other}");
    assert!(
        cpu.contains("CpuDevice") && !cpu.contains("gust_wgpu"),
        "{cpu}"
    );
    assert!(
        other.contains("CpuDevice") && other.contains("gust_wgpu"),
        "{other}"
    );
    // The twin of the other profile is gated, so a build without `gust_dual` does not carry it; under `cpu` it is the
    // run path and is not.
    assert!(other.contains("cfg (any (test , gust_dual))"), "{other}");
    assert!(!cpu.contains("cfg (any (test , gust_dual))"), "{cpu}");
    // The same Slang either way: one module name.
    assert_eq!(module_name(&cpu), module_name(&other));
}

#[test]
fn the_descriptor_form_expands_to_the_kernels_descriptor() {
    let scratch = Scratch::new("descriptor");
    let tokens: TokenStream = format!("descriptor; {BODY}").parse().unwrap();
    let expansion = par_for_in(
        tokens,
        &environment(&scratch, "src/lib.rs"),
        Profile::Other,
        1,
    )
    .to_string();
    assert!(!failed(&expansion), "{expansion}");
    assert!(expansion.contains("Kernel > :: DESCRIPTOR"), "{expansion}");
    assert!(!expansion.contains("launch_default ("), "{expansion}");
}

// ---- rejections ----

/// The expansion of `input` as an error text, which must carry `code`.
fn rejected(input: &str, profile: Profile, code: Code) {
    let scratch = Scratch::new("rejected");
    let expansion = expand(input, &scratch, "src/lib.rs", 1, profile);
    assert!(expansion.contains("compile_error"), "{input}\n{expansion}");
    assert_rendered_code(&expansion, code);
    // Nothing is written for a call that is refused.
    assert!(scratch.files("").is_empty(), "{:?}", scratch.files(""));
}

#[test]
fn a_call_that_is_not_one_of_the_forms_is_e0719() {
    for input in [
        // No body, no binding list.
        "i in 0..n, []",
        "i in 0..n",
        // The pattern.
        "(x) in 0..n, [] { }",
        "(a, b, c, d) in (0..1, 0..1, 0..1, 0..1), [] { }",
        "(x, x) in (0..w, 0..h), [] { }",
        "Some(i) in 0..n, [] { }",
        "mut i in 0..n, [] { }",
        // The ranges.
        "i in 1..n, [] { }",
        "i in 0..=n, [] { }",
        "i in ..n, [] { }",
        "i in 0.., [] { }",
        "i in n, [] { }",
        "(x, y) in 0..n, [] { }",
        "(x, y) in (0..w), [] { }",
        "(x, y) in (0..w, 0..h, 0..d), [] { }",
        "i in (0..w, 0..h), [] { }",
        // The rest of the input.
        "i in 0..n, [] { } { }",
        "i in 0..n, [] { } ;",
        // The options.
        "on(a); on(b); i in 0..n, [] { }",
    ] {
        rejected(input, Profile::Other, Code::ParForForm);
    }
}

#[test]
fn the_workgroup_option_uses_the_workgroup_codes() {
    rejected(
        "workgroup_size(8, 8); i in 0..n, [] { }",
        Profile::Other,
        Code::WorkgroupDimensionCount,
    );
    rejected(
        "workgroup_size(8, 0, 1); i in 0..n, [] { }",
        Profile::Other,
        Code::WorkgroupDimensionZero,
    );
    rejected(
        "workgroup_size(8, 1, 1); workgroup_size(8, 1, 1); i in 0..n, [] { }",
        Profile::Other,
        Code::DuplicateWorkgroupSize,
    );
}

#[test]
fn a_binding_that_cannot_be_one_is_e0720() {
    for input in [
        // Types.
        "i in 0..n, [a: Vec<f32>] { }",
        "i in 0..n, [a: &Vec<f32>] { }",
        "i in 0..n, [a: &[bool]] { }",
        "i in 0..n, [a: &[u64]] { }",
        "i in 0..n, [a: &mut f32] { }",
        "i in 0..n, [a: u64] { }",
        "i in 0..n, [a: &&[f32]] { }",
        "i in 0..n, [a: &GpuVec<f32, f32>] { }",
        "i in 0..n, [a: &Other<f32>] { }",
        "i in 0..n, [a: [f32; 4]] { }",
        // Names.
        "i in 0..n, [a: &[f32], a: &[f32]] { }",
        "i in 0..n, [i: u32] { }",
        "(x, y) in (0..w, 0..h), [y: u32] { }",
    ] {
        rejected(input, Profile::Other, Code::ParForBinding);
    }
    // No GPU API under the cpu profile.
    rejected(
        "i in 0..n, [a: &GpuVec<f32>] { }",
        Profile::Cpu,
        Code::ParForBinding,
    );
    rejected(
        "i in 0..n, [a: &mut GpuVec<f32>] { }",
        Profile::Cpu,
        Code::ParForBinding,
    );
    // A 3-wide vector has no buffer layout.
    rejected(
        "i in 0..n, [a: &[Vec3]] { }",
        Profile::Other,
        Code::ThreeWideInBuffer,
    );
}

#[test]
fn a_gpu_vec_binding_is_accepted_off_the_cpu_profile() {
    let scratch = Scratch::new("gpu vec");
    let input =
        "i in 0..n, [a: &GpuVec<f32>, out: &mut ::gust_wgpu::GpuVec<f32>] { out[i] = a[i]; }";
    let expansion = expand(input, &scratch, "src/lib.rs", 1, Profile::Other);
    assert!(!failed(&expansion), "{expansion}");
    assert!(expansion.contains("GpuVec"), "{expansion}");
}

/// The error of `input` and the line it points at.
fn error_line(input: &str) -> (String, usize) {
    let scratch = Scratch::new("error line");
    let tokens: TokenStream = input.parse().unwrap();
    let error = expand_input(
        tokens,
        &environment(&scratch, "src/lib.rs"),
        Profile::Other,
        1,
        &mut false,
    )
    .expect_err(input);
    (error.to_string(), error.span().start().line)
}

#[test]
fn a_dialect_error_in_the_body_is_reported_at_the_body() {
    // D33: the tokens of the body are the user's, so the error carries its code and the line of the offending
    // statement; the generated parts (the guard, the extent record) are never what an error points at.
    let input = "i in 0..n, [a: &[f32], out: &mut [f32]] {\n    let v = a[i];\n    while v > 0.0 {\n        break;\n    }\n}";
    let (message, line) = error_line(input);
    assert!(
        message.contains(Code::WhileNotSupported.text()),
        "{message}"
    );
    assert_eq!(line, 3, "{message}");
    let (message, line) =
        error_line("i in 0..n, [out: &mut [f32]] {\n\n\n    let __gust_x = 1.0;\n}");
    assert!(message.contains(Code::ReservedPrefix.text()), "{message}");
    assert_eq!(line, 4, "{message}");
    // The same through the whole expansion: a coded `compile_error!`.
    rejected(input, Profile::Other, Code::WhileNotSupported);
}

#[test]
fn the_names_in_front_of_the_body_cannot_be_reserved_ones() {
    let (message, _) = error_line("i in 0..n, [__gust_a: &[f32]] { }");
    assert!(message.contains(Code::ReservedPrefix.text()), "{message}");
}

#[test]
fn a_directory_that_cannot_be_written_is_a_coded_export_error() {
    // The export failure is the expansion's own (GUST-E0712 and the others), not a panic of the macro.
    let scratch = Scratch::new("export failure");
    let blocker = scratch.0.join("blocked");
    fs::write(&blocker, "a file where the package directory should be").unwrap();
    let environment = Environment {
        manifest_dir: None,
        export_dir: Some(blocker),
        package: Some("demo".to_owned()),
        reported_file: Some("src/lib.rs".to_owned()),
    };
    let tokens: TokenStream = BODY.parse().unwrap();
    let expansion = par_for_in(tokens, &environment, Profile::Other, 1).to_string();
    assert_rendered_code(&expansion, Code::ExportWriteFailed);
}
