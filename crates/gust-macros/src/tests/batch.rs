//! `gpu_batch!` (slice m14.s2a2): how the macro reads the block and rewrites its stages, how a stage of a batch is
//! lowered, and the two diagnostics of the family that this slice owns (E1005, E1006).

use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use proc_macro2::TokenStream;

use crate::{
    batch::gpu_batch,
    diagnostics::{Code, testing::assert_rendered_code},
    expand::{Environment, Profile},
    task::gpu_task_in,
};

/// A scratch directory under the system temp directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "gust batch {} {}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    /// The Slang of every kernel exported, joined.
    fn slang(&self) -> String {
        fs::read_dir(self.0.join("demo"))
            .map(|entries| {
                entries
                    .map(|entry| entry.unwrap().path())
                    .filter(|path| {
                        path.extension()
                            .is_some_and(|extension| extension == "slang")
                    })
                    .map(|path| fs::read_to_string(path).unwrap().replace("\r\n", "\n"))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The text of a `gpu_batch!` expansion.
fn batch(input: &str) -> String {
    let tokens: TokenStream = input.parse().unwrap();
    gpu_batch(tokens).to_string()
}

/// The text of the expansion of a stage of a batch (`gpu_task!` with the hidden clause the batch macro writes).
fn stage(input: &str, scratch: &Scratch) -> String {
    let tokens: TokenStream = input.parse().unwrap();
    let environment = Environment {
        manifest_dir: None,
        export_dir: Some(scratch.0.clone()),
        package: Some("demo".to_owned()),
        reported_file: Some("src/demo.rs".to_owned()),
    };
    gpu_task_in(tokens, &environment, Profile::Other, 1).to_string()
}

fn failed(expansion: &str) -> bool {
    expansion.contains("compile_error")
}

fn batch_rejected(input: &str, code: Code) {
    let text = batch(input);
    assert!(failed(&text), "{input}\n{text}");
    assert_rendered_code(&text, code);
    assert_eq!(text.matches("compile_error").count(), 1, "{text}");
}

fn stage_rejected(input: &str, code: Code) {
    let scratch = Scratch::new();
    let text = stage(input, &scratch);
    assert!(failed(&text), "{input}\n{text}");
    assert_rendered_code(&text, code);
    assert_eq!(text.matches("compile_error").count(), 1, "{text}");
}

const BLOCK: &str = r#""exposure", on(&device); in_flight(1); |b| {
    let size = b.param::<(u32, u32)>((16, 8));
    let gain = b.param::<f32>(1.0);
    let pixels = b.input::<f32>(&size);
    let lum = gust::gpu_task!(&size, |x, y| -> f32 { pixels[(x, y)] * gain.get() });
    let flat = gust::gpu_task!(n, |i| -> f32 { lum[i] });
    gust::gpu_task!(&mut flat, |i| { flat[i] = flat[i] * 0.5; });
    (pixels, lum, flat)
}"#;

// ---- the block ----

#[test]
fn a_block_is_a_build_of_the_device_with_its_tail_exposed() {
    let text = batch(BLOCK);
    assert!(!failed(&text), "{text}");
    assert!(text.starts_with(":: gust :: batch :: build"), "{text}");
    // The handles of the tail are exposed to the batch.
    for name in ["pixels", "lum", "flat"] {
        assert!(
            text.contains(&format!("b . expose (& {name})")),
            "{name}: {text}"
        );
    }
    // An input is `b.input::<T, _>`, and each `let` names what it records.
    assert!(text.contains("input :: < f32 , _ >"), "{text}");
    for name in ["size", "gain", "pixels"] {
        assert!(
            text.contains(&format!("b . next_name (\"{name}\")")),
            "{name}: {text}"
        );
    }
    // The result is a value of the block's own type.
    assert!(text.contains("& (& device)"), "{text}");
}

#[test]
fn a_stage_is_given_the_hidden_clause_and_the_entries_the_block_declares() {
    let text = batch(BLOCK);
    // Each stage names its `let` and carries the handle it reads, with the type the block gave it.
    assert!(text.contains("batch (b , \"lum\")"), "{text}");
    assert!(text.contains("pixels : Stage < f32 >"), "{text}");
    assert!(text.contains("gain : Param < f32 >"), "{text}");
    assert!(text.contains("lum : Stage < f32 >"), "{text}");
    // The in-place target the block declares has an entry made for it.
    assert!(text.contains("flat : & mut Stage < f32 >"), "{text}");
    assert!(text.contains("batch (b , \"\")"), "{text}");
}

#[test]
fn a_written_scalar_and_a_written_entry_of_a_declared_target_stay_as_they_are() {
    let text = batch(
        r#""t", on(&device); |b| {
            let d = b.input::<f32>(4u32);
            gust::gpu_task!(&mut d, [d: &mut [f32], k: f32], |i| { d[i] = d[i] * k; });
            d
        }"#,
    );
    assert!(!failed(&text), "{text}");
    assert!(text.contains("k : f32"), "{text}");
    assert!(!text.contains("& mut [f32]"), "{text}");
    assert!(text.contains("d : & mut Stage < f32 >"), "{text}");
}

#[test]
fn the_crate_option_is_passed_to_each_stage() {
    let text = batch(
        r#""t", on(&device); crate(facade::gust); |b| {
            let a = b.input::<f32>(4u32);
            let c = gust::gpu_task!(4u32, |i| -> f32 { a[i] });
            c
        }"#,
    );
    assert!(!failed(&text), "{text}");
    assert!(text.contains("crate (facade :: gust) ;"), "{text}");
    assert!(
        text.starts_with("facade :: gust :: batch :: build"),
        "{text}"
    );
}

// ---- E1005 ----

#[test]
fn in_flight_outside_one_to_four_is_e1005() {
    for count in ["5", "8"] {
        let text = batch(&format!("\"t\", on(&d); in_flight({count}); |b| {{ }}"));
        assert!(failed(&text), "{text}");
        assert!(text.contains("from 1 to 4"), "{text}");
        assert_rendered_code(&text, Code::BatchForm);
    }
    batch_rejected("\"t\", on(&d); in_flight(0); |b| { }", Code::BatchForm);
    batch_rejected("\"t\", on(&d); in_flight(n); |b| { }", Code::BatchForm);
    batch_rejected(
        "\"t\", on(&d); in_flight(2); in_flight(2); |b| { }",
        Code::BatchForm,
    );
}

#[test]
fn in_flight_one_to_four_is_accepted_and_reaches_the_build() {
    for count in ["1", "2", "4"] {
        let text = batch(&format!("\"t\", on(&d); in_flight({count}); |b| {{ }}"));
        assert!(!failed(&text), "{text}");
        assert!(text.contains(&format!("(& d) , {count}u32 ,")), "{text}");
    }
    // Left out, it is 1.
    let text = batch("\"t\", on(&d); |b| { }");
    assert!(text.contains("(& d) , 1u32 ,"), "{text}");
}

#[test]
fn a_batch_without_a_device_is_e1005() {
    batch_rejected("\"t\", in_flight(1); |b| { }", Code::BatchForm);
}

#[test]
fn a_stage_in_a_closure_or_a_function_is_e1005() {
    batch_rejected(
        "\"t\", on(&d); |b| { let a = b.input::<f32>(4u32); let f = || { gust::gpu_task!(4u32, |i| -> f32 { a[i] }) }; f(); }",
        Code::BatchForm,
    );
    batch_rejected(
        "\"t\", on(&d); |b| { let a = b.input::<f32>(4u32); fn g() { gust::gpu_task!(4u32, |i| -> f32 { 1.0 }); } g(); }",
        Code::BatchForm,
    );
}

#[test]
fn a_stage_in_an_async_block_or_a_loop_is_e1006() {
    for scope in [
        "async { gust::gpu_task!(4u32, |i| -> f32 { a[i] }); }",
        "for _ in 0..2 { gust::gpu_task!(4u32, |i| -> f32 { a[i] }); }",
        "while true { gust::gpu_task!(4u32, |i| -> f32 { a[i] }); }",
        "loop { gust::gpu_task!(4u32, |i| -> f32 { a[i] }); }",
    ] {
        batch_rejected(
            &format!("\"t\", on(&d); |b| {{ let a = b.input::<f32>(4u32); {scope} }}"),
            Code::BatchBody,
        );
    }
}

#[test]
fn a_stage_in_an_impl_method_is_e1005() {
    batch_rejected(
        "\"t\", on(&d); |b| { let a = b.input::<f32>(4u32); impl S { fn g() { gust::gpu_task!(4u32, |i| -> f32 { 1.0 }); } } }",
        Code::BatchForm,
    );
}

#[test]
fn a_block_name_of_a_type_the_macro_cannot_see_is_e1005() {
    batch_rejected(
        "\"t\", on(&d); |b| { let a = make(); let c = gust::gpu_task!(4u32, |i| -> f32 { a[i] }); }",
        Code::BatchForm,
    );
    // Written in the list, it is a scalar the user typed: fine.
    let text = batch(
        "\"t\", on(&d); |b| { let k = 2.0f32; let c = gust::gpu_task!(4u32, [k: f32], |i| -> f32 { k }); }",
    );
    assert!(!failed(&text), "{text}");
}

#[test]
fn a_host_slice_in_a_stage_of_a_batch_is_e1005() {
    for input in [
        "batch(b, \"s\"); n, [a: &[f32]], |i| -> f32 { a[i] }",
        "batch(b, \"s\"); &mut d, [d: &mut [f32]], |i| { d[i] = 1.0; }",
    ] {
        stage_rejected(input, Code::BatchForm);
    }
}

#[test]
fn on_in_a_stage_of_a_batch_is_e1005() {
    stage_rejected(
        "batch(b, \"s\"); on(&device); n, [h: Stage<f32>], |i| -> f32 { h[i] }",
        Code::BatchForm,
    );
}

#[test]
fn a_shape_parameter_read_in_a_body_is_e1005() {
    stage_rejected(
        "batch(b, \"s\"); n, [p: Param<(u32, u32)>], |i| -> u32 { i }",
        Code::BatchForm,
    );
}

// ---- E1006 ----

#[test]
fn a_late_read_and_a_3d_index_are_e1006() {
    for input in [
        "batch(b, \"s\"); n, [h: Stage<f32>], |i| -> f32 { h.late() }",
        "batch(b, \"s\"); n, [h: Stage<f32>], |i| -> f32 { h[(i, 0u32, 0u32)] }",
    ] {
        stage_rejected(input, Code::BatchBody);
    }
}

// ---- the lowering ----

#[test]
fn a_2d_handle_is_indexed_row_major_over_its_width_and_a_parameter_is_its_scalar() {
    let scratch = Scratch::new();
    let text = stage(
        "batch(b, \"lum\"); (w, h), [px: Stage<f32>, gain: Param<f32>], |x, y| -> f32 { px[(x, y)] * gain.get() }",
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    assert!(text.contains("__gust_batch"), "{text}");
    let slang = scratch.slang();
    // The width is a scalar of the extent record, and the view is flat.
    assert!(slang.contains("par_for_w_px"), "{slang}");
    assert!(slang.contains("gain"), "{slang}");
    assert!(!slang.contains("get("), "{slang}");
}

#[test]
fn a_1d_handle_keeps_its_index() {
    let scratch = Scratch::new();
    let text = stage(
        "batch(b, \"s\"); n, [h: Stage<f32>], |i| -> f32 { h[i] * 2.0 }",
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    let slang = scratch.slang();
    assert!(slang.contains("h[i]"), "{slang}");
    assert!(!slang.contains("par_for_w_h"), "{slang}");
}

#[test]
fn a_mixed_rank_use_of_one_handle_is_the_rank_error() {
    stage_rejected(
        "batch(b, \"s\"); (w, h), [px: Stage<f32>], |x, y| -> f32 { px[(x, y)] + px[x] }",
        Code::TaskIndexRank,
    );
}

// ---- an enum parameter (m14.s2e2b, D270) ----

const ENUM_BLOCK: &str = r#""shade", on(&device); |b| {
    let mode = b.param::<Mode>(Mode::Linear);
    let x = b.input::<u32>(8u32);
    let y = gust::gpu_task!(8u32, |i| -> u32 {
        let v: u32 = match mode.get() { Mode::Linear => x[i], Mode::Square => x[i] * x[i] };
        v
    });
    (mode, x, y)
}"#;

/// The words a user's enum hands the batch, as the derive's callback macro writes them.
const MODE_DEFINITION: &str = "# [repr (u32)] enum Mode { Square = 5u32 , Linear = 0u32 }";

#[test]
fn a_block_with_no_type_of_the_users_expands_in_one_step_as_before() {
    // The whole expansion, verbatim (captured from the macro after the trailing-comma fix; the base emitter writes the
    // same tokens for a block with no type of the user's): no hidden clause, no callback, no `enums`.
    let text = batch(BLOCK);
    assert_eq!(text, include_str!("batch_no_enum.golden").trim_end());
    // And a stage of it: the host function has the generics it had before enums (no trailing comma, no enum bound).
    let scratch = Scratch::new();
    let stage_text = stage(
        "batch(b, \"lum\"); (w, h), [px: Stage<f32>, gain: Param<f32>], |x, y| -> f32 { px[(x, y)] * gain.get() }",
        &scratch,
    );
    assert!(!failed(&stage_text), "{stage_text}");
    assert!(
        stage_text.contains("__gust_batch < '__gust_b , __GustD > ("),
        "{stage_text}"
    );
    for hidden in ["GpuEnum", "par_for_enum", "__gust_enum"] {
        assert!(!stage_text.contains(hidden), "{hidden}: {stage_text}");
    }
    assert!(!scratch.slang().contains("enum "), "{}", scratch.slang());
}

#[test]
fn a_block_with_an_enum_parameter_first_checks_the_bound_and_calls_the_callback_of_the_enum() {
    let text = batch(ENUM_BLOCK);
    assert!(!failed(&text), "{text}");
    assert!(text.contains("__gust_enum_check :: < Mode >"), "{text}");
    assert!(text.contains("Mode ! (acc = [] ;"), "{text}");
    assert!(text.contains(":: gust :: gpu_batch"), "{text}");
    // The original tokens ride along for step two.
    assert!(text.contains("orig = {"), "{text}");
}

#[test]
fn the_callback_chain_names_every_enum_and_a_primitive_is_not_one() {
    let text = batch(
        r#""s", on(&device); |b| {
            let a = b.param::<Mode>(Mode::Linear);
            let t = b.param::<types::Tint>(Tint::Warm);
            let g = b.param::<f32>(1.0);
            let v = b.param::<(u32, u32)>((1, 2));
            (a, t, g, v)
        }"#,
    );
    assert!(text.contains("Mode ! (acc = [] ;"), "{text}");
    assert!(text.contains("next = [{ types :: Tint }]"), "{text}");
    assert_eq!(text.matches("__gust_enum_check ::").count(), 2, "{text}");
}

#[test]
fn step_two_gives_each_stage_the_enums_clause() {
    let text = batch(&format!(
        "@ enums [ {{ {MODE_DEFINITION} }} ] ; {ENUM_BLOCK}"
    ));
    assert!(!failed(&text), "{text}");
    assert!(text.contains("enums { # [repr (u32)] enum Mode"), "{text}");
    assert!(!text.contains("@ enums"), "{text}");
}

#[test]
fn two_enums_of_one_name_are_e1005() {
    batch_rejected(
        &format!("@ enums [ {{ {MODE_DEFINITION} }} {{ {MODE_DEFINITION} }} ] ; {ENUM_BLOCK}"),
        Code::BatchForm,
    );
}

#[test]
fn a_stage_with_an_enum_parameter_matches_on_its_words_in_the_kernel() {
    let scratch = Scratch::new();
    let text = stage(
        &format!(
            "batch(b, \"s\"); enums {{ {MODE_DEFINITION} }} 8u32, [x: Stage<u32>, mode: Param<Mode>], |i| -> u32 {{ \
             let v: u32 = match mode.get() {{ Mode::Linear => x[i], Mode::Square => x[i] * x[i] }}; v }}"
        ),
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    let slang = scratch.slang();
    // The record carries the word; the kernel decodes it into the module's own copy, and the decoder compares the
    // word the definition gave (`Square` is 5, not its position).
    assert!(slang.contains("if ((word == uint(5)))"), "{slang}");
    assert!(slang.contains("return Mode::Square;"), "{slang}");
    assert!(slang.contains("par_for_enum_Mode"), "{slang}");
}

#[test]
fn an_enum_parameter_outside_a_match_is_still_a_parameter() {
    let scratch = Scratch::new();
    let text = stage(
        &format!(
            "batch(b, \"s\"); enums {{ {MODE_DEFINITION} }} 8u32, [x: Stage<u32>, mode: Param<Mode>], |i| -> u32 {{ \
             let m: Mode = mode.get(); let v: u32 = match m {{ Mode::Linear => x[i], Mode::Square => 2u32 }}; v }}"
        ),
        &scratch,
    );
    assert!(!failed(&text), "{text}");
}

#[test]
fn two_enums_that_differ_in_case_keep_two_decoders() {
    let scratch = Scratch::new();
    let text = stage(
        "batch(b, \"s\"); enums { # [repr (u32)] enum Mode { A = 0u32 , B = 1u32 } \
         # [repr (u32)] enum MODE { C = 3u32 , D = 4u32 } } 8u32, [x: Stage<u32>, m: Param<Mode>, u: Param<MODE>], \
         |i| -> u32 { let a: u32 = match m.get() { Mode::A => x[i], Mode::B => 2u32 }; \
         let c: u32 = match u.get() { MODE::C => a, MODE::D => 3u32 }; c }",
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    let slang = scratch.slang();
    for name in ["par_for_enum_Mode", "par_for_enum_MODE"] {
        assert!(slang.contains(name), "{name}: {slang}");
    }
}

#[test]
fn a_stage_asserts_every_word_of_its_copy_against_the_type_the_parameter_names() {
    let scratch = Scratch::new();
    let text = stage(
        &format!(
            "batch(b, \"s\"); enums {{ {MODE_DEFINITION} }} 8u32, [x: Stage<u32>, mode: Param<types::Mode>, \
             again: Param<types::Mode>], |i| -> u32 {{ let v: u32 = match mode.get() {{ Mode::Linear => x[i], \
             Mode::Square => 1u32 }}; v }}"
        ),
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    // One block for the one path (two parameters of it), a word per variant of the copy, and the exhaustive `match`.
    assert_eq!(text.matches("__gust_enum_variants").count(), 1, "{text}");
    for expected in [
        "assert ! (types :: Mode :: Square as :: core :: primitive :: u32 == 5u32)",
        "assert ! (types :: Mode :: Linear as :: core :: primitive :: u32 == 0u32)",
        "match value { types :: Mode :: Square => { } types :: Mode :: Linear => { } }",
    ] {
        assert!(text.contains(expected), "{expected} not in {text}");
    }
}

#[test]
fn a_parameter_that_names_none_of_the_enums_of_the_block_says_so() {
    let scratch = Scratch::new();
    let text = stage(
        &format!(
            "batch(b, \"s\"); enums {{ {MODE_DEFINITION} }} 8u32, [x: Stage<u32>, shade: Param<Shade>], \
             |i| -> u32 {{ x[i] }}"
        ),
        &scratch,
    );
    assert!(failed(&text), "{text}");
    assert_rendered_code(&text, Code::BatchForm);
    assert!(
        text.contains(
            "`Shade` is not the name of an enum the block's derives handed over (`Mode`)"
        ),
        "{text}"
    );
    // A primitive keeps its own text.
    let other = stage(
        &format!(
            "batch(b, \"s\"); enums {{ {MODE_DEFINITION} }} 8u32, [x: Stage<u32>, big: Param<u64>], \
             |i| -> u32 {{ x[i] }}"
        ),
        &Scratch::new(),
    );
    assert!(other.contains("a parameter a body cannot read"), "{other}");
}

#[test]
fn enums_that_differ_in_one_word_make_different_kernels() {
    let source = |square: u32| {
        format!(
            "batch(b, \"s\"); enums {{ # [repr (u32)] enum Mode {{ Square = {square}u32 , Linear = 0u32 }} }} 8u32, \
             [x: Stage<u32>, mode: Param<Mode>], |i| -> u32 {{ let v: u32 = match mode.get() {{ \
             Mode::Linear => x[i], Mode::Square => x[i] * x[i] }}; v }}"
        )
    };
    let (five, six) = (Scratch::new(), Scratch::new());
    let (text_five, text_six) = (stage(&source(5), &five), stage(&source(6), &six));
    assert!(!failed(&text_five) && !failed(&text_six));
    assert!(five.slang().contains("if ((word == uint(5)))"));
    assert!(six.slang().contains("if ((word == uint(6)))"));
    assert_ne!(five.slang(), six.slang());
    // The module's name carries the hash of the lowered kernel, so the artifact key moves with the word.
    assert_ne!(text_five, text_six);
    let module = |text: &str| {
        text.split("mod __gust_pf_")
            .nth(1)
            .and_then(|rest| rest.split_whitespace().next())
            .map(str::to_owned)
    };
    assert_ne!(module(&text_five), module(&text_six));
}

// ---- an enum stored as an element (m14.s2e2c, D275) ----

/// An enum with no variant whose word is 0.
const SHADE_DEFINITION: &str = "# [repr (u32)] enum Shade { Dim = 2u32 , Lit = 5u32 }";

const ELEMENT_BLOCK: &str = r#""shade", on(&device); |b| {
    let modes = b.input::<Mode>(8u32);
    let next = gust::gpu_task!(8u32, |i| -> Mode {
        match modes[i] { Mode::Linear => Mode::Square, Mode::Square => Mode::Linear }
    });
    (modes, next)
}"#;

#[test]
fn an_input_and_a_result_of_enum_type_are_collected_like_a_parameter() {
    let text = batch(ELEMENT_BLOCK);
    assert!(!failed(&text), "{text}");
    // One bound check and one callback for the one enum, however many places name it.
    assert_eq!(text.matches("__gust_enum_check ::").count(), 1, "{text}");
    assert!(text.contains("Mode ! (acc = [] ;"), "{text}");
    // A numeric element is not an enum.
    let numeric = batch(
        r#""s", on(&device); |b| { let a = b.input::<f32>(4u32); let c = b.input::<Vec4>(4u32);
            let d = gust::gpu_task!(4u32, |i| -> u32 { 0u32 }); (a, c, d) }"#,
    );
    assert!(!numeric.contains("__gust_enum_check"), "{numeric}");
}

#[test]
fn a_stage_reads_an_enum_view_through_the_one_decoder_and_stores_the_word() {
    let scratch = Scratch::new();
    let text = stage(
        &format!(
            "batch(b, \"s\"); enums {{ {MODE_DEFINITION} }} 8u32, [x: Stage<Mode>], |i| -> Mode {{ \
             match x[i] {{ Mode::Linear => Mode::Square, Mode::Square => Mode::Linear }} }}"
        ),
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    let slang = scratch.slang();
    // The buffers are words, a load decodes through the decoder the parameters use, and the result is stored as its word.
    assert!(slang.contains("StructuredBuffer<uint> x"), "{slang}");
    assert!(slang.contains("par_for_enum_Mode(x["), "{slang}");
    assert!(slang.contains("uint(par_for_enum_result)"), "{slang}");
    assert!(slang.contains("if ((word == uint(5)))"), "{slang}");
    // The handle of the stage is the enum's again, and the view is handed over as its words.
    assert!(text.contains("retype :: < Mode >"), "{text}");
    assert!(text.contains("x . words ()"), "{text}");
}

#[test]
fn a_two_dimensional_enum_view_decodes_its_loads() {
    let scratch = Scratch::new();
    let text = stage(
        &format!(
            "batch(b, \"s\"); enums {{ {MODE_DEFINITION} }} (w, h), [g: Stage<Mode>], |x, y| -> u32 {{ \
             let v: u32 = match g[(x, y)] {{ Mode::Linear => x, Mode::Square => y }}; v }}"
        ),
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    let slang = scratch.slang();
    assert!(slang.contains("par_for_enum_Mode(g["), "{slang}");
    assert!(slang.contains("par_for_w_g"), "{slang}");
}

#[test]
fn elements_that_differ_in_one_word_make_different_kernels() {
    let source = |square: u32| {
        format!(
            "batch(b, \"s\"); enums {{ # [repr (u32)] enum Mode {{ Square = {square}u32 , Linear = 0u32 }} }} 8u32, \
             [x: Stage<Mode>], |i| -> Mode {{ match x[i] {{ Mode::Linear => Mode::Square, Mode::Square => Mode::Linear }} }}"
        )
    };
    let (five, six) = (Scratch::new(), Scratch::new());
    let (text_five, text_six) = (stage(&source(5), &five), stage(&source(6), &six));
    assert!(!failed(&text_five) && !failed(&text_six));
    assert_ne!(five.slang(), six.slang());
    assert_ne!(text_five, text_six);
}

#[test]
fn an_enum_element_without_a_word_of_zero_is_e1012_at_the_element_type() {
    // An input, a result type and a stage's view each carry the rule; a parameter of the same enum does not.
    let input = format!(
        "@ enums [ {{ {SHADE_DEFINITION} }} ] ; \"s\", on(&device); |b| {{ let x = b.input::<Shade>(4u32); x }}"
    );
    batch_rejected(&input, Code::BatchEnumNoZero);
    let result = format!(
        "@ enums [ {{ {SHADE_DEFINITION} }} ] ; \"s\", on(&device); |b| {{ \
         let x = gust::gpu_task!(4u32, |i| -> Shade {{ Shade::Dim }}); x }}"
    );
    batch_rejected(&result, Code::BatchEnumNoZero);
    stage_rejected(
        &format!(
            "batch(b, \"s\"); enums {{ {SHADE_DEFINITION} }} 8u32, [x: Stage<Shade>], |i| -> u32 {{ 0u32 }}"
        ),
        Code::BatchEnumNoZero,
    );
    let parameter = batch(&format!(
        "@ enums [ {{ {SHADE_DEFINITION} }} ] ; \"s\", on(&device); |b| {{ let p = b.param::<Shade>(Shade::Dim); p }}"
    ));
    assert!(!failed(&parameter), "{parameter}");
}

#[test]
fn an_enum_in_place_target_loads_through_the_decoder_and_stores_the_encoded_word() {
    let scratch = Scratch::new();
    let text = stage(
        &format!(
            "batch(b, \"s\"); enums {{ {MODE_DEFINITION} }} &mut t, [t: &mut Stage<Mode>], |i| {{ \
             t[i] = match t[i] {{ Mode::Linear => Mode::Square, Mode::Square => Mode::Linear }}; }}"
        ),
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    let slang = scratch.slang();
    // The target is a buffer of words, the load decodes and the store writes the word of the value.
    assert!(slang.contains("RWStructuredBuffer<uint> t"), "{slang}");
    assert!(slang.contains("par_for_enum_Mode(t["), "{slang}");
    assert!(slang.contains("uint(par_for_enum_store)"), "{slang}");
    assert!(text.contains("t . words ()"), "{text}");
}

#[test]
fn a_two_dimensional_enum_target_is_still_the_rank_error() {
    stage_rejected(
        &format!(
            "batch(b, \"s\"); enums {{ {MODE_DEFINITION} }} &mut t, [t: &mut Stage<Mode>], |i| {{ \
             t[(i, i)] = Mode::Linear; }}"
        ),
        Code::TaskIndexRank,
    );
}

#[test]
fn a_compound_assignment_to_an_enum_target_is_e0122() {
    stage_rejected(
        &format!(
            "batch(b, \"s\"); enums {{ {MODE_DEFINITION} }} &mut t, [t: &mut Stage<Mode>], |i| {{ \
             t[i] += Mode::Linear; }}"
        ),
        Code::EnumShape,
    );
}

#[test]
fn an_atomic_on_an_enum_target_is_e0122() {
    stage_rejected(
        &format!(
            "batch(b, \"s\"); enums {{ {MODE_DEFINITION} }} &mut t, [t: &mut Stage<Mode>], |i| {{ \
             atomic_add(&mut t[i], 1u32); }}"
        ),
        Code::EnumShape,
    );
}

#[test]
fn an_enum_target_without_a_word_of_zero_is_e1012() {
    stage_rejected(
        &format!(
            "batch(b, \"s\"); enums {{ {SHADE_DEFINITION} }} &mut t, [t: &mut Stage<Shade>], |i| {{ \
             t[i] = Shade::Dim; }}"
        ),
        Code::BatchEnumNoZero,
    );
}

#[test]
fn a_host_slice_of_enums_is_refused_like_every_host_slice() {
    let slice = stage(
        &format!(
            "batch(b, \"s\"); enums {{ {MODE_DEFINITION} }} 8u32, [m: &[Mode]], |i| -> u32 {{ 0u32 }}"
        ),
        &Scratch::new(),
    );
    assert!(failed(&slice), "{slice}");
    assert_rendered_code(&slice, Code::BatchForm);
    assert!(slice.contains("arrives in a later release"), "{slice}");
}

#[test]
fn a_target_that_differs_in_one_word_makes_a_different_kernel() {
    let source = |square: u32| {
        format!(
            "batch(b, \"s\"); enums {{ # [repr (u32)] enum Mode {{ Square = {square}u32 , Linear = 0u32 }} }} \
             &mut t, [t: &mut Stage<Mode>], |i| {{ t[i] = Mode::Square; }}"
        )
    };
    let (five, six) = (Scratch::new(), Scratch::new());
    let (text_five, text_six) = (stage(&source(5), &five), stage(&source(6), &six));
    assert!(!failed(&text_five) && !failed(&text_six), "{text_five}");
    assert_ne!(five.slang(), six.slang());
    assert_ne!(text_five, text_six);
}

#[test]
fn an_enum_handle_is_no_operand_of_a_library_op() {
    batch_rejected(
        &format!(
            "@ enums [ {{ {MODE_DEFINITION} }} ] ; \"s\", on(&device); |b| {{ \
             let x = b.input::<Mode>(4u32); let n = gust::sum(&x); (x, n) }}"
        ),
        Code::BatchOpForm,
    );
}

#[test]
fn a_store_to_an_enum_capture_is_still_the_write_error() {
    stage_rejected(
        &format!(
            "batch(b, \"s\"); enums {{ {MODE_DEFINITION} }} 8u32, [x: Stage<Mode>], |i| -> u32 {{ \
             x[i] = Mode::Linear; 0u32 }}"
        ),
        Code::TaskWrite,
    );
}

#[test]
fn a_parameter_and_an_element_load_call_the_one_decoder_whose_last_variant_is_the_fallback() {
    let scratch = Scratch::new();
    let text = stage(
        &format!(
            "batch(b, \"s\"); enums {{ {MODE_DEFINITION} }} 8u32, [x: Stage<Mode>, mode: Param<Mode>], |i| -> u32 {{ \
             let a: u32 = match mode.get() {{ Mode::Linear => 1u32, Mode::Square => 2u32 }}; \
             let b: u32 = match x[i] {{ Mode::Linear => 10u32, Mode::Square => 20u32 }}; a + b }}"
        ),
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    let slang = scratch.slang();
    // One decoder serves the parameter and the element load: defined once, called from both places.
    for module in slang
        .split("// Generated by GUST")
        .filter(|module| !module.trim().is_empty())
    {
        assert_eq!(
            module.matches("par_for_enum_Mode(uint word)").count(),
            1,
            "{module}"
        );
        assert!(module.contains("par_for_enum_Mode(x["), "{module}");
        assert!(
            module.contains("par_for_enum_Mode(__gust_ext.mode)"),
            "{module}"
        );
    }
    // The decoder compares the earlier variants' words only (`Square` is 5); the last variant (`Linear`, word 0) is
    // what every other word gives, on a parameter and on an element alike: it has no comparison of its own.
    assert!(slang.contains("if ((word == uint(5)))"), "{slang}");
    assert!(!slang.contains("word == uint(0)"), "{slang}");
    assert!(slang.contains("return Mode::Linear;"), "{slang}");
}

#[test]
fn a_user_struct_as_an_input_is_taken_for_an_enum_in_a_block() {
    // D275 (review finding S4): a `GpuPod` struct of the user's as `b.input::<Particle>(n)` is a type step one cannot tell
    // from an enum, so it calls the type's own macro (rustc then says "cannot find macro `Particle`", and
    // "`Particle` is not a GPU enum"). Documented in CHANGELOG and GUST-E1005; `gust::batch::build` is the way around.
    let text = batch(r#""s", on(&device); |b| { let p = b.input::<Particle>(4u32); p }"#);
    assert!(text.contains("Particle ! (acc = [] ;"), "{text}");
    assert_eq!(text.matches("__gust_enum_check ::").count(), 1, "{text}");
}
