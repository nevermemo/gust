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
fn a_block_with_an_enum_parameter_first_calls_the_callback_of_the_enum_and_step_two_checks_the_bound()
 {
    let text = batch(ENUM_BLOCK);
    assert!(!failed(&text), "{text}");
    assert!(text.contains("Mode ! (acc = [] ;"), "{text}");
    assert!(text.contains(":: gust :: gpu_batch"), "{text}");
    // The original tokens ride along for step two, which knows the kind of the type and checks it.
    assert!(text.contains("orig = {"), "{text}");
    assert!(!text.contains("__gust_enum_check"), "{text}");
    let resolved = batch(&format!(
        "@ enums [ {{ {MODE_DEFINITION} }} ] ; {ENUM_BLOCK}"
    ));
    assert!(!failed(&resolved), "{resolved}");
    assert!(
        resolved.contains("__gust_enum_check :: < Mode >"),
        "{resolved}"
    );
    assert!(!resolved.contains("__gust_struct_check"), "{resolved}");
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
    assert!(!text.contains("__gust_enum_check"), "{text}");
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
    // One callback for the one enum, however many places name it, and one bound check in step two.
    assert!(text.contains("Mode ! (acc = [] ;"), "{text}");
    assert_eq!(text.matches("Mode ! (").count(), 1, "{text}");
    let resolved = batch(&format!(
        "@ enums [ {{ {MODE_DEFINITION} }} ] ; {ELEMENT_BLOCK}"
    ));
    assert!(!failed(&resolved), "{resolved}");
    assert_eq!(
        resolved.matches("__gust_enum_check ::").count(),
        1,
        "{resolved}"
    );
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
fn a_user_struct_as_an_input_calls_the_callback_of_the_type() {
    // D275 (review finding S4), settled by D279: a struct of the user's as `b.input::<Particle>(n)` is a type step one
    // cannot tell from an enum, so it calls the type's own macro, which a `#[derive(GpuStruct)]` struct has (a struct that
    // no derive handed over has none: rustc says "cannot find macro `Particle`"). The kind is the derive's.
    let text = batch(r#""s", on(&device); |b| { let p = b.input::<Particle>(4u32); p }"#);
    assert!(text.contains("Particle ! (acc = [] ;"), "{text}");
    assert!(!text.contains("__gust_enum_check"), "{text}");
}

// ---- counted stages (D278) ----

#[test]
fn a_counted_stage_has_a_linear_index_over_a_two_dimensional_grid_and_an_args_kernel() {
    let scratch = Scratch::new();
    let text = stage(
        "batch(b, \"s\"); count(n); 8u32, [n: Stage<u32>, src: Stage<f32>], |i| -> f32 { src[i] }",
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    // The stage is sealed through the counted form, over the pair of kernels.
    assert!(text.contains("finish_counted"), "{text}");
    assert!(text.contains("require_count"), "{text}");
    assert!(text.contains("run_args"), "{text}");
    let slang = scratch.slang();
    // The counted kernel: the row is the linear index over the grid, clamped by the count and the capacity.
    assert!(slang.contains("65535"), "{slang}");
    assert!(slang.contains("min("), "{slang}");
    // The args kernel writes three words.
    assert!(
        slang.matches("65535").count() >= 4,
        "the grid is clamped in the args kernel too: {slang}"
    );
}

#[test]
fn an_uncounted_stage_has_no_args_kernel() {
    let scratch = Scratch::new();
    let text = stage(
        "batch(b, \"s\"); 8u32, [src: Stage<f32>], |i| -> f32 { src[i] }",
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    assert!(
        !text.contains("finish_counted") && !text.contains("run_args"),
        "{text}"
    );
    assert!(!scratch.slang().contains("65535"));
}

#[test]
fn count_outside_a_batch_is_e1013() {
    stage_rejected(
        "count(n); 8u32, [n: &[u32], src: &[f32]], |i| -> f32 { src[i] }",
        Code::BatchCount,
    );
    stage_rejected(
        "batch(b, \"s\"); count(n); count(m); 8u32, [n: Stage<u32>, src: Stage<f32>], |i| -> f32 { src[i] }",
        Code::BatchCount,
    );
}

#[test]
fn a_counted_two_dimensional_stage_or_a_tall_workgroup_is_e1014() {
    stage_rejected(
        "batch(b, \"s\"); count(n); (4u32, 4u32), [n: Stage<u32>, src: Stage<f32>], |x, y| -> f32 { src[(x, y)] }",
        Code::BatchCountBody,
    );
    stage_rejected(
        "batch(b, \"s\"); workgroup_size(8, 8, 1); count(n); 8u32, [n: Stage<u32>, src: Stage<f32>], |i| -> f32 { src[i] }",
        Code::BatchCountBody,
    );
}

#[test]
fn a_count_that_is_not_a_u32_handle_the_stage_only_reads_is_e1015() {
    // A handle of another element type, and the stage's own target.
    stage_rejected(
        "batch(b, \"s\"); count(n); 8u32, [n: Stage<f32>, src: Stage<f32>], |i| -> f32 { src[i] }",
        Code::BatchCountHandle,
    );
    stage_rejected(
        "batch(b, \"s\"); count(t); &mut t, [t: &mut Stage<u32>], |i| { t[i] = 1u32; }",
        Code::BatchCountHandle,
    );
    // In a block, a name that is not a handle (a parameter) is refused at the name.
    batch_rejected(
        r#""s", on(&device); |b| { let p = b.param::<u32>(2); let v = b.input::<u32>(8u32); gust::gpu_task!(count(p); 8u32, |i| -> u32 { v[i] }); v }"#,
        Code::BatchCountHandle,
    );
}

#[test]
fn a_block_gives_a_counted_stage_the_count_handle_it_names() {
    let text = batch(
        r#""s", on(&device); |b| { let n = b.input::<u32>(1u32); let v = b.input::<u32>(8u32); let o = gust::gpu_task!(count(n); 8u32, |i| -> u32 { v[i] }); o }"#,
    );
    assert!(!failed(&text), "{text}");
    assert!(text.contains("count (n) ;"), "{text}");
    assert!(text.contains("n : Stage < u32 >"), "{text}");
}

// ---- a struct stored as an element (m14.s2s part b, D279) ----

/// How many bindings of `declaration` the Slang module of the stage's own kernel declares (a stage writes one module
/// per kernel, the args kernel of a counted stage first, and `Scratch::slang` joins them).
fn bindings(slang: &str, declaration: &str) -> usize {
    slang
        .split("// Generated by GUST")
        .find(|module| module.contains("StructuredBuffer<Row>") && !module.contains("run_args"))
        .unwrap_or_default()
        .matches(declaration)
        .count()
}

/// The definition a `#[derive(GpuStruct)]` struct hands over, as the derive's callback macro writes it.
const ROW_DEFINITION: &str =
    "struct Row { pos : Vec2 , life : f32 , flags : u32 , _pad0 : u32 , _pad1 : u32 }";

const ROW_BLOCK: &str = r#""rows", on(&device); |b| {
    let rows = b.input::<Row>(8u32);
    let life = gust::gpu_task!(8u32, |i| -> f32 { rows[i].life });
    (rows, life)
}"#;

#[test]
fn a_struct_input_calls_the_callback_and_step_two_checks_the_marker_and_the_fields() {
    let text = batch(ROW_BLOCK);
    assert!(!failed(&text), "{text}");
    assert!(text.contains("Row ! (acc = [] ;"), "{text}");
    assert!(!text.contains("__gust_struct_check"), "{text}");
    let resolved = batch(&format!("@ enums [ {{ {ROW_DEFINITION} }} ] ; {ROW_BLOCK}"));
    assert!(!failed(&resolved), "{resolved}");
    // The marker bound, then every field of the definition named and typed as its name stands for.
    assert!(
        resolved.contains("__gust_struct_check :: < Row >"),
        "{resolved}"
    );
    assert!(
        resolved.contains("let Row { pos , life , flags , _pad0 , _pad1 } = value"),
        "{resolved}"
    );
    assert!(
        resolved.contains("let _ : :: gust :: math :: Vec2 = pos"),
        "{resolved}"
    );
    assert!(
        resolved.contains("let _ : :: core :: primitive :: f32 = life"),
        "{resolved}"
    );
    assert!(!resolved.contains("__gust_enum_check"), "{resolved}");
    // The hidden clause carries the struct into the stages.
    assert!(resolved.contains("enums { struct Row"), "{resolved}");
}

#[test]
fn a_stage_over_struct_rows_declares_the_struct_and_reaches_the_users_type() {
    let scratch = Scratch::new();
    let text = stage(
        &format!(
            "batch(b, \"s\"); enums {{ {ROW_DEFINITION} }} 8u32, [rows: Stage<Row>], |i| -> f32 {{ rows[i].life }}"
        ),
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    // The kernel sees the struct as a buffer element, and no word, decoder or retype is involved.
    let slang = scratch.slang();
    assert!(slang.contains("struct Row"), "{slang}");
    assert!(slang.contains("StructuredBuffer<Row> rows"), "{slang}");
    assert!(!text.contains("words ()"), "{text}");
    assert!(!text.contains("retype"), "{text}");
    // The Rust side of the module is the user's own type, reached from the module (a copy would be another type).
    assert!(text.contains("use super :: Row ;"), "{text}");
    assert!(text.contains("Stage < super :: Row"), "{text}");
    assert!(!text.contains("pub struct Row"), "{text}");
}

#[test]
fn a_struct_result_is_a_struct_literal_stored_whole() {
    let scratch = Scratch::new();
    let text = stage(
        &format!(
            "batch(b, \"s\"); enums {{ {ROW_DEFINITION} }} 8u32, [rows: Stage<Row>], |i| -> Row {{ \
             Row {{ pos: rows[i].pos, life: rows[i].life * 0.5f32, flags: rows[i].flags, _pad0: 0u32, _pad1: 0u32 }} }}"
        ),
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    let slang = scratch.slang();
    assert!(slang.contains("RWStructuredBuffer<Row>"), "{slang}");
    assert!(slang.contains("__gust_struct_1.life = "), "{slang}");
    // The rows are two bindings, one read and one written, however many fields the struct has.
    assert_eq!(bindings(&slang, "StructuredBuffer<"), 2, "{slang}");
}

#[test]
fn a_counted_stage_over_a_struct_target_is_one_binding_for_the_rows_and_one_for_the_count() {
    let scratch = Scratch::new();
    let text = stage(
        &format!(
            "batch(b, \"s\"); enums {{ {ROW_DEFINITION} }} count(n); &mut t, [n: Stage<u32>, t: &mut Stage<Row>], |i| {{ \
             t[i].flags = 1u32; }}"
        ),
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    let slang = scratch.slang();
    // One in-place struct binding (not one per field), and the count as the one added binding.
    assert_eq!(bindings(&slang, "StructuredBuffer<Row>"), 1, "{slang}");
    assert!(slang.contains("StructuredBuffer<uint> n"), "{slang}");
}

#[test]
fn a_counted_stage_gathers_whole_struct_rows_by_an_index_handle() {
    // m14.s2s2: `rows[idx[i]]` is a struct read at a computed index and `-> Row` under `count(n)` is a counted struct
    // result; the stage is four bindings (rows, index, count, result) whatever the fields.
    let scratch = Scratch::new();
    let text = stage(
        &format!(
            "batch(b, \"s\"); enums {{ {ROW_DEFINITION} }} count(n); 8u32, \
             [n: Stage<u32>, idx: Stage<u32>, rows: Stage<Row>], |i| -> Row {{ rows[idx[i]] }}"
        ),
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    let slang = scratch.slang();
    assert!(slang.contains("RWStructuredBuffer<Row>"), "{slang}");
    assert!(slang.contains("rows[idx[i]]"), "{slang}");
    assert_eq!(bindings(&slang, "StructuredBuffer<Row>"), 2, "{slang}");
    assert_eq!(bindings(&slang, "StructuredBuffer<uint>"), 2, "{slang}");
}

#[test]
fn a_counted_stage_gathers_a_field_built_struct_row_by_an_index_handle() {
    let scratch = Scratch::new();
    let text = stage(
        &format!(
            "batch(b, \"s\"); enums {{ {ROW_DEFINITION} }} count(n); 8u32, \
             [n: Stage<u32>, idx: Stage<u32>, rows: Stage<Row>], |i| -> Row {{ \
             Row {{ pos: rows[idx[i]].pos, life: rows[idx[i]].life * 0.5f32, flags: rows[idx[i]].flags, \
             _pad0: 0u32, _pad1: 0u32 }} }}"
        ),
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    let slang = scratch.slang();
    assert!(slang.contains("rows[idx[i]].life"), "{slang}");
}

#[test]
fn an_in_place_target_of_structs_stores_a_field_or_the_whole_element() {
    let scratch = Scratch::new();
    let text = stage(
        &format!(
            "batch(b, \"s\"); enums {{ {ROW_DEFINITION} }} &mut t, [t: &mut Stage<Row>], |i| {{ \
             t[i].life = t[i].life * 0.5f32; t[i].flags += 1u32; \
             t[i] = Row {{ pos: t[i].pos, life: t[i].life, flags: t[i]._pad0, _pad0: t[i].flags, _pad1: 0u32 }}; }}"
        ),
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    let slang = scratch.slang();
    assert!(slang.contains("RWStructuredBuffer<Row> t"), "{slang}");
    assert!(slang.contains("t[i].life"), "{slang}");
    // The whole element is stored from a temporary built first, so a swap of two fields reads the old values.
    assert!(
        slang.contains("__gust_struct_3.flags = t[i]._pad0;"),
        "{slang}"
    );
    assert!(slang.contains("t[i] = __gust_struct_3;"), "{slang}");
    // An in-place target is one binding.
    assert_eq!(bindings(&slang, "StructuredBuffer<"), 1, "{slang}");
}

#[test]
fn rows_that_differ_in_one_field_make_different_kernels() {
    let source = |field: &str| {
        format!(
            "batch(b, \"s\"); enums {{ struct Row {{ pos : Vec2 , {field} : f32 , flags : u32 , _pad0 : u32 , _pad1 : u32 }} }} \
             8u32, [rows: Stage<Row>], |i| -> u32 {{ rows[i].flags }}"
        )
    };
    let (one, two) = (Scratch::new(), Scratch::new());
    let (text_one, text_two) = (stage(&source("life"), &one), stage(&source("mass"), &two));
    assert!(!failed(&text_one) && !failed(&text_two), "{text_one}");
    assert_ne!(one.slang(), two.slang());
    // The module's name carries the hash of the lowered kernel, so the artifact key moves with a field's name.
    assert_ne!(text_one, text_two);
    let module = |text: &str| {
        text.split("mod __gust_pf_")
            .nth(1)
            .and_then(|rest| rest.split_whitespace().next())
            .map(str::to_owned)
    };
    assert!(module(&text_one).is_some(), "{text_one}");
    assert_ne!(module(&text_one), module(&text_two));
    // A retyped field and a reordered pair of fields are each another struct, so another kernel and another module.
    let with_definition = |definition: &str, scratch: &Scratch| {
        stage(
            &format!(
                "batch(b, \"s\"); enums {{ {definition} }} 8u32, [rows: Stage<Row>], |i| -> u32 {{ rows[i]._pad0 }}"
            ),
            scratch,
        )
    };
    let (base, retype, swapped) = (Scratch::new(), Scratch::new(), Scratch::new());
    let text_base = with_definition(ROW_DEFINITION, &base);
    let text_retype = with_definition(
        "struct Row { pos : Vec2 , life : f32 , flags : i32 , _pad0 : u32 , _pad1 : u32 }",
        &retype,
    );
    let text_swapped = with_definition(
        "struct Row { pos : Vec2 , flags : u32 , life : f32 , _pad0 : u32 , _pad1 : u32 }",
        &swapped,
    );
    for text in [&text_base, &text_retype, &text_swapped] {
        assert!(!failed(text), "{text}");
    }
    assert_ne!(base.slang(), retype.slang());
    assert_ne!(base.slang(), swapped.slang());
    assert_ne!(module(&text_base), module(&text_retype));
    assert_ne!(module(&text_base), module(&text_swapped));
}

#[test]
fn the_slang_struct_lists_the_fields_in_the_order_of_the_definition() {
    // The layout the GPU sees is the order the fields are declared in, the order of the user's `#[repr(C)]` struct.
    let scratch = Scratch::new();
    let text = stage(
        &format!(
            "batch(b, \"s\"); enums {{ {ROW_DEFINITION} }} 8u32, [rows: Stage<Row>], |i| -> u32 {{ rows[i].flags }}"
        ),
        &scratch,
    );
    assert!(!failed(&text), "{text}");
    let slang = scratch.slang();
    let declared = slang
        .split("struct Row")
        .nth(1)
        .and_then(|rest| rest.split('}').next())
        .unwrap_or_default();
    let positions: Vec<Option<usize>> = ["pos", "life", "flags", "_pad0", "_pad1"]
        .iter()
        .map(|field| declared.find(&format!(" {field};")))
        .collect();
    assert!(
        positions.iter().all(Option::is_some) && positions.is_sorted(),
        "{declared}"
    );
}

#[test]
fn a_struct_handle_is_no_operand_of_a_library_op() {
    batch_rejected(
        &format!(
            "@ enums [ {{ {ROW_DEFINITION} }} ] ; \"s\", on(&device); |b| {{ \
             let x = b.input::<Row>(4u32); let n = gust::sum(&x); (x, n) }}"
        ),
        Code::BatchStructOperand,
    );
}

#[test]
fn the_refusal_of_a_struct_operand_names_the_form_that_compacts_rows() {
    // m14.s2s2: `compact` takes flags, so its message says what to write; the other ops keep their own text.
    let refused = |call: &str| {
        let text = batch(&format!(
            "@ enums [ {{ {ROW_DEFINITION} }} ] ; \"s\", on(&device); |b| {{ \
             let x = b.input::<Row>(4u32); let n = {call}; (x, n) }}"
        ));
        assert!(failed(&text), "{text}");
        assert_rendered_code(&text, Code::BatchStructOperand);
        text
    };
    let compact = refused("gust::compact(&x)");
    for part in [
        "the operand of `compact` is a buffer of `u32` flags",
        "build the flags in a stage",
        "gather the rows in a counted stage",
        "count(n); CAP, |i| -> Row { rows[idx[i]] }",
    ] {
        assert!(compact.contains(part), "`{part}` missing from: {compact}");
    }
    let sum = refused("gust::sum(&x)");
    assert!(
        sum.contains("an op over rows is not in this release"),
        "{sum}"
    );
    assert!(!sum.contains("counted stage"), "{sum}");
}

#[test]
fn a_struct_is_no_parameter() {
    stage_rejected(
        &format!(
            "batch(b, \"s\"); enums {{ {ROW_DEFINITION} }} 8u32, [p: Param<Row>], |i| -> u32 {{ 0u32 }}"
        ),
        Code::BatchStructParam,
    );
}

#[test]
fn a_struct_written_with_two_paths_in_one_stage_is_refused() {
    stage_rejected(
        &format!(
            "batch(b, \"s\"); enums {{ {ROW_DEFINITION} }} 8u32, [a: Stage<Row>, c: Stage<rows::Row>], |i| -> u32 {{ \
             a[i].flags + c[i].flags }}"
        ),
        Code::BatchForm,
    );
}

#[test]
fn a_struct_and_an_enum_of_one_name_are_refused() {
    batch_rejected(
        &format!(
            "@ enums [ {{ {ROW_DEFINITION} }} {{ # [repr (u32)] enum Row {{ A = 0u32 }} }} ] ; {ROW_BLOCK}"
        ),
        Code::BatchForm,
    );
}

#[test]
fn a_struct_as_the_count_of_a_stage_is_still_the_count_error() {
    stage_rejected(
        &format!(
            "batch(b, \"s\"); enums {{ {ROW_DEFINITION} }} count(rows); 8u32, [rows: Stage<Row>], |i| -> u32 {{ \
             rows[i].flags }}"
        ),
        Code::BatchCountHandle,
    );
}
