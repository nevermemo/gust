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
fn a_late_read_a_stage_level_match_and_a_3d_index_are_e1006() {
    for input in [
        "batch(b, \"s\"); n, [h: Stage<f32>], |i| -> f32 { h.late() }",
        "batch(b, \"s\"); n, [k: u32], |i| -> u32 { match k { 0u32 => 1u32, _ => 2u32 } }",
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
