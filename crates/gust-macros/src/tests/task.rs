//! `gpu_task!` (slice m14.s2a1): the lowering to the `par_for!` generator (the kernel's Slang, the identity of an
//! expansion, the export records), the host function it adds, and the five diagnostics of the E10xx family.

use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use proc_macro2::TokenStream;

use crate::{
    diagnostics::{Code, testing::assert_rendered_code},
    expand::{Environment, Profile},
    task::gpu_task_in,
};

/// A scratch directory under the system temp directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "gust task {name} {} {}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    /// The names of the files of the BODIES that end with `suffix`, sorted: a grid kernel's split (D209) is a second
    /// record under a key of its own, and `every_file` lists it too.
    fn files(&self, suffix: &str) -> Vec<String> {
        self.every_file(suffix)
            .into_iter()
            .filter(|name| !self.is_split(name))
            .collect()
    }

    /// Whether the file `name` (`<key>.slang`, `<key>.kernel.json`, `<key>.<site>.site.json`) belongs to a split.
    fn is_split(&self, name: &str) -> bool {
        let key = &name[..16];
        fs::read_to_string(self.0.join("demo").join(format!("{key}.kernel.json")))
            .is_ok_and(|record| record.contains("\"split_of\""))
    }

    /// The names of every exported file that ends with `suffix`, bodies and splits, sorted.
    fn every_file(&self, suffix: &str) -> Vec<String> {
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
        fs::read_to_string(self.0.join("demo").join(name))
            .unwrap()
            .replace("\r\n", "\n")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn environment(scratch: &Scratch) -> Environment {
    Environment {
        manifest_dir: None,
        export_dir: Some(scratch.0.clone()),
        package: Some("demo".to_owned()),
        reported_file: Some("src/demo.rs".to_owned()),
    }
}

/// The text of a `gpu_task!` expansion for `input`, with the call on `call_line`.
fn expand(input: &str, scratch: &Scratch, call_line: u32, profile: Profile) -> String {
    let tokens: TokenStream = input.parse().unwrap();
    gpu_task_in(tokens, &environment(scratch), profile, call_line).to_string()
}

/// Whether an expansion is an error (`{ ::core::compile_error!{..} Task::new(..) }`) rather than a kernel.
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

/// The Slang of the one kernel a call exports.
fn slang_of(input: &str, profile: Profile) -> String {
    let scratch = Scratch::new("slang");
    let expansion = expand(input, &scratch, 1, profile);
    assert!(!failed(&expansion), "{expansion}");
    let files = scratch.files(".slang");
    assert_eq!(files.len(), 1, "{files:?}");
    scratch.read(&files[0])
}

const FREE_1D: &str = "n, [a: &[f32], k: f32], |i| -> f32 { a[i] * k }";
const FREE_2D: &str = "(w, h), [a: &[u32]], |x, y| -> u32 { a[x] + y }";
const IN_PLACE: &str = "&mut data, [data: &mut [f32]], |i| { data[i] = data[i] * 0.5; }";

// ---- the lowering ----

#[test]
fn a_1d_stage_writes_its_tail_to_the_result_buffer_at_its_own_index() {
    let slang = slang_of(FREE_1D, Profile::Other);
    assert!(slang.contains("[numthreads(256, 1, 1)]"), "{slang}");
    // The kernel is `par_for!`'s: the extent record, the guard, the captures, then the result buffer last.
    assert!(slang.contains("struct __GustExtent"), "{slang}");
    assert!(
        slang.contains("RWStructuredBuffer<float> __gust_task_out"),
        "{slang}"
    );
    assert!(
        slang.contains("__gust_task_out[__gust_gid.x] = (a[i] * k);"),
        "{slang}"
    );
    let order = |needle: &str| {
        slang
            .find(needle)
            .unwrap_or_else(|| panic!("{needle} in {slang}"))
    };
    assert!(
        order("StructuredBuffer<float> a") < order("RWStructuredBuffer<float> __gust_task_out")
    );
}

#[test]
fn a_2d_stage_is_a_2d_launch_and_writes_row_major() {
    let slang = slang_of(FREE_2D, Profile::Other);
    assert!(slang.contains("[numthreads(16, 16, 1)]"), "{slang}");
    assert!(
        slang.contains(
            "__gust_task_out[((__gust_gid.y * __gust_ext.__gust_ex) + __gust_gid.x)] = (a[x] + y);"
        ),
        "{slang}"
    );
}

#[test]
fn an_in_place_stage_has_no_result_buffer_and_writes_its_target() {
    let slang = slang_of(IN_PLACE, Profile::Other);
    assert!(!slang.contains("task_out"), "{slang}");
    assert!(slang.contains("RWStructuredBuffer<float> data"), "{slang}");
    assert!(slang.contains("data[i] = (data[i] * 0.5);"), "{slang}");
}

#[test]
fn a_stage_exports_through_the_par_for_path_with_no_new_record_kind() {
    let scratch = Scratch::new("records");
    let expansion = expand(FREE_1D, &scratch, 7, Profile::Other);
    assert!(!failed(&expansion), "{expansion}");
    // One artifact (`.slang`, `.kernel.json`) and one sidecar per call site: exactly the files `par_for!` writes.
    assert_eq!(scratch.files(".slang").len(), 1);
    let kernels = scratch.files(".kernel.json");
    assert_eq!(kernels.len(), 1, "{kernels:?}");
    let sidecars = scratch.files(".site.json");
    assert_eq!(sidecars.len(), 1, "{sidecars:?}");
    let record = scratch.read(&kernels[0]);
    assert!(
        record.contains("\"schema\": 2") || record.contains("\"schema\":2"),
        "{record}"
    );
    assert!(!record.contains("\"source\""), "{record}");
    let sidecar = scratch.read(&sidecars[0]);
    assert!(sidecar.contains('7'), "{sidecar}");
}

#[test]
fn two_calls_of_one_body_are_one_artifact_and_two_sidecars() {
    let scratch = Scratch::new("identity");
    let first = expand(FREE_1D, &scratch, 3, Profile::Other);
    let spelled = "n , [ a : &[float] , k : float ] , | i | -> float { a [ i ] * k }";
    let second = expand(spelled, &scratch, 9, Profile::Other);
    assert_eq!(module_name(&first), module_name(&second));
    assert_eq!(scratch.files(".kernel.json").len(), 1);
    assert_eq!(scratch.files(".site.json").len(), 2);
}

#[test]
fn bodies_that_lower_differently_get_different_module_names() {
    let scratch = Scratch::new("different");
    let one = expand(FREE_1D, &scratch, 1, Profile::Other);
    let other = expand(
        "n, [a: &[f32], k: f32], |i| -> f32 { a[i] + k }",
        &scratch,
        1,
        Profile::Other,
    );
    assert_ne!(module_name(&one), module_name(&other));
}

// ---- the host side ----

#[test]
fn the_host_function_builds_the_call_and_submits_through_the_one_runtime_entry_point() {
    let scratch = Scratch::new("host");
    let text = expand(FREE_1D, &scratch, 1, Profile::Other);
    // The closure builds the `Call` and `submit_and_read` makes every submission: the host function names neither
    // a submission (`submit_extent`, `submit_call`, `dispatch_extent`; the kernel's module defines them, the host
    // function must not call them), nor the read-back, nor a wait.
    let host = &text[text.find("fn __gust_task").expect(&text)..];
    assert!(
        host.contains(":: gust :: task :: submit_and_read"),
        "{host}"
    );
    assert!(host.contains("run :: call_extent"), "{host}");
    for submission in [
        "submit_extent",
        "submit_call",
        "dispatch_extent",
        "dispatch_call",
        "readback",
        "to_vec_async",
    ] {
        assert!(!host.contains(submission), "{submission} in {host}");
    }
    // A task runs on `wasm32` too: it has no `compile_error!` behind that target, unlike `par_for!`.
    assert!(!text.contains("blocks until the device is done"), "{text}");
    assert!(text.contains(":: gust :: task :: Task :: new"), "{text}");
}

/// The result buffer has a reserved name no capture can have, so a capture called `task_out` is an ordinary one.
#[test]
fn a_capture_named_task_out_does_not_collide_with_the_result_buffer() {
    let slang = slang_of(
        "n, [task_out: &[f32]], |i| -> f32 { task_out[i] }",
        Profile::Other,
    );
    assert!(
        slang.contains("StructuredBuffer<float> task_out"),
        "{slang}"
    );
    assert!(
        slang.contains("RWStructuredBuffer<float> __gust_task_out"),
        "{slang}"
    );
}

#[test]
fn the_cpu_profile_names_no_gpu_api_and_the_other_profile_names_the_default_device() {
    let scratch = Scratch::new("profiles");
    let cpu = expand(FREE_1D, &scratch, 1, Profile::Cpu);
    assert!(!failed(&cpu), "{cpu}");
    assert!(!cpu.contains("gust_wgpu"), "{cpu}");
    assert!(cpu.contains("CpuDevice :: new ()"), "{cpu}");
    let other = expand(FREE_1D, &scratch, 1, Profile::Other);
    assert!(other.contains("gust_wgpu :: default_device"), "{other}");
}

#[test]
fn the_on_clause_names_the_device_and_the_default_is_not_opened() {
    let scratch = Scratch::new("on");
    let text = expand(
        "on(&device); n, [a: &[f32], k: f32], |i| -> f32 { a[i] * k }",
        &scratch,
        1,
        Profile::Other,
    );
    assert!(!failed(&text), "{text}");
    assert!(text.contains("let __gust_given = & device ;"), "{text}");
    assert!(!text.contains("default_device"), "{text}");
}

#[test]
fn a_stage_with_no_captures_needs_no_list() {
    let scratch = Scratch::new("nolist");
    let text = expand("n, |i| -> u32 { i * 2u32 }", &scratch, 1, Profile::Other);
    assert!(!failed(&text), "{text}");
}

// ---- the diagnostics ----

fn rejected(input: &str, code: Code) {
    let scratch = Scratch::new("rejected");
    let text = expand(input, &scratch, 1, Profile::Other);
    assert!(failed(&text), "{input}\n{text}");
    assert_rendered_code(&text, code);
    // One mistake is one error: the value that follows is a task of the right type, and nothing was exported.
    assert_eq!(text.matches("compile_error").count(), 1, "{text}");
    assert!(scratch.files(".slang").is_empty(), "{input}");
}

#[test]
fn a_call_that_is_not_one_of_the_forms_is_e1001() {
    for input in [
        // no closure
        "n, [a: &[f32]], 5",
        // three parameters
        "n, |a, b, c| -> u32 { a }",
        // a pattern that is not an identifier
        "n, |(a, b)| -> u32 { a }",
        // no result type on a free stage
        "n, [a: &[f32]], |i| { a[i]; }",
        // a result type on an in-place stage
        "&mut data, [data: &mut [f32]], |i| -> f32 { data[i] }",
        // a 2-D in-place stage
        "&mut data, [data: &mut [f32]], |x, y| { data[x] = 1.0; }",
        // a tuple shape for a 1-D stage, a literal for a 2-D one, a triple for a 2-D one
        "(a, b), |i| -> u32 { i }",
        "4, |x, y| -> u32 { x }",
        "(a, b, c), |x, y| -> u32 { x }",
        // the body does not end in its result
        "n, |i| -> u32 { let a = i; }",
        // a `move` closure
        "n, move |i| -> u32 { i }",
        // a resident capture
        "n, [a: &GpuVec<f32>], |i| -> f32 { a[i] }",
        // an in-place target that is not a variable
        "&mut self.data, [data: &mut [f32]], |i| { data[i] = 1.0; }",
        // something after the closure
        "n, |i| -> u32 { i } extra",
        // the hidden form of `par_for!`
        "descriptor; n, |i| -> u32 { i }",
        // no shape
        "|i| -> u32 { i }",
    ] {
        rejected(input, Code::TaskForm);
    }
}

#[test]
fn a_capture_whose_element_type_the_macro_cannot_see_is_e1002_and_names_the_entry() {
    for (input, entry) in [
        ("n, [a: &Vec<f32>], |i| -> f32 { a[i] }", "`a: &[f32]`"),
        ("n, [a: &[Foo]], |i| -> f32 { 1.0 }", "`a: &[f32]`"),
        ("n, [a: &[bool]], |i| -> u32 { 1u32 }", "`a: &[u32]`"),
        ("n, [a: f64], |i| -> f32 { 1.0 }", "`a: &[f32]`"),
        ("n, [a: Vec<u32>], |i| -> f32 { 1.0 }", "`a: &[f32]`"),
        // a name of the body that the list does not have: the entry to write is named, for a slice and a scalar
        ("n, |i| -> f32 { a[i] }", "`a: &[f32]`"),
        ("n, [k: f32], |i| -> f32 { a[i] * k }", "`a: &[f32]`"),
        (
            "n, [a: &[f32]], |i| -> f32 { a[i] * scale }",
            "`scale: f32`",
        ),
        ("n, |i| -> u32 { n }", "`n: f32`"),
        (
            "&mut data, [data: &mut [f32]], |i| { data[i] = data[i] * k; }",
            "`k: f32`",
        ),
        // the form follows the use: an index use never suggests a scalar, any other use never a slice
        ("n, |i| -> f32 { (a)[i] }", "`a: &[f32]`"),
        ("n, |i| -> f32 { a[i].x }", "`a: &[f32]`"),
        // an in-place target with no entry, and one whose entry is read-only
        ("&mut data, |i| { data[i] = 1.0; }", "`data: &mut [f32]`"),
        (
            "&mut data, [data: &[f32]], |i| { data[i] = 1.0; }",
            "`data: &mut [T]`",
        ),
    ] {
        rejected(input, Code::TaskCapture);
        let scratch = Scratch::new("entry");
        let text = expand(input, &scratch, 1, Profile::Other);
        assert!(text.contains(entry), "{entry} not in {text}");
    }
    // A name the body uses that the list lacks names one form: the slice for an index use, the scalar otherwise.
    for (input, name, scalar) in [
        ("n, |i| -> f32 { a[i] }", "a", false),
        ("n, [k: f32], |i| -> f32 { a[i] * k }", "a", false),
        ("n, [a: &[f32]], |i| -> f32 { a[i] * scale }", "scale", true),
        ("n, |i| -> u32 { n }", "n", true),
        (
            "&mut data, [data: &mut [f32]], |i| { data[i] = data[i] * k; }",
            "k",
            true,
        ),
    ] {
        let scratch = Scratch::new("form");
        let text = expand(input, &scratch, 1, Profile::Other);
        let (this, other) = if scalar {
            (format!("`{name}: f32`"), format!("`{name}: &[f32]`"))
        } else {
            (format!("`{name}: &[f32]`"), format!("`{name}: f32`"))
        };
        assert!(text.contains(&this), "{this} not in {text}");
        assert!(!text.contains(&other), "{other} in {text}");
    }
}

/// Names bound in the body, parameters, list entries and callees are not missing entries.
#[test]
fn names_the_body_binds_are_not_missing_list_entries() {
    let scratch = Scratch::new("bound");
    for input in [
        "n, [a: &[f32], k: f32], |i| -> f32 { let v = a[i]; let w = v * k; w + v }",
        "n, [a: &[f32]], |i| -> f32 { let mut s = 0.0; for j in 0u32..4u32 { s = s + a[j]; } s }",
        "(w, h), [a: &[u32]], |x, y| -> u32 { a[x] + y }",
        "n, [a: &[u32], k: u32], |_| -> u32 { k }",
    ] {
        let text = expand(input, &scratch, 1, Profile::Other);
        assert!(!failed(&text), "{input}\n{text}");
    }
    // The shapes that bind or name something without being a list entry: whatever else the dialect says of them, it is
    // not E1002.
    for input in [
        // a nested closure parameter
        "n, [a: &[f32]], |i| -> f32 { let f = |t: f32| t + 1.0; f(a[i]) }",
        // a `match` binding
        "n, [a: &[f32]], |i| -> f32 { match i { 0u32 => 1.0, k => a[k] } }",
        // a prelude call
        "n, [a: &[f32]], |i| -> f32 { subgroup_add(a[i]) }",
        "n, [a: &[f32]], |i| -> f32 { abs(a[i]) }",
        // a method receiver (a parameter and a local)
        "n, [a: &[f32]], |i| -> f32 { let v = a[i]; v.abs() }",
        "n, [a: &[f32]], |i| -> u32 { i.min(3u32) }",
        // struct shorthand
        "n, [a: &[f32]], |i| -> Vec2 { let x = a[i]; let y = 1.0; Vec2 { x, y } }",
    ] {
        let text = expand(input, &scratch, 1, Profile::Other);
        assert!(!text.contains("GUST-E1002"), "{input}\n{text}");
    }
}

#[test]
fn a_write_to_a_capture_is_e1003() {
    for input in [
        // an assignment, a compound assignment and a write through a field or a nested index
        "n, [a: &[f32]], |i| -> f32 { a[i] = 1.0; a[i] }",
        "n, [a: &[f32]], |i| -> f32 { a[i] += 1.0; a[i] }",
        "n, [a: &[Vec2]], |i| -> f32 { a[i].x = 1.0; a[i].x }",
        "n, [k: f32], |i| -> f32 { k = 2.0; k }",
        // `&mut` on anything but the target, in a free stage and in an in-place one
        "n, [a: &[f32]], |i| -> f32 { let r = &mut a[i]; *r }",
        "&mut data, [data: &mut [f32], other: &[f32]], |i| { let r = &mut other[i]; data[i] = *r; }",
        // a `&mut [T]` host slice that is not the target
        "n, [a: &mut [f32]], |i| -> f32 { a[i] }",
        "&mut data, [data: &mut [f32], b: &mut [f32]], |i| { data[i] = b[i]; }",
    ] {
        rejected(input, Code::TaskWrite);
    }
}

#[test]
fn the_in_place_stage_may_write_its_own_target() {
    let scratch = Scratch::new("target");
    for input in [
        IN_PLACE,
        "&mut data, [data: &mut [f32]], |i| { data[i] += 1.0; }",
        "&mut data, [data: &mut [f32], k: f32], |i| { let v = data[i]; data[i] = v * k; }",
    ] {
        let text = expand(input, &scratch, 1, Profile::Other);
        assert!(!failed(&text), "{input}\n{text}");
    }
}

#[test]
fn an_index_of_the_wrong_rank_is_e1007() {
    for input in [
        // a tuple index of a host slice, in a 1-D and a 2-D stage, and of the in-place target
        "n, [a: &[f32]], |i| -> f32 { a[(i, i)] }",
        "(w, h), [a: &[f32]], |x, y| -> f32 { a[(x, y)] }",
        "(w, h), [a: &[f32]], |x, y| -> f32 { a[(x, y, 0u32)] }",
        "&mut data, [data: &mut [f32]], |i| { data[(i, i)] = 1.0; }",
    ] {
        rejected(input, Code::TaskIndexRank);
    }
}

#[test]
fn a_read_of_the_in_place_target_at_another_index_is_e1008() {
    for input in [
        "&mut data, [data: &mut [f32]], |i| { data[i] = data[i + 1u32]; }",
        "&mut data, [data: &mut [f32]], |i| { data[i + 1u32] = 1.0; }",
        "&mut data, [data: &mut [f32]], |i| { data[0u32] = data[i]; }",
        "&mut data, [data: &mut [f32], p: &[u32]], |i| { data[i] = data[p[i]]; }",
        // the own index bound again: a `let`, a `for`, a closure parameter and a `match` arm of that name hide a shifted
        // index (read or write) behind it, so the two would differ
        "&mut data, [data: &mut [f32]], |i| { let i = i + 1u32; data[i] = data[i] * 0.5; }",
        "&mut data, [data: &mut [f32]], |i| { for i in 0u32..4u32 { data[i] = 0.0; } }",
        "&mut data, [data: &mut [f32]], |i| { let f = |i| i + 1u32; data[f(i)] = 0.0; }",
        "&mut data, [data: &mut [f32]], |i| { match i + 1u32 { i => { data[i] = 0.0; } } }",
        "&mut data, [data: &mut [f32]], |i| { let (i, _) = (i + 1u32, 0u32); data[i] = 0.0; }",
    ] {
        rejected(input, Code::TaskInPlaceRead);
    }
    // A read at another index of a capture that is not the target is an ordinary read.
    let scratch = Scratch::new("neighbour");
    let text = expand(
        "&mut data, [data: &mut [f32], other: &[f32]], |i| { data[i] = other[i + 1u32]; }",
        &scratch,
        1,
        Profile::Other,
    );
    assert!(!failed(&text), "{text}");
    // Another name for the closure's parameter is the same stage, and a new value under another name is not a
    // rebinding of the index.
    for input in [
        "&mut data, [data: &mut [f32]], |k| { data[k] = data[k] * 0.5; }",
        "&mut data, [data: &mut [f32]], |i| { let j = i + 1u32; data[i] = data[i] + 1.0; }",
        "&mut data, [data: &mut [f32]], |i| { for step in 0u32..4u32 { data[i] = data[i] + 1.0; } }",
    ] {
        let text = expand(input, &scratch, 1, Profile::Other);
        assert!(!failed(&text), "{input}\n{text}");
    }
}

/// The probe of fix round 2: a name bound to the target expanded cleanly before the rule, so the write and the read
/// through it escaped E1008.
#[test]
fn an_alias_of_the_in_place_target_is_e1008() {
    for input in [
        "&mut data, [data: &mut [f32]], |i| { let d = data; d[i + 1u32] = 0.0; }",
        "&mut data, [data: &mut [f32]], |i| { let d = data; data[i] = d[i + 1u32]; }",
        "&mut data, [data: &mut [f32]], |i| { let d = (data); d[i + 1u32] = 0.0; }",
        "&mut data, [data: &mut [f32], o: &[f32]], |i| { let d = o; data[i] = d[i]; let e = data; e[i] = 0.0; }",
        "&mut data, [data: &mut [f32]], |i| { match data { d => { d[i + 1u32] = 0.0; } } }",
        "&mut data, [data: &mut [f32]], |i| { if let d = data { d[i + 1u32] = 0.0; } }",
    ] {
        rejected(input, Code::TaskInPlaceRead);
    }
    // A free stage has no target to hide, and a copy of a value is not an alias.
    let scratch = Scratch::new("alias");
    for input in [
        "n, [a: &[f32]], |i| -> f32 { let d = a; d[i + 1u32] }",
        "&mut data, [data: &mut [f32]], |i| { let v = data[i]; data[i] = v + 1.0; }",
    ] {
        let text = expand(input, &scratch, 1, Profile::Other);
        assert!(!failed(&text), "{input}\n{text}");
    }
}

#[test]
fn a_dialect_error_in_the_body_keeps_its_own_code() {
    // A call to a function outside the module is E0302, as in `par_for!`: the stage is a kernel like any other.
    let scratch = Scratch::new("body");
    let text = expand("n, |i| -> u32 { outside(i) }", &scratch, 1, Profile::Other);
    assert!(failed(&text), "{text}");
    assert_rendered_code(&text, Code::UnknownCall);
}
