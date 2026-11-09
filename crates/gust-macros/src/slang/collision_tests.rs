//! Collision-only identifier renaming (M6 plan 2.5): the table, the pure rename function, the six name positions, the
//! reserved prefixes and the goldens. These are Unit-tier tests (no slangc); the Slangc-tier table test in
//! `crates/gust/tests/diagnostics.rs` compiles the same table.

use std::collections::BTreeSet;

use super::{
    collisions::{
        COLLISIONS, MAPPED_TYPE_NAMES, MEASURED_CLASS1, PREFIX, Position, collides, emit_name,
    },
    emit_ident, emit_kernel,
};
use crate::{
    diagnostics::{Code, testing::assert_code},
    expand::kernel_options,
    validate::validate_module,
};

/// The emitted Slang of every kernel of a module source, in declaration order.
fn emit(source: &str) -> syn::Result<String> {
    let module: syn::ItemMod = crate::diagnostics::testing::parse_str(source)?;
    validate_module(&module)?;
    let (_, items) = module.content.as_ref().expect("inline module");
    let mut text = String::new();
    for item in items {
        let syn::Item::Fn(function) = item else {
            continue;
        };
        let Some(options) = kernel_options(function)? else {
            continue;
        };
        text.push_str(&emit_kernel(&module, function, options)?.source);
    }
    Ok(text)
}

/// A module that declares `name` in `position`, written as a raw identifier so that a Rust keyword is spellable.
fn rust_source(position: Position, name: &str) -> String {
    let n = format!("r#{name}");
    let body = match position {
        Position::Local => format!(
            "#[kernel] pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) {{ \
             let {n} = 1.0; out[id.x] = {n}; }}"
        ),
        Position::Field => format!(
            "pub struct S {{ pub {n}: float }} \
             #[kernel] pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<S>) {{ \
             let s = S {{ {n}: 1.0 }}; out[id.x] = s; out[id.x].{n} = 2.0; let back = out[id.x].{n}; }}"
        ),
        Position::Helper => format!(
            "fn {n}(v: float) -> float {{ v + 1.0 }} \
             #[kernel] pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) {{ \
             out[id.x] = {n}(2.0); }}"
        ),
        Position::Struct => format!(
            "pub struct {n} {{ pub v: float }} \
             #[kernel] pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<{n}>) {{ \
             let s = {n} {{ v: 1.0 }}; out[id.x] = s; }}"
        ),
        Position::Global => format!(
            "#[kernel] pub fn run(id: SV_DispatchThreadID, mut {n}: RWStructuredBuffer<float>) {{ \
             {n}[id.x] = 1.0; let k = {n}.len(); }}"
        ),
        Position::HelperParameter => format!(
            "fn f({n}: float) -> float {{ {n} + 1.0 }} \
             #[kernel] pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) {{ \
             out[id.x] = f(2.0); }}"
        ),
    };
    format!("mod table {{ {body} }}")
}

/// The pieces of the emitted text that must carry `emitted` (the Slang spelling) at `position`.
fn expected_pieces(position: Position, emitted: &str) -> Vec<String> {
    let e = emitted;
    match position {
        Position::Local => vec![format!("var {e} = 1.0;"), format!("out[id.x] = {e};")],
        Position::Field => vec![
            format!("    float {e};"),
            format!(".{e} = 1.0;"),
            format!("out[id.x].{e} = 2.0;"),
            format!("var back = out[id.x].{e};"),
        ],
        Position::Helper => vec![
            format!("float {e}(float v)"),
            format!("out[id.x] = {e}(2.0);"),
        ],
        Position::Struct => vec![
            format!("struct {e}\n"),
            format!("RWStructuredBuffer<{e}> out;"),
            format!("    {e} __gust_struct_0;"),
        ],
        Position::Global => vec![
            format!("RWStructuredBuffer<float> {e};"),
            format!("{e}[id.x] = 1.0;"),
            format!("{e}.GetDimensions(__gpu_len_{e}, __gpu_stride_{e});"),
            format!("var k = __gpu_len_{e};"),
        ],
        Position::HelperParameter => vec![
            format!("float f(float {e})"),
            format!("return ({e} + 1.0);"),
        ],
    }
}

/// Whether a user can write `name` at `position` and reach the renaming: E0901 denies `__Gust` names, and a struct
/// named like a type the macro maps itself (`float`, `int`, `uint`, `bool`) is read as that type.
fn reachable(position: Position, name: &str) -> bool {
    !name.starts_with("__Gust")
        && !(position == Position::Struct && MAPPED_TYPE_NAMES.contains(&name))
}

#[test]
fn the_table_is_sorted_and_without_duplicates() {
    for pair in COLLISIONS.windows(2) {
        assert!(
            pair[0].name < pair[1].name,
            "`{}` must sort before `{}` (byte order, no duplicates)",
            pair[0].name,
            pair[1].name
        );
    }
}

#[test]
fn every_entry_has_a_reason_and_a_clean_spelling() {
    for entry in COLLISIONS {
        assert!(
            !entry.fails_in.is_empty() || entry.emitted,
            "`{}` is in the table for no reason",
            entry.name
        );
        let positions: BTreeSet<_> = entry.fails_in.iter().map(|p| p.probe_label()).collect();
        assert_eq!(
            positions.len(),
            entry.fails_in.len(),
            "`{}` lists a position twice",
            entry.name
        );
        assert!(
            entry
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "`{}` is not an identifier",
            entry.name
        );
        assert!(
            !entry.name.starts_with("__gust_") && !entry.name.starts_with("__gpu_"),
            "`{}` is a reserved generated name, never a user name",
            entry.name
        );
    }
}

/// The measurement is the ground truth for class 1: dropping an entry, or recording a different position set, fails
/// here before any slangc runs (the Slangc-tier test then re-measures every row).
#[test]
fn the_table_holds_exactly_the_measured_class_1_record() {
    let class_1: Vec<_> = COLLISIONS
        .iter()
        .filter(|entry| !entry.fails_in.is_empty())
        .collect();
    assert_eq!(class_1.len(), MEASURED_CLASS1.len());
    for (name, labels) in MEASURED_CLASS1 {
        let entry = COLLISIONS
            .iter()
            .find(|entry| entry.name == *name)
            .unwrap_or_else(|| panic!("the measured class-1 name `{name}` is not in the table"));
        let mut recorded: Vec<_> = entry.fails_in.iter().map(|p| p.probe_label()).collect();
        let mut measured = labels.to_vec();
        recorded.sort_unstable();
        measured.sort_unstable();
        assert_eq!(recorded, measured, "positions of `{name}`");
    }
}

#[test]
fn the_class_2_entries_are_the_names_the_emitted_text_uses() {
    let emitted: BTreeSet<_> = COLLISIONS
        .iter()
        .filter(|entry| entry.emitted)
        .map(|entry| entry.name)
        .collect();
    // R1-A.4 and S6.3b, plus the measured additions (`Atomic`, `none`, and `true`, `false` and `var`, which the review
    // of 2026-11-01 found captured by a local of that name).
    let expected: BTreeSet<_> = [
        "Atomic",
        "ConstantBuffer",
        "Optional",
        "RWStructuredBuffer",
        "StructuredBuffer",
        "__GustResult",
        "bool",
        "dot",
        "false",
        "float",
        "float2",
        "float2x2",
        "float3",
        "float3x3",
        "float4",
        "float4x4",
        "int",
        "int2",
        "int4",
        "matrix",
        "mul",
        "none",
        "this",
        "transpose",
        "true",
        "uint",
        "uint2",
        "uint3",
        "uint4",
        "var",
        "vector",
    ]
    .into_iter()
    .collect();
    assert_eq!(emitted, expected);
    // `int3` and `bool2` are in no class: M6 emits neither (R1-A.4).
    assert!(!collides("int3") && !collides("bool2"));
}

#[test]
fn emit_name_renames_only_table_names_and_is_injective() {
    assert_eq!(emit_name("new"), "__gust_r_new");
    assert_eq!(emit_name("uint3"), "__gust_r_uint3");
    assert_eq!(emit_name("id"), "id");
    assert_eq!(emit_name("match"), "match");
    assert_eq!(emit_name("int3"), "int3");
    assert_eq!(emit_name("New"), "New", "the table is case sensitive");
    let plain = [
        "id", "i", "tmp", "match", "loop", "else", "fn", "int3", "bool2", "New",
    ];
    let mut outputs = BTreeSet::new();
    for entry in COLLISIONS {
        let out = emit_name(entry.name);
        assert_eq!(out, format!("{PREFIX}{}", entry.name));
        // The inverse is a text rule: delete the prefix.
        assert_eq!(out.replace(PREFIX, ""), entry.name);
        assert!(outputs.insert(out.into_owned()));
    }
    for name in plain {
        assert!(outputs.insert(emit_name(name).into_owned()), "{name}");
    }
    // No output is itself a table name, so renaming is not applied twice by a second pass over the emitted text.
    assert!(outputs.iter().all(|out| !collides(out)));
}

#[test]
fn raw_identifiers_are_stripped_and_renamed_by_their_unrawed_text() {
    let ident =
        |text: &str| -> syn::Ident { crate::diagnostics::testing::parse_str(text).unwrap() };
    assert_eq!(emit_ident(&ident("r#new")), "__gust_r_new");
    assert_eq!(emit_ident(&ident("new")), "__gust_r_new");
    assert_eq!(emit_ident(&ident("r#if")), "__gust_r_if");
    assert_eq!(emit_ident(&ident("r#match")), "match");
    assert_eq!(emit_ident(&ident("r#else")), "else");
    assert_eq!(emit_ident(&ident("r#loop")), "loop");
    assert_eq!(emit_ident(&ident("r#plain")), "plain");
}

/// `r#true` and `r#false` are variables to every literal GUST writes after them: the emitted text must carry the
/// renamed declaration and the bare literals, so the literals keep their meaning (slangc takes the unrenamed program
/// and stores the other value, so the Slangc-tier capture test pins the generated code as well).
#[test]
fn a_local_named_true_or_false_is_renamed_so_the_literals_keep_their_meaning() {
    let text = emit(
        "mod m { #[kernel] pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) { \
         let r#true = false; let r#false = true; let x = true; let y = false; \
         if x && !y { out[id.x] = 1; } } }",
    )
    .unwrap();
    for piece in [
        "var __gust_r_true = false;",
        "var __gust_r_false = true;",
        "var x = true;",
        "var y = false;",
    ] {
        assert!(text.contains(piece), "missing `{piece}` in\n{text}");
    }
    assert!(
        !text.contains("var true") && !text.contains("var false"),
        "{text}"
    );
    // And `var`, which every emitted local starts with.
    let text = emit(
        "mod m { #[kernel] pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) { \
         let var = 1u32; let after = 2u32; out[id.x] = var + after; } }",
    )
    .unwrap();
    assert!(text.contains("var __gust_r_var = uint(1);"), "{text}");
    assert!(text.contains("var after = uint(2);"), "{text}");
}

/// The invocation parameter and a `ConstantBuffer` parameter are name sites of their own in `emit_kernel` (the
/// invocation name is used for the entry point's parameter and every `id.x`; the uniform is a global).
#[test]
fn the_invocation_and_uniform_parameters_are_renamed_like_every_other_name() {
    let text = emit(
        "mod m { pub struct Gain { pub v: uint, pub a: uint, pub b: uint, pub c: uint } \
         #[kernel] pub fn run(new: SV_DispatchThreadID, class: ConstantBuffer<Gain>, \
         mut This: RWStructuredBuffer<uint>) { This[new.x] = class.v; } }",
    )
    .unwrap();
    for piece in [
        "ConstantBuffer<Gain> __gust_r_class;",
        "RWStructuredBuffer<uint> __gust_r_This;",
        "uint3 __gust_r_new : SV_DispatchThreadID",
        "__gust_r_This[__gust_r_new.x] = __gust_r_class.v;",
    ] {
        assert!(text.contains(piece), "missing `{piece}` in\n{text}");
    }
    for bare in [" new", "(new", " class;", " This;", "[new."] {
        assert!(!text.contains(bare), "unrenamed `{bare}` in\n{text}");
    }
}

/// One name site per position, for every entry: the renamed spelling appears at the declaration and at every use.
#[test]
fn every_entry_is_renamed_at_every_name_site_of_every_position() {
    let mut checked = 0;
    for entry in COLLISIONS {
        for position in Position::ALL {
            if !reachable(position, entry.name) {
                continue;
            }
            let source = rust_source(position, entry.name);
            let text = emit(&source).unwrap_or_else(|error| {
                panic!("{} / {}: {error}\n{source}", entry.name, position.label())
            });
            let renamed = format!("{PREFIX}{}", entry.name);
            for piece in expected_pieces(position, &renamed) {
                assert!(
                    text.contains(&piece),
                    "`{}` at {}: missing `{piece}` in\n{text}",
                    entry.name,
                    position.label()
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 600, "{checked}");
}

/// A name outside the table keeps its spelling at every site, so nothing is renamed that does not need it.
#[test]
fn a_name_outside_the_table_is_emitted_unchanged_at_every_site() {
    for name in ["tmp", "match", "loop", "else", "int3", "bool2", "idx"] {
        for position in Position::ALL {
            let text = emit(&rust_source(position, name)).unwrap();
            assert!(
                !text.contains(PREFIX),
                "{name} / {}:\n{text}",
                position.label()
            );
            for piece in expected_pieces(position, name) {
                assert!(
                    text.contains(&piece),
                    "{name} / {}: `{piece}` in\n{text}",
                    position.label()
                );
            }
        }
    }
}

#[test]
fn if_let_match_for_and_atomic_names_are_renamed() {
    let text = emit(
        "mod table {
            fn present(v: Option<uint>) -> uint {
                let mut total = 0u32;
                if let Some(r#new) = v {
                    total = total + r#new;
                }
                total
            }
            #[kernel]
            pub fn run(id: SV_DispatchThreadID, mut r#operator: RWStructuredBuffer<uint>, mut other: RWStructuredBuffer<uint>) {
                let settled: Result<uint, uint> = Ok(2u32);
                let mut total = 0u32;
                match settled {
                    Ok(r#class) => {
                        total = total + r#class;
                    }
                    Err(r#switch) => {
                        total = total + r#switch;
                    }
                };
                for r#default in 0u32..3u32 {
                    atomic_add(&mut r#operator[id.x], r#default);
                }
                other[id.x] = total + present(Some(1u32));
            }
        }",
    )
    .unwrap();
    for piece in [
        "var __gust_r_new = __gust_opt_",
        "total = (total + __gust_r_new);",
        "var __gust_r_class = __gust_match_",
        "var __gust_r_switch = __gust_match_",
        "total = (total + __gust_r_class);",
        "for (var __gust_r_default = ",
        "__gust_r_default < __gust_end_",
        "__gust_r_default++",
        "RWStructuredBuffer<Atomic<uint>> __gust_r_operator;",
        "__gust_r_operator[id.x].add(__gust_r_default);",
    ] {
        assert!(
            text.contains(piece),
            "missing `{piece}` in
{text}"
        );
    }
}

#[test]
fn reserved_prefixes_are_rejected_also_when_raw() {
    let local = |name: &str| {
        format!(
            "mod table {{ #[kernel] pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) {{ \
             let {name} = 1.0; out[id.x] = {name}; }} }}"
        )
    };
    // `__Gust` (the generated generics and `__GustResult`), the old two prefixes, and a name that would print like
    // the rename of `new`, in raw form too: emission strips `r#`, so the check reads the unrawed text.
    for name in [
        "__GustCount",
        "__Gust",
        "r#__GustX",
        "__gust_count",
        "__gpu_count",
        "r#__gust_r_new",
        "__gust_r_new",
    ] {
        assert_code(&emit(&local(name)).unwrap_err(), Code::ReservedPrefix);
    }
    // Other fields, parameters and types are covered by the same identifier visit.
    let param = "mod table { #[kernel] pub fn run(id: SV_DispatchThreadID, mut r#__GustOut: RWStructuredBuffer<float>) {} }";
    assert_code(&emit(param).unwrap_err(), Code::ReservedPrefix);
    let structure = "mod table { pub struct __GustS { pub v: float } #[kernel] pub fn run(id: SV_DispatchThreadID) {} }";
    assert_code(&emit(structure).unwrap_err(), Code::ReservedPrefix);
    // `_Gust` and `__gust` (no trailing underscore) are not generated spellings.
    assert!(emit(&local("__gustavo")).is_ok());
    assert!(emit(&local("_Gust")).is_ok());
}

/// No rename fires in the goldens of `tests/fixtures`, so `bless` stays unchanged for them. The goldens' own tests
/// compare the emitted text byte for byte; this one names the property, and fails when a name in them starts to be
/// renamed even if someone blesses the new text.
#[test]
fn no_rename_fires_in_the_goldens() {
    let fixtures: [(&str, &str, &str); 11] = [
        (
            "atomics",
            include_str!("../../../../tests/fixtures/atomics.rs"),
            include_str!("../../../../tests/fixtures/atomics.slang"),
        ),
        (
            "coretests",
            include_str!("../../../../tests/fixtures/coretests.rs"),
            include_str!("../../../../tests/fixtures/coretests.slang"),
        ),
        (
            "loops",
            include_str!("../../../../tests/fixtures/loops.rs"),
            include_str!("../../../../tests/fixtures/loops.slang"),
        ),
        (
            "mismatch",
            include_str!("../../../../tests/fixtures/mismatch.rs"),
            include_str!("../../../../tests/fixtures/mismatch.slang"),
        ),
        (
            "numeric",
            include_str!("../../../../tests/fixtures/numeric.rs"),
            include_str!("../../../../tests/fixtures/numeric.slang"),
        ),
        (
            "option",
            include_str!("../../../../tests/fixtures/option.rs"),
            include_str!("../../../../tests/fixtures/option.slang"),
        ),
        (
            "result",
            include_str!("../../../../tests/fixtures/result.rs"),
            include_str!("../../../../tests/fixtures/result.slang"),
        ),
        (
            "products",
            include_str!("../../../../tests/fixtures/products.rs"),
            include_str!("../../../../tests/fixtures/products.slang"),
        ),
        (
            "semantics",
            include_str!("../../../../tests/fixtures/semantics.rs"),
            include_str!("../../../../tests/fixtures/semantics.slang"),
        ),
        (
            "uniforms",
            include_str!("../../../../tests/fixtures/uniforms.rs"),
            include_str!("../../../../tests/fixtures/uniforms.slang"),
        ),
        (
            "vectors",
            include_str!("../../../../tests/fixtures/vectors.rs"),
            include_str!("../../../../tests/fixtures/vectors.slang"),
        ),
    ];
    for (name, rust, golden) in fixtures {
        let emitted = emit(rust).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert!(
            emitted.len() > 200,
            "{name}: the fixture emitted no kernel text"
        );
        assert!(
            !emitted.contains(PREFIX),
            "{name}: a rename fired:\n{emitted}"
        );
        assert!(
            !golden.contains(PREFIX),
            "{name}: the golden holds a rename"
        );
    }
}

/// Descriptors carry the emitted names (R1-A.3): `D17` compares them with the names slangc reflects, which are the
/// compiled text's. `rust_type` keeps the Rust spelling.
#[test]
fn descriptor_names_are_the_emitted_names() {
    use crate::expand::expand_with_directories;

    let module: syn::ItemMod = crate::diagnostics::testing::parse_str(
        "mod table {
            pub struct S { pub r#new: float, pub plain: float }
            #[kernel]
            pub fn run(id: SV_DispatchThreadID, r#class: StructuredBuffer<S>, mut r#float: RWStructuredBuffer<S>) {
                r#float[id.x] = r#class[id.x];
            }
        }",
    )
    .unwrap();
    let text = expand_with_directories(module, None, None, None)
        .unwrap()
        .to_string();
    for piece in [
        "name : \"__gust_r_class\" , rust_type : \"StructuredBuffer < S >\"",
        "name : \"__gust_r_float\" , rust_type : \"RWStructuredBuffer < S >\"",
        "name : \"id\" , rust_type : \"SV_DispatchThreadID\"",
        "name : \"__gust_r_new\" , offset :",
        "name : \"plain\" , offset :",
    ] {
        assert!(text.contains(piece), "missing `{piece}` in\n{text}");
    }
    assert!(!text.contains("name : \"class\""), "{text}");
    assert!(!text.contains("name : \"r#"), "{text}");
}
