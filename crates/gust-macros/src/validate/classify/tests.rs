//! The shape classifier (mx.c): each rule of plan 5.1 has a kernel that keeps `pixel` and a kernel that loses it, the
//! pin `#[kernel(shape = ..)]` is checked against the classification with the code of the construct, and the kernels
//! the macro itself writes (the map generator's, `par_for!`'s) are `grid`-only.

use proc_macro2::TokenStream;

use super::*;
use crate::{
    diagnostics::testing::{assert_code, parse_str},
    expand::{Environment, Profile, expand, kernel_attribute_facts, map_kernel},
    par_for::par_for_in,
    profile::{
        BindingRules, BodyChecks, Builtin, ContractTier, ExtraBinding, GRID, KernelFacts,
        OptionRules, OwnIndexWrite, ReflectionKind, StageProfile, TwinHarness,
    },
    validate::{validate_module, validate_module_in},
};

// ---------------------------------------------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------------------------------------------

fn module(source: &str) -> syn::ItemMod {
    parse_str(source).unwrap()
}

fn kernel_of(module: &syn::ItemMod) -> &syn::ItemFn {
    module
        .content
        .as_ref()
        .unwrap()
        .1
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(function)
                if function
                    .attrs
                    .iter()
                    .any(|attribute| attribute.path().is_ident("kernel")) =>
            {
                Some(function)
            }
            _ => None,
        })
        .expect("a kernel")
}

/// The classification of `function`, a kernel of the `grid` profile.
fn classify_function(function: &syn::ItemFn) -> Classification {
    let (_, _, facts) = kernel_attribute_facts(&GRID, function).unwrap().unwrap();
    classify(&GRID, function, &facts)
}

/// The classification of the kernel of `source`, a module the validator accepts.
fn classification(source: &str) -> Classification {
    let module = module(source);
    validate_module(&module).unwrap_or_else(|error| panic!("{error}\n{source}"));
    classify_function(kernel_of(&module))
}

/// A kernel `k` with the attribute `attribute`, the parameters `parameters` and the body `body`.
fn kernel(attribute: &str, parameters: &str, body: &str) -> String {
    format!("mod m {{ {attribute} pub fn k({parameters}) {{ {body} }} }}")
}

const PARAMETERS: &str =
    "id: SV_DispatchThreadID, a: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>";

/// The kernel with the standard parameters (an input and one output) and the body `body`.
fn body(body: &str) -> String {
    kernel("#[kernel]", PARAMETERS, body)
}

/// The kernel keeps both shapes.
#[track_caller]
fn assert_both(source: &str) {
    let classification = classification(source);
    assert_eq!(
        classification.shapes,
        [Shape::Grid, Shape::Pixel],
        "{:?}\n{source}",
        classification.exclusions
    );
    assert!(classification.exclusions.is_empty());
}

/// The kernel is `grid`-only because of `why`; returns that exclusion.
#[track_caller]
fn assert_excluded(source: &str, why: Why) -> Exclusion {
    let classification = classification(source);
    assert!(classification.allows(Shape::Grid), "{source}");
    assert!(
        !classification.allows(Shape::Pixel),
        "pixel was not excluded:\n{source}"
    );
    classification
        .exclusions
        .iter()
        .find(|exclusion| exclusion.why == why)
        .unwrap_or_else(|| panic!("no {why:?} in {:?}\n{source}", classification.exclusions))
        .clone()
}

fn without_whitespace(text: impl ToString) -> String {
    text.to_string()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect()
}

// ---------------------------------------------------------------------------------------------------------------
// The rules of plan 5.1.
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn a_kernel_that_writes_each_element_of_one_output_is_grid_and_pixel() {
    assert_both(&body("out[id.x] = a[id.x] * 2.0f32;"));
}

#[test]
fn the_spellings_of_the_own_index_are_the_own_index() {
    for write in [
        "out[id.x] = a[id.x];",
        "out[id.global_id().x] = a[id.global_id().x];",
        "let i = id.x; out[i] = a[i];",
        "let i: u32 = id.x; out[i] = a[i];",
        "let i = id.global_id().x; out[i] = a[i];",
    ] {
        assert_both(&body(write));
    }
}

#[test]
fn the_invocation_is_immutable() {
    let mutable = kernel(
        "#[kernel]",
        "mut id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>",
        "out[id.x] = 1.0f32;",
    );
    let exclusion = assert_excluded(&mutable, Why::MutableInvocation);
    assert_eq!(exclusion.why.code(), Code::ShapeMutableInvocation);
    assert!(exclusion.construct.contains("mut id"), "{exclusion:?}");
    // The invocation is the invocation under any name.
    assert_both(&kernel(
        "#[kernel]",
        "gid: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>",
        "out[gid.x] = 1.0f32;",
    ));
}

#[test]
fn an_assignment_to_the_invocation_or_a_field_of_it_is_found() {
    let scan = |source: &str| {
        let function: syn::ItemFn = parse_str(source).unwrap();
        let mut scan = BodyScan {
            id: "id".to_owned(),
            ..BodyScan::default()
        };
        scan.visit_block(&function.block);
        scan
    };
    let assigned = scan(
        "fn k(mut id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>) { id.x = 3u32; id.y += 1u32; out[id.x] = 1.0f32; }",
    );
    assert_eq!(assigned.id_assignments.len(), 2);
    let untouched = scan(
        "fn k(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>) { out[id.x] = 1.0f32; }",
    );
    assert!(untouched.id_assignments.is_empty());
}

#[test]
fn a_write_at_any_index_but_the_own_one_is_refused() {
    for write in [
        "out[id.x + 1u32] = a[id.x];",
        "out[0u32] = a[id.x];",
        "let j = id.x + 1u32; out[j] = a[id.x];",
        "out[id.y] = a[id.x];",
    ] {
        let exclusion = assert_excluded(&body(write), Why::OwnIndexWrite);
        assert_eq!(exclusion.why.code(), Code::ShapeOwnIndex);
    }
}

#[test]
fn an_alias_of_the_own_index_is_one_immutable_name_bound_once() {
    // Two aliases.
    assert_excluded(
        &body("let i = id.x; let j = id.x; out[i] = a[j];"),
        Why::OwnIndexAlias,
    );
    // A `mut` alias.
    assert_excluded(
        &body("let mut i = id.x; out[i] = a[i];"),
        Why::OwnIndexAlias,
    );
    // The name bound again, by a second `let` and by a loop variable.
    assert_excluded(
        &body("let i = id.x; let i = 3u32; out[i] = a[i];"),
        Why::OwnIndexAlias,
    );
    assert_excluded(
        &body("let i = id.x; for i in 0u32..2u32 { let t = a[i]; } out[i] = 1.0f32;"),
        Why::OwnIndexAlias,
    );
    // A name that merely holds a value derived from the id is not the own index.
    assert_excluded(
        &body("let i = id.x + 0u32; out[i] = a[id.x];"),
        Why::OwnIndexWrite,
    );
    assert_eq!(Why::OwnIndexAlias.code(), Code::ShapeOwnIndex);
}

#[test]
fn an_output_is_only_written_at_the_own_index_and_asked_for_its_length() {
    assert_both(&body("let n = out.len(); out[id.x] = n as f32;"));
    assert_both(&body("if id.x < out.len() { out[id.x] = 1.0f32; }"));

    // A read of an element.
    let read = assert_excluded(
        &body("let old = out[id.x]; out[id.x] = old + 1.0f32;"),
        Why::OutputRead,
    );
    assert_eq!(read.why.code(), Code::ShapeOutputUse);
    assert!(read.construct.contains("out[id.x]"), "{read:?}");

    // A compound assignment.
    let compound = assert_excluded(&body("out[id.x] += a[id.x];"), Why::OutputCompound);
    assert_eq!(compound.why.code(), Code::ShapeOutputUse);

    // One component of an element.
    let component = assert_excluded(
        &kernel(
            "#[kernel]",
            "id: SV_DispatchThreadID, mut out: RWStructuredBuffer<Vec4>",
            "out[id.x].x = 1.0f32;",
        ),
        Why::OutputComponent,
    );
    assert_eq!(component.why.code(), Code::ShapeOutputUse);
}

#[test]
fn a_mutable_buffer_that_is_never_written_is_an_input() {
    let read = kernel(
        "#[kernel]",
        "id: SV_DispatchThreadID, mut scratch: RWStructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>",
        "out[id.x] = scratch[id.x];",
    );
    let exclusion = assert_excluded(&read, Why::OutputRead);
    assert_eq!(exclusion.why.code(), Code::ShapeOutputUse);
    let unused = kernel(
        "#[kernel]",
        "id: SV_DispatchThreadID, mut scratch: RWStructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>",
        "out[id.x] = scratch.len() as f32;",
    );
    let exclusion = assert_excluded(&unused, Why::OutputUnwritten);
    assert_eq!(exclusion.why.code(), Code::ShapeOutputUse);
    assert!(exclusion.construct.contains("scratch"), "{exclusion:?}");
}

#[test]
fn every_output_is_definitely_written() {
    // Both arms, every arm, and the transparent guard in its spellings.
    assert_both(&body(
        "if a[id.x] > 0.0f32 { out[id.x] = 1.0f32; } else { out[id.x] = 2.0f32; }",
    ));
    assert_both(&body(
        "let r: Result<u32, u32> = Ok(1u32); \
         match r { Ok(x) => { out[id.x] = x as f32; } Err(e) => { out[id.x] = e as f32; } };",
    ));
    assert_both(&body("let i = id.x; if i < out.len() { out[i] = 1.0f32; }"));
    assert_both(&body(
        "let i = id.x; if i < out.len() && out.len() > i { out[i] = 1.0f32; }",
    ));
    assert_both(&body(
        "let i = id.x; if out.len() > i { if i < out.len() { out[i] = 1.0f32; } }",
    ));

    // An opaque condition, a loop, an arm without the write, and a guard on something else.
    for write in [
        "if a[id.x] > 0.0f32 { out[id.x] = 1.0f32; }",
        "for t in 0u32..2u32 { out[id.x] = 1.0f32; }",
        "if a[id.x] > 0.0f32 { out[id.x] = 1.0f32; } else { let t = 1.0f32; }",
        "let r: Result<u32, u32> = Ok(1u32); \
         match r { Ok(x) => { out[id.x] = x as f32; } Err(e) => { let t = e; } };",
        "if id.x < a.len() { out[id.x] = 1.0f32; }",
        "if id.x < out.len() && a[id.x] > 0.0f32 { out[id.x] = 1.0f32; }",
        "if id.x <= out.len() { out[id.x] = 1.0f32; }",
    ] {
        let exclusion = assert_excluded(&body(write), Why::ConditionalWrite);
        assert_eq!(exclusion.why.code(), Code::ShapeConditionalWrite);
        assert!(exclusion.detail.contains("out"), "{exclusion:?}");
    }
}

#[test]
fn every_output_of_several_is_definitely_written() {
    let two = |write: &str| {
        kernel(
            "#[kernel]",
            "id: SV_DispatchThreadID, mut first: RWStructuredBuffer<f32>, mut second: RWStructuredBuffer<u32>",
            write,
        )
    };
    assert_both(&two("first[id.x] = 1.0f32; second[id.x] = 2u32;"));
    // One output written on every path does not cover the other.
    let exclusion = assert_excluded(
        &two("first[id.x] = 1.0f32; if id.x == 0u32 { second[id.x] = 2u32; }"),
        Why::ConditionalWrite,
    );
    assert!(exclusion.detail.contains("second"), "{exclusion:?}");
}

#[test]
fn a_conditional_write_names_the_construct_that_makes_it_conditional() {
    let looped = assert_excluded(
        &body("for t in 0u32..2u32 { out[id.x] = 1.0f32; }"),
        Why::ConditionalWrite,
    );
    assert!(looped.construct.starts_with("for t in"), "{looped:?}");
    let guard = assert_excluded(
        &body("if id.x < a.len() { out[id.x] = 1.0f32; }"),
        Why::ConditionalWrite,
    );
    assert!(guard.construct.contains("a.len()"), "{guard:?}");
}

#[test]
fn a_return_in_the_kernel_body_is_refused() {
    let exclusion = assert_excluded(
        &body("if id.x >= out.len() { return; } out[id.x] = 1.0f32;"),
        Why::Return,
    );
    assert_eq!(exclusion.why.code(), Code::ShapeReturn);
    // The same kernel without the early exit.
    assert_both(&body("if id.x < out.len() { out[id.x] = 1.0f32; }"));
}

#[test]
fn a_barrier_workgroup_memory_and_the_other_invocation_ids_are_refused() {
    let source = kernel(
        "#[kernel(workgroup_size(256, 1, 1), cpu = \"unsupported\")]",
        "id: SV_DispatchThreadID, lid: LocalId, gid: GroupId, a: StructuredBuffer<f32>, \
         mut out: RWStructuredBuffer<f32>, mut scratch: Workgroup<[f32; 256]>",
        "scratch[lid.x] = a[id.x]; workgroup_barrier(); out[id.x] = scratch[0u32] + gid.x as f32;",
    );
    let classification = classification(&source);
    let whys: Vec<Why> = classification
        .exclusions
        .iter()
        .map(|exclusion| exclusion.why)
        .collect();
    for expected in [
        Why::WorkgroupSize,
        Why::InvocationBuiltin,
        Why::WorkgroupMemory,
        Why::Barrier,
    ] {
        assert!(whys.contains(&expected), "{expected:?} in {whys:?}");
        assert_eq!(expected.code(), Code::ShapeExcluded);
    }
    let barrier = classification
        .exclusions
        .iter()
        .find(|exclusion| exclusion.why == Why::Barrier)
        .unwrap();
    assert_eq!(barrier.construct, "workgroup_barrier()");

    // Each id alone.
    for (id, ty) in [("lid", "LocalId"), ("gid", "GroupId")] {
        assert_excluded(
            &kernel(
                "#[kernel]",
                &format!("id: SV_DispatchThreadID, {id}: {ty}, mut out: RWStructuredBuffer<f32>"),
                &format!("out[id.x] = {id}.x as f32;"),
            ),
            Why::InvocationBuiltin,
        );
    }
}

#[test]
fn an_atomic_operation_is_refused() {
    let source = kernel(
        "#[kernel]",
        "id: SV_DispatchThreadID, mut counter: RWStructuredBuffer<u32>, mut out: RWStructuredBuffer<f32>",
        "atomic_add(&mut counter[0], 1u32); out[id.x] = 1.0f32;",
    );
    let exclusion = assert_excluded(&source, Why::Atomic);
    assert_eq!(exclusion.why.code(), Code::ShapeExcluded);
    assert!(
        exclusion.construct.starts_with("atomic_add("),
        "{exclusion:?}"
    );
}

// A storage texture is the one texture parameter that rules `pixel` out (mx.t3); the fixture with the kernel is
// `tests/ui/shape-pixel-storage-texture`.
#[test]
fn a_storage_texture_parameter_is_refused() {
    for (ty, expected) in [
        ("StorageTexture2D<R32Float>", true),
        ("WriteTexture2D<Rgba32Float>", true),
        ("Texture2D<f32>", false),
        ("Sampler", false),
        ("f32", false),
        ("StructuredBuffer<f32>", false),
    ] {
        let parsed: syn::Type = parse_str(ty).unwrap();
        let flavor = crate::slang::parameter_flavor(&parsed);
        let refused = matches!(
            flavor,
            crate::slang::ParameterFlavor::StorageTexture
                | crate::slang::ParameterFlavor::WriteTexture
        );
        assert_eq!(refused, expected, "{ty}");
    }
    assert_eq!(Why::StorageTexture.code(), Code::ShapeExcluded);
}

#[test]
fn an_explicit_workgroup_size_is_refused_and_a_default_one_is_not() {
    let sized = kernel(
        "#[kernel(workgroup_size(64, 1, 1))]",
        PARAMETERS,
        "out[id.x] = a[id.x];",
    );
    let exclusion = assert_excluded(&sized, Why::WorkgroupSize);
    assert_eq!(exclusion.why.code(), Code::ShapeExcluded);
    assert_eq!(exclusion.construct, "workgroup_size(64, 1, 1)");
    assert_both(&kernel(
        "#[kernel(cpu = \"unsupported\")]",
        PARAMETERS,
        "out[id.x] = a[id.x];",
    ));
    assert_both(&body("out[id.x] = a[id.x];"));
}

#[test]
fn a_kernel_with_no_output_has_no_pixel() {
    let source = kernel(
        "#[kernel]",
        "id: SV_DispatchThreadID, a: StructuredBuffer<f32>",
        "let x = a[id.x] + 1.0f32;",
    );
    let exclusion = assert_excluded(&source, Why::NoOutput);
    assert_eq!(exclusion.why.code(), Code::ShapeExcluded);
}

#[test]
fn the_outputs_have_colour_target_formats_within_the_bound() {
    for (element, value) in [
        ("u32", "1u32"),
        ("i32", "1i32"),
        ("f32", "1.0f32"),
        ("Vec2", "Vec2::splat(1.0f32)"),
        ("Vec4", "Vec4::splat(1.0f32)"),
        ("IVec2", "IVec2::splat(1i32)"),
        ("IVec4", "IVec4::splat(1i32)"),
        ("UVec2", "UVec2::splat(1u32)"),
        ("UVec4", "UVec4::splat(1u32)"),
    ] {
        assert_both(&kernel(
            "#[kernel]",
            &format!("id: SV_DispatchThreadID, mut out: RWStructuredBuffer<{element}>"),
            &format!("out[id.x] = {value};"),
        ));
    }
    // A struct has no colour-target format.
    let structure = "mod m { pub struct P { pub x: f32, pub y: f32 } #[kernel] \
        pub fn k(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<P>) { \
        out[id.x] = P { x: 1.0f32, y: 2.0f32 }; } }";
    let exclusion = assert_excluded(structure, Why::TargetElement);
    assert_eq!(exclusion.why.code(), Code::ShapeExcluded);
    assert!(exclusion.detail.contains('P'), "{exclusion:?}");
}

#[test]
fn at_most_eight_outputs_and_thirty_two_bytes_of_them() {
    let outputs = |count: usize, element: &str, value: &str| {
        let parameters: Vec<String> = (0..count)
            .map(|k| format!("mut o{k}: RWStructuredBuffer<{element}>"))
            .collect();
        let writes: Vec<String> = (0..count)
            .map(|k| format!("o{k}[id.x] = {value};"))
            .collect();
        kernel(
            "#[kernel]",
            &format!("id: SV_DispatchThreadID, {}", parameters.join(", ")),
            &writes.join(" "),
        )
    };
    assert_both(&outputs(8, "f32", "1.0f32"));
    assert_both(&outputs(2, "Vec4", "Vec4::splat(1.0f32)"));
    let count = assert_excluded(&outputs(9, "f32", "1.0f32"), Why::TargetCount);
    assert_eq!(count.why.code(), Code::ShapeExcluded);
    assert!(count.detail.contains('9'), "{count:?}");
    let bytes = assert_excluded(&outputs(3, "Vec4", "Vec4::splat(1.0f32)"), Why::TargetBytes);
    assert_eq!(bytes.why.code(), Code::ShapeExcluded);
    assert!(bytes.detail.contains("48"), "{bytes:?}");
}

#[test]
fn the_transparent_guard_forms_of_the_difftest_corpus_keep_both_shapes() {
    // The shape of every kernel `tests/difftest` emits, without the explicit workgroup size (see the next test).
    let corpus = |attribute: &str| {
        kernel(
            attribute,
            "id: SV_DispatchThreadID, a: StructuredBuffer<u32>, b: StructuredBuffer<u32>, \
             mut out: RWStructuredBuffer<u32>",
            "let i = id.x; if i < out.len() { let v0: u32 = a[i] + 1u32; \
             for v1 in 0u32..3u32 { let v2: u32 = v0 + v1; } \
             if a[i] > b[i] { let v3: u32 = 1u32; } else { let v3: u32 = 2u32; } \
             if v0 > 4u32 { out[i] = 1u32; } else { out[i] = 0u32; } }",
        )
    };
    assert_both(&corpus("#[kernel]"));
    assert_both(&body(
        "let i = id.x; if i < out.len() { out[i] = a[i] * 2.0f32; }",
    ));
}

#[test]
fn the_difftest_kernels_carry_a_workgroup_size_and_so_are_grid_only() {
    // `tests/difftest/src/emit.rs` writes `#[kernel(workgroup_size(64, 1, 1))]`: an explicit size is the author saying
    // "compute", so the literal rule of plan 5.1 keeps those kernels `grid`-only.
    let source = kernel(
        "#[kernel(workgroup_size(64, 1, 1))]",
        "id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>",
        "let i = id.x; if i < out.len() { out[i] = 1u32; }",
    );
    let classification = classification(&source);
    assert_eq!(classification.shapes, [Shape::Grid]);
    assert_eq!(classification.exclusions.len(), 1);
    assert_eq!(classification.exclusions[0].why, Why::WorkgroupSize);
}

// ---------------------------------------------------------------------------------------------------------------
// The kernels the macro writes.
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn the_map_generators_kernel_is_grid_only_by_the_rules() {
    let function: syn::ItemFn = parse_str("fn add(x: f32, y: f32) -> f32 { x + y }").unwrap();
    let generated = map_kernel(&function, Profile::Other).unwrap();
    let kernel = generated
        .module
        .content
        .as_ref()
        .unwrap()
        .1
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(function) if function.sig.ident == "map_kernel" => Some(function),
            _ => None,
        })
        .unwrap();
    let classification = classify_function(kernel);
    assert_eq!(classification.shapes, [Shape::Grid]);
    let whys: Vec<Why> = classification
        .exclusions
        .iter()
        .map(|exclusion| exclusion.why)
        .collect();
    assert!(whys.contains(&Why::WorkgroupSize), "{whys:?}");
    // The write is at the grid-stride index, not at the own one.
    assert!(whys.contains(&Why::OwnIndexWrite), "{whys:?}");

    // And the descriptor the expansion gives it says so.
    let item: TokenStream = "#[gpu] fn add(x: f32, y: f32) -> f32 { x + y }"
        .parse()
        .unwrap();
    let expansion = without_whitespace(crate::gpu_impl_with(
        TokenStream::new(),
        item,
        Profile::Other,
    ));
    assert!(
        expansion.contains("compatible_shapes:&[::gust::Shape::Grid],"),
        "{expansion}"
    );
    assert!(!expansion.contains("Shape::Pixel"));
}

#[test]
fn the_kernel_of_par_for_is_grid_only_by_the_rules() {
    let input: TokenStream =
        "on(device); i in 0..n, [a: &[f32], b: &[f32], out: &mut [f32]] { out[i] = a[i] * b[i]; }"
            .parse()
            .unwrap();
    let expansion = without_whitespace(par_for_in(
        input,
        &Environment::default(),
        Profile::Other,
        1,
    ));
    assert!(
        expansion.contains("compatible_shapes:&[::gust::Shape::Grid],"),
        "{expansion}"
    );
    assert!(!expansion.contains("Shape::Pixel"), "{expansion}");
}

// ---------------------------------------------------------------------------------------------------------------
// The pin.
// ---------------------------------------------------------------------------------------------------------------

fn expanded(source: &str) -> syn::Result<String> {
    expand(module(source)).map(without_whitespace)
}

#[test]
fn the_descriptor_lists_the_compatible_shapes() {
    let both = expanded(&body("out[id.x] = a[id.x];")).unwrap();
    assert!(
        both.contains("compatible_shapes:&[::gust::Shape::Grid,::gust::Shape::Pixel],"),
        "{both}"
    );
    // A pixel-compatible kernel is still built as the profile's `grid`.
    assert!(both.contains("shape:::gust::Shape::Grid,"), "{both}");
    let sized = expanded(&kernel(
        "#[kernel(workgroup_size(64, 1, 1))]",
        PARAMETERS,
        "out[id.x] = a[id.x];",
    ))
    .unwrap();
    assert!(
        sized.contains("compatible_shapes:&[::gust::Shape::Grid],"),
        "{sized}"
    );
}

#[test]
fn a_grid_pin_is_always_accepted_and_changes_nothing() {
    let pinned = kernel(
        "#[kernel(shape = grid)]",
        PARAMETERS,
        "out[id.x] = a[id.x];",
    );
    let unpinned = body("out[id.x] = a[id.x];");
    assert_eq!(expanded(&pinned).unwrap(), expanded(&unpinned).unwrap());
    let sized = kernel(
        "#[kernel(shape = grid, workgroup_size(64, 1, 1))]",
        PARAMETERS,
        "if a[id.x] > 0.0f32 { out[id.x] = a[id.x]; }",
    );
    assert!(
        expanded(&sized)
            .unwrap()
            .contains("compatible_shapes:&[::gust::Shape::Grid],")
    );
}

#[test]
fn a_pixel_pin_on_a_compatible_kernel_waits_for_the_pixel_profile() {
    let source = kernel(
        "#[kernel(shape = pixel)]",
        PARAMETERS,
        "out[id.x] = a[id.x];",
    );
    let error = expanded(&source).unwrap_err();
    assert_code(&error, Code::ShapeNotAvailable);
    let text = error.to_string();
    assert!(text.contains("no `pixel` profile yet"), "{text}");
}

#[test]
fn a_pixel_pin_outside_the_compatible_set_is_an_error_with_the_code_of_the_construct() {
    let pinned =
        |parameters: &str, body: &str| kernel("#[kernel(shape = pixel)]", parameters, body);
    let sized = |parameters: &str, body: &str| {
        kernel(
            "#[kernel(shape = pixel, workgroup_size(64, 1, 1))]",
            parameters,
            body,
        )
    };
    let cases: Vec<(String, Code, &str)> = vec![
        (
            sized(PARAMETERS, "out[id.x] = a[id.x];"),
            Code::ShapeExcluded,
            "workgroup_size(64, 1, 1)",
        ),
        (
            pinned(
                "mut id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>",
                "out[id.x] = 1.0f32;",
            ),
            Code::ShapeMutableInvocation,
            "mut id",
        ),
        (
            pinned(PARAMETERS, "out[id.x + 1u32] = a[id.x];"),
            Code::ShapeOwnIndex,
            "out[id.x + 1u32]",
        ),
        (
            pinned(PARAMETERS, "let i = id.x; let i = 2u32; out[i] = a[i];"),
            Code::ShapeOwnIndex,
            "i",
        ),
        (
            pinned(PARAMETERS, "if a[id.x] > 0.0f32 { out[id.x] = 1.0f32; }"),
            Code::ShapeConditionalWrite,
            "a[id.x] > 0.0f32",
        ),
        (
            pinned(
                PARAMETERS,
                "if id.x >= out.len() { return; } out[id.x] = 1.0f32;",
            ),
            Code::ShapeReturn,
            "return",
        ),
        (
            pinned(PARAMETERS, "out[id.x] += a[id.x];"),
            Code::ShapeOutputUse,
            "out[id.x] += a[id.x]",
        ),
        (
            kernel(
                "#[kernel(shape = pixel, workgroup_size(8, 1, 1), cpu = \"unsupported\")]",
                "id: SV_DispatchThreadID, lid: LocalId, mut out: RWStructuredBuffer<f32>, \
                 mut scratch: Workgroup<[f32; 8]>",
                "scratch[lid.x] = 1.0f32; workgroup_barrier(); out[id.x] = scratch[0u32];",
            ),
            Code::ShapeExcluded,
            "workgroup_size(8, 1, 1)",
        ),
    ];
    for (source, code, construct) in cases {
        let error = expanded(&source).unwrap_err();
        assert_code(&error, code);
        let text = error.to_string();
        assert!(
            text.contains("shape `pixel` is not compatible with kernel `k`"),
            "{text}"
        );
        assert!(text.contains(construct), "{construct} in {text}");
        assert!(text.contains("grid`-only"), "{text}");
    }
}

#[test]
fn a_pin_that_is_refused_is_one_error_spanned_at_the_pin() {
    let source = kernel(
        "#[kernel(shape = pixel)]",
        PARAMETERS,
        "if a[id.x] > 0.0f32 { out[id.x] = 1.0f32; }",
    );
    let error = expanded(&source).unwrap_err();
    assert_eq!(error.into_iter().count(), 1);
}

#[test]
fn a_shape_the_dialect_does_not_have_is_an_unknown_shape() {
    for value in ["vulkan", "compute", "Grid", "fragment"] {
        let source = kernel(
            &format!("#[kernel(shape = {value})]"),
            PARAMETERS,
            "out[id.x] = a[id.x];",
        );
        let error = expanded(&source).unwrap_err();
        assert_code(&error, Code::ShapeUnknown);
        assert!(error.to_string().contains(value), "{error}");
    }
}

#[test]
fn cpu_is_a_target_not_a_shape() {
    let source = kernel("#[kernel(shape = cpu)]", PARAMETERS, "out[id.x] = a[id.x];");
    let error = expanded(&source).unwrap_err();
    assert_code(&error, Code::ShapeOption);
    assert!(error.to_string().contains("GUST_PROFILE=cpu"), "{error}");
}

#[test]
fn the_shape_option_is_given_once_with_a_bare_name() {
    for attribute in [
        "#[kernel(shape = grid, shape = grid)]",
        "#[kernel(shape = grid, shape = pixel)]",
        "#[kernel(shape = \"grid\")]",
        "#[kernel(shape = 1)]",
        "#[kernel(shape)]",
    ] {
        let source = kernel(attribute, PARAMETERS, "out[id.x] = a[id.x];");
        let error = expanded(&source).unwrap_err();
        assert_code(&error, Code::ShapeOption);
    }
}

#[test]
fn the_shape_option_combines_with_the_others() {
    for attribute in [
        "#[kernel(shape = grid, cpu = \"unsupported\")]",
        "#[kernel(cpu = \"unsupported\", shape = grid)]",
        "#[kernel(workgroup_size(8, 8, 1), shape = grid)]",
    ] {
        let source = kernel(attribute, PARAMETERS, "out[id.x] = a[id.x];");
        expanded(&source).unwrap();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The refusal of a profile (GUST-E0729).
// ---------------------------------------------------------------------------------------------------------------

/// `grid` in everything but the body checks, which refuse a barrier.
struct Refuser;

impl StageProfile for Refuser {
    fn shape(&self) -> crate::profile::Shape {
        GRID.shape()
    }

    fn slang_stage(&self) -> &'static str {
        GRID.slang_stage()
    }

    fn invocation(&self, ty: &syn::Type) -> Option<Builtin> {
        GRID.invocation(ty)
    }

    fn binding_rules(&self) -> BindingRules {
        GRID.binding_rules()
    }

    fn options(&self) -> OptionRules {
        GRID.options()
    }

    fn validate_body(&self, cx: &mut BodyChecks) {
        if cx.barrier {
            cx.forbidden
                .push("the refusing stage has no barriers".to_owned());
        }
    }

    fn entry_signature(&self, k: &KernelFacts<'_>) -> String {
        GRID.entry_signature(k)
    }

    fn prologue(&self, k: &KernelFacts<'_>) -> String {
        GRID.prologue(k)
    }

    fn epilogue(&self, k: &KernelFacts<'_>) -> String {
        GRID.epilogue(k)
    }

    fn lower_output_write(&self, write: &OwnIndexWrite<'_>) -> Option<String> {
        GRID.lower_output_write(write)
    }

    fn extra_bindings(&self, k: &KernelFacts<'_>) -> Vec<ExtraBinding> {
        GRID.extra_bindings(k)
    }

    fn twin(&self) -> TwinHarness {
        GRID.twin()
    }

    fn reflection(&self) -> ReflectionKind {
        GRID.reflection()
    }

    fn contract_tier(&self) -> ContractTier {
        GRID.contract_tier()
    }
}

#[test]
fn a_construct_a_profile_refuses_has_its_own_code() {
    let barrier = module(&kernel(
        "#[kernel(cpu = \"unsupported\")]",
        "id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>",
        "out[id.x] = 1.0f32; workgroup_barrier();",
    ));
    // `grid` takes it; the refusing profile does not.
    validate_module_in(&GRID, &barrier).unwrap();
    let error = validate_module_in(&Refuser, &barrier).unwrap_err();
    assert_code(&error, Code::ProfileRefusal);
    assert!(
        error
            .to_string()
            .contains("the refusing stage has no barriers"),
        "{error}"
    );
}

#[test]
fn a_construct_is_written_as_source_text() {
    let tokens = |text: &str| render(&text.parse::<TokenStream>().unwrap());
    assert_eq!(
        tokens("id.x < out.len() && a[id.x] > 0.0f32"),
        "id.x < out.len() && a[id.x] > 0.0f32"
    );
    assert_eq!(tokens("out [ i + 1 ] = & mut x"), "out[i + 1] = &mut x");
    assert_eq!(tokens("f ( a , b )"), "f(a, b)");
}

#[test]
fn a_qualified_element_type_is_judged_by_its_last_segment() {
    for (ty, bytes) in [
        ("f32", Some(4)),
        ("gust::Vec4", Some(16)),
        ("::gust::math::Vec2", Some(8)),
        ("Vec3", None),
        ("Option<f32>", None),
    ] {
        let parsed: syn::Type = parse_str(ty).unwrap();
        assert_eq!(target_bytes(&parsed), bytes, "{ty}");
    }
}

#[test]
fn a_refused_pin_counts_the_other_constructs_that_rule_the_shape_out() {
    let one = kernel(
        "#[kernel(shape = pixel, workgroup_size(64, 1, 1))]",
        PARAMETERS,
        "out[id.x] = a[id.x];",
    );
    let text = expanded(&one).unwrap_err().to_string();
    assert!(!text.contains("more construct"), "{text}");
    let several = kernel(
        "#[kernel(shape = pixel, workgroup_size(64, 1, 1))]",
        "mut id: SV_DispatchThreadID, a: StructuredBuffer<f32>, mut out: RWStructuredBuffer<f32>",
        "out[id.x + 1u32] = a[id.x];",
    );
    let text = expanded(&several).unwrap_err().to_string();
    assert!(text.contains("and 2 more constructs rule it out"), "{text}");
}
