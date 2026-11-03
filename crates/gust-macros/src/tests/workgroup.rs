//! Workgroup memory and the workgroup barrier (m7.g1): the golden Slang of `tests/fixtures/workgroup.rs`, every
//! rejection of the barrier rules, and what the expansion leaves out of the descriptor, the typed call and the twin.

use proc_macro2::TokenStream;

use crate::{
    diagnostics::{
        Code,
        testing::{assert_rendered_code, parse_str},
    },
    expand::{Profile, kernel_options},
    regression_tests::assert_golden,
    slang::emit_kernel,
    validate::validate_module,
};

const FIXTURE: &str = include_str!("../../../../tests/fixtures/workgroup.rs");
const GOLDEN: &str = include_str!("../../../../tests/fixtures/workgroup.slang");

/// The expansion of a `#[gpu]` module given as source without its attribute.
fn expansion(source: &str, profile: Profile) -> TokenStream {
    let item: TokenStream = source.parse().unwrap();
    crate::gpu_impl_with(TokenStream::new(), item, profile)
}

/// The fixture module as `#[gpu]` receives it (the attribute is the macro's own).
fn fixture_module() -> String {
    FIXTURE.replacen("#[gust::gpu]", "", 1)
}

/// [`expansion`] with all whitespace removed, so assertions do not depend on token spacing.
fn expanded(source: &str, profile: Profile) -> String {
    expansion(source, profile)
        .to_string()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect()
}

/// Asserts the expansion of `source` is a `compile_error!` carrying `code` and no other code.
#[track_caller]
fn assert_rejected(source: &str, code: Code) {
    let output = expansion(source, Profile::Other).to_string();
    assert!(
        output.starts_with(":: core :: compile_error !"),
        "{output}\nfor {source}"
    );
    assert_rendered_code(&output, code);
    assert_eq!(
        output.matches("[GUST-E").count(),
        1,
        "exactly one diagnostic expected: {output}\nfor {source}"
    );
}

/// A module with one kernel: `attribute` is the `#[kernel(..)]` argument list, `parameters` follow the dispatch-thread
/// parameter, `items` come before the kernel.
fn module(attribute: &str, items: &str, parameters: &str, body: &str) -> String {
    format!(
        "mod t {{ {items} #[kernel({attribute})] pub fn k(id: SV_DispatchThreadID{}{parameters}) {{ {body} }} }}",
        if parameters.is_empty() { "" } else { ", " }
    )
}

const UNSUPPORTED: &str = "workgroup_size(64, 1, 1), cpu = \"unsupported\"";
const SCRATCH: &str = "lid: LocalId, mut scratch: Workgroup<[f32; 64]>";

/// A kernel with workgroup memory and the given body.
fn barrier_kernel(body: &str) -> String {
    module(UNSUPPORTED, "", SCRATCH, body)
}

// ---------------------------------------------------------------------------------------------------------------
// The fixture.
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn kernels_of_the_workgroup_fixture_match_the_golden() {
    let module: syn::ItemMod = parse_str(fixture_module().trim()).unwrap();
    validate_module(&module).unwrap();
    let mut actual = String::new();
    for item in &module.content.as_ref().unwrap().1 {
        let syn::Item::Fn(function) = item else {
            continue;
        };
        let Some(options) = kernel_options(function).unwrap() else {
            continue;
        };
        actual.push_str(&format!("// ==== kernel {} ====\n", function.sig.ident));
        actual.push_str(
            &emit_kernel(&module, function, options)
                .unwrap()
                .source
                .replace("\r\n", "\n"),
        );
    }
    assert_golden(&actual, "workgroup.slang", &GOLDEN.replace("\r\n", "\n"));
}

#[test]
fn the_golden_shows_the_lowering() {
    let golden = GOLDEN.replace("\r\n", "\n");
    // Workgroup memory is a module-scope `groupshared` array named like the parameter.
    assert!(
        golden.contains("groupshared float scratch[256];"),
        "{golden}"
    );
    // The barrier is the combined group memory barrier and sync.
    assert_eq!(
        golden.matches("GroupMemoryBarrierWithGroupSync();").count(),
        2,
        "{golden}"
    );
    // The ids are derived from the dispatch-thread id and the workgroup size.
    assert!(
        golden.contains("uint3 lid = uint3(id.x % 256u, id.y % 1u, id.z % 1u);"),
        "{golden}"
    );
    assert!(
        golden.contains("uint3 gid = uint3(id.x / 256u, id.y / 1u, id.z / 1u);"),
        "{golden}"
    );
    // The entry point keeps its one semantic parameter.
    assert_eq!(golden.matches("SV_DispatchThreadID").count(), 2, "{golden}");
    assert!(!golden.contains("SV_GroupThreadID"), "{golden}");
    assert!(!golden.contains("SV_GroupID"), "{golden}");
}

#[test]
fn two_kernels_with_a_workgroup_parameter_of_the_same_name_keep_their_own_array() {
    // The `groupshared` array is named like the parameter (not `<kernel>__<name>`), which is safe because each
    // kernel is emitted as its own Slang source with only its own parameters. This pins that property.
    let source = "mod t {
        #[kernel(workgroup_size(64, 1, 1), cpu = \"unsupported\")]
        pub fn first(id: SV_DispatchThreadID, lid: LocalId, mut out: RWStructuredBuffer<f32>,
                     mut scratch: Workgroup<[f32; 64]>) {
            scratch[lid.x] = 1.0f32;
            workgroup_barrier();
            out[id.x] = scratch[0u32];
        }
        #[kernel(workgroup_size(32, 1, 1), cpu = \"unsupported\")]
        pub fn second(id: SV_DispatchThreadID, lid: LocalId, mut out: RWStructuredBuffer<u32>,
                      mut scratch: Workgroup<[u32; 32]>) {
            scratch[lid.x] = 2u32;
            workgroup_barrier();
            out[id.x] = scratch[0u32];
        }
    }";
    let module: syn::ItemMod = parse_str(source).unwrap();
    validate_module(&module).unwrap();
    let mut sources = Vec::new();
    for item in &module.content.as_ref().unwrap().1 {
        let syn::Item::Fn(function) = item else {
            continue;
        };
        let options = kernel_options(function).unwrap().unwrap();
        sources.push(emit_kernel(&module, function, options).unwrap().source);
    }
    assert_eq!(sources.len(), 2);
    for (source, own, other) in [
        (
            &sources[0],
            "groupshared float scratch[64];",
            "uint scratch[32]",
        ),
        (
            &sources[1],
            "groupshared uint scratch[32];",
            "float scratch[64]",
        ),
    ] {
        assert_eq!(source.matches("groupshared").count(), 1, "{source}");
        assert_eq!(source.matches(own).count(), 1, "{source}");
        assert!(!source.contains(other), "{source}");
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The expansion.
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn derived_ids_and_workgroup_memory_are_not_in_the_descriptor() {
    let text = expanded(&fixture_module(), Profile::Other);
    for present in [
        "name:\"id\"",
        "name:\"input\"",
        "name:\"partial\"",
        "name:\"locals\"",
    ] {
        assert!(text.contains(present), "missing {present}");
    }
    for absent in ["name:\"scratch\"", "name:\"lid\"", "name:\"gid\""] {
        assert!(
            !text.contains(absent),
            "{absent} must not be a descriptor parameter"
        );
    }
}

#[test]
fn the_layout_assertions_index_the_descriptor_not_the_parameter_list() {
    // `block_sum(id, lid, gid, input, partial, scratch)`: the descriptor holds id, input and partial, so the two
    // buffers are descriptor entries 1 and 2, whatever their place among the parameters.
    let text = expanded(&fixture_module(), Profile::Other);
    assert!(text.contains("block_sumas::gust::Kernel>::DESCRIPTOR.parameters[1]"));
    assert!(text.contains("block_sumas::gust::Kernel>::DESCRIPTOR.parameters[2]"));
    assert!(!text.contains("block_sumas::gust::Kernel>::DESCRIPTOR.parameters[3]"));
    assert!(!text.contains("block_sumas::gust::Kernel>::DESCRIPTOR.parameters[4]"));
}

#[test]
fn the_typed_call_takes_the_buffers_only() {
    let text = expanded(&fixture_module(), Profile::Other);
    assert!(text.contains("input:impl::core::convert::Into<"), "{text}");
    assert!(
        text.contains("partial:impl::core::convert::Into<"),
        "{text}"
    );
    for absent in ["scratch:impl", "lid:impl", "gid:impl"] {
        assert!(
            !text.contains(absent),
            "{absent} must not be an argument of the call"
        );
    }
}

#[test]
fn a_barrier_kernel_has_no_cpu_twin_in_any_profile() {
    for profile in [Profile::Other, Profile::Cpu] {
        let text = expanded(&fixture_module(), profile);
        assert!(
            !text.contains("Run<block_sum,"),
            "an unsupported kernel has no Run impl"
        );
        assert!(
            text.contains("Run<local_ids,"),
            "the kernel without a barrier keeps its twin"
        );
    }
}

#[test]
fn the_twin_derives_the_ids_like_the_gpu_does() {
    for profile in [Profile::Other, Profile::Cpu] {
        let text = expanded(&fixture_module(), profile);
        assert!(
            text.contains("::gust::UVec3{x:__gust_id.x%256u32,y:0u32,z:0u32,}"),
            "{text}"
        );
        assert!(
            text.contains("::gust::UVec3{x:__gust_id.x/256u32,y:__gust_id.y,z:__gust_id.z,}"),
            "{text}"
        );
        // No `% 1` or `/ 1` is generated: clippy denies `modulo_one` in the user's crate.
        assert!(!text.contains("%1u32") && !text.contains("/1u32"), "{text}");
    }
}

#[test]
fn the_pending_slang_text_in_the_descriptor_holds_the_lowering() {
    let text = expanded(&fixture_module(), Profile::Other);
    assert!(text.contains("groupsharedfloatscratch[256];"));
    assert!(text.contains("GroupMemoryBarrierWithGroupSync();"));
}

// ---------------------------------------------------------------------------------------------------------------
// The accepted shapes.
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn accepted_shapes_expand() {
    for body in [
        // A barrier in the body, and one in the literal-range loop at the top level.
        "scratch[lid.x] = 1.0f32; workgroup_barrier(); for r in 0u32..4u32 { workgroup_barrier(); }",
        // The loop variable is free, the range is two literals of any integer type.
        "for r in 0i32..4i32 { scratch[lid.x] = 2.0f32; workgroup_barrier(); }",
        // A loop without a barrier may break and continue, and nest, and sit under an `if`.
        "workgroup_barrier(); for r in 0u32..4u32 { if r == 2u32 { break; } if r == 1u32 { continue; } }",
        "workgroup_barrier(); if lid.x == 0u32 { for r in 0u32..4u32 { if r == 2u32 { break; } } }",
    ] {
        let output = expansion(&barrier_kernel(body), Profile::Other).to_string();
        assert!(!output.contains("compile_error"), "{output}\nfor {body}");
    }
}

#[test]
fn memory_up_to_the_limit_is_accepted() {
    // 16384 bytes exactly: 4096 f32, or 2048 two-wide vectors, or 1024 four-wide ones, or a mix.
    for parameters in [
        "mut a: Workgroup<[f32; 4096]>",
        "mut a: Workgroup<[Vec2; 2048]>",
        "mut a: Workgroup<[Vec4; 1024]>",
        "mut a: Workgroup<[Vec3; 1024]>",
        "mut a: Workgroup<[bool; 4096]>",
        "mut a: Workgroup<[u32; 2048]>, mut b: Workgroup<[i32; 2048]>",
        "mut a: Workgroup<[Vec4; 512]>, mut b: Workgroup<[Vec2; 512]>, mut c: Workgroup<[f32; 1024]>",
    ] {
        let source = module(UNSUPPORTED, "", parameters, "workgroup_barrier();");
        let output = expansion(&source, Profile::Other).to_string();
        assert!(
            !output.contains("compile_error"),
            "{output}\nfor {parameters}"
        );
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The rejections.
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn a_barrier_outside_the_approved_positions_is_e0414() {
    for body in [
        "if lid.x < 8u32 { workgroup_barrier(); }",
        "for r in 0u32..4u32 { if r == 1u32 { workgroup_barrier(); } }",
        "for r in 0u32..4u32 { for q in 0u32..4u32 { workgroup_barrier(); } }",
        "for r in 0u32..4u32 { { workgroup_barrier(); } }",
        // The bounds are not two literals.
        "let n = 4u32; for r in 0u32..n { workgroup_barrier(); }",
        "for r in 0u32..scratch_length() { workgroup_barrier(); }",
        // Under a loop that is not at the top level.
        "if lid.x == 0u32 { for r in 0u32..4u32 { workgroup_barrier(); } }",
    ] {
        let source = module(
            UNSUPPORTED,
            "fn scratch_length() -> u32 { 4u32 }",
            SCRATCH,
            body,
        );
        assert_rejected(&source, Code::BarrierPosition);
    }
}

#[test]
fn a_barrier_in_a_helper_is_e0414() {
    let source = module(
        UNSUPPORTED,
        "fn sync() { workgroup_barrier(); }",
        SCRATCH,
        "sync();",
    );
    assert_rejected(&source, Code::BarrierPosition);
}

#[test]
fn a_barrier_in_a_kernel_without_workgroup_memory_is_still_a_barrier_kernel() {
    // The barrier alone makes it one: a bad position is E0414, and the missing strategy is E0717.
    let nested = module(
        UNSUPPORTED,
        "",
        "",
        "if id.x < 4u32 { workgroup_barrier(); }",
    );
    assert_rejected(&nested, Code::BarrierPosition);
    let emitted = module("workgroup_size(64, 1, 1)", "", "", "workgroup_barrier();");
    assert_rejected(&emitted, Code::BarrierKernelCpu);
}

#[test]
fn a_return_in_a_barrier_kernel_is_e0415() {
    for body in [
        "workgroup_barrier(); return;",
        "if lid.x == 0u32 { return; } workgroup_barrier();",
        "for r in 0u32..4u32 { if r == 2u32 { return; } }",
    ] {
        assert_rejected(&barrier_kernel(body), Code::ReturnInBarrierKernel);
    }
}

#[test]
fn a_return_in_a_helper_of_a_barrier_kernel_is_fine() {
    let source = module(
        UNSUPPORTED,
        "fn pick(x: u32) -> u32 { if x > 3u32 { return 3u32; } x }",
        SCRATCH,
        "workgroup_barrier(); scratch[lid.x] = pick(lid.x) as f32;",
    );
    let output = expansion(&source, Profile::Other).to_string();
    assert!(!output.contains("compile_error"), "{output}");
}

#[test]
fn break_and_continue_in_a_loop_with_a_barrier_are_e0416() {
    for body in [
        "for r in 0u32..4u32 { if r == 2u32 { break; } workgroup_barrier(); }",
        "for r in 0u32..4u32 { if r == 2u32 { continue; } workgroup_barrier(); }",
        "for r in 0u32..4u32 { workgroup_barrier(); break; }",
        "for r in 0u32..4u32 { workgroup_barrier(); continue; }",
    ] {
        assert_rejected(&barrier_kernel(body), Code::JumpInBarrierLoop);
    }
}

#[test]
fn constructs_rejected_for_their_own_sake_stay_rejected_once() {
    // `while`, a loop label and `?` are not in the dialect at all; a barrier kernel does not add a second error.
    assert_rejected(
        &barrier_kernel("while lid.x < 4u32 { workgroup_barrier(); }"),
        Code::WhileNotSupported,
    );
    assert_rejected(
        &barrier_kernel("'outer: for r in 0u32..4u32 { workgroup_barrier(); }"),
        Code::LoopLabel,
    );
    assert_rejected(
        &barrier_kernel(
            "for r in 0u32..4u32 { if r == 1u32 { break 'outer; } } workgroup_barrier();",
        ),
        Code::LoopLabel,
    );
    assert_rejected(
        &module(
            UNSUPPORTED,
            "fn half(x: u32) -> Option<u32> { Some(x) }",
            SCRATCH,
            "let v = half(lid.x)?; workgroup_barrier();",
        ),
        Code::TryOperator,
    );
}

#[test]
fn the_barrier_is_a_statement_with_no_arguments_e0327() {
    for body in [
        "workgroup_barrier(1u32);",
        "let x = workgroup_barrier();",
        "scratch[lid.x] = (workgroup_barrier(), 1.0f32).1;",
        "let f = workgroup_barrier;",
        "if lid.x < 4u32 { workgroup_barrier() }",
        "workgroup_barrier()",
    ] {
        assert_rejected(&barrier_kernel(body), Code::BarrierForm);
    }
}

#[test]
fn a_workgroup_parameter_is_for_kernels_only_e0119() {
    let source = module(
        UNSUPPORTED,
        "fn help(mut s: Workgroup<[f32; 4]>) {}",
        SCRATCH,
        "workgroup_barrier();",
    );
    assert_rejected(&source, Code::WorkgroupParameterOnHelper);
}

#[test]
fn the_shape_of_workgroup_memory_is_e0210() {
    for parameter in [
        "mut s: Workgroup<f32>",
        "mut s: Workgroup",
        "mut s: Workgroup<[f32; 0]>",
        "mut s: Workgroup<[f32; LEN]>",
        "mut s: Workgroup<[f32; 4], u32>",
        "mut s: Workgroup<[f64; 4]>",
        "mut s: Workgroup<[Mat4; 4]>",
        "mut s: Workgroup<[Item; 4]>",
        "mut s: Workgroup<[[f32; 4]; 4]>",
    ] {
        let source = module(
            UNSUPPORTED,
            "pub struct Item { pub a: f32 }",
            parameter,
            "workgroup_barrier();",
        );
        assert_rejected(&source, Code::WorkgroupShape);
    }
}

#[test]
fn workgroup_memory_stands_only_as_a_kernel_parameter_type_e0210() {
    for (items, body) in [
        ("", "let s: Workgroup<[f32; 4]> = id.x;"),
        (
            "fn help() -> Workgroup<[f32; 4]> { id }",
            "workgroup_barrier();",
        ),
        (
            "pub struct Holder { pub s: Workgroup<[f32; 4]> }",
            "workgroup_barrier();",
        ),
    ] {
        let source = module(UNSUPPORTED, items, SCRATCH, body);
        assert_rejected(&source, Code::WorkgroupShape);
    }
}

#[test]
fn a_kernel_with_workgroup_memory_or_a_barrier_must_say_cpu_unsupported_e0717() {
    let plain = "workgroup_size(64, 1, 1)";
    for (parameters, body) in [
        (SCRATCH, "workgroup_barrier();"),
        (SCRATCH, "scratch[lid.x] = 1.0f32;"),
        ("", "workgroup_barrier();"),
    ] {
        assert_rejected(&module(plain, "", parameters, body), Code::BarrierKernelCpu);
    }
    // The message names the construct and the fix.
    let output = expansion(
        &module(plain, "", SCRATCH, "workgroup_barrier();"),
        Profile::Other,
    )
    .to_string();
    assert!(output.contains("cpu = \\\"unsupported\\\""), "{output}");
    assert!(output.contains("Workgroup"), "{output}");
}

#[test]
fn cpu_lockstep_is_not_available_yet_e0709() {
    let source = module(
        "workgroup_size(64, 1, 1), cpu = \"lockstep\"",
        "",
        SCRATCH,
        "workgroup_barrier();",
    );
    assert_rejected(&source, Code::UnknownCpuValue);
    let output = expansion(&source, Profile::Other).to_string();
    assert!(output.contains("lockstep"), "{output}");
    assert!(output.contains("not available yet"), "{output}");
}

#[test]
fn workgroup_memory_over_16_kib_is_e0718() {
    for parameters in [
        "mut a: Workgroup<[f32; 4097]>",
        "mut a: Workgroup<[Vec4; 1025]>",
        "mut a: Workgroup<[Vec3; 1025]>",
        "mut a: Workgroup<[Vec2; 2049]>",
        "mut a: Workgroup<[bool; 4097]>",
        "mut a: Workgroup<[f32; 4096]>, mut b: Workgroup<[u32; 1]>",
        "mut a: Workgroup<[f32; 2048]>, mut b: Workgroup<[Vec4; 513]>",
    ] {
        let source = module(UNSUPPORTED, "", parameters, "workgroup_barrier();");
        assert_rejected(&source, Code::WorkgroupMemoryLimit);
    }
}

#[test]
fn a_local_cannot_shadow_a_derived_id_or_workgroup_memory() {
    for body in [
        "let lid = 1u32; workgroup_barrier();",
        "let scratch = 1u32; workgroup_barrier();",
        "for scratch in 0u32..4u32 { }",
    ] {
        assert_rejected(&barrier_kernel(body), Code::ShadowedParameter);
    }
}

#[test]
fn global_id_is_the_dispatch_thread_parameters_alone() {
    assert_rejected(
        &barrier_kernel("let g = lid.global_id(); workgroup_barrier();"),
        Code::GlobalIdReceiver,
    );
}
