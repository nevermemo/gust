//! The map generator (m7.a1, D60): `#[gpu]` on a free scalar function.

use proc_macro2::TokenStream;

use crate::{
    diagnostics::{
        Code,
        testing::{assert_rendered_code, parse_str, parse2},
    },
    expand::{Profile, kernel_options, map_kernel},
    regression_tests::assert_golden,
    slang::emit_kernel,
};

const FIXTURE: &str = include_str!("../../../../tests/fixtures/map.rs");
const GOLDEN: &str = include_str!("../../../../tests/fixtures/map.slang");

/// The Slang of the kernel the generator makes for `function`.
fn kernel_slang(function: &syn::ItemFn) -> String {
    let generated = map_kernel(function, Profile::Other).unwrap();
    let (_, items) = generated.module.content.as_ref().unwrap();
    let kernel = items
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(function) if function.sig.ident == "map_kernel" => Some(function),
            _ => None,
        })
        .unwrap();
    let options = kernel_options(kernel).unwrap().unwrap();
    emit_kernel(&generated.module, kernel, options)
        .unwrap()
        .source
}

/// The expansion of `source` (a `#[gpu]` item) for `profile`.
fn expansion(source: &str, profile: Profile) -> TokenStream {
    let item: TokenStream = source.parse().unwrap();
    crate::gpu_impl_with(TokenStream::new(), item, profile)
}

/// [`expansion`] with all whitespace removed, so assertions do not depend on token spacing.
fn expanded(source: &str, profile: Profile) -> String {
    expansion(source, profile)
        .to_string()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect()
}

/// Asserts the expansion of `source` is a `compile_error!` carrying `code`, followed by the item as written.
#[track_caller]
fn assert_rejected(source: &str, code: Code) {
    let output = expansion(source, Profile::Other).to_string();
    assert!(
        output.starts_with(":: core :: compile_error !"),
        "{output}\nfor {source}"
    );
    assert_rendered_code(&output, code);
    let item: TokenStream = source.parse().unwrap();
    assert!(
        output.ends_with(&item.to_string()),
        "the item must follow the error as written: {output}"
    );
}

const SAXPY: &str =
    "#[inline] pub fn saxpy(a: f32, x: f32, flag: bool) -> f32 { if flag { a * x } else { x } }";

// ---------------------------------------------------------------------------------------------------------------
// The kernel.
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn kernels_of_the_map_fixture_match_the_golden() {
    let file: syn::File = parse_str(FIXTURE).unwrap();
    let mut actual = String::new();
    for item in file.items {
        let syn::Item::Fn(mut function) = item else {
            panic!("the fixture holds functions only");
        };
        function.attrs.clear();
        actual.push_str(&format!("// ==== kernel {} ====\n", function.sig.ident));
        actual.push_str(&kernel_slang(&function));
        actual.push('\n');
    }
    assert_golden(&actual, "map.slang", &GOLDEN.replace("\r\n", "\n"));
}

#[test]
fn kernel_is_the_grid_stride_form_with_a_broadcast_read() {
    let function: syn::ItemFn = parse_str("pub fn add(a: u32, b: u32) -> u32 { a + b }").unwrap();
    let slang = kernel_slang(&function);
    assert!(
        slang.contains("16776960"),
        "the ceiling of one round:\n{slang}"
    );
    assert!(slang.contains("numthreads(256, 1, 1)"), "{slang}");
    assert!(slang.contains("gpu_add_map_kernel"), "{slang}");
    // The length-1 broadcast read at index 0, once per input.
    assert!(slang.matches("== uint(1)").count() >= 2, "{slang}");
}

// ---------------------------------------------------------------------------------------------------------------
// The expansion.
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn function_stays_and_gains_a_module_of_its_name() {
    let file: syn::File = parse2(expansion(SAXPY, Profile::Other)).unwrap();
    let syn::Item::Fn(function) = &file.items[0] else {
        panic!("the function comes first");
    };
    assert_eq!(function.sig.ident, "saxpy");
    assert!(
        function
            .attrs
            .iter()
            .any(|attribute| attribute.path().is_ident("inline"))
    );
    let syn::Item::Mod(module) = &file.items[1] else {
        panic!("the module follows");
    };
    assert_eq!(module.ident, "saxpy");
    assert!(matches!(module.vis, syn::Visibility::Public(_)));
}

#[test]
fn gpu_profile_has_the_six_calls_the_descriptor_and_the_own_index_stamp() {
    let text = expanded(SAXPY, Profile::Other);
    for needle in [
        "pubfnmap_on<__GustD,",
        "pubasyncfnmap_on_async<",
        "pubfnmap<",
        "pubfnmap_gpu_on<",
        "pubfnmap_gpu<",
        "pubasyncfnmap_async<",
        "pubconstDESCRIPTOR:",
        "pubconstOWN_INDEX:bool=true;",
        "impl::gust::map::OwnIndexformap_kernel{}",
        "::gust_wgpu::default_device()",
        "::gust_wgpu::default_device_async()",
        "::gust::map::CEILING",
    ] {
        assert!(text.contains(needle), "missing {needle}");
    }
    // The blocking calls do not exist on wasm32 (nothing may block there); the async ones do.
    let gate = "#[cfg(not(target_arch=\"wasm32\"))]#[allow(clippy::too_many_arguments)]";
    assert!(text.contains(&format!("{gate}pubfnmap<")));
    assert!(text.contains(&format!("{gate}pubfnmap_on<")));
    assert!(!text.contains(&format!("{gate}pubasyncfn")));
    // The cpu-only chunk body is not in this profile.
    assert!(!text.contains("__gust_map_cpu_chunk"));
}

#[test]
fn cpu_profile_names_no_gpu_api_and_has_the_chunk_body() {
    let text = expanded(SAXPY, Profile::Cpu);
    assert!(
        !text.contains("gust_wgpu"),
        "the cpu profile must not name the GPU API"
    );
    for needle in [
        "pubfnmap<",
        "pubfnmap_on<",
        "::gust::cpu::CpuDevice::new()",
        "pubfn__gust_map_cpu_chunk(",
        "__gust_lo:u32,__gust_hi:u32,",
        "pubconstOWN_INDEX:bool=true;",
    ] {
        assert!(text.contains(needle), "missing {needle}");
    }
    for absent in ["map_gpu", "map_async", "default_device"] {
        assert!(
            !text.contains(absent),
            "the cpu profile must not have {absent}"
        );
    }
    // The chunk body calls the user's function, with the broadcast rule on each input and `bool` read as `!= 0`.
    assert!(text.contains("let__gust_r=saxpy(__gust_x0,__gust_x1,__gust_x2);"));
    assert!(text.contains(".len()==1{0}else{__gust_iasusize}"));
    assert!(text.contains("!=0;"));
}

#[test]
fn elements_are_the_buffer_types_and_bool_is_a_u32_word() {
    let text = expanded(SAXPY, Profile::Other);
    assert!(text.contains("MapArg<::core::primitive::f32,__GustD>"));
    // The argument bound is keyed on the host type: a `bool` parameter takes `bool`s, never `u32`s.
    assert!(
        text.contains("MapArg<::core::primitive::bool,__GustD>"),
        "bool arguments are keyed on bool"
    );
    assert!(
        !text.contains("MapArg<::core::primitive::u32,__GustD>"),
        "a bool argument is not a u32 argument"
    );
    assert!(text.contains("Result<::std::vec::Vec<::core::primitive::f32>,::gust::map::Error>"));
    let boolean = expanded(
        "pub fn is_odd(n: u32) -> bool { n % 2u32 == 1u32 }",
        Profile::Other,
    );
    assert!(
        boolean.contains("Result<::std::vec::Vec<::core::primitive::bool>,::gust::map::Error>")
    );
    assert!(boolean.contains("::<::core::primitive::u32>(__gust_device,&__gust_out)"));
}

#[test]
fn a_parameter_named_id_does_not_meet_the_dispatch_id() {
    // The kernel's own dispatch-id parameter has a reserved name, so the user's `id` is just a buffer name.
    let function: syn::ItemFn =
        parse_str("pub fn hash(id: u32, salt: u32) -> u32 { (id + salt) * 3u32 }").unwrap();
    let slang = kernel_slang(&function);
    assert!(slang.contains("StructuredBuffer<uint> id;"), "{slang}");
    assert!(
        slang.contains("uint3 __gust_gid : SV_DispatchThreadID"),
        "{slang}"
    );
    assert!(slang.contains("__gust_gid.x"), "{slang}");
    let text = expansion(
        "pub fn hash(id: u32, salt: u32) -> u32 { (id + salt) * 3u32 }",
        Profile::Other,
    )
    .to_string();
    assert!(!text.contains("compile_error"), "{text}");
}

#[test]
fn the_module_has_the_visibility_of_the_function() {
    for (written, visibility) in [
        ("fn f(x: f32) -> f32 { x }", ""),
        ("pub fn f(x: f32) -> f32 { x }", "pub"),
        ("pub(crate) fn f(x: f32) -> f32 { x }", "pub(crate)"),
    ] {
        let file: syn::File = parse2(expansion(written, Profile::Other)).unwrap();
        let syn::Item::Mod(module) = &file.items[1] else {
            panic!("the module follows the function: {written}");
        };
        let found: String = quote::ToTokens::to_token_stream(&module.vis)
            .to_string()
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        assert_eq!(found, visibility, "{written}");
    }
}

#[test]
fn cfg_attributes_of_the_function_gate_the_module_too() {
    let text = expanded(
        "#[cfg(feature = \"x\")] pub fn f(x: f32) -> f32 { x }",
        Profile::Other,
    );
    assert!(text.starts_with("#[cfg(feature=\"x\")]pubfnf("), "{text}");
    assert!(text.contains("#[cfg(feature=\"x\")]#[doc="), "{text}");
}

#[test]
fn vector_and_matrix_elements_are_accepted() {
    for source in [
        "pub fn f(v: Vec2, s: f32) -> Vec2 { Vec2::new(v.x * s, v.y) }",
        "pub fn f(v: IVec4) -> IVec4 { v }",
        "pub fn f(v: UVec2) -> UVec2 { v }",
        "pub fn f(m: Mat4) -> Mat4 { m }",
        "pub fn f(x: float, n: int, u: uint) -> float { x }",
    ] {
        let text = expansion(source, Profile::Other).to_string();
        assert!(!text.contains("compile_error"), "{source}: {text}");
    }
}

#[test]
fn three_wide_values_are_fine_inside_the_body() {
    let source = "pub fn f(v: Vec2) -> f32 { let w = Vec3::new(v.x, v.y, 1.0); w.x + w.y + w.z }";
    let text = expansion(source, Profile::Other).to_string();
    assert!(!text.contains("compile_error"), "{text}");
}

// ---------------------------------------------------------------------------------------------------------------
// The diagnostics.
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn signature_errors_are_e0714() {
    for source in [
        "pub fn f() -> f32 { 1.0 }",
        "pub fn f(x: f32) {}",
        "pub fn f(x: f64) -> f32 { 1.0 }",
        "pub fn f(x: f32) -> f64 { 1.0 }",
        "pub fn f(x: &f32) -> f32 { *x }",
        "pub fn f(x: Option<f32>) -> f32 { 1.0 }",
        "pub fn f(x: f32) -> (f32, f32) { (x, x) }",
        "pub fn f(x: [f32; 4]) -> f32 { 1.0 }",
        "pub fn f(x: IVec3) -> f32 { 1.0 }",
        "pub fn f(x: gust::math::Vec2) -> f32 { 1.0 }",
        "#[kernel] pub fn f(x: f32) -> f32 { x }",
    ] {
        assert_rejected(source, Code::MapSignature);
    }
}

#[test]
fn names_the_module_defines_are_e0715() {
    for name in [
        "map",
        "map_on",
        "map_gpu",
        "map_gpu_on",
        "map_async",
        "map_on_async",
        "map_kernel",
        "DESCRIPTOR",
        "MODULE_DESCRIPTOR",
        "OWN_INDEX",
    ] {
        assert_rejected(
            &format!("pub fn {name}(x: f32) -> f32 {{ x }}"),
            Code::MapNameCollision,
        );
    }
    assert_rejected("pub fn f(f: f32) -> f32 { f }", Code::MapNameCollision);
}

#[test]
fn nested_gpu_functions_are_e0716() {
    for attribute in ["gpu", "gust::gpu"] {
        let source = format!("mod m {{ #[{attribute}] pub fn f(x: f32) -> f32 {{ x }} }}");
        let output = expansion(&source, Profile::Other).to_string();
        assert!(output.starts_with(":: core :: compile_error !"), "{output}");
        assert_rendered_code(&output, Code::NestedGpuFunction);
    }
}

#[test]
fn a_rejected_module_returns_without_the_nested_attribute() {
    // Left on, the inner `#[gust::gpu]` would expand by itself in the user's crate and add errors that only follow.
    let source =
        "mod m { #[gust::gpu] pub fn f(x: f32) -> f32 { x } pub fn g(x: f32) -> f32 { x } }";
    let output = expansion(source, Profile::Other).to_string();
    assert!(output.starts_with(":: core :: compile_error !"), "{output}");
    assert!(!output.contains("# [gust :: gpu]"), "{output}");
    assert!(
        output.contains("pub fn f") && output.contains("pub fn g"),
        "{output}"
    );
}

#[test]
fn three_wide_types_are_e0204() {
    for source in [
        "pub fn f(v: Vec3) -> f32 { v.x }",
        "pub fn f(v: UVec3) -> f32 { 1.0 }",
        "pub fn f(m: Mat3) -> f32 { 1.0 }",
        "pub fn f(x: f32) -> Vec3 { Vec3::new(x, x, x) }",
    ] {
        assert_rejected(source, Code::ThreeWideInBuffer);
    }
}

#[test]
fn the_dialect_rules_apply_to_the_body() {
    // The validator sees the function as a helper of a module: its codes, not a map code.
    for source in [
        "pub fn f(x: f32) -> f32 { let v = Vec::new(); x }",
        "pub async fn f(x: f32) -> f32 { x }",
    ] {
        let output = expansion(source, Profile::Other).to_string();
        assert!(output.starts_with(":: core :: compile_error !"), "{output}");
        assert!(!output.contains("GUST-E0714"), "{output}");
    }
}

#[test]
fn other_items_are_e0702() {
    for source in [
        "struct S;",
        "const X: u32 = 1;",
        "static Y: u32 = 2;",
        "use std::fmt;",
        "trait T {}",
    ] {
        assert_rejected(source, Code::ItemNotModule);
    }
}

#[test]
fn arguments_are_still_rejected_on_a_function() {
    let output = crate::gpu_impl_with(
        "workgroup(4)".parse().unwrap(),
        "pub fn f(x: f32) -> f32 { x }".parse().unwrap(),
        Profile::Other,
    )
    .to_string();
    assert!(output.starts_with(":: core :: compile_error !"), "{output}");
    assert_rendered_code(&output, Code::GpuArguments);
    assert!(
        output.contains("pub fn f"),
        "the function follows the error: {output}"
    );
}
