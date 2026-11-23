//! The map generator (m7.a1, D60): `#[gpu_fn]` on a free scalar function, and (m14.mapw1, D221) on a function of a
//! `#[gpu_module]` module, which may take and return the module's structs.

use proc_macro2::TokenStream;

use crate::{
    diagnostics::{
        Code,
        testing::{assert_rendered_code, parse_str, parse2},
    },
    expand::{Profile, kernel_options, map_kernel, map_module_kernels},
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

/// The function name and Slang of each kernel the generator adds to `module` (a `#[gpu_module]` module with
/// `#[gpu_fn]` functions), in the module's order.
fn module_kernel_slang(module: syn::ItemMod) -> Vec<(String, String)> {
    let generated = map_module_kernels(module, Profile::Other).unwrap();
    let (_, items) = generated.content.as_ref().unwrap();
    items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Fn(function) => {
                let name = function.sig.ident.to_string();
                let function_name = name.strip_suffix("_map_kernel")?.to_owned();
                let options = kernel_options(function).unwrap().unwrap();
                let source = emit_kernel(&generated, function, options).unwrap().source;
                Some((function_name, source))
            }
            _ => None,
        })
        .collect()
}

/// The expansion of `source` (a `#[gpu_module]` item) for `profile`.
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
        match item {
            syn::Item::Fn(mut function) => {
                function.attrs.clear();
                actual.push_str(&format!("// ==== kernel {} ====\n", function.sig.ident));
                actual.push_str(&kernel_slang(&function));
                actual.push('\n');
            }
            // A `#[gpu_module]` module with `#[gpu_fn]` functions (m14.mapw1): one section per function, in order.
            syn::Item::Mod(mut module) => {
                module.attrs.clear();
                for (function, source) in module_kernel_slang(module) {
                    actual.push_str(&format!("// ==== kernel {function} ====\n"));
                    actual.push_str(&source);
                    actual.push('\n');
                }
            }
            _ => panic!("the fixture holds functions and modules only"),
        }
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

/// The chunked `cpu` map (m7.c): `map_on` and `map_on_async` are concrete over `CpuDevice`, check lengths and the
/// thread count before uploading, borrow the inputs as whole-length slices, and run the chunk body through
/// `run_map_into_vec`; the chunk body records the element it evaluates. No other profile has any of it.
#[test]
fn cpu_profile_map_runs_the_chunk_body_in_parallel_pieces() {
    let text = expanded(SAXPY, Profile::Cpu);
    for needle in [
        "pubfnmap_on<__GustA0,__GustA1,__GustA2>(__gust_device:&::gust::cpu::CpuDevice,",
        "pubasyncfnmap_on_async<__GustA0,__GustA1,__GustA2>(__gust_device:&::gust::cpu::CpuDevice,",
        "::gust::cpu::parallel::map_threads(__gust_n)",
        "::gust::cpu::parallel::require_own_index::<map_kernel>();",
        "::gust::cpu::parallel::input::<::core::primitive::f32,_>(",
        "let__gust_s0:&[::core::primitive::f32]=&__gust_r0;",
        "::gust::cpu::parallel::run_map_into_vec(__gust_out,__gust_threads,",
        "__gust_map_cpu_chunk(__gust_lo,__gust_hi,__gust_s0,__gust_s1,__gust_s2,__gust_chunk,__gust_cur)",
        "__gust_cur:&::core::cell::Cell<u32>,",
        "__gust_cur.set(__gust_i);",
    ] {
        assert!(text.contains(needle), "missing {needle}");
    }
    // Not generic over the device, and no trip through the sequential `Run` impl (which stays for typed calls).
    for absent in ["__GustD", "MapHost+::gust::Run", "::dispatch(", "::submit("] {
        assert!(!text.contains(absent), "the cpu map must not have {absent}");
    }
    // The element count and the thread count are checked before anything is bound, and an empty call returns early.
    let order = [
        "element_count",
        "map_threads",
        "Vec::new()",
        "bind(a,__gust_device)",
        "::zeroed::<::core::primitive::f32>",
    ];
    let positions: Vec<usize> = order
        .iter()
        .map(|needle| {
            text.find(needle)
                .unwrap_or_else(|| panic!("missing {needle}"))
        })
        .collect();
    assert!(
        positions.windows(2).all(|pair| pair[0] < pair[1]),
        "{positions:?}"
    );
    // A `bool` input is the converted `u32` slice and a `bool` result is converted back.
    let flag = expanded(
        "pub fn pick(flag: bool, x: u32) -> bool { flag && x > 1u32 }",
        Profile::Cpu,
    );
    assert!(
        flag.contains("::gust::cpu::parallel::input::<::core::primitive::u32,_>("),
        "{flag}"
    );
    assert!(flag.contains("let__gust_s0:&[::core::primitive::u32]=&__gust_r0;"));
    assert!(flag.contains(".map(|__gust_word|__gust_word!=0u32)"));
    // Every other profile is unchanged: generic over the device, through the kernel's `Run` impl.
    let other = expanded(SAXPY, Profile::Other);
    assert!(!other.contains("gust::cpu::parallel"));
    assert!(!other.contains("__gust_cur"));
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

// ---------------------------------------------------------------------------------------------------------------
// `#[gpu_fn]` inside a `#[gpu_module]` module: structs (m14.mapw1, D221).
// ---------------------------------------------------------------------------------------------------------------

/// A module with a struct, a struct-to-struct map with a scalar beside it and a struct-to-scalar map.
const RECORDS: &str = "mod records { \
    pub struct Item { pub pos: Vec2, pub id: u32, pub weight: f32 } \
    #[gust::gpu_fn] pub fn scale(item: Item, k: f32) -> Item { \
        let moved = Item { pos: item.pos, id: item.id + 1u32, weight: item.weight * k }; moved } \
    #[gpu_fn] pub fn total(item: Item) -> f32 { item.weight + item.id as f32 } }";

/// Asserts the expansion of the module `source` is a `compile_error!` carrying `code` and that the module comes back
/// without any `#[gpu_fn]` attribute (left on, it would expand by itself in the user's crate).
#[track_caller]
fn assert_module_rejected(source: &str, code: Code) -> String {
    let output = expansion(source, Profile::Other).to_string();
    assert!(
        output.starts_with(":: core :: compile_error !"),
        "{output}\nfor {source}"
    );
    assert_rendered_code(&output, code);
    assert!(
        !output.contains("gpu_fn ]"),
        "the attribute is dropped: {output}"
    );
    output
}

/// The functions of the module `outer::inner` of `file`, by name.
fn functions_of(file: &syn::File, outer: &str, inner: &str) -> Vec<String> {
    let syn::Item::Mod(module) = file
        .items
        .iter()
        .find(|item| matches!(item, syn::Item::Mod(module) if module.ident == outer))
        .unwrap_or_else(|| panic!("no module {outer}"))
    else {
        unreachable!()
    };
    let (_, items) = module.content.as_ref().unwrap();
    let seen: Vec<String> = items
        .iter()
        .map(|item| match item {
            syn::Item::Mod(m) => format!("mod {}", m.ident),
            syn::Item::Fn(f) => format!("fn {}", f.sig.ident),
            other => format!("{:?}", std::mem::discriminant(other)),
        })
        .collect();
    let Some(syn::Item::Mod(inner)) = items
        .iter()
        .find(|item| matches!(item, syn::Item::Mod(module) if module.ident == inner))
    else {
        panic!("no module {outer}::{inner}: {seen:?}");
    };
    let (_, items) = inner.content.as_ref().unwrap();
    items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Fn(function) => Some(function.sig.ident.to_string()),
            _ => None,
        })
        .collect()
}

#[test]
fn a_gpu_fn_in_a_module_gets_its_calls_in_a_module_of_its_name() {
    let file: syn::File = parse2(expansion(RECORDS, Profile::Other)).unwrap();
    for function in ["scale", "total"] {
        let mut names = functions_of(&file, "records", function);
        names.sort();
        assert_eq!(
            names,
            [
                "map",
                "map_async",
                "map_gpu",
                "map_gpu_on",
                "map_on",
                "map_on_async"
            ],
            "{function}"
        );
    }
    let text = expanded(RECORDS, Profile::Other);
    // The kernels are the module's own and are named for their function; the function stays a helper.
    for needle in [
        "modscale{",
        "modtotal{",
        "scale_map_kernel",
        "total_map_kernel",
        "pubfnscale(item:Item,k:f32)->Item",
    ] {
        assert!(text.contains(needle), "{needle}: {text}");
    }
    assert!(
        !text.contains("gpu_fn]pubfn"),
        "the attribute is taken off the helper: {text}"
    );
}

#[test]
fn the_cpu_profile_has_the_cpu_calls_and_the_chunk() {
    let file: syn::File = parse2(expansion(RECORDS, Profile::Cpu)).unwrap();
    let mut names = functions_of(&file, "records", "scale");
    names.sort();
    assert_eq!(
        names,
        ["__gust_map_cpu_chunk", "map", "map_on", "map_on_async"]
    );
}

#[test]
fn a_struct_the_module_does_not_declare_is_e0714() {
    // Outside the module the macro cannot see the struct: a type outside the element types is GUST-E0714, whose message
    // names the struct of the module as the way out.
    for source in [
        "mod m { #[gpu_fn] pub fn f(item: Outside) -> f32 { 1.0 } }",
        "mod m { #[gpu_fn] pub fn f(x: f32) -> super::Outside { x } }",
        "mod m { pub struct W<T> { pub x: T } #[gpu_fn] pub fn f(w: W<f32>) -> f32 { w.x } }",
    ] {
        let output = assert_module_rejected(source, Code::MapSignature);
        assert!(
            output.contains("a struct declared in the same `#[gpu_module]` module"),
            "{output}"
        );
    }
    // A free function has no module to declare it in.
    let output = expansion("pub fn f(item: Item) -> f32 { 1.0 }", Profile::Other).to_string();
    assert_rendered_code(&output, Code::MapSignature);
}

#[test]
fn the_module_refuses_what_a_struct_element_cannot_hold_with_its_own_codes() {
    let module = |fields: &str| {
        format!("mod m {{ pub struct S {{ {fields} }} #[gpu_fn] pub fn f(s: S) -> f32 {{ 1.0 }} }}")
    };
    // A 3-wide field, an array of 3-wide elements, an enum field, a storage gap, a generic struct.
    assert_module_rejected(&module("pub v: Vec3, pub w: f32"), Code::ThreeWideInBuffer);
    assert_module_rejected(&module("pub v: [Vec3; 2]"), Code::ArrayStride);
    assert_module_rejected(&module("pub a: f32, pub v: Vec2"), Code::ImplicitPadding);
    assert_module_rejected(
        "mod m { pub enum E { A, B } pub struct S { pub e: E } #[gpu_fn] pub fn f(s: S) -> f32 { 1.0 } }",
        Code::EnumInResource,
    );
    assert_module_rejected(
        "mod m { pub struct S<T> { pub x: T } #[gpu_fn] pub fn f(x: f32) -> f32 { x } }",
        Code::GenericStruct,
    );
}

#[test]
fn a_gpu_fn_in_a_module_takes_no_arguments_and_no_cfg() {
    assert_module_rejected(
        "mod m { #[gpu_fn(route = \"x\")] pub fn f(x: f32) -> f32 { x } }",
        Code::GpuArguments,
    );
    assert_module_rejected(
        "mod m { #[cfg(test)] #[gpu_fn] pub fn f(x: f32) -> f32 { x } }",
        Code::MapSignature,
    );
}

#[test]
fn the_kernel_name_of_a_module_function_may_not_be_taken() {
    assert_module_rejected(
        "mod m { pub fn f_map_kernel(x: f32) -> f32 { x } #[gpu_fn] pub fn f(x: f32) -> f32 { x } }",
        Code::MapNameCollision,
    );
}

#[test]
fn a_gpu_module_attribute_on_a_function_in_a_module_is_still_e0716() {
    let message = "`#[gpu_module]` goes on a module, not on a function: a map function takes \
                   `#[gpu_fn]`, on a free scalar function or, when it takes or returns a struct, inside the \
                   `#[gpu_module]` module that declares the struct; the other functions of a module are \
                   helpers or `#[kernel]`s";
    for attribute in ["gpu_module", "gust::gpu_module"] {
        let source = format!("mod m {{ #[{attribute}] pub fn f(x: f32) -> f32 {{ x }} }}");
        let output = expansion(&source, Profile::Other).to_string();
        assert!(output.starts_with(":: core :: compile_error !"), "{output}");
        assert_rendered_code(&output, Code::NestedGpuFunction);
        assert!(output.contains(message), "{attribute}: {output}");
    }
}

#[test]
fn a_rejected_module_returns_without_the_nested_attribute() {
    // Left on, the inner `#[gust::gpu_fn]` would expand by itself in the user's crate and add errors that only
    // follow from the first.
    let source = "mod m { static X: u32 = 1; #[gust::gpu_fn] pub fn f(x: f32) -> f32 { x } pub fn g(x: f32) -> f32 { x } }";
    let output = expansion(source, Profile::Other).to_string();
    assert!(output.starts_with(":: core :: compile_error !"), "{output}");
    assert!(!output.contains("# [gust :: gpu_fn]"), "{output}");
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

/// `item` under the attribute of `form`, as the proc macro gets it, with no arguments.
fn under(form: crate::Form, item: &str) -> String {
    crate::gpu_impl_routed(
        form,
        TokenStream::new(),
        item.parse().unwrap(),
        Profile::Other,
        &|_| None,
    )
    .to_string()
}

#[test]
fn gpu_module_on_a_function_is_e0702_and_names_gpu_fn() {
    let output = under(crate::Form::Module, "pub fn f(x: f32) -> f32 { x }");
    assert!(output.starts_with(":: core :: compile_error !"), "{output}");
    assert_rendered_code(&output, Code::ItemNotModule);
    assert!(
        output.contains("`#[gpu_module]` goes on an inline module"),
        "{output}"
    );
    assert!(output.contains("takes `#[gpu_fn]`"), "{output}");
    // The function follows the error as written: no second diagnostic.
    assert!(output.contains("pub fn f"), "{output}");
    assert_eq!(output.matches("[GUST-E").count(), 1, "{output}");
}

#[test]
fn gpu_module_on_a_function_returns_it_without_kernel_and_variant() {
    // `#[kernel]` and `#[variant]` are unknown attributes outside a module: left on, they would be a second error.
    let output = under(
        crate::Form::Module,
        "#[kernel] #[variant(subgroups)] pub fn f(x: f32) -> f32 { x }",
    );
    assert_eq!(output.matches("[GUST-E").count(), 1, "{output}");
    assert!(!output.contains("# [kernel]"), "{output}");
    assert!(!output.contains("variant"), "{output}");
    assert!(output.contains("pub fn f"), "{output}");
}

#[test]
fn gpu_fn_on_a_module_is_e0702_and_names_gpu_module() {
    let output = under(
        crate::Form::Function,
        "mod m { #[kernel] pub fn run() {} pub fn g(x: f32) -> f32 { x } }",
    );
    assert!(output.starts_with(":: core :: compile_error !"), "{output}");
    assert_rendered_code(&output, Code::ItemNotModule);
    assert!(
        output.contains("`#[gpu_fn]` goes on a free scalar function"),
        "{output}"
    );
    assert!(output.contains("takes `#[gpu_module]`"), "{output}");
    // The module follows the error, and the one diagnostic is the only one.
    assert!(output.contains("mod m"), "{output}");
    assert_eq!(output.matches("[GUST-E").count(), 1, "{output}");
}

#[test]
fn each_attribute_on_another_item_names_its_own_form() {
    for (form, own) in [
        (
            crate::Form::Module,
            "`#[gpu_module]` goes on an inline module",
        ),
        (
            crate::Form::Function,
            "`#[gpu_fn]` goes on a free scalar function",
        ),
    ] {
        for source in ["struct S;", "const X: u32 = 1;", "use std::fmt;"] {
            let output = under(form, source);
            assert_rendered_code(&output, Code::ItemNotModule);
            assert!(output.contains(own), "{output}\nfor {source}");
            assert_eq!(output.matches("[GUST-E").count(), 1, "{output}");
        }
    }
}

#[test]
fn a_gpu_fn_written_twice_is_refused() {
    // `lift` takes the first attribute off; a second would expand by itself, so it is the error.
    assert_module_rejected(
        "mod m { #[gpu_fn] #[gpu_fn] pub fn f(x: f32) -> f32 { x } }",
        Code::MapSignature,
    );
}

/// A scratch crate whose manifest declares the route `physics`, removed when dropped.
struct RoutedCrate(std::path::PathBuf);

impl RoutedCrate {
    fn new(name: &str) -> Self {
        let directory =
            std::env::temp_dir().join(format!("gust-mapw1-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("Cargo.toml"),
            "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n\
             [package.metadata.gust.routes]\nphysics = {}\n",
        )
        .unwrap();
        Self(directory)
    }
}

impl Drop for RoutedCrate {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The expansion of the module `RECORDS` under the route `physics` with the variable set to `value` (if any).
fn routed_records(scratch: &RoutedCrate, value: Option<&str>) -> String {
    use crate::expand::{Environment, expand_module_in_routed, routing};
    let lookup = |name: &str| {
        (name == "GUST_ROUTE__demo__physics")
            .then(|| value.map(str::to_owned))
            .flatten()
    };
    let request = routing::Request {
        tag: Some(routing::Tag {
            text: "physics".to_owned(),
            span: proc_macro2::Span::call_site(),
        }),
        variables: &lookup,
    };
    let environment = Environment {
        manifest_dir: Some(scratch.0.clone()),
        export_dir: None,
        package: Some("demo".to_owned()),
        reported_file: None,
    };
    let module: syn::ItemMod = parse_str(RECORDS).unwrap();
    expand_module_in_routed(module, &environment, Profile::Other, &request)
        .unwrap()
        .to_string()
}

#[test]
fn a_routed_module_with_a_gpu_fn_follows_its_route_in_the_host_calls() {
    let scratch = RoutedCrate::new("records");
    // Routed to the GPU (the default): the GPU calls, and the kernel has its GPU target.
    let gpu = routed_records(&scratch, None);
    assert!(gpu.contains("map_gpu"), "{gpu}");
    assert!(gpu.contains("Target :: Gpu"), "{gpu}");
    // Routed to the CPU by the variable: the CPU family of host calls, the chunk body, no GPU API, and the kernel's
    // CPU target.
    let cpu = routed_records(&scratch, Some("cpu"));
    assert!(cpu.contains("Target :: Cpu"), "{cpu}");
    assert!(!cpu.contains("GpuKernel"), "{cpu}");
    assert!(
        !cpu.contains("map_gpu") && !cpu.contains("gust_wgpu"),
        "{cpu}"
    );
    assert!(cpu.contains("__gust_map_cpu_chunk"), "{cpu}");
    assert!(cpu.contains("scale_map_kernel"), "{cpu}");
}

#[test]
fn a_parameter_named_like_an_item_of_the_module_is_refused() {
    // Probed on the 890M before the check: the parameter becomes a global `StructuredBuffer<float> total;` beside the
    // module's `float total(..)`, and slangc refuses the kernel at run time (E30200, conflicting declaration).
    for source in [
        "mod m { #[gpu_fn] pub fn total(x: f32) -> f32 { x } #[gpu_fn] pub fn scale(total: f32, k: f32) -> f32 { total * k } }",
        "mod m { const LIMIT: u32 = 4; #[gpu_fn] pub fn f(LIMIT: u32) -> u32 { LIMIT } }",
        "mod m { pub struct S { pub a: f32 } #[gpu_fn] pub fn f(S: f32) -> f32 { S } }",
    ] {
        let output = assert_module_rejected(source, Code::MapNameCollision);
        assert!(
            output.contains("has the name of an item of the module"),
            "{output}"
        );
    }
    // Another name is fine.
    let ok = expansion(
        "mod m { #[gpu_fn] pub fn total(x: f32) -> f32 { x } #[gpu_fn] pub fn scale(sum: f32, k: f32) -> f32 { sum * k } }",
        Profile::Other,
    )
    .to_string();
    assert!(!ok.starts_with(":: core :: compile_error !"), "{ok}");
}
