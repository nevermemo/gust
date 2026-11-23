//! Tests of the descriptor, the pod records and the artifact checks.

use super::*;

/// Not reachable through `#[gpu_module]`: the validator rejects a struct without named fields
/// (E0104) before the host side is generated. Listed in the guard's `UNREACHABLE`.
#[test]
fn pod_struct_without_named_fields_is_coded_when_validation_is_skipped() {
    let item: syn::ItemStruct = syn::parse_quote! { struct P(u32, u32); };
    crate::diagnostics::testing::assert_code(
        &gpu_pod_impl(&item, &PodRules::Storage).unwrap_err(),
        Code::PodStructNamedFields,
    );
}

#[test]
fn a_structs_asserts_share_one_const_after_its_impl_in_a_fixed_order() {
    let item: syn::ItemStruct = syn::parse_quote! { struct Holder { a: uint, w: Wide8 } };
    let text = gpu_pod_impl(&item, &PodRules::Storage).unwrap().to_string();
    // One block: one cause is one E0080.
    assert_eq!(text.matches("const _").count(), 1, "{text}");
    let at = |needle: &str| {
        text.find(needle)
            .unwrap_or_else(|| panic!("no `{needle}` in {text}"))
    };
    let order = [
        ":: gust :: GpuPod for Holder",
        "const _",
        "field `a` of `Holder` is not at its padding-free offset",
        "field `w` of `Holder` is not at its padding-free offset",
        "`Holder` has tail padding",
        "`Holder` must have the largest alignment of its fields",
        "`Holder` layout is not StorageV2",
    ];
    let positions: Vec<usize> = order.iter().map(|needle| at(needle)).collect();
    assert!(
        positions.windows(2).all(|pair| pair[0] < pair[1]),
        "{positions:?} for {order:?} in {text}"
    );
    // The unnamed size-sum assert carries a message too, so no bare `assertion failed` can reach a user.
    assert!(
            text.contains("== 0 + :: core :: mem :: size_of :: < uint > () + :: core :: mem :: size_of :: < Wide8 > () , \"GUST StorageV2: `Holder` has tail padding"),
            "{text}"
        );
    assert!(!text.contains("u32 :: MAX as usize ) ;"), "{text}");
}

#[test]
fn element_text_keeps_the_spaces_that_mean_something() {
    let text = |ty: syn::Type| type_text(&ty);
    assert_eq!(text(syn::parse_quote!(super::Outside)), "super::Outside");
    assert_eq!(text(syn::parse_quote!(Vec<Vec<u32>>)), "Vec<Vec<u32>>");
    assert_eq!(
        text(syn::parse_quote!(Box<dyn core::fmt::Debug>)),
        "Box<dyn core::fmt::Debug>"
    );
}

#[test]
fn descriptor_contains_slang_not_custom_ir_or_spirv() {
    let module: syn::ItemMod = syn::parse_quote! {
        mod layout {
            #[kernel]
            pub fn inspect(id: SV_DispatchThreadID, values: StructuredBuffer<f32>) {
                let i = id.x;
            }
        }
    };
    let output = expand(module).unwrap().to_string();
    assert!(output.contains("slang_source"));
    assert!(!output.contains("spirv_words"));
    assert!(!output.contains("fn cpu"));
}

#[test]
fn descriptor_origin_is_unknown_when_the_compiler_gave_no_real_file() {
    // Outside a compiler invocation spans carry no real path, so the table is
    // dropped: lines without a file cannot be reported.
    let module: syn::ItemMod = crate::diagnostics::testing::parse_str(
            "mod layout {\n#[kernel]\npub fn inspect(id: SV_DispatchThreadID, values: StructuredBuffer<f32>) {\nlet i = id.x;\n}\n}",
        )
        .unwrap();
    let output = expand(module).unwrap().to_string();
    assert!(output.contains("source : :: gust :: SourceOrigin :: UNKNOWN"));
    assert!(!output.contains("LineSpan"));
}

#[test]
fn every_kernel_checks_its_artifact_spanned_at_the_kernel() {
    // `first` is on line 3, `second` on line 7.
    let module: syn::ItemMod = crate::diagnostics::testing::parse_str(
            "mod layout {\n#[kernel]\npub fn first(id: SV_DispatchThreadID, values: StructuredBuffer<f32>) {\nlet i = id.x;\n}\n#[kernel]\npub fn second(id: SV_DispatchThreadID, values: StructuredBuffer<f32>) {\nlet i = id.x;\n}\n}",
        )
        .unwrap();
    let output = expand(module).unwrap();
    let text = output.to_string();
    // No `gust-artifacts/` exists in this crate, so the argument is `false`; one check per kernel, none per module.
    assert_eq!(
        text.matches("const _ : () = :: gust :: profile :: require_artifact (false) ;")
            .count(),
        2,
        "{text}"
    );
    assert_eq!(text.matches("require_artifact").count(), 2, "{text}");
    // Each check carries the line of its own kernel, which is where a strict build reports the error.
    let mut lines = Vec::new();
    ident_lines(output, "require_artifact", &mut lines);
    assert_eq!(lines, [3, 7], "{text}");
}

#[test]
fn the_checks_use_the_value_the_artifacts_field_reports() {
    let artifacts = artifact::artifact_tokens("0123456789abcdef", &Lookup::Absent);
    assert_eq!(artifacts.to_string(), "& []");
    let item = require_artifact_item(true, proc_macro2::Span::call_site()).to_string();
    assert_eq!(
        item,
        "const _ : () = :: gust :: profile :: require_artifact (true) ;"
    );
    let item = require_manifest_item(false, proc_macro2::Span::call_site()).to_string();
    assert_eq!(
        item,
        "const _ : () = :: gust :: profile :: require_manifest (false) ;"
    );
}
