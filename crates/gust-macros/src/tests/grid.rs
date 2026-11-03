//! Multi-dimensional dispatch (m7.d): the golden Slang of `tests/fixtures/grid.rs` and what the expansion of its
//! kernels gives the typed call (the `*_extent` forms next to the one-dimensional ones).

use proc_macro2::TokenStream;

use crate::{
    diagnostics::testing::parse_str,
    expand::{Profile, kernel_options},
    regression_tests::assert_golden,
    slang::emit_kernel,
    validate::validate_module,
};

const FIXTURE: &str = include_str!("../../../../tests/fixtures/grid.rs");
const GOLDEN: &str = include_str!("../../../../tests/fixtures/grid.slang");

/// The fixture module as `#[gpu]` receives it (the attribute is the macro's own).
fn fixture_module() -> String {
    FIXTURE.replacen("#[gust::gpu]", "", 1)
}

/// The expansion of the fixture with all whitespace removed, so assertions do not depend on token spacing.
fn expanded() -> String {
    let item: TokenStream = fixture_module().parse().unwrap();
    crate::gpu_impl_with(TokenStream::new(), item, Profile::Other)
        .to_string()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect()
}

#[test]
fn kernels_of_the_grid_fixture_match_the_golden() {
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
    assert_golden(&actual, "grid.slang", &GOLDEN.replace("\r\n", "\n"));
}

#[test]
fn the_golden_shows_a_two_dimensional_workgroup_and_ids_derived_per_axis() {
    let golden = GOLDEN.replace("\r\n", "\n");
    assert_eq!(
        golden.matches("[numthreads(16, 16, 1)]").count(),
        2,
        "{golden}"
    );
    assert_eq!(
        golden.matches("[numthreads(4, 4, 4)]").count(),
        1,
        "{golden}"
    );
    // The ids of `cells` are the dispatch-thread id modulo and over the workgroup size, on every axis.
    assert!(
        golden.contains("uint3 lid = uint3(id.x % 16u, id.y % 16u, id.z % 1u);"),
        "{golden}"
    );
    assert!(
        golden.contains("uint3 gid = uint3(id.x / 16u, id.y / 16u, id.z / 1u);"),
        "{golden}"
    );
}

#[test]
fn every_kernel_of_the_grid_fixture_has_the_extent_forms_of_the_typed_call() {
    let text = expanded();
    // One `call_extent`, `submit_extent` and `dispatch_extent` per kernel, beside the three one-dimensional forms.
    for form in ["call_extent", "submit_extent", "dispatch_extent"] {
        assert_eq!(
            text.matches(&format!("fn{form}<")).count(),
            3,
            "{form} in {text}"
        );
    }
    assert!(text.contains("__gust_extent:::gust::Extent"), "{text}");
    // The descriptor records the workgroup size the extent is divided by.
    assert!(text.contains("workgroup_size:[16u32,16u32,1u32]"), "{text}");
    assert!(text.contains("workgroup_size:[4u32,4u32,4u32]"), "{text}");
}
