//! Unit tests of the validator.

use crate::diagnostics::testing::{assert_code, assert_codes, codes_of_result};

use super::*;

#[test]
fn rejects_cpu_allocation() {
    let module: syn::ItemMod = syn::parse_quote! {
        mod bad {
            #[kernel]
            pub fn bad(ctx: Invocation) {
                let values = Vec::new();
            }
        }
    };
    let error = validate_module(&module).expect_err("Vec must be rejected");
    assert_code(&error, Code::CpuOnlyName);
}

#[test]
fn independent_causes_report_every_code_in_source_order() {
    let module: syn::ItemMod = syn::parse_quote! {
        mod bad {
            #[kernel]
            pub fn bad(ctx: Invocation) {
                loop { }
                while true { }
                let values = Vec::new();
            }
        }
    };
    let error = validate_module(&module).expect_err("three causes must be rejected");
    assert_codes(
        &error,
        &[
            Code::LoopNotSupported,
            Code::WhileNotSupported,
            Code::CpuOnlyName,
        ],
    );
}

#[test]
fn an_accepted_module_reports_no_code() {
    let module: syn::ItemMod = syn::parse_quote! {
        mod good {
            #[kernel]
            pub fn add(ctx: Invocation, mut out: StorageMut<f32>) {
                let i = ctx.global_id().x;
                if i < out.len() { out[i] = 1.0; }
            }
        }
    };
    assert!(codes_of_result(&validate_module(&module)).is_empty());
}

#[test]
fn a_module_without_inline_content_and_stray_items_are_coded() {
    let outline: syn::ItemMod = syn::parse_quote! { mod outline; };
    assert_code(
        &validate_module(&outline).expect_err("outline module"),
        Code::InlineModuleRequired,
    );
    let stray: syn::ItemMod = syn::parse_quote! { mod bad { static LIMIT: u32 = 1; } };
    assert_code(
        &validate_module(&stray).expect_err("static item"),
        Code::UnsupportedModuleItem,
    );
}

#[test]
fn signature_restrictions_are_coded() {
    for (module, code) in [
        (
            syn::parse_quote! { mod bad { async fn f() {} } },
            Code::AsyncFunction,
        ),
        (
            syn::parse_quote! { mod bad { unsafe fn f() {} } },
            Code::UnsafeFunction,
        ),
        (
            syn::parse_quote! { mod bad { fn f<const N: usize>(x: u32) {} } },
            Code::ExternOrGenericFunction,
        ),
        (
            syn::parse_quote! { mod bad { extern "C" fn f() {} } },
            Code::ExternOrGenericFunction,
        ),
        (
            syn::parse_quote! { mod bad { struct Pair(uint, uint); } },
            Code::StructNeedsNamedFields,
        ),
    ] {
        let module: syn::ItemMod = module;
        assert_code(&validate_module(&module).expect_err("rejected"), code);
    }
}

#[test]
fn a_derive_that_does_not_parse_is_wrapped_with_a_code() {
    let module: syn::ItemMod = syn::parse_quote! {
        mod bad {
            #[derive = "Debug"]
            struct Params { value: f32 }
        }
    };
    // `syn` words this failure itself; the code is added in front of it.
    let error = validate_module(&module).expect_err("malformed derive");
    assert_code(&error, Code::MalformedDerive);
}

#[test]
fn accepts_vector_add_subset() {
    let module: syn::ItemMod = syn::parse_quote! {
        mod good {
            #[kernel]
            pub fn add(ctx: Invocation, a: Storage<f32>, b: Storage<f32>, mut out: StorageMut<f32>) {
                let i = ctx.global_id().x;
                if i < out.len() {
                    out[i] = a[i] + b[i];
                }
            }
        }
    };
    validate_module(&module).expect("vector add should validate");
}

#[test]
fn rejects_empty_or_manually_represented_gpu_structs() {
    let empty: syn::ItemMod = syn::parse_quote! {
        mod bad {
            struct Empty {}
        }
    };
    assert_code(
        &validate_module(&empty).expect_err("empty GPU struct must fail"),
        Code::EmptyStruct,
    );

    let represented: syn::ItemMod = syn::parse_quote! {
        mod bad {
            #[repr(C)]
            struct Params { value: f32 }
        }
    };
    assert_code(
        &validate_module(&represented).expect_err("manual representation must fail"),
        Code::ManualRepr,
    );

    let copied: syn::ItemMod = syn::parse_quote! {
        mod bad {
            #[derive(Debug, Copy)]
            struct Params { value: f32 }
        }
    };
    assert_code(
        &validate_module(&copied).expect_err("manual Copy derive must fail"),
        Code::ManualCloneCopy,
    );
}

#[test]
fn accepts_non_abi_derives_on_gpu_structs() {
    let module: syn::ItemMod = syn::parse_quote! {
        mod good {
            #[derive(Debug, PartialEq)]
            struct Params { scale: f32, bias: f32 }
        }
    };
    validate_module(&module).expect("non-ABI derives should remain available");
}
