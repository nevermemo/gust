//! The Slang emitter: turns a validated `#[gpu]` module and one of its kernels into Slang text and a line table.
//!
//! The entry points are [`emit_kernel_in`] (and the test-only `emit_kernel`); the other files hold the emitter by
//! concern, and this one declares them and re-exports what the rest of the crate names.
//!
//! - `kernel`: the entry point, one kernel of a module into Slang
//! - `declarations`: enums, structs, interfaces, extensions, helper functions and functions
//! - `statements`: statements, `let` bindings, struct values and repeat initialisers
//! - `control`: `if`, `match` and `for`, with their case labels and variant paths
//! - `value_if`: `if` and `match` in value position, and the type inference of their branches
//! - `expressions`: expressions and literals
//! - `operators`: operator predicates, division helper selection and the `__gust_keep` rule
//! - `calls`: math, subgroup and atomic calls, constructors and argument lists
//! - `types`: identifiers, parameter flavors and the Slang spelling of Rust types
//! - `closures`: closure bindings, their hoisted functions and scopes
//! - `arrays`: array lengths and the tables that hold them
//! - `lowering`: the whole-module lowerings and the `Self` scope
//! - `helpers`: the text of the helper functions a kernel defines on use
//! - `usage`: the one walk that finds which helpers a kernel needs
//! - `records`: the line records and the table that builds them
//! - `collisions`, `consts`, `div_table`, `guards`, `mul_table`, `neg_table`, `texture`: existing tables and rules

#[cfg(test)]
mod collision_tests;
pub(crate) mod collisions;
pub(crate) mod consts;
pub(crate) mod div_table;
pub(crate) mod guards;
pub(crate) mod mul_table;
pub(crate) mod neg_table;
pub(crate) mod texture;

mod arrays;
mod calls;
mod closures;
mod control;
mod declarations;
mod expressions;
mod helpers;
mod kernel;
mod lowering;
mod operators;
mod records;
mod statements;
mod types;
mod usage;
mod value_if;

use expressions::{emit_expression, emit_literal, integer_literal};
use types::emit_type;

pub(crate) use arrays::{
    array_field_lengths, array_length_literal, array_lengths, array_place_length,
};
pub(crate) use closures::{closure_binding, closure_function, closure_sites};
#[cfg(test)]
pub use kernel::emit_kernel;
pub(crate) use kernel::emit_kernel_in;
#[cfg(test)]
pub(crate) use operators::is_pure_place;
#[cfg(test)]
pub use records::EmittedKernel;
pub use records::LineRecord;
pub use statements::local_binding;
pub(crate) use statements::tuple_pattern;
pub use types::{ParameterFlavor, parameter_flavor, wrapper_element_type};
pub(crate) use types::{emit_ident, parameter_flavor_in, type_name};

#[cfg(test)]
mod tests {
    use super::types::typed_argument;
    use super::*;
    use crate::{diagnostics::Code, expand::KernelOptions};

    #[test]
    fn emits_slang_resources_semantic_and_body() {
        let module: syn::ItemMod = syn::parse_quote! {
            mod vector_add {
                #[kernel(workgroup_size(64, 1, 1))]
                fn add(id: SV_DispatchThreadID, a: StructuredBuffer<float>, mut out: RWStructuredBuffer<float>) {
                    let i = id.x;
                    if i < out.len() {
                        out[i] = a[i] + 1.0;
                    }
                }
            }
        };
        let (_, items) = module.content.as_ref().unwrap();
        let kernel = items
            .iter()
            .find_map(|item| match item {
                syn::Item::Fn(function) => Some(function),
                _ => None,
            })
            .unwrap();
        let source = emit_kernel(
            &module,
            kernel,
            KernelOptions {
                workgroup_size: [64, 1, 1],
            },
        )
        .unwrap()
        .source;
        assert!(source.contains("StructuredBuffer<float> a;"));
        assert!(source.contains("RWStructuredBuffer<float> out;"));
        assert!(source.contains("void gpu_vector_add_add(uint3 id : SV_DispatchThreadID)"));
        assert!(source.contains("out[i] = (a[i] + 1.0);"));
        assert!(source.contains("i < __gpu_len_out"));
    }

    /// The emitter re-checks shapes the validator already rejects, so these sites are
    /// unreachable through `#[gpu]`; they are exercised on the emitter functions directly
    /// and must still carry their own codes.
    #[test]
    fn emitter_rejections_carry_their_codes() {
        use crate::diagnostics::testing::assert_code;

        let ty = |ty: syn::Type| emit_type(&ty).unwrap_err();
        assert_code(&ty(syn::parse_quote!([u32; 2])), Code::TypeNeedsPath);
        assert_code(
            &ty(syn::parse_quote!(Result<'a>)),
            Code::ResultPayloadMissing,
        );

        let argument: syn::FnArg = syn::parse_quote!(&self);
        assert_code(&typed_argument(&argument).unwrap_err(), Code::SelfParameter);
        let argument: syn::FnArg = syn::parse_quote!((a, b): u32);
        assert_code(
            &typed_argument(&argument).unwrap_err(),
            Code::ParameterNeedsIdentifier,
        );

        let element = |ty: syn::Type| wrapper_element_type(&ty).unwrap_err();
        assert_code(
            &element(syn::parse_quote!(*const u32)),
            Code::ResourceElementPath,
        );
        assert_code(
            &element(syn::parse_quote!(StructuredBuffer)),
            Code::ResourceElementArity,
        );
        assert_code(
            &element(syn::parse_quote!(StructuredBuffer<'a>)),
            Code::ResourceElementMissing,
        );

        let expression = |expression: syn::Expr| emit_expression(&expression).unwrap_err();
        assert_code(
            &expression(syn::parse_quote!(atomic_add(counter, 1u32))),
            Code::AtomicEmitterReference,
        );
        assert_code(
            &expression(syn::parse_quote!(atomic_add(&mut counter, 1u32))),
            Code::AtomicEmitterElement,
        );
        assert_code(
            &expression(syn::parse_quote!(|x| x)),
            Code::UnsupportedExpression,
        );
        assert_code(
            &expression(syn::parse_quote!('a')),
            Code::UnsupportedLiteral,
        );

        let pattern: syn::Pat = syn::parse_quote!((a, b));
        assert_code(&local_binding(&pattern).unwrap_err(), Code::LetBindingForm);
    }
}
