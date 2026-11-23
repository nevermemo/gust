//! The items that check a kernel's artifacts and carry its parameter records.

use super::*;

/// The profile checks of one kernel, all spanned at the kernel so that a strict build fails at the kernel's own line in
/// the user's file, not inside `gust`:
/// - no artifact: `require_artifact(false)`, which a strict profile (D44) turns into a compile error;
/// - a complete pair: `require_artifact(true)`. The module as a whole then also gets one
///   `require_manifest(<the manifest exists>)` (see `expand_in`), which a strict profile turns into the "run
///   `cargo gust init` and `cargo gust build`" error (R4.3). A kernel without an artifact reports only the artifact,
///   so that one cause is one error; its fix, `cargo gust build`, writes the manifest too;
/// - half a pair (outside export mode, where it is absent): a coded `compile_error!` naming the missing file and the
///   one to delete, and nothing else, so that it is never a silent `&[]`.
///
/// Under `dev` the profile checks do nothing. They are emitted in every expansion.
pub(super) fn artifact_check_items(lookup: &Lookup, span: proc_macro2::Span) -> TokenStream {
    match lookup {
        Lookup::Absent => require_artifact_item(false, span),
        Lookup::Present => require_artifact_item(true, span),
        Lookup::Incomplete { left, missing } => artifact::error_item(
            Code::HalfArtifactPair,
            span,
            &artifact::incomplete_message(left, missing),
        ),
    }
}

/// `gust::profile::require_artifact(present)` in a `const _`. Under `dev` it does nothing.
pub(super) fn require_artifact_item(present: bool, span: proc_macro2::Span) -> TokenStream {
    quote_spanned! {span=>
        const _: () = ::gust::profile::require_artifact(#present);
    }
}

/// `gust::profile::require_manifest(present)` in a `const _`. Under `dev` it does nothing.
pub(super) fn require_manifest_item(present: bool, span: proc_macro2::Span) -> TokenStream {
    quote_spanned! {span=>
        const _: () = ::gust::profile::require_manifest(#present);
    }
}

/// The parameters of a kernel as `kernel.json` records them: the facts [`parameter_descriptors`] turns into tokens,
/// plus the element layout of each storage parameter (`structs`, keyed as the source spells the names) and of each
/// uniform parameter (`structs_by_unraw_name`, keyed as [`crate::uniform::module_structs`] keys them).
pub(super) fn parameter_records(
    stage: &dyn StageProfile,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
    structs: &HashMap<String, &syn::ItemStruct>,
    structs_by_unraw_name: &HashMap<String, &syn::ItemStruct>,
) -> syn::Result<Vec<artifact::ParameterRecord>> {
    Ok(parameter_facts(stage, inputs)?
        .into_iter()
        .map(|fact| {
            let (kind, access) = match fact.flavor {
                ParameterFlavor::Invocation => ("invocation", "not_applicable"),
                ParameterFlavor::StorageRead => ("storage", "read_only"),
                ParameterFlavor::StorageReadWrite => ("storage", "read_write"),
                ParameterFlavor::Uniform => ("uniform", "read_only"),
                ParameterFlavor::Value => ("value", "not_applicable"),
                ParameterFlavor::SampledTexture => ("texture", "read_only"),
                ParameterFlavor::StorageTexture | ParameterFlavor::WriteTexture => {
                    ("texture", "read_write")
                }
                ParameterFlavor::Sampler => ("sampler", "not_applicable"),
                ParameterFlavor::LocalId
                | ParameterFlavor::GroupId
                | ParameterFlavor::Workgroup => {
                    unreachable!("`parameter_facts` leaves out what is not in the descriptor")
                }
            };
            // The numbers are the ones the `const` assertions of the expansion check against the real types (D80): the
            // storage layout is the running sum of the field sizes, the uniform layout is `uniform::std140_struct`.
            // A texture's layout is its texel type and the texture facts (D91).
            let layout = match (&fact.element, &fact.uniform_element, &fact.texture) {
                (Some(element), _, _) => artifact::storage_resource(element, structs),
                (None, Some(element), _) => {
                    artifact::uniform_resource(element, structs_by_unraw_name)
                }
                (None, None, Some(texture)) => artifact::texture_resource(
                    &texture.record_element,
                    artifact::TextureRecord {
                        format: texture.format.map(|format| format.name),
                        access: texture.access.record_name(),
                        class: texture.scalar.name(),
                    },
                ),
                (None, None, None) => None,
            };
            artifact::ParameterRecord {
                name: fact.name,
                rust_type: fact.rust_type,
                kind,
                access,
                binding: fact.binding,
                layout,
            }
        })
        .collect())
}

/// Reads the two variables that decide what a kernel crate's expansion contains, so that rustc records them as
/// dependencies of the crate (`option_env!` is tracked) and Cargo re-expands the `#[gpu_module]` module when either one
/// changes: `GUST_PROFILE` (the profile, D44) and `GUST_EXPORT_DIR` (export mode, D46). The values are not used.
pub(super) fn tracked_environment() -> TokenStream {
    quote! {
        const _: ::core::option::Option<&::core::primitive::str> = ::core::option_env!("GUST_PROFILE");
        const _: ::core::option::Option<&::core::primitive::str> = ::core::option_env!("GUST_EXPORT_DIR");
    }
}
