//! Kernel parameters: facts, bindings and descriptors.

use super::*;

/// What the expander knows about one kernel parameter. The descriptor tokens ([`parameter_descriptors`]) and the
/// export record ([`parameter_records`]) are both made from it, so they cannot disagree about a binding.
pub(super) struct ParameterFacts {
    pub(super) name: String,
    pub(super) rust_type: String,
    pub(super) flavor: ParameterFlavor,
    /// The binding in descriptor set 0 of a storage or uniform parameter, numbered in declaration order.
    pub(super) binding: Option<u32>,
    /// The element type of a storage parameter.
    pub(super) element: Option<syn::Type>,
    /// The struct type of a uniform parameter.
    pub(super) uniform_element: Option<syn::Type>,
    /// The facts of a texture parameter.
    pub(super) texture: Option<TextureParameter>,
}

/// What the descriptor and the export record say about a texture parameter (mx.t3, D64, D91).
pub(super) struct TextureParameter {
    pub(super) access: crate::slang::texture::Access,
    pub(super) format: Option<&'static crate::slang::texture::FormatInfo>,
    /// The Rust type of one texel as the descriptor names it: the user's type for a sampled texture, the format's
    /// texel for a storage one.
    pub(super) element: syn::Type,
    /// The same type by its bare name, which is what the record's layout writer reads.
    pub(super) record_element: syn::Type,
    pub(super) scalar: crate::slang::texture::Scalar,
}

pub(super) fn parameter_facts(
    stage: &dyn StageProfile,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
) -> syn::Result<Vec<ParameterFacts>> {
    let rules = stage.binding_rules();
    let mut next_binding = 0_u32;
    inputs
        .iter()
        .map(|input| {
            let syn::FnArg::Typed(input) = input else {
                return Err(diagnostics::error(
                    Code::SelfParameter,
                    input,
                    "GPU functions cannot use self",
                ));
            };
            let syn::Pat::Ident(pattern) = input.pat.as_ref() else {
                return Err(diagnostics::error(
                    Code::ParameterNeedsIdentifier,
                    &input.pat,
                    "GPU parameters need identifier names",
                ));
            };
            let flavor = parameter_flavor_in(stage, &input.ty);
            // The ids derived from the dispatch-thread id and the workgroup memory are no parameter of the entry
            // point: neither the descriptor nor the export record lists them.
            if !flavor.in_descriptor() {
                return Ok(None);
            }
            let binding = rules.gets_slot(flavor).then(|| {
                let binding = next_binding;
                next_binding += 1;
                binding
            });
            let element = if matches!(
                flavor,
                ParameterFlavor::StorageRead | ParameterFlavor::StorageReadWrite
            ) {
                Some(crate::slang::wrapper_element_type(&input.ty)?.clone())
            } else {
                None
            };
            let uniform_element = if flavor == ParameterFlavor::Uniform {
                Some(crate::slang::wrapper_element_type(&input.ty)?.clone())
            } else {
                None
            };
            let texture = if flavor.is_texture() {
                let spec = crate::slang::texture::texture_spec(flavor, &input.ty)
                    .map_err(|message| diagnostics::error(Code::TextureType, &input.ty, message))?;
                let (element, record_element) = match (spec.sampled_texel, spec.format) {
                    (Some(texel), _) => (texel.clone(), crate::slang::texture::leaf_type(texel)),
                    (None, Some(format)) => (format.texel_type(), format.texel_leaf_type()),
                    (None, None) => unreachable!("a texture has a texel type or a format"),
                };
                Some(TextureParameter {
                    access: spec.access,
                    format: spec.format,
                    element,
                    record_element,
                    scalar: spec.scalar,
                })
            } else {
                None
            };
            Ok(Some(ParameterFacts {
                name: crate::slang::emit_ident(&pattern.ident),
                rust_type: input.ty.to_token_stream().to_string(),
                flavor,
                binding,
                element,
                uniform_element,
                texture,
            }))
        })
        .filter_map(Result::transpose)
        .collect()
}

/// The name and descriptor binding of each parameter the descriptor lists, for the test that the emitter numbers the
/// same slots.
#[cfg(test)]
pub(crate) fn parameter_bindings(
    stage: &dyn StageProfile,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
) -> syn::Result<Vec<(String, Option<u32>)>> {
    Ok(parameter_facts(stage, inputs)?
        .into_iter()
        .map(|fact| (fact.name, fact.binding))
        .collect())
}

pub(super) fn parameter_descriptors(
    stage: &dyn StageProfile,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
) -> syn::Result<Vec<TokenStream>> {
    parameter_facts(stage, inputs)?
        .into_iter()
        .map(|fact| {
            let ParameterFacts {
                name,
                rust_type,
                flavor,
                binding,
                element,
                uniform_element,
                texture,
            } = fact;
            let (kind, access) = match flavor {
                ParameterFlavor::Invocation => (
                    quote!(::gust::ParameterKind::Invocation),
                    quote!(::gust::Access::NotApplicable),
                ),
                ParameterFlavor::StorageRead => (
                    quote!(::gust::ParameterKind::Storage),
                    quote!(::gust::Access::ReadOnly),
                ),
                ParameterFlavor::StorageReadWrite => (
                    quote!(::gust::ParameterKind::Storage),
                    quote!(::gust::Access::ReadWrite),
                ),
                ParameterFlavor::Uniform => (
                    quote!(::gust::ParameterKind::Uniform),
                    quote!(::gust::Access::ReadOnly),
                ),
                ParameterFlavor::Value => (
                    quote!(::gust::ParameterKind::Value),
                    quote!(::gust::Access::NotApplicable),
                ),
                // A sampled texture is read-only; a storage or write-only one is bound read-write (D64).
                ParameterFlavor::SampledTexture => (
                    quote!(::gust::ParameterKind::Texture),
                    quote!(::gust::Access::ReadOnly),
                ),
                ParameterFlavor::StorageTexture | ParameterFlavor::WriteTexture => (
                    quote!(::gust::ParameterKind::Texture),
                    quote!(::gust::Access::ReadWrite),
                ),
                ParameterFlavor::Sampler => (
                    quote!(::gust::ParameterKind::Sampler),
                    quote!(::gust::Access::NotApplicable),
                ),
                ParameterFlavor::LocalId
                | ParameterFlavor::GroupId
                | ParameterFlavor::Workgroup => {
                    unreachable!("`parameter_facts` leaves out what is not in the descriptor")
                }
            };
            let binding = match binding {
                Some(binding) => {
                    quote!(Some(::gust::ResourceBinding { group: 0, binding: #binding }))
                }
                None => quote!(None),
            };
            let resource_layout = match (element, uniform_element) {
                // A texture: its texel type is the element, and the facts the shader witnesses (D91) follow. The
                // element is named once, so a texel type that is no `GpuPod` still reports one E0277.
                (None, None) if texture.is_some() => {
                    let texture = texture.expect("guarded by the match arm");
                    let element = &texture.element;
                    let format = match texture.format {
                        Some(format) => {
                            let tokens = format.format_tokens();
                            quote!(::core::option::Option::Some(#tokens))
                        }
                        None => quote!(::core::option::Option::None),
                    };
                    let access = texture.access.tokens();
                    let class = texture.scalar.class_tokens();
                    quote!(Some(::gust::ResourceLayout::texture_v1(
                        <#element as ::gust::GpuPod>::LAYOUT,
                        ::gust::TextureLayout {
                            format: #format,
                            dimension: ::gust::TextureDimension::D2,
                            access: #access,
                            class: #class,
                        }
                    )))
                }
                // StorageV1 when the element is scalars and padding-free structs of them, StorageV2 when it holds
                // vectors or matrices. The element is named once, so a type that is no `GpuPod` still reports one
                // E0277.
                (Some(element), _) => quote!(Some({
                    const __GUST_ELEMENT: ::gust::TypeLayout = <#element as ::gust::GpuPod>::LAYOUT;
                    if __GUST_ELEMENT.is_storage_v1() {
                        ::gust::ResourceLayout::storage_v1(__GUST_ELEMENT)
                    } else {
                        ::gust::ResourceLayout::storage_v2(__GUST_ELEMENT)
                    }
                })),
                // A uniform is `Std140V1` always. Whether the type really is one is asserted next to the descriptor
                // (`element_layout_assertions`), with the same single E0277 for a type that is no `GpuPod`.
                (None, Some(element)) => quote!(Some(::gust::ResourceLayout::std140_v1(
                    <#element as ::gust::GpuPod>::LAYOUT
                ))),
                (None, None) => quote!(None),
            };
            Ok(quote! {
                ::gust::ParameterDescriptor {
                    name: #name,
                    rust_type: #rust_type,
                    kind: #kind,
                    access: #access,
                    resource_layout: #resource_layout,
                    binding: #binding,
                }
            })
        })
        .collect()
}

/// An argument of a typed call in declaration order: the index into the resources or into the samplers.
#[derive(Clone, Copy)]
pub(super) enum CallParameter {
    Resource(usize),
    Sampler(usize),
}

/// One storage, uniform or texture parameter of a kernel, as the typed call needs it.
pub(super) struct ResourceArgument {
    pub(super) name: syn::Ident,
    pub(super) element: syn::Type,
    pub(super) access: TokenStream,
    /// `::gust::kind::Storage`, `Uniform` or `Texture`: the `Kind` the bound of the generated call names.
    pub(super) kind: TokenStream,
    /// Whether the parameter is a texture (its `kind` is `::gust::kind::Texture`).
    pub(super) texture: bool,
}
