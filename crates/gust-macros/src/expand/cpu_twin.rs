//! The CPU twin of a kernel.

use super::*;

/// The CPU twin of one kernel (roadmap M3): `impl Run<Kernel, N> for ::gust::cpu::CpuDevice` under
/// `cfg(any(test, gust_dual))`, or ungated under [`Profile::Cpu`] (the production CPU target, D21: the same impl with
/// no `cfg` copy, so `cpu` plus `cfg(test)` or `--cfg gust_dual` still has exactly one). It calls the shadow body (the
/// renamed typecheck function) as written, once per invocation, and reaches the bindings only through
/// `::gust::cpu::{validate, read, write, invocations}`: no body copy, no rewrite, no IR and no downcast in the
/// emitted code. `gust/src/cpu.rs` holds a hand-written impl of this exact shape in its tests, which is the
/// specification.
///
/// Nothing is emitted for `cpu = "unsupported"`. A uniform parameter is read by copy (`::gust::cpu::read_uniform`) and
/// passed as `::gust::Uniform(copy)` (D57). Locals take the user's parameter names (the validator reserves `__gust_`,
/// so they cannot collide with `__gust_call` and `__gust_id`).
pub(super) fn cpu_twin_items(
    stage: &dyn StageProfile,
    kernel: &syn::Ident,
    typecheck: &syn::Ident,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
    workgroup_size: [u32; 3],
    cpu: CpuTwin,
    profile: Profile,
) -> syn::Result<TokenStream> {
    if cpu == CpuTwin::Unsupported {
        return Ok(TokenStream::new());
    }
    let mut names = Vec::new();
    let mut sampler_names: Vec<syn::Ident> = Vec::new();
    let mut acquisitions = Vec::new();
    let mut arguments = Vec::new();
    for input in inputs {
        // `parameter_descriptors` has already rejected a receiver or a non-identifier pattern.
        let syn::FnArg::Typed(input) = input else {
            continue;
        };
        let syn::Pat::Ident(pattern) = input.pat.as_ref() else {
            continue;
        };
        let name = &pattern.ident;
        let label = name.to_string();
        match parameter_flavor_in(stage, &input.ty) {
            ParameterFlavor::Invocation => {
                let ty = &input.ty;
                arguments.push(quote! {
                    <#ty as ::gust::cpu::FromGlobalId>::from_global_id(__gust_id)
                });
            }
            // The local and the group id of the invocation, as the GPU derives them from the dispatch-thread id and
            // the workgroup size (slice m7.g1). A kernel with workgroup memory has no twin, so this serves a kernel
            // that only reads the ids.
            ParameterFlavor::LocalId | ParameterFlavor::GroupId => {
                let [x, y, z] = workgroup_size;
                let is_group = parameter_flavor_in(stage, &input.ty) == ParameterFlavor::GroupId;
                // An axis of size 1 is spelled out (`id % 1` is 0, `id / 1` is `id`): clippy denies `modulo_one`
                // in the generated twin, and the default size is `64, 1, 1`.
                let axis = |component: TokenStream, size: u32| {
                    if size == 1 {
                        if is_group { component } else { quote!(0u32) }
                    } else if is_group {
                        quote!(#component / #size)
                    } else {
                        quote!(#component % #size)
                    }
                };
                let (ax, ay, az) = (
                    axis(quote!(__gust_id.x), x),
                    axis(quote!(__gust_id.y), y),
                    axis(quote!(__gust_id.z), z),
                );
                arguments.push(quote! {
                    ::gust::UVec3 {
                        x: #ax,
                        y: #ay,
                        z: #az,
                    }
                });
            }
            // Validation requires `cpu = "unsupported"` of a kernel with workgroup memory, so this is not reached.
            ParameterFlavor::Workgroup => return Ok(TokenStream::new()),
            ParameterFlavor::StorageRead => {
                let element = crate::slang::wrapper_element_type(&input.ty)?;
                names.push(name.clone());
                acquisitions.push(quote! {
                    let #name = ::gust::cpu::read::<#element>(#name, #label)?;
                });
                arguments.push(quote!(::gust::Storage::new(&#name)));
            }
            ParameterFlavor::StorageReadWrite => {
                let element = crate::slang::wrapper_element_type(&input.ty)?;
                names.push(name.clone());
                acquisitions.push(quote! {
                    let mut #name = ::gust::cpu::write::<#element>(#name, #label)?;
                });
                arguments.push(quote!(::gust::StorageMut::new(&mut #name)));
            }
            ParameterFlavor::Uniform => {
                // The uniform is read once, by copy, and each invocation gets `Uniform(copy)`: the shadow body cannot
                // write through it, like the GPU's read-only constant buffer. `Uniform` has `Deref` only, and the
                // validator refuses `mut` on the parameter and `.0` on it (GUST-E0321), the two ways around that.
                let element = crate::slang::wrapper_element_type(&input.ty)?;
                names.push(name.clone());
                acquisitions.push(quote! {
                    let #name = ::gust::cpu::read_uniform::<#element>(#name, #label)?;
                });
                arguments.push(quote!(::gust::Uniform(#name)));
            }
            // A texture is acquired by its texel type and viewed through the kernel-side type over the same texels
            // (`gust/src/cpu/texture.rs` has the hand-written impl of this shape, which is the specification).
            ParameterFlavor::SampledTexture => {
                let spec = crate::slang::texture::texture_spec(
                    parameter_flavor_in(stage, &input.ty),
                    &input.ty,
                )
                .map_err(|message| diagnostics::error(Code::TextureType, &input.ty, message))?;
                let texel = spec.element_type();
                names.push(name.clone());
                acquisitions.push(quote! {
                    let #name = ::gust::cpu::texture::read::<#texel>(#name, #label)?;
                });
                arguments
                    .push(quote!(::gust::Texture2D::new(&#name.texels, #name.width, #name.height)));
            }
            ParameterFlavor::StorageTexture | ParameterFlavor::WriteTexture => {
                let flavor = parameter_flavor_in(stage, &input.ty);
                let spec = crate::slang::texture::texture_spec(flavor, &input.ty)
                    .map_err(|message| diagnostics::error(Code::TextureType, &input.ty, message))?;
                let texel = spec.element_type();
                let marker = spec
                    .format
                    .expect("a storage texture has a format")
                    .marker_tokens();
                let view = if flavor == ParameterFlavor::StorageTexture {
                    quote!(::gust::StorageTexture2D)
                } else {
                    quote!(::gust::WriteTexture2D)
                };
                names.push(name.clone());
                acquisitions.push(quote! {
                    let mut #name = ::gust::cpu::texture::write::<#texel>(#name, #label)?;
                });
                arguments.push(quote!(
                    #view::<#marker>::new(&mut #name.texels, #name.width, #name.height)
                ));
            }
            ParameterFlavor::Sampler => {
                sampler_names.push(name.clone());
                arguments.push(quote!(::gust::cpu::texture::sampler(#name)));
            }
            ParameterFlavor::Value => return Ok(TokenStream::new()),
        }
    }
    let count = proc_macro2::Literal::usize_unsuffixed(names.len());
    // A kernel with samplers is `Run<K, N, [CpuSamplerBinding; S]>`, and reads the sampler bindings of its call.
    let sampler_count = proc_macro2::Literal::usize_unsuffixed(sampler_names.len());
    let (run_trait, call_type, sampler_binding) = if sampler_names.is_empty() {
        (
            quote!(::gust::Run<#kernel, #count>),
            quote!(::gust::Call<'_, Self, #kernel, #count>),
            TokenStream::new(),
        )
    } else {
        (
            quote!(::gust::Run<
                #kernel,
                #count,
                [::gust::cpu::texture::CpuSamplerBinding<'_>; #sampler_count]
            >),
            quote!(::gust::Call<
                '_,
                Self,
                #kernel,
                #count,
                [::gust::cpu::texture::CpuSamplerBinding<'_>; #sampler_count]
            >),
            quote!(let [#(#sampler_names),*] = *__gust_call.samplers();),
        )
    };
    let gate = if profile.is_cpu() {
        TokenStream::new()
    } else {
        quote!(#[cfg(any(test, gust_dual))])
    };
    // How the twin visits the invocations of the call: the shape's harness.
    let invocations = match stage.twin() {
        TwinHarness::Linear => {
            quote!(::gust::cpu::invocations(
                __gust_call.kernel(),
                __gust_call.extent()
            ))
        }
    };
    Ok(quote! {
        #gate
        impl #run_trait for ::gust::cpu::CpuDevice {
            fn submit_call(
                &self,
                __gust_call: &#call_type,
            ) -> ::core::result::Result<::gust::cpu::CpuJob, ::gust::cpu::Error> {
                ::gust::cpu::validate(self, __gust_call)?;
                let [#(#names),*] = *__gust_call.bindings();
                #sampler_binding
                #(#acquisitions)*
                for __gust_id in #invocations {
                    #typecheck(#(#arguments),*);
                }
                ::core::result::Result::Ok(::gust::cpu::CpuJob::complete())
            }
        }
    })
}
