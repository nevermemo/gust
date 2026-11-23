//! The typed `capture` of a kernel (slice g2.cap): copy the inputs of one GPU call and replay it on the CPU twin.

use super::*;

/// The generated `capture` of one kernel whose twin exists. It lives beside the twin and under the twin's gate (a test
/// build of the crate that defines the kernel, or `--cfg gust_dual`); the `cpu` profile has no GPU to capture from, so
/// it gets none. A kernel with a texture or a sampler gets a `capture` that says it cannot capture them yet.
///
/// The body of the replay is the twin's own, built from the same acquisitions and arguments (`TwinParts`), once for
/// each id of an iterator the caller chooses; the twin's `submit_call` itself is left as it is.
pub(super) fn capture_items(
    kernel: &syn::Ident,
    typecheck: &syn::Ident,
    parts: &TwinParts<'_>,
    profile: Profile,
) -> TokenStream {
    if profile.is_cpu() {
        return TokenStream::new();
    }
    let sampled = !parts.sampler_labels.is_empty();
    let first_unsupported = parts
        .resources
        .iter()
        .find(|resource| matches!(resource.kind, TwinResourceKind::Texture))
        .map(|resource| (resource.label.as_str(), "a texture"))
        .or_else(|| {
            parts
                .sampler_labels
                .first()
                .map(|label| (label.as_str(), "a sampler"))
        });
    let count = parts.resources.len();
    if let Some((parameter, what)) = first_unsupported {
        let kernel_text = kernel.to_string();
        return capture_fn(
            kernel,
            count,
            parts.sampler_labels.len(),
            &format!(
                "Capture is not supported yet for `{kernel}`: it has {what} parameter, `{parameter}`. Always returns \
                 `gust::capture::Error::Unsupported`."
            ),
            quote! {
                let _ = (__gust_device, __gust_call);
                ::core::result::Result::Err(::gust::capture::Error::Unsupported {
                    kernel: #kernel_text,
                    parameter: #parameter,
                    what: #what,
                })
            },
        );
    }
    debug_assert!(!sampled, "a sampler is the other unsupported parameter");
    let mut captures = Vec::new();
    let mut rebinds = Vec::new();
    for (index, resource) in parts.resources.iter().enumerate() {
        let label = &resource.label;
        match &resource.kind {
            TwinResourceKind::Read(element) | TwinResourceKind::Write(element) => {
                let access = if matches!(resource.kind, TwinResourceKind::Read(_)) {
                    quote!(::gust::Access::ReadOnly)
                } else {
                    quote!(::gust::Access::ReadWrite)
                };
                captures.push(quote! {
                    ::gust::capture::capture_storage::<__GustD, #element>(
                        __gust_device,
                        &__gust_bindings[#index],
                        #label,
                        &__gust_cpu,
                    )?
                });
                rebinds.push(quote! {
                    ::gust::capture::bind_storage::<#element>(&__gust_slots[#index], #access)?
                });
            }
            TwinResourceKind::Uniform(element) => {
                captures.push(quote! {
                    ::gust::capture::capture_uniform::<__GustD, #element>(
                        __gust_device,
                        &__gust_bindings[#index],
                        #label,
                        &__gust_cpu,
                    )?
                });
                rebinds.push(quote! {
                    ::gust::capture::bind_uniform::<#element>(&__gust_slots[#index])?
                });
            }
            TwinResourceKind::Texture => unreachable!("a texture is the unsupported case"),
        }
    }
    let names = parts.names;
    let acquisitions = parts.acquisitions;
    let arguments = parts.arguments;
    capture_fn(
        kernel,
        count,
        0,
        &format!(
            "Capture the call `__gust_call` of `{kernel}` for replay on its CPU twin: copy every bound buffer and \
             uniform from `__gust_device` as they are now, so call it before the dispatch. The result owns the copies, \
             and `run_element` and `run_until` run chosen invocations of the twin over them (a debugger can break in \
             the kernel body there). A replay reproduces the twin's order, not the GPU's schedule: for a race-free \
             kernel without atomics that equals the GPU's result; with atomics or an order-dependent result it is one \
             member of the allowed set, not the history the GPU had. A read-only buffer can be captured only on a \
             device opened with capture-capable allocation (`DeviceOptions::with_capture`). Textures and samplers are \
             not supported yet."
        ),
        quote! {
            #[allow(unused_variables)]
            fn __gust_replay(
                __gust_cpu: &::gust::cpu::CpuDevice,
                __gust_extent: ::gust::Extent,
                __gust_slots: &[::gust::capture::Slot],
                __gust_ids: &mut dyn ::core::iter::Iterator<Item = ::gust::UVec3>,
            ) -> ::core::result::Result<(), ::gust::cpu::Error> {
                let __gust_call =
                    ::gust::Call::<::gust::cpu::CpuDevice, #kernel, #count>::new_extent(
                        __gust_extent,
                        [#(#rebinds),*],
                    );
                ::gust::cpu::validate(__gust_cpu, &__gust_call)?;
                let [#(#names),*] = *__gust_call.bindings();
                #(#acquisitions)*
                for __gust_id in __gust_ids {
                    #typecheck(#(#arguments),*);
                }
                ::core::result::Result::Ok(())
            }
            let __gust_cpu = ::gust::cpu::CpuDevice::new();
            let __gust_bindings = __gust_call.bindings();
            let __gust_slots = ::std::vec![#(#captures),*];
            ::core::result::Result::Ok(::gust::capture::Captured::new(
                __gust_cpu,
                __gust_call.extent(),
                __gust_slots,
                __gust_replay,
            ))
        },
    )
}

/// The `capture` of a kernel with `cpu = "unsupported"`: there is no twin, so it always returns
/// `gust::capture::Error::NoTwin` (and the call is still type-checked like every other kernel's). Nothing for a kernel
/// that has no typed call (a plain value parameter, or a sampler without a resource) or under the `cpu` profile.
pub(super) fn no_twin_items(
    stage: &dyn StageProfile,
    kernel: &syn::Ident,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
    profile: Profile,
) -> syn::Result<TokenStream> {
    if profile.is_cpu() {
        return Ok(TokenStream::new());
    }
    let (mut resources, mut samplers) = (0_usize, 0_usize);
    for input in inputs {
        let syn::FnArg::Typed(input) = input else {
            continue;
        };
        match parameter_flavor_in(stage, &input.ty) {
            ParameterFlavor::Sampler => samplers += 1,
            ParameterFlavor::StorageRead
            | ParameterFlavor::StorageReadWrite
            | ParameterFlavor::Uniform
            | ParameterFlavor::SampledTexture
            | ParameterFlavor::StorageTexture
            | ParameterFlavor::WriteTexture => resources += 1,
            ParameterFlavor::Value => return Ok(TokenStream::new()),
            ParameterFlavor::Invocation
            | ParameterFlavor::LocalId
            | ParameterFlavor::GroupId
            | ParameterFlavor::Workgroup => {}
        }
    }
    // The typed call of such a kernel does not exist (see `typed_call_items`), so there is nothing to capture.
    if resources == 0 && samplers > 0 {
        return Ok(TokenStream::new());
    }
    let kernel_text = kernel.to_string();
    Ok(capture_fn(
        kernel,
        resources,
        samplers,
        &format!(
            "`{kernel}` is `cpu = \"unsupported\"`, so it has no CPU twin to replay on: always returns \
             `gust::capture::Error::NoTwin`."
        ),
        quote! {
            let _ = (__gust_device, __gust_call);
            ::core::result::Result::Err(::gust::capture::Error::NoTwin {
                kernel: #kernel_text,
            })
        },
    ))
}

/// The signature of `capture` over the typed call of `count` resources and `samplers` samplers, with `body`.
fn capture_fn(
    kernel: &syn::Ident,
    count: usize,
    samplers: usize,
    doc: &str,
    body: TokenStream,
) -> TokenStream {
    let count = proc_macro2::Literal::usize_unsuffixed(count);
    let (call_type, bounds) = if samplers == 0 {
        (
            quote!(::gust::Call<'r, __GustD, Self, #count>),
            quote!(::gust::capture::CaptureDevice),
        )
    } else {
        let samplers = proc_macro2::Literal::usize_unsuffixed(samplers);
        (
            quote!(::gust::SampledCall<'r, __GustD, Self, #count, #samplers>),
            quote!(::gust::capture::CaptureDevice + ::gust::SamplerDevice),
        )
    };
    quote! {
        #[cfg(any(test, gust_dual))]
        impl #kernel {
            #[doc = #doc]
            #[allow(unused_variables, clippy::type_complexity)]
            pub fn capture<'r, __GustD>(
                __gust_device: &__GustD,
                __gust_call: &#call_type,
            ) -> ::core::result::Result<
                ::gust::capture::Captured<Self>,
                ::gust::capture::Error<<__GustD as ::gust::Device>::Error>,
            >
            where
                __GustD: #bounds,
            {
                #body
            }
        }
    }
}
