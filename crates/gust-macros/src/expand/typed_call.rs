//! The typed host call of a kernel.

use super::*;

/// The typed host API of one kernel (roadmap M2): `call`, `submit` and `dispatch` on the handle, plus a
/// non-generic constant assertion that the generated argument count equals the descriptor's number of
/// storage plus uniform parameters (so `cargo check` catches a mismatch, not only monomorphisation).
///
/// Each storage or uniform parameter becomes one `impl Into<Arg<'r, Ri>>` argument, in declaration order, under its
/// own name. `Ri` must be a `gust::Resource` of the parameter's element type and of the device of the
/// first argument, and of the parameter's kind: `Kind = ::gust::kind::Storage` for a buffer and `Kind =
/// ::gust::kind::Uniform` for a uniform (D57), so a `GpuUniform` where a buffer is expected, or the other way round,
/// is a compile error and not a refusal at submit. Every `Ri` carries `'r` (S0 decision A: `Arg<'r, Ri>` needs
/// `Ri: 'r`, and rustc does not infer that for a function signature). Generated value names carry the reserved
/// `__gust_` prefix (the validator rejects it in user code) and the generic parameters carry `__Gust`, so a user
/// parameter named `element_count` or `device`, or an element type named `R0` or `D`, cannot collide with them.
///
/// A plain value parameter never reaches this function: the emitter rejects it at expansion (GUST-E0801).
///
/// Returns the doc text for the kernel's handle struct (users read it on `kernel_name`; it states the rules above
/// for the user) and the items.
pub(super) fn typed_call_items(
    stage: &dyn StageProfile,
    kernel: &syn::Ident,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
) -> syn::Result<(String, TokenStream)> {
    let mut arguments = Vec::new();
    // The sampler parameters, and the declaration order of the call's arguments (a resource or a sampler).
    let mut samplers: Vec<syn::Ident> = Vec::new();
    let mut call_order: Vec<CallParameter> = Vec::new();
    for input in inputs {
        // `parameter_descriptors` has already rejected a receiver or a non-identifier pattern.
        let syn::FnArg::Typed(input) = input else {
            continue;
        };
        let syn::Pat::Ident(pattern) = input.pat.as_ref() else {
            continue;
        };
        let flavor = parameter_flavor_in(stage, &input.ty);
        // A texture is a resource of kind `Texture` whose element is its texel type; a sampler is not a resource at
        // all but a `SamplerArg` of the same device (D64).
        let mut texel = None;
        let (access, kind) = match flavor {
            ParameterFlavor::Sampler => {
                call_order.push(CallParameter::Sampler(samplers.len()));
                samplers.push(pattern.ident.clone());
                continue;
            }
            ParameterFlavor::SampledTexture
            | ParameterFlavor::StorageTexture
            | ParameterFlavor::WriteTexture => {
                let spec = crate::slang::texture::texture_spec(flavor, &input.ty)
                    .map_err(|message| diagnostics::error(Code::TextureType, &input.ty, message))?;
                texel = Some(spec.element_type());
                let access = if spec.access == crate::slang::texture::Access::Sampled {
                    quote!(::gust::Access::ReadOnly)
                } else {
                    quote!(::gust::Access::ReadWrite)
                };
                (access, quote!(::gust::kind::Texture))
            }
            // The derived ids and the workgroup memory are not arguments of a call: no buffer stands for them.
            ParameterFlavor::Invocation
            | ParameterFlavor::LocalId
            | ParameterFlavor::GroupId
            | ParameterFlavor::Workgroup => continue,
            ParameterFlavor::StorageRead => (
                quote!(::gust::Access::ReadOnly),
                quote!(::gust::kind::Storage),
            ),
            ParameterFlavor::StorageReadWrite => (
                quote!(::gust::Access::ReadWrite),
                quote!(::gust::kind::Storage),
            ),
            // A uniform is read-only for the kernel and has the kind `Uniform`: a `GpuUniform` is the only resource
            // that fits its bound, and a `GpuBuffer` the only one that fits a storage bound.
            ParameterFlavor::Uniform => (
                quote!(::gust::Access::ReadOnly),
                quote!(::gust::kind::Uniform),
            ),
            // `Value` is defensive: the emitter rejects a plain value parameter (GUST-E0801) before this runs.
            ParameterFlavor::Value => {
                return Ok((
                    format!(
                        "Handle of the GUST kernel `{kernel}`. It has a plain value parameter, which no Slang compute \
                         entry point accepts, so it has no typed `call`, `submit` or `dispatch`."
                    ),
                    TokenStream::new(),
                ));
            }
        };
        let element = match texel {
            Some(texel) => texel,
            None => crate::slang::wrapper_element_type(&input.ty)?.clone(),
        };
        call_order.push(CallParameter::Resource(arguments.len()));
        arguments.push(ResourceArgument {
            name: pattern.ident.clone(),
            element,
            access,
            kind,
            texture: flavor.is_texture(),
        });
    }
    let total = arguments.len();
    let count = proc_macro2::Literal::usize_unsuffixed(total);
    let sampler_total = samplers.len();
    let sampler_count = proc_macro2::Literal::usize_unsuffixed(sampler_total);
    let count_message = format!(
        "`{kernel}`: the generated typed call takes a different number of arguments than its descriptor has storage, \
         uniform and texture parameters, or a different number of samplers than its sampler parameters"
    );
    let names: Vec<String> = call_order
        .iter()
        .map(|parameter| match *parameter {
            CallParameter::Resource(index) => arguments[index].name.to_string(),
            CallParameter::Sampler(index) => samplers[index].to_string(),
        })
        .collect();
    // The wording of a kernel without textures is the one it had before they existed.
    let textured = !samplers.is_empty() || arguments.iter().any(|argument| argument.texture);
    let order = if names.is_empty() {
        "no resource arguments".to_owned()
    } else if textured {
        format!(
            "then the storage, uniform, texture and sampler arguments in declaration order: {}",
            names.join(", ")
        )
    } else {
        format!(
            "then the storage and uniform arguments in declaration order: {}",
            names.join(", ")
        )
    };
    let turbofish = if arguments.is_empty() {
        " The kernel has no resource arguments to infer the device from, so name it: `call::<Device>(..)`."
    } else {
        ""
    };
    let call_doc = format!(
        "Typed call of `{kernel}`: `__gust_element_count`, {order}. A storage argument is a reference to a \
         buffer of the kernel's element type, optionally wrapped in `gust::Independent` to let its length \
         differ from the element count; a uniform argument is a reference to a uniform of the kernel's struct \
         (it has no length).{turbofish}"
    );
    let handle_doc = format!(
        "Handle of the GUST kernel `{kernel}`, with the typed host API `call`, `submit` and `dispatch` \
         (one argument per storage buffer or uniform parameter, in parameter order).{turbofish} \
         Under `cfg(test)` of the crate that defines the kernel, or `--cfg gust_dual`, the kernel also runs on \
         `gust::cpu::CpuDevice` (its CPU twin), unless `#[kernel(cpu = \"unsupported\")]`. Under \
         `GUST_PROFILE=cpu` it runs on `gust::cpu::CpuDevice` in every build of the crate."
    );
    let submit_doc = format!(
        "Submit `{kernel}`: `__gust_device`, `__gust_element_count`, {order}. Returns a job to wait on."
    );
    let dispatch_doc = format!(
        "Run `{kernel}` and wait (on `wasm32`, a device whose job cannot block returns its wait error after \
         submitting): `__gust_device`, `__gust_element_count`, {order}."
    );
    // The multi-dimensional forms (slice m7.d): the same three functions over a `gust::Extent` in place of the element
    // count. The one-dimensional forms are `Extent { x: n, y: 1, z: 1 }`.
    let extent_note = "The extent is `gust::Extent { x, y, z }`: each axis is rounded up to whole workgroups, a shared \
         buffer holds exactly `x * y * z` elements, an axis of 0 runs nothing, and the rounded launch is at most \
         2^32 invocations on every backend.";
    let call_extent_doc = format!(
        "Typed call of `{kernel}` over an extent: `__gust_extent`, {order}. Like `call`, with a `gust::Extent` \
         in place of the element count. {extent_note}{turbofish}"
    );
    let submit_extent_doc = format!(
        "Submit `{kernel}` over an extent: `__gust_device`, `__gust_extent`, {order}. Returns a job to wait on. \
         {extent_note}"
    );
    let dispatch_extent_doc = format!(
        "Run `{kernel}` over an extent and wait (on `wasm32`, a device whose job cannot block returns its wait \
         error after submitting): `__gust_device`, `__gust_extent`, {order}. {extent_note}"
    );

    let assertion = if textured {
        quote! {
            const _: () = ::core::assert!(
                <#kernel as ::gust::Kernel>::DESCRIPTOR.storage_parameter_count()
                    + <#kernel as ::gust::Kernel>::DESCRIPTOR.uniform_parameter_count()
                    + <#kernel as ::gust::Kernel>::DESCRIPTOR.texture_parameter_count()
                    == #count
                    && <#kernel as ::gust::Kernel>::DESCRIPTOR.sampler_parameter_count()
                        == #sampler_count,
                #count_message
            );
        }
    } else {
        quote! {
            const _: () = ::core::assert!(
                <#kernel as ::gust::Kernel>::DESCRIPTOR.storage_parameter_count()
                    + <#kernel as ::gust::Kernel>::DESCRIPTOR.uniform_parameter_count()
                    == #count,
                #count_message
            );
        }
    };

    // A sampler is used through a texture's `sample`, so a kernel with a sampler and no resource has nothing to take
    // the device from: it gets no typed call (its descriptor, record and twin are unaffected).
    if arguments.is_empty() && !samplers.is_empty() {
        return Ok((
            format!(
                "Handle of the GUST kernel `{kernel}`. It has a sampler parameter but no buffer or texture, so it has \
                 no typed `call`, `submit` or `dispatch`."
            ),
            TokenStream::new(),
        ));
    }
    if arguments.is_empty() {
        // The forms over a `gust::Extent`: a shape whose invocations are not a grid has the one-dimensional forms only.
        let extent_items = if stage.binding_rules().extent_calls {
            quote! {
                #[doc = #call_extent_doc]
                pub fn call_extent<'r, __GustD: ::gust::TypedDevice>(
                    __gust_extent: ::gust::Extent,
                ) -> ::gust::Call<'r, __GustD, Self, 0> {
                    ::gust::Call::new_extent(__gust_extent, [])
                }

                #[doc = #submit_extent_doc]
                #[allow(clippy::type_complexity)]
                pub fn submit_extent<'d, __GustD: ::gust::Run<Self, 0>>(
                    __gust_device: &'d __GustD,
                    __gust_extent: ::gust::Extent,
                ) -> ::core::result::Result<
                    <__GustD as ::gust::Device>::Job<'d>,
                    <__GustD as ::gust::Device>::Error,
                > {
                    <__GustD as ::gust::Run<Self, 0>>::submit_call(__gust_device, &Self::call_extent(__gust_extent))
                }

                #[doc = #dispatch_extent_doc]
                pub fn dispatch_extent<__GustD: ::gust::Run<Self, 0>>(
                    __gust_device: &__GustD,
                    __gust_extent: ::gust::Extent,
                ) -> ::core::result::Result<(), <__GustD as ::gust::Device>::Error> {
                    <__GustD as ::gust::Run<Self, 0>>::dispatch_call(
                        __gust_device,
                        &Self::call_extent(__gust_extent),
                    )
                }
            }
        } else {
            TokenStream::new()
        };
        return Ok((
            handle_doc,
            quote! {
                #assertion

                impl #kernel {
                    #[doc = #call_doc]
                    pub fn call<'r, __GustD: ::gust::TypedDevice>(
                        __gust_element_count: ::core::primitive::u32,
                    ) -> ::gust::Call<'r, __GustD, Self, 0> {
                        ::gust::Call::new(__gust_element_count, [])
                    }

                    #[doc = #submit_doc]
                    #[allow(clippy::type_complexity)]
                    pub fn submit<'d, __GustD: ::gust::Run<Self, 0>>(
                        __gust_device: &'d __GustD,
                        __gust_element_count: ::core::primitive::u32,
                    ) -> ::core::result::Result<
                        <__GustD as ::gust::Device>::Job<'d>,
                        <__GustD as ::gust::Device>::Error,
                    > {
                        <__GustD as ::gust::Run<Self, 0>>::submit_call(__gust_device, &Self::call(__gust_element_count))
                    }

                    #[doc = #dispatch_doc]
                    pub fn dispatch<__GustD: ::gust::Run<Self, 0>>(
                        __gust_device: &__GustD,
                        __gust_element_count: ::core::primitive::u32,
                    ) -> ::core::result::Result<(), <__GustD as ::gust::Device>::Error> {
                        <__GustD as ::gust::Run<Self, 0>>::dispatch_call(__gust_device, &Self::call(__gust_element_count))
                    }

                    #extent_items
                }
            },
        ));
    }

    let resources: Vec<syn::Ident> = (0..total)
        .map(|index| format_ident!("__GustR{}", index))
        .collect();
    let sampler_types: Vec<syn::Ident> = (0..sampler_total)
        .map(|index| format_ident!("__GustS{}", index))
        .collect();
    let first = &resources[0];
    let device = quote!(<#first as ::gust::Resource>::Device);
    // In declaration order: a resource is any `Into<Arg>` of its type, a sampler a reference to its type.
    let parameters: Vec<TokenStream> = call_order
        .iter()
        .map(|parameter| match *parameter {
            CallParameter::Resource(index) => {
                let (name, resource) = (&arguments[index].name, &resources[index]);
                quote!(#name: impl ::core::convert::Into<::gust::Arg<'r, #resource>>)
            }
            CallParameter::Sampler(index) => {
                let (name, sampler) = (&samplers[index], &sampler_types[index]);
                quote!(#name: &'r #sampler)
            }
        })
        .collect();
    let mut bounds: Vec<TokenStream> = arguments
        .iter()
        .zip(&resources)
        .enumerate()
        .map(|(index, (argument, resource))| {
            let (element, kind) = (&argument.element, &argument.kind);
            // Spanned on the parameter, so rustc's "required by this bound" note points at the
            // kernel parameter the bound belongs to, not at `#[gpu_module]`.
            let span = argument.name.span();
            if index == 0 {
                quote_spanned!(span=> #resource: ::gust::Resource<Element = #element, Kind = #kind> + 'r)
            } else {
                quote_spanned!(span=> #resource: ::gust::Resource<Element = #element, Device = #device, Kind = #kind> + 'r)
            }
        })
        .collect();
    // A sampler is an argument of the same device as the resources, and that device must have samplers (D64).
    for (name, sampler) in samplers.iter().zip(&sampler_types) {
        let span = name.span();
        bounds.push(quote_spanned!(span=> #sampler: ::gust::SamplerArg<Device = #device> + 'r));
    }
    if !samplers.is_empty() {
        bounds.push(quote!(#device: ::gust::SamplerDevice));
    }
    let forwarded: Vec<&syn::Ident> = call_order
        .iter()
        .map(|parameter| match *parameter {
            CallParameter::Resource(index) => &arguments[index].name,
            CallParameter::Sampler(index) => &samplers[index],
        })
        .collect();
    let bindings: Vec<TokenStream> = arguments
        .iter()
        .map(|argument| {
            let (name, access) = (&argument.name, &argument.access);
            quote!(::core::convert::Into::into(#name).bind(#access))
        })
        .collect();
    let sampler_bindings: Vec<TokenStream> = samplers
        .iter()
        .map(|name| quote!(::gust::SamplerArg::bind(#name)))
        .collect();
    // The call type, its constructors and the `Run` bound: `Call<.., N>` for a kernel without samplers, and the
    // `SampledCall` of its `S` samplers for one with them.
    let (call_type, new_call, new_extent_call, run_bound) = if samplers.is_empty() {
        (
            quote!(::gust::Call<'r, #device, Self, #count>),
            quote!(::gust::Call::new(__gust_element_count, [#(#bindings),*])),
            quote!(::gust::Call::new_extent(__gust_extent, [#(#bindings),*])),
            quote!(#device: ::gust::Run<Self, #count>),
        )
    } else {
        (
            quote!(::gust::SampledCall<'r, #device, Self, #count, #sampler_count>),
            quote!(::gust::Call::new_with_samplers(
                __gust_element_count,
                [#(#bindings),*],
                [#(#sampler_bindings),*]
            )),
            quote!(::gust::Call::new_extent_with_samplers(
                __gust_extent,
                [#(#bindings),*],
                [#(#sampler_bindings),*]
            )),
            quote!(#device: ::gust::Run<
                Self,
                #count,
                [<#device as ::gust::SamplerDevice>::SamplerBinding<'r>; #sampler_count]
            >),
        )
    };
    let run_trait = if samplers.is_empty() {
        quote!(::gust::Run<Self, #count>)
    } else {
        quote!(::gust::Run<
            Self,
            #count,
            [<#device as ::gust::SamplerDevice>::SamplerBinding<'r>; #sampler_count]
        >)
    };
    let generics: Vec<&syn::Ident> = resources.iter().chain(&sampler_types).collect();
    // The forms over a `gust::Extent`: a shape whose invocations are not a grid has the one-dimensional forms only.
    let extent_items = if stage.binding_rules().extent_calls {
        quote! {
            #[doc = #call_extent_doc]
            #[allow(clippy::too_many_arguments)]
            pub fn call_extent<'r, #(#generics),*>(
                __gust_extent: ::gust::Extent,
                #(#parameters),*
            ) -> #call_type
            where
                #(#bounds,)*
            {
                #new_extent_call
            }

            #[doc = #submit_extent_doc]
            #[allow(clippy::too_many_arguments, clippy::type_complexity)]
            pub fn submit_extent<'d, 'r, #(#generics),*>(
                __gust_device: &'d #device,
                __gust_extent: ::gust::Extent,
                #(#parameters),*
            ) -> ::core::result::Result<
                <#device as ::gust::Device>::Job<'d>,
                <#device as ::gust::Device>::Error,
            >
            where
                #(#bounds,)*
                #run_bound,
            {
                <#device as #run_trait>::submit_call(
                    __gust_device,
                    &Self::call_extent(__gust_extent, #(#forwarded),*),
                )
            }

            #[doc = #dispatch_extent_doc]
            #[allow(clippy::too_many_arguments)]
            pub fn dispatch_extent<'r, #(#generics),*>(
                __gust_device: &#device,
                __gust_extent: ::gust::Extent,
                #(#parameters),*
            ) -> ::core::result::Result<(), <#device as ::gust::Device>::Error>
            where
                #(#bounds,)*
                #run_bound,
            {
                <#device as #run_trait>::dispatch_call(
                    __gust_device,
                    &Self::call_extent(__gust_extent, #(#forwarded),*),
                )
            }
        }
    } else {
        TokenStream::new()
    };

    Ok((
        handle_doc,
        quote! {
            #assertion

            impl #kernel {
                #[doc = #call_doc]
                #[allow(clippy::too_many_arguments)]
                pub fn call<'r, #(#generics),*>(
                    __gust_element_count: ::core::primitive::u32,
                    #(#parameters),*
                ) -> #call_type
                where
                    #(#bounds,)*
                {
                    #new_call
                }

                #[doc = #submit_doc]
                #[allow(clippy::too_many_arguments, clippy::type_complexity)]
                pub fn submit<'d, 'r, #(#generics),*>(
                    __gust_device: &'d #device,
                    __gust_element_count: ::core::primitive::u32,
                    #(#parameters),*
                ) -> ::core::result::Result<
                    <#device as ::gust::Device>::Job<'d>,
                    <#device as ::gust::Device>::Error,
                >
                where
                    #(#bounds,)*
                    #run_bound,
                {
                    <#device as #run_trait>::submit_call(
                        __gust_device,
                        &Self::call(__gust_element_count, #(#forwarded),*),
                    )
                }

                #[doc = #dispatch_doc]
                #[allow(clippy::too_many_arguments)]
                pub fn dispatch<'r, #(#generics),*>(
                    __gust_device: &#device,
                    __gust_element_count: ::core::primitive::u32,
                    #(#parameters),*
                ) -> ::core::result::Result<(), <#device as ::gust::Device>::Error>
                where
                    #(#bounds,)*
                    #run_bound,
                {
                    <#device as #run_trait>::dispatch_call(
                        __gust_device,
                        &Self::call(__gust_element_count, #(#forwarded),*),
                    )
                }

                #extent_items
            }
        },
    ))
}
