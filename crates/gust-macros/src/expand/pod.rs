//! Plain-data records: element layout assertions and the `GpuPod` implementation.

use super::*;

/// A type as written, for a message: rustc's token spacing (`super :: Outside`, `Vec < u32 >`) without the spaces
/// around `::`, `<`, `>` and `,`. Other spaces (`dyn Trait`) are kept.
pub(super) fn type_text(ty: &syn::Type) -> String {
    ty.to_token_stream()
        .to_string()
        .replace(" :: ", "::")
        .replace(":: ", "::")
        .replace(" < ", "<")
        .replace("< ", "<")
        .replace(" >", ">")
        .replace(" ,", ",")
}

/// One `const _` per storage parameter of `kernel`: the element layout recorded in the kernel's descriptor (the
/// element's `GpuPod::LAYOUT`) has the size and alignment of the Rust type and is a StorageV2 layout (every StorageV1
/// layout is one) whose stride
/// is the Rust size, so a hand-written `GpuPod` impl that lies is a compile error here. It covers everything the
/// runtime checks (`create_typed_buffer`, dispatch validation) cover. The assertion reads the
/// descriptor rather than naming `GpuPod` again, so an element that is no `GpuPod` still reports the one
/// E0277 the descriptor already gives (the `gpupod-missing` fixture expects exactly one). A uniform parameter gets the
/// same assertion for `Std140V1` (its rules, the Rust size and alignment of the struct, and
/// `TypeLayout::is_std140_v1`). The message is a static string (const panics cannot format) that names the element,
/// the parameter and the kernel.
pub(super) fn element_layout_assertions(
    stage: &dyn StageProfile,
    kernel: &syn::Ident,
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
) -> syn::Result<TokenStream> {
    let mut assertions = Vec::new();
    // The position in the descriptor: the derived ids and the workgroup memory take none.
    let mut position = 0_usize;
    for input in inputs {
        let syn::FnArg::Typed(input) = input else {
            continue;
        };
        let syn::Pat::Ident(pattern) = input.pat.as_ref() else {
            continue;
        };
        let flavor = parameter_flavor_in(stage, &input.ty);
        if !flavor.in_descriptor() {
            continue;
        }
        let index = position;
        position += 1;
        if !matches!(
            flavor,
            ParameterFlavor::StorageRead
                | ParameterFlavor::StorageReadWrite
                | ParameterFlavor::Uniform
        ) {
            continue;
        }
        let element = crate::slang::wrapper_element_type(&input.ty)?;
        // `assert!` reads its message as a format string.
        let text = type_text(element).replace('{', "{{").replace('}', "}}");
        let parameter = &pattern.ident;
        // `parameter_descriptors` emits one descriptor per input, in order, so `index` is this parameter's.
        let index = proc_macro2::Literal::usize_unsuffixed(index);
        if flavor == ParameterFlavor::Uniform {
            let message = format!(
                "GUST Std140V1: uniform element `{text}` layout does not match its Rust representation \
                 (parameter `{parameter}` of kernel `{kernel}`)"
            );
            assertions.push(quote! {
                const _: () = ::core::assert!(
                    match <#kernel as ::gust::Kernel>::DESCRIPTOR.parameters[#index].resource_layout {
                        ::core::option::Option::Some(__gust_layout) => {
                            ::core::matches!(
                                __gust_layout.rules,
                                ::gust::ShaderLayoutRules::Std140V1
                            ) && __gust_layout.element.size as usize
                                == ::core::mem::size_of::<#element>()
                                && __gust_layout.element.alignment as usize
                                    == ::core::mem::align_of::<#element>()
                                && __gust_layout.element_stride as usize
                                    == ::core::mem::size_of::<#element>()
                                && __gust_layout.element.is_std140_v1()
                        }
                        ::core::option::Option::None => false,
                    },
                    #message
                );
            });
            continue;
        }
        let message = format!(
            "GUST StorageV1: element `{text}` layout does not match its Rust representation \
             (parameter `{parameter}` of kernel `{kernel}`)"
        );
        assertions.push(quote! {
            const _: () = ::core::assert!(
                match <#kernel as ::gust::Kernel>::DESCRIPTOR.parameters[#index].resource_layout {
                    ::core::option::Option::Some(__gust_layout) => {
                        __gust_layout.element.size as usize == ::core::mem::size_of::<#element>()
                            && __gust_layout.element.alignment as usize
                                == ::core::mem::align_of::<#element>()
                            && __gust_layout.element_stride as usize
                                == ::core::mem::size_of::<#element>()
                            && __gust_layout.element.is_storage_v2()
                    }
                    ::core::option::Option::None => false,
                },
                #message
            );
        });
    }
    Ok(quote!(#(#assertions)*))
}

/// The layout rules the `const` assertions of a struct's `GpuPod` impl check.
#[derive(Clone, Debug)]
pub(super) enum PodRules {
    /// `StorageV1` or `StorageV2`: `#[repr(C)]`, the largest field alignment, and no padding.
    Storage,
    /// `Std140V1` (the struct is reachable from a uniform parameter): `#[repr(C, align(16))]`, alignment and size
    /// rounded up to 16, and still no padding. The model is the macro's own constant-buffer layout of the struct (the
    /// validator rejected every module where it has a gap or a tail), `None` when a field type is one the macro cannot
    /// see into.
    Std140(Option<crate::uniform::Std140Struct>),
}

pub(super) fn gpu_pod_impl(item: &syn::ItemStruct, rules: &PodRules) -> syn::Result<TokenStream> {
    let ident = &item.ident;
    let syn::Fields::Named(fields) = &item.fields else {
        return Err(diagnostics::error(
            Code::PodStructNamedFields,
            item,
            "GPU POD structs require named fields",
        ));
    };
    let field_layouts = fields.named.iter().map(|field| {
        let field_ident = field.ident.as_ref().expect("named field");
        let field_name = crate::slang::emit_ident(field_ident);
        let ty = &field.ty;
        quote! {
            ::gust::StructFieldLayout {
                name: #field_name,
                offset: ::core::mem::offset_of!(#ident, #field_ident) as u32,
                ty: <#ty as ::gust::GpuPod>::LAYOUT,
            }
        }
    });
    let struct_name = ident.to_string();
    // One assertion per field, in declaration order: the field sits at the sum of the sizes of the fields
    // before it. Const panics cannot format, so the message is a static string that names the field.
    let types: Vec<&syn::Type> = fields.named.iter().map(|field| &field.ty).collect();
    let model = match rules {
        PodRules::Std140(Some(model)) => Some(model),
        _ => None,
    };
    let offset_assertions = fields.named.iter().enumerate().map(|(index, field)| {
        let field_ident = field.ident.as_ref().expect("named field");
        let earlier = &types[..index];
        if let Some(model) = model {
            // The constant-buffer offset the macro computed, named in the message (const panics cannot format).
            let offset = model.offsets[index] as usize;
            let message = format!(
                "GUST Std140V1: field `{field_ident}` of `{struct_name}` is not at byte {offset} of the \
                 constant-buffer layout (it is at another offset in Rust)"
            );
            return quote! {
                ::core::assert!(
                    ::core::mem::offset_of!(#ident, #field_ident) == #offset
                        && ::core::mem::offset_of!(#ident, #field_ident)
                            == 0 #( + ::core::mem::size_of::<#earlier>() )*,
                    #message
                );
            };
        }
        let message = match rules {
            PodRules::Storage => format!(
                // The wording keeps the name `StorageV1` because tests/ui pins it; the rule is StorageV2's (the offset
                // of a field is the sum of the sizes before it, which a vector's alignment can break).
                "GUST StorageV1: field `{field_ident}` of `{struct_name}` is not at its padding-free offset"
            ),
            PodRules::Std140(_) => format!(
                "GUST Std140V1: field `{field_ident}` of `{struct_name}` is not at its padding-free offset"
            ),
        };
        quote! {
            ::core::assert!(
                ::core::mem::offset_of!(#ident, #field_ident)
                    == 0 #( + ::core::mem::size_of::<#earlier>() )*,
                #message
            );
        }
    });
    let (padding_message, alignment_message, storage_message) = match rules {
        PodRules::Storage => (
            format!(
                "GUST StorageV2: `{struct_name}` has tail padding (its fields do not add up to its size)"
            ),
            format!(
                "GUST StorageV2: `{struct_name}` must have the largest alignment of its fields"
            ),
            format!("GUST StorageV2: `{struct_name}` layout is not StorageV2"),
        ),
        PodRules::Std140(_) => (
            format!(
                "GUST Std140V1: `{struct_name}` has padding (its fields do not add up to its size, which a \
                 constant buffer rounds up to 16): add explicit fields"
            ),
            format!(
                "GUST Std140V1: `{struct_name}` must have alignment 16, or the largest alignment of its fields if that is \
                 more"
            ),
            format!("GUST Std140V1: `{struct_name}` layout is not Std140V1"),
        ),
    };
    // The alignment the struct must have: the largest field alignment, and for a constant buffer at least 16.
    let floor: usize = match rules {
        PodRules::Storage => 1,
        PodRules::Std140(_) => 16,
    };
    let layout_check = match rules {
        PodRules::Storage => quote!(is_storage_v2()),
        PodRules::Std140(_) => quote!(is_std140_v1()),
    };
    let model_size = model.map(|model| {
        let size = model.size as usize;
        let message = format!(
            "GUST Std140V1: `{struct_name}` is not {size} bytes, the size of its constant-buffer layout"
        );
        quote! {
            ::core::assert!(::core::mem::size_of::<#ident>() == #size, #message);
        }
    });
    // One block, after the impl whose `LAYOUT` it reads (const items are order-independent), so one cause is one
    // E0080 and the asserts run in this order: the field that is off its offset is named first, then the total,
    // then the alignment (a lying alignment also fails the last one), then the whole layout. Vectors and matrices make
    // padding possible (`{ a: f32, v: Vec2 }`), and the validator (E0208) rejects what it can see; these asserts are
    // the check for what it cannot (a field type the macro cannot see into).
    Ok(quote! {
        unsafe impl ::gust::GpuPod for #ident {
            const LAYOUT: ::gust::TypeLayout = ::gust::TypeLayout::structure(
                stringify!(#ident),
                ::core::mem::size_of::<Self>() as u32,
                ::core::mem::align_of::<Self>() as u32,
                &[#(#field_layouts),*],
            );
        }

        const _: () = {
            ::core::assert!(::core::mem::size_of::<#ident>() > 0);
            ::core::assert!(
                ::core::mem::size_of::<#ident>() <= ::core::primitive::u32::MAX as ::core::primitive::usize
            );
            #(#offset_assertions)*
            ::core::assert!(
                ::core::mem::size_of::<#ident>()
                    == 0 #( + ::core::mem::size_of::<#types>() )*,
                #padding_message
            );
            #model_size
            ::core::assert!(
                ::core::mem::align_of::<#ident>()
                    == { let mut __gust_alignment = #floor; #(
                        if ::core::mem::align_of::<#types>() > __gust_alignment {
                            __gust_alignment = ::core::mem::align_of::<#types>();
                        }
                    )* __gust_alignment },
                #alignment_message
            );
            ::core::assert!(
                <#ident as ::gust::GpuPod>::LAYOUT.#layout_check,
                #storage_message
            );
        };
    })
}
