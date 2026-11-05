//! The map generator (M7, D60, slice m7.a1): `#[gpu]` on a free scalar function.
//!
//! ```text
//! #[gpu]
//! pub fn saxpy(a: f32, x: f32, y: f32) -> f32 { a * x + y }
//! ```
//!
//! stays a plain Rust function and gains a module of the same name (a function and a module live in different
//! namespaces) with the kernel and the host functions of the call:
//!
//! ```text
//! pub mod saxpy {
//!     fn saxpy(a: f32, x: f32, y: f32) -> f32 { a * x + y }   // the lowered copy (the user's tokens)
//!     #[kernel(workgroup_size(256, 1, 1))]
//!     pub fn map_kernel(__gust_gid, a: StructuredBuffer<f32>, .., mut __gust_out: RWStructuredBuffer<f32>) { .. }
//!     pub const DESCRIPTOR: KernelDescriptor = ..;
//!     pub fn map(..) / map_on(..) / map_gpu(..) / map_gpu_on(..) / map_async(..) / map_on_async(..)
//! }
//! ```
//!
//! The generator does not emit a second pipeline. It builds the module above as syntax (the user's function validated
//! as a helper of a module, the kernel written in the dialect around it) and hands it to the expansion that every
//! `#[gpu]` module goes through, so the Slang, the descriptor, the typed calls, the CPU `Run` impl and the artifact
//! handling are the ones of any kernel, and `extra` carries the host functions. The kernel is the grid-stride form of
//! spike m7.s1: every buffer is bound `Independent`, the element count is the output's length, a dispatch has
//! `min(n, CEILING)` invocations, and each invocation covers `ceil(n / CEILING)` rounds. An input buffer of length 1
//! is a [`Broadcast`](https://docs.rs/) argument and is read at index 0, so one kernel serves every mix of vectors
//! and broadcasts.
//!
//! Profiles. Under `GUST_PROFILE=cpu` the host functions name only `::gust::cpu` (no GPU API), `map_gpu` and
//! `map_async` are not emitted, and `map_on` and `map_on_async` are concrete over `CpuDevice` and run the CPU-only
//! chunk body of the plan's section 6 (`__gust_map_cpu_chunk`) in parallel pieces of the output through
//! `gust::cpu::parallel` (m7.c). Under every other profile `map`, `map_on`, `map_gpu`, `map_gpu_on`, `map_async` and
//! `map_on_async` are emitted; the blocking ones are not emitted on `wasm32` (nothing may block there, D69).

use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote};
use syn::ext::IdentExt as _;

use super::{Environment, Identity, Profile, expand_validated};
use crate::{
    diagnostics::{self, Code},
    math_types,
    profile::GRID,
};

/// The name of the generated kernel, and so of its handle struct in the function's module.
const KERNEL: &str = "map_kernel";

/// Names the generator defines in the function's module; the function (whose lowered copy is a module item too) may
/// not be called any of them.
const GENERATED_NAMES: [&str; 10] = [
    "DESCRIPTOR",
    "MODULE_DESCRIPTOR",
    KERNEL,
    "map",
    "map_on",
    "map_gpu",
    "map_gpu_on",
    "map_async",
    "map_on_async",
    "OWN_INDEX",
];

/// The 1D ceiling of the grid-stride kernel: 65 535 workgroups of 256 (`gust::map::CEILING`, which a test in `gust`
/// pins to this number).
const CEILING: u32 = 65_535 * 256;

/// The types one element type is spelled with: the result of the lookup `element_types`, which the map generator (a
/// parameter or result of a map function) and `par_for!` (a buffer binding) share.
#[derive(Clone)]
pub(crate) struct ElementTypes {
    /// The element type as the kernel spells it (a plain name of the dialect): `bool` is `u32` there.
    pub(crate) kernel: TokenStream,
    /// The same type as host code spells it (an absolute path).
    pub(crate) buffer: TokenStream,
    /// The type of the value in host code: `bool` stays `bool`.
    pub(crate) host: TokenStream,
    /// Whether the type looked up is `bool`. A map function takes it; `par_for!` has no `bool` buffer element and
    /// refuses it at its own site (a `bool` there is a scalar binding).
    pub(crate) boolean: bool,
}

struct Parameter {
    /// The parameter's name as the user wrote it (without `mut`).
    name: syn::Ident,
    element: ElementTypes,
}

/// A map function understood: its name, parameters and result.
struct Signature {
    name: syn::Ident,
    parameters: Vec<Parameter>,
    result: ElementTypes,
}

/// What the generator makes of one function: the module to expand (the kernel around the user's function) and the
/// host functions that go inside it. Tests emit the module's Slang from it.
#[cfg(test)]
pub(crate) struct MapKernel {
    pub(crate) module: syn::ItemMod,
    #[allow(dead_code)]
    pub(crate) host: TokenStream,
}

/// The module the generator builds around `function`, and the host functions, for a function that passed
/// [`check_signature`] and validation.
#[cfg(test)]
pub(crate) fn map_kernel(function: &syn::ItemFn, profile: Profile) -> syn::Result<MapKernel> {
    let signature = check_signature(function)?;
    validate(function)?;
    let (module, host) = build(function, &signature, profile);
    Ok(MapKernel { module, host })
}

/// Expands `#[gpu]` on a free function for `profile`: the function as written, then its module.
pub(crate) fn expand_map(function: syn::ItemFn, profile: Profile) -> syn::Result<TokenStream> {
    expand_map_in(function, &Environment::from_process(), profile)
}

/// [`expand_map`] with the environment given, as the module expansion has it.
pub(super) fn expand_map_in(
    function: syn::ItemFn,
    environment: &Environment,
    profile: Profile,
) -> syn::Result<TokenStream> {
    let signature = check_signature(&function)?;
    validate(&function)?;
    let (module, host) = build(&function, &signature, profile);
    let expanded = expand_validated(module, environment, profile, &GRID, &host, Identity::Module)?;
    Ok(quote! {
        #function

        #expanded
    })
}

// ---------------------------------------------------------------------------------------------------------------
// The signature.
// ---------------------------------------------------------------------------------------------------------------

/// The signature rules (GUST-E0714, GUST-E0715, and GUST-E0204 for a 3-wide type): at least one parameter, each a
/// plain identifier with a scalar, vector or matrix type that can be a buffer element, a result of such a type, no
/// `#[kernel]`, and a name the generated module does not use itself. Everything else about the function (its body,
/// `async`, generics, `self`) is the validator's, which sees the function as a helper of a module.
fn check_signature(function: &syn::ItemFn) -> syn::Result<Signature> {
    let name = function.sig.ident.clone();
    if function
        .attrs
        .iter()
        .any(|attribute| attribute.path().is_ident("kernel"))
    {
        return Err(diagnostics::error(
            Code::MapSignature,
            &function.sig,
            "a map function is not a `#[kernel]`: `#[gpu]` on a function generates the kernel; write a kernel \
             inside `#[gpu] mod name { .. }` to control it",
        ));
    }
    let unraw = name.unraw().to_string();
    if GENERATED_NAMES.contains(&unraw.as_str()) {
        return Err(diagnostics::error(
            Code::MapNameCollision,
            &name,
            format!(
                "the module `{unraw}` that `#[gpu]` generates for this function would define `{unraw}` twice: \
                 the generated names are {}; rename the function",
                GENERATED_NAMES
                    .iter()
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ));
    }
    let mut parameters = Vec::new();
    for input in &function.sig.inputs {
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
        let parameter = pattern.ident.unraw().to_string();
        if parameter == unraw {
            return Err(diagnostics::error(
                Code::MapNameCollision,
                &pattern.ident,
                format!(
                    "the parameter `{parameter}` has the function's name, so the generated kernel could not call the \
                     function; rename the parameter"
                ),
            ));
        }
        let element = element(&input.ty, &format!("parameter `{parameter}`"))?;
        parameters.push(Parameter {
            name: pattern.ident.clone(),
            element,
        });
    }
    if parameters.is_empty() {
        return Err(diagnostics::error(
            Code::MapSignature,
            &function.sig,
            "a map function needs at least one parameter: the element count of a call comes from its arguments",
        ));
    }
    let result = match &function.sig.output {
        syn::ReturnType::Type(_, ty) => element(ty, "the return type")?,
        syn::ReturnType::Default => {
            return Err(diagnostics::error(
                Code::MapSignature,
                &function.sig,
                "a map function returns one value per element; this one returns `()`",
            ));
        }
    };
    Ok(Signature {
        name,
        parameters,
        result,
    })
}

/// The element a map parameter or result of type `ty` is: `f32`, `i32`, `u32`, `bool` (or their dialect spellings
/// `float`, `int`, `uint`) and the vector and matrix types that can be buffer elements (GUST-E0714 otherwise,
/// GUST-E0204 for a 3-wide type or `Mat3`).
fn element(ty: &syn::Type, what: &str) -> syn::Result<ElementTypes> {
    element_types(ty).map_err(|reject| match reject {
        ElementReject::Unsupported => diagnostics::error(
            Code::MapSignature,
            ty,
            format!(
                "{what} has a type a map function cannot take: use `f32`, `i32`, `u32`, `bool`, `Vec2`, `Vec4`, \
                 `IVec2`, `IVec4`, `UVec2`, `UVec4`, `Mat2` or `Mat4`, by value"
            ),
        ),
        ElementReject::ThreeWide(name) => diagnostics::error(
            Code::ThreeWideInBuffer,
            ty,
            format!(
                "{what} is `{name}`, which a map call cannot take: a map's arguments and results are buffer \
                 elements, and 3-wide types have stride 16 in a buffer and no Rust layout that matches in \
                 every position; use the 4-wide type or scalars"
            ),
        ),
    })
}

/// Why a type is not an element. The lookup names the reason only; each caller words it in its own diagnostic.
pub(crate) enum ElementReject {
    /// Not a plain type name, or a name that is none of the element types.
    Unsupported,
    /// A 3-wide vector or `Mat3` (the type's name), which has no matching layout in a buffer.
    ThreeWide(&'static str),
}

/// The element types a plain type name stands for, shared by the map generator and `par_for!`: `f32`, `i32`, `u32`,
/// `bool` (`ElementTypes::boolean`; a caller that has no `bool` element refuses it) and the buffer-element vector and
/// matrix types, with the dialect spellings `float`, `int`, `uint`.
pub(crate) fn element_types(ty: &syn::Type) -> Result<ElementTypes, ElementReject> {
    let syn::Type::Path(path) = ty else {
        return Err(ElementReject::Unsupported);
    };
    let Some(ident) = path.path.get_ident().filter(|_| path.qself.is_none()) else {
        return Err(ElementReject::Unsupported);
    };
    let name = ident.unraw().to_string();
    let scalar = |kernel: TokenStream, buffer: TokenStream| ElementTypes {
        kernel,
        host: buffer.clone(),
        buffer,
        boolean: false,
    };
    Ok(match name.as_str() {
        "f32" | "float" => scalar(quote!(f32), quote!(::core::primitive::f32)),
        "i32" | "int" => scalar(quote!(i32), quote!(::core::primitive::i32)),
        "u32" | "uint" => scalar(quote!(u32), quote!(::core::primitive::u32)),
        "bool" => ElementTypes {
            kernel: quote!(u32),
            buffer: quote!(::core::primitive::u32),
            host: quote!(::core::primitive::bool),
            boolean: true,
        },
        other => {
            let Some(math) = math_types::math_type(other) else {
                return Err(ElementReject::Unsupported);
            };
            if !math.buffer {
                return Err(ElementReject::ThreeWide(math.name));
            }
            let path = format_ident!("{}", math.name);
            scalar(quote!(#path), quote!(::gust::math::#path))
        }
    })
}

/// The validator's verdict on the function, as a helper of a module of its own (the context its body is lowered in).
fn validate(function: &syn::ItemFn) -> syn::Result<()> {
    // The attributes belong to the function as plain Rust (`#[inline]`, `#[must_use]`, `#[cfg]`); the lowered copy
    // has none.
    let mut function = function.clone();
    function.attrs.clear();
    let module: syn::ItemMod = syn::parse_quote! {
        mod __gust_validated {
            #function
        }
    };
    crate::validate::validate_module_in(&GRID, &module)
}

// ---------------------------------------------------------------------------------------------------------------
// The module.
// ---------------------------------------------------------------------------------------------------------------

/// The attributes of the function that carry over to the generated module: `cfg` (a gated function has a gated
/// module) and the function's documentation is not copied (the module has its own).
fn module_attributes(
    function: &syn::ItemFn,
    signature: &Signature,
    profile: Profile,
) -> Vec<syn::Attribute> {
    let mut attributes: Vec<syn::Attribute> = function
        .attrs
        .iter()
        .filter(|attribute| attribute.path().is_ident("cfg"))
        .cloned()
        .collect();
    let name = &signature.name;
    let calls = if profile == Profile::Cpu {
        "`map` and `map_on` take one argument per parameter (a slice or a `gust::map::Broadcast`) and return the \
         results as a `Vec`; `map_on_async` does not block"
    } else {
        "`map` and `map_on` take one argument per parameter (a slice, a `gust::map::Broadcast`, or a resident \
         buffer) and return the results as a `Vec`; `map_gpu` and `map_gpu_on` keep them on the GPU; `map_async` \
         and `map_on_async` do not block"
    };
    let text = format!(
        "The element-wise calls of [`{name}`] and its generated kernel (`#[gpu]` on a function, D60): {calls}. All \
         arguments of one call that have a length must be equally long."
    );
    attributes.push(syn::parse_quote!(#[doc = #text]));
    attributes
}

/// The synthesised module around the user's function and the host functions that go inside it.
fn build(
    function: &syn::ItemFn,
    signature: &Signature,
    profile: Profile,
) -> (syn::ItemMod, TokenStream) {
    let name = &signature.name;
    let kernel = format_ident!("{}", KERNEL);

    // The user's function as a private helper of the module: their tokens, so errors in the body point at their code.
    let mut helper = function.clone();
    helper.attrs.clear();
    helper.vis = syn::Visibility::Inherited;

    let kernel_parameters = signature.parameters.iter().map(|parameter| {
        let (name, element) = (&parameter.name, &parameter.element.kernel);
        quote!(#name: StructuredBuffer<#element>)
    });
    let indices = (0..signature.parameters.len()).map(|index| {
        let (parameter, local) = (
            &signature.parameters[index],
            format_ident!("__gust_i{}", index),
        );
        let name = &parameter.name;
        quote! {
            let mut #local = __gust_j;
            if #name.len() == 1u32 {
                #local = 0u32;
            }
        }
    });
    let values = signature
        .parameters
        .iter()
        .enumerate()
        .map(|(index, parameter)| {
            let (name, index_local, value) = (
                &parameter.name,
                format_ident!("__gust_i{}", index),
                format_ident!("__gust_v{}", index),
            );
            if parameter.element.boolean {
                quote!(let #value = #name[#index_local] != 0u32;)
            } else {
                quote!(let #value = #name[#index_local];)
            }
        });
    let arguments = (0..signature.parameters.len()).map(|index| format_ident!("__gust_v{}", index));
    let result_element = &signature.result.kernel;
    let store = if signature.result.boolean {
        quote!(__gust_out[__gust_j] = __gust_r as u32;)
    } else {
        quote!(__gust_out[__gust_j] = __gust_r;)
    };
    let ceiling = Literal::u32_suffixed(CEILING);
    let module_name = name.clone();
    let visibility = &function.vis;
    let attributes = module_attributes(function, signature, profile);
    let module: syn::ItemMod = syn::parse_quote! {
        #(#attributes)*
        #visibility mod #module_name {
            #helper

            #[kernel(workgroup_size(256, 1, 1))]
            pub fn #kernel(
                __gust_gid: SV_DispatchThreadID,
                #(#kernel_parameters,)*
                mut __gust_out: RWStructuredBuffer<#result_element>,
            ) {
                let __gust_n = __gust_out.len();
                let __gust_rounds = __gust_n / #ceiling + (__gust_n % #ceiling != 0u32) as u32;
                for __gust_round in 0u32..__gust_rounds {
                    let __gust_j = __gust_gid.x + __gust_round * #ceiling;
                    if __gust_j < __gust_n {
                        #(#indices)*
                        #(#values)*
                        let __gust_r = #name(#(#arguments),*);
                        #store
                    }
                }
            }
        }
    };
    let host = host_items(signature, profile);
    (module, host)
}

// ---------------------------------------------------------------------------------------------------------------
// The host functions.
// ---------------------------------------------------------------------------------------------------------------

/// The generics, parameters and where clause of one host function, written once for all of them.
struct Shape {
    /// `__GustA0, __GustA1, ..`
    argument_types: Vec<syn::Ident>,
    /// `a: __GustA0, x: __GustA1, ..`
    parameters: Vec<TokenStream>,
    /// `__GustA0: MapArg<E0, D>, ..` for the device type `device`.
    bounds: Vec<TokenStream>,
    /// The user's parameter names, for forwarding.
    names: Vec<syn::Ident>,
}

fn shape(signature: &Signature, device: &TokenStream) -> Shape {
    let argument_types: Vec<syn::Ident> = (0..signature.parameters.len())
        .map(|index| format_ident!("__GustA{}", index))
        .collect();
    let mut parameters = Vec::new();
    let mut bounds = Vec::new();
    let mut names = Vec::new();
    for (parameter, argument) in signature.parameters.iter().zip(&argument_types) {
        let (name, element) = (&parameter.name, &parameter.element.host);
        parameters.push(quote!(#name: #argument));
        bounds.push(quote!(#argument: ::gust::map::MapArg<#element, #device>));
        names.push(name.clone());
    }
    Shape {
        argument_types,
        parameters,
        bounds,
        names,
    }
}

/// Everything after the arguments are known: the element count, the bindings, the output, the dispatch and the
/// final step (`sync`: read back and return; otherwise return the output buffer, a `GpuVec`, and the caller decides).
/// `device` is the expression of type `&D` that names the device. `asynchronous` submits without waiting and awaits
/// the read-back.
fn call_body(
    signature: &Signature,
    shape: &Shape,
    device_type: &TokenStream,
    device: &TokenStream,
    finish: Finish,
) -> TokenStream {
    let kernel = format_ident!("{}", KERNEL);
    let counted: Vec<TokenStream> = signature
        .parameters
        .iter()
        .zip(&shape.argument_types)
        .map(|(parameter, _)| {
            let (name, label) = (&parameter.name, parameter.name.unraw().to_string());
            let element = &parameter.element.host;
            quote!((#label, ::gust::map::MapArg::<#element, #device_type>::shape(&#name)))
        })
        .collect();
    let bound: Vec<syn::Ident> = (0..signature.parameters.len())
        .map(|index| format_ident!("__gust_b{}", index))
        .collect();
    let bindings = signature
        .parameters
        .iter()
        .zip(&bound)
        .map(|(parameter, local)| {
            let (name, element) = (&parameter.name, &parameter.element.host);
            quote! {
                let #local = ::gust::map::MapArg::<#element, #device_type>::bind(#name, #device)?;
            }
        });
    let resources = signature
        .parameters
        .iter()
        .zip(&bound)
        .zip(&shape.argument_types)
        .map(|((parameter, local), argument)| {
            let element = &parameter.element.host;
            quote!(::gust::Independent(
                ::gust::map::resource::<#element, #device_type, #argument>(&#local)
            ))
        })
        .collect::<Vec<_>>();
    let result = &signature.result.buffer;
    let output = match finish {
        Finish::GpuVec => quote! {
            let __gust_out = ::gust_wgpu::GpuVec::<#result>::zeroed_on(__gust_handle, __gust_n)
                .map_err(<#device_type as ::gust::map::MapHost>::map_error)?;
        },
        _ => quote! {
            let __gust_out = <#device_type as ::gust::map::MapHost>::zeroed::<#result>(#device, __gust_n)
                .map_err(<#device_type as ::gust::map::MapHost>::map_error)?;
        },
    };
    let launch = match finish {
        Finish::Await => quote! {
            let __gust_job = #kernel::submit(
                #device,
                __gust_n.min(::gust::map::CEILING),
                #(#resources,)*
                ::gust::Independent(&__gust_out),
            )
            .map_err(<#device_type as ::gust::map::MapHost>::map_error)?;
        },
        // On wasm32 nothing may block (D69): submit and drop the job. The result is a resident `GpuVec` that the caller
        // reads with `to_vec_async`, and queue order makes that read see the kernel's writes. The `cfg` is evaluated in
        // the user's crate; the native expansion is the `dispatch` below, as before.
        Finish::GpuVec => quote! {
            #[cfg(target_arch = "wasm32")]
            let __gust_job = #kernel::submit(
                #device,
                __gust_n.min(::gust::map::CEILING),
                #(#resources,)*
                ::gust::Independent(&__gust_out),
            )
            .map_err(<#device_type as ::gust::map::MapHost>::map_error)?;
            #[cfg(target_arch = "wasm32")]
            ::core::mem::drop(__gust_job);
            #[cfg(not(target_arch = "wasm32"))]
            #kernel::dispatch(
                #device,
                __gust_n.min(::gust::map::CEILING),
                #(#resources,)*
                ::gust::Independent(&__gust_out),
            )
            .map_err(<#device_type as ::gust::map::MapHost>::map_error)?;
        },
        Finish::Return => quote! {
            #kernel::dispatch(
                #device,
                __gust_n.min(::gust::map::CEILING),
                #(#resources,)*
                ::gust::Independent(&__gust_out),
            )
            .map_err(<#device_type as ::gust::map::MapHost>::map_error)?;
        },
    };
    let convert = if signature.result.boolean {
        quote!(
            __gust_host
                .into_iter()
                .map(|__gust_word| __gust_word != 0u32)
                .collect()
        )
    } else {
        quote!(__gust_host)
    };
    // An empty call dispatches nothing: a `Vec` call returns an empty `Vec`; a resident call returns an empty
    // `GpuVec` (created before the bindings, so the empty arguments are never uploaded).
    let empty = match finish {
        Finish::GpuVec => quote! {
            return ::gust_wgpu::GpuVec::<#result>::zeroed_on(__gust_handle, 0u32)
                .map_err(<#device_type as ::gust::map::MapHost>::map_error);
        },
        _ => quote!(return ::core::result::Result::Ok(::std::vec::Vec::new());),
    };
    let finish_tokens = match finish {
        Finish::Return => quote! {
            let __gust_host = <#device_type as ::gust::map::MapHost>::readback::<#result>(#device, &__gust_out)
                .map_err(<#device_type as ::gust::map::MapHost>::map_error)?;
            ::core::result::Result::Ok(#convert)
        },
        Finish::Await => quote! {
            let __gust_host = <#device_type as ::gust::map::MapHost>::readback_async::<#result>(#device, &__gust_out)
                .await
                .map_err(<#device_type as ::gust::map::MapHost>::map_error)?;
            ::core::mem::drop(__gust_job);
            ::core::result::Result::Ok(#convert)
        },
        Finish::GpuVec => quote!(::core::result::Result::Ok(__gust_out)),
    };
    quote! {
        let __gust_n = ::gust::map::element_count(&[#(#counted),*])?;
        if __gust_n == 0u32 {
            #empty
        }
        #(#bindings)*
        #output
        #launch
        #finish_tokens
    }
}

/// How a call ends.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Finish {
    /// Dispatch, wait, read back, return the `Vec`.
    Return,
    /// Submit, await the read-back, return the `Vec`.
    Await,
    /// Dispatch and return the output `GpuVec` (resident).
    GpuVec,
}

fn host_items(signature: &Signature, profile: Profile) -> TokenStream {
    let name = &signature.name;
    let kernel = format_ident!("{}", KERNEL);
    let count = signature.parameters.len() + 1;
    let result_host = &signature.result.host;
    let result_buffer = &signature.result.buffer;
    let not_wasm = quote!(#[cfg(not(target_arch = "wasm32"))]);
    let cpu = profile == Profile::Cpu;

    let generic_device = quote!(__GustD);
    let generic = shape(signature, &generic_device);
    let generic_arguments = &generic.argument_types;
    let generic_parameters = &generic.parameters;
    let generic_bounds = &generic.bounds;
    let sync_body = call_body(
        signature,
        &generic,
        &generic_device,
        &quote!(__gust_device),
        Finish::Return,
    );
    let async_body = call_body(
        signature,
        &generic,
        &generic_device,
        &quote!(__gust_device),
        Finish::Await,
    );
    let run_bound = quote!(__GustD: ::gust::map::MapHost + ::gust::Run<#kernel, #count>);

    // The concrete device of `map`, and the device `map` opens.
    let (concrete_device, open_device) = if cpu {
        (
            quote!(::gust::cpu::CpuDevice),
            quote!(let __gust_device = ::gust::cpu::CpuDevice::new();),
        )
    } else {
        (
            quote!(::gust_wgpu::HeadlessDevice),
            quote! {
                let __gust_device = ::gust_wgpu::default_device().map_err(::gust::map::Error::device)?;
            },
        )
    };
    let concrete = shape(signature, &concrete_device);
    let concrete_parameters = &concrete.parameters;
    let concrete_bounds = &concrete.bounds;
    let concrete_arguments = &concrete.argument_types;
    let forwarded = &concrete.names;
    let deref_device = if cpu {
        quote!(&__gust_device)
    } else {
        quote!(&*__gust_device)
    };

    let doc_on = format!(
        "[`{name}`] over the elements of the arguments on `device`, with the results read back into a `Vec`. \
         `Err` before any device work when the lengths differ (`gust::map::Error::LengthMismatch`) or the count \
         does not fit `u32`; an empty call returns an empty `Vec` without dispatching."
    );
    let doc_map = if cpu {
        format!(
            "[`{name}`] over the elements of the arguments on the CPU (`GUST_PROFILE=cpu`), as a `Vec`."
        )
    } else {
        format!(
            "[`{name}`] over the elements of the arguments on the GPU (the default device), as a `Vec`."
        )
    };
    let doc_async = "[`map_on`] that does not block: submit, then await the read-back, as a `Vec`.";

    // `map_on` and `map_on_async`. The `cpu` profile has one device, so they are concrete there and the work is the
    // parallel chunked run of `gust::cpu::parallel` (m7.c); every other profile is generic over the device and goes
    // through the kernel's `Run` impl, which on the CPU twin stays sequential.
    let on_items = if cpu {
        let device = quote!(::gust::cpu::CpuDevice);
        let cpu_shape = shape(signature, &device);
        let cpu_body = cpu_body(signature, &cpu_shape);
        let (cpu_arguments, cpu_parameters, cpu_bounds) = (
            &cpu_shape.argument_types,
            &cpu_shape.parameters,
            &cpu_shape.bounds,
        );
        let doc_on_cpu = format!(
            "[`{name}`] over the elements of the arguments on `device`, as a `Vec`, in parallel chunks of the output \
             (`GUST_CPU_THREADS`, see `docs/SEMANTICS.md`): the result is the one a single thread gives, and if \
             elements panic, the panic of the lowest index is raised. `Err` before any work when the lengths differ \
             (`gust::map::Error::LengthMismatch`), the count does not fit `u32` or `GUST_CPU_THREADS` is invalid; an \
             empty call returns an empty `Vec`."
        );
        let doc_async_cpu = "[`map_on`] as an `async fn`: the CPU work runs inside the first poll, since there is \
             nothing to wait for.";
        quote! {
            #[doc = #doc_on_cpu]
            #not_wasm
            #[allow(clippy::too_many_arguments)]
            pub fn map_on<#(#cpu_arguments),*>(
                __gust_device: &#device,
                #(#cpu_parameters),*
            ) -> ::core::result::Result<::std::vec::Vec<#result_host>, ::gust::map::Error>
            where
                #(#cpu_bounds,)*
            {
                #cpu_body
            }

            #[doc = #doc_async_cpu]
            #[allow(clippy::too_many_arguments, clippy::unused_async)]
            pub async fn map_on_async<#(#cpu_arguments),*>(
                __gust_device: &#device,
                #(#cpu_parameters),*
            ) -> ::core::result::Result<::std::vec::Vec<#result_host>, ::gust::map::Error>
            where
                #(#cpu_bounds,)*
            {
                #cpu_body
            }
        }
    } else {
        quote! {
            #[doc = #doc_on]
            #not_wasm
            #[allow(clippy::too_many_arguments)]
            pub fn map_on<#generic_device, #(#generic_arguments),*>(
                __gust_device: &#generic_device,
                #(#generic_parameters),*
            ) -> ::core::result::Result<::std::vec::Vec<#result_host>, ::gust::map::Error>
            where
                #run_bound,
                #(#generic_bounds,)*
            {
                #sync_body
            }

            #[doc = #doc_async]
            #[allow(clippy::too_many_arguments)]
            pub async fn map_on_async<#generic_device, #(#generic_arguments),*>(
                __gust_device: &#generic_device,
                #(#generic_parameters),*
            ) -> ::core::result::Result<::std::vec::Vec<#result_host>, ::gust::map::Error>
            where
                #run_bound,
                #(#generic_bounds,)*
            {
                #async_body
            }
        }
    };

    let mut items = TokenStream::new();
    items.extend(quote! {
        /// The descriptor of the generated kernel.
        pub const DESCRIPTOR: ::gust::KernelDescriptor =
            <#kernel as ::gust::Kernel>::DESCRIPTOR;

        // The stamp (D60, plan section 6): the kernel writes only `out[i]` for its own element index.
        #[doc(hidden)]
        pub const OWN_INDEX: bool = true;
        impl ::gust::map::OwnIndex for #kernel {}

        #on_items

        #[doc = #doc_map]
        #not_wasm
        #[allow(clippy::too_many_arguments)]
        pub fn map<#(#concrete_arguments),*>(
            #(#concrete_parameters),*
        ) -> ::core::result::Result<::std::vec::Vec<#result_host>, ::gust::map::Error>
        where
            #(#concrete_bounds,)*
        {
            #open_device
            map_on(#deref_device, #(#forwarded),*)
        }
    });
    if !cpu {
        let gpu = shape(signature, &quote!(::gust_wgpu::HeadlessDevice));
        let gpu_parameters = &gpu.parameters;
        let gpu_bounds = &gpu.bounds;
        let gpu_arguments = &gpu.argument_types;
        let gpu_names = &gpu.names;
        let gpu_body = call_body(
            signature,
            &gpu,
            &quote!(::gust_wgpu::HeadlessDevice),
            &quote!(&**__gust_handle),
            Finish::GpuVec,
        );
        let doc_gpu = format!(
            "[`{name}`] over the arguments on the default GPU device (`gust_wgpu::default_device`), leaving the \
             results there as a `gust_wgpu::GpuVec` (no read-back: pass it to the next call). Every resident \
             argument must live on the default device too (`gust::map::Error::MixedDevices` otherwise); for \
             another device use [`map_gpu_on`]. In the browser (wasm32) `default_device` is `NoAdapter` until \
             `gust_wgpu::default_device_async` has run, so open the device first or call [`map_async`]."
        );
        let doc_gpu_on = "[`map_gpu`] on `__gust_handle`, any device (the default one included), which must be the \
             device of every resident argument (`gust::map::Error::MixedDevices` otherwise).";
        let doc_map_async = format!(
            "[`{name}`] over the arguments on the default GPU device without blocking: the device opens on the \
             first call (in the browser too, through `gust_wgpu::default_device_async`), and the results are awaited."
        );
        items.extend(quote! {
            #[doc = #doc_gpu_on]
            #[allow(clippy::too_many_arguments)]
            pub fn map_gpu_on<#(#gpu_arguments),*>(
                __gust_handle: &::gust_wgpu::DeviceHandle,
                #(#gpu_parameters),*
            ) -> ::core::result::Result<::gust_wgpu::GpuVec<#result_buffer>, ::gust::map::Error>
            where
                #(#gpu_bounds,)*
            {
                #gpu_body
            }

            #[doc = #doc_gpu]
            #[allow(clippy::too_many_arguments)]
            pub fn map_gpu<#(#gpu_arguments),*>(
                #(#gpu_parameters),*
            ) -> ::core::result::Result<::gust_wgpu::GpuVec<#result_buffer>, ::gust::map::Error>
            where
                #(#gpu_bounds,)*
            {
                let __gust_handle = ::gust_wgpu::default_device().map_err(::gust::map::Error::device)?;
                map_gpu_on(&__gust_handle, #(#gpu_names),*)
            }

            #[doc = #doc_map_async]
            #[allow(clippy::too_many_arguments)]
            pub async fn map_async<#(#gpu_arguments),*>(
                #(#gpu_parameters),*
            ) -> ::core::result::Result<::std::vec::Vec<#result_host>, ::gust::map::Error>
            where
                #(#gpu_bounds,)*
            {
                let __gust_handle = ::gust_wgpu::default_device_async()
                    .await
                    .map_err(::gust::map::Error::device)?;
                map_on_async(&*__gust_handle, #(#gpu_names),*).await
            }
        });
    } else {
        items.extend(cpu_chunk(signature));
    }
    items
}

/// The body of a `cpu` profile `map_on` and `map_on_async` (slice m7.c): the checks and the binding of every other map
/// call, but the work is `gust::cpu::parallel::run_map_into_vec` over [`cpu_chunk`]. The inputs are borrowed as
/// whole-length slices (a broadcast is the slice of one element, a `bool` the converted `u32` words) and shared by
/// every chunk; the output is the zeroed buffer of the device, split into disjoint pieces by the runner. Lengths and
/// the thread count are checked before anything is uploaded, so an error leaves nothing behind.
fn cpu_body(signature: &Signature, shape: &Shape) -> TokenStream {
    let kernel = format_ident!("{}", KERNEL);
    let device_type = quote!(::gust::cpu::CpuDevice);
    let map_error = quote!(<#device_type as ::gust::map::MapHost>::map_error);
    let count = signature.parameters.len();
    let counted: Vec<TokenStream> = signature
        .parameters
        .iter()
        .map(|parameter| {
            let (name, label) = (&parameter.name, parameter.name.unraw().to_string());
            let element = &parameter.element.host;
            quote!((#label, ::gust::map::MapArg::<#element, #device_type>::shape(&#name)))
        })
        .collect();
    let bound: Vec<syn::Ident> = (0..count)
        .map(|index| format_ident!("__gust_b{}", index))
        .collect();
    let borrowed: Vec<syn::Ident> = (0..count)
        .map(|index| format_ident!("__gust_r{}", index))
        .collect();
    let slices: Vec<syn::Ident> = (0..count)
        .map(|index| format_ident!("__gust_s{}", index))
        .collect();
    let bindings = signature.parameters.iter().zip(&bound).map(|(parameter, local)| {
        let (name, element) = (&parameter.name, &parameter.element.host);
        quote! {
            let #local = ::gust::map::MapArg::<#element, #device_type>::bind(#name, __gust_device)?;
        }
    });
    let reads = signature
        .parameters
        .iter()
        .zip(&shape.argument_types)
        .zip(bound.iter().zip(borrowed.iter().zip(&slices)))
        .map(|((parameter, argument), (local, (borrow, slice)))| {
            let (host, buffer) = (&parameter.element.host, &parameter.element.buffer);
            let label = parameter.name.unraw().to_string();
            quote! {
                let #borrow = ::gust::cpu::parallel::input::<#buffer, _>(
                    __gust_device,
                    ::gust::map::resource::<#host, #device_type, #argument>(&#local),
                    #label,
                )
                .map_err(#map_error)?;
                let #slice: &[#buffer] = &#borrow;
            }
        });
    let result = &signature.result.buffer;
    let convert = if signature.result.boolean {
        quote!(
            __gust_host
                .into_iter()
                .map(|__gust_word| __gust_word != 0u32)
                .collect()
        )
    } else {
        quote!(__gust_host)
    };
    quote! {
        let __gust_n = ::gust::map::element_count(&[#(#counted),*])?;
        let __gust_threads = ::gust::cpu::parallel::map_threads(__gust_n).map_err(#map_error)?;
        if __gust_n == 0u32 {
            return ::core::result::Result::Ok(::std::vec::Vec::new());
        }
        ::gust::cpu::parallel::require_own_index::<#kernel>();
        #(#bindings)*
        #(#reads)*
        let __gust_out = <#device_type as ::gust::map::MapHost>::zeroed::<#result>(__gust_device, __gust_n)
            .map_err(#map_error)?;
        let __gust_host = ::gust::cpu::parallel::run_map_into_vec(
            __gust_out,
            __gust_threads,
            |__gust_lo, __gust_hi, __gust_chunk, __gust_cur| {
                __gust_map_cpu_chunk(__gust_lo, __gust_hi, #(#slices,)* __gust_chunk, __gust_cur)
            },
        )
        .map_err(#map_error)?;
        ::core::result::Result::Ok(#convert)
    }
}

/// The CPU-only chunk body of the plan's section 6 (`cpu` profile only; the parallel `map_on` of [`cpu_body`] runs it):
/// the function over the element indexes `lo..hi`, reading whole-length input slices at the global index (a
/// one-element slice at index 0, a broadcast) and writing only `chunk[i - lo]`, the invocation's own element. The
/// generator owns `i`. It sets `cur` to `i` before it evaluates element `i`, so that the runner can name the element
/// of a panic and re-raise the lowest one.
fn cpu_chunk(signature: &Signature) -> TokenStream {
    let name = &signature.name;
    let inputs: Vec<syn::Ident> = (0..signature.parameters.len())
        .map(|index| format_ident!("__gust_in{}", index))
        .collect();
    let slices = signature
        .parameters
        .iter()
        .zip(&inputs)
        .map(|(parameter, input)| {
            let element = &parameter.element.buffer;
            quote!(#input: &[#element])
        });
    let reads =
        signature
            .parameters
            .iter()
            .zip(&inputs)
            .enumerate()
            .map(|(index, (parameter, input))| {
                let value = format_ident!("__gust_x{}", index);
                let read = quote!(#input[if #input.len() == 1 { 0 } else { __gust_i as usize }]);
                if parameter.element.boolean {
                    quote!(let #value = #read != 0;)
                } else {
                    quote!(let #value = #read;)
                }
            });
    let arguments = (0..signature.parameters.len()).map(|index| format_ident!("__gust_x{}", index));
    let chunk_element = &signature.result.buffer;
    let store = if signature.result.boolean {
        quote!(u32::from(__gust_r))
    } else {
        quote!(__gust_r)
    };
    quote! {
        #[doc(hidden)]
        #[allow(clippy::too_many_arguments)]
        pub fn __gust_map_cpu_chunk(
            __gust_lo: u32,
            __gust_hi: u32,
            #(#slices,)*
            __gust_chunk: &mut [#chunk_element],
            __gust_cur: &::core::cell::Cell<u32>,
        ) {
            for __gust_i in __gust_lo..__gust_hi {
                __gust_cur.set(__gust_i);
                #(#reads)*
                let __gust_r = #name(#(#arguments),*);
                __gust_chunk[(__gust_i - __gust_lo) as usize] = #store;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expand::artifact::{TWIN_TEXT, tests::Scratch};

    /// The export records of a `#[gpu] fn` hold no CPU twin text (GUST-E0712): the map kernel's dispatch id is not
    /// spelled `__gust_id`, the marker the guard reserves for the twin, so `cargo gust build` can export a map.
    #[test]
    fn the_export_records_of_a_gpu_fn_pass_the_twin_text_guard() {
        let function: syn::ItemFn = crate::diagnostics::testing::parse_str(
            "pub fn saxpy(a: f32, x: f32, y: f32) -> f32 { a * x + y }",
        )
        .unwrap();
        let scratch = Scratch::new("map-records");
        let records = scratch.0.join("records");
        let environment = Environment {
            manifest_dir: None,
            export_dir: Some(records.clone()),
            package: Some("demo-crate".to_owned()),
            reported_file: None,
        };
        let expanded = expand_map_in(function, &environment, Profile::Other)
            .unwrap_or_else(|error| panic!("a map must export: {error}"));
        // A record with twin text is refused (GUST-E0712) as a `compile_error!` beside the item, not as an `Err`, so
        // the failure names the code instead of a missing directory.
        let text = expanded.to_string();
        assert!(!text.contains("compile_error"), "{text}");
        assert!(text.contains("map_kernel"));
        let mut checked = 0;
        for entry in std::fs::read_dir(records.join("demo-crate")).unwrap() {
            let record = std::fs::read_to_string(entry.unwrap().path()).unwrap();
            for marker in TWIN_TEXT {
                assert!(!record.contains(marker), "{marker} in {record}");
            }
            checked += 1;
        }
        assert_eq!(checked, 2, "one .slang and one .kernel.json");
    }
}
