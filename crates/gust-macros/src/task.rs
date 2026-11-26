//! `gust::gpu_task!` (plan m14.s2, slice s2a1): a stage written at the call site, run as a free task.
//!
//! ```text
//! gust::gpu_task!(n, [a: &[f32], k: f32], |i| -> f32 { a[i] * k })           // 1-D, a result of type f32
//! gust::gpu_task!((w, h), [a: &[u32]], |x, y| -> u32 { a[x] + y })           // 2-D, the shape is a (u32, u32)
//! gust::gpu_task!(&mut data, [data: &mut [f32]], |i| { data[i] = data[i] * 0.5; })  // in place, no result type
//! ```
//!
//! The expression evaluates to a `gust::task::Task`, which is run once with `.submit()` (asynchronous read-back, every
//! target) or `.dispatch()` (waits, native targets).
//!
//! The macro lowers the call to the `par_for!` generator ([`crate::par_for`]) and adds to it only what a stage has
//! that a loop does not: a result buffer (the closure's tail expression is written to it, at the invocation's own
//! index, row-major for a 2-D stage), and host code that submits and reads back instead of launching and waiting. So
//! the kernel, its Slang, its artifacts, its records, its routing and its CPU twin are those of any kernel, named by
//! the content hash of the lowered body, and the export record is the `par_for!` one (no new kind).
//!
//! The diagnostics of the E10xx family that this slice owns are raised here, on the closure's own tokens, before the
//! generator sees the body: E1001 (not one of the forms), E1002 (a capture whose element type is not visible, or a name
//! the capture list does not have), E1003 (a write to a capture), E1004 (a `bool` element of a stage of a batch: a
//! `Stage<bool>` entry or a `bool` result type), E1007 (an index of the wrong rank), E1008 (an in-place stage reading
//! or writing its target at another index than its own, or binding its own index again). Everything else about the
//! body is the validator's, with its own codes.

use std::collections::HashSet;

use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, format_ident, quote, quote_spanned};
use syn::{
    Token,
    ext::IdentExt as _,
    parse::{Parse, ParseStream},
    spanned::Spanned as _,
    visit::Visit,
    visit_mut::VisitMut,
};

use crate::{
    diagnostics::{self, Code},
    expand::{ElementReject, Environment, Profile, element_types},
    par_for::{Binding, Kind, Options, Plan, Raw, RawBinding, Scalar, parse_options},
};

/// The name of the result buffer of a stage that has a result type: an ordinary binding of the plan, under a reserved
/// `__gust_` name that no capture can have (the validated view of the kernel gives it a name of its own, see
/// `par_for::Names`).
pub(crate) const OUT: &str = "__gust_task_out";

fn form(tokens: impl ToTokens, message: impl std::fmt::Display) -> syn::Error {
    diagnostics::error(Code::TaskForm, tokens, message)
}

/// The hidden clause `gpu_batch!` puts in front of the stages it rewrites: `batch(<builder>, "<name of the let>");`.
pub(crate) struct BatchClause {
    /// The builder of the block (`b` of `|b| { .. }`).
    pub(crate) builder: syn::Ident,
    /// The name of the stage's `let`; `None` for a stage that no `let` names.
    pub(crate) name: Option<String>,
    /// The enums the block's parameters name, as their derive handed them over (`enums { enum Mode { .. } }`): the
    /// stage declares a private copy of each one its parameters use.
    pub(crate) enums: Vec<syn::ItemEnum>,
}

/// The macro input as written: the options, the shape (or the in-place target), the capture list, the closure.
pub(crate) struct Input {
    batch: Option<BatchClause>,
    pub(crate) options: Options,
    pub(crate) shape: syn::Expr,
    pub(crate) list: Vec<RawBinding>,
    pub(crate) closure: syn::ExprClosure,
}

/// The hidden batch clause, when the input starts with one.
fn parse_batch_clause(input: ParseStream) -> syn::Result<Option<BatchClause>> {
    let starts = input
        .cursor()
        .ident()
        .is_some_and(|(keyword, _)| keyword == "batch")
        && input.peek2(syn::token::Paren);
    if !starts {
        return Ok(None);
    }
    input.parse::<syn::Ident>()?;
    let content;
    syn::parenthesized!(content in input);
    let builder: syn::Ident = content.parse()?;
    content.parse::<Token![,]>()?;
    let name: syn::LitStr = content.parse()?;
    input.parse::<Token![;]>()?;
    let name = name.value();
    let mut enums = Vec::new();
    if input
        .cursor()
        .ident()
        .is_some_and(|(keyword, _)| keyword == "enums")
        && input.peek2(syn::token::Brace)
    {
        input.parse::<syn::Ident>()?;
        let content;
        syn::braced!(content in input);
        while !content.is_empty() {
            enums.push(content.parse()?);
        }
    }
    Ok(Some(BatchClause {
        builder,
        name: (!name.is_empty()).then_some(name),
        enums,
    }))
}

impl Parse for Input {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let batch = parse_batch_clause(input)?;
        let options = parse_options(input)?;
        let shape: syn::Expr = input.parse()?;
        input.parse::<Token![,]>()?;
        let mut list = Vec::new();
        if input.peek(syn::token::Bracket) {
            let content;
            syn::bracketed!(content in input);
            list = content
                .parse_terminated(RawBinding::parse, Token![,])?
                .into_iter()
                .collect();
            input.parse::<Token![,]>()?;
        }
        let closure = match input.parse::<syn::Expr>()? {
            syn::Expr::Closure(closure) => closure,
            other => {
                return Err(form(
                    other,
                    "the last part of a `gpu_task!` is its closure: `|i| -> T { .. }`, `|x, y| -> T { .. }`, or \
                     `|i| { .. }` for an in-place stage",
                ));
            }
        };
        let _ = input.parse::<Option<Token![,]>>()?;
        if !input.is_empty() {
            return Err(form(
                input.cursor().token_stream(),
                "`gpu_task!` ends with its closure",
            ));
        }
        Ok(Self {
            batch,
            options,
            shape,
            list,
            closure,
        })
    }
}

/// How a stage's extent is found.
enum Extent {
    /// A free stage: one expression (1-D) or a pair that the call destructures (2-D).
    Shape(Box<syn::Expr>),
    /// An in-place stage: the length of its target.
    Target,
}

/// A `gpu_task!` understood, as far as the generated kernel and its host function need it.
pub(crate) struct TaskPlan {
    /// The number of closure parameters: 1 or 2.
    rank: usize,
    extent: Extent,
    /// The closure body's statements, before its tail (a free stage) or all of them (an in-place stage).
    stmts: Vec<syn::Stmt>,
    /// The closure body's result expression; `None` for an in-place stage.
    tail: Option<syn::Expr>,
    /// The in-place target: its name in the list and the place it is taken from.
    target: Option<(syn::Ident, syn::Expr)>,
    /// A stage of a `gpu_batch!`: what its captures are and where it is recorded.
    batch: Option<BatchStage>,
}

/// What a stage of a batch captures: handles of the block (flat views of its buffers), parameters, and host scalars.
struct BatchStage {
    builder: syn::Ident,
    name: Option<String>,
    /// The stage's own buffers in the order of the list: the handles it reads and, last, the in-place target.
    views: Vec<View>,
    /// The `Param`s the body reads (`p.get()`): the name, which is also the scalar's name in the kernel.
    params: Vec<syn::Ident>,
    /// The host scalars of the list: values taken when the stage is recorded.
    scalars: Vec<syn::Ident>,
    /// The enums the stage's parameters have: a private copy of each is declared in the stage's module (D270).
    enums: Vec<(syn::ItemEnum, syn::ItemFn)>,
    /// The parameters that are enums: the parameter and the enum it holds. The kernel's record has the word, and the
    /// kernel turns it into the enum before the body runs.
    enum_params: Vec<(syn::Ident, syn::Ident)>,
    /// Per enum path a parameter's type names, the statements that check the copy against it ([`enum_copy_check`]).
    enum_checks: Vec<TokenStream>,
    /// The views whose elements are enums (D275): the view and the enum's name. The kernel binds the words (`u32`), and
    /// a load of an element decodes the word through the enum's one decoder.
    enum_views: Vec<(syn::Ident, syn::Ident)>,
    /// The enum the stage's result holds, when it is one: the type as written and the enum's name. The kernel stores
    /// the word (`as u32`), and the call site types the handle back as the enum.
    enum_result: Option<(syn::Type, syn::Ident)>,
}

impl BatchStage {
    /// Whether the view `name` holds enum elements.
    fn is_enum_view(&self, name: &syn::Ident) -> bool {
        self.enum_views.iter().any(|(view, _)| view == name)
    }

    /// Whether the parameter `name` holds an enum.
    fn is_enum(&self, name: &syn::Ident) -> bool {
        self.enum_params.iter().any(|(param, _)| param == name)
    }
}

/// The name of the function that turns the word of an enum into the enum, in the module of a stage. Not a reserved
/// `__gust_` name, so the stage's validated view and its expanded module share it. The enum's exact name, so that two
/// enums that differ in case (`Mode` and `MODE`) keep two functions.
fn enum_decoder(definition: &syn::Ident) -> syn::Ident {
    format_ident!("par_for_enum_{}", definition.unraw())
}

/// The function that decodes the word of `definition`: an `if` and `return` per discriminant, the last variant being
/// what any other word gives. A parameter's word is always a variant's (`set` takes the enum); a stored element's word is
/// whatever a stage or the host put there, so the last variant is the rule for it too (D275).
fn enum_decoder_fn(definition: &syn::ItemEnum) -> syn::Result<syn::ItemFn> {
    let name = &definition.ident;
    let variants = enum_variants(definition)?;
    let Some(((last, _), earlier)) = variants.split_last() else {
        unreachable!("`enum_variants` refuses an enum with no variant")
    };
    let decoder = enum_decoder(name);
    let (words, idents): (Vec<u32>, Vec<&syn::Ident>) =
        earlier.iter().map(|(ident, word)| (*word, ident)).unzip();
    Ok(syn::parse_quote! {
        #[allow(non_snake_case)]
        fn #decoder(word: u32) -> #name {
            #(
                if word == #words {
                    return #name::#idents;
                }
            )*
            #name::#last
        }
    })
}

/// The variants of a handed-over enum with their words, in order: each has a discriminant that is an integer literal
/// (the derive writes them all out), and there is at least one.
fn enum_variants(definition: &syn::ItemEnum) -> syn::Result<Vec<(syn::Ident, u32)>> {
    let name = &definition.ident;
    let mut variants = Vec::new();
    for variant in &definition.variants {
        let word = match &variant.discriminant {
            Some((
                _,
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Int(literal),
                    ..
                }),
            )) => literal.base10_parse::<u32>()?,
            _ => {
                return Err(batch_form(
                    variant,
                    "an enum a batch parameter holds has a discriminant written on each variant: derive it with \
                     `#[derive(GpuEnum)]`",
                ));
            }
        };
        variants.push((variant.ident.clone(), word));
    }
    if variants.is_empty() {
        return Err(batch_form(
            name,
            "an enum a batch parameter holds has a variant",
        ));
    }
    Ok(variants)
}

/// The statements, in the user's scope, that check the enum a parameter's type names against the copy a stage was
/// given: every variant of the copy is a variant of `path` with the same word (`P::V as u32 == W`), and every variant of
/// `path` is one of the copy's (an exhaustive `match` in a `const fn`). The copy came from the macro that carries the
/// enum's name, and rustc resolves a macro and a type apart, so the two can differ (a `type` alias, a glob import, the
/// hidden form written by hand): the decoder would then read the words of one enum as the variants of another, with
/// no error at all. This is a compile error at the parameter's type instead.
fn enum_copy_check(path: &syn::Path, variants: &[(syn::Ident, u32)]) -> TokenStream {
    let (idents, words): (Vec<&syn::Ident>, Vec<u32>) =
        variants.iter().map(|(ident, word)| (ident, *word)).unzip();
    quote_spanned! {path.span()=>
        const _: () = {
            #(::core::assert!(#path::#idents as ::core::primitive::u32 == #words);)*
            #[allow(dead_code)]
            const fn __gust_enum_variants(value: #path) {
                match value {
                    #(#path::#idents => {})*
                }
            }
        };
    }
}

/// A handle of the block that a batch stage reads or writes.
struct View {
    name: syn::Ident,
    /// 1 for a flat view, 2 for a row-major view indexed by a pair.
    rank: usize,
    /// The in-place target.
    target: bool,
}

/// The name of the kernel scalar that holds the width of the 2-D view `name`.
fn width_name(name: &syn::Ident) -> syn::Ident {
    format_ident!("par_for_w_{}", name.unraw())
}

impl TaskPlan {
    /// The statements of the kernel body: the closure's own, then (a free stage) the write of its result at the
    /// invocation's own index. `gid`, `ext`, `width` and `out` are the names the module being built gives the dispatch
    /// id, the extent record, its first extent and the result buffer.
    pub(crate) fn statements(
        &self,
        gid: &syn::Ident,
        ext: &syn::Ident,
        width: &syn::Ident,
        out: &syn::Ident,
    ) -> Vec<syn::Stmt> {
        let mut statements = self.stmts.clone();
        if let Some(tail) = &self.tail {
            let index = if self.rank == 1 {
                quote!(#gid.x)
            } else {
                quote!(#gid.y * #ext.#width + #gid.x)
            };
            match self
                .batch
                .as_ref()
                .and_then(|batch| batch.enum_result.as_ref())
            {
                // An enum result is stored as its word: the value is bound with its type (a `match` value needs one).
                Some((_, name)) => {
                    let local = format_ident!("par_for_enum_result");
                    statements.push(syn::parse_quote!(let #local: #name = #tail;));
                    statements.push(syn::parse_quote!(#out[#index] = #local as u32;));
                }
                None => statements.push(syn::parse_quote!(#out[#index] = #tail;)),
            }
        }
        statements
    }

    /// The enums of the stage's parameters and the function that decodes each one's word, to be declared in the stage's
    /// module in front of the record and the kernel.
    pub(crate) fn enum_items(&self) -> Vec<syn::Item> {
        let Some(batch) = &self.batch else {
            return Vec::new();
        };
        batch
            .enums
            .iter()
            .flat_map(|(definition, decoder)| {
                [
                    syn::Item::Enum(definition.clone()),
                    syn::Item::Fn(decoder.clone()),
                ]
            })
            .collect()
    }

    /// The function that decodes the word of the enum parameter `name`, when it is one.
    pub(crate) fn enum_decoder_of(&self, name: &syn::Ident) -> Option<syn::Ident> {
        let (_, definition) = self
            .batch
            .as_ref()?
            .enum_params
            .iter()
            .find(|(param, _)| param == name)?;
        Some(enum_decoder(definition))
    }

    /// The name of the binding that the kernel writes its result to: the result buffer, or the in-place target.
    fn output_name(&self) -> syn::Ident {
        match &self.target {
            Some((name, _)) => name.clone(),
            None => format_ident!("{}", OUT),
        }
    }

    fn output<'a>(&self, plan: &'a Plan) -> &'a Binding {
        let name = self.output_name();
        plan.bindings
            .iter()
            .find(|binding| binding.name == name)
            .expect("the plan holds the output binding")
    }

    fn output_element(&self, plan: &Plan) -> TokenStream {
        match &self.output(plan).kind {
            Kind::Slice { element, .. } => element.host.clone(),
            _ => unreachable!("the output of a stage is a slice binding"),
        }
    }

    /// What the stage's future resolves to: the values read back, or nothing for an in-place stage.
    fn done(&self, plan: &Plan) -> TokenStream {
        if self.target.is_some() {
            quote!(())
        } else {
            let element = self.output_element(plan);
            quote!(::std::vec::Vec<#element>)
        }
    }

    /// The host function inside the generated module: it uploads the inputs, records the extent, and hands the closure
    /// that builds the kernel's `Call` to `gust::task::submit_and_read`, which makes the output buffer and submits the
    /// dispatch and the read-back through `MapHost::start_read` (one submission on a GPU device). (The one exception:
    /// an empty captured slice on a recycled pool buffer is cleared in its own submission, from generated code,
    /// through `MapArg::bind`.)
    pub(crate) fn host_items(&self, plan: &Plan) -> TokenStream {
        if self.batch.is_some() {
            return self.batch_items(plan);
        }
        let record = format_ident!("{}", "__GustExtent");
        let count = proc_macro2::Literal::usize_unsuffixed(1 + plan.buffers().count());
        let output = self.output_name();
        let output_element = self.output_element(plan);
        let done = self.done(plan);
        let lifetime = quote!('__gust_t);

        // The extents: a free stage takes one value per axis, an in-place stage the length of its target.
        let axes = ["x", "y"];
        let extent_types: Vec<syn::Ident> = (0..self.rank)
            .map(|axis| format_ident!("__GustE{}", axis))
            .collect();
        let extent_names: Vec<syn::Ident> = (0..self.rank)
            .map(|axis| format_ident!("__gust_e{}", axis))
            .collect();
        let (extent_generics, extent_parameters, extent_bounds, conversions) = match &self.extent {
            Extent::Shape(_) => {
                let bounds: Vec<TokenStream> = extent_types
                    .iter()
                    .map(|ty| {
                        quote! {
                            #ty: ::core::convert::TryInto<u32>,
                            <#ty as ::core::convert::TryInto<u32>>::Error:
                                ::core::error::Error + ::core::marker::Send + ::core::marker::Sync + 'static,
                        }
                    })
                    .collect();
                let parameters: Vec<TokenStream> = extent_names
                    .iter()
                    .zip(&extent_types)
                    .map(|(name, ty)| quote!(#name: #ty))
                    .collect();
                let conversions: Vec<TokenStream> = extent_names
                    .iter()
                    .zip(axes)
                    .map(|(name, axis)| {
                        let converted = format_ident!("__gust_{}", axis);
                        quote! {
                            let #converted = ::core::convert::TryInto::<u32>::try_into(#name)
                                .map_err(::gust::map::Error::device)?;
                        }
                    })
                    .collect();
                (extent_types.clone(), parameters, bounds, conversions)
            }
            Extent::Target => {
                let target = &self.output_name();
                let conversion = quote! {
                    let __gust_x = ::core::primitive::u32::try_from(#target.len())
                        .map_err(|_| ::gust::map::Error::TooLong { len: #target.len() })?;
                };
                (Vec::new(), Vec::new(), Vec::new(), vec![conversion])
            }
        };
        let value_x = quote!(__gust_x);
        let value_y = if self.rank == 2 {
            quote!(__gust_y)
        } else {
            quote!(1u32)
        };

        // The user's captures as host parameters, and the resource each kernel parameter is bound to.
        let host_parameters: Vec<TokenStream> = plan
            .bindings
            .iter()
            .filter(|binding| binding.name != format_ident!("{}", OUT))
            .map(|binding| {
                let name = &binding.name;
                match &binding.kind {
                    Kind::Slice { mutable, element } => {
                        let element = &element.host;
                        if *mutable {
                            quote!(#name: &#lifetime mut [#element])
                        } else {
                            quote!(#name: &#lifetime [#element])
                        }
                    }
                    Kind::Scalar(Scalar::U32) => quote!(#name: ::core::primitive::u32),
                    Kind::Scalar(Scalar::I32) => quote!(#name: ::core::primitive::i32),
                    Kind::Scalar(Scalar::F32) => quote!(#name: ::core::primitive::f32),
                    Kind::Scalar(Scalar::Bool) => quote!(#name: ::core::primitive::bool),
                    Kind::Resident { .. } => unreachable!("a stage has no resident capture"),
                }
            })
            .collect();
        let bound: Vec<syn::Ident> = (0..plan.buffers().count())
            .map(|index| format_ident!("__gust_b{}", index))
            .collect();
        // The uploaded captures (every buffer but the output), in the order of the kernel's parameters.
        let inputs: Vec<syn::Ident> = plan
            .buffers()
            .zip(&bound)
            .filter(|((name, ..), _)| **name != output)
            .map(|(_, local)| local.clone())
            .collect();
        let binds = plan
            .buffers()
            .zip(&bound)
            .filter(|((name, ..), _)| **name != output)
            .map(|((name, _, element, _), local)| {
                let element = &element.host;
                quote! {
                    let #local = ::gust::map::MapArg::<#element, __GustD>::bind(&*#name, &*__gust_device)?;
                }
            });
        let resources = plan.buffers().zip(&bound).map(|((name, _, element, _), local)| {
            let element = &element.host;
            if *name == output {
                quote!(::gust::Independent(__gust_out))
            } else {
                // `local` is the reference the call's context gives back: the uploaded capture, borrowed.
                quote! {
                    ::gust::Independent(::gust::map::resource::<#element, __GustD, &[#element]>(#local))
                }
            }
        });
        let scalar_values = plan.scalars().map(|(name, scalar)| {
            if scalar == Scalar::Bool {
                quote!(#name: ::core::primitive::u32::from(#name))
            } else {
                quote!(#name)
            }
        });
        let pads = (0..(4 - (3 + plan.scalars().count()) % 4) % 4).map(|index| {
            let pad = format_ident!("__gust_pad{}", index);
            quote!(#pad: 0u32)
        });

        // The result sink and the length of the output.
        let (sink, empty_sink, length) = match &self.extent {
            Extent::Shape(_) => (
                quote!(::gust::task::Fresh(__gust_len)),
                quote!(::gust::task::Fresh(0)),
                quote! {
                    let __gust_len = ::core::primitive::usize::try_from(
                        u64::from(#value_x) * u64::from(#value_y),
                    )
                    .map_err(|_| ::gust::map::Error::TooLong { len: ::core::primitive::usize::MAX })?;
                },
            ),
            Extent::Target => (
                quote!(::gust::task::InPlace(#output)),
                quote!(::gust::task::InPlace(#output)),
                TokenStream::new(),
            ),
        };
        let zero = if self.rank == 2 {
            quote!(#value_x == 0u32 || #value_y == 0u32)
        } else {
            quote!(#value_x == 0u32)
        };

        quote! {
            /// Upload the captures and start the stage; the read-back is the returned future.
            #[doc(hidden)]
            #[allow(clippy::too_many_arguments)]
            pub fn __gust_task<#lifetime, __GustH, __GustD, #(#extent_generics,)*>(
                __gust_device: __GustH,
                #(#extent_parameters,)*
                #(#host_parameters,)*
            ) -> ::core::result::Result<::gust::task::Pending<#lifetime, #done>, ::gust::map::Error>
            where
                __GustH: ::core::ops::Deref<Target = __GustD> + #lifetime,
                __GustD: ::gust::map::MapHost
                    + __GustPfHost
                    + ::gust::task::TaskDevice<run, #count>
                    + #lifetime,
                <__GustD as ::gust::map::MapHost>::Buffer<#output_element>: #lifetime,
                #(#extent_bounds)*
            {
                #(#conversions)*
                if #zero {
                    return ::core::result::Result::Ok(
                        ::gust::task::nothing_to_run::<#output_element, _>(#empty_sink),
                    );
                }
                #length
                let __gust_extent = ::gust::Extent::new(#value_x, #value_y, 1u32);
                #[cfg(not(target_arch = "wasm32"))]
                __gust_check_cap::<__GustD>(__gust_extent)?;
                let __gust_record = #record {
                    __gust_ex: #value_x,
                    __gust_ey: #value_y,
                    __gust_ez: 1u32,
                    #(#scalar_values,)*
                    #(#pads,)*
                };
                let __gust_uniform = <__GustD as __GustPfHost>::__gust_uniform(&*__gust_device, __gust_record)?;
                #(#binds)*
                ::gust::task::submit_and_read(
                    __gust_device,
                    #sink,
                    (__gust_uniform, (#(#inputs,)*)),
                    |__gust_context, __gust_out: &<__GustD as ::gust::map::MapHost>::Buffer<#output_element>| {
                        let (__gust_uniform, (#(#inputs,)*)) = __gust_context;
                        run::call_extent(__gust_extent, __gust_uniform, #(#resources,)*)
                    },
                )
            }
        }
    }

    /// The value of the call: the arguments are taken at the call (a slice is borrowed, a scalar copied, the in-place
    /// target reborrowed), and the task that holds them calls the host function when it is submitted.
    pub(crate) fn value(&self, plan: &Plan, module: &syn::Ident) -> TokenStream {
        if self.batch.is_some() {
            return self.batch_value(module);
        }
        let mut lets = TokenStream::new();
        let mut shape_arguments = Vec::new();
        if let Extent::Shape(shape) = &self.extent {
            if self.rank == 2 {
                lets.extend(quote!(let (__gust_s0, __gust_s1) = #shape;));
                shape_arguments.push(quote!(__gust_s0));
                shape_arguments.push(quote!(__gust_s1));
            } else {
                lets.extend(quote!(let __gust_s0 = #shape;));
                shape_arguments.push(quote!(__gust_s0));
            }
        }
        let mut arguments = Vec::new();
        for (index, binding) in plan
            .bindings
            .iter()
            .filter(|binding| binding.name != format_ident!("{}", OUT))
            .enumerate()
        {
            let name = &binding.name;
            let local = format_ident!("__gust_a{}", index);
            match &binding.kind {
                Kind::Slice { mutable: false, .. } => lets.extend(quote!(let #local = &#name[..];)),
                Kind::Slice { mutable: true, .. } => {
                    let (_, place) = self.target.as_ref().expect("a mutable slice is the target");
                    lets.extend(quote!(let #local = &mut (#place)[..];));
                }
                Kind::Scalar(_) => lets.extend(quote!(let #local = #name;)),
                Kind::Resident { .. } => unreachable!("a stage has no resident capture"),
            }
            arguments.push(local);
        }
        let (device_let, device) = match &plan.options.on {
            Some(device) => (quote!(let __gust_given = #device;), quote!(__gust_given)),
            None if plan.is_cpu() => (
                TokenStream::new(),
                quote!(::std::sync::Arc::new(::gust::cpu::CpuDevice::new())),
            ),
            None => (
                TokenStream::new(),
                quote!(::gust_wgpu::default_device().map_err(::gust::map::Error::device)?),
            ),
        };
        quote! {
            {
                #device_let
                #lets
                ::gust::task::Task::new(move || {
                    #module::__gust_task(#device, #(#shape_arguments,)* #(#arguments),*)
                })
            }
        }
    }
}

/// The name a pattern of a closure parameter gives an axis: a plain identifier or `_`.
fn parameter_axis(pattern: &syn::Pat) -> syn::Result<syn::Pat> {
    match pattern {
        syn::Pat::Ident(ident)
            if ident.by_ref.is_none() && ident.mutability.is_none() && ident.subpat.is_none() =>
        {
            Ok(pattern.clone())
        }
        syn::Pat::Wild(_) => Ok(pattern.clone()),
        other => Err(form(
            other,
            "a stage's closure parameters are plain identifiers (or `_`): `|i|` or `|x, y|`",
        )),
    }
}

/// The root name of a place: `a` of `a`, `a[i]`, `a[i].x` and `(a)`.
fn root_name(expression: &syn::Expr) -> Option<&syn::Ident> {
    match expression {
        syn::Expr::Path(path) if path.qself.is_none() => path.path.get_ident(),
        syn::Expr::Index(index) => root_name(&index.expr),
        syn::Expr::Field(field) => root_name(&field.base),
        syn::Expr::Paren(paren) => root_name(&paren.expr),
        _ => None,
    }
}

/// Whether `expression` is exactly the path `name` (parentheses ignored).
fn is_name(expression: &syn::Expr, name: &syn::Ident) -> bool {
    match expression {
        syn::Expr::Paren(paren) => is_name(&paren.expr, name),
        syn::Expr::Path(path) if path.qself.is_none() => {
            path.path.get_ident().is_some_and(|ident| ident == name)
        }
        _ => false,
    }
}

/// The names that the body itself binds: a `let`, a `for`, a closure parameter and a `match` arm (a pattern), and the
/// items declared in it. Every scope counts as one, so a name is "unknown" only when nothing in the body ever binds it.
#[derive(Default)]
struct Bound(HashSet<String>);

impl<'ast> Visit<'ast> for Bound {
    fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
        self.0.insert(pattern.ident.unraw().to_string());
        syn::visit::visit_pat_ident(self, pattern);
    }

    fn visit_item_const(&mut self, item: &'ast syn::ItemConst) {
        self.0.insert(item.ident.unraw().to_string());
        syn::visit::visit_item_const(self, item);
    }

    fn visit_item_static(&mut self, item: &'ast syn::ItemStatic) {
        self.0.insert(item.ident.unraw().to_string());
        syn::visit::visit_item_static(self, item);
    }

    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.0.insert(item.sig.ident.unraw().to_string());
        syn::visit::visit_item_fn(self, item);
    }
}

/// Reads the closure's tokens for the rules of the E10xx family; the first rule broken is the error.
struct Rules<'a> {
    /// The names of the captures, read-only (host slices and scalars).
    captures: &'a [syn::Ident],
    /// The names that are host slices (captures and the target): the views a body indexes.
    views: &'a [syn::Ident],
    /// The in-place target, when there is one.
    target: Option<&'a syn::Ident>,
    /// The closure's own index of an in-place stage.
    own: Option<&'a syn::Ident>,
    /// The names a value path of the body may have: the closure's parameters, the list's entries and every name that a
    /// pattern or an item of the body binds.
    known: &'a HashSet<String>,
    error: Option<syn::Error>,
}

impl Rules<'_> {
    fn refuse(&mut self, code: Code, tokens: impl ToTokens, message: impl std::fmt::Display) {
        if self.error.is_none() {
            self.error = Some(diagnostics::error(code, tokens, message));
        }
    }

    fn is_capture(&self, name: &syn::Ident) -> bool {
        self.captures
            .iter()
            .any(|capture| capture.unraw() == name.unraw())
    }

    fn is_target(&self, name: &syn::Ident) -> bool {
        self.target
            .is_some_and(|target| target.unraw() == name.unraw())
    }

    fn is_view(&self, name: &syn::Ident) -> bool {
        self.views.iter().any(|view| view.unraw() == name.unraw())
    }

    /// A value path of the body. A single name that is not a parameter, a capture or a local is a capture the list does
    /// not have: the kernel sees nothing of the call site, so the entry is the only way to name it.
    fn used(&mut self, path: &syn::ExprPath, indexed: bool) {
        let Some(ident) = path.path.get_ident().filter(|_| path.qself.is_none()) else {
            return;
        };
        let name = ident.unraw().to_string();
        if self.known.contains(&name)
            || matches!(name.as_str(), "self" | "Self" | "super" | "crate")
        {
            return;
        }
        // What the body does with the name tells the form: an index use is a host slice, any other use a scalar.
        let entry = if indexed {
            format!(
                "write the entry `{name}: &[f32]` for a host slice (with the element type of `{name}` in place of `f32`)"
            )
        } else {
            format!(
                "write the entry `{name}: f32` for a scalar (with the type of `{name}`, `u32`, `i32`, `f32` or `bool`, \
                 in place of `f32`)"
            )
        };
        self.refuse(
            Code::TaskCapture,
            ident,
            format!(
                "`{name}` is used in the body, but a stage sees only its parameters and the entries of its capture \
                 list: {entry}"
            ),
        );
    }

    /// A binding (a `let`, a `match` or `if let` scrutinee) of a bare view or target name in an in-place body: the new
    /// name would index the target where E1008 cannot see it (`let d = data; d[i + 1u32] = 0.0;`).
    fn aliased(&mut self, value: &syn::Expr) {
        if self.target.is_none() {
            return;
        }
        let mut value = value;
        while let syn::Expr::Paren(paren) = value {
            value = &paren.expr;
        }
        let syn::Expr::Path(path) = value else {
            return;
        };
        if let Some(name) = path
            .path
            .get_ident()
            .filter(|name| path.qself.is_none() && self.is_view(name))
        {
            self.refuse(
                Code::TaskInPlaceRead,
                value,
                format!(
                    "an in-place stage does not bind `{0}` to another name: a read or a write through the alias would \
                     reach `{0}` at an index the check cannot see, so the GPU and the CPU twin could differ; index \
                     `{0}` itself",
                    name.unraw()
                ),
            );
        }
    }

    /// An assignment (or a compound one) to `place`.
    fn written(&mut self, place: &syn::Expr) {
        if let Some(name) = root_name(place).filter(|name| self.is_capture(name)) {
            self.refuse(
                Code::TaskWrite,
                place,
                format!(
                    "`{}` is a capture, and a stage's captures are read-only; only an in-place stage writes, and only \
                     its target (`gpu_task!(&mut target, [target: &mut [T]], |i| {{ .. }})`)",
                    name.unraw()
                ),
            );
        }
    }
}

impl<'ast> Visit<'ast> for Rules<'_> {
    fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
        if let Some(own) = self.own.filter(|own| own.unraw() == pattern.ident.unraw()) {
            self.refuse(
                Code::TaskInPlaceRead,
                &pattern.ident,
                format!(
                    "an in-place stage's own index `{0}` is not bound again in its body (a `let`, a `for`, a closure \
                     parameter or a `match` arm named `{0}`): a read or a write of its target at a shifted index \
                     would hide behind the name, so the GPU and the CPU twin would differ; give the new value another \
                     name and index the target with `{0}`",
                    own.unraw()
                ),
            );
        }
        syn::visit::visit_pat_ident(self, pattern);
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let Some(init) = &local.init {
            self.aliased(&init.expr);
        }
        syn::visit::visit_local(self, local);
    }

    fn visit_expr(&mut self, expression: &'ast syn::Expr) {
        match expression {
            syn::Expr::Match(matched) => self.aliased(&matched.expr),
            syn::Expr::Let(binding) => self.aliased(&binding.expr),
            _ => {}
        }
        match expression {
            // The callee of a call is a function of the dialect, not a value: only its arguments are names used.
            syn::Expr::Call(call) if matches!(&*call.func, syn::Expr::Path(_)) => {
                for argument in &call.args {
                    self.visit_expr(argument);
                }
                return;
            }
            syn::Expr::Path(path) => self.used(path, false),
            syn::Expr::Assign(assign) => self.written(&assign.left),
            syn::Expr::Binary(binary) if is_compound_assignment(&binary.op) => {
                self.written(&binary.left);
            }
            syn::Expr::Reference(reference) if reference.mutability.is_some() => {
                let on_target = root_name(&reference.expr).is_some_and(|name| self.is_target(name));
                if !on_target {
                    self.refuse(
                        Code::TaskWrite,
                        reference,
                        "`&mut` is for the stage's own target only; a stage's captures are read-only and it has no \
                         other place to borrow mutably",
                    );
                }
            }
            syn::Expr::Index(index) => {
                // `a[..]` of a name nothing binds: the slice form is the entry to write.
                let mut base = &*index.expr;
                while let syn::Expr::Paren(paren) = base {
                    base = &paren.expr;
                }
                if let syn::Expr::Path(path) = base {
                    self.used(path, true);
                }
                if let Some(name) = root_name(&index.expr).filter(|name| self.is_view(name)) {
                    let mut at = &*index.index;
                    while let syn::Expr::Paren(paren) = at {
                        at = &paren.expr;
                    }
                    if matches!(at, syn::Expr::Tuple(_)) {
                        self.refuse(
                            Code::TaskIndexRank,
                            &index.index,
                            format!(
                                "`{}` is a host slice, a flat view of rank 1: index it with one index, `{}[i]` (a \
                                 view of rank 2 is a stage's result, which a later slice lets a stage read)",
                                name.unraw(),
                                name.unraw()
                            ),
                        );
                    } else if self.is_target(name) && !self.own.is_some_and(|own| is_name(at, own))
                    {
                        self.refuse(
                            Code::TaskInPlaceRead,
                            &index.index,
                            format!(
                                "an in-place stage reads or writes its target `{0}` only at its own index (`{0}[{1}]`): \
                                 another element may be mid-write on the GPU while the CPU twin reads a settled \
                                 value, so the two would differ; read a neighbour from a second buffer",
                                name.unraw(),
                                self.own.map_or_else(String::new, |own| own.unraw().to_string()),
                            ),
                        );
                    }
                }
            }
            _ => {}
        }
        syn::visit::visit_expr(self, expression);
    }
}

fn is_compound_assignment(op: &syn::BinOp) -> bool {
    matches!(
        op,
        syn::BinOp::AddAssign(_)
            | syn::BinOp::SubAssign(_)
            | syn::BinOp::MulAssign(_)
            | syn::BinOp::DivAssign(_)
            | syn::BinOp::RemAssign(_)
            | syn::BinOp::BitXorAssign(_)
            | syn::BinOp::BitAndAssign(_)
            | syn::BinOp::BitOrAssign(_)
            | syn::BinOp::ShlAssign(_)
            | syn::BinOp::ShrAssign(_)
    )
}

/// GUST-E1004: a `bool` as the element of a stage of a batch. `what` names the thing that has it.
fn bool_element(tokens: impl ToTokens, what: &str) -> syn::Error {
    diagnostics::error(
        Code::BatchElementBool,
        tokens,
        format!(
            "{what} has `bool` as its element, and a buffer holds a `bool` as a `u32` word: use `u32` (`0` and `1`); a \
             `#[derive(GpuEnum)]` enum is a parameter type or an element of a batch (`b.param::<Mode>(..)`, \
             `b.input::<Mode>(..)`), which a `bool` is not"
        ),
    )
}

/// What the list's entry `name: ty` is: checked for the forms a stage takes, with the codes of this slice. A capture is
/// a host slice `&[T]` of a buffer element type, or a scalar `u32`, `i32`, `f32` or `bool`; `&mut [T]` is the in-place
/// target's entry only. In a stage of a batch (`batch`) a `bool` element is GUST-E1004 where a stand-alone stage has
/// GUST-E1002.
fn check_entry(entry: &RawBinding, target: Option<&syn::Ident>, batch: bool) -> syn::Result<()> {
    let name = entry.name.unraw().to_string();
    let write = |element: &str| format!("`{name}: &[{element}]`");
    match &entry.ty {
        syn::Type::Reference(reference) => match reference.elem.as_ref() {
            syn::Type::Slice(slice) => {
                let is_target = target.is_some_and(|target| target.unraw() == entry.name.unraw());
                if reference.mutability.is_some() && !is_target {
                    return Err(diagnostics::error(
                        Code::TaskWrite,
                        &entry.ty,
                        format!(
                            "`{name}` is a `&mut [T]` host slice, and a stage's captures are read-only: write \
                             `{name}: &[T]`, or make `{name}` the target of an in-place stage with \
                             `gpu_task!(&mut {name}, ..)`"
                        ),
                    ));
                }
                match element_types(&slice.elem) {
                    Ok(types) if !types.boolean => Ok(()),
                    Ok(_) if batch => Err(bool_element(&slice.elem, &format!("`{name}`"))),
                    Ok(_) => Err(diagnostics::error(
                        Code::TaskCapture,
                        &slice.elem,
                        format!(
                            "`{name}` is a `bool` slice, and a buffer holds `u32` words: write {}",
                            write("u32")
                        ),
                    )),
                    Err(ElementReject::ThreeWide(_)) => Ok(()),
                    Err(ElementReject::Unsupported) => Err(diagnostics::error(
                        Code::TaskCapture,
                        &slice.elem,
                        format!(
                            "the macro cannot see an element type it can use for `{name}`: write the list entry as \
                             {} with a buffer element type (`f32`, `i32`, `u32`, `Vec2`, `Vec4`, `IVec2`, `IVec4`, \
                             `UVec2`, `UVec4`, `Mat2` or `Mat4`)",
                            write("f32")
                        ),
                    )),
                }
            }
            syn::Type::Path(path)
                if path
                    .path
                    .segments
                    .last()
                    .is_some_and(|segment| segment.ident == "GpuVec") =>
            {
                Err(form(
                    &entry.ty,
                    format!(
                        "`{name}` is a `GpuVec`: a free task captures host slices (`&[T]`) and scalars; a resident \
                         buffer arrives with the batch"
                    ),
                ))
            }
            _ => Err(diagnostics::error(
                Code::TaskCapture,
                &entry.ty,
                format!(
                    "the macro cannot see an element type for `{name}`: write the list entry as {} (a slice, not a \
                     `Vec` or an array)",
                    write("f32")
                ),
            )),
        },
        syn::Type::Path(path)
            if path.qself.is_none()
                && path.path.get_ident().is_some_and(|ident| {
                    matches!(
                        ident.unraw().to_string().as_str(),
                        "u32" | "uint" | "i32" | "int" | "f32" | "float" | "bool"
                    )
                }) =>
        {
            Ok(())
        }
        other => Err(diagnostics::error(
            Code::TaskCapture,
            other,
            format!(
                "the macro cannot see what `{name}` is: write the list entry as a host slice {} or a scalar \
                 (`{name}: u32`, `i32`, `f32` or `bool`)",
                write("f32")
            ),
        )),
    }
}

/// Everything the macro makes of the call, or the first error.
fn plan_of(input: Input, profile: Profile) -> syn::Result<Plan> {
    let Input {
        batch,
        options,
        shape,
        list,
        closure,
    } = input;
    if options.descriptor {
        return Err(form(&shape, "`descriptor;` is not a clause of `gpu_task!`"));
    }
    if let (Some(device), Some(_)) = (&options.on, &batch) {
        return Err(diagnostics::error(
            Code::BatchForm,
            device,
            "a stage of a batch runs on the batch's device: `on(..)` goes on the `gpu_batch!`",
        ));
    }
    // The closure: no `move`, `async`, `const` or `static`; one or two plain parameters.
    if closure.asyncness.is_some()
        || closure.constness.is_some()
        || closure.movability.is_some()
        || closure.capture.is_some()
        || closure.lifetimes.is_some()
    {
        return Err(form(
            &closure,
            "a stage's closure is a plain closure: no `move`, `async`, `const` or `static`",
        ));
    }
    let parameters = closure
        .inputs
        .iter()
        .map(parameter_axis)
        .collect::<syn::Result<Vec<_>>>()?;
    let rank = parameters.len();
    if !(1..=2).contains(&rank) {
        return Err(form(
            &closure.inputs,
            "a stage has one closure parameter (`|i|`, 1-D) or two (`|x, y|`, 2-D)",
        ));
    }
    let pattern: syn::Pat = if rank == 1 {
        parameters[0].clone()
    } else {
        let (x, y) = (&parameters[0], &parameters[1]);
        syn::parse_quote!((#x, #y))
    };

    // Free stage or in place.
    let target = match &shape {
        syn::Expr::Reference(reference) if reference.mutability.is_some() => {
            let place = (*reference.expr).clone();
            let Some(name) = root_name(&place).filter(|_| matches!(place, syn::Expr::Path(_)))
            else {
                return Err(form(
                    reference,
                    "the in-place form is `gpu_task!(&mut target, [..], |i| { .. })`, with the target a variable",
                ));
            };
            Some((name.clone(), place))
        }
        _ => None,
    };
    let result = match (&closure.output, &target) {
        (syn::ReturnType::Type(_, ty), None) => Some((**ty).clone()),
        (syn::ReturnType::Default, None) => {
            return Err(form(
                &closure,
                "a stage names its result type, as in `|i| -> f32 { .. }`; a stage that writes its own target \
                 takes `&mut target` as its shape instead",
            ));
        }
        (syn::ReturnType::Type(..), Some(_)) => {
            return Err(form(
                &closure.output,
                "an in-place stage has no result type: its result is its target",
            ));
        }
        (syn::ReturnType::Default, Some(_)) => None,
    };
    if target.is_some() && rank != 1 {
        return Err(form(
            &closure.inputs,
            "an in-place stage is 1-D in this release (`|i| { .. }`): a 2-D stage names its shape and its result type",
        ));
    }
    let extent = if target.is_some() {
        Extent::Target
    } else {
        match (&shape, rank) {
            (syn::Expr::Tuple(_), 1) => {
                return Err(form(
                    &shape,
                    "a 1-D stage's shape is one extent; write `|x, y|` for a 2-D stage",
                ));
            }
            (syn::Expr::Tuple(tuple), 2) if tuple.elems.len() != 2 => {
                return Err(form(
                    &shape,
                    "a 2-D stage's shape is a pair `(width, height)`",
                ));
            }
            (syn::Expr::Lit(_), 2) => {
                return Err(form(
                    &shape,
                    "a 2-D stage's shape is a pair `(width, height)`; write `|i|` for a 1-D stage",
                ));
            }
            _ => Extent::Shape(Box::new(shape.clone())),
        }
    };

    // The body: a block; a free stage ends with its result expression.
    let syn::Expr::Block(block) = *closure.body.clone() else {
        return Err(form(
            &closure.body,
            "the body of a stage is a block: `|i| -> T { .. }`",
        ));
    };
    let mut stmts = block.block.stmts;
    let mut tail = if result.is_some() {
        match stmts.pop() {
            Some(syn::Stmt::Expr(expression, None)) => Some(expression),
            _ => {
                return Err(diagnostics::error_at(
                    Code::TaskForm,
                    block.block.brace_token.span.join(),
                    "a stage's body ends with its result expression, without a `;`",
                ));
            }
        }
    } else {
        None
    };

    // A stage of a batch: its handles, parameters and the body lowered to what the kernel sees.
    let mut result = result;
    let (list, batch) = match batch {
        Some(clause) => {
            let lowered = lower_batch(clause, list, result.as_ref(), &mut stmts, &mut tail)?;
            if let Some(word) = lowered.result {
                result = Some(word);
            }
            (lowered.list, Some(lowered.stage))
        }
        None => (list, None),
    };

    // The list: each entry is a form this slice takes; an in-place target has its `&mut [T]` entry.
    let target_name = target.as_ref().map(|(name, _)| name);
    for entry in &list {
        check_entry(entry, target_name, batch.is_some())?;
    }
    if let Some((name, _)) = &target {
        let entry = list.iter().find(|entry| entry.name.unraw() == name.unraw());
        let Some(entry) = entry else {
            let n = name.unraw().to_string();
            return Err(diagnostics::error(
                Code::TaskCapture,
                name,
                format!(
                    "the in-place target `{n}` has no entry in the list, so the macro cannot see its element \
                     type: write `{n}: &mut [f32]` (with the element type of `{n}`) in the capture list"
                ),
            ));
        };
        if !matches!(&entry.ty, syn::Type::Reference(reference) if reference.mutability.is_some()) {
            let n = name.unraw().to_string();
            return Err(diagnostics::error(
                Code::TaskCapture,
                &entry.ty,
                format!(
                    "the in-place target's entry is `{n}: &mut [T]`, which is how the stage may write it"
                ),
            ));
        }
    }

    // The rules of the body.
    let captures: Vec<syn::Ident> = list
        .iter()
        .filter(|entry| target_name.is_none_or(|target| target.unraw() != entry.name.unraw()))
        .map(|entry| entry.name.clone())
        .collect();
    let views: Vec<syn::Ident> = list
        .iter()
        .filter(|entry| matches!(entry.ty, syn::Type::Reference(_)))
        .map(|entry| entry.name.clone())
        .collect();
    let own = match (&target, &parameters[0]) {
        (Some(_), syn::Pat::Ident(ident)) => Some(&ident.ident),
        _ => None,
    };
    let mut known = Bound::default();
    for statement in &stmts {
        known.visit_stmt(statement);
    }
    if let Some(tail) = &tail {
        known.visit_expr(tail);
    }
    let mut known = known.0;
    known.extend(list.iter().map(|entry| entry.name.unraw().to_string()));
    known.extend(parameters.iter().filter_map(|axis| match axis {
        syn::Pat::Ident(ident) => Some(ident.ident.unraw().to_string()),
        _ => None,
    }));
    let mut rules = Rules {
        captures: &captures,
        views: &views,
        target: target_name,
        own,
        known: &known,
        error: None,
    };
    for statement in &stmts {
        rules.visit_stmt(statement);
    }
    if let Some(tail) = &tail {
        rules.visit_expr(tail);
    }
    if let Some(error) = rules.error {
        return Err(error);
    }

    // The generator's input: the list, then the result buffer.
    let mut bindings = list;
    if let Some(ty) = &result {
        if batch.is_some() && element_types(ty).is_ok_and(|types| types.boolean) {
            return Err(bool_element(ty, "the result type of a stage"));
        }
        crate::par_for::element(ty, "the result type of a stage")?;
        bindings.push(RawBinding {
            name: format_ident!("{}", OUT),
            ty: syn::parse_quote!(&mut [#ty]),
        });
    }
    let ranges: syn::Expr = if rank == 1 {
        syn::parse_quote!(0..__gust_n)
    } else {
        syn::parse_quote!((0..__gust_w, 0..__gust_h))
    };
    let raw = Raw {
        options,
        pattern,
        ranges,
        bindings,
        body: syn::parse_quote!({}),
    };
    let mut plan = Plan::new(raw, profile)?;
    plan.task = Some(TaskPlan {
        rank,
        extent,
        stmts,
        tail,
        target,
        batch,
    });
    Ok(plan)
}

pub(crate) fn batch_form(tokens: impl ToTokens, message: impl std::fmt::Display) -> syn::Error {
    diagnostics::error(Code::BatchForm, tokens, message)
}

/// The `T` of `Name<T>` (or `Name<T, R>`: the rank is the body's to say, not written).
pub(crate) fn type_argument<'a>(ty: &'a syn::Type, name: &str) -> Option<&'a syn::Type> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    let segment = path.path.segments.last()?;
    if segment.ident != name {
        return None;
    }
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    match arguments.args.first()? {
        syn::GenericArgument::Type(inner) => Some(inner),
        _ => None,
    }
}

/// Whether `ty` is `u32`, `i32`, `f32` or `bool`.
pub(crate) fn is_scalar_type(ty: &syn::Type) -> bool {
    matches!(ty, syn::Type::Path(path) if path.qself.is_none()
        && path.path.get_ident().is_some_and(|ident| matches!(ident.to_string().as_str(), "u32" | "i32" | "f32" | "bool")))
}

/// A stage of a batch lowered to what the kernel sees: its list in kernel terms, and what its captures are.
struct Lowered {
    list: Vec<RawBinding>,
    /// The type the result buffer is bound as, when it is not the type written (an enum result is bound as `u32`).
    result: Option<syn::Type>,
    stage: BatchStage,
}

/// The handles and parameters `gpu_batch!` wrote into a stage's list (`h: Stage<T>`, `p: Param<T>`, `t: &mut
/// Stage<T>`), and the body lowered: a 2-D view `h[(x, y)]` is a flat `h[y * w + x]` over the width `par_for_w_h` that
/// the extent record holds, and `p.get()` is the scalar `p`.
fn lower_batch(
    clause: BatchClause,
    list: Vec<RawBinding>,
    result: Option<&syn::Type>,
    stmts: &mut [syn::Stmt],
    tail: &mut Option<syn::Expr>,
) -> syn::Result<Lowered> {
    let mut entries = Vec::new();
    let mut views = Vec::new();
    let mut params = Vec::new();
    let mut scalars = Vec::new();
    let mut book = EnumBook::default();
    let mut enum_params = Vec::new();
    let mut enum_views = Vec::new();
    for entry in list {
        let RawBinding { name, ty } = entry;
        if let Some(element) = type_argument(&ty, "Stage") {
            views.push(View {
                name: name.clone(),
                rank: 1,
                target: false,
            });
            // A view of enum elements: the kernel binds the words, and a load decodes them.
            if let Some((path, definition)) = enum_named(&clause.enums, element) {
                require_zero_word(element, definition)?;
                book.take(path, definition)?;
                enum_views.push((name.clone(), definition.ident.clone()));
                entries.push(RawBinding {
                    name,
                    ty: syn::parse_quote!(&[u32]),
                });
                continue;
            }
            let element = element.clone();
            entries.push(RawBinding {
                name,
                ty: syn::parse_quote!(&[#element]),
            });
            continue;
        }
        if let syn::Type::Reference(reference) = &ty {
            if let (Some(_), Some(element)) = (
                reference.mutability,
                type_argument(&reference.elem, "Stage"),
            ) {
                if enum_named(&clause.enums, element).is_some() {
                    return Err(diagnostics::error(
                        Code::TaskCapture,
                        element,
                        format!(
                            "`{}` is an in-place target of enum elements, which is not in this release: a stage reads \
                             enum elements and writes them as its result (`|i| -> {}`), and an in-place stage takes a \
                             numeric element",
                            name.unraw(),
                            element.to_token_stream()
                        ),
                    ));
                }
                views.push(View {
                    name: name.clone(),
                    rank: 1,
                    target: true,
                });
                let element = element.clone();
                entries.push(RawBinding {
                    name,
                    ty: syn::parse_quote!(&mut [#element]),
                });
                continue;
            }
            if let syn::Type::Slice(slice) = &*reference.elem
                && enum_named(&clause.enums, &slice.elem).is_some()
            {
                return Err(diagnostics::error(
                    Code::TaskCapture,
                    &slice.elem,
                    format!(
                        "`{}` is a host slice of enum elements, which a stage does not capture: write the enums into \
                         an input (`b.input::<{}>(..)`) and read the input",
                        name.unraw(),
                        slice.elem.to_token_stream()
                    ),
                ));
            }
            if matches!(*reference.elem, syn::Type::Slice(_)) {
                return Err(batch_form(
                    &ty,
                    format!(
                        "`{}` is a host slice, and a batch captures the handles of its block and scalars: a host slice \
                         arrives in a later release (s2g); upload it into an `input` and write the input",
                        name.unraw()
                    ),
                ));
            }
        }
        if let Some(element) = type_argument(&ty, "Param") {
            // An enum the block's derive handed over: the record holds its word, and the kernel decodes it.
            if let Some((path, definition)) = enum_named(&clause.enums, element) {
                params.push(name.clone());
                enum_params.push((name.clone(), definition.ident.clone()));
                book.take(path, definition)?;
                entries.push(RawBinding {
                    name,
                    ty: syn::parse_quote!(u32),
                });
                continue;
            }
            if !clause.enums.is_empty() && crate::batch::is_user_type(element) {
                let handed: Vec<String> = clause
                    .enums
                    .iter()
                    .map(|definition| format!("`{}`", definition.ident.unraw()))
                    .collect();
                return Err(batch_form(
                    element,
                    format!(
                        "`{}` is not the name of an enum the block's derives handed over ({}): write the enum's own \
                         name as the type of a parameter or an element (a `use .. as` rename and a `type` alias are not \
                         supported)",
                        element.to_token_stream(),
                        handed.join(", ")
                    ),
                ));
            }
            if !is_scalar_type(element) {
                return Err(batch_form(
                    element,
                    format!(
                        "`{}` is a parameter a body cannot read: only a `u32`, `i32`, `f32`, `bool` or enum parameter \
                         is read (a `(u32, u32)` parameter is a shape, which sizes a stage)",
                        name.unraw()
                    ),
                ));
            }
            params.push(name.clone());
            let element = element.clone();
            entries.push(RawBinding { name, ty: element });
            continue;
        }
        if is_scalar_type(&ty) {
            scalars.push(name.clone());
        }
        entries.push(RawBinding { name, ty });
    }
    // A result of enum elements: the kernel stores its word.
    let mut enum_result = None;
    let mut result_word = None;
    if let Some(ty) = result
        && let Some((path, definition)) = enum_named(&clause.enums, ty)
    {
        require_zero_word(ty, definition)?;
        book.take(path, definition)?;
        enum_result = Some((ty.clone(), definition.ident.clone()));
        result_word = Some(syn::parse_quote!(u32));
    }
    let mut lowering = Lowering {
        views: views
            .iter()
            .map(|view| {
                let decoder = enum_views
                    .iter()
                    .find(|(name, _)| *name == view.name)
                    .map(|(_, definition)| enum_decoder(definition));
                (view.name.clone(), view.target, None, decoder)
            })
            .collect(),
        params: params.clone(),
        lvalue: false,
        error: None,
    };
    for statement in stmts.iter_mut() {
        lowering.visit_stmt_mut(statement);
    }
    if let Some(tail) = tail {
        lowering.visit_expr_mut(tail);
    }
    if let Some(error) = lowering.error {
        return Err(error);
    }
    for (view, (_, _, rank, _)) in views.iter_mut().zip(&lowering.views) {
        view.rank = rank.unwrap_or(1);
        if view.rank == 2 {
            entries.push(RawBinding {
                name: width_name(&view.name),
                ty: syn::parse_quote!(u32),
            });
        }
    }
    Ok(Lowered {
        list: entries,
        result: result_word,
        stage: BatchStage {
            builder: clause.builder,
            name: clause.name,
            views,
            params,
            scalars,
            enums: book.enums,
            enum_params,
            enum_checks: book.checks,
            enum_views,
            enum_result,
        },
    })
}

/// The enums a stage declares a private copy of, and the checks of each copy against the user's enum: once per enum
/// name, and once per path the enum is written with.
#[derive(Default)]
struct EnumBook {
    enums: Vec<(syn::ItemEnum, syn::ItemFn)>,
    checks: Vec<TokenStream>,
    paths: Vec<String>,
}

impl EnumBook {
    /// Declare the copy of `definition`, the enum that `path` names, unless it is declared already.
    fn take(&mut self, path: &syn::Path, definition: &syn::ItemEnum) -> syn::Result<()> {
        if !self
            .enums
            .iter()
            .any(|(copy, _)| copy.ident == definition.ident)
        {
            let decoder = enum_decoder_fn(definition)?;
            let mut copy = definition.clone();
            copy.attrs.push(syn::parse_quote!(#[derive(PartialEq)]));
            self.enums.push((copy, decoder));
        }
        let written = path.to_token_stream().to_string();
        if !self.paths.contains(&written) {
            self.checks
                .push(enum_copy_check(path, &enum_variants(definition)?));
            self.paths.push(written);
        }
        Ok(())
    }
}

/// GUST-E1012: an enum stored as an element has a variant whose word is 0, because a buffer starts (and is resized) as
/// zeros. Raised at the element type's token. `definition` is the enum a handed-over derive wrote, so each variant has a
/// literal word.
pub(crate) fn require_zero_word(ty: impl ToTokens, definition: &syn::ItemEnum) -> syn::Result<()> {
    if enum_variants(definition)?
        .iter()
        .any(|(_, word)| *word == 0)
    {
        return Ok(());
    }
    Err(diagnostics::error(
        Code::BatchEnumNoZero,
        ty,
        format!(
            "`{0}` is stored as the element of a buffer, and a buffer starts as zeros (and a resize makes zeroed \
             buffers), so `{0}` needs a variant whose word is 0 (`Name = 0`): add one, or keep `{0}` a parameter",
            definition.ident.unraw()
        ),
    ))
}

/// The enum, among those the block's parameters name, that `ty` is: a plain path (`Mode`, `types::Mode`, no generics)
/// whose last segment is the enum's name. The stage's module declares the enum under that name, so a body writes
/// `Mode::Linear` whatever path the parameter's type was written with.
fn enum_named<'a>(
    enums: &'a [syn::ItemEnum],
    ty: &'a syn::Type,
) -> Option<(&'a syn::Path, &'a syn::ItemEnum)> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    let last = path.path.segments.last().filter(|_| path.qself.is_none())?;
    if !last.arguments.is_none() {
        return None;
    }
    enums
        .iter()
        .find(|definition| definition.ident == last.ident)
        .map(|definition| (&path.path, definition))
}

/// The body of a batch stage, rewritten in place, and the first rule it breaks.
struct Lowering {
    /// Each handle: its name, whether it is the in-place target, the rank the body indexes it with, and the function
    /// that decodes a load when its elements are enums.
    views: Vec<(syn::Ident, bool, Option<usize>, Option<syn::Ident>)>,
    params: Vec<syn::Ident>,
    /// The expression being visited is the place of an assignment: a store, not a load.
    lvalue: bool,
    error: Option<syn::Error>,
}

impl Lowering {
    fn refuse(&mut self, code: Code, tokens: impl ToTokens, message: impl std::fmt::Display) {
        if self.error.is_none() {
            self.error = Some(diagnostics::error(code, tokens, message));
        }
    }

    fn slot(&self, expression: &syn::Expr) -> Option<usize> {
        let mut expression = expression;
        while let syn::Expr::Paren(paren) = expression {
            expression = &paren.expr;
        }
        let syn::Expr::Path(path) = expression else {
            return None;
        };
        let name = path.path.get_ident().filter(|_| path.qself.is_none())?;
        self.views
            .iter()
            .position(|(view, ..)| view.unraw() == name.unraw())
    }

    fn is_param(&self, expression: &syn::Expr) -> bool {
        matches!(expression, syn::Expr::Path(path) if path.qself.is_none()
            && path.path.get_ident().is_some_and(|name| self.params.iter().any(|param| param.unraw() == name.unraw())))
    }
}

impl VisitMut for Lowering {
    fn visit_expr_mut(&mut self, expression: &mut syn::Expr) {
        // Only the node that is the place of an assignment is a store; what is inside it is read.
        let store = core::mem::take(&mut self.lvalue);
        match expression {
            syn::Expr::Assign(assign) => {
                self.visit_expr_mut(&mut assign.right);
                self.lvalue = true;
                self.visit_expr_mut(&mut assign.left);
                return;
            }
            syn::Expr::Binary(binary) if is_compound_assignment(&binary.op) => {
                self.visit_expr_mut(&mut binary.right);
                self.lvalue = true;
                self.visit_expr_mut(&mut binary.left);
                return;
            }
            _ => {}
        }
        syn::visit_mut::visit_expr_mut(self, expression);
        let mut replacement = None;
        match expression {
            syn::Expr::Index(index) => {
                if let Some(slot) = self.slot(&index.expr) {
                    let mut at = &*index.index;
                    while let syn::Expr::Paren(paren) = at {
                        at = &paren.expr;
                    }
                    let (name, target, seen, decoder) = self.views[slot].clone();
                    let rank = match at {
                        syn::Expr::Tuple(tuple) => tuple.elems.len(),
                        _ => 1,
                    };
                    if rank == 0 || (rank == 1 && matches!(at, syn::Expr::Tuple(_))) {
                        self.refuse(
                            Code::TaskIndexRank,
                            &index.index,
                            format!(
                                "`{}` is indexed with one index, `{0}[i]`, or a pair, `{0}[(x, y)]`",
                                name.unraw()
                            ),
                        );
                    } else if rank > 2 {
                        self.refuse(
                            Code::BatchBody,
                            &index.index,
                            format!(
                                "`{}` is indexed with {rank} indices: a view of rank 3 is not in this release; index it \
                                 with one index or a pair",
                                name.unraw()
                            ),
                        );
                    } else if rank == 2 && target {
                        self.refuse(
                            Code::TaskIndexRank,
                            &index.index,
                            format!(
                                "`{}` is the in-place target, which is a flat view of rank 1 in this release: index it \
                                 with its own index",
                                name.unraw()
                            ),
                        );
                    } else if seen.is_some_and(|seen| seen != rank) {
                        self.refuse(
                            Code::TaskIndexRank,
                            &index.index,
                            format!(
                                "`{}` is indexed with one index in one place and with a pair in another: a view has one \
                                 rank",
                                name.unraw()
                            ),
                        );
                    } else {
                        self.views[slot].2 = Some(rank);
                        if let syn::Expr::Tuple(tuple) = at {
                            let (x, y) = (&tuple.elems[0], &tuple.elems[1]);
                            let width = width_name(&name);
                            let flat: syn::Expr = syn::parse_quote!(((#y) * #width + (#x)));
                            *index.index = flat;
                        }
                        // A load of an enum element decodes the word; a store (a capture's, which E1003 refuses)
                        // stays as written.
                        if let (Some(decoder), false) = (&decoder, store) {
                            let load = syn::Expr::Index(index.clone());
                            replacement = Some(syn::parse_quote!(#decoder(#load)));
                        }
                    }
                }
            }
            syn::Expr::MethodCall(call) => {
                if call.method == "late" {
                    self.refuse(
                        Code::BatchBody,
                        &call.method,
                        "`.late()` (a read of a stage's value that is still being written) is not in this release",
                    );
                } else if call.method == "get"
                    && call.args.is_empty()
                    && self.is_param(&call.receiver)
                {
                    replacement = Some((*call.receiver).clone());
                }
            }
            _ => {}
        }
        if let Some(replacement) = replacement {
            *expression = replacement;
        }
    }
}

fn rank_type(rank: usize) -> TokenStream {
    if rank == 2 {
        quote!(::gust::batch::D2)
    } else {
        quote!(::gust::batch::D1)
    }
}

fn scalar_tokens(scalar: Scalar) -> TokenStream {
    match scalar {
        Scalar::U32 => quote!(::core::primitive::u32),
        Scalar::I32 => quote!(::core::primitive::i32),
        Scalar::F32 => quote!(::core::primitive::f32),
        Scalar::Bool => quote!(::core::primitive::bool),
    }
}

fn binding_of<'a>(plan: &'a Plan, name: &syn::Ident) -> &'a Binding {
    plan.bindings
        .iter()
        .find(|binding| binding.name == *name)
        .expect("the plan holds every capture")
}

impl TaskPlan {
    /// The host function of a stage of a batch, in the module of its kernel: it takes the handles and parameters the
    /// stage captures, makes the extent record and its uniform, and seals the kernel call into the batch
    /// (`StageDraft::finish`). The record is made again, from the parameters' values, when a frame sets one of them.
    fn batch_items(&self, plan: &Plan) -> TokenStream {
        let batch = self.batch.as_ref().expect("a stage of a batch");
        let record = format_ident!("{}", "__GustExtent");
        let count = proc_macro2::Literal::usize_unsuffixed(1 + plan.buffers().count());
        let output_element = self.output_element(plan);
        let output_rank = rank_type(self.rank);
        let out = format_ident!("{}", OUT);
        let element_of = |name: &syn::Ident| match &binding_of(plan, name).kind {
            Kind::Slice { element, .. } => element.host.clone(),
            _ => unreachable!("a handle is a slice binding"),
        };
        let scalar_of = |name: &syn::Ident| match binding_of(plan, name).kind {
            Kind::Scalar(scalar) => scalar,
            _ => unreachable!("a parameter is a scalar binding"),
        };
        let param_id = |name: &syn::Ident| format_ident!("__gust_p_{}", name.unraw());

        let view_parameters = batch.views.iter().map(|view| {
            let (name, element, rank) = (&view.name, element_of(&view.name), rank_type(view.rank));
            quote!(#name: &::gust::batch::Stage<#element, #rank>)
        });
        // An enum parameter is generic: the user's own type, which the stage's module cannot name (it declares a copy of
        // the enum under the same name), is whatever the call passes, and the bound is the derive's trait.
        let enum_generic = |name: &syn::Ident| {
            let position = batch
                .enum_params
                .iter()
                .position(|(param, _)| param == name)
                .expect("an enum parameter has a generic");
            format_ident!("__GustP{}", position)
        };
        let enum_generics: Vec<syn::Ident> = batch
            .enum_params
            .iter()
            .map(|(name, _)| enum_generic(name))
            .collect();
        let param_parameters = batch.params.iter().map(|name| {
            let ty = if batch.is_enum(name) {
                let generic = enum_generic(name);
                quote!(#generic)
            } else {
                scalar_tokens(scalar_of(name))
            };
            quote!(#name: &::gust::batch::Param<#ty>)
        });
        let scalar_parameters = batch.scalars.iter().map(|name| {
            let ty = scalar_tokens(scalar_of(name));
            quote!(#name: #ty)
        });
        let bounds = plan.buffers().map(|(_, _, element, _)| {
            let element = &element.host;
            quote!(<__GustD as ::gust::map::MapHost>::Buffer<#element>: 'static,)
        });
        let locals: Vec<syn::Ident> = (0..plan.buffers().count())
            .map(|index| format_ident!("__gust_b{}", index))
            .collect();
        let acquire = plan
            .buffers()
            .zip(&locals)
            .map(|((name, _, element, _), local)| {
                let element = &element.host;
                if *name == out {
                    quote!(let #local = __gust_draft.output::<#element>();)
                } else if batch
                    .views
                    .iter()
                    .any(|view| view.name == *name && view.target)
                {
                    quote!(let #local = __gust_draft.target(#name);)
                } else {
                    quote!(let #local = __gust_draft.view(#name);)
                }
            });
        let param_ids = batch.params.iter().map(|name| {
            let id = param_id(name);
            quote!(let #id = __gust_draft.param(#name);)
        });
        // The number of each 2-D view's buffer, taken when the stage is recorded; the width is read from the frame's
        // extents each time the record is made, so a resize changes it.
        let width_ids = batch
            .views
            .iter()
            .filter(|view| view.rank == 2)
            .map(|view| {
                let (name, id) = (
                    &view.name,
                    format_ident!("__gust_wid_{}", view.name.unraw()),
                );
                quote!(let #id = #name.index();)
            });
        let width_lets = batch
            .views
            .iter()
            .filter(|view| view.rank == 2)
            .map(|view| {
                let (width, id) = (
                    width_name(&view.name),
                    format_ident!("__gust_wid_{}", view.name.unraw()),
                );
                quote!(let #width = __gust_values.dims(#id)[0];)
            });
        let remade = plan
            .buffers()
            .enumerate()
            .map(|(index, (_, _, element, _))| {
                let element = &element.host;
                quote!(__gust_layout.buffer::<__GustD, #element>(__gust_ids[#index]))
            });
        let scalar_fields = plan.scalars().map(|(name, scalar)| {
            let value = if batch.is_enum(name) {
                let (id, generic) = (param_id(name), enum_generic(name));
                quote!(::gust::GpuEnum::to_word(__gust_values.get::<#generic>(#id)))
            } else if batch.params.contains(name) {
                let (id, ty) = (param_id(name), scalar_tokens(scalar));
                quote!(__gust_values.get::<#ty>(#id))
            } else {
                // A host scalar, or the width local of a 2-D view: a local of the function with the field's name.
                quote!(#name)
            };
            if scalar == Scalar::Bool {
                quote!(#name: ::core::primitive::u32::from(#value))
            } else {
                quote!(#name: #value)
            }
        });
        let pads = (0..(4 - (3 + plan.scalars().count()) % 4) % 4).map(|index| {
            let pad = format_ident!("__gust_pad{}", index);
            quote!(#pad: 0u32)
        });
        let resources = locals
            .iter()
            .map(|local| quote!(::gust::Independent(&**#local)));
        let some_locals = locals
            .iter()
            .map(|local| quote!(::core::option::Option::Some(#local),));

        quote! {
            /// Record the stage into the batch being built: seal its kernel call and return the handle of its result.
            #[doc(hidden)]
            #[allow(clippy::too_many_arguments)]
            pub fn __gust_batch<'__gust_b, __GustD #(, #enum_generics)*>(
                mut __gust_draft: ::gust::batch::StageDraft<'__gust_b, __GustD>,
                #(#view_parameters,)*
                #(#param_parameters,)*
                #(#scalar_parameters,)*
            ) -> ::gust::batch::Stage<#output_element, #output_rank>
            where
                __GustD: ::gust::batch::BatchDevice + __GustPfHost + ::gust::Run<run, #count>,
                #(#enum_generics: ::gust::GpuEnum,)*
                #(#bounds)*
            {
                let __gust_dims = __gust_draft.dims();
                let (__gust_x, __gust_y) = (__gust_dims[0], __gust_dims[1]);
                #(#acquire)*
                #(#param_ids)*
                #(#width_ids)*
                if !__gust_draft.ok() {
                    return __gust_draft.abandon();
                }
                let (#(#some_locals)*) = (#(#locals,)*) else {
                    return __gust_draft.abandon();
                };
                let __gust_write = __gust_draft.written();
                let __gust_ids = __gust_draft.bound();
                let __gust_extent = ::gust::Extent::new(__gust_x, __gust_y, 1u32);
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if let ::core::result::Result::Err(__gust_error) = __gust_check_cap::<__GustD>(__gust_extent) {
                        __gust_draft.fail_map(__gust_error);
                        return __gust_draft.abandon();
                    }
                }
                // The extent record is made from the frame's extents and parameters, so a resize changes it.
                let __gust_record = move |__gust_values: &::gust::batch::Values| {
                    let [__gust_x, __gust_y] = __gust_values.dims(__gust_write);
                    #(#width_lets)*
                    #record {
                        __gust_ex: __gust_x,
                        __gust_ey: __gust_y,
                        __gust_ez: 1u32,
                        #(#scalar_fields,)*
                        #(#pads,)*
                    }
                };
                let __gust_device = __gust_draft.device();
                // The uniform is shared by the stage and its update; the CPU twin's is single-threaded.
                #[allow(clippy::arc_with_non_send_sync)]
                let __gust_uniform = match <__GustD as __GustPfHost>::__gust_uniform(
                    &*__gust_device,
                    __gust_record(&__gust_draft.values()),
                ) {
                    ::core::result::Result::Ok(uniform) => ::std::sync::Arc::new(uniform),
                    ::core::result::Result::Err(__gust_error) => {
                        __gust_draft.fail_map(__gust_error);
                        return __gust_draft.abandon();
                    }
                };
                let __gust_rewritten = ::std::sync::Arc::clone(&__gust_uniform);
                let __gust_again = ::std::sync::Arc::clone(&__gust_uniform);
                __gust_draft.finish::<#output_element, #output_rank, run, _, #count>(
                    (__gust_uniform, __gust_extent, #(#locals,)*),
                    |__gust_keep| {
                        let (__gust_uniform, __gust_extent, #(#locals,)*) = __gust_keep;
                        run::call_extent(*__gust_extent, &**__gust_uniform, #(#resources,)*)
                    },
                    move |__gust_device, __gust_frame, __gust_values| {
                        ::gust::batch::BatchDevice::frame_uniform(
                            __gust_device,
                            __gust_frame,
                            &__gust_rewritten,
                            __gust_record(__gust_values),
                        )
                    },
                    // The same stage over the buffers of a resized frame: the new extent and the new buffers.
                    move |__gust_layout| {
                        let [__gust_nx, __gust_ny] = __gust_layout.dims(__gust_write);
                        let __gust_extent = ::gust::Extent::new(__gust_nx, __gust_ny, 1u32);
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            if let ::core::result::Result::Err(__gust_error) =
                                __gust_check_cap::<__GustD>(__gust_extent)
                            {
                                return ::core::result::Result::Err(
                                    ::gust::batch::Error::Device(__gust_error),
                                );
                            }
                        }
                        ::core::result::Result::Ok((
                            ::std::sync::Arc::clone(&__gust_again),
                            __gust_extent,
                            #(#remade,)*
                        ))
                    },
                )
            }
        }
    }

    /// The value of a stage of a batch: its draft (named by the stage's `let`, sized by its shape) handed to the host
    /// function with the handles, parameters and scalars it captures.
    fn batch_value(&self, module: &syn::Ident) -> TokenStream {
        let batch = self.batch.as_ref().expect("a stage of a batch");
        let builder = &batch.builder;
        let name = batch
            .name
            .as_ref()
            .map(|name| quote!(#builder.next_name(#name);));
        let rank = rank_type(self.rank);
        let shape = match (&self.extent, &self.target) {
            (Extent::Shape(shape), _) => quote!(#shape),
            // The target's own shape (a rule that follows its buffer through a resize, not the extents it had at the seal).
            (Extent::Target, Some((name, _))) => quote!(&#name),
            (Extent::Target, None) => unreachable!("an in-place stage has its target"),
        };
        // A view of enum elements is handed to the stage as the handle of its words.
        let views = batch.views.iter().map(|view| {
            let name = &view.name;
            if batch.is_enum_view(name) {
                quote!(&#name.words())
            } else {
                quote!(&#name)
            }
        });
        let params = &batch.params;
        let scalars = &batch.scalars;
        let checks = &batch.enum_checks;
        // The stage records its result as words; the handle is the enum's again for everyone who holds it.
        let retype = batch
            .enum_result
            .as_ref()
            .map(|(ty, _)| quote!(.retype::<#ty>()));
        quote! {
            {
                #(#checks)*
                #name
                let __gust_draft = #builder.draft::<_, #rank>(&(#shape));
                #module::__gust_batch(__gust_draft, #(#views,)* #(&#params,)* #(#scalars,)*) #retype
            }
        }
    }
}

/// What the macro learned about the call besides its expansion, for the value that follows an error: the type of the
/// task's future, and the `crate(..)` clause.
#[derive(Default)]
struct Shape {
    krate: Option<syn::Path>,
    done: Option<TokenStream>,
    /// A stage of a batch: the value after an error is a handle, not a task.
    batch: bool,
}

/// `gpu_task!` for the process: the profile and the environment come from the process, the call line from the span of
/// the invocation.
pub(crate) fn gpu_task(input: TokenStream, profile: Profile) -> TokenStream {
    let call_line = crate::par_for::line_of(Span::call_site())
        .or_else(|| {
            input
                .clone()
                .into_iter()
                .find_map(|token| crate::par_for::line_of(token.span()))
        })
        .unwrap_or(0);
    gpu_task_in(input, &Environment::from_process(), profile, call_line)
}

/// [`gpu_task`] with the environment and the line of the call given, so tests choose them.
pub(crate) fn gpu_task_in(
    input: TokenStream,
    environment: &Environment,
    profile: Profile,
    call_line: u32,
) -> TokenStream {
    let mut shape = Shape::default();
    let tokens = match expand_input(input, environment, profile, call_line, &mut shape) {
        Ok(tokens) => tokens,
        Err(error) => {
            // The errors, then a task of the type the call would have had, so that one mistake is one error.
            let errors = error.to_compile_error();
            if shape.batch {
                return quote!({ #errors ::core::unreachable!() });
            }
            let done = shape.done.take().unwrap_or_else(|| quote!(()));
            quote!({ #errors ::gust::task::Task::<'_, #done>::new(|| ::core::unreachable!()) })
        }
    };
    match &shape.krate {
        Some(path) => crate::forward::forward_crate(tokens, path),
        None => tokens,
    }
}

fn expand_input(
    input: TokenStream,
    environment: &Environment,
    profile: Profile,
    call_line: u32,
    shape: &mut Shape,
) -> syn::Result<TokenStream> {
    let parsed = diagnostics::parse2::<Input>(Code::TaskForm, input)?;
    shape.krate.clone_from(&parsed.options.krate);
    shape.batch = parsed.batch.is_some();
    if let (syn::ReturnType::Type(_, ty), false) = (
        &parsed.closure.output,
        matches!(&parsed.shape, syn::Expr::Reference(reference) if reference.mutability.is_some()),
    ) {
        shape.done = Some(quote!(::std::vec::Vec<#ty>));
    }
    plan_of(parsed, profile)?.expand(environment, call_line)
}
