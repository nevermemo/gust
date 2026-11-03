//! `gust::par_for!` (M7, slice m7.p, plan section 3.1): a parallel loop written at the call site.
//!
//! ```text
//! gust::par_for!(i in 0..n, [a: &[f32], b: &[f32], out: &mut [f32]] {
//!     out[i] = a[i] * b[i];
//! });
//! gust::par_for!((x, y) in (0..w, 0..h), [img: &[f32], out: &mut [f32], w: u32] {
//!     out[y * w + x] = img[y * w + x] * 0.5;
//! });
//! ```
//!
//! The macro builds a kernel in the dialect around the body, hands it to the expansion every `#[gpu]` module goes
//! through (so the Slang, the descriptor, the typed calls, the CPU `Run` impl and the artifact handling are those of
//! any kernel), and adds the host function that launches it:
//!
//! ```text
//! { mod __gust_pf_<hash16> { struct __GustExtent {..}  #[kernel(..)] fn run(..) {..}  fn launch(..) {..} .. }
//!   __gust_pf_<hash16>::launch_default(n, a, b, out) }
//! ```
//!
//! The kernel is `run(gid, extent: Uniform<__GustExtent>, buffers..)`. The extent record holds the three extents, the
//! scalar bindings (a `bool` as a `u32` word) and explicit padding up to 16 bytes (GUST never inserts padding). The
//! body runs inside the guard `if x < extent.x && y < extent.y && z < extent.z` over all three axes, whatever the arity
//! of the loop (an unused axis has extent 1, so a taller workgroup runs nothing on it).
//!
//! Identity (R1.1 S5, M5). `hash16` is the content key of the *lowered Slang*, not of the tokens: the macro lowers
//! once as the placeholder module `__gust_pf` (entry `gpu___gust_pf_run`), takes [`content_key`] of that clean text,
//! and expands for real as `__gust_pf_<hash16>`. Two spellings of one body (`f32` and `float`, other whitespace)
//! lower to one text, so they get one module name, one entry point and one artifact key, and the block around the
//! module scopes it, so two calls never clash. The export record is written as schema 2 (no `source`) with one
//! sidecar per call site ([`Identity::Site`]).
//!
//! The body is validated as a kernel of a module of its own whose generated names do not use the reserved `__gust_`
//! prefix (the *view*), so a dialect error is reported at the body's tokens with its code (D33). The module that is
//! expanded uses the reserved names, which the validator refuses in user code.

use proc_macro2::{Literal, Span, TokenStream};
use quote::{format_ident, quote};
use syn::{
    Token,
    ext::IdentExt as _,
    parse::{Parse, ParseStream},
};

use crate::{
    diagnostics::{self, Code},
    expand::{Environment, Identity, KernelOptions, Profile, content_key, expand_validated},
    math_types,
};

/// The names the generated kernel uses for its own parts, in the module that is expanded (reserved `__gust_` names) and
/// in the view that is validated (names the validator accepts in user code).
struct Names {
    /// The dispatch thread id.
    gid: &'static str,
    /// The extent record parameter.
    ext: &'static str,
    /// The extent record struct.
    record: &'static str,
    /// The record fields that hold the three extents.
    axes: [&'static str; 3],
    /// The prefix of the padding fields.
    pad: &'static str,
}

const FINAL: Names = Names {
    gid: "__gust_gid",
    ext: "__gust_ext",
    record: "__GustExtent",
    axes: ["__gust_ex", "__gust_ey", "__gust_ez"],
    pad: "__gust_pad",
};

const VIEW: Names = Names {
    gid: "par_for_gid",
    ext: "par_for_ext",
    record: "ParForExtent",
    axes: ["par_for_ex", "par_for_ey", "par_for_ez"],
    pad: "par_for_pad",
};

/// The field names of a dispatch id.
const FIELDS: [&str; 3] = ["x", "y", "z"];

/// The placeholder module of the first lowering and its entry point (R1.1 S5).
const PLACEHOLDER: &str = "__gust_pf";

/// What the keyword options in front of the loop say.
#[derive(Default)]
struct Options {
    workgroup: Option<[u32; 3]>,
    on: Option<syn::Expr>,
    /// The hidden `descriptor;` form, which expands to the kernel's descriptor and launches nothing. It is for the
    /// follow-up that runs the naga gate over `par_for!` kernels (the fixtures are skipped there until then); it is
    /// not public API and has no entry in the documentation of the macro.
    descriptor: bool,
}

struct RawBinding {
    name: syn::Ident,
    ty: syn::Type,
}

/// The macro input as written: shape only, no meaning.
struct Raw {
    options: Options,
    pattern: syn::Pat,
    ranges: syn::Expr,
    bindings: Vec<RawBinding>,
    body: syn::Block,
}

impl Parse for RawBinding {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![:]>()?;
        Ok(Self {
            name,
            ty: input.parse()?,
        })
    }
}

impl Parse for Raw {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut options = Options::default();
        // The keyword options come first, each ended by `;`. A loop pattern is never an identifier followed by `(`
        // or `;`, so these are unambiguous.
        while let Some((keyword, _)) = input.cursor().ident() {
            let text = keyword.to_string();
            if text == "workgroup_size" && input.peek2(syn::token::Paren) {
                input.parse::<syn::Ident>()?;
                let content;
                syn::parenthesized!(content in input);
                let sizes = content.parse_terminated(syn::LitInt::parse, Token![,])?;
                input.parse::<Token![;]>()?;
                if options.workgroup.is_some() {
                    return Err(diagnostics::error(
                        Code::DuplicateWorkgroupSize,
                        &keyword,
                        "`workgroup_size` given twice",
                    ));
                }
                if sizes.len() != 3 {
                    return Err(diagnostics::error(
                        Code::WorkgroupDimensionCount,
                        &sizes,
                        "`workgroup_size` takes exactly three dimensions, for example `workgroup_size(64, 4, 1)`",
                    ));
                }
                let mut size = [0_u32; 3];
                for (slot, literal) in size.iter_mut().zip(&sizes) {
                    *slot = literal
                        .base10_parse::<u32>()
                        .ok()
                        .filter(|&value| value > 0)
                        .ok_or_else(|| {
                            diagnostics::error(
                                Code::WorkgroupDimensionZero,
                                literal,
                                "a workgroup dimension is a positive integer",
                            )
                        })?;
                }
                options.workgroup = Some(size);
            } else if text == "on" && input.peek2(syn::token::Paren) {
                input.parse::<syn::Ident>()?;
                let content;
                syn::parenthesized!(content in input);
                let device: syn::Expr = content.parse()?;
                input.parse::<Token![;]>()?;
                if options.on.is_some() {
                    return Err(diagnostics::error(
                        Code::ParForForm,
                        &keyword,
                        "`on(..)` given twice",
                    ));
                }
                options.on = Some(device);
            } else if text == "descriptor" && input.peek2(Token![;]) {
                input.parse::<syn::Ident>()?;
                input.parse::<Token![;]>()?;
                options.descriptor = true;
            } else {
                break;
            }
        }
        let pattern = syn::Pat::parse_single(input)?;
        input.parse::<Token![in]>()?;
        let ranges: syn::Expr = input.parse()?;
        input.parse::<Token![,]>()?;
        let content;
        syn::bracketed!(content in input);
        let bindings = content
            .parse_terminated(RawBinding::parse, Token![,])?
            .into_iter()
            .collect();
        let body: syn::Block = input.parse()?;
        if !input.is_empty() {
            return Err(diagnostics::error(
                Code::ParForForm,
                input.cursor().token_stream(),
                "`par_for!` ends with the body block",
            ));
        }
        Ok(Self {
            options,
            pattern,
            ranges,
            bindings,
            body,
        })
    }
}

/// The element of a buffer binding: its type as the kernel spells it and as host code spells it.
#[derive(Clone)]
struct Element {
    kernel: TokenStream,
    host: TokenStream,
}

/// A scalar binding: a field of the extent record.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scalar {
    U32,
    I32,
    F32,
    Bool,
}

enum Kind {
    /// `&[T]` or `&mut [T]`: uploaded for the launch, read back afterwards when mutable.
    Slice {
        mutable: bool,
        element: Element,
    },
    /// `&GpuVec<T>` or `&mut GpuVec<T>`: bound where it is.
    Resident {
        mutable: bool,
        element: Element,
    },
    Scalar(Scalar),
}

struct Binding {
    name: syn::Ident,
    kind: Kind,
}

/// A `par_for!` call understood.
struct Plan {
    options: Options,
    /// The loop variable of each axis (`None` for `_`); the length is the number of dimensions.
    axes: Vec<Option<syn::Ident>>,
    extents: Vec<syn::Expr>,
    workgroup: [u32; 3],
    bindings: Vec<Binding>,
    body: syn::Block,
    profile: Profile,
}

fn form(tokens: impl quote::ToTokens, message: impl std::fmt::Display) -> syn::Error {
    diagnostics::error(Code::ParForForm, tokens, message)
}

fn binding_error(tokens: impl quote::ToTokens, message: impl std::fmt::Display) -> syn::Error {
    diagnostics::error(Code::ParForBinding, tokens, message)
}

/// The names of a loop pattern: `i`, `(x, y)` or `(x, y, z)`, each name an identifier or `_`.
fn pattern_axes(pattern: &syn::Pat) -> syn::Result<Vec<Option<syn::Ident>>> {
    let single = |pattern: &syn::Pat| match pattern {
        syn::Pat::Ident(ident)
            if ident.by_ref.is_none() && ident.mutability.is_none() && ident.subpat.is_none() =>
        {
            Ok(Some(ident.ident.clone()))
        }
        syn::Pat::Wild(_) => Ok(None),
        other => Err(form(
            other,
            "a loop variable is a plain identifier or `_`; the pattern is `i`, `(x, y)` or `(x, y, z)`",
        )),
    };
    let axes = match pattern {
        syn::Pat::Tuple(tuple) if (2..=3).contains(&tuple.elems.len()) => tuple
            .elems
            .iter()
            .map(single)
            .collect::<syn::Result<Vec<_>>>()?,
        syn::Pat::Tuple(tuple) => {
            return Err(form(
                tuple,
                "a `par_for!` loop has one, two or three dimensions: `i`, `(x, y)` or `(x, y, z)`",
            ));
        }
        other => vec![single(other)?],
    };
    let mut seen = Vec::new();
    for name in axes.iter().flatten() {
        let text = name.unraw().to_string();
        if seen.contains(&text) {
            return Err(form(
                name,
                format!("the loop variable `{text}` is named twice"),
            ));
        }
        seen.push(text);
    }
    Ok(axes)
}

/// The end of a range `0..n`: the start must be the literal 0 and the range half-open.
fn range_end(range: &syn::Expr) -> syn::Result<syn::Expr> {
    let message = "a `par_for!` range is `0..n`: it starts at the literal 0 and ends before `n`";
    let syn::Expr::Range(range) = range else {
        return Err(form(range, message));
    };
    let starts_at_zero = matches!(
        range.start.as_deref(),
        Some(syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(literal), .. })) if literal.base10_digits() == "0"
    );
    match (&range.end, range.limits) {
        (Some(end), syn::RangeLimits::HalfOpen(_)) if starts_at_zero => Ok((**end).clone()),
        _ => Err(form(range, message)),
    }
}

/// The extent expressions, one per dimension.
fn range_ends(ranges: &syn::Expr, dimensions: usize) -> syn::Result<Vec<syn::Expr>> {
    let mut ranges = ranges;
    while let syn::Expr::Paren(paren) = ranges {
        ranges = &paren.expr;
    }
    if dimensions == 1 {
        return Ok(vec![range_end(ranges)?]);
    }
    let syn::Expr::Tuple(tuple) = ranges else {
        return Err(form(
            ranges,
            format!(
                "this loop has {dimensions} variables, so it ranges over a tuple of {dimensions} ranges"
            ),
        ));
    };
    if tuple.elems.len() != dimensions {
        return Err(form(
            tuple,
            format!(
                "this loop has {dimensions} variables but {} ranges; give one range per variable",
                tuple.elems.len()
            ),
        ));
    }
    tuple.elems.iter().map(range_end).collect()
}

/// The element of a buffer binding: `f32`, `i32`, `u32` (or `float`, `int`, `uint`) and the vector and matrix types
/// that can be buffer elements. `bool` is not one (a buffer holds `u32` words).
fn element(ty: &syn::Type, what: &str) -> syn::Result<Element> {
    let reject = || {
        binding_error(
            ty,
            format!(
                "{what} has a type a buffer cannot hold: use `f32`, `i32`, `u32`, `Vec2`, `Vec4`, `IVec2`, `IVec4`, \
                 `UVec2`, `UVec4`, `Mat2` or `Mat4` (a `bool` buffer is a `u32` buffer)"
            ),
        )
    };
    let syn::Type::Path(path) = ty else {
        return Err(reject());
    };
    let Some(ident) = path.path.get_ident().filter(|_| path.qself.is_none()) else {
        return Err(reject());
    };
    let name = ident.unraw().to_string();
    let scalar = |kernel: TokenStream, host: TokenStream| Element { kernel, host };
    Ok(match name.as_str() {
        "f32" | "float" => scalar(quote!(f32), quote!(::core::primitive::f32)),
        "i32" | "int" => scalar(quote!(i32), quote!(::core::primitive::i32)),
        "u32" | "uint" => scalar(quote!(u32), quote!(::core::primitive::u32)),
        other => {
            let Some(math) = math_types::math_type(other) else {
                return Err(reject());
            };
            if !math.buffer {
                return Err(diagnostics::error(
                    Code::ThreeWideInBuffer,
                    ty,
                    format!(
                        "{what} is `{}`, which a buffer cannot hold: 3-wide types have stride 16 in a buffer and no \
                         Rust layout that matches in every position; use the 4-wide type or scalars",
                        math.name
                    ),
                ));
            }
            let path = format_ident!("{}", math.name);
            scalar(quote!(#path), quote!(::gust::math::#path))
        }
    })
}

/// The one type argument of a `GpuVec<T>` path.
fn gpu_vec_element(path: &syn::TypePath) -> Option<&syn::Type> {
    if path.qself.is_some() {
        return None;
    }
    let segment = path.path.segments.last()?;
    if segment.ident != "GpuVec" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    match arguments.args.iter().collect::<Vec<_>>().as_slice() {
        [syn::GenericArgument::Type(ty)] => Some(ty),
        _ => None,
    }
}

fn classify(name: &syn::Ident, ty: &syn::Type, profile: Profile) -> syn::Result<Kind> {
    let what = format!("the binding `{}`", name.unraw());
    let unsupported = || {
        binding_error(
            ty,
            format!(
                "{what} has a type a `par_for!` binding cannot have: use `&[T]`, `&mut [T]`, `&GpuVec<T>`, \
                 `&mut GpuVec<T>`, or a scalar `u32`, `i32`, `f32` or `bool`"
            ),
        )
    };
    match ty {
        syn::Type::Reference(reference) => {
            let mutable = reference.mutability.is_some();
            match reference.elem.as_ref() {
                syn::Type::Slice(slice) => Ok(Kind::Slice {
                    mutable,
                    element: element(&slice.elem, &what)?,
                }),
                syn::Type::Path(path) => {
                    let Some(inner) = gpu_vec_element(path) else {
                        return Err(unsupported());
                    };
                    if profile == Profile::Cpu {
                        return Err(binding_error(
                            ty,
                            format!(
                                "{what} is a `GpuVec`, and under `GUST_PROFILE=cpu` there is no GPU API; bind a \
                                 slice instead"
                            ),
                        ));
                    }
                    Ok(Kind::Resident {
                        mutable,
                        element: element(inner, &what)?,
                    })
                }
                _ => Err(unsupported()),
            }
        }
        syn::Type::Path(path) => {
            let Some(ident) = path.path.get_ident().filter(|_| path.qself.is_none()) else {
                return Err(unsupported());
            };
            Ok(Kind::Scalar(match ident.unraw().to_string().as_str() {
                "u32" | "uint" => Scalar::U32,
                "i32" | "int" => Scalar::I32,
                "f32" | "float" => Scalar::F32,
                "bool" => Scalar::Bool,
                _ => return Err(unsupported()),
            }))
        }
        _ => Err(unsupported()),
    }
}

/// The names the expansion defines in the block of the call that a binding's name would shadow: the kernel `run` (a
/// unit struct as well as a function, so a binding of that name is read as a pattern for it), the constant
/// `DESCRIPTOR` (likewise a pattern), and the two launch functions, one of which `launch_default` calls. The other
/// names the block defines start with `__gust` or `__Gust`, which are reserved for every name in a call.
const BLOCK_NAMES: [&str; 4] = ["run", "launch", "launch_default", "DESCRIPTOR"];

impl Plan {
    fn new(raw: Raw, profile: Profile) -> syn::Result<Self> {
        let axes = pattern_axes(&raw.pattern)?;
        let extents = range_ends(&raw.ranges, axes.len())?;
        let mut bindings: Vec<Binding> = Vec::new();
        for binding in raw.bindings {
            let text = binding.name.unraw().to_string();
            let taken = bindings
                .iter()
                .any(|other| other.name.unraw() == binding.name.unraw())
                || axes
                    .iter()
                    .flatten()
                    .any(|axis| axis.unraw() == binding.name.unraw());
            if taken {
                return Err(binding_error(
                    &binding.name,
                    format!(
                        "the name `{text}` is used twice among the loop variables and the bindings"
                    ),
                ));
            }
            if BLOCK_NAMES.contains(&text.as_str()) {
                return Err(binding_error(
                    &binding.name,
                    format!(
                        "the name `{text}` is one the expansion defines in the block of the call ({}); rename the \
                         variable (a binding reads the variable of its own name)",
                        BLOCK_NAMES.join(", ")
                    ),
                ));
            }
            let kind = classify(&binding.name, &binding.ty, profile)?;
            bindings.push(Binding {
                name: binding.name,
                kind,
            });
        }
        let workgroup = raw.options.workgroup.unwrap_or(match axes.len() {
            1 => [256, 1, 1],
            2 => [16, 16, 1],
            _ => [8, 8, 4],
        });
        Ok(Self {
            options: raw.options,
            axes,
            extents,
            workgroup,
            bindings,
            body: raw.body,
            profile,
        })
    }

    fn scalars(&self) -> impl Iterator<Item = (&syn::Ident, Scalar)> {
        self.bindings
            .iter()
            .filter_map(|binding| match binding.kind {
                Kind::Scalar(scalar) => Some((&binding.name, scalar)),
                _ => None,
            })
    }

    fn buffers(&self) -> impl Iterator<Item = (&syn::Ident, bool, &Element, bool)> {
        self.bindings
            .iter()
            .filter_map(|binding| match &binding.kind {
                Kind::Slice { mutable, element } => Some((&binding.name, *mutable, element, false)),
                Kind::Resident { mutable, element } => {
                    Some((&binding.name, *mutable, element, true))
                }
                Kind::Scalar(_) => None,
            })
    }

    /// The kernel module named `ident`, with the names of `names`.
    fn module(&self, ident: &syn::Ident, names: &Names) -> syn::ItemMod {
        let gid = format_ident!("{}", names.gid);
        let ext = format_ident!("{}", names.ext);
        let record = format_ident!("{}", names.record);
        let axis_fields: Vec<syn::Ident> = names
            .axes
            .iter()
            .map(|name| format_ident!("{}", name))
            .collect();
        let scalar_fields = self.scalars().map(|(name, scalar)| {
            let ty = match scalar {
                Scalar::U32 | Scalar::Bool => quote!(u32),
                Scalar::I32 => quote!(i32),
                Scalar::F32 => quote!(f32),
            };
            quote!(pub #name: #ty)
        });
        // Three extents and the scalars are 4-byte words; the record fills a multiple of 16 bytes.
        let words = 3 + self.scalars().count();
        let pads = (0..(4 - words % 4) % 4).map(|index| format_ident!("{}{}", names.pad, index));
        let parameters = self.buffers().map(|(name, mutable, element, _)| {
            let element = &element.kernel;
            if mutable {
                quote!(mut #name: RWStructuredBuffer<#element>)
            } else {
                quote!(#name: StructuredBuffer<#element>)
            }
        });
        let mut guard = TokenStream::new();
        // All three axes, whatever the loop's arity: a `workgroup_size` taller than the loop runs invocations on
        // the unused axes too, and their extent is 1.
        for axis in 0..3 {
            let (field, extent) = (format_ident!("{}", FIELDS[axis]), &axis_fields[axis]);
            if axis > 0 {
                guard.extend(quote!(&&));
            }
            guard.extend(quote!(#gid.#field < #ext.#extent));
        }
        let axis_lets = self.axes.iter().enumerate().filter_map(|(axis, name)| {
            let (name, field) = (name.as_ref()?, format_ident!("{}", FIELDS[axis]));
            Some(quote!(let #name = #gid.#field;))
        });
        let scalar_lets = self.scalars().map(|(name, scalar)| {
            if scalar == Scalar::Bool {
                quote!(let #name = #ext.#name != 0u32;)
            } else {
                quote!(let #name = #ext.#name;)
            }
        });
        let [x, y, z] = self.workgroup.map(Literal::u32_unsuffixed);
        let statements = &self.body.stmts;
        let [ex, ey, ez] = &axis_fields[..] else {
            unreachable!("three axes")
        };
        syn::parse_quote! {
            mod #ident {
                pub struct #record {
                    pub #ex: u32,
                    pub #ey: u32,
                    pub #ez: u32,
                    #(#scalar_fields,)*
                    #(pub #pads: u32,)*
                }

                #[kernel(workgroup_size(#x, #y, #z))]
                pub fn run(#gid: SV_DispatchThreadID, #ext: Uniform<#record>, #(#parameters,)*) {
                    if #guard {
                        #(#axis_lets)*
                        #(#scalar_lets)*
                        #(#statements)*
                    }
                }
            }
        }
    }

    /// The hash of the lowered Slang (S5): the module lowered once under the placeholder name.
    fn lowered_hash(&self) -> syn::Result<String> {
        let module = self.module(&format_ident!("{}", PLACEHOLDER), &FINAL);
        let (_, items) = module.content.as_ref().expect("inline module");
        let kernel = items
            .iter()
            .find_map(|item| match item {
                syn::Item::Fn(function) if function.sig.ident == "run" => Some(function),
                _ => None,
            })
            .expect("the kernel `run`");
        let options = KernelOptions {
            workgroup_size: self.workgroup,
        };
        let emitted = crate::slang::emit_kernel(&module, kernel, options)?;
        Ok(content_key(
            &format!("gpu_{PLACEHOLDER}_run"),
            &emitted.source,
        ))
    }

    /// The host side that goes inside the module: the extent record's uniform, `launch` and `launch_default`.
    fn host_items(&self) -> TokenStream {
        let cpu = self.profile == Profile::Cpu;
        let record = format_ident!("{}", FINAL.record);
        let dimensions = self.axes.len();
        let count = Literal::usize_unsuffixed(1 + self.buffers().count());

        // Generics: the device, one type per extent expression.
        let extent_types: Vec<syn::Ident> = (0..dimensions)
            .map(|axis| format_ident!("__GustE{}", axis))
            .collect();
        let extent_names: Vec<syn::Ident> = (0..dimensions)
            .map(|axis| format_ident!("__gust_e{}", axis))
            .collect();
        let extent_bounds: Vec<TokenStream> = extent_types
            .iter()
            .map(|ty| {
                quote! {
                    #ty: ::core::convert::TryInto<u32>,
                    <#ty as ::core::convert::TryInto<u32>>::Error:
                        ::core::error::Error + ::core::marker::Send + ::core::marker::Sync + 'static,
                }
            })
            .collect();
        let extent_parameters: Vec<TokenStream> = extent_names
            .iter()
            .zip(&extent_types)
            .map(|(name, ty)| quote!(#name: #ty))
            .collect();

        // The user's bindings as host parameters, with absolute types.
        let host_parameters: Vec<TokenStream> = self
            .bindings
            .iter()
            .map(|binding| {
                let name = &binding.name;
                match &binding.kind {
                    Kind::Slice { mutable, element } => {
                        let element = &element.host;
                        if *mutable {
                            quote!(#name: &mut [#element])
                        } else {
                            quote!(#name: &[#element])
                        }
                    }
                    Kind::Resident { mutable, element } => {
                        let element = &element.host;
                        if *mutable {
                            quote!(#name: &mut ::gust_wgpu::GpuVec<#element>)
                        } else {
                            quote!(#name: &::gust_wgpu::GpuVec<#element>)
                        }
                    }
                    Kind::Scalar(Scalar::U32) => quote!(#name: ::core::primitive::u32),
                    Kind::Scalar(Scalar::I32) => quote!(#name: ::core::primitive::i32),
                    Kind::Scalar(Scalar::F32) => quote!(#name: ::core::primitive::f32),
                    Kind::Scalar(Scalar::Bool) => quote!(#name: ::core::primitive::bool),
                }
            })
            .collect();
        let resident_bounds =
            self.buffers()
                .filter(|(_, _, _, resident)| *resident)
                .map(|(_, _, element, _)| {
                    let element = &element.host;
                    quote! {
                        for<'__gust_a> &'__gust_a ::gust_wgpu::GpuVec<#element>:
                            ::gust::map::MapArg<#element, __GustD>,
                    }
                });

        // The body of `launch`.
        let conversions = extent_names
            .iter()
            .zip(0..)
            .map(|(name, axis)| {
                let converted = format_ident!("__gust_{}", FIELDS[axis]);
                quote! {
                    let #converted = ::core::convert::TryInto::<u32>::try_into(#name)
                        .map_err(::gust::map::Error::device)?;
                }
            })
            .collect::<Vec<_>>();
        let axis_value = |axis: usize| {
            if axis < dimensions {
                let converted = format_ident!("__gust_{}", FIELDS[axis]);
                quote!(#converted)
            } else {
                quote!(1u32)
            }
        };
        let (value_x, value_y, value_z) = (axis_value(0), axis_value(1), axis_value(2));
        let converted_names: Vec<syn::Ident> = (0..dimensions)
            .map(|axis| format_ident!("__gust_{}", FIELDS[axis]))
            .collect();
        let scalar_values = self.scalars().map(|(name, scalar)| {
            if scalar == Scalar::Bool {
                quote!(#name: ::core::primitive::u32::from(#name))
            } else {
                quote!(#name)
            }
        });
        let pads = (0..(4 - (3 + self.scalars().count()) % 4) % 4).map(|index| {
            let pad = format_ident!("{}{}", FINAL.pad, index);
            quote!(#pad: 0u32)
        });
        let bound: Vec<syn::Ident> = (0..self.buffers().count())
            .map(|index| format_ident!("__gust_b{}", index))
            .collect();
        let binds = self.buffers().zip(&bound).map(|((name, _, element, _), local)| {
            let element = &element.host;
            quote! {
                let #local = ::gust::map::MapArg::<#element, __GustD>::bind(&*#name, __gust_device)?;
            }
        });
        let resources = self.buffers().zip(&bound).map(|((_, _, element, resident), local)| {
            let element = &element.host;
            let argument = if resident {
                quote!(&::gust_wgpu::GpuVec<#element>)
            } else {
                quote!(&[#element])
            };
            quote! {
                ::gust::Independent(::gust::map::resource::<#element, __GustD, #argument>(&#local))
            }
        });
        let readbacks = self.buffers().zip(&bound).filter_map(|((name, mutable, element, resident), local)| {
            if !mutable || resident {
                return None;
            }
            let element = &element.host;
            Some(quote! {
                let __gust_host = <__GustD as ::gust::map::MapHost>::readback::<#element>(__gust_device, &#local)
                    .map_err(<__GustD as ::gust::map::MapHost>::map_error)?;
                #name.copy_from_slice(&__gust_host);
            })
        });
        let doc_launch = "Run the loop on `__gust_device` and wait. The extents are converted with `TryInto<u32>` \
             (`Err` when one does not fit); an axis of 0 runs nothing and returns `Ok`. A `&mut [T]` binding is \
             uploaded and read back into the slice; the body guards its own indexing, because no buffer length is \
             checked against the extents.";
        let doc_default = "[`launch`] on the default device.";
        let (open_device, device_argument) = if cpu {
            (
                quote!(let __gust_device = ::gust::cpu::CpuDevice::new();),
                quote!(&__gust_device),
            )
        } else {
            (
                quote!(let __gust_device = ::gust_wgpu::default_device().map_err(::gust::map::Error::device)?;),
                quote!(&*__gust_device),
            )
        };
        let concrete_device = if cpu {
            quote!(::gust::cpu::CpuDevice)
        } else {
            quote!(::gust_wgpu::HeadlessDevice)
        };
        let resident_defaults =
            self.buffers()
                .filter(|(_, _, _, resident)| *resident)
                .map(|(_, _, element, _)| {
                    let element = &element.host;
                    quote! {
                        for<'__gust_a> &'__gust_a ::gust_wgpu::GpuVec<#element>:
                            ::gust::map::MapArg<#element, #concrete_device>,
                    }
                });
        let names: Vec<&syn::Ident> = self.bindings.iter().map(|binding| &binding.name).collect();

        let mut items = quote! {
            /// The descriptor of the generated kernel.
            pub const DESCRIPTOR: ::gust::KernelDescriptor = <run as ::gust::Kernel>::DESCRIPTOR;

            /// The device side of the extent record: its uniform buffer.
            #[doc(hidden)]
            pub trait __GustPfHost: ::gust::map::MapHost {
                type Uniform: ::gust::Resource<
                    Element = #record,
                    Device = Self,
                    Kind = ::gust::kind::Uniform,
                >;
                fn __gust_uniform(&self, value: #record) -> ::core::result::Result<Self::Uniform, ::gust::map::Error>;
                /// The launch needs more invocations than the cap of `global_invocation_id` (the device's own error).
                fn __gust_over_cap(requested: u64) -> ::gust::map::Error;
            }

            /// Refuse a launch past the invocation cap before anything is created or uploaded: every axis rounded up
            /// to whole workgroups, multiplied, at most 2^32 (`Extent::MAX_INVOCATIONS`, the rule of both backends).
            /// The device repeats the check when it dispatches; this one only comes first.
            #[cfg(not(target_arch = "wasm32"))]
            fn __gust_check_cap<__GustD: __GustPfHost>(
                extent: ::gust::Extent,
            ) -> ::core::result::Result<(), ::gust::map::Error> {
                match extent.rounded_total(DESCRIPTOR.workgroup_size) {
                    ::core::option::Option::Some(total) if total > u128::from(::gust::Extent::MAX_INVOCATIONS) => {
                        ::core::result::Result::Err(<__GustD as __GustPfHost>::__gust_over_cap(
                            u64::try_from(total).unwrap_or(u64::MAX),
                        ))
                    }
                    _ => ::core::result::Result::Ok(()),
                }
            }

            impl __GustPfHost for ::gust::cpu::CpuDevice {
                type Uniform = ::gust::cpu::CpuUniform<#record>;
                fn __gust_uniform(&self, value: #record) -> ::core::result::Result<Self::Uniform, ::gust::map::Error> {
                    self.uniform(value).map_err(<Self as ::gust::map::MapHost>::map_error)
                }
                fn __gust_over_cap(requested: u64) -> ::gust::map::Error {
                    <Self as ::gust::map::MapHost>::map_error(::gust::cpu::Error::DeviceLimit {
                        kernel: DESCRIPTOR.qualified_name().to_string(),
                        limit: "global_invocation_id",
                        requested,
                        maximum: ::gust::Extent::MAX_INVOCATIONS,
                    })
                }
            }
        };
        if !cpu {
            items.extend(quote! {
                impl __GustPfHost for ::gust_wgpu::HeadlessDevice {
                    type Uniform = ::gust_wgpu::GpuUniform<#record>;
                    fn __gust_uniform(
                        &self,
                        value: #record,
                    ) -> ::core::result::Result<Self::Uniform, ::gust::map::Error> {
                        self.create_uniform("par_for extent", &value)
                            .map_err(<Self as ::gust::map::MapHost>::map_error)
                    }
                    fn __gust_over_cap(requested: u64) -> ::gust::map::Error {
                        <Self as ::gust::map::MapHost>::map_error(::gust_wgpu::Error::DeviceLimit {
                            kernel: DESCRIPTOR.qualified_name().to_string(),
                            limit: "global_invocation_id",
                            requested,
                            maximum: ::gust::Extent::MAX_INVOCATIONS,
                        })
                    }
                }
            });
        }
        items.extend(quote! {
            #[doc = #doc_launch]
            #[cfg(not(target_arch = "wasm32"))]
            #[allow(clippy::too_many_arguments)]
            pub fn launch<__GustD, #(#extent_types,)*>(
                __gust_device: &__GustD,
                #(#extent_parameters,)*
                #(#host_parameters,)*
            ) -> ::core::result::Result<(), ::gust::map::Error>
            where
                __GustD: ::gust::map::MapHost + __GustPfHost + ::gust::Run<run, #count>,
                #(#extent_bounds)*
                #(#resident_bounds)*
            {
                #(#conversions)*
                if #value_x == 0u32 || #value_y == 0u32 || #value_z == 0u32 {
                    return ::core::result::Result::Ok(());
                }
                let __gust_extent = ::gust::Extent::new(#value_x, #value_y, #value_z);
                __gust_check_cap::<__GustD>(__gust_extent)?;
                let __gust_record = #record {
                    __gust_ex: #value_x,
                    __gust_ey: #value_y,
                    __gust_ez: #value_z,
                    #(#scalar_values,)*
                    #(#pads,)*
                };
                let __gust_uniform = <__GustD as __GustPfHost>::__gust_uniform(__gust_device, __gust_record)?;
                #(#binds)*
                run::dispatch_extent(__gust_device, __gust_extent, &__gust_uniform, #(#resources,)*)
                    .map_err(<__GustD as ::gust::map::MapHost>::map_error)?;
                #(#readbacks)*
                ::core::result::Result::Ok(())
            }

            #[doc = #doc_default]
            #[cfg(not(target_arch = "wasm32"))]
            #[allow(clippy::too_many_arguments)]
            pub fn launch_default<#(#extent_types),*>(
                #(#extent_parameters,)*
                #(#host_parameters,)*
            ) -> ::core::result::Result<(), ::gust::map::Error>
            where
                #(#extent_bounds)*
                #(#resident_defaults)*
            {
                // The extents are converted and the cap is checked before the device is opened; `launch` then takes
                // the converted ones (a `u32` converts to itself).
                #(#conversions)*
                if #value_x == 0u32 || #value_y == 0u32 || #value_z == 0u32 {
                    return ::core::result::Result::Ok(());
                }
                __gust_check_cap::<#concrete_device>(::gust::Extent::new(#value_x, #value_y, #value_z))?;
                #open_device
                launch(#device_argument, #(#converted_names,)* #(#names),*)
            }
        });
        if !self.options.descriptor {
            items.extend(quote! {
                #[cfg(target_arch = "wasm32")]
                ::core::compile_error!(
                    "`par_for!` blocks until the device is done, which nothing may do on wasm32 (D69)"
                );
            });
        }
        items
    }

    /// The expansion: the kernel module and either the launch call or, for the hidden `descriptor;` form, the
    /// kernel's descriptor.
    fn expand(&self, environment: &Environment, call_line: u32) -> syn::Result<TokenStream> {
        // The body as a kernel of a module with names the validator accepts: a dialect error is reported at the
        // body's own tokens with its code.
        crate::validate::validate_module(&self.module(&format_ident!("par_for_view"), &VIEW))?;
        let hash = self.lowered_hash()?;
        let name = format_ident!("{}_{}", PLACEHOLDER, hash);
        let module = self.module(&name, &FINAL);
        let host = self.host_items();
        let expanded = expand_validated(
            module,
            environment,
            self.profile,
            &host,
            Identity::Site { call_line },
        )?;
        let value = if self.options.descriptor {
            quote!(<#name::run as ::gust::Kernel>::DESCRIPTOR)
        } else {
            let extents = &self.extents;
            let arguments = self.bindings.iter().map(|binding| {
                let name = &binding.name;
                match &binding.kind {
                    Kind::Slice { mutable: false, .. } => quote!(&#name[..]),
                    Kind::Slice { mutable: true, .. } => quote!(&mut #name[..]),
                    Kind::Resident { mutable: false, .. } => quote!(#name),
                    Kind::Resident { mutable: true, .. } => quote!(&mut *#name),
                    Kind::Scalar(_) => quote!(#name),
                }
            });
            match &self.options.on {
                Some(device) => quote!(#name::launch(#device, #(#extents,)* #(#arguments),*)),
                None => quote!(#name::launch_default(#(#extents,)* #(#arguments),*)),
            }
        };
        Ok(quote! {
            {
                #expanded
                #value
            }
        })
    }
}

/// `par_for!` for the process: the profile and the environment come from the process, the call line from the span of
/// the invocation.
pub(crate) fn par_for(input: TokenStream, profile: Profile) -> TokenStream {
    let call_line = line_of(Span::call_site())
        .or_else(|| {
            input
                .clone()
                .into_iter()
                .find_map(|token| line_of(token.span()))
        })
        .unwrap_or(0);
    par_for_in(input, &Environment::from_process(), profile, call_line)
}

fn line_of(span: Span) -> Option<u32> {
    u32::try_from(span.start().line)
        .ok()
        .filter(|&line| line > 0)
}

/// [`par_for`] with the environment and the line of the call given, so tests choose them.
pub(crate) fn par_for_in(
    input: TokenStream,
    environment: &Environment,
    profile: Profile,
    call_line: u32,
) -> TokenStream {
    let mut descriptor = false;
    match expand_input(input, environment, profile, call_line, &mut descriptor) {
        Ok(tokens) => tokens,
        Err(error) => {
            // The errors, then a value of the type the call would have had, so that one mistake is one error.
            let errors = error.to_compile_error();
            if descriptor {
                quote!({ #errors ::core::unreachable!() })
            } else {
                quote!({ #errors ::core::result::Result::<(), ::gust::map::Error>::Ok(()) })
            }
        }
    }
}

/// The expansion, or the first error. `descriptor` says whether the hidden `descriptor;` form was read, which decides
/// what value follows an error.
pub(crate) fn expand_input(
    input: TokenStream,
    environment: &Environment,
    profile: Profile,
    call_line: u32,
    descriptor: &mut bool,
) -> syn::Result<TokenStream> {
    let raw = diagnostics::parse2::<Raw>(Code::ParForForm, input)?;
    *descriptor = raw.options.descriptor;
    Plan::new(raw, profile)?.expand(environment, call_line)
}
