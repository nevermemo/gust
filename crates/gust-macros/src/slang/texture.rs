//! Texture and sampler parameters (roadmap MX, slice mx.t3; decisions D62, D64, D91, D120).
//!
//! A kernel may take `Texture2D<T>` (sampled), `StorageTexture2D<F>` (read and write, three formats), `WriteTexture2D<F>`
//! (write only, nine formats) and `Sampler`. This module is the dialect's one table of them: which type arguments are
//! accepted, what each parameter declares in Slang, what the methods lower to, and the facts the descriptor, the export
//! record and the typed call need. The validator, the emitter and the expander all read it, so they cannot disagree.
//!
//! Lowering (probed with slangc 2026.20, WGSL and SPIR-V, `spirv-val` clean):
//!
//! | Rust | Slang |
//! | --- | --- |
//! | `Texture2D<T>` | `[[vk::binding(n, 0)]] Texture2D<T> name;` |
//! | `StorageTexture2D<F>` | `[[vk::binding(n, 0)]] [format("r32f")] RWTexture2D<T> name;` |
//! | `WriteTexture2D<F>` | `[[vk::binding(n, 0)]] [format("rgba32f")] WTexture2D<T> name;` |
//! | `Sampler` | `[[vk::binding(n, 0)]] SamplerState name;` |
//! | `t.load(x, y)` | `t[uint2(x, y)]` (level 0 of a sampled texture, a storage texel of the others) |
//! | `t.store(x, y, v)` | `t.Store(uint2(x, y), v)` |
//! | `t.sample(s, u, v)` | `t.SampleLevel(s, float2(u, v), 0.0)` |
//! | `t.width()`, `t.height()` | `__gpu_width_t`, `__gpu_height_t`, set in the prologue by `t.GetDimensions(..)` |

use std::fmt::Write as _;

use syn::{ext::IdentExt as _, visit::Visit};

use super::ParameterFlavor;

/// The access a texture parameter has in the shader: the `TextureAccess` of its `TextureLayout`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Access {
    Sampled,
    ReadWrite,
    WriteOnly,
}

impl Access {
    /// The spelling in an export record (`texture.access`).
    pub(crate) fn record_name(self) -> &'static str {
        match self {
            Self::Sampled => "sampled",
            Self::ReadWrite => "read_write",
            Self::WriteOnly => "write_only",
        }
    }

    /// The `gust::TextureAccess` variant, as tokens.
    pub(crate) fn tokens(self) -> proc_macro2::TokenStream {
        match self {
            Self::Sampled => quote::quote!(::gust::TextureAccess::Sampled),
            Self::ReadWrite => quote::quote!(::gust::TextureAccess::ReadWrite),
            Self::WriteOnly => quote::quote!(::gust::TextureAccess::WriteOnly),
        }
    }
}

/// The scalar a texel is made of, which is also the sample class of the texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scalar {
    F32,
    U32,
    I32,
}

impl Scalar {
    /// The name in a record's `texture.class`: `f32`, `u32` or `i32` (`SampleClass::name`).
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::U32 => "u32",
            Self::I32 => "i32",
        }
    }

    /// The `gust::texture::SampleClass` variant, as tokens.
    pub(crate) fn class_tokens(self) -> proc_macro2::TokenStream {
        match self {
            Self::F32 => quote::quote!(::gust::texture::SampleClass::Float),
            Self::U32 => quote::quote!(::gust::texture::SampleClass::Uint),
            Self::I32 => quote::quote!(::gust::texture::SampleClass::Sint),
        }
    }
}

/// One texel format of the dialect (the nine 32-bit formats of D62/D64), as `gust::texture::Format` names them.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FormatInfo {
    /// The marker type: `R32Float`, `Rgba32Uint`, ...
    pub marker: &'static str,
    /// The `gust::texture::Format` variant (the same spelling as the marker).
    variant: &'static str,
    /// WebGPU's name of the format (`Format::name`): the record's `texture.format`.
    pub name: &'static str,
    /// Slang's name of the format (`Format::slang_name`): the argument of `[format("..")]`.
    pub slang: &'static str,
    pub scalar: Scalar,
    pub channels: u32,
    /// Whether a `StorageTexture2D` (read and write) may have this format.
    pub read_write: bool,
    /// The Rust type of one texel, as a path from the crate root of `gust` (`f32`, `::gust::math::Vec2`, ...).
    texel: &'static str,
}

const fn format(
    marker: &'static str,
    name: &'static str,
    slang: &'static str,
    scalar: Scalar,
    channels: u32,
    texel: &'static str,
) -> FormatInfo {
    FormatInfo {
        marker,
        variant: marker,
        name,
        slang,
        scalar,
        channels,
        read_write: channels == 1,
        texel,
    }
}

/// The nine formats, in the order of `gust::texture::Format::ALL`.
pub(crate) const FORMATS: [FormatInfo; 9] = [
    format("R32Uint", "r32uint", "r32ui", Scalar::U32, 1, "u32"),
    format("R32Sint", "r32sint", "r32i", Scalar::I32, 1, "i32"),
    format("R32Float", "r32float", "r32f", Scalar::F32, 1, "f32"),
    format(
        "Rg32Uint",
        "rg32uint",
        "rg32ui",
        Scalar::U32,
        2,
        "::gust::math::UVec2",
    ),
    format(
        "Rg32Sint",
        "rg32sint",
        "rg32i",
        Scalar::I32,
        2,
        "::gust::math::IVec2",
    ),
    format(
        "Rg32Float",
        "rg32float",
        "rg32f",
        Scalar::F32,
        2,
        "::gust::math::Vec2",
    ),
    format(
        "Rgba32Uint",
        "rgba32uint",
        "rgba32ui",
        Scalar::U32,
        4,
        "::gust::math::UVec4",
    ),
    format(
        "Rgba32Sint",
        "rgba32sint",
        "rgba32i",
        Scalar::I32,
        4,
        "::gust::math::IVec4",
    ),
    format(
        "Rgba32Float",
        "rgba32float",
        "rgba32f",
        Scalar::F32,
        4,
        "::gust::math::Vec4",
    ),
];

/// A type of the texel table, which holds valid Rust types only.
fn texel_type_of(spelling: &str) -> syn::Type {
    let tokens: proc_macro2::TokenStream = spelling
        .parse()
        .expect("the texel table holds valid tokens");
    crate::diagnostics::parse2(crate::diagnostics::Code::TextureType, tokens)
        .expect("the texel table holds valid types")
}

impl FormatInfo {
    /// The marker type by its path from the crate root of `gust`: `::gust::texture::R32Float`.
    pub(crate) fn marker_tokens(&self) -> proc_macro2::TokenStream {
        let marker = quote::format_ident!("{}", self.marker);
        quote::quote!(::gust::texture::#marker)
    }

    /// The Rust type of one texel (`TextureFormat::Texel`).
    pub(crate) fn texel_type(&self) -> syn::Type {
        texel_type_of(self.texel)
    }

    /// The texel type by its bare name (`Vec2`, `f32`), the spelling the export record's layout writer knows.
    pub(crate) fn texel_leaf_type(&self) -> syn::Type {
        let leaf = self
            .texel
            .rsplit("::")
            .next()
            .expect("a path has a last segment");
        texel_type_of(leaf)
    }

    /// `::gust::texture::Format::<variant>`, as tokens.
    pub(crate) fn format_tokens(&self) -> proc_macro2::TokenStream {
        let variant = quote::format_ident!("{}", self.variant);
        quote::quote!(::gust::texture::Format::#variant)
    }
}

fn slang_vector(scalar: Scalar, channels: u32) -> String {
    let base = match scalar {
        Scalar::F32 => "float",
        Scalar::U32 => "uint",
        Scalar::I32 => "int",
    };
    if channels == 1 {
        base.to_owned()
    } else {
        format!("{base}{channels}")
    }
}

/// The texel types a sampled `Texture2D<T>` accepts: spelling, scalar and width. There is no 3-wide texel (D62).
const SAMPLED_TEXELS: [(&str, Scalar, u32); 12] = [
    ("f32", Scalar::F32, 1),
    ("float", Scalar::F32, 1),
    ("u32", Scalar::U32, 1),
    ("uint", Scalar::U32, 1),
    ("i32", Scalar::I32, 1),
    ("int", Scalar::I32, 1),
    ("Vec2", Scalar::F32, 2),
    ("Vec4", Scalar::F32, 4),
    ("UVec2", Scalar::U32, 2),
    ("UVec4", Scalar::U32, 4),
    ("IVec2", Scalar::I32, 2),
    ("IVec4", Scalar::I32, 4),
];

/// What a texture parameter's type says.
#[derive(Debug)]
pub(crate) struct TextureSpec<'a> {
    pub access: Access,
    /// The format of a storage or write-only texture.
    pub format: Option<&'static FormatInfo>,
    /// The texel type as the user wrote it, for a sampled texture.
    pub sampled_texel: Option<&'a syn::Type>,
    pub scalar: Scalar,
    pub channels: u32,
}

impl TextureSpec<'_> {
    /// Whether `sample` exists: a sampled texture of float texels.
    pub(crate) fn filterable(&self) -> bool {
        self.access == Access::Sampled && self.scalar == Scalar::F32
    }

    /// The Rust type of one texel: the typed call's `Element`.
    pub(crate) fn element_type(&self) -> syn::Type {
        match (self.sampled_texel, self.format) {
            (Some(texel), _) => texel.clone(),
            (None, Some(format)) => format.texel_type(),
            (None, None) => unreachable!("a texture has a texel type or a format"),
        }
    }

    /// The Slang type of one texel.
    fn slang_texel(&self) -> String {
        slang_vector(self.scalar, self.channels)
    }
}

/// The type arguments of the last path segment: `Ok(types)` (lifetimes are not counted), or the reason the arguments
/// are not a plain list of types.
fn type_arguments(ty: &syn::Type) -> Result<Vec<&syn::Type>, String> {
    let syn::Type::Path(path) = ty else {
        return Err("a texture type is written as a path".to_owned());
    };
    let segment = path.path.segments.last().expect("a path has a segment");
    match &segment.arguments {
        syn::PathArguments::None => Ok(Vec::new()),
        syn::PathArguments::AngleBracketed(arguments) => {
            let mut types = Vec::new();
            for argument in &arguments.args {
                match argument {
                    syn::GenericArgument::Type(ty) => types.push(ty),
                    syn::GenericArgument::Lifetime(_) => {}
                    _ => return Err("a texture type takes types only".to_owned()),
                }
            }
            Ok(types)
        }
        syn::PathArguments::Parenthesized(_) => {
            Err("a texture type takes no parentheses".to_owned())
        }
    }
}

/// The last segment name of a bare type path (`R32Float`, `gust::texture::R32Float` and `f32` all work), or `None`.
fn bare_name(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    let segment = path.path.segments.last()?;
    segment
        .arguments
        .is_none()
        .then(|| segment.ident.unraw().to_string())
}

/// A texel type by its last segment alone: `gust::math::Vec4` is `Vec4`, which is the spelling the export record's
/// layout writer knows. The record then describes a path-spelled texel the way it describes a bare one.
pub(crate) fn leaf_type(ty: &syn::Type) -> syn::Type {
    if let syn::Type::Path(path) = ty
        && path.qself.is_none()
        && let Some(segment) = path.path.segments.last()
    {
        let mut leaf = segment.clone();
        leaf.arguments = syn::PathArguments::None;
        return syn::Type::Path(syn::TypePath {
            qself: None,
            path: syn::Path::from(leaf),
        });
    }
    ty.clone()
}

fn format_named(name: &str) -> Option<&'static FormatInfo> {
    FORMATS.iter().find(|format| format.marker == name)
}

const SAMPLED_LIST: &str =
    "`f32`, `u32`, `i32`, `Vec2`, `Vec4`, `UVec2`, `UVec4`, `IVec2` or `IVec4`";
const READ_WRITE_LIST: &str = "`R32Uint`, `R32Sint` or `R32Float`";
const WRITE_LIST: &str = "`R32Uint`, `R32Sint`, `R32Float`, `Rg32Uint`, `Rg32Sint`, `Rg32Float`, `Rgba32Uint`, \
                          `Rgba32Sint` or `Rgba32Float`";

/// What the type of a `Texture2D`, `StorageTexture2D` or `WriteTexture2D` parameter says, or the reason it is not a
/// texture type of the dialect (GUST-E0731). `flavor` is what [`super::parameter_flavor_in`] made of the type.
pub(crate) fn texture_spec(
    flavor: ParameterFlavor,
    ty: &syn::Type,
) -> Result<TextureSpec<'_>, String> {
    let arguments = type_arguments(ty)?;
    let [argument] = arguments.as_slice() else {
        return Err(match flavor {
            ParameterFlavor::SampledTexture => {
                format!("`Texture2D` takes one texel type: {SAMPLED_LIST}")
            }
            ParameterFlavor::StorageTexture => {
                format!("`StorageTexture2D` takes one format: {READ_WRITE_LIST}")
            }
            _ => format!("`WriteTexture2D` takes one format: {WRITE_LIST}"),
        });
    };
    match flavor {
        ParameterFlavor::SampledTexture => {
            let name = bare_name(argument).unwrap_or_default();
            let Some((_, scalar, channels)) = SAMPLED_TEXELS.iter().find(|row| row.0 == name)
            else {
                return Err(format!(
                    "`Texture2D<T>` takes a texel type of {SAMPLED_LIST}; a 3-wide texel has no texture format"
                ));
            };
            Ok(TextureSpec {
                access: Access::Sampled,
                format: None,
                sampled_texel: Some(argument),
                scalar: *scalar,
                channels: *channels,
            })
        }
        ParameterFlavor::StorageTexture | ParameterFlavor::WriteTexture => {
            let write_only = flavor == ParameterFlavor::WriteTexture;
            let name = bare_name(argument).unwrap_or_default();
            let Some(format) = format_named(&name) else {
                let list = if write_only {
                    WRITE_LIST
                } else {
                    READ_WRITE_LIST
                };
                let wrapper = if write_only {
                    "WriteTexture2D"
                } else {
                    "StorageTexture2D"
                };
                return Err(format!("`{wrapper}<F>` takes a format of {list}"));
            };
            if !write_only && !format.read_write {
                return Err(format!(
                    "`StorageTexture2D<{}>` is not available: only {READ_WRITE_LIST} can be read and written in a \
                     kernel; `WriteTexture2D<{}>` is the write-only spelling",
                    format.marker, format.marker
                ));
            }
            Ok(TextureSpec {
                access: if write_only {
                    Access::WriteOnly
                } else {
                    Access::ReadWrite
                },
                format: Some(format),
                sampled_texel: None,
                scalar: format.scalar,
                channels: format.channels,
            })
        }
        _ => unreachable!("texture_spec is called for a texture flavor"),
    }
}

/// Whether `ty` is a `Sampler` with no type arguments.
pub(crate) fn sampler_has_no_arguments(ty: &syn::Type) -> bool {
    matches!(type_arguments(ty), Ok(arguments) if arguments.is_empty())
        && matches!(ty, syn::Type::Path(path) if path.path.segments.last().is_some_and(|s| s.arguments.is_none()))
}

/// The Slang declaration of a texture or sampler parameter bound at `binding` in set 0.
pub(crate) fn declaration(
    flavor: ParameterFlavor,
    ty: &syn::Type,
    name: &str,
    binding: u32,
) -> syn::Result<String> {
    if flavor == ParameterFlavor::Sampler {
        return Ok(format!(
            "[[vk::binding({binding}, 0)]] SamplerState {name};"
        ));
    }
    let spec = texture_spec(flavor, ty).map_err(|message| {
        crate::diagnostics::error(crate::diagnostics::Code::TextureType, ty, &message)
    })?;
    let texel = match spec.sampled_texel {
        Some(texel) => super::emit_type(texel)?,
        None => spec.slang_texel(),
    };
    Ok(match (flavor, spec.format) {
        (ParameterFlavor::SampledTexture, _) => {
            format!("[[vk::binding({binding}, 0)]] Texture2D<{texel}> {name};")
        }
        (ParameterFlavor::StorageTexture, Some(format)) => format!(
            "[[vk::binding({binding}, 0)]] [format(\"{}\")] RWTexture2D<{texel}> {name};",
            format.slang
        ),
        (_, Some(format)) => format!(
            "[[vk::binding({binding}, 0)]] [format(\"{}\")] WTexture2D<{texel}> {name};",
            format.slang
        ),
        (_, None) => unreachable!("a storage texture has a format"),
    })
}

/// The methods a texture parameter has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Method {
    Load,
    Store,
    Sample,
    Width,
    Height,
}

impl Method {
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "load" => Self::Load,
            "store" => Self::Store,
            "sample" => Self::Sample,
            "width" => Self::Width,
            "height" => Self::Height,
            _ => return None,
        })
    }

    /// The number of arguments, the receiver not counted.
    pub(crate) fn arity(self) -> usize {
        match self {
            Self::Load => 2,
            Self::Store | Self::Sample => 3,
            Self::Width | Self::Height => 0,
        }
    }

    /// Whether a texture of `access` has the method (`sample` also needs float texels, which the caller checks).
    pub(crate) fn exists_on(self, access: Access) -> bool {
        match self {
            Self::Width | Self::Height => true,
            Self::Load => access != Access::WriteOnly,
            Self::Store => access != Access::Sampled,
            Self::Sample => access == Access::Sampled,
        }
    }
}

/// Whether `call` is spelled like a texture method. The validator has already refused every such call whose receiver
/// is not a texture parameter, so the emitter needs the name only.
pub(crate) fn is_texture_method(call: &syn::ExprMethodCall) -> bool {
    Method::from_name(&crate::names::name_of(&call.method))
        .is_some_and(|method| method.arity() == call.args.len())
}

/// The identifier of a receiver written as a bare name, with any parentheses or groups around it taken off:
/// `out`, `(out)` and `((out))` are all `out`. `width()` and `height()` name a prologue local after it, so they must
/// agree with the validator's `direct_name` about what a plain receiver is.
fn receiver_ident(expression: &syn::Expr) -> Option<&syn::Ident> {
    match expression {
        syn::Expr::Path(path) if path.qself.is_none() => path.path.get_ident(),
        syn::Expr::Paren(paren) => receiver_ident(&paren.expr),
        syn::Expr::Group(group) => receiver_ident(&group.expr),
        _ => None,
    }
}

/// The Slang of a texture method call (the table in the module comment).
pub(crate) fn emit_method(call: &syn::ExprMethodCall) -> syn::Result<String> {
    let method = Method::from_name(&crate::names::name_of(&call.method))
        .expect("guarded by is_texture_method");
    let receiver = super::emit_expression(&call.receiver)?;
    // The size of a texture is a local named after the parameter (the prologue), so it is lowered from the stripped
    // identifier and a parenthesised receiver gives the same name as a bare one.
    let size_name = || {
        receiver_ident(&call.receiver)
            .map(super::emit_ident)
            .ok_or_else(|| {
                crate::diagnostics::error(
                    crate::diagnostics::Code::TextureUse,
                    call,
                    "`width()` and `height()` are called on a texture parameter by name",
                )
            })
    };
    let argument = |index: usize| super::emit_expression(&call.args[index]);
    Ok(match method {
        Method::Load => format!("{receiver}[uint2({}, {})]", argument(0)?, argument(1)?),
        Method::Store => format!(
            "{receiver}.Store(uint2({}, {}), {})",
            argument(0)?,
            argument(1)?,
            argument(2)?
        ),
        Method::Sample => format!(
            "{receiver}.SampleLevel({}, float2({}, {}), 0.0)",
            argument(0)?,
            argument(1)?,
            argument(2)?
        ),
        Method::Width => format!("__gpu_width_{}", size_name()?),
        Method::Height => format!("__gpu_height_{}", size_name()?),
    })
}

/// The texture parameters of a kernel whose `width()` or `height()` the body calls.
struct DimensionUses<'a> {
    textures: &'a [String],
    used: Vec<String>,
}

impl<'ast> Visit<'ast> for DimensionUses<'_> {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if matches!(
            crate::names::name_of(&call.method).as_str(),
            "width" | "height"
        ) && let Some(ident) = receiver_ident(&call.receiver)
        {
            let name = super::emit_ident(ident);
            if self.textures.contains(&name) && !self.used.contains(&name) {
                self.used.push(name);
            }
        }
        syn::visit::visit_expr_method_call(self, call);
    }
}

/// The statements that read the size of each texture whose `width()` or `height()` the body calls, in parameter
/// order: `__gpu_width_<t>` and `__gpu_height_<t>` are locals set by `GetDimensions`. Empty for a kernel that never
/// asks, so the Slang of every existing kernel is unchanged.
pub(crate) fn dimension_prologue(textures: &[String], kernel: &syn::ItemFn) -> String {
    let mut uses = DimensionUses {
        textures,
        used: Vec::new(),
    };
    uses.visit_block(&kernel.block);
    let mut text = String::new();
    for name in textures.iter().filter(|name| uses.used.contains(name)) {
        writeln!(text, "    uint __gpu_width_{name};").unwrap();
        writeln!(text, "    uint __gpu_height_{name};").unwrap();
        writeln!(
            text,
            "    {name}.GetDimensions(__gpu_width_{name}, __gpu_height_{name});"
        )
        .unwrap();
    }
    text
}
