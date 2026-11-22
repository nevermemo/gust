//! Identifiers, parameter flavors and the Slang spelling of Rust types.

use syn::ext::IdentExt as _;

use super::arrays::array_length_literal;
use super::collisions;
use super::declarations::IN_INTERFACE;
use super::lowering::self_struct;
use crate::{
    diagnostics::{self, Code},
    math_types,
    names::peel_type,
    profile::{Builtin, GRID, StageProfile},
};

/// The Slang spelling of a user identifier (M6 plan 2.5): a raw prefix is stripped, and a name in the collision table
/// becomes `__gust_r_<name>`. Every site that writes a user name into Slang, and every descriptor `name:` string, goes
/// through this one function, so a renamed name is the same text everywhere.
pub(crate) fn emit_ident(ident: &syn::Ident) -> String {
    collisions::emit_name(&ident.unraw().to_string()).into_owned()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterFlavor {
    Invocation,
    /// `LocalId` or `SV_GroupThreadID` (slice m7.g1): `uint3` derived from the dispatch-thread id and the workgroup
    /// size, not an entry-point parameter and not in the descriptor.
    LocalId,
    /// `GroupId` or `SV_GroupID` (slice m7.g1): derived like [`ParameterFlavor::LocalId`].
    GroupId,
    /// `Workgroup<[T; N]>` (slice m7.g1): a module-scope `groupshared` array, not a resource and not in the
    /// descriptor.
    Workgroup,
    StorageRead,
    StorageReadWrite,
    Uniform,
    /// `Texture2D<T>` (slice mx.t3): a sampled texture, read with `load` and `sample`.
    SampledTexture,
    /// `StorageTexture2D<F>` (slice mx.t3): a read-write storage texture of one of the three 32-bit single-channel
    /// formats.
    StorageTexture,
    /// `WriteTexture2D<F>` (slice mx.t3): a write-only storage texture of any of the nine formats.
    WriteTexture,
    /// `Sampler` (slice mx.t3): a sampler, passed to `sample`.
    Sampler,
    Value,
}

impl ParameterFlavor {
    /// Whether the parameter is part of the kernel's host contract (the descriptor, the export record and the typed
    /// call): the dispatch-thread id and the resources. The derived ids and the workgroup arrays are the body's own.
    pub(crate) fn in_descriptor(self) -> bool {
        !matches!(self, Self::LocalId | Self::GroupId | Self::Workgroup)
    }

    /// Whether the parameter is a texture (not a sampler).
    pub(crate) fn is_texture(self) -> bool {
        matches!(
            self,
            Self::SampledTexture | Self::StorageTexture | Self::WriteTexture
        )
    }

    /// Whether the parameter is a texture or a sampler: it has a shader binding and the dialect uses it only through
    /// texture methods (GUST-E0736).
    pub(crate) fn is_texture_or_sampler(self) -> bool {
        self.is_texture() || self == Self::Sampler
    }
}

/// [`parameter_flavor_in`] for the `grid` profile, for the callers that classify a type without a kernel (the
/// reachability of uniform structs, the nested checks of the validator): they tell resources from values, and the
/// invocation names do not matter to them.
pub fn parameter_flavor(ty: &syn::Type) -> ParameterFlavor {
    parameter_flavor_in(&GRID, ty)
}

/// What a kernel parameter of type `ty` is: the invocation built-ins are the profile's (D63), the resources are the
/// dialect's.
pub(crate) fn parameter_flavor_in(stage: &dyn StageProfile, ty: &syn::Type) -> ParameterFlavor {
    match stage.invocation(ty) {
        Some(Builtin::Global) => return ParameterFlavor::Invocation,
        Some(Builtin::Local) => return ParameterFlavor::LocalId,
        Some(Builtin::Group) => return ParameterFlavor::GroupId,
        None => {}
    }
    let Some(name) = type_name(ty) else {
        return ParameterFlavor::Value;
    };
    match name.as_str() {
        "Workgroup" => ParameterFlavor::Workgroup,
        "Storage" | "StructuredBuffer" => ParameterFlavor::StorageRead,
        "StorageMut" | "RWStructuredBuffer" => ParameterFlavor::StorageReadWrite,
        "Uniform" | "ConstantBuffer" => ParameterFlavor::Uniform,
        "Texture2D" => ParameterFlavor::SampledTexture,
        "StorageTexture2D" => ParameterFlavor::StorageTexture,
        "WriteTexture2D" => ParameterFlavor::WriteTexture,
        "Sampler" => ParameterFlavor::Sampler,
        _ => ParameterFlavor::Value,
    }
}

pub(super) fn emit_type(ty: &syn::Type) -> syn::Result<String> {
    if let syn::Type::Tuple(tuple) = ty {
        if tuple.elems.is_empty() {
            return Ok("void".to_owned());
        }
        // `(A, B)` is Slang's `Tuple<A, B>` (spike M9.s0, P3).
        let elements = tuple
            .elems
            .iter()
            .map(emit_type)
            .collect::<syn::Result<Vec<_>>>()?;
        return Ok(format!("Tuple<{}>", elements.join(", ")));
    }
    let syn::Type::Path(path) = ty else {
        return Err(diagnostics::error(
            Code::TypeNeedsPath,
            ty,
            "Slang types must be named paths",
        ));
    };
    let segment = path.path.segments.last().expect("nonempty type path");
    let source_name = segment.ident.unraw().to_string();
    let name = match source_name.as_str() {
        "f32" | "float" => "float".to_owned(),
        "i32" | "int" => "int".to_owned(),
        "u32" | "uint" => "uint".to_owned(),
        "bool" => "bool".to_owned(),
        "Option" => "Optional".to_owned(),
        "Result" => "__GustResult".to_owned(),
        "SV_DispatchThreadID" | "LocalId" | "GroupId" | "SV_GroupThreadID" | "SV_GroupID" => {
            "uint3".to_owned()
        }
        // `Self` in the declaration of a trait is the implementing type, Slang's `This` (slice m9.g1).
        "Self" if IN_INTERFACE.with(std::cell::Cell::get) => "This".to_owned(),
        // `Self` in an `impl` block is the struct of the block (slice m9.a2).
        "Self" if self_struct().is_some() => {
            collisions::emit_name(&self_struct().expect("guarded by the match arm")).into_owned()
        }
        name if math_types::math_type(name).is_some() => math_types::math_type(name)
            .expect("guarded by the match arm")
            .slang
            .to_owned(),
        // Any other type name is a struct of the module, so it is renamed exactly like its declaration.
        name => collisions::emit_name(name).into_owned(),
    };
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return Ok(name);
    };
    let parameters = arguments
        .args
        .iter()
        .filter_map(|argument| match argument {
            syn::GenericArgument::Type(ty) => Some(emit_type(ty)),
            _ => None,
        })
        .collect::<syn::Result<Vec<_>>>()?;
    if source_name == "Result" {
        let payload = parameters.first().ok_or_else(|| {
            diagnostics::error(
                Code::ResultPayloadMissing,
                ty,
                "`Result` needs a payload type",
            )
        })?;
        return Ok(format!("{name}<{payload}>"));
    }
    Ok(format!("{name}<{}>", parameters.join(", ")))
}

/// Whether `ty` is the path `usize`.
pub(super) fn is_usize_type(ty: &syn::Type) -> bool {
    matches!(peel_type(ty), syn::Type::Path(path)
        if path.qself.is_none() && crate::names::path_is(&path.path, "usize"))
}

/// The Slang type of a local, a parameter or a result (slice m9.r1): [`emit_type`], and `T[N]` for `[T; N]`. A struct
/// field, a buffer or uniform element and the payload of an `Option` or `Result` still go through [`emit_type`], which
/// has no array arm, so an array there is refused exactly as before (GUST-E0813).
pub(super) fn emit_value_type(ty: &syn::Type) -> syn::Result<String> {
    let syn::Type::Array(array) = peel_type(ty) else {
        return emit_type(ty);
    };
    let length = array_length_literal(&array.len).ok_or_else(|| {
        diagnostics::error(
            Code::ArrayLength,
            &array.len,
            "an array length is an integer literal greater than zero",
        )
    })?;
    // A nested array has no arm in `emit_type` either (the validator refused it, GUST-E0217).
    Ok(format!("{}[{length}]", emit_type(&array.elem)?))
}

pub(super) fn typed_argument(argument: &syn::FnArg) -> syn::Result<(&syn::Ident, &syn::Type)> {
    let syn::FnArg::Typed(argument) = argument else {
        return Err(diagnostics::error(
            Code::SelfParameter,
            argument,
            "GPU functions cannot use self",
        ));
    };
    let syn::Pat::Ident(pattern) = argument.pat.as_ref() else {
        return Err(diagnostics::error(
            Code::ParameterNeedsIdentifier,
            &argument.pat,
            "GPU parameters need identifier names",
        ));
    };
    Ok((&pattern.ident, &argument.ty))
}

pub fn wrapper_element_type(ty: &syn::Type) -> syn::Result<&syn::Type> {
    let syn::Type::Path(path) = ty else {
        return Err(diagnostics::error(
            Code::ResourceElementPath,
            ty,
            "GPU resource requires a named element type",
        ));
    };
    let segment = path.path.segments.last().expect("nonempty resource type");
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return Err(diagnostics::error(
            Code::ResourceElementArity,
            segment,
            "GPU resource requires one element type",
        ));
    };
    arguments
        .args
        .iter()
        .find_map(|argument| match argument {
            syn::GenericArgument::Type(ty) => Some(ty),
            _ => None,
        })
        .ok_or_else(|| {
            diagnostics::error(
                Code::ResourceElementMissing,
                arguments,
                "GPU resource requires an element type",
            )
        })
}

pub(crate) fn type_name(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    path.path
        .segments
        .last()
        .map(|segment| segment.ident.unraw().to_string())
}
