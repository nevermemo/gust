//! Uniform parameters (slice m6.u2): which module structs a constant buffer reaches, the `Std140V1`
//! layout the macro computes for them, and the checks that keep a Rust struct and a Slang constant buffer one layout.
//!
//! The macro has no type information, so everything here is by name, like [`crate::math_types`]. A struct is
//! *uniform-reachable* when a `Uniform<T>` parameter of a kernel names it, or a uniform-reachable struct has it as a
//! field. Such a struct is declared `#[repr(C, align(16))]` (a constant-buffer struct has alignment and size rounded up
//! to 16) and must be padding-free in that layout: GUST never inserts padding, because the upload copies the bytes of
//! the Rust value and padding bytes are uninitialised (`gust::GpuPod`). A gap or a tail is therefore a compile error
//! that names the field and the explicit padding to write (`GUST-E0209`).

use std::collections::{HashMap, HashSet};

use syn::ext::IdentExt as _;

use crate::{
    diagnostics::{self, Code},
    math_types,
    slang::{ParameterFlavor, parameter_flavor, wrapper_element_type},
};

/// The alignment of a constant-buffer struct at least: structs are rounded up to 16 (std140).
const STRUCT_ALIGNMENT: u32 = 16;

/// How deep a struct may nest before the model gives up (a recursive struct is a Rust error, but the macro runs before
/// that error is found).
const MAX_NESTING: usize = 16;

/// The module's structs by unraw name.
pub(crate) fn module_structs(items: &[syn::Item]) -> HashMap<String, &syn::ItemStruct> {
    items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Struct(item) => Some((item.ident.unraw().to_string(), item)),
            _ => None,
        })
        .collect()
}

/// A resource parameter of a kernel and its element type.
struct Resource<'a> {
    flavor: ParameterFlavor,
    parameter: &'a syn::Type,
    element: &'a syn::Type,
}

/// Every storage and uniform parameter of every kernel of the module, with its element type. A resource type with no
/// element (`StructuredBuffer` alone) is left out: the emitter reports it.
fn kernel_resources(items: &[syn::Item]) -> syn::Result<Vec<Resource<'_>>> {
    let mut resources = Vec::new();
    for item in items {
        let syn::Item::Fn(function) = item else {
            continue;
        };
        if crate::expand::kernel_options(function)?.is_none() {
            continue;
        }
        for argument in &function.sig.inputs {
            let syn::FnArg::Typed(argument) = argument else {
                continue;
            };
            let flavor = parameter_flavor(&argument.ty);
            if !matches!(
                flavor,
                ParameterFlavor::StorageRead
                    | ParameterFlavor::StorageReadWrite
                    | ParameterFlavor::Uniform
            ) {
                continue;
            }
            if let Ok(element) = wrapper_element_type(&argument.ty) {
                resources.push(Resource {
                    flavor,
                    parameter: &argument.ty,
                    element,
                });
            }
        }
    }
    Ok(resources)
}

/// The name of the module struct `ty` is, when it is a bare path to one.
fn struct_name(ty: &syn::Type, structs: &HashMap<String, &syn::ItemStruct>) -> Option<String> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() || path.path.leading_colon.is_some() {
        return None;
    }
    let ident = path.path.get_ident()?;
    let name = ident.unraw().to_string();
    structs.contains_key(&name).then_some(name)
}

/// `roots` and every module struct reachable from them through fields.
fn close_over_fields(
    roots: impl IntoIterator<Item = String>,
    structs: &HashMap<String, &syn::ItemStruct>,
) -> HashSet<String> {
    let mut seen = HashSet::new();
    let mut pending: Vec<String> = roots.into_iter().collect();
    while let Some(name) = pending.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let Some(item) = structs.get(&name) else {
            continue;
        };
        if let syn::Fields::Named(fields) = &item.fields {
            pending.extend(
                fields
                    .named
                    .iter()
                    .filter_map(|field| struct_name(&field.ty, structs)),
            );
        }
    }
    seen
}

/// The module structs a constant buffer reaches (`uniform`) and the ones a storage element reaches (`storage`).
pub(crate) struct Reach {
    pub(crate) uniform: HashSet<String>,
    pub(crate) storage: HashSet<String>,
}

/// The reach of the kernels of a module. Total on any module: an element that is no module struct reaches nothing
/// (the checks of [`check_parameters`] reject a uniform one).
pub(crate) fn reach(items: &[syn::Item]) -> syn::Result<Reach> {
    let structs = module_structs(items);
    let resources = kernel_resources(items)?;
    let roots = |uniform: bool| {
        resources
            .iter()
            .filter(|resource| (resource.flavor == ParameterFlavor::Uniform) == uniform)
            .filter_map(|resource| struct_name(resource.element, &structs))
            .collect::<Vec<_>>()
    };
    Ok(Reach {
        uniform: close_over_fields(roots(true), &structs),
        storage: close_over_fields(roots(false), &structs),
    })
}

/// E0206, E0207 and E0205: the shape of the uniform parameters and of the structs they reach.
pub(crate) fn check_parameters(items: &[syn::Item]) -> syn::Result<()> {
    let structs = module_structs(items);
    for resource in kernel_resources(items)? {
        if resource.flavor == ParameterFlavor::Uniform
            && struct_name(resource.element, &structs).is_none()
        {
            return Err(diagnostics::error(
                Code::UniformElement,
                resource.parameter,
                "a uniform parameter holds one struct declared in the same `#[gpu]` module: write `Uniform<Params>` \
                 with `Params` a struct of the module (a scalar, a vector or a matrix goes in a field of it)",
            ));
        }
    }
    let reach = reach(items)?;
    for item in items {
        let syn::Item::Struct(item) = item else {
            continue;
        };
        let name = item.ident.unraw().to_string();
        if reach.uniform.contains(&name) && reach.storage.contains(&name) {
            return Err(diagnostics::error(
                Code::UniformAndStorageStruct,
                &item.ident,
                format!(
                    "`{name}` is reachable from both a uniform parameter and a storage buffer element: the two layouts \
                     differ (a constant-buffer struct is rounded up to 16 bytes) and copying between them crashes \
                     slangc on WGSL; declare a separate struct for each use"
                ),
            ));
        }
    }
    for item in items {
        let syn::Item::Struct(item) = item else {
            continue;
        };
        if !reach.uniform.contains(&item.ident.unraw().to_string()) {
            continue;
        }
        let syn::Fields::Named(fields) = &item.fields else {
            continue;
        };
        for field in &fields.named {
            let Some(name) = field_type_name(&field.ty) else {
                continue;
            };
            if name == "Mat2" || name == "bool" {
                return Err(diagnostics::error(
                    Code::UniformFieldType,
                    &field.ty,
                    format!(
                        "`{name}` cannot be a field of a struct a uniform reaches: {}",
                        if name == "Mat2" {
                            "a 2x2 matrix has column stride 16 in a constant buffer but 8 in Rust, so no Rust type \
                             matches it; use a `Vec4` holding the four entries, or a `Mat4`"
                        } else {
                            "a `bool` is one byte in Rust and a 4-byte value on the GPU, so the two layouts differ; use a \
                             `u32` holding 0 or 1"
                        }
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// The last segment of a field type that is a bare name.
fn field_type_name(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    path.path.get_ident().map(|ident| ident.unraw().to_string())
}

/// The `Std140V1` layout of a struct: the offsets and sizes of its fields, and its own alignment and size (both
/// rounded up to 16).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Std140Struct {
    pub(crate) size: u32,
    pub(crate) alignment: u32,
    /// Per field, in declaration order: the byte offset.
    pub(crate) offsets: Vec<u32>,
    /// Per field, in declaration order: the size in bytes.
    pub(crate) sizes: Vec<u32>,
}

/// The size and alignment of a field type in a constant buffer, when the macro can compute them from names alone: a
/// scalar, a vector, a `Mat4`, or a struct of the module whose fields are such types. `Mat2` has no layout here
/// (E0205), and a type the macro cannot see into has none either.
fn std140_field(
    ty: &syn::Type,
    structs: &HashMap<String, &syn::ItemStruct>,
    depth: usize,
) -> Option<(u32, u32)> {
    let name = field_type_name(ty)?;
    if name == "Mat2" {
        return None;
    }
    if let Some(layout) = math_types::known_layout(&name) {
        return Some(layout);
    }
    if depth > MAX_NESTING {
        return None;
    }
    let nested = std140_struct(structs.get(&name)?, structs, depth + 1)?;
    Some((nested.size, nested.alignment))
}

/// The constant-buffer layout of `item`, when every field has a known one.
pub(crate) fn std140_struct(
    item: &syn::ItemStruct,
    structs: &HashMap<String, &syn::ItemStruct>,
    depth: usize,
) -> Option<Std140Struct> {
    let syn::Fields::Named(fields) = &item.fields else {
        return None;
    };
    let mut offset = 0_u32;
    let mut alignment = STRUCT_ALIGNMENT;
    let mut offsets = Vec::new();
    let mut sizes = Vec::new();
    for field in &fields.named {
        let (size, field_alignment) = std140_field(&field.ty, structs, depth)?;
        offset = offset.checked_next_multiple_of(field_alignment)?;
        offsets.push(offset);
        sizes.push(size);
        offset = offset.checked_add(size)?;
        alignment = alignment.max(field_alignment);
    }
    Some(Std140Struct {
        size: offset.checked_next_multiple_of(alignment)?,
        alignment,
        offsets,
        sizes,
    })
}

/// `bytes` of explicit padding as the fields to write, such as "3 `f32` fields".
fn padding_fields(bytes: u32) -> String {
    let count = bytes / 4;
    if count == 1 {
        "one `f32` field".to_owned()
    } else {
        format!("{count} `f32` fields")
    }
}

/// E0209: a struct a uniform reaches has no gap before a field and no tail in its constant-buffer layout. The model
/// covers scalars, vectors, `Mat4` and the module's structs; a struct with a field the macro cannot see into is left to
/// the `const` assertions of the expansion, which fail compilation instead.
pub(crate) fn check_padding(items: &[syn::Item]) -> syn::Result<()> {
    let structs = module_structs(items);
    let reach = reach(items)?;
    for item in items {
        let syn::Item::Struct(item) = item else {
            continue;
        };
        let name = item.ident.unraw().to_string();
        if !reach.uniform.contains(&name) {
            continue;
        }
        let (Some(layout), syn::Fields::Named(fields)) =
            (std140_struct(item, &structs, 0), &item.fields)
        else {
            continue;
        };
        let mut end = 0_u32;
        for ((field, offset), size) in fields.named.iter().zip(&layout.offsets).zip(&layout.sizes) {
            if *offset != end {
                let gap = offset - end;
                let field_name = field
                    .ident
                    .as_ref()
                    .map_or_else(String::new, |ident| ident.unraw().to_string());
                return Err(diagnostics::error(
                    Code::UniformPadding,
                    field,
                    format!(
                        "field `{field_name}` of `{name}` would sit at byte {offset} of a constant buffer, {gap} bytes \
                         after the previous field ends at byte {end}: a struct a uniform reaches may contain no \
                         implicit padding; add {gap} bytes of explicit padding before it ({}), or reorder the fields",
                        padding_fields(gap),
                    ),
                ));
            }
            end = offset + size;
        }
        if layout.size != end {
            let tail = layout.size - end;
            return Err(diagnostics::error(
                Code::UniformPadding,
                &item.ident,
                format!(
                    "`{name}` ends at byte {end} but a constant-buffer struct is rounded up to {} bytes: a struct a \
                     uniform reaches may contain no implicit padding; add {tail} bytes of explicit padding at the end \
                     ({})",
                    layout.size,
                    padding_fields(tail),
                ),
            ));
        }
    }
    Ok(())
}
