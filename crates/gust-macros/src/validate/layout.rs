//! Types and layout: struct, enum, tuple and array layout checks and the stored-array lengths.

use super::*;

/// The size and alignment of a field type the macro can compute from names alone: a scalar, a vector, a matrix or a
/// struct of the module whose fields are all such types. `None` when any part is unknown (a type the macro cannot see
/// into), so the `const` assertions the expansion emits stay the check for it.
pub(super) fn field_layout(
    ty: &syn::Type,
    structs: &HashMap<String, &syn::ItemStruct>,
    depth: usize,
) -> Option<(u32, u32)> {
    // `[T; N]` is `N` elements of `T` end to end with the alignment of `T` (a Rust array has no padding between
    // elements, and `GpuPod for [T; N]` admits only elements whose stride is their size; slice m9.r2a).
    if let syn::Type::Array(array) = peel_type(ty) {
        let length = crate::slang::array_length_literal(&array.len)?;
        if depth > 16 {
            return None;
        }
        let (size, alignment) = field_layout(&array.elem, structs, depth + 1)?;
        return Some((size.checked_mul(length)?, alignment));
    }
    let syn::Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() || path.path.leading_colon.is_some() {
        return None;
    }
    let ident = path.path.get_ident()?;
    let name = ident.unraw().to_string();
    if let Some(layout) = math_types::known_layout(&name) {
        return Some(layout);
    }
    // A struct cannot contain itself, but the guard keeps a malformed module from recursing.
    if depth > 16 {
        return None;
    }
    struct_layout(structs.get(&name)?, structs, depth + 1)
        .map(|(size, alignment, _)| (size, alignment))
}

/// A struct's `#[repr(C)]` size, alignment and the offsets of its fields, when every field has a known layout.
pub(super) fn struct_layout(
    item: &syn::ItemStruct,
    structs: &HashMap<String, &syn::ItemStruct>,
    depth: usize,
) -> Option<(u32, u32, Vec<u32>)> {
    let syn::Fields::Named(fields) = &item.fields else {
        return None;
    };
    let mut offset = 0_u32;
    let mut alignment = 1_u32;
    let mut offsets = Vec::new();
    for field in &fields.named {
        let (size, field_alignment) = field_layout(&field.ty, structs, depth)?;
        offset = offset.checked_next_multiple_of(field_alignment)?;
        offsets.push(offset);
        offset = offset.checked_add(size)?;
        alignment = alignment.max(field_alignment);
    }
    Some((
        offset.checked_next_multiple_of(alignment)?,
        alignment,
        offsets,
    ))
}

/// E0208: a struct whose layout has implicit padding is not a `GpuPod` type (`pod_slice_as_bytes` would read the
/// padding bytes, which are uninitialised). The model covers scalars, vectors, matrices and the module's structs;
/// anything else is left to the expansion's `const` assertions.
pub(super) fn check_padding(
    item: &syn::ItemStruct,
    structs: &HashMap<String, &syn::ItemStruct>,
    uniform_reached: bool,
) -> syn::Result<()> {
    // A struct a uniform reaches may be marked `#[pad]` (m14.pad, D239), which fills the gap and the tail.
    let hint = if uniform_reached {
        ", or mark the struct `#[pad]` and GUST inserts the padding"
    } else {
        ""
    };
    let syn::Fields::Named(fields) = &item.fields else {
        return Ok(());
    };
    let Some((size, _, offsets)) = struct_layout(item, structs, 0) else {
        return Ok(());
    };
    let mut end = 0_u32;
    for (field, offset) in fields.named.iter().zip(offsets) {
        let Some((field_size, _)) = field_layout(&field.ty, structs, 0) else {
            return Ok(());
        };
        let name = field
            .ident
            .as_ref()
            .map_or_else(String::new, ToString::to_string);
        if offset != end {
            return Err(diagnostics::error(
                Code::ImplicitPadding,
                field,
                format!(
                    "field `{name}` of `{}` would sit at offset {offset}, {} bytes after the previous field ends at {end}: GPU structs may not contain implicit padding; order the fields by decreasing alignment, or fill the gap with an explicit field{hint}",
                    item.ident,
                    offset - end,
                ),
            ));
        }
        end = offset + field_size;
    }
    if size != end {
        return Err(diagnostics::error(
            Code::ImplicitPadding,
            &item.ident,
            format!(
                "`{}` ends at byte {end} but its size is {size}: GPU structs may not contain implicit tail padding; add explicit fields up to a multiple of the largest field alignment{hint}",
                item.ident,
            ),
        ));
    }
    Ok(())
}

/// GUST-E0211: an enum is a value of the invocation, never a stored one. No `GpuPod` exists for a type with invalid bit
/// patterns, so a struct field, a buffer or uniform element (or a `Workgroup` element) naming an enum is refused.
pub(super) fn check_enum_layout(
    stage: &dyn StageProfile,
    items: &[syn::Item],
    enums: &HashMap<String, EnumFacts>,
) -> syn::Result<()> {
    use crate::slang::{ParameterFlavor, parameter_flavor_in};

    if enums.is_empty() {
        return Ok(());
    }
    /// The first plain path to an enum anywhere inside a type.
    struct Mention<'e> {
        enums: &'e HashMap<String, EnumFacts>,
        found: Option<String>,
    }
    impl<'ast> Visit<'ast> for Mention<'_> {
        fn visit_type_path(&mut self, ty: &'ast syn::TypePath) {
            if let Some(name) = ty.path.get_ident().map(|ident| ident.unraw().to_string())
                && self.enums.contains_key(&name)
            {
                self.found.get_or_insert(name);
            }
            visit::visit_type_path(self, ty);
        }
    }
    let mention = |ty: &syn::Type| {
        let mut scan = Mention { enums, found: None };
        scan.visit_type(ty);
        scan.found
    };
    for item in items {
        match item {
            syn::Item::Struct(item) => {
                for field in &item.fields {
                    if let Some(name) = mention(&field.ty) {
                        return Err(diagnostics::error(
                            Code::EnumInResource,
                            &field.ty,
                            format!(
                                "`{name}` is an enum, and an enum is not a field of a struct: it has no `GpuPod` layout (a stored value could be none of its variants); keep the enum in a local or a parameter and store its integer (`mode as u32`)"
                            ),
                        ));
                    }
                }
            }
            syn::Item::Fn(function)
                if function
                    .attrs
                    .iter()
                    .any(|attribute| attribute.path().is_ident("kernel")) =>
            {
                for argument in &function.sig.inputs {
                    let syn::FnArg::Typed(argument) = argument else {
                        continue;
                    };
                    if matches!(
                        parameter_flavor_in(stage, &argument.ty),
                        ParameterFlavor::StorageRead
                            | ParameterFlavor::StorageReadWrite
                            | ParameterFlavor::Uniform
                            | ParameterFlavor::Workgroup
                    ) && let Some(name) = mention(&argument.ty)
                    {
                        return Err(diagnostics::error(
                            Code::EnumInResource,
                            &argument.ty,
                            format!(
                                "`{name}` is an enum, and an enum is not the element of a buffer, a uniform or workgroup memory: it has no `GpuPod` layout; store its integer (`mode as u32`) and match on it where it is used"
                            ),
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// GUST-E0212: a tuple is a value of the invocation, never a stored one. Rust gives a tuple no layout (the order and
/// the padding of its elements are unspecified), so no `GpuPod` can describe one: a struct field, a buffer or uniform
/// element and a `Workgroup` element naming a tuple anywhere inside the type are refused. A struct without named fields
/// is left to GUST-E0104.
pub(super) fn check_tuple_layout(stage: &dyn StageProfile, items: &[syn::Item]) -> syn::Result<()> {
    use crate::slang::{ParameterFlavor, parameter_flavor_in};

    /// Whether a type holds a tuple with elements anywhere inside it.
    struct Mention(bool);
    impl<'ast> Visit<'ast> for Mention {
        fn visit_type_tuple(&mut self, ty: &'ast syn::TypeTuple) {
            self.0 |= !ty.elems.is_empty();
            visit::visit_type_tuple(self, ty);
        }
    }
    let mentions = |ty: &syn::Type| {
        let mut scan = Mention(false);
        scan.visit_type(ty);
        scan.0
    };
    for item in items {
        match item {
            syn::Item::Struct(item) if matches!(item.fields, syn::Fields::Named(_)) => {
                for field in &item.fields {
                    if mentions(&field.ty) {
                        return Err(diagnostics::error(
                            Code::TupleInResource,
                            &field.ty,
                            "a tuple is not a field of a struct: Rust gives a tuple no layout, so it has no `GpuPod` description; keep the tuple in a local, a parameter or a result and store its elements as fields of their own",
                        ));
                    }
                }
            }
            syn::Item::Fn(function)
                if function
                    .attrs
                    .iter()
                    .any(|attribute| attribute.path().is_ident("kernel")) =>
            {
                for argument in &function.sig.inputs {
                    let syn::FnArg::Typed(argument) = argument else {
                        continue;
                    };
                    if matches!(
                        parameter_flavor_in(stage, &argument.ty),
                        ParameterFlavor::StorageRead
                            | ParameterFlavor::StorageReadWrite
                            | ParameterFlavor::Uniform
                            | ParameterFlavor::Workgroup
                    ) && mentions(&argument.ty)
                    {
                        return Err(diagnostics::error(
                            Code::TupleInResource,
                            &argument.ty,
                            "a tuple is not the element of a buffer, a uniform or workgroup memory: Rust gives a tuple no layout, so it has no `GpuPod` description; use a struct with named fields",
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// The lengths a stored array may have: the ones `gust::GpuPod` is implemented for (D159).
pub(super) const STORED_ARRAY_LENGTHS: &str = "1 to 32 and 64, 128, 256, 512, 1024, 2048 and 4096";

/// Whether `[T; length]` has a `GpuPod` impl (D159): 1 to 32, and the powers of two from 64 to 4096.
pub(super) fn stored_array_length_ok(length: u32) -> bool {
    (1..=32).contains(&length) || ((64..=4096).contains(&length) && length.is_power_of_two())
}

/// The outermost array type inside `ty`, if there is one.
pub(super) fn first_array(ty: &syn::Type) -> Option<&syn::TypeArray> {
    struct First<'ast>(Option<&'ast syn::TypeArray>);
    impl<'ast> Visit<'ast> for First<'ast> {
        fn visit_type_array(&mut self, ty: &'ast syn::TypeArray) {
            self.0.get_or_insert(ty);
        }
    }
    let mut scan = First(None);
    scan.visit_type(ty);
    scan.0
}

/// GUST-E0214, GUST-E0216 and GUST-E0217 for an array that is stored: a struct field or the element of a buffer. A
/// stored array has the layout of `GpuPod for [T; N]` (D159), so `N` is one of [`STORED_ARRAY_LENGTHS`] and the element
/// is one whose std430 stride is its Rust size. `place` says where the array stands, for the message.
pub(super) fn check_stored_array(array: &syn::TypeArray, place: &str) -> syn::Result<()> {
    let Some(length) = crate::slang::array_length_literal(&array.len) else {
        return Err(diagnostics::error(
            Code::ArrayLength,
            &array.len,
            "an array length is an integer literal greater than zero (`[f32; 4]`): not `0`, a constant or an expression",
        ));
    };
    if !stored_array_length_ok(length) {
        return Err(diagnostics::error(
            Code::ArrayLength,
            &array.len,
            format!(
                "an array that is {place} has one of the lengths {STORED_ARRAY_LENGTHS}, the ones `GpuPod` is implemented for, and `{length}` is none of them: use one of those lengths, or keep the longer array in a buffer of its own elements (a local array may have any length above zero)"
            ),
        ));
    }
    if let syn::Type::Path(path) = peel_type(&array.elem)
        && let Some(ident) = path.path.get_ident()
        && let Some(math) = math_types::math_type(&ident.unraw().to_string())
        && !math.buffer
    {
        return Err(diagnostics::error(
            Code::ArrayStride,
            &array.elem,
            format!(
                "`{}` cannot be the element of an array that is {place}: its stride in a buffer differs from its size in Rust, so no Rust array matches the buffer; use the 4-wide type, or one array per component",
                math.name
            ),
        ));
    }
    if let Some(problem) = array_element_problem(&array.elem) {
        return Err(diagnostics::error(Code::ArrayElement, &array.elem, problem));
    }
    Ok(())
}

/// Arrays that are stored (slice m9.r2a, D159): a struct field, a buffer element, and never a uniform. GUST-E0215 for a
/// uniform element that is or holds an array and for a field array of a struct a uniform reaches (std140 would round
/// the stride of the array up to 16; the scope choice of D159, not a limit of Slang); GUST-E0216, GUST-E0217 and
/// GUST-E0214 through [`check_stored_array`] for the rest. `Workgroup<[T; N]>` is its own shape (GUST-E0210), so it is
/// left alone: a nested array or a struct element stays refused there.
pub(super) fn check_array_layout(stage: &dyn StageProfile, items: &[syn::Item]) -> syn::Result<()> {
    use crate::slang::{ParameterFlavor, parameter_flavor_in};

    let kernels = items.iter().filter_map(|item| match item {
        syn::Item::Fn(function) => Some(function),
        _ => None,
    });
    let mut resources: Vec<(ParameterFlavor, &syn::Type)> = Vec::new();
    for function in kernels {
        if crate::expand::kernel_attribute_in(stage, function)?.is_none() {
            continue;
        }
        for argument in &function.sig.inputs {
            let syn::FnArg::Typed(argument) = argument else {
                continue;
            };
            let flavor = parameter_flavor_in(stage, &argument.ty);
            if matches!(
                flavor,
                ParameterFlavor::StorageRead
                    | ParameterFlavor::StorageReadWrite
                    | ParameterFlavor::Uniform
            ) && let Ok(element) = crate::slang::wrapper_element_type(&argument.ty)
            {
                resources.push((flavor, element));
            }
        }
    }
    for &(flavor, element) in &resources {
        if flavor == ParameterFlavor::Uniform
            && let Some(array) = first_array(element)
        {
            return Err(diagnostics::error(
                Code::ArrayInUniform,
                array,
                "an array is not part of a uniform: a constant buffer lays an array out with std140 rules, which no `GpuPod` array promises, and arrays are storage-buffer types in this release; put the array in a storage buffer (`StructuredBuffer<[T; N]>`), or pass its elements as fields",
            ));
        }
    }
    let uniform_structs = crate::uniform::reach(stage, items)?.uniform;
    for item in items {
        let syn::Item::Struct(item) = item else {
            continue;
        };
        if !matches!(item.fields, syn::Fields::Named(_)) {
            continue;
        }
        let uniform = uniform_structs.contains(&item.ident.unraw().to_string());
        for field in &item.fields {
            let Some(array) = first_array(&field.ty) else {
                continue;
            };
            if uniform {
                return Err(diagnostics::error(
                    Code::ArrayInUniform,
                    array,
                    format!(
                        "`{}` is reached from a uniform parameter, and an array is not part of a uniform: a constant buffer lays an array out with std140 rules, which no `GpuPod` array promises; keep the array in a struct that only storage buffers use, or give the uniform one field per element",
                        item.ident
                    ),
                ));
            }
            check_stored_array(array, "a struct field")?;
        }
    }
    for &(flavor, element) in &resources {
        if flavor != ParameterFlavor::Uniform
            && let Some(array) = first_array(element)
        {
            check_stored_array(array, "the element of a buffer")?;
        }
    }
    Ok(())
}

/// Whether `ty` is a tuple type with at least one element (`()` is the unit type, which is no tuple here).
pub(super) fn is_tuple_type(ty: &syn::Type) -> bool {
    match peel_type(ty) {
        syn::Type::Tuple(tuple) => !tuple.elems.is_empty(),
        _ => false,
    }
}

/// The most elements a tuple has in the dialect (GUST-E0213). Slang has no such limit (spike M9.s0, P3: a 9-tuple
/// compiles and validates); the bound is the dialect's own, to keep the surface that is tested small.
pub(super) const MAX_TUPLE_ELEMENTS: usize = 8;

/// Why `element` cannot be an element type of a tuple, or `None`. An element is a path type other than `Option` and
/// `Result` (scalars, vectors, matrices, module structs and enums) or a tuple; arrays, references and the rest are
/// refused (GUST-E0213).
pub(super) fn tuple_element_problem(element: &syn::Type) -> Option<&'static str> {
    match peel_type(element) {
        syn::Type::Tuple(_) => None,
        syn::Type::Path(path) => {
            let last = path.path.segments.last()?.ident.unraw().to_string();
            matches!(last.as_str(), "Option" | "Result").then_some(
                "a tuple element cannot be an `Option` or a `Result`: keep the optional value in its own local",
            )
        }
        _ => Some(
            "a tuple element is a scalar, vector, matrix, struct, enum or tuple type (no array, reference or other type)",
        ),
    }
}

/// Why `element` cannot be the element type of an array (GUST-E0217): `None` for a scalar, vector, matrix or struct or
/// enum of the module (slice m9.r1).
pub(super) fn array_element_problem(element: &syn::Type) -> Option<&'static str> {
    match peel_type(element) {
        syn::Type::Array(_) => Some(
            "an array element cannot be an array: nested arrays are not supported, use one flat array and index `i * N + j`",
        ),
        syn::Type::Path(path) => {
            let last = path.path.segments.last()?.ident.unraw().to_string();
            if matches!(last.as_str(), "Option" | "Result") {
                Some(
                    "an array element cannot be an `Option` or a `Result`: keep the optional value in its own local",
                )
            } else if crate::slang::parameter_flavor(&syn::Type::Path(path.clone()))
                != crate::slang::ParameterFlavor::Value
            {
                Some(
                    "an array element must be a value type: a buffer, uniform, workgroup array, texture, sampler or invocation id is a resource of the kernel",
                )
            } else {
                None
            }
        }
        _ => Some(
            "an array element is a scalar, vector, matrix, struct or enum type of the module (no tuple, reference or other type)",
        ),
    }
}
