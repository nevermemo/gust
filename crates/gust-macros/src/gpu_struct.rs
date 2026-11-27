//! `#[derive(GpuStruct)]` (plan m14.s2, slice s2s, part a, D279).
//!
//! A derive cannot rewrite the struct, so it checks the shape the struct was written in and refuses what a GPU row may
//! not be. The shape is the one a `#[gpu_module]` accepts for a storage struct (`validate/layout.rs`), checked here by
//! name: named fields, written `#[repr(C)]` (the derive checks it and does not add it), no generics, at least one field,
//! and every field a scalar (`f32`, `u32`, `i32`), a vector or a matrix of `gust::math` that a buffer may hold, written
//! by its bare name. The layout is the std430 one of `#[gpu_module]`: the offsets are the `#[repr(C)]` ones and no byte
//! of the struct may be padding (GUST-E0208), because `GpuPod` is the padding-free subset; the message names the field
//! to add and where. The refusals that are new carry GUST-E1020 (the shape) and GUST-E1021 (a field type); a 3-wide
//! field, a tuple and implicit padding keep the codes the module validator gives them (GUST-E0204, GUST-E0212,
//! GUST-E0208), so one mistake has one code on either route.
//!
//! The expansion is `#[gpu_module]`'s own `GpuPod` impl and assertions (`expand::pod::gpu_pod_impl`, storage rules: the
//! offset of a field is the sum of the sizes before it and the size is the sum of all, whatever the field types are,
//! so no padding can hide behind a type alias), a per-field check that the type IS the canonical type its name stands
//! for (`type UVec2 = u32;` is refused by rustc at the field), assertions of the offsets, size and alignment against the
//! model the derive computed from the names, and `impl ::gust::GpuStruct`, the marker a batch asks for. The host half of a batch element is the blanket `impl<T: GpuPod> Element for T`, so nothing else is written.
//!
//! It also hands the struct to a batch stage, by the protocol of `gpu_enum.rs` (D270): a hidden `macro_rules!
//! __gust_struct_<Name>`, re-exported under the struct's own name, appends the struct's definition to `acc` and calls
//! the next macro of the chain, or the continuation. The definition is the name and the fields in source order, with
//! no `#[repr]` and no derive: the module that receives it adds `#[repr(C)]`, `Clone` and `Copy` itself and refuses a
//! hand-written one (GUST-E0106, GUST-E0107). A proc macro cannot see a type declared elsewhere, so this is the only
//! way the stage's module can declare its private copy. Same crate only.

use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote, quote_spanned};
use syn::ext::IdentExt as _;
use syn::spanned::Spanned as _;

use crate::diagnostics::{self, Code};
use crate::math_types;

/// The derive: the impls, or the errors and nothing else (the struct itself is the user's own tokens; a derive does not
/// return them, so it cannot repeat them beside an error).
pub(crate) fn derive(input: TokenStream) -> TokenStream {
    match expand(input) {
        Ok(tokens) => tokens,
        Err(error) => error.into_compile_error(),
    }
}

fn expand(input: TokenStream) -> syn::Result<TokenStream> {
    let item = diagnostics::parse2::<syn::DeriveInput>(Code::GpuStructShape, input)?;
    let syn::Data::Struct(data) = &item.data else {
        return Err(diagnostics::error(
            Code::GpuStructShape,
            &item.ident,
            "`#[derive(GpuStruct)]` goes on a struct: a GPU struct is a row of scalar, vector and matrix fields",
        ));
    };
    let (fields, model) = check(&item, data)?;
    emit(&item.ident, fields, &model)
}

/// The layout of a struct as the derive computes it from the names of its field types (`#[repr(C)]`, std430 sizes and
/// alignments), which the expansion asserts against rustc's.
#[derive(Debug, PartialEq, Eq)]
struct Model {
    /// The byte offset of each field, in order.
    offsets: Vec<u32>,
    /// The size, a multiple of `alignment`.
    size: u32,
    /// The largest alignment of a field.
    alignment: u32,
}

/// Checks the shape and returns the layout. Every refusal found is reported, each at its own token; the padding is
/// checked only when every field has a layout, because the gap after a refused field has no meaning.
fn check<'a>(
    item: &syn::DeriveInput,
    data: &'a syn::DataStruct,
) -> syn::Result<(&'a syn::FieldsNamed, Model)> {
    let mut errors = diagnostics::Collector::default();
    // Source order: the attributes come before the generics.
    check_repr(item, &mut errors);
    if !item.generics.params.is_empty() || item.generics.where_clause.is_some() {
        errors.reject(
            Code::GpuStructShape,
            &item.generics,
            "generic structs are not supported: a GPU struct is a row of scalar, vector and matrix fields",
        );
    }
    let syn::Fields::Named(fields) = &data.fields else {
        errors.reject(
            Code::GpuStructShape,
            &item.ident,
            "a GPU struct has named fields (`struct Row { x: f32 }`): a tuple or unit struct has no field names for a stage to read",
        );
        return errors
            .finish()
            .map(|()| unreachable!("an error was collected"));
    };
    if fields.named.is_empty() {
        errors.reject(
            Code::GpuStructShape,
            &item.ident,
            "a GPU struct needs at least one field: a buffer element has a size",
        );
    }
    let mut layouts = Vec::new();
    for field in &fields.named {
        match field_layout(&field.ty) {
            Ok(layout) => layouts.push(layout),
            Err(error) => errors.push(error),
        }
    }
    errors.finish()?;
    let mut errors = diagnostics::Collector::default();
    let model = layout(&item.ident, fields, &layouts, &mut errors);
    errors.finish().map(|()| (fields, model))
}

/// `#[repr(C)]` is written, and is the only representation. At the struct's name when it is missing, at the attribute
/// when it is another one (`#[repr(C, align(16))]`, `#[repr(transparent)]`, `#[repr(u32)]` and a list are refused).
fn check_repr(item: &syn::DeriveInput, errors: &mut diagnostics::Collector) {
    let mut any = false;
    for attribute in &item.attrs {
        if !attribute.path().is_ident("repr") {
            continue;
        }
        any = true;
        match diagnostics::parse_args_with(
            attribute,
            Code::GpuStructShape,
            syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
        ) {
            Ok(reprs) if reprs.len() == 1 && reprs[0].path().is_ident("C") => {
                if !matches!(reprs[0], syn::Meta::Path(_)) {
                    errors.reject(
                        Code::GpuStructShape,
                        attribute,
                        "a GPU struct is `#[repr(C)]` and nothing else",
                    );
                }
            }
            Ok(_) => errors.reject(
                Code::GpuStructShape,
                attribute,
                "a GPU struct is `#[repr(C)]` and nothing else: a layout of its own (`align(16)`, `packed`, `transparent`) is not the std430 one",
            ),
            Err(error) => errors.push(error),
        }
    }
    if !any {
        errors.reject(
            Code::GpuStructShape,
            &item.ident,
            "a GPU struct is written `#[repr(C)]`: the derive checks it and does not add it",
        );
    }
}

/// The size and alignment of a field type, or the refusal at the type. The accepted types are the ones
/// `math_types::known_layout` knows that a buffer may hold, written by their bare Rust names.
fn field_layout(ty: &syn::Type) -> syn::Result<(u32, u32)> {
    let refuse = |message: &str| Err(diagnostics::error(Code::GpuStructField, ty, message));
    match ty {
        syn::Type::Tuple(tuple) if !tuple.elems.is_empty() => Err(diagnostics::error(
            Code::TupleInResource,
            ty,
            "a tuple is not a field of a GPU struct: Rust gives a tuple no layout; store its elements as fields of their own",
        )),
        syn::Type::Array(_) => refuse(
            "an array is not a field of a GPU struct yet: write the elements as fields of their own",
        ),
        syn::Type::Path(path) if path.qself.is_none() && path.path.leading_colon.is_none() => {
            let Some(ident) = path.path.get_ident() else {
                return refuse(
                    "a field type is written by its bare name (`f32`, `Vec2`): a path or a generic type is not a GPU field type",
                );
            };
            let name = ident.unraw().to_string();
            if let Some(math) = math_types::math_type(&name)
                && !math.buffer
            {
                return Err(diagnostics::error(
                    Code::ThreeWideInBuffer,
                    ty,
                    format!(
                        "`{name}` is not a field of a GPU struct: Rust's {}-byte, {}-aligned layout has no std430 match; use `Vec4` (`UVec4`, `Mat4`) or separate scalar fields (`f32`, `u32`, three of them)",
                        math.size, math.alignment
                    ),
                ));
            }
            let accepted = matches!(name.as_str(), "f32" | "u32" | "i32")
                || math_types::math_type(&name).is_some();
            if let Some(layout) = math_types::known_layout(&name).filter(|_| accepted) {
                return Ok(layout);
            }
            refuse(&format!(
                "`{name}` is not a field type of a GPU struct: a field is `f32`, `u32`, `i32`, a vector (`Vec2`, `Vec4`, `IVec2`, `IVec4`, `UVec2`, `UVec4`) or a matrix (`Mat2`, `Mat4`); a struct, an enum, a `bool` and any other type are not supported yet (a flag is a `u32` bit in a flags word)"
            ))
        }
        _ => refuse(
            "a field type is a scalar, a vector or a matrix written by its bare name (`f32`, `Vec2`): not a reference, a pointer or a unit",
        ),
    }
}

/// "1 `u32` or `f32` pad field" and "3 `u32` or `f32` pad fields", for a gap of `count` four-byte fields.
fn pad_fields(count: u64) -> String {
    let plural = if count == 1 { "" } else { "s" };
    format!("{count} `u32` or `f32` pad field{plural}")
}

/// The offsets, the size and the alignment, refusing implicit padding (GUST-E0208) at the field that sits late or at
/// the struct for a tail, with the number of 4-byte fields to add and where.
fn layout(
    name: &syn::Ident,
    fields: &syn::FieldsNamed,
    layouts: &[(u32, u32)],
    errors: &mut diagnostics::Collector,
) -> Model {
    let mut offsets = Vec::new();
    let mut alignment = 1_u32;
    let mut end = 0_u64;
    let mut previous: Option<&syn::Ident> = None;
    for (field, &(size, field_alignment)) in fields.named.iter().zip(layouts) {
        let ident = field.ident.as_ref().expect("named field");
        let align = u64::from(field_alignment);
        let offset = end.next_multiple_of(align);
        if offset != end {
            let gap = offset - end;
            // A gap needs a field before it: the first field sits at 0.
            let place =
                previous.map_or_else(String::new, |previous| format!("after `{previous}`, "));
            errors.reject(
                Code::ImplicitPadding,
                field,
                format!(
                    "field `{ident}` of `{name}` would sit at byte {offset}, {gap} bytes after the previous field ends at {end}: GPU structs may not contain implicit padding; add {} (`_pad0` ...) {place}before `{ident}`, or order the fields by decreasing alignment",
                    pad_fields(gap / 4)
                ),
            );
        }
        offsets.push(u32::try_from(offset).unwrap_or(u32::MAX));
        end = offset + u64::from(size);
        alignment = alignment.max(field_alignment);
        previous = Some(ident);
    }
    let size = end.next_multiple_of(u64::from(alignment));
    if size != end {
        let last = previous.map_or_else(String::new, ToString::to_string);
        errors.reject(
            Code::ImplicitPadding,
            name,
            format!(
                "`{name}` ends at byte {end} but its size is {size}: GPU structs may not contain implicit tail padding; add {} (`_pad0` ...) after the last field `{last}`",
                pad_fields((size - end) / 4)
            ),
        );
    }
    if size > u64::from(u32::MAX) {
        errors.reject(
            Code::GpuStructShape,
            name,
            "a GPU struct is smaller than 4 GiB",
        );
    }
    Model {
        offsets,
        size: u32::try_from(size).unwrap_or(u32::MAX),
        alignment,
    }
}

/// The struct as a stage's module reads it: the name and the fields in source order, nothing else. The stage adds
/// `#[repr(C)]`, `Clone` and `Copy`; a field type is the bare name the derive accepted, so the module must have it in
/// scope (it is the content of the stage's hash: a field added, retyped or reordered changes it).
fn definition(name: &syn::Ident, fields: &syn::FieldsNamed) -> TokenStream {
    let idents = fields.named.iter().map(|field| &field.ident);
    let types = fields.named.iter().map(|field| &field.ty);
    quote! {
        struct #name { #(#idents: #types),* }
    }
}

/// The canonical type a field name stands for: the primitive or the `gust::math` type itself.
pub(crate) fn canonical(name: &str) -> TokenStream {
    let ident = format_ident!("{}", name);
    if matches!(name, "f32" | "u32" | "i32") {
        quote!(::core::primitive::#ident)
    } else {
        quote!(::gust::math::#ident)
    }
}

/// The impls, the layout assertions and the callback macro.
///
/// The `GpuPod` impl and its assertions are `#[gpu_module]`'s own (`expand::pod::gpu_pod_impl`, storage rules): the offset
/// of a field is the sum of the `size_of` of the fields before it and the size is the sum of all, so no padding can hide
/// whatever the field types are. The derive adds two things on top. Each field's type must BE the canonical type its
/// name stands for (`type UVec2 = u32;` would make the model and the real layout disagree, and in a stage the copy would
/// read the name as the `gust::math` type), and the offsets, size and alignment must equal the model the derive computed
/// from the names, which makes a disagreement an error that names the struct.
fn emit(name: &syn::Ident, fields: &syn::FieldsNamed, model: &Model) -> syn::Result<TokenStream> {
    let idents: Vec<&syn::Ident> = fields
        .named
        .iter()
        .filter_map(|field| field.ident.as_ref())
        .collect();
    let identity = fields.named.iter().map(|field| {
        let ty = &field.ty;
        let canonical = match ty {
            syn::Type::Path(path) => path
                .path
                .get_ident()
                .map(|ident| canonical(&ident.unraw().to_string())),
            _ => None,
        };
        // At the field's own type, so that rustc's mismatch points there.
        quote_spanned! { ty.span() =>
            const _: fn(#ty) -> #canonical = |value| value;
        }
    });
    let struct_name = name.unraw().to_string();
    let offset_assertions = idents.iter().zip(&model.offsets).map(|(ident, offset)| {
        let offset = Literal::usize_unsuffixed(*offset as usize);
        let message = format!(
            "GUST StorageV2: field `{}` of `{struct_name}` is not at its std430 offset",
            ident.unraw()
        );
        quote! {
            ::core::assert!(::core::mem::offset_of!(#name, #ident) == #offset, #message);
        }
    });
    let size = Literal::usize_unsuffixed(model.size as usize);
    let alignment = Literal::usize_unsuffixed(model.alignment as usize);
    let size_message =
        format!("GUST StorageV2: `{struct_name}` is not the size of its std430 layout");
    let alignment_message =
        format!("GUST StorageV2: `{struct_name}` must have the largest alignment of its fields");
    let item = syn::ItemStruct {
        attrs: Vec::new(),
        vis: syn::Visibility::Inherited,
        struct_token: syn::Token![struct](name.span()),
        ident: name.clone(),
        generics: syn::Generics::default(),
        fields: syn::Fields::Named(fields.clone()),
        semi_token: None,
    };
    let pod = crate::expand::pod::gpu_pod_impl(&item, &crate::expand::pod::PodRules::Storage)?;
    let definition = definition(name, fields);
    let callback = format_ident!("__gust_struct_{}", name.unraw());
    Ok(quote! {
        #pod

        #[automatically_derived]
        impl ::gust::__GpuStructSealed for #name {}

        #[automatically_derived]
        impl ::gust::__GpuElementKind for #name {}

        #[automatically_derived]
        impl ::gust::GpuStruct for #name {}

        #(#identity)*

        const _: () = {
            ::core::assert!(::core::mem::size_of::<#name>() == #size, #size_message);
            ::core::assert!(::core::mem::align_of::<#name>() == #alignment, #alignment_message);
            #(#offset_assertions)*
        };

        #[allow(unused_macros)]
        macro_rules! #callback {
            (
                acc = [ $($acc:tt)* ] ;
                next = [ { $($head:tt)+ } $($tail:tt)* ] ;
                cont = { $($cont:tt)+ } ;
                orig = { $($orig:tt)* }
            ) => {
                $($head)+ ! {
                    acc = [ $($acc)* { #definition } ] ;
                    next = [ $($tail)* ] ;
                    cont = { $($cont)+ } ;
                    orig = { $($orig)* }
                }
            };
            (
                acc = [ $($acc:tt)* ] ;
                next = [ ] ;
                cont = { $($cont:tt)+ } ;
                orig = { $($orig:tt)* }
            ) => {
                $($cont)+ ! {
                    // The `@enums` tag carries structs too: the receiver tells an enum from a struct by the item.
                    @enums [ $($acc)* { #definition } ] ;
                    $($orig)*
                }
            };
        }

        #[allow(unused_imports)]
        pub(crate) use #callback as #name;
    })
}

#[cfg(test)]
mod tests {
    use super::{Model, check, derive, expand};
    use crate::diagnostics::{
        Code,
        testing::{assert_code, assert_codes, parse_str, parse2},
    };

    const HEAD: &str = "#[repr(C)]";

    fn expansion(source: &str) -> syn::Result<proc_macro2::TokenStream> {
        expand(source.parse().unwrap())
    }

    fn model(source: &str) -> Model {
        let item: syn::DeriveInput = parse_str(source).unwrap();
        let syn::Data::Struct(data) = &item.data else {
            panic!("a struct");
        };
        check(&item, data)
            .unwrap_or_else(|error| panic!("{error}: {source}"))
            .1
    }

    /// The one refusal of `source`: one error, with the code, and its text.
    fn refused(source: &str, code: Code) -> String {
        let error = expansion(source).unwrap_err();
        assert_code(&error, code);
        assert_eq!(error.clone().into_iter().count(), 1, "{error}");
        error.to_string()
    }

    fn assert_refused(source: &str, code: Code, needle: &str) {
        let text = refused(source, code);
        assert!(text.contains(needle), "{needle} not in {text}");
    }

    #[test]
    fn a_row_with_no_padding_has_the_std430_offsets() {
        // Layout (i) of the plan: two `Vec2`, an `f32` and a `u32`, 24 bytes.
        let model = model(&format!(
            "{HEAD} struct Row {{ a: Vec2, b: Vec2, c: f32, d: u32 }}"
        ));
        assert_eq!(
            model,
            Model {
                offsets: vec![0, 8, 16, 20],
                size: 24,
                alignment: 8
            }
        );
    }

    #[test]
    fn a_row_with_explicit_pad_fields_has_the_std430_offsets() {
        // Layout (ii): a `Vec4`, a `Vec2` and two pad `f32`, 32 bytes.
        let model = model(&format!(
            "{HEAD} struct Row {{ a: Vec4, b: Vec2, _pad0: f32, _pad1: f32 }}"
        ));
        assert_eq!(
            model,
            Model {
                offsets: vec![0, 16, 24, 28],
                size: 32,
                alignment: 16
            }
        );
    }

    #[test]
    fn a_matrix_row_has_the_std430_offsets() {
        // Layout (iii): a `Mat4` and scalars up to a multiple of 16.
        let model = model(&format!(
            "{HEAD} struct Row {{ m: Mat4, a: f32, b: u32, c: i32, _pad: u32 }}"
        ));
        assert_eq!(
            model,
            Model {
                offsets: vec![0, 64, 68, 72, 76],
                size: 80,
                alignment: 16
            }
        );
    }

    #[test]
    fn every_accepted_field_type_has_its_std430_size_and_alignment() {
        // Each type once, ordered so that only the three pad fields before `UVec4` are needed.
        let model = model(&format!(
            "{HEAD} struct All {{ a: Vec4, b: IVec4, c: UVec2, d: IVec2, e: Vec2, f: Mat2, g: f32, h: u32, i: i32, _pad0: u32, _pad1: u32, _pad2: u32, j: UVec4 }}"
        ));
        assert_eq!(
            model.offsets,
            [0, 16, 32, 40, 48, 56, 72, 76, 80, 84, 88, 92, 96]
        );
        assert_eq!((model.size, model.alignment), (112, 16));
    }

    #[test]
    fn the_impls_and_the_layout_assertions_are_written() {
        let output = expansion(&format!(
            "{HEAD} struct Row {{ a: Vec2, b: Vec2, c: f32, d: u32 }}"
        ))
        .unwrap();
        parse2::<syn::File>(output.clone()).unwrap();
        let text = output.to_string();
        for expected in [
            "unsafe impl :: gust :: GpuPod for Row",
            "impl :: gust :: GpuStruct for Row",
            "offset_of ! (Row , a) as u32",
            "offset_of ! (Row , d) as u32",
            "size_of :: < Row > () == 24",
            "align_of :: < Row > () == 8",
            "offset_of ! (Row , c) == 16",
            "offset_of ! (Row , d) == 20",
            "is_storage_v2 ()",
            // The per-field identity check against the canonical type.
            "const _ : fn (Vec2) -> :: gust :: math :: Vec2 = | value | value",
            "const _ : fn (f32) -> :: core :: primitive :: f32 = | value | value",
            "const _ : fn (u32) -> :: core :: primitive :: u32 = | value | value",
            "impl :: gust :: __GpuStructSealed for Row",
            // Only `gpu_pod_impl` writes these: the name-independent proof (offsets and size as sums of `size_of`).
            "has tail padding",
            "== 0 + :: core :: mem :: size_of :: < Vec2 > ()",
        ] {
            assert!(text.contains(expected), "{expected} not in {text}");
        }
        // The names the layout reports are the Slang names, in source order.
        let first = text.find("name : \"a\"").unwrap();
        let last = text.find("name : \"d\"").unwrap();
        assert!(first < last, "{text}");
    }

    #[test]
    fn the_derive_adds_the_hidden_callback_macro_and_its_alias_for_a_batch() {
        let text = expansion(&format!("{HEAD} struct Row {{ a: f32, b: u32 }}"))
            .unwrap()
            .to_string();
        assert!(text.contains("macro_rules ! __gust_struct_Row"), "{text}");
        assert!(
            text.contains("pub (crate) use __gust_struct_Row as Row"),
            "{text}"
        );
        // The definition handed on is the name and the fields only: the module adds the representation and the derives.
        assert!(text.contains("struct Row { a : f32 , b : u32 }"), "{text}");
        assert!(!text.contains("# [repr"), "{text}");
        assert!(!text.contains("# [derive"), "{text}");
        // The same protocol as the enum's: acc, next, cont, orig and the `@enums` hand-over.
        for expected in ["acc = [", "next = [", "cont = {", "orig = {", "@ enums"] {
            assert!(text.contains(expected), "{expected} not in {text}");
        }
    }

    /// The definition a stage hashes: it changes with a field added, retyped or reordered, and with nothing else.
    #[test]
    fn the_definition_a_stage_hashes_follows_the_fields() {
        let definition = |source: &str| {
            let item: syn::DeriveInput = parse_str(source).unwrap();
            let syn::Data::Struct(syn::DataStruct {
                fields: syn::Fields::Named(fields),
                ..
            }) = &item.data
            else {
                panic!("a struct with named fields");
            };
            super::definition(&item.ident, fields).to_string()
        };
        let base = definition(&format!("{HEAD} struct Row {{ a: f32, b: u32 }}"));
        assert_eq!(
            base,
            definition(&format!(
                "{HEAD} /// docs\n struct Row {{ a: f32, b: u32 }}"
            )),
            "attributes and documentation are not part of it"
        );
        for changed in [
            "struct Row { a: f32, b: u32, c: u32, d: u32 }",
            "struct Row { a: f32, b: i32 }",
            "struct Row { b: u32, a: f32 }",
            "struct Row { a: f32, c: u32 }",
        ] {
            assert_ne!(base, definition(&format!("{HEAD} {changed}")), "{changed}");
        }
    }

    #[test]
    fn a_refused_struct_gets_the_error_and_no_impl() {
        let text = derive("struct Row { a: f32 }".parse().unwrap()).to_string();
        assert!(text.contains("compile_error"), "{text}");
        assert!(!text.contains("impl"), "{text}");
    }

    #[test]
    fn a_missing_or_other_repr_is_refused() {
        assert_refused("struct Row { a: f32 }", Code::GpuStructShape, "is written");
        for repr in [
            "u32",
            "transparent",
            "C, align(16)",
            "C, packed",
            "align(16)",
            "C(1)",
        ] {
            assert_refused(
                &format!("#[repr({repr})] struct Row {{ a: f32 }}"),
                Code::GpuStructShape,
                "`#[repr(C)]` and nothing else",
            );
        }
    }

    #[test]
    fn generics_are_refused() {
        assert_refused(
            &format!("{HEAD} struct Row<T> {{ a: f32 }}"),
            Code::GpuStructShape,
            "generic structs",
        );
        assert_refused(
            &format!("{HEAD} struct Row where u32: Copy {{ a: f32 }}"),
            Code::GpuStructShape,
            "generic structs",
        );
    }

    #[test]
    fn a_tuple_unit_or_empty_struct_and_an_enum_are_refused() {
        assert_refused(
            &format!("{HEAD} struct Row(f32);"),
            Code::GpuStructShape,
            "named fields",
        );
        assert_refused(
            &format!("{HEAD} struct Row;"),
            Code::GpuStructShape,
            "named fields",
        );
        assert_refused(
            &format!("{HEAD} struct Row {{}}"),
            Code::GpuStructShape,
            "at least one field",
        );
        assert_refused(
            &format!("{HEAD} enum Row {{ A }}"),
            Code::GpuStructShape,
            "goes on a struct",
        );
    }

    #[test]
    fn a_three_wide_field_is_refused_with_the_existing_code() {
        for ty in ["Vec3", "UVec3", "Mat3"] {
            assert_refused(
                &format!("{HEAD} struct Row {{ a: f32, b: {ty} }}"),
                Code::ThreeWideInBuffer,
                &format!("`{ty}` is not a field of a GPU struct"),
            );
        }
    }

    #[test]
    fn a_tuple_field_is_refused_with_the_existing_code() {
        assert_refused(
            &format!("{HEAD} struct Row {{ a: (u32, u32) }}"),
            Code::TupleInResource,
            "a tuple is not a field",
        );
    }

    #[test]
    fn an_array_nested_struct_enum_or_other_type_is_refused() {
        assert_refused(
            &format!("{HEAD} struct Row {{ a: [f32; 4] }}"),
            Code::GpuStructField,
            "an array is not a field",
        );
        for ty in ["Inner", "Mode", "bool", "usize", "f64", "u8", "float"] {
            assert_refused(
                &format!("{HEAD} struct Row {{ a: {ty} }}"),
                Code::GpuStructField,
                &format!("`{ty}` is not a field type"),
            );
        }
        for ty in [
            "gust::prelude::Vec2",
            "::core::primitive::u32",
            "Option<u32>",
            "&'static u32",
            "*const u32",
            "()",
        ] {
            refused(
                &format!("{HEAD} struct Row {{ a: {ty} }}"),
                Code::GpuStructField,
            );
        }
    }

    #[test]
    fn implicit_padding_is_refused_with_the_existing_code_and_names_the_field_to_add() {
        // A gap before `b`: 4 bytes after `a`.
        let text = refused(
            &format!("{HEAD} struct Row {{ a: f32, b: Vec2 }}"),
            Code::ImplicitPadding,
        );
        for needle in [
            "field `b` of `Row` would sit at byte 8",
            "4 bytes after the previous field ends at 4",
            "add 1 `u32` or `f32` pad field",
            "after `a`, before `b`",
        ] {
            assert!(text.contains(needle), "{needle} not in {text}");
        }
        // A gap at the start of the struct cannot exist, so a gap names the field before it; a tail names the last.
        let tail = refused(
            &format!("{HEAD} struct Row {{ a: Vec2, b: f32 }}"),
            Code::ImplicitPadding,
        );
        for needle in [
            "`Row` ends at byte 12 but its size is 16",
            "add 1 `u32` or `f32` pad field",
            "after the last field `b`",
        ] {
            assert!(tail.contains(needle), "{needle} not in {tail}");
        }
        let wide = refused(
            &format!("{HEAD} struct Row {{ a: Vec4, b: f32 }}"),
            Code::ImplicitPadding,
        );
        assert!(wide.contains("add 3 `u32` or `f32` pad fields"), "{wide}");
    }

    #[test]
    fn every_offending_token_is_reported_once_in_source_order() {
        let error = expansion(
            "#[repr(u32)] struct Row<T> { a: Vec3, b: [f32; 2], c: (u32, u32), d: bool }",
        )
        .unwrap_err();
        assert_codes(
            &error,
            &[
                Code::GpuStructShape,
                Code::GpuStructShape,
                Code::ThreeWideInBuffer,
                Code::GpuStructField,
                Code::TupleInResource,
                Code::GpuStructField,
            ],
        );
    }

    #[test]
    fn padding_is_not_reported_beside_a_refused_field() {
        let error =
            expansion(&format!("{HEAD} struct Row {{ a: f32, b: bool, c: Vec2 }}")).unwrap_err();
        assert_codes(&error, &[Code::GpuStructField]);
    }
}
