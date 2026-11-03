//! Element layouts read from the Slang helper's reflection JSON, for the parameters whose layout the macro could not
//! write into the record (`layout: null`). A storage parameter is listed as `export_unchecked` in the manifest; a
//! uniform is not (see the last paragraph).
//!
//! The macro describes an element type from the module's own syntax: scalars and structs of this module. A `GpuPod`
//! type from outside the module, a type alias or a nested type from elsewhere is invisible to it, so the record says
//! `layout: null` and the layout comes from here. That layout is the compiler's own answer, so comparing it with the
//! compiler's answer proves nothing about the Rust type: the descriptor-side half of D17 is skipped for those
//! parameters at export time, and the manifest says so. D17 still holds at run time, because `verify_artifact` checks
//! the real `GpuPod::LAYOUT` of the program that loads the artifact.
//!
//! The fill is not trusted blindly: the completed descriptor goes through the same cross-check as any other, which
//! parses the same JSON with `gust`'s own reader, so a fill that disagrees with that reader fails the export, and so
//! does a reflected layout that is not StorageV1 (padding, alignment other than 4).
//!
//! A uniform (`Uniform<T>`, a constant buffer; D80) is always in this state: the record schema has no uniform layout,
//! so its `layout` is `null` whatever `T` is. [`reflected_uniform_layout`] reads the constant buffer's element, and the
//! export declares it `Std140V1`, so the same cross-check holds it to std140 (padding fields spelled out, nested
//! structs at multiples of 16) and to the reflection it came from. It is not listed as `export_unchecked`, and the
//! macro's own layout of a uniform is compared with the reflection only at run time (D17, `verify_artifact`).

use gust::ScalarKind;

use crate::{
    json::{self, Json},
    record::{FieldRecord, LayoutKindRecord, LayoutRecord, ResourceRecord},
};

/// The element layout the reflection reports for the storage parameter `parameter`.
pub(crate) fn reflected_layout(
    reflection_json: &str,
    parameter: &str,
) -> Result<ResourceRecord, String> {
    let root = json::parse(reflection_json).map_err(|error| format!("reflection JSON: {error}"))?;
    let resource = resource_named(&root, parameter)?;
    let context = format!("reflection parameter `{parameter}`");
    let element_stride = resource
        .field("elementStride", &context)?
        .as_u32()
        .ok_or_else(|| format!("{context}: `elementStride` must be a non-negative integer"))?;
    let element = resource
        .field("type", &context)?
        .field("resultType", &context)?;
    Ok(ResourceRecord {
        element_stride,
        element: layout(element, &format!("{context}.resultType"))?,
    })
}

/// The element layout the reflection reports for the uniform parameter `parameter`, a constant buffer (D80).
///
/// A constant buffer has no array stride: the document's `size` of the buffer is its element's size, and that is what a
/// `Std140V1` [`ResourceRecord`] keeps in `element_stride`. The caller declares the result `Std140V1` and the
/// cross-check judges it: the nested sizes and offsets must obey std140, and `element_stride` must equal the size of
/// the element, so a reflection whose `size` and `elementType.size` disagree fails there.
pub(crate) fn reflected_uniform_layout(
    reflection_json: &str,
    parameter: &str,
) -> Result<ResourceRecord, String> {
    let root = json::parse(reflection_json).map_err(|error| format!("reflection JSON: {error}"))?;
    let resource = resource_named(&root, parameter)?;
    let context = format!("reflection parameter `{parameter}`");
    let buffer = resource.field("type", &context)?;
    let kind = buffer
        .field("kind", &format!("{context}.type"))?
        .as_str()
        .ok_or_else(|| format!("{context}.type: `kind` must be a string"))?;
    if kind != "constantBuffer" {
        return Err(format!(
            "{context}: the macro says uniform, the reflection reports `{kind}`, not a constantBuffer"
        ));
    }
    let element = buffer.field("elementType", &context)?;
    Ok(ResourceRecord {
        element_stride: number(buffer, "size", &format!("{context}.type"))?,
        element: layout(element, &format!("{context}.elementType"))?,
    })
}

fn resource_named<'a>(root: &'a Json, parameter: &str) -> Result<&'a Json, String> {
    let resources = root
        .field("parameters", "reflection")?
        .as_array()
        .ok_or("reflection: `parameters` must be an array")?;
    resources
        .iter()
        .find(|resource| resource.get("name").and_then(Json::as_str) == Some(parameter))
        .ok_or_else(|| {
            format!(
                "the reflection reports no resource named `{parameter}` to take its layout from"
            )
        })
}

fn number(value: &Json, key: &str, context: &str) -> Result<u32, String> {
    value
        .field(key, context)?
        .as_u32()
        .ok_or_else(|| format!("{context}: `{key}` must be a non-negative integer"))
}

fn layout(value: &Json, context: &str) -> Result<LayoutRecord, String> {
    let kind = value
        .field("kind", context)?
        .as_str()
        .ok_or_else(|| format!("{context}: `kind` must be a string"))?;
    let (name, kind) = match kind {
        "scalar" => {
            let scalar = match value.field("scalarType", context)?.as_str() {
                Some("float32") => ScalarKind::F32,
                Some("int32") => ScalarKind::I32,
                Some("uint32") => ScalarKind::U32,
                _ => {
                    return Err(format!(
                        "{context}: unsupported scalar type (only 32-bit float, int and uint)"
                    ));
                }
            };
            let name = match scalar {
                ScalarKind::F32 => "f32",
                ScalarKind::I32 => "i32",
                ScalarKind::U32 => "u32",
            };
            (name.to_owned(), LayoutKindRecord::Scalar(scalar))
        }
        "struct" => {
            let fields = value
                .field("fields", context)?
                .as_array()
                .ok_or_else(|| format!("{context}: `fields` must be an array"))?
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    let at = format!("{context}.fields[{index}]");
                    let name = field
                        .field("name", &at)?
                        .as_str()
                        .ok_or_else(|| format!("{at}: `name` must be a string"))?
                        .to_owned();
                    Ok(FieldRecord {
                        offset: number(
                            field.field("binding", &at)?,
                            "offset",
                            &format!("{at}.binding"),
                        )?,
                        ty: layout(field.field("type", &at)?, &format!("{at}.type"))?,
                        name,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let name = value
                .get("name")
                .and_then(Json::as_str)
                .unwrap_or("struct")
                .to_owned();
            (name, LayoutKindRecord::Struct(fields))
        }
        other => {
            return Err(format!(
                "{context}: unsupported element type `{other}` (only 32-bit scalars and structs of them)"
            ));
        }
    };
    Ok(LayoutRecord {
        name,
        size: number(value, "size", context)?,
        alignment: number(value, "alignment", context)?,
        stride: number(value, "stride", context)?,
        kind,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata::{PAIRS_REFLECTION, VECTOR_ADD_REFLECTION};

    /// The helper's reflection of `scale` (one uniform `Gain`) and `shifted` (uniforms `Outer`, which nests `Inner`,
    /// and `Gain`), as `cargo gust build` wrote it (see `export_uniform_tests.rs`).
    const GAIN_REFLECTION: &str =
        include_str!("testdata/uniform/bfb6e436140611c0.wgsl.reflection.json");
    const OUTER_REFLECTION: &str =
        include_str!("testdata/uniform/f6dd8e3fe6e29694.wgsl.reflection.json");

    #[test]
    fn a_scalar_element_is_read_with_the_canonical_name() {
        let layout = reflected_layout(VECTOR_ADD_REFLECTION, "out").unwrap();
        assert_eq!(layout.element_stride, 4);
        assert_eq!(
            layout.element,
            LayoutRecord {
                name: "f32".to_owned(),
                size: 4,
                alignment: 4,
                stride: 4,
                kind: LayoutKindRecord::Scalar(ScalarKind::F32),
            }
        );
    }

    #[test]
    fn a_struct_element_is_read_with_its_field_offsets_and_nested_types() {
        let layout = reflected_layout(PAIRS_REFLECTION, "pairs").unwrap();
        assert_eq!(layout.element_stride, 8);
        let element = layout.element;
        assert_eq!(
            (element.name.as_str(), element.size, element.stride),
            ("Pair", 8, 8)
        );
        let LayoutKindRecord::Struct(fields) = element.kind else {
            panic!("expected a struct");
        };
        let seen: Vec<(&str, u32, &str)> = fields
            .iter()
            .map(|field| (field.name.as_str(), field.offset, field.ty.name.as_str()))
            .collect();
        assert_eq!(seen, [("a", 0, "u32"), ("b", 4, "f32")]);
    }

    #[test]
    fn what_the_reflection_cannot_give_is_an_error_that_names_the_parameter() {
        let missing = reflected_layout(VECTOR_ADD_REFLECTION, "nope").unwrap_err();
        assert!(missing.contains("no resource named `nope`"), "{missing}");
        for (what, bad, expected) in [
            ("not json", "{".to_owned(), "reflection JSON"),
            (
                "an unsupported scalar",
                PAIRS_REFLECTION.replace("float32", "float64"),
                "unsupported scalar type",
            ),
            (
                "a vector element",
                PAIRS_REFLECTION.replace("\"kind\":\"struct\"", "\"kind\":\"vector\""),
                "unsupported element type `vector`",
            ),
            (
                "no stride",
                PAIRS_REFLECTION.replace("\"elementStride\":8,", ""),
                "elementStride",
            ),
            (
                "no field offset",
                PAIRS_REFLECTION.replace("\"offset\":0,", ""),
                "offset",
            ),
        ] {
            let error = reflected_layout(&bad, "pairs").unwrap_err();
            assert!(error.contains(expected), "{what}: {error}");
        }
    }

    fn seen(layout: &LayoutRecord) -> Vec<(&str, u32, &str)> {
        let LayoutKindRecord::Struct(fields) = &layout.kind else {
            panic!("expected a struct");
        };
        fields
            .iter()
            .map(|field| (field.name.as_str(), field.offset, field.ty.name.as_str()))
            .collect()
    }

    #[test]
    fn a_uniform_struct_is_read_from_the_constant_buffer_with_the_element_size_as_its_stride() {
        let layout = reflected_uniform_layout(GAIN_REFLECTION, "g").unwrap();
        assert_eq!(layout.element_stride, 16);
        let element = &layout.element;
        assert_eq!(
            (
                element.name.as_str(),
                element.size,
                element.alignment,
                element.stride
            ),
            ("Gain", 16, 16, 16)
        );
        assert_eq!(
            seen(element),
            [
                ("gain", 0, "f32"),
                ("pad0", 4, "f32"),
                ("pad1", 8, "f32"),
                ("pad2", 12, "f32")
            ]
        );
    }

    #[test]
    fn a_nested_uniform_struct_keeps_its_offsets_and_its_rounded_up_nested_size() {
        let outer = reflected_uniform_layout(OUTER_REFLECTION, "o").unwrap();
        assert_eq!(outer.element_stride, 32);
        assert_eq!(
            (
                outer.element.name.as_str(),
                outer.element.size,
                outer.element.alignment
            ),
            ("Outer", 32, 16)
        );
        assert_eq!(
            seen(&outer.element),
            [
                ("scale", 0, "f32"),
                ("pad0", 4, "f32"),
                ("pad1", 8, "f32"),
                ("pad2", 12, "f32"),
                ("inner", 16, "Inner"),
            ]
        );
        let LayoutKindRecord::Struct(fields) = &outer.element.kind else {
            panic!("expected a struct");
        };
        let inner = &fields[4].ty;
        assert_eq!((inner.size, inner.alignment, inner.stride), (16, 16, 16));
        assert_eq!(
            seen(inner),
            [
                ("count", 0, "u32"),
                ("bias", 4, "i32"),
                ("pad0", 8, "f32"),
                ("pad1", 12, "f32")
            ]
        );
        // The second uniform of the same document is found by name.
        assert_eq!(
            reflected_uniform_layout(OUTER_REFLECTION, "g")
                .unwrap()
                .element_stride,
            16
        );
    }

    #[test]
    fn what_a_uniform_cannot_take_from_the_reflection_is_an_error_that_names_the_parameter() {
        let missing = reflected_uniform_layout(GAIN_REFLECTION, "nope").unwrap_err();
        assert!(missing.contains("no resource named `nope`"), "{missing}");
        // A storage buffer is not a constant buffer, and a constant buffer is not a storage buffer.
        let storage = reflected_uniform_layout(GAIN_REFLECTION, "src").unwrap_err();
        assert!(
            storage.contains("reflection parameter `src`")
                && storage.contains("`resource`, not a constantBuffer"),
            "{storage}"
        );
        assert!(
            reflected_layout(GAIN_REFLECTION, "g")
                .unwrap_err()
                .contains("elementStride")
        );
        for (what, bad, expected) in [
            ("not json", "{".to_owned(), "reflection JSON"),
            (
                "no buffer size",
                GAIN_REFLECTION.replace("\"constantBuffer\",\"size\":16,", "\"constantBuffer\","),
                "`size`",
            ),
            (
                "no element type",
                GAIN_REFLECTION.replace("\"elementType\"", "\"elementTypo\""),
                "elementType",
            ),
            (
                "a vector field",
                GAIN_REFLECTION.replacen("\"kind\":\"scalar\"", "\"kind\":\"vector\"", 1),
                "unsupported element type `vector`",
            ),
            (
                "no field offset",
                GAIN_REFLECTION.replacen("\"offset\":0,", "", 1),
                "offset",
            ),
        ] {
            let error = reflected_uniform_layout(&bad, "g").unwrap_err();
            assert!(error.contains(expected), "{what}: {error}");
        }
    }
}
