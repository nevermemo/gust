//! Element layouts read from the Slang helper's reflection JSON, for the storage and uniform parameters whose layout
//! the macro could not write into the record (`layout: null`). Such a parameter is listed as `export_unchecked` in the
//! manifest.
//!
//! The macro describes an element type from the module's own syntax: scalars, vectors and matrices of `gust::math`,
//! and structs of this module (a uniform's element is always one). A `GpuPod` type from outside the module, a type
//! alias or a nested type from elsewhere is invisible to it, so the record says `layout: null` and the layout comes
//! from here. That layout is the compiler's own answer, so comparing it with the compiler's answer proves nothing about
//! the Rust type: the descriptor-side half of D17 is skipped for those parameters at export time, and the manifest says
//! so. D17 still holds at run time, because `verify_artifact` checks the real `GpuPod::LAYOUT` of the program that
//! loads the artifact.
//!
//! The fill is not trusted blindly: the completed descriptor goes through the same cross-check as any other, which
//! parses the same JSON with `gust`'s own reader, so a fill that disagrees with that reader fails the export, and so
//! does a reflected layout that does not follow its rules. A storage element is declared `StorageV1` when it holds
//! no vector and no matrix and `StorageV2` when it does ([`reflected_layout`]); a uniform (`Uniform<T>`, a constant
//! buffer; D80) is declared `Std140V1` ([`reflected_uniform_layout`]), so the cross-check holds it to std140 (padding
//! fields spelled out, nested structs at multiples of 16) and to the reflection it came from.

use gust::{ScalarKind, ShaderLayoutRules};

use crate::{
    json::{self, Json},
    record::{FieldRecord, LayoutKindRecord, LayoutRecord, ResourceRecord},
};

/// The element layout the reflection reports for the storage parameter `parameter`, with `StorageV1` rules when the
/// element holds no vector and no matrix and `StorageV2` rules when it does (the choice the macro's descriptor makes).
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
    let element = layout(element, &format!("{context}.resultType"))?;
    Ok(ResourceRecord {
        rules: if element.holds_math_type() {
            ShaderLayoutRules::StorageV2
        } else {
            ShaderLayoutRules::StorageV1
        },
        element_stride,
        element,
    })
}

/// The element layout the reflection reports for the uniform parameter `parameter`, a constant buffer (D80).
///
/// A constant buffer has no array stride: the document's `size` of the buffer is its element's size, and that is what a
/// `Std140V1` [`ResourceRecord`] keeps in `element_stride`. The result is declared `Std140V1` and the cross-check
/// judges it: the nested sizes and offsets must obey std140, and `element_stride` must equal the size of the element,
/// so a reflection whose `size` and `elementType.size` disagree fails there.
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
        rules: ShaderLayoutRules::Std140V1,
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

/// The `scalarType` of a reflected scalar, vector or matrix.
fn scalar_type(value: &Json, context: &str) -> Result<ScalarKind, String> {
    match value.field("scalarType", context)?.as_str() {
        Some("float32") => Ok(ScalarKind::F32),
        Some("int32") => Ok(ScalarKind::I32),
        Some("uint32") => Ok(ScalarKind::U32),
        _ => Err(format!(
            "{context}: unsupported scalar type (only 32-bit float, int and uint)"
        )),
    }
}

fn layout(value: &Json, context: &str) -> Result<LayoutRecord, String> {
    let kind = value
        .field("kind", context)?
        .as_str()
        .ok_or_else(|| format!("{context}: `kind` must be a string"))?;
    // The type's stride in the record is `TypeLayout::stride`: the reflection's own, except for a matrix.
    let mut stride = None;
    let (name, kind) = match kind {
        "scalar" => {
            let scalar = scalar_type(value, context)?;
            let name = match scalar {
                ScalarKind::F32 => "f32",
                ScalarKind::I32 => "i32",
                ScalarKind::U32 => "u32",
            };
            (name.to_owned(), LayoutKindRecord::Scalar(scalar))
        }
        "vector" => {
            let scalar = scalar_type(value, context)?;
            let count = number(value, "elementCount", context)?;
            // The Rust spelling of `gust::math`, as the macro names it.
            let prefix = match scalar {
                ScalarKind::F32 => "",
                ScalarKind::I32 => "I",
                ScalarKind::U32 => "U",
            };
            (
                format!("{prefix}Vec{count}"),
                LayoutKindRecord::Vector { scalar, count },
            )
        }
        "matrix" => {
            let scalar = scalar_type(value, context)?;
            let layout = value
                .field("matrixLayout", context)?
                .as_str()
                .ok_or_else(|| format!("{context}: `matrixLayout` must be a string"))?;
            if layout != "column" {
                return Err(format!(
                    "{context}: unsupported matrix layout `{layout}` (only column-major)"
                ));
            }
            let rows = number(value, "rows", context)?;
            let cols = number(value, "columns", context)?;
            // The document's stride of a matrix is its column stride; `TypeLayout::stride` is the whole matrix's.
            let column_stride = number(value, "stride", context)?;
            stride = Some(number(value, "size", context)?);
            let name = if rows == cols {
                format!("Mat{rows}")
            } else {
                format!("Mat{rows}x{cols}")
            };
            (
                name,
                LayoutKindRecord::Matrix {
                    scalar,
                    rows,
                    cols,
                    column_stride,
                },
            )
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
                "{context}: unsupported element type `{other}` (only 32-bit scalars, vectors, matrices and \r
                 structs of them)"
            ));
        }
    };
    Ok(LayoutRecord {
        name,
        size: number(value, "size", context)?,
        alignment: number(value, "alignment", context)?,
        stride: match stride {
            Some(stride) => stride,
            None => number(value, "stride", context)?,
        },
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
                "a texture element",
                PAIRS_REFLECTION.replace("\"kind\":\"struct\"", "\"kind\":\"texture\""),
                "unsupported element type `texture`",
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
                "a texture field",
                GAIN_REFLECTION.replacen("\"kind\":\"scalar\"", "\"kind\":\"texture\"", 1),
                "unsupported element type `texture`",
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

    /// The helper's reflection of the `uniforms` fixture's `run` (a `Uniform<Frame>` with a `Mat4`, `Vec4`s, a nested
    /// struct and `Vec2`s; two `Vec4` buffers) and of the `vectors` fixture's `run`, as `cargo gust build` wrote them
    /// (see `export_vector_tests.rs`).
    const FRAME_REFLECTION: &str =
        include_str!("testdata/uniform/98a66a209fa92eb1.wgsl.reflection.json");
    const VECTORS_REFLECTION: &str =
        include_str!("testdata/uniform/7ac7b445a03f9ce0.wgsl.reflection.json");

    #[test]
    fn a_vector_element_is_read_with_storage_v2_rules_and_its_own_stride() {
        let points = reflected_layout(FRAME_REFLECTION, "points").unwrap();
        assert_eq!(points.rules, ShaderLayoutRules::StorageV2);
        assert_eq!(points.element_stride, 16);
        assert_eq!(
            (
                points.element.name.as_str(),
                points.element.size,
                points.element.alignment
            ),
            ("Vec4", 16, 16)
        );
        assert_eq!(
            points.element.kind,
            LayoutKindRecord::Vector {
                scalar: ScalarKind::F32,
                count: 4
            }
        );
        for (name, scalar, count) in [
            ("counts", ScalarKind::U32, 2),
            ("signed_in", ScalarKind::I32, 4),
            ("wide_in", ScalarKind::U32, 4),
            ("velocities", ScalarKind::F32, 2),
        ] {
            let layout = reflected_layout(VECTORS_REFLECTION, name).unwrap();
            assert_eq!(layout.rules, ShaderLayoutRules::StorageV2, "{name}");
            assert_eq!(layout.element_stride, count * 4, "{name}");
            assert_eq!(
                layout.element.kind,
                LayoutKindRecord::Vector { scalar, count },
                "{name}"
            );
        }
        // A scalar buffer of the same kernel stays `StorageV1`.
        let sums = reflected_layout(VECTORS_REFLECTION, "sums").unwrap();
        assert_eq!(sums.rules, ShaderLayoutRules::StorageV1);
        assert_eq!(sums.element.kind, LayoutKindRecord::Scalar(ScalarKind::F32));
    }

    #[test]
    fn a_struct_of_vectors_is_read_as_storage_v2_with_its_field_offsets() {
        let particles = reflected_layout(VECTORS_REFLECTION, "particles").unwrap();
        assert_eq!(particles.rules, ShaderLayoutRules::StorageV2);
        assert_eq!(particles.element_stride, 24);
        let LayoutKindRecord::Struct(fields) = &particles.element.kind else {
            panic!("Particle is a struct");
        };
        let shape: Vec<(&str, u32)> = fields
            .iter()
            .map(|field| (field.name.as_str(), field.offset))
            .collect();
        assert_eq!(
            shape,
            [("position", 0), ("velocity", 8), ("mass", 16), ("tag", 20)]
        );
    }

    #[test]
    fn a_uniform_with_a_matrix_vectors_and_a_nested_struct_is_read_as_std140() {
        let frame = reflected_uniform_layout(FRAME_REFLECTION, "frame").unwrap();
        assert_eq!(frame.rules, ShaderLayoutRules::Std140V1);
        assert_eq!((frame.element.size, frame.element.alignment), (128, 16));
        assert_eq!(frame.element_stride, 128);
        let LayoutKindRecord::Struct(fields) = &frame.element.kind else {
            panic!("Frame is a struct");
        };
        let shape: Vec<(&str, u32)> = fields
            .iter()
            .map(|field| (field.name.as_str(), field.offset))
            .collect();
        assert_eq!(
            shape,
            [
                ("view", 0),
                ("tint", 64),
                ("light", 80),
                ("offset", 112),
                ("scale", 120),
                ("count", 124)
            ]
        );
        // The matrix: 64 bytes, columns 16 apart; its stride in the record is its size, not the reflection's column.
        assert_eq!(
            fields[0].ty.kind,
            LayoutKindRecord::Matrix {
                scalar: ScalarKind::F32,
                rows: 4,
                cols: 4,
                column_stride: 16
            }
        );
        assert_eq!((fields[0].ty.size, fields[0].ty.stride), (64, 64));
        assert_eq!(
            fields[3].ty.kind,
            LayoutKindRecord::Vector {
                scalar: ScalarKind::F32,
                count: 2
            }
        );
    }
}
