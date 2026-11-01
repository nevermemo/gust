//! Element layouts read from the Slang helper's reflection JSON, for the parameters whose layout the macro could not
//! write into the record (`layout: null`, listed as `export_unchecked` in the manifest).
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
    let resources = root
        .field("parameters", "reflection")?
        .as_array()
        .ok_or("reflection: `parameters` must be an array")?;
    let resource = resources
        .iter()
        .find(|resource| resource.get("name").and_then(Json::as_str) == Some(parameter))
        .ok_or_else(|| {
            format!(
                "the reflection reports no resource named `{parameter}` to take its layout from"
            )
        })?;
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
}
