use super::{
    error::{Error, invalid},
    json::{Json, Parser},
    model::{
        ReflectedField, ReflectedResource, ReflectedResourceKind, ReflectedSampler,
        ReflectedTarget, ReflectedTexture, ReflectedType, ReflectedTypeKind, Reflection, Stage,
    },
};
use crate::{
    ScalarKind,
    abi::{MATRIX_LAYOUT, MatrixLayout, TextureAccess, TextureDimension},
    slang::Target,
    texture::Format as TexelFormat,
};

impl Reflection {
    /// Parse global structured buffers and the generated dispatch-ID parameter.
    /// Entry-point resources and ordinary value/uniform parameters are unsupported.
    pub fn from_json(target: Target, source: &str) -> Result<Self, Error> {
        let root = Parser::parse(source)?;
        Self::read_json(target, &root, false, 1)
    }

    /// `schema` is the document's `schemaVersion` (1 to 4): under 1 exactly the pre-schema-2 grammar is read, and
    /// the texture, sampler, `stage` and `targets` members exist from 3 on only.
    pub(super) fn read_json(
        target: Target,
        root: &Json,
        complete: bool,
        schema: u32,
    ) -> Result<Self, Error> {
        let stage = read_stage(root, schema)?;
        if schema >= 2 {
            let text = root.get("matrixLayout")?.string()?;
            match MatrixLayout::parse(text) {
                Some(layout) if layout == MATRIX_LAYOUT => {}
                Some(layout) => {
                    return Err(Error::Mismatch(format!(
                        "matrixLayout: host {}, compiler {}",
                        MATRIX_LAYOUT.as_str(),
                        layout.as_str()
                    )));
                }
                None => return Err(invalid("unknown matrixLayout")),
            }
        }
        if let Some(entries) = root.optional("entryPoints")? {
            for entry in entries.array()? {
                if let Some(parameters) = entry.optional("parameters")? {
                    for parameter in parameters.array()? {
                        if parameter.optional("binding")?.is_some()
                            || parameter.optional("bindings")?.is_some()
                        {
                            return Err(invalid(
                                "entry-point resource/uniform bindings are unsupported",
                            ));
                        }
                        let ty = parameter.get("type")?;
                        if parameter
                            .optional("semanticName")?
                            .map(Json::string)
                            .transpose()?
                            != Some("SV_DISPATCHTHREADID")
                            || ty.get("kind")?.string()? != "vector"
                            || ty.get("elementCount")?.u32()? != 3
                            || ty.get("elementType")?.get("kind")?.string()? != "scalar"
                            || ty.get("elementType")?.get("scalarType")?.string()? != "uint32"
                        {
                            return Err(invalid(
                                "only dispatch-thread-ID entry-point parameters are supported",
                            ));
                        }
                    }
                }
            }
        }
        let mut resources: Vec<ReflectedResource> = Vec::new();
        let mut textures: Vec<ReflectedTexture> = Vec::new();
        let mut samplers: Vec<ReflectedSampler> = Vec::new();
        for value in root.get("parameters")?.array()? {
            let name = value.get("name")?.string()?.to_owned();
            let binding = value.get("binding")?;
            if binding.get("kind")?.string()? != "descriptorTableSlot" {
                return Err(invalid("unsupported resource binding kind"));
            }
            let index = binding.get("index")?.u32()?;
            let space = if complete {
                binding.get("space")?.u32()?
            } else {
                binding
                    .optional("space")?
                    .map(Json::u32)
                    .transpose()?
                    .unwrap_or(0)
            };
            let ty = value.get("type")?;
            // Schema 3 only: a texture or a sampler. Schemas 1 and 2 never evaluate this and read exactly what they
            // always did (a texture there is refused as "only structured buffers are supported").
            if schema >= 3 && matches!(ty.get("kind")?.string()?, "texture" | "sampler") {
                if complete && binding.get("count")?.u32()? != 1 {
                    return Err(invalid("binding arrays are unsupported"));
                }
                if let Some(count) = binding.optional("count")?
                    && count.u32()? != 1
                {
                    return Err(invalid("binding arrays are unsupported"));
                }
                if slot_taken(&resources, &textures, &samplers, &name, index, space) {
                    return Err(invalid("duplicate resource name or binding"));
                }
                if ty.get("kind")?.string()? == "sampler" {
                    samplers.push(ReflectedSampler {
                        name,
                        binding: index,
                        space,
                    });
                } else {
                    textures.push(read_texture(name, index, space, ty)?);
                }
                continue;
            }
            // Schema 2 only: a constant buffer. Schema 1 never evaluates this and reads exactly what it always did.
            let constant = schema >= 2 && ty.get("kind")?.string()? == "constantBuffer";
            if !constant
                && (ty.get("kind")?.string()? != "resource"
                    || ty.get("baseShape")?.string()? != "structuredBuffer")
            {
                return Err(invalid("only structured buffers are supported"));
            }
            let access = if constant {
                crate::Access::ReadOnly
            } else {
                match ty.optional("access")?.map(Json::string).transpose()? {
                    Some("read") => crate::Access::ReadOnly,
                    None if !complete => crate::Access::ReadOnly,
                    Some("readWrite") => crate::Access::ReadWrite,
                    _ => return Err(invalid("unsupported resource access")),
                }
            };
            if complete && binding.get("count")?.u32()? != 1 {
                return Err(invalid("binding arrays are unsupported"));
            }
            if let Some(count) = binding.optional("count")?
                && count.u32()? != 1
            {
                return Err(invalid("binding arrays are unsupported"));
            }
            if slot_taken(&resources, &textures, &samplers, &name, index, space) {
                return Err(invalid("duplicate resource name or binding"));
            }
            let (kind, element, element_stride) = if constant {
                // A constant buffer has no array stride: `size` is its element's size, rounded up by the layout rule.
                (
                    ReflectedResourceKind::ConstantBuffer,
                    read_type(ty.get("elementType")?, complete, schema)?,
                    if complete {
                        Some(ty.get("size")?.u32()?)
                    } else {
                        None
                    },
                )
            } else {
                (
                    ReflectedResourceKind::StructuredBuffer,
                    read_type(ty.get("resultType")?, complete, schema)?,
                    if complete {
                        Some(value.get("elementStride")?.u32()?)
                    } else {
                        None
                    },
                )
            };
            resources.push(ReflectedResource {
                name,
                kind,
                access,
                binding: index,
                space,
                element,
                element_stride,
            });
        }
        let targets = if schema >= 3 {
            read_targets(root, stage)?
        } else {
            Vec::new()
        };
        Ok(Self {
            target,
            schema_version: schema,
            stage,
            resources,
            textures,
            samplers,
            targets,
        })
    }
}

/// The document's `stage`: the member from schema 3 on, compute for the schemas that predate it.
pub(super) fn read_stage(root: &Json, schema: u32) -> Result<Stage, Error> {
    if schema < 3 {
        return Ok(Stage::Compute);
    }
    match root.get("stage")?.string()? {
        "compute" => Ok(Stage::Compute),
        "fragment" => Ok(Stage::Fragment),
        _ => Err(invalid("unknown stage")),
    }
}

/// Whether a resource of the same name or the same `(space, binding)` is already read, in any of the three lists.
fn slot_taken(
    resources: &[ReflectedResource],
    textures: &[ReflectedTexture],
    samplers: &[ReflectedSampler],
    name: &str,
    binding: u32,
    space: u32,
) -> bool {
    resources
        .iter()
        .map(|r| (r.name.as_str(), r.binding, r.space))
        .chain(
            textures
                .iter()
                .map(|t| (t.name.as_str(), t.binding, t.space)),
        )
        .chain(
            samplers
                .iter()
                .map(|s| (s.name.as_str(), s.binding, s.space)),
        )
        .any(|(other, other_binding, other_space)| {
            other == name || (other_binding == binding && other_space == space)
        })
}

/// One `kind: "texture"` parameter of a schema-3 document.
fn read_texture(
    name: String,
    binding: u32,
    space: u32,
    ty: &Json,
) -> Result<ReflectedTexture, Error> {
    let dimension = match ty.get("shape")?.string()? {
        "2d" => TextureDimension::D2,
        _ => return Err(invalid("unsupported texture shape (only 2d)")),
    };
    let access = match ty.get("access")?.string()? {
        "read" => TextureAccess::Sampled,
        "readWrite" => TextureAccess::ReadWrite,
        "write" => TextureAccess::WriteOnly,
        _ => return Err(invalid("unsupported texture access")),
    };
    let result = read_type(ty.get("result")?, false, 3)?;
    if !matches!(
        result.kind,
        ReflectedTypeKind::Scalar(_) | ReflectedTypeKind::Vector { .. }
    ) {
        return Err(invalid(
            "unsupported texture result type (a scalar or a vector)",
        ));
    }
    let format = match ty.get("format")?.string()? {
        "unknown" => None,
        text => Some(
            TexelFormat::ALL
                .iter()
                .copied()
                .find(|format| format.slang_name() == text)
                .ok_or_else(|| invalid(format!("unsupported texture format `{text}`")))?,
        ),
    };
    Ok(ReflectedTexture {
        name,
        binding,
        space,
        dimension,
        access,
        result,
        format,
    })
}

/// The `targets` of a schema-3 document: present with a fragment stage (possibly empty), absent with a compute one.
fn read_targets(root: &Json, stage: Stage) -> Result<Vec<ReflectedTarget>, Error> {
    let Some(targets) = root.optional("targets")? else {
        return if stage == Stage::Fragment {
            Err(invalid("a fragment reflection needs targets"))
        } else {
            Ok(Vec::new())
        };
    };
    if stage == Stage::Compute {
        return Err(invalid("a compute reflection has no targets"));
    }
    let mut read: Vec<ReflectedTarget> = Vec::new();
    for target in targets.array()? {
        let index = target.get("index")?.u32()?;
        if read.last().is_some_and(|previous| previous.index >= index) {
            return Err(invalid(
                "targets must be in ascending SV_Target order without repeats",
            ));
        }
        let ty = read_type(target.get("type")?, false, 3)?;
        if !matches!(
            ty.kind,
            ReflectedTypeKind::Scalar(_) | ReflectedTypeKind::Vector { .. }
        ) {
            return Err(invalid("unsupported target type (a scalar or a vector)"));
        }
        read.push(ReflectedTarget { index, ty });
    }
    Ok(read)
}

/// The document's `schemaVersion`: 1, 2, 3 or 4, nothing else.
pub(super) fn read_schema_version(root: &Json) -> Result<u32, Error> {
    match root.get("schemaVersion")?.u32()? {
        version @ (1..=4) => Ok(version),
        _ => Err(invalid("unsupported native reflection schema version")),
    }
}

fn read_scalar(value: &Json) -> Result<ScalarKind, Error> {
    Ok(match value.get("scalarType")?.string()? {
        "float32" => ScalarKind::F32,
        "int32" => ScalarKind::I32,
        "uint32" => ScalarKind::U32,
        _ => return Err(invalid("unsupported scalar type")),
    })
}

/// A vector width or matrix dimension of the dialect: two to four.
fn read_dimension(value: &Json, key: &str) -> Result<u32, Error> {
    let n = value.get(key)?.u32()?;
    if (2..=4).contains(&n) {
        Ok(n)
    } else {
        Err(invalid(format!("unsupported {key} {n}")))
    }
}

/// `schema` is the document's `schemaVersion`; the `vector` and `matrix` kinds exist from 2 on, `array` from 4 on.
fn read_type(value: &Json, complete: bool, schema: u32) -> Result<ReflectedType, Error> {
    let kind = match value.get("kind")?.string()? {
        "scalar" => ReflectedTypeKind::Scalar(read_scalar(value)?),
        "vector" if schema >= 2 => ReflectedTypeKind::Vector {
            scalar: read_scalar(value)?,
            count: read_dimension(value, "elementCount")?,
        },
        "matrix" if schema >= 2 => ReflectedTypeKind::Matrix {
            scalar: read_scalar(value)?,
            rows: read_dimension(value, "rows")?,
            cols: read_dimension(value, "columns")?,
            layout: MatrixLayout::parse(value.get("matrixLayout")?.string()?)
                .ok_or_else(|| invalid("unknown matrixLayout"))?,
        },
        "struct" => {
            let mut fields = Vec::new();
            for field in value.get("fields")?.array()? {
                let name = field.get("name")?.string()?.to_owned();
                if fields.iter().any(|f: &ReflectedField| f.name == name) {
                    return Err(invalid("duplicate struct field"));
                }
                let binding = field.get("binding")?;
                if binding.get("kind")?.string()? != "uniform" {
                    return Err(invalid("unsupported field layout category"));
                }
                fields.push(ReflectedField {
                    name,
                    offset: binding.get("offset")?.u32()?,
                    size: binding.get("size")?.u32()?,
                    ty: read_type(field.get("type")?, complete, schema)?,
                });
            }
            if fields.is_empty() {
                return Err(invalid("empty struct layout"));
            }
            ReflectedTypeKind::Struct(fields)
        }
        "array" if schema >= 4 => {
            if !complete {
                return Err(invalid("an array needs a complete layout document"));
            }
            let count = value.get("elementCount")?.u32()?;
            let element_stride = value.get("elementStride")?.u32()?;
            if count == 0 || element_stride == 0 {
                return Err(invalid(
                    "an array needs a positive elementCount and elementStride",
                ));
            }
            let element = read_type(value.get("elementType")?, complete, schema)?;
            // Lower bounds only: an HLSL-style rule set may leave the last element unpadded, so `size` need not be
            // `count * stride`, but no element may overlap the next, sit off its alignment, or fall outside the array.
            let (size, element_size, alignment) = (
                value.get("size")?.u32()?,
                element.size.unwrap_or(0),
                element.alignment.unwrap_or(0),
            );
            if element_stride < element_size {
                return Err(invalid(
                    "an array's elementStride is smaller than its element's size",
                ));
            }
            if alignment == 0 || element_stride % alignment != 0 {
                return Err(invalid(
                    "an array's elementStride is not a multiple of its element's alignment",
                ));
            }
            let needed = (count - 1)
                .checked_mul(element_stride)
                .and_then(|used| used.checked_add(element_size));
            if needed.is_none_or(|needed| size < needed) {
                return Err(invalid(
                    "an array's size is smaller than its elements need (elementCount - 1) * elementStride + element size",
                ));
            }
            ReflectedTypeKind::Array {
                element: Box::new(element),
                count,
                element_stride,
            }
        }
        _ if schema >= 4 => {
            return Err(invalid(
                "unsupported element type (only 32-bit scalars, vectors, matrices, arrays and structs)",
            ));
        }
        _ if schema >= 2 => {
            return Err(invalid(
                "unsupported element type (only 32-bit scalars, vectors, matrices and structs)",
            ));
        }
        _ => {
            return Err(invalid(
                "unsupported element type (only 32-bit scalars and structs)",
            ));
        }
    };
    Ok(ReflectedType {
        name: value
            .optional("name")?
            .map(Json::string)
            .transpose()?
            .map(str::to_owned),
        size: if complete {
            Some(value.get("size")?.u32()?)
        } else {
            None
        },
        alignment: if complete {
            Some(value.get("alignment")?.u32()?)
        } else {
            None
        },
        stride: if complete {
            Some(value.get("stride")?.u32()?)
        } else {
            None
        },
        kind,
    })
}
