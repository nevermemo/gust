use super::{
    error::{Error, invalid},
    model::{PodCrossCheck, ReflectedResourceKind, ReflectedTypeKind, Reflection, Stage},
    pod::{cross_check_layout, equal},
};
use crate::{
    KernelDescriptor, ScalarKind, ShaderLayoutRules, Shape, TypeLayout, TypeLayoutKind,
    abi::TextureAccess,
    texture::{Format as TexelFormat, SampleClass},
};
use std::collections::BTreeMap;

impl Reflection {
    /// Require complete layout evidence for every descriptor binding, with no extras: `StorageV1` and `StorageV2`
    /// storage buffers, and `Std140V1` uniforms (constant buffers, schema 2 only).
    ///
    /// A schema-1 document cannot describe a vector, a matrix or a uniform, so a descriptor that has one is refused
    /// with `reflection schema 1 cannot describe <kind>; re-export with cargo gust build`.
    pub fn cross_check_kernel(
        &self,
        kernel: &KernelDescriptor,
    ) -> Result<Vec<PodCrossCheck>, Error> {
        for (index, resource) in self.resources.iter().enumerate() {
            if self.resources[..index].iter().any(|previous| {
                previous.name == resource.name
                    || (previous.space, previous.binding) == (resource.space, resource.binding)
            }) {
                return Err(invalid("duplicate resource name or binding"));
            }
        }
        // The stage first: a compute reflection checked against a pixel descriptor (or the reverse) has nothing else
        // worth comparing (D91).
        check_stage(kernel.shape, self.stage)?;
        let mut reports = Vec::new();
        let mut bindings = BTreeMap::new();
        let (mut textures, mut samplers) = (0_usize, 0_usize);
        let mut outputs: Vec<(&'static str, TypeLayout)> = Vec::new();
        for (index, parameter) in kernel.parameters.iter().enumerate() {
            if kernel.parameters[..index]
                .iter()
                .any(|previous| previous.name == parameter.name)
            {
                return Err(Error::Mismatch(
                    "duplicate descriptor parameter name".into(),
                ));
            }
            if parameter.kind == crate::ParameterKind::Invocation {
                if parameter.binding.is_some()
                    || parameter.resource_layout.is_some()
                    || parameter.access != crate::Access::NotApplicable
                {
                    return Err(Error::Mismatch(
                        "invocation parameter has a resource contract".into(),
                    ));
                }
                continue;
            }
            if self.schema_version < 3
                && matches!(
                    parameter.kind,
                    crate::ParameterKind::Texture | crate::ParameterKind::Sampler
                )
            {
                return Err(Error::Mismatch(format!(
                    "reflection schema {} cannot describe {}; re-export with cargo gust build",
                    self.schema_version,
                    if parameter.kind == crate::ParameterKind::Texture {
                        "texture"
                    } else {
                        "sampler"
                    }
                )));
            }
            if self.schema_version < 2
                && let Some(what) = schema_one_gap(parameter)
            {
                return Err(Error::Mismatch(format!(
                    "reflection schema 1 cannot describe {what}; re-export with cargo gust build"
                )));
            }
            // Under `pixel` a render-target output (no binding; see `is_pixel_output`) is a colour target.
            if kernel.shape == Shape::Pixel && is_pixel_output(parameter) {
                let layout = parameter.resource_layout.ok_or_else(|| {
                    Error::Mismatch(format!(
                        "{}: missing descriptor element layout",
                        parameter.name
                    ))
                })?;
                outputs.push((parameter.name, layout.element));
                continue;
            }
            if matches!(
                parameter.kind,
                crate::ParameterKind::Texture | crate::ParameterKind::Sampler
            ) {
                let binding = parameter
                    .binding
                    .ok_or_else(|| Error::Mismatch("missing descriptor binding".into()))?;
                if bindings
                    .insert((binding.group, binding.binding), parameter.name)
                    .is_some()
                {
                    return Err(Error::Mismatch("duplicate descriptor binding".into()));
                }
                if parameter.kind == crate::ParameterKind::Texture {
                    self.check_texture(parameter, binding)?;
                    textures += 1;
                } else {
                    self.check_sampler(parameter, binding)?;
                    samplers += 1;
                }
                continue;
            }
            let expected_kind = match parameter.kind {
                crate::ParameterKind::Storage => ReflectedResourceKind::StructuredBuffer,
                crate::ParameterKind::Uniform => ReflectedResourceKind::ConstantBuffer,
                _ => {
                    return Err(Error::Mismatch(
                        "only storage and uniform descriptor resources are supported".into(),
                    ));
                }
            };
            let binding = parameter
                .binding
                .ok_or_else(|| Error::Mismatch("missing descriptor binding".into()))?;
            let layout = parameter
                .resource_layout
                .ok_or_else(|| Error::Mismatch("missing descriptor element layout".into()))?;
            if bindings
                .insert((binding.group, binding.binding), parameter.name)
                .is_some()
            {
                return Err(Error::Mismatch("duplicate descriptor binding".into()));
            }
            let rules_fit = matches!(
                (expected_kind, layout.rules),
                (
                    ReflectedResourceKind::StructuredBuffer,
                    ShaderLayoutRules::StorageV1 | ShaderLayoutRules::StorageV2,
                ) | (
                    ReflectedResourceKind::ConstantBuffer,
                    ShaderLayoutRules::Std140V1
                )
            );
            if !rules_fit {
                return Err(Error::Mismatch(format!(
                    "{}.rules: {:?} does not fit this parameter kind",
                    parameter.name, layout.rules
                )));
            }
            // A constant buffer has no array stride: its stride slot holds the element size.
            let expected_stride = if layout.rules == ShaderLayoutRules::Std140V1 {
                layout.element.size
            } else {
                layout.element.stride
            };
            if layout.element_stride != expected_stride {
                return Err(Error::Mismatch(format!(
                    "{}.stride: inconsistent host resource layout",
                    parameter.name
                )));
            }
            let resource = self
                .resources
                .iter()
                .find(|resource| resource.name == parameter.name)
                .ok_or_else(|| {
                    Error::Mismatch(format!("missing compiler binding {}", parameter.name))
                })?;
            if resource.kind != expected_kind {
                return Err(Error::Mismatch(format!(
                    "{}.kind: host {expected_kind:?}, compiler {:?}",
                    parameter.name, resource.kind
                )));
            }
            equal(
                binding.group,
                resource.space,
                &format!("{}.group", parameter.name),
            )?;
            equal(
                binding.binding,
                resource.binding,
                &format!("{}.binding", parameter.name),
            )?;
            if !matches!(
                parameter.access,
                crate::Access::ReadOnly | crate::Access::ReadWrite
            ) || parameter.access != resource.access
            {
                return Err(Error::Mismatch(format!(
                    "{}.access differs",
                    parameter.name
                )));
            }
            let report = cross_check_layout(layout.element, layout.rules, resource)?;
            if !report.is_full_abi_match() {
                return Err(Error::Incomplete(format!(
                    "{} requires all nested sizes, alignments and strides",
                    parameter.name
                )));
            }
            reports.push(report);
        }
        if reports.len() != self.resources.len()
            || textures != self.textures.len()
            || samplers != self.samplers.len()
        {
            return Err(Error::Mismatch(
                "unexpected compiler resource bindings".into(),
            ));
        }
        self.check_targets(&outputs)?;
        Ok(reports)
    }

    /// A texture parameter against the document: the descriptor's own consistency first (the layout is the texture
    /// rule, the class and format agree with the result type, the access with the parameter's), then slot, dimension
    /// and access. A **sampled** texture is held to the sample class and the width of its result type and to having no
    /// format (its concrete format is checked at submit against the bound texture); a **storage** or **write-only**
    /// texture to its exact result type and exact format, which must have the result's channel count. Each mismatch
    /// names `<parameter>.<field>`.
    fn check_texture(
        &self,
        parameter: &crate::ParameterDescriptor,
        binding: crate::ResourceBinding,
    ) -> Result<(), Error> {
        let name = parameter.name;
        let layout = parameter
            .resource_layout
            .ok_or_else(|| Error::Mismatch(format!("{name}: missing descriptor element layout")))?;
        let ShaderLayoutRules::TextureV1(texture) = layout.rules else {
            return Err(Error::Mismatch(format!(
                "{name}.rules: {:?} does not fit this parameter kind",
                layout.rules
            )));
        };
        // The descriptor's own coherence: these are host-side facts, so a disagreement is a malformed descriptor.
        let Some((scalar, count)) = result_shape(layout.element).filter(|(_, count)| *count != 3)
        else {
            return Err(Error::Mismatch(format!(
                "{name}.result: a texture result is a 32-bit scalar or a 2- or 4-wide vector"
            )));
        };
        if layout.element_stride != layout.element.stride {
            return Err(Error::Mismatch(format!(
                "{name}.stride: inconsistent host resource layout"
            )));
        }
        if texture.class != class_of(scalar) {
            return Err(Error::Mismatch(format!(
                "{name}.class: host {}, result type {}",
                texture.class.name(),
                class_of(scalar).name()
            )));
        }
        let access_fits = match texture.access {
            TextureAccess::Sampled => parameter.access == crate::Access::ReadOnly,
            TextureAccess::ReadWrite | TextureAccess::WriteOnly => {
                parameter.access == crate::Access::ReadWrite
            }
        };
        if !access_fits {
            return Err(Error::Mismatch(format!("{name}.access differs")));
        }
        match (texture.access, texture.format) {
            (TextureAccess::Sampled, None) => {}
            (TextureAccess::Sampled, Some(format)) => {
                return Err(Error::Mismatch(format!(
                    "{name}.format: a sampled texture has no format in the shader, host {format}"
                )));
            }
            (_, None) => {
                return Err(Error::Mismatch(format!(
                    "{name}.format: a storage texture needs a format"
                )));
            }
            (access, Some(format)) => {
                if format.sample_class() != texture.class
                    || (access == TextureAccess::ReadWrite && !format.read_write())
                {
                    return Err(Error::Mismatch(format!(
                        "{name}.format: {format} does not fit class {} and access {}",
                        texture.class.name(),
                        access.name()
                    )));
                }
                if format.channels() != count {
                    return Err(Error::Mismatch(format!(
                        "{name}.format: {format} has channel count {}, the result type has width {count}",
                        format.channels()
                    )));
                }
            }
        }
        let reflected = self
            .textures
            .iter()
            .find(|candidate| candidate.name == name);
        let Some(reflected) = reflected else {
            return Err(self.missing(name, "texture"));
        };
        equal(binding.group, reflected.space, &format!("{name}.group"))?;
        equal(
            binding.binding,
            reflected.binding,
            &format!("{name}.binding"),
        )?;
        if texture.dimension != reflected.dimension {
            return Err(Error::Mismatch(format!(
                "{name}.dimension: host {}, compiler {}",
                texture.dimension.name(),
                reflected.dimension.name()
            )));
        }
        if texture.access != reflected.access {
            return Err(Error::Mismatch(format!(
                "{name}.access: host {}, compiler {}",
                texture.access.name(),
                reflected.access.name()
            )));
        }
        let Some((reflected_scalar, reflected_count)) = reflected_shape(&reflected.result.kind)
        else {
            return Err(Error::Mismatch(format!(
                "{name}.result: the compiler's result type is not a scalar or a vector"
            )));
        };
        if texture.access == TextureAccess::Sampled && texture.class != class_of(reflected_scalar) {
            return Err(Error::Mismatch(format!(
                "{name}.class: host {}, compiler {}",
                texture.class.name(),
                class_of(reflected_scalar).name()
            )));
        }
        // The result type for every texture (D91): a sampled one differs from its twin in width as well.
        if (scalar, count) != (reflected_scalar, reflected_count) {
            return Err(Error::Mismatch(format!(
                "{name}.result: host {}, compiler {}",
                shape_text(scalar, count),
                shape_text(reflected_scalar, reflected_count)
            )));
        }
        if texture.format != reflected.format {
            return Err(Error::Mismatch(format!(
                "{name}.format: host {}, compiler {}",
                format_text(texture.format),
                format_text(reflected.format)
            )));
        }
        Ok(())
    }

    /// A sampler parameter against the document: only its slot is compared (the options are bound at submit).
    fn check_sampler(
        &self,
        parameter: &crate::ParameterDescriptor,
        binding: crate::ResourceBinding,
    ) -> Result<(), Error> {
        let name = parameter.name;
        if parameter.resource_layout.is_some() || parameter.access != crate::Access::NotApplicable {
            return Err(Error::Mismatch(format!(
                "{name}: a sampler has no resource contract"
            )));
        }
        let Some(reflected) = self.samplers.iter().find(|sampler| sampler.name == name) else {
            return Err(self.missing(name, "sampler"));
        };
        equal(binding.group, reflected.space, &format!("{name}.group"))?;
        equal(
            binding.binding,
            reflected.binding,
            &format!("{name}.binding"),
        )
    }

    /// The error for a descriptor texture or sampler the document has under that name in no list of its own kind: a
    /// buffer or the other kind of binding under that name is a kind mismatch, nothing at all a missing binding.
    fn missing(&self, name: &str, host: &str) -> Error {
        let other = if self.resources.iter().any(|r| r.name == name) {
            Some("a buffer")
        } else if self.textures.iter().any(|t| t.name == name) {
            Some("a texture")
        } else if self.samplers.iter().any(|s| s.name == name) {
            Some("a sampler")
        } else {
            None
        };
        match other {
            Some(other) => Error::Mismatch(format!("{name}.kind: host {host}, compiler {other}")),
            None => Error::Mismatch(format!("missing compiler binding {name}")),
        }
    }

    /// The colour outputs of a pixel descriptor against the document's `targets`: the same count, the output `k`
    /// (in parameter order) at `SV_Target` index `k`, and the same scalar kind and width.
    fn check_targets(&self, outputs: &[(&'static str, TypeLayout)]) -> Result<(), Error> {
        if self.targets.len() != outputs.len() {
            return Err(Error::Mismatch(format!(
                "targets: host {} outputs, compiler {}",
                outputs.len(),
                self.targets.len()
            )));
        }
        for (position, ((name, element), target)) in outputs.iter().zip(&self.targets).enumerate() {
            let host_index = u32::try_from(position).unwrap_or(u32::MAX);
            equal(
                host_index,
                target.index,
                &format!("targets[{position}].index"),
            )?;
            let Some((scalar, count)) = result_shape(*element) else {
                return Err(Error::Mismatch(format!(
                    "{name}: an output is a 32-bit scalar or a 2- to 4-wide vector"
                )));
            };
            let Some((reflected_scalar, reflected_count)) = reflected_shape(&target.ty.kind) else {
                return Err(Error::Mismatch(format!(
                    "targets[{position}].type: the compiler's type is not a scalar or a vector"
                )));
            };
            if (scalar, count) != (reflected_scalar, reflected_count) {
                return Err(Error::Mismatch(format!(
                    "targets[{position}].type: host {}, compiler {}",
                    shape_text(scalar, count),
                    shape_text(reflected_scalar, reflected_count)
                )));
            }
        }
        Ok(())
    }
}

/// Whether a pixel descriptor's parameter is a render-target output: a storage parameter without a shader binding.
/// Its `SV_Target` index is its position among such parameters.
///
/// This is the **render-target model** (D93: an output is a colour target, not a resource), which D120 keeps only for
/// targets the host provides itself (Breeze). MX pixel outputs under D120 are fragment *storage writes*: their
/// parameters carry a `binding`, this function is false for them, and they are cross-checked as buffers, so the
/// document's `targets` must then be empty: the fragment of a `pixel` kernel returns nothing, and the never-written
/// `r8unorm` attachment of the pass is a host object the shader does not name (mx.p1a). Under D120 no kernel the macro
/// writes reaches the branch below, so this function is dead for MX until Breeze; the host (mx.p1b) lists it. The rule
/// is the descriptor-side one until the descriptor names a target explicitly (plan section 4.1's
/// `target: Some(k)`): then this function reads that field instead and nothing else changes. It does not look at
/// `access`: a read-only input that lost its binding counts as an output, and fails as a `targets` count mismatch.
fn is_pixel_output(parameter: &crate::ParameterDescriptor) -> bool {
    parameter.kind == crate::ParameterKind::Storage && parameter.binding.is_none()
}

/// The stage of the document against the kernel's shape. `Mismatch` naming `stage`.
pub(super) fn check_stage(shape: Shape, stage: Stage) -> Result<(), Error> {
    let host = Stage::of(shape);
    if host == stage {
        Ok(())
    } else {
        Err(Error::Mismatch(format!(
            "stage: host {}, compiler {}",
            host.name(),
            stage.name()
        )))
    }
}

/// The scalar kind and width (1 for a scalar) of a host type layout, if it is a scalar or a vector.
fn result_shape(layout: TypeLayout) -> Option<(ScalarKind, u32)> {
    match layout.kind {
        TypeLayoutKind::Scalar(scalar) => Some((scalar, 1)),
        TypeLayoutKind::Vector { scalar, count } => Some((scalar, count)),
        _ => None,
    }
}

/// The same for a reflected type.
fn reflected_shape(kind: &ReflectedTypeKind) -> Option<(ScalarKind, u32)> {
    match kind {
        ReflectedTypeKind::Scalar(scalar) => Some((*scalar, 1)),
        ReflectedTypeKind::Vector { scalar, count } => Some((*scalar, *count)),
        _ => None,
    }
}

/// The sample class of a result type's scalar.
const fn class_of(scalar: ScalarKind) -> SampleClass {
    match scalar {
        ScalarKind::F32 => SampleClass::Float,
        ScalarKind::U32 => SampleClass::Uint,
        ScalarKind::I32 => SampleClass::Sint,
    }
}

fn shape_text(scalar: ScalarKind, count: u32) -> String {
    let name = scalar_text(scalar);
    if count == 1 {
        name.to_owned()
    } else {
        format!("{name}x{count}")
    }
}

fn scalar_text(scalar: ScalarKind) -> &'static str {
    match scalar {
        ScalarKind::F32 => "f32",
        ScalarKind::I32 => "i32",
        ScalarKind::U32 => "u32",
    }
}

fn format_text(format: Option<TexelFormat>) -> &'static str {
    format.map_or("none", TexelFormat::name)
}

/// What a schema-1 document cannot describe about this parameter, if anything: a uniform, or a vector or matrix
/// anywhere in its element.
fn schema_one_gap(parameter: &crate::ParameterDescriptor) -> Option<&'static str> {
    fn layout_gap(layout: TypeLayout) -> Option<&'static str> {
        match layout.kind {
            TypeLayoutKind::Scalar(_) => None,
            TypeLayoutKind::Vector { .. } => Some("vector"),
            TypeLayoutKind::Matrix { .. } => Some("matrix"),
            TypeLayoutKind::Struct(fields) => fields.iter().find_map(|field| layout_gap(field.ty)),
            // Only schema 4 describes an array (m9.r2b).
            TypeLayoutKind::Array { .. } => Some("array"),
        }
    }
    if parameter.kind == crate::ParameterKind::Uniform {
        return Some("uniform");
    }
    parameter
        .resource_layout
        .and_then(|layout| layout_gap(layout.element))
}
