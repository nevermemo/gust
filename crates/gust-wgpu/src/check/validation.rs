use crate::texture;
use crate::{BufferBinding, Error, GpuSamplerBinding, binding_resource_layout, render};
#[cfg(not(target_arch = "wasm32"))]
#[allow(deprecated)]
use crate::{F32Binding, U32Binding};
#[cfg(not(target_arch = "wasm32"))]
use gust::GpuPod;
use gust::target::Target;
use gust::{
    Access, BindingKind, Extent, KernelDescriptor, ParameterDescriptor, ParameterKind,
    ResourceLayout, Shape,
};

pub(crate) fn validate_binding_layout(
    parameter: &ParameterDescriptor,
    actual: ResourceLayout,
) -> Result<(), Error> {
    let expected = parameter
        .resource_layout
        .expect("validated resource layout");
    if expected != actual {
        return Err(Error::UnsupportedResourceLayout {
            parameter: parameter.name.to_owned(),
            expected: Box::new(expected),
            actual: Some(Box::new(actual)),
        });
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(deprecated)]
pub(crate) fn validate_dispatch(
    kernel: &KernelDescriptor,
    element_count: u32,
    bindings: &[F32Binding<'_>],
) -> Result<Vec<&'static ParameterDescriptor>, Error> {
    render::require_grid(kernel, render::ONE_SHOT_REASON)?;
    let resources = validate_kernel(kernel)?;
    if resources.len() != bindings.len() {
        return Err(Error::BindingCount {
            expected: resources.len(),
            actual: bindings.len(),
        });
    }
    for (parameter, binding) in resources.iter().copied().zip(bindings.iter().copied()) {
        validate_binding_layout(
            parameter,
            ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT),
        )?;
        if parameter.access != binding.access() {
            return Err(Error::BindingAccess {
                parameter: parameter.name.to_owned(),
                expected: parameter.access,
                actual: binding.access(),
            });
        }
        if binding.values().len() != element_count as usize {
            return Err(Error::BufferLength {
                parameter: parameter.name.to_owned(),
                expected: element_count as usize,
                actual: binding.values().len(),
            });
        }
    }
    Ok(resources)
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(deprecated)]
pub(crate) fn validate_u32_dispatch(
    kernel: &KernelDescriptor,
    element_count: u32,
    bindings: &[U32Binding<'_>],
) -> Result<Vec<&'static ParameterDescriptor>, Error> {
    render::require_grid(kernel, render::ONE_SHOT_REASON)?;
    let resources = validate_kernel(kernel)?;
    if resources.len() != bindings.len() {
        return Err(Error::BindingCount {
            expected: resources.len(),
            actual: bindings.len(),
        });
    }
    for (parameter, binding) in resources.iter().copied().zip(bindings.iter().copied()) {
        validate_binding_layout(
            parameter,
            ResourceLayout::storage_v1(<u32 as GpuPod>::LAYOUT),
        )?;
        if parameter.access != binding.access() {
            return Err(Error::BindingAccess {
                parameter: parameter.name.to_owned(),
                expected: parameter.access,
                actual: binding.access(),
            });
        }
        if binding.values().len() != element_count as usize {
            return Err(Error::BufferLength {
                parameter: parameter.name.to_owned(),
                expected: element_count as usize,
                actual: binding.values().len(),
            });
        }
    }
    Ok(resources)
}

/// The binding checks of a one-dimensional dispatch of `element_count` invocations of a buffer-only kernel: what the
/// staged graph validates a node with. A kernel with a texture or sampler parameter is refused
/// ([`Error::UnsupportedParameter`]): the graph tracks hazards by buffer and does not learn textures in MX.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) fn validate_persistent_dispatch(
    device_id: u64,
    kernel: &KernelDescriptor,
    element_count: u32,
    bindings: &[BufferBinding<'_>],
) -> Result<Vec<&'static ParameterDescriptor>, Error> {
    let resources = validate_kernel(kernel)?;
    if let Some(parameter) = resources.iter().find(|parameter| {
        matches!(
            parameter.kind,
            ParameterKind::Texture | ParameterKind::Sampler
        )
    }) {
        return Err(Error::UnsupportedParameter {
            name: parameter.name.to_owned(),
            kind: parameter.kind,
        });
    }
    validate_dispatch_extent(
        device_id,
        false,
        kernel,
        Extent::linear(element_count),
        bindings,
        &[],
    )
}

/// The binding checks of a persistent dispatch over `extent`: the kernel, then per binding the kind, layout, device,
/// access and length, then aliasing. A shared binding holds exactly `x * y * z` elements (the checked product;
/// [`Error::Extent`] when it overflows a `u64`, before any binding is looked at), an independent one only has to be
/// non-empty when the extent is. The launch limits are `check_launch_extent`'s.
///
/// A texture binding has no length rule. It is checked for its format (a storage parameter's exact format, a sampled
/// one's sample class: [`Error::TextureFormat`]), its device, its access and its use ([`Error::TextureUse`]). A sampler
/// binding is checked for its device and its filter against the device's layout variant (`float32_filterable`:
/// [`Error::TextureFilter`]). `samplers` are the bindings of the kernel's sampler parameters in declaration order.
pub(crate) fn validate_dispatch_extent(
    device_id: u64,
    float32_filterable: bool,
    kernel: &KernelDescriptor,
    extent: Extent,
    bindings: &[BufferBinding<'_>],
    samplers: &[GpuSamplerBinding<'_>],
) -> Result<Vec<&'static ParameterDescriptor>, Error> {
    let resources = validate_kernel(kernel)?;
    // A `pixel` call is one-dimensional (the fragment index is `y * 8192 + x` of the attachment, not a call's y).
    render::check_pixel_extent(kernel, extent)?;
    let total = extent.checked_total().ok_or(Error::Extent {
        extent: extent.to_array().map(u64::from),
        reason: "x * y * z overflows a u64",
    })?;
    // A product above `usize::MAX` is longer than any buffer: it saturates, so the length check fails for all of them.
    let expected = usize::try_from(total).unwrap_or(usize::MAX);
    // The resource parameters (buffers and textures) pair with `bindings`, the sampler parameters with `samplers`.
    let (sampler_parameters, resource_parameters): (
        Vec<&ParameterDescriptor>,
        Vec<&ParameterDescriptor>,
    ) = resources
        .iter()
        .copied()
        .partition(|parameter| parameter.kind == ParameterKind::Sampler);
    if resource_parameters.len() != bindings.len() {
        return Err(Error::BindingCount {
            expected: resource_parameters.len(),
            actual: bindings.len(),
        });
    }
    if sampler_parameters.len() != samplers.len() {
        return Err(Error::BindingCount {
            expected: sampler_parameters.len(),
            actual: samplers.len(),
        });
    }
    for (parameter, binding) in resource_parameters
        .iter()
        .copied()
        .zip(bindings.iter().copied())
    {
        let buffer = binding;
        let expected_kind =
            BindingKind::of(parameter.kind).expect("validate_kernel keeps only bound parameters");
        if buffer.kind != expected_kind {
            return Err(Error::BindingKind {
                parameter: parameter.name.to_owned(),
                expected: expected_kind,
                actual: buffer.kind,
            });
        }
        if let Some(texture) = binding.texture {
            texture::validate_texture_binding(device_id, parameter, &binding, texture)?;
            continue;
        }
        validate_binding_layout(
            parameter,
            binding_resource_layout(buffer.kind, buffer.layout),
        )?;
        if buffer.device_id != device_id {
            return Err(Error::ForeignPersistentBuffer);
        }
        if parameter.access != binding.access {
            return Err(Error::BindingAccess {
                parameter: parameter.name.to_owned(),
                expected: parameter.access,
                actual: binding.access,
            });
        }
        if !buffer.buffer_access.allows(binding.access) {
            return Err(Error::PersistentBufferAccess {
                parameter: parameter.name.to_owned(),
                buffer: buffer.buffer_access,
                binding: binding.access,
            });
        }
        // An output a `pixel` kernel writes in place must have exactly `n` elements (D127): there is no clamp-free way
        // to bound a store to a shorter one, so the twin refuses it too (`cpu::Error::PixelIndependentOutput`).
        if kernel.shape == Shape::Pixel
            && parameter.kind == ParameterKind::Storage
            && parameter.access == Access::ReadWrite
            && binding.independent_length
        {
            return Err(Error::ShapeIndependentOutput {
                kernel: kernel.qualified_name().to_string(),
                parameter: parameter.name.to_owned(),
            });
        }
        if binding.independent_length {
            if buffer.len == 0 && total != 0 {
                return Err(Error::IndependentLengthEmpty {
                    parameter: parameter.name.to_owned(),
                });
            }
        } else if buffer.len != expected {
            return Err(Error::BufferLength {
                parameter: parameter.name.to_owned(),
                expected,
                actual: buffer.len,
            });
        }
    }
    for (parameter, sampler) in sampler_parameters.iter().copied().zip(samplers) {
        texture::validate_sampler_binding(device_id, float32_filterable, parameter, sampler)?;
    }
    // One resource may be bound twice only when both bindings are read-only. The access of each binding
    // equals its parameter's access here, and the check names both parameters in declaration order. A texture
    // is compared by its own identity (`Raw`): a storage texture bound twice is refused, a sampled one twice is
    // accepted (wgpu allows two read-only uses of one texture), and a texture never equals a buffer.
    for (index, (first_parameter, first)) in resource_parameters.iter().zip(bindings).enumerate() {
        for (second_parameter, second) in resource_parameters.iter().zip(bindings).skip(index + 1) {
            if first.raw == second.raw
                && (first.access == Access::ReadWrite || second.access == Access::ReadWrite)
            {
                return Err(Error::AliasedBinding {
                    kernel: kernel.qualified_name().to_string(),
                    first: first_parameter.name.to_owned(),
                    second: second_parameter.name.to_owned(),
                });
            }
        }
    }
    Ok(resources)
}

pub(crate) fn validate_kernel(
    kernel: &KernelDescriptor,
) -> Result<Vec<&'static ParameterDescriptor>, Error> {
    // A kernel built for the CPU has no GPU form (D137). The typed entry points refuse it at compile time; a kernel given
    // by its descriptor is refused here, first of this function's checks, on every path, whatever its extent. (The
    // one-shot paths and the generated host snippet refuse a shape that is not `grid` before they call this, so a
    // CPU-target `pixel` kernel given to one of them is `ShapeUnsupported`.)
    if kernel.target != Target::Gpu {
        return Err(Error::KernelTarget {
            kernel: kernel.qualified_name().to_string(),
            target: kernel.target,
        });
    }
    // `grid` runs as a compute pass and `pixel` as a render pass (slice mx.p1b); a shape added later has no path yet.
    if !matches!(kernel.shape, Shape::Grid | Shape::Pixel) {
        return Err(render::unknown_shape(kernel));
    }
    if kernel.slang_source.trim().is_empty() {
        return Err(Error::MissingSlangSource(
            kernel.qualified_name().to_string(),
        ));
    }
    // Workgroups of one, two or three dimensions (slice m7.d); an axis of 0 is no workgroup. The device limits of each
    // axis and of their product are `check_kernel_limits`'s.
    if kernel.workgroup_size.contains(&0) {
        return Err(Error::UnsupportedWorkgroupSize(kernel.workgroup_size));
    }

    let resources = kernel
        .parameters
        .iter()
        .filter(|parameter| parameter.binding.is_some())
        .collect::<Vec<_>>();
    for parameter in resources.iter().copied() {
        // The rule is a function of the element (`binding_resource_layout`): a `Vec4` labelled StorageV1, or a storage
        // parameter under the Std140V1 rule, is refused here and not only later at the persistent dispatch. A uniform
        // is a constant buffer: read-only, and `Std140V1` with an element that satisfies it.
        let layout_ok = match parameter.kind {
            ParameterKind::Storage => parameter.resource_layout.is_some_and(|layout| {
                layout.element.is_storage_v2()
                    && layout == binding_resource_layout(BindingKind::Storage, layout.element)
            }),
            ParameterKind::Uniform => {
                if parameter.access != Access::ReadOnly {
                    return Err(Error::BindingAccess {
                        parameter: parameter.name.to_owned(),
                        expected: Access::ReadOnly,
                        actual: parameter.access,
                    });
                }
                parameter.resource_layout.is_some_and(|layout| {
                    layout.element.is_std140_v1()
                        && layout == binding_resource_layout(BindingKind::Uniform, layout.element)
                })
            }
            // A texture's contract is the `TextureV1` rule (D91) and a sampler has none: `texture.rs` judges both.
            ParameterKind::Texture => texture::texture_parameter_is_valid(parameter),
            ParameterKind::Sampler => texture::sampler_parameter_is_valid(parameter),
            _ => {
                return Err(Error::UnsupportedParameter {
                    name: parameter.name.to_owned(),
                    kind: parameter.kind,
                });
            }
        };
        if !layout_ok {
            return Err(Error::InvalidResourceLayout {
                parameter: parameter.name.to_owned(),
                actual: parameter.resource_layout.map(Box::new),
            });
        }
        let resource_binding = parameter.binding.expect("filtered resource binding");
        if resource_binding.group != 0 {
            return Err(Error::UnsupportedBindGroup(resource_binding.group));
        }
    }
    Ok(resources)
}

#[cfg(test)]
mod tests;
