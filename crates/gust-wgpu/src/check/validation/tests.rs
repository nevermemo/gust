use super::*;
use crate::render_wgpu_source;
#[allow(deprecated)]
use crate::{
    DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION, Error, F32Binding, GpuBufferAccess,
    HeadlessDevice, Raw, assert_pod_matches_storage_layout, binding_resource_layout,
    buffer_resource_layout, check_uniform_type, padding_free,
};
use gust::{
    Access, BindingKind, Extent, GpuPod, KernelDescriptor, ParameterDescriptor, ParameterKind,
    ResourceBinding, ResourceLayout, Shape,
};

const PARAMETERS: &[ParameterDescriptor] = &[ParameterDescriptor {
    name: "out",
    rust_type: "StorageMut<f32>",
    kind: ParameterKind::Storage,
    access: Access::ReadWrite,
    resource_layout: Some(ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT)),
    binding: Some(ResourceBinding {
        group: 0,
        binding: 0,
    }),
}];

const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new(
    "test",
    "kernel",
    "gpu_test_kernel",
    [64, 1, 1],
    PARAMETERS,
    r#"
            [[vk::binding(0, 0)]] RWStructuredBuffer<float> out;
            [shader("compute")]
            [numthreads(64, 1, 1)]
            void gpu_test_kernel(uint3 id : SV_DispatchThreadID) { if (id.x < 1) out[id.x] = 0.0; }
        "#,
);

/// An honest hand-written `GpuPod` over vectors: `Vec4` (16, align 16) then two `Vec2` at 16 and 24, size 32, no
/// padding.
#[derive(Clone, Copy)]
#[repr(C)]
struct HonestBody {
    a: gust::math::Vec4,
    b: gust::math::Vec2,
    c: gust::math::Vec2,
}

// SAFETY: `repr(C)` over three `repr(C)` vector types, no padding (16 + 8 + 8 = 32), offsets 0, 16, 24.
unsafe impl GpuPod for HonestBody {
    const LAYOUT: gust::TypeLayout = gust::TypeLayout::structure(
        "HonestBody",
        32,
        16,
        &[
            gust::StructFieldLayout {
                name: "a",
                offset: 0,
                ty: <gust::math::Vec4 as GpuPod>::LAYOUT,
            },
            gust::StructFieldLayout {
                name: "b",
                offset: 16,
                ty: <gust::math::Vec2 as GpuPod>::LAYOUT,
            },
            gust::StructFieldLayout {
                name: "c",
                offset: 24,
                ty: <gust::math::Vec2 as GpuPod>::LAYOUT,
            },
        ],
    );
}

#[test]
fn rejects_incorrect_binding_count_before_gpu_work() {
    assert!(matches!(
        validate_dispatch(&DESCRIPTOR, 0, &[]),
        Err(Error::BindingCount {
            expected: 1,
            actual: 0,
        })
    ));
}

#[test]
#[allow(deprecated)]
fn rejects_resource_layout_that_does_not_match_f32_backend() {
    let parameters = Box::leak(Box::new([ParameterDescriptor {
        resource_layout: Some(ResourceLayout::storage_v1(<u32 as GpuPod>::LAYOUT)),
        ..PARAMETERS[0]
    }]));
    let descriptor = DESCRIPTOR.with_parameters(parameters);
    assert!(matches!(
        validate_dispatch(&descriptor, 0, &[F32Binding::ReadWrite(&[])]),
        Err(Error::UnsupportedResourceLayout { parameter, .. }) if parameter == "out"
    ));
}

#[test]
fn renders_descriptor_specific_wgpu_source() {
    let source = render_wgpu_source(&DESCRIPTOR).expect("source should render");
    assert!(source.contains("pub struct TestKernelPipeline"));
    assert!(source.contains("impl TestKernelPipeline"));
    assert!(source.contains("pub fn encode("));
    assert!(source.contains("binding: 0,"));
    assert!(source.contains("read_only: false"));
    assert!(source.contains("gust::slang::compile_wgsl"));
    assert!(source.contains("gust::artifact::verify_artifact(descriptor, artifact)"));
    assert!(source.contains("artifact.format == gust::Format::Wgsl"));
    assert!(source.contains("out: f32 [storage-v1; size=4, align=4, stride=4]"));
    assert!(source.contains("pass.dispatch_workgroups(element_count.div_ceil(64), 1, 1);"));
    // A kernel with a one-dimensional workgroup takes the element count and nothing else (m7.d).
    assert!(!source.contains("extent"), "{source}");
}

#[test]
fn a_kernel_with_a_two_or_three_dimensional_workgroup_renders_a_host_that_takes_an_extent() {
    for (size, dispatch) in [
        (
            [16, 16, 1],
            "pass.dispatch_workgroups(extent[0].div_ceil(16), extent[1].div_ceil(16), extent[2].div_ceil(1));",
        ),
        (
            [4, 4, 4],
            "pass.dispatch_workgroups(extent[0].div_ceil(4), extent[1].div_ceil(4), extent[2].div_ceil(4));",
        ),
        (
            [8, 1, 2],
            "pass.dispatch_workgroups(extent[0].div_ceil(8), extent[1].div_ceil(1), extent[2].div_ceil(2));",
        ),
    ] {
        let descriptor = DESCRIPTOR.with_workgroup_size(size);
        let source = render_wgpu_source(&descriptor).expect("source should render");
        assert!(
            source.contains("        extent: [u32; 3],"),
            "{size:?}: {source}"
        );
        assert!(source.contains(dispatch), "{size:?}: {source}");
        assert!(!source.contains("element_count"), "{size:?}: {source}");
    }
    // A zero axis is refused, as for every dispatch.
    let flat = DESCRIPTOR.with_workgroup_size([16, 0, 1]);
    assert!(matches!(
        render_wgpu_source(&flat),
        Err(Error::UnsupportedWorkgroupSize([16, 0, 1]))
    ));
}

#[test]
fn the_workgroup_constant_is_wgpus_default() {
    assert_eq!(
        DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
        wgpu::Limits::default().max_compute_workgroups_per_dimension
    );
}

#[test]
fn the_resource_rule_follows_the_element() {
    use gust::math::{Mat2, Mat4, Vec2, Vec4};
    let scalar_struct = gust::TypeLayout::structure(
        "Pair",
        8,
        4,
        &[
            gust::StructFieldLayout {
                name: "a",
                offset: 0,
                ty: <u32 as GpuPod>::LAYOUT,
            },
            gust::StructFieldLayout {
                name: "b",
                offset: 4,
                ty: <f32 as GpuPod>::LAYOUT,
            },
        ],
    );
    for element in [
        <f32 as GpuPod>::LAYOUT,
        <u32 as GpuPod>::LAYOUT,
        scalar_struct,
    ] {
        assert_eq!(
            buffer_resource_layout(element),
            ResourceLayout::storage_v1(element),
            "{}",
            element.name
        );
    }
    for element in [
        <Vec2 as GpuPod>::LAYOUT,
        <Vec4 as GpuPod>::LAYOUT,
        <Mat2 as GpuPod>::LAYOUT,
        <Mat4 as GpuPod>::LAYOUT,
        <HonestBody as GpuPod>::LAYOUT,
    ] {
        assert_eq!(
            buffer_resource_layout(element),
            ResourceLayout::storage_v2(element),
            "{}",
            element.name
        );
    }
}

/// Slice m9.r2b (D159): array layouts pass the binding checks and a mismatch fails them, with the existing errors.
#[test]
fn array_layouts_pass_the_binding_checks_and_a_mismatch_fails() {
    use gust::math::{Mat4, Vec2};
    const F: gust::TypeLayout = <f32 as GpuPod>::LAYOUT;
    const PAIR: gust::TypeLayout = <[f32; 2] as GpuPod>::LAYOUT;
    const WITH_ARRAY: gust::TypeLayout = gust::TypeLayout::structure(
        "WithArray",
        24,
        4,
        &[
            gust::StructFieldLayout {
                name: "a",
                offset: 0,
                ty: <[f32; 4] as GpuPod>::LAYOUT,
            },
            gust::StructFieldLayout {
                name: "p",
                offset: 16,
                ty: PAIR,
            },
        ],
    );
    // The assertion every typed allocation runs: arrays of honest elements are accepted.
    assert_pod_matches_storage_layout::<[f32; 4]>();
    assert_pod_matches_storage_layout::<[Vec2; 3]>();
    assert_pod_matches_storage_layout::<[Mat4; 2]>();
    for layout in [
        <[f32; 4] as GpuPod>::LAYOUT,
        <[Vec2; 3] as GpuPod>::LAYOUT,
        <[Mat4; 2] as GpuPod>::LAYOUT,
        WITH_ARRAY,
    ] {
        assert!(padding_free(layout), "{}", layout.name);
    }
    // A wrong total, and an element whose stride is not its size, are not padding-free.
    let mut wrong = <[Vec2; 3] as GpuPod>::LAYOUT;
    wrong.size = 32;
    assert!(!padding_free(wrong));
    const PADDED_ELEMENT: gust::TypeLayout = gust::TypeLayout {
        name: "P",
        size: 12,
        alignment: 4,
        stride: 16,
        kind: gust::TypeLayoutKind::Scalar(gust::ScalarKind::U32),
    };
    const PADDED_ARRAY: gust::TypeLayout = gust::TypeLayout {
        name: "array",
        size: 24,
        alignment: 4,
        stride: 24,
        kind: gust::TypeLayoutKind::Array {
            element: &PADDED_ELEMENT,
            count: 2,
        },
    };
    assert!(!padding_free(PADDED_ARRAY));
    // The resource rule: a float array is StorageV1, one with vectors StorageV2.
    assert_eq!(
        buffer_resource_layout(<[f32; 4] as GpuPod>::LAYOUT),
        ResourceLayout::storage_v1(<[f32; 4] as GpuPod>::LAYOUT)
    );
    assert_eq!(
        buffer_resource_layout(WITH_ARRAY),
        ResourceLayout::storage_v1(WITH_ARRAY)
    );
    assert_eq!(
        buffer_resource_layout(<[Vec2; 3] as GpuPod>::LAYOUT),
        ResourceLayout::storage_v2(<[Vec2; 3] as GpuPod>::LAYOUT)
    );
    // The binding check: the descriptor's layout against the buffer's.
    let four = binding_resource_layout(BindingKind::Storage, <[f32; 4] as GpuPod>::LAYOUT);
    let parameter = ParameterDescriptor {
        resource_layout: Some(four),
        ..PARAMETERS[0]
    };
    assert!(validate_binding_layout(&parameter, four).is_ok());
    for actual in [
        // A different count, a different element, a scalar, a struct holding the array.
        binding_resource_layout(BindingKind::Storage, <[f32; 3] as GpuPod>::LAYOUT),
        binding_resource_layout(BindingKind::Storage, <[u32; 4] as GpuPod>::LAYOUT),
        binding_resource_layout(BindingKind::Storage, F),
        binding_resource_layout(BindingKind::Storage, WITH_ARRAY),
    ] {
        assert!(
            matches!(
                validate_binding_layout(&parameter, actual),
                Err(Error::UnsupportedResourceLayout { .. })
            ),
            "{actual:?}"
        );
    }
    // An array is not a constant-buffer type.
    assert!(check_uniform_type::<[f32; 4]>().is_err());
}

/// `DESCRIPTOR` with its one storage parameter relabelled.
fn descriptor_with(resource_layout: ResourceLayout) -> KernelDescriptor {
    let parameters = Box::leak(Box::new([ParameterDescriptor {
        resource_layout: Some(resource_layout),
        ..PARAMETERS[0]
    }]));
    DESCRIPTOR.with_parameters(parameters)
}

/// Slice mx.p1b: a `pixel` kernel is valid (it runs as a render pass); the callers that only know compute passes
/// refuse it with a typed error before any compile or pipeline (no device is involved), not a wgpu panic.
#[test]
fn a_pixel_kernel_is_valid_and_the_compute_only_callers_refuse_it_by_name() {
    let pixel = descriptor_with(ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT))
        .with_shape(Shape::Pixel)
        .with_compatible_shapes(&[Shape::Pixel]);
    assert!(validate_kernel(&pixel).is_ok());
    #[cfg(not(target_arch = "wasm32"))]
    {
        let error = validate_dispatch(&pixel, 0, &[]).unwrap_err();
        assert!(
            matches!(
                error,
                Error::ShapeUnsupported {
                    shape: Shape::Pixel,
                    ..
                }
            ),
            "{error:?}"
        );
        let text = error.to_string();
        assert!(
            text.contains("Pixel") && text.contains("one-shot"),
            "{text}"
        );
        assert!(matches!(
            validate_u32_dispatch(&pixel, 0, &[]).unwrap_err(),
            Error::ShapeUnsupported { .. }
        ));
    }
    assert!(
        validate_kernel(&descriptor_with(ResourceLayout::storage_v1(
            <f32 as GpuPod>::LAYOUT
        )))
        .is_ok()
    );
}

#[test]
fn a_pixel_call_over_more_than_one_dimension_is_refused_before_any_binding_is_looked_at() {
    let pixel = descriptor_with(ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT))
        .with_shape(Shape::Pixel)
        .with_compatible_shapes(&[Shape::Pixel]);
    for extent in [Extent::new(4, 2, 1), Extent::new(4, 1, 2)] {
        let error = validate_dispatch_extent(0, false, &pixel, extent, &[], &[]).unwrap_err();
        assert!(
            matches!(
                error,
                Error::ShapeUnsupported {
                    shape: Shape::Pixel,
                    ..
                }
            ),
            "{extent:?}: {error:?}"
        );
    }
}

#[test]
fn raw_compares_the_resource_and_a_buffer_never_equals_a_texture() {
    let device = HeadlessDevice::new().expect("a GPU adapter");
    let first = device.create_typed_buffer("raw first", &[0.0], GpuBufferAccess::ReadWrite);
    let second = device.create_typed_buffer("raw second", &[0.0], GpuBufferAccess::ReadWrite);
    let texture = device
        .create_texture_2d::<gust::texture::R32Float>(2, 2, gust::texture::TextureUse::SAMPLED)
        .expect("a texture");
    let other = device
        .create_texture_2d::<gust::texture::R32Float>(2, 2, gust::texture::TextureUse::SAMPLED)
        .expect("a texture");
    assert!(Raw::Buffer(&first.raw) == Raw::Buffer(&first.raw));
    assert!(Raw::Buffer(&first.raw) != Raw::Buffer(&second.raw));
    assert!(Raw::Texture(texture.as_wgpu_texture()) == Raw::Texture(texture.as_wgpu_texture()));
    assert!(Raw::Texture(texture.as_wgpu_texture()) != Raw::Texture(other.as_wgpu_texture()));
    assert!(Raw::Buffer(&first.raw) != Raw::Texture(texture.as_wgpu_texture()));
}

#[test]
fn a_kernel_is_valid_only_with_the_rule_that_its_element_implies() {
    use gust::math::{Mat4, Vec4};
    for (label, valid) in [
        (ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT), true),
        (ResourceLayout::storage_v2(<Vec4 as GpuPod>::LAYOUT), true),
        (ResourceLayout::storage_v2(<Mat4 as GpuPod>::LAYOUT), true),
        // A V2 claim on a scalar, a V1 claim on a vector and a constant-buffer rule on a storage parameter.
        (ResourceLayout::storage_v2(<f32 as GpuPod>::LAYOUT), false),
        (ResourceLayout::storage_v1(<Vec4 as GpuPod>::LAYOUT), false),
        (ResourceLayout::std140_v1(<Vec4 as GpuPod>::LAYOUT), false),
        (ResourceLayout::std140_v1(<f32 as GpuPod>::LAYOUT), false),
    ] {
        let result = validate_kernel(&descriptor_with(label));
        if valid {
            assert!(result.is_ok(), "{label:?}");
        } else {
            assert!(
                matches!(result, Err(Error::InvalidResourceLayout { .. })),
                "{label:?}"
            );
        }
    }
}

#[test]
#[allow(deprecated)]
fn a_vector_buffer_does_not_match_the_f32_backend() {
    let descriptor = descriptor_with(ResourceLayout::storage_v2(
        <gust::math::Vec4 as GpuPod>::LAYOUT,
    ));
    assert!(matches!(
        validate_dispatch(&descriptor, 0, &[F32Binding::ReadWrite(&[])]),
        Err(Error::UnsupportedResourceLayout { parameter, .. }) if parameter == "out"
    ));
}

#[test]
fn the_generated_host_text_names_the_storage_v2_rule_of_a_vector_buffer() {
    let descriptor = descriptor_with(ResourceLayout::storage_v2(
        <gust::math::Vec4 as GpuPod>::LAYOUT,
    ));
    let source = render_wgpu_source(&descriptor).expect("source should render");
    assert!(
        source.contains("out: Vec4 [storage-v2; size=16, align=16, stride=16]"),
        "{source}"
    );
}

/// `DESCRIPTOR` with its one parameter replaced by a uniform of the given access and layout.
fn uniform_descriptor(access: Access, resource_layout: ResourceLayout) -> KernelDescriptor {
    let parameters = Box::leak(Box::new([ParameterDescriptor {
        name: "factor",
        rust_type: "Uniform<Vec4>",
        kind: ParameterKind::Uniform,
        access,
        resource_layout: Some(resource_layout),
        binding: Some(ResourceBinding {
            group: 0,
            binding: 0,
        }),
    }]));
    DESCRIPTOR.with_parameters(parameters)
}

#[test]
fn a_uniform_parameter_is_valid_only_as_a_read_only_std140_v1_constant_buffer() {
    use gust::math::{Mat2, Mat4, Vec4};
    for (label, valid) in [
        (ResourceLayout::std140_v1(<Vec4 as GpuPod>::LAYOUT), true),
        (ResourceLayout::std140_v1(<Mat4 as GpuPod>::LAYOUT), true),
        (ResourceLayout::std140_v1(<f32 as GpuPod>::LAYOUT), true),
        // The storage rules on a uniform, even for an element that is valid under both.
        (ResourceLayout::storage_v2(<Vec4 as GpuPod>::LAYOUT), false),
        (ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT), false),
        // A constant buffer has no 2x2 matrix.
        (ResourceLayout::std140_v1(<Mat2 as GpuPod>::LAYOUT), false),
    ] {
        let result = validate_kernel(&uniform_descriptor(Access::ReadOnly, label));
        if valid {
            assert!(result.is_ok(), "{label:?}");
        } else {
            assert!(
                matches!(result, Err(Error::InvalidResourceLayout { .. })),
                "{label:?}"
            );
        }
    }
    assert!(matches!(
        validate_kernel(&uniform_descriptor(
            Access::ReadWrite,
            ResourceLayout::std140_v1(<Vec4 as GpuPod>::LAYOUT)
        )),
        Err(Error::BindingAccess {
            parameter,
            expected: Access::ReadOnly,
            actual: Access::ReadWrite,
        }) if parameter == "factor"
    ));
}

#[test]
fn a_parameter_that_is_neither_storage_nor_uniform_is_still_unsupported() {
    let parameters = Box::leak(Box::new([ParameterDescriptor {
        kind: ParameterKind::Value,
        ..PARAMETERS[0]
    }]));
    let descriptor = DESCRIPTOR.with_parameters(parameters);
    assert!(matches!(
        validate_kernel(&descriptor),
        Err(Error::UnsupportedParameter { name, kind: ParameterKind::Value }) if name == "out"
    ));
}

#[test]
fn the_generated_host_text_names_the_std140_rule_and_the_uniform_binding_of_a_uniform() {
    let descriptor = uniform_descriptor(
        Access::ReadOnly,
        ResourceLayout::std140_v1(<gust::math::Vec4 as GpuPod>::LAYOUT),
    );
    let source = render_wgpu_source(&descriptor).expect("source should render");
    assert!(
        source.contains("factor: Vec4 [std140-v1; size=16, align=16, stride=16]"),
        "{source}"
    );
    assert!(
        source.contains("ty: wgpu::BufferBindingType::Uniform,"),
        "{source}"
    );
    assert!(!source.contains("BufferBindingType::Storage"), "{source}");
}
