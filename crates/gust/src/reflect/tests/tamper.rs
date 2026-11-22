use super::{fragment_support::*, *};
use crate::TextureLayout;

// ---- the tamper cases (plan section 3.3): every one is a `Mismatch` that names the field ----

#[test]
fn the_untouched_schema_three_documents_pass_their_cross_checks() {
    let compute = texture_kernel();
    let compiled = read(&texture_document(), &compute).unwrap();
    assert!(compiled.reflection.cross_check_kernel(&compute).is_ok());
    let fragment = fragment_kernel();
    let compiled = read(&fragment_document(), &fragment).unwrap();
    assert!(compiled.reflection.cross_check_kernel(&fragment).is_ok());
}

#[test]
fn a_stage_that_differs_from_the_shape_is_a_mismatch_naming_stage() {
    // A compute reflection against a pixel descriptor.
    let pixel = fragment_kernel();
    let compute_document = document3("compute", &fragment_parameters(), &[]);
    assert_eq!(
        any_mismatch(&compute_document, &pixel),
        "stage: host fragment, compiler compute"
    );
    // The same at the cross-check, which does not depend on the read having caught it: the reflection was read for
    // a grid kernel and is checked against a pixel one.
    let grid = kernel_with(Vec::new());
    let compiled = read(&document3("compute", &[], &[]), &grid).unwrap();
    assert_eq!(
        compiled
            .reflection
            .cross_check_kernel(&pixel)
            .unwrap_err()
            .to_string(),
        "POD reflection mismatch: stage: host fragment, compiler compute"
    );
    // A fragment reflection against a grid descriptor, at the read and at the cross-check.
    assert_eq!(
        any_mismatch(&fragment_document(), &texture_kernel()),
        "stage: host compute, compiler fragment"
    );
    let compiled = read(&fragment_document(), &fragment_kernel()).unwrap();
    assert_eq!(
        compiled
            .reflection
            .cross_check_kernel(&texture_kernel())
            .unwrap_err()
            .to_string(),
        "POD reflection mismatch: stage: host compute, compiler fragment"
    );
    // A schema-1 and a schema-2 reflection are compute reflections.
    let compiled = read(&document(2, Some("column"), &[]), &grid).unwrap();
    assert_eq!(
        compiled
            .reflection
            .cross_check_kernel(&pixel)
            .unwrap_err()
            .to_string(),
        "POD reflection mismatch: stage: host fragment, compiler compute"
    );
}

#[test]
fn a_target_index_or_type_that_differs_is_a_mismatch_naming_the_target() {
    let kernel = fragment_kernel();
    let parameters = fragment_parameters();
    for (what, targets, expected) in [
        (
            "the index",
            vec![target_json(1, F4)],
            "targets[0].index: host 0, compiler 1",
        ),
        (
            "the scalar type",
            vec![target_json(0, ("uint32", 4))],
            "targets[0].type: host f32x4, compiler u32x4",
        ),
        (
            "the width",
            vec![target_json(0, ("float32", 2))],
            "targets[0].type: host f32x4, compiler f32x2",
        ),
        (
            "a scalar for a vector",
            vec![target_json(0, ("float32", 1))],
            "targets[0].type: host f32x4, compiler f32",
        ),
        (
            "an extra target",
            vec![target_json(0, F4), target_json(1, F4)],
            "targets: host 1 outputs, compiler 2",
        ),
        (
            "a missing target",
            Vec::new(),
            "targets: host 1 outputs, compiler 0",
        ),
    ] {
        let json = document3("fragment", &parameters, &targets);
        assert_eq!(any_mismatch(&json, &kernel), expected, "{what}");
    }
    // A gap in the indices (the helper reports `SV_Target0` and `SV_Target2`) is not outputs in parameter order.
    let two = pixel_kernel(vec![
        output_parameter("a", U32_LAYOUT),
        output_parameter("b", VEC4),
    ]);
    let json = document3(
        "fragment",
        &[],
        &[target_json(0, ("uint32", 1)), target_json(2, F4)],
    );
    assert_eq!(
        any_mismatch(&json, &two),
        "targets[1].index: host 1, compiler 2"
    );
    // Swapping the two outputs' order is a type mismatch at both positions' first.
    let swapped = pixel_kernel(vec![
        output_parameter("b", VEC4),
        output_parameter("a", U32_LAYOUT),
    ]);
    let json = document3(
        "fragment",
        &[],
        &[target_json(0, ("uint32", 1)), target_json(1, F4)],
    );
    assert_eq!(
        any_mismatch(&json, &swapped),
        "targets[0].type: host f32x4, compiler u32"
    );
}

#[test]
fn a_texture_that_differs_is_a_mismatch_naming_the_field() {
    let kernel = texture_kernel();
    let control = texture_parameters();
    let with = |index: usize, parameter: String| {
        let mut parameters = control.clone();
        parameters[index] = parameter;
        document3("compute", &parameters, &[])
    };
    for (what, json, expected) in [
        (
            "access of a read-write texture",
            with(1, texture_json("rw", 1, "write", ("float32", 1), "r32f")),
            "rw.access: host readWrite, compiler write",
        ),
        (
            "access of a write-only texture",
            with(
                2,
                texture_json("wo", 2, "readWrite", ("uint32", 2), "rg32ui"),
            ),
            "wo.access: host write, compiler readWrite",
        ),
        (
            "access of a sampled texture",
            with(0, texture_json("tex", 0, "readWrite", F4, "unknown")),
            "tex.access: host read, compiler readWrite",
        ),
        (
            "format of a read-write texture",
            with(
                1,
                texture_json("rw", 1, "readWrite", ("float32", 1), "rgba32f"),
            ),
            "rw.format: host r32float, compiler rgba32float",
        ),
        (
            "format of a write-only texture",
            with(2, texture_json("wo", 2, "write", ("uint32", 2), "rg32i")),
            "wo.format: host rg32uint, compiler rg32sint",
        ),
        (
            "format of a storage texture reported as unknown",
            with(
                1,
                texture_json("rw", 1, "readWrite", ("float32", 1), "unknown"),
            ),
            "rw.format: host r32float, compiler none",
        ),
        (
            "a format on a sampled texture",
            with(0, texture_json("tex", 0, "read", F4, "r32f")),
            "tex.format: host none, compiler r32float",
        ),
        (
            "result type of a storage texture",
            with(
                1,
                texture_json("rw", 1, "readWrite", ("float32", 2), "r32f"),
            ),
            "rw.result: host f32, compiler f32x2",
        ),
        (
            "result scalar of a write-only texture",
            with(2, texture_json("wo", 2, "write", ("int32", 2), "rg32ui")),
            "wo.result: host u32x2, compiler i32x2",
        ),
        (
            "sample class of a sampled texture",
            with(0, texture_json("tex", 0, "read", ("uint32", 4), "unknown")),
            "tex.class: host f32, compiler u32",
        ),
        (
            "result width of a sampled texture",
            with(0, texture_json("tex", 0, "read", ("float32", 1), "unknown")),
            "tex.result: host f32x4, compiler f32",
        ),
        (
            "slot of a texture",
            with(
                0,
                texture_json("tex", 0, "read", F4, "unknown").replacen(
                    r#""index":0"#,
                    r#""index":6"#,
                    1,
                ),
            ),
            "tex.binding: host 0, compiler 6",
        ),
        (
            "space of a texture",
            with(
                0,
                texture_json("tex", 0, "read", F4, "unknown").replacen(
                    r#""space":0"#,
                    r#""space":1"#,
                    1,
                ),
            ),
            "tex.group: host 0, compiler 1",
        ),
        (
            "a texture the document reports as a buffer",
            with(0, storage_json("tex", 0, F32_LAYOUT, "read")),
            "tex.kind: host texture, compiler a buffer",
        ),
        (
            "a texture the document reports as a sampler",
            with(0, sampler_json("tex", 0)),
            "tex.kind: host texture, compiler a sampler",
        ),
        (
            "a texture the document names otherwise",
            with(0, texture_json("tex2", 0, "read", F4, "unknown")),
            "missing compiler binding tex",
        ),
    ] {
        assert_eq!(any_mismatch(&json, &kernel), expected, "{what}");
    }
    // A texture the descriptor does not have.
    let mut parameters = texture_parameters();
    parameters.push(texture_json("extra", 7, "read", F4, "unknown"));
    assert_eq!(
        any_mismatch(&document3("compute", &parameters, &[]), &kernel),
        "unexpected compiler resource bindings"
    );
}

#[test]
fn a_sampler_that_differs_is_a_mismatch_naming_the_field() {
    let kernel = texture_kernel();
    let control = texture_parameters();
    let with = |parameter: String| {
        let mut parameters = control.clone();
        parameters[3] = parameter;
        document3("compute", &parameters, &[])
    };
    for (what, json, expected) in [
        (
            "a moved slot",
            with(sampler_json("samp", 5)),
            "samp.binding: host 3, compiler 5",
        ),
        (
            "a moved space",
            with(sampler_json("samp", 3).replacen(r#""space":0"#, r#""space":2"#, 1)),
            "samp.group: host 0, compiler 2",
        ),
        (
            "a sampler the document reports as a texture",
            with(texture_json("samp", 3, "read", F4, "unknown")),
            "samp.kind: host sampler, compiler a texture",
        ),
        (
            "a sampler the document names otherwise",
            with(sampler_json("other", 3)),
            "missing compiler binding samp",
        ),
    ] {
        assert_eq!(any_mismatch(&json, &kernel), expected, "{what}");
    }
    let mut parameters = texture_parameters();
    parameters.push(sampler_json("extra", 7));
    assert_eq!(
        any_mismatch(&document3("compute", &parameters, &[]), &kernel),
        "unexpected compiler resource bindings"
    );
    // A descriptor with a sampler the document lacks.
    let mut parameters = texture_parameters();
    parameters.remove(3);
    assert_eq!(
        any_mismatch(&document3("compute", &parameters, &[]), &kernel),
        "missing compiler binding samp"
    );
}

#[test]
fn a_descriptor_that_contradicts_itself_is_a_mismatch_even_against_its_own_document() {
    let document = texture_document();
    let kernel_with_tex = |texture: ParameterDescriptor| {
        let mut kernel = texture_kernel();
        let mut parameters = kernel.parameters.to_vec();
        parameters[0] = texture;
        kernel.parameters = Box::leak(parameters.into_boxed_slice());
        kernel
    };
    let kernel_with_rw = |texture: ParameterDescriptor| {
        let mut kernel = texture_kernel();
        let mut parameters = kernel.parameters.to_vec();
        parameters[1] = texture;
        kernel.parameters = Box::leak(parameters.into_boxed_slice());
        kernel
    };
    let rgba = storage(TexelFormat::Rgba32Float, TextureAccess::ReadWrite);
    let cases: Vec<(&str, KernelDescriptor, &str)> = vec![
        (
            "a class that is not the result type's",
            kernel_with_tex(texture_parameter(
                "tex",
                Access::ReadOnly,
                VEC4,
                sampled(SampleClass::Uint),
                0,
            )),
            "tex.class: host u32, result type f32",
        ),
        (
            "a format on a sampled texture",
            kernel_with_tex(texture_parameter(
                "tex",
                Access::ReadOnly,
                VEC4,
                TextureLayout {
                    format: Some(TexelFormat::Rgba32Float),
                    ..sampled(SampleClass::Float)
                },
                0,
            )),
            "tex.format: a sampled texture has no format in the shader, host rgba32float",
        ),
        (
            "a storage texture without a format",
            kernel_with_rw(texture_parameter(
                "rw",
                Access::ReadWrite,
                VEC4,
                TextureLayout {
                    format: None,
                    ..rgba
                },
                1,
            )),
            "rw.format: a storage texture needs a format",
        ),
        (
            "a read-write texture of a wide format",
            kernel_with_rw(texture_parameter("rw", Access::ReadWrite, VEC4, rgba, 1)),
            "rw.format: rgba32float does not fit class f32 and access readWrite",
        ),
        (
            "a format of another class",
            kernel_with_rw(texture_parameter(
                "rw",
                Access::ReadWrite,
                VEC4,
                TextureLayout {
                    format: Some(TexelFormat::Rgba32Uint),
                    access: TextureAccess::WriteOnly,
                    ..rgba
                },
                1,
            )),
            "rw.format: rgba32uint does not fit class f32 and access write",
        ),
        (
            "a sampled texture with write access",
            kernel_with_tex(texture_parameter(
                "tex",
                Access::ReadWrite,
                VEC4,
                sampled(SampleClass::Float),
                0,
            )),
            "tex.access differs",
        ),
        (
            "a texture with a buffer rule",
            kernel_with_tex(parameter(
                "tex",
                ParameterKind::Texture,
                Access::ReadOnly,
                ResourceLayout::storage_v2(VEC4),
                0,
            )),
            "tex.rules: StorageV2 does not fit this parameter kind",
        ),
        (
            "a texture result that is a matrix",
            kernel_with_tex(texture_parameter(
                "tex",
                Access::ReadOnly,
                MAT4,
                sampled(SampleClass::Float),
                0,
            )),
            "tex.result: a texture result is a 32-bit scalar or a 2- or 4-wide vector",
        ),
        (
            "a texture result that is three wide",
            kernel_with_tex(texture_parameter(
                "tex",
                Access::ReadOnly,
                TypeLayout::vector("Vec3", ScalarKind::F32, 3),
                sampled(SampleClass::Float),
                0,
            )),
            "tex.result: a texture result is a 32-bit scalar or a 2- or 4-wide vector",
        ),
        (
            "a format with more channels than the result type has",
            kernel_with_rw(texture_parameter(
                "rw",
                Access::ReadWrite,
                UVEC2,
                TextureLayout {
                    format: Some(TexelFormat::Rgba32Uint),
                    access: TextureAccess::WriteOnly,
                    ..storage(TexelFormat::Rgba32Uint, TextureAccess::WriteOnly)
                },
                1,
            )),
            "rw.format: rgba32uint has channel count 4, the result type has width 2",
        ),
        (
            "a format with fewer channels than the result type has",
            kernel_with_rw(texture_parameter(
                "rw",
                Access::ReadWrite,
                VEC4,
                storage(TexelFormat::R32Float, TextureAccess::ReadWrite),
                1,
            )),
            "rw.format: r32float has channel count 1, the result type has width 4",
        ),
        (
            "a sampler with a layout",
            {
                let mut kernel = texture_kernel();
                let mut parameters = kernel.parameters.to_vec();
                parameters[3].resource_layout = Some(ResourceLayout::storage_v1(F32_LAYOUT));
                kernel.parameters = Box::leak(parameters.into_boxed_slice());
                kernel
            },
            "samp: a sampler has no resource contract",
        ),
    ];
    for (what, kernel, expected) in cases {
        assert_eq!(any_mismatch(&document, &kernel), expected, "{what}");
    }
    // A buffer parameter that carries the texture rule.
    let buffer = kernel_with(vec![parameter(
        "out",
        ParameterKind::Storage,
        Access::ReadWrite,
        ResourceLayout::texture_v1(F32_LAYOUT, sampled(SampleClass::Float)),
        0,
    )]);
    let json = document3(
        "compute",
        &[storage_json("out", 0, F32_LAYOUT, "readWrite")],
        &[],
    );
    assert!(
        any_mismatch(&json, &buffer).starts_with("out.rules: TextureV1("),
        "{}",
        any_mismatch(&json, &buffer)
    );
    // Two descriptor bindings in one slot, across kinds.
    let clash = kernel_with(vec![
        texture_parameter(
            "tex",
            Access::ReadOnly,
            VEC4,
            sampled(SampleClass::Float),
            0,
        ),
        sampler_parameter("samp", 0),
    ]);
    assert_eq!(
        any_mismatch(
            &document3(
                "compute",
                &[
                    texture_json("tex", 0, "read", F4, "unknown"),
                    sampler_json("samp", 1)
                ],
                &[]
            ),
            &clash
        ),
        "duplicate descriptor binding"
    );
}

#[test]
fn a_pixel_kernel_may_have_no_outputs_in_its_descriptor_only_if_the_document_has_none() {
    let no_outputs = pixel_kernel(vec![texture_parameter(
        "tex",
        Access::ReadOnly,
        VEC4,
        sampled(SampleClass::Float),
        0,
    )]);
    let json = document3(
        "fragment",
        &[texture_json("tex", 0, "read", F4, "unknown")],
        &[],
    );
    let compiled = read(&json, &no_outputs).unwrap();
    assert!(compiled.reflection.cross_check_kernel(&no_outputs).is_ok());
    assert_eq!(
        any_mismatch(&fragment_document(), &no_outputs),
        "unexpected compiler resource bindings"
    );
}
