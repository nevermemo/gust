use gust::{
    Access, GpuPod, KernelDescriptor, ParameterDescriptor, ParameterKind, ResourceBinding,
    ResourceLayout, reflect::Error as ReflectionError, slang::Target,
};
use gust_wgpu::{BufferBinding, Error, GpuBufferAccess, HeadlessDevice};

const PARAMETERS: &[ParameterDescriptor] = &[ParameterDescriptor {
    name: "output",
    rust_type: "RWStructuredBuffer<float>",
    kind: ParameterKind::Storage,
    access: Access::ReadWrite,
    resource_layout: Some(ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT)),
    binding: Some(ResourceBinding {
        group: 0,
        binding: 0,
    }),
}];

const KERNEL: KernelDescriptor = KernelDescriptor::new(
    "reflection_test",
    "write",
    "write_value",
    [64, 1, 1],
    PARAMETERS,
    r#"
        [[vk::binding(0, 0)]] RWStructuredBuffer<float> output;
        [shader("compute")] [numthreads(64, 1, 1)]
        void write_value(uint3 id : SV_DispatchThreadID) {
            if (id.x == 0) output[0] = 17.0;
        }
    "#,
);

fn assert_mismatch(error: Error, expected: &str) {
    let Error::Reflection(ReflectionError::Mismatch(message)) = error else {
        panic!("expected a reflection mismatch, got {error}");
    };
    assert_eq!(message, expected);
}

#[test]
fn rejects_workgroup_mismatch_before_dispatch() {
    let device = HeadlessDevice::new().expect("Vulkan adapter required");
    let output = device.create_f32_buffer("reflection gate", &[3.0], GpuBufferAccess::ReadWrite);
    let descriptor = KERNEL.with_workgroup_size([32, 1, 1]);
    let error = device
        .dispatch_buffers(&descriptor, 1, &[BufferBinding::read_write(&output)])
        .expect_err("descriptor/shader workgroup mismatch must fail before dispatch");
    assert_mismatch(error, "compute workgroup dimensions differ");
    assert_eq!(device.read_f32_buffer(&output).unwrap(), [3.0]);
    let stats = device.pipeline_cache_stats();
    assert_eq!((stats.entries, stats.misses, stats.hits), (0, 0, 0));
}

#[test]
fn rejects_resource_mismatches_before_dispatch() {
    let device = HeadlessDevice::new().expect("Vulkan adapter required");
    let floats = device.create_f32_buffer("reflection floats", &[3.0], GpuBufferAccess::ReadWrite);
    let integers =
        device.create_typed_buffer("reflection uints", &[5u32], GpuBufferAccess::ReadWrite);
    for (parameter, binding, expected) in [
        (
            ParameterDescriptor {
                name: "renamed",
                ..PARAMETERS[0]
            },
            BufferBinding::read_write(&floats),
            "missing compiler binding renamed",
        ),
        (
            ParameterDescriptor {
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: 1,
                }),
                ..PARAMETERS[0]
            },
            BufferBinding::read_write(&floats),
            "output.binding: host 1, compiler 0",
        ),
        (
            ParameterDescriptor {
                access: Access::ReadOnly,
                ..PARAMETERS[0]
            },
            BufferBinding::read_only(&floats),
            "output.access differs",
        ),
        (
            ParameterDescriptor {
                resource_layout: Some(ResourceLayout::storage_v1(<u32 as GpuPod>::LAYOUT)),
                ..PARAMETERS[0]
            },
            BufferBinding::read_write(&integers),
            "output: type or field count differs",
        ),
    ] {
        let descriptor = KERNEL.with_parameters(Box::leak(Box::new([parameter])));
        let error = device
            .dispatch_buffers(&descriptor, 1, &[binding])
            .expect_err("resource mismatch must fail before dispatch");
        assert_mismatch(error, expected);
    }
    let descriptor = KERNEL.with_parameters(&[]);
    assert_mismatch(
        device.dispatch_buffers(&descriptor, 1, &[]).unwrap_err(),
        "unexpected compiler resource bindings",
    );
    assert_eq!(device.read_f32_buffer(&floats).unwrap(), [3.0]);
    assert_eq!(device.read_typed_buffer(&integers).unwrap(), [5]);
    let stats = device.pipeline_cache_stats();
    assert_eq!((stats.entries, stats.misses, stats.hits), (0, 0, 0));
}

#[test]
fn cached_pipeline_does_not_authorize_a_changed_contract() {
    let device = HeadlessDevice::new().expect("Vulkan adapter required");
    let output = device.create_f32_buffer("cached reflection", &[3.0], GpuBufferAccess::ReadWrite);
    let bindings = [BufferBinding::read_write(&output)];
    device.dispatch_buffers(&KERNEL, 1, &bindings).unwrap();
    let descriptor = KERNEL.with_workgroup_size([32, 1, 1]);
    assert_mismatch(
        device
            .dispatch_buffers(&descriptor, 1, &bindings)
            .unwrap_err(),
        "compute workgroup dimensions differ",
    );
    let descriptor = KERNEL.with_parameters(Box::leak(Box::new([ParameterDescriptor {
        name: "renamed",
        ..PARAMETERS[0]
    }])));
    assert_mismatch(
        device
            .dispatch_buffers(&descriptor, 1, &bindings)
            .unwrap_err(),
        "missing compiler binding renamed",
    );
    device.dispatch_buffers(&KERNEL, 1, &bindings).unwrap();
    assert_eq!(device.read_f32_buffer(&output).unwrap(), [17.0]);
    let stats = device.pipeline_cache_stats();
    assert_eq!((stats.entries, stats.misses, stats.hits), (1, 1, 1));
}

/// A kernel that rustc and the GUST validator accept and only Slang rejects: a
/// local named `new` (M1.3, the Reflect-tier check that the native helper honours
/// the compile copy's `#line` directives). The macro now emits that local as `__gust_r_new`, which slangc takes, so
/// the test deletes the prefix from the Slang text and keeps the macro's line table: the bare `new` is the
/// Slang-only error, on the same lines.
#[gust::gpu]
mod rejected_by_slang {
    #[kernel]
    pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<uint>) {
        if id.x < out.len() {
            let new = id.x;
            out[id.x] = new; // FAILS: native helper
        }
    }
}

#[test]
fn the_native_helper_reports_a_slang_failure_at_the_rust_line() {
    let marker = "// FAILS: native helper";
    let expected = include_str!("reflection.rs")
        .lines()
        .position(|line| line.trim_end().ends_with(marker) && line.contains("out[id.x]"))
        .map(|index| index + 1)
        .expect("marker line");
    let renamed = rejected_by_slang::run::DESCRIPTOR;
    assert!(renamed.source.file.ends_with("reflection.rs"));
    let bare = renamed.slang_source.replace("__gust_r_", "");
    assert_ne!(
        bare, renamed.slang_source,
        "the macro renamed the local `new`"
    );
    let kernel = renamed.with_slang_source(Box::leak(bare.into_boxed_str()));
    for target in [Target::Wgsl, Target::Spirv] {
        let error = gust::reflect::compile_reflected(&kernel, target).unwrap_err();
        let gust::reflect::Error::Compiler(gust::slang::Error::CompilationFailed {
            diagnostics,
            ..
        }) = error
        else {
            panic!("{target:?}: expected the helper to refuse the source, got {error}");
        };
        // The helper's own error location is the Rust file and line ...
        let location = format!("--> {}:{expected}:", kernel.source.file);
        assert!(
            diagnostics.contains(&location),
            "{target:?}: no `{location}` in:\n{diagnostics}"
        );
        // ... and the annotation names the same line, the kernel and the construct.
        assert!(
            diagnostics.contains(&format!(
                "originating Rust source: {}:{expected}\n",
                kernel.source.file
            )),
            "{target:?}: {diagnostics}"
        );
        assert!(
            diagnostics.contains("originating Rust kernel: rejected_by_slang::run"),
            "{target:?}: {diagnostics}"
        );
        assert!(
            diagnostics.contains("originating Rust construct: assignment #"),
            "{target:?}: {diagnostics}"
        );
        assert!(
            diagnostics.contains("reflection helper: "),
            "{target:?}: {diagnostics}"
        );
    }
    // Without a table the helper reads the clean text and reports Slang coordinates.
    let mut clean = kernel;
    clean.source = gust::SourceOrigin::UNKNOWN;
    let error = gust::reflect::compile_reflected(&clean, Target::Wgsl).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("kernel.slang:"), "{text}");
    assert!(!text.contains("reflection.rs"), "{text}");
}

/// A hand-written descriptor (no macro, no validator) whose Slang has an
/// undeclared identifier on line 4, with a two-record line table whose faulty
/// record points at the marked line below (M1 risk 5, native helper leg).
const HANDWRITTEN_FAILING: KernelDescriptor = KernelDescriptor::new(
    "reflection_test",
    "handwritten_failure",
    "handwritten_write",
    [64, 1, 1],
    PARAMETERS,
    concat!(
        "[[vk::binding(0, 0)]] RWStructuredBuffer<float> output;\n",
        "[shader(\"compute\")] [numthreads(64, 1, 1)]\n",
        "void handwritten_write(uint3 id : SV_DispatchThreadID) {\n",
        "    output[id.x] = missing_value;\n",
        "}\n",
    ),
)
.with_source(gust::SourceOrigin {
    file: "crates/gust-wgpu/tests/reflection.rs",
    lines: &[
        gust::LineSpan {
            slang_start: 3,
            slang_end: 5,
            rust_line: line!(),
        },
        gust::LineSpan {
            slang_start: 4,
            slang_end: 4,
            rust_line: line!(), // FAILS: native handwritten
        },
    ],
});

#[test]
fn the_native_helper_reports_a_handwritten_failure_at_the_rust_line() {
    let marker = "// FAILS: native handwritten";
    let expected = include_str!("reflection.rs")
        .lines()
        .position(|line| line.trim_end().ends_with(marker) && line.contains("rust_line"))
        .map(|index| index + 1)
        .expect("marker line");
    let kernel = HANDWRITTEN_FAILING;
    // The recorded file is the one the macro records for this test file.
    assert_eq!(
        kernel.source.file,
        rejected_by_slang::run::DESCRIPTOR.source.file
    );
    // The outer record is another Rust line, so only the innermost one is the marker.
    assert_eq!(kernel.source.lines[1].rust_line as usize, expected);
    assert_ne!(kernel.source.lines[0].rust_line as usize, expected);
    for target in [Target::Wgsl, Target::Spirv] {
        let error = gust::reflect::compile_reflected(&kernel, target).unwrap_err();
        let gust::reflect::Error::Compiler(gust::slang::Error::CompilationFailed {
            diagnostics,
            ..
        }) = error
        else {
            panic!("{target:?}: expected the helper to refuse the source, got {error}");
        };
        assert!(
            diagnostics.contains("missing_value"),
            "{target:?}: the failure is the undeclared identifier: {diagnostics}"
        );
        // The helper's own error location is the Rust file and line: only the
        // compile copy's `#line` directives can put it there ...
        let location = format!("--> {}:{expected}:", kernel.source.file);
        assert!(
            diagnostics.contains(&location),
            "{target:?}: no `{location}` in:\n{diagnostics}"
        );
        // ... and the annotation names the same line.
        assert!(
            diagnostics.contains(&format!(
                "originating Rust source: {}:{expected}\n",
                kernel.source.file
            )),
            "{target:?}: {diagnostics}"
        );
        assert!(
            diagnostics.contains("reflection helper: "),
            "{target:?}: {diagnostics}"
        );
    }
    // With an empty table the helper reads the clean text: Slang coordinates.
    let mut unmapped = kernel;
    unmapped.source.lines = &[];
    let error = gust::reflect::compile_reflected(&unmapped, Target::Wgsl).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("kernel.slang:4:"), "{text}");
    assert!(!text.contains("reflection.rs"), "{text}");
}

#[gust::gpu]
mod nested {
    pub struct Inner {
        pub value: float,
        pub tag: uint,
    }
    pub struct Outer {
        pub inner: Inner,
        pub weight: int,
    }
    #[kernel]
    pub fn transform(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<Outer>,
        tags: StructuredBuffer<uint>,
        mut output: RWStructuredBuffer<Outer>,
        mut results: RWStructuredBuffer<uint>,
    ) {
        if id.x < output.len() {
            output[id.x] = input[id.x];
            output[id.x].inner.value = input[id.x].inner.value * 2.0 + 1.0;
            output[id.x].inner.tag = tags[id.x] + 7;
            output[id.x].weight = -input[id.x].weight;
            results[id.x] = tags[id.x] * 3 + id.x;
        }
    }
}

#[test]
fn reflected_nested_and_scalar_buffers_execute_with_cache_hits() {
    let device = HeadlessDevice::new().expect("Vulkan adapter required; no silent GPU skip");
    for count in [0, 1, 63, 64, 65, 257] {
        let values: Vec<_> = (0..count)
            .map(|index| nested::Outer {
                inner: nested::Inner {
                    value: index as f32 * 0.5 - 19.0,
                    tag: 91,
                },
                weight: index as i32 - 33,
            })
            .collect();
        let tags: Vec<_> = (0..count).map(|index| index as u32 * 11 + 2).collect();
        let zeros = vec![
            nested::Outer {
                inner: nested::Inner { value: 0.0, tag: 0 },
                weight: 0,
            };
            count
        ];
        let input = device.create_typed_buffer("nested input", &values, GpuBufferAccess::ReadOnly);
        let tag_input =
            device.create_typed_buffer("scalar input", &tags, GpuBufferAccess::ReadOnly);
        let output =
            device.create_typed_buffer("nested output", &zeros, GpuBufferAccess::ReadWrite);
        let results = device.create_typed_buffer(
            "scalar output",
            &vec![0u32; count],
            GpuBufferAccess::ReadWrite,
        );
        for _ in 0..2 {
            device
                .dispatch_buffers(
                    &nested::transform::DESCRIPTOR,
                    count as u32,
                    &[
                        BufferBinding::read_only(&input),
                        BufferBinding::read_only(&tag_input),
                        BufferBinding::read_write(&output),
                        BufferBinding::read_write(&results),
                    ],
                )
                .unwrap();
            let actual = device.read_typed_buffer(&output).unwrap();
            let actual_tags = device.read_typed_buffer(&results).unwrap();
            assert_eq!(actual.len(), count);
            assert_eq!(actual_tags.len(), count);
            for index in 0..count {
                assert_eq!(actual[index].inner.value, index as f32 - 37.0);
                assert_eq!(actual[index].inner.tag, index as u32 * 11 + 9);
                assert_eq!(actual[index].weight, 33 - index as i32);
                assert_eq!(actual_tags[index], index as u32 * 34 + 6);
            }
        }
    }
    let stats = device.pipeline_cache_stats();
    assert_eq!((stats.entries, stats.misses, stats.hits), (1, 1, 9));
}
