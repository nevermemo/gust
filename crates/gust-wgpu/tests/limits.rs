//! Launch-limit checks (roadmap M2, m2.b). Every over-limit launch must come back as
//! `Error::DeviceLimit` before any pipeline is compiled or looked up (cache statistics
//! stay at zero), where wgpu would otherwise panic through its uncaptured-error
//! handler. The exact-limit launches must still run on the GPU. These tests cover single
//! dispatches and a one-node graph; for a multi-node graph, earlier nodes may already be
//! compiled before an over-limit node fails (the graph-level pass belongs to m2.e).
//!
//! The kernels are hand-written descriptors; their Slang `numthreads` matches the
//! descriptor, so the only failure a test can see is the limit.

use gust::{
    Access, GpuPod, KernelDescriptor, ParameterDescriptor, ParameterKind, ResourceBinding,
    ResourceLayout, TypeLayout,
};
use gust_wgpu::{
    BufferBinding, BufferDispatch, Error, F32Binding, GpuBufferAccess, HeadlessDevice, StagedGraph,
    U32Binding,
};

#[derive(Clone, Copy)]
enum Element {
    F32,
    U32,
}

impl Element {
    fn layout(self) -> TypeLayout {
        match self {
            Self::F32 => <f32 as GpuPod>::LAYOUT,
            Self::U32 => <u32 as GpuPod>::LAYOUT,
        }
    }

    fn slang_type(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::U32 => "uint",
        }
    }

    fn rust_type(self) -> &'static str {
        match self {
            Self::F32 => "StorageMut<f32>",
            Self::U32 => "StorageMut<u32>",
        }
    }

    fn literal(self) -> &'static str {
        match self {
            Self::F32 => "17.0",
            Self::U32 => "17u",
        }
    }
}

/// A kernel with `count` read-write storage parameters `p0..`, workgroup width
/// `workgroup`, whose invocation 0 writes 17 into element 0 of `p0`.
fn kernel(element: Element, workgroup: u32, count: usize) -> KernelDescriptor {
    let parameters = (0..count)
        .map(|index| ParameterDescriptor {
            name: Box::leak(format!("p{index}").into_boxed_str()),
            rust_type: element.rust_type(),
            kind: ParameterKind::Storage,
            access: Access::ReadWrite,
            resource_layout: Some(ResourceLayout::storage_v1(element.layout())),
            binding: Some(ResourceBinding {
                group: 0,
                binding: index as u32,
            }),
        })
        .collect::<Vec<_>>();
    let mut source = String::new();
    for index in 0..count {
        source.push_str(&format!(
            "[[vk::binding({index}, 0)]] RWStructuredBuffer<{}> p{index};\n",
            element.slang_type()
        ));
    }
    source.push_str(&format!(
        "[shader(\"compute\")] [numthreads({workgroup}, 1, 1)]\n\
         void probe_k(uint3 id : SV_DispatchThreadID) {{ if (id.x == 0) p0[0] = {}; }}\n",
        element.literal()
    ));
    KernelDescriptor {
        module: "limits_test",
        name: "k",
        entry_point: "probe_k",
        workgroup_size: [workgroup, 1, 1],
        parameters: Box::leak(parameters.into_boxed_slice()),
        slang_source: Box::leak(source.into_boxed_str()),
        source: gust::SourceOrigin::UNKNOWN,
        artifacts: &[],
    }
}

fn device() -> HeadlessDevice {
    HeadlessDevice::new().expect("Vulkan adapter required")
}

fn lowered(device: HeadlessDevice, change: impl FnOnce(&mut wgpu::Limits)) -> HeadlessDevice {
    let mut limits = device.limits().clone();
    change(&mut limits);
    device
        .with_limits(limits)
        .expect("lowering limits is allowed")
}

fn assert_cold(device: &HeadlessDevice) {
    let stats = device.pipeline_cache_stats();
    assert_eq!(
        (stats.entries, stats.misses, stats.hits),
        (0, 0, 0),
        "a limit error must precede compilation and the cache lookup"
    );
}

#[track_caller]
fn assert_limit(error: Error, limit: &str, requested: u64, maximum: u64) {
    match error {
        Error::DeviceLimit {
            kernel,
            limit: actual_limit,
            requested: actual_requested,
            maximum: actual_maximum,
        } => {
            assert_eq!(kernel, "limits_test::k");
            assert_eq!(
                (actual_limit, actual_requested, actual_maximum),
                (limit, requested, maximum)
            );
        }
        other => panic!("expected DeviceLimit for `{limit}`, got {other:?}"),
    }
}

#[test]
fn limits_are_the_opened_devices_and_the_replacement_lowers_them() {
    let device = device();
    let opened = device.limits().clone();
    // Deliberate S2 tripwire, the only place that compares with constants instead of
    // reading the device: it fails if wgpu changes its defaults or if GUST starts
    // requesting the adapter's limits, so the other tests' assumptions get re-checked.
    assert_eq!(opened, wgpu::Limits::default());
    assert_eq!(
        opened.max_compute_workgroups_per_dimension,
        gust_wgpu::DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
        "the exported constant the examples use must be what the opened device reports"
    );
    println!(
        "adapter {:?}; max_compute_workgroup_size_x={} max_compute_invocations_per_workgroup={} \
         max_compute_workgroups_per_dimension={} max_storage_buffer_binding_size={} \
         max_buffer_size={} max_storage_buffers_per_shader_stage={}",
        device.adapter_info().name,
        opened.max_compute_workgroup_size_x,
        opened.max_compute_invocations_per_workgroup,
        opened.max_compute_workgroups_per_dimension,
        opened.max_storage_buffer_binding_size,
        opened.max_buffer_size,
        opened.max_storage_buffers_per_shader_stage,
    );
    let device = lowered(device, |limits| limits.max_compute_workgroup_size_x = 32);
    assert_eq!(device.limits().max_compute_workgroup_size_x, 32);
    // Raising back up to the opened device's value is allowed: the comparison is
    // against the device, not against the previously installed limits.
    let device = device
        .with_limits(opened.clone())
        .expect("restoring is allowed");
    assert_eq!(device.limits(), &opened);
}

#[test]
fn with_limits_rejects_a_raised_maximum() {
    for (name, raise) in [
        (
            "max_compute_workgroups_per_dimension",
            (|limits: &mut wgpu::Limits| limits.max_compute_workgroups_per_dimension += 1)
                as fn(&mut wgpu::Limits),
        ),
        ("max_storage_buffer_binding_size", |limits| {
            limits.max_storage_buffer_binding_size += 1
        }),
        ("max_compute_workgroup_size_x", |limits| {
            limits.max_compute_workgroup_size_x += 1
        }),
        ("max_storage_buffers_per_shader_stage", |limits| {
            limits.max_storage_buffers_per_shader_stage += 1
        }),
    ] {
        let device = device();
        let opened = device.limits().clone();
        let mut limits = opened.clone();
        raise(&mut limits);
        match device.with_limits(limits) {
            Err(Error::LimitRaised {
                limit,
                requested,
                device,
            }) => {
                assert_eq!(limit, name);
                assert_eq!(requested, device + 1, "{name}");
            }
            Err(other) => panic!("expected LimitRaised for `{name}`, got {other:?}"),
            Ok(_) => panic!("raising `{name}` above the device must be rejected"),
        }
    }
}

#[test]
fn workgroup_width_above_the_limit_is_an_error_before_compilation() {
    let device = device();
    let maximum = device.limits().max_compute_workgroup_size_x;
    let output = device.create_f32_buffer("width", &[3.0], GpuBufferAccess::ReadWrite);
    let descriptor = kernel(Element::F32, maximum + 1, 1);
    let error = device
        .dispatch_buffers(&descriptor, 1, &[BufferBinding::read_write(&output)])
        .expect_err("a workgroup wider than the device allows must fail");
    assert_limit(
        error,
        "max_compute_workgroup_size_x",
        u64::from(maximum) + 1,
        u64::from(maximum),
    );
    assert_cold(&device);
    assert_eq!(device.read_f32_buffer(&output).unwrap(), [3.0]);
}

#[test]
fn invocations_per_workgroup_above_the_limit_is_an_error_before_compilation() {
    let device = lowered(device(), |limits| {
        limits.max_compute_invocations_per_workgroup = 32;
    });
    let output = device.create_f32_buffer("invocations", &[3.0], GpuBufferAccess::ReadWrite);
    let error = device
        .dispatch_buffers(
            &kernel(Element::F32, 64, 1),
            1,
            &[BufferBinding::read_write(&output)],
        )
        .expect_err("64 invocations exceed the lowered limit of 32");
    assert_limit(error, "max_compute_invocations_per_workgroup", 64, 32);
    assert_cold(&device);
    assert_eq!(device.read_f32_buffer(&output).unwrap(), [3.0]);
}

#[test]
fn storage_buffers_above_the_limit_are_an_error_before_compilation() {
    let device = device();
    let maximum = device.limits().max_storage_buffers_per_shader_stage;
    let count = maximum as usize + 1;
    let buffers = (0..count)
        .map(|index| {
            device.create_f32_buffer(&format!("p{index}"), &[3.0], GpuBufferAccess::ReadWrite)
        })
        .collect::<Vec<_>>();
    let bindings = buffers
        .iter()
        .map(BufferBinding::read_write)
        .collect::<Vec<_>>();
    let error = device
        .dispatch_buffers(&kernel(Element::F32, 64, count), 1, &bindings)
        .expect_err("one storage buffer more than the stage allows must fail");
    assert_limit(
        error,
        "max_storage_buffers_per_shader_stage",
        u64::from(maximum) + 1,
        u64::from(maximum),
    );
    assert_cold(&device);
    for buffer in &buffers {
        assert_eq!(device.read_f32_buffer(buffer).unwrap(), [3.0]);
    }
}

#[test]
fn lowering_limits_after_the_cache_is_warm_still_rejects() {
    let device = device();
    let opened = device.limits().clone();
    let output = device.create_f32_buffer("warm", &[3.0], GpuBufferAccess::ReadWrite);
    let descriptor = kernel(Element::F32, 64, 1);
    let bindings = [BufferBinding::read_write(&output)];
    device.dispatch_buffers(&descriptor, 1, &bindings).unwrap();
    assert_eq!(device.read_f32_buffer(&output).unwrap(), [17.0]);
    let stats = device.pipeline_cache_stats();
    assert_eq!((stats.entries, stats.misses, stats.hits), (1, 1, 0));

    let device = lowered(device, |limits| limits.max_compute_workgroup_size_x = 32);
    device.write_f32_buffer(&output, &[3.0]).unwrap();
    let error = device
        .dispatch_buffers(&descriptor, 1, &bindings)
        .expect_err("the cached pipeline must not authorize a launch the limits forbid");
    assert_limit(error, "max_compute_workgroup_size_x", 64, 32);
    let stats = device.pipeline_cache_stats();
    assert_eq!(
        (stats.entries, stats.misses, stats.hits),
        (1, 1, 0),
        "the limit check runs before the cache lookup, so no hit is counted"
    );
    assert_eq!(device.read_f32_buffer(&output).unwrap(), [3.0]);

    let device = device.with_limits(opened).unwrap();
    device.dispatch_buffers(&descriptor, 1, &bindings).unwrap();
    assert_eq!(device.read_f32_buffer(&output).unwrap(), [17.0]);
    assert_eq!(device.pipeline_cache_stats().hits, 1);
}

#[test]
fn an_empty_dispatch_is_not_limit_checked() {
    let device = device();
    let maximum = device.limits().max_compute_workgroup_size_x;
    let output = device.create_f32_buffer("empty", &[], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &kernel(Element::F32, maximum + 1, 1),
            0,
            &[BufferBinding::read_write(&output)],
        )
        .expect("a dispatch of zero elements does no GPU work, so no limit applies");
    assert_cold(&device);
}

/// `(maximum workgroups, over-limit element count, exact-limit element count)` for a
/// 64-wide kernel.
fn count_cases(device: &HeadlessDevice) -> (u32, u32, u32) {
    let maximum = u64::from(device.limits().max_compute_workgroups_per_dimension);
    let narrow = |value: u64| u32::try_from(value).expect("element count fits in u32");
    (
        narrow(maximum),
        narrow((maximum + 1) * 64),
        narrow(maximum * 64),
    )
}

#[test]
fn workgroup_count_through_dispatch_buffers() {
    let device = device();
    let (maximum, over, exact) = count_cases(&device);
    let output = device.create_f32_buffer("count", &[3.0], GpuBufferAccess::ReadWrite);
    let descriptor = kernel(Element::F32, 64, 1);
    let bindings = [BufferBinding::read_write(&output).independent_length()];
    let error = device
        .dispatch_buffers(&descriptor, over, &bindings)
        .expect_err("65536 workgroups exceed the per-dimension limit");
    assert_limit(
        error,
        "max_compute_workgroups_per_dimension",
        u64::from(maximum) + 1,
        u64::from(maximum),
    );
    assert_cold(&device);
    assert_eq!(device.read_f32_buffer(&output).unwrap(), [3.0]);

    device
        .dispatch_buffers(&descriptor, exact, &bindings)
        .unwrap();
    assert_eq!(device.read_f32_buffer(&output).unwrap(), [17.0]);
}

#[test]
fn workgroup_count_through_dispatch_f32() {
    let device = device();
    let (maximum, over, exact) = count_cases(&device);
    let descriptor = kernel(Element::F32, 64, 1);
    let mut values = vec![0.0_f32; over as usize];
    values[0] = 3.0;
    let error = device
        .dispatch_f32(&descriptor, over, &[F32Binding::ReadWrite(&values)])
        .expect_err("65536 workgroups exceed the per-dimension limit");
    assert_limit(
        error,
        "max_compute_workgroups_per_dimension",
        u64::from(maximum) + 1,
        u64::from(maximum),
    );
    assert_cold(&device);

    let output = device
        .dispatch_f32(
            &descriptor,
            exact,
            &[F32Binding::ReadWrite(&values[..exact as usize])],
        )
        .unwrap();
    assert_eq!(output.writable_buffers.len(), 1);
    assert_eq!(output.writable_buffers[0].len(), exact as usize);
    assert_eq!(output.writable_buffers[0][0], 17.0);
}

#[test]
fn workgroup_count_through_dispatch_u32() {
    let device = device();
    let (maximum, over, exact) = count_cases(&device);
    let descriptor = kernel(Element::U32, 64, 1);
    let mut values = vec![0_u32; over as usize];
    values[0] = 3;
    let error = device
        .dispatch_u32(&descriptor, over, &[U32Binding::ReadWrite(&values)])
        .expect_err("65536 workgroups exceed the per-dimension limit");
    assert_limit(
        error,
        "max_compute_workgroups_per_dimension",
        u64::from(maximum) + 1,
        u64::from(maximum),
    );
    assert_cold(&device);

    let output = device
        .dispatch_u32(
            &descriptor,
            exact,
            &[U32Binding::ReadWrite(&values[..exact as usize])],
        )
        .unwrap();
    assert_eq!(output.writable_buffers.len(), 1);
    assert_eq!(output.writable_buffers[0].len(), exact as usize);
    assert_eq!(output.writable_buffers[0][0], 17);
}

#[test]
fn workgroup_count_through_a_one_node_graph() {
    let device = device();
    let (maximum, over, exact) = count_cases(&device);
    let output = device.create_f32_buffer("graph", &[3.0], GpuBufferAccess::ReadWrite);
    let descriptor = kernel(Element::F32, 64, 1);
    let bindings = [BufferBinding::read_write(&output).independent_length()];

    let mut graph = StagedGraph::new();
    graph.dispatch(BufferDispatch::new(&descriptor, over, &bindings), &[]);
    let error = device
        .execute_graph(&graph)
        .err()
        .expect("the graph inherits the dispatch checks");
    assert_limit(
        error,
        "max_compute_workgroups_per_dimension",
        u64::from(maximum) + 1,
        u64::from(maximum),
    );
    assert_cold(&device);
    assert_eq!(device.read_f32_buffer(&output).unwrap(), [3.0]);

    let mut graph = StagedGraph::new();
    graph.dispatch(BufferDispatch::new(&descriptor, exact, &bindings), &[]);
    device.execute_graph(&graph).unwrap();
    assert_eq!(device.read_f32_buffer(&output).unwrap(), [17.0]);
}

/// Four elements are 16 bytes, exactly the lowered binding limit; five are 20.
fn small_binding_device() -> HeadlessDevice {
    lowered(device(), |limits| {
        limits.max_storage_buffer_binding_size = 16;
    })
}

#[test]
fn binding_size_through_dispatch_buffers() {
    let device = small_binding_device();
    let descriptor = kernel(Element::U32, 64, 1);
    let too_big = device.create_typed_buffer("big", &[3_u32; 5], GpuBufferAccess::ReadWrite);
    let error = device
        .dispatch_buffers(&descriptor, 5, &[BufferBinding::read_write(&too_big)])
        .expect_err("20 bytes exceed the 16-byte binding limit");
    assert_limit(error, "max_storage_buffer_binding_size", 20, 16);
    assert_cold(&device);
    assert_eq!(device.read_typed_buffer(&too_big).unwrap(), [3_u32; 5]);

    let fits = device.create_typed_buffer("fits", &[3_u32; 4], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(&descriptor, 4, &[BufferBinding::read_write(&fits)])
        .unwrap();
    assert_eq!(device.read_typed_buffer(&fits).unwrap(), [17, 3, 3, 3]);
}

#[test]
fn binding_size_through_dispatch_f32() {
    let device = small_binding_device();
    let descriptor = kernel(Element::F32, 64, 1);
    let error = device
        .dispatch_f32(&descriptor, 5, &[F32Binding::ReadWrite(&[3.0; 5])])
        .expect_err("20 bytes exceed the 16-byte binding limit");
    assert_limit(error, "max_storage_buffer_binding_size", 20, 16);
    assert_cold(&device);

    let output = device
        .dispatch_f32(&descriptor, 4, &[F32Binding::ReadWrite(&[3.0; 4])])
        .unwrap();
    assert_eq!(output.writable_buffers, [vec![17.0, 3.0, 3.0, 3.0]]);
}

#[test]
fn binding_size_through_dispatch_u32() {
    let device = small_binding_device();
    let descriptor = kernel(Element::U32, 64, 1);
    let error = device
        .dispatch_u32(&descriptor, 5, &[U32Binding::ReadWrite(&[3; 5])])
        .expect_err("20 bytes exceed the 16-byte binding limit");
    assert_limit(error, "max_storage_buffer_binding_size", 20, 16);
    assert_cold(&device);

    let output = device
        .dispatch_u32(&descriptor, 4, &[U32Binding::ReadWrite(&[3; 4])])
        .unwrap();
    assert_eq!(output.writable_buffers, [vec![17, 3, 3, 3]]);
}

/// A hand-written `GpuPod` that claims alignment 8 for a four-byte value, so its layout is not
/// StorageV1. It is never uploaded: every allocation path must refuse it first.
#[derive(Clone, Copy)]
#[repr(C)]
struct LyingPod(u32);

// SAFETY: deliberately dishonest, to prove the runtime layout assertion fires.
unsafe impl GpuPod for LyingPod {
    const LAYOUT: TypeLayout = TypeLayout {
        name: "LyingPod",
        size: 4,
        alignment: 8,
        stride: 4,
        kind: gust::TypeLayoutKind::Scalar(gust::ScalarKind::U32),
    };
}

#[test]
#[should_panic(expected = "GpuPod implementation must match the Rust StorageV1 representation")]
fn typed_buffers_refuse_a_lying_pod() {
    device().create_typed_buffer("lying", &[LyingPod(1)], GpuBufferAccess::ReadWrite);
}

#[test]
#[should_panic(expected = "GpuPod implementation must match the Rust StorageV1 representation")]
fn pool_buffers_refuse_a_lying_pod() {
    device().create_pool("lying", &[LyingPod(1)], 4);
}

/// A four-byte Rust type whose layout claims to be exactly `DispatchIndirectArgs` (three `u32`s,
/// size 12), so it passes the indirect-arguments layout check and only the size assertion can
/// refuse it.
#[derive(Clone, Copy)]
#[repr(C)]
struct LyingArgs(u32);

// SAFETY: deliberately dishonest, to prove the runtime layout assertion fires.
unsafe impl GpuPod for LyingArgs {
    const LAYOUT: TypeLayout = TypeLayout {
        name: "LyingArgs",
        size: 12,
        alignment: 4,
        stride: 12,
        kind: gust::TypeLayoutKind::Struct(&[
            gust::StructFieldLayout {
                name: "x",
                offset: 0,
                ty: TypeLayout::scalar("u32", gust::ScalarKind::U32),
            },
            gust::StructFieldLayout {
                name: "y",
                offset: 4,
                ty: TypeLayout::scalar("u32", gust::ScalarKind::U32),
            },
            gust::StructFieldLayout {
                name: "z",
                offset: 8,
                ty: TypeLayout::scalar("u32", gust::ScalarKind::U32),
            },
        ]),
    };
}

#[test]
#[should_panic(expected = "GpuPod implementation must match the Rust StorageV1 representation")]
fn indirect_buffers_refuse_a_lying_pod() {
    let _ = device().create_indirect_buffer("lying", &[LyingArgs(1)]);
}
