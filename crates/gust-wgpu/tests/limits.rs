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
    Access, Extent, GpuPod, KernelDescriptor, ParameterDescriptor, ParameterKind, ResourceBinding,
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
    kernel_sized(element, [workgroup, 1, 1], count)
}

/// [`kernel`] with a workgroup of `size` on every axis (every invocation with `id.x == 0` writes the same 17).
fn kernel_sized(element: Element, size: [u32; 3], count: usize) -> KernelDescriptor {
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
        "[shader(\"compute\")] [numthreads({}, {}, {})]\n\
         void probe_k(uint3 id : SV_DispatchThreadID) {{ if (id.x == 0) p0[0] = {}; }}\n",
        size[0],
        size[1],
        size[2],
        element.literal()
    ));
    KernelDescriptor::new(
        "limits_test",
        "k",
        "probe_k",
        size,
        Box::leak(parameters.into_boxed_slice()),
        Box::leak(source.into_boxed_str()),
    )
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

// ---------------------------------------------------------------------------------------------------------------
// m7.a0 / S7.6: `max_buffer_size` is checked before any wgpu call by the fallible buffer functions, real limits are
// requested on demand, and a zeroed 1 GiB buffer is created where the limit allows it.
// ---------------------------------------------------------------------------------------------------------------

/// `DeviceLimit` for a buffer: `kernel` carries the buffer's label.
#[track_caller]
fn assert_buffer_limit(error: Error, label: &str, requested: u64, maximum: u64) {
    match error {
        Error::DeviceLimit {
            kernel,
            limit,
            requested: actual_requested,
            maximum: actual_maximum,
        } => assert_eq!(
            (kernel.as_str(), limit, actual_requested, actual_maximum),
            (label, "max_buffer_size", requested, maximum)
        ),
        other => panic!("expected DeviceLimit for `max_buffer_size`, got {other:?}"),
    }
}

fn open(index: usize, limits: gust_wgpu::LimitsPolicy) -> HeadlessDevice {
    HeadlessDevice::with_options(
        gust_wgpu::DeviceOptions::default()
            .with_backends(wgpu::Backends::VULKAN)
            .with_adapter(gust_wgpu::AdapterChoice::Index(index))
            .with_limits(limits)
            .with_features(gust_wgpu::FeaturesPolicy::Default),
    )
    .expect("the adapter opens")
}

#[test]
fn a_zeroed_buffer_above_max_buffer_size_is_a_device_limit_with_default_limits() {
    let device = device();
    let maximum = device.limits().max_buffer_size;
    assert_eq!(maximum, 256 << 20, "wgpu's default max_buffer_size");
    // 1 GiB of u32.
    let error = device
        .create_zeroed_buffer::<u32>("one gib", 1 << 28, GpuBufferAccess::ReadWrite)
        .err()
        .expect("1 GiB is above 256 MiB");
    assert_buffer_limit(error, "one gib", 1 << 30, maximum);
    // One element above the limit fails, the limit itself succeeds.
    let error = device
        .create_zeroed_buffer::<u32>(
            "just above",
            (maximum / 4) as usize + 1,
            GpuBufferAccess::ReadWrite,
        )
        .err()
        .expect("four bytes above the limit");
    assert_buffer_limit(error, "just above", maximum + 4, maximum);
    let exact = device
        .create_zeroed_buffer::<u32>("exact", (maximum / 4) as usize, GpuBufferAccess::ReadWrite)
        .expect("exactly max_buffer_size is allowed");
    assert_eq!(exact.len(), (maximum / 4) as usize);
    drop(exact);
    // The typed upload has the same pre-check, decided before the upload's staging allocation.
    let values = vec![7_u32; (maximum / 4) as usize + 1];
    let error = device
        .try_create_typed_buffer("upload", &values, GpuBufferAccess::ReadOnly)
        .err()
        .expect("one element too many");
    assert_buffer_limit(error, "upload", maximum + 4, maximum);
    // `with_limits` lowers the limit the pre-check reads.
    let device = lowered(device, |limits| limits.max_buffer_size = 1 << 20);
    let error = device
        .create_zeroed_buffer::<u32>("lowered", (1 << 18) + 1, GpuBufferAccess::ReadWrite)
        .err()
        .expect("above the lowered limit");
    assert_buffer_limit(error, "lowered", (1 << 20) + 4, 1 << 20);
}

#[test]
fn a_64_gib_request_is_a_device_limit_before_any_wgpu_call() {
    for policy in [
        gust_wgpu::LimitsPolicy::Default,
        gust_wgpu::LimitsPolicy::Adapter,
    ] {
        let device = open(0, policy);
        let maximum = device.limits().max_buffer_size;
        let error = device
            .create_zeroed_buffer::<u32>("sixty-four gib", 1 << 34, GpuBufferAccess::ReadWrite)
            .err()
            .expect("64 GiB is above every max_buffer_size");
        assert_buffer_limit(error, "sixty-four gib", 1 << 36, maximum);
        // A length whose byte size overflows `u64` saturates and is refused the same way.
        let error = device
            .create_zeroed_buffer::<u32>("overflow", usize::MAX, GpuBufferAccess::ReadWrite)
            .err()
            .expect("an absurd length");
        assert_buffer_limit(error, "overflow", u64::MAX, maximum);
        // The pooled path decides the same way, before the pool or wgpu is touched: no miss is counted.
        let handle = std::sync::Arc::new(device);
        let error = gust_wgpu::GpuVec::<u32>::zeroed_on(&handle, u32::MAX)
            .expect_err("16 GiB is above every max_buffer_size here");
        assert_buffer_limit(error, "GpuVec", u64::from(u32::MAX) * 4, maximum);
        assert_eq!(handle.buffer_pool_stats().misses, 0);
        assert_eq!(handle.buffer_pool_stats().outstanding, 0);
        // The device is unharmed: a small buffer still allocates and reads back.
        let small = handle.create_typed_buffer("small", &[1_u32, 2, 3], GpuBufferAccess::ReadWrite);
        assert_eq!(handle.read_typed_buffer(&small).unwrap(), [1, 2, 3]);
    }
}

/// S7.6: on every adapter of this machine, a 1 GiB `create_zeroed_buffer` succeeds under `LimitsPolicy::Adapter` where
/// the adapter's `max_buffer_size` allows it, and is `DeviceLimit` (never a panic) where it does not.
#[test]
fn a_1_gib_zeroed_buffer_is_created_where_the_adapter_allows_it() {
    let adapters = HeadlessDevice::enumerate(wgpu::Backends::VULKAN);
    assert!(!adapters.is_empty(), "a Vulkan adapter is required");
    for (index, info) in adapters.iter().enumerate() {
        let device = open(index, gust_wgpu::LimitsPolicy::Adapter);
        let maximum = device.limits().max_buffer_size;
        let outcome =
            device.create_zeroed_buffer::<u32>("one gib", 1 << 28, GpuBufferAccess::ReadWrite);
        println!(
            "S7.6 adapter {index} {:?}: max_buffer_size={maximum}; 1 GiB create_zeroed_buffer -> {}",
            info.name,
            match &outcome {
                Ok(buffer) => format!("Ok ({} elements)", buffer.len()),
                Err(error) => format!("Err({error})"),
            }
        );
        if maximum >= 1 << 30 {
            let buffer = outcome.expect("1 GiB is inside this adapter's max_buffer_size");
            assert_eq!(buffer.len(), 1 << 28);
            assert_eq!(buffer.access(), GpuBufferAccess::ReadWrite);
        } else {
            assert_buffer_limit(
                outcome.err().expect("above the limit"),
                "one gib",
                1 << 30,
                maximum,
            );
        }
    }
}

// ---- the y and z axes of a multi-dimensional dispatch (m7.d) ----

/// One dispatch of `descriptor` over `extent`, through the batch path every typed call takes.
fn dispatch_extent(
    device: &HeadlessDevice,
    descriptor: &KernelDescriptor,
    extent: Extent,
    bindings: &[BufferBinding<'_>],
) -> Result<(), Error> {
    device.dispatch_batch(&[BufferDispatch::new_extent(descriptor, extent, bindings)])
}

#[test]
fn workgroup_size_y_and_z_above_the_limit_are_errors_before_compilation() {
    let device = device();
    let limits = device.limits().clone();
    let output = device.create_f32_buffer("axes", &[3.0], GpuBufferAccess::ReadWrite);
    for (axis, name, maximum) in [
        (
            1,
            "max_compute_workgroup_size_y",
            limits.max_compute_workgroup_size_y,
        ),
        (
            2,
            "max_compute_workgroup_size_z",
            limits.max_compute_workgroup_size_z,
        ),
    ] {
        let mut size = [1, 1, 1];
        size[axis] = maximum + 1;
        let error = device
            .dispatch_buffers(
                &kernel_sized(Element::F32, size, 1),
                1,
                &[BufferBinding::read_write(&output)],
            )
            .expect_err("a workgroup deeper than the device allows must fail");
        assert_limit(error, name, u64::from(maximum) + 1, u64::from(maximum));
        assert_cold(&device);
        assert_eq!(device.read_f32_buffer(&output).unwrap(), [3.0], "{name}");
    }
}

#[test]
fn workgroup_size_y_and_z_at_the_limit_still_run() {
    let device = device();
    let limits = device.limits().clone();
    for (axis, maximum) in [
        (1, limits.max_compute_workgroup_size_y),
        (2, limits.max_compute_workgroup_size_z),
    ] {
        let mut size = [1, 1, 1];
        size[axis] = maximum;
        // The invocations per workgroup limit still applies to the product; keep inside it.
        if u64::from(maximum) > u64::from(limits.max_compute_invocations_per_workgroup) {
            continue;
        }
        let output = device.create_f32_buffer("exact", &[3.0], GpuBufferAccess::ReadWrite);
        device
            .dispatch_buffers(
                &kernel_sized(Element::F32, size, 1),
                1,
                &[BufferBinding::read_write(&output)],
            )
            .unwrap_or_else(|error| panic!("axis {axis} at its limit {maximum}: {error:?}"));
        assert_eq!(
            device.read_f32_buffer(&output).unwrap(),
            [17.0],
            "axis {axis}"
        );
    }
}

#[test]
fn the_workgroup_counts_of_every_axis_are_checked_against_the_per_dimension_limit() {
    let device = lowered(device(), |limits| {
        limits.max_compute_workgroups_per_dimension = 3;
    });
    let output = device.create_f32_buffer("counts", &[3.0], GpuBufferAccess::ReadWrite);
    let descriptor = kernel_sized(Element::F32, [4, 4, 4], 1);
    let bindings = [BufferBinding::read_write(&output).independent_length()];
    // Each over-limit axis is named in the error. The first one over is the one reported: x, then y, then z.
    for (extent, limit) in [
        (
            Extent::new(16, 12, 12),
            "max_compute_workgroups_per_dimension",
        ),
        (
            Extent::new(12, 13, 12),
            "max_compute_workgroups_per_dimension (y)",
        ),
        (
            Extent::new(12, 12, 13),
            "max_compute_workgroups_per_dimension (z)",
        ),
        (
            Extent::new(12, 16, 16),
            "max_compute_workgroups_per_dimension (y)",
        ),
        (
            Extent::new(16, 16, 16),
            "max_compute_workgroups_per_dimension",
        ),
    ] {
        let error = dispatch_extent(&device, &descriptor, extent, &bindings)
            .expect_err("four workgroups along an axis exceed the lowered limit of three");
        assert_limit(error, limit, 4, 3);
        assert_cold(&device);
        assert_eq!(
            device.read_f32_buffer(&output).unwrap(),
            [3.0],
            "{extent:?}"
        );
    }
    // Three workgroups on every axis, and a ragged extent that rounds up to three on every axis, both run.
    for extent in [Extent::new(12, 12, 12), Extent::new(9, 9, 9)] {
        device.write_f32_buffer(&output, &[3.0]).unwrap();
        dispatch_extent(&device, &descriptor, extent, &bindings).unwrap();
        assert_eq!(
            device.read_f32_buffer(&output).unwrap(),
            [17.0],
            "{extent:?}"
        );
    }
}

#[test]
fn an_extent_with_a_zero_dimension_is_ok_and_not_limit_checked() {
    let device = device();
    let output = device.create_f32_buffer("zero", &[3.0], GpuBufferAccess::ReadWrite);
    // A workgroup too deep for the device: any dispatch of it that is not empty is an error.
    let too_deep = device.limits().max_compute_workgroup_size_z + 1;
    let descriptor = kernel_sized(Element::F32, [1, 1, too_deep], 1);
    let bindings = [BufferBinding::read_write(&output).independent_length()];
    for extent in [
        Extent::new(0, 1, 1),
        Extent::new(1, 0, 1),
        Extent::new(1, 1, 0),
        Extent::new(u32::MAX, u32::MAX, 0),
        Extent::new(0, u32::MAX, u32::MAX),
    ] {
        dispatch_extent(&device, &descriptor, extent, &bindings)
            .unwrap_or_else(|error| panic!("{extent:?} is empty and must be Ok: {error:?}"));
    }
    assert_cold(&device);
    assert_eq!(device.read_f32_buffer(&output).unwrap(), [3.0]);
    // The same kernel over one invocation is the error the empty launches did not report.
    let error = dispatch_extent(&device, &descriptor, Extent::new(1, 1, 1), &bindings).unwrap_err();
    assert_limit(
        error,
        "max_compute_workgroup_size_z",
        u64::from(too_deep),
        u64::from(too_deep) - 1,
    );
}

#[test]
fn an_extent_product_beyond_a_u64_is_an_extent_error_before_compilation() {
    let device = device();
    let output = device.create_f32_buffer("overflow", &[3.0], GpuBufferAccess::ReadWrite);
    let descriptor = kernel_sized(Element::F32, [4, 4, 4], 1);
    let bindings = [BufferBinding::read_write(&output).independent_length()];
    let everything = Extent::new(u32::MAX, u32::MAX, u32::MAX);
    match dispatch_extent(&device, &descriptor, everything, &bindings) {
        Err(Error::Extent { extent, .. }) => assert_eq!(extent, [u64::from(u32::MAX); 3]),
        other => panic!("expected Error::Extent, got {other:?}"),
    }
    assert_cold(&device);
    assert_eq!(device.read_f32_buffer(&output).unwrap(), [3.0]);
}

#[test]
fn a_staged_graph_refuses_a_dispatch_with_a_y_or_z_extent() {
    let device = device();
    let output = device.create_f32_buffer("graph", &[3.0], GpuBufferAccess::ReadWrite);
    let descriptor = kernel_sized(Element::F32, [4, 4, 4], 1);
    let bindings = [BufferBinding::read_write(&output).independent_length()];
    for extent in [Extent::new(8, 2, 1), Extent::new(8, 1, 2)] {
        let mut graph = StagedGraph::new();
        graph.dispatch(
            BufferDispatch::new_extent(&descriptor, extent, &bindings),
            &[],
        );
        match device.execute_graph(&graph) {
            Err(Error::Extent {
                extent: reported, ..
            }) => {
                assert_eq!(reported, extent.to_array().map(u64::from));
            }
            other => panic!(
                "expected Error::Extent for {extent:?}, got {:?}",
                other.err()
            ),
        }
        assert_cold(&device);
        assert_eq!(device.read_f32_buffer(&output).unwrap(), [3.0]);
    }
    // A linear extent is the one-dimensional dispatch it always was.
    let mut graph = StagedGraph::new();
    graph.dispatch(
        BufferDispatch::new_extent(&descriptor, Extent::linear(8), &bindings),
        &[],
    );
    device.execute_graph(&graph).unwrap();
    assert_eq!(device.read_f32_buffer(&output).unwrap(), [17.0]);
}

#[test]
fn assigning_element_count_to_a_dispatch_still_sets_what_runs() {
    let device = device();
    let output = device.create_f32_buffer("assign", &[3.0], GpuBufferAccess::ReadWrite);
    let descriptor = kernel(Element::F32, 64, 1);
    let bindings = [BufferBinding::read_write(&output).independent_length()];
    // The pattern of code written before the extent existed: build with a count, then assign the real one.
    let mut dispatch = BufferDispatch::new(&descriptor, 0, &bindings);
    dispatch.element_count = 64;
    assert_eq!(dispatch.extent(), Extent::linear(64));
    device.dispatch_batch(&[dispatch]).unwrap();
    assert_eq!(device.read_f32_buffer(&output).unwrap(), [17.0]);

    // The same on a dispatch built over an extent: x follows the assignment, y and z stay.
    let mut dispatch = BufferDispatch::new_extent(&descriptor, Extent::new(0, 2, 3), &bindings);
    dispatch.element_count = 5;
    assert_eq!(dispatch.extent(), Extent::new(5, 2, 3));
    // And the setter replaces all three axes, `element_count` included.
    dispatch.set_extent(Extent::new(9, 8, 7));
    assert_eq!(
        (dispatch.element_count, dispatch.extent()),
        (9, Extent::new(9, 8, 7))
    );

    // A count assigned to a zero count dispatches nothing when it is assigned zero, too.
    device.write_f32_buffer(&output, &[3.0]).unwrap();
    let mut dispatch = BufferDispatch::new(&descriptor, 64, &bindings);
    dispatch.element_count = 0;
    device.dispatch_batch(&[dispatch]).unwrap();
    assert_eq!(device.read_f32_buffer(&output).unwrap(), [3.0]);
}
