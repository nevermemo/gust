//! Capabilities on the device (M8, slice m8.w1; D23, D137, D142): a kernel whose descriptor says `requires: [Subgroups]`
//! runs only on a device opened with the capability, and a descriptor built for the CPU target is refused. The kernels
//! here are hand-written descriptors with plain Slang (`out[i] = out[i] + 1`, no subgroup operation): the macro does
//! not emit subgroup code before slice m8.s, and what is under test is the device's decision, which depends on the
//! descriptor's `requires` and `target` only. The adapter is the one `DeviceOptions::default()` picks (the
//! high-performance Vulkan adapter, the RTX 5090 on the project's PC); a test whose adapter lacks subgroups fails, it
//! never skips. A test that means "no capabilities" says so with `with_capabilities(&[])`, so it does not change when
//! the default options come to request some (m8.w2).
//!
//! The open-refusal path (an adapter that lacks the capability) cannot be reached on an adapter that has it, so it is
//! unit-tested over fake feature sets in `src/device.rs` (`required_capability_features`).

use gust::{
    Access, Arg, Call, GpuPod, Kernel, KernelDescriptor, ParameterDescriptor, ParameterKind,
    Resource, ResourceBinding, ResourceLayout, Run, capability::Capability, target::Target,
};
use gust_wgpu::{
    BufferBinding, DeviceOptions, Error, F32Binding, GpuBuffer, GpuBufferAccess, HeadlessDevice,
    U32Binding,
};

const PARAMETERS: &[ParameterDescriptor] = &[ParameterDescriptor {
    name: "p",
    rust_type: "float",
    kind: ParameterKind::Storage,
    access: Access::ReadWrite,
    resource_layout: Some(ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT)),
    binding: Some(ResourceBinding {
        group: 0,
        binding: 0,
    }),
}];

/// `p[i] = p[i] + 1` over 64-wide workgroups.
const SLANG: &str = r#"
[[vk::binding(0, 0)]] RWStructuredBuffer<float> p;
[shader("compute")] [numthreads(64, 1, 1)]
void subgroup_tests_add_one(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; p.GetDimensions(n, stride);
    if (id.x < n) { p[id.x] = p[id.x] + 1.0; }
}
"#;

const BASE: KernelDescriptor = KernelDescriptor::new(
    "subgroup_tests",
    "add_one",
    "subgroup_tests_add_one",
    [64, 1, 1],
    PARAMETERS,
    SLANG,
);

const PARAMETERS_U32: &[ParameterDescriptor] = &[ParameterDescriptor {
    name: "p",
    rust_type: "uint",
    kind: ParameterKind::Storage,
    access: Access::ReadWrite,
    resource_layout: Some(ResourceLayout::storage_v1(<u32 as GpuPod>::LAYOUT)),
    binding: Some(ResourceBinding {
        group: 0,
        binding: 0,
    }),
}];

/// The same kernel over `u32`, for the second one-shot entry point (`dispatch_u32`). Only launched empty, so its source
/// (the `float` one) is never compiled.
const BASE_U32: KernelDescriptor = KernelDescriptor::new(
    "subgroup_tests",
    "add_one_u32",
    "subgroup_tests_add_one",
    [64, 1, 1],
    PARAMETERS_U32,
    SLANG,
);

/// The kernel as a type: it requires subgroups, so only a device opened with them runs it.
struct NeedsSubgroups;
impl Kernel for NeedsSubgroups {
    const DESCRIPTOR: KernelDescriptor = BASE.with_requires(&[Capability::Subgroups]);
}
impl gust::target::GpuKernel for NeedsSubgroups {}

/// The same kernel with no requirement: the control that the refusal is the `requires` and nothing else.
struct NeedsNothing;
impl Kernel for NeedsNothing {
    const DESCRIPTOR: KernelDescriptor = BASE;
}
impl gust::target::GpuKernel for NeedsNothing {}

fn call<'r, R0, K: Kernel>(
    count: u32,
    p: impl Into<Arg<'r, R0>>,
) -> Call<'r, <R0 as Resource>::Device, K, 1>
where
    R0: Resource<Element = f32> + 'r,
{
    Call::new(count, [p.into().bind(Access::ReadWrite)])
}

const COUNT: usize = 70;

fn without_capabilities() -> HeadlessDevice {
    HeadlessDevice::with_options(DeviceOptions::default().with_capabilities(&[]))
        .expect("Vulkan adapter required; this test must not silently skip")
}

fn with_subgroups() -> HeadlessDevice {
    HeadlessDevice::with_options(
        DeviceOptions::default().with_capabilities(&[Capability::Subgroups]),
    )
    .expect(
        "a Vulkan adapter with subgroup operations is required; this test must not silently skip",
    )
}

fn buffer(device: &HeadlessDevice) -> GpuBuffer<f32> {
    let values: Vec<f32> = (0..COUNT).map(|index| index as f32).collect();
    device.create_typed_buffer("p", &values, GpuBufferAccess::ReadWrite)
}

fn plus_one() -> Vec<f32> {
    (0..COUNT).map(|index| index as f32 + 1.0).collect()
}

fn unchanged() -> Vec<f32> {
    (0..COUNT).map(|index| index as f32).collect()
}

/// A device that compiled and cached nothing: a refusal comes before the pipeline cache.
fn nothing_compiled(device: &HeadlessDevice) {
    let stats = device.pipeline_cache_stats();
    assert_eq!(
        (stats.entries, stats.hits, stats.misses),
        (0, 0, 0),
        "a refused launch must not reach the pipeline cache"
    );
}

#[track_caller]
fn assert_missing_subgroups(error: &Error, device: &HeadlessDevice, kernel: &str) {
    match error {
        Error::MissingCapability {
            capability,
            adapter,
            kernel: Some(name),
        } => {
            assert_eq!(*capability, Capability::Subgroups);
            assert_eq!(adapter, &device.adapter_info().name);
            assert_eq!(name, kernel);
        }
        other => panic!("expected MissingCapability for {kernel}, got {other:?}"),
    }
    let text = error.to_string();
    assert!(
        text.contains(&format!(
            "kernel `{kernel}` requires the `subgroups` capability"
        )),
        "{text}"
    );
    assert!(text.contains("with_capabilities"), "{text}");
}

#[test]
fn a_device_opened_without_the_capability_refuses_a_kernel_that_requires_it() {
    let device = without_capabilities();
    assert!(device.capabilities().is_empty());
    let target = buffer(&device);

    // The erased path.
    let error = device
        .dispatch_buffers(
            &NeedsSubgroups::DESCRIPTOR,
            COUNT as u32,
            &[BufferBinding::read_write(&target)],
        )
        .unwrap_err();
    assert_missing_subgroups(&error, &device, "subgroup_tests::add_one");

    // The typed path reaches the same check.
    let error = device
        .dispatch_call(&call::<_, NeedsSubgroups>(COUNT as u32, &target))
        .unwrap_err();
    assert_missing_subgroups(&error, &device, "subgroup_tests::add_one");

    // Nothing ran and nothing was compiled.
    assert_eq!(device.read_typed_buffer(&target).unwrap(), unchanged());
    nothing_compiled(&device);

    // The control: the same kernel without the requirement runs on the same device.
    device
        .dispatch_call(&call::<_, NeedsNothing>(COUNT as u32, &target))
        .unwrap();
    assert_eq!(device.read_typed_buffer(&target).unwrap(), plus_one());
}

/// An empty launch is judged by the kernel's `requires` like a full one (the same kind of fact as the target): a program
/// tested with empty input must not pass where real input would fail. Every path that returns early for no work is
/// covered: the erased and the typed batch path, and the two one-shot functions.
#[test]
fn an_empty_launch_of_a_kernel_with_an_unmet_requires_is_refused() {
    let device = without_capabilities();
    let empty = device.create_typed_buffer::<f32>("p", &[], GpuBufferAccess::ReadWrite);
    let requires = BASE.with_requires(&[Capability::Subgroups]);
    let requires_u32 = BASE_U32.with_requires(&[Capability::Subgroups]);

    let error = device
        .dispatch_buffers(&requires, 0, &[BufferBinding::read_write(&empty)])
        .unwrap_err();
    assert_missing_subgroups(&error, &device, "subgroup_tests::add_one");

    let error = device
        .dispatch_call(&call::<_, NeedsSubgroups>(0, &empty))
        .unwrap_err();
    assert_missing_subgroups(&error, &device, "subgroup_tests::add_one");

    let error = device
        .dispatch_f32(&requires, 0, &[F32Binding::ReadWrite(&[])])
        .unwrap_err();
    assert_missing_subgroups(&error, &device, "subgroup_tests::add_one");

    let error = device
        .dispatch_u32(&requires_u32, 0, &[U32Binding::ReadWrite(&[])])
        .unwrap_err();
    assert_missing_subgroups(&error, &device, "subgroup_tests::add_one_u32");
    nothing_compiled(&device);

    // The controls: a kernel without `requires` launches empty, and so does the same kernel on a device with the capability.
    device
        .dispatch_buffers(&BASE, 0, &[BufferBinding::read_write(&empty)])
        .unwrap();
    device
        .dispatch_f32(&BASE, 0, &[F32Binding::ReadWrite(&[])])
        .unwrap();
    let device = with_subgroups();
    let empty = device.create_typed_buffer::<f32>("p", &[], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(&requires, 0, &[BufferBinding::read_write(&empty)])
        .unwrap();
    device
        .dispatch_call(&call::<_, NeedsSubgroups>(0, &empty))
        .unwrap();
    device
        .dispatch_f32(&requires, 0, &[F32Binding::ReadWrite(&[])])
        .unwrap();
    device
        .dispatch_u32(&requires_u32, 0, &[U32Binding::ReadWrite(&[])])
        .unwrap();
    nothing_compiled(&device);
}

#[test]
fn a_device_opened_with_the_capability_runs_a_kernel_that_requires_it() {
    let device = with_subgroups();
    assert_eq!(device.capabilities(), &[Capability::Subgroups]);
    let sizes = device
        .subgroup_sizes()
        .expect("an adapter that opened with subgroups reports its range");
    println!(
        "{}: subgroup sizes {}..{}",
        device.adapter_info().name,
        sizes.0,
        sizes.1
    );
    // wgpu's range for the adapter, not one size: only its order and its being sane are asserted.
    assert!(
        0 < sizes.0 && sizes.0 <= sizes.1 && sizes.1 <= 128,
        "{sizes:?}"
    );

    let target = buffer(&device);
    device
        .dispatch_call(&call::<_, NeedsSubgroups>(COUNT as u32, &target))
        .unwrap();
    assert_eq!(device.read_typed_buffer(&target).unwrap(), plus_one());

    // The erased path accepts it too, and the control still runs: a device with more capabilities runs the portable kernel.
    let other = buffer(&device);
    device
        .dispatch_buffers(
            &NeedsSubgroups::DESCRIPTOR,
            COUNT as u32,
            &[BufferBinding::read_write(&other)],
        )
        .unwrap();
    assert_eq!(device.read_typed_buffer(&other).unwrap(), plus_one());
    let third = buffer(&device);
    let compiled = device.pipeline_cache_stats().misses;
    device
        .dispatch_call(&call::<_, NeedsNothing>(COUNT as u32, &third))
        .unwrap();
    assert_eq!(device.read_typed_buffer(&third).unwrap(), plus_one());
    // The `requires` set is part of the cache key: the kernel that differs only in it is compiled on its own, it does
    // not borrow the pipeline of the one that needs subgroups.
    assert_eq!(device.pipeline_cache_stats().misses, compiled + 1);
}

#[test]
fn the_subgroup_range_is_reported_without_the_capability_being_enabled() {
    // The range comes from wgpu's description of the adapter, so a device opened without the capability reports the same one.
    let without = without_capabilities();
    let with = with_subgroups();
    assert_eq!(without.subgroup_sizes(), with.subgroup_sizes());
    assert!(without.subgroup_sizes().is_some());
}

#[test]
fn a_subgroup_kernel_needs_whole_subgroups() {
    let device = with_subgroups();
    let (_, maximum) = device.subgroup_sizes().expect("a subgroup range");
    // 33 invocations are never whole subgroups of a power-of-two size above 1.
    let ragged = BASE
        .with_workgroup_size([33, 1, 1])
        .with_requires(&[Capability::Subgroups]);
    let target = buffer(&device);
    let error = device
        .dispatch_buffers(&ragged, COUNT as u32, &[BufferBinding::read_write(&target)])
        .unwrap_err();
    match &error {
        Error::SubgroupSize {
            kernel,
            invocations,
            subgroup_max,
        } => {
            assert_eq!(kernel, "subgroup_tests::add_one");
            assert_eq!((*invocations, *subgroup_max), (33, maximum));
        }
        other => panic!("expected SubgroupSize, got {other:?}"),
    }
    assert!(
        error
            .to_string()
            .contains("the largest subgroup size wgpu reports for this adapter; a subgroup kernel needs full subgroups"),
        "{error}"
    );
    assert_eq!(device.read_typed_buffer(&target).unwrap(), unchanged());
    nothing_compiled(&device);
}

#[test]
fn a_kernel_built_for_the_cpu_target_is_refused_on_a_gpu_device() {
    let device = without_capabilities();
    let target = buffer(&device);
    let cpu = BASE.with_target(Target::Cpu);

    let error = device
        .dispatch_buffers(&cpu, COUNT as u32, &[BufferBinding::read_write(&target)])
        .unwrap_err();
    match &error {
        Error::KernelTarget { kernel, target } => {
            assert_eq!(kernel, "subgroup_tests::add_one");
            assert_eq!(*target, Target::Cpu);
        }
        other => panic!("expected KernelTarget, got {other:?}"),
    }
    assert!(
        error
            .to_string()
            .starts_with("kernel `subgroup_tests::add_one` is built for the `cpu` target"),
        "{error}"
    );

    // An empty launch is refused as well: the target is a fact of the descriptor, not of the work.
    assert!(matches!(
        device.dispatch_buffers(&cpu, 0, &[BufferBinding::read_write(&target)]),
        Err(Error::KernelTarget { .. })
    ));
    assert_eq!(device.read_typed_buffer(&target).unwrap(), unchanged());
    nothing_compiled(&device);

    // The control: the same descriptor for the GPU target runs.
    device
        .dispatch_buffers(
            &BASE.with_target(Target::Gpu),
            COUNT as u32,
            &[BufferBinding::read_write(&target)],
        )
        .unwrap();
    assert_eq!(device.read_typed_buffer(&target).unwrap(), plus_one());
}
