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
//!
//! Subgroup operations (slice m8.s; D140): the second half of the file runs the kernels of `tests/fixtures/subgroup.rs`,
//! whose bodies the macro now emits. The file is built under `dev` (the portable bodies are selected: they write zeros,
//! copies or per-element flags) and under `dev+subgroups` (the `_subgroups` bodies are selected: `cargo xtask test
//! --suite Gpu --profile dev+subgroups --test subgroup`), and the same tests judge both from
//! `gust::capability::enabled()`. A subgroup result is judged against a host model rebuilt from the subgroup size and
//! the lane each invocation wrote beside its results, because the size is a property of the pipeline (32 on the
//! RTX 5090, 64 on the Radeon 890M, 8 on lavapipe, 4 on WARP) and the model must hold for every one of them.

include!("../../../tests/fixtures/subgroup.rs");

use gust::cpu::CpuDevice;
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

// ---------------------------------------------------------------------------------------------------------------
// Subgroup operations (m8.s)

/// Whether this build selected the `_subgroups` bodies of the fixture (`GUST_PROFILE=dev+subgroups`).
fn selected() -> bool {
    gust::capability::enabled().contains(&Capability::Subgroups)
}

fn headless() -> HeadlessDevice {
    HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip")
}

/// Runs a fixture kernel over `input` (read-only) with an output of `words` per element, every word starting as
/// `sentinel` so that a word no body wrote shows.
fn run<T: GpuPod + Copy>(
    device: &HeadlessDevice,
    kernel: &KernelDescriptor,
    input: &[T],
    words: usize,
    sentinel: T,
) -> Vec<T> {
    let data = device.create_typed_buffer("subgroup data", input, GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer(
        "subgroup out",
        &vec![sentinel; input.len() * words],
        GpuBufferAccess::ReadWrite,
    );
    device
        .dispatch_buffers(
            kernel,
            input.len() as u32,
            &[
                BufferBinding::read_only(&data),
                BufferBinding::read_write(&out).independent_length(),
            ],
        )
        .unwrap();
    device.read_typed_buffer(&out).unwrap()
}

/// The subgroup of invocation `index` as the kernel reported it: its size and lane. The model assumes the subgroups of
/// a 1D workgroup are consecutive runs of `size` invocations, `lane = index % size`; this is checked rather than
/// assumed, so a device that numbered its lanes otherwise fails here and not in a misleading comparison.
#[track_caller]
fn geometry(index: usize, size: usize, lane: usize) -> usize {
    assert!(
        (2..=128).contains(&size) && size.is_power_of_two(),
        "invocation {index}: subgroup size {size}"
    );
    assert_eq!(
        lane,
        index % size,
        "invocation {index}: lane {lane} of size {size}"
    );
    index - lane
}

/// The adapter and the sizes it reports, printed so that a run records which subgroup size its results are for.
fn report(device: &HeadlessDevice, what: &str) {
    println!(
        "{what}: {} (subgroup size range {:?}), build capabilities {:?}",
        device.adapter_info().name,
        device.subgroup_sizes(),
        gust::capability::enabled()
    );
}

#[test]
fn the_selected_body_follows_the_build_and_keeps_the_layout_of_the_kernel() {
    let kernels: [&KernelDescriptor; 6] = [
        &subgroup::partial_sums::DESCRIPTOR,
        &subgroup::count_if::DESCRIPTOR,
        &subgroup::lanes_u32::DESCRIPTOR,
        &subgroup::lanes_i32::DESCRIPTOR,
        &subgroup::lanes_f32::DESCRIPTOR,
        &subgroup::votes::DESCRIPTOR,
    ];
    for descriptor in kernels {
        // The kernel keeps its name and its handle; the variant is the body the descriptor describes.
        assert_eq!(descriptor.requires, gust::capability::enabled());
        assert_eq!(
            descriptor.variant.ends_with("_subgroups"),
            selected(),
            "{}",
            descriptor.name
        );
        assert_eq!(
            descriptor.entry_point.ends_with("_subgroups"),
            selected(),
            "{}",
            descriptor.entry_point
        );
        assert_eq!(descriptor.workgroup_size, [64, 1, 1]);
        // The bound layout is the portable body's: the same parameters, in the same order, at the same bindings
        // (m8.b1 review N7). `requires(subgroups)` on a body changes nothing a caller binds.
        let names: Vec<&str> = descriptor.parameters.iter().map(|p| p.name).collect();
        assert_eq!(names, ["id", "data", "out"], "{}", descriptor.name);
        let bindings: Vec<_> = descriptor.parameters.iter().map(|p| p.binding).collect();
        assert_eq!(
            bindings,
            [
                None,
                Some(ResourceBinding {
                    group: 0,
                    binding: 0
                }),
                Some(ResourceBinding {
                    group: 0,
                    binding: 1
                })
            ],
            "{}",
            descriptor.name
        );
    }
}

#[test]
fn partial_sums_keeps_its_contract_in_both_builds() {
    let device = headless();
    report(&device, "partial_sums");
    let (minimum, maximum) = device.subgroup_sizes().expect("a subgroup range");
    for count in [1usize, 63, 64, 1000, 4097] {
        let input: Vec<u32> = (0..count).map(|i| ((i * 7 + 3) % 97 + 1) as u32).collect();
        let out = run(
            &device,
            &subgroup::partial_sums::DESCRIPTOR,
            &input,
            1,
            u32::MAX,
        );
        // The contract of both bodies: the sum of the output is the sum of the input.
        let wanted: u64 = input.iter().map(|v| u64::from(*v)).sum();
        let got: u64 = out.iter().map(|v| u64::from(*v)).sum();
        assert_eq!(got, wanted, "count {count}");
        let written = out.iter().filter(|v| **v != 0).count();
        if selected() {
            // One total per subgroup that starts inside the data, written by its elected lane, and zeros elsewhere.
            let fewest = count.div_ceil(maximum as usize);
            let most = count.div_ceil(minimum as usize);
            assert!(
                (fewest..=most).contains(&written),
                "count {count}: {written} totals, expected {fewest}..={most}"
            );
        } else {
            // The portable body copies the data (every element is at least 1).
            assert_eq!(out, input, "count {count}");
        }
        // CPU dispatch under `dev+subgroups` succeeds and returns the contract's output (the portable body copies the
        // data). This does not show which body ran: with the one-lane shadows (`subgroup_add` is its argument,
        // `subgroup_elect` is true, `subgroup_count` is 0 or 1) the subgroup body gives the same output on every input.
        let cpu = CpuDevice::new();
        let cpu_input = cpu.buffer(&input);
        let cpu_out = cpu.buffer(&vec![u32::MAX; count]);
        subgroup::partial_sums::dispatch(&cpu, count as u32, &cpu_input, &cpu_out).unwrap();
        assert_eq!(
            cpu_out.read(),
            input,
            "count {count}: the CPU form is the portable body"
        );
    }
}

#[test]
fn count_if_keeps_its_contract_in_both_builds() {
    let device = headless();
    report(&device, "count_if");
    let (minimum, maximum) = device.subgroup_sizes().expect("a subgroup range");
    for count in [1usize, 63, 64, 1000, 4097] {
        let input: Vec<u32> = (0..count).map(|i| ((i * 13 + 5) % 23) as u32).collect();
        let out = run(
            &device,
            &subgroup::count_if::DESCRIPTOR,
            &input,
            1,
            u32::MAX,
        );
        let wanted = input.iter().filter(|v| **v > 10).count() as u64;
        let got: u64 = out.iter().map(|v| u64::from(*v)).sum();
        assert_eq!(got, wanted, "count {count}");
        if selected() {
            // Each element holds the count of its subgroup or 0; a count is never more than the subgroup size.
            assert!(
                out.iter().all(|v| *v <= maximum),
                "count {count}: a subgroup count over {maximum}"
            );
            let written = out.iter().filter(|v| **v != 0).count();
            assert!(
                written <= count.div_ceil(minimum as usize),
                "count {count}: {written} counts"
            );
        } else {
            let flags: Vec<u32> = input.iter().map(|v| u32::from(*v > 10)).collect();
            assert_eq!(out, flags, "count {count}");
        }
        // CPU dispatch under `dev+subgroups` succeeds and returns the contract's output (one flag per element). This
        // does not show which body ran: with the one-lane shadows (`subgroup_add` is its argument, `subgroup_elect` is
        // true, `subgroup_count` is 0 or 1) the subgroup body gives the same output on every input.
        let cpu = CpuDevice::new();
        let cpu_input = cpu.buffer(&input);
        let cpu_out = cpu.buffer(&vec![u32::MAX; count]);
        subgroup::count_if::dispatch(&cpu, count as u32, &cpu_input, &cpu_out).unwrap();
        let cpu_flags: Vec<u32> = input.iter().map(|v| u32::from(*v > 10)).collect();
        assert_eq!(
            cpu_out.read(),
            cpu_flags,
            "count {count}: the CPU form is the portable body"
        );
    }
}

/// Checks the words one invocation wrote against `expected` and the words after them, which no body writes and so keep
/// the sentinel.
#[track_caller]
fn words_equal<T: PartialEq + std::fmt::Debug + Copy>(
    index: usize,
    got: &[T],
    expected: &[T],
    sentinel: T,
    what: &str,
) {
    assert_eq!(
        &got[..expected.len()],
        expected,
        "{what}: invocation {index}"
    );
    assert!(
        got[expected.len()..].iter().all(|word| *word == sentinel),
        "{what}: invocation {index}: unwritten words {:?}",
        &got[expected.len()..]
    );
}

#[test]
fn lanes_u32_match_the_host_model() {
    let device = headless();
    report(&device, "lanes_u32");
    for count in [256usize, 300] {
        let input: Vec<u32> = (0..count)
            .map(|i| ((i * 37 + 11) % 29 + 1) as u32)
            .collect();
        let got = run(
            &device,
            &subgroup::lanes_u32::DESCRIPTOR,
            &input,
            16,
            0xDEAD_BEEF,
        );
        if !selected() {
            assert!(
                got.iter().all(|word| *word == 0),
                "the portable body writes zeros"
            );
            continue;
        }
        // An invocation past the end of the data holds 0 and still takes part.
        let value = |j: usize| input.get(j).copied().unwrap_or(0);
        for index in 0..count {
            let words = &got[index * 16..index * 16 + 16];
            let (size, lane) = (words[14] as usize, words[15] as usize);
            let base = geometry(index, size, lane);
            let members: Vec<u32> = (base..base + size).map(value).collect();
            let before = &members[..lane];
            let through = &members[..=lane];
            let expected = [
                members.iter().fold(0u32, |a, b| a.wrapping_add(*b)),
                members.iter().fold(1u32, |a, b| a.wrapping_mul(*b)),
                *members.iter().min().unwrap(),
                *members.iter().max().unwrap(),
                members.iter().fold(u32::MAX, |a, b| a & *b),
                members.iter().fold(0u32, |a, b| a | *b),
                members.iter().fold(0u32, |a, b| a ^ *b),
                // The exclusive scans give the identity at lane 0 (docs/SEMANTICS.md).
                before.iter().fold(0u32, |a, b| a.wrapping_add(*b)),
                before.iter().fold(1u32, |a, b| a.wrapping_mul(*b)),
                through.iter().fold(0u32, |a, b| a.wrapping_add(*b)),
                through.iter().fold(1u32, |a, b| a.wrapping_mul(*b)),
                members[0],
                members[2],
                members[lane ^ 1],
                size as u32,
                lane as u32,
            ];
            words_equal(index, words, &expected, 0xDEAD_BEEF, "lanes_u32");
        }
    }
}

#[test]
fn lanes_i32_match_the_host_model() {
    let device = headless();
    report(&device, "lanes_i32");
    for count in [256usize, 300] {
        let input: Vec<i32> = (0..count)
            .map(|i| {
                let v = ((i * 37 + 11) % 28) as i32 + 1;
                if v > 14 { v - 29 } else { v }
            })
            .collect();
        let got = run(
            &device,
            &subgroup::lanes_i32::DESCRIPTOR,
            &input,
            16,
            0x5EED_i32,
        );
        if !selected() {
            assert!(
                got.iter().all(|word| *word == 0),
                "the portable body writes zeros"
            );
            continue;
        }
        let value = |j: usize| input.get(j).copied().unwrap_or(0);
        for index in 0..count {
            let words = &got[index * 16..index * 16 + 16];
            let (size, lane) = (words[14] as usize, words[15] as usize);
            let base = geometry(index, size, lane);
            let members: Vec<i32> = (base..base + size).map(value).collect();
            let before = &members[..lane];
            let through = &members[..=lane];
            let expected = [
                members.iter().fold(0i32, |a, b| a.wrapping_add(*b)),
                members.iter().fold(1i32, |a, b| a.wrapping_mul(*b)),
                *members.iter().min().unwrap(),
                *members.iter().max().unwrap(),
                members.iter().fold(-1i32, |a, b| a & *b),
                members.iter().fold(0i32, |a, b| a | *b),
                members.iter().fold(0i32, |a, b| a ^ *b),
                before.iter().fold(0i32, |a, b| a.wrapping_add(*b)),
                before.iter().fold(1i32, |a, b| a.wrapping_mul(*b)),
                through.iter().fold(0i32, |a, b| a.wrapping_add(*b)),
                through.iter().fold(1i32, |a, b| a.wrapping_mul(*b)),
                members[0],
                // A broadcast of a lane that is not a literal, and a shuffle by a computed lane.
                members[size / 2],
                members[(lane * 5 + 3) % size],
                size as i32,
                lane as i32,
            ];
            words_equal(index, words, &expected, 0x5EED_i32, "lanes_i32");
        }
    }
}

#[test]
fn lanes_f32_match_the_host_model() {
    let device = headless();
    report(&device, "lanes_f32");
    // Powers of two: every sum, product and scan of up to 128 of them is exact in `f32` in any order, so the model needs
    // no rounding bound and the comparison is bit for bit.
    let powers = [0.5f32, 1.0, 2.0, 1.0, 0.5, 2.0, 1.0];
    let sentinel = 12_345.0f32;
    for count in [256usize, 300] {
        let input: Vec<f32> = (0..count).map(|i| powers[(i * 5 + 1) % 7]).collect();
        let got = run(
            &device,
            &subgroup::lanes_f32::DESCRIPTOR,
            &input,
            16,
            sentinel,
        );
        if !selected() {
            assert!(
                got.iter().all(|word| *word == 0.0),
                "the portable body writes zeros"
            );
            continue;
        }
        let value = |j: usize| input.get(j).copied().unwrap_or(0.0);
        for index in 0..count {
            let words = &got[index * 16..index * 16 + 16];
            let (size, lane) = (words[11] as usize, words[12] as usize);
            let base = geometry(index, size, lane);
            let members: Vec<f32> = (base..base + size).map(value).collect();
            let before = &members[..lane];
            let through = &members[..=lane];
            let expected = [
                members.iter().fold(0.0f32, |a, b| a + *b),
                members.iter().fold(1.0f32, |a, b| a * *b),
                members.iter().copied().fold(f32::INFINITY, f32::min),
                members.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                before.iter().fold(0.0f32, |a, b| a + *b),
                before.iter().fold(1.0f32, |a, b| a * *b),
                through.iter().fold(0.0f32, |a, b| a + *b),
                through.iter().fold(1.0f32, |a, b| a * *b),
                members[0],
                members[2],
                members[lane ^ 2],
                size as f32,
                lane as f32,
            ];
            let bits = |words: &[f32]| words.iter().map(|w| w.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&words[..13]),
                bits(&expected),
                "lanes_f32: invocation {index}"
            );
            assert!(
                words[13..]
                    .iter()
                    .all(|w| w.to_bits() == sentinel.to_bits()),
                "lanes_f32: invocation {index}: unwritten words {:?}",
                &words[13..]
            );
        }
    }
}

#[test]
fn votes_match_the_host_model() {
    let device = headless();
    report(&device, "votes");
    for count in [256usize, 300] {
        // Blocks of 64: all true, all false, every third true, and only the first of the block true.
        let input: Vec<u32> = (0..count)
            .map(|i| {
                let over = match (i / 64) % 4 {
                    0 => true,
                    1 => false,
                    2 => i % 3 == 0,
                    _ => i % 64 == 0,
                };
                if over { 20 } else { 3 }
            })
            .collect();
        let got = run(
            &device,
            &subgroup::votes::DESCRIPTOR,
            &input,
            16,
            0xDEAD_BEEF,
        );
        if !selected() {
            assert!(
                got.iter().all(|word| *word == 0),
                "the portable body writes zeros"
            );
            continue;
        }
        // A padded invocation holds 0, which is not over 10.
        let over = |j: usize| input.get(j).is_some_and(|v| *v > 10);
        for index in 0..count {
            let words = &got[index * 16..index * 16 + 16];
            let (size, lane) = (words[8] as usize, words[9] as usize);
            let base = geometry(index, size, lane);
            let votes: Vec<bool> = (base..base + size).map(over).collect();
            let ballot = |word: usize| {
                (0..32usize)
                    .filter(|bit| word * 32 + bit < size && votes[word * 32 + bit])
                    .fold(0u32, |bits, bit| bits | (1 << bit))
            };
            let expected = [
                u32::from(votes.iter().all(|v| *v)),
                u32::from(votes.iter().any(|v| *v)),
                ballot(0),
                ballot(1),
                ballot(2),
                ballot(3),
                votes.iter().filter(|v| **v).count() as u32,
                u32::from(lane == 0),
                size as u32,
                lane as u32,
            ];
            words_equal(index, words, &expected, 0xDEAD_BEEF, "votes");
        }
    }
}

#[test]
fn the_wgsl_of_a_subgroup_body_is_fixed_up_and_its_spirv_is_valid() {
    // `compile_wgsl` applies the fix-up (wgpu 5555 and 10818) to what slangc wrote, so the text wgpu parses has no
    // `enable subgroups;` and a `u32` lane in `subgroupShuffle`/`subgroupBroadcast`; applying it again changes nothing.
    for descriptor in [
        &subgroup::lanes_u32::DESCRIPTOR,
        &subgroup::lanes_i32::DESCRIPTOR,
        &subgroup::lanes_f32::DESCRIPTOR,
        &subgroup::votes::DESCRIPTOR,
    ] {
        let wgsl = gust::slang::compile_wgsl(descriptor).unwrap();
        assert!(!wgsl.contains("enable subgroups"), "{wgsl}");
        assert_eq!(gust::slang::wgsl::fixup_subgroups(&wgsl), wgsl);
        assert_eq!(
            wgsl.contains("subgroupAdd(") || wgsl.contains("subgroupBallot("),
            selected(),
            "{}: the subgroup operations are in the WGSL of the selected body only",
            descriptor.name
        );
        let words = gust::slang::compile_spirv(descriptor).unwrap();
        gust::spirv::validate_structure(&words).unwrap();
    }
    if selected() {
        let wgsl = gust::slang::compile_wgsl(&subgroup::lanes_i32::DESCRIPTOR).unwrap();
        // The lane arguments of the broadcast of a computed lane and of the shuffle are `u32` after the fix-up.
        for call in ["subgroupShuffle(", "subgroupBroadcast("] {
            for part in wgsl.split(call).skip(1) {
                let argument = part.split(')').next().unwrap_or_default();
                assert!(
                    !argument.contains("i32("),
                    "{call} keeps an i32 lane: {argument}"
                );
            }
        }
    }
}

#[test]
fn a_macro_kernel_with_a_ragged_workgroup_is_refused_before_it_compiles() {
    let device = headless();
    if !selected() {
        // The portable body has no requirement and so no subgroup rule.
        assert!(subgroup::partial_sums::DESCRIPTOR.requires.is_empty());
        return;
    }
    let (_, maximum) = device.subgroup_sizes().expect("a subgroup range");
    let ragged = subgroup::partial_sums::DESCRIPTOR.with_workgroup_size([33, 1, 1]);
    let input = [1u32; 70];
    let data = device.create_typed_buffer("ragged data", &input, GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("ragged out", &input, GpuBufferAccess::ReadWrite);
    let error = device
        .dispatch_buffers(
            &ragged,
            input.len() as u32,
            &[
                BufferBinding::read_only(&data),
                BufferBinding::read_write(&out),
            ],
        )
        .unwrap_err();
    match &error {
        Error::SubgroupSize {
            invocations,
            subgroup_max,
            ..
        } => assert_eq!((*invocations, *subgroup_max), (33, maximum)),
        other => panic!("expected SubgroupSize, got {other:?}"),
    }
    assert_eq!(device.read_typed_buffer(&out).unwrap(), input);
}
