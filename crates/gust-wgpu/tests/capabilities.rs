//! The default capability set of a device (M8, slice m8.w2; D23, D142): `DeviceOptions::default()`,
//! `HeadlessDevice::new()` and `default_device()` request the capabilities this build was compiled for,
//! `gust::capability::enabled()`, which is what `GUST_PROFILE` named after its target. The file reads that set at run
//! time, so one file pins both builds: under `dev` the set is empty and a kernel that requires subgroups is refused,
//! under `dev+subgroups` the set is `[Subgroups]` and the same kernel runs (`cargo xtask test --suite Gpu --profile
//! dev+subgroups --test capabilities`, and the lavapipe step of `gpu-software`, whose adapter reports subgroup sizes
//! 8..8). A test that means "no capabilities" says so with `with_capabilities(&[])`.
//!
//! The kernel is a hand-written descriptor with plain Slang (no subgroup operation), as in `tests/subgroup.rs`: what is
//! under test is the device's decision, which depends on the descriptor's `requires` and on the capabilities the
//! device was opened with. The refusal to open (an adapter without the capability) and the browser's refusal are
//! unit-tested over fake feature sets in `src/device.rs`.

use gust::{
    Access, KernelDescriptor, ParameterDescriptor, ParameterKind, ResourceBinding, ResourceLayout,
    capability::{self, Capability},
};
use gust_wgpu::{
    BufferBinding, DeviceOptions, Error, GpuBuffer, GpuBufferAccess, HeadlessDevice, default_device,
};

const PARAMETERS: &[ParameterDescriptor] = &[ParameterDescriptor {
    name: "p",
    rust_type: "float",
    kind: ParameterKind::Storage,
    access: Access::ReadWrite,
    resource_layout: Some(ResourceLayout::storage_v1(<f32 as gust::GpuPod>::LAYOUT)),
    binding: Some(ResourceBinding {
        group: 0,
        binding: 0,
    }),
}];

/// `p[i] = p[i] + 1` over 64-wide workgroups, a multiple of every subgroup size an adapter reports.
const SLANG: &str = r#"
[[vk::binding(0, 0)]] RWStructuredBuffer<float> p;
[shader("compute")] [numthreads(64, 1, 1)]
void capabilities_tests_add_one(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; p.GetDimensions(n, stride);
    if (id.x < n) { p[id.x] = p[id.x] + 1.0; }
}
"#;

const BASE: KernelDescriptor = KernelDescriptor::new(
    "capabilities_tests",
    "add_one",
    "capabilities_tests_add_one",
    [64, 1, 1],
    PARAMETERS,
    SLANG,
);

/// The kernel when it requires subgroups: the one capability this release has.
const NEEDS_SUBGROUPS: KernelDescriptor = BASE.with_requires(&[Capability::Subgroups]);

const COUNT: usize = 70;

fn buffer(device: &HeadlessDevice) -> GpuBuffer<f32> {
    let values: Vec<f32> = (0..COUNT).map(|index| index as f32).collect();
    device.create_typed_buffer("p", &values, GpuBufferAccess::ReadWrite)
}

fn unchanged() -> Vec<f32> {
    (0..COUNT).map(|index| index as f32).collect()
}

fn plus_one() -> Vec<f32> {
    (0..COUNT).map(|index| index as f32 + 1.0).collect()
}

/// Launch `NEEDS_SUBGROUPS` on `device` and check the outcome against whether the device holds the capability: it runs
/// and adds one, or it is `MissingCapability { kernel: Some(..) }` and the buffer is untouched.
#[track_caller]
fn assert_runs_iff_opened_with_subgroups(device: &HeadlessDevice) {
    let target = buffer(device);
    let result = device.dispatch_buffers(
        &NEEDS_SUBGROUPS,
        COUNT as u32,
        &[BufferBinding::read_write(&target)],
    );
    if device.capabilities().contains(&Capability::Subgroups) {
        result.expect("a device opened with subgroups runs a kernel that requires them");
        assert_eq!(device.read_typed_buffer(&target).unwrap(), plus_one());
    } else {
        match result {
            Err(Error::MissingCapability {
                capability,
                adapter,
                kernel: Some(kernel),
            }) => {
                assert_eq!(capability, Capability::Subgroups);
                assert_eq!(adapter, device.adapter_info().name);
                assert_eq!(kernel, "capabilities_tests::add_one");
            }
            other => panic!("expected MissingCapability for the kernel, got {other:?}"),
        }
        assert_eq!(device.read_typed_buffer(&target).unwrap(), unchanged());
    }
}

#[test]
fn the_default_options_request_the_builds_capabilities() {
    assert_eq!(DeviceOptions::default().capabilities, capability::enabled());
    // The explicit empty set overrides it, and the setter touches nothing else.
    let none = DeviceOptions::default().with_capabilities(&[]);
    assert!(none.capabilities.is_empty());
    assert_eq!(none.backends, DeviceOptions::default().backends);
    assert_eq!(none.limits, DeviceOptions::default().limits);
    assert_eq!(none.features, DeviceOptions::default().features);
}

#[test]
fn new_and_new_async_open_with_the_builds_capabilities() {
    let device = HeadlessDevice::new()
        .expect("a Vulkan adapter is required; this test must not silently skip");
    assert_eq!(device.capabilities(), capability::enabled());
    let device = pollster::block_on(HeadlessDevice::new_async(wgpu::Backends::VULKAN))
        .expect("a Vulkan adapter is required; this test must not silently skip");
    assert_eq!(device.capabilities(), capability::enabled());
}

#[test]
fn options_open_with_the_builds_capabilities_unless_told_otherwise() {
    let device = HeadlessDevice::with_options(DeviceOptions::default())
        .expect("a Vulkan adapter is required; this test must not silently skip");
    assert_eq!(device.capabilities(), capability::enabled());
    let device = HeadlessDevice::with_options(DeviceOptions::default().with_capabilities(&[]))
        .expect("a Vulkan adapter is required; this test must not silently skip");
    assert!(device.capabilities().is_empty());
}

/// The only test of this file that touches the process-wide default device.
#[test]
fn the_default_device_opens_with_the_builds_capabilities() {
    let device =
        default_device().expect("a Vulkan adapter is required; this test must not silently skip");
    assert_eq!(device.capabilities(), capability::enabled());
}

/// The point of the default: a device opened without choosing runs the kernels of its own build. A kernel that requires
/// subgroups runs on it exactly when the build named them (`dev+subgroups`), and is refused when it did not (`dev`); a
/// device opened with an explicit empty set always refuses it, and one opened with the capability always runs it.
#[test]
fn a_default_device_runs_a_kernel_that_requires_exactly_the_builds_capabilities() {
    println!("capabilities of this build: {:?}", capability::enabled());
    let device = HeadlessDevice::new()
        .expect("a Vulkan adapter is required; this test must not silently skip");
    assert_eq!(
        device.capabilities().contains(&Capability::Subgroups),
        capability::enabled().contains(&Capability::Subgroups)
    );
    assert_runs_iff_opened_with_subgroups(&device);

    let without = HeadlessDevice::with_options(DeviceOptions::default().with_capabilities(&[]))
        .expect("a Vulkan adapter is required; this test must not silently skip");
    assert!(without.capabilities().is_empty());
    assert_runs_iff_opened_with_subgroups(&without);

    let with = HeadlessDevice::with_options(
        DeviceOptions::default().with_capabilities(&[Capability::Subgroups]),
    )
    .expect(
        "a Vulkan adapter with subgroup operations is required; this test must not silently skip",
    );
    assert_eq!(with.capabilities(), &[Capability::Subgroups]);
    assert_runs_iff_opened_with_subgroups(&with);
}

/// The build's set follows the `GUST_PROFILE` of the cargo run that built it. Cargo hands its own environment to the test
/// binary, so the variable seen here is the one the build script and the macro saw: `dev+subgroups` names subgroups and
/// every other value (or none) names none. This is what shows that `cargo xtask test --profile` reached the build.
#[test]
fn the_builds_capabilities_follow_the_profile_variable() {
    let profile = std::env::var("GUST_PROFILE").unwrap_or_default();
    let named = profile.split('+').skip(1).any(|name| name == "subgroups");
    println!(
        "GUST_PROFILE={profile:?}; capabilities of this build: {:?}",
        capability::enabled()
    );
    assert_eq!(
        capability::enabled().contains(&Capability::Subgroups),
        named
    );
}
