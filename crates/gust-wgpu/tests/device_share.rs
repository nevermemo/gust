//! Sharing a device with a renderer (m14.devshare.1; Gpu tier): a device the caller opened in its own `wgpu::Instance`
//! from `DeviceOptions::device_request`, then `HeadlessDevice::adopt`ed. Test 2 is the independent oracle: the set of
//! features and limits the open asks for is written out here by hand, so the open and `device_request` cannot drift
//! together. The real pair of test 5 (two adapters of one instance) is opt-in: it needs `GUST_DEVSHARE_REAL_PAIR=1`.

mod common;

use gust::capability::Capability;
use gust::cpu::CpuDevice;
use gust::map::{Broadcast, Error as MapError};
use gust::math::Vec2;
use gust_wgpu::wgpu;
use gust_wgpu::{
    AdapterChoice, DeviceHandle, DeviceOptions, Error, FeaturesPolicy, GpuVec, HeadlessDevice,
    LimitsPolicy,
};

include!("../../../tests/fixtures/map.rs");

/// The set the environment variable of test 5 names.
const REAL_PAIR_VARIABLE: &str = "GUST_DEVSHARE_REAL_PAIR";

/// A device and queue the caller opened in an instance of its own, with the adapter they came from.
struct Caller {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn caller_instance() -> wgpu::Instance {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    wgpu::Instance::new(descriptor)
}

fn caller_adapter() -> wgpu::Adapter {
    pollster::block_on(
        caller_instance().request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }),
    )
    .expect("a Vulkan adapter is required; this test must not silently skip")
}

fn open_on(
    adapter: wgpu::Adapter,
    features: wgpu::Features,
    limits: wgpu::Limits,
) -> Result<Caller, wgpu::RequestDeviceError> {
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("caller device"),
        required_features: features,
        required_limits: limits,
        ..Default::default()
    }))?;
    Ok(Caller {
        adapter,
        device,
        queue,
    })
}

/// What a renderer does: the features and limits of `device_request`, in its own instance.
fn open_from_request(options: &DeviceOptions) -> Caller {
    let adapter = caller_adapter();
    let request = options
        .device_request(&adapter)
        .expect("the adapter offers the requested capabilities");
    open_on(adapter, request.features, request.limits).expect("the adapter opens a device")
}

fn adopt(caller: Caller, capabilities: &'static [Capability]) -> Result<HeadlessDevice, Error> {
    HeadlessDevice::adopt(&caller.adapter, caller.device, caller.queue, capabilities)
}

fn log_adapter(tag: &str, device: &HeadlessDevice) {
    let info = device.adapter_info();
    println!("{tag} adapter: {} backend: {:?}", info.name, info.backend);
}

/// The kernel of tests 1, 4 and 6: `a * x + y` over 1000 elements, in the range where `f32` is exact.
fn inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..1000).map(|i| (i % 37) as f32 - 18.0).collect();
    let y: Vec<f32> = (0..1000).map(|i| (i % 11) as f32 * 0.5).collect();
    let expected = x.iter().zip(&y).map(|(x, y)| 2.0 * x + y).collect();
    (x, y, expected)
}

#[test]
fn an_adopted_device_runs_a_kernel_like_one_gust_opened_and_like_the_cpu_twin() {
    let (x, y, expected) = inputs();
    let options = DeviceOptions::default().with_backends(wgpu::Backends::VULKAN);
    let adopted = adopt(open_from_request(&options), options.capabilities).expect("adopts");
    log_adapter("adopted", &adopted);
    let on_adopted = saxpy::map_on(&adopted, Broadcast(2.0), &x, &y).unwrap();
    assert_eq!(on_adopted, expected, "the device the caller opened");

    let opened = HeadlessDevice::with_options(options).expect("GUST opens a device");
    log_adapter("opened", &opened);
    let on_opened = saxpy::map_on(&opened, Broadcast(2.0), &x, &y).unwrap();
    assert_eq!(on_adopted, on_opened, "the device GUST opened");

    let on_cpu = saxpy::map_on(&CpuDevice::new(), Broadcast(2.0), &x, &y).unwrap();
    assert_eq!(on_adopted, on_cpu, "the CPU twin");
}

/// The features `options` must ask `adapter` for, written out from the rules and not from `device_request`.
fn expected_features(adapter: &wgpu::Adapter, options: &DeviceOptions) -> wgpu::Features {
    let offered = adapter.features();
    let mut features = wgpu::Features::empty();
    // GUST's own: timestamps and coherent storage wherever the adapter has them.
    for feature in [
        wgpu::Features::TIMESTAMP_QUERY,
        wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        wgpu::Features::MEMORY_DECORATION_COHERENT,
    ] {
        if offered.contains(feature) {
            features |= feature;
        }
    }
    // The policy's: `float32-filterable` under `Adapter` where offered, never under `Default`.
    if options.features == FeaturesPolicy::Adapter
        && offered.contains(wgpu::Features::FLOAT32_FILTERABLE)
    {
        features |= wgpu::Features::FLOAT32_FILTERABLE;
    }
    // The SPIR-V route's: only in a build with the feature, only on Vulkan, and only where offered.
    if cfg!(feature = "spirv-route")
        && adapter.get_info().backend == wgpu::Backend::Vulkan
        && offered.contains(wgpu::Features::PASSTHROUGH_SHADERS)
    {
        features |= wgpu::Features::PASSTHROUGH_SHADERS;
    }
    // A capability's: `SUBGROUP` only for a requested `Subgroups`.
    if options.capabilities.contains(&Capability::Subgroups) {
        features |= wgpu::Features::SUBGROUP;
    }
    features
}

/// The limits: wgpu's defaults, with the adapter's three values under `LimitsPolicy::Adapter`.
fn expected_limits(adapter: &wgpu::Adapter, options: &DeviceOptions) -> wgpu::Limits {
    let mut limits = wgpu::Limits::default();
    if options.limits == LimitsPolicy::Adapter {
        let offered = adapter.limits();
        limits.max_buffer_size = offered.max_buffer_size;
        limits.max_storage_buffer_binding_size = offered.max_storage_buffer_binding_size;
        limits.max_compute_workgroups_per_dimension = offered.max_compute_workgroups_per_dimension;
    }
    limits
}

#[test]
fn device_request_equals_the_hand_written_set_and_the_open_asks_exactly_that() {
    let adapter = caller_adapter();
    let offered = adapter.features();
    println!(
        "oracle adapter: {} offers TIMESTAMP_QUERY={} MEMORY_DECORATION_COHERENT={} FLOAT32_FILTERABLE={} SUBGROUP={}",
        adapter.get_info().name,
        offered.contains(wgpu::Features::TIMESTAMP_QUERY),
        offered.contains(wgpu::Features::MEMORY_DECORATION_COHERENT),
        offered.contains(wgpu::Features::FLOAT32_FILTERABLE),
        offered.contains(wgpu::Features::SUBGROUP),
    );
    let base = DeviceOptions::default()
        .with_backends(wgpu::Backends::VULKAN)
        .with_adapter(AdapterChoice::HighPerformance)
        .with_capabilities(&[]);
    let mut cases = vec![
        base.clone()
            .with_limits(LimitsPolicy::Default)
            .with_features(FeaturesPolicy::Default),
        base.clone()
            .with_limits(LimitsPolicy::Adapter)
            .with_features(FeaturesPolicy::Adapter),
        base.clone()
            .with_limits(LimitsPolicy::Default)
            .with_features(FeaturesPolicy::Adapter),
    ];
    if offered.contains(wgpu::Features::SUBGROUP) {
        println!("oracle branch: SUBGROUP offered, a case with the subgroups capability runs");
        cases.push(base.clone().with_capabilities(&[Capability::Subgroups]));
    } else {
        println!("oracle branch: SUBGROUP not offered, the capability case is test 3's refusal");
    }
    for options in cases {
        let label = format!(
            "limits {:?}, features {:?}, capabilities {:?}",
            options.limits, options.features, options.capabilities
        );
        let features = expected_features(&adapter, &options);
        let limits = expected_limits(&adapter, &options);
        let request = options.device_request(&adapter).expect(&label);
        assert_eq!(request.features, features, "features of {label}");
        assert_eq!(request.limits, limits, "limits of {label}");

        // The open asks for exactly that: the device it opened was granted nothing else and nothing less.
        let opened = HeadlessDevice::with_options(options.clone()).expect(&label);
        assert_eq!(
            opened.wgpu_device().features(),
            features,
            "features granted to the device GUST opened under {label}"
        );
        assert_eq!(
            opened.wgpu_device().limits(),
            limits,
            "limits granted to the device GUST opened under {label}"
        );
        assert_eq!(opened.capabilities(), options.capabilities);

        // Positive seam: a device adopted from the request reports what the adapter offered.
        let caller = open_on(caller_adapter(), request.features, request.limits).expect(&label);
        let adopted = adopt(caller, options.capabilities).expect(&label);
        let timestamps = offered.contains(wgpu::Features::TIMESTAMP_QUERY);
        println!(
            "seam {label}: adapter offers TIMESTAMP_QUERY={timestamps}, supports_gpu_timestamps={}, float32_filterable={}",
            adopted.supports_gpu_timestamps(),
            adopted.float32_filterable()
        );
        assert_eq!(adopted.supports_gpu_timestamps(), timestamps, "{label}");
        assert_eq!(
            adopted.float32_filterable(),
            options.features == FeaturesPolicy::Adapter
                && offered.contains(wgpu::Features::FLOAT32_FILTERABLE),
            "{label}"
        );
        assert_eq!(
            adopted.supports_coherent_storage(),
            offered.contains(wgpu::Features::MEMORY_DECORATION_COHERENT),
            "{label}"
        );
    }
}

#[test]
fn a_device_without_the_features_of_a_capability_is_missing_features() {
    let adapter = caller_adapter();
    let name = adapter.get_info().name;
    let bare =
        open_on(adapter, wgpu::Features::empty(), wgpu::Limits::default()).expect("a bare device");
    let error =
        adopt(bare, &[Capability::Subgroups]).expect_err("no subgroup feature on the device");
    match &error {
        Error::MissingFeatures {
            adapter,
            features,
            needed_by,
        } => {
            assert_eq!(adapter, &name);
            assert_eq!(*features, wgpu::Features::SUBGROUP);
            assert_eq!(needed_by, "the `subgroups` capability");
        }
        other => panic!("expected MissingFeatures, got {other:?}"),
    }
    assert_eq!(
        error.to_string(),
        format!(
            "the device on `{name}` lacks SUBGROUP, needed by the `subgroups` capability; request the features of `DeviceOptions::device_request` when opening it"
        )
    );
    println!("refusal adapter: {name}");
}

#[test]
fn a_bare_device_adopted_without_capabilities_degrades_and_still_runs_a_kernel() {
    let bare = open_on(
        caller_adapter(),
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    )
    .expect("a bare device");
    let adopted = adopt(bare, &[]).expect("adopts: optional features are not required");
    log_adapter("bare", &adopted);
    assert!(!adopted.supports_gpu_timestamps());
    assert!(!adopted.float32_filterable());
    assert!(!adopted.supports_coherent_storage());
    assert_eq!(adopted.capabilities(), &[] as &[Capability]);
    let (x, y, expected) = inputs();
    assert_eq!(
        saxpy::map_on(&adopted, Broadcast(2.0), &x, &y).unwrap(),
        expected
    );
}

/// Whether the opt-in of the real pair is set.
fn real_pair_requested() -> bool {
    std::env::var(REAL_PAIR_VARIABLE).is_ok_and(|value| value == "1")
}

#[test]
fn an_adapter_that_is_not_the_devices_is_an_adapter_mismatch() {
    // The comparison itself runs on two fake `AdapterInfo`s in the crate's unit tests (`headless/adopt.rs`). This is
    // the real pair: two adapters of one instance, the device from the second, the first given to `adopt`.
    if !real_pair_requested() {
        println!(
            "real pair: not run, set {REAL_PAIR_VARIABLE}=1 to opt in (it opens a device on a second adapter)"
        );
        return;
    }
    let instance = caller_instance();
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
    if adapters.len() < 2 {
        println!(
            "real pair: not run, the instance lists {} adapter(s)",
            adapters.len()
        );
        return;
    }
    // Enumeration order, not "the 890M, then the 5090": which card is which depends on the machine. The log names both.
    let (first, second) = (adapters[0].clone(), adapters[1].clone());
    println!(
        "real pair: adapter given to adopt: {}, adapter of the device: {}",
        first.get_info().name,
        second.get_info().name
    );
    assert_ne!(first.get_info(), second.get_info());
    let (device, queue) =
        pollster::block_on(second.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("the second adapter opens a device");
    let error = HeadlessDevice::adopt(&first, device, queue, &[]).expect_err("the wrong adapter");
    match &error {
        Error::AdapterMismatch {
            adapter,
            device_adapter,
        } => {
            assert_eq!(adapter, &first.get_info().name);
            assert_eq!(device_adapter, &second.get_info().name);
        }
        other => panic!("expected AdapterMismatch, got {other:?}"),
    }
}

#[test]
fn two_adoptions_of_one_device_refuse_each_others_buffers() {
    let options = DeviceOptions::default().with_backends(wgpu::Backends::VULKAN);
    let caller = open_from_request(&options);
    let first = HeadlessDevice::adopt(
        &caller.adapter,
        caller.device.clone(),
        caller.queue.clone(),
        options.capabilities,
    )
    .expect("the first adoption");
    let second = HeadlessDevice::adopt(
        &caller.adapter,
        caller.device,
        caller.queue,
        options.capabilities,
    )
    .expect("the second adoption");
    assert_ne!(
        first.id(),
        second.id(),
        "two adoptions are two GUST devices"
    );
    let (first, second): (DeviceHandle, DeviceHandle) =
        (std::sync::Arc::new(first), std::sync::Arc::new(second));

    let on_first = GpuVec::from_slice_on(&first, &[1u32, 2, 3]).unwrap();
    let refused = add_one::map_gpu_on(&second, &on_first).map(|vec| vec.len());
    assert!(
        matches!(refused, Err(MapError::MixedDevices)),
        "{refused:?}"
    );
    let buffer =
        first.create_typed_buffer("mine", &[1u32, 2, 3], gust_wgpu::GpuBufferAccess::ReadWrite);
    match second.read_typed_buffer(&buffer) {
        Err(Error::ForeignPersistentBuffer) => {}
        other => panic!("expected ForeignPersistentBuffer, got {other:?}"),
    }
    // On its own device each works.
    assert_eq!(first.read_typed_buffer(&buffer).unwrap(), [1, 2, 3]);
    assert_eq!(
        add_one::map_gpu_on(&first, &on_first)
            .unwrap()
            .to_vec()
            .unwrap(),
        [2, 3, 4]
    );
}
