//! Device options (m7.a0): adapter enumeration and choice, the two limits policies, and the process's default device.
//!
//! S7.6 numbers are printed (`--nocapture`): the limits each policy was granted on the adapters of this machine.
//! The tests assert relations (the adapter policy grants what the adapter offers, the default policy grants wgpu's
//! defaults), not the numbers of one GPU.

use gust::capability::Capability;
use gust_wgpu::{
    AdapterChoice, DeviceOptions, Error, FeaturesPolicy, HeadlessDevice, LimitsPolicy,
};

fn vulkan() -> wgpu::Backends {
    wgpu::Backends::VULKAN
}

fn options(adapter: AdapterChoice, limits: LimitsPolicy) -> DeviceOptions {
    DeviceOptions::default()
        .with_backends(vulkan())
        .with_adapter(adapter)
        .with_limits(limits)
        .with_features(FeaturesPolicy::Default)
}

#[test]
fn enumerate_lists_the_vulkan_adapters() {
    let adapters = HeadlessDevice::enumerate(vulkan());
    assert!(!adapters.is_empty(), "a Vulkan adapter is required");
    for (index, info) in adapters.iter().enumerate() {
        println!(
            "adapter {index}: {:?} ({:?}, {:?})",
            info.name, info.device_type, info.backend
        );
        assert!(!info.name.is_empty());
        assert_eq!(info.backend, wgpu::Backend::Vulkan);
    }
    // The same call twice lists the same adapters in the same order: `Index` counts by it.
    assert_eq!(adapters, HeadlessDevice::enumerate(vulkan()));
}

#[test]
fn an_adapter_is_chosen_by_index_and_by_name() {
    let adapters = HeadlessDevice::enumerate(vulkan());
    let first = &adapters[0];
    let by_index =
        HeadlessDevice::with_options(options(AdapterChoice::Index(0), LimitsPolicy::Default))
            .expect("the first adapter opens");
    assert_eq!(by_index.adapter_info().name, first.name);

    // Upper case, and only a part of the name: the match ignores case and takes a substring.
    let fragment = first
        .name
        .chars()
        .take(6)
        .collect::<String>()
        .to_uppercase();
    let by_name = HeadlessDevice::with_options(options(
        AdapterChoice::NameContains(fragment),
        LimitsPolicy::Default,
    ))
    .expect("an adapter whose name contains the fragment opens");
    assert!(
        by_name.adapter_info().name.to_lowercase().starts_with(
            &first
                .name
                .chars()
                .take(6)
                .collect::<String>()
                .to_lowercase()
        )
    );
}

#[test]
fn an_adapter_that_does_not_exist_is_no_adapter() {
    let missing = HeadlessDevice::enumerate(vulkan()).len();
    match HeadlessDevice::with_options(options(
        AdapterChoice::Index(missing),
        LimitsPolicy::Default,
    )) {
        Err(Error::NoAdapter(message)) => {
            assert!(
                message.contains("Index"),
                "the message names the choice: {message}"
            );
        }
        Err(other) => panic!("expected NoAdapter, got {other:?}"),
        Ok(_) => panic!("index {missing} is one past the adapters, so no device can open"),
    }
    match HeadlessDevice::with_options(options(
        AdapterChoice::NameContains("no such adapter anywhere".to_owned()),
        LimitsPolicy::Default,
    )) {
        Err(Error::NoAdapter(_)) => {}
        Err(other) => panic!("expected NoAdapter, got {other:?}"),
        Ok(_) => panic!("no adapter has that name"),
    }
}

#[test]
fn both_power_preferences_open_a_device() {
    for choice in [AdapterChoice::HighPerformance, AdapterChoice::LowPower] {
        let device = HeadlessDevice::with_options(options(choice.clone(), LimitsPolicy::Default))
            .unwrap_or_else(|error| panic!("{choice:?} did not open: {error}"));
        println!("{choice:?}: {:?}", device.adapter_info().name);
    }
}

/// S7.6: what each policy grants, printed for every adapter of the machine, and the relations between them.
#[test]
fn the_adapter_policy_grants_the_adapters_buffer_limits() {
    let defaults = wgpu::Limits::default();
    for index in 0..HeadlessDevice::enumerate(vulkan()).len() {
        let narrow = HeadlessDevice::with_options(options(
            AdapterChoice::Index(index),
            LimitsPolicy::Default,
        ))
        .expect("the adapter opens with default limits");
        let wide = HeadlessDevice::with_options(options(
            AdapterChoice::Index(index),
            LimitsPolicy::Adapter,
        ))
        .expect("the adapter opens with its own limits");
        let (narrow_limits, wide_limits) = (narrow.limits(), wide.limits());
        println!(
            "S7.6 adapter {index} {:?}: Default max_buffer_size={} max_storage_buffer_binding_size={} \
             max_compute_workgroups_per_dimension={}",
            narrow.adapter_info().name,
            narrow_limits.max_buffer_size,
            narrow_limits.max_storage_buffer_binding_size,
            narrow_limits.max_compute_workgroups_per_dimension,
        );
        println!(
            "S7.6 adapter {index} {:?}: Adapter max_buffer_size={} max_storage_buffer_binding_size={} \
             max_compute_workgroups_per_dimension={}",
            wide.adapter_info().name,
            wide_limits.max_buffer_size,
            wide_limits.max_storage_buffer_binding_size,
            wide_limits.max_compute_workgroups_per_dimension,
        );
        assert_eq!(
            narrow_limits, &defaults,
            "the Default policy is wgpu's defaults"
        );
        assert!(wide_limits.max_buffer_size >= defaults.max_buffer_size);
        assert!(
            wide_limits.max_storage_buffer_binding_size >= defaults.max_storage_buffer_binding_size
        );
        assert!(
            wide_limits.max_compute_workgroups_per_dimension
                >= defaults.max_compute_workgroups_per_dimension
        );
        // Every limit but these three stays at wgpu's default.
        let mut rest = wide_limits.clone();
        rest.max_buffer_size = defaults.max_buffer_size;
        rest.max_storage_buffer_binding_size = defaults.max_storage_buffer_binding_size;
        rest.max_compute_workgroups_per_dimension = defaults.max_compute_workgroups_per_dimension;
        assert_eq!(
            rest, defaults,
            "only the three buffer and dispatch limits are the adapter's"
        );

        // `with_limits` keeps lowering from what was granted, and can raise back up to it.
        let granted = wide_limits.clone();
        let mut lower = granted.clone();
        lower.max_buffer_size = 1 << 20;
        let wide = wide.with_limits(lower).expect("lowering is allowed");
        assert_eq!(wide.limits().max_buffer_size, 1 << 20);
        let wide = wide
            .with_limits(granted.clone())
            .expect("raising back is allowed");
        assert_eq!(wide.limits(), &granted);
    }
}

#[test]
fn the_default_options_ask_for_the_adapters_limits_on_native() {
    let defaults = DeviceOptions::default();
    assert_eq!(defaults.backends, vulkan());
    assert_eq!(defaults.adapter, AdapterChoice::HighPerformance);
    assert_eq!(defaults.limits, LimitsPolicy::Adapter);
    // D90: the optional features follow the same two defaults as the limits.
    assert_eq!(defaults.features, FeaturesPolicy::Adapter);
    // `HeadlessDevice::new` is unchanged: wgpu's default limits, and no optional feature. It requests the capabilities
    // of the build, as the default options do (m8.w2; `tests/capabilities.rs` pins that under both profiles).
    let plain = HeadlessDevice::new().expect("a Vulkan adapter");
    assert_eq!(plain.limits(), &wgpu::Limits::default());
    assert!(!plain.float32_filterable());
    assert_eq!(plain.capabilities(), gust::capability::enabled());
}

/// Capabilities (M8, m8.w1, m8.w2): the options hold the build's set (`gust::capability::enabled()`) until asked
/// otherwise, `with_capabilities` sets exactly the set it is given and nothing else, and a device reports wgpu's
/// subgroup range for the adapter whether or not it was opened with the capability.
/// A request the adapter cannot meet is unit-tested over fake feature sets (`required_capability_features`).
#[test]
fn capabilities_are_an_option_of_their_own() {
    // The line that says what the default is: the build's own set, empty under `dev` and `[Subgroups]` under
    // `dev+subgroups`. Everything below opens with an explicit set, so it does not depend on the profile.
    assert_eq!(
        DeviceOptions::default().capabilities,
        gust::capability::enabled()
    );
    let none = DeviceOptions::default().with_capabilities(&[]);
    let asked = none.clone().with_capabilities(&[Capability::Subgroups]);
    assert_eq!(asked.capabilities, &[Capability::Subgroups]);
    assert_eq!(asked.clone().with_capabilities(&[]), none);

    let plain = HeadlessDevice::with_options(
        options(AdapterChoice::Index(0), LimitsPolicy::Default).with_capabilities(&[]),
    )
    .expect("the first adapter opens");
    assert!(plain.capabilities().is_empty());
    let info = plain.adapter_info();
    if let Some((smallest, largest)) = plain.subgroup_sizes() {
        // The range is the one wgpu reports for the adapter, read from its description.
        assert_eq!(
            (smallest, largest),
            (info.subgroup_min_size, info.subgroup_max_size)
        );
        assert!(smallest <= largest);
    }
    println!(
        "{:?}: subgroup sizes {:?}",
        info.name,
        plain.subgroup_sizes()
    );
}

/// The one test of this file that touches the process-wide default device: options set before the first use apply to
/// it; setting them afterwards is an error; every call returns the same device.
#[test]
fn the_default_device_opens_once_with_the_options_set_before_it() {
    gust_wgpu::set_default_options(options(
        AdapterChoice::HighPerformance,
        LimitsPolicy::Default,
    ))
    .expect("options can be set before the first use");
    // Setting them twice before use is allowed: the last one wins.
    gust_wgpu::set_default_options(options(AdapterChoice::Index(0), LimitsPolicy::Default))
        .expect("options can be replaced before the first use");
    let first = gust_wgpu::default_device().expect("the default device opens");
    assert_eq!(first.limits(), &wgpu::Limits::default());
    // `set_default_options` governs the optional features too (D90): `Default` turns `float32-filterable` off even
    // on an adapter that offers it.
    assert!(!first.float32_filterable());
    // The options came from `DeviceOptions::default()`, so the device holds the build's capabilities (m8.w2).
    assert_eq!(first.capabilities(), gust::capability::enabled());
    let second = gust_wgpu::default_device().expect("and is still there");
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "one device per process"
    );
    assert_eq!(first.id(), second.id());
    match gust_wgpu::set_default_options(DeviceOptions::default()) {
        Err(Error::DefaultDeviceOpened) => {}
        other => panic!("expected DefaultDeviceOpened after the first use, got {other:?}"),
    }
    let from_async = pollster::block_on(gust_wgpu::default_device_async()).expect("same device");
    assert!(std::sync::Arc::ptr_eq(&first, &from_async));
}
