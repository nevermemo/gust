//! More than 8 storage buffers per shader stage (slice m14.lim-a, D84 as amended): `DeviceOptions::with_storage_buffers`.
//!
//! The option is off by default, takes a number the developer states, requests it from the adapter and refuses at open
//! an adapter that offers fewer. These tests run a plain `par_for!` kernel with ten storage buffers on a device opened
//! with 10 and with 16, compare it with the CPU twin and with an independent host reference (outputs sentinel-filled
//! first), and pin the refusals: the default device, the adapter that offers fewer, a number below 8, `with_limits`,
//! `device_request`, `adopt`, and the generated snippet's comment. Every test runs on the GPU the environment selects
//! and fails without an adapter, like the rest of the Gpu suite.

mod common;

use common::assert_cold;

use gust::{
    Access, KernelDescriptor, ParameterDescriptor, ParameterKind, ResourceBinding, ResourceLayout,
};
use gust_wgpu::{DeviceOptions, Error, HeadlessDevice, LimitsPolicy};

/// The storage buffers of the kernel below: five read, five written.
const BUFFERS: u32 = 10;

/// One `par_for!` over ten slices, written once for each backend: `device` is the type of the `on(..)` argument.
macro_rules! ten_buffer_loop {
    ($device:ty) => {
        #[allow(clippy::too_many_arguments)]
        pub fn run(
            a0: &[u32],
            a1: &[u32],
            a2: &[u32],
            a3: &[u32],
            a4: &[u32],
            o0: &mut [u32],
            o1: &mut [u32],
            o2: &mut [u32],
            o3: &mut [u32],
            o4: &mut [u32],
            device: &$device,
        ) -> Result<(), gust::map::Error> {
            let n = o0.len() as u32;
            gust::par_for!(on(device); i in 0..n, [
                a0: &[u32], a1: &[u32], a2: &[u32], a3: &[u32], a4: &[u32],
                o0: &mut [u32], o1: &mut [u32], o2: &mut [u32], o3: &mut [u32], o4: &mut [u32]
            ] {
                o0[i] = a0[i] * 3u32 + a1[i];
                o1[i] = a1[i] * 3u32 + a2[i] + 1u32;
                o2[i] = a2[i] * 3u32 + a3[i] + 2u32;
                o3[i] = a3[i] * 3u32 + a4[i] + 3u32;
                o4[i] = a4[i] * 3u32 + a0[i] + 4u32;
            })
        }
    };
}

mod on_gpu {
    ten_buffer_loop!(gust_wgpu::HeadlessDevice);
}

mod on_twin {
    ten_buffer_loop!(gust::cpu::CpuDevice);
}

/// A value no kernel writes, to show an element the loop never reached.
const UNWRITTEN: u32 = 0xDEAD_BEEF;
const ELEMENTS: usize = 1000;

fn inputs() -> [Vec<u32>; 5] {
    std::array::from_fn(|k| {
        (0..ELEMENTS as u32)
            .map(|i| i * (k as u32 + 1) + 7)
            .collect()
    })
}

/// The independent host reference: the five outputs of the loop body.
fn reference(a: &[Vec<u32>; 5]) -> [Vec<u32>; 5] {
    std::array::from_fn(|k| {
        (0..ELEMENTS)
            .map(|i| a[k][i] * 3 + a[(k + 1) % 5][i] + k as u32)
            .collect()
    })
}

fn sentinels() -> [Vec<u32>; 5] {
    std::array::from_fn(|_| vec![UNWRITTEN; ELEMENTS])
}

fn run_on_gpu(device: &HeadlessDevice) -> Result<[Vec<u32>; 5], gust::map::Error> {
    let a = inputs();
    let [mut o0, mut o1, mut o2, mut o3, mut o4] = sentinels();
    on_gpu::run(
        &a[0], &a[1], &a[2], &a[3], &a[4], &mut o0, &mut o1, &mut o2, &mut o3, &mut o4, device,
    )?;
    Ok([o0, o1, o2, o3, o4])
}

fn run_on_twin() -> [Vec<u32>; 5] {
    let a = inputs();
    let [mut o0, mut o1, mut o2, mut o3, mut o4] = sentinels();
    on_twin::run(
        &a[0],
        &a[1],
        &a[2],
        &a[3],
        &a[4],
        &mut o0,
        &mut o1,
        &mut o2,
        &mut o3,
        &mut o4,
        &gust::cpu::CpuDevice::new(),
    )
    .expect("the twin runs the loop");
    [o0, o1, o2, o3, o4]
}

fn options(n: u32) -> DeviceOptions {
    DeviceOptions::default()
        .with_backends(wgpu::Backends::VULKAN)
        .with_storage_buffers(n)
}

/// A device opened with `n` storage buffers, or `None` after a printed `SKIP` line when the adapter offers fewer
/// (a software adapter, say): the test then says so by name instead of failing for want of a feature it cannot have.
fn open(n: u32) -> Option<HeadlessDevice> {
    open_with(options(n))
}

/// [`open`] for options that already state their number.
fn open_with(options: DeviceOptions) -> Option<HeadlessDevice> {
    match HeadlessDevice::with_options(options.clone()) {
        Ok(device) => Some(device),
        Err(Error::DeviceLimit { maximum, .. }) => {
            println!(
                "SKIP {:?}: the adapter offers {maximum} storage buffers per stage",
                options.storage_buffers
            );
            None
        }
        Err(error) => panic!("{:?}: {error}", options.storage_buffers),
    }
}

/// A refused `par_for!` launch built no pipeline (`assert_cold`) and left no pooled buffer held. The loop's front door
/// stages its ten slices in pooled buffers before it launches (measured: ten pool misses), so the limit check cannot
/// precede those; it precedes the pipeline, and the refusal leaves nothing held (`outstanding == 0`).
#[track_caller]
fn assert_nothing_built(device: &HeadlessDevice, why: &str) {
    assert_cold(device, why);
    assert_eq!(
        device.buffer_pool_stats().outstanding,
        0,
        "{why} must leave no pooled buffer held"
    );
}

/// A refused launch of buffers the caller created itself (`dispatch_buffers`) takes nothing from the pool: the check
/// precedes the pipeline and any buffer of GUST's own.
#[track_caller]
fn assert_no_pool_buffer_taken(device: &HeadlessDevice, why: &str) {
    assert_cold(device, why);
    let pool = device.buffer_pool_stats();
    assert_eq!(
        (pool.misses, pool.outstanding),
        (0, 0),
        "{why} must precede any pooled buffer"
    );
}

/// The limit error of a `par_for!` call, with the loop's numbers.
#[track_caller]
fn device_limit(error: &gust::map::Error) -> (&str, u64, u64) {
    match error.downcast_ref::<Error>() {
        Some(Error::DeviceLimit {
            limit,
            requested,
            maximum,
            ..
        }) => (limit, *requested, *maximum),
        other => panic!("expected a DeviceLimit, got {other:?} ({error})"),
    }
}

#[test]
fn a_default_device_refuses_the_ten_buffer_kernel_before_any_pipeline() {
    let device = common::device();
    assert_eq!(device.limits().max_storage_buffers_per_shader_stage, 8);
    let a = inputs();
    let [mut o0, mut o1, mut o2, mut o3, mut o4] = sentinels();
    let error = on_gpu::run(
        &a[0], &a[1], &a[2], &a[3], &a[4], &mut o0, &mut o1, &mut o2, &mut o3, &mut o4, &device,
    )
    .expect_err("ten storage buffers exceed the default 8");
    assert_eq!(
        device_limit(&error),
        ("max_storage_buffers_per_shader_stage", 10, 8)
    );
    assert_nothing_built(&device, "a storage buffer limit error");
    for output in [&o0, &o1, &o2, &o3, &o4] {
        assert!(output.iter().all(|&word| word == UNWRITTEN));
    }
}

#[test]
fn a_device_opened_with_ten_or_sixteen_runs_the_kernel_equal_to_the_twin() {
    let twin = run_on_twin();
    assert_eq!(twin, reference(&inputs()), "the twin against the host");
    for n in [BUFFERS, 16] {
        let Some(device) = open(n) else {
            continue;
        };
        println!(
            "n={n} adapter: {}; granted {}",
            device.adapter_info().name,
            device.limits().max_storage_buffers_per_shader_stage
        );
        assert_eq!(device.limits().max_storage_buffers_per_shader_stage, n);
        let outputs = run_on_gpu(&device).unwrap_or_else(|error| panic!("n={n}: {error}"));
        assert_eq!(outputs, twin, "n={n}: the GPU against the twin");
        assert_eq!(outputs, reference(&inputs()), "n={n}: against the host");
    }
}

#[test]
fn the_launch_check_follows_the_limit_the_device_was_opened_with_at_n_and_n_plus_one() {
    // The kernel binds 10. A device opened with 10 and lowered to 9 refuses it (one over the limit), the same device
    // at 10 runs it (exactly at the limit, `a_device_opened_with_ten...`), and 9 is a legal request (8 is the floor).
    let Some(device) = open(BUFFERS) else {
        return;
    };
    let mut limits = device.limits().clone();
    limits.max_storage_buffers_per_shader_stage = BUFFERS - 1;
    let lowered = device.with_limits(limits).expect("lowering is allowed");
    let error = run_on_gpu(&lowered).expect_err("ten buffers exceed the lowered 9");
    assert_eq!(
        device_limit(&error),
        ("max_storage_buffers_per_shader_stage", 10, 9)
    );
    assert_nothing_built(&lowered, "a storage buffer limit error");

    let nine = open(BUFFERS - 1).expect("9 is offered wherever 10 is");
    let error = run_on_gpu(&nine).expect_err("ten buffers exceed a device opened with 9");
    assert_eq!(
        device_limit(&error),
        ("max_storage_buffers_per_shader_stage", 10, 9)
    );
    assert_nothing_built(&nine, "a storage buffer limit error");
    // `with_limits` cannot raise above what the device was opened with.
    let mut limits = nine.limits().clone();
    limits.max_storage_buffers_per_shader_stage += 1;
    match nine.with_limits(limits) {
        Err(Error::LimitRaised {
            limit,
            requested,
            device,
        }) => assert_eq!(
            (limit, requested, device),
            ("max_storage_buffers_per_shader_stage", 10, 9)
        ),
        other => panic!("expected LimitRaised, got {:?}", other.err()),
    }
}

#[test]
fn a_number_above_what_the_adapter_offers_is_device_limit_before_any_device_exists() {
    match HeadlessDevice::with_options(options(u32::MAX)) {
        Err(Error::DeviceLimit {
            limit,
            requested,
            maximum,
            kernel,
        }) => {
            assert_eq!(limit, "max_storage_buffers_per_shader_stage");
            assert_eq!(requested, u64::from(u32::MAX));
            assert!(
                maximum >= u64::from(BUFFERS),
                "the adapter offers {maximum}"
            );
            assert!(kernel.contains("with_storage_buffers"), "{kernel}");
        }
        other => panic!("expected DeviceLimit, got {:?}", other.err()),
    }
}

#[test]
fn a_number_below_8_is_refused_by_name() {
    for n in [0, 1, 7] {
        match HeadlessDevice::with_options(options(n)) {
            Err(Error::RequestDevice(message)) => {
                assert!(
                    message.contains(&format!("with_storage_buffers({n})"))
                        && message.contains("at least 8"),
                    "{message}"
                );
            }
            other => panic!("n={n}: expected RequestDevice, got {:?}", other.err()),
        }
    }
    // 8 itself is the default and is accepted.
    let device = HeadlessDevice::with_options(options(8)).expect("8 is the default");
    assert_eq!(device.limits().max_storage_buffers_per_shader_stage, 8);
}

#[test]
fn the_option_is_off_by_default_and_every_default_open_keeps_8() {
    assert_eq!(DeviceOptions::default().storage_buffers, None);
    assert_eq!(options(10).storage_buffers, Some(10));
    let default_limits = wgpu::Limits::default();
    assert_eq!(default_limits.max_storage_buffers_per_shader_stage, 8);
    for (what, device) in [
        ("HeadlessDevice::new", common::device()),
        (
            "DeviceOptions::default",
            HeadlessDevice::with_options(
                DeviceOptions::default().with_backends(wgpu::Backends::VULKAN),
            )
            .expect("the default options open"),
        ),
        (
            "LimitsPolicy::Default",
            HeadlessDevice::with_options(
                DeviceOptions::default()
                    .with_backends(wgpu::Backends::VULKAN)
                    .with_limits(LimitsPolicy::Default),
            )
            .expect("the default policy opens"),
        ),
    ] {
        assert_eq!(
            device.limits().max_storage_buffers_per_shader_stage,
            8,
            "{what}"
        );
    }
    // The option is valid under either policy.
    for policy in [LimitsPolicy::Default, LimitsPolicy::Adapter] {
        let Some(device) = open_with(options(BUFFERS).with_limits(policy)) else {
            continue;
        };
        assert_eq!(
            device.limits().max_storage_buffers_per_shader_stage,
            BUFFERS
        );
    }
}

/// A caller's device, opened in its own Vulkan instance on the high-performance adapter (the one `HeadlessDevice::new`
/// opens) from what `device_request` asks for under `options`; `None` after a printed `SKIP` when the adapter offers
/// fewer storage buffers than the options state.
fn caller_device(options: &DeviceOptions) -> Option<(wgpu::Limits, HeadlessDevice)> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("a Vulkan adapter is required; this test must not silently skip");
    let request = match options.device_request(&adapter) {
        Ok(request) => request,
        Err(Error::DeviceLimit { maximum, .. }) => {
            println!(
                "SKIP {:?}: the adapter offers {maximum} storage buffers per stage",
                options.storage_buffers
            );
            return None;
        }
        Err(error) => panic!("the request is accepted: {error}"),
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: request.features,
        required_limits: request.limits.clone(),
        ..Default::default()
    }))
    .expect("the caller's device opens");
    let adopted = HeadlessDevice::adopt(&adapter, device, queue, gust::capability::enabled())
        .expect("adopt refuses nothing about limits");
    Some((request.limits, adopted))
}

#[test]
fn device_request_carries_n_and_an_adopted_device_follows_what_its_caller_opened() {
    let Some((limits, raised)) =
        caller_device(&DeviceOptions::default().with_storage_buffers(BUFFERS))
    else {
        return;
    };
    assert_eq!(limits.max_storage_buffers_per_shader_stage, BUFFERS);
    assert_eq!(
        raised.limits().max_storage_buffers_per_shader_stage,
        BUFFERS
    );
    let outputs = run_on_gpu(&raised).expect("the adopted device opened with 10 runs the kernel");
    assert_eq!(outputs, reference(&inputs()));

    let (limits, plain) = caller_device(&DeviceOptions::default()).expect("the default request");
    assert_eq!(limits.max_storage_buffers_per_shader_stage, 8);
    let error = run_on_gpu(&plain).expect_err("an adopted default device refuses ten buffers");
    assert_eq!(
        device_limit(&error),
        ("max_storage_buffers_per_shader_stage", 10, 8)
    );
    assert_nothing_built(&plain, "a storage buffer limit error");
}

/// A hand-written kernel with `count` read-write storage parameters, for the snippet and the precompiled checks.
fn kernel(count: usize) -> KernelDescriptor {
    let parameters = (0..count)
        .map(|index| ParameterDescriptor {
            name: Box::leak(format!("p{index}").into_boxed_str()),
            rust_type: "StorageMut<f32>",
            kind: ParameterKind::Storage,
            access: Access::ReadWrite,
            resource_layout: Some(ResourceLayout::storage_v1(<f32 as gust::GpuPod>::LAYOUT)),
            binding: Some(ResourceBinding {
                group: 0,
                binding: index as u32,
            }),
        })
        .collect::<Vec<_>>();
    let mut source = String::new();
    for index in 0..count {
        source.push_str(&format!(
            "[[vk::binding({index}, 0)]] RWStructuredBuffer<float> p{index};\n"
        ));
    }
    source.push_str(
        "[shader(\"compute\")] [numthreads(64, 1, 1)]\n\
         void probe_k(uint3 id : SV_DispatchThreadID) { if (id.x == 0) p0[0] = 17.0; }\n",
    );
    KernelDescriptor::new(
        "storage_buffers_test",
        "k",
        "probe_k",
        [64, 1, 1],
        Box::leak(parameters.into_boxed_slice()),
        Box::leak(source.into_boxed_str()),
    )
}

#[test]
fn the_snippet_of_a_kernel_over_8_names_the_device_it_needs_and_others_keep_their_bytes() {
    let comment = "// This kernel binds 10 storage buffers: its device must be opened with max_storage_buffers_per_shader_stage of at least 10";
    let over = gust_wgpu::render_wgpu_source(&kernel(10)).expect("the snippet renders");
    assert_eq!(over.matches(comment).count(), 1, "{over}");
    for count in [1, 8] {
        let text = gust_wgpu::render_wgpu_source(&kernel(count)).expect("the snippet renders");
        assert!(!text.contains("storage buffers: its device"), "{count}");
        assert!(
            !text.contains("with_storage_buffers"),
            "a kernel at {count} buffers keeps its bytes"
        );
    }
    // The comment starts at 9 buffers, and is the one line that names the option.
    let nine = gust_wgpu::render_wgpu_source(&kernel(9)).expect("the snippet renders");
    assert_eq!(
        nine.lines()
            .filter(|line| line.contains("with_storage_buffers"))
            .count(),
        1
    );
}

// The precompiled half of check 6. The kernel carries a WGSL artifact (the fixture of `tests/precompiled.rs`, which
// belongs to another kernel and is never verified here: the limit check runs before any artifact is looked at), so a
// build that has only artifacts takes this path.
const WGSL: &[u8] = include_bytes!("fixtures/precompiled/52e85c5f03f72272.wgsl");
const REFLECTION: &str = include_str!("fixtures/precompiled/52e85c5f03f72272.wgsl.reflection.json");

#[test]
fn a_ten_binding_kernel_with_an_artifact_on_a_default_device_is_a_typed_device_limit() {
    let artifact = gust::Artifact::new(gust::Format::Wgsl, "52e85c5f03f72272", WGSL, REFLECTION);
    let descriptor = kernel(10).with_artifacts(Box::leak(Box::new([artifact])));
    let device = common::device();
    let buffers = (0..10)
        .map(|index| {
            device.create_typed_buffer(
                &format!("p{index}"),
                &[3.0_f32],
                gust_wgpu::GpuBufferAccess::ReadWrite,
            )
        })
        .collect::<Vec<_>>();
    #[allow(deprecated)]
    let bindings = buffers
        .iter()
        .map(gust_wgpu::BufferBinding::read_write)
        .collect::<Vec<_>>();
    #[allow(deprecated)]
    let error = device
        .dispatch_buffers(&descriptor, 1, &bindings)
        .expect_err("ten bindings exceed the default 8");
    match error {
        Error::DeviceLimit {
            limit,
            requested,
            maximum,
            ..
        } => assert_eq!(
            (limit, requested, maximum),
            ("max_storage_buffers_per_shader_stage", 10, 8)
        ),
        other => panic!("expected the typed DeviceLimit, got {other:?}"),
    }
    assert_no_pool_buffer_taken(&device, "a storage buffer limit error");
}
