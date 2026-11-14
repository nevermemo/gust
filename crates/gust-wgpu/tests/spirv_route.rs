//! The opt-in SPIR-V route (slice m9.rt2; D172, D174), on a real adapter. Compiled only with the Cargo feature
//! `spirv-route`; the target has `required-features` in Cargo.toml, so a feature-off build does not build it at all
//! and the feature-off Gpu suite does not see this file.
//!
//! Three things: the sparse binding layouts that crashed passthrough in the spike (binding 3, 999 and `{0, 999}`), the
//! guards of the emitted Slang (index clamp, division helpers, the literal `%`; D39 limitations pinned at the values
//! naga's guards give), and the fallback record (`HeadlessDevice::route_events`). The workgroup zero-initialisation
//! probes (D174a) are in `workgroup_zero.rs`, which runs on both routes. On a hardware adapter every kernel here must
//! take the SPIR-V route; on a software adapter (lavapipe) either route is accepted as long as the record says why.
#![cfg(feature = "spirv-route")]

use gust::{
    Access, GpuPod, Kernel, KernelDescriptor, ParameterDescriptor, ParameterKind, ResourceBinding,
    ResourceLayout,
};
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice, Route};

fn device() -> HeadlessDevice {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("spirv_route adapter: {}", device.adapter_info().name);
    device
}

/// The route the kernel `qualified` took on `device`, asserting the expectation: SPIR-V on every hardware adapter, and on
/// a software one SPIR-V or WGSL with the recorded reason.
fn assert_spirv(device: &HeadlessDevice, qualified: &str) {
    let events: Vec<_> = device
        .route_events()
        .into_iter()
        .filter(|event| event.kernel == qualified)
        .collect();
    assert_eq!(
        events.len(),
        1,
        "one route event for {qualified}: {events:?}"
    );
    let event = &events[0];
    if device.adapter_info().device_type == wgpu::DeviceType::Cpu {
        if event.route == Route::Wgsl {
            assert!(event.reason.is_some(), "{event:?}");
        }
        return;
    }
    assert_eq!(
        (event.route, event.reason.as_deref()),
        (Route::Spirv, None),
        "{qualified} must take the SPIR-V route on a hardware adapter"
    );
}

// ---- sparse binding layouts (m9.rt1 item 3) ----

const fn storage(name: &'static str, access: Access, binding: u32) -> ParameterDescriptor {
    ParameterDescriptor {
        name,
        rust_type: "float",
        kind: ParameterKind::Storage,
        access,
        resource_layout: Some(ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT)),
        binding: Some(ResourceBinding { group: 0, binding }),
    }
}

/// One read-write buffer at binding N: `p[0] = 17`.
macro_rules! one {
    ($k:ident, $n:literal) => {
        struct $k;
        impl Kernel for $k {
            const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new(
                "spirv_route",
                concat!("one", $n),
                concat!("route_one", $n),
                [64, 1, 1],
                &[storage("p", Access::ReadWrite, $n)],
                concat!(
                    "[[vk::binding(",
                    $n,
                    ", 0)]] RWStructuredBuffer<float> p;\n",
                    "[shader(\"compute\")] [numthreads(64, 1, 1)]\n",
                    "void route_one",
                    $n,
                    "(uint3 id : SV_DispatchThreadID) { if (id.x == 0) { p[0] = 17.0; } }\n"
                ),
            );
        }
    };
}

/// Two buffers (0 and N) in the descriptor; the kernel touches only N, so slangc keeps one binding in the SPIR-V while the
/// layout holds two (the module's numbering is then not the layout's rank without the renumbering).
macro_rules! unused0 {
    ($k:ident, $n:literal) => {
        struct $k;
        impl Kernel for $k {
            const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new(
                "spirv_route",
                concat!("unused0_", $n),
                concat!("route_unused0_", $n),
                [64, 1, 1],
                &[
                    storage("q", Access::ReadWrite, 0),
                    storage("p", Access::ReadWrite, $n),
                ],
                concat!(
                    "[[vk::binding(0, 0)]] RWStructuredBuffer<float> q;\n",
                    "[[vk::binding(",
                    $n,
                    ", 0)]] RWStructuredBuffer<float> p;\n",
                    "[shader(\"compute\")] [numthreads(64, 1, 1)]\n",
                    "void route_unused0_",
                    $n,
                    "(uint3 id : SV_DispatchThreadID) { if (id.x == 0) { p[0] = 17.0; } }\n"
                ),
            );
        }
    };
}

/// Buffers at 0 and N, both written: `q[0] = 19`, `p[0] = 17`.
macro_rules! both {
    ($k:ident, $n:literal) => {
        struct $k;
        impl Kernel for $k {
            const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new(
                "spirv_route",
                concat!("both_", $n),
                concat!("route_both_", $n),
                [64, 1, 1],
                &[storage("q", Access::ReadWrite, 0), storage("p", Access::ReadWrite, $n)],
                concat!(
                    "[[vk::binding(0, 0)]] RWStructuredBuffer<float> q;\n",
                    "[[vk::binding(", $n, ", 0)]] RWStructuredBuffer<float> p;\n",
                    "[shader(\"compute\")] [numthreads(64, 1, 1)]\n",
                    "void route_both_", $n, "(uint3 id : SV_DispatchThreadID) { if (id.x == 0) { p[0] = 17.0; q[0] = 19.0; } }\n"
                ),
            );
        }
    };
}

one!(O3, 3);
one!(O31, 31);
one!(O999, 999);
unused0!(U3, 3);
unused0!(U999, 999);
both!(B999, 999);

/// Run a kernel on `p` (and `q` when `two`); the returned values are `(p[0], q[0])`.
fn run<K: Kernel>(device: &HeadlessDevice, two: bool) -> (f32, f32) {
    let p = device.create_typed_buffer("p", &[3.0f32], GpuBufferAccess::ReadWrite);
    let q = device.create_typed_buffer("q", &[5.0f32], GpuBufferAccess::ReadWrite);
    if two {
        device
            .dispatch_buffers(
                &K::DESCRIPTOR,
                1,
                &[BufferBinding::read_write(&q), BufferBinding::read_write(&p)],
            )
            .unwrap();
    } else {
        device
            .dispatch_buffers(&K::DESCRIPTOR, 1, &[BufferBinding::read_write(&p)])
            .unwrap();
    }
    (
        device.read_typed_buffer(&p).unwrap()[0],
        device.read_typed_buffer(&q).unwrap()[0],
    )
}

#[test]
fn sparse_bindings_run_on_the_spirv_route() {
    let device = device();
    // A single buffer at binding 3, 31 and 999: unrenumbered, a lost write on the RTX 5090 and a crash on the Radeon 890M.
    assert_eq!(run::<O3>(&device, false), (17.0, 5.0));
    assert_eq!(run::<O31>(&device, false), (17.0, 5.0));
    assert_eq!(run::<O999>(&device, false), (17.0, 5.0));
    // {0, N} in the layout with the kernel using N only: slangc prunes binding 0 from the module.
    assert_eq!(run::<U3>(&device, true), (17.0, 5.0));
    assert_eq!(run::<U999>(&device, true), (17.0, 5.0));
    // {0, 999} both used.
    assert_eq!(run::<B999>(&device, true), (17.0, 19.0));
    for qualified in [
        "spirv_route::one3",
        "spirv_route::one31",
        "spirv_route::one999",
        "spirv_route::unused0_3",
        "spirv_route::unused0_999",
        "spirv_route::both_999",
    ] {
        assert_spirv(&device, qualified);
    }
}

// ---- the kernels with a guard ----

#[gust::gpu]
mod probes {
    // Dynamic indices out of range: a local array, a workgroup array, from run-time values (the index clamp).
    #[kernel(workgroup_size(64, 1, 1), cpu = "unsupported")]
    pub fn clamps(
        id: SV_DispatchThreadID,
        lid: LocalId,
        idx: StructuredBuffer<u32>,
        mut local_out: RWStructuredBuffer<u32>,
        mut wg_out: RWStructuredBuffer<u32>,
        mut scratch: Workgroup<[u32; 8]>,
    ) {
        let i = id.x;
        let k = idx[i];
        let mut d: [u32; 4] = [10u32, 20u32, 30u32, 40u32];
        local_out[i] = d[k as usize];
        d[k as usize] = 7u32;
        let mut sum = 0u32;
        let mut weight = 1u32;
        for j in 0u32..4u32 {
            sum += d[j as usize] * weight;
            weight += 2u32;
        }
        local_out[i + 64u32] = sum;
        if lid.x < 8u32 {
            scratch[lid.x] = 100u32 + lid.x;
        }
        workgroup_barrier();
        wg_out[i] = scratch[k];
    }

    // Integer division and remainder: the awkward operands, signed and unsigned, and a literal divisor.
    #[kernel(workgroup_size(64, 1, 1), cpu = "unsupported")]
    pub fn divs(
        id: SV_DispatchThreadID,
        a: StructuredBuffer<i32>,
        b: StructuredBuffer<i32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        let x = a[i];
        let y = b[i];
        let ux = x as u32;
        let uy = y as u32;
        out[i * 6u32] = (x / y) as u32;
        out[i * 6u32 + 1u32] = (x % y) as u32;
        out[i * 6u32 + 2u32] = ux / uy;
        out[i * 6u32 + 3u32] = ux % uy;
        out[i * 6u32 + 4u32] = (x % 3i32) as u32;
        out[i * 6u32 + 5u32] = (x % -3i32) as u32;
    }
}

#[test]
fn out_of_range_array_indices_are_clamped_to_the_last_element() {
    let device = device();
    let n = 64usize;
    // Indices in range, just past the end, far past it, and with the top bit set.
    let idx: Vec<u32> = (0..n as u32)
        .map(|i| match i {
            0..=15 => i % 8,
            16..=23 => 1000 + i,
            24..=31 => u32::MAX - i,
            32..=39 => 0x8000_0000 + i,
            _ => i % 4,
        })
        .collect();
    let idx_b = device.create_typed_buffer("idx", &idx, GpuBufferAccess::ReadOnly);
    let local = device.create_typed_buffer("l", &vec![0u32; 2 * n], GpuBufferAccess::ReadWrite);
    let wg = device.create_typed_buffer("w", &vec![0u32; n], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &probes::clamps::DESCRIPTOR,
            n as u32,
            &[
                BufferBinding::read_only(&idx_b),
                BufferBinding::read_write(&local).independent_length(),
                BufferBinding::read_write(&wg),
            ],
        )
        .unwrap();
    let local = device.read_typed_buffer(&local).unwrap();
    let wg = device.read_typed_buffer(&wg).unwrap();
    for (i, &k) in idx.iter().enumerate() {
        let d = [10u32, 20, 30, 40];
        let kc = (k as usize).min(3);
        assert_eq!(local[i], d[kc], "local read at index {k}");
        let mut written = d;
        written[kc] = 7;
        let sum = written[0] + written[1] * 3 + written[2] * 5 + written[3] * 7;
        assert_eq!(local[i + 64], sum, "local write at index {k}");
        // The workgroup array holds 100 + lid in its first 8 elements; lid < 8 wrote, the rest of the group did not and
        // read an element a lane with lid < 8 wrote (the barrier makes them visible), or the zero it started as.
        let kw = (k as usize).min(7);
        assert_eq!(wg[i], 100 + kw as u32, "workgroup read at index {k}");
    }
    assert_spirv(&device, "probes::clamps");
}

/// Rust-defined values where the divisor is not zero (and not `MIN / -1`); naga's guarded values where it is: `x / 0` is
/// `x`, `x % 0` is `0`, `MIN / -1` is `MIN`, `MIN % -1` is `0` (D39: documented limitations of the GPU, not Rust).
fn expected_divs(x: i32, y: i32) -> [u32; 6] {
    let signed_divisor = if y == 0 || (x == i32::MIN && y == -1) {
        1
    } else {
        y
    };
    let ux = x as u32;
    let uy = y as u32;
    let unsigned_divisor = if uy == 0 { 1 } else { uy };
    [
        (x / signed_divisor) as u32,
        (x % signed_divisor) as u32,
        ux / unsigned_divisor,
        ux % unsigned_divisor,
        (x % 3) as u32,
        (x % -3) as u32,
    ]
}

#[test]
fn division_and_remainder_have_the_guarded_values_and_the_literal_remainder_is_signed() {
    let device = device();
    let values: [i32; 8] = [0, 1, -1, 7, -7, i32::MIN, i32::MAX, 1000];
    let mut a = Vec::new();
    let mut b = Vec::new();
    for &x in &values {
        for &y in &values {
            a.push(x);
            b.push(y);
        }
    }
    let a_b = device.create_typed_buffer("a", &a, GpuBufferAccess::ReadOnly);
    let b_b = device.create_typed_buffer("b", &b, GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer("o", &vec![0u32; a.len() * 6], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &probes::divs::DESCRIPTOR,
            a.len() as u32,
            &[
                BufferBinding::read_only(&a_b),
                BufferBinding::read_only(&b_b),
                BufferBinding::read_write(&out).independent_length(),
            ],
        )
        .unwrap();
    let out = device.read_typed_buffer(&out).unwrap();
    for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(
            &out[i * 6..i * 6 + 6],
            &expected_divs(x, y),
            "x = {x}, y = {y}"
        );
    }
    assert_spirv(&device, "probes::divs");
}

// ---- the fallback record ----

#[test]
fn a_kernel_the_route_took_is_recorded_once_per_device() {
    let device = device();
    assert!(device.route_events().is_empty());
    assert_eq!(run::<O3>(&device, false), (17.0, 5.0));
    assert_eq!(run::<O3>(&device, false), (17.0, 5.0));
    let events = device.route_events();
    assert_eq!(
        events.len(),
        1,
        "the pipeline cache builds it once: {events:?}"
    );
    assert_eq!(events[0].kernel, "spirv_route::one3");
}

/// A backend that is not Vulkan stays on the WGSL route and says why (review M1): DX12, Metal and GL advertise
/// `PASSTHROUGH_SHADERS` but take no SPIR-V, so asking for the feature there would panic at the first kernel. This end
/// to end test can run only where wgpu itself is built with a non-Vulkan backend: this workspace builds wgpu with
/// `vulkan` only (`std`, `vulkan`, `wgsl`), so no DX12 adapter can open on any machine of the project and the test
/// returns early. The decision itself (a backend that is not Vulkan stays on WGSL) is covered by the unit test in
/// `passthrough.rs`; this test matters to a downstream build whose own wgpu dependency enables `dx12`, `metal` or `gles`.
#[test]
fn a_dx12_device_falls_back_to_wgsl_with_the_backend_reason() {
    let options = gust_wgpu::DeviceOptions::default().with_backends(wgpu::Backends::DX12);
    let Ok(device) = HeadlessDevice::with_options(options) else {
        eprintln!(
            "spirv_route: no DX12 adapter can open: the workspace builds wgpu with Vulkan only, so the end-to-end \
             fallback was not run (the decision is covered by the unit test)"
        );
        return;
    };
    eprintln!(
        "spirv_route: DX12 adapter {} ({:?})",
        device.adapter_info().name,
        device.adapter_info().backend
    );
    assert_eq!(run::<O3>(&device, false), (17.0, 5.0));
    let events = device.route_events();
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(events[0].route, Route::Wgsl, "{events:?}");
    assert_eq!(
        events[0].reason.as_deref(),
        Some("backend is not Vulkan"),
        "{events:?}"
    );
}
