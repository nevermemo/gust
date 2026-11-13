//! Workgroup memory starts as zeros (D174a), on whichever route the build takes (slice m9.rt2, review S2).
//!
//! SEMANTICS promises that a read of workgroup memory before any write gives zero. With the `spirv-route` feature off
//! that rests on naga's own zero-initialisation of the WGSL route, so this file is not gated: the Gpu suite checks the
//! promise on the default route too, and a change of naga's default would fail here. With the feature on the guards
//! are in the emitted Slang, and the probes also assert that the kernels took the SPIR-V route (a software adapter
//! may take either, as long as the record says why).

use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

fn device() -> HeadlessDevice {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    eprintln!("workgroup_zero adapter: {}", device.adapter_info().name);
    device
}

/// With the feature on, the kernel `qualified` took the SPIR-V route on a hardware adapter. Without it, nothing: the
/// kernels are on the WGSL route by construction.
#[cfg(feature = "spirv-route")]
fn assert_spirv(device: &HeadlessDevice, qualified: &str) {
    use gust_wgpu::Route;
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

#[cfg(not(feature = "spirv-route"))]
fn assert_spirv(_device: &HeadlessDevice, _qualified: &str) {}

#[gust::gpu]
mod probes {
    // Workgroup memory read before any write (D174a). `wg_dirty` fills the group with a pattern first; `wg_read_first`
    // reads before writing, in a dispatch that may reuse the same shared memory.
    #[kernel(workgroup_size(64, 1, 1), cpu = "unsupported")]
    pub fn wg_dirty(
        id: SV_DispatchThreadID,
        lid: LocalId,
        mut out: RWStructuredBuffer<u32>,
        mut scratch: Workgroup<[u32; 64]>,
    ) {
        scratch[lid.x] = 0xDEADBEEFu32;
        workgroup_barrier();
        out[id.x] = scratch[(lid.x + 1u32) % 64u32];
    }

    #[kernel(workgroup_size(64, 1, 1), cpu = "unsupported")]
    pub fn wg_read_first(
        id: SV_DispatchThreadID,
        lid: LocalId,
        mut out: RWStructuredBuffer<u32>,
        scratch: Workgroup<[u32; 64]>,
    ) {
        out[id.x] = scratch[lid.x];
    }

    // A workgroup array that is not a multiple of the group, of a vector type and a bool: zero loop edge cases.
    #[kernel(workgroup_size(64, 1, 1), cpu = "unsupported")]
    pub fn wg_odd_dirty(
        id: SV_DispatchThreadID,
        lid: LocalId,
        mut out: RWStructuredBuffer<u32>,
        mut a: Workgroup<[Vec2; 100]>,
        mut b: Workgroup<[bool; 3]>,
    ) {
        a[lid.x] = Vec2::new(1.0f32, 2.0f32);
        if lid.x < 3u32 {
            b[lid.x] = true;
        }
        workgroup_barrier();
        out[id.x] = 1u32;
    }

    #[kernel(workgroup_size(64, 1, 1), cpu = "unsupported")]
    pub fn wg_odd_read_first(
        id: SV_DispatchThreadID,
        lid: LocalId,
        mut out: RWStructuredBuffer<u32>,
        a: Workgroup<[Vec2; 100]>,
        b: Workgroup<[bool; 3]>,
    ) {
        let mut seen = 0u32;
        if a[lid.x].x != 0.0f32 || a[lid.x + 36u32].y != 0.0f32 {
            seen += 1u32;
        }
        if b[lid.x % 3u32] {
            seen += 2u32;
        }
        out[id.x] = seen;
    }
}

fn dispatch_zero_probe(
    device: &HeadlessDevice,
    dirty: &gust::KernelDescriptor,
    read_first: &gust::KernelDescriptor,
    groups: usize,
) -> Vec<Vec<u32>> {
    let n = groups * 64;
    let mut rounds = Vec::new();
    for _ in 0..3 {
        let dirty_out = device.create_typed_buffer("d", &vec![0u32; n], GpuBufferAccess::ReadWrite);
        device
            .dispatch_buffers(dirty, n as u32, &[BufferBinding::read_write(&dirty_out)])
            .unwrap();
        let _ = device.read_typed_buffer(&dirty_out).unwrap();
        let out =
            device.create_typed_buffer("c", &vec![0x5555_5555u32; n], GpuBufferAccess::ReadWrite);
        device
            .dispatch_buffers(read_first, n as u32, &[BufferBinding::read_write(&out)])
            .unwrap();
        rounds.push(device.read_typed_buffer(&out).unwrap());
    }
    rounds
}

#[test]
fn workgroup_memory_reads_as_zero_before_any_write_on_the_route() {
    let device = device();
    // 256 groups, 3 rounds: the previous dispatch's pattern would still sit in shared memory without the zeroing (the
    // spike saw 0xDEADBEEF on the RTX 5090 and garbage on the Radeon 890M).
    let rounds = dispatch_zero_probe(
        &device,
        &probes::wg_dirty::DESCRIPTOR,
        &probes::wg_read_first::DESCRIPTOR,
        256,
    );
    for (round, got) in rounds.iter().enumerate() {
        let bad = got.iter().filter(|word| **word != 0).count();
        assert_eq!(
            bad,
            0,
            "round {round}: {bad} of {} words are not zero",
            got.len()
        );
    }
    assert_spirv(&device, "probes::wg_read_first");
    assert_spirv(&device, "probes::wg_dirty");
}

#[test]
fn workgroup_zeroing_covers_arrays_that_are_not_a_multiple_of_the_group() {
    let device = device();
    let rounds = dispatch_zero_probe(
        &device,
        &probes::wg_odd_dirty::DESCRIPTOR,
        &probes::wg_odd_read_first::DESCRIPTOR,
        64,
    );
    for (round, got) in rounds.iter().enumerate() {
        let bad = got.iter().filter(|word| **word != 0).count();
        assert_eq!(
            bad,
            0,
            "round {round}: {bad} of {} words are not zero",
            got.len()
        );
    }
    assert_spirv(&device, "probes::wg_odd_read_first");
}
