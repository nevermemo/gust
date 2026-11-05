//! The `pixel` shape on a real device (roadmap MX, slice mx.p1b; decisions D61, D62, D120 and D127).
//!
//! A `pixel` kernel is a fragment shader whose outputs are storage buffers; the host draws one fragment per element
//! over a full-screen triangle, scissored to the live texels (Option S). What this file shows, each from an
//! independent reference and never from the kernel run on the same path:
//!
//! - Identity: for u32, i32, f32, Vec2 and Vec4, a `pixel` kernel and a `grid` kernel with the same body write the
//!   same bits, at X-S1's 17 squares and at the sizes around the row width (8192), the tail row and the cap. Both are
//!   also compared with a host reference that replays the body, so "equal to each other" cannot hide "both wrong".
//! - Tail rows: n = 8192k + 1 and 8192k + 2, where the last row is covered by one or two texels and its quad has helper
//!   invocations. They run but cannot store, so the output must be exact. (A stray store cannot be seen here, because
//!   robustBufferAccess2 drops an out-of-bounds store; the fragments that really run are counted with an atomic by
//!   `a_pixel_pass_launches_exactly_n_fragments` in `src/render.rs`.)
//! - The cap: n = 2^26 runs where the adapter can hold a 256 MiB binding and names the limit where it cannot.
//! - The fixture kernels `shade`, `pack` and `row` against the CPU twin, a pixel kernel through the f32 compatibility
//!   path, a batch mixing both shapes, the timed family, the staged graph, and two threads on one device.
//!
//! Sizes above 1 000 003 are skipped on an adapter whose type is `Cpu` (lavapipe, which runs the whole binary in CI);
//! the skip is printed as a `BRANCH` line (run with `--nocapture` to see it). The RTX 5090 runs every size.

#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::disallowed_methods)]

include!("../../../tests/fixtures/pixel.rs");

use std::thread;

use gust::{
    Extent, GpuPod, Independent, PIXEL_MAX_ELEMENTS, PIXEL_ROW_WIDTH, Shape,
    cpu::CpuDevice,
    math::{Vec2, Vec4},
    pod_slice_as_bytes,
    texture::{Address, Filter, Rgba32Float, SamplerOptions, TextureUse},
};
use gust_wgpu::{
    BufferDispatch, DeviceOptions, Error, F32BufferBinding, GpuBuffer, GpuBufferAccess,
    HeadlessDevice, StagedGraph,
};

/// The identity kernels: for each element type a `grid` and a `pixel` kernel over one shared helper, so the only
/// difference between the two is the shape. `seed_*` fill an input on the GPU (a host-made input at 16M elements
/// would dominate the test), `iota` writes a function of the index alone. Most grid kernels take the default workgroup
/// of 64; `grid_iota` sets 128 beside the `pixel` kernels of the module (a mix the emitter refused with E0707 before
/// slice mx.p1b's macro fix).
#[gust::gpu]
mod ident {
    fn mix(i: u32) -> u32 {
        i * 2654435761u32 + 12345u32
    }

    fn seed_i32(i: u32) -> i32 {
        (mix(i) & 0xFFFFFu32) as i32 - 524288
    }

    fn seed_f32(i: u32) -> f32 {
        (mix(i) & 0xFFFFu32) as f32 * 0.25f32 - 4096.0f32
    }

    fn seed_v2(i: u32) -> Vec2 {
        let s = mix(i);
        Vec2::new(
            (s & 0xFFFFu32) as f32 * 0.25f32 - 4096.0f32,
            ((s >> 16u32) & 0xFFFFu32) as f32 * 0.5f32 - 2048.0f32,
        )
    }

    fn seed_v4(i: u32) -> Vec4 {
        let s = mix(i);
        Vec4::new(
            (s & 0xFFu32) as f32,
            ((s >> 8u32) & 0xFFu32) as f32 * 0.5f32,
            ((s >> 16u32) & 0xFFu32) as f32 - 128.0f32,
            (s >> 24u32) as f32 * 0.25f32,
        )
    }

    fn h_u32(x: u32, i: u32) -> u32 {
        x * 3u32 + i
    }

    fn h_i32(x: i32, i: u32) -> i32 {
        x - (i as i32)
    }

    fn h_f32(x: f32, i: u32) -> f32 {
        x + (i as f32)
    }

    fn h_v2(v: Vec2, i: u32) -> Vec2 {
        Vec2::new(v.y, v.x + (i as f32))
    }

    fn h_v4(v: Vec4, i: u32) -> Vec4 {
        Vec4::new(v.w, v.z + (i as f32), v.y, v.x)
    }

    #[kernel]
    pub fn seed_u32_input(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let i = id.y * 2_097_152 + id.x;
        if i < out.len() {
            out[i] = mix(i);
        }
    }

    #[kernel]
    pub fn seed_i32_input(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<i32>) {
        let i = id.y * 2_097_152 + id.x;
        if i < out.len() {
            out[i] = seed_i32(i);
        }
    }

    #[kernel]
    pub fn seed_f32_input(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<f32>) {
        let i = id.y * 2_097_152 + id.x;
        if i < out.len() {
            out[i] = seed_f32(i);
        }
    }

    #[kernel]
    pub fn seed_v2_input(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<Vec2>) {
        let i = id.y * 2_097_152 + id.x;
        if i < out.len() {
            out[i] = seed_v2(i);
        }
    }

    #[kernel]
    pub fn seed_v4_input(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<Vec4>) {
        let i = id.y * 2_097_152 + id.x;
        if i < out.len() {
            out[i] = seed_v4(i);
        }
    }

    #[kernel]
    pub fn grid_u32(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let i = id.y * 2_097_152 + id.x;
        if i < out.len() {
            out[i] = h_u32(input[i], i);
        }
    }

    #[kernel(shape = pixel)]
    pub fn pixel_u32(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<u32>,
        mut out: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        out[i] = h_u32(input[i], i);
    }

    #[kernel]
    pub fn grid_i32(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<i32>,
        mut out: RWStructuredBuffer<i32>,
    ) {
        let i = id.y * 2_097_152 + id.x;
        if i < out.len() {
            out[i] = h_i32(input[i], i);
        }
    }

    #[kernel(shape = pixel)]
    pub fn pixel_i32(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<i32>,
        mut out: RWStructuredBuffer<i32>,
    ) {
        let i = id.x;
        out[i] = h_i32(input[i], i);
    }

    #[kernel]
    pub fn grid_f32(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<f32>,
    ) {
        let i = id.y * 2_097_152 + id.x;
        if i < out.len() {
            out[i] = h_f32(input[i], i);
        }
    }

    #[kernel(shape = pixel)]
    pub fn pixel_f32(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<f32>,
    ) {
        let i = id.x;
        out[i] = h_f32(input[i], i);
    }

    #[kernel]
    pub fn grid_v2(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<Vec2>,
        mut out: RWStructuredBuffer<Vec2>,
    ) {
        let i = id.y * 2_097_152 + id.x;
        if i < out.len() {
            out[i] = h_v2(input[i], i);
        }
    }

    #[kernel(shape = pixel)]
    pub fn pixel_v2(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<Vec2>,
        mut out: RWStructuredBuffer<Vec2>,
    ) {
        let i = id.x;
        out[i] = h_v2(input[i], i);
    }

    #[kernel]
    pub fn grid_v4(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<Vec4>,
        mut out: RWStructuredBuffer<Vec4>,
    ) {
        let i = id.y * 2_097_152 + id.x;
        if i < out.len() {
            out[i] = h_v4(input[i], i);
        }
    }

    #[kernel(shape = pixel)]
    pub fn pixel_v4(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<Vec4>,
        mut out: RWStructuredBuffer<Vec4>,
    ) {
        let i = id.x;
        out[i] = h_v4(input[i], i);
    }

    /// No input: the output is a function of the index alone (the 2^26 case, and the helper-invocation case).
    #[kernel(workgroup_size(128, 1, 1))]
    pub fn grid_iota(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let i = id.y * 2_097_152 + id.x;
        if i < out.len() {
            out[i] = mix(i);
        }
    }

    #[kernel(shape = pixel)]
    pub fn pixel_iota(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<u32>) {
        let i = id.x;
        out[i] = mix(i);
    }
}

// ---- host references (the bodies above, replayed on the host) ----

fn host_mix(i: u32) -> u32 {
    i.wrapping_mul(2_654_435_761).wrapping_add(12_345)
}

fn host_seed_u32(i: u32) -> u32 {
    host_mix(i)
}

fn host_seed_i32(i: u32) -> i32 {
    (host_mix(i) & 0xF_FFFF) as i32 - 524_288
}

fn host_seed_f32(i: u32) -> f32 {
    (host_mix(i) & 0xFFFF) as f32 * 0.25 - 4096.0
}

fn host_seed_v2(i: u32) -> Vec2 {
    let s = host_mix(i);
    Vec2::new(
        (s & 0xFFFF) as f32 * 0.25 - 4096.0,
        ((s >> 16) & 0xFFFF) as f32 * 0.5 - 2048.0,
    )
}

fn host_seed_v4(i: u32) -> Vec4 {
    let s = host_mix(i);
    Vec4::new(
        (s & 0xFF) as f32,
        ((s >> 8) & 0xFF) as f32 * 0.5,
        ((s >> 16) & 0xFF) as f32 - 128.0,
        (s >> 24) as f32 * 0.25,
    )
}

fn host_h_u32(x: u32, i: u32) -> u32 {
    x.wrapping_mul(3).wrapping_add(i)
}

fn host_h_i32(x: i32, i: u32) -> i32 {
    x.wrapping_sub(i as i32)
}

fn host_h_f32(x: f32, i: u32) -> f32 {
    x + i as f32
}

fn host_h_v2(v: Vec2, i: u32) -> Vec2 {
    Vec2::new(v.y, v.x + i as f32)
}

fn host_h_v4(v: Vec4, i: u32) -> Vec4 {
    Vec4::new(v.w, v.z + i as f32, v.y, v.x)
}

// ---- helpers ----

fn device() -> HeadlessDevice {
    HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip")
}

/// A device with the adapter's own limits: the 2^26 case needs a 256 MiB storage binding (wgpu's default is 128 MiB)
/// and the larger squares of the wide element types need more than the default as well.
fn wide_device() -> HeadlessDevice {
    HeadlessDevice::with_options(DeviceOptions::default())
        .expect("Vulkan adapter required; this test must not silently skip")
}

/// The launch of a `grid` kernel over `n` elements: one row of at most 2^21 threads (at most 32 768 workgroups, under
/// the 65 535 limit of one dimension) and as many rows as needed. The kernels index `id.y * 2^21 + id.x`.
fn grid_extent(n: u32) -> Extent {
    const ROW: u32 = 2_097_152;
    Extent::new(n.min(ROW), n.div_ceil(ROW).max(1), 1)
}

/// The most elements a `Cpu` adapter (a software rasteriser) is asked to draw: its cost is the CPU's, not a bug's.
const CPU_ADAPTER_MAX_ELEMENTS: u32 = 1_000_003;

fn is_cpu_adapter(device: &HeadlessDevice) -> bool {
    device.adapter_info().device_type == wgpu::DeviceType::Cpu
}

/// The sizes of the identity test, ascending and without repeats: X-S1's 17 squares, 64^3 + 1, 0, 1, 63, 65, the row
/// width and its neighbours, `8192k + 1` for k in 1, 2, 7 and 1000, two rows and a bit, and 1 000 003.
fn sizes() -> Vec<u32> {
    let mut sizes: Vec<u32> = [
        1u32, 2, 3, 5, 17, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097,
    ]
    .iter()
    .map(|side| side * side)
    .collect();
    sizes.extend([64 * 64 * 64 + 1, 0, 1, 63, 65, 8191, 8192, 8193, 16_385]);
    sizes.extend([2 * 8192 + 1, 1_000_003]);
    sizes.extend([1u32, 2, 7, 1000].map(|k| PIXEL_ROW_WIDTH * k + 1));
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

/// Whether `n` elements of `stride` bytes are skipped on this adapter, with the reason printed (a `BRANCH` line).
fn skipped(device: &HeadlessDevice, test: &str, n: u32, stride: usize) -> bool {
    if is_cpu_adapter(device) && n > CPU_ADAPTER_MAX_ELEMENTS {
        eprintln!(
            "BRANCH {test}: n = {n} skipped, the adapter is of type Cpu (sizes above {CPU_ADAPTER_MAX_ELEMENTS} are run on hardware)"
        );
        return true;
    }
    let bytes = u64::from(n) * stride as u64;
    let limits = device.limits();
    if bytes > limits.max_storage_buffer_binding_size || bytes > limits.max_buffer_size {
        eprintln!(
            "BRANCH {test}: n = {n} skipped, {bytes} bytes exceed the adapter's storage binding or buffer limit"
        );
        return true;
    }
    false
}

/// The 32-bit words in which `left` and `right` differ (all of the longer one's, when the lengths differ).
fn differing_words<T: GpuPod>(left: &[T], right: &[T]) -> usize {
    let (left, right) = (pod_slice_as_bytes(left), pod_slice_as_bytes(right));
    let common = left
        .as_chunks::<4>()
        .0
        .iter()
        .zip(right.as_chunks::<4>().0.iter())
        .filter(|(left, right)| left != right)
        .count();
    common + left.len().abs_diff(right.len()).div_ceil(4)
}

fn typed<T: GpuPod>(device: &HeadlessDevice, label: &str, values: &[T]) -> GpuBuffer<T> {
    device.create_typed_buffer(label, values, GpuBufferAccess::ReadWrite)
}

/// A read-write buffer of `count` zeroed elements, allocated on the device without a host vector.
fn zeroed<T: GpuPod>(device: &HeadlessDevice, label: &str, count: usize) -> GpuBuffer<T> {
    device
        .create_zeroed_buffer::<T>(label, count, GpuBufferAccess::ReadWrite)
        .unwrap()
}

/// One element type of the identity test: n elements are seeded on the GPU, run through the `grid` and the `pixel`
/// kernel (each into an output with a different starting value, so a cell a kernel never wrote cannot match), and
/// both are compared bit for bit with each other and with the host reference.
macro_rules! identity_test {
    (
        $test:ident, $ty:ty, $seed:ident, $grid:ident, $pixel:ident,
        init: ($zero:expr, $other:expr), host: ($host_seed:ident, $host_h:ident)
    ) => {
        #[test]
        fn $test() {
            let device = wide_device();
            let (mut ran, mut skipped_sizes) = (0usize, 0usize);
            for n in sizes() {
                if skipped(&device, stringify!($test), n, std::mem::size_of::<$ty>()) {
                    skipped_sizes += 1;
                    continue;
                }
                let count = n as usize;
                let input = zeroed::<$ty>(&device, "input", count);
                let grid_out = zeroed::<$ty>(&device, "grid out", count);
                let pixel_out = typed::<$ty>(&device, "pixel out", &vec![$other; count]);
                ident::$seed::dispatch_extent(&device, grid_extent(n), Independent(&input))
                    .unwrap();
                ident::$grid::dispatch_extent(
                    &device,
                    grid_extent(n),
                    Independent(&input),
                    Independent(&grid_out),
                )
                .unwrap();
                ident::$pixel::dispatch(&device, n, &input, &pixel_out).unwrap();
                let grid = device.read_typed_buffer(&grid_out).unwrap();
                let pixel = device.read_typed_buffer(&pixel_out).unwrap();
                let want: Vec<$ty> = (0..n).map(|i| $host_h($host_seed(i), i)).collect();
                assert_eq!(
                    differing_words(&pixel, &grid),
                    0,
                    "{}: the pixel and the grid kernel differ at n = {n}",
                    stringify!($test)
                );
                assert_eq!(
                    differing_words(&grid, &want),
                    0,
                    "{}: the kernels differ from the host reference at n = {n}",
                    stringify!($test)
                );
                ran += 1;
            }
            eprintln!(
                "BRANCH {}: {ran} sizes run, {skipped_sizes} skipped, 0 differing words",
                stringify!($test)
            );
            assert!(ran >= 20, "the identity test must run the small sizes");
        }
    };
}

identity_test!(
    u32_pixel_equals_grid_bit_for_bit,
    u32,
    seed_u32_input,
    grid_u32,
    pixel_u32,
    init: (0u32, 0xA5A5_A5A5u32),
    host: (host_seed_u32, host_h_u32)
);

identity_test!(
    i32_pixel_equals_grid_bit_for_bit,
    i32,
    seed_i32_input,
    grid_i32,
    pixel_i32,
    init: (0i32, -1515870811i32),
    host: (host_seed_i32, host_h_i32)
);

identity_test!(
    f32_pixel_equals_grid_bit_for_bit,
    f32,
    seed_f32_input,
    grid_f32,
    pixel_f32,
    init: (0.0f32, -1.0e9f32),
    host: (host_seed_f32, host_h_f32)
);

identity_test!(
    vec2_pixel_equals_grid_bit_for_bit,
    Vec2,
    seed_v2_input,
    grid_v2,
    pixel_v2,
    init: (Vec2::new(0.0, 0.0), Vec2::new(-1.0e9, -1.0e9)),
    host: (host_seed_v2, host_h_v2)
);

identity_test!(
    vec4_pixel_equals_grid_bit_for_bit,
    Vec4,
    seed_v4_input,
    grid_v4,
    pixel_v4,
    init: (Vec4::new(0.0, 0.0, 0.0, 0.0), Vec4::new(-1.0e9, -1.0e9, -1.0e9, -1.0e9)),
    host: (host_seed_v4, host_h_v4)
);

// ---- helper invocations ----

/// n = 8192k + 1 and 8192k + 2: the tail row holds one or two live texels, and the rest of its quad is helper
/// invocations. Helpers run but cannot store, so every element must hold exactly `mix(i)` and none may keep its
/// sentinel. This does not detect a stray fragment: robustBufferAccess2 drops an out-of-bounds store, so a fragment
/// for an index past `n` would leave no trace here. The count test in `src/render.rs` detects those with an atomic.
#[test]
fn a_tail_row_of_one_or_two_texels_leaves_the_output_exact() {
    let device = wide_device();
    let mut checked = 0;
    for k in [1u32, 2, 7, 1000] {
        for extra in [1u32, 2] {
            let n = PIXEL_ROW_WIDTH * k + extra;
            if skipped(&device, "helper_invocations", n, 4) {
                continue;
            }
            let out = typed::<u32>(&device, "iota", &vec![0xDEAD_BEEFu32; n as usize]);
            ident::pixel_iota::dispatch(&device, n, &out).unwrap();
            let got = device.read_typed_buffer(&out).unwrap();
            let want: Vec<u32> = (0..n).map(host_mix).collect();
            assert_eq!(differing_words(&got, &want), 0, "n = {n}");
            assert_eq!(
                got[n as usize - 1],
                host_mix(n - 1),
                "n = {n}: the last element"
            );
            checked += 1;
        }
    }
    assert!(checked >= 4, "the small tails must run on every adapter");
    eprintln!("BRANCH helper_invocations: {checked} sizes run, 0 differing words");
}

// ---- the cap ----

/// n = 2^26 is the cap: 8192 rows of 8192 texels, a 256 MiB `u32` output. It runs where the adapter's storage binding
/// holds that, against both the `grid` kernel and the host reference; where it does not, the named `DeviceLimit` is
/// the answer and the branch is printed.
#[test]
fn the_cap_runs_where_the_adapter_can_bind_it_and_names_the_limit_where_it_cannot() {
    let device = wide_device();
    let n = PIXEL_MAX_ELEMENTS;
    let bytes = u64::from(n) * 4;
    let out = match device.create_zeroed_buffer::<u32>(
        "cap out",
        n as usize,
        GpuBufferAccess::ReadWrite,
    ) {
        Ok(out) => out,
        Err(Error::DeviceLimit { limit, .. }) => {
            eprintln!(
                "BRANCH cap: the adapter cannot allocate {bytes} bytes ({limit}); the named limit is the answer"
            );
            assert_eq!(limit, "max_buffer_size");
            return;
        }
        Err(other) => panic!("unexpected allocation error: {other:?}"),
    };
    let binding_limit = device.limits().max_storage_buffer_binding_size;
    if binding_limit < bytes {
        eprintln!(
            "BRANCH cap: the storage binding limit {binding_limit} is below {bytes}; asserting the named limit"
        );
        // Refused before anything is encoded, so it costs nothing on any adapter.
        match ident::pixel_iota::dispatch(&device, n, &out) {
            Err(Error::DeviceLimit {
                limit,
                requested,
                maximum,
                ..
            }) => {
                assert_eq!(limit, "max_storage_buffer_binding_size");
                assert_eq!((requested, maximum), (bytes, binding_limit));
            }
            other => panic!("expected the binding-size DeviceLimit, got {other:?}"),
        }
        return;
    }
    if is_cpu_adapter(&device) {
        // A software rasteriser is not asked for 67 M fragments: nothing is dispatched.
        eprintln!("BRANCH cap: the adapter is of type Cpu; n = {n} is not dispatched");
        return;
    }
    eprintln!("BRANCH cap: ran n = {n} on the adapter ({bytes} bytes)");
    ident::pixel_iota::dispatch(&device, n, &out).unwrap();
    let reference = device
        .create_zeroed_buffer::<u32>("cap grid", n as usize, GpuBufferAccess::ReadWrite)
        .unwrap();
    ident::grid_iota::dispatch_extent(&device, grid_extent(n), Independent(&reference)).unwrap();
    let got = device.read_typed_buffer(&out).unwrap();
    let grid = device.read_typed_buffer(&reference).unwrap();
    assert_eq!(
        differing_words(&got, &grid),
        0,
        "the pixel and the grid kernel differ at 2^26"
    );
    let want: Vec<u32> = (0..n).map(host_mix).collect();
    assert_eq!(
        differing_words(&got, &want),
        0,
        "the pixel kernel differs from the host at 2^26"
    );
}

// ---- the fixture kernels against the CPU twin ----

/// Sizes around the row width, small enough for the twin to run the body once per element.
const TWIN_SIZES: [u32; 6] = [1, 7, 4096, 8191, 8193, 20_011];

#[test]
fn shade_on_the_gpu_matches_the_twin() {
    let device = device();
    let cpu = CpuDevice::new();
    for n in TWIN_SIZES {
        let values: Vec<f32> = (0..n).map(|i| (i % 101) as f32 / 100.0 - 0.1).collect();
        let input = typed(&device, "input", &values);
        let scaled = typed(&device, "scaled", &vec![f32::NAN; n as usize]);
        let flags = typed(&device, "flags", &vec![7u32; n as usize]);
        pixel::shade::dispatch(&device, n, &input, &scaled, &flags).unwrap();
        let (cpu_input, cpu_scaled, cpu_flags) = (
            cpu.buffer(&values),
            cpu.buffer(&vec![f32::NAN; n as usize]),
            cpu.buffer(&vec![7u32; n as usize]),
        );
        pixel::shade::dispatch(&cpu, n, &cpu_input, &cpu_scaled, &cpu_flags).unwrap();
        assert_eq!(
            differing_words(
                &device.read_typed_buffer(&scaled).unwrap(),
                &cpu_scaled.read()
            ),
            0,
            "shade scaled, n = {n}"
        );
        assert_eq!(
            device.read_typed_buffer(&flags).unwrap(),
            cpu_flags.read(),
            "shade flags, n = {n}"
        );
    }
}

#[test]
fn pack_on_the_gpu_matches_the_twin() {
    let device = device();
    let cpu = CpuDevice::new();
    for n in TWIN_SIZES {
        let a: Vec<Vec2> = (0..n)
            .map(|i| Vec2::new(i as f32 * 0.5, -(i as f32)))
            .collect();
        let b: Vec<i32> = (0..n as i32).map(|i| 3 - i).collect();
        let (ga, gb) = (typed(&device, "a", &a), typed(&device, "b", &b));
        let sums = typed(
            &device,
            "sums",
            &vec![Vec2::new(f32::NAN, f32::NAN); n as usize],
        );
        let wide = typed(
            &device,
            "wide",
            &vec![Vec4::new(f32::NAN, f32::NAN, f32::NAN, f32::NAN); n as usize],
        );
        let steps = typed(&device, "steps", &vec![0i32; n as usize]);
        pixel::pack::dispatch(&device, n, &ga, &gb, &sums, &wide, &steps).unwrap();
        let (ca, cb) = (cpu.buffer(&a), cpu.buffer(&b));
        let csums = cpu.buffer(&vec![Vec2::new(f32::NAN, f32::NAN); n as usize]);
        let cwide = cpu.buffer(&vec![
            Vec4::new(f32::NAN, f32::NAN, f32::NAN, f32::NAN);
            n as usize
        ]);
        let csteps = cpu.buffer(&vec![0i32; n as usize]);
        pixel::pack::dispatch(&cpu, n, &ca, &cb, &csums, &cwide, &csteps).unwrap();
        assert_eq!(
            differing_words(&device.read_typed_buffer(&sums).unwrap(), &csums.read()),
            0,
            "sums, n = {n}"
        );
        assert_eq!(
            differing_words(&device.read_typed_buffer(&wide).unwrap(), &cwide.read()),
            0,
            "wide, n = {n}"
        );
        assert_eq!(
            device.read_typed_buffer(&steps).unwrap(),
            csteps.read(),
            "steps, n = {n}"
        );
    }
}

/// `row` samples a 16 by 1 texture at the centre of texel `x % 16`'s column with a nearest sampler (a linear one needs
/// `float32-filterable`), so the sample is the texel itself on the GPU and on the twin.
#[test]
fn row_on_the_gpu_matches_the_twin() {
    let device = device();
    let cpu = CpuDevice::new();
    let texels: Vec<Vec4> = (0..16)
        .map(|j| Vec4::new(j as f32, 2.0 * j as f32, 3.0, 1.0))
        .collect();
    let image = device
        .create_texture_2d::<Rgba32Float>(16, 1, TextureUse::SAMPLED)
        .unwrap();
    device.write_texture(&image, &texels).unwrap();
    let nearest = device
        .create_sampler(SamplerOptions::new(Filter::Nearest, Address::ClampToEdge))
        .unwrap();
    let cpu_image = cpu
        .texture_2d::<Rgba32Float>(16, 1, TextureUse::SAMPLED)
        .unwrap();
    cpu_image.write(&texels).unwrap();
    let cpu_nearest = cpu.sampler(SamplerOptions::new(Filter::Nearest, Address::ClampToEdge));
    for n in [1u32, 15, 16, 4097, 8193] {
        // `x` past the strip clamps (the u of the kernel is `(x + 0.5) / 16`), the same rule on both.
        let colours = typed(
            &device,
            "colours",
            &vec![Vec4::new(f32::NAN, f32::NAN, f32::NAN, f32::NAN); n as usize],
        );
        pixel::row::dispatch(&device, n, &image, &nearest, &colours).unwrap();
        let cpu_colours = cpu.buffer(&vec![
            Vec4::new(f32::NAN, f32::NAN, f32::NAN, f32::NAN);
            n as usize
        ]);
        pixel::row::dispatch(&cpu, n, &cpu_image, &cpu_nearest, &cpu_colours).unwrap();
        assert_eq!(
            differing_words(
                &device.read_typed_buffer(&colours).unwrap(),
                &cpu_colours.read()
            ),
            0,
            "row, n = {n}"
        );
    }
}

// ---- other entry points ----

fn descriptor(name: &str) -> &'static gust::KernelDescriptor {
    ident::MODULE_DESCRIPTOR
        .kernels
        .iter()
        .find(|kernel| kernel.name == name)
        .unwrap_or_else(|| panic!("no kernel `{name}`"))
}

/// A `pixel` kernel through the f32 compatibility path (`dispatch_f32_buffers`) writes what the `grid` kernel does.
#[test]
fn a_pixel_kernel_through_the_f32_buffer_path_matches_the_grid_kernel() {
    let device = device();
    for n in [1u32, 8191, 8193, 100_003] {
        let values: Vec<f32> = (0..n).map(host_seed_f32).collect();
        let input = typed(&device, "input", &values);
        let grid_out = typed(&device, "grid out", &vec![0.0f32; n as usize]);
        let pixel_out = typed(&device, "pixel out", &vec![-1.0e9f32; n as usize]);
        for (name, out) in [("grid_f32", &grid_out), ("pixel_f32", &pixel_out)] {
            device
                .dispatch_f32_buffers(
                    descriptor(name),
                    n,
                    &[
                        F32BufferBinding::ReadOnly(&input),
                        F32BufferBinding::ReadWrite(out),
                    ],
                )
                .unwrap();
        }
        let want: Vec<f32> = (0..n).map(|i| host_h_f32(values[i as usize], i)).collect();
        let got = device.read_typed_buffer(&pixel_out).unwrap();
        assert_eq!(
            differing_words(&got, &device.read_typed_buffer(&grid_out).unwrap()),
            0,
            "n = {n}"
        );
        assert_eq!(differing_words(&got, &want), 0, "n = {n}");
    }
}

fn stats(device: &HeadlessDevice) -> (usize, u64, u64) {
    let stats = device.pipeline_cache_stats();
    (stats.entries, stats.misses, stats.hits)
}

/// Three `pixel` calls of distinct n and a `grid` call between them in one batch: one command buffer, a pass per call.
/// The second batch finds every pipeline in the cache.
#[test]
fn a_batch_mixes_pixel_calls_of_different_sizes_with_a_grid_call() {
    let device = device();
    let sizes = [1u32, 8193, 100_003];
    let values: Vec<Vec<u32>> = sizes
        .iter()
        .map(|&n| (0..n).map(host_seed_u32).collect())
        .collect();
    let inputs: Vec<_> = values
        .iter()
        .map(|values| typed(&device, "input", values))
        .collect();
    let outputs: Vec<_> = sizes
        .iter()
        .map(|&n| typed(&device, "out", &vec![0u32; n as usize]))
        .collect();
    let grid_values: Vec<u32> = (0..257).map(host_seed_u32).collect();
    let grid_input = typed(&device, "grid input", &grid_values);
    let grid_out = typed(&device, "grid out", &vec![0u32; 257]);

    let first = ident::pixel_u32::call(sizes[0], &inputs[0], &outputs[0]);
    let second = ident::pixel_u32::call(sizes[1], &inputs[1], &outputs[1]);
    let between = ident::grid_u32::call(257, &grid_input, &grid_out);
    let third = ident::pixel_u32::call(sizes[2], &inputs[2], &outputs[2]);
    let batch: [BufferDispatch<'_>; 4] = [
        (&first).into(),
        (&second).into(),
        (&between).into(),
        (&third).into(),
    ];

    device.dispatch_batch(&batch).unwrap();
    // Two kernels (one pixel, one grid) compiled once each.
    assert_eq!(stats(&device), (2, 2, 2));
    for ((output, values), &n) in outputs.iter().zip(&values).zip(&sizes) {
        let want: Vec<u32> = values
            .iter()
            .zip(0..n)
            .map(|(&x, i)| host_h_u32(x, i))
            .collect();
        assert_eq!(
            differing_words(&device.read_typed_buffer(output).unwrap(), &want),
            0,
            "n = {n}"
        );
    }
    let want: Vec<u32> = grid_values
        .iter()
        .zip(0..257)
        .map(|(&x, i)| host_h_u32(x, i))
        .collect();
    assert_eq!(
        differing_words(&device.read_typed_buffer(&grid_out).unwrap(), &want),
        0
    );

    // The second batch hits the cache for every call: no new entry, no new miss.
    let before = stats(&device);
    device.dispatch_batch(&batch).unwrap();
    let after = stats(&device);
    assert_eq!((after.0, after.1), (before.0, before.1));
    assert!(after.2 > before.2);
}

/// A `pixel` call whose output is the next `grid` call's input, in one `dispatch_batch` (one encoder, two passes of
/// different kinds): the second must see all of the first's stores. n is not a multiple of 64 or of the row width.
#[test]
fn a_pixel_call_feeds_a_grid_call_in_one_batch() {
    let device = device();
    let n = 100_003u32;
    let values: Vec<u32> = (0..n).map(host_seed_u32).collect();
    let input = typed(&device, "input", &values);
    let middle = zeroed::<u32>(&device, "middle", n as usize);
    let last = zeroed::<u32>(&device, "last", n as usize);
    let first = ident::pixel_u32::call(n, &input, &middle);
    let second = ident::grid_u32::call(n, &middle, &last);
    device
        .dispatch_batch(&[(&first).into(), (&second).into()])
        .unwrap();
    let want: Vec<u32> = values
        .iter()
        .zip(0..n)
        .map(|(&x, i)| host_h_u32(host_h_u32(x, i), i))
        .collect();
    assert_eq!(
        differing_words(&device.read_typed_buffer(&last).unwrap(), &want),
        0
    );
}

/// n = 0 is `Ok` and does no work, no compile and no cache lookup, as for a `grid` call.
#[test]
fn an_empty_pixel_call_is_ok_and_touches_no_cache() {
    let device = device();
    let input = typed::<u32>(&device, "input", &[]);
    let out = typed::<u32>(&device, "out", &[]);
    ident::pixel_u32::dispatch(&device, 0, &input, &out).unwrap();
    assert_eq!(stats(&device), (0, 0, 0));
}

#[test]
fn the_timed_family_runs_a_pixel_call() {
    let device = device();
    let n = 20_011u32;
    let values: Vec<u32> = (0..n).map(host_seed_u32).collect();
    let input = typed(&device, "input", &values);
    let out = typed(&device, "out", &vec![0u32; n as usize]);
    let other = typed(&device, "other", &vec![0u32; n as usize]);
    let kernel = descriptor("pixel_u32");
    assert_eq!(kernel.shape, Shape::Pixel);
    let bindings = [
        gust_wgpu::BufferBinding::read_only(&input),
        gust_wgpu::BufferBinding::read_write(&out),
    ];
    let timing = device.dispatch_buffers_timed(kernel, n, &bindings).unwrap();
    assert_eq!(
        timing.gpu_elapsed.is_some(),
        device.supports_gpu_timestamps()
    );
    let want: Vec<u32> = values
        .iter()
        .zip(0..n)
        .map(|(&x, i)| host_h_u32(x, i))
        .collect();
    assert_eq!(
        differing_words(&device.read_typed_buffer(&out).unwrap(), &want),
        0
    );

    // A timed batch whose first and last passes are of different shapes: the stamps go to a compute and a render pass.
    let grid = ident::grid_u32::call(n, &input, &other);
    let pixel = ident::pixel_u32::call(n, &input, &out);
    let batch: [BufferDispatch<'_>; 2] = [(&grid).into(), (&pixel).into()];
    let timing = device.dispatch_batch_timed(&batch).unwrap();
    assert_eq!(
        timing.gpu_elapsed.is_some(),
        device.supports_gpu_timestamps()
    );
    assert_eq!(
        differing_words(&device.read_typed_buffer(&other).unwrap(), &want),
        0
    );
    let batch: [BufferDispatch<'_>; 2] = [(&pixel).into(), (&grid).into()];
    let timing = device.dispatch_batch_timed(&batch).unwrap();
    assert_eq!(
        timing.gpu_elapsed.is_some(),
        device.supports_gpu_timestamps()
    );
}

// ---- the staged graph ----

/// grid -> pixel -> grid: the pixel node reads what the first grid node wrote and the second grid node reads what the
/// pixel node wrote; the typed methods derive the edges, the explicit ones state them, and the same graph with the
/// middle edge missing is a `GraphMissingDependency`.
#[test]
fn a_graph_orders_a_pixel_node_between_two_grid_nodes() {
    let device = device();
    let n = 20_011u32;
    let a = typed(&device, "a", &vec![0u32; n as usize]);
    let b = typed(&device, "b", &vec![0u32; n as usize]);
    let c = typed(&device, "c", &vec![0u32; n as usize]);
    let seed = ident::seed_u32_input::call(n, &a);
    let middle = ident::pixel_u32::call(n, &a, &b);
    let last = ident::grid_u32::call(n, &b, &c);
    let want: Vec<u32> = (0..n)
        .map(|i| host_h_u32(host_h_u32(host_seed_u32(i), i), i))
        .collect();

    let mut derived = StagedGraph::new();
    let seed_node = derived.run(&seed);
    let middle_node = derived.run(&middle);
    let last_node = derived.run(&last);
    let read = derived.read(&c);
    assert_eq!(derived.dependency_indices(middle_node), [seed_node.index()]);
    assert_eq!(derived.dependency_indices(last_node), [middle_node.index()]);
    let output = device.execute_graph(&derived).unwrap();
    assert_eq!(
        differing_words(&output.readback::<u32>(read).unwrap(), &want),
        0
    );
    assert_eq!(output.report.dispatches, 3);

    // Explicit edges, all stated: the same result.
    let mut explicit = StagedGraph::new();
    let seed_node = explicit.dispatch(BufferDispatch::from(&seed), &[]);
    let middle_node = explicit.dispatch(BufferDispatch::from(&middle), &[seed_node]);
    let last_node = explicit.dispatch(BufferDispatch::from(&last), &[middle_node]);
    let read = explicit.readback(&c, &[last_node]);
    let output = device.execute_graph(&explicit).unwrap();
    assert_eq!(
        differing_words(&output.readback::<u32>(read).unwrap(), &want),
        0
    );

    // The pixel node without its edge to the node that wrote its input.
    let mut missing = StagedGraph::new();
    let seed_node = missing.dispatch(BufferDispatch::from(&seed), &[]);
    let middle_node = missing.dispatch(BufferDispatch::from(&middle), &[]);
    let last_node = missing.dispatch(BufferDispatch::from(&last), &[middle_node]);
    let _ = (seed_node, last_node);
    match device.execute_graph(&missing) {
        Err(Error::GraphMissingDependency { node, producer }) => {
            assert_eq!((node, producer), (1, 0))
        }
        Ok(_) => panic!("a missing edge must be refused"),
        Err(other) => panic!("expected GraphMissingDependency, got {other:?}"),
    }
}

// ---- threads ----

/// A smoke test of the queue and of the concurrent first compile (Option S has no shared uniform, so there is no hazard
/// left to test): two threads share one device and submit batches of three `pixel` calls of distinct n for 100 rounds
/// each; every
/// output is compared with the host reference every round, and the cache holds the three kernels once each (a miss is
/// counted when an entry is inserted, so two threads compiling the same kernel at once still make one miss).
#[test]
fn two_threads_submit_pixel_batches_on_one_device() {
    let device = device();
    let sets: [[u32; 3]; 2] = [[1, 8193, 20_011], [2, 16_385, 30_011]];
    thread::scope(|scope| {
        for sizes in sets {
            let device = &device;
            scope.spawn(move || {
                let [n_u32, n_i32, n_f32] = sizes;
                let u: Vec<u32> = (0..n_u32).map(host_seed_u32).collect();
                let i: Vec<i32> = (0..n_i32).map(host_seed_i32).collect();
                let f: Vec<f32> = (0..n_f32).map(host_seed_f32).collect();
                let (u_in, i_in, f_in) = (
                    typed(device, "u in", &u),
                    typed(device, "i in", &i),
                    typed(device, "f in", &f),
                );
                let u_out = typed(device, "u out", &vec![0u32; u.len()]);
                let i_out = typed(device, "i out", &vec![0i32; i.len()]);
                let f_out = typed(device, "f out", &vec![0.0f32; f.len()]);
                let want_u: Vec<u32> = u.iter().zip(0..).map(|(&x, k)| host_h_u32(x, k)).collect();
                let want_i: Vec<i32> = i.iter().zip(0..).map(|(&x, k)| host_h_i32(x, k)).collect();
                let want_f: Vec<f32> = f.iter().zip(0..).map(|(&x, k)| host_h_f32(x, k)).collect();
                for round in 0..100 {
                    let calls = (
                        ident::pixel_u32::call(n_u32, &u_in, &u_out),
                        ident::pixel_i32::call(n_i32, &i_in, &i_out),
                        ident::pixel_f32::call(n_f32, &f_in, &f_out),
                    );
                    let batch: [BufferDispatch<'_>; 3] =
                        [(&calls.0).into(), (&calls.1).into(), (&calls.2).into()];
                    device.dispatch_batch(&batch).unwrap();
                    assert_eq!(
                        differing_words(&device.read_typed_buffer(&u_out).unwrap(), &want_u),
                        0,
                        "u32, round {round}"
                    );
                    assert_eq!(
                        differing_words(&device.read_typed_buffer(&i_out).unwrap(), &want_i),
                        0,
                        "i32, round {round}"
                    );
                    assert_eq!(
                        differing_words(&device.read_typed_buffer(&f_out).unwrap(), &want_f),
                        0,
                        "f32, round {round}"
                    );
                }
            });
        }
    });
    let stats = device.pipeline_cache_stats();
    assert_eq!((stats.entries, stats.misses), (3, 3));
}

/// The kernels of this file are the shape they say they are.
#[test]
fn the_identity_kernels_have_the_shapes_their_names_say() {
    for kernel in ident::MODULE_DESCRIPTOR.kernels {
        let want = if kernel.name.starts_with("pixel_") {
            Shape::Pixel
        } else {
            Shape::Grid
        };
        assert_eq!(kernel.shape, want, "{}", kernel.name);
    }
    for kernel in pixel::MODULE_DESCRIPTOR.kernels {
        assert_eq!(kernel.shape, Shape::Pixel, "{}", kernel.name);
    }
}
