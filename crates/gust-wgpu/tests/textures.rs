//! Textures in `gust-wgpu` (roadmap MX, slice mx.t2, decision D64): `GpuTexture2d<F>`, `GpuSampler`, the host
//! transfers, the two bind-group layout variants, and the submit-time checks of texture and sampler parameters.
//!
//! The kernels are hand-written descriptors with hand-written Slang (the `#[gpu]` macro does not emit texture kernels
//! until slice mx.t3), built for one property each. Expectations come from host loops and constants, never from a
//! shader shadow body:
//!
//! - `read_texture` returns exactly what `write_texture` stored, bit for bit, on all nine formats.
//! - A sampled `Texture2D` reads through a `Nearest` sampler on the default device (the non-filterable layout variant)
//!   and on a device that enabled `float32-filterable` (the filterable variant); a `Linear` sampler needs the feature.
//! - A read-write `r32` storage texture and the write-only wide formats (bound `WriteOnly`, D120) round-trip through
//!   a kernel.
//! - Each mistake of the dispatch (use, format, filter, sampler count, limits) is its own error before any pipeline is
//!   compiled.
//!
//! Slice mx.t4 adds the evidence for kernels written in Rust, in the second half of this file (from the line that
//! starts with `==== slice mx.t4`): the twin against the GPU on all nine formats, sampling against the twin's formula and
//! an independent host reference, and the refusals. One typed call is built by hand in the first half (`Call::new_with_samplers`).

#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]

use std::{
    future::Future,
    pin::pin,
    task::{Context, Poll, Waker},
    thread,
    time::{Duration, Instant},
};

use gust::{
    Access, Call, GpuPod, Kernel, KernelDescriptor, LengthRule, ParameterDescriptor, ParameterKind,
    Resource, ResourceBinding, ResourceLayout, Run, SamplerArg, SourceOrigin, TextureAccess,
    TextureDimension, TextureLayout, TextureUse, TypeLayout,
    math::{IVec2, IVec4, UVec2, UVec4, Vec2, Vec4},
    pod_slice_as_bytes,
    texture::{
        Address, Filter, Format, R32Float, R32Sint, R32Uint, Rg32Float, Rg32Sint, Rg32Uint,
        Rgba32Float, Rgba32Sint, Rgba32Uint, SampleClass, SamplerOptions, TextureFormat,
    },
};
use gust_wgpu::{
    AdapterChoice, BufferBinding, BufferDispatch, DeviceOptions, Error, FeaturesPolicy, GpuBuffer,
    GpuBufferAccess, GpuSampler, GpuSamplerBinding, GpuTexture2d, HeadlessDevice, LimitsPolicy,
    StagedGraph, render_wgpu_source,
};

fn device() -> HeadlessDevice {
    HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip")
}

/// Options as `device()` opens (Vulkan, the portable limits) with the given optional features.
fn options(features: FeaturesPolicy) -> DeviceOptions {
    DeviceOptions {
        backends: wgpu::Backends::VULKAN,
        adapter: AdapterChoice::HighPerformance,
        limits: LimitsPolicy::Default,
        features,
    }
}

/// A device opened with `FeaturesPolicy::Adapter`: it has `float32-filterable` when the adapter offers it.
fn feature_device() -> HeadlessDevice {
    HeadlessDevice::with_options(options(FeaturesPolicy::Adapter))
        .expect("Vulkan adapter required; this test must not silently skip")
}

/// Poll `future` every millisecond until it is ready (see `async_readback.rs`).
fn drive<F: Future>(future: F) -> F::Output {
    let mut context = Context::from_waker(Waker::noop());
    let mut future = pin!(future);
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
        assert!(
            Instant::now() < deadline,
            "the texture readback was still pending after 60 s"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

fn assert_cold(device: &HeadlessDevice) {
    let stats = device.pipeline_cache_stats();
    assert_eq!(
        (stats.entries, stats.misses, stats.hits),
        (0, 0, 0),
        "the error must precede compilation and the cache lookup"
    );
}

// ---- host transfers: create, write, read ----

/// The words of texel `index`: the first texels are the awkward float bit patterns (negative zero, infinities, the
/// smallest denormal, a quiet NaN with a payload, the largest finite value), the rest an arithmetic mix. Integer
/// formats take the same words as `u32`/`i32`.
fn word(index: u32, lane: u32) -> u32 {
    const SPECIAL: [u32; 8] = [
        0x8000_0000, // -0.0
        0x7f80_0000, // +inf
        0xff80_0000, // -inf
        0x0000_0001, // the smallest denormal
        0x7fc1_2345, // a quiet NaN with a payload
        0x7f7f_ffff, // the largest finite f32
        0x8000_0001, // the smallest negative denormal
        0xffff_ffff, // a NaN with every payload bit set
    ];
    let slot = (index * 4 + lane) as usize;
    if slot < SPECIAL.len() {
        SPECIAL[slot]
    } else {
        (index.wrapping_mul(2_654_435_761)) ^ lane.wrapping_mul(0x9e37_79b9) ^ (index >> 3)
    }
}

/// A texel type built from the words of [`word`].
trait FromWords: GpuPod {
    fn from_words(index: u32) -> Self;
}

impl FromWords for u32 {
    fn from_words(index: u32) -> Self {
        word(index, 0)
    }
}
impl FromWords for i32 {
    fn from_words(index: u32) -> Self {
        word(index, 0) as i32
    }
}
impl FromWords for f32 {
    fn from_words(index: u32) -> Self {
        f32::from_bits(word(index, 0))
    }
}
impl FromWords for UVec2 {
    fn from_words(index: u32) -> Self {
        UVec2::new(word(index, 0), word(index, 1))
    }
}
impl FromWords for IVec2 {
    fn from_words(index: u32) -> Self {
        IVec2::new(word(index, 0) as i32, word(index, 1) as i32)
    }
}
impl FromWords for Vec2 {
    fn from_words(index: u32) -> Self {
        Vec2::new(
            f32::from_bits(word(index, 0)),
            f32::from_bits(word(index, 1)),
        )
    }
}
impl FromWords for UVec4 {
    fn from_words(index: u32) -> Self {
        UVec4::new(
            word(index, 0),
            word(index, 1),
            word(index, 2),
            word(index, 3),
        )
    }
}
impl FromWords for IVec4 {
    fn from_words(index: u32) -> Self {
        IVec4::new(
            word(index, 0) as i32,
            word(index, 1) as i32,
            word(index, 2) as i32,
            word(index, 3) as i32,
        )
    }
}
impl FromWords for Vec4 {
    fn from_words(index: u32) -> Self {
        Vec4::new(
            f32::from_bits(word(index, 0)),
            f32::from_bits(word(index, 1)),
            f32::from_bits(word(index, 2)),
            f32::from_bits(word(index, 3)),
        )
    }
}

/// 67 texels wide: 67 * 4 bytes is not a multiple of 256, so the readback has to strip row padding for every format
/// (the widest rows, 67 * 16 bytes, are not 256-aligned either).
const WIDTH: u32 = 67;
const HEIGHT: u32 = 3;

fn texels<F: TextureFormat>() -> Vec<F::Texel>
where
    F::Texel: FromWords,
{
    (0..WIDTH * HEIGHT)
        .map(<F::Texel as FromWords>::from_words)
        .collect()
}

fn assert_round_trip<F: TextureFormat>(device: &HeadlessDevice)
where
    F::Texel: FromWords,
{
    let texture: GpuTexture2d<F> = device
        .create_texture_2d(WIDTH, HEIGHT, TextureUse::SAMPLED)
        .unwrap();
    assert_eq!((texture.width(), texture.height()), (WIDTH, HEIGHT));
    assert_eq!(texture.format(), F::FORMAT);
    assert_eq!(texture.uses(), TextureUse::SAMPLED);
    let written = texels::<F>();
    device.write_texture(&texture, &written).unwrap();
    let blocking = device.read_texture(&texture).unwrap();
    assert_eq!(
        pod_slice_as_bytes(&blocking),
        pod_slice_as_bytes(&written),
        "{} must read back bit for bit",
        F::FORMAT
    );
    let asynchronous = drive(device.read_texture_async(&texture)).unwrap();
    assert_eq!(
        pod_slice_as_bytes(&asynchronous),
        pod_slice_as_bytes(&written),
        "{} must read back bit for bit through the async path",
        F::FORMAT
    );
}

#[test]
fn every_format_round_trips_bit_for_bit() {
    let device = device();
    assert_round_trip::<R32Uint>(&device);
    assert_round_trip::<R32Sint>(&device);
    assert_round_trip::<R32Float>(&device);
    assert_round_trip::<Rg32Uint>(&device);
    assert_round_trip::<Rg32Sint>(&device);
    assert_round_trip::<Rg32Float>(&device);
    assert_round_trip::<Rgba32Uint>(&device);
    assert_round_trip::<Rgba32Sint>(&device);
    assert_round_trip::<Rgba32Float>(&device);
    // The nine markers are the nine formats.
    assert_eq!(Format::ALL.len(), 9);
}

#[test]
fn a_new_texture_reads_back_as_zeros() {
    let device = device();
    let texture: GpuTexture2d<Rgba32Float> =
        device.create_texture_2d(5, 2, TextureUse::SAMPLED).unwrap();
    let read = device.read_texture(&texture).unwrap();
    assert_eq!(pod_slice_as_bytes(&read), vec![0u8; 10 * 16]);
}

#[test]
fn texture_creation_and_transfer_errors() {
    let device = device();
    // A zero side.
    for (width, height) in [(0, 4), (4, 0), (0, 0)] {
        match device.create_texture_2d::<R32Float>(width, height, TextureUse::SAMPLED) {
            Err(Error::TextureSize {
                width: w,
                height: h,
                ..
            }) => assert_eq!((w, h), (width, height)),
            other => panic!(
                "expected TextureSize for {width}x{height}, got {:?}",
                other.err()
            ),
        }
    }
    // A side above the device limit.
    let maximum = device.limits().max_texture_dimension_2d;
    match device.create_texture_2d::<R32Float>(maximum + 1, 1, TextureUse::SAMPLED) {
        Err(Error::DeviceLimit {
            limit,
            requested,
            maximum: reported,
            ..
        }) => {
            assert_eq!(limit, "max_texture_dimension_2d");
            assert_eq!(requested, u64::from(maximum) + 1);
            assert_eq!(reported, u64::from(maximum));
        }
        other => panic!("expected DeviceLimit, got {:?}", other.err()),
    }
    // A host slice of the wrong length.
    let texture: GpuTexture2d<R32Uint> =
        device.create_texture_2d(4, 4, TextureUse::SAMPLED).unwrap();
    for length in [0usize, 15, 17] {
        let result = device.write_texture(&texture, &vec![1u32; length]);
        assert!(
            matches!(
                result,
                Err(Error::TextureSize {
                    width: 4,
                    height: 4,
                    ..
                })
            ),
            "a slice of {length} texels for a 4x4 texture: {result:?}"
        );
    }
    // A texture of another device.
    let other = HeadlessDevice::new().unwrap();
    assert!(matches!(
        other.write_texture(&texture, &[0u32; 16]),
        Err(Error::ForeignPersistentBuffer)
    ));
    assert!(matches!(
        other.read_texture(&texture),
        Err(Error::ForeignPersistentBuffer)
    ));
    assert!(matches!(
        drive(other.read_texture_async(&texture)),
        Err(Error::ForeignPersistentBuffer)
    ));
    // The texture itself is intact after the refused calls.
    device.write_texture(&texture, &[7u32; 16]).unwrap();
    assert_eq!(device.read_texture(&texture).unwrap(), vec![7u32; 16]);
}

#[test]
fn samplers_carry_their_options_and_device() {
    let device = device();
    for filter in [Filter::Nearest, Filter::Linear] {
        for address in [Address::ClampToEdge, Address::Repeat] {
            let options = SamplerOptions::new(filter, address);
            let sampler = device.create_sampler(options).unwrap();
            assert_eq!(sampler.options(), options);
            assert_eq!(GpuSamplerBinding::new(&sampler).options(), options);
        }
    }
}

// ---- features ----

#[test]
fn the_default_device_has_no_optional_feature() {
    let device = device();
    assert!(!device.float32_filterable());
    // `FeaturesPolicy::Default` turns it off even through the native `DeviceOptions::default()` (Adapter limits too).
    let policy_default = HeadlessDevice::with_options(DeviceOptions {
        features: FeaturesPolicy::Default,
        ..DeviceOptions::default()
    })
    .unwrap();
    assert!(!policy_default.float32_filterable());
}

/// D90: `DeviceOptions::default()` opens the same features as `default_device()` (the policy is `Adapter` on native).
#[test]
fn the_default_options_enable_the_feature_when_the_adapter_offers_it() {
    let device = HeadlessDevice::with_options(DeviceOptions::default()).unwrap();
    assert_eq!(
        device.float32_filterable(),
        adapter_offers_float32_filterable(&device)
    );
}

#[test]
fn the_adapter_policy_enables_the_feature_exactly_when_the_adapter_offers_it() {
    let device = feature_device();
    let offered = adapter_offers_float32_filterable(&device);
    assert_eq!(
        device.float32_filterable(),
        offered,
        "FeaturesPolicy::Adapter must enable float32-filterable on an adapter that has it, and nothing else"
    );
    eprintln!(
        "adapter {:?}: float32-filterable offered = {offered}",
        device.adapter_info().name
    );
}

/// Whether the adapter `device` was opened on offers `float32-filterable`, asked of a fresh wgpu instance.
fn adapter_offers_float32_filterable(device: &HeadlessDevice) -> bool {
    let name = device.adapter_info().name.clone();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .find(|adapter| adapter.get_info().name == name)
        .expect("the adapter the device was opened on")
        .features()
        .contains(wgpu::Features::FLOAT32_FILTERABLE)
}

// ---- descriptors ----

const F32: TypeLayout = <f32 as GpuPod>::LAYOUT;
const U32: TypeLayout = <u32 as GpuPod>::LAYOUT;
const VEC2: TypeLayout = <Vec2 as GpuPod>::LAYOUT;
const VEC4: TypeLayout = <Vec4 as GpuPod>::LAYOUT;
const UVEC2: TypeLayout = <UVec2 as GpuPod>::LAYOUT;

const fn bound(binding: u32) -> Option<ResourceBinding> {
    Some(ResourceBinding { group: 0, binding })
}

/// A sampled `Texture2D<element>` (parameter access `ReadOnly`, no exact format).
const fn sampled(
    name: &'static str,
    element: TypeLayout,
    class: SampleClass,
    binding: u32,
) -> ParameterDescriptor {
    ParameterDescriptor {
        name,
        rust_type: "Texture2D",
        kind: ParameterKind::Texture,
        access: Access::ReadOnly,
        resource_layout: Some(ResourceLayout::texture_v1(
            element,
            TextureLayout {
                format: None,
                dimension: TextureDimension::D2,
                access: TextureAccess::Sampled,
                class,
            },
        )),
        binding: bound(binding),
    }
}

/// A storage texture of an exact format (parameter access `ReadWrite`; `access` is the texture's own).
const fn storage_texture(
    name: &'static str,
    element: TypeLayout,
    format: Format,
    access: TextureAccess,
    binding: u32,
) -> ParameterDescriptor {
    ParameterDescriptor {
        name,
        rust_type: "StorageTexture2D",
        kind: ParameterKind::Texture,
        access: Access::ReadWrite,
        resource_layout: Some(ResourceLayout::texture_v1(
            element,
            TextureLayout {
                format: Some(format),
                dimension: TextureDimension::D2,
                access,
                class: format.sample_class(),
            },
        )),
        binding: bound(binding),
    }
}

const fn sampler(name: &'static str, binding: u32) -> ParameterDescriptor {
    ParameterDescriptor {
        name,
        rust_type: "Sampler",
        kind: ParameterKind::Sampler,
        access: Access::NotApplicable,
        resource_layout: None,
        binding: bound(binding),
    }
}

const fn buffer(name: &'static str, element: TypeLayout, binding: u32) -> ParameterDescriptor {
    ParameterDescriptor {
        name,
        rust_type: "RWStructuredBuffer",
        kind: ParameterKind::Storage,
        access: Access::ReadWrite,
        resource_layout: Some(ResourceLayout::storage_v1(element)),
        binding: bound(binding),
    }
}

const fn descriptor(
    name: &'static str,
    entry_point: &'static str,
    parameters: &'static [ParameterDescriptor],
    slang_source: &'static str,
) -> KernelDescriptor {
    KernelDescriptor {
        module: "texture_tests",
        name,
        entry_point,
        shape: gust::Shape::Grid,
        compatible_shapes: &[gust::Shape::Grid],
        workgroup_size: [64, 1, 1],
        parameters,
        slang_source,
        source: SourceOrigin::UNKNOWN,
        artifacts: &[],
    }
}

/// `sample(tex, samp, out)`: `out[i] = tex.SampleLevel(samp, ((i + 0.5) / n, 0.5), 0)`, one output per eighth of the
/// unit width, over a 4 x 1 texture. Texel centres are at 1/8, 3/8, 5/8, 7/8, so the outputs sit a quarter texel
/// either side of a centre: `Nearest` yields texel `i / 2`, `Linear` the interpolation of `i / 2 - 0.25` and
/// `i / 2 + 0.25` texel steps.
struct Sample;
impl Kernel for Sample {
    const DESCRIPTOR: KernelDescriptor = descriptor(
        "sample",
        "tex_sample",
        &[
            sampled("tex", F32, SampleClass::Float, 0),
            sampler("samp", 1),
            buffer("out", F32, 2),
        ],
        r#"
[[vk::binding(0, 0)]] Texture2D<float> tex;
[[vk::binding(1, 0)]] SamplerState samp;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> out;
[shader("compute")] [numthreads(64, 1, 1)]
void tex_sample(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; out.GetDimensions(n, stride);
    if (id.x < n)
    {
        float u = (float(id.x) + 0.5) / float(n);
        out[id.x] = tex.SampleLevel(samp, float2(u, 0.5), 0.0);
    }
}
"#,
    );
}

/// `load_u(tex, out)`: `out[i] = tex.Load((i % 5, i / 5, 0))` over a 5 x 3 `Texture2D<uint>`.
struct LoadUint;
impl Kernel for LoadUint {
    const DESCRIPTOR: KernelDescriptor = descriptor(
        "load_u",
        "tex_load_u",
        &[
            sampled("tex", U32, SampleClass::Uint, 0),
            buffer("out", U32, 1),
        ],
        r#"
[[vk::binding(0, 0)]] Texture2D<uint> tex;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> out;
[shader("compute")] [numthreads(64, 1, 1)]
void tex_load_u(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; out.GetDimensions(n, stride);
    if (id.x < n) { out[id.x] = tex.Load(int3(int(id.x % 5u), int(id.x / 5u), 0)); }
}
"#,
    );
}

/// `rw(img)`: `img[p] = img[p] * 2 + 1` over a 6 x 2 `r32f` read-write storage texture.
struct ReadWrite;
impl Kernel for ReadWrite {
    const DESCRIPTOR: KernelDescriptor = descriptor(
        "rw",
        "tex_rw",
        &[storage_texture(
            "img",
            F32,
            Format::R32Float,
            TextureAccess::ReadWrite,
            0,
        )],
        r#"
[[vk::binding(0, 0)]] [format("r32f")] RWTexture2D<float> img;
[shader("compute")] [numthreads(64, 1, 1)]
void tex_rw(uint3 id : SV_DispatchThreadID)
{
    if (id.x < 12u)
    {
        uint2 p = uint2(id.x % 6u, id.x / 6u);
        img[p] = img[p] * 2.0 + 1.0;
    }
}
"#,
    );
}

/// `fill(wide, pair)`: two write-only storage textures over an 8 x 4 grid, `rgba32f` and `rg32ui`. Neither format
/// allows read-write access, so both bind `WriteOnly` (D120).
struct Fill;
impl Kernel for Fill {
    const DESCRIPTOR: KernelDescriptor = descriptor(
        "fill",
        "tex_fill",
        &[
            storage_texture(
                "wide",
                VEC4,
                Format::Rgba32Float,
                TextureAccess::WriteOnly,
                0,
            ),
            storage_texture("pair", UVEC2, Format::Rg32Uint, TextureAccess::WriteOnly, 1),
        ],
        r#"
[[vk::binding(0, 0)]] [format("rgba32f")] WTexture2D<float4> wide;
[[vk::binding(1, 0)]] [format("rg32ui")] WTexture2D<uint2> pair;
[shader("compute")] [numthreads(64, 1, 1)]
void tex_fill(uint3 id : SV_DispatchThreadID)
{
    if (id.x < 32u)
    {
        uint2 p = uint2(id.x % 8u, id.x / 8u);
        wide.Store(p, float4(float(p.x), float(p.y), float(p.x) * 0.5, 1.0));
        pair.Store(p, uint2(p.x * 100u + p.y, 0xdeadbeefu));
    }
}
"#,
    );
}

/// `two_wide(a, b)`: two write-only `rgba32f` storage textures, so one texture can be bound to both.
struct TwoWide;
impl Kernel for TwoWide {
    const DESCRIPTOR: KernelDescriptor = descriptor(
        "two_wide",
        "tex_two_wide",
        &[
            storage_texture("a", VEC4, Format::Rgba32Float, TextureAccess::WriteOnly, 0),
            storage_texture("b", VEC4, Format::Rgba32Float, TextureAccess::WriteOnly, 1),
        ],
        r#"
[[vk::binding(0, 0)]] [format("rgba32f")] WTexture2D<float4> a;
[[vk::binding(1, 0)]] [format("rgba32f")] WTexture2D<float4> b;
[shader("compute")] [numthreads(64, 1, 1)]
void tex_two_wide(uint3 id : SV_DispatchThreadID)
{
    if (id.x < 32u)
    {
        uint2 p = uint2(id.x % 8u, id.x / 8u);
        a.Store(p, float4(1.0, 0.0, 0.0, 1.0));
        b.Store(p, float4(0.0, 1.0, 0.0, 1.0));
    }
}
"#,
    );
}

/// `load_v(tex, out)`: `out[i] = tex.Load((i % 3, i / 3, 0))` over a 3 x 2 `Texture2D<float4>`.
struct LoadVec4;
impl Kernel for LoadVec4 {
    const DESCRIPTOR: KernelDescriptor = descriptor(
        "load_v",
        "tex_load_v",
        &[
            sampled("tex", VEC4, SampleClass::Float, 0),
            buffer("out", F32, 1),
        ],
        r#"
[[vk::binding(0, 0)]] Texture2D<float4> tex;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> out;
[shader("compute")] [numthreads(64, 1, 1)]
void tex_load_v(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; out.GetDimensions(n, stride);
    if (id.x < n / 4u)
    {
        float4 v = tex.Load(int3(int(id.x % 3u), int(id.x / 3u), 0));
        out[4u * id.x] = v.x;
        out[4u * id.x + 1u] = v.y;
        out[4u * id.x + 2u] = v.z;
        out[4u * id.x + 3u] = v.w;
    }
}
"#,
    );
}

/// `sum_load(a, b, out)`: `out[i] = a.Load((i, 0, 0)) + b.Load((i, 0, 0))` over 4 x 1 `Texture2D<float>`s. The same
/// texture may be bound to both: two reads do not alias.
struct SumLoad;
impl Kernel for SumLoad {
    const DESCRIPTOR: KernelDescriptor = descriptor(
        "sum_load",
        "tex_sum_load",
        &[
            sampled("a", F32, SampleClass::Float, 0),
            sampled("b", F32, SampleClass::Float, 1),
            buffer("out", F32, 2),
        ],
        r#"
[[vk::binding(0, 0)]] Texture2D<float> a;
[[vk::binding(1, 0)]] Texture2D<float> b;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> out;
[shader("compute")] [numthreads(64, 1, 1)]
void tex_sum_load(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; out.GetDimensions(n, stride);
    if (id.x < n) { out[id.x] = a.Load(int3(int(id.x), 0, 0)) + b.Load(int3(int(id.x), 0, 0)); }
}
"#,
    );
}

/// `copy_in(tex, img)`: `img[(i, 0)] = tex.Load((i, 0, 0))`, a sampled and a read-write storage texture of one format.
struct SampledAndStorage;
impl Kernel for SampledAndStorage {
    const DESCRIPTOR: KernelDescriptor = descriptor(
        "copy_in",
        "tex_copy_in",
        &[
            sampled("tex", F32, SampleClass::Float, 0),
            storage_texture("img", F32, Format::R32Float, TextureAccess::ReadWrite, 1),
        ],
        r#"
[[vk::binding(0, 0)]] Texture2D<float> tex;
[[vk::binding(1, 0)]] [format("r32f")] RWTexture2D<float> img;
[shader("compute")] [numthreads(64, 1, 1)]
void tex_copy_in(uint3 id : SV_DispatchThreadID)
{
    if (id.x < 4u) { img[uint2(id.x, 0u)] = tex.Load(int3(int(id.x), 0, 0)); }
}
"#,
    );
}

// ---- sampled textures and the two layout variants ----

/// A 4 x 1 `R32Float` texture of 0, 10, 20, 40.
fn ramp(device: &HeadlessDevice) -> GpuTexture2d<R32Float> {
    let texture = device.create_texture_2d(4, 1, TextureUse::SAMPLED).unwrap();
    device
        .write_texture(&texture, &[0.0, 10.0, 20.0, 40.0])
        .unwrap();
    texture
}

fn run_sample(
    device: &HeadlessDevice,
    texture: &GpuTexture2d<R32Float>,
    sampler: &GpuSampler,
) -> Result<Vec<f32>, Error> {
    let out: GpuBuffer<f32> =
        device.create_typed_buffer("out", &[-1.0f32; 8], GpuBufferAccess::ReadWrite);
    let bindings = [
        texture.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&out),
    ];
    let samplers = [GpuSamplerBinding::new(sampler)];
    device.dispatch_batch(&[
        BufferDispatch::new(&Sample::DESCRIPTOR, 8, &bindings).with_samplers(&samplers)
    ])?;
    device.read_typed_buffer(&out)
}

#[test]
fn a_nearest_sampler_reads_the_texel_under_the_coordinate_on_both_variants() {
    let expected = [0.0f32, 0.0, 10.0, 10.0, 20.0, 20.0, 40.0, 40.0];
    for device in [device(), feature_device()] {
        let texture = ramp(&device);
        let sampler = device
            .create_sampler(SamplerOptions::new(Filter::Nearest, Address::ClampToEdge))
            .unwrap();
        let read = run_sample(&device, &texture, &sampler).unwrap();
        assert_eq!(
            read,
            expected,
            "float32-filterable enabled = {}",
            device.float32_filterable()
        );
    }
}

#[test]
fn a_linear_sampler_needs_float32_filterable() {
    // Without the feature: the named error, before any pipeline exists.
    let plain = device();
    let texture = ramp(&plain);
    let linear = plain
        .create_sampler(SamplerOptions::new(Filter::Linear, Address::ClampToEdge))
        .unwrap();
    match run_sample(&plain, &texture, &linear) {
        Err(Error::TextureFilter {
            parameter,
            filter,
            feature,
        }) => {
            assert_eq!(parameter, "samp");
            assert_eq!(filter, Filter::Linear);
            assert_eq!(feature, "float32-filterable");
        }
        other => panic!("expected TextureFilter, got {other:?}"),
    }
    assert_cold(&plain);
    // The device is still usable: a nearest sampler binds.
    let nearest = plain.create_sampler(SamplerOptions::default()).unwrap();
    assert_eq!(
        run_sample(&plain, &texture, &nearest).unwrap(),
        [0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 40.0, 40.0]
    );

    // With the feature (when the adapter has it): bilinear values; otherwise the same error again.
    let featured = feature_device();
    let texture = ramp(&featured);
    let linear = featured
        .create_sampler(SamplerOptions::new(Filter::Linear, Address::ClampToEdge))
        .unwrap();
    let result = run_sample(&featured, &texture, &linear);
    if featured.float32_filterable() {
        // Output i samples u = (i + 0.5) / 8, which is 4u = (i + 0.5) / 2 texels from the left edge, and
        // (i + 0.5) / 2 - 0.5 = i / 2 - 0.25 texels from the first texel centre. Clamped at both ends.
        let texels = [0.0f32, 10.0, 20.0, 40.0];
        let expected: Vec<f32> = (0..8)
            .map(|i| {
                let position = (i as f32 / 2.0 - 0.25).clamp(0.0, 3.0);
                let low = position.floor() as usize;
                let high = (low + 1).min(3);
                let fraction = position - low as f32;
                texels[low] + (texels[high] - texels[low]) * fraction
            })
            .collect();
        let read = result.unwrap();
        // Plan 3.5's bound: a hardware filter weighs texels with 8 fractional bits, so 2^-8 of the 40-wide range,
        // plus two ulps of the largest value for the interpolation itself.
        let tolerance = 40.0 / 256.0 + 2.0 * 40.0 * f32::EPSILON;
        for (index, (got, want)) in read.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() <= tolerance,
                "output {index}: {got} against {want} (all: {read:?} against {expected:?})"
            );
        }
    } else {
        eprintln!(
            "bilinear values not checked: adapter {:?} lacks float32-filterable; the refusal is checked instead",
            featured.adapter_info().name
        );
        assert!(
            matches!(result, Err(Error::TextureFilter { .. })),
            "{result:?}"
        );
    }
}

#[test]
fn the_same_descriptor_runs_under_each_device_layout_variant_with_one_pipeline_each() {
    // One descriptor serves both devices; each compiles its own pipeline for its own variant. That the variants
    // differ is not asserted here: wgpu validates the sampler and texture entries against each other, so the Linear
    // branch of `a_linear_sampler_needs_float32_filterable` (a filtering sampler over a float texture) is what proves
    // the filterable variant, and the Nearest runs above prove the non-filterable one.
    let plain = device();
    let featured = feature_device();
    for device in [&plain, &featured] {
        let texture = ramp(device);
        let sampler = device.create_sampler(SamplerOptions::default()).unwrap();
        run_sample(device, &texture, &sampler).unwrap();
        let stats = device.pipeline_cache_stats();
        assert_eq!(stats.entries, 1, "one pipeline per device");
    }
}

// ---- integer textures: Load, a different format of the same class, and the format check ----

#[test]
fn a_sampled_texture_loads_and_needs_its_exact_texel_type() {
    let device = device();
    let uint_texture: GpuTexture2d<R32Uint> =
        device.create_texture_2d(5, 3, TextureUse::SAMPLED).unwrap();
    let values: Vec<u32> = (0..15).map(|i| 1000 + i * 3).collect();
    device.write_texture(&uint_texture, &values).unwrap();
    let out: GpuBuffer<u32> =
        device.create_typed_buffer("out", &[0u32; 15], GpuBufferAccess::ReadWrite);
    let bindings = [
        uint_texture.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&out),
    ];
    device
        .dispatch_buffers(&LoadUint::DESCRIPTOR, 15, &bindings)
        .unwrap();
    assert_eq!(device.read_typed_buffer(&out).unwrap(), values);

    // A two-channel texture of the same class is refused too (S1): `Texture2D<uint>` reads one `uint`, as the typed
    // call (`Element = F::Texel`) and the twin do, not the first channel of a wider texel.
    let pair_texture: GpuTexture2d<Rg32Uint> =
        device.create_texture_2d(5, 3, TextureUse::SAMPLED).unwrap();
    let out: GpuBuffer<u32> =
        device.create_typed_buffer("out", &[0u32; 15], GpuBufferAccess::ReadWrite);
    let bindings = [
        pair_texture.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&out),
    ];
    match device.dispatch_buffers(&LoadUint::DESCRIPTOR, 15, &bindings) {
        Err(Error::TextureFormat {
            parameter, actual, ..
        }) => {
            assert_eq!(parameter, "tex");
            assert_eq!(actual, Format::Rg32Uint);
        }
        other => panic!("expected TextureFormat, got {other:?}"),
    }

    // A texture of another class is refused.
    let float_texture: GpuTexture2d<R32Float> =
        device.create_texture_2d(5, 3, TextureUse::SAMPLED).unwrap();
    let out: GpuBuffer<u32> =
        device.create_typed_buffer("out", &[0u32; 15], GpuBufferAccess::ReadWrite);
    let bindings = [
        float_texture.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&out),
    ];
    match device.dispatch_buffers(&LoadUint::DESCRIPTOR, 15, &bindings) {
        Err(Error::TextureFormat {
            parameter, actual, ..
        }) => {
            assert_eq!(parameter, "tex");
            assert_eq!(actual, Format::R32Float);
        }
        other => panic!("expected TextureFormat, got {other:?}"),
    }
}

#[test]
fn a_float4_texture_parameter_needs_a_four_channel_texture() {
    let device = device();
    let wide: GpuTexture2d<Rgba32Float> =
        device.create_texture_2d(3, 2, TextureUse::SAMPLED).unwrap();
    let texels: Vec<Vec4> = (0..6)
        .map(|i| Vec4::new(i as f32, -(i as f32), 0.5, 100.0 + i as f32))
        .collect();
    device.write_texture(&wide, &texels).unwrap();
    // The kernel writes the four channels of texel `i` to `out[4i..4i+4]`.
    let out: GpuBuffer<f32> =
        device.create_typed_buffer("out", &[0.0f32; 24], GpuBufferAccess::ReadWrite);
    let bindings = [
        wide.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&out).independent_length(),
    ];
    device
        .dispatch_buffers(&LoadVec4::DESCRIPTOR, 6, &bindings)
        .unwrap();
    let read = device.read_typed_buffer(&out).unwrap();
    let flat: Vec<f32> = texels.iter().flat_map(|t| [t.x, t.y, t.z, t.w]).collect();
    assert_eq!(read, flat);

    // An `r32float` texture has the class but not the width: refused, not read as `(r, 0, 0, 1)`.
    let narrow: GpuTexture2d<R32Float> =
        device.create_texture_2d(3, 2, TextureUse::SAMPLED).unwrap();
    let bindings = [
        narrow.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&out).independent_length(),
    ];
    match device.dispatch_buffers(&LoadVec4::DESCRIPTOR, 6, &bindings) {
        Err(Error::TextureFormat {
            parameter, actual, ..
        }) => {
            assert_eq!(parameter, "tex");
            assert_eq!(actual, Format::R32Float);
        }
        other => panic!("expected TextureFormat, got {other:?}"),
    }
    // So is a two-channel one.
    let pair: GpuTexture2d<Rg32Float> =
        device.create_texture_2d(3, 2, TextureUse::SAMPLED).unwrap();
    let bindings = [
        pair.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&out).independent_length(),
    ];
    let result = device.dispatch_buffers(&LoadVec4::DESCRIPTOR, 6, &bindings);
    assert!(
        matches!(
            result,
            Err(Error::TextureFormat {
                actual: Format::Rg32Float,
                ..
            })
        ),
        "{result:?}"
    );
}

#[test]
fn one_texture_bound_to_two_sampled_parameters_is_accepted() {
    let device = device();
    let texture = ramp(&device);
    let out: GpuBuffer<f32> =
        device.create_typed_buffer("out", &[0.0f32; 4], GpuBufferAccess::ReadWrite);
    let bindings = [
        texture.bind(Access::ReadOnly, LengthRule::Independent),
        texture.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&out),
    ];
    device
        .dispatch_buffers(&SumLoad::DESCRIPTOR, 4, &bindings)
        .unwrap();
    assert_eq!(
        device.read_typed_buffer(&out).unwrap(),
        [0.0, 20.0, 40.0, 80.0]
    );
}

#[test]
fn one_texture_bound_sampled_and_as_storage_is_aliased() {
    let device = device();
    let texture: GpuTexture2d<R32Float> = device
        .create_texture_2d(4, 1, TextureUse::SAMPLED.union(TextureUse::STORAGE))
        .unwrap();
    let bindings = [
        texture.bind(Access::ReadOnly, LengthRule::Independent),
        texture.bind(Access::ReadWrite, LengthRule::Independent),
    ];
    let result = device.dispatch_buffers(&SampledAndStorage::DESCRIPTOR, 4, &bindings);
    assert!(
        matches!(result, Err(Error::AliasedBinding { .. })),
        "{result:?}"
    );
    assert_cold(&device);
}

#[test]
fn a_texture_whose_readback_exceeds_max_buffer_size_is_a_device_limit() {
    let device = device();
    // 67 x 3 r32float: 3 rows of 512 padded bytes = 1536 bytes of staging.
    let texture: GpuTexture2d<R32Float> = device
        .create_texture_2d(WIDTH, HEIGHT, TextureUse::SAMPLED)
        .unwrap();
    let device = lowered(device, |limits| limits.max_buffer_size = 1024);
    match device.read_texture(&texture) {
        Err(Error::DeviceLimit {
            limit,
            requested,
            maximum,
            ..
        }) => assert_eq!((limit, requested, maximum), ("max_buffer_size", 1536, 1024)),
        other => panic!("expected DeviceLimit, got {other:?}"),
    }
    let asynchronous = drive(device.read_texture_async(&texture));
    assert!(
        matches!(
            asynchronous,
            Err(Error::DeviceLimit {
                limit: "max_buffer_size",
                ..
            })
        ),
        "{asynchronous:?}"
    );
}

#[test]
fn a_texture_without_the_sampled_use_is_refused_for_a_sampled_parameter() {
    let device = device();
    let texture: GpuTexture2d<R32Uint> =
        device.create_texture_2d(5, 3, TextureUse::STORAGE).unwrap();
    let out: GpuBuffer<u32> =
        device.create_typed_buffer("out", &[0u32; 15], GpuBufferAccess::ReadWrite);
    let bindings = [
        texture.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&out),
    ];
    match device.dispatch_buffers(&LoadUint::DESCRIPTOR, 15, &bindings) {
        Err(Error::TextureUse {
            parameter,
            required,
            actual,
        }) => {
            assert_eq!(parameter, "tex");
            assert_eq!(required, TextureUse::SAMPLED);
            assert_eq!(actual, TextureUse::STORAGE);
        }
        other => panic!("expected TextureUse, got {other:?}"),
    }
    assert_cold(&device);
}

// ---- storage textures ----

#[test]
fn a_read_write_r32_storage_texture_is_read_and_written_by_a_kernel() {
    let device = device();
    let texture: GpuTexture2d<R32Float> =
        device.create_texture_2d(6, 2, TextureUse::STORAGE).unwrap();
    let input: Vec<f32> = (0..12).map(|i| i as f32 * 1.5 - 4.0).collect();
    device.write_texture(&texture, &input).unwrap();
    let bindings = [texture.bind(Access::ReadWrite, LengthRule::Independent)];
    device
        .dispatch_buffers(&ReadWrite::DESCRIPTOR, 12, &bindings)
        .unwrap();
    let expected: Vec<f32> = input.iter().map(|value| value * 2.0 + 1.0).collect();
    assert_eq!(device.read_texture(&texture).unwrap(), expected);
}

#[test]
fn a_storage_parameter_needs_the_storage_use_and_its_exact_format() {
    let device = device();
    // Without STORAGE.
    let sampled_only: GpuTexture2d<R32Float> =
        device.create_texture_2d(6, 2, TextureUse::SAMPLED).unwrap();
    let bindings = [sampled_only.bind(Access::ReadWrite, LengthRule::Independent)];
    match device.dispatch_buffers(&ReadWrite::DESCRIPTOR, 12, &bindings) {
        Err(Error::TextureUse {
            parameter,
            required,
            actual,
        }) => {
            assert_eq!(parameter, "img");
            assert_eq!(required, TextureUse::STORAGE);
            assert_eq!(actual, TextureUse::SAMPLED);
        }
        other => panic!("expected TextureUse, got {other:?}"),
    }
    // The wrong format: a storage parameter wants exactly `r32f`.
    let wrong: GpuTexture2d<R32Uint> = device.create_texture_2d(6, 2, TextureUse::STORAGE).unwrap();
    let bindings = [wrong.bind(Access::ReadWrite, LengthRule::Independent)];
    match device.dispatch_buffers(&ReadWrite::DESCRIPTOR, 12, &bindings) {
        Err(Error::TextureFormat {
            parameter, actual, ..
        }) => {
            assert_eq!(parameter, "img");
            assert_eq!(actual, Format::R32Uint);
        }
        other => panic!("expected TextureFormat, got {other:?}"),
    }
    assert_cold(&device);
}

#[test]
fn write_only_wide_formats_are_bound_write_only() {
    let device = device();
    let wide: GpuTexture2d<Rgba32Float> =
        device.create_texture_2d(8, 4, TextureUse::STORAGE).unwrap();
    let pair: GpuTexture2d<Rg32Uint> = device.create_texture_2d(8, 4, TextureUse::STORAGE).unwrap();
    let bindings = [
        wide.bind(Access::ReadWrite, LengthRule::Independent),
        pair.bind(Access::ReadWrite, LengthRule::Independent),
    ];
    device
        .dispatch_buffers(&Fill::DESCRIPTOR, 32, &bindings)
        .unwrap();
    let expected_wide: Vec<Vec4> = (0..32)
        .map(|i| {
            let (x, y) = ((i % 8) as f32, (i / 8) as f32);
            Vec4::new(x, y, x * 0.5, 1.0)
        })
        .collect();
    let expected_pair: Vec<UVec2> = (0..32u32)
        .map(|i| UVec2::new((i % 8) * 100 + i / 8, 0xdead_beef))
        .collect();
    let read_wide = device.read_texture(&wide).unwrap();
    assert_eq!(
        pod_slice_as_bytes(&read_wide),
        pod_slice_as_bytes(&expected_wide)
    );
    let read_pair = device.read_texture(&pair).unwrap();
    assert_eq!(
        pod_slice_as_bytes(&read_pair),
        pod_slice_as_bytes(&expected_pair)
    );
}

// ---- descriptors and bindings: the validation that needs no kernel run ----

#[test]
fn incoherent_texture_descriptors_are_refused() {
    let device = device();
    // A wide format with read-write texture access (D120: only the r32 formats allow it).
    static WIDE_READ_WRITE: [ParameterDescriptor; 1] = [storage_texture(
        "img",
        VEC2,
        Format::Rg32Float,
        TextureAccess::ReadWrite,
        0,
    )];
    // A sampler with a resource contract.
    static SAMPLER_WITH_LAYOUT: [ParameterDescriptor; 1] = [ParameterDescriptor {
        resource_layout: Some(ResourceLayout::storage_v1(F32)),
        ..sampler("samp", 0)
    }];
    // A sampled texture with an exact format.
    static SAMPLED_WITH_FORMAT: [ParameterDescriptor; 1] = [storage_texture(
        "tex",
        F32,
        Format::R32Float,
        TextureAccess::Sampled,
        0,
    )];
    // A sampled texture whose class disagrees with its element.
    static CLASS_MISMATCH: [ParameterDescriptor; 1] = [sampled("tex", F32, SampleClass::Uint, 0)];
    for (name, parameters) in [
        ("wide_read_write", &WIDE_READ_WRITE[..]),
        ("sampler_with_layout", &SAMPLER_WITH_LAYOUT[..]),
        ("sampled_with_format", &SAMPLED_WITH_FORMAT[..]),
        ("class_mismatch", &CLASS_MISMATCH[..]),
    ] {
        let kernel = descriptor(name, "main", parameters, "// not compiled");
        let result = device.dispatch_batch(&[BufferDispatch::new(&kernel, 1, &[])]);
        assert!(
            matches!(result, Err(Error::InvalidResourceLayout { .. })),
            "{name}: {result:?}"
        );
        assert!(
            matches!(
                render_wgpu_source(&kernel),
                Err(Error::InvalidResourceLayout { .. })
            ),
            "{name}: the generated host refuses it too"
        );
    }
    assert_cold(&device);
}

#[test]
fn a_missing_or_extra_sampler_is_a_binding_count_error() {
    let device = device();
    let texture = ramp(&device);
    let out: GpuBuffer<f32> =
        device.create_typed_buffer("out", &[0.0f32; 8], GpuBufferAccess::ReadWrite);
    let bindings = [
        texture.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&out),
    ];
    // None.
    let result = device.dispatch_batch(&[BufferDispatch::new(&Sample::DESCRIPTOR, 8, &bindings)]);
    assert!(
        matches!(result, Err(Error::BindingCount { .. })),
        "{result:?}"
    );
    // Two.
    let sampler = device.create_sampler(SamplerOptions::default()).unwrap();
    let samplers = [
        GpuSamplerBinding::new(&sampler),
        GpuSamplerBinding::new(&sampler),
    ];
    let result = device.dispatch_batch(&[
        BufferDispatch::new(&Sample::DESCRIPTOR, 8, &bindings).with_samplers(&samplers)
    ]);
    assert!(
        matches!(result, Err(Error::BindingCount { .. })),
        "{result:?}"
    );
    // A sampler on a kernel without one.
    let one = [GpuSamplerBinding::new(&sampler)];
    let uint_texture: GpuTexture2d<R32Uint> =
        device.create_texture_2d(5, 3, TextureUse::SAMPLED).unwrap();
    let uint_out: GpuBuffer<u32> =
        device.create_typed_buffer("out", &[0u32; 15], GpuBufferAccess::ReadWrite);
    let uint_bindings = [
        uint_texture.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&uint_out),
    ];
    let result =
        device.dispatch_batch(&[
            BufferDispatch::new(&LoadUint::DESCRIPTOR, 15, &uint_bindings).with_samplers(&one),
        ]);
    assert!(
        matches!(result, Err(Error::BindingCount { .. })),
        "{result:?}"
    );
    assert_cold(&device);
}

#[test]
fn a_texture_or_sampler_of_another_device_is_refused() {
    let device = device();
    let other = HeadlessDevice::new().unwrap();
    let out: GpuBuffer<f32> =
        device.create_typed_buffer("out", &[0.0f32; 8], GpuBufferAccess::ReadWrite);
    let own_texture = ramp(&device);
    let own_sampler = device.create_sampler(SamplerOptions::default()).unwrap();
    let foreign_texture = ramp(&other);
    let foreign_sampler = other.create_sampler(SamplerOptions::default()).unwrap();

    let bindings = [
        foreign_texture.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&out),
    ];
    let samplers = [GpuSamplerBinding::new(&own_sampler)];
    let result = device.dispatch_batch(&[
        BufferDispatch::new(&Sample::DESCRIPTOR, 8, &bindings).with_samplers(&samplers)
    ]);
    assert!(
        matches!(result, Err(Error::ForeignPersistentBuffer)),
        "{result:?}"
    );

    let bindings = [
        own_texture.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&out),
    ];
    let samplers = [GpuSamplerBinding::new(&foreign_sampler)];
    let result = device.dispatch_batch(&[
        BufferDispatch::new(&Sample::DESCRIPTOR, 8, &bindings).with_samplers(&samplers)
    ]);
    assert!(
        matches!(result, Err(Error::ForeignPersistentBuffer)),
        "{result:?}"
    );
}

#[test]
fn one_texture_bound_to_two_parameters_with_a_write_is_aliased() {
    let device = device();
    // `TwoWide` writes two textures of one format; binding one texture to both is the aliasing buffers refuse.
    let texture: GpuTexture2d<Rgba32Float> =
        device.create_texture_2d(8, 4, TextureUse::STORAGE).unwrap();
    let bindings = [
        texture.bind(Access::ReadWrite, LengthRule::Independent),
        texture.bind(Access::ReadWrite, LengthRule::Independent),
    ];
    let result = device.dispatch_buffers(&TwoWide::DESCRIPTOR, 32, &bindings);
    assert!(
        matches!(result, Err(Error::AliasedBinding { .. })),
        "{result:?}"
    );
}

// ---- the typed call and the staged graph ----

#[test]
fn a_typed_call_with_a_sampler_runs_through_run() {
    let device = device();
    let texture = ramp(&device);
    let sampler = device.create_sampler(SamplerOptions::default()).unwrap();
    let out: GpuBuffer<f32> =
        device.create_typed_buffer("out", &[-1.0f32; 8], GpuBufferAccess::ReadWrite);
    let call = Call::<HeadlessDevice, Sample, 2, _>::new_with_samplers(
        8,
        [
            texture.bind(Access::ReadOnly, LengthRule::Independent),
            out.bind(Access::ReadWrite, LengthRule::Shared),
        ],
        [sampler.bind()],
    );
    device.dispatch_call(&call).unwrap();
    assert_eq!(
        device.read_typed_buffer(&out).unwrap(),
        [0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 40.0, 40.0]
    );
}

#[test]
fn the_staged_graph_refuses_texture_and_sampler_kernels() {
    let device = device();
    let texture: GpuTexture2d<R32Uint> =
        device.create_texture_2d(5, 3, TextureUse::SAMPLED).unwrap();
    let out: GpuBuffer<u32> =
        device.create_typed_buffer("out", &[0u32; 15], GpuBufferAccess::ReadWrite);
    let bindings = [
        texture.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&out),
    ];
    let mut graph = StagedGraph::new();
    graph.dispatch(
        BufferDispatch::new(&LoadUint::DESCRIPTOR, 15, &bindings),
        &[],
    );
    match device.execute_graph(&graph) {
        Err(Error::UnsupportedParameter { name, kind }) => {
            assert_eq!(name, "tex");
            assert_eq!(kind, ParameterKind::Texture);
        }
        other => panic!("expected UnsupportedParameter, got {:?}", other.err()),
    }
    assert_cold(&device);
}

// ---- limits ----

fn lowered(device: HeadlessDevice, change: impl FnOnce(&mut wgpu::Limits)) -> HeadlessDevice {
    let mut limits = device.limits().clone();
    change(&mut limits);
    device
        .with_limits(limits)
        .expect("lowering limits is allowed")
}

#[track_caller]
fn assert_limit(error: Option<Error>, name: &str, requested: u64, maximum: u64) {
    match error {
        Some(Error::DeviceLimit {
            limit,
            requested: actual_requested,
            maximum: actual_maximum,
            ..
        }) => assert_eq!(
            (limit, actual_requested, actual_maximum),
            (name, requested, maximum)
        ),
        other => panic!("expected DeviceLimit for `{name}`, got {other:?}"),
    }
}

#[test]
fn sampled_textures_above_the_limit_are_an_error_before_compilation() {
    let device = device();
    let texture = ramp(&device);
    let sampler = device.create_sampler(SamplerOptions::default()).unwrap();
    let out: GpuBuffer<f32> =
        device.create_typed_buffer("out", &[0.0f32; 8], GpuBufferAccess::ReadWrite);
    let device = lowered(device, |limits| {
        limits.max_sampled_textures_per_shader_stage = 0
    });
    let bindings = [
        texture.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&out),
    ];
    let samplers = [GpuSamplerBinding::new(&sampler)];
    let error = device
        .dispatch_batch(&[
            BufferDispatch::new(&Sample::DESCRIPTOR, 8, &bindings).with_samplers(&samplers)
        ])
        .err();
    assert_limit(error, "max_sampled_textures_per_shader_stage", 1, 0);
    assert_cold(&device);
}

#[test]
fn storage_textures_above_the_limit_are_an_error_before_compilation() {
    let device = device();
    let wide: GpuTexture2d<Rgba32Float> =
        device.create_texture_2d(8, 4, TextureUse::STORAGE).unwrap();
    let pair: GpuTexture2d<Rg32Uint> = device.create_texture_2d(8, 4, TextureUse::STORAGE).unwrap();
    let device = lowered(device, |limits| {
        limits.max_storage_textures_per_shader_stage = 1
    });
    let bindings = [
        wide.bind(Access::ReadWrite, LengthRule::Independent),
        pair.bind(Access::ReadWrite, LengthRule::Independent),
    ];
    let error = device
        .dispatch_buffers(&Fill::DESCRIPTOR, 32, &bindings)
        .err();
    assert_limit(error, "max_storage_textures_per_shader_stage", 2, 1);
    assert_cold(&device);
}

#[test]
fn samplers_above_the_limit_are_an_error_before_compilation() {
    let device = device();
    let texture = ramp(&device);
    let sampler = device.create_sampler(SamplerOptions::default()).unwrap();
    let out: GpuBuffer<f32> =
        device.create_typed_buffer("out", &[0.0f32; 8], GpuBufferAccess::ReadWrite);
    let device = lowered(device, |limits| limits.max_samplers_per_shader_stage = 0);
    let bindings = [
        texture.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&out),
    ];
    let samplers = [GpuSamplerBinding::new(&sampler)];
    let error = device
        .dispatch_batch(&[
            BufferDispatch::new(&Sample::DESCRIPTOR, 8, &bindings).with_samplers(&samplers)
        ])
        .err();
    assert_limit(error, "max_samplers_per_shader_stage", 1, 0);
    assert_cold(&device);
}

#[test]
fn a_texture_side_above_the_limit_is_an_error_at_creation_and_at_launch() {
    let device = device();
    // A texture made before the limit is lowered.
    let texture: GpuTexture2d<R32Uint> =
        device.create_texture_2d(5, 3, TextureUse::SAMPLED).unwrap();
    let out: GpuBuffer<u32> =
        device.create_typed_buffer("out", &[0u32; 15], GpuBufferAccess::ReadWrite);
    let device = lowered(device, |limits| limits.max_texture_dimension_2d = 4);
    // Creation honours the lowered limit.
    let error = device
        .create_texture_2d::<R32Uint>(5, 1, TextureUse::SAMPLED)
        .err();
    assert_limit(error, "max_texture_dimension_2d", 5, 4);
    // A launch with the older, wider texture does too.
    let bindings = [
        texture.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&out),
    ];
    let error = device
        .dispatch_buffers(&LoadUint::DESCRIPTOR, 15, &bindings)
        .err();
    assert_limit(error, "max_texture_dimension_2d", 5, 4);
    assert_cold(&device);
}

// ---- the generated host ----

#[test]
fn the_generated_host_renders_texture_and_sampler_bindings() {
    let sample = render_wgpu_source(&Sample::DESCRIPTOR).unwrap();
    assert!(sample.contains("wgpu::BindingType::Texture"), "{sample}");
    // Both layout variants are rendered, chosen by the `float32_filterable` argument of `new` (S2).
    assert!(sample.contains("float32_filterable: bool"), "{sample}");
    assert!(
        sample.contains("Float { filterable: float32_filterable }"),
        "{sample}"
    );
    assert!(
        sample.contains("wgpu::SamplerBindingType::Filtering"),
        "{sample}"
    );
    assert!(
        sample.contains("wgpu::SamplerBindingType::NonFiltering"),
        "{sample}"
    );
    assert!(sample.contains("&wgpu::TextureView"), "{sample}");
    assert!(sample.contains("&wgpu::Sampler"), "{sample}");
    assert!(
        sample.contains("wgpu::BindingResource::TextureView"),
        "{sample}"
    );
    assert!(
        sample.contains("wgpu::BindingResource::Sampler"),
        "{sample}"
    );

    let fill = render_wgpu_source(&Fill::DESCRIPTOR).unwrap();
    assert!(
        fill.contains("wgpu::StorageTextureAccess::WriteOnly"),
        "{fill}"
    );
    assert!(!fill.contains("StorageTextureAccess::ReadWrite"), "{fill}");
    assert!(fill.contains("wgpu::TextureFormat::Rgba32Float"), "{fill}");
    assert!(fill.contains("wgpu::TextureFormat::Rg32Uint"), "{fill}");

    let read_write = render_wgpu_source(&ReadWrite::DESCRIPTOR).unwrap();
    assert!(
        read_write.contains("wgpu::StorageTextureAccess::ReadWrite"),
        "{read_write}"
    );
    assert!(
        read_write.contains("wgpu::TextureFormat::R32Float"),
        "{read_write}"
    );
}

/// The generated `new` takes `float32_filterable` only when a layout entry depends on it (S2): a sampler or a sampled
/// float texture. A kernel with a sampled `uint` texture or storage textures keeps the two-argument form.
#[test]
fn the_generated_host_takes_the_feature_only_when_the_layout_depends_on_it() {
    // A sampled float texture without a sampler is filterable only on a feature device.
    let sum = render_wgpu_source(&SumLoad::DESCRIPTOR).unwrap();
    assert!(sum.contains("float32_filterable: bool"), "{sum}");
    assert!(
        sum.contains("Float { filterable: float32_filterable }"),
        "{sum}"
    );
    assert!(!sum.contains("SamplerBindingType"), "{sum}");
    for (name, source) in [
        ("load_u", render_wgpu_source(&LoadUint::DESCRIPTOR).unwrap()),
        ("fill", render_wgpu_source(&Fill::DESCRIPTOR).unwrap()),
        ("rw", render_wgpu_source(&ReadWrite::DESCRIPTOR).unwrap()),
    ] {
        assert!(!source.contains("float32_filterable"), "{name}: {source}");
        assert!(
            source.contains(
                "pub fn new(device: &wgpu::Device, descriptor: &gust::KernelDescriptor) -> Self"
            ),
            "{name}: {source}"
        );
    }
}

// ==== slice mx.t4: `#[gpu]` texture kernels on the GPU, against the CPU twin and an independent host reference ====
//
// Everything above this line drives hand-written descriptors. From here on the kernels are written in Rust: the module
// `textures` is the fixture of `tests/fixtures/textures.rs` (its Slang is the golden `textures.slang`), and the module
// `evidence` below holds the kernels that the fixture does not have (a copy kernel per format class, in-place swaps of
// the three storage formats, kernels that sample at coordinates read from a buffer, and two that address a texel
// from a buffer). Each kernel runs twice, on the GPU through the generated typed call and on the CPU twin through the
// same call, and the expectation is computed on the host from the inputs alone: never from a shadow body and never
// from the twin (a twin assertion is added to the GPU assertion, never in place of it, D22). `load` and `store` are
// compared bit for bit; `sample` is compared under the error domain of SEMANTICS (`linear_error_bound`).

include!("../../../tests/fixtures/textures.rs");

#[gust::gpu]
mod evidence {
    // One copy kernel per format class: every texel of three sampled textures is `load`ed and `store`d into three
    // write-only textures of the same formats. With the three classes this is `load` and `store` on all nine formats.
    #[kernel(workgroup_size(8, 8, 1))]
    pub fn copy_floats(
        id: SV_DispatchThreadID,
        one_in: Texture2D<f32>,
        two_in: Texture2D<Vec2>,
        four_in: Texture2D<Vec4>,
        mut one: WriteTexture2D<R32Float>,
        mut two: WriteTexture2D<Rg32Float>,
        mut four: WriteTexture2D<Rgba32Float>,
    ) {
        if id.x < one.width() && id.y < one.height() {
            let a = one_in.load(id.x, id.y);
            let b = two_in.load(id.x, id.y);
            let c = four_in.load(id.x, id.y);
            one.store(id.x, id.y, a);
            two.store(id.x, id.y, b);
            four.store(id.x, id.y, c);
        }
    }

    #[kernel(workgroup_size(8, 8, 1))]
    pub fn copy_uints(
        id: SV_DispatchThreadID,
        one_in: Texture2D<u32>,
        two_in: Texture2D<UVec2>,
        four_in: Texture2D<UVec4>,
        mut one: WriteTexture2D<R32Uint>,
        mut two: WriteTexture2D<Rg32Uint>,
        mut four: WriteTexture2D<Rgba32Uint>,
    ) {
        if id.x < one.width() && id.y < one.height() {
            let a = one_in.load(id.x, id.y);
            let b = two_in.load(id.x, id.y);
            let c = four_in.load(id.x, id.y);
            one.store(id.x, id.y, a);
            two.store(id.x, id.y, b);
            four.store(id.x, id.y, c);
        }
    }

    #[kernel(workgroup_size(8, 8, 1))]
    pub fn copy_ints(
        id: SV_DispatchThreadID,
        one_in: Texture2D<i32>,
        two_in: Texture2D<IVec2>,
        four_in: Texture2D<IVec4>,
        mut one: WriteTexture2D<R32Sint>,
        mut two: WriteTexture2D<Rg32Sint>,
        mut four: WriteTexture2D<Rgba32Sint>,
    ) {
        if id.x < one.width() && id.y < one.height() {
            let a = one_in.load(id.x, id.y);
            let b = two_in.load(id.x, id.y);
            let c = four_in.load(id.x, id.y);
            one.store(id.x, id.y, a);
            two.store(id.x, id.y, b);
            four.store(id.x, id.y, c);
        }
    }

    // The read-write storage path (the three r32 formats): each texel of two storage textures is loaded, and each is
    // stored with the other's value. Every invocation touches only its own texel, so there is no race.
    #[kernel(workgroup_size(8, 8, 1))]
    pub fn swap_uint(
        id: SV_DispatchThreadID,
        mut left: StorageTexture2D<R32Uint>,
        mut right: StorageTexture2D<R32Uint>,
    ) {
        if id.x < left.width() && id.y < left.height() {
            let a = left.load(id.x, id.y);
            let b = right.load(id.x, id.y);
            left.store(id.x, id.y, b);
            right.store(id.x, id.y, a);
        }
    }

    #[kernel(workgroup_size(8, 8, 1))]
    pub fn swap_sint(
        id: SV_DispatchThreadID,
        mut left: StorageTexture2D<R32Sint>,
        mut right: StorageTexture2D<R32Sint>,
    ) {
        if id.x < left.width() && id.y < left.height() {
            let a = left.load(id.x, id.y);
            let b = right.load(id.x, id.y);
            left.store(id.x, id.y, b);
            right.store(id.x, id.y, a);
        }
    }

    #[kernel(workgroup_size(8, 8, 1))]
    pub fn swap_float(
        id: SV_DispatchThreadID,
        mut left: StorageTexture2D<R32Float>,
        mut right: StorageTexture2D<R32Float>,
    ) {
        if id.x < left.width() && id.y < left.height() {
            let a = left.load(id.x, id.y);
            let b = right.load(id.x, id.y);
            left.store(id.x, id.y, b);
            right.store(id.x, id.y, a);
        }
    }

    // `sample` at the coordinates of two buffers, one result per coordinate, for the three float texel types. The
    // filter and the address mode belong to the sampler the call is given.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn sample_scalar(
        id: SV_DispatchThreadID,
        image: Texture2D<f32>,
        filter: Sampler,
        us: StructuredBuffer<f32>,
        vs: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<f32>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = image.sample(filter, us[i], vs[i]);
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn sample_pair(
        id: SV_DispatchThreadID,
        image: Texture2D<Vec2>,
        filter: Sampler,
        us: StructuredBuffer<f32>,
        vs: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<Vec2>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = image.sample(filter, us[i], vs[i]);
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn sample_quad(
        id: SV_DispatchThreadID,
        image: Texture2D<Vec4>,
        filter: Sampler,
        us: StructuredBuffer<f32>,
        vs: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<Vec4>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = image.sample(filter, us[i], vs[i]);
        }
    }

    // One texel, addressed from a buffer (`at` is `[x, y]`): the twin's out-of-range panics are tested through these.
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn poke(
        id: SV_DispatchThreadID,
        at: StructuredBuffer<u32>,
        mut image: StorageTexture2D<R32Uint>,
    ) {
        image.store(at[0u32], at[1u32], 7u32 + id.x);
    }

    #[kernel(workgroup_size(1, 1, 1))]
    pub fn peek(
        id: SV_DispatchThreadID,
        at: StructuredBuffer<u32>,
        image: StorageTexture2D<R32Uint>,
        mut seen: RWStructuredBuffer<u32>,
    ) {
        seen[id.x] = image.load(at[0u32], at[1u32]);
    }
}

use std::panic::{AssertUnwindSafe, catch_unwind};

use gust::{
    Extent, Independent,
    cpu::{
        CpuDevice,
        texture::{CpuSampler, CpuTexture2d, linear_error_bound, ulp},
    },
};

fn gpu_texture<F: TextureFormat>(
    device: &HeadlessDevice,
    width: u32,
    height: u32,
    uses: TextureUse,
    texels: &[F::Texel],
) -> GpuTexture2d<F> {
    let texture = device.create_texture_2d(width, height, uses).unwrap();
    device.write_texture(&texture, texels).unwrap();
    texture
}

fn twin_texture<F: TextureFormat>(
    cpu: &CpuDevice,
    width: u32,
    height: u32,
    uses: TextureUse,
    texels: &[F::Texel],
) -> CpuTexture2d<F> {
    let texture = cpu.texture_2d(width, height, uses).unwrap();
    texture.write(texels).unwrap();
    texture
}

/// Texels of the format `F` that no kernel has touched.
fn zeros<F: TextureFormat>(count: u32) -> Vec<F::Texel> {
    vec![F::Texel::default(); count as usize]
}

/// Bytes of a texel slice: bit-exact comparison, so a NaN payload or a negative zero that changes is a difference.
fn raw<T: GpuPod>(texels: &[T]) -> Vec<u8> {
    pod_slice_as_bytes(texels).to_vec()
}

// ---- load and store on all nine formats ----

/// One copy kernel of a class on the GPU and on the twin. The input texels are `texels::<F>()` (the awkward bit
/// patterns of `word` first), so the expectation is the input itself, which no kernel computed.
macro_rules! check_copy_class {
    ($device:expr, $cpu:expr, $kernel:path, $one:ty, $two:ty, $four:ty) => {{
        let extent = Extent::new(WIDTH, HEIGHT, 1);
        let (i1, i2, i4) = (texels::<$one>(), texels::<$two>(), texels::<$four>());
        let count = WIDTH * HEIGHT;
        // The GPU.
        let sampled = TextureUse::SAMPLED;
        let storage = TextureUse::STORAGE;
        let (g1, g2, g4) = (
            gpu_texture::<$one>($device, WIDTH, HEIGHT, sampled, &i1),
            gpu_texture::<$two>($device, WIDTH, HEIGHT, sampled, &i2),
            gpu_texture::<$four>($device, WIDTH, HEIGHT, sampled, &i4),
        );
        let (o1, o2, o4) = (
            gpu_texture::<$one>($device, WIDTH, HEIGHT, storage, &zeros::<$one>(count)),
            gpu_texture::<$two>($device, WIDTH, HEIGHT, storage, &zeros::<$two>(count)),
            gpu_texture::<$four>($device, WIDTH, HEIGHT, storage, &zeros::<$four>(count)),
        );
        $kernel($device, extent, &g1, &g2, &g4, &o1, &o2, &o4)
            .expect(concat!("the GPU refused ", stringify!($kernel)));
        let gpu = (
            $device.read_texture(&o1).unwrap(),
            $device.read_texture(&o2).unwrap(),
            $device.read_texture(&o4).unwrap(),
        );
        // The twin.
        let (t1, t2, t4) = (
            twin_texture::<$one>($cpu, WIDTH, HEIGHT, sampled, &i1),
            twin_texture::<$two>($cpu, WIDTH, HEIGHT, sampled, &i2),
            twin_texture::<$four>($cpu, WIDTH, HEIGHT, sampled, &i4),
        );
        let (p1, p2, p4) = (
            $cpu.texture_2d::<$one>(WIDTH, HEIGHT, storage).unwrap(),
            $cpu.texture_2d::<$two>(WIDTH, HEIGHT, storage).unwrap(),
            $cpu.texture_2d::<$four>(WIDTH, HEIGHT, storage).unwrap(),
        );
        $kernel($cpu, extent, &t1, &t2, &t4, &p1, &p2, &p4)
            .expect(concat!("the twin refused ", stringify!($kernel)));
        let twin = (p1.read(), p2.read(), p4.read());
        for (format, input, gpu, twin) in [
            (
                <$one as TextureFormat>::FORMAT,
                raw(&i1),
                raw(&gpu.0),
                raw(&twin.0),
            ),
            (
                <$two as TextureFormat>::FORMAT,
                raw(&i2),
                raw(&gpu.1),
                raw(&twin.1),
            ),
            (
                <$four as TextureFormat>::FORMAT,
                raw(&i4),
                raw(&gpu.2),
                raw(&twin.2),
            ),
        ] {
            assert_eq!(
                gpu, input,
                "{format}: the GPU copy differs from the input, bit for bit"
            );
            assert_eq!(
                twin, input,
                "{format}: the twin copy differs from the input, bit for bit"
            );
            assert_eq!(
                gpu, twin,
                "{format}: the GPU and the twin differ, bit for bit"
            );
        }
    }};
}

/// `load` from a sampled texture and `store` into a write-only one, on the nine formats, GPU against twin against the
/// input, bit for bit (including the negative zero, the infinities, the denormals and the NaN payloads of `word`).
#[test]
fn gpu_and_twin_load_and_store_agree_bit_for_bit_on_all_nine_formats() {
    let device = device();
    let cpu = CpuDevice::new();
    check_copy_class!(
        &device,
        &cpu,
        evidence::copy_floats::dispatch_extent,
        R32Float,
        Rg32Float,
        Rgba32Float
    );
    check_copy_class!(
        &device,
        &cpu,
        evidence::copy_uints::dispatch_extent,
        R32Uint,
        Rg32Uint,
        Rgba32Uint
    );
    check_copy_class!(
        &device,
        &cpu,
        evidence::copy_ints::dispatch_extent,
        R32Sint,
        Rg32Sint,
        Rgba32Sint
    );
}

/// The read-write storage path on the three r32 formats: both textures are loaded and stored in one kernel.
macro_rules! check_swap {
    ($device:expr, $cpu:expr, $kernel:path, $format:ty) => {{
        let extent = Extent::new(WIDTH, HEIGHT, 1);
        let left = texels::<$format>();
        let right: Vec<<$format as TextureFormat>::Texel> = (0..WIDTH * HEIGHT)
            .map(|index| FromWords::from_words(index + 7919))
            .collect();
        let storage = TextureUse::STORAGE;
        let (gl, gr) = (
            gpu_texture::<$format>($device, WIDTH, HEIGHT, storage, &left),
            gpu_texture::<$format>($device, WIDTH, HEIGHT, storage, &right),
        );
        $kernel($device, extent, &gl, &gr).expect("the GPU refused the swap");
        let (tl, tr) = (
            twin_texture::<$format>($cpu, WIDTH, HEIGHT, storage, &left),
            twin_texture::<$format>($cpu, WIDTH, HEIGHT, storage, &right),
        );
        $kernel($cpu, extent, &tl, &tr).expect("the twin refused the swap");
        let format = <$format as TextureFormat>::FORMAT;
        // After the swap the left texture holds the right's input and the other way round.
        assert_eq!(
            raw(&$device.read_texture(&gl).unwrap()),
            raw(&right),
            "{format}: GPU left"
        );
        assert_eq!(
            raw(&$device.read_texture(&gr).unwrap()),
            raw(&left),
            "{format}: GPU right"
        );
        assert_eq!(raw(&tl.read()), raw(&right), "{format}: twin left");
        assert_eq!(raw(&tr.read()), raw(&left), "{format}: twin right");
    }};
}

#[test]
fn gpu_and_twin_load_and_store_in_place_agree_on_the_three_storage_formats() {
    let device = device();
    let cpu = CpuDevice::new();
    check_swap!(&device, &cpu, evidence::swap_uint::dispatch_extent, R32Uint);
    check_swap!(&device, &cpu, evidence::swap_sint::dispatch_extent, R32Sint);
    check_swap!(
        &device,
        &cpu,
        evidence::swap_float::dispatch_extent,
        R32Float
    );
}

/// The fixture kernels that compute: `invert`, `accumulate`, `shift` and the three `write_*`, against loops over the
/// input. `shift` reads its neighbour from a separate sampled texture (it used to read the storage texture it writes:
/// invocation `x` read texel `x + 1`, which invocation `x + 1` writes, a race that only the twin's ascending order
/// resolves, so it ran nowhere).
#[test]
fn the_fixture_kernels_match_host_loops_on_the_gpu_and_the_twin() {
    let device = device();
    let cpu = CpuDevice::new();
    let extent = Extent::new(WIDTH, HEIGHT, 1);
    let count = WIDTH * HEIGHT;
    let storage = TextureUse::STORAGE;

    // invert: 1 - v, exact for sixteenths.
    let image: Vec<f32> = (0..count)
        .map(|i| ((i * 7) % 33) as f32 / 16.0 - 1.0)
        .collect();
    let inverted: Vec<f32> = image.iter().map(|v| 1.0 - v).collect();
    let g = gpu_texture::<R32Float>(&device, WIDTH, HEIGHT, storage, &image);
    textures::invert::dispatch_extent(&device, extent, &g).unwrap();
    assert_eq!(device.read_texture(&g).unwrap(), inverted, "GPU invert");
    let t = twin_texture::<R32Float>(&cpu, WIDTH, HEIGHT, storage, &image);
    textures::invert::dispatch_extent(&cpu, extent, &t).unwrap();
    assert_eq!(t.read(), inverted, "twin invert");

    // accumulate: totals += ids.
    let ids: Vec<u32> = (0..count).map(|i| i * 3 + 1).collect();
    let before: Vec<u32> = (0..count).map(|i| 1000 + i * 5).collect();
    let totals: Vec<u32> = ids.iter().zip(&before).map(|(a, b)| a + b).collect();
    let (gi, gt) = (
        gpu_texture::<R32Uint>(&device, WIDTH, HEIGHT, TextureUse::SAMPLED, &ids),
        gpu_texture::<R32Uint>(&device, WIDTH, HEIGHT, storage, &before),
    );
    textures::accumulate::dispatch_extent(&device, extent, &gi, &gt).unwrap();
    assert_eq!(device.read_texture(&gt).unwrap(), totals, "GPU accumulate");
    let (ti, tt) = (
        twin_texture::<R32Uint>(&cpu, WIDTH, HEIGHT, TextureUse::SAMPLED, &ids),
        twin_texture::<R32Uint>(&cpu, WIDTH, HEIGHT, storage, &before),
    );
    textures::accumulate::dispatch_extent(&cpu, extent, &ti, &tt).unwrap();
    assert_eq!(tt.read(), totals, "twin accumulate");

    // shift: field[x, y] = source[x + 1, y] - 1, the last column untouched. The source and the field are two textures.
    let source: Vec<i32> = (0..count as i32).map(|i| i * 3 - 100).collect();
    let field_before: Vec<i32> = (0..count as i32).map(|i| 5000 + i).collect();
    let shifted: Vec<i32> = (0..count)
        .map(|i| {
            if i % WIDTH + 1 < WIDTH {
                source[i as usize + 1] - 1
            } else {
                field_before[i as usize]
            }
        })
        .collect();
    let (gs, gf) = (
        gpu_texture::<R32Sint>(&device, WIDTH, HEIGHT, TextureUse::SAMPLED, &source),
        gpu_texture::<R32Sint>(&device, WIDTH, HEIGHT, storage, &field_before),
    );
    textures::shift::dispatch_extent(&device, extent, &gs, &gf).unwrap();
    assert_eq!(device.read_texture(&gf).unwrap(), shifted, "GPU shift");
    let (ts, tf) = (
        twin_texture::<R32Sint>(&cpu, WIDTH, HEIGHT, TextureUse::SAMPLED, &source),
        twin_texture::<R32Sint>(&cpu, WIDTH, HEIGHT, storage, &field_before),
    );
    textures::shift::dispatch_extent(&cpu, extent, &ts, &tf).unwrap();
    assert_eq!(tf.read(), shifted, "twin shift");

    // write_floats, write_uints, write_ints: constants and the texel's own address.
    let at = |i: u32| (i % WIDTH, i / WIDTH);
    let (f1, f2, f4): (Vec<f32>, Vec<Vec2>, Vec<Vec4>) = (
        vec![1.0; count as usize],
        vec![Vec2::new(1.0, 2.0); count as usize],
        vec![Vec4::new(1.0, 2.0, 3.0, 4.0); count as usize],
    );
    let (u1, u2, u4): (Vec<u32>, Vec<UVec2>, Vec<UVec4>) = (
        (0..count).map(|i| at(i).0).collect(),
        (0..count).map(|i| UVec2::new(at(i).0, at(i).1)).collect(),
        (0..count)
            .map(|i| UVec4::new(at(i).0, at(i).1, 3, 4))
            .collect(),
    );
    let (s1, s2, s4): (Vec<i32>, Vec<IVec2>, Vec<IVec4>) = (
        vec![-1; count as usize],
        vec![IVec2::new(-1, 2); count as usize],
        vec![IVec4::new(-1, 2, -3, 4); count as usize],
    );
    macro_rules! check_write {
        ($kernel:path, $one:ty, $two:ty, $four:ty, $expect:expr) => {{
            let (e1, e2, e4) = $expect;
            let out = (
                gpu_texture::<$one>(&device, WIDTH, HEIGHT, storage, &zeros::<$one>(count)),
                gpu_texture::<$two>(&device, WIDTH, HEIGHT, storage, &zeros::<$two>(count)),
                gpu_texture::<$four>(&device, WIDTH, HEIGHT, storage, &zeros::<$four>(count)),
            );
            $kernel(&device, extent, &out.0, &out.1, &out.2).unwrap();
            assert_eq!(
                raw(&device.read_texture(&out.0).unwrap()),
                raw(&e1),
                "GPU {}",
                stringify!($kernel)
            );
            assert_eq!(
                raw(&device.read_texture(&out.1).unwrap()),
                raw(&e2),
                "GPU {}",
                stringify!($kernel)
            );
            assert_eq!(
                raw(&device.read_texture(&out.2).unwrap()),
                raw(&e4),
                "GPU {}",
                stringify!($kernel)
            );
            let twin = (
                twin_texture::<$one>(&cpu, WIDTH, HEIGHT, storage, &zeros::<$one>(count)),
                twin_texture::<$two>(&cpu, WIDTH, HEIGHT, storage, &zeros::<$two>(count)),
                twin_texture::<$four>(&cpu, WIDTH, HEIGHT, storage, &zeros::<$four>(count)),
            );
            $kernel(&cpu, extent, &twin.0, &twin.1, &twin.2).unwrap();
            assert_eq!(
                raw(&twin.0.read()),
                raw(&e1),
                "twin {}",
                stringify!($kernel)
            );
            assert_eq!(
                raw(&twin.1.read()),
                raw(&e2),
                "twin {}",
                stringify!($kernel)
            );
            assert_eq!(
                raw(&twin.2.read()),
                raw(&e4),
                "twin {}",
                stringify!($kernel)
            );
        }};
    }
    check_write!(
        textures::write_floats::dispatch_extent,
        R32Float,
        Rg32Float,
        Rgba32Float,
        (f1, f2, f4)
    );
    check_write!(
        textures::write_uints::dispatch_extent,
        R32Uint,
        Rg32Uint,
        Rgba32Uint,
        (u1, u2, u4)
    );
    check_write!(
        textures::write_ints::dispatch_extent,
        R32Sint,
        Rg32Sint,
        Rgba32Sint,
        (s1, s2, s4)
    );
}

// ---- the twin refuses an address outside the texture; the same texture cannot be bound twice ----

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "<non-string panic payload>".to_owned()
    }
}

/// An address outside the texture is a panic "index out of bounds" on the twin, for `store` and for `load`, also when
/// the flat index `y * width + x` would exist (`x == width` is texel `(0, y + 1)` of a flat array). The texture is
/// untouched by a refused store. The addresses inside the texture work, so the panic is the address and nothing else.
#[test]
fn a_twin_store_or_load_outside_the_texture_panics() {
    const W: u32 = 4;
    const H: u32 = 2;
    let cpu = CpuDevice::new();
    let initial: Vec<u32> = (10..10 + W * H).collect();
    let texture = twin_texture::<R32Uint>(&cpu, W, H, TextureUse::STORAGE, &initial);
    let seen = cpu.buffer(&[0xAAAA_AAAAu32]);

    // Inside: the corner texel.
    let corner = cpu.buffer(&[W - 1, H - 1]);
    evidence::poke::dispatch(&cpu, 1, Independent(&corner), &texture).unwrap();
    let mut expected = initial.clone();
    expected[(W * H - 1) as usize] = 7;
    assert_eq!(texture.read(), expected);
    evidence::peek::dispatch(&cpu, 1, Independent(&corner), &texture, Independent(&seen)).unwrap();
    assert_eq!(seen.read(), [7]);

    for (x, y) in [(W, 0), (0, H), (W, H), (W, 1), (u32::MAX, 0), (0, u32::MAX)] {
        let at = cpu.buffer(&[x, y]);
        let stored = catch_unwind(AssertUnwindSafe(|| {
            evidence::poke::dispatch(&cpu, 1, Independent(&at), &texture)
        }));
        match stored {
            Err(payload) => assert!(
                panic_text(payload).starts_with("index out of bounds"),
                "store at ({x}, {y}): the panic message"
            ),
            Ok(result) => panic!("store at ({x}, {y}) did not panic: {result:?}"),
        }
        assert_eq!(
            texture.read(),
            expected,
            "a refused store at ({x}, {y}) wrote"
        );
        let loaded = catch_unwind(AssertUnwindSafe(|| {
            evidence::peek::dispatch(&cpu, 1, Independent(&at), &texture, Independent(&seen))
        }));
        match loaded {
            Err(payload) => assert!(
                panic_text(payload).starts_with("index out of bounds"),
                "load at ({x}, {y}): the panic message"
            ),
            Ok(result) => panic!("load at ({x}, {y}) did not panic: {result:?}"),
        }
        assert_eq!(seen.read(), [7], "a refused load at ({x}, {y}) wrote");
    }
}

/// One texture bound to a sampled parameter and a storage parameter of one call. Both backends refuse it before any
/// texel moves, and the texture keeps its contents. The twin names the two parameters in `Error::Aliased` (its
/// `validate` finds the alias by identity before any borrow is taken, so `Error::Borrowed` is not reachable through a
/// call that binds the texture twice); the GPU reports `Error::AliasedBinding` for the same call.
#[test]
fn binding_the_same_texture_twice_is_refused_before_anything_runs() {
    let device = device();
    let cpu = CpuDevice::new();
    let extent = Extent::new(4, 2, 1);
    let both = TextureUse::SAMPLED.union(TextureUse::STORAGE);
    let initial: Vec<u32> = (1..=8).collect();

    let twin = twin_texture::<R32Uint>(&cpu, 4, 2, both, &initial);
    match textures::accumulate::dispatch_extent(&cpu, extent, &twin, &twin) {
        Err(gust::cpu::Error::Aliased { first, second, .. }) => {
            assert_eq!((first, second), ("ids", "totals"));
        }
        other => panic!("the twin: expected Aliased, got {other:?}"),
    }
    assert_eq!(
        twin.read(),
        initial,
        "the twin wrote through a refused call"
    );

    let gpu = gpu_texture::<R32Uint>(&device, 4, 2, both, &initial);
    match textures::accumulate::dispatch_extent(&device, extent, &gpu, &gpu) {
        Err(Error::AliasedBinding { .. }) => {}
        other => panic!("the GPU: expected AliasedBinding, got {other:?}"),
    }
    assert_eq!(
        device.read_texture(&gpu).unwrap(),
        initial,
        "the GPU wrote through a refused call"
    );
}

// ---- sampling ----

/// A float texel type: one of the three the sampling kernels take.
trait Sampled: GpuPod + Copy + Default {
    type F: TextureFormat<Texel = Self>;
    const CHANNELS: usize;
    fn channel(&self, index: usize) -> f32;
    /// The texel `index` of the test texture: channels spread over about -8 to 8 with full mantissas.
    fn make(index: u32) -> Self;
    fn gpu(
        device: &HeadlessDevice,
        texture: &GpuTexture2d<Self::F>,
        sampler: &GpuSampler,
        us: &[f32],
        vs: &[f32],
    ) -> Result<Vec<Self>, Error>;
    fn twin(
        cpu: &CpuDevice,
        texture: &CpuTexture2d<Self::F>,
        sampler: &CpuSampler,
        us: &[f32],
        vs: &[f32],
    ) -> Vec<Self>;
}

/// A channel value: a hash of (texel, channel) mapped to [-8, 8) with a full 24-bit mantissa.
fn spread(index: u32, channel: u32) -> f32 {
    let mixed = index
        .wrapping_mul(2_654_435_761)
        .wrapping_add(channel.wrapping_mul(0x9e37_79b9))
        .rotate_left(13)
        .wrapping_mul(0x85eb_ca6b);
    (mixed >> 8) as f32 / 16_777_216.0 * 16.0 - 8.0
}

macro_rules! sampled_impl {
    ($texel:ty, $format:ty, $kernel:ident, $channels:expr, |$t:ident, $i:ident| $channel:expr, |$k:ident| $make:expr) => {
        impl Sampled for $texel {
            type F = $format;
            const CHANNELS: usize = $channels;
            fn channel(&self, $i: usize) -> f32 {
                let $t = self;
                $channel
            }
            fn make($k: u32) -> Self {
                $make
            }
            fn gpu(
                device: &HeadlessDevice,
                texture: &GpuTexture2d<Self::F>,
                sampler: &GpuSampler,
                us: &[f32],
                vs: &[f32],
            ) -> Result<Vec<Self>, Error> {
                let count = us.len() as u32;
                let us = device.create_typed_buffer("us", us, GpuBufferAccess::ReadOnly);
                let vs = device.create_typed_buffer("vs", vs, GpuBufferAccess::ReadOnly);
                let out = device.create_typed_buffer(
                    "out",
                    &vec![Self::default(); count as usize],
                    GpuBufferAccess::ReadWrite,
                );
                evidence::$kernel::dispatch(device, count, texture, sampler, &us, &vs, &out)?;
                device.read_typed_buffer(&out)
            }
            fn twin(
                cpu: &CpuDevice,
                texture: &CpuTexture2d<Self::F>,
                sampler: &CpuSampler,
                us: &[f32],
                vs: &[f32],
            ) -> Vec<Self> {
                let (us_b, vs_b) = (cpu.buffer(us), cpu.buffer(vs));
                let out = cpu.buffer(&vec![Self::default(); us.len()]);
                evidence::$kernel::dispatch(
                    cpu,
                    us.len() as u32,
                    texture,
                    sampler,
                    &us_b,
                    &vs_b,
                    &out,
                )
                .expect("the twin refused the sampling kernel");
                out.read()
            }
        }
    };
}

sampled_impl!(
    f32,
    R32Float,
    sample_scalar,
    1,
    |t, i| {
        let _ = i;
        *t
    },
    |k| spread(k, 0)
);
sampled_impl!(
    Vec2,
    Rg32Float,
    sample_pair,
    2,
    |t, i| if i == 0 { t.x } else { t.y },
    |k| Vec2::new(spread(k, 0), spread(k, 1))
);
sampled_impl!(
    Vec4,
    Rgba32Float,
    sample_quad,
    4,
    |t, i| [t.x, t.y, t.z, t.w][i],
    |k| Vec4::new(spread(k, 0), spread(k, 1), spread(k, 2), spread(k, 3))
);

const SAMPLE_W: u32 = 5;
const SAMPLE_H: u32 = 4;

/// The distances from a texel edge, in texels, as powers of two, at which the nearest band is probed: 2^-20, 2^-12,
/// 2^-9, 2^-8 and 2^-7.
const BAND_EXPONENTS: [i32; 5] = [-20, -12, -9, -8, -7];

/// The sample coordinates: a grid from -1.3 to 2.3 on both axes (outside the unit square on every side, so the clamp
/// and the repeat both matter), plus every texel centre.
fn coordinates() -> (Vec<f32>, Vec<f32>) {
    let (mut us, mut vs) = (Vec::new(), Vec::new());
    for j in 0..20 {
        for i in 0..24 {
            us.push((-1.3 + 3.6 * f64::from(i) / 23.0) as f32);
            vs.push((-1.3 + 3.6 * f64::from(j) / 19.0) as f32);
        }
    }
    for y in 0..SAMPLE_H {
        for x in 0..SAMPLE_W {
            us.push((f64::from(x) + 0.5) as f32 / SAMPLE_W as f32);
            vs.push((f64::from(y) + 0.5) as f32 / SAMPLE_H as f32);
        }
    }
    // The edges between texels, on both axes: the coordinate of each edge and the floats one step either side of it
    // (a nearest read there is the case of KM-14). For the edge at 0 the step below is the bits of 0 minus one, which
    // is a NaN, and `.max(0.0)` turns a NaN into 0: that probe is the coordinate 0 again, not a negative one.
    let step = |value: f32, by: i32| f32::from_bits(value.to_bits().wrapping_add_signed(by));
    for k in 0..=SAMPLE_W {
        let edge = k as f32 / SAMPLE_W as f32;
        for by in [-1, 0, 1] {
            us.push(step(edge, by).max(0.0));
            vs.push(0.375);
        }
    }
    for k in 0..=SAMPLE_H {
        let edge = k as f32 / SAMPLE_H as f32;
        for by in [-1, 0, 1] {
            us.push(0.3);
            vs.push(step(edge, by).max(0.0));
        }
    }
    // The nearest band, measured (fix round 1): each edge and each border, at `BAND_EXPONENTS` texels either side. A
    // position below 0 is clamped to 0 (the border itself, not an edge that a clamp can cross).
    for k in 0..=SAMPLE_W {
        for exponent in BAND_EXPONENTS {
            for side in [-1.0f64, 1.0] {
                let position = f64::from(k) + side * 2f64.powi(exponent);
                us.push((position / f64::from(SAMPLE_W)).max(0.0) as f32);
                vs.push(0.375);
            }
        }
    }
    for k in 0..=SAMPLE_H {
        for exponent in BAND_EXPONENTS {
            for side in [-1.0f64, 1.0] {
                let position = f64::from(k) + side * 2f64.powi(exponent);
                us.push(0.3);
                vs.push((position / f64::from(SAMPLE_H)).max(0.0) as f32);
            }
        }
    }
    (us, vs)
}

/// The independent host reference of one sample: the WGSL conventions evaluated in `f64`, from the texels alone (it
/// shares no code with the twin). Returns the value of each channel, the four texels each channel reads (for the
/// error bound), the texel a nearest read picks, and the distance of the unnormalised position to the nearest texel
/// edge (an integer), in texels, on the nearer axis.
struct Reference {
    value: Vec<f64>,
    corners: Vec<[f32; 4]>,
    nearest: (usize, usize),
    edge_distance: f64,
}

fn address(coordinate: f64, mode: Address) -> f64 {
    match mode {
        Address::ClampToEdge => coordinate.clamp(0.0, 1.0),
        Address::Repeat => coordinate - coordinate.floor(),
        other => panic!("unknown address mode {other:?}"),
    }
}

fn texel_index_of(index: i64, size: u32, mode: Address) -> usize {
    match mode {
        Address::ClampToEdge => index.clamp(0, i64::from(size) - 1) as usize,
        Address::Repeat => index.rem_euclid(i64::from(size)) as usize,
        other => panic!("unknown address mode {other:?}"),
    }
}

/// The distance of the position `x` (in texels, after the address mode) to the nearest edge between two texels that a
/// nearest read can cross. Clamped to the first or the last edge nothing is crossed (the read is the edge texel), so
/// that distance is infinite; under repeat the border wraps to the opposite texel and is an edge like any other.
fn edge(x: f64, size: u32, mode: Address) -> f64 {
    let nearest = x.round();
    if mode == Address::ClampToEdge && (nearest <= 0.0 || nearest >= f64::from(size)) {
        return f64::INFINITY;
    }
    (x - nearest).abs()
}

fn reference<T: Sampled>(texels: &[T], mode: Address, filter: Filter, u: f32, v: f32) -> Reference {
    let (w, h) = (SAMPLE_W, SAMPLE_H);
    let x = address(f64::from(u), mode) * f64::from(w);
    let y = address(f64::from(v), mode) * f64::from(h);
    let at = |i: usize, j: usize, c: usize| texels[j * w as usize + i].channel(c);
    let nearest = (
        (x.floor() as i64).clamp(0, i64::from(w) - 1) as usize,
        (y.floor() as i64).clamp(0, i64::from(h) - 1) as usize,
    );
    let edge_distance = edge(x, w, mode).min(edge(y, h, mode));
    let mut value = Vec::new();
    let mut corners = Vec::new();
    for c in 0..T::CHANNELS {
        match filter {
            Filter::Nearest => {
                value.push(f64::from(at(nearest.0, nearest.1, c)));
                corners.push([at(nearest.0, nearest.1, c); 4]);
            }
            Filter::Linear => {
                let (px, py) = (x - 0.5, y - 0.5);
                let (i0, j0) = (px.floor() as i64, py.floor() as i64);
                let (fx, fy) = (px - px.floor(), py - py.floor());
                let (i0, i1) = (texel_index_of(i0, w, mode), texel_index_of(i0 + 1, w, mode));
                let (j0, j1) = (texel_index_of(j0, h, mode), texel_index_of(j0 + 1, h, mode));
                let t = [at(i0, j0, c), at(i1, j0, c), at(i0, j1, c), at(i1, j1, c)];
                let lerp =
                    |a: f32, b: f32, f: f64| f64::from(a) + (f64::from(b) - f64::from(a)) * f;
                let top = lerp(t[0], t[1], fx);
                let bottom = lerp(t[2], t[3], fx);
                value.push(top + (bottom - top) * fy);
                corners.push(t);
            }
            other => panic!("unknown filter {other:?}"),
        }
    }
    Reference {
        value,
        corners,
        nearest,
        edge_distance,
    }
}

/// What one sampling run observed, for the report and for the ledger rows KM-13 and KM-14.
#[derive(Debug, Default)]
struct Observation {
    /// Samples compared.
    samples: usize,
    /// The largest `|gpu - twin|` over the bound of that channel (`linear`), 0 for an exact comparison.
    worst_ratio: f64,
    /// Nearest reads within the edge band: compared, and how many differ from the formula's texel.
    near_edge: usize,
    near_edge_other_texel: usize,
    /// Every nearest read less than 0.05 texel from an edge: its distance from the edge (texels) and whether the GPU
    /// picked a texel other than the formula's.
    edge_samples: Vec<(f64, bool)>,
}

/// 1/128 of a texel: a nearest read this close to an edge is not asserted exactly (KM-14), farther away it must be.
///
/// This widens SEMANTICS section 3.5, which allows a neighbour only "within an ulp" of an edge. A sampler works on a
/// coordinate with a finite number of sub-texel bits; 1/128 covers 7 or more of them. That count is not read from any
/// device (inferred: Vulkan guarantees only `subTexelPrecisionBits >= 4`, which would need 1/16). What a device does
/// is measured instead: the nearest test prints, per device and address mode, the picks at `BAND_EXPONENTS`.
const EDGE_BAND: f64 = 1.0 / 128.0;

fn check_sampling<T: Sampled>(
    device: &HeadlessDevice,
    filter: Filter,
    address: Address,
) -> Observation {
    let options = SamplerOptions::new(filter, address);
    let (us, vs) = coordinates();
    let texels: Vec<T> = (0..SAMPLE_W * SAMPLE_H).map(T::make).collect();
    let texture = gpu_texture::<T::F>(device, SAMPLE_W, SAMPLE_H, TextureUse::SAMPLED, &texels);
    let sampler = device.create_sampler(options).unwrap();
    let gpu = T::gpu(device, &texture, &sampler, &us, &vs).unwrap_or_else(|error| {
        panic!(
            "{filter:?} {address:?} on {}: the GPU refused the call: {error}",
            <T::F as TextureFormat>::FORMAT
        )
    });
    let cpu = CpuDevice::new();
    let twin_tex = twin_texture::<T::F>(&cpu, SAMPLE_W, SAMPLE_H, TextureUse::SAMPLED, &texels);
    let twin = T::twin(&cpu, &twin_tex, &cpu.sampler(options), &us, &vs);
    let label = format!("{} {filter:?} {address:?}", <T::F as TextureFormat>::FORMAT);
    let mut seen = Observation::default();
    for n in 0..us.len() {
        let reference = reference(&texels, address, filter, us[n], vs[n]);
        for c in 0..T::CHANNELS {
            let (g, t, r) = (gpu[n].channel(c), twin[n].channel(c), reference.value[c]);
            let position = format!(
                "{label} at (u, v) = ({}, {}), channel {c}: gpu {g}, twin {t}, reference {r}",
                us[n], vs[n]
            );
            match filter {
                Filter::Nearest => {
                    if reference.edge_distance >= EDGE_BAND {
                        // Away from an edge every texel pick is decided: exact on the twin, the reference and the GPU.
                        assert_eq!(
                            t.to_bits(),
                            (r as f32).to_bits(),
                            "twin against the reference: {position}"
                        );
                        assert_eq!(g.to_bits(), t.to_bits(), "GPU against the twin: {position}");
                    } else if c == 0 {
                        // Within the band the GPU may pick the neighbouring texel; it picks one of the four texels
                        // around the edge, never another.
                        seen.near_edge += 1;
                        if g.to_bits() != t.to_bits() {
                            seen.near_edge_other_texel += 1;
                        }
                    }
                    if c == 0 && reference.edge_distance < 0.05 {
                        seen.edge_samples
                            .push((reference.edge_distance, g.to_bits() != t.to_bits()));
                    }
                    if reference.edge_distance < EDGE_BAND {
                        // The texels one step from the formula's, in each direction, with the address mode applied to
                        // the step (so the repeat's border wraps to the opposite texel).
                        let (nx, ny) = (reference.nearest.0 as i64, reference.nearest.1 as i64);
                        let mut around = Vec::new();
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                let i = texel_index_of(nx + dx, SAMPLE_W, address);
                                let j = texel_index_of(ny + dy, SAMPLE_H, address);
                                around.push(texels[j * SAMPLE_W as usize + i].channel(c).to_bits());
                            }
                        }
                        assert!(
                            around.contains(&g.to_bits()),
                            "an edge read is a neighbouring texel: {position}"
                        );
                    }
                }
                Filter::Linear => {
                    let bound = f64::from(linear_error_bound(reference.corners[c]));
                    let from_twin = (f64::from(g) - f64::from(t)).abs();
                    let from_reference = (f64::from(g) - r).abs();
                    let twin_from_reference = (f64::from(t) - r).abs();
                    assert!(
                        from_twin <= bound,
                        "GPU against the twin, {from_twin} > {bound}: {position}"
                    );
                    assert!(
                        from_reference <= bound,
                        "GPU against the reference, {from_reference} > {bound}: {position}"
                    );
                    assert!(
                        twin_from_reference <= bound,
                        "twin against the reference, {twin_from_reference} > {bound}: {position}"
                    );
                    if bound > 0.0 {
                        seen.worst_ratio = seen.worst_ratio.max(from_twin / bound);
                    }
                }
                other => panic!("unknown filter {other:?}"),
            }
        }
        seen.samples += 1;
    }
    eprintln!(
        "MX-T4 sampling {label}: {} samples, worst ratio {:.4}, {} in the edge band, {} of them another texel",
        seen.samples, seen.worst_ratio, seen.near_edge, seen.near_edge_other_texel
    );
    seen
}

fn all_sampled(device: &HeadlessDevice, filter: Filter, address: Address) -> Observation {
    let one = check_sampling::<f32>(device, filter, address);
    let two = check_sampling::<Vec2>(device, filter, address);
    let four = check_sampling::<Vec4>(device, filter, address);
    Observation {
        samples: one.samples + two.samples + four.samples,
        worst_ratio: one.worst_ratio.max(two.worst_ratio).max(four.worst_ratio),
        near_edge: one.near_edge + two.near_edge + four.near_edge,
        near_edge_other_texel: one.near_edge_other_texel
            + two.near_edge_other_texel
            + four.near_edge_other_texel,
        edge_samples: [one.edge_samples, two.edge_samples, four.edge_samples].concat(),
    }
}

/// Nearest `sample`, clamp and repeat, on the default device (the non-filterable layout) and on the device that
/// enabled `float32-filterable`: away from a texel edge the GPU picks the texel the formula picks, exactly, and the
/// independent reference agrees with the twin.
#[test]
fn nearest_sampling_picks_the_texel_of_the_formula_under_clamp_and_repeat() {
    for device in [device(), feature_device()] {
        for address in [Address::ClampToEdge, Address::Repeat] {
            let seen = all_sampled(&device, Filter::Nearest, address);
            assert!(seen.samples > 0);
            eprintln!(
                "MX-T4 nearest {address:?} float32-filterable={}: {} samples, {} in the edge band, {} of them a texel other than the formula's",
                device.float32_filterable(),
                seen.samples,
                seen.near_edge,
                seen.near_edge_other_texel
            );
            // The band, measured: at each probed distance, how many reads picked a texel other than the formula's.
            let mut table = String::new();
            for exponent in BAND_EXPONENTS {
                let at: Vec<&(f64, bool)> = seen
                    .edge_samples
                    .iter()
                    .filter(|(distance, _)| distance.log2().round() == f64::from(exponent))
                    .collect();
                assert!(!at.is_empty(), "no sample at 2^{exponent} texel");
                let other = at.iter().filter(|(_, other)| *other).count();
                table += &format!(" 2^{exponent}: {other}/{};", at.len());
            }
            let farthest = seen
                .edge_samples
                .iter()
                .filter(|(_, other)| *other)
                .map(|(distance, _)| *distance)
                .fold(0.0, f64::max);
            eprintln!(
                "MX-T4 nearest band {address:?} float32-filterable={}: picks of another texel / samples at each distance:{table} the farthest pick of another texel is at {farthest:e} texel",
                device.float32_filterable()
            );
        }
    }
}

/// Linear `sample`, clamp and repeat, on three float texel types: `|gpu - twin|`, `|gpu - reference|` and
/// `|twin - reference|` are each within `2^-8 * (max - min of the four texels) + 2 ulp(max |texel|)` per channel.
/// A device without `float32-filterable` refuses a linear sampler on these formats (`Error::TextureFilter`), and the
/// test then asserts that refusal instead: nothing is skipped silently.
#[test]
fn linear_sampling_is_within_the_error_domain_under_clamp_and_repeat() {
    let device = feature_device();
    if !device.float32_filterable() {
        let texels: Vec<f32> = (0..SAMPLE_W * SAMPLE_H)
            .map(<f32 as Sampled>::make)
            .collect();
        let texture =
            gpu_texture::<R32Float>(&device, SAMPLE_W, SAMPLE_H, TextureUse::SAMPLED, &texels);
        let sampler = device
            .create_sampler(SamplerOptions::new(Filter::Linear, Address::ClampToEdge))
            .unwrap();
        let (us, vs) = coordinates();
        let result = <f32 as Sampled>::gpu(&device, &texture, &sampler, &us, &vs);
        assert!(
            matches!(result, Err(Error::TextureFilter { .. })),
            "{result:?}"
        );
        eprintln!(
            "MX-T4 linear: {:?} does not offer float32-filterable; asserted the TextureFilter refusal",
            device.adapter_info().name
        );
        return;
    }
    for address in [Address::ClampToEdge, Address::Repeat] {
        let seen = all_sampled(&device, Filter::Linear, address);
        assert!(seen.samples > 0);
        eprintln!(
            "MX-T4 linear {address:?}: {} samples, the largest |gpu - twin| is {:.4} of its bound",
            seen.samples, seen.worst_ratio
        );
    }
}

/// The two fixture kernels that sample (`blur`, four bilinear reads per texel, and `gather`, one nearest and one linear
/// read beside integer loads) on the GPU against the twin. A linear read is within `linear_error_bound` of the twin's
/// over the whole texture's range, so `blur` is within the mean of four such bounds plus the rounding of its sum, and
/// `gather` is within one bound plus one ulp on its two sampled channels and exact on its two loaded ones.
#[test]
fn the_fixture_kernels_that_sample_agree_with_the_twin_within_the_bound() {
    let device = feature_device();
    if !device.float32_filterable() {
        // Without the feature a linear sampler of a float32 texture is refused, as in the linear test.
        let sampled = TextureUse::SAMPLED;
        let image = gpu_texture::<Rgba32Float>(&device, 2, 2, sampled, &zeros::<Rgba32Float>(4));
        let target = gpu_texture::<Rgba32Float>(
            &device,
            2,
            2,
            TextureUse::STORAGE,
            &zeros::<Rgba32Float>(4),
        );
        let linear = SamplerOptions::new(Filter::Linear, Address::ClampToEdge);
        let sampler = device.create_sampler(linear).unwrap();
        let result = textures::blur::dispatch_extent(
            &device,
            Extent::new(2, 2, 1),
            &image,
            &sampler,
            &target,
        );
        assert!(
            matches!(result, Err(Error::TextureFilter { .. })),
            "{result:?}"
        );
        eprintln!(
            "MX-T4 blur/gather: no float32-filterable on this adapter; asserted the TextureFilter refusal"
        );
        return;
    }
    let cpu = CpuDevice::new();
    let (w, h) = (6u32, 5u32);
    let count = w * h;
    let extent = Extent::new(w, h, 1);
    let storage = TextureUse::STORAGE;
    let sampled = TextureUse::SAMPLED;
    let range_bound = |channel: &dyn Fn(u32) -> f32| {
        let values: Vec<f32> = (0..count).map(channel).collect();
        let (low, high) = values
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, hi), &v| (l.min(v), hi.max(v)));
        f64::from(linear_error_bound([low, high, low, high]))
    };

    // blur: Vec4 -> Vec4 through four bilinear reads, clamped.
    let image: Vec<Vec4> = (0..count).map(<Vec4 as Sampled>::make).collect();
    let options = SamplerOptions::new(Filter::Linear, Address::ClampToEdge);
    let (gi, go) = (
        gpu_texture::<Rgba32Float>(&device, w, h, sampled, &image),
        gpu_texture::<Rgba32Float>(&device, w, h, storage, &zeros::<Rgba32Float>(count)),
    );
    let sampler = device.create_sampler(options).unwrap();
    textures::blur::dispatch_extent(&device, extent, &gi, &sampler, &go).unwrap();
    let (ti, to) = (
        twin_texture::<Rgba32Float>(&cpu, w, h, sampled, &image),
        twin_texture::<Rgba32Float>(&cpu, w, h, storage, &zeros::<Rgba32Float>(count)),
    );
    textures::blur::dispatch_extent(&cpu, extent, &ti, &cpu.sampler(options), &to).unwrap();
    let (gpu, twin) = (device.read_texture(&go).unwrap(), to.read());
    for c in 0..4 {
        let bound = range_bound(&|i| image[i as usize].channel(c));
        let mut worst = 0.0f64;
        for n in 0..count as usize {
            let (g, t) = (f64::from(gpu[n].channel(c)), f64::from(twin[n].channel(c)));
            // The mean of four reads, each within `bound`, then the rounding of three additions and a product.
            let limit = bound + 4.0 * f64::from(ulp(twin[n].channel(c).abs().max(1.0)));
            assert!(
                (g - t).abs() <= limit,
                "blur texel {n} channel {c}: gpu {g}, twin {t}, limit {limit}"
            );
            worst = worst.max((g - t).abs() / limit);
        }
        eprintln!("MX-T4 blur channel {c}: the largest |gpu - twin| is {worst:.4} of its limit");
    }

    // gather: nearest `scalar`, linear `pair`, integer loads of `counts` and `offsets`.
    let scalar: Vec<f32> = (0..count).map(<f32 as Sampled>::make).collect();
    let pair: Vec<Vec2> = (0..count).map(<Vec2 as Sampled>::make).collect();
    let counts: Vec<UVec4> = (0..count).map(|i| UVec4::new(i, 2, 3, 100 + i)).collect();
    let offsets: Vec<IVec2> = (0..count)
        .map(|i| IVec2::new(i as i32 * 3, 7 - i as i32))
        .collect();
    let nearest = SamplerOptions::new(Filter::Nearest, Address::ClampToEdge);
    let (g_scalar, g_pair, g_counts, g_offsets, g_out) = (
        gpu_texture::<R32Float>(&device, w, h, sampled, &scalar),
        gpu_texture::<Rg32Float>(&device, w, h, sampled, &pair),
        gpu_texture::<Rgba32Uint>(&device, w, h, sampled, &counts),
        gpu_texture::<Rg32Sint>(&device, w, h, sampled, &offsets),
        gpu_texture::<Rgba32Float>(&device, w, h, storage, &zeros::<Rgba32Float>(count)),
    );
    let (s_nearest, s_linear) = (
        device.create_sampler(nearest).unwrap(),
        device.create_sampler(options).unwrap(),
    );
    textures::gather::dispatch_extent(
        &device, extent, &g_scalar, &g_pair, &g_counts, &g_offsets, &s_nearest, &s_linear, &g_out,
    )
    .unwrap();
    let (t_scalar, t_pair, t_counts, t_offsets, t_out) = (
        twin_texture::<R32Float>(&cpu, w, h, sampled, &scalar),
        twin_texture::<Rg32Float>(&cpu, w, h, sampled, &pair),
        twin_texture::<Rgba32Uint>(&cpu, w, h, sampled, &counts),
        twin_texture::<Rg32Sint>(&cpu, w, h, sampled, &offsets),
        twin_texture::<Rgba32Float>(&cpu, w, h, storage, &zeros::<Rgba32Float>(count)),
    );
    textures::gather::dispatch_extent(
        &cpu,
        extent,
        &t_scalar,
        &t_pair,
        &t_counts,
        &t_offsets,
        &cpu.sampler(nearest),
        &cpu.sampler(options),
        &t_out,
    )
    .unwrap();
    let (gpu, twin) = (device.read_texture(&g_out).unwrap(), t_out.read());
    let bound_x = range_bound(&|i| pair[i as usize].x);
    let bound_y = range_bound(&|i| pair[i as usize].y);
    for n in 0..count as usize {
        let (g, t) = (gpu[n], twin[n]);
        // Channel 0 is `nearest scalar + linear pair.x`: the nearest read is the same texel at a texel centre, so its
        // error is the linear read's plus one rounding of the sum.
        let limit_x = bound_x + f64::from(ulp(t.x.abs().max(1.0)));
        assert!(
            (f64::from(g.x) - f64::from(t.x)).abs() <= limit_x,
            "gather texel {n} x: {g:?} against {t:?}"
        );
        assert!(
            (f64::from(g.y) - f64::from(t.y)).abs() <= bound_y,
            "gather texel {n} y: {g:?} against {t:?}"
        );
        assert_eq!(
            (g.z.to_bits(), g.w.to_bits()),
            (t.z.to_bits(), t.w.to_bits()),
            "gather texel {n} loads"
        );
    }
}
