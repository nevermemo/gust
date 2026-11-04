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
//! What waits for mx.t3 and mx.t4: kernels written in Rust with `Texture2D` parameters, and the generated typed
//! `call` functions for them. One typed call is built by hand here (`Call::new_with_samplers`).

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
    BufferBinding, BufferDispatch, DeviceOptions, Error, FeaturesPolicy, GpuBuffer,
    GpuBufferAccess, GpuSampler, GpuSamplerBinding, GpuTexture2d, HeadlessDevice, StagedGraph,
    render_wgpu_source,
};

fn device() -> HeadlessDevice {
    HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip")
}

/// A device opened with `FeaturesPolicy::Adapter`: it has `float32-filterable` when the adapter offers it.
fn feature_device() -> HeadlessDevice {
    HeadlessDevice::with_options_and_features(DeviceOptions::default(), FeaturesPolicy::Adapter)
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
    let policy_default = HeadlessDevice::with_options_and_features(
        DeviceOptions::default(),
        FeaturesPolicy::Default,
    )
    .unwrap();
    assert!(!policy_default.float32_filterable());
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
        // Output i samples texel coordinate i / 2 + 0.25 - 0.25 + ...: u = (i + 0.5) / 8, texel space 4u, minus the
        // half-texel centre offset: (i + 0.5) / 2 - 0.5 = i / 2 - 0.25. Clamped at both ends.
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
        for (index, (got, want)) in read.iter().zip(&expected).enumerate() {
            // Hardware filters weigh texels with 8 fractional bits: up to 1/512 of the 40-wide range is 0.08.
            assert!(
                (got - want).abs() < 0.25,
                "output {index}: {got} against {want} (all: {read:?} against {expected:?})"
            );
        }
    } else {
        assert!(
            matches!(result, Err(Error::TextureFilter { .. })),
            "{result:?}"
        );
    }
}

#[test]
fn a_filterable_layout_variant_binds_the_same_texture_as_the_non_filterable_one() {
    // The sampled float texture is `Float { filterable: false }` on the default device and
    // `Float { filterable: true }` on a feature device; the same kernel descriptor serves both.
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
fn a_sampled_uint_texture_loads_and_accepts_any_format_of_its_class() {
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

    // A two-channel texture of the same class is bound to the same parameter: the kernel reads the first channel.
    let pair_texture: GpuTexture2d<Rg32Uint> =
        device.create_texture_2d(5, 3, TextureUse::SAMPLED).unwrap();
    let pairs: Vec<UVec2> = (0..15).map(|i| UVec2::new(7 + i, 900 + i)).collect();
    device.write_texture(&pair_texture, &pairs).unwrap();
    let out: GpuBuffer<u32> =
        device.create_typed_buffer("out", &[0u32; 15], GpuBufferAccess::ReadWrite);
    let bindings = [
        pair_texture.bind(Access::ReadOnly, LengthRule::Independent),
        BufferBinding::read_write(&out),
    ];
    device
        .dispatch_buffers(&LoadUint::DESCRIPTOR, 15, &bindings)
        .unwrap();
    let firsts: Vec<u32> = pairs.iter().map(|pair| pair.x).collect();
    assert_eq!(device.read_typed_buffer(&out).unwrap(), firsts);

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
    assert!(sample.contains("filterable: false"), "{sample}");
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
