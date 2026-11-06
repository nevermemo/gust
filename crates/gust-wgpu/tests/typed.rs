//! Typed calls on the GPU (roadmap M2, m2.c): `gust-wgpu` implements `TypedDevice`, `Run` and
//! `Resource`, runs every `Call` through the existing persistent-dispatch validation against
//! `call.kernel()`, rejects one buffer bound twice when either side is written, and checks binding
//! indices against `max_bindings_per_bind_group` before any wgpu object exists.
//!
//! `semantics::run::call` is the `call` function the macro generates (slice m2.d1; m2.f1 swapped the hand-written
//! form for it). The other kernels are hand-written descriptors (Slang source included) built for one property
//! each, with calls from the small `f32_call!` macro.
//! Expectations come from independent host references, not from running a shader shadow body.

include!("../../../tests/fixtures/semantics.rs");

use gust::{
    Access, Arg, Call, Extent, GpuPod, Independent, Kernel, KernelDescriptor, LengthRule,
    ParameterDescriptor, ParameterKind, Resource, ResourceBinding, ResourceLayout, Run,
};
use gust_wgpu::{
    BufferBinding, BufferDispatch, Error, F32Binding, GpuBuffer, GpuBufferAccess, HeadlessDevice,
    StagedGraph, U32Binding,
};

// ---- hand-written kernels over f32 storage buffers ----

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

const fn descriptor(
    name: &'static str,
    entry_point: &'static str,
    parameters: &'static [ParameterDescriptor],
    slang_source: &'static str,
) -> KernelDescriptor {
    KernelDescriptor::new(
        "typed_tests",
        name,
        entry_point,
        [64, 1, 1],
        parameters,
        slang_source,
    )
}

/// `sum2(a: RO, b: RO, total: RW)`: `total[i] = a[i] + b[i]`.
struct Sum2;
impl Kernel for Sum2 {
    const DESCRIPTOR: KernelDescriptor = descriptor(
        "sum2",
        "typed_sum2",
        &[
            storage("a", Access::ReadOnly, 0),
            storage("b", Access::ReadOnly, 1),
            storage("total", Access::ReadWrite, 2),
        ],
        r#"
[[vk::binding(0, 0)]] StructuredBuffer<float> a;
[[vk::binding(1, 0)]] StructuredBuffer<float> b;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> total;
[shader("compute")] [numthreads(64, 1, 1)]
void typed_sum2(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; total.GetDimensions(n, stride);
    if (id.x < n) { total[id.x] = a[id.x] + b[id.x]; }
}
"#,
    );
}
impl gust::target::GpuKernel for Sum2 {}

/// `add_one(src: RO, dst: RW)`: `dst[i] = src[i] + 1`.
struct AddOne;
impl Kernel for AddOne {
    const DESCRIPTOR: KernelDescriptor = descriptor(
        "add_one",
        "typed_add_one",
        &[
            storage("src", Access::ReadOnly, 0),
            storage("dst", Access::ReadWrite, 1),
        ],
        r#"
[[vk::binding(0, 0)]] StructuredBuffer<float> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dst;
[shader("compute")] [numthreads(64, 1, 1)]
void typed_add_one(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; dst.GetDimensions(n, stride);
    if (id.x < n) { dst[id.x] = src[id.x] + 1.0; }
}
"#,
    );
}
impl gust::target::GpuKernel for AddOne {}

/// `accumulate(acc: RW, src: RO)`: `acc[i] += src[i]` (the read-write parameter comes first).
struct Accumulate;
impl Kernel for Accumulate {
    const DESCRIPTOR: KernelDescriptor = descriptor(
        "accumulate",
        "typed_accumulate",
        &[
            storage("acc", Access::ReadWrite, 0),
            storage("src", Access::ReadOnly, 1),
        ],
        r#"
[[vk::binding(0, 0)]] RWStructuredBuffer<float> acc;
[[vk::binding(1, 0)]] StructuredBuffer<float> src;
[shader("compute")] [numthreads(64, 1, 1)]
void typed_accumulate(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; acc.GetDimensions(n, stride);
    if (id.x < n) { acc[id.x] = acc[id.x] + src[id.x]; }
}
"#,
    );
}
impl gust::target::GpuKernel for Accumulate {}

/// `fill(x: RW, y: RW)`: `x[i] = 1`, `y[i] = 2`.
struct Fill;
impl Kernel for Fill {
    const DESCRIPTOR: KernelDescriptor = descriptor(
        "fill",
        "typed_fill",
        &[
            storage("x", Access::ReadWrite, 0),
            storage("y", Access::ReadWrite, 1),
        ],
        r#"
[[vk::binding(0, 0)]] RWStructuredBuffer<float> x;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> y;
[shader("compute")] [numthreads(64, 1, 1)]
void typed_fill(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; x.GetDimensions(n, stride);
    if (id.x < n) { x[id.x] = 1.0; y[id.x] = 2.0; }
}
"#,
    );
}
impl gust::target::GpuKernel for Fill {}

/// `scale(a: RO, settings: RO, scaled: RW)`: `scaled[i] = a[i] * settings[0]`; `settings` is one element.
struct Scale;
impl Kernel for Scale {
    const DESCRIPTOR: KernelDescriptor = descriptor(
        "scale",
        "typed_scale",
        &[
            storage("a", Access::ReadOnly, 0),
            storage("settings", Access::ReadOnly, 1),
            storage("scaled", Access::ReadWrite, 2),
        ],
        r#"
[[vk::binding(0, 0)]] StructuredBuffer<float> a;
[[vk::binding(1, 0)]] StructuredBuffer<float> settings;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> scaled;
[shader("compute")] [numthreads(64, 1, 1)]
void typed_scale(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; scaled.GetDimensions(n, stride);
    uint m; settings.GetDimensions(m, stride);
    if (id.x < n && m > 0) { scaled[id.x] = a[id.x] * settings[0]; }
}
"#,
    );
}
impl gust::target::GpuKernel for Scale {}

/// One read-write parameter at a sparse binding index: `p[0] = 17`.
macro_rules! sparse_kernel {
    ($kernel:ident, $name:literal, $entry:literal, $binding:literal) => {
        struct $kernel;
        impl Kernel for $kernel {
            const DESCRIPTOR: KernelDescriptor = descriptor(
                $name,
                $entry,
                &[storage("p", Access::ReadWrite, $binding)],
                concat!(
                    "[[vk::binding(",
                    $binding,
                    ", 0)]] RWStructuredBuffer<float> p;\n",
                    "[shader(\"compute\")] [numthreads(64, 1, 1)]\n",
                    "void ",
                    $entry,
                    "(uint3 id : SV_DispatchThreadID) { if (id.x == 0) { p[0] = 17.0; } }\n"
                ),
            );
        }
        impl gust::target::GpuKernel for $kernel {}
    };
}

sparse_kernel!(Sparse3, "sparse3", "typed_sparse3", 3);
sparse_kernel!(Sparse999, "sparse999", "typed_sparse999", 999);
sparse_kernel!(Sparse1000, "sparse1000", "typed_sparse1000", 1000);

/// `sparse1000` over `u32` storage, for the `dispatch_u32` path: `p[0] = 17`.
struct Sparse1000U32;
impl Kernel for Sparse1000U32 {
    const DESCRIPTOR: KernelDescriptor = descriptor(
        "sparse1000_u32",
        "typed_sparse1000_u32",
        &[ParameterDescriptor {
            name: "p",
            rust_type: "uint",
            kind: ParameterKind::Storage,
            access: Access::ReadWrite,
            resource_layout: Some(ResourceLayout::storage_v1(<u32 as GpuPod>::LAYOUT)),
            binding: Some(ResourceBinding {
                group: 0,
                binding: 1000,
            }),
        }],
        "[[vk::binding(1000, 0)]] RWStructuredBuffer<uint> p;\n[shader(\"compute\")] [numthreads(64, 1, 1)]\nvoid typed_sparse1000_u32(uint3 id : SV_DispatchThreadID) { if (id.x == 0) { p[0] = 17u; } }\n",
    );
}
impl gust::target::GpuKernel for Sparse1000U32 {}

/// A generated-form `call` function for a kernel whose parameters are all `f32` storage.
macro_rules! f32_call {
    (
        $call:ident, $kernel:ty, $count:literal,
        $first:ident: $r0:ident = $access0:ident $(, $param:ident: $r:ident = $access:ident)*
    ) => {
        fn $call<'r, $r0 $(, $r)*>(
            __gust_element_count: u32,
            $first: impl Into<Arg<'r, $r0>>
            $(, $param: impl Into<Arg<'r, $r>>)*
        ) -> Call<'r, <$r0 as Resource>::Device, $kernel, $count>
        where
            $r0: Resource<Element = f32> + 'r,
            $($r: Resource<Element = f32, Device = <$r0 as Resource>::Device> + 'r,)*
        {
            Call::new(
                __gust_element_count,
                [
                    $first.into().bind(Access::$access0)
                    $(, $param.into().bind(Access::$access))*
                ],
            )
        }
    };
}

f32_call!(sum2_call, Sum2, 3, a: R0 = ReadOnly, b: R1 = ReadOnly, total: R2 = ReadWrite);
f32_call!(add_one_call, AddOne, 2, src: R0 = ReadOnly, dst: R1 = ReadWrite);
f32_call!(accumulate_call, Accumulate, 2, acc: R0 = ReadWrite, src: R1 = ReadOnly);
f32_call!(fill_call, Fill, 2, x: R0 = ReadWrite, y: R1 = ReadWrite);
f32_call!(scale_call, Scale, 3, a: R0 = ReadOnly, settings: R1 = ReadOnly, scaled: R2 = ReadWrite);
f32_call!(sparse3_call, Sparse3, 1, p: R0 = ReadWrite);
f32_call!(sparse999_call, Sparse999, 1, p: R0 = ReadWrite);
f32_call!(sparse1000_call, Sparse1000, 1, p: R0 = ReadWrite);

// ---- helpers ----

const COUNT: usize = 70;

fn device() -> HeadlessDevice {
    HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip")
}

fn ramp() -> Vec<f32> {
    (0..COUNT).map(|index| index as f32 + 1.0).collect()
}

fn buffer(
    device: &HeadlessDevice,
    label: &str,
    values: &[f32],
    access: GpuBufferAccess,
) -> GpuBuffer<f32> {
    device.create_typed_buffer(label, values, access)
}

fn stats(device: &HeadlessDevice) -> (usize, u64, u64) {
    let stats = device.pipeline_cache_stats();
    (stats.entries, stats.misses, stats.hits)
}

#[track_caller]
fn assert_aliased(error: Error, kernel: &str, first: &str, second: &str) {
    match error {
        Error::AliasedBinding {
            kernel: actual_kernel,
            first: actual_first,
            second: actual_second,
        } => assert_eq!(
            (
                actual_kernel.as_str(),
                actual_first.as_str(),
                actual_second.as_str()
            ),
            (kernel, first, second)
        ),
        other => panic!("expected AliasedBinding, got {other:?}"),
    }
}

#[track_caller]
fn assert_binding_limit(error: Error, kernel: &str, requested: u64, maximum: u64) {
    match error {
        Error::DeviceLimit {
            kernel: actual_kernel,
            limit,
            requested: actual_requested,
            maximum: actual_maximum,
        } => assert_eq!(
            (
                actual_kernel.as_str(),
                limit,
                actual_requested,
                actual_maximum
            ),
            (kernel, "max_bindings_per_bind_group", requested, maximum)
        ),
        other => panic!("expected DeviceLimit for max_bindings_per_bind_group, got {other:?}"),
    }
}

// ---- the typed path reproduces the existing untyped results ----

#[test]
fn typed_call_reproduces_the_semantics_fixture() {
    let device = device();
    for count in [1, 63, 64, 65, 257] {
        let values: Vec<f32> = (0..count).map(|i| (i % 7) as f32 - 3.0).collect();
        let input = buffer(&device, "typed input", &values, GpuBufferAccess::ReadOnly);
        let floats = buffer(
            &device,
            "typed floats",
            &vec![0.0f32; count],
            GpuBufferAccess::ReadWrite,
        );
        let integers = device.create_typed_buffer(
            "typed integers",
            &vec![0u32; count],
            GpuBufferAccess::ReadWrite,
        );
        let call = semantics::run::call(count as u32, &input, &floats, &integers);
        assert_eq!(call.element_count(), count as u32);
        assert_eq!(call.kernel().qualified_name().to_string(), "semantics::run");
        device.dispatch_call(&call).unwrap();
        let actual_floats = device.read_typed_buffer(&floats).unwrap();
        let actual_integers = device.read_typed_buffer(&integers).unwrap();
        for (i, &x) in values.iter().enumerate() {
            let expected = if x > 0.0 {
                x * 2.5 + 1.0
            } else if x < 0.0 {
                -x
            } else {
                7.0
            };
            assert_eq!(actual_floats[i], expected + if i >= 2 { 1.0 } else { 0.0 });
            let i = i as u32;
            let expected = if i.is_multiple_of(2) {
                255 - (i % 256) + 2 + (i / 2) + 1
            } else {
                i + 1024
            };
            assert_eq!(actual_integers[i as usize], expected);
        }
    }
    // One kernel: one compilation, then cache hits for the other four counts.
    assert_eq!(stats(&device), (1, 1, 4));
}

#[test]
fn a_call_is_reusable_and_submit_call_returns_a_job() {
    let device = device();
    let source = buffer(&device, "src", &ramp(), GpuBufferAccess::ReadOnly);
    let target = buffer(
        &device,
        "dst",
        &vec![0.0; COUNT],
        GpuBufferAccess::ReadWrite,
    );
    let call = add_one_call(COUNT as u32, &source, &target);
    for _ in 0..2 {
        device.submit_call(&call).unwrap().wait().unwrap();
    }
    let expected: Vec<f32> = ramp().iter().map(|x| x + 1.0).collect();
    assert_eq!(device.read_typed_buffer(&target).unwrap(), expected);
    assert_eq!(stats(&device), (1, 1, 1));
}

// ---- aliasing (R11, S0 decision A) ----

#[test]
fn one_buffer_bound_twice_read_only_is_accepted() {
    let device = device();
    let values = ramp();
    let input = buffer(&device, "x", &values, GpuBufferAccess::ReadOnly);
    let total = buffer(
        &device,
        "total",
        &vec![0.0; COUNT],
        GpuBufferAccess::ReadWrite,
    );
    let expected: Vec<f32> = values.iter().map(|x| x + x).collect();

    device
        .dispatch_call(&sum2_call(COUNT as u32, &input, &input, &total))
        .unwrap();
    assert_eq!(device.read_typed_buffer(&total).unwrap(), expected);

    // The erased path accepts it as well.
    let total = buffer(
        &device,
        "total again",
        &vec![0.0; COUNT],
        GpuBufferAccess::ReadWrite,
    );
    device
        .dispatch_buffers(
            &Sum2::DESCRIPTOR,
            COUNT as u32,
            &[
                BufferBinding::read_only(&input),
                BufferBinding::read_only(&input),
                BufferBinding::read_write(&total),
            ],
        )
        .unwrap();
    assert_eq!(device.read_typed_buffer(&total).unwrap(), expected);
}

#[test]
fn distinct_buffers_with_equal_contents_are_not_aliases() {
    let device = device();
    let first = buffer(&device, "first", &ramp(), GpuBufferAccess::ReadWrite);
    let second = buffer(&device, "second", &ramp(), GpuBufferAccess::ReadWrite);
    device
        .dispatch_call(&accumulate_call(COUNT as u32, &first, &second))
        .unwrap();
    let expected: Vec<f32> = ramp().iter().map(|x| x + x).collect();
    assert_eq!(device.read_typed_buffer(&first).unwrap(), expected);
}

/// The buffer a rejected call must leave untouched, and the cache it must not have reached.
fn assert_untouched(device: &HeadlessDevice, buffers: &[&GpuBuffer<f32>], expected: &[f32]) {
    assert_eq!(
        stats(device),
        (0, 0, 0),
        "an alias error must precede compilation and the cache lookup"
    );
    for buffer in buffers {
        assert_eq!(device.read_typed_buffer(buffer).unwrap(), expected);
    }
}

#[test]
fn a_read_only_and_a_read_write_binding_of_one_buffer_are_rejected() {
    let device = device();
    let values = ramp();
    let shared = buffer(&device, "shared", &values, GpuBufferAccess::ReadWrite);

    // read-only first, read-write second
    let error = device
        .dispatch_call(&add_one_call(COUNT as u32, &shared, &shared))
        .unwrap_err();
    assert_aliased(error, "typed_tests::add_one", "src", "dst");
    assert_untouched(&device, &[&shared], &values);

    // read-write first, read-only second
    let error = device
        .dispatch_call(&accumulate_call(COUNT as u32, &shared, &shared))
        .unwrap_err();
    assert_aliased(error, "typed_tests::accumulate", "acc", "src");
    assert_untouched(&device, &[&shared], &values);

    // Not adjacent: the third parameter writes the buffer the first one reads.
    let other = buffer(&device, "other", &values, GpuBufferAccess::ReadWrite);
    let error = device
        .dispatch_call(&sum2_call(COUNT as u32, &shared, &other, &shared))
        .unwrap_err();
    assert_aliased(error, "typed_tests::sum2", "a", "total");
    assert_untouched(&device, &[&shared, &other], &values);

    // Through the erased path as well.
    let error = device
        .dispatch_buffers(
            &AddOne::DESCRIPTOR,
            COUNT as u32,
            &[
                BufferBinding::read_only(&shared),
                BufferBinding::read_write(&shared),
            ],
        )
        .unwrap_err();
    assert_aliased(error, "typed_tests::add_one", "src", "dst");
    assert_untouched(&device, &[&shared], &values);
}

#[test]
fn two_read_write_bindings_of_one_buffer_are_rejected() {
    let device = device();
    let values = ramp();
    let shared = buffer(&device, "shared", &values, GpuBufferAccess::ReadWrite);

    let error = device
        .dispatch_call(&fill_call(COUNT as u32, &shared, &shared))
        .unwrap_err();
    assert_aliased(error, "typed_tests::fill", "x", "y");
    assert_untouched(&device, &[&shared], &values);

    let error = device
        .dispatch_buffers(
            &Fill::DESCRIPTOR,
            COUNT as u32,
            &[
                BufferBinding::read_write(&shared),
                BufferBinding::read_write(&shared),
            ],
        )
        .unwrap_err();
    assert_aliased(error, "typed_tests::fill", "x", "y");
    assert_untouched(&device, &[&shared], &values);

    // The same kernel with two buffers still runs.
    let y = buffer(&device, "y", &vec![0.0; COUNT], GpuBufferAccess::ReadWrite);
    device
        .dispatch_call(&fill_call(COUNT as u32, &shared, &y))
        .unwrap();
    assert_eq!(device.read_typed_buffer(&shared).unwrap(), vec![1.0; COUNT]);
    assert_eq!(device.read_typed_buffer(&y).unwrap(), vec![2.0; COUNT]);
}

#[test]
fn an_alias_in_one_dispatch_of_a_batch_rejects_the_whole_batch() {
    let device = device();
    let values = ramp();
    let source = buffer(&device, "src", &values, GpuBufferAccess::ReadOnly);
    let target = buffer(
        &device,
        "dst",
        &vec![0.0; COUNT],
        GpuBufferAccess::ReadWrite,
    );
    let shared = buffer(&device, "shared", &values, GpuBufferAccess::ReadWrite);
    let good = add_one_call(COUNT as u32, &source, &target);
    let bad = fill_call(COUNT as u32, &shared, &shared);
    let error = device
        .dispatch_batch(&[(&good).into(), (&bad).into()])
        .unwrap_err();
    assert_aliased(error, "typed_tests::fill", "x", "y");
    assert_untouched(&device, &[&shared], &values);
    assert_eq!(device.read_typed_buffer(&target).unwrap(), vec![0.0; COUNT]);
}

// ---- batches ----

#[test]
fn typed_calls_of_different_kernels_share_one_batch() {
    // Two calls of different kernels over disjoint outputs: one compilation per kernel.
    let device = device();
    let values = ramp();
    let source = buffer(&device, "src", &values, GpuBufferAccess::ReadOnly);
    let plus_one = buffer(
        &device,
        "plus one",
        &vec![0.0; COUNT],
        GpuBufferAccess::ReadWrite,
    );
    let total = buffer(
        &device,
        "total",
        &vec![0.0; COUNT],
        GpuBufferAccess::ReadWrite,
    );
    let first = add_one_call(COUNT as u32, &source, &plus_one);
    let second = sum2_call(COUNT as u32, &source, &source, &total);
    let batch: [BufferDispatch<'_>; 2] = [(&first).into(), (&second).into()];
    device.dispatch_batch(&batch).unwrap();
    assert_eq!(stats(&device), (2, 2, 0));
    let plus_one_values: Vec<f32> = values.iter().map(|x| x + 1.0).collect();
    assert_eq!(
        device.read_typed_buffer(&plus_one).unwrap(),
        plus_one_values
    );
    let doubled: Vec<f32> = values.iter().map(|x| x + x).collect();
    assert_eq!(device.read_typed_buffer(&total).unwrap(), doubled);
    device.dispatch_batch(&batch).unwrap();
    assert_eq!(stats(&device), (2, 2, 2));
}

// ---- the length rule (D15) ----

#[test]
fn independent_lets_a_one_element_buffer_through_where_the_plain_argument_fails() {
    let device = device();
    let values = ramp();
    let input = buffer(&device, "a", &values, GpuBufferAccess::ReadOnly);
    let settings = buffer(&device, "settings", &[3.0], GpuBufferAccess::ReadOnly);
    let scaled = buffer(
        &device,
        "scaled",
        &vec![0.0; COUNT],
        GpuBufferAccess::ReadWrite,
    );

    let error = device
        .dispatch_call(&scale_call(COUNT as u32, &input, &settings, &scaled))
        .unwrap_err();
    match error {
        Error::BufferLength {
            parameter,
            expected,
            actual,
        } => assert_eq!(
            (parameter.as_str(), expected, actual),
            ("settings", COUNT, 1)
        ),
        other => panic!("expected BufferLength, got {other:?}"),
    }
    assert_eq!(stats(&device), (0, 0, 0));

    device
        .dispatch_call(&scale_call(
            COUNT as u32,
            &input,
            Independent(&settings),
            &scaled,
        ))
        .unwrap();
    let expected: Vec<f32> = values.iter().map(|x| x * 3.0).collect();
    assert_eq!(device.read_typed_buffer(&scaled).unwrap(), expected);
}

#[test]
fn an_independent_empty_buffer_is_rejected_when_the_dispatch_has_work() {
    let device = device();
    let input = buffer(&device, "a", &ramp(), GpuBufferAccess::ReadOnly);
    let settings = buffer(&device, "settings", &[], GpuBufferAccess::ReadOnly);
    let scaled = buffer(
        &device,
        "scaled",
        &vec![0.0; COUNT],
        GpuBufferAccess::ReadWrite,
    );
    let error = device
        .dispatch_call(&scale_call(
            COUNT as u32,
            &input,
            Independent(&settings),
            &scaled,
        ))
        .unwrap_err();
    match error {
        Error::IndependentLengthEmpty { parameter } => assert_eq!(parameter, "settings"),
        other => panic!("expected IndependentLengthEmpty, got {other:?}"),
    }
}

#[test]
fn bind_maps_the_length_rule_onto_the_binding() {
    // `bind` is public: the rule it records decides whether a one-element buffer passes.
    let device = device();
    let input = buffer(&device, "a", &ramp(), GpuBufferAccess::ReadOnly);
    let settings = buffer(&device, "settings", &[2.0], GpuBufferAccess::ReadOnly);
    let scaled = buffer(
        &device,
        "scaled",
        &vec![0.0; COUNT],
        GpuBufferAccess::ReadWrite,
    );
    let call = |rule: LengthRule| {
        Call::<HeadlessDevice, Scale, 3>::new(
            COUNT as u32,
            [
                input.bind(Access::ReadOnly, LengthRule::Shared),
                settings.bind(Access::ReadOnly, rule),
                scaled.bind(Access::ReadWrite, LengthRule::Shared),
            ],
        )
    };
    assert!(matches!(
        device.dispatch_call(&call(LengthRule::Shared)),
        Err(Error::BufferLength { .. })
    ));
    device
        .dispatch_call(&call(LengthRule::Independent))
        .unwrap();
    let expected: Vec<f32> = ramp().iter().map(|x| x * 2.0).collect();
    assert_eq!(device.read_typed_buffer(&scaled).unwrap(), expected);
}

// ---- a typed call is checked against the descriptor at submit ----

#[test]
fn a_read_only_buffer_where_the_call_binds_read_write_is_a_runtime_error() {
    // Access is not part of the buffer type, so this compiles; the backend refuses it at submit.
    let device = device();
    let values = vec![1.0f32; 4];
    let input = buffer(&device, "values", &values, GpuBufferAccess::ReadOnly);
    let floats = buffer(&device, "floats", &values, GpuBufferAccess::ReadOnly);
    let integers = device.create_typed_buffer("integers", &[0u32; 4], GpuBufferAccess::ReadWrite);
    let error = device
        .dispatch_call(&semantics::run::call(4, &input, &floats, &integers))
        .unwrap_err();
    match error {
        Error::PersistentBufferAccess {
            parameter,
            buffer,
            binding,
        } => {
            assert_eq!(parameter, "floats");
            assert_eq!(buffer, GpuBufferAccess::ReadOnly);
            assert_eq!(binding, Access::ReadWrite);
        }
        other => panic!("expected PersistentBufferAccess, got {other:?}"),
    }
    assert_eq!(stats(&device), (0, 0, 0));
}

#[test]
fn hand_built_calls_are_checked_against_the_descriptor_at_submit() {
    let device = device();
    let values = ramp();
    let a = buffer(&device, "a", &values, GpuBufferAccess::ReadWrite);
    let b = buffer(&device, "b", &values, GpuBufferAccess::ReadWrite);
    let total = buffer(
        &device,
        "total",
        &vec![0.0; COUNT],
        GpuBufferAccess::ReadWrite,
    );
    let shared = LengthRule::Shared;
    let n = COUNT as u32;

    // Wrong access, written side: `total` is read-write in the descriptor, bound read-only.
    let call = Call::<HeadlessDevice, Sum2, 3>::new(
        n,
        [
            a.bind(Access::ReadOnly, shared),
            b.bind(Access::ReadOnly, shared),
            total.bind(Access::ReadOnly, shared),
        ],
    );
    match device.dispatch_call(&call).unwrap_err() {
        Error::BindingAccess {
            parameter,
            expected,
            actual,
        } => assert_eq!(
            (parameter.as_str(), expected, actual),
            ("total", Access::ReadWrite, Access::ReadOnly)
        ),
        other => panic!("expected BindingAccess, got {other:?}"),
    }

    // Wrong access, read side: `a` is read-only in the descriptor, bound read-write.
    let call = Call::<HeadlessDevice, Sum2, 3>::new(
        n,
        [
            a.bind(Access::ReadWrite, shared),
            b.bind(Access::ReadOnly, shared),
            total.bind(Access::ReadWrite, shared),
        ],
    );
    match device.dispatch_call(&call).unwrap_err() {
        Error::BindingAccess {
            parameter,
            expected,
            actual,
        } => assert_eq!(
            (parameter.as_str(), expected, actual),
            ("a", Access::ReadOnly, Access::ReadWrite)
        ),
        other => panic!("expected BindingAccess, got {other:?}"),
    }

    // Wrong element: a `u32` buffer for an `f32` parameter.
    let integers =
        device.create_typed_buffer("integers", &vec![0u32; COUNT], GpuBufferAccess::ReadWrite);
    let call = Call::<HeadlessDevice, Sum2, 3>::new(
        n,
        [
            a.bind(Access::ReadOnly, shared),
            b.bind(Access::ReadOnly, shared),
            integers.bind(Access::ReadWrite, shared),
        ],
    );
    match device.dispatch_call(&call).unwrap_err() {
        Error::UnsupportedResourceLayout { parameter, .. } => assert_eq!(parameter, "total"),
        other => panic!("expected UnsupportedResourceLayout, got {other:?}"),
    }

    // Nothing above reached the pipeline cache or wrote anything.
    assert_eq!(stats(&device), (0, 0, 0));
    assert_eq!(device.read_typed_buffer(&total).unwrap(), vec![0.0; COUNT]);
}

#[test]
fn a_buffer_of_another_device_instance_is_rejected_although_the_type_matches() {
    let first = device();
    let second = device();
    let values = ramp();
    let source = buffer(&first, "src", &values, GpuBufferAccess::ReadOnly);
    let foreign = buffer(
        &second,
        "dst",
        &vec![0.0; COUNT],
        GpuBufferAccess::ReadWrite,
    );
    // Both are `GpuBuffer<f32>`: the type system cannot tell the devices apart.
    let call = add_one_call(COUNT as u32, &source, &foreign);
    assert!(matches!(
        first.dispatch_call(&call),
        Err(Error::ForeignPersistentBuffer)
    ));
    assert_eq!(stats(&first), (0, 0, 0));
    assert_eq!(
        second.read_typed_buffer(&foreign).unwrap(),
        vec![0.0; COUNT]
    );
}

// ---- binding indices against max_bindings_per_bind_group (gap from the m2.b review) ----

#[test]
fn a_binding_index_at_the_limit_is_an_error_before_compilation() {
    let device = device();
    let maximum = u64::from(device.limits().max_bindings_per_bind_group);
    assert_eq!(
        maximum, 1000,
        "the sparse kernels assume wgpu's default limit"
    );
    let target = buffer(&device, "p", &[3.0], GpuBufferAccess::ReadWrite);

    let error = device
        .dispatch_call(&sparse1000_call(1, &target))
        .unwrap_err();
    assert_binding_limit(error, "typed_tests::sparse1000", 1001, maximum);
    assert_eq!(stats(&device), (0, 0, 0));

    let error = device
        .dispatch_buffers(
            &Sparse1000::DESCRIPTOR,
            1,
            &[BufferBinding::read_write(&target)],
        )
        .unwrap_err();
    assert_binding_limit(error, "typed_tests::sparse1000", 1001, maximum);
    assert_eq!(stats(&device), (0, 0, 0));
    assert_eq!(device.read_typed_buffer(&target).unwrap(), [3.0]);
}

#[test]
fn the_highest_allowed_binding_index_runs() {
    let device = device();
    let target = buffer(&device, "p", &[3.0], GpuBufferAccess::ReadWrite);
    device.dispatch_call(&sparse999_call(1, &target)).unwrap();
    assert_eq!(device.read_typed_buffer(&target).unwrap(), [17.0]);
}

#[test]
fn lowered_binding_limits_apply_to_the_binding_index_not_the_binding_count() {
    let device = device();
    let mut limits = device.limits().clone();
    limits.max_bindings_per_bind_group = 3;
    let device = device.with_limits(limits).expect("lowering is allowed");
    let values = ramp();
    let a = buffer(&device, "a", &values, GpuBufferAccess::ReadOnly);
    let b = buffer(&device, "b", &values, GpuBufferAccess::ReadOnly);
    let total = buffer(
        &device,
        "total",
        &vec![0.0; COUNT],
        GpuBufferAccess::ReadWrite,
    );
    // Indices 0, 1 and 2 against a limit of 3: allowed.
    device
        .dispatch_call(&sum2_call(COUNT as u32, &a, &b, &total))
        .unwrap();
    let expected: Vec<f32> = values.iter().map(|x| x + x).collect();
    assert_eq!(device.read_typed_buffer(&total).unwrap(), expected);
    let compiled = stats(&device);

    // One parameter at index 3: the count is one, the index is not below the limit.
    let target = buffer(&device, "p", &[3.0], GpuBufferAccess::ReadWrite);
    let error = device.dispatch_call(&sparse3_call(1, &target)).unwrap_err();
    assert_binding_limit(error, "typed_tests::sparse3", 4, 3);
    assert_eq!(stats(&device), compiled, "no compilation or cache lookup");
    assert_eq!(device.read_typed_buffer(&target).unwrap(), [3.0]);

    // An empty dispatch does no GPU work and is not limit checked, as for the other limits.
    let empty = buffer(&device, "empty", &[], GpuBufferAccess::ReadWrite);
    device.dispatch_call(&sparse3_call(0, &empty)).unwrap();
}

#[test]
fn the_binding_index_limit_applies_to_the_slice_paths_too() {
    let device = device();
    let maximum = u64::from(device.limits().max_bindings_per_bind_group);
    let error = device
        .dispatch_f32(&Sparse1000::DESCRIPTOR, 1, &[F32Binding::ReadWrite(&[3.0])])
        .unwrap_err();
    assert_binding_limit(error, "typed_tests::sparse1000", 1001, maximum);
    assert_eq!(stats(&device), (0, 0, 0));

    let error = device
        .dispatch_u32(
            &Sparse1000U32::DESCRIPTOR,
            1,
            &[U32Binding::ReadWrite(&[3])],
        )
        .unwrap_err();
    assert_binding_limit(error, "typed_tests::sparse1000_u32", 1001, maximum);
    assert_eq!(stats(&device), (0, 0, 0));
}

// ---- aliasing through a staged graph ----

#[test]
fn a_graph_dispatch_node_that_aliases_a_written_buffer_is_rejected() {
    let device = device();
    let values = ramp();
    let shared = buffer(&device, "shared", &values, GpuBufferAccess::ReadWrite);
    let other = buffer(
        &device,
        "other",
        &vec![0.0; COUNT],
        GpuBufferAccess::ReadWrite,
    );

    // Control: the same kernel with two buffers executes in a graph.
    let fine = add_one_call(COUNT as u32, &shared, &other);
    let mut graph = StagedGraph::new();
    let _ = graph.dispatch((&fine).into(), &[]);
    assert!(device.execute_graph(&graph).is_ok());

    // One node binds the same buffer read-only and read-write.
    let read_write_alias = add_one_call(COUNT as u32, &shared, &shared);
    let mut graph = StagedGraph::new();
    let _ = graph.dispatch((&read_write_alias).into(), &[]);
    let error = device
        .execute_graph(&graph)
        .err()
        .expect("alias must be rejected");
    assert_aliased(error, "typed_tests::add_one", "src", "dst");

    // One node binds the same buffer read-write twice.
    let write_write_alias = fill_call(COUNT as u32, &shared, &shared);
    let mut graph = StagedGraph::new();
    let _ = graph.dispatch((&write_write_alias).into(), &[]);
    let error = device
        .execute_graph(&graph)
        .err()
        .expect("alias must be rejected");
    assert_aliased(error, "typed_tests::fill", "x", "y");
    assert_eq!(device.read_typed_buffer(&shared).unwrap(), values);
}

// ---- uniforms (m6.u1) ----
//
// A uniform parameter is bound to a `GpuUniform`: one `Std140V1` value in a constant buffer. The macro generates the
// `call` for a `Uniform<T>` parameter (m6.u2; `uniforms.rs` dispatches such kernels), but the kernels below are
// hand-written for one property each, with a `call` function in the shape the macro generates (`Kind =
// gust::kind::Storage` / `gust::kind::Uniform` bounds).

use gust::math::{Mat2, Vec4};

const fn uniform(
    name: &'static str,
    rust_type: &'static str,
    element: gust::TypeLayout,
    binding: u32,
) -> ParameterDescriptor {
    ParameterDescriptor {
        name,
        rust_type,
        kind: ParameterKind::Uniform,
        access: Access::ReadOnly,
        resource_layout: Some(ResourceLayout::std140_v1(element)),
        binding: Some(ResourceBinding { group: 0, binding }),
    }
}

/// `shade(src: RO, factor: Uniform<f32>, dst: RW)`: `dst[i] = src[i] * factor`.
struct Shade;
impl Kernel for Shade {
    const DESCRIPTOR: KernelDescriptor = descriptor(
        "shade",
        "typed_shade",
        &[
            storage("src", Access::ReadOnly, 0),
            uniform("factor", "float", <f32 as GpuPod>::LAYOUT, 1),
            storage("dst", Access::ReadWrite, 2),
        ],
        r#"
[[vk::binding(0, 0)]] StructuredBuffer<float> src;
[[vk::binding(1, 0)]] ConstantBuffer<float> factor;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> dst;
[shader("compute")] [numthreads(64, 1, 1)]
void typed_shade(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; dst.GetDimensions(n, stride);
    if (id.x < n) { dst[id.x] = src[id.x] * factor; }
}
"#,
    );
}
impl gust::target::GpuKernel for Shade {}

/// A padding-free `Std140V1` struct: a `Vec4` then four `f32`, size 32, alignment 16.
#[derive(Clone, Copy)]
#[repr(C)]
struct Packed {
    v: Vec4,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
}

// SAFETY: `repr(C)`, a `Vec4` (16 bytes, align 16) then four `f32` at 16, 20, 24 and 28: no padding, size 32.
unsafe impl GpuPod for Packed {
    const LAYOUT: gust::TypeLayout = gust::TypeLayout::structure(
        "Packed",
        32,
        16,
        &[
            gust::StructFieldLayout {
                name: "v",
                offset: 0,
                ty: <Vec4 as GpuPod>::LAYOUT,
            },
            gust::StructFieldLayout {
                name: "a",
                offset: 16,
                ty: <f32 as GpuPod>::LAYOUT,
            },
            gust::StructFieldLayout {
                name: "b",
                offset: 20,
                ty: <f32 as GpuPod>::LAYOUT,
            },
            gust::StructFieldLayout {
                name: "c",
                offset: 24,
                ty: <f32 as GpuPod>::LAYOUT,
            },
            gust::StructFieldLayout {
                name: "d",
                offset: 28,
                ty: <f32 as GpuPod>::LAYOUT,
            },
        ],
    );
}

/// A `Std140V1` struct with an honest layout and Rust size that is nevertheless padded: a `Vec4` then one `f32`, so
/// bytes 20 to 31 are padding that Rust leaves uninitialised.
#[derive(Clone, Copy)]
#[repr(C)]
struct Padded {
    v: Vec4,
    a: f32,
}

// SAFETY: deliberately breaks the no-padding rule, to show that `create_uniform` refuses it; it is never uploaded.
unsafe impl GpuPod for Padded {
    const LAYOUT: gust::TypeLayout = gust::TypeLayout::structure(
        "Padded",
        32,
        16,
        &[
            gust::StructFieldLayout {
                name: "v",
                offset: 0,
                ty: <Vec4 as GpuPod>::LAYOUT,
            },
            gust::StructFieldLayout {
                name: "a",
                offset: 16,
                ty: <f32 as GpuPod>::LAYOUT,
            },
        ],
    );
}

/// `unpack(p: Uniform<Packed>, dst: RW)`: `dst[0..8] = [p.v.x, p.v.y, p.v.z, p.v.w, p.a, p.b, p.c, p.d]`.
struct Unpack;
impl Kernel for Unpack {
    const DESCRIPTOR: KernelDescriptor = descriptor(
        "unpack",
        "typed_unpack",
        &[
            uniform("p", "Packed", <Packed as GpuPod>::LAYOUT, 0),
            storage("dst", Access::ReadWrite, 1),
        ],
        r#"
struct Packed { float4 v; float a; float b; float c; float d; };
[[vk::binding(0, 0)]] ConstantBuffer<Packed> p;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dst;
[shader("compute")] [numthreads(64, 1, 1)]
void typed_unpack(uint3 id : SV_DispatchThreadID)
{
    float all[8] = { p.v.x, p.v.y, p.v.z, p.v.w, p.a, p.b, p.c, p.d };
    uint n; uint stride; dst.GetDimensions(n, stride);
    if (id.x < n && id.x < 8) { dst[id.x] = all[id.x]; }
}
"#,
    );
}
impl gust::target::GpuKernel for Unpack {}

/// The `Shade` kernel with its uniform labelled `StorageV2`: valid for the element but not for a constant buffer.
struct ShadeStorageRule;
impl Kernel for ShadeStorageRule {
    const DESCRIPTOR: KernelDescriptor = descriptor(
        "shade_storage_rule",
        "typed_shade",
        &[
            storage("src", Access::ReadOnly, 0),
            ParameterDescriptor {
                resource_layout: Some(ResourceLayout::storage_v2(<Vec4 as GpuPod>::LAYOUT)),
                ..uniform("factor", "float4", <Vec4 as GpuPod>::LAYOUT, 1)
            },
            storage("dst", Access::ReadWrite, 2),
        ],
        Shade::DESCRIPTOR.slang_source,
    );
}
impl gust::target::GpuKernel for ShadeStorageRule {}

fn shade_call<'r, R0, R1, R2>(
    __gust_element_count: u32,
    src: impl Into<Arg<'r, R0>>,
    factor: impl Into<Arg<'r, R1>>,
    dst: impl Into<Arg<'r, R2>>,
) -> Call<'r, <R0 as Resource>::Device, Shade, 3>
where
    R0: Resource<Element = f32, Kind = gust::kind::Storage> + 'r,
    R1: Resource<Element = f32, Device = <R0 as Resource>::Device, Kind = gust::kind::Uniform> + 'r,
    R2: Resource<Element = f32, Device = <R0 as Resource>::Device, Kind = gust::kind::Storage> + 'r,
{
    Call::new(
        __gust_element_count,
        [
            src.into().bind(Access::ReadOnly),
            factor.into().bind(Access::ReadOnly),
            dst.into().bind(Access::ReadWrite),
        ],
    )
}

fn unpack_call<'r, R0, R1>(
    __gust_element_count: u32,
    p: impl Into<Arg<'r, R0>>,
    dst: impl Into<Arg<'r, R1>>,
) -> Call<'r, <R0 as Resource>::Device, Unpack, 2>
where
    R0: Resource<Element = Packed, Kind = gust::kind::Uniform> + 'r,
    R1: Resource<Element = f32, Device = <R0 as Resource>::Device, Kind = gust::kind::Storage> + 'r,
{
    Call::new(
        __gust_element_count,
        [
            p.into().bind(Access::ReadOnly),
            dst.into().bind(Access::ReadWrite),
        ],
    )
}

fn zeros(device: &HeadlessDevice, label: &str, count: usize) -> GpuBuffer<f32> {
    buffer(device, label, &vec![0.0; count], GpuBufferAccess::ReadWrite)
}

#[test]
fn a_uniform_is_read_by_a_kernel() {
    let device = device();
    let src = buffer(&device, "src", &ramp(), GpuBufferAccess::ReadOnly);
    let dst = zeros(&device, "dst", COUNT);
    let factor = device.create_uniform("factor", &2.5f32).unwrap();
    // The uniform holds one value while the dispatch covers `COUNT` elements: it is `Independent` by construction.
    let call = shade_call(COUNT as u32, &src, &factor, &dst);
    assert_eq!(call.bindings().len(), 3);
    device.dispatch_call(&call).unwrap();
    let expected: Vec<f32> = ramp().iter().map(|x| x * 2.5).collect();
    assert_eq!(device.read_typed_buffer(&dst).unwrap(), expected);
}

#[test]
fn a_uniform_written_between_two_calls_is_seen_by_the_second() {
    let device = device();
    let src = buffer(&device, "src", &ramp(), GpuBufferAccess::ReadOnly);
    let first = zeros(&device, "first", COUNT);
    let second = zeros(&device, "second", COUNT);
    let factor = device.create_uniform("factor", &2.0f32).unwrap();
    device
        .dispatch_call(&shade_call(COUNT as u32, &src, &factor, &first))
        .unwrap();
    device.write_uniform(&factor, &-3.0f32).unwrap();
    device
        .dispatch_call(&shade_call(COUNT as u32, &src, &factor, &second))
        .unwrap();
    let doubled: Vec<f32> = ramp().iter().map(|x| x * 2.0).collect();
    let negated: Vec<f32> = ramp().iter().map(|x| x * -3.0).collect();
    assert_eq!(device.read_typed_buffer(&first).unwrap(), doubled);
    assert_eq!(device.read_typed_buffer(&second).unwrap(), negated);
}

#[test]
fn a_struct_uniform_keeps_its_std140_offsets() {
    let device = device();
    let packed = Packed {
        v: Vec4::new(1.0, 2.0, 3.0, 4.0),
        a: 5.0,
        b: 6.0,
        c: 7.0,
        d: 8.0,
    };
    let uniform = device.create_uniform("packed", &packed).unwrap();
    assert_eq!(uniform.layout(), <Packed as GpuPod>::LAYOUT);
    // Eight elements against a uniform of length one.
    let dst = zeros(&device, "dst", 8);
    device
        .dispatch_call(&unpack_call(8, &uniform, &dst))
        .unwrap();
    assert_eq!(
        device.read_typed_buffer(&dst).unwrap(),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
}

#[test]
fn a_padded_or_non_std140_type_is_refused_with_the_fix_named() {
    let device = device();
    let padded = Padded {
        v: Vec4::splat(1.0),
        a: 2.0,
    };
    match device.create_uniform("padded", &padded) {
        Err(Error::InvalidUniformType { type_name, reason }) => {
            assert_eq!(type_name, "Padded");
            assert!(reason.contains("padding bytes"), "{reason}");
            assert!(reason.contains("explicit padding fields"), "{reason}");
        }
        Err(other) => panic!("expected InvalidUniformType, got {other:?}"),
        Ok(_) => panic!("a padded type must be refused"),
    }
    // A 2x2 matrix is a storage type but has no constant-buffer layout.
    match device.create_uniform(
        "matrix",
        &Mat2::from_cols(
            gust::math::Vec2::new(1.0, 0.0),
            gust::math::Vec2::new(0.0, 1.0),
        ),
    ) {
        Err(Error::InvalidUniformType { type_name, reason }) => {
            assert_eq!(type_name, Mat2::LAYOUT.name);
            assert!(reason.contains("not Std140V1"), "{reason}");
        }
        Err(other) => panic!("expected InvalidUniformType, got {other:?}"),
        Ok(_) => panic!("a 2x2 matrix must be refused"),
    }
}

#[test]
fn a_resource_of_the_other_kind_is_refused_before_anything_else() {
    let device = device();
    let src = buffer(&device, "src", &ramp(), GpuBufferAccess::ReadOnly);
    let dst = zeros(&device, "dst", COUNT);
    let factor = device.create_uniform("factor", &2.0f32).unwrap();
    let vector =
        device.create_typed_buffer("vector", &[Vec4::splat(1.0)], GpuBufferAccess::ReadOnly);

    // A storage buffer in the uniform slot. `Call::new` takes the bindings as they are: the typed `call` would not
    // compile, so a hand-built call is what reaches the runtime check.
    let storage_for_uniform: Call<'_, HeadlessDevice, Shade, 3> = Call::new(
        COUNT as u32,
        [
            src.bind(Access::ReadOnly, LengthRule::Shared),
            vector.bind(Access::ReadOnly, LengthRule::Independent),
            dst.bind(Access::ReadWrite, LengthRule::Shared),
        ],
    );
    match device.dispatch_call(&storage_for_uniform).unwrap_err() {
        Error::BindingKind {
            parameter,
            expected,
            actual,
        } => assert_eq!(
            (parameter.as_str(), expected, actual),
            (
                "factor",
                gust::BindingKind::Uniform,
                gust::BindingKind::Storage
            )
        ),
        other => panic!("expected BindingKind, got {other:?}"),
    }

    // A uniform in the storage slot: the kind is checked before the layout (a `f32` uniform would also differ in the
    // rule it presents, `Std140V1` against `StorageV1`).
    let uniform_for_storage: Call<'_, HeadlessDevice, Shade, 3> = Call::new(
        COUNT as u32,
        [
            factor.bind(Access::ReadOnly, LengthRule::Independent),
            factor.bind(Access::ReadOnly, LengthRule::Independent),
            dst.bind(Access::ReadWrite, LengthRule::Shared),
        ],
    );
    match device.dispatch_call(&uniform_for_storage).unwrap_err() {
        Error::BindingKind {
            parameter,
            expected,
            actual,
        } => assert_eq!(
            (parameter.as_str(), expected, actual),
            (
                "src",
                gust::BindingKind::Storage,
                gust::BindingKind::Uniform
            )
        ),
        other => panic!("expected BindingKind, got {other:?}"),
    }
    assert_eq!(stats(&device), (0, 0, 0), "refused before any compilation");

    // A uniform is read-only: a read-write binding of it is an access error, however the call was built.
    let writable: Call<'_, HeadlessDevice, Shade, 3> = Call::new(
        COUNT as u32,
        [
            src.bind(Access::ReadOnly, LengthRule::Shared),
            factor.bind(Access::ReadWrite, LengthRule::Independent),
            dst.bind(Access::ReadWrite, LengthRule::Shared),
        ],
    );
    match device.dispatch_call(&writable).unwrap_err() {
        Error::BindingAccess {
            parameter,
            expected,
            actual,
        } => assert_eq!(
            (parameter.as_str(), expected, actual),
            ("factor", Access::ReadOnly, Access::ReadWrite)
        ),
        other => panic!("expected BindingAccess, got {other:?}"),
    }
    assert_eq!(stats(&device), (0, 0, 0));
}

#[test]
fn a_std140_layout_mismatch_is_refused() {
    let device = device();
    let src = buffer(&device, "src", &ramp(), GpuBufferAccess::ReadOnly);
    let dst = zeros(&device, "dst", COUNT);

    // The kernel declares `Uniform<f32>` and the uniform holds a `Vec4`: same kind, different element.
    let vector = device.create_uniform("vector", &Vec4::splat(1.0)).unwrap();
    let wrong_element: Call<'_, HeadlessDevice, Shade, 3> = Call::new(
        COUNT as u32,
        [
            src.bind(Access::ReadOnly, LengthRule::Shared),
            vector.bind(Access::ReadOnly, LengthRule::Independent),
            dst.bind(Access::ReadWrite, LengthRule::Shared),
        ],
    );
    match device.dispatch_call(&wrong_element).unwrap_err() {
        Error::UnsupportedResourceLayout {
            parameter,
            expected,
            actual,
        } => {
            assert_eq!(parameter, "factor");
            assert_eq!(
                *expected,
                ResourceLayout::std140_v1(<f32 as GpuPod>::LAYOUT)
            );
            assert_eq!(
                actual.as_deref(),
                Some(&ResourceLayout::std140_v1(<Vec4 as GpuPod>::LAYOUT))
            );
        }
        other => panic!("expected UnsupportedResourceLayout, got {other:?}"),
    }

    // The descriptor itself claims the storage rule for a uniform.
    let factor = device.create_uniform("factor", &Vec4::splat(1.0)).unwrap();
    let error = device
        .dispatch_buffers(
            &ShadeStorageRule::DESCRIPTOR,
            COUNT as u32,
            &[
                BufferBinding::read_only(&src),
                factor.bind(Access::ReadOnly, LengthRule::Independent),
                BufferBinding::read_write(&dst),
            ],
        )
        .unwrap_err();
    match error {
        Error::InvalidResourceLayout { parameter, .. } => assert_eq!(parameter, "factor"),
        other => panic!("expected InvalidResourceLayout, got {other:?}"),
    }
    assert_eq!(stats(&device), (0, 0, 0), "refused before any compilation");
}

#[test]
fn a_uniform_of_another_device_is_refused() {
    let device = device();
    let other = HeadlessDevice::new().expect("a second device");
    let src = buffer(&device, "src", &ramp(), GpuBufferAccess::ReadOnly);
    let dst = zeros(&device, "dst", COUNT);
    let foreign = other.create_uniform("foreign", &2.0f32).unwrap();
    let error = device
        .dispatch_call(&shade_call(COUNT as u32, &src, &foreign, &dst))
        .unwrap_err();
    assert!(matches!(error, Error::ForeignPersistentBuffer), "{error:?}");
    assert!(matches!(
        device.write_uniform(&foreign, &1.0f32),
        Err(Error::ForeignPersistentBuffer)
    ));
}

#[test]
fn the_uniform_launch_limits_apply_before_any_compilation() {
    // The uniform count: a kernel with one uniform on a device that allows none.
    let device = device();
    let mut limits = device.limits().clone();
    limits.max_uniform_buffers_per_shader_stage = 0;
    let device = device.with_limits(limits).unwrap();
    let src = buffer(&device, "src", &ramp(), GpuBufferAccess::ReadOnly);
    let dst = zeros(&device, "dst", COUNT);
    let factor = device.create_uniform("factor", &2.0f32).unwrap();
    match device
        .dispatch_call(&shade_call(COUNT as u32, &src, &factor, &dst))
        .unwrap_err()
    {
        Error::DeviceLimit {
            kernel,
            limit,
            requested,
            maximum,
        } => assert_eq!(
            (kernel.as_str(), limit, requested, maximum),
            (
                "typed_tests::shade",
                "max_uniform_buffers_per_shader_stage",
                1,
                0
            )
        ),
        other => panic!("expected DeviceLimit, got {other:?}"),
    }
    assert_eq!(stats(&device), (0, 0, 0));

    // The binding size: the 32-byte `Packed` against a limit of 16.
    let device = self::device();
    let mut limits = device.limits().clone();
    limits.max_uniform_buffer_binding_size = 16;
    let device = device.with_limits(limits).unwrap();
    let packed = device
        .create_uniform(
            "packed",
            &Packed {
                v: Vec4::splat(0.0),
                a: 0.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
            },
        )
        .unwrap();
    let dst = zeros(&device, "dst", 8);
    match device
        .dispatch_call(&unpack_call(8, &packed, &dst))
        .unwrap_err()
    {
        Error::DeviceLimit {
            kernel,
            limit,
            requested,
            maximum,
        } => assert_eq!(
            (kernel.as_str(), limit, requested, maximum),
            (
                "typed_tests::unpack",
                "max_uniform_buffer_binding_size",
                32,
                16
            )
        ),
        other => panic!("expected DeviceLimit, got {other:?}"),
    }
    assert_eq!(stats(&device), (0, 0, 0));
}

// ---- the extent forms of a typed call (m7.d) ----

#[test]
fn a_linear_extent_call_is_the_one_dimensional_call() {
    let device = device();
    let values: Vec<f32> = (0..100).map(|i| (i % 7) as f32 - 3.0).collect();
    let input = buffer(&device, "extent input", &values, GpuBufferAccess::ReadOnly);
    let outputs = |label: &str| {
        (
            buffer(
                &device,
                label,
                &vec![0.0; values.len()],
                GpuBufferAccess::ReadWrite,
            ),
            device.create_typed_buffer(
                label,
                &vec![0u32; values.len()],
                GpuBufferAccess::ReadWrite,
            ),
        )
    };
    let (floats, integers) = outputs("one-dimensional");
    let (extent_floats, extent_integers) = outputs("extent");
    let count = values.len() as u32;
    let linear = semantics::run::call(count, &input, &floats, &integers);
    let extent = semantics::run::call_extent(
        Extent::linear(count),
        &input,
        &extent_floats,
        &extent_integers,
    );
    assert_eq!(linear.extent(), Extent::new(count, 1, 1));
    assert_eq!(extent.extent(), linear.extent());
    assert_eq!(extent.element_count(), count);
    device.dispatch_call(&linear).unwrap();
    device.dispatch_call(&extent).unwrap();
    assert_eq!(
        device.read_typed_buffer(&extent_floats).unwrap(),
        device.read_typed_buffer(&floats).unwrap()
    );
    assert_eq!(
        device.read_typed_buffer(&extent_integers).unwrap(),
        device.read_typed_buffer(&integers).unwrap()
    );
    // The other two forms of the extent family bind the same way.
    let (submit_floats, submit_integers) = outputs("submit_extent");
    semantics::run::submit_extent(
        &device,
        Extent::linear(count),
        &input,
        &submit_floats,
        &submit_integers,
    )
    .unwrap()
    .wait()
    .unwrap();
    let (dispatch_floats, dispatch_integers) = outputs("dispatch_extent");
    semantics::run::dispatch_extent(
        &device,
        Extent::linear(count),
        &input,
        &dispatch_floats,
        &dispatch_integers,
    )
    .unwrap();
    for (floats_, integers_) in [
        (&submit_floats, &submit_integers),
        (&dispatch_floats, &dispatch_integers),
    ] {
        assert_eq!(
            device.read_typed_buffer(floats_).unwrap(),
            device.read_typed_buffer(&floats).unwrap()
        );
        assert_eq!(
            device.read_typed_buffer(integers_).unwrap(),
            device.read_typed_buffer(&integers).unwrap()
        );
    }
}

#[test]
fn a_shared_buffer_holds_the_product_of_the_extent_and_that_product_does_not_wrap() {
    let device = device();
    let input = buffer(
        &device,
        "shared input",
        &[1.0; 12],
        GpuBufferAccess::ReadOnly,
    );
    let floats = |len: usize| {
        buffer(
            &device,
            "shared floats",
            &vec![0.0; len],
            GpuBufferAccess::ReadWrite,
        )
    };
    let integers = |len: usize| {
        device.create_typed_buffer(
            "shared integers",
            &vec![0u32; len],
            GpuBufferAccess::ReadWrite,
        )
    };
    // 4 x 3 x 1 is twelve elements: twelve is accepted, eleven and thirteen are not, each naming the length.
    let extent = Extent::new(4, 3, 1);
    let (floats_ok, integers_ok) = (floats(12), integers(12));
    device
        .dispatch_call(&semantics::run::call_extent(
            extent,
            &input,
            &floats_ok,
            &integers_ok,
        ))
        .unwrap();
    for wrong in [11usize, 13] {
        let (floats_wrong, integers_wrong) = (floats(wrong), integers(wrong));
        let error = device
            .dispatch_call(&semantics::run::call_extent(
                extent,
                &input,
                &floats_wrong,
                &integers_wrong,
            ))
            .unwrap_err();
        match error {
            Error::BufferLength {
                expected, actual, ..
            } => assert_eq!((expected, actual), (12, wrong)),
            other => panic!("expected BufferLength, got {other:?}"),
        }
    }
    // 65536 x 65536 is 2^32 elements, not the 0 a product computed in 32 bits would be. The first buffer (twelve
    // elements) is the one reported.
    let wrapping = Extent::new(65_536, 65_536, 1);
    let (floats_one, integers_one) = (floats(1), integers(1));
    match device
        .dispatch_call(&semantics::run::call_extent(
            wrapping,
            &input,
            &floats_one,
            &integers_one,
        ))
        .unwrap_err()
    {
        Error::BufferLength {
            expected, actual, ..
        } => assert_eq!((expected, actual), (1 << 32, 12)),
        other => panic!("expected BufferLength for the 32-bit wrap, got {other:?}"),
    }
}
