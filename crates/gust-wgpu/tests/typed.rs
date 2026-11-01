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
    Access, Arg, Call, GpuPod, Independent, Kernel, KernelDescriptor, LengthRule,
    ParameterDescriptor, ParameterKind, Resource, ResourceBinding, ResourceLayout, Run,
    SourceOrigin,
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
    KernelDescriptor {
        module: "typed_tests",
        name,
        entry_point,
        workgroup_size: [64, 1, 1],
        parameters,
        slang_source,
        source: SourceOrigin::UNKNOWN,
        artifacts: &[],
    }
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
