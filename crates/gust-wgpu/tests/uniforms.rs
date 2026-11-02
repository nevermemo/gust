//! Uniforms on the GPU (roadmap M6, slice m6.g2; Gpu tier).
//!
//! Every kernel below is a `#[gpu]` kernel with a `Uniform<T>` parameter, dispatched through the typed `call` the macro
//! generates. Each runs three ways: on the GPU (`HeadlessDevice`, the WGSL route; the native reflection helper checks
//! the descriptor first, D17, now with a `constantBuffer` binding), as the CPU twin (the same typed call through
//! `CpuDevice`, D22), and against an independent host reference written in this file with plain `f64` arithmetic (it
//! does not call `gust::math`). The twin is a second assertion and never GPU evidence. Every input is a small integer
//! or a binary fraction and every operation is a product or a sum of such values, so each result is exactly
//! representable and the comparisons are exact (no tolerance is involved).
//!
//! GUST never inserts padding, so a uniform struct spells out what the constant-buffer layout needs: `Gain` is one
//! `f32` followed by three explicit padding fields (a struct holding only one `f32` is GUST-E0209), and `Tint`, `Xform`
//! are one-field structs of a `Vec4` and a `Mat4` (a bare `Uniform<Vec4>` is GUST-E0206).

include!("../../../tests/fixtures/uniforms.rs");

use std::mem::{offset_of, size_of};

use gust::cpu::CpuDevice;
use gust::math::{Mat4, Vec2, Vec4};
use gust::reflect::{ReflectedResourceKind, ReflectedTypeKind};
use gust::slang::Target;
use gust::{KernelDescriptor, ParameterKind, Run};
use gust_wgpu::{GpuBuffer, GpuBufferAccess, HeadlessDevice, StagedGraph};

/// The element count of every dispatch: not a multiple of the workgroup width (64), so the guard matters.
const N: usize = 70;

#[gust::gpu]
mod probes {
    // `gain` at 0 and three explicit padding fields: 16 bytes, no implicit padding.
    pub struct Gain {
        pub gain: f32,
        pub pad0: f32,
        pub pad1: f32,
        pub pad2: f32,
    }

    // One field: a vector.
    pub struct Tint {
        pub color: Vec4,
    }

    // One field: a matrix.
    pub struct Xform {
        pub m: Mat4,
    }

    // `v` 0, `x` 8, `y` 12: 16 bytes.
    pub struct Inner {
        pub v: Vec2,
        pub x: f32,
        pub y: f32,
    }

    // `a`..`d` 0..16, `inner` 16..32 (a nested struct starts at a multiple of 16): 32 bytes, no padding.
    pub struct Outer {
        pub a: f32,
        pub b: f32,
        pub c: f32,
        pub d: f32,
        pub inner: Inner,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn scale(
        id: SV_DispatchThreadID,
        g: Uniform<Gain>,
        src: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<f32>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = src[i] * g.gain;
        }
    }

    // The same `Gain` as `scale`, read by a second kernel.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn offset(
        id: SV_DispatchThreadID,
        g: Uniform<Gain>,
        src: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<f32>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = src[i] + g.gain;
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn tinted(
        id: SV_DispatchThreadID,
        t: Uniform<Tint>,
        src: StructuredBuffer<Vec4>,
        mut out: RWStructuredBuffer<Vec4>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = src[i] * t.color;
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn transform(
        id: SV_DispatchThreadID,
        x: Uniform<Xform>,
        src: StructuredBuffer<Vec4>,
        mut out: RWStructuredBuffer<Vec4>,
    ) {
        let i = id.x;
        if i < out.len() {
            out[i] = x.m.mul_vec4(src[i]);
        }
    }

    // Every field of a nested struct, one slot each (eight slots; the dispatch covers eight elements).
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn fields(id: SV_DispatchThreadID, u: Uniform<Outer>, mut out: RWStructuredBuffer<f32>) {
        let i = id.x;
        if i == 0 {
            out[0] = u.a;
            out[1] = u.b;
            out[2] = u.c;
            out[3] = u.d;
            out[4] = u.inner.v.x;
            out[5] = u.inner.v.y;
            out[6] = u.inner.x;
            out[7] = u.inner.y;
        }
    }
}

use probes::{Gain, Inner, Outer, Tint, Xform};
use uniforms::{Frame, Light};

fn device() -> HeadlessDevice {
    HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip")
}

fn gain(value: f32) -> Gain {
    Gain {
        gain: value,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

fn input(device: &HeadlessDevice, label: &str, values: &[f32]) -> GpuBuffer<f32> {
    device.create_typed_buffer(label, values, GpuBufferAccess::ReadOnly)
}

/// An output buffer filled with NaN, so a slot the kernel does not write fails every comparison.
fn nan_floats(device: &HeadlessDevice, label: &str, count: usize) -> GpuBuffer<f32> {
    device.create_typed_buffer(label, &vec![f32::NAN; count], GpuBufferAccess::ReadWrite)
}

fn nan_vectors(device: &HeadlessDevice, label: &str, count: usize) -> GpuBuffer<Vec4> {
    device.create_typed_buffer(
        label,
        &vec![Vec4::splat(f32::NAN); count],
        GpuBufferAccess::ReadWrite,
    )
}

fn ramp() -> Vec<f32> {
    (0..N).map(|i| i as f32 + 1.0).collect()
}

/// Integer-valued points in [-3, 3]; the first three are the golden's hand-worked points.
fn points() -> Vec<[f32; 4]> {
    let mut points = vec![
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for i in 3..N {
        points.push([
            (i % 7) as f32 - 3.0,
            (i % 5) as f32 - 2.0,
            (i % 3) as f32 - 1.0,
            (i % 4) as f32 - 1.0,
        ]);
    }
    points
}

fn vec4s(values: &[[f32; 4]]) -> Vec<Vec4> {
    values
        .iter()
        .map(|v| Vec4::new(v[0], v[1], v[2], v[3]))
        .collect()
}

fn quads(values: &[Vec4]) -> Vec<[f32; 4]> {
    values.iter().map(|v| [v.x, v.y, v.z, v.w]).collect()
}

/// Independent host references (`f64`; no `gust::math`).
mod host {
    /// `rows * p` for a matrix given by its rows: `result[r] = sum over c of rows[r][c] * p[c]`.
    pub fn mat_vec(rows: &[[f32; 4]; 4], p: &[f32; 4]) -> [f64; 4] {
        let mut out = [0.0; 4];
        for r in 0..4 {
            for c in 0..4 {
                out[r] += f64::from(rows[r][c]) * f64::from(p[c]);
            }
        }
        out
    }

    /// The fields of the golden's `Frame`, as plain arrays.
    pub struct Frame {
        pub view: [[f32; 4]; 4],
        pub tint: [f32; 4],
        pub color: [f32; 4],
        pub intensity: f32,
        pub falloff: f32,
        pub offset: [f32; 2],
        pub scale: f32,
        pub count: u32,
    }

    /// The golden kernel: `lit = (view * p + tint) * intensity + color * falloff`, then
    /// `(lit.x + offset.x, lit.y + offset.y, lit.z * scale, lit.w + count)`.
    pub fn golden(frame: &Frame, p: &[f32; 4]) -> [f32; 4] {
        let moved = mat_vec(&frame.view, p);
        let mut lit = [0.0f64; 4];
        for k in 0..4 {
            lit[k] = (moved[k] + f64::from(frame.tint[k])) * f64::from(frame.intensity)
                + f64::from(frame.color[k]) * f64::from(frame.falloff);
        }
        [
            (lit[0] + f64::from(frame.offset[0])) as f32,
            (lit[1] + f64::from(frame.offset[1])) as f32,
            (lit[2] * f64::from(frame.scale)) as f32,
            (lit[3] + f64::from(frame.count)) as f32,
        ]
    }
}

fn golden_spec() -> host::Frame {
    host::Frame {
        // Rows (1 2 0 0), (0 1 0 3), (0 0 2 0), (1 0 0 1).
        view: [
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 2.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        tint: [1.0, 0.0, 2.0, 1.0],
        color: [1.0, 0.0, 0.0, 2.0],
        intensity: 2.0,
        falloff: 3.0,
        offset: [1.0, 2.0],
        scale: 2.0,
        count: 3,
    }
}

fn golden_frame(spec: &host::Frame) -> Frame {
    let row = |r: usize| {
        Vec4::new(
            spec.view[r][0],
            spec.view[r][1],
            spec.view[r][2],
            spec.view[r][3],
        )
    };
    Frame {
        view: Mat4::from_rows(row(0), row(1), row(2), row(3)),
        tint: Vec4::new(spec.tint[0], spec.tint[1], spec.tint[2], spec.tint[3]),
        light: Light {
            color: Vec4::new(spec.color[0], spec.color[1], spec.color[2], spec.color[3]),
            intensity: spec.intensity,
            falloff: spec.falloff,
            pad0: 0.0,
            pad1: 0.0,
        },
        offset: Vec2::new(spec.offset[0], spec.offset[1]),
        scale: spec.scale,
        count: spec.count,
    }
}

/// The golden kernel on the GPU, with the uniform already created.
fn run_golden_gpu(
    device: &HeadlessDevice,
    frame: &gust_wgpu::GpuUniform<Frame>,
    points: &[[f32; 4]],
) -> Vec<[f32; 4]> {
    let src = device.create_typed_buffer("points", &vec4s(points), GpuBufferAccess::ReadOnly);
    let out = nan_vectors(device, "out", points.len());
    let call = uniforms::run::call(points.len() as u32, frame, &src, &out);
    device.dispatch_call(&call).unwrap();
    quads(&device.read_typed_buffer(&out).unwrap())
}

fn run_golden_twin(
    cpu: &CpuDevice,
    frame: &gust::cpu::CpuUniform<Frame>,
    points: &[[f32; 4]],
) -> Vec<[f32; 4]> {
    let src = cpu.buffer(&vec4s(points));
    let out = cpu.buffer(&vec![Vec4::splat(f32::NAN); points.len()]);
    uniforms::run::dispatch(cpu, points.len() as u32, frame, &src, &out).unwrap();
    quads(&out.read())
}

#[test]
fn the_golden_uniform_kernel_matches_the_host_reference_and_the_twin() {
    let device = device();
    eprintln!("uniforms adapter: {:?}", device.adapter_info());
    let spec = golden_spec();
    let points = points();
    let expected: Vec<[f32; 4]> = points.iter().map(|p| host::golden(&spec, p)).collect();
    // The hand-worked values of `crates/gust/tests/uniforms_twin.rs` pin the host reference itself.
    assert_eq!(
        expected[..3],
        [
            [16.0, 30.0, 32.0, 21.0],
            [10.0, 4.0, 8.0, 11.0],
            [6.0, 8.0, 8.0, 13.0]
        ]
    );

    let frame = golden_frame(&spec);
    let uniform = device.create_uniform("frame", &frame).unwrap();
    assert_eq!(run_golden_gpu(&device, &uniform, &points), expected);

    let cpu = CpuDevice::new();
    let twin_uniform = cpu.uniform(golden_frame(&spec)).unwrap();
    assert_eq!(run_golden_twin(&cpu, &twin_uniform, &points), expected);

    // A second value through `write_uniform` (outside a graph): the next dispatch sees it, on both.
    let mut changed = golden_spec();
    changed.scale = 1.0;
    changed.count = 0;
    changed.offset = [-1.0, 0.0];
    changed.view[0] = [0.0, 0.0, 1.0, 0.0];
    let expected: Vec<[f32; 4]> = points.iter().map(|p| host::golden(&changed, p)).collect();
    device
        .write_uniform(&uniform, &golden_frame(&changed))
        .unwrap();
    assert_eq!(run_golden_gpu(&device, &uniform, &points), expected);
    twin_uniform.write(golden_frame(&changed)).unwrap();
    assert_eq!(run_golden_twin(&cpu, &twin_uniform, &points), expected);
}

#[test]
fn a_one_field_scalar_struct_uniform_scales_a_buffer() {
    let device = device();
    let src = ramp();
    let expected: Vec<f32> = src
        .iter()
        .map(|x| (f64::from(*x) * f64::from(2.5f32)) as f32)
        .collect();

    let (gpu_src, out) = (input(&device, "src", &src), nan_floats(&device, "out", N));
    let uniform = device.create_uniform("gain", &gain(2.5)).unwrap();
    let call = probes::scale::call(N as u32, &uniform, &gpu_src, &out);
    device.dispatch_call(&call).unwrap();
    assert_eq!(device.read_typed_buffer(&out).unwrap(), expected);

    let cpu = CpuDevice::new();
    let twin_uniform = cpu.uniform(gain(2.5)).unwrap();
    let twin_out = cpu.buffer(&vec![f32::NAN; N]);
    probes::scale::dispatch(&cpu, N as u32, &twin_uniform, &cpu.buffer(&src), &twin_out).unwrap();
    assert_eq!(twin_out.read(), expected);
}

#[test]
fn a_vec4_field_uniform_multiplies_component_wise() {
    let device = device();
    let src = points();
    let color = [2.0f32, -1.0, 0.5, 4.0];
    let expected: Vec<[f32; 4]> = src
        .iter()
        .map(|p| {
            let mut out = [0.0f32; 4];
            for k in 0..4 {
                out[k] = (f64::from(p[k]) * f64::from(color[k])) as f32;
            }
            out
        })
        .collect();
    let tint = Tint {
        color: Vec4::new(color[0], color[1], color[2], color[3]),
    };

    let gpu_src = device.create_typed_buffer("src", &vec4s(&src), GpuBufferAccess::ReadOnly);
    let out = nan_vectors(&device, "out", N);
    let uniform = device.create_uniform("tint", &tint).unwrap();
    let call = probes::tinted::call(N as u32, &uniform, &gpu_src, &out);
    device.dispatch_call(&call).unwrap();
    assert_eq!(quads(&device.read_typed_buffer(&out).unwrap()), expected);

    let cpu = CpuDevice::new();
    let twin_out = cpu.buffer(&vec![Vec4::splat(f32::NAN); N]);
    probes::tinted::dispatch(
        &cpu,
        N as u32,
        &cpu.uniform(tint).unwrap(),
        &cpu.buffer(&vec4s(&src)),
        &twin_out,
    )
    .unwrap();
    assert_eq!(quads(&twin_out.read()), expected);
}

#[test]
fn a_mat4_field_uniform_transforms_vectors_with_the_row_convention() {
    let device = device();
    // Not symmetric, so a transposed read of the constant buffer gives different numbers.
    let rows = [
        [1.0f32, 2.0, 0.0, -1.0],
        [0.0, 1.0, 3.0, 0.5],
        [4.0, 0.0, 2.0, 1.0],
        [0.0, -2.0, 1.0, 1.0],
    ];
    let src = points();
    let expected: Vec<[f32; 4]> = src
        .iter()
        .map(|p| host::mat_vec(&rows, p).map(|x| x as f32))
        .collect();
    let row = |r: usize| Vec4::new(rows[r][0], rows[r][1], rows[r][2], rows[r][3]);
    let xform = Xform {
        m: Mat4::from_rows(row(0), row(1), row(2), row(3)),
    };

    let gpu_src = device.create_typed_buffer("src", &vec4s(&src), GpuBufferAccess::ReadOnly);
    let out = nan_vectors(&device, "out", N);
    let uniform = device.create_uniform("xform", &xform).unwrap();
    let call = probes::transform::call(N as u32, &uniform, &gpu_src, &out);
    device.dispatch_call(&call).unwrap();
    assert_eq!(quads(&device.read_typed_buffer(&out).unwrap()), expected);

    let cpu = CpuDevice::new();
    let twin_out = cpu.buffer(&vec![Vec4::splat(f32::NAN); N]);
    probes::transform::dispatch(
        &cpu,
        N as u32,
        &cpu.uniform(xform).unwrap(),
        &cpu.buffer(&vec4s(&src)),
        &twin_out,
    )
    .unwrap();
    assert_eq!(quads(&twin_out.read()), expected);
}

#[test]
fn a_nested_struct_uniform_is_read_field_by_field() {
    let device = device();
    // Eight distinct values, so a swap of any two fields (or a wrong offset) changes the result.
    let outer = Outer {
        a: 1.0,
        b: 2.0,
        c: 3.0,
        d: 4.0,
        inner: Inner {
            v: Vec2::new(5.0, 6.0),
            x: 7.0,
            y: 8.0,
        },
    };
    let expected = [
        outer.a,
        outer.b,
        outer.c,
        outer.d,
        outer.inner.v.x,
        outer.inner.v.y,
        outer.inner.x,
        outer.inner.y,
    ];
    assert_eq!(expected, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let out = nan_floats(&device, "out", 8);
    let uniform = device.create_uniform("outer", &outer).unwrap();
    let call = probes::fields::call(8, &uniform, &out);
    device.dispatch_call(&call).unwrap();
    assert_eq!(device.read_typed_buffer(&out).unwrap(), expected);

    let cpu = CpuDevice::new();
    let twin_out = cpu.buffer(&[f32::NAN; 8]);
    probes::fields::dispatch(&cpu, 8, &cpu.uniform(outer).unwrap(), &twin_out).unwrap();
    assert_eq!(twin_out.read(), expected);
}

#[test]
fn one_uniform_bound_to_two_kernels_and_changed_on_the_host_without_a_write() {
    let device = device();
    let src = ramp();
    let gpu_src = input(&device, "src", &src);
    let (scaled, shifted) = (
        nan_floats(&device, "scaled", N),
        nan_floats(&device, "shifted", N),
    );

    // The same `GpuUniform<Gain>` as the uniform of two kernels.
    let mut host_value = gain(2.5);
    let uniform = device.create_uniform("gain", &host_value).unwrap();
    // Changing the host value after `create_uniform` is not a write: the buffer holds the copy made at creation, and
    // only `write_uniform` (or a graph's) changes it. This is the documented behaviour.
    host_value.gain = 99.0;
    let (scale_call, offset_call) = (
        probes::scale::call(N as u32, &uniform, &gpu_src, &scaled),
        probes::offset::call(N as u32, &uniform, &gpu_src, &shifted),
    );
    device.dispatch_call(&scale_call).unwrap();
    device.dispatch_call(&offset_call).unwrap();
    let times: Vec<f32> = src.iter().map(|x| (f64::from(*x) * 2.5) as f32).collect();
    let plus: Vec<f32> = src.iter().map(|x| (f64::from(*x) + 2.5) as f32).collect();
    assert_eq!(device.read_typed_buffer(&scaled).unwrap(), times);
    assert_eq!(device.read_typed_buffer(&shifted).unwrap(), plus);
    assert_eq!(host_value.gain, 99.0);

    // Both kernels in one graph on the same uniform, no write: both read 2.5.
    let (scaled_again, shifted_again) = (
        nan_floats(&device, "scaled2", N),
        nan_floats(&device, "shifted2", N),
    );
    let (scale_again, offset_again) = (
        probes::scale::call(N as u32, &uniform, &gpu_src, &scaled_again),
        probes::offset::call(N as u32, &uniform, &gpu_src, &shifted_again),
    );
    let mut graph = StagedGraph::new();
    graph.run(&scale_again);
    graph.run(&offset_again);
    device.execute_graph(&graph).unwrap();
    assert_eq!(device.read_typed_buffer(&scaled_again).unwrap(), times);
    assert_eq!(device.read_typed_buffer(&shifted_again).unwrap(), plus);

    // The twin: one `CpuUniform` for both kernels.
    let cpu = CpuDevice::new();
    let twin_uniform = cpu.uniform(gain(2.5)).unwrap();
    let (twin_src, twin_scaled, twin_shifted) = (
        cpu.buffer(&src),
        cpu.buffer(&vec![f32::NAN; N]),
        cpu.buffer(&vec![f32::NAN; N]),
    );
    probes::scale::dispatch(&cpu, N as u32, &twin_uniform, &twin_src, &twin_scaled).unwrap();
    probes::offset::dispatch(&cpu, N as u32, &twin_uniform, &twin_src, &twin_shifted).unwrap();
    assert_eq!(twin_scaled.read(), times);
    assert_eq!(twin_shifted.read(), plus);
}

#[test]
fn a_staged_graph_update_between_two_dispatches_is_seen_by_the_second() {
    let device = device();
    let src = ramp();
    let gpu_src = input(&device, "src", &src);
    let (first, second, third) = (
        nan_floats(&device, "first", N),
        nan_floats(&device, "second", N),
        nan_floats(&device, "third", N),
    );
    let uniform = device.create_uniform("gain", &gain(2.5)).unwrap();
    let next = gain(-4.0);
    let (call_one, call_two, call_three) = (
        probes::scale::call(N as u32, &uniform, &gpu_src, &first),
        probes::scale::call(N as u32, &uniform, &gpu_src, &second),
        probes::offset::call(N as u32, &uniform, &gpu_src, &third),
    );

    let mut graph = StagedGraph::new();
    let one = graph.run(&call_one);
    let update = graph.write_uniform(&uniform, &next);
    let two = graph.run(&call_two);
    let three = graph.run(&call_three);
    // The write waits for the dispatch that read the old value; the later dispatches wait for the write.
    assert_eq!(graph.dependency_indices(one), Vec::<usize>::new());
    assert_eq!(graph.dependency_indices(update), [0]);
    assert_eq!(graph.dependency_indices(two), [1]);
    assert_eq!(graph.dependency_indices(three), [1]);
    let output = device.execute_graph(&graph).unwrap();
    assert_eq!(output.report.dispatches, 3);
    // 16 bytes: the whole `Gain`.
    assert_eq!(output.report.upload_bytes, 16);

    let times = |k: f32| -> Vec<f32> {
        src.iter()
            .map(|x| (f64::from(*x) * f64::from(k)) as f32)
            .collect()
    };
    // One command buffer, three dispatches: each saw the value its position implies.
    assert_eq!(device.read_typed_buffer(&first).unwrap(), times(2.5));
    assert_eq!(device.read_typed_buffer(&second).unwrap(), times(-4.0));
    let plus: Vec<f32> = src.iter().map(|x| x - 4.0).collect();
    assert_eq!(device.read_typed_buffer(&third).unwrap(), plus);

    // The uniform keeps the last value for a later submission.
    let again = nan_floats(&device, "again", N);
    let call_again = probes::scale::call(N as u32, &uniform, &gpu_src, &again);
    device.dispatch_call(&call_again).unwrap();
    assert_eq!(device.read_typed_buffer(&again).unwrap(), times(-4.0));

    // The twin of the same sequence: a write between two dispatches.
    let cpu = CpuDevice::new();
    let twin_uniform = cpu.uniform(gain(2.5)).unwrap();
    let (twin_src, twin_first, twin_second) = (
        cpu.buffer(&src),
        cpu.buffer(&vec![f32::NAN; N]),
        cpu.buffer(&vec![f32::NAN; N]),
    );
    probes::scale::dispatch(&cpu, N as u32, &twin_uniform, &twin_src, &twin_first).unwrap();
    twin_uniform.write(gain(-4.0)).unwrap();
    probes::scale::dispatch(&cpu, N as u32, &twin_uniform, &twin_src, &twin_second).unwrap();
    assert_eq!(twin_first.read(), times(2.5));
    assert_eq!(twin_second.read(), times(-4.0));
}

/// D17 with a `constantBuffer` binding: the helper reflects each uniform of the dispatched kernels as a constant
/// buffer whose size is Rust's `size_of` of the element, and `cross_check_kernel` accepts the descriptor.
fn constant_buffer_size(descriptor: &KernelDescriptor, name: &str, rust_size: usize) -> u32 {
    let compiled = gust::reflect::compile_reflected(descriptor, Target::Wgsl)
        .unwrap_or_else(|e| panic!("{}: {e}", descriptor.name));
    assert_eq!(compiled.reflection.schema_version, 2, "{}", descriptor.name);
    let resource = compiled
        .reflection
        .resources
        .iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("{}: no resource `{name}`", descriptor.name));
    assert_eq!(resource.kind, ReflectedResourceKind::ConstantBuffer);
    assert!(matches!(
        resource.element.kind,
        ReflectedTypeKind::Struct(_)
    ));
    let size = resource.element_stride.expect("a constant buffer size");
    assert_eq!(
        (size as usize, resource.element.size.map(|s| s as usize)),
        (rust_size, Some(rust_size)),
        "{}: `{name}`",
        descriptor.name
    );
    // Every parameter of the descriptor is checked against the document, the uniform among them.
    let reports = compiled
        .reflection
        .cross_check_kernel(descriptor)
        .unwrap_or_else(|e| panic!("{}: {e}", descriptor.name));
    assert!(
        reports.iter().all(|r| r.is_full_abi_match()),
        "{}: {reports:?}",
        descriptor.name
    );
    assert!(
        descriptor
            .parameters
            .iter()
            .any(|p| p.name == name && p.kind == ParameterKind::Uniform)
    );
    size
}

#[test]
fn the_dispatched_kernels_pass_d17_with_constant_buffer_bindings() {
    // The golden: 128 bytes, and every field where Rust has it.
    let compiled =
        gust::reflect::compile_reflected(&uniforms::run::DESCRIPTOR, Target::Wgsl).unwrap();
    let frame = compiled
        .reflection
        .resources
        .iter()
        .find(|r| r.name == "frame")
        .unwrap();
    let ReflectedTypeKind::Struct(fields) = &frame.element.kind else {
        panic!("the Frame element is not a struct");
    };
    let offsets: Vec<(&str, u32)> = fields.iter().map(|f| (f.name.as_str(), f.offset)).collect();
    assert_eq!(
        offsets,
        [
            ("view", offset_of!(Frame, view) as u32),
            ("tint", offset_of!(Frame, tint) as u32),
            ("light", offset_of!(Frame, light) as u32),
            ("offset", offset_of!(Frame, offset) as u32),
            ("scale", offset_of!(Frame, scale) as u32),
            ("count", offset_of!(Frame, count) as u32),
        ]
    );
    assert_eq!(
        offsets.iter().map(|(_, o)| *o).collect::<Vec<_>>(),
        [0, 64, 80, 112, 120, 124]
    );
    let sizes = [
        (
            "uniforms::run frame",
            constant_buffer_size(&uniforms::run::DESCRIPTOR, "frame", size_of::<Frame>()),
        ),
        (
            "probes::scale g",
            constant_buffer_size(&probes::scale::DESCRIPTOR, "g", size_of::<Gain>()),
        ),
        (
            "probes::offset g",
            constant_buffer_size(&probes::offset::DESCRIPTOR, "g", size_of::<Gain>()),
        ),
        (
            "probes::tinted t",
            constant_buffer_size(&probes::tinted::DESCRIPTOR, "t", size_of::<Tint>()),
        ),
        (
            "probes::transform x",
            constant_buffer_size(&probes::transform::DESCRIPTOR, "x", size_of::<Xform>()),
        ),
        (
            "probes::fields u",
            constant_buffer_size(&probes::fields::DESCRIPTOR, "u", size_of::<Outer>()),
        ),
    ];
    for (name, size) in sizes {
        eprintln!("constantBuffer {name}: {size} bytes");
    }
    assert_eq!(sizes.map(|(_, size)| size), [128, 16, 16, 16, 64, 32]);
}
