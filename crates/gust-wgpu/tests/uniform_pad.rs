//! `#[pad]` uniforms on the GPU (slice m14.pad, D233 and D239; Gpu tier).
//!
//! Three structs are marked `#[pad]`, so GUST inserts the padding of their constant-buffer layouts: `Gain` is one `f32`
//! (a tail of 12 bytes), `Mixed` has a gap before a `Vec2` and a tail, and `Outer` holds a nested struct (a gap before
//! the 16-byte start of `inner`, and a tail after `b`). Each is built with the generated `new`, which takes the user
//! fields only. The first test compares the reflection record of each uniform with the layout of the padded Rust
//! struct (offsets, names and size, `_padN` included); the second uploads each struct and reads every user field back
//! from a kernel on the GPU, with an exact comparison (every value is a small integer). Run on the adapter `device()`
//! finds; the CPU twin is a second assertion and never GPU evidence.

mod common;

use common::device;

use std::mem::size_of;

use gust::cpu::CpuDevice;
use gust::math::Vec2;
use gust::reflect::{ReflectedResourceKind, ReflectedTypeKind};
use gust::slang::Target;
use gust::{GpuPod, KernelDescriptor, Run, TypeLayoutKind};
use gust_wgpu::{GpuBuffer, GpuBufferAccess, HeadlessDevice};

#[gust::gpu]
mod padded {
    // `gain` at 0, then 12 bytes of padding (`_pad0` to `_pad2`): 16 bytes.
    #[pad]
    pub struct Gain {
        pub gain: f32,
    }

    // `a` 0, `_pad0` 4, `v` 8, `b` 16, then 12 bytes of padding: 32 bytes.
    #[pad]
    pub struct Mixed {
        pub a: f32,
        pub v: Vec2,
        pub b: f32,
    }

    // Four `f32`: a whole row, nothing to insert, and still a constructor.
    #[pad]
    pub struct Inner {
        pub x: f32,
        pub y: f32,
        pub z: f32,
        pub w: f32,
    }

    // `a` 0, `_pad0` to `_pad2` 4..16, `inner` 16..32, `b` 32..40, `_pad3` and `_pad4` 40..48: 48 bytes.
    #[pad]
    pub struct Outer {
        pub a: f32,
        pub inner: Inner,
        pub b: Vec2,
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn read_gain(id: SV_DispatchThreadID, u: Uniform<Gain>, mut out: RWStructuredBuffer<f32>) {
        if id.x == 0 {
            out[0] = u.gain;
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn read_mixed(
        id: SV_DispatchThreadID,
        u: Uniform<Mixed>,
        mut out: RWStructuredBuffer<f32>,
    ) {
        if id.x == 0 {
            out[0] = u.a;
            out[1] = u.v.x;
            out[2] = u.v.y;
            out[3] = u.b;
        }
    }

    #[kernel(workgroup_size(64, 1, 1))]
    pub fn read_outer(
        id: SV_DispatchThreadID,
        u: Uniform<Outer>,
        mut out: RWStructuredBuffer<f32>,
    ) {
        if id.x == 0 {
            out[0] = u.a;
            out[1] = u.inner.x;
            out[2] = u.inner.y;
            out[3] = u.inner.z;
            out[4] = u.inner.w;
            out[5] = u.b.x;
            out[6] = u.b.y;
        }
    }
}

use padded::{Gain, Inner, Mixed, Outer};

fn nan_floats(device: &HeadlessDevice, count: usize) -> GpuBuffer<f32> {
    device.create_typed_buffer("out", &vec![f32::NAN; count], GpuBufferAccess::ReadWrite)
}

/// The field names and byte offsets of the struct layout of `T` (the macro's `GpuPod` impl, which reads `offset_of!`
/// of the padded Rust struct).
fn rust_fields<T: GpuPod>() -> Vec<(String, u32)> {
    let TypeLayoutKind::Struct(fields) = T::LAYOUT.kind else {
        panic!("{} is not a struct layout", T::LAYOUT.name);
    };
    fields
        .iter()
        .map(|field| (field.name.to_owned(), field.offset))
        .collect()
}

/// The names and offsets of the uniform `u` of `descriptor` in the reflection record, and the record's size for it.
fn reflected(descriptor: &KernelDescriptor) -> (Vec<(String, u32)>, u32) {
    let compiled = gust::reflect::compile_reflected(descriptor, Target::Wgsl)
        .unwrap_or_else(|e| panic!("{}: {e}", descriptor.name));
    let resource = compiled
        .reflection
        .resources
        .iter()
        .find(|r| r.name == "u")
        .unwrap_or_else(|| panic!("{}: no resource `u`", descriptor.name));
    assert_eq!(resource.kind, ReflectedResourceKind::ConstantBuffer);
    let ReflectedTypeKind::Struct(fields) = &resource.element.kind else {
        panic!("{}: the element is not a struct", descriptor.name);
    };
    let fields = fields.iter().map(|f| (f.name.clone(), f.offset)).collect();
    (
        fields,
        resource.element_stride.expect("a constant buffer size"),
    )
}

#[test]
fn a_pad_uniform_has_in_the_reflection_record_the_layout_of_its_padded_rust_struct() {
    fn check<T: GpuPod>(descriptor: &KernelDescriptor, expected: &[(&str, u32)], size: usize) {
        let (reflected, reflected_size) = reflected(descriptor);
        let expected: Vec<(String, u32)> = expected
            .iter()
            .map(|(name, offset)| ((*name).to_owned(), *offset))
            .collect();
        // The hand-written expectation, the macro's record of the Rust struct, and Slang's reflection all agree.
        assert_eq!(
            rust_fields::<T>(),
            expected,
            "{}: Rust layout",
            descriptor.name
        );
        assert_eq!(
            reflected, expected,
            "{}: reflection record",
            descriptor.name
        );
        assert_eq!(size_of::<T>(), size, "{}: size_of", descriptor.name);
        assert_eq!(
            reflected_size as usize, size,
            "{}: reflected size",
            descriptor.name
        );
    }
    check::<Gain>(
        &padded::read_gain::DESCRIPTOR,
        &[("gain", 0), ("_pad0", 4), ("_pad1", 8), ("_pad2", 12)],
        16,
    );
    check::<Mixed>(
        &padded::read_mixed::DESCRIPTOR,
        &[
            ("a", 0),
            ("_pad0", 4),
            ("v", 8),
            ("b", 16),
            ("_pad1", 20),
            ("_pad2", 24),
            ("_pad3", 28),
        ],
        32,
    );
    check::<Outer>(
        &padded::read_outer::DESCRIPTOR,
        &[
            ("a", 0),
            ("_pad0", 4),
            ("_pad1", 8),
            ("_pad2", 12),
            ("inner", 16),
            ("b", 32),
            ("_pad3", 40),
            ("_pad4", 44),
        ],
        48,
    );
    // The nested struct is a struct of the record too, with no padding of its own.
    assert_eq!(
        rust_fields::<Inner>(),
        [
            ("x".to_owned(), 0),
            ("y".to_owned(), 4),
            ("z".to_owned(), 8),
            ("w".to_owned(), 12)
        ]
    );
}

#[test]
fn a_pad_uniform_built_with_new_is_read_field_by_field_on_the_gpu() {
    let device = device();
    let cpu = CpuDevice::new();

    // `Gain`: one user field, three padding fields.
    let gain = Gain::new(2.5);
    let out = nan_floats(&device, 1);
    let uniform = device.create_uniform("gain", &gain).unwrap();
    device
        .dispatch_call(&padded::read_gain::call(1, &uniform, &out))
        .unwrap();
    assert_eq!(device.read_typed_buffer(&out).unwrap(), [2.5]);
    let twin = cpu.buffer(&[f32::NAN]);
    padded::read_gain::dispatch(&cpu, 1, &cpu.uniform(gain).unwrap(), &twin).unwrap();
    assert_eq!(twin.read(), [2.5]);

    // `Mixed`: four distinct values, so a swap of two fields or a wrong offset changes the result.
    let mixed = Mixed::new(1.0, Vec2::new(2.0, 3.0), 4.0);
    let out = nan_floats(&device, 4);
    let uniform = device.create_uniform("mixed", &mixed).unwrap();
    device
        .dispatch_call(&padded::read_mixed::call(4, &uniform, &out))
        .unwrap();
    assert_eq!(
        device.read_typed_buffer(&out).unwrap(),
        [1.0, 2.0, 3.0, 4.0]
    );
    let twin = cpu.buffer(&[f32::NAN; 4]);
    padded::read_mixed::dispatch(&cpu, 4, &cpu.uniform(mixed).unwrap(), &twin).unwrap();
    assert_eq!(twin.read(), [1.0, 2.0, 3.0, 4.0]);

    // `Outer`: a nested struct built with its own `new`, every user field read back.
    let outer = Outer::new(1.0, Inner::new(2.0, 3.0, 4.0, 5.0), Vec2::new(6.0, 7.0));
    let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let out = nan_floats(&device, 7);
    let uniform = device.create_uniform("outer", &outer).unwrap();
    device
        .dispatch_call(&padded::read_outer::call(7, &uniform, &out))
        .unwrap();
    assert_eq!(device.read_typed_buffer(&out).unwrap(), expected);
    let twin = cpu.buffer(&[f32::NAN; 7]);
    padded::read_outer::dispatch(&cpu, 7, &cpu.uniform(outer).unwrap(), &twin).unwrap();
    assert_eq!(twin.read(), expected);
}
