//! The uniforms fixture on the CPU twin (roadmap M6, slice m6.u2; Unit tier, no adapter, no slangc).
//!
//! `tests/fixtures/uniforms.rs` is the module of the golden `uniforms.slang`. Including it here compiles its expansion
//! with rustc, which the string tests of the macro cannot do: the `#[repr(C, align(16))]` declarations, the `Std140V1`
//! const assertions, the typed call with its `Kind` bounds and the twin that reads the uniform by copy. The kernel runs
//! on the twin and each output is compared with a hand-computed reference (the arithmetic is in the comments, and
//! every value is exactly representable, so the comparison is exact). The twin is never GPU evidence.

include!("../../../tests/fixtures/uniforms.rs");

use gust::cpu::CpuDevice;
use gust::math::{Mat4, Vec2, Vec4};
use uniforms::{Frame, Light};

fn frame() -> Frame {
    Frame {
        // Rows (1 2 0 0), (0 1 0 3), (0 0 2 0), (1 0 0 1): every entry of the product below is an integer.
        view: Mat4::from_rows(
            Vec4::new(1.0, 2.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 3.0),
            Vec4::new(0.0, 0.0, 2.0, 0.0),
            Vec4::new(1.0, 0.0, 0.0, 1.0),
        ),
        tint: Vec4::new(1.0, 0.0, 2.0, 1.0),
        light: Light {
            color: Vec4::new(1.0, 0.0, 0.0, 2.0),
            intensity: 2.0,
            falloff: 3.0,
            pad0: 0.0,
            pad1: 0.0,
        },
        offset: Vec2::new(1.0, 2.0),
        scale: 2.0,
        count: 3,
    }
}

/// The kernel on the points below, worked by hand: `out = (lit.x + offset.x, lit.y + offset.y, lit.z * 2, lit.w + 3)`
/// with `lit = (view * p + tint) * 2 + color * 3`.
///
/// - `p = (1, 2, 3, 4)`: `view * p = (1 + 4, 2 + 12, 6, 1 + 4) = (5, 14, 6, 5)`; plus the tint `(6, 14, 8, 6)`;
///   `lit = (12 + 3, 28, 16, 12 + 6) = (15, 28, 16, 18)`; `out = (16, 30, 32, 21)`.
/// - `p = (0, 1, 0, 0)`: `view * p = (2, 1, 0, 0)`; `(3, 1, 2, 1)`; `lit = (6 + 3, 2, 4, 2 + 6) = (9, 2, 4, 8)`;
///   `out = (10, 4, 8, 11)`.
/// - `p = (0, 0, 0, 1)`: `view * p = (0, 3, 0, 1)`; `(1, 3, 2, 2)`; `lit = (2 + 3, 6, 4, 4 + 6) = (5, 6, 4, 10)`;
///   `out = (6, 8, 8, 13)`.
const POINTS: [Vec4; 3] = [
    Vec4::new(1.0, 2.0, 3.0, 4.0),
    Vec4::new(0.0, 1.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, 0.0, 1.0),
];
const EXPECTED: [[f32; 4]; 3] = [
    [16.0, 30.0, 32.0, 21.0],
    [10.0, 4.0, 8.0, 11.0],
    [6.0, 8.0, 8.0, 13.0],
];

fn run(device: &CpuDevice, uniform: &gust::cpu::CpuUniform<Frame>) -> Vec<[f32; 4]> {
    let points = device.buffer(&POINTS);
    let out = device.buffer(&[Vec4::new(f32::NAN, f32::NAN, f32::NAN, f32::NAN); 3]);
    uniforms::run::dispatch(device, 3, uniform, &points, &out)
        .expect("the twin of `run` refused the call");
    out.read().iter().map(|v| [v.x, v.y, v.z, v.w]).collect()
}

#[test]
fn the_twin_of_a_uniform_kernel_matches_the_hand_reference() {
    let device = CpuDevice::new();
    let uniform = device.uniform(frame()).expect("Frame is padding-free");
    assert_eq!(run(&device, &uniform), EXPECTED);
}

#[test]
fn a_new_uniform_value_is_seen_by_the_next_dispatch_and_the_kernel_cannot_change_it() {
    let device = CpuDevice::new();
    let uniform = device.uniform(frame()).unwrap();
    assert_eq!(run(&device, &uniform), EXPECTED);
    // The same points under `scale = 1, count = 0`: only `z` and `w` of each output change (z * 1, w + 0).
    let mut changed = frame();
    changed.scale = 1.0;
    changed.count = 0;
    uniform.write(changed).unwrap();
    let again = run(&device, &uniform);
    assert_eq!(again[0], [16.0, 30.0, 16.0, 18.0]);
    assert_eq!(again[1], [10.0, 4.0, 4.0, 8.0]);
    assert_eq!(again[2], [6.0, 8.0, 4.0, 10.0]);
    // The kernel read a copy: the uniform still holds what was written.
    assert_eq!(uniform.read().scale, 1.0);
    assert_eq!(uniform.read().count, 0);
}

#[test]
fn the_structs_a_uniform_reaches_have_the_std140_layout() {
    use std::mem::{align_of, offset_of, size_of};
    // Light: color 0..16, intensity 16, falloff 20, two padding fields 24 and 28, rounded to 16: 32 bytes.
    assert_eq!(size_of::<Light>(), 32);
    assert_eq!(align_of::<Light>(), 16);
    // Frame: view 0..64, tint 64..80, light 80..112, offset 112..120, scale 120, count 124: 128 bytes, no padding.
    assert_eq!(size_of::<Frame>(), 128);
    assert_eq!(align_of::<Frame>(), 16);
    assert_eq!(offset_of!(Frame, view), 0);
    assert_eq!(offset_of!(Frame, tint), 64);
    assert_eq!(offset_of!(Frame, light), 80);
    assert_eq!(offset_of!(Frame, offset), 112);
    assert_eq!(offset_of!(Frame, scale), 120);
    assert_eq!(offset_of!(Frame, count), 124);
}
