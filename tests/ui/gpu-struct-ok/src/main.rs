// A free-standing `#[derive(GpuStruct)]` struct compiles, and its layout is the std430 one (D279): offsets, size and
// alignment of three rows, one of them with explicit pad fields and one with a matrix. `check-ui` only runs `cargo check`:
// the derive's const assertions are what this fixture checks; the asserts in `main` are not run.

use gust::prelude::*;
use gust::{GpuPod, GpuStruct, TypeLayoutKind};

#[derive(Clone, Copy, GpuStruct)]
#[repr(C)]
struct Packed {
    a: Vec2,
    b: Vec2,
    c: f32,
    d: u32,
}

#[derive(Clone, Copy, GpuStruct)]
#[repr(C)]
struct Padded {
    a: Vec4,
    b: Vec2,
    _pad0: f32,
    _pad1: f32,
}

#[derive(Clone, Copy, GpuStruct)]
#[repr(C)]
struct Matrixed {
    m: Mat4,
    a: f32,
    b: u32,
    c: i32,
    _pad: u32,
}

fn offsets<T: GpuStruct>() -> (u32, u32, Vec<u32>) {
    let layout = T::LAYOUT;
    let TypeLayoutKind::Struct(fields) = layout.kind else {
        panic!("a struct layout");
    };
    (
        layout.size,
        layout.alignment,
        fields.iter().map(|field| field.offset).collect(),
    )
}

fn main() {
    assert_eq!(offsets::<Packed>(), (24, 8, vec![0, 8, 16, 20]));
    assert_eq!(offsets::<Padded>(), (32, 16, vec![0, 16, 24, 28]));
    assert_eq!(offsets::<Matrixed>(), (80, 16, vec![0, 64, 68, 72, 76]));
    let _ = (Packed { a: Vec2::new(0.0, 0.0), b: Vec2::new(0.0, 0.0), c: 0.0, d: 0 }, Padded { a: Vec4::new(0.0, 0.0, 0.0, 0.0), b: Vec2::new(0.0, 0.0), _pad0: 0.0, _pad1: 0.0 });
}
