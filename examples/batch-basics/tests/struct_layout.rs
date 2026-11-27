//! The std430 layouts of a struct element (m14.s2s, D279), one test pair per layout, each compared with the host
//! reference and with the CPU twin: (a) two `Vec2`, an `f32` and a `u32` (24 bytes, align 8); (b) a `Vec4`, a `Vec2`
//! and two pad words (32 bytes, align 16); (c) a `Mat4` and four scalar words (80 bytes, align 16).
//!
//! Every stage reads or stores a member that sits after a wide member, so a wrong offset, alignment or stride shows as
//! a wrong value. Building the first frame builds each pipeline, which runs the D17 reflection cross-check (host
//! `TypeLayout` against the compiler's reflected layout, `crates/gust/src/reflect/pod.rs`); a mismatch is an error
//! that fails the `unwrap`s below, so a passing GPU test also means the cross-check passed. The `twin_` tests run on
//! the CPU twin; the `gpu_` ones are `#[ignore]`d like the example's other GPU tests and run in full verification on
//! the adapter the environment selects.

use std::fmt::Debug;
use std::sync::Arc;

use batch_basics::app::block_on;
use gust::batch::{Batch, BatchDevice, D1, Element, Error, Stage};
use gust::cpu::CpuDevice;
use gust::prelude::{Mat4, Vec2, Vec4};
use gust::{GpuPod, gpu_batch, gpu_task};
use gust_wgpu::HeadlessDevice;

const N: u32 = 40;
const FRAMES: u32 = 4;

fn private_device() -> Arc<HeadlessDevice> {
    Arc::new(HeadlessDevice::new().unwrap_or_else(|error| {
        panic!("a GPU adapter is required; this test must not silently skip: {error}")
    }))
}

/// One frame over `input`: the rows written, the moved rows and the probe read.
fn run<D, T, P>(
    batch: &mut Batch<D>,
    rows: &Stage<T, D1>,
    moved: &Stage<T, D1>,
    probe: &Stage<P, D1>,
    input: &[T],
) -> (Vec<T>, Vec<P>)
where
    D: BatchDevice,
    T: Element + GpuPod + Send,
    T::Word: Send,
    P: Element + GpuPod + Send,
    P::Word: Send,
    D::Buffer<T>: 'static,
    D::Buffer<P>: 'static,
{
    block_on(async {
        let mut frame = batch.begin().await?;
        frame.write(rows, input)?;
        let moved = frame.ticket(moved)?;
        let probe = frame.ticket(probe)?;
        let _submitted = frame.submit()?;
        Ok::<_, Error>((moved.await?, probe.await?))
    })
    .unwrap()
}

/// `Debug` text, for rows that have no `PartialEq` (`Mat4`); the values are small and exact, so text equality is bit
/// equality.
fn text<T: Debug>(value: &T) -> String {
    format!("{value:?}")
}

/// The body of a layout's twin test, from the module's own build function, input and host reference.
macro_rules! twin_body {
    ($build_cpu:ident, $input:ident, $expected:ident) => {
        let (mut batch, h) = $build_cpu(&Arc::new(CpuDevice::new())).unwrap();
        for seed in 0..FRAMES {
            let input = $input(seed);
            let out = run(&mut batch, &h.rows, &h.moved, &h.probe, &input);
            assert_eq!(text(&out), text(&$expected(&input)), "seed {seed}");
        }
    };
}

/// The body of a layout's GPU test: the device's rows equal the host reference and the CPU twin's.
macro_rules! gpu_body {
    ($build_cpu:ident, $build_gpu:ident, $input:ident, $expected:ident) => {
        let device = private_device();
        println!("adapter: {}", device.adapter_info().name);
        let (mut on_gpu, g) = $build_gpu(&device).unwrap();
        let (mut on_cpu, c) = $build_cpu(&Arc::new(CpuDevice::new())).unwrap();
        for seed in 0..FRAMES {
            let input = $input(seed);
            let out = run(&mut on_gpu, &g.rows, &g.moved, &g.probe, &input);
            assert_eq!(text(&out), text(&$expected(&input)), "host, seed {seed}");
            let twin = run(&mut on_cpu, &c.rows, &c.moved, &c.probe, &input);
            assert_eq!(text(&out), text(&twin), "CPU twin, seed {seed}");
        }
    };
}

/// Two `Vec2`, an `f32` and a `u32`: the second `Vec2` starts at 8, the scalars at 16 and 20.
mod pair {
    use super::*;

    #[derive(Clone, Copy, Debug, gust::GpuStruct)]
    #[repr(C)]
    pub struct Row {
        pub a: Vec2,
        pub b: Vec2,
        pub f: f32,
        pub u: u32,
    }

    #[derive(Clone, Copy)]
    pub struct Handles {
        pub rows: Stage<Row, D1>,
        pub moved: Stage<Row, D1>,
        pub probe: Stage<u32, D1>,
    }

    macro_rules! define_build {
        ($build:ident, $device:ty) => {
            pub fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles), Error> {
                gpu_batch!("layout_pair", on(device); in_flight(1); |b| {
                    let rows = b.input::<Row>(N);
                    let moved = gpu_task!(N, |i| -> Row {
                        Row { a: rows[i].b, b: rows[i].a, f: rows[i].f + 1.0, u: rows[i].u + 2u32 }
                    });
                    let probe = gpu_task!(N, |i| -> u32 { rows[i].u });
                    Handles { rows, moved, probe }
                })
            }
        };
    }
    define_build!(build_cpu, CpuDevice);
    define_build!(build_gpu, HeadlessDevice);

    pub fn input(seed: u32) -> Vec<Row> {
        (0..N)
            .map(|i| Row {
                a: Vec2::new((i + seed) as f32, 0.5 * i as f32),
                b: Vec2::new(-(i as f32), (seed * 3 + i) as f32 * 0.25),
                f: (i * 2 + seed) as f32 * 0.5,
                u: 100 + i * 7 + seed,
            })
            .collect()
    }

    pub fn expected(input: &[Row]) -> (Vec<Row>, Vec<u32>) {
        (
            input
                .iter()
                .map(|r| Row {
                    a: r.b,
                    b: r.a,
                    f: r.f + 1.0,
                    u: r.u + 2,
                })
                .collect(),
            input.iter().map(|r| r.u).collect(),
        )
    }
}

#[test]
fn twin_a_struct_of_two_vec2_and_two_scalars_matches_the_host() {
    twin_body!(build_cpu_pair, input_pair, expected_pair);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_struct_of_two_vec2_and_two_scalars_matches_the_host_and_the_twin() {
    gpu_body!(build_cpu_pair, build_gpu_pair, input_pair, expected_pair);
}
use pair::{
    build_cpu as build_cpu_pair, build_gpu as build_gpu_pair, expected as expected_pair,
    input as input_pair,
};

/// A `Vec4`, a `Vec2` and two pad words: 32 bytes, the `Vec2` at 16, the pads at 24 and 28.
mod wide {
    use super::*;

    #[derive(Clone, Copy, Debug, gust::GpuStruct)]
    #[repr(C)]
    pub struct Row {
        pub a: Vec4,
        pub b: Vec2,
        pub p0: f32,
        pub p1: f32,
    }

    #[derive(Clone, Copy)]
    pub struct Handles {
        pub rows: Stage<Row, D1>,
        pub moved: Stage<Row, D1>,
        pub probe: Stage<f32, D1>,
    }

    macro_rules! define_build {
        ($build:ident, $device:ty) => {
            pub fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles), Error> {
                gpu_batch!("layout_wide", on(device); in_flight(1); |b| {
                    let rows = b.input::<Row>(N);
                    let moved = gpu_task!(N, |i| -> Row {
                        Row { a: rows[i].a, b: rows[i].b, p0: rows[i].p1, p1: rows[i].p0 }
                    });
                    let probe = gpu_task!(N, |i| -> f32 { rows[i].p1 + rows[i].b.y });
                    Handles { rows, moved, probe }
                })
            }
        };
    }
    define_build!(build_cpu, CpuDevice);
    define_build!(build_gpu, HeadlessDevice);

    pub fn input(seed: u32) -> Vec<Row> {
        (0..N)
            .map(|i| Row {
                a: Vec4::new(i as f32, (i + seed) as f32, 0.25, -1.0),
                b: Vec2::new(0.5 * i as f32, (seed + 2 * i) as f32),
                p0: (i + 11) as f32,
                p1: (seed * 5 + i) as f32 * 2.0,
            })
            .collect()
    }

    pub fn expected(input: &[Row]) -> (Vec<Row>, Vec<f32>) {
        (
            input
                .iter()
                .map(|r| Row {
                    a: r.a,
                    b: r.b,
                    p0: r.p1,
                    p1: r.p0,
                })
                .collect(),
            input.iter().map(|r| r.p1 + r.b.y).collect(),
        )
    }
}

#[test]
fn twin_a_struct_of_vec4_vec2_and_two_pads_matches_the_host() {
    twin_body!(build_cpu_wide, input_wide, expected_wide);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_struct_of_vec4_vec2_and_two_pads_matches_the_host_and_the_twin() {
    gpu_body!(build_cpu_wide, build_gpu_wide, input_wide, expected_wide);
}
use wide::{
    build_cpu as build_cpu_wide, build_gpu as build_gpu_wide, expected as expected_wide,
    input as input_wide,
};

/// A `Mat4` and four scalar words: 80 bytes, the scalars at 64, 68, 72 and 76 (three kinds, so a misread shows).
mod matrix {
    use super::*;

    #[derive(Clone, Copy, Debug, gust::GpuStruct)]
    #[repr(C)]
    pub struct Row {
        pub m: Mat4,
        pub a: f32,
        pub b: u32,
        pub c: i32,
        pub p: u32,
    }

    #[derive(Clone, Copy)]
    pub struct Handles {
        pub rows: Stage<Row, D1>,
        pub moved: Stage<Row, D1>,
        pub probe: Stage<i32, D1>,
    }

    macro_rules! define_build {
        ($build:ident, $device:ty) => {
            pub fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles), Error> {
                gpu_batch!("layout_matrix", on(device); in_flight(1); |b| {
                    let rows = b.input::<Row>(N);
                    let moved = gpu_task!(N, |i| -> Row {
                        Row {
                            m: rows[i].m,
                            a: rows[i].a + 1.0,
                            b: rows[i].b + 2u32,
                            c: rows[i].c - 3i32,
                            p: rows[i].p,
                        }
                    });
                    let probe = gpu_task!(N, |i| -> i32 { rows[i].c });
                    Handles { rows, moved, probe }
                })
            }
        };
    }
    define_build!(build_cpu, CpuDevice);
    define_build!(build_gpu, HeadlessDevice);

    pub fn input(seed: u32) -> Vec<Row> {
        (0..N)
            .map(|i| {
                let k = (i + seed) as f32;
                Row {
                    m: Mat4::from_cols(
                        Vec4::new(k, 1.0, 2.0, 3.0),
                        Vec4::new(4.0, k + 1.0, 5.0, 6.0),
                        Vec4::new(7.0, 8.0, k + 2.0, 9.0),
                        Vec4::new(10.0, 11.0, 12.0, k + 3.0),
                    ),
                    a: 0.5 * k,
                    b: 1000 + i,
                    c: i as i32 - 20,
                    p: 7 * seed + i,
                }
            })
            .collect()
    }

    pub fn expected(input: &[Row]) -> (Vec<Row>, Vec<i32>) {
        (
            input
                .iter()
                .map(|r| Row {
                    m: r.m,
                    a: r.a + 1.0,
                    b: r.b + 2,
                    c: r.c - 3,
                    p: r.p,
                })
                .collect(),
            input.iter().map(|r| r.c).collect(),
        )
    }
}

#[test]
fn twin_a_struct_of_a_mat4_and_scalars_matches_the_host() {
    twin_body!(build_cpu_matrix, input_matrix, expected_matrix);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_a_struct_of_a_mat4_and_scalars_matches_the_host_and_the_twin() {
    gpu_body!(
        build_cpu_matrix,
        build_gpu_matrix,
        input_matrix,
        expected_matrix
    );
}
use matrix::{
    build_cpu as build_cpu_matrix, build_gpu as build_gpu_matrix, expected as expected_matrix,
    input as input_matrix,
};
