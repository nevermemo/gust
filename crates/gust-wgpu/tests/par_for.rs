//! `par_for!` on a real device (slice m7.p, plan section 3.1): the calls of `tests/fixtures/par_for.rs` run on the GPU
//! and on the CPU twin and compared with independent host references, over one, two and three dimensions, extents that
//! are not a multiple of the workgroup, extents shorter than the buffer (only the bounds guard keeps the last
//! workgroup out of the rest of it), a zero extent, an extent past the launch cap and one past `u32`; the default
//! device; and resident `GpuVec` bindings.
//!
//! The fixture is included twice, once per backend, with `Dev` naming the device type. Buffers are small (a few
//! thousand elements), so every test fits the 128 MiB storage bindings of the CPU adapter that public CI runs on.

#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]

use std::sync::Arc;

use gust::cpu;
use gust_wgpu::{DeviceHandle, GpuVec, HeadlessDevice};

mod on_gpu {
    use gust_wgpu::HeadlessDevice as Dev;

    include!("../../../tests/fixtures/par_for.rs");
}

mod on_twin {
    use gust::cpu::CpuDevice as Dev;

    include!("../../../tests/fixtures/par_for.rs");
}

/// A value no kernel writes, to show an element the loop never reached.
const UNWRITTEN: u32 = 0xDEAD_BEEF;

fn handle() -> DeviceHandle {
    Arc::new(
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip"),
    )
}

/// The device of each backend, as the fixture's functions take it.
trait Backend {
    type Dev;
    fn open() -> Self::Dev;
    /// The error's device cause says "the launch is over the cap" (`DeviceLimit`).
    fn is_over_cap(error: &gust::map::Error) -> bool;
}

struct Gpu;
struct Twin;

impl Backend for Gpu {
    type Dev = DeviceHandle;
    fn open() -> DeviceHandle {
        handle()
    }
    fn is_over_cap(error: &gust::map::Error) -> bool {
        matches!(
            error.downcast_ref::<gust_wgpu::Error>(),
            Some(gust_wgpu::Error::DeviceLimit { .. })
        )
    }
}

impl Backend for Twin {
    type Dev = cpu::CpuDevice;
    fn open() -> cpu::CpuDevice {
        cpu::CpuDevice::new()
    }
    fn is_over_cap(error: &gust::map::Error) -> bool {
        matches!(
            error.downcast_ref::<cpu::Error>(),
            Some(cpu::Error::DeviceLimit { .. })
        )
    }
}

fn gpu_device(device: &DeviceHandle) -> &HeadlessDevice {
    device
}

fn twin_device(device: &cpu::CpuDevice) -> &cpu::CpuDevice {
    device
}

/// Every test of one backend, written once: `$backend` says how its device opens and what its over-cap error is,
/// `$module` holds the fixture's functions for it, and `$device` turns the opened device into the `&Dev` they take.
macro_rules! scenarios {
    ($name:ident, $backend:ty, $module:ident, $device:expr) => {
        mod $name {
            use super::*;

            fn device() -> <$backend as Backend>::Dev {
                <$backend as Backend>::open()
            }

            fn pair(n: usize) -> (Vec<f32>, Vec<f32>) {
                // Small integers: every product and sum below is exact in f32, so a bit-exact comparison is fair.
                let a = (0..n).map(|i| (i % 17) as f32 + 1.0).collect();
                let b = (0..n).map(|i| (i % 5) as f32 - 2.0).collect();
                (a, b)
            }

            #[test]
            fn one_dimension_over_lengths_around_the_workgroup() {
                let device = device();
                let dev = $device(&device);
                for n in [1_usize, 2, 255, 256, 257, 1000, 4097] {
                    let (a, b) = pair(n);
                    let mut out = vec![-1.0_f32; n];
                    $module::product_1d(&a, &b, &mut out, dev).unwrap();
                    let expected: Vec<f32> = a.iter().zip(&b).map(|(a, b)| a * b).collect();
                    assert_eq!(out, expected, "n = {n}");
                }
            }

            #[test]
            fn the_workgroup_size_can_be_overridden() {
                let device = device();
                let dev = $device(&device);
                let (a, b) = pair(1000);
                let mut out = vec![-1.0_f32; 1000];
                $module::product_64(&a, &b, &mut out, dev).unwrap();
                let expected: Vec<f32> = a.iter().zip(&b).map(|(a, b)| a * b).collect();
                assert_eq!(out, expected);
            }

            #[test]
            fn the_bounds_guard_keeps_the_last_workgroup_out_of_the_rest_of_the_buffer() {
                // (Mutation: dropping the guard writes `out[1000..1024]`.)
                let device = device();
                let dev = $device(&device);
                for n in [1_u32, 255, 256, 257, 1000, 1023] {
                    let mut out = vec![UNWRITTEN; 1024];
                    $module::fill_prefix(&mut out, n, dev).unwrap();
                    for (index, value) in out.iter().enumerate() {
                        let expected = if (index as u32) < n {
                            index as u32 + 1
                        } else {
                            UNWRITTEN
                        };
                        assert_eq!(*value, expected, "n = {n}, index = {index}");
                    }
                }
            }

            #[test]
            fn two_dimensions_with_extents_off_the_workgroup() {
                let device = device();
                let dev = $device(&device);
                for (w, h) in [(1_u32, 1_u32), (16, 16), (17, 3), (37, 21), (3, 40)] {
                    let count = (w * h) as usize;
                    let img: Vec<f32> = (0..count).map(|i| (i % 23) as f32).collect();
                    let mut out = vec![-1.0_f32; count];
                    $module::halve_2d(&img, &mut out, w, h, dev).unwrap();
                    let expected: Vec<f32> = img.iter().map(|v| v * 0.5).collect();
                    assert_eq!(out, expected, "{w} x {h}");
                }
            }

            #[test]
            fn two_dimensions_guard_each_axis() {
                let device = device();
                let dev = $device(&device);
                let (w, h, stride) = (37_u32, 21_u32, 48_u32);
                let mut out = vec![UNWRITTEN; (stride * 32) as usize];
                $module::tag_2d(&mut out, w, h, stride, dev).unwrap();
                for row in 0..32_u32 {
                    for column in 0..stride {
                        let expected = if row < h && column < w {
                            row * 1000 + column + 1
                        } else {
                            UNWRITTEN
                        };
                        assert_eq!(
                            out[(row * stride + column) as usize],
                            expected,
                            "({column}, {row})"
                        );
                    }
                }
            }

            #[test]
            fn three_dimensions_with_extents_off_the_workgroup() {
                let device = device();
                let dev = $device(&device);
                for (w, h, d) in [(8_u32, 8_u32, 4_u32), (10, 9, 5), (1, 1, 1), (3, 2, 9)] {
                    let count = (w * h * d) as usize;
                    let mut out = vec![UNWRITTEN; count];
                    $module::index_3d(&mut out, w, h, d, dev).unwrap();
                    let mut expected = Vec::with_capacity(count);
                    for z in 0..d {
                        for y in 0..h {
                            for x in 0..w {
                                expected.push(z * 1_000_000 + y * 1000 + x);
                            }
                        }
                    }
                    assert_eq!(out, expected, "{w} x {h} x {d}");
                }
            }

            #[test]
            fn scalars_of_every_kind_reach_the_body() {
                let device = device();
                let dev = $device(&device);
                let (a, b) = pair(300);
                for (t, invert, bias) in
                    [(0.25_f32, false, 3_i32), (0.5, true, -7), (1.0, false, 0)]
                {
                    let mut out = vec![0.0_f32; 300];
                    $module::blend(&a, &b, &mut out, t, invert, bias, dev).unwrap();
                    let expected: Vec<f32> = a
                        .iter()
                        .zip(&b)
                        .map(|(a, b)| {
                            let mixed = a * (1.0 - t) + b * t + bias as f32;
                            if invert { 0.0 - mixed } else { mixed }
                        })
                        .collect();
                    assert_eq!(out, expected, "t = {t}, invert = {invert}, bias = {bias}");
                }
            }

            #[test]
            fn a_zero_extent_runs_nothing_and_is_ok() {
                let device = device();
                let dev = $device(&device);
                let mut ones = vec![1.0_f32; 12];
                let mut tags = vec![UNWRITTEN; 12];
                let (a, b) = pair(12);
                // Zero along x, along y, and along z.
                $module::halve_2d(&a, &mut ones, 0, 4, dev).unwrap();
                $module::halve_2d(&a, &mut ones, 3, 0, dev).unwrap();
                $module::index_3d(&mut tags, 3, 4, 0, dev).unwrap();
                $module::fill_prefix(&mut tags, 0, dev).unwrap();
                $module::product_1d(&a, &b, &mut [], dev).unwrap();
                assert_eq!(ones, vec![1.0_f32; 12]);
                assert_eq!(tags, vec![UNWRITTEN; 12]);
            }

            #[test]
            fn an_extent_past_the_launch_cap_is_an_error_and_runs_nothing() {
                let device = device();
                let dev = $device(&device);
                // 65_536 x 65_552 invocations after rounding y up to 16: a little over 2^32.
                let mut out = vec![UNWRITTEN; 16];
                let error = $module::tag_2d(&mut out, 65_536, 65_537, 4, dev).unwrap_err();
                assert!(<$backend as Backend>::is_over_cap(&error), "{error:?}");
                assert_eq!(out, vec![UNWRITTEN; 16]);
            }

            #[test]
            fn an_extent_that_does_not_fit_u32_is_an_error() {
                let device = device();
                let dev = $device(&device);
                let mut out = vec![UNWRITTEN; 8];
                $module::fill_u64(&mut out, 5, dev).unwrap();
                assert_eq!(out[..5], [7; 5]);
                assert_eq!(out[5..], [UNWRITTEN; 3]);
                let mut out = vec![UNWRITTEN; 8];
                let error = $module::fill_u64(&mut out, u64::from(u32::MAX) + 1, dev).unwrap_err();
                assert!(matches!(error, gust::map::Error::Device(_)), "{error:?}");
                assert!(error.to_string().contains("out of range"), "{error}");
                assert_eq!(out, vec![UNWRITTEN; 8]);
            }
        }
    };
}

scenarios!(gpu, Gpu, on_gpu, gpu_device);
scenarios!(twin, Twin, on_twin, twin_device);

// ---- the default device ----

/// `par_for!` without `on(..)`: opens `gust_wgpu::default_device()`.
fn double_on_the_default_device(values: &[f32]) -> Result<Vec<f32>, gust::map::Error> {
    let mut out = vec![0.0_f32; values.len()];
    let n = values.len() as u32;
    gust::par_for!(i in 0..n, [values: &[f32], out: &mut [f32]] {
        out[i] = values[i] * 2.0;
    })?;
    Ok(out)
}

#[test]
fn without_on_the_default_device_runs_the_loop() {
    let values: Vec<f32> = (0..777).map(|i| i as f32).collect();
    let doubled = double_on_the_default_device(&values).unwrap();
    assert_eq!(doubled, values.iter().map(|v| v * 2.0).collect::<Vec<_>>());
}

// ---- resident buffers ----

#[test]
fn resident_bindings_are_bound_where_they_are() {
    let device = handle();
    let input: Vec<f32> = (0..1500).map(|i| (i % 31) as f32).collect();
    let source = GpuVec::<f32>::from_slice_on(&device, &input).unwrap();
    let mut doubled = GpuVec::<f32>::zeroed_on(&device, 1500).unwrap();
    let n = source.len();
    // A resident binding is the variable of its declared type: a reference.
    let (source, doubled) = (&source, &mut doubled);
    gust::par_for!(on(&*device); i in 0..n, [source: &GpuVec<f32>, doubled: &mut GpuVec<f32>] {
        doubled[i] = source[i] * 2.0;
    })
    .unwrap();
    // In place, on the same buffer, as a second loop.
    gust::par_for!(on(&*device); i in 0..n, [doubled: &mut GpuVec<f32>] {
        doubled[i] += 1.0;
    })
    .unwrap();
    let expected: Vec<f32> = input.iter().map(|v| v * 2.0 + 1.0).collect();
    assert_eq!(doubled.to_vec().unwrap(), expected);
    assert_eq!(source.to_vec().unwrap(), input);
}
