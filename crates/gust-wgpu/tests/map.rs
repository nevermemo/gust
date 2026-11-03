//! The element-wise map calls of `#[gpu]` on a function (m7.a1, D60; Gpu tier).
//!
//! `tests/fixtures/map.rs` holds the functions; each gets `f::map`, `f::map_on`, `f::map_gpu`, `f::map_async` and the
//! rest. Every comparison is against a host reference written here with plain Rust arithmetic: the float inputs are
//! small integer-valued floats, so `a * x + y` is exact on the GPU and on the host and the comparisons are `==`. The
//! CPU twin (`map_on` over `CpuDevice`, which the macro generates for the test build) runs the same kernel body with
//! the workgroup-rounded launch of the GPU, so a missing bound check in the kernel panics there even where the GPU's
//! robust buffer access hides it.
//!
//! Sizes: 0, 1, 255, 256, 257, 1000 and 65537 (either side of the workgroup width and of 256 workgroups), and one
//! element count above the 1D ceiling of one grid-stride round (`CEILING + 1000`), which takes a second round.

#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]

use std::{
    future::Future,
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Waker},
    thread,
    time::{Duration, Instant},
};

use gust::{
    cpu::CpuDevice,
    map::{Broadcast, CEILING, Error as MapError},
    math::Vec2,
};
use gust_wgpu::{DeviceHandle, GpuVec, HeadlessDevice, default_device};

include!("../../../tests/fixtures/map.rs");

/// Drive a future that completes only while the device is polled.
fn drive<T>(future: impl Future<Output = T>) -> T {
    let mut future = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    let started = Instant::now();
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "the call never completed"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

const SIZES: [usize; 8] = [0, 1, 2, 255, 256, 257, 1000, 65_537];

/// Small integer-valued floats, different in every position and between `salt`s.
fn floats(len: usize, salt: u32) -> Vec<f32> {
    (0..len as u32)
        .map(|i| ((i.wrapping_mul(2_654_435_761).wrapping_add(salt) >> 7) % 1000) as f32 - 500.0)
        .collect()
}

/// The components of vectors, which have no `PartialEq`.
fn components(vectors: &[Vec2]) -> Vec<(f32, f32)> {
    vectors.iter().map(|v| (v.x, v.y)).collect()
}

fn saxpy_reference(a: f32, x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y).map(|(x, y)| a * x + y).collect()
}

#[test]
fn saxpy_on_every_size_matches_the_host_and_the_cpu_twin() {
    let cpu = CpuDevice::new();
    for len in SIZES {
        let x = floats(len, 1);
        let y = floats(len, 2);
        let expected = saxpy_reference(2.0, &x, &y);
        let gpu = saxpy::map(Broadcast(2.0), &x, &y).unwrap();
        assert_eq!(gpu, expected, "GPU, {len} elements");
        let twin = saxpy::map_on(&cpu, Broadcast(2.0), &x, &y).unwrap();
        assert_eq!(twin, expected, "CPU twin, {len} elements");
    }
}

#[test]
fn a_broadcast_is_read_at_index_zero_in_any_position() {
    let cpu = CpuDevice::new();
    let a = floats(300, 3);
    let x = floats(300, 4);
    let y = floats(300, 5);
    // Each argument in turn is the scalar.
    let cases = [
        (
            saxpy::map(Broadcast(-3.0), &x, &y).unwrap(),
            saxpy::map_on(&cpu, Broadcast(-3.0), &x, &y).unwrap(),
            x.iter()
                .zip(&y)
                .map(|(x, y)| -3.0 * x + y)
                .collect::<Vec<_>>(),
        ),
        (
            saxpy::map(&a, Broadcast(7.0), &y).unwrap(),
            saxpy::map_on(&cpu, &a, Broadcast(7.0), &y).unwrap(),
            a.iter().zip(&y).map(|(a, y)| a * 7.0 + y).collect(),
        ),
        (
            saxpy::map(&a, &x, Broadcast(11.0)).unwrap(),
            saxpy::map_on(&cpu, &a, &x, Broadcast(11.0)).unwrap(),
            a.iter().zip(&x).map(|(a, x)| a * x + 11.0).collect(),
        ),
        (
            saxpy::map(&a, Broadcast(2.0), Broadcast(5.0)).unwrap(),
            saxpy::map_on(&cpu, &a, Broadcast(2.0), Broadcast(5.0)).unwrap(),
            a.iter().map(|a| a * 2.0 + 5.0).collect(),
        ),
    ];
    for (index, (gpu, twin, expected)) in cases.into_iter().enumerate() {
        assert_eq!(gpu, expected, "GPU, case {index}");
        assert_eq!(twin, expected, "CPU twin, case {index}");
    }
    // Arrays, `Vec`s and slices are all arguments.
    let array = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(
        saxpy::map(Broadcast(10.0), &array, &vec![1.0f32; 4]).unwrap(),
        [11.0, 21.0, 31.0, 41.0]
    );
}

#[test]
fn lengths_must_agree_and_the_count_must_exist() {
    let long = [1.0f32, 2.0, 3.0];
    let short = [1.0f32, 2.0];
    for result in [
        saxpy::map(Broadcast(1.0), &long, &short),
        saxpy::map_on(&CpuDevice::new(), Broadcast(1.0), &long, &short),
    ] {
        let Err(MapError::LengthMismatch {
            parameter,
            expected,
            found,
        }) = result
        else {
            panic!("a mismatch is an error: {result:?}");
        };
        assert_eq!((parameter, expected, found), ("y", 3, 2));
    }
    // The first mismatch in parameter order is the one reported.
    let Err(MapError::LengthMismatch { parameter, .. }) =
        saxpy::map(&[1.0f32; 4][..], &long, &short)
    else {
        panic!("a mismatch is an error");
    };
    assert_eq!(parameter, "x");
    // A call of scalars only has no length.
    assert!(matches!(
        saxpy::map(Broadcast(1.0), Broadcast(2.0), Broadcast(3.0)),
        Err(MapError::Empty)
    ));
    // An empty slice is a zero count, not an error, and runs nothing.
    let empty: [f32; 0] = [];
    assert!(
        saxpy::map(Broadcast(1.0), &empty, &empty)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn bool_arguments_and_results_are_bool_vectors() {
    let cpu = CpuDevice::new();
    for len in [1usize, 255, 1000] {
        let n: Vec<u32> = (0..len as u32).collect();
        let flags: Vec<bool> = (0..len).map(|i| i % 3 != 0).collect();
        let expected: Vec<bool> = n
            .iter()
            .zip(&flags)
            .map(|(n, f)| n % 2 == 0 && *f)
            .collect();
        assert_eq!(even_and::map(&n, &flags).unwrap(), expected, "GPU, {len}");
        assert_eq!(
            even_and::map_on(&cpu, &n, &flags).unwrap(),
            expected,
            "CPU twin, {len}"
        );
        let all_true: Vec<bool> = n.iter().map(|n| n % 2 == 0).collect();
        assert_eq!(even_and::map(&n, Broadcast(true)).unwrap(), all_true);
        assert_eq!(
            even_and::map(&n, Broadcast(false)).unwrap(),
            vec![false; len]
        );
    }
}

#[test]
fn signed_integers_and_branches() {
    let cpu = CpuDevice::new();
    let x: Vec<i32> = (-600..600).collect();
    let expected: Vec<i32> = x.iter().map(|x| (*x).clamp(-100, 250)).collect();
    assert_eq!(
        clamp_signed::map(&x, Broadcast(-100), Broadcast(250)).unwrap(),
        expected
    );
    assert_eq!(
        clamp_signed::map_on(&cpu, &x, Broadcast(-100), Broadcast(250)).unwrap(),
        expected
    );
    // The bounds vary by element, too.
    let low: Vec<i32> = x.iter().map(|x| x / 4 - 50).collect();
    let high: Vec<i32> = x.iter().map(|x| x / 4 + 50).collect();
    let expected: Vec<i32> = x
        .iter()
        .zip(low.iter().zip(&high))
        .map(|(x, (low, high))| (*x).clamp(*low, *high))
        .collect();
    assert_eq!(clamp_signed::map(&x, &low, &high).unwrap(), expected);
}

#[test]
fn vector_elements() {
    let cpu = CpuDevice::new();
    let v: Vec<Vec2> = floats(777, 6)
        .into_iter()
        .zip(floats(777, 7))
        .map(|(x, y)| Vec2::new(x, y))
        .collect();
    let expected: Vec<(f32, f32)> = v.iter().map(|v| (v.x * 0.5, -v.y * 0.5)).collect();
    assert_eq!(
        components(&mirror::map(&v, Broadcast(0.5)).unwrap()),
        expected
    );
    assert_eq!(
        components(&mirror::map_on(&cpu, &v, Broadcast(0.5)).unwrap()),
        expected
    );
    let scales: Vec<f32> = (0..777).map(|i| (i % 5) as f32).collect();
    let expected: Vec<(f32, f32)> = v
        .iter()
        .zip(&scales)
        .map(|(v, s)| (v.x * s, -v.y * s))
        .collect();
    assert_eq!(components(&mirror::map(&v, &scales).unwrap()), expected);
}

#[test]
fn resident_arguments_chain_without_a_round_trip() {
    let x = floats(5000, 8);
    let y = floats(5000, 9);
    let resident_x = GpuVec::from_slice(&x).unwrap();
    // A resident argument beside a slice and a scalar.
    let sum = saxpy::map_gpu(Broadcast(2.0), &resident_x, &y).unwrap();
    assert_eq!(sum.len(), 5000);
    assert_eq!(sum.to_vec().unwrap(), saxpy_reference(2.0, &x, &y));
    // The result feeds the next call as it is; the host sees only the last one.
    let once = add_one::map_gpu(&GpuVec::from_slice(&[0u32, 1, 2, 3]).unwrap()).unwrap();
    let twice = add_one::map_gpu(&once).unwrap();
    let thrice = add_one::map_gpu(&twice).unwrap();
    assert_eq!(thrice.to_vec().unwrap(), [3, 4, 5, 6]);
    // `map` takes a resident argument and returns a `Vec`.
    assert_eq!(add_one::map(&thrice).unwrap(), [4, 5, 6, 7]);
    // An empty resident vector is an empty call.
    let nothing: GpuVec<u32> = GpuVec::zeroed(0).unwrap();
    assert_eq!(add_one::map_gpu(&nothing).unwrap().len(), 0);
}

#[test]
fn resident_arguments_of_two_devices_are_mixed_devices() {
    let default = default_device().unwrap();
    let other: DeviceHandle = Arc::new(HeadlessDevice::new().expect("Vulkan adapter required"));
    assert_ne!(default.id(), other.id());
    let on_other = GpuVec::from_slice_on(&other, &[1u32, 2, 3]).unwrap();
    let on_default = GpuVec::from_slice_on(&default, &[1u32, 2, 3]).unwrap();
    // The default device is `map_gpu`'s; the vector lives on the other one.
    let probe = add_one::map_gpu(&on_other).map(|v| v.len());
    assert!(matches!(probe, Err(MapError::MixedDevices)), "{probe:?}");
    assert!(matches!(
        add_one::map(&on_other),
        Err(MapError::MixedDevices)
    ));
    assert!(matches!(
        add_one::map_gpu_on(&other, &on_default),
        Err(MapError::MixedDevices)
    ));
    // On its own device the same call works.
    let result = add_one::map_gpu_on(&other, &on_other).unwrap();
    assert_eq!(result.to_vec().unwrap(), [2, 3, 4]);
    assert_eq!(result.device_id(), other.id());
}

#[test]
fn the_async_calls_do_not_block() {
    let x = floats(3000, 10);
    let y = floats(3000, 11);
    let expected = saxpy_reference(4.0, &x, &y);
    assert_eq!(
        drive(saxpy::map_async(Broadcast(4.0), &x, &y)).unwrap(),
        expected
    );
    let default = default_device().unwrap();
    assert_eq!(
        drive(saxpy::map_on_async(&*default, Broadcast(4.0), &x, &y)).unwrap(),
        expected
    );
    // The same call through the CPU twin, awaited.
    assert_eq!(
        drive(saxpy::map_on_async(
            &CpuDevice::new(),
            Broadcast(4.0),
            &x,
            &y
        ))
        .unwrap(),
        expected
    );
    // Errors come back from the future, before any device work.
    let short = [1.0f32];
    assert!(matches!(
        drive(saxpy::map_async(Broadcast(1.0), &x, &short)),
        Err(MapError::LengthMismatch { .. })
    ));
}

// A parameter called `id` is a plain input (the kernel's dispatch id has a reserved name).
#[gust::gpu]
pub fn hash(id: u32, salt: u32) -> u32 {
    (id + salt) * 3u32
}

#[test]
fn a_parameter_named_id_is_an_ordinary_input() {
    let ids: Vec<u32> = (0..700).collect();
    let expected: Vec<u32> = ids.iter().map(|id| (id + 9) * 3).collect();
    assert_eq!(hash::map(&ids, Broadcast(9)).unwrap(), expected);
    assert_eq!(
        hash::map_on(&CpuDevice::new(), &ids, Broadcast(9)).unwrap(),
        expected
    );
}

#[test]
fn the_descriptor_is_the_generated_kernels() {
    let descriptor = saxpy::DESCRIPTOR;
    assert_eq!(descriptor.name, "map_kernel");
    assert_eq!(descriptor.entry_point, "gpu_saxpy_map_kernel");
    assert_eq!(descriptor.workgroup_size, [256, 1, 1]);
    let names: Vec<&str> = descriptor.parameters.iter().map(|p| p.name).collect();
    assert_eq!(names, ["__gust_gid", "a", "x", "y", "__gust_out"]);
    assert_eq!(saxpy::MODULE_DESCRIPTOR.name, "saxpy");
    const { assert!(saxpy::OWN_INDEX) };
    assert_eq!(CEILING, 65_535 * 256);
}

/// One element count above the ceiling of a round: every invocation runs its second round for the elements past
/// 16 776 960. The input is a counting sequence, so a wrong index in either round shows as a wrong value.
#[test]
fn more_elements_than_one_round_takes_a_second_round() {
    let len = CEILING as usize + 1000;
    let input: Vec<u32> = (0..len as u32).collect();
    let output = add_one::map(&input).unwrap();
    assert_eq!(output.len(), len);
    let wrong = output
        .iter()
        .zip(&input)
        .position(|(out, input)| *out != input + 1);
    assert_eq!(wrong, None, "the first wrong element");
}
