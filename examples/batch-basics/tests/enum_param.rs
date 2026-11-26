//! An enum parameter and a `match` in a stage of a batch (m14.s2e2b, D270): `b.param::<Mode>(..)`, `frame.set(&mode,
//! Mode::Curve)`, and `match mode.get() { Mode::Linear => .., .. }` in a stage body.
//!
//! The enums derive `GpuEnum`, each declared in a sibling module. `Mode` is brought in by `use modes::Mode;` and
//! written by its own name, and `Tint` is written by path (`types::Tint`, although `use types::Tint;` also names it in
//! the body), so one batch has an enum of each spelling. The stage declares a private copy of each enum in its
//! module and the kernel reads the parameter's word through it, so a body writes the enum's own name (`Mode::Linear`)
//! whatever path the parameter's type was written with. Every value is small, so a result equals the host reference bit
//! for bit. The `twin_` tests run on the CPU twin; the `gpu_` ones are `#[ignore]`d like the example's other GPU tests
//! and run in full verification on the adapter the environment selects.

use std::sync::Arc;

use batch_basics::app::block_on;
use gust::batch::{Batch, BatchDevice, D1, Error, Param, Stage};
use gust::cpu::CpuDevice;
use gust::{gpu_batch, gpu_task};
use gust_wgpu::HeadlessDevice;

mod types {
    /// Words with a gap, so that a copy whose words were counted instead of copied would read the wrong arm.
    #[derive(Clone, Copy, Debug, PartialEq, gust::GpuEnum)]
    #[repr(u32)]
    pub enum Tint {
        Warm = 4,
        Cool = 9,
    }
}

mod modes {
    #[derive(Clone, Copy, Debug, PartialEq, gust::GpuEnum)]
    #[repr(u32)]
    pub enum Mode {
        Linear = 0,
        Curve = 1,
        Step = 5,
    }
}

use modes::Mode;
use types::Tint;

const N: u32 = 64;

const MODES: [Mode; 3] = [Mode::Linear, Mode::Curve, Mode::Step];
const TINTS: [Tint; 2] = [Tint::Warm, Tint::Cool];

#[derive(Clone, Copy)]
struct Handles {
    mode: Param<Mode>,
    tint: Param<Tint>,
    x: Stage<u32, D1>,
    y: Stage<u32, D1>,
}

macro_rules! define_build {
    ($build:ident, $device:ty, $depth:tt) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles), Error> {
            gpu_batch!("enum_param", on(device); in_flight($depth); |b| {
                let mode = b.param::<Mode>(Mode::Linear);
                let tint = b.param::<types::Tint>(Tint::Warm);
                let x = b.input::<u32>(N);
                let y = gpu_task!(N, |i| -> u32 {
                    let scaled: u32 = match mode.get() {
                        Mode::Linear => x[i] * 3u32,
                        Mode::Curve => x[i] * x[i],
                        Mode::Step => x[i] + 100u32,
                    };
                    let shift: u32 = match tint.get() {
                        Tint::Warm => 1u32,
                        Tint::Cool => 2u32,
                    };
                    scaled + shift
                });
                Handles { mode, tint, x, y }
            })
        }
    };
}

define_build!(build_cpu, CpuDevice, 1);
define_build!(build_cpu2, CpuDevice, 2);
define_build!(build_gpu, HeadlessDevice, 1);

fn private_device() -> Arc<HeadlessDevice> {
    Arc::new(HeadlessDevice::new().unwrap_or_else(|error| {
        panic!("a GPU adapter is required; this test must not silently skip: {error}")
    }))
}

fn data(seed: u32) -> Vec<u32> {
    (0..N).map(|i| (i * 7 + seed) % 64).collect()
}

/// The host reference: the same two `match`es on the real enums.
fn reference(x: &[u32], mode: Mode, tint: Tint) -> Vec<u32> {
    x.iter()
        .map(|&v| {
            let scaled = match mode {
                Mode::Linear => v * 3,
                Mode::Curve => v * v,
                Mode::Step => v + 100,
            };
            scaled
                + match tint {
                    Tint::Warm => 1,
                    Tint::Cool => 2,
                }
        })
        .collect()
}

/// One frame: the parameters set when given, the input written when given, and the result.
fn frame<D: BatchDevice>(
    batch: &mut Batch<D>,
    h: &Handles,
    set: Option<(Mode, Tint)>,
    input: Option<&[u32]>,
) -> Vec<u32>
where
    D::Buffer<u32>: 'static,
{
    block_on(async {
        let mut frame = batch.begin().await?;
        if let Some((mode, tint)) = set {
            frame.set(&h.mode, mode)?;
            frame.set(&h.tint, tint)?;
        }
        if let Some(input) = input {
            frame.write(&h.x, input)?;
        }
        let ticket = frame.ticket(&h.y)?;
        let _submitted = frame.submit()?;
        ticket.await
    })
    .unwrap()
}

/// Every pair of values, each in a frame of its own, then the first pair again, then frames that set nothing: each
/// result equals the host reference, so both arms of both `match`es are taken and a parameter held across frames is
/// still its value.
fn every_value_equals_the_host<D: BatchDevice>(mut batch: Batch<D>, h: &Handles)
where
    D::Buffer<u32>: 'static,
{
    let x = data(1);
    let first = frame(&mut batch, h, None, Some(&x));
    assert_eq!(
        first,
        reference(&x, Mode::Linear, Tint::Warm),
        "the starting values"
    );
    for mode in MODES {
        for tint in TINTS {
            let got = frame(&mut batch, h, Some((mode, tint)), None);
            assert_eq!(got, reference(&x, mode, tint), "{mode:?} {tint:?}");
        }
    }
    let held = frame(&mut batch, h, None, None);
    assert_eq!(
        held,
        reference(&x, Mode::Step, Tint::Cool),
        "a frame that sets nothing holds the last values"
    );
    let again = frame(&mut batch, h, Some((Mode::Linear, Tint::Warm)), None);
    assert_eq!(again, first, "back to the starting values");
    let y = data(2);
    let fresh = frame(&mut batch, h, Some((Mode::Curve, Tint::Cool)), Some(&y));
    assert_eq!(fresh, reference(&y, Mode::Curve, Tint::Cool), "new data");
}

#[test]
fn twin_an_enum_parameter_and_a_match_equal_the_host() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu(&device).unwrap();
    every_value_equals_the_host(batch, &h);
}

#[test]
fn twin_an_enum_parameter_with_two_frames_in_flight_equals_the_host() {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu2(&device).unwrap();
    every_value_equals_the_host(batch, &h);
}

#[test]
fn twin_an_enum_parameter_set_back_to_its_value_changes_nothing() {
    let device = Arc::new(CpuDevice::new());
    let (mut batch, h) = build_cpu(&device).unwrap();
    let x = data(3);
    let a = frame(&mut batch, &h, Some((Mode::Step, Tint::Cool)), Some(&x));
    let b = frame(&mut batch, &h, Some((Mode::Step, Tint::Cool)), None);
    assert_eq!(a, reference(&x, Mode::Step, Tint::Cool));
    assert_eq!(b, a, "the same values set again");
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_an_enum_parameter_and_a_match_equal_the_host() {
    let device = private_device();
    println!("adapter: {}", device.adapter_info().name);
    let (batch, h) = build_gpu(&device).unwrap();
    every_value_equals_the_host(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_an_enum_parameter_equals_the_cpu_twin_at_every_pair() {
    let gpu = private_device();
    let cpu = Arc::new(CpuDevice::new());
    println!("adapter: {}", gpu.adapter_info().name);
    let (mut on_gpu, gpu_h) = build_gpu(&gpu).unwrap();
    let (mut on_cpu, cpu_h) = build_cpu(&cpu).unwrap();
    let x = data(5);
    for (index, (mode, tint)) in MODES
        .into_iter()
        .flat_map(|mode| TINTS.map(|tint| (mode, tint)))
        .enumerate()
    {
        let input = (index == 0).then_some(&x[..]);
        let from_gpu = frame(&mut on_gpu, &gpu_h, Some((mode, tint)), input);
        let from_cpu = frame(&mut on_cpu, &cpu_h, Some((mode, tint)), input);
        assert_eq!(from_gpu, from_cpu, "{mode:?} {tint:?}");
        assert_eq!(from_gpu, reference(&x, mode, tint), "{mode:?} {tint:?}");
    }
}
