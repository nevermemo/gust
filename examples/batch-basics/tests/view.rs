//! A caller's stable view of a batch's buffers (m14.s2v, decision D261): `Batch::view`, `View` and
//! `Batch::generation` on the CPU twin (no adapter, no GPU; check-cpu requires these tests by name).
//!
//! The batch is `y = x * 3 + gain`, then `y = y * 2 + 1` in place, over `u32` values small enough never to overflow, so
//! a result equals the host reference bit for bit. The GPU half of the same contract is
//! `crates/gust-wgpu/tests/batch_view.rs`.

use std::sync::Arc;

use batch_basics::app::block_on;
use gust::batch::{Batch, BatchDevice, D1, Error, Param, Stage, View};
use gust::cpu::CpuDevice;
use gust::map::MapHost;
use gust::{gpu_batch, gpu_task};

const N: u32 = 64;

struct Handles {
    gain: Param<u32>,
    x: Stage<u32, D1>,
    y: Stage<u32, D1>,
}

macro_rules! define_build {
    ($build:ident, $depth:tt) => {
        fn $build(device: &Arc<CpuDevice>) -> Result<(Batch<CpuDevice>, Handles), Error> {
            gpu_batch!("viewed", on(device); in_flight($depth); |b| {
                let gain = b.param::<u32>(1);
                let x = b.input::<u32>(N);
                let y = gpu_task!(N, |i| -> u32 { x[i] * 3u32 + gain.get() });
                let _double = gpu_task!(&mut y, |i| { y[i] = y[i] * 2u32 + 1u32; });
                Handles { gain, x, y }
            })
        }
    };
}

define_build!(build1, 1);
define_build!(build2, 2);

fn data(seed: u32) -> Vec<u32> {
    (0..N).map(|i| (i * 7 + seed) % 1_000).collect()
}

fn reference(x: &[u32], gain: u32) -> Vec<u32> {
    x.iter().map(|v| (v * 3 + gain) * 2 + 1).collect()
}

/// One frame by `submit`: writes `x`, sets `gain`, reserves a ticket of `y`, and returns what the ticket resolves to.
fn frame(batch: &mut Batch<CpuDevice>, h: &Handles, x: Option<&[u32]>, gain: u32) -> Vec<u32> {
    let mut frame = block_on(batch.begin()).unwrap();
    if let Some(x) = x {
        frame.write(&h.x, x).unwrap();
    }
    frame.set(&h.gain, gain).unwrap();
    let ticket = frame.ticket(&h.y).unwrap();
    let _submitted = frame.submit().unwrap();
    block_on(ticket).unwrap()
}

fn read(device: &CpuDevice, view: &View<CpuDevice, u32>) -> Vec<u32> {
    device.readback(view.buffer()).unwrap()
}

#[test]
fn twin_a_view_of_an_input_and_of_a_stage_holds_what_the_frame_left() {
    let device = Arc::new(CpuDevice::new());
    let (mut batch, h) = build1(&device).unwrap();
    let x_view = batch.view(&h.x).unwrap();
    let y_view = batch.view(&h.y).unwrap();
    assert_eq!(
        (x_view.len(), x_view.width(), x_view.height()),
        (N as usize, N, 1)
    );
    assert!(!y_view.is_empty());
    assert_eq!(
        read(&device, &y_view),
        vec![0; N as usize],
        "zeros at the seal"
    );

    // The views were taken before the frame: the batch's own buffers are what they show afterwards.
    let x = data(1);
    let ticket = frame(&mut batch, &h, Some(&x), 5);
    assert_eq!(ticket, reference(&x, 5));
    assert_eq!(
        read(&device, &y_view),
        ticket,
        "the stage view holds the frame's ticket"
    );
    assert_eq!(
        read(&device, &x_view),
        x,
        "the input view holds the frame's write"
    );
    assert_eq!(
        read(&device, &batch.view(&h.y).unwrap()),
        ticket,
        "a view taken after the frame shows the same buffer"
    );
}

#[test]
fn twin_a_two_dimensional_view_keeps_its_extents() {
    let device = Arc::new(CpuDevice::new());
    let (batch, (grid, flat)) = gpu_batch!("grid", on(&device); |b| {
        let grid = b.input::<u32>((8u32, 4u32));
        let flat = b.input::<u32>(5u32);
        (grid, flat)
    })
    .unwrap();
    let view = batch.view(&grid).unwrap();
    assert_eq!((view.width(), view.height(), view.len()), (8, 4, 32));
    let view = batch.view(&flat).unwrap();
    assert_eq!((view.width(), view.height(), view.len()), (5, 1, 5));
}

#[test]
fn twin_a_view_refuses_a_foreign_handle_and_one_the_block_did_not_return() {
    let device = Arc::new(CpuDevice::new());
    let mut leaked = None;
    let (batch, shown) = gpu_batch!("one", on(&device); |b| {
        let a = b.input::<u32>(4u32);
        let hidden = gpu_task!(4u32, |i| -> u32 { a[i] + 1u32 });
        let shown = gpu_task!(4u32, |i| -> u32 { hidden[i] * 2u32 });
        leaked = Some(hidden);
        shown
    })
    .unwrap();
    let hidden = leaked.unwrap();
    assert!(batch.view(&shown).is_ok());
    assert!(matches!(batch.view(&hidden), Err(Error::NotOutput { .. })));

    let (other, other_handle) =
        gpu_batch!("two", on(&device); |b| { let a = b.input::<u32>(4u32); a }).unwrap();
    assert!(matches!(
        batch.view(&other_handle),
        Err(Error::Foreign { .. })
    ));
    assert!(other.view(&other_handle).is_ok());
}

#[test]
fn twin_the_generation_is_equal_across_three_frames_and_with_two_in_flight() {
    let device = Arc::new(CpuDevice::new());
    for (depth, build) in [(1, build1 as fn(&_) -> _), (2, build2)] {
        let (mut batch, h) = build(&device).unwrap();
        let view = batch.view(&h.y).unwrap();
        let generation = batch.generation();
        assert_eq!(view.generation(), generation);
        let x = data(2);
        for (k, gain) in [3_u32, 4, 5].into_iter().enumerate() {
            let ticket = frame(&mut batch, &h, (k == 0).then_some(&x[..]), gain);
            assert_eq!(
                batch.generation(),
                generation,
                "frame {k}, in_flight({depth})"
            );
            assert_eq!(
                read(&device, &view),
                ticket,
                "frame {k}, in_flight({depth})"
            );
        }
        assert_eq!(batch.view(&h.y).unwrap().generation(), generation);
    }
}

#[test]
fn twin_a_view_is_after_encode_the_frame_and_before_it_the_frame_before() {
    let device = Arc::new(CpuDevice::new());
    let (mut batch, h) = build1(&device).unwrap();
    let view = batch.view(&h.y).unwrap();
    let x = data(3);
    let first = frame(&mut batch, &h, Some(&x), 2);

    // A read after `encode` but before the encoder runs stands in for the caller's own pass recorded before `encode`:
    // it sees the frame before. Once the encoder has run, the view shows this frame.
    let mut encoder = BatchDevice::new_encoder(&*device);
    let mut run = block_on(batch.begin()).unwrap();
    run.set(&h.gain, 9).unwrap();
    let ticket = run.ticket(&h.y).unwrap();
    let encoded = run.encode(&mut encoder).unwrap();
    let before = read(&device, &view);
    assert_eq!(
        before, first,
        "after `encode` and before the encoder runs, the buffer holds the frame before"
    );
    let job = encoder.run(&device).unwrap();
    let _submitted = encoded.submitted(&job);
    assert_eq!(read(&device, &view), block_on(ticket).unwrap());
    assert_eq!(read(&device, &view), reference(&x, 9));
}
