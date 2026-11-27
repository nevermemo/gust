//! Whole struct rows moved by the indices of `compact` (m14.s2s2): a stage reads the flags from a field, `gust::compact`
//! gives `(idx, n)`, and a counted stage gathers the rows, `gpu_task!(count(n); CAP, |i| -> Row { rows[idx[i]] })`. There
//! is no library op over rows: `compact` is an op over `u32` flags, and the gather is a stage that reads a struct at an
//! index taken from another handle. The stage binds the rows, the indices, the count and its result, four storage
//! buffers whatever the number of fields.
//!
//! A counted result stage writes only the rows below its count; the others keep what the previous frame left (zero in
//! every byte before the first frame and after a resize), so every test compares the whole buffer with a host model that
//! carries the buffer from frame to frame. Every value is small and exactly representable, so a result equals the model
//! bit for bit. The `twin_` tests run on the CPU twin; the `gpu_` ones are `#[ignore]`d like the example's other GPU
//! tests, print the adapter and run in full verification on the adapter the environment selects, each under both
//! counted launches (the indirect one and the worst-case grid, D277).

use std::fmt::Debug;
use std::sync::Arc;

use batch_basics::app::block_on;
use gust::batch::{
    Batch, BatchDevice, CountedLaunch, D1, Element, Error, Param, Stage, set_counted_launch,
};
use gust::cpu::CpuDevice;
use gust::map::MapHost;
use gust::prelude::{Mat4, Vec2, Vec4};
use gust::{GpuPod, gpu_batch, gpu_task};
use gust_wgpu::HeadlessDevice;

mod rows {
    /// Three words of two kinds, so that a field read at the wrong offset reads the wrong kind.
    #[derive(Clone, Copy, Debug, PartialEq, gust::GpuStruct)]
    #[repr(C)]
    pub struct Row {
        pub life: f32,
        pub flags: u32,
        pub tag: u32,
    }
}

use rows::Row;

const CAP: u32 = 1000;
const LAUNCHES: [CountedLaunch; 2] = [CountedLaunch::Indirect, CountedLaunch::Bounded];

/// Forces a launch for the batches this thread builds until it is dropped.
struct Launch;

impl Launch {
    fn set(launch: CountedLaunch) -> Self {
        set_counted_launch(Some(launch));
        Self
    }
}

impl Drop for Launch {
    fn drop(&mut self) {
        set_counted_launch(None);
    }
}

fn private_device() -> Arc<HeadlessDevice> {
    let device = Arc::new(HeadlessDevice::new().unwrap_or_else(|error| {
        panic!("a GPU adapter is required; this test must not silently skip: {error}")
    }));
    println!("adapter: {}", device.adapter_info().name);
    device
}

const ZERO: Row = Row {
    life: 0.0,
    flags: 0,
    tag: 0,
};

/// The rows of frame `k`: a row is one the flags select (`flags % 3 == 0`) exactly when `mask` says so.
fn rows_for(k: u32, mask: &[bool]) -> Vec<Row> {
    (0..mask.len() as u32)
        .map(|i| Row {
            life: i as f32 * 0.25 + k as f32,
            flags: i * 3 + u32::from(!mask[i as usize]),
            tag: 1000 + i,
        })
        .collect()
}

fn selected(input: &[Row]) -> Vec<Row> {
    input
        .iter()
        .filter(|row| row.flags % 3 == 0)
        .copied()
        .collect()
}

/// A share of the rows, going 0, 1/12, .. 12/12 and back, moving along the rows every frame: the count changes every
/// frame and passes through 0 and through the capacity.
fn share_mask(k: u32) -> Vec<bool> {
    let t = k % 24;
    let thresh = t.min(24 - t);
    (0..CAP).map(|i| (i + k * 3) % 12 < thresh).collect()
}

fn share_masks() -> Vec<Vec<bool>> {
    (0..25).map(share_mask).collect()
}

/// The edges: every row, none, the last one, all but one in the middle, every row, the first one, none.
fn edge_masks() -> Vec<Vec<bool>> {
    let all = vec![true; CAP as usize];
    let none = vec![false; CAP as usize];
    let mut last = none.clone();
    last[CAP as usize - 1] = true;
    let mut first = none.clone();
    first[0] = true;
    let mut all_but_one = all.clone();
    all_but_one[500] = false;
    vec![
        all.clone(),
        none.clone(),
        last,
        all_but_one,
        all,
        first,
        none,
    ]
}

// ---- the batch: flags from a field, compact, and the gathers ----

#[derive(Clone, Copy)]
struct Handles {
    rows: Stage<Row, D1>,
    n: Stage<u32, D1>,
    dense: Stage<Row, D1>,
    shifted: Stage<Row, D1>,
    guarded: Stage<Row, D1>,
}

macro_rules! define_compact {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles), Error> {
            gpu_batch!("struct_compact", on(device); in_flight(1); |b| {
                let rows = b.input::<Row>(CAP);
                let alive = gpu_task!(CAP, |i| -> u32 {
                    let v: u32 = if rows[i].flags % 3u32 == 0u32 { 1u32 } else { 0u32 };
                    v
                });
                let (idx, n) = gust::compact(&alive);
                let dense = gpu_task!(count(n); CAP, |i| -> Row { rows[idx[i]] });
                let shifted = gpu_task!(count(n); CAP, |i| -> Row {
                    Row {
                        life: rows[idx[i]].life * 0.5,
                        flags: rows[idx[i]].flags,
                        tag: rows[idx[i]].tag,
                    }
                });
                let guarded = gpu_task!(CAP, |i| -> Row {
                    let mut v: Row = rows[i];
                    if i < n[0u32] {
                        v = rows[idx[i]];
                    }
                    v
                });
                Handles { rows, n, dense, shifted, guarded }
            })
        }
    };
}

define_compact!(build_cpu, CpuDevice);
define_compact!(build_gpu, HeadlessDevice);

/// What a frame reads: the count, the counted gather, the counted field-built gather and the guarded gather.
type Out = (u32, Vec<Row>, Vec<Row>, Vec<Row>);

fn frame<D: BatchDevice>(batch: &mut Batch<D>, h: &Handles, input: &[Row]) -> Out
where
    D::Buffer<u32>: 'static,
    D::Buffer<Row>: 'static,
{
    block_on(async {
        let mut frame = batch.begin().await?;
        frame.write(&h.rows, input)?;
        let n = frame.ticket(&h.n)?;
        let dense = frame.ticket(&h.dense)?;
        let shifted = frame.ticket(&h.shifted)?;
        let guarded = frame.ticket(&h.guarded)?;
        let _submitted = frame.submit()?;
        let n = n.await?;
        assert_eq!(n.len(), 1, "the count is one element");
        Ok::<_, Error>((n[0], dense.await?, shifted.await?, guarded.await?))
    })
    .unwrap()
}

/// The host model: the counted results carry their rows from frame to frame, the guarded one is written whole.
struct Model {
    dense: Vec<Row>,
    shifted: Vec<Row>,
}

impl Model {
    fn new() -> Self {
        Self {
            dense: vec![ZERO; CAP as usize],
            shifted: vec![ZERO; CAP as usize],
        }
    }

    fn step(&mut self, input: &[Row]) -> Out {
        let picked = selected(input);
        let m = picked.len();
        self.dense[..m].copy_from_slice(&picked);
        for (slot, row) in self.shifted.iter_mut().zip(&picked) {
            *slot = Row {
                life: row.life * 0.5,
                ..*row
            };
        }
        let guarded = (0..input.len())
            .map(|i| if i < m { picked[i] } else { input[i] })
            .collect();
        (m as u32, self.dense.clone(), self.shifted.clone(), guarded)
    }
}

/// Every frame of `masks` equal to the model over the whole buffers; returns what the frames read.
fn frames_equal_the_model<D: BatchDevice>(
    mut batch: Batch<D>,
    h: &Handles,
    masks: &[Vec<bool>],
    what: &str,
) -> Vec<Out>
where
    D::Buffer<u32>: 'static,
    D::Buffer<Row>: 'static,
{
    let mut model = Model::new();
    masks
        .iter()
        .enumerate()
        .map(|(k, mask)| {
            let input = rows_for(k as u32, mask);
            let got = frame(&mut batch, h, &input);
            let want = model.step(&input);
            assert_eq!(got.0, want.0, "{what}: the count, frame {k}");
            assert_eq!(got.1, want.1, "{what}: the gathered rows, frame {k}");
            assert_eq!(got.2, want.2, "{what}: the field-built rows, frame {k}");
            assert_eq!(got.3, want.3, "{what}: the guarded gather, frame {k}");
            got
        })
        .collect()
}

fn twin_frames(masks: &[Vec<bool>]) -> Vec<Out> {
    let device = Arc::new(CpuDevice::new());
    let (batch, h) = build_cpu(&device).unwrap();
    frames_equal_the_model(batch, &h, masks, "twin")
}

/// The GPU's frames equal the model and the twin's frames under each counted launch.
fn gpu_frames(masks: &[Vec<bool>]) {
    let device = private_device();
    let twin = twin_frames(masks);
    for launch in LAUNCHES {
        let _launch = Launch::set(launch);
        let (batch, h) = build_gpu(&device).unwrap();
        let got = frames_equal_the_model(batch, &h, masks, &format!("{launch:?}"));
        assert_eq!(got, twin, "{launch:?} against the twin");
    }
}

#[test]
fn twin_compact_gathers_whole_struct_rows() {
    let outs = twin_frames(&share_masks());
    let counts: Vec<u32> = outs.iter().map(|out| out.0).collect();
    assert_eq!(counts[0], 0, "the count passes through zero");
    assert_eq!(counts[12], CAP, "and through the capacity");
    assert_eq!(
        counts[1], 83,
        "the rows with `i % 12 == 9`, a twelfth of them"
    );
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_compact_gathers_whole_struct_rows() {
    gpu_frames(&share_masks());
}

#[test]
fn twin_compact_rows_at_the_edges() {
    let outs = twin_frames(&edge_masks());
    let counts: Vec<u32> = outs.iter().map(|out| out.0).collect();
    assert_eq!(counts, [CAP, 0, 1, CAP - 1, CAP, 1, 0]);
    // The row of the count of one is the last one, found through idx[0].
    assert_eq!(outs[2].1[0].tag, 1000 + CAP - 1);
    // A count of zero writes nothing: the buffers are what the frame before left.
    assert_eq!(outs[1].1, outs[0].1);
    assert_eq!(outs[1].2, outs[0].2);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_compact_rows_at_the_edges() {
    gpu_frames(&edge_masks());
}

// ---- the view of a gathered buffer ----

#[test]
fn twin_compact_rows_through_a_view() {
    let device = Arc::new(CpuDevice::new());
    let (mut batch, h) = build_cpu(&device).unwrap();
    let view = batch.view(&h.dense).unwrap();
    assert_eq!(view.len(), CAP as usize);
    let mask = share_mask(5);
    let out = frame(&mut batch, &h, &rows_for(5, &mask));
    assert_eq!(device.readback(view.buffer()).unwrap(), out.1);
    assert_eq!(out.0 as usize, mask.iter().filter(|set| **set).count());
}

// ---- layouts with pad fields ----

/// The handles of a layout's batch: the flags, the rows, the count and the gathered rows.
type Layout<'a, T> = (
    &'a Stage<u32, D1>,
    &'a Stage<T, D1>,
    &'a Stage<u32, D1>,
    &'a Stage<T, D1>,
);

/// The frames of one layout: the rows written, flags set from `mask`, the count and the gathered rows read. Rows have no
/// `PartialEq` here (`Mat4`), so they are compared as `Debug` text, which is bit equality for these exact values.
fn layout_frames<D, T>(
    batch: &mut Batch<D>,
    stages: Layout<'_, T>,
    input: &dyn Fn(u32) -> Vec<T>,
    masks: &[Vec<bool>],
    what: &str,
) -> Vec<(u32, String)>
where
    D: BatchDevice,
    T: Element + GpuPod + Send + Copy + Debug,
    T::Word: Send,
    D::Buffer<u32>: 'static,
    D::Buffer<T>: 'static,
{
    let (flags, rows, n, dense) = stages;
    let mut state: Vec<T> = Vec::new();
    masks
        .iter()
        .enumerate()
        .map(|(k, mask)| {
            let data = input(k as u32);
            let set: Vec<u32> = mask.iter().map(|on| u32::from(*on)).collect();
            let (count, got) = block_on(async {
                let mut frame = batch.begin().await?;
                frame.write(flags, &set)?;
                frame.write(rows, &data)?;
                let n = frame.ticket(n)?;
                let dense = frame.ticket(dense)?;
                let _submitted = frame.submit()?;
                Ok::<_, Error>((n.await?[0], dense.await?))
            })
            .unwrap();
            let picked: Vec<T> = (0..data.len())
                .filter(|i| mask[*i])
                .map(|i| data[i])
                .collect();
            if state.is_empty() {
                state = data.clone();
            }
            state[..picked.len()].copy_from_slice(&picked);
            assert_eq!(count as usize, picked.len(), "{what}: the count, frame {k}");
            let got = format!("{got:?}");
            assert_eq!(got, format!("{state:?}"), "{what}: the rows, frame {k}");
            (count, got)
        })
        .collect()
}

const LCAP: u32 = 40;

/// Every row, a third of them, none, six sevenths, the last one only.
fn layout_masks() -> Vec<Vec<bool>> {
    let at = |keep: &dyn Fn(u32) -> bool| (0..LCAP).map(keep).collect::<Vec<_>>();
    vec![
        at(&|_| true),
        at(&|i| i % 3 == 0),
        at(&|_| false),
        at(&|i| i % 7 != 0),
        at(&|i| i == LCAP - 1),
    ]
}

/// A `Vec4`, a `Vec2` and two pad words: 32 bytes, the `Vec2` at 16, the pads at 24 and 28.
mod wide {
    use super::{
        Arc, Batch, CpuDevice, D1, Error, HeadlessDevice, LCAP, Stage, Vec2, Vec4, gpu_batch,
        gpu_task,
    };

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
        pub flags: Stage<u32, D1>,
        pub rows: Stage<Row, D1>,
        pub n: Stage<u32, D1>,
        pub dense: Stage<Row, D1>,
    }

    macro_rules! define_build {
        ($build:ident, $device:ty) => {
            pub fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles), Error> {
                gpu_batch!("compact_wide", on(device); in_flight(1); |b| {
                    let flags = b.input::<u32>(LCAP);
                    let rows = b.input::<Row>(LCAP);
                    let (idx, n) = gust::compact(&flags);
                    let dense = gpu_task!(count(n); LCAP, |i| -> Row { rows[idx[i]] });
                    Handles { flags, rows, n, dense }
                })
            }
        };
    }
    define_build!(build_cpu, CpuDevice);
    define_build!(build_gpu, HeadlessDevice);

    pub fn input(k: u32) -> Vec<Row> {
        (0..LCAP)
            .map(|i| Row {
                a: Vec4::new(i as f32, (i + k) as f32, 0.25, -1.0),
                b: Vec2::new(0.5 * i as f32, (k + 2 * i) as f32),
                p0: (i + 11) as f32,
                p1: (k * 5 + i) as f32 * 2.0,
            })
            .collect()
    }
}

/// A `Mat4` and four scalar words: 80 bytes, the scalars at 64, 68, 72 and 76.
mod matrix {
    use super::{
        Arc, Batch, CpuDevice, D1, Error, HeadlessDevice, LCAP, Mat4, Stage, Vec4, gpu_batch,
        gpu_task,
    };

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
        pub flags: Stage<u32, D1>,
        pub rows: Stage<Row, D1>,
        pub n: Stage<u32, D1>,
        pub dense: Stage<Row, D1>,
    }

    macro_rules! define_build {
        ($build:ident, $device:ty) => {
            pub fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles), Error> {
                gpu_batch!("compact_matrix", on(device); in_flight(1); |b| {
                    let flags = b.input::<u32>(LCAP);
                    let rows = b.input::<Row>(LCAP);
                    let (idx, n) = gust::compact(&flags);
                    let dense = gpu_task!(count(n); LCAP, |i| -> Row { rows[idx[i]] });
                    Handles { flags, rows, n, dense }
                })
            }
        };
    }
    define_build!(build_cpu, CpuDevice);
    define_build!(build_gpu, HeadlessDevice);

    pub fn input(k: u32) -> Vec<Row> {
        (0..LCAP)
            .map(|i| {
                let x = (i + k) as f32;
                Row {
                    m: Mat4::from_cols(
                        Vec4::new(x, 1.0, 2.0, 3.0),
                        Vec4::new(4.0, x + 1.0, 5.0, 6.0),
                        Vec4::new(7.0, 8.0, x + 2.0, 9.0),
                        Vec4::new(10.0, 11.0, 12.0, x + 3.0),
                    ),
                    a: 0.5 * x,
                    b: 1000 + i,
                    c: i as i32 - 20,
                    p: 7 * k + i,
                }
            })
            .collect()
    }
}

macro_rules! layout_runs {
    ($twin:ident, $gpu:ident, $layout:ident) => {
        fn $twin() -> Vec<(u32, String)> {
            let (mut batch, h) = $layout::build_cpu(&Arc::new(CpuDevice::new())).unwrap();
            let stages = (&h.flags, &h.rows, &h.n, &h.dense);
            layout_frames(&mut batch, stages, &$layout::input, &layout_masks(), "twin")
        }

        fn $gpu() {
            let device = private_device();
            let twin = $twin();
            for launch in LAUNCHES {
                let _launch = Launch::set(launch);
                let (mut batch, h) = $layout::build_gpu(&device).unwrap();
                let stages = (&h.flags, &h.rows, &h.n, &h.dense);
                let what = format!("{launch:?}");
                let got =
                    layout_frames(&mut batch, stages, &$layout::input, &layout_masks(), &what);
                assert_eq!(got, twin, "{what} against the twin");
            }
        }
    };
}

layout_runs!(wide_on_the_twin, wide_on_the_gpu, wide);
layout_runs!(matrix_on_the_twin, matrix_on_the_gpu, matrix);

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_compact_rows_with_a_vec4_a_vec2_and_two_pads() {
    wide_on_the_gpu();
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_compact_rows_with_a_mat4_and_scalars() {
    matrix_on_the_gpu();
}

#[test]
fn twin_compact_rows_with_pad_fields() {
    assert_eq!(wide_on_the_twin().len(), 5);
    assert_eq!(matrix_on_the_twin().len(), 5);
    let counts: Vec<u32> = wide_on_the_twin().iter().map(|frame| frame.0).collect();
    assert_eq!(counts, [LCAP, 14, 0, 34, 1]);
}

// ---- four bindings whatever the fields ----

/// Ten words in five `Vec2`: in columns the same gather would bind the indices, the count, ten source columns and ten
/// result columns.
mod ten {
    use super::{
        Arc, Batch, CpuDevice, D1, Error, HeadlessDevice, LCAP, Stage, Vec2, gpu_batch, gpu_task,
    };

    #[derive(Clone, Copy, Debug, gust::GpuStruct)]
    #[repr(C)]
    pub struct Row {
        pub a: Vec2,
        pub b: Vec2,
        pub c: Vec2,
        pub d: Vec2,
        pub e: Vec2,
    }

    #[derive(Clone, Copy)]
    pub struct Handles {
        pub flags: Stage<u32, D1>,
        pub rows: Stage<Row, D1>,
        pub n: Stage<u32, D1>,
        pub dense: Stage<Row, D1>,
    }

    macro_rules! define_build {
        ($build:ident, $device:ty) => {
            pub fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Handles), Error> {
                gpu_batch!("compact_ten", on(device); in_flight(1); |b| {
                    let flags = b.input::<u32>(LCAP);
                    let rows = b.input::<Row>(LCAP);
                    let (idx, n) = gust::compact(&flags);
                    let dense = gpu_task!(count(n); LCAP, |i| -> Row { rows[idx[i]] });
                    Handles { flags, rows, n, dense }
                })
            }
        };
    }
    define_build!(build_cpu, CpuDevice);
    define_build!(build_gpu, HeadlessDevice);

    pub fn input(k: u32) -> Vec<Row> {
        (0..LCAP)
            .map(|i| {
                let x = (i * 10 + k) as f32;
                Row {
                    a: Vec2::new(x, x + 1.0),
                    b: Vec2::new(x + 2.0, x + 3.0),
                    c: Vec2::new(x + 4.0, x + 5.0),
                    d: Vec2::new(x + 6.0, x + 7.0),
                    e: Vec2::new(x + 8.0, x + 9.0),
                }
            })
            .collect()
    }
}

fn ten_frames<D: BatchDevice>(
    batch: &mut Batch<D>,
    h: &ten::Handles,
    what: &str,
) -> Vec<(u32, String)>
where
    D::Buffer<u32>: 'static,
    D::Buffer<ten::Row>: 'static,
{
    let stages = (&h.flags, &h.rows, &h.n, &h.dense);
    layout_frames(batch, stages, &ten::input, &layout_masks(), what)
}

#[test]
fn twin_compact_gather_uses_four_bindings_whatever_the_fields() {
    let (mut batch, h) = ten::build_cpu(&Arc::new(CpuDevice::new())).unwrap();
    assert_eq!(ten_frames(&mut batch, &h, "twin").len(), 5);
}

/// A device whose limit of storage buffers per shader stage is lowered to `buffers` (`with_limits` may only lower).
fn device_with(buffers: u32) -> Arc<HeadlessDevice> {
    let opened = HeadlessDevice::new().expect("a GPU adapter is required");
    let mut limits = opened.limits().clone();
    limits.max_storage_buffers_per_shader_stage = buffers;
    Arc::new(
        opened
            .with_limits(limits)
            .expect("a lowered limit is accepted"),
    )
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_compact_gather_uses_four_bindings_whatever_the_fields() {
    let (mut twin, th) = ten::build_cpu(&Arc::new(CpuDevice::new())).unwrap();
    let want = ten_frames(&mut twin, &th, "twin");
    // A device limited to four storage buffers builds and runs the gather of ten-word rows: the rows, the indices,
    // the count and the result.
    let device = device_with(4);
    println!("adapter: {}", device.adapter_info().name);
    for launch in LAUNCHES {
        let _launch = Launch::set(launch);
        let (mut batch, h) = ten::build_gpu(&device).unwrap();
        assert_eq!(
            ten_frames(&mut batch, &h, &format!("{launch:?}")),
            want,
            "{launch:?}"
        );
    }
    // With three it is refused at build, asking for four.
    let Err(error) = ten::build_gpu(&device_with(3)) else {
        panic!(
            "the gather binds four storage buffers: it must not build on a device limited to three"
        )
    };
    let Error::StageDevice {
        batch,
        stage,
        source,
    } = &error
    else {
        panic!("a stage the device refuses at build is `StageDevice`: {error:?}")
    };
    // The refusal names the gathering stage, not another stage of the batch.
    assert_eq!(
        (batch.as_str(), stage.as_str()),
        ("compact_ten", "dense"),
        "the refused stage is the gather: {error}"
    );
    match source.downcast_ref::<gust_wgpu::Error>() {
        Some(gust_wgpu::Error::DeviceLimit {
            limit,
            requested,
            maximum,
            ..
        }) => assert_eq!(
            (*limit, *requested, *maximum),
            ("max_storage_buffers_per_shader_stage", 4, 3)
        ),
        other => panic!("the source of the refusal is a `DeviceLimit`: {other:?}"),
    }
}

// ---- an in-place stage over the gathered rows ----

#[derive(Clone, Copy)]
struct Bumped {
    flags: Stage<u32, D1>,
    rows: Stage<Row, D1>,
    n: Stage<u32, D1>,
    dense: Stage<Row, D1>,
}

macro_rules! define_bumped {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Bumped), Error> {
            gpu_batch!("compact_bumped", on(device); in_flight(1); |b| {
                let flags = b.input::<u32>(CAP);
                let rows = b.input::<Row>(CAP);
                let (idx, n) = gust::compact(&flags);
                let dense = gpu_task!(count(n); CAP, |i| -> Row { rows[idx[i]] });
                let _bump = gpu_task!(count(n); &mut dense, |i| {
                    dense[i].life += 1.0;
                    dense[i].tag = i;
                });
                Bumped { flags, rows, n, dense }
            })
        }
    };
}

define_bumped!(bumped_cpu, CpuDevice);
define_bumped!(bumped_gpu, HeadlessDevice);

/// The gathered rows, then a field store and a compound over the same count: the rows below the count are the bumped
/// gathered ones, the rows from the count on keep what the earlier frames left.
fn bumped_frames<D: BatchDevice>(mut batch: Batch<D>, h: &Bumped, what: &str) -> Vec<Vec<Row>>
where
    D::Buffer<u32>: 'static,
    D::Buffer<Row>: 'static,
{
    let masks = [
        vec![true; CAP as usize],
        share_mask(5),
        vec![false; CAP as usize],
        share_mask(14),
    ];
    let mut state = vec![ZERO; CAP as usize];
    masks
        .iter()
        .enumerate()
        .map(|(k, mask)| {
            let input = rows_for(k as u32, mask);
            let set: Vec<u32> = mask.iter().map(|on| u32::from(*on)).collect();
            let (count, got) = block_on(async {
                let mut frame = batch.begin().await?;
                frame.write(&h.flags, &set)?;
                frame.write(&h.rows, &input)?;
                let n = frame.ticket(&h.n)?;
                let dense = frame.ticket(&h.dense)?;
                let _submitted = frame.submit()?;
                Ok::<_, Error>((n.await?[0], dense.await?))
            })
            .unwrap();
            let picked: Vec<Row> = (0..CAP as usize)
                .filter(|i| mask[*i])
                .map(|i| input[i])
                .collect();
            for (i, row) in picked.iter().enumerate() {
                state[i] = Row {
                    life: row.life + 1.0,
                    flags: row.flags,
                    tag: i as u32,
                };
            }
            assert_eq!(count as usize, picked.len(), "{what}: the count, frame {k}");
            assert_eq!(got, state, "{what}: the rows, frame {k}");
            got
        })
        .collect()
}

#[test]
fn twin_compact_rows_with_an_in_place_consumer() {
    let (batch, h) = bumped_cpu(&Arc::new(CpuDevice::new())).unwrap();
    let outs = bumped_frames(batch, &h, "twin");
    assert_eq!(
        outs[2], outs[1],
        "a count of zero writes no row of either stage"
    );
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_compact_rows_with_an_in_place_consumer() {
    let device = private_device();
    let (twin, th) = bumped_cpu(&Arc::new(CpuDevice::new())).unwrap();
    let want = bumped_frames(twin, &th, "twin");
    for launch in LAUNCHES {
        let _launch = Launch::set(launch);
        let (batch, h) = bumped_gpu(&device).unwrap();
        let what = format!("{launch:?}");
        assert_eq!(
            bumped_frames(batch, &h, &what),
            want,
            "{what} against the twin"
        );
    }
}

// ---- a resize of the rows ----

#[derive(Clone, Copy)]
struct Resized {
    len: Param<u32>,
    flags: Stage<u32, D1>,
    rows: Stage<Row, D1>,
    n: Stage<u32, D1>,
    dense: Stage<Row, D1>,
}

macro_rules! define_resized {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Resized), Error> {
            gpu_batch!("compact_resize", on(device); in_flight(1); |b| {
                let len = b.param::<u32>(8);
                let flags = b.input::<u32>(&len);
                let rows = b.input::<Row>(&len);
                let (idx, n) = gust::compact(&flags);
                let dense = gpu_task!(count(n); &len, |i| -> Row { rows[idx[i]] });
                Resized { len, flags, rows, n, dense }
            })
        }
    };
}

define_resized!(resized_cpu, CpuDevice);
define_resized!(resized_gpu, HeadlessDevice);

/// A grow, a shrink and a grow with no write: a resized buffer is zero in every byte (so the flags select nothing and
/// the gather writes nothing), and while the length stays the buffer carries its rows. `step` is the new length, if any,
/// and the mask of the rows written, if any.
fn a_resize_zero_fills_and_the_count_bounds_the_gather<D: BatchDevice>(
    mut batch: Batch<D>,
    h: &Resized,
    what: &str,
) where
    D::Buffer<u32>: 'static,
    D::Buffer<Row>: 'static,
{
    type Step = (Option<u32>, Option<Box<dyn Fn(u32) -> bool>>);
    let steps: Vec<Step> = vec![
        (None, Some(Box::new(|i| i % 2 == 0))),
        (Some(20), Some(Box::new(|i| i % 3 == 0))),
        (Some(5), Some(Box::new(|_| true))),
        (Some(11), None),
        (None, Some(Box::new(|i| i % 4 == 0))),
        (None, Some(Box::new(|_| false))),
    ];
    let mut len = 8_u32;
    let mut state = vec![ZERO; len as usize];
    for (k, (new_len, mask)) in steps.iter().enumerate() {
        if let Some(new_len) = new_len {
            len = *new_len;
            state = vec![ZERO; len as usize];
        }
        let written = mask.as_ref().map(|keep| {
            let mask: Vec<bool> = (0..len).map(keep).collect();
            rows_for(k as u32, &mask)
        });
        let (count, got) = block_on(async {
            let mut frame = batch.begin().await?;
            if let Some(new_len) = new_len {
                frame.set(&h.len, *new_len)?;
            }
            if let Some(input) = &written {
                let set: Vec<u32> = input
                    .iter()
                    .map(|row| u32::from(row.flags % 3 == 0))
                    .collect();
                frame.write(&h.flags, &set)?;
                frame.write(&h.rows, input)?;
            }
            let n = frame.ticket(&h.n)?;
            let dense = frame.ticket(&h.dense)?;
            let _submitted = frame.submit()?;
            Ok::<_, Error>((n.await?[0], dense.await?))
        })
        .unwrap();
        // A frame that writes nothing after a resize sees zero flags: the count is zero.
        let picked = written.as_deref().map(selected).unwrap_or_default();
        state[..picked.len()].copy_from_slice(&picked);
        assert_eq!(count as usize, picked.len(), "{what}: the count, step {k}");
        assert_eq!(got, state, "{what}: the rows, step {k}");
    }
}

#[test]
fn twin_compact_rows_follow_a_resize() {
    let (batch, h) = resized_cpu(&Arc::new(CpuDevice::new())).unwrap();
    a_resize_zero_fills_and_the_count_bounds_the_gather(batch, &h, "twin");
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_compact_rows_follow_a_resize() {
    let device = private_device();
    for launch in LAUNCHES {
        let _launch = Launch::set(launch);
        let (batch, h) = resized_gpu(&device).unwrap();
        a_resize_zero_fills_and_the_count_bounds_the_gather(batch, &h, &format!("{launch:?}"));
    }
}

// ---- an index handle built by a stage ----

const M: u32 = 64;

#[derive(Clone, Copy)]
struct Permuted {
    rows: Stage<Row, D1>,
    reversed: Stage<Row, D1>,
    order: Stage<u32, D1>,
    sorted: Stage<Row, D1>,
}

macro_rules! define_permuted {
    ($build:ident, $device:ty) => {
        fn $build(device: &Arc<$device>) -> Result<(Batch<$device>, Permuted), Error> {
            gpu_batch!("permuted", on(device); in_flight(1); |b| {
                let rows = b.input::<Row>(M);
                // A reversal: the index handle is a stage, not the output of `compact`.
                let rev = gpu_task!(M, |i| -> u32 { 63u32 - i });
                let reversed = gpu_task!(M, |i| -> Row { rows[rev[i]] });
                // The order of a stable counting sort by `flags % 4`: the rank of every row, then the row of every rank.
                let rank = gpu_task!(M, |j| -> u32 {
                    let mut r = 0u32;
                    for m in 0u32..64u32 {
                        if rows[m].flags % 4u32 < rows[j].flags % 4u32 {
                            r += 1u32;
                        }
                        if rows[m].flags % 4u32 == rows[j].flags % 4u32 && m < j {
                            r += 1u32;
                        }
                    }
                    r
                });
                let order = gpu_task!(M, |i| -> u32 {
                    let mut found = 0u32;
                    for j in 0u32..64u32 {
                        if rank[j] == i {
                            found = j;
                        }
                    }
                    found
                });
                let sorted = gpu_task!(M, |i| -> Row { rows[order[i]] });
                Permuted { rows, reversed, order, sorted }
            })
        }
    };
}

define_permuted!(permuted_cpu, CpuDevice);
define_permuted!(permuted_gpu, HeadlessDevice);

fn permuted_rows(seed: u32) -> Vec<Row> {
    (0..M)
        .map(|i| Row {
            life: (i * 3 + seed) as f32 * 0.25,
            flags: (i * 7 + seed * 5) % 13,
            tag: 1000 + i,
        })
        .collect()
}

/// A reversal and the order of a stable counting sort, each equal to the host over 16 frames.
fn rows_are_gathered_through_a_permutation<D: BatchDevice>(mut batch: Batch<D>, h: &Permuted)
where
    D::Buffer<u32>: 'static,
    D::Buffer<Row>: 'static,
{
    for seed in 0..16 {
        let input = permuted_rows(seed);
        let (reversed, order, sorted) = block_on(async {
            let mut frame = batch.begin().await?;
            frame.write(&h.rows, &input)?;
            let reversed = frame.ticket(&h.reversed)?;
            let order = frame.ticket(&h.order)?;
            let sorted = frame.ticket(&h.sorted)?;
            let _submitted = frame.submit()?;
            Ok::<_, Error>((reversed.await?, order.await?, sorted.await?))
        })
        .unwrap();
        let mut want_order: Vec<u32> = (0..M).collect();
        want_order.sort_by_key(|i| input[*i as usize].flags % 4);
        assert_eq!(
            order, want_order,
            "the order of the counting sort, frame {seed}"
        );
        assert_eq!(
            reversed,
            input.iter().rev().copied().collect::<Vec<_>>(),
            "the reversal, frame {seed}"
        );
        assert_eq!(
            sorted,
            want_order
                .iter()
                .map(|i| input[*i as usize])
                .collect::<Vec<_>>(),
            "the rows in the order of the sort, frame {seed}"
        );
    }
}

#[test]
fn twin_gather_rows_through_a_stage_built_permutation() {
    let (batch, h) = permuted_cpu(&Arc::new(CpuDevice::new())).unwrap();
    rows_are_gathered_through_a_permutation(batch, &h);
}

#[test]
#[ignore = "example validation runs only in full verification"]
fn gpu_gather_rows_through_a_stage_built_permutation() {
    let device = private_device();
    let (batch, h) = permuted_gpu(&device).unwrap();
    rows_are_gathered_through_a_permutation(batch, &h);
}
