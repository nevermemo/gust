//! A `gpu_batch!` with two frames in flight, in a browser (m14.s2c, plan section 2.5). `cargo xtask check-web` builds the
//! crate under `GUST_PROFILE=web` (the strict profile: the stages' kernels are the WGSL artifacts committed in
//! `gust-artifacts/`, made by `cargo gust build -p web-batch-frames`), runs `wasm-bindgen --target web`, serves
//! `index.html` with the result from a loopback server, opens it in headless Chrome and reads the one line that the page
//! logs to the console:
//!
//! - `GUST_OK <vendor> <architecture> | frame time read | two frames in flight exact, wide stage refused on 8 and exact
//!   on 16` when the page recorded and submitted frame N+1 while frame N's ticket was still pending (nothing returns to
//!   the event loop in between, so the ticket cannot have resolved: that is checked, not guessed), a third `begin` was
//!   still waiting for frame N at that point, and then every frame's ticket, awaited, equalled the host computation
//!   exactly. The third frame's `begin`, awaited after the first two tickets, learned from the browser's
//!   `on_submitted_work_done` that the oldest frame had completed, and its ticket was exact too. The last part of the
//!   line is the wide stage
//!   (ten storage buffers): [`build_wide`] was refused by name on the default device (8), and on a second device opened
//!   with `with_storage_buffers(16)` it built and its result was exact. The middle part says which branch of the frame
//!   timing ran (m14.ftweb): the batch is built with `FrameTiming::Passes`, and the third frame asks for its GPU time. On
//!   an adapter where the browser granted `timestamp-query` the time resolved to a value with one entry per stage that
//!   has a pass ([`TIMED_STAGES`] of them, with distinct names, none longer than the span), a span no longer than the
//!   CPU wall time of that frame (plus the resolution of the clock), and the batch reported a 1 ns tick: `frame time
//!   read`. Where it did not, the time resolved to `None` and the batch reported no timing: `frame time absent`. The
//!   values are quantized by the browser (measured 65536 ns on headless Chrome 158, Radeon 890M only; RTX 5090
//!   pending). The two adapter words are what the browser says about its adapter (`GPUAdapter.info`; wgpu's own name is
//!   empty there);
//! - `GUST_FAIL <reason>` on any error or panic.
//!
//! The batch builder, [`build`], is compiled on the host as well, which is how `cargo gust build` finds the kernels;
//! everything that talks to the browser is `cfg(target_arch = "wasm32")` (the `page` module).

use std::sync::Arc;

use gust::batch::{Batch, D1, Error, FrameTiming, Param, Stage};
use gust::{gpu_batch, gpu_task};
use gust_wgpu::HeadlessDevice;

/// The elements of the input and of each frame's result: more than one workgroup and not a multiple of it.
pub const LENGTH: u32 = 1000;

/// The stages of [`build`] that have a pass, and so the entries of a frame's time at level `Passes` (the browser records
/// each stage in passes of its own): `y`, the in-place doubling of `y`, `z`, `mixed`, `rotated`, the reversal index
/// stage, `gathered` and the scatter stage (`_scatter`).
pub const TIMED_STAGES: usize = 8;

/// The element of the batch's struct stages (m14.s2s, D279): three words, so no padding, and exact `u32` arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, gust::GpuStruct)]
#[repr(C)]
pub struct Row {
    /// The first word.
    pub a: u32,
    /// The second word.
    pub b: u32,
    /// The third word.
    pub c: u32,
}

/// The handles of the batch: the gain, the input, and the stage whose values every frame's ticket reads.
pub struct Handles {
    /// Added to every element in the first stage.
    pub gain: Param<u32>,
    /// The input, written by a frame.
    pub x: Stage<u32, D1>,
    /// The result: `y = (x * 3 + gain) * 2 + 1`.
    pub y: Stage<u32, D1>,
    /// How many rows the counted stage runs, written by a frame (a count above [`LENGTH`] is clamped).
    pub count: Stage<u32, D1>,
    /// The counted stage's result: row `i` is `x[i] + 1` for the rows below the last count that reached it and keeps
    /// the value of the last frame that wrote it otherwise (zero before any).
    pub z: Stage<u32, D1>,
    /// A struct as the element of an input, written by a frame.
    pub rows: Stage<Row, D1>,
    /// A field read of `rows`: `a + b * 2`.
    pub mixed: Stage<u32, D1>,
    /// A whole row as a stage's result: the fields of `rows` rotated by one (`a, b, c` become `c, a, b`).
    pub rotated: Stage<Row, D1>,
    /// The rows of `rows` reversed, gathered by a counted stage through an index a stage built.
    pub gathered: Stage<Row, D1>,
    /// The rows of `rows` reversed, stored by a scatter stage (m14.s2sc) at the place `rev` names for each row.
    pub scattered: Stage<Row, D1>,
}

/// The batch: `y = x * 3 + gain`, then `y = y * 2 + 1` in place, with `in_flight(2)`. `u32` arithmetic on small values,
/// so the host computation ([`reference`]) is exact.
///
/// # Errors
///
/// The batch's build errors (an unusable device, a kernel without its artifact on the `web` profile).
///
/// # Panics
///
/// When `n` is not [`LENGTH`]: the stage that reverses the rows (`rev`) writes the index `999 - i`, which is `LENGTH - 1
/// - i` only for that length, and the kernels of the committed artifacts are made for it.
pub fn build(
    device: &Arc<HeadlessDevice>,
    n: u32,
) -> Result<(Batch<HeadlessDevice>, Handles), Error> {
    assert_eq!(
        n, LENGTH,
        "the reversal of `rev` is written for LENGTH rows"
    );
    gpu_batch!("frames", on(device); in_flight(2); |b| {
        // Pass boundaries only in a browser (WebGPU has no stamps inside passes); `Ok(None)` where it is not granted.
        b.frame_timing(FrameTiming::Passes);
        let gain = b.param::<u32>(1);
        let x = b.input::<u32>(n);
        let y = gpu_task!(n, |i| -> u32 { x[i] * 3u32 + gain.get() });
        let _double = gpu_task!(&mut y, |i| { y[i] = y[i] * 2u32 + 1u32; });
        let count = b.input::<u32>(1u32);
        // A counted stage (m14.s2i, D278): the browser runs the bounded launch, the one `wasm32` always takes.
        let z = gpu_task!(count(count); n, |i| -> u32 { x[i] + 1u32 });
        // A struct stored as the element (m14.s2s): the browser runs kernels that read a field and return a whole row.
        let rows = b.input::<Row>(n);
        let mixed = gpu_task!(n, |i| -> u32 { rows[i].a + rows[i].b * 2u32 });
        let rotated = gpu_task!(n, |i| -> Row { Row { a: rows[i].c, b: rows[i].a, c: rows[i].b } });
        // Whole rows through an index handle (m14.s2s2): a stage builds a reversal (the length is `LENGTH`), and a counted
        // stage gathers the rows through it. The ops of a batch (`compact`) run Slang kernels at run time, which a
        // browser build has not, so the index is a stage's here. `999` is `LENGTH - 1` (`build` checks `n`).
        let rev = gpu_task!(n, |i| -> u32 { 999u32 - i });
        let gathered = gpu_task!(count(count); n, |i| -> Row { rows[rev[i]] });
        // A scatter stage (m14.s2sc) stores each row at the place the same index names: the places are distinct, as the
        // stage declares. Its `let` is a unit value and names the stage in the twin's panic message.
        let scattered = b.input::<Row>(n);
        #[allow(clippy::let_unit_value)]
        let _scatter = gpu_task!(n, scatter(&mut scattered), |i| { scattered[rev[i]] = rows[i]; });
        Handles { gain, x, y, count, z, rows, mixed, rotated, gathered, scattered }
    })
}

/// The handles of the wide batch: its nine inputs and the stage that reads them all.
pub struct WideHandles {
    /// The inputs `a0` to `a8`, written by a frame.
    pub inputs: [Stage<u32, D1>; 9],
    /// The result: `out[i] = a0[i] * 1 + a1[i] * 2 + ... + a8[i] * 9`.
    pub out: Stage<u32, D1>,
}

/// A batch whose one stage binds ten storage buffers (nine inputs and its result), over the limit of eight that a
/// browser device opens with unless it is asked for more (m14.lim-b, D280; m14.lim-web): a default device refuses it
/// when it is built, naming the stage `out`, and a device opened with `with_storage_buffers(16)` builds and runs it.
///
/// # Errors
///
/// [`Error::StageDevice`] on a device that offers fewer than ten storage buffers per shader stage, which is every
/// browser device opened without the option.
// The stage is named by its `let`; its generated kernel function takes one argument per buffer.
#[allow(clippy::too_many_arguments)]
pub fn build_wide(
    device: &Arc<HeadlessDevice>,
) -> Result<(Batch<HeadlessDevice>, WideHandles), Error> {
    gpu_batch!("wide", on(device); in_flight(1); |b| {
        let a0 = b.input::<u32>(LENGTH);
        let a1 = b.input::<u32>(LENGTH);
        let a2 = b.input::<u32>(LENGTH);
        let a3 = b.input::<u32>(LENGTH);
        let a4 = b.input::<u32>(LENGTH);
        let a5 = b.input::<u32>(LENGTH);
        let a6 = b.input::<u32>(LENGTH);
        let a7 = b.input::<u32>(LENGTH);
        let a8 = b.input::<u32>(LENGTH);
        let out = gpu_task!(LENGTH, |i| -> u32 {
            a0[i] + a1[i] * 2u32 + a2[i] * 3u32 + a3[i] * 4u32 + a4[i] * 5u32 + a5[i] * 6u32
                + a6[i] * 7u32 + a7[i] * 8u32 + a8[i] * 9u32
        });
        WideHandles { inputs: [a0, a1, a2, a3, a4, a5, a6, a7, a8], out }
    })
}

/// Input `k` of the wide batch (`k` below nine): small values, so the weighted sum is exact in `u32`.
#[must_use]
pub fn wide_input(k: u32) -> Vec<u32> {
    (0..LENGTH).map(|i| i * 7 + k).collect()
}

/// The host computation of the wide stage: for each row, the sum over `k` of `(k + 1) * wide_input(k)[i]`.
#[must_use]
pub fn wide_reference() -> Vec<u32> {
    (0..LENGTH)
        .map(|i| (0..9u32).map(|k| (k + 1) * (i * 7 + k)).sum())
        .collect()
}

/// The host computation of one frame's result.
#[must_use]
pub fn reference(x: &[u32], gain: u32) -> Vec<u32> {
    x.iter().map(|v| (v * 3 + gain) * 2 + 1).collect()
}

/// The counted stage's result after a frame that wrote `x` and the count `count`: the first `min(count, LENGTH)` rows
/// are `x[i] + 1`, the others keep what `state` held (the frames share their buffers).
pub fn counted_reference(state: &mut [u32], x: &[u32], count: u32) {
    for (row, value) in state.iter_mut().zip(x).take(count as usize) {
        *row = value + 1;
    }
}

/// The rows of frame `seed`, and what the struct stages make of them: `a + b * 2` and the rotated rows.
#[must_use]
pub fn rows_of(seed: u32) -> Vec<Row> {
    (0..LENGTH)
        .map(|i| Row {
            a: i + seed,
            b: i * 3,
            c: i * 5 + seed,
        })
        .collect()
}

/// The values of the `mixed` stage for `rows`.
#[must_use]
pub fn mixed_reference(rows: &[Row]) -> Vec<u32> {
    rows.iter().map(|row| row.a + row.b * 2).collect()
}

/// The rows of the `rotated` stage for `rows`.
#[must_use]
pub fn rotated_reference(rows: &[Row]) -> Vec<Row> {
    rows.iter()
        .map(|row| Row {
            a: row.c,
            b: row.a,
            c: row.b,
        })
        .collect()
}

/// The rows of the `gathered` stage after the frame that wrote `rows` and the count `count`: the first
/// `min(count, LENGTH)` rows are the rows reversed, the others are zero rows (the frames before wrote no rows, so each
/// of their gathers moved zero rows, and the stage's buffer holds zeros past the count).
#[must_use]
pub fn gathered_reference(rows: &[Row], count: u32) -> Vec<Row> {
    let mut gathered: Vec<Row> = rows.iter().rev().copied().collect();
    gathered.truncate(count as usize);
    gathered.resize(rows.len(), Row { a: 0, b: 0, c: 0 });
    gathered
}

/// The rows of the scatter stage for `rows`: each row stored at the place `LENGTH - 1 - i`, so the rows reversed.
#[must_use]
pub fn scattered_reference(rows: &[Row]) -> Vec<Row> {
    rows.iter().rev().copied().collect()
}

/// The input of frame `seed`.
#[must_use]
pub fn input(seed: u32) -> Vec<u32> {
    (0..LENGTH).map(|i| i * 7 + seed).collect()
}

#[cfg(target_arch = "wasm32")]
mod page;
