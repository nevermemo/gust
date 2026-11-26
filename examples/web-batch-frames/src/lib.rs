//! A `gpu_batch!` with two frames in flight, in a browser (m14.s2c, plan section 2.5). `cargo xtask check-web` builds the
//! crate under `GUST_PROFILE=web` (the strict profile: the stages' kernels are the WGSL artifacts committed in
//! `gust-artifacts/`, made by `cargo gust build -p web-batch-frames`), runs `wasm-bindgen --target web`, serves
//! `index.html` with the result from a loopback server, opens it in headless Chrome and reads the one line that the page
//! logs to the console:
//!
//! - `GUST_OK <vendor> <architecture> | two frames in flight exact` when the page recorded and submitted frame N+1
//!   while frame N's ticket was still pending (nothing returns to the event loop in between, so the ticket cannot have
//!   resolved: that is checked, not guessed), a third `begin` was still waiting for frame N at that point, and then
//!   every frame's ticket, awaited, equalled the host computation exactly. The third frame's `begin`, awaited after the
//!   first two tickets, learned from the browser's `on_submitted_work_done` that the oldest frame had completed, and
//!   its ticket was exact too. The two words are what the browser says about its adapter (`GPUAdapter.info`; wgpu's own
//!   name is empty there);
//! - `GUST_FAIL <reason>` on any error or panic.
//!
//! The batch builder, [`build`], is compiled on the host as well, which is how `cargo gust build` finds the kernels;
//! everything that talks to the browser is `cfg(target_arch = "wasm32")` (the `page` module).

use std::sync::Arc;

use gust::batch::{Batch, D1, Error, Param, Stage};
use gust::{gpu_batch, gpu_task};
use gust_wgpu::HeadlessDevice;

/// The elements of the input and of each frame's result: more than one workgroup and not a multiple of it.
pub const LENGTH: u32 = 1000;

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
}

/// The batch: `y = x * 3 + gain`, then `y = y * 2 + 1` in place, with `in_flight(2)`. `u32` arithmetic on small values,
/// so the host computation ([`reference`]) is exact.
///
/// # Errors
///
/// The batch's build errors (an unusable device, a kernel without its artifact on the `web` profile).
pub fn build(
    device: &Arc<HeadlessDevice>,
    n: u32,
) -> Result<(Batch<HeadlessDevice>, Handles), Error> {
    gpu_batch!("frames", on(device); in_flight(2); |b| {
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
        Handles { gain, x, y, count, z, rows, mixed, rotated }
    })
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

/// The input of frame `seed`.
#[must_use]
pub fn input(seed: u32) -> Vec<u32> {
    (0..LENGTH).map(|i| i * 7 + seed).collect()
}

#[cfg(target_arch = "wasm32")]
mod page;
