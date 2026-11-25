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

/// The handles of the batch: the gain, the input, and the stage whose values every frame's ticket reads.
pub struct Handles {
    /// Added to every element in the first stage.
    pub gain: Param<u32>,
    /// The input, written by a frame.
    pub x: Stage<u32, D1>,
    /// The result: `y = (x * 3 + gain) * 2 + 1`.
    pub y: Stage<u32, D1>,
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
        Handles { gain, x, y }
    })
}

/// The host computation of one frame's result.
#[must_use]
pub fn reference(x: &[u32], gain: u32) -> Vec<u32> {
    x.iter().map(|v| (v * 3 + gain) * 2 + 1).collect()
}

/// The input of frame `seed`.
#[must_use]
pub fn input(seed: u32) -> Vec<u32> {
    (0..LENGTH).map(|i| i * 7 + seed).collect()
}

#[cfg(target_arch = "wasm32")]
mod page;
