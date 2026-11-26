//! Counted stages (plan m14.s2, slice m14.s2i, D278): a stage that runs over the rows `0 .. min(n, capacity)`, where `n`
//! is a one-element `u32` buffer of the same batch that a stage on the GPU writes.
//!
//! The stage keeps its ordinary extent as its capacity (the size of its buffers, and what a resize changes), and the
//! macro emits two kernels for it: the counted kernel, which runs over a two-dimensional grid of workgroups (the
//! per-dimension limit of 65535 groups is no ceiling) and computes a linear row index from it, and a one-invocation
//! args kernel that turns the count into the grid. A device launches the pair in one of two ways ([`CountedLaunch`]):
//! indirectly (the args kernel writes the grid into a buffer the next pass reads) or at the worst-case grid of the
//! capacity (the same counted kernel, no args kernel). Both mean one thing: the first `min(n, capacity)` rows are
//! written and the others are not. The CPU twin reads `n` from its own buffer when the stage runs and loops over the
//! same rows.
//!
//! Everything here is macro support, outside the pre-1.0 promise (D111), except what the batch module re-exports.

use core::cell::Cell;
use std::sync::Arc;

use super::{BatchDevice, Error, Layout, Remake, Stage, StageDraft, StageRec, Values};
use crate::cpu::CpuDevice;
use crate::{Call, Extent, Kernel, Run};

/// The most workgroups a dispatch has along one dimension on every device GUST targets (the WebGPU default and the
/// Vulkan one). A counted stage's grid is two-dimensional so that this is no ceiling on its rows.
pub const GROUP_LIMIT: u32 = 65_535;

/// How a device launches a counted stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CountedLaunch {
    /// The args kernel writes the grid for `min(n, capacity)` rows and the counted kernel is dispatched from it: no
    /// workgroup beyond the count is launched. The default on native targets.
    Indirect,
    /// The counted kernel runs over the grid of the capacity and the rows past the count do nothing. No args kernel
    /// runs. The only launch on `wasm32` in this release.
    Bounded,
}

std::thread_local! {
    static LAUNCH: Cell<Option<CountedLaunch>> = const { Cell::new(None) };
}

/// The launch the batches sealed on this thread use for their counted stages: [`CountedLaunch::Indirect`] on native
/// targets and [`CountedLaunch::Bounded`] on `wasm32`, unless [`set_counted_launch`] chose one.
#[must_use]
pub fn counted_launch() -> CountedLaunch {
    LAUNCH
        .with(Cell::get)
        .unwrap_or(if cfg!(target_arch = "wasm32") {
            CountedLaunch::Bounded
        } else {
            CountedLaunch::Indirect
        })
}

/// Choose the launch of the counted stages of the batches this thread seals next (`None` restores the default). A
/// batch keeps the launch it was sealed with, through resizes. Test and measurement support: the launches mean the
/// same, and on `wasm32` only [`CountedLaunch::Bounded`] runs.
#[doc(hidden)]
pub fn set_counted_launch(launch: Option<CountedLaunch>) {
    LAUNCH.with(|cell| cell.set(launch));
}

/// The grid of a counted launch over `rows` rows of a kernel whose workgroup has `workgroup` invocations along x: the
/// workgroups along x and y, as the args kernel writes them. Zero rows give no group.
#[doc(hidden)]
#[must_use]
pub fn grid_of(rows: u32, workgroup: u32) -> [u32; 2] {
    let groups = rows / workgroup + u32::from(!rows.is_multiple_of(workgroup));
    [
        groups.min(GROUP_LIMIT),
        groups / GROUP_LIMIT + u32::from(!groups.is_multiple_of(GROUP_LIMIT)),
    ]
}

/// A device that can run a counted stage.
#[doc(hidden)]
pub trait CountedDevice: BatchDevice {
    /// Seal a counted stage. `counted` makes the call of the counted kernel over the extent the device chooses (the
    /// invocations of the grid, or of the rows); `args` makes the call of the args kernel over the buffer the device
    /// gives it; `count` is the buffer `n` and `capacity` the rows of the stage when it is not counted.
    ///
    /// # Errors
    ///
    /// The device's refusal of the calls.
    #[allow(clippy::too_many_arguments)]
    fn prepare_counted<K, KA, R, const N: usize, const NA: usize>(
        &self,
        keep: R,
        count: Arc<Self::Buffer<u32>>,
        capacity: u32,
        workgroup: u32,
        launch: CountedLaunch,
        counted: impl for<'o> Fn(&'o R, Extent) -> Call<'o, Self, K, N> + 'static,
        args: impl for<'o> Fn(&'o R, &'o Self::Buffer<u32>) -> Call<'o, Self, KA, NA> + 'static,
    ) -> Result<Self::Prepared, Self::Error>
    where
        Self: Run<K, N> + Run<KA, NA>,
        K: Kernel,
        KA: Kernel,
        R: 'static;
}

impl CountedDevice for CpuDevice {
    /// The twin reads `n` from its own buffer each time the stage runs (the stages before it have run), and runs the
    /// counted kernel over `min(n, capacity)` rows. It never runs the args kernel.
    fn prepare_counted<K, KA, R, const N: usize, const NA: usize>(
        &self,
        keep: R,
        count: Arc<Self::Buffer<u32>>,
        capacity: u32,
        _workgroup: u32,
        _launch: CountedLaunch,
        counted: impl for<'o> Fn(&'o R, Extent) -> Call<'o, Self, K, N> + 'static,
        _args: impl for<'o> Fn(&'o R, &'o Self::Buffer<u32>) -> Call<'o, Self, KA, NA> + 'static,
    ) -> Result<Self::Prepared, Self::Error>
    where
        Self: Run<K, N> + Run<KA, NA>,
        K: Kernel,
        KA: Kernel,
        R: 'static,
    {
        Ok(self.prepare_fn(move |device| {
            let rows = count.read().first().copied().unwrap_or(0).min(capacity);
            let call = counted(&keep, Extent::new(rows, 1, 1));
            <CpuDevice as Run<K, N>>::dispatch_call(device, &call)
        }))
    }
}

impl<'b, D: BatchDevice> StageDraft<'b, D> {
    /// Check that the buffer numbered `id` is a count: one element. The stage is dropped (an error is in the builder)
    /// when it is not. Macro support.
    pub fn require_count(&mut self, id: usize) {
        let found = {
            let state = self.builder.state.borrow();
            state
                .buffers
                .get(id)
                .map(|entry| (entry.name.clone(), entry.len))
        };
        match found {
            Some((_, 1)) | None => {}
            Some((buffer, len)) => {
                let mut state = self.builder.state.borrow_mut();
                let (batch, stage) = (state.name.clone(), self.name.clone());
                state.fail(Error::NotCount {
                    batch,
                    stage,
                    buffer,
                    len,
                });
                self.ok = false;
            }
        }
    }

    /// Seal a counted stage: [`StageDraft::finish`] over the pair of kernels. `keep_with` makes the stage's `keep`, its
    /// count buffer and its capacity again over the buffers of a resized frame. Macro support.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_counted<T, Rk, K, KA, R, const N: usize, const NA: usize>(
        self,
        keep: R,
        count: Arc<D::Buffer<u32>>,
        capacity: u32,
        workgroup: u32,
        counted: impl for<'o> Fn(&'o R, Extent) -> Call<'o, D, K, N> + Copy + 'static,
        args: impl for<'o> Fn(&'o R, &'o D::Buffer<u32>) -> Call<'o, D, KA, NA> + Copy + 'static,
        update: impl Fn(&D, &mut D::Frame, &Values) -> Result<(), D::Error> + 'static,
        keep_with: impl Fn(&Layout<'_>) -> Result<(R, Arc<D::Buffer<u32>>, u32), Error> + 'static,
    ) -> Stage<T, Rk>
    where
        D: CountedDevice + Run<K, N> + Run<KA, NA>,
        K: Kernel,
        KA: Kernel,
        R: 'static,
        D::Buffer<u32>: 'static,
    {
        let Self {
            builder,
            name,
            dims,
            reads,
            params,
            write,
            ok,
            ..
        } = self;
        let mut state = builder.state.borrow_mut();
        let dud = Stage::new(state.id, usize::MAX, dims);
        let (true, Some(write)) = (ok, write) else {
            return dud;
        };
        let launch = counted_launch();
        match state
            .device
            .prepare_counted(keep, count, capacity, workgroup, launch, counted, args)
        {
            Ok(prepared) => {
                let remake: Remake<D> = Box::new(move |device, layout| {
                    let (keep, count, capacity) = keep_with(layout)?;
                    device
                        .prepare_counted(keep, count, capacity, workgroup, launch, counted, args)
                        .map_err(|error| Error::Device(D::map_error(error)))
                });
                let rec = StageRec::kernel(name, prepared, Box::new(update), params, reads, write)
                    .with_remake(remake);
                state.stages.push(rec);
                Stage::new(state.id, write, dims)
            }
            Err(error) => {
                state.fail(Error::Device(D::map_error(error)));
                dud
            }
        }
    }
}
