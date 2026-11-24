//! Opt-in GPU timestamps (D194): [`timed`] runs a closure of gust-algo calls and of [`dispatch_batch_in_scope`] calls
//! and reports the GPU time of the batches they dispatched.
//!
//! Every public GPU operation of this crate that dispatches itself submits its dispatches as one or more batches through
//! [`dispatch_batch_in_scope`] below, which a caller's own batches can use too. Outside a [`timed`] scope that is
//! exactly `HeadlessDevice::dispatch_batch`; inside one it is `HeadlessDevice::dispatch_batch_timed`, and the result
//! is added to the scope's accumulator.
//! The exception is [`Recording::with_dispatches`](crate::record::Recording::with_dispatches), which hands its
//! dispatches to the caller instead.
//!
//! Native only: the batches it counts go through the blocking `dispatch_batch`, which `wasm32` does not have.

use std::cell::RefCell;
use std::time::Duration;

use gust_wgpu::{BufferDispatch, HeadlessDevice};

use crate::Error;

/// The GPU time of the batches a [`timed`] scope dispatched.
///
/// See [`timed`] for what `gpu_elapsed` brackets and excludes.
///
/// ```
/// use gust_algo::{GpuTime, reduce, timed};
/// use gust_wgpu::{GpuVec, default_device};
///
/// let device = default_device()?;
/// let input = GpuVec::from_slice_on(&device, &[1u32, 2, 3, 4])?;
/// let result = timed(|| reduce::sum(&input))?;
/// let time: GpuTime = result.time;
/// assert_eq!(time.batches, 1);
/// assert_eq!(time.gpu_elapsed.is_some(), device.supports_gpu_timestamps());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct GpuTime {
    /// The sum of the GPU time of every batch of the scope, or `None` when the scope dispatched no batch or a batch ran
    /// on an adapter without timestamp queries (`HeadlessDevice::supports_gpu_timestamps`).
    pub gpu_elapsed: Option<Duration>,
    /// How many batches the scope dispatched (an empty input dispatches none).
    pub batches: usize,
}

/// The value of a [`timed`] scope and its GPU time.
///
/// ```
/// use gust_algo::{Timed, reduce, timed};
/// use gust_wgpu::{GpuVec, default_device};
///
/// let device = default_device()?;
/// let input = GpuVec::from_slice_on(&device, &[5u32, 6])?;
/// let result: Timed<u32> = timed(|| reduce::sum(&input))?;
/// assert_eq!(result.value, 11);
/// assert!(result.time.batches >= 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Timed<R> {
    /// What the closure returned.
    pub value: R,
    /// The GPU time of the batches it dispatched.
    pub time: GpuTime,
}

/// The running total of one scope.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Accum {
    gpu: Duration,
    unsupported: bool,
    batches: usize,
}

impl Accum {
    /// One finished batch: its `gpu_elapsed` and whether the device can report one. A batch with no time on a device
    /// that can report one did no GPU work (every dispatch was empty) and adds zero.
    pub(crate) fn record(&mut self, gpu_elapsed: Option<Duration>, supports_timestamps: bool) {
        self.batches = self.batches.saturating_add(1);
        match gpu_elapsed {
            Some(elapsed) => self.gpu = self.gpu.saturating_add(elapsed),
            None if supports_timestamps => {}
            None => self.unsupported = true,
        }
    }

    fn merge(&mut self, inner: &Self) {
        self.gpu = self.gpu.saturating_add(inner.gpu);
        self.unsupported |= inner.unsupported;
        self.batches = self.batches.saturating_add(inner.batches);
    }

    pub(crate) fn time(&self) -> GpuTime {
        GpuTime {
            gpu_elapsed: (self.batches > 0 && !self.unsupported).then_some(self.gpu),
            batches: self.batches,
        }
    }
}

thread_local! {
    /// The accumulator of the innermost [`timed`] scope of this thread, `None` outside one.
    static SCOPE: RefCell<Option<Accum>> = const { RefCell::new(None) };
}

/// Whether a [`timed`] scope is open on this thread (tests: a leaked scope shows as `true` after the scope ended).
#[cfg(test)]
pub(crate) fn scope_is_open() -> bool {
    SCOPE.with(|scope| scope.borrow().is_some())
}

/// Count one finished batch in the scope of this thread, if there is one.
pub(crate) fn note_batch(gpu_elapsed: Option<Duration>, supports_timestamps: bool) {
    SCOPE.with(|scope| {
        if let Some(accum) = scope.borrow_mut().as_mut() {
            accum.record(gpu_elapsed, supports_timestamps);
        }
    });
}

/// Restores the enclosing scope (and folds this scope's totals into it) when a scope ends, however it ends.
struct Scope {
    outer: Option<Option<Accum>>,
}

impl Scope {
    fn enter() -> Self {
        let outer = SCOPE.with(|scope| scope.replace(Some(Accum::default())));
        Self { outer: Some(outer) }
    }

    fn finish(&mut self) -> Accum {
        let Some(outer) = self.outer.take() else {
            return Accum::default();
        };
        SCOPE.with(|scope| {
            let mine = scope.replace(outer).unwrap_or_default();
            if let Some(enclosing) = scope.borrow_mut().as_mut() {
                enclosing.merge(&mine);
            }
            mine
        })
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Run `run` and report the GPU time of the batches it dispatched (gust-algo's own and those sent through
/// [`dispatch_batch_in_scope`]): opt-in GPU timestamps (D194).
///
/// Wrap any calls of this crate: a single operation (`timed(|| reduce::sum(&input))`), or several in one closure. The
/// operations behave exactly as outside the scope and return the same values; the scope only measures.
///
/// # What is counted
///
/// Only gust-algo operations and [`dispatch_batch_in_scope`] calls on the calling thread inside the closure are
/// counted: the scope is per thread, so work the closure moves to another thread is not, and a batch sent straight to
/// `HeadlessDevice::dispatch_batch` is not. Batches on more than one device are summed.
///
/// An operation dispatches one or more batches (`reduce::sum` and the other reductions, `scan::inclusive`, the
/// gathers: one; `compact::count`: two; `compact::indices`: four; an empty input: none). `gpu_elapsed` is the sum, over
/// the batches, of the time from the beginning of the first dispatch's compute pass of the batch to the end of the
/// last dispatch's (each dispatch is a pass of its own; the transitions between them are included).
///
/// It does not include the gaps between batches, the clearing of pooled buffers (each is a submission of its own), the
/// readback of a result, or host time for encoding, submission and waiting, so for a multi-batch operation it is a
/// lower bound of the wall time of the call. Work that other threads submit to the same queue can overlap the bracket.
///
/// A scope with zero batches gives `gpu_elapsed == None` and `batches == 0` (an empty input, or a gather with empty
/// values, dispatches none). When `batches > 0`, `gpu_elapsed` is `Some` exactly when every batch ran on a device that
/// supports timestamp queries (`HeadlessDevice::supports_gpu_timestamps`). A batch of only empty dispatches on such a
/// device adds zero.
///
/// If the closure returns an error, so does `timed`, and no time is reported. Scopes nest: an inner scope reports its
/// own time and the outer one includes it too, also when the inner scope returned an error or panicked and the
/// outer closure went on (the batches ran, so the outer scope counts them). A batch whose own dispatch returned an
/// error is not counted, even if its submission ran.
///
/// # Cost when not used
///
/// A call outside any scope pays one thread-local read per batch and then makes the same `dispatch_batch` call as
/// before: no timestamp query, no extra submission, no allocation. Inside a scope the batches use
/// `dispatch_batch_timed`: the same passes in an encoder of its own, plus two timestamp writes, a query resolve and a
/// 16-byte readback per batch. Host time inside a scope is therefore larger than outside it, and timed batches on one
/// device take turns across threads (the device has one timestamp profiler behind a lock).
///
/// # Errors
///
/// Whatever the closure returns.
///
/// ```
/// use gust_algo::{compact, reduce, timed};
/// use gust_wgpu::{GpuVec, default_device};
///
/// let device = default_device()?;
/// let input = GpuVec::from_slice_on(&device, &[1.0f32, 2.0, 3.0])?;
/// let sum = timed(|| reduce::sum(&input))?;
/// assert_eq!(sum.value, 6.0);
/// assert_eq!(sum.time.gpu_elapsed.is_some(), device.supports_gpu_timestamps());
///
/// // Several calls in one scope add up: count is two batches.
/// let flags = GpuVec::from_slice_on(&device, &[1u32, 0, 7])?;
/// let counted = timed(|| compact::count(&flags))?;
/// assert_eq!(counted.time.batches, 2);
/// assert_eq!(counted.value.to_vec()?, [2]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn timed<R>(run: impl FnOnce() -> Result<R, Error>) -> Result<Timed<R>, Error> {
    let mut scope = Scope::enter();
    let value = run()?;
    let time = scope.finish().time();
    Ok(Timed { value, time })
}

/// Dispatch one batch of your own kernels, counted in the enclosing [`timed`] scope: a caller's own kernels can be
/// timed in the same scope as gust-algo calls.
///
/// This is `device.dispatch_batch(batch)` with one difference. Outside a [`timed`] scope of this thread it is exactly
/// that call. Inside one it runs the batch through `HeadlessDevice::dispatch_batch_timed` and adds its GPU time to the
/// scope and one to `batches`, as it does for the batches of this crate's own operations; the rules of [`timed`] for
/// timestamp support, a batch of only empty dispatches and errors apply unchanged. An empty slice (no dispatch at all)
/// is not a batch: it counts nothing, in a scope or outside.
///
/// # Counted once
///
/// The function dispatches the batch itself and counts the time it measured in that same call, so a batch is counted
/// exactly once when it goes through here instead of `HeadlessDevice::dispatch_batch`. There is no way to report a time
/// without dispatching, so a batch cannot be counted twice by this function. Two things stay with the caller: a batch
/// sent straight to `HeadlessDevice::dispatch_batch` or `dispatch_batch_timed` is not counted at all, and the same
/// batch passed here twice is dispatched twice and counted twice, as it ran twice. Do not send the batches of
/// gust-algo calls through here: those operations already count their own.
///
/// # Errors
///
/// The error of the dispatch, unchanged. A batch whose dispatch returned an error is not counted.
///
/// ```
/// use gust::Independent;
/// use gust_algo::{dispatch_batch_in_scope, reduce, timed};
/// use gust_wgpu::{BufferDispatch, GpuVec, default_device};
///
/// #[gust::gpu_module]
/// mod double {
///     #[kernel(workgroup_size(64, 1, 1))]
///     pub fn run(
///         id: SV_DispatchThreadID,
///         values: StructuredBuffer<uint>,
///         mut out: RWStructuredBuffer<uint>,
///     ) {
///         let i = id.x;
///         if i < out.len() {
///             out[i] = values[i] * 2u32;
///         }
///     }
/// }
///
/// let device = default_device()?;
/// let values = GpuVec::from_slice_on(&device, &[1u32, 2, 3, 4])?;
/// let doubled = GpuVec::<u32>::zeroed_on(&device, 4)?;
///
/// // One scope: an own batch and a gust-algo call, two batches.
/// let result = timed(|| {
///     let call = double::run::call(4, Independent(&values), Independent(&doubled));
///     dispatch_batch_in_scope(&device, &[BufferDispatch::from(&call)])?;
///     reduce::sum(&doubled)
/// })?;
/// assert_eq!(result.value, 20);
/// assert_eq!(result.time.batches, 2);
/// assert_eq!(result.time.gpu_elapsed.is_some(), device.supports_gpu_timestamps());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn dispatch_batch_in_scope(
    device: &HeadlessDevice,
    batch: &[BufferDispatch<'_>],
) -> Result<(), gust_wgpu::Error> {
    if batch.is_empty() || !SCOPE.with(|scope| scope.borrow().is_some()) {
        return device.dispatch_batch(batch);
    }
    let timing = device.dispatch_batch_timed(batch)?;
    note_batch(timing.gpu_elapsed, device.supports_gpu_timestamps());
    Ok(())
}
