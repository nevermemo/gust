//! The host side of `gpu_task!`: a stage written at the call site, run as a free task (plan m14.s2, slice s2a1).
//!
//! `gust::gpu_task!` builds a kernel around its closure (the `par_for!` generator, so the artifacts, the records, the
//! routing and the CPU twin are those of any kernel) and evaluates to a [`Task`]. The task is run once, in one of two
//! ways: [`Task::submit`] starts the work at once and returns the asynchronous read-back, and [`Task::dispatch`] (native
//! targets only) waits for it.
//!
//! ```text
//! let doubled = gust::gpu_task!(xs.len(), [xs: &[f32]], |i| -> f32 { xs[i] * 2.0 }).dispatch()?;
//! ```
//!
//! (Not a doctest: the kernel needs its CPU twin to run on a `CpuDevice`, and a doctest is no test build of its
//! crate. `examples/task-basics` runs every form.)
//!
//! # What `submit` costs on the queue
//!
//! Every runtime call that [`Task::submit`] makes goes through [`submit_and_read`] and nowhere else, so the number of
//! queue submissions is that one function's business and the generated code does not depend on it. Today it is a
//! known gap against decision D251 (one command encoder, one `queue.submit` per submit): the dispatch is one
//! submission, and the read-back copy is a second one that is made when the returned future is first polled. A zeroed
//! output that the device's buffer pool recycles adds a third, the pool's own clear. So a task makes 2 submissions
//! with a fresh output and 3 with a recycled one ([`SUBMISSIONS_FRESH_OUTPUT`], [`SUBMISSIONS_RECYCLED_OUTPUT`]); an
//! in-place task uploads its target (no clear) and makes 2. Slice m14.encb adds an encode-plus-read-back entry point
//! to the GPU backend, and the swap is a change to [`submit_and_read`] alone.

use core::future::Future;
use core::ops::Deref;
use core::pin::Pin;

use crate::map::{Error, MapHost};
use crate::{GpuPod, Kernel, Run};

/// The asynchronous read-back of a task: a future of the task's result, a `Vec<T>` for a task that has a result type and
/// `()` for an in-place task (its target slice then holds the result). It is the future that `map_async` returns today,
/// boxed so that it owns the device handle and the output buffer; `Ticket<T>` (slice s2a2) wraps it.
///
/// The read-back copy is submitted when the future is first polled. On a native GPU device the future completes only
/// while something polls it (each poll polls the device once), so a caller without a runtime polls it in a loop;
/// [`Task::dispatch`] does that.
pub type Pending<'t, O> = Pin<Box<dyn Future<Output = Result<O, Error>> + 't>>;

/// The queue submissions of one [`Task::submit`] whose output is a fresh buffer of the device: the dispatch and the
/// read-back copy.
pub const SUBMISSIONS_FRESH_OUTPUT: u32 = 2;

/// The queue submissions of one [`Task::submit`] whose output buffer the device's pool recycled: the pool's clear, the
/// dispatch and the read-back copy.
pub const SUBMISSIONS_RECYCLED_OUTPUT: u32 = 3;

/// A stage written at the call site, ready to run once.
///
/// It borrows what the call's list names for `'t`. [`submit`](Self::submit) and [`dispatch`](Self::dispatch) take the
/// task by value.
#[must_use = "a task does nothing until it is submitted or dispatched"]
pub struct Task<'t, O> {
    start: Box<dyn FnOnce() -> Result<Pending<'t, O>, Error> + 't>,
}

impl<'t, O> Task<'t, O> {
    /// A task that starts by calling `start`. Macro support.
    #[doc(hidden)]
    pub fn new(start: impl FnOnce() -> Result<Pending<'t, O>, Error> + 't) -> Self {
        Self {
            start: Box::new(start),
        }
    }

    /// Upload the captures, start the kernel and return at once with the read-back of its result.
    ///
    /// The work is submitted before this returns. The read-back copy is submitted when the returned future is first
    /// polled, so a native caller drives it (see [`Pending`]); in the browser a plain `.await` does. Available on every
    /// target.
    ///
    /// **Known gap (decision D251).** This is two queue submissions, the dispatch and the copy, three when the
    /// device recycles the output buffer ([`SUBMISSIONS_FRESH_OUTPUT`], [`SUBMISSIONS_RECYCLED_OUTPUT`]); the design
    /// asks for one. All of it goes through [`submit_and_read`], which slice m14.encb changes.
    ///
    /// An empty shape runs nothing and returns an empty result.
    ///
    /// # Errors
    ///
    /// The device's own failures, as [`Error::Device`]; an extent that does not fit `u32`; a shape whose element count
    /// does not fit `u32` ([`Error::TooLong`]). Nothing ran in that case.
    pub fn submit(self) -> Result<Pending<'t, O>, Error> {
        (self.start)()
    }

    /// [`submit`](Self::submit), then wait for the result. Not available on `wasm32`, where nothing may block (D69).
    ///
    /// # Errors
    ///
    /// Those of [`submit`](Self::submit), and the device's failure to read the result back.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dispatch(self) -> Result<O, Error> {
        block_on(self.submit()?)
    }
}

/// Drive `future` on this thread until it completes.
#[cfg(not(target_arch = "wasm32"))]
fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = core::pin::pin!(future);
    let mut context = core::task::Context::from_waker(core::task::Waker::noop());
    loop {
        match future.as_mut().poll(&mut context) {
            core::task::Poll::Ready(value) => return value,
            core::task::Poll::Pending => std::thread::yield_now(),
        }
    }
}

/// Where a task's result goes: a fresh buffer of the device, or the target slice of an in-place task. Macro support;
/// sealed by being hidden, and free to change.
#[doc(hidden)]
pub trait Sink<'t, T: GpuPod>: 't {
    /// What the task's future resolves to.
    type Done: 't;

    /// The element count of the output buffer.
    fn count(&self) -> usize;

    /// The values the output buffer starts with: the target of an in-place task, `None` for a zeroed fresh buffer.
    fn initial(&self) -> Option<&[T]>;

    /// Take the values read back.
    fn finish(self, values: Vec<T>) -> Self::Done;
}

/// A fresh, zeroed output of the given length; the task's result is the `Vec` read back. Macro support.
#[doc(hidden)]
pub struct Fresh(pub usize);

impl<'t, T: GpuPod> Sink<'t, T> for Fresh {
    type Done = Vec<T>;

    fn count(&self) -> usize {
        self.0
    }

    fn initial(&self) -> Option<&[T]> {
        None
    }

    fn finish(self, values: Vec<T>) -> Vec<T> {
        values
    }
}

/// The target of an in-place task: uploaded as the output buffer, and overwritten with the read-back values. Macro
/// support.
#[doc(hidden)]
pub struct InPlace<'t, T>(pub &'t mut [T]);

impl<'t, T: GpuPod> Sink<'t, T> for InPlace<'t, T> {
    type Done = ();

    fn count(&self) -> usize {
        self.0.len()
    }

    fn initial(&self) -> Option<&[T]> {
        Some(&*self.0)
    }

    fn finish(self, values: Vec<T>) {
        self.0.copy_from_slice(&values);
    }
}

/// What a device must offer to run the generated kernel of a task: [`MapHost`] (upload, a zeroed output, read-back) and
/// the typed `Run` of the kernel. The generated code names this trait and nothing finer, so the bound that the
/// one-submission entry point needs later is added here.
#[doc(hidden)]
pub trait TaskDevice<K: Kernel, const N: usize>: MapHost + Run<K, N> {}

impl<D: MapHost + Run<K, N>, K: Kernel, const N: usize> TaskDevice<K, N> for D {}

/// The one place where a task reaches the runtime (see the module's section on the queue).
///
/// It makes the output buffer (`sink`), calls `launch` with the device and that buffer (the generated closure binds
/// the kernel's resources, submits the dispatch and drops its job), and returns the read-back as a future that owns the
/// device handle and the buffer. Macro support.
///
/// # Errors
///
/// A length above `u32::MAX` is [`Error::TooLong`]; the device's failures are [`Error::Device`] (or the device's
/// own variant, see [`MapHost::map_error`]); `launch`'s error is returned as it is.
#[doc(hidden)]
pub fn submit_and_read<'t, H, D, T, S>(
    device: H,
    sink: S,
    launch: impl FnOnce(&D, &D::Buffer<T>) -> Result<(), Error>,
) -> Result<Pending<'t, S::Done>, Error>
where
    H: Deref<Target = D> + 't,
    D: MapHost + 't,
    D::Buffer<T>: 't,
    T: GpuPod + 't,
    S: Sink<'t, T>,
{
    let len = u32::try_from(sink.count()).map_err(|_| Error::TooLong { len: sink.count() })?;
    let output = match sink.initial() {
        Some(values) => device.upload(values),
        None => device.zeroed::<T>(len),
    }
    .map_err(D::map_error)?;
    launch(&device, &output)?;
    Ok(Box::pin(async move {
        let values = device.readback_async(&output).await.map_err(D::map_error)?;
        Ok(sink.finish(values))
    }))
}

/// The read-back of a task that has nothing to run (an empty shape): a future that is ready with the empty result.
/// Macro support.
#[doc(hidden)]
#[must_use]
pub fn nothing_to_run<'t, T: GpuPod, S: Sink<'t, T>>(sink: S) -> Pending<'t, S::Done> {
    Box::pin(core::future::ready(Ok(sink.finish(Vec::new()))))
}

#[cfg(test)]
mod tests {
    use core::cell::RefCell;
    use core::convert::Infallible;

    use crate::cpu::{CpuBinding, CpuBuffer, CpuDevice, CpuJob};
    use crate::{Access, LengthRule, Resource, kind};
    use crate::{Device, GpuPod, TypedDevice};

    use super::*;

    /// A CPU buffer that belongs to the spy.
    struct SpyBuffer<T: GpuPod>(CpuBuffer<T>);

    impl<T: GpuPod> Resource for SpyBuffer<T> {
        type Element = T;
        type Device = Spy;
        type Kind = kind::Storage;

        fn bind<'r>(&'r self, access: Access, length: LengthRule) -> CpuBinding<'r> {
            self.0.bind(access, length)
        }
    }

    /// A CPU device that writes down the calls a task makes on it, in order.
    struct Spy {
        inner: CpuDevice,
        log: RefCell<Vec<&'static str>>,
    }

    impl Spy {
        fn new() -> Self {
            Self {
                inner: CpuDevice::new(),
                log: RefCell::new(Vec::new()),
            }
        }

        fn note(&self, what: &'static str) {
            self.log.borrow_mut().push(what);
        }
    }

    impl Device for Spy {
        type Error = crate::cpu::Error;
        type Dispatch<'r> = Infallible;
        type Job<'d> = CpuJob;

        fn submit(&self, _dispatches: &[Infallible]) -> Result<CpuJob, Self::Error> {
            Ok(CpuJob::complete())
        }
    }

    impl TypedDevice for Spy {
        type Binding<'r> = <CpuDevice as TypedDevice>::Binding<'r>;
    }

    impl MapHost for Spy {
        type Buffer<T: GpuPod> = SpyBuffer<T>;

        fn upload<T: GpuPod>(&self, values: &[T]) -> Result<SpyBuffer<T>, Self::Error> {
            self.note("upload");
            self.inner.upload(values).map(SpyBuffer)
        }

        fn zeroed<T: GpuPod>(&self, len: u32) -> Result<SpyBuffer<T>, Self::Error> {
            self.note("zeroed");
            self.inner.zeroed(len).map(SpyBuffer)
        }

        fn readback<T: GpuPod>(&self, buffer: &SpyBuffer<T>) -> Result<Vec<T>, Self::Error> {
            self.note("readback");
            self.inner.readback(&buffer.0)
        }

        async fn readback_async<T: GpuPod>(
            &self,
            buffer: &SpyBuffer<T>,
        ) -> Result<Vec<T>, Self::Error> {
            self.note("copy");
            self.inner.readback_async(&buffer.0).await
        }
    }

    /// Every runtime call of a task goes through `submit_and_read`: with a fresh output the device sees exactly the
    /// output's creation, the launch (one submission) and the read-back copy (one submission, made when the future is
    /// polled), which is the documented 2 (3 when the pool recycles the output and clears it itself).
    #[test]
    fn a_fresh_output_is_one_dispatch_and_one_copy() {
        let spy = Spy::new();
        let launched = RefCell::new(0_u32);
        let pending = submit_and_read(&spy, Fresh(3), |device: &Spy, output: &SpyBuffer<f32>| {
            device.note("dispatch");
            *launched.borrow_mut() += 1;
            output.0.write(&[1.0, 2.0, 3.0]).map_err(Error::device)
        })
        .unwrap();
        // The dispatch is made before `submit` returns; the copy waits for the first poll.
        assert_eq!(*spy.log.borrow(), ["zeroed", "dispatch"]);
        assert_eq!(block_on(pending).unwrap(), [1.0, 2.0, 3.0]);
        assert_eq!(*spy.log.borrow(), ["zeroed", "dispatch", "copy"]);
        assert_eq!(*launched.borrow(), 1);
        // The submissions the device sees: the dispatch and the copy; the pool's clear is the third on a GPU device
        // when it recycles the output (`zeroed` there), which this device has no pool for.
        let seen = spy
            .log
            .borrow()
            .iter()
            .filter(|call| matches!(**call, "dispatch" | "copy"))
            .count();
        assert_eq!(u32::try_from(seen).unwrap(), SUBMISSIONS_FRESH_OUTPUT);
        assert_eq!(SUBMISSIONS_RECYCLED_OUTPUT, SUBMISSIONS_FRESH_OUTPUT + 1);
    }

    #[test]
    fn an_in_place_target_is_uploaded_and_overwritten() {
        let spy = Spy::new();
        let mut target = [1_u32, 2, 3];
        let pending = submit_and_read(
            &spy,
            InPlace(&mut target),
            |_: &Spy, output: &SpyBuffer<u32>| output.0.write(&[10, 20, 30]).map_err(Error::device),
        )
        .unwrap();
        assert_eq!(*spy.log.borrow(), ["upload"]);
        block_on(pending).unwrap();
        assert_eq!(target, [10, 20, 30]);
    }

    #[test]
    fn a_launch_error_is_returned_and_nothing_is_read() {
        let spy = Spy::new();
        let result = submit_and_read(&spy, Fresh(2), |_: &Spy, _: &SpyBuffer<u32>| {
            Err(Error::Empty)
        });
        assert!(matches!(result, Err(Error::Empty)));
        assert_eq!(*spy.log.borrow(), ["zeroed"]);
    }

    #[test]
    fn nothing_to_run_is_ready_and_empty() {
        let pending = nothing_to_run::<f32, _>(Fresh(0));
        assert_eq!(block_on(pending).unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn a_task_starts_when_it_is_submitted_and_dispatch_waits() {
        let started = RefCell::new(false);
        let task = Task::new(|| {
            *started.borrow_mut() = true;
            Ok(nothing_to_run::<u32, _>(Fresh(0)))
        });
        assert!(!*started.borrow());
        assert!(task.dispatch().unwrap().is_empty());
        assert!(*started.borrow());
    }
}
