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
//! The generated code builds the kernel's [`Call`](crate::Call) (`run::call_extent`) and nothing else about the
//! submission: [`submit_and_read`] asks the device's `MapHost::start_read` to make the output buffer, run that call
//! and start the read-back. The default of that method is two device calls (the dispatch, then the read-back copy when
//! the returned future is first polled); the GPU device overrides it to record the dispatch, the clear of a recycled
//! output and the read-back copy in one command buffer. Decision D251 (one command encoder, one `queue.submit` per
//! submit) is met on the GPU device without a change to generated code.
//!
//! Measured with the device's `gust_submissions()` counter (Radeon 890M only; the RTX 5090 run is pending): a `submit`
//! makes exactly 1 submission, for a 1-D task with a fresh or a recycled output, a 2-D task and an in-place task alike,
//! and the read-back adds none. The uploads of the captures and the extent record are `write_buffer` calls and buffer
//! creations, not submissions, with one exception that generated code makes outside `submit_and_read`: an empty
//! captured slice on a recycled pool buffer is cleared in a submission of its own. The launch refuses such a task
//! (`IndependentLengthEmpty`, an empty capture with a non-empty extent), so that submission is made by a task that
//! fails: it makes 1 and no dispatch. A device that keeps the default makes two submissions (the dispatch and the
//! copy), three when its pool clears a recycled output; the CPU device has no queue.
//!
//! The ignored tests `gpu_*` of `examples/task-basics` count these on a GPU; the test
//! `a_task_starts_one_dispatch_and_one_read_back` of this file counts the default path on a device that records its
//! calls, through [`submit_and_read`] itself.

use core::future::Future;
use core::ops::Deref;
use core::pin::Pin;

use crate::map::{Error, MapHost};
use crate::{Call, GpuPod, Kernel, Run};

/// The asynchronous read-back of a task: a future of the task's result, a `Vec<T>` for a task that has a result type and
/// `()` for an in-place task (its target slice then holds the result). It is the future that `map_async` returns today,
/// boxed so that it owns the device handle and the output buffer.
///
/// Where the read-back copy is made depends on the device. On the GPU device it is already in the one submission that
/// [`Task::submit`] made, and the mapping of the result is requested at once. A device that keeps the default of
/// `MapHost::start_read` makes the copy when the future is first polled. On a native GPU device the future completes
/// only while something polls it (each poll polls the device once), so a caller without a runtime polls it in a loop;
/// [`Task::dispatch`] does that.
///
/// # What it is and what it holds
///
/// - It is not `Send`: it can be polled on the thread that made it only.
/// - It keeps everything the task's list names (the host slices and the in-place target) borrowed until the read-back
///   ends, for one lifetime `'t`, although the slices were uploaded when the task was submitted. A read-only slice
///   can be used again only after the future is ready or dropped.
/// - Dropping it before the first poll leaves nothing running on the host and discards the values. The dispatch was
///   submitted by [`Task::submit`] and is not undone: on a GPU device it still runs and the copy was already made, so
///   the staging buffer goes back to the pool when its mapping completes. A device that keeps the default starts no
///   read-back at all. An in-place target is written only when the future resolves, so it is left exactly as it was
///   before the task. The output buffer is released with the future.
///
/// A later slice (the ticket of the batch) may replace this type; it is not a stable surface.
pub type Pending<'t, O> = Pin<Box<dyn Future<Output = Result<O, Error>> + 't>>;

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
    /// The work is submitted before this returns. On the GPU device the read-back copy is part of that submission and
    /// the mapping is requested at once; a device that keeps the default of `MapHost::start_read` makes the copy when
    /// the returned future is first polled. Either way a native caller drives the future (see [`Pending`]); in the
    /// browser a plain `.await` does. Available on every target.
    ///
    /// On the GPU device this is one queue submission, the dispatch and the read-back copy together (decision D251;
    /// measured on the Radeon 890M, see the module's section on the queue). The one exception is a task with an empty
    /// captured slice on a recycled pool buffer, which clears it in a submission of its own and is then refused. A
    /// device that does not override `MapHost::start_read` makes two, the dispatch and the copy.
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

/// What a device must offer to run the generated kernel of a task: [`MapHost`] (upload, a zeroed output, read-back and
/// [`MapHost::start_read`], where a backend makes the dispatch and the read-back one submission) and the typed `Run`
/// of the kernel. The generated code names this trait and nothing finer.
#[doc(hidden)]
pub trait TaskDevice<K: Kernel, const N: usize>: MapHost + Run<K, N> {}

impl<D: MapHost + Run<K, N>, K: Kernel, const N: usize> TaskDevice<K, N> for D {}

/// The place where a task's dispatch and read-back reach the queue (see the module's section on it). The one other
/// submission is an empty captured slice's clear on a recycled pool buffer, which generated code makes in its own
/// submission.
///
/// It hands the sink's initial values (or its length), `build` and `context` to [`MapHost::start_read`], which makes
/// the output buffer, asks `build` for the kernel's [`Call`] over the context and that buffer (the generated closure
/// binds the kernel's resources and does nothing else), submits the call and starts the read-back. The result is a
/// future that owns the device handle and the buffer. On the GPU device that is one queue submission, as measured by
/// its `gust_submissions()` counter; the default of `start_read` makes two. `context` is whatever the call borrows
/// besides the output (the extent record and the uploaded captures): it is moved in so that the call can borrow it.
/// Macro support.
///
/// # Errors
///
/// A length above `u32::MAX` is [`Error::TooLong`]; the device's failures are [`Error::Device`] (or the device's
/// own variant, see [`MapHost::map_error`]).
#[doc(hidden)]
pub fn submit_and_read<'t, H, D, T, S, C, K, const N: usize>(
    device: H,
    sink: S,
    context: C,
    build: impl for<'o> FnOnce(&'o C, &'o D::Buffer<T>) -> Call<'o, D, K, N>,
) -> Result<Pending<'t, S::Done>, Error>
where
    H: Deref<Target = D> + 't,
    D: TaskDevice<K, N> + 't,
    D::Buffer<T>: 't,
    T: GpuPod + 't,
    S: Sink<'t, T>,
    K: Kernel,
{
    let len = u32::try_from(sink.count()).map_err(|_| Error::TooLong { len: sink.count() })?;
    let read = D::start_read(device, sink.initial(), len, context, build).map_err(D::map_error)?;
    Ok(Box::pin(async move {
        let values = read.await.map_err(D::map_error)?;
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
    use std::rc::Rc;

    use crate::cpu::{CpuBinding, CpuBuffer, CpuDevice, CpuJob};
    use crate::{
        Access, Arg, Device, KernelDescriptor, LengthRule, ParameterDescriptor, ParameterKind,
        Resource, TypedDevice, kind,
    };

    use super::*;

    /// A kernel with one storage parameter: all the spy needs of one.
    struct Probe;

    impl Kernel for Probe {
        const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new(
            "m",
            "probe",
            "m_probe",
            [1, 1, 1],
            &[ParameterDescriptor {
                name: "out",
                rust_type: "f32",
                kind: ParameterKind::Storage,
                access: Access::ReadWrite,
                resource_layout: None,
                binding: None,
            }],
            "",
        );
    }

    /// A CPU buffer that belongs to the spy, shared so that the spy's dispatch can write it.
    struct SpyBuffer<T: GpuPod>(Rc<CpuBuffer<T>>);

    impl<T: GpuPod> Resource for SpyBuffer<T> {
        type Element = T;
        type Device = Spy;
        type Kind = kind::Storage;

        fn bind<'r>(&'r self, access: Access, length: LengthRule) -> CpuBinding<'r> {
            self.0.bind(access, length)
        }
    }

    /// A CPU device that writes down the calls a task makes on it, in order, and runs a script when it is asked to
    /// dispatch. A call that submits to a queue is `dispatch` (`Run::submit_call`) and `copy` (`readback_async`).
    struct Spy {
        inner: CpuDevice,
        log: RefCell<Vec<&'static str>>,
        script: RefCell<Option<Box<dyn FnOnce()>>>,
        fail: bool,
    }

    impl Spy {
        fn new() -> Self {
            Self {
                inner: CpuDevice::new(),
                log: RefCell::new(Vec::new()),
                script: RefCell::new(None),
                fail: false,
            }
        }

        fn note(&self, what: &'static str) {
            self.log.borrow_mut().push(what);
        }

        /// What the next dispatch does to the device's memory.
        fn on_dispatch(&self, script: impl FnOnce() + 'static) {
            *self.script.borrow_mut() = Some(Box::new(script));
        }

        /// The calls made so far that submit to a queue on a GPU device.
        fn submitting_calls(&self) -> Vec<&'static str> {
            self.log
                .borrow()
                .iter()
                .copied()
                .filter(|call| matches!(*call, "dispatch" | "copy"))
                .collect()
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

    impl Run<Probe, 1> for Spy {
        fn submit_call(&self, call: &Call<'_, Self, Probe, 1>) -> Result<CpuJob, Self::Error> {
            self.note("dispatch");
            assert_eq!(call.kernel().name, "probe");
            if self.fail {
                return Err(crate::cpu::Error::Borrowed { parameter: "out" });
            }
            if let Some(script) = self.script.borrow_mut().take() {
                script();
            }
            Ok(CpuJob::complete())
        }
    }

    impl MapHost for Spy {
        type Buffer<T: GpuPod> = SpyBuffer<T>;

        fn upload<T: GpuPod>(&self, values: &[T]) -> Result<SpyBuffer<T>, Self::Error> {
            self.note("upload");
            self.inner
                .upload(values)
                .map(|buffer| SpyBuffer(Rc::new(buffer)))
        }

        fn zeroed<T: GpuPod>(&self, len: u32) -> Result<SpyBuffer<T>, Self::Error> {
            self.note("zeroed");
            self.inner
                .zeroed(len)
                .map(|buffer| SpyBuffer(Rc::new(buffer)))
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

    /// The call of the probe kernel over `output`, as the generated closure builds one.
    fn probe_call<'o, T: GpuPod>(len: u32, output: &'o SpyBuffer<T>) -> Call<'o, Spy, Probe, 1> {
        Call::new(len, [Arg::from(output).bind(Access::ReadWrite)])
    }

    /// What `submit_and_read` asks of a device for one task, counted on a device that records its calls: the output is
    /// created, and then exactly one call that submits is made by `submit` (the dispatch) and exactly one by the first
    /// poll (the read-back copy). On a GPU device `zeroed` adds the pool's clear when it recycles a buffer, which this
    /// device has no pool for. The count is of calls through `submit_and_read`, not of `queue.submit`.
    #[test]
    fn a_task_starts_one_dispatch_and_one_read_back() {
        let spy = Spy::new();
        let pending = submit_and_read(&spy, Fresh(3), (), |(): &(), output: &SpyBuffer<f32>| {
            let out = Rc::clone(&output.0);
            spy.on_dispatch(move || out.write(&[1.0, 2.0, 3.0]).unwrap());
            probe_call(3, output)
        })
        .unwrap();
        // The dispatch is made before `submit` returns; the copy waits for the first poll.
        assert_eq!(*spy.log.borrow(), ["zeroed", "dispatch"]);
        assert_eq!(spy.submitting_calls(), ["dispatch"]);
        assert_eq!(block_on(pending).unwrap(), [1.0, 2.0, 3.0]);
        assert_eq!(*spy.log.borrow(), ["zeroed", "dispatch", "copy"]);
        assert_eq!(spy.submitting_calls(), ["dispatch", "copy"]);
    }

    #[test]
    fn an_in_place_target_is_uploaded_and_overwritten() {
        let spy = Spy::new();
        let mut target = [1_u32, 2, 3];
        let pending = submit_and_read(
            &spy,
            InPlace(&mut target),
            (),
            |(): &(), output: &SpyBuffer<u32>| {
                let out = Rc::clone(&output.0);
                spy.on_dispatch(move || out.write(&[10, 20, 30]).unwrap());
                probe_call(3, output)
            },
        )
        .unwrap();
        assert_eq!(*spy.log.borrow(), ["upload", "dispatch"]);
        block_on(pending).unwrap();
        assert_eq!(target, [10, 20, 30]);
        assert_eq!(spy.submitting_calls(), ["dispatch", "copy"]);
    }

    /// Dropping a `Pending` before its first poll: the dispatch was made, no read-back starts, and the in-place target
    /// is as it was (it is written when the future resolves).
    #[test]
    fn a_pending_dropped_unpolled_leaves_an_in_place_target_unchanged() {
        let spy = Spy::new();
        let mut target = [1_u32, 2, 3];
        let pending = submit_and_read(
            &spy,
            InPlace(&mut target),
            (),
            |(): &(), output: &SpyBuffer<u32>| {
                let out = Rc::clone(&output.0);
                spy.on_dispatch(move || out.write(&[10, 20, 30]).unwrap());
                probe_call(3, output)
            },
        )
        .unwrap();
        drop(pending);
        assert_eq!(target, [1, 2, 3]);
        assert_eq!(*spy.log.borrow(), ["upload", "dispatch"]);
    }

    #[test]
    fn a_context_is_borrowed_by_the_call_and_given_back_by_reference() {
        let spy = Spy::new();
        let pending = submit_and_read(
            &spy,
            Fresh(2),
            7_u32,
            |seven: &u32, output: &SpyBuffer<u32>| {
                assert_eq!(*seven, 7);
                probe_call(2, output)
            },
        )
        .unwrap();
        assert_eq!(block_on(pending).unwrap(), [0, 0]);
    }

    #[test]
    fn a_dispatch_error_is_returned_and_nothing_is_read() {
        let mut spy = Spy::new();
        spy.fail = true;
        let result = submit_and_read(&spy, Fresh(2), (), |(): &(), output: &SpyBuffer<u32>| {
            probe_call(2, output)
        });
        assert!(matches!(result, Err(Error::Device(_))));
        assert_eq!(*spy.log.borrow(), ["zeroed", "dispatch"]);
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
