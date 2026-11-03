//! A one-shot completion future built on `std` alone (R4.9).
//!
//! wgpu reports `map_async` through a callback, and the browser drives that callback itself, so the part of a readback
//! that waits is a future that the callback completes. [`Completion`] is that future: a shared slot plus the waker of
//! whoever awaits it. It adds no dependency and holds no executor.

use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    task::{Context, Poll, Waker},
};

enum State<T> {
    /// Not completed yet; holds the waker of the last poll.
    Pending(Option<Waker>),
    Ready(T),
    /// The value was handed to the awaiting side. Completing again stays rejected.
    Taken,
}

/// A value that is delivered once, from a callback, to a future.
///
/// Clones share one slot: keep one clone to `.await` and move another into the callback. The first
/// [`complete`](Self::complete) wins and wakes the waker stored by the last poll; any later one is rejected and hands its
/// value back. Awaiting it again after it returned its value panics.
pub(crate) struct Completion<T> {
    state: Arc<Mutex<State<T>>>,
}

impl<T> Clone for Completion<T> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

impl<T> Completion<T> {
    pub(crate) fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(State::Pending(None))),
        }
    }

    /// Deliver `value` and wake the awaiting task. A completion that already holds or has delivered a value
    /// rejects this one: it returns `Err(value)` and changes nothing, so the callback of a slot is safe to
    /// call twice.
    pub(crate) fn complete(&self, value: T) -> Result<(), T> {
        let previous = {
            let mut state = self.lock();
            if !matches!(*state, State::Pending(_)) {
                return Err(value);
            }
            std::mem::replace(&mut *state, State::Ready(value))
        };
        // Wake outside the lock: a waker may poll the future on this thread at once.
        if let State::Pending(Some(waker)) = previous {
            waker.wake();
        }
        Ok(())
    }

    fn lock(&self) -> MutexGuard<'_, State<T>> {
        // No code runs under the lock that can leave the state half-written, so a poisoned lock is still consistent.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<T> Future for Completion<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        let mut state = self.lock();
        match &mut *state {
            State::Pending(waker) => {
                if !waker
                    .as_ref()
                    .is_some_and(|stored| stored.will_wake(cx.waker()))
                {
                    *waker = Some(cx.waker().clone());
                }
                Poll::Pending
            }
            State::Ready(_) => match std::mem::replace(&mut *state, State::Taken) {
                State::Ready(value) => Poll::Ready(value),
                _ => unreachable!("the state was Ready under the same lock"),
            },
            State::Taken => panic!("Completion polled after it returned its value"),
        }
    }
}

/// The wait of the blocking readbacks (native only: the browser cannot block, and its readbacks are awaited).
///
/// `Device::poll` runs the `map_async` callbacks that its own `maintain` collected, and `maintain` collects those of
/// every finished submission, whoever requested them. When another thread's poll collects this thread's callback,
/// this thread's `poll(Wait)` returns before the callback has run, so one device poll followed by one check of the
/// result is a race. A blocking readback therefore waits on the completion itself, which the callback signals from
/// whichever thread runs it.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod blocking {
    use std::{
        future::Future,
        pin::pin,
        sync::{Arc, mpsc},
        task::{Context, Poll, Wake, Waker},
        thread,
        time::{Duration, Instant},
    };

    use crate::Error;

    /// How long a blocking readback waits for its callback after the device wait returned. The GPU work is done by
    /// then, so a callback that does not arrive in this time is lost; the readback fails with `Error::Map`.
    pub(crate) const MAPPING_WAIT: Duration = Duration::from_secs(30);

    /// How long the thread parks before it polls the device again, in case no callback is in flight at all.
    const REPOLL: Duration = Duration::from_millis(10);

    struct ThreadWake(thread::Thread);

    impl Wake for ThreadWake {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }

    /// Drive `future` on this thread: poll it, park until it wakes the thread (or [`REPOLL`] passes, which polls the
    /// device again), and give up after `limit` with `None`. A readback future polls the device on every poll, so a
    /// callback that sits unfired in this device's queue is run here.
    pub(crate) fn block_on_for<F: Future>(future: F, limit: Duration) -> Option<F::Output> {
        let mut future = pin!(future);
        let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
        let mut context = Context::from_waker(&waker);
        let deadline = Instant::now() + limit;
        loop {
            if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
                return Some(output);
            }
            let remaining = deadline.checked_duration_since(Instant::now())?;
            thread::park_timeout(remaining.min(REPOLL));
        }
    }

    /// The error of a mapping whose callback did not arrive within [`MAPPING_WAIT`].
    pub(crate) fn mapping_timeout() -> Error {
        Error::Map(format!(
            "mapping not complete {} s after the device wait",
            MAPPING_WAIT.as_secs()
        ))
    }

    /// Drive a readback future to its result, or `Error::Map` when it is still pending after [`MAPPING_WAIT`].
    pub(crate) fn block_on_mapping<T>(
        future: impl Future<Output = Result<T, Error>>,
    ) -> Result<T, Error> {
        block_on_for(future, MAPPING_WAIT).unwrap_or_else(|| Err(mapping_timeout()))
    }

    /// The result of a `map_async` callback that sends on a channel, waiting at most [`MAPPING_WAIT`].
    pub(crate) fn recv_mapping(receiver: &mpsc::Receiver<Result<(), String>>) -> Result<(), Error> {
        match receiver.recv_timeout(MAPPING_WAIT) {
            Ok(result) => result.map_err(Error::Map),
            Err(mpsc::RecvTimeoutError::Timeout) => Err(mapping_timeout()),
            Err(error @ mpsc::RecvTimeoutError::Disconnected) => Err(Error::Map(error.to_string())),
        }
    }

    #[cfg(test)]
    mod tests {
        use std::sync::atomic::{AtomicBool, Ordering};

        use super::*;
        use crate::completion::Completion;

        #[test]
        fn a_ready_future_returns_at_once() {
            assert_eq!(block_on_for(std::future::ready(5), Duration::ZERO), Some(5));
        }

        #[test]
        fn a_completion_finished_by_another_thread_ends_the_blocked_wait() {
            let completion = Completion::new();
            let callback = completion.clone();
            let started = Instant::now();
            let sender = thread::spawn(move || {
                thread::sleep(Duration::from_millis(100));
                let _ = callback.complete("late");
            });
            // The limit is far above the delay, so it is the callback's wake that ends the wait.
            assert_eq!(
                block_on_for(completion, Duration::from_secs(20)),
                Some("late")
            );
            assert!(started.elapsed() < Duration::from_secs(10));
            sender.join().unwrap();
        }

        #[test]
        fn a_completion_that_never_arrives_times_out_instead_of_hanging() {
            let completion = Completion::<()>::new();
            let started = Instant::now();
            assert_eq!(block_on_for(completion, Duration::from_millis(60)), None);
            let waited = started.elapsed();
            assert!(
                waited >= Duration::from_millis(60),
                "gave up early: {waited:?}"
            );
            assert!(
                waited < Duration::from_secs(10),
                "waited too long: {waited:?}"
            );
        }

        #[test]
        fn a_timed_out_mapping_is_a_map_error_that_names_the_wait() {
            let error = mapping_timeout();
            assert!(
                matches!(&error, Error::Map(text) if text == "mapping not complete 30 s after the device wait"),
                "{error}"
            );
        }

        #[test]
        fn a_channel_result_arrives_from_another_thread_and_errors_are_kept() {
            let (sender, receiver) = mpsc::channel();
            let flag = Arc::new(AtomicBool::new(false));
            let seen = Arc::clone(&flag);
            let worker = thread::spawn(move || {
                thread::sleep(Duration::from_millis(50));
                seen.store(true, Ordering::SeqCst);
                let _ = sender.send(Err("device lost".to_owned()));
            });
            let result = recv_mapping(&receiver);
            assert!(
                flag.load(Ordering::SeqCst),
                "returned before the sender ran"
            );
            assert!(matches!(result, Err(Error::Map(text)) if text == "device lost"));
            worker.join().unwrap();
            // The sender is gone and nothing is queued: a disconnect, not a hang.
            assert!(matches!(recv_mapping(&receiver), Err(Error::Map(_))));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc,
        },
        task::Wake,
        thread,
        time::Duration,
    };

    use super::*;

    /// A waker that counts how often it was woken.
    struct CountingWake(AtomicUsize);

    impl CountingWake {
        fn new() -> (Arc<Self>, Waker) {
            let counter = Arc::new(Self(AtomicUsize::new(0)));
            (Arc::clone(&counter), Waker::from(counter))
        }

        fn count(&self) -> usize {
            self.0.load(Ordering::SeqCst)
        }
    }

    impl Wake for CountingWake {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn poll_once<T>(completion: &mut Completion<T>, waker: &Waker) -> Poll<T> {
        Pin::new(completion).poll(&mut Context::from_waker(waker))
    }

    #[test]
    fn completes_once_and_rejects_a_second_value() {
        let (_, waker) = CountingWake::new();
        let mut completion = Completion::new();
        assert!(matches!(poll_once(&mut completion, &waker), Poll::Pending));

        assert_eq!(completion.complete(1), Ok(()));
        assert_eq!(completion.complete(2), Err(2));
        assert!(matches!(poll_once(&mut completion, &waker), Poll::Ready(1)));
        // After the value was taken, a late callback is still rejected and nothing is stored.
        assert_eq!(completion.complete(3), Err(3));
    }

    #[test]
    fn completing_wakes_the_stored_waker_exactly_once() {
        let (wakes, waker) = CountingWake::new();
        let mut completion = Completion::new();
        assert!(matches!(poll_once(&mut completion, &waker), Poll::Pending));
        assert_eq!(wakes.count(), 0, "polling alone never wakes");

        assert_eq!(completion.clone().complete("done"), Ok(()));
        assert_eq!(wakes.count(), 1);
        assert_eq!(completion.complete("again"), Err("again"));
        assert_eq!(wakes.count(), 1, "a rejected completion does not wake");
        assert!(matches!(
            poll_once(&mut completion, &waker),
            Poll::Ready("done")
        ));
    }

    #[test]
    fn a_later_poll_replaces_the_stored_waker() {
        let (first_wakes, first) = CountingWake::new();
        let (second_wakes, second) = CountingWake::new();
        let mut completion = Completion::new();
        assert!(matches!(poll_once(&mut completion, &first), Poll::Pending));
        assert!(matches!(poll_once(&mut completion, &second), Poll::Pending));

        assert_eq!(completion.complete(()), Ok(()));
        assert_eq!((first_wakes.count(), second_wakes.count()), (0, 1));
    }

    #[test]
    fn a_value_set_before_the_first_poll_is_ready_without_a_wake() {
        let (wakes, waker) = CountingWake::new();
        let mut completion = Completion::new();
        assert_eq!(completion.complete(7), Ok(()));
        assert!(matches!(poll_once(&mut completion, &waker), Poll::Ready(7)));
        assert_eq!(wakes.count(), 0);
    }

    #[test]
    fn a_callback_on_another_thread_wakes_a_blocked_await() {
        let (done, outcome) = mpsc::channel();
        thread::spawn(move || {
            let completion = Completion::new();
            let callback = completion.clone();
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(50));
                let _ = callback.complete(42);
            });
            let _ = done.send(pollster::block_on(completion));
        });
        // A completion that never wakes would park `block_on` forever; the timeout turns that into a failure.
        assert_eq!(
            outcome.recv_timeout(Duration::from_secs(20)),
            Ok(42),
            "the await was not woken by the callback"
        );
    }

    #[test]
    #[should_panic(expected = "polled after it returned its value")]
    fn polling_after_the_value_was_taken_panics() {
        let (_, waker) = CountingWake::new();
        let mut completion = Completion::new();
        assert_eq!(completion.complete(1), Ok(()));
        assert!(matches!(poll_once(&mut completion, &waker), Poll::Ready(1)));
        let _ = poll_once(&mut completion, &waker);
    }
}
