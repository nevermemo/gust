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
