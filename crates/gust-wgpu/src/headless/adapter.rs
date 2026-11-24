use crate::Error;
use std::future::Future;
use std::time::Duration;

/// Adapter requests made by `HeadlessDevice::new` before `NoAdapter` is returned.
pub(crate) const ADAPTER_ATTEMPTS: u32 = 3;
/// Pauses after the first and second failed adapter request.
pub(crate) const ADAPTER_BACKOFF: [Duration; 2] =
    [Duration::from_millis(100), Duration::from_millis(400)];

/// Run `request` up to `attempts` times, pausing `backoff[n]` through `pause` after the
/// `n`th failure. Only `Error::NoAdapter` is retried; any other result returns at once.
/// The final `NoAdapter` keeps its message and gains the attempt count. When a request
/// succeeds after at least one failed attempt, one line goes to `log`, so the transient
/// failure stays visible instead of being absorbed by the retry.
///
/// Both the request and the pause are futures, so the same loop serves the browser, where
/// nothing may block. `pause` is awaited between attempts; on native it sleeps the thread and
/// returns a ready future (`pause_between_attempts`).
pub(crate) async fn retry_no_adapter<
    T,
    Request: Future<Output = Result<T, Error>>,
    Pause: Future<Output = ()>,
>(
    attempts: u32,
    backoff: &[Duration],
    mut pause: impl FnMut(Duration) -> Pause,
    mut log: impl FnMut(&str),
    mut request: impl FnMut() -> Request,
) -> Result<T, Error> {
    let mut attempt = 1;
    let mut first_failure: Option<String> = None;
    loop {
        match request().await {
            Err(Error::NoAdapter(message)) if attempt >= attempts => {
                return Err(Error::NoAdapter(format!(
                    "{message} (after {attempt} attempts)"
                )));
            }
            Err(Error::NoAdapter(message)) => {
                first_failure.get_or_insert(message);
                if let Some(delay) = backoff.get(attempt as usize - 1) {
                    pause(*delay).await;
                }
                attempt += 1;
            }
            other => {
                if let (Ok(_), Some(message)) = (&other, &first_failure) {
                    log(&format!(
                        "gust-compute-wgpu: adapter request succeeded on attempt {attempt} of {attempts} after a transient NoAdapter failure: {message}"
                    ));
                }
                return other;
            }
        }
    }
}

/// Run `list` up to `attempts` times until it returns a non-empty list, pausing `backoff[n]` through `pause`
/// after the `n`th empty one. A machine with no adapter still gets an empty list, only after the last attempt.
/// When a list is non-empty after at least one empty attempt, one line goes to `log`, so the transient failure
/// stays visible. Shares the attempts, backoff and futures contract of [`retry_no_adapter`].
pub(crate) async fn retry_empty_list<
    T,
    List: Future<Output = Vec<T>>,
    Pause: Future<Output = ()>,
>(
    attempts: u32,
    backoff: &[Duration],
    pause: impl FnMut(Duration) -> Pause,
    log: impl FnMut(&str),
    mut list: impl FnMut() -> List,
) -> Vec<T> {
    retry_no_adapter(attempts, backoff, pause, log, || {
        let pending = list();
        async move {
            let found = pending.await;
            if found.is_empty() {
                Err(Error::NoAdapter("the adapter list was empty".to_owned()))
            } else {
                Ok(found)
            }
        }
    })
    .await
    .unwrap_or_default()
}

/// The pause between two adapter requests. Native sleeps the calling thread and returns a ready future.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn pause_between_attempts(delay: Duration) -> std::future::Ready<()> {
    std::thread::sleep(delay);
    std::future::ready(())
}

/// The pause between two adapter requests. A browser thread cannot sleep (`std::thread::sleep` panics on
/// `wasm32-unknown-unknown`), so the next request follows at once.
#[cfg(target_arch = "wasm32")]
pub(crate) fn pause_between_attempts(_delay: Duration) -> std::future::Ready<()> {
    std::future::ready(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use std::pin::Pin;
    use std::task::{Context, Poll};
    use std::time::Duration;

    fn no_adapter() -> Error {
        Error::NoAdapter("loader failed".to_string())
    }

    /// The retry loop with synchronous closures, driven to completion: these tests pin the loop's decisions, not its
    /// futures.
    fn retry_no_adapter<T>(
        attempts: u32,
        backoff: &[Duration],
        mut pause: impl FnMut(Duration),
        log: impl FnMut(&str),
        mut request: impl FnMut() -> Result<T, Error>,
    ) -> Result<T, Error> {
        pollster::block_on(super::retry_no_adapter(
            attempts,
            backoff,
            |delay| {
                pause(delay);
                std::future::ready(())
            },
            log,
            || std::future::ready(request()),
        ))
    }

    /// Pending on its first poll (waking itself), ready on the next.
    struct YieldOnce(bool);

    impl Future for YieldOnce {
        type Output = ();

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if self.0 {
                return Poll::Ready(());
            }
            self.0 = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }

    #[test]
    fn adapter_retry_awaits_requests_and_pauses_that_are_not_ready() {
        let mut calls = 0;
        let mut pauses = Vec::new();
        let result = pollster::block_on(super::retry_no_adapter(
            ADAPTER_ATTEMPTS,
            &ADAPTER_BACKOFF,
            |delay| {
                pauses.push(delay);
                YieldOnce(false)
            },
            |_| {},
            || {
                calls += 1;
                let outcome = if calls < ADAPTER_ATTEMPTS {
                    Err(no_adapter())
                } else {
                    Ok(calls)
                };
                async move {
                    YieldOnce(false).await;
                    outcome
                }
            },
        ));
        assert!(matches!(result, Ok(n) if n == ADAPTER_ATTEMPTS));
        assert_eq!(pauses, ADAPTER_BACKOFF);
    }

    #[test]
    fn adapter_retry_succeeds_on_first_attempt_without_pausing() {
        let mut calls = 0;
        let mut pauses = Vec::new();
        let mut logs = Vec::new();
        let result = retry_no_adapter(
            ADAPTER_ATTEMPTS,
            &ADAPTER_BACKOFF,
            |delay| pauses.push(delay),
            |line| logs.push(line.to_string()),
            || {
                calls += 1;
                Ok(7)
            },
        );
        assert!(matches!(result, Ok(7)));
        assert_eq!(calls, 1);
        assert!(pauses.is_empty());
        assert!(logs.is_empty(), "no retry happened, so nothing is logged");
    }

    fn retry_empty_list<T>(
        mut pause: impl FnMut(Duration),
        log: impl FnMut(&str),
        mut list: impl FnMut() -> Vec<T>,
    ) -> Vec<T> {
        pollster::block_on(super::retry_empty_list(
            ADAPTER_ATTEMPTS,
            &ADAPTER_BACKOFF,
            |delay| {
                pause(delay);
                std::future::ready(())
            },
            log,
            || std::future::ready(list()),
        ))
    }

    #[test]
    fn adapter_enumeration_retries_an_empty_list_and_logs_once() {
        let mut calls = 0;
        let mut pauses = Vec::new();
        let mut logs = Vec::new();
        let found = retry_empty_list(
            |delay| pauses.push(delay),
            |line| logs.push(line.to_string()),
            || {
                calls += 1;
                if calls < 2 { Vec::new() } else { vec![calls] }
            },
        );
        assert_eq!(found, [2]);
        assert_eq!(pauses, ADAPTER_BACKOFF[..1]);
        assert_eq!(
            logs,
            [format!(
                "gust-compute-wgpu: adapter request succeeded on attempt 2 of {ADAPTER_ATTEMPTS} after a transient NoAdapter failure: the adapter list was empty"
            )]
        );
    }

    #[test]
    fn adapter_enumeration_returns_empty_after_the_attempts() {
        let mut calls = 0;
        let mut pauses = Vec::new();
        let mut logs = Vec::new();
        let found: Vec<u32> = retry_empty_list(
            |delay| pauses.push(delay),
            |line| logs.push(line.to_string()),
            || {
                calls += 1;
                Vec::new()
            },
        );
        assert!(found.is_empty());
        assert_eq!(calls, ADAPTER_ATTEMPTS);
        assert_eq!(pauses, ADAPTER_BACKOFF);
        assert!(
            logs.is_empty(),
            "no adapter was found, so nothing is logged"
        );
    }

    #[test]
    fn adapter_enumeration_returns_a_first_list_at_once() {
        let mut calls = 0;
        let mut pauses = Vec::new();
        let mut logs = Vec::new();
        let found = retry_empty_list(
            |delay| pauses.push(delay),
            |line| logs.push(line.to_string()),
            || {
                calls += 1;
                vec![7]
            },
        );
        assert_eq!(found, [7]);
        assert_eq!(calls, 1);
        assert!(pauses.is_empty());
        assert!(logs.is_empty());
    }

    #[test]
    fn adapter_backoff_has_one_pause_between_each_pair_of_attempts() {
        assert_eq!(ADAPTER_BACKOFF.len() as u32, ADAPTER_ATTEMPTS - 1);
    }

    #[test]
    fn adapter_retry_succeeds_on_a_later_attempt() {
        let mut calls = 0;
        let mut pauses = Vec::new();
        let mut logs = Vec::new();
        let result = retry_no_adapter(
            ADAPTER_ATTEMPTS,
            &ADAPTER_BACKOFF,
            |delay| pauses.push(delay),
            |line| logs.push(line.to_string()),
            || {
                calls += 1;
                if calls < ADAPTER_ATTEMPTS {
                    Err(no_adapter())
                } else {
                    Ok(calls)
                }
            },
        );
        assert!(matches!(result, Ok(n) if n == ADAPTER_ATTEMPTS));
        assert_eq!(calls, ADAPTER_ATTEMPTS);
        assert_eq!(pauses, ADAPTER_BACKOFF);
        assert_eq!(
            pauses.iter().sum::<Duration>(),
            ADAPTER_BACKOFF.iter().sum::<Duration>()
        );
        // The transient failure stays visible: exactly one line, naming the attempt
        // that succeeded and the loader message of the failed one.
        assert_eq!(
            logs,
            [format!(
                "gust-compute-wgpu: adapter request succeeded on attempt {ADAPTER_ATTEMPTS} of {ADAPTER_ATTEMPTS} after a transient NoAdapter failure: loader failed"
            )]
        );
    }

    #[test]
    fn adapter_retry_fails_after_the_limit_with_the_attempt_count() {
        let mut calls = 0;
        let mut pauses = Vec::new();
        let mut logs = Vec::new();
        let result: Result<(), Error> = retry_no_adapter(
            ADAPTER_ATTEMPTS,
            &ADAPTER_BACKOFF,
            |delay| pauses.push(delay),
            |line| logs.push(line.to_string()),
            || {
                calls += 1;
                Err(no_adapter())
            },
        );
        assert_eq!(calls, ADAPTER_ATTEMPTS);
        assert_eq!(
            pauses.iter().sum::<Duration>(),
            ADAPTER_BACKOFF.iter().sum::<Duration>()
        );
        // A final failure is reported by the error itself, not by the retry log.
        assert!(logs.is_empty());
        match result {
            Err(Error::NoAdapter(message)) => {
                assert_eq!(
                    message,
                    format!("loader failed (after {ADAPTER_ATTEMPTS} attempts)")
                );
            }
            other => panic!("expected NoAdapter after the limit, got {other:?}"),
        }
    }

    #[test]
    fn adapter_retry_returns_other_errors_immediately() {
        let mut calls = 0;
        let mut pauses = 0;
        let result: Result<(), Error> = retry_no_adapter(
            ADAPTER_ATTEMPTS,
            &ADAPTER_BACKOFF,
            |_| pauses += 1,
            |_| panic!("an immediate non-adapter error is not a retry"),
            || {
                calls += 1;
                Err(Error::RequestDevice("lost".to_string()))
            },
        );
        assert!(matches!(result, Err(Error::RequestDevice(_))));
        assert_eq!(calls, 1);
        assert_eq!(pauses, 0);
    }
}
