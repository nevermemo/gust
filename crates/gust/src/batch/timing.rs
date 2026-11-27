//! Frame timing: the GPU time of the passes of a frame, opt-in at the build of a batch ([`FrameTiming`]), asked for per
//! frame with [`FrameRun::time`](super::FrameRun::time) and read, without blocking, through a [`FrameTime`].

use core::any::Any;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
#[cfg(target_arch = "wasm32")]
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use super::{BoxFuture, Error};
use crate::map;

/// How much GPU time a batch can report, chosen when the batch is built (the timestamp objects are made then, and only
/// when the level is not `Off` and the device has timestamp queries).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FrameTiming {
    /// No timing: the batch makes no timestamp object and a frame records none (the default).
    #[default]
    Off,
    /// One span per timed frame ([`FrameTimes::span`]).
    Frame,
    /// The span, and one entry per stage that has a pass ([`FrameTimes::passes`]). A device that merges the passes of
    /// neighbouring stages when a frame is untimed or at `Frame` (the GPU device does) and can write timestamps inside a
    /// pass (a native device with `TIMESTAMP_QUERY_INSIDE_PASSES`) records the frame as it records it untimed, with a
    /// begin and an end timestamp for each stage. A device that merges but cannot (a browser device, or one opened
    /// without the feature) records every stage in passes of its own at this level, so the frame is not the one that
    /// runs at the other levels: a stage's time includes its pass boundary, and the times can sum to more than the
    /// `Frame` span. In the browser this is the only form, because WebGPU has no timestamps inside passes.
    Passes,
}

/// What a batch can tell about its frames' time, from [`Batch::frame_timing`](super::Batch::frame_timing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct TimingInfo {
    /// The level the batch was built with (never `Off`: a batch with no timing has no `TimingInfo`).
    pub level: FrameTiming,
    /// The adapter's stated timestamp tick. A value is a whole number of ticks times this; the spread of repeated
    /// frames is a measured fact of each adapter, not part of this type. In the browser the tick is stated as 1 ns and
    /// the values are quantized by the browser (measured 65536 ns on Chrome 158, Radeon 890M only; RTX 5090 pending), so
    /// a span is a multiple of that step and a frame shorter than the step can read as 0.
    pub tick: Duration,
}

/// The time of the passes of one stage of a frame (or, when passes are fused, of one group of stages).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PassTime {
    /// The stages the passes cover, by name.
    pub stages: Vec<String>,
    /// From the begin of the first pass to the end of the last pass of the entry.
    pub duration: Duration,
}

/// The GPU time of one frame.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct FrameTimes {
    /// From the begin of the frame's first pass to the end of its last one, whatever lies between them. The
    /// frame's upload copies (before the first pass) and its ticket copies (after the last) are **not** in it. It is
    /// the quantity the benchmarks call `gpu_batch_per_dispatch` times the number of passes.
    pub span: Duration,
    /// One entry per stage that has a pass, in the order of the stages; empty at level [`FrameTiming::Frame`].
    pub passes: Vec<PassTime>,
}

/// The timing objects of a device for one batch, kept by the batch: `Arc<dyn Any + Send + Sync>` on native targets and
/// `Rc<dyn Any>` on `wasm32`, where the browser device's objects are not `Send`. Macro support.
#[doc(hidden)]
#[cfg(not(target_arch = "wasm32"))]
pub type DeviceTiming = Arc<dyn Any + Send + Sync>;

/// The timing objects of a device for one batch, kept by the batch (see the native alias). Macro support.
#[doc(hidden)]
#[cfg(target_arch = "wasm32")]
pub type DeviceTiming = Rc<dyn Any>;

/// What a frame holds while it is recorded and until it is submitted. Macro support.
#[doc(hidden)]
pub type FrameClock = Box<dyn Any>;

/// What a device made for a batch's timing. Macro support.
#[doc(hidden)]
pub struct OpenedTiming {
    /// The device's objects.
    pub handle: DeviceTiming,
    /// The length of one timestamp tick, in nanoseconds.
    pub period_ns: f64,
}

/// The timing a batch was built with: the level and the device's objects.
pub(crate) struct TimingState {
    pub(crate) level: FrameTiming,
    pub(crate) handle: DeviceTiming,
    pub(crate) period_ns: f64,
}

impl TimingState {
    pub(crate) fn info(&self) -> TimingInfo {
        TimingInfo {
            level: self.level,
            tick: duration_of(1, self.period_ns),
        }
    }
}

/// `ticks` timestamp ticks of `period_ns` nanoseconds each.
fn duration_of(ticks: u64, period_ns: f64) -> Duration {
    Duration::from_secs_f64(ticks as f64 * period_ns * 1e-9)
}

/// How a frame's raw timestamps turn into [`FrameTimes`]: the stages that were timed, in order, and the level.
pub(crate) struct TimePlan {
    pub(crate) level: FrameTiming,
    pub(crate) names: Vec<String>,
    pub(crate) period_ns: f64,
}

impl TimePlan {
    /// The slots a timed stage writes at this level: the first stage's first pass writes slot 0 and the last stage's last
    /// pass the last slot used at level `Frame`; at level `Passes` stage `group` writes `2 * group` and `2 * group + 1`.
    pub(crate) fn slots(&self, group: usize) -> (Option<u32>, Option<u32>) {
        let last = self.names.len() - 1;
        match self.level {
            FrameTiming::Passes => {
                let begin = u32::try_from(2 * group).expect("a batch has fewer than 2^31 stages");
                (Some(begin), Some(begin + 1))
            }
            _ => ((group == 0).then_some(0), (group == last).then_some(1)),
        }
    }

    /// How many slots a frame of this plan writes.
    pub(crate) fn used(&self) -> u32 {
        match self.level {
            FrameTiming::Passes => {
                u32::try_from(2 * self.names.len()).expect("a batch has fewer than 2^31 stages")
            }
            _ => 2,
        }
    }

    /// The times the raw timestamps `ticks` make, or `None` when there are not as many as the plan wrote.
    fn times(&self, ticks: &[u64]) -> Option<FrameTimes> {
        let used = self.used() as usize;
        if ticks.len() != used {
            return None;
        }
        let span = duration_of(ticks[used - 1].saturating_sub(ticks[0]), self.period_ns);
        let passes = match self.level {
            FrameTiming::Passes => self
                .names
                .iter()
                .zip(ticks.as_chunks::<2>().0)
                .map(|(name, pair)| PassTime {
                    stages: vec![name.clone()],
                    duration: duration_of(pair[1].saturating_sub(pair[0]), self.period_ns),
                })
                .collect(),
            _ => Vec::new(),
        };
        Some(FrameTimes { span, passes })
    }
}

/// How a frame's time ended.
#[derive(Clone)]
enum Resolved {
    Times(Option<FrameTimes>),
    NotSubmitted,
    Failed(String),
}

enum TimeState {
    /// The frame has not been submitted; holds the wakers of the last polls.
    Reserved(Vec<Waker>),
    /// The frame was submitted: the read of its timestamps, in ticks.
    Running(
        BoxFuture<Result<Option<Vec<u64>>, map::Error>>,
        Arc<TimePlan>,
    ),
    /// The outcome, which every handle reads.
    Done(Resolved),
}

pub(crate) struct TimeCell {
    batch: String,
    state: Mutex<TimeState>,
}

impl TimeCell {
    pub(crate) fn new(batch: &str) -> Self {
        Self {
            batch: batch.to_owned(),
            state: Mutex::new(TimeState::Reserved(Vec::new())),
        }
    }

    fn lock(&self) -> MutexGuard<'_, TimeState> {
        // No code runs under the lock that can leave the state half-written.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn settle(&self, next: impl FnOnce() -> TimeState) {
        let wakers = {
            let mut state = self.lock();
            let TimeState::Reserved(wakers) = &mut *state else {
                return;
            };
            let wakers = core::mem::take(wakers);
            *state = next();
            wakers
        };
        for waker in wakers {
            waker.wake();
        }
    }

    /// The frame was submitted and its timestamps are read by `read`.
    pub(crate) fn start(
        &self,
        read: BoxFuture<Result<Option<Vec<u64>>, map::Error>>,
        plan: TimePlan,
    ) {
        self.settle(|| TimeState::Running(read, Arc::new(plan)));
    }

    /// The frame was submitted and there is nothing to read.
    pub(crate) fn none(&self) {
        self.settle(|| TimeState::Done(Resolved::Times(None)));
    }

    /// The frame was never submitted.
    pub(crate) fn fail(&self) {
        self.settle(|| TimeState::Done(Resolved::NotSubmitted));
    }

    fn outcome(&self, resolved: Resolved) -> Result<Option<FrameTimes>, Error> {
        match resolved {
            Resolved::Times(times) => Ok(times),
            Resolved::NotSubmitted => Err(Error::NotSubmitted {
                batch: self.batch.clone(),
            }),
            Resolved::Failed(message) => Err(Error::Device(map::Error::device(
                std::io::Error::other(message),
            ))),
        }
    }
}

/// The GPU time of one frame, asked for with [`FrameRun::time`](super::FrameRun::time): a future of the
/// [`FrameTimes`], read the way a [`Ticket`](super::Ticket) is (it is polled, never waited for, and it may arrive
/// frames after the frame). A frame the device gave no time for resolves to `Ok(None)`: a batch built with
/// [`FrameTiming::Off`], a device without timestamp queries (including the CPU twin, and a browser that does not grant
/// the feature), a frame with no pass, or no free read-back buffer when the frame was recorded (the batch keeps a
/// bounded set, replaced as frames drop or fail, so this is rare). A frame that is never submitted resolves to
/// [`Error::NotSubmitted`], and so does a frame whose timestamps were recorded into a caller's encoder that never ran
/// (`Encoded::submitted` with the index of another submission); a frame that recorded none (built `Off`, no pass) is
/// `Ok(None)` there.
///
/// In the browser the values are quantized by the browser (measured 65536 ns on Chrome 158, Radeon 890M only; RTX 5090
/// pending), so a span is a multiple of that step and a frame shorter than the step can read as 0.
///
/// Clones read the same measurement; a second `.await` of the same handle returns it again. Native GPU devices complete
/// it only while the device is polled, which polling the future does (see [`Ticket`](super::Ticket)).
#[derive(Clone)]
#[must_use = "a frame time is a future: await it, or poll it"]
pub struct FrameTime {
    pub(crate) cell: Arc<TimeCell>,
}

impl core::fmt::Debug for FrameTime {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FrameTime").finish_non_exhaustive()
    }
}

impl Future for FrameTime {
    type Output = Result<Option<FrameTimes>, Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.cell.lock();
        let resolved = match &mut *state {
            TimeState::Reserved(wakers) => {
                if !wakers.iter().any(|stored| stored.will_wake(cx.waker())) {
                    wakers.push(cx.waker().clone());
                }
                return Poll::Pending;
            }
            TimeState::Running(read, plan) => match read.as_mut().poll(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Ok(Some(ticks))) => Resolved::Times(plan.times(&ticks)),
                // The copy never ran: the encoder it was recorded into was not submitted.
                Poll::Ready(Ok(None)) => Resolved::NotSubmitted,
                Poll::Ready(Err(error)) => Resolved::Failed(error.to_string()),
            },
            TimeState::Done(resolved) => return Poll::Ready(self.cell.outcome(resolved.clone())),
        };
        *state = TimeState::Done(resolved.clone());
        Poll::Ready(self.cell.outcome(resolved))
    }
}

// A frame time outlives its batch and moves between threads on native targets, as a ticket does.
#[cfg(not(target_arch = "wasm32"))]
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<FrameTime>();
};

/// What a timed frame keeps between `time()` and its submission.
pub(crate) struct TimedFrame {
    pub(crate) cell: Arc<TimeCell>,
    /// Set by the recording of the frame when it wrote timestamps.
    pub(crate) recorded: Option<Recorded>,
}

pub(crate) struct Recorded {
    pub(crate) clock: FrameClock,
    pub(crate) plan: TimePlan,
}
