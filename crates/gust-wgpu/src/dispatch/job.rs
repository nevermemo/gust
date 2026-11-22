use crate::Error;
#[cfg(doc)]
use crate::HeadlessDevice;
use gust::JobStatus;
use std::time::Duration;

/// Fence for one submitted wgpu command buffer.
///
/// # On `wasm32`
///
/// The browser cannot block, so the job reports completion without polling and refuses to wait:
///
/// - [`status`](WgpuJob::status) (and [`is_complete`](WgpuJob::is_complete)) is `Pending` until wgpu's
///   `Queue::on_submitted_work_done` callback for this submission has run, and `Complete` after it. A job that
///   submitted nothing (an empty batch) is `Complete`. The callback runs on the browser's event loop, so a status
///   read in the same synchronous run of code as the submission is `Pending`.
/// - [`wait`](WgpuJob::wait) and [`wait_timeout`](WgpuJob::wait_timeout) always return [`Error::WouldBlock`], never
///   `Ok`, so a caller cannot take a finished GPU for granted. Any work the job submitted has been submitted and will
///   run; only the wait is refused. Synchronise with an awaited readback
///   ([`read_typed_buffer_async`](HeadlessDevice::read_typed_buffer_async)), which resolves when the work and the copy
///   are done.
///
/// Every blocking convenience built on `wait` fails the same way on `wasm32`, after it has submitted the work: the
/// generated `dispatch` and `dispatch_extent` of a kernel, `gust::Run::dispatch_call` and `gust::Device::dispatch`
/// return that error at run time (use `submit` and an awaited readback there), and the `HeadlessDevice` helpers that
/// wait are compiled out.
///
/// On native, `submit` starts a call and returns the job; `wait` blocks until that submission is done:
///
/// ```
/// use gust::{JobStatus, gpu};
/// use gust_wgpu::{GpuBufferAccess, HeadlessDevice};
///
/// #[gpu]
/// mod double {
///     #[kernel(workgroup_size(64, 1, 1))]
///     pub fn run(id: SV_DispatchThreadID, mut out: RWStructuredBuffer<float>) {
///         let i = id.x;
///         if i < out.len() {
///             out[i] = out[i] * 2.0;
///         }
///     }
/// }
///
/// fn main() -> Result<(), gust_wgpu::Error> {
///     let device = HeadlessDevice::new()?;
///     let out = device.create_typed_buffer("out", &[1.0f32, 2.0], GpuBufferAccess::ReadWrite);
///     let job = double::run::submit(&device, 2, &out)?;
///     job.wait()?;
///     assert!(job.is_complete()?);
///     assert_eq!(job.status()?, JobStatus::Complete);
///     assert_eq!(device.read_typed_buffer(&out)?, [2.0, 4.0]);
///     Ok(())
/// }
/// ```
#[must_use = "GPU work is asynchronous; poll, await a readback or wait on the returned job when completion matters"]
pub struct WgpuJob<'device> {
    // Only the native `poll_with_timeout` reads these two.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) device: &'device wgpu::Device,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) submission: Option<wgpu::SubmissionIndex>,
    /// Set by the `on_submitted_work_done` callback of this job's submission (`true` from the start for a job that
    /// submitted nothing). An `Arc<AtomicBool>` is `Send + Sync` and the callback must be `Send`; the field exists on
    /// `wasm32` only, so the auto traits of `WgpuJob` on native are what they were.
    #[cfg(target_arch = "wasm32")]
    done: Arc<std::sync::atomic::AtomicBool>,
}

impl WgpuJob<'_> {
    /// Poll the exact submission once without blocking.
    ///
    /// On `wasm32` this reads the flag that `Queue::on_submitted_work_done` sets: `Pending` until the callback has
    /// run, then `Complete`; a job with no submission is `Complete`. A lost device also reads `Complete` there (the
    /// WebGPU specification runs the completion steps when the device is lost, and wgpu calls back when the promise
    /// rejects), where native returns `Err(Error::Poll)`.
    pub fn status(&self) -> Result<JobStatus, Error> {
        #[cfg(target_arch = "wasm32")]
        {
            Ok(if self.done.load(Ordering::Acquire) {
                JobStatus::Complete
            } else {
                JobStatus::Pending
            })
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_with_timeout(Some(Duration::ZERO))
    }

    /// Whether the submission has completed, from one non-blocking poll ([`WgpuJob::status`]).
    pub fn is_complete(&self) -> Result<bool, Error> {
        self.status().map(JobStatus::is_complete)
    }

    /// Wait for at most `timeout`.
    ///
    /// # On `wasm32`
    ///
    /// Always returns [`Error::WouldBlock`]: the browser cannot block, and `timeout` is ignored. Use
    /// [`status`](WgpuJob::status) or an awaited readback
    /// ([`read_typed_buffer_async`](HeadlessDevice::read_typed_buffer_async)).
    pub fn wait_timeout(&self, timeout: Duration) -> Result<JobStatus, Error> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = timeout;
            Err(Error::WouldBlock("wait_timeout"))
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_with_timeout(Some(timeout))
    }

    /// Wait indefinitely for this submission, without waiting for later work.
    ///
    /// # On `wasm32` this returns an error
    ///
    /// The browser cannot block, so on `wasm32` this always returns [`Error::WouldBlock`], never `Ok`, even for a job
    /// that has already completed or submitted nothing. Any work the job submitted will run; only the wait is
    /// refused. The synchronisation on the web is an awaited readback
    /// ([`read_typed_buffer_async`](HeadlessDevice::read_typed_buffer_async)), which resolves only when the work and the
    /// copy are done; [`status`](WgpuJob::status) reports completion without waiting.
    pub fn wait(&self) -> Result<(), Error> {
        #[cfg(target_arch = "wasm32")]
        {
            Err(Error::WouldBlock("wait"))
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_with_timeout(None).map(|_| ())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn poll_with_timeout(&self, timeout: Option<Duration>) -> Result<JobStatus, Error> {
        let Some(submission) = &self.submission else {
            return Ok(JobStatus::Complete);
        };
        match self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission.clone()),
            timeout,
        }) {
            Ok(status) if status.wait_finished() => Ok(JobStatus::Complete),
            Ok(_) | Err(wgpu::PollError::Timeout) => Ok(JobStatus::Pending),
            Err(error) => Err(Error::Poll(error.to_string())),
        }
    }
}

impl std::fmt::Debug for WgpuJob<'_> {
    /// Whether the job submitted work; never the raw wgpu handles.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuJob")
            .field("submitted", &self.submission.is_some())
            .finish_non_exhaustive()
    }
}

impl gust::Job for WgpuJob<'_> {
    type Error = Error;

    fn status(&self) -> Result<JobStatus, Self::Error> {
        Self::status(self)
    }

    fn wait_timeout(&self, timeout: Duration) -> Result<JobStatus, Self::Error> {
        Self::wait_timeout(self, timeout)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        Self::wait(self)
    }
}
