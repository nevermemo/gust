use std::time::Duration;

/// Backend-neutral state of an asynchronous GPU submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobStatus {
    /// The work has not finished.
    Pending,
    /// The work has finished.
    Complete,
}

impl JobStatus {
    /// Whether the state is [`JobStatus::Complete`].
    pub const fn is_complete(self) -> bool {
        matches!(self, Self::Complete)
    }
}

/// Backend-neutral completion contract for submitted GPU work.
pub trait Job {
    /// The error a poll or a wait can return.
    type Error;

    /// Poll once without blocking.
    fn status(&self) -> Result<JobStatus, Self::Error>;

    /// Wait for at most `timeout` and return the resulting state.
    fn wait_timeout(&self, timeout: Duration) -> Result<JobStatus, Self::Error>;

    /// Wait until the submission has completed.
    fn wait(&self) -> Result<(), Self::Error>;

    /// Whether the submission has completed, from one non-blocking poll ([`Job::status`]).
    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.status().map(JobStatus::is_complete)
    }
}

/// Backend-neutral device contract for dispatch descriptions and asynchronous jobs.
pub trait Device {
    /// The error a submission, a poll or a wait can return.
    type Error;
    /// One dispatch of the backend, borrowing the resources it reads and writes.
    type Dispatch<'resources>;
    /// The handle of a submission that has not necessarily finished, borrowing the device.
    type Job<'device>: Job<Error = Self::Error>
    where
        Self: 'device;

    /// Submit `dispatches` for execution and return without waiting for them.
    fn submit<'device, 'resources>(
        &'device self,
        dispatches: &[Self::Dispatch<'resources>],
    ) -> Result<Self::Job<'device>, Self::Error>;

    /// Submit `dispatches` and wait until they have completed.
    ///
    /// A device whose job cannot block (`gust-compute-wgpu` on `wasm32`) submits the dispatches, which will run, and then
    /// returns its wait error instead of waiting: use [`Device::submit`] and an awaited readback there.
    fn dispatch<'resources>(
        &self,
        dispatches: &[Self::Dispatch<'resources>],
    ) -> Result<(), Self::Error> {
        self.submit(dispatches)?.wait()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_state_is_explicit() {
        assert!(!JobStatus::Pending.is_complete());
        assert!(JobStatus::Complete.is_complete());
    }
}
