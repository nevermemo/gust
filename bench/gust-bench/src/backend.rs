//! The backend trait every comparator implements (research section 5: prepare, run, sync, read).

use std::{error::Error, time::Duration};

use crate::workloads::{Inputs, Workload};

pub type BenchResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Gpu,
    Cpu,
}

/// One comparator. Every method that completes work waits for it: a method returns only when its result is final.
pub trait Backend {
    /// Stable short id used in the JSON and the tables.
    fn id(&self) -> &'static str;
    /// One line for the README and the table legend.
    fn description(&self) -> &'static str;
    fn kind(&self) -> Kind;

    /// Allocates the buffers for `inputs`, uploads them and builds the pipeline. Not timed. After it returns,
    /// everything is resident and idle.
    fn prepare(&mut self, workload: Workload, inputs: &Inputs) -> BenchResult<()>;

    /// Fills the device-side output with NaN and waits for it, so a dispatch that does not run cannot pass a check.
    /// Not timed.
    fn poison(&mut self) -> BenchResult<()>;

    /// End to end: host inputs in, host output in `out`, all work complete. For a GPU this is upload, dispatch and
    /// read-back; for a CPU backend it is the computation on host memory.
    fn end_to_end(&mut self, inputs: &Inputs, out: &mut Vec<f32>) -> BenchResult<()>;

    /// One dispatch on resident data, complete on return, host-timed. For the CPU backends it is the computation on
    /// their own resident copies.
    fn dispatch(&mut self) -> BenchResult<()>;

    /// GPU time of `batch` back-to-back dispatches in one submission, from timestamps bracketing the first to the last
    /// pass. `None` when the backend has no GPU clock. Divide by `batch` for one dispatch.
    fn gpu_time(&mut self, _batch: usize) -> BenchResult<Option<Duration>> {
        Ok(None)
    }

    /// Copies the device-side output to the host. Not timed by the harness.
    fn read_back(&mut self, out: &mut Vec<f32>) -> BenchResult<()>;

    /// Drops the compiled pipeline state, then times the first dispatch (compile included). `None` for backends
    /// with no compile step.
    fn cold_start(&mut self) -> BenchResult<Option<Duration>> {
        Ok(None)
    }

    /// Pipeline-cache misses so far, where the backend counts them. The harness asserts it does not move while
    /// samples are taken.
    fn cache_misses(&self) -> Option<u64> {
        None
    }
}
