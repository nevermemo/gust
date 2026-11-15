//! The backend trait every comparator implements (research section 5: prepare, run, sync, read).

use std::{error::Error, time::Duration};

use serde_json::Value;

use crate::workloads::{Inputs, Workload};

pub type BenchResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Gpu,
    Cpu,
}

/// One comparator. Every method that completes work waits for it: a method returns only when its result is final.
///
/// The output is `Vec<f32>` words for every workload; a `u32` histogram's words are the counts' bit patterns (see
/// `workloads`). `out` of `end_to_end` and `read_back` is replaced or resized by the backend to the workload's output
/// length (`Workload::out_len`).
pub trait Backend {
    /// Stable short id used in the JSON and the tables.
    fn id(&self) -> &'static str;
    /// One line for the README and the table legend.
    fn description(&self) -> &'static str;
    fn kind(&self) -> Kind;

    /// Whether the backend has the workload at all (`gust-cpu-par` has a map form of W1 and W2 only). The run leaves
    /// a backend out of the cells of a workload it does not support, instead of recording a failure.
    fn supports(&self, _workload: Workload) -> bool {
        true
    }

    /// Allocates the buffers for `inputs`, uploads them and builds the pipeline. Not timed. After it returns,
    /// everything is resident and idle.
    fn prepare(&mut self, workload: Workload, inputs: &Inputs) -> BenchResult<()>;

    /// Resets the device-side output so a dispatch that does not run cannot pass a check: NaN for the f32 workloads,
    /// zero for a histogram (which accumulates). Waits for it. Not timed.
    fn poison(&mut self) -> BenchResult<()>;

    /// End to end: host inputs in, host output in `out`, all work complete. For a GPU this is upload, dispatch and
    /// read-back (a histogram also clears its bins, a host call needs a clean result); for a CPU backend it is the
    /// computation on host memory.
    fn end_to_end(&mut self, inputs: &Inputs, out: &mut Vec<f32>) -> BenchResult<()>;

    /// One dispatch on resident data, complete on return, host-timed. For the CPU backends it is the computation on
    /// their own resident copies. A histogram dispatch adds into the bins: `read_back` after `p` dispatches since the
    /// last `poison` returns the counts of one pass, which it can only do when the bins were identical passes.
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

    /// The route record of the last `cold_start` (see `route`): which path produced the shader. `None` where the
    /// backend has no such path.
    fn cold_route(&self) -> Option<Value> {
        None
    }

    /// The route check after a cell: no route event of the cell's own may be off the route of the build, and the
    /// kernel's latest event over the device's life must exist and be on the route (for a warm cell, which compiles
    /// nothing, that event is the build of the cached pipeline it ran). An `Err` names both routes and fails the cell.
    /// Backends without route events pass.
    fn route_check(&self) -> Result<(), String> {
        Ok(())
    }

    /// What `route_check` saw, for the run file (m10.b1c, R1.3-6): the build's route, how many route events the device
    /// holds for this kernel and on which routes. `None` where the backend has no route events (every build without
    /// `spirv-route`, and the hand-WGSL backends).
    fn route_evidence(&self) -> Option<Value> {
        None
    }

    /// Pipeline-cache misses so far, where the backend counts them. The harness asserts it does not move while
    /// samples are taken.
    fn cache_misses(&self) -> Option<u64> {
        None
    }
}
