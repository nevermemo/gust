//! Sealing a stage: the dispatch of a kernel call, validated and prepared once, with what it binds kept alive.

use core::any::Any;

use gust::{Call, Kernel, Run};

use crate::{BufferDispatch, Error, HeadlessDevice, PreparedDispatch};

/// A stage sealed with its batch: the prepared pass (the pipeline, the bind group, the workgroup counts), and the
/// buffers and extent record it binds, which the bind group refers to.
#[doc(hidden)]
pub struct GpuStage {
    pub(super) dispatches: Vec<PreparedDispatch>,
    _keep: Box<dyn Any>,
}

impl HeadlessDevice {
    /// Validate and prepare the call that `build` makes over `keep`, as `submit_batch` would at each run.
    pub(super) fn prepare_stage<K, R, const N: usize>(
        &self,
        keep: R,
        build: impl for<'o> Fn(&'o R) -> Call<'o, Self, K, N> + 'static,
    ) -> Result<GpuStage, Error>
    where
        Self: Run<K, N>,
        K: Kernel,
        R: 'static,
    {
        let dispatches = {
            let call = build(&keep);
            let dispatch =
                BufferDispatch::new_extent(call.kernel(), call.extent(), call.bindings());
            self.prepare_extent_batch(&[dispatch])?
        };
        Ok(GpuStage {
            dispatches,
            _keep: Box::new(keep),
        })
    }
}
