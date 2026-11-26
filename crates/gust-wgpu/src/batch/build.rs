//! Sealing a stage: the dispatch of a kernel call, validated and prepared once, with what it binds kept alive.

use core::any::Any;

use gust::{Call, Kernel, Run};

use crate::{BufferDispatch, Error, HeadlessDevice, PreparedDispatch};

/// A stage sealed with its batch: the prepared pass (the pipeline, the bind group, the workgroup counts), and the
/// buffers and extent record it binds, which the bind group refers to.
#[doc(hidden)]
pub struct GpuStage {
    pub(super) dispatches: Vec<PreparedDispatch>,
    /// A counted stage launched indirectly: the buffer its grid is read from (the second dispatch is the counted one).
    pub(super) indirect: Option<wgpu::Buffer>,
    pub(super) _keep: Box<dyn Any>,
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
            indirect: None,
            _keep: Box::new(keep),
        })
    }
}

impl HeadlessDevice {
    /// Validate and prepare a list of dispatches as one stage, as `prepare_stage` does for one call:
    /// the library ops of a batch (gust-algo's reductions, scans and compaction) are several dispatches in order.
    ///
    /// `keep` is moved into the stage and dropped with it, so it holds what the dispatches name for as long as the
    /// stage lives: the pooled scratch buffers of a plan (which go back to the pool when it drops) and the operands the
    /// bind groups refer to. The dispatches borrow what they name, so a caller that builds them over `keep` keeps a
    /// second handle to it (an `Arc` clone) for the borrow and hands the first one here.
    ///
    /// Macro support, outside the pre-1.0 promise (D111).
    ///
    /// # Errors
    ///
    /// The refusal of the preparation: a buffer of another device, a binding the kernel does not take, a limit.
    #[doc(hidden)]
    pub fn prepare_list(
        &self,
        keep: impl Any,
        dispatches: &[BufferDispatch<'_>],
    ) -> Result<GpuStage, Error> {
        Ok(GpuStage {
            dispatches: self.prepare_extent_batch(dispatches)?,
            indirect: None,
            _keep: Box::new(keep),
        })
    }
}
