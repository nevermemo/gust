//! Counted stages on the GPU device (slice m14.s2i, D278): a stage over the rows `0 .. min(n, capacity)` of a count `n`
//! that a stage before it wrote.
//!
//! Indirect launch (native): the stage is two prepared dispatches, the one-invocation args kernel that writes the grid
//! of `min(n, capacity)` rows into a three-word buffer, and the counted kernel, dispatched from that buffer with
//! `dispatch_workgroups_indirect` in a pass of its own. Both go into the frame's one encoder, so the frame is still one
//! queue submission, and queue order keeps each frame's args kernel in front of its counted stage when frames are in
//! flight (the args buffer is the stage's, one per stage; a resize that remakes the stage makes a new one). Bounded launch: the counted kernel alone, over
//! the worst-case grid of the capacity (what `wasm32` runs until the browser's indirect path arrives).

use std::sync::Arc;

use gust::batch::{CountedDevice, CountedLaunch, GROUP_LIMIT, grid_of};
use gust::{Call, Extent, Kernel, Run};

use crate::{BufferDispatch, Error, GpuBuffer, HeadlessDevice, PassTimestamps, PreparedDispatch};

use super::GpuStage;
use crate::cache::Pipeline;

/// The label of the pass of an indirectly launched counted stage.
const COUNTED_PASS_LABEL: &str = "GUST counted batch pass";
/// The label of the pass of the args kernel.
const ARGS_PASS_LABEL: &str = "GUST counted batch args pass";

/// What a test may see of the counted stages the device sealed and recorded on this thread (D278, check 3): the args
/// buffers and the number of indirect passes recorded. Test support, off until [`HeadlessDevice::counted_probe`] turns
/// it on; not API.
#[derive(Default)]
struct Probe {
    on: bool,
    args: Vec<std::sync::Weak<GpuBuffer<u32>>>,
    passes: u64,
}

std::thread_local! {
    static PROBE: std::cell::RefCell<Probe> = std::cell::RefCell::new(Probe::default());
}

impl HeadlessDevice {
    /// Turn the counted-stage probe of this thread on or off. Turning it on forgets what it saw.
    #[doc(hidden)]
    pub fn counted_probe(on: bool) {
        PROBE.with(|probe| {
            *probe.borrow_mut() = Probe {
                on,
                ..Probe::default()
            }
        });
    }

    /// The args buffers of the counted stages sealed with the indirect launch since the probe was turned on, oldest
    /// first, those still alive.
    #[doc(hidden)]
    #[must_use]
    pub fn counted_probe_args() -> Vec<Arc<GpuBuffer<u32>>> {
        PROBE.with(|probe| {
            probe
                .borrow()
                .args
                .iter()
                .filter_map(std::sync::Weak::upgrade)
                .collect()
        })
    }

    /// The indirect counted passes recorded since the probe was turned on.
    #[doc(hidden)]
    #[must_use]
    pub fn counted_probe_passes() -> u64 {
        PROBE.with(|probe| probe.borrow().passes)
    }
}

impl HeadlessDevice {
    /// The three-word buffer a counted stage's grid is written to and dispatched from.
    #[cfg(not(target_arch = "wasm32"))]
    fn counted_args_buffer(&self) -> GpuBuffer<u32> {
        let raw = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GUST counted stage args"),
            size: 3 * u64::from(<u32 as gust::GpuPod>::LAYOUT.stride),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        GpuBuffer {
            raw,
            len: 3,
            access: crate::GpuBufferAccess::ReadWrite,
            layout: <u32 as gust::GpuPod>::LAYOUT,
            device_id: self.id,
            indirect: true,
            lease: None,
            marker: std::marker::PhantomData,
        }
    }

    /// Record the args pass and the indirect counted pass of a stage sealed with the indirect launch.
    pub(super) fn record_counted(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        dispatches: &[PreparedDispatch],
        args: &wgpu::Buffer,
        stamps: Option<(&wgpu::QuerySet, Option<u32>, Option<u32>)>,
    ) -> Result<(), Error> {
        let [grid, counted] = dispatches else {
            unreachable!(
                "an indirectly launched counted stage holds the args dispatch and the counted one"
            )
        };
        let args_stamps = stamps.and_then(|(query_set, begin, _)| {
            begin.map(|begin| PassTimestamps {
                query_set,
                begin: Some(begin),
                end: None,
            })
        });
        crate::count_passes(2);
        grid.encode(&self.device, encoder, ARGS_PASS_LABEL, args_stamps)?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(COUNTED_PASS_LABEL),
            timestamp_writes: stamps.and_then(|(query_set, _, end)| {
                end.map(|end| wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: None,
                    end_of_pass_write_index: Some(end),
                })
            }),
        });
        let Pipeline::Compute(pipeline) = &counted.cached_kernel.pipeline else {
            unreachable!("a counted stage is a grid kernel")
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &counted.bind_group, &[]);
        pass.dispatch_workgroups_indirect(args, 0);
        PROBE.with(|probe| {
            let mut probe = probe.borrow_mut();
            if probe.on {
                probe.passes += 1;
            }
        });
        Ok(())
    }
}

impl CountedDevice for HeadlessDevice {
    fn prepare_counted<K, KA, R, const N: usize, const NA: usize>(
        &self,
        keep: R,
        _count: Arc<GpuBuffer<u32>>,
        capacity: u32,
        workgroup: u32,
        launch: CountedLaunch,
        counted: impl for<'o> Fn(&'o R, Extent) -> Call<'o, Self, K, N> + 'static,
        args: impl for<'o> Fn(&'o R, &'o GpuBuffer<u32>) -> Call<'o, Self, KA, NA> + 'static,
    ) -> Result<GpuStage, Error>
    where
        Self: Run<K, N> + Run<KA, NA>,
        K: Kernel,
        KA: Kernel,
        R: 'static,
    {
        let launch = effective_launch(launch, self.indirect_execution());
        #[cfg(not(target_arch = "wasm32"))]
        if launch == CountedLaunch::Indirect {
            // The counted kernel is prepared first: its device limits (the storage buffer count above all) are checked
            // before the args buffer or the args pipeline exist, so a refused stage leaves neither behind (m14.lim-b).
            let counted_dispatches = {
                let call = counted(&keep, Extent::new(workgroup, 1, 1));
                let dispatch =
                    BufferDispatch::new_extent(call.kernel(), call.extent(), call.bindings());
                self.prepare_indirect(&dispatch, &crate::BindGroupSlot::default())?
            };
            let buffer = Arc::new(self.counted_args_buffer());
            PROBE.with(|probe| {
                let mut probe = probe.borrow_mut();
                if probe.on {
                    probe.args.push(Arc::downgrade(&buffer));
                }
            });
            let dispatches = {
                let grid_call = args(&keep, &buffer);
                let grid = BufferDispatch::new_extent(
                    grid_call.kernel(),
                    grid_call.extent(),
                    grid_call.bindings(),
                );
                let mut prepared = self.prepare_extent_batch(&[grid])?;
                prepared.extend(counted_dispatches);
                prepared
            };
            let raw = buffer.raw.clone();
            return Ok(GpuStage {
                dispatches,
                indirect: Some(raw),
                _keep: Box::new((keep, buffer)),
            });
        }
        let [x, y] = grid_of(capacity, workgroup);
        debug_assert!(x <= GROUP_LIMIT && (y <= GROUP_LIMIT || capacity == 0));
        let dispatches = {
            let call = counted(&keep, Extent::new(x * workgroup, y, 1));
            let dispatch =
                BufferDispatch::new_extent(call.kernel(), call.extent(), call.bindings());
            self.prepare_extent_batch(&[dispatch])?
        };
        let _ = (&args, launch);
        Ok(GpuStage {
            dispatches,
            indirect: None,
            _keep: Box::new(keep),
        })
    }
}

/// The launch a device takes for a counted stage: the wanted one, except that an adapter without indirect execution
/// takes the bounded launch (wgpu would otherwise fail at encode). The flags are the input so that the choice can be
/// tested without a device.
#[must_use]
pub(crate) const fn effective_launch(
    wanted: CountedLaunch,
    indirect_execution: bool,
) -> CountedLaunch {
    match wanted {
        CountedLaunch::Indirect if !indirect_execution => CountedLaunch::Bounded,
        other => other,
    }
}

impl HeadlessDevice {
    /// Whether the adapter runs `dispatch_workgroups_indirect`: wgpu's `DownlevelFlags::INDIRECT_EXECUTION`, read when
    /// the device was built.
    fn indirect_execution(&self) -> bool {
        self.indirect_execution
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_adapter_without_indirect_execution_takes_the_bounded_launch() {
        assert_eq!(
            effective_launch(CountedLaunch::Indirect, true),
            CountedLaunch::Indirect
        );
        assert_eq!(
            effective_launch(CountedLaunch::Indirect, false),
            CountedLaunch::Bounded
        );
        for indirect_execution in [true, false] {
            assert_eq!(
                effective_launch(CountedLaunch::Bounded, indirect_execution),
                CountedLaunch::Bounded
            );
        }
    }
}
