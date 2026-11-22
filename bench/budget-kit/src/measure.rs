//! Timing: one sample, the median of the counted steps, one batch dispatch.

use gust_wgpu::{BufferDispatch, HeadlessDevice};

use crate::setup::{STEPS, WARMUP, dbg};

// ---- measuring ----

/// One step's time: the GPU's own, where the device has timestamps, and the wall clock around the step.
pub(crate) struct Sample {
    pub(crate) gpu_ms: Option<f64>,
    pub(crate) wall_ms: f64,
}

pub(crate) fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

/// What a measured row reports.
pub(crate) struct Measured {
    pub(crate) ms: f64,
    pub(crate) method: &'static str,
}

/// The median of `STEPS` steps after `WARMUP` that are not counted. The GPU time is used only if every counted step
/// has one; otherwise the wall clock is, for all of them.
pub(crate) fn measure(
    mut step: impl FnMut() -> Result<Sample, String>,
) -> Result<Measured, String> {
    for _ in 0..WARMUP {
        step()?;
    }
    let mut gpu = Vec::new();
    let mut wall = Vec::new();
    let mut every_gpu = true;
    for _ in 0..STEPS {
        let sample = step()?;
        wall.push(sample.wall_ms);
        match sample.gpu_ms {
            Some(ms) => gpu.push(ms),
            None => every_gpu = false,
        }
    }
    Ok(if every_gpu {
        Measured {
            ms: median(&gpu),
            method: "gpu timestamps",
        }
    } else {
        Measured {
            ms: median(&wall),
            method: "wall clock",
        }
    })
}

/// Dispatches a batch once, timed.
pub(crate) fn dispatch(
    device: &HeadlessDevice,
    batch: &[BufferDispatch<'_>],
) -> Result<Sample, String> {
    let timing = device.dispatch_batch_timed(batch).map_err(dbg)?;
    Ok(Sample {
        gpu_ms: timing.gpu_elapsed.map(|d| d.as_secs_f64() * 1e3),
        wall_ms: timing.host_elapsed.as_secs_f64() * 1e3,
    })
}
