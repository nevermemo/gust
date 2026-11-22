//! W4: spatial binning, and its scatter row.

use std::time::Instant;

use gust::{Independent, math::Vec4};
use gust_algo::{dispatch_batch_in_scope, scan, timed};
use gust_wgpu::{BufferDispatch, DeviceHandle, GpuVec};

use crate::kernels::kit;
use crate::measure::{Sample, measure};
use crate::report::{Row, skip_reason};
use crate::setup::{
    W4_GRID_SIDE, bin_of_host, cpu_counts, cpu_exclusive, dbg, extent, particle_init,
};

pub(crate) fn w4(device: &DeviceHandle, count: u32) -> Row {
    let cells = W4_GRID_SIDE * W4_GRID_SIDE;
    let mut row = Row {
        workload: "W4 spatial binning (count + scan)",
        size: format!("{count} particles, {W4_GRID_SIDE}x{W4_GRID_SIDE}"),
        result: Err("not run".into()),
        note: String::new(),
    };
    row.result = (|| {
        if let Some(reason) = skip_reason(device, u64::from(count) * 16) {
            return Err(reason);
        }
        let parts_host: Vec<Vec4> = (0..count).map(|i| particle_init(i, W4_GRID_SIDE)).collect();
        let parts = GpuVec::from_slice_on(device, &parts_host).map_err(dbg)?;
        // Non-zero to begin with (every cell holds a value the histogram cannot produce), so a clear that does
        // nothing leaves a difference the check below sees; zeroed counts would let it pass.
        let poisoned: Vec<u32> = (0..cells).map(|i| 0x8000_0000 | i).collect();
        let counts = GpuVec::from_slice_on(device, &poisoned).map_err(dbg)?;
        let clear_params = device
            .create_uniform(
                "w4 clear",
                &kit::Params {
                    n: cells,
                    side: W4_GRID_SIDE,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let count_params = device
            .create_uniform(
                "w4 count",
                &kit::Params {
                    n: count,
                    side: W4_GRID_SIDE,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let clear = kit::w4_clear::call_extent(extent(cells), &clear_params, Independent(&counts));
        let bin = kit::w4_count::call_extent(
            extent(count),
            &count_params,
            Independent(&parts),
            Independent(&counts),
        );
        let batch = [BufferDispatch::from(&clear), BufferDispatch::from(&bin)];
        // The scan's output, made once: every step writes all of it (`exclusive_into`), so no step allocates it.
        let positions = GpuVec::<u32>::zeroed_on(device, cells).map_err(dbg)?;
        // One step: the counts the kernel made against the host's histogram, and the scan of them against the host's
        // exclusive prefix, both exactly (integers).
        // The clear and count batch goes through `dispatch_batch_in_scope` inside the same `timed` scope as the scan, so
        // the scope's GPU time covers both and its `batches` is the clear and count batch plus the scan's own.
        let step = || -> Result<Sample, String> {
            let started = Instant::now();
            let scoped = timed(|| {
                dispatch_batch_in_scope(device, &batch)?;
                scan::exclusive_into(&counts, &positions)
            })
            .map_err(dbg)?;
            let wall_ms = started.elapsed().as_secs_f64() * 1e3;
            if scoped.time.batches != 2 {
                return Err(format!(
                    "the timing scope counted {} batch(es), expected the clear and count batch and the scan",
                    scoped.time.batches
                ));
            }
            let gpu_ms = scoped.time.gpu_elapsed.map(|d| d.as_secs_f64() * 1e3);
            Ok(Sample { gpu_ms, wall_ms })
        };
        step()?;
        let cpu = cpu_counts(&parts_host, W4_GRID_SIDE);
        if counts.to_vec().map_err(dbg)? != cpu {
            return Err("the atomic counts differ from the host histogram".into());
        }
        if positions.to_vec().map_err(dbg)? != cpu_exclusive(&cpu) {
            return Err("the scan differs from the host exclusive prefix".into());
        }
        let measured = measure(step)?;
        // A later step as well: counts that were not cleared between steps have grown by now.
        if counts.to_vec().map_err(dbg)? != cpu {
            return Err(
                "the atomic counts differ from the host histogram after the timed steps".into(),
            );
        }
        // The particle read, the atomic update of one counter (a read and a write), the clear and the scan of the counts.
        let bytes = 16.0 * f64::from(count) + 8.0 * f64::from(count) + 12.0 * f64::from(cells);
        Ok((measured, Some(bytes)))
    })();
    row.note = "one timing scope covers the clear and count batch and the scan, so the time is their summed GPU time (the wall clock around the whole step on a device without timestamps, which no longer includes allocating the scan output: it is made once before the steps, so do not compare wall figures with earlier reports); bytes: 16 + 8 per particle, 12 per cell, nominal (the atomic counted as 8), so the GB/s is not a memory bandwidth".into();
    row
}

/// Up to the first 16 entries of a slice, for a failure message.
fn head(values: &[u32]) -> String {
    let shown: Vec<String> = values.iter().take(16).map(u32::to_string).collect();
    let more = if values.len() > 16 { ", .." } else { "" };
    format!("[{}{more}] ({} entries)", shown.join(", "), values.len())
}

/// The scatter against the host: for every cell the slots `starts[c] .. starts[c] + counts[c]` of `order` hold, as a
/// set, the indices of the particles the host puts in that cell (each range is sorted first: the order inside a cell is
/// unspecified), and every index appears once. The host's sets come from a counting sort of the host's own particles,
/// binned on a `bins x bins` grid over a field of side `field` (`grid` is `(bins, field)`).
pub(crate) fn check_scatter(
    order: &[u32],
    starts: &[u32],
    counts: &[u32],
    parts_host: &[Vec4],
    grid: (u32, u32),
) -> Result<(), String> {
    let cell_of_host = |q: &Vec4| bin_of_host(q, grid.0, grid.1);
    if order.len() != parts_host.len() {
        return Err(format!(
            "order holds {} slots, expected {}",
            order.len(),
            parts_host.len()
        ));
    }
    let mut fill = vec![0u32; counts.len()];
    let mut expected = vec![u32::MAX; parts_host.len()];
    for (i, q) in parts_host.iter().enumerate() {
        let c = cell_of_host(q);
        expected[(starts[c] + fill[c]) as usize] = i as u32;
        fill[c] += 1;
    }
    let mut seen = vec![false; parts_host.len()];
    let mut got = Vec::new();
    for (c, &count) in counts.iter().enumerate() {
        let from = starts[c] as usize;
        let to = from + count as usize;
        got.clear();
        got.extend_from_slice(&order[from..to]);
        got.sort_unstable();
        if got != expected[from..to] {
            return Err(format!(
                "cell {c}: GPU set {} against host set {}",
                head(&got),
                head(&expected[from..to])
            ));
        }
        for &index in &got {
            if std::mem::replace(&mut seen[index as usize], true) {
                return Err(format!("cell {c}: index {index} appears twice"));
            }
        }
    }
    if let Some(missing) = seen.iter().position(|s| !s) {
        return Err(format!("index {missing} appears in no cell"));
    }
    Ok(())
}

pub(crate) fn w4_scatter(device: &DeviceHandle, count: u32) -> Row {
    let cells = W4_GRID_SIDE * W4_GRID_SIDE;
    let mut row = Row {
        workload: "W4 spatial binning (count + scan + scatter)",
        size: format!("{count} particles, {W4_GRID_SIDE}x{W4_GRID_SIDE}"),
        result: Err("not run".into()),
        note: String::new(),
    };
    row.result = (|| {
        if let Some(reason) = skip_reason(device, u64::from(count) * 16) {
            return Err(reason);
        }
        let parts_host: Vec<Vec4> = (0..count).map(|i| particle_init(i, W4_GRID_SIDE)).collect();
        let parts = GpuVec::from_slice_on(device, &parts_host).map_err(dbg)?;
        // The counts and the cursors begin non-zero, and `order` begins at a value no index can equal, so a clear that
        // does nothing or a slot never written shows in the check below.
        let counts_init: Vec<u32> = (0..cells).map(|i| 0x8000_0000 | i).collect();
        let cursor_init: Vec<u32> = (0..cells).map(|i| 0x4000_0000 | i).collect();
        let counts = GpuVec::from_slice_on(device, &counts_init).map_err(dbg)?;
        let cursor = GpuVec::from_slice_on(device, &cursor_init).map_err(dbg)?;
        let order = GpuVec::from_slice_on(device, &vec![u32::MAX; count as usize]).map_err(dbg)?;
        let clear_params = device
            .create_uniform(
                "w4 scatter clear",
                &kit::Params {
                    n: cells,
                    side: W4_GRID_SIDE,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let count_params = device
            .create_uniform(
                "w4 scatter count",
                &kit::Params {
                    n: count,
                    side: W4_GRID_SIDE,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let clear_counts =
            kit::w4_clear::call_extent(extent(cells), &clear_params, Independent(&counts));
        let clear_cursor =
            kit::w4_clear::call_extent(extent(cells), &clear_params, Independent(&cursor));
        let bin = kit::w4_count::call_extent(
            extent(count),
            &count_params,
            Independent(&parts),
            Independent(&counts),
        );
        let first = [
            BufferDispatch::from(&clear_counts),
            BufferDispatch::from(&clear_cursor),
            BufferDispatch::from(&bin),
        ];
        // The scan's output and the scatter call that reads it are made once (`exclusive_into` writes all of `starts`
        // every step), so a step allocates and builds neither; the scan's own calls are still made inside it.
        let starts = GpuVec::<u32>::zeroed_on(device, cells).map_err(dbg)?;
        let scatter = kit::w4_scatter::call_extent(
            extent(count),
            &count_params,
            Independent(&parts),
            Independent(&starts),
            Independent(&cursor),
            Independent(&order),
        );
        let second = [BufferDispatch::from(&scatter)];
        // One step: the clears and the count (one batch), the scan, the scatter (one batch): three batches in the one
        // `timed` scope, so its GPU time covers the whole binning.
        let step = || -> Result<Sample, String> {
            let started = Instant::now();
            let scoped = timed(|| {
                dispatch_batch_in_scope(device, &first)?;
                scan::exclusive_into(&counts, &starts)?;
                dispatch_batch_in_scope(device, &second)?;
                Ok(())
            })
            .map_err(dbg)?;
            let wall_ms = started.elapsed().as_secs_f64() * 1e3;
            if scoped.time.batches != 3 {
                return Err(format!(
                    "the timing scope counted {} batch(es), expected the clear and count batch, the scan and the scatter batch",
                    scoped.time.batches
                ));
            }
            let gpu_ms = scoped.time.gpu_elapsed.map(|d| d.as_secs_f64() * 1e3);
            Ok(Sample { gpu_ms, wall_ms })
        };
        let cpu = cpu_counts(&parts_host, W4_GRID_SIDE);
        let cpu_starts = cpu_exclusive(&cpu);
        let verify = |after: &str| -> Result<(), String> {
            if counts.to_vec().map_err(dbg)? != cpu {
                return Err(format!(
                    "the atomic counts differ from the host histogram{after}"
                ));
            }
            if starts.to_vec().map_err(dbg)? != cpu_starts {
                return Err(format!(
                    "the scan differs from the host exclusive prefix{after}"
                ));
            }
            check_scatter(
                &order.to_vec().map_err(dbg)?,
                &cpu_starts,
                &cpu,
                &parts_host,
                (W4_GRID_SIDE, W4_GRID_SIDE),
            )
            .map_err(|reason| format!("scatter{after}: {reason}"))
        };
        step()?;
        verify("")?;
        let measured = measure(step)?;
        // A later step as well: counters or cursors that were not cleared between steps have grown by now.
        verify(" after the timed steps")?;
        // Per particle: the particle read (16), the count atomic (8), the scatter's particle read (16), the start read
        // (4), the cursor atomic (8) and the order write (4); per cell: the two clears (8) and the scan (12).
        let bytes = 56.0 * f64::from(count) + 20.0 * f64::from(cells);
        Ok((measured, Some(bytes)))
    })();
    row.note = "the order inside a cell is unspecified (checked as a set per cell, each range sorted first, and every index exactly once); one timing scope covers the clears and count batch, the scan and the scatter batch, so the time is their summed GPU time (the wall clock around the whole step on a device without timestamps, which no longer includes allocating the scan output or building the scatter call: both are made once before the steps, so do not compare wall figures with earlier reports); the particles never move and `order` is not re-poisoned between steps, so the check after the timed steps does not catch a write lost in a later step; bytes: nominal (each atomic counted as 8), so the GB/s is not a memory bandwidth".into();
    row
}
