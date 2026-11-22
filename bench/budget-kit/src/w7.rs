//! W7: a sustained composite step, and the checks and statistics W8 shares.

use std::time::Instant;

use gust::{Independent, math::Vec4};
use gust_wgpu::{BufferDispatch, DeviceHandle, GpuVec};

use crate::kernels::kit;
use crate::measure::{Measured, dispatch, median};
use crate::report::{Row, skip_reason};
use crate::setup::{
    W3_FIELD_SIDE, WARMUP, cpu_counts, cpu_diffuse, cpu_particle, dbg, extent, field_init, lanes,
    particle_init,
};

/// W7 particle counts, and the counted steps of its sustained run (after the warm-up, for each of its two runs).
pub(crate) const W7_SIZES: [u32; 2] = [1 << 18, 1 << 20];
pub(crate) const W7_STEPS: usize = 600;
pub(crate) const W7_STEPS_SMALL: usize = 120;

/// The value at or below which at least `p` of the samples lie (nearest rank), for `p` in (0, 1]. `values` must not
/// be empty.
pub(crate) fn percentile(values: &[f64], p: f64) -> f64 {
    debug_assert!(!values.is_empty(), "percentile of no samples");
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (p * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// W2's comparison of a field with the host's, with W2's tolerance.
pub(crate) fn check_field(gpu: &[Vec4], cpu: &[Vec4]) -> Result<(), String> {
    for (at, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        for (lane, (gv, cv)) in lanes(*g).into_iter().zip(lanes(*c)).enumerate() {
            if !(gv.is_finite() && (gv - cv).abs() <= 5e-6 * cv.abs()) {
                return Err(format!(
                    "field cell {at} lane {lane}: GPU {gv} against CPU {cv} after 1 step"
                ));
            }
        }
    }
    Ok(())
}

/// W3's comparison of particles with the host's one step from `start` over `field`, with W3's tolerance.
pub(crate) fn check_particles(
    gpu: &[Vec4],
    start: &[Vec4],
    field: &[Vec4],
    side_u: u32,
) -> Result<(), String> {
    let side = side_u as f32;
    for (at, (g, q)) in gpu.iter().zip(start).enumerate() {
        let c = cpu_particle(*q, field, side_u);
        for (lane, (gv, cv)) in lanes(*g).into_iter().zip(lanes(c)).enumerate() {
            if lane < 2 && !(0.0..=side).contains(&gv) {
                return Err(format!(
                    "particle {at} lane {lane}: GPU {gv} is outside [0, {side}] (the wrap)"
                ));
            }
            let mut diff = (gv - cv).abs();
            if lane < 2 && diff > side / 2.0 {
                diff = side - diff;
            }
            if !(gv.is_finite() && diff <= 8.0 * (1.0 / 8_388_608.0) * cv.abs().max(1.0)) {
                return Err(format!(
                    "particle {at} lane {lane}: GPU {gv} against CPU {cv}"
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn w7(device: &DeviceHandle, count: u32, steps: usize) -> Row {
    let side = W3_FIELD_SIDE;
    let cells = side * side;
    let mut row = Row {
        workload: "W7 composite step",
        size: format!("{count} particles, {side}x{side} field"),
        result: Err("not run".into()),
        note: String::new(),
    };
    // What the row says whether or not it succeeds; a success puts the statistics before it.
    let fixed = format!(
        "{steps} counted steps after {WARMUP} warm-up steps, in two runs of that length (blocking, then submit); one batch per step: w2_diffuse on {side}x{side}, w3_step, w4_clear and w4_count into {side}x{side} cells; checked after step 1 and again after step 2, the same three ways (field 5e-6 relative as W2, particles as W3 over the GPU's own field, counts exact against the GPU's own positions), and once more after the runs (particles in range, counts exact)"
    );
    row.note = fixed.clone();
    row.result = (|| {
        if let Some(reason) = skip_reason(device, u64::from(count) * 16) {
            return Err(reason);
        }
        let field_host: Vec<Vec4> = (0..cells).map(field_init).collect();
        let parts_host: Vec<Vec4> = (0..count).map(|i| particle_init(i, side)).collect();
        let a = GpuVec::from_slice_on(device, &field_host).map_err(dbg)?;
        let b = GpuVec::<Vec4>::zeroed_on(device, cells).map_err(dbg)?;
        let parts = GpuVec::from_slice_on(device, &parts_host).map_err(dbg)?;
        // Non-zero to begin with, so a clear that does nothing shows in the count check.
        let poisoned: Vec<u32> = (0..cells).map(|i| 0x8000_0000 | i).collect();
        let counts = GpuVec::from_slice_on(device, &poisoned).map_err(dbg)?;
        let uniform = |label: &str, n: u32| {
            device
                .create_uniform(
                    label,
                    &kit::Params {
                        n,
                        side,
                        pad0: 0,
                        pad1: 0,
                    },
                )
                .map_err(dbg)
        };
        let field_params = uniform("w7 field", cells)?;
        let part_params = uniform("w7 particles", count)?;
        let diffuse_ab = kit::w2_diffuse::call_extent(
            extent(cells),
            &field_params,
            Independent(&a),
            Independent(&b),
        );
        let diffuse_ba = kit::w2_diffuse::call_extent(
            extent(cells),
            &field_params,
            Independent(&b),
            Independent(&a),
        );
        let step_b = kit::w3_step::call_extent(
            extent(count),
            &part_params,
            Independent(&b),
            Independent(&parts),
        );
        let step_a = kit::w3_step::call_extent(
            extent(count),
            &part_params,
            Independent(&a),
            Independent(&parts),
        );
        let clear = kit::w4_clear::call_extent(extent(cells), &field_params, Independent(&counts));
        let bin = kit::w4_count::call_extent(
            extent(count),
            &part_params,
            Independent(&parts),
            Independent(&counts),
        );
        // The field alternates between `a` and `b` as in W2; the particles sample the field the same batch diffused.
        let batches = [
            [
                BufferDispatch::from(&diffuse_ab),
                BufferDispatch::from(&step_b),
                BufferDispatch::from(&clear),
                BufferDispatch::from(&bin),
            ],
            [
                BufferDispatch::from(&diffuse_ba),
                BufferDispatch::from(&step_a),
                BufferDispatch::from(&clear),
                BufferDispatch::from(&bin),
            ],
        ];
        // The first step, checked: the field (in `b`) against the host's diffusion, the particles against the host's
        // step over the GPU's own diffused field (the one they sampled), and the counts exactly against the host's histogram of the GPU's own
        // particle positions (a particle at a cell edge could bin differently from a host position that differs in the
        // last bit).
        dispatch(device, &batches[0])?;
        let cpu_field = cpu_diffuse(&field_host, side);
        let gpu_field = b.to_vec().map_err(dbg)?;
        check_field(&gpu_field, &cpu_field)?;
        let gpu_parts = parts.to_vec().map_err(dbg)?;
        // The particles sampled the GPU's own field, so that is the field of the reference step.
        check_particles(&gpu_parts, &parts_host, &gpu_field, side)?;
        if counts.to_vec().map_err(dbg)? != cpu_counts(&gpu_parts, side) {
            return Err("the counts differ from the host histogram after 1 step".into());
        }
        // The second step, checked the same way (the field is back in `a`): the host diffuses the GPU's own
        // first-step field, the particles are checked against the host's step from the GPU's first-step particles
        // over the GPU's own second-step field, and the counts against the host histogram of the GPU's own particles.
        dispatch(device, &batches[1])?;
        let cpu_field_2 = cpu_diffuse(&gpu_field, side);
        let gpu_field_2 = a.to_vec().map_err(dbg)?;
        check_field(&gpu_field_2, &cpu_field_2).map_err(|e| format!("second step: {e}"))?;
        let gpu_parts_2 = parts.to_vec().map_err(dbg)?;
        check_particles(&gpu_parts_2, &gpu_parts, &gpu_field_2, side)
            .map_err(|e| format!("second step: {e}"))?;
        if counts.to_vec().map_err(dbg)? != cpu_counts(&gpu_parts_2, side) {
            return Err("the counts differ from the host histogram after 2 steps".into());
        }
        // Two steps ran, so the alternation continues with `batches[0]` (field in `a`).
        let mut next = 2usize;
        // Run 1: blocking `dispatch_batch_timed`: the GPU time and `host_elapsed` of every counted step.
        for _ in 0..WARMUP {
            dispatch(device, &batches[next % 2])?;
            next += 1;
        }
        let mut blocking = Vec::with_capacity(steps);
        for _ in 0..steps {
            blocking.push(dispatch(device, &batches[next % 2])?);
            next += 1;
        }
        // Run 2: the wall time for `submit_batch` to return (encode and submit, no wait); the wait for the job is
        // outside the measured span.
        for _ in 0..WARMUP {
            device.dispatch_batch(&batches[next % 2]).map_err(dbg)?;
            next += 1;
        }
        let mut submit_ms = Vec::with_capacity(steps);
        for _ in 0..steps {
            let started = Instant::now();
            let job = device.submit_batch(&batches[next % 2]).map_err(dbg)?;
            submit_ms.push(started.elapsed().as_secs_f64() * 1e3);
            job.wait().map_err(dbg)?;
            next += 1;
        }
        // After the sustained runs: the particles are still inside the field, and the counts of the last step are the
        // host's histogram of the particles as they now are.
        let gpu_parts = parts.to_vec().map_err(dbg)?;
        let side_f = side as f32;
        if let Some(at) = gpu_parts.iter().position(|q| {
            !(q.x.is_finite()
                && q.y.is_finite()
                && (0.0..=side_f).contains(&q.x)
                && (0.0..=side_f).contains(&q.y))
        }) {
            return Err(format!(
                "particle {at} is outside [0, {side_f}] after the sustained runs"
            ));
        }
        if counts.to_vec().map_err(dbg)? != cpu_counts(&gpu_parts, side) {
            return Err(
                "the counts differ from the host histogram after the sustained runs".into(),
            );
        }
        let host_a: Vec<f64> = blocking.iter().map(|s| s.wall_ms).collect();
        let (gpu_series, method) = if blocking.iter().all(|s| s.gpu_ms.is_some()) {
            (
                blocking
                    .iter()
                    .filter_map(|s| s.gpu_ms)
                    .collect::<Vec<f64>>(),
                "gpu timestamps",
            )
        } else {
            (host_a.clone(), "wall clock")
        };
        let measured = Measured {
            ms: median(&gpu_series),
            method,
        };
        row.note = format!(
            "the time column is the median {method} time, p95 {:.4} ms, max {:.4} ms; host (a) host_elapsed of the blocking dispatch_batch_timed (it includes the GPU work and the timestamp readback): median {:.4} ms, p95 {:.4} ms; host (b) wall time for submit_batch to return (encode and submit, no wait; the job is waited for after the span; measured with an idle queue, so it shows no back-pressure): median {:.4} ms, p95 {:.4} ms; {fixed}",
            percentile(&gpu_series, 0.95),
            gpu_series.iter().copied().fold(f64::MIN, f64::max),
            median(&host_a),
            percentile(&host_a, 0.95),
            median(&submit_ms),
            percentile(&submit_ms, 0.95)
        );
        Ok((measured, None))
    })();
    row
}
