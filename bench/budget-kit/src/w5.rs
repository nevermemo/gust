//! W5: reduction.

use std::time::Instant;

use gust_algo::{bound, reduce, timed};
use gust_wgpu::{DeviceHandle, GpuVec};

use crate::measure::{Sample, measure};
use crate::report::{Row, skip_reason};
use crate::setup::{dbg, mix, unit};

pub(crate) fn w5(device: &DeviceHandle, n: u32) -> Row {
    let mut row = Row {
        workload: "W5 reduction (f32 sum)",
        size: format!("{n} f32"),
        result: Err("not run".into()),
        note: String::new(),
    };
    row.result = (|| {
        if let Some(reason) = skip_reason(device, u64::from(n) * 4) {
            return Err(reason);
        }
        let host: Vec<f32> = (0..n).map(|i| unit(mix(i))).collect();
        let input = GpuVec::from_slice_on(device, &host).map_err(dbg)?;
        let abs_sum = bound::abs_sum(&host);
        let reference = bound::sum_f64(&host);
        let allowed =
            bound::f32_sum_bound(n, bound::C_GPU, abs_sum) + bound::f64_reference_error(n, abs_sum);
        // The exact companion: a `u32` sum of `n` values, compared bit for bit with the host's wrapping sum, so a lost block
        // or a double-counted one cannot hide inside the float bound.
        let ints: Vec<u32> = (0..n).map(|i| mix(i) >> 8).collect();
        let int_input = GpuVec::from_slice_on(device, &ints).map_err(dbg)?;
        let int_sum = reduce::sum_wrapping(&int_input).map_err(dbg)?;
        let int_host = ints.iter().fold(0u32, |a, v| a.wrapping_add(*v));
        if int_sum != int_host {
            return Err(format!(
                "u32 sum {int_sum} differs from the host's {int_host}"
            ));
        }
        let first = timed(|| reduce::sum(&input)).map_err(dbg)?;
        let error = (f64::from(first.value) - reference).abs();
        row.note = if first.value.is_finite() {
            format!(
                "observed error {:.3e} of an allowed {:.3e} (bound::f32_sum_bound plus bound::f64_reference_error); bytes: one read per element; the f32 bound is wide (about {:.0}x the observed error), so a structural check is the exact u32 sum, which matched",
                error,
                allowed,
                allowed / error.max(f64::MIN_POSITIVE)
            )
        } else {
            format!(
                "the f32 sum was not finite ({}), so there is no observed error to compare with the allowed {:.3e}; the exact u32 sum matched",
                first.value, allowed
            )
        };
        // Finite and affirmatively within the bound, so a NaN cannot pass.
        if !(first.value.is_finite() && error <= allowed) {
            return Err(format!(
                "sum {} differs from the f64 reference {reference} by {error:.3e}, over {allowed:.3e}",
                first.value
            ));
        }
        let measured = measure(|| {
            let started = Instant::now();
            let sum = timed(|| reduce::sum(&input)).map_err(dbg)?;
            Ok(Sample {
                gpu_ms: sum.time.gpu_elapsed.map(|d| d.as_secs_f64() * 1e3),
                wall_ms: started.elapsed().as_secs_f64() * 1e3,
            })
        })?;
        Ok((measured, Some(4.0 * f64::from(n))))
    })();
    row
}
