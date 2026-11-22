//! W2: field diffusion.

use gust::{Independent, math::Vec4};
use gust_wgpu::{BufferDispatch, DeviceHandle, GpuVec};

use crate::kernels::kit;
use crate::measure::{dispatch, measure};
use crate::report::{Row, skip_reason};
use crate::setup::{W2_CHECKED_STEPS, cpu_diffuse, dbg, extent, field_init, lanes};

pub(crate) fn w2(device: &DeviceHandle, side: u32) -> Row {
    let n = side * side;
    let mut row = Row {
        workload: "W2 field diffusion",
        size: format!("{side}x{side}"),
        result: Err("not run".into()),
        note: String::new(),
    };
    row.result = (|| {
        if let Some(reason) = skip_reason(device, u64::from(n) * 16) {
            return Err(reason);
        }
        let host: Vec<Vec4> = (0..n).map(field_init).collect();
        let a = GpuVec::from_slice_on(device, &host).map_err(dbg)?;
        let b = GpuVec::<Vec4>::zeroed_on(device, n).map_err(dbg)?;
        let params = device
            .create_uniform(
                "w2",
                &kit::Params {
                    n,
                    side,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let a_to_b =
            kit::w2_diffuse::call_extent(extent(n), &params, Independent(&a), Independent(&b));
        let b_to_a =
            kit::w2_diffuse::call_extent(extent(n), &params, Independent(&b), Independent(&a));
        let batches = [
            [BufferDispatch::from(&a_to_b)],
            [BufferDispatch::from(&b_to_a)],
        ];
        // The checked steps alternate from `a`; with an odd count the result is in `b`.
        let mut done = 0usize;
        let mut cpu = host;
        for _ in 0..W2_CHECKED_STEPS {
            dispatch(device, &batches[done % 2])?;
            done += 1;
            cpu = cpu_diffuse(&cpu, side);
        }
        let gpu = if done % 2 == 1 { &b } else { &a }.to_vec().map_err(dbg)?;
        // No cancellation (every operand is non-negative), so the relative error of each step is at most five
        // roundings, each 2^-23 on the GPU and 2^-24 on the CPU: 3 steps x 5 x 1.5 x 2^-23 is below 3e-6. The
        // tolerance is 5e-6.
        for (at, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
            for (lane, (gv, cv)) in lanes(*g).into_iter().zip(lanes(*c)).enumerate() {
                // Finite and affirmatively within the tolerance, so a NaN cannot pass.
                if !(gv.is_finite() && (gv - cv).abs() <= 5e-6 * cv.abs()) {
                    return Err(format!(
                        "cell {at} lane {lane}: GPU {gv} against CPU {cv} after {W2_CHECKED_STEPS} steps"
                    ));
                }
            }
        }
        let measured = measure(|| {
            let sample = dispatch(device, &batches[done % 2]);
            done += 1;
            sample
        })?;
        Ok((measured, Some(32.0 * f64::from(n))))
    })();
    row.note = "bytes: one read plus one write of the 16-byte cell (neighbour reads assumed cached); checked after 3 steps".into();
    row
}
