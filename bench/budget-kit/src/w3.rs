//! W3: particle step.

use gust::{Independent, math::Vec4};
use gust_wgpu::{BufferDispatch, DeviceHandle, GpuVec};

use crate::kernels::kit;
use crate::measure::{dispatch, measure};
use crate::report::{Row, skip_reason};
use crate::setup::{W3_FIELD_SIDE, cpu_particle, dbg, extent, field_init, lanes, particle_init};

pub(crate) fn w3(device: &DeviceHandle, count: u32) -> Row {
    let mut row = Row {
        workload: "W3 particle step",
        size: format!("{count} particles"),
        result: Err("not run".into()),
        note: String::new(),
    };
    row.result = (|| {
        let field_n = W3_FIELD_SIDE * W3_FIELD_SIDE;
        if let Some(reason) = skip_reason(device, u64::from(count) * 16) {
            return Err(reason);
        }
        let field_host: Vec<Vec4> = (0..field_n).map(field_init).collect();
        let parts_host: Vec<Vec4> = (0..count)
            .map(|i| particle_init(i, W3_FIELD_SIDE))
            .collect();
        let field = GpuVec::from_slice_on(device, &field_host).map_err(dbg)?;
        let parts = GpuVec::from_slice_on(device, &parts_host).map_err(dbg)?;
        let params = device
            .create_uniform(
                "w3",
                &kit::Params {
                    n: count,
                    side: W3_FIELD_SIDE,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let call = kit::w3_step::call_extent(
            extent(count),
            &params,
            Independent(&field),
            Independent(&parts),
        );
        let batch = [BufferDispatch::from(&call)];
        // One step from the initial state, so every particle samples the cell the host also computes. The tolerance is
        // 8 x 2^-23 of max(|expected|, 1): a step is at most four roundings of 2u on the GPU, with a magnitude of at
        // most the field side. The GPU positions must also lie in [0, side] (`side` itself is possible: -tiny + side
        // rounds to side), so a missing wrap shows even though the modulo comparison below hides a difference of a whole
        // side. A wrap decided differently by the two sides differs by the side, so a difference is
        // taken modulo the side.
        dispatch(device, &batch)?;
        let gpu = parts.to_vec().map_err(dbg)?;
        let side = W3_FIELD_SIDE as f32;
        for (at, (g, q)) in gpu.iter().zip(&parts_host).enumerate() {
            let c = cpu_particle(*q, &field_host, W3_FIELD_SIDE);
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
                // Finite and affirmatively within the tolerance, so a NaN cannot pass.
                if !(gv.is_finite() && diff <= 8.0 * (1.0 / 8_388_608.0) * cv.abs().max(1.0)) {
                    return Err(format!(
                        "particle {at} lane {lane}: GPU {gv} against CPU {cv}"
                    ));
                }
            }
        }
        let measured = measure(|| dispatch(device, &batch))?;
        Ok((measured, Some(48.0 * f64::from(count))))
    })();
    row.note = format!(
        "bytes: particle read and write (16 each) plus one 16-byte field sample; field {W3_FIELD_SIDE}x{W3_FIELD_SIDE} held still; checked after 1 step"
    );
    row
}
