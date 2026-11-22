//! W1: copy (reference).

use gust::Independent;
use gust_wgpu::{BufferDispatch, DeviceHandle, GpuVec};

use crate::kernels::kit;
use crate::measure::{dispatch, measure};
use crate::report::{Row, skip_reason};
use crate::setup::{dbg, extent, mix};

pub(crate) fn w1(device: &DeviceHandle, n: u32) -> Row {
    let mut row = Row {
        workload: "W1 copy",
        size: format!("{n} u32"),
        result: Err("not run".into()),
        note: String::new(),
    };
    row.result = (|| {
        if let Some(reason) = skip_reason(device, u64::from(n) * 4) {
            return Err(reason);
        }
        let host: Vec<u32> = (0..n).map(mix).collect();
        let input = GpuVec::from_slice_on(device, &host).map_err(dbg)?;
        let out = GpuVec::<u32>::zeroed_on(device, n).map_err(dbg)?;
        let params = device
            .create_uniform(
                "w1",
                &kit::Params {
                    n,
                    side: 0,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let call =
            kit::w1_copy::call_extent(extent(n), &params, Independent(&input), Independent(&out));
        let batch = [BufferDispatch::from(&call)];
        dispatch(device, &batch)?;
        if out.to_vec().map_err(dbg)? != host {
            return Err("the copy differs from the input".into());
        }
        let measured = measure(|| dispatch(device, &batch))?;
        Ok((measured, Some(8.0 * f64::from(n))))
    })();
    row.note = "bytes: one read plus one write per element".into();
    row
}
