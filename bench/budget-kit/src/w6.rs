//! W6: readback delay.

use std::time::Instant;

use gust::Independent;
use gust_wgpu::{BufferDispatch, DeviceHandle, GpuVec};

use crate::kernels::kit;
use crate::measure::{Sample, measure, median};
use crate::report::{Row, skip_reason};
use crate::setup::{STEPS, dbg, extent};

pub(crate) fn w6(device: &DeviceHandle, n: u32) -> Row {
    let mut row = Row {
        workload: "W6 readback delay",
        size: format!("{n} u32"),
        result: Err("not run".into()),
        note: String::new(),
    };
    row.result = (|| {
        if let Some(reason) = skip_reason(device, u64::from(n) * 4) {
            return Err(reason);
        }
        let host: Vec<u32> = (0..n).map(|i| i.wrapping_mul(7).wrapping_add(3)).collect();
        let expected: Vec<u32> = host.iter().map(|v| v.wrapping_add(1)).collect();
        let input = GpuVec::from_slice_on(device, &host).map_err(dbg)?;
        let out = GpuVec::<u32>::zeroed_on(device, n).map_err(dbg)?;
        let params = device
            .create_uniform(
                "w6",
                &kit::Params {
                    n,
                    side: 0,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let call = kit::w6_plus_one::call_extent(
            extent(n),
            &params,
            Independent(&input),
            Independent(&out),
        );
        let batch = [BufferDispatch::from(&call)];
        let mut submit_ms = Vec::new();
        let mut read_ms = Vec::new();
        // Submit, wait for the submission, read the result back: the time until it is on the CPU. The result is compared
        // after the clock stops on every step, but only the first step proves the kernel wrote it (`out` starts zeroed;
        // later steps find the previous step's identical result there)."
        let mut step = || -> Result<Sample, String> {
            let started = Instant::now();
            device.dispatch_batch(&batch).map_err(dbg)?;
            let submitted = started.elapsed();
            let result = out.to_vec().map_err(dbg)?;
            let total = started.elapsed();
            if result != expected {
                return Err("the result differs from the host's".into());
            }
            submit_ms.push(submitted.as_secs_f64() * 1e3);
            read_ms.push((total - submitted).as_secs_f64() * 1e3);
            Ok(Sample {
                gpu_ms: None,
                wall_ms: total.as_secs_f64() * 1e3,
            })
        };
        let measured = measure(&mut step)?;
        // The split of the counted steps (the last STEPS entries).
        let from = submit_ms.len() - STEPS;
        row.note = format!(
            "submit and wait {:.4} ms, readback {:.4} ms (medians of the same steps); wall clock around submit, wait and readback; the result is proven written on the first step only",
            median(&submit_ms[from..]),
            median(&read_ms[from..])
        );
        Ok((measured, None))
    })();
    row
}
