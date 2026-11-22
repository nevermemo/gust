//! The G4 probe: a launch one workgroup above the portable limit, and the remedy.

use gust::{Extent, Independent};
use gust_wgpu::{BufferDispatch, DeviceHandle, GpuVec};

use crate::kernels::kit;
use crate::setup::dbg;

// ---- the G4 probe ----

/// The G4 probe launches 65 536 workgroups of 256 on x, one above the portable limit of 65 535.
const G4_WORKGROUPS: u32 = 65_536;
const G4_THREADS: u32 = G4_WORKGROUPS * 256;
/// The remedy's row width, in invocations.
const G4_WIDTH: u32 = 4096;

fn g4_expected(i: u32) -> u32 {
    i.wrapping_mul(3).wrapping_add(1)
}

/// The limits policy of the device `main` opens: `HeadlessDevice::new` requests wgpu's default limits
/// (`LimitsPolicy::Default`). Keep this text in step with the call in `main`.
const KIT_LIMITS_POLICY: &str = "Default (HeadlessDevice::new)";

/// The remedy's element count: not a multiple of the row width or the workgroup size, so the last row is partly past
/// the end and the kernel's `i >= n` guard has work to do.
const G4_REMEDY_N: u32 = G4_THREADS - 1000;
/// What the remedy's buffer holds before the launch, and in the slots past `n` after it.
const G4_SENTINEL: u32 = 0xDEAD_BEEF;

/// The probe of gap G4 (a linear launch above the workgroup limit on x) and its remedy. It is not a workload: it
/// reports what happens on this device. Since slice m11.g4b gust-wgpu splits such a launch into pieces and the probe
/// succeeds; a refusal, where there is one, is its own host-side check of the launch against `limits()`, made before
/// anything is submitted. Returns the lines to print and whether a check failed.
pub(crate) fn g4(device: &DeviceHandle) -> (Vec<String>, bool) {
    let max = device.limits().max_compute_workgroups_per_dimension;
    let context = format!(
        "device max_compute_workgroups_per_dimension {max}, requested limits policy {KIT_LIMITS_POLICY}"
    );
    let n = G4_THREADS;
    let mut failed = false;
    // `Ok(None)` is a launch whose result matched the host exactly; `Ok(Some(reason))` one that ran and differed.
    let run = || -> Result<Option<String>, gust_wgpu::Error> {
        let out = GpuVec::<u32>::zeroed_on(device, n)?;
        let params = device.create_uniform(
            "g4",
            &kit::Params {
                n,
                side: 0,
                pad0: 0,
                pad1: 0,
            },
        )?;
        let call = kit::g4_linear::call_extent(Extent::new(n, 1, 1), &params, Independent(&out));
        device.dispatch_batch(&[BufferDispatch::from(&call)])?;
        let got = out.to_vec()?;
        Ok(got
            .iter()
            .enumerate()
            .find(|(i, v)| **v != g4_expected(*i as u32))
            .map(|(i, v)| format!("element {i} is {v}")))
    };
    let line = match run() {
        Ok(None) => format!(
            "G4: probe, a linear launch of {G4_WORKGROUPS} workgroups on x ({n} invocations) succeeded and matched the host exactly; {context}"
        ),
        Ok(Some(reason)) => {
            failed = true;
            format!(
                "G4: probe, a linear launch of {G4_WORKGROUPS} workgroups on x ({n} invocations) ran but FAILED the host check ({reason}); {context}"
            )
        }
        // The intended rejection and nothing else: the workgroup limit of x, the requested count and the maximum the
        // device reports.
        Err(gust_wgpu::Error::DeviceLimit {
            limit,
            requested,
            maximum,
            ..
        }) if limit == "max_compute_workgroups_per_dimension"
            && requested == u64::from(G4_WORKGROUPS)
            && maximum == u64::from(max) =>
        {
            format!(
                "G4: probe, a linear launch of {G4_WORKGROUPS} workgroups on x ({n} invocations) was refused with DeviceLimit: limit {limit}, requested {requested}, maximum {maximum}; {context}"
            )
        }
        Err(other) => {
            failed = true;
            format!(
                "G4: probe, a linear launch of {G4_WORKGROUPS} workgroups on x ({n} invocations) FAILED with an error other than the intended DeviceLimit: {other:?}; {context}"
            )
        }
    };
    let rows = G4_REMEDY_N.div_ceil(G4_WIDTH);
    let slots = rows * G4_WIDTH;
    let remedy = || -> Result<(), String> {
        let out = GpuVec::from_slice_on(device, &vec![G4_SENTINEL; slots as usize]).map_err(dbg)?;
        let params = device
            .create_uniform(
                "g4 grid",
                &kit::Params {
                    n: G4_REMEDY_N,
                    side: G4_WIDTH,
                    pad0: 0,
                    pad1: 0,
                },
            )
            .map_err(dbg)?;
        let call =
            kit::g4_grid::call_extent(Extent::new(G4_WIDTH, rows, 1), &params, Independent(&out));
        device
            .dispatch_batch(&[BufferDispatch::from(&call)])
            .map_err(dbg)?;
        let got = out.to_vec().map_err(dbg)?;
        let (values, tail) = got.split_at(G4_REMEDY_N as usize);
        if let Some(at) = values
            .iter()
            .enumerate()
            .position(|(i, v)| *v != g4_expected(i as u32))
        {
            return Err(format!("element {at} differs from the host"));
        }
        if let Some(at) = tail.iter().position(|v| *v != G4_SENTINEL) {
            return Err(format!(
                "slot {} past n was written (the guard did not hold)",
                G4_REMEDY_N as usize + at
            ));
        }
        Ok(())
    };
    let result = remedy();
    failed |= result.is_err();
    let remedy_line = match result {
        Ok(()) => format!(
            "G4 remedy: ok, {G4_REMEDY_N} elements (not a multiple of the width) as Extent::new({G4_WIDTH}, {rows}, 1) with `i = id.y * width + id.x; if i >= n {{ return; }}`: all values exact against the host and all {} slots past n still hold the sentinel",
            slots - G4_REMEDY_N
        ),
        Err(reason) => format!("G4 remedy: FAILED, {reason}"),
    };
    (vec![line, remedy_line], failed)
}
