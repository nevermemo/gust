//! `gust-bench trace`: wgpu's own API trace of the hand-WGSL backends (and of GUST, see `trace_gust`), a `wgpu-trace` build.
//!
//! One directory per (backend, workload, size, operation) under `--out`: a fresh device opened with
//! `wgpu::Trace::Directory`, the backend prepared and poisoned, then exactly one call (`batch16` is `gpu_time(16)`, the
//! 16-pass batch; `end_to_end` is one host-in, host-out call). The trace holds the whole device life, so the setup comes
//! first and the measured call is the tail. GUST's device is opened inside `gust-wgpu` with the default (off) trace and
//! `HeadlessDevice` has no way to pass one, so GUST can be traced only in a build of `gust-wgpu` that carries the
//! (never merged) probe hook, see `trace_gust`; `profile` records its ordered API call sequence from wgpu-core's log
//! without it.

use std::{path::PathBuf, rc::Rc};

use crate::{
    Options,
    backend::{Backend, BenchResult},
    harness::prepare_case,
    wgsl_wgpu::{HandWgsl, Mode, WgpuContext},
    workloads::Workload,
};

pub fn run(options: &Options) -> BenchResult<bool> {
    let root = PathBuf::from(options.out.as_deref().ok_or("--out DIR is required")?);
    let workloads = options
        .workloads
        .clone()
        .unwrap_or_else(|| vec![Workload::VectorAdd]);
    let sizes = options
        .sizes
        .clone()
        .unwrap_or_else(|| vec![1_048_576, 4_194_304]);
    let backends = options
        .backends
        .clone()
        .unwrap_or_else(|| vec!["wgsl-prebuilt".to_owned(), "wgsl-percall".to_owned()]);
    // `--label batch16` or `--label end_to_end` picks one operation; the default is both.
    let ops: Vec<&str> = match options.label.as_deref() {
        Some("batch16") => vec!["batch16"],
        Some("end_to_end") => vec!["end_to_end"],
        None => vec!["batch16", "end_to_end"],
        Some(other) => return Err(format!("--label is batch16 or end_to_end, not {other}").into()),
    };
    for workload in workloads {
        for &n in &sizes {
            let (inputs, _) = prepare_case(n, workload);
            for id in &backends {
                let mode = match id.as_str() {
                    "wgsl-prebuilt" => Mode::Prebuilt,
                    "wgsl-percall" => Mode::PerCall,
                    "gust-gpu" | "gust-gpu-spirv" => {
                        for op in &ops {
                            trace_gust(&root, id, workload, n, &inputs, op)?;
                        }
                        continue;
                    }
                    other => {
                        return Err(format!(
                            "trace covers wgsl-prebuilt, wgsl-percall and gust-gpu, not {other}"
                        )
                        .into());
                    }
                };
                for op in &ops {
                    let dir = root.join(format!("{id}-{}-{n}-{op}", workload.id()));
                    std::fs::create_dir_all(&dir)?;
                    let ctx = Rc::new(WgpuContext::open(Some(&dir))?);
                    let mut backend = HandWgsl::new(Rc::clone(&ctx), mode);
                    backend.prepare(workload, &inputs)?;
                    backend.poison()?;
                    match *op {
                        "batch16" => {
                            backend.gpu_time(16)?;
                        }
                        _ => {
                            let mut out = Vec::new();
                            backend.end_to_end(&inputs, &mut out)?;
                        }
                    }
                    drop(backend);
                    drop(ctx);
                    eprintln!("traced {}", dir.display());
                }
            }
        }
    }
    Ok(true)
}

/// Traces GUST's own device. `HeadlessDevice` has no trace option, so this needs a `gust-wgpu` build that reads
/// `GUST_WGPU_TRACE_DIR` when it opens its device (the m10.b1c probe patch, built with `--features
/// gust-wgpu/probe-trace`; it is not part of the crate). Without that hook the run fails here and removes the (empty)
/// directory it made, instead of reporting a trace that is not there.
fn trace_gust(
    root: &std::path::Path,
    id: &str,
    workload: Workload,
    n: usize,
    inputs: &crate::workloads::Inputs,
    op: &str,
) -> BenchResult<()> {
    let dir = root.join(format!("{id}-{}-{n}-{op}", workload.id()));
    std::fs::create_dir_all(&dir)?;
    // SAFETY: `set_var` is only unsound when another thread reads or writes the process environment at the same time.
    // `trace` is a diagnostic subcommand that runs no sampler or rayon pool and spawns no thread of its own (a driver
    // opened by an earlier loop iteration may own threads). What carries the soundness is that on Windows, the only
    // platform it is run on, the call is `SetEnvironmentVariableW`, which is not racy with the loader's own reads. Do not
    // run `trace` from a multi-threaded caller on another platform.
    unsafe { std::env::set_var("GUST_WGPU_TRACE_DIR", &dir) };
    let device = Rc::new(gust_wgpu::HeadlessDevice::new()?);
    let mut backend = crate::gust_gpu::GustGpu::new(Rc::clone(&device));
    backend.prepare(workload, inputs)?;
    backend.poison()?;
    match op {
        "batch16" => {
            backend.gpu_time(16)?;
        }
        _ => {
            let mut out = Vec::new();
            backend.end_to_end(inputs, &mut out)?;
        }
    }
    drop(backend);
    drop(device);
    if !dir.join("trace.ron").exists() {
        // Do not leave an empty directory behind for a trace that is not there.
        let _ = std::fs::remove_dir(&dir);
        return Err(format!(
            "no trace.ron in {}: gust-wgpu was built without the probe trace hook",
            dir.display()
        )
        .into());
    }
    eprintln!("traced {}", dir.display());
    Ok(())
}
