//! `gust-bench trace`: wgpu's own API trace of the hand-WGSL backends , a `wgpu-trace` build.
//!
//! One directory per (backend, workload, size, operation) under `--out`: a fresh device opened with
//! `wgpu::Trace::Directory`, the backend prepared and poisoned, then exactly one call (`batch16` is `gpu_time(16)`, the
//! 16-pass batch; `end_to_end` is one host-in, host-out call). The trace holds the whole device life, so the setup comes
//! first and the measured call is the tail. GUST's device is opened inside `gust-wgpu` with the default (off) trace and
//! `HeadlessDevice` has no way to pass one, so GUST cannot be traced from this crate; `profile` records its ordered API
//! call sequence from wgpu-core's log instead.

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
                    other => {
                        return Err(format!(
                            "trace covers wgsl-prebuilt and wgsl-percall, not {other}"
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
