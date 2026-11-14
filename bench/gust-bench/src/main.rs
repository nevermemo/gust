//! `gust-bench`: a custom benchmark harness for GUST (M10 spike B1).
//!
//! ```text
//! gust-bench run --out run1.json [--label NAME] [--quick] [--workloads W1,W2] [--sizes N,N] [--backends a,b]
//!                [--samples N] [--cap-secs S] [--ablation bind-group-reuse,staging-reuse,one-submission|all]
//!                [--settle-secs S] [--window-secs S] [--no-sampler]
//! gust-bench aggregate --json out.json --md out.md run1.json run2.json run3.json
//! gust-bench dump-wgsl            # the WGSL GUST generates for each kernel, on stdout
//! gust-bench probe-adapters       # DXGI adapters, the LUID mapping, and which LUID this process's GPU work shows on
//! gust-bench list                 # backends, workloads, sizes
//! ```
//!
//! `run` writes `--out FILE` and, beside it, `FILE.samples.json` (the quiet-window sampler's record, see `sampler`).
//!
//! Build with `RUSTFLAGS="--cfg gust_dual"` to include the GUST CPU twin (D26). See `bench/README.md`.

mod backend;
mod gust_gpu;
#[cfg(gust_dual)]
mod gust_twin;
mod harness;
mod meta;
mod rayon_cpu;
mod report;
mod sampler;
mod wgsl_wgpu;
mod workloads;

use std::{process::ExitCode, rc::Rc, time::Duration, time::Instant};

use backend::{Backend, BenchResult};
use gust_wgpu::HeadlessDevice;
use harness::{CellInput, Config, prepare_case, run_cell};
use sampler::Sampler;
use serde_json::json;
use wgsl_wgpu::{HandWgsl, Mode, Switch, WgpuContext};
use workloads::{SIZES, Workload};

const USAGE: &str = "usage: gust-bench run --out FILE [--label NAME] [--quick] [--workloads W1,W2] [--sizes N,N] [--backends a,b] [--samples N] [--cap-secs S] [--ablation LIST|all] [--settle-secs S] [--window-secs S] [--no-sampler]\n       gust-bench aggregate --json FILE --md FILE RUN.json...\n       gust-bench dump-wgsl\n       gust-bench probe-adapters\n       gust-bench list";

#[derive(Default)]
struct Options {
    out: Option<String>,
    json: Option<String>,
    md: Option<String>,
    label: Option<String>,
    quick: bool,
    workloads: Option<Vec<Workload>>,
    sizes: Option<Vec<usize>>,
    backends: Option<Vec<String>>,
    samples: Option<usize>,
    cap_secs: Option<f64>,
    /// Ablation backends to add (`wgsl-percall` with one switch on each).
    ablation: Vec<Switch>,
    settle_secs: Option<f64>,
    window_secs: Option<f64>,
    no_sampler: bool,
    positional: Vec<String>,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut options = Options::default();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let mut value = |name: &str| {
            iter.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match arg.as_str() {
            "--out" => options.out = Some(value("--out")?),
            "--json" => options.json = Some(value("--json")?),
            "--md" => options.md = Some(value("--md")?),
            "--label" => options.label = Some(value("--label")?),
            "--quick" => options.quick = true,
            "--workloads" => {
                options.workloads = Some(
                    value("--workloads")?
                        .split(',')
                        .map(|w| Workload::parse(w).ok_or_else(|| format!("unknown workload {w}")))
                        .collect::<Result<_, _>>()?,
                );
            }
            "--sizes" => {
                options.sizes = Some(
                    value("--sizes")?
                        .split(',')
                        .map(|s| s.parse::<usize>().map_err(|e| format!("size {s}: {e}")))
                        .collect::<Result<_, _>>()?,
                );
            }
            "--backends" => {
                options.backends =
                    Some(value("--backends")?.split(',').map(str::to_owned).collect());
            }
            "--samples" => {
                options.samples = Some(value("--samples")?.parse().map_err(|e| format!("{e}"))?);
            }
            "--cap-secs" => {
                options.cap_secs = Some(value("--cap-secs")?.parse().map_err(|e| format!("{e}"))?);
            }
            "--ablation" => {
                let list = value("--ablation")?;
                options.ablation = if list == "all" {
                    Switch::ALL.to_vec()
                } else {
                    list.split(',')
                        .map(|s| Switch::parse(s).ok_or_else(|| format!("unknown switch {s}")))
                        .collect::<Result<_, _>>()?
                };
            }
            "--settle-secs" => {
                options.settle_secs = Some(
                    value("--settle-secs")?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                );
            }
            "--window-secs" => {
                options.window_secs = Some(
                    value("--window-secs")?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                );
            }
            "--no-sampler" => options.no_sampler = true,
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            other => options.positional.push(other.to_owned()),
        }
    }
    Ok(options)
}

fn all_backends(
    device: &Rc<HeadlessDevice>,
    ctx: &Rc<WgpuContext>,
    threads: usize,
    ablation: &[Switch],
) -> Vec<Box<dyn Backend>> {
    let mut backends: Vec<Box<dyn Backend>> = vec![
        Box::new(gust_gpu::GustGpu::new(Rc::clone(device))),
        Box::new(HandWgsl::new(Rc::clone(ctx), Mode::Prebuilt)),
        Box::new(HandWgsl::new(Rc::clone(ctx), Mode::PerCall)),
    ];
    for switch in ablation {
        backends.push(Box::new(HandWgsl::ablation(Rc::clone(ctx), *switch)));
    }
    #[cfg(gust_dual)]
    backends.push(Box::new(gust_twin::GustTwin::new()));
    backends.push(Box::new(rayon_cpu::CpuLoop::rayon(1)));
    backends.push(Box::new(rayon_cpu::CpuLoop::rayon(threads)));
    backends.push(Box::new(rayon_cpu::CpuLoop::serial()));
    backends
}

fn run(options: &Options) -> BenchResult<bool> {
    let out = options.out.as_deref().ok_or("--out is required")?;
    let gust_device = Rc::new(HeadlessDevice::new()?);
    let ctx = Rc::new(WgpuContext::new()?);
    let adapter = gust_device.adapter_info();
    if matches!(adapter.device_type, wgpu::DeviceType::Cpu) {
        return Err(format!("software adapter selected: {}", adapter.name).into());
    }
    if adapter.name != ctx.info.name {
        return Err(format!(
            "GUST ran on `{}` but the hand-WGSL device on `{}`; refusing to compare",
            adapter.name, ctx.info.name
        )
        .into());
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());

    let mut config = Config::default();
    if options.quick {
        config.samples = 5;
        config.min_samples = 3;
        config.cap = Duration::from_secs(2);
        config.cold_repeats = 1;
    }
    if let Some(samples) = options.samples {
        config.samples = samples;
    }
    if let Some(secs) = options.cap_secs {
        config.cap = Duration::from_secs_f64(secs);
    }
    let sizes = options.sizes.clone().unwrap_or_else(|| {
        if options.quick {
            vec![SIZES[0], SIZES[1]]
        } else {
            SIZES.to_vec()
        }
    });
    let workload_list = options
        .workloads
        .clone()
        .unwrap_or_else(|| Workload::ALL.to_vec());
    let mut backends = all_backends(&gust_device, &ctx, threads, &options.ablation);
    if let Some(wanted) = &options.backends {
        for name in wanted {
            if !backends.iter().any(|b| b.id() == name) {
                return Err(format!("unknown or unavailable backend {name}").into());
            }
        }
        backends.retain(|b| wanted.iter().any(|w| w == b.id()));
    }

    let limits = gust_device.limits();
    let machine = meta::machine(
        adapter,
        gust_device.supports_gpu_timestamps(),
        threads,
        limits,
    );
    let load_before = meta::load_snapshot();
    let gpu_before = meta::gpu_snapshot();
    eprintln!("gpu before: {gpu_before}\nload before: {load_before}");
    let mut timing = sampler::Timing::default();
    if let Some(secs) = options.settle_secs {
        timing.settle = Duration::from_secs_f64(secs);
    }
    if let Some(secs) = options.window_secs {
        timing.window = Duration::from_secs_f64(secs);
    }
    let sampler = (!options.no_sampler).then(|| Sampler::open(timing, adapter));
    if let Some(sampler) = &sampler {
        eprintln!("sampler: {}", sampler.header()["luid_mapping"]);
    }
    let mut window_records = Vec::new();
    let started = Instant::now();

    let mut cells = Vec::new();
    let mut failures = Vec::new();
    for workload in &workload_list {
        for (size_index, n) in sizes.iter().enumerate() {
            let (inputs, want) = prepare_case(*n, *workload);
            for backend in &mut backends {
                if let Some(sampler) = &sampler {
                    window_records.push(sampler.sample_cell(json!({
                        "backend": backend.id(), "workload": workload.id(), "n": n,
                    })));
                }
                let input = CellInput {
                    workload: *workload,
                    inputs: &inputs,
                    reference: &want,
                    cold: size_index == 0,
                };
                let cell = run_cell(backend.as_mut(), &input, &config);
                let summary = ["end_to_end", "dispatch_host", "gpu_single"].map(|metric| {
                    cell.json["metrics"][metric]["median_ms"]
                        .as_f64()
                        .map_or("-".to_owned(), |v| format!("{v:.4}"))
                });
                eprintln!(
                    "{} n={n} {}: e2e {} ms, dispatch {} ms, gpu {} ms, checks {}{}",
                    workload.id(),
                    backend.id(),
                    summary[0],
                    summary[1],
                    summary[2],
                    cell.json["checks_passed"],
                    if cell.failures.is_empty() {
                        ""
                    } else {
                        "  FAILED"
                    }
                );
                failures.extend(cell.failures);
                cells.push(cell.json);
            }
        }
    }

    let sidecar = sampler::sidecar_path(out);
    let sampler_json = match &sampler {
        Some(sampler) => {
            let document = json!({ "header": sampler.header(), "windows": window_records });
            std::fs::write(&sidecar, serde_json::to_string(&document)? + "\n")?;
            json!({
                "enabled": true,
                // Only the file name: results files never carry a path.
                "sidecar": sidecar.file_name().and_then(|n| n.to_str()),
                "windows": window_records.len(),
                "settle_s": timing.settle.as_secs_f64(), "window_s": timing.window.as_secs_f64(),
                "rate_hz": timing.rate_hz,
            })
        }
        None => json!({ "enabled": false }),
    };
    let mut document = json!({
        "label": options.label.clone().unwrap_or_else(|| "run".to_owned()),
        "banner": "shared machine, not publishable",
        // The publishable judgement (plan 2.4) is b1b's; until then no file is publishable, and says why.
        "publishable": meta::PUBLISHABLE,
        "publishable_reason": meta::PUBLISHABLE_REASON,
        "sampler": sampler_json,
        "meta": machine,
        "config": {
            "samples": config.samples, "min_samples": config.min_samples,
            "cap_s": config.cap.as_secs_f64(), "batch": config.batch,
            "warmup_end_to_end": config.warmup_end_to_end, "warmup_dispatch": config.warmup_dispatch,
            "workgroup_size": workloads::WORKGROUP, "sizes": sizes,
            "workloads": workload_list.iter().map(|w| w.id()).collect::<Vec<_>>(),
            "backends": backends.iter().map(|b| json!({ "id": b.id(), "description": b.description() })).collect::<Vec<_>>(),
            "hand_wgsl_adapter": ctx.info.name,
        },
        "load_before": load_before,
        "load_after": meta::load_snapshot(),
        "gpu_before": gpu_before,
        "gpu_after": meta::gpu_snapshot(),
        "wall_s": started.elapsed().as_secs_f64(),
        "cells": cells,
        "failures": failures,
    });
    // Route, build and adapter at the top level of every results file (plan 2.5 item 8).
    if let (Some(top), Some(extra)) = (
        document.as_object_mut(),
        meta::provenance(adapter).as_object(),
    ) {
        top.extend(extra.clone());
    }
    std::fs::write(out, serde_json::to_string_pretty(&document)? + "\n")?;
    for failure in &failures {
        eprintln!("FAILURE: {failure}");
    }
    Ok(failures.is_empty())
}

fn dump_wgsl() -> BenchResult<()> {
    let kernels = [
        &workloads::vector_add::kernel::add::DESCRIPTOR,
        &workloads::saxpy::kernel::saxpy::DESCRIPTOR,
    ];
    for descriptor in kernels {
        println!("// ===== {} =====", descriptor.qualified_name());
        println!("{}", gust::slang::compile_wgsl(descriptor)?);
    }
    Ok(())
}

/// Verifies the DXGI-to-PDH LUID mapping on this machine: keeps the adapter under test busy from this process while
/// the sampler observes the GPU-engine counters, then reports on which LUID the process's own utilisation shows up.
/// Exit 0 only when it shows up on the mapped LUID and nowhere else.
#[cfg(windows)]
fn probe_adapters() -> BenchResult<bool> {
    let device = HeadlessDevice::new()?;
    let adapter = device.adapter_info();
    let ctx = Rc::new(WgpuContext::new()?);
    if adapter.name != ctx.info.name {
        return Err(format!(
            "the probe's two devices are on different adapters: `{}` and `{}`",
            adapter.name, ctx.info.name
        )
        .into());
    }
    let sampler = Sampler::open(sampler::Timing::default(), adapter);
    let own = std::process::id();
    let window = Duration::from_secs(3);
    let n = *SIZES.last().expect("sizes");
    let (inputs, _) = prepare_case(n, Workload::VectorAdd);
    let mut busy = HandWgsl::new(Rc::clone(&ctx), Mode::Prebuilt);
    busy.prepare(Workload::VectorAdd, &inputs)?;
    let readings = std::thread::scope(|scope| {
        let observer = scope.spawn(|| sampler.observe(window));
        let until = Instant::now() + window + Duration::from_millis(500);
        while Instant::now() < until {
            if let Err(error) = busy.dispatch() {
                eprintln!("busy dispatch failed: {error}");
                break;
            }
        }
        observer.join().expect("the observer thread does not panic")
    })?;
    let max = sampler::max_by_luid(&readings, own);
    let mapped = sampler.mapped_luid();
    let on_mapped = mapped
        .and_then(|luid| max.get(&luid).copied())
        .unwrap_or(0.0);
    let elsewhere = max
        .iter()
        .filter(|(luid, _)| Some(**luid) != mapped)
        .map(|(_, pct)| *pct)
        .fold(0.0, f64::max);
    let verified = mapped.is_some() && on_mapped > 0.0 && elsewhere == 0.0;
    let report = json!({
        "adapter_under_test": adapter.name,
        "vendor_id": adapter.vendor, "device_id": adapter.device,
        "header": sampler.header(),
        "own_pid": own,
        "own_max_utilisation_by_luid": max.iter().map(|(luid, pct)| (luid.text(), json!(pct))).collect::<serde_json::Map<_, _>>(),
        "mapped_luid": mapped.map(sampler::Luid::text),
        "own_on_mapped_luid_pct": on_mapped,
        "own_on_other_luids_pct": elsewhere,
        "samples": readings.len(),
        "verified": verified,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(verified)
}

#[cfg(not(windows))]
fn probe_adapters() -> BenchResult<bool> {
    Err("probe-adapters needs the Windows GPU-engine counters".into())
}

fn real_main() -> BenchResult<bool> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let Some((command, rest)) = args.split_first() else {
        return Err(USAGE.into());
    };
    let options = parse(rest)?;
    match command.as_str() {
        "run" => run(&options),
        "aggregate" => {
            let json = options.json.as_deref().ok_or("--json is required")?;
            let md = options.md.as_deref().ok_or("--md is required")?;
            report::aggregate(&options.positional, json, md)?;
            Ok(true)
        }
        "dump-wgsl" => dump_wgsl().map(|()| true),
        "probe-adapters" => probe_adapters(),
        "list" => {
            println!("workloads: W1 vector-add, W2 saxpy");
            println!("sizes: {SIZES:?}");
            println!("gust_dual (CPU twin built in): {}", cfg!(gust_dual));
            Ok(true)
        }
        _ => Err(USAGE.into()),
    }
}

fn main() -> ExitCode {
    match real_main() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}
