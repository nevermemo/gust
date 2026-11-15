//! `gust-bench`: a custom benchmark harness for GUST (M10 spike B1).
//!
//! ```text
//! gust-bench run --out run1.json [--label NAME] [--quick] [--workloads W1,W3,W4-256-zipf] [--sizes N,N] [--backends a,b]
//!                [--group publishable|ablation] [--ablation bind-group-reuse,staging-reuse,one-submission,all-on|all]
//!                [--samples N] [--cap-secs S] [--settle-secs S] [--window-secs S] [--no-sampler] [--warmup-secs S]
//!                [--cold-only] [--cold-repeats N] [--seed N]
//! gust-bench aggregate --json out.json --md out.md run1.json run2.json run3.json
//! gust-bench dump-wgsl            # the WGSL GUST generates for each kernel, on stdout (slang-runtime builds)
//! gust-bench probe-adapters       # DXGI adapters, the LUID mapping, and which LUID this process's GPU work shows on
//! gust-bench list                 # backends, workloads, sizes, groups
//! ```
//!
//! `run` writes `--out FILE` and, beside it, `FILE.samples.json` (the quiet-window sampler's record, see `sampler`).
//!
//! Build with `RUSTFLAGS="--cfg gust_dual"` to include the GUST CPU twin (D26). The `GUST_PROFILE=cpu` build has no GPU
//! runtime (`--no-default-features`): it runs the CPU backends only, among them `gust-cpu`. See `bench/README.md`.

mod backend;
mod group;
#[cfg(any(gust_dual, not(feature = "gpu")))]
mod gust_cpu;
#[cfg(feature = "gpu")]
mod gust_gpu;
mod harness;
mod meta;
mod rayon_cpu;
mod report;
#[cfg(feature = "gpu")]
mod route;
#[cfg(feature = "gpu")]
mod sampler;
#[cfg(feature = "gpu")]
mod wgsl_wgpu;
mod workloads;

#[cfg(feature = "gpu")]
use std::rc::Rc;
use std::{process::ExitCode, time::Duration, time::Instant};

use backend::{Backend, BenchResult, Kind};
use group::Group;
#[cfg(feature = "gpu")]
use gust_wgpu::HeadlessDevice;
use harness::{CellInput, Config, prepare_case, run_cell, run_gpu_group};
#[cfg(feature = "gpu")]
use sampler::Sampler;
use serde_json::{Value, json};
#[cfg(feature = "gpu")]
use wgsl_wgpu::{HandWgsl, Mode, Switch, WgpuContext};
use workloads::Workload;

const USAGE: &str = "usage: gust-bench run --out FILE [--label NAME] [--quick] [--workloads W1,W3,...] [--sizes N,N] [--backends a,b] [--group publishable|ablation] [--ablation LIST|all-on|all] [--samples N] [--cap-secs S] [--settle-secs S] [--window-secs S] [--no-sampler] [--warmup-secs S] [--cold-only] [--cold-repeats N] [--seed N]\n       gust-bench aggregate --json FILE --md FILE RUN.json...\n       gust-bench dump-wgsl\n       gust-bench probe-adapters\n       gust-bench list";

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
    group: Option<Group>,
    samples: Option<usize>,
    cap_secs: Option<f64>,
    /// The raw `--ablation` list (switch names, `all-on`, `all`), parsed where the hand-WGSL backends exist.
    ablation: Option<String>,
    settle_secs: Option<f64>,
    window_secs: Option<f64>,
    no_sampler: bool,
    cold_only: bool,
    cold_repeats: Option<usize>,
    seed: Option<u64>,
    warmup_secs: Option<f64>,
    positional: Vec<String>,
}

/// The `--ablation` list: switch names, `all-on` (every switch on, result `Vec` still fresh per call) and `all`,
/// which is each switch alone plus `all-on`: the whole ablation table.
#[cfg(feature = "gpu")]
fn parse_ablation(list: &str) -> Result<(Vec<Switch>, bool), String> {
    let mut switches = Vec::new();
    let mut all_on = false;
    for item in list.split(',') {
        match item {
            "all" => {
                switches = Switch::ALL.to_vec();
                all_on = true;
            }
            "all-on" => all_on = true,
            other => {
                let switch =
                    Switch::parse(other).ok_or_else(|| format!("unknown switch {other}"))?;
                if !switches.contains(&switch) {
                    switches.push(switch);
                }
            }
        }
    }
    Ok((switches, all_on))
}

/// The group of a run and its ablation list. The ablation backends run in their own group (plan R1.2-3), compared only
/// against the wgsl-percall and wgsl-prebuilt of that group, so `--ablation` needs `--group ablation`; the ablation
/// group without a list is the whole table.
fn resolve_group(
    group: Option<Group>,
    ablation: Option<&str>,
) -> Result<(Group, Option<String>), String> {
    match (group.unwrap_or(Group::Publishable), ablation) {
        (Group::Publishable, None) => Ok((Group::Publishable, None)),
        (Group::Publishable, Some(_)) => Err(
            "--ablation needs --group ablation: the ablation backends run in a group of their own"
                .to_owned(),
        ),
        (Group::Ablation, None) => Ok((Group::Ablation, Some("all".to_owned()))),
        (Group::Ablation, Some(list)) => Ok((Group::Ablation, Some(list.to_owned()))),
    }
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
                let mut list = Vec::new();
                for item in value("--workloads")?.split(',') {
                    list.extend(
                        Workload::parse_family(item)
                            .ok_or_else(|| format!("unknown workload {item}"))?,
                    );
                }
                options.workloads = Some(list);
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
            "--group" => {
                let text = value("--group")?;
                options.group =
                    Some(Group::parse(&text).ok_or_else(|| {
                        format!("unknown group {text} (publishable or ablation)")
                    })?);
            }
            "--samples" => {
                options.samples = Some(value("--samples")?.parse().map_err(|e| format!("{e}"))?);
            }
            "--cap-secs" => {
                options.cap_secs = Some(value("--cap-secs")?.parse().map_err(|e| format!("{e}"))?);
            }
            "--ablation" => options.ablation = Some(value("--ablation")?),
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
            "--cold-only" => options.cold_only = true,
            "--cold-repeats" => {
                options.cold_repeats = Some(
                    value("--cold-repeats")?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                );
            }
            "--seed" => {
                options.seed = Some(value("--seed")?.parse().map_err(|e| format!("{e}"))?);
            }
            "--warmup-secs" => {
                options.warmup_secs = Some(
                    value("--warmup-secs")?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                );
            }
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            other => options.positional.push(other.to_owned()),
        }
    }
    Ok(options)
}

/// The sizes of one workload in a run: `--sizes` for all of them, else the workload's own ladder (the first two rungs
/// of W1 and W2, the first of W3 and W4, with `--quick`).
fn sizes_for(options: &Options, workload: Workload) -> Vec<usize> {
    match &options.sizes {
        Some(sizes) => sizes.clone(),
        None if options.quick => workload
            .sizes()
            .iter()
            .copied()
            .take(if matches!(workload.family(), "W1" | "W2") {
                2
            } else {
                1
            })
            .collect(),
        None => workload.sizes().to_vec(),
    }
}

/// The GPU backends of a group. The publishable group is GUST (the route of this build: `gust-gpu`, or `gust-gpu-spirv`
/// in the `spirv-route` build), wgsl-prebuilt and wgsl-percall; the ablation group is wgsl-percall and wgsl-prebuilt
/// plus the switch ablations, with no GUST backend.
#[cfg(feature = "gpu")]
fn gpu_backends(
    group: Group,
    device: &Rc<HeadlessDevice>,
    ctx: &Rc<WgpuContext>,
    ablation: &[Switch],
    all_on: bool,
) -> Vec<Box<dyn Backend>> {
    let mut backends: Vec<Box<dyn Backend>> = match group {
        Group::Publishable => vec![
            Box::new(gust_gpu::GustGpu::new(Rc::clone(device))),
            Box::new(HandWgsl::new(Rc::clone(ctx), Mode::Prebuilt)),
            Box::new(HandWgsl::new(Rc::clone(ctx), Mode::PerCall)),
        ],
        Group::Ablation => vec![
            Box::new(HandWgsl::new(Rc::clone(ctx), Mode::PerCall)),
            Box::new(HandWgsl::new(Rc::clone(ctx), Mode::Prebuilt)),
        ],
    };
    for switch in ablation {
        backends.push(Box::new(HandWgsl::ablation(Rc::clone(ctx), *switch)));
    }
    if all_on {
        backends.push(Box::new(HandWgsl::all_on(Rc::clone(ctx))));
    }
    backends
}

/// The CPU backends, run beside the publishable group (not in its rotation). In the `GUST_PROFILE=cpu` build they
/// include `gust-cpu` and `gust-cpu-par`; in a `--cfg gust_dual` GPU build, `gust-twin`.
#[allow(clippy::vec_init_then_push)]
fn cpu_backends(threads: usize) -> Vec<Box<dyn Backend>> {
    let mut backends: Vec<Box<dyn Backend>> = Vec::new();
    #[cfg(any(gust_dual, not(feature = "gpu")))]
    backends.push(Box::new(gust_cpu::GustCpu::new()));
    #[cfg(not(feature = "gpu"))]
    backends.push(Box::new(gust_cpu::GustCpuPar::new()));
    backends.push(Box::new(rayon_cpu::CpuLoop::rayon(1)));
    backends.push(Box::new(rayon_cpu::CpuLoop::rayon(threads)));
    backends.push(Box::new(rayon_cpu::CpuLoop::serial()));
    backends
}

/// Everything a results file holds, before it is written. Its own type so that a test can build the file the way
/// `run` does and read the real output.
struct RunRecord {
    label: String,
    run_seed: u64,
    /// `publishable` or `ablation`: the group every sample of the file came from.
    group: &'static str,
    /// The kill rule of the publishable group (`group::kill_rule`), null for the ablation group.
    kill_rule: Value,
    sampler: Value,
    machine: Value,
    config: Value,
    load_before: Value,
    load_after: Value,
    gpu_before: Value,
    gpu_after: Value,
    wall_s: f64,
    cells: Vec<Value>,
    failures: Vec<String>,
    /// Route, build and adapter (plan 2.5 item 8), merged into the top level.
    provenance: Value,
}

impl RunRecord {
    fn into_document(self) -> Value {
        let mut document = json!({
            "label": self.label,
            "run_seed": self.run_seed,
            "group": self.group,
            "kill_rule": self.kill_rule,
            "banner": "shared machine, not publishable",
            // The publishable judgement (plan 2.4) is b1b's; until then no file is publishable, and says why.
            "publishable": meta::PUBLISHABLE,
            "publishable_reason": meta::PUBLISHABLE_REASON,
            "sampler": self.sampler,
            "meta": self.machine,
            "config": self.config,
            "load_before": self.load_before,
            "load_after": self.load_after,
            "gpu_before": self.gpu_before,
            "gpu_after": self.gpu_after,
            "wall_s": self.wall_s,
            "cells": self.cells,
            "failures": self.failures,
        });
        if let (Some(top), Some(extra)) = (document.as_object_mut(), self.provenance.as_object()) {
            top.extend(extra.clone());
        }
        document
    }
}

fn build_config(options: &Options, group: Group) -> Config {
    let mut config = Config::default();
    if options.quick {
        config.samples = 5;
        config.min_samples = 3;
        config.cap = Duration::from_secs(2);
        config.cold_repeats = 1;
    }
    if let Some(secs) = options.warmup_secs {
        config.warmup_cap = Duration::from_secs_f64(secs);
    }
    // The run seed fixes every rotation order of the run (recorded in the file); the default comes from the clock.
    config.run_seed = options.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(1, |d| d.as_nanos() as u64)
    });
    config.cold_only = options.cold_only;
    config.group = group.name();
    if let Some(repeats) = options.cold_repeats {
        config.cold_repeats = repeats;
    }
    if let Some(samples) = options.samples {
        config.samples = samples;
    }
    if let Some(secs) = options.cap_secs {
        config.cap = Duration::from_secs_f64(secs);
    }
    config
}

fn config_json(
    config: &Config,
    options: &Options,
    workload_list: &[Workload],
    backends: &[Box<dyn Backend>],
    extra: Value,
) -> Value {
    let mut document = json!({
        "samples": config.samples, "min_samples": config.min_samples,
        "cap_s": config.cap.as_secs_f64(), "batch": config.batch,
        "warmup_end_to_end": config.warmup_end_to_end, "warmup_dispatch": config.warmup_dispatch,
        "workgroup_size": workloads::WORKGROUP,
        "sizes": workload_list.iter().map(|w| (w.id().to_owned(), json!(sizes_for(options, *w)))).collect::<serde_json::Map<_, _>>(),
        "warmup_cap_s": config.warmup_cap.as_secs_f64(), "warmup_batch": config.warmup_batch,
        "group": config.group,
        "sampling": "the group's gpu backends interleaved in rounds of seeded random order, paired against wgsl-prebuilt",
        "cold_only": config.cold_only,
        "cold_repeats": config.cold_repeats,
        "workloads": workload_list.iter().map(|w| w.id()).collect::<Vec<_>>(),
        "backends": backends.iter().map(|b| json!({ "id": b.id(), "description": b.description() })).collect::<Vec<_>>(),
    });
    if let (Some(top), Some(extra)) = (document.as_object_mut(), extra.as_object()) {
        top.extend(extra.clone());
    }
    document
}

/// Keeps the backends the user named; an unknown name is an error.
fn select_backends(
    backends: &mut Vec<Box<dyn Backend>>,
    wanted: Option<&Vec<String>>,
) -> BenchResult<()> {
    if let Some(wanted) = wanted {
        for name in wanted {
            if !backends.iter().any(|b| b.id() == name) {
                return Err(format!("unknown or unavailable backend {name}").into());
            }
        }
        backends.retain(|b| wanted.iter().any(|w| w == b.id()));
    }
    Ok(())
}

/// The cells of a run: for each workload and size, the group's GPU backends as one paired rotation, then each CPU
/// backend alone. `before_cell` is called with a label before each (the sampler's idle window, in a GPU build).
fn run_cells(
    backends: &mut [Box<dyn Backend>],
    workload_list: &[Workload],
    options: &Options,
    config: &Config,
    before_cell: &mut dyn FnMut(&str, Workload, usize),
) -> (Vec<Value>, Vec<String>) {
    let mut cells = Vec::new();
    let mut failures = Vec::new();
    for workload in workload_list {
        for (size_index, n) in sizes_for(options, *workload).iter().enumerate() {
            let (inputs, want) = prepare_case(*n, *workload);
            let input = CellInput {
                workload: *workload,
                inputs: &inputs,
                reference: &want,
                cold: size_index == 0,
            };
            // The GPU backends are one paired group: one idle window, one shared warm-up, interleaved samples.
            let mut gpu: Vec<&mut dyn Backend> = backends
                .iter_mut()
                .filter(|b| b.kind() == Kind::Gpu && b.supports(*workload))
                .map(|b| -> &mut dyn Backend { b.as_mut() })
                .collect();
            let mut produced = Vec::new();
            if !gpu.is_empty() {
                before_cell("gpu-group", *workload, *n);
                produced.extend(run_gpu_group(&mut gpu, &input, config));
            }
            for backend in backends
                .iter_mut()
                .filter(|b| b.kind() != Kind::Gpu && b.supports(*workload))
            {
                before_cell(backend.id(), *workload, *n);
                produced.push(run_cell(backend.as_mut(), &input, config));
            }
            for cell in produced {
                let summary = ["end_to_end", "dispatch_host", "gpu_single"].map(|metric| {
                    cell.json["metrics"][metric]["median_ms"]
                        .as_f64()
                        .map_or("-".to_owned(), |v| format!("{v:.4}"))
                });
                let paired = ["end_to_end", "gpu_single", "gpu_batch_per_dispatch"]
                    .iter()
                    .filter_map(|metric| {
                        cell.json["metrics"][metric]["paired_vs_prebuilt"]["median_ratio"]
                            .as_f64()
                            .map(|r| format!("{metric} {r:.3}x"))
                    })
                    .collect::<Vec<_>>();
                eprintln!(
                    "{} n={n} {}: e2e {} ms, dispatch {} ms, gpu {} ms, checks {}{}{}",
                    workload.id(),
                    cell.json["backend"].as_str().unwrap_or("?"),
                    summary[0],
                    summary[1],
                    summary[2],
                    cell.json["checks_passed"],
                    if paired.is_empty() {
                        String::new()
                    } else {
                        format!(", paired vs prebuilt: {}", paired.join(", "))
                    },
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
    (cells, failures)
}

/// The kill rule of the publishable group over the cells, printed; null for the ablation group.
#[cfg(feature = "gpu")]
fn kill_rule_of(group: Group, cells: &[Value]) -> Value {
    if group != Group::Publishable {
        return Value::Null;
    }
    let record = group::kill_rule(cells);
    eprintln!(
        "kill rule: C re-opened = {} ({} paired GPU-time ratios checked, offenders {})",
        record["c_reopened"], record["checked"], record["offenders"]
    );
    record
}

#[cfg(feature = "gpu")]
fn run(options: &Options) -> BenchResult<bool> {
    let out = options.out.as_deref().ok_or("--out is required")?;
    let (group, ablation_list) = resolve_group(options.group, options.ablation.as_deref())?;
    let (ablation, all_on) = match &ablation_list {
        Some(list) => parse_ablation(list)?,
        None => (Vec::new(), false),
    };
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

    let config = build_config(options, group);
    let workload_list = options
        .workloads
        .clone()
        .unwrap_or_else(|| Workload::ALL.to_vec());
    let mut backends = gpu_backends(group, &gust_device, &ctx, &ablation, all_on);
    if group == Group::Publishable {
        backends.extend(cpu_backends(threads));
    }
    select_backends(&mut backends, options.backends.as_ref())?;

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

    let (cells, failures) = run_cells(
        &mut backends,
        &workload_list,
        options,
        &config,
        &mut |label, workload, n| {
            if let Some(sampler) = &sampler {
                window_records.push(sampler.sample_cell(json!({
                    "backend": label, "workload": workload.id(), "n": n,
                })));
            }
        },
    );

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
    let kill_rule = kill_rule_of(group, &cells);
    let document = RunRecord {
        label: options.label.clone().unwrap_or_else(|| "run".to_owned()),
        run_seed: config.run_seed,
        group: group.name(),
        kill_rule,
        sampler: sampler_json,
        machine,
        config: config_json(
            &config,
            options,
            &workload_list,
            &backends,
            json!({
                "hand_wgsl_adapter": ctx.info.name,
                // Which shader source gust-gpu's cold start used (plan 2.3): a kernel with embedded artifacts is
                // loaded from them, otherwise it is compiled at run time. A run of one kind does not aggregate with
                // the other.
                "shader_source": shader_source(),
            }),
        ),
        load_before,
        load_after: meta::load_snapshot(),
        gpu_before,
        gpu_after: meta::gpu_snapshot(),
        wall_s: started.elapsed().as_secs_f64(),
        cells,
        failures: failures.clone(),
        provenance: meta::provenance(adapter),
    }
    .into_document();
    std::fs::write(out, serde_json::to_string_pretty(&document)? + "\n")?;
    for failure in &failures {
        eprintln!("FAILURE: {failure}");
    }
    Ok(failures.is_empty())
}

/// The `GUST_PROFILE=cpu` build has no GPU runtime: only the CPU backends, one group, no sampler (there is no GPU
/// to idle).
#[cfg(not(feature = "gpu"))]
fn run(options: &Options) -> BenchResult<bool> {
    let out = options.out.as_deref().ok_or("--out is required")?;
    let (group, _) = resolve_group(options.group, options.ablation.as_deref())?;
    if group != Group::Publishable {
        return Err(
            "this build has no GPU runtime: the ablation group needs the default build".into(),
        );
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let config = build_config(options, group);
    let workload_list = options
        .workloads
        .clone()
        .unwrap_or_else(|| Workload::ALL.to_vec());
    let mut backends = cpu_backends(threads);
    select_backends(&mut backends, options.backends.as_ref())?;
    let load_before = meta::load_snapshot();
    let started = Instant::now();
    let (cells, failures) = run_cells(
        &mut backends,
        &workload_list,
        options,
        &config,
        &mut |_, _, _| {},
    );
    let document = RunRecord {
        label: options.label.clone().unwrap_or_else(|| "run".to_owned()),
        run_seed: config.run_seed,
        group: group.name(),
        kill_rule: Value::Null,
        sampler: json!({ "enabled": false }),
        machine: meta::machine_cpu(threads),
        config: config_json(&config, options, &workload_list, &backends, json!({})),
        load_before,
        load_after: meta::load_snapshot(),
        gpu_before: Value::Null,
        gpu_after: Value::Null,
        wall_s: started.elapsed().as_secs_f64(),
        cells,
        failures: failures.clone(),
        provenance: meta::provenance_cpu(),
    }
    .into_document();
    std::fs::write(out, serde_json::to_string_pretty(&document)? + "\n")?;
    for failure in &failures {
        eprintln!("FAILURE: {failure}");
    }
    Ok(failures.is_empty())
}

/// `precompiled` when every benchmark kernel carries an embedded artifact, `run-time compile` when none does.
#[cfg(feature = "gpu")]
fn shader_source() -> &'static str {
    let kernels = workloads::all_descriptors();
    let with = kernels.iter().filter(|k| !k.artifacts.is_empty()).count();
    match with {
        0 => "run-time compile",
        n if n == kernels.len() => "precompiled",
        _ => "mixed",
    }
}

#[cfg(all(feature = "gpu", feature = "slang-runtime"))]
fn dump_wgsl() -> BenchResult<()> {
    for descriptor in workloads::all_descriptors() {
        println!("// ===== {} =====", descriptor.qualified_name());
        println!("{}", gust::slang::compile_wgsl(descriptor)?);
    }
    Ok(())
}

#[cfg(not(all(feature = "gpu", feature = "slang-runtime")))]
fn dump_wgsl() -> BenchResult<()> {
    Err("dump-wgsl needs a build with the gpu and slang-runtime features".into())
}

/// Verifies the DXGI-to-PDH LUID mapping on this machine: keeps the adapter under test busy from this process while
/// the sampler observes the GPU-engine counters, then reports on which LUID the process's own utilisation shows up.
/// Exit 0 only when it shows up on the mapped LUID and nowhere else.
#[cfg(all(windows, feature = "gpu"))]
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
    let n = *Workload::VectorAdd.sizes().last().expect("sizes");
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

#[cfg(not(all(windows, feature = "gpu")))]
fn probe_adapters() -> BenchResult<bool> {
    Err("probe-adapters needs the Windows GPU-engine counters and the gpu feature".into())
}

fn list() {
    println!("workloads and sizes:");
    for workload in Workload::ALL {
        println!(
            "  {:<18} {}: {:?}",
            workload.id(),
            workload.name(),
            workload.sizes()
        );
    }
    println!("groups: publishable (default), ablation");
    println!("gpu runtime in this build: {}", cfg!(feature = "gpu"));
    println!("features: {:?}", meta::build_features());
    println!("GUST profile: {}", gust::profile::NAME);
    println!("gust_dual (CPU twin built in): {}", cfg!(gust_dual));
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
            list();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    #[test]
    fn ablation_lists_parse_and_all_is_the_whole_table() {
        let (switches, all_on) = parse_ablation("all").unwrap();
        assert_eq!(switches, Switch::ALL.to_vec());
        assert!(all_on);
        let (switches, all_on) = parse_ablation("all-on").unwrap();
        assert!(switches.is_empty() && all_on);
        let (switches, all_on) = parse_ablation("staging-reuse,staging-reuse").unwrap();
        assert_eq!(switches, vec![Switch::StagingReuse]);
        assert!(!all_on);
        assert!(parse_ablation("nope").is_err());
    }

    #[test]
    fn the_ablation_backends_run_only_in_the_ablation_group() {
        assert_eq!(
            resolve_group(None, None).unwrap(),
            (Group::Publishable, None)
        );
        assert!(resolve_group(Some(Group::Publishable), Some("all")).is_err());
        assert!(resolve_group(None, Some("all-on")).is_err());
        assert_eq!(
            resolve_group(Some(Group::Ablation), None).unwrap(),
            (Group::Ablation, Some("all".to_owned()))
        );
        assert_eq!(
            resolve_group(Some(Group::Ablation), Some("all-on")).unwrap(),
            (Group::Ablation, Some("all-on".to_owned()))
        );
    }

    #[test]
    fn the_group_switch_parses_and_workload_families_expand() {
        let args = |list: &[&str]| list.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        let options = parse(&args(&[
            "--group",
            "ablation",
            "--workloads",
            "W3,W4-256-zipf",
        ]))
        .unwrap();
        assert_eq!(options.group, Some(Group::Ablation));
        let workloads = options.workloads.unwrap();
        assert_eq!(workloads.len(), 4);
        assert!(workloads.contains(&Workload::Hist256Zipf));
        assert!(parse(&args(&["--group", "nope"])).is_err());
        assert!(parse(&args(&["--workloads", "W9"])).is_err());
    }

    #[test]
    fn each_workload_runs_its_own_ladder_and_quick_shortens_it() {
        let mut options = Options::default();
        assert_eq!(sizes_for(&options, Workload::Poly16), workloads::POLY_SIZES);
        assert_eq!(
            sizes_for(&options, Workload::Hist256Zipf),
            workloads::HIST_SIZES
        );
        assert_eq!(sizes_for(&options, Workload::VectorAdd).len(), 5);
        options.quick = true;
        assert_eq!(sizes_for(&options, Workload::VectorAdd).len(), 2);
        assert_eq!(sizes_for(&options, Workload::Poly4).len(), 1);
        options.sizes = Some(vec![4096]);
        assert_eq!(sizes_for(&options, Workload::Hist256Uniform), vec![4096]);
    }

    #[test]
    fn the_per_run_file_says_not_publishable_with_the_reason_and_carries_the_provenance() {
        let document = RunRecord {
            label: "t".to_owned(),
            run_seed: 7,
            group: "publishable",
            kill_rule: json!({ "c_reopened": false }),
            sampler: json!({ "enabled": false }),
            machine: json!({}),
            config: json!({ "samples": 5 }),
            load_before: json!(null),
            load_after: json!(null),
            gpu_before: json!(null),
            gpu_after: json!(null),
            wall_s: 1.0,
            cells: Vec::new(),
            failures: Vec::new(),
            provenance: json!({ "route": "wgsl", "build": { "commit": "abc" }, "adapter": { "name": "A" } }),
        }
        .into_document();
        assert_eq!(document["publishable"], false);
        assert_eq!(document["publishable_reason"], meta::PUBLISHABLE_REASON);
        assert_eq!(document["banner"], "shared machine, not publishable");
        assert_eq!(document["route"], "wgsl");
        assert_eq!(document["build"]["commit"], "abc");
        assert_eq!(document["adapter"]["name"], "A");
        assert_eq!(document["config"]["samples"], 5);
        assert_eq!(document["group"], "publishable");
        assert_eq!(document["kill_rule"]["c_reopened"], false);
    }
}
