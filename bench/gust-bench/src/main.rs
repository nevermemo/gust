//! `gust-bench`: a custom benchmark harness for GUST (M10 spike B1).
//!
//! ```text
//! gust-bench run --out run1.json [--label NAME] [--quick] [--workloads W1,W3,W4-256-zipf] [--sizes N,N] [--backends a,b]
//!                [--group publishable|spirv|ablation] [--ablation bind-group-reuse,staging-reuse,one-submission,all-on|all]
//!                [--samples N] [--cap-secs S] [--settle-secs S] [--window-secs S] [--no-sampler] [--warmup-secs S]
//!                [--cold-only] [--cold-repeats N] [--seed N] [--with gust-setup,gust-shader,gust-device,gust-unchecked,hand-unchecked,wgsl-prebuilt-b]
//!                [--functional]   # correctness only, quick sizes, any adapter (lavapipe included), no sampler, no timing published
//!                [--adapter "NAME"] # the one adapter whose name contains NAME (letter case ignored), for both devices of the run
//! gust-bench aggregate --json out.json --md out.md run1.json run2.json run3.json
//! gust-bench dump-wgsl            # the WGSL GUST generates for each kernel, on stdout (slang-runtime builds)
//! gust-bench dump-kernels --out DIR # each kernel's Slang source, WGSL and slangc SPIR-V, as files (slang-runtime builds)
//! gust-bench profile --out p.json [--workloads W1,W2] [--sizes N,N] [--backends a,b] [--samples CALLS]
//!                                 # host-side counts per call (allocations, wgpu API calls); needs --features profile-counts
//! gust-bench trace --out DIR [--workloads W1] [--sizes N,N] [--backends wgsl-prebuilt,wgsl-percall,gust-gpu] [--label batch16|end_to_end]
//!                                 # wgpu API traces of the hand-WGSL backends (and, with the gust-wgpu probe-trace hook, GUST); needs --features wgpu-trace
//! gust-bench probe-adapters       # DXGI adapters, the LUID mapping, and which LUID this process's GPU work shows on
//! gust-bench list                 # backends, workloads, sizes, groups
//! ```
//!
//! `run` writes `--out FILE` and, beside it, `FILE.samples.json` (the quiet-window sampler's record, see `sampler`).
//!
//! Build with `RUSTFLAGS="--cfg gust_dual"` to include the GUST CPU twin (D26). The `GUST_PROFILE=cpu` build has no GPU
//! runtime (`--no-default-features`): it runs the CPU backends only, among them `gust-cpu`. See `bench/README.md`.

#[cfg(feature = "gpu")]
mod adapter;
mod backend;
#[cfg(feature = "cubecl")]
mod cubecl_backend;
mod group;
#[cfg(any(gust_dual, not(feature = "gpu")))]
mod gust_cpu;
#[cfg(feature = "gpu")]
mod gust_gpu;
mod harness;
mod judge;
mod meta;
#[cfg(feature = "profile-counts")]
mod profile;
mod rayon_cpu;
mod report;
#[cfg(feature = "gpu")]
mod route;
#[cfg(feature = "rust-gpu")]
mod rustgpu;
#[cfg(any(test, feature = "rust-gpu"))]
mod rustgpu_module;
mod sampler;
#[cfg(feature = "wgpu-trace")]
mod trace;
#[cfg(feature = "gpu")]
mod wgsl_wgpu;
mod workloads;

#[cfg(feature = "gpu")]
use std::rc::Rc;
use std::{process::ExitCode, time::Duration, time::Instant};

use backend::{Backend, BenchResult, Kind};
use group::Group;
#[cfg(feature = "gpu")]
use gust_wgpu::{DeviceHandle, DeviceOptions, FeaturesPolicy, HeadlessDevice, LimitsPolicy};
use harness::{CellInput, Config, prepare_case, run_cell, run_gpu_group};
#[cfg(feature = "gpu")]
use sampler::AdapterIdent;
use sampler::Sampler;
use serde_json::{Value, json};
#[cfg(feature = "gpu")]
use wgsl_wgpu::{HandWgsl, Mode, Switch, WgpuContext};
use workloads::Workload;

const USAGE: &str = "usage: gust-bench run --out FILE [--label NAME] [--quick] [--workloads W1,W3,...] [--sizes N,N] [--backends a,b] [--group publishable|spirv|ablation] [--ablation LIST|all-on|all] [--with gust-setup,gust-shader,gust-device,gust-unchecked,hand-unchecked,wgsl-prebuilt-b,gust-default,gust-perpass,gust-fused,gust-frame,gust-frame-perpass] [--gust-only] [--device-order checked-first|unchecked-first] [--samples N] [--cap-secs S] [--settle-secs S] [--window-secs S] [--no-sampler] [--warmup-secs S] [--cold-only] [--cold-repeats N] [--seed N] [--functional] [--adapter NAME]\n       gust-bench aggregate --json FILE --md FILE RUN.json...\n       gust-bench dump-wgsl\n       gust-bench dump-kernels --out DIR (slang-runtime build)\n       gust-bench probe-adapters\n       gust-bench profile --out FILE [--workloads W1,W2] [--sizes N,N] [--backends a,b] [--samples CALLS] (profile-counts build)\n       gust-bench trace --out DIR [--workloads W1] [--sizes N,N] [--backends wgsl-prebuilt,wgsl-percall,gust-gpu] [--label batch16|end_to_end] (wgpu-trace build)\n       gust-bench list";

/// The order in which the checked and the unchecked GUST device of a pf.trust run are opened.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum DeviceOrder {
    #[default]
    CheckedFirst,
    UncheckedFirst,
}

impl DeviceOrder {
    fn parse(text: &str) -> Option<Self> {
        match text {
            "checked-first" => Some(Self::CheckedFirst),
            "unchecked-first" => Some(Self::UncheckedFirst),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::CheckedFirst => "checked-first",
            Self::UncheckedFirst => "unchecked-first",
        }
    }
}

#[derive(Default)]
struct Options {
    out: Option<String>,
    json: Option<String>,
    md: Option<String>,
    label: Option<String>,
    quick: bool,
    /// `--functional`: a correctness-only run (every cell's checks on the `--quick` sizes), accepted on a software
    /// adapter, with no sampler; no timing of it is ever published.
    functional: bool,
    /// `--adapter NAME`: the adapter whose name contains NAME (see `adapter`); `None` keeps wgpu's high-performance choice.
    adapter: Option<String>,
    workloads: Option<Vec<Workload>>,
    sizes: Option<Vec<usize>>,
    backends: Option<Vec<String>>,
    group: Option<Group>,
    samples: Option<usize>,
    cap_secs: Option<f64>,
    /// The raw `--ablation` list (switch names, `all-on`, `all`), parsed where the hand-WGSL backends exist.
    ablation: Option<String>,
    /// `--with gust-setup,gust-shader,gust-device,gust-unchecked,hand-unchecked,wgsl-prebuilt-b`: the probe backends, appended to the
    /// publishable group in the order given (see `gpu_backends`). The first three are the m10.b1c hand probes;
    /// `gust-unchecked` is `gust-gpu` on a second device opened with `DeviceOptions::unchecked_kernels` and
    /// `hand-unchecked` is the hand text with wgpu's runtime checks off (pf.trust, its reproduction control);
    /// `wgsl-prebuilt-b` is the reference again on a device of its own, opened like the hand one (a noise floor).
    /// The pf.onepass GUST rows `gust-default`, `gust-perpass`, `gust-fused`, `gust-frame` and `gust-frame-perpass` are also
    /// `--with` names (W6-hand only, probe rows).
    with_probe: Vec<String>,
    /// `--device-order checked-first|unchecked-first`: which of the two GUST devices of `--with gust-unchecked` is opened
    /// first (default `checked-first`). The first-created variant can be slower on the Radeon 890M, so a series swaps it.
    device_order: Option<DeviceOrder>,
    settle_secs: Option<f64>,
    window_secs: Option<f64>,
    no_sampler: bool,
    /// `--gust-only`: the GUST rows (the base `gust-gpu` and the `gust-default`, `gust-perpass`, `gust-fused`,
    /// `gust-frame` and `gust-frame-perpass` probe
    /// rows) run alone, the form Breeze runs; no hand row, no CPU row, no paired reference. Such a run is never
    /// publishable.
    gust_only: bool,
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
///
/// The default group follows the build's route (D193): `publishable` in the WGSL build, `spirv` in the `spirv-route`
/// build, which refuses `--group publishable` (and the WGSL build refuses `--group spirv`).
fn resolve_group(
    group: Option<Group>,
    ablation: Option<&str>,
    spirv_route: bool,
) -> Result<(Group, Option<String>), String> {
    let group = group.unwrap_or_else(|| group::default_group(spirv_route));
    group::check_route(group, spirv_route)?;
    match (group, ablation) {
        (Group::Publishable | Group::Spirv, None) => Ok((group, None)),
        (Group::Publishable | Group::Spirv, Some(_)) => Err(
            "--ablation needs --group ablation: the ablation backends run in a group of their own"
                .to_owned(),
        ),
        (Group::Ablation, None) => Ok((Group::Ablation, Some("all".to_owned()))),
        (Group::Ablation, Some(list)) => Ok((Group::Ablation, Some(list.to_owned()))),
    }
}

/// `--with` probes are hand-WGSL probes of the checked baseline: they run in the `publishable` group only.
#[cfg(feature = "gpu")]
fn check_probes_group(group: Group, with_probe: &[String]) -> Result<(), String> {
    if !with_probe.is_empty() && group != Group::Publishable {
        return Err(format!(
            "--with runs in the publishable group; the {} group takes no probes (its rows are fixed)",
            group.name()
        ));
    }
    Ok(())
}

/// `--gust-only`: GUST's own rows run alone (no hand row, no CPU row), the form Breeze runs. The run names them with
/// `--backends` (`gust-gpu` or the group's GUST id, and `gust-gpu+...` probe rows), so no other row can sneak in, and it
/// is never publishable. The ablation group has no GUST row.
#[cfg(feature = "gpu")]
fn check_gust_only(options: &Options, group: Group) -> Result<(), String> {
    if !options.gust_only {
        return Ok(());
    }
    if group == Group::Ablation {
        return Err(
            "--gust-only needs a group with a GUST row; the ablation group has none".into(),
        );
    }
    let Some(wanted) = &options.backends else {
        return Err("--gust-only needs --backends naming the GUST rows to run".into());
    };
    let gust_id = group.gust_backend();
    if let Some(other) = wanted
        .iter()
        .find(|name| name.as_str() != gust_id && !name.starts_with("gust-gpu+"))
    {
        return Err(format!(
            "--gust-only runs GUST rows alone; {other} is not one ({gust_id}, or a gust-gpu+ probe row)"
        ));
    }
    if options
        .with_probe
        .iter()
        .any(|name| gust_gpu::Variant::parse(name).is_none())
    {
        return Err(
            "--gust-only takes no hand probe (--with gust-setup, gust-shader, gust-device)".into(),
        );
    }
    Ok(())
}

/// Whether this build runs GUST on the SPIR-V route (the `spirv-route` feature; the GPU-less build has no route).
fn spirv_route() -> bool {
    #[cfg(feature = "gpu")]
    {
        gust_gpu::backend_id() == group::SPIRV_GUST
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
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
            "--functional" => options.functional = true,
            "--adapter" => options.adapter = Some(value("--adapter")?),
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
                options.group = Some(Group::parse(&text).ok_or_else(|| {
                    format!("unknown group {text} (publishable, spirv or ablation)")
                })?);
            }
            "--samples" => {
                options.samples = Some(value("--samples")?.parse().map_err(|e| format!("{e}"))?);
            }
            "--cap-secs" => {
                options.cap_secs = Some(value("--cap-secs")?.parse().map_err(|e| format!("{e}"))?);
            }
            "--ablation" => options.ablation = Some(value("--ablation")?),
            "--device-order" => {
                let text = value("--device-order")?;
                options.device_order = Some(DeviceOrder::parse(&text).ok_or_else(|| {
                    format!("unknown --device-order {text} (checked-first or unchecked-first)")
                })?);
            }
            "--with" => {
                for item in value("--with")?.split(',') {
                    match item {
                        "gust-setup" | "gust-shader" | "gust-device" | "gust-unchecked"
                        | "hand-unchecked" | "wgsl-prebuilt-b" | "gust-default"
                        | "gust-perpass" | "gust-fused" | "gust-frame" | "gust-frame-perpass" => {
                            options.with_probe.push(item.to_owned());
                        }
                        other => {
                            return Err(if other == "unchecked" {
                                "--with unchecked is gone: the unchecked hand rows are the spirv group's own (D193); run them with --group spirv in the spirv-route build".to_owned()
                            } else {
                                format!(
                                    "--with takes gust-setup, gust-shader, gust-device, gust-unchecked, hand-unchecked, wgsl-prebuilt-b, or the W6-hand GUST rows gust-default, gust-perpass, gust-fused, gust-frame and gust-frame-perpass, not {other}"
                                )
                            });
                        }
                    }
                }
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
            "--gust-only" => options.gust_only = true,
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
    // A functional run is the quick sizes and samples: it checks every cell's correctness and nothing else.
    if options.functional {
        options.quick = true;
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

/// The largest `--sizes` value W9 takes: its size is the matrix side `N` (the matrices hold `N * N` words), so 4096 is
/// 64 MiB a matrix; an element count such as `--sizes 1048576` meant for W1 would ask for 2^40 words.
#[cfg(feature = "gpu")]
const MATMUL_MAX_SIDE: usize = 4096;

/// Refuses a `--sizes` list that cannot be a matrix side for a W9 selected beside other workloads (the ladder is not
/// affected: it is always sides). Errors name the cause.
#[cfg(feature = "gpu")]
fn check_sizes(options: &Options, workloads: &[Workload]) -> Result<(), String> {
    let Some(sizes) = &options.sizes else {
        return Ok(());
    };
    if workloads.contains(&Workload::MatMul)
        && let Some(n) = sizes.iter().find(|n| **n > MATMUL_MAX_SIDE)
    {
        return Err(format!(
            "--sizes {n} is too large for W9: its size is the matrix side N (at most {MATMUL_MAX_SIDE}), not an element count; run W9 with its own --sizes in a separate run (--workloads W9 --sizes 256,512)"
        ));
    }
    Ok(())
}

/// The GPU backends of a group. The publishable group is `gust-gpu`, wgsl-prebuilt and wgsl-percall (the WGSL build); the
/// spirv group (the `spirv-route` build, D193) is `gust-gpu-spirv` with the hand WGSL compiled without wgpu's runtime checks
/// (`wgsl-prebuilt-unchecked`, the reference, and `wgsl-percall-unchecked`, the control); the ablation group is wgsl-percall and wgsl-prebuilt
/// plus the switch ablations, with no GUST backend.
#[cfg(feature = "gpu")]
fn gpu_backends(
    group: Group,
    device: &DeviceHandle,
    ctx: &Rc<WgpuContext>,
    ablation: &[Switch],
    all_on: bool,
    probes: &[(String, Probe)],
    variants: &[gust_gpu::Variant],
) -> Vec<Box<dyn Backend>> {
    let mut backends: Vec<Box<dyn Backend>> = match group {
        Group::Publishable => vec![
            Box::new(gust_gpu::GustGpu::new(DeviceHandle::clone(device))),
            Box::new(HandWgsl::new(Rc::clone(ctx), Mode::Prebuilt)),
            Box::new(HandWgsl::new(Rc::clone(ctx), Mode::PerCall)),
        ],
        Group::Spirv => vec![
            Box::new(gust_gpu::GustGpu::new(DeviceHandle::clone(device))),
            Box::new(HandWgsl::unchecked(Rc::clone(ctx), Mode::Prebuilt)),
            Box::new(HandWgsl::unchecked(Rc::clone(ctx), Mode::PerCall)),
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
    // The m10.b1c probes (`--with`): more hand backends in the publishable rotation, `gust-setup` and `gust-device` on a
    // device of their own set up like GUST's, `gust-shader` on the shared hand device. The kill rule
    // ignores them (it judges GUST against wgsl-prebuilt only). The pf.trust probes: `gust-unchecked` is GUST itself on a
    // second device opened with `DeviceOptions::unchecked_kernels` (its id follows the device, `gust-gpu-unchecked`), and
    // `hand-unchecked` is the hand text with the checks off (`wgsl-prebuilt-unchecked`), the reproduction control.
    // `wgsl-prebuilt-b` is the reference again on a device of its own, opened like the hand one.
    for (index, (name, probe)) in probes.iter().enumerate() {
        let backend: Box<dyn Backend> = match probe {
            Probe::Gust(unchecked) => {
                Box::new(gust_gpu::GustGpu::new(DeviceHandle::clone(unchecked)))
            }
            Probe::Hand(hand) if name == "hand-unchecked" => {
                Box::new(HandWgsl::unchecked(Rc::clone(hand), Mode::Prebuilt))
            }
            Probe::Hand(hand) => Box::new(HandWgsl::gust_probe(Rc::clone(hand), name)),
        };
        // `gust-unchecked` as the first item of `--with` is also first in the group, before `gust-gpu`: the order in
        // which a pair's buffers and pipelines are created matters on the Radeon 890M, so a series swaps it.
        if index == 0 && matches!(probe, Probe::Gust(_)) {
            backends.insert(0, backend);
        } else {
            backends.push(backend);
        }
    }
    // The W6-hand GUST rows (all five `--with` names of `Variant`) on GUST's own device, after the hand probes,
    // in the order given.
    for variant in variants {
        backends.push(Box::new(gust_gpu::GustGpu::with_variant(
            DeviceHandle::clone(device),
            *variant,
        )));
    }
    backends
}

/// The device a `--with` probe runs on: the shared or a private hand context, or a GUST device of its own.
#[cfg(feature = "gpu")]
enum Probe {
    Hand(Rc<WgpuContext>),
    Gust(DeviceHandle),
}

/// The second GUST device of `--with gust-unchecked`, on the adapter and with the limits of the run's first one, opened
/// with `DeviceOptions::unchecked_kernels` (pf.trust). The SPIR-V route has no wgpu checks to turn off, so the row would
/// repeat `gust-gpu-spirv` under another name: it is refused there.
#[cfg(feature = "gpu")]
fn open_unchecked_gust_device(
    selection: Option<&adapter::Selection>,
    adapter_limits: bool,
) -> BenchResult<DeviceHandle> {
    if spirv_route() {
        return Err("--with gust-unchecked needs the WGSL build: the SPIR-V route has no wgpu runtime checks to turn off".into());
    }
    let options = match (selection, adapter_limits) {
        (Some(selected), _) => selected.options(adapter_limits),
        (None, true) => DeviceOptions::default(),
        (None, false) => DeviceOptions::default()
            .with_limits(LimitsPolicy::Default)
            .with_features(FeaturesPolicy::Default),
    };
    // SAFETY: this device exists to time `gust-gpu-unchecked` against `gust-gpu`. The bench kernels read and write inside
    // their buffers (the host sizes every buffer for its kernel), divide by values the host keeps non-zero, and loop a
    // host-chosen number of times (the degree for W3, a constant elsewhere). Every sample is compared with the serial
    // reference, which catches a wrong result but not undefined behaviour (a hang, or a write into another buffer), so
    // this rests on the kernels above, not on the comparison. The row is for measurement; no published table rests on it.
    let options = unsafe { options.unchecked_kernels() };
    Ok(DeviceHandle::new(HeadlessDevice::with_options(options)?))
}

/// What `rust_gpu_backend` returns: the backend when it can run, and the backends left out with their reasons.
#[cfg(feature = "gpu")]
type RustGpuSetup = (Option<Box<dyn Backend>>, Vec<Value>);

/// The rust-gpu comparator `rustgpu-wgpu` (m10.b4), in the publishable group's rotation of a `rust-gpu` build: its own
/// device (the run's third) on the run's adapter selection, opened with wgpu's passthrough feature, only when the run
/// selects the backend (no `--backends` list, or one naming it). An adapter that lacks a capability the module needs
/// leaves the backend out with the reason (printed and recorded in the run file as `backends_not_run`), never a
/// fallback to another shader path; naming it with `--backends` then makes that an error, and any other failure to open
/// the device fails the run (`rustgpu_module::decide`).
#[cfg(all(feature = "gpu", feature = "rust-gpu"))]
fn rust_gpu_backend(
    options: &Options,
    group: Group,
    adapter: &wgpu::AdapterInfo,
) -> BenchResult<RustGpuSetup> {
    use rustgpu_module::{Decision, decide};
    if group != Group::Publishable {
        return Ok((None, Vec::new()));
    }
    if let Some(wanted) = &options.backends
        && !wanted.iter().any(|name| name == "rustgpu-wgpu")
    {
        return Ok((None, Vec::new()));
    }
    let opened = WgpuContext::open_passthrough_on(options.adapter.as_deref());
    match decide(opened, options.backends.as_deref())? {
        Decision::Run(ctx) => {
            adapter::require_same(&[("GUST device", adapter), ("rust-gpu device", &ctx.info)])?;
            Ok((Some(Box::new(HandWgsl::rust_gpu(Rc::new(ctx)))), Vec::new()))
        }
        Decision::NotRun(entry) => {
            eprintln!(
                "rustgpu-wgpu not run: {}",
                entry["reason"].as_str().unwrap_or_default()
            );
            Ok((None, vec![entry]))
        }
    }
}

/// A build without the `rust-gpu` feature has no such backend; naming it is an unknown backend, and nothing is recorded.
#[cfg(all(feature = "gpu", not(feature = "rust-gpu")))]
fn rust_gpu_backend(
    _options: &Options,
    _group: Group,
    _adapter: &wgpu::AdapterInfo,
) -> BenchResult<RustGpuSetup> {
    Ok((None, Vec::new()))
}

/// What `cubecl_backends` returns: the rows when they can run, the rows left out with their reasons, and the record of the
/// comparator for the run file's config (`Value::Null` when it did not run).
#[cfg(feature = "gpu")]
type CubeclSetup = (Vec<Box<dyn Backend>>, Vec<Value>, Value);

/// The CubeCL comparator's two rows (m10.b2), in the publishable group's rotation of a `cubecl` build: one runtime on a
/// device of its own (besides GUST's and the hand device) on the run's adapter selection, opened only when the run selects a CubeCL
/// row (no `--backends` list, or one naming one). An adapter CubeCL cannot run its kernels on leaves both rows out with the
/// reason (printed and recorded in the run file as `backends_not_run`), never another adapter; naming a row with
/// `--backends` then makes that an error, and any other failure to open the runtime fails the run
/// (`cubecl_backend::decide`).
#[cfg(all(feature = "gpu", feature = "cubecl"))]
fn cubecl_backends(
    options: &Options,
    group: Group,
    adapter: &wgpu::AdapterInfo,
    adapter_limits: bool,
) -> BenchResult<CubeclSetup> {
    use cubecl_backend::{Cubecl, CubeclContext, Decision, LIKE_FOR_LIKE, TUNED, decide};
    if group != Group::Publishable {
        return Ok((Vec::new(), Vec::new(), Value::Null));
    }
    if let Some(wanted) = &options.backends
        && !wanted
            .iter()
            .any(|name| name == LIKE_FOR_LIKE || name == TUNED)
    {
        return Ok((Vec::new(), Vec::new(), Value::Null));
    }
    let opened = CubeclContext::open_on(options.adapter.as_deref(), adapter_limits);
    match decide(opened, options.backends.as_deref())? {
        Decision::Run(ctx) => {
            adapter::require_same(&[("GUST device", adapter), ("CubeCL device", &ctx.info)])?;
            let record =
                json!({ "adapter": ctx.info.name, "identity": cubecl_backend::identity() });
            let ctx = Rc::new(ctx);
            Ok((
                vec![
                    Box::new(Cubecl::new(Rc::clone(&ctx), false)),
                    Box::new(Cubecl::new(ctx, true)),
                ],
                Vec::new(),
                record,
            ))
        }
        Decision::NotRun(entries) => {
            for entry in &entries {
                eprintln!(
                    "{} not run: {}",
                    entry["backend"].as_str().unwrap_or_default(),
                    entry["reason"].as_str().unwrap_or_default()
                );
            }
            Ok((Vec::new(), entries, Value::Null))
        }
    }
}

/// A build without the `cubecl` feature has no such backends; naming one is an unknown backend, and nothing is recorded.
#[cfg(all(feature = "gpu", not(feature = "cubecl")))]
fn cubecl_backends(
    _options: &Options,
    _group: Group,
    _adapter: &wgpu::AdapterInfo,
    _adapter_limits: bool,
) -> BenchResult<CubeclSetup> {
    Ok((Vec::new(), Vec::new(), Value::Null))
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
    /// `publishable`, `spirv` or `ablation`: the group every sample of the file came from.
    group: &'static str,
    /// The kill rule of the run's table group (`group::kill_rule`), null for the ablation group.
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
    /// `publishable` and its reasons (`judge::judge_run`), written with the banner of a run that is not.
    publishable: bool,
    publishable_reasons: Vec<String>,
    functional: bool,
    /// Start and end of the run by the harness's own clock (plan 2.4 item 5).
    started_unix_s: u64,
    ended_unix_s: u64,
    /// The git state at the start and at the end of the run (plan 2.4 item 1).
    git_start: meta::GitState,
    git_end: meta::GitState,
}

impl RunRecord {
    fn into_document(self) -> Value {
        let git = |state: &meta::GitState| {
            json!({
                "head": state.head, "tree_dirty": state.dirty,
                "origin_main": state.origin_main, "ancestor_of_origin_main": state.ancestor_of_origin_main,
            })
        };
        let mut document = json!({
            "label": self.label,
            "run_seed": self.run_seed,
            "group": self.group,
            "kill_rule": self.kill_rule,
            // Plan 2.4: a run the harness does not judge publishable carries the banner and the reasons.
            "banner": (!self.publishable).then_some(report::BANNER),
            "publishable": self.publishable,
            "publishable_reason": if self.publishable {
                "every check of plan 2.4 items 1 to 3 passed; the attestation (item 5) is the results commit's".to_owned()
            } else {
                self.publishable_reasons.join("; ")
            },
            "publishable_reasons": self.publishable_reasons,
            "functional": self.functional,
            "window": {
                "started_utc": meta::utc_text(self.started_unix_s),
                "ended_utc": meta::utc_text(self.ended_unix_s),
                "started_unix_s": self.started_unix_s,
                "ended_unix_s": self.ended_unix_s,
            },
            "source": { "start": git(&self.git_start), "end": git(&self.git_end) },
            "thresholds": judge::THRESHOLDS.json(),
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
    config.group = group;
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
        "group": config.group.name(),
        "sampling": format!("the group's gpu backends interleaved in rounds of seeded random order, paired against {}", config.group.reference()),
        "cold_only": config.cold_only,
        "cold_repeats": config.cold_repeats,
        "workloads": workload_list.iter().map(|w| w.id()).collect::<Vec<_>>(),
        "backends": backends.iter().map(|b| json!({ "id": b.id(), "description": b.description() })).collect::<Vec<_>>(),
        // The `--with` order is the creation order of the probe rows; `device_order` is the open order of the two GUST
        // devices (null without `gust-unchecked`).
        "with": options.with_probe,
        "device_order": options
            .with_probe
            .iter()
            .any(|p| p == "gust-unchecked")
            .then(|| options.device_order.unwrap_or_default().name()),
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
/// backend alone. `before_cell` is called with a label before each (the sampler's settle-and-window, retried until the
/// window is quiet) and returns the quiet record the following cells are stamped with, together with `blockers` (the
/// run-level reasons no cell is publishable), by `judge::stamp_cell`.
fn run_cells(
    backends: &mut [Box<dyn Backend>],
    workload_list: &[Workload],
    options: &Options,
    config: &Config,
    blockers: &[String],
    before_cell: &mut dyn FnMut(&str, Workload, usize) -> Value,
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
                let quiet = before_cell("gpu-group", *workload, *n);
                for mut cell in run_gpu_group(&mut gpu, &input, config) {
                    judge::stamp_cell(&mut cell.json, &quiet, blockers);
                    produced.push(cell);
                }
            }
            for backend in backends
                .iter_mut()
                .filter(|b| b.kind() != Kind::Gpu && b.supports(*workload))
            {
                let quiet = before_cell(backend.id(), *workload, *n);
                let mut cell = run_cell(backend.as_mut(), &input, config);
                judge::stamp_cell(&mut cell.json, &quiet, blockers);
                produced.push(cell);
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
                        format!(
                            ", paired vs {}: {}",
                            config.group.reference(),
                            paired.join(", ")
                        )
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

/// The kill rule of the run's table group over the cells, printed; null for the ablation group.
#[cfg(feature = "gpu")]
fn kill_rule_of(group: Group, cells: &[Value]) -> Value {
    if !group.is_table() {
        return Value::Null;
    }
    let record = group::kill_rule(group, cells);
    eprintln!(
        "kill rule ({} group): C re-opened = {}, reading step = {} (judged {}: {} paired GPU-time ratios checked against {}, {} unpaired, offenders {}); control {} outside the window, worst {}",
        record["group"],
        record["c_reopened"],
        record["reading_step"],
        record["judged"],
        record["checked"],
        record["reference"],
        record["unpaired"],
        record["offenders"],
        record["control"]["outside_tolerance"],
        record["control"]["worst_abs_deviation"]
    );
    record
}

/// What a run does before each cell: settle, sample and judge the window (plan 2.4 item 3), retrying up to `N`
/// times, when the sampler is on; otherwise a record that says the window was not judged.
struct Quiet<'a> {
    sampler: Option<&'a Sampler>,
    timing: sampler::Timing,
    /// Every window taken, with its attempt number and verdict: the sidecar.
    windows: Vec<Value>,
    why_off: &'static str,
}

impl Quiet<'_> {
    fn before_cell(&mut self, label: &str, workload: Workload, n: usize) -> Value {
        let Some(sampler) = self.sampler else {
            return judge::unjudged_record(self.why_off);
        };
        let (windows, record) =
            judge::settle_until_quiet(&judge::THRESHOLDS, self.timing.samples(), || {
                sampler.sample_cell(json!({
                    "backend": label, "workload": workload.id(), "n": n,
                }))
            });
        self.windows.extend(windows);
        if record["quiet"] != true {
            eprintln!(
                "not quiet: {label} {} n={n} after {} attempts: {}",
                workload.id(),
                record["attempts"],
                record["reasons"][0].as_str().unwrap_or("?")
            );
        }
        record
    }
}

/// Writes the sampler's sidecar beside `out` and returns the run file's `sampler` block.
fn write_sidecar(
    out: &str,
    sampler: Option<&Sampler>,
    windows: &[Value],
    timing: &sampler::Timing,
) -> BenchResult<Value> {
    let Some(sampler) = sampler else {
        return Ok(json!({ "enabled": false }));
    };
    let sidecar = sampler::sidecar_path(out);
    let document = json!({ "header": sampler.header(), "windows": windows });
    std::fs::write(&sidecar, serde_json::to_string(&document)? + "\n")?;
    Ok(json!({
        "enabled": true,
        // Only the file name: results files never carry a path.
        "sidecar": sidecar.file_name().and_then(|n| n.to_str()),
        "windows": windows.len(),
        "settle_s": timing.settle.as_secs_f64(), "window_s": timing.window.as_secs_f64(),
        "rate_hz": timing.rate_hz,
    }))
}

/// The sampling timing of a run: the plan's defaults, or the `--settle-secs` and `--window-secs` overrides.
fn run_timing(options: &Options) -> sampler::Timing {
    let mut timing = sampler::Timing::default();
    if let Some(secs) = options.settle_secs {
        timing.settle = Duration::from_secs_f64(secs);
    }
    if let Some(secs) = options.window_secs {
        timing.window = Duration::from_secs_f64(secs);
    }
    timing
}

/// The facts of the start of a run that decide whether any of its cells can be published (`judge::run_blockers`).
fn run_inputs(
    options: &Options,
    config: &Config,
    group: Group,
    sampler_on: bool,
    software_adapter: bool,
    git: &meta::GitState,
    timing: &sampler::Timing,
) -> judge::RunInputs {
    // Anything that changes the measured method (plan 2.3, 2.4 item 3) away from its defaults is named.
    let defaults = Config::default();
    let default_timing = sampler::Timing::default();
    let mut overrides = Vec::new();
    if timing.settle != default_timing.settle {
        overrides.push("--settle-secs".to_owned());
    }
    if timing.window != default_timing.window || timing.rate_hz != default_timing.rate_hz {
        overrides.push("--window-secs".to_owned());
    }
    if config.cap != defaults.cap && !options.quick {
        overrides.push("--cap-secs".to_owned());
    }
    if config.warmup_cap != defaults.warmup_cap {
        overrides.push("--warmup-secs".to_owned());
    }
    judge::RunInputs {
        overrides,
        build_head: meta::build_head(),
        build_dirty: meta::build_dirty(),
        functional: options.functional,
        quick: options.quick,
        cold_only: options.cold_only,
        sizes_overridden: options.sizes.is_some(),
        samples: config.samples,
        group_is_publishable: group.is_table(),
        release_build: !cfg!(debug_assertions),
        cargo_profile: meta::build_profile().to_owned(),
        tainting_features: meta::build_features()
            .into_iter()
            .filter(|feature| {
                matches!(
                    *feature,
                    "profile-counts" | "wgpu-trace" | "rust-gpu" | "cubecl"
                )
            })
            .map(str::to_owned)
            .collect(),
        probe_backends: !options.with_probe.is_empty(),
        sampler_enabled: sampler_on,
        software_adapter,
        head: git.head.clone(),
        tree_dirty: git.dirty,
        ancestor_of_origin_main: git.ancestor_of_origin_main,
    }
}

#[cfg(feature = "gpu")]
fn run(options: &Options) -> BenchResult<bool> {
    let out = options.out.as_deref().ok_or("--out is required")?;
    let started_unix_s = meta::unix_now();
    let git_start = meta::git_state();
    let (group, ablation_list) =
        resolve_group(options.group, options.ablation.as_deref(), spirv_route())?;
    let (ablation, all_on) = match &ablation_list {
        Some(list) => parse_ablation(list)?,
        None => (Vec::new(), false),
    };
    let workload_list = options
        .workloads
        .clone()
        .unwrap_or_else(|| Workload::ALL.to_vec());
    check_sizes(options, &workload_list)?;
    // The default wgpu limits cap a storage binding at 128 MiB, which a W7 input above 2^25 words exceeds: a run with such
    // a case opens both devices (GUST's and the hand one) with the adapter's own limits; every other run keeps the
    // portable defaults. The granted limits are recorded in the run file's machine block.
    let adapter_limits = workload_list.iter().any(|w| {
        sizes_for(options, *w)
            .iter()
            .any(|n| workloads::needs_adapter_limits(*w, *n))
    });
    // `--adapter NAME`: one adapter for both devices (zero or several matches are an error that lists the adapters).
    let selection = options
        .adapter
        .as_deref()
        .map(adapter::select)
        .transpose()?;
    // `--device-order`: the unchecked GUST device is opened before the checked one when asked, else after it (in the
    // probe loop below). Only a run that has the unchecked device can name an order.
    let wants_unchecked = options.with_probe.iter().any(|p| p == "gust-unchecked");
    if options.device_order.is_some() && !wants_unchecked {
        return Err("--device-order needs --with gust-unchecked".into());
    }
    let device_order = options.device_order.unwrap_or_default();
    let mut unchecked_device = None;
    if wants_unchecked && device_order == DeviceOrder::UncheckedFirst {
        unchecked_device = Some(open_unchecked_gust_device(
            selection.as_ref(),
            adapter_limits,
        )?);
    }
    let gust_device = DeviceHandle::new(match (&selection, adapter_limits) {
        (Some(selected), _) => HeadlessDevice::with_options(selected.options(adapter_limits))?,
        (None, true) => HeadlessDevice::with_options(DeviceOptions::default())?,
        (None, false) => HeadlessDevice::new()?,
    });
    let ctx = Rc::new(WgpuContext::open_on(
        options.adapter.as_deref(),
        adapter_limits,
    )?);
    let adapter = gust_device.adapter_info();
    // Prove, not assume, that every device of the run is on one adapter (the selection too, when there is one).
    let mut devices = vec![("GUST device", adapter), ("hand-WGSL device", &ctx.info)];
    if let Some(selected) = &selection {
        devices.push(("--adapter selection", &selected.info));
    }
    adapter::require_same(&devices)?;
    // A software adapter (lavapipe) runs only the functional check: its times are never published (plan 2.4).
    let software = matches!(adapter.device_type, wgpu::DeviceType::Cpu);
    if software && !options.functional {
        return Err(format!(
            "software adapter selected: {} (only `--functional` runs on one, and it publishes no timing)",
            adapter.name
        )
        .into());
    }
    // `nvidia-smi` describes only an NVIDIA adapter: on any other it would read a different GPU, so it is not queried.
    let smi_on = meta::limit_smi_to_adapter(adapter.vendor);
    if !smi_on {
        eprintln!(
            "nvidia-smi is not queried: the adapter's vendor id is {:#06x}, not NVIDIA's, so it would describe another GPU",
            adapter.vendor
        );
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());

    let config = build_config(options, group);
    check_probes_group(group, &options.with_probe)?;
    check_gust_only(options, group)?;
    let mut probes = Vec::new();
    let mut variants = Vec::new();
    for name in &options.with_probe {
        // The W6-hand GUST rows run on GUST's own device and open no hand context.
        if let Some(variant) = gust_gpu::Variant::parse(name) {
            variants.push(variant);
            continue;
        }
        // `gust-setup` and `gust-device` have a device of their own (set up like GUST's), `gust-unchecked` a GUST device
        // of its own opened with unchecked kernels, `wgsl-prebuilt-b` one set up like the hand device; the others
        // share the hand device.
        let probe = if name == "gust-unchecked" {
            Probe::Gust(match unchecked_device.take() {
                Some(opened) => opened,
                None => open_unchecked_gust_device(selection.as_ref(), adapter_limits)?,
            })
        } else if name == "wgsl-prebuilt-b" {
            Probe::Hand(Rc::new(WgpuContext::open_on(
                options.adapter.as_deref(),
                adapter_limits,
            )?))
        } else if name == "gust-setup" || name == "gust-device" {
            Probe::Hand(Rc::new(WgpuContext::open_like_gust_on(
                options.adapter.as_deref(),
            )?))
        } else {
            Probe::Hand(Rc::clone(&ctx))
        };
        let info = match &probe {
            Probe::Hand(hand) => &hand.info,
            Probe::Gust(unchecked) => unchecked.adapter_info(),
        };
        adapter::require_same(&[("GUST device", adapter), (name.as_str(), info)])?;
        probes.push((name.clone(), probe));
    }
    let mut backends = gpu_backends(
        group,
        &gust_device,
        &ctx,
        &ablation,
        all_on,
        &probes,
        &variants,
    );
    let (rust_gpu, mut backends_not_run) = rust_gpu_backend(options, group, adapter)?;
    backends.extend(rust_gpu);
    let (cubecl_rows, cubecl_not_run, cubecl_record) =
        cubecl_backends(options, group, adapter, adapter_limits)?;
    backends.extend(cubecl_rows);
    backends_not_run.extend(cubecl_not_run);
    if group.is_table() {
        backends.extend(cpu_backends(threads));
    }
    // A `--gust-only` run names GUST rows alone, which the full-set rule of a table group refuses by design.
    if group.is_table()
        && !options.gust_only
        && let Some(wanted) = &options.backends
    {
        group::require_full_set(group, wanted)?;
    }
    select_backends(&mut backends, options.backends.as_ref())?;

    let limits = gust_device.limits();
    let hand_limits = ctx.device.limits();
    let mut machine = meta::machine(
        adapter,
        gust_device.supports_gpu_timestamps(),
        threads,
        limits,
        &hand_limits,
    );
    // Which adapter the run asked for: the substring and the adapter it matched (null: the default choice, no flag).
    machine["adapter_selection"] = selection
        .as_ref()
        .map_or(Value::Null, adapter::Selection::record);
    machine["hand_wgsl_adapter_info"] = adapter::info_record(&ctx.info);
    eprintln!(
        "adapter: {} ({})",
        adapter.name,
        match &selection {
            Some(selected) => format!("--adapter {}, index {}", selected.substring, selected.index),
            None => "default choice".to_owned(),
        }
    );
    let load_before = meta::load_snapshot();
    let gpu_before = meta::gpu_snapshot();
    if smi_on {
        eprintln!("gpu before: {gpu_before}");
    }
    eprintln!("load before: {load_before}");
    let timing = run_timing(options);
    // A functional run checks correctness only: no window, no judgement.
    let sampler_on = !options.no_sampler && !options.functional;
    let sampler = sampler_on.then(|| {
        Sampler::open(
            timing,
            Some(AdapterIdent {
                vendor: adapter.vendor,
                device: adapter.device,
                name: &adapter.name,
            }),
        )
    });
    if let Some(sampler) = &sampler {
        eprintln!("sampler: {}", sampler.header()["luid_mapping"]);
    }
    let blockers = judge::run_blockers(&run_inputs(
        options, &config, group, sampler_on, software, &git_start, &timing,
    ));
    for blocker in &blockers {
        eprintln!("not publishable: {blocker}");
    }
    let mut quiet = Quiet {
        sampler: sampler.as_ref(),
        timing,
        windows: Vec::new(),
        why_off: if options.functional {
            "functional run: no sampler"
        } else {
            "no sampler (--no-sampler)"
        },
    };
    let started = Instant::now();

    let (cells, failures) = run_cells(
        &mut backends,
        &workload_list,
        options,
        &config,
        &blockers,
        &mut |label, workload, n| quiet.before_cell(label, workload, n),
    );
    let ended_unix_s = meta::unix_now();
    let git_end = meta::git_state();

    let sampler_json = write_sidecar(out, sampler.as_ref(), &quiet.windows, &timing)?;
    let kill_rule = if options.functional {
        Value::Null
    } else {
        kill_rule_of(group, &cells)
    };
    let source_changed = meta::source_change(&git_start, &git_end);
    let (mut publishable, mut publishable_reasons) =
        judge::judge_run(&blockers, &cells, &failures, source_changed.as_deref());
    if options.gust_only {
        publishable = false;
        publishable_reasons.push("--gust-only: GUST rows alone, no paired reference".to_owned());
    }
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
                "gust_only": options.gust_only,
                "hand_wgsl_adapter": ctx.info.name,
                // A backend of this build that was left out, with the reason (the rust-gpu comparator on an adapter
                // without passthrough shaders): "not run", never a fallback.
                "backends_not_run": backends_not_run,
                // The CubeCL comparator's adapter and what each of its rows is (null in a build without it, or when it
                // did not run).
                "cubecl": cubecl_record,
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
        provenance: meta::provenance(adapter, &git_start),
        publishable,
        publishable_reasons,
        functional: options.functional,
        started_unix_s,
        ended_unix_s,
        git_start,
        git_end,
    }
    .into_document();
    std::fs::write(out, serde_json::to_string_pretty(&document)? + "\n")?;
    for failure in &failures {
        eprintln!("FAILURE: {failure}");
    }
    eprintln!(
        "publishable: {} ({})",
        document["publishable"], document["publishable_reason"]
    );
    Ok(failures.is_empty())
}

/// The `GUST_PROFILE=cpu` build has no GPU runtime: only the CPU backends, one group. The sampler still runs between
/// cells (no adapter under test, so every adapter's activity counts: the stricter fallback), so the CPU rows are
/// judged like any others.
#[cfg(not(feature = "gpu"))]
fn run(options: &Options) -> BenchResult<bool> {
    let out = options.out.as_deref().ok_or("--out is required")?;
    let started_unix_s = meta::unix_now();
    let git_start = meta::git_state();
    let (group, _) = resolve_group(options.group, options.ablation.as_deref(), spirv_route())?;
    if options.adapter.is_some() {
        return Err("--adapter needs the default build: this build has no GPU runtime".into());
    }
    if options.gust_only {
        return Err("--gust-only needs the default build: this build has no GPU runtime".into());
    }
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
    let timing = run_timing(options);
    let sampler_on = !options.no_sampler && !options.functional;
    let sampler = sampler_on.then(|| Sampler::open(timing, None));
    let blockers = judge::run_blockers(&run_inputs(
        options, &config, group, sampler_on, false, &git_start, &timing,
    ));
    for blocker in &blockers {
        eprintln!("not publishable: {blocker}");
    }
    let mut quiet = Quiet {
        sampler: sampler.as_ref(),
        timing,
        windows: Vec::new(),
        why_off: if options.functional {
            "functional run: no sampler"
        } else {
            "no sampler (--no-sampler)"
        },
    };
    let started = Instant::now();
    let (cells, failures) = run_cells(
        &mut backends,
        &workload_list,
        options,
        &config,
        &blockers,
        &mut |label, workload, n| quiet.before_cell(label, workload, n),
    );
    let ended_unix_s = meta::unix_now();
    let git_end = meta::git_state();
    let sampler_json = write_sidecar(out, sampler.as_ref(), &quiet.windows, &timing)?;
    let source_changed = meta::source_change(&git_start, &git_end);
    let (publishable, publishable_reasons) =
        judge::judge_run(&blockers, &cells, &failures, source_changed.as_deref());
    let document = RunRecord {
        label: options.label.clone().unwrap_or_else(|| "run".to_owned()),
        run_seed: config.run_seed,
        group: group.name(),
        kill_rule: Value::Null,
        sampler: sampler_json,
        machine: meta::machine_cpu(threads),
        config: config_json(&config, options, &workload_list, &backends, json!({})),
        load_before,
        load_after: meta::load_snapshot(),
        gpu_before: Value::Null,
        gpu_after: Value::Null,
        wall_s: started.elapsed().as_secs_f64(),
        cells,
        failures: failures.clone(),
        provenance: meta::provenance_cpu(&git_start),
        publishable,
        publishable_reasons,
        functional: options.functional,
        started_unix_s,
        ended_unix_s,
        git_start,
        git_end,
    }
    .into_document();
    std::fs::write(out, serde_json::to_string_pretty(&document)? + "\n")?;
    for failure in &failures {
        eprintln!("FAILURE: {failure}");
    }
    eprintln!(
        "publishable: {} ({})",
        document["publishable"], document["publishable_reason"]
    );
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

/// `dump-kernels --out DIR`: for every benchmark kernel its Slang source, the WGSL and the SPIR-V that slangc makes of it
/// (`DIR/<kernel>.slang`, `.wgsl`, `.spv`). In a `spirv-route` build the Slang source is the one that route emits.
#[cfg(all(feature = "gpu", feature = "slang-runtime"))]
fn dump_kernels(options: &Options) -> BenchResult<()> {
    let dir = std::path::PathBuf::from(options.out.as_deref().ok_or("--out DIR is required")?);
    std::fs::create_dir_all(&dir)?;
    for descriptor in workloads::all_descriptors() {
        let stem = descriptor.qualified_name().to_string().replace("::", "_");
        std::fs::write(dir.join(format!("{stem}.slang")), descriptor.slang_source)?;
        std::fs::write(
            dir.join(format!("{stem}.wgsl")),
            gust::slang::compile_wgsl(descriptor)?,
        )?;
        let words = gust::slang::compile_spirv(descriptor)?;
        let bytes = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<u8>>();
        std::fs::write(dir.join(format!("{stem}.spv")), bytes)?;
        eprintln!("dumped {stem}");
    }
    Ok(())
}

#[cfg(not(all(feature = "gpu", feature = "slang-runtime")))]
fn dump_kernels(_options: &Options) -> BenchResult<()> {
    Err("dump-kernels needs a build with the gpu and slang-runtime features".into())
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
    let sampler = Sampler::open(
        sampler::Timing::default(),
        Some(AdapterIdent {
            vendor: adapter.vendor,
            device: adapter.device,
            name: &adapter.name,
        }),
    );
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

/// `profile`: host-side counts per call (needs a `profile-counts` build, which carries the counting allocator).
#[cfg(feature = "profile-counts")]
fn profile_command(options: &Options) -> BenchResult<bool> {
    profile::run(options)
}

#[cfg(not(feature = "profile-counts"))]
fn profile_command(_options: &Options) -> BenchResult<bool> {
    Err("profile needs a build with the profile-counts feature (it installs the counting allocator)".into())
}

#[cfg(feature = "wgpu-trace")]
fn trace_command(options: &Options) -> BenchResult<bool> {
    trace::run(options)
}

#[cfg(not(feature = "wgpu-trace"))]
fn trace_command(_options: &Options) -> BenchResult<bool> {
    Err("trace needs a build with the wgpu-trace feature".into())
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
    println!(
        "groups: publishable (default of the WGSL build), spirv (default of the spirv-route build), ablation"
    );
    println!("gpu runtime in this build: {}", cfg!(feature = "gpu"));
    println!("features: {:?}", meta::build_features());
    println!("GUST profile: {}", gust::profile::NAME);
    println!("cargo profile: {}", meta::build_profile());
    println!("gust_dual (CPU twin built in): {}", cfg!(gust_dual));
}

/// `--adapter` belongs to `run`: `profile` and `trace` open their own devices on the default adapter, so they refuse it
/// rather than ignore it.
fn adapter_only_for_run(command: &str, options: &Options) -> BenchResult<()> {
    if options.adapter.is_some() && command != "run" {
        return Err(format!(
            "--adapter is an option of `run`; `{command}` always uses the default adapter"
        )
        .into());
    }
    Ok(())
}

fn real_main() -> BenchResult<bool> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let Some((command, rest)) = args.split_first() else {
        return Err(USAGE.into());
    };
    let options = parse(rest)?;
    adapter_only_for_run(command, &options)?;
    match command.as_str() {
        "run" => run(&options),
        "aggregate" => {
            let json = options.json.as_deref().ok_or("--json is required")?;
            let md = options.md.as_deref().ok_or("--md is required")?;
            report::aggregate(&options.positional, json, md)?;
            Ok(true)
        }
        "dump-wgsl" => dump_wgsl().map(|()| true),
        "dump-kernels" => dump_kernels(&options).map(|()| true),
        "probe-adapters" => probe_adapters(),
        "profile" => profile_command(&options),
        "trace" => trace_command(&options),
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
            resolve_group(None, None, false).unwrap(),
            (Group::Publishable, None)
        );
        assert!(resolve_group(Some(Group::Publishable), Some("all"), false).is_err());
        assert!(resolve_group(None, Some("all-on"), false).is_err());
        assert!(resolve_group(Some(Group::Spirv), Some("all"), true).is_err());
        for route in [false, true] {
            assert_eq!(
                resolve_group(Some(Group::Ablation), None, route).unwrap(),
                (Group::Ablation, Some("all".to_owned()))
            );
            assert_eq!(
                resolve_group(Some(Group::Ablation), Some("all-on"), route).unwrap(),
                (Group::Ablation, Some("all-on".to_owned()))
            );
        }
    }

    /// D193: the SPIR-V route is judged in the `spirv` group and only there; the default group follows the build.
    #[test]
    fn the_spirv_route_build_runs_the_spirv_group_and_refuses_the_publishable_one() {
        assert_eq!(
            resolve_group(None, None, true).unwrap(),
            (Group::Spirv, None)
        );
        assert_eq!(
            resolve_group(Some(Group::Spirv), None, true).unwrap(),
            (Group::Spirv, None)
        );
        let refused = resolve_group(Some(Group::Publishable), None, true).unwrap_err();
        assert!(
            refused.contains("D193") && refused.contains("--group spirv"),
            "{refused}"
        );
        // The WGSL build has no SPIR-V route to judge.
        let refused = resolve_group(Some(Group::Spirv), None, false).unwrap_err();
        assert!(refused.contains("spirv-route"), "{refused}");
    }

    /// `--with` probes (hand-WGSL host-shape probes of the checked baseline) run in the publishable group only.
    #[cfg(feature = "gpu")]
    #[test]
    fn with_probes_are_refused_outside_the_publishable_group() {
        let probes = vec!["gust-shader".to_owned()];
        assert!(check_probes_group(Group::Publishable, &probes).is_ok());
        assert!(check_probes_group(Group::Publishable, &[]).is_ok());
        for group in [Group::Spirv, Group::Ablation] {
            let refused = check_probes_group(group, &probes).unwrap_err();
            assert!(
                refused.contains("--with") && refused.contains(group.name()),
                "{refused}"
            );
            assert!(check_probes_group(group, &[]).is_ok());
        }
    }

    #[test]
    fn the_device_order_parses_and_defaults_to_checked_first() {
        let args = |list: &[&str]| list.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        assert_eq!(parse(&[]).unwrap().device_order, None);
        assert_eq!(DeviceOrder::default(), DeviceOrder::CheckedFirst);
        let order = |text: &str| parse(&args(&["--device-order", text])).map(|o| o.device_order);
        assert_eq!(
            order("unchecked-first"),
            Ok(Some(DeviceOrder::UncheckedFirst))
        );
        assert_eq!(order("checked-first"), Ok(Some(DeviceOrder::CheckedFirst)));
        assert!(order("sideways").is_err());
    }

    #[test]
    fn the_spirv_group_parses_and_the_unchecked_probe_is_gone() {
        let args = |list: &[&str]| list.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        assert_eq!(
            parse(&args(&["--group", "spirv"])).unwrap().group,
            Some(Group::Spirv)
        );
        let Err(error) = parse(&args(&["--with", "unchecked"])) else {
            panic!("--with unchecked must be refused");
        };
        assert!(
            error.contains("--group spirv") && error.contains("D193"),
            "{error}"
        );
        assert!(parse(&args(&["--with", "gust-shader"])).is_ok());
        assert!(parse(&args(&["--with", "nope"])).is_err());
        // The W6-hand GUST rows keep the order given, which fixes their creation order.
        let with = parse(&args(&["--with", "gust-perpass,gust-fused,gust-default"])).unwrap();
        assert_eq!(
            with.with_probe,
            ["gust-perpass", "gust-fused", "gust-default"]
        );
        // The spike's names are gone.
        assert!(parse(&args(&["--with", "gust-onepass"])).is_err());
        assert!(parse(&args(&["--with", "gust-unfused"])).is_err());
        // The pf.trust probes keep the order they are given in: it is the creation order of the rows.
        let probes = parse(&args(&["--with", "hand-unchecked,gust-unchecked"])).unwrap();
        assert_eq!(probes.with_probe, ["hand-unchecked", "gust-unchecked"]);
    }

    /// `--gust-only` parses, is refused without `--backends` and with any row that is not GUST's, and lifts nothing else.
    #[cfg(feature = "gpu")]
    #[test]
    fn gust_only_names_gust_rows_alone() {
        let args = |list: &[&str]| list.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        assert!(!parse(&[]).unwrap().gust_only);
        let only = parse(&args(&["--gust-only", "--backends", "gust-gpu"])).unwrap();
        assert!(only.gust_only);
        assert!(check_gust_only(&only, Group::Publishable).is_ok());
        let none = parse(&args(&["--gust-only"])).unwrap();
        assert!(check_gust_only(&none, Group::Publishable).is_err());
        let mixed = parse(&args(&[
            "--gust-only",
            "--backends",
            "gust-gpu,wgsl-prebuilt",
        ]))
        .unwrap();
        assert!(check_gust_only(&mixed, Group::Publishable).is_err());
        let rows = parse(&args(&[
            "--gust-only",
            "--backends",
            "gust-gpu,gust-gpu+default,gust-gpu+perpass,gust-gpu+fused,gust-gpu+frame,gust-gpu+frame-perpass",
            "--with",
            "gust-default,gust-perpass,gust-fused,gust-frame,gust-frame-perpass",
        ]))
        .unwrap();
        assert!(check_gust_only(&rows, Group::Publishable).is_ok());
        let hand = parse(&args(&[
            "--gust-only",
            "--backends",
            "gust-gpu",
            "--with",
            "gust-shader",
        ]))
        .unwrap();
        assert!(check_gust_only(&hand, Group::Publishable).is_err());
        assert!(check_gust_only(&only, Group::Ablation).is_err());
        // Without the flag nothing is checked, and the full-set rule of a table group still applies.
        let plain = parse(&args(&["--backends", "gust-gpu"])).unwrap();
        assert!(check_gust_only(&plain, Group::Publishable).is_ok());
        assert!(group::require_full_set(Group::Publishable, &["gust-gpu".to_owned()]).is_err());
    }

    #[test]
    fn the_run_config_pairs_against_the_groups_reference() {
        let options = parse(&[]).unwrap();
        assert_eq!(
            build_config(&options, Group::Publishable).group.reference(),
            "wgsl-prebuilt"
        );
        assert_eq!(
            build_config(&options, Group::Spirv).group.reference(),
            "wgsl-prebuilt-unchecked"
        );
        assert_eq!(build_config(&options, Group::Spirv).group.name(), "spirv");
    }

    #[test]
    fn adapter_is_refused_outside_run() {
        let args = |list: &[&str]| list.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        let options = parse(&args(&["--adapter", "5090"])).unwrap();
        assert!(adapter_only_for_run("run", &options).is_ok());
        for command in ["profile", "trace"] {
            let error = adapter_only_for_run(command, &options)
                .unwrap_err()
                .to_string();
            assert!(error.contains(command), "{error}");
        }
        assert!(adapter_only_for_run("profile", &parse(&args(&[])).unwrap()).is_ok());
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
        // W6 is the two staged-pipeline forms; W8 is not a workload.
        let pipeline = parse(&args(&["--workloads", "W6"]))
            .unwrap()
            .workloads
            .unwrap();
        assert_eq!(
            pipeline,
            vec![Workload::PipelineGraph, Workload::PipelineHand]
        );
        assert!(parse(&args(&["--workloads", "W8"])).is_err());
        let sums = parse(&args(&["--workloads", "W7,W9"]))
            .unwrap()
            .workloads
            .unwrap();
        assert_eq!(
            sums,
            vec![Workload::SumU32, Workload::SumF32, Workload::MatMul]
        );
    }

    /// The method flags a run changes away from the defaults, as `run_inputs` names them, for a command line.
    fn overrides_of(list: &[&str]) -> Vec<String> {
        let args = list.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        let options = parse(&args).unwrap();
        let config = build_config(&options, Group::Publishable);
        let timing = run_timing(&options);
        let git = meta::GitState {
            head: Some("0123456789abcdef".to_owned()),
            dirty: Some(false),
            origin_main: None,
            ancestor_of_origin_main: Some(true),
        };
        run_inputs(
            &options,
            &config,
            Group::Publishable,
            true,
            false,
            &git,
            &timing,
        )
        .overrides
    }

    #[test]
    fn the_method_flags_map_to_the_overrides_the_judge_names() {
        assert!(overrides_of(&[]).is_empty());
        assert_eq!(overrides_of(&["--settle-secs", "1"]), ["--settle-secs"]);
        assert_eq!(overrides_of(&["--window-secs", "1"]), ["--window-secs"]);
        assert_eq!(overrides_of(&["--cap-secs", "3"]), ["--cap-secs"]);
        assert_eq!(overrides_of(&["--warmup-secs", "0"]), ["--warmup-secs"]);
        assert_eq!(
            overrides_of(&[
                "--settle-secs",
                "1",
                "--window-secs",
                "1",
                "--cap-secs",
                "3",
                "--warmup-secs",
                "0"
            ]),
            [
                "--settle-secs",
                "--window-secs",
                "--cap-secs",
                "--warmup-secs"
            ]
        );
        // The values of the defaults are not overrides.
        assert!(overrides_of(&["--warmup-secs", "1", "--cap-secs", "8"]).is_empty());
        // `--quick` sets its own cap, which is not an override; an explicit `--cap-secs` on top of it is not named either.
        assert!(overrides_of(&["--quick"]).is_empty());
        assert!(overrides_of(&["--quick", "--cap-secs", "5"]).is_empty());
    }

    /// `--sizes` is a matrix side for W9: an element count meant for the other workloads is refused beside it.
    #[cfg(feature = "gpu")]
    #[test]
    fn a_sizes_list_too_large_for_a_matrix_side_is_refused_beside_w9() {
        let mut options = parse(&["--out".to_owned(), "x.json".to_owned()]).unwrap();
        let both = [Workload::VectorAdd, Workload::MatMul];
        assert!(
            check_sizes(&options, &both).is_ok(),
            "no --sizes: the ladders"
        );
        options.sizes = Some(vec![1_048_576]);
        let error = check_sizes(&options, &both).unwrap_err();
        assert!(error.contains("matrix side"), "{error}");
        assert!(check_sizes(&options, &[Workload::VectorAdd]).is_ok());
        options.sizes = Some(vec![256, 4096]);
        assert!(check_sizes(&options, &both).is_ok());
    }

    #[test]
    fn functional_implies_quick_and_the_flag_is_parsed() {
        let args = |list: &[&str]| list.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        let options = parse(&args(&["--functional", "--out", "x.json"])).unwrap();
        assert!(options.functional);
        assert!(options.quick);
        assert!(!parse(&args(&["--out", "x.json"])).unwrap().functional);
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
            publishable: false,
            publishable_reasons: vec!["--quick run: reduced samples and sizes".to_owned()],
            functional: false,
            started_unix_s: 1_700_000_000,
            ended_unix_s: 1_700_000_060,
            git_start: meta::GitState::default(),
            git_end: meta::GitState::default(),
        }
        .into_document();
        assert_eq!(document["publishable"], false);
        assert_eq!(
            document["publishable_reason"],
            "--quick run: reduced samples and sizes"
        );
        assert_eq!(document["window"]["started_unix_s"], 1_700_000_000u64);
        assert_eq!(document["window"]["started_utc"], "2023-11-14T22:13:20Z");
        assert_eq!(document["functional"], false);
        assert_eq!(document["banner"], "shared machine, not publishable");
        assert_eq!(document["route"], "wgsl");
        assert_eq!(document["build"]["commit"], "abc");
        assert_eq!(document["adapter"]["name"], "A");
        assert_eq!(document["config"]["samples"], 5);
        assert_eq!(document["group"], "publishable");
        assert_eq!(document["kill_rule"]["c_reopened"], false);
    }
}
