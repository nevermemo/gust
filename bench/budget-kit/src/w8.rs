//! W8: a full step with a lifecycle, and its variants W8a to W8d.

use std::time::Instant;

use gust::{Independent, math::Vec4};
use gust_algo::{dispatch_batch_in_scope, scan, timed};
use gust_wgpu::{BufferDispatch, DeviceHandle, GpuVec};

use crate::kernels::kit;
use crate::measure::{Measured, median};
use crate::report::{Row, skip_reason};
use crate::setup::{
    STEPS, W3_FIELD_SIDE, WARMUP, cpu_bin_counts, cpu_counts, cpu_diffuse, cpu_exclusive, dbg,
    extent, field_init, particle_init,
};
use crate::w4::check_scatter;
use crate::w7::{check_field, check_particles, percentile};
use crate::w8_model::{
    LifeGpu, W8_SPAWN_DIV, W8_START_DIV, check_life, first_diff, life_plan, storage_buffers,
    w8_buffer_note,
};

/// The variants of the W8 step (slice m11.cost): `Base` is W8, each other changes one thing.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Var {
    Base,
    /// No bins: no count, no clears of the counts and cursors, no scan of the counts and no scatter.
    NoBins,
    /// The bins are a `W8_COARSE_BINS` x `W8_COARSE_BINS` grid over the same field.
    CoarseBins,
    /// The step, the count and the deposit in one particle kernel.
    Fused,
    /// The lifecycle (retire, flag scan, compaction, spawn) on every `W8_LIFE_PERIOD`th step only.
    Sparse,
}

/// The bins per side of `Var::CoarseBins`, and the period of `Var::Sparse`.
const W8_COARSE_BINS: u32 = 256;
const W8_LIFE_PERIOD: usize = 4;

impl Var {
    pub(crate) fn workload(self) -> &'static str {
        match self {
            Self::Base => "W8 full step",
            Self::NoBins => "W8a no bins",
            Self::CoarseBins => "W8b bins 256x256",
            Self::Fused => "W8c fused step, count and deposit",
            Self::Sparse => "W8d lifecycle every 4th step",
        }
    }

    /// Steps per lifecycle: the lifecycle runs on every `period`th step.
    pub(crate) fn period(self) -> usize {
        if self == Self::Sparse {
            W8_LIFE_PERIOD
        } else {
            1
        }
    }

    pub(crate) fn bins(self) -> bool {
        self != Self::NoBins
    }

    /// What differs from W8, for the row's note (empty for W8 itself).
    pub(crate) fn describe(self) -> String {
        match self {
            Self::Base => String::new(),
            Self::NoBins => "variant a, no bins: the step has no w4_count, no clears of the counts and the cursors, no scan of the counts and no scatter, so it is three batches: (1) w2_diffuse, w3_step, the clear of the deposit grid, w8_deposit, w8_fold, w8_retire; (2) the scan of the alive flags; (3) w8_compact and w8_spawn; the checks of the counts, their scan and the scatter are dropped, the rest are W8's; the description below applies except where this says otherwise; ".to_owned(),
            Self::CoarseBins => format!(
                "variant b, the bins are a {W8_COARSE_BINS}x{W8_COARSE_BINS} grid over the same field: w8_bin_count and w8_bin_scatter bin a position scaled by {W8_COARSE_BINS}/{} (exact), so the counts, the cursors and their scan are {} cells; the field and the deposit grid stay {side}x{side}; the checks bin the GPU's own positions the same way on the host, the rest are W8's; storage buffers: w8_bin_count {}, w8_bin_scatter {}; the description below applies except where this says otherwise; ",
                W3_FIELD_SIDE,
                W8_COARSE_BINS * W8_COARSE_BINS,
                storage_buffers(&kit::w8_bin_count::DESCRIPTOR),
                storage_buffers(&kit::w8_bin_scatter::DESCRIPTOR),
                side = W3_FIELD_SIDE
            ),
            Self::Fused => format!(
                "variant c, w8_step_count_deposit does the step, the count and the deposit in one particle kernel (the same three effects, one read of each particle; storage buffers {}), so the first batch has no w3_step, w4_count or w8_deposit; the checks are W8's unchanged; the description below applies except where this says otherwise; ",
                storage_buffers(&kit::w8_step_count_deposit::DESCRIPTOR)
            ),
            Self::Sparse => format!(
                "variant d, the lifecycle (w8_retire, the scan of the alive flags, w8_compact, w8_spawn) runs on every {W8_LIFE_PERIOD}th step only; the steps between move, bin and deposit (three batches: the first without w8_retire, the scan of the counts, the scatter) and the live count changes on a lifecycle step only (the plan is per round of {W8_LIFE_PERIOD} steps and the retire draw uses the round number); the time column is the amortised mean GPU time per step over the counted steps, which are whole rounds from a round boundary (the warm-up is a whole number of rounds too); steps 1 and 2 (no lifecycle), the first clamped lifecycle step and the last step (a lifecycle step) are checked, a step without a lifecycle on everything but the lifecycle; the description below applies except where this says otherwise; "
            ),
        }
    }
}

/// One piece of a W8 step: a kernel (with the name the reports use) or a scan of one `u32` buffer into another.
enum Part<'a> {
    Kernel(&'static str, BufferDispatch<'a>),
    Scan(&'static str, &'a GpuVec<u32>, &'a GpuVec<u32>),
}

impl Part<'_> {
    fn name(&self) -> &'static str {
        match self {
            Self::Kernel(name, _) | Self::Scan(name, ..) => name,
        }
    }
}

/// The part of a kernel's call, or nothing for a call the variant does not make.
fn kernel_part<'a, C>(name: &'static str, call: &'a Option<C>) -> Option<Part<'a>>
where
    BufferDispatch<'a>: From<&'a C>,
{
    call.as_ref()
        .map(|call| Part::Kernel(name, BufferDispatch::from(call)))
}

/// How the parts of a step are grouped into batches (one batch and one timing scope each).
#[derive(Clone, Copy, PartialEq)]
enum Grain {
    /// A run of kernels is one batch and a scan is one of its own: the kit's step.
    Fused,
    /// Every kernel is a batch of its own, and every scan one.
    Kernel,
}

/// The ranges of `parts` that run as one batch each.
fn groups(parts: &[Part<'_>], grain: Grain) -> Vec<std::ops::Range<usize>> {
    let mut out: Vec<std::ops::Range<usize>> = Vec::new();
    for (at, part) in parts.iter().enumerate() {
        let joins = grain == Grain::Fused
            && matches!(part, Part::Kernel(..))
            && at > 0
            && matches!(parts[at - 1], Part::Kernel(..));
        match out.last_mut() {
            Some(last) if joins => last.end = at + 1,
            _ => out.push(at..at + 1),
        }
    }
    out
}

/// The dispatches of a group of kernels.
fn batch_of<'a>(group: &[Part<'a>]) -> Vec<BufferDispatch<'a>> {
    group
        .iter()
        .filter_map(|part| match part {
            Part::Kernel(_, dispatch) => Some(*dispatch),
            Part::Scan(..) => None,
        })
        .collect()
}

/// Runs one group: a scan, or a batch of kernels, through `dispatch_batch_in_scope`.
fn run_group(device: &DeviceHandle, group: &[Part<'_>]) -> Result<(), gust_algo::Error> {
    if let [Part::Scan(_, input, out)] = group {
        return scan::exclusive_into(input, out);
    }
    dispatch_batch_in_scope(device, &batch_of(group))?;
    Ok(())
}

/// A group's name in a report: the kernel, the scan, or the first and last kernel of a batch.
fn group_label(group: &[Part<'_>]) -> String {
    match (group.first(), group.last()) {
        (Some(first), Some(last)) if group.len() > 1 => {
            format!("{}..{}", first.name(), last.name())
        }
        (Some(only), _) => only.name().to_owned(),
        _ => String::new(),
    }
}

/// What a per-group pass finds: each group's name and median GPU time, and the median of the per-step sums.
type PassMedians = (Vec<(String, f64)>, f64);

/// The note text of a per-group pass: each group's median GPU time and its share of the sum of the medians, the sum,
/// and the median of the per-step sums.
fn pass_note(what: &str, pass: Option<&PassMedians>) -> String {
    let Some((groups, step_sums)) = pass else {
        return format!("{what}: n/a, this device has no GPU timestamps");
    };
    let total: f64 = groups.iter().map(|(_, ms)| ms).sum();
    let list: Vec<String> = groups
        .iter()
        .map(|(label, ms)| format!("{label} {ms:.4} ms ({:.1}%)", 100.0 * ms / total))
        .collect();
    format!(
        "{what}: {}; sum of the medians {total:.4} ms, median of the per-step sums {step_sums:.4} ms",
        list.join(", ")
    )
}

/// How W8 runs one step.
#[derive(Clone, Copy)]
enum Mode {
    /// Every batch and the scans inside one `timed` scope: the GPU time of the batches and the wall clock.
    Timed,
    /// The same dispatches with no scope: plain blocking dispatches, the wall clock only (used for the checked steps).
    Plain,
    /// As `Plain`, with the first batch sent through `submit_batch` (only the call is timed, then the job is waited
    /// for): the host cost of encoding and submitting it.
    Submit,
    /// Every group of the grain in a `timed` scope of its own: the GPU time of each.
    Split(Grain),
}

/// What one W8 step took.
struct StepTime {
    gpu_ms: Option<f64>,
    wall_ms: f64,
    submit_ms: Option<f64>,
    /// For `Mode::Split`: each group's name and GPU time.
    groups: Vec<(String, Option<f64>)>,
}

pub(crate) fn w8(device: &DeviceHandle, cap: u32, steps: usize, var: Var) -> Row {
    let side = W3_FIELD_SIDE;
    let cells = side * side;
    let bin_side = if var == Var::CoarseBins {
        W8_COARSE_BINS
    } else {
        side
    };
    let bin_cells = bin_side * bin_side;
    let period = var.period();
    // A whole number of lifecycle periods, so the counted steps start on a round boundary.
    let warm = WARMUP.next_multiple_of(period);
    let spawn = cap / W8_SPAWN_DIV;
    let start = cap / W8_START_DIV;
    let mut row = Row {
        workload: var.workload(),
        size: format!("{cap} particles capacity, {side}x{side} field"),
        result: Err("not run".into()),
        note: String::new(),
    };
    let fixed = format!(
        "{steps} counted steps after {warm} warm-up steps, in two runs of that length (the timed scope, then no scope with the first batch submitted through submit_batch); one step is four batches in one timed scope: (1) w2_diffuse on {side}x{side}, w3_step over the diffused field, w4_clear of the counts, the cursors and the deposit grid, w4_count, w8_deposit (atomic_add of {} into a u32 grid), w8_fold (the grid into lane w of the field), w8_retire (alive flags over the capacity); (2) gust-algo exclusive scan of the {cells} counts; (3) the same scan of the {cap} alive flags; (4) w4_scatter, w8_compact (survivors to slots given by the scan), w8_spawn ({spawn} new particles after the survivors, those past the capacity dropped); the population starts at {start} live particles, one in {} retires per step by a hash of (step, slot); the live counts come from a host model, not from a readback (a design that reads the count back each step pays W6's delay on top); checked after step 1, after step 2, on the first step where the survivors and the spawns do not all fit (the capacity clamp; the code requires that step to exist and says which it is in the note) and after the runs (one more step, checked on its own); each checked step re-poisons the scatter's `order` first: the field against the host (diffuse then fold of the GPU's previous field, 5e-6 relative as W2), the moved particles as W3, the deposit grid, the counts and their scan exactly, the scatter as sets per cell as W4, the alive flags, their scan and the survivors (the total of the scan) exactly against the host model, and the live set (survivors in slot order, then the spawned that fit; compared slot by slot, which fixes the live count)",
        kit::DEPOSIT,
        65_536 / kit::RETIRE
    );
    row.note = format!("{}{fixed}; {}", var.describe(), w8_buffer_note(cells, cap));
    row.result = (|| {
        if let Some(reason) = skip_reason(device, u64::from(cap.max(cells)) * 16) {
            return Err(reason);
        }
        // The counted steps must be whole lifecycle periods, so the mean covers whole rounds and the last step is a
        // lifecycle step.
        if !steps.is_multiple_of(period) {
            return Err(format!(
                "{steps} counted steps are not a whole number of {period}-step rounds"
            ));
        }
        // The round a step belongs to (a lifecycle period of steps), and whether the step ends it with the lifecycle.
        let round = |g: usize| (g - 1) / period + 1;
        let is_life = |g: usize| g.is_multiple_of(period);
        // The first round where the capacity clamps the population (survivors plus spawns past the capacity), found in
        // a plan long enough to reach it; its lifecycle step is checked on its own. Steps run: 1, 2 and the clamped
        // step checked, two runs of warm-up and counted steps, for W8 itself the two per-group passes, and one last
        // checked step.
        let clamp_round = life_plan(cap, spawn, start, 400)
            .iter()
            .position(|s| s.surv + spawn > cap)
            .map(|at| at + 1)
            .ok_or("no round of the first 400 reaches the capacity clamp")?;
        let clamp_at = clamp_round * period;
        let checked_to = clamp_at.max(2);
        let passes = if var == Var::Base {
            2 * (WARMUP + STEPS)
        } else {
            0
        };
        let total = checked_to + 2 * (warm + steps) + passes + period;
        let plan = life_plan(cap, spawn, start, round(total));
        if plan[clamp_round - 1].surv + spawn <= cap {
            return Err(format!("round {clamp_round} is not clamped"));
        }
        let live_of = |g: usize| plan[round(g) - 1].live;
        let field_host: Vec<Vec4> = (0..cells).map(field_init).collect();
        let parts_host: Vec<Vec4> = (0..cap).map(|i| particle_init(i, side)).collect();
        // The field alternates between two buffers each step; the particles between two buffers each round.
        let fields = [
            GpuVec::from_slice_on(device, &field_host).map_err(dbg)?,
            GpuVec::<Vec4>::zeroed_on(device, cells).map_err(dbg)?,
        ];
        let pbufs = [
            GpuVec::from_slice_on(device, &parts_host).map_err(dbg)?,
            GpuVec::from_slice_on(
                device,
                &vec![Vec4::new(-1.0, -1.0, -1.0, -1.0); cap as usize],
            )
            .map_err(dbg)?,
        ];
        // Non-zero to begin with, so a clear or a write that does nothing shows in the checks.
        let poisoned = |tag: u32, n: u32| -> Result<GpuVec<u32>, String> {
            let host: Vec<u32> = (0..n).map(|i| tag | i).collect();
            GpuVec::from_slice_on(device, &host).map_err(dbg)
        };
        // `Var::NoBins` binds none of the bin buffers (counts, cursor, starts, order), so they are one element there.
        let (bin_n, order_n) = if var.bins() { (bin_cells, cap) } else { (1, 1) };
        let counts = poisoned(0x8000_0000, bin_n)?;
        let cursor = poisoned(0x4000_0000, bin_n)?;
        let dep = poisoned(0x2000_0000, cells)?;
        let alive = poisoned(0x8000_0000, cap)?;
        let starts = GpuVec::<u32>::zeroed_on(device, bin_n).map_err(dbg)?;
        let astarts = GpuVec::<u32>::zeroed_on(device, cap).map_err(dbg)?;
        let order =
            GpuVec::from_slice_on(device, &vec![u32::MAX; order_n as usize]).map_err(dbg)?;
        let uniform = |label: &str, n: u32, side: u32| {
            device
                .create_uniform(
                    label,
                    &kit::Params {
                        n,
                        side,
                        pad0: 0,
                        pad1: 0,
                    },
                )
                .map_err(dbg)
        };
        let cell_params = uniform("w8 cells", cells, side)?;
        let bin_clear_params = uniform("w8 bin cells", bin_cells, bin_side)?;
        // One set of uniforms per step: the live count and the round change, so they are made before the runs.
        let part_params = (1..=total)
            .map(|g| uniform("w8 particles", live_of(g), side))
            .collect::<Result<Vec<_>, String>>()?;
        let life_params = (1..=total)
            .map(|g| {
                device
                    .create_uniform(
                        "w8 life",
                        &kit::LifeParams {
                            cap,
                            live: live_of(g),
                            step: round(g) as u32,
                            surv: plan[round(g) - 1].surv,
                            spawn,
                            side,
                            pad0: 0,
                            pad1: 0,
                        },
                    )
                    .map_err(dbg)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let bin_params = if var == Var::CoarseBins {
            (1..=total)
                .map(|g| {
                    device
                        .create_uniform(
                            "w8 bins",
                            &kit::BinParams {
                                n: live_of(g),
                                bins: bin_side,
                                field: side,
                                pad0: 0,
                            },
                        )
                        .map_err(dbg)
                })
                .collect::<Result<Vec<_>, String>>()?
        } else {
            Vec::new()
        };
        // Step `g` (1-based) diffuses `fields[(g - 1) % 2]` into the other one and moves the particles of
        // `pbufs[(round - 1) % 2]` in place; a lifecycle step compacts them into the other buffer. The calls are built
        // before the clock starts.
        let run_step = |g: usize, mode: Mode| -> Result<StepTime, String> {
            let (f_in, f_out) = (&fields[(g - 1) % 2], &fields[g % 2]);
            let cur = (round(g) - 1) % 2;
            let (p_cur, p_next) = (&pbufs[cur], &pbufs[1 - cur]);
            let (pp, lp) = (&part_params[g - 1], &life_params[g - 1]);
            let live = live_of(g);
            let life = is_life(g);
            let fused = var == Var::Fused;
            let coarse = var == Var::CoarseBins;
            let bins = var.bins();
            let diffuse = kit::w2_diffuse::call_extent(
                extent(cells),
                &cell_params,
                Independent(f_in),
                Independent(f_out),
            );
            let step = (!fused).then(|| {
                kit::w3_step::call_extent(extent(live), pp, Independent(f_out), Independent(p_cur))
            });
            let clear_counts = bins.then(|| {
                kit::w4_clear::call_extent(
                    extent(bin_cells),
                    &bin_clear_params,
                    Independent(&counts),
                )
            });
            let clear_cursor = bins.then(|| {
                kit::w4_clear::call_extent(
                    extent(bin_cells),
                    &bin_clear_params,
                    Independent(&cursor),
                )
            });
            let clear_dep =
                kit::w4_clear::call_extent(extent(cells), &cell_params, Independent(&dep));
            let count = (bins && !fused && !coarse).then(|| {
                kit::w4_count::call_extent(
                    extent(live),
                    pp,
                    Independent(p_cur),
                    Independent(&counts),
                )
            });
            let bin_count = coarse.then(|| {
                kit::w8_bin_count::call_extent(
                    extent(live),
                    &bin_params[g - 1],
                    Independent(p_cur),
                    Independent(&counts),
                )
            });
            let step_count_deposit = fused.then(|| {
                kit::w8_step_count_deposit::call_extent(
                    extent(live),
                    pp,
                    Independent(f_out),
                    Independent(p_cur),
                    Independent(&counts),
                    Independent(&dep),
                )
            });
            let deposit = (!fused).then(|| {
                kit::w8_deposit::call_extent(
                    extent(live),
                    pp,
                    Independent(p_cur),
                    Independent(&dep),
                )
            });
            let fold = kit::w8_fold::call_extent(
                extent(cells),
                &cell_params,
                Independent(&dep),
                Independent(f_out),
            );
            let retire =
                life.then(|| kit::w8_retire::call_extent(extent(cap), lp, Independent(&alive)));
            let scatter = (bins && !coarse).then(|| {
                kit::w4_scatter::call_extent(
                    extent(live),
                    pp,
                    Independent(p_cur),
                    Independent(&starts),
                    Independent(&cursor),
                    Independent(&order),
                )
            });
            let bin_scatter = coarse.then(|| {
                kit::w8_bin_scatter::call_extent(
                    extent(live),
                    &bin_params[g - 1],
                    Independent(p_cur),
                    Independent(&starts),
                    Independent(&cursor),
                    Independent(&order),
                )
            });
            let compact = life.then(|| {
                kit::w8_compact::call_extent(
                    extent(live),
                    lp,
                    Independent(p_cur),
                    Independent(&alive),
                    Independent(&astarts),
                    Independent(p_next),
                )
            });
            let spawn_call =
                life.then(|| kit::w8_spawn::call_extent(extent(spawn), lp, Independent(p_next)));
            let parts: Vec<Part<'_>> = [
                Some(Part::Kernel("diffuse", BufferDispatch::from(&diffuse))),
                kernel_part("step", &step),
                kernel_part("clear counts", &clear_counts),
                kernel_part("clear cursor", &clear_cursor),
                Some(Part::Kernel("clear dep", BufferDispatch::from(&clear_dep))),
                kernel_part("count", &count),
                kernel_part("bin count", &bin_count),
                kernel_part("step count deposit", &step_count_deposit),
                kernel_part("deposit", &deposit),
                Some(Part::Kernel("fold", BufferDispatch::from(&fold))),
                kernel_part("retire", &retire),
                bins.then_some(Part::Scan("scan counts", &counts, &starts)),
                life.then_some(Part::Scan("scan alive", &alive, &astarts)),
                kernel_part("scatter", &scatter),
                kernel_part("bin scatter", &bin_scatter),
                kernel_part("compact", &compact),
                kernel_part("spawn", &spawn_call),
            ]
            .into_iter()
            .flatten()
            .collect();
            let fused_groups = groups(&parts, Grain::Fused);
            let started = Instant::now();
            match mode {
                Mode::Timed => {
                    let scoped = timed(|| -> Result<(), gust_algo::Error> {
                        for range in &fused_groups {
                            run_group(device, &parts[range.clone()])?;
                        }
                        Ok(())
                    })
                    .map_err(dbg)?;
                    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
                    // W8 is four batches: the kernels, the scan of the counts, the scan of the alive flags, the kernels.
                    let expected = if var == Var::Base {
                        4
                    } else {
                        fused_groups.len()
                    };
                    if scoped.time.batches != expected {
                        return Err(format!(
                            "the timing scope counted {} batch(es), expected {expected} (the batches of kernels and the scans)",
                            scoped.time.batches
                        ));
                    }
                    Ok(StepTime {
                        gpu_ms: scoped.time.gpu_elapsed.map(|d| d.as_secs_f64() * 1e3),
                        wall_ms,
                        submit_ms: None,
                        groups: Vec::new(),
                    })
                }
                Mode::Plain => {
                    for range in &fused_groups {
                        run_group(device, &parts[range.clone()]).map_err(dbg)?;
                    }
                    Ok(StepTime {
                        gpu_ms: None,
                        wall_ms: started.elapsed().as_secs_f64() * 1e3,
                        submit_ms: None,
                        groups: Vec::new(),
                    })
                }
                Mode::Submit => {
                    // The batch is built before the clock of the submit span starts.
                    let first = batch_of(&parts[fused_groups[0].clone()]);
                    let started = Instant::now();
                    let job = device.submit_batch(&first).map_err(dbg)?;
                    let submit_ms = started.elapsed().as_secs_f64() * 1e3;
                    job.wait().map_err(dbg)?;
                    for range in &fused_groups[1..] {
                        run_group(device, &parts[range.clone()]).map_err(dbg)?;
                    }
                    Ok(StepTime {
                        gpu_ms: None,
                        wall_ms: started.elapsed().as_secs_f64() * 1e3,
                        submit_ms: Some(submit_ms),
                        groups: Vec::new(),
                    })
                }
                Mode::Split(grain) => {
                    let mut timed_groups = Vec::new();
                    for range in groups(&parts, grain) {
                        let group = &parts[range];
                        let scoped = timed(|| run_group(device, group)).map_err(dbg)?;
                        if scoped.time.batches != 1 {
                            return Err(format!(
                                "the timing scope of {} counted {} batch(es), expected 1",
                                group_label(group),
                                scoped.time.batches
                            ));
                        }
                        timed_groups.push((
                            group_label(group),
                            scoped.time.gpu_elapsed.map(|d| d.as_secs_f64() * 1e3),
                        ));
                    }
                    Ok(StepTime {
                        gpu_ms: None,
                        wall_ms: started.elapsed().as_secs_f64() * 1e3,
                        submit_ms: None,
                        groups: timed_groups,
                    })
                }
            }
        };
        let order_poison =
            GpuVec::from_slice_on(device, &vec![u32::MAX; order_n as usize]).map_err(dbg)?;
        let cap_params = uniform("w8 order", order_n, 0)?;
        let repoison = var.bins().then(|| {
            kit::w1_copy::call_extent(
                extent(order_n),
                &cap_params,
                Independent(&order_poison),
                Independent(&order),
            )
        });
        // One checked step: the GPU's state before it is read back, the step runs, and everything it produced is
        // compared with the host (the field and the moved particles from the GPU's own previous state, as W7 does).
        let verify = |g: usize| -> Result<(), String> {
            let live = live_of(g) as usize;
            let cur = (round(g) - 1) % 2;
            let prev_field = fields[(g - 1) % 2].to_vec().map_err(dbg)?;
            let prev_parts = pbufs[cur].to_vec().map_err(dbg)?;
            // The scatter's `order` is poisoned again (a copy of `u32::MAX`s, outside any timed scope), so a slot this
            // step does not write shows in the scatter check.
            if let Some(repoison) = &repoison {
                device
                    .dispatch_batch(&[BufferDispatch::from(repoison)])
                    .map_err(dbg)?;
            }
            run_step(g, Mode::Plain)?;
            let at_step = |e: String| format!("step {g}: {e}");
            let gpu_field = fields[g % 2].to_vec().map_err(dbg)?;
            let moved_all = pbufs[cur].to_vec().map_err(dbg)?;
            let stepped = &moved_all[..live];
            check_particles(stepped, &prev_parts[..live], &gpu_field, side).map_err(at_step)?;
            // The deposit grid: every live particle's amount in its cell, exactly.
            let dep_host: Vec<u32> = cpu_counts(stepped, side)
                .iter()
                .map(|c| c * kit::DEPOSIT)
                .collect();
            if let Some(reason) = first_diff(&dep.to_vec().map_err(dbg)?, &dep_host) {
                return Err(at_step(format!("deposit grid: {reason}")));
            }
            // The field: the host's diffusion of the GPU's previous field, then the deposit folded into lane w.
            let folded: Vec<Vec4> = cpu_diffuse(&prev_field, side)
                .iter()
                .zip(&dep_host)
                .map(|(v, d)| Vec4::new(v.x, v.y, v.z, v.w + *d as f32 / 1024.0))
                .collect();
            check_field(&gpu_field, &folded).map_err(at_step)?;
            // The bins: counts and their scan exactly, the scatter as sets per cell.
            if var.bins() {
                let counts_host = cpu_bin_counts(stepped, bin_side, side);
                if let Some(reason) = first_diff(&counts.to_vec().map_err(dbg)?, &counts_host) {
                    return Err(at_step(format!("counts: {reason}")));
                }
                let starts_host = cpu_exclusive(&counts_host);
                if let Some(reason) = first_diff(&starts.to_vec().map_err(dbg)?, &starts_host) {
                    return Err(at_step(format!("scan of the counts: {reason}")));
                }
                check_scatter(
                    &order.to_vec().map_err(dbg)?[..live],
                    &starts_host,
                    &counts_host,
                    stepped,
                    (bin_side, side),
                )
                .map_err(|reason| at_step(format!("scatter: {reason}")))?;
            }
            // The lifecycle.
            if is_life(g) {
                let (alive_gpu, astarts_gpu) =
                    (alive.to_vec().map_err(dbg)?, astarts.to_vec().map_err(dbg)?);
                let next_gpu = pbufs[1 - cur].to_vec().map_err(dbg)?;
                check_life(
                    round(g) as u32,
                    cap,
                    spawn,
                    side,
                    &plan[round(g) - 1],
                    &LifeGpu {
                        stepped,
                        alive: &alive_gpu,
                        astarts: &astarts_gpu,
                        next: &next_gpu,
                    },
                )
                .map_err(at_step)?;
            }
            Ok(())
        };
        verify(1)?;
        verify(2)?;
        let mut g = 2usize;
        // Plain steps up to the clamped one, which is checked.
        while g + 1 < clamp_at {
            g += 1;
            run_step(g, Mode::Plain)?;
        }
        if clamp_at > 2 {
            g += 1;
            verify(g).map_err(|e| format!("clamped step, {e}"))?;
        }
        // Run 1: the timed scope, the GPU time and the wall clock of every counted step.
        for _ in 0..warm {
            g += 1;
            run_step(g, Mode::Timed)?;
        }
        let mut blocking = Vec::with_capacity(steps);
        for _ in 0..steps {
            g += 1;
            blocking.push(run_step(g, Mode::Timed)?);
        }
        // Run 2: no scope; the first batch is submitted (the call timed, the wait outside), the rest blocks.
        for _ in 0..warm {
            g += 1;
            run_step(g, Mode::Submit)?;
        }
        let mut plain = Vec::with_capacity(steps);
        for _ in 0..steps {
            g += 1;
            plain.push(run_step(g, Mode::Submit)?);
        }
        // The per-group passes (W8 itself): each batch, then each kernel, in a timed scope of its own, the median of
        // `STEPS` steps after `WARMUP`; `None` where the device has no GPU timestamps.
        let pass = |grain: Grain, g: &mut usize| -> Result<Option<PassMedians>, String> {
            for _ in 0..WARMUP {
                *g += 1;
                run_step(*g, Mode::Split(grain))?;
            }
            let mut labels = Vec::new();
            let mut series: Vec<Vec<f64>> = Vec::new();
            let mut sums = Vec::new();
            let mut timestamps = true;
            for at in 0..STEPS {
                *g += 1;
                let step = run_step(*g, Mode::Split(grain))?;
                if at == 0 {
                    labels = step.groups.iter().map(|(label, _)| label.clone()).collect();
                    series = vec![Vec::new(); labels.len()];
                }
                let times: Vec<Option<f64>> = step.groups.iter().map(|(_, ms)| *ms).collect();
                match times.iter().copied().sum::<Option<f64>>() {
                    Some(sum) => sums.push(sum),
                    None => timestamps = false,
                }
                for (values, ms) in series.iter_mut().zip(times) {
                    values.extend(ms);
                }
            }
            Ok(timestamps.then(|| {
                (
                    labels
                        .into_iter()
                        .zip(&series)
                        .map(|(label, values)| (label, median(values)))
                        .collect(),
                    median(&sums),
                )
            }))
        };
        let passes_note = if var == Var::Base {
            let batches = pass(Grain::Fused, &mut g)?;
            let kernels = pass(Grain::Kernel, &mut g)?;
            format!(
                "; {}; {}; the sum of single-kernel batches may differ from the fused batches (a batch of kernels is one compute pass with one timestamp bracket; a one-kernel batch is a pass of its own)",
                pass_note(
                    &format!(
                        "per-batch GPU time (a separate pass after the sustained runs: each of the four batches in a timed scope of its own, median of {STEPS} steps after {WARMUP} warm-up steps)"
                    ),
                    batches.as_ref()
                ),
                pass_note(
                    &format!(
                        "per-kernel GPU time (a separate pass: every kernel a one-kernel batch in a timed scope of its own, each scan one scope, median of {STEPS} steps after {WARMUP} warm-up steps)"
                    ),
                    kernels.as_ref()
                )
            )
        } else {
            String::new()
        };
        // After the sustained runs: steps to the last, which is checked as the first two were, from whatever state the
        // runs left.
        while g + 1 < total {
            g += 1;
            run_step(g, Mode::Plain)?;
        }
        g += 1;
        if g != total {
            return Err(format!("ran {g} steps, planned {total}"));
        }
        verify(g).map_err(|e| format!("after the sustained runs, {e}"))?;
        let host_a: Vec<f64> = blocking.iter().map(|s| s.wall_ms).collect();
        let host_b: Vec<f64> = plain.iter().map(|s| s.wall_ms).collect();
        let submit_ms: Vec<f64> = plain.iter().filter_map(|s| s.submit_ms).collect();
        let (gpu_series, method) = if blocking.iter().all(|s| s.gpu_ms.is_some()) {
            (
                blocking
                    .iter()
                    .filter_map(|s| s.gpu_ms)
                    .collect::<Vec<f64>>(),
                "gpu timestamps",
            )
        } else {
            (host_a.clone(), "wall clock")
        };
        // The counted steps start on a round boundary, so every `period`th one is a lifecycle step. A variant with a
        // sparse lifecycle reports the mean per step (the amortised cost); the others the median.
        let (stat, ms) = if period > 1 {
            (
                "amortised mean",
                gpu_series.iter().sum::<f64>() / steps as f64,
            )
        } else {
            ("median", median(&gpu_series))
        };
        let measured = Measured { ms, method };
        let sparse_note = if period > 1 {
            let (life, rest): (Vec<_>, Vec<_>) = gpu_series
                .iter()
                .enumerate()
                .partition(|(at, _)| (at + 1) % period == 0);
            let medians = |part: &[(usize, &f64)]| {
                median(&part.iter().map(|(_, ms)| **ms).collect::<Vec<f64>>())
            };
            format!(
                "; median of the lifecycle steps {:.4} ms, median of the other steps {:.4} ms",
                medians(&life),
                medians(&rest)
            )
        } else {
            String::new()
        };
        let live_last = plan[round(total) - 1].next;
        row.note = format!(
            "the time column is the {stat} {method} time{}, p95 {:.4} ms, max {:.4} ms{sparse_note}; host (a) host_elapsed-style wall clock around the whole timed step (blocking batches with timestamp readback): median {:.4} ms, p95 {:.4} ms; host (b) wall clock around the whole step of the second run (the first batch through submit_batch and waited for, the scans and the other batches blocking, no timestamps): median {:.4} ms, p95 {:.4} ms; host (c) wall time for submit_batch of the first batch to return (encode and submit, no wait; the job is waited for after the span; measured with an idle queue, so it shows no back-pressure): median {:.4} ms, p95 {:.4} ms (the scans block, so unlike W7 the whole step cannot be submitted without a wait); the capacity clamp is checked at step {clamp_at}; live particles at the last step {live_last} of {cap}{passes_note}; {}",
            if method == "gpu timestamps" {
                " (the sum of the batches' GPU time; the gaps between the batches are not in it)"
            } else {
                " (the wall clock around the whole timed step: this device has no GPU timestamps)"
            },
            percentile(&gpu_series, 0.95),
            gpu_series.iter().copied().fold(f64::MIN, f64::max),
            median(&host_a),
            percentile(&host_a, 0.95),
            median(&host_b),
            percentile(&host_b, 0.95),
            median(&submit_ms),
            percentile(&submit_ms, 0.95),
            row.note
        );
        Ok((measured, None))
    })();
    row
}
