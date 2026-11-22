//! The host model of W8's lifecycle, its checks and the note of the buffers each kernel binds.

use gust::math::Vec4;

use crate::kernels::kit;
use crate::setup::{cpu_exclusive, lanes, unit};

// ---- W8: a full step with a lifecycle (m11.kit4) ----

/// W8 particle capacities. The live count starts at the capacity over `W8_START_DIV`; every step retires about one
/// particle in sixteen (`kit::RETIRE`) and spawns the capacity over `W8_SPAWN_DIV` new ones, so the population climbs
/// to the capacity and then stays near it, with the spawns that do not fit dropped.
pub(crate) const W8_SIZES: [u32; 2] = [1 << 17, 1 << 20];
pub(crate) const W8_SPAWN_DIV: u32 = 16;
pub(crate) const W8_START_DIV: u32 = 2;

/// The "lowbias32" finaliser (Chris Wellons' hash-prospector, public domain), written again from
/// `examples/walker-life/src/app.rs` (`mix`), as is `lb_hash` (`hash`) after it.
pub(crate) fn lb_mix(x: u32) -> u32 {
    let mut h = x ^ (x >> 16);
    h = h.wrapping_mul(2_146_121_005);
    h ^= h >> 15;
    h = h.wrapping_mul(2_221_713_035);
    h ^ (h >> 16)
}

pub(crate) fn lb_hash(seed: u32, index: u32) -> u32 {
    lb_mix(lb_mix(seed) ^ index)
}

/// Whether slot `i` retires in step `step`: the host's form of `w8_retire`.
fn retires(step: u32, i: u32) -> bool {
    (lb_hash(lb_hash(kit::RETIRE_SEED, step), i) & 0xffff) < kit::RETIRE
}

/// Particle `k` spawned in step `step`: the host's form of `w8_spawn` (every operation is exact in `f32`).
fn spawn_particle(step: u32, k: u32, side: u32) -> Vec4 {
    let h = lb_hash(lb_hash(kit::SPAWN_SEED, step), k);
    let s = side as f32;
    Vec4::new(
        unit(lb_hash(h, 0)) * s,
        unit(lb_hash(h, 1)) * s,
        unit(lb_hash(h, 2)) * 2.0 - 1.0,
        unit(lb_hash(h, 3)) * 2.0 - 1.0,
    )
}

/// The host model's counts of one step: the live particles before it, the survivors, the live particles after it.
pub(crate) struct LifeStep {
    pub(crate) live: u32,
    pub(crate) surv: u32,
    pub(crate) next: u32,
}

/// The counts of steps 1 to `steps` from `start` live particles: every number is an integer function of the step and
/// the slot, so the host knows them without the GPU.
pub(crate) fn life_plan(cap: u32, spawn: u32, start: u32, steps: usize) -> Vec<LifeStep> {
    let mut live = start;
    let mut plan = Vec::with_capacity(steps);
    for step in 1..=steps as u32 {
        let inner = lb_mix(lb_hash(kit::RETIRE_SEED, step));
        let surv = (0..live)
            .filter(|&i| (lb_mix(inner ^ i) & 0xffff) >= kit::RETIRE)
            .count() as u32;
        let next = (surv + spawn).min(cap);
        plan.push(LifeStep { live, surv, next });
        live = next;
    }
    plan
}

/// The first index at which two integer slices differ, as a message part.
pub(crate) fn first_diff(got: &[u32], want: &[u32]) -> Option<String> {
    if got.len() != want.len() {
        return Some(format!("{} entries against {}", got.len(), want.len()));
    }
    got.iter()
        .zip(want)
        .position(|(g, w)| g != w)
        .map(|at| format!("entry {at}: GPU {}, host {}", got[at], want[at]))
}

/// What the GPU left after one W8 step: its own moved live particles (the ones the compaction read), the alive flags
/// and their exclusive scan over the capacity, and the second particle buffer.
pub(crate) struct LifeGpu<'a> {
    pub(crate) stepped: &'a [Vec4],
    pub(crate) alive: &'a [u32],
    pub(crate) astarts: &'a [u32],
    pub(crate) next: &'a [Vec4],
}

/// The lifecycle of one step against the host's model, all exact. The host recomputes the flags from the retire rule,
/// the scan from the flags, the survivors and live count from the scan and the plan, and the live set: the survivors
/// in slot order, then the spawned particles (those that fit), compared slot by slot, which also makes them equal as
/// a set.
pub(crate) fn check_life(
    step: u32,
    cap: u32,
    spawn: u32,
    side: u32,
    plan: &LifeStep,
    gpu: &LifeGpu<'_>,
) -> Result<(), String> {
    let LifeGpu {
        stepped,
        alive,
        astarts,
        next,
    } = *gpu;
    let host_alive: Vec<u32> = (0..cap)
        .map(|i| u32::from(i < plan.live && !retires(step, i)))
        .collect();
    if let Some(reason) = first_diff(alive, &host_alive) {
        return Err(format!("alive flags: {reason}"));
    }
    let host_starts = cpu_exclusive(&host_alive);
    if let Some(reason) = first_diff(astarts, &host_starts) {
        return Err(format!("scan of the alive flags: {reason}"));
    }
    let surv = astarts[cap as usize - 1] + alive[cap as usize - 1];
    if surv != plan.surv {
        return Err(format!(
            "survivors: GPU scan says {surv}, host model {}",
            plan.surv
        ));
    }
    // The live count after the step is `min(surv + spawn, cap)`, from the survivors just compared; the live set below
    // covers exactly that many slots (the spawns past the capacity are dropped).
    let live_next = plan.next;
    let mut expected: Vec<Vec4> = stepped
        .iter()
        .zip(&host_alive)
        .filter(|(_, a)| **a == 1)
        .map(|(q, _)| *q)
        .collect();
    for k in 0..spawn {
        if surv + k < cap {
            expected.push(spawn_particle(step, k, side));
        }
    }
    if let Some(at) = expected
        .iter()
        .zip(next)
        .position(|(e, g)| lanes(*e) != lanes(*g))
    {
        return Err(format!(
            "live set: slot {at} of {live_next}: GPU {:?}, host {:?}",
            lanes(next[at]),
            lanes(expected[at])
        ));
    }
    Ok(())
}

/// How many storage buffers a kernel binds, from its generated descriptor (uniform and invocation parameters are not
/// storage buffers).
pub(crate) fn storage_buffers(kernel: &gust::KernelDescriptor) -> usize {
    kernel
        .parameters
        .iter()
        .filter(|parameter| matches!(parameter.kind, gust::ParameterKind::Storage))
        .count()
}

/// The storage buffers each kernel of the W8 step binds, counted from the generated descriptors, and the largest.
pub(crate) fn w8_buffer_note(cells: u32, cap: u32) -> String {
    use gust_algo::scan_kernels as sk;
    use gust_algo::schedule::TileSchedule;
    use gust_algo::variant::{Operation, Variant, shipped};
    // The scan kernels of the shipped exclusive variant for these two lengths (gust-algo may still fall back to v0 on
    // a device with too little workgroup memory; that decision is inside the library and not visible here).
    let tiles = shipped(Operation::Exclusive) == Variant::V1
        && TileSchedule::new(cells).is_some()
        && TileSchedule::new(cap).is_some();
    let scan_kernels = if tiles {
        vec![
            (
                "scan v1 tiles exclusive",
                storage_buffers(&sk::scan_tiles_exclusive_u32::DESCRIPTOR),
            ),
            (
                "scan v1 tiles sums",
                storage_buffers(&sk::scan_tiles_sums_u32::DESCRIPTOR),
            ),
            (
                "scan v1 tiles top",
                storage_buffers(&sk::scan_tiles_top_u32::DESCRIPTOR),
            ),
            (
                "scan v1 add carry",
                storage_buffers(&sk::add_carry_u32::DESCRIPTOR),
            ),
        ]
    } else {
        vec![
            (
                "scan v0 chunks",
                storage_buffers(&sk::scan_chunks_u32::DESCRIPTOR),
            ),
            (
                "scan v0 sums",
                storage_buffers(&sk::scan_sums_u32::DESCRIPTOR),
            ),
            (
                "scan v0 top",
                storage_buffers(&sk::scan_top_u32::DESCRIPTOR),
            ),
            (
                "scan v0 add back sums",
                storage_buffers(&sk::add_back_sums_u32::DESCRIPTOR),
            ),
            (
                "scan v0 add back exclusive",
                storage_buffers(&sk::add_back_exclusive_u32::DESCRIPTOR),
            ),
        ]
    };
    let mut kernels = vec![
        ("w2_diffuse", storage_buffers(&kit::w2_diffuse::DESCRIPTOR)),
        ("w3_step", storage_buffers(&kit::w3_step::DESCRIPTOR)),
        ("w4_clear", storage_buffers(&kit::w4_clear::DESCRIPTOR)),
        ("w4_count", storage_buffers(&kit::w4_count::DESCRIPTOR)),
        ("w8_deposit", storage_buffers(&kit::w8_deposit::DESCRIPTOR)),
        ("w8_fold", storage_buffers(&kit::w8_fold::DESCRIPTOR)),
        ("w8_retire", storage_buffers(&kit::w8_retire::DESCRIPTOR)),
        ("w4_scatter", storage_buffers(&kit::w4_scatter::DESCRIPTOR)),
        ("w8_compact", storage_buffers(&kit::w8_compact::DESCRIPTOR)),
        ("w8_spawn", storage_buffers(&kit::w8_spawn::DESCRIPTOR)),
    ];
    kernels.extend(scan_kernels);
    let most = kernels.iter().map(|(_, n)| *n).max().unwrap_or(0);
    let list: Vec<String> = kernels
        .iter()
        .map(|(name, n)| format!("{name} {n}"))
        .collect();
    format!(
        "storage buffers bound per kernel (counted from each kernel's generated descriptor: the parameters of kind Storage, uniforms excluded): {}; the most in one kernel is {most}{}",
        list.join(", "),
        if most > 8 {
            " (OVER the 8 per stage GUST opens devices with)"
        } else {
            " (within the 8 per stage GUST opens devices with)"
        }
    )
}
