//! Component pool, indirect workload, and GPU active-count proofs (T08, T13).
//!
//! A `GpuPool<Particle>` grows across frames while GPU-authored state survives
//! each growth (a GPU-to-GPU copy, never a readback). Every frame runs a graph:
//! `classify` atomically counts the live particles that have spawned by the
//! current frame into a one-element count buffer; `prepare_dispatch` derives the
//! active count and indirect workgroup arguments on the GPU from that
//! GPU-authored count, clamped to the allocation and a settings budget;
//! `integrate` then runs as an indirect dispatch over exactly that many
//! particles. The host never learns the dispatched count before the work is
//! submitted. The kernels are typed calls, the graph edges are declared
//! (classify -> prepare_dispatch -> integrate; the graph checks them and never
//! infers or reorders them), and the graph is rebuilt after growth.
//!
//! Particles spawn in slot order (an emitter releasing `emit_per_frame` per
//! frame), so the active particles always form a prefix of the pool and the
//! indirect consumer's `i < active` guard covers exactly them. There is no
//! compaction stage yet (that is T14): the prefix invariant is what makes the
//! count usable as a dispatch bound.

use std::{path::PathBuf, time::Instant};

use gust::{Independent, KernelDescriptor, gpu};
#[cfg(test)]
use gust_wgpu::BufferBinding;
use gust_wgpu::{
    BufferDispatch, GpuBufferAccess, GpuPool, GraphReport, HeadlessDevice, StagedGraph,
    render_wgpu_source,
};

#[gpu]
mod pool {
    #[derive(Debug, PartialEq)]
    pub struct Particle {
        pub x: float,
        pub y: float,
        pub vx: float,
        pub vy: float,
        pub steps: uint,
        /// Frame at which the emitter releases this particle.
        pub spawn_frame: uint,
    }

    #[derive(Debug, PartialEq)]
    pub struct Settings {
        pub dt: float,
        pub gravity: float,
        /// Upper bound on particles integrated per frame.
        pub budget: uint,
        /// Current frame; live particles with `spawn_frame <= frame` are active.
        pub frame: uint,
    }

    #[derive(Debug, PartialEq)]
    pub struct DispatchArgs {
        pub x: uint,
        pub y: uint,
        pub z: uint,
    }

    #[derive(Debug, PartialEq)]
    pub struct Active {
        pub count: uint,
    }

    fn min_uint(a: uint, b: uint) -> uint {
        if a < b { a } else { b }
    }

    // One invocation per pool slot: count the live particles that have spawned by
    // this frame. `live` mirrors the pool's logical length; slots at or beyond it
    // hold unspecified bytes and are never classified. The count buffer is only
    // touched atomically here; the host zeroes it through an ordered graph upload.
    #[allow(clippy::len_zero)]
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn classify(
        id: SV_DispatchThreadID,
        live: StructuredBuffer<uint>,
        settings: StructuredBuffer<Settings>,
        particles: StructuredBuffer<Particle>,
        mut count: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if 0u32 < live.len()
            && 0u32 < settings.len()
            && 0u32 < count.len()
            && i < live[0u32]
            && i < particles.len()
            && particles[i].spawn_frame <= settings[0u32].frame
        {
            atomic_add(&mut count[0u32], 1u32);
        }
    }

    // One invocation: derive the active count and the indirect arguments on the
    // GPU. `count` is the GPU-authored classification result; it is clamped to the
    // allocation (`particles.len()`) so a malformed count can never dispatch past
    // the buffer, and to the settings budget.
    #[allow(clippy::len_zero)]
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn prepare_dispatch(
        id: SV_DispatchThreadID,
        count: StructuredBuffer<uint>,
        settings: StructuredBuffer<Settings>,
        particles: StructuredBuffer<Particle>,
        mut active: RWStructuredBuffer<Active>,
        mut args: RWStructuredBuffer<DispatchArgs>,
    ) {
        if id.x == 0u32 && 0u32 < count.len() && 0u32 < settings.len() {
            let live = min_uint(count[0u32], particles.len());
            let n = min_uint(live, settings[0u32].budget);
            active[0u32].count = n;
            // Overflow-free ceiling division: `n + 63` could wrap for a huge `n`.
            args[0u32].x = n / 64u32 + min_uint(n % 64u32, 1u32);
            args[0u32].y = 1u32;
            args[0u32].z = 1u32;
        }
    }

    // Indirect dispatch: the workgroup count came from `args`, so the guard uses
    // the GPU-derived active count and the allocation length, never a host count.
    #[allow(clippy::len_zero, clippy::assign_op_pattern)]
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn integrate(
        id: SV_DispatchThreadID,
        active: StructuredBuffer<Active>,
        settings: StructuredBuffer<Settings>,
        mut particles: RWStructuredBuffer<Particle>,
    ) {
        let i = id.x;
        if 0u32 < active.len()
            && 0u32 < settings.len()
            && i < active[0u32].count
            && i < particles.len()
        {
            let s = settings[0u32];
            let mut p = particles[i];
            p.vy = p.vy + s.gravity * s.dt;
            p.x = p.x + p.vx * s.dt;
            p.y = p.y + p.vy * s.dt;
            p.steps = p.steps + 1u32;
            particles[i] = p;
        }
    }
}

pub use pool::{Active, DispatchArgs, Particle, Settings};

/// Particles integrated per frame at most. The indirect dispatch needs `BUDGET / 64` = 16 384 workgroups,
/// within the 65 535 workgroups per dimension of the default limits; wgpu would silently zero a larger
/// indirect dispatch, and the runtime cannot check it, so the kernel's clamp to the budget is what holds.
const BUDGET: u32 = 1 << 20;

/// Spawn `count` particles for slots `first..`, released by the emitter in slot order at `emit_per_frame` particles
/// per frame, so `spawn_frame` never decreases along the pool: the particles spawned by any frame are a prefix.
pub fn spawn(first: usize, count: usize, emit_per_frame: usize) -> Vec<Particle> {
    let emit_per_frame = emit_per_frame.max(1);
    (first..first + count)
        .map(|index| Particle {
            x: (index % 17) as f32 * 0.5,
            y: (index % 13) as f32 * 0.25,
            vx: 1.0 + (index % 5) as f32 * 0.1,
            vy: (index % 7) as f32 * 0.2 - 0.6,
            steps: 0,
            spawn_frame: (index / emit_per_frame) as u32,
        })
        .collect()
}

/// Independent host reference for the classification: the particles spawned by `settings.frame`.
pub fn count_spawned(particles: &[Particle], settings: &Settings) -> usize {
    particles
        .iter()
        .filter(|particle| particle.spawn_frame <= settings.frame)
        .count()
}

/// Independent host reference for one frame: the spawned count, clamped to the budget, selects how many slots
/// integrate from the front of the pool (the prefix invariant: the spawned particles are the first ones).
pub fn step_cpu(particles: &mut [Particle], settings: &Settings) -> usize {
    let spawned = count_spawned(particles, settings);
    assert!(
        particles[..spawned]
            .iter()
            .all(|particle| particle.spawn_frame <= settings.frame),
        "the spawned particles must form a prefix of the pool (no compaction yet)"
    );
    let active = spawned.min(settings.budget as usize);
    for p in &mut particles[..active] {
        p.vy += settings.gravity * settings.dt;
        p.x += p.vx * settings.dt;
        p.y += p.vy * settings.dt;
        p.steps += 1;
    }
    active
}

pub struct Frame {
    pub settings: gust_wgpu::GpuBuffer<Settings>,
    /// GPU-authored classification count (one element); zeroed by the graph before `classify`.
    pub count: gust_wgpu::GpuBuffer<u32>,
    pub active: gust_wgpu::GpuBuffer<Active>,
    pub args: gust_wgpu::GpuBuffer<DispatchArgs>,
}

impl Frame {
    pub fn new(device: &HeadlessDevice) -> Result<Self, gust_wgpu::Error> {
        Ok(Self {
            settings: device.create_typed_buffer(
                "pool settings",
                &[Settings {
                    dt: 0.0,
                    gravity: 0.0,
                    budget: 0,
                    frame: 0,
                }],
                GpuBufferAccess::ReadOnly,
            ),
            count: device.create_typed_buffer(
                "pool active count",
                &[0u32],
                GpuBufferAccess::ReadWrite,
            ),
            active: device.create_typed_buffer(
                "pool active",
                &[Active { count: 0 }],
                GpuBufferAccess::ReadWrite,
            ),
            args: device.create_indirect_buffer(
                "pool dispatch args",
                &[DispatchArgs { x: 0, y: 0, z: 0 }],
            )?,
        })
    }
}

/// [`run_frame_traced`] without the edges.
#[cfg(test)]
pub fn run_frame(
    device: &HeadlessDevice,
    frame: &Frame,
    particles: &GpuPool<Particle>,
    settings: &[Settings; 1],
) -> Result<(u32, GraphReport), gust_wgpu::Error> {
    let (count, report, _edges) = run_frame_traced(device, frame, particles, settings)?;
    Ok((count, report))
}

/// One frame: settings write and count reset → classify → prepare_dispatch → indirect integrate → read the
/// GPU-derived active count back (the only readback; particles stay resident). Returns the count, the report,
/// and the dependency edges the graph holds, per node (node indices, in declaration order).
///
/// The graph is rebuilt every frame: growth replaces the pool's allocation, and the typed calls (locals
/// built before the graph, which borrows them) bind the allocation the pool holds right now. Every edge is
/// declared (`upload`, `dispatch`, `dispatch_indirect`, `readback` take their dependencies): the graph
/// checks them against the buffers the nodes touch and refuses a missing one, and it neither infers nor
/// reorders any. The classification dispatch is sized by the host-tracked pool length, never by a value
/// read back from the GPU; the readback of `active` waits for `integrate`, which follows its writer
/// `prepare_dispatch`.
pub fn run_frame_traced(
    device: &HeadlessDevice,
    frame: &Frame,
    particles: &GpuPool<Particle>,
    settings: &[Settings; 1],
) -> Result<(u32, GraphReport, Vec<Vec<usize>>), gust_wgpu::Error> {
    // Every buffer is bound `Independent`: none has the dispatch's element count as its length (the
    // pool's allocation is its capacity, the count has one element), and an indirect dispatch has no
    // host element count at all.
    let classify_call = pool::classify::call(
        particles.len() as u32,
        Independent(particles.count_buffer()),
        Independent(&frame.settings),
        Independent(particles.buffer()),
        Independent(&frame.count),
    );
    let prepare_call = pool::prepare_dispatch::call(
        1,
        Independent(&frame.count),
        Independent(&frame.settings),
        Independent(particles.buffer()),
        Independent(&frame.active),
        Independent(&frame.args),
    );
    let integrate_call = pool::integrate::call(
        0,
        Independent(&frame.active),
        Independent(&frame.settings),
        Independent(particles.buffer()),
    );
    let zero = [0u32];

    let mut graph = StagedGraph::new();
    let write = graph.upload(&frame.settings, settings, &[]);
    let reset = graph.upload(&frame.count, &zero, &[]);
    let classify = graph.dispatch(BufferDispatch::from(&classify_call), &[write, reset]);
    let prepare = graph.dispatch(BufferDispatch::from(&prepare_call), &[classify]);
    let integrate = graph.dispatch_indirect(
        BufferDispatch::from(&integrate_call),
        &frame.args,
        &[prepare],
    );
    let active = graph.readback(&frame.active, &[integrate]);
    let edges = [write, reset, classify, prepare, integrate, active]
        .map(|node| graph.dependency_indices(node))
        .to_vec();
    let output = device.execute_graph(&graph)?;
    let count = output.readback::<Active>(active)?[0].count;
    Ok((count, output.report, edges))
}

pub fn descriptors() -> [&'static KernelDescriptor; 3] {
    [
        &pool::classify::DESCRIPTOR,
        &pool::prepare_dispatch::DESCRIPTOR,
        &pool::integrate::DESCRIPTOR,
    ]
}

pub fn assert_close(actual: &[Particle], expected: &[Particle]) {
    assert_eq!(actual.len(), expected.len());
    let close = |a: f32, b: f32| (a - b).abs() <= 1.0e-4 * b.abs().max(1.0);
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            close(actual.x, expected.x)
                && close(actual.y, expected.y)
                && close(actual.vx, expected.vx)
                && close(actual.vy, expected.vy),
            "particle[{index}]: {actual:?} vs {expected:?}"
        );
        assert_eq!(actual.steps, expected.steps, "steps[{index}]");
    }
}

fn export_artifacts() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../generated-wgpu");
    std::fs::create_dir_all(&directory)?;
    for descriptor in descriptors() {
        let stem = format!("{}__{}", descriptor.module, descriptor.name);
        std::fs::write(
            directory.join(format!("{stem}.slang")),
            descriptor.slang_source,
        )?;
        std::fs::write(
            directory.join(format!("{stem}.wgsl")),
            gust::slang::compile_wgsl(descriptor)?,
        )?;
        std::fs::write(
            directory.join(format!("{stem}.spv")),
            gust::spirv::words_as_le_bytes(&gust::slang::compile_spirv(descriptor)?),
        )?;
        std::fs::write(
            directory.join(format!("{stem}.rs")),
            render_wgpu_source(descriptor)?,
        )?;
    }
    Ok(directory.canonicalize().unwrap_or(directory))
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = HeadlessDevice::new()?;
    let frame = Frame::new(&device)?;

    let initial = if cfg!(debug_assertions) {
        1 << 12
    } else {
        1 << 16
    };
    // The emitter releases `initial` particles per frame; the pool is grown ahead of it, so later frames
    // hold particles that have not spawned yet.
    let emit_per_frame = initial;
    let mut reference = spawn(0, initial, emit_per_frame);
    let mut particles = device.create_pool("particles", &reference, initial);
    println!(
        "Component pool: start {initial} particles, capacity {}, emitter {emit_per_frame}/frame",
        particles.capacity()
    );

    let frames = 6;
    let mut total_copied = 0;
    let mut last_edges = Vec::new();
    for frame_index in 0..frames {
        let settings = Settings {
            dt: 0.016,
            gravity: -9.8,
            budget: BUDGET,
            frame: frame_index,
        };
        let live = particles.len();
        let started = Instant::now();
        let (active, report, edges) = run_frame_traced(&device, &frame, &particles, &[settings])?;
        last_edges = edges;
        let frame_elapsed = started.elapsed();
        let cpu_started = Instant::now();
        let expected_active = step_cpu(&mut reference, &settings);
        let cpu_elapsed = cpu_started.elapsed();
        assert_eq!(active as usize, expected_active);

        // Grow after each frame: GPU-authored positions must survive the copy.
        let spawned = spawn(reference.len(), reference.len(), emit_per_frame);
        let growth = device.pool_push(&mut particles, &spawned)?;
        reference.extend(spawned);
        if let Some(growth) = growth {
            total_copied += growth.copied_bytes;
        }
        let reclaimed = device.pool_reclaim(&mut particles)?;
        println!(
            "frame {frame_index}: active {active} of {live} (GPU-classified), graph {frame_elapsed:?} (CPU reference {cpu_elapsed:?}), upload {} B, readback {} B, resident {} B, growth {growth:?}, reclaimed {reclaimed}",
            report.upload_bytes, report.readback_bytes, report.resident_bytes
        );
    }

    let actual = device.pool_read(&particles)?;
    assert_close(&actual, &reference);
    let artifacts = export_artifacts()?;
    println!(
        "final: {} particles, capacity {}, generation {}, {} growths, {} bytes copied GPU-to-GPU, {} retired allocations pending",
        particles.len(),
        particles.capacity(),
        particles.generation(),
        particles.growth_history().len(),
        total_copied,
        particles.retired_allocations()
    );
    println!("adapter: {}", device.adapter_info().name);
    println!(
        "declared dependencies per node (upload settings, reset count, classify, prepare_dispatch, integrate, \
         read; derived: none): {last_edges:?}"
    );
    println!("pipeline cache: {:?}", device.pipeline_cache_stats());
    println!("generated artifacts: {}", artifacts.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use gust::cpu::{CpuDevice, Error as CpuError};

    use super::*;
    use gust_wgpu::Error;

    /// The label of every test below that runs a kernel's CPU twin. The graph, its edges, the indirect launch that consumes the arguments and
    /// the pool growth are not executed by the twin: they stay GPU-only (docs/SEMANTICS.md, section 1). The twin of
    /// `prepare_dispatch` does compute the arguments, which the ignored test compares with the GPU's.
    const LABEL: &str = "kernel-level twin evidence; graph and indirect dispatch GPU-only";

    /// `2^-24`.
    const U: f64 = 1.0 / 16_777_216.0;

    /// The twin-versus-GPU bounds of the float outputs of `integrate`, absolute, in units of `U`. The twin is fed the
    /// particles, the settings and the active count that the GPU frame started from (the pool's allocation read back
    /// before the frame, the count and indirect arguments read back after it), so a bound covers one frame's own
    /// roundings only.
    ///
    /// Derived from the kernel text (docs/SEMANTICS.md section 4, "Deriving a tolerance"), not from a measurement.
    /// With the domain asserted by `assert_derivation_domain` and `assert_state_domain` (`0 < dt <= 0.5`,
    /// `abs(gravity * dt) <= 1`, and for each particle the frame steps `abs(x) <= 16`, `abs(y) <= 16`,
    /// `abs(vx) <= 1.5`, `abs(vy) <= 4`), one rounding of at most `u` times the result's magnitude on the twin and
    /// `2u` on the GPU for `+ - *`, first-order propagation, magnitudes `M`, twin error `T` and GPU error `G` in units
    /// of `u`, the GPU side without fusion:
    ///
    /// `vy` = `vy + gravity * dt`, `abs(vy) <= 5` afterwards:
    ///
    /// | step | M | T | G |
    /// | --- | --- | --- | --- |
    /// | `gravity * dt` | 1 | 1 | 2 |
    /// | `vy + ...` | 5 | 6 | 12 |
    ///
    /// Twin plus GPU: 18u, plus 1u for the second-order terms is 19u.
    ///
    /// `x` = `x + vx * dt`:
    ///
    /// | step | M | T | G |
    /// | --- | --- | --- | --- |
    /// | `vx * dt` | 0.75 | 0.75 | 1.5 |
    /// | `x + ...` | 16.75 | 17.5 | 35 |
    ///
    /// Twin plus GPU: 52.5u gives 53u, plus 1u is 54u.
    ///
    /// `y` = `y + vy * dt` uses the new `vy`, whose error (T 6, G 12) enters scaled by `dt <= 0.5`:
    ///
    /// | step | M | T | G |
    /// | --- | --- | --- | --- |
    /// | `vy * dt`, carrying the error of `vy` | 2.5 | 3 + 2.5 | 6 + 5 |
    /// | `y + ...` | 18.5 | 24 | 48 |
    ///
    /// Twin plus GPU: 72u, plus 1u is 73u.
    ///
    /// `vx`, `steps` and every field of a particle that is not stepped are copies or integer counts and must be equal.
    /// `prepare_dispatch` is integer-only (`min`, `/`, `%`, `+` on `u32`) and exact: the test compares the active
    /// count and the three indirect arguments with the twin and with a host formula.
    ///
    /// Observed maxima are printed separately by the ignored test and never become the tolerance.
    const VY_BOUND: f64 = 19.0 * U;
    const X_BOUND: f64 = 54.0 * U;
    const Y_BOUND: f64 = 73.0 * U;

    /// Everything spawned; only the budget can clamp.
    fn settings(budget: u32) -> Settings {
        settings_at(u32::MAX, budget)
    }

    fn settings_at(frame: u32, budget: u32) -> Settings {
        Settings {
            dt: 0.5,
            gravity: -2.0,
            budget,
            frame,
        }
    }

    /// A pool with `live` particles plus three spawned-then-retired tail slots (one particle per frame), so a
    /// classification that ignored the live length would over-count. Returns the pool and the live reference.
    fn pool_with_retired_tail(
        device: &HeadlessDevice,
        live: usize,
    ) -> (GpuPool<Particle>, Vec<Particle>) {
        let mut reference = spawn(0, live + 3, 1);
        let mut pool = device.create_pool("particles", &reference, reference.len());
        device.pool_truncate(&mut pool, live).unwrap();
        reference.truncate(live);
        (pool, reference)
    }

    /// Workgroup width of `integrate`, which the indirect arguments count in.
    fn width() -> u32 {
        pool::integrate::DESCRIPTOR.workgroup_size[0]
    }

    fn blank_active() -> Active {
        Active { count: u32::MAX }
    }

    fn blank_args() -> DispatchArgs {
        DispatchArgs {
            x: u32::MAX,
            y: u32::MAX,
            z: u32::MAX,
        }
    }

    /// The independent host formula for `prepare_dispatch`: the live count clamped to the allocation and the budget,
    /// and the workgroups that cover it.
    fn expected_prepare(count: u32, capacity: usize, budget: u32) -> (Active, DispatchArgs) {
        let active = count.min(capacity as u32).min(budget);
        (
            Active { count: active },
            DispatchArgs {
                x: active.div_ceil(width()),
                y: 1,
                z: 1,
            },
        )
    }

    /// The domain the bounds are derived for.
    fn assert_derivation_domain(settings: &Settings) {
        assert!(
            settings.dt > 0.0 && settings.dt <= 0.5,
            "dt {}",
            settings.dt
        );
        assert!(
            (f64::from(settings.gravity) * f64::from(settings.dt)).abs() <= 1.0,
            "gravity * dt"
        );
    }

    /// The particles a frame steps (the first `stepped`) stay inside the domain of the bounds.
    fn assert_state_domain(particles: &[Particle], stepped: usize) {
        for particle in &particles[..stepped] {
            assert!(particle.x.abs() <= 16.0, "x {}", particle.x);
            assert!(particle.y.abs() <= 16.0, "y {}", particle.y);
            assert!(particle.vx.abs() <= 1.5, "vx {}", particle.vx);
            assert!(particle.vy.abs() <= 4.0, "vy {}", particle.vy);
        }
    }

    /// `prepare_dispatch` on the twin. The two outputs are seeded with values no result equals, so a kernel that
    /// writes nothing fails. `allocation` is the pool's whole allocation: the kernel reads only its length.
    fn prepare_on_cpu(
        count: u32,
        settings: &Settings,
        allocation: &[Particle],
    ) -> Result<(Active, DispatchArgs), CpuError> {
        let device = CpuDevice::new();
        let count_buffer = device.buffer(&[count]);
        let settings_buffer = device.buffer(&[*settings]);
        let particles_buffer = device.buffer(allocation);
        let active = device.buffer(&[blank_active()]);
        let args = device.buffer(&[blank_args()]);
        pool::prepare_dispatch::dispatch(
            &device,
            1,
            Independent(&count_buffer),
            Independent(&settings_buffer),
            Independent(&particles_buffer),
            Independent(&active),
            Independent(&args),
        )?;
        Ok((active.read()[0], args.read()[0]))
    }

    /// `integrate` on the twin over `groups` workgroups of the given `active` count and `allocation`. The indirect
    /// dispatch is GPU-only: the twin is launched with the invocations the arguments of the frame ask for.
    fn integrate_on_cpu(
        active: Active,
        groups: u32,
        settings: &Settings,
        allocation: &[Particle],
    ) -> Result<Vec<Particle>, CpuError> {
        let device = CpuDevice::new();
        let active_buffer = device.buffer(&[active]);
        let settings_buffer = device.buffer(&[*settings]);
        let particles_buffer = device.buffer(allocation);
        pool::integrate::dispatch(
            &device,
            groups * width(),
            Independent(&active_buffer),
            Independent(&settings_buffer),
            Independent(&particles_buffer),
        )?;
        Ok(particles_buffer.read())
    }

    #[test]
    fn a_nan_in_a_compared_value_fails_the_bound_checks() {
        // `fold(0.0, f64::max)` returned 0 for a NaN difference, so NaN output passed a bound.
        let finite = [1.0f32, 2.0, 3.0];
        let nan_second = [1.0f32, f32::NAN, 3.0];
        let nan_first = [f32::NAN, 2.0, 3.0];
        assert_eq!(largest_abs(finite.into_iter(), finite.into_iter()), 0.0);
        assert_eq!(
            largest_abs(finite.into_iter(), [1.5f32, 2.0, 0.0].into_iter()),
            3.0
        );
        for (left, right) in [
            (finite, nan_second),
            (nan_second, finite),
            (nan_first, finite),
        ] {
            let largest = largest_abs(left.into_iter(), right.into_iter());
            assert!(largest.is_nan(), "{largest}");
            let passes = largest <= 1.0e9;
            assert!(!passes, "a NaN must not pass the bound");
        }
    }

    /// The largest absolute difference (computed in `f64`) between two float sequences.
    fn largest_abs(left: impl Iterator<Item = f32>, right: impl Iterator<Item = f32>) -> f64 {
        left.zip(right)
            .map(|(l, r)| (f64::from(l) - f64::from(r)).abs())
            .fold(0.0, |max: f64, diff| {
                // `f64::max` would return the other operand for a NaN and turn it into 0; a NaN stays NaN here,
                // so `largest <= bound` fails.
                if max.is_nan() || diff.is_nan() {
                    f64::NAN
                } else {
                    max.max(diff)
                }
            })
    }

    #[test]
    fn the_derivation_domain_holds_for_the_settings_and_the_spawned_particles() {
        for (dt, gravity) in [(0.016f32, -9.8f32), (0.5, -2.0)] {
            assert_derivation_domain(&Settings {
                dt,
                gravity,
                budget: 0,
                frame: 0,
            });
        }
        assert_state_domain(&spawn(0, 4096, 1), 4096);
    }

    /// The independent host reference for the emitter: with one particle per frame, the particles spawned by frame
    /// `f` are exactly slots `0..=f`, a prefix, and `step_cpu` steps exactly that prefix.
    #[test]
    fn the_emitter_reference_spawns_a_prefix() {
        for emit_per_frame in [1usize, 3, 64] {
            let pool = spawn(0, 200, emit_per_frame);
            assert!(
                pool.windows(2)
                    .all(|pair| pair[0].spawn_frame <= pair[1].spawn_frame)
            );
            for frame in [0u32, 1, 5, 66, u32::MAX] {
                let settings = settings_at(frame, 1000);
                let expected = (frame as usize)
                    .saturating_add(1)
                    .saturating_mul(emit_per_frame)
                    .min(pool.len());
                assert_eq!(
                    count_spawned(&pool, &settings),
                    expected,
                    "{emit_per_frame}/frame, frame {frame}"
                );
                let mut stepped = pool.clone();
                assert_eq!(step_cpu(&mut stepped, &settings), expected);
                assert!(
                    stepped[..expected]
                        .iter()
                        .all(|particle| particle.steps == 1)
                );
                assert!(
                    stepped[expected..]
                        .iter()
                        .all(|particle| particle.steps == 0)
                );
            }
        }
    }

    /// `classify` on the twin. `live` is the pool's logical length, `allocation` the whole allocation and `initial`
    /// the value the count buffer starts from (the graph zeroes it on the GPU; the twin needs no reset to be seeded).
    fn classify_on_cpu(
        live: u32,
        settings: &Settings,
        allocation: &[Particle],
        initial: u32,
        invocations: u32,
    ) -> Result<u32, CpuError> {
        let device = CpuDevice::new();
        let live_buffer = device.buffer(&[live]);
        let settings_buffer = device.buffer(&[*settings]);
        let particles_buffer = device.buffer(allocation);
        let count = device.buffer(&[initial]);
        pool::classify::dispatch(
            &device,
            invocations,
            Independent(&live_buffer),
            Independent(&settings_buffer),
            Independent(&particles_buffer),
            Independent(&count),
        )?;
        Ok(count.read()[0])
    }

    /// Kernel-level twin evidence; graph and indirect dispatch GPU-only (`LABEL`). The twin of `classify` counts the
    /// spawned live slots, never the retired tail, at the boundaries of the workgroup width, and agrees with the
    /// independent host reference `count_spawned`. The count is integer-only and exact.
    #[test]
    fn twin_classify_counts_the_spawned_live_slots() {
        let group = pool::classify::DESCRIPTOR.workgroup_size[0];
        for live in [0usize, 1, 63, 64, 65, 257] {
            let allocation = spawn(0, live + 3, 1);
            let reference = &allocation[..live];
            for frame in [
                0u32,
                live as u32 / 2,
                (live as u32).saturating_sub(1),
                u32::MAX,
            ] {
                let settings = settings_at(frame, 1000);
                let expected = count_spawned(reference, &settings);
                assert_eq!(expected, (frame as usize).saturating_add(1).min(live));
                // The graph's dispatch covers the live length rounded up to whole workgroups; the second launch also
                // covers the retired tail, which `i < live` must keep out of the count.
                for slots in [live, live + 3] {
                    let invocations = (slots as u32).div_ceil(group) * group;
                    let counted =
                        classify_on_cpu(live as u32, &settings, &allocation, 0, invocations)
                            .expect("classify twin");
                    assert_eq!(
                        counted as usize, expected,
                        "live {live}, frame {frame}, {invocations} invocations; {LABEL}"
                    );
                    // The count accumulates onto what it holds: the graph's reset upload is what zeroes it.
                    let more = classify_on_cpu(live as u32, &settings, &allocation, 7, invocations)
                        .expect("classify twin");
                    assert_eq!(
                        more as usize,
                        expected + 7,
                        "live {live}, frame {frame} from 7; {LABEL}"
                    );
                }
            }
        }
    }

    /// Kernel-level twin evidence; graph and indirect dispatch GPU-only (`LABEL`). The two twins run in sequence over
    /// frames that clamp to the budget, to the live count and to the allocation, and the allocation is compared with
    /// the host reference `step_cpu`. The shadow bodies perform the operations of `step_cpu` in the same order and
    /// Rust does not contract them, so floats agree to the last bit.
    #[test]
    fn twin_frames_match_host_reference() {
        let capacity = 130;
        for (dt, gravity) in [(0.016f32, -9.8f32), (0.5, -2.0)] {
            let mut allocation = spawn(0, capacity, 1);
            let mut reference = allocation.clone();
            // (live count in the count buffer, budget). 1000 is a malformed count past the allocation.
            for (frame, (count, budget)) in [
                (130, 100),
                (130, 1000),
                (65, 1000),
                (0, 1000),
                (1000, 1000),
                (130, 0),
                (129, 64),
                (130, 65),
            ]
            .into_iter()
            .enumerate()
            {
                let settings = Settings {
                    dt,
                    gravity,
                    budget,
                    frame: u32::MAX,
                };
                let (active, args) =
                    prepare_on_cpu(count, &settings, &allocation).expect("prepare twin");
                let (expected_active, expected_args) = expected_prepare(count, capacity, budget);
                assert_eq!(active, expected_active, "active, frame {frame}; {LABEL}");
                assert_eq!(args, expected_args, "args, frame {frame}; {LABEL}");

                allocation = integrate_on_cpu(active, args.x, &settings, &allocation)
                    .expect("integrate twin");
                let live = count as usize;
                let stepped = step_cpu(&mut reference[..live.min(capacity)], &settings);
                assert_eq!(stepped, active.count as usize, "stepped, frame {frame}");
                assert_eq!(
                    allocation, reference,
                    "allocation after frame {frame}; {LABEL}"
                );
            }
            // The first two frames step 100 and then all 130 particles, so every particle moved at least once (the
            // frames (0, 1000) and (130, 0) step nothing).
            assert!(reference.iter().all(|particle| particle.steps >= 1));
        }
    }

    /// Kernel-level twin evidence; graph and indirect dispatch GPU-only (`LABEL`). Each frame of a pool that is
    /// truncated and grown between frames runs on the GPU, and then both kernels run on the twin from the state the
    /// GPU started the frame from: `prepare_dispatch` from the allocation read back before the frame, `integrate` from
    /// that allocation, the active count the frame's graph read back and the indirect arguments read back after it.
    /// The growth and the indirect launch stay on the GPU.
    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn twin_matches_reference_and_gpu_within_the_derived_bounds() {
        let device = HeadlessDevice::new().expect("Vulkan adapter");
        let frame = Frame::new(&device).unwrap();
        for (dt, gravity) in [(0.016f32, -9.8f32), (0.5, -2.0)] {
            let mut reference = spawn(0, 130, 1);
            let mut particles = device.create_pool("particles", &reference, 130);
            // The actions before each frame: grow, truncate, or nothing. Each is GPU-only.
            for (index, (budget, action)) in [
                (100, "none"),
                (1000, "grow"),
                (1000, "none"),
                (1000, "truncate 65"),
                (1000, "truncate 0"),
            ]
            .into_iter()
            .enumerate()
            {
                match action {
                    "grow" => {
                        let spawned = spawn(reference.len(), 70, 1);
                        let growth = device.pool_push(&mut particles, &spawned).unwrap();
                        assert!(growth.is_some(), "130 -> 200 must grow the allocation");
                        reference.extend(spawned);
                    }
                    "truncate 65" => {
                        device.pool_truncate(&mut particles, 65).unwrap();
                        reference.truncate(65);
                    }
                    "truncate 0" => {
                        device.pool_truncate(&mut particles, 0).unwrap();
                        reference.clear();
                    }
                    _ => {}
                }
                let settings = Settings {
                    dt,
                    gravity,
                    budget,
                    frame: u32::MAX,
                };
                assert_derivation_domain(&settings);

                let len = particles.len();
                let capacity = particles.capacity();
                let before = device.read_typed_buffer(particles.buffer()).unwrap();
                assert_eq!(before.len(), capacity);
                let (expected_active, expected_args) =
                    expected_prepare(len as u32, capacity, budget);
                let stepped = expected_active.count as usize;
                assert_state_domain(&before, stepped);

                let (gpu_count, _report, _edges) =
                    run_frame_traced(&device, &frame, &particles, &[settings]).unwrap();
                let gpu_active = Active { count: gpu_count };
                let gpu_args = device.read_typed_buffer(&frame.args).unwrap()[0];
                let after = device.read_typed_buffer(particles.buffer()).unwrap();
                assert_eq!(after.len(), capacity);

                // The GPU against the independent host references comes first so that a failing twin cannot hide it.
                assert_eq!(gpu_active, expected_active, "GPU active, frame {index}");
                assert_eq!(gpu_args, expected_args, "GPU args, frame {index}");
                assert_eq!(step_cpu(&mut reference, &settings), stepped);
                assert_close(&after[..len], &reference);

                // classify: integer-only, exact. Every live particle has spawned (`frame` is `u32::MAX`), so the
                // GPU-authored count is the logical length, which is what the twin counts too.
                let gpu_classified = device.read_typed_buffer(&frame.count).unwrap()[0];
                let twin_classified =
                    classify_on_cpu(len as u32, &settings, &before, 0, capacity as u32).unwrap();
                assert_eq!(
                    gpu_classified as usize, len,
                    "GPU classified, frame {index}"
                );
                assert_eq!(
                    twin_classified, gpu_classified,
                    "classified, frame {index}; {LABEL}"
                );

                // prepare_dispatch: integer-only, exact, fed the classification count.
                let (twin_active, twin_args) =
                    prepare_on_cpu(gpu_classified, &settings, &before).unwrap();
                assert_eq!(twin_active, expected_active, "twin active, frame {index}");
                assert_eq!(twin_args, expected_args, "twin args, frame {index}");
                assert_eq!(twin_active, gpu_active, "active, frame {index}; {LABEL}");
                assert_eq!(twin_args, gpu_args, "args, frame {index}; {LABEL}");

                // integrate: fed the GPU's active count and indirect arguments, over the GPU's starting allocation.
                let twin_after =
                    integrate_on_cpu(gpu_active, gpu_args.x, &settings, &before).unwrap();
                assert_eq!(twin_after.len(), after.len());
                for (slot, ((twin, gpu), start)) in
                    twin_after.iter().zip(&after).zip(&before).enumerate()
                {
                    assert_eq!(twin.vx, gpu.vx, "vx[{slot}], frame {index}");
                    assert_eq!(twin.steps, gpu.steps, "steps[{slot}], frame {index}");
                    if slot >= stepped {
                        assert_eq!(twin, start, "twin moved particle {slot}, frame {index}");
                        assert_eq!(gpu, start, "GPU moved particle {slot}, frame {index}");
                    }
                }
                let vy = largest_abs(twin_after.iter().map(|p| p.vy), after.iter().map(|p| p.vy));
                let x = largest_abs(twin_after.iter().map(|p| p.x), after.iter().map(|p| p.x));
                let y = largest_abs(twin_after.iter().map(|p| p.y), after.iter().map(|p| p.y));
                println!(
                    "S4 component-pool integrate, dt {dt}, frame {index}, {stepped} stepped of {len}: twin versus GPU max abs error vy {vy:e} (bound {VY_BOUND:e}), x {x:e} (bound {X_BOUND:e}), y {y:e} (bound {Y_BOUND:e}); {LABEL}"
                );
                assert!(vy <= VY_BOUND, "vy {vy:e}");
                assert!(x <= X_BOUND, "x {x:e}");
                assert!(y <= Y_BOUND, "y {y:e}");
            }
        }
    }

    #[test]
    fn the_budget_keeps_the_indirect_dispatch_within_the_default_workgroup_limit() {
        let groups = BUDGET.div_ceil(pool::integrate::DESCRIPTOR.workgroup_size[0]);
        assert!(
            groups <= gust_wgpu::DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
            "a budget of {BUDGET} particles needs {groups} workgroups"
        );
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn every_stage_compiles_to_wgsl() {
        for descriptor in descriptors() {
            let wgsl = gust::slang::compile_wgsl(descriptor).unwrap();
            assert!(wgsl.contains("@compute"));
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_authored_state_survives_growth_without_readback() {
        let device = HeadlessDevice::new().expect("Vulkan adapter");
        let frame = Frame::new(&device).unwrap();
        let mut reference = spawn(0, 3, 1);
        let mut particles = device.create_pool("particles", &reference, 4);
        assert_eq!((particles.len(), particles.capacity()), (3, 4));

        // Frame 1 mutates the three live particles on the GPU.
        let (active, _) = run_frame(&device, &frame, &particles, &[settings(1000)]).unwrap();
        assert_eq!(active, 3);
        step_cpu(&mut reference, &settings(1000));

        // Growth boundaries: 4 -> 8 (geometric), 8 -> 70 (required exceeds doubling).
        let spawned = spawn(3, 2, 1);
        let growth = device.pool_push(&mut particles, &spawned).unwrap().unwrap();
        reference.extend(spawned);
        assert_eq!(
            growth,
            gust_wgpu::GrowthRecord {
                old_capacity: 4,
                new_capacity: 8,
                copied_bytes: 3 * size_of::<Particle>() as u64
            }
        );
        let spawned = spawn(5, 65, 1);
        let growth = device.pool_push(&mut particles, &spawned).unwrap().unwrap();
        reference.extend(spawned);
        assert_eq!((growth.old_capacity, growth.new_capacity), (8, 70));
        assert_eq!(growth.copied_bytes, 5 * size_of::<Particle>() as u64);
        assert!(device.pool_push(&mut particles, &[]).unwrap().is_none());
        assert_eq!(
            (
                particles.len(),
                particles.capacity(),
                particles.generation()
            ),
            (70, 70, 2)
        );

        // The mutated state from frame 1 is intact after two copies, and frame 2
        // integrates all 70 (two workgroups, partial second) through the indirect path.
        assert_close(&device.pool_read(&particles).unwrap(), &reference);
        let (active, report, edges) =
            run_frame_traced(&device, &frame, &particles, &[settings(1000)]).unwrap();
        assert_eq!(active, 70);
        // Declared edges, in node order: the settings write and the count reset have none, classify follows both,
        // prepare_dispatch follows classify, the indirect integrate follows prepare_dispatch, and the readback
        // follows integrate.
        assert_eq!(
            edges,
            [vec![], vec![], vec![0, 1], vec![2], vec![3], vec![4]]
        );
        assert_eq!((report.derived_edges, report.declared_edges), (0, 5));
        assert_eq!(report.indirect_dispatches, 1);
        assert_eq!(report.dispatches, 3);
        assert_eq!(report.readback_bytes, size_of::<Active>() as u64);
        assert_eq!(
            report.upload_bytes,
            (size_of::<Settings>() + size_of::<u32>()) as u64
        );
        step_cpu(&mut reference, &settings(1000));
        assert_close(&device.pool_read(&particles).unwrap(), &reference);

        // Retirement: both copies have completed by now.
        assert_eq!(particles.retired_allocations(), 2);
        assert_eq!(device.pool_reclaim(&mut particles).unwrap(), 2);
        assert_eq!(particles.retired_allocations(), 0);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn active_count_is_derived_on_the_gpu_and_clamped() {
        let device = HeadlessDevice::new().expect("Vulkan adapter");
        let frame = Frame::new(&device).unwrap();
        let mut reference = spawn(0, 130, 1);
        let mut particles = device.create_pool("particles", &reference, 130);

        // Budget below the length: only the first 100 move.
        let (active, _) = run_frame(&device, &frame, &particles, &[settings(100)]).unwrap();
        assert_eq!(active, 100);
        assert_eq!(step_cpu(&mut reference, &settings(100)), 100);
        assert_close(&device.pool_read(&particles).unwrap(), &reference);

        // Truncate: the count buffer shrinks, so fewer particles are active, and
        // the read-back prefix shrinks with it.
        device.pool_truncate(&mut particles, 65).unwrap();
        reference.truncate(65);
        let (active, _) = run_frame(&device, &frame, &particles, &[settings(1000)]).unwrap();
        assert_eq!(active, 65);
        step_cpu(&mut reference, &settings(1000));
        assert_close(&device.pool_read(&particles).unwrap(), &reference);
        assert!(matches!(
            device.pool_truncate(&mut particles, 66),
            Err(Error::PoolTruncateGrows {
                len: 65,
                requested: 66
            })
        ));

        // Zero length: no particle is active, nothing changes, nothing panics.
        device.pool_truncate(&mut particles, 0).unwrap();
        let (active, report) = run_frame(&device, &frame, &particles, &[settings(1000)]).unwrap();
        assert_eq!(active, 0);
        assert_eq!(report.indirect_dispatches, 1);
        assert!(device.pool_read(&particles).unwrap().is_empty());

        // Push after truncation reuses capacity and resumes from the logical length.
        let spawned = spawn(0, 2, 1);
        assert!(
            device
                .pool_push(&mut particles, &spawned)
                .unwrap()
                .is_none()
        );
        assert_eq!(particles.len(), 2);
        assert_eq!(device.pool_read(&particles).unwrap(), spawned);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn growth_while_a_frame_is_in_flight_preserves_its_writes() {
        // Queue order: the growth copy runs after the still-executing integrate,
        // so the new allocation observes that frame's writes.
        let device = HeadlessDevice::new().expect("Vulkan adapter");
        let frame = Frame::new(&device).unwrap();
        let mut reference = spawn(0, 64, 1);
        let mut particles = device.create_pool("particles", &reference, 64);
        device
            .write_typed_buffer(&frame.settings, &[settings(1000)])
            .unwrap();
        device
            .write_typed_buffer(&frame.active, &[Active { count: 64 }])
            .unwrap();
        // Submit without waiting, then grow immediately; the typed call borrowed the old allocation
        // only inside `submit`, so `pool_push` can take the pool mutably.
        let job = pool::integrate::submit(
            &device,
            64,
            Independent(&frame.active),
            Independent(&frame.settings),
            Independent(particles.buffer()),
        )
        .unwrap();
        let spawned = spawn(64, 64, 1);
        let growth = device.pool_push(&mut particles, &spawned).unwrap().unwrap();
        assert_eq!((growth.old_capacity, growth.new_capacity), (64, 128));
        job.wait().unwrap();
        step_cpu(&mut reference, &settings(1000));
        reference.extend(spawned);
        assert_close(&device.pool_read(&particles).unwrap(), &reference);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn indirect_dispatch_rejects_wrong_buffers_and_bindings() {
        let device = HeadlessDevice::new().expect("Vulkan adapter");
        let frame = Frame::new(&device).unwrap();
        let particles = device.create_pool("particles", &spawn(0, 8, 1), 8);
        let settings = [settings(1000)];
        let integrate_bindings = [
            BufferBinding::read_only(&frame.active).independent_length(),
            BufferBinding::read_only(&frame.settings).independent_length(),
            BufferBinding::read_write(particles.buffer()).independent_length(),
        ];
        let integrate = BufferDispatch::new(&pool::integrate::DESCRIPTOR, 0, &integrate_bindings);

        // Args must come from create_indirect_buffer; a plain storage buffer of
        // the right layout is rejected before any GPU work.
        let plain = device.create_typed_buffer(
            "plain args",
            &[DispatchArgs { x: 1, y: 1, z: 1 }],
            GpuBufferAccess::ReadWrite,
        );
        let mut graph = StagedGraph::new();
        graph.dispatch_indirect(integrate, &plain, &[]);
        assert!(matches!(
            device.execute_graph(&graph),
            Err(Error::IndirectArgsLayout(_))
        ));

        // The wrong element layout cannot even create an indirect buffer.
        assert!(matches!(
            device.create_indirect_buffer("wrong", &[Active { count: 1 }]),
            Err(Error::IndirectArgsLayout("Active"))
        ));
        assert!(matches!(
            device.create_indirect_buffer("wrong", &[1u32, 1, 1]),
            Err(Error::IndirectArgsLayout("u32"))
        ));

        // Every binding of an indirect dispatch must opt in to an independent length.
        let strict = [
            BufferBinding::read_only(&frame.active),
            BufferBinding::read_only(&frame.settings).independent_length(),
            BufferBinding::read_write(particles.buffer()).independent_length(),
        ];
        let mut graph = StagedGraph::new();
        graph.dispatch_indirect(
            BufferDispatch::new(&pool::integrate::DESCRIPTOR, 0, &strict),
            &frame.args,
            &[],
        );
        assert!(matches!(
            device.execute_graph(&graph),
            Err(Error::IndirectBindingLength { parameter }) if parameter == "active"
        ));

        // The args buffer is a read for hazard checking: a dispatch that reads
        // args written by prepare_dispatch must declare that edge.
        let prepare_bindings = [
            BufferBinding::read_only(&frame.count).independent_length(),
            BufferBinding::read_only(&frame.settings).independent_length(),
            BufferBinding::read_only(particles.buffer()).independent_length(),
            BufferBinding::read_write(&frame.active).independent_length(),
            BufferBinding::read_write(&frame.args).independent_length(),
        ];
        let mut graph = StagedGraph::new();
        let upload = graph.upload(&frame.settings, &settings, &[]);
        graph.dispatch(
            BufferDispatch::new(&pool::prepare_dispatch::DESCRIPTOR, 1, &prepare_bindings),
            &[upload],
        );
        graph.dispatch_indirect(integrate, &frame.args, &[upload]);
        assert!(matches!(
            device.execute_graph(&graph),
            Err(Error::GraphMissingDependency {
                node: 2,
                producer: 1
            })
        ));

        // The derived graph over the same buffers gets the edge the rejected graph lacked: integrate
        // depends on prepare_dispatch (through `args`, `active` and the particles) without a declared
        // edge, and the readback depends on prepare_dispatch, not integrate.
        let prepare_call = pool::prepare_dispatch::call(
            1,
            Independent(particles.count_buffer()),
            Independent(&frame.settings),
            Independent(particles.buffer()),
            Independent(&frame.active),
            Independent(&frame.args),
        );
        let integrate_call = pool::integrate::call(
            0,
            Independent(&frame.active),
            Independent(&frame.settings),
            Independent(particles.buffer()),
        );
        let mut derived = StagedGraph::new();
        let write = derived.write(&frame.settings, &settings);
        let prepare = derived.run(&prepare_call);
        let integrate = derived.run_indirect(&integrate_call, &frame.args);
        let read = derived.read(&frame.active);
        let edges = |node| derived.dependency_indices(node);
        assert_eq!(
            [edges(write), edges(prepare), edges(integrate), edges(read)],
            [vec![], vec![0], vec![0, 1], vec![1]]
        );
        assert!(
            [write, prepare, integrate, read]
                .into_iter()
                .all(|node| derived.is_derived(node))
        );
        let output = device.execute_graph(&derived).unwrap();
        assert_eq!(
            (output.report.derived_edges, output.report.declared_edges),
            (4, 0)
        );
        assert_eq!(output.readback::<Active>(read).unwrap()[0].count, 8);
    }

    // T13: the count that sizes the indirect dispatch is authored on the GPU by `classify` (atomic increments) and
    // read back only to compare with the independent host reference; the graph never reads it to decide work.
    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_active_count_matches_cpu_reference_at_boundaries() {
        let device = HeadlessDevice::new().expect("Vulkan adapter");
        let frame = Frame::new(&device).unwrap();
        for n in [0usize, 1, 63, 64, 65, 257] {
            let (particles, mut reference) = pool_with_retired_tail(&device, n);
            let n32 = n as u32;
            // Nothing (or one), about half, everything live, and the u32 extreme.
            for frame_index in [0, n32 / 2, n32.saturating_sub(1), u32::MAX] {
                let settings = settings_at(frame_index, 1000);
                let (active, report) = run_frame(&device, &frame, &particles, &[settings]).unwrap();
                let classified = device.read_typed_buffer(&frame.count).unwrap()[0];
                let expected_classified = count_spawned(&reference, &settings);
                let expected_active = step_cpu(&mut reference, &settings);
                assert_eq!(
                    (classified as usize, active as usize),
                    (expected_classified, expected_active),
                    "n {n}, frame {frame_index}"
                );
                assert_eq!(
                    expected_classified,
                    (frame_index as usize).saturating_add(1).min(n),
                    "emitter reference for n {n}, frame {frame_index}"
                );
                assert_eq!(report.indirect_dispatches, 1, "n {n}, frame {frame_index}");
                // An empty pool launches no classification (a zero-element dispatch is skipped).
                assert_eq!(
                    report.dispatches,
                    if n == 0 { 2 } else { 3 },
                    "n {n}, frame {frame_index}"
                );
                assert_close(&device.pool_read(&particles).unwrap(), &reference);
            }
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn indirect_arguments_are_clamped_to_budget_and_allocation() {
        let device = HeadlessDevice::new().expect("Vulkan adapter");
        let frame = Frame::new(&device).unwrap();
        let (particles, mut reference) = pool_with_retired_tail(&device, 65);
        assert_eq!(particles.capacity(), 68);

        // Budget below the classified count: 65 classified, 40 active, one workgroup.
        let (active, _) = run_frame(&device, &frame, &particles, &[settings(40)]).unwrap();
        assert_eq!(device.read_typed_buffer(&frame.count).unwrap(), vec![65]);
        assert_eq!(active, 40);
        assert_eq!(
            device.read_typed_buffer(&frame.args).unwrap(),
            vec![DispatchArgs { x: 1, y: 1, z: 1 }]
        );
        assert_eq!(step_cpu(&mut reference, &settings(40)), 40);
        assert_close(&device.pool_read(&particles).unwrap(), &reference);

        // Unclamped budget: 65 active needs a partial second workgroup.
        let (active, _) = run_frame(&device, &frame, &particles, &[settings(1000)]).unwrap();
        assert_eq!(active, 65);
        assert_eq!(
            device.read_typed_buffer(&frame.args).unwrap(),
            vec![DispatchArgs { x: 2, y: 1, z: 1 }]
        );
        step_cpu(&mut reference, &settings(1000));
        assert_close(&device.pool_read(&particles).unwrap(), &reference);

        // A malformed count (injected by the host; no classification runs) degrades to "the whole allocation",
        // never past it and never to zero; the budget still caps it.
        for (budget, expected_active, expected_groups) in [(1000, 68, 2), (10, 10, 1)] {
            device
                .write_typed_buffer(&frame.count, &[u32::MAX])
                .unwrap();
            let settings = [settings(budget)];
            let prepare_bindings = [
                BufferBinding::read_only(&frame.count).independent_length(),
                BufferBinding::read_only(&frame.settings).independent_length(),
                BufferBinding::read_only(particles.buffer()).independent_length(),
                BufferBinding::read_write(&frame.active).independent_length(),
                BufferBinding::read_write(&frame.args).independent_length(),
            ];
            let integrate_bindings = [
                BufferBinding::read_only(&frame.active).independent_length(),
                BufferBinding::read_only(&frame.settings).independent_length(),
                BufferBinding::read_write(particles.buffer()).independent_length(),
            ];
            let mut graph = StagedGraph::new();
            let upload = graph.upload(&frame.settings, &settings, &[]);
            let prepare = graph.dispatch(
                BufferDispatch::new(&pool::prepare_dispatch::DESCRIPTOR, 1, &prepare_bindings),
                &[upload],
            );
            let integrate = graph.dispatch_indirect(
                BufferDispatch::new(&pool::integrate::DESCRIPTOR, 0, &integrate_bindings),
                &frame.args,
                &[prepare],
            );
            let active = graph.readback(&frame.active, &[integrate]);
            let output = device.execute_graph(&graph).unwrap();
            assert_eq!(
                output.readback::<Active>(active).unwrap(),
                vec![Active {
                    count: expected_active
                }]
            );
            assert_eq!(
                device.read_typed_buffer(&frame.args).unwrap(),
                vec![DispatchArgs {
                    x: expected_groups,
                    y: 1,
                    z: 1
                }]
            );
            // The live prefix stepped once more; the retired slots are unspecified and not compared.
            step_cpu(&mut reference, &settings[0]);
            assert_close(&device.pool_read(&particles).unwrap(), &reference);
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn graph_rejects_undeclared_classification_edges() {
        let device = HeadlessDevice::new().expect("Vulkan adapter");
        let frame = Frame::new(&device).unwrap();
        let particles = device.create_pool("particles", &spawn(0, 8, 1), 8);
        let settings = [settings(1000)];
        let classify_bindings = [
            BufferBinding::read_only(particles.count_buffer()).independent_length(),
            BufferBinding::read_only(&frame.settings).independent_length(),
            BufferBinding::read_only(particles.buffer()).independent_length(),
            BufferBinding::read_write(&frame.count).independent_length(),
        ];
        let prepare_bindings = [
            BufferBinding::read_only(&frame.count).independent_length(),
            BufferBinding::read_only(&frame.settings).independent_length(),
            BufferBinding::read_only(particles.buffer()).independent_length(),
            BufferBinding::read_write(&frame.active).independent_length(),
            BufferBinding::read_write(&frame.args).independent_length(),
        ];
        let classify = BufferDispatch::new(&pool::classify::DESCRIPTOR, 8, &classify_bindings);
        let prepare =
            BufferDispatch::new(&pool::prepare_dispatch::DESCRIPTOR, 1, &prepare_bindings);

        // prepare_dispatch reads the count classify writes: the edge must be declared.
        let mut graph = StagedGraph::new();
        let upload = graph.upload(&frame.settings, &settings, &[]);
        let reset = graph.upload(&frame.count, &[0u32], &[]);
        graph.dispatch(classify, &[upload, reset]);
        graph.dispatch(prepare, &[upload, reset]);
        assert!(matches!(
            device.execute_graph(&graph),
            Err(Error::GraphMissingDependency {
                node: 3,
                producer: 2
            })
        ));

        // classify accumulates into the count the reset upload zeroes: same rule.
        let mut graph = StagedGraph::new();
        let upload = graph.upload(&frame.settings, &settings, &[]);
        graph.upload(&frame.count, &[0u32], &[]);
        graph.dispatch(classify, &[upload]);
        assert!(matches!(
            device.execute_graph(&graph),
            Err(Error::GraphMissingDependency {
                node: 2,
                producer: 1
            })
        ));
        assert_eq!(device.read_typed_buffer(&frame.count).unwrap(), vec![0]);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn count_bindings_reject_empty_and_mismatched_buffers() {
        let device = HeadlessDevice::new().expect("Vulkan adapter");
        let frame = Frame::new(&device).unwrap();
        let particles = device.create_pool("particles", &spawn(0, 8, 1), 8);
        device
            .write_typed_buffer(&frame.settings, &[settings(1000)])
            .unwrap();
        let empty =
            device.create_typed_buffer::<u32>("empty count", &[], GpuBufferAccess::ReadWrite);

        // A zero-length independent count cannot receive the classification.
        let classify_bindings = [
            BufferBinding::read_only(particles.count_buffer()).independent_length(),
            BufferBinding::read_only(&frame.settings).independent_length(),
            BufferBinding::read_only(particles.buffer()).independent_length(),
            BufferBinding::read_write(&empty).independent_length(),
        ];
        assert!(matches!(
            device.dispatch_buffers(&pool::classify::DESCRIPTOR, 8, &classify_bindings),
            Err(Error::IndependentLengthEmpty { parameter }) if parameter == "count"
        ));

        // Nor can prepare_dispatch read one.
        let prepare_bindings = [
            BufferBinding::read_only(&empty).independent_length(),
            BufferBinding::read_only(&frame.settings).independent_length(),
            BufferBinding::read_only(particles.buffer()).independent_length(),
            BufferBinding::read_write(&frame.active).independent_length(),
            BufferBinding::read_write(&frame.args).independent_length(),
        ];
        assert!(matches!(
            device.dispatch_buffers(&pool::prepare_dispatch::DESCRIPTOR, 1, &prepare_bindings),
            Err(Error::IndependentLengthEmpty { parameter }) if parameter == "count"
        ));

        // The count must be a `uint` buffer; a struct of the same size is rejected.
        let prepare_bindings = [
            BufferBinding::read_only(&frame.active).independent_length(),
            BufferBinding::read_only(&frame.settings).independent_length(),
            BufferBinding::read_only(particles.buffer()).independent_length(),
            BufferBinding::read_write(&frame.active).independent_length(),
            BufferBinding::read_write(&frame.args).independent_length(),
        ];
        assert!(matches!(
            device.dispatch_buffers(&pool::prepare_dispatch::DESCRIPTOR, 1, &prepare_bindings),
            Err(Error::UnsupportedResourceLayout { parameter, .. }) if parameter == "count"
        ));

        // The one-element count cannot pretend to share the per-slot element count.
        let classify_bindings = [
            BufferBinding::read_only(particles.count_buffer()).independent_length(),
            BufferBinding::read_only(&frame.settings).independent_length(),
            BufferBinding::read_only(particles.buffer()).independent_length(),
            BufferBinding::read_write(&frame.count),
        ];
        assert!(matches!(
            device.dispatch_buffers(&pool::classify::DESCRIPTOR, 8, &classify_bindings),
            Err(Error::BufferLength {
                parameter,
                expected: 8,
                actual: 1
            }) if parameter == "count"
        ));
        assert_eq!(device.read_typed_buffer(&frame.count).unwrap(), vec![0]);
    }
}
