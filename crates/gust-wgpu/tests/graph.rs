//! `StagedGraph` derives dependency edges from typed calls (roadmap M2, m2.e).
//!
//! The typed methods `write`, `run`, `run_indirect` and `read` record the same nodes as their explicit
//! twins (`upload`, `dispatch`, `dispatch_indirect`, `readback`) and compute the edges themselves: for each
//! buffer a node touches, the latest earlier writer, plus, when the node writes, every reader since that
//! writer. The expected edge sets below were written down before the derivation ran (spike S4) and are
//! not adjusted to its output. Kernels are macro-generated; the calls are the generated `call` functions.
//! Expectations come from independent host references, never from running a shader shadow body.

mod common;

use common::device;

use gust::Independent;
use gust_wgpu::{Error, GpuBuffer, GpuBufferAccess, HeadlessDevice, NodeId, StagedGraph};

// The two shapes of the staged-graph and component-pool examples. Kernel signatures (parameter types,
// access, order) and bodies are those of `examples/staged-graph` and `examples/component-pool`.
#[gust::gpu_module]
mod staged {
    #[derive(Debug, PartialEq)]
    pub struct Settings {
        pub scale: float,
        pub offset: float,
        pub threshold: float,
        pub block: uint,
    }

    #[derive(Debug, PartialEq)]
    pub struct Summary {
        pub total: float,
        pub peak: float,
        pub above_threshold: uint,
        pub count: uint,
    }

    #[allow(clippy::len_zero)]
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn transform(
        id: SV_DispatchThreadID,
        samples: StructuredBuffer<float>,
        settings: StructuredBuffer<Settings>,
        mut intermediate: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < intermediate.len() && 0u32 < settings.len() {
            let parameters = settings[0u32];
            intermediate[i] = samples[i] * parameters.scale + parameters.offset;
        }
    }

    #[allow(clippy::assign_op_pattern)]
    fn fold(acc: Summary, value: float, threshold: float) -> Summary {
        let mut next = Summary {
            total: acc.total + value,
            peak: acc.peak,
            above_threshold: acc.above_threshold,
            count: acc.count + 1u32,
        };
        if value > next.peak {
            next.peak = value;
        }
        if value > threshold {
            next.above_threshold = next.above_threshold + 1u32;
        }
        next
    }

    #[allow(clippy::len_zero)]
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn summarize(
        id: SV_DispatchThreadID,
        intermediate: StructuredBuffer<float>,
        settings: StructuredBuffer<Settings>,
        mut summaries: RWStructuredBuffer<Summary>,
    ) {
        let b = id.x;
        if b < summaries.len() && 0u32 < settings.len() {
            let parameters = settings[0u32];
            let start = b * parameters.block;
            let limit = intermediate.len();
            let mut acc = Summary {
                total: 0.0f32,
                peak: 0.0f32,
                above_threshold: 0u32,
                count: 0u32,
            };
            if start < limit {
                acc = fold(acc, intermediate[start], parameters.threshold);
            }
            if start + 1u32 < limit {
                acc = fold(acc, intermediate[start + 1u32], parameters.threshold);
            }
            if start + 2u32 < limit {
                acc = fold(acc, intermediate[start + 2u32], parameters.threshold);
            }
            if start + 3u32 < limit {
                acc = fold(acc, intermediate[start + 3u32], parameters.threshold);
            }
            if start + 4u32 < limit {
                acc = fold(acc, intermediate[start + 4u32], parameters.threshold);
            }
            if start + 5u32 < limit {
                acc = fold(acc, intermediate[start + 5u32], parameters.threshold);
            }
            if start + 6u32 < limit {
                acc = fold(acc, intermediate[start + 6u32], parameters.threshold);
            }
            if start + 7u32 < limit {
                acc = fold(acc, intermediate[start + 7u32], parameters.threshold);
            }
            summaries[b] = acc;
        }
    }
}

#[gust::gpu_module]
mod pool {
    #[derive(Debug, PartialEq)]
    pub struct Particle {
        pub x: float,
        pub y: float,
        pub vx: float,
        pub vy: float,
        pub steps: uint,
    }

    #[derive(Debug, PartialEq)]
    pub struct Settings {
        pub dt: float,
        pub gravity: float,
        pub budget: uint,
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
            args[0u32].x = n / 64u32 + min_uint(n % 64u32, 1u32);
            args[0u32].y = 1u32;
            args[0u32].z = 1u32;
        }
    }

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

/// The edges of `node` as plain indices.
#[track_caller]
fn edges(graph: &StagedGraph<'_>, node: NodeId) -> Vec<usize> {
    graph.dependency_indices(node)
}

// ---- spike S4: the two example shapes, derived against the pre-registered sets ----

const SAMPLES: usize = 20;
const BLOCK: u32 = 8;

struct StagedState {
    samples: GpuBuffer<f32>,
    settings: GpuBuffer<staged::Settings>,
    intermediate: GpuBuffer<f32>,
    summaries: GpuBuffer<staged::Summary>,
}

fn staged_state(device: &HeadlessDevice) -> StagedState {
    let samples: Vec<f32> = (0..SAMPLES).map(|index| (index % 5) as f32).collect();
    let summaries = SAMPLES.div_ceil(BLOCK as usize);
    StagedState {
        samples: device.create_typed_buffer("s4 samples", &samples, GpuBufferAccess::ReadOnly),
        settings: device.create_typed_buffer(
            "s4 settings",
            &[staged_settings()],
            GpuBufferAccess::ReadOnly,
        ),
        intermediate: device.create_typed_buffer(
            "s4 intermediate",
            &[0.0f32; SAMPLES],
            GpuBufferAccess::ReadWrite,
        ),
        summaries: device.create_typed_buffer(
            "s4 summaries",
            &vec![
                staged::Summary {
                    total: 0.0,
                    peak: 0.0,
                    above_threshold: 0,
                    count: 0,
                };
                summaries
            ],
            GpuBufferAccess::ReadWrite,
        ),
    }
}

fn staged_settings() -> staged::Settings {
    staged::Settings {
        scale: 2.0,
        offset: 1.0,
        threshold: 5.0,
        block: BLOCK,
    }
}

/// Independent host reference: ordinary Rust over the same inputs.
fn staged_reference() -> Vec<staged::Summary> {
    let samples: Vec<f32> = (0..SAMPLES).map(|index| (index % 5) as f32).collect();
    let settings = staged_settings();
    samples
        .chunks(BLOCK as usize)
        .map(|block| {
            let mut summary = staged::Summary {
                total: 0.0,
                peak: 0.0,
                above_threshold: 0,
                count: 0,
            };
            for &sample in block {
                let value = sample * settings.scale + settings.offset;
                summary.total += value;
                if value > summary.peak {
                    summary.peak = value;
                }
                if value > settings.threshold {
                    summary.above_threshold += 1;
                }
                summary.count += 1;
            }
            summary
        })
        .collect()
}

#[test]
fn s4_staged_graph_shape_derives_the_pre_registered_edges() {
    let device = device();
    let state = staged_state(&device);
    let settings = [staged_settings()];
    let transform = staged::transform::call(
        SAMPLES as u32,
        &state.samples,
        Independent(&state.settings),
        &state.intermediate,
    );
    let summarize = staged::summarize::call(
        state.summaries.len() as u32,
        Independent(&state.intermediate),
        Independent(&state.settings),
        &state.summaries,
    );

    // Declared edges, as the example builds them today.
    let mut declared = StagedGraph::new();
    let upload = declared.upload(&state.settings, &settings, &[]);
    let stage_a = declared.dispatch((&transform).into(), &[upload]);
    let stage_b = declared.dispatch((&summarize).into(), &[stage_a]);
    let declared_read = declared.readback(&state.summaries, &[stage_b]);

    // Derived edges from the typed methods.
    let mut derived = StagedGraph::new();
    let write = derived.write(&state.settings, &settings);
    let first = derived.run(&transform);
    let second = derived.run(&summarize);
    let derived_read = derived.read(&state.summaries);

    let declared_sets: Vec<_> = [upload, stage_a, stage_b, declared_read]
        .into_iter()
        .map(|node| edges(&declared, node))
        .collect();
    let derived_sets: Vec<_> = [write, first, second, derived_read]
        .into_iter()
        .map(|node| edges(&derived, node))
        .collect();
    println!("S4 staged-graph declared: {declared_sets:?}");
    println!("S4 staged-graph derived:  {derived_sets:?}");

    // Pre-registered: summarize also reads `settings`, whose last writer is node 0.
    assert_eq!(declared_sets, [vec![], vec![0], vec![1], vec![2]]);
    assert_eq!(derived_sets, [vec![], vec![0], vec![0, 1], vec![2]]);
    assert!(
        [write, first, second, derived_read]
            .into_iter()
            .all(|node| derived.is_derived(node))
    );
    assert!(
        [upload, stage_a, stage_b, declared_read]
            .into_iter()
            .all(|node| !declared.is_derived(node))
    );

    let from_declared = device.execute_graph(&declared).unwrap();
    let declared_summaries = from_declared
        .readback::<staged::Summary>(declared_read)
        .unwrap();
    let from_derived = device.execute_graph(&derived).unwrap();
    let derived_summaries = from_derived
        .readback::<staged::Summary>(derived_read)
        .unwrap();
    assert_eq!(declared_summaries, staged_reference());
    assert_eq!(derived_summaries, declared_summaries);
}

const PARTICLES: usize = 20;
const BUDGET: u32 = 10;

struct PoolState {
    count: GpuBuffer<u32>,
    settings: GpuBuffer<pool::Settings>,
    particles: GpuBuffer<pool::Particle>,
    active: GpuBuffer<pool::Active>,
    args: GpuBuffer<pool::DispatchArgs>,
}

fn pool_state(device: &HeadlessDevice) -> PoolState {
    let particles: Vec<pool::Particle> = (0..PARTICLES)
        .map(|index| pool::Particle {
            x: index as f32,
            y: 0.0,
            vx: 1.0,
            vy: 0.0,
            steps: 0,
        })
        .collect();
    PoolState {
        count: device.create_typed_buffer(
            "s4 count",
            &[PARTICLES as u32],
            GpuBufferAccess::ReadOnly,
        ),
        settings: device.create_typed_buffer(
            "s4 pool settings",
            &[pool_settings()],
            GpuBufferAccess::ReadOnly,
        ),
        particles: device.create_typed_buffer(
            "s4 particles",
            &particles,
            GpuBufferAccess::ReadWrite,
        ),
        active: device.create_typed_buffer(
            "s4 active",
            &[pool::Active { count: 0 }],
            GpuBufferAccess::ReadWrite,
        ),
        args: device
            .create_indirect_buffer("s4 args", &[pool::DispatchArgs { x: 0, y: 0, z: 0 }])
            .unwrap(),
    }
}

fn pool_settings() -> pool::Settings {
    pool::Settings {
        dt: 0.5,
        gravity: -1.0,
        budget: BUDGET,
    }
}

#[test]
fn s4_component_pool_shape_derives_the_pre_registered_edges() {
    let device = device();
    let state = pool_state(&device);
    let settings = [pool_settings()];
    let prepare = pool::prepare_dispatch::call(
        1,
        Independent(&state.count),
        Independent(&state.settings),
        Independent(&state.particles),
        Independent(&state.active),
        Independent(&state.args),
    );
    let integrate = pool::integrate::call(
        0,
        Independent(&state.active),
        Independent(&state.settings),
        Independent(&state.particles),
    );

    // Declared edges, as the example builds them today (`app.rs` run_frame).
    let mut declared = StagedGraph::new();
    let upload = declared.upload(&state.settings, &settings, &[]);
    let prepared = declared.dispatch((&prepare).into(), &[upload]);
    let integrated = declared.dispatch_indirect((&integrate).into(), &state.args, &[prepared]);
    let declared_read = declared.readback(&state.active, &[integrated]);

    // Derived edges from the typed methods.
    let mut derived = StagedGraph::new();
    let write = derived.write(&state.settings, &settings);
    let first = derived.run(&prepare);
    let second = derived.run_indirect(&integrate, &state.args);
    let derived_read = derived.read(&state.active);

    let declared_sets: Vec<_> = [upload, prepared, integrated, declared_read]
        .into_iter()
        .map(|node| edges(&declared, node))
        .collect();
    let derived_sets: Vec<_> = [write, first, second, derived_read]
        .into_iter()
        .map(|node| edges(&derived, node))
        .collect();
    println!("S4 component-pool declared: {declared_sets:?}");
    println!("S4 component-pool derived:  {derived_sets:?}");

    // Pre-registered: the readback of `active` depends on `prepare`, its last writer, not on
    // `integrate`, which only reads it. The declared graph says `[integrate]`; the hazard rule explains
    // the difference, and `integrate` itself depends on `prepare` through `active` and `args`.
    assert_eq!(declared_sets, [vec![], vec![0], vec![1], vec![2]]);
    assert_eq!(derived_sets, [vec![], vec![0], vec![0, 1], vec![1]]);

    let from_declared = device.execute_graph(&declared).unwrap();
    assert_eq!(
        from_declared
            .readback::<pool::Active>(declared_read)
            .unwrap(),
        [pool::Active { count: BUDGET }]
    );
    assert_eq!(from_declared.report.indirect_dispatches, 1);
    let from_derived = device.execute_graph(&derived).unwrap();
    assert_eq!(
        from_derived.readback::<pool::Active>(derived_read).unwrap(),
        [pool::Active { count: BUDGET }]
    );
    assert_eq!(from_derived.report.indirect_dispatches, 1);

    // Both executions ran `integrate` over exactly the first `BUDGET` particles.
    let particles = device.read_typed_buffer(&state.particles).unwrap();
    assert_eq!(particles.len(), PARTICLES);
    for (index, particle) in particles.iter().enumerate() {
        assert_eq!(particle.steps, if index < BUDGET as usize { 2 } else { 0 });
    }
}

// ---- small kernels for the hazard rules ----

#[gust::gpu_module]
mod small {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn copy(
        id: SV_DispatchThreadID,
        src: StructuredBuffer<float>,
        mut dst: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < dst.len() && i < src.len() {
            dst[i] = src[i];
        }
    }
}

const COUNT: usize = 70;

fn ramp(offset: f32) -> Vec<f32> {
    (0..COUNT).map(|index| index as f32 + offset).collect()
}

fn floats(device: &HeadlessDevice, label: &str, values: &[f32]) -> GpuBuffer<f32> {
    device.create_typed_buffer(label, values, GpuBufferAccess::ReadWrite)
}

#[test]
fn a_write_after_readers_depends_on_each_reader_and_readers_do_not_depend_on_each_other() {
    let device = device();
    let (first, second) = (ramp(1.0), ramp(500.0));
    let x = floats(&device, "x", &vec![0.0; COUNT]);
    let y = floats(&device, "y", &vec![0.0; COUNT]);
    let z = floats(&device, "z", &vec![0.0; COUNT]);
    let copy_y = small::copy::call(COUNT as u32, &x, &y);
    let copy_z = small::copy::call(COUNT as u32, &x, &z);

    let mut graph = StagedGraph::new();
    let initial = graph.write(&x, &first);
    let reader_a = graph.run(&copy_y);
    let reader_b = graph.run(&copy_z);
    let overwrite = graph.write(&x, &second);
    assert_eq!(edges(&graph, initial), Vec::<usize>::new());
    assert_eq!(edges(&graph, reader_a), [0]);
    // Two consecutive readers of one buffer are independent of each other.
    assert_eq!(edges(&graph, reader_b), [0]);
    // WAR: the overwrite waits for both readers and for the writer before them.
    assert_eq!(edges(&graph, overwrite), [0, 1, 2]);

    device.execute_graph(&graph).unwrap();
    assert_eq!(device.read_typed_buffer(&y).unwrap(), first);
    assert_eq!(device.read_typed_buffer(&z).unwrap(), first);
    assert_eq!(device.read_typed_buffer(&x).unwrap(), second);
}

#[test]
fn a_write_after_a_write_depends_on_the_first_write_only() {
    let device = device();
    let (first, second) = (ramp(1.0), ramp(500.0));
    let x = floats(&device, "x", &vec![0.0; COUNT]);

    let mut graph = StagedGraph::new();
    let a = graph.write(&x, &first);
    let b = graph.write(&x, &second);
    let c = graph.read(&x);
    assert_eq!(edges(&graph, a), Vec::<usize>::new());
    // WAW with no reader in between: one edge.
    assert_eq!(edges(&graph, b), [0]);
    // RAW: a read depends on the latest writer only.
    assert_eq!(edges(&graph, c), [1]);

    let output = device.execute_graph(&graph).unwrap();
    assert_eq!(output.readback::<f32>(c).unwrap(), second);
}

#[test]
fn the_derived_edge_set_is_conservative_not_a_transitive_reduction() {
    let device = device();
    let (first, second) = (ramp(1.0), ramp(500.0));
    let x = floats(&device, "x", &vec![0.0; COUNT]);
    let y = floats(&device, "y", &vec![0.0; COUNT]);
    let copy = small::copy::call(COUNT as u32, &x, &y);

    let mut graph = StagedGraph::new();
    let w0 = graph.write(&x, &first);
    let r1 = graph.run(&copy);
    let w2 = graph.write(&x, &second);
    assert_eq!(edges(&graph, w0), Vec::<usize>::new());
    assert_eq!(edges(&graph, r1), [0]);
    // W0 -> R1 -> W2: W2 also lists W0, although it is reachable through R1.
    assert_eq!(edges(&graph, w2), [0, 1]);
    device.execute_graph(&graph).unwrap();
    assert_eq!(device.read_typed_buffer(&y).unwrap(), first);
}

// ---- mixed graphs ----

#[test]
fn a_derived_node_cannot_repair_an_invalid_explicit_node() {
    let device = device();
    let values = ramp(1.0);
    let x = floats(&device, "x", &vec![0.0; COUNT]);
    let y = floats(&device, "y", &vec![0.0; COUNT]);
    let z = floats(&device, "z", &vec![0.0; COUNT]);
    let x_to_y = small::copy::call(COUNT as u32, &x, &y);
    let y_to_z = small::copy::call(COUNT as u32, &y, &z);
    let x_to_z = small::copy::call(COUNT as u32, &x, &z);

    let mut graph = StagedGraph::new();
    let upload = graph.upload(&x, &values, &[]);
    // Explicit node 1 reads `x`, which node 0 writes, and declares no edge.
    let missing = graph.dispatch((&x_to_y).into(), &[]);
    let after = graph.run(&y_to_z);
    let later = graph.run(&x_to_z);
    assert_eq!(upload.index(), 0);
    assert_eq!(missing.index(), 1);
    assert!(!graph.is_derived(missing));
    assert_eq!(edges(&graph, missing), Vec::<usize>::new());
    // The derived nodes see the explicit ones as predecessors like any other.
    assert_eq!(edges(&graph, after), [1]);
    // `later` reads `x` (last written by node 0, so the reader at node 1 is no conflict) and writes `z` (last
    // written by node 2).
    assert_eq!(edges(&graph, later), [0, 2]);

    match device
        .execute_graph(&graph)
        .err()
        .expect("the explicit node stays invalid")
    {
        Error::GraphMissingDependency { node, producer } => assert_eq!((node, producer), (1, 0)),
        other => panic!("expected GraphMissingDependency, got {other:?}"),
    }
    assert_eq!(device.read_typed_buffer(&z).unwrap(), vec![0.0; COUNT]);
}

#[test]
fn a_derived_node_follows_a_valid_explicit_node() {
    let device = device();
    let values = ramp(1.0);
    let x = floats(&device, "x", &vec![0.0; COUNT]);
    let y = floats(&device, "y", &vec![0.0; COUNT]);
    let z = floats(&device, "z", &vec![0.0; COUNT]);
    let x_to_y = small::copy::call(COUNT as u32, &x, &y);
    let y_to_z = small::copy::call(COUNT as u32, &y, &z);

    let mut graph = StagedGraph::new();
    let upload = graph.upload(&x, &values, &[]);
    let copy = graph.dispatch((&x_to_y).into(), &[upload]);
    let derived = graph.run(&y_to_z);
    let read = graph.read(&z);
    assert!(!graph.is_derived(upload) && !graph.is_derived(copy));
    assert!(graph.is_derived(derived) && graph.is_derived(read));
    assert_eq!(edges(&graph, copy), [0]);
    assert_eq!(edges(&graph, derived), [1]);
    assert_eq!(edges(&graph, read), [2]);

    let output = device.execute_graph(&graph).unwrap();
    assert_eq!(output.readback::<f32>(read).unwrap(), values);
    // One declared edge (node 1) and two derived ones (nodes 2 and 3).
    assert_eq!(
        (output.report.declared_edges, output.report.derived_edges),
        (1, 2)
    );
}

#[test]
fn the_report_counts_edges_by_origin() {
    let device = device();
    let values = ramp(1.0);
    let x = floats(&device, "x", &vec![0.0; COUNT]);
    let y = floats(&device, "y", &vec![0.0; COUNT]);
    let copy = small::copy::call(COUNT as u32, &x, &y);

    let mut declared = StagedGraph::new();
    let upload = declared.upload(&x, &values, &[]);
    let dispatch = declared.dispatch((&copy).into(), &[upload]);
    // A duplicated declaration (`dispatch` twice) counts once.
    declared.readback(&y, &[dispatch, upload, dispatch]);
    let report = device.execute_graph(&declared).unwrap().report;
    assert_eq!((report.declared_edges, report.derived_edges), (3, 0));

    let mut derived = StagedGraph::new();
    derived.write(&x, &values);
    derived.run(&copy);
    derived.read(&y);
    let report = device.execute_graph(&derived).unwrap().report;
    assert_eq!((report.declared_edges, report.derived_edges), (0, 2));
}

// ---- indirect dispatch ----

#[test]
fn run_indirect_needs_independent_arguments() {
    let device = device();
    let state = pool_state(&device);
    // `active` is bound without `Independent`.
    let integrate = pool::integrate::call(
        0,
        &state.active,
        Independent(&state.settings),
        Independent(&state.particles),
    );
    let mut graph = StagedGraph::new();
    graph.run_indirect(&integrate, &state.args);
    match device.execute_graph(&graph).err().expect("rejected") {
        Error::IndirectBindingLength { parameter } => assert_eq!(parameter, "active"),
        other => panic!("expected IndirectBindingLength, got {other:?}"),
    }
}

// ---- limits are checked for the whole graph before any node compiles ----

#[test]
fn a_limit_error_in_a_later_node_precedes_the_compilation_of_an_earlier_one() {
    let device = device();
    let mut limits = device.limits().clone();
    // `copy` has 64 invocations per workgroup: 70 elements need 2 workgroups, 129 need 3.
    limits.max_compute_workgroups_per_dimension = 2;
    let device = device.with_limits(limits).unwrap();
    let small_input = floats(&device, "small in", &ramp(1.0));
    let small_output = floats(&device, "small out", &vec![0.0; COUNT]);
    let big_input = floats(&device, "big in", &vec![1.0; 129]);
    let big_output = floats(&device, "big out", &vec![0.0; 129]);
    let fits = small::copy::call(COUNT as u32, &small_input, &small_output);
    let too_big = small::copy::call(129, &big_input, &big_output);

    let mut graph = StagedGraph::new();
    graph.run(&fits);
    graph.run(&too_big);
    match device
        .execute_graph(&graph)
        .err()
        .expect("node 1 is over the limit")
    {
        Error::DeviceLimit {
            kernel,
            limit,
            requested,
            maximum,
        } => assert_eq!(
            (kernel.as_str(), limit, requested, maximum),
            ("small::copy", "max_compute_workgroups_per_dimension", 3, 2)
        ),
        other => panic!("expected DeviceLimit, got {other:?}"),
    }
    // Node 0 was never compiled, cached or run.
    let stats = device.pipeline_cache_stats();
    assert_eq!((stats.entries, stats.misses, stats.hits), (0, 0, 0));
    assert_eq!(
        device.read_typed_buffer(&small_output).unwrap(),
        vec![0.0; COUNT]
    );

    // The same holds for the explicit methods, and without the over-limit node the graph runs.
    let mut explicit = StagedGraph::new();
    explicit.dispatch((&fits).into(), &[]);
    explicit.dispatch((&too_big).into(), &[]);
    assert!(matches!(
        device.execute_graph(&explicit),
        Err(Error::DeviceLimit { requested: 3, .. })
    ));
    assert_eq!(device.pipeline_cache_stats().entries, 0);

    let mut ok = StagedGraph::new();
    ok.run(&fits);
    device.execute_graph(&ok).unwrap();
    assert_eq!(device.read_typed_buffer(&small_output).unwrap(), ramp(1.0));
}

#[test]
fn an_indirect_node_is_limit_checked_before_an_earlier_node_compiles() {
    let device = device();
    let mut limits = device.limits().clone();
    limits.max_compute_invocations_per_workgroup = 32;
    limits.max_compute_workgroup_size_x = 32;
    let device = device.with_limits(limits).unwrap();
    let state = pool_state(&device);
    // `prepare_dispatch` has one invocation per workgroup and passes; `integrate` has 64 and does not.
    let prepare = pool::prepare_dispatch::call(
        1,
        Independent(&state.count),
        Independent(&state.settings),
        Independent(&state.particles),
        Independent(&state.active),
        Independent(&state.args),
    );
    let integrate = pool::integrate::call(
        0,
        Independent(&state.active),
        Independent(&state.settings),
        Independent(&state.particles),
    );
    let mut graph = StagedGraph::new();
    graph.run(&prepare);
    graph.run_indirect(&integrate, &state.args);
    match device.execute_graph(&graph).err().expect("over the limit") {
        Error::DeviceLimit {
            limit,
            requested,
            maximum,
            ..
        } => assert_eq!(
            (limit, requested, maximum),
            ("max_compute_workgroup_size_x", 64, 32)
        ),
        other => panic!("expected DeviceLimit, got {other:?}"),
    }
    assert_eq!(device.pipeline_cache_stats().entries, 0);
}

// ---- the arguments buffer of an indirect dispatch ----

#[gust::gpu_module]
mod indirect {
    #[derive(Debug, PartialEq)]
    pub struct DispatchArgs {
        pub x: uint,
        pub y: uint,
        pub z: uint,
    }

    // Writes the arguments buffer it is bound to.
    #[allow(clippy::len_zero)]
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn rewrite(id: SV_DispatchThreadID, mut args: RWStructuredBuffer<DispatchArgs>) {
        if id.x == 0u32 && 0u32 < args.len() {
            args[0u32].x = 1u32;
            args[0u32].y = 1u32;
            args[0u32].z = 1u32;
        }
    }

    // Reads the arguments buffer it is bound to.
    #[allow(clippy::len_zero)]
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn inspect(
        id: SV_DispatchThreadID,
        args: StructuredBuffer<DispatchArgs>,
        mut seen: RWStructuredBuffer<uint>,
    ) {
        if id.x == 0u32 && 0u32 < args.len() && 0u32 < seen.len() {
            seen[0u32] = args[0u32].x + 100u32;
        }
    }
}

#[test]
fn an_indirect_node_whose_arguments_buffer_is_also_bound_read_write_is_rejected() {
    let device = device();
    let args = device
        .create_indirect_buffer("args", &[indirect::DispatchArgs { x: 1, y: 1, z: 1 }])
        .unwrap();
    let rewrite = indirect::rewrite::call(0, Independent(&args));

    // Explicit twin.
    let mut graph = StagedGraph::new();
    graph.dispatch_indirect((&rewrite).into(), &args, &[]);
    let error = device.execute_graph(&graph).err().expect("rejected");
    assert_aliased(error, "args");

    // Typed twin.
    let mut graph = StagedGraph::new();
    graph.run_indirect(&rewrite, &args);
    let error = device.execute_graph(&graph).err().expect("rejected");
    assert_aliased(error, "args");

    // Rejected before any wgpu object exists.
    assert_eq!(device.pipeline_cache_stats().entries, 0);
}

#[track_caller]
fn assert_aliased(error: Error, parameter: &str) {
    let text = error.to_string();
    match error {
        Error::AliasedBinding {
            kernel,
            first,
            second,
        } => assert_eq!(
            (kernel.as_str(), first.as_str(), second.as_str()),
            (
                "indirect::rewrite",
                parameter,
                "<indirect dispatch arguments>"
            )
        ),
        other => panic!("expected AliasedBinding, got {other:?}"),
    }
    assert_eq!(
        text,
        format!(
            "the dispatch-arguments buffer of kernel `indirect::rewrite` is also bound read-write by parameter `{parameter}`, which wgpu does not allow while the GPU reads it as indirect arguments (bind `{parameter}` read-only, or use a separate arguments buffer)"
        )
    );
}

#[test]
fn the_arguments_buffer_of_an_indirect_node_is_a_derived_edge_of_its_own() {
    let device = device();
    let values = [indirect::DispatchArgs { x: 1, y: 1, z: 1 }];
    let args = device.create_indirect_buffer("args", &values).unwrap();
    let x = floats(&device, "x", &vec![0.0; COUNT]);
    let y = floats(&device, "y", &vec![0.0; COUNT]);
    // `copy` does not bind `args`: the only link between the nodes below is the arguments buffer.
    let copy = small::copy::call(0, Independent(&x), Independent(&y));

    let mut graph = StagedGraph::new();
    let first = graph.write(&args, &values);
    let indirect_node = graph.run_indirect(&copy, &args);
    let second = graph.write(&args, &values);
    assert_eq!(edges(&graph, first), Vec::<usize>::new());
    // RAW: the indirect node reads `args` as its arguments, so it waits for the write.
    assert_eq!(edges(&graph, indirect_node), [0]);
    // WAR: a later write waits for the indirect node.
    assert_eq!(edges(&graph, second), [0, 1]);
}

#[test]
fn an_indirect_node_may_read_its_arguments_buffer() {
    let device = device();
    let args = device
        .create_indirect_buffer("args", &[indirect::DispatchArgs { x: 1, y: 1, z: 1 }])
        .unwrap();
    let seen = device.create_typed_buffer("seen", &[0_u32], GpuBufferAccess::ReadWrite);
    let inspect = indirect::inspect::call(0, Independent(&args), Independent(&seen));

    let mut graph = StagedGraph::new();
    graph.run_indirect(&inspect, &args);
    let output = device.execute_graph(&graph).unwrap();
    assert_eq!(output.report.indirect_dispatches, 1);
    assert_eq!(device.read_typed_buffer(&seen).unwrap(), [101]);
}

#[test]
fn an_indirect_node_ignores_the_workgroup_count_limit_but_not_the_others() {
    let device = device();
    let mut limits = device.limits().clone();
    // No direct launch fits, however small; the host never sees an indirect count.
    limits.max_compute_workgroups_per_dimension = 0;
    let device = device.with_limits(limits.clone()).unwrap();
    let args = device
        .create_indirect_buffer("args", &[indirect::DispatchArgs { x: 1, y: 1, z: 1 }])
        .unwrap();
    let seen = device.create_typed_buffer("seen", &[0_u32], GpuBufferAccess::ReadWrite);

    // The same kernel launched directly still fails the count check: one element needs one workgroup.
    let direct = indirect::inspect::call(1, Independent(&args), Independent(&seen));
    let mut graph = StagedGraph::new();
    graph.run(&direct);
    match device.execute_graph(&graph).err().expect("over the limit") {
        Error::DeviceLimit {
            limit,
            requested,
            maximum,
            ..
        } => assert_eq!(
            (limit, requested, maximum),
            ("max_compute_workgroups_per_dimension", 1, 0)
        ),
        other => panic!("expected DeviceLimit, got {other:?}"),
    }

    // Indirect launches, explicit and typed, run.
    let inspect = indirect::inspect::call(0, Independent(&args), Independent(&seen));
    let mut graph = StagedGraph::new();
    graph.run_indirect(&inspect, &args);
    let output = device.execute_graph(&graph).unwrap();
    assert_eq!(output.report.indirect_dispatches, 1);
    assert_eq!(device.read_typed_buffer(&seen).unwrap(), [101]);
    let mut explicit = StagedGraph::new();
    explicit.dispatch_indirect((&inspect).into(), &args, &[]);
    let output = device.execute_graph(&explicit).unwrap();
    assert_eq!(output.report.indirect_dispatches, 1);
    assert_eq!(output.report.dispatches, 1);

    // Every other limit still applies to an indirect node: `inspect` binds two storage buffers.
    limits.max_storage_buffers_per_shader_stage = 1;
    let device = device.with_limits(limits).unwrap();
    let mut graph = StagedGraph::new();
    graph.run_indirect(&inspect, &args);
    match device.execute_graph(&graph).err().expect("over the limit") {
        Error::DeviceLimit {
            limit,
            requested,
            maximum,
            ..
        } => assert_eq!(
            (limit, requested, maximum),
            ("max_storage_buffers_per_shader_stage", 2, 1)
        ),
        other => panic!("expected DeviceLimit, got {other:?}"),
    }
}

// ---- uniforms in a graph (m6.u1) ----
//
// `StagedGraph::write_uniform` is an encoded copy from a staging buffer, so an update falls between the dispatches of
// one submission. The kernel is hand-written for the `edges` and report checks; `uniforms.rs` runs `#[gpu_module]`
// kernels in a graph the same way.

use gust::{
    Access, Arg, Call, GpuPod, Kernel, KernelDescriptor, ParameterDescriptor, ParameterKind,
    Resource, ResourceBinding, ResourceLayout,
};

const fn graph_storage(name: &'static str, access: Access, binding: u32) -> ParameterDescriptor {
    ParameterDescriptor {
        name,
        rust_type: "float",
        kind: ParameterKind::Storage,
        access,
        resource_layout: Some(ResourceLayout::storage_v1(<f32 as GpuPod>::LAYOUT)),
        binding: Some(ResourceBinding { group: 0, binding }),
    }
}

/// `shade(src: RO, factor: Uniform<f32>, dst: RW)`: `dst[i] = src[i] * factor`.
struct Shade;
impl Kernel for Shade {
    const DESCRIPTOR: KernelDescriptor = KernelDescriptor::new(
        "graph_tests",
        "shade",
        "graph_shade",
        [64, 1, 1],
        &[
            graph_storage("src", Access::ReadOnly, 0),
            ParameterDescriptor {
                name: "factor",
                rust_type: "float",
                kind: ParameterKind::Uniform,
                access: Access::ReadOnly,
                resource_layout: Some(ResourceLayout::std140_v1(<f32 as GpuPod>::LAYOUT)),
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: 1,
                }),
            },
            graph_storage("dst", Access::ReadWrite, 2),
        ],
        r#"
[[vk::binding(0, 0)]] StructuredBuffer<float> src;
[[vk::binding(1, 0)]] ConstantBuffer<float> factor;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> dst;
[shader("compute")] [numthreads(64, 1, 1)]
void graph_shade(uint3 id : SV_DispatchThreadID)
{
    uint n; uint stride; dst.GetDimensions(n, stride);
    if (id.x < n) { dst[id.x] = src[id.x] * factor; }
}
"#,
    );
}
impl gust::target::GpuKernel for Shade {}

fn shade_call<'r, R0, R1, R2>(
    __gust_element_count: u32,
    src: impl Into<Arg<'r, R0>>,
    factor: impl Into<Arg<'r, R1>>,
    dst: impl Into<Arg<'r, R2>>,
) -> Call<'r, <R0 as Resource>::Device, Shade, 3>
where
    R0: Resource<Element = f32, Kind = gust::kind::Storage> + 'r,
    R1: Resource<Element = f32, Device = <R0 as Resource>::Device, Kind = gust::kind::Uniform> + 'r,
    R2: Resource<Element = f32, Device = <R0 as Resource>::Device, Kind = gust::kind::Storage> + 'r,
{
    Call::new(
        __gust_element_count,
        [
            src.into().bind(Access::ReadOnly),
            factor.into().bind(Access::ReadOnly),
            dst.into().bind(Access::ReadWrite),
        ],
    )
}

fn scaled(factor: f32) -> Vec<f32> {
    ramp(1.0).iter().map(|x| x * factor).collect()
}

#[test]
fn two_uniform_updates_between_dispatches_are_observed_in_order() {
    let device = device();
    let src = floats(&device, "src", &ramp(1.0));
    let (out_one, out_two, out_three) = (
        floats(&device, "one", &vec![0.0; COUNT]),
        floats(&device, "two", &vec![0.0; COUNT]),
        floats(&device, "three", &vec![0.0; COUNT]),
    );
    let factor = device.create_uniform("factor", &1.0f32).unwrap();
    let (two, three) = (2.0f32, 3.0f32);
    let (call_one, call_two, call_three) = (
        shade_call(COUNT as u32, &src, &factor, &out_one),
        shade_call(COUNT as u32, &src, &factor, &out_two),
        shade_call(COUNT as u32, &src, &factor, &out_three),
    );

    let mut graph = StagedGraph::new();
    let first = graph.run(&call_one);
    let set_two = graph.write_uniform(&factor, &two);
    let second = graph.run(&call_two);
    let set_three = graph.write_uniform(&factor, &three);
    let third = graph.run(&call_three);
    // `Debug` (W2): counts and ids, never a raw wgpu handle.
    assert_eq!(format!("{graph:?}"), "StagedGraph { nodes: 5 }");
    assert_eq!(
        format!("{factor:?}"),
        format!("GpuUniform {{ device_id: {}, .. }}", device.id())
    );
    assert_eq!(
        format!("{src:?}"),
        format!(
            "GpuBuffer {{ len: {COUNT}, access: ReadWrite, device_id: {}, indirect: false, .. }}",
            device.id()
        )
    );
    assert!(
        format!("{device:?}")
            .starts_with(&format!("HeadlessDevice {{ id: {}, adapter: ", device.id()))
    );
    // Written down before the first run. A write of the uniform waits for the dispatches that read the old value
    // (WAR) and the one before it (WAW); a dispatch waits for the write it reads.
    assert_eq!(edges(&graph, first), Vec::<usize>::new());
    assert_eq!(edges(&graph, set_two), [0]);
    assert_eq!(edges(&graph, second), [1]);
    assert_eq!(edges(&graph, set_three), [1, 2]);
    assert_eq!(edges(&graph, third), [3]);
    assert!(graph.is_derived(set_two));

    let output = device.execute_graph(&graph).unwrap();
    // All three dispatches are in one command buffer, so each saw the value its position implies: with
    // `Queue::write_buffer` they would all have seen the last one.
    assert_eq!(device.read_typed_buffer(&out_one).unwrap(), scaled(1.0));
    assert_eq!(device.read_typed_buffer(&out_two).unwrap(), scaled(2.0));
    assert_eq!(device.read_typed_buffer(&out_three).unwrap(), scaled(3.0));
    assert_eq!(output.report.dispatches, 3);
    assert_eq!(output.report.upload_bytes, 8);
    assert_eq!(output.report.derived_edges, 5);

    // The uniform keeps the last value for the next submission.
    let mut again = StagedGraph::new();
    again.run(&call_one);
    device.execute_graph(&again).unwrap();
    assert_eq!(device.read_typed_buffer(&out_one).unwrap(), scaled(3.0));
}

#[test]
fn a_dispatch_that_reads_a_uniform_must_declare_its_edge_to_the_write() {
    let device = device();
    let src = floats(&device, "src", &ramp(1.0));
    let out = floats(&device, "out", &vec![0.0; COUNT]);
    let factor = device.create_uniform("factor", &1.0f32).unwrap();
    let four = 4.0f32;
    let call = shade_call(COUNT as u32, &src, &factor, &out);

    // The write is a write of the uniform for the hazard check, like an upload of a buffer: an explicit dispatch
    // that reads the uniform after it, with no edge, is rejected.
    let mut missing = StagedGraph::new();
    missing.write_uniform(&factor, &four);
    missing.dispatch((&call).into(), &[]);
    match device
        .execute_graph(&missing)
        .err()
        .expect("the edge is missing")
    {
        Error::GraphMissingDependency { node, producer } => assert_eq!((node, producer), (1, 0)),
        other => panic!("expected GraphMissingDependency, got {other:?}"),
    }
    assert_eq!(device.read_typed_buffer(&out).unwrap(), vec![0.0; COUNT]);

    // The control declares it.
    let mut declared = StagedGraph::new();
    let write = declared.write_uniform(&factor, &four);
    declared.dispatch((&call).into(), &[write]);
    device.execute_graph(&declared).unwrap();
    assert_eq!(device.read_typed_buffer(&out).unwrap(), scaled(4.0));
}

#[test]
fn a_uniform_of_another_device_cannot_be_written_in_a_graph() {
    let device = device();
    let other = HeadlessDevice::new().expect("a second device");
    let foreign = other.create_uniform("foreign", &1.0f32).unwrap();
    let value = 2.0f32;
    let mut graph = StagedGraph::new();
    graph.write_uniform(&foreign, &value);
    assert!(matches!(
        device.execute_graph(&graph).err().expect("foreign uniform"),
        Error::ForeignPersistentBuffer
    ));
}

// ---- opt-in GPU timestamps (m10.ts2, D194) ----
//
// `execute_graph_timed` runs the same graph and reports `gpu_elapsed` from the first dispatch pass's begin to the end
// of the last. The value is `Some` exactly when the adapter has timestamp queries and a dispatch pass ran.

#[gust::gpu_module]
mod stamped {
    #[kernel(shape = pixel)]
    pub fn pixel_copy(
        id: SV_DispatchThreadID,
        src: StructuredBuffer<u32>,
        mut dst: RWStructuredBuffer<u32>,
    ) {
        let i = id.x;
        dst[i] = src[i] + 1u32;
    }
}

/// What a timed graph that ran exactly `passes` dispatch passes must report on this device: `Some` only where the
/// adapter has timestamp queries and a pass ran, and then a duration that is neither a wrapped difference nor an
/// absolute clock reading (a wrong stamp slot would give one of those).
#[track_caller]
fn assert_stamped(device: &HeadlessDevice, report: &gust_wgpu::GraphReport, passes: usize) {
    assert_eq!(report.dispatches, passes);
    let adapter = device.adapter_info().name.clone();
    assert_eq!(
        report.gpu_elapsed.is_some(),
        device.supports_gpu_timestamps() && passes > 0,
        "adapter {adapter:?}"
    );
    if let Some(elapsed) = report.gpu_elapsed {
        assert!(
            elapsed < std::time::Duration::from_secs(10),
            "implausible GPU time {elapsed:?} on {adapter:?}"
        );
    }
}

#[test]
fn a_timed_graph_returns_the_outputs_of_an_untimed_one() {
    let device = device();
    let run = |timed: bool| {
        let input = floats(&device, "in", &ramp(1.0));
        let middle = floats(&device, "mid", &vec![0.0; COUNT]);
        let output = floats(&device, "out", &vec![0.0; COUNT]);
        let first = small::copy::call(COUNT as u32, &input, &middle);
        let second = small::copy::call(COUNT as u32, &middle, &output);
        let mut graph = StagedGraph::new();
        let host = ramp(1.0);
        graph.write(&input, &host);
        graph.run(&first);
        graph.run(&second);
        let read = graph.read(&output);
        let result = if timed {
            device.execute_graph_timed(&graph)
        } else {
            device.execute_graph(&graph)
        }
        .unwrap();
        let values = result.readback::<f32>(read).unwrap();
        (result.report, values)
    };
    let (untimed, untimed_values) = run(false);
    let (timed, timed_values) = run(true);
    assert_eq!(untimed_values, ramp(1.0));
    assert_eq!(timed_values, untimed_values);
    assert_eq!(untimed.gpu_elapsed, None);
    assert_stamped(&device, &timed, 2);
    // Evidence line for the gate log (`-- --nocapture`): whether the `Some` path ran on this adapter.
    println!(
        "ts2 evidence: adapter={:?} supports_gpu_timestamps={} gpu_elapsed={:?}",
        device.adapter_info().name,
        device.supports_gpu_timestamps(),
        timed.gpu_elapsed
    );
    // Every field except the two elapsed times is deterministic and must agree.
    let strip = |report: &gust_wgpu::GraphReport| {
        let mut report = report.clone();
        report.host_elapsed = std::time::Duration::ZERO;
        report.gpu_elapsed = None;
        report
    };
    assert_eq!(strip(&timed), strip(&untimed));
}

#[test]
fn a_timed_graph_with_an_indirect_pass_first_or_last_is_bracketed() {
    let device = device();
    let args = device
        .create_indirect_buffer("args", &[indirect::DispatchArgs { x: 1, y: 1, z: 1 }])
        .unwrap();
    let seen = device.create_typed_buffer("seen", &[0_u32], GpuBufferAccess::ReadWrite);
    let inspect = indirect::inspect::call(0, Independent(&args), Independent(&seen));
    let input = floats(&device, "in", &ramp(1.0));
    let output = floats(&device, "out", &vec![0.0; COUNT]);
    let copy = small::copy::call(COUNT as u32, &input, &output);

    // Indirect then direct, direct then indirect, indirect twice; then one indirect pass alone (first and last are the same pass).
    for order in [[true, false], [false, true], [true, true]] {
        device.write_typed_buffer(&seen, &[0_u32]).unwrap();
        let mut graph = StagedGraph::new();
        for indirect_node in order {
            if indirect_node {
                graph.run_indirect(&inspect, &args);
            } else {
                graph.run(&copy);
            }
        }
        let result = device.execute_graph_timed(&graph).unwrap();
        assert_eq!(
            result.report.indirect_dispatches,
            order.iter().filter(|node| **node).count()
        );
        assert_stamped(&device, &result.report, 2);
        assert_eq!(device.read_typed_buffer(&seen).unwrap(), [101]);
    }

    let mut only = StagedGraph::new();
    only.run_indirect(&inspect, &args);
    let result = device.execute_graph_timed(&only).unwrap();
    assert_eq!(result.report.indirect_dispatches, 1);
    assert_stamped(&device, &result.report, 1);
}

#[test]
fn a_timed_graph_with_pixel_passes_first_and_last_is_bracketed() {
    let device = device();
    let src = device.create_typed_buffer("src", &[1_u32, 2, 3, 4], GpuBufferAccess::ReadWrite);
    let mid = device.create_typed_buffer("mid", &[0_u32; 4], GpuBufferAccess::ReadWrite);
    let dst = device.create_typed_buffer("dst", &[0_u32; 4], GpuBufferAccess::ReadWrite);
    let first = stamped::pixel_copy::call(4, &src, &mid);
    let second = stamped::pixel_copy::call(4, &mid, &dst);
    let mut graph = StagedGraph::new();
    graph.run(&first);
    graph.run(&second);
    let result = device.execute_graph_timed(&graph).unwrap();
    assert_eq!(result.report.dispatches, 2);
    assert_stamped(&device, &result.report, 2);
    assert_eq!(device.read_typed_buffer(&dst).unwrap(), [3, 4, 5, 6]);
}

#[test]
fn a_timed_graph_without_a_dispatch_pass_reports_none() {
    let device = device();
    let buffer = device.create_typed_buffer("values", &[0_u32; 4], GpuBufferAccess::ReadWrite);
    let mut graph = StagedGraph::new();
    graph.write(&buffer, &[1_u32, 2, 3, 4]);
    let read = graph.read(&buffer);
    let result = device.execute_graph_timed(&graph).unwrap();
    assert_eq!(result.readback::<u32>(read).unwrap(), [1, 2, 3, 4]);
    assert_eq!(result.report.gpu_elapsed, None);

    // A zero-element dispatch runs no pass either.
    let input = floats(&device, "in", &[]);
    let out = floats(&device, "out", &[]);
    let empty = small::copy::call(0, &input, &out);
    let mut graph = StagedGraph::new();
    graph.run(&empty);
    let result = device.execute_graph_timed(&graph).unwrap();
    assert_eq!(result.report.dispatches, 0);
    assert_eq!(result.report.gpu_elapsed, None);
}

#[test]
fn zero_element_dispatches_at_either_end_do_not_carry_the_stamps() {
    let device = device();
    let input = floats(&device, "in", &ramp(1.0));
    let output = floats(&device, "out", &vec![0.0; COUNT]);
    let empty_in = floats(&device, "e-in", &[]);
    let empty_out = floats(&device, "e-out", &[]);
    let empty = small::copy::call(0, &empty_in, &empty_out);
    let copy = small::copy::call(COUNT as u32, &input, &output);
    let mut graph = StagedGraph::new();
    graph.run(&empty);
    graph.run(&copy);
    graph.run(&empty);
    let result = device.execute_graph_timed(&graph).unwrap();
    assert_eq!(result.report.dispatches, 1);
    assert_stamped(&device, &result.report, 1);
    assert_eq!(device.read_typed_buffer(&output).unwrap(), ramp(1.0));
}

#[test]
fn timed_graphs_and_timed_batches_share_the_profiler_without_deadlock() {
    const ROUNDS: usize = 20;
    let device = device();
    let input = floats(&device, "in", &ramp(1.0));
    let graph_out = floats(&device, "g-out", &vec![0.0; COUNT]);
    let batch_out = floats(&device, "b-out", &vec![0.0; COUNT]);
    let graph_call = small::copy::call(COUNT as u32, &input, &graph_out);
    let batch_call = small::copy::call(COUNT as u32, &input, &batch_out);
    // Both threads meet at a barrier, so the loops start together and overlap; each does a bounded number of rounds
    // and the scope joins them; no sleep, no timeout.
    let start = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            start.wait();
            for _ in 0..ROUNDS {
                let mut graph = StagedGraph::new();
                graph.run(&graph_call);
                let result = device.execute_graph_timed(&graph).unwrap();
                assert_stamped(&device, &result.report, 1);
            }
        });
        scope.spawn(|| {
            start.wait();
            for _ in 0..ROUNDS {
                let timing = device
                    .dispatch_batch_timed(&[(&batch_call).into()])
                    .unwrap();
                assert_eq!(
                    timing.gpu_elapsed.is_some(),
                    device.supports_gpu_timestamps()
                );
            }
        });
    });
    assert_eq!(device.read_typed_buffer(&graph_out).unwrap(), ramp(1.0));
    assert_eq!(device.read_typed_buffer(&batch_out).unwrap(), ramp(1.0));
}

// ---- the hazard check on graphs of 40 or more nodes (m14.rtfix1) ----

/// Nodes of the long graphs below: more than the 40 the check was measured at.
const LONG: usize = 44;

#[test]
fn a_graph_of_forty_four_derived_nodes_validates_and_runs() {
    let device = device();
    let input = ramp(3.0);
    let chain: Vec<GpuBuffer<f32>> = (0..=LONG)
        .map(|index| floats(&device, &format!("chain {index}"), &vec![0.0; COUNT]))
        .collect();
    let calls: Vec<_> = (0..LONG)
        .map(|index| small::copy::call(COUNT as u32, &chain[index], &chain[index + 1]))
        .collect();

    let mut graph = StagedGraph::new();
    graph.write(&chain[0], &input);
    for call in &calls {
        graph.run(call);
    }
    let read = graph.read(&chain[LONG]);
    assert_eq!(graph.len(), LONG + 2);

    // Twice: the second execution reuses what the first kept, and must see the same.
    for _ in 0..2 {
        let output = device.execute_graph(&graph).unwrap();
        assert_eq!(output.report.dispatches, LONG);
        assert_eq!(output.readback::<f32>(read).unwrap(), input);
    }
}

/// `LONG - 2` writers of one buffer, each depending on the one before, and a reader of that buffer that declares
/// `dependencies` (given the writers' ids).
fn reader_after_a_long_writer_chain(
    dependencies: impl Fn(&[NodeId]) -> Vec<NodeId>,
) -> Result<(), Error> {
    let device = device();
    let source = floats(&device, "source", &ramp(1.0));
    let shared = floats(&device, "shared", &vec![0.0; COUNT]);
    let out = floats(&device, "out", &vec![0.0; COUNT]);
    let write = small::copy::call(COUNT as u32, &source, &shared);
    let read = small::copy::call(COUNT as u32, &shared, &out);

    let mut graph = StagedGraph::new();
    let mut writers: Vec<NodeId> = Vec::new();
    for _ in 0..LONG - 2 {
        let previous: Vec<NodeId> = writers.last().copied().into_iter().collect();
        writers.push(graph.dispatch((&write).into(), &previous));
    }
    let reader = graph.dispatch((&read).into(), &dependencies(&writers));
    assert_eq!(reader.index(), LONG - 2);
    device.execute_graph(&graph).map(|_| {
        assert_eq!(device.read_typed_buffer(&out).unwrap(), ramp(1.0));
    })
}

#[test]
fn a_missing_edge_names_the_lowest_producer_the_node_does_not_follow() {
    // No edge at all: node 0 is the lowest of the 42 writers.
    match reader_after_a_long_writer_chain(|_| Vec::new()).expect_err("no edge") {
        Error::GraphMissingDependency { node, producer } => {
            assert_eq!((node, producer), (LONG - 2, 0));
        }
        other => panic!("expected GraphMissingDependency, got {other:?}"),
    }
    // An edge to writer 20 covers writers 0 to 20 (each depends on the one before): 21 is the lowest left.
    match reader_after_a_long_writer_chain(|writers| vec![writers[20]]).expect_err("a short edge") {
        Error::GraphMissingDependency { node, producer } => {
            assert_eq!((node, producer), (LONG - 2, 21));
        }
        other => panic!("expected GraphMissingDependency, got {other:?}"),
    }
    // An edge to the last writer covers all of them: the graph is valid.
    reader_after_a_long_writer_chain(|writers| vec![*writers.last().unwrap()]).unwrap();
}

/// `LONG - 3` readers of one uploaded buffer, then an overwrite of it that declares an edge to the upload and to
/// every reader except `skipped` (when given).
fn overwrite_after_many_readers(skipped: Option<usize>) -> Result<(), Error> {
    let device = device();
    let (first, second) = (ramp(1.0), ramp(500.0));
    let shared = floats(&device, "shared", &vec![0.0; COUNT]);
    let outs: Vec<GpuBuffer<f32>> = (0..LONG - 3)
        .map(|index| floats(&device, &format!("out {index}"), &vec![0.0; COUNT]))
        .collect();
    let calls: Vec<_> = outs
        .iter()
        .map(|out| small::copy::call(COUNT as u32, &shared, out))
        .collect();

    let mut graph = StagedGraph::new();
    let upload = graph.upload(&shared, &first, &[]);
    let readers: Vec<NodeId> = calls
        .iter()
        .map(|call| graph.dispatch(call.into(), &[upload]))
        .collect();
    let mut edges: Vec<NodeId> = vec![upload];
    edges.extend(
        readers
            .iter()
            .filter(|reader| Some(reader.index()) != skipped),
    );
    let overwrite = graph.upload(&shared, &second, &edges);
    assert_eq!(overwrite.index(), LONG - 2);
    device.execute_graph(&graph).map(|_| {
        for out in &outs {
            assert_eq!(device.read_typed_buffer(out).unwrap(), first);
        }
        assert_eq!(device.read_typed_buffer(&shared).unwrap(), second);
    })
}

#[test]
fn an_overwrite_must_follow_every_reader_of_a_long_graph() {
    overwrite_after_many_readers(None).expect("every reader is an edge");
    // The reader at node index 17 is the only one left out; the readers before it are covered.
    match overwrite_after_many_readers(Some(17))
        .expect_err("the reader at node index 17 is not an ancestor")
    {
        Error::GraphMissingDependency { node, producer } => {
            assert_eq!((node, producer), (LONG - 2, 17));
        }
        other => panic!("expected GraphMissingDependency, got {other:?}"),
    }
}

// ---- the staging buffers of readbacks come from the device's pool (m14.rtfix1) ----

#[test]
fn a_graph_takes_its_readback_staging_from_the_pool_and_gives_it_back() {
    let device = device();
    let input = ramp(2.0);
    let x = floats(&device, "x", &vec![0.0; COUNT]);
    let y = floats(&device, "y", &vec![0.0; COUNT]);
    let copy = small::copy::call(COUNT as u32, &x, &y);
    let mut graph = StagedGraph::new();
    graph.write(&x, &input);
    graph.run(&copy);
    let read = graph.read(&y);

    // The first execution finds the pool empty: one staging buffer for the readback (the upload uses a fresh one).
    let before = device.buffer_pool_stats();
    let output = device.execute_graph(&graph).unwrap();
    assert_eq!(output.readback::<f32>(read).unwrap(), input);
    let first = device.buffer_pool_stats();
    assert_eq!(first.misses - before.misses, 1);
    assert_eq!(first.hits - before.hits, 0);
    assert_eq!(first.outstanding, before.outstanding);
    assert_eq!(first.free_buffers - before.free_buffers, 1);
    assert_eq!(first.discarded_mapped, before.discarded_mapped);

    // Every later execution reuses it, and the reused readback buffer was unmapped on return.
    for _ in 0..3 {
        let output = device.execute_graph(&graph).unwrap();
        assert_eq!(output.readback::<f32>(read).unwrap(), input);
    }
    let later = device.buffer_pool_stats();
    assert_eq!(later.misses, first.misses);
    assert_eq!(later.hits - first.hits, 3);
    assert_eq!(later.free_buffers, first.free_buffers);
    assert_eq!(later.outstanding, before.outstanding);
    assert_eq!(later.discarded_mapped, before.discarded_mapped);
}

#[test]
fn a_readback_above_the_staging_ceiling_is_a_fresh_buffer_and_leaves_the_pool_alone() {
    // 17_000_000 floats are 68_000_000 bytes: the next power of two, 128 MiB, is above the 64 MiB ceiling.
    const HUGE: usize = 17_000_000;
    let device = device();
    let values: Vec<f32> = (0..HUGE).map(|index| (index % 1000) as f32).collect();
    let x = floats(&device, "x", &vec![0.0; HUGE]);
    let before = device.buffer_pool_stats();
    let mut graph = StagedGraph::new();
    graph.write(&x, &values);
    let read = graph.read(&x);
    let output = device.execute_graph(&graph).unwrap();
    assert_eq!(output.readback::<f32>(read).unwrap(), values);
    assert_eq!(device.buffer_pool_stats(), before);
}

#[test]
fn a_readback_inside_the_staging_ceiling_is_pooled_and_returned() {
    // 10_000_000 floats are 40_000_000 bytes: the next power of two, 64 MiB, is at the ceiling and still pooled.
    const BIG: usize = 10_000_000;
    const CLASS: u64 = 64 << 20;
    let device = device();
    let values: Vec<f32> = (0..BIG).map(|index| (index % 1000) as f32).collect();
    let x = floats(&device, "x", &vec![0.0; BIG]);
    let before = device.buffer_pool_stats();
    let mut graph = StagedGraph::new();
    graph.write(&x, &values);
    let read = graph.read(&x);
    let output = device.execute_graph(&graph).unwrap();
    assert_eq!(output.readback::<f32>(read).unwrap(), values);
    let mut expected = before;
    expected.misses += 1;
    expected.returned += 1;
    expected.free_buffers += 1;
    expected.free_bytes += CLASS;
    assert_eq!(device.buffer_pool_stats(), expected);
}

#[test]
fn a_pooled_readback_buffer_larger_than_the_data_is_read_over_the_data_only() {
    // 70_000 floats are 280_000 bytes, past the exact classes: the readback staging buffer is the next power of two.
    const BIG: usize = 70_000;
    let device = device();
    let first: Vec<f32> = (0..BIG).map(|index| index as f32).collect();
    let second: Vec<f32> = (0..BIG).map(|index| (BIG - index) as f32).collect();
    let x = floats(&device, "x", &vec![0.0; BIG]);

    for values in [&first, &second, &first] {
        let mut graph = StagedGraph::new();
        graph.write(&x, values);
        let read = graph.read(&x);
        let output = device.execute_graph(&graph).unwrap();
        assert_eq!(output.readback::<f32>(read).unwrap(), *values);
        assert_eq!(output.report.upload_bytes, (BIG * 4) as u64);
        assert_eq!(output.report.readback_bytes, (BIG * 4) as u64);
    }
    // The three graphs shared one readback buffer.
    let stats = device.buffer_pool_stats();
    assert_eq!((stats.misses, stats.hits), (1, 2));
}
