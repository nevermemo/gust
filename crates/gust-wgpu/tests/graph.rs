//! `StagedGraph` derives dependency edges from typed calls (roadmap M2, m2.e).
//!
//! The typed methods `write`, `run`, `run_indirect` and `read` record the same nodes as their explicit
//! twins (`upload`, `dispatch`, `dispatch_indirect`, `readback`) and compute the edges themselves: for each
//! buffer a node touches, the latest earlier writer, plus, when the node writes, every reader since that
//! writer. The expected edge sets below were written down before the derivation ran (spike S4) and are
//! not adjusted to its output. Kernels are macro-generated; the calls are the generated `call` functions.
//! Expectations come from independent host references, never from running a shader shadow body.

use gust::Independent;
use gust_wgpu::{Error, GpuBuffer, GpuBufferAccess, HeadlessDevice, NodeId, StagedGraph};

// The two shapes of the staged-graph and component-pool examples. Kernel signatures (parameter types,
// access, order) and bodies are those of `examples/staged-graph` and `examples/component-pool`.
#[gust::gpu]
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

#[gust::gpu]
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

fn device() -> HeadlessDevice {
    HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip")
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

#[gust::gpu]
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

#[gust::gpu]
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
// one submission. The kernel is hand-written for the `edges` and report checks; `uniforms.rs` runs `#[gpu]` kernels in a
// graph the same way.

use gust::{
    Access, Arg, Call, GpuPod, Kernel, KernelDescriptor, ParameterDescriptor, ParameterKind,
    Resource, ResourceBinding, ResourceLayout, SourceOrigin,
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
    const DESCRIPTOR: KernelDescriptor = KernelDescriptor {
        module: "graph_tests",
        name: "shade",
        entry_point: "graph_shade",
        shape: gust::Shape::Grid,
        compatible_shapes: &[gust::Shape::Grid],
        workgroup_size: [64, 1, 1],
        parameters: &[
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
        slang_source: r#"
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
        source: SourceOrigin::UNKNOWN,
        artifacts: &[],
    };
}

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
