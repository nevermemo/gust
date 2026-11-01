//! Staged execution graph (T07) on typed calls.
//!
//! One CPU settings update is uploaded, GPU stage A scales and shifts a resident
//! sample buffer into a resident intermediate, GPU stage B reduces every block of
//! `BLOCK` intermediates into one summary, and only the small summary buffer is
//! read back. The graph derives the dependencies from the buffers each typed node
//! touches (`write`, `run`, `read`); the runtime still checks the whole graph. The
//! rejection tests keep the explicit, declared-edge API where they test it.

use std::{path::PathBuf, time::Instant};

use gust::{Independent, KernelDescriptor, gpu};
#[cfg(test)]
use gust_wgpu::{BufferBinding, BufferDispatch};
use gust_wgpu::{GpuBufferAccess, GraphReport, HeadlessDevice, StagedGraph, render_wgpu_source};

/// Intermediates reduced into one summary element.
pub const BLOCK: u32 = 8;

#[gpu]
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

    // Stage A: one element per sample. `settings` is a one-element buffer, bound
    // with an independent length and guarded by its own `.len()` (the dialect has
    // no `.is_empty()`, hence the lint allowance).
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

    // Stage B: one element per block. Reads `block` consecutive intermediates and
    // writes one summary; both `intermediate` and `settings` have independent lengths.
    // The block is unrolled because loops are not in the dialect yet.
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

pub use staged::{Settings, Summary};

pub fn make_samples(count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| ((index % 97) as f32) * 0.25 - 6.0)
        .collect()
}

pub fn summary_count(sample_count: usize) -> usize {
    sample_count.div_ceil(BLOCK as usize)
}

/// Independent host reference: ordinary Rust over the same inputs.
pub fn run_cpu(samples: &[f32], settings: &Settings) -> Vec<Summary> {
    samples
        .chunks(settings.block as usize)
        .map(|block| {
            let mut summary = Summary {
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

/// Resident GPU state that survives across settings updates.
pub struct ResidentState {
    pub samples: gust_wgpu::GpuBuffer<f32>,
    pub settings: gust_wgpu::GpuBuffer<Settings>,
    pub intermediate: gust_wgpu::GpuBuffer<f32>,
    pub summaries: gust_wgpu::GpuBuffer<Summary>,
}

impl ResidentState {
    pub fn new(device: &HeadlessDevice, samples: &[f32]) -> Self {
        let summaries = summary_count(samples.len());
        Self {
            samples: device.create_typed_buffer(
                "graph samples",
                samples,
                GpuBufferAccess::ReadOnly,
            ),
            settings: device.create_typed_buffer(
                "graph settings",
                &[Settings {
                    scale: 0.0,
                    offset: 0.0,
                    threshold: 0.0,
                    block: BLOCK,
                }],
                GpuBufferAccess::ReadOnly,
            ),
            intermediate: device.create_typed_buffer(
                "graph intermediate",
                &vec![0.0f32; samples.len()],
                GpuBufferAccess::ReadWrite,
            ),
            summaries: device.create_typed_buffer(
                "graph summaries",
                &vec![
                    Summary {
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
}

/// Per node of a graph, the indices of the nodes it depends on.
pub type Edges = Vec<Vec<usize>>;

/// [`run_graph_traced`] without the edges.
#[cfg(test)]
pub fn run_graph(
    device: &HeadlessDevice,
    state: &ResidentState,
    settings: &[Settings; 1],
) -> Result<(Vec<Summary>, GraphReport), gust_wgpu::Error> {
    let (summaries, report, _edges) = run_graph_traced(device, state, settings)?;
    Ok((summaries, report))
}

/// Build and run the four-node graph for one settings update: write the settings, run both stages,
/// read the summaries back. Returns the summaries, the report, and the dependency edges the graph
/// derived, per node (node indices, ascending).
///
/// The typed calls are locals built before the graph, which borrows them. Nothing is declared by the
/// host: `write`, `run` and `read` derive each node's edges from the buffers it touches and the access
/// the kernel declares (`Independent` only lifts the shared-length rule, not the access).
pub fn run_graph_traced(
    device: &HeadlessDevice,
    state: &ResidentState,
    settings: &[Settings; 1],
) -> Result<(Vec<Summary>, GraphReport, Edges), gust_wgpu::Error> {
    let sample_count = state.samples.len() as u32;
    let summary_count = state.summaries.len() as u32;
    // `settings` has one element and `intermediate` one per sample, not one per summary: `Independent`
    // is the call-site opt-in for a length that differs from the element count.
    let transform = staged::transform::call(
        sample_count,
        &state.samples,
        Independent(&state.settings),
        &state.intermediate,
    );
    let summarize = staged::summarize::call(
        summary_count,
        Independent(&state.intermediate),
        Independent(&state.settings),
        &state.summaries,
    );

    let mut graph = StagedGraph::new();
    let write = graph.write(&state.settings, settings);
    let stage_a = graph.run(&transform);
    let stage_b = graph.run(&summarize);
    let read = graph.read(&state.summaries);
    let edges = [write, stage_a, stage_b, read]
        .map(|node| graph.dependency_indices(node))
        .to_vec();

    let output = device.execute_graph(&graph)?;
    Ok((output.readback::<Summary>(read)?, output.report, edges))
}

pub fn descriptors() -> [&'static KernelDescriptor; 2] {
    [
        &staged::transform::DESCRIPTOR,
        &staged::summarize::DESCRIPTOR,
    ]
}

pub fn assert_close(actual: &[Summary], expected: &[Summary]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let close = |a: f32, b: f32| (a - b).abs() <= 1.0e-4 * b.abs().max(1.0);
        assert!(
            close(actual.total, expected.total),
            "total[{index}]: {actual:?} vs {expected:?}"
        );
        assert!(
            close(actual.peak, expected.peak),
            "peak[{index}]: {actual:?} vs {expected:?}"
        );
        assert_eq!(
            actual.above_threshold, expected.above_threshold,
            "above[{index}]"
        );
        assert_eq!(actual.count, expected.count, "count[{index}]");
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

/// Sample count of a debug run.
const DEBUG_COUNT: usize = 1 << 18;
/// Sample count of a release run. Stage A dispatches `count / workgroup_size`
/// groups, which must stay within the default per-dimension workgroup limit of
/// 65 535; 2^21 samples need 32 768 groups of 64.
const RELEASE_COUNT: usize = 1 << 21;

fn demo_count() -> usize {
    if cfg!(debug_assertions) {
        DEBUG_COUNT
    } else {
        RELEASE_COUNT
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let count = demo_count();
    let samples = make_samples(count);
    let device = HeadlessDevice::new()?;
    let state = ResidentState::new(&device, &samples);

    let updates = [
        Settings {
            scale: 1.5,
            offset: 2.0,
            threshold: 4.0,
            block: BLOCK,
        },
        Settings {
            scale: -0.5,
            offset: 10.0,
            threshold: 11.0,
            block: BLOCK,
        },
    ];
    let mut last_report = None;
    let mut last_edges = Vec::new();
    for settings in updates {
        let cpu_started = Instant::now();
        let expected = run_cpu(&samples, &settings);
        let cpu_elapsed = cpu_started.elapsed();
        let (actual, report, edges) = run_graph_traced(&device, &state, &[settings])?;
        last_edges = edges;
        assert_close(&actual, &expected);
        println!(
            "settings {settings:?}: CPU reference {cpu_elapsed:?}, graph {:?}, upload {} B, readback {} B, resident {} B",
            report.host_elapsed, report.upload_bytes, report.readback_bytes, report.resident_bytes
        );
        last_report = Some(report);
    }
    let artifacts = export_artifacts()?;

    println!(
        "Staged graph: {count} samples -> {} summaries",
        summary_count(count)
    );
    println!("adapter: {}", device.adapter_info().name);
    println!("nodes per execution: write -> transform -> summarize -> read");
    println!("derived dependencies per node (declared: none): {last_edges:?}");
    println!("pipeline cache: {:?}", device.pipeline_cache_stats());
    println!("last report: {:?}", last_report.expect("two updates ran"));
    println!("generated artifacts: {}", artifacts.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use gust::cpu::{CpuDevice, Error as CpuError};

    use super::*;
    use gust_wgpu::{Error, TransferDirection};

    /// The label of every test below that runs a kernel's CPU twin. The graph, the dependency edges and the
    /// transfers are not executed by the twin: they stay GPU-only (docs/SEMANTICS.md, section 1).
    const LABEL: &str = "kernel-level twin evidence; graph and indirect dispatch GPU-only";

    /// `2^-24`.
    const U: f64 = 1.0 / 16_777_216.0;

    /// The twin-versus-GPU bounds of the float outputs, absolute, in units of `U`. Each kernel is compared on identical
    /// inputs (`summarize` is fed the intermediate the GPU read back from stage A), so a bound covers that kernel's own
    /// roundings only.
    ///
    /// Derived from the kernel text (docs/SEMANTICS.md section 4, "Deriving a tolerance"), not from a measurement.
    /// With the domain asserted by `assert_derivation_domain` and `assert_intermediate_domain` (`abs(sample) <= 18`,
    /// `abs(scale) <= 1.5`, `abs(offset) <= 10`, so `abs(intermediate) <= 27 + 10 = 37`), one rounding of at most `u`
    /// times the result's magnitude on the twin and `2u` on the GPU for `+ - *`, first-order propagation, magnitudes
    /// `M`, twin error `T` and GPU error `G` in units of `u`, the GPU side without fusion:
    ///
    /// `transform` (`samples[i] * scale + offset`):
    ///
    /// | step | M | T | G |
    /// | --- | --- | --- | --- |
    /// | `samples[i] * scale` | 27 | 27 | 54 |
    /// | `... + offset` | 37 | 64 | 128 |
    ///
    /// Twin plus GPU: 192u, plus 1u for the second-order terms is 193u.
    ///
    /// `summarize`, `total`: eight values of magnitude at most 37 are added into `0.0`. The first addition (`0.0 + v`)
    /// is exact. The twin adds in the written order, so its seven roundings have partial sums of at most 37 k for
    /// k = 2 to 8: T = 37 * 35 = 1295. WGSL 15.7.5 lets the GPU regroup the additions; the grouping with the largest
    /// partial sums is the same left-deep chain (a balanced tree has smaller ones), so G = 2 * 1295 = 2590. Twin plus
    /// GPU: 3885u, plus 1u is 3886u.
    ///
    /// `summarize`, `peak` is a value of the input copied by comparisons, `above_threshold` compares an input with
    /// `threshold` and `count` counts: with identical inputs these are exact, and the test asserts them equal.
    ///
    /// Observed maxima are printed separately by the ignored test and never become the tolerance.
    const TRANSFORM_BOUND: f64 = 193.0 * U;
    const TOTAL_BOUND: f64 = 3886.0 * U;

    /// Two settings with exactly representable scale and offset (no rounding happens), and a third whose scale and
    /// offset are not representable, so the products and sums round. The thresholds lie off the grid of the
    /// intermediates (`assert_decisions_are_stable`).
    fn test_settings() -> [Settings; 3] {
        [
            Settings {
                scale: 0.1,
                offset: 0.3,
                threshold: 2.07,
                block: BLOCK,
            },
            Settings {
                scale: 1.5,
                offset: 2.0,
                threshold: 4.05,
                block: BLOCK,
            },
            Settings {
                scale: -0.5,
                offset: 10.0,
                threshold: 11.05,
                block: BLOCK,
            },
        ]
    }

    /// The domain the bounds are derived for. `make_samples` has period 97, so 1000 samples reach every extreme.
    fn assert_derivation_domain(samples: &[f32], settings: &Settings) {
        assert!(
            samples.iter().all(|sample| (-6.0..=18.0).contains(sample)),
            "sample"
        );
        assert!(settings.scale.abs() <= 1.5, "scale");
        assert!(settings.offset.abs() <= 10.0, "offset");
        assert_eq!(settings.block, BLOCK, "block");
    }

    /// The domain of the values that cross from stage A to stage B, which the bound of `total` assumes.
    fn assert_intermediate_domain(intermediate: &[f32]) {
        assert!(
            intermediate.iter().all(|value| value.abs() <= 37.0),
            "intermediate"
        );
    }

    /// `above_threshold` is a decision on a float: the targets agree on it unless an intermediate lies within the
    /// `transform` bound of the threshold (WGSL lets a product or sum round to either neighbour). Checked on the host
    /// intermediates, so the GPU's (within the bound) fall on the same side. The thresholds of `test_settings` lie off
    /// the grid of the samples, and this guard fails the test, never skips, if a change moves one onto it. (The
    /// twin-versus-GPU comparison of `summarize` is unaffected: its inputs are identical.)
    fn assert_decisions_are_stable(samples: &[f32], settings: &Settings) {
        // The f32 value the kernel compares with, not the decimal literal it was written as.
        let threshold = f64::from(settings.threshold);
        for (index, value) in intermediate_on_host(samples, settings).iter().enumerate() {
            assert!(
                (f64::from(*value) - threshold).abs() > TRANSFORM_BOUND,
                "intermediate[{index}] = {value} lies within the derived bound of the threshold {threshold}"
            );
        }
    }

    fn blank_summary() -> Summary {
        Summary {
            total: -1.0e9,
            peak: -1.0e9,
            above_threshold: u32::MAX,
            count: u32::MAX,
        }
    }

    /// The independent host reference of stage A.
    fn intermediate_on_host(samples: &[f32], settings: &Settings) -> Vec<f32> {
        samples
            .iter()
            .map(|&sample| sample * settings.scale + settings.offset)
            .collect()
    }

    /// Stage A on the twin. The output is seeded with a value no result equals, so a kernel that writes nothing fails.
    fn transform_on_cpu(samples: &[f32], settings: &Settings) -> Result<Vec<f32>, CpuError> {
        let device = CpuDevice::new();
        let samples_buffer = device.buffer(samples);
        let settings_buffer = device.buffer(&[*settings]);
        let out = device.buffer(&vec![-1.0e9f32; samples.len()]);
        staged::transform::dispatch(
            &device,
            samples.len() as u32,
            &samples_buffer,
            Independent(&settings_buffer),
            &out,
        )?;
        Ok(out.read())
    }

    /// Stage B on the twin, over the `intermediate` it is given (the twin's own, or the one the GPU read back).
    fn summarize_on_cpu(
        intermediate: &[f32],
        settings: &Settings,
        summaries: usize,
    ) -> Result<Vec<Summary>, CpuError> {
        let device = CpuDevice::new();
        let intermediate_buffer = device.buffer(intermediate);
        let settings_buffer = device.buffer(&[*settings]);
        let out = device.buffer(&vec![blank_summary(); summaries]);
        staged::summarize::dispatch(
            &device,
            summaries as u32,
            Independent(&intermediate_buffer),
            Independent(&settings_buffer),
            &out,
        )?;
        Ok(out.read())
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
    fn inputs_stay_inside_the_derivation_domain() {
        for count in [0, 1, 97, 1000, DEBUG_COUNT] {
            let samples = make_samples(count);
            for settings in test_settings() {
                assert_derivation_domain(&samples, &settings);
                assert_decisions_are_stable(&samples, &settings);
                assert_intermediate_domain(&intermediate_on_host(&samples, &settings));
            }
        }
    }

    /// Kernel-level twin evidence; graph and indirect dispatch GPU-only (`LABEL`).
    #[test]
    fn twin_chain_matches_host_reference() {
        // The two twins run the shadow bodies, which perform the operations of `run_cpu` in the same order, and Rust
        // does not contract them, so floats agree to the last bit and integers are equal. The counts straddle the
        // block of 8 and the workgroup of 64.
        for count in [0, 1, 7, 8, 9, 63, 64, 65, 257, 1000] {
            let samples = make_samples(count);
            for settings in test_settings() {
                let intermediate = transform_on_cpu(&samples, &settings).expect("transform twin");
                assert_eq!(
                    intermediate,
                    intermediate_on_host(&samples, &settings),
                    "transform, {count} samples; {LABEL}"
                );
                assert_intermediate_domain(&intermediate);
                let summaries = summarize_on_cpu(&intermediate, &settings, summary_count(count))
                    .expect("summarize twin");
                assert_eq!(
                    summaries,
                    run_cpu(&samples, &settings),
                    "summarize, {count} samples; {LABEL}"
                );
            }
        }
    }

    /// Kernel-level twin evidence; graph and indirect dispatch GPU-only (`LABEL`). The twin's `transform` is compared
    /// with the GPU's intermediate on the same samples and settings, and its `summarize` is fed the GPU's intermediate,
    /// read back once per run, so both kernels are compared on identical inputs.
    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn twin_matches_reference_and_gpu_within_the_derived_bounds() {
        let device = HeadlessDevice::new().expect("Vulkan adapter");
        for count in [0, 1, 7, 8, 9, 63, 64, 65, 257, 1000] {
            let samples = make_samples(count);
            let state = ResidentState::new(&device, &samples);
            for settings in test_settings() {
                assert_derivation_domain(&samples, &settings);
                assert_decisions_are_stable(&samples, &settings);
                let host = run_cpu(&samples, &settings);
                let (gpu_summaries, _report, _edges) =
                    run_graph_traced(&device, &state, &[settings]).unwrap();
                // The GPU against the host reference comes first so that a failing twin cannot hide it.
                assert_close(&gpu_summaries, &host);
                let gpu_intermediate = device.read_typed_buffer(&state.intermediate).unwrap();
                assert_eq!(gpu_intermediate.len(), count);

                let twin_intermediate = transform_on_cpu(&samples, &settings).unwrap();
                assert_eq!(
                    twin_intermediate,
                    intermediate_on_host(&samples, &settings),
                    "transform twin versus host, {count} samples"
                );
                let transform = largest_abs(
                    twin_intermediate.iter().copied(),
                    gpu_intermediate.iter().copied(),
                );
                println!(
                    "S4 staged-graph transform, {count} samples, scale {}: twin versus GPU max abs error {transform:e} (bound {TRANSFORM_BOUND:e}); {LABEL}",
                    settings.scale
                );
                assert!(transform <= TRANSFORM_BOUND, "transform {transform:e}");

                // From here on the twin is fed what the GPU produced for stage A: identical inputs.
                assert_intermediate_domain(&gpu_intermediate);
                let twin_summaries =
                    summarize_on_cpu(&gpu_intermediate, &settings, summary_count(count)).unwrap();
                assert_eq!(
                    twin_summaries.len(),
                    gpu_summaries.len(),
                    "summarize output length, {count} samples"
                );
                let total = largest_abs(
                    twin_summaries.iter().map(|s| s.total),
                    gpu_summaries.iter().map(|s| s.total),
                );
                println!(
                    "S4 staged-graph summarize, {count} samples, scale {}: twin versus GPU max abs error total {total:e} (bound {TOTAL_BOUND:e}); {LABEL}",
                    settings.scale
                );
                assert!(total <= TOTAL_BOUND, "total {total:e}");
                for (index, (twin, gpu)) in twin_summaries.iter().zip(&gpu_summaries).enumerate() {
                    assert_eq!(twin.peak, gpu.peak, "summarize peak[{index}]");
                    assert_eq!(
                        twin.above_threshold, gpu.above_threshold,
                        "summarize above_threshold[{index}]"
                    );
                    assert_eq!(twin.count, gpu.count, "summarize count[{index}]");
                }
            }
        }
    }

    #[test]
    fn demo_sizes_stay_within_the_default_workgroup_limit() {
        for count in [DEBUG_COUNT, RELEASE_COUNT] {
            // Stage A runs one element per sample, stage B one per summary.
            let stages = [
                (&staged::transform::DESCRIPTOR, count),
                (&staged::summarize::DESCRIPTOR, summary_count(count)),
            ];
            for (descriptor, elements) in stages {
                let groups = u32::try_from(elements)
                    .unwrap()
                    .div_ceil(descriptor.workgroup_size[0]);
                assert!(
                    groups <= gust_wgpu::DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                    "{}: {elements} elements need {groups} workgroups",
                    descriptor.entry_point
                );
            }
        }
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
    fn graph_matches_cpu_and_keeps_intermediate_resident() {
        let device = HeadlessDevice::new().expect("Vulkan adapter");
        for count in [0, 1, 7, 8, 9, 63, 64, 65, 257, 1000] {
            let samples = make_samples(count);
            let state = ResidentState::new(&device, &samples);
            for settings in [
                Settings {
                    scale: 1.5,
                    offset: 2.0,
                    threshold: 4.0,
                    block: BLOCK,
                },
                Settings {
                    scale: -0.5,
                    offset: 10.0,
                    threshold: 11.0,
                    block: BLOCK,
                },
            ] {
                let expected = run_cpu(&samples, &settings);
                let (actual, report, edges) =
                    run_graph_traced(&device, &state, &[settings]).unwrap();
                assert_close(&actual, &expected);
                // Derived from the buffers: transform reads the settings the write replaces; summarize
                // reads the settings and the intermediate transform wrote; the read follows summarize.
                assert_eq!(edges, [vec![], vec![0], vec![0, 1], vec![2]]);
                assert_eq!((report.derived_edges, report.declared_edges), (4, 0));
                assert_eq!(report.upload_bytes, size_of::<Settings>() as u64);
                assert_eq!(
                    report.readback_bytes,
                    (summary_count(count) * size_of::<Summary>()) as u64
                );
                // Samples and the intermediate never cross the host boundary.
                assert_eq!(report.resident_bytes, 2 * (count * size_of::<f32>()) as u64);
                assert_eq!(report.dispatches, if count == 0 { 0 } else { 2 });
                let expected_workgroups = u64::from((count as u32).div_ceil(64))
                    + u64::from((summary_count(count) as u32).div_ceil(64));
                assert_eq!(report.workgroups, expected_workgroups);
                assert_eq!(report.transfers.len(), 2);
                assert_eq!(report.transfers[0].direction, TransferDirection::Upload);
                assert_eq!(report.transfers[1].direction, TransferDirection::Readback);
            }
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn second_update_observes_new_settings_in_order() {
        // If the upload were not ordered before stage A, the second run would
        // still see the first settings.
        let device = HeadlessDevice::new().expect("Vulkan adapter");
        let samples = make_samples(64);
        let state = ResidentState::new(&device, &samples);
        let first = Settings {
            scale: 1.0,
            offset: 0.0,
            threshold: 0.0,
            block: BLOCK,
        };
        let second = Settings {
            scale: 2.0,
            offset: 100.0,
            threshold: 50.0,
            block: BLOCK,
        };
        run_graph(&device, &state, &[first]).unwrap();
        let (actual, _) = run_graph(&device, &state, &[second]).unwrap();
        assert_close(&actual, &run_cpu(&samples, &second));
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn undeclared_dependencies_and_invalid_resources_are_rejected() {
        let device = HeadlessDevice::new().expect("Vulkan adapter");
        let samples = make_samples(64);
        let state = ResidentState::new(&device, &samples);
        let settings = [Settings {
            scale: 1.0,
            offset: 0.0,
            threshold: 0.0,
            block: BLOCK,
        }];
        let transform_bindings = [
            BufferBinding::read_only(&state.samples),
            BufferBinding::read_only(&state.settings).independent_length(),
            BufferBinding::read_write(&state.intermediate),
        ];
        let summarize_bindings = [
            BufferBinding::read_only(&state.intermediate).independent_length(),
            BufferBinding::read_only(&state.settings).independent_length(),
            BufferBinding::read_write(&state.summaries),
        ];
        let transform =
            BufferDispatch::new(&staged::transform::DESCRIPTOR, 64, &transform_bindings);
        let summarize = BufferDispatch::new(&staged::summarize::DESCRIPTOR, 8, &summarize_bindings);

        // Stage A reads the settings the upload writes, but declares no edge.
        let mut graph = StagedGraph::new();
        let _upload = graph.upload(&state.settings, &settings, &[]);
        graph.dispatch(transform, &[]);
        assert!(matches!(
            device.execute_graph(&graph),
            Err(Error::GraphMissingDependency {
                node: 1,
                producer: 0
            })
        ));

        // Stage B reads the intermediate stage A writes; depending only on the
        // upload does not cover that hazard.
        let mut graph = StagedGraph::new();
        let upload = graph.upload(&state.settings, &settings, &[]);
        let _stage_a = graph.dispatch(transform, &[upload]);
        graph.dispatch(summarize, &[upload]);
        assert!(matches!(
            device.execute_graph(&graph),
            Err(Error::GraphMissingDependency {
                node: 2,
                producer: 1
            })
        ));

        // Transitive coverage is accepted: readback depends on B, B on A, A on upload.
        let mut graph = StagedGraph::new();
        let upload = graph.upload(&state.settings, &settings, &[]);
        let stage_a = graph.dispatch(transform, &[upload]);
        let stage_b = graph.dispatch(summarize, &[stage_a]);
        let readback = graph.readback(&state.summaries, &[stage_b]);
        let output = device.execute_graph(&graph).unwrap();
        assert_eq!(output.readback::<Summary>(readback).unwrap().len(), 8);
        assert!(matches!(
            output.readback::<f32>(readback),
            Err(Error::GraphReadbackType { .. })
        ));
        assert!(matches!(
            output.readback::<Summary>(stage_b),
            Err(Error::GraphUnknownReadback(2))
        ));

        // The derived graph over the same buffers needs no declared edge and gets exactly the edges the
        // two rejected graphs above lacked (typed calls are locals the graph borrows).
        let transform_call = staged::transform::call(
            64,
            &state.samples,
            Independent(&state.settings),
            &state.intermediate,
        );
        let summarize_call = staged::summarize::call(
            8,
            Independent(&state.intermediate),
            Independent(&state.settings),
            &state.summaries,
        );
        let mut derived = StagedGraph::new();
        let write = derived.write(&state.settings, &settings);
        let stage_a = derived.run(&transform_call);
        let stage_b = derived.run(&summarize_call);
        let read = derived.read(&state.summaries);
        let edges = |node| derived.dependency_indices(node);
        assert_eq!(
            [edges(write), edges(stage_a), edges(stage_b), edges(read)],
            [vec![], vec![0], vec![0, 1], vec![2]]
        );
        assert!(
            [write, stage_a, stage_b, read]
                .into_iter()
                .all(|node| derived.is_derived(node))
        );
        let output = device.execute_graph(&derived).unwrap();
        assert_eq!(
            (output.report.derived_edges, output.report.declared_edges),
            (4, 0)
        );
        assert_eq!(output.readback::<Summary>(read).unwrap().len(), 8);

        // A node id from another graph is not an earlier node of this one.
        let mut other = StagedGraph::new();
        other.upload(&state.settings, &settings, &[]);
        let foreign = other.dispatch(transform, &[]);
        let mut graph = StagedGraph::new();
        graph.dispatch(transform, &[foreign]);
        assert!(matches!(
            device.execute_graph(&graph),
            Err(Error::GraphDependencyOrder {
                node: 0,
                dependency: 1
            })
        ));

        // Reading back a read-only buffer is rejected before any GPU work.
        let mut graph = StagedGraph::new();
        graph.readback(&state.samples, &[]);
        assert!(matches!(
            device.execute_graph(&graph),
            Err(Error::PersistentBufferNotReadable)
        ));

        // An upload of the wrong length is rejected.
        let mut graph = StagedGraph::new();
        graph.upload(&state.intermediate, &[1.0f32, 2.0], &[]);
        assert!(matches!(
            device.execute_graph(&graph),
            Err(Error::PersistentBufferLength {
                expected: 64,
                actual: 2
            })
        ));

        // The strict shared-length rule still applies without the opt-in.
        let strict = [
            BufferBinding::read_only(&state.samples),
            BufferBinding::read_only(&state.settings),
            BufferBinding::read_write(&state.intermediate),
        ];
        assert!(matches!(
            device.dispatch_buffers(&staged::transform::DESCRIPTOR, 64, &strict),
            Err(Error::BufferLength { .. })
        ));

        // An empty independent-length buffer would expose a placeholder allocation.
        let empty = device.create_typed_buffer::<Settings>("empty", &[], GpuBufferAccess::ReadOnly);
        let placeholder = [
            BufferBinding::read_only(&state.samples),
            BufferBinding::read_only(&empty).independent_length(),
            BufferBinding::read_write(&state.intermediate),
        ];
        assert!(matches!(
            device.dispatch_buffers(&staged::transform::DESCRIPTOR, 64, &placeholder),
            Err(Error::IndependentLengthEmpty { .. })
        ));
    }
}
