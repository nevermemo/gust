use std::{path::PathBuf, time::Instant};

#[cfg(gust_dual)]
use gust::cpu::CpuDevice;
use gust::{JobStatus, gpu};
use gust_wgpu::{
    BufferDispatch, GpuBufferAccess, HeadlessDevice, TransferTiming, render_wgpu_source,
};

const PERSISTENT_ITERATIONS: u32 = 20;
const DEBUG_COUNT: usize = 1 << 18;
/// 32 768 workgroups of 128 invocations, within the 65 535 workgroups per dimension of the default limits.
const RELEASE_COUNT: usize = 1 << 22;

#[gpu]
mod polynomial {
    #[kernel(workgroup_size(128, 1, 1))]
    pub fn evaluate(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            let value = input[i];
            let squared = value * value;
            let cubed = squared * value;
            let curved = cubed - (2.0 * squared);
            let linear = 0.5 * value;
            out[i] = (curved + linear) + 1.0;
        }
    }
}

pub fn main() {
    let element_count = benchmark_element_count();
    let input = (0..element_count)
        .map(|index| (index % 4096) as f32 / 2048.0 - 1.0)
        .collect::<Vec<_>>();
    let mut cpu_output = vec![0.0; element_count];

    let cpu_started = Instant::now();
    run_cpu(&input, &mut cpu_output);
    let cpu_elapsed = cpu_started.elapsed();

    let device = HeadlessDevice::new().expect("a Vulkan compute adapter should be available");
    let initial_output = vec![0.0; element_count];
    let _warmup = dispatch_gpu(&device, &input, &initial_output).expect("polynomial GPU warmup");
    let gpu_started = Instant::now();
    let gpu_output =
        dispatch_gpu(&device, &input, &initial_output).expect("polynomial GPU dispatch");
    let gpu_elapsed = gpu_started.elapsed();
    let error = max_abs_error(&cpu_output, &gpu_output);
    assert!(error <= 1.0e-5, "CPU/GPU error was {error}");

    let persistent_input =
        device.create_f32_buffer("polynomial input", &input, GpuBufferAccess::ReadOnly);
    let persistent_output = device.create_f32_buffer(
        "polynomial output",
        &initial_output,
        GpuBufferAccess::ReadWrite,
    );
    let upload_timing = device
        .write_f32_buffer_timed(&persistent_input, &input)
        .expect("timed polynomial upload");
    polynomial::evaluate::dispatch(
        &device,
        element_count as u32,
        &persistent_input,
        &persistent_output,
    )
    .expect("persistent polynomial warmup");
    let persistent_started = Instant::now();
    for _ in 0..PERSISTENT_ITERATIONS {
        polynomial::evaluate::dispatch(
            &device,
            element_count as u32,
            &persistent_input,
            &persistent_output,
        )
        .expect("persistent polynomial dispatch");
    }
    let persistent_elapsed = persistent_started.elapsed() / PERSISTENT_ITERATIONS;
    // One typed call, repeated: a `Call` is a reusable value and converts to a batch entry.
    let call =
        polynomial::evaluate::call(element_count as u32, &persistent_input, &persistent_output);
    let batch = vec![BufferDispatch::from(&call); PERSISTENT_ITERATIONS as usize];
    let batch_started = Instant::now();
    device
        .dispatch_batch(&batch)
        .expect("batched polynomial dispatch");
    let batch_elapsed = batch_started.elapsed();
    let profiled_batch = device
        .dispatch_batch_timed(&batch)
        .expect("timestamped polynomial batch");
    let mut overlap_output = vec![0.0; element_count];
    let async_submit_started = Instant::now();
    let async_job = device
        .submit_batch(&batch)
        .expect("asynchronous polynomial batch");
    let async_submit_elapsed = async_submit_started.elapsed();
    let status_after_submit = async_job.status().expect("polynomial job status");
    let cpu_overlap_started = Instant::now();
    run_cpu(&input, &mut overlap_output);
    let cpu_overlap_elapsed = cpu_overlap_started.elapsed();
    let status_after_cpu = async_job
        .status()
        .expect("polynomial job status after CPU work");
    let final_wait_started = Instant::now();
    async_job
        .wait()
        .expect("asynchronous polynomial completion");
    let final_wait_elapsed = final_wait_started.elapsed();
    assert!(max_abs_error(&cpu_output, &overlap_output) <= 1.0e-5);
    let async_timing = AsyncBatchTiming {
        submit: async_submit_elapsed,
        status_after_submit,
        cpu_overlap: cpu_overlap_elapsed,
        status_after_cpu,
        final_wait: final_wait_elapsed,
    };
    let (persistent_result, readback_timing) = device
        .read_f32_buffer_timed(&persistent_output)
        .expect("timed persistent polynomial readback");
    let persistent_error = max_abs_error(&cpu_output, &persistent_result);
    assert!(
        persistent_error <= 1.0e-5,
        "persistent CPU/GPU error was {persistent_error}"
    );

    let generated_source = export_generated_source();
    println!("Polynomial benchmark");
    println!(
        "kernel: {}",
        polynomial::evaluate::DESCRIPTOR.qualified_name()
    );
    println!("adapter: {}", device.adapter_info().name);
    println!("elements: {element_count}");
    println!("operations per element: 4 mul, 1 sub, 2 add");
    println!(
        "hardware GPU timestamps: {}",
        device.supports_gpu_timestamps()
    );
    println!("pipeline cache: {:?}", device.pipeline_cache_stats());
    print_comparison(
        cpu_elapsed,
        gpu_elapsed,
        persistent_elapsed,
        batch_elapsed,
        profiled_batch.gpu_elapsed,
        upload_timing,
        readback_timing,
    );
    print_async_overlap(async_timing);
    println!("maximum absolute error: {error:e}");
    println!("persistent maximum absolute error: {persistent_error:e}");
    println!("generated wgpu source: {}", generated_source.display());

    // Only the dual test build (`cargo xtask check-dual`, `--cfg gust_dual`, D26) has the twin in a binary.
    #[cfg(gust_dual)]
    report_twin(&input, &cpu_output, &gpu_output, cpu_elapsed);
}

/// The twin-versus-GPU bound, absolute, over the whole input domain: 48 * 2^-24, about 2.86e-6.
///
/// Derived from the kernel text (docs/SEMANTICS.md section 4, "Worked example"), not from a measurement. With
/// `u = 2^-24` and `|v| <= 1` (inputs are `k / 2048 - 1`), multiplication by a power of two is exact, and:
/// - twin, in the written order: `squared` u, `cubed` 2u, `2.0 * squared` 2u, `curved` 7u (2u + 2u plus u times
///   its magnitude bound 3), plus `linear` 10.5u, plus `1.0` 15u;
/// - GPU, in the worst grouping WGSL allows for the four terms: `squared` 2u, `cubed` 4u, `2.0 * squared` 4u,
///   then three additions of partial sums of magnitude up to 3, 4 and 4.5 at 2u each, so 8u + 23u = 31u;
/// - twin versus GPU: 15u + 31u = 46u, and 48u asserted, the extra 2u being the margin for second-order terms.
///
/// `main`'s inputs, `(index % 4096) / 2048 - 1`, lie inside that domain, so the dual `main` asserts the same bound.
/// Observed maxima are printed separately and never become the tolerance.
#[cfg(any(test, gust_dual))]
const TWIN_VS_GPU_BOUND: f64 = 48.0 / 16_777_216.0;

/// The twin on the same input as `main`'s `run_cpu` and GPU runs. It must equal `run_cpu` to the last bit (the same
/// operations in the same order) and stay within the derived `TWIN_VS_GPU_BOUND` of the GPU result, which `main`'s
/// inputs are inside the domain of (the ignored test `twin_matches_reference_and_gpu_within_the_derived_bound`
/// asserts the same bound on that domain).
/// Prints the line `cargo xtask check-dual` requires, then the twin's wall time beside `run_cpu`'s: both are
/// single-threaded runs of the same input on one thread, and the debug profile is not a benchmark.
#[cfg(gust_dual)]
fn report_twin(
    input: &[f32],
    run_cpu_output: &[f32],
    gpu_output: &[f32],
    run_cpu_elapsed: std::time::Duration,
) {
    let device = CpuDevice::new();
    let input_buffer = device.buffer(input);
    let output = device.buffer(&vec![0.0f32; input.len()]);
    let started = Instant::now();
    polynomial::evaluate::dispatch(&device, input.len() as u32, &input_buffer, &output)
        .expect("the CPU twin should dispatch");
    let twin_elapsed = started.elapsed();
    let twin = output.read();
    let (max_ulps, _) = largest_difference(&twin, run_cpu_output);
    assert_eq!(max_ulps, 0, "CPU twin versus run_cpu");
    let (_, gpu_abs) = largest_difference(&twin, gpu_output);
    assert!(
        gpu_abs <= TWIN_VS_GPU_BOUND,
        "CPU twin versus GPU: maximum absolute error {gpu_abs:e} exceeds the derived bound {TWIN_VS_GPU_BOUND:e}"
    );
    println!(
        "CPU twin: matches ({} elements, max ULP {max_ulps})",
        twin.len()
    );
    println!("CPU twin versus GPU: maximum absolute error {gpu_abs:e}");
    let profile = if cfg!(debug_assertions) {
        "debug profile"
    } else {
        "release profile"
    };
    println!(
        "CPU twin time: {twin_elapsed:?} (run_cpu: {run_cpu_elapsed:?}; {profile}, single-threaded, not a benchmark)"
    );
}

/// `a.max(b)` that keeps a NaN. `f64::max` returns the other operand when one is NaN, so a `fold(0.0, f64::max)` of
/// differences turns a NaN difference into 0 and a bound check on the result passes; this fold poisons the maximum
/// instead and `maximum <= bound` fails.
#[cfg(any(test, gust_dual))]
fn propagating_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// The largest distance between two results in ULPs and in absolute terms (computed in `f64`).
#[cfg(any(test, gust_dual))]
fn largest_difference(left: &[f32], right: &[f32]) -> (u64, f64) {
    assert_eq!(left.len(), right.len());
    left.iter().zip(right).fold((0, 0.0), |(ulp, abs), (l, r)| {
        (
            ulp.max(gust::cpu::ulps(*l, *r)),
            propagating_max(abs, (f64::from(*l) - f64::from(*r)).abs()),
        )
    })
}

fn run_cpu(input: &[f32], output: &mut [f32]) {
    for (output, value) in output.iter_mut().zip(input.iter().copied()) {
        let squared = value * value;
        let cubed = squared * value;
        let curved = cubed - 2.0 * squared;
        let linear = 0.5 * value;
        *output = curved + linear + 1.0;
    }
}

/// One-shot GPU path: upload both host slices, run the generated typed `dispatch`, read the output back.
fn dispatch_gpu(
    device: &HeadlessDevice,
    input: &[f32],
    initial_output: &[f32],
) -> Result<Vec<f32>, gust_wgpu::Error> {
    let input = device.create_f32_buffer(
        "polynomial one-shot input",
        input,
        GpuBufferAccess::ReadOnly,
    );
    let output = device.create_f32_buffer(
        "polynomial one-shot output",
        initial_output,
        GpuBufferAccess::ReadWrite,
    );
    polynomial::evaluate::dispatch(device, initial_output.len() as u32, &input, &output)?;
    device.read_f32_buffer(&output)
}

fn benchmark_element_count() -> usize {
    if cfg!(debug_assertions) {
        DEBUG_COUNT
    } else {
        RELEASE_COUNT
    }
}

/// The largest absolute difference; NaN when any difference is NaN, so that `error <= bound` fails
/// (`f32::max` would return the other operand and a fold with it would turn the NaN into 0).
fn max_abs_error(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, |max: f32, diff| {
            if max.is_nan() || diff.is_nan() {
                f32::NAN
            } else {
                max.max(diff)
            }
        })
}

fn print_comparison(
    cpu: std::time::Duration,
    gpu: std::time::Duration,
    persistent: std::time::Duration,
    batch: std::time::Duration,
    gpu_batch: Option<std::time::Duration>,
    upload: TransferTiming,
    readback: TransferTiming,
) {
    let end_to_end_ratio = cpu.as_secs_f64() / gpu.as_secs_f64();
    let persistent_ratio = cpu.as_secs_f64() / persistent.as_secs_f64();
    let batch_per_dispatch = batch / PERSISTENT_ITERATIONS;
    let batching_speedup = persistent.as_secs_f64() / batch_per_dispatch.as_secs_f64();
    println!("CPU reference: {cpu:?}");
    println!("GPU end-to-end: {gpu:?}");
    println!("GPU persistent dispatch: {persistent:?}");
    println!(
        "GPU batch ({} dispatches): {batch:?} total, {batch_per_dispatch:?} per dispatch",
        PERSISTENT_ITERATIONS
    );
    println!(
        "GPU input upload ({} bytes): {:?}",
        upload.bytes, upload.elapsed
    );
    match gpu_batch {
        Some(elapsed) => println!(
            "GPU batch timestamp: {elapsed:?} total, {:?} per dispatch",
            elapsed / PERSISTENT_ITERATIONS
        ),
        None => println!("GPU batch timestamp: unavailable"),
    }
    println!(
        "GPU output readback ({} bytes): {:?}",
        readback.bytes, readback.elapsed
    );
    if end_to_end_ratio >= 1.0 {
        println!("GPU end-to-end is {end_to_end_ratio:.2}x faster than CPU");
    } else {
        println!(
            "CPU is {:.2}x faster than GPU end-to-end",
            end_to_end_ratio.recip()
        );
    }
    if persistent_ratio >= 1.0 {
        println!("GPU persistent dispatch is {persistent_ratio:.2}x faster than CPU");
    } else {
        println!(
            "CPU is {:.2}x faster than GPU persistent dispatch",
            persistent_ratio.recip()
        );
    }
    println!("batching improves host throughput by {batching_speedup:.2}x");
}

#[derive(Clone, Copy)]
struct AsyncBatchTiming {
    submit: std::time::Duration,
    status_after_submit: JobStatus,
    cpu_overlap: std::time::Duration,
    status_after_cpu: JobStatus,
    final_wait: std::time::Duration,
}

fn print_async_overlap(timing: AsyncBatchTiming) {
    println!("GPU asynchronous batch submission: {:?}", timing.submit);
    println!(
        "job status after submission: {:?}",
        timing.status_after_submit
    );
    println!("overlapped CPU reference: {:?}", timing.cpu_overlap);
    println!("job status after CPU work: {:?}", timing.status_after_cpu);
    println!("final job wait: {:?}", timing.final_wait);
}

fn export_generated_source() -> PathBuf {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../generated-wgpu");
    std::fs::create_dir_all(&directory).expect("create generated-wgpu directory");
    let descriptor = &polynomial::evaluate::DESCRIPTOR;
    let path = directory.join("polynomial__evaluate.rs");
    let source = render_wgpu_source(descriptor).expect("render polynomial wgpu source");
    std::fs::write(&path, source).expect("write polynomial wgpu source");
    std::fs::write(
        directory.join("polynomial__evaluate.slang"),
        descriptor.slang_source,
    )
    .expect("write polynomial Slang source");
    std::fs::write(
        directory.join("polynomial__evaluate.wgsl"),
        gust::slang::compile_wgsl(descriptor).expect("compile polynomial WGSL"),
    )
    .expect("write polynomial WGSL");
    std::fs::write(
        directory.join("polynomial__evaluate.spv"),
        gust::spirv::words_as_le_bytes(
            &gust::slang::compile_spirv(descriptor).expect("compile polynomial SPIR-V"),
        ),
    )
    .expect("write polynomial SPIR-V");
    path.canonicalize().unwrap_or(path)
}

#[cfg(test)]
mod tests {
    use gust::cpu::{CpuDevice, Error as CpuError};

    use super::*;

    /// `k / 2048 - 1` for `k` in `0..4096`, the input domain of `main`.
    fn domain() -> Vec<f32> {
        (0..4096).map(|k| k as f32 / 2048.0 - 1.0).collect()
    }

    /// The host reference: `run_cpu` over a fresh output.
    fn reference(input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        run_cpu(input, &mut output);
        output
    }

    #[test]
    fn a_nan_in_a_compared_value_fails_the_bound_checks() {
        // `fold(0.0, f64::max)` returned 0 for a NaN difference, so a GPU that wrote NaN passed the bounds.
        let finite = [1.0f32, 2.0, 3.0];
        let mut poisoned = finite;
        poisoned[1] = f32::NAN;
        let (_, clean) = largest_difference(&finite, &finite);
        assert!(clean <= TWIN_VS_GPU_BOUND);
        for (left, right) in [(&finite, &poisoned), (&poisoned, &finite)] {
            let (ulps, abs) = largest_difference(left, right);
            assert!(abs.is_nan(), "the absolute error must stay NaN, got {abs}");
            let passes = abs <= TWIN_VS_GPU_BOUND;
            assert!(!passes, "a NaN must not pass the bound");
            assert_eq!(ulps, u64::MAX);
            assert!(max_abs_error(left, right).is_nan());
            let passes = max_abs_error(left, right) <= 1.0e-5;
            assert!(!passes, "a NaN must not pass the bound");
        }
        // A NaN in the first or the last position and a larger finite error elsewhere still poison the maximum.
        let late = [f32::NAN, 5.0, 9.0];
        assert!(largest_difference(&late, &[0.0, 0.0, 0.0]).1.is_nan());
        assert!(max_abs_error(&late, &[0.0, 0.0, 0.0]).is_nan());
        assert_eq!(max_abs_error(&[1.0, 4.0], &[1.5, 1.0]), 3.0);
    }

    /// The same typed `dispatch` on the CPU twin, which the macro emits under `cfg(test)`: no GPU involved.
    fn evaluate_on_cpu(input: &[f32]) -> Result<Vec<f32>, CpuError> {
        let device = CpuDevice::new();
        let input_buffer = device.buffer(input);
        let output = device.buffer(&vec![0.0f32; input.len()]);
        polynomial::evaluate::dispatch(&device, input.len() as u32, &input_buffer, &output)?;
        Ok(output.read())
    }

    #[test]
    fn twin_matches_host_reference() {
        // The twin runs the shadow body, which performs the operations of `run_cpu` in the same order, and Rust
        // does not contract them, so the two agree to the last bit. 4096 values fill 32 workgroups; 1000 end
        // mid-workgroup (the kernel's `i < out.len()` guard is what keeps the tail inside the buffer).
        let domain = domain();
        for count in [domain.len(), 1000, 1, 0] {
            let input = &domain[..count];
            let twin = evaluate_on_cpu(input).expect("twin dispatch");
            let (max_ulps, max_abs) = largest_difference(&twin, &reference(input));
            println!(
                "twin versus run_cpu, {count} elements: max {max_ulps} ULP, max abs error {max_abs:e}"
            );
            assert_eq!(max_ulps, 0, "{count} elements");
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn twin_matches_reference_and_gpu_within_the_derived_bound() {
        let device = match HeadlessDevice::new() {
            Ok(device) => device,
            Err(error) => panic!(
                "Vulkan adapter required; this test must not silently skip: could not initialize headless wgpu: {error}"
            ),
        };
        let input = domain();
        let twin = evaluate_on_cpu(&input).expect("twin dispatch");
        let gpu = dispatch_gpu(&device, &input, &vec![0.0; input.len()]).expect("GPU dispatch");
        let host = reference(&input);

        let (twin_host_ulps, twin_host_abs) = largest_difference(&twin, &host);
        let (twin_gpu_ulps, twin_gpu_abs) = largest_difference(&twin, &gpu);
        // Observations on this adapter, recorded in the slice log (probe/s4.log); not tolerances.
        println!("S4 adapter: {}", device.adapter_info().name);
        println!(
            "S4 twin versus run_cpu: max {twin_host_ulps} ULP, max abs error {twin_host_abs:e}"
        );
        println!("S4 twin versus GPU: max {twin_gpu_ulps} ULP, max abs error {twin_gpu_abs:e}");
        println!("S4 derived bound (48 * 2^-24): {TWIN_VS_GPU_BOUND:e}");

        assert_eq!(twin_host_ulps, 0, "twin versus run_cpu");
        assert!(
            twin_gpu_abs <= TWIN_VS_GPU_BOUND,
            "twin versus GPU: max abs error {twin_gpu_abs:e} exceeds the derived bound {TWIN_VS_GPU_BOUND:e}"
        );
    }

    #[test]
    fn demo_sizes_stay_within_the_default_workgroup_limit() {
        for count in [DEBUG_COUNT, RELEASE_COUNT] {
            let groups = u32::try_from(count)
                .unwrap()
                .div_ceil(polynomial::evaluate::DESCRIPTOR.workgroup_size[0]);
            assert!(
                groups <= gust_wgpu::DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                "{count} elements need {groups} workgroups"
            );
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn cpu_reference_evaluates_all_steps() {
        let input = [-1.0, 0.0, 0.5, 1.0];
        let mut output = [0.0; 4];
        run_cpu(&input, &mut output);
        let expected = input.map(|value| {
            let squared = value * value;
            value * squared - 2.0 * squared + 0.5 * value + 1.0
        });
        assert_eq!(output, expected);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn headless_gpu_matches_cpu_reference() {
        let device = match HeadlessDevice::new() {
            Ok(device) => device,
            Err(error) => panic!(
                "Vulkan adapter required; this test must not silently skip: could not initialize headless wgpu: {error}"
            ),
        };
        let input = [-1.0, -0.25, 0.0, 0.5, 1.0, 1.5];
        let mut cpu_output = [0.0; 6];
        let initial_output = [0.0; 6];
        run_cpu(&input, &mut cpu_output);
        let gpu_output =
            dispatch_gpu(&device, &input, &initial_output).expect("GPU polynomial dispatch");
        assert!(max_abs_error(&cpu_output, &gpu_output) <= 1.0e-6);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn descriptor_and_generated_host_code_are_executable() {
        let descriptor = polynomial::evaluate::DESCRIPTOR;
        assert!(
            descriptor
                .slang_source
                .contains("void gpu_polynomial_evaluate")
        );
        let source = render_wgpu_source(&descriptor).expect("render source");
        assert!(source.contains("pub struct PolynomialEvaluatePipeline"));
        assert!(source.contains("pub fn encode("));
        assert!(source.contains("input.as_entire_binding()"));
        assert!(source.contains("out.as_entire_binding()"));
    }
}
