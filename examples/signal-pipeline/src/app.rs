use std::{path::PathBuf, time::Instant};

use gust::{JobStatus, gpu};
use gust_wgpu::{
    BufferDispatch, GpuBufferAccess, HeadlessDevice, TransferTiming, render_wgpu_source,
};

const PERSISTENT_ITERATIONS: u32 = 20;
const DEBUG_COUNT: usize = 1 << 18;
/// 32 768 workgroups of 128 invocations, within the 65 535 workgroups per dimension of the default limits.
const RELEASE_COUNT: usize = 1 << 22;

#[gpu]
mod signal_pipeline {
    #[kernel(workgroup_size(128, 1, 1))]
    pub fn transform(
        id: SV_DispatchThreadID,
        signal: StructuredBuffer<float>,
        baseline: StructuredBuffer<float>,
        gain: StructuredBuffer<float>,
        bias: StructuredBuffer<float>,
        scale: StructuredBuffer<float>,
        mut out: RWStructuredBuffer<float>,
    ) {
        let i = id.x;
        if i < out.len() {
            let centered = signal[i] - baseline[i];
            let amplified = centered * gain[i];
            let biased = amplified + bias[i];
            let normalized = biased / scale[i];
            let energy = normalized * normalized;
            out[i] = energy + (0.25 * normalized);
        }
    }
}

pub fn main() {
    let element_count = benchmark_element_count();
    let inputs = Inputs::new(element_count);
    let mut cpu_output = vec![0.0; element_count];

    let cpu_started = Instant::now();
    run_cpu(&inputs, &mut cpu_output);
    let cpu_elapsed = cpu_started.elapsed();

    let device = HeadlessDevice::new().expect("a Vulkan compute adapter should be available");
    let initial_output = vec![0.0; element_count];
    let _warmup = dispatch_gpu(&device, &inputs, &initial_output).expect("signal GPU warmup");
    let gpu_started = Instant::now();
    let gpu_result = dispatch_gpu(&device, &inputs, &initial_output).expect("signal GPU dispatch");
    let gpu_elapsed = gpu_started.elapsed();
    let error = max_abs_error(&cpu_output, &gpu_result);
    assert!(error <= 2.0e-5, "CPU/GPU error was {error}");

    let persistent_signal =
        device.create_f32_buffer("signal", &inputs.signal, GpuBufferAccess::ReadOnly);
    let persistent_baseline =
        device.create_f32_buffer("baseline", &inputs.baseline, GpuBufferAccess::ReadOnly);
    let persistent_gain = device.create_f32_buffer("gain", &inputs.gain, GpuBufferAccess::ReadOnly);
    let persistent_bias = device.create_f32_buffer("bias", &inputs.bias, GpuBufferAccess::ReadOnly);
    let persistent_scale =
        device.create_f32_buffer("scale", &inputs.scale, GpuBufferAccess::ReadOnly);
    let persistent_output =
        device.create_f32_buffer("signal output", &initial_output, GpuBufferAccess::ReadWrite);
    let upload_timing = [
        (&persistent_signal, inputs.signal.as_slice()),
        (&persistent_baseline, inputs.baseline.as_slice()),
        (&persistent_gain, inputs.gain.as_slice()),
        (&persistent_bias, inputs.bias.as_slice()),
        (&persistent_scale, inputs.scale.as_slice()),
    ]
    .into_iter()
    .try_fold(TransferTiming::default(), |total, (buffer, values)| {
        let timing = device.write_f32_buffer_timed(buffer, values)?;
        Ok::<_, gust_wgpu::Error>(TransferTiming {
            bytes: total.bytes + timing.bytes,
            elapsed: total.elapsed + timing.elapsed,
        })
    })
    .expect("timed signal input uploads");
    signal_pipeline::transform::dispatch(
        &device,
        element_count as u32,
        &persistent_signal,
        &persistent_baseline,
        &persistent_gain,
        &persistent_bias,
        &persistent_scale,
        &persistent_output,
    )
    .expect("persistent signal warmup");
    let persistent_started = Instant::now();
    for _ in 0..PERSISTENT_ITERATIONS {
        signal_pipeline::transform::dispatch(
            &device,
            element_count as u32,
            &persistent_signal,
            &persistent_baseline,
            &persistent_gain,
            &persistent_bias,
            &persistent_scale,
            &persistent_output,
        )
        .expect("persistent signal dispatch");
    }
    let persistent_elapsed = persistent_started.elapsed() / PERSISTENT_ITERATIONS;
    // One typed call, repeated: a `Call` is a reusable value and converts to a batch entry.
    let call = signal_pipeline::transform::call(
        element_count as u32,
        &persistent_signal,
        &persistent_baseline,
        &persistent_gain,
        &persistent_bias,
        &persistent_scale,
        &persistent_output,
    );
    let batch = vec![BufferDispatch::from(&call); PERSISTENT_ITERATIONS as usize];
    let batch_started = Instant::now();
    device
        .dispatch_batch(&batch)
        .expect("batched signal dispatch");
    let batch_elapsed = batch_started.elapsed();
    let profiled_batch = device
        .dispatch_batch_timed(&batch)
        .expect("timestamped signal batch");
    let mut overlap_output = vec![0.0; element_count];
    let async_submit_started = Instant::now();
    let async_job = device
        .submit_batch(&batch)
        .expect("asynchronous signal batch");
    let async_submit_elapsed = async_submit_started.elapsed();
    let status_after_submit = async_job.status().expect("signal job status");
    let cpu_overlap_started = Instant::now();
    run_cpu(&inputs, &mut overlap_output);
    let cpu_overlap_elapsed = cpu_overlap_started.elapsed();
    let status_after_cpu = async_job
        .status()
        .expect("signal job status after CPU work");
    let final_wait_started = Instant::now();
    async_job.wait().expect("asynchronous signal completion");
    let final_wait_elapsed = final_wait_started.elapsed();
    assert!(max_abs_error(&cpu_output, &overlap_output) <= 2.0e-5);
    let async_timing = AsyncBatchTiming {
        submit: async_submit_elapsed,
        status_after_submit,
        cpu_overlap: cpu_overlap_elapsed,
        status_after_cpu,
        final_wait: final_wait_elapsed,
    };
    let (persistent_result, readback_timing) = device
        .read_f32_buffer_timed(&persistent_output)
        .expect("timed persistent signal readback");
    let persistent_error = max_abs_error(&cpu_output, &persistent_result);
    assert!(
        persistent_error <= 2.0e-5,
        "persistent CPU/GPU error was {persistent_error}"
    );

    let generated_source = export_generated_source();
    println!("Signal pipeline benchmark");
    println!(
        "kernel: {}",
        signal_pipeline::transform::DESCRIPTOR.qualified_name()
    );
    println!("adapter: {}", device.adapter_info().name);
    println!("elements: {element_count}");
    println!("operations per element: 3 mul, 1 sub, 2 add, 1 div");
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
}

struct Inputs {
    signal: Vec<f32>,
    baseline: Vec<f32>,
    gain: Vec<f32>,
    bias: Vec<f32>,
    scale: Vec<f32>,
}

impl Inputs {
    fn new(element_count: usize) -> Self {
        let indices = 0..element_count;
        Self {
            signal: indices
                .clone()
                .map(|index| (index % 2048) as f32 / 256.0 - 4.0)
                .collect(),
            baseline: indices
                .clone()
                .map(|index| (index % 31) as f32 * 0.01 - 0.15)
                .collect(),
            gain: indices
                .clone()
                .map(|index| 0.75 + (index % 17) as f32 * 0.025)
                .collect(),
            bias: indices
                .clone()
                .map(|index| (index % 13) as f32 * 0.02 - 0.12)
                .collect(),
            scale: indices
                .map(|index| 1.0 + (index % 23) as f32 * 0.03)
                .collect(),
        }
    }
}

fn run_cpu(inputs: &Inputs, output: &mut [f32]) {
    for (index, output) in output.iter_mut().enumerate() {
        let centered = inputs.signal[index] - inputs.baseline[index];
        let amplified = centered * inputs.gain[index];
        let biased = amplified + inputs.bias[index];
        let normalized = biased / inputs.scale[index];
        let energy = normalized * normalized;
        *output = energy + 0.25 * normalized;
    }
}

fn dispatch_gpu(
    device: &HeadlessDevice,
    inputs: &Inputs,
    initial_output: &[f32],
) -> Result<Vec<f32>, gust_wgpu::Error> {
    let read_only =
        |label, values: &[f32]| device.create_f32_buffer(label, values, GpuBufferAccess::ReadOnly);
    let signal = read_only("signal one-shot", &inputs.signal);
    let baseline = read_only("baseline one-shot", &inputs.baseline);
    let gain = read_only("gain one-shot", &inputs.gain);
    let bias = read_only("bias one-shot", &inputs.bias);
    let scale = read_only("scale one-shot", &inputs.scale);
    let output = device.create_f32_buffer(
        "signal output one-shot",
        initial_output,
        GpuBufferAccess::ReadWrite,
    );
    signal_pipeline::transform::dispatch(
        device,
        inputs.signal.len() as u32,
        &signal,
        &baseline,
        &gain,
        &bias,
        &scale,
        &output,
    )?;
    device.read_f32_buffer(&output)
}

fn benchmark_element_count() -> usize {
    if cfg!(debug_assertions) {
        DEBUG_COUNT
    } else {
        RELEASE_COUNT
    }
}

fn max_abs_error(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
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
    let descriptor = &signal_pipeline::transform::DESCRIPTOR;
    let path = directory.join("signal_pipeline__transform.rs");
    let source = render_wgpu_source(descriptor).expect("render signal pipeline wgpu source");
    std::fs::write(&path, source).expect("write signal pipeline wgpu source");
    std::fs::write(
        directory.join("signal_pipeline__transform.slang"),
        descriptor.slang_source,
    )
    .expect("write signal pipeline Slang source");
    std::fs::write(
        directory.join("signal_pipeline__transform.wgsl"),
        gust::slang::compile_wgsl(descriptor).expect("compile signal pipeline WGSL"),
    )
    .expect("write signal pipeline WGSL");
    std::fs::write(
        directory.join("signal_pipeline__transform.spv"),
        gust::spirv::words_as_le_bytes(
            &gust::slang::compile_spirv(descriptor).expect("compile signal pipeline SPIR-V"),
        ),
    )
    .expect("write signal pipeline SPIR-V");
    path.canonicalize().unwrap_or(path)
}

#[cfg(test)]
mod tests {
    use gust::cpu::{CpuDevice, Error as CpuError, ulps};

    use super::*;

    /// The twin-versus-GPU bound for `transform`, absolute, over the input domain of `Inputs`: 900 * 2^-24, about
    /// 5.4e-5.
    ///
    /// Derived from the kernel text (docs/SEMANTICS.md section 4, "Deriving a tolerance"), not from a measurement.
    /// With `u = 2^-24`, the domain below (asserted by `assert_derivation_domain`) and one rounding of at most `u`
    /// times the result's magnitude on the twin, `2u` on the GPU for `+ - *` and `5u` for `/` (divisor in
    /// `[1, 1.66]`, inside the `[2^-126, 2^126]` range of the 2.5 ULP row), first-order propagation, magnitudes
    /// `M`, twin error `T` and GPU error `G` in units of `u`, the GPU side without fusion (its maximum):
    ///
    /// | step | operation | M | T | G |
    /// | --- | --- | --- | --- | --- |
    /// | `centered` | `signal - baseline`, `abs(signal) <= 4`, `abs(baseline) <= 0.16` | 4.16 | 4.16 | 8.32 |
    /// | `amplified` | `* gain`, `gain <= 1.16` | 4.83 | 9.65 | 19.30 |
    /// | `biased` | `+ bias`, `abs(bias) <= 0.13` | 4.96 | 14.61 | 29.21 |
    /// | `normalized` | `/ scale`, `scale >= 1`, so the inputs' error is divided by at most 1 | 4.96 | 19.56 | 53.99 |
    /// | `energy` | `normalized * normalized`, error `2 M T` plus the rounding | 24.56 | 218.44 | 584.24 |
    /// | `out` | `energy + 0.25 * normalized` (the product is exact: a power of two) | 25.80 | 249.13 | 649.33 |
    ///
    /// Twin plus GPU is 898.46u, so 899u rounded up, and 900u asserted, the extra 1u covering the second-order terms
    /// (below 0.001u here). Reassociation cannot raise it: every sum has two operands, and a chain of products has
    /// the same relative error in any grouping. Fusing `centered * gain + bias` or `energy + 0.25 * normalized`
    /// only removes a rounding. A sign change near a root stays inside the bound.
    ///
    /// Observed maxima are printed separately by the ignored test and never become the tolerance.
    const TWIN_VS_GPU_BOUND: f64 = 900.0 / 16_777_216.0;

    /// The domain the bound is derived for: every input of `Inputs::new` lies in these ranges. All inputs are periodic
    /// with periods of at most 2048, so 4096 elements reach every extreme.
    fn assert_derivation_domain(inputs: &Inputs) {
        let largest = |values: &[f32]| values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(largest(&inputs.signal) <= 4.0, "signal");
        assert!(largest(&inputs.baseline) <= 0.16, "baseline");
        assert!(largest(&inputs.gain) <= 1.16, "gain");
        assert!(largest(&inputs.bias) <= 0.13, "bias");
        assert!(
            inputs.scale.iter().all(|scale| *scale >= 1.0),
            "scale (the divisor) is at least 1"
        );
    }

    /// The host reference: `run_cpu` over a fresh output.
    fn reference(inputs: &Inputs) -> Vec<f32> {
        let mut output = vec![0.0; inputs.signal.len()];
        run_cpu(inputs, &mut output);
        output
    }

    /// The same typed `dispatch` on the CPU twin, which the macro emits under `cfg(test)`: no GPU involved.
    fn transform_on_cpu(inputs: &Inputs) -> Result<Vec<f32>, CpuError> {
        let device = CpuDevice::new();
        let signal = device.buffer(&inputs.signal);
        let baseline = device.buffer(&inputs.baseline);
        let gain = device.buffer(&inputs.gain);
        let bias = device.buffer(&inputs.bias);
        let scale = device.buffer(&inputs.scale);
        let out = device.buffer(&vec![-1.0f32; inputs.signal.len()]);
        signal_pipeline::transform::dispatch(
            &device,
            inputs.signal.len() as u32,
            &signal,
            &baseline,
            &gain,
            &bias,
            &scale,
            &out,
        )?;
        Ok(out.read())
    }

    /// The largest distance between two results in ULPs and in absolute terms (computed in `f64`).
    fn largest_difference(left: &[f32], right: &[f32]) -> (u64, f64) {
        assert_eq!(left.len(), right.len());
        left.iter().zip(right).fold((0, 0.0), |(ulp, abs), (l, r)| {
            (
                ulp.max(ulps(*l, *r)),
                abs.max((f64::from(*l) - f64::from(*r)).abs()),
            )
        })
    }

    #[test]
    fn inputs_stay_inside_the_derivation_domain() {
        assert_derivation_domain(&Inputs::new(4096));
        assert_derivation_domain(&Inputs::new(DEBUG_COUNT));
    }

    #[test]
    fn twin_matches_host_reference() {
        // The twin runs the shadow body, which performs the operations of `run_cpu` in the same order, and Rust
        // does not contract them, so the two agree to the last bit. 257 elements end mid-workgroup of 128 (the
        // kernel's `i < out.len()` guard keeps the tail inside the buffer); 0 and 1 are the edges.
        for count in [0, 1, 127, 128, 129, 257, 4096] {
            let inputs = Inputs::new(count);
            let twin = transform_on_cpu(&inputs).expect("twin dispatch");
            let (max_ulps, max_abs) = largest_difference(&twin, &reference(&inputs));
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
        let inputs = Inputs::new(DEBUG_COUNT);
        assert_derivation_domain(&inputs);
        let host = reference(&inputs);
        let gpu = dispatch_gpu(&device, &inputs, &vec![0.0; DEBUG_COUNT]).expect("GPU dispatch");
        let twin = transform_on_cpu(&inputs).expect("twin dispatch");

        let (gpu_host_ulps, gpu_host_abs) = largest_difference(&gpu, &host);
        let (twin_host_ulps, twin_host_abs) = largest_difference(&twin, &host);
        let (twin_gpu_ulps, twin_gpu_abs) = largest_difference(&twin, &gpu);
        // Observations on this adapter, recorded in the slice log (probe/s4-e1.log); not tolerances.
        println!("S4 adapter: {}", device.adapter_info().name);
        println!("S4 GPU versus run_cpu: max {gpu_host_ulps} ULP, max abs error {gpu_host_abs:e}");
        println!(
            "S4 twin versus run_cpu: max {twin_host_ulps} ULP, max abs error {twin_host_abs:e}"
        );
        println!("S4 twin versus GPU: max {twin_gpu_ulps} ULP, max abs error {twin_gpu_abs:e}");
        println!("S4 derived bound (900 * 2^-24): {TWIN_VS_GPU_BOUND:e}");

        // The GPU against the host reference is the same pair as the twin against the GPU, because the twin
        // equals the host reference bit for bit; it is asserted first so that a failing twin cannot hide it.
        assert!(
            gpu_host_abs <= TWIN_VS_GPU_BOUND,
            "GPU versus run_cpu: max abs error {gpu_host_abs:e} exceeds the derived bound {TWIN_VS_GPU_BOUND:e}"
        );
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
                .div_ceil(signal_pipeline::transform::DESCRIPTOR.workgroup_size[0]);
            assert!(
                groups <= gust_wgpu::DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
                "{count} elements need {groups} workgroups"
            );
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn cpu_reference_runs_the_full_pipeline() {
        let inputs = Inputs::new(8);
        let mut output = [0.0; 8];
        run_cpu(&inputs, &mut output);
        for (index, actual) in output.into_iter().enumerate() {
            let centered = inputs.signal[index] - inputs.baseline[index];
            let normalized =
                (centered * inputs.gain[index] + inputs.bias[index]) / inputs.scale[index];
            let expected = normalized * normalized + 0.25 * normalized;
            assert_eq!(actual, expected);
        }
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
        let inputs = Inputs::new(257);
        let mut cpu_output = vec![0.0; 257];
        let initial_output = vec![0.0; 257];
        run_cpu(&inputs, &mut cpu_output);
        let gpu_output =
            dispatch_gpu(&device, &inputs, &initial_output).expect("GPU signal dispatch");
        assert!(max_abs_error(&cpu_output, &gpu_output) <= 2.0e-5);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn descriptor_and_generated_host_code_are_executable() {
        let descriptor = signal_pipeline::transform::DESCRIPTOR;
        assert!(
            descriptor
                .slang_source
                .contains("void gpu_signal_pipeline_transform")
        );
        let source = render_wgpu_source(&descriptor).expect("render source");
        assert!(source.contains("pub struct SignalPipelineTransformPipeline"));
        assert!(source.contains("pub fn encode("));
        assert!(source.contains("binding: 5,"));
        assert!(source.contains("scale.as_entire_binding()"));
    }
}
