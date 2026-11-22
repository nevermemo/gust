use gust::{Independent, gpu_module};
use gust_wgpu::{GpuBuffer, GpuBufferAccess, HeadlessDevice};

/// Indices hashed by the demo and by the tests. A multiple of the workgroup size, so no thread needs a bounds check.
const COUNT: u32 = 4096;
/// Histogram bins. The kernel masks with `BINS - 1`, so this must stay a power of two (63 is written in the kernel).
const BINS: usize = 64;

#[gpu_module]
mod histogram {
    /// A multiply-xorshift finaliser. Every shift amount is a literal below 32 (D39).
    pub fn mix(x: u32) -> u32 {
        let mut h = x ^ (x >> 16u32);
        h = h.wrapping_mul(2146121005u32);
        h = h ^ (h >> 15u32);
        h = h.wrapping_mul(2221713035u32);
        h ^ (h >> 16u32)
    }

    /// Hashes the thread index into one of 64 bins and counts it. Integer atomics are exact, so the counts do not
    /// depend on the order the threads run in.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn count(id: SV_DispatchThreadID, mut bins: RWStructuredBuffer<uint>) {
        let bin = mix(id.x) & 63u32;
        atomic_add(&mut bins[bin], 1u32);
    }
}

/// The consumer: a compute pass written by hand in WGSL, not by GUST. One thread per bin; each bin that differs adds one
/// to the single mismatch counter.
const COMPARE_WGSL: &str = r"
@group(0) @binding(0) var<storage, read> actual: array<u32>;
@group(0) @binding(1) var<storage, read> expected: array<u32>;
@group(0) @binding(2) var<storage, read_write> mismatches: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&actual) && actual[id.x] != expected[id.x]) {
        atomicAdd(&mismatches[0], 1u);
    }
}
";

/// The histogram as the CPU computes it, with its own copy of the arithmetic (plain Rust, outside the module).
fn reference_histogram() -> Vec<u32> {
    fn mix(x: u32) -> u32 {
        let mut h = x ^ (x >> 16);
        h = h.wrapping_mul(2146121005);
        h ^= h >> 15;
        h = h.wrapping_mul(2221713035);
        h ^ (h >> 16)
    }
    let mut bins = vec![0u32; BINS];
    for i in 0..COUNT {
        bins[(mix(i) & (BINS as u32 - 1)) as usize] += 1;
    }
    bins
}

/// The kernel's CPU twin, which the macro emits under `cfg(test)`.
#[cfg(test)]
fn twin_histogram() -> Result<Vec<u32>, gust::cpu::Error> {
    use gust::cpu::CpuDevice;
    let device = CpuDevice::new();
    let bins = device.buffer(&vec![0u32; BINS]);
    histogram::count::dispatch(&device, COUNT, Independent(&bins))?;
    Ok(bins.read())
}

/// Counts the bins of `actual` that differ from `expected`. `actual` is used by the compare pass straight from the
/// device: the only readback is the one-element mismatch counter.
fn count_mismatches(
    device: &HeadlessDevice,
    actual: &GpuBuffer<u32>,
    expected: &[u32],
) -> Result<u32, Box<dyn std::error::Error>> {
    let expected = device.create_typed_buffer("expected", expected, GpuBufferAccess::ReadOnly);
    // Read-write so that the host can read it back; it is the only buffer that is.
    let mismatches = device.create_typed_buffer("mismatches", &[0u32], GpuBufferAccess::ReadWrite);

    let raw = device.wgpu_device();
    let shader = raw.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("compare shader"),
        source: wgpu::ShaderSource::Wgsl(COMPARE_WGSL.into()),
    });
    let pipeline = raw.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compare pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = raw.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compare bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                // `as_entire_binding()` is right only because these three buffers are unpooled (`create_typed_buffer`): a
                // pooled `GpuVec` buffer is a whole size class and would need a sized binding.
                resource: actual.as_wgpu_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: expected.as_wgpu_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: mismatches.as_wgpu_buffer().as_entire_binding(),
            },
        ],
    });
    let mut encoder = raw.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("compare encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compare pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((actual.len() as u32).div_ceil(64), 1, 1);
    }
    // Not waited on: the readback below is a later submission on the same queue, so it runs after this one.
    device.wgpu_queue().submit([encoder.finish()]);
    Ok(device.read_typed_buffer(&mismatches)?[0])
}

/// How the GUST kernel is started before the example records its own pass.
#[derive(Clone, Copy)]
enum Handoff {
    /// The typed `dispatch`, which waits for its submission.
    Waited,
    /// The typed `submit`, whose job is dropped unwaited: only the queue order links the two submissions.
    Unwaited,
}

/// Runs the GUST kernel, then hands its histogram buffer to the compare pass with `expected`.
fn run(
    device: &HeadlessDevice,
    expected: &[u32],
    handoff: Handoff,
) -> Result<u32, Box<dyn std::error::Error>> {
    let bins = device.create_typed_buffer("histogram", &[0u32; BINS], GpuBufferAccess::ReadWrite);
    // The histogram has `BINS` elements, not `COUNT`: `Independent` is the call-site opt-in for that.
    match handoff {
        Handoff::Waited => histogram::count::dispatch(device, COUNT, Independent(&bins))?,
        Handoff::Unwaited => drop(histogram::count::submit(device, COUNT, Independent(&bins))?),
    }
    count_mismatches(device, &bins, expected)
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = HeadlessDevice::new()?;
    let info = device.adapter_info();
    println!("adapter: {} backend: {:?}", info.name, info.backend);
    let expected = reference_histogram();
    println!(
        "matching expected buffer: {} mismatches",
        run(&device, &expected, Handoff::Waited)?
    );
    let mut wrong = expected;
    wrong[0] += 1;
    println!(
        "one bin changed: {} mismatches",
        run(&device, &wrong, Handoff::Waited)?
    );
    println!(
        "one bin changed, kernel submitted without waiting: {} mismatches",
        run(&device, &wrong, Handoff::Unwaited)?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> HeadlessDevice {
        let device = HeadlessDevice::new().unwrap_or_else(|error| {
            panic!(
                "Vulkan adapter required; this test must not silently skip: could not initialize headless wgpu: {error}"
            )
        });
        let info = device.adapter_info();
        println!("adapter: {} backend: {:?}", info.name, info.backend);
        device
    }

    #[test]
    fn the_reference_counts_every_index_once() {
        let bins = reference_histogram();
        assert_eq!(bins.len(), BINS);
        assert_eq!(bins.iter().sum::<u32>(), COUNT);
        assert!(bins.iter().all(|&count| count > 0), "{bins:?}");
    }

    #[test]
    fn twin_matches_host_reference() {
        assert_eq!(twin_histogram().unwrap(), reference_histogram());
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_histogram_consumed_in_place_has_zero_mismatches() {
        let device = device();
        assert_eq!(
            run(&device, &reference_histogram(), Handoff::Waited).unwrap(),
            0
        );
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn one_changed_bin_gives_exactly_one_mismatch() {
        let device = device();
        let mut wrong = reference_histogram();
        wrong[17] += 1;
        assert_eq!(run(&device, &wrong, Handoff::Waited).unwrap(), 1);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn unwaited_submit_then_the_compare_pass_has_zero_mismatches() {
        let device = device();
        let result = run(&device, &reference_histogram(), Handoff::Unwaited).unwrap();
        assert_eq!(result, 0);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn unwaited_submit_with_one_changed_bin_gives_exactly_one_mismatch() {
        let device = device();
        let mut wrong = reference_histogram();
        wrong[17] += 1;
        assert_eq!(run(&device, &wrong, Handoff::Unwaited).unwrap(), 1);
    }
}
