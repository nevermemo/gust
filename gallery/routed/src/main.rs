// ANCHOR: imports
use gust::gpu;
use gust::routed::RoutedDevice;
use gust_wgpu::HeadlessDevice;
// ANCHOR_END: imports

// ANCHOR: kernel
/// Stage one: saturate every sample at 255. The tag `tidy` is declared in `Cargo.toml`, and `gust.toml` routes it to
/// the CPU: a small, cheap stage that is not worth a trip through the GPU.
#[gpu(route = "tidy")]
mod tidy {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn saturate(
        id: SV_DispatchThreadID,
        raw: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < raw.len() && i < out.len() {
            let mut v = raw[i];
            if v > 255u32 {
                v = 255u32;
            }
            out[i] = v;
        }
    }
}

/// Stage two: a weighted sum of each sample and its two neighbours (weights 1, 2, 1; the ends repeat themselves). The
/// tag `smooth` is routed to the GPU.
#[gpu(route = "smooth")]
mod smooth {
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn blur(
        id: SV_DispatchThreadID,
        input: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        let n = input.len();
        if i < n && i < out.len() {
            let mut left = i;
            if i > 0u32 {
                left = i - 1u32;
            }
            let mut right = i;
            if i + 1u32 < n {
                right = i + 1u32;
            }
            out[i] = input[left] + 2u32 * input[i] + input[right];
        }
    }
}
// ANCHOR_END: kernel

// ANCHOR: host
/// The samples: some of them over 255, so that the first stage has work to do.
fn samples(n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| (u32::try_from(i).expect("fewer than 2^32 samples") * 37 + 11) % 400)
        .collect()
}

/// The independent host reference: both stages as plain Rust over a slice, never through the kernels (D26, D42).
fn host_reference(raw: &[u32]) -> Vec<u32> {
    let tidy: Vec<u32> = raw.iter().map(|&v| v.min(255)).collect();
    (0..tidy.len())
        .map(|i| {
            let left = tidy[i.saturating_sub(1)];
            let right = tidy[(i + 1).min(tidy.len() - 1)];
            left + 2 * tidy[i] + right
        })
        .collect()
}
// ANCHOR_END: host

// ANCHOR: main
fn main() {
    let n = 1000;
    let raw = samples(n);
    let gpu = HeadlessDevice::new().expect("a Vulkan compute adapter");
    println!("adapter: {}", gpu.adapter_info().name);
    // One handle for both stages: each call runs on the target its kernel was built for, and the buffers move between
    // the CPU and the GPU as the calls need. Nothing below names a target, so moving a stage in `gust.toml` changes no
    // line.
    let device = RoutedDevice::with_gpu(gpu);
    let raw_buffer = device.buffer(&raw).expect("input buffer");
    let tidy_buffer = device.buffer(&vec![0u32; n]).expect("middle buffer");
    let out_buffer = device.buffer(&vec![0u32; n]).expect("output buffer");
    let count = u32::try_from(n).expect("fewer than 2^32 samples");
    tidy::saturate::dispatch(&device, count, &raw_buffer, &tidy_buffer).expect("stage one");
    smooth::blur::dispatch(&device, count, &tidy_buffer, &out_buffer).expect("stage two");
    let result = out_buffer.read(&device).expect("read back");
    assert_eq!(result, host_reference(&raw), "both stages are exact");
    let checksum: u64 = result.iter().map(|&v| u64::from(v)).sum();
    println!(
        "routed: {n} samples, tidy on {:?}, smooth on {:?}, checksum {checksum}",
        tidy::saturate::DESCRIPTOR.target,
        smooth::blur::DESCRIPTOR.target
    );
}
// ANCHOR_END: main

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use gust::cpu::CpuDevice;
    use gust::routed::{Error, NoGpu, RoutedDevice};

    use super::{host_reference, samples, smooth, tidy};

    /// Both stages on the CPU twin: `tidy` is CPU-routed, so it runs there in every build; `smooth` is GPU-routed, and
    /// the test build gives it a twin (D26) that runs the same body on `CpuDevice`.
    fn on_twin(raw: &[u32], threads: u32) -> Vec<u32> {
        let device = CpuDevice::new();
        let input = device.buffer(raw);
        let middle = device.buffer(&vec![0u32; raw.len()]);
        let out = device.buffer(&vec![0u32; raw.len()]);
        tidy::saturate::dispatch(&device, threads, &input, &middle).expect("stage one");
        smooth::blur::dispatch(&device, threads, &middle, &out).expect("stage two on the twin");
        out.read()
    }

    #[test]
    fn the_twin_matches_hand_computed_values() {
        // 300 saturates to 255. Neighbours of the middle value: 10 + 2 * 255 + 20 = 540; the ends repeat themselves.
        let got = on_twin(&[10, 300, 20], 3);
        assert_eq!(got, [10 + 2 * 10 + 255, 540, 255 + 2 * 20 + 20]);
    }

    #[test]
    fn the_twin_matches_the_host_reference_exactly() {
        // Sizes below, at and above the workgroup size of 64, ones that are not a multiple of it, and 1000 samples.
        for n in [1usize, 2, 63, 64, 65, 1000] {
            let raw = samples(n);
            let got = on_twin(&raw, u32::try_from(n).expect("small"));
            assert_eq!(got, host_reference(&raw), "n = {n}");
        }
    }

    #[test]
    fn the_targets_are_the_routes_of_gust_toml() {
        // Plain `cargo test` builds the declared defaults, which `gust.toml` repeats; `cargo gust test` builds the
        // file.
        assert_eq!(format!("{:?}", tidy::saturate::DESCRIPTOR.target), "Cpu");
        assert_eq!(format!("{:?}", smooth::blur::DESCRIPTOR.target), "Gpu");
    }

    #[test]
    fn a_handle_without_a_gpu_runs_the_cpu_stage_and_refuses_the_gpu_one() {
        // The handle follows the route: no fallback (D65). The CPU-routed stage runs, the GPU-routed one is a typed
        // error that leaves its output untouched.
        let device = RoutedDevice::<NoGpu>::cpu_only();
        let raw = samples(70);
        let input = device.buffer(&raw).expect("input");
        let middle = device.buffer(&vec![0u32; raw.len()]).expect("middle");
        let out = device.buffer(&vec![7u32; raw.len()]).expect("out");
        tidy::saturate::dispatch(&device, 70, &input, &middle).expect("the CPU stage runs");
        let refused = smooth::blur::dispatch(&device, 70, &middle, &out);
        assert!(matches!(refused, Err(Error::NoGpu { .. })), "{refused:?}");
        assert_eq!(out.read(&device).expect("read"), vec![7u32; raw.len()]);
        let want: Vec<u32> = raw.iter().map(|&v| v.min(255)).collect();
        assert_eq!(middle.read(&device).expect("read"), want);
    }
}
// ANCHOR_END: tests
