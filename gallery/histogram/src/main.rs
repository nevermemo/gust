// ANCHOR: imports
use gust::{Independent, gpu};
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};
// ANCHOR_END: imports

// ANCHOR: kernel
#[gpu]
mod histogram {
    /// Every invocation adds one to the bin of its key. Many invocations hit the same few bins at once, so the
    /// increment must be atomic: a plain `bins[k] += 1` would lose counts on a GPU.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn count(
        id: SV_DispatchThreadID,
        keys: StructuredBuffer<uint>,
        mut bins: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < keys.len() {
            let bin = keys[i] % bins.len();
            atomic_add(&mut bins[bin], 1u32);
        }
    }
}
// ANCHOR_END: kernel

/// The number of bins, kept small on purpose: with 100 000 keys every bin is hit by about ten thousand invocations.
const BINS: usize = 8;

/// The keys: a multiplicative hash of the index, squared so that the bins are far from equally likely (the counts
/// below are the same on every run, because the keys are).
fn keys(n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| {
            let h = (i as u32).wrapping_mul(2_654_435_761) >> 20;
            (h * h) >> 21
        })
        .collect()
}

/// The independent host reference: a plain count over the keys, which never goes through the kernel (D26, D42).
fn host_count(keys: &[u32], bins: usize) -> Vec<u32> {
    let mut counts = vec![0u32; bins];
    for key in keys {
        counts[*key as usize % bins] += 1;
    }
    counts
}

// ANCHOR: main
fn main() {
    let keys = keys(100_000);
    let device = HeadlessDevice::new().expect("a Vulkan compute adapter");
    let gpu_keys = device.create_typed_buffer("keys", &keys, GpuBufferAccess::ReadOnly);
    let bins = device.create_typed_buffer("bins", &[0u32; BINS], GpuBufferAccess::ReadWrite);
    // One invocation per key; `bins` is shorter than the dispatch, which `Independent` says is meant.
    histogram::count::dispatch(&device, keys.len() as u32, &gpu_keys, Independent(&bins))
        .expect("dispatch");
    let counts = device.read_typed_buffer(&bins).expect("read back");
    assert_eq!(counts, host_count(&keys, BINS), "an atomic count is exact");
    println!("histogram: {} keys in {BINS} bins {counts:?}", keys.len());
}
// ANCHOR_END: main

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use gust::{Independent, cpu::CpuDevice};

    use super::{BINS, histogram, host_count, keys};

    /// The kernel on the CPU twin (the macro emits the same typed `dispatch` for `CpuDevice` under `cfg(test)`). The
    /// twin runs the invocations one after another, so it cannot lose a count; what it shows is that the kernel's
    /// indexing and bin choice are right. Contention itself is the GPU's, checked by `cargo run`.
    fn on_twin(keys: &[u32], bins: usize, threads: u32) -> Vec<u32> {
        let device = CpuDevice::new();
        let gpu_keys = device.buffer(keys);
        let counts = device.buffer(&vec![0u32; bins]);
        histogram::count::dispatch(&device, threads, &gpu_keys, Independent(&counts))
            .expect("the twin dispatches");
        counts.read()
    }

    #[test]
    fn the_twin_matches_hand_counted_values() {
        // Keys 0 1 2 3 4 5 5 5 into 4 bins: bin 0 gets 0 and 4, bin 1 gets 1 and the three 5s, then 2 and 3.
        let counts = on_twin(&[0, 1, 2, 3, 4, 5, 5, 5], 4, 8);
        assert_eq!(counts, [2, 4, 1, 1]);
    }

    #[test]
    fn the_twin_matches_the_host_count_exactly() {
        // Sizes below, at and above the workgroup size of 64, ones that are not a multiple of it, and a large one.
        for n in [1usize, 63, 64, 65, 1000, 100_000] {
            let keys = keys(n);
            for bins in [1usize, 2, BINS, 256] {
                let threads = u32::try_from(n).expect("small");
                let counts = on_twin(&keys, bins, threads);
                assert_eq!(counts, host_count(&keys, bins), "n = {n}, bins = {bins}");
                assert_eq!(
                    counts.iter().sum::<u32>(),
                    threads,
                    "every key is counted once, n = {n}"
                );
            }
        }
    }

    #[test]
    fn the_keys_are_skewed() {
        // A guard on the example itself: with equal bins the demonstration of contention would be weaker.
        let counts = host_count(&keys(100_000), BINS);
        let (max, min) = (counts.iter().max().unwrap(), counts.iter().min().unwrap());
        assert!(*max > 4 * *min, "{counts:?}");
    }
}
// ANCHOR_END: tests
