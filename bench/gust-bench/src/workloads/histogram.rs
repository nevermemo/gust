//! W4: histogram, `bins[keys[i]] += 1` with a `u32` atomic add, 256 or 65536 bins, uniform or Zipf keys. The one
//! GPU-specific feature every comparator has: many invocations contend on few counters (Zipf keys make a handful of
//! bins hot). The result is an integer count per bin, so the check is exact equality.
//!
//! One invocation takes [`KEYS_PER_THREAD`](super::KEYS_PER_THREAD) = 4 keys, strided by 256 inside a block of 1024
//! contiguous keys that a workgroup owns (coalesced loads). That is what lets 2^24 keys launch inside the default
//! 65535-workgroup cap. The kernel reads its bounds from the buffers (`keys.len()`, `bins.len()`), like W1 and W2.
//!
//! The device output accumulates: a dispatch adds into the bins and never clears them, so a sample zeroes the bins
//! first (the harness's `poison`, and `end_to_end` clears inside its timed span because a host call needs a clean
//! result). After `p` dispatches since the last clear every count is `p` times the reference.

use gust::gpu;

/// The GUST kernel. Workgroup size 256 (see `workloads::WORKGROUP`).
#[gpu]
pub mod kernel {
    #[kernel(workgroup_size(256, 1, 1))]
    pub fn histogram(
        id: SV_DispatchThreadID,
        keys: StructuredBuffer<uint>,
        mut bins: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        let n = keys.len();
        let first = (i / 256u32) * 1024u32 + i % 256u32;
        for k in 0u32..4u32 {
            let index = first + k * 256u32;
            if index >= n {
                break;
            }
            let bin = keys[index];
            if bin < bins.len() {
                atomic_add(&mut bins[bin], 1u32);
            }
        }
    }
}

/// The serial reference: the count of each bin.
pub fn counts(keys: &[u32], bins: usize) -> Vec<u32> {
    let mut counts = vec![0u32; bins];
    for key in keys {
        counts[*key as usize] += 1;
    }
    counts
}

/// Zipf-distributed keys, exponent 1: bin `k` has probability proportional to `1 / (k + 1)`, drawn by inverse CDF from
/// a well-mixed counter, so the stream is deterministic and independent of the thread count.
pub fn zipf_keys(n: usize, bins: usize) -> Vec<u32> {
    let mut cdf = Vec::with_capacity(bins);
    let mut total = 0.0f64;
    for k in 0..bins {
        total += 1.0 / (k as f64 + 1.0);
        cdf.push(total);
    }
    (0..n as u64)
        .map(|i| {
            let u =
                (super::mix(i ^ 0x5a17_5a17_5a17_5a17) >> 11) as f64 / (1u64 << 53) as f64 * total;
            // The first bin whose cumulative weight exceeds `u`.
            cdf.partition_point(|c| *c <= u).min(bins - 1) as u32
        })
        .collect()
}
