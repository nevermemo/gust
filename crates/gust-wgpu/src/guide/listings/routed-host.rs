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
