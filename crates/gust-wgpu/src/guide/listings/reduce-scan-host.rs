/// The lanes of a tile: the workgroup size of the kernels above.
const TILE: usize = 64;

/// The input values: small, so that no sum or prefix of the sizes below leaves `u32`.
fn values(n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| u32::try_from((i * 7 + 3) % 10).expect("below ten"))
        .collect()
}

/// The independent host reference of a sum: a plain loop, which never goes through a kernel (D26, D42).
fn host_sum(values: &[u32]) -> u32 {
    values.iter().sum()
}

/// The independent host reference of an inclusive scan: a running sum.
fn host_scan(values: &[u32]) -> Vec<u32> {
    let mut total = 0u32;
    values
        .iter()
        .map(|value| {
            total += value;
            total
        })
        .collect()
}
