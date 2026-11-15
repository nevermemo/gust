/// The values: small numbers, a bit under half of them over 100.
fn values(n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| (u32::try_from(i).expect("fewer than 2^32 values") * 37 + 11) % 251)
        .collect()
}

/// The three cells as a host computes them, with plain iterators that never go through the kernel (D26, D42).
fn host_cells(data: &[u32]) -> [u32; 3] {
    [
        data.iter().sum(),
        u32::try_from(data.iter().filter(|&&x| x > 100).count()).expect("fewer than 2^32 values"),
        data.iter().copied().max().unwrap_or(0),
    ]
}

/// Whether the body this build selected is the subgroup one (the build has the `subgroups` capability).
fn subgroup_body_selected() -> bool {
    stats::tally::DESCRIPTOR
        .requires
        .contains(&Capability::Subgroups)
}
