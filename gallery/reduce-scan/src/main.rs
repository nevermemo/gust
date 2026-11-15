// ANCHOR: imports
use gust::{Independent, gpu};
use gust_wgpu::{GpuBuffer, GpuBufferAccess, HeadlessDevice};
// ANCHOR_END: imports

// ANCHOR: kernel
#[gpu]
mod blocks {
    /// The sum of every tile of 64 elements, one tile per workgroup: each lane loads one element into workgroup memory,
    /// then six rounds of a tree (a barrier after the load and after every round) leave the tile's sum in cell 0, which
    /// lane 0 writes to `sums[gid]`. A lane past the end of the input holds 0, so the last tile may be partial.
    ///
    /// Workgroup memory and barriers have no CPU twin (SEMANTICS, D101): the kernel says `cpu = "unsupported"`, and its
    /// result is checked against a host reference on the GPU instead (`main`).
    #[kernel(workgroup_size(64, 1, 1), cpu = "unsupported")]
    pub fn tile_sum(
        _id: SV_DispatchThreadID,
        lid: LocalId,
        gid: GroupId,
        input: StructuredBuffer<uint>,
        mut sums: RWStructuredBuffer<uint>,
        mut scratch: Workgroup<[u32; 64]>,
    ) {
        let i = gid.x * 64u32 + lid.x;
        let mut value = 0u32;
        if i < input.len() {
            value = input[i];
        }
        scratch[lid.x] = value;
        workgroup_barrier();
        for round in 0u32..6u32 {
            let stride = 32u32 >> round;
            if lid.x < stride {
                scratch[lid.x] += scratch[lid.x + stride];
            }
            workgroup_barrier();
        }
        if lid.x == 0u32 {
            sums[gid.x] = scratch[0u32];
        }
    }

    /// The inclusive scan of every tile of 64 elements (Hillis and Steele): round `r` adds the value `2^r` lanes to the
    /// left. Every lane reads its neighbour, a barrier follows, every lane writes its own cell, a barrier follows: the
    /// barriers sit at the top level of the loop body, where every lane reaches them. The tile's total is the last
    /// lane's value (lanes past the input hold 0), written to `totals[gid]`.
    #[kernel(workgroup_size(64, 1, 1), cpu = "unsupported")]
    pub fn tile_scan(
        _id: SV_DispatchThreadID,
        lid: LocalId,
        gid: GroupId,
        input: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
        mut totals: RWStructuredBuffer<uint>,
        mut scratch: Workgroup<[u32; 64]>,
    ) {
        let i = gid.x * 64u32 + lid.x;
        let mut value = 0u32;
        if i < input.len() {
            value = input[i];
        }
        scratch[lid.x] = value;
        workgroup_barrier();
        for round in 0u32..6u32 {
            let offset = 1u32 << round;
            let mut left = 0u32;
            if lid.x >= offset {
                left = scratch[lid.x - offset];
            }
            workgroup_barrier();
            scratch[lid.x] += left;
            workgroup_barrier();
        }
        if i < out.len() {
            out[i] = scratch[lid.x];
        }
        if lid.x == 63u32 {
            totals[gid.x] = scratch[63u32];
        }
    }

    /// Adds to every element of tile `t > 0` the inclusive scan of the tile totals before it (`offsets[t - 1]`). No
    /// workgroup memory, no barrier: this kernel has a CPU twin like the ones of the other entries.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn add_offsets(
        id: SV_DispatchThreadID,
        offsets: StructuredBuffer<uint>,
        mut out: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < out.len() {
            let tile = i / 64u32;
            if tile > 0u32 {
                out[i] += offsets[tile - 1u32];
            }
        }
    }
}
// ANCHOR_END: kernel

// ANCHOR: host
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
// ANCHOR_END: host

// ANCHOR: main
/// The number of invocations of a call over `n` elements: the typed call rounds it up to whole workgroups, and each
/// workgroup is one tile.
fn invocations(n: usize) -> u32 {
    u32::try_from(n).expect("a small input")
}

/// The sum of the first `n` elements of `input`, on the GPU: one `tile_sum` pass per level, each pass leaves one value
/// per tile, until a single tile is left. The source of a pass holds exactly one value per element, so it is bound plainly and the call checks its
/// length; `sums` holds one value per tile, fewer than the pass has invocations, so it is `Independent`.
fn reduce(device: &HeadlessDevice, input: &GpuBuffer<u32>, n: usize) -> u32 {
    let mut levels: Vec<GpuBuffer<u32>> = Vec::new();
    let mut len = n;
    loop {
        let tiles = len.div_ceil(TILE);
        let sums =
            device.create_typed_buffer("sums", &vec![0u32; tiles], GpuBufferAccess::ReadWrite);
        let source = levels.last().unwrap_or(input);
        blocks::tile_sum::dispatch(device, invocations(len), source, Independent(&sums))
            .expect("the sum pass");
        levels.push(sums);
        if tiles == 1 {
            break;
        }
        len = tiles;
    }
    let last = levels.last().expect("at least one level");
    device.read_typed_buffer(last).expect("read back")[0]
}

/// The inclusive scan of the first `n` elements of `input`, on the GPU, as a new buffer: scan every tile, scan the tile
/// totals the same way (a call of this function, one level up), then add each tile's offset.
fn scan(device: &HeadlessDevice, input: &GpuBuffer<u32>, n: usize) -> GpuBuffer<u32> {
    let tiles = n.div_ceil(TILE);
    let out = device.create_typed_buffer("scan", &vec![0u32; n], GpuBufferAccess::ReadWrite);
    let totals =
        device.create_typed_buffer("totals", &vec![0u32; tiles], GpuBufferAccess::ReadWrite);
    blocks::tile_scan::dispatch(device, invocations(n), input, &out, Independent(&totals))
        .expect("the tile scan");
    if tiles > 1 {
        let offsets = scan(device, &totals, tiles);
        blocks::add_offsets::dispatch(device, invocations(n), Independent(&offsets), &out)
            .expect("the offsets pass");
    }
    out
}

/// Both operations on `n` values of the GPU, each asserted against the host reference before it returns.
fn check(device: &HeadlessDevice, n: usize) -> (u32, Vec<u32>) {
    let values = values(n);
    let input = device.create_typed_buffer("input", &values, GpuBufferAccess::ReadOnly);
    let sum = reduce(device, &input, n);
    assert_eq!(sum, host_sum(&values), "the sum, n = {n}");
    let scanned = device
        .read_typed_buffer(&scan(device, &input, n))
        .expect("read back");
    assert_eq!(scanned, host_scan(&values), "the scan, n = {n}");
    (sum, scanned)
}

/// Sizes below, at and above the tile of 64, ones that are not a multiple of it (a partial last workgroup), 4097 (65
/// tiles: the totals need a second tile, so the scan recurses) and 100 000 (three levels).
const SIZES: [usize; 8] = [1, 63, 64, 65, 1000, 4096, 4097, 100_000];

fn main() {
    let device = HeadlessDevice::new().expect("a Vulkan compute adapter");
    let results: Vec<_> = SIZES.iter().map(|n| check(&device, *n)).collect();
    let (sum, scanned) = results.last().expect("there are sizes");
    let sum = *sum;
    let checksum: u64 = scanned.iter().map(|value| u64::from(*value)).sum();
    println!(
        "reduce-scan: {} values in {} sizes, sum {sum}, scan ends at {} with checksum {checksum}",
        scanned.len(),
        SIZES.len(),
        scanned.last().expect("a non-empty scan")
    );
    println!("adapter: {}", device.adapter_info().name);
}
// ANCHOR_END: main

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use gust::{Independent, cpu::CpuDevice};
    use gust_wgpu::HeadlessDevice;

    use super::{SIZES, TILE, blocks, check, host_scan, host_sum, values};

    #[test]
    fn the_host_references_match_hand_computed_values() {
        // Values 3, 0, 7, 4, 1 (`(7 i + 3) % 10`): the sum is 15 and the running sums are 3, 3, 10, 14, 15.
        let five = values(5);
        assert_eq!(five, [3, 0, 7, 4, 1]);
        assert_eq!(host_sum(&five), 15);
        assert_eq!(host_scan(&five), [3, 3, 10, 14, 15]);
        assert_eq!(host_sum(&[]), 0);
        assert_eq!(host_scan(&[]), Vec::<u32>::new());
    }

    /// The one kernel of the entry with a twin: `add_offsets` on the CPU twin, against the host scan. The input is the
    /// scan of each tile alone (made by the host), the offsets the scan of the tile totals (also by the host), and the
    /// sizes include ones with a partial last tile.
    #[test]
    fn the_add_offsets_twin_completes_the_host_scan_exactly() {
        for n in [1usize, 63, 64, 65, 129, 1000, 4097] {
            let values = values(n);
            let tile_scans: Vec<u32> = values.chunks(TILE).flat_map(host_scan).collect();
            let totals: Vec<u32> = values.chunks(TILE).map(host_sum).collect();
            let offsets = host_scan(&totals);
            let device = CpuDevice::new();
            let out = device.buffer(&tile_scans);
            blocks::add_offsets::dispatch(
                &device,
                u32::try_from(n).expect("small"),
                Independent(&device.buffer(&offsets)),
                &out,
            )
            .expect("the twin dispatches");
            assert_eq!(out.read(), host_scan(&values), "n = {n}");
        }
    }

    /// The barrier kernels have no twin, so this is the test of them: the GPU against the host at every size, exact.
    /// It needs an adapter and fails without one (it never skips silently), so it only runs on request:
    /// `cargo test -- --ignored`. `cargo run --release` runs the same `check` for every size, which is the evidence
    /// `cargo xtask check-gallery` collects.
    #[test]
    #[ignore = "needs a Vulkan compute adapter"]
    fn the_gpu_matches_the_host_references_exactly() {
        let device = HeadlessDevice::new().expect("a Vulkan compute adapter");
        for n in SIZES {
            check(&device, n);
        }
    }
}
// ANCHOR_END: tests
