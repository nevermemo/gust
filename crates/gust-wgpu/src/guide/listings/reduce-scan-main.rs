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
