/// Runs the selected body over `n` values on `device`, asserts the three cells against the host reference and returns
/// them.
fn check(device: &HeadlessDevice, n: usize, body: &str) -> Vec<u32> {
    let data = values(n);
    let gpu_data = device.create_typed_buffer("data", &data, GpuBufferAccess::ReadOnly);
    let cells = device.create_typed_buffer("cells", &[0u32; 3], GpuBufferAccess::ReadWrite);
    let invocations = u32::try_from(n).expect("fewer than 2^32 values");
    stats::tally::dispatch(device, invocations, &gpu_data, Independent(&cells)).expect("dispatch");
    let got = device.read_typed_buffer(&cells).expect("read back");
    assert_eq!(got, host_cells(&data), "the cells of body={body}, n = {n}");
    got
}

fn main() {
    let body = if subgroup_body_selected() {
        "subgroups"
    } else {
        "portable"
    };
    // A device opened with the capabilities of the build. An adapter without subgroup operations cannot open one
    // under `native+subgroups`: that is a typed error here, never a wrong result. The portable build runs on every
    // adapter.
    let device = match HeadlessDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("variants: no device for body={body}: {error}");
            std::process::exit(1);
        }
    };
    // Sizes that leave the last workgroup and the last subgroup partly past the end of the data: 1, 37 (a subgroup of
    // 32 holds 5 real values) and 100 003 (3 real values in the last subgroup of 8 and of 32), then 100 000.
    for n in [1, 37, 100_003] {
        check(&device, n, body);
    }
    let got = check(&device, 100_000, body);
    println!(
        "variants: body={body}, 100000 values, sum {}, {} over 100, max {}",
        got[0], got[1], got[2]
    );
    println!("subgroup sizes: {:?}", device.subgroup_sizes());
    println!("adapter: {}", device.adapter_info().name);
}
