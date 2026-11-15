// ANCHOR: imports
use gust::capability::Capability;
use gust::{Independent, gpu};
use gust_wgpu::{GpuBufferAccess, HeadlessDevice};
// ANCHOR_END: imports

// ANCHOR: kernel
#[gpu]
mod stats {
    /// Sum, count over a limit and maximum of a stream, into three cells. The portable body: every invocation does its
    /// own atomic operations, so the three cells see one operation per element.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn tally(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<uint>,
        mut cells: RWStructuredBuffer<uint>,
    ) {
        if id.x < data.len() {
            let x = data[id.x];
            atomic_add(&mut cells[0u32], x);
            if x > 100u32 {
                atomic_add(&mut cells[1u32], 1u32);
            }
            atomic_max(&mut cells[2u32], x);
        }
    }

    /// The same kernel for a device with subgroup operations: the lanes of a subgroup combine their values first, and
    /// one lane of the subgroup does the atomic operations. The signature is the portable body's, token for token. An
    /// invocation past the end of the data holds 0 and still takes part in the subgroup operations (a 0 changes none of
    /// the three results), and every subgroup operation is the whole value of a `let` at the top level of the body.
    #[variant(of = tally, requires(subgroups))]
    pub fn tally_subgroups(
        id: SV_DispatchThreadID,
        data: StructuredBuffer<uint>,
        mut cells: RWStructuredBuffer<uint>,
    ) {
        let mut x = 0u32;
        if id.x < data.len() {
            x = data[id.x];
        }
        let sum = subgroup_add(x);
        let over = subgroup_count(x > 100u32);
        let top = subgroup_max(x);
        let lane = subgroup_lane();
        if lane == 0u32 {
            atomic_add(&mut cells[0u32], sum);
            atomic_add(&mut cells[1u32], over);
            atomic_max(&mut cells[2u32], top);
        }
    }
}
// ANCHOR_END: kernel

// ANCHOR: host
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
// ANCHOR_END: host

// ANCHOR: main
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
// ANCHOR_END: main

// ANCHOR: tests
#[cfg(test)]
mod tests {
    use gust::{Independent, cpu::CpuDevice};

    use super::{host_cells, stats, values};

    /// The kernel on the CPU twin. The twin always runs the portable body (a subgroup body has no CPU form), and the
    /// twin runs the invocations one after another, so what it shows is the body's indexing and conditions.
    fn on_twin(data: &[u32], threads: u32) -> Vec<u32> {
        let device = CpuDevice::new();
        let input = device.buffer(data);
        let cells = device.buffer(&[0u32; 3]);
        stats::tally::dispatch(&device, threads, &input, Independent(&cells))
            .expect("the twin dispatches");
        cells.read()
    }

    #[test]
    fn the_twin_matches_hand_computed_values() {
        // 5, 150 and 101 give the sum 256, two values over 100 and the maximum 150.
        assert_eq!(on_twin(&[5, 150, 101], 3), [256, 2, 150]);
    }

    #[test]
    fn the_twin_matches_the_host_reference_exactly() {
        // Sizes below, at and above the workgroup size of 64 and the sizes of a subgroup (8 to 128), ones that are not
        // a multiple of either, and the size of the program.
        for n in [1usize, 7, 63, 64, 65, 127, 129, 1000, 100_000] {
            let data = values(n);
            let threads = u32::try_from(n).expect("small");
            assert_eq!(on_twin(&data, threads), host_cells(&data), "n = {n}");
        }
    }

    #[test]
    fn the_data_exercises_every_cell() {
        // A guard on the example: some values are over the limit and some are not, and the maximum is not the last
        // value.
        let data = values(1000);
        let cells = host_cells(&data);
        let over = usize::try_from(cells[1]).expect("small");
        assert!(over > 0 && over < data.len());
        assert_eq!(cells[2], 250);
    }
}
// ANCHOR_END: tests
