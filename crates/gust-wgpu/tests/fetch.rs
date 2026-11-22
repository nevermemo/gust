//! The old value of `atomic_add` on the GPU and on the CPU twin (slice m11.fetch; Gpu tier).
//!
//! `scatter`: `ITEMS` items are hashed to `CELLS` cells on the host (one cell takes every fifth item, so it is heavily
//! contended). Each item does `slot = atomic_add(&mut counts[cell], 1)` and writes its id to `out[cell * CAP + slot]`.
//! Which item gets which slot is schedule-dependent and unspecified (docs/SEMANTICS.md, `m11.fetch`), so nothing here
//! compares a per-item slot: the counts are checked exactly against the host, and for every cell the ids written in
//! slots `0..count` are compared as a multiset with the host's list for that cell, with every other slot untouched.
//! `scatter_wide` is the same scatter in 64-wide workgroups with a bounds check. `ticket_signed` checks the old values of
//! a contended `i32` cell as a set, and `ticket_wrap` those of a `u32` cell that wraps past 2^32. The twin is checked the same way, and
//! its own order (ascending, one invocation at a time) is asserted separately.

include!("../../../tests/fixtures/fetch.rs");

mod common;

use gust::{Independent, cpu::CpuDevice};
use gust_wgpu::{BufferBinding, GpuBufferAccess};

const ITEMS: u32 = 4099;
const CELLS: u32 = 61;
/// The slots per cell of the kernel (`cell * 1024 + slot`).
const CAP: usize = 1024;
const UNWRITTEN: u32 = u32::MAX;

/// The host's cell of an item: a multiplicative hash, except that every fifth item goes to cell 7.
fn cell_of(item: u32) -> u32 {
    if item.is_multiple_of(5) {
        7
    } else {
        (item.wrapping_mul(2_654_435_761) >> 11) % CELLS
    }
}

/// The host's reference: the keys and, per cell, the ids hashed to it in ascending order.
fn host_cells() -> (Vec<u32>, Vec<Vec<u32>>) {
    let keys: Vec<u32> = (0..ITEMS).map(cell_of).collect();
    let mut cells = vec![Vec::new(); CELLS as usize];
    for (item, &key) in keys.iter().enumerate() {
        cells[key as usize].push(item as u32);
    }
    (keys, cells)
}

/// The scatter outcome against the host: counts exact, per cell the multiset of written ids equal to the host's list,
/// slots `0..count` filled, every other slot untouched. Returns the number of cells checked.
fn check_scatter(who: &str, counts: &[u32], out: &[u32], cells: &[Vec<u32>]) -> usize {
    assert_eq!(counts.len(), CELLS as usize, "{who}: counts length");
    assert_eq!(out.len(), CELLS as usize * CAP, "{who}: out length");
    let mut checked = 0;
    for (cell, expected) in cells.iter().enumerate() {
        assert_eq!(
            counts[cell] as usize,
            expected.len(),
            "{who}: count of cell {cell}"
        );
        let row = &out[cell * CAP..(cell + 1) * CAP];
        let (written, rest) = row.split_at(expected.len());
        assert!(
            written.iter().all(|&id| id != UNWRITTEN),
            "{who}: a slot below the count of cell {cell} was never filled"
        );
        assert!(
            rest.iter().all(|&id| id == UNWRITTEN),
            "{who}: a slot at or above the count of cell {cell} was written"
        );
        let mut sorted = written.to_vec();
        sorted.sort_unstable();
        // `expected` is ascending and has no duplicates, so equality also rules out duplicate ids.
        assert_eq!(
            &sorted, expected,
            "{who}: the ids of cell {cell} are not the host's set"
        );
        checked += 1;
    }
    assert_eq!(
        counts.iter().map(|&c| c as usize).sum::<usize>(),
        ITEMS as usize,
        "{who}: the counts must sum to the item count"
    );
    checked
}

/// A scatter test of the fixture kernel `$kernel` (the same checks on the GPU and on the twin).
macro_rules! scatter_test {
    ($name:ident, $kernel:ident) => {
#[test]
fn $name() {
    let (keys, cells) = host_cells();
    let heaviest = cells.iter().map(Vec::len).max().unwrap();
    assert!(heaviest <= CAP, "the host's heaviest cell must fit CAP");
    assert!(
        heaviest > 500,
        "the test needs a contended cell, got {heaviest}"
    );

    let device = common::device();
    let info = device.adapter_info();
    eprintln!("adapter: {} backend: {:?}", info.name, info.backend);

    let key_buffer = device.create_typed_buffer("fetch keys", &keys, GpuBufferAccess::ReadOnly);
    let counts = device.create_typed_buffer(
        "fetch counts",
        &[0u32; CELLS as usize],
        GpuBufferAccess::ReadWrite,
    );
    let out = device.create_typed_buffer(
        "fetch out",
        &vec![UNWRITTEN; CELLS as usize * CAP],
        GpuBufferAccess::ReadWrite,
    );
    device
        .dispatch_buffers(
            &fetch::$kernel::DESCRIPTOR,
            ITEMS,
            &[
                BufferBinding::read_only(&key_buffer).independent_length(),
                BufferBinding::read_write(&counts).independent_length(),
                BufferBinding::read_write(&out).independent_length(),
            ],
        )
        .unwrap();
    let gpu_counts = device.read_typed_buffer(&counts).unwrap();
    let gpu_out = device.read_typed_buffer(&out).unwrap();
    let checked = check_scatter("gpu", &gpu_counts, &gpu_out, &cells);
    eprintln!(
        "fetch scatter: items={ITEMS} cells={CELLS} heaviest_cell={heaviest} cells_checked={checked} \
         counts_exact=true multiset_equal=true slots_filled=true"
    );

    // The twin: the same typed call with the device swapped, checked the same way (never against the GPU's slots).
    let cpu = CpuDevice::new();
    let twin_keys = cpu.buffer(&keys);
    let twin_counts = cpu.buffer(&[0u32; CELLS as usize]);
    let twin_out = cpu.buffer(&vec![UNWRITTEN; CELLS as usize * CAP]);
    fetch::$kernel::dispatch(
        &cpu,
        ITEMS,
        Independent(&twin_keys),
        Independent(&twin_counts),
        Independent(&twin_out),
    )
    .unwrap();
    let twin_counts = twin_counts.read();
    let twin_out = twin_out.read();
    check_scatter("twin", &twin_counts, &twin_out, &cells);
    assert_eq!(twin_counts, gpu_counts, "twin counts against the GPU");
    // The twin's own order: one invocation at a time in ascending order, so a cell's slots hold its ids ascending.
    for (cell, expected) in cells.iter().enumerate() {
        assert_eq!(
            &twin_out[cell * CAP..cell * CAP + expected.len()],
            expected.as_slice(),
            "twin order of cell {cell}"
        );
    }
}
    };
}

scatter_test!(atomic_add_old_values_scatter_items_to_their_cells, scatter);
scatter_test!(
    atomic_add_old_values_scatter_in_64_wide_workgroups,
    scatter_wide
);

#[test]
fn signed_atomic_add_old_values_are_a_set_of_prefix_sums() {
    const N: u32 = 257;
    const START: i32 = -300;
    let expected: Vec<i32> = (0..N as i32).map(|k| START + 3 * k).collect();

    let device = common::device();
    let info = device.adapter_info();
    eprintln!("adapter: {} backend: {:?}", info.name, info.backend);
    let cell = device.create_typed_buffer("fetch cell", &[START], GpuBufferAccess::ReadWrite);
    let out = device.create_typed_buffer(
        "fetch signed out",
        &vec![i32::MIN; N as usize],
        GpuBufferAccess::ReadWrite,
    );
    device
        .dispatch_buffers(
            &fetch::ticket_signed::DESCRIPTOR,
            N,
            &[
                BufferBinding::read_write(&cell).independent_length(),
                BufferBinding::read_write(&out).independent_length(),
            ],
        )
        .unwrap();
    assert_eq!(
        device.read_typed_buffer(&cell).unwrap(),
        vec![START + 3 * N as i32],
        "gpu: the cell total is exact"
    );
    let mut gpu = device.read_typed_buffer(&out).unwrap();
    gpu.sort_unstable();
    assert_eq!(
        gpu, expected,
        "gpu: the old values are the prefix sums, as a set"
    );

    let cpu = CpuDevice::new();
    let twin_cell = cpu.buffer(&[START]);
    let twin_out = cpu.buffer(&vec![i32::MIN; N as usize]);
    fetch::ticket_signed::dispatch(&cpu, N, Independent(&twin_cell), Independent(&twin_out))
        .unwrap();
    assert_eq!(twin_cell.read(), vec![START + 3 * N as i32], "twin total");
    let twin = twin_out.read();
    // The twin's own order is ascending, so invocation `k` gets `START + 3 * k`.
    assert_eq!(twin, expected, "twin: its own (ascending) order");
    let mut sorted = twin;
    sorted.sort_unstable();
    assert_eq!(sorted, expected, "twin: the same set as the GPU");
}

#[test]
fn unsigned_atomic_add_old_values_wrap_modulo_2_pow_32() {
    const N: u32 = 257;
    const START: u32 = u32::MAX - 100;
    // The prefix sums modulo 2^32: 101 values below the wrap, then 0, 1, ..
    let expected: Vec<u32> = (0..N).map(|k| START.wrapping_add(k)).collect();
    let mut expected_sorted = expected.clone();
    expected_sorted.sort_unstable();
    assert!(
        expected.contains(&0) && expected.contains(&u32::MAX),
        "the case must cross zero"
    );

    let device = common::device();
    let info = device.adapter_info();
    eprintln!("adapter: {} backend: {:?}", info.name, info.backend);
    let cell = device.create_typed_buffer("fetch wrap cell", &[START], GpuBufferAccess::ReadWrite);
    let out = device.create_typed_buffer(
        "fetch wrap out",
        &vec![0u32; N as usize],
        GpuBufferAccess::ReadWrite,
    );
    device
        .dispatch_buffers(
            &fetch::ticket_wrap::DESCRIPTOR,
            N,
            &[
                BufferBinding::read_write(&cell).independent_length(),
                BufferBinding::read_write(&out).independent_length(),
            ],
        )
        .unwrap();
    assert_eq!(
        device.read_typed_buffer(&cell).unwrap(),
        vec![START.wrapping_add(N)],
        "gpu: the cell total wraps exactly"
    );
    let mut gpu = device.read_typed_buffer(&out).unwrap();
    gpu.sort_unstable();
    assert_eq!(gpu, expected_sorted, "gpu: the old values, as a set");

    let cpu = CpuDevice::new();
    let twin_cell = cpu.buffer(&[START]);
    let twin_out = cpu.buffer(&vec![0u32; N as usize]);
    fetch::ticket_wrap::dispatch(&cpu, N, Independent(&twin_cell), Independent(&twin_out)).unwrap();
    assert_eq!(twin_cell.read(), vec![START.wrapping_add(N)], "twin total");
    let mut twin = twin_out.read();
    twin.sort_unstable();
    assert_eq!(twin, expected_sorted, "twin: the same set as the GPU");
}
