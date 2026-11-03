//! Multi-dimensional dispatch on a real device (slice m7.d): the kernels of `tests/fixtures/grid.rs` over two and
//! three dimensional extents, run on the GPU and on the CPU twin and compared with an independent host reference; the
//! local and group ids in two dimensions; a zero dimension; and the invocation cap at its boundary on both backends.
//!
//! The cap itself cannot be run on the twin: a launch of exactly 2^32 invocations is a few seconds of GPU time and
//! days of CPU time. The twin's decision about every boundary extent is therefore taken from `gust::cpu::validate`
//! (the check `CpuDevice::dispatch` makes before it runs anything), and the GPU's from a real dispatch.

include!("../../../tests/fixtures/grid.rs");

use gust::{
    Extent, Independent,
    cpu::{self, CpuDevice},
};
use gust_wgpu::{Error, GpuBufferAccess, HeadlessDevice};

/// The workgroup of `cells` and `corner`, per axis.
const TILE: u32 = 16;

fn device() -> HeadlessDevice {
    HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip")
}

/// A value no kernel writes, to show a cell the dispatch never reached.
const UNWRITTEN: u32 = u32::MAX;

/// What `cells` writes, per output, computed from the coordinates alone (not from the kernel's body).
struct Cells {
    coords: Vec<u32>,
    locals: Vec<u32>,
    groups: Vec<u32>,
}

/// The expected outputs of `cells` over a `width` x `height` extent, row-major.
fn expected_cells(width: u32, height: u32) -> Cells {
    let mut cells = Cells {
        coords: Vec::new(),
        locals: Vec::new(),
        groups: Vec::new(),
    };
    for y in 0..height {
        for x in 0..width {
            cells.coords.push(y * 65_536 + x);
            cells.locals.push((y % TILE) * TILE + x % TILE);
            cells.groups.push((y / TILE) * 65_536 + x / TILE);
        }
    }
    cells
}

/// `cells` on the GPU over `width` x `height`.
fn gpu_cells(device: &HeadlessDevice, width: u32, height: u32) -> Cells {
    let count = width as usize * height as usize;
    let size = device.create_typed_buffer("size", &[width, height], GpuBufferAccess::ReadOnly);
    let output = |label| {
        device.create_typed_buffer(label, &vec![UNWRITTEN; count], GpuBufferAccess::ReadWrite)
    };
    let (coords, locals, groups) = (output("coords"), output("locals"), output("groups"));
    grid::cells::dispatch_extent(
        device,
        Extent::new(width, height, 1),
        Independent(&size),
        &coords,
        &locals,
        &groups,
    )
    .unwrap();
    let read = |buffer| device.read_typed_buffer(buffer).unwrap();
    Cells {
        coords: read(&coords),
        locals: read(&locals),
        groups: read(&groups),
    }
}

/// `cells` on the CPU twin over `width` x `height`.
fn twin_cells(width: u32, height: u32) -> Cells {
    let cpu = CpuDevice::new();
    let count = width as usize * height as usize;
    let size = cpu.buffer(&[width, height]);
    let output = || cpu.buffer(&vec![UNWRITTEN; count]);
    let (coords, locals, groups) = (output(), output(), output());
    grid::cells::dispatch_extent(
        &cpu,
        Extent::new(width, height, 1),
        Independent(&size),
        &coords,
        &locals,
        &groups,
    )
    .unwrap();
    Cells {
        coords: coords.read(),
        locals: locals.read(),
        groups: groups.read(),
    }
}

#[track_caller]
fn assert_cells(actual: &Cells, wanted: &Cells, what: &str) {
    assert_eq!(actual.coords, wanted.coords, "coords, {what}");
    assert_eq!(actual.locals, wanted.locals, "locals, {what}");
    assert_eq!(actual.groups, wanted.groups, "groups, {what}");
}

/// Extents that are, in turn: one cell, not a multiple of the workgroup on either axis (and on one axis only), exactly
/// one workgroup, one past it, a thin strip in each direction, and a large square.
const EXTENTS: [(u32, u32); 10] = [
    (1, 1),
    (15, 17),
    (16, 16),
    (17, 16),
    (16, 17),
    (33, 1),
    (1, 33),
    (100, 37),
    (31, 255),
    (256, 256),
];

#[test]
fn cells_match_the_host_reference_on_the_gpu_and_on_the_twin() {
    let device = device();
    eprintln!("grid adapter: {:?}", device.adapter_info());
    for (width, height) in EXTENTS {
        let wanted = expected_cells(width, height);
        let what = format!("extent {width} x {height}");
        assert_cells(
            &gpu_cells(&device, width, height),
            &wanted,
            &format!("GPU, {what}"),
        );
        assert_cells(
            &twin_cells(width, height),
            &wanted,
            &format!("twin, {what}"),
        );
    }
}

#[test]
fn local_and_group_ids_are_exact_in_two_dimensions() {
    let device = device();
    // 20 x 18 is two workgroups wide and two high, with a ragged edge on both axes.
    let cells = gpu_cells(&device, 20, 18);
    let at = |x: u32, y: u32| (y * 20 + x) as usize;
    // The origin cell, the last cell of the first workgroup, the first cell of the next workgroup on each axis, and
    // the last cell of the extent (in the fourth workgroup).
    for (x, y, local, group) in [
        (0, 0, 0, 0),
        (15, 15, 255, 0),
        (16, 0, 0, 1),
        (0, 16, 0, 65_536),
        (17, 16, 1, 65_536 + 1),
        (19, 17, 16 + 3, 65_536 + 1),
    ] {
        assert_eq!(cells.locals[at(x, y)], local, "local id of ({x}, {y})");
        assert_eq!(cells.groups[at(x, y)], group, "group id of ({x}, {y})");
    }
    // And the identity that defines them, for every cell: the dispatch id is `group * 16 + local` on each axis.
    for y in 0..18u32 {
        for x in 0..20u32 {
            let local = cells.locals[at(x, y)];
            let group = cells.groups[at(x, y)];
            let (local_x, local_y) = (local % TILE, local / TILE);
            let (group_x, group_y) = (group % 65_536, group / 65_536);
            assert_eq!((group_x * TILE + local_x, group_y * TILE + local_y), (x, y));
            assert!(local < TILE * TILE && group_x < 2 && group_y < 2);
        }
    }
}

/// `volume`'s packed coordinate of a cell.
fn packed(x: u32, y: u32, z: u32) -> u32 {
    z * 1_048_576 + y * 1024 + x
}

fn expected_volume(width: u32, height: u32, depth: u32) -> Vec<u32> {
    let mut cells = Vec::new();
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                cells.push(packed(x, y, z));
            }
        }
    }
    cells
}

#[test]
fn a_three_dimensional_extent_runs_on_the_gpu_and_on_the_twin() {
    let device = device();
    // One cell, a single workgroup, ragged on every axis, a thin column of layers, and a deeper block.
    for (width, height, depth) in [
        (1, 1, 1),
        (4, 4, 4),
        (5, 3, 7),
        (9, 2, 5),
        (1, 1, 9),
        (13, 9, 11),
    ] {
        let count = (width * height * depth) as usize;
        let wanted = expected_volume(width, height, depth);
        let extent = Extent::new(width, height, depth);
        let size =
            device.create_typed_buffer("size", &[width, height, depth], GpuBufferAccess::ReadOnly);
        let cells = device.create_typed_buffer(
            "cells",
            &vec![UNWRITTEN; count],
            GpuBufferAccess::ReadWrite,
        );
        grid::volume::dispatch_extent(&device, extent, Independent(&size), &cells).unwrap();
        assert_eq!(
            device.read_typed_buffer(&cells).unwrap(),
            wanted,
            "GPU, extent {width} x {height} x {depth}"
        );
        let cpu = CpuDevice::new();
        let size = cpu.buffer(&[width, height, depth]);
        let cells = cpu.buffer(&vec![UNWRITTEN; count]);
        grid::volume::dispatch_extent(&cpu, extent, Independent(&size), &cells).unwrap();
        assert_eq!(
            cells.read(),
            wanted,
            "twin, extent {width} x {height} x {depth}"
        );
    }
}

#[test]
fn a_zero_dimension_is_ok_and_dispatches_nothing_on_both_backends() {
    let device = device();
    let cpu = CpuDevice::new();
    // `corner` would write the last cell it reaches into `last`: a dispatch that ran leaves `last` changed.
    for (width, height) in [(0, 16), (16, 0), (0, 0), (0, u32::MAX), (u32::MAX, 0)] {
        let size = device.create_typed_buffer("size", &[width, height], GpuBufferAccess::ReadOnly);
        let last = device.create_typed_buffer("last", &[7u32, 7], GpuBufferAccess::ReadWrite);
        grid::corner::dispatch_extent(
            &device,
            Extent::new(width, height, 1),
            Independent(&size),
            Independent(&last),
        )
        .unwrap_or_else(|error| panic!("GPU, {width} x {height}: {error:?}"));
        assert_eq!(
            device.read_typed_buffer(&last).unwrap(),
            [7, 7],
            "GPU, {width} x {height}"
        );

        let size = cpu.buffer(&[width, height]);
        let last = cpu.buffer(&[7u32, 7]);
        grid::corner::dispatch_extent(
            &cpu,
            Extent::new(width, height, 1),
            Independent(&size),
            Independent(&last),
        )
        .unwrap_or_else(|error| panic!("twin, {width} x {height}: {error:?}"));
        assert_eq!(last.read(), [7, 7], "twin, {width} x {height}");
    }
    // A zero in the third dimension of a three dimensional kernel, with the others beyond any limit.
    let size = device.create_typed_buffer("size", &[1u32, 1, 1], GpuBufferAccess::ReadOnly);
    let cells = device.create_typed_buffer("cells", &[7u32], GpuBufferAccess::ReadWrite);
    for extent in [
        Extent::new(4, 4, 0),
        Extent::new(u32::MAX, u32::MAX, 0),
        Extent::new(0, 4, 4),
    ] {
        grid::volume::dispatch_extent(&device, extent, Independent(&size), Independent(&cells))
            .unwrap();
        assert_eq!(device.read_typed_buffer(&cells).unwrap(), [7], "{extent:?}");
    }
    // A shared buffer of an empty extent is empty.
    let empty = device.create_typed_buffer("empty", &Vec::<u32>::new(), GpuBufferAccess::ReadWrite);
    grid::volume::dispatch_extent(&device, Extent::new(0, 5, 5), Independent(&size), &empty)
        .unwrap();
    // A zero dimension does not excuse a shared buffer that is not empty: the length rule is the product, which is 0.
    let wrong = device.create_typed_buffer("wrong", &[1u32], GpuBufferAccess::ReadWrite);
    assert!(matches!(
        grid::volume::dispatch_extent(&device, Extent::new(0, 5, 5), Independent(&size), &wrong),
        Err(Error::BufferLength {
            expected: 0,
            actual: 1,
            ..
        })
    ));
}

/// The number of invocations a launch of `extent` over the 16 x 16 workgroup of `corner` runs: every axis rounded up
/// to whole workgroups, then multiplied.
fn rounded_invocations(width: u32, height: u32) -> u64 {
    let round = |axis: u32| u64::from(axis).div_ceil(u64::from(TILE)) * u64::from(TILE);
    round(width) * round(height)
}

/// 2^32, the most invocations one launch may run.
const CAP: u64 = 1 << 32;

/// What `gust::cpu::validate` says about launching `corner` over `extent`, with the error reduced to the three numbers
/// that identify a limit.
fn twin_decision(extent: Extent) -> Result<(), (&'static str, u64, u64)> {
    let cpu = CpuDevice::new();
    let size = cpu.buffer(&[extent.x, extent.y]);
    let last = cpu.buffer(&[0u32, 0]);
    let call = grid::corner::call_extent(extent, Independent(&size), Independent(&last));
    match cpu::validate(&cpu, &call) {
        Ok(()) => Ok(()),
        Err(cpu::Error::DeviceLimit {
            limit,
            requested,
            maximum,
            ..
        }) => Err((limit, requested, maximum)),
        Err(other) => panic!("{extent:?}: expected a DeviceLimit, got {other:?}"),
    }
}

/// What the GPU says about launching `corner` over `width` x `height`: a launch that is accepted runs, and its last
/// cell (`last`) is returned; a rejected one returns the three numbers of its limit and leaves `last` alone.
fn gpu_decision(
    device: &HeadlessDevice,
    width: u32,
    height: u32,
) -> Result<[u32; 2], (&'static str, u64, u64)> {
    let size = device.create_typed_buffer("size", &[width, height], GpuBufferAccess::ReadOnly);
    let last = device.create_typed_buffer("last", &[UNWRITTEN; 2], GpuBufferAccess::ReadWrite);
    let result = grid::corner::dispatch_extent(
        device,
        Extent::new(width, height, 1),
        Independent(&size),
        Independent(&last),
    );
    let last = device.read_typed_buffer(&last).unwrap();
    match result {
        Ok(()) => Ok([last[0], last[1]]),
        Err(Error::DeviceLimit {
            limit,
            requested,
            maximum,
            ..
        }) => {
            assert_eq!(last, [UNWRITTEN; 2], "a rejected launch must not run");
            Err((limit, requested, maximum))
        }
        Err(other) => panic!("{width} x {height}: expected a DeviceLimit, got {other:?}"),
    }
}

#[test]
fn the_invocation_cap_is_the_same_on_the_gpu_and_the_twin_at_its_boundary() {
    let device = device();
    // Exactly the cap, rounded up to exactly the cap, and the two neighbours that round past it.
    //   65536 x 65536: 2^16 * 2^16 = 2^32.
    //   65536 x 65535: the second axis rounds up to 65536, so also 2^32 (a product of 2^32 - 2^16 before rounding).
    //   65537 x 65535: a raw product of 2^32 - 1, which is below the cap, but 65552 * 65536 after rounding.
    //   65537 x 65536 and 65536 x 65537: one element past the cap on one axis.
    let accepted = [(65_536u32, 65_536u32), (65_536, 65_535)];
    let rejected = [(65_537u32, 65_535u32), (65_537, 65_536), (65_536, 65_537)];
    for (width, height) in accepted {
        assert_eq!(
            rounded_invocations(width, height),
            CAP,
            "{width} x {height}"
        );
        assert_eq!(
            twin_decision(Extent::new(width, height, 1)),
            Ok(()),
            "twin, {width} x {height}"
        );
        // The GPU runs it, and the last cell of the extent is reached.
        assert_eq!(
            gpu_decision(&device, width, height),
            Ok([width - 1, height - 1]),
            "GPU, {width} x {height}"
        );
    }
    let raw = u64::from(65_537u32) * u64::from(65_535u32);
    assert!(
        raw < CAP,
        "the raw product of 65537 x 65535 is below the cap"
    );
    for (width, height) in rejected {
        let requested = rounded_invocations(width, height);
        assert!(requested > CAP, "{width} x {height}");
        let wanted = ("global_invocation_id", requested, CAP);
        assert_eq!(
            twin_decision(Extent::new(width, height, 1)),
            Err(wanted),
            "twin, {width} x {height}"
        );
        assert_eq!(
            gpu_decision(&device, width, height),
            Err(wanted),
            "GPU, {width} x {height}"
        );
    }
}

#[test]
fn the_invocation_cap_counts_all_three_axes() {
    let device = device();
    // `volume` has a 4 x 4 x 4 workgroup: 2048 * 2048 * 1024 is exactly 2^32 and each axis is a whole number of
    // workgroups (512, 512, 256 of them), so it is accepted; one more layer rounds the third axis up to 1028.
    let run = |extent: Extent| {
        let size = device.create_typed_buffer(
            "size",
            &[extent.x, extent.y, extent.z],
            GpuBufferAccess::ReadOnly,
        );
        // The kernel's write is guarded by the length of `packed`, so one cell is enough for 2^32 invocations.
        let packed_cell =
            device.create_typed_buffer("packed", &[UNWRITTEN], GpuBufferAccess::ReadWrite);
        let result = grid::volume::dispatch_extent(
            &device,
            extent,
            Independent(&size),
            Independent(&packed_cell),
        );
        (result, device.read_typed_buffer(&packed_cell).unwrap())
    };
    let twin = |extent: Extent| {
        let cpu = CpuDevice::new();
        let size = cpu.buffer(&[extent.x, extent.y, extent.z]);
        let packed_cell = cpu.buffer(&[UNWRITTEN]);
        let call = grid::volume::call_extent(extent, Independent(&size), Independent(&packed_cell));
        cpu::validate(&cpu, &call)
    };

    let exact = Extent::new(2048, 2048, 1024);
    assert_eq!(twin(exact), Ok(()));
    let (result, cell) = run(exact);
    result.unwrap();
    assert_eq!(
        cell,
        [0],
        "the first cell is written by the first invocation"
    );

    let over = Extent::new(2048, 2048, 1025);
    let requested = 2048u64 * 2048 * 1028;
    assert_eq!(
        twin(over),
        Err(cpu::Error::DeviceLimit {
            kernel: "grid::volume".to_owned(),
            limit: "global_invocation_id",
            requested,
            maximum: CAP,
        })
    );
    let (result, cell) = run(over);
    match result {
        Err(Error::DeviceLimit {
            limit,
            requested: actual,
            maximum,
            ..
        }) => assert_eq!(
            (limit, actual, maximum),
            ("global_invocation_id", requested, CAP)
        ),
        other => panic!("expected the cap, got {other:?}"),
    }
    assert_eq!(cell, [UNWRITTEN], "a rejected launch must not run");
}

#[test]
fn the_per_dimension_workgroup_limit_is_the_gpus_alone_and_the_cap_is_both() {
    let device = device();
    // 2^28 x 1 is exactly 2^32 invocations once the second axis is rounded up to 16, so the twin accepts it. The GPU
    // would need 2^24 workgroups along x, and a device allows 65535 along an axis.
    let extent = Extent::new(1 << 28, 1, 1);
    assert_eq!(twin_decision(extent), Ok(()));
    let maximum = u64::from(device.limits().max_compute_workgroups_per_dimension);
    assert_eq!(
        gpu_decision(&device, 1 << 28, 1),
        Err(("max_compute_workgroups_per_dimension", 1 << 24, maximum))
    );
}

#[test]
fn a_product_beyond_a_u64_is_rejected_on_both_backends_before_anything_runs() {
    let device = device();
    let everything = u32::MAX;
    // The twin has no extent error: the launch is over its cap (2^64 once rounded, saturated to a u64). The GPU
    // reports the overflow itself, because it cannot describe the launch (`Error::Extent`). Both refuse it before a buffer is touched.
    assert_eq!(
        twin_decision(Extent::new(everything, everything, 1)),
        Err(("global_invocation_id", u64::MAX, CAP))
    );
    let size = device.create_typed_buffer("size", &[1u32, 1, 1], GpuBufferAccess::ReadOnly);
    let packed_cell =
        device.create_typed_buffer("packed", &[UNWRITTEN], GpuBufferAccess::ReadWrite);
    let extent = Extent::new(everything, everything, everything);
    assert_eq!(extent.checked_total(), None);
    match grid::volume::dispatch_extent(
        &device,
        extent,
        Independent(&size),
        Independent(&packed_cell),
    ) {
        Err(Error::Extent { extent, .. }) => assert_eq!(extent, [u64::from(everything); 3]),
        other => panic!("expected Error::Extent, got {other:?}"),
    }
    assert_eq!(device.read_typed_buffer(&packed_cell).unwrap(), [UNWRITTEN]);
    // The twin refuses the same extent for the cap, saturated to a u64.
    let cpu = CpuDevice::new();
    let size = cpu.buffer(&[1u32, 1, 1]);
    let packed_cell = cpu.buffer(&[UNWRITTEN]);
    let call = grid::volume::call_extent(extent, Independent(&size), Independent(&packed_cell));
    match cpu::validate(&cpu, &call) {
        Err(cpu::Error::DeviceLimit {
            limit, requested, ..
        }) => assert_eq!((limit, requested), ("global_invocation_id", u64::MAX)),
        other => panic!("expected the cap, got {other:?}"),
    }
}
