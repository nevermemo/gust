//! The checks of the bins (see [`crate::bins`] for the contract): cases, a host reference, and the per-cell comparison.
//!
//! The host reference is a counting sort written out here, with its own copy of the cell arithmetic, not the `#[gpu_module]`
//! bodies. The order inside a cell is unspecified on the GPU, so the comparison sorts each cell's slice and compares
//! sets; the counts and the starts are exact `u32` values.
//!
//! The hash-spread positions come from the `mix` and `hash` of `examples/hash-rng` (lowbias32, as copied by
//! `examples/mc-histogram`), on the host: `x = (hash(seed, 2 i) >> 8) / 2^24 * side` and `y` likewise with `2 i + 1`.
//! The 24-bit fraction times a power-of-two side is exact in `f32` and below `side`.

use gust_wgpu::{DeviceHandle, GpuVec, HeadlessDevice, default_device};

#[cfg(any(test, gust_dual))]
use crate::bins::TwinBins;
use crate::bins::{Bins, Snapshot};

/// One set of element positions on a `side x side` grid.
pub struct Case {
    pub name: String,
    pub side: u32,
    pub xs: Vec<f32>,
    pub ys: Vec<f32>,
}

/// The step of a position just below the last cell edge: `side - EDGE`, exact in `f32` for a side below 2^14.
const EDGE: f32 = 1.0 / 1024.0;

/// A multiply-xorshift finaliser: "lowbias32" of Chris Wellons' hash-prospector (public domain), as in `hash-rng`.
fn mix(x: u32) -> u32 {
    let mut h = x ^ (x >> 16);
    h = h.wrapping_mul(2146121005);
    h ^= h >> 15;
    h = h.wrapping_mul(2221713035);
    h ^ (h >> 16)
}

fn hash(seed: u32, index: u32) -> u32 {
    mix(mix(seed) ^ index)
}

/// No elements at all.
pub fn empty_case(side: u32) -> Case {
    Case {
        name: format!("empty, side {side}"),
        side,
        xs: Vec::new(),
        ys: Vec::new(),
    }
}

/// `n` elements, all inside the cell `(side / 2, side / 3)`, at 64 different offsets in it.
pub fn one_cell_case(side: u32, n: u32) -> Case {
    let (cx, cy) = ((side / 2) as f32, (side / 3) as f32);
    let offset = |k: u32| (k % 8) as f32 / 8.0;
    Case {
        name: format!("{n} elements in one cell, side {side}"),
        side,
        xs: (0..n).map(|i| cx + offset(i)).collect(),
        ys: (0..n).map(|i| cy + offset(i / 8)).collect(),
    }
}

/// Elements only in the last column (x just below `side`), only in the last row (y just below `side`), and one in
/// the last cell.
pub fn last_edge_case(side: u32) -> Case {
    let last = side as f32 - EDGE;
    let along = |k: u32| k as f32 + 0.5;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for k in 0..side {
        xs.push(last);
        ys.push(along(k));
    }
    for k in 0..side {
        xs.push(along(k));
        ys.push(last);
    }
    xs.push(last);
    ys.push(last);
    Case {
        name: format!("last row and column, side {side}"),
        side,
        xs,
        ys,
    }
}

/// Positions far past the grid, up to `f32::MAX` and infinity, mixed with ordinary ones: each goes to the last cell of
/// its axis.
pub fn far_case(side: u32) -> Case {
    let far = [
        side as f32,
        1.0e10,
        4_294_967_296.0,
        f32::MAX,
        f32::INFINITY,
    ];
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (k, &value) in far.iter().enumerate() {
        let near = (k as u32 % side) as f32 + 0.5;
        xs.push(value);
        ys.push(near);
        xs.push(near);
        ys.push(value);
        xs.push(value);
        ys.push(value);
    }
    Case {
        name: format!("far and infinite positions, side {side}"),
        side,
        xs,
        ys,
    }
}

/// `n` hash-spread elements; `side` must be a power of two.
pub fn spread_case(side: u32, n: u32, seed: u32) -> Case {
    assert!(side.is_power_of_two());
    let unit = |h: u32| (h >> 8) as f32 / 16_777_216.0 * side as f32;
    Case {
        name: format!("{n} hash-spread elements, side {side}, seed {seed}"),
        side,
        xs: (0..n).map(|i| unit(hash(seed, 2 * i))).collect(),
        ys: (0..n).map(|i| unit(hash(seed, 2 * i + 1))).collect(),
    }
}

/// Cases rebuilt one after the other into the same bins: a spread of `n`, one cell, the last row and column, far
/// positions, a smaller spread (another seed), nothing, and a spread of 700.
pub fn sequence(side: u32, n: u32) -> Vec<Case> {
    vec![
        spread_case(side, n, 1),
        one_cell_case(side, 1000),
        last_edge_case(side),
        far_case(side),
        spread_case(side, n / 2 + 11, 2),
        empty_case(side),
        spread_case(side, 700, 3),
    ]
}

/// The host's cell of a position.
fn host_cell(x: f32, y: f32, side: u32) -> usize {
    let cx = (x as u32).min(side - 1);
    let cy = (y as u32).min(side - 1);
    (cy * side + cx) as usize
}

/// A cell's set of elements as a message: up to the first 8 indices.
fn head(values: &[u32]) -> String {
    let shown: Vec<String> = values.iter().take(8).map(u32::to_string).collect();
    let more = if values.len() > 8 { ", .." } else { "" };
    format!("[{}{more}] ({} elements)", shown.join(", "), values.len())
}

/// Checks the bins of `case` against the host's counting sort: counts and starts exactly, the total, and for every cell
/// the set of elements (each cell's slice sorted first).
pub fn check(case: &Case, bins: &Snapshot) -> Result<(), String> {
    let side = case.side;
    let cells = (side * side) as usize;
    let n = case.xs.len();
    let mut sets: Vec<Vec<u32>> = vec![Vec::new(); cells];
    for (i, (&x, &y)) in case.xs.iter().zip(&case.ys).enumerate() {
        sets[host_cell(x, y, side)].push(i as u32);
    }
    if bins.counts.len() != cells || bins.starts.len() != cells {
        return Err(format!(
            "{} counts and {} starts for {cells} cells",
            bins.counts.len(),
            bins.starts.len()
        ));
    }
    let mut start = 0u32;
    for (c, set) in sets.iter().enumerate() {
        if bins.counts[c] != set.len() as u32 {
            return Err(format!(
                "cell {c} (x {}, y {}): count {} against host {}",
                c as u32 % side,
                c as u32 / side,
                bins.counts[c],
                set.len()
            ));
        }
        if bins.starts[c] != start {
            return Err(format!(
                "cell {c}: start {} against host {start}",
                bins.starts[c]
            ));
        }
        start += set.len() as u32;
    }
    let total: u64 = bins.counts.iter().map(|&c| u64::from(c)).sum();
    if total != n as u64 || start as usize != n {
        return Err(format!("the counts total {total} for {n} elements"));
    }
    if bins.order.len() < n {
        return Err(format!(
            "order holds {} slots for {n} elements",
            bins.order.len()
        ));
    }
    for (c, set) in sets.iter().enumerate() {
        let mut got = bins.cell(c).to_vec();
        got.sort_unstable();
        if got != *set {
            return Err(format!(
                "cell {c}: set {} against host set {}",
                head(&got),
                head(set)
            ));
        }
    }
    Ok(())
}

/// The side and the capacity one set of bins needs for `cases`.
fn shape(cases: &[Case]) -> Result<(u32, u32), String> {
    let side = cases.first().ok_or("no cases")?.side;
    if cases.iter().any(|case| case.side != side) {
        return Err("the cases have different sides".into());
    }
    let capacity = cases
        .iter()
        .map(|case| case.xs.len() as u32)
        .max()
        .unwrap_or(0);
    Ok((side, capacity))
}

/// Rebuilds the same bins for each case in turn on the device and checks each against the host.
pub fn run_gpu(handle: &DeviceHandle, cases: &[Case]) -> Result<(), Box<dyn std::error::Error>> {
    let (side, capacity) = shape(cases)?;
    let mut bins = Bins::new(handle, side, capacity)?;
    for case in cases {
        let xs = GpuVec::from_slice_on(handle, &case.xs)?;
        let ys = GpuVec::from_slice_on(handle, &case.ys)?;
        bins.rebuild(&xs, &ys)?;
        check(case, &bins.read()?).map_err(|message| format!("gpu, {}: {message}", case.name))?;
    }
    Ok(())
}

/// The same on the CPU twin.
#[cfg(any(test, gust_dual))]
pub fn run_twin(cases: &[Case]) -> Result<(), Box<dyn std::error::Error>> {
    let (side, capacity) = shape(cases)?;
    let mut bins = TwinBins::new(side, capacity)?;
    for case in cases {
        let xs = bins.device().buffer(&case.xs);
        let ys = bins.device().buffer(&case.ys);
        bins.rebuild(&xs, &ys)?;
        check(case, &bins.read()).map_err(|message| format!("twin, {}: {message}", case.name))?;
    }
    Ok(())
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let handle = default_device()?;
    let device: &HeadlessDevice = &handle;
    let info = device.adapter_info();
    println!("adapter: {} backend: {:?}", info.name, info.backend);
    let cases = sequence(256, (1 << 20) + 37);
    run_gpu(&handle, &cases)?;
    for case in &cases {
        println!(
            "{}: counts, starts and every cell's set equal to the host",
            case.name
        );
    }
    println!(
        "{} rebuilds into the same bins (the bins' own buffers are reused; the scan's temporaries come from the device pool)",
        cases.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIDE: u32 = 64;
    /// More than one workgroup (256) and, with 4096 cells, more than one workgroup of the scan.
    const SPREAD: u32 = 100_037;

    #[test]
    fn twin_empty_grid() {
        run_twin(&[empty_case(SIDE)]).unwrap();
    }

    #[test]
    fn twin_all_elements_in_one_cell() {
        run_twin(&[one_cell_case(SIDE, 1000)]).unwrap();
    }

    #[test]
    fn twin_last_row_and_column() {
        run_twin(&[last_edge_case(SIDE)]).unwrap();
    }

    #[test]
    fn twin_far_and_infinite_positions_go_to_the_last_cell() {
        let case = far_case(SIDE);
        run_twin(std::slice::from_ref(&case)).unwrap();
        // Five elements are far on both axes (the last cell); the others are far on one axis only.
        let bins = run_snapshot(&case);
        assert_eq!(bins.counts[(SIDE * SIDE - 1) as usize], 5);
    }

    #[test]
    fn twin_hash_spread_past_a_workgroup() {
        run_twin(&[spread_case(SIDE, SPREAD, 1)]).unwrap();
    }

    #[test]
    fn twin_rebuilds_into_the_same_bins() {
        run_twin(&sequence(SIDE, SPREAD)).unwrap();
    }

    #[test]
    fn the_twin_rejects_a_bad_grid_or_bad_elements() {
        assert!(TwinBins::new(0, 4).is_err(), "side 0");
        assert!(
            TwinBins::new(65_536, 4).is_err(),
            "side * side overflows u32"
        );
        let mut bins = TwinBins::new(4, 3).unwrap();
        let two = bins.device().buffer(&[0.5f32, 1.5]);
        let three = bins.device().buffer(&[0.5f32, 1.5, 2.5]);
        let four = bins.device().buffer(&[0.5f32, 1.5, 2.5, 3.5]);
        let message =
            |result: Result<(), Box<dyn std::error::Error>>| result.unwrap_err().to_string();
        assert!(message(bins.rebuild(&two, &three)).contains("2 x positions for 3 y"));
        assert!(message(bins.rebuild(&four, &four)).contains("capacity 3"));
        bins.rebuild(&three, &three).unwrap();
    }

    #[test]
    fn the_cases_put_elements_where_they_say() {
        let one = one_cell_case(SIDE, 1000);
        let cell = host_cell(one.xs[0], one.ys[0], SIDE);
        assert!(
            one.xs
                .iter()
                .zip(&one.ys)
                .all(|(&x, &y)| host_cell(x, y, SIDE) == cell)
        );
        let edge = last_edge_case(SIDE);
        assert_eq!(edge.xs.len(), 2 * SIDE as usize + 1);
        assert!(
            edge.xs
                .iter()
                .zip(&edge.ys)
                .all(|(&x, &y)| x >= (SIDE - 1) as f32 || y >= (SIDE - 1) as f32)
        );
        assert!(edge.xs.iter().chain(&edge.ys).all(|&v| v < SIDE as f32));
        let spread = spread_case(SIDE, SPREAD, 1);
        assert!(
            spread
                .xs
                .iter()
                .chain(&spread.ys)
                .all(|&v| (0.0..SIDE as f32).contains(&v))
        );
        let mut used = vec![false; (SIDE * SIDE) as usize];
        for (&x, &y) in spread.xs.iter().zip(&spread.ys) {
            used[host_cell(x, y, SIDE)] = true;
        }
        assert!(used.iter().all(|&u| u), "every cell holds a spread element");
    }

    #[test]
    fn the_check_rejects_a_wrong_count_start_set_or_total() {
        let case = spread_case(8, 300, 1);
        let right = run_snapshot(&case);
        check(&case, &right).unwrap();

        let mut bad = right.clone();
        bad.counts[3] += 1;
        assert!(check(&case, &bad).unwrap_err().contains("cell 3"));

        let mut bad = right.clone();
        bad.starts[5] += 1;
        assert!(check(&case, &bad).unwrap_err().contains("cell 5: start"));

        // One element moved to another cell's slice: counts and starts agree, the set does not.
        let mut bad = right.clone();
        let (from, to) = (bad.starts[1] as usize, bad.starts[2] as usize);
        bad.order.swap(from, to);
        assert!(check(&case, &bad).unwrap_err().contains("set"));

        let mut bad = right.clone();
        bad.order.truncate(10);
        assert!(check(&case, &bad).is_err());
    }

    /// The twin's bins of one case.
    fn run_snapshot(case: &Case) -> Snapshot {
        let mut bins = TwinBins::new(case.side, case.xs.len() as u32).unwrap();
        let xs = bins.device().buffer(&case.xs);
        let ys = bins.device().buffer(&case.ys);
        bins.rebuild(&xs, &ys).unwrap();
        bins.read()
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_matches_the_host_reference() {
        let handle = default_device().unwrap_or_else(|error| {
            panic!("Vulkan adapter required; this test must not silently skip: {error}")
        });
        let info = handle.adapter_info();
        println!("adapter: {} backend: {:?}", info.name, info.backend);
        assert!(Bins::new(&handle, 0, 4).is_err(), "side 0");
        assert!(
            Bins::new(&handle, 65_536, 4).is_err(),
            "side * side overflows u32"
        );
        let mut bins = Bins::new(&handle, 4, 3).unwrap();
        let three = GpuVec::from_slice_on(&handle, &[0.5f32, 1.5, 2.5]).unwrap();
        let four = GpuVec::from_slice_on(&handle, &[0.5f32, 1.5, 2.5, 3.5]).unwrap();
        assert!(bins.rebuild(&three, &four).is_err(), "mismatched lengths");
        assert!(bins.rebuild(&four, &four).is_err(), "over capacity");
        for case in [
            empty_case(SIDE),
            one_cell_case(SIDE, 1000),
            last_edge_case(SIDE),
            far_case(SIDE),
            spread_case(SIDE, SPREAD, 1),
        ] {
            run_gpu(&handle, std::slice::from_ref(&case)).unwrap();
            println!("{}: equal to the host", case.name);
        }
        let cases = sequence(SIDE, SPREAD);
        run_gpu(&handle, &cases).unwrap();
        println!(
            "{} rebuilds into the same bins: each equal to the host",
            cases.len()
        );
    }
}
