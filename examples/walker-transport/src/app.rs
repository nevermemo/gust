//! A composite step of bins, walkers with a crowding rule and a field spread, exact on host, twin and GPU (m11.transport).
//!
//! Many walkers move over a `u32` field on a periodic `SIDE x SIDE` grid. Everything is integer, so the host
//! reference, the CPU twin and the GPU must agree bit for bit after every step: the field, each walker's cell and
//! heading, and the per-cell counts of the bins. The walkers grouped by cell agree per cell as sets, compared after
//! sorting each cell's slice (`Snapshot::grouped`): the raw order inside a cell is unspecified on the GPU. One step is
//! four stages in this order (the bins stage itself dispatches clear, count, scan and scatter):
//!
//! 1. **bins** (`crate::bins`, copied from `examples/cell-bins` and adapted to integer cells): the per-cell counts of
//!    the walkers' current cells, and the walkers grouped by cell.
//! 2. `walk`, one invocation per walker, as in `examples/walker-steer`. A walker is a cell index and a heading 0..7 (E,
//!    NE, N, NW, W, SW, S, SE, counterclockwise). It
//!    - senses the field `dist` cells ahead in the headings `heading - 1`, `heading` and `heading + 1` (mod 8) and
//!      turns toward the largest sample (straight wins a tie with either side, `heading - 1` wins a tie with
//!      `heading + 1`);
//!    - draws `r = hash(hash(seed, step), walker)`. With probability `prob / 2^16` (the low 16 bits of `r` below
//!      `prob`) the choice is replaced by `(r >> 16) % 3`, as turn 0, 1 or 2 for `heading - 1`, `heading` or
//!      `heading + 1` (the modulo is slightly biased: turn 0 has probability 21846/65536, the others 21845/65536);
//!    - **crowding rule:** if its current cell holds more than `crowd` walkers (the count of step 1), the choice is
//!      replaced by that same turn `(r >> 16) % 3` regardless of `prob`. It is the *same draw* `r` as the random
//!      turn, not a second one, so a walker that is both crowded and drawn for a random turn takes one turn;
//!    - moves one cell in its new heading, wrapping, and adds `amount` to the deposit buffer at the new cell with
//!      `atomic_add`.
//! 3. `spread`, one invocation per cell: with `g = field + deposit` per cell, the next field of a cell is
//!    `g - 4 * (g >> 3) + sum of (g >> 3) of its four neighbours` (the exact `u32` channel of `examples/field-relax`,
//!    applied to the field with the deposit added). It writes a second field buffer, so no cell reads a buffer it
//!    writes.
//! 4. `clear_deposit`, one invocation per cell: the deposit to zero. It is its own dispatch because `spread` reads the
//!    deposit of the neighbours, so no invocation of `spread` may clear it.
//!
//! The host then swaps the two field buffers. The walkers read only the field of the previous step and the bins
//! rebuilt in phase 1 of the same step (from the positions after the previous step), and the deposits are integer sums, which commute, so the order of the walkers never matters.
//!
//! One `Uniform<Params>` is created once and rewritten before every step.
//!
//! The hash is the "lowbias32" finaliser of `examples/hash-rng` (`hash(seed, index) = mix(mix(seed) ^ index)`), written
//! again here in the module and on the host.
//!
//! # No `u32` overflow
//!
//! Let `A` be the amount, `W` the number of walkers and `T0` the sum of the starting field. The spread conserves the
//! sum: a cell keeps `g - 4 * (g >> 3)` and each of its four neighbours receives `g >> 3` from it, and every cell is
//! a neighbour of four cells, so the total of the next field is the total of `g`. Every walker adds
//! `A` to the deposit, so the total after `s` steps is exactly `T0 + s * A * W`. Every cell is at most the total, and
//! `g` (field plus deposit) at step `s` is at most the total after it, so no value in any kernel exceeds
//! `T0 + STEPS * A * W`: the bound `field_bound`. The documented range is `T0 + STEPS * A * W < 2^32`; the tests
//! assert it for their inputs, check that the reference's total equals `T0 + s * A * W` after every step and that no
//! cell exceeds the bound. `g - 4 * (g >> 3)` never underflows (`4 * (g >> 3) <= g`), and a sum of nonnegative parts
//! of one cell is at most the total.
//!
//! # Why two modules
//!
//! `SIDE` is a module constant, so one `#[gpu]` module has one grid size. The macro below writes the module and the
//! host code that drives it once per size: `big` (128, the run of `main` and the GPU tests) and, under `cfg(test)`,
//! `small` (16).

#[cfg(any(test, gust_dual))]
use gust::cpu::CpuBuffer;
use gust::{Independent, gpu};
use gust_wgpu::{DeviceHandle, GpuVec, HeadlessDevice, default_device};

use crate::bins::Bins;
#[cfg(any(test, gust_dual))]
use crate::bins::TwinBins;

/// Steps of every run.
pub const STEPS: usize = 24;
/// Seeds of the starting field, the walkers' cells and their headings.
const SEEDS: [u32; 3] = [0x1234_5678, 0x9abc_def0, 0x0bad_cafe];
/// The eight headings as `(dx, dy)`, counterclockwise from east.
const DIRS: [(i64, i64); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

// ---- Host side shared by every size: inputs and the reference, plain Rust, not the `#[gpu]` bodies. ----

/// The hash-rng finaliser ("lowbias32", Chris Wellons' hash-prospector, public domain).
fn mix(x: u32) -> u32 {
    let mut h = x ^ (x >> 16);
    h = h.wrapping_mul(2146121005);
    h ^= h >> 15;
    h = h.wrapping_mul(2221713035);
    h ^ (h >> 16)
}

/// A hash of a seed and an index, as in `examples/hash-rng`.
fn hash(seed: u32, index: u32) -> u32 {
    mix(mix(seed) ^ index)
}

/// Where the walkers start.
#[derive(Clone, Copy, Debug)]
pub enum Start {
    /// Hash-chosen cells and headings.
    Spread,
    /// One cell, hash-chosen headings.
    OneCell,
    /// One cell and one heading: every walker makes the same choices unless a random turn splits them, so one
    /// deposit cell takes an `atomic_add` from every walker.
    OneCellOneHeading,
}

/// One run: the walkers and the parameters.
#[derive(Clone, Debug)]
pub struct Case {
    pub name: &'static str,
    pub walkers: usize,
    pub start: Start,
    /// How many cells ahead the walkers sense: `1 <= dist <= SIDE` (checked by `make_start`).
    pub dist: u32,
    /// The probability of a random turn, over `2^16`.
    pub prob: u32,
    /// What a walker adds to the deposit buffer.
    pub amount: u32,
    /// A walker in a cell holding more than `crowd` walkers takes the hash-chosen turn: `u32::MAX` never fires.
    pub crowd: u32,
    pub seed: u32,
}

/// The state after a step: the field, per walker its cell and heading, and the bins of those cells: the walker count
/// of each cell and the walker indices grouped by cell (ascending inside a cell).
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub field: Vec<u32>,
    pub pos: Vec<u32>,
    pub head: Vec<u32>,
    pub counts: Vec<u32>,
    pub order: Vec<u32>,
}

/// `T0 + STEPS * A * W`, the bound on every value of a run (module documentation, "No `u32` overflow").
#[cfg(test)]
pub fn field_bound(case: &Case, start: &State) -> u64 {
    start.field.iter().map(|&f| u64::from(f)).sum::<u64>()
        + STEPS as u64 * u64::from(case.amount) * case.walkers as u64
}

/// The bins of the cells `pos`, on the host: the count of each of `cells` cells and the walker indices grouped by cell,
/// ascending inside a cell.
fn host_bins(cells: usize, pos: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); cells];
    for (walker, &cell) in pos.iter().enumerate() {
        members[cell as usize].push(walker as u32);
    }
    let counts = members.iter().map(|m| m.len() as u32).collect();
    let order = members.into_iter().flatten().collect();
    (counts, order)
}

/// The starting state of a `side x side` grid: a field below 32 (small, so equal samples are common and the tie rule
/// matters) and the walkers.
///
/// # Panics
///
/// When `dist` is outside `1 <= dist <= side`: every run starts here, on the host and on the device.
pub fn make_start(side: usize, case: &Case) -> State {
    assert!(
        (1..=side).contains(&(case.dist as usize)),
        "{}: dist {} is outside 1..={side}",
        case.name,
        case.dist
    );
    let cells = side * side;
    let field = (0..cells).map(|i| hash(SEEDS[0], i as u32) >> 27).collect();
    let pos: Vec<u32> = (0..case.walkers)
        .map(|w| match case.start {
            Start::Spread => hash(SEEDS[1], w as u32) % cells as u32,
            Start::OneCell | Start::OneCellOneHeading => (cells / 2 + side / 2) as u32,
        })
        .collect();
    let head = (0..case.walkers)
        .map(|w| match case.start {
            Start::Spread | Start::OneCell => hash(SEEDS[2], w as u32) % 8,
            Start::OneCellOneHeading => 3,
        })
        .collect();
    let (counts, order) = host_bins(cells, &pos);
    State {
        field,
        pos,
        head,
        counts,
        order,
    }
}

/// One step on the host, written from the description with a table of headings and `rem_euclid` for the wrap.
pub fn reference_step(side: usize, case: &Case, step: usize, now: &State) -> State {
    let n = side as i64;
    let at = |cell: u32, heading: u32, k: i64| -> u32 {
        let (dx, dy) = DIRS[heading as usize];
        let x = i64::from(cell) % n + dx * k;
        let y = i64::from(cell) / n + dy * k;
        (y.rem_euclid(n) * n + x.rem_euclid(n)) as u32
    };
    let mut deposit = vec![0u32; side * side];
    let mut pos = Vec::with_capacity(case.walkers);
    let mut head = Vec::with_capacity(case.walkers);
    for w in 0..case.walkers {
        let (cell, heading) = (now.pos[w], now.head[w]);
        let headings = [(heading + 7) % 8, heading, (heading + 1) % 8];
        let samples = headings.map(|h| now.field[at(cell, h, i64::from(case.dist)) as usize]);
        // Straight first; a side wins only when strictly larger than the best so far (`heading - 1` before
        // `heading + 1`).
        let mut best = 1;
        for side_index in [0, 2] {
            if samples[side_index] > samples[best] {
                best = side_index;
            }
        }
        let r = hash(hash(case.seed, step as u32), w as u32);
        let crowded = now.counts[cell as usize] > case.crowd;
        if crowded || (r & 0xffff) < case.prob {
            best = ((r >> 16) % 3) as usize;
        }
        let new_heading = headings[best];
        let new_cell = at(cell, new_heading, 1);
        deposit[new_cell as usize] += case.amount;
        pos.push(new_cell);
        head.push(new_heading);
    }
    // The field with the deposit added, then each cell keeps `g - 4 * (g >> 3)` and takes `g >> 3` from each of its
    // four wrapped neighbours.
    let grown: Vec<u32> = now
        .field
        .iter()
        .zip(&deposit)
        .map(|(&f, &d)| f + d)
        .collect();
    let g = |x: usize, y: usize| grown[y * side + x];
    let mut field = Vec::with_capacity(side * side);
    for y in 0..side {
        for x in 0..side {
            let c = g(x, y);
            field.push(
                c - 4 * (c >> 3)
                    + (g(x, (y + side - 1) % side) >> 3)
                    + (g(x, (y + 1) % side) >> 3)
                    + (g((x + 1) % side, y) >> 3)
                    + (g((x + side - 1) % side, y) >> 3),
            );
        }
    }
    let (counts, order) = host_bins(side * side, &pos);
    State {
        field,
        pos,
        head,
        counts,
        order,
    }
}

/// The reference trace: index 0 is the start, index `s` the state after `s` steps.
pub fn reference_trace(side: usize, case: &Case) -> Vec<State> {
    let mut trace = vec![make_start(side, case)];
    for step in 1..=STEPS {
        let next = reference_step(side, case, step, &trace[step - 1]);
        trace.push(next);
    }
    trace
}

/// Compares a device trace with the reference after every step: the first differing walker, then the first
/// differing field cell, bin count and bin slot, as a message.
pub fn verify(name: &str, side: usize, want: &[State], got: &[State]) -> Result<(), String> {
    if want.len() != got.len() {
        return Err(format!("{name}: {} states for {}", got.len(), want.len()));
    }
    for (step, (want, got)) in want.iter().zip(got).enumerate() {
        if want.pos.len() != got.pos.len()
            || want.head.len() != got.head.len()
            || want.field.len() != got.field.len()
            || want.counts.len() != got.counts.len()
            || want.order.len() != got.order.len()
        {
            return Err(format!(
                "{name}: step {step}: a buffer has the wrong length"
            ));
        }
        if let Some(w) =
            (0..want.pos.len()).find(|&w| want.pos[w] != got.pos[w] || want.head[w] != got.head[w])
        {
            return Err(format!(
                "{name}: step {step}: walker {w}: got cell {} heading {}, host cell {} heading {}",
                got.pos[w], got.head[w], want.pos[w], want.head[w]
            ));
        }
        if let Some(i) = (0..want.field.len()).find(|&i| want.field[i] != got.field[i]) {
            return Err(format!(
                "{name}: step {step}: field cell {i} (x {}, y {}): got {}, host {}",
                i % side,
                i / side,
                got.field[i],
                want.field[i]
            ));
        }
        if let Some(i) = (0..want.counts.len()).find(|&i| want.counts[i] != got.counts[i]) {
            return Err(format!(
                "{name}: step {step}: bin count of cell {i} (x {}, y {}): got {}, host {}",
                i % side,
                i / side,
                got.counts[i],
                want.counts[i]
            ));
        }
        if let Some(i) = (0..want.order.len()).find(|&i| want.order[i] != got.order[i]) {
            return Err(format!(
                "{name}: step {step}: bin slot {i}: got walker {}, host walker {}",
                got.order[i], want.order[i]
            ));
        }
    }
    Ok(())
}

/// The cases of the small grid: fewer walkers than a workgroup, then more, a crowd the walkers never reach.
#[cfg(test)]
pub fn small_cases() -> Vec<Case> {
    vec![
        case("spread, crowd 1, p = 0", 50, Start::Spread, 3, 0, 1),
        case(
            "spread, crowd 2, p = 1/4",
            300,
            Start::Spread,
            5,
            1 << 14,
            2,
        ),
        case("one cell, crowd 5, p = 0", 300, Start::OneCell, 2, 0, 5),
        case(
            "one cell and heading, crowd 9, p = 1/16",
            300,
            Start::OneCellOneHeading,
            3,
            1 << 12,
            9,
        ),
        case(
            "spread, crowd 1000, p = 1/4",
            300,
            Start::Spread,
            7,
            1 << 14,
            1000,
        ),
    ]
}

/// The cases of the large grid: many workgroups, all walkers in one cell, every walker crowded (`crowd = 0`), and a
/// crowd the walkers never reach.
pub fn big_cases() -> Vec<Case> {
    vec![
        case("spread, crowd 2, p = 0", 3000, Start::Spread, 4, 0, 2),
        case(
            "spread, crowd 2, p = 1/16",
            3000,
            Start::Spread,
            6,
            1 << 12,
            2,
        ),
        case("one cell, crowd 4, p = 0", 3000, Start::OneCell, 3, 0, 4),
        case(
            "one cell and heading, crowd 3, p = 0",
            3000,
            Start::OneCellOneHeading,
            4,
            0,
            3,
        ),
        case("spread, crowd 0, p = 0", 3000, Start::Spread, 4, 0, 0),
        case(
            "spread, crowd 100000, p = 1/16",
            3000,
            Start::Spread,
            6,
            1 << 12,
            100_000,
        ),
    ]
}

fn case(
    name: &'static str,
    walkers: usize,
    start: Start,
    dist: u32,
    prob: u32,
    crowd: u32,
) -> Case {
    Case {
        name,
        walkers,
        start,
        dist,
        prob,
        amount: 7,
        crowd,
        seed: 0xdead_beef,
    }
}

/// The same case with the crowding rule removed.
#[cfg(test)]
fn rule_off(case: &Case) -> Case {
    Case {
        crowd: u32::MAX,
        ..case.clone()
    }
}

/// The state of a device run read back: the field, the walkers and the bins of their cells.
fn read_state(
    bins: &Bins,
    field: &GpuVec<u32>,
    pos: &GpuVec<u32>,
    head: &GpuVec<u32>,
) -> Result<State, Box<dyn std::error::Error>> {
    let snapshot = bins.read()?;
    Ok(State {
        field: field.to_vec()?,
        pos: pos.to_vec()?,
        head: head.to_vec()?,
        order: snapshot.grouped()?,
        counts: snapshot.counts,
    })
}

/// The same on the CPU twin.
#[cfg(any(test, gust_dual))]
fn read_twin_state(
    bins: &TwinBins,
    field: &CpuBuffer<u32>,
    pos: &CpuBuffer<u32>,
    head: &CpuBuffer<u32>,
) -> Result<State, String> {
    let snapshot = bins.read();
    Ok(State {
        field: field.read(),
        pos: pos.read(),
        head: head.read(),
        order: snapshot.grouped()?,
        counts: snapshot.counts,
    })
}

/// Writes, per size, the `#[gpu]` module and the host code that drives it.
macro_rules! transport_program {
    ($name:ident, $side:tt) => {
        pub mod $name {
            use super::*;

            #[gpu]
            pub mod k {
                /// Cells per side: a power of two.
                pub const SIDE: u32 = $side;
                /// Mask of a coordinate: `SIDE - 1`.
                pub const MASK: u32 = SIDE - 1;
                /// Cells of the grid.
                pub const CELLS: u32 = SIDE * SIDE;

                /// The parameters of a step, eight `u32` (32 bytes, a multiple of 16, so no gap): the seed and
                /// the step of the hash, how far ahead to sense, the random-turn probability over `2^16`, the
                /// deposit amount, the crowd above which a walker always turns, and two pad fields.
                pub struct Params {
                    pub seed: u32,
                    pub step: u32,
                    pub dist: u32,
                    pub prob: u32,
                    pub amount: u32,
                    pub crowd: u32,
                    pub pad0: u32,
                    pub pad1: u32,
                }

                /// A multiply-xorshift finaliser. Every shift amount is a literal below 32 (D39).
                ///
                /// The constants and shifts are those of "lowbias32" from Chris Wellons' hash-prospector
                /// (https://github.com/skeeto/hash-prospector; public domain, Unlicense), as in
                /// `examples/hash-rng`.
                pub fn mix(x: u32) -> u32 {
                    let mut h = x ^ (x >> 16u32);
                    h = h.wrapping_mul(2146121005u32);
                    h = h ^ (h >> 15u32);
                    h = h.wrapping_mul(2221713035u32);
                    h ^ (h >> 16u32)
                }

                /// A hash of a seed and an index, as in `examples/hash-rng`.
                pub fn hash(seed: u32, index: u32) -> u32 {
                    mix(mix(seed) ^ index)
                }

                /// The x step of heading `d`: 1 for the headings with an east part, -1 for a west part.
                pub fn dir_x(d: u32) -> i32 {
                    let t: [i32; 8] = [1i32, 1i32, 0i32, -1i32, -1i32, -1i32, 0i32, 1i32];
                    t[d as usize]
                }

                /// The y step of heading `d`: 1 for the headings with a north part, -1 for a south part.
                pub fn dir_y(d: u32) -> i32 {
                    let t: [i32; 8] = [0i32, 1i32, 1i32, 1i32, 0i32, -1i32, -1i32, -1i32];
                    t[d as usize]
                }

                /// The cell `k` steps from `(x, y)` in heading `d`, wrapping.
                pub fn ahead(x: u32, y: u32, d: u32, k: u32) -> u32 {
                    let n = SIDE as i32;
                    let nx = ((x as i32) + dir_x(d) * (k as i32)).rem_euclid(n);
                    let ny = ((y as i32) + dir_y(d) * (k as i32)).rem_euclid(n);
                    (ny as u32) * SIDE + (nx as u32)
                }

                /// One walker: sense, turn (the crowding rule), move, deposit. Reads `field` and `counts` (never
                /// written here), writes its own `pos` and `head`, and adds into `deposit`.
                #[kernel(workgroup_size(256, 1, 1))]
                pub fn walk(
                    id: SV_DispatchThreadID,
                    p: Uniform<Params>,
                    field: StructuredBuffer<uint>,
                    counts: StructuredBuffer<uint>,
                    mut pos: RWStructuredBuffer<uint>,
                    mut head: RWStructuredBuffer<uint>,
                    mut deposit: RWStructuredBuffer<uint>,
                ) {
                    let i = id.x;
                    if i < pos.len() {
                        let c = pos[i];
                        let h = head[i];
                        let x = c & MASK;
                        let y = c / SIDE;
                        let minus = (h + 7u32) & 7u32;
                        let plus = (h + 1u32) & 7u32;
                        let vm = field[ahead(x, y, minus, p.dist)];
                        let vs = field[ahead(x, y, h, p.dist)];
                        let vp = field[ahead(x, y, plus, p.dist)];
                        // 0 is heading - 1 (clockwise), 1 straight, 2 is heading + 1. Straight wins ties;
                        // heading - 1 wins a tie with heading + 1.
                        let mut turn = 1u32;
                        let mut best = vs;
                        if vm > best {
                            turn = 0u32;
                            best = vm;
                        }
                        if vp > best {
                            turn = 2u32;
                        }
                        // One draw serves the random turn and the crowding rule.
                        let r = hash(hash(p.seed, p.step), i);
                        if counts[c] > p.crowd {
                            turn = (r >> 16u32) % 3u32;
                        }
                        if (r & 65535u32) < p.prob {
                            turn = (r >> 16u32) % 3u32;
                        }
                        let nh = (h + 7u32 + turn) & 7u32;
                        let nc = ahead(x, y, nh, 1u32);
                        atomic_add(&mut deposit[nc], p.amount);
                        pos[i] = nc;
                        head[i] = nh;
                    }
                }

                /// One cell: with `g = field + deposit`, `dst = g - 4 * (g >> 3)` plus `g >> 3` of each of the four
                /// wrapped neighbours. Reads `field` and `deposit`, writes only `dst`.
                #[kernel(workgroup_size(256, 1, 1))]
                pub fn spread(
                    id: SV_DispatchThreadID,
                    field: StructuredBuffer<uint>,
                    deposit: StructuredBuffer<uint>,
                    mut dst: RWStructuredBuffer<uint>,
                ) {
                    let i = id.x;
                    if i < CELLS {
                        let x = i & MASK;
                        let y = i / SIDE;
                        let row = y * SIDE;
                        let e = row + ((x + 1u32) & MASK);
                        let w = row + ((x + MASK) & MASK);
                        let n = ((y + MASK) & MASK) * SIDE + x;
                        let s = ((y + 1u32) & MASK) * SIDE + x;
                        let c = field[i] + deposit[i];
                        dst[i] = c - 4u32 * (c >> 3u32)
                            + ((field[n] + deposit[n]) >> 3u32)
                            + ((field[s] + deposit[s]) >> 3u32)
                            + ((field[e] + deposit[e]) >> 3u32)
                            + ((field[w] + deposit[w]) >> 3u32);
                    }
                }

                /// One cell: the deposit to zero.
                #[kernel(workgroup_size(256, 1, 1))]
                pub fn clear_deposit(
                    id: SV_DispatchThreadID,
                    mut deposit: RWStructuredBuffer<uint>,
                ) {
                    let i = id.x;
                    if i < deposit.len() {
                        deposit[i] = 0u32;
                    }
                }
            }

            pub const SIDE: usize = k::SIDE as usize;

            /// The parameter struct of step `step`.
            /// `step` is 1-based.
            pub fn params(case: &Case, step: usize) -> k::Params {
                k::Params {
                    seed: case.seed,
                    step: step as u32,
                    dist: case.dist,
                    prob: case.prob,
                    amount: case.amount,
                    crowd: case.crowd,
                    pad0: 0,
                    pad1: 0,
                }
            }

            /// `STEPS` steps on the device, reading everything back before every step and after the last.
            pub fn run_gpu(
                handle: &DeviceHandle,
                case: &Case,
            ) -> Result<Vec<State>, Box<dyn std::error::Error>> {
                let device: &HeadlessDevice = handle;
                let start = make_start(SIDE, case);
                let mut field = GpuVec::from_slice_on(handle, &start.field)?;
                let mut spread = GpuVec::<u32>::zeroed_on(handle, k::CELLS)?;
                let deposit = GpuVec::<u32>::zeroed_on(handle, k::CELLS)?;
                let pos = GpuVec::from_slice_on(handle, &start.pos)?;
                let head = GpuVec::from_slice_on(handle, &start.head)?;
                let mut bins = Bins::new(handle, k::CELLS, case.walkers as u32)?;
                let uniform = device.create_uniform("walker-transport params", &params(case, 1))?;
                let mut trace = Vec::with_capacity(STEPS + 1);
                // `field`, the bin counts and `deposit` are `Independent` in `walk`: their length is the cell
                // count, not the number of walkers it is dispatched for.
                for step in 1..=STEPS {
                    bins.rebuild(&pos)?;
                    trace.push(read_state(&bins, &field, &pos, &head)?);
                    device.write_uniform(&uniform, &params(case, step))?;
                    k::walk::dispatch(
                        device,
                        case.walkers as u32,
                        &uniform,
                        Independent(&field),
                        Independent(bins.counts()),
                        &pos,
                        &head,
                        Independent(&deposit),
                    )?;
                    k::spread::dispatch(device, k::CELLS, &field, &deposit, &spread)?;
                    k::clear_deposit::dispatch(device, k::CELLS, &deposit)?;
                    std::mem::swap(&mut field, &mut spread);
                }
                bins.rebuild(&pos)?;
                trace.push(read_state(&bins, &field, &pos, &head)?);
                Ok(trace)
            }

            /// The same steps on the CPU twin (the kernels' own Rust bodies).
            #[cfg(any(test, gust_dual))]
            pub fn run_twin(case: &Case) -> Result<Vec<State>, Box<dyn std::error::Error>> {
                let mut bins = TwinBins::new(k::CELLS, case.walkers as u32)?;
                let start = make_start(SIDE, case);
                let mut field = bins.device().buffer(&start.field);
                let mut spread = bins.device().buffer(&vec![0u32; SIDE * SIDE]);
                let deposit = bins.device().buffer(&vec![0u32; SIDE * SIDE]);
                let pos = bins.device().buffer(&start.pos);
                let head = bins.device().buffer(&start.head);
                let uniform = bins.device().uniform(params(case, 1))?;
                let mut trace = Vec::with_capacity(STEPS + 1);
                for step in 1..=STEPS {
                    bins.rebuild(&pos)?;
                    trace.push(read_twin_state(&bins, &field, &pos, &head)?);
                    uniform.write(params(case, step))?;
                    k::walk::dispatch(
                        bins.device(),
                        case.walkers as u32,
                        &uniform,
                        Independent(&field),
                        Independent(bins.counts()),
                        &pos,
                        &head,
                        Independent(&deposit),
                    )?;
                    k::spread::dispatch(bins.device(), k::CELLS, &field, &deposit, &spread)?;
                    k::clear_deposit::dispatch(bins.device(), k::CELLS, &deposit)?;
                    std::mem::swap(&mut field, &mut spread);
                }
                bins.rebuild(&pos)?;
                trace.push(read_twin_state(&bins, &field, &pos, &head)?);
                Ok(trace)
            }
        }
    };
}

transport_program!(big, 128u32);
#[cfg(test)]
transport_program!(small, 16u32);

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let handle = default_device()?;
    let info = handle.adapter_info();
    println!("adapter: {} backend: {:?}", info.name, info.backend);
    println!(
        "grid {0} x {0}, {STEPS} steps, {1} cases",
        big::SIDE,
        big_cases().len()
    );
    for case in big_cases() {
        let want = reference_trace(big::SIDE, &case);
        let got = big::run_gpu(&handle, &case)?;
        verify(case.name, big::SIDE, &want, &got)?;
        println!(
            "gpu, {}: {} walkers equal to the host on every step (field, walkers and bins)",
            case.name, case.walkers
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bins::Snapshot;

    fn gpu_handle() -> DeviceHandle {
        default_device().unwrap_or_else(|error| {
            panic!("Vulkan adapter required; this test must not silently skip: {error}")
        })
    }

    #[test]
    fn the_host_reads_the_modules_constants() {
        assert_eq!(big::SIDE, 128);
        assert_eq!(small::SIDE, 16);
        assert!(big::SIDE.is_power_of_two() && small::SIDE.is_power_of_two());
        assert_eq!(big::k::MASK as usize, big::SIDE - 1);
        assert_eq!(small::k::CELLS as usize, small::SIDE * small::SIDE);
        // The struct is 32 bytes, a multiple of 16, without a gap.
        assert_eq!(std::mem::size_of::<small::k::Params>(), 32);
        // More walkers than one workgroup of 256 in some cases.
        assert!(small_cases().iter().any(|c| c.walkers > 256));
        assert!(big_cases().iter().all(|c| c.walkers > 256));
    }

    #[test]
    fn the_inputs_stay_inside_the_derivation_domain() {
        for (side, cases) in [(small::SIDE, small_cases()), (big::SIDE, big_cases())] {
            for case in cases {
                let start = make_start(side, &case);
                let bound = field_bound(&case, &start);
                assert!(bound < 1 << 32, "{}: {bound}", case.name);
                let trace = reference_trace(side, &case);
                let start_total: u64 = start.field.iter().map(|&f| u64::from(f)).sum();
                for (step, state) in trace.iter().enumerate() {
                    let largest = u64::from(*state.field.iter().max().unwrap());
                    assert!(
                        largest <= bound,
                        "{} step {step}: {largest} > {bound}",
                        case.name
                    );
                    // The spread conserves the sum and each walker deposits once a step.
                    let total: u64 = state.field.iter().map(|&f| u64::from(f)).sum();
                    assert_eq!(
                        total,
                        start_total + step as u64 * u64::from(case.amount) * case.walkers as u64,
                        "{} step {step}",
                        case.name
                    );
                    assert_eq!(
                        state.counts.iter().map(|&c| c as usize).sum::<usize>(),
                        case.walkers,
                        "{} step {step}",
                        case.name
                    );
                }
                // The rule fires when some walker is in a cell above the crowd. The inputs are chosen so that this
                // happens exactly when `crowd < walkers` (a property of these inputs, asserted below); no count can
                // exceed the walker count, so at `crowd == walkers` the rule never fires. When it fires the run
                // differs from the run without the rule, and when it does not the two runs are equal.
                let fires = trace[..STEPS]
                    .iter()
                    .any(|s| s.counts.iter().any(|&c| c > case.crowd));
                assert_eq!(
                    fires,
                    (case.crowd as usize) < case.walkers,
                    "{}: the rule fires",
                    case.name
                );
                if fires {
                    assert_ne!(
                        trace,
                        reference_trace(side, &rule_off(&case)),
                        "{}: the crowding rule changes nothing",
                        case.name
                    );
                } else {
                    assert_eq!(
                        trace,
                        reference_trace(side, &rule_off(&case)),
                        "{}",
                        case.name
                    );
                }
                // Contention is exercised, and the bound is guaranteed. With every walker in one cell and heading
                // (A = amount, W = walkers, all crowded), the walkers take one of three headings, so by pigeonhole
                // some deposit cell gets at least A*W/3. The spread keeps `g - 4*(g >> 3) >= g/2` of a cell's own
                // `g`, so after step 1 that cell holds at least A*W/6, which is more than A*W/8. A start spread out
                // over the grid deposits far less per cell and stays far below.
                if matches!(case.start, Start::OneCellOneHeading) && case.prob == 0 {
                    let largest = u64::from(*trace[1].field.iter().max().unwrap());
                    assert!(
                        largest >= u64::from(case.amount) * case.walkers as u64 / 8,
                        "{largest}"
                    );
                }
            }
        }
    }

    #[test]
    fn dist_outside_one_to_side_is_refused() {
        let side = small::SIDE;
        for dist in [0, side as u32 + 1] {
            let mut case = small_cases().remove(0);
            case.dist = dist;
            let result = std::panic::catch_unwind(|| make_start(side, &case));
            assert!(result.is_err(), "dist {dist}");
        }
        let mut case = small_cases().remove(0);
        case.dist = side as u32;
        assert_eq!(make_start(side, &case).pos.len(), case.walkers);
    }

    #[test]
    fn the_kernel_hash_is_the_host_hash() {
        for (seed, index) in [
            (0, 0),
            (1, 2),
            (2, 1),
            (0xdead_beef, 4095),
            (u32::MAX, u32::MAX),
        ] {
            assert_eq!(small::k::hash(seed, index), hash(seed, index));
        }
        assert_ne!(hash(1, 2), hash(2, 1));
    }

    #[test]
    fn the_host_bins_group_the_walkers_by_cell() {
        let (counts, order) = host_bins(4, &[2, 0, 2, 2, 3]);
        assert_eq!(counts, [1, 0, 3, 1]);
        assert_eq!(order, [1, 0, 2, 3, 4]);
    }

    #[test]
    fn grouped_sorts_each_cell_and_refuses_wrong_bins() {
        let snapshot = Snapshot {
            counts: vec![1, 0, 3, 1],
            starts: vec![0, 1, 1, 4],
            order: vec![1, 3, 0, 2, 4],
        };
        assert_eq!(snapshot.grouped().unwrap(), [1, 0, 2, 3, 4]);
        let wrong_start = Snapshot {
            starts: vec![0, 1, 2, 4],
            ..snapshot
        };
        assert!(wrong_start.grouped().unwrap_err().contains("starts[2]"));
        let short = Snapshot {
            counts: vec![1, 0, 3, 0],
            starts: vec![0, 1, 1, 4],
            order: vec![1, 3, 0, 2, 4],
        };
        assert!(short.grouped().unwrap_err().contains("sum to 4 for 5"));
    }

    #[test]
    fn a_wrong_walker_a_wrong_cell_and_wrong_bins_are_reported() {
        let case = &small_cases()[0];
        let want = reference_trace(small::SIDE, case);
        let mut got = want.clone();
        got[2].pos[7] ^= 1;
        let message = verify("t", small::SIDE, &want, &got).err().unwrap();
        assert!(message.contains("step 2: walker 7:"), "{message}");
        let mut got = want.clone();
        got[3].field[40] += 1;
        let message = verify("t", small::SIDE, &want, &got).err().unwrap();
        assert!(
            message.contains("step 3: field cell 40 (x 8, y 2)"),
            "{message}"
        );
        let mut got = want.clone();
        got[4].counts[9] += 1;
        let message = verify("t", small::SIDE, &want, &got).err().unwrap();
        assert!(
            message.contains("step 4: bin count of cell 9 (x 9, y 0)"),
            "{message}"
        );
        let mut got = want.clone();
        got[5].order[3] ^= 1;
        let message = verify("t", small::SIDE, &want, &got).err().unwrap();
        assert!(message.contains("step 5: bin slot 3:"), "{message}");
        assert!(verify("t", small::SIDE, &want, &want).is_ok());
    }

    #[test]
    fn twin_matches_the_host_on_the_small_grid() {
        for case in small_cases() {
            let want = reference_trace(small::SIDE, &case);
            let got = small::run_twin(&case).unwrap();
            verify(case.name, small::SIDE, &want, &got).unwrap();
            assert_ne!(want[0].field, want[STEPS].field, "{}", case.name);
        }
    }

    #[test]
    fn twin_matches_the_host_on_the_medium_grid() {
        for case in big_cases() {
            let want = reference_trace(big::SIDE, &case);
            let got = big::run_twin(&case).unwrap();
            verify(case.name, big::SIDE, &want, &got).unwrap();
        }
    }

    #[test]
    fn twin_a_crowd_above_the_walker_count_is_the_rule_off() {
        let mut checked = 0;
        for case in small_cases()
            .into_iter()
            .filter(|c| c.crowd as usize > c.walkers)
        {
            let off = reference_trace(small::SIDE, &rule_off(&case));
            let got = small::run_twin(&case).unwrap();
            verify(case.name, small::SIDE, &off, &got).unwrap();
            checked += 1;
        }
        assert!(checked > 0);
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_matches_the_host_on_the_small_grid() {
        let handle = gpu_handle();
        for case in small_cases() {
            let want = reference_trace(small::SIDE, &case);
            let got = small::run_gpu(&handle, &case).unwrap();
            verify(case.name, small::SIDE, &want, &got).unwrap();
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_matches_the_host_on_the_medium_grid() {
        let handle = gpu_handle();
        let info = handle.adapter_info();
        println!("adapter: {} backend: {:?}", info.name, info.backend);
        for case in big_cases() {
            let want = reference_trace(big::SIDE, &case);
            let got = big::run_gpu(&handle, &case).unwrap();
            verify(case.name, big::SIDE, &want, &got).unwrap();
            println!("gpu on {}, {}: equal on every step", info.name, case.name);
        }
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_a_crowd_above_the_walker_count_is_the_rule_off() {
        let handle = gpu_handle();
        let mut checked = 0;
        for case in small_cases()
            .into_iter()
            .filter(|c| c.crowd as usize > c.walkers)
        {
            let off = reference_trace(small::SIDE, &rule_off(&case));
            let got = small::run_gpu(&handle, &case).unwrap();
            verify(case.name, small::SIDE, &off, &got).unwrap();
            checked += 1;
        }
        for case in big_cases()
            .into_iter()
            .filter(|c| c.crowd as usize > c.walkers)
        {
            let off = reference_trace(big::SIDE, &rule_off(&case));
            let got = big::run_gpu(&handle, &case).unwrap();
            verify(case.name, big::SIDE, &off, &got).unwrap();
            checked += 1;
        }
        assert_eq!(checked, 2);
    }
}
