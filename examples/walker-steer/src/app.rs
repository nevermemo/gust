//! Walkers that sample a field ahead, turn toward the strongest sample, move and deposit, exact on host, twin and GPU.
//!
//! Many elements (walkers) move over a scalar field on a periodic `SIDE x SIDE` grid. Everything is integer, so the
//! host reference, the CPU twin and the GPU must agree bit for bit after every step. One step is two kernels:
//!
//! 1. `walk`, one invocation per walker. A walker is a cell index and a heading 0..7 (E, NE, N, NW, W, SW, S, SE, the
//!    eight neighbours counterclockwise). It
//!    - senses the field `dist` cells ahead in the headings `heading - 1`, `heading` and `heading + 1` (mod 8);
//!    - turns toward the largest sample. **Tie rule:** straight wins a tie with either side, and left (`heading - 1`)
//!      wins a tie with right (`heading + 1`); a side is chosen only when it is strictly larger than the best before it;
//!    - with probability `prob / 2^16` replaces that choice by one of the three headings, picked by the hash
//!      `hash(hash(seed, step), walker)` (the low 16 bits are compared with `prob`, the high 16 bits modulo 3 pick
//!      left, straight or right), so the choice is a pure function of `(seed, step, walker)`;
//!    - moves one cell in its new heading, wrapping at the edges, and adds `amount` to the deposit buffer at the new
//!      cell with `atomic_add`.
//! 2. `relax`, one invocation per cell: `field = field - (field >> shift) + deposit`, then the deposit is cleared.
//!
//! The walkers read only the field of the previous step (they never write it) and the deposit sums are integer
//! additions, which commute, so the order of the walkers never matters and a step is deterministic.
//!
//! The hash is the "lowbias32" finaliser of `examples/hash-rng` (`hash(seed, index) = mix(mix(seed) ^ index)`), written
//! again here in the module and on the host.
//!
//! # No `u32` overflow
//!
//! Let `W` be the number of walkers, `A` the amount and `s` the shift. A cell receives at most `A * W` in a step.
//! Claim: if every cell is at most `M = A * W * 2^s` before a step, it is at most `M` after it, for any number of
//! steps. A cell `f` becomes `f - floor(f / 2^s) + d` with `d <= A * W`; `floor(f / 2^s) > f / 2^s - 1`, so the new
//! value is below `f (1 - 2^-s) + 1 + A * W <= M - A * W + 1 + A * W = M + 1`, hence at most `M`. The documented
//! range is `A * W * 2^s < 2^32` with a starting field at most `M`, and `field_bound` is that `M`; the tests assert it
//! for their inputs and that no cell of the reference ever exceeds it. The subtraction never underflows (`f >> s <= f`)
//! and the deposit buffer holds at most `A * W` per cell, itself below `2^32`.
//!
//! # Why two modules
//!
//! `SIDE` is a module constant, so one `#[gpu]` module has one grid size. The macro below writes the module and the
//! host code that drives it once per size: `big` (128, the run of `main` and the GPU tests) and, under `cfg(test)`,
//! `small` (16).

#[cfg(any(test, gust_dual))]
use gust::cpu::CpuDevice;
use gust::{Independent, gpu};
use gust_wgpu::{DeviceHandle, GpuVec, HeadlessDevice, default_device};

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
    /// One cell and one heading: every walker makes the same choices unless the random turn splits them, so one
    /// deposit cell takes an `atomic_add` from every walker.
    OneCellOneHeading,
}

/// One run: the walkers and the parameters.
#[derive(Clone, Debug)]
pub struct Case {
    pub name: &'static str,
    pub walkers: usize,
    pub start: Start,
    /// How many cells ahead the walkers sense.
    pub dist: u32,
    /// The probability of a random turn, over `2^16`.
    pub prob: u32,
    /// What a walker adds to the deposit buffer.
    pub amount: u32,
    /// The field loses `field >> shift` a step.
    pub shift: u32,
    pub seed: u32,
}

/// The state after a step: the field, and per walker its cell and heading.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub field: Vec<u32>,
    pub pos: Vec<u32>,
    pub head: Vec<u32>,
}

/// `A * W * 2^s`, the bound on every cell of the field (module documentation, "No `u32` overflow").
#[cfg(test)]
pub fn field_bound(case: &Case) -> u64 {
    (u64::from(case.amount) * case.walkers as u64) << case.shift
}

/// The starting state of a `side x side` grid: a field below 32 (small, so equal samples are common and the tie rule
/// matters) and the walkers.
pub fn make_start(side: usize, case: &Case) -> State {
    let cells = side * side;
    let field = (0..cells).map(|i| hash(SEEDS[0], i as u32) >> 27).collect();
    let pos = (0..case.walkers)
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
    State { field, pos, head }
}

/// The parameter values of step `step` (1-based): `(seed, step, dist, prob, amount, shift)`.
fn step_params(case: &Case, step: usize) -> [u32; 6] {
    [
        case.seed,
        step as u32,
        case.dist,
        case.prob,
        case.amount,
        case.shift,
    ]
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
        // Straight first; a side wins only when strictly larger than the best so far (left before right).
        let mut best = 1;
        for side_index in [0, 2] {
            if samples[side_index] > samples[best] {
                best = side_index;
            }
        }
        let r = hash(hash(case.seed, step as u32), w as u32);
        if (r & 0xffff) < case.prob {
            best = ((r >> 16) % 3) as usize;
        }
        let new_heading = headings[best];
        let new_cell = at(cell, new_heading, 1);
        deposit[new_cell as usize] += case.amount;
        pos.push(new_cell);
        head.push(new_heading);
    }
    let field = now
        .field
        .iter()
        .zip(&deposit)
        .map(|(&f, &d)| f - (f >> case.shift) + d)
        .collect();
    State { field, pos, head }
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
/// differing cell, as a message.
pub fn verify(name: &str, side: usize, want: &[State], got: &[State]) -> Result<(), String> {
    if want.len() != got.len() {
        return Err(format!("{name}: {} states for {}", got.len(), want.len()));
    }
    for (step, (want, got)) in want.iter().zip(got).enumerate() {
        if want.pos.len() != got.pos.len()
            || want.head.len() != got.head.len()
            || want.field.len() != got.field.len()
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
    }
    Ok(())
}

/// The cases of the small grid: fewer walkers than a workgroup, then more, then all random turns.
#[cfg(test)]
pub fn small_cases() -> Vec<Case> {
    vec![
        case("spread, p = 0", 50, Start::Spread, 3, 0),
        case("spread, p = 1/4", 300, Start::Spread, 5, 1 << 14),
        case("spread, p = 1, wraps", 300, Start::Spread, 7, 1 << 16),
        case("one cell, p = 0", 300, Start::OneCell, 2, 0),
    ]
}

/// The cases of the large grid: many workgroups, heavy contention, `p = 0` and `p = 2^16`.
pub fn big_cases() -> Vec<Case> {
    vec![
        case("spread, p = 0", 3000, Start::Spread, 4, 0),
        case("spread, p = 1/16", 3000, Start::Spread, 6, 1 << 12),
        case("spread, p = 1", 3000, Start::Spread, 4, 1 << 16),
        case(
            "one cell and heading, p = 0",
            3000,
            Start::OneCellOneHeading,
            4,
            0,
        ),
        case(
            "one cell and heading, p = 1",
            3000,
            Start::OneCellOneHeading,
            4,
            1 << 16,
        ),
        case("one cell, p = 1/16", 3000, Start::OneCell, 3, 1 << 12),
    ]
}

fn case(name: &'static str, walkers: usize, start: Start, dist: u32, prob: u32) -> Case {
    Case {
        name,
        walkers,
        start,
        dist,
        prob,
        amount: 7,
        shift: 3,
        seed: 0xdead_beef,
    }
}

/// Writes, per size, the `#[gpu]` module and the host code that drives it.
macro_rules! walker_program {
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
                /// deposit amount, the field's shift, and two pad fields.
                pub struct Params {
                    pub seed: u32,
                    pub step: u32,
                    pub dist: u32,
                    pub prob: u32,
                    pub amount: u32,
                    pub shift: u32,
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
                    let mut r = 0i32;
                    if d == 0u32 || d == 1u32 || d == 7u32 {
                        r = 1i32;
                    }
                    if d == 3u32 || d == 4u32 || d == 5u32 {
                        r = -1i32;
                    }
                    r
                }

                /// The y step of heading `d`: 1 for the headings with a north part, -1 for a south part.
                pub fn dir_y(d: u32) -> i32 {
                    let mut r = 0i32;
                    if d == 1u32 || d == 2u32 || d == 3u32 {
                        r = 1i32;
                    }
                    if d == 5u32 || d == 6u32 || d == 7u32 {
                        r = -1i32;
                    }
                    r
                }

                /// The cell `k` steps from `(x, y)` in heading `d`, wrapping.
                pub fn ahead(x: u32, y: u32, d: u32, k: u32) -> u32 {
                    let n = SIDE as i32;
                    let nx = ((x as i32) + dir_x(d) * (k as i32)).rem_euclid(n);
                    let ny = ((y as i32) + dir_y(d) * (k as i32)).rem_euclid(n);
                    (ny as u32) * SIDE + (nx as u32)
                }

                /// One walker: sense, turn, move, deposit. Reads `field` (never written here), writes its own
                /// `pos` and `head`, and adds into `deposit`.
                #[kernel(workgroup_size(256, 1, 1))]
                pub fn walk(
                    id: SV_DispatchThreadID,
                    p: Uniform<Params>,
                    field: StructuredBuffer<uint>,
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
                        let left = (h + 7u32) & 7u32;
                        let right = (h + 1u32) & 7u32;
                        let vl = field[ahead(x, y, left, p.dist)];
                        let vs = field[ahead(x, y, h, p.dist)];
                        let vr = field[ahead(x, y, right, p.dist)];
                        // 0 is left, 1 straight, 2 right. Straight wins ties; left wins a tie with right.
                        let mut turn = 1u32;
                        let mut best = vs;
                        if vl > best {
                            turn = 0u32;
                            best = vl;
                        }
                        if vr > best {
                            turn = 2u32;
                        }
                        let r = hash(hash(p.seed, p.step), i);
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

                /// One cell: `field = field - (field >> shift) + deposit`, then the deposit is cleared.
                #[kernel(workgroup_size(256, 1, 1))]
                pub fn relax(
                    id: SV_DispatchThreadID,
                    p: Uniform<Params>,
                    mut field: RWStructuredBuffer<uint>,
                    mut deposit: RWStructuredBuffer<uint>,
                ) {
                    let i = id.x;
                    if i < CELLS {
                        let f = field[i];
                        field[i] = f - (f >> p.shift) + deposit[i];
                        deposit[i] = 0u32;
                    }
                }
            }

            pub const SIDE: usize = k::SIDE as usize;

            /// The parameter struct of step `step`.
            pub fn params(case: &Case, step: usize) -> k::Params {
                let [seed, step, dist, prob, amount, shift] = step_params(case, step);
                k::Params {
                    seed,
                    step,
                    dist,
                    prob,
                    amount,
                    shift,
                    pad0: 0,
                    pad1: 0,
                }
            }

            /// `STEPS` steps on the device, reading everything back after every step.
            pub fn run_gpu(
                handle: &DeviceHandle,
                case: &Case,
            ) -> Result<Vec<State>, Box<dyn std::error::Error>> {
                let device: &HeadlessDevice = handle;
                let start = make_start(SIDE, case);
                let field = GpuVec::from_slice_on(handle, &start.field)?;
                let deposit = GpuVec::<u32>::zeroed_on(handle, k::CELLS)?;
                let pos = GpuVec::from_slice_on(handle, &start.pos)?;
                let head = GpuVec::from_slice_on(handle, &start.head)?;
                let mut trace = vec![start];
                // `field` and `deposit` are `Independent` in `walk`: their length is the cell count, not the
                // number of walkers it is dispatched for.
                for step in 1..=STEPS {
                    let uniform =
                        device.create_uniform("walker-steer params", &params(case, step))?;
                    k::walk::dispatch(
                        device,
                        case.walkers as u32,
                        &uniform,
                        Independent(&field),
                        &pos,
                        &head,
                        Independent(&deposit),
                    )?;
                    k::relax::dispatch(device, k::CELLS, &uniform, &field, &deposit)?;
                    trace.push(State {
                        field: field.to_vec()?,
                        pos: pos.to_vec()?,
                        head: head.to_vec()?,
                    });
                }
                Ok(trace)
            }

            /// The same steps on the CPU twin (the kernels' own Rust bodies).
            #[cfg(any(test, gust_dual))]
            pub fn run_twin(case: &Case) -> Result<Vec<State>, Box<dyn std::error::Error>> {
                let device = CpuDevice::new();
                let start = make_start(SIDE, case);
                let field = device.buffer(&start.field);
                let deposit = device.buffer(&vec![0u32; SIDE * SIDE]);
                let pos = device.buffer(&start.pos);
                let head = device.buffer(&start.head);
                let mut trace = vec![start];
                for step in 1..=STEPS {
                    let uniform = device.uniform(params(case, step))?;
                    k::walk::dispatch(
                        &device,
                        case.walkers as u32,
                        &uniform,
                        Independent(&field),
                        &pos,
                        &head,
                        Independent(&deposit),
                    )?;
                    k::relax::dispatch(&device, k::CELLS, &uniform, &field, &deposit)?;
                    trace.push(State {
                        field: field.read(),
                        pos: pos.read(),
                        head: head.read(),
                    });
                }
                Ok(trace)
            }
        }
    };
}

walker_program!(big, 128u32);
#[cfg(test)]
walker_program!(small, 16u32);

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
            "gpu, {}: {} walkers equal to the host on every step (field and walkers)",
            case.name, case.walkers
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
                let bound = field_bound(&case);
                assert!(bound < 1 << 32, "{}: {bound}", case.name);
                let trace = reference_trace(side, &case);
                for (step, state) in trace.iter().enumerate() {
                    let largest = u64::from(*state.field.iter().max().unwrap());
                    assert!(
                        largest <= bound,
                        "{} step {step}: {largest} > {bound}",
                        case.name
                    );
                }
                // Contention is exercised: with every walker in one cell and heading, one cell takes `amount * walkers`.
                if matches!(case.start, Start::OneCellOneHeading) && case.prob == 0 {
                    let largest = u64::from(*trace[1].field.iter().max().unwrap());
                    assert!(
                        largest >= u64::from(case.amount) * case.walkers as u64,
                        "{largest}"
                    );
                }
            }
        }
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
    fn the_random_turn_changes_the_run_and_p_zero_does_not_use_it() {
        let side = small::SIDE;
        let mut cases = small_cases();
        let zero = cases.remove(0);
        let mut random = zero.clone();
        random.prob = 1 << 16;
        let a = reference_trace(side, &zero);
        let b = reference_trace(side, &random);
        assert_ne!(a[STEPS].head, b[STEPS].head);
        // With `p = 0` a different seed changes nothing.
        let mut other_seed = zero.clone();
        other_seed.seed ^= 1;
        assert_eq!(a, reference_trace(side, &other_seed));
        // Walkers do turn on their own with `p = 0` (the field steers them).
        assert_ne!(a[0].head, a[STEPS].head);
    }

    #[test]
    fn a_wrong_walker_and_a_wrong_cell_are_reported() {
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
}
