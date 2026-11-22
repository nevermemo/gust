//! Walkers that retire and spawn, kept dense in their buffers by an exclusive scan, exact on host, twin and GPU.
//!
//! Many elements (walkers) move on a periodic `SIDE x SIDE` grid and the population changes: some walkers retire each
//! step and others spawn a child, and the live walkers must stay dense in their buffers, so later kernels can launch
//! over `0..n`. Everything is integer and the result is a pure function of the previous state, so the host reference,
//! the CPU twin and the GPU must agree bit for bit after every step: every field of every walker and the count `n`.
//!
//! A walker is a cell index, a heading 0..7 (E, NE, N, NW, W, SW, S, SE, counterclockwise) and an energy (`u32`, at
//! least 1 while the walker lives), in buffers of a fixed capacity `cap`; the live count `n <= cap` is known to the
//! host before each step. One step is three phases:
//!
//! 1. `act`, one invocation per slot (`cap` of them, so the counts past `n` are written too). A live slot `i < n`
//!    - turns by `heading - 1`, `heading` or `heading + 1` and moves one cell, wrapping at the edges (the turn is the
//!      low 16 bits of a second hash draw modulo 3, slightly biased);
//!    - loses one energy, to `e`;
//!    - **retires** when `e` is 0 or the low 16 bits of the draw `hash(hash(seed, step), i)` are below `retire`
//!      (over `2^16`); `keep` is 0 then, else 1;
//!    - **spawns** one child (`kids` 1) when `e` is above `threshold` (at least 1) and the high 16 bits of the same
//!      draw are below `spawn` (over `2^16`). A retiring walker can still spawn: the child takes its place.
//!
//!    It writes `flags[i]` (bit 0 `keep`, bit 1 `kids`; `compact` reads this one buffer, which keeps its buffers
//!    within the eight storage buffers per stage that an adapter guarantees) and `counts[i] = keep + kids`; a slot
//!    at or past `n` writes a count of 0.
//! 2. `exclusive_into` (`gust_algo::scan`) over the `cap` counts gives each walker its first output slot, `starts[i]`.
//!    The slots past `n` count 0, so the scan over `cap` is the scan over the `n` live walkers, and the sum of the
//!    counts is `starts[cap - 1] + counts[cap - 1]`: the host reads those two back (the whole buffers, one `to_vec`
//!    each; no extra kernel) and the next count is that sum clamped to `cap`.
//! 3. `compact`, one invocation per live walker (`n` of them). A kept walker writes itself to slot `starts[i]`; a
//!    child (energy `e / 2`, the heading `(draw2 >> 16) & 7`, the parent's new cell) to `starts[i] + keep`; the
//!    parent keeps `e - e / 2` when it spawned, else `e`. The output goes to a second set of buffers and the host
//!    swaps the sets. **Clamp:** a slot at or past `cap` is dropped, so the population is clamped at `cap`,
//!    deterministically, by slot: the walkers of the lowest slots survive.
//!
//! No `atomic_add` assigns the slots: an atomic append would make the order of the spawned walkers depend on the
//! schedule, and the walkers could not be compared one by one. The scan makes every slot a pure function of the
//! counts, so there is exactly one correct result, independent of the order of the invocations.
//!
//! The host reference is written plainly from these rules (a loop over the walkers that pushes into vectors and
//! truncates at `cap`), not from the kernels. With no live walker (`n = 0`) a step does nothing on every side.
//!
//! The hash is the "lowbias32" finaliser of `examples/hash-rng` (`hash(seed, index) = mix(mix(seed) ^ index)`), written
//! again here in the module and on the host.
//!
//! # Why two modules
//!
//! `SIDE` is a module constant, so one `#[gpu]` module has one grid size. The macro below writes the module and the
//! host code that drives it once per size: `big` (128, the run of `main` and the GPU tests) and, under `cfg(test)`,
//! `small` (16).

#[cfg(any(test, gust_dual))]
use gust::cpu::{CpuBuffer, CpuDevice};
use gust::{Independent, gpu};
use gust_wgpu::{DeviceHandle, GpuVec, HeadlessDevice, default_device};

/// Steps of every run.
pub const STEPS: usize = 24;
/// Seeds of the walkers' cells, headings and energies.
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
/// Xored into the seed for the second draw of a walker (the golden-ratio constant), so it is independent of the first.
const SALT: u32 = 0x9e37_79b9;

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

/// One run: the starting population and the rules.
#[derive(Clone, Debug)]
pub struct Case {
    pub name: &'static str,
    /// Walkers at the start: `1 <= start <= cap` (checked by `make_start`).
    pub start: usize,
    /// The capacity of every buffer: the population never exceeds it.
    pub cap: usize,
    /// Starting energies are hash-chosen in `1..=max_energy`.
    pub max_energy: u32,
    /// The probability of retiring a step, over `2^16`.
    pub retire: u32,
    /// The probability of spawning a step (when the energy allows), over `2^16`.
    pub spawn: u32,
    /// A walker spawns only with an energy above this: `threshold >= 1`, so a child has at least 1.
    pub threshold: u32,
    pub seed: u32,
}

/// The live walkers: per walker its cell, heading and energy. Its length is the live count `n`.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub pos: Vec<u32>,
    pub head: Vec<u32>,
    pub energy: Vec<u32>,
}

impl State {
    fn empty() -> Self {
        Self {
            pos: Vec::new(),
            head: Vec::new(),
            energy: Vec::new(),
        }
    }
}

/// The starting state of a `side x side` grid.
///
/// # Panics
///
/// When `start` is outside `1..=cap`, `max_energy` is 0 or `threshold` is 0: every run starts here.
pub fn make_start(side: usize, case: &Case) -> State {
    assert!(
        (1..=case.cap).contains(&case.start),
        "{}: start {} is outside 1..={}",
        case.name,
        case.start,
        case.cap
    );
    assert!(case.max_energy >= 1, "{}: max_energy is 0", case.name);
    assert!(case.threshold >= 1, "{}: threshold is 0", case.name);
    let cells = (side * side) as u32;
    let walkers = 0..case.start as u32;
    State {
        pos: walkers.clone().map(|w| hash(SEEDS[0], w) % cells).collect(),
        head: walkers.clone().map(|w| hash(SEEDS[1], w) % 8).collect(),
        energy: walkers
            .map(|w| 1 + hash(SEEDS[2], w) % case.max_energy)
            .collect(),
    }
}

/// One step on the host, written from the rules: every walker's products in walker order, before the clamp. Its
/// length may exceed `cap`.
pub fn step_unclamped(side: usize, case: &Case, step: usize, now: &State) -> State {
    let n = side as i64;
    let mut out = State::empty();
    for w in 0..now.pos.len() {
        let r = hash(hash(case.seed, step as u32), w as u32);
        let r2 = hash(hash(case.seed ^ SALT, step as u32), w as u32);
        let heading = (now.head[w] + 7 + (r2 & 0xffff) % 3) % 8;
        let (dx, dy) = DIRS[heading as usize];
        let x = i64::from(now.pos[w]) % n + dx;
        let y = i64::from(now.pos[w]) / n + dy;
        let cell = (y.rem_euclid(n) * n + x.rem_euclid(n)) as u32;
        let energy = now.energy[w] - 1;
        let keeps = energy != 0 && (r & 0xffff) >= case.retire;
        let spawns = energy > case.threshold && (r >> 16) < case.spawn;
        if keeps {
            out.pos.push(cell);
            out.head.push(heading);
            out.energy
                .push(if spawns { energy - energy / 2 } else { energy });
        }
        if spawns {
            out.pos.push(cell);
            out.head.push((r2 >> 16) % 8);
            out.energy.push(energy / 2);
        }
    }
    out
}

/// One step on the host: [`step_unclamped`], cut at `cap` (the walkers of the lowest slots survive).
pub fn reference_step(side: usize, case: &Case, step: usize, now: &State) -> State {
    let mut out = step_unclamped(side, case, step, now);
    out.pos.truncate(case.cap);
    out.head.truncate(case.cap);
    out.energy.truncate(case.cap);
    out
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

/// The live count after a step: the sum of the counts, clamped at the capacity.
fn next_count(total: u32, cap: u32) -> u32 {
    total.min(cap)
}

/// Compares a device trace with the reference after every step: the live count, then the first differing walker, as
/// a message.
pub fn verify(name: &str, want: &[State], got: &[State]) -> Result<(), String> {
    if want.len() != got.len() {
        return Err(format!("{name}: {} states for {}", got.len(), want.len()));
    }
    for (step, (want, got)) in want.iter().zip(got).enumerate() {
        if got.pos.len() != want.pos.len()
            || got.head.len() != want.pos.len()
            || got.energy.len() != want.pos.len()
        {
            return Err(format!(
                "{name}: step {step}: {} walkers, host {}",
                got.pos.len(),
                want.pos.len()
            ));
        }
        if let Some(w) = (0..want.pos.len()).find(|&w| {
            want.pos[w] != got.pos[w]
                || want.head[w] != got.head[w]
                || want.energy[w] != got.energy[w]
        }) {
            return Err(format!(
                "{name}: step {step}: walker {w}: got cell {} heading {} energy {}, host cell {} heading {} energy {}",
                got.pos[w], got.head[w], got.energy[w], want.pos[w], want.head[w], want.energy[w]
            ));
        }
    }
    Ok(())
}

/// The cases of the small grid: fewer walkers than a workgroup that die out, a steady population, one that hits the
/// capacity, and more walkers than a workgroup.
#[cfg(test)]
pub fn small_cases() -> Vec<Case> {
    vec![
        case("dies out", 50, 100, 4, 0, 0, 1),
        case("steady", 100, 400, 80, 1 << 11, 1 << 11, 3),
        case("hits the capacity", 200, 300, 40, 0, 1 << 15, 2),
        case("more than a workgroup", 600, 1000, 10, 1 << 12, 1 << 13, 2),
    ]
}

/// The cases of the large grid: many workgroups and a scan of several workgroups.
pub fn big_cases() -> Vec<Case> {
    vec![
        case("dies out", 3000, 4096, 6, 0, 0, 1),
        case("steady", 3000, 5000, 80, 1 << 11, 1 << 11, 4),
        case("hits the capacity", 3000, 4096, 30, 1 << 10, 1 << 15, 2),
    ]
}

fn case(
    name: &'static str,
    start: usize,
    cap: usize,
    max_energy: u32,
    retire: u32,
    spawn: u32,
    threshold: u32,
) -> Case {
    Case {
        name,
        start,
        cap,
        max_energy,
        retire,
        spawn,
        threshold,
        seed: 0xdead_beef,
    }
}

/// Writes, per size, the `#[gpu]` module and the host code that drives it.
macro_rules! life_program {
    ($name:ident, $side:tt) => {
        pub mod $name {
            use super::*;

            #[gpu]
            pub mod k {
                /// Cells per side: a power of two.
                pub const SIDE: u32 = $side;
                /// Mask of a coordinate: `SIDE - 1`.
                pub const MASK: u32 = SIDE - 1;

                /// The parameters of a step, eight `u32` (32 bytes, a multiple of 16, so no gap): the seed and the
                /// step of the hash, the live count, the capacity, the retire and spawn probabilities over `2^16`,
                /// the spawn threshold, and a pad field.
                pub struct Params {
                    pub seed: u32,
                    pub step: u32,
                    pub n: u32,
                    pub cap: u32,
                    pub retire: u32,
                    pub spawn: u32,
                    pub threshold: u32,
                    pub pad0: u32,
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

                /// The second draw of walker `i` in a step: independent of the first (the seed is salted). Its
                /// low 16 bits give the turn, its high 16 bits the heading of a child.
                pub fn draw2(seed: u32, step: u32, i: u32) -> u32 {
                    hash(hash(seed ^ 2654435769u32, step), i)
                }

                /// One slot: a live walker (`i < n`) turns, moves, loses one energy and decides to retire and
                /// to spawn; it writes its new state in place, `flags` and `counts`. A slot at or past `n` writes
                /// a count of 0, so the scan over `cap` counts only the live walkers.
                #[kernel(workgroup_size(256, 1, 1))]
                pub fn act(
                    id: SV_DispatchThreadID,
                    p: Uniform<Params>,
                    mut pos: RWStructuredBuffer<uint>,
                    mut head: RWStructuredBuffer<uint>,
                    mut energy: RWStructuredBuffer<uint>,
                    mut flags: RWStructuredBuffer<uint>,
                    mut counts: RWStructuredBuffer<uint>,
                ) {
                    let i = id.x;
                    if i < counts.len() {
                        if i < p.n {
                            let c = pos[i];
                            let r = hash(hash(p.seed, p.step), i);
                            let r2 = draw2(p.seed, p.step, i);
                            let nh = (head[i] + 7u32 + (r2 & 65535u32) % 3u32) & 7u32;
                            let nc = ahead(c & MASK, c / SIDE, nh, 1u32);
                            let e = energy[i] - 1u32;
                            let mut alive = 1u32;
                            if e == 0u32 {
                                alive = 0u32;
                            }
                            if (r & 65535u32) < p.retire {
                                alive = 0u32;
                            }
                            let mut kid = 0u32;
                            if e > p.threshold {
                                if (r >> 16u32) < p.spawn {
                                    kid = 1u32;
                                }
                            }
                            pos[i] = nc;
                            head[i] = nh;
                            energy[i] = e;
                            flags[i] = alive + 2u32 * kid;
                            counts[i] = alive + kid;
                        }
                        if i >= p.n {
                            counts[i] = 0u32;
                        }
                    }
                }

                /// One live walker: a kept walker goes to slot `starts[i]`, its child to `starts[i] + keep`,
                /// in the second set of buffers; a slot at or past `cap` is dropped (the clamp).
                #[allow(clippy::too_many_arguments)]
                #[kernel(workgroup_size(256, 1, 1))]
                pub fn compact(
                    id: SV_DispatchThreadID,
                    p: Uniform<Params>,
                    pos: StructuredBuffer<uint>,
                    head: StructuredBuffer<uint>,
                    energy: StructuredBuffer<uint>,
                    flags: StructuredBuffer<uint>,
                    starts: StructuredBuffer<uint>,
                    mut out_pos: RWStructuredBuffer<uint>,
                    mut out_head: RWStructuredBuffer<uint>,
                    mut out_energy: RWStructuredBuffer<uint>,
                ) {
                    let i = id.x;
                    if i < p.n {
                        let s = starts[i];
                        let f = flags[i];
                        let kept = f & 1u32;
                        let kid = f >> 1u32;
                        let e = energy[i];
                        let half = e / 2u32;
                        let mut own = e;
                        if kid == 1u32 {
                            own = e - half;
                        }
                        if kept == 1u32 {
                            if s < p.cap {
                                out_pos[s] = pos[i];
                                out_head[s] = head[i];
                                out_energy[s] = own;
                            }
                        }
                        if kid == 1u32 {
                            let t = s + kept;
                            if t < p.cap {
                                out_pos[t] = pos[i];
                                out_head[t] = (draw2(p.seed, p.step, i) >> 16u32) & 7u32;
                                out_energy[t] = half;
                            }
                        }
                    }
                }
            }

            pub const SIDE: usize = k::SIDE as usize;

            /// The parameter struct of step `step` (1-based) with `n` live walkers.
            pub fn params(case: &Case, step: usize, n: u32) -> k::Params {
                k::Params {
                    seed: case.seed,
                    step: step as u32,
                    n,
                    cap: case.cap as u32,
                    retire: case.retire,
                    spawn: case.spawn,
                    threshold: case.threshold,
                    pad0: 0,
                }
            }

            /// `v` as a buffer of `cap` elements, the tail zero.
            fn padded(v: &[u32], cap: usize) -> Vec<u32> {
                let mut out = v.to_vec();
                out.resize(cap, 0);
                out
            }

            /// The three walker buffers of one set, on the device.
            struct Set {
                pos: GpuVec<u32>,
                head: GpuVec<u32>,
                energy: GpuVec<u32>,
            }

            impl Set {
                fn from_state(
                    handle: &DeviceHandle,
                    state: &State,
                    cap: usize,
                ) -> Result<Self, Box<dyn std::error::Error>> {
                    Ok(Self {
                        pos: GpuVec::from_slice_on(handle, &padded(&state.pos, cap))?,
                        head: GpuVec::from_slice_on(handle, &padded(&state.head, cap))?,
                        energy: GpuVec::from_slice_on(handle, &padded(&state.energy, cap))?,
                    })
                }

                /// The first `n` walkers, read back.
                fn read(&self, n: usize) -> Result<State, Box<dyn std::error::Error>> {
                    let mut state = State {
                        pos: self.pos.to_vec()?,
                        head: self.head.to_vec()?,
                        energy: self.energy.to_vec()?,
                    };
                    state.pos.truncate(n);
                    state.head.truncate(n);
                    state.energy.truncate(n);
                    Ok(state)
                }
            }

            /// `STEPS` steps on the device, reading the live walkers back after every step.
            pub fn run_gpu(
                handle: &DeviceHandle,
                case: &Case,
            ) -> Result<Vec<State>, Box<dyn std::error::Error>> {
                let device: &HeadlessDevice = handle;
                let start = make_start(SIDE, case);
                let cap = case.cap;
                let mut n = start.pos.len() as u32;
                let mut cur = Set::from_state(handle, &start, cap)?;
                let mut next = Set::from_state(handle, &State::empty(), cap)?;
                let flags = GpuVec::<u32>::zeroed_on(handle, cap as u32)?;
                let counts = GpuVec::<u32>::zeroed_on(handle, cap as u32)?;
                let starts = GpuVec::<u32>::zeroed_on(handle, cap as u32)?;
                let mut trace = vec![start];
                let uniform = device.create_uniform("walker-life params", &params(case, 1, n))?;
                for step in 1..=STEPS {
                    // With no live walker a step does nothing.
                    if n > 0 {
                        device.write_uniform(&uniform, &params(case, step, n))?;
                        k::act::dispatch(
                            device,
                            cap as u32,
                            &uniform,
                            &cur.pos,
                            &cur.head,
                            &cur.energy,
                            &flags,
                            &counts,
                        )?;
                        gust_algo::scan::exclusive_into(&counts, &starts)?;
                        // The buffers hold `cap` elements and `compact` runs over the `n` live walkers.
                        k::compact::dispatch(
                            device,
                            n,
                            &uniform,
                            Independent(&cur.pos),
                            Independent(&cur.head),
                            Independent(&cur.energy),
                            Independent(&flags),
                            Independent(&starts),
                            Independent(&next.pos),
                            Independent(&next.head),
                            Independent(&next.energy),
                        )?;
                        let last = cap - 1;
                        let total = starts.to_vec()?[last] + counts.to_vec()?[last];
                        n = next_count(total, cap as u32);
                        std::mem::swap(&mut cur, &mut next);
                    }
                    trace.push(cur.read(n as usize)?);
                }
                Ok(trace)
            }

            /// The same steps on the CPU twin (the kernels' own Rust bodies; the scan is a plain host prefix sum,
            /// as in `examples/cell-bins`: the twin of `gust_algo::scan` is not available outside that crate).
            #[cfg(any(test, gust_dual))]
            pub fn run_twin(case: &Case) -> Result<Vec<State>, Box<dyn std::error::Error>> {
                let device = CpuDevice::new();
                let start = make_start(SIDE, case);
                let cap = case.cap;
                let mut n = start.pos.len() as u32;
                let buffer = |v: &[u32]| device.buffer(&padded(v, cap));
                let mut cur = [
                    buffer(&start.pos),
                    buffer(&start.head),
                    buffer(&start.energy),
                ];
                let mut next = [buffer(&[]), buffer(&[]), buffer(&[])];
                let flags = buffer(&[]);
                let counts = buffer(&[]);
                let starts = buffer(&[]);
                let mut trace = vec![start];
                let uniform = device.uniform(params(case, 1, n))?;
                let read = |set: &[CpuBuffer<u32>; 3], n: u32| State {
                    pos: set[0].read()[..n as usize].to_vec(),
                    head: set[1].read()[..n as usize].to_vec(),
                    energy: set[2].read()[..n as usize].to_vec(),
                };
                for step in 1..=STEPS {
                    if n > 0 {
                        uniform.write(params(case, step, n))?;
                        k::act::dispatch(
                            &device, cap as u32, &uniform, &cur[0], &cur[1], &cur[2], &flags,
                            &counts,
                        )?;
                        let sums: Vec<u32> = counts
                            .read()
                            .iter()
                            .scan(0u32, |total, &count| {
                                let start = *total;
                                *total += count;
                                Some(start)
                            })
                            .collect();
                        starts.write(&sums)?;
                        k::compact::dispatch(
                            &device,
                            n,
                            &uniform,
                            Independent(&cur[0]),
                            Independent(&cur[1]),
                            Independent(&cur[2]),
                            Independent(&flags),
                            Independent(&starts),
                            Independent(&next[0]),
                            Independent(&next[1]),
                            Independent(&next[2]),
                        )?;
                        let last = cap - 1;
                        let total = starts.read()[last] + counts.read()[last];
                        n = next_count(total, cap as u32);
                        std::mem::swap(&mut cur, &mut next);
                    }
                    trace.push(read(&cur, n));
                }
                Ok(trace)
            }
        }
    };
}

life_program!(big, 128u32);
#[cfg(test)]
life_program!(small, 16u32);

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
        verify(case.name, &want, &got)?;
        println!(
            "gpu, {}: {} walkers at the start, {} after {STEPS} steps, equal to the host on every step",
            case.name,
            case.start,
            got[STEPS].pos.len()
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

    fn counts(trace: &[State]) -> Vec<usize> {
        trace.iter().map(|state| state.pos.len()).collect()
    }

    #[test]
    fn the_host_reads_the_modules_constants() {
        assert_eq!(big::SIDE, 128);
        assert_eq!(small::SIDE, 16);
        assert!(big::SIDE.is_power_of_two() && small::SIDE.is_power_of_two());
        assert_eq!(big::k::MASK as usize, big::SIDE - 1);
        // The struct is 32 bytes, a multiple of 16, without a gap.
        assert_eq!(std::mem::size_of::<small::k::Params>(), 32);
        // More walkers than one workgroup of 256 in some cases.
        assert!(small_cases().iter().any(|c| c.start > 256));
        assert!(big_cases().iter().all(|c| c.start > 256));
    }

    #[test]
    fn the_cases_exercise_the_rules() {
        for (side, cases) in [(small::SIDE, small_cases()), (big::SIDE, big_cases())] {
            let mut died = 0;
            let mut clamped = 0;
            for case in cases {
                let trace = reference_trace(side, &case);
                // Every live walker has at least 1 energy, a cell on the grid and a heading, within the capacity.
                for state in &trace {
                    assert!(state.pos.len() <= case.cap, "{}", case.name);
                    assert!(state.energy.iter().all(|&e| e >= 1), "{}", case.name);
                    assert!(state.pos.iter().all(|&c| (c as usize) < side * side));
                    assert!(state.head.iter().all(|&h| h < 8));
                }
                if case.name == "dies out" {
                    died += 1;
                    // After it died out the next step is a no-op: the same empty state, every step.
                    let at = trace.iter().position(|s| s.pos.is_empty()).unwrap();
                    assert!(at < STEPS, "{}: dies out at {at}", case.name);
                    assert!(trace[at..].iter().all(|s| s.pos.is_empty()));
                }
                if case.name == "hits the capacity" {
                    clamped += 1;
                    // Some step makes more walkers than the capacity holds: the clamp drops some.
                    assert!(
                        (1..=STEPS).any(|step| {
                            step_unclamped(side, &case, step, &trace[step - 1])
                                .pos
                                .len()
                                > case.cap
                        }),
                        "{}",
                        case.name
                    );
                    assert!(trace.iter().any(|s| s.pos.len() == case.cap));
                }
                if case.name == "steady" {
                    // Alive at the end, so the late steps scan a real population.
                    assert!(!trace[STEPS].pos.is_empty(), "{}", case.name);
                }
            }
            assert_eq!(died, 1, "one case dies out");
            assert_eq!(clamped, 1, "one case hits the capacity");
        }
    }

    #[test]
    fn the_population_changes_in_the_steady_case() {
        let case = &small_cases()[1];
        let trace = reference_trace(small::SIDE, case);
        let n = counts(&trace);
        // Both retiring and spawning happen: some step loses and some step gains walkers.
        assert!(n.windows(2).any(|w| w[1] < w[0]), "{n:?}");
        assert!(n.windows(2).any(|w| w[1] > w[0]), "{n:?}");
    }

    #[test]
    fn a_start_outside_the_capacity_is_refused() {
        let side = small::SIDE;
        for start in [0, small_cases()[0].cap + 1] {
            let mut case = small_cases().remove(0);
            case.start = start;
            assert!(std::panic::catch_unwind(|| make_start(side, &case)).is_err());
        }
        let mut case = small_cases().remove(0);
        case.start = case.cap;
        assert_eq!(make_start(side, &case).pos.len(), case.cap);
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
            assert_eq!(
                small::k::draw2(seed, 5, index),
                hash(hash(seed ^ SALT, 5), index)
            );
        }
        assert_ne!(hash(1, 2), hash(2, 1));
    }

    #[test]
    fn a_wrong_walker_and_a_wrong_count_are_reported() {
        let case = &small_cases()[1];
        let want = reference_trace(small::SIDE, case);
        let mut got = want.clone();
        got[2].energy[7] += 1;
        let message = verify("t", &want, &got).err().unwrap();
        assert!(message.contains("step 2: walker 7:"), "{message}");
        let mut got = want.clone();
        got[3].pos.pop();
        got[3].head.pop();
        got[3].energy.pop();
        let message = verify("t", &want, &got).err().unwrap();
        assert!(
            message.contains("step 3:") && message.contains("walkers, host"),
            "{message}"
        );
        assert!(verify("t", &want, &want).is_ok());
    }

    #[test]
    fn twin_matches_the_host_on_the_small_grid() {
        for case in small_cases() {
            let want = reference_trace(small::SIDE, &case);
            let got = small::run_twin(&case).unwrap();
            verify(case.name, &want, &got).unwrap();
            assert_ne!(want[0], want[STEPS], "{}", case.name);
        }
    }

    #[test]
    fn twin_matches_the_host_on_the_medium_grid() {
        for case in big_cases() {
            let want = reference_trace(big::SIDE, &case);
            let got = big::run_twin(&case).unwrap();
            verify(case.name, &want, &got).unwrap();
        }
    }

    #[test]
    fn twin_a_population_that_dies_out_stays_empty() {
        let case = &small_cases()[0];
        let got = small::run_twin(case).unwrap();
        let n = counts(&got);
        assert_eq!(n[STEPS], 0, "{n:?}");
        let at = n.iter().position(|&n| n == 0).unwrap();
        assert!(at > 0 && at < STEPS, "{n:?}");
        assert!(n[at..].iter().all(|&n| n == 0), "{n:?}");
    }

    #[test]
    fn twin_a_population_at_the_capacity_is_cut_by_slot() {
        let case = &small_cases()[2];
        let want = reference_trace(small::SIDE, case);
        let got = small::run_twin(case).unwrap();
        verify(case.name, &want, &got).unwrap();
        let n = counts(&got);
        assert!(n.iter().skip(1).any(|&n| n == case.cap), "{n:?}");
    }

    #[test]
    fn twin_more_walkers_than_a_workgroup() {
        let case = &small_cases()[3];
        assert!(case.start > 256);
        let want = reference_trace(small::SIDE, case);
        let got = small::run_twin(case).unwrap();
        verify(case.name, &want, &got).unwrap();
        assert!(got.iter().any(|state| state.pos.len() > 256));
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_matches_the_host_on_the_small_grid() {
        let handle = gpu_handle();
        for case in small_cases() {
            let want = reference_trace(small::SIDE, &case);
            let got = small::run_gpu(&handle, &case).unwrap();
            verify(case.name, &want, &got).unwrap();
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
            verify(case.name, &want, &got).unwrap();
            println!("gpu on {}, {}: equal on every step", info.name, case.name);
        }
    }
}
