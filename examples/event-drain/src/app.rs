//! A bounded GPU-to-host event queue that the host drains late, without blocking (m11.events).
//!
//! A simulation on the GPU produces rare events the host must react to (here: a walker stepped onto a cell whose index
//! is a multiple of 64). Reading whole buffers back every step is costly and a blocking read stalls the frame, so the
//! kernel appends the events to a bounded queue, and the host reads that queue some steps later, when the GPU is done
//! with it.
//!
//! # The queue
//!
//! Two buffers per queue: `counters`, two words, word 0 the head (how many events were produced this step, stored or
//! not) and word 1 the overflow (how many did not fit); and `events`, `cap` events of three words each: `(step, walker,
//! cell)`. They are two buffers because a buffer that takes an atomic operation in a kernel refuses plain reads and
//! writes there (GUST-E0607).
//!
//! - `clear` (one invocation) zeroes the counters of the queue a step is about to write. The stale events
//!   stay in place; only `min(head, cap)` of them are meaningful.
//! - `walk` (one invocation per walker) moves a walker one cell in a hash-drawn direction on a periodic `SIDE x SIDE`
//!   grid. A walker that lands on a cell whose index is a multiple of 64 takes a slot with
//!   `let slot = atomic_add(&mut counters[0], 1)`; with `slot < cap` it writes its event there, else it counts the event
//!   with `atomic_add(&mut counters[1], 1)`. So `head` is the number of events, `overflow` is `head - cap` when positive,
//!   and which walker gets which slot is unspecified (the schedule decides): the host compares events as a set, and
//!   when the queue overflowed it checks that the stored events are distinct members of the host's set.
//!
//! # The host loop
//!
//! `QUEUES = LAG + 1` queues form a ring and step `s` writes queue `s % QUEUES`. Per frame `f` (`1 <= f`) the host
//!
//! 1. submits step `f` (the `clear` and the `walk`, with `submit`, no wait), while `f <= STEPS`, and starts the
//!    read-back of its queue at once, `GpuVec::to_vec_async` for the counters and the events: the first poll of each
//!    submits the copy into a staging buffer (behind the step's kernels in the queue) and requests the mapping;
//! 2. looks at every step `s` that is not drained yet and has `f >= s + LAG`: when its job reports `Complete` from one
//!    non-blocking `status` poll, it polls the read-back once with a no-op waker. A poll polls the device once and
//!    never waits (on native nothing else runs the mapping callback);
//! 3. goes on to the next frame. The loop never waits on the GPU (the first submit of a kernel may compile its Slang).
//!
//! A drain is *late* when it completes after frame `s + LAG`; its lateness is the number of steps after that one.
//! After frame `LAST_FRAME = STEPS + LAG` the loop spins (a yield each) until every queue is drained, at most
//! `MAX_FRAMES`; drains that complete in those spins are counted apart, not as late, and a step still undrained at
//! the end is drained at shutdown, with a blocking `wait` on its job, and counted apart too.
//!
//! **Ring reuse.** Step `s` reuses the queue of step `s - QUEUES`. Its copies were submitted right behind its kernels,
//! before the new `clear`, and the staging buffers keep the data, so a late drain never sees newer events.
//!
//! The uniform holding the parameters of a step is written with `write_uniform` before each submission; that write
//! is ordered after every earlier submission and before the next, so the kernels already submitted keep their value.
//!
//! The host reference is written plainly from the rules, with a table of directions, not from the kernels. The hash is
//! the "lowbias32" finaliser of `examples/hash-rng`, written again here in the module and on the host.

#[cfg(any(test, gust_dual))]
use gust::cpu::CpuDevice;
use gust::{Independent, gpu_module};
use gust_wgpu::{DeviceHandle, GpuVec, HeadlessDevice, WgpuJob, default_device};
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll, Waker},
};

/// Steps of every run.
pub const STEPS: usize = 24;
/// Steps between the submission of a step and the first look at its queue (`L`, at least 1).
pub const LAG: usize = 2;
/// Queues in the ring: the `LAG` steps in flight and the one being written.
pub const QUEUES: usize = LAG + 1;
/// The last frame that counts for lateness: the frame of the last step plus the lag.
const LAST_FRAME: usize = STEPS + LAG;
/// The most frames (spins included) the loop runs before it falls back to waiting at shutdown.
const MAX_FRAMES: usize = 200_000;
/// The seed of the walkers' starting cells.
const START_SEED: u32 = 0x1234_5678;
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

// The ring must hold the steps in flight and the one being written.
const _: () = assert!(LAG >= 1 && QUEUES > LAG);

/// Cells per side of the grid: a power of two.
pub const SIDE: usize = 64;

// ---- Host side: inputs and the reference, plain Rust, not the `#[gpu_module]` bodies. ----

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

/// One run: how many walkers and how many events a queue holds.
#[derive(Clone, Debug)]
pub struct Case {
    pub name: &'static str,
    pub walkers: usize,
    /// Events a queue stores; the rest are counted in the overflow.
    pub cap: usize,
    pub seed: u32,
}

/// One event: the step, the walker and the cell it stepped onto.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Event {
    pub walker: u32,
    pub step: u32,
    pub cell: u32,
}

/// What the host reads of one queue: the counters and the stored events (`min(head, cap)` of them, in slot order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Drained {
    pub head: u32,
    pub overflow: u32,
    pub events: Vec<Event>,
}

/// The starting cell of every walker.
fn make_start(case: &Case) -> Vec<u32> {
    let cells = (SIDE * SIDE) as u32;
    (0..case.walkers as u32)
        .map(|w| hash(START_SEED, w) % cells)
        .collect()
}

/// The cell one step from `cell` in heading `heading`, wrapping at the edges.
fn move_cell(cell: u32, heading: u32) -> u32 {
    let n = SIDE as i64;
    let (dx, dy) = DIRS[heading as usize];
    let x = i64::from(cell) % n + dx;
    let y = i64::from(cell) / n + dy;
    (y.rem_euclid(n) * n + x.rem_euclid(n)) as u32
}

/// Whether a walker that lands on `cell` raises an event.
fn raises_event(cell: u32) -> bool {
    cell.is_multiple_of(64)
}

/// The host reference: per step (index `step - 1`) the events in walker order, and the final cells.
pub fn reference(case: &Case) -> (Vec<Vec<Event>>, Vec<u32>) {
    let mut cells = make_start(case);
    let mut steps = Vec::new();
    for step in 1..=STEPS as u32 {
        let mut events = Vec::new();
        for (w, cell) in cells.iter_mut().enumerate() {
            let r = hash(hash(case.seed, step), w as u32);
            *cell = move_cell(*cell, r & 7);
            if raises_event(*cell) {
                events.push(Event {
                    walker: w as u32,
                    step,
                    cell: *cell,
                });
            }
        }
        steps.push(events);
    }
    (steps, cells)
}

/// Compares one drained queue with the host's events of that step (`want`, in walker order): the head is the number
/// of events; with room for all of them the stored events are exactly the host's, as a set, and the overflow is 0;
/// else the overflow is `events - cap` and the stored events are `cap` distinct members of the host's set.
pub fn verify_step(
    name: &str,
    step: usize,
    cap: usize,
    want: &[Event],
    got: &Drained,
) -> Result<(), String> {
    let who = format!("{name}: step {step}");
    if got.head as usize != want.len() {
        return Err(format!(
            "{who}: head {}, host {} events",
            got.head,
            want.len()
        ));
    }
    let stored = want.len().min(cap);
    if got.events.len() != stored {
        return Err(format!(
            "{who}: {} stored events, expected {stored}",
            got.events.len()
        ));
    }
    let overflow = (want.len() - stored) as u32;
    if got.overflow != overflow {
        return Err(format!(
            "{who}: overflow {}, expected {overflow}",
            got.overflow
        ));
    }
    let mut sorted = got.events.clone();
    sorted.sort_unstable();
    if let Some(pair) = sorted.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(format!("{who}: event {:?} stored twice", pair[0]));
    }
    if overflow == 0 {
        if sorted != want {
            return Err(format!("{who}: stored {sorted:?}, host {want:?}"));
        }
    } else if let Some(event) = sorted.iter().find(|e| !want.contains(e)) {
        return Err(format!(
            "{who}: stored event {event:?} is not one of the host's"
        ));
    }
    Ok(())
}

/// Compares a whole run with the reference: every step's queue, then the final cells.
pub fn verify(case: &Case, want: &(Vec<Vec<Event>>, Vec<u32>), got: &Run) -> Result<(), String> {
    if got.steps.len() != want.0.len() {
        return Err(format!(
            "{}: {} queues for {} steps",
            case.name,
            got.steps.len(),
            want.0.len()
        ));
    }
    for (i, (want, got)) in want.0.iter().zip(&got.steps).enumerate() {
        verify_step(case.name, i + 1, case.cap, want, got)?;
    }
    match got.cells.iter().zip(&want.1).position(|(a, b)| a != b) {
        Some(w) => Err(format!(
            "{}: walker {w} ends on cell {}, host {}",
            case.name, got.cells[w], want.1[w]
        )),
        None => Ok(()),
    }
}

/// A run: the drained queue of every step and the final cells of the walkers.
#[derive(Clone, Debug)]
pub struct Run {
    pub steps: Vec<Drained>,
    pub cells: Vec<u32>,
}

/// The cases: many events and room for all, many events and little room, and few walkers with quiet steps.
pub fn cases() -> Vec<Case> {
    let case = |name, walkers, cap| Case {
        name,
        walkers,
        cap,
        seed: 0xdead_beef,
    };
    vec![
        case("room for all", 1024, 256),
        case("overflowing", 1024, 4),
        case("quiet", 7, 8),
    ]
}

/// The two buffers of a queue on the device: the counters (word 0 the head, word 1 the overflow) and `cap` events of
/// three words.
struct Queue {
    counters: GpuVec<u32>,
    events: GpuVec<u32>,
}

impl Queue {
    fn new(handle: &DeviceHandle, cap: usize) -> Result<Self, gust_wgpu::Error> {
        Ok(Self {
            counters: GpuVec::zeroed_on(handle, 2)?,
            events: GpuVec::zeroed_on(handle, 3 * cap as u32)?,
        })
    }
}

/// The counters and the events of a queue, read back.
fn decode(cap: usize, counters: &[u32], events: &[u32]) -> Result<Drained, String> {
    if counters.len() != 2 || events.len() != 3 * cap {
        return Err(format!(
            "a queue of {} counters and {} event words, expected 2 and {}",
            counters.len(),
            events.len(),
            3 * cap
        ));
    }
    let stored = (counters[0] as usize).min(cap);
    let events = events
        .as_chunks::<3>()
        .0
        .iter()
        .take(stored)
        .map(|words| Event {
            step: words[0],
            walker: words[1],
            cell: words[2],
        })
        .collect();
    Ok(Drained {
        head: counters[0],
        overflow: counters[1],
        events,
    })
}

#[gpu_module]
pub mod k {
    /// Cells per side: a power of two.
    pub const SIDE: u32 = 64;
    /// Mask of a coordinate: `SIDE - 1`.
    pub const MASK: u32 = SIDE - 1;

    /// The parameters of a step, four `u32` (16 bytes, no gap): the seed and the step of the hash, the events a
    /// queue stores, and a pad field.
    pub struct Params {
        pub seed: u32,
        pub step: u32,
        pub cap: u32,
        pub pad0: u32,
    }

    /// A multiply-xorshift finaliser. Every shift amount is a literal below 32 (D39).
    ///
    /// The constants and shifts are those of "lowbias32" from Chris Wellons' hash-prospector
    /// (https://github.com/skeeto/hash-prospector; public domain, Unlicense), as in `examples/hash-rng`.
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

    /// The cell one step from `(x, y)` in heading `d`, wrapping.
    pub fn ahead(x: u32, y: u32, d: u32) -> u32 {
        let n = SIDE as i32;
        let nx = ((x as i32) + dir_x(d)).rem_euclid(n);
        let ny = ((y as i32) + dir_y(d)).rem_euclid(n);
        (ny as u32) * SIDE + (nx as u32)
    }

    /// Zeroes the head (word 0) and the overflow (word 1) of the queue the next step writes.
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn clear(id: SV_DispatchThreadID, mut counters: RWStructuredBuffer<uint>) {
        if id.x == 0u32 {
            counters[0u32] = 0u32;
            counters[1u32] = 0u32;
        }
    }

    /// One walker: moves one cell in a hash-drawn heading and, on a cell that is a multiple of 64, appends the event
    /// `(step, walker, cell)` to the queue: the slot is the old head; a slot past `cap` counts in the overflow.
    /// The counters are only touched atomically here (a buffer with an atomic operation refuses plain access), which
    /// is why the events are a buffer of their own.
    #[kernel(workgroup_size(64, 1, 1))]
    pub fn walk(
        id: SV_DispatchThreadID,
        p: Uniform<Params>,
        mut pos: RWStructuredBuffer<uint>,
        mut counters: RWStructuredBuffer<uint>,
        mut events: RWStructuredBuffer<uint>,
    ) {
        let i = id.x;
        if i < pos.len() {
            let r = hash(hash(p.seed, p.step), i);
            let c = pos[i];
            let nc = ahead(c & MASK, c / SIDE, r & 7u32);
            pos[i] = nc;
            if (nc & 63u32) == 0u32 {
                let slot = atomic_add(&mut counters[0u32], 1u32);
                if slot < p.cap {
                    let at = 3u32 * slot;
                    events[at] = p.step;
                    events[at + 1u32] = i;
                    events[at + 2u32] = nc;
                }
                if slot >= p.cap {
                    atomic_add(&mut counters[1u32], 1u32);
                }
            }
        }
    }
}

/// The parameter struct of step `step` (1-based).
pub fn params(case: &Case, step: usize) -> k::Params {
    k::Params {
        seed: case.seed,
        step: step as u32,
        cap: case.cap as u32,
        pad0: 0,
    }
}

/// How the drains went. A frame is one iteration of the host loop; frame `f` submits step `f`, so lateness counts
/// steps. Frames after `LAST_FRAME` are only spins waiting for the device and are not counted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Lateness {
    /// Steps drained inside the loop, without a blocking call.
    pub in_loop: usize,
    /// Of those, the drains that completed later than frame `step + LAG`, up to `LAST_FRAME`.
    pub late: usize,
    /// The steps those late drains were late by, summed, and the largest.
    pub late_steps: usize,
    pub max_late: usize,
    /// Of the drains in the loop, those that completed only after `LAST_FRAME`, in the spins that follow it.
    pub after_last_frame: usize,
    /// Steps still undrained when the loop ended, drained with a blocking wait.
    pub at_shutdown: usize,
}

impl Lateness {
    /// Counts a drain of `step` that completed in `frame`.
    fn record(&mut self, step: usize, frame: usize) {
        self.in_loop += 1;
        if frame > LAST_FRAME {
            self.after_last_frame += 1;
            return;
        }
        let late = frame.saturating_sub(step + LAG);
        if late > 0 {
            self.late += 1;
            self.late_steps += late;
            self.max_late = self.max_late.max(late);
        }
    }
}

impl std::fmt::Display for Lateness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} drains in the loop, {} of them late (by {} steps in all, at most {}), {} only after the last frame, {} at shutdown",
            self.in_loop,
            self.late,
            self.late_steps,
            self.max_late,
            self.after_last_frame,
            self.at_shutdown
        )
    }
}

type Readback<'a> = Pin<Box<dyn Future<Output = Result<Vec<u32>, gust_wgpu::Error>> + 'a>>;

/// One read-back of a buffer: the future, until it resolves, then its words.
enum Part<'a> {
    Waiting(Readback<'a>),
    Done(Vec<u32>),
}

impl<'a> Part<'a> {
    /// Starts the read-back of `buffer`: the first poll submits the copy, so this is where the copy joins the queue.
    fn start(buffer: &'a GpuVec<u32>) -> Result<Self, gust_wgpu::Error> {
        let mut readback: Readback<'a> = Box::pin(buffer.to_vec_async());
        Self::poll(&mut readback).map(|done| done.map_or(Self::Waiting(readback), Self::Done))
    }

    /// One poll with a no-op waker: it polls the device once and never waits.
    fn poll(readback: &mut Readback<'_>) -> Result<Option<Vec<u32>>, gust_wgpu::Error> {
        match readback
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
        {
            Poll::Ready(words) => words.map(Some),
            Poll::Pending => Ok(None),
        }
    }

    /// Polls once more unless done.
    fn advance(&mut self) -> Result<(), gust_wgpu::Error> {
        if let Self::Waiting(readback) = self
            && let Some(words) = Self::poll(readback)?
        {
            *self = Self::Done(words);
        }
        Ok(())
    }
}

/// The counters and the event words of a queue, read back.
type Words<'a> = (&'a [u32], &'a [u32]);

/// The read-back of one queue: the counters and the events, started together so both copies sit in the queue right
/// after the step's kernels.
struct Read<'a> {
    counters: Part<'a>,
    events: Part<'a>,
}

impl<'a> Read<'a> {
    fn start(queue: &'a Queue) -> Result<Self, gust_wgpu::Error> {
        Ok(Self {
            counters: Part::start(&queue.counters)?,
            events: Part::start(&queue.events)?,
        })
    }

    /// Polls what is not done; the words of both once they are.
    fn advance(&mut self) -> Result<Option<Words<'_>>, gust_wgpu::Error> {
        self.counters.advance()?;
        self.events.advance()?;
        Ok(match (&self.counters, &self.events) {
            (Part::Done(counters), Part::Done(events)) => Some((counters, events)),
            _ => None,
        })
    }
}

/// A submitted step that is not drained yet, with its read-back already started.
struct Pending<'a> {
    step: usize,
    job: WgpuJob<'a>,
    read: Read<'a>,
}

impl Pending<'_> {
    /// Polls the read-back once and decodes the queue when it is complete.
    fn poll(&mut self, cap: usize) -> Result<Option<Drained>, Box<dyn std::error::Error>> {
        Ok(match self.read.advance()? {
            Some((counters, events)) => Some(decode(cap, counters, events)?),
            None => None,
        })
    }
}

/// `STEPS` steps on the device, draining the queues by the loop of the module comment. Returns the run and how the
/// drains went.
pub fn run_gpu(
    handle: &DeviceHandle,
    case: &Case,
) -> Result<(Run, Lateness), Box<dyn std::error::Error>> {
    let device: &HeadlessDevice = handle;
    let start = make_start(case);
    let pos = GpuVec::from_slice_on(handle, &start)?;
    let queues = (0..QUEUES)
        .map(|_| Queue::new(handle, case.cap))
        .collect::<Result<Vec<_>, _>>()?;
    let uniform = device.create_uniform("event-drain params", &params(case, 1))?;
    let mut pending: Vec<Pending<'_>> = Vec::new();
    let mut drained: Vec<Option<Drained>> = vec![None; STEPS];
    let mut lateness = Lateness::default();
    let mut frame = 0;
    while (frame < STEPS || !pending.is_empty()) && frame < MAX_FRAMES {
        frame += 1;
        if frame <= STEPS {
            let step = frame;
            device.write_uniform(&uniform, &params(case, step))?;
            let queue = &queues[step % QUEUES];
            drop(k::clear::submit(device, 1, Independent(&queue.counters))?);
            let job = k::walk::submit(
                device,
                case.walkers as u32,
                &uniform,
                Independent(&pos),
                Independent(&queue.counters),
                Independent(&queue.events),
            )?;
            // The copies go into the queue right behind the step's kernels, so the ring reuse of a later step
            // cannot overwrite what this step wrote before the host has it (the staging buffers keep it).
            let read = Read::start(queue)?;
            pending.push(Pending { step, job, read });
        }
        let mut i = 0;
        while i < pending.len() {
            let p = &mut pending[i];
            if frame >= p.step + LAG
                && p.job.status()?.is_complete()
                && let Some(done) = p.poll(case.cap)?
            {
                drained[p.step - 1] = Some(done);
                lateness.record(p.step, frame);
                pending.remove(i);
                continue;
            }
            i += 1;
        }
        if frame >= LAST_FRAME && !pending.is_empty() {
            std::thread::yield_now();
        }
    }
    // Shutdown: the only blocking calls.
    for mut p in pending {
        p.job.wait()?;
        lateness.at_shutdown += 1;
        let done = loop {
            if let Some(done) = p.poll(case.cap)? {
                break done;
            }
            std::thread::yield_now();
        };
        drained[p.step - 1] = Some(done);
    }
    let steps = drained
        .into_iter()
        .enumerate()
        .map(|(i, d)| d.ok_or_else(|| format!("step {} was never drained", i + 1)))
        .collect::<Result<Vec<_>, _>>()?;
    let cells = pos.to_vec()?;
    Ok((Run { steps, cells }, lateness))
}

/// The same steps on the CPU twin (the kernels' own Rust bodies, one invocation at a time in ascending order).
#[cfg(any(test, gust_dual))]
pub fn run_twin(case: &Case) -> Result<Run, Box<dyn std::error::Error>> {
    let device = CpuDevice::new();
    let pos = device.buffer(&make_start(case));
    let counters = device.buffer(&[0u32; 2]);
    let events = device.buffer(&vec![0u32; 3 * case.cap]);
    let uniform = device.uniform(params(case, 1))?;
    let mut steps = Vec::new();
    for step in 1..=STEPS {
        uniform.write(params(case, step))?;
        k::clear::dispatch(&device, 1, Independent(&counters))?;
        k::walk::dispatch(
            &device,
            case.walkers as u32,
            &uniform,
            Independent(&pos),
            Independent(&counters),
            Independent(&events),
        )?;
        steps.push(decode(case.cap, &counters.read(), &events.read())?);
    }
    Ok(Run {
        steps,
        cells: pos.read().to_vec(),
    })
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let handle = default_device()?;
    let info = handle.adapter_info();
    println!("adapter: {} backend: {:?}", info.name, info.backend);
    println!(
        "grid {0} x {0}, {STEPS} steps, {QUEUES} queues, drained {LAG} steps late, {1} cases",
        SIDE,
        cases().len()
    );
    for case in cases() {
        let want = reference(&case);
        let (got, lateness) = run_gpu(&handle, &case)?;
        verify(&case, &want, &got)?;
        let events: usize = want.0.iter().map(Vec::len).sum();
        let dropped: usize = got.steps.iter().map(|d| d.overflow as usize).sum();
        println!(
            "gpu, {}: {} walkers, {events} events, {dropped} dropped by the cap {}, equal to the host on every step; {lateness}",
            case.name, case.walkers, case.cap
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

    fn case_named(name: &str) -> Case {
        cases().into_iter().find(|c| c.name == name).unwrap()
    }

    #[test]
    fn the_host_reads_the_modules_constants() {
        assert_eq!(k::SIDE as usize, SIDE);
        assert!(SIDE.is_power_of_two());
        assert_eq!(k::MASK as usize, SIDE - 1);
        // The struct is 16 bytes, a multiple of 16, without a gap.
        assert_eq!(std::mem::size_of::<k::Params>(), 16);
    }

    #[test]
    fn the_cases_exercise_the_queue() {
        let ample = reference(&case_named("room for all")).0;
        let tight = reference(&case_named("overflowing")).0;
        let quiet = reference(&case_named("quiet")).0;
        let cap = |name| case_named(name).cap;
        // Room for all: events every step, never more than the capacity.
        assert!(
            ample
                .iter()
                .all(|s| !s.is_empty() && s.len() <= cap("room for all"))
        );
        // The same events overflow a queue of 4 in most steps.
        assert!(
            tight
                .iter()
                .filter(|s| s.len() > cap("overflowing"))
                .count()
                >= STEPS / 2
        );
        // Few walkers: empty steps, and at least one event somewhere.
        assert!(quiet.iter().any(Vec::is_empty));
        assert!(quiet.iter().any(|s| !s.is_empty()));
        // Every event is on a multiple of 64 and names its own step.
        for (i, events) in ample.iter().enumerate() {
            assert!(
                events
                    .iter()
                    .all(|e| e.cell % 64 == 0 && e.step as usize == i + 1)
            );
            assert!(events.windows(2).all(|w| w[0].walker < w[1].walker));
        }
    }

    #[test]
    fn the_kernel_hash_and_move_are_the_host_ones() {
        for (seed, index) in [
            (0, 0),
            (1, 2),
            (2, 1),
            (0xdead_beef, 4095),
            (u32::MAX, u32::MAX),
        ] {
            assert_eq!(k::hash(seed, index), hash(seed, index));
        }
        assert_ne!(hash(1, 2), hash(2, 1));
        for cell in [0u32, 1, 63, 64, 65, 4095, 2000] {
            for heading in 0..8 {
                assert_eq!(
                    k::ahead(cell & k::MASK, cell / k::SIDE, heading),
                    move_cell(cell, heading)
                );
            }
        }
    }

    #[test]
    fn a_wrong_queue_is_reported() {
        let want = vec![
            Event {
                walker: 3,
                step: 5,
                cell: 64,
            },
            Event {
                walker: 9,
                step: 5,
                cell: 128,
            },
            Event {
                walker: 20,
                step: 5,
                cell: 0,
            },
        ];
        let good = Drained {
            head: 3,
            overflow: 0,
            events: vec![want[2], want[0], want[1]],
        };
        assert!(verify_step("t", 5, 8, &want, &good).is_ok());
        let fail = |got: &Drained, cap, text: &str| {
            let message = verify_step("t", 5, cap, &want, got).unwrap_err();
            assert!(message.contains(text), "{message}");
        };
        fail(
            &Drained {
                head: 2,
                ..good.clone()
            },
            8,
            "head 2, host 3 events",
        );
        fail(
            &Drained {
                overflow: 1,
                ..good.clone()
            },
            8,
            "overflow 1, expected 0",
        );
        let mut wrong = good.clone();
        wrong.events[1].cell = 192;
        fail(&wrong, 8, "stored [");
        fail(
            &Drained {
                events: vec![want[0]; 3],
                ..good.clone()
            },
            8,
            "stored twice",
        );
        fail(
            &Drained {
                events: good.events[..2].to_vec(),
                ..good.clone()
            },
            8,
            "2 stored events, expected 3",
        );
        // A queue of 2 that overflowed by one: two distinct members of the host's set are fine, a foreign one is not.
        let tight = Drained {
            head: 3,
            overflow: 1,
            events: vec![want[1], want[2]],
        };
        assert!(verify_step("t", 5, 2, &want, &tight).is_ok());
        fail(
            &Drained {
                overflow: 0,
                ..tight.clone()
            },
            2,
            "overflow 0, expected 1",
        );
        let foreign = Event {
            walker: 4,
            ..want[0]
        };
        fail(
            &Drained {
                events: vec![foreign, want[2]],
                ..tight.clone()
            },
            2,
            "not one of the host's",
        );
        fail(
            &Drained {
                events: vec![want[2], want[2]],
                ..tight
            },
            2,
            "stored twice",
        );
    }

    #[test]
    fn a_wrong_final_cell_is_reported() {
        let case = case_named("quiet");
        let want = reference(&case);
        let mut got = Run {
            steps: want
                .0
                .iter()
                .map(|events| Drained {
                    head: events.len() as u32,
                    overflow: 0,
                    events: events.clone(),
                })
                .collect(),
            cells: want.1.clone(),
        };
        assert!(verify(&case, &want, &got).is_ok());
        got.cells[3] += 1;
        let message = verify(&case, &want, &got).unwrap_err();
        assert!(message.contains("walker 3 ends on cell"), "{message}");
        got.steps.pop();
        assert!(
            verify(&case, &want, &got)
                .unwrap_err()
                .contains("queues for")
        );
    }

    #[test]
    fn the_lateness_counts_steps_after_step_plus_lag() {
        let mut lateness = Lateness::default();
        lateness.record(1, 1 + LAG);
        lateness.record(2, 2 + LAG + 3);
        lateness.record(3, 3 + LAG + 1);
        // A drain in the spins after the last frame is not counted as late.
        lateness.record(STEPS, LAST_FRAME + 5);
        assert_eq!(
            lateness,
            Lateness {
                in_loop: 4,
                late: 2,
                late_steps: 4,
                max_late: 3,
                after_last_frame: 1,
                ..Lateness::default()
            }
        );
    }

    #[test]
    fn twin_matches_the_host() {
        for case in cases() {
            let want = reference(&case);
            let got = run_twin(&case).unwrap();
            verify(&case, &want, &got).unwrap();
        }
    }

    #[test]
    fn twin_counts_the_events_that_do_not_fit() {
        let case = case_named("overflowing");
        let want = reference(&case).0;
        let got = run_twin(&case).unwrap();
        let mut overflowed = 0;
        for (events, drained) in want.iter().zip(&got.steps) {
            assert_eq!(drained.head as usize, events.len());
            assert_eq!(drained.events.len(), events.len().min(case.cap));
            assert_eq!(
                drained.overflow as usize,
                events.len().saturating_sub(case.cap)
            );
            overflowed += usize::from(drained.overflow > 0);
        }
        assert!(overflowed >= STEPS / 2, "{overflowed}");
    }

    #[test]
    fn twin_a_quiet_step_leaves_an_empty_queue() {
        let case = case_named("quiet");
        let got = run_twin(&case).unwrap();
        let quiet = got.steps.iter().filter(|d| d.head == 0).collect::<Vec<_>>();
        assert!(!quiet.is_empty());
        assert!(quiet.iter().all(|d| d.overflow == 0 && d.events.is_empty()));
    }

    #[test]
    #[ignore = "example validation runs only in full verification"]
    fn gpu_matches_the_host() {
        let handle = gpu_handle();
        let info = handle.adapter_info();
        println!("adapter: {} backend: {:?}", info.name, info.backend);
        for case in cases() {
            let want = reference(&case);
            let (got, lateness) = run_gpu(&handle, &case).unwrap();
            verify(&case, &want, &got).unwrap();
            // Every step was drained inside the loop: a job that never reported complete would land at shutdown.
            assert_eq!(lateness.at_shutdown, 0, "{}: {lateness}", case.name);
            assert_eq!(lateness.in_loop, STEPS, "{}: {lateness}", case.name);
            println!(
                "gpu on {}, {}: equal on every step; {lateness}",
                info.name, case.name
            );
        }
    }
}
