//! Parallel CPU execution of the generated map kernels (M7, D119, slice m7.c). Macro support.
//!
//! Under `GUST_PROFILE=cpu` the map generator (`#[gpu]` on a free function) emits a CPU-only chunk body,
//! `__gust_map_cpu_chunk`: the function over the element indexes `lo..hi`, reading every input as a whole-length slice
//! at the global index and writing only `chunk[i - lo]`, the invocation's own element. [`run_map_chunked`] runs that
//! body over disjoint pieces of the output (`chunks_mut` under `std::thread::scope`), so the result is the one a
//! single thread gives, element for element. Nothing else in the `cpu` target is parallel: `par_for!` bodies and
//! user-written kernels stay sequential (the generator owns `i` and the write set, which is what makes this safe).
//!
//! * Thread count: [`THREADS_VARIABLE`] (`GUST_CPU_THREADS`, a positive integer; anything else is
//!   [`Error::InvalidThreads`], never a silent fallback) or else `std::thread::available_parallelism()`, then bounded
//!   by the input: `min(configured, max(1, n / ELEMENTS_PER_THREAD))`, because a spawned thread costs about 0.1 ms
//!   and a small call must not pay for threads it cannot use.
//! * Panics: each chunk runs under `catch_unwind` and records the first element index that panicked; after the join
//!   the panic of the lowest index is re-raised with its own payload, which is the panic a single thread meets first.
//!   Nothing is returned from a call that panicked.
//!
//! Safe Rust and `std` only.

use core::cell::{Cell, Ref};
use std::any::Any;
use std::ffi::OsStr;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::OnceLock;

use super::{Access, CpuBuffer, CpuDevice, Error, LengthRule, Resource, read};
use crate::GpuPod;

/// The environment variable that sets the thread count of a parallel map.
pub const THREADS_VARIABLE: &str = "GUST_CPU_THREADS";

/// How many elements one thread is worth (the factor `F` of D119): a call of `n` elements uses at most
/// `max(1, n / ELEMENTS_PER_THREAD)` threads. Chosen by measurement (plans/m7, `time-m7c.sh`); one constant, so a
/// better number is a one-line change.
pub const ELEMENTS_PER_THREAD: usize = 1 << 18;

/// The thread count that `value` (the content of `GUST_CPU_THREADS`, `None` when it is not set) asks for, with
/// `available` (the machine's parallelism) when it is not set.
///
/// A set value must be a positive decimal integer: digits only, no sign or spaces, not zero, not out of range. Anything
/// else is [`Error::InvalidThreads`] naming the value.
pub fn threads_from(value: Option<&OsStr>, available: usize) -> Result<usize, Error> {
    let Some(value) = value else {
        return Ok(available.max(1));
    };
    let invalid = || Error::InvalidThreads {
        value: value.to_string_lossy().into_owned(),
    };
    let text = value.to_str().ok_or_else(invalid)?;
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid());
    }
    match text.parse::<usize>() {
        Ok(threads) if threads > 0 => Ok(threads),
        _ => Err(invalid()),
    }
}

/// The threads a call of `elements` elements uses when `configured` threads are allowed:
/// `min(configured, max(1, elements / ELEMENTS_PER_THREAD))`, and at least 1.
pub fn bounded_threads(configured: usize, elements: usize) -> usize {
    configured
        .min((elements / ELEMENTS_PER_THREAD).max(1))
        .max(1)
}

/// The machine's parallelism, asked once. `wasm32` has no threads to spawn, so it is 1 there.
fn available() -> usize {
    static AVAILABLE: OnceLock<usize> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        if cfg!(target_arch = "wasm32") {
            1
        } else {
            std::thread::available_parallelism().map_or(1, usize::from)
        }
    })
}

/// The thread count of one map call over `elements` elements: [`threads_from`] with the process environment, then
/// [`bounded_threads`]. An invalid `GUST_CPU_THREADS` is an error even for a call that would run one thread. On
/// `wasm32` it is always 1.
pub fn map_threads(elements: u32) -> Result<usize, Error> {
    let configured = threads_from(std::env::var_os(THREADS_VARIABLE).as_deref(), available())?;
    if cfg!(target_arch = "wasm32") {
        return Ok(1);
    }
    Ok(bounded_threads(configured, elements as usize))
}

/// What a chunk that panicked leaves behind: the element it was evaluating and the panic's own payload.
struct Failure {
    index: u32,
    payload: Box<dyn Any + Send>,
}

/// `value` as an element index. The callers have checked that the output is at most `u32::MAX` long.
fn index(value: usize) -> u32 {
    u32::try_from(value).expect("an element index of an output of at most u32::MAX elements")
}

/// Runs `chunk` over `lo..lo + piece.len()` under `catch_unwind` and reports where it stopped if it panicked.
fn run_guarded<R, C>(chunk: &C, lo: usize, piece: &mut [R]) -> Option<Failure>
where
    C: Fn(u32, u32, &mut [R], &Cell<u32>),
{
    let (first, end) = (index(lo), index(lo + piece.len()));
    let current = Cell::new(first);
    catch_unwind(AssertUnwindSafe(|| chunk(first, end, piece, &current)))
        .err()
        .map(|payload| Failure {
            index: current.get(),
            payload,
        })
}

/// Runs the chunk body `chunk(lo, hi, piece, current)` over all of `out`, split into at most `threads` pieces that
/// are processed at the same time. `piece` is `&mut out[lo..hi]`; the body must write only `piece[i - lo]` for the
/// elements `i` in `lo..hi` and set `current` to `i` before it evaluates element `i` (the generated body does both),
/// so that a panic can be attributed to its element.
///
/// With one thread (or one element) the body runs on the calling thread over the whole output and a panic simply
/// unwinds. With more, the pieces are `ceil(n / threads)` long, the first runs on the calling thread and the others on
/// scoped threads, and after the join the panic of the lowest element index, if any, is re-raised with its own payload
/// (`resume_unwind`). The output is then partly written and must be dropped by the caller, which the generated map
/// does by unwinding.
///
/// # Panics
///
/// If the body panics (as above), and if `out` is longer than `u32::MAX` elements.
pub fn run_map_chunked<R, C>(out: &mut [R], threads: usize, chunk: C)
where
    R: Send,
    C: Fn(u32, u32, &mut [R], &Cell<u32>) + Sync,
{
    let len = out.len();
    if len == 0 {
        return;
    }
    let end = index(len);
    let threads = threads.clamp(1, len);
    if threads == 1 {
        chunk(0, end, out, &Cell::new(0));
        return;
    }
    let size = len.div_ceil(threads);
    let chunk = &chunk;
    let failures: Vec<Option<Failure>> = std::thread::scope(|scope| {
        let mut pieces = out.chunks_mut(size).enumerate();
        let first = pieces.next();
        let handles: Vec<_> = pieces
            .map(|(number, piece)| scope.spawn(move || run_guarded(chunk, number * size, piece)))
            .collect();
        let mut failures = Vec::with_capacity(handles.len() + 1);
        if let Some((_, piece)) = first {
            failures.push(run_guarded(chunk, 0, piece));
        }
        for handle in handles {
            failures.push(
                handle
                    .join()
                    .expect("a chunk catches its own panic, so its thread ends normally"),
            );
        }
        failures
    });
    if let Some(failure) = failures
        .into_iter()
        .flatten()
        .min_by_key(|failure| failure.index)
    {
        resume_unwind(failure.payload);
    }
}

/// [`run_map_chunked`] over the contents of a buffer that the caller owns, returned as a `Vec` when the body is done.
/// The generated map creates the output with `MapHost::zeroed` and hands it here, so the result needs no copy.
pub fn run_map_into_vec<R, C>(mut out: CpuBuffer<R>, threads: usize, chunk: C) -> Vec<R>
where
    R: GpuPod + Send,
    C: Fn(u32, u32, &mut [R], &Cell<u32>) + Sync,
{
    run_map_chunked(out.cells.get_mut(), threads, chunk);
    out.cells.into_inner()
}

/// Borrows the contents of an input of the map call for reading, after checking that it belongs to `device`
/// ([`Error::WrongDevice`]), has the element type `T` ([`Error::ElementType`]) and is not borrowed mutably outside
/// the call ([`Error::Borrowed`]). `parameter` names the map function's parameter in those errors.
pub fn input<'b, T, B>(
    device: &CpuDevice,
    resource: &'b B,
    parameter: &'static str,
) -> Result<Ref<'b, Vec<T>>, Error>
where
    T: GpuPod,
    B: Resource<Element = T, Device = CpuDevice>,
{
    let binding = resource.bind(Access::ReadOnly, LengthRule::Independent);
    if binding.owner != device.id {
        return Err(Error::WrongDevice { parameter });
    }
    read::<T>(binding, parameter)
}

/// Compile-time proof that the kernel `K` carries the generator's own-index stamp (`gust::map::OwnIndex`): the
/// chunked execution is defined for stamped kernels only, and the generated map names its kernel here.
pub fn require_own_index<K: crate::map::OwnIndex>() {}
