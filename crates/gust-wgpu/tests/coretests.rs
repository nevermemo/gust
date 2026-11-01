//! The scalar coretests transcription on the GPU (roadmap M5, slice m5.d2; Gpu tier).
//!
//! `tests/fixtures/coretests.rs` holds the five `#[gpu]` kernels of the cases that `docs/development/CORETESTS.md`
//! transcribes from the Rust core tests (tag 1.98.0). This file runs every kernel on the adapter and on its CPU twin
//! and compares each output slot with the expected literal of the table under the acceptance set of the row (M5 plan
//! R1-7): exact for integers and `bool` inside the domain, every `f32` within 2.5 ULP of the literal for float `/`,
//! and the two neighbours of the integer for `i32 as f32` above 2^24. The literal is the independent reference (D26);
//! the twin is asserted exact in every row and is never GPU evidence. The input arrays and the output lengths are the
//! constants of `crates/gust/tests/coretests_twin.rs`, element for element.
//!
//! Guards: the executed ids are recorded as the cases run and must equal the ids of the table (every id exactly once,
//! none unknown); the acceptance label of every table row equals the one the case uses. A failure names the case id,
//! the expected set, and the GPU and twin values. No adapter is skipped silently, and no set is loosened.

// CT-69 transcribes `u32::MIN <= 0` (uint_macros.rs:11), which is always true for `u32`;
// the kernel must still run it, so the including test allows the lint.
#![allow(clippy::absurd_extreme_comparisons)]

include!("../../../tests/fixtures/coretests.rs");

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use gust::Independent;
use gust::cpu::{CpuDevice, ulps};
use gust_wgpu::{BufferBinding, GpuBufferAccess, HeadlessDevice};

const TABLE: &str = include_str!("../../../docs/development/CORETESTS.md");

// ---- kernel inputs (the names are the variables of the kernels; equal to coretests_twin.rs) ----

const INTS_A: [i32; 13] = [
    0,
    0,
    1,
    0b1010,
    2,
    i32::MAX,
    i32::MIN,
    0x1010_1010,
    0b1100,
    0b1010,
    0b111,
    0b1110,
    0b11,
];
const INTS_N: [u32; 1] = [0];
const UINTS_A: [u32; 13] = [
    0,
    0,
    1,
    0b1010,
    2,
    u32::MAX,
    0,
    0b1100,
    0b1010,
    0b111,
    0b1110,
    0b1011,
    0,
];
const FLOATS_A: [f32; 6] = [0.0, 0.0, 1.0, 10.0, 2.0, -0.0];
const BOOLS_A: [u32; 3] = [0, 0, 1];
const CASTS_A: [i32; 8] = [
    16777214, 16777215, 16777216, 16777217, -16777214, -16777215, -16777216, -16777217,
];
const INTS_OUT: usize = 50;
const UINTS_OUT: usize = 34;
const FLOATS_OUT: usize = 14;
const BOOLS_OUT: usize = 7;

/// The largest ULP distance the float `/` set accepts: 2.5 ULP rounds down to 2 whole ULPs (R1-7).
const DIV_ULPS: u64 = 2;

// ---- literals, acceptance sets, cases ----

/// A literal of the table, in the type of the output slot. Floats compare with `==`, so the sign of a zero is ignored
/// (SEMANTICS.md section 4); a comparison slot stores 1 or 0 and is read as a `bool`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Lit {
    I(i32),
    U(u32),
    F(f32),
    B(bool),
}

impl Lit {
    fn render(self) -> String {
        match self {
            Self::I(value) => value.to_string(),
            Self::U(value) => value.to_string(),
            Self::F(value) => format!("{value:?}"),
            Self::B(value) => value.to_string(),
        }
    }
}

/// What the GPU may return for a row (R1-7). The twin is exact in every row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Accept {
    /// Bit equality for integers and `bool`; `==` for floats (WGSL allows any sign of a zero).
    Exact,
    /// Float `/`: every `f32` within 2.5 ULP of the literal, asserted as `gust::cpu::ulps(gpu, literal) <= 2`.
    Ulps25,
    /// An integer above 2^24 converted to `f32`: either neighbour of the integer (WGSL 15.7.6), and the same for the
    /// round trip back to `i32`.
    Neighbours,
}

impl Accept {
    /// The spelling of the "Acceptance" column of the table.
    fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Ulps25 => "GPU: 2.5 ULP set",
            Self::Neighbours => "GPU: two-neighbour set",
        }
    }
}

/// An output buffer of a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Stream {
    Ints,
    Uints,
    Floats,
    Bools,
    CastFloat,
    CastInt,
}

impl Stream {
    const ALL: [Self; 6] = [
        Self::Ints,
        Self::Uints,
        Self::Floats,
        Self::Bools,
        Self::CastFloat,
        Self::CastInt,
    ];

    /// The kernel that writes the buffer, as the table names it.
    fn kernel(self) -> &'static str {
        match self {
            Self::Ints => "ints",
            Self::Uints => "uints",
            Self::Floats => "floats",
            Self::Bools => "bools",
            Self::CastFloat | Self::CastInt => "casts",
        }
    }

    fn len(self) -> usize {
        match self {
            Self::Ints => INTS_OUT,
            Self::Uints => UINTS_OUT,
            Self::Floats => FLOATS_OUT,
            Self::Bools => BOOLS_OUT,
            Self::CastFloat | Self::CastInt => CASTS_A.len(),
        }
    }
}

/// Everything one executor (the adapter or the twin) wrote.
struct Outputs {
    ints: Vec<i32>,
    uints: Vec<u32>,
    floats: Vec<f32>,
    bools: Vec<u32>,
    cast_float: Vec<f32>,
    cast_int: Vec<i32>,
}

/// Dispatches one invocation of every kernel on the twin and reads the outputs back.
fn run_twins() -> Outputs {
    let cpu = CpuDevice::new();

    let a = cpu.buffer(&INTS_A);
    let n = cpu.buffer(&INTS_N);
    let out = cpu.buffer(&[i32::MIN; INTS_OUT]);
    coretests::ints::dispatch(&cpu, 1, Independent(&a), Independent(&n), Independent(&out))
        .expect("the twin of `ints` refused the call");
    let ints = out.read();

    let a = cpu.buffer(&UINTS_A);
    let out = cpu.buffer(&[u32::MAX; UINTS_OUT]);
    coretests::uints::dispatch(&cpu, 1, Independent(&a), Independent(&out))
        .expect("the twin of `uints` refused the call");
    let uints = out.read();

    let a = cpu.buffer(&FLOATS_A);
    let out = cpu.buffer(&[f32::NAN; FLOATS_OUT]);
    coretests::floats::dispatch(&cpu, 1, Independent(&a), Independent(&out))
        .expect("the twin of `floats` refused the call");
    let floats = out.read();

    let a = cpu.buffer(&BOOLS_A);
    let out = cpu.buffer(&[u32::MAX; BOOLS_OUT]);
    coretests::bools::dispatch(&cpu, 1, Independent(&a), Independent(&out))
        .expect("the twin of `bools` refused the call");
    let bools = out.read();

    let a = cpu.buffer(&CASTS_A);
    let float_out = cpu.buffer(&[f32::NAN; CASTS_A.len()]);
    let int_out = cpu.buffer(&[i32::MIN; CASTS_A.len()]);
    coretests::casts::dispatch(&cpu, CASTS_A.len() as u32, &a, &float_out, &int_out)
        .expect("the twin of `casts` refused the call");

    Outputs {
        ints,
        uints,
        floats,
        bools,
        cast_float: float_out.read(),
        cast_int: int_out.read(),
    }
}

/// Dispatches every kernel on the adapter and reads the outputs back.
fn run_gpu(device: &HeadlessDevice) -> Outputs {
    let ro = GpuBufferAccess::ReadOnly;
    let rw = GpuBufferAccess::ReadWrite;

    let a = device.create_typed_buffer("coretests ints a", &INTS_A, ro);
    let n = device.create_typed_buffer("coretests ints n", &INTS_N, ro);
    let out = device.create_typed_buffer("coretests ints out", &[i32::MIN; INTS_OUT], rw);
    device
        .dispatch_buffers(
            &coretests::ints::DESCRIPTOR,
            1,
            &[
                BufferBinding::read_only(&a).independent_length(),
                BufferBinding::read_only(&n).independent_length(),
                BufferBinding::read_write(&out).independent_length(),
            ],
        )
        .expect("the GPU refused `ints`");
    let ints = device.read_typed_buffer(&out).unwrap();

    let a = device.create_typed_buffer("coretests uints a", &UINTS_A, ro);
    let out = device.create_typed_buffer("coretests uints out", &[u32::MAX; UINTS_OUT], rw);
    device
        .dispatch_buffers(
            &coretests::uints::DESCRIPTOR,
            1,
            &[
                BufferBinding::read_only(&a).independent_length(),
                BufferBinding::read_write(&out).independent_length(),
            ],
        )
        .expect("the GPU refused `uints`");
    let uints = device.read_typed_buffer(&out).unwrap();

    let a = device.create_typed_buffer("coretests floats a", &FLOATS_A, ro);
    let out = device.create_typed_buffer("coretests floats out", &[f32::NAN; FLOATS_OUT], rw);
    device
        .dispatch_buffers(
            &coretests::floats::DESCRIPTOR,
            1,
            &[
                BufferBinding::read_only(&a).independent_length(),
                BufferBinding::read_write(&out).independent_length(),
            ],
        )
        .expect("the GPU refused `floats`");
    let floats = device.read_typed_buffer(&out).unwrap();

    let a = device.create_typed_buffer("coretests bools a", &BOOLS_A, ro);
    let out = device.create_typed_buffer("coretests bools out", &[u32::MAX; BOOLS_OUT], rw);
    device
        .dispatch_buffers(
            &coretests::bools::DESCRIPTOR,
            1,
            &[
                BufferBinding::read_only(&a).independent_length(),
                BufferBinding::read_write(&out).independent_length(),
            ],
        )
        .expect("the GPU refused `bools`");
    let bools = device.read_typed_buffer(&out).unwrap();

    let a = device.create_typed_buffer("coretests casts a", &CASTS_A, ro);
    let float_out = device.create_typed_buffer("coretests casts f", &[f32::NAN; CASTS_A.len()], rw);
    let int_out = device.create_typed_buffer("coretests casts r", &[i32::MIN; CASTS_A.len()], rw);
    device
        .dispatch_buffers(
            &coretests::casts::DESCRIPTOR,
            CASTS_A.len() as u32,
            &[
                BufferBinding::read_only(&a),
                BufferBinding::read_write(&float_out),
                BufferBinding::read_write(&int_out),
            ],
        )
        .expect("the GPU refused `casts`");

    Outputs {
        ints,
        uints,
        floats,
        bools,
        cast_float: device.read_typed_buffer(&float_out).unwrap(),
        cast_int: device.read_typed_buffer(&int_out).unwrap(),
    }
}

/// One output slot of one kernel and the literal it must hold.
type Check = (Stream, usize, Lit);

/// One row of the table: its id, the acceptance set and the slots it asserts.
struct Case {
    id: &'static str,
    accept: Accept,
    checks: &'static [Check],
}

const fn case(id: &'static str, accept: Accept, checks: &'static [Check]) -> Case {
    Case { id, accept, checks }
}

/// The value in a slot, read in the type of the literal the slot is compared with.
fn observed(outputs: &Outputs, stream: Stream, index: usize, want: Lit) -> Lit {
    let truth = |stored: u32| match stored {
        0 => Lit::B(false),
        1 => Lit::B(true),
        other => panic!("{stream:?}[{index}] holds {other}, a comparison stores 0 or 1"),
    };
    match (stream, want) {
        (Stream::Ints, Lit::I(_)) => Lit::I(outputs.ints[index]),
        (Stream::Ints, Lit::B(_)) => truth(outputs.ints[index] as u32),
        (Stream::Uints, Lit::U(_)) => Lit::U(outputs.uints[index]),
        (Stream::Uints, Lit::B(_)) => truth(outputs.uints[index]),
        (Stream::Floats, Lit::F(_)) => Lit::F(outputs.floats[index]),
        (Stream::Floats, Lit::B(_)) => {
            let stored = outputs.floats[index];
            if stored == 0.0 {
                Lit::B(false)
            } else if stored == 1.0 {
                Lit::B(true)
            } else {
                panic!("{stream:?}[{index}] holds {stored}, a comparison stores 0.0 or 1.0")
            }
        }
        (Stream::Bools, Lit::B(_)) => truth(outputs.bools[index]),
        (Stream::CastFloat, Lit::F(_)) => Lit::F(outputs.cast_float[index]),
        (Stream::CastInt, Lit::I(_)) => Lit::I(outputs.cast_int[index]),
        (stream, want) => {
            panic!("a case reads {stream:?}[{index}] as {want:?}, which that kernel does not write")
        }
    }
}

/// `observed` for the adapter's outputs, which a broken readback may fill with a value that is neither 0 nor 1 for a
/// comparison; that is a failure of the row, not a panic of the harness.
fn observed_or_text(
    outputs: &Outputs,
    stream: Stream,
    index: usize,
    want: Lit,
) -> Result<Lit, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        observed(outputs, stream, index, want)
    }))
    .map_err(|_| format!("{stream:?}[{index}] is not a valid {want:?} slot"))
}

/// Whether `got` is in the set `accept` allows for the literal `want` of the slot `(stream, index)`.
/// `index` selects the source integer of the two-neighbour set.
fn accepts(accept: Accept, stream: Stream, index: usize, want: Lit, got: Lit) -> bool {
    match accept {
        Accept::Exact => got == want,
        Accept::Ulps25 => match (got, want) {
            (Lit::F(got), Lit::F(want)) => ulps(got, want) <= DIV_ULPS,
            _ => false,
        },
        Accept::Neighbours => {
            let source = CASTS_A[index];
            match (stream, got) {
                (Stream::CastFloat, Lit::F(got)) => {
                    got == (source - 1) as f32 || got == (source + 1) as f32
                }
                (Stream::CastInt, Lit::I(got)) => got == source - 1 || got == source + 1,
                _ => false,
            }
        }
    }
}

/// The accepted set in words, for a failure message.
fn set_text(accept: Accept, stream: Stream, index: usize, want: Lit) -> String {
    match accept {
        Accept::Exact => format!("exactly {}", want.render()),
        Accept::Ulps25 => format!(
            "within {DIV_ULPS} ULP of {} (the 2.5 ULP set)",
            want.render()
        ),
        Accept::Neighbours => {
            let source = CASTS_A[index];
            match stream {
                Stream::CastFloat => format!(
                    "{:?} or {:?} (the neighbours of {source})",
                    (source - 1) as f32,
                    (source + 1) as f32
                ),
                _ => format!(
                    "{} or {} (the neighbours of {source})",
                    source - 1,
                    source + 1
                ),
            }
        }
    }
}

/// The rows of `docs/development/CORETESTS.md`, in order. The slot numbers are the `out[k]` of the kernels.
fn cases() -> Vec<Case> {
    vec![
        case("CT-01", Accept::Exact, &[(Stream::Ints, 0, Lit::I(0))]),
        case(
            "CT-02",
            Accept::Exact,
            &[(Stream::Ints, 1, Lit::I(0)), (Stream::Ints, 2, Lit::I(0))],
        ),
        case(
            "CT-03",
            Accept::Exact,
            &[(Stream::Ints, 3, Lit::I(0)), (Stream::Ints, 4, Lit::I(0))],
        ),
        case(
            "CT-04",
            Accept::Exact,
            &[(Stream::Ints, 5, Lit::I(0)), (Stream::Ints, 6, Lit::I(0))],
        ),
        case(
            "CT-05",
            Accept::Exact,
            &[(Stream::Ints, 7, Lit::I(0)), (Stream::Ints, 8, Lit::I(0))],
        ),
        case(
            "CT-06",
            Accept::Exact,
            &[(Stream::Ints, 9, Lit::I(0)), (Stream::Ints, 10, Lit::I(0))],
        ),
        case(
            "CT-07",
            Accept::Exact,
            &[(Stream::Ints, 11, Lit::I(0)), (Stream::Ints, 12, Lit::I(0))],
        ),
        case(
            "CT-08",
            Accept::Exact,
            &[(Stream::Ints, 13, Lit::I(0)), (Stream::Ints, 14, Lit::I(0))],
        ),
        case(
            "CT-09",
            Accept::Exact,
            &[(Stream::Ints, 15, Lit::I(0)), (Stream::Ints, 16, Lit::I(0))],
        ),
        case(
            "CT-10",
            Accept::Exact,
            &[(Stream::Ints, 17, Lit::I(0)), (Stream::Ints, 18, Lit::I(0))],
        ),
        case(
            "CT-11",
            Accept::Exact,
            &[(Stream::Ints, 19, Lit::I(0)), (Stream::Ints, 20, Lit::I(0))],
        ),
        case(
            "CT-12",
            Accept::Exact,
            &[(Stream::Uints, 0, Lit::U(0)), (Stream::Uints, 1, Lit::U(0))],
        ),
        case(
            "CT-13",
            Accept::Exact,
            &[(Stream::Uints, 2, Lit::U(0)), (Stream::Uints, 3, Lit::U(0))],
        ),
        case(
            "CT-14",
            Accept::Exact,
            &[(Stream::Uints, 4, Lit::U(0)), (Stream::Uints, 5, Lit::U(0))],
        ),
        case(
            "CT-15",
            Accept::Exact,
            &[(Stream::Uints, 6, Lit::U(0)), (Stream::Uints, 7, Lit::U(0))],
        ),
        case(
            "CT-16",
            Accept::Exact,
            &[(Stream::Uints, 8, Lit::U(0)), (Stream::Uints, 9, Lit::U(0))],
        ),
        case(
            "CT-17",
            Accept::Exact,
            &[
                (Stream::Uints, 10, Lit::U(0)),
                (Stream::Uints, 11, Lit::U(0)),
            ],
        ),
        case(
            "CT-18",
            Accept::Exact,
            &[
                (Stream::Uints, 12, Lit::U(0)),
                (Stream::Uints, 13, Lit::U(0)),
            ],
        ),
        case(
            "CT-19",
            Accept::Exact,
            &[
                (Stream::Uints, 14, Lit::U(0)),
                (Stream::Uints, 15, Lit::U(0)),
            ],
        ),
        case(
            "CT-20",
            Accept::Exact,
            &[
                (Stream::Uints, 16, Lit::U(0)),
                (Stream::Uints, 17, Lit::U(0)),
            ],
        ),
        case(
            "CT-21",
            Accept::Exact,
            &[
                (Stream::Uints, 18, Lit::U(0)),
                (Stream::Uints, 19, Lit::U(0)),
            ],
        ),
        case("CT-22", Accept::Exact, &[(Stream::Floats, 0, Lit::F(0.0))]),
        case(
            "CT-23",
            Accept::Exact,
            &[
                (Stream::Floats, 1, Lit::F(0.0)),
                (Stream::Floats, 2, Lit::F(0.0)),
            ],
        ),
        case(
            "CT-24",
            Accept::Exact,
            &[
                (Stream::Floats, 3, Lit::F(0.0)),
                (Stream::Floats, 4, Lit::F(0.0)),
            ],
        ),
        case(
            "CT-25",
            Accept::Exact,
            &[
                (Stream::Floats, 5, Lit::F(0.0)),
                (Stream::Floats, 6, Lit::F(0.0)),
            ],
        ),
        case(
            "CT-26",
            Accept::Ulps25,
            &[
                (Stream::Floats, 7, Lit::F(0.0)),
                (Stream::Floats, 8, Lit::F(0.0)),
            ],
        ),
        case("CT-27", Accept::Exact, &[(Stream::Bools, 0, Lit::B(false))]),
        case(
            "CT-28",
            Accept::Exact,
            &[
                (Stream::Bools, 1, Lit::B(false)),
                (Stream::Bools, 2, Lit::B(false)),
            ],
        ),
        case(
            "CT-29",
            Accept::Exact,
            &[
                (Stream::Bools, 3, Lit::B(false)),
                (Stream::Bools, 4, Lit::B(false)),
            ],
        ),
        case(
            "CT-30",
            Accept::Exact,
            &[
                (Stream::Bools, 5, Lit::B(false)),
                (Stream::Bools, 6, Lit::B(false)),
            ],
        ),
        case("CT-31", Accept::Exact, &[(Stream::Ints, 21, Lit::I(12))]),
        case("CT-32", Accept::Exact, &[(Stream::Ints, 22, Lit::I(8))]),
        case("CT-33", Accept::Exact, &[(Stream::Ints, 23, Lit::I(20))]),
        case("CT-34", Accept::Exact, &[(Stream::Ints, 24, Lit::I(5))]),
        case("CT-35", Accept::Exact, &[(Stream::Ints, 25, Lit::I(0))]),
        case("CT-36", Accept::Exact, &[(Stream::Uints, 20, Lit::U(12))]),
        case("CT-37", Accept::Exact, &[(Stream::Uints, 21, Lit::U(8))]),
        case("CT-38", Accept::Exact, &[(Stream::Uints, 22, Lit::U(20))]),
        case("CT-39", Accept::Exact, &[(Stream::Uints, 23, Lit::U(5))]),
        case("CT-40", Accept::Exact, &[(Stream::Uints, 24, Lit::U(0))]),
        case("CT-41", Accept::Exact, &[(Stream::Floats, 9, Lit::F(12.0))]),
        case("CT-42", Accept::Exact, &[(Stream::Floats, 10, Lit::F(8.0))]),
        case(
            "CT-43",
            Accept::Exact,
            &[(Stream::Floats, 11, Lit::F(20.0))],
        ),
        case(
            "CT-44",
            Accept::Ulps25,
            &[(Stream::Floats, 12, Lit::F(5.0))],
        ),
        case("CT-45", Accept::Exact, &[(Stream::Ints, 26, Lit::I(90))]),
        case("CT-46", Accept::Exact, &[(Stream::Ints, 27, Lit::I(0))]),
        case("CT-47", Accept::Exact, &[(Stream::Ints, 28, Lit::I(1))]),
        case("CT-48", Accept::Exact, &[(Stream::Ints, 29, Lit::I(-10))]),
        case("CT-49", Accept::Exact, &[(Stream::Ints, 30, Lit::I(10240))]),
        case(
            "CT-50",
            Accept::Exact,
            &[(Stream::Ints, 31, Lit::I(655360))],
        ),
        case("CT-51", Accept::Exact, &[(Stream::Ints, 32, Lit::I(160))]),
        case("CT-52", Accept::Exact, &[(Stream::Ints, 33, Lit::I(1000))]),
        case("CT-53", Accept::Exact, &[(Stream::Ints, 34, Lit::I(10000))]),
        case("CT-54", Accept::Exact, &[(Stream::Ints, 35, Lit::I(100))]),
        case(
            "CT-55",
            Accept::Exact,
            &[(Stream::Ints, 36, Lit::I(368640))],
        ),
        case(
            "CT-56",
            Accept::Exact,
            &[(Stream::Ints, 37, Lit::I(269488144))],
        ),
        case("CT-57", Accept::Exact, &[(Stream::Ints, 38, Lit::I(0))]),
        case(
            "CT-58",
            Accept::Exact,
            &[(Stream::Ints, 39, Lit::I(808464432))],
        ),
        case("CT-59", Accept::Exact, &[(Stream::Ints, 40, Lit::B(true))]),
        case("CT-60", Accept::Exact, &[(Stream::Ints, 41, Lit::B(true))]),
        case("CT-61", Accept::Exact, &[(Stream::Ints, 42, Lit::I(0))]),
        case("CT-62", Accept::Exact, &[(Stream::Ints, 43, Lit::I(14))]),
        case("CT-63", Accept::Exact, &[(Stream::Ints, 44, Lit::I(8))]),
        case("CT-64", Accept::Exact, &[(Stream::Ints, 45, Lit::I(6))]),
        case("CT-65", Accept::Exact, &[(Stream::Ints, 46, Lit::I(14))]),
        case("CT-66", Accept::Exact, &[(Stream::Ints, 47, Lit::I(7))]),
        case(
            "CT-67",
            Accept::Exact,
            &[
                (Stream::Ints, 48, Lit::I(-4)),
                (Stream::Ints, 49, Lit::I(-4)),
            ],
        ),
        case("CT-68", Accept::Exact, &[(Stream::Uints, 25, Lit::B(true))]),
        case("CT-69", Accept::Exact, &[(Stream::Uints, 26, Lit::B(true))]),
        case("CT-70", Accept::Exact, &[(Stream::Uints, 27, Lit::U(14))]),
        case("CT-71", Accept::Exact, &[(Stream::Uints, 28, Lit::U(8))]),
        case("CT-72", Accept::Exact, &[(Stream::Uints, 29, Lit::U(6))]),
        case("CT-73", Accept::Exact, &[(Stream::Uints, 30, Lit::U(14))]),
        case("CT-74", Accept::Exact, &[(Stream::Uints, 31, Lit::U(7))]),
        case(
            "CT-75",
            Accept::Exact,
            &[
                (Stream::Uints, 32, Lit::U(4294967284)),
                (Stream::Uints, 33, Lit::U(4294967284)),
            ],
        ),
        case(
            "CT-76",
            Accept::Exact,
            &[(Stream::Floats, 13, Lit::B(true))],
        ),
        case(
            "CT-77",
            Accept::Exact,
            &[
                (Stream::CastFloat, 0, Lit::F(16777214.0)),
                (Stream::CastInt, 0, Lit::I(16777214)),
            ],
        ),
        case(
            "CT-78",
            Accept::Exact,
            &[
                (Stream::CastFloat, 1, Lit::F(16777215.0)),
                (Stream::CastInt, 1, Lit::I(16777215)),
            ],
        ),
        case(
            "CT-79",
            Accept::Exact,
            &[
                (Stream::CastFloat, 2, Lit::F(16777216.0)),
                (Stream::CastInt, 2, Lit::I(16777216)),
            ],
        ),
        case(
            "CT-80",
            Accept::Neighbours,
            &[
                (Stream::CastFloat, 3, Lit::F(16777216.0)),
                (Stream::CastInt, 3, Lit::I(16777216)),
            ],
        ),
        case(
            "CT-81",
            Accept::Exact,
            &[
                (Stream::CastFloat, 4, Lit::F(-16777214.0)),
                (Stream::CastInt, 4, Lit::I(-16777214)),
            ],
        ),
        case(
            "CT-82",
            Accept::Exact,
            &[
                (Stream::CastFloat, 5, Lit::F(-16777215.0)),
                (Stream::CastInt, 5, Lit::I(-16777215)),
            ],
        ),
        case(
            "CT-83",
            Accept::Exact,
            &[
                (Stream::CastFloat, 6, Lit::F(-16777216.0)),
                (Stream::CastInt, 6, Lit::I(-16777216)),
            ],
        ),
        case(
            "CT-84",
            Accept::Neighbours,
            &[
                (Stream::CastFloat, 7, Lit::F(-16777216.0)),
                (Stream::CastInt, 7, Lit::I(-16777216)),
            ],
        ),
    ]
}

// ---- the table ----

/// The cells of a table row; `\|` is a literal bar inside a cell.
fn cells(line: &str) -> Vec<String> {
    let mut cells = vec![String::new()];
    let mut chars = line.trim().chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' if chars.peek() == Some(&'|') => {
                chars.next();
                cells.last_mut().unwrap().push('|');
            }
            '|' => cells.push(String::new()),
            other => cells.last_mut().unwrap().push(other),
        }
    }
    // The line starts and ends with a bar, so the first and the last cell are empty.
    let mut cells: Vec<String> = cells.iter().map(|cell| cell.trim().to_owned()).collect();
    cells.remove(0);
    cells.pop();
    cells
}

/// The rows of the table: lines that start with `| CT-`, keyed by id. A duplicate row is a failure.
fn table_rows() -> BTreeMap<String, Vec<String>> {
    let mut rows = BTreeMap::new();
    for line in TABLE.lines().filter(|line| line.starts_with("| CT-")) {
        let row = cells(line);
        assert_eq!(row.len(), 9, "a table row has nine cells: {line}");
        assert!(
            rows.insert(row[0].clone(), row).is_none(),
            "a duplicate row in the table: {line}"
        );
    }
    rows
}

/// `name[1,2]` groups, joined by spaces: the kernel slots a case asserts.
fn slots_cell(case: &Case) -> String {
    let mut by_kernel: BTreeMap<&str, BTreeSet<usize>> = BTreeMap::new();
    for (stream, index, _) in case.checks {
        by_kernel.entry(stream.kernel()).or_default().insert(*index);
    }
    by_kernel
        .iter()
        .map(|(kernel, slots)| {
            let slots: Vec<String> = slots.iter().map(usize::to_string).collect();
            format!("{kernel}[{}]", slots.join(","))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The expected literals of a case in the spelling of the table: distinct values in order, joined by `; `.
fn expected_cell(case: &Case) -> String {
    let mut seen: Vec<String> = Vec::new();
    for (_, _, expected) in case.checks {
        let text = expected.render();
        if !seen.contains(&text) {
            seen.push(text);
        }
    }
    seen.join("; ")
}

/// Prints the adapter the way the other Gpu tests do, with the identity triple of R1-5.
fn print_adapter(device: &HeadlessDevice) {
    let info = device.adapter_info();
    eprintln!(
        "coretests adapter: name={:?} backend={:?} device_type={:?} driver={:?} driver_info={:?}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );
}

// ---- tests ----

#[test]
fn scalar_coretests_run_on_the_gpu_and_the_twin_against_the_literals() {
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    print_adapter(&device);

    let gpu = run_gpu(&device);
    let twin = run_twins();

    let mut executed: Vec<&'static str> = Vec::new();
    let mut failures = String::new();
    let mut slots = 0;
    let mut max_div_ulps = 0u64;
    for case in cases() {
        executed.push(case.id);
        for &(stream, index, want) in case.checks {
            slots += 1;
            let twin_value = observed_or_text(&twin, stream, index, want);
            let gpu_value = observed_or_text(&gpu, stream, index, want);
            let show = |value: &Result<Lit, String>| match value {
                Ok(lit) => lit.render(),
                Err(text) => text.clone(),
            };

            // The twin is Rust's own arithmetic: exact in every row (including the 2.5 ULP and two-neighbour rows).
            let twin_ok =
                matches!(twin_value, Ok(got) if accepts(Accept::Exact, stream, index, want, got));
            let gpu_ok =
                matches!(gpu_value, Ok(got) if accepts(case.accept, stream, index, want, got));

            if let (Accept::Ulps25, Ok(Lit::F(got)), Lit::F(literal)) =
                (case.accept, &gpu_value, want)
            {
                let distance = ulps(*got, literal);
                eprintln!(
                    "coretests {} {stream:?}[{index}]: GPU {got:?} is {distance} ULP from the literal {}",
                    case.id,
                    want.render()
                );
                if distance != u64::MAX {
                    max_div_ulps = max_div_ulps.max(distance);
                }
            }
            if case.accept == Accept::Neighbours {
                eprintln!(
                    "coretests {} {stream:?}[{index}]: GPU {} (literal {}, set {})",
                    case.id,
                    show(&gpu_value),
                    want.render(),
                    set_text(case.accept, stream, index, want)
                );
            }

            if !twin_ok || !gpu_ok {
                let _ = writeln!(
                    failures,
                    "{} {stream:?}[{index}]: expected {} ({}); GPU {} [{}], twin {} [{}]",
                    case.id,
                    set_text(case.accept, stream, index, want),
                    case.accept.label(),
                    show(&gpu_value),
                    if gpu_ok { "ok" } else { "FAIL" },
                    show(&twin_value),
                    if twin_ok { "ok" } else { "FAIL" },
                );
            }
        }
    }
    eprintln!(
        "coretests: {} cases, {slots} output slots; largest float `/` distance {max_div_ulps} ULP",
        executed.len()
    );

    // The id guard: every id of the table is executed exactly once, and no executed id is unknown.
    let table = table_rows();
    let documented_set: BTreeSet<&str> = table.keys().map(String::as_str).collect();
    let executed_set: BTreeSet<&str> = executed.iter().copied().collect();
    assert_eq!(
        executed_set.len(),
        executed.len(),
        "a case id was executed twice"
    );
    let unknown: Vec<&&str> = executed_set.difference(&documented_set).collect();
    let missing: Vec<&&str> = documented_set.difference(&executed_set).collect();
    assert!(
        unknown.is_empty() && missing.is_empty(),
        "docs/development/CORETESTS.md and the executed cases differ: executed but not in the table {unknown:?}, \
         in the table but not executed {missing:?}"
    );
    assert_eq!(executed.len(), 84, "CT-01 to CT-84 are executed");
    assert_eq!(slots, 121, "the output slots asserted");

    assert!(
        failures.is_empty(),
        "coretests failed on the adapter or the twin:\n{failures}"
    );
}

#[test]
fn the_table_states_the_slots_the_literals_and_the_acceptance_the_cases_assert() {
    let table = table_rows();
    let cases = cases();
    assert_eq!(
        table.len(),
        cases.len(),
        "the table and the cases have the same number of rows"
    );
    for case in &cases {
        let row = table
            .get(case.id)
            .unwrap_or_else(|| panic!("{} is not in the table", case.id));
        assert_eq!(row[4], slots_cell(case), "{}: kernel slots", case.id);
        assert_eq!(row[5], expected_cell(case), "{}: expected literal", case.id);
        assert_eq!(row[6], case.accept.label(), "{}: acceptance set", case.id);
    }
}

#[test]
fn every_output_slot_of_every_kernel_is_asserted_by_exactly_one_case() {
    let mut claimed: BTreeMap<Stream, Vec<usize>> = BTreeMap::new();
    for case in cases() {
        for &(stream, index, _) in case.checks {
            claimed.entry(stream).or_default().push(index);
        }
    }
    for stream in Stream::ALL {
        let mut slots = claimed.remove(&stream).unwrap_or_default();
        slots.sort_unstable();
        let all: Vec<usize> = (0..stream.len()).collect();
        assert_eq!(slots, all, "the slots of {stream:?} claimed by the cases");
    }
}

/// `value` moved by `steps` representable `f32` values (positive: up in magnitude for positive `value`).
fn nudge(value: f32, steps: i32) -> f32 {
    f32::from_bits(value.to_bits().wrapping_add_signed(steps))
}

#[test]
fn the_acceptance_sets_accept_what_r1_7_says_and_nothing_else() {
    let slot = (Stream::Floats, 12);
    let f = Lit::F;
    let check =
        |accept: Accept, want: f32, got: f32| accepts(accept, slot.0, slot.1, f(want), f(got));

    // Float `/`: the literal and up to 2 ULP either way pass, 3 ULP and NaN do not.
    for want in [5.0f32, 1.0, 0.1, 1234.5, -5.0] {
        for steps in [0, 1, -1, 2, -2] {
            assert!(
                check(Accept::Ulps25, want, nudge(want, steps)),
                "{want} {steps:+} ULP must be in the 2.5 ULP set"
            );
        }
        for steps in [3, -3, 4, -4] {
            assert!(
                !check(Accept::Ulps25, want, nudge(want, steps)),
                "{want} {steps:+} ULP must not be in the 2.5 ULP set"
            );
        }
        assert!(!check(Accept::Ulps25, want, f32::NAN));
        assert!(!check(Accept::Ulps25, want, f32::INFINITY));
    }
    assert!(!check(Accept::Ulps25, f32::NAN, 5.0));
    // Zeros of either sign are within 0 ULP of each other, and a small float is not near zero.
    assert!(check(Accept::Ulps25, 0.0, -0.0));
    assert!(check(Accept::Ulps25, -0.0, 0.0));
    assert!(!check(Accept::Ulps25, 0.0, 1.0));
    // A non-float literal or value is never in the set.
    assert!(!accepts(
        Accept::Ulps25,
        slot.0,
        slot.1,
        Lit::I(5),
        Lit::I(5)
    ));

    // Exact: equal values, and a zero of either sign; one ULP, NaN and a different type are not.
    assert!(check(Accept::Exact, 0.0, -0.0));
    assert!(check(Accept::Exact, -0.0, 0.0));
    assert!(check(Accept::Exact, 5.0, 5.0));
    assert!(!check(Accept::Exact, 5.0, nudge(5.0, 1)));
    assert!(!check(Accept::Exact, 5.0, f32::NAN));
    assert!(!check(Accept::Exact, f32::NAN, f32::NAN));
    assert!(!accepts(
        Accept::Exact,
        slot.0,
        slot.1,
        Lit::I(5),
        Lit::U(5)
    ));

    // Two-neighbour set, both streams, both signs (CASTS_A[3] = 16777217, CASTS_A[7] = -16777217): the integer on
    // either side passes; the source integer (not representable as `f32`), the far side and a wrong type do not.
    for (index, source) in [(3usize, 16_777_217i32), (7, -16_777_217)] {
        assert_eq!(CASTS_A[index], source);
        let cast = |stream, want, got| accepts(Accept::Neighbours, stream, index, want, got);
        for got in [source - 1, source + 1] {
            assert!(cast(Stream::CastInt, Lit::I(0), Lit::I(got)), "{got}");
            assert!(
                cast(Stream::CastFloat, Lit::F(0.0), Lit::F(got as f32)),
                "{got}"
            );
        }
        assert!(!cast(Stream::CastInt, Lit::I(0), Lit::I(source)));
        assert!(!cast(Stream::CastInt, Lit::I(0), Lit::I(source + 3)));
        assert!(!cast(Stream::CastInt, Lit::I(0), Lit::I(source - 3)));
        assert!(!cast(
            Stream::CastFloat,
            Lit::F(0.0),
            Lit::F((source + 3) as f32)
        ));
        assert!(!cast(Stream::CastFloat, Lit::F(0.0), Lit::F(f32::NAN)));
        assert!(!cast(Stream::CastFloat, Lit::F(0.0), Lit::I(source + 1)));
    }
}
