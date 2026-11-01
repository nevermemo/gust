//! The scalar coretests transcription on the CPU twin (roadmap M5, slice m5.d1; Unit tier, no adapter).
//!
//! `tests/fixtures/coretests.rs` holds the `#[gpu]` kernels of the cases that `docs/development/CORETESTS.md`
//! transcribes from the Rust core tests (tag 1.98.0), and this file runs the CPU twin of every kernel and compares
//! each output slot with the expected literal of the table. The literal is the independent reference (D26): it comes
//! from the Rust sources or is derived by hand, never from the twin. The twin is never GPU evidence; the GPU half is
//! slice m5.d2, which reads the same table and applies the acceptance sets of the "Acceptance" column. The twin is
//! asserted exact in every row, including the float `/` rows and the casts above 2^24.
//!
//! Guards: every `CT-nn` of the table has a case here, every case is in the table, the kernel slots, the expected
//! literals, the acceptance sets and the unique-coverage numbers of the table equal what these cases say, every
//! output slot of every kernel is claimed by exactly one case, and every id appears in the fixture.

// CT-69 transcribes `u32::MIN <= 0` (uint_macros.rs:11), which is always true for `u32`;
// the kernel must still run it, so the including test allows the lint.
#![allow(clippy::absurd_extreme_comparisons)]

include!("../../../tests/fixtures/coretests.rs");

use std::collections::{BTreeMap, BTreeSet};

use gust::Independent;
use gust::cpu::CpuDevice;

const TABLE: &str = include_str!("../../../docs/development/CORETESTS.md");
const FIXTURE: &str = include_str!("../../../tests/fixtures/coretests.rs");

// ---- kernel inputs (the names are the variables of the kernels) ----

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
    /// The spelling of the table.
    fn render(self) -> String {
        match self {
            Self::I(value) => value.to_string(),
            Self::U(value) => value.to_string(),
            Self::F(value) => format!("{value:?}"),
            Self::B(value) => value.to_string(),
        }
    }
}

/// What the GPU may return for a row (R1-7). The twin is exact in every row; the set is carried here so that the
/// table and the cases cannot disagree about it, and m5.d2 applies it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Accept {
    /// Bit equality for integers and `bool`; `==` for floats (WGSL allows any sign of a zero).
    Exact,
    /// Float `/`: every `f32` within 2.5 ULP of the literal, asserted as `gust::cpu::ulps(gpu, literal) <= 2`.
    Ulps25,
    /// An integer above 2^24 converted to `f32`: either neighbour of the integer (WGSL 15.7.6).
    Neighbours,
}

impl Accept {
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

/// Everything the twins wrote.
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

/// The rows of the table: lines that start with `| CT-`.
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

// ---- tests ----

#[test]
fn the_twin_of_every_kernel_equals_every_coretests_literal() {
    let outputs = run_twins();
    let mut asserted = 0;
    for case in cases() {
        for &(stream, index, expected) in case.checks {
            let actual = observed(&outputs, stream, index, expected);
            assert_eq!(
                actual,
                expected,
                "{}: the twin of `{}` slot {index} differs from the coretests literal",
                case.id,
                stream.kernel()
            );
            asserted += 1;
        }
    }
    assert_eq!(asserted, 121, "the output slots asserted");
}

#[test]
fn every_table_id_has_a_case_and_every_case_is_in_the_table() {
    let cases = cases();
    let ids: Vec<&str> = cases.iter().map(|case| case.id).collect();
    let unique: BTreeSet<&str> = ids.iter().copied().collect();
    assert_eq!(unique.len(), ids.len(), "a case id is used twice");
    for (position, id) in ids.iter().enumerate() {
        assert_eq!(
            *id,
            format!("CT-{:02}", position + 1),
            "ids run CT-01, CT-02, ... in order"
        );
    }
    let table = table_rows();
    let documented: BTreeSet<&str> = table.keys().map(String::as_str).collect();
    assert_eq!(
        documented, unique,
        "docs/development/CORETESTS.md ids and the cases of tests/coretests_twin.rs differ"
    );
    assert_eq!(ids.len(), 84, "CT-01 to CT-84");
    for id in &ids {
        assert!(
            FIXTURE.contains(id),
            "{id} is not mentioned in tests/fixtures/coretests.rs"
        );
    }
}

#[test]
fn the_table_states_the_slots_the_literals_and_the_acceptance_the_cases_assert() {
    let table = table_rows();
    for case in cases() {
        let row = &table[case.id];
        assert_eq!(row[4], slots_cell(&case), "{}: kernel slots", case.id);
        assert_eq!(
            row[5],
            expected_cell(&case),
            "{}: expected literal",
            case.id
        );
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

#[test]
fn the_unique_coverage_numbers_of_the_table_are_what_its_rows_say() {
    let table = table_rows();
    // `op type class [new|existing]`, separated by `;` in the last-but-one cell.
    let mut flags: BTreeMap<String, String> = BTreeMap::new();
    for row in table.values() {
        for triple in row[7].split(';').map(str::trim) {
            let (key, flag) = triple
                .rsplit_once(" [")
                .unwrap_or_else(|| panic!("a triple ends in [new] or [existing]: {triple}"));
            let flag = flag.trim_end_matches(']');
            assert!(flag == "new" || flag == "existing", "{triple}");
            let previous = flags.insert(key.to_owned(), flag.to_owned());
            assert!(
                previous.is_none() || previous.as_deref() == Some(flag),
                "the triple `{key}` is both new and existing"
            );
        }
    }
    let new = flags.values().filter(|flag| flag.as_str() == "new").count();
    let line = TABLE
        .lines()
        .find(|line| line.starts_with("Unique coverage:"))
        .expect("the table states `Unique coverage: triples=N new=M existing=K`");
    let stated = format!(
        "Unique coverage: triples={} new={} existing={}",
        flags.len(),
        new,
        flags.len() - new
    );
    assert_eq!(
        line.trim(),
        stated,
        "the stated unique coverage and the table differ"
    );

    // The operator-type pairs: the first two words of a triple. Their split into new and existing rests on a reading
    // of the fixtures, so only the total and the sum are checked.
    let pairs: BTreeSet<String> = flags
        .keys()
        .map(|key| key.split(' ').take(2).collect::<Vec<_>>().join(" "))
        .collect();
    let line = TABLE
        .lines()
        .find(|line| line.starts_with("Unique operator-type pairs:"))
        .expect("the table states `Unique operator-type pairs: pairs=N new=M existing=K`");
    let numbers: Vec<usize> = line
        .split(['=', ' '])
        .filter_map(|word| word.parse().ok())
        .collect();
    assert_eq!(numbers.len(), 3, "{line}");
    assert_eq!(
        numbers[0],
        pairs.len(),
        "the stated pair total and the table differ"
    );
    assert_eq!(
        numbers[1] + numbers[2],
        numbers[0],
        "new and existing pairs add up to the total"
    );
}
