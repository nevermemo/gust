//! The known-mismatch ledger (roadmap M3, slice m3.a1; D22, D38; `docs/SEMANTICS.md`).
//!
//! Every row pins three outcomes of one construct whose result is not portable and that no
//! diagnostic can prevent, because the operands are run-time values: what the GPU does
//! (measured here, through `HeadlessDevice`), what plain Rust does (a closure with
//! `black_box` operands under `catch_unwind`, in this test crate's own profile) and what the
//! CPU twin does (the macro-emitted `Run` impl of the same kernel on `gust::cpu::CpuDevice`,
//! one case per dispatch under `catch_unwind`; slice m3.d). The twin runs the shadow body
//! compiled in this crate, so it must equal the plain-Rust column in the same mode; it is
//! compared with Rust and never with the GPU (D39: an edge case that the GPU leaves
//! unspecified is recorded, not matched). Rust's and the twin's expectation is a function of
//! the overflow-checks mode of this crate, read with `gust::cpu::overflow_checks!()`; no row
//! is ignored in any profile.
//!
//! `DOMAIN` rows are controls inside the defined domain (results that fit, non-zero
//! divisors, shift amounts below 32, normal finite floats): GPU and Rust must agree on
//! both columns in both modes. A disagreement there is not a ledger row, it means the
//! contract's defined domain is wrong (spike S2 kill rule), and the test fails.
//!
//! Two kinds of adapter run this file (slice m5.b, D52): hardware and software (`device_type` `Cpu`, lavapipe).
//! An enforced row and every `DOMAIN` control have one pin that holds on both. An unenforced row (KM-07, KM-08,
//! KM-11, KM-12 to KM-14 and the NaN casts of KM-05 and KM-06) has its hardware observation in `Case::gpu` and, where
//! lavapipe differs, its software observation in `Case::software`; the NOTE compares with the running kind's pin. Each
//! run writes one `LEDGER` line per case to `ledger-<kind>.txt` under the target's tmp directory.
//!
//! Labels (docs/SEMANTICS.md): the GPU values pinned here are *measured* on one adapter
//! and are never promoted to *proven* by this file. Each failing row names its id.

include!("../../../tests/fixtures/mismatch.rs");

use std::{
    any::Any,
    collections::BTreeSet,
    hint::black_box,
    panic::{AssertUnwindSafe, catch_unwind},
};

use gust::cpu::texture::linear_error_bound;
use gust::cpu::{CpuBuffer, CpuDevice, Error as CpuError, OverflowChecks};
use gust::texture::{Address, Filter, R32Float, R32Uint, SamplerOptions, TextureUse};
use gust::{GpuPod, Independent, KernelDescriptor};
use gust_wgpu::{BufferBinding, DeviceOptions, GpuBufferAccess, HeadlessDevice};

// KM-12 to KM-14 (slice mx.t4, D62, D64): the three texture constructs whose GPU outcome the specification leaves open
// or bounds. As in the fixture, every operand comes from a buffer, so the validator cannot reject a kernel and the
// compiler cannot fold it. `at` is the texel address `[x, y]` for the first two kernels and the normalised
// coordinate `[u, v]` for `sample_at`.
#[gust::gpu]
mod texture_edges {
    // KM-12: `load` at the address of `at`; the loaded texel goes to `seen`.
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn load_texel(
        id: SV_DispatchThreadID,
        at: StructuredBuffer<u32>,
        image: StorageTexture2D<R32Uint>,
        mut seen: RWStructuredBuffer<u32>,
    ) {
        seen[id.x] = image.load(at[0u32], at[1u32]);
    }

    // KM-12: `store` of 77 at the address of `at`.
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn store_texel(
        id: SV_DispatchThreadID,
        at: StructuredBuffer<u32>,
        mut image: StorageTexture2D<R32Uint>,
    ) {
        image.store(at[0u32], at[1u32], 77u32 + id.x);
    }

    // KM-13 and KM-14: one `sample` at the coordinate of `at`; the filter is the sampler's.
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn sample_at(
        id: SV_DispatchThreadID,
        image: Texture2D<f32>,
        filter: Sampler,
        at: StructuredBuffer<f32>,
        mut out: RWStructuredBuffer<f32>,
    ) {
        out[id.x] = image.sample(filter, at[0u32], at[1u32]);
    }
}

// ---- overflow-checks detection (the consumer-side macro of gust::cpu, D38) ----

/// The overflow mode of this test crate, which is also the crate that holds the shadow bodies the twin runs. The
/// macro expands its probe here, so it answers for this crate's build whatever `gust` itself was compiled with. It is
/// cached per call site, so this is the only call site. In a checked build the first call prints one panic message to
/// stderr (the default hook); the ledger test calls it first.
fn overflow_checks() -> OverflowChecks {
    gust::cpu::overflow_checks!()
}

/// Picks the checked or the wrapping outcome of a row.
trait Pick {
    fn pick(self, checked: Outcome, wrapping: Outcome) -> Outcome;
}

impl Pick for OverflowChecks {
    fn pick(self, checked: Outcome, wrapping: Outcome) -> Outcome {
        match self {
            Self::Checked => checked,
            Self::Wrapping => wrapping,
            // `OverflowChecks` is non-exhaustive: a future mode needs a decision here, not a silent default.
            other => panic!("the ledger has no column for the overflow mode {other:?}"),
        }
    }
}

// ---- outcomes ----

/// A pinned outcome. Values are 32-bit patterns (`i32` and `f32` as their bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    Value(u32),
    /// Any NaN bit pattern (the sign and payload of a NaN are not part of the contract).
    Nan,
    /// A panic whose message starts with this text.
    Panic(&'static str),
    /// Set-valued GPU outcomes (KM-09): every observation lies in the inclusive range.
    Within(u32, u32),
    /// As `Within`, or this one extra value (the unchanged sentinel of a weak compare-exchange, WGSL 17.8.5).
    WithinOr(u32, u32, u32),
}

/// What one side actually did.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Observed {
    Value(u32),
    Panic(String),
}

impl Outcome {
    fn accepts(self, observed: &Observed) -> bool {
        match (self, observed) {
            (Self::Value(want), Observed::Value(got)) => want == *got,
            (Self::Nan, Observed::Value(got)) => f32::from_bits(*got).is_nan(),
            (Self::Panic(prefix), Observed::Panic(message)) => message.starts_with(prefix),
            (Self::Within(low, high), Observed::Value(got)) => (low..=high).contains(got),
            (Self::WithinOr(low, high, extra), Observed::Value(got)) => {
                (low..=high).contains(got) || *got == extra
            }
            _ => false,
        }
    }
}

/// A pin in hex where it is one value (the derived `Debug` prints the bits in decimal).
fn pin_text(pin: Outcome) -> String {
    match pin {
        Outcome::Value(bits) => format!("0x{bits:08x}"),
        other => format!("{other:?}"),
    }
}

fn show(observed: &Observed) -> String {
    match observed {
        Observed::Value(bits) => format!("0x{bits:08x}"),
        Observed::Panic(message) => format!("panic({message})"),
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "<non-string panic payload>".to_owned()
    }
}

// ---- adapter kinds ----

/// The kind of adapter that runs the ledger (R1-5). Unenforced rows are pinned once per kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdapterKind {
    /// Any adapter that is not a CPU implementation.
    Hardware,
    /// A CPU implementation of Vulkan (`wgpu::DeviceType::Cpu`, for example lavapipe).
    Software,
}

impl AdapterKind {
    fn of(device_type: wgpu::DeviceType) -> Self {
        if device_type == wgpu::DeviceType::Cpu {
            Self::Software
        } else {
            Self::Hardware
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Hardware => "hardware",
            Self::Software => "software",
        }
    }
}

// ---- cases ----

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    UDiv,
    IDiv,
    UShift,
    IShift,
    UArith,
    IArith,
    UWrap,
    IWrap,
    CastInt,
    CastUint,
    FloatOp,
    FloatCmp,
    OobLoad,
    OobStoreLast,
    OobStoreElsewhere,
    ContendedExchange,
    ContendedCompareExchange,
    UncontendedCompareExchange,
    /// KM-12: `load` at the texel `(operands.1, operands.2)` of the 4 x 2 texture, outside it.
    TexLoadOob,
    /// KM-12: `store` of 77 there; the observation is the row-major index of the texel that holds 77 afterwards.
    TexStoreOob,
    /// KM-13: a linear `sample` at `(f32::from_bits(operands.1), f32::from_bits(operands.2))`.
    LinearSample,
    /// KM-14: a nearest `sample` at `u = f32::from_bits(operands.1)`, `v = 0.5`.
    NearestSample,
}

struct Case {
    /// `KM-nn`, or `DOMAIN` for a control inside the defined domain.
    id: &'static str,
    what: &'static str,
    kind: Kind,
    /// Operation selector, first operand, second operand (bit patterns).
    operands: (u32, u32, u32),
    /// The pinned GPU outcome: the requirement of an enforced row, the hardware observation of an unenforced one.
    gpu: Outcome,
    /// The observation on a software adapter (lavapipe), for an unenforced row where it differs from `gpu`. `None` is
    /// "no software pin": the hardware pin accepted m5.b0's value there. Never set on an enforced row or on a
    /// set-valued KM-09 row.
    software: Option<Outcome>,
    rust: fn(OverflowChecks) -> Outcome,
    /// Whether the pinned GPU value is a requirement the test enforces (see `gpu_is_enforced`).
    enforced: bool,
}

impl Case {
    /// Adds the software-adapter observation of an unenforced row.
    fn software(mut self, observation: Outcome) -> Self {
        self.software = Some(observation);
        self
    }

    /// The pin that the running kind compares with: the one pin of an enforced row, else the kind's own.
    fn pin(&self, kind: AdapterKind) -> Option<Outcome> {
        if self.enforced {
            return Some(self.gpu);
        }
        match kind {
            AdapterKind::Hardware => Some(self.gpu),
            AdapterKind::Software => self.software,
        }
    }
}

fn case(
    id: &'static str,
    what: &'static str,
    kind: Kind,
    operands: (u32, u32, u32),
    gpu: Outcome,
    rust: fn(OverflowChecks) -> Outcome,
) -> Case {
    Case {
        id,
        what,
        kind,
        operands,
        gpu,
        software: None,
        rust,
        enforced: gpu_is_enforced(id, kind, operands.1),
    }
}

const fn i(value: i32) -> u32 {
    value as u32
}

const fn f(value: f32) -> u32 {
    value.to_bits()
}

const fn v(bits: u32) -> Outcome {
    Outcome::Value(bits)
}

const fn p(prefix: &'static str) -> Outcome {
    Outcome::Panic(prefix)
}

const DIV_ZERO: &str = "attempt to divide by zero";
const REM_ZERO: &str = "attempt to calculate the remainder with a divisor of zero";
const DIV_OVERFLOW: &str = "attempt to divide with overflow";
const REM_OVERFLOW: &str = "attempt to calculate the remainder with overflow";
const SHL_OVERFLOW: &str = "attempt to shift left with overflow";
const SHR_OVERFLOW: &str = "attempt to shift right with overflow";
const ADD_OVERFLOW: &str = "attempt to add with overflow";
const SUB_OVERFLOW: &str = "attempt to subtract with overflow";
const MUL_OVERFLOW: &str = "attempt to multiply with overflow";
const NEG_OVERFLOW: &str = "attempt to negate with overflow";
const OOB: &str = "index out of bounds";

/// KM-11: the index passed to `out_of_bounds`, equal to the length of `sink` (4).
const OOB_INDEX: u32 = 4;
/// KM-11 load row: `source` is as long as `sink`, so the load `source[4]` is out of bounds (and comes first).
const OOB_LOAD_SOURCE_LEN: usize = 4;
/// KM-11 store rows: `source` is longer than `sink`, so the load `source[4]` is in bounds by construction and the
/// store `sink[4] = 77` is the only out-of-bounds access of the kernel.
const OOB_STORE_SOURCE_LEN: usize = 8;

/// The contents of `source`: 10, 11, 12, ... (so `source[4]` is 14 where it exists).
fn oob_source(len: usize) -> Vec<u32> {
    (10..).take(len).collect()
}

const NAN: u32 = f32::NAN.to_bits();
const INF: u32 = f32::INFINITY.to_bits();
const NEG_INF: u32 = f32::NEG_INFINITY.to_bits();
const DENORMAL: f32 = 1.0e-40;

/// Number of invocations that contend in the KM-09 kernels.
const CONTENDERS: u32 = 257;
/// How many times each contended kernel is dispatched to sample the GPU's order.
const CONTENDED_RUNS: usize = 32;

fn cases() -> Vec<Case> {
    use Kind::*;
    // Both modes expect the same outcome.
    let mut rows = Vec::new();
    let mut add = |row: Case| rows.push(row);

    // ---- DOMAIN controls (integers) ----
    add(case("DOMAIN", "u32 7 / 2", UDiv, (0, 7, 2), v(3), |_| v(3)));
    add(case("DOMAIN", "u32 7 % 2", UDiv, (1, 7, 2), v(1), |_| v(1)));
    add(case(
        "DOMAIN",
        "i32 -7 / 2",
        IDiv,
        (0, i(-7), 2),
        v(i(-3)),
        |_| v(i(-3)),
    ));
    add(case(
        "DOMAIN",
        "i32 -7 % 2",
        IDiv,
        (1, i(-7), 2),
        v(i(-1)),
        |_| v(i(-1)),
    ));
    add(case(
        "DOMAIN",
        "i32 MIN / 1",
        IDiv,
        (0, i(i32::MIN), 1),
        v(i(i32::MIN)),
        |_| v(i(i32::MIN)),
    ));
    add(case(
        "DOMAIN",
        "u32 1 << 31",
        UShift,
        (0, 1, 31),
        v(0x8000_0000),
        |_| v(0x8000_0000),
    ));
    add(case(
        "DOMAIN",
        "u32 0x8000_0000 >> 31",
        UShift,
        (1, 0x8000_0000, 31),
        v(1),
        |_| v(1),
    ));
    add(case(
        "DOMAIN",
        "i32 1 << 30",
        IShift,
        (0, 1, 30),
        v(0x4000_0000),
        |_| v(0x4000_0000),
    ));
    add(case(
        "DOMAIN",
        "i32 -8 >> 31",
        IShift,
        (1, i(-8), 31),
        v(i(-1)),
        |_| v(i(-1)),
    ));
    add(case(
        "DOMAIN",
        "i32 -8 >> 1",
        IShift,
        (1, i(-8), 1),
        v(i(-4)),
        |_| v(i(-4)),
    ));
    add(case("DOMAIN", "u32 2 + 3", UArith, (0, 2, 3), v(5), |_| {
        v(5)
    }));
    add(case("DOMAIN", "u32 5 - 3", UArith, (1, 5, 3), v(2), |_| {
        v(2)
    }));
    add(case(
        "DOMAIN",
        "u32 6 * 7",
        UArith,
        (2, 6, 7),
        v(42),
        |_| v(42),
    ));
    add(case(
        "DOMAIN",
        "i32 -5 + 3",
        IArith,
        (0, i(-5), 3),
        v(i(-2)),
        |_| v(i(-2)),
    ));
    add(case(
        "DOMAIN",
        "i32 3 - 5",
        IArith,
        (1, 3, 5),
        v(i(-2)),
        |_| v(i(-2)),
    ));
    add(case(
        "DOMAIN",
        "i32 -6 * 7",
        IArith,
        (2, i(-6), 7),
        v(i(-42)),
        |_| v(i(-42)),
    ));
    add(case(
        "DOMAIN",
        "i32 -(5)",
        IArith,
        (3, 5, 0),
        v(i(-5)),
        |_| v(i(-5)),
    ));
    add(case(
        "DOMAIN",
        "f32 3.99 as i32",
        CastInt,
        (0, f(3.99), 0),
        v(3),
        |_| v(3),
    ));
    add(case(
        "DOMAIN",
        "f32 -3.99 as i32",
        CastInt,
        (0, f(-3.99), 0),
        v(i(-3)),
        |_| v(i(-3)),
    ));
    add(case(
        "DOMAIN",
        "f32 3.99 as u32",
        CastUint,
        (0, f(3.99), 0),
        v(3),
        |_| v(3),
    ));
    add(case(
        "DOMAIN",
        "f32 0.0 as u32",
        CastUint,
        (0, f(0.0), 0),
        v(0),
        |_| v(0),
    ));

    add(case(
        "DOMAIN",
        "f32 -1.0 as i32",
        CastInt,
        (0, f(-1.0), 0),
        v(i(-1)),
        |_| v(i(-1)),
    ));
    // ---- KM-01 division by zero ----
    add(case("KM-01", "u32 7 / 0", UDiv, (0, 7, 0), v(7), |_| {
        p(DIV_ZERO)
    }));
    add(case(
        "KM-01",
        "u32 MAX / 0",
        UDiv,
        (0, u32::MAX, 0),
        v(u32::MAX),
        |_| p(DIV_ZERO),
    ));
    add(case(
        "KM-01",
        "i32 -7 / 0",
        IDiv,
        (0, i(-7), 0),
        v(i(-7)),
        |_| p(DIV_ZERO),
    ));
    // ---- KM-02 remainder by zero ----
    add(case("KM-02", "u32 7 % 0", UDiv, (1, 7, 0), v(0), |_| {
        p(REM_ZERO)
    }));
    add(case(
        "KM-02",
        "i32 -7 % 0",
        IDiv,
        (1, i(-7), 0),
        v(0),
        |_| p(REM_ZERO),
    ));
    // ---- KM-03 MIN / -1 and MIN % -1 ----
    add(case(
        "KM-03",
        "i32 MIN / -1",
        IDiv,
        (0, i(i32::MIN), i(-1)),
        v(i(i32::MIN)),
        |_| p(DIV_OVERFLOW),
    ));
    add(case(
        "KM-03",
        "i32 MIN % -1",
        IDiv,
        (1, i(i32::MIN), i(-1)),
        v(0),
        |_| p(REM_OVERFLOW),
    ));

    // ---- KM-04 shift amounts of 32 or more ----
    add(case(
        "KM-04",
        "u32 1 << 32",
        UShift,
        (0, 1, 32),
        v(1),
        |m| m.pick(p(SHL_OVERFLOW), v(1)),
    ));
    add(case(
        "KM-04",
        "u32 1 << 33",
        UShift,
        (0, 1, 33),
        v(2),
        |m| m.pick(p(SHL_OVERFLOW), v(2)),
    ));
    add(case(
        "KM-04",
        "u32 1 << 64",
        UShift,
        (0, 1, 64),
        v(1),
        |m| m.pick(p(SHL_OVERFLOW), v(1)),
    ));
    add(case(
        "KM-04",
        "u32 0x8000_0000 >> 32",
        UShift,
        (1, 0x8000_0000, 32),
        v(0x8000_0000),
        |m| m.pick(p(SHR_OVERFLOW), v(0x8000_0000)),
    ));
    add(case(
        "KM-04",
        "u32 0x8000_0000 >> 33",
        UShift,
        (1, 0x8000_0000, 33),
        v(0x4000_0000),
        |m| m.pick(p(SHR_OVERFLOW), v(0x4000_0000)),
    ));
    add(case(
        "KM-04",
        "u32 0x8000_0000 >> 64",
        UShift,
        (1, 0x8000_0000, 64),
        v(0x8000_0000),
        |m| m.pick(p(SHR_OVERFLOW), v(0x8000_0000)),
    ));
    add(case(
        "KM-04",
        "i32 1 << 32",
        IShift,
        (0, 1, 32),
        v(1),
        |m| m.pick(p(SHL_OVERFLOW), v(1)),
    ));
    add(case(
        "KM-04",
        "i32 -8 >> 32",
        IShift,
        (1, i(-8), 32),
        v(i(-8)),
        |m| m.pick(p(SHR_OVERFLOW), v(i(-8))),
    ));
    add(case(
        "KM-04",
        "i32 -8 >> 33",
        IShift,
        (1, i(-8), 33),
        v(i(-4)),
        |m| m.pick(p(SHR_OVERFLOW), v(i(-4))),
    ));
    add(case(
        "KM-04",
        "i32 -8 >> 64",
        IShift,
        (1, i(-8), 64),
        v(i(-8)),
        |m| m.pick(p(SHR_OVERFLOW), v(i(-8))),
    ));

    // ---- KM-05 f32 as i32 out of range, NaN ----
    add(case(
        "KM-05",
        "f32 1.0e10 as i32",
        CastInt,
        (0, f(1.0e10), 0),
        v(0x7fff_ff80),
        |_| v(i(i32::MAX)),
    ));
    add(case(
        "KM-05",
        "f32 -1.0e10 as i32",
        CastInt,
        (0, f(-1.0e10), 0),
        v(i(i32::MIN)),
        |_| v(i(i32::MIN)),
    ));
    add(case(
        "KM-05",
        "f32 NaN as i32",
        CastInt,
        (0, NAN, 0),
        v(0x8000_0000),
        |_| v(0),
    ));
    add(case(
        "KM-05",
        "f32 2147483648.0 (2^31) as i32",
        CastInt,
        (0, f(2_147_483_648.0), 0),
        v(0x7fff_ff80),
        |_| v(i(i32::MAX)),
    ));
    // ---- KM-06 f32 as u32 of a negative, NaN or too large value ----
    add(case(
        "KM-06",
        "f32 -1.0 as u32",
        CastUint,
        (0, f(-1.0), 0),
        v(0),
        |_| v(0),
    ));
    add(case(
        "KM-06",
        "f32 -3.99 as u32",
        CastUint,
        (0, f(-3.99), 0),
        v(0),
        |_| v(0),
    ));
    add(case(
        "KM-06",
        "f32 NaN as u32",
        CastUint,
        (0, NAN, 0),
        v(0),
        |_| v(0),
    ));
    add(case(
        "KM-06",
        "f32 1.0e10 as u32",
        CastUint,
        (0, f(1.0e10), 0),
        v(0xffff_ff00),
        |_| v(u32::MAX),
    ));
    add(case(
        "KM-06",
        "f32 -1.0e10 as u32",
        CastUint,
        (0, f(-1.0e10), 0),
        v(0),
        |_| v(0),
    ));
    add(case(
        "KM-06",
        "f32 4294967296.0 (2^32) as u32",
        CastUint,
        (0, f(4_294_967_296.0), 0),
        v(0xffff_ff00),
        |_| v(u32::MAX),
    ));

    // ---- KM-10 overflow of + - * and negation ----
    add(case(
        "KM-10",
        "u32 MAX + 1",
        UArith,
        (0, u32::MAX, 1),
        v(0),
        |m| m.pick(p(ADD_OVERFLOW), v(0)),
    ));
    add(case(
        "KM-10",
        "i32 MAX + 1",
        IArith,
        (0, i(i32::MAX), 1),
        v(i(i32::MIN)),
        |m| m.pick(p(ADD_OVERFLOW), v(i(i32::MIN))),
    ));
    add(case(
        "KM-10",
        "u32 0 - 1",
        UArith,
        (1, 0, 1),
        v(u32::MAX),
        |m| m.pick(p(SUB_OVERFLOW), v(u32::MAX)),
    ));
    add(case(
        "KM-10",
        "i32 MIN - 1",
        IArith,
        (1, i(i32::MIN), 1),
        v(i(i32::MAX)),
        |m| m.pick(p(SUB_OVERFLOW), v(i(i32::MAX))),
    ));
    add(case(
        "KM-10",
        "u32 MAX * 2",
        UArith,
        (2, u32::MAX, 2),
        v(u32::MAX - 1),
        |m| m.pick(p(MUL_OVERFLOW), v(u32::MAX - 1)),
    ));
    add(case(
        "KM-10",
        "i32 MAX * 2",
        IArith,
        (2, i(i32::MAX), 2),
        v(i(-2)),
        |m| m.pick(p(MUL_OVERFLOW), v(i(-2))),
    ));
    add(case(
        "KM-10",
        "-(i32 MIN)",
        IArith,
        (3, i(i32::MIN), 0),
        v(i(i32::MIN)),
        |m| m.pick(p(NEG_OVERFLOW), v(i(i32::MIN))),
    ));

    // ---- the wrapping forms of KM-10 (slice m7.w): inside the domain for every input ----
    // `wrapping_add`, `wrapping_sub`, `wrapping_mul` and `wrapping_neg` give the same wrapped value on the GPU, in
    // plain Rust and on the twin, with checks on and with checks off, so every row is a `DOMAIN` control. The
    // expected values are written out; `rust_observed` computes each one again with Rust's own `wrapping_*`.
    macro_rules! wrapping_row {
        ($kind:ident, $what:expr, $op:expr, $a:expr, $b:expr, $want:expr) => {
            add(case(
                "DOMAIN",
                $what,
                $kind,
                ($op, $a, $b),
                v($want),
                |_| v($want),
            ))
        };
    }
    wrapping_row!(UWrap, "u32 2.wrapping_add(3)", 0, 2, 3, 5);
    wrapping_row!(UWrap, "u32 MAX.wrapping_add(1)", 0, u32::MAX, 1, 0);
    wrapping_row!(
        UWrap,
        "u32 MAX.wrapping_add(MAX)",
        0,
        u32::MAX,
        u32::MAX,
        u32::MAX - 1
    );
    wrapping_row!(UWrap, "u32 0.wrapping_sub(1)", 1, 0, 1, u32::MAX);
    wrapping_row!(UWrap, "u32 0.wrapping_sub(MAX)", 1, 0, u32::MAX, 1);
    wrapping_row!(UWrap, "u32 MAX.wrapping_sub(MAX)", 1, u32::MAX, u32::MAX, 0);
    wrapping_row!(
        UWrap,
        "u32 MAX.wrapping_mul(2)",
        2,
        u32::MAX,
        2,
        u32::MAX - 1
    );
    wrapping_row!(UWrap, "u32 MAX.wrapping_mul(MAX)", 2, u32::MAX, u32::MAX, 1);
    wrapping_row!(UWrap, "u32 0.wrapping_mul(MAX)", 2, 0, u32::MAX, 0);
    wrapping_row!(UWrap, "u32 65536.wrapping_mul(65536)", 2, 65_536, 65_536, 0);
    wrapping_row!(UWrap, "u32 0.wrapping_neg()", 3, 0, 0, 0);
    wrapping_row!(UWrap, "u32 1.wrapping_neg()", 3, 1, 0, u32::MAX);
    wrapping_row!(UWrap, "u32 MAX.wrapping_neg()", 3, u32::MAX, 0, 1);
    wrapping_row!(
        UWrap,
        "u32 2^31.wrapping_neg()",
        3,
        0x8000_0000,
        0,
        0x8000_0000
    );
    wrapping_row!(IWrap, "i32 -2.wrapping_add(-3)", 0, i(-2), i(-3), i(-5));
    wrapping_row!(
        IWrap,
        "i32 MAX.wrapping_add(1)",
        0,
        i(i32::MAX),
        1,
        i(i32::MIN)
    );
    wrapping_row!(
        IWrap,
        "i32 MIN.wrapping_add(-1)",
        0,
        i(i32::MIN),
        i(-1),
        i(i32::MAX)
    );
    wrapping_row!(
        IWrap,
        "i32 MIN.wrapping_add(MIN)",
        0,
        i(i32::MIN),
        i(i32::MIN),
        0
    );
    wrapping_row!(IWrap, "i32 -1.wrapping_add(-1)", 0, i(-1), i(-1), i(-2));
    wrapping_row!(
        IWrap,
        "i32 MIN.wrapping_sub(1)",
        1,
        i(i32::MIN),
        1,
        i(i32::MAX)
    );
    wrapping_row!(
        IWrap,
        "i32 0.wrapping_sub(MIN)",
        1,
        0,
        i(i32::MIN),
        i(i32::MIN)
    );
    wrapping_row!(
        IWrap,
        "i32 MAX.wrapping_sub(-1)",
        1,
        i(i32::MAX),
        i(-1),
        i(i32::MIN)
    );
    wrapping_row!(
        IWrap,
        "i32 MIN.wrapping_sub(MAX)",
        1,
        i(i32::MIN),
        i(i32::MAX),
        1
    );
    wrapping_row!(IWrap, "i32 MAX.wrapping_mul(2)", 2, i(i32::MAX), 2, i(-2));
    wrapping_row!(
        IWrap,
        "i32 MIN.wrapping_mul(-1)",
        2,
        i(i32::MIN),
        i(-1),
        i(i32::MIN)
    );
    wrapping_row!(
        IWrap,
        "i32 MIN.wrapping_mul(MIN)",
        2,
        i(i32::MIN),
        i(i32::MIN),
        0
    );
    wrapping_row!(
        IWrap,
        "i32 MAX.wrapping_mul(MAX)",
        2,
        i(i32::MAX),
        i(i32::MAX),
        1
    );
    wrapping_row!(IWrap, "i32 -1.wrapping_mul(-1)", 2, i(-1), i(-1), 1);
    wrapping_row!(IWrap, "i32 MIN.wrapping_mul(2)", 2, i(i32::MIN), 2, 0);
    wrapping_row!(
        IWrap,
        "i32 MIN.wrapping_neg()",
        3,
        i(i32::MIN),
        0,
        i(i32::MIN)
    );
    wrapping_row!(IWrap, "i32 0.wrapping_neg()", 3, 0, 0, 0);
    wrapping_row!(IWrap, "i32 -1.wrapping_neg()", 3, i(-1), 0, 1);
    wrapping_row!(
        IWrap,
        "i32 MAX.wrapping_neg()",
        3,
        i(i32::MAX),
        0,
        i(i32::MIN) + 1
    );

    // ---- DOMAIN controls (floats): exactly representable values, normal results ----
    add(case(
        "DOMAIN",
        "f32 1.5 + 2.25",
        FloatOp,
        (0, f(1.5), f(2.25)),
        v(f(3.75)),
        |_| v(f(3.75)),
    ));
    add(case(
        "DOMAIN",
        "f32 1.0 - 2.0",
        FloatOp,
        (1, f(1.0), f(2.0)),
        v(f(-1.0)),
        |_| v(f(-1.0)),
    ));
    add(case(
        "DOMAIN",
        "f32 2.0 * 3.0",
        FloatOp,
        (2, f(2.0), f(3.0)),
        v(f(6.0)),
        |_| v(f(6.0)),
    ));
    add(case(
        "DOMAIN",
        "f32 1.0 / 4.0",
        FloatOp,
        (3, f(1.0), f(4.0)),
        v(f(0.25)),
        |_| v(f(0.25)),
    ));
    add(case(
        "DOMAIN",
        "f32 1.0 == 1.0",
        FloatCmp,
        (0, f(1.0), f(1.0)),
        v(1),
        |_| v(1),
    ));
    add(case(
        "DOMAIN",
        "f32 1.0 != 1.0",
        FloatCmp,
        (1, f(1.0), f(1.0)),
        v(0),
        |_| v(0),
    ));
    add(case(
        "DOMAIN",
        "f32 1.0 < 2.0",
        FloatCmp,
        (2, f(1.0), f(2.0)),
        v(1),
        |_| v(1),
    ));
    add(case(
        "DOMAIN",
        "f32 3.0 > 2.0",
        FloatCmp,
        (3, f(3.0), f(2.0)),
        v(1),
        |_| v(1),
    ));
    add(case(
        "DOMAIN",
        "f32 !(1.0 < 2.0)",
        FloatCmp,
        (4, f(1.0), f(2.0)),
        v(0),
        |_| v(0),
    ));
    add(case(
        "DOMAIN",
        "f32 0.0 == 0.0",
        FloatCmp,
        (0, f(0.0), f(0.0)),
        v(1),
        |_| v(1),
    ));
    add(case(
        "DOMAIN",
        "f32 0.0 < 1.0",
        FloatCmp,
        (2, f(0.0), f(1.0)),
        v(1),
        |_| v(1),
    ));
    add(case(
        "DOMAIN",
        "f32 !(1.0 < 1.0)",
        FloatCmp,
        (4, f(1.0), f(1.0)),
        v(1),
        |_| v(1),
    ));

    // ---- KM-07 NaN and infinity through arithmetic and comparison ----
    add(case(
        "KM-07",
        "f32 NaN + 1.0",
        FloatOp,
        (0, NAN, f(1.0)),
        Outcome::Nan,
        |_| Outcome::Nan,
    )
    .software(v(0x7fc00000)));
    add(case(
        "KM-07",
        "f32 +inf + 1.0",
        FloatOp,
        (0, INF, f(1.0)),
        v(INF),
        |_| v(INF),
    ));
    add(case(
        "KM-07",
        "f32 -inf + 1.0",
        FloatOp,
        (0, NEG_INF, f(1.0)),
        v(NEG_INF),
        |_| v(NEG_INF),
    ));
    add(case(
        "KM-07",
        "f32 NaN * 0.0",
        FloatOp,
        (2, NAN, f(0.0)),
        Outcome::Nan,
        |_| Outcome::Nan,
    )
    .software(v(0x7fc00000)));
    add(case(
        "KM-07",
        "f32 +inf * 0.0",
        FloatOp,
        (2, INF, f(0.0)),
        Outcome::Nan,
        |_| Outcome::Nan,
    )
    .software(v(0xffc00000)));
    add(case(
        "KM-07",
        "f32 -inf * 0.0",
        FloatOp,
        (2, NEG_INF, f(0.0)),
        Outcome::Nan,
        |_| Outcome::Nan,
    )
    .software(v(0xffc00000)));
    add(case(
        "KM-07",
        "f32 NaN - NaN",
        FloatOp,
        (1, NAN, NAN),
        Outcome::Nan,
        |_| Outcome::Nan,
    )
    .software(v(0x7fc00000)));
    add(case(
        "KM-07",
        "f32 +inf - +inf",
        FloatOp,
        (1, INF, INF),
        Outcome::Nan,
        |_| Outcome::Nan,
    )
    .software(v(0xffc00000)));
    add(case(
        "KM-07",
        "f32 NaN / NaN",
        FloatOp,
        (3, NAN, NAN),
        Outcome::Nan,
        |_| Outcome::Nan,
    )
    .software(v(0x7fc00000)));
    add(case(
        "KM-07",
        "f32 +inf / +inf",
        FloatOp,
        (3, INF, INF),
        Outcome::Nan,
        |_| Outcome::Nan,
    )
    .software(v(0xffc00000)));
    add(case(
        "KM-07",
        "f32 0.0 / 0.0",
        FloatOp,
        (3, f(0.0), f(0.0)),
        Outcome::Nan,
        |_| Outcome::Nan,
    )
    .software(v(0xffc00000)));
    add(case(
        "KM-07",
        "f32 1.0 / 0.0",
        FloatOp,
        (3, f(1.0), f(0.0)),
        v(INF),
        |_| v(INF),
    ));
    add(case(
        "KM-07",
        "f32 -1.0 / 0.0",
        FloatOp,
        (3, f(-1.0), f(0.0)),
        v(NEG_INF),
        |_| v(NEG_INF),
    ));
    add(case(
        "KM-07",
        "f32 3.0e38 * 10.0",
        FloatOp,
        (2, f(3.0e38), f(10.0)),
        v(INF),
        |_| v(INF),
    ));
    add(case(
        "KM-07",
        "f32 -inf - -inf",
        FloatOp,
        (1, NEG_INF, NEG_INF),
        Outcome::Nan,
        |_| Outcome::Nan,
    )
    .software(v(0xffc00000)));
    add(case(
        "KM-07",
        "f32 -inf / -inf",
        FloatOp,
        (3, NEG_INF, NEG_INF),
        Outcome::Nan,
        |_| Outcome::Nan,
    )
    .software(v(0xffc00000)));
    // The remaining pre-registered inputs (0.0, 1.0, 3.0e38) through the same operations. Their
    // results are finite or (0.0 / 0.0, above) NaN, so they are controls.
    macro_rules! control {
        ($what:expr, $kind:expr, $operands:expr, $result:expr) => {
            add(case("DOMAIN", $what, $kind, $operands, v($result), |_| {
                v($result)
            }))
        };
    }
    control!("f32 0.0 + 1.0", FloatOp, (0, f(0.0), f(1.0)), f(1.0));
    control!("f32 0.0 * 0.0", FloatOp, (2, f(0.0), f(0.0)), f(0.0));
    control!("f32 0.0 - 0.0", FloatOp, (1, f(0.0), f(0.0)), f(0.0));
    control!("f32 0.0 != 0.0", FloatCmp, (1, f(0.0), f(0.0)), 0);
    control!("f32 0.0 > 1.0", FloatCmp, (3, f(0.0), f(1.0)), 0);
    control!("f32 !(0.0 < 1.0)", FloatCmp, (4, f(0.0), f(1.0)), 0);
    control!("f32 1.0 + 1.0", FloatOp, (0, f(1.0), f(1.0)), f(2.0));
    control!("f32 1.0 * 0.0", FloatOp, (2, f(1.0), f(0.0)), f(0.0));
    control!("f32 1.0 - 1.0", FloatOp, (1, f(1.0), f(1.0)), f(0.0));
    control!("f32 1.0 / 1.0", FloatOp, (3, f(1.0), f(1.0)), f(1.0));
    control!("f32 1.0 < 1.0", FloatCmp, (2, f(1.0), f(1.0)), 0);
    control!("f32 1.0 > 1.0", FloatCmp, (3, f(1.0), f(1.0)), 0);
    control!(
        "f32 3.0e38 + 1.0",
        FloatOp,
        (0, f(3.0e38), f(1.0)),
        f(3.0e38)
    );
    control!("f32 3.0e38 * 0.0", FloatOp, (2, f(3.0e38), f(0.0)), f(0.0));
    control!(
        "f32 3.0e38 - 3.0e38",
        FloatOp,
        (1, f(3.0e38), f(3.0e38)),
        f(0.0)
    );
    // Found by S2 (pre-registered input 3.0e38): the divisor is above 2^126, outside WGSL's division
    // accuracy range, and the GPU returned 0, not 1. Unspecified, so a KM-07 observation.
    add(case(
        "KM-07",
        "f32 3.0e38 / 3.0e38 (divisor above 2^126)",
        FloatOp,
        (3, f(3.0e38), f(3.0e38)),
        v(0),
        |_| v(f(1.0)),
    )
    .software(v(0x3f800000)));
    control!(
        "f32 3.0e38 == 3.0e38",
        FloatCmp,
        (0, f(3.0e38), f(3.0e38)),
        1
    );
    control!(
        "f32 3.0e38 != 3.0e38",
        FloatCmp,
        (1, f(3.0e38), f(3.0e38)),
        0
    );
    control!("f32 3.0e38 < 1.0", FloatCmp, (2, f(3.0e38), f(1.0)), 0);
    control!("f32 3.0e38 > 1.0", FloatCmp, (3, f(3.0e38), f(1.0)), 1);
    control!("f32 !(3.0e38 < 1.0)", FloatCmp, (4, f(3.0e38), f(1.0)), 1);
    // Comparisons: `f == f`, `f != f`, `f < 1.0`, `f > 1.0`, `!(f < 1.0)`.
    add(case(
        "KM-07",
        "f32 NaN == NaN",
        FloatCmp,
        (0, NAN, NAN),
        v(0),
        |_| v(0),
    ));
    add(case(
        "KM-07",
        "f32 NaN != NaN",
        FloatCmp,
        (1, NAN, NAN),
        v(0),
        |_| v(1),
    ));
    add(case(
        "KM-07",
        "f32 NaN < 1.0",
        FloatCmp,
        (2, NAN, f(1.0)),
        v(0),
        |_| v(0),
    ));
    add(case(
        "KM-07",
        "f32 NaN > 1.0",
        FloatCmp,
        (3, NAN, f(1.0)),
        v(0),
        |_| v(0),
    ));
    add(case(
        "KM-07",
        "f32 !(NaN < 1.0)",
        FloatCmp,
        (4, NAN, f(1.0)),
        v(1),
        |_| v(1),
    ));
    add(case(
        "KM-07",
        "f32 +inf == +inf",
        FloatCmp,
        (0, INF, INF),
        v(1),
        |_| v(1),
    ));
    add(case(
        "KM-07",
        "f32 +inf != +inf",
        FloatCmp,
        (1, INF, INF),
        v(0),
        |_| v(0),
    ));
    add(case(
        "KM-07",
        "f32 +inf < 1.0",
        FloatCmp,
        (2, INF, f(1.0)),
        v(0),
        |_| v(0),
    ));
    add(case(
        "KM-07",
        "f32 +inf > 1.0",
        FloatCmp,
        (3, INF, f(1.0)),
        v(1),
        |_| v(1),
    ));
    add(case(
        "KM-07",
        "f32 !(+inf < 1.0)",
        FloatCmp,
        (4, INF, f(1.0)),
        v(1),
        |_| v(1),
    ));
    add(case(
        "KM-07",
        "f32 -inf == -inf",
        FloatCmp,
        (0, NEG_INF, NEG_INF),
        v(1),
        |_| v(1),
    ));
    add(case(
        "KM-07",
        "f32 -inf != -inf",
        FloatCmp,
        (1, NEG_INF, NEG_INF),
        v(0),
        |_| v(0),
    ));
    add(case(
        "KM-07",
        "f32 -inf < 1.0",
        FloatCmp,
        (2, NEG_INF, f(1.0)),
        v(1),
        |_| v(1),
    ));
    add(case(
        "KM-07",
        "f32 -inf > 1.0",
        FloatCmp,
        (3, NEG_INF, f(1.0)),
        v(0),
        |_| v(0),
    ));
    add(case(
        "KM-07",
        "f32 !(-inf < 1.0)",
        FloatCmp,
        (4, NEG_INF, f(1.0)),
        v(0),
        |_| v(0),
    ));

    // ---- KM-08 denormal operands and results (Rust keeps them) ----
    add(case(
        "KM-08",
        "f32 1.0e-40 * 1.0",
        FloatOp,
        (2, f(DENORMAL), f(1.0)),
        v(0),
        |_| v((DENORMAL * 1.0).to_bits()),
    )
    .software(v(0x000116c2)));
    add(case(
        "KM-08",
        "f32 1.0e-40 + 0.0",
        FloatOp,
        (0, f(DENORMAL), f(0.0)),
        v(0),
        |_| v((DENORMAL + 0.0).to_bits()),
    )
    .software(v(0x000116c2)));
    add(case(
        "KM-08",
        "f32 1.0e-40 * 2.0",
        FloatOp,
        (2, f(DENORMAL), f(2.0)),
        v(0),
        |_| v((DENORMAL * 2.0).to_bits()),
    )
    .software(v(0x00022d84)));
    add(case(
        "KM-08",
        "f32 1.0e-40 / 2.0",
        FloatOp,
        (3, f(DENORMAL), f(2.0)),
        v(0),
        |_| v((DENORMAL / 2.0).to_bits()),
    )
    .software(v(0x00008b61)));
    add(case(
        "KM-08",
        "f32 1.0e-40 == 0.0",
        FloatCmp,
        (0, f(DENORMAL), f(0.0)),
        v(1),
        |_| v(0),
    )
    .software(v(0x00000000)));
    // Revision 1 wrote `1.0e-38 / 4.0` as "a normal input"; 1.0e-38 is itself below the
    // smallest normal f32 (1.17549435e-38), so the normal input used here is 4.0e-38.
    add(case(
        "KM-08",
        "f32 4.0e-38 / 4.0 (normal input, denormal result)",
        FloatOp,
        (3, f(4.0e-38), f(4.0)),
        v(0),
        |_| v((4.0e-38f32 / 4.0).to_bits()),
    )
    .software(v(0x006ce3ee)));
    assert!(DENORMAL.is_subnormal());

    // ---- KM-11 out-of-bounds index equal to the buffer length ----
    add(case(
        "KM-11",
        "load source[len] (len 4)",
        OobLoad,
        (0, 0, 0),
        v(0),
        |_| p(OOB),
    ));
    add(case(
        "KM-11",
        "store sink[len] = 77 (load in bounds): sink[len - 1] afterwards",
        OobStoreLast,
        (0, 0, 0),
        v(0),
        |_| p(OOB),
    ));
    add(case(
        "KM-11",
        "store sink[len] = 77 (load in bounds): sink[0..len - 1] afterwards (OR of them)",
        OobStoreElsewhere,
        (0, 0, 0),
        v(0),
        |_| p(OOB),
    ));

    // ---- KM-09 contended exchange and compare-exchange (set-valued on the GPU) ----
    add(case(
        "KM-09",
        "257 x atomic_exchange(cell, id): final cell",
        ContendedExchange,
        (0, 0, 0),
        Outcome::Within(0, CONTENDERS - 1),
        |_| v(CONTENDERS - 1),
    ));
    add(case(
        "KM-09",
        "257 x atomic_compare_exchange(cell, 999, id + 1): final cell",
        ContendedCompareExchange,
        (0, 0, 0),
        // 999 stays when every weak attempt fails spuriously (WGSL 17.8.5): part of the spec-allowed set.
        Outcome::WithinOr(1, CONTENDERS, 999),
        |_| v(1),
    ));
    // The same kernel with one invocation, so nothing contends. The weak compare-exchange may still fail spuriously
    // (WGSL 17.8.5; D47): the cell ends at 1 (exchanged) or stays 999. Plain Rust and the twin always end at 1.
    add(case(
        "KM-09",
        "1 x atomic_compare_exchange(cell, 999, id + 1): uncontended success",
        UncontendedCompareExchange,
        (0, 0, 0),
        Outcome::WithinOr(1, 1, 999),
        |_| v(1),
    ));

    // ---- KM-12 a texel address outside the texture ----
    // The texture is 4 x 2 (`TEX_W` x `TEX_H`) and holds 10, 11, .. 17 row-major. The address `(4, 0)` has a flat
    // index (it would be `(0, 1)` of a flat array) and `(0, 2)` has none: both are outside.
    // Measured on an NVIDIA RTX 5090 (Vulkan): a load outside returns 0 and a store outside changes no texel. Both are
    // observations of one adapter kind (D52), not requirements; the twin and plain Rust panic.
    add(case(
        "KM-12",
        "load tex[x = width, y = 0] (4 x 2)",
        TexLoadOob,
        (0, TEX_W, 0),
        v(0),
        |_| p(OOB),
    ));
    add(case(
        "KM-12",
        "load tex[x = 0, y = height] (4 x 2)",
        TexLoadOob,
        (0, 0, TEX_H),
        v(0),
        |_| p(OOB),
    ));
    add(case(
        "KM-12",
        "store tex[x = width, y = 0] = 77: row-major index of the texel holding 77 (none: 0xffffffff)",
        TexStoreOob,
        (0, TEX_W, 0),
        v(u32::MAX),
        |_| p(OOB),
    ));
    add(case(
        "KM-12",
        "store tex[x = 0, y = height] = 77: row-major index of the texel holding 77 (none: 0xffffffff)",
        TexStoreOob,
        (0, 0, TEX_H),
        v(u32::MAX),
        |_| p(OOB),
    ));

    // ---- KM-13 linear sample weights ----
    // A 3 x 2 `r32float` texture, clamp-to-edge: the formula in `f32` (plain Rust and the twin) against the GPU, whose
    // result is pinned as an observation and, on every adapter, held to `linear_error_bound`.
    // Measured: the GPU is within the bound but not equal to the formula at the first probe. That the weights are
    // quantised is an inference from this one value, not a measurement.
    add(case(
        "KM-13",
        "linear sample of a 3 x 2 texture at (0.37, 0.61)",
        LinearSample,
        (0, f(0.37), f(0.61)),
        v(0x4076_0000),
        |_| v(linear_probe(0.37, 0.61).0.to_bits()),
    ));
    add(case(
        "KM-13",
        "linear sample of a 3 x 2 texture at (0.93, 0.08)",
        LinearSample,
        (0, f(0.93), f(0.08)),
        v(0x4080_0000),
        |_| v(linear_probe(0.93, 0.08).0.to_bits()),
    ));

    // ---- KM-14 nearest sample within one ulp of a texel edge ----
    // A 4 x 1 `r32float` texture of 10, 20, 30, 40 under clamp-to-edge. The edges between texels are at u = 0.25, 0.5
    // and 0.75; the formula picks the texel `floor(u * 4)`, so one ulp below an edge is the texel on its left.
    // Measured: the GPU picked the formula's texel on all four probes. A GPU may pick the neighbour (SEMANTICS), so the
    // pin is an observation.
    add(case(
        "KM-14",
        "nearest sample at u = 0.5 - 1 ulp (4 x 1)",
        NearestSample,
        (0, 0.5f32.next_down().to_bits(), 0),
        v(0x41a0_0000),
        |_| v(nearest_probe(0.5f32.next_down()).to_bits()),
    ));
    add(case(
        "KM-14",
        "nearest sample at u = 0.5 exactly (4 x 1)",
        NearestSample,
        (0, f(0.5), 0),
        v(0x41f0_0000),
        |_| v(nearest_probe(0.5).to_bits()),
    ));
    add(case(
        "KM-14",
        "nearest sample at u = 0.75 - 1 ulp (4 x 1)",
        NearestSample,
        (0, 0.75f32.next_down().to_bits(), 0),
        v(0x41f0_0000),
        |_| v(nearest_probe(0.75f32.next_down()).to_bits()),
    ));
    add(case(
        "KM-14",
        "nearest sample at u = 0.25 + 1 ulp (4 x 1)",
        NearestSample,
        (0, 0.25f32.next_up().to_bits(), 0),
        v(0x41a0_0000),
        |_| v(nearest_probe(0.25f32.next_up()).to_bits()),
    ));

    rows
}

// ---- the plain Rust column ----

fn rust_observed(case: &Case) -> Observed {
    let (op, a, b) = case.operands;
    let kind = case.kind;
    let outcome = catch_unwind(AssertUnwindSafe(move || -> u32 {
        let (op, a, b) = (black_box(op), black_box(a), black_box(b));
        match kind {
            Kind::UDiv => {
                if op == 0 {
                    a / b
                } else {
                    a % b
                }
            }
            Kind::IDiv => {
                let (a, b) = (a as i32, b as i32);
                (if op == 0 { a / b } else { a % b }) as u32
            }
            Kind::UShift => {
                if op == 0 {
                    a << b
                } else {
                    a >> b
                }
            }
            Kind::IShift => {
                let a = a as i32;
                (if op == 0 { a << b } else { a >> b }) as u32
            }
            Kind::UArith => match op {
                0 => a + b,
                1 => a - b,
                _ => a * b,
            },
            Kind::IArith => {
                let (a, b) = (a as i32, b as i32);
                (match op {
                    0 => a + b,
                    1 => a - b,
                    2 => a * b,
                    _ => -a,
                }) as u32
            }
            Kind::UWrap => match op {
                0 => a.wrapping_add(b),
                1 => a.wrapping_sub(b),
                2 => a.wrapping_mul(b),
                _ => a.wrapping_neg(),
            },
            Kind::IWrap => {
                let (a, b) = (a as i32, b as i32);
                (match op {
                    0 => a.wrapping_add(b),
                    1 => a.wrapping_sub(b),
                    2 => a.wrapping_mul(b),
                    _ => a.wrapping_neg(),
                }) as u32
            }
            Kind::CastInt => (f32::from_bits(a) as i32) as u32,
            Kind::CastUint => f32::from_bits(a) as u32,
            Kind::FloatOp => {
                let (a, b) = (f32::from_bits(a), f32::from_bits(b));
                match op {
                    0 => a + b,
                    1 => a - b,
                    2 => a * b,
                    _ => a / b,
                }
                .to_bits()
            }
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            Kind::FloatCmp => {
                let (a, b) = (f32::from_bits(a), f32::from_bits(b));
                u32::from(match op {
                    0 => a == b,
                    1 => a != b,
                    2 => a < b,
                    3 => a > b,
                    _ => !(a < b),
                })
            }
            Kind::OobLoad => {
                let source = black_box(vec![10u32, 11, 12, 13]);
                source[black_box(source.len())]
            }
            Kind::OobStoreLast | Kind::OobStoreElsewhere => {
                let mut sink = black_box(vec![0u32; 4]);
                let index = black_box(sink.len());
                sink[index] = 77;
                sink[3]
            }
            // A texture as nested vectors (the plain Rust column of KM-12): the row, then the column.
            Kind::TexLoadOob => {
                let rows = black_box(texel_rows());
                rows[b as usize][a as usize]
            }
            Kind::TexStoreOob => {
                let mut rows = black_box(texel_rows());
                rows[b as usize][a as usize] = 77;
                row_major_index_of_77(&rows)
            }
            Kind::LinearSample => linear_probe(f32::from_bits(a), f32::from_bits(b))
                .0
                .to_bits(),
            Kind::NearestSample => nearest_probe(f32::from_bits(a)).to_bits(),
            // The sequential host loop: ascending ids, as the CPU twin runs them.
            Kind::ContendedExchange => {
                let mut cell = black_box(9999u32);
                for id in 0..CONTENDERS {
                    cell = black_box(id);
                }
                cell
            }
            Kind::ContendedCompareExchange => {
                let mut cell = black_box(999u32);
                for id in 0..CONTENDERS {
                    if cell == 999 {
                        cell = black_box(id + 1);
                    }
                }
                cell
            }
            Kind::UncontendedCompareExchange => {
                let mut cell = black_box(999u32);
                if cell == 999 {
                    cell = black_box(1u32);
                }
                cell
            }
        }
    }));
    match outcome {
        Ok(bits) => Observed::Value(bits),
        Err(payload) => Observed::Panic(panic_message(payload.as_ref())),
    }
}

// ---- the CPU twin column ----

/// Dispatches `kernel(op, a, b, out)` on the twin for one case (one element, so one workgroup) and returns the result
/// as bits. A call the twin refuses is a test defect and panics with its own message.
fn twin_ops<A: Bits, B: Bits, O: Bits>(
    (op, a, b): (u32, u32, u32),
    dispatch: impl FnOnce(
        &CpuDevice,
        &CpuBuffer<u32>,
        &CpuBuffer<A>,
        &CpuBuffer<B>,
        &CpuBuffer<O>,
    ) -> Result<(), CpuError>,
) -> u32 {
    let cpu = CpuDevice::new();
    let ops = cpu.buffer(&[op]);
    let first = cpu.buffer(&[A::from_bits32(a)]);
    let second = cpu.buffer(&[B::from_bits32(b)]);
    let out = cpu.buffer(&[O::from_bits32(0)]);
    dispatch(&cpu, &ops, &first, &second, &out)
        .unwrap_or_else(|error| panic!("the twin refused the call: {error}"));
    out.read()[0].to_bits32()
}

/// Dispatches `kernel(f, out)` on the twin for one case.
fn twin_casts<O: Bits>(
    bits: u32,
    dispatch: impl FnOnce(&CpuDevice, &CpuBuffer<f32>, &CpuBuffer<O>) -> Result<(), CpuError>,
) -> u32 {
    let cpu = CpuDevice::new();
    let floats = cpu.buffer(&[f32::from_bits(bits)]);
    let out = cpu.buffer(&[O::from_bits32(0)]);
    dispatch(&cpu, &floats, &out)
        .unwrap_or_else(|error| panic!("the twin refused the call: {error}"));
    out.read()[0].to_bits32()
}

/// KM-11 on the twin: the same kernel and buffers as `run_out_of_bounds`. Returns what the GPU column extracts:
/// the loaded value and the sink contents (only reached when the twin does not panic).
fn twin_out_of_bounds(source_len: usize) -> (u32, [u32; 4]) {
    let cpu = CpuDevice::new();
    let idx = cpu.buffer(&[OOB_INDEX]);
    let source = cpu.buffer(&oob_source(source_len));
    let sink = cpu.buffer(&[0u32; 4]);
    let seen = cpu.buffer(&[0xAAAA_AAAAu32]);
    mismatch::out_of_bounds::dispatch(
        &cpu,
        1,
        Independent(&idx),
        Independent(&source),
        Independent(&sink),
        Independent(&seen),
    )
    .unwrap_or_else(|error| panic!("the twin refused the call: {error}"));
    let sink = sink.read();
    (seen.read()[0], [sink[0], sink[1], sink[2], sink[3]])
}

/// KM-09 on the twin: the final cell of a `count`-way dispatch (257 contended, or 1 uncontended), run in ascending
/// invocation order.
fn twin_contended(
    initial: u32,
    count: u32,
    dispatch: impl FnOnce(&CpuDevice, u32, Independent<&CpuBuffer<u32>>) -> Result<(), CpuError>,
) -> u32 {
    let cpu = CpuDevice::new();
    let cell = cpu.buffer(&[initial]);
    dispatch(&cpu, count, Independent(&cell))
        .unwrap_or_else(|error| panic!("the twin refused the call: {error}"));
    cell.read()[0]
}

/// What the CPU twin does for one case: the macro-emitted `Run` impl of the same kernel, called through the typed
/// `dispatch` exactly as the GPU column calls it, under `catch_unwind`.
fn twin_observed(case: &Case) -> Observed {
    let (kind, operands) = (case.kind, case.operands);
    let outcome = catch_unwind(AssertUnwindSafe(move || -> u32 {
        match kind {
            Kind::UDiv => twin_ops::<u32, u32, u32>(operands, |d, op, a, b, out| {
                mismatch::udivide::dispatch(d, 1, op, a, b, out)
            }),
            Kind::IDiv => twin_ops::<i32, i32, i32>(operands, |d, op, a, b, out| {
                mismatch::idivide::dispatch(d, 1, op, a, b, out)
            }),
            Kind::UShift => twin_ops::<u32, u32, u32>(operands, |d, op, a, n, out| {
                mismatch::ushift::dispatch(d, 1, op, a, n, out)
            }),
            Kind::IShift => twin_ops::<i32, u32, i32>(operands, |d, op, a, n, out| {
                mismatch::ishift::dispatch(d, 1, op, a, n, out)
            }),
            Kind::UArith => twin_ops::<u32, u32, u32>(operands, |d, op, a, b, out| {
                mismatch::uarith::dispatch(d, 1, op, a, b, out)
            }),
            Kind::IArith => twin_ops::<i32, i32, i32>(operands, |d, op, a, b, out| {
                mismatch::iarith::dispatch(d, 1, op, a, b, out)
            }),
            Kind::UWrap => twin_ops::<u32, u32, u32>(operands, |d, op, a, b, out| {
                mismatch::uwrap::dispatch(d, 1, op, a, b, out)
            }),
            Kind::IWrap => twin_ops::<i32, i32, i32>(operands, |d, op, a, b, out| {
                mismatch::iwrap::dispatch(d, 1, op, a, b, out)
            }),
            Kind::CastInt => twin_casts::<i32>(operands.1, |d, f, out| {
                mismatch::cast_int::dispatch(d, 1, f, out)
            }),
            Kind::CastUint => twin_casts::<u32>(operands.1, |d, f, out| {
                mismatch::cast_uint::dispatch(d, 1, f, out)
            }),
            Kind::FloatOp => twin_ops::<f32, f32, f32>(operands, |d, op, a, b, out| {
                mismatch::float_op::dispatch(d, 1, op, a, b, out)
            }),
            Kind::FloatCmp => twin_ops::<f32, f32, u32>(operands, |d, op, a, b, out| {
                mismatch::float_cmp::dispatch(d, 1, op, a, b, out)
            }),
            Kind::OobLoad => twin_out_of_bounds(OOB_LOAD_SOURCE_LEN).0,
            Kind::OobStoreLast => twin_out_of_bounds(OOB_STORE_SOURCE_LEN).1[3],
            Kind::OobStoreElsewhere => {
                let sink = twin_out_of_bounds(OOB_STORE_SOURCE_LEN).1;
                sink[0] | sink[1] | sink[2]
            }
            Kind::TexLoadOob => twin_texel_load(operands.1, operands.2),
            Kind::TexStoreOob => twin_texel_store(operands.1, operands.2),
            Kind::LinearSample => twin_sample(
                Filter::Linear,
                f32::from_bits(operands.1),
                f32::from_bits(operands.2),
            ),
            Kind::NearestSample => twin_sample(Filter::Nearest, f32::from_bits(operands.1), 0.5),
            Kind::ContendedExchange => twin_contended(9999, CONTENDERS, |d, count, cell| {
                mismatch::contend_exchange::dispatch(d, count, cell)
            }),
            Kind::ContendedCompareExchange => twin_contended(999, CONTENDERS, |d, count, cell| {
                mismatch::contend_compare_exchange::dispatch(d, count, cell)
            }),
            Kind::UncontendedCompareExchange => twin_contended(999, 1, |d, count, cell| {
                mismatch::contend_compare_exchange::dispatch(d, count, cell)
            }),
        }
    }));
    match outcome {
        Ok(bits) => Observed::Value(bits),
        Err(payload) => Observed::Panic(panic_message(payload.as_ref())),
    }
}

// ---- KM-12 to KM-14: the texture probes (slice mx.t4) ----

/// The 4 x 2 texture of KM-12.
const TEX_W: u32 = 4;
const TEX_H: u32 = 2;

/// The texel `(x, y)` of the KM-12 texture: 10 plus the row-major index.
fn texel_value(x: u32, y: u32) -> u32 {
    10 + y * TEX_W + x
}

/// The KM-12 texture as nested vectors, row then column.
fn texel_rows() -> Vec<Vec<u32>> {
    (0..TEX_H)
        .map(|y| (0..TEX_W).map(|x| texel_value(x, y)).collect())
        .collect()
}

/// The KM-12 texture row-major.
fn texel_contents() -> Vec<u32> {
    texel_rows().into_iter().flatten().collect()
}

/// The row-major index of the first texel that holds 77, or `u32::MAX` when none does.
fn row_major_index_of_77(rows: &[Vec<u32>]) -> u32 {
    rows.iter()
        .flatten()
        .position(|&texel| texel == 77)
        .map_or(u32::MAX, |index| index as u32)
}

/// The 3 x 2 texture of KM-13, row-major.
const LINEAR_W: u32 = 3;
const LINEAR_H: u32 = 2;
const LINEAR_TEXELS: [f32; 6] = [0.0, 10.0, 4.0, 6.0, 1.0, 12.5];

/// The 4 x 1 texture of KM-14.
const NEAREST_W: u32 = 4;
const NEAREST_TEXELS: [f32; 4] = [10.0, 20.0, 30.0, 40.0];

/// KM-13 in plain Rust: the linear formula of docs/SEMANTICS.md (clamp-to-edge) in `f32` in its written order, and the
/// four texels it reads (the input of `linear_error_bound`).
fn linear_probe(u: f32, v: f32) -> (f32, [f32; 4]) {
    let x = u.clamp(0.0, 1.0) * LINEAR_W as f32 - 0.5;
    let y = v.clamp(0.0, 1.0) * LINEAR_H as f32 - 0.5;
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let at = |i: f32, j: f32| {
        let column = i.clamp(0.0, (LINEAR_W - 1) as f32) as usize;
        let row = j.clamp(0.0, (LINEAR_H - 1) as f32) as usize;
        LINEAR_TEXELS[row * LINEAR_W as usize + column]
    };
    let (t00, t10) = (at(x0, y0), at(x0 + 1.0, y0));
    let (t01, t11) = (at(x0, y0 + 1.0), at(x0 + 1.0, y0 + 1.0));
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    (
        lerp(lerp(t00, t10, fx), lerp(t01, t11, fx), fy),
        [t00, t10, t01, t11],
    )
}

/// KM-14 in plain Rust: the texel `min(floor(u * 4), 3)` of the 4 x 1 texture, clamp-to-edge.
fn nearest_probe(u: f32) -> f32 {
    let x = u.clamp(0.0, 1.0) * NEAREST_W as f32;
    NEAREST_TEXELS[(x.floor() as usize).min(NEAREST_W as usize - 1)]
}

/// The texture of a sampling probe: KM-13 (linear) or KM-14 (nearest), with its size.
fn probe_texture(filter: Filter) -> (&'static [f32], u32, u32) {
    match filter {
        Filter::Linear => (&LINEAR_TEXELS, LINEAR_W, LINEAR_H),
        Filter::Nearest => (&NEAREST_TEXELS, NEAREST_W, 1),
        other => panic!("the ledger has no probe for the filter {other:?}"),
    }
}

/// KM-12 on the twin: `load` at `(x, y)` of the KM-12 texture. The twin panics outside it, so a value is only
/// returned inside.
fn twin_texel_load(x: u32, y: u32) -> u32 {
    let cpu = CpuDevice::new();
    let image = cpu
        .texture_2d::<R32Uint>(TEX_W, TEX_H, TextureUse::STORAGE)
        .unwrap();
    image.write(&texel_contents()).unwrap();
    let at = cpu.buffer(&[x, y]);
    let seen = cpu.buffer(&[0xAAAA_AAAAu32]);
    texture_edges::load_texel::dispatch(&cpu, 1, Independent(&at), &image, Independent(&seen))
        .unwrap_or_else(|error| panic!("the twin refused the call: {error}"));
    seen.read()[0]
}

/// KM-12 on the twin: `store` of 77 at `(x, y)`; the row-major index of the texel that holds 77.
fn twin_texel_store(x: u32, y: u32) -> u32 {
    let cpu = CpuDevice::new();
    let image = cpu
        .texture_2d::<R32Uint>(TEX_W, TEX_H, TextureUse::STORAGE)
        .unwrap();
    image.write(&texel_contents()).unwrap();
    let at = cpu.buffer(&[x, y]);
    texture_edges::store_texel::dispatch(&cpu, 1, Independent(&at), &image)
        .unwrap_or_else(|error| panic!("the twin refused the call: {error}"));
    row_major_index_of_77(&[image.read()])
}

/// KM-13 and KM-14 on the twin: one `sample` of the probe's texture, as bits.
fn twin_sample(filter: Filter, u: f32, v: f32) -> u32 {
    let cpu = CpuDevice::new();
    let (texels, width, height) = probe_texture(filter);
    let image = cpu
        .texture_2d::<R32Float>(width, height, TextureUse::SAMPLED)
        .unwrap();
    image.write(texels).unwrap();
    let sampler = cpu.sampler(SamplerOptions::new(filter, Address::ClampToEdge));
    let at = cpu.buffer(&[u, v]);
    let out = cpu.buffer(&[f32::from_bits(0xAAAA_AAAA)]);
    texture_edges::sample_at::dispatch(&cpu, 1, &image, &sampler, Independent(&at), &out)
        .unwrap_or_else(|error| panic!("the twin refused the call: {error}"));
    out.read()[0].to_bits()
}

/// KM-12 on the GPU: `load` at `(x, y)`; the loaded texel (`0xAAAAAAAA` if the kernel never wrote it).
fn gpu_texel_load(device: &HeadlessDevice, x: u32, y: u32) -> u32 {
    let image = device
        .create_texture_2d::<R32Uint>(TEX_W, TEX_H, TextureUse::STORAGE)
        .unwrap();
    device.write_texture(&image, &texel_contents()).unwrap();
    let at = device.create_typed_buffer("texel at", &[x, y], GpuBufferAccess::ReadOnly);
    let seen =
        device.create_typed_buffer("texel seen", &[0xAAAA_AAAAu32], GpuBufferAccess::ReadWrite);
    texture_edges::load_texel::dispatch(device, 1, Independent(&at), &image, Independent(&seen))
        .unwrap();
    device.read_typed_buffer(&seen).unwrap()[0]
}

/// KM-12 on the GPU: `store` of 77 at `(x, y)`; the row-major index of the texel that holds 77 (`u32::MAX`: none).
fn gpu_texel_store(device: &HeadlessDevice, x: u32, y: u32) -> u32 {
    let image = device
        .create_texture_2d::<R32Uint>(TEX_W, TEX_H, TextureUse::STORAGE)
        .unwrap();
    device.write_texture(&image, &texel_contents()).unwrap();
    let at = device.create_typed_buffer("texel at", &[x, y], GpuBufferAccess::ReadOnly);
    texture_edges::store_texel::dispatch(device, 1, Independent(&at), &image).unwrap();
    row_major_index_of_77(&[device.read_texture(&image).unwrap()])
}

/// KM-13 and KM-14 on the GPU: one `sample` of the probe's texture, as bits.
fn gpu_sample(device: &HeadlessDevice, filter: Filter, u: f32, v: f32) -> u32 {
    gpu_sample_result(device, filter, u, v).unwrap()
}

/// As `gpu_sample`, but returns the refusal of the dispatch (a linear sampler of a float32 texture on a device
/// without `float32-filterable` is `Error::TextureFilter`).
fn gpu_sample_result(
    device: &HeadlessDevice,
    filter: Filter,
    u: f32,
    v: f32,
) -> Result<u32, gust_wgpu::Error> {
    let (texels, width, height) = probe_texture(filter);
    let image = device
        .create_texture_2d::<R32Float>(width, height, TextureUse::SAMPLED)
        .unwrap();
    device.write_texture(&image, texels).unwrap();
    let sampler = device
        .create_sampler(SamplerOptions::new(filter, Address::ClampToEdge))
        .unwrap();
    let at = device.create_typed_buffer("sample at", &[u, v], GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer(
        "sample out",
        &[f32::from_bits(0xAAAA_AAAA)],
        GpuBufferAccess::ReadWrite,
    );
    texture_edges::sample_at::dispatch(device, 1, &image, &sampler, Independent(&at), &out)?;
    Ok(device.read_typed_buffer(&out).unwrap()[0].to_bits())
}

// ---- the GPU column ----

trait Bits: GpuPod {
    fn from_bits32(bits: u32) -> Self;
    fn to_bits32(self) -> u32;
}

impl Bits for u32 {
    fn from_bits32(bits: u32) -> Self {
        bits
    }
    fn to_bits32(self) -> u32 {
        self
    }
}

impl Bits for i32 {
    fn from_bits32(bits: u32) -> Self {
        bits as i32
    }
    fn to_bits32(self) -> u32 {
        self as u32
    }
}

impl Bits for f32 {
    fn from_bits32(bits: u32) -> Self {
        f32::from_bits(bits)
    }
    fn to_bits32(self) -> u32 {
        self.to_bits()
    }
}

/// Dispatch `kernel(op, a, b, out)` over every case of one kind; one element per case.
fn run_ops<A: Bits, B: Bits, O: Bits>(
    device: &HeadlessDevice,
    kernel: &KernelDescriptor,
    cases: &[&Case],
) -> Vec<u32> {
    let ops: Vec<u32> = cases.iter().map(|c| c.operands.0).collect();
    let first: Vec<A> = cases.iter().map(|c| A::from_bits32(c.operands.1)).collect();
    let second: Vec<B> = cases.iter().map(|c| B::from_bits32(c.operands.2)).collect();
    let ops = device.create_typed_buffer("mismatch op", &ops, GpuBufferAccess::ReadOnly);
    let first = device.create_typed_buffer("mismatch a", &first, GpuBufferAccess::ReadOnly);
    let second = device.create_typed_buffer("mismatch b", &second, GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer(
        "mismatch out",
        &vec![O::from_bits32(0); cases.len()],
        GpuBufferAccess::ReadWrite,
    );
    device
        .dispatch_buffers(
            kernel,
            cases.len() as u32,
            &[
                BufferBinding::read_only(&ops),
                BufferBinding::read_only(&first),
                BufferBinding::read_only(&second),
                BufferBinding::read_write(&out),
            ],
        )
        .unwrap();
    device
        .read_typed_buffer(&out)
        .unwrap()
        .into_iter()
        .map(Bits::to_bits32)
        .collect()
}

/// Dispatch `kernel(f, out)` over every case of one kind.
fn run_casts<O: Bits>(
    device: &HeadlessDevice,
    kernel: &KernelDescriptor,
    cases: &[&Case],
) -> Vec<u32> {
    let floats: Vec<f32> = cases.iter().map(|c| f32::from_bits(c.operands.1)).collect();
    let floats = device.create_typed_buffer("mismatch f", &floats, GpuBufferAccess::ReadOnly);
    let out = device.create_typed_buffer(
        "mismatch cast out",
        &vec![O::from_bits32(0); cases.len()],
        GpuBufferAccess::ReadWrite,
    );
    device
        .dispatch_buffers(
            kernel,
            cases.len() as u32,
            &[
                BufferBinding::read_only(&floats),
                BufferBinding::read_write(&out),
            ],
        )
        .unwrap();
    device
        .read_typed_buffer(&out)
        .unwrap()
        .into_iter()
        .map(Bits::to_bits32)
        .collect()
}

/// KM-11: the index equals the length of `sink` (4); `source` has `source_len` elements. Returns the loaded value
/// (`seen[0]`, 0xAAAA_AAAA if the invocation never reached the store) and the sink contents.
fn run_out_of_bounds(device: &HeadlessDevice, source_len: usize) -> (u32, [u32; 4]) {
    let idx = device.create_typed_buffer("oob idx", &[OOB_INDEX], GpuBufferAccess::ReadOnly);
    let source = device.create_typed_buffer(
        "oob source",
        &oob_source(source_len),
        GpuBufferAccess::ReadOnly,
    );
    let sink = device.create_typed_buffer("oob sink", &[0u32; 4], GpuBufferAccess::ReadWrite);
    let seen =
        device.create_typed_buffer("oob seen", &[0xAAAA_AAAAu32], GpuBufferAccess::ReadWrite);
    device
        .dispatch_buffers(
            &mismatch::out_of_bounds::DESCRIPTOR,
            1,
            &[
                BufferBinding::read_only(&idx).independent_length(),
                BufferBinding::read_only(&source).independent_length(),
                BufferBinding::read_write(&sink).independent_length(),
                BufferBinding::read_write(&seen).independent_length(),
            ],
        )
        .unwrap();
    let sink = device.read_typed_buffer(&sink).unwrap();
    (
        device.read_typed_buffer(&seen).unwrap()[0],
        [sink[0], sink[1], sink[2], sink[3]],
    )
}

/// KM-09: final cell values of repeated `count`-way dispatches (257 contended, or 1 uncontended).
fn run_contended(
    device: &HeadlessDevice,
    kernel: &KernelDescriptor,
    initial: u32,
    count: u32,
) -> Vec<u32> {
    (0..CONTENDED_RUNS)
        .map(|_| {
            let cell = device.create_typed_buffer("cell", &[initial], GpuBufferAccess::ReadWrite);
            device
                .dispatch_buffers(
                    kernel,
                    count,
                    &[BufferBinding::read_write(&cell).independent_length()],
                )
                .unwrap();
            device.read_typed_buffer(&cell).unwrap()[0]
        })
        .collect()
}

/// All GPU observations, aligned with `cases` (several per case for the set-valued KM-09).
fn gpu_observed(device: &HeadlessDevice, cases: &[Case]) -> Vec<Vec<u32>> {
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); cases.len()];
    let of_kind = |kind: Kind| -> Vec<usize> {
        (0..cases.len())
            .filter(|&n| cases[n].kind == kind)
            .collect()
    };
    let mut put = |indices: &[usize], values: Vec<u32>| {
        for (&n, value) in indices.iter().zip(values) {
            out[n] = vec![value];
        }
    };
    let pick = |indices: &[usize]| -> Vec<&Case> { indices.iter().map(|&n| &cases[n]).collect() };

    let n = of_kind(Kind::UDiv);
    put(
        &n,
        run_ops::<u32, u32, u32>(device, &mismatch::udivide::DESCRIPTOR, &pick(&n)),
    );
    let n = of_kind(Kind::IDiv);
    put(
        &n,
        run_ops::<i32, i32, i32>(device, &mismatch::idivide::DESCRIPTOR, &pick(&n)),
    );
    let n = of_kind(Kind::UShift);
    put(
        &n,
        run_ops::<u32, u32, u32>(device, &mismatch::ushift::DESCRIPTOR, &pick(&n)),
    );
    let n = of_kind(Kind::IShift);
    put(
        &n,
        run_ops::<i32, u32, i32>(device, &mismatch::ishift::DESCRIPTOR, &pick(&n)),
    );
    let n = of_kind(Kind::UArith);
    put(
        &n,
        run_ops::<u32, u32, u32>(device, &mismatch::uarith::DESCRIPTOR, &pick(&n)),
    );
    let n = of_kind(Kind::IArith);
    put(
        &n,
        run_ops::<i32, i32, i32>(device, &mismatch::iarith::DESCRIPTOR, &pick(&n)),
    );
    let n = of_kind(Kind::UWrap);
    put(
        &n,
        run_ops::<u32, u32, u32>(device, &mismatch::uwrap::DESCRIPTOR, &pick(&n)),
    );
    let n = of_kind(Kind::IWrap);
    put(
        &n,
        run_ops::<i32, i32, i32>(device, &mismatch::iwrap::DESCRIPTOR, &pick(&n)),
    );
    let n = of_kind(Kind::CastInt);
    put(
        &n,
        run_casts::<i32>(device, &mismatch::cast_int::DESCRIPTOR, &pick(&n)),
    );
    let n = of_kind(Kind::CastUint);
    put(
        &n,
        run_casts::<u32>(device, &mismatch::cast_uint::DESCRIPTOR, &pick(&n)),
    );
    let n = of_kind(Kind::FloatOp);
    put(
        &n,
        run_ops::<f32, f32, f32>(device, &mismatch::float_op::DESCRIPTOR, &pick(&n)),
    );
    let n = of_kind(Kind::FloatCmp);
    put(
        &n,
        run_ops::<f32, f32, u32>(device, &mismatch::float_cmp::DESCRIPTOR, &pick(&n)),
    );

    // The load row: both accesses are out of bounds (the load first). The store rows: the load is in bounds, so the
    // store is the only access out of bounds and its effect on `sink` is what is observed.
    let (seen, _) = run_out_of_bounds(device, OOB_LOAD_SOURCE_LEN);
    let (stored_seen, sink) = run_out_of_bounds(device, OOB_STORE_SOURCE_LEN);
    eprintln!(
        "S2 raw KM-11: load row seen[0] = 0x{seen:08x}; store rows seen[0] = 0x{stored_seen:08x}, sink = {sink:08x?}"
    );
    for n in of_kind(Kind::OobLoad) {
        out[n] = vec![seen];
    }
    for n in of_kind(Kind::OobStoreLast) {
        out[n] = vec![sink[3]];
    }
    for n in of_kind(Kind::OobStoreElsewhere) {
        out[n] = vec![sink[0] | sink[1] | sink[2]];
    }
    // KM-12 to KM-14 (slice mx.t4). A linear read of a 32-bit float texture needs `float32-filterable`, which this
    // ledger's device does not enable: the linear rows run on a second device of the same adapter that opened with
    // `DeviceOptions::default()`. An adapter without the feature cannot be measured for KM-13, and the test says so.
    for n in of_kind(Kind::TexLoadOob) {
        let (_, x, y) = cases[n].operands;
        out[n] = vec![gpu_texel_load(device, x, y)];
    }
    for n in of_kind(Kind::TexStoreOob) {
        let (_, x, y) = cases[n].operands;
        out[n] = vec![gpu_texel_store(device, x, y)];
    }
    for n in of_kind(Kind::NearestSample) {
        let u = f32::from_bits(cases[n].operands.1);
        out[n] = vec![gpu_sample(device, Filter::Nearest, u, 0.5)];
    }
    let linear = of_kind(Kind::LinearSample);
    if !linear.is_empty() {
        let filtering = HeadlessDevice::with_options(DeviceOptions::default())
            .expect("Vulkan adapter required; this test must not silently skip");
        for n in linear {
            let (_, u, v) = cases[n].operands;
            let (u, v) = (f32::from_bits(u), f32::from_bits(v));
            if filtering.float32_filterable() {
                out[n] = vec![gpu_sample(&filtering, Filter::Linear, u, v)];
            } else {
                // KM-13 is not measurable here: the refusal is what the specification of this crate says happens
                // (`textures.rs` asserts it too). The row has no observation; the test prints a NOTE and enforces
                // no bound, because there is no value to hold to it.
                let refused = gpu_sample_result(&filtering, Filter::Linear, u, v);
                assert!(
                    matches!(refused, Err(gust_wgpu::Error::TextureFilter { .. })),
                    "a linear sampler of a float32 texture without float32-filterable must be Error::TextureFilter: {refused:?}"
                );
                out[n] = Vec::new();
            }
        }
    }
    for n in of_kind(Kind::ContendedExchange) {
        out[n] = run_contended(
            device,
            &mismatch::contend_exchange::DESCRIPTOR,
            9999,
            CONTENDERS,
        );
    }
    for n in of_kind(Kind::ContendedCompareExchange) {
        out[n] = run_contended(
            device,
            &mismatch::contend_compare_exchange::DESCRIPTOR,
            999,
            CONTENDERS,
        );
    }
    for n in of_kind(Kind::UncontendedCompareExchange) {
        out[n] = run_contended(
            device,
            &mismatch::contend_compare_exchange::DESCRIPTOR,
            999,
            1,
        );
    }
    out
}

// ---- the tests ----

/// Whether the pinned GPU value of a case is a requirement the test enforces (per case). WGSL
/// defines the result of integer division and remainder by zero, `MIN / -1`, shift amounts of
/// 32 or more and integer overflow (KM-01 to KM-04, KM-10), bounds the set-valued KM-09 outcome,
/// and defines every finite float-to-integer conversion (section 15.7.6: the value in the target
/// type closest to the truncated source that the source float type represents exactly, e.g.
/// `2147483520i`, `4294967040u`, `-1e20f` gives `i32::MIN`): those are enforced, including the
/// finite KM-05 and KM-06 cases. Only the NaN conversions are indeterminate, and NaN or infinity
/// results and comparisons (KM-07), denormal handling (KM-08), out-of-bounds access (KM-11), a
/// texel address outside a texture (KM-12), the weights of a linear sample (KM-13) and a nearest
/// sample within an ulp of a texel edge (KM-14) are unspecified or implementation-dependent (KM-13
/// is also held to `linear_error_bound` on every adapter, see the test): the pinned GPU value there is a measured
/// one-adapter observation per adapter kind, and a different value is reported as a NOTE, never a failure
/// (owner ruling D39). The plain Rust column is always enforced.
fn gpu_is_enforced(id: &str, kind: Kind, first: u32) -> bool {
    match id {
        "KM-07" | "KM-08" | "KM-11" | "KM-12" | "KM-13" | "KM-14" => false,
        "KM-05" | "KM-06" => {
            !(matches!(kind, Kind::CastInt | Kind::CastUint) && f32::from_bits(first).is_nan())
        }
        _ => true,
    }
}

fn gpu_summary(values: &[u32]) -> String {
    if values.is_empty() {
        return "not measurable".to_owned();
    }
    let distinct: BTreeSet<u32> = values.iter().copied().collect();
    if distinct.len() == 1 {
        format!("0x{:08x}", values[0])
    } else {
        format!(
            "{} runs, {} distinct values, min {}, max {}",
            values.len(),
            distinct.len(),
            distinct.first().unwrap(),
            distinct.last().unwrap()
        )
    }
}

/// The ledger file of an adapter kind: one `LEDGER` line per case, written by the ledger test below (R1-5). Cargo
/// creates `CARGO_TARGET_TMPDIR` (`target/tmp/`) for integration tests.
fn ledger_path(kind: AdapterKind) -> &'static str {
    match kind {
        AdapterKind::Hardware => concat!(env!("CARGO_TARGET_TMPDIR"), "/ledger-hardware.txt"),
        AdapterKind::Software => concat!(env!("CARGO_TARGET_TMPDIR"), "/ledger-software.txt"),
    }
}

#[test]
fn known_mismatches_pin_the_gpu_the_plain_rust_and_the_cpu_twin_outcome() {
    let mode = overflow_checks();
    let device =
        HeadlessDevice::new().expect("Vulkan adapter required; this test must not silently skip");
    let info = device.adapter_info();
    let kind = AdapterKind::of(info.device_type);
    let identity = format!(
        "driver={:?} backend={:?} driver_info={:?}",
        info.driver, info.backend, info.driver_info
    );
    eprintln!(
        "S2 adapter: name={:?} backend={:?} device_type={:?} driver={:?} driver_info={:?} kind={}",
        info.name,
        info.backend,
        info.device_type,
        info.driver,
        info.driver_info,
        kind.name()
    );
    eprintln!("S2 overflow checks detected in this build: {mode:?}");

    let cases = cases();
    let observed = gpu_observed(&device, &cases);
    let mut failures = Vec::new();
    let mut ledger = Vec::new();
    for (case, gpu) in cases.iter().zip(&observed) {
        let failures_before = failures.len();
        let rust = rust_observed(case);
        let twin = twin_observed(case);
        let rust_expected = (case.rust)(mode);
        // An enforced row is checked against its one pin on every adapter. An unenforced row is compared with the
        // running kind's own pin; a row without one has no software pin (the hardware pin accepted
        // m5.b0's value), and the observation is then only compared with the hardware pin to tell whether it is new.
        let pin = case.pin(kind);
        let reference = pin.unwrap_or(case.gpu);
        let gpu_ok = gpu
            .iter()
            .all(|&bits| reference.accepts(&Observed::Value(bits)));
        let rust_ok = rust_expected.accepts(&rust);
        // The twin runs the shadow body compiled in this crate, so it must do what plain Rust does here.
        let twin_ok = rust_expected.accepts(&twin);
        let enforced = case.enforced;
        eprintln!(
            "S2 | {} | {} | gpu expected {:?} observed {} {} | rust[{mode:?}] expected {:?} observed {} {} | twin observed {} {}",
            case.id,
            case.what,
            reference,
            gpu_summary(gpu),
            if gpu_ok { "ok" } else { "DEVIATES" },
            rust_expected,
            show(&rust),
            if rust_ok { "ok" } else { "DEVIATES" },
            show(&twin),
            if twin_ok { "ok" } else { "DEVIATES" },
        );
        let mut noted = false;
        if gpu.is_empty() {
            // No observation (KM-13 on an adapter without float32-filterable): reported, never failed.
            noted = true;
            eprintln!(
                "NOTE {} `{}`: not measurable on this adapter (no float32-filterable; the TextureFilter refusal was asserted); {identity}",
                case.id, case.what
            );
        }
        if !gpu_ok && !enforced {
            // Owner ruling D39: an unspecified outcome is a per-kind observation, not a requirement. A change is
            // reported, never failed.
            noted = true;
            if pin.is_some() {
                eprintln!(
                    "NOTE {} `{}`: the {} observation changed (was {}, now {}); unspecified by WGSL, not enforced; {identity}",
                    case.id,
                    case.what,
                    kind.name(),
                    pin_text(reference),
                    gpu_summary(gpu)
                );
            } else {
                eprintln!(
                    "NOTE {} `{}`: no {} pin (the hardware pin accepted m5.b0's value); observed {}, which the hardware pin {} does not accept; unspecified by WGSL, not enforced; {identity}",
                    case.id,
                    case.what,
                    kind.name(),
                    gpu_summary(gpu),
                    pin_text(reference)
                );
            }
        }
        if !gpu_ok && enforced {
            failures.push(format!(
                "{} `{}`: GPU expected {:?}, observed {}",
                case.id,
                case.what,
                case.gpu,
                gpu_summary(gpu)
            ));
        }
        if !rust_ok {
            failures.push(format!(
                "{} `{}`: plain Rust [{mode:?}] expected {:?}, observed {}",
                case.id,
                case.what,
                rust_expected,
                show(&rust)
            ));
        }
        if !twin_ok {
            failures.push(format!(
                "{} `{}`: CPU twin [{mode:?}] expected the plain-Rust outcome {:?}, observed {}",
                case.id,
                case.what,
                rust_expected,
                show(&twin)
            ));
        }
        if case.kind == Kind::LinearSample && !gpu.is_empty() {
            // The error domain of docs/SEMANTICS.md holds on every adapter, whatever the pin says (D52: a
            // specification bound is not an observation).
            let (_, u, v) = case.operands;
            let (formula, corners) = linear_probe(f32::from_bits(u), f32::from_bits(v));
            let bound = linear_error_bound(corners);
            let observed = f32::from_bits(gpu[0]);
            let distance = (observed - formula).abs();
            let within = distance <= bound;
            if !within {
                failures.push(format!(
                    "{} `{}`: the GPU returned {observed:e}, {distance:e} from the formula {formula:e}, outside the error domain {bound:e}",
                    case.id, case.what
                ));
            }
        }
        if case.id == "DOMAIN" {
            // Inside the defined domain nothing may differ: the pins themselves must agree.
            for each in [OverflowChecks::Checked, OverflowChecks::Wrapping] {
                if (case.rust)(each) != case.gpu {
                    failures.push(format!(
                        "DOMAIN `{}`: the pinned GPU and Rust outcomes differ in {each:?} mode",
                        case.what
                    ));
                }
            }
        }
        if enforced && case.software.is_some() {
            failures.push(format!(
                "{} `{}`: an enforced row has one pin for every adapter and takes no software observation",
                case.id, case.what
            ));
        }
        let status = if failures.len() > failures_before {
            "fail"
        } else if noted {
            "note"
        } else {
            "ok"
        };
        ledger.push(format!(
            "LEDGER kind={} driver={:?} backend={:?} version={:?} row={} what={:?} observed={:?} status={status}",
            kind.name(),
            info.driver,
            info.backend,
            info.driver_info,
            case.id,
            case.what,
            gpu_summary(gpu),
        ));
    }

    // One line per case, written before anything can fail so that the file survives a red run, and counted back
    // from disk.
    let path = ledger_path(kind);
    std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap())
        .expect("create the ledger directory");
    std::fs::write(path, ledger.join("\n") + "\n").expect("write the ledger file");
    let written = std::fs::read_to_string(path).expect("read the ledger file back");
    assert_eq!(
        written.lines().count(),
        cases.len(),
        "the ledger file {path} must hold exactly one line per case"
    );
    eprintln!("S2 ledger file: {} lines in {path}", cases.len());

    wrapping_edge_grid(&device, &mut failures);

    assert!(
        failures.is_empty(),
        "{} ledger expectation(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The ids of the table rows of `docs/SEMANTICS.md`: lines that start with `| KM-nn |`.
fn documented_ids() -> BTreeSet<String> {
    include_str!("../../../docs/SEMANTICS.md")
        .lines()
        .filter_map(|line| line.strip_prefix("| KM-"))
        .map(|rest| format!("KM-{}", rest.split(' ').next().unwrap_or_default()))
        .collect()
}

fn pinned_ids() -> BTreeSet<String> {
    cases()
        .iter()
        .map(|case| case.id)
        .filter(|id| id.starts_with("KM-"))
        .map(str::to_owned)
        .collect()
}

#[test]
fn every_pinned_ledger_id_is_documented_and_every_documented_id_is_pinned() {
    let documented = documented_ids();
    let pinned = pinned_ids();
    assert_eq!(
        documented, pinned,
        "docs/SEMANTICS.md ledger ids and the ids pinned in tests/mismatch.rs differ"
    );
    assert_eq!(pinned.len(), 14, "KM-01 to KM-14");
    // Every row has at least one case that is not a control, and ids follow KM-nn.
    for id in &pinned {
        assert!(
            id.len() == 5 && id[3..].chars().all(|c| c.is_ascii_digit()),
            "{id}"
        );
    }
}

/// The wrapping methods over a grid of edge values (slice m7.w): every operation on every pair of {0, 1, 2, MAX - 1,
/// MAX, 2^31 - 1, 2^31} for `u32` and {MIN, MIN + 1, -2, -1, 0, 1, MAX} for `i32`, so that sums, differences and
/// products wrap in both directions. The GPU, the twin and Rust's own `wrapping_*` (the plain Rust column) must give
/// one value, and so must an oracle that shares nothing with them: exact integer arithmetic in `i128` reduced
/// modulo 2^32. Part of the ledger test, so it runs in both overflow modes of that test and in the release ledger
/// (`check-ledger`) without a ledger name of its own to register. A disagreement goes into the ledger's `failures`,
/// so the test reports a count and the KM rows are still checked.
fn wrapping_edge_grid(device: &HeadlessDevice, failures: &mut Vec<String>) {
    let unsigned_edges = [0, 1, 2, u32::MAX - 1, u32::MAX, 0x7fff_ffff, 0x8000_0000];
    let signed_edges = [i32::MIN, i32::MIN + 1, -2, -1, 0, 1, i32::MAX].map(i);
    for (kind, edges, signed) in [
        (Kind::UWrap, unsigned_edges, false),
        (Kind::IWrap, signed_edges, true),
    ] {
        let mut grid = Vec::new();
        for op in 0..4 {
            for &a in &edges {
                for &b in &edges {
                    grid.push(case("DOMAIN", "edge grid", kind, (op, a, b), v(0), |_| {
                        v(0)
                    }));
                }
            }
        }
        assert_eq!(grid.len(), 4 * 7 * 7);
        let refs: Vec<&Case> = grid.iter().collect();
        let gpu = if signed {
            run_ops::<i32, i32, i32>(device, &mismatch::iwrap::DESCRIPTOR, &refs)
        } else {
            run_ops::<u32, u32, u32>(device, &mismatch::uwrap::DESCRIPTOR, &refs)
        };
        for (case, gpu) in grid.iter().zip(gpu) {
            let (op, a, b) = case.operands;
            let (x, y) = if signed {
                (i128::from(a as i32), i128::from(b as i32))
            } else {
                (i128::from(a), i128::from(b))
            };
            let exact = match op {
                0 => x + y,
                1 => x - y,
                2 => x * y,
                _ => -x,
            };
            let oracle = exact.rem_euclid(1 << 32) as u32;
            let what = format!("{kind:?} op {op} on 0x{a:08x} and 0x{b:08x}");
            let rust = rust_observed(case);
            if Observed::Value(gpu) != rust {
                failures.push(format!(
                    "GPU, {what}: {:?} vs {rust:?}",
                    Observed::Value(gpu)
                ));
            }
            if twin_observed(case) != rust {
                failures.push(format!(
                    "twin, {what}: {:?} vs {rust:?}",
                    twin_observed(case)
                ));
            }
            if gpu != oracle {
                failures.push(format!(
                    "exact oracle, {what}: 0x{gpu:08x} vs 0x{oracle:08x}"
                ));
            }
        }
    }
}

#[test]
fn the_emitted_wgsl_uses_plain_operators_and_conversions() {
    // The enforced KM rows rest on WGSL's defined results (sections 6.2.3, 8.8, 8.10, 15.7.6),
    // which hold for the WGSL that slangc emits only if it keeps the plain operators. Slang's
    // own language leaves integer division by zero undefined, so this is checked, not assumed.
    //
    // Since slice m6.dv (D74) a `/` or `%` with a divisor that is not a literal is the call `__gust_div(a, b)` or
    // `__gust_rem(a, b)`, whose body is the plain operator, and slangc keeps each helper as a WGSL function
    // (`_gust_div_0`, `_gust_rem_0`): the call stands where the operator stood, and the plain operator is in the
    // helper's body, on the helper's own parameters.
    let wgsl = |kernel: &KernelDescriptor| gust::slang::compile_wgsl(kernel).unwrap();
    // The text of the WGSL function `name`, from its `fn` line to the closing brace.
    fn function<'a>(wgsl: &'a str, name: &str) -> &'a str {
        let start = wgsl
            .find(&format!("fn {name}("))
            .unwrap_or_else(|| panic!("no function {name} in:\n{wgsl}"));
        let end = wgsl[start..].find("\n}").expect("the function ends") + start;
        &wgsl[start..end]
    }
    // The expression a helper assigns to its `ty` local, up to the semicolon: `a_ OP b_` in this order, so a swapped
    // pair of operands fails here as well as in the grid test.
    fn expression<'a>(function: &'a str, ty: &str) -> Option<&'a str> {
        let marker = format!(" : {ty} = ");
        let start = function.find(&marker)? + marker.len();
        let length = function[start..].find(';')?;
        Some(&function[start..start + length])
    }
    for (kernel, ty) in [
        (&mismatch::udivide::DESCRIPTOR, "u32"),
        (&mismatch::idivide::DESCRIPTOR, "i32"),
    ] {
        let divide = wgsl(kernel);
        assert!(
            divide.contains("_gust_div_0(a_0[i_0], b_0[i_0])"),
            "{divide}"
        );
        assert!(
            divide.contains("_gust_rem_0(a_0[i_0], b_0[i_0])"),
            "{divide}"
        );
        let quotient = function(&divide, "_gust_div_0");
        assert!(quotient.contains(&format!("-> {ty}")), "{quotient}");
        assert!(
            expression(quotient, ty)
                .is_some_and(|text| text.starts_with("a_") && text.contains(" / b_")),
            "{quotient}"
        );
        let remainder = function(&divide, "_gust_rem_0");
        assert!(remainder.contains(&format!("-> {ty}")), "{remainder}");
        assert!(
            expression(remainder, ty)
                .is_some_and(|text| text.starts_with("a_") && text.contains(" % b_")),
            "{remainder}"
        );
    }
    for kernel in [&mismatch::ushift::DESCRIPTOR, &mismatch::ishift::DESCRIPTOR] {
        let shift = wgsl(kernel);
        assert!(shift.contains("a_0[i_0] << (n_0[i_0])"), "{shift}");
        assert!(shift.contains("a_0[i_0] >> (n_0[i_0])"), "{shift}");
    }
    assert!(wgsl(&mismatch::cast_int::DESCRIPTOR).contains("i32(f_0[i_0])"));
    assert!(wgsl(&mismatch::cast_uint::DESCRIPTOR).contains("u32(f_0[i_0])"));
    let signed_arithmetic = wgsl(&mismatch::iarith::DESCRIPTOR);
    assert!(signed_arithmetic.contains("a_0[i_0] - b_0[i_0]"));
    assert!(signed_arithmetic.contains("a_0[i_0] * b_0[i_0]"));
    assert!(signed_arithmetic.contains("- a_0[i_0]"));
    // The wrapping methods (slice m7.w) are the same plain operators; the negation is `0 - a`, because WGSL has no
    // negation of a `u32` and the macro sees no types.
    for (kernel, ty) in [
        (&mismatch::uwrap::DESCRIPTOR, "u32"),
        (&mismatch::iwrap::DESCRIPTOR, "i32"),
    ] {
        let wrapping = wgsl(kernel);
        assert!(wrapping.contains("a_0[i_0] + b_0[i_0]"), "{wrapping}");
        assert!(wrapping.contains("a_0[i_0] - b_0[i_0]"), "{wrapping}");
        assert!(wrapping.contains("a_0[i_0] * b_0[i_0]"), "{wrapping}");
        assert!(
            wrapping.contains(&format!("{ty}(0) - a_0[i_0]")),
            "{wrapping}"
        );
        assert!(!wrapping.contains("_gust_"), "{wrapping}");
    }
}

/// The mode that the ledger keys on comes from the consumer-side macro and is this workspace's profile for this
/// crate: `cargo test` is checked and `cargo test --release` wraps, because the root `Cargo.toml` pins neither. The
/// row outcomes that depend on the mode (KM-04, KM-10) are checked by the ledger test above in whichever mode runs.
#[test]
fn the_ledger_mode_is_the_overflow_mode_of_this_crate() {
    let expected = if cfg!(debug_assertions) {
        OverflowChecks::Checked
    } else {
        OverflowChecks::Wrapping
    };
    assert_eq!(overflow_checks(), expected);
    // The macro and the probe of a build agree here: no per-package override exists in this workspace.
    assert_eq!(overflow_checks(), gust::cpu::overflow_checks());
}

// A kernel without storage parameters: the only observable effect of its twin is a panic, so it divides by a value
// that reaches zero at invocation 2 (the shadow body is rustc's own arithmetic, as in the ledger rows above).
#[gust::gpu]
mod zero_storage {
    #[kernel(workgroup_size(1, 1, 1))]
    pub fn divide_by_distance(id: SV_DispatchThreadID) {
        let distance: uint = 2u32 - id.x;
        let _quotient: uint = 10u32 / distance;
    }
}

/// KM-11 on the twin, store rows, without a GPU: `source` is longer than `sink`, so the load `source[4]` is in
/// bounds and the only out-of-bounds access is the store `sink[4] = 77`. The panic therefore comes from the store:
/// `seen` holds the loaded `source[4]` (14, it was 0xAAAA_AAAA) although the invocation panicked, so the load ran
/// and passed first, and `sink` is unchanged. The control with index 3 shows that the same invocation does reach the
/// store and write it when the index is in bounds.
#[test]
fn the_out_of_bounds_store_twin_stops_at_the_store_after_a_load_that_succeeded() {
    fn dispatch_with_index(index: u32) -> (CpuBuffer<u32>, CpuBuffer<u32>, Result<(), String>) {
        let cpu = CpuDevice::new();
        let idx = cpu.buffer(&[index]);
        let source = cpu.buffer(&oob_source(OOB_STORE_SOURCE_LEN));
        let sink = cpu.buffer(&[0u32; 4]);
        let seen = cpu.buffer(&[0xAAAA_AAAAu32]);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            mismatch::out_of_bounds::dispatch(
                &cpu,
                1,
                Independent(&idx),
                Independent(&source),
                Independent(&sink),
                Independent(&seen),
            )
            .unwrap_or_else(|error| panic!("the twin refused the call: {error}"));
        }));
        let outcome = outcome.map_err(|payload| panic_message(payload.as_ref()));
        (seen, sink, outcome)
    }

    let (seen, sink, outcome) = dispatch_with_index(OOB_INDEX);
    let message = outcome.expect_err("the store sink[4] is out of bounds");
    assert!(message.starts_with(OOB), "{message}");
    assert!(
        message.contains("the len is 4 but the index is 4"),
        "{message}"
    );
    assert_eq!(
        seen.read(),
        [14],
        "the load source[4] ran and was in bounds"
    );
    assert_eq!(sink.read(), [0; 4], "the store did not complete");

    let (seen, sink, outcome) = dispatch_with_index(3);
    outcome.expect("index 3 is in bounds for both buffers");
    assert_eq!(seen.read(), [13]);
    assert_eq!(sink.read(), [0, 0, 0, 77], "the store ran");
}

/// The twin of a kernel with no storage parameters (`Run<K, 0>`): the typed `dispatch` infers the device from its
/// first argument, and runs the body. A zero-storage body has no observable except a panic, so the test shows only
/// this: counts 0 and 2 do not reach id 2 and return `Ok`, and count 3 does reach it, where the body divides by zero
/// (KM-01) and panics. It does not show that ids 0 and 1 ran.
#[test]
fn a_kernel_without_storage_parameters_runs_on_the_twin() {
    let cpu = CpuDevice::new();
    zero_storage::divide_by_distance::dispatch(&cpu, 0).unwrap();
    zero_storage::divide_by_distance::dispatch(&cpu, 2).unwrap();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        zero_storage::divide_by_distance::dispatch(&cpu, 3)
    }));
    let message = panic_message(
        outcome
            .expect_err("invocation 2 divides by zero, so the third element must panic")
            .as_ref(),
    );
    assert!(message.starts_with(DIV_ZERO), "{message}");
}
