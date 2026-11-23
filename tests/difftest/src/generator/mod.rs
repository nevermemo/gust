//! The seeded program generator: seed -> byte stream -> AST (the decoding of `arbitrary`, written in place).
//!
//! A program is a pure function of `(seed, attempt)`: `byte_stream` gives `BYTES_PER_PROGRAM` bytes and `decode`
//! turns them into a `Program`; reading past the end of the stream yields zero bytes, which select the first (leaf)
//! option of every choice, so truncating a stream (a shrinking pass of m5.c1c) always decodes to a valid, smaller
//! program. `generate_batch` keeps the programs that the interpreter accepts as in-domain until it has
//! `PROGRAMS_PER_SEED` of them and counts the discards by reason.
//!
//! Anything that changes what a seed decodes to (this module, `prng.rs`, the operator or type lists) is a change of
//! the format and must bump `DIFFTEST_FORMAT` and regenerate `seeds.txt` knowingly.

mod array_gen;
mod base;
mod batch;
mod blocks;
mod calls;
mod closure_gen;
mod decoding;
mod enums;
mod floats_aggs;
mod scalar;
#[cfg(test)]
mod tests;
mod trait_gen;
mod tuples;

use std::collections::BTreeMap;

use crate::ast::{
    Arm, BinOp, ConstDef, Elem, EnumDef, Expr, FnDef, GenParam, Imp, InputBuf, Lane, MAX_ENUMS,
    MAX_FLAT, MAX_TRIPS, MAX_TUPLE_FLAT, MAX_VARIANTS, MathFn, Methods, Pat, Place, Program, Recv,
    Root, Stmt, StructDef, TupleTy, Ty, WIDTHS, chain_terms,
};
use crate::interp;
use crate::prng::{Prng, byte_stream};
use closure_gen::{ArgBias, Cl};

use base::*;
pub use batch::*;
use calls::*;
pub use decoding::*;
use tuples::*;

/// Bump when a change alters what any seed decodes to, or the emitted text. 2: the program space gained `f32`
/// (m5.c1a2, R1-4). 3: the condition and literal biases of the float space (m5.c1b): an `if` condition is never a bare
/// `true` or `false`, it often compares an input element with a value of the data or tests the element index, and a
/// literal is often a value that the data holds, so that both sides of a condition are taken by some elements. 4: no
/// decoding change, a new refusal (m5.c2): an integer `/` or `%` whose divisor slangc folds to zero (the first
/// iteration of the loops, a zero factor, identical operands, the branch that a known condition takes), live or dead,
/// is discarded as `static`, because slangc refuses the kernel (E30002). 5: that refusal is gone (slice m6.dv, D74): the
/// macro lowers a `/` or `%` with a divisor that is not a literal to `__gust_div` and `__gust_rem`, which slangc
/// compiles in every shape, so a program is no longer discarded for a divisor slangc would fold; the decoding is
/// unchanged, the programs that format 4 refused for that reason come back. 6: the vector programs (slice m6.d): a
/// seed's batch is its first ten in-domain programs of the float space (the same programs as in format 5, in the same
/// order) followed by six in-domain programs of the vector space, drawn from the attempt numbers
/// `VECTOR_ATTEMPT_BASE..`; the scalar decoding is unchanged, the last six programs of every seed are new. 7: no
/// decoding change (m8.x1): the result file of every program carries the `capabilities` of its build, a field the reader
/// (`classify::parse_result`) requires, and the reader refuses a file whose `format` is not this number; the emitted text
/// differs from format 6 by its stamp only, so `seeds.txt` changes by its stamp and its hashes.
/// 8: the method programs (slice m9.d1a): a seed's batch is eight float, five vector and three method programs, and a
/// method program (`Space::Method`, attempts from `METHOD_ATTEMPT_BASE`) has module structs, an `impl` with `&self`,
/// `&mut self` and by-value `self` methods and associated functions, and helpers that call them; the decoding of the
/// float and vector spaces is unchanged (a seed's first eight float and first five vector programs are those of
/// format 7), the batch cut is new.
/// 9: enums and the statement `match` (slice m9.d1b, D166): a seed's batch is seven float, five vector, one method and
/// three enum programs; an enum program (`Space::Enum`, attempts from `ENUM_ATTEMPT_BASE`) is a method program with
/// field-less enums (`u32` or `i32` repr, literal discriminants), enum locals, parameters and returns, `==`/`!=` and `as`
/// on them, statement `match` (grouped arms, `_` or the `default` rule, nested, in a `for`, arms that assign, call or
/// `return`), biased toward the order-sensitive shapes. The method space also changed (the biased shift amount is
/// `call & 31`, and same-root `&=` and `>>=` are reachable), so its format-8 programs and pin are re-taken; the float and
/// vector spaces are unchanged (a seed's first seven float and first five vector programs are those of format 8).
/// 10: the value form of `match`, integer patterns and tuples (slice m9.d1c): a seed's batch is six float, five vector,
/// one method, two enum and two value programs. A value program (`Space::Value`, attempts from `VALUE_ATTEMPT_BASE`) is
/// an enum program that also writes `let v: T = match scrut { .. };` (on an enum or on an integer with literal patterns,
/// grouped with `|`, and a final `_`; arms that are expressions, blocks, nested value matches or, in a helper, a
/// `return`) and tuples of two to four elements (locals, parameters and results, `t.0`, element assignment, `f().1`,
/// `let (a, b) = f(..)`), biased toward the order-sensitive shapes. The four older spaces decode as in format 9 (a
/// seed's first six float, five vector, one method and two enum programs are format 9's); the batch cut is new.
/// 11: arrays (slice m9.d1d): a seed's batch is five float, four vector, one method, two enum, two value and two array
/// programs; the array space (`Space::Array`, attempts from `ARRAY_ATTEMPT_BASE`) is the enum space plus fixed-size
/// arrays, with the NVIDIA hole guard (D168).
/// 12: non-capturing closures (slice m9.d1e, D166): a seed's batch is three float, four vector, one method, two enum,
/// two value, two array and two closure programs. A closure program (`Space::Closure`, attempts from
/// `CLOSURE_ATTEMPT_BASE`) is an array program that also binds `let clN = |p: T, ..| -> R { .. };` (typed parameters,
/// written result, block body, immutable binding) in the kernel, helper and method bodies and calls it as `clN(a, b)`
/// wherever a helper call is accepted, biased toward the order-sensitive shapes. A closure body sees its parameters
/// and nothing else, calls helpers and methods but no closure, and holds no barrier, atomic, subgroup operation,
/// resource, `i` or `return`; no capture, `move`, closure value, untyped, unit or nested closure is generated. The
/// older spaces decode as in format 11 (their pins are read with their own stamp); the batch cut is new.
///
/// 13: traits and generic helpers (slice m9.d2, D166, D170): a seed's batch is two float, three vector, one method, two
/// enum, two value, two array, two closure and two trait programs. A trait program (`Space::Trait`, attempts from
/// `TRAIT_ATTEMPT_BASE`) is an enum program (no floats, tuples, arrays or closures) that also declares one to three
/// module traits `Tr<n>` (one or two methods `t<n>m<k>` with a receiver and scalar parameters), each method
/// implemented by exactly two structs or field-less enums, calls them (`s.m(a)`, `Tr::m(&s, a)`,
/// `<S as Tr>::m(&s, a)`) and declares generic helpers and struct methods `fn g<T: Tr0 + Tr1, U>(t: T, ..) -> ..` (inline or `where`
/// bounds, every parameter bounded by one or two module traits, `Copy` sometimes added) called with at least two
/// different types. No turbofish, std bound, arithmetic on `T`, generic trait method, recursion, array or
/// barrier/atomic/subgroup operation is generated. The older spaces decode as in format 12; the batch cut is new.
///
/// 14: the scalar math methods and module constants (slice m11.dt2): a seed's batch is 17 programs: one float, three
/// vector, one method, two enum, two value, two array, two closure, two trait and two math programs (the float part
/// gives up one program). A math program (`Space::Math`, attempts from `MATH_ATTEMPT_BASE`) is a float program (scalars
/// only) that also writes `a.rem_euclid(b)` on `u32` and `i32`, `a.mul_add(b, c)`, `x.abs()`, `x.floor()`, `x.ceil()`,
/// `x.trunc()`, `a.min(b)`, `a.max(b)` and `x.clamp(lo, hi)` on `f32`, and one to three module `const` items
/// (`u32` or `i32`, expressions of literals and earlier constants, D210) that its integer expressions read. The
/// older spaces decode as in format 13 (a seed's first float, vector, method, enum, value, array, closure and trait
/// programs are those of format 13); the batch cut is new.
///
/// 15: the attribute of the emitted modules is spelled `#[gust::gpu_module]` (slice m14.rename, D237); it was
/// `#[gust::gpu]`. The programs are format 14's, and so is every pin of an older format, which is read with the old
/// spelling (`batch_hash_as_format`).
pub const DIFFTEST_FORMAT: u32 = 15;

/// Which programs the generator writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    /// Integers and booleans only. Reads exactly the bytes, and makes exactly the choices, that DIFFTEST_FORMAT 1 did,
    /// so the programs are those of m5.c1a (a unit test pins their discard counts).
    Integer,
    /// Integers, booleans and `f32` (the exact-interval subset of R1-4). The scalar part of a batch.
    Float,
    /// The float space plus `gust::math` vectors and matrices (slice m6.d): `Vec2`/`Vec4`, `IVec2`/`IVec4`,
    /// `UVec2`/`UVec4`, `Mat2`/`Mat4` as locals and intermediates, read back through components, `dot`, and
    /// `Mat * Vec`. The vector part of a batch; selected by an attempt number of `VECTOR_ATTEMPT_BASE` or more.
    Vector,
    /// Integers and booleans with module structs and their methods (slice m9.d1a, format 8): no floats, no vectors.
    /// Selected by an attempt number of `METHOD_ATTEMPT_BASE` or more.
    Method,
    /// The method space plus enums and the statement `match` (slice m9.d1b, format 9). Selected by an attempt number of
    /// `ENUM_ATTEMPT_BASE` or more.
    Enum,
    /// The enum space plus the value `match`, integer patterns and tuples (slice m9.d1c, format 10). Selected by an
    /// attempt number of `VALUE_ATTEMPT_BASE` or more.
    Value,
    /// The enum space plus fixed-size arrays (slice m9.d1d, format 11): typed array locals, literals and repeats,
    /// indexed reads and element sets, array parameters and results, method calls on elements and an indexed store into
    /// the output buffer. No floats, no tuples, no value `match`. Selected by an attempt number of
    /// `ARRAY_ATTEMPT_BASE` or more.
    Array,
    /// The array space plus non-capturing closures (slice m9.d1e, format 12): `let f = |x: T, ..| -> R { .. };` in the
    /// kernel, helper and method bodies, called in every position a helper call is accepted. Selected by an attempt
    /// number of `CLOSURE_ATTEMPT_BASE` or more.
    Closure,
    /// The enum space plus module traits and generic helpers (slice m9.d2, format 13). Selected by an attempt number of
    /// `TRAIT_ATTEMPT_BASE` or more.
    Trait,
    /// The float space (scalars only) plus the scalar math methods and module `const` items (slice m11.dt2, format 14).
    /// Selected by an attempt number of `MATH_ATTEMPT_BASE` or more.
    Math,
}

/// The first attempt number of the vector space. A program is a pure function of `(seed, attempt)`, and the attempt
/// number alone says which space decodes it, so the reducer and the shrinker re-decode a vector program without being
/// told. The scalar part of a batch never gets near it (`MAX_ATTEMPTS` is 100 000).
pub const VECTOR_ATTEMPT_BASE: u64 = 1_000_000;
/// The first attempt number of the method space (above every vector attempt a seed can make).
pub const METHOD_ATTEMPT_BASE: u64 = 2_000_000;
/// In-domain programs of a seed's batch that are scalar (the float space), then vector, then method
/// (`PROGRAMS_PER_SEED` in all).
pub const SCALAR_PROGRAMS_PER_SEED: usize = 1;
pub const VECTOR_PROGRAMS_PER_SEED: usize = 3;
pub const METHOD_PROGRAMS_PER_SEED: usize = 1;
pub const ENUM_PROGRAMS_PER_SEED: usize = 2;
pub const ARRAY_PROGRAMS_PER_SEED: usize = 2;
pub const CLOSURE_PROGRAMS_PER_SEED: usize = 2;
pub const TRAIT_PROGRAMS_PER_SEED: usize = 2;
pub const MATH_PROGRAMS_PER_SEED: usize = 2;
/// The first attempt number of the trait space (above every closure attempt a seed can make).
pub const TRAIT_ATTEMPT_BASE: u64 = 7_000_000;
/// The first attempt number of the math space (above every trait attempt a seed can make).
pub const MATH_ATTEMPT_BASE: u64 = 8_000_000;
pub const VALUE_PROGRAMS_PER_SEED: usize = PROGRAMS_PER_SEED
    - SCALAR_PROGRAMS_PER_SEED
    - VECTOR_PROGRAMS_PER_SEED
    - METHOD_PROGRAMS_PER_SEED
    - ENUM_PROGRAMS_PER_SEED
    - ARRAY_PROGRAMS_PER_SEED
    - CLOSURE_PROGRAMS_PER_SEED
    - TRAIT_PROGRAMS_PER_SEED
    - MATH_PROGRAMS_PER_SEED;
/// The first attempt number of the array space (above every value attempt a seed can make).
pub const ARRAY_ATTEMPT_BASE: u64 = 5_000_000;
/// The first attempt number of the closure space (above every array attempt a seed can make).
pub const CLOSURE_ATTEMPT_BASE: u64 = 6_000_000;
/// The first attempt number of the value space (above every enum attempt a seed can make).
pub const VALUE_ATTEMPT_BASE: u64 = 4_000_000;
/// The first attempt number of the enum space (above every method attempt a seed can make).
pub const ENUM_ATTEMPT_BASE: u64 = 3_000_000;
/// Where the vector programs of a batch end and the method programs begin, where those end and the enum programs begin.
pub const VECTOR_END: usize = SCALAR_PROGRAMS_PER_SEED + VECTOR_PROGRAMS_PER_SEED;
pub const METHOD_END: usize = VECTOR_END + METHOD_PROGRAMS_PER_SEED;
pub const ENUM_END: usize = METHOD_END + ENUM_PROGRAMS_PER_SEED;
pub const VALUE_END: usize = ENUM_END + VALUE_PROGRAMS_PER_SEED;
pub const ARRAY_END: usize = VALUE_END + ARRAY_PROGRAMS_PER_SEED;
/// Where the closure programs end and the trait programs begin, where those end and the math programs begin.
pub const CLOSURE_END: usize = ARRAY_END + CLOSURE_PROGRAMS_PER_SEED;
pub const TRAIT_END: usize = CLOSURE_END + TRAIT_PROGRAMS_PER_SEED;

/// The space whose decoder reads the program of this attempt number.
pub fn space_of_attempt(attempt: u64) -> Space {
    if attempt >= MATH_ATTEMPT_BASE {
        Space::Math
    } else if attempt >= TRAIT_ATTEMPT_BASE {
        Space::Trait
    } else if attempt >= CLOSURE_ATTEMPT_BASE {
        Space::Closure
    } else if attempt >= ARRAY_ATTEMPT_BASE {
        Space::Array
    } else if attempt >= VALUE_ATTEMPT_BASE {
        Space::Value
    } else if attempt >= ENUM_ATTEMPT_BASE {
        Space::Enum
    } else if attempt >= METHOD_ATTEMPT_BASE {
        Space::Method
    } else if attempt >= VECTOR_ATTEMPT_BASE {
        Space::Vector
    } else {
        Space::Float
    }
}
/// Bytes of the stream one program may consume; the decoder reads zeros beyond.
pub const BYTES_PER_PROGRAM: usize = 4096;
/// In-domain programs per seed (R1-13).
pub const PROGRAMS_PER_SEED: usize = 17;
/// In-domain programs of a seed in one space (`generate_batch_in`): the batch size of formats 5 to 13, which the pins of
/// the older spaces hash.
pub const SPACE_PROGRAMS_PER_SEED: usize = 16;
/// Attempts per seed before `generate_batch` gives up (the discard rate is far below this limit).
pub const MAX_ATTEMPTS: u64 = 100_000;
/// Expression depth limit (root = 6).
pub const MAX_DEPTH: u32 = 6;
