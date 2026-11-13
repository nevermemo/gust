//! The seeded program generator: seed -> byte stream -> AST (the decoding of `arbitrary`, written in place).
//!
//! A program is a pure function of `(seed, attempt)`: `byte_stream` gives `BYTES_PER_PROGRAM` bytes and `decode`
//! turns them into a `Program`; reading past the end of the stream yields zero bytes, which select the first (leaf)
//! option of every choice, so truncating a stream (a shrinking pass of m5.c1c) always decodes to a valid, smaller
//! program. `generate_batch` keeps the programs that the interpreter accepts as in-domain until it has
//! `PROGRAMS_PER_SEED` of them and counts the discards by reason.
//!
//! Anything that changes what a seed decodes to (this file, `prng.rs`, the operator or type lists) is a change of
//! the format and must bump `DIFFTEST_FORMAT` and regenerate `seeds.txt` knowingly.

use std::collections::BTreeMap;

use crate::ast::{
    Arm, BinOp, Elem, EnumDef, Expr, FnDef, GenParam, Imp, InputBuf, Lane, MAX_ENUMS, MAX_FLAT,
    MAX_TRIPS, MAX_TUPLE_FLAT, MAX_VARIANTS, Methods, Pat, Place, Program, Recv, Root, Stmt,
    StructDef, TupleTy, Ty, WIDTHS, chain_terms,
};
use crate::interp;
mod array_gen;
mod closure_gen;
mod trait_gen;
use crate::prng::{Prng, byte_stream};
use closure_gen::{ArgBias, Cl};

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
/// 13: traits and generic helpers (slice m9.d2, D166, D170): a seed's batch is three float, four vector, one method, one
/// enum, one value, two array, two closure and two trait programs. A trait program (`Space::Trait`, attempts from
/// `TRAIT_ATTEMPT_BASE`) is an enum program (no floats, tuples, arrays or closures) that also declares one to three
/// module traits `Tr<n>` (one or two methods `t<n>m<k>` with a receiver and scalar parameters), each method
/// implemented by exactly two structs or field-less enums, calls them (`s.m(a)`, `Tr::m(&s, a)`,
/// `<S as Tr>::m(&s, a)`) and declares generic helpers `fn g<T: Tr0 + Tr1, U>(t: T, ..) -> ..` (inline or `where`
/// bounds, every parameter bounded by one or two module traits, `Copy` sometimes added) called with at least two
/// different types. No turbofish, std bound, arithmetic on `T`, generic trait method, recursion, array or
/// barrier/atomic/subgroup operation is generated. The older spaces decode as in format 12; the batch cut is new.
pub const DIFFTEST_FORMAT: u32 = 13;

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
}

/// The first attempt number of the vector space. A program is a pure function of `(seed, attempt)`, and the attempt
/// number alone says which space decodes it, so the reducer and the shrinker re-decode a vector program without being
/// told. The scalar part of a batch never gets near it (`MAX_ATTEMPTS` is 100 000).
pub const VECTOR_ATTEMPT_BASE: u64 = 1_000_000;
/// The first attempt number of the method space (above every vector attempt a seed can make).
pub const METHOD_ATTEMPT_BASE: u64 = 2_000_000;
/// In-domain programs of a seed's batch that are scalar (the float space), then vector, then method
/// (`PROGRAMS_PER_SEED` in all).
pub const SCALAR_PROGRAMS_PER_SEED: usize = 3;
pub const VECTOR_PROGRAMS_PER_SEED: usize = 4;
pub const METHOD_PROGRAMS_PER_SEED: usize = 1;
pub const ENUM_PROGRAMS_PER_SEED: usize = 1;
pub const ARRAY_PROGRAMS_PER_SEED: usize = 2;
pub const CLOSURE_PROGRAMS_PER_SEED: usize = 2;
pub const TRAIT_PROGRAMS_PER_SEED: usize = 2;
/// The first attempt number of the trait space (above every closure attempt a seed can make).
pub const TRAIT_ATTEMPT_BASE: u64 = 7_000_000;
pub const VALUE_PROGRAMS_PER_SEED: usize = PROGRAMS_PER_SEED
    - SCALAR_PROGRAMS_PER_SEED
    - VECTOR_PROGRAMS_PER_SEED
    - METHOD_PROGRAMS_PER_SEED
    - ENUM_PROGRAMS_PER_SEED
    - ARRAY_PROGRAMS_PER_SEED
    - CLOSURE_PROGRAMS_PER_SEED
    - TRAIT_PROGRAMS_PER_SEED;
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

/// The space whose decoder reads the program of this attempt number.
pub fn space_of_attempt(attempt: u64) -> Space {
    if attempt >= TRAIT_ATTEMPT_BASE {
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
pub const PROGRAMS_PER_SEED: usize = 16;
/// Attempts per seed before `generate_batch` gives up (the discard rate is far below this limit).
pub const MAX_ATTEMPTS: u64 = 100_000;
/// Expression depth limit (root = 6).
pub const MAX_DEPTH: u32 = 6;

const COUNTS: [u32; 5] = [1, 63, 64, 65, 257];

struct Bytes<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Bytes<'_> {
    fn byte(&mut self) -> u8 {
        let byte = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        byte
    }

    fn u64(&mut self) -> u64 {
        (0..8).fold(0, |acc, k| acc | u64::from(self.byte()) << (8 * k))
    }

    /// A value in `0..bound`; a zero byte (or a truncated stream) gives 0.
    fn below(&mut self, bound: u32) -> u32 {
        match bound {
            0 | 1 => 0,
            2..=256 => u32::from(self.byte()) % bound,
            _ => (u32::from(self.byte()) | u32::from(self.byte()) << 8) % bound,
        }
    }

    fn chance(&mut self, percent: u32) -> bool {
        self.below(100) < percent
    }

    /// An index chosen with the given weights; zero bytes choose index 0.
    fn weighted(&mut self, weights: &[u32]) -> usize {
        let total: u32 = weights.iter().sum();
        let mut r = self.below(total);
        for (k, weight) in weights.iter().enumerate() {
            if r < *weight {
                return k;
            }
            r -= weight;
        }
        weights.len() - 1
    }
}

struct Gen<'a> {
    b: Bytes<'a>,
    floats: bool,
    /// The vector space: vectors and matrices as locals and intermediates (implies `floats`).
    vectors: bool,
    inputs: Vec<Ty>,
    /// The data of each input buffer (same order as `inputs`); the float space draws literals and conditions from it.
    data: Vec<Vec<u32>>,
    count: u32,
    next_var: u32,
    /// (variable, type, mutable), innermost last.
    scope: Vec<(u32, Ty, bool)>,
    /// The method space: structs and functions (the tables are complete before the kernel is drawn; while function `k`
    /// is drawn they hold the functions below `k`).
    methods: bool,
    table: Methods,
    /// The function being drawn (`None` in the kernel): it may call the functions below it only.
    cur_fn: Option<usize>,
    /// The receiver of the function being drawn: the struct type and how `self` is held.
    this: Option<(Ty, Recv)>,
    /// Roots of the receivers of the calls whose arguments are being drawn: no `&mut self` call on them there.
    forbid: Vec<Root>,
    /// The enum space (implies `methods`): enums, `match` and `return`. Every draw they add sits behind this flag.
    enums: bool,
    /// While a helper function is drawn: its result type, so that an arm or a `then` block may end in `return`.
    cur_ret: Option<Ty>,
    /// The function being drawn must return an enum (the tables must hold at least one such function).
    force_enum_ret: bool,
    /// The value space (implies `enums`): tuples, the value `match` and integer patterns. Every draw they add sits
    /// behind this flag.
    values: bool,
    /// Lets that drawing an expression hoisted (a struct literal cannot be a tuple element, E0810): the statement that
    /// is being drawn gets them in front of it.
    pre: Vec<Stmt>,
    /// The helpers that hold a `return` in an arm of a `match` (calls prefer them in the value space).
    arm_return_fns: Vec<usize>,
    /// The array space (implies `enums`, not `values`): arrays, `store` and the array helpers.
    arrays: bool,
    /// The final store of the program being drawn (the array space).
    store: Option<crate::ast::Store>,
    /// The closure space (implies `arrays`): non-capturing closures and their calls. Every draw they add sits behind
    /// this flag.
    closures: bool,
    /// A closure body is being drawn: no `i`, no buffer, no `self`, no `return`, no closure.
    in_closure: bool,
    /// The probe writer closures (closure, struct local, field, field type) of the kernel (see `closure_probe_lets`).
    probe_writers: Vec<(u32, u32, u8, Ty)>,
    /// The closures bound in the open blocks.
    cls: Vec<Cl>,
    /// The trait space (implies `enums`, not `values`, `arrays` or `closures`): module traits, their impls and generic
    /// helpers. Every draw they add sits behind this flag.
    traits: bool,
    /// The type parameters of the generic function being drawn (empty elsewhere).
    cur_gens: Vec<GenParam>,
    /// The (generic function, type) instantiations drawn so far: a call prefers a type not yet used.
    inst_used: Vec<(usize, Ty)>,
}

/// The signature a struct impl of a trait method is drawn with (format 13).
#[derive(Clone)]
struct Forced {
    owner: Option<u8>,
    recv: Recv,
    params: Vec<Ty>,
    ret: Option<Ty>,
    imp: Imp,
    /// The index below which the body may call: the first impl of the pair.
    limit: usize,
}

/// A data value as a literal; i32::MIN is written as the expression -2147483647 - 1 (the emitter never writes the
/// literal -2147483648i32).
fn data_value(ty: Ty, bits: u32) -> Expr {
    if ty == Ty::I32 && bits == 0x8000_0000 {
        Expr::i32_min()
    } else {
        Expr::Lit(ty, bits)
    }
}

const SMALL_DIVISORS: [u32; 8] = [1, 2, 3, 4, 7, 10, 16, 255];
const MASKS: [u32; 6] = [0x1, 0xF, 0xFF, 0xFFF, 0xFFFF, 0x7FFF_FFFF];

impl Gen<'_> {
    fn fresh(&mut self) -> u32 {
        let id = self.next_var;
        self.next_var += 1;
        id
    }

    fn vars(&self, ty: Ty) -> Vec<u32> {
        self.scope
            .iter()
            .filter(|(_, t, _)| *t == ty)
            .map(|(id, ..)| *id)
            .collect()
    }

    fn inputs_of(&self, ty: Ty) -> Vec<usize> {
        self.inputs
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == ty)
            .map(|(k, _)| k)
            .collect()
    }

    /// The type of a local: a scalar, or in the vector space often a vector or a matrix.
    fn pick_ty(&mut self) -> Ty {
        if self.values && self.b.chance(24) {
            return self.tuple_ty();
        }
        if self.enums && self.b.chance(28) {
            return self.enum_ty();
        }
        if self.vectors {
            return match self.b.weighted(&[18, 18, 8, 16, 40]) {
                0 => Ty::U32,
                1 => Ty::I32,
                2 => Ty::Bool,
                3 => Ty::F32,
                _ => self.agg_ty(),
            };
        }
        self.scalar_ty()
    }

    /// The type of a result: always a scalar (the output buffer is scalar).
    fn scalar_ty(&mut self) -> Ty {
        if self.floats {
            return [Ty::U32, Ty::I32, Ty::Bool, Ty::F32][self.b.weighted(&[30, 30, 15, 25])];
        }
        [Ty::U32, Ty::I32, Ty::Bool][self.b.weighted(&[40, 40, 20])]
    }

    fn int_ty(&mut self) -> Ty {
        [Ty::U32, Ty::I32][self.b.below(2) as usize]
    }

    /// A value that an input buffer of type `ty` holds, as a literal; `None` when the program has no such buffer.
    fn data_literal(&mut self, ty: Ty) -> Option<Expr> {
        let buffers = self.inputs_of(ty);
        if buffers.is_empty() {
            return None;
        }
        let buf = buffers[self.b.below(buffers.len() as u32) as usize];
        let at = self.b.below(self.count) as usize;
        Some(data_value(ty, self.data[buf][at]))
    }

    fn literal(&mut self, ty: Ty) -> Expr {
        // The float space: one literal in eight is a value the data holds, so that a comparison with it has both
        // outcomes over the elements (a literal drawn from the edge lists almost never equals or separates the data).
        if self.floats
            && ty != Ty::Bool
            && self.b.chance(12)
            && let Some(literal) = self.data_literal(ty)
        {
            return literal;
        }
        match ty {
            Ty::Vec(..)
            | Ty::Mat(_)
            | Ty::Struct(_)
            | Ty::Enum(_)
            | Ty::Tuple(_)
            | Ty::Array(_)
            | Ty::Param(_) => {
                unreachable!("a vector, matrix, struct or enum has no literal")
            }
            Ty::F32 => self.float_literal(),
            Ty::Bool => Expr::bool(self.b.below(2) == 1),
            Ty::U32 => Expr::u32(match self.b.weighted(&[55, 12, 8, 8, 12, 5]) {
                0 => self.b.below(17),
                1 => 1 << self.b.below(32),
                2 => u32::MAX >> self.b.below(32),
                3 => [u32::MAX, u32::MAX - 1, 0x8000_0000, 0x7FFF_FFFF][self.b.below(4) as usize],
                4 => self.b.below(65_536),
                _ => (self.b.u64() >> 16) as u32,
            }),
            Ty::I32 => {
                let kind = self.b.weighted(&[40, 15, 10, 15, 12, 5, 3]);
                match kind {
                    0 => Expr::i32(self.b.below(17) as i32),
                    1 => Expr::i32(-(self.b.below(16) as i32) - 1),
                    2 => Expr::i32(1 << self.b.below(31)),
                    3 => Expr::i32(-(1 << self.b.below(31))),
                    4 => Expr::i32(self.b.below(65_536) as i32 - 32_768),
                    5 => [
                        Expr::i32(i32::MAX),
                        Expr::i32(i32::MIN + 1),
                        Expr::i32(i32::MAX - 1),
                        Expr::i32(-1),
                    ][self.b.below(4) as usize]
                        .clone(),
                    _ => Expr::i32_min(),
                }
            }
        }
    }

    fn input_read(&mut self, buf: usize, ty: Ty) -> Expr {
        let index = match self.b.weighted(&[80, 10, 7, 3]) {
            0 => Expr::Gid,
            1 => {
                let inner = self.expr(Ty::U32, 1);
                // The closure space has `&mut self` calls everywhere (the counter methods), and this index may stand in an
                // assigned place (E0421): it keeps the draw but drops a call that mutates.
                let mut roots = Vec::new();
                crate::ast::mutating_roots(&self.table, &inner, &mut roots);
                let inner = if self.closures && !roots.is_empty() {
                    Expr::Gid
                } else {
                    inner
                };
                Expr::bin(BinOp::Rem, inner, Expr::Len(buf))
            }
            2 => Expr::bin(
                BinOp::Sub,
                Expr::bin(BinOp::Sub, Expr::Len(buf), Expr::u32(1)),
                Expr::Gid,
            ),
            _ => Expr::bin(BinOp::Add, Expr::Gid, Expr::u32(1 + self.b.below(3))),
        };
        Expr::input(buf, ty, index)
    }

    fn leaf(&mut self, ty: Ty) -> Expr {
        if ty.is_aggregate() {
            return self.agg_leaf(ty);
        }
        if ty.is_enum() {
            return self.enum_leaf(ty);
        }
        let vars = self.vars(ty);
        let inputs = self.inputs_of(ty);
        // 0 literal, 1 variable, 2 input read, 3 gid, 4 len.
        // A function sees no buffer and no element index.
        let kernel = self.cur_fn.is_none() && !self.in_closure;
        let mut weights = vec![
            18,
            if vars.is_empty() { 0 } else { 55 },
            if inputs.is_empty() || !kernel { 0 } else { 40 },
            if ty == Ty::U32 && kernel { 5 } else { 0 },
            if ty == Ty::U32 && kernel { 3 } else { 0 },
        ];
        // 5: a field of a struct local or of `self` (the method space only).
        let fields = if self.methods {
            self.scalar_places(ty)
        } else {
            Vec::new()
        };
        if self.methods {
            weights.push(if fields.is_empty() { 0 } else { 45 });
        }
        match self.b.weighted(&weights) {
            5 => {
                let (place, _) = fields[self.b.below(fields.len() as u32) as usize].clone();
                Expr::Read { place, ty }
            }
            1 => Expr::Var(vars[self.b.below(vars.len() as u32) as usize], ty),
            2 => {
                let buf = inputs[self.b.below(inputs.len() as u32) as usize];
                self.input_read(buf, ty)
            }
            3 => Expr::Gid,
            4 => {
                let buf = self.b.below(self.inputs.len() as u32) as usize;
                Expr::Len(buf)
            }
            _ => self.literal(ty),
        }
    }

    fn masked(&mut self, e: Expr) -> Expr {
        let mask = MASKS[self.b.below(MASKS.len() as u32) as usize];
        let ty = e.ty();
        Expr::bin(BinOp::BitAnd, e, Expr::Lit(ty, mask))
    }

    fn divisor(&mut self, ty: Ty, depth: u32) -> Expr {
        match self.b.weighted(&[32, 32, 26, 10]) {
            0 => {
                let value = SMALL_DIVISORS[self.b.below(SMALL_DIVISORS.len() as u32) as usize];
                if ty == Ty::I32 && self.b.chance(25) {
                    Expr::i32(-(value as i32))
                } else {
                    Expr::Lit(ty, value)
                }
            }
            1 => {
                let e = self.expr(ty, depth);
                Expr::bin(BinOp::BitOr, e, Expr::Lit(ty, 1))
            }
            2 => match self.expr(ty, depth) {
                // A literal zero divisor is rejected by the `#[gpu]` validator (`DivisionByLiteralZero`) even in a
                // branch no input reaches, so it is never emitted.
                Expr::Lit(_, 0) => Expr::Lit(ty, 1),
                other => other,
            },
            // The all-ones divisor: `-1` for `i32` (so `MIN / -1` is reached), `u32::MAX` for `u32`.
            _ => Expr::Lit(ty, u32::MAX),
        }
    }

    fn shift_amount(&mut self, depth: u32) -> Expr {
        match self.b.weighted(&[55, 35, 7, 3]) {
            0 => Expr::u32(self.b.below(32)),
            1 => {
                let e = self.expr(Ty::U32, depth);
                Expr::bin(BinOp::BitAnd, e, Expr::u32(31))
            }
            2 => self.expr(Ty::U32, depth),
            _ => Expr::u32(32 + self.b.below(3)),
        }
    }

    fn arithmetic(&mut self, op: BinOp, ty: Ty, depth: u32) -> Expr {
        let mut l = self.expr(ty, depth - 1);
        let mut r = self.expr(ty, depth - 1);
        // Masking keeps a share of the products and sums inside the domain, so the discard rate stays workable.
        let mask_chance = if op == BinOp::Mul { 45 } else { 20 };
        if self.b.chance(mask_chance) {
            l = self.masked(l);
        }
        if self.b.chance(mask_chance) {
            r = self.masked(r);
        }
        Expr::bin(op, l, r)
    }

    fn expr(&mut self, ty: Ty, depth: u32) -> Expr {
        if ty.is_aggregate() {
            return self.agg_expr(ty, depth);
        }
        if ty.is_enum() {
            return self.enum_expr(ty, depth);
        }
        if ty.is_tuple() {
            return self.tuple_expr(ty, depth);
        }
        let leaf_chance = if depth == 0 {
            100
        } else {
            15 + (MAX_DEPTH - depth.min(MAX_DEPTH)) * 12
        };
        if self.b.chance(leaf_chance) {
            return self.leaf(ty);
        }
        let d = depth - 1;
        if self.methods
            && self.b.chance(24)
            && let Some(call) = self.value_call(ty, d)
        {
            return call;
        }
        // A closure call (the closure space): `f(c.bump(), c.bump())`, `f(c.bump()) ^ f(c.bump())`.
        if self.closures && !self.cls.is_empty() && self.b.chance(26) {
            let roll = self.b.below(100);
            let special = if roll < 22 {
                self.closure_twice(ty, d)
            } else if roll < 55 {
                self.closure_observed_call(ty, d)
            } else {
                None
            };
            if let Some(call) = special.or_else(|| self.closure_call(ty, d, None)) {
                return call;
            }
        }
        // An element of a call's tuple result (`f().1`).
        if self.values
            && self.b.chance(14)
            && let Some(field) = self.tuple_field_call(ty, d, None)
        {
            return field;
        }
        // The order-sensitive pairs: a read and a `&mut self` call on one root, a call right of `&&`/`||`.
        if self.methods
            && self.b.chance(22)
            && let Some(pair) = self.root_binary(ty, d)
        {
            return pair;
        }
        // An enum read back as its repr type (`e as u32`).
        if self.enums && ty.is_int() && self.b.chance(9) {
            let reprs: Vec<u8> = (0..self.table.enums.len() as u8)
                .filter(|&k| self.table.enums[usize::from(k)].repr == ty)
                .collect();
            if !reprs.is_empty() {
                let k = reprs[self.b.below(reprs.len() as u32) as usize];
                let inner = self.enum_expr(Ty::Enum(k), d);
                return Expr::EnumAs {
                    to: ty,
                    e: Box::new(inner),
                };
            }
        }
        if ty == Ty::Bool {
            return self.bool_expr(d);
        }
        if ty == Ty::F32 {
            return self.float_expr(d);
        }
        // add sub mul div rem and or xor shl shr neg not cast (and, in the float space, cast from f32)
        let neg_weight = if ty == Ty::I32 { 4 } else { 0 };
        let mut weights = vec![14, 9, 8, 6, 6, 7, 5, 5, 5, 6, neg_weight, 3, 6];
        if self.floats {
            weights.push(5);
        }
        if self.vectors {
            weights.push(8);
        }
        match self.b.weighted(&weights) {
            0 => self.arithmetic(BinOp::Add, ty, depth),
            1 => self.arithmetic(BinOp::Sub, ty, depth),
            2 => self.arithmetic(BinOp::Mul, ty, depth),
            3 | 4 => {
                let op = if self.b.below(2) == 0 {
                    BinOp::Div
                } else {
                    BinOp::Rem
                };
                let mut l = self.expr(ty, d);
                let r = self.divisor(ty, d);
                // `-1` as a divisor is only interesting if the dividend can be `i32::MIN`, which only the
                // edge-valued input buffers hold; so the dividend is often a plain read of such a buffer.
                if r == Expr::Lit(Ty::I32, u32::MAX)
                    && self.cur_fn.is_none()
                    && !self.in_closure
                    && self.b.chance(60)
                {
                    let buffers = self.inputs_of(Ty::I32);
                    if !buffers.is_empty() {
                        let buf = buffers[self.b.below(buffers.len() as u32) as usize];
                        l = self.input_read(buf, Ty::I32);
                    }
                }
                Expr::bin(op, l, r)
            }
            5 => self.arithmetic(BinOp::BitAnd, ty, depth),
            6 => self.arithmetic(BinOp::BitOr, ty, depth),
            7 => self.arithmetic(BinOp::BitXor, ty, depth),
            8 | 9 => {
                let op = if self.b.below(2) == 0 {
                    BinOp::Shl
                } else {
                    BinOp::Shr
                };
                let l = self.expr(ty, d);
                let r = self.shift_amount(d);
                Expr::bin(op, l, r)
            }
            10 => Expr::negated(self.expr(Ty::I32, d)),
            11 => Expr::inverted(self.expr(ty, d)),
            12 => {
                let other = if ty == Ty::U32 { Ty::I32 } else { Ty::U32 };
                Expr::cast(ty, self.expr(other, d))
            }
            13 => Expr::cast(ty, self.expr(Ty::F32, d)),
            _ => self.agg_readout(ty, d),
        }
    }

    /// An `f32` expression (R1-4): `+ - * /`, unary `-`, and casts from integers. A `+`/`-` chain has at most four
    /// written terms; a divisor is mostly kept away from zero by construction.
    fn float_expr(&mut self, d: u32) -> Expr {
        let weights: &[u32] = if self.vectors {
            &[16, 10, 14, 8, 6, 8, 14]
        } else {
            &[16, 10, 14, 8, 6, 8]
        };
        match self.b.weighted(weights) {
            kind @ (0 | 1) => {
                let op = if kind == 0 { BinOp::Add } else { BinOp::Sub };
                let l = self.expr(Ty::F32, d);
                let r = self.expr(Ty::F32, d);
                let (l, r) = self.fit_chain(l, r, 4);
                Expr::bin(op, l, r)
            }
            2 => {
                let l = self.expr(Ty::F32, d);
                let r = self.expr(Ty::F32, d);
                Expr::bin(BinOp::Mul, l, r)
            }
            3 => {
                let l = self.expr(Ty::F32, d);
                let r = self.float_divisor(d);
                Expr::bin(BinOp::Div, l, r)
            }
            4 => Expr::negated(self.expr(Ty::F32, d)),
            6 => self.agg_readout(Ty::F32, d),
            _ => {
                let ty = self.int_ty();
                let e = self.expr(ty, d);
                // A converted integer must stay within 2^24: mostly masked to 16 bits.
                let e = if self.b.chance(70) {
                    Expr::bin(BinOp::BitAnd, e, Expr::Lit(ty, 0xFFFF))
                } else {
                    e
                };
                Expr::cast(Ty::F32, e)
            }
        }
    }

    /// Replaces operands of a `+`/`-` chain by leaves until it has at most `max` written terms.
    fn fit_chain(&mut self, mut l: Expr, mut r: Expr, max: usize) -> (Expr, Expr) {
        while chain_terms(&l) + chain_terms(&r) > max {
            if chain_terms(&l) >= chain_terms(&r) {
                l = self.leaf(l.ty());
            } else {
                r = self.leaf(r.ty());
            }
        }
        (l, r)
    }

    /// The type of a vector or matrix local: mostly `f32` vectors and matrices, which carry the interval bounds.
    fn agg_ty(&mut self) -> Ty {
        const TYS: [Ty; 8] = [
            Ty::Vec(Lane::F32, 2),
            Ty::Vec(Lane::F32, 4),
            Ty::Vec(Lane::I32, 2),
            Ty::Vec(Lane::I32, 4),
            Ty::Vec(Lane::U32, 2),
            Ty::Vec(Lane::U32, 4),
            Ty::Mat(2),
            Ty::Mat(4),
        ];
        TYS[self.b.weighted(&[20, 24, 7, 7, 7, 7, 10, 18])]
    }

    /// A vector or matrix leaf: a local of the type, or a fresh construction from scalar leaves.
    fn agg_leaf(&mut self, ty: Ty) -> Expr {
        let vars = self.vars(ty);
        if !vars.is_empty() && self.b.chance(65) {
            return Expr::Var(vars[self.b.below(vars.len() as u32) as usize], ty);
        }
        self.build(ty, 1)
    }

    /// `Vec4::new(..)`, `Vec4::splat(..)`, `Mat4::from_cols(..)` or `Mat4::from_rows(..)` of operands of depth `d`. A
    /// `Mat2` always has an input read among its columns (SEMANTICS section 4: a constant `Mat2` crashes naga's SPIR-V
    /// front end, so the generator never writes one).
    fn build(&mut self, ty: Ty, d: u32) -> Expr {
        match ty {
            Ty::Vec(lane, n) => {
                if self.b.chance(15) {
                    let e = self.expr(lane.ty(), d);
                    return Expr::Splat { ty, e: Box::new(e) };
                }
                let args = (0..n).map(|_| self.expr(lane.ty(), d)).collect();
                Expr::vec_new(ty, args)
            }
            Ty::Mat(n) => {
                let column = Ty::Vec(Lane::F32, n);
                let rows = self.b.chance(30);
                let mut args: Vec<Expr> = (0..n).map(|_| self.agg_expr(column, d)).collect();
                if n == 2 && !args.iter().any(Expr::reads_f32_input_directly) {
                    args[0] = self.vec2_from_input();
                }
                Expr::Make { ty, rows, args }
            }
            other => unreachable!("build of {}", other.name()),
        }
    }

    /// A `Vec2` that reads an `f32` input buffer (the vector space always has one).
    fn vec2_from_input(&mut self) -> Expr {
        let buffers = self.inputs_of(Ty::F32);
        let buf = buffers[self.b.below(buffers.len() as u32) as usize];
        let x = self.input_read(buf, Ty::F32);
        let y = self.float_literal();
        Expr::vec_new(Ty::Vec(Lane::F32, 2), vec![x, y])
    }

    /// A scalar read out of a vector: a component, or the `dot` of two vectors of one type.
    fn agg_readout(&mut self, ty: Ty, d: u32) -> Expr {
        let lane = match ty {
            Ty::U32 => Lane::U32,
            Ty::I32 => Lane::I32,
            _ => Lane::F32,
        };
        let n = WIDTHS[self.b.below(2) as usize];
        let vector = Ty::Vec(lane, n);
        let first = self.agg_expr(vector, d);
        if self.b.chance(40) {
            let second = self.agg_expr(vector, d);
            Expr::dot(first, second)
        } else {
            let index = self.b.below(u32::from(n)) as u8;
            Expr::comp(first, index)
        }
    }

    /// A divisor for one lane of a vector: never a literal zero.
    fn lane_divisor(&mut self, lane: Lane, d: u32) -> Expr {
        if lane == Lane::F32 {
            self.float_divisor(d)
        } else {
            self.divisor(lane.ty(), d)
        }
    }

    fn agg_expr(&mut self, ty: Ty, depth: u32) -> Expr {
        let leaf_chance = if depth == 0 {
            100
        } else {
            15 + (MAX_DEPTH - depth.min(MAX_DEPTH)) * 12
        };
        if self.b.chance(leaf_chance) {
            return self.agg_leaf(ty);
        }
        let d = depth - 1;
        match ty {
            Ty::Vec(lane, n) => self.vec_expr(ty, lane, n, d),
            Ty::Mat(n) => self.mat_expr(ty, n, d),
            other => unreachable!("agg_expr of {}", other.name()),
        }
    }

    fn vec_expr(&mut self, ty: Ty, lane: Lane, n: u8, d: u32) -> Expr {
        let float = lane == Lane::F32;
        let weights = [
            14,
            8,
            5,
            if lane == Lane::U32 { 0 } else { 5 },
            8,
            8,
            4,
            if float { 10 } else { 0 },
        ];
        match self.b.weighted(&weights) {
            0 => {
                let op = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div]
                    [self.b.weighted(&[10, 8, 10, 5])];
                let l = self.agg_expr(ty, d);
                if op == BinOp::Div {
                    let args = (0..n).map(|_| self.lane_divisor(lane, d)).collect();
                    return Expr::agg_bin(op, l, Expr::vec_new(ty, args));
                }
                let r = self.agg_expr(ty, d);
                let (l, r) = if float && matches!(op, BinOp::Add | BinOp::Sub) {
                    self.fit_chain(l, r, 4)
                } else {
                    (l, r)
                };
                Expr::agg_bin(op, l, r)
            }
            1 => {
                let op = if self.b.below(2) == 0 {
                    BinOp::Mul
                } else {
                    BinOp::Div
                };
                let l = self.agg_expr(ty, d);
                let r = if op == BinOp::Div {
                    self.lane_divisor(lane, d)
                } else {
                    self.expr(lane.ty(), d)
                };
                Expr::agg_bin(op, l, r)
            }
            2 => {
                let s = self.expr(lane.ty(), d);
                let v = self.agg_expr(ty, d);
                Expr::agg_bin(BinOp::Mul, s, v)
            }
            3 => Expr::AggNeg {
                e: Box::new(self.agg_expr(ty, d)),
            },
            4 => {
                let width = WIDTHS[self.b.below(2) as usize];
                let source = self.agg_expr(Ty::Vec(lane, width), d);
                let lanes = (0..n)
                    .map(|_| self.b.below(u32::from(width)) as u8)
                    .collect();
                Expr::swizzle(source, lanes)
            }
            5 => self.build(ty, d),
            6 => {
                let e = self.expr(lane.ty(), d);
                Expr::Splat { ty, e: Box::new(e) }
            }
            _ => {
                let m = self.agg_expr(Ty::Mat(n), d);
                let v = self.agg_expr(ty, d);
                Expr::agg_bin(BinOp::Mul, m, v)
            }
        }
    }

    fn mat_expr(&mut self, ty: Ty, n: u8, d: u32) -> Expr {
        match self.b.weighted(&[8, 10, 14, 8, 6, 4]) {
            0 => self.build(ty, d),
            1 => {
                let op = if self.b.below(2) == 0 {
                    BinOp::Add
                } else {
                    BinOp::Sub
                };
                let l = self.agg_expr(ty, d);
                let r = self.agg_expr(ty, d);
                let (l, r) = self.fit_chain(l, r, 4);
                Expr::agg_bin(op, l, r)
            }
            2 => {
                let l = self.agg_expr(ty, d);
                let r = self.agg_expr(ty, d);
                Expr::agg_bin(BinOp::Mul, l, r)
            }
            3 => {
                let m = self.agg_expr(ty, d);
                let s = self.expr(Ty::F32, d);
                if self.b.below(2) == 0 {
                    Expr::agg_bin(BinOp::Mul, m, s)
                } else {
                    Expr::agg_bin(BinOp::Mul, s, m)
                }
            }
            4 => Expr::Transpose {
                e: Box::new(self.agg_expr(Ty::Mat(n), d)),
            },
            _ => Expr::AggNeg {
                e: Box::new(self.agg_expr(ty, d)),
            },
        }
    }

    /// An assignment to a mutable vector or matrix local: one component, the accumulating form, or a fresh value.
    fn assign_agg(&mut self, var: u32, ty: Ty) -> Stmt {
        if let Ty::Vec(lane, n) = ty
            && self.b.chance(40)
        {
            let index = self.b.below(u32::from(n)) as u8;
            let value = self.expr(lane.ty(), 3);
            return Stmt::AssignComp { var, index, value };
        }
        let value = if self.b.chance(45) {
            let current = Expr::Var(var, ty);
            let rhs = self.agg_expr(ty, 2);
            let op = [BinOp::Add, BinOp::Add, BinOp::Sub, BinOp::Mul][self.b.below(4) as usize];
            let (current, rhs) = if matches!(op, BinOp::Add | BinOp::Sub) && ty.scalar() == Ty::F32
            {
                self.fit_chain(current, rhs, 4)
            } else {
                (current, rhs)
            };
            Expr::agg_bin(op, current, rhs)
        } else {
            self.agg_expr(ty, 3)
        };
        Stmt::Assign { var, value }
    }

    fn float_divisor(&mut self, d: u32) -> Expr {
        const DIVISORS: [f32; 10] = [0.5, 1.0, 2.0, 3.0, 4.0, 0.25, 10.0, -2.0, -4.0, 8.0];
        match self.b.weighted(&[40, 30, 30]) {
            0 => Expr::f32(DIVISORS[self.b.below(DIVISORS.len() as u32) as usize]),
            1 => match self.expr(Ty::F32, d) {
                // Never a literal zero divisor (the validator refuses one even in dead code).
                Expr::Lit(_, 0) => Expr::f32(1.0),
                other => other,
            },
            _ => {
                // x * x + c with c >= 1: away from zero whatever x is.
                let e = self.expr(Ty::F32, d);
                let square = Expr::bin(BinOp::Mul, e.clone(), e);
                let c = Expr::f32(1.0 + self.b.below(4) as f32);
                Expr::bin(BinOp::Add, square, c)
            }
        }
    }

    /// A literal from a spread of values: small integers, powers of two, decimals, zero; a third of them negative.
    fn float_literal(&mut self) -> Expr {
        let value = match self.b.weighted(&[30, 25, 20, 8, 17]) {
            0 => 1.0 + self.b.below(16) as f32,
            1 => f32::from_bits((127 - 8 + self.b.below(17)) << 23),
            2 => {
                let mantissa = self.b.below(1 << 16) as f32 / 65_536.0;
                (1.0 + mantissa) * f32::from_bits((127 - 6 + self.b.below(13)) << 23)
            }
            3 => return Expr::f32(0.0),
            _ => [0.5, 1.0, 2.0, 0.1, 0.25, 1.5][self.b.below(6) as usize],
        };
        Expr::f32(if self.b.chance(33) { -value } else { value })
    }

    /// A condition on the element index: it splits the elements whenever there are two or more.
    fn index_condition(&mut self) -> Expr {
        let cmp = |op, l, r| Expr::bin(op, l, r);
        match self.b.weighted(&[40, 35, 25]) {
            0 => {
                let modulus = [2, 3, 4, 5, 8][self.b.below(5) as usize];
                let remainder = self.b.below(modulus);
                cmp(
                    BinOp::Eq,
                    cmp(BinOp::Rem, Expr::Gid, Expr::u32(modulus)),
                    Expr::u32(remainder),
                )
            }
            1 => {
                let bound = 1 + self.b.below(self.count.max(2) - 1);
                cmp(BinOp::Lt, Expr::Gid, Expr::u32(bound))
            }
            _ => {
                let mask = [1, 2, 4, 8, 16, 32][self.b.below(6) as usize];
                cmp(
                    BinOp::Ne,
                    cmp(BinOp::BitAnd, Expr::Gid, Expr::u32(mask)),
                    Expr::u32(0),
                )
            }
        }
    }

    /// An element of an input buffer compared with a value of the data. For `f32` the pivot is the midpoint of two
    /// elements (a literal equal to an element would make the intervals overlap, and the pair would be discarded).
    fn pivot_condition(&mut self) -> Expr {
        let buf = self.b.below(self.inputs.len() as u32) as usize;
        let ty = self.inputs[buf];
        let p = self.b.below(self.count) as usize;
        let q = self.b.below(self.count) as usize;
        let (first, second) = (self.data[buf][p], self.data[buf][q]);
        let pivot = if ty == Ty::F32 {
            let mid = (f32::from_bits(first) + f32::from_bits(second)) / 2.0;
            Expr::Lit(
                ty,
                if mid.is_finite() {
                    mid.to_bits()
                } else {
                    first
                },
            )
        } else {
            data_value(ty, first)
        };
        let ops = if ty == Ty::F32 {
            [
                BinOp::Lt,
                BinOp::Le,
                BinOp::Gt,
                BinOp::Ge,
                BinOp::Lt,
                BinOp::Gt,
            ]
        } else {
            [
                BinOp::Lt,
                BinOp::Le,
                BinOp::Gt,
                BinOp::Ge,
                BinOp::Eq,
                BinOp::Ne,
            ]
        };
        let op = ops[self.b.below(6) as usize];
        Expr::bin(op, Expr::input(buf, ty, Expr::Gid), pivot)
    }

    /// The condition of an `if`. The default space never writes a bare `true` or `false` (a third of the conditions
    /// of format 2 were one, and only 2 of 55 evaluated `if`s took both sides over the elements, measured at
    /// m5.c1b): it compares an input element with a value of the data, tests the element index, or writes the general
    /// boolean expression of format 2 (replaced by an index condition if it came out as a bare literal).
    fn condition(&mut self) -> Expr {
        // The method space splits the elements in the kernel the way the float space does; a function has no element.
        if !self.floats && !(self.methods && self.cur_fn.is_none() && !self.in_closure) {
            return self.expr(Ty::Bool, 4);
        }
        match self.b.weighted(&[35, 25, 40]) {
            0 => self.pivot_condition(),
            1 => self.index_condition(),
            _ => match self.expr(Ty::Bool, 4) {
                Expr::Lit(..) => self.index_condition(),
                other => other,
            },
        }
    }

    fn bool_expr(&mut self, d: u32) -> Expr {
        if self.enums && self.b.chance(40) {
            return self.enum_compare(d);
        }
        let mut weights = vec![55, 20, 10, 5];
        if self.floats {
            weights = vec![50, 20, 10, 8, 18];
        }
        match self.b.weighted(&weights) {
            4 => {
                // Only comparisons of disjoint intervals are kept (R1-4), `==` and `!=` included.
                let op = [
                    BinOp::Eq,
                    BinOp::Ne,
                    BinOp::Lt,
                    BinOp::Le,
                    BinOp::Gt,
                    BinOp::Ge,
                ][self.b.below(6) as usize];
                Expr::bin(op, self.expr(Ty::F32, d), self.expr(Ty::F32, d))
            }
            0 => {
                let ty = self.int_ty();
                let op = [
                    BinOp::Eq,
                    BinOp::Ne,
                    BinOp::Lt,
                    BinOp::Le,
                    BinOp::Gt,
                    BinOp::Ge,
                ][self.b.below(6) as usize];
                Expr::bin(op, self.expr(ty, d), self.expr(ty, d))
            }
            1 => {
                let op = if self.b.below(2) == 0 {
                    BinOp::And
                } else {
                    BinOp::Or
                };
                Expr::bin(op, self.expr(Ty::Bool, d), self.expr(Ty::Bool, d))
            }
            2 => Expr::inverted(self.expr(Ty::Bool, d)),
            _ => {
                let op = if self.b.below(2) == 0 {
                    BinOp::Eq
                } else {
                    BinOp::Ne
                };
                Expr::bin(op, self.expr(Ty::Bool, d), self.expr(Ty::Bool, d))
            }
        }
    }

    fn block(&mut self, depth: u32, min: u32, max: u32) -> Vec<Stmt> {
        let mark = self.scope.len();
        let cl_mark = self.cls.len();
        let count = min + self.b.below(max - min + 1);
        let mut stmts = Vec::new();
        for _ in 0..count {
            let can_assign = self
                .scope
                .iter()
                .any(|(_, ty, mutable)| *mutable && !ty.is_struct() && !ty.is_tuple());
            let nest = if depth < 2 { 1 } else { 0 };
            let mut weights = vec![40, if can_assign { 24 } else { 0 }, 18 * nest, 14 * nest];
            // 4: an assignment to a field or a struct place, 5: a call as a statement, 6: a struct local.
            let targets = if self.methods {
                self.set_targets()
            } else {
                Vec::new()
            };
            let statement_calls = if self.methods {
                self.statement_calls()
            } else {
                Vec::new()
            };
            if self.methods {
                weights.push(if targets.is_empty() { 0 } else { 30 });
                weights.push(if statement_calls.is_empty() { 0 } else { 22 });
                weights.push(10);
            }
            // 7: a statement `match` (the enum space).
            if self.enums {
                weights.push(26 * nest);
            }
            // 8: a value `match`, 9: a destructuring, 10: an assignment to a tuple element (the value space).
            let tuple_targets = if self.values {
                self.tuple_set_targets()
            } else {
                Vec::new()
            };
            if self.values {
                // A helper writes more value matches: they are where its arms `return`.
                weights.push(match (depth < 3, self.cur_ret.is_some()) {
                    (false, _) => 0,
                    (true, true) => 55,
                    (true, false) => 24,
                });
                weights.push(22);
                let field_sets = self.tuple_field_set_candidates();
                weights.push(match (tuple_targets.is_empty(), field_sets.is_empty()) {
                    (true, true) => 0,
                    (_, false) => 120,
                    (false, true) => 50,
                });
            }
            // 11: a closure binding (the closure space; none inside a closure body).
            let closure_kind = weights.len();
            if self.closures {
                weights.push(if self.in_closure { 0 } else { 16 });
            }
            let (pre_mark, at) = (self.pre.len(), stmts.len());
            let kind = self.b.weighted(&weights);
            match kind {
                k if self.closures && k == closure_kind => stmts.push(self.closure_let(depth)),
                8 => stmts.push(self.let_match(depth)),
                9 => stmts.push(self.let_tuple()),
                10 => stmts.extend(self.tuple_set_stmt(&tuple_targets)),
                7 => stmts.push(self.match_stmt(depth)),
                4 => stmts.push(self.set_stmt(&targets)),
                5 => stmts.extend(self.statement_call(&statement_calls)),
                6 => {
                    let (j, init) = self.struct_init();
                    let ty = Ty::Struct(j);
                    let var = self.fresh();
                    let mutable = self.b.chance(65);
                    self.scope.push((var, ty, mutable));
                    stmts.push(Stmt::Let {
                        var,
                        ty,
                        mutable,
                        init,
                    });
                }
                0 => {
                    let ty = match self.values && self.b.chance(12) {
                        true => self.field_set_tuple_ty().unwrap_or_else(|| self.pick_ty()),
                        false => self.pick_ty(),
                    };
                    // A vector or matrix initialiser is shallower: its products multiply the bounds of its operands.
                    let init = self.expr(ty, if ty.is_aggregate() { 3 } else { MAX_DEPTH - 1 });
                    let var = self.fresh();
                    // A tuple is mutable more often (the value space writes its elements).
                    let mutable = self.b.chance(if ty.is_tuple() { 80 } else { 55 });
                    self.scope.push((var, ty, mutable));
                    stmts.push(Stmt::Let {
                        var,
                        ty,
                        mutable,
                        init,
                    });
                }
                1 => stmts.push(self.assign()),
                2 => {
                    let cond = self.condition();
                    let mut then = self.block(depth + 1, 1, 3);
                    self.maybe_return(&mut then, 25);
                    let els = self.block(depth + 1, 0, 2);
                    stmts.push(Stmt::If { cond, then, els });
                }
                _ => {
                    let trips = 1 + self.b.below(MAX_TRIPS);
                    let var = self.fresh();
                    // The float accumulator: a `let mut` before the loop and `acc = acc + term` first in the body, a
                    // chain of up to `trips + 1` terms (R1-4).
                    let mut accumulate = Vec::new();
                    if self.floats && self.b.chance(40) {
                        let init = if self.b.chance(50) {
                            Expr::f32(0.0)
                        } else {
                            self.expr(Ty::F32, 2)
                        };
                        let acc = self.fresh();
                        stmts.push(Stmt::Let {
                            var: acc,
                            ty: Ty::F32,
                            mutable: true,
                            init,
                        });
                        self.scope.push((acc, Ty::F32, true));
                        self.scope.push((var, Ty::U32, false));
                        let term = self.expr(Ty::F32, 3);
                        let current = Expr::Var(acc, Ty::F32);
                        let (current, term) = self.fit_chain(current, term, 4);
                        let op = [BinOp::Add, BinOp::Add, BinOp::Sub][self.b.below(3) as usize];
                        accumulate.push(Stmt::Assign {
                            var: acc,
                            value: Expr::bin(op, current, term),
                        });
                    } else {
                        self.scope.push((var, Ty::U32, false));
                    }
                    let mut body = accumulate;
                    body.extend(self.block(depth + 1, 1, 3));
                    // The loop variable leaves the scope with the loop.
                    self.scope.retain(|(id, ..)| *id != var);
                    stmts.push(Stmt::For { var, trips, body });
                }
            }
            self.splice_pre(pre_mark, at, &mut stmts);
        }
        self.scope.truncate(mark);
        self.cls.truncate(cl_mark);
        stmts
    }

    fn assign(&mut self) -> Stmt {
        let mutable: Vec<(u32, Ty)> = self
            .scope
            .iter()
            .filter(|(_, ty, mutable)| *mutable && !ty.is_struct() && !ty.is_tuple())
            .map(|&(id, ty, _)| (id, ty))
            .collect();
        let (var, ty) = mutable[self.b.below(mutable.len() as u32) as usize];
        if ty.is_aggregate() {
            return self.assign_agg(var, ty);
        }
        let value = if !ty.is_enum() && self.b.chance(45) {
            // The accumulating form: the target is one operand.
            let current = Expr::Var(var, ty);
            let rhs = self.expr(ty, 3);
            let op = match ty {
                Ty::F32 => {
                    [BinOp::Add, BinOp::Add, BinOp::Sub, BinOp::Mul][self.b.below(4) as usize]
                }
                Ty::Bool => [BinOp::And, BinOp::Or][self.b.below(2) as usize],
                _ => [
                    BinOp::Add,
                    BinOp::Add,
                    BinOp::Sub,
                    BinOp::BitXor,
                    BinOp::BitOr,
                    BinOp::Mul,
                ][self.b.below(6) as usize],
            };
            let (current, rhs) = if matches!(op, BinOp::Add | BinOp::Sub) && ty == Ty::F32 {
                self.fit_chain(current, rhs, 4)
            } else {
                (current, rhs)
            };
            Expr::bin(op, current, rhs)
        } else {
            self.expr(ty, 4)
        };
        Stmt::Assign { var, value }
    }
}

/// An `f32` bit pattern of magnitude `2^e * (1 + m / 2^23)` for a random mantissa `m`, with `e` in `e_lo..=e_hi` (the
/// bounded input set of R1-4 is `2^-20 <= |x| < 2^20` or zero), negative if `negative`.
fn float_in_set(prng: &mut Prng, e_lo: i32, e_hi: i32, negative: bool) -> u32 {
    let e = e_lo + prng.below((e_hi - e_lo + 1) as u32) as i32;
    let mantissa = prng.next_u32() & 0x7F_FFFF;
    (u32::from(negative) << 31) | (((e + 127) as u32) << 23) | mantissa
}

/// How `data` is drawn for one `f32` input buffer: always inside the bounded set of R1-4 (magnitudes in
/// `[2^-20, 2^20]`, or exactly zero; no negative zero).
fn float_input_data(class: usize, count: u32, sub_seed: u64) -> Vec<u32> {
    let mut prng = Prng::new(sub_seed);
    (0..count)
        .map(|_| {
            let negative = prng.next_u32() & 1 == 1;
            match class {
                0 => float_in_set(&mut prng, -2, 1, negative),
                1 => float_in_set(&mut prng, -20, 19, negative),
                2 if prng.below(100) < 30 => 0,
                2 => float_in_set(&mut prng, -6, 6, negative),
                3 => {
                    // Small exact integers: the values casts to integers see.
                    let value = 1 + prng.below(1000);
                    (value as f32).to_bits() | (u32::from(negative) << 31)
                }
                4 => float_in_set(&mut prng, 0, 3, false),
                5 => {
                    // Multiples of 1/16: exact in every grouping.
                    let value = 1 + prng.below(64);
                    (value as f32 / 16.0).to_bits() | (u32::from(negative) << 31)
                }
                _ => {
                    const EDGES: [f32; 9] = [0.0, 1.0, -1.0, 0.5, 2.0, 1.5, -2.5, 0.1, 1_048_575.0];
                    EDGES[prng.below(EDGES.len() as u32) as usize].to_bits()
                }
            }
        })
        .collect()
}

/// How `data` is drawn for one input buffer.
fn input_data(ty: Ty, class: usize, count: u32, sub_seed: u64) -> Vec<u32> {
    if ty == Ty::F32 {
        return float_input_data(class, count, sub_seed);
    }
    const U32_EDGES: [u32; 13] = [
        0,
        1,
        2,
        0x7FFF_FFFF,
        0x8000_0000,
        0x8000_0001,
        0xFFFF_FFFE,
        0xFFFF_FFFF,
        31,
        32,
        33,
        0xFFFF,
        0x1_0000,
    ];
    const I32_EDGES: [i32; 14] = [
        0,
        1,
        -1,
        2,
        -2,
        i32::MAX,
        i32::MAX - 1,
        i32::MIN,
        i32::MIN + 1,
        31,
        32,
        -31,
        0x7FFF,
        0x1_0000,
    ];
    const BOUNDS: [u64; 5] = [4, 16, 256, 65_536, 1 << 31];
    let mut prng = Prng::new(sub_seed);
    (0..count)
        .map(|_| match (class, ty) {
            (0..=4, Ty::U32) => prng.next_u32() % BOUNDS[class] as u32,
            (0..=4, _) => {
                let magnitude = (u64::from(prng.next_u32()) % BOUNDS[class]) as i64;
                let negative = prng.next_u32() & 1 == 1;
                (if negative { -magnitude } else { magnitude }) as i32 as u32
            }
            (5, _) => prng.next_u32(),
            (_, Ty::U32) => U32_EDGES[prng.below(U32_EDGES.len() as u32) as usize],
            (_, _) => I32_EDGES[prng.below(I32_EDGES.len() as u32) as usize] as u32,
        })
        .collect()
}

/// Decodes one program from a byte stream, in the space that its attempt number selects (`space_of_attempt`). Pure:
/// the same bytes give the same program.
pub fn decode(bytes: &[u8], seed: u64, attempt: u64) -> Program {
    decode_in(space_of_attempt(attempt), bytes, seed, attempt)
}

/// Decodes one program of `space` from a byte stream.
pub fn decode_in(space: Space, bytes: &[u8], seed: u64, attempt: u64) -> Program {
    let mut g = Gen {
        b: Bytes {
            data: bytes,
            pos: 0,
        },
        floats: matches!(space, Space::Float | Space::Vector),
        vectors: space == Space::Vector,
        inputs: Vec::new(),
        data: Vec::new(),
        count: 1,
        next_var: 0,
        scope: Vec::new(),
        methods: matches!(
            space,
            Space::Method
                | Space::Enum
                | Space::Value
                | Space::Array
                | Space::Closure
                | Space::Trait
        ),
        table: Methods::default(),
        cur_fn: None,
        this: None,
        forbid: Vec::new(),
        enums: matches!(
            space,
            Space::Enum | Space::Value | Space::Array | Space::Closure | Space::Trait
        ),
        cur_ret: None,
        force_enum_ret: false,
        values: space == Space::Value,
        pre: Vec::new(),
        arm_return_fns: Vec::new(),
        arrays: matches!(space, Space::Array | Space::Closure),
        store: None,
        closures: space == Space::Closure,
        in_closure: false,
        probe_writers: Vec::new(),
        cls: Vec::new(),
        traits: space == Space::Trait,
        cur_gens: Vec::new(),
        inst_used: Vec::new(),
    };
    let count = match g.b.below(6) {
        k @ 0..=4 => COUNTS[k as usize],
        _ => 1 + g.b.below(4095),
    };
    g.count = count;
    let input_count = 1 + g.b.below(3);
    let mut inputs = Vec::new();
    for k in 0..input_count {
        let ty = if g.vectors {
            // Mostly `f32` (the vectors are mostly `f32`), and at least one `f32` buffer: a `Mat2` reads one.
            let ty = [Ty::U32, Ty::I32, Ty::F32][g.b.weighted(&[25, 25, 50])];
            if k + 1 == input_count && !g.inputs.contains(&Ty::F32) {
                Ty::F32
            } else {
                ty
            }
        } else if g.floats {
            [Ty::U32, Ty::I32, Ty::F32][g.b.weighted(&[35, 35, 30])]
        } else {
            g.int_ty()
        };
        let class = g.b.weighted(&[3, 4, 4, 3, 2, 1, 3]);
        let sub_seed = g.b.u64();
        let data = input_data(ty, class, count, sub_seed);
        g.inputs.push(ty);
        g.data.push(data.clone());
        inputs.push(InputBuf { ty, data });
    }
    // The method space draws its structs and functions first; the kernel starts with one or two struct locals.
    let mut body = Vec::new();
    if g.methods {
        g.gen_tables();
        for _ in 0..1 + g.b.below(2) {
            let (pre_mark, at) = (g.pre.len(), body.len());
            let (j, init) = g.struct_init();
            let var = g.fresh();
            let mutable = g.b.chance(80);
            g.scope.push((var, Ty::Struct(j), mutable));
            body.push(Stmt::Let {
                var,
                ty: Ty::Struct(j),
                mutable,
                init,
            });
            g.splice_pre(pre_mark, at, &mut body);
        }
    }
    // The closure space binds one or two closures in the kernel before its ordinary block.
    if g.closures {
        for _ in 0..1 + g.b.below(2) {
            let (pre_mark, at) = (g.pre.len(), body.len());
            let stmt = g.closure_let(1);
            body.push(stmt);
            g.splice_pre(pre_mark, at, &mut body);
        }
        g.closure_probe_lets(&mut body);
    }
    // The trait space adds a local of every implementing type, so that the kernel can call the impls.
    if g.traits {
        g.trait_locals(&mut body);
    }
    body.extend(g.block(0, 2, 7));
    // The value space ends the kernel body, often, with `t.k.f op= t.k.m()` on a top-level tuple local (fix round 1),
    // and the result reads the field: the order-sensitive set then decides the output of the program.
    let mut forced_set = false;
    if g.values && g.b.chance(80) {
        if g.tuple_field_set_candidates().is_empty()
            && let Some(ty) = g.field_set_tuple_ty()
        {
            let (pre_mark, at) = (g.pre.len(), body.len());
            let init = g.expr(ty, 3);
            let var = g.fresh();
            g.scope.push((var, ty, true));
            body.push(Stmt::Let {
                var,
                ty,
                mutable: true,
                init,
            });
            g.splice_pre(pre_mark, at, &mut body);
        }
        let (pre_mark, at) = (g.pre.len(), body.len());
        if let Some(stmt) = g.tuple_field_set_stmt() {
            forced_set = true;
            body.push(stmt);
            g.splice_pre(pre_mark, at, &mut body);
        }
    }
    // The array space ends the kernel with its array statements, whose element reads join the result.
    let mut array_folds = Vec::new();
    if g.arrays {
        g.restore_scope(&body);
        g.restore_closures(&body);
        array_folds = g.array_section(&mut body);
    }
    // A boolean result would hide the field (`x != 0`): the forced set gets an integer result.
    let mut ty = g.scalar_ty();
    // A `bool` element reaches the output only through a `bool` result: often one.
    if array_folds.iter().any(|e| e.ty() == Ty::Bool) && g.b.chance(60) {
        ty = Ty::Bool;
    }
    if forced_set && ty == Ty::Bool {
        ty = Ty::U32;
    }
    // The closure space wants an integer result: the probe calls join it with a `^` that every path evaluates.
    if (g.closures || g.traits) && ty == Ty::Bool && g.b.chance(70) {
        ty = Ty::U32;
    }
    if g.methods {
        // The result may call on the top-level locals (`out[i] = c.next()`).
        for stmt in &body {
            if let Stmt::Let {
                var, ty, mutable, ..
            }
            | Stmt::LetMatch {
                var, ty, mutable, ..
            } = stmt
                && !g.scope.iter().any(|(id, ..)| id == var)
            {
                g.scope.push((*var, *ty, *mutable));
            }
            if let Stmt::LetTuple {
                vars,
                ty: Ty::Tuple(t),
                ..
            } = stmt
            {
                for (&(var, mutable), elem) in vars.iter().zip(t.elems()) {
                    if !g.scope.iter().any(|(id, ..)| *id == var) {
                        g.scope.push((var, elem.ty(), mutable));
                    }
                }
            }
        }
    }
    let mut result = g.expr(ty, MAX_DEPTH);
    // The trait space joins calls of its impls and generic helpers to the result: every path evaluates them.
    if g.traits {
        for probe in g.trait_probes() {
            result = fold_in(result, probe);
        }
    }
    // The value space joins a call of a helper that returns from inside an arm to the result (nit R1 of the m9.d1b
    // re-check: such a `return` must run in many programs, not in a handful).
    let mut arm_call = None;
    if g.values && g.b.chance(75) {
        // A return type that has a scalar to fold: an integer, an enum (its discriminant), or an element of a tuple.
        let usable = |ret: Option<Ty>| match ret {
            Some(Ty::U32 | Ty::I32 | Ty::Enum(_)) => true,
            Some(Ty::Tuple(t)) => t.elems().iter().any(|e| !matches!(e, Elem::Struct(_))),
            _ => false,
        };
        let fns: Vec<usize> = g
            .arm_return_fns
            .iter()
            .copied()
            .filter(|&k| usable(g.table.fns[k].ret))
            .collect();
        if !fns.is_empty() {
            let f = fns[g.b.below(fns.len() as u32) as usize];
            let ret = g.table.fns[f].ret.expect("a return type");
            if let Some((recv, args)) = g.call_parts(f, 3) {
                let owner = if g.table.fns[f].recv == Recv::None {
                    g.table.fns[f].owner
                } else {
                    None
                };
                let call = Expr::Call {
                    f,
                    owner,
                    recv,
                    args,
                    ty: ret,
                };
                let as_repr = |g: &Gen, e: u8, value: Expr| Expr::EnumAs {
                    to: g.table.enums[usize::from(e)].repr,
                    e: Box::new(value),
                };
                arm_call = Some(match ret {
                    Ty::Enum(e) => as_repr(&g, e, call),
                    Ty::Tuple(t) => {
                        let ks: Vec<u8> = (0..t.n)
                            .filter(|&k| !matches!(t.elems()[usize::from(k)], Elem::Struct(_)))
                            .collect();
                        let index = ks[g.b.below(ks.len() as u32) as usize];
                        let field = Expr::TupleField {
                            e: Box::new(call),
                            index,
                        };
                        match t.elems()[usize::from(index)] {
                            Elem::Enum(e) => as_repr(&g, e, field),
                            _ => field,
                        }
                    }
                    _ => call,
                });
            }
        }
    }
    // Lets that the result hoisted come after the body and before the result.
    body.append(&mut g.pre);
    // Fold the final value of up to two top-level locals into the result, so that the computed values are live.
    let live: Vec<(u32, Ty)> = body
        .iter()
        .flat_map(|stmt| match stmt {
            Stmt::Let { var, ty, .. } if !ty.is_struct() && !ty.is_tuple() && !ty.is_array() => {
                vec![(*var, *ty)]
            }
            Stmt::LetMatch { var, ty, .. } => vec![(*var, *ty)],
            Stmt::LetTuple {
                vars,
                ty: Ty::Tuple(t),
                ..
            } => vars
                .iter()
                .zip(t.elems())
                .filter(|(_, elem)| !matches!(elem, Elem::Struct(_)))
                .map(|(&(var, _), elem)| (var, elem.ty()))
                .collect(),
            _ => Vec::new(),
        })
        .collect();
    // ... and one element of up to two tuple locals (the value space).
    let tuple_locals: Vec<(u32, TupleTy)> = body
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::Let {
                var,
                ty: Ty::Tuple(t),
                ..
            } => Some((*var, *t)),
            _ => None,
        })
        .collect();
    // ... and a scalar field of up to two struct locals (the method space), so that the state the calls left is live.
    if g.methods {
        let structs: Vec<(u32, u8)> = body
            .iter()
            .filter_map(|stmt| match stmt {
                Stmt::Let {
                    var,
                    ty: Ty::Struct(j),
                    ..
                } => Some((*var, *j)),
                _ => None,
            })
            .collect();
        let mut folded = Vec::new();
        for _ in 0..2 {
            if structs.is_empty() || !g.b.chance(85) {
                continue;
            }
            let (var, j) = structs[g.b.below(structs.len() as u32) as usize];
            let fields = g.table.structs[usize::from(j)].fields.clone();
            let k = g.b.below(fields.len() as u32) as u8;
            // A field is folded once: `x ^ x` is zero and would hide the state.
            if fields[usize::from(k)].is_struct() || folded.contains(&(var, k)) {
                continue;
            }
            folded.push((var, k));
            let place = Place::var(var, Ty::Struct(j)).field(k);
            let read = Expr::Read {
                place,
                ty: fields[usize::from(k)],
            };
            result = fold_in(result, read);
        }
    }
    if let Some(call) = arm_call {
        result = fold_in(result, call);
    }
    if g.values {
        for _ in 0..2 {
            if tuple_locals.is_empty() || !g.b.chance(85) {
                continue;
            }
            let (var, t) = tuple_locals[g.b.below(tuple_locals.len() as u32) as usize];
            let scalars: Vec<u8> = (0..t.n)
                .filter(|&k| !matches!(t.elems()[usize::from(k)], Elem::Struct(_)))
                .collect();
            if scalars.is_empty() {
                continue;
            }
            let k = scalars[g.b.below(scalars.len() as u32) as usize];
            let elem_ty = t.elems()[usize::from(k)].ty();
            let place = Place::var(var, Ty::Tuple(t)).field(k);
            let read = Expr::Read { place, ty: elem_ty };
            let read = match elem_ty {
                Ty::Enum(e) => Expr::EnumAs {
                    to: g.table.enums[usize::from(e)].repr,
                    e: Box::new(read),
                },
                _ => read,
            };
            result = fold_in(result, read);
        }
    }
    // The field that a top-level `t.k.f op= t.k.m()` set (fix round 1): read in the result, so that the order of the
    // place read and the call decides the output. A field is folded once.
    if g.values {
        let mut folded: Vec<Place> = Vec::new();
        for stmt in &body {
            if let Stmt::Set {
                place,
                ty,
                op: Some(op),
                value,
            } = stmt
                && crate::ast::tuple_set_names(&g.table, place, Some(*op), value)
                    .contains(&crate::ast::SHAPE_TUPLE_SET_CALL)
                && tuple_locals
                    .iter()
                    .any(|(var, _)| place.root == Root::Var(*var))
                && !folded.contains(place)
            {
                folded.push(place.clone());
                let read = Expr::Read {
                    place: place.clone(),
                    ty: *ty,
                };
                result = fold_in(result, read);
            }
        }
    }
    for read in array_folds {
        result = fold_in(result, read);
    }
    // ... and often a call on one struct root, so that the final store has a `&mut self` call in its value.
    if g.arrays
        && matches!(result.ty(), Ty::U32 | Ty::I32)
        && g.b.chance(50)
        && let Some(root) = g.bias_root()
        && let Some(call) = g.root_call(result.ty(), root, 3)
    {
        result = fold_in(result, call);
    }
    if g.closures && matches!(result.ty(), Ty::U32 | Ty::I32) {
        for probe in g.closure_probe_calls() {
            result = fold_in(result, probe);
        }
    }
    // ... and, in the closure space, a call of a closure whose arguments call it: the store's value then holds the call.
    if g.closures
        && matches!(result.ty(), Ty::U32 | Ty::I32)
        && g.b.chance(45)
        && let Some(root) = g.bias_root()
        && let Some(call) = g.closure_call(result.ty(), 3, Some(root))
    {
        result = fold_in(result, call);
    }
    g.store = g.array_store(&result);
    let (folds, fold_chance) = if g.vectors { (3, 85) } else { (2, 60) };
    for _ in 0..folds {
        if live.is_empty() || !g.b.chance(fold_chance) {
            continue;
        }
        let (var, var_ty) = live[g.b.below(live.len() as u32) as usize];
        let local = match var_ty {
            // A vector local joins the result through a component or (`f32`) its squared length; a matrix through a
            // component of its product with a fresh vector.
            Ty::Vec(lane, n) => {
                let vector = Expr::Var(var, var_ty);
                if lane == Lane::F32 && g.b.chance(35) {
                    Expr::dot(vector.clone(), vector)
                } else {
                    Expr::comp(vector, g.b.below(u32::from(n)) as u8)
                }
            }
            Ty::Mat(n) => {
                let column = g.build(Ty::Vec(Lane::F32, n), 1);
                let product = Expr::agg_bin(BinOp::Mul, Expr::Var(var, var_ty), column);
                Expr::comp(product, g.b.below(u32::from(n)) as u8)
            }
            // An enum local joins the result as its repr value.
            Ty::Enum(k) => Expr::EnumAs {
                to: g.table.enums[usize::from(k)].repr,
                e: Box::new(Expr::Var(var, var_ty)),
            },
            _ => Expr::Var(var, var_ty),
        };
        result = fold_in(result, local);
    }
    Program {
        seed,
        attempt,
        count,
        inputs,
        body,
        result,
        methods: g.table,
        store: g.store.take(),
    }
}

/// `result` combined with a local (`bit-exact` operators only: no new overflow, division or shift conditions).
fn fold_in(result: Expr, local: Expr) -> Expr {
    let (ty, local_ty) = (result.ty(), local.ty());
    match (ty, local_ty) {
        // A float local joins a float result as one more term, if the chain stays within four written terms.
        (Ty::F32, Ty::F32) if chain_terms(&result) < 4 => Expr::bin(BinOp::Add, result, local),
        (Ty::F32, _) | (Ty::Bool, Ty::F32) => result,
        (Ty::Bool, Ty::Bool) => Expr::bin(BinOp::Or, result, local),
        (Ty::Bool, _) => Expr::bin(
            BinOp::And,
            result,
            Expr::bin(BinOp::Ne, local, Expr::Lit(local_ty, 0)),
        ),
        (_, Ty::Bool) => result,
        (_, _) if ty == local_ty => Expr::bin(BinOp::BitXor, result, local),
        _ => Expr::bin(BinOp::BitXor, result, Expr::cast(ty, local)),
    }
}

/// The program of `seed` at `attempt`, whether or not it is in the domain, in the space that the attempt number
/// selects (`space_of_attempt`).
pub fn generate_attempt(seed: u64, attempt: u64) -> Program {
    generate_attempt_in(space_of_attempt(attempt), seed, attempt)
}

/// The program of `seed` at `attempt` in `space`, whether or not it is in the domain.
pub fn generate_attempt_in(space: Space, seed: u64, attempt: u64) -> Program {
    decode_in(
        space,
        &byte_stream(seed, attempt, BYTES_PER_PROGRAM),
        seed,
        attempt,
    )
}

/// The in-domain programs of one seed and what was discarded on the way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Batch {
    pub seed: u64,
    pub programs: Vec<Program>,
    /// Attempts made, accepted or not (both parts of the batch).
    pub attempts: u64,
    /// Discarded attempts by reason (`Violation::label`, `static`, `no-input-read` or, in the vector part,
    /// `no-aggregate`), both parts of the batch.
    pub discarded: BTreeMap<&'static str, u32>,
    /// Of `attempts`, the attempts of the vector part, and of `discarded`, the discards of the vector part.
    pub vector_attempts: u64,
    pub vector_discarded: BTreeMap<&'static str, u32>,
    /// Of `attempts`, the attempts of the method part, and of `discarded`, the discards of the method part.
    pub method_attempts: u64,
    pub method_discarded: BTreeMap<&'static str, u32>,
    /// Of `attempts`, the attempts of the enum part, and of `discarded`, the discards of the enum part.
    pub enum_attempts: u64,
    pub enum_discarded: BTreeMap<&'static str, u32>,
    /// Of `attempts`, the attempts of the value part, and of `discarded`, the discards of the value part.
    pub value_attempts: u64,
    pub value_discarded: BTreeMap<&'static str, u32>,
    /// Of `attempts`, the attempts of the array part, and of `discarded`, the discards of the array part.
    pub array_attempts: u64,
    pub array_discarded: BTreeMap<&'static str, u32>,
    /// Of `attempts`, the attempts of the closure part, and of `discarded`, the discards of the closure part.
    pub closure_attempts: u64,
    pub closure_discarded: BTreeMap<&'static str, u32>,
    /// Of `attempts`, the attempts of the trait part, and of `discarded`, the discards of the trait part.
    pub trait_attempts: u64,
    pub trait_discarded: BTreeMap<&'static str, u32>,
}

impl Batch {
    pub fn discarded_total(&self) -> u32 {
        self.discarded.values().sum()
    }
}

/// Why a program is not a test: it reads no input at all, holds a closed constant outside the domain anywhere (`static`,
/// dead code included), or leaves the domain for some input.
pub fn discard_reason(program: &Program) -> Option<&'static str> {
    if !program.reads_input() {
        return Some("no-input-read");
    }
    if interp::static_violation(program).is_some() {
        return Some("static");
    }
    interp::run(program)
        .err()
        .map(|failure| failure.violation.label())
}

/// Generates the `PROGRAMS_PER_SEED` in-domain programs of `seed`: the first `SCALAR_PROGRAMS_PER_SEED` of the float
/// space (attempts `0..`, the programs of DIFFTEST_FORMAT 5), then `VECTOR_PROGRAMS_PER_SEED` of the vector space
/// (attempts `VECTOR_ATTEMPT_BASE..`), then `METHOD_PROGRAMS_PER_SEED` of the method space (attempts
/// `METHOD_ATTEMPT_BASE..`).
pub fn generate_batch(seed: u64) -> Result<Batch, String> {
    let mut batch = empty_batch(seed);
    fill(&mut batch, Space::Float, SCALAR_PROGRAMS_PER_SEED)?;
    fill(&mut batch, Space::Vector, VECTOR_PROGRAMS_PER_SEED)?;
    fill(&mut batch, Space::Method, METHOD_PROGRAMS_PER_SEED)?;
    fill(&mut batch, Space::Enum, ENUM_PROGRAMS_PER_SEED)?;
    fill(&mut batch, Space::Value, VALUE_PROGRAMS_PER_SEED)?;
    fill(&mut batch, Space::Array, ARRAY_PROGRAMS_PER_SEED)?;
    fill(&mut batch, Space::Closure, CLOSURE_PROGRAMS_PER_SEED)?;
    fill(&mut batch, Space::Trait, TRAIT_PROGRAMS_PER_SEED)?;
    Ok(batch)
}

/// Generates the `PROGRAMS_PER_SEED` in-domain programs of `seed` all in `space`, in attempt order.
pub fn generate_batch_in(space: Space, seed: u64) -> Result<Batch, String> {
    let mut batch = empty_batch(seed);
    fill(&mut batch, space, PROGRAMS_PER_SEED)?;
    Ok(batch)
}

fn empty_batch(seed: u64) -> Batch {
    Batch {
        seed,
        programs: Vec::new(),
        attempts: 0,
        discarded: BTreeMap::new(),
        vector_attempts: 0,
        vector_discarded: BTreeMap::new(),
        method_attempts: 0,
        method_discarded: BTreeMap::new(),
        enum_attempts: 0,
        enum_discarded: BTreeMap::new(),
        value_attempts: 0,
        value_discarded: BTreeMap::new(),
        array_attempts: 0,
        array_discarded: BTreeMap::new(),
        closure_attempts: 0,
        closure_discarded: BTreeMap::new(),
        trait_attempts: 0,
        trait_discarded: BTreeMap::new(),
    }
}

/// Adds `want` more in-domain programs of `space` to `batch`, from the first attempt number of the space.
fn fill(batch: &mut Batch, space: Space, want: usize) -> Result<(), String> {
    let first = match space {
        Space::Vector => VECTOR_ATTEMPT_BASE,
        Space::Method => METHOD_ATTEMPT_BASE,
        Space::Enum => ENUM_ATTEMPT_BASE,
        Space::Value => VALUE_ATTEMPT_BASE,
        Space::Array => ARRAY_ATTEMPT_BASE,
        Space::Closure => CLOSURE_ATTEMPT_BASE,
        Space::Trait => TRAIT_ATTEMPT_BASE,
        _ => 0,
    };
    let target = batch.programs.len() + want;
    for attempt in first..first + MAX_ATTEMPTS {
        batch.attempts += 1;
        batch.vector_attempts += u64::from(space == Space::Vector);
        batch.method_attempts += u64::from(space == Space::Method);
        batch.enum_attempts += u64::from(space == Space::Enum);
        batch.value_attempts += u64::from(space == Space::Value);
        batch.array_attempts += u64::from(space == Space::Array);
        batch.closure_attempts += u64::from(space == Space::Closure);
        batch.trait_attempts += u64::from(space == Space::Trait);
        let program = generate_attempt_in(space, batch.seed, attempt);
        // A vector program that came out scalar tests nothing new, nor does a method program that calls nothing, nor
        // an enum program without a `match`.
        let reason = if space == Space::Vector && !program.has_aggregate() {
            Some("no-aggregate")
        } else if space == Space::Method && !program.has_call() {
            Some("no-call")
        } else if space == Space::Enum && !program.has_match() {
            Some("no-match")
        } else if space == Space::Enum && !program.has_call() {
            Some("no-call")
        } else if space == Space::Value && !program.has_value_form() {
            Some("no-value")
        } else if space == Space::Closure && !program.has_closure() {
            Some("no-closure")
        } else if space == Space::Trait && !program.has_trait_forms() {
            Some("no-trait")
        } else if matches!(space, Space::Array | Space::Closure) && !program.has_array() {
            Some("no-array")
        } else if space == Space::Trait && !program.has_match() {
            Some("no-match")
        } else if matches!(space, Space::Value | Space::Array | Space::Closure)
            && !program.has_call()
        {
            Some("no-call")
        } else {
            discard_reason(&program)
        };
        match reason {
            Some(reason) => {
                *batch.discarded.entry(reason).or_default() += 1;
                if space == Space::Vector {
                    *batch.vector_discarded.entry(reason).or_default() += 1;
                }
                if space == Space::Method {
                    *batch.method_discarded.entry(reason).or_default() += 1;
                }
                if space == Space::Enum {
                    *batch.enum_discarded.entry(reason).or_default() += 1;
                }
                if space == Space::Array {
                    *batch.array_discarded.entry(reason).or_default() += 1;
                }
                if space == Space::Closure {
                    *batch.closure_discarded.entry(reason).or_default() += 1;
                }
                if space == Space::Trait {
                    *batch.trait_discarded.entry(reason).or_default() += 1;
                }
                if space == Space::Value {
                    *batch.value_discarded.entry(reason).or_default() += 1;
                }
            }
            None => batch.programs.push(program),
        }
        if batch.programs.len() == target {
            return Ok(());
        }
    }
    Err(format!(
        "seed {}: only {} in-domain programs of {space:?} in {MAX_ATTEMPTS} attempts",
        batch.seed,
        batch.programs.len()
    ))
}

/// A copy of a struct place as an expression (a bare local is a `Var`, a field path or `self` a `Read`).
fn place_value(place: &Place, ty: Ty) -> Expr {
    match place.root {
        Root::Var(var) if place.path.is_empty() => Expr::Var(var, ty),
        _ => Expr::Read {
            place: place.clone(),
            ty,
        },
    }
}

/// The method space (format 8): the structs and functions of a program, and the calls and assignments that use them.
/// Every rule of `ast::Check` about receivers, borrows and the call graph is kept here by construction, so that no
/// refused form is drawn (a unit test compiles a sample of seeds through the macro).
impl Gen<'_> {
    /// Every struct-typed place in reach (a local, a parameter, `self`, and the struct fields nested in them), with
    /// the struct and whether the place may be written.
    fn struct_places(&self) -> Vec<(Place, u8, bool)> {
        let mut work = Vec::new();
        for &(var, ty, mutable) in &self.scope {
            if let Ty::Struct(j) = ty {
                work.push((Place::var(var, ty), j, mutable));
            }
        }
        if let Some((ty @ Ty::Struct(j), recv)) = self.this {
            work.push((Place::this(ty), j, recv == Recv::Mut));
        }
        // The struct elements of tuple locals and parameters (the value space).
        for &(var, ty, mutable) in &self.scope {
            if let Ty::Tuple(t) = ty {
                for (k, elem) in t.elems().iter().enumerate() {
                    if let Elem::Struct(j) = elem {
                        work.push((Place::var(var, ty).field(k as u8), *j, mutable));
                    }
                }
            }
        }
        let mut out = Vec::new();
        while let Some((place, j, writable)) = work.pop() {
            for (k, field) in self.table.structs[usize::from(j)].fields.iter().enumerate() {
                if let Ty::Struct(inner) = field {
                    work.push((place.field(k as u8), *inner, writable));
                }
            }
            out.push((place, j, writable));
        }
        out
    }

    /// The scalar fields of type `ty` of the struct places in reach, with whether they may be written.
    fn scalar_places(&self, ty: Ty) -> Vec<(Place, bool)> {
        let mut out = Vec::new();
        for (place, j, writable) in self.struct_places() {
            for (k, field) in self.table.structs[usize::from(j)].fields.iter().enumerate() {
                if *field == ty {
                    out.push((place.field(k as u8), writable));
                }
            }
        }
        // The scalar elements of tuple locals and parameters.
        for &(var, vty, mutable) in &self.scope {
            if let Ty::Tuple(t) = vty {
                for (k, elem) in t.elems().iter().enumerate() {
                    if !matches!(elem, Elem::Struct(_)) && elem.ty() == ty {
                        out.push((Place::var(var, vty).field(k as u8), mutable));
                    }
                }
            }
        }
        out
    }

    /// Whether an ordinary call may name function `k`: not a generic function (its arguments are drawn by `generic_call`) and
    /// not the impl of an enum (its receiver is no struct place).
    fn plain(&self, k: usize) -> bool {
        let def = &self.table.fns[k];
        def.gens.is_empty() && !(def.recv != Recv::None && def.owner.is_none())
    }

    /// The functions a call may name here: those below the one being drawn (the kernel may name all).
    fn callable(&self) -> usize {
        self.cur_fn.unwrap_or(self.table.fns.len())
    }

    /// The places an assignment may write: scalar fields, and (never `*self`) whole struct places.
    fn set_targets(&self) -> Vec<(Place, Ty)> {
        let mut out = Vec::new();
        for (place, j, writable) in self.struct_places() {
            if !writable {
                continue;
            }
            for (k, field) in self.table.structs[usize::from(j)].fields.iter().enumerate() {
                if !field.is_struct() {
                    out.push((place.field(k as u8), *field));
                }
            }
            if !(place.root == Root::SelfV && place.path.is_empty()) {
                out.push((place, Ty::Struct(j)));
            }
        }
        out
    }

    fn set_stmt(&mut self, targets: &[(Place, Ty)]) -> Stmt {
        let (place, ty) = targets[self.b.below(targets.len() as u32) as usize].clone();
        let (op, value) = match ty {
            // A whole-struct assignment; if no value of the struct is in reach, the place is assigned to itself.
            Ty::Struct(j) => (
                None,
                self.struct_expr(j, 3, true)
                    .unwrap_or_else(|| place_value(&place, ty)),
            ),
            _ => {
                debug_assert!(ty.is_int(), "a struct field is a u32 or an i32");
                let op = self.b.chance(70).then(|| {
                    [
                        BinOp::Add,
                        BinOp::Add,
                        BinOp::Sub,
                        BinOp::BitXor,
                        BinOp::BitXor,
                        BinOp::BitOr,
                        BinOp::Mul,
                        BinOp::Div,
                        BinOp::Rem,
                        BinOp::BitAnd,
                        BinOp::Shl,
                        BinOp::Shr,
                        // `&=`, `>>=` and `%=` again: the same-root forms of each are reached with the call as the value.
                        BinOp::BitAnd,
                        BinOp::Shr,
                        BinOp::Rem,
                    ][self.b.below(15) as usize]
                });
                // A shift takes a `u32` amount whatever the place's type.
                let value_ty = if op.is_some_and(|op| op.is_shift()) {
                    Ty::U32
                } else {
                    ty
                };
                // A share of the values call a `&mut self` method on the place's own root: the order rules of
                // `place op= value` (the place is read after the value) are only observable then.
                let biased = if self.b.chance(75) {
                    // A closure call whose arguments call the place's root: the order of the place read and of the
                    // arguments decides the stored value.
                    let via = if self.closures && !self.cls.is_empty() && self.b.chance(40) {
                        self.closure_call(value_ty, 3, Some(place.root))
                    } else {
                        None
                    };
                    via.or_else(|| self.root_call(value_ty, place.root, 3))
                } else {
                    None
                };
                let value = match (biased, op) {
                    (Some(call), Some(BinOp::Div | BinOp::Rem)) => {
                        Expr::bin(BinOp::BitOr, call, Expr::Lit(value_ty, 1))
                    }
                    // A shift amount is written `call & 31`, so the call's value stays below the width.
                    (Some(call), Some(BinOp::Shl | BinOp::Shr)) => {
                        Expr::bin(BinOp::BitAnd, call, Expr::u32(31))
                    }
                    (Some(call), _) => call,
                    (None, Some(BinOp::Div | BinOp::Rem)) => self.divisor(ty, 3),
                    (None, Some(BinOp::Shl | BinOp::Shr)) => self.shift_amount(3),
                    (None, _) => self.expr(ty, 3),
                };
                (op, value)
            }
        };
        Stmt::Set {
            place,
            ty,
            op,
            value,
        }
    }

    /// The `&mut self` methods that can be called as a statement here.
    fn statement_calls(&self) -> Vec<usize> {
        let places = self.struct_places();
        (0..self.callable())
            .filter(|&k| {
                let def = &self.table.fns[k];
                self.plain(k)
                    && def.recv == Recv::Mut
                    && places
                        .iter()
                        .any(|(_, j, writable)| *writable && Some(*j) == def.owner)
            })
            .collect()
    }

    /// A `&mut self` call as a statement; `None` when an argument of struct type has no value in reach.
    fn statement_call(&mut self, calls: &[usize]) -> Option<Stmt> {
        let f = calls[self.b.below(calls.len() as u32) as usize];
        let (recv, args) = self.call_parts(f, 3)?;
        Some(Stmt::Call {
            f,
            owner: None,
            recv,
            args,
        })
    }

    /// The receiver and the arguments of a call of function `f`, or `None` when no receiver is in reach. A `&mut
    /// self` call needs a writable receiver whose root no enclosing call is borrowing; while the arguments of a
    /// call that borrows its receiver (`&self`, `&mut self`) are drawn, no `&mut self` call on that root is drawn.
    fn call_parts(&mut self, f: usize, depth: u32) -> Option<(Option<Place>, Vec<Expr>)> {
        self.call_parts_on(f, depth, None)
    }

    /// `call_parts` with the receiver chosen by the caller (`forced`), or drawn when `None`.
    fn call_parts_on(
        &mut self,
        f: usize,
        depth: u32,
        forced: Option<Place>,
    ) -> Option<(Option<Place>, Vec<Expr>)> {
        let def = self.table.fns[f].clone();
        let recv = if def.recv == Recv::None {
            None
        } else if forced.is_some() {
            forced
        } else {
            let places: Vec<Place> = self
                .struct_places()
                .into_iter()
                .filter(|(place, j, writable)| {
                    Some(*j) == def.owner
                        && (def.recv != Recv::Mut
                            || (*writable && !self.forbid.contains(&place.root)))
                })
                .map(|(place, ..)| place)
                .collect();
            if places.is_empty() {
                return None;
            }
            Some(places[self.b.below(places.len() as u32) as usize].clone())
        };
        let pushed = recv
            .as_ref()
            .filter(|_| def.recv != Recv::Value)
            .map(|place| place.root);
        if let Some(root) = pushed {
            self.forbid.push(root);
        }
        let mut args = Vec::new();
        let mut complete = true;
        let mut enum_root: Option<Root> = None;
        // A trait method called on one root often gets calls on another root among its arguments.
        let trait_root = if self.traits && def.imp.is_some() && def.recv != Recv::Value {
            self.pick_writable_root()
        } else {
            None
        };
        for (slot, (_, ty)) in def.params.iter().enumerate() {
            match ty {
                scalar if trait_root.is_some() && self.b.chance(65) => {
                    let root = trait_root.expect("a root");
                    let call = self.root_call(*scalar, root, depth.saturating_sub(1));
                    args.push(call.unwrap_or_else(|| self.expr(*scalar, depth)));
                }
                // A comparison before the last argument: the shape that made the macro emit `__gust_keep` (m9.f1).
                Ty::Bool if self.values && slot + 1 < def.params.len() && self.b.chance(45) => {
                    let operand = self.int_ty();
                    args.push(Expr::bin(
                        BinOp::Lt,
                        self.expr(operand, depth.min(2)),
                        self.expr(operand, depth.min(2)),
                    ));
                }
                Ty::Tuple(_) => args.push(self.tuple_expr(*ty, depth)),
                Ty::Array(a) => args.push(self.array_arg(*a, depth)),
                // An enum argument is often built from a call, and two of them from calls on one root: the arguments
                // run left to right, so the first call's effect on the root is what the second one sees.
                Ty::Enum(_) => args.push(self.enum_arg(*ty, depth, &mut enum_root)),
                // A struct literal is only a `let` initializer or the right side of an assignment (E0810): an
                // argument is a local, a field, or the value of a call.
                Ty::Struct(j) => match self.struct_expr(*j, depth, false) {
                    Some(arg) => args.push(arg),
                    None => complete = false,
                },
                // A by-value call on `R` often gets a `&mut self` call on `R` among its arguments: the receiver is
                // copied before they run.
                scalar if def.recv == Recv::Value && self.b.chance(75) => {
                    let root = recv.as_ref().map(|place| place.root);
                    let call = root
                        .and_then(|root| self.root_call(*scalar, root, depth.saturating_sub(1)));
                    args.push(call.unwrap_or_else(|| self.expr(*scalar, depth)));
                }
                scalar => args.push(self.expr(*scalar, depth)),
            }
        }
        if pushed.is_some() {
            self.forbid.pop();
        }
        complete.then_some((recv, args))
    }

    /// A call whose value has type `ty` (a scalar, or a struct for an associated function), if one is in reach.
    fn value_call(&mut self, ty: Ty, depth: u32) -> Option<Expr> {
        if self.traits {
            if self.b.chance(30)
                && let Some(call) = self.generic_call(ty, depth)
            {
                return Some(call);
            }
            if matches!(ty, Ty::Enum(_) | Ty::U32 | Ty::I32 | Ty::Bool)
                && self.b.chance(25)
                && let Some(call) = self.trait_call(ty, depth)
            {
                return Some(call);
            }
        }
        let candidates: Vec<usize> = (0..self.callable())
            .filter(|&k| self.table.fns[k].ret == Some(ty) && self.plain(k))
            .collect();
        if candidates.is_empty() {
            return None;
        }
        // A third of the calls prefer a by-value method, whose receiver is copied before its arguments run.
        let by_value: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|&k| self.table.fns[k].recv == Recv::Value)
            .collect();
        let candidates = if !by_value.is_empty() && self.b.chance(33) {
            by_value
        } else {
            candidates
        };
        // The value space prefers the helpers that return from inside an arm: a `return` in an arm that runs.
        let returning: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|k| self.arm_return_fns.contains(k))
            .collect();
        let candidates = if self.values && !returning.is_empty() && self.b.chance(50) {
            returning
        } else {
            candidates
        };
        let f = candidates[self.b.below(candidates.len() as u32) as usize];
        let owner = {
            let def = &self.table.fns[f];
            if def.recv == Recv::None {
                def.owner
            } else {
                None
            }
        };
        let (recv, args) = self.call_parts(f, depth)?;
        let call = Expr::Call {
            f,
            owner,
            recv,
            args,
            ty,
        };
        Some(if self.traits {
            self.maybe_path(call)
        } else {
            call
        })
    }

    /// A `&mut self` call of a method returning `ty` on a writable place of root `root`, if one is in reach and no
    /// enclosing call is borrowing that root: the call whose effect on `root` a surrounding read or place can see.
    pub(super) fn root_call(&mut self, ty: Ty, root: Root, depth: u32) -> Option<Expr> {
        if depth == 0 || self.forbid.contains(&root) {
            return None;
        }
        // `c.pair().1`: an element of a `&mut self` method's tuple result.
        if self.values
            && self.b.chance(30)
            && let Some(field) = self.tuple_field_call(ty, depth, Some(root))
        {
            return Some(field);
        }
        let places = self.struct_places();
        let on_root = |owner: Option<u8>| {
            places
                .iter()
                .filter(|(place, j, writable)| *writable && place.root == root && Some(*j) == owner)
                .map(|(place, ..)| place.clone())
                .collect::<Vec<Place>>()
        };
        let candidates: Vec<usize> = (0..self.callable())
            .filter(|&k| {
                let def = &self.table.fns[k];
                self.plain(k)
                    && def.recv == Recv::Mut
                    && def.ret == Some(ty)
                    && !on_root(def.owner).is_empty()
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let f = candidates[self.b.below(candidates.len() as u32) as usize];
        let receivers = on_root(self.table.fns[f].owner);
        let recv = receivers[self.b.below(receivers.len() as u32) as usize].clone();
        let (recv, args) = self.call_parts_on(f, depth, Some(recv))?;
        Some(Expr::Call {
            f,
            owner: None,
            recv,
            args,
            ty,
        })
    }

    /// A binary whose operands are a read of a place and a `&mut self` call on its root (in either order, so the
    /// read comes before or after the call), or, for `bool`, a condition reading a place and a `&mut self` call on
    /// its root as the right operand of `&&`/`||`. `None` when there is no such pair in reach.
    fn root_binary(&mut self, ty: Ty, depth: u32) -> Option<Expr> {
        let mut readable: Vec<(Place, Ty)> = Vec::new();
        for scalar in [Ty::U32, Ty::I32] {
            readable.extend(
                self.scalar_places(scalar)
                    .into_iter()
                    .map(|(p, _)| (p, scalar)),
            );
        }
        if readable.is_empty() {
            return None;
        }
        let (place, read_ty) = readable[self.b.below(readable.len() as u32) as usize].clone();
        let depth = depth.max(1);
        if ty == Ty::Bool {
            let call = self.root_call(Ty::Bool, place.root, depth)?;
            let bound = self.expr(read_ty, 1);
            let cmp = [BinOp::Lt, BinOp::Ge, BinOp::Ne, BinOp::Eq][self.b.below(4) as usize];
            let read = Expr::Read { place, ty: read_ty };
            let op = [BinOp::And, BinOp::Or][self.b.below(2) as usize];
            return Some(Expr::bin(op, Expr::bin(cmp, read, bound), call));
        }
        if read_ty != ty {
            return None;
        }
        let call = self.root_call(ty, place.root, depth)?;
        let read = Expr::Read { place, ty };
        let op = [
            BinOp::Add,
            BinOp::Add,
            BinOp::Sub,
            BinOp::BitXor,
            BinOp::BitOr,
            BinOp::Mul,
        ][self.b.below(6) as usize];
        Some(if self.b.chance(50) {
            Expr::bin(op, read, call)
        } else {
            Expr::bin(op, call, read)
        })
    }

    /// The type and the initializer of a new struct local: struct `S0` (scalar fields only, so a literal is always
    /// possible) when the random struct has no value in reach.
    fn struct_init(&mut self) -> (u8, Expr) {
        let j = self.b.below(self.table.structs.len() as u32) as u8;
        if let Some(init) = self.struct_expr(j, 3, true) {
            return (j, init);
        }
        let init = self
            .struct_expr(0, 3, true)
            .expect("a literal of S0 is always possible");
        (0, init)
    }

    /// A value of struct `j`: a literal (only where `top` says a literal is allowed: a `let` initializer or the right
    /// side of an assignment), a call of an associated function, or a copy of a struct place. `None` when the kind
    /// drawn, and every other, has no value in reach.
    fn struct_expr(&mut self, j: u8, depth: u32, top: bool) -> Option<Expr> {
        let ty = Ty::Struct(j);
        if self.closures
            && depth > 0
            && !self.cls.is_empty()
            && self.b.chance(22)
            && let Some(call) = self.closure_call(ty, depth - 1, None)
        {
            return Some(call);
        }
        let bare_self_ok = matches!(self.this, Some((_, Recv::Value)));
        let places: Vec<Place> = self
            .struct_places()
            .into_iter()
            .filter(|(place, pj, _)| {
                *pj == j && !(place.root == Root::SelfV && place.path.is_empty() && !bare_self_ok)
            })
            .map(|(place, ..)| place)
            .collect();
        let assoc = (0..self.callable()).any(|k| self.table.fns[k].ret == Some(ty));
        let weights = [
            if top { 50 } else { 0 },
            if assoc && depth > 0 { 30 } else { 0 },
            if places.is_empty() { 0 } else { 25 },
        ];
        if weights.iter().sum::<u32>() == 0 {
            return None;
        }
        let first = self.b.weighted(&weights);
        // The kind drawn first, then the others in a fixed order if it has no value in reach.
        for kind in [first, 0, 1, 2] {
            if weights[kind] == 0 {
                continue;
            }
            match kind {
                0 => {
                    let fields = self.table.structs[usize::from(j)].fields.clone();
                    let mut values = Vec::new();
                    for field in fields {
                        let value = match field {
                            Ty::Struct(inner) => {
                                self.struct_expr(inner, depth.saturating_sub(1), false)
                            }
                            scalar => Some(self.expr(scalar, depth.min(2))),
                        };
                        match value {
                            Some(value) => values.push(value),
                            None => break,
                        }
                    }
                    if values.len() == self.table.structs[usize::from(j)].fields.len() {
                        // Written in a permuted order a quarter of the time: Rust evaluates the fields as written.
                        let mut order: Vec<u8> = (0..values.len() as u8).collect();
                        if values.len() > 1 && self.b.chance(25) {
                            for k in (1..order.len()).rev() {
                                let other = self.b.below(k as u32 + 1) as usize;
                                order.swap(k, other);
                                values.swap(k, other);
                            }
                        }
                        return Some(Expr::StructLit {
                            ty,
                            fields: values,
                            order,
                        });
                    }
                }
                1 => {
                    if let Some(call) = self.value_call(ty, depth - 1) {
                        return Some(call);
                    }
                }
                _ => {
                    let place = places[self.b.below(places.len() as u32) as usize].clone();
                    return Some(place_value(&place, ty));
                }
            }
        }
        None
    }

    /// The structs (one to three, each of one to three fields, scalars or earlier structs) and three to six functions
    /// (helpers, `&self`, `&mut self` and by-value `self` methods, associated functions), each calling only the
    /// functions below it.
    fn gen_tables(&mut self) {
        let structs = 1 + self.b.below(3) as usize;
        for j in 0..structs {
            let count = 1 + self.b.below(3);
            if self.traits && j > 0 && self.b.chance(55) {
                // A copy of the struct before: two types of one shape, so that a wrong impl stays observable.
                let fields = self.table.structs[j - 1].fields.clone();
                self.table.structs.push(StructDef { fields });
                continue;
            }
            let mut fields = Vec::new();
            for _ in 0..count {
                // No `bool` field: a GPU struct is a `GpuPod` type, and `bool` is not one.
                fields.push(
                    match self.b.weighted(&[50, 50, if j > 0 { 25 } else { 0 }]) {
                        0 => Ty::U32,
                        1 => Ty::I32,
                        _ => Ty::Struct(self.b.below(j as u32) as u8),
                    },
                );
            }
            self.table.structs.push(StructDef { fields });
            if self.table.flat_len(j as u8) > MAX_FLAT {
                self.table.structs[j].fields = vec![Ty::U32, Ty::I32];
            }
        }
        if self.enums {
            self.gen_enums();
        }
        if self.traits {
            self.gen_trait_tables();
            return;
        }
        let total = 3 + self.b.below(4) as usize;
        for k in 0..total {
            // The enum space needs a function that returns an enum: the last one is forced to if none is yet.
            self.force_enum_ret = self.enums
                && k + 1 == total
                && !self
                    .table
                    .fns
                    .iter()
                    .any(|def| matches!(def.ret, Some(Ty::Enum(_))));
            self.gen_fn(k);
        }
        self.force_enum_ret = false;
        if self.closures {
            self.counter_methods();
        }
    }

    fn gen_fn(&mut self, k: usize) {
        self.gen_fn_with(k, None);
    }

    /// `gen_fn`, or (a trait program, format 13) the struct impl of a trait method with the signature the trait gives:
    /// `forced.limit` is the index below which the body may call (the first impl of the pair).
    fn gen_fn_with(&mut self, k: usize, forced: Option<Forced>) {
        let structs = self.table.structs.len() as u32;
        // The first function is often a `&mut self` method returning `bool`: the right operand of an `&&`/`||`.
        let first_bool = forced.is_none() && k == 0 && self.b.chance(35);
        let kind = if forced.is_some() {
            1
        } else if first_bool {
            2
        } else if self.force_enum_ret {
            // Not an associated function: it returns its struct.
            [0, 1, 2, 3][self.b.weighted(&[10, 15, 55, 20])]
        } else {
            self.b.weighted(&[18, 18, 26, 18, 20])
        };
        let recv = forced.as_ref().map_or(
            [Recv::None, Recv::Shared, Recv::Mut, Recv::Value, Recv::None][kind],
            |fo| fo.recv,
        );
        // An associated function builds its struct from a literal, which needs every field to be a scalar.
        let scalar_structs: Vec<u8> = (0..structs as u8)
            .filter(|&j| {
                self.table.structs[usize::from(j)]
                    .fields
                    .iter()
                    .all(|f| !f.is_struct())
            })
            .collect();
        let owner = if let Some(fo) = &forced {
            fo.owner
        } else {
            (kind != 0).then(|| {
                if kind == 4 {
                    scalar_structs[self.b.below(scalar_structs.len() as u32) as usize]
                } else {
                    self.b.below(structs) as u8
                }
            })
        };
        self.cur_fn = Some(forced.as_ref().map_or(k, |fo| fo.limit));
        self.scope.clear();
        self.cls.clear();
        self.this = owner
            .filter(|_| recv != Recv::None)
            .map(|o| (Ty::Struct(o), recv));
        let mut params = Vec::new();
        // A by-value method takes at least one parameter, mostly a `u32`: the slot of a `&mut self` call's value.
        let count = if forced.is_some() {
            0
        } else if self.enums {
            1 + self.b.below(3)
        } else {
            self.b.below(3).max(u32::from(kind == 3))
        };
        let mut prev_enum = false;
        for slot in 0..count {
            let ty = if kind == 3 && slot == 0 && self.b.chance(70) {
                Ty::U32
            } else if self.enums && prev_enum && self.b.chance(60) {
                self.enum_ty()
            } else if self.b.chance(18) {
                Ty::Struct(self.b.below(structs) as u8)
            } else if self.arrays && self.b.chance(18) {
                self.array_param_ty()
            } else if self.values && self.b.chance(14) {
                self.tuple_ty()
            } else if self.enums && self.b.chance(30) {
                self.enum_ty()
            } else {
                [Ty::U32, Ty::I32, Ty::Bool][self.b.weighted(&[40, 40, 20])]
            };
            let var = self.fresh();
            prev_enum = ty.is_enum();
            self.scope.push((var, ty, false));
            params.push((var, ty));
        }
        if let Some(fo) = forced.as_ref() {
            for &ty in &fo.params {
                let var = self.fresh();
                self.scope.push((var, ty, false));
                params.push((var, ty));
            }
        }
        let ret = if let Some(fo) = &forced {
            fo.ret
        } else {
            match kind {
                _ if first_bool => Some(Ty::Bool),
                4 => owner.map(Ty::Struct),
                // A helper that takes an array often returns it (modified): the array in, the array out.
                0 if self.arrays
                    && params.iter().any(|(_, ty)| ty.is_array())
                    && self.b.chance(50) =>
                {
                    params.iter().map(|(_, ty)| *ty).find(|ty| ty.is_array())
                }
                _ if self.force_enum_ret => Some(self.enum_ty()),
                2 if self.b.chance(40) => None,
                _ if self.values && self.b.chance(28) => Some(self.tuple_ty()),
                _ if self.enums && self.b.chance(30) => Some(self.enum_ty()),
                _ => Some([Ty::U32, Ty::I32, Ty::Bool][self.b.weighted(&[40, 40, 20])]),
            }
        };
        // A helper (no receiver, no owner) may end a block in `return`.
        self.cur_ret = if self.enums
            && recv == Recv::None
            && owner.is_none()
            && !ret.is_some_and(Ty::is_array)
        {
            ret
        } else {
            None
        };
        let mut body = Vec::new();
        if recv == Recv::Mut {
            // A `&mut self` method writes the receiver at least once.
            let targets = self.set_targets();
            if !targets.is_empty() {
                let pre_mark = self.pre.len();
                body.push(self.set_stmt(&targets));
                self.splice_pre(pre_mark, 0, &mut body);
            }
        }
        // A helper of the value space often opens with a value `match` whose arms (all but the last) `return`.
        if self.values
            && let Some(ret) = self.cur_ret
            && !ret.is_struct()
            && self.b.chance(70)
        {
            let pre_mark = self.pre.len();
            let at = body.len();
            let forced = self.returning_match(ret);
            body.push(forced);
            self.splice_pre(pre_mark, at, &mut body);
        }
        // A function of the closure space often binds a closure first: the body and the tail then call it.
        if self.closures && self.b.chance(70) {
            let pre_mark = self.pre.len();
            let stmt = self.closure_let(1);
            body.push(stmt);
            self.splice_pre(pre_mark, 0, &mut body);
        }
        body.extend(self.block(1, 0, 3));
        let array_reads = if self.arrays {
            self.array_param_uses(&params, &mut body)
        } else {
            Vec::new()
        };
        let pre_mark = self.pre.len();
        let at = body.len();
        let tail = ret.map(|ty| match ty {
            Ty::Struct(j) => {
                // A struct literal is no tail (E0810): bind it first.
                let init = self
                    .struct_expr(j, 3, true)
                    .expect("a literal of a scalar-field struct is always possible");
                let var = self.fresh();
                body.push(Stmt::Let {
                    var,
                    ty,
                    mutable: false,
                    init,
                });
                Expr::Var(var, ty)
            }
            Ty::Array(_) => {
                let (var, _) = *params
                    .iter()
                    .find(|(_, p)| *p == ty)
                    .expect("an array result has its array parameter");
                Expr::Var(var, ty)
            }
            scalar => {
                let mut value = self.expr(scalar, 4);
                for read in array_reads
                    .clone()
                    .into_iter()
                    .filter(|_| matches!(scalar, Ty::U32 | Ty::I32 | Ty::Bool))
                {
                    value = fold_in(value, read);
                }
                if self.closures
                    && !self.cls.is_empty()
                    && matches!(scalar, Ty::U32 | Ty::I32 | Ty::Bool)
                    && self.b.chance(85)
                    && let Some(call) = self.closure_call(scalar, 3, None)
                {
                    value = fold_in(value, call);
                }
                value
            }
        });
        // A trait impl weighs its integer parameters into an integer result, so that the order of two arguments shows.
        let tail = if forced.is_some() {
            tail.map(|t| self.weight_params(t, &params))
        } else {
            tail
        };
        // Lets that the tail hoisted come before it (and before the `let` of a struct tail).
        self.splice_pre(pre_mark, at, &mut body);
        if self.values && arm_returns(&body, false) {
            self.arm_return_fns.push(k);
        }
        self.table.fns.push(FnDef {
            imp: forced.as_ref().map(|fo| fo.imp),
            gens: Vec::new(),
            owner,
            recv,
            params,
            ret,
            body,
            tail,
        });
        self.cur_fn = None;
        self.cur_ret = None;
        self.this = None;
        self.scope.clear();
        self.cls.clear();
    }
}

/// The enum space (format 9, slice m9.d1b): the enum tables, enum values and the statement `match`.
impl Gen<'_> {
    /// One to two enums of two to five variants, a `u32` or `i32` repr, literal discriminants: an explicit value from a
    /// list of boundaries, or the count-up of the macro (the previous value plus one). Values are distinct.
    fn gen_enums(&mut self) {
        const UNSIGNED: [u32; 13] = [
            0,
            1,
            2,
            3,
            5,
            7,
            10,
            100,
            255,
            65535,
            0x7FFF_FFFF,
            0x8000_0000,
            0xFFFF_FFF0,
        ];
        const SIGNED: [i32; 13] = [
            -100,
            -7,
            -2,
            -1,
            0,
            1,
            2,
            5,
            10,
            255,
            65535,
            i32::MAX - 3,
            i32::MIN + 5,
        ];
        let count = 1 + self.b.below(MAX_ENUMS as u32) as usize;
        for _ in 0..count {
            let repr = if self.b.chance(50) { Ty::U32 } else { Ty::I32 };
            let n = 2 + self.b.below(MAX_VARIANTS as u32 - 1) as usize;
            let mut variants: Vec<(u32, bool)> = Vec::new();
            for _ in 0..n {
                let up = match variants.last() {
                    None => Some(0u32),
                    Some(&(p, _)) if repr == Ty::I32 => (p as i32).checked_add(1).map(|v| v as u32),
                    Some(&(p, _)) => p.checked_add(1),
                };
                let taken = |v: u32, variants: &[(u32, bool)]| variants.iter().any(|x| x.0 == v);
                if !self.b.chance(45)
                    && let Some(v) = up
                    && !taken(v, &variants)
                {
                    variants.push((v, false));
                    continue;
                }
                let start = self.b.below(13) as usize;
                let value = (0..13)
                    .map(|k| {
                        let pick = (start + k) % 13;
                        if repr == Ty::I32 {
                            SIGNED[pick] as u32
                        } else {
                            UNSIGNED[pick]
                        }
                    })
                    .find(|&v| !taken(v, &variants))
                    .expect("thirteen values, at most four taken");
                variants.push((value, true));
            }
            self.table.enums.push(EnumDef { repr, variants });
        }
    }

    fn enum_ty(&mut self) -> Ty {
        Ty::Enum(self.b.below(self.table.enums.len() as u32) as u8)
    }

    /// A variable of the enum type or one of its variants.
    fn enum_leaf(&mut self, ty: Ty) -> Expr {
        let Ty::Enum(e) = ty else {
            unreachable!("an enum type")
        };
        let vars = self.vars(ty);
        if !vars.is_empty() && self.b.chance(60) {
            return Expr::Var(vars[self.b.below(vars.len() as u32) as usize], ty);
        }
        let v = self
            .b
            .below(self.table.enums[usize::from(e)].variants.len() as u32) as u8;
        Expr::Variant { e, v }
    }

    /// An enum value comes from a variant path, a variable (a local or a parameter) or the result of a call, never from
    /// an integer.
    fn enum_expr(&mut self, ty: Ty, depth: u32) -> Expr {
        if self.closures
            && depth > 0
            && !self.cls.is_empty()
            && self.b.chance(25)
            && let Some(call) = self.closure_call(ty, depth - 1, None)
        {
            return call;
        }
        if self.methods
            && depth > 0
            && self.b.chance(35)
            && let Some(call) = self.value_call(ty, depth - 1)
        {
            return call;
        }
        self.enum_leaf(ty)
    }

    /// A root with a writable struct place that no enclosing call is borrowing.
    fn pick_writable_root(&mut self) -> Option<Root> {
        let mut roots: Vec<Root> = Vec::new();
        for (place, _, writable) in self.struct_places() {
            if writable && !self.forbid.contains(&place.root) && !roots.contains(&place.root) {
                roots.push(place.root);
            }
        }
        if roots.is_empty() {
            return None;
        }
        Some(roots[self.b.below(roots.len() as u32) as usize])
    }

    /// An enum argument: often a `&mut self` call on a root shared with the other enum arguments of the call.
    fn enum_arg(&mut self, ty: Ty, depth: u32, shared: &mut Option<Root>) -> Expr {
        if depth >= 2 && self.b.chance(85) {
            let root = match *shared {
                Some(root) => Some(root),
                None => self.pick_writable_root(),
            };
            if let Some(root) = root
                && let Some(call) = self.root_call(ty, root, depth - 1)
            {
                *shared = Some(root);
                return call;
            }
        }
        self.enum_expr(ty, depth)
    }

    /// `==` or `!=` of two enum values; often two `&mut self` calls on one root (the left one runs first).
    fn enum_compare(&mut self, d: u32) -> Expr {
        let ty = self.enum_ty();
        let op = if self.b.chance(50) {
            BinOp::Eq
        } else {
            BinOp::Ne
        };
        if self.b.chance(75)
            && let Some(root) = self.pick_writable_root()
            && let Some(l) = self.root_call(ty, root, 2)
            && let Some(r) = self.root_call(ty, root, 2)
        {
            return Expr::bin(op, l, r);
        }
        let l = self.enum_expr(ty, d);
        let r = self.enum_expr(ty, d);
        Expr::bin(op, l, r)
    }

    /// A block ending in `return value` of the function's result type, in a helper, `percent` of the time.
    fn maybe_return(&mut self, body: &mut Vec<Stmt>, percent: u32) {
        if !self.enums {
            return;
        }
        // The value space returns from inside an arm twice as often (m9.d1b re-check, nit R1).
        let percent = if self.values { percent * 2 } else { percent };
        if let Some(ret) = self.cur_ret
            && self.b.chance(percent)
        {
            let value = self.expr(ret, 3);
            body.push(Stmt::Return { value });
        }
    }

    /// A statement `match` on an enum: the variants are shuffled and cut into groups (grouped arms); `_` replaces the
    /// last group 45 % of the time, otherwise every variant is named and the last arm is the `default` of the Slang
    /// `switch`. The scrutinee is a variable, a call (often a `&mut self` call on a root that an arm uses), or a path.
    fn match_stmt(&mut self, depth: u32) -> Stmt {
        let e = self.b.below(self.table.enums.len() as u32) as u8;
        let ty = Ty::Enum(e);
        let n = self.table.enums[usize::from(e)].variants.len();
        let mut bias_root = None;
        let mut scrut = None;
        match self.b.weighted(&[30, 45, 18, 7]) {
            1 => {
                if let Some(root) = self.pick_writable_root()
                    && let Some(call) = self.root_call(ty, root, 3)
                {
                    bias_root = Some(root);
                    scrut = Some(call);
                }
            }
            2 => scrut = self.value_call(ty, 2),
            3 => {
                let v = self.b.below(n as u32) as u8;
                scrut = Some(Expr::Variant { e, v });
            }
            _ => {}
        }
        let scrut = match scrut {
            Some(scrut) => scrut,
            None => self.enum_leaf(ty),
        };
        let scrut = self.enum_scrutinee(e, scrut);
        let mut order: Vec<u8> = (0..n as u8).collect();
        for k in (1..order.len()).rev() {
            let other = self.b.below(k as u32 + 1) as usize;
            order.swap(k, other);
        }
        let mut groups: Vec<Vec<u8>> = vec![vec![order[0]]];
        for &v in &order[1..] {
            if self.b.chance(40) {
                groups.last_mut().expect("a group").push(v);
            } else {
                groups.push(vec![v]);
            }
        }
        let wild = self.b.chance(45);
        if wild {
            if groups.len() == 1 {
                let last = groups[0].pop().expect("two variants at least");
                groups.push(vec![last]);
            }
            groups.pop();
        }
        let mut pats: Vec<Pat> = groups.into_iter().map(Pat::Variants).collect();
        if wild {
            pats.push(Pat::Wild);
        }
        let mut arms = Vec::new();
        for pat in pats {
            let mut body = self.block(depth + 1, 0, 2);
            // An arm that uses the root the scrutinee's call mutated: the call's effect is what the arm sees.
            if let Some(root) = bias_root
                && self.b.chance(70)
            {
                let targets: Vec<(Place, Ty)> = self
                    .set_targets()
                    .into_iter()
                    .filter(|(place, _)| place.root == root)
                    .collect();
                if !targets.is_empty() {
                    body.insert(0, self.set_stmt(&targets));
                }
            }
            self.maybe_return(&mut body, 30);
            arms.push(Arm { pat, body });
        }
        Stmt::Match { scrut, arms }
    }
}

/// The value space (format 10, slice m9.d1c): tuples, the value form of `match` and integer patterns. Every draw sits
/// behind `Gen::values`, so the older spaces decode as before.
impl Gen<'_> {
    /// A tuple `(S, u32)` or `(S, i32)` whose struct `S` has a `&mut self` method returning one of its scalar field
    /// types: the local that lets `tuple_field_set_stmt` write `t.0.f op= t.0.m()` (fix round 1).
    fn field_set_tuple_ty(&mut self) -> Option<Ty> {
        let mut options = Vec::new();
        for (j, def) in self.table.structs.iter().enumerate() {
            if def.fields.iter().any(|f| f.is_struct()) {
                continue;
            }
            let fits = (0..self.callable()).any(|m| {
                let f = &self.table.fns[m];
                f.recv == Recv::Mut
                    && f.owner == Some(j as u8)
                    && matches!(f.ret, Some(Ty::U32 | Ty::I32))
                    && f.ret.is_some_and(|r| {
                        def.fields
                            .iter()
                            .enumerate()
                            .any(|(k, field)| *field == r && writes_self_field(&f.body, k as u8))
                    })
            });
            if fits {
                options.push(j as u8);
            }
        }
        if options.is_empty() {
            return None;
        }
        let j = options[self.b.below(options.len() as u32) as usize];
        let other = if self.b.chance(50) {
            Elem::U32
        } else {
            Elem::I32
        };
        let ty = Ty::Tuple(TupleTy::new(&[Elem::Struct(j), other]));
        (self.table.ty_len(ty) <= MAX_TUPLE_FLAT).then_some(ty)
    }

    /// A tuple type of two to four elements (scalars, a struct of scalar fields, an enum), at most `MAX_TUPLE_FLAT`
    /// scalars once the structs are flattened.
    fn tuple_ty(&mut self) -> Ty {
        let n = 2 + self.b.weighted(&[50, 35, 15]);
        let scalar_structs: Vec<u8> = (0..self.table.structs.len() as u8)
            .filter(|&j| {
                self.table.structs[usize::from(j)]
                    .fields
                    .iter()
                    .all(|f| !f.is_struct())
            })
            .collect();
        let enums = self.table.enums.len() as u32;
        let mut elems = Vec::new();
        for _ in 0..n {
            let weights = [
                30,
                26,
                12,
                if scalar_structs.is_empty() { 0 } else { 30 },
                if enums == 0 { 0 } else { 14 },
            ];
            elems.push(match self.b.weighted(&weights) {
                0 => Elem::U32,
                1 => Elem::I32,
                2 => Elem::Bool,
                3 => {
                    Elem::Struct(scalar_structs[self.b.below(scalar_structs.len() as u32) as usize])
                }
                _ => Elem::Enum(self.b.below(enums) as u8),
            });
        }
        while self.table.ty_len(Ty::Tuple(TupleTy::new(&elems))) > MAX_TUPLE_FLAT {
            let last = elems
                .iter()
                .rposition(|e| matches!(e, Elem::Struct(_)))
                .expect("only structs make a tuple too long");
            elems[last] = Elem::U32;
        }
        Ty::Tuple(TupleTy::new(&elems))
    }

    /// A value of tuple type `ty`: a literal, a local, or the result of a call.
    fn tuple_expr(&mut self, ty: Ty, depth: u32) -> Expr {
        let Ty::Tuple(t) = ty else {
            unreachable!("a tuple type")
        };
        let vars = self.vars(ty);
        let calls = depth > 0 && (0..self.callable()).any(|k| self.table.fns[k].ret == Some(ty));
        let weights = [
            60,
            if vars.is_empty() { 0 } else { 18 },
            if calls { 26 } else { 0 },
        ];
        match self.b.weighted(&weights) {
            1 => Expr::Var(vars[self.b.below(vars.len() as u32) as usize], ty),
            2 => match self.value_call(ty, depth - 1) {
                Some(call) => call,
                None => self.tuple_literal(t, depth),
            },
            _ => self.tuple_literal(t, depth),
        }
    }

    /// `(a, b, c)`. Often the elements share one root: a `&mut self` call and a read or a copy of the same root, so that
    /// the order the elements run in is what the value shows.
    fn tuple_literal(&mut self, t: TupleTy, depth: u32) -> Expr {
        let root = if depth >= 1 && self.b.chance(65) {
            self.pick_writable_root()
        } else {
            None
        };
        let n = usize::from(t.n);
        let mut elems = Vec::new();
        for (k, elem) in t.elems().iter().enumerate() {
            elems.push(self.tuple_elem(*elem, depth.saturating_sub(1), root, k + 1 == n));
        }
        Expr::Tuple {
            ty: Ty::Tuple(t),
            elems,
        }
    }

    fn tuple_elem(&mut self, elem: Elem, depth: u32, root: Option<Root>, last: bool) -> Expr {
        let ty = elem.ty();
        match elem {
            Elem::Struct(j) => {
                if let Some(root) = root
                    && self.b.chance(55)
                    && let Some(copy) = self.struct_copy_on(j, root)
                {
                    return copy;
                }
                match self.struct_expr(j, depth, false) {
                    Some(value) => value,
                    None => self.hoist_struct(j),
                }
            }
            Elem::Enum(_) => {
                if let Some(root) = root
                    && self.b.chance(60)
                    && let Some(call) = self.root_call(ty, root, depth)
                {
                    return call;
                }
                self.enum_expr(ty, depth)
            }
            Elem::Bool if !last && self.b.chance(40) => {
                let operand = self.int_ty();
                Expr::bin(
                    BinOp::Lt,
                    self.expr(operand, depth.min(2)),
                    self.expr(operand, depth.min(2)),
                )
            }
            _ => {
                if let Some(root) = root {
                    if self.b.chance(50)
                        && let Some(call) = self.root_call(ty, root, depth)
                    {
                        return call;
                    }
                    if self.b.chance(30) {
                        let reads: Vec<Place> = self
                            .scalar_places(ty)
                            .into_iter()
                            .filter(|(place, _)| place.root == root)
                            .map(|(place, _)| place)
                            .collect();
                        if !reads.is_empty() {
                            let place = reads[self.b.below(reads.len() as u32) as usize].clone();
                            return Expr::Read { place, ty };
                        }
                    }
                }
                self.expr(ty, depth)
            }
        }
    }

    /// A copy of a struct-`j` place on `root` (the struct as it is when the element runs).
    fn struct_copy_on(&mut self, j: u8, root: Root) -> Option<Expr> {
        let bare_self_ok = matches!(self.this, Some((_, Recv::Value)));
        let places: Vec<Place> = self
            .struct_places()
            .into_iter()
            .filter(|(place, pj, _)| {
                *pj == j
                    && place.root == root
                    && !(place.root == Root::SelfV && place.path.is_empty() && !bare_self_ok)
            })
            .map(|(place, ..)| place)
            .collect();
        if places.is_empty() {
            return None;
        }
        let place = places[self.b.below(places.len() as u32) as usize].clone();
        Some(place_value(&place, Ty::Struct(j)))
    }

    /// A struct value that is a variable: a struct literal cannot be a tuple element (E0810), so it is bound by a `let`
    /// before the statement being drawn (`Gen::pre`).
    fn hoist_struct(&mut self, j: u8) -> Expr {
        let ty = Ty::Struct(j);
        let init = match self.struct_expr(j, 3, true) {
            Some(init) => init,
            // A nested struct field with no value in reach: hoist that one too, then a literal of variables.
            None => {
                let fields = self.table.structs[usize::from(j)].fields.clone();
                let values: Vec<Expr> = fields
                    .into_iter()
                    .map(|field| match field {
                        Ty::Struct(inner) => self.hoist_struct(inner),
                        scalar => self.expr(scalar, 2),
                    })
                    .collect();
                Expr::StructLit {
                    ty,
                    order: (0..values.len() as u8).collect(),
                    fields: values,
                }
            }
        };
        let var = self.fresh();
        let mutable = self.b.chance(50);
        self.scope.push((var, ty, mutable));
        self.pre.push(Stmt::Let {
            var,
            ty,
            mutable,
            init,
        });
        Expr::Var(var, ty)
    }

    /// Moves the lets that drawing since `mark` hoisted to `at` in `stmts` (before the statement that needs them).
    fn splice_pre(&mut self, mark: usize, at: usize, stmts: &mut Vec<Stmt>) {
        if self.pre.len() > mark {
            let hoisted = self.pre.split_off(mark);
            stmts.splice(at..at, hoisted);
        }
    }

    /// `f().k` for a function returning a tuple with an element of type `ty`; with `root`, a `&mut self` method on a
    /// writable place of that root (the call runs before the element is read, whatever follows).
    fn tuple_field_call(&mut self, ty: Ty, depth: u32, root: Option<Root>) -> Option<Expr> {
        if depth == 0 || root.is_some_and(|root| self.forbid.contains(&root)) {
            return None;
        }
        let places = self.struct_places();
        let on_root = |owner: Option<u8>, root: Root| {
            places
                .iter()
                .filter(|(place, j, writable)| *writable && place.root == root && Some(*j) == owner)
                .map(|(place, ..)| place.clone())
                .collect::<Vec<Place>>()
        };
        let matching = |t: TupleTy| -> Vec<u8> {
            (0..t.n)
                .filter(|&k| t.elems()[usize::from(k)].ty() == ty)
                .collect()
        };
        let candidates: Vec<usize> = (0..self.callable())
            .filter(|&k| {
                let def = &self.table.fns[k];
                let Some(Ty::Tuple(t)) = def.ret else {
                    return false;
                };
                !matching(t).is_empty()
                    && match root {
                        Some(root) => def.recv == Recv::Mut && !on_root(def.owner, root).is_empty(),
                        None => true,
                    }
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let f = candidates[self.b.below(candidates.len() as u32) as usize];
        let def = self.table.fns[f].clone();
        let Some(Ty::Tuple(t)) = def.ret else {
            unreachable!("a tuple result")
        };
        let forced = root.map(|root| {
            let receivers = on_root(def.owner, root);
            receivers[self.b.below(receivers.len() as u32) as usize].clone()
        });
        let (recv, args) = self.call_parts_on(f, depth, forced)?;
        let indexes = matching(t);
        let index = indexes[self.b.below(indexes.len() as u32) as usize];
        let owner = if def.recv == Recv::None {
            def.owner
        } else {
            None
        };
        Some(Expr::TupleField {
            e: Box::new(Expr::Call {
                f,
                owner,
                recv,
                args,
                ty: Ty::Tuple(t),
            }),
            index,
        })
    }

    /// The element places of tuple locals that may be assigned: `u32` and `i32` elements of a `let mut` tuple.
    fn tuple_set_targets(&self) -> Vec<(Place, Ty)> {
        let mut out = Vec::new();
        for &(var, ty, mutable) in &self.scope {
            let Ty::Tuple(t) = ty else { continue };
            if !mutable {
                continue;
            }
            for (k, elem) in t.elems().iter().enumerate() {
                if matches!(elem, Elem::U32 | Elem::I32) {
                    out.push((Place::var(var, ty).field(k as u8), elem.ty()));
                }
            }
        }
        out
    }

    /// The scalar fields of the struct elements of mutable tuple locals, each with a `&mut self` method of that struct
    /// returning the field's type: `(tuple var, tuple type, element, field, field type)`. A compound set of such a
    /// field whose value calls the method on the same element, `t.k.f *= t.k.m()`, is the one tuple-place assignment
    /// where the order of the place read and the value is observable (m9.d1c fix round 1).
    fn tuple_field_set_candidates(&self) -> Vec<(u32, Ty, u8, u8, Ty)> {
        let mut out = Vec::new();
        for &(var, ty, mutable) in &self.scope {
            let Ty::Tuple(t) = ty else { continue };
            if !mutable || self.forbid.contains(&Root::Var(var)) {
                continue;
            }
            for (k, elem) in t.elems().iter().enumerate() {
                let Elem::Struct(j) = elem else { continue };
                for (f, field) in self.table.structs[usize::from(*j)]
                    .fields
                    .iter()
                    .enumerate()
                {
                    let scalar = matches!(field, Ty::U32 | Ty::I32);
                    let method = (0..self.callable()).any(|m| {
                        let def = &self.table.fns[m];
                        def.recv == Recv::Mut
                            && def.owner == Some(*j)
                            && def.ret == Some(*field)
                            && writes_self_field(&def.body, f as u8)
                    });
                    if scalar && method {
                        out.push((var, ty, k as u8, f as u8, *field));
                    }
                }
            }
        }
        out
    }

    /// `t.k.f op= t.k.m();` for a `&mut self` method `m` of the struct at element `k` returning `f`'s type: Rust runs
    /// the call first, so the place is read after the call changed it.
    fn tuple_field_set_stmt(&mut self) -> Option<Stmt> {
        let candidates = self.tuple_field_set_candidates();
        if candidates.is_empty() {
            return None;
        }
        let (var, ty, k, f, field) = candidates[self.b.below(candidates.len() as u32) as usize];
        let Ty::Tuple(t) = ty else {
            unreachable!("a tuple local")
        };
        let Elem::Struct(j) = t.elems()[usize::from(k)] else {
            unreachable!("a struct element")
        };
        let methods: Vec<usize> = (0..self.callable())
            .filter(|&m| {
                let def = &self.table.fns[m];
                def.recv == Recv::Mut
                    && def.owner == Some(j)
                    && def.ret == Some(field)
                    && writes_self_field(&def.body, f)
            })
            .collect();
        let m = methods[self.b.below(methods.len() as u32) as usize];
        let element = Place::var(var, ty).field(k);
        let (recv, args) = self.call_parts_on(m, 3, Some(element.clone()))?;
        let op = [
            BinOp::Add,
            BinOp::Sub,
            BinOp::BitXor,
            BinOp::BitOr,
            BinOp::Mul,
        ][self.b.below(5) as usize];
        Some(Stmt::Set {
            place: element.field(f),
            ty: field,
            op: Some(op),
            value: Expr::Call {
                f: m,
                owner: None,
                recv,
                args,
                ty: field,
            },
        })
    }

    /// `t.a = t.b + c.bump();` or `t.a += c.bump();`: the value runs first, then the place is read (for `op=`).
    fn tuple_set_stmt(&mut self, targets: &[(Place, Ty)]) -> Option<Stmt> {
        if (targets.is_empty() || self.b.chance(85))
            && let Some(stmt) = self.tuple_field_set_stmt()
        {
            return Some(stmt);
        }
        if targets.is_empty() {
            return None;
        }
        let (place, ty) = targets[self.b.below(targets.len() as u32) as usize].clone();
        let op = self.b.chance(55).then(|| {
            [
                BinOp::Add,
                BinOp::Add,
                BinOp::Sub,
                BinOp::BitXor,
                BinOp::BitOr,
                BinOp::BitAnd,
            ][self.b.below(6) as usize]
        });
        // A `&mut self` call on a writable root, tried on up to three roots.
        let mut call = None;
        for _ in 0..3 {
            if call.is_some() {
                break;
            }
            call = self
                .pick_writable_root()
                .and_then(|root| self.root_call(ty, root, 3));
        }
        let value = match call {
            Some(call) if op.is_none() && self.b.chance(85) => {
                // Another element of the same tuple (or the target itself) read before the call runs.
                let Ty::Tuple(t) = place.root_ty else {
                    unreachable!("a tuple place")
                };
                let same: Vec<u8> = (0..t.n)
                    .filter(|&k| t.elems()[usize::from(k)].ty() == ty)
                    .collect();
                let k = same[self.b.below(same.len() as u32) as usize];
                let read = Expr::Read {
                    place: Place {
                        root: place.root,
                        root_ty: place.root_ty,
                        path: vec![k],
                        index: None,
                    },
                    ty,
                };
                let joint = [BinOp::Add, BinOp::Sub, BinOp::BitXor][self.b.below(3) as usize];
                if self.b.chance(50) {
                    Expr::bin(joint, read, call)
                } else {
                    Expr::bin(joint, call, read)
                }
            }
            Some(call) => call,
            None => self.expr(ty, 3),
        };
        Some(Stmt::Set {
            place,
            ty,
            op,
            value,
        })
    }

    /// `let (a, mut b): (T, U) = init;`. Often the source is a call of a function that returns a tuple, a `&mut self`
    /// method first, so that the root it changed is read afterwards.
    fn let_tuple(&mut self) -> Stmt {
        let rets: Vec<usize> = (0..self.callable())
            .filter(|&k| matches!(self.table.fns[k].ret, Some(Ty::Tuple(_))))
            .collect();
        let mut init = None;
        if !rets.is_empty() && self.b.chance(60) {
            let muts: Vec<usize> = rets
                .iter()
                .copied()
                .filter(|&k| self.table.fns[k].recv == Recv::Mut)
                .collect();
            let pool = if !muts.is_empty() && self.b.chance(65) {
                muts
            } else {
                rets
            };
            let f = pool[self.b.below(pool.len() as u32) as usize];
            let def = self.table.fns[f].clone();
            if let Some((recv, args)) = self.call_parts(f, 3) {
                init = Some(Expr::Call {
                    f,
                    owner: if def.recv == Recv::None {
                        def.owner
                    } else {
                        None
                    },
                    recv,
                    args,
                    ty: def.ret.expect("a tuple result"),
                });
            }
        }
        let init = match init {
            Some(init) => init,
            None => {
                let ty = self.tuple_ty();
                self.tuple_expr(ty, 3)
            }
        };
        let ty = init.ty();
        let Ty::Tuple(t) = ty else {
            unreachable!("a tuple initializer")
        };
        let annotate = t.elems().iter().any(|e| matches!(e, Elem::Enum(_))) || self.b.chance(50);
        let mut vars = Vec::new();
        for elem in t.elems() {
            let var = self.fresh();
            let mutable = self.b.chance(40);
            self.scope.push((var, elem.ty(), mutable));
            vars.push((var, mutable));
        }
        Stmt::LetTuple {
            vars,
            ty,
            annotate,
            init,
        }
    }

    /// The type of a value `match`: mostly an integer, sometimes `bool` or an enum.
    fn match_result_ty(&mut self) -> Ty {
        if self.b.chance(22) {
            self.enum_ty()
        } else {
            [Ty::U32, Ty::I32, Ty::Bool][self.b.weighted(&[45, 35, 20])]
        }
    }

    /// `let v: T = match scrut { p0 => return .., p1 => return .., _ => value };` in a helper that returns `ret`: every
    /// arm but the last returns, so that a `return` in an arm runs for some element (nit R1 of the m9.d1c brief).
    fn returning_match(&mut self, ret: Ty) -> Stmt {
        let ty = self.match_result_ty();
        let (scrut, mut arms) = self.value_match_parts(ty, 1);
        let last = arms.len() - 1;
        for arm in &mut arms[..last] {
            let (mark, scope_mark) = (self.pre.len(), self.scope.len());
            let value = self.expr(ret, 3);
            let mut body = vec![Stmt::Return { value }];
            self.splice_pre(mark, 0, &mut body);
            // A local that the arm hoisted leaves the scope with the arm.
            self.scope.truncate(scope_mark);
            arm.body = body;
        }
        let var = self.fresh();
        self.scope.push((var, ty, false));
        Stmt::LetMatch {
            var,
            ty,
            mutable: false,
            scrut,
            arms,
        }
    }

    /// `let v: T = match scrut { .. };`
    fn let_match(&mut self, depth: u32) -> Stmt {
        let ty = self.match_result_ty();
        let (scrut, arms) = self.value_match_parts(ty, depth);
        let var = self.fresh();
        let mutable = self.b.chance(40);
        self.scope.push((var, ty, mutable));
        Stmt::LetMatch {
            var,
            ty,
            mutable,
            scrut,
            arms,
        }
    }

    /// An enum scrutinee is a variable, a call or a variant path (E0515): any other expression (an element of a tuple)
    /// becomes the first variant.
    fn enum_scrutinee(&mut self, e: u8, scrut: Expr) -> Expr {
        if matches!(
            scrut,
            Expr::Var(..) | Expr::Call { .. } | Expr::PathCall { .. } | Expr::Variant { .. }
        ) {
            scrut
        } else {
            Expr::Variant { e, v: 0 }
        }
    }

    /// The scrutinee and the arms of a value `match` of type `ty`: on an enum (a third) or on an integer.
    fn value_match_parts(&mut self, ty: Ty, depth: u32) -> (Expr, Vec<Arm>) {
        let mut bias_root = None;
        let (scrut, pats) = if self.b.chance(35) {
            let e = self.b.below(self.table.enums.len() as u32) as u8;
            let enum_ty = Ty::Enum(e);
            let n = self.table.enums[usize::from(e)].variants.len();
            let mut scrut = None;
            match self.b.weighted(&[30, 45, 18, 7]) {
                1 => {
                    if let Some(root) = self.pick_writable_root()
                        && let Some(call) = self.root_call(enum_ty, root, 3)
                    {
                        bias_root = Some(root);
                        scrut = Some(call);
                    }
                }
                2 => scrut = self.value_call(enum_ty, 2),
                3 => {
                    let v = self.b.below(n as u32) as u8;
                    scrut = Some(Expr::Variant { e, v });
                }
                _ => {}
            }
            let scrut = match scrut {
                Some(scrut) => scrut,
                None => self.enum_leaf(enum_ty),
            };
            let scrut = self.enum_scrutinee(e, scrut);
            (scrut, self.enum_patterns(e))
        } else {
            let int = self.int_ty();
            let mut scrut = None;
            match self.b.weighted(&[38, 38, 12, 12]) {
                0 => {
                    if let Some(root) = self.pick_writable_root()
                        && let Some(call) = self.root_call(int, root, 3)
                    {
                        bias_root = Some(root);
                        scrut = Some(Expr::bin(BinOp::BitAnd, call, Expr::Lit(int, 7)));
                    }
                }
                2 => {
                    let vars = self.vars(int);
                    if !vars.is_empty() {
                        scrut = Some(Expr::Var(
                            vars[self.b.below(vars.len() as u32) as usize],
                            int,
                        ));
                    }
                }
                3 => {
                    let operand = self.expr(int, 3);
                    scrut = Some(Expr::bin(BinOp::BitAnd, operand, Expr::Lit(int, 7)));
                }
                _ => {}
            }
            let scrut = match scrut {
                Some(scrut) => scrut,
                None => {
                    let operand = self.expr(int, 3);
                    let k = [3, 4, 5, 8][self.b.below(4) as usize];
                    Expr::bin(BinOp::Rem, operand, Expr::Lit(int, k))
                }
            };
            let pats = self.int_patterns(int);
            (scrut, pats)
        };
        let mut arms = Vec::new();
        for pat in pats {
            let body = self.value_arm(ty, depth, bias_root);
            arms.push(Arm { pat, body });
        }
        (scrut, arms)
    }

    /// The patterns of an enum value `match`: the variants shuffled and cut into groups, `_` replacing the last group
    /// 45 % of the time.
    fn enum_patterns(&mut self, e: u8) -> Vec<Pat> {
        let n = self.table.enums[usize::from(e)].variants.len();
        let mut order: Vec<u8> = (0..n as u8).collect();
        for k in (1..order.len()).rev() {
            let other = self.b.below(k as u32 + 1) as usize;
            order.swap(k, other);
        }
        let mut groups: Vec<Vec<u8>> = vec![vec![order[0]]];
        for &v in &order[1..] {
            if self.b.chance(40) {
                groups.last_mut().expect("a group").push(v);
            } else {
                groups.push(vec![v]);
            }
        }
        let wild = self.b.chance(45);
        if wild {
            if groups.len() == 1 {
                let last = groups[0].pop().expect("two variants at least");
                groups.push(vec![last]);
            }
            groups.pop();
        }
        let mut pats: Vec<Pat> = groups.into_iter().map(Pat::Variants).collect();
        if wild {
            pats.push(Pat::Wild);
        }
        pats
    }

    /// The patterns of an integer value `match`: one to four distinct small literals (an `i32` also negative ones, a
    /// rare wide one), joined by `|` in groups 40 % of the time, written with or without the suffix, then `_`.
    fn int_patterns(&mut self, int: Ty) -> Vec<Pat> {
        let mut pool: Vec<u32> = (0..8).collect();
        if int == Ty::I32 {
            pool.extend([-1i32 as u32, -2i32 as u32, -3i32 as u32]);
        }
        for k in (1..pool.len()).rev() {
            let other = self.b.below(k as u32 + 1) as usize;
            pool.swap(k, other);
        }
        let count = 1 + self.b.below(4) as usize;
        pool.truncate(count);
        if self.b.chance(12) {
            let wide = if int == Ty::I32 {
                i32::MAX as u32
            } else {
                u32::MAX
            };
            if !pool.contains(&wide) {
                pool[0] = wide;
            }
        }
        let mut groups: Vec<Vec<u32>> = vec![vec![pool[0]]];
        for &v in &pool[1..] {
            if self.b.chance(40) {
                groups.last_mut().expect("a group").push(v);
            } else {
                groups.push(vec![v]);
            }
        }
        let mut pats = Vec::new();
        for group in groups {
            let lits = group
                .into_iter()
                .map(|bits| (bits, self.b.chance(40)))
                .collect();
            pats.push(Pat::Lits(lits));
        }
        pats.push(Pat::Wild);
        pats
    }

    /// The block of one arm of a value `match` of type `ty`: statements, then its value: an expression, a nested value
    /// `match`, or (in a helper) a `return`. An arm after a scrutinee that mutated `bias_root` often writes it first.
    fn value_arm(&mut self, ty: Ty, depth: u32, bias_root: Option<Root>) -> Vec<Stmt> {
        let scope_mark = self.scope.len();
        let pre_mark = self.pre.len();
        let kind = self.b.weighted(&[
            38,
            22,
            if depth < 2 { 12 } else { 0 },
            if self.cur_ret.is_some() { 45 } else { 0 },
        ]);
        let mut body = match kind {
            0 => Vec::new(),
            1 => self.block(depth + 1, 1, 2),
            _ => self.block(depth + 1, 0, 1),
        };
        if let Some(root) = bias_root
            && self.b.chance(70)
        {
            let targets: Vec<(Place, Ty)> = self
                .set_targets()
                .into_iter()
                .filter(|(place, _)| place.root == root)
                .collect();
            if !targets.is_empty() {
                body.insert(0, self.set_stmt(&targets));
            }
        }
        let last = match kind {
            2 => {
                let (scrut, arms) = self.value_match_parts(ty, depth + 1);
                Stmt::TailMatch { scrut, arms }
            }
            3 => {
                let ret = self.cur_ret.expect("a return type");
                Stmt::Return {
                    value: self.expr(ret, 3),
                }
            }
            _ => Stmt::Yield {
                value: self.expr(ty, 3),
            },
        };
        let at = body.len();
        body.push(last);
        self.splice_pre(pre_mark, at, &mut body);
        self.scope.truncate(scope_mark);
        body
    }
}

/// Whether a `return` sits inside an arm of a `match` in `stmts`.
fn arm_returns(stmts: &[Stmt], in_arm: bool) -> bool {
    stmts.iter().any(|stmt| match stmt {
        Stmt::Return { .. } => in_arm,
        Stmt::If { then, els, .. } => arm_returns(then, in_arm) || arm_returns(els, in_arm),
        Stmt::For { body, .. } => arm_returns(body, in_arm),
        Stmt::Match { arms, .. } | Stmt::LetMatch { arms, .. } | Stmt::TailMatch { arms, .. } => {
            arms.iter().any(|arm| arm_returns(&arm.body, true))
        }
        _ => false,
    })
}

/// Whether a method body assigns the field `field` of `self` anywhere (so that a call of it changes that place).
fn writes_self_field(stmts: &[Stmt], field: u8) -> bool {
    stmts.iter().any(|stmt| match stmt {
        Stmt::Set { place, .. } => place.root == Root::SelfV && place.path.first() == Some(&field),
        Stmt::If { then, els, .. } => {
            writes_self_field(then, field) || writes_self_field(els, field)
        }
        Stmt::For { body, .. } => writes_self_field(body, field),
        Stmt::Match { arms, .. } | Stmt::LetMatch { arms, .. } | Stmt::TailMatch { arms, .. } => {
            arms.iter().any(|arm| writes_self_field(&arm.body, field))
        }
        _ => false,
    })
}
#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::ast::{BinOp, operator_counts};
    use crate::emit::emit_crate_source;
    use crate::interp::Violation;

    /// Everything about a program except the seed and attempt labels.
    fn content(program: &Program) -> (u32, &[InputBuf], &[Stmt], &Expr) {
        (
            program.count,
            &program.inputs,
            &program.body,
            &program.result,
        )
    }

    #[test]
    fn determinism_the_same_seed_gives_byte_identical_programs() {
        let first = generate_batch(3).unwrap();
        let second = generate_batch(3).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            emit_crate_source(&first.programs).as_bytes(),
            emit_crate_source(&second.programs).as_bytes()
        );
        assert_eq!(generate_attempt(5, 17), generate_attempt(5, 17));
        // The attempt number selects the program, independently of the attempts before it.
        let batch = generate_batch(4).unwrap();
        for program in &batch.programs {
            assert_eq!(*program, generate_attempt(4, program.attempt));
        }
    }

    #[test]
    fn determinism_different_seeds_and_attempts_give_different_programs() {
        let one = generate_batch(1).unwrap();
        let two = generate_batch(2).unwrap();
        assert_ne!(content(&one.programs[0]), content(&two.programs[0]));
        assert_ne!(
            one.programs.iter().map(content).collect::<Vec<_>>(),
            two.programs.iter().map(content).collect::<Vec<_>>()
        );
        assert_ne!(
            content(&generate_attempt(1, 0)),
            content(&generate_attempt(1, 1))
        );
        // All 16 programs of a batch differ from each other.
        let distinct: BTreeSet<String> = one
            .programs
            .iter()
            .map(|p| format!("{:?}", content(p)))
            .collect();
        assert_eq!(distinct.len(), PROGRAMS_PER_SEED);
        // A pinned decode of the integer space (it reads the bytes DIFFTEST_FORMAT 1 read, so these are the m5.c1a
        // values): they change only with a deliberate format change (and then seeds.txt changes too).
        let one = generate_batch_in(Space::Integer, 1).unwrap();
        let two = generate_batch_in(Space::Integer, 2).unwrap();
        let first = &one.programs[0];
        assert_eq!((first.seed, first.attempt, first.count), (1, 0, 1));
        let first = &two.programs[0];
        assert_eq!((first.seed, first.attempt, first.count), (2, 0, 64));
        assert_eq!((one.programs[1].attempt, one.programs[1].count), (3, 2898));
    }

    #[test]
    fn the_integer_space_decodes_to_the_programs_of_m5_c1a() {
        // The discard counts of the m5.c1a corpus (its seeds.txt at 3bcebaf; the hashes are read with the format stamp
        // as 1): the integer space must reproduce them, so it is the same program space. Format 4 relabelled some
        // discards (seeds 1, 3, 4, 5, 7 and 8) with a divisor rule for slangc, and format 5 (m6.dv) took the rule
        // back, so the counts are the m5.c1a counts again, label for label; the accepted programs, and so the hashes
        // below, are those of m5.c1a in every format.
        let expected = [
            "1 16 57 division-by-zero=3 index=6 no-input-read=3 overflow=34 shift-amount=3 static=8",
            "2 16 36 division-by-zero=6 index=2 min-over-minus-one=1 overflow=21 static=6",
            "3 16 38 division-by-zero=8 no-input-read=2 overflow=20 static=8",
            "4 16 53 division-by-zero=7 index=5 no-input-read=1 overflow=25 shift-amount=1 static=14",
            "5 16 42 division-by-zero=5 no-input-read=3 overflow=21 shift-amount=3 static=10",
            "6 16 36 division-by-zero=1 index=3 no-input-read=1 overflow=25 shift-amount=1 static=5",
            "7 16 61 division-by-zero=8 index=3 no-input-read=2 overflow=37 shift-amount=1 static=10",
            "8 16 63 division-by-zero=4 index=1 no-input-read=2 overflow=45 static=11",
        ];
        // The m5.c1a hashes of the same programs: the emitted text and the input data, byte for byte, with the format
        // stamp read as 1 (the stamp is the only difference of format 2 for an integer program).
        let hashes = [
            0x371f_a252_2e66_81c7_u64,
            0x811a_4d82_af66_a078,
            0x7816_c1f1_77b7_08c2,
            0xa2e3_1f20_063e_9b07,
            0xd44f_7f81_932a_d6cd,
            0x3531_64f5_a863_23f9,
            0x6155_afa9_9718_9c32,
            0x9a61_7649_d659_8029,
        ];
        let mut lines = Vec::new();
        for (seed, hash) in (1..=8).zip(hashes) {
            let batch = generate_batch_in(Space::Integer, seed).unwrap();
            assert_eq!(
                crate::corpus::batch_hash_as_format(&batch.programs, 1),
                hash,
                "seed {seed}"
            );
            let mut line = format!(
                "{seed} {} {}",
                batch.programs.len(),
                batch.discarded_total()
            );
            for (reason, count) in &batch.discarded {
                line.push_str(&format!(" {reason}={count}"));
            }
            lines.push(line);
            for program in &batch.programs {
                assert!(!program.has_float());
            }
        }
        assert_eq!(lines, expected);
    }

    #[test]
    fn a_batch_holds_sixteen_in_domain_programs_and_counts_its_discards() {
        for seed in 1..=8 {
            let batch = generate_batch(seed).unwrap();
            assert_eq!(batch.programs.len(), PROGRAMS_PER_SEED);
            assert_eq!(
                u64::from(batch.discarded_total()) + PROGRAMS_PER_SEED as u64,
                batch.attempts
            );
            for program in &batch.programs {
                assert!(program.reads_input());
                let out = crate::interp::run(program).unwrap();
                assert_eq!(out.len(), program.count as usize);
            }
            let attempts: Vec<u64> = batch.programs.iter().map(|p| p.attempt).collect();
            assert!(attempts.windows(2).all(|w| w[0] < w[1]));
        }
    }

    #[test]
    fn every_attempt_decodes_to_a_well_typed_program() {
        for seed in [1, 2, 3] {
            for attempt in 0..300 {
                let program = generate_attempt(seed, attempt);
                program
                    .check()
                    .unwrap_or_else(|e| panic!("seed {seed} attempt {attempt}: {e}"));
            }
        }
    }

    #[test]
    fn a_truncated_or_empty_stream_still_decodes_to_a_valid_program() {
        let stream = byte_stream(7, 11, BYTES_PER_PROGRAM);
        for len in [0, 1, 2, 7, 8, 9, 31, 100, 500, 2000, BYTES_PER_PROGRAM - 1] {
            let program = decode(&stream[..len], 7, 11);
            program.check().unwrap_or_else(|e| panic!("len {len}: {e}"));
        }
        let empty = decode(&[], 7, 11);
        assert_eq!(empty.count, 1);
        // The all-zero stream selects the first option everywhere: one element, `u32` input, a literal result.
        assert_eq!(empty.inputs.len(), 1);
    }

    #[test]
    fn discards_cover_the_domain_conditions_the_generator_can_reach() {
        let mut reasons = BTreeSet::new();
        let mut float_reasons = BTreeSet::new();
        for attempt in 0..3000 {
            if let Some(reason) = discard_reason(&generate_attempt_in(Space::Integer, 1, attempt)) {
                reasons.insert(reason);
            }
            if let Some(reason) = discard_reason(&generate_attempt(1, attempt)) {
                float_reasons.insert(reason);
            }
        }
        // The four float labels of R1-4, so the narrowing is visible in every run's counts.
        for violation in [
            Violation::FloatDomain,
            Violation::ComparisonOverlap,
            Violation::Unboundable,
            Violation::FloatConstruct,
        ] {
            assert!(
                float_reasons.contains(violation.label()),
                "{float_reasons:?}"
            );
        }
        assert!(
            !reasons.iter().any(|r| r.starts_with("float-")
                || *r == "unboundable"
                || *r == "comparison-overlap"),
            "{reasons:?}"
        );
        for violation in [
            Violation::Overflow,
            Violation::DivisionByZero,
            Violation::ShiftAmount,
            Violation::Index,
        ] {
            assert!(reasons.contains(violation.label()), "{reasons:?}");
        }
        assert!(reasons.contains("no-input-read"), "{reasons:?}");
        assert!(reasons.contains("static"), "{reasons:?}");
    }

    #[test]
    fn element_counts_include_the_workgroup_boundaries() {
        let counts: BTreeSet<u32> = (0..200).map(|a| generate_attempt(1, a).count).collect();
        for expected in [1, 63, 64, 65, 257] {
            assert!(counts.contains(&expected), "{counts:?}");
        }
        assert!(counts.iter().any(|c| ![1, 63, 64, 65, 257].contains(c)));
        assert!(counts.iter().all(|c| (1..4096).contains(c)));
    }

    #[test]
    fn the_integer_corpus_covers_the_full_operator_set_for_both_types() {
        let mut total: BTreeMap<String, usize> = BTreeMap::new();
        for seed in 1..=8 {
            for program in generate_batch_in(Space::Integer, seed).unwrap().programs {
                for (key, count) in operator_counts(&program) {
                    *total.entry(key).or_default() += count;
                }
            }
        }
        let mut expected = BTreeSet::new();
        for ty in ["u32", "i32"] {
            for op in BinOp::ALL {
                if !op.is_logical() {
                    expected.insert(format!("{} {ty}", op.symbol()));
                }
            }
            expected.insert(format!("!{ty}"));
            expected.insert(format!("as {ty}"));
        }
        for op in ["&&", "||", "==", "!="] {
            expected.insert(format!("{op} bool"));
        }
        expected.insert("!bool".into());
        expected.insert("-".into());
        expected.insert("for".into());
        let missing: Vec<&String> = expected
            .iter()
            .filter(|key| !total.contains_key(*key))
            .collect();
        assert!(missing.is_empty(), "not covered: {missing:?}");
    }

    /// The literal zero divisors of `program`, dead code included.
    fn literal_zero_divisors(program: &Program) -> usize {
        let mut found = 0;
        crate::ast::visit_exprs(program, &mut |e| {
            if let Expr::Bin {
                op: BinOp::Div | BinOp::Rem,
                r,
                ..
            } = e
            {
                found += usize::from(matches!(**r, Expr::Lit(_, 0)));
            }
        });
        found
    }

    /// The literal shift amounts of 32 or more in `program`, dead code included.
    fn literal_wide_shifts(program: &Program) -> usize {
        let mut found = 0;
        crate::ast::visit_exprs(program, &mut |e| {
            if let Expr::Bin {
                op: BinOp::Shl | BinOp::Shr,
                r,
                ..
            } = e
            {
                found += usize::from(matches!(**r, Expr::Lit(_, 32..)));
            }
        });
        found
    }

    #[test]
    fn no_decoded_program_has_a_literal_zero_divisor() {
        // 40 seeds x 500 attempts = 20000 raw programs, accepted or not (the validator rejects the literal everywhere).
        for seed in 1..=40 {
            for attempt in 0..500 {
                let program = generate_attempt(seed, attempt);
                assert_eq!(
                    literal_zero_divisors(&program),
                    0,
                    "seed {seed} attempt {attempt}"
                );
            }
        }
    }

    #[test]
    fn no_accepted_program_holds_a_static_constant_outside_the_domain() {
        // 100 seeds x 16 = 1600 accepted programs.
        for seed in 1..=100 {
            for program in generate_batch(seed).unwrap().programs {
                let label = format!("seed {seed} attempt {}", program.attempt);
                assert_eq!(interp::static_violation(&program), None, "{label}");
                assert_eq!(literal_wide_shifts(&program), 0, "{label}");
                assert_eq!(literal_zero_divisors(&program), 0, "{label}");
            }
        }
    }

    /// Seed 4, attempt 26 (a `v2 % v2` on the counter of an unrolled loop, in a branch no element takes) was accepted
    /// until DIFFTEST_FORMAT 3 and made slangc refuse the kernel (E30002, slice m5.c2); formats 4 refused it. Since
    /// format 5 the macro lowers that `%` to `__gust_rem`, which slangc compiles, so the program is accepted again
    /// (the program of that attempt is the same program: the rule changed acceptance, not decoding).
    #[test]
    fn the_seed_4_dead_divisor_of_format_3_is_accepted_again() {
        let program = generate_attempt(4, 26);
        let text = crate::emit::emit_module(&program);
        assert!(text.contains("(v2 % v2)"), "{text}");
        assert_eq!(discard_reason(&program), None);
        assert!(
            generate_batch_in(Space::Float, 4)
                .unwrap()
                .programs
                .iter()
                .any(|accepted| accepted.attempt == 26)
        );
    }

    #[test]
    fn the_static_discard_is_live_and_only_counts_what_the_interpreter_cannot_see() {
        let mut static_programs = 0;
        let mut wide_shifts = 0;
        for attempt in 0..3000 {
            let program = generate_attempt(1, attempt);
            wide_shifts += literal_wide_shifts(&program);
            if discard_reason(&program) == Some("static") {
                static_programs += 1;
                assert!(program.reads_input());
            }
        }
        assert!(static_programs > 50, "{static_programs}");
        assert!(
            wide_shifts > 0,
            "the generator no longer writes out-of-range literal shifts"
        );
    }

    #[test]
    fn the_float_corpus_covers_the_float_subset() {
        let mut total: BTreeMap<String, usize> = BTreeMap::new();
        let mut loop_accumulators = 0;
        let mut float_outputs = 0;
        let mut float_to_int = 0;
        for seed in 1..=16 {
            for program in generate_batch(seed).unwrap().programs {
                for (key, count) in operator_counts(&program) {
                    *total.entry(key).or_default() += count;
                }
                float_outputs += usize::from(program.out_ty() == Ty::F32);
                crate::ast::visit_exprs(&program, &mut |e| {
                    if let Expr::Cast { to, e } = e {
                        float_to_int += usize::from(to.is_int() && e.ty() == Ty::F32);
                    }
                });
                fn has_accumulator(stmts: &[Stmt]) -> bool {
                    stmts.iter().any(|stmt| match stmt {
                        Stmt::For { body, .. } => body.iter().any(
                            |s| matches!(s, Stmt::Assign { value, .. } if value.ty() == Ty::F32),
                        ) || has_accumulator(body),
                        Stmt::If { then, els, .. } => has_accumulator(then) || has_accumulator(els),
                        _ => false,
                    })
                }
                loop_accumulators += usize::from(has_accumulator(&program.body));
            }
        }
        let mut expected: Vec<String> = ["+", "-", "*", "/", "==", "!=", "<", "<=", ">", ">="]
            .iter()
            .map(|op| format!("{op} f32"))
            .collect();
        expected.extend(["-f32".to_owned(), "as f32".to_owned()]);
        let missing: Vec<&String> = expected
            .iter()
            .filter(|key| !total.contains_key(*key))
            .collect();
        assert!(missing.is_empty(), "not covered: {missing:?}\n{total:?}");
        assert!(
            loop_accumulators > 0,
            "no float loop accumulator in the corpus"
        );
        assert!(float_outputs > 0, "no f32 output in the corpus");
        assert!(float_to_int > 0, "no f32 -> integer cast in the corpus");
    }

    #[test]
    fn float_programs_stay_inside_the_subset() {
        let mut floats = 0;
        for seed in 1..=3 {
            for attempt in 0..400 {
                let program = generate_attempt(seed, attempt);
                floats += usize::from(program.has_float());
                for input in program.inputs.iter().filter(|input| input.ty == Ty::F32) {
                    for bits in &input.data {
                        let value = f32::from_bits(*bits);
                        assert!(
                            *bits == 0
                                || (value.abs() >= 2f32.powi(-20) && value.abs() < 2f32.powi(20)),
                            "input {value:e} is outside the bounded set"
                        );
                    }
                }
                crate::ast::visit_exprs(&program, &mut |e| {
                    // At most four written terms per chain; no float `%`; no bool buffer (`check` refuses both).
                    assert!(
                        crate::ast::chain_terms(e) <= 4,
                        "seed {seed} attempt {attempt}"
                    );
                });
                program.check().unwrap();
            }
        }
        assert!(floats > 400, "{floats} float programs in 1200 attempts");
    }

    #[test]
    fn no_float_divisor_is_a_literal_zero_in_the_float_space_either() {
        // Covered by `no_decoded_program_has_a_literal_zero_divisor`, which scans the float space; this checks the
        // scan can see a float divisor at all.
        let mut float_divisions = 0;
        for attempt in 0..2000 {
            crate::ast::visit_exprs(&generate_attempt(1, attempt), &mut |e| {
                if let Expr::Bin {
                    op: BinOp::Div,
                    ty: Ty::F32,
                    ..
                } = e
                {
                    float_divisions += 1;
                }
            });
        }
        assert!(float_divisions > 20, "{float_divisions}");
    }
}
