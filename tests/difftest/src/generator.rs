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

use crate::ast::{BinOp, Expr, InputBuf, MAX_TRIPS, Program, Stmt, Ty, chain_terms};
use crate::interp;
use crate::prng::{Prng, byte_stream};

/// Bump when a change alters what any seed decodes to, or the emitted text. 2: the program space gained `f32`
/// (m5.c1a2, R1-4). 3: the condition and literal biases of the float space (m5.c1b): an `if` condition is never a bare
/// `true` or `false`, it often compares an input element with a value of the data or tests the element index, and a
/// literal is often a value that the data holds, so that both sides of a condition are taken by some elements. 4: no
/// decoding change, a new refusal (m5.c2): an integer `/` or `%` whose divisor slangc folds to zero (the first
/// iteration of the loops, a zero factor, identical operands, the branch that a known condition takes), live or dead,
/// is discarded as `static`, because slangc refuses the kernel (E30002). 5: that refusal is gone (slice m6.dv, D74): the
/// macro lowers a `/` or `%` with a divisor that is not a literal to `__gust_div` and `__gust_rem`, which slangc
/// compiles in every shape, so a program is no longer discarded for a divisor slangc would fold; the decoding is
/// unchanged, the programs that format 4 refused for that reason come back.
pub const DIFFTEST_FORMAT: u32 = 5;

/// Which programs the generator writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    /// Integers and booleans only. Reads exactly the bytes, and makes exactly the choices, that DIFFTEST_FORMAT 1 did,
    /// so the programs are those of m5.c1a (a unit test pins their discard counts).
    Integer,
    /// Integers, booleans and `f32` (the exact-interval subset of R1-4). The default.
    Float,
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
    inputs: Vec<Ty>,
    /// The data of each input buffer (same order as `inputs`); the float space draws literals and conditions from it.
    data: Vec<Vec<u32>>,
    count: u32,
    next_var: u32,
    /// (variable, type, mutable), innermost last.
    scope: Vec<(u32, Ty, bool)>,
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

    fn pick_ty(&mut self) -> Ty {
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
        let vars = self.vars(ty);
        let inputs = self.inputs_of(ty);
        // 0 literal, 1 variable, 2 input read, 3 gid, 4 len.
        let weights = [
            18,
            if vars.is_empty() { 0 } else { 55 },
            if inputs.is_empty() { 0 } else { 40 },
            if ty == Ty::U32 { 5 } else { 0 },
            if ty == Ty::U32 { 3 } else { 0 },
        ];
        match self.b.weighted(&weights) {
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
        let leaf_chance = if depth == 0 {
            100
        } else {
            15 + (MAX_DEPTH - depth.min(MAX_DEPTH)) * 12
        };
        if self.b.chance(leaf_chance) {
            return self.leaf(ty);
        }
        let d = depth - 1;
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
                if r == Expr::Lit(Ty::I32, u32::MAX) && self.b.chance(60) {
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
            _ => Expr::cast(ty, self.expr(Ty::F32, d)),
        }
    }

    /// An `f32` expression (R1-4): `+ - * /`, unary `-`, and casts from integers. A `+`/`-` chain has at most four
    /// written terms; a divisor is mostly kept away from zero by construction.
    fn float_expr(&mut self, d: u32) -> Expr {
        match self.b.weighted(&[16, 10, 14, 8, 6, 8]) {
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
                l = self.leaf(Ty::F32);
            } else {
                r = self.leaf(Ty::F32);
            }
        }
        (l, r)
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
        if !self.floats {
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
        let count = min + self.b.below(max - min + 1);
        let mut stmts = Vec::new();
        for _ in 0..count {
            let can_assign = self.scope.iter().any(|(_, _, mutable)| *mutable);
            let nest = if depth < 2 { 1 } else { 0 };
            let kind =
                self.b
                    .weighted(&[40, if can_assign { 24 } else { 0 }, 18 * nest, 14 * nest]);
            match kind {
                0 => {
                    let ty = self.pick_ty();
                    let init = self.expr(ty, MAX_DEPTH - 1);
                    let var = self.fresh();
                    let mutable = self.b.chance(55);
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
                    let then = self.block(depth + 1, 1, 3);
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
        }
        self.scope.truncate(mark);
        stmts
    }

    fn assign(&mut self) -> Stmt {
        let mutable: Vec<(u32, Ty)> = self
            .scope
            .iter()
            .filter(|(_, _, mutable)| *mutable)
            .map(|&(id, ty, _)| (id, ty))
            .collect();
        let (var, ty) = mutable[self.b.below(mutable.len() as u32) as usize];
        let value = if self.b.chance(45) {
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

/// Decodes one program of the default (`Space::Float`) space from a byte stream. Pure: the same bytes give the
/// same program.
pub fn decode(bytes: &[u8], seed: u64, attempt: u64) -> Program {
    decode_in(Space::Float, bytes, seed, attempt)
}

/// Decodes one program of `space` from a byte stream.
pub fn decode_in(space: Space, bytes: &[u8], seed: u64, attempt: u64) -> Program {
    let mut g = Gen {
        b: Bytes {
            data: bytes,
            pos: 0,
        },
        floats: space == Space::Float,
        inputs: Vec::new(),
        data: Vec::new(),
        count: 1,
        next_var: 0,
        scope: Vec::new(),
    };
    let count = match g.b.below(6) {
        k @ 0..=4 => COUNTS[k as usize],
        _ => 1 + g.b.below(4095),
    };
    g.count = count;
    let input_count = 1 + g.b.below(3);
    let mut inputs = Vec::new();
    for _ in 0..input_count {
        let ty = if g.floats {
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
    let body = g.block(0, 2, 7);
    let ty = g.pick_ty();
    let mut result = g.expr(ty, MAX_DEPTH);
    // Fold the final value of up to two top-level locals into the result, so that the computed values are live.
    let live: Vec<(u32, Ty)> = body
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::Let { var, ty, .. } => Some((*var, *ty)),
            _ => None,
        })
        .collect();
    for _ in 0..2 {
        if live.is_empty() || !g.b.chance(60) {
            continue;
        }
        let (var, var_ty) = live[g.b.below(live.len() as u32) as usize];
        result = fold_in(result, Expr::Var(var, var_ty));
    }
    Program {
        seed,
        attempt,
        count,
        inputs,
        body,
        result,
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

/// The program of `seed` at `attempt` in the default (`Space::Float`) space, whether or not it is in the domain.
pub fn generate_attempt(seed: u64, attempt: u64) -> Program {
    generate_attempt_in(Space::Float, seed, attempt)
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
    /// Attempts made, accepted or not.
    pub attempts: u64,
    /// Discarded attempts by reason (`Violation::label`, `static` or `no-input-read`).
    pub discarded: BTreeMap<&'static str, u32>,
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

/// Generates the `PROGRAMS_PER_SEED` in-domain programs of `seed`, in attempt order, in the default
/// (`Space::Float`) space.
pub fn generate_batch(seed: u64) -> Result<Batch, String> {
    generate_batch_in(Space::Float, seed)
}

/// Generates the `PROGRAMS_PER_SEED` in-domain programs of `seed` in `space`, in attempt order.
pub fn generate_batch_in(space: Space, seed: u64) -> Result<Batch, String> {
    let mut batch = Batch {
        seed,
        programs: Vec::new(),
        attempts: 0,
        discarded: BTreeMap::new(),
    };
    for attempt in 0..MAX_ATTEMPTS {
        batch.attempts = attempt + 1;
        let program = generate_attempt_in(space, seed, attempt);
        match discard_reason(&program) {
            Some(reason) => *batch.discarded.entry(reason).or_default() += 1,
            None => batch.programs.push(program),
        }
        if batch.programs.len() == PROGRAMS_PER_SEED {
            return Ok(batch);
        }
    }
    Err(format!(
        "seed {seed}: only {} in-domain programs in {MAX_ATTEMPTS} attempts",
        batch.programs.len()
    ))
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
            generate_batch(4)
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
        for seed in 1..=8 {
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
