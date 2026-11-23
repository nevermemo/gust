//! The decoding state: the byte reader, the generator state, and the input data.

use super::*;

pub(super) const COUNTS: [u32; 5] = [1, 63, 64, 65, 257];

pub(super) struct Bytes<'a> {
    pub(super) data: &'a [u8],
    pub(super) pos: usize,
}

impl Bytes<'_> {
    fn byte(&mut self) -> u8 {
        let byte = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        byte
    }

    pub(super) fn u64(&mut self) -> u64 {
        (0..8).fold(0, |acc, k| acc | u64::from(self.byte()) << (8 * k))
    }

    /// A value in `0..bound`; a zero byte (or a truncated stream) gives 0.
    pub(super) fn below(&mut self, bound: u32) -> u32 {
        match bound {
            0 | 1 => 0,
            2..=256 => u32::from(self.byte()) % bound,
            _ => (u32::from(self.byte()) | u32::from(self.byte()) << 8) % bound,
        }
    }

    pub(super) fn chance(&mut self, percent: u32) -> bool {
        self.below(100) < percent
    }

    /// An index chosen with the given weights; zero bytes choose index 0.
    pub(super) fn weighted(&mut self, weights: &[u32]) -> usize {
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

pub(super) struct Gen<'a> {
    pub(super) b: Bytes<'a>,
    pub(super) floats: bool,
    /// The vector space: vectors and matrices as locals and intermediates (implies `floats`).
    pub(super) vectors: bool,
    pub(super) inputs: Vec<Ty>,
    /// The data of each input buffer (same order as `inputs`); the float space draws literals and conditions from it.
    pub(super) data: Vec<Vec<u32>>,
    pub(super) count: u32,
    pub(super) next_var: u32,
    /// (variable, type, mutable), innermost last.
    pub(super) scope: Vec<(u32, Ty, bool)>,
    /// The method space: structs and functions (the tables are complete before the kernel is drawn; while function `k`
    /// is drawn they hold the functions below `k`).
    pub(super) methods: bool,
    pub(super) table: Methods,
    /// The function being drawn (`None` in the kernel): it may call the functions below it only.
    pub(super) cur_fn: Option<usize>,
    /// The receiver of the function being drawn: the struct type and how `self` is held.
    pub(super) this: Option<(Ty, Recv)>,
    /// Roots of the receivers of the calls whose arguments are being drawn: no `&mut self` call on them there.
    pub(super) forbid: Vec<Root>,
    /// The enum space (implies `methods`): enums, `match` and `return`. Every draw they add sits behind this flag.
    pub(super) enums: bool,
    /// While a helper function is drawn: its result type, so that an arm or a `then` block may end in `return`.
    pub(super) cur_ret: Option<Ty>,
    /// The function being drawn must return an enum (the tables must hold at least one such function).
    pub(super) force_enum_ret: bool,
    /// The value space (implies `enums`): tuples, the value `match` and integer patterns. Every draw they add sits
    /// behind this flag.
    pub(super) values: bool,
    /// Lets that drawing an expression hoisted (a struct literal cannot be a tuple element, E0810): the statement that
    /// is being drawn gets them in front of it.
    pub(super) pre: Vec<Stmt>,
    /// The helpers that hold a `return` in an arm of a `match` (calls prefer them in the value space).
    pub(super) arm_return_fns: Vec<usize>,
    /// The array space (implies `enums`, not `values`): arrays, `store` and the array helpers.
    pub(super) arrays: bool,
    /// The final store of the program being drawn (the array space).
    pub(super) store: Option<crate::ast::Store>,
    /// The closure space (implies `arrays`): non-capturing closures and their calls. Every draw they add sits behind
    /// this flag.
    pub(super) closures: bool,
    /// A closure body is being drawn: no `i`, no buffer, no `self`, no `return`, no closure.
    pub(super) in_closure: bool,
    /// The probe writer closures (closure, struct local, field, field type) of the kernel (see `closure_probe_lets`).
    pub(super) probe_writers: Vec<(u32, u32, u8, Ty)>,
    /// The closures bound in the open blocks.
    pub(super) cls: Vec<Cl>,
    /// The trait space (implies `enums`, not `values`, `arrays` or `closures`): module traits, their impls and generic
    /// helpers. Every draw they add sits behind this flag.
    pub(super) traits: bool,
    /// The type parameters of the generic function being drawn (empty elsewhere).
    pub(super) cur_gens: Vec<GenParam>,
    /// The (generic function, type) instantiations drawn so far: a call prefers a type not yet used.
    pub(super) inst_used: Vec<(usize, Ty)>,
    /// The math space (format 14, implies `floats`, not `vectors` or `methods`): the scalar math methods and module
    /// constants. Every draw they add sits behind this flag.
    pub(super) math: bool,
}

/// The signature a struct impl of a trait method is drawn with (format 13).
#[derive(Clone)]
pub(super) struct Forced {
    pub(super) owner: Option<u8>,
    pub(super) recv: Recv,
    pub(super) params: Vec<Ty>,
    pub(super) ret: Option<Ty>,
    pub(super) imp: Imp,
    /// The index below which the body may call: the first impl of the pair.
    pub(super) limit: usize,
}

/// A data value as a literal; i32::MIN is written as the expression -2147483647 - 1 (the emitter never writes the
/// literal -2147483648i32).
pub(super) fn data_value(ty: Ty, bits: u32) -> Expr {
    if ty == Ty::I32 && bits == 0x8000_0000 {
        Expr::i32_min()
    } else {
        Expr::Lit(ty, bits)
    }
}

pub(super) const SMALL_DIVISORS: [u32; 8] = [1, 2, 3, 4, 7, 10, 16, 255];
pub(super) const MASKS: [u32; 6] = [0x1, 0xF, 0xFF, 0xFFF, 0xFFFF, 0x7FFF_FFFF];

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
pub(super) fn input_data(ty: Ty, class: usize, count: u32, sub_seed: u64) -> Vec<u32> {
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
