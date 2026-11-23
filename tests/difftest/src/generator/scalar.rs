//! Scalar, float and condition expressions of the generator.

use super::*;

impl Gen<'_> {
    pub(super) fn fresh(&mut self) -> u32 {
        let id = self.next_var;
        self.next_var += 1;
        id
    }

    pub(super) fn vars(&self, ty: Ty) -> Vec<u32> {
        self.scope
            .iter()
            .filter(|(_, t, _)| *t == ty)
            .map(|(id, ..)| *id)
            .collect()
    }

    pub(super) fn inputs_of(&self, ty: Ty) -> Vec<usize> {
        self.inputs
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == ty)
            .map(|(k, _)| k)
            .collect()
    }

    /// The type of a local: a scalar, or in the vector space often a vector or a matrix.
    pub(super) fn pick_ty(&mut self) -> Ty {
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
    pub(super) fn scalar_ty(&mut self) -> Ty {
        if self.floats {
            return [Ty::U32, Ty::I32, Ty::Bool, Ty::F32][self.b.weighted(&[30, 30, 15, 25])];
        }
        [Ty::U32, Ty::I32, Ty::Bool][self.b.weighted(&[40, 40, 20])]
    }

    pub(super) fn int_ty(&mut self) -> Ty {
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

    pub(super) fn input_read(&mut self, buf: usize, ty: Ty) -> Expr {
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
        // 5 in the math space: a module constant of this type.
        let consts: Vec<u8> = (0..self.table.consts.len() as u8)
            .filter(|&k| self.table.consts[usize::from(k)].ty == ty)
            .collect();
        if self.math {
            weights.push(if consts.is_empty() { 0 } else { 16 });
        }
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
            5 if self.math => Expr::Const {
                k: consts[self.b.below(consts.len() as u32) as usize],
                ty,
            },
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

    pub(super) fn divisor(&mut self, ty: Ty, depth: u32) -> Expr {
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
                // A literal zero divisor is rejected by the `#[gpu_module]` validator (`DivisionByLiteralZero`) even in
                // a branch no input reaches, so it is never emitted.
                Expr::Lit(_, 0) => Expr::Lit(ty, 1),
                other => other,
            },
            // The all-ones divisor: `-1` for `i32` (so `MIN / -1` is reached), `u32::MAX` for `u32`.
            _ => Expr::Lit(ty, u32::MAX),
        }
    }

    pub(super) fn shift_amount(&mut self, depth: u32) -> Expr {
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

    pub(super) fn expr(&mut self, ty: Ty, depth: u32) -> Expr {
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
        // 14 in the math space (which has no vectors): `a.rem_euclid(b)`.
        if self.math {
            weights.push(10);
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
            14 if self.math => {
                let l = self.expr(ty, d);
                let r = self.divisor(ty, d);
                Expr::Math {
                    f: MathFn::RemEuclid,
                    ty,
                    args: vec![l, r],
                }
            }
            _ => self.agg_readout(ty, d),
        }
    }

    /// The module constants of a math program (format 14, D210): one to three `u32` or `i32` items, each an expression
    /// of literals and the items before it. Rust refuses a constant that overflows, divides by zero or shifts by 32 or
    /// more, so a draw that does not evaluate (`interp::const_value`) is tried again and, failing that, written as a
    /// literal.
    pub(super) fn gen_consts(&mut self) {
        for k in 0..1 + self.b.below(3) as usize {
            let ty = self.int_ty();
            let mut init = None;
            for _ in 0..3 {
                let candidate = self.const_term(ty, k, 2);
                self.table.consts.push(ConstDef {
                    ty,
                    init: candidate.clone(),
                });
                let valid = interp::const_value(&self.table, k).is_ok();
                self.table.consts.pop();
                if valid {
                    init = Some(candidate);
                    break;
                }
            }
            let init = init.unwrap_or_else(|| self.literal(ty));
            self.table.consts.push(ConstDef { ty, init });
        }
    }

    /// An expression of the initializer of constant `K<k>` of type `ty`: literals, the constants of that type below it,
    /// and the operators of D210 (a shift amount is a `u32` literal or a `u32` constant).
    fn const_term(&mut self, ty: Ty, k: usize, depth: u32) -> Expr {
        let below = |g: &Self, ty: Ty| -> Vec<u8> {
            (0..k as u8)
                .filter(|&j| g.table.consts[usize::from(j)].ty == ty)
                .collect()
        };
        if depth == 0 || self.b.chance(30) {
            let same = below(self, ty);
            if !same.is_empty() && self.b.chance(50) {
                let j = same[self.b.below(same.len() as u32) as usize];
                return Expr::Const { k: j, ty };
            }
            return self.literal(ty);
        }
        let d = depth - 1;
        match self.b.weighted(&[14, 10, 12, 8, 8, 6, 6, 6, 8, 8, 7, 7]) {
            kind @ 0..=7 => {
                let op = [
                    BinOp::Add,
                    BinOp::Sub,
                    BinOp::Mul,
                    BinOp::Div,
                    BinOp::Rem,
                    BinOp::BitAnd,
                    BinOp::BitOr,
                    BinOp::BitXor,
                ][kind];
                let l = self.const_term(ty, k, d);
                let r = self.const_term(ty, k, d);
                Expr::bin(op, l, r)
            }
            kind @ (8 | 9) => {
                let op = if kind == 8 { BinOp::Shl } else { BinOp::Shr };
                let l = self.const_term(ty, k, d);
                let amounts = below(self, Ty::U32);
                let r = if !amounts.is_empty() && self.b.chance(40) {
                    Expr::Const {
                        k: amounts[self.b.below(amounts.len() as u32) as usize],
                        ty: Ty::U32,
                    }
                } else {
                    Expr::u32(self.b.below(32))
                };
                Expr::bin(op, l, r)
            }
            10 if ty == Ty::I32 => Expr::negated(self.const_term(ty, k, d)),
            _ => Expr::inverted(self.const_term(ty, k, d)),
        }
    }

    /// An `f32` expression (R1-4): `+ - * /`, unary `-`, and casts from integers. A `+`/`-` chain has at most four
    /// written terms; a divisor is mostly kept away from zero by construction.
    fn float_expr(&mut self, d: u32) -> Expr {
        let weights: &[u32] = if self.vectors {
            &[16, 10, 14, 8, 6, 8, 14]
        } else if self.math {
            // 6 is the vector readout (never drawn here), 7 a math method.
            &[16, 10, 14, 8, 6, 8, 0, 34]
        } else {
            &[16, 10, 14, 8, 6, 8]
        };
        match self.b.weighted(weights) {
            7 => self.float_math(d),
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

    /// A scalar math method on `f32` (format 14): `a.mul_add(b, c)`, `x.abs()`, `x.floor()`, `x.ceil()`, `x.trunc()`,
    /// `a.min(b)`, `a.max(b)` or `x.clamp(lo, hi)`. A clamp mostly takes literal bounds with `lo < hi`, because the
    /// interval bounds need `lo <= hi` on every party (anything else is discarded as `comparison-overlap`).
    fn float_math(&mut self, d: u32) -> Expr {
        const LOWS: [f32; 6] = [-8.0, -2.0, -1.0, -0.5, 0.0, 0.25];
        const HIGHS: [f32; 6] = [0.5, 1.0, 2.0, 4.0, 16.0, 1000.0];
        let f = [
            MathFn::MulAdd,
            MathFn::Abs,
            MathFn::Floor,
            MathFn::Ceil,
            MathFn::Trunc,
            MathFn::Min,
            MathFn::Max,
            MathFn::Clamp,
        ][self.b.weighted(&[22, 12, 10, 10, 8, 12, 12, 14])];
        let mut args = vec![self.expr(Ty::F32, d)];
        if f == MathFn::Clamp && self.b.chance(70) {
            args.push(Expr::f32(LOWS[self.b.below(LOWS.len() as u32) as usize]));
            args.push(Expr::f32(HIGHS[self.b.below(HIGHS.len() as u32) as usize]));
        } else {
            for _ in 1..f.arity() {
                args.push(self.expr(Ty::F32, d));
            }
        }
        Expr::Math {
            f,
            ty: Ty::F32,
            args,
        }
    }

    /// Replaces operands of a `+`/`-` chain by leaves until it has at most `max` written terms.
    pub(super) fn fit_chain(&mut self, mut l: Expr, mut r: Expr, max: usize) -> (Expr, Expr) {
        while chain_terms(&l) + chain_terms(&r) > max {
            if chain_terms(&l) >= chain_terms(&r) {
                l = self.leaf(l.ty());
            } else {
                r = self.leaf(r.ty());
            }
        }
        (l, r)
    }
}
