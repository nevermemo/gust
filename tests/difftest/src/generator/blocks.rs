//! Conditions and statement blocks of the generator.

use super::*;

impl Gen<'_> {
    pub(super) fn float_divisor(&mut self, d: u32) -> Expr {
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
    pub(super) fn float_literal(&mut self) -> Expr {
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

    pub(super) fn bool_expr(&mut self, d: u32) -> Expr {
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

    pub(super) fn block(&mut self, depth: u32, min: u32, max: u32) -> Vec<Stmt> {
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
