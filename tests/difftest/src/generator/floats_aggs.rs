//! Vector and matrix expressions of the generator.

use super::*;

impl Gen<'_> {
    /// The type of a vector or matrix local: mostly `f32` vectors and matrices, which carry the interval bounds.
    pub(super) fn agg_ty(&mut self) -> Ty {
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
    pub(super) fn agg_leaf(&mut self, ty: Ty) -> Expr {
        let vars = self.vars(ty);
        if !vars.is_empty() && self.b.chance(65) {
            return Expr::Var(vars[self.b.below(vars.len() as u32) as usize], ty);
        }
        self.build(ty, 1)
    }

    /// `Vec4::new(..)`, `Vec4::splat(..)`, `Mat4::from_cols(..)` or `Mat4::from_rows(..)` of operands of depth `d`. A
    /// `Mat2` always has an input read among its columns (SEMANTICS section 4: a constant `Mat2` crashes naga's SPIR-V
    /// front end, so the generator never writes one).
    pub(super) fn build(&mut self, ty: Ty, d: u32) -> Expr {
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
    pub(super) fn agg_readout(&mut self, ty: Ty, d: u32) -> Expr {
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

    pub(super) fn agg_expr(&mut self, ty: Ty, depth: u32) -> Expr {
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
    pub(super) fn assign_agg(&mut self, var: u32, ty: Ty) -> Stmt {
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
}
