//! Expression evaluation.

use super::*;

impl Machine<'_> {
    /// Evaluates `expr` and, for `observed_values`, remembers the first value each node takes (a scalar's pattern; a
    /// vector or matrix has no single pattern and is not remembered).
    pub(super) fn expr(&mut self, expr: &Expr) -> Result<Val, Violation> {
        let value = self.eval(expr)?;
        if let Some(seen) = &mut self.observed {
            let bits = match value {
                Val::I(bits) => Some(bits),
                Val::F(reference, _) => Some(reference.to_bits()),
                Val::A(_) => None,
            };
            if let Some(bits) = bits {
                seen.entry(std::ptr::from_ref(expr) as usize)
                    .or_insert(bits);
            }
        }
        Ok(value)
    }

    /// The components of an operand: a scalar is one, an aggregate its own.
    fn lanes(value: Val) -> Agg {
        match value {
            Val::A(agg) => agg,
            scalar => Agg::new(&[scalar.scalar()]),
        }
    }

    /// The aggregate operators (`AggBin` and the nodes that build, read or rearrange vectors and matrices).
    fn eval_aggregate(&mut self, expr: &Expr) -> Result<Val, Violation> {
        self.count_aggregate(expr);
        match expr {
            Expr::Make { ty, rows, args } => {
                let mut parts = Vec::with_capacity(ty.lanes());
                for arg in args {
                    let value = self.expr(arg)?;
                    match value {
                        Val::A(agg) => parts.extend_from_slice(agg.lanes()),
                        scalar => parts.push(scalar.scalar()),
                    }
                }
                if *rows {
                    let Ty::Mat(n) = ty else {
                        unreachable!("from_rows of {}", ty.name())
                    };
                    parts = agg::from_rows(usize::from(*n), &parts);
                }
                Ok(Val::A(Agg::new(&parts)))
            }
            Expr::Splat { ty, e } => {
                let value = self.expr(e)?.scalar();
                Ok(Val::A(Agg::new(&vec![value; ty.lanes()])))
            }
            Expr::Comp { e, index } => {
                let agg = self.expr(e)?.agg();
                Ok(agg.c[usize::from(*index)].into())
            }
            Expr::Swizzle { e, lanes } => {
                let agg = self.expr(e)?.agg();
                let picked: Vec<Scalar> = lanes.iter().map(|&k| agg.c[usize::from(k)]).collect();
                Ok(Val::A(Agg::new(&picked)))
            }
            Expr::Dot { l, r } => {
                let lane = l.ty().scalar();
                let left = self.expr(l)?.agg();
                let right = self.expr(r)?.agg();
                Ok(agg::dot(lane, left.lanes(), right.lanes())?.into())
            }
            Expr::Transpose { e } => {
                let Ty::Mat(n) = e.ty() else {
                    unreachable!("transpose of a non-matrix")
                };
                let agg = self.expr(e)?.agg();
                Ok(Val::A(Agg::new(&agg::transpose(
                    usize::from(n),
                    agg.lanes(),
                ))))
            }
            Expr::AggNeg { e } => {
                let lane = e.ty().scalar();
                let agg = self.expr(e)?.agg();
                Ok(Val::A(agg::negate(lane, agg.lanes())?))
            }
            Expr::AggBin { op, ty, l, r } => {
                let types = (l.ty(), r.ty());
                let left = Self::lanes(self.expr(l)?);
                let right = Self::lanes(self.expr(r)?);
                Ok(Val::A(agg::binary_op(
                    *op,
                    *ty,
                    types,
                    left.lanes(),
                    right.lanes(),
                )?))
            }
            other => unreachable!("not an aggregate operator: {other:?}"),
        }
    }

    fn eval(&mut self, expr: &Expr) -> Result<Val, Violation> {
        match expr {
            Expr::Lit(Ty::F32, bits) => {
                let value = f32::from_bits(*bits);
                Ok(Val::F(value, in_domain(Fl::exact(value))?))
            }
            Expr::Lit(_, bits) => Ok(Val::I(*bits)),
            Expr::Var(var, _) => Ok(self.vars[*var as usize]),
            Expr::Gid => Ok(Val::I(self.element)),
            Expr::Len(buf) => Ok(Val::I(self.program.inputs[*buf].data.len() as u32)),
            Expr::Input { buf, index, ty } => {
                let index = self.expr(index)?.int();
                let bits = self.program.inputs[*buf]
                    .data
                    .get(index as usize)
                    .copied()
                    .ok_or(Violation::Index)?;
                if *ty == Ty::F32 {
                    let value = f32::from_bits(bits);
                    Ok(Val::F(value, in_domain(Fl::exact(value))?))
                } else {
                    Ok(Val::I(bits))
                }
            }
            Expr::Un {
                op: UnOp::Neg,
                ty: Ty::F32,
                e,
            } => {
                self.count(KIND_UNARY, "-f32", "");
                let (reference, fl) = self.expr(e)?.float();
                Ok(checked_value(-reference, fl.negated()))
            }
            Expr::Un { op, ty, e } => {
                self.count(
                    KIND_UNARY,
                    match op {
                        UnOp::Neg => "-",
                        UnOp::Not => "!",
                    },
                    ty.name(),
                );
                let value = self.expr(e)?.int();
                Ok(Val::I(unary(*op, *ty, value)?))
            }
            Expr::Bin {
                op: op @ (BinOp::And | BinOp::Or),
                l,
                r,
                ..
            } => {
                self.count(KIND_BINARY, op.symbol(), "bool");
                let left = self.expr(l)?.int();
                if (*op == BinOp::And) == (left == 0) {
                    if let Some(counting) = &mut self.coverage {
                        counting.rhs_skipped += 1;
                    }
                    return Ok(Val::I(left));
                }
                if let Some(counting) = &mut self.coverage {
                    counting.rhs_evaluated += 1;
                }
                if self.coverage.is_some()
                    && let Some(shape) = bin_shape(&self.program.methods, *op, l, r)
                {
                    self.count_name(shape);
                }
                self.expr(r)
            }
            Expr::Bin {
                op,
                ty: Ty::F32,
                l,
                r,
            } => {
                self.count(KIND_BINARY, op.symbol(), "f32");
                let left = self.expr(l)?.float();
                let right = self.expr(r)?.float();
                if op.is_comparison() {
                    let truth = float::compare(op.symbol(), &left.1, &right.1)?;
                    debug_assert_eq!(
                        truth,
                        match op {
                            BinOp::Eq => left.0 == right.0,
                            BinOp::Ne => left.0 != right.0,
                            BinOp::Lt => left.0 < right.0,
                            BinOp::Le => left.0 <= right.0,
                            BinOp::Gt => left.0 > right.0,
                            _ => left.0 >= right.0,
                        },
                        "the reference comparison disagrees with the disjoint intervals"
                    );
                    return Ok(Val::I(u32::from(truth)));
                }
                float_binary(*op, left, right).map(Val::from)
            }
            Expr::Make { .. }
            | Expr::Splat { .. }
            | Expr::Comp { .. }
            | Expr::Swizzle { .. }
            | Expr::Dot { .. }
            | Expr::Transpose { .. }
            | Expr::AggBin { .. }
            | Expr::AggNeg { .. } => self.eval_aggregate(expr),
            Expr::Read { place, .. } => {
                if place.root_ty.is_tuple() {
                    self.count_name(TUPLE_READ);
                }
                if self.coverage.is_some() {
                    for name in array_expr_names(&self.program.methods, expr) {
                        self.count_name(name);
                    }
                }
                let loc = self.locate(place)?;
                Ok(self.loc_get(&loc))
            }
            Expr::ArrayLit { elems, .. } => {
                if self.coverage.is_some() {
                    for name in array_expr_names(&self.program.methods, expr) {
                        self.count_name(name);
                    }
                }
                // The elements run left to right, each completely before the next (Rust Reference, "Array
                // expressions" and "Evaluation order of operands").
                let mut values = vec![None; elems.len()];
                let order: Vec<usize> = if self.mutant == Some(Mutant::ArrayLitRightToLeft) {
                    (0..elems.len()).rev().collect()
                } else {
                    (0..elems.len()).collect()
                };
                for k in order {
                    values[k] = Some(self.expr(&elems[k])?);
                }
                let mut parts = Vec::new();
                for value in values.into_iter().flatten() {
                    match value {
                        Val::A(agg) => parts.extend_from_slice(agg.lanes()),
                        scalar => parts.push(scalar.scalar()),
                    }
                }
                Ok(Val::A(Agg::new(&parts)))
            }
            Expr::Repeat { ty, e } => {
                if self.coverage.is_some() {
                    for name in array_expr_names(&self.program.methods, expr) {
                        self.count_name(name);
                    }
                }
                let Ty::Array(a) = ty else {
                    unreachable!("Program::check requires an array type")
                };
                // The element is evaluated once and copied (Reference, "Array expressions": `[expr; N]`).
                let mut parts = Vec::new();
                let mut value = self.expr(e)?;
                for k in 0..a.n {
                    if self.mutant == Some(Mutant::RepeatPerElement) && k > 0 {
                        value = self.expr(e)?;
                    }
                    match value {
                        Val::A(agg) => parts.extend_from_slice(agg.lanes()),
                        scalar => parts.push(scalar.scalar()),
                    }
                }
                Ok(Val::A(Agg::new(&parts)))
            }
            Expr::ArrLen { place } => {
                if self.coverage.is_some() {
                    for name in array_expr_names(&self.program.methods, expr) {
                        self.count_name(name);
                    }
                }
                let Ty::Array(a) = place.root_ty else {
                    unreachable!("Program::check requires an array under `.len()`")
                };
                Ok(Val::I(u32::from(a.n)))
            }
            Expr::Tuple { elems, .. } => {
                if self.coverage.is_some() {
                    for name in tuple_lit_names(&self.program.methods, elems) {
                        self.count_name(name);
                    }
                }
                // The elements run left to right, each completely before the next (Rust Reference, "Tuple
                // expressions" and "Evaluation order of operands").
                let mut values = vec![None; elems.len()];
                let order: Vec<usize> = if self.mutant == Some(Mutant::TupleRightToLeft) {
                    (0..elems.len()).rev().collect()
                } else {
                    (0..elems.len()).collect()
                };
                for k in order {
                    values[k] = Some(self.expr(&elems[k])?);
                }
                let mut parts = Vec::new();
                for value in values.into_iter().flatten() {
                    match value {
                        Val::A(agg) => parts.extend_from_slice(agg.lanes()),
                        scalar => parts.push(scalar.scalar()),
                    }
                }
                Ok(Val::A(Agg::new(&parts)))
            }
            Expr::TupleField { e, index } => {
                if self.coverage.is_some() {
                    for name in tuple_field_names(&self.program.methods, e) {
                        self.count_name(name);
                    }
                }
                let Ty::Tuple(t) = e.ty() else {
                    unreachable!("Program::check requires a tuple under `.k`")
                };
                let agg = self.expr(e)?.agg();
                Ok(self.tuple_element(t, *index, &agg))
            }
            Expr::Variant { e, v } => Ok(Val::I(
                self.program.methods.enums[usize::from(*e)].variants[usize::from(*v)].0,
            )),
            Expr::Const { k, ty } => {
                if self.coverage.is_some() {
                    self.count_name(&format!("const {}", ty.name()));
                }
                Ok(Val::I(const_value(&self.program.methods, usize::from(*k))?))
            }
            Expr::Math { f, ty, args } => {
                if self.coverage.is_some() {
                    self.count_name(&f.counted_name(*ty));
                }
                // The receiver and the arguments run left to right, each completely.
                let mut values = Vec::with_capacity(args.len());
                for arg in args {
                    values.push(self.expr(arg)?);
                }
                if *f == MathFn::RemEuclid {
                    return Ok(Val::I(rem_euclid(*ty, values[0].int(), values[1].int())?));
                }
                let operands: Vec<(f32, Fl)> = values.into_iter().map(Val::float).collect();
                float_math(*f, &operands).map(Val::from)
            }
            Expr::EnumAs { to, e } => {
                self.count_name(&enum_as_name(*to));
                // The discriminant bits are already the repr value (an `i32` repr holds its two's complement).
                Ok(Val::I(self.expr(e)?.int()))
            }
            Expr::Call { f, recv, args, .. } => {
                if self.program.methods.fns[*f].recv == Recv::Mut {
                    self.count_name(SHAPE_EXPR_CALL);
                }
                self.count_enum_args(args);
                let value = self.call(*f, recv, args)?;
                Ok(value.expect("a value call has a tail"))
            }
            Expr::ClosureCall { var, args, .. } => self.closure_call(*var, args),
            Expr::PathCall { form, call } => {
                self.count_name(if *form == PathForm::Trait {
                    TRAIT_PATH_CALL
                } else {
                    TRAIT_PATH_QUALIFIED
                });
                self.expr(call)
            }
            Expr::StructLit { fields, order, .. } => {
                self.count_name(STRUCT_LIT);
                // The fields in the order written, as Rust evaluates them; a nested struct contributes its
                // components. The components are laid out in declaration order.
                let mut by_decl = vec![Vec::new(); fields.len()];
                for (field, &decl) in fields.iter().zip(order) {
                    by_decl[usize::from(decl)] = match self.expr(field)? {
                        Val::A(agg) => agg.lanes().to_vec(),
                        scalar => vec![scalar.scalar()],
                    };
                }
                let parts: Vec<_> = by_decl.into_iter().flatten().collect();
                Ok(Val::A(Agg::new(&parts)))
            }
            Expr::Bin { op, ty, l, r } => {
                self.count(KIND_BINARY, op.symbol(), ty.name());
                if self.coverage.is_some()
                    && let Some(shape) = bin_shape(&self.program.methods, *op, l, r)
                {
                    self.count_name(shape);
                }
                let left = self.expr(l)?.int();
                let right = self.expr(r)?.int();
                Ok(Val::I(binary(*op, *ty, left, right)?))
            }
            Expr::Cast { to, e } => {
                self.count(KIND_CAST, "as", to.name());
                let from = e.ty();
                let value = self.expr(e)?;
                match (from, *to) {
                    (Ty::F32, Ty::I32 | Ty::U32) => {
                        let (reference, fl) = value.float();
                        Ok(Val::I(float_to_int(*to, reference, &fl)?))
                    }
                    (Ty::I32 | Ty::U32, Ty::F32) => {
                        let bits = value.int();
                        let magnitude = if from == Ty::I32 {
                            u64::from((bits as i32).unsigned_abs())
                        } else {
                            u64::from(bits)
                        };
                        if magnitude > float::EXACT_INT_LIMIT {
                            return Err(Violation::FloatConstruct);
                        }
                        let number = if from == Ty::I32 {
                            (bits as i32) as f32
                        } else {
                            bits as f32
                        };
                        Ok(Val::F(number, Fl::exact(number)))
                    }
                    _ => Ok(value),
                }
            }
        }
    }
}
