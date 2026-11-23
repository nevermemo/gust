//! The static fold: closed constants outside the domain.

use super::*;

/// A constant of the static check.
#[allow(clippy::large_enum_variant)] // as `Val`: copied by value, folded once per node
#[derive(Clone, Copy)]
enum Const {
    I(u32),
    F(f32),
    /// A closed vector or matrix; its `f32` components carry exact intervals and are folded with the interpreter's
    /// own arithmetic (`agg`), so a closed aggregate that would leave the domain when run is a static violation too.
    A(Agg),
}

impl Const {
    fn int(self) -> u32 {
        match self {
            Const::I(bits) => bits,
            Const::F(_) | Const::A(_) => unreachable!("typed program"),
        }
    }

    fn float(self) -> f32 {
        match self {
            Const::F(value) => value,
            Const::I(_) | Const::A(_) => unreachable!("typed program"),
        }
    }

    fn agg(self) -> Agg {
        match self {
            Const::A(agg) => agg,
            Const::I(_) | Const::F(_) => unreachable!("typed program"),
        }
    }

    /// This constant as a component of an aggregate (a float with the exact interval of its value).
    fn scalar(self) -> Scalar {
        match self {
            Const::I(bits) => Scalar::I(bits),
            Const::F(value) => Scalar::F(value, Fl::exact(value)),
            Const::A(_) => unreachable!("typed program"),
        }
    }

    /// The constant that a component of an aggregate is (a float as its reference value).
    fn of(scalar: Scalar) -> Const {
        match scalar {
            Scalar::I(bits) => Const::I(bits),
            Scalar::F(reference, _) => Const::F(reference),
        }
    }

    /// The components of an operand of `AggBin`, `Dot` and the like: a scalar is one component.
    fn lanes(self) -> Agg {
        match self {
            Const::A(agg) => agg,
            scalar => Agg::new(&[scalar.scalar()]),
        }
    }
}

/// A closed `f32` constant must be zero or inside `+-[2^-100, 2^100]`.
fn float_constant(value: f32) -> Result<Option<Const>, Violation> {
    let v = f64::from(value);
    if float::in_range(v, v) {
        Ok(Some(Const::F(value)))
    } else {
        Err(Violation::FloatDomain)
    }
}

/// The first domain violation in a closed sub-expression (one made of literals only) anywhere in the program,
/// dead code included, or `None`. Closed sub-expressions are folded eagerly, without short-circuiting, with the same
/// `binary` and `unary` as the interpreter. A shift by a literal 32 or more and a division by a literal zero count
/// even when the other operand is not closed. Such a constant is a compile-time error in some shader languages
/// (const-expression rules), so the generator keeps none, in executed or dead code (labelled `static`). Closed
/// `f32` constants are folded in `f32` and must be zero or inside `+-[2^-100, 2^100]`; a closed float division by
/// zero, an integer above `2^24` converted to `f32` and a closed cast out of the integer range count too. A divisor
/// that slangc folds to zero without being a literal (the first iteration of a loop, `i * 0`, `i - i`) is not counted
/// since DIFFTEST_FORMAT 5: the macro lowers such a `/` and `%` to `__gust_div` and `__gust_rem` (slice m6.dv, D74),
/// which slangc compiles.
pub fn static_violation(program: &Program) -> Option<Violation> {
    fn fold(methods: &Methods, e: &Expr) -> Result<Option<Const>, Violation> {
        match e {
            Expr::Lit(Ty::F32, bits) => float_constant(f32::from_bits(*bits)),
            Expr::Lit(_, bits) => Ok(Some(Const::I(*bits))),
            Expr::Var(..) | Expr::Gid | Expr::Len(_) | Expr::Variant { .. } => Ok(None),
            // A module constant folds to its value (an initializer that overflows is a violation), so a shift by `K0`
            // of 32 or more counts like the literal.
            Expr::Const { k, .. } => {
                const_value(methods, usize::from(*k)).map(|v| Some(Const::I(v)))
            }
            Expr::Math {
                f: MathFn::RemEuclid,
                ty,
                args,
            } => match (fold(methods, &args[0])?, fold(methods, &args[1])?) {
                (_, Some(Const::I(0))) => Err(Violation::DivisionByZero),
                (Some(l), Some(r)) => rem_euclid(*ty, l.int(), r.int()).map(|v| Some(Const::I(v))),
                _ => Ok(None),
            },
            // The float methods fold their operands (each is checked as a node of its own) and stay open.
            Expr::Math { args, .. } => args
                .iter()
                .try_for_each(|arg| fold(methods, arg).map(|_| ()))
                .map(|()| None),
            // A field, a call or a struct literal is open; its operands are folded as nodes of their own.
            Expr::Read { .. }
            | Expr::Call { .. }
            | Expr::PathCall { .. }
            | Expr::ClosureCall { .. }
            | Expr::StructLit { .. }
            | Expr::Tuple { .. }
            | Expr::TupleField { .. }
            | Expr::ArrayLit { .. }
            | Expr::Repeat { .. }
            | Expr::ArrLen { .. } => Ok(None),
            Expr::Input { index, .. } => fold(methods, index).map(|_| None),
            Expr::Un {
                op: UnOp::Neg,
                ty: Ty::F32,
                e,
            } => fold(methods, e)?
                .map(|v| float_constant(-v.float()))
                .transpose()
                .map(Option::flatten),
            Expr::Un { op, ty, e } => fold(methods, e)?
                .map(|v| unary(*op, *ty, v.int()).map(Const::I))
                .transpose(),
            Expr::Bin { op, ty, l, r } if *ty == Ty::F32 => {
                let (left, right) = (fold(methods, l)?, fold(methods, r)?);
                if *op == BinOp::Div && right.is_some_and(|v| v.float() == 0.0) {
                    return Err(Violation::DivisionByZero);
                }
                let (Some(a), Some(b)) = (left, right) else {
                    return Ok(None);
                };
                let (a, b) = (a.float(), b.float());
                let value = match op {
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Div => a / b,
                    BinOp::Eq => return Ok(Some(Const::I(u32::from(a == b)))),
                    BinOp::Ne => return Ok(Some(Const::I(u32::from(a != b)))),
                    BinOp::Lt => return Ok(Some(Const::I(u32::from(a < b)))),
                    BinOp::Le => return Ok(Some(Const::I(u32::from(a <= b)))),
                    BinOp::Gt => return Ok(Some(Const::I(u32::from(a > b)))),
                    BinOp::Ge => return Ok(Some(Const::I(u32::from(a >= b)))),
                    other => unreachable!("float operator {}", other.symbol()),
                };
                float_constant(value)
            }
            Expr::Bin { op, ty, l, r } => {
                let (left, right) = (fold(methods, l)?, fold(methods, r)?);
                match (op, right) {
                    (BinOp::Div | BinOp::Rem, Some(Const::I(0))) => {
                        return Err(Violation::DivisionByZero);
                    }
                    (BinOp::Shl | BinOp::Shr, Some(Const::I(amount))) if amount >= 32 => {
                        return Err(Violation::ShiftAmount);
                    }
                    _ => {}
                }
                match (left, right) {
                    (Some(l), Some(r)) => {
                        binary(*op, *ty, l.int(), r.int()).map(|v| Some(Const::I(v)))
                    }
                    _ => Ok(None),
                }
            }
            Expr::Make { ty, rows, args } => {
                let mut parts = Vec::with_capacity(ty.lanes());
                let mut closed = true;
                for arg in args {
                    match fold(methods, arg)? {
                        Some(Const::A(agg)) => parts.extend_from_slice(agg.lanes()),
                        Some(scalar) => parts.push(scalar.scalar()),
                        None => closed = false,
                    }
                }
                if !closed {
                    return Ok(None);
                }
                if *rows {
                    let Ty::Mat(n) = ty else {
                        unreachable!("from_rows of {}", ty.name())
                    };
                    parts = agg::from_rows(usize::from(*n), &parts);
                }
                Ok(Some(Const::A(Agg::new(&parts))))
            }
            Expr::Splat { ty, e } => Ok(fold(methods, e)?
                .map(|value| Const::A(Agg::new(&vec![value.scalar(); ty.lanes()])))),
            Expr::Comp { e, index } => {
                Ok(fold(methods, e)?.map(|value| Const::of(value.agg().c[usize::from(*index)])))
            }
            Expr::Swizzle { e, lanes } => Ok(fold(methods, e)?.map(|value| {
                let agg = value.agg();
                let picked: Vec<Scalar> = lanes.iter().map(|&k| agg.c[usize::from(k)]).collect();
                Const::A(Agg::new(&picked))
            })),
            Expr::Dot { l, r } => {
                let (left, right) = (fold(methods, l)?, fold(methods, r)?);
                let (Some(a), Some(b)) = (left, right) else {
                    return Ok(None);
                };
                let sum = agg::dot(l.ty().scalar(), a.agg().lanes(), b.agg().lanes())?;
                Ok(Some(Const::of(sum)))
            }
            Expr::Transpose { e } => {
                let Ty::Mat(n) = e.ty() else {
                    unreachable!("transpose of a non-matrix")
                };
                Ok(fold(methods, e)?.map(|value| {
                    Const::A(Agg::new(&agg::transpose(
                        usize::from(n),
                        value.agg().lanes(),
                    )))
                }))
            }
            Expr::AggNeg { e } => match fold(methods, e)? {
                Some(value) => Ok(Some(Const::A(agg::negate(
                    e.ty().scalar(),
                    value.agg().lanes(),
                )?))),
                None => Ok(None),
            },
            Expr::AggBin { op, ty, l, r } => {
                let types = (l.ty(), r.ty());
                let (left, right) = (fold(methods, l)?, fold(methods, r)?);
                if matches!(op, BinOp::Div | BinOp::Rem)
                    && right.is_some_and(|divisor| {
                        divisor.lanes().lanes().iter().any(|part| match *part {
                            Scalar::I(bits) => bits == 0,
                            Scalar::F(value, _) => value == 0.0,
                        })
                    })
                {
                    return Err(Violation::DivisionByZero);
                }
                let (Some(a), Some(b)) = (left, right) else {
                    return Ok(None);
                };
                let (a, b) = (a.lanes(), b.lanes());
                Ok(Some(Const::A(agg::binary_op(
                    *op,
                    *ty,
                    types,
                    a.lanes(),
                    b.lanes(),
                )?)))
            }
            Expr::EnumAs { e, .. } => fold(methods, e).map(|_| None),
            Expr::Cast { to, e } => {
                let from = e.ty();
                let Some(value) = fold(methods, e)? else {
                    return Ok(None);
                };
                match (from, *to) {
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
                        float_constant(number)
                    }
                    (Ty::F32, Ty::I32 | Ty::U32) => {
                        let truncated = f64::from(value.float()).trunc();
                        let (min, max) = if *to == Ty::I32 {
                            (-(f64::from(i32::MAX)), f64::from(i32::MAX) - 1.0)
                        } else {
                            (0.0, f64::from(u32::MAX) - 1.0)
                        };
                        if truncated < min || truncated > max {
                            return Err(Violation::FloatDomain);
                        }
                        Ok(Some(Const::I(truncated as i64 as u32)))
                    }
                    _ => Ok(Some(value)),
                }
            }
        }
    }
    let mut found = None;
    // Every node folds its own subtree: quadratic in depth, but programs are small.
    crate::ast::visit_exprs(program, &mut |e| {
        if found.is_none() {
            found = fold(&program.methods, e).err();
        }
    });
    // Format 10 (m9.d1c): the right-hand side of a compound assignment is the operand of an implicit shift, division or
    // remainder, so a closed constant outside the domain there (`self.f <<= 34u32` in code that is reachable from the
    // kernel but never executes) is a static violation too: naga rejects the module although the interpreter never runs
    // the statement. Every space applies it (fix round 1; round 0 gated it to the value space).
    if found.is_none() {
        fn sets(
            stmts: &[Stmt],
            methods: &Methods,
            fold: fn(&Methods, &Expr) -> Result<Option<Const>, Violation>,
        ) -> Option<Violation> {
            for stmt in stmts {
                let violation = match stmt {
                    Stmt::Set {
                        op: Some(op),
                        value,
                        ..
                    } => match (op, fold(methods, value)) {
                        (BinOp::Shl | BinOp::Shr, Ok(Some(Const::I(amount)))) if amount >= 32 => {
                            Some(Violation::ShiftAmount)
                        }
                        (BinOp::Div | BinOp::Rem, Ok(Some(Const::I(0)))) => {
                            Some(Violation::DivisionByZero)
                        }
                        _ => None,
                    },
                    Stmt::If { then, els, .. } => {
                        sets(then, methods, fold).or_else(|| sets(els, methods, fold))
                    }
                    Stmt::For { body, .. } => sets(body, methods, fold),
                    Stmt::LetClosure { def } => sets(&def.body, methods, fold),
                    Stmt::Match { arms, .. }
                    | Stmt::LetMatch { arms, .. }
                    | Stmt::TailMatch { arms, .. } => {
                        arms.iter().find_map(|arm| sets(&arm.body, methods, fold))
                    }
                    _ => None,
                };
                if violation.is_some() {
                    return violation;
                }
            }
            None
        }
        found = sets(&program.body, &program.methods, fold).or_else(|| {
            program
                .methods
                .fns
                .iter()
                .find_map(|def| sets(&def.body, &program.methods, fold))
        });
    }
    found
}
