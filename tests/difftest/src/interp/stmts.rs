//! Statement blocks.

use super::*;

impl Machine<'_> {
    pub(super) fn block(&mut self, stmts: &[Stmt]) -> Result<(), Violation> {
        for stmt in stmts {
            match stmt {
                Stmt::Let { var, init, .. } => {
                    if self.coverage.is_some() {
                        for name in crate::ast::array_value_names(init) {
                            self.count_name(name);
                        }
                    }
                    let value = self.expr(init)?;
                    self.vars[*var as usize] = value;
                }
                Stmt::Assign { var, value } => {
                    if self.coverage.is_some() {
                        for name in crate::ast::array_value_names(value) {
                            self.count_name(name);
                        }
                    }
                    let value = self.expr(value)?;
                    self.vars[*var as usize] = value;
                }
                Stmt::LetTuple {
                    vars,
                    ty,
                    init,
                    annotate,
                } => {
                    if self.coverage.is_some() {
                        for name in destruct_names(&self.program.methods, *annotate, init) {
                            self.count_name(name);
                        }
                    }
                    let Ty::Tuple(t) = ty else {
                        unreachable!("Program::check requires a tuple type")
                    };
                    let agg = self.expr(init)?.agg();
                    for (k, &(var, _)) in vars.iter().enumerate() {
                        self.vars[var as usize] = self.tuple_element(*t, k as u8, &agg);
                    }
                }
                Stmt::LetMatch {
                    var, scrut, arms, ..
                } => {
                    if let Some(value) = self.value_match(scrut, arms, false)? {
                        self.vars[*var as usize] = value;
                    }
                }
                Stmt::Yield { .. } | Stmt::TailMatch { .. } => {
                    unreachable!("a `Yield` or `TailMatch` only ends a value arm (Program::check)")
                }
                Stmt::AssignComp { var, index, value } => {
                    if let Some(counting) = &mut self.coverage {
                        *counting.named.entry(ASSIGN_COMP.to_string()).or_default() += 1;
                    }
                    let value = self.expr(value)?.scalar();
                    let Val::A(mut agg) = self.vars[*var as usize] else {
                        unreachable!("a component assignment to a scalar")
                    };
                    agg.c[usize::from(*index)] = value;
                    self.vars[*var as usize] = Val::A(agg);
                }
                Stmt::Set {
                    place,
                    ty,
                    op,
                    value,
                } => {
                    self.count_name(&set_name(*op));
                    if self.coverage.is_some()
                        && let Some(shape) = set_shape(&self.program.methods, place, *op, value)
                    {
                        self.count_name(&shape);
                    }
                    if self.coverage.is_some() {
                        for name in tuple_set_names(&self.program.methods, place, *op, value) {
                            self.count_name(name);
                        }
                    }
                    // Rust evaluates the assigned value first. For `place op= value` on primitives the right operand
                    // also comes before the place is read (Reference, "Compound assignment expressions"), so the
                    // place holds whatever the value's calls left in it.
                    // That is true of primitive operands only: for a `Vec*`/`Mat*` `AddAssign` the place comes first. The
                    // generator writes a compound set on `u32`/`i32` only; a slice that adds other operand types must
                    // give this arm the type-dependent order.
                    debug_assert!(op.is_none() || ty.is_int(), "a compound set on a primitive");
                    if self.coverage.is_some() {
                        for name in array_set_names(&self.program.methods, place, *op, value) {
                            self.count_name(name);
                        }
                        for name in closure_set_names(&self.program.methods, place, *op, value) {
                            self.count_name(name);
                        }
                    }
                    // The place is located (its index evaluated, once) after the value, as Rust does; the mutants
                    // `index-before-value` and `compound-place-first` locate it first.
                    let place_first =
                        self.mutant == Some(Mutant::CompoundPlaceFirst) && op.is_some();
                    let mut loc = None;
                    if place_first || self.mutant == Some(Mutant::IndexBeforeValue) {
                        loc = Some(self.locate(place)?);
                    }
                    let early = match (place_first, loc.as_ref()) {
                        (true, Some(loc)) => Some(self.loc_get(loc).int()),
                        _ => None,
                    };
                    let value = self.expr(value)?;
                    let loc = match loc {
                        Some(loc) => loc,
                        None => self.locate(place)?,
                    };
                    let stored = match op {
                        None => value,
                        Some(op) => {
                            let current = early.unwrap_or_else(|| self.loc_get(&loc).int());
                            Val::I(binary(*op, *ty, current, value.int())?)
                        }
                    };
                    self.loc_set(&loc, stored);
                }
                Stmt::Call { f, recv, args, .. } => {
                    if self.program.methods.fns[*f].recv == Recv::Mut {
                        self.count_name(SHAPE_STMT_CALL);
                    }
                    self.count_enum_args(args);
                    self.call(*f, recv, args)?;
                }
                Stmt::LetClosure { def } => {
                    if self.coverage.is_some() {
                        let site = self.closures.get(&def.var).and_then(|c| c.1);
                        for name in closure_let_names(&self.program.methods, def, site) {
                            self.count_name(name);
                        }
                    }
                }
                Stmt::Return { value } => {
                    self.count_name("return");
                    if self.arm_depth > 0 {
                        self.count_name(MATCH_ARM_RETURN_RUN);
                    }
                    let value = self.expr(value)?;
                    self.returned = Some(value);
                }
                Stmt::Match { scrut, arms } => {
                    if self.coverage.is_some() {
                        let (in_for, nested) = (self.for_depth > 0, self.arm_depth > 0);
                        for name in match_names(arms, in_for, nested) {
                            self.count_name(name);
                        }
                        if let Some(shape) = match_shape(&self.program.methods, scrut, arms) {
                            self.count_name(shape);
                        }
                    }
                    // The scrutinee is evaluated once, then the first arm naming the variant (`_` names the rest).
                    let bits = self.expr(scrut)?.int();
                    let Ty::Enum(e) = scrut.ty() else {
                        unreachable!("Program::check requires an enum scrutinee")
                    };
                    let variant = self.program.methods.enums[usize::from(e)]
                        .variants
                        .iter()
                        .position(|v| v.0 == bits)
                        .expect("an enum value is a discriminant")
                        as u8;
                    let mut taken = None;
                    for (k, arm) in arms.iter().enumerate() {
                        if self.mutant == Some(Mutant::ScrutineeTwice) && k > 0 {
                            self.expr(scrut)?;
                        }
                        let hit = match &arm.pat {
                            Pat::Variants(vs) => vs.contains(&variant),
                            Pat::Lits(_) => unreachable!("an integer pattern in a statement match"),
                            Pat::Wild => true,
                        };
                        if hit {
                            let no_wild = arms.iter().all(|a| a.pat != Pat::Wild);
                            let wrong = self.mutant == Some(Mutant::DefaultTakesFirstArm)
                                && no_wild
                                && k + 1 == arms.len();
                            taken = Some(if wrong { &arms[0] } else { arm });
                            break;
                        }
                    }
                    let arm = taken.expect("a match is exhaustive");
                    self.arm_depth += 1;
                    let done = self.block(&arm.body);
                    self.arm_depth -= 1;
                    done?;
                }
                Stmt::If { cond, then, els } => {
                    let taken = self.expr(cond)?.int() != 0;
                    if let Some(counting) = &mut self.coverage {
                        let entry = counting
                            .branches
                            .entry(std::ptr::from_ref(stmt) as usize)
                            .or_default();
                        if taken {
                            entry.0 += 1;
                        } else {
                            entry.1 += 1;
                        }
                    }
                    if taken {
                        self.block(then)?;
                    } else {
                        self.block(els)?;
                    }
                }
                Stmt::For { var, trips, body } => {
                    self.count(KIND_FOR, "for", "");
                    self.for_depth += 1;
                    for k in 0..*trips {
                        self.vars[*var as usize] = Val::I(k);
                        if let Err(violation) = self.block(body) {
                            self.for_depth -= 1;
                            return Err(violation);
                        }
                        if self.returned.is_some() {
                            break;
                        }
                    }
                    self.for_depth -= 1;
                }
            }
            if self.returned.is_some() {
                break;
            }
        }
        Ok(())
    }
}
