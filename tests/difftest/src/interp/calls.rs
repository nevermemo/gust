//! Places, calls, closures and value matches.

use super::*;

impl Machine<'_> {
    /// Element `k` of a tuple value `agg` of type `t`: a scalar, or a struct as its flat components.
    pub(super) fn tuple_element(&self, t: crate::ast::TupleTy, k: u8, agg: &Agg) -> Val {
        let methods = &self.program.methods;
        let offset = methods.tuple_offset(t, k);
        match t.elems()[usize::from(k)] {
            Elem::Struct(j) => Val::A(Agg::new(&agg.c[offset..offset + methods.flat_len(j)])),
            _ => agg.c[offset].into(),
        }
    }

    /// Counts the enum-argument shapes of a call.
    pub(super) fn count_enum_args(&mut self, args: &[Expr]) {
        if self.coverage.is_some() {
            for shape in enum_call_shapes(&self.program.methods, args) {
                self.count_name(shape);
            }
        }
    }

    /// Counts a construct by its name (calls, `set`, struct literals); only `run_covered` pays for it.
    pub(super) fn count_name(&mut self, name: &str) {
        if let Some(counting) = &mut self.coverage {
            *counting.named.entry(name.to_owned()).or_default() += 1;
        }
    }

    /// Resolves a place: evaluates its index, once, and checks it against the array's length (a `Violation::Index`
    /// otherwise: Rust would panic), and returns where the value lives. A place without an index evaluates nothing.
    pub(super) fn locate(&mut self, place: &Place) -> Result<Loc, Violation> {
        let program = self.program;
        let methods = &program.methods;
        let mut base = 0;
        let mut ty = self.resolve(place.root_ty);
        if let Some(index) = place.index.as_deref() {
            let Ty::Array(a) = place.root_ty else {
                unreachable!("Program::check requires an array under an index")
            };
            let bits = self.expr(index)?.int();
            let in_range = if index.ty() == Ty::I32 {
                (bits as i32) >= 0 && (bits as usize) < usize::from(a.n)
            } else {
                (bits as usize) < usize::from(a.n)
            };
            if !in_range {
                return Err(Violation::Index);
            }
            base = bits as usize * methods.ty_len(a.e.ty());
            ty = a.e.ty();
        }
        let (ty, offset) = methods.walk(ty, &place.path);
        Ok(Loc {
            root: place.root,
            offset: base + offset,
            ty,
            whole: place.index.is_none() && place.path.is_empty(),
        })
    }

    fn root_val(&self, root: Root) -> Val {
        match root {
            Root::Var(var) => self.vars[var as usize],
            Root::SelfV => *self.self_stack.last().expect("`self` outside a call"),
        }
    }

    /// The value at a resolved place: a scalar, or a struct, a vector, a matrix or an array as its flat components.
    pub(super) fn loc_get(&self, loc: &Loc) -> Val {
        let root = self.root_val(loc.root);
        if loc.whole {
            return root;
        }
        let Val::A(agg) = root else {
            unreachable!("a part of a scalar: {loc:?} holds {root:?}")
        };
        match loc.ty {
            Ty::U32 | Ty::I32 | Ty::Bool | Ty::F32 | Ty::Enum(_) => agg.c[loc.offset].into(),
            other => {
                let len = self.program.methods.ty_len(other);
                Val::A(Agg::new(&agg.c[loc.offset..loc.offset + len]))
            }
        }
    }

    /// Stores `value` at a resolved place (the root is rewritten whole, as Rust writes the part in place).
    pub(super) fn loc_set(&mut self, loc: &Loc, value: Val) {
        if loc.whole {
            match loc.root {
                Root::Var(var) => self.vars[var as usize] = value,
                Root::SelfV => *self.self_stack.last_mut().expect("`self` outside a call") = value,
            }
            return;
        }
        let root = match loc.root {
            Root::Var(var) => &mut self.vars[var as usize],
            Root::SelfV => self.self_stack.last_mut().expect("`self` outside a call"),
        };
        let Val::A(agg) = root else {
            unreachable!("a part of a scalar")
        };
        match value {
            Val::A(part) => agg.c[loc.offset..loc.offset + part.n].copy_from_slice(part.lanes()),
            scalar => agg.c[loc.offset] = scalar.scalar(),
        }
    }

    /// A type resolved in the binding of the generic function that is running: `T` is the type it was called with.
    fn resolve(&self, ty: Ty) -> Ty {
        match ty {
            Ty::Param(g) => self
                .binds
                .last()
                .and_then(|b| b[usize::from(g)])
                .expect("a bound type parameter"),
            other => other,
        }
    }

    /// A call, in Rust's order: the receiver is read first (a by-value `self` is copied now; for `&self` and `&mut
    /// self` the arguments cannot change it, `ast::Check` refuses a `&mut self` call on its root inside them), then
    /// the arguments left to right, then the body. `&mut self` writes the final receiver back to its place.
    pub(super) fn call(
        &mut self,
        f: usize,
        recv: &Option<Place>,
        args: &[Expr],
    ) -> Result<Option<Val>, Violation> {
        let program = self.program;
        let mut f = f;
        let mut def = &program.methods.fns[f];
        self.count_name(call_name(def.recv, def.owner));
        if self.coverage.is_some() {
            for name in trait_call_names(&program.methods, f, recv, args) {
                self.count_name(name);
            }
        }
        if self.coverage.is_some()
            && let Some(shape) = call_shape(&program.methods, f, recv, args)
        {
            self.count_name(shape);
        }
        if self.coverage.is_some() {
            for name in call_arg_names(def.recv != Recv::None, args) {
                self.count_name(name);
            }
        }
        if self.coverage.is_some() {
            for name in array_call_names(recv)
                .into_iter()
                .chain(crate::ast::array_arg_names(args))
            {
                self.count_name(name);
            }
        }
        if self.coverage.is_some()
            && let Some((closure, _)) = self.closure_ctx
        {
            self.count_name(if def.recv == Recv::None && def.owner.is_none() {
                CLOSURE_BODY_HELPER
            } else {
                CLOSURE_BODY_METHOD
            });
            if recv.as_ref().is_some_and(|place| {
                closure
                    .params
                    .iter()
                    .any(|p| p.1.is_struct() && place.root == Root::Var(p.0))
            }) {
                self.count_name(CLOSURE_PARAM_METHOD);
            }
        }
        let loc = recv.as_ref().map(|place| self.locate(place)).transpose()?;
        let receiver = loc.as_ref().map(|loc| self.loc_get(loc));
        // A trait method is resolved by the concrete type of its receiver (a type parameter through its binding), from the
        // trait method the node names and not from the impl it happens to carry.
        if let (Some(imp), Some(loc)) = (def.imp, loc.as_ref()) {
            let rty = self.resolve(loc.ty);
            let via_param = recv
                .as_ref()
                .is_some_and(|p| matches!(p.root_ty, Ty::Param(_)));
            let mut target = program
                .methods
                .impl_fn(imp.tr, imp.m, rty)
                .expect("the receiver's type implements the trait");
            if self.mutant == Some(Mutant::TraitWrongImpl) && !via_param {
                // The wrong model (a call on a concrete type): the impl of the other implementing type runs.
                let other = program
                    .methods
                    .implementors(imp.tr)
                    .into_iter()
                    .find(|ty| *ty != rty)
                    .expect("a trait has two implementors");
                target = program.methods.impl_fn(imp.tr, imp.m, other).expect("impl");
            }
            debug_assert!(
                (self.mutant == Some(Mutant::TraitWrongImpl) && !via_param)
                    || target == f
                    || via_param,
                "the node's impl is not the receiver's: f={f} target={target} imp={imp:?} rty={rty:?} loc={:?}",
                loc.ty
            );
            f = target;
            def = &program.methods.fns[f];
        }
        let reverse = matches!(self.mutant, Some(Mutant::TraitArgsRightToLeft))
            && (def.imp.is_some() || !def.gens.is_empty());
        let mut slots: Vec<Option<Val>> = vec![None; args.len()];
        let order: Vec<usize> = if reverse {
            (0..args.len()).rev().collect()
        } else {
            (0..args.len()).collect()
        };
        for k in order {
            slots[k] = Some(self.expr(&args[k])?);
        }
        let values: Vec<Val> = slots.into_iter().flatten().collect();
        // A generic function binds each type parameter to the type of the arguments declared with it (resolved in the
        // caller's binding) for the length of its body.
        let mut bind: [Option<Ty>; 2] = [None, None];
        let generic = !def.gens.is_empty();
        if generic {
            for (arg, (_, pty)) in args.iter().zip(&def.params) {
                if let Ty::Param(g) = pty {
                    bind[usize::from(*g)] = Some(self.resolve(arg.ty()));
                }
            }
            if self.mutant == Some(Mutant::GenericReuse) {
                // The wrong model: the first instantiation's specialisation serves every later call.
                bind = *self.first_bind.entry(f).or_insert(bind);
            }
            if let (Some(coverage), Some(t)) = (self.coverage.as_mut(), bind[0]) {
                coverage.instantiations.insert((f, format!("{t:?}")));
            }
        }
        // The arguments of a borrowing call (`&self`, `&mut self`) must not change the receiver, or the copy-in and
        // copy-out model of this interpreter would differ from Rust's references (the generator and `ast::Check`
        // keep it so).
        if def.recv != Recv::Value
            && let (Some(loc), Some(before)) = (loc.as_ref(), receiver.as_ref())
        {
            debug_assert_eq!(
                format!("{before:?}"),
                format!("{:?}", self.loc_get(loc)),
                "the arguments of a borrowing call changed its receiver"
            );
        }
        for ((var, _), value) in def.params.iter().zip(values) {
            self.vars[*var as usize] = value;
        }
        if let Some(receiver) = receiver {
            self.self_stack.push(receiver);
        }
        let ctx = self.closure_ctx.take();
        let depths = (self.for_depth, self.arm_depth);
        (self.for_depth, self.arm_depth) = (0, 0);
        if generic {
            self.binds.push(bind);
        }
        let ran = self.block(&def.body);
        (self.for_depth, self.arm_depth) = depths;
        let value = match ran {
            Ok(()) => match self.returned.take() {
                Some(value) => Ok(Some(value)),
                None => def.tail.as_ref().map(|tail| self.expr(tail)).transpose(),
            },
            Err(violation) => Err(violation),
        };
        if generic {
            self.binds.pop();
        }
        let value = value?;
        self.closure_ctx = ctx;
        if self.mutant == Some(Mutant::ArrayArgByRef) {
            // The wrong model: an array argument that is a variable is the callee's parameter itself.
            for (arg, (var, _)) in args.iter().zip(&def.params) {
                if let Expr::Var(outer, Ty::Array(_)) = arg {
                    self.vars[*outer as usize] = self.vars[*var as usize];
                }
            }
        }
        if let Some(loc) = loc.as_ref() {
            let last = self.self_stack.pop().expect("a receiver was pushed");
            if def.recv == Recv::Mut {
                self.loc_set(loc, last);
            }
        }
        Ok(value)
    }

    /// A closure call, in Rust's order: the arguments left to right, each completely, then the parameters are bound by
    /// value (an array or a struct is copied: the slot of the parameter is its own), then the body runs once and the
    /// block's value is the result. A closure holds no capture, so the caller's variables are untouched by the body.
    pub(super) fn closure_call(&mut self, var: u32, args: &[Expr]) -> Result<Val, Violation> {
        let program = self.program;
        let &(def, site) = self.closures.get(&var).expect("a call of a bound closure");
        if self.coverage.is_some() {
            for name in closure_call_names(&program.methods, var, args) {
                self.count_name(name);
            }
        }
        let mut values = vec![None; args.len()];
        let order: Vec<usize> = if self.mutant == Some(Mutant::ClosureArgsRightToLeft) {
            (0..args.len()).rev().collect()
        } else {
            (0..args.len()).collect()
        };
        for k in order {
            values[k] = Some(self.expr(&args[k])?);
        }
        for (&(param, ..), value) in def.params.iter().zip(values.into_iter().flatten()) {
            self.vars[param as usize] = value;
        }
        if self.mutant == Some(Mutant::ClosureMemo)
            && let Some(&memo) = self.memo.get(&var)
        {
            return Ok(memo);
        }
        let in_method = site.is_some_and(|k| program.methods.fns[k].owner.is_some());
        let saved = self.closure_ctx.replace((def, in_method));
        let depths = (self.for_depth, self.arm_depth);
        (self.for_depth, self.arm_depth) = (0, 0);
        let ran = self.block(&def.body).and_then(|()| self.expr(&def.tail));
        (self.for_depth, self.arm_depth) = depths;
        self.closure_ctx = saved;
        let value = ran?;
        if self.mutant == Some(Mutant::ClosureParamByRef) {
            for (arg, p) in args.iter().zip(&def.params) {
                if let Expr::Var(outer, ty) = arg
                    && (ty.is_array() || ty.is_struct())
                {
                    self.vars[*outer as usize] = self.vars[p.0 as usize];
                }
            }
        }
        if self.mutant == Some(Mutant::ClosureMemo) {
            self.memo.insert(var, value);
        }
        Ok(value)
    }

    /// A value `match` (`tail`: the value of an arm of another one): the scrutinee once, the first arm that names it,
    /// then that arm's block and its value. `None` if the arm returned from the function (`self.returned` is set).
    pub(super) fn value_match(
        &mut self,
        scrut: &Expr,
        arms: &[Arm],
        tail: bool,
    ) -> Result<Option<Val>, Violation> {
        if self.coverage.is_some() {
            for name in value_match_names(&self.program.methods, scrut, arms, tail) {
                self.count_name(name);
            }
        }
        let bits = self.expr(scrut)?.int();
        let variant = match scrut.ty() {
            Ty::Enum(e) => self.program.methods.enums[usize::from(e)]
                .variants
                .iter()
                .position(|v| v.0 == bits)
                .expect("an enum value is a discriminant") as u8,
            _ => 0,
        };
        let mut taken = None;
        for (k, arm) in arms.iter().enumerate() {
            if self.mutant == Some(Mutant::ValueScrutineeTwice) && k > 0 {
                self.expr(scrut)?;
            }
            let hit = match &arm.pat {
                Pat::Variants(vs) => vs.contains(&variant),
                Pat::Lits(lits) if self.mutant == Some(Mutant::GroupFirstOnly) => lits[0].0 == bits,
                Pat::Lits(lits) => lits.iter().any(|&(lit, _)| lit == bits),
                Pat::Wild => true,
            };
            if hit {
                taken = Some(arm);
                break;
            }
        }
        let arm = taken.expect("a value match is exhaustive");
        self.arm_depth += 1;
        let done = self.arm_value(&arm.body);
        self.arm_depth -= 1;
        done
    }

    /// The block of a value arm: its statements, then its value (`None` after a `return`).
    fn arm_value(&mut self, body: &[Stmt]) -> Result<Option<Val>, Violation> {
        let (last, rest) = body.split_last().expect("a value arm is not empty");
        self.block(rest)?;
        if self.returned.is_some() {
            return Ok(None);
        }
        match last {
            Stmt::Yield { value } => Ok(Some(self.expr(value)?)),
            Stmt::TailMatch { scrut, arms } => self.value_match(scrut, arms, true),
            other => {
                self.block(std::slice::from_ref(other))?;
                Ok(None)
            }
        }
    }
}
