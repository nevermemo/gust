//! `Program::check`: the invariants the generator promises.

use super::*;

pub(super) struct Check<'a> {
    pub(super) inputs: Vec<Ty>,
    pub(super) methods: &'a Methods,
    /// (variable, type, mutable), innermost last.
    pub(super) scope: Vec<(u32, Ty, bool)>,
    pub(super) seen: BTreeSet<u32>,
    /// The index of the function being checked (`None` in the kernel). A function sees no buffer, no `i`.
    pub(super) in_fn: Option<usize>,
    /// Inside a closure body (format 12): no `i`, buffer, `self`, `return` or closure.
    pub(super) in_closure: bool,
    /// The closures bound in the open blocks: (variable, parameter types, result type).
    pub(super) closures: Vec<(u32, Vec<Ty>, Ty)>,
    /// The struct parameters of closures: a parameter is a read-only value to Slang, so it is no `&mut self` receiver
    /// (E0333), `mut` or not.
    pub(super) closure_params: Vec<u32>,
}

/// A `return` is the last statement of its block (the statements after one would be dead code).
fn returns_last(stmts: &[Stmt]) -> Result<(), String> {
    match stmts.split_last() {
        Some((_, rest)) if rest.iter().any(|s| matches!(s, Stmt::Return { .. })) => {
            Err("a `return` that is not the last statement of its block".into())
        }
        _ => Ok(()),
    }
}

/// The roots of the `&mut self` calls inside `expr` (a call's receiver, and the calls in its arguments).
pub fn mutating_roots(methods: &Methods, expr: &Expr, out: &mut Vec<Root>) {
    if let Expr::Call {
        f,
        recv: Some(place),
        ..
    } = expr
        && methods.fns[*f].recv == Recv::Mut
    {
        out.push(place.root);
    }
    expr.for_each_child(&mut |child| mutating_roots(methods, child, out));
}

impl Check<'_> {
    /// The type of a place, or why it is not one.
    fn place_ty(&self, place: &Place) -> Result<Ty, String> {
        let root = match place.root {
            Root::Var(var) => match self.lookup(var) {
                Some((ty, _)) => ty,
                None => return Err(format!("place rooted at unknown v{var}")),
            },
            Root::SelfV if self.in_closure => {
                return Err("a closure captures `self` (E0329)".into());
            }
            Root::SelfV => match self.in_fn.and_then(|k| {
                let def = &self.methods.fns[k];
                def.owner.map(Ty::Struct).or(def.imp.map(|i| i.ty))
            }) {
                Some(ty) if self.self_recv() != Recv::None => ty,
                _ => return Err("`self` outside a method".into()),
            },
        };
        if root != place.root_ty {
            return Err("a place with the wrong root type".into());
        }
        let mut ty = root;
        if place.index.is_some() {
            ty = match ty {
                Ty::Array(a) => a.e.ty(),
                other => return Err(format!("an index into {}", other.name())),
            };
        }
        for &k in &place.path {
            ty = match ty {
                Ty::Struct(j) => *self.methods.structs[usize::from(j)]
                    .fields
                    .get(usize::from(k))
                    .ok_or_else(|| format!("S{j} has no field {k}"))?,
                Ty::Tuple(t) if k < t.n => t.elems()[usize::from(k)].ty(),
                other => return Err(format!("a field of {}", other.name())),
            };
        }
        Ok(ty)
    }

    /// The index of an indexed place: an integer expression with no `&mut self` call (E0421), checked as an expression.
    fn place_index(&mut self, place: &Place, write: bool) -> Result<(), String> {
        let Some(index) = place.index.as_deref() else {
            return Ok(());
        };
        self.expr(index)?;
        if !matches!(index.ty(), Ty::U32 | Ty::I32) {
            return Err("an array index that is not an integer".into());
        }
        let mut roots = Vec::new();
        mutating_roots(self.methods, index, &mut roots);
        if write && !roots.is_empty() {
            return Err("a `&mut self` call inside an index (E0421)".into());
        }
        Ok(())
    }

    fn self_recv(&self) -> Recv {
        self.in_fn.map_or(Recv::None, |k| self.methods.fns[k].recv)
    }

    /// Whether the place may be written: a `let mut` local, or `self` of a `&mut self` method.
    fn writable(&self, place: &Place) -> bool {
        match place.root {
            Root::Var(var) => self.lookup(var).is_some_and(|(_, mutable)| mutable),
            Root::SelfV => self.self_recv() == Recv::Mut,
        }
    }

    /// A call's shape: the callee exists below the caller, the arguments match, the receiver is a place of the
    /// owner's type that may be borrowed as the callee needs, and no `&mut self` call on the receiver's root sits in
    /// the arguments of a call that borrows it (rustc's E0499 and E0502).
    fn call(
        &mut self,
        f: usize,
        owner: Option<u8>,
        recv: &Option<Place>,
        args: &[Expr],
    ) -> Result<Option<Ty>, String> {
        let methods = self.methods;
        let Some(def) = methods.fns.get(f) else {
            return Err(format!("call of unknown m{f}"));
        };
        if def.recv == Recv::None && owner != def.owner {
            return Err(format!("m{f} is called with the wrong path"));
        }
        if self.in_fn.is_some_and(|caller| f >= caller) {
            return Err(format!(
                "m{f} is called from a function that is not above it"
            ));
        }
        if args.len() != def.params.len() {
            return Err(format!("m{f} takes {} arguments", def.params.len()));
        }
        let mut bound: [Option<Ty>; 2] = [None, None];
        let mut self_param: Option<Ty> = None;
        for (arg, (_, ty)) in args.iter().zip(&def.params) {
            self.expr(arg)?;
            match ty {
                Ty::Param(g) => {
                    let slot = &mut bound[usize::from(*g)];
                    if slot.is_some_and(|t| t != arg.ty()) {
                        return Err(format!("m{f}: one type parameter bound to two types"));
                    }
                    *slot = Some(arg.ty());
                }
                _ if arg.ty() != *ty => {
                    return Err(format!("argument of m{f} has the wrong type"));
                }
                _ => {}
            }
        }
        for (g, gen_param) in def.gens.iter().enumerate() {
            let Some(actual) = bound[g] else {
                return Err(format!("m{f}: a type parameter nothing binds"));
            };
            for &b in &gen_param.bounds {
                let ok = match actual {
                    Ty::Param(c) => self.in_fn.is_some_and(|k| {
                        methods.fns[k]
                            .gens
                            .get(usize::from(c))
                            .is_some_and(|p| p.bounds.contains(&b))
                    }),
                    Ty::Struct(_) | Ty::Enum(_) => methods.implements(actual, b),
                    _ => false,
                };
                if !ok {
                    return Err(format!("m{f}: the type bound to a parameter lacks Tr{b}"));
                }
            }
        }
        match (def.recv, recv) {
            (Recv::None, None) => {}
            (Recv::None, Some(_)) | (_, None) => {
                return Err(format!("m{f}: receiver and signature disagree"));
            }
            (kind, Some(place)) => {
                self.place_index(place, true)?;
                let rty = self.place_ty(place)?;
                let fits = match (def.imp, rty) {
                    (Some(imp), Ty::Param(c)) => self.in_fn.is_some_and(|k| {
                        methods.fns[k]
                            .gens
                            .get(usize::from(c))
                            .is_some_and(|p| p.bounds.contains(&imp.tr))
                    }),
                    (Some(imp), ty) => ty == imp.ty,
                    (None, ty) => ty == Ty::Struct(def.owner.unwrap_or(u8::MAX)),
                };
                if !fits {
                    return Err(format!("receiver of m{f} has the wrong type"));
                }
                if let (Some(imp), Ty::Param(_)) = (def.imp, rty)
                    && methods.traits[usize::from(imp.tr)].methods[usize::from(imp.m)].ret
                        == TRet::SelfTy
                {
                    self_param = Some(rty);
                }
                if kind == Recv::Mut
                    && matches!(place.root, Root::Var(v) if self.closure_params.contains(&v))
                {
                    return Err(format!(
                        "`&mut self` call m{f} on a closure parameter (E0333)"
                    ));
                }
                if kind == Recv::Mut && !self.writable(place) {
                    return Err(format!(
                        "`&mut self` call m{f} on a place that is not writable"
                    ));
                }
                if kind != Recv::Value {
                    let mut roots = Vec::new();
                    for arg in args {
                        mutating_roots(methods, arg, &mut roots);
                    }
                    if roots.contains(&place.root) {
                        return Err(format!(
                            "an argument of m{f} calls a `&mut self` method on its receiver's root"
                        ));
                    }
                }
            }
        }
        Ok(match def.ret {
            _ if self_param.is_some() => self_param,
            Some(Ty::Param(g)) => bound[usize::from(g)],
            other => other,
        })
    }

    /// One function in a scope of its own: its parameters, its body and its tail.
    pub(super) fn function(&mut self, k: usize, def: &FnDef) -> Result<(), String> {
        self.in_fn = Some(k);
        self.scope.clear();
        self.closures.clear();
        for &(var, ty) in &def.params {
            self.declare(var, ty, ty.is_array())?;
        }
        returns_last(&def.body)?;
        for stmt in &def.body {
            self.stmt(stmt)?;
        }
        match (&def.tail, def.ret) {
            (Some(tail), Some(ty)) => {
                self.expr(tail)?;
                if tail.ty() != ty {
                    return Err(format!("the tail of m{k} has the wrong type"));
                }
            }
            (None, None) => {}
            _ => return Err(format!("m{k}: the tail and the result type disagree")),
        }
        // A bare `self` as a value is only a `self` method's: in `&mut self` it binds the reference (E0116) and in
        // `&self` it has the type `&Self`.
        if def.recv != Recv::Value {
            let mut bare = false;
            walk_fn_exprs(def, &mut |e| {
                bare |= matches!(e, Expr::Read { place, .. } if place.root == Root::SelfV && place.path.is_empty());
            });
            if bare {
                return Err(format!("m{k} uses `self` as a value"));
            }
        }
        Ok(())
    }

    /// A closure binding: the parameters are the only names its body sees (no capture), the body calls no closure and
    /// holds no `return`, `i`, buffer or `self`, and the tail has the written result type. The closure is bound after
    /// its body is checked (no recursion), for the rest of the block.
    fn let_closure(&mut self, def: &ClosureDef) -> Result<(), String> {
        if self.in_closure {
            return Err("a closure inside a closure (E0332)".into());
        }
        if !self.seen.insert(def.var) {
            return Err(format!("closure cl{} declared twice", def.var));
        }
        for ty in def.params.iter().map(|p| p.1).chain([def.ret]) {
            if ty.is_aggregate() || ty.is_tuple() {
                return Err(format!("a closure signature with {} (E0331)", ty.name()));
            }
        }
        let saved_scope = std::mem::take(&mut self.scope);
        let saved_closures = std::mem::take(&mut self.closures);
        self.in_closure = true;
        let outcome = self.closure_body(def);
        self.in_closure = false;
        self.scope = saved_scope;
        self.closures = saved_closures;
        outcome?;
        self.closures
            .push((def.var, def.params.iter().map(|p| p.1).collect(), def.ret));
        Ok(())
    }

    fn closure_body(&mut self, def: &ClosureDef) -> Result<(), String> {
        for &(var, ty, mutable) in &def.params {
            if ty.is_array() && !mutable {
                return Err("an array closure parameter that is not `mut`".into());
            }
            self.declare(var, ty, mutable)?;
            if ty.is_struct() {
                self.closure_params.push(var);
            }
        }
        for stmt in &def.body {
            self.stmt(stmt)?;
        }
        self.expr(&def.tail)?;
        if def.tail.ty() != def.ret {
            return Err(format!("the tail of cl{} has the wrong type", def.var));
        }
        Ok(())
    }

    fn declare(&mut self, var: u32, ty: Ty, mutable: bool) -> Result<(), String> {
        if !self.seen.insert(var) {
            return Err(format!("variable v{var} declared twice"));
        }
        self.scope.push((var, ty, mutable));
        Ok(())
    }

    fn lookup(&self, var: u32) -> Option<(Ty, bool)> {
        self.scope
            .iter()
            .rev()
            .find(|(id, ..)| *id == var)
            .map(|&(_, ty, mutable)| (ty, mutable))
    }

    fn block(&mut self, stmts: &[Stmt]) -> Result<(), String> {
        let mark = self.scope.len();
        let closure_mark = self.closures.len();
        returns_last(stmts)?;
        for stmt in stmts {
            self.stmt(stmt)?;
        }
        self.scope.truncate(mark);
        self.closures.truncate(closure_mark);
        Ok(())
    }

    /// A statement `match` on an enum: the scrutinee is a local, a parameter, a call or a variant path (E0515), the
    /// patterns name each variant at most once (E0521), `_` is last and stands for at least one variant, and without
    /// `_` every variant is named, so Rust accepts the match and the last arm is the `default` of the Slang `switch`.
    fn match_stmt(&mut self, scrut: &Expr, arms: &[Arm], value: Option<Ty>) -> Result<(), String> {
        self.expr(scrut)?;
        let e = match scrut.ty() {
            Ty::Enum(e) => e,
            Ty::U32 | Ty::I32 if value.is_some() => return self.int_match(scrut.ty(), arms, value),
            other => return Err(format!("a match on {}", other.name())),
        };
        if !matches!(
            scrut,
            Expr::Var(..) | Expr::Call { .. } | Expr::Variant { .. }
        ) {
            return Err("a match scrutinee is a local, a call or a variant".into());
        }
        let Some(def) = self.methods.enums.get(usize::from(e)) else {
            return Err(format!("a match on unknown E{e}"));
        };
        let mut named = BTreeSet::new();
        let mut wild = false;
        let mut named_arms = 0;
        for arm in arms {
            if wild {
                return Err("an arm after `_`".into());
            }
            match &arm.pat {
                Pat::Wild => wild = true,
                Pat::Lits(_) => return Err("an integer pattern in an enum match".into()),
                Pat::Variants(variants) => {
                    named_arms += 1;
                    if variants.is_empty() {
                        return Err("an arm with no variant".into());
                    }
                    for &v in variants {
                        if usize::from(v) >= def.variants.len() {
                            return Err(format!("E{e} has no variant V{v}"));
                        }
                        if !named.insert(v) {
                            return Err(format!("E{e}::V{v} is named twice"));
                        }
                    }
                }
            }
            self.arm_body(&arm.body, value)?;
        }
        if named_arms == 0 {
            return Err("a match with no variant arm".into());
        }
        match (wild, named.len() == def.variants.len()) {
            (false, false) => Err("a match without `_` that does not name every variant".into()),
            (true, true) => Err("a `_` arm that no variant reaches".into()),
            _ => Ok(()),
        }
    }

    /// The body of a `match` arm: a block, and in the value form (`value` the type of the `let`) a block that ends in
    /// the value of the arm: a `Yield` of that type, a `Return` (helpers) or a nested value `match`.
    fn arm_body(&mut self, body: &[Stmt], value: Option<Ty>) -> Result<(), String> {
        let Some(ty) = value else {
            return self.block(body);
        };
        let mark = self.scope.len();
        returns_last(body)?;
        let Some((last, rest)) = body.split_last() else {
            return Err("an empty value arm".into());
        };
        for stmt in rest {
            self.stmt(stmt)?;
        }
        match last {
            Stmt::Yield { value } => {
                self.expr(value)?;
                if value.ty() != ty {
                    return Err(format!(
                        "a value arm yields {} for {}",
                        value.ty().name(),
                        ty.name()
                    ));
                }
            }
            Stmt::Return { .. } => self.stmt(last)?,
            Stmt::TailMatch { scrut, arms } => self.match_stmt(scrut, arms, Some(ty))?,
            _ => return Err("a value arm that does not end in its value".into()),
        }
        self.scope.truncate(mark);
        Ok(())
    }

    /// A value `match` on an integer: distinct `u32`/`i32` literal patterns (E0521 for a value twice), the literals of
    /// one arm joined by `|`, and a final `_` (an integer `match` is not exhaustive without it).
    fn int_match(&mut self, scrut_ty: Ty, arms: &[Arm], value: Option<Ty>) -> Result<(), String> {
        let mut seen = BTreeSet::new();
        let mut wild = false;
        let mut literal_arms = 0;
        for arm in arms {
            if wild {
                return Err("an arm after `_`".into());
            }
            match &arm.pat {
                Pat::Wild => wild = true,
                Pat::Variants(_) => return Err("a variant pattern in an integer match".into()),
                Pat::Lits(lits) => {
                    literal_arms += 1;
                    if lits.is_empty() {
                        return Err("an arm with no literal".into());
                    }
                    for &(bits, _) in lits {
                        if scrut_ty == Ty::I32 && bits == i32::MIN as u32 {
                            return Err("an i32::MIN pattern".into());
                        }
                        if !seen.insert(bits) {
                            return Err(format!("the pattern {bits} is written twice"));
                        }
                    }
                }
            }
            self.arm_body(&arm.body, value)?;
        }
        if literal_arms == 0 || !wild {
            return Err("an integer match needs a literal arm and a final `_`".into());
        }
        Ok(())
    }

    pub(super) fn stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::Let {
                var,
                ty,
                mutable,
                init,
            } => {
                self.expr(init)?;
                if init.ty() != *ty {
                    return Err(format!(
                        "let v{var}: {} initialised with {}",
                        ty.name(),
                        init.ty().name()
                    ));
                }
                self.declare(*var, *ty, *mutable)
            }
            Stmt::Assign { var, value } => {
                self.expr(value)?;
                match self.lookup(*var) {
                    Some((ty, true)) if ty == value.ty() => Ok(()),
                    Some((_, true)) => Err(format!("assignment to v{var} has the wrong type")),
                    Some((_, false)) => Err(format!("assignment to immutable v{var}")),
                    None => Err(format!("assignment to unknown v{var}")),
                }
            }
            Stmt::AssignComp { var, index, value } => {
                self.expr(value)?;
                match self.lookup(*var) {
                    Some((Ty::Vec(lane, width), true)) => {
                        if *index >= width {
                            Err(format!("component {index} of a {width}-wide v{var}"))
                        } else if value.ty() != lane.ty() {
                            Err(format!("component assignment to v{var} has the wrong type"))
                        } else {
                            Ok(())
                        }
                    }
                    Some((_, true)) => Err(format!("component assignment to a non-vector v{var}")),
                    Some((_, false)) => Err(format!("component assignment to immutable v{var}")),
                    None => Err(format!("component assignment to unknown v{var}")),
                }
            }
            Stmt::If { cond, then, els } => {
                self.expr(cond)?;
                if cond.ty() != Ty::Bool {
                    return Err("`if` condition is not bool".into());
                }
                self.block(then)?;
                self.block(els)
            }
            Stmt::For { var, trips, body } => {
                if !(1..=MAX_TRIPS).contains(trips) {
                    return Err(format!("loop trip count {trips}"));
                }
                let mark = self.scope.len();
                self.declare(*var, Ty::U32, false)?;
                self.block(body)?;
                self.scope.truncate(mark);
                Ok(())
            }
            Stmt::Set {
                place,
                ty,
                op,
                value,
            } => {
                self.expr(value)?;
                self.place_index(place, true)?;
                if self.place_ty(place)? != *ty {
                    return Err("assignment to a place of another type".into());
                }
                if !self.writable(place) {
                    return Err("assignment to a place that is not writable".into());
                }
                match op {
                    None if value.ty() == *ty => Ok(()),
                    None => Err("assignment of the wrong type".into()),
                    Some(op) => {
                        let right = if op.is_shift() { Ty::U32 } else { *ty };
                        let allowed = matches!(
                            op,
                            BinOp::Add
                                | BinOp::Sub
                                | BinOp::Mul
                                | BinOp::Div
                                | BinOp::Rem
                                | BinOp::BitAnd
                                | BinOp::BitOr
                                | BinOp::BitXor
                                | BinOp::Shl
                                | BinOp::Shr
                        );
                        if !allowed || !ty.is_int() || value.ty() != right {
                            return Err(format!("compound `{}=` on {}", op.symbol(), ty.name()));
                        }
                        Ok(())
                    }
                }
            }
            Stmt::Call {
                f,
                owner,
                recv,
                args,
            } => self.call(*f, *owner, recv, args).map(|_| ()),
            Stmt::Match { scrut, arms } => self.match_stmt(scrut, arms, None),
            Stmt::Yield { .. } | Stmt::TailMatch { .. } => {
                Err("a `Yield` or `TailMatch` that does not end a value arm".into())
            }
            Stmt::LetMatch {
                var,
                ty,
                mutable,
                scrut,
                arms,
            } => {
                if !matches!(ty, Ty::U32 | Ty::I32 | Ty::Bool | Ty::Enum(_)) {
                    return Err(format!("a value match of type {}", ty.name()));
                }
                self.match_stmt(scrut, arms, Some(*ty))?;
                self.declare(*var, *ty, *mutable)
            }
            Stmt::LetTuple {
                vars,
                ty,
                annotate,
                init,
            } => {
                self.expr(init)?;
                let Ty::Tuple(t) = ty else {
                    return Err("a destructuring of a non-tuple".into());
                };
                self.methods.check_tuple(*t)?;
                if init.ty() != *ty || vars.len() != usize::from(t.n) {
                    return Err("a destructuring that does not fit its tuple".into());
                }
                if !annotate && t.elems().iter().any(|e| matches!(e, Elem::Enum(_))) {
                    return Err("an unannotated destructuring of an enum".into());
                }
                for (&(var, mutable), elem) in vars.iter().zip(t.elems()) {
                    self.declare(var, elem.ty(), mutable)?;
                }
                Ok(())
            }
            Stmt::LetClosure { def } => self.let_closure(def),
            Stmt::Return { .. } if self.in_closure => Err("a `return` in a closure body".into()),
            Stmt::Return { value } => {
                self.expr(value)?;
                let helper = self.in_fn.map(|k| &self.methods.fns[k]);
                match helper {
                    Some(def)
                        if def.recv == Recv::None
                            && def.owner.is_none()
                            && def.ret == Some(value.ty()) =>
                    {
                        Ok(())
                    }
                    _ => Err("a `return` outside a helper, or of the wrong type".into()),
                }
            }
        }
    }

    pub(super) fn expr(&mut self, expr: &Expr) -> Result<(), String> {
        match expr {
            Expr::Lit(Ty::Bool, bits) if *bits > 1 => Err("bool literal out of range".into()),
            Expr::Lit(Ty::I32, bits) if *bits == i32::MIN as u32 => Err("i32::MIN literal".into()),
            Expr::Lit(Ty::F32, bits) if !f32::from_bits(*bits).is_finite() => {
                Err("non-finite f32 literal".into())
            }
            Expr::Lit(Ty::F32, bits) if *bits == (-0.0f32).to_bits() => {
                Err("negative zero literal".into())
            }
            Expr::Lit(ty, _) if ty.is_aggregate() || ty.is_struct() || ty.is_enum() => {
                Err("a vector, matrix, struct or enum literal".into())
            }
            Expr::Variant { e, v } => match self.methods.enums.get(usize::from(*e)) {
                Some(def) if usize::from(*v) < def.variants.len() => Ok(()),
                _ => Err(format!("E{e}::V{v} does not exist")),
            },
            Expr::EnumAs { to, e } => {
                self.expr(e)?;
                match e.ty() {
                    Ty::Enum(k)
                        if self.methods.enums.get(usize::from(k)).map(|def| def.repr)
                            == Some(*to) =>
                    {
                        Ok(())
                    }
                    other => Err(format!("{} as {}", other.name(), to.name())),
                }
            }
            Expr::Gid if self.in_fn.is_some() || self.in_closure => {
                Err("`i` inside a function or a closure".into())
            }
            Expr::Lit(..) | Expr::Gid => Ok(()),
            Expr::Var(var, ty) => match self.lookup(*var) {
                Some((declared, _)) if declared == *ty => Ok(()),
                Some(_) => Err(format!("v{var} used at the wrong type")),
                None => Err(format!("v{var} used out of scope")),
            },
            Expr::Len(_) | Expr::Input { .. } if self.in_fn.is_some() || self.in_closure => {
                Err("a buffer inside a function or a closure".into())
            }
            Expr::Len(buf) => {
                if *buf < self.inputs.len() {
                    Ok(())
                } else {
                    Err(format!("len of unknown buffer {buf}"))
                }
            }
            Expr::Input { buf, ty, index } => {
                self.expr(index)?;
                if ty.is_aggregate()
                    || ty.is_struct()
                    || ty.is_enum()
                    || self.inputs.get(*buf) != Some(ty)
                {
                    return Err(format!("input {buf} read at the wrong type"));
                }
                if index.ty() != Ty::U32 {
                    return Err("buffer index is not u32".into());
                }
                Ok(())
            }
            Expr::Un { op, ty, e } => {
                self.expr(e)?;
                if e.ty() != *ty {
                    return Err("unary operand type mismatch".into());
                }
                match op {
                    UnOp::Neg if !matches!(*ty, Ty::I32 | Ty::F32) => {
                        Err("negation of a non-i32, non-f32".into())
                    }
                    UnOp::Neg if matches!(**e, Expr::Lit(Ty::I32, _)) => {
                        Err("negated literal not folded".into())
                    }
                    UnOp::Neg if matches!(**e, Expr::Lit(Ty::F32, bits) if f32::from_bits(bits) != 0.0) => {
                        Err("negated literal not folded".into())
                    }
                    UnOp::Not if *ty == Ty::F32 => Err("`!` on f32".into()),
                    UnOp::Not if ty.is_aggregate() || ty.is_struct() || ty.is_enum() => {
                        Err("`!` on a vector, matrix, struct or enum".into())
                    }
                    UnOp::Neg if ty.is_struct() => Err("negation of a struct".into()),
                    _ => Ok(()),
                }
            }
            Expr::Bin { op, ty, l, r } => {
                self.expr(l)?;
                self.expr(r)?;
                if l.ty() != *ty {
                    return Err("binary left operand type mismatch".into());
                }
                if ty.is_aggregate() {
                    return Err(format!("`{}` on {} is an `AggBin`", op.symbol(), ty.name()));
                }
                if ty.is_struct() {
                    return Err(format!("`{}` on a struct", op.symbol()));
                }
                if ty.is_enum() {
                    // `==` and `!=` only: the enum derives `PartialEq` and nothing else.
                    return if matches!(op, BinOp::Eq | BinOp::Ne) && r.ty() == *ty {
                        Ok(())
                    } else {
                        Err(format!("`{}` on an enum", op.symbol()))
                    };
                }
                let (need_int, right) = if op.is_shift() {
                    (true, Ty::U32)
                } else if op.is_logical() {
                    (false, Ty::Bool)
                } else if op.is_comparison()
                    || (*ty == Ty::F32
                        && matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div))
                {
                    (false, *ty)
                } else {
                    (true, *ty)
                };
                if op.is_logical() && *ty != Ty::Bool {
                    return Err("logical operator on a non-bool".into());
                }
                if op.is_comparison() && !matches!(op, BinOp::Eq | BinOp::Ne) && *ty == Ty::Bool {
                    return Err(format!("`{}` on bool", op.symbol()));
                }
                if need_int && !ty.is_int() {
                    return Err(format!("`{}` on {}", op.symbol(), ty.name()));
                }
                if r.ty() != right {
                    return Err(format!("`{}` right operand type", op.symbol()));
                }
                Ok(())
            }
            Expr::Cast { to, e } => {
                self.expr(e)?;
                let castable = |ty: Ty| {
                    ty != Ty::Bool && !ty.is_aggregate() && !ty.is_struct() && !ty.is_enum()
                };
                if !castable(*to) || !castable(e.ty()) || e.ty() == *to {
                    return Err("cast outside u32, i32 and f32".into());
                }
                Ok(())
            }
            Expr::Make { ty, rows, args } => {
                for arg in args {
                    self.expr(arg)?;
                }
                let want = match ty {
                    Ty::Vec(lane, width) if WIDTHS.contains(width) && !rows => {
                        (usize::from(*width), lane.ty())
                    }
                    Ty::Mat(n) if WIDTHS.contains(n) => (usize::from(*n), Ty::Vec(Lane::F32, *n)),
                    _ => return Err(format!("`{}` cannot be built from components", ty.name())),
                };
                if args.len() != want.0 || args.iter().any(|arg| arg.ty() != want.1) {
                    return Err(format!(
                        "`{}` needs {} arguments of type {}",
                        ty.name(),
                        want.0,
                        want.1.name()
                    ));
                }
                // SEMANTICS section 4: a compile-time constant `Mat2` crashes naga's SPIR-V front end. A matrix whose
                // constructor reads no `f32` input buffer directly (a read behind a cast can fold to a constant) is a
                // candidate for constant folding, so the generator never writes one.
                if *ty == Ty::Mat(2) && !args.iter().any(Expr::reads_f32_input_directly) {
                    return Err(
                        "a `Mat2` with no direct `f32` input read could be a compile-time constant"
                            .into(),
                    );
                }
                Ok(())
            }
            Expr::Splat { ty, e } => {
                self.expr(e)?;
                match ty {
                    Ty::Vec(lane, width) if WIDTHS.contains(width) && e.ty() == lane.ty() => Ok(()),
                    _ => Err(format!("`{}::splat` of {}", ty.name(), e.ty().name())),
                }
            }
            Expr::Comp { e, index } => {
                self.expr(e)?;
                match e.ty() {
                    Ty::Vec(_, width) if *index < width => Ok(()),
                    other => Err(format!("component {index} of {}", other.name())),
                }
            }
            Expr::Swizzle { e, lanes } => {
                self.expr(e)?;
                match e.ty() {
                    Ty::Vec(_, width)
                        if WIDTHS.contains(&(lanes.len() as u8))
                            && lanes.iter().all(|&k| k < width) =>
                    {
                        Ok(())
                    }
                    other => Err(format!(
                        "a {}-letter swizzle of {}",
                        lanes.len(),
                        other.name()
                    )),
                }
            }
            Expr::Dot { l, r } => {
                self.expr(l)?;
                self.expr(r)?;
                match (l.ty(), r.ty()) {
                    (a @ Ty::Vec(..), b) if a == b => Ok(()),
                    (a, b) => Err(format!("dot of {} and {}", a.name(), b.name())),
                }
            }
            Expr::Transpose { e } => {
                self.expr(e)?;
                match e.ty() {
                    Ty::Mat(_) => Ok(()),
                    other => Err(format!("transpose of {}", other.name())),
                }
            }
            Expr::AggBin { op, ty, l, r } => {
                self.expr(l)?;
                self.expr(r)?;
                if agg_bin_ty(*op, l.ty(), r.ty()) != Some(*ty) {
                    return Err(format!(
                        "`{}` on {} and {} is not {}",
                        op.symbol(),
                        l.ty().name(),
                        r.ty().name(),
                        ty.name()
                    ));
                }
                Ok(())
            }
            Expr::AggNeg { e } => {
                self.expr(e)?;
                match e.ty() {
                    Ty::Vec(Lane::F32 | Lane::I32, _) | Ty::Mat(_) => Ok(()),
                    other => Err(format!("negation of {}", other.name())),
                }
            }
            Expr::Read { place, ty } => {
                self.place_index(place, false)?;
                if self.place_ty(place)? != *ty {
                    return Err("a field read at the wrong type".into());
                }
                match place.root {
                    Root::Var(_) if place.path.is_empty() && place.index.is_none() => {
                        Err("a bare local is a `Var`, not a `Read`".into())
                    }
                    Root::SelfV if place.path.is_empty() && self.self_recv() != Recv::Value => {
                        Err("`self` as a value outside a `self` method".into())
                    }
                    _ => Ok(()),
                }
            }
            Expr::Call {
                f,
                owner,
                recv,
                args,
                ty,
            } => match self.call(*f, *owner, recv, args)? {
                Some(ret) if ret == *ty => Ok(()),
                _ => Err(format!("a call of m{f} has the wrong type")),
            },
            Expr::Const { k, ty } => match self.methods.consts.get(usize::from(*k)) {
                Some(def) if def.ty == *ty => Ok(()),
                Some(_) => Err(format!("K{k} used at the wrong type")),
                None => Err(format!("K{k} does not exist")),
            },
            Expr::Math { f, ty, args } => {
                for arg in args {
                    self.expr(arg)?;
                }
                let operand = if f.is_float() { Ty::F32 } else { *ty };
                if *ty != operand || (!f.is_float() && !ty.is_int()) {
                    return Err(format!("`{}` on {}", f.name(), ty.name()));
                }
                if args.len() != f.arity() || args.iter().any(|arg| arg.ty() != operand) {
                    return Err(format!(
                        "`{}` needs {} operands of {}",
                        f.name(),
                        f.arity(),
                        operand.name()
                    ));
                }
                Ok(())
            }
            Expr::PathCall { call, .. } => {
                let Expr::Call {
                    f,
                    recv: Some(place),
                    ..
                } = &**call
                else {
                    return Err("a path call that is not a method call".into());
                };
                if self.methods.fns[*f].imp.is_none() {
                    return Err("a path call of a function that implements no trait".into());
                }
                if matches!(place.root_ty, Ty::Param(_)) || place.index.is_some() {
                    return Err("a path call on a type parameter or an element".into());
                }
                self.expr(call)
            }
            Expr::ClosureCall { var, args, ty } => {
                let Some((_, params, ret)) = self.closures.iter().rev().find(|c| c.0 == *var)
                else {
                    return Err(format!("a call of unbound closure cl{var}"));
                };
                let (params, ret) = (params.clone(), *ret);
                if ret != *ty {
                    return Err(format!("a call of cl{var} has the wrong type"));
                }
                if args.len() != params.len() {
                    return Err(format!("cl{var} takes {} arguments", params.len()));
                }
                for (arg, want) in args.iter().zip(&params) {
                    self.expr(arg)?;
                    if arg.ty() != *want {
                        return Err(format!("an argument of cl{var} has the wrong type"));
                    }
                }
                Ok(())
            }
            Expr::Tuple { ty, elems } => {
                let Ty::Tuple(t) = ty else {
                    return Err("a tuple literal of a non-tuple".into());
                };
                self.methods.check_tuple(*t)?;
                if elems.len() != usize::from(t.n) {
                    return Err("a tuple literal of the wrong length".into());
                }
                for (elem, want) in elems.iter().zip(t.elems()) {
                    // E0810: a struct literal is only a `let` initializer or the right side of an assignment.
                    if matches!(elem, Expr::StructLit { .. }) {
                        return Err("a struct literal as a tuple element".into());
                    }
                    self.expr(elem)?;
                    if elem.ty() != want.ty() {
                        return Err("a tuple element of the wrong type".into());
                    }
                }
                Ok(())
            }
            Expr::ArrayLit { ty, elems } => {
                let Ty::Array(a) = ty else {
                    return Err("an array literal of a non-array".into());
                };
                self.methods.check_array(*a)?;
                if elems.len() != usize::from(a.n) {
                    return Err("an array literal of the wrong length".into());
                }
                for elem in elems {
                    // E0810: a struct literal is only a `let` initializer or the right side of an assignment.
                    if matches!(elem, Expr::StructLit { .. }) {
                        return Err("a struct literal as an array element".into());
                    }
                    self.expr(elem)?;
                    if elem.ty() != a.e.ty() {
                        return Err("an array element of the wrong type".into());
                    }
                }
                Ok(())
            }
            Expr::Repeat { ty, e } => {
                let Ty::Array(a) = ty else {
                    return Err("a repeat of a non-array".into());
                };
                self.methods.check_array(*a)?;
                if matches!(**e, Expr::StructLit { .. }) {
                    return Err("a struct literal as a repeated element".into());
                }
                self.expr(e)?;
                if e.ty() != a.e.ty() {
                    return Err("a repeated element of the wrong type".into());
                }
                Ok(())
            }
            Expr::ArrLen { place } => {
                self.place_index(place, false)?;
                if place.index.is_some() || !matches!(self.place_ty(place)?, Ty::Array(_)) {
                    return Err("`.len()` of a non-array".into());
                }
                Ok(())
            }
            Expr::TupleField { e, index } => {
                self.expr(e)?;
                match e.ty() {
                    Ty::Tuple(t) if *index < t.n && matches!(**e, Expr::Call { .. }) => Ok(()),
                    other => Err(format!("element {index} of {} (not a call)", other.name())),
                }
            }
            Expr::StructLit { ty, fields, order } => {
                let Ty::Struct(j) = ty else {
                    return Err("a struct literal of a non-struct".into());
                };
                let Some(def) = self.methods.structs.get(usize::from(*j)) else {
                    return Err(format!("a literal of unknown S{j}"));
                };
                if fields.len() != def.fields.len() {
                    return Err(format!("S{j} literal has {} fields", fields.len()));
                }
                let mut seen = order.clone();
                seen.sort_unstable();
                if order.len() != fields.len()
                    || seen.iter().enumerate().any(|(k, &d)| usize::from(d) != k)
                {
                    return Err(format!("S{j} literal order is not a permutation"));
                }
                for (field, &d) in fields.iter().zip(order) {
                    let want = &def.fields[usize::from(d)];
                    self.expr(field)?;
                    if field.ty() != *want {
                        return Err(format!("a field of the S{j} literal has the wrong type"));
                    }
                }
                Ok(())
            }
        }
    }
}
