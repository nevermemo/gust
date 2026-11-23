//! The tuple and value-match space.

use super::*;

/// The value space (format 10, slice m9.d1c): tuples, the value form of `match` and integer patterns. Every draw sits
/// behind `Gen::values`, so the older spaces decode as before.
impl Gen<'_> {
    /// A tuple `(S, u32)` or `(S, i32)` whose struct `S` has a `&mut self` method returning one of its scalar field
    /// types: the local that lets `tuple_field_set_stmt` write `t.0.f op= t.0.m()` (fix round 1).
    pub(super) fn field_set_tuple_ty(&mut self) -> Option<Ty> {
        let mut options = Vec::new();
        for (j, def) in self.table.structs.iter().enumerate() {
            if def.fields.iter().any(|f| f.is_struct()) {
                continue;
            }
            let fits = (0..self.callable()).any(|m| {
                let f = &self.table.fns[m];
                f.recv == Recv::Mut
                    && f.owner == Some(j as u8)
                    && matches!(f.ret, Some(Ty::U32 | Ty::I32))
                    && f.ret.is_some_and(|r| {
                        def.fields
                            .iter()
                            .enumerate()
                            .any(|(k, field)| *field == r && writes_self_field(&f.body, k as u8))
                    })
            });
            if fits {
                options.push(j as u8);
            }
        }
        if options.is_empty() {
            return None;
        }
        let j = options[self.b.below(options.len() as u32) as usize];
        let other = if self.b.chance(50) {
            Elem::U32
        } else {
            Elem::I32
        };
        let ty = Ty::Tuple(TupleTy::new(&[Elem::Struct(j), other]));
        (self.table.ty_len(ty) <= MAX_TUPLE_FLAT).then_some(ty)
    }

    /// A tuple type of two to four elements (scalars, a struct of scalar fields, an enum), at most `MAX_TUPLE_FLAT`
    /// scalars once the structs are flattened.
    pub(super) fn tuple_ty(&mut self) -> Ty {
        let n = 2 + self.b.weighted(&[50, 35, 15]);
        let scalar_structs: Vec<u8> = (0..self.table.structs.len() as u8)
            .filter(|&j| {
                self.table.structs[usize::from(j)]
                    .fields
                    .iter()
                    .all(|f| !f.is_struct())
            })
            .collect();
        let enums = self.table.enums.len() as u32;
        let mut elems = Vec::new();
        for _ in 0..n {
            let weights = [
                30,
                26,
                12,
                if scalar_structs.is_empty() { 0 } else { 30 },
                if enums == 0 { 0 } else { 14 },
            ];
            elems.push(match self.b.weighted(&weights) {
                0 => Elem::U32,
                1 => Elem::I32,
                2 => Elem::Bool,
                3 => {
                    Elem::Struct(scalar_structs[self.b.below(scalar_structs.len() as u32) as usize])
                }
                _ => Elem::Enum(self.b.below(enums) as u8),
            });
        }
        while self.table.ty_len(Ty::Tuple(TupleTy::new(&elems))) > MAX_TUPLE_FLAT {
            let last = elems
                .iter()
                .rposition(|e| matches!(e, Elem::Struct(_)))
                .expect("only structs make a tuple too long");
            elems[last] = Elem::U32;
        }
        Ty::Tuple(TupleTy::new(&elems))
    }

    /// A value of tuple type `ty`: a literal, a local, or the result of a call.
    pub(super) fn tuple_expr(&mut self, ty: Ty, depth: u32) -> Expr {
        let Ty::Tuple(t) = ty else {
            unreachable!("a tuple type")
        };
        let vars = self.vars(ty);
        let calls = depth > 0 && (0..self.callable()).any(|k| self.table.fns[k].ret == Some(ty));
        let weights = [
            60,
            if vars.is_empty() { 0 } else { 18 },
            if calls { 26 } else { 0 },
        ];
        match self.b.weighted(&weights) {
            1 => Expr::Var(vars[self.b.below(vars.len() as u32) as usize], ty),
            2 => match self.value_call(ty, depth - 1) {
                Some(call) => call,
                None => self.tuple_literal(t, depth),
            },
            _ => self.tuple_literal(t, depth),
        }
    }

    /// `(a, b, c)`. Often the elements share one root: a `&mut self` call and a read or a copy of the same root, so that
    /// the order the elements run in is what the value shows.
    fn tuple_literal(&mut self, t: TupleTy, depth: u32) -> Expr {
        let root = if depth >= 1 && self.b.chance(65) {
            self.pick_writable_root()
        } else {
            None
        };
        let n = usize::from(t.n);
        let mut elems = Vec::new();
        for (k, elem) in t.elems().iter().enumerate() {
            elems.push(self.tuple_elem(*elem, depth.saturating_sub(1), root, k + 1 == n));
        }
        Expr::Tuple {
            ty: Ty::Tuple(t),
            elems,
        }
    }

    fn tuple_elem(&mut self, elem: Elem, depth: u32, root: Option<Root>, last: bool) -> Expr {
        let ty = elem.ty();
        match elem {
            Elem::Struct(j) => {
                if let Some(root) = root
                    && self.b.chance(55)
                    && let Some(copy) = self.struct_copy_on(j, root)
                {
                    return copy;
                }
                match self.struct_expr(j, depth, false) {
                    Some(value) => value,
                    None => self.hoist_struct(j),
                }
            }
            Elem::Enum(_) => {
                if let Some(root) = root
                    && self.b.chance(60)
                    && let Some(call) = self.root_call(ty, root, depth)
                {
                    return call;
                }
                self.enum_expr(ty, depth)
            }
            Elem::Bool if !last && self.b.chance(40) => {
                let operand = self.int_ty();
                Expr::bin(
                    BinOp::Lt,
                    self.expr(operand, depth.min(2)),
                    self.expr(operand, depth.min(2)),
                )
            }
            _ => {
                if let Some(root) = root {
                    if self.b.chance(50)
                        && let Some(call) = self.root_call(ty, root, depth)
                    {
                        return call;
                    }
                    if self.b.chance(30) {
                        let reads: Vec<Place> = self
                            .scalar_places(ty)
                            .into_iter()
                            .filter(|(place, _)| place.root == root)
                            .map(|(place, _)| place)
                            .collect();
                        if !reads.is_empty() {
                            let place = reads[self.b.below(reads.len() as u32) as usize].clone();
                            return Expr::Read { place, ty };
                        }
                    }
                }
                self.expr(ty, depth)
            }
        }
    }

    /// A copy of a struct-`j` place on `root` (the struct as it is when the element runs).
    fn struct_copy_on(&mut self, j: u8, root: Root) -> Option<Expr> {
        let bare_self_ok = matches!(self.this, Some((_, Recv::Value)));
        let places: Vec<Place> = self
            .struct_places()
            .into_iter()
            .filter(|(place, pj, _)| {
                *pj == j
                    && place.root == root
                    && !(place.root == Root::SelfV && place.path.is_empty() && !bare_self_ok)
            })
            .map(|(place, ..)| place)
            .collect();
        if places.is_empty() {
            return None;
        }
        let place = places[self.b.below(places.len() as u32) as usize].clone();
        Some(place_value(&place, Ty::Struct(j)))
    }

    /// A struct value that is a variable: a struct literal cannot be a tuple element (E0810), so it is bound by a `let`
    /// before the statement being drawn (`Gen::pre`).
    fn hoist_struct(&mut self, j: u8) -> Expr {
        let ty = Ty::Struct(j);
        let init = match self.struct_expr(j, 3, true) {
            Some(init) => init,
            // A nested struct field with no value in reach: hoist that one too, then a literal of variables.
            None => {
                let fields = self.table.structs[usize::from(j)].fields.clone();
                let values: Vec<Expr> = fields
                    .into_iter()
                    .map(|field| match field {
                        Ty::Struct(inner) => self.hoist_struct(inner),
                        scalar => self.expr(scalar, 2),
                    })
                    .collect();
                Expr::StructLit {
                    ty,
                    order: (0..values.len() as u8).collect(),
                    fields: values,
                }
            }
        };
        let var = self.fresh();
        let mutable = self.b.chance(50);
        self.scope.push((var, ty, mutable));
        self.pre.push(Stmt::Let {
            var,
            ty,
            mutable,
            init,
        });
        Expr::Var(var, ty)
    }

    /// Moves the lets that drawing since `mark` hoisted to `at` in `stmts` (before the statement that needs them).
    pub(super) fn splice_pre(&mut self, mark: usize, at: usize, stmts: &mut Vec<Stmt>) {
        if self.pre.len() > mark {
            let hoisted = self.pre.split_off(mark);
            stmts.splice(at..at, hoisted);
        }
    }

    /// `f().k` for a function returning a tuple with an element of type `ty`; with `root`, a `&mut self` method on a
    /// writable place of that root (the call runs before the element is read, whatever follows).
    pub(super) fn tuple_field_call(
        &mut self,
        ty: Ty,
        depth: u32,
        root: Option<Root>,
    ) -> Option<Expr> {
        if depth == 0 || root.is_some_and(|root| self.forbid.contains(&root)) {
            return None;
        }
        let places = self.struct_places();
        let on_root = |owner: Option<u8>, root: Root| {
            places
                .iter()
                .filter(|(place, j, writable)| *writable && place.root == root && Some(*j) == owner)
                .map(|(place, ..)| place.clone())
                .collect::<Vec<Place>>()
        };
        let matching = |t: TupleTy| -> Vec<u8> {
            (0..t.n)
                .filter(|&k| t.elems()[usize::from(k)].ty() == ty)
                .collect()
        };
        let candidates: Vec<usize> = (0..self.callable())
            .filter(|&k| {
                let def = &self.table.fns[k];
                let Some(Ty::Tuple(t)) = def.ret else {
                    return false;
                };
                !matching(t).is_empty()
                    && match root {
                        Some(root) => def.recv == Recv::Mut && !on_root(def.owner, root).is_empty(),
                        None => true,
                    }
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let f = candidates[self.b.below(candidates.len() as u32) as usize];
        let def = self.table.fns[f].clone();
        let Some(Ty::Tuple(t)) = def.ret else {
            unreachable!("a tuple result")
        };
        let forced = root.map(|root| {
            let receivers = on_root(def.owner, root);
            receivers[self.b.below(receivers.len() as u32) as usize].clone()
        });
        let (recv, args) = self.call_parts_on(f, depth, forced)?;
        let indexes = matching(t);
        let index = indexes[self.b.below(indexes.len() as u32) as usize];
        let owner = if def.recv == Recv::None {
            def.owner
        } else {
            None
        };
        Some(Expr::TupleField {
            e: Box::new(Expr::Call {
                f,
                owner,
                recv,
                args,
                ty: Ty::Tuple(t),
            }),
            index,
        })
    }

    /// The element places of tuple locals that may be assigned: `u32` and `i32` elements of a `let mut` tuple.
    pub(super) fn tuple_set_targets(&self) -> Vec<(Place, Ty)> {
        let mut out = Vec::new();
        for &(var, ty, mutable) in &self.scope {
            let Ty::Tuple(t) = ty else { continue };
            if !mutable {
                continue;
            }
            for (k, elem) in t.elems().iter().enumerate() {
                if matches!(elem, Elem::U32 | Elem::I32) {
                    out.push((Place::var(var, ty).field(k as u8), elem.ty()));
                }
            }
        }
        out
    }

    /// The scalar fields of the struct elements of mutable tuple locals, each with a `&mut self` method of that struct
    /// returning the field's type: `(tuple var, tuple type, element, field, field type)`. A compound set of such a
    /// field whose value calls the method on the same element, `t.k.f *= t.k.m()`, is the one tuple-place assignment
    /// where the order of the place read and the value is observable (m9.d1c fix round 1).
    pub(super) fn tuple_field_set_candidates(&self) -> Vec<(u32, Ty, u8, u8, Ty)> {
        let mut out = Vec::new();
        for &(var, ty, mutable) in &self.scope {
            let Ty::Tuple(t) = ty else { continue };
            if !mutable || self.forbid.contains(&Root::Var(var)) {
                continue;
            }
            for (k, elem) in t.elems().iter().enumerate() {
                let Elem::Struct(j) = elem else { continue };
                for (f, field) in self.table.structs[usize::from(*j)]
                    .fields
                    .iter()
                    .enumerate()
                {
                    let scalar = matches!(field, Ty::U32 | Ty::I32);
                    let method = (0..self.callable()).any(|m| {
                        let def = &self.table.fns[m];
                        def.recv == Recv::Mut
                            && def.owner == Some(*j)
                            && def.ret == Some(*field)
                            && writes_self_field(&def.body, f as u8)
                    });
                    if scalar && method {
                        out.push((var, ty, k as u8, f as u8, *field));
                    }
                }
            }
        }
        out
    }

    /// `t.k.f op= t.k.m();` for a `&mut self` method `m` of the struct at element `k` returning `f`'s type: Rust runs
    /// the call first, so the place is read after the call changed it.
    pub(super) fn tuple_field_set_stmt(&mut self) -> Option<Stmt> {
        let candidates = self.tuple_field_set_candidates();
        if candidates.is_empty() {
            return None;
        }
        let (var, ty, k, f, field) = candidates[self.b.below(candidates.len() as u32) as usize];
        let Ty::Tuple(t) = ty else {
            unreachable!("a tuple local")
        };
        let Elem::Struct(j) = t.elems()[usize::from(k)] else {
            unreachable!("a struct element")
        };
        let methods: Vec<usize> = (0..self.callable())
            .filter(|&m| {
                let def = &self.table.fns[m];
                def.recv == Recv::Mut
                    && def.owner == Some(j)
                    && def.ret == Some(field)
                    && writes_self_field(&def.body, f)
            })
            .collect();
        let m = methods[self.b.below(methods.len() as u32) as usize];
        let element = Place::var(var, ty).field(k);
        let (recv, args) = self.call_parts_on(m, 3, Some(element.clone()))?;
        let op = [
            BinOp::Add,
            BinOp::Sub,
            BinOp::BitXor,
            BinOp::BitOr,
            BinOp::Mul,
        ][self.b.below(5) as usize];
        Some(Stmt::Set {
            place: element.field(f),
            ty: field,
            op: Some(op),
            value: Expr::Call {
                f: m,
                owner: None,
                recv,
                args,
                ty: field,
            },
        })
    }

    /// `t.a = t.b + c.bump();` or `t.a += c.bump();`: the value runs first, then the place is read (for `op=`).
    pub(super) fn tuple_set_stmt(&mut self, targets: &[(Place, Ty)]) -> Option<Stmt> {
        if (targets.is_empty() || self.b.chance(85))
            && let Some(stmt) = self.tuple_field_set_stmt()
        {
            return Some(stmt);
        }
        if targets.is_empty() {
            return None;
        }
        let (place, ty) = targets[self.b.below(targets.len() as u32) as usize].clone();
        let op = self.b.chance(55).then(|| {
            [
                BinOp::Add,
                BinOp::Add,
                BinOp::Sub,
                BinOp::BitXor,
                BinOp::BitOr,
                BinOp::BitAnd,
            ][self.b.below(6) as usize]
        });
        // A `&mut self` call on a writable root, tried on up to three roots.
        let mut call = None;
        for _ in 0..3 {
            if call.is_some() {
                break;
            }
            call = self
                .pick_writable_root()
                .and_then(|root| self.root_call(ty, root, 3));
        }
        let value = match call {
            Some(call) if op.is_none() && self.b.chance(85) => {
                // Another element of the same tuple (or the target itself) read before the call runs.
                let Ty::Tuple(t) = place.root_ty else {
                    unreachable!("a tuple place")
                };
                let same: Vec<u8> = (0..t.n)
                    .filter(|&k| t.elems()[usize::from(k)].ty() == ty)
                    .collect();
                let k = same[self.b.below(same.len() as u32) as usize];
                let read = Expr::Read {
                    place: Place {
                        root: place.root,
                        root_ty: place.root_ty,
                        path: vec![k],
                        index: None,
                    },
                    ty,
                };
                let joint = [BinOp::Add, BinOp::Sub, BinOp::BitXor][self.b.below(3) as usize];
                if self.b.chance(50) {
                    Expr::bin(joint, read, call)
                } else {
                    Expr::bin(joint, call, read)
                }
            }
            Some(call) => call,
            None => self.expr(ty, 3),
        };
        Some(Stmt::Set {
            place,
            ty,
            op,
            value,
        })
    }

    /// `let (a, mut b): (T, U) = init;`. Often the source is a call of a function that returns a tuple, a `&mut self`
    /// method first, so that the root it changed is read afterwards.
    pub(super) fn let_tuple(&mut self) -> Stmt {
        let rets: Vec<usize> = (0..self.callable())
            .filter(|&k| matches!(self.table.fns[k].ret, Some(Ty::Tuple(_))))
            .collect();
        let mut init = None;
        if !rets.is_empty() && self.b.chance(60) {
            let muts: Vec<usize> = rets
                .iter()
                .copied()
                .filter(|&k| self.table.fns[k].recv == Recv::Mut)
                .collect();
            let pool = if !muts.is_empty() && self.b.chance(65) {
                muts
            } else {
                rets
            };
            let f = pool[self.b.below(pool.len() as u32) as usize];
            let def = self.table.fns[f].clone();
            if let Some((recv, args)) = self.call_parts(f, 3) {
                init = Some(Expr::Call {
                    f,
                    owner: if def.recv == Recv::None {
                        def.owner
                    } else {
                        None
                    },
                    recv,
                    args,
                    ty: def.ret.expect("a tuple result"),
                });
            }
        }
        let init = match init {
            Some(init) => init,
            None => {
                let ty = self.tuple_ty();
                self.tuple_expr(ty, 3)
            }
        };
        let ty = init.ty();
        let Ty::Tuple(t) = ty else {
            unreachable!("a tuple initializer")
        };
        let annotate = t.elems().iter().any(|e| matches!(e, Elem::Enum(_))) || self.b.chance(50);
        let mut vars = Vec::new();
        for elem in t.elems() {
            let var = self.fresh();
            let mutable = self.b.chance(40);
            self.scope.push((var, elem.ty(), mutable));
            vars.push((var, mutable));
        }
        Stmt::LetTuple {
            vars,
            ty,
            annotate,
            init,
        }
    }

    /// The type of a value `match`: mostly an integer, sometimes `bool` or an enum.
    fn match_result_ty(&mut self) -> Ty {
        if self.b.chance(22) {
            self.enum_ty()
        } else {
            [Ty::U32, Ty::I32, Ty::Bool][self.b.weighted(&[45, 35, 20])]
        }
    }

    /// `let v: T = match scrut { p0 => return .., p1 => return .., _ => value };` in a helper that returns `ret`: every
    /// arm but the last returns, so that a `return` in an arm runs for some element (nit R1 of the m9.d1c brief).
    pub(super) fn returning_match(&mut self, ret: Ty) -> Stmt {
        let ty = self.match_result_ty();
        let (scrut, mut arms) = self.value_match_parts(ty, 1);
        let last = arms.len() - 1;
        for arm in &mut arms[..last] {
            let (mark, scope_mark) = (self.pre.len(), self.scope.len());
            let value = self.expr(ret, 3);
            let mut body = vec![Stmt::Return { value }];
            self.splice_pre(mark, 0, &mut body);
            // A local that the arm hoisted leaves the scope with the arm.
            self.scope.truncate(scope_mark);
            arm.body = body;
        }
        let var = self.fresh();
        self.scope.push((var, ty, false));
        Stmt::LetMatch {
            var,
            ty,
            mutable: false,
            scrut,
            arms,
        }
    }

    /// `let v: T = match scrut { .. };`
    pub(super) fn let_match(&mut self, depth: u32) -> Stmt {
        let ty = self.match_result_ty();
        let (scrut, arms) = self.value_match_parts(ty, depth);
        let var = self.fresh();
        let mutable = self.b.chance(40);
        self.scope.push((var, ty, mutable));
        Stmt::LetMatch {
            var,
            ty,
            mutable,
            scrut,
            arms,
        }
    }

    /// An enum scrutinee is a variable, a call or a variant path (E0515): any other expression (an element of a tuple)
    /// becomes the first variant.
    pub(super) fn enum_scrutinee(&mut self, e: u8, scrut: Expr) -> Expr {
        if matches!(
            scrut,
            Expr::Var(..) | Expr::Call { .. } | Expr::PathCall { .. } | Expr::Variant { .. }
        ) {
            scrut
        } else {
            Expr::Variant { e, v: 0 }
        }
    }

    /// The scrutinee and the arms of a value `match` of type `ty`: on an enum (a third) or on an integer.
    fn value_match_parts(&mut self, ty: Ty, depth: u32) -> (Expr, Vec<Arm>) {
        let mut bias_root = None;
        let (scrut, pats) = if self.b.chance(35) {
            let e = self.b.below(self.table.enums.len() as u32) as u8;
            let enum_ty = Ty::Enum(e);
            let n = self.table.enums[usize::from(e)].variants.len();
            let mut scrut = None;
            match self.b.weighted(&[30, 45, 18, 7]) {
                1 => {
                    if let Some(root) = self.pick_writable_root()
                        && let Some(call) = self.root_call(enum_ty, root, 3)
                    {
                        bias_root = Some(root);
                        scrut = Some(call);
                    }
                }
                2 => scrut = self.value_call(enum_ty, 2),
                3 => {
                    let v = self.b.below(n as u32) as u8;
                    scrut = Some(Expr::Variant { e, v });
                }
                _ => {}
            }
            let scrut = match scrut {
                Some(scrut) => scrut,
                None => self.enum_leaf(enum_ty),
            };
            let scrut = self.enum_scrutinee(e, scrut);
            (scrut, self.enum_patterns(e))
        } else {
            let int = self.int_ty();
            let mut scrut = None;
            match self.b.weighted(&[38, 38, 12, 12]) {
                0 => {
                    if let Some(root) = self.pick_writable_root()
                        && let Some(call) = self.root_call(int, root, 3)
                    {
                        bias_root = Some(root);
                        scrut = Some(Expr::bin(BinOp::BitAnd, call, Expr::Lit(int, 7)));
                    }
                }
                2 => {
                    let vars = self.vars(int);
                    if !vars.is_empty() {
                        scrut = Some(Expr::Var(
                            vars[self.b.below(vars.len() as u32) as usize],
                            int,
                        ));
                    }
                }
                3 => {
                    let operand = self.expr(int, 3);
                    scrut = Some(Expr::bin(BinOp::BitAnd, operand, Expr::Lit(int, 7)));
                }
                _ => {}
            }
            let scrut = match scrut {
                Some(scrut) => scrut,
                None => {
                    let operand = self.expr(int, 3);
                    let k = [3, 4, 5, 8][self.b.below(4) as usize];
                    Expr::bin(BinOp::Rem, operand, Expr::Lit(int, k))
                }
            };
            let pats = self.int_patterns(int);
            (scrut, pats)
        };
        let mut arms = Vec::new();
        for pat in pats {
            let body = self.value_arm(ty, depth, bias_root);
            arms.push(Arm { pat, body });
        }
        (scrut, arms)
    }

    /// The patterns of an enum value `match`: the variants shuffled and cut into groups, `_` replacing the last group
    /// 45 % of the time.
    pub(super) fn enum_patterns(&mut self, e: u8) -> Vec<Pat> {
        let n = self.table.enums[usize::from(e)].variants.len();
        let mut order: Vec<u8> = (0..n as u8).collect();
        for k in (1..order.len()).rev() {
            let other = self.b.below(k as u32 + 1) as usize;
            order.swap(k, other);
        }
        let mut groups: Vec<Vec<u8>> = vec![vec![order[0]]];
        for &v in &order[1..] {
            if self.b.chance(40) {
                groups.last_mut().expect("a group").push(v);
            } else {
                groups.push(vec![v]);
            }
        }
        let wild = self.b.chance(45);
        if wild {
            if groups.len() == 1 {
                let last = groups[0].pop().expect("two variants at least");
                groups.push(vec![last]);
            }
            groups.pop();
        }
        let mut pats: Vec<Pat> = groups.into_iter().map(Pat::Variants).collect();
        if wild {
            pats.push(Pat::Wild);
        }
        pats
    }

    /// The patterns of an integer value `match`: one to four distinct small literals (an `i32` also negative ones, a
    /// rare wide one), joined by `|` in groups 40 % of the time, written with or without the suffix, then `_`.
    fn int_patterns(&mut self, int: Ty) -> Vec<Pat> {
        let mut pool: Vec<u32> = (0..8).collect();
        if int == Ty::I32 {
            pool.extend([-1i32 as u32, -2i32 as u32, -3i32 as u32]);
        }
        for k in (1..pool.len()).rev() {
            let other = self.b.below(k as u32 + 1) as usize;
            pool.swap(k, other);
        }
        let count = 1 + self.b.below(4) as usize;
        pool.truncate(count);
        if self.b.chance(12) {
            let wide = if int == Ty::I32 {
                i32::MAX as u32
            } else {
                u32::MAX
            };
            if !pool.contains(&wide) {
                pool[0] = wide;
            }
        }
        let mut groups: Vec<Vec<u32>> = vec![vec![pool[0]]];
        for &v in &pool[1..] {
            if self.b.chance(40) {
                groups.last_mut().expect("a group").push(v);
            } else {
                groups.push(vec![v]);
            }
        }
        let mut pats = Vec::new();
        for group in groups {
            let lits = group
                .into_iter()
                .map(|bits| (bits, self.b.chance(40)))
                .collect();
            pats.push(Pat::Lits(lits));
        }
        pats.push(Pat::Wild);
        pats
    }

    /// The block of one arm of a value `match` of type `ty`: statements, then its value: an expression, a nested value
    /// `match`, or (in a helper) a `return`. An arm after a scrutinee that mutated `bias_root` often writes it first.
    fn value_arm(&mut self, ty: Ty, depth: u32, bias_root: Option<Root>) -> Vec<Stmt> {
        let scope_mark = self.scope.len();
        let pre_mark = self.pre.len();
        let kind = self.b.weighted(&[
            38,
            22,
            if depth < 2 { 12 } else { 0 },
            if self.cur_ret.is_some() { 45 } else { 0 },
        ]);
        let mut body = match kind {
            0 => Vec::new(),
            1 => self.block(depth + 1, 1, 2),
            _ => self.block(depth + 1, 0, 1),
        };
        if let Some(root) = bias_root
            && self.b.chance(70)
        {
            let targets: Vec<(Place, Ty)> = self
                .set_targets()
                .into_iter()
                .filter(|(place, _)| place.root == root)
                .collect();
            if !targets.is_empty() {
                body.insert(0, self.set_stmt(&targets));
            }
        }
        let last = match kind {
            2 => {
                let (scrut, arms) = self.value_match_parts(ty, depth + 1);
                Stmt::TailMatch { scrut, arms }
            }
            3 => {
                let ret = self.cur_ret.expect("a return type");
                Stmt::Return {
                    value: self.expr(ret, 3),
                }
            }
            _ => Stmt::Yield {
                value: self.expr(ty, 3),
            },
        };
        let at = body.len();
        body.push(last);
        self.splice_pre(pre_mark, at, &mut body);
        self.scope.truncate(scope_mark);
        body
    }
}

/// Whether a `return` sits inside an arm of a `match` in `stmts`.
pub(super) fn arm_returns(stmts: &[Stmt], in_arm: bool) -> bool {
    stmts.iter().any(|stmt| match stmt {
        Stmt::Return { .. } => in_arm,
        Stmt::If { then, els, .. } => arm_returns(then, in_arm) || arm_returns(els, in_arm),
        Stmt::For { body, .. } => arm_returns(body, in_arm),
        Stmt::Match { arms, .. } | Stmt::LetMatch { arms, .. } | Stmt::TailMatch { arms, .. } => {
            arms.iter().any(|arm| arm_returns(&arm.body, true))
        }
        _ => false,
    })
}

/// Whether a method body assigns the field `field` of `self` anywhere (so that a call of it changes that place).
fn writes_self_field(stmts: &[Stmt], field: u8) -> bool {
    stmts.iter().any(|stmt| match stmt {
        Stmt::Set { place, .. } => place.root == Root::SelfV && place.path.first() == Some(&field),
        Stmt::If { then, els, .. } => {
            writes_self_field(then, field) || writes_self_field(els, field)
        }
        Stmt::For { body, .. } => writes_self_field(body, field),
        Stmt::Match { arms, .. } | Stmt::LetMatch { arms, .. } | Stmt::TailMatch { arms, .. } => {
            arms.iter().any(|arm| writes_self_field(&arm.body, field))
        }
        _ => false,
    })
}
