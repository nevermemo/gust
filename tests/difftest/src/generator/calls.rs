//! The method space: struct places, calls, tables and functions.

use super::*;

/// A copy of a struct place as an expression (a bare local is a `Var`, a field path or `self` a `Read`).
pub(super) fn place_value(place: &Place, ty: Ty) -> Expr {
    match place.root {
        Root::Var(var) if place.path.is_empty() => Expr::Var(var, ty),
        _ => Expr::Read {
            place: place.clone(),
            ty,
        },
    }
}

/// The method space (format 8): the structs and functions of a program, and the calls and assignments that use them.
/// Every rule of `ast::Check` about receivers, borrows and the call graph is kept here by construction, so that no
/// refused form is drawn (a unit test compiles a sample of seeds through the macro).
impl Gen<'_> {
    /// Every struct-typed place in reach (a local, a parameter, `self`, and the struct fields nested in them), with
    /// the struct and whether the place may be written.
    pub(super) fn struct_places(&self) -> Vec<(Place, u8, bool)> {
        let mut work = Vec::new();
        for &(var, ty, mutable) in &self.scope {
            if let Ty::Struct(j) = ty {
                work.push((Place::var(var, ty), j, mutable));
            }
        }
        if let Some((ty @ Ty::Struct(j), recv)) = self.this {
            work.push((Place::this(ty), j, recv == Recv::Mut));
        }
        // The struct elements of tuple locals and parameters (the value space).
        for &(var, ty, mutable) in &self.scope {
            if let Ty::Tuple(t) = ty {
                for (k, elem) in t.elems().iter().enumerate() {
                    if let Elem::Struct(j) = elem {
                        work.push((Place::var(var, ty).field(k as u8), *j, mutable));
                    }
                }
            }
        }
        let mut out = Vec::new();
        while let Some((place, j, writable)) = work.pop() {
            for (k, field) in self.table.structs[usize::from(j)].fields.iter().enumerate() {
                if let Ty::Struct(inner) = field {
                    work.push((place.field(k as u8), *inner, writable));
                }
            }
            out.push((place, j, writable));
        }
        out
    }

    /// The scalar fields of type `ty` of the struct places in reach, with whether they may be written.
    pub(super) fn scalar_places(&self, ty: Ty) -> Vec<(Place, bool)> {
        let mut out = Vec::new();
        for (place, j, writable) in self.struct_places() {
            for (k, field) in self.table.structs[usize::from(j)].fields.iter().enumerate() {
                if *field == ty {
                    out.push((place.field(k as u8), writable));
                }
            }
        }
        // The scalar elements of tuple locals and parameters.
        for &(var, vty, mutable) in &self.scope {
            if let Ty::Tuple(t) = vty {
                for (k, elem) in t.elems().iter().enumerate() {
                    if !matches!(elem, Elem::Struct(_)) && elem.ty() == ty {
                        out.push((Place::var(var, vty).field(k as u8), mutable));
                    }
                }
            }
        }
        out
    }

    /// Whether an ordinary call may name function `k`: not a generic function (its arguments are drawn by `generic_call`) and
    /// not the impl of an enum (its receiver is no struct place).
    fn plain(&self, k: usize) -> bool {
        let def = &self.table.fns[k];
        def.gens.is_empty() && !(def.recv != Recv::None && def.owner.is_none())
    }

    /// The functions a call may name here: those below the one being drawn (the kernel may name all).
    pub(super) fn callable(&self) -> usize {
        self.cur_fn.unwrap_or(self.table.fns.len())
    }

    /// The places an assignment may write: scalar fields, and (never `*self`) whole struct places.
    pub(super) fn set_targets(&self) -> Vec<(Place, Ty)> {
        let mut out = Vec::new();
        for (place, j, writable) in self.struct_places() {
            if !writable {
                continue;
            }
            for (k, field) in self.table.structs[usize::from(j)].fields.iter().enumerate() {
                if !field.is_struct() {
                    out.push((place.field(k as u8), *field));
                }
            }
            if !(place.root == Root::SelfV && place.path.is_empty()) {
                out.push((place, Ty::Struct(j)));
            }
        }
        out
    }

    pub(super) fn set_stmt(&mut self, targets: &[(Place, Ty)]) -> Stmt {
        let (place, ty) = targets[self.b.below(targets.len() as u32) as usize].clone();
        let (op, value) = match ty {
            // A whole-struct assignment; if no value of the struct is in reach, the place is assigned to itself.
            Ty::Struct(j) => (
                None,
                self.struct_expr(j, 3, true)
                    .unwrap_or_else(|| place_value(&place, ty)),
            ),
            _ => {
                debug_assert!(ty.is_int(), "a struct field is a u32 or an i32");
                let op = self.b.chance(70).then(|| {
                    [
                        BinOp::Add,
                        BinOp::Add,
                        BinOp::Sub,
                        BinOp::BitXor,
                        BinOp::BitXor,
                        BinOp::BitOr,
                        BinOp::Mul,
                        BinOp::Div,
                        BinOp::Rem,
                        BinOp::BitAnd,
                        BinOp::Shl,
                        BinOp::Shr,
                        // `&=`, `>>=` and `%=` again: the same-root forms of each are reached with the call as the value.
                        BinOp::BitAnd,
                        BinOp::Shr,
                        BinOp::Rem,
                    ][self.b.below(15) as usize]
                });
                // A shift takes a `u32` amount whatever the place's type.
                let value_ty = if op.is_some_and(|op| op.is_shift()) {
                    Ty::U32
                } else {
                    ty
                };
                // A share of the values call a `&mut self` method on the place's own root: the order rules of
                // `place op= value` (the place is read after the value) are only observable then.
                let biased = if self.b.chance(75) {
                    // A closure call whose arguments call the place's root: the order of the place read and of the
                    // arguments decides the stored value.
                    let via = if self.closures && !self.cls.is_empty() && self.b.chance(40) {
                        self.closure_call(value_ty, 3, Some(place.root))
                    } else {
                        None
                    };
                    via.or_else(|| self.root_call(value_ty, place.root, 3))
                } else {
                    None
                };
                let value = match (biased, op) {
                    (Some(call), Some(BinOp::Div | BinOp::Rem)) => {
                        Expr::bin(BinOp::BitOr, call, Expr::Lit(value_ty, 1))
                    }
                    // A shift amount is written `call & 31`, so the call's value stays below the width.
                    (Some(call), Some(BinOp::Shl | BinOp::Shr)) => {
                        Expr::bin(BinOp::BitAnd, call, Expr::u32(31))
                    }
                    (Some(call), _) => call,
                    (None, Some(BinOp::Div | BinOp::Rem)) => self.divisor(ty, 3),
                    (None, Some(BinOp::Shl | BinOp::Shr)) => self.shift_amount(3),
                    (None, _) => self.expr(ty, 3),
                };
                (op, value)
            }
        };
        Stmt::Set {
            place,
            ty,
            op,
            value,
        }
    }

    /// The `&mut self` methods that can be called as a statement here.
    pub(super) fn statement_calls(&self) -> Vec<usize> {
        let places = self.struct_places();
        (0..self.callable())
            .filter(|&k| {
                let def = &self.table.fns[k];
                self.plain(k)
                    && def.recv == Recv::Mut
                    && places
                        .iter()
                        .any(|(_, j, writable)| *writable && Some(*j) == def.owner)
            })
            .collect()
    }

    /// A `&mut self` call as a statement; `None` when an argument of struct type has no value in reach.
    pub(super) fn statement_call(&mut self, calls: &[usize]) -> Option<Stmt> {
        let f = calls[self.b.below(calls.len() as u32) as usize];
        let (recv, args) = self.call_parts(f, 3)?;
        Some(Stmt::Call {
            f,
            owner: None,
            recv,
            args,
        })
    }

    /// The receiver and the arguments of a call of function `f`, or `None` when no receiver is in reach. A `&mut
    /// self` call needs a writable receiver whose root no enclosing call is borrowing; while the arguments of a
    /// call that borrows its receiver (`&self`, `&mut self`) are drawn, no `&mut self` call on that root is drawn.
    pub(super) fn call_parts(
        &mut self,
        f: usize,
        depth: u32,
    ) -> Option<(Option<Place>, Vec<Expr>)> {
        self.call_parts_on(f, depth, None)
    }

    /// `call_parts` with the receiver chosen by the caller (`forced`), or drawn when `None`.
    pub(super) fn call_parts_on(
        &mut self,
        f: usize,
        depth: u32,
        forced: Option<Place>,
    ) -> Option<(Option<Place>, Vec<Expr>)> {
        let def = self.table.fns[f].clone();
        let recv = if def.recv == Recv::None {
            None
        } else if forced.is_some() {
            forced
        } else {
            let places: Vec<Place> = self
                .struct_places()
                .into_iter()
                .filter(|(place, j, writable)| {
                    Some(*j) == def.owner
                        && (def.recv != Recv::Mut
                            || (*writable && !self.forbid.contains(&place.root)))
                })
                .map(|(place, ..)| place)
                .collect();
            if places.is_empty() {
                return None;
            }
            Some(places[self.b.below(places.len() as u32) as usize].clone())
        };
        let pushed = recv
            .as_ref()
            .filter(|_| def.recv != Recv::Value)
            .map(|place| place.root);
        if let Some(root) = pushed {
            self.forbid.push(root);
        }
        let mut args = Vec::new();
        let mut complete = true;
        let mut enum_root: Option<Root> = None;
        // A trait method called on one root often gets calls on another root among its arguments.
        let trait_root = if self.traits && def.imp.is_some() && def.recv != Recv::Value {
            self.pick_writable_root()
        } else {
            None
        };
        for (slot, (_, ty)) in def.params.iter().enumerate() {
            match ty {
                scalar if trait_root.is_some() && self.b.chance(80) => {
                    let root = trait_root.expect("a root");
                    let call = self.root_call(*scalar, root, depth.saturating_sub(1));
                    args.push(call.unwrap_or_else(|| self.expr(*scalar, depth)));
                }
                // A comparison before the last argument: the shape that made the macro emit `__gust_keep` (m9.f1).
                Ty::Bool if self.values && slot + 1 < def.params.len() && self.b.chance(45) => {
                    let operand = self.int_ty();
                    args.push(Expr::bin(
                        BinOp::Lt,
                        self.expr(operand, depth.min(2)),
                        self.expr(operand, depth.min(2)),
                    ));
                }
                Ty::Tuple(_) => args.push(self.tuple_expr(*ty, depth)),
                Ty::Array(a) => args.push(self.array_arg(*a, depth)),
                // An enum argument is often built from a call, and two of them from calls on one root: the arguments
                // run left to right, so the first call's effect on the root is what the second one sees.
                Ty::Enum(_) => args.push(self.enum_arg(*ty, depth, &mut enum_root)),
                // A struct literal is only a `let` initializer or the right side of an assignment (E0810): an
                // argument is a local, a field, or the value of a call.
                Ty::Struct(j) => match self.struct_expr(*j, depth, false) {
                    Some(arg) => args.push(arg),
                    None => complete = false,
                },
                // A by-value call on `R` often gets a `&mut self` call on `R` among its arguments: the receiver is
                // copied before they run.
                scalar if def.recv == Recv::Value && self.b.chance(75) => {
                    let root = recv.as_ref().map(|place| place.root);
                    let call = root
                        .and_then(|root| self.root_call(*scalar, root, depth.saturating_sub(1)));
                    args.push(call.unwrap_or_else(|| self.expr(*scalar, depth)));
                }
                scalar => args.push(self.expr(*scalar, depth)),
            }
        }
        if pushed.is_some() {
            self.forbid.pop();
        }
        complete.then_some((recv, args))
    }

    /// A call whose value has type `ty` (a scalar, or a struct for an associated function), if one is in reach.
    pub(super) fn value_call(&mut self, ty: Ty, depth: u32) -> Option<Expr> {
        if self.traits {
            if self.b.chance(30)
                && let Some(call) = self.generic_call(ty, depth)
            {
                return Some(call);
            }
            if matches!(ty, Ty::Enum(_) | Ty::U32 | Ty::I32 | Ty::Bool)
                && self.b.chance(25)
                && let Some(call) = self.trait_call(ty, depth)
            {
                return Some(call);
            }
        }
        let candidates: Vec<usize> = (0..self.callable())
            .filter(|&k| self.table.fns[k].ret == Some(ty) && self.plain(k))
            .collect();
        if candidates.is_empty() {
            return None;
        }
        // A third of the calls prefer a by-value method, whose receiver is copied before its arguments run.
        let by_value: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|&k| self.table.fns[k].recv == Recv::Value)
            .collect();
        let candidates = if !by_value.is_empty() && self.b.chance(33) {
            by_value
        } else {
            candidates
        };
        // The value space prefers the helpers that return from inside an arm: a `return` in an arm that runs.
        let returning: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|k| self.arm_return_fns.contains(k))
            .collect();
        let candidates = if self.values && !returning.is_empty() && self.b.chance(50) {
            returning
        } else {
            candidates
        };
        let f = candidates[self.b.below(candidates.len() as u32) as usize];
        let owner = {
            let def = &self.table.fns[f];
            if def.recv == Recv::None {
                def.owner
            } else {
                None
            }
        };
        let (recv, args) = self.call_parts(f, depth)?;
        let call = Expr::Call {
            f,
            owner,
            recv,
            args,
            ty,
        };
        Some(if self.traits {
            self.maybe_path(call)
        } else {
            call
        })
    }

    /// A `&mut self` call of a method returning `ty` on a writable place of root `root`, if one is in reach and no
    /// enclosing call is borrowing that root: the call whose effect on `root` a surrounding read or place can see.
    pub(super) fn root_call(&mut self, ty: Ty, root: Root, depth: u32) -> Option<Expr> {
        if depth == 0 || self.forbid.contains(&root) {
            return None;
        }
        // `c.pair().1`: an element of a `&mut self` method's tuple result.
        if self.values
            && self.b.chance(30)
            && let Some(field) = self.tuple_field_call(ty, depth, Some(root))
        {
            return Some(field);
        }
        let places = self.struct_places();
        let on_root = |owner: Option<u8>| {
            places
                .iter()
                .filter(|(place, j, writable)| *writable && place.root == root && Some(*j) == owner)
                .map(|(place, ..)| place.clone())
                .collect::<Vec<Place>>()
        };
        let candidates: Vec<usize> = (0..self.callable())
            .filter(|&k| {
                let def = &self.table.fns[k];
                self.plain(k)
                    && def.recv == Recv::Mut
                    && def.ret == Some(ty)
                    && !on_root(def.owner).is_empty()
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let f = candidates[self.b.below(candidates.len() as u32) as usize];
        let receivers = on_root(self.table.fns[f].owner);
        let recv = receivers[self.b.below(receivers.len() as u32) as usize].clone();
        let (recv, args) = self.call_parts_on(f, depth, Some(recv))?;
        Some(Expr::Call {
            f,
            owner: None,
            recv,
            args,
            ty,
        })
    }

    /// A binary whose operands are a read of a place and a `&mut self` call on its root (in either order, so the
    /// read comes before or after the call), or, for `bool`, a condition reading a place and a `&mut self` call on
    /// its root as the right operand of `&&`/`||`. `None` when there is no such pair in reach.
    pub(super) fn root_binary(&mut self, ty: Ty, depth: u32) -> Option<Expr> {
        let mut readable: Vec<(Place, Ty)> = Vec::new();
        for scalar in [Ty::U32, Ty::I32] {
            readable.extend(
                self.scalar_places(scalar)
                    .into_iter()
                    .map(|(p, _)| (p, scalar)),
            );
        }
        if readable.is_empty() {
            return None;
        }
        let (place, read_ty) = readable[self.b.below(readable.len() as u32) as usize].clone();
        let depth = depth.max(1);
        if ty == Ty::Bool {
            let call = self.root_call(Ty::Bool, place.root, depth)?;
            let bound = self.expr(read_ty, 1);
            let cmp = [BinOp::Lt, BinOp::Ge, BinOp::Ne, BinOp::Eq][self.b.below(4) as usize];
            let read = Expr::Read { place, ty: read_ty };
            let op = [BinOp::And, BinOp::Or][self.b.below(2) as usize];
            return Some(Expr::bin(op, Expr::bin(cmp, read, bound), call));
        }
        if read_ty != ty {
            return None;
        }
        let call = self.root_call(ty, place.root, depth)?;
        let read = Expr::Read { place, ty };
        let op = [
            BinOp::Add,
            BinOp::Add,
            BinOp::Sub,
            BinOp::BitXor,
            BinOp::BitOr,
            BinOp::Mul,
        ][self.b.below(6) as usize];
        Some(if self.b.chance(50) {
            Expr::bin(op, read, call)
        } else {
            Expr::bin(op, call, read)
        })
    }

    /// The type and the initializer of a new struct local: struct `S0` (scalar fields only, so a literal is always
    /// possible) when the random struct has no value in reach.
    pub(super) fn struct_init(&mut self) -> (u8, Expr) {
        let j = self.b.below(self.table.structs.len() as u32) as u8;
        if let Some(init) = self.struct_expr(j, 3, true) {
            return (j, init);
        }
        let init = self
            .struct_expr(0, 3, true)
            .expect("a literal of S0 is always possible");
        (0, init)
    }

    /// A value of struct `j`: a literal (only where `top` says a literal is allowed: a `let` initializer or the right
    /// side of an assignment), a call of an associated function, or a copy of a struct place. `None` when the kind
    /// drawn, and every other, has no value in reach.
    pub(super) fn struct_expr(&mut self, j: u8, depth: u32, top: bool) -> Option<Expr> {
        let ty = Ty::Struct(j);
        if self.closures
            && depth > 0
            && !self.cls.is_empty()
            && self.b.chance(22)
            && let Some(call) = self.closure_call(ty, depth - 1, None)
        {
            return Some(call);
        }
        let bare_self_ok = matches!(self.this, Some((_, Recv::Value)));
        let places: Vec<Place> = self
            .struct_places()
            .into_iter()
            .filter(|(place, pj, _)| {
                *pj == j && !(place.root == Root::SelfV && place.path.is_empty() && !bare_self_ok)
            })
            .map(|(place, ..)| place)
            .collect();
        let assoc = (0..self.callable()).any(|k| self.table.fns[k].ret == Some(ty));
        let weights = [
            if top { 50 } else { 0 },
            if assoc && depth > 0 { 30 } else { 0 },
            if places.is_empty() { 0 } else { 25 },
        ];
        if weights.iter().sum::<u32>() == 0 {
            return None;
        }
        let first = self.b.weighted(&weights);
        // The kind drawn first, then the others in a fixed order if it has no value in reach.
        for kind in [first, 0, 1, 2] {
            if weights[kind] == 0 {
                continue;
            }
            match kind {
                0 => {
                    let fields = self.table.structs[usize::from(j)].fields.clone();
                    let mut values = Vec::new();
                    for field in fields {
                        let value = match field {
                            Ty::Struct(inner) => {
                                self.struct_expr(inner, depth.saturating_sub(1), false)
                            }
                            scalar => Some(self.expr(scalar, depth.min(2))),
                        };
                        match value {
                            Some(value) => values.push(value),
                            None => break,
                        }
                    }
                    if values.len() == self.table.structs[usize::from(j)].fields.len() {
                        // Written in a permuted order a quarter of the time: Rust evaluates the fields as written.
                        let mut order: Vec<u8> = (0..values.len() as u8).collect();
                        if values.len() > 1 && self.b.chance(25) {
                            for k in (1..order.len()).rev() {
                                let other = self.b.below(k as u32 + 1) as usize;
                                order.swap(k, other);
                                values.swap(k, other);
                            }
                        }
                        return Some(Expr::StructLit {
                            ty,
                            fields: values,
                            order,
                        });
                    }
                }
                1 => {
                    if let Some(call) = self.value_call(ty, depth - 1) {
                        return Some(call);
                    }
                }
                _ => {
                    let place = places[self.b.below(places.len() as u32) as usize].clone();
                    return Some(place_value(&place, ty));
                }
            }
        }
        None
    }

    /// The structs (one to three, each of one to three fields, scalars or earlier structs) and three to six functions
    /// (helpers, `&self`, `&mut self` and by-value `self` methods, associated functions), each calling only the
    /// functions below it.
    pub(super) fn gen_tables(&mut self) {
        let structs = 1 + self.b.below(3) as usize;
        for j in 0..structs {
            let count = 1 + self.b.below(3);
            if self.traits && j > 0 && self.b.chance(55) {
                // A copy of the struct before: two types of one shape, so that a wrong impl stays observable.
                let fields = self.table.structs[j - 1].fields.clone();
                self.table.structs.push(StructDef { fields });
                continue;
            }
            let mut fields = Vec::new();
            for _ in 0..count {
                // No `bool` field: a GPU struct is a `GpuPod` type, and `bool` is not one.
                fields.push(
                    match self.b.weighted(&[50, 50, if j > 0 { 25 } else { 0 }]) {
                        0 => Ty::U32,
                        1 => Ty::I32,
                        _ => Ty::Struct(self.b.below(j as u32) as u8),
                    },
                );
            }
            self.table.structs.push(StructDef { fields });
            if self.table.flat_len(j as u8) > MAX_FLAT {
                self.table.structs[j].fields = vec![Ty::U32, Ty::I32];
            }
        }
        if self.enums {
            self.gen_enums();
        }
        if self.traits {
            self.gen_trait_tables();
            return;
        }
        let total = 3 + self.b.below(4) as usize;
        for k in 0..total {
            // The enum space needs a function that returns an enum: the last one is forced to if none is yet.
            self.force_enum_ret = self.enums
                && k + 1 == total
                && !self
                    .table
                    .fns
                    .iter()
                    .any(|def| matches!(def.ret, Some(Ty::Enum(_))));
            self.gen_fn(k);
        }
        self.force_enum_ret = false;
        if self.closures {
            self.counter_methods();
        }
    }

    pub(super) fn gen_fn(&mut self, k: usize) {
        self.gen_fn_with(k, None);
    }

    /// `gen_fn`, or (a trait program, format 13) the struct impl of a trait method with the signature the trait gives:
    /// `forced.limit` is the index below which the body may call (the first impl of the pair).
    pub(super) fn gen_fn_with(&mut self, k: usize, forced: Option<Forced>) {
        let structs = self.table.structs.len() as u32;
        // The first function is often a `&mut self` method returning `bool`: the right operand of an `&&`/`||`.
        let first_bool = forced.is_none() && k == 0 && self.b.chance(35);
        let kind = if forced.is_some() {
            1
        } else if first_bool {
            2
        } else if self.force_enum_ret {
            // Not an associated function: it returns its struct.
            [0, 1, 2, 3][self.b.weighted(&[10, 15, 55, 20])]
        } else {
            self.b.weighted(&[18, 18, 26, 18, 20])
        };
        let recv = forced.as_ref().map_or(
            [Recv::None, Recv::Shared, Recv::Mut, Recv::Value, Recv::None][kind],
            |fo| fo.recv,
        );
        // An associated function builds its struct from a literal, which needs every field to be a scalar.
        let scalar_structs: Vec<u8> = (0..structs as u8)
            .filter(|&j| {
                self.table.structs[usize::from(j)]
                    .fields
                    .iter()
                    .all(|f| !f.is_struct())
            })
            .collect();
        let owner = if let Some(fo) = &forced {
            fo.owner
        } else {
            (kind != 0).then(|| {
                if kind == 4 {
                    scalar_structs[self.b.below(scalar_structs.len() as u32) as usize]
                } else {
                    self.b.below(structs) as u8
                }
            })
        };
        self.cur_fn = Some(forced.as_ref().map_or(k, |fo| fo.limit));
        self.scope.clear();
        self.cls.clear();
        self.this = owner
            .filter(|_| recv != Recv::None)
            .map(|o| (Ty::Struct(o), recv));
        let mut params = Vec::new();
        // A by-value method takes at least one parameter, mostly a `u32`: the slot of a `&mut self` call's value.
        let count = if forced.is_some() {
            0
        } else if self.enums {
            1 + self.b.below(3)
        } else {
            self.b.below(3).max(u32::from(kind == 3))
        };
        let mut prev_enum = false;
        for slot in 0..count {
            let ty = if kind == 3 && slot == 0 && self.b.chance(70) {
                Ty::U32
            } else if self.enums && prev_enum && self.b.chance(60) {
                self.enum_ty()
            } else if self.b.chance(18) {
                Ty::Struct(self.b.below(structs) as u8)
            } else if self.arrays && self.b.chance(18) {
                self.array_param_ty()
            } else if self.values && self.b.chance(14) {
                self.tuple_ty()
            } else if self.enums && self.b.chance(30) {
                self.enum_ty()
            } else {
                [Ty::U32, Ty::I32, Ty::Bool][self.b.weighted(&[40, 40, 20])]
            };
            let var = self.fresh();
            prev_enum = ty.is_enum();
            self.scope.push((var, ty, false));
            params.push((var, ty));
        }
        if let Some(fo) = forced.as_ref() {
            for &ty in &fo.params {
                let var = self.fresh();
                self.scope.push((var, ty, false));
                params.push((var, ty));
            }
        }
        let ret = if let Some(fo) = &forced {
            fo.ret
        } else {
            match kind {
                _ if first_bool => Some(Ty::Bool),
                4 => owner.map(Ty::Struct),
                // A helper that takes an array often returns it (modified): the array in, the array out.
                0 if self.arrays
                    && params.iter().any(|(_, ty)| ty.is_array())
                    && self.b.chance(50) =>
                {
                    params.iter().map(|(_, ty)| *ty).find(|ty| ty.is_array())
                }
                _ if self.force_enum_ret => Some(self.enum_ty()),
                2 if self.b.chance(40) => None,
                _ if self.values && self.b.chance(28) => Some(self.tuple_ty()),
                _ if self.enums && self.b.chance(30) => Some(self.enum_ty()),
                _ => Some([Ty::U32, Ty::I32, Ty::Bool][self.b.weighted(&[40, 40, 20])]),
            }
        };
        // A helper (no receiver, no owner) may end a block in `return`.
        self.cur_ret = if self.enums
            && recv == Recv::None
            && owner.is_none()
            && !ret.is_some_and(Ty::is_array)
        {
            ret
        } else {
            None
        };
        let mut body = Vec::new();
        if recv == Recv::Mut {
            // A `&mut self` method writes the receiver at least once.
            let targets = self.set_targets();
            if !targets.is_empty() {
                let pre_mark = self.pre.len();
                body.push(self.set_stmt(&targets));
                self.splice_pre(pre_mark, 0, &mut body);
            }
        }
        // A helper of the value space often opens with a value `match` whose arms (all but the last) `return`.
        if self.values
            && let Some(ret) = self.cur_ret
            && !ret.is_struct()
            && self.b.chance(70)
        {
            let pre_mark = self.pre.len();
            let at = body.len();
            let forced = self.returning_match(ret);
            body.push(forced);
            self.splice_pre(pre_mark, at, &mut body);
        }
        // A function of the closure space often binds a closure first: the body and the tail then call it.
        if self.closures && self.b.chance(70) {
            let pre_mark = self.pre.len();
            let stmt = self.closure_let(1);
            body.push(stmt);
            self.splice_pre(pre_mark, 0, &mut body);
        }
        body.extend(self.block(1, 0, 3));
        let array_reads = if self.arrays {
            self.array_param_uses(&params, &mut body)
        } else {
            Vec::new()
        };
        let pre_mark = self.pre.len();
        let at = body.len();
        let tail = ret.map(|ty| match ty {
            Ty::Struct(j) => {
                // A struct literal is no tail (E0810): bind it first.
                let init = self
                    .struct_expr(j, 3, true)
                    .expect("a literal of a scalar-field struct is always possible");
                let var = self.fresh();
                body.push(Stmt::Let {
                    var,
                    ty,
                    mutable: false,
                    init,
                });
                Expr::Var(var, ty)
            }
            Ty::Array(_) => {
                let (var, _) = *params
                    .iter()
                    .find(|(_, p)| *p == ty)
                    .expect("an array result has its array parameter");
                Expr::Var(var, ty)
            }
            scalar => {
                let mut value = self.expr(scalar, 4);
                for read in array_reads
                    .clone()
                    .into_iter()
                    .filter(|_| matches!(scalar, Ty::U32 | Ty::I32 | Ty::Bool))
                {
                    value = fold_in(value, read);
                }
                if self.closures
                    && !self.cls.is_empty()
                    && matches!(scalar, Ty::U32 | Ty::I32 | Ty::Bool)
                    && self.b.chance(85)
                    && let Some(call) = self.closure_call(scalar, 3, None)
                {
                    value = fold_in(value, call);
                }
                value
            }
        });
        // A trait impl weighs its integer parameters into an integer result, so that the order of two arguments shows.
        let tail = if forced.is_some() {
            tail.map(|t| {
                let t = self.weight_params(t, &params);
                match owner {
                    Some(j) if recv != Recv::None => self.weight_self(t, j),
                    _ => t,
                }
            })
        } else {
            tail
        };
        // Lets that the tail hoisted come before it (and before the `let` of a struct tail).
        self.splice_pre(pre_mark, at, &mut body);
        if self.values && arm_returns(&body, false) {
            self.arm_return_fns.push(k);
        }
        self.table.fns.push(FnDef {
            imp: forced.as_ref().map(|fo| fo.imp),
            gens: Vec::new(),
            owner,
            recv,
            params,
            ret,
            body,
            tail,
        });
        self.cur_fn = None;
        self.cur_ret = None;
        self.this = None;
        self.scope.clear();
        self.cls.clear();
    }
}
