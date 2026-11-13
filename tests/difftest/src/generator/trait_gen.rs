//! The trait space (DIFFTEST_FORMAT 13, slice m9.d2): module traits with two impls per method, calls of the impls, and
//! generic helpers bounded by the traits.
//!
//! The refused forms (SEMANTICS sections 13 and 14, E0123 to E0126, E0218, E0328) are not written, by construction:
//!
//! - every trait method has a receiver and scalar parameters; a `&mut self` method is implemented by structs only
//!   (E0124), an enum impl takes `self` by value; no default body, no generic trait method, no associated item;
//! - names: traits `Tr<n>`, methods `t<n>m<k>`, type parameters `T` and `U`, none of which is `m<k>`, `S<j>`, `E<j>`,
//!   `f<k>`, `v<n>`, `cl<n>`, a Slang keyword, a built-in id type or a `__gust_` name; two traits never share a method
//!   name, so no type has two methods of one name;
//! - the two impls of a trait method are adjacent in the function table and both are drawn with the index of the first
//!   as their call limit, so the call graph (by name, D162) stays acyclic: every caller of the name lies above both;
//! - every type parameter has one or two module-trait bounds (`Copy` may join, never alone), inline or in a `where`
//!   clause, and is determined by a parameter of type `T`; no turbofish, no std bound, no arithmetic on `T`, no
//!   recursion, no `T` used by value twice without `Copy`;
//! - the space has no array, so no trait or generic body holds one and the D168/D169 hole guard has nothing to guard;
//!   no barrier, atomic or subgroup operation is generated in the space at all.

use super::{Forced, Gen, fold_in};
use crate::ast::{
    Arm, BinOp, Expr, FnDef, GenParam, Imp, PathForm, Place, Recv, Root, Stmt, TRet, TraitDef,
    TraitMethod, Ty,
};

const SCALARS: [Ty; 3] = [Ty::U32, Ty::I32, Ty::Bool];

impl Gen<'_> {
    fn scalar_pick(&mut self) -> Ty {
        SCALARS[self.b.weighted(&[40, 40, 20])]
    }

    /// The struct types (scalar-field ones first in the flag) and the enum types of the module.
    fn impl_candidates(&self) -> Vec<Ty> {
        let mut out: Vec<Ty> = (0..self.table.structs.len() as u8)
            .map(Ty::Struct)
            .collect();
        out.extend((0..self.table.enums.len() as u8).map(Ty::Enum));
        out
    }

    fn scalar_fielded(&self, ty: Ty) -> bool {
        match ty {
            Ty::Struct(j) => self.table.structs[usize::from(j)]
                .fields
                .iter()
                .all(|f| !f.is_struct()),
            _ => true,
        }
    }

    /// The traits, their impls, the helpers that call the impls and the generic helpers (after the structs and enums).
    pub(super) fn gen_trait_tables(&mut self) {
        // Counter methods first: every later function may call them (the order-sensitive shapes need a `&mut self` method).
        self.counter_methods();
        for _ in 0..1 + self.b.below(2) {
            let k = self.table.fns.len();
            self.gen_fn(k);
        }
        // The traits: two implementing types each (a later trait often reuses a pair, so that two bounds can hold).
        let candidates = self.impl_candidates();
        let n_traits = 1 + self.b.below(3) as usize;
        let mut pairs: Vec<(Ty, Ty)> = Vec::new();
        for _ in 0..n_traits {
            let pair = if !pairs.is_empty() && self.b.chance(45) {
                pairs[self.b.below(pairs.len() as u32) as usize]
            } else {
                let a = candidates[self.b.below(candidates.len() as u32) as usize];
                let rest: Vec<Ty> = candidates.iter().copied().filter(|t| *t != a).collect();
                (a, rest[self.b.below(rest.len() as u32) as usize])
            };
            pairs.push(pair);
            let all_structs = matches!(pair.0, Ty::Struct(_)) && matches!(pair.1, Ty::Struct(_));
            let self_ok = self.scalar_fielded(pair.0) && self.scalar_fielded(pair.1);
            let n_methods = 1 + self.b.below(2) as usize;
            let mut methods = Vec::new();
            for m in 0..n_methods {
                let recv = if all_structs {
                    [Recv::Shared, Recv::Mut, Recv::Value][self.b.weighted(&[35, 30, 35])]
                } else {
                    Recv::Value
                };
                let n_params = [0, 1, 2][self.b.weighted(&[5, 25, 70])];
                let params: Vec<Ty> = (0..n_params).map(|_| self.scalar_pick()).collect();
                let ret = match recv {
                    Recv::Mut if self.b.chance(40) => TRet::Unit,
                    Recv::Value if self_ok && self.b.chance(35) => TRet::SelfTy,
                    _ => TRet::Ty(self.scalar_pick()),
                };
                // Method 0 of a trait returns a scalar most of the time: a generic body then has something to read.
                let ret = if m == 0 && ret == TRet::Unit && self.b.chance(60) {
                    TRet::Ty(self.scalar_pick())
                } else {
                    ret
                };
                methods.push(TraitMethod { recv, params, ret });
            }
            self.table.traits.push(TraitDef { methods });
        }
        // The impls: for each trait method, the two types' functions side by side.
        for (tr, &(a, b)) in pairs.iter().enumerate() {
            for m in 0..self.table.traits[tr].methods.len() {
                let limit = self.table.fns.len();
                for ty in [a, b] {
                    self.gen_impl(tr as u8, m as u8, ty, limit);
                }
            }
        }
        // Helpers above the impls call them.
        for _ in 0..1 + self.b.below(2) {
            let k = self.table.fns.len();
            self.gen_fn(k);
        }
        // Generic helpers and methods, each calling the functions below it; the first is a helper with an integer result.
        for n in 0..1 + self.b.below(3) {
            let k = self.table.fns.len();
            self.gen_generic_fn(k, n == 0);
        }
        // A helper above the generic ones calls them with concrete types.
        if self.b.chance(70) {
            let k = self.table.fns.len();
            self.gen_fn(k);
        }
    }

    /// `value ^ (p & 255) * (2k + 1)` for every integer parameter `p` (the `k`th), when `value` is an integer.
    pub(super) fn weight_params(&mut self, mut value: Expr, params: &[(u32, Ty)]) -> Expr {
        let ty = value.ty();
        if !ty.is_int() {
            return value;
        }
        for (k, &(var, pty)) in params.iter().enumerate() {
            if !pty.is_int() {
                continue;
            }
            let read = if pty == ty {
                Expr::Var(var, pty)
            } else {
                Expr::cast(ty, Expr::Var(var, pty))
            };
            let masked = Expr::bin(BinOp::BitAnd, read, Expr::Lit(ty, 255));
            let term = Expr::bin(BinOp::Mul, masked, Expr::Lit(ty, 2 * k as u32 + 1));
            value = Expr::bin(BinOp::BitXor, value, term);
        }
        value
    }

    /// `value ^ (self.f & 255) * (2k + 1)` for every integer field `f` (the `k`th) of the receiver's struct `j`, when
    /// `value` is an integer: a by-value `self` that an argument's call changes later shows in the result.
    pub(super) fn weight_self(&mut self, mut value: Expr, j: u8) -> Expr {
        let ty = value.ty();
        if !ty.is_int() {
            return value;
        }
        let this = Ty::Struct(j);
        let fields = self.table.structs[usize::from(j)].fields.clone();
        for (k, &fty) in fields.iter().enumerate() {
            if !fty.is_int() {
                continue;
            }
            let read = Expr::Read {
                place: Place::this(this).field(k as u8),
                ty: fty,
            };
            let read = if fty == ty {
                read
            } else {
                Expr::cast(ty, read)
            };
            let masked = Expr::bin(BinOp::BitAnd, read, Expr::Lit(ty, 255));
            let term = Expr::bin(BinOp::Mul, masked, Expr::Lit(ty, 2 * k as u32 + 1));
            value = Expr::bin(BinOp::BitXor, value, term);
        }
        value
    }

    /// The impl of trait method `(tr, m)` for `ty`, drawn with the index `limit` as the bound on what it may call.
    fn gen_impl(&mut self, tr: u8, m: u8, ty: Ty, limit: usize) {
        let method = self.table.traits[usize::from(tr)].methods[usize::from(m)].clone();
        let ret = match method.ret {
            TRet::Unit => None,
            TRet::Ty(t) => Some(t),
            TRet::SelfTy => Some(ty),
        };
        let imp = Imp { tr, m, ty };
        match ty {
            Ty::Struct(j) => {
                let forced = Forced {
                    owner: Some(j),
                    recv: method.recv,
                    ret,
                    imp,
                    limit,
                    params: method.params.clone(),
                };
                let k = self.table.fns.len();
                self.gen_fn_with(k, Some(forced));
            }
            Ty::Enum(e) => self.gen_enum_impl(imp, e, &method, ret, limit),
            _ => unreachable!("an impl for a struct or an enum"),
        }
    }

    /// An enum impl: `self` by value, read as a value (`let v: E = self;`, `self as u32`, a `match` on it).
    fn gen_enum_impl(
        &mut self,
        imp: Imp,
        e: u8,
        method: &TraitMethod,
        ret: Option<Ty>,
        limit: usize,
    ) {
        let ety = Ty::Enum(e);
        self.cur_fn = Some(limit);
        self.scope.clear();
        self.cls.clear();
        self.this = None;
        self.cur_ret = None;
        let mut params = Vec::new();
        for &ty in &method.params {
            let var = self.fresh();
            self.scope.push((var, ty, false));
            params.push((var, ty));
        }
        let me = Expr::Read {
            place: Place::this(ety),
            ty: ety,
        };
        let repr = self.table.enums[usize::from(e)].repr;
        let mut body = Vec::new();
        let tail = ret.map(|ty| {
            if ty == ety {
                // `Self`: a value `match` on the receiver (or a parameter-free copy of it), arms yielding variants.
                let v = self.fresh();
                body.push(Stmt::Let {
                    var: v,
                    ty: ety,
                    mutable: false,
                    init: me.clone(),
                });
                let r = self.fresh();
                let n = self.table.enums[usize::from(e)].variants.len() as u32;
                let pats = self.enum_patterns(e);
                let mut arms = Vec::new();
                for pat in pats {
                    let variant = self.b.below(n) as u8;
                    arms.push(Arm {
                        pat,
                        body: vec![Stmt::Yield {
                            value: Expr::Variant { e, v: variant },
                        }],
                    });
                }
                body.push(Stmt::LetMatch {
                    var: r,
                    ty: ety,
                    mutable: false,
                    scrut: Expr::Var(v, ety),
                    arms,
                });
                Expr::Var(r, ety)
            } else {
                let as_repr = Expr::EnumAs {
                    to: repr,
                    e: Box::new(me.clone()),
                };
                let base = self.expr(ty, 3);
                let mut value = fold_in(base, as_repr);
                if self.b.chance(50) {
                    // The receiver is read again through a bound local: a copy of the same value.
                    let v = self.fresh();
                    body.insert(
                        0,
                        Stmt::Let {
                            var: v,
                            ty: ety,
                            mutable: false,
                            init: me.clone(),
                        },
                    );
                    let again = Expr::EnumAs {
                        to: repr,
                        e: Box::new(Expr::Var(v, ety)),
                    };
                    value = fold_in(value, again);
                }
                value
            }
        });
        let tail = tail.map(|t| self.weight_params(t, &params));
        self.table.fns.push(FnDef {
            imp: Some(imp),
            gens: Vec::new(),
            owner: None,
            recv: method.recv,
            params,
            ret,
            body,
            tail,
        });
        self.cur_fn = None;
        self.scope.clear();
    }

    /// One local of every implementing enum type and of every implementing struct type, so that the kernel can call.
    pub(super) fn trait_locals(&mut self, body: &mut Vec<Stmt>) {
        let mut want: Vec<Ty> = Vec::new();
        for tr in 0..self.table.traits.len() {
            for ty in self.table.implementors(tr as u8) {
                if !want.contains(&ty) {
                    want.push(ty);
                }
            }
        }
        for ty in want {
            match ty {
                Ty::Enum(e) => {
                    let n = self.table.enums[usize::from(e)].variants.len() as u32;
                    let v = self.b.below(n) as u8;
                    let var = self.fresh();
                    self.scope.push((var, ty, false));
                    body.push(Stmt::Let {
                        var,
                        ty,
                        mutable: false,
                        init: Expr::Variant { e, v },
                    });
                }
                Ty::Struct(j) => {
                    let has = self.scope.iter().filter(|(_, t, _)| *t == ty).count();
                    if has >= 1 && self.b.chance(40) {
                        continue;
                    }
                    let (pre_mark, at) = (self.pre.len(), body.len());
                    if let Some(init) = self.struct_expr(j, 3, true) {
                        let var = self.fresh();
                        let mutable = self.b.chance(85);
                        self.scope.push((var, ty, mutable));
                        body.push(Stmt::Let {
                            var,
                            ty,
                            mutable,
                            init,
                        });
                        self.splice_pre(pre_mark, at, body);
                    }
                }
                _ => {}
            }
        }
    }

    /// Calls that join the kernel's result (format 13), all integer-valued and joined with `^`, so that every path
    /// evaluates them: one to three trait-impl calls on locals, one generic helper called once with each of two
    /// different implementing types (a wrong impl, a reused specialisation or a reversed argument order reaches the
    /// output), and up to two more generic calls.
    pub(super) fn trait_probes(&mut self) -> Vec<Expr> {
        let mut out = Vec::new();
        let impls: Vec<usize> = (0..self.table.fns.len())
            .filter(|&k| {
                let def = &self.table.fns[k];
                def.imp.is_some() && def.ret.is_some_and(Ty::is_int)
            })
            .collect();
        for n in 0..3 {
            if impls.is_empty() || (n > 0 && !self.b.chance(70)) {
                continue;
            }
            // The first probe is not optional: a trait program calls an impl where it counts.
            for _ in 0..if n == 0 { 4 } else { 1 } {
                if let Some(call) = self.impl_probe(&impls) {
                    out.push(call);
                    break;
                }
            }
        }
        // One helper with two instantiations of its first type parameter (M3 of the m9.d2 review).
        let helpers: Vec<usize> = (0..self.table.fns.len())
            .filter(|&k| {
                let def = &self.table.fns[k];
                !def.gens.is_empty() && def.recv == Recv::None && def.ret.is_some_and(Ty::is_int)
            })
            .collect();
        if !helpers.is_empty() {
            let f = helpers[self.b.below(helpers.len() as u32) as usize];
            let rty = self.table.fns[f].ret.expect("an integer result");
            let bounds = self.table.fns[f].gens[0].bounds.clone();
            let types = self.instantiations(&bounds);
            if types.len() >= 2 {
                let a = self.b.below(types.len() as u32) as usize;
                let b = (a + 1 + self.b.below(types.len() as u32 - 1) as usize) % types.len();
                for t in [types[a], types[b]] {
                    if let Some(call) = self.generic_call_f(f, rty, 3, Some(t)) {
                        out.push(call);
                    }
                }
            }
        }
        for _ in 0..2 {
            let ty = [Ty::U32, Ty::I32][self.b.weighted(&[50, 50])];
            if self.b.chance(60)
                && let Some(call) = self.generic_call(ty, 3)
            {
                out.push(call);
            }
        }
        out
    }

    /// A call of one of the integer-valued `impls` on a local (an enum local, or a struct place `call_parts` draws).
    fn impl_probe(&mut self, impls: &[usize]) -> Option<Expr> {
        let f = impls[self.b.below(impls.len() as u32) as usize];
        let def = self.table.fns[f].clone();
        let imp = def.imp.expect("an impl");
        let place = match imp.ty {
            Ty::Enum(_) => {
                let vars = self.vars(imp.ty);
                if vars.is_empty() {
                    return None;
                }
                Some(Place::var(
                    vars[self.b.below(vars.len() as u32) as usize],
                    imp.ty,
                ))
            }
            _ => None,
        };
        let (recv, args) = match place {
            Some(place) => {
                let args = def.params.iter().map(|&(_, t)| self.expr(t, 2)).collect();
                (Some(place), args)
            }
            None => self.call_parts(f, 3)?,
        };
        let call = Expr::Call {
            f,
            owner: None,
            recv,
            args,
            ty: def.ret.expect("a result"),
        };
        Some(self.maybe_path(call))
    }

    /// A trait-method call on an enum local whose value has type `ty`, sometimes written as a path call.
    pub(super) fn trait_call(&mut self, ty: Ty, depth: u32) -> Option<Expr> {
        let candidates: Vec<usize> = (0..self.callable())
            .filter(|&k| {
                let def = &self.table.fns[k];
                def.imp.is_some_and(|imp| matches!(imp.ty, Ty::Enum(_)))
                    && def.ret == Some(ty)
                    && !self.vars(def.imp.expect("an impl").ty).is_empty()
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let f = candidates[self.b.below(candidates.len() as u32) as usize];
        let def = self.table.fns[f].clone();
        let imp = def.imp.expect("an impl");
        let vars = self.vars(imp.ty);
        let var = vars[self.b.below(vars.len() as u32) as usize];
        let mut args = Vec::new();
        for (_, pty) in &def.params {
            args.push(self.expr(*pty, depth.saturating_sub(1).min(2)));
        }
        let call = Expr::Call {
            f,
            owner: None,
            recv: Some(Place::var(var, imp.ty)),
            args,
            ty,
        };
        Some(self.maybe_path(call))
    }

    /// A call of a trait-impl method, written as a path call (`Tr0::m(&s, a)`, `<S as Tr0>::m(&s, a)`) 40 % of the time.
    pub(super) fn maybe_path(&mut self, call: Expr) -> Expr {
        let Expr::Call {
            f,
            recv: Some(place),
            ..
        } = &call
        else {
            return call;
        };
        // `Tr::m(&mut v, args)` does not get the two-phase borrow of `v.m(args)` (an argument may read `v`), and a bare `self`
        // is a reference in a method: only a shared or by-value receiver that is not `self` itself is written as a path.
        let bare_self = place.root == Root::SelfV && place.path.is_empty();
        if self.table.fns[*f].imp.is_none()
            || self.table.fns[*f].recv == Recv::Mut
            || bare_self
            || place.index.is_some()
            || !self.b.chance(40)
        {
            return call;
        }
        let form = if self.b.chance(50) {
            PathForm::Trait
        } else {
            PathForm::Qualified
        };
        Expr::PathCall {
            form,
            call: Box::new(call),
        }
    }

    /// The types that implement every trait of `bounds`.
    fn instantiations(&self, bounds: &[u8]) -> Vec<Ty> {
        self.impl_candidates()
            .into_iter()
            .filter(|&t| bounds.iter().all(|&b| self.table.implements(t, b)))
            .collect()
    }

    /// A call of a generic helper or method whose value has type `ty`: the type parameters bound to different module
    /// types over the program, a `T` argument often a struct place whose root a later argument mutates.
    pub(super) fn generic_call(&mut self, ty: Ty, depth: u32) -> Option<Expr> {
        if depth == 0 {
            return None;
        }
        let candidates: Vec<usize> = (0..self.callable())
            .filter(|&k| {
                let def = &self.table.fns[k];
                if def.gens.is_empty() {
                    return false;
                }
                match def.ret {
                    Some(Ty::Param(g)) => {
                        matches!(ty, Ty::Struct(_) | Ty::Enum(_))
                            && def.gens[usize::from(g)]
                                .bounds
                                .iter()
                                .all(|&b| self.table.implements(ty, b))
                    }
                    Some(r) => r == ty,
                    None => false,
                }
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let f = candidates[self.b.below(candidates.len() as u32) as usize];
        self.generic_call_f(f, ty, depth, None)
    }

    /// A call of the generic function `f` (a helper, or a method of a struct) whose value has type `ty`; `first`
    /// forces the type bound to its first type parameter.
    pub(super) fn generic_call_f(
        &mut self,
        f: usize,
        ty: Ty,
        depth: u32,
        first: Option<Ty>,
    ) -> Option<Expr> {
        if depth == 0 || f >= self.callable() {
            return None;
        }
        let def = self.table.fns[f].clone();
        // A method of a struct is called on a place of that struct, whose root the arguments leave alone.
        let recv = if def.recv == Recv::None {
            None
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
        let pushed = recv.as_ref().map(|place| place.root);
        if let Some(root) = pushed {
            self.forbid.push(root);
        }
        let parts = self.generic_args(f, &def, ty, depth, first);
        if pushed.is_some() {
            self.forbid.pop();
        }
        let (args, rty) = parts?;
        Some(Expr::Call {
            f,
            owner: None,
            recv,
            args,
            ty: rty,
        })
    }

    /// The arguments of a call of the generic `def` (function `f`) and the type of its value.
    fn generic_args(
        &mut self,
        f: usize,
        def: &FnDef,
        ty: Ty,
        depth: u32,
        first: Option<Ty>,
    ) -> Option<(Vec<Expr>, Ty)> {
        // Bind each type parameter; one used by the result is bound to the result type.
        let mut inst: Vec<Ty> = Vec::new();
        for (g, gp) in def.gens.iter().enumerate() {
            if g == 0
                && let Some(t) = first
            {
                inst.push(t);
                continue;
            }
            if def.ret == Some(Ty::Param(g as u8)) {
                inst.push(ty);
                continue;
            }
            let options = self.instantiations(&gp.bounds);
            // Prefer a type this function has not been instantiated with yet.
            let fresh: Vec<Ty> = options
                .iter()
                .copied()
                .filter(|t| !self.inst_used.contains(&(f, *t)))
                .collect();
            let pool = if !fresh.is_empty() && self.b.chance(75) {
                fresh
            } else {
                options
            };
            if pool.is_empty() {
                return None;
            }
            inst.push(pool[self.b.below(pool.len() as u32) as usize]);
        }
        let mut bias_root: Option<Root> = None;
        let mut enum_root: Option<Root> = None;
        let mut scalar_root: Option<Root> = None;
        let mut args = Vec::new();
        for (_, pty) in &def.params {
            match *pty {
                Ty::Param(g) => match inst[usize::from(g)] {
                    t @ Ty::Struct(j) => {
                        let places: Vec<Place> = self
                            .struct_places()
                            .into_iter()
                            .filter(|(place, pj, writable)| {
                                *pj == j
                                    && *writable
                                    && !self.forbid.contains(&place.root)
                                    && !(place.root == Root::SelfV && place.path.is_empty())
                            })
                            .map(|(place, ..)| place)
                            .collect();
                        if !places.is_empty() && self.b.chance(60) {
                            let place = places[self.b.below(places.len() as u32) as usize].clone();
                            if bias_root.is_none() {
                                bias_root = Some(place.root);
                            }
                            args.push(super::place_value(&place, t));
                        } else {
                            args.push(self.struct_expr(j, depth - 1, false)?);
                        }
                    }
                    t @ Ty::Enum(_) => args.push(self.enum_arg(t, depth, &mut enum_root)),
                    _ => return None,
                },
                scalar => {
                    // Two integer arguments are often calls on one root: the order they run in shows.
                    if bias_root.is_none() && scalar_root.is_none() && self.b.chance(60) {
                        scalar_root = self.pick_writable_root();
                    }
                    let root = bias_root.or(scalar_root).filter(|_| self.b.chance(75));
                    let call =
                        root.and_then(|r| self.root_call(scalar, r, depth.saturating_sub(1)));
                    args.push(call.unwrap_or_else(|| self.expr(scalar, depth - 1)));
                }
            }
        }
        for t in &inst {
            self.inst_used.push((f, *t));
        }
        let rty = match def.ret {
            Some(Ty::Param(g)) => inst[usize::from(g)],
            Some(other) => other,
            None => return None,
        };
        Some((args, rty))
    }

    /// The impl function of trait method `(tr, m)`, either of the two (the call node names one; the interpreter
    /// dispatches on the receiver).
    fn any_impl(&mut self, tr: u8, m: u8) -> usize {
        let types = self.table.implementors(tr);
        let ty = types[self.b.below(types.len() as u32) as usize];
        self.table
            .impl_fn(tr, m, ty)
            .expect("both impls of a trait method exist")
    }

    /// A generic helper `fn gK<T: Tr0 + Tr1, U: Tr2>(t: T, u: U, a: u32) -> R` (format 13), or a generic method of a
    /// struct (`&self` or `&mut self`): the parameters are bound to copies `let mut c: T = t;`, and the body calls the
    /// bound traits' methods through them, reads a scalar through every `T`, and returns an integer or a `T`. An
    /// integer result folds in every `T` read, every scalar `let` and the integer parameters (weighted), so that the
    /// order of the arguments of a call, a `&mut self` call through `T` and a copy taken before a later argument all
    /// reach the result. `plain_int` makes a helper (no receiver) with an integer result: the call that every trait
    /// program makes with two different types.
    fn gen_generic_fn(&mut self, k: usize, plain_int: bool) {
        let n_traits = self.table.traits.len();
        let ngen = if self.b.chance(35) { 2 } else { 1 };
        let mut gens: Vec<GenParam> = Vec::new();
        for _ in 0..ngen {
            let t0 = self.b.below(n_traits as u32) as u8;
            let mut bounds = vec![t0];
            if n_traits > 1 && self.b.chance(50) {
                let t1 = (t0 + 1 + self.b.below(n_traits as u32 - 1) as u8) % n_traits as u8;
                if self.instantiations(&[t0, t1]).len() >= 2 {
                    bounds.push(t1);
                }
            }
            gens.push(GenParam {
                bounds,
                copy: self.b.chance(50),
                where_clause: self.b.chance(40),
            });
        }
        // A method of a struct (never an enum: an enum method is not generated, see DIFFTEST.md).
        let method: Option<(u8, Recv)> = if !plain_int && self.b.chance(40) {
            let j = self.b.below(self.table.structs.len() as u32) as u8;
            let recv = if self.b.chance(45) {
                Recv::Mut
            } else {
                Recv::Shared
            };
            Some((j, recv))
        } else {
            None
        };
        self.cur_fn = Some(k);
        self.scope.clear();
        self.cls.clear();
        self.this = method.map(|(j, recv)| (Ty::Struct(j), recv));
        self.cur_ret = None;
        self.cur_gens = gens.clone();
        let nlast = gens.len();
        let returns_t = !plain_int && method.is_none() && self.b.chance(35);
        let g_ret = if returns_t {
            self.b.below(nlast as u32) as usize
        } else {
            0
        };
        // Whether the body reads the `T` of parameter `g` through an integer method (the last `T` may be consumed).
        let reads: Vec<bool> = (0..nlast)
            .map(|g| {
                !self
                    .read_methods(&gens[g], gens[g].copy || g + 1 == nlast)
                    .is_empty()
            })
            .collect();
        // Parameters: one `T`-typed parameter per type parameter, scalars mixed in.
        let mut params: Vec<(u32, Ty)> = Vec::new();
        let mut scalars: Vec<(u32, Ty)> = Vec::new();
        let mut t_vars = Vec::new();
        for g in 0..gens.len() {
            if self.b.chance(40) {
                let ty = self.scalar_pick();
                let var = self.fresh();
                self.scope.push((var, ty, false));
                params.push((var, ty));
                scalars.push((var, ty));
            }
            let var = self.fresh();
            params.push((var, Ty::Param(g as u8)));
            t_vars.push(var);
        }
        if self.b.chance(60) {
            let ty = self.scalar_pick();
            let var = self.fresh();
            self.scope.push((var, ty, false));
            params.push((var, ty));
            scalars.push((var, ty));
        }
        // Two integer parameters are likely: the order their arguments run in shows in the result.
        for _ in 0..2 {
            if scalars.iter().filter(|(_, t)| t.is_int()).count() < 2 && self.b.chance(65) {
                let ty = [Ty::U32, Ty::I32][self.b.weighted(&[50, 50])];
                let var = self.fresh();
                self.scope.push((var, ty, false));
                params.push((var, ty));
                scalars.push((var, ty));
            }
        }
        // `let mut c: T = t;` for every type parameter.
        let mut body = Vec::new();
        let mut c_vars = Vec::new();
        for (g, &t_var) in t_vars.iter().enumerate() {
            let c = self.fresh();
            let pty = Ty::Param(g as u8);
            body.push(Stmt::Let {
                var: c,
                ty: pty,
                mutable: true,
                init: Expr::Var(t_var, pty),
            });
            c_vars.push(c);
        }
        // The integer results of the statements below, folded into an integer result.
        let mut lets: Vec<(u32, Ty)> = Vec::new();
        // Statements: calls through `T` (a by-value method needs `Copy`). A `T` is changed only when the body reads it
        // afterwards (or returns it).
        for _ in 0..1 + self.b.below(3) {
            let g = self.b.below(gens.len() as u32) as usize;
            if (returns_t && g != g_ret) || (!returns_t && !reads[g]) {
                continue;
            }
            let pty = Ty::Param(g as u8);
            let place = Place::var(c_vars[g], pty);
            let mut methods = self.bound_methods(&gens[g], gens[g].copy);
            if methods.is_empty() {
                continue;
            }
            // Half the statements are `&mut self` calls that return nothing: the `T` changes, the body reads it later.
            let changing: Vec<(u8, u8)> = methods
                .iter()
                .copied()
                .filter(|&(tr, m)| {
                    let method = &self.table.traits[usize::from(tr)].methods[usize::from(m)];
                    method.recv == Recv::Mut && method.ret == TRet::Unit
                })
                .collect();
            if !changing.is_empty() && self.b.chance(50) {
                methods = changing;
            }
            let (tr, m) = methods[self.b.below(methods.len() as u32) as usize];
            let method = self.table.traits[usize::from(tr)].methods[usize::from(m)].clone();
            let f = self.any_impl(tr, m);
            let mut args = Vec::new();
            for (slot, &p) in method.params.iter().enumerate() {
                // Under a by-value method (the `T` is copied first) an argument calls a `&mut self` method on the same
                // `T`: the copy was taken before.
                if method.recv == Recv::Value
                    && gens[g].copy
                    && slot == 0
                    && self.b.chance(60)
                    && let Some(call) = self.param_mut_call(&gens[g], c_vars[g], g, p)
                {
                    args.push(call);
                    continue;
                }
                args.push(self.self_arg(p));
            }
            match method.ret {
                TRet::Unit => body.push(Stmt::Call {
                    f,
                    owner: None,
                    recv: Some(place),
                    args,
                }),
                TRet::Ty(t) => {
                    let var = self.fresh();
                    body.push(Stmt::Let {
                        var,
                        ty: t,
                        mutable: false,
                        init: Expr::Call {
                            f,
                            owner: None,
                            recv: Some(place),
                            args,
                            ty: t,
                        },
                    });
                    self.scope.push((var, t, false));
                    if t.is_int() {
                        lets.push((var, t));
                    }
                }
                // A `Self` result consumes the `T`: only in the tail.
                TRet::SelfTy => {}
            }
        }
        // A call of a lower generic helper with one of the `T`s (it needs `Copy`: the callee takes the value).
        if !returns_t && self.b.chance(35) {
            let g = self.b.below(gens.len() as u32) as usize;
            if gens[g].copy {
                let pty = Ty::Param(g as u8);
                let callees: Vec<usize> = (0..k)
                    .filter(|&f| {
                        let def = &self.table.fns[f];
                        def.gens.len() == 1
                            && def.recv == Recv::None
                            && def.ret.is_some_and(|r| r != Ty::Param(0))
                            && def.gens[0]
                                .bounds
                                .iter()
                                .all(|b| gens[g].bounds.contains(b))
                    })
                    .collect();
                if !callees.is_empty() {
                    let f = callees[self.b.below(callees.len() as u32) as usize];
                    let def = self.table.fns[f].clone();
                    let mut args = Vec::new();
                    for (_, p) in &def.params {
                        args.push(match *p {
                            Ty::Param(_) => Expr::Var(c_vars[g], pty),
                            scalar => self.expr(scalar, 2),
                        });
                    }
                    let rty = def.ret.expect("a scalar result");
                    let var = self.fresh();
                    body.push(Stmt::Let {
                        var,
                        ty: rty,
                        mutable: false,
                        init: Expr::Call {
                            f,
                            owner: None,
                            recv: None,
                            args,
                            ty: rty,
                        },
                    });
                    self.scope.push((var, rty, false));
                    if rty.is_int() {
                        lets.push((var, rty));
                    }
                }
            }
        }
        // The result: an integer, or a `T`.
        let (ret, tail) = if returns_t {
            let g = g_ret;
            let pty = Ty::Param(g as u8);
            let selfs: Vec<(u8, u8)> = self
                .bound_methods(&gens[g], true)
                .into_iter()
                .filter(|&(tr, m)| {
                    self.table.traits[usize::from(tr)].methods[usize::from(m)].ret == TRet::SelfTy
                })
                .collect();
            let tail = if !selfs.is_empty() && self.b.chance(70) {
                let (tr, m) = selfs[self.b.below(selfs.len() as u32) as usize];
                let method = self.table.traits[usize::from(tr)].methods[usize::from(m)].clone();
                let f = self.any_impl(tr, m);
                let args: Vec<Expr> = method.params.iter().map(|&p| self.expr(p, 2)).collect();
                Expr::Call {
                    f,
                    owner: None,
                    recv: Some(Place::var(c_vars[g], pty)),
                    args,
                    ty: pty,
                }
            } else {
                Expr::Var(c_vars[g], pty)
            };
            (Some(pty), tail)
        } else {
            let ty = [Ty::U32, Ty::I32][self.b.weighted(&[50, 50])];
            let mut value = self.expr(ty, 3);
            // Method results read through every `T`.
            for g in 0..gens.len() {
                let methods = self.read_methods(&gens[g], gens[g].copy || g + 1 == nlast);
                if methods.is_empty() {
                    continue;
                }
                let (tr, m) = methods[self.b.below(methods.len() as u32) as usize];
                let method = self.table.traits[usize::from(tr)].methods[usize::from(m)].clone();
                let TRet::Ty(rt) = method.ret else { continue };
                let f = self.any_impl(tr, m);
                let args: Vec<Expr> = method.params.iter().map(|&p| self.self_arg(p)).collect();
                let call = Expr::Call {
                    f,
                    owner: None,
                    recv: Some(Place::var(c_vars[g], Ty::Param(g as u8))),
                    args,
                    ty: rt,
                };
                value = fold_in(value, call);
            }
            for &(var, t) in &lets {
                value = fold_in(value, Expr::Var(var, t));
            }
            let mut value = self.weight_params(value, &scalars);
            if let Some((j, _)) = method {
                value = self.weight_self(value, j);
            }
            (Some(ty), value)
        };
        self.table.fns.push(FnDef {
            imp: None,
            gens,
            owner: method.map(|(j, _)| j),
            recv: method.map_or(Recv::None, |(_, recv)| recv),
            params,
            ret,
            body,
            tail: Some(tail),
        });
        self.cur_fn = None;
        self.cur_gens.clear();
        self.this = None;
        self.scope.clear();
    }

    /// An argument of type `p` in a generic body: in a `&mut self` method often a call that changes `self` (a copy of
    /// the `T` in a local is another root), so that the order of the arguments and the state they see show.
    fn self_arg(&mut self, p: Ty) -> Expr {
        if p.is_int()
            && matches!(self.this, Some((_, Recv::Mut)))
            && self.b.chance(60)
            && let Some(call) = self.root_call(p, Root::SelfV, 2)
        {
            return call;
        }
        self.expr(p, 2)
    }

    /// The trait methods through which a type parameter's value is read as an integer; without `by_value` the by-value
    /// ones are left out.
    fn read_methods(&self, gp: &GenParam, by_value: bool) -> Vec<(u8, u8)> {
        self.bound_methods(gp, by_value)
            .into_iter()
            .filter(|&(tr, m)| {
                matches!(
                    self.table.traits[usize::from(tr)].methods[usize::from(m)].ret,
                    TRet::Ty(t) if t.is_int()
                )
            })
            .collect()
    }

    /// The trait methods a type parameter's bounds give; without `by_value` the by-value ones are left out.
    fn bound_methods(&self, gp: &GenParam, by_value: bool) -> Vec<(u8, u8)> {
        let mut out = Vec::new();
        for &tr in &gp.bounds {
            for (m, method) in self.table.traits[usize::from(tr)]
                .methods
                .iter()
                .enumerate()
            {
                if by_value || method.recv != Recv::Value {
                    out.push((tr, m as u8));
                }
            }
        }
        out
    }

    /// A `&mut self` call through the `T` held in `c` whose value has type `ty`, if its bounds have one.
    fn param_mut_call(&mut self, gp: &GenParam, c: u32, g: usize, ty: Ty) -> Option<Expr> {
        let methods: Vec<(u8, u8)> = self
            .bound_methods(gp, false)
            .into_iter()
            .filter(|&(tr, m)| {
                let method = &self.table.traits[usize::from(tr)].methods[usize::from(m)];
                method.recv == Recv::Mut && method.ret == TRet::Ty(ty)
            })
            .collect();
        if methods.is_empty() {
            return None;
        }
        let (tr, m) = methods[self.b.below(methods.len() as u32) as usize];
        let method = self.table.traits[usize::from(tr)].methods[usize::from(m)].clone();
        let f = self.any_impl(tr, m);
        let args: Vec<Expr> = method.params.iter().map(|&p| self.expr(p, 1)).collect();
        Some(Expr::Call {
            f,
            owner: None,
            recv: Some(Place::var(c, Ty::Param(g as u8))),
            args,
            ty,
        })
    }
}
