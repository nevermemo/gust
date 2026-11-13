//! The closure space (DIFFTEST_FORMAT 12, slice m9.d1e): `let cl3 = |p: T, mut q: [u32; 3]| -> R { .. };` in kernel,
//! helper and method bodies, and the calls `cl3(a, b)` in every position a helper call is accepted.
//!
//! The refused forms (SEMANTICS section 11, E0328 to E0332) are not written, by construction: a closure body is drawn
//! with the scope reset to its parameters, no `self`, no `i`, no buffer and no `return` (`in_closure`), and with the
//! list of closures emptied, so it captures nothing and calls no closure; every parameter is typed, the result type is
//! written, the binding is immutable and a closure is never a value (it is only ever called by name). The name is
//! `cl<N>`, N the program-wide variable id, so it is distinct from `m<k>` (functions), `v<n>` (variables), `i`, `id`,
//! `S<j>`, `E<j>`, `f<k>` and every `__gust_` name.
//!
//! D168 (the NVIDIA hole, plans/m9/arrbug.md): the guard of format 11 stays on every array a closure touches. A closure
//! writes an array parameter only through `array_param_uses` (a literal index, or a literal modulo `.len()`, the same
//! indices as a helper's array parameter: no dynamic index and no array local inside a closure). An array a closure
//! returns is a call result, registered with `uniform` and `equal` like a helper's, so the kernel's compound sets on it
//! take the `non_identity` operand or a literal index.

use super::{Gen, fold_in};
use crate::ast::{ArrTy, BinOp, ClosureDef, Expr, Place, Recv, Root, Stmt, Ty};

/// A closure in scope: its variable, its parameter types and its result type.
#[derive(Clone, Debug)]
pub(super) struct Cl {
    pub var: u32,
    pub params: Vec<Ty>,
    /// Whether each parameter is `mut` (a struct or array the body may write).
    pub muts: Vec<bool>,
    /// Whether the body writes each aggregate parameter (the post-call read of a caller variable can show a difference).
    pub writes: Vec<bool>,
    /// Whether each integer parameter enters an integer result with a weight of its own (the order of two arguments shows).
    pub weighted: Vec<bool>,
    pub ret: Ty,
}

/// How the scalar arguments of a closure call are biased toward the order-sensitive shapes.
#[derive(Clone, Debug)]
pub(super) enum ArgBias {
    /// Ordinary expressions.
    None,
    /// Often a `&mut self` call on this root (two of them in one call: `f(c.bump(), c.bump())`).
    Root(Root),
    /// Often a `&mut self` call on an element of this array (at this index, if given).
    Elem(u32, ArrTy, Option<Expr>),
}

impl Gen<'_> {
    /// `let cl<N> = |..| -> R { .. };`: draws the parameters, the body in a scope of its own, and the result; the
    /// closure is in scope (`cls`) for the rest of the block that holds the statement.
    pub(super) fn closure_let(&mut self, depth: u32) -> Stmt {
        let structs = self.table.structs.len() as u32;
        let count = 1 + self.b.below(3);
        let mut params: Vec<(u32, Ty, bool)> = Vec::new();
        for slot in 0..count {
            let ty = if slot == 0 && self.b.chance(50) {
                [Ty::U32, Ty::I32][self.b.below(2) as usize]
            } else {
                match self.b.weighted(&[
                    30,
                    26,
                    7,
                    10,
                    if self.cur_fn.is_some() { 34 } else { 15 },
                    12,
                ]) {
                    0 => Ty::U32,
                    1 => Ty::I32,
                    2 => Ty::Bool,
                    3 => self.enum_ty(),
                    4 => Ty::Struct(self.b.below(structs) as u8),
                    _ => self.array_param_ty(),
                }
            };
            let mutable = match ty {
                Ty::Array(_) => true,
                Ty::Struct(_) => self.b.chance(55),
                _ => false,
            };
            params.push((self.fresh(), ty, mutable));
        }
        // The body sees its parameters and nothing else.
        let saved_scope = std::mem::take(&mut self.scope);
        let saved_cls = std::mem::take(&mut self.cls);
        let saved_pre = std::mem::take(&mut self.pre);
        let saved_forbid = std::mem::take(&mut self.forbid);
        let saved_this = self.this.take();
        let saved_ret = self.cur_ret.take();
        self.in_closure = true;
        for &(var, ty, mutable) in &params {
            // An array parameter is `mut` in the text but is not an assignable local of the block drawing.
            self.scope.push((var, ty, mutable && !ty.is_array()));
            // A parameter is no `&mut self` receiver (E0333); its fields may still be written.
            if ty.is_struct() {
                self.forbid.push(Root::Var(var));
            }
        }
        let mut body = self.block(depth.max(1), 0, 2);
        let pairs: Vec<(u32, Ty)> = params.iter().map(|p| (p.0, p.1)).collect();
        let reads = self.array_param_uses(&pairs, &mut body);
        let array_params: Vec<(u32, Ty)> =
            pairs.iter().copied().filter(|p| p.1.is_array()).collect();
        let pre_mark = self.pre.len();
        let at = body.len();
        let weights = [
            62,
            if self.table.enums.is_empty() { 0 } else { 8 },
            8,
            if array_params.is_empty() { 0 } else { 22 },
        ];
        let kind = self.b.weighted(&weights);
        let (ret, tail) = match kind {
            1 => {
                let ty = self.enum_ty();
                (ty, self.expr(ty, 3))
            }
            2 => {
                let j = self.b.below(structs) as u8;
                let ty = Ty::Struct(j);
                // A struct literal is no tail (E0810): bind it first.
                match self.struct_expr(j, 3, true) {
                    Some(init) => {
                        let var = self.fresh();
                        body.push(Stmt::Let {
                            var,
                            ty,
                            mutable: false,
                            init,
                        });
                        (ty, Expr::Var(var, ty))
                    }
                    None => self.closure_scalar_tail(&reads, &pairs),
                }
            }
            3 => {
                let (var, ty) = array_params[self.b.below(array_params.len() as u32) as usize];
                (ty, Expr::Var(var, ty))
            }
            _ => self.closure_scalar_tail(&reads, &pairs),
        };
        self.splice_pre(pre_mark, at, &mut body);
        self.in_closure = false;
        self.scope = saved_scope;
        self.cls = saved_cls;
        self.pre = saved_pre;
        self.forbid = saved_forbid;
        self.this = saved_this;
        self.cur_ret = saved_ret;
        let var = self.fresh();
        self.bind_closure(ClosureDef {
            var,
            params,
            ret,
            body,
            tail,
        })
    }

    /// The closure space ends its method table with a counter for every struct that has an integer field:
    /// `fn mK(&mut self) -> uint { self.fJ = (self.fJ & 255) + 1; self.fJ }`. Its results differ from call to call, so
    /// two calls on one root show their order (a closure's arguments `c.mK()`, `c.mK()`), without drawing anything
    /// from the generator's stream.
    pub(super) fn counter_methods(&mut self) {
        for j in 0..self.table.structs.len() {
            let Some(k) = self.table.structs[j]
                .fields
                .iter()
                .position(|f| matches!(f, Ty::U32 | Ty::I32))
            else {
                continue;
            };
            let fty = self.table.structs[j].fields[k];
            let sty = Ty::Struct(j as u8);
            let mut place = Place::this(sty);
            place.path.push(k as u8);
            let masked = Expr::bin(
                BinOp::BitAnd,
                Expr::Read {
                    place: place.clone(),
                    ty: fty,
                },
                Expr::Lit(fty, 255),
            );
            let value = Expr::bin(BinOp::Add, masked, Expr::Lit(fty, 1));
            self.table.fns.push(crate::ast::FnDef {
                imp: None,
                gens: Vec::new(),
                owner: Some(j as u8),
                recv: Recv::Mut,
                params: Vec::new(),
                ret: Some(fty),
                body: vec![Stmt::Set {
                    place: place.clone(),
                    ty: fty,
                    op: None,
                    value,
                }],
                tail: Some(Expr::Read { place, ty: fty }),
            });
        }
    }

    /// Registers `def` as a closure in reach and returns its binding statement.
    pub(super) fn bind_closure(&mut self, def: ClosureDef) -> Stmt {
        let info = crate::ast::closure_info(&self.table, &def);
        self.cls.push(Cl {
            var: def.var,
            params: def.params.iter().map(|p| p.1).collect(),
            muts: def.params.iter().map(|p| p.2).collect(),
            writes: info.writes,
            weighted: info.weighted,
            ret: def.ret,
        });
        Stmt::LetClosure { def: Box::new(def) }
    }

    /// A scalar result type and its tail: the array reads of the written parameters fold into an integer or `bool` one.
    fn closure_scalar_tail(&mut self, reads: &[Expr], params: &[(u32, Ty)]) -> (Ty, Expr) {
        let ty = [Ty::U32, Ty::I32, Ty::Bool][self.b.weighted(&[45, 40, 15])];
        let mut value = self.expr(ty, 4);
        // The integer parameters enter the result with weights of their own (`(p & 255) * (2k + 1)`), so the order of the
        // arguments of a call shows in its value and no overflow can come from them.
        let ints: Vec<(u32, Ty)> = params
            .iter()
            .copied()
            .filter(|p| matches!(p.1, Ty::U32 | Ty::I32))
            .collect();
        for (k, (var, pty)) in ints.into_iter().enumerate().take(3) {
            if self.b.chance(95) {
                let masked = Expr::bin(BinOp::BitAnd, Expr::Var(var, pty), Expr::Lit(pty, 255));
                let weighted = Expr::bin(BinOp::Mul, masked, Expr::Lit(pty, 2 * k as u32 + 1));
                value = fold_in(value, weighted);
            }
        }
        // A method of a struct parameter that takes `&self` or `self` (a parameter is no `&mut self` receiver).
        for &(var, pty) in params {
            let Ty::Struct(j) = pty else { continue };
            if !self.b.chance(80) {
                continue;
            }
            let candidates: Vec<usize> = (0..self.callable())
                .filter(|&f| {
                    let def = &self.table.fns[f];
                    def.owner == Some(j)
                        && matches!(def.recv, Recv::Shared | Recv::Value)
                        && matches!(def.ret, Some(Ty::U32 | Ty::I32 | Ty::Bool))
                })
                .collect();
            if candidates.is_empty() {
                continue;
            }
            let f = candidates[self.b.below(candidates.len() as u32) as usize];
            let ret = self.table.fns[f].ret.unwrap_or(Ty::U32);
            if let Some((recv, args)) = self.call_parts_on(f, 2, Some(Place::var(var, pty))) {
                let call = Expr::Call {
                    f,
                    owner: None,
                    recv,
                    args,
                    ty: ret,
                };
                value = fold_in(value, call);
            }
        }
        for read in reads {
            value = fold_in(value, read.clone());
        }
        (ty, value)
    }

    /// The arguments of a call of `cl`, left to right.
    pub(super) fn closure_args(
        &mut self,
        cl: &Cl,
        depth: u32,
        bias: &ArgBias,
    ) -> Option<Vec<Expr>> {
        let mut args = Vec::new();
        let mut enum_root: Option<Root> = None;
        for &pty in &cl.params {
            let arg = match pty {
                Ty::Array(a) => self.array_arg(a, depth),
                // A struct literal is only a `let` initializer or the right side of an assignment (E0810): an
                // argument is a local, a field, or the value of a call.
                Ty::Struct(j) => self.struct_expr(j, depth, false)?,
                Ty::Enum(_) => self.enum_arg(pty, depth, &mut enum_root),
                scalar => {
                    let call = match bias {
                        ArgBias::Root(root) if self.b.chance(78) => {
                            self.root_call(scalar, *root, depth.max(1))
                        }
                        ArgBias::Elem(var, aty, at) if self.b.chance(78) => {
                            self.elem_call(*var, *aty, scalar, depth.max(1), at.clone())
                        }
                        _ => None,
                    };
                    call.unwrap_or_else(|| self.expr(scalar, depth.min(2)))
                }
            };
            args.push(arg);
        }
        Some(args)
    }

    /// A call of a closure in scope that returns `ty`. With `bias`, the arguments are often `&mut self` calls on that
    /// root; without, half of the calls pick a root of their own.
    pub(super) fn closure_call(&mut self, ty: Ty, depth: u32, bias: Option<Root>) -> Option<Expr> {
        if !self.cls.iter().any(|c| c.ret == ty) {
            return None;
        }
        let bias = match bias {
            Some(root) => ArgBias::Root(root),
            None if self.b.chance(50) => self.bias_root().map_or(ArgBias::None, ArgBias::Root),
            None => ArgBias::None,
        };
        self.closure_call_with(ty, depth, &bias)
    }

    /// [`Gen::closure_call`] with the argument bias given.
    pub(super) fn closure_call_with(&mut self, ty: Ty, depth: u32, bias: &ArgBias) -> Option<Expr> {
        let candidates: Vec<Cl> = self.cls.iter().filter(|c| c.ret == ty).cloned().collect();
        if candidates.is_empty() {
            return None;
        }
        // Under a root bias, closures with two integer parameters are preferred: the order of their arguments shows.
        let two_ints: Vec<Cl> = candidates
            .iter()
            .filter(|c| {
                c.params
                    .iter()
                    .filter(|p| matches!(p, Ty::U32 | Ty::I32))
                    .count()
                    >= 2
            })
            .cloned()
            .collect();
        let pool = if matches!(bias, ArgBias::Root(_)) && !two_ints.is_empty() && self.b.chance(75)
        {
            two_ints
        } else {
            candidates
        };
        let cl = pool[self.b.below(pool.len() as u32) as usize].clone();
        let args = self.closure_args(&cl, depth, bias)?;
        Some(Expr::ClosureCall {
            var: cl.var,
            args,
            ty,
        })
    }

    /// A call of a closure that takes a `mut` struct or array, passed a variable of the caller, followed in the same
    /// expression by a read of that variable: `f(v) + v.f0`. The call runs first, so the read shows whether the callee
    /// changed the caller's value (it must not: the parameter is a copy).
    pub(super) fn closure_observed_call(&mut self, ty: Ty, depth: u32) -> Option<Expr> {
        // The read joins the call with a `^` (no short circuit, no dropped arm): an integer result and an integer read.
        if !matches!(ty, Ty::U32 | Ty::I32) {
            return None;
        }
        let mut options: Vec<(Cl, usize, u32)> = Vec::new();
        for cl in self.cls.iter().filter(|c| c.ret == ty) {
            for (slot, pty) in cl.params.iter().enumerate() {
                if !cl.muts[slot]
                    || !cl.writes[slot]
                    || !matches!(pty, Ty::Struct(_) | Ty::Array(_))
                {
                    continue;
                }
                for &(var, vty, _) in &self.scope {
                    if vty == *pty && !self.forbid.contains(&Root::Var(var)) {
                        options.push((cl.clone(), slot, var));
                    }
                }
            }
        }
        if options.is_empty() {
            return None;
        }
        let (cl, slot, var) = options[self.b.below(options.len() as u32) as usize].clone();
        let pty = cl.params[slot];
        let read = match pty {
            Ty::Struct(j) => {
                let k = self.table.structs[usize::from(j)]
                    .fields
                    .iter()
                    .position(|f| matches!(f, Ty::U32 | Ty::I32))?;
                let fty = self.table.structs[usize::from(j)].fields[k];
                Expr::Read {
                    place: Place::var(var, pty).field(k as u8),
                    ty: fty,
                }
            }
            Ty::Array(aty) => Expr::Read {
                place: Place {
                    index: Some(Box::new(Expr::Lit(Ty::U32, 0))),
                    ..Place::var(var, pty)
                },
                ty: aty.e.ty(),
            },
            _ => return None,
        };
        let mut args = self.closure_args(&cl, depth, &ArgBias::None)?;
        args[slot] = Expr::Var(var, pty);
        let call = Expr::ClosureCall {
            var: cl.var,
            args,
            ty,
        };
        Some(fold_in(call, read))
    }

    /// `f(c.bump()) ^ f(c.bump())`: two calls of one closure whose arguments call `&mut self` on one root, the left
    /// one running first.
    pub(super) fn closure_twice(&mut self, ty: Ty, depth: u32) -> Option<Expr> {
        if !matches!(ty, Ty::U32 | Ty::I32) {
            return None;
        }
        let root = self.bias_root()?;
        let candidates: Vec<Cl> = self
            .cls
            .iter()
            .filter(|c| {
                c.ret == ty
                    && c.params
                        .iter()
                        .any(|p| matches!(p, Ty::U32 | Ty::I32 | Ty::Bool))
            })
            .cloned()
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let cl = candidates[self.b.below(candidates.len() as u32) as usize].clone();
        let mut calls = Vec::new();
        for _ in 0..2 {
            let args = self.closure_args(&cl, depth, &ArgBias::Root(root))?;
            calls.push(Expr::ClosureCall {
                var: cl.var,
                args,
                ty,
            });
        }
        let r = calls.pop()?;
        let l = calls.pop()?;
        // Mostly a combination that is not symmetric in the two calls (`l ^ ((r & 255) * 3)`), so that the order of the
        // calls shows in the value and not only the number of body runs; sometimes the symmetric `^` and `|`.
        if self.b.chance(75) {
            let masked = Expr::bin(BinOp::BitAnd, r, Expr::Lit(ty, 255));
            let scaled = Expr::bin(BinOp::Mul, masked, Expr::Lit(ty, 3));
            return Some(Expr::bin(BinOp::BitXor, l, scaled));
        }
        let op = [BinOp::BitXor, BinOp::BitOr][self.b.below(2) as usize];
        Some(Expr::bin(op, l, r))
    }

    /// `let c = cl(a, ..);`: a call of a closure that returns an array, bound as a new array local. The result is a
    /// call result: registered like a helper's (the D168 guard).
    pub(super) fn closure_array_stmt(&mut self, arrays: &mut Vec<(u32, ArrTy)>) -> Option<Stmt> {
        let candidates: Vec<Cl> = self
            .cls
            .iter()
            .filter(|c| matches!(c.ret, Ty::Array(_)))
            .cloned()
            .collect();
        if candidates.is_empty() || arrays.len() >= 4 {
            return None;
        }
        let cl = candidates[self.b.below(candidates.len() as u32) as usize].clone();
        let Ty::Array(aty) = cl.ret else { return None };
        let bias = match self.bias_root() {
            Some(root) if self.b.chance(50) => ArgBias::Root(root),
            _ => ArgBias::None,
        };
        let args = self.closure_args(&cl, 3, &bias)?;
        let var = self.fresh();
        self.scope.push((var, cl.ret, false));
        arrays.push((var, aty));
        Some(Stmt::Let {
            var,
            ty: cl.ret,
            mutable: true,
            init: Expr::ClosureCall {
                var: cl.var,
                args,
                ty: cl.ret,
            },
        })
    }

    /// Kernel-start probes (fix round 1 of m9.d1e): a closure of two weighted integer parameters, and a closure that
    /// writes a struct parameter. The end of the kernel calls them in the forms the order and by-value rules show in
    /// (`closure_probe_calls`), so that a wrong argument order or a parameter passed by reference changes the output of
    /// many closure programs, not of a few.
    pub(super) fn closure_probe_lets(&mut self, body: &mut Vec<Stmt>) {
        if self.b.chance(88) {
            let tys = [Ty::U32, Ty::I32];
            let pa = tys[self.b.below(2) as usize];
            let pb = tys[self.b.below(2) as usize];
            let ret = tys[self.b.below(2) as usize];
            let (a, b) = (self.fresh(), self.fresh());
            let term = |var: u32, ty: Ty, weight: u32| {
                let masked = Expr::bin(BinOp::BitAnd, Expr::Var(var, ty), Expr::Lit(ty, 255));
                let weighted = Expr::bin(BinOp::Mul, masked, Expr::Lit(ty, weight));
                if ty == ret {
                    weighted
                } else {
                    Expr::cast(ret, weighted)
                }
            };
            let tail = Expr::bin(BinOp::BitXor, term(a, pa, 1), term(b, pb, 3));
            let var = self.fresh();
            let stmt = self.bind_closure(ClosureDef {
                var,
                params: vec![(a, pa, false), (b, pb, false)],
                ret,
                body: Vec::new(),
                tail,
            });
            body.push(stmt);
        }
        if self.b.chance(85) {
            let mut options: Vec<(u32, Ty, u8, Ty)> = Vec::new();
            for &(var, vty, _) in &self.scope {
                if let Ty::Struct(j) = vty {
                    for (k, f) in self.table.structs[usize::from(j)].fields.iter().enumerate() {
                        if matches!(f, Ty::U32 | Ty::I32) {
                            options.push((var, vty, k as u8, *f));
                        }
                    }
                }
            }
            if !options.is_empty() {
                let (sv, sty, k, fty) = options[self.b.below(options.len() as u32) as usize];
                let p = self.fresh();
                let place = Place::var(p, sty).field(k);
                let set = Stmt::Set {
                    place: place.clone(),
                    ty: fty,
                    op: Some(BinOp::BitXor),
                    value: Expr::Lit(fty, 1),
                };
                let tail = Expr::Read { place, ty: fty };
                let var = self.fresh();
                let stmt = self.bind_closure(ClosureDef {
                    var,
                    params: vec![(p, sty, true)],
                    ret: fty,
                    body: vec![set],
                    tail,
                });
                self.probe_writers.push((var, sv, k, fty));
                body.push(stmt);
            }
        }
    }

    /// The arguments of a probe call: an integer parameter is a `&mut self` call on `root` (of either integer type, cast
    /// to the parameter's), the others ordinary.
    fn probe_args(&mut self, cl: &Cl, root: Root) -> Vec<Expr> {
        let mut args = Vec::new();
        for (slot, &pty) in cl.params.iter().enumerate() {
            let arg = match pty {
                Ty::U32 | Ty::I32 if cl.weighted[slot] => {
                    let other = if pty == Ty::U32 { Ty::I32 } else { Ty::U32 };
                    match self.root_call(pty, root, 2) {
                        Some(call) => call,
                        None => match self.root_call(other, root, 2) {
                            Some(call) => Expr::cast(pty, call),
                            None => self.expr(pty, 2),
                        },
                    }
                }
                scalar => self.expr(scalar, 2),
            };
            args.push(arg);
        }
        args
    }

    /// The probe calls of the end of the kernel: `f(c.bump(), c.bump())` on the two-parameter closure (both arguments
    /// call `&mut self` on one root, retried until they do), and `g(v) ^ v.k` on the writer closure (the post-call
    /// read of the caller's variable joins the result with a `^`, which does not short-circuit).
    pub(super) fn closure_probe_calls(&mut self) -> Vec<Expr> {
        let mut out = Vec::new();
        let ordered: Vec<Cl> = self
            .cls
            .iter()
            .filter(|c| {
                matches!(c.ret, Ty::U32 | Ty::I32)
                    && c.params.iter().all(|p| matches!(p, Ty::U32 | Ty::I32))
                    && c.weighted.iter().filter(|w| **w).count() >= 2
            })
            .cloned()
            .collect();
        if !ordered.is_empty() && self.b.chance(90) {
            let cl = ordered[self.b.below(ordered.len() as u32) as usize].clone();
            for _ in 0..10 {
                let Some(root) = self.bias_root() else { break };
                let args = self.probe_args(&cl, root);
                let roots: Vec<Vec<Root>> = args
                    .iter()
                    .map(|arg| {
                        let mut r = Vec::new();
                        crate::ast::mutating_roots(&self.table, arg, &mut r);
                        r
                    })
                    .collect();
                let both = (0..args.len()).any(|a| {
                    cl.weighted[a]
                        && ((a + 1)..args.len()).any(|b| {
                            cl.weighted[b] && roots[a].iter().any(|r| roots[b].contains(r))
                        })
                });
                if both {
                    out.push(Expr::ClosureCall {
                        var: cl.var,
                        args,
                        ty: cl.ret,
                    });
                    break;
                }
            }
        }
        let writers: Vec<(u32, u32, u8, Ty)> = self
            .probe_writers
            .iter()
            .copied()
            .filter(|w| {
                self.cls.iter().any(|c| c.var == w.0) && self.scope.iter().any(|(v, ..)| *v == w.1)
            })
            .collect();
        if !writers.is_empty() && self.b.chance(90) {
            let (var, sv, k, fty) = writers[self.b.below(writers.len() as u32) as usize];
            let sty = self.scope.iter().find(|s| s.0 == sv).map(|s| s.1);
            if let Some(sty) = sty {
                let call = Expr::ClosureCall {
                    var,
                    args: vec![Expr::Var(sv, sty)],
                    ty: fty,
                };
                let read = Expr::Read {
                    place: Place::var(sv, sty).field(k),
                    ty: fty,
                };
                out.push(Expr::bin(BinOp::BitXor, call, read));
            }
        }
        out
    }

    /// Puts the closures bound at the top level of `body` back in reach (the block drawing popped them), as the array
    /// section and the result draw need.
    pub(super) fn restore_closures(&mut self, body: &[Stmt]) {
        for stmt in body {
            if let Stmt::LetClosure { def } = stmt
                && !self.cls.iter().any(|c| c.var == def.var)
            {
                let info = crate::ast::closure_info(&self.table, def);
                self.cls.push(Cl {
                    var: def.var,
                    params: def.params.iter().map(|p| p.1).collect(),
                    muts: def.params.iter().map(|p| p.2).collect(),
                    writes: info.writes,
                    weighted: info.weighted,
                    ret: def.ret,
                });
            }
        }
    }
}
