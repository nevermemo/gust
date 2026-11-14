//! The array space (DIFFTEST_FORMAT 11, slice m9.d1d): typed array locals, literals and repeats, indexed reads and
//! element sets, arrays as parameters and results of functions, method calls on struct elements, and the final store
//! `out[idx] op= result`.
//!
//! The rules of `ast::Check` and of the macro's validator (E0421, E0810, E0333, SEMANTICS section 11) are kept here by
//! construction: a repeat is a whole `let` initializer or assignment value; a struct literal is never an element
//! (a struct element is a variable, a call or a copy); the index of a write place names no variable (a literal, `i`,
//! an input read), or only the array itself when the value calls nothing on it, so that `reject_stale_index` has
//! nothing to refuse; a compound `/= %= ^=` on an indexed place has a value without a `&mut self` call; an element
//! index is in bounds by construction (a literal below the length, or `x % len`), and the interpreter still reports
//! `Violation::Index` as the net. Arrays of vectors and matrices, and arrays of enums, are not drawn (DIFFTEST.md).

use super::Gen;
use crate::ast::{
    ArrTy, BinOp, Elem, Expr, MAX_ARRAY, MAX_ARRAY_FLAT, Place, Recv, Root, Stmt, Store, Ty,
    walk_expr,
};

impl Gen<'_> {
    /// An array type for a parameter or a result: `u32` or `i32` elements, two to four of them.
    pub(super) fn array_param_ty(&mut self) -> Ty {
        let e = if self.b.chance(55) {
            Elem::U32
        } else {
            Elem::I32
        };
        Ty::Array(ArrTy::new(e, 2 + self.b.below(3) as u8))
    }

    /// The type of an array local: scalar, `bool` or struct elements, at most 16 scalars in all.
    fn array_local_ty(&mut self) -> ArrTy {
        let structs = self.table.structs.len() as u32;
        let e = match self.b.weighted(&[30, 22, 8, 40]) {
            0 => Elem::U32,
            1 => Elem::I32,
            2 => Elem::Bool,
            _ => Elem::Struct(self.b.below(structs) as u8),
        };
        let per = self.table.ty_len(e.ty()).max(1);
        let most = (MAX_ARRAY_FLAT / per).min(MAX_ARRAY);
        let n = if most < 2 {
            1
        } else {
            2 + self.b.below(most as u32 - 1)
        };
        ArrTy::new(e, n as u8)
    }

    /// A place of an element of array local `var`: `a[index]`, then `path` into a struct element.
    fn elem_place(var: u32, aty: ArrTy, index: Expr, path: &[u8]) -> Place {
        Place {
            root: Root::Var(var),
            root_ty: Ty::Array(aty),
            path: path.to_vec(),
            index: Some(Box::new(index)),
        }
    }

    /// An index into an array of `n` elements that names no variable: a literal below `n`, or `x % n` with `x` a
    /// `u32` made of `i` and literals or an input read (an `i32` index sometimes). In bounds whatever `x` is.
    fn array_index(&mut self, n: u8) -> Expr {
        let n32 = u32::from(n);
        match self.b.weighted(&[30, 28, 14, 14, 14]) {
            0 => Expr::u32(self.b.below(n32)),
            1 => Expr::bin(BinOp::Rem, Expr::Gid, Expr::u32(n32)),
            2 => {
                let k = Expr::u32(self.b.below(256));
                Expr::bin(
                    BinOp::Rem,
                    Expr::bin(BinOp::BitXor, Expr::Gid, k),
                    Expr::u32(n32),
                )
            }
            3 => {
                // `i` as an `i32`: non-negative, so `% n` is too.
                Expr::bin(
                    BinOp::Rem,
                    Expr::cast(Ty::I32, Expr::Gid),
                    Expr::i32(i32::from(n)),
                )
            }
            _ => match self.inputs_of(Ty::U32).first().copied() {
                Some(buf) => {
                    let read = self.input_read(buf, Ty::U32);
                    Expr::bin(BinOp::Rem, read, Expr::u32(n32))
                }
                None => Expr::bin(BinOp::Rem, Expr::Gid, Expr::u32(n32)),
            },
        }
    }

    /// An index inside a function (no `i`, no buffer, no call there): a literal, or a literal modulo `.len()`.
    fn fn_array_index(&mut self, var: u32, aty: ArrTy) -> Expr {
        let k = Expr::u32(self.b.below(256));
        if self.b.chance(30) {
            let place = Place::var(var, Ty::Array(aty));
            Expr::bin(BinOp::Rem, k, Expr::ArrLen { place })
        } else {
            Expr::u32(self.b.below(u32::from(aty.n)))
        }
    }

    /// `x % a.len()` for the array local `var`, `x` made of `i`: the length through `.len()` (an index only).
    fn array_len_index(var: u32, aty: ArrTy) -> Expr {
        let place = Place::var(var, Ty::Array(aty));
        Expr::bin(BinOp::Rem, Expr::Gid, Expr::ArrLen { place })
    }

    /// A writable struct root with a `&mut self` method in reach, if any: the root a literal's calls share.
    pub(super) fn bias_root(&mut self) -> Option<Root> {
        let mut roots: Vec<Root> = Vec::new();
        for (place, _, writable) in self.struct_places() {
            if writable && !self.forbid.contains(&place.root) && !roots.contains(&place.root) {
                roots.push(place.root);
            }
        }
        if roots.is_empty() {
            None
        } else {
            Some(roots[self.b.below(roots.len() as u32) as usize])
        }
    }

    /// The fields of struct `j` that a `&mut self` method of `j` sets in its top-level statements (so a read of one
    /// after a call to such a method can see the write).
    fn written_fields(&self, j: u8) -> Vec<u8> {
        let mut out = Vec::new();
        for def in &self.table.fns {
            if def.recv != Recv::Mut || def.owner != Some(j) {
                continue;
            }
            for stmt in &def.body {
                if let Stmt::Set { place, .. } = stmt
                    && place.root == Root::SelfV
                    && place.index.is_none()
                    && let [k] = place.path[..]
                    && !out.contains(&k)
                {
                    out.push(k);
                }
            }
        }
        out
    }

    /// A read of a scalar field of type `ty` on a struct place rooted at `root`: after a `&mut self` call on the same
    /// root it sees the call's writes, before it does not, so an array literal mixing calls and reads shows the order.
    fn root_field_read(&mut self, ty: Ty, root: Root) -> Option<Expr> {
        let places = self.struct_places();
        let mut reads: Vec<Place> = Vec::new();
        for (place, j, _) in places {
            if place.root != root {
                continue;
            }
            let written = self.written_fields(j);
            for (k, field) in self.table.structs[usize::from(j)].fields.iter().enumerate() {
                if *field == ty && written.contains(&(k as u8)) {
                    reads.push(place.field(k as u8));
                }
            }
        }
        if reads.is_empty() {
            return None;
        }
        let place = reads[self.b.below(reads.len() as u32) as usize].clone();
        Some(Expr::Read { place, ty })
    }

    /// One element of an array literal or repeat.
    fn array_elem_value(&mut self, e: Elem, bias: Option<Root>) -> Expr {
        match e {
            Elem::U32 | Elem::I32 => {
                let ty = e.ty();
                // A closure call whose arguments call the bias root: `[f(c.bump()), c.bump()]` (the closure space).
                if let Some(root) = bias
                    && self.closures
                    && !self.cls.is_empty()
                    && self.b.chance(70)
                    && let Some(call) = self.closure_call(ty, 2, Some(root))
                {
                    return call;
                }
                if let Some(root) = bias
                    && self.b.chance(90)
                    && let Some(call) = self
                        .b
                        .chance(45)
                        .then(|| self.root_field_read(ty, root))
                        .flatten()
                        .or_else(|| self.root_call(ty, root, 2))
                {
                    return call;
                }
                self.expr(ty, 2)
            }
            Elem::Bool => self.expr(Ty::Bool, 2),
            Elem::Struct(j) => self.fresh_struct(j),
            Elem::Enum(_) | Elem::Vec(..) | Elem::Mat(_) => {
                unreachable!("the generator draws no enum, vector or matrix element")
            }
        }
    }

    /// A struct value for an array element: a struct is not `Copy` in the Rust twin, and a struct literal cannot be an
    /// element (E0810), so the element is a call, or a variable bound just before the statement, used here and
    /// nowhere else (it is not put in scope: reading it again would be a use after the move).
    fn fresh_struct(&mut self, j: u8) -> Expr {
        let ty = Ty::Struct(j);
        if self.b.chance(40)
            && let Some(call) = self.value_call(ty, 2)
        {
            return call;
        }
        let fields = self.table.structs[usize::from(j)].fields.clone();
        let values: Vec<Expr> = fields
            .into_iter()
            .map(|field| match field {
                Ty::Struct(inner) => self.fresh_struct(inner),
                scalar => self.expr(scalar, 2),
            })
            .collect();
        let var = self.fresh();
        self.pre.push(Stmt::Let {
            var,
            ty,
            mutable: false,
            init: Expr::StructLit {
                ty,
                order: (0..values.len() as u8).collect(),
                fields: values,
            },
        });
        Expr::Var(var, ty)
    }

    /// An array value of `aty`: a literal, or (`top`: a whole initializer or assignment value, scalar elements) a
    /// repeat. With a bias root, the elements are often calls on that one root, left to right.
    fn array_value(&mut self, aty: ArrTy, top: bool, bias: Option<Root>) -> Expr {
        let ty = Ty::Array(aty);
        if top
            && !matches!(aty.e, Elem::Struct(_))
            && self.b.chance(match (bias.is_some(), self.closures) {
                (true, true) => 55,
                (true, false) => 40,
                _ => 28,
            })
        {
            let element = self.array_elem_value(aty.e, bias);
            return Expr::Repeat {
                ty,
                e: Box::new(element),
            };
        }
        let elems = (0..aty.n)
            .map(|_| self.array_elem_value(aty.e, bias))
            .collect();
        Expr::ArrayLit { ty, elems }
    }

    /// An argument of an array parameter: a copy of an array local of the type (scalar elements), or a literal.
    pub(super) fn array_arg(&mut self, aty: ArrTy, depth: u32) -> Expr {
        let ty = Ty::Array(aty);
        let locals: Vec<u32> = self
            .scope
            .iter()
            .filter(|(_, t, _)| *t == ty)
            .map(|(v, ..)| *v)
            .collect();
        if !locals.is_empty() && self.b.chance(75) {
            return Expr::Var(locals[self.b.below(locals.len() as u32) as usize], ty);
        }
        let elems = (0..aty.n)
            .map(|_| self.expr(aty.e.ty(), depth.min(2)))
            .collect();
        Expr::ArrayLit { ty, elems }
    }

    /// The statements a function with array parameters runs on them, and an element read to fold into its tail: the
    /// arrays are `mut` parameters, so a function may write them.
    pub(super) fn array_param_uses(
        &mut self,
        params: &[(u32, Ty)],
        body: &mut Vec<Stmt>,
    ) -> Vec<Expr> {
        let mut reads = Vec::new();
        for &(var, ty) in params {
            let Ty::Array(aty) = ty else { continue };
            let elem = aty.e.ty();
            if self.b.chance(85) {
                let index = self.fn_array_index(var, aty);
                let place = Self::elem_place(var, aty, index, &[]);
                // A shift, a division or an overflow would discard programs for nothing: xor, or and add of a mask.
                let (op, value) = match self.b.below(4) {
                    0 => (None, self.expr(elem, 2)),
                    1 => (Some(BinOp::BitXor), self.expr(elem, 2)),
                    2 => (Some(BinOp::BitOr), self.expr(elem, 2)),
                    _ => (
                        Some(BinOp::Add),
                        Expr::bin(BinOp::BitAnd, self.expr(elem, 2), Expr::Lit(elem, 0xFF)),
                    ),
                };
                // E0421: `^=` on an indexed place refuses a value with a `&mut self` call (the counter methods of the
                // closure space make such a value likely).
                let op = match op {
                    Some(BinOp::BitXor) if self.has_mut_call(&value) => Some(BinOp::BitOr),
                    other => other,
                };
                let (place, value) = self.guard_set(var, aty, &[], elem, op, place, value);
                body.push(Stmt::Set {
                    place,
                    ty: elem,
                    op,
                    value,
                });
            }
            let k = self.b.below(u32::from(aty.n)) as u8;
            reads.push(Expr::Read {
                place: Self::elem_place(var, aty, Expr::u32(u32::from(k)), &[]),
                ty: elem,
            });
        }
        reads
    }

    /// A call of a `&mut self` method of struct element `j` returning `ty`, on element `index` of array local
    /// `var`: the call whose effect on the array a surrounding place or read can see.
    pub(super) fn elem_call(
        &mut self,
        var: u32,
        aty: ArrTy,
        ty: Ty,
        depth: u32,
        at: Option<Expr>,
    ) -> Option<Expr> {
        let Elem::Struct(j) = aty.e else { return None };
        let root = Root::Var(var);
        if depth == 0 || self.forbid.contains(&root) {
            return None;
        }
        let candidates: Vec<usize> = (0..self.callable())
            .filter(|&k| {
                let def = &self.table.fns[k];
                def.recv == Recv::Mut && def.owner == Some(j) && def.ret == Some(ty)
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let f = candidates[self.b.below(candidates.len() as u32) as usize];
        // A literal `at` (the index the place of the set also takes) keeps the call and the place on one element.
        let index = match at {
            Some(at @ Expr::Lit(..)) => at,
            _ => self.call_index(var, aty),
        };
        let recv = Self::elem_place(var, aty, index, &[]);
        let (recv, args) = self.call_parts_on(f, depth, Some(recv))?;
        Some(Expr::Call {
            f,
            owner: None,
            recv,
            args,
            ty,
        })
    }

    /// `operand` for `place op= operand` made provably not the identity of `op` at run time (see `guard_set`). `+= -= ^=` take `v | 1` (never 0: a non-zero addend or an odd mask changes every value), `<<= >>=`
    /// take `(v & 31) | 1` (1 to 31) and `*= /=` take `v | 2` (never 0 or 1); these leave an element unchanged only for
    /// special values (0, or -1 under an `i32` `>>`) that the driver cannot prove for a runtime element. `|= &= %=` are
    /// not rewritten: an element with known bits or range can be provably unchanged by them (`(c | 1) | 1`,
    /// `(c & 0xFE) & !1`, `(c & 7) % 9`), so `guard_set` gives them a literal index instead.
    fn non_identity(op: BinOp, v: Expr, ty: Ty) -> Expr {
        match op {
            BinOp::Add | BinOp::Sub | BinOp::BitXor => Expr::bin(BinOp::BitOr, v, Expr::Lit(ty, 1)),
            BinOp::Shl | BinOp::Shr => Expr::bin(
                BinOp::BitOr,
                Expr::bin(BinOp::BitAnd, v, Expr::u32(31)),
                Expr::u32(1),
            ),
            BinOp::Mul | BinOp::Div => Expr::bin(BinOp::BitOr, v, Expr::Lit(ty, 2)),
            _ => v,
        }
    }

    /// The known driver hole (D168, D169; NVIDIA 610.88): a load and store-back of
    /// an unchanged value at a dynamic index into any function-local array (any length, uniform or not) can zero
    /// elements. `place op= value` at a dynamic index therefore gets a provably non-identity operand for
    /// `+= -= ^= <<= >>= *= /=` (every operand: a call or `%` that folds to 0 after inlining triggered it) and a
    /// literal index for `|= &= %=`; a plain set that stores the place's own value back at a dynamic index takes a
    /// literal value instead. A literal index is never in the hole. Removed when a driver fixes it.
    #[allow(clippy::too_many_arguments)]
    fn guard_set(
        &mut self,
        var: u32,
        aty: ArrTy,
        path: &[u8],
        ty: Ty,
        op: Option<BinOp>,
        place: Place,
        value: Expr,
    ) -> (Place, Expr) {
        if matches!(place.index.as_deref(), Some(Expr::Lit(..))) {
            return (place, value);
        }
        match op {
            Some(BinOp::BitOr | BinOp::BitAnd | BinOp::Rem) => (
                Self::elem_place(var, aty, Expr::u32(self.b.below(u32::from(aty.n))), path),
                value,
            ),
            Some(o) => (place, Self::non_identity(o, value, ty)),
            None => {
                let mut inner = &value;
                while let Expr::Cast { e, .. } = inner {
                    inner = e;
                }
                if matches!(inner, Expr::Read { place: p, .. } if *p == place) {
                    let lit = if ty == Ty::Bool { 1 } else { 5 };
                    (place, Expr::Lit(ty, lit))
                } else {
                    (place, value)
                }
            }
        }
    }

    /// An element index for a `&mut self` call on element of array `var`: always a literal (D168, D169: the call's
    /// write-back of `self` is a copy store, which the NVIDIA driver loses at a dynamic index on any local array).
    fn call_index(&mut self, _var: u32, aty: ArrTy) -> Expr {
        Expr::u32(self.b.below(u32::from(aty.n)))
    }

    /// Whether `e` holds a `&mut self` call.
    fn has_mut_call(&self, e: &Expr) -> bool {
        let mut found = false;
        walk_expr(e, &mut |n| {
            if let Expr::Call { f, .. } = n
                && self.table.fns[*f].recv == Recv::Mut
            {
                found = true;
            }
        });
        found
    }

    /// Whether `e` holds a `&mut self` call on `root`.
    fn has_mut_call_on(&self, e: &Expr, root: Root) -> bool {
        let mut found = false;
        walk_expr(e, &mut |n| {
            if let Expr::Call {
                f,
                recv: Some(place),
                ..
            } = n
                && self.table.fns[*f].recv == Recv::Mut
                && place.root == root
            {
                found = true;
            }
        });
        found
    }

    /// `a[k] = value` or `a[k] op= value` on a scalar element, or on a scalar field of a struct element (value often a
    /// `&mut self` call on the array's own struct elements: `ps[k].n += ps[j].bump()`). `None` if the array has no
    /// such place.
    fn array_set_stmt(&mut self, arrays: &[(u32, ArrTy)]) -> Option<Stmt> {
        // Half the time a struct array, when there is one: its elements' methods are what the order shapes need.
        let structs: Vec<(u32, ArrTy)> = arrays
            .iter()
            .copied()
            .filter(|(_, aty)| matches!(aty.e, Elem::Struct(_)))
            .collect();
        let pool: &[(u32, ArrTy)] = if !structs.is_empty() && self.b.chance(85) {
            &structs
        } else {
            arrays
        };
        let (var, aty) = pool[self.b.below(pool.len() as u32) as usize];
        let (path, ty): (Vec<u8>, Ty) = match aty.e {
            Elem::U32 | Elem::I32 => (Vec::new(), aty.e.ty()),
            Elem::Bool => (Vec::new(), Ty::Bool),
            Elem::Struct(j) => {
                let fields: Vec<(u8, Ty)> = self.table.structs[usize::from(j)]
                    .fields
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| !f.is_struct())
                    .map(|(k, f)| (k as u8, *f))
                    .collect();
                if fields.is_empty() {
                    return None;
                }
                // Often a field that one of the struct's `&mut self` methods writes: the value's call on the same
                // element then changes what the place reads, so the order shows.
                let written = self.written_fields(j);
                let prefer: Vec<(u8, Ty)> = fields
                    .iter()
                    .copied()
                    .filter(|(k, f)| written.contains(k) && matches!(f, Ty::U32 | Ty::I32))
                    .collect();
                let pool = if !prefer.is_empty() && self.b.chance(70) {
                    &prefer
                } else {
                    &fields
                };
                let (k, f) = pool[self.b.below(pool.len() as u32) as usize];
                (vec![k], f)
            }
            _ => return None,
        };
        let index = self.array_index(aty.n);
        // The element a call in the value and the place share: a literal (a call at a dynamic index is in the D168/D169
        // hole, so the place takes the call's literal rather than the call taking the place's dynamic index).
        let same_index = Expr::u32(self.b.below(u32::from(aty.n)));
        let mut same = false;
        let place = Self::elem_place(var, aty, index, &path);
        let (op, value) = if ty == Ty::Bool {
            (None, self.expr(Ty::Bool, 3))
        } else {
            let op = self.b.chance(72).then(|| {
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
                ][self.b.below(12) as usize]
            });
            let value_ty = if op.is_some_and(|op| op.is_shift()) {
                Ty::U32
            } else {
                ty
            };
            // The value often calls `&mut self` on the array's own elements, or on one struct root: the place is
            // read after the value, so the order shows.
            let biased = if self.b.chance(85) && matches!(aty.e, Elem::Struct(_)) {
                let at = self.b.chance(80).then(|| same_index.clone());
                same = at.is_some();
                // The closure space: the call is a closure's, with arguments that call the array's own elements.
                let via = if self.closures && !self.cls.is_empty() && self.b.chance(35) {
                    self.closure_call_with(value_ty, 3, &super::ArgBias::Elem(var, aty, at.clone()))
                } else {
                    None
                };
                via.or_else(|| self.elem_call(var, aty, value_ty, 3, at))
            } else if self.b.chance(40) {
                self.bias_root()
                    .and_then(|root| self.root_call(value_ty, root, 3))
            } else {
                None
            };
            let value = match (biased, op) {
                (Some(call), Some(BinOp::Div | BinOp::Rem)) => {
                    Expr::bin(BinOp::BitOr, call, Expr::Lit(value_ty, 1))
                }
                (Some(call), Some(BinOp::Shl | BinOp::Shr)) => {
                    Expr::bin(BinOp::BitAnd, call, Expr::u32(31))
                }
                (Some(call), _) => call,
                (None, Some(BinOp::Div | BinOp::Rem)) => self.divisor(ty, 3),
                (None, Some(BinOp::Shl | BinOp::Shr)) => self.shift_amount(3),
                (None, _) => self.expr(ty, 3),
            };
            (op, value)
        };
        // E0421: `/= %= ^=` on an indexed place refuse a value with a `&mut self` call (the macro cannot order them).
        let op = match op {
            Some(BinOp::Div | BinOp::Rem | BinOp::BitXor) if self.has_mut_call(&value) => {
                Some(BinOp::Add)
            }
            other => other,
        };
        let place = if same {
            Self::elem_place(var, aty, same_index.clone(), &path)
        } else {
            place
        };
        // `x % a.len()` names the array: only when the value calls nothing on it (`reject_stale_index`).
        let place = if self.b.chance(18) && !self.has_mut_call_on(&value, Root::Var(var)) {
            Self::elem_place(var, aty, Self::array_len_index(var, aty), &path)
        } else {
            place
        };
        let (place, value) = self.guard_set(var, aty, &path, ty, op, place, value);
        Some(Stmt::Set {
            place,
            ty,
            op,
            value,
        })
    }

    /// `ps[k].m(args);`: a `&mut self` method on an element, as a statement.
    fn elem_call_stmt(&mut self, arrays: &[(u32, ArrTy)]) -> Option<Stmt> {
        let structs: Vec<(u32, ArrTy)> = arrays
            .iter()
            .copied()
            .filter(|(_, aty)| matches!(aty.e, Elem::Struct(_)))
            .collect();
        if structs.is_empty() {
            return None;
        }
        let (var, aty) = structs[self.b.below(structs.len() as u32) as usize];
        let Elem::Struct(j) = aty.e else { return None };
        let candidates: Vec<usize> = (0..self.callable())
            .filter(|&k| {
                let def = &self.table.fns[k];
                def.recv == Recv::Mut && def.owner == Some(j)
            })
            .collect();
        if candidates.is_empty() || self.forbid.contains(&Root::Var(var)) {
            return None;
        }
        let f = candidates[self.b.below(candidates.len() as u32) as usize];
        let index = self.call_index(var, aty);
        let recv = Self::elem_place(var, aty, index, &[]);
        let (recv, args) = self.call_parts_on(f, 3, Some(recv))?;
        Some(Stmt::Call {
            f,
            owner: None,
            recv,
            args,
        })
    }

    /// The array statements of the kernel, after its ordinary block: array locals, sets, element calls, copies, whole
    /// assignments, helper calls with array arguments and results. Returns the expressions to fold into the result.
    pub(super) fn array_section(&mut self, body: &mut Vec<Stmt>) -> Vec<Expr> {
        let mut arrays: Vec<(u32, ArrTy)> = Vec::new();
        for _ in 0..1 + self.b.below(2) {
            let aty = self.array_local_ty();
            let bias = if self.b.chance(75) {
                self.bias_root()
            } else {
                None
            };
            let (pre_mark, at) = (self.pre.len(), body.len());
            let init = self.array_value(aty, true, bias);
            let var = self.fresh();
            self.scope.push((var, Ty::Array(aty), false));
            body.push(Stmt::Let {
                var,
                ty: Ty::Array(aty),
                mutable: true,
                init,
            });
            self.splice_pre(pre_mark, at, body);
            arrays.push((var, aty));
        }
        for _ in 0..2 + self.b.below(5) {
            let (pre_mark, at) = (self.pre.len(), body.len());
            let mut weights = vec![38, 18, 8, 8, 24, 12];
            if self.closures {
                weights.push(26);
            }
            let kind = self.b.weighted(&weights);
            let stmt = match kind {
                0 => self.array_set_stmt(&arrays),
                1 => self.elem_call_stmt(&arrays),
                2 => self.array_copy_stmt(&mut arrays),
                3 => self.array_assign_stmt(&arrays),
                4 => self.array_call_stmt(&mut arrays),
                6 => self.closure_array_stmt(&mut arrays),
                _ => self.array_set_stmt(&arrays),
            };
            let drew = stmt.is_some();
            body.extend(stmt);
            // The write to the copy of `let mut b = a;` that the copy must not show in `a`.
            if kind == 2 && drew {
                let (copy, aty) = *arrays.last().expect("a copy was added");
                body.extend(self.array_set_stmt(&[(copy, aty)]));
            }
            // Lets that drawing hoisted (a struct element is a variable) come before the statement.
            self.splice_pre(pre_mark, at, body);
        }
        // Every element of every array joins the result, so that the state the statements left is live.
        let mut folds = Vec::new();
        for &(var, aty) in &arrays {
            for k in 0..aty.n {
                let index = Expr::u32(u32::from(k));
                match aty.e {
                    Elem::Struct(j) => {
                        let fields = self.table.structs[usize::from(j)].fields.clone();
                        for (f, field) in fields.iter().enumerate() {
                            if !field.is_struct() {
                                folds.push(Expr::Read {
                                    place: Self::elem_place(var, aty, index.clone(), &[f as u8]),
                                    ty: *field,
                                });
                            }
                        }
                    }
                    other => folds.push(Expr::Read {
                        place: Self::elem_place(var, aty, index, &[]),
                        ty: other.ty(),
                    }),
                }
            }
        }
        if self.b.chance(75)
            && let Some(pair) = self.array_index_pair(&arrays)
        {
            folds.push(pair);
        }
        folds
    }

    /// `(a[c.f() % N] ^ k) ^ a[c.g() % N]`: two element reads whose indices call `&mut self` methods on one root, the
    /// left index running first (the shape `a[f()] .. a[g()]`). A read is no assignee, so its index may call.
    fn array_index_pair(&mut self, arrays: &[(u32, ArrTy)]) -> Option<Expr> {
        let wide: Vec<(u32, ArrTy)> = arrays
            .iter()
            .copied()
            .filter(|(_, aty)| aty.e == Elem::U32)
            .collect();
        if wide.is_empty() {
            return None;
        }
        let root = self.bias_root()?;
        let mut reads = Vec::new();
        for _ in 0..2 {
            let (var, aty) = wide[self.b.below(wide.len() as u32) as usize];
            let call = self.root_call(Ty::U32, root, 2)?;
            let index = Expr::bin(BinOp::Rem, call, Expr::u32(u32::from(aty.n)));
            reads.push(Expr::Read {
                place: Self::elem_place(var, aty, index, &[]),
                ty: Ty::U32,
            });
        }
        let right = reads.pop()?;
        let left = reads.pop()?;
        Some(Expr::bin(
            BinOp::BitXor,
            Expr::bin(BinOp::BitXor, left, Expr::u32(self.b.below(256))),
            right,
        ))
    }

    /// `let mut b = a;` (a copy: the elements are `Copy`), then a write to `b`, so that `a` must stay as it was.
    fn array_copy_stmt(&mut self, arrays: &mut Vec<(u32, ArrTy)>) -> Option<Stmt> {
        let (var, aty) = arrays[self.b.below(arrays.len() as u32) as usize];
        if arrays.len() >= 4 || matches!(aty.e, Elem::Struct(_)) {
            return None;
        }
        let copy = self.fresh();
        self.scope.push((copy, Ty::Array(aty), false));
        arrays.push((copy, aty));
        // The write to the copy comes from the loop in `array_section` (any array may be drawn); the copy itself:
        Some(Stmt::Let {
            var: copy,
            ty: Ty::Array(aty),
            mutable: true,
            init: Expr::Var(var, Ty::Array(aty)),
        })
    }

    /// `a = [..];` or `a = [e; N];`: a whole assignment.
    fn array_assign_stmt(&mut self, arrays: &[(u32, ArrTy)]) -> Option<Stmt> {
        let (var, aty) = arrays[self.b.below(arrays.len() as u32) as usize];
        let bias = if self.b.chance(50) {
            self.bias_root()
        } else {
            None
        };
        let value = self.array_value(aty, true, bias);
        Some(Stmt::Assign { var, value })
    }

    /// `let c = f(a, ..);`: a call of a function with array parameters, whose result is an array or a scalar; the
    /// array result is a new array local.
    fn array_call_stmt(&mut self, arrays: &mut Vec<(u32, ArrTy)>) -> Option<Stmt> {
        let candidates: Vec<usize> = (0..self.callable())
            .filter(|&k| {
                let def = &self.table.fns[k];
                def.recv == Recv::None
                    && def.owner.is_none()
                    && matches!(def.ret, Some(Ty::Array(_)))
            })
            .collect();
        if candidates.is_empty() || arrays.len() >= 4 {
            return None;
        }
        let f = candidates[self.b.below(candidates.len() as u32) as usize];
        let ret = self.table.fns[f].ret?;
        let Ty::Array(aty) = ret else { return None };
        let (recv, args) = self.call_parts_on(f, 3, None)?;
        let var = self.fresh();
        self.scope.push((var, ret, false));
        arrays.push((var, aty));
        Some(Stmt::Let {
            var,
            ty: ret,
            mutable: true,
            init: Expr::Call {
                f,
                owner: None,
                recv,
                args,
                ty: ret,
            },
        })
    }

    /// The final store of an array program: `out[idx] op= result`, `idx` equal to `i` by construction. The initial
    /// output is `init`. `None` leaves `out[i] = result`.
    pub(super) fn array_store(&mut self, result: &Expr) -> Option<Store> {
        if !self.arrays || !matches!(result.ty(), Ty::U32 | Ty::I32) || !self.b.chance(70) {
            return None;
        }
        let call_in_result = self.has_mut_call(result);
        let op = self.b.chance(75).then(|| {
            let ops = [
                BinOp::Add,
                BinOp::Add,
                BinOp::BitOr,
                BinOp::BitAnd,
                BinOp::Sub,
                BinOp::BitXor,
                BinOp::Mul,
                BinOp::Div,
                BinOp::Rem,
            ];
            ops[self.b.below(ops.len() as u32) as usize]
        });
        // `/= %= ^=` refuse a value with a `&mut self` call on an indexed place (E0421).
        let op = match op {
            Some(BinOp::Div | BinOp::Rem | BinOp::BitXor) if call_in_result => Some(BinOp::Add),
            other => other,
        };
        let index = match self.b.weighted(&[40, 20, 20, 20]) {
            0 => Expr::Gid,
            1 => {
                let k = Expr::u32(self.b.below(1000));
                Expr::bin(
                    BinOp::BitXor,
                    Expr::bin(BinOp::BitXor, Expr::Gid, k.clone()),
                    k,
                )
            }
            2 => Expr::bin(BinOp::Add, Expr::Gid, Expr::u32(0)),
            _ => match self.inputs_of(Ty::U32).first().copied() {
                // `i | (in[i] & 0)`: names only a buffer.
                Some(buf) => Expr::bin(
                    BinOp::BitOr,
                    Expr::Gid,
                    Expr::bin(
                        BinOp::BitAnd,
                        Expr::input(buf, Ty::U32, Expr::Gid),
                        Expr::u32(0),
                    ),
                ),
                None => Expr::Gid,
            },
        };
        // A non-zero start so that `/=`, `%=` and `-=` have something to work on; small enough for `+=` and `*=`.
        let init = self.b.below(32) + 1;
        Some(Store { index, op, init })
    }
}

impl Gen<'_> {
    /// Puts the top-level locals of `body` back in scope (the block drawing popped them), as the result draw needs.
    pub(super) fn restore_scope(&mut self, body: &[Stmt]) {
        for stmt in body {
            if let Stmt::Let {
                var, ty, mutable, ..
            }
            | Stmt::LetMatch {
                var, ty, mutable, ..
            } = stmt
                && !self.scope.iter().any(|(id, ..)| id == var)
            {
                self.scope.push((*var, *ty, *mutable));
            }
        }
    }
}
