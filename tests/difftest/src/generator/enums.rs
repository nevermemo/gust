//! The enum space: enum declarations, expressions and matches.

use super::*;

/// The enum space (format 9, slice m9.d1b): the enum tables, enum values and the statement `match`.
impl Gen<'_> {
    /// One to two enums of two to five variants, a `u32` or `i32` repr, literal discriminants: an explicit value from a
    /// list of boundaries, or the count-up of the macro (the previous value plus one). Values are distinct.
    pub(super) fn gen_enums(&mut self) {
        const UNSIGNED: [u32; 13] = [
            0,
            1,
            2,
            3,
            5,
            7,
            10,
            100,
            255,
            65535,
            0x7FFF_FFFF,
            0x8000_0000,
            0xFFFF_FFF0,
        ];
        const SIGNED: [i32; 13] = [
            -100,
            -7,
            -2,
            -1,
            0,
            1,
            2,
            5,
            10,
            255,
            65535,
            i32::MAX - 3,
            i32::MIN + 5,
        ];
        let count = 1 + self.b.below(MAX_ENUMS as u32) as usize;
        for _ in 0..count {
            let repr = if self.b.chance(50) { Ty::U32 } else { Ty::I32 };
            let n = 2 + self.b.below(MAX_VARIANTS as u32 - 1) as usize;
            let mut variants: Vec<(u32, bool)> = Vec::new();
            for _ in 0..n {
                let up = match variants.last() {
                    None => Some(0u32),
                    Some(&(p, _)) if repr == Ty::I32 => (p as i32).checked_add(1).map(|v| v as u32),
                    Some(&(p, _)) => p.checked_add(1),
                };
                let taken = |v: u32, variants: &[(u32, bool)]| variants.iter().any(|x| x.0 == v);
                if !self.b.chance(45)
                    && let Some(v) = up
                    && !taken(v, &variants)
                {
                    variants.push((v, false));
                    continue;
                }
                let start = self.b.below(13) as usize;
                let value = (0..13)
                    .map(|k| {
                        let pick = (start + k) % 13;
                        if repr == Ty::I32 {
                            SIGNED[pick] as u32
                        } else {
                            UNSIGNED[pick]
                        }
                    })
                    .find(|&v| !taken(v, &variants))
                    .expect("thirteen values, at most four taken");
                variants.push((value, true));
            }
            self.table.enums.push(EnumDef { repr, variants });
        }
    }

    pub(super) fn enum_ty(&mut self) -> Ty {
        Ty::Enum(self.b.below(self.table.enums.len() as u32) as u8)
    }

    /// A variable of the enum type or one of its variants.
    pub(super) fn enum_leaf(&mut self, ty: Ty) -> Expr {
        let Ty::Enum(e) = ty else {
            unreachable!("an enum type")
        };
        let vars = self.vars(ty);
        if !vars.is_empty() && self.b.chance(60) {
            return Expr::Var(vars[self.b.below(vars.len() as u32) as usize], ty);
        }
        let v = self
            .b
            .below(self.table.enums[usize::from(e)].variants.len() as u32) as u8;
        Expr::Variant { e, v }
    }

    /// An enum value comes from a variant path, a variable (a local or a parameter) or the result of a call, never from
    /// an integer.
    pub(super) fn enum_expr(&mut self, ty: Ty, depth: u32) -> Expr {
        if self.closures
            && depth > 0
            && !self.cls.is_empty()
            && self.b.chance(25)
            && let Some(call) = self.closure_call(ty, depth - 1, None)
        {
            return call;
        }
        if self.methods
            && depth > 0
            && self.b.chance(35)
            && let Some(call) = self.value_call(ty, depth - 1)
        {
            return call;
        }
        self.enum_leaf(ty)
    }

    /// A root with a writable struct place that no enclosing call is borrowing.
    pub(super) fn pick_writable_root(&mut self) -> Option<Root> {
        let mut roots: Vec<Root> = Vec::new();
        for (place, _, writable) in self.struct_places() {
            if writable && !self.forbid.contains(&place.root) && !roots.contains(&place.root) {
                roots.push(place.root);
            }
        }
        if roots.is_empty() {
            return None;
        }
        Some(roots[self.b.below(roots.len() as u32) as usize])
    }

    /// An enum argument: often a `&mut self` call on a root shared with the other enum arguments of the call.
    pub(super) fn enum_arg(&mut self, ty: Ty, depth: u32, shared: &mut Option<Root>) -> Expr {
        if depth >= 2 && self.b.chance(85) {
            let root = match *shared {
                Some(root) => Some(root),
                None => self.pick_writable_root(),
            };
            if let Some(root) = root
                && let Some(call) = self.root_call(ty, root, depth - 1)
            {
                *shared = Some(root);
                return call;
            }
        }
        self.enum_expr(ty, depth)
    }

    /// `==` or `!=` of two enum values; often two `&mut self` calls on one root (the left one runs first).
    pub(super) fn enum_compare(&mut self, d: u32) -> Expr {
        let ty = self.enum_ty();
        let op = if self.b.chance(50) {
            BinOp::Eq
        } else {
            BinOp::Ne
        };
        if self.b.chance(75)
            && let Some(root) = self.pick_writable_root()
            && let Some(l) = self.root_call(ty, root, 2)
            && let Some(r) = self.root_call(ty, root, 2)
        {
            return Expr::bin(op, l, r);
        }
        let l = self.enum_expr(ty, d);
        let r = self.enum_expr(ty, d);
        Expr::bin(op, l, r)
    }

    /// A block ending in `return value` of the function's result type, in a helper, `percent` of the time.
    pub(super) fn maybe_return(&mut self, body: &mut Vec<Stmt>, percent: u32) {
        if !self.enums {
            return;
        }
        // The value space returns from inside an arm twice as often (m9.d1b re-check, nit R1).
        let percent = if self.values { percent * 2 } else { percent };
        if let Some(ret) = self.cur_ret
            && self.b.chance(percent)
        {
            let value = self.expr(ret, 3);
            body.push(Stmt::Return { value });
        }
    }

    /// A statement `match` on an enum: the variants are shuffled and cut into groups (grouped arms); `_` replaces the
    /// last group 45 % of the time, otherwise every variant is named and the last arm is the `default` of the Slang
    /// `switch`. The scrutinee is a variable, a call (often a `&mut self` call on a root that an arm uses), or a path.
    pub(super) fn match_stmt(&mut self, depth: u32) -> Stmt {
        let e = self.b.below(self.table.enums.len() as u32) as u8;
        let ty = Ty::Enum(e);
        let n = self.table.enums[usize::from(e)].variants.len();
        let mut bias_root = None;
        let mut scrut = None;
        match self.b.weighted(&[30, 45, 18, 7]) {
            1 => {
                if let Some(root) = self.pick_writable_root()
                    && let Some(call) = self.root_call(ty, root, 3)
                {
                    bias_root = Some(root);
                    scrut = Some(call);
                }
            }
            2 => scrut = self.value_call(ty, 2),
            3 => {
                let v = self.b.below(n as u32) as u8;
                scrut = Some(Expr::Variant { e, v });
            }
            _ => {}
        }
        let scrut = match scrut {
            Some(scrut) => scrut,
            None => self.enum_leaf(ty),
        };
        let scrut = self.enum_scrutinee(e, scrut);
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
        let mut arms = Vec::new();
        for pat in pats {
            let mut body = self.block(depth + 1, 0, 2);
            // An arm that uses the root the scrutinee's call mutated: the call's effect is what the arm sees.
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
            self.maybe_return(&mut body, 30);
            arms.push(Arm { pat, body });
        }
        Stmt::Match { scrut, arms }
    }
}
