//! The user-item tables of a program (`impl Methods`): constants, calls and signatures.

use super::*;

impl Methods {
    /// Whether `e`, the initializer of `K<k>`, is a constant expression of D210 over the type of the item: literals,
    /// the constants below `k`, unary `-` (`i32`) and `!`, and the integer binary operators; a shift amount is a `u32`.
    fn const_init(&self, k: usize, e: &Expr) -> Result<(), String> {
        let ty = self.consts[k].ty;
        match e {
            Expr::Lit(lit, bits) if *lit == ty && (ty == Ty::U32 || *bits as i32 != i32::MIN) => {
                Ok(())
            }
            Expr::Const { k: j, ty: used }
                if usize::from(*j) < k && *used == self.consts[usize::from(*j)].ty =>
            {
                Ok(())
            }
            Expr::Un {
                op,
                ty: t,
                e: inner,
            } if *t == ty => match (op, ty) {
                (UnOp::Neg, Ty::U32) => Err(format!("K{k}: `-` on a u32")),
                (UnOp::Neg, _) if matches!(**inner, Expr::Lit(..)) => {
                    Err(format!("K{k}: a negated literal is not folded"))
                }
                _ => self.const_init(k, inner),
            },
            Expr::Bin { op, ty: t, l, r }
                if *t == ty && !op.is_comparison() && !op.is_logical() =>
            {
                if op.is_shift() && r.ty() != Ty::U32 {
                    return Err(format!("K{k}: a shift amount that is not a u32"));
                }
                self.const_init(k, l)?;
                if op.is_shift() {
                    match &**r {
                        Expr::Lit(Ty::U32, _) | Expr::Const { ty: Ty::U32, .. } => {}
                        _ => {
                            return Err(format!(
                                "K{k}: a shift amount that is no literal or constant"
                            ));
                        }
                    }
                    if let Expr::Const { k: j, .. } = &**r
                        && usize::from(*j) >= k
                    {
                        return Err(format!("K{k}: a constant below it"));
                    }
                    Ok(())
                } else {
                    self.const_init(k, r)
                }
            }
            other => Err(format!("K{k}: {other:?} is not a constant expression")),
        }
    }

    /// An array type of a parameter, a result or a local: 1 to `MAX_ARRAY` elements of an element that exists, at most
    /// `MAX_ARRAY_FLAT` scalars in all.
    pub fn check_array(&self, a: ArrTy) -> Result<(), String> {
        if !(1..=MAX_ARRAY).contains(&usize::from(a.n)) {
            return Err(format!("an array of {} elements", a.n));
        }
        match a.e {
            Elem::Struct(j) if usize::from(j) >= self.structs.len() => {
                return Err(format!("an array element of unknown S{j}"));
            }
            Elem::Enum(k) if usize::from(k) >= self.enums.len() => {
                return Err(format!("an array element of unknown E{k}"));
            }
            _ => {}
        }
        if self.ty_len(Ty::Array(a)) > MAX_ARRAY_FLAT {
            return Err("an array of more than 16 scalars".into());
        }
        Ok(())
    }
}

impl Methods {
    /// A tuple type of a parameter, a result or a local: 2 to 4 elements that exist, at most `MAX_TUPLE_FLAT` scalars.
    pub fn check_tuple(&self, t: TupleTy) -> Result<(), String> {
        if !(2..=MAX_TUPLE).contains(&usize::from(t.n)) {
            return Err(format!("a tuple of {} elements", t.n));
        }
        for e in t.elems() {
            match e {
                Elem::Struct(j) if usize::from(*j) >= self.structs.len() => {
                    return Err(format!("a tuple element of unknown S{j}"));
                }
                Elem::Enum(k) if usize::from(*k) >= self.enums.len() => {
                    return Err(format!("a tuple element of unknown E{k}"));
                }
                _ => {}
            }
        }
        if self.ty_len(Ty::Tuple(t)) > MAX_TUPLE_FLAT {
            return Err("a tuple of more than 16 scalars".into());
        }
        Ok(())
    }

    /// The structural rules of the tables: struct fields refer to earlier structs, sizes are bounded, every function has
    /// a consistent receiver and return type, and calls only go downwards (checked with the bodies).
    pub fn check_tables(&self) -> Result<(), String> {
        for (k, def) in self.consts.iter().enumerate() {
            if !matches!(def.ty, Ty::U32 | Ty::I32) || def.init.ty() != def.ty {
                return Err(format!("K{k} is not an integer of its own type"));
            }
            self.const_init(k, &def.init)?;
            // Rust refuses a constant whose evaluation overflows, divides by zero or shifts by 32 or more.
            if let Err(violation) = crate::interp::const_value(self, k) {
                return Err(format!("K{k} does not evaluate: {}", violation.label()));
            }
        }
        if self.structs.len() > MAX_STRUCTS {
            return Err(format!("{} structs", self.structs.len()));
        }
        if self.enums.len() > MAX_ENUMS {
            return Err(format!("{} enums", self.enums.len()));
        }
        for (k, def) in self.enums.iter().enumerate() {
            if !matches!(def.repr, Ty::U32 | Ty::I32) {
                return Err(format!("E{k} has the repr {}", def.repr.name()));
            }
            if !(2..=MAX_VARIANTS).contains(&def.variants.len()) {
                return Err(format!("E{k} has {} variants", def.variants.len()));
            }
            let mut seen = BTreeSet::new();
            for (v, &(bits, explicit)) in def.variants.iter().enumerate() {
                let value = def.value(v as u8);
                if !explicit {
                    // A variant without `= value` counts up from the one before (the first is 0).
                    let expected = if v == 0 {
                        0
                    } else {
                        def.value(v as u8 - 1) + 1
                    };
                    if value != expected {
                        return Err(format!(
                            "E{k}::V{v} is {value}, counting up gives {expected}"
                        ));
                    }
                }
                if !seen.insert(bits) {
                    return Err(format!("E{k} has the value {value} twice"));
                }
            }
        }
        for (j, def) in self.structs.iter().enumerate() {
            if def.fields.is_empty() || def.fields.len() > 3 {
                return Err(format!("struct S{j} has {} fields", def.fields.len()));
            }
            for ty in &def.fields {
                match ty {
                    // A GPU struct is a `GpuPod` type: its fields are `u32`, `i32` (no `bool`) or an earlier struct.
                    Ty::U32 | Ty::I32 => {}
                    Ty::Struct(inner) if usize::from(*inner) < j => {}
                    other => return Err(format!("S{j} has a field of {}", other.name())),
                }
            }
            if self.flat_len(j as u8) > MAX_FLAT {
                return Err(format!("S{j} is {} scalars", self.flat_len(j as u8)));
            }
        }
        for (k, def) in self.fns.iter().enumerate() {
            if let Some(owner) = def.owner
                && usize::from(owner) >= self.structs.len()
            {
                return Err(format!("m{k} is a function of an unknown struct"));
            }
            if def.recv != Recv::None && def.owner.is_none() && def.imp.is_none() {
                return Err(format!("m{k} has a receiver and no struct"));
            }
            for (_, ty) in &def.params {
                match ty {
                    Ty::U32 | Ty::I32 | Ty::Bool => {}
                    Ty::Struct(j) if usize::from(*j) < self.structs.len() => {}
                    Ty::Enum(e) if usize::from(*e) < self.enums.len() => {}
                    Ty::Tuple(t) => self.check_tuple(*t)?,
                    Ty::Array(a) => self.check_array(*a)?,
                    Ty::Param(_) => {}
                    other => return Err(format!("m{k} has a parameter of {}", other.name())),
                }
            }
            if let Some(Ty::Enum(e)) = def.ret
                && usize::from(e) >= self.enums.len()
            {
                return Err(format!("m{k} returns an unknown enum"));
            }
            if let Some(Ty::Array(a)) = def.ret {
                self.check_array(a)?;
                if def.recv != Recv::None || def.owner.is_some() {
                    return Err(format!("m{k}: only a helper returns an array"));
                }
            }
            if let Some(Ty::Tuple(t)) = def.ret {
                self.check_tuple(t)?;
            }
            match (def.recv, def.owner, def.ret) {
                (Recv::Shared | Recv::Value, _, Some(Ty::Struct(_))) if def.imp.is_some() => {}
                (Recv::None, None, Some(Ty::Param(_))) => {}
                (Recv::None, Some(owner), ret) if ret != Some(Ty::Struct(owner)) => {
                    return Err(format!("associated m{k} does not return its struct"));
                }
                (
                    Recv::None,
                    None,
                    Some(Ty::U32 | Ty::I32 | Ty::Bool | Ty::Enum(_) | Ty::Tuple(_) | Ty::Array(_)),
                ) => {}
                (Recv::None, None, _) => return Err(format!("helper m{k} has no scalar result")),
                (
                    Recv::Shared | Recv::Value,
                    _,
                    Some(Ty::U32 | Ty::I32 | Ty::Bool | Ty::Enum(_) | Ty::Tuple(_) | Ty::Array(_)),
                ) => {}
                (Recv::Shared | Recv::Value, _, _) => {
                    return Err(format!("method m{k} has no scalar result"));
                }
                (
                    Recv::Mut,
                    _,
                    None
                    | Some(Ty::U32 | Ty::I32 | Ty::Bool | Ty::Enum(_) | Ty::Tuple(_) | Ty::Array(_)),
                ) => {}
                (Recv::Mut, _, _) => return Err(format!("method m{k} returns a struct")),
                _ => {}
            }
            if def.ret.is_some() != def.tail.is_some() {
                return Err(format!("m{k}: the tail and the result type disagree"));
            }
        }
        self.check_traits()?;
        Ok(())
    }

    /// Format 13: the traits, their impls and the generic functions. A trait method is implemented by exactly two types,
    /// the same two for every method of the trait, with the signature the trait gives (`Self` as the type); a `&mut self`
    /// method only by structs; an impl is no generic function; a type parameter has one or two module traits as bounds.
    fn check_traits(&self) -> Result<(), String> {
        if self.traits.len() > 3 {
            return Err(format!("{} traits", self.traits.len()));
        }
        for (tr, def) in self.traits.iter().enumerate() {
            if def.methods.is_empty() || def.methods.len() > 2 {
                return Err(format!("Tr{tr} has {} methods", def.methods.len()));
            }
            for (m, method) in def.methods.iter().enumerate() {
                if method.recv == Recv::None {
                    return Err(format!("Tr{tr}::m{m} has no receiver"));
                }
                if method
                    .params
                    .iter()
                    .any(|p| !matches!(p, Ty::U32 | Ty::I32 | Ty::Bool))
                {
                    return Err(format!("Tr{tr}::m{m} has a non-scalar parameter"));
                }
                match (method.recv, method.ret) {
                    (Recv::Mut, TRet::Unit | TRet::Ty(_)) => {}
                    (Recv::Mut, TRet::SelfTy) => {
                        return Err("a `&mut self` method returns Self".into());
                    }
                    (_, TRet::Unit) => return Err(format!("Tr{tr}::m{m} returns nothing")),
                    (Recv::Shared, TRet::SelfTy) => {
                        return Err("a `&self` method returns Self".into());
                    }
                    _ => {}
                }
                if let TRet::Ty(t) = method.ret
                    && !matches!(t, Ty::U32 | Ty::I32 | Ty::Bool)
                {
                    return Err(format!("Tr{tr}::m{m} returns {}", t.name()));
                }
                let types: Vec<Ty> = self
                    .fns
                    .iter()
                    .filter_map(|f| f.imp.filter(|i| i.tr as usize == tr && i.m as usize == m))
                    .map(|i| i.ty)
                    .collect();
                if types.len() != 2 || types[0] == types[1] {
                    return Err(format!("Tr{tr}::m{m} has {} impls", types.len()));
                }
                if m > 0 && types != self.implementors(tr as u8) {
                    return Err(format!("the impls of Tr{tr} differ between its methods"));
                }
                for ty in types {
                    match ty {
                        Ty::Struct(j) if usize::from(j) < self.structs.len() => {}
                        Ty::Enum(e) if usize::from(e) < self.enums.len() => {
                            if method.recv != Recv::Value {
                                return Err("an enum impl whose receiver is not `self`".into());
                            }
                        }
                        other => return Err(format!("an impl for {}", other.name())),
                    }
                }
            }
        }
        for (k, def) in self.fns.iter().enumerate() {
            if let Some(imp) = def.imp {
                let Some(method) = self
                    .traits
                    .get(usize::from(imp.tr))
                    .and_then(|t| t.methods.get(usize::from(imp.m)))
                else {
                    return Err(format!("m{k} implements an unknown trait method"));
                };
                let ret = match method.ret {
                    TRet::Unit => None,
                    TRet::Ty(t) => Some(t),
                    TRet::SelfTy => Some(imp.ty),
                };
                let owner = match imp.ty {
                    Ty::Struct(j) => Some(j),
                    _ => None,
                };
                if def.recv != method.recv
                    || def.ret != ret
                    || def.owner != owner
                    || !def.gens.is_empty()
                    || def.params.iter().map(|p| p.1).collect::<Vec<_>>() != method.params
                {
                    return Err(format!(
                        "m{k} does not match the signature of its trait method"
                    ));
                }
            }
            if def.gens.len() > 2 {
                return Err(format!("m{k} has {} type parameters", def.gens.len()));
            }
            for (g, gen_param) in def.gens.iter().enumerate() {
                if gen_param.bounds.is_empty() || gen_param.bounds.len() > 2 {
                    return Err(format!(
                        "m{k}: a type parameter with {} bounds",
                        gen_param.bounds.len()
                    ));
                }
                let mut seen = BTreeSet::new();
                for &b in &gen_param.bounds {
                    if usize::from(b) >= self.traits.len() || !seen.insert(b) {
                        return Err(format!("m{k}: a bound that is not a trait, or twice"));
                    }
                }
                if !def.params.iter().any(|p| p.1 == Ty::Param(g as u8)) {
                    return Err(format!("m{k}: nothing determines the type parameter {g}"));
                }
            }
            let in_range = |ty: Ty| match ty {
                Ty::Param(g) => usize::from(g) < def.gens.len(),
                _ => true,
            };
            if !def.params.iter().all(|p| in_range(p.1)) || !def.ret.is_none_or(in_range) {
                return Err(format!("m{k} names a type parameter it does not have"));
            }
        }
        Ok(())
    }
}
