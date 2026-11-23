//! The expression constructors and queries (`impl Expr`).

use super::*;

/// The counted name of a struct literal.
pub const STRUCT_LIT: &str = "struct literal";

/// The most structs a method program declares, and the most scalars a struct holds once nested structs are flattened.
pub const MAX_STRUCTS: usize = 3;
pub const MAX_FLAT: usize = 8;
/// The most enums an enum program declares, and the most variants of one.
pub const MAX_ENUMS: usize = 2;
pub const MAX_VARIANTS: usize = 5;

impl Expr {
    pub fn u32(value: u32) -> Expr {
        Expr::Lit(Ty::U32, value)
    }

    /// An `i32` literal. `i32::MIN` has no literal in the emitted dialect, so it is refused here; use `i32_min`.
    pub fn i32(value: i32) -> Expr {
        assert!(value != i32::MIN, "i32::MIN has no literal form");
        Expr::Lit(Ty::I32, value as u32)
    }

    /// An `f32` literal. A negative zero and a non-finite value have no literal here.
    pub fn f32(value: f32) -> Expr {
        assert!(value.is_finite(), "non-finite f32 literal");
        assert!(
            value.to_bits() != (-0.0f32).to_bits(),
            "negative zero has no literal form"
        );
        Expr::Lit(Ty::F32, value.to_bits())
    }

    pub fn bool(value: bool) -> Expr {
        Expr::Lit(Ty::Bool, u32::from(value))
    }

    /// `i32::MIN` as `-2147483647i32 - 1i32` (the exact value, no overflow), the only spelling the emitter uses.
    pub fn i32_min() -> Expr {
        Expr::bin(BinOp::Sub, Expr::i32(i32::MIN + 1), Expr::i32(1))
    }

    pub fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
        let ty = l.ty();
        Expr::Bin {
            op,
            ty,
            l: Box::new(l),
            r: Box::new(r),
        }
    }

    /// `-e`. A negated literal is folded into the literal (the emitted text is the same, and the re-parser reads
    /// `-5i32` as one literal), so the AST never holds `Un(Neg, Lit)`.
    pub fn negated(e: Expr) -> Expr {
        match e {
            Expr::Lit(Ty::I32, bits) if bits as i32 != i32::MIN => {
                return Expr::i32(-(bits as i32));
            }
            // `-0.0` stays an operation: the literal form of a negative zero does not exist.
            Expr::Lit(Ty::F32, bits) if f32::from_bits(bits) != 0.0 => {
                return Expr::f32(-f32::from_bits(bits));
            }
            _ => {}
        }
        Expr::Un {
            op: UnOp::Neg,
            ty: e.ty(),
            e: Box::new(e),
        }
    }

    pub fn inverted(e: Expr) -> Expr {
        let ty = e.ty();
        Expr::Un {
            op: UnOp::Not,
            ty,
            e: Box::new(e),
        }
    }

    pub fn cast(to: Ty, e: Expr) -> Expr {
        Expr::Cast { to, e: Box::new(e) }
    }

    pub fn input(buf: usize, ty: Ty, index: Expr) -> Expr {
        Expr::Input {
            buf,
            ty,
            index: Box::new(index),
        }
    }

    pub fn ty(&self) -> Ty {
        match self {
            Expr::PathCall { call, .. } => call.ty(),
            Expr::Lit(ty, _) | Expr::Var(_, ty) | Expr::Input { ty, .. } | Expr::Un { ty, .. } => {
                *ty
            }
            Expr::Read { ty, .. }
            | Expr::Math { ty, .. }
            | Expr::Const { ty, .. }
            | Expr::Call { ty, .. }
            | Expr::ClosureCall { ty, .. }
            | Expr::StructLit { ty, .. }
            | Expr::Tuple { ty, .. }
            | Expr::ArrayLit { ty, .. }
            | Expr::Repeat { ty, .. } => *ty,
            Expr::ArrLen { .. } => Ty::U32,
            Expr::TupleField { e, index } => match e.ty() {
                Ty::Tuple(t) => t.elems()[usize::from(*index)].ty(),
                other => panic!("a tuple field of {}", other.name()),
            },
            Expr::Variant { e, .. } => Ty::Enum(*e),
            Expr::EnumAs { to, .. } => *to,
            Expr::Gid | Expr::Len(_) => Ty::U32,
            Expr::Bin { op, ty, .. } => op.result_ty(*ty),
            Expr::Cast { to, .. } => *to,
            Expr::Make { ty, .. } | Expr::Splat { ty, .. } | Expr::AggBin { ty, .. } => *ty,
            Expr::Comp { e, .. } | Expr::Dot { l: e, .. } => e.ty().scalar(),
            Expr::Swizzle { e, lanes } => Ty::Vec(
                match e.ty() {
                    Ty::Vec(lane, _) => lane,
                    other => panic!("a swizzle of {}", other.name()),
                },
                lanes.len() as u8,
            ),
            Expr::Transpose { e } | Expr::AggNeg { e } => e.ty(),
        }
    }

    /// `Vec4::new(..)` or `Vec4::splat(..)`-style construction of a vector from its scalars.
    pub fn vec_new(ty: Ty, args: Vec<Expr>) -> Expr {
        Expr::Make {
            ty,
            rows: false,
            args,
        }
    }

    /// `l op r` on aggregates; panics if `gust::math` has no such operator (use `agg_bin_ty` to ask).
    pub fn agg_bin(op: BinOp, l: Expr, r: Expr) -> Expr {
        let ty = agg_bin_ty(op, l.ty(), r.ty()).unwrap_or_else(|| {
            panic!(
                "no `{}` on {} and {}",
                op.symbol(),
                l.ty().name(),
                r.ty().name()
            )
        });
        Expr::AggBin {
            op,
            ty,
            l: Box::new(l),
            r: Box::new(r),
        }
    }

    pub fn comp(e: Expr, index: u8) -> Expr {
        Expr::Comp {
            e: Box::new(e),
            index,
        }
    }

    pub fn swizzle(e: Expr, lanes: Vec<u8>) -> Expr {
        Expr::Swizzle {
            e: Box::new(e),
            lanes,
        }
    }

    pub fn dot(l: Expr, r: Expr) -> Expr {
        Expr::Dot {
            l: Box::new(l),
            r: Box::new(r),
        }
    }

    /// Calls `f` on each direct child, in the order every traversal of the program uses (the order of
    /// `visit_exprs`, which the reducer's node numbering relies on).
    pub fn for_each_child(&self, f: &mut dyn FnMut(&Expr)) {
        match self {
            Expr::PathCall { call, .. } => f(call),
            Expr::Lit(..)
            | Expr::Var(..)
            | Expr::Gid
            | Expr::Len(_)
            | Expr::Variant { .. }
            | Expr::Const { .. } => {}
            Expr::Math { args, .. } => args.iter().for_each(f),
            Expr::Input { index, .. } => f(index),
            Expr::Un { e, .. }
            | Expr::Cast { e, .. }
            | Expr::EnumAs { e, .. }
            | Expr::Splat { e, .. }
            | Expr::Comp { e, .. }
            | Expr::Swizzle { e, .. }
            | Expr::Transpose { e }
            | Expr::AggNeg { e }
            | Expr::TupleField { e, .. } => f(e),
            Expr::Bin { l, r, .. } | Expr::Dot { l, r } | Expr::AggBin { l, r, .. } => {
                f(l);
                f(r);
            }
            Expr::Make { args, .. } => args.iter().for_each(f),
            Expr::Read { place, .. } | Expr::ArrLen { place } => {
                if let Some(i) = &place.index {
                    f(i);
                }
            }
            Expr::Call { recv, args, .. } => {
                if let Some(i) = recv.as_ref().and_then(|p| p.index.as_deref()) {
                    f(i);
                }
                args.iter().for_each(f);
            }
            Expr::StructLit { fields, .. } => fields.iter().for_each(f),
            Expr::ClosureCall { args, .. } => args.iter().for_each(f),
            Expr::Tuple { elems, .. } | Expr::ArrayLit { elems, .. } => elems.iter().for_each(f),
            Expr::Repeat { e, .. } => f(e),
        }
    }

    /// [`Expr::for_each_child`] with mutable access.
    pub fn for_each_child_mut(&mut self, f: &mut dyn FnMut(&mut Expr)) {
        match self {
            Expr::PathCall { call, .. } => f(call),
            Expr::Lit(..)
            | Expr::Var(..)
            | Expr::Gid
            | Expr::Len(_)
            | Expr::Variant { .. }
            | Expr::Const { .. } => {}
            Expr::Math { args, .. } => args.iter_mut().for_each(f),
            Expr::Input { index, .. } => f(index),
            Expr::Un { e, .. }
            | Expr::Cast { e, .. }
            | Expr::EnumAs { e, .. }
            | Expr::Splat { e, .. }
            | Expr::Comp { e, .. }
            | Expr::Swizzle { e, .. }
            | Expr::Transpose { e }
            | Expr::AggNeg { e }
            | Expr::TupleField { e, .. } => f(e),
            Expr::Bin { l, r, .. } | Expr::Dot { l, r } | Expr::AggBin { l, r, .. } => {
                f(l);
                f(r);
            }
            Expr::Make { args, .. } => args.iter_mut().for_each(f),
            Expr::Read { place, .. } | Expr::ArrLen { place } => {
                if let Some(i) = place.index.as_deref_mut() {
                    f(i);
                }
            }
            Expr::Call { recv, args, .. } => {
                if let Some(i) = recv.as_mut().and_then(|p| p.index.as_deref_mut()) {
                    f(i);
                }
                args.iter_mut().for_each(f);
            }
            Expr::StructLit { fields, .. } => fields.iter_mut().for_each(f),
            Expr::ClosureCall { args, .. } => args.iter_mut().for_each(f),
            Expr::Tuple { elems, .. } | Expr::ArrayLit { elems, .. } => {
                elems.iter_mut().for_each(f)
            }
            Expr::Repeat { e, .. } => f(e),
        }
    }

    /// The number of nodes.
    pub fn size(&self) -> usize {
        let mut total = 1;
        self.for_each_child(&mut |child| total += child.size());
        total
    }

    pub fn reads_input(&self) -> bool {
        if matches!(self, Expr::Input { .. }) {
            return true;
        }
        let mut found = false;
        self.for_each_child(&mut |child| found |= child.reads_input());
        found
    }

    /// Whether the expression reads an `f32` input buffer directly: an `f32` `Input` node that no cast sits above.
    /// A read behind a cast (`(in1[i] * 0u32) as f32`) can fold to a constant, so it does not count.
    pub fn reads_f32_input_directly(&self) -> bool {
        match self {
            Expr::Input { ty, .. } => *ty == Ty::F32,
            Expr::Cast { .. } => false,
            _ => {
                let mut found = false;
                self.for_each_child(&mut |child| found |= child.reads_f32_input_directly());
                found
            }
        }
    }

    /// The counted name of a vector or matrix operator (`new Vec4`, `* Mat4 Vec4`, `.xy Vec4`, ...), the same string in
    /// `operator_counts` and in the interpreter's coverage; `None` for every other node.
    pub fn aggregate_name(&self) -> Option<String> {
        Some(match self {
            Expr::Make {
                ty: ty @ Ty::Mat(_),
                rows,
                ..
            } => format!(
                "{} {}",
                if *rows { "from_rows" } else { "from_cols" },
                ty.name()
            ),
            Expr::Make { ty, .. } => format!("new {}", ty.name()),
            Expr::Splat { ty, .. } => format!("splat {}", ty.name()),
            Expr::Comp { e, .. } => format!("comp {}", e.ty().name()),
            Expr::Swizzle { e, lanes } => format!("swizzle {}->{}", e.ty().name(), lanes.len()),
            Expr::Dot { l, .. } => format!("dot {}", l.ty().name()),
            Expr::Transpose { e } => format!("transpose {}", e.ty().name()),
            Expr::AggBin { op, l, r, .. } => {
                format!("{} {} {}", op.symbol(), l.ty().name(), r.ty().name())
            }
            Expr::AggNeg { e } => format!("-{}", e.ty().name()),
            _ => return None,
        })
    }
}
