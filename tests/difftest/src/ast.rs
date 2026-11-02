//! The AST of the generated kernels: integers, booleans, `f32` (from DIFFTEST_FORMAT 2) and, for the vector programs
//! (DIFFTEST_FORMAT 6, slice m6.d), the `gust::math` vectors and matrices.
//!
//! Every scalar value is a 32-bit pattern (`u32`); `bool` is 0 or 1. Nodes that the interpreter and the emitter need to
//! type carry their operand type, so `Expr::ty` is O(1). `Program::check` verifies the invariants the generator
//! promises, and the unit tests run it over every generated program.
//!
//! The vector types never reach a buffer: inputs and the output are scalar, so the result file, the classifier and the
//! shrinker see scalars only. A vector exists as a local or an intermediate, is built from scalars (`Vec4::new`,
//! `splat`, `Mat4::from_cols`) and is read back through a component, a `dot`, or a swizzle that is read in turn. The
//! aggregate operators have their own nodes (`AggBin`, `AggNeg`, `Dot`, ...), so every pattern that matches `Bin` or
//! `Un` still sees scalars only.

use std::collections::{BTreeMap, BTreeSet};

/// The element type of a vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Lane {
    U32,
    I32,
    F32,
}

impl Lane {
    /// The scalar type of the elements.
    pub fn ty(self) -> Ty {
        match self {
            Lane::U32 => Ty::U32,
            Lane::I32 => Ty::I32,
            Lane::F32 => Ty::F32,
        }
    }
}

/// The widths the program space uses: `gust::math` has 2- and 4-wide vectors and matrices with a buffer layout; the
/// 3-wide types are locals only and are not generated.
pub const WIDTHS: [u8; 2] = [2, 4];

/// The types of the program space. There is no `bool` buffer (`bool` has no `GpuPod`), and no vector or matrix buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Ty {
    U32,
    I32,
    Bool,
    /// `f32` (R1-4 of the M5 plan: the exact-interval subset). A value is the 32-bit IEEE pattern.
    F32,
    /// `Vec2`/`Vec4`, `IVec2`/`IVec4`, `UVec2`/`UVec4`: the element type and the width (2 or 4).
    Vec(Lane, u8),
    /// `Mat2`/`Mat4`: square, `f32`, column-major.
    Mat(u8),
}

impl Ty {
    /// The Rust spelling.
    pub fn name(self) -> &'static str {
        match self {
            Ty::U32 => "u32",
            Ty::I32 => "i32",
            Ty::Bool => "bool",
            Ty::F32 => "f32",
            Ty::Vec(Lane::F32, 2) => "Vec2",
            Ty::Vec(Lane::F32, 4) => "Vec4",
            Ty::Vec(Lane::I32, 2) => "IVec2",
            Ty::Vec(Lane::I32, 4) => "IVec4",
            Ty::Vec(Lane::U32, 2) => "UVec2",
            Ty::Vec(Lane::U32, 4) => "UVec4",
            Ty::Mat(2) => "Mat2",
            Ty::Mat(4) => "Mat4",
            Ty::Vec(..) | Ty::Mat(_) => panic!("a width outside the program space: {self:?}"),
        }
    }

    /// The dialect spelling used for buffer element types (scalars only: no vector or matrix buffer is generated).
    pub fn dialect_name(self) -> &'static str {
        match self {
            Ty::U32 => "uint",
            Ty::I32 => "int",
            Ty::Bool => "bool",
            Ty::F32 => "float",
            Ty::Vec(..) | Ty::Mat(_) => panic!("{} is not a buffer element type", self.name()),
        }
    }

    pub fn is_int(self) -> bool {
        matches!(self, Ty::U32 | Ty::I32)
    }

    /// A vector or a matrix.
    pub fn is_aggregate(self) -> bool {
        matches!(self, Ty::Vec(..) | Ty::Mat(_))
    }

    /// The type of one component: the type itself for a scalar, the lane type for a vector, `f32` for a matrix.
    pub fn scalar(self) -> Ty {
        match self {
            Ty::Vec(lane, _) => lane.ty(),
            Ty::Mat(_) => Ty::F32,
            other => other,
        }
    }

    /// The number of components: 1 for a scalar, `n` for a vector, `n * n` for a matrix.
    pub fn lanes(self) -> usize {
        match self {
            Ty::Vec(_, n) => usize::from(n),
            Ty::Mat(n) => usize::from(n) * usize::from(n),
            _ => 1,
        }
    }
}

/// The result type of the aggregate operator `l op r`, if `gust::math` has the operator for those operand types (the
/// product table, the `Div` impls and the matrix `Add`/`Sub` of `crates/gust/src/math.rs`).
pub fn agg_bin_ty(op: BinOp, l: Ty, r: Ty) -> Option<Ty> {
    let arithmetic = matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div);
    match (l, r) {
        (Ty::Vec(a, n), Ty::Vec(b, m)) if a == b && n == m && arithmetic => Some(l),
        (Ty::Vec(a, _), s) if s == a.ty() && matches!(op, BinOp::Mul | BinOp::Div) => Some(l),
        (s, Ty::Vec(a, _)) if s == a.ty() && op == BinOp::Mul => Some(r),
        (Ty::Mat(n), Ty::Mat(m))
            if n == m && matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul) =>
        {
            Some(l)
        }
        (Ty::Mat(_), Ty::F32) if op == BinOp::Mul => Some(l),
        (Ty::F32, Ty::Mat(_)) if op == BinOp::Mul => Some(r),
        (Ty::Mat(n), Ty::Vec(Lane::F32, m)) if n == m && op == BinOp::Mul => Some(r),
        _ => None,
    }
}

/// The letters of the component fields and swizzle methods, by component index.
pub const COMPONENT_LETTERS: [char; 4] = ['x', 'y', 'z', 'w'];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    /// Arithmetic negation, `i32` and `f32` only (Rust has no `-` on `u32`).
    Neg,
    /// `!`: bitwise on integers, logical on `bool`.
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    /// The right operand is always `u32`.
    Shl,
    /// The right operand is always `u32`; arithmetic on `i32`.
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// Short-circuit, as in Rust.
    And,
    Or,
}

impl BinOp {
    pub const ALL: [BinOp; 18] = [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Rem,
        BinOp::BitAnd,
        BinOp::BitOr,
        BinOp::BitXor,
        BinOp::Shl,
        BinOp::Shr,
        BinOp::Eq,
        BinOp::Ne,
        BinOp::Lt,
        BinOp::Le,
        BinOp::Gt,
        BinOp::Ge,
        BinOp::And,
        BinOp::Or,
    ];

    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::And => "&&",
            BinOp::Or => "||",
        }
    }

    /// Rust's binary operator precedence (higher binds tighter); used by the re-parser of the emitted text.
    pub fn precedence(self) -> u8 {
        match self {
            BinOp::Mul | BinOp::Div | BinOp::Rem => 10,
            BinOp::Add | BinOp::Sub => 9,
            BinOp::Shl | BinOp::Shr => 8,
            BinOp::BitAnd => 7,
            BinOp::BitXor => 6,
            BinOp::BitOr => 5,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => 4,
            BinOp::And => 3,
            BinOp::Or => 2,
        }
    }

    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        )
    }

    pub fn is_shift(self) -> bool {
        matches!(self, BinOp::Shl | BinOp::Shr)
    }

    pub fn is_logical(self) -> bool {
        matches!(self, BinOp::And | BinOp::Or)
    }

    /// The type of `l op r` when the left operand has type `operand`.
    pub fn result_ty(self, operand: Ty) -> Ty {
        if self.is_comparison() || self.is_logical() {
            Ty::Bool
        } else {
            operand
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    /// A literal: the 32-bit pattern. `i32::MIN` has no literal form (see `Expr::i32_min`). An `f32` literal is a
    /// finite value that is not a negative zero.
    Lit(Ty, u32),
    /// A local variable (a `let` binding or a loop variable).
    Var(u32, Ty),
    /// The invocation index `i` (`id.x`), `u32`.
    Gid,
    /// `<input buffer>.len()`, `u32`.
    Len(usize),
    /// `<input buffer>[index]`; `index` is `u32`.
    Input {
        buf: usize,
        ty: Ty,
        index: Box<Expr>,
    },
    /// `ty` is the operand and the result type.
    Un { op: UnOp, ty: Ty, e: Box<Expr> },
    /// `ty` is the type of the left operand; the result type follows `BinOp::result_ty`.
    Bin {
        op: BinOp,
        ty: Ty,
        l: Box<Expr>,
        r: Box<Expr>,
    },
    /// `e as to` among `u32`, `i32` and `f32`. Between `u32` and `i32` the bits are unchanged; to and from `f32` the
    /// value is converted (the interpreter's rules for the cases it accepts are in `interp.rs`).
    Cast { to: Ty, e: Box<Expr> },
    /// `Vec4::new(a, b, c, d)` (`ty` a vector, `args` its scalars in order), or for a matrix `Mat4::from_cols(c0, ..)`
    /// (`rows` false: the columns, each a `Vec` of the matrix size) or `Mat4::from_rows(r0, ..)` (`rows` true).
    Make { ty: Ty, rows: bool, args: Vec<Expr> },
    /// `Vec4::splat(e)`, `e` a scalar of the lane type.
    Splat { ty: Ty, e: Box<Expr> },
    /// `e.x`, `e.y`, ...: one component of a vector (a matrix has no component access).
    Comp { e: Box<Expr>, index: u8 },
    /// `e.xy()`, `e.wzyx()`: the named components as a vector of the word's length (2 or 4 letters).
    Swizzle { e: Box<Expr>, lanes: Vec<u8> },
    /// `l.dot(r)`: two vectors of one type, a scalar of the lane type.
    Dot { l: Box<Expr>, r: Box<Expr> },
    /// `e.transpose()`.
    Transpose { e: Box<Expr> },
    /// `l op r` where at least one operand is a vector or a matrix; `ty` is the result type (`agg_bin_ty`).
    AggBin {
        op: BinOp,
        ty: Ty,
        l: Box<Expr>,
        r: Box<Expr>,
    },
    /// `-e` on a float vector, a signed integer vector or a matrix.
    AggNeg { e: Box<Expr> },
}

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
            Expr::Lit(ty, _) | Expr::Var(_, ty) | Expr::Input { ty, .. } | Expr::Un { ty, .. } => {
                *ty
            }
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
            Expr::Lit(..) | Expr::Var(..) | Expr::Gid | Expr::Len(_) => {}
            Expr::Input { index, .. } => f(index),
            Expr::Un { e, .. }
            | Expr::Cast { e, .. }
            | Expr::Splat { e, .. }
            | Expr::Comp { e, .. }
            | Expr::Swizzle { e, .. }
            | Expr::Transpose { e }
            | Expr::AggNeg { e } => f(e),
            Expr::Bin { l, r, .. } | Expr::Dot { l, r } | Expr::AggBin { l, r, .. } => {
                f(l);
                f(r);
            }
            Expr::Make { args, .. } => args.iter().for_each(f),
        }
    }

    /// [`Expr::for_each_child`] with mutable access.
    pub fn for_each_child_mut(&mut self, f: &mut dyn FnMut(&mut Expr)) {
        match self {
            Expr::Lit(..) | Expr::Var(..) | Expr::Gid | Expr::Len(_) => {}
            Expr::Input { index, .. } => f(index),
            Expr::Un { e, .. }
            | Expr::Cast { e, .. }
            | Expr::Splat { e, .. }
            | Expr::Comp { e, .. }
            | Expr::Swizzle { e, .. }
            | Expr::Transpose { e }
            | Expr::AggNeg { e } => f(e),
            Expr::Bin { l, r, .. } | Expr::Dot { l, r } | Expr::AggBin { l, r, .. } => {
                f(l);
                f(r);
            }
            Expr::Make { args, .. } => args.iter_mut().for_each(f),
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stmt {
    Let {
        var: u32,
        ty: Ty,
        mutable: bool,
        init: Expr,
    },
    Assign {
        var: u32,
        value: Expr,
    },
    /// `v.x = value;`: one component of a mutable vector local (`index` 0 is `x`).
    AssignComp {
        var: u32,
        index: u8,
        value: Expr,
    },
    If {
        cond: Expr,
        then: Vec<Stmt>,
        els: Vec<Stmt>,
    },
    /// `for var in 0u32..trips`, `trips` a literal in `1..=MAX_TRIPS`.
    For {
        var: u32,
        trips: u32,
        body: Vec<Stmt>,
    },
}

/// The largest loop trip count the generator writes.
pub const MAX_TRIPS: u32 = 8;

/// One read-only input buffer; `data` holds one 32-bit pattern per element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputBuf {
    pub ty: Ty,
    pub data: Vec<u32>,
}

/// A generated kernel: `if i < out.len() { <body>; out[i] = <result>; }` over `count` elements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub seed: u64,
    pub attempt: u64,
    pub count: u32,
    pub inputs: Vec<InputBuf>,
    pub body: Vec<Stmt>,
    pub result: Expr,
}

impl Program {
    /// The element type of the output buffer: the result type, with `bool` written as `u32` 0 or 1.
    pub fn out_ty(&self) -> Ty {
        match self.result.ty() {
            Ty::Bool => Ty::U32,
            ty => ty,
        }
    }

    /// Whether any value of the program is an `f32`: an input or the result (a local always descends from one of
    /// them, or from a cast of an integer, which `visit_exprs` sees).
    pub fn has_float(&self) -> bool {
        let mut found = self.inputs.iter().any(|input| input.ty == Ty::F32);
        visit_exprs(self, &mut |e| found |= e.ty().scalar() == Ty::F32);
        found
    }

    /// Whether any expression is a vector or a matrix operation (the vector programs of `Space::Vector`).
    pub fn has_aggregate(&self) -> bool {
        let mut found = false;
        visit_exprs(self, &mut |e| found |= e.ty().is_aggregate());
        found
    }

    /// Whether any expression reads an input buffer (a program that reads none is a constant, not a test).
    pub fn reads_input(&self) -> bool {
        fn block(stmts: &[Stmt]) -> bool {
            stmts.iter().any(|stmt| match stmt {
                Stmt::Let { init, .. } => init.reads_input(),
                Stmt::Assign { value, .. } | Stmt::AssignComp { value, .. } => value.reads_input(),
                Stmt::If { cond, then, els } => cond.reads_input() || block(then) || block(els),
                Stmt::For { body, .. } => block(body),
            })
        }
        block(&self.body) || self.result.reads_input()
    }

    /// Verifies the typing and scoping invariants. `Err` names the first violation.
    pub fn check(&self) -> Result<(), String> {
        if self.count == 0 {
            return Err("count is zero".into());
        }
        if self.inputs.is_empty() || self.inputs.len() > 3 {
            return Err(format!("{} input buffers", self.inputs.len()));
        }
        for (k, input) in self.inputs.iter().enumerate() {
            if input.ty == Ty::Bool {
                return Err(format!("input {k} is a bool buffer"));
            }
            if input.ty.is_aggregate() {
                return Err(format!("input {k} is a vector or matrix buffer"));
            }
            if input.data.len() != self.count as usize {
                return Err(format!("input {k} has {} elements", input.data.len()));
            }
        }
        if self.result.ty().is_aggregate() {
            return Err("the result is a vector or a matrix (the output buffer is scalar)".into());
        }
        let mut cx = Check {
            inputs: self.inputs.iter().map(|input| input.ty).collect(),
            scope: Vec::new(),
            seen: BTreeSet::new(),
        };
        // The body and the result share the guarded block's scope.
        for stmt in &self.body {
            cx.stmt(stmt)?;
        }
        cx.expr(&self.result)
    }
}

struct Check {
    inputs: Vec<Ty>,
    /// (variable, type, mutable), innermost last.
    scope: Vec<(u32, Ty, bool)>,
    seen: BTreeSet<u32>,
}

impl Check {
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
        for stmt in stmts {
            self.stmt(stmt)?;
        }
        self.scope.truncate(mark);
        Ok(())
    }

    fn stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
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
        }
    }

    fn expr(&mut self, expr: &Expr) -> Result<(), String> {
        match expr {
            Expr::Lit(Ty::Bool, bits) if *bits > 1 => Err("bool literal out of range".into()),
            Expr::Lit(Ty::I32, bits) if *bits == i32::MIN as u32 => Err("i32::MIN literal".into()),
            Expr::Lit(Ty::F32, bits) if !f32::from_bits(*bits).is_finite() => {
                Err("non-finite f32 literal".into())
            }
            Expr::Lit(Ty::F32, bits) if *bits == (-0.0f32).to_bits() => {
                Err("negative zero literal".into())
            }
            Expr::Lit(ty, _) if ty.is_aggregate() => Err("a vector or matrix literal".into()),
            Expr::Lit(..) | Expr::Gid => Ok(()),
            Expr::Var(var, ty) => match self.lookup(*var) {
                Some((declared, _)) if declared == *ty => Ok(()),
                Some(_) => Err(format!("v{var} used at the wrong type")),
                None => Err(format!("v{var} used out of scope")),
            },
            Expr::Len(buf) => {
                if *buf < self.inputs.len() {
                    Ok(())
                } else {
                    Err(format!("len of unknown buffer {buf}"))
                }
            }
            Expr::Input { buf, ty, index } => {
                self.expr(index)?;
                if ty.is_aggregate() || self.inputs.get(*buf) != Some(ty) {
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
                    UnOp::Not if ty.is_aggregate() => Err("`!` on a vector or matrix".into()),
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
                let castable = |ty: Ty| ty != Ty::Bool && !ty.is_aggregate();
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
        }
    }
}

/// Calls `f` on every expression node of `program` (pre-order): the body's statements, both branches of every `if`,
/// loop bodies and the result, dead code included.
pub fn visit_exprs(program: &Program, f: &mut dyn FnMut(&Expr)) {
    fn expr(e: &Expr, f: &mut dyn FnMut(&Expr)) {
        f(e);
        e.for_each_child(&mut |child| expr(child, f));
    }
    fn block(stmts: &[Stmt], f: &mut dyn FnMut(&Expr)) {
        for stmt in stmts {
            match stmt {
                Stmt::Let { init, .. } => expr(init, f),
                Stmt::Assign { value, .. } | Stmt::AssignComp { value, .. } => expr(value, f),
                Stmt::If { cond, then, els } => {
                    expr(cond, f);
                    block(then, f);
                    block(els, f);
                }
                Stmt::For { body, .. } => block(body, f),
            }
        }
    }
    block(&program.body, f);
    expr(&program.result, f);
}

/// The number of terms of the maximal `+`/`-` chain that starts at `expr` (1 if `expr` is not an `f32` addition or
/// subtraction, or the same on float vectors or matrices, which add lane by lane). This is the "written terms" count of
/// R1-4; a chain continued through a variable is not counted.
pub fn chain_terms(expr: &Expr) -> usize {
    match expr {
        Expr::Bin {
            op: BinOp::Add | BinOp::Sub,
            ty: Ty::F32,
            l,
            r,
        } => chain_terms(l) + chain_terms(r),
        Expr::AggBin {
            op: BinOp::Add | BinOp::Sub,
            ty,
            l,
            r,
        } if ty.scalar() == Ty::F32 => chain_terms(l) + chain_terms(r),
        _ => 1,
    }
}

/// The counted name of `AssignComp` statements.
pub const ASSIGN_COMP: &str = "assign comp";

/// How many times each operator is used in `program` (for coverage counts in tests and reports).
pub fn operator_counts(program: &Program) -> BTreeMap<String, usize> {
    fn expr(e: &Expr, counts: &mut BTreeMap<String, usize>) {
        let name = match e {
            Expr::Lit(..) | Expr::Var(..) | Expr::Gid | Expr::Len(_) | Expr::Input { .. } => None,
            Expr::Un { op, ty, .. } => Some(match op {
                UnOp::Neg if *ty == Ty::F32 => "-f32".to_owned(),
                UnOp::Neg => "-".to_owned(),
                UnOp::Not => format!("!{}", ty.name()),
            }),
            Expr::Bin { op, ty, .. } => Some(format!("{} {}", op.symbol(), ty.name())),
            Expr::Cast { to, .. } => Some(format!("as {}", to.name())),
            other => other.aggregate_name(),
        };
        if let Some(name) = name {
            *counts.entry(name).or_default() += 1;
        }
        e.for_each_child(&mut |child| expr(child, counts));
    }
    fn block(stmts: &[Stmt], counts: &mut BTreeMap<String, usize>) {
        for stmt in stmts {
            match stmt {
                Stmt::Let { init, .. } => expr(init, counts),
                Stmt::Assign { value, .. } => expr(value, counts),
                Stmt::AssignComp { value, .. } => {
                    *counts.entry(ASSIGN_COMP.to_owned()).or_default() += 1;
                    expr(value, counts);
                }
                Stmt::If { cond, then, els } => {
                    expr(cond, counts);
                    block(then, counts);
                    block(els, counts);
                }
                Stmt::For { body, .. } => {
                    *counts.entry("for".to_owned()).or_default() += 1;
                    block(body, counts);
                }
            }
        }
    }
    let mut counts = BTreeMap::new();
    block(&program.body, &mut counts);
    expr(&program.result, &mut counts);
    counts
}

/// Every type a value can have, for the tables below.
fn value_types() -> Vec<Ty> {
    let mut types = vec![Ty::U32, Ty::I32, Ty::F32];
    for lane in [Lane::U32, Lane::I32, Lane::F32] {
        for n in WIDTHS {
            types.push(Ty::Vec(lane, n));
        }
    }
    types.extend(WIDTHS.map(Ty::Mat));
    types
}

/// The operator names (as `operator_counts` writes them) of every row the generator can write: each `agg_bin_ty` row
/// with the four arithmetic operators, each negation (`i32` and `f32` vectors, matrices) and each swizzle shape
/// (a vector of width 2 or 4 read as a vector of width 2 or 4).
pub fn writable_aggregate_forms() -> BTreeSet<String> {
    let mut forms = BTreeSet::new();
    let types = value_types();
    for op in [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div] {
        for &l in &types {
            for &r in &types {
                if agg_bin_ty(op, l, r).is_some() {
                    forms.insert(format!("{} {} {}", op.symbol(), l.name(), r.name()));
                }
            }
        }
    }
    for &ty in &types {
        match ty {
            Ty::Vec(Lane::I32 | Lane::F32, _) | Ty::Mat(_) => {
                forms.insert(format!("-{}", ty.name()));
            }
            _ => {}
        }
        if let Ty::Vec(..) = ty {
            for width in WIDTHS {
                forms.insert(format!("swizzle {}->{}", ty.name(), width));
            }
        }
    }
    forms
}
