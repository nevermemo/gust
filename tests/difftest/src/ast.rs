//! The scalar AST of the generated kernels (integers, booleans and, from DIFFTEST_FORMAT 2, `f32`).
//!
//! Every value is a 32-bit pattern (`u32`); `bool` is 0 or 1. Nodes that the interpreter and the emitter need to
//! type carry their operand type, so `Expr::ty` is O(1). `Program::check` verifies the invariants the generator
//! promises, and the unit tests run it over every generated program.

use std::collections::{BTreeMap, BTreeSet};

/// The scalar types of the program space. There is no `bool` buffer (`bool` has no `GpuPod`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Ty {
    U32,
    I32,
    Bool,
    /// `f32` (R1-4 of the M5 plan: the exact-interval subset). A value is the 32-bit IEEE pattern.
    F32,
}

impl Ty {
    /// The Rust spelling.
    pub fn name(self) -> &'static str {
        match self {
            Ty::U32 => "u32",
            Ty::I32 => "i32",
            Ty::Bool => "bool",
            Ty::F32 => "f32",
        }
    }

    /// The dialect spelling used for buffer element types.
    pub fn dialect_name(self) -> &'static str {
        match self {
            Ty::U32 => "uint",
            Ty::I32 => "int",
            Ty::Bool => "bool",
            Ty::F32 => "float",
        }
    }

    pub fn is_int(self) -> bool {
        matches!(self, Ty::U32 | Ty::I32)
    }

    pub fn is_float(self) -> bool {
        self == Ty::F32
    }
}

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
        }
    }

    /// The number of nodes.
    pub fn size(&self) -> usize {
        1 + match self {
            Expr::Lit(..) | Expr::Var(..) | Expr::Gid | Expr::Len(_) => 0,
            Expr::Input { index, .. } => index.size(),
            Expr::Un { e, .. } | Expr::Cast { e, .. } => e.size(),
            Expr::Bin { l, r, .. } => l.size() + r.size(),
        }
    }

    pub fn reads_input(&self) -> bool {
        match self {
            Expr::Input { .. } => true,
            Expr::Lit(..) | Expr::Var(..) | Expr::Gid | Expr::Len(_) => false,
            Expr::Un { e, .. } | Expr::Cast { e, .. } => e.reads_input(),
            Expr::Bin { l, r, .. } => l.reads_input() || r.reads_input(),
        }
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
        visit_exprs(self, &mut |e| found |= e.ty() == Ty::F32);
        found
    }

    /// Whether any expression reads an input buffer (a program that reads none is a constant, not a test).
    pub fn reads_input(&self) -> bool {
        fn block(stmts: &[Stmt]) -> bool {
            stmts.iter().any(|stmt| match stmt {
                Stmt::Let { init, .. } => init.reads_input(),
                Stmt::Assign { value, .. } => value.reads_input(),
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
            if input.data.len() != self.count as usize {
                return Err(format!("input {k} has {} elements", input.data.len()));
            }
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
                if self.inputs.get(*buf) != Some(ty) {
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
                    _ => Ok(()),
                }
            }
            Expr::Bin { op, ty, l, r } => {
                self.expr(l)?;
                self.expr(r)?;
                if l.ty() != *ty {
                    return Err("binary left operand type mismatch".into());
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
                let castable = |ty: Ty| ty != Ty::Bool;
                if !castable(*to) || !castable(e.ty()) || e.ty() == *to {
                    return Err("cast outside u32, i32 and f32".into());
                }
                Ok(())
            }
        }
    }
}

/// Calls `f` on every expression node of `program` (pre-order): the body's statements, both branches of every `if`,
/// loop bodies and the result, dead code included.
pub fn visit_exprs(program: &Program, f: &mut dyn FnMut(&Expr)) {
    fn expr(e: &Expr, f: &mut dyn FnMut(&Expr)) {
        f(e);
        match e {
            Expr::Lit(..) | Expr::Var(..) | Expr::Gid | Expr::Len(_) => {}
            Expr::Input { index, .. } => expr(index, f),
            Expr::Un { e, .. } | Expr::Cast { e, .. } => expr(e, f),
            Expr::Bin { l, r, .. } => {
                expr(l, f);
                expr(r, f);
            }
        }
    }
    fn block(stmts: &[Stmt], f: &mut dyn FnMut(&Expr)) {
        for stmt in stmts {
            match stmt {
                Stmt::Let { init, .. } => expr(init, f),
                Stmt::Assign { value, .. } => expr(value, f),
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
/// subtraction). This is the "written terms" count of R1-4; a chain continued through a variable is not counted.
pub fn chain_terms(expr: &Expr) -> usize {
    match expr {
        Expr::Bin {
            op: BinOp::Add | BinOp::Sub,
            ty: Ty::F32,
            l,
            r,
        } => chain_terms(l) + chain_terms(r),
        _ => 1,
    }
}

/// How many times each operator is used in `program` (for coverage counts in tests and reports).
pub fn operator_counts(program: &Program) -> BTreeMap<String, usize> {
    fn expr(e: &Expr, counts: &mut BTreeMap<String, usize>) {
        match e {
            Expr::Lit(..) | Expr::Var(..) | Expr::Gid | Expr::Len(_) => {}
            Expr::Input { index, .. } => expr(index, counts),
            Expr::Un { op, ty, e } => {
                let name = match op {
                    UnOp::Neg if *ty == Ty::F32 => "-f32".to_owned(),
                    UnOp::Neg => "-".to_owned(),
                    UnOp::Not => format!("!{}", ty.name()),
                };
                *counts.entry(name).or_default() += 1;
                expr(e, counts);
            }
            Expr::Bin { op, ty, l, r } => {
                *counts
                    .entry(format!("{} {}", op.symbol(), ty.name()))
                    .or_default() += 1;
                expr(l, counts);
                expr(r, counts);
            }
            Expr::Cast { to, e } => {
                *counts.entry(format!("as {}", to.name())).or_default() += 1;
                expr(e, counts);
            }
        }
    }
    fn block(stmts: &[Stmt], counts: &mut BTreeMap<String, usize>) {
        for stmt in stmts {
            match stmt {
                Stmt::Let { init, .. } => expr(init, counts),
                Stmt::Assign { value, .. } => expr(value, counts),
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
