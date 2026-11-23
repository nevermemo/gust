//! Statements and the whole program, with its invariants.

use super::*;

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
    /// `place = value;` (`op` None) or `place op= value;` on a field place or a local, `ty` the type of the place
    /// (an integer type when `op` is set; `bool` has no compound assignment here). Rust evaluates `value` first, then
    /// (for `op=` on primitives) reads the place. (Format 8.)
    Set {
        place: Place,
        ty: Ty,
        op: Option<BinOp>,
        value: Expr,
    },
    /// `recv.m(args);` or `m(args);` with the value, if any, dropped. (Format 8.)
    Call {
        f: usize,
        owner: Option<u8>,
        recv: Option<Place>,
        args: Vec<Expr>,
    },
    /// `match scrut { E0::V0 | E0::V2 => { .. } E0::V1 => { .. } _ => { .. } }` on an enum: the scrutinee is evaluated
    /// once, the first arm whose pattern names the value runs. Without `_` every variant is named once and the last arm
    /// is the `default` of the Slang `switch`. (Format 9.)
    Match {
        scrut: Expr,
        arms: Vec<Arm>,
    },
    /// `return value;` in a helper, the last statement of its block. (Format 9.)
    Return {
        value: Expr,
    },
    /// `let (a, mut b): (u32, S0) = init;`: destructure a tuple into plain names (`vars`: the variable and whether it is
    /// `mut`), the annotation written when `annotate` (always when an element is an enum). `ty` is the tuple type. (Format 10.)
    LetTuple {
        vars: Vec<(u32, bool)>,
        ty: Ty,
        annotate: bool,
        init: Expr,
    },
    /// `let v: ty = match scrut { pat => value, .. };`: the value form of `match`, only as a typed `let` initializer. Each
    /// arm body is a block of statements ending in a `Yield`, a `Return` (helpers) or a `TailMatch`. (Format 10.)
    LetMatch {
        var: u32,
        ty: Ty,
        mutable: bool,
        scrut: Expr,
        arms: Vec<Arm>,
    },
    /// The value of an arm of a value `match`: the last statement of the arm's block, written as the bare expression. (Format 10.)
    Yield {
        value: Expr,
    },
    /// A nested value `match` as the value of an arm: the last statement of the arm's block. (Format 10.)
    TailMatch {
        scrut: Expr,
        arms: Vec<Arm>,
    },
    /// `let cl3 = |p: T, ..| -> R { .. };`: bind a non-capturing closure for the rest of the block. (Format 12.)
    LetClosure {
        def: Box<ClosureDef>,
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

/// The final store of an array program: `out[index as usize] op= result;` (`op` None: `=`), over an output buffer whose
/// elements all start as `init`. The index is `u32` and equals `i` by construction (the validator accepts only an index
/// that names no root the result calls `&mut self` on, E0421), so the stores of different invocations do not overlap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Store {
    pub index: Expr,
    pub op: Option<BinOp>,
    pub init: u32,
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
    /// The structs and functions of a method program (format 8); empty for every other space.
    pub methods: Methods,
    /// The final store of an array program (format 11): `out[index] op= result` in place of `out[i] = result`, over an
    /// output buffer that starts as `init` in every element. `None` is `out[i] = result`.
    pub store: Option<Store>,
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

    /// Whether the kernel itself calls a method or an associated function, in an expression or as a statement (a
    /// method program whose kernel calls nothing, or only free helpers, tests nothing new).
    pub fn has_call(&self) -> bool {
        fn block(methods: &Methods, stmts: &[Stmt]) -> bool {
            stmts.iter().any(|stmt| match stmt {
                Stmt::Call { f, .. } => is_method_or_assoc(methods, *f),
                Stmt::If { then, els, .. } => block(methods, then) || block(methods, els),
                Stmt::For { body, .. } => block(methods, body),
                Stmt::Match { arms, .. }
                | Stmt::LetMatch { arms, .. }
                | Stmt::TailMatch { arms, .. } => arms.iter().any(|arm| block(methods, &arm.body)),
                _ => false,
            })
        }
        fn is_method_or_assoc(methods: &Methods, f: usize) -> bool {
            methods.fns[f].recv != Recv::None || methods.fns[f].owner.is_some()
        }
        let methods = &self.methods;
        let mut found = block(methods, &self.body);
        let mut look = |e: &Expr| {
            found |= matches!(e, Expr::Call { f, .. } if is_method_or_assoc(methods, *f));
        };
        walk_stmts(&self.body, &mut look);
        walk_expr(&self.result, &mut look);
        found
    }

    /// Whether the kernel or a function holds a statement `match` (an enum program without one tests nothing new).
    pub fn has_match(&self) -> bool {
        fn block(stmts: &[Stmt]) -> bool {
            stmts.iter().any(|stmt| match stmt {
                Stmt::Match { .. } => true,
                Stmt::If { then, els, .. } => block(then) || block(els),
                Stmt::For { body, .. } => block(body),
                _ => false,
            })
        }
        block(&self.body) || self.methods.fns.iter().any(|def| block(&def.body))
    }

    /// Whether the program, when it runs, calls a trait method and a generic function, and one generic function with
    /// two different types bound to its first type parameter (a trait program without these tests nothing new). A
    /// program outside the domain passes: its discard reason is the domain one.
    pub fn has_trait_forms(&self) -> bool {
        let Ok((_, coverage)) = crate::interp::run_covered(self) else {
            return true;
        };
        let ran = |name: &str| coverage.operators.get(name).copied().unwrap_or(0) > 0;
        ran(TRAIT_CALL) && ran(GENERIC_CALL) && coverage.two_instantiations()
    }

    /// Whether the program, when it runs, executes a math method and reads a module constant (a math program without
    /// both tests nothing new). A program outside the domain passes: its discard reason is the domain one.
    pub fn has_math_forms(&self) -> bool {
        let Ok((_, coverage)) = crate::interp::run_covered(self) else {
            return true;
        };
        let ran = |prefix: &str| {
            coverage
                .operators
                .keys()
                .any(|name| name.starts_with(prefix))
        };
        let math_method = MathFn::ALL.iter().any(|f| {
            [Ty::U32, Ty::I32, Ty::F32]
                .iter()
                .any(|ty| ran(&f.counted_name(*ty)))
        });
        math_method && ran("const ")
    }

    /// Whether the program binds a closure and calls one (a closure program without both tests nothing new).
    pub fn has_closure(&self) -> bool {
        let mut bound = false;
        walk_all_stmts(&self.body, &mut |s| {
            bound |= matches!(s, Stmt::LetClosure { .. });
        });
        for def in &self.methods.fns {
            walk_all_stmts(&def.body, &mut |s| {
                bound |= matches!(s, Stmt::LetClosure { .. });
            });
        }
        let mut called = false;
        visit_exprs(self, &mut |e| {
            called |= matches!(e, Expr::ClosureCall { .. })
        });
        bound && called
    }

    /// Whether any expression or statement is an array form (an array program without one tests nothing new).
    pub fn has_array(&self) -> bool {
        let mut found = self.store.is_some();
        visit_exprs(self, &mut |e| {
            found |= e.ty().is_array()
                || matches!(e, Expr::Read { place, .. } | Expr::ArrLen { place } if place.index.is_some());
        });
        found
    }

    /// Whether the program holds a value `match` or a tuple (a value program without either tests nothing new).
    pub fn has_value_form(&self) -> bool {
        fn block(stmts: &[Stmt]) -> bool {
            stmts.iter().any(|stmt| match stmt {
                Stmt::LetMatch { .. } | Stmt::LetTuple { .. } => true,
                Stmt::If { then, els, .. } => block(then) || block(els),
                Stmt::For { body, .. } => block(body),
                Stmt::Match { arms, .. } => arms.iter().any(|arm| block(&arm.body)),
                _ => false,
            })
        }
        let mut tuple = false;
        visit_exprs(self, &mut |e| {
            tuple |= matches!(e, Expr::Tuple { .. } | Expr::TupleField { .. }) || e.ty().is_tuple();
        });
        tuple || block(&self.body) || self.methods.fns.iter().any(|def| block(&def.body))
    }

    /// Whether any expression reads an input buffer (a program that reads none is a constant, not a test).
    pub fn reads_input(&self) -> bool {
        fn block(stmts: &[Stmt]) -> bool {
            stmts.iter().any(|stmt| match stmt {
                Stmt::Let { init, .. } | Stmt::LetTuple { init, .. } => init.reads_input(),
                Stmt::Assign { value, .. }
                | Stmt::AssignComp { value, .. }
                | Stmt::Yield { value }
                | Stmt::Return { value } => value.reads_input(),
                Stmt::Set { place, value, .. } => {
                    place.index.as_deref().is_some_and(Expr::reads_input) || value.reads_input()
                }
                Stmt::Call { recv, args, .. } => {
                    recv.as_ref()
                        .and_then(|p| p.index.as_deref())
                        .is_some_and(Expr::reads_input)
                        || args.iter().any(Expr::reads_input)
                }
                Stmt::If { cond, then, els } => cond.reads_input() || block(then) || block(els),
                Stmt::For { body, .. } => block(body),
                Stmt::Match { scrut, arms }
                | Stmt::LetMatch { scrut, arms, .. }
                | Stmt::TailMatch { scrut, arms } => {
                    scrut.reads_input() || arms.iter().any(|arm| block(&arm.body))
                }
                Stmt::LetClosure { .. } => false,
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
            if input.ty.is_aggregate() || input.ty.is_struct() || input.ty.is_enum() {
                return Err(format!(
                    "input {k} is a vector, matrix, struct or enum buffer"
                ));
            }
            if input.data.len() != self.count as usize {
                return Err(format!("input {k} has {} elements", input.data.len()));
            }
        }
        if self.result.ty().is_aggregate()
            || self.result.ty().is_struct()
            || self.result.ty().is_enum()
            || self.result.ty().is_array()
        {
            return Err(
                "the result is a vector, a matrix, a struct or an enum (the output buffer is scalar)"
                    .into(),
            );
        }
        self.methods.check_tables()?;
        let mut cx = Check {
            inputs: self.inputs.iter().map(|input| input.ty).collect(),
            methods: &self.methods,
            scope: Vec::new(),
            seen: BTreeSet::new(),
            in_fn: None,
            in_closure: false,
            closures: Vec::new(),
            closure_params: Vec::new(),
        };
        // Functions first (their variables are numbered in the program's one id space), each in a scope of its own.
        for (k, def) in self.methods.fns.iter().enumerate() {
            cx.function(k, def)?;
        }
        cx.in_fn = None;
        cx.scope.clear();
        cx.closures.clear();
        // The body and the result share the guarded block's scope.
        for stmt in &self.body {
            cx.stmt(stmt)?;
        }
        cx.expr(&self.result)?;
        if let Some(store) = &self.store {
            if self.result.ty() == Ty::Bool {
                return Err("an output store of a bool result".into());
            }
            cx.expr(&store.index)?;
            if store.index.ty() != Ty::U32 {
                return Err("the output store index is not u32".into());
            }
            if let Some(op) = store.op {
                let allowed = matches!(
                    op,
                    BinOp::Add
                        | BinOp::Sub
                        | BinOp::Mul
                        | BinOp::Div
                        | BinOp::Rem
                        | BinOp::BitAnd
                        | BinOp::BitOr
                        | BinOp::BitXor
                );
                if !allowed || !matches!(self.result.ty(), Ty::U32 | Ty::I32) {
                    return Err(format!(
                        "compound `{}=` on the output of {}",
                        op.symbol(),
                        self.result.ty().name()
                    ));
                }
            }
        }
        Ok(())
    }
}
