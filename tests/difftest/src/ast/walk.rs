//! Walks over the AST and the operator counts.

use super::*;

/// Calls `f` on `e` and every node below it (pre-order).
pub fn walk_expr(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    f(e);
    e.for_each_child(&mut |child| walk_expr(child, f));
}

/// Calls `f` on every expression node of a statement list (pre-order): both branches of every `if`, loop bodies.
pub fn walk_stmts(stmts: &[Stmt], f: &mut dyn FnMut(&Expr)) {
    for stmt in stmts {
        match stmt {
            Stmt::Let { init, .. } | Stmt::LetTuple { init, .. } => walk_expr(init, f),
            Stmt::Assign { value, .. }
            | Stmt::AssignComp { value, .. }
            | Stmt::Yield { value }
            | Stmt::Return { value } => walk_expr(value, f),
            Stmt::Set { place, value, .. } => {
                if let Some(index) = place.index.as_deref() {
                    walk_expr(index, f);
                }
                walk_expr(value, f);
            }
            Stmt::Call { recv, args, .. } => {
                if let Some(index) = recv.as_ref().and_then(|p| p.index.as_deref()) {
                    walk_expr(index, f);
                }
                args.iter().for_each(|arg| walk_expr(arg, f));
            }
            Stmt::If { cond, then, els } => {
                walk_expr(cond, f);
                walk_stmts(then, f);
                walk_stmts(els, f);
            }
            Stmt::For { body, .. } => walk_stmts(body, f),
            Stmt::LetClosure { def } => {
                walk_stmts(&def.body, f);
                walk_expr(&def.tail, f);
            }
            Stmt::Match { scrut, arms }
            | Stmt::LetMatch { scrut, arms, .. }
            | Stmt::TailMatch { scrut, arms } => {
                walk_expr(scrut, f);
                for arm in arms {
                    walk_stmts(&arm.body, f);
                }
            }
        }
    }
}

/// Calls `f` on every statement of a statement list (pre-order), closure bodies and match arms included.
pub fn walk_all_stmts(stmts: &[Stmt], f: &mut dyn FnMut(&Stmt)) {
    for stmt in stmts {
        f(stmt);
        match stmt {
            Stmt::If { then, els, .. } => {
                walk_all_stmts(then, f);
                walk_all_stmts(els, f);
            }
            Stmt::For { body, .. } => walk_all_stmts(body, f),
            Stmt::LetClosure { def } => walk_all_stmts(&def.body, f),
            Stmt::Match { arms, .. }
            | Stmt::LetMatch { arms, .. }
            | Stmt::TailMatch { arms, .. } => {
                for arm in arms {
                    walk_all_stmts(&arm.body, f);
                }
            }
            _ => {}
        }
    }
}

/// [`walk_stmts`] over a function: its body, then its tail.
pub fn walk_fn_exprs(def: &FnDef, f: &mut dyn FnMut(&Expr)) {
    walk_stmts(&def.body, f);
    if let Some(tail) = &def.tail {
        walk_expr(tail, f);
    }
}

/// Calls `f` on every expression node of `program` (pre-order): the body's statements, both branches of every `if`,
/// loop bodies and the result, dead code included, then the body and tail of every function in table order. (The
/// reducer's node numbering is this order.)
pub fn visit_exprs(program: &Program, f: &mut dyn FnMut(&Expr)) {
    walk_stmts(&program.body, f);
    walk_expr(&program.result, f);
    for def in &program.methods.fns {
        walk_fn_exprs(def, f);
    }
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
    register_closures(program);
    fn expr(methods: &Methods, e: &Expr, counts: &mut BTreeMap<String, usize>) {
        let name = match e {
            Expr::Lit(..) | Expr::Var(..) | Expr::Gid | Expr::Len(_) | Expr::Input { .. } => None,
            Expr::Const { ty, .. } => Some(format!("const {}", ty.name())),
            Expr::Math { f, ty, .. } => Some(f.counted_name(*ty)),
            Expr::Read { place, .. } => place.root_ty.is_tuple().then(|| TUPLE_READ.to_owned()),
            Expr::Tuple { .. } | Expr::TupleField { .. } => None,
            Expr::Un { op, ty, .. } => Some(match op {
                UnOp::Neg if *ty == Ty::F32 => "-f32".to_owned(),
                UnOp::Neg => "-".to_owned(),
                UnOp::Not => format!("!{}", ty.name()),
            }),
            Expr::Bin { op, ty, .. } => Some(format!("{} {}", op.symbol(), ty.name())),
            Expr::Cast { to, .. } => Some(format!("as {}", to.name())),
            Expr::Call { f, .. } => {
                let def = &methods.fns[*f];
                Some(call_name(def.recv, def.owner).to_owned())
            }
            Expr::StructLit { .. } => Some(STRUCT_LIT.to_owned()),
            Expr::EnumAs { to, .. } => Some(enum_as_name(*to)),
            Expr::Variant { .. } => None,
            other => other.aggregate_name(),
        };
        if let Some(name) = name {
            *counts.entry(name).or_default() += 1;
        }
        let shape = match e {
            Expr::Bin { op, l, r, .. } => bin_shape(methods, *op, l, r),
            Expr::Call { f, recv, args, .. } => {
                if methods.fns[*f].recv == Recv::Mut {
                    *counts.entry(SHAPE_EXPR_CALL.to_owned()).or_default() += 1;
                }
                call_shape(methods, *f, recv, args)
            }
            _ => None,
        };
        if let Some(shape) = shape {
            *counts.entry(shape.to_owned()).or_default() += 1;
        }
        if let Expr::Call { f, recv, args, .. } = e {
            for name in trait_call_names(methods, *f, recv, args) {
                *counts.entry(name.to_owned()).or_default() += 1;
            }
        }
        if let Expr::PathCall { form, .. } = e {
            let name = if *form == PathForm::Trait {
                TRAIT_PATH_CALL
            } else {
                TRAIT_PATH_QUALIFIED
            };
            *counts.entry(name.to_owned()).or_default() += 1;
        }
        if let Expr::Call { f, args, .. } = e {
            for shape in enum_call_shapes(methods, args) {
                *counts.entry(shape.to_owned()).or_default() += 1;
            }
            for name in call_arg_names(methods.fns[*f].recv != Recv::None, args) {
                *counts.entry(name.to_owned()).or_default() += 1;
            }
        }
        match e {
            Expr::Tuple { elems, .. } => {
                for name in tuple_lit_names(methods, elems) {
                    *counts.entry(name.to_owned()).or_default() += 1;
                }
            }
            Expr::TupleField { e: inner, .. } => {
                for name in tuple_field_names(methods, inner) {
                    *counts.entry(name.to_owned()).or_default() += 1;
                }
            }
            _ => {}
        }
        for name in array_expr_names(methods, e) {
            *counts.entry(name.to_owned()).or_default() += 1;
        }
        if let Expr::ClosureCall { var, args, .. } = e {
            for name in closure_call_names(methods, *var, args) {
                *counts.entry(name.to_owned()).or_default() += 1;
            }
        }
        if let Expr::Call { recv, args, .. } = e {
            for name in array_call_names(recv)
                .into_iter()
                .chain(array_arg_names(args))
            {
                *counts.entry(name.to_owned()).or_default() += 1;
            }
        }
        e.for_each_child(&mut |child| expr(methods, child, counts));
    }
    fn block(
        methods: &Methods,
        stmts: &[Stmt],
        in_for: bool,
        nested: bool,
        counts: &mut BTreeMap<String, usize>,
    ) {
        for stmt in stmts {
            match stmt {
                Stmt::Let { init, .. } => {
                    for name in array_value_names(init) {
                        *counts.entry(name.to_owned()).or_default() += 1;
                    }
                    expr(methods, init, counts);
                }
                Stmt::LetTuple { annotate, init, .. } => {
                    for name in destruct_names(methods, *annotate, init) {
                        *counts.entry(name.to_owned()).or_default() += 1;
                    }
                    expr(methods, init, counts);
                }
                Stmt::Yield { value } => expr(methods, value, counts),
                Stmt::LetMatch { scrut, arms, .. } | Stmt::TailMatch { scrut, arms } => {
                    let tail = matches!(stmt, Stmt::TailMatch { .. });
                    for name in value_match_names(methods, scrut, arms, tail) {
                        *counts.entry(name.to_owned()).or_default() += 1;
                    }
                    expr(methods, scrut, counts);
                    for arm in arms {
                        block(methods, &arm.body, in_for, true, counts);
                    }
                }
                Stmt::Assign { value, .. } => {
                    for name in array_value_names(value) {
                        *counts.entry(name.to_owned()).or_default() += 1;
                    }
                    expr(methods, value, counts);
                }
                Stmt::AssignComp { value, .. } => {
                    *counts.entry(ASSIGN_COMP.to_owned()).or_default() += 1;
                    expr(methods, value, counts);
                }
                Stmt::Set {
                    place, op, value, ..
                } => {
                    *counts.entry(set_name(*op)).or_default() += 1;
                    if let Some(shape) = set_shape(methods, place, *op, value) {
                        *counts.entry(shape).or_default() += 1;
                    }
                    for name in array_set_names(methods, place, *op, value) {
                        *counts.entry(name.to_owned()).or_default() += 1;
                    }
                    for name in closure_set_names(methods, place, *op, value) {
                        *counts.entry(name.to_owned()).or_default() += 1;
                    }
                    if let Some(index) = place.index.as_deref() {
                        expr(methods, index, counts);
                    }
                    for name in tuple_set_names(methods, place, *op, value) {
                        *counts.entry(name.to_owned()).or_default() += 1;
                    }
                    expr(methods, value, counts);
                }
                Stmt::Call { f, recv, args, .. } => {
                    let def = &methods.fns[*f];
                    for name in trait_call_names(methods, *f, recv, args) {
                        *counts.entry(name.to_owned()).or_default() += 1;
                    }
                    if def.recv == Recv::Mut {
                        *counts.entry(SHAPE_STMT_CALL.to_owned()).or_default() += 1;
                    }
                    if let Some(shape) = call_shape(methods, *f, recv, args) {
                        *counts.entry(shape.to_owned()).or_default() += 1;
                    }
                    *counts
                        .entry(call_name(def.recv, def.owner).to_owned())
                        .or_default() += 1;
                    for shape in enum_call_shapes(methods, args) {
                        *counts.entry(shape.to_owned()).or_default() += 1;
                    }
                    for name in call_arg_names(def.recv != Recv::None, args) {
                        *counts.entry(name.to_owned()).or_default() += 1;
                    }
                    for name in array_call_names(recv)
                        .into_iter()
                        .chain(array_arg_names(args))
                    {
                        *counts.entry(name.to_owned()).or_default() += 1;
                    }
                    if let Some(index) = recv.as_ref().and_then(|p| p.index.as_deref()) {
                        expr(methods, index, counts);
                    }
                    args.iter().for_each(|arg| expr(methods, arg, counts));
                }
                Stmt::If { cond, then, els } => {
                    expr(methods, cond, counts);
                    block(methods, then, in_for, nested, counts);
                    block(methods, els, in_for, nested, counts);
                }
                Stmt::For { body, .. } => {
                    *counts.entry("for".to_owned()).or_default() += 1;
                    block(methods, body, true, nested, counts);
                }
                Stmt::Match { scrut, arms } => {
                    for name in match_names(arms, in_for, nested) {
                        *counts.entry(name.to_owned()).or_default() += 1;
                    }
                    if let Some(shape) = match_shape(methods, scrut, arms) {
                        *counts.entry(shape.to_owned()).or_default() += 1;
                    }
                    expr(methods, scrut, counts);
                    for arm in arms {
                        block(methods, &arm.body, in_for, true, counts);
                    }
                }
                Stmt::Return { value } => {
                    *counts.entry("return".to_owned()).or_default() += 1;
                    expr(methods, value, counts);
                }
                Stmt::LetClosure { def } => {
                    block(methods, &def.body, in_for, nested, counts);
                    expr(methods, &def.tail, counts);
                }
            }
        }
    }
    let mut counts = BTreeMap::new();
    let methods = &program.methods;
    block(methods, &program.body, false, false, &mut counts);
    expr(methods, &program.result, &mut counts);
    if let Some(store) = &program.store {
        expr(methods, &store.index, &mut counts);
        for name in array_store_names(methods, &program.result) {
            *counts.entry(name.to_owned()).or_default() += 1;
        }
    }
    for def in &methods.fns {
        block(methods, &def.body, false, false, &mut counts);
        if let Some(tail) = &def.tail {
            expr(methods, tail, &mut counts);
        }
    }
    closure_site_counts(program, &mut counts);
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
