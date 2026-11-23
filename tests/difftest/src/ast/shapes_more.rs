//! The counted construct names and shape rules for arrays, closures and match forms.

use super::*;

/// Names of the constructs of the array space (format 11, slice m9.d1d), counted statically by `operator_counts` and when
/// executed by the interpreter, under the same names.
pub const ARRAY_LIT: &str = "array literal";
pub const ARRAY_REPEAT: &str = "array repeat";
pub const ARRAY_READ: &str = "array element read";
pub const ARRAY_LEN: &str = "array len";
pub const ARRAY_SET: &str = "array element set";
pub const ARRAY_SET_OP: &str = "array element compound set";
pub const ARRAY_COPY: &str = "array copy";
pub const ARRAY_ARG: &str = "array argument";
pub const ARRAY_ELEM_CALL: &str = "array element method call";
pub const ARRAY_STORE: &str = "array output store";
pub const SHAPE_ARRAY_LIT_ROOT: &str = "shape array literal of &mut self calls on one root";
pub const SHAPE_ARRAY_REPEAT_CALL: &str = "shape array repeat of a &mut self call";
pub const SHAPE_ARRAY_INDEX_CALLS: &str =
    "shape array reads indexed by &mut self calls on one root";
pub const SHAPE_ARRAY_SET_CALL: &str =
    "shape array element set with a &mut self call on the array in the value";
pub const SHAPE_ARRAY_STORE_CALL: &str =
    "shape array output store with a &mut self call in the value";

/// The roots of the `&mut self` calls inside the index of an indexed read in `e`.
fn index_call_roots(methods: &Methods, e: &Expr) -> Vec<Root> {
    let mut roots = Vec::new();
    walk_expr(e, &mut |n| {
        if let Expr::Read { place, .. } = n
            && let Some(index) = place.index.as_deref()
        {
            mutating_roots(methods, index, &mut roots);
        }
    });
    roots
}

/// `a[f()] .. a[g()]`: the two operands of a binary both read an element through an index that calls `&mut self` on
/// one root; the left index must run first.
pub(super) fn array_index_shape(methods: &Methods, l: &Expr, r: &Expr) -> Option<&'static str> {
    let (left, right) = (index_call_roots(methods, l), index_call_roots(methods, r));
    left.iter()
        .any(|root| right.contains(root))
        .then_some(SHAPE_ARRAY_INDEX_CALLS)
}

/// The names an expression node counts under besides its operator: the array constructs.
pub fn array_expr_names(methods: &Methods, e: &Expr) -> Vec<&'static str> {
    match e {
        Expr::ArrayLit { elems, .. } => {
            let mut names = vec![ARRAY_LIT];
            let mut roots = Vec::new();
            for elem in elems {
                mutating_roots(methods, elem, &mut roots);
            }
            if roots
                .iter()
                .enumerate()
                .any(|(k, root)| roots[..k].contains(root))
            {
                names.push(SHAPE_ARRAY_LIT_ROOT);
            }
            names.extend(closure_array_lit_shape(methods, elems));
            names
        }
        Expr::Repeat { e: inner, .. } => {
            let mut names = vec![ARRAY_REPEAT];
            let mut roots = Vec::new();
            mutating_roots(methods, inner, &mut roots);
            if !roots.is_empty() {
                names.push(SHAPE_ARRAY_REPEAT_CALL);
            }
            names
        }
        Expr::Read { place, .. } if place.index.is_some() => vec![ARRAY_READ],
        Expr::ArrLen { .. } => vec![ARRAY_LEN],
        _ => Vec::new(),
    }
}

/// The names an assignment to an element of an array counts under besides `set_name`.
pub fn array_set_names(
    methods: &Methods,
    place: &Place,
    op: Option<BinOp>,
    value: &Expr,
) -> Vec<&'static str> {
    if place.index.is_none() {
        return Vec::new();
    }
    let mut names = vec![if op.is_some() {
        ARRAY_SET_OP
    } else {
        ARRAY_SET
    }];
    if contains_mut_call_on(methods, value, place.root) {
        names.push(SHAPE_ARRAY_SET_CALL);
    }
    names
}

/// The names a call counts under when its receiver is an element of an array.
pub fn array_call_names(recv: &Option<Place>) -> Vec<&'static str> {
    match recv {
        Some(place) if place.index.is_some() => vec![ARRAY_ELEM_CALL],
        _ => Vec::new(),
    }
}

/// The names the arguments of a call and the initializer of a binding count under: an array copied by value.
pub fn array_value_names(e: &Expr) -> Vec<&'static str> {
    if e.ty().is_array() && matches!(e, Expr::Var(..)) {
        vec![ARRAY_COPY]
    } else {
        Vec::new()
    }
}

/// The names a call's arguments count under: an array passed by value.
pub fn array_arg_names(args: &[Expr]) -> Vec<&'static str> {
    if args.iter().any(|arg| arg.ty().is_array()) {
        vec![ARRAY_ARG]
    } else {
        Vec::new()
    }
}

/// The names the final store counts under: itself, and the shape when the result calls `&mut self`.
pub fn array_store_names(methods: &Methods, result: &Expr) -> Vec<&'static str> {
    let mut names = vec![ARRAY_STORE];
    let mut roots = Vec::new();
    mutating_roots(methods, result, &mut roots);
    if !roots.is_empty() {
        names.push(SHAPE_ARRAY_STORE_CALL);
    }
    names.extend(closure_store_names(methods, result));
    names
}

/// Names of the closure constructs and shapes (format 12, slice m9.d1e), counted statically by `operator_counts` and when
/// executed by the interpreter, under the same names.
pub const CLOSURE_LET: &str = "closure let";
pub const CLOSURE_CALL: &str = "closure call";
pub const CLOSURE_IN_KERNEL: &str = "closure bound in the kernel";
pub const CLOSURE_IN_HELPER: &str = "closure bound in a helper";
pub const CLOSURE_IN_METHOD: &str = "closure bound in a method";
pub const CLOSURE_BODY_HELPER: &str = "closure body calls a helper";
pub const CLOSURE_BODY_METHOD: &str = "closure body calls a method";
pub const CLOSURE_MUT_ARRAY: &str = "closure takes a mut array";
pub const CLOSURE_MUT_STRUCT: &str = "closure takes a mut struct";
pub const CLOSURE_ARRAY_RESULT: &str = "closure returns an array";
pub const SHAPE_CLOSURE_ARGS_ROOT: &str = "shape closure arguments built from calls on one root";
pub const SHAPE_CLOSURE_ARRAY_LIT: &str =
    "shape closure call in an array literal beside a call on its root";
pub const SHAPE_CLOSURE_SET_OP: &str =
    "shape closure call in a compound set value on the place's root";
pub const SHAPE_CLOSURE_SET_INDEX: &str =
    "shape closure call in an indexed set value on the array's root";
pub const SHAPE_CLOSURE_STORE: &str = "shape closure call in the output store's value";
pub const SHAPE_CLOSURE_TWICE: &str =
    "shape closure called twice with a side effect in an argument";
pub const SHAPE_CLOSURE_MUT_PARAM: &str =
    "shape closure mutates its parameter, caller variable passed";
pub const CLOSURE_PARAM_METHOD: &str = "closure method call on a struct parameter";

/// The roots of the `&mut self` calls in the arguments of the closure calls inside `e` (each call's arguments run
/// before its body, left to right).
fn closure_arg_roots(methods: &Methods, e: &Expr) -> Vec<Root> {
    let mut roots = Vec::new();
    walk_expr(e, &mut |n| {
        if let Expr::ClosureCall { args, .. } = n {
            for arg in args {
                mutating_roots(methods, arg, &mut roots);
            }
        }
    });
    roots
}

/// Each closure call inside `e` with the roots its arguments call `&mut self` on.
fn closure_calls_with_roots(methods: &Methods, e: &Expr) -> Vec<(u32, Vec<Root>)> {
    let mut calls = Vec::new();
    walk_expr(e, &mut |n| {
        if let Expr::ClosureCall { var, args, .. } = n {
            let mut roots = Vec::new();
            for arg in args {
                mutating_roots(methods, arg, &mut roots);
            }
            calls.push((*var, roots));
        }
    });
    calls
}

/// What the order shapes need to know about a closure's definition (slice m9.d1e, fix round 1): `weighted[k]` is true
/// when parameter `k` is an integer that enters an integer result with a weight of its own (`(p & 255) * (2j + 1)`,
/// exclusive-or'ed into the tail), so swapping two such arguments changes the result; `writes[k]` is true when the body
/// writes the aggregate parameter `k`.
#[derive(Clone, Debug, Default)]
pub struct ClosureInfo {
    pub weighted: Vec<bool>,
    pub writes: Vec<bool>,
}

thread_local! {
    static CLOSURE_INFO: std::cell::RefCell<BTreeMap<u32, ClosureInfo>> = const {
        std::cell::RefCell::new(BTreeMap::new())
    };
}

/// The terms of a chain of `^` (the shape the generator folds a tail into).
fn xor_terms<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    if let Expr::Bin {
        op: BinOp::BitXor,
        l,
        r,
        ..
    } = e
    {
        xor_terms(l, out);
        xor_terms(r, out);
    } else {
        out.push(e);
    }
}

/// Whether `term` is `(var & 255) * w`, possibly cast.
fn is_weighted_term(term: &Expr, var: u32) -> bool {
    let term = match term {
        Expr::Cast { e, .. } => e,
        other => other,
    };
    matches!(term, Expr::Bin { op: BinOp::Mul, l, .. }
        if matches!(&**l, Expr::Bin { op: BinOp::BitAnd, l: inner, .. }
            if matches!(&**inner, Expr::Var(v, _) if *v == var)))
}

/// The order facts about a closure definition.
pub fn closure_info(methods: &Methods, def: &ClosureDef) -> ClosureInfo {
    let mut terms = Vec::new();
    if matches!(def.ret, Ty::U32 | Ty::I32) {
        xor_terms(&def.tail, &mut terms);
    }
    ClosureInfo {
        weighted: def
            .params
            .iter()
            .map(|p| {
                matches!(p.1, Ty::U32 | Ty::I32) && terms.iter().any(|t| is_weighted_term(t, p.0))
            })
            .collect(),
        writes: def
            .params
            .iter()
            .map(|p| (p.1.is_array() || p.1.is_struct()) && writes_var(methods, &def.body, p.0))
            .collect(),
    }
}

/// Records the facts about every closure of `program` for the shape functions of this thread; `operator_counts` and the
/// interpreter call it before they count.
pub fn register_closures(program: &Program) {
    let methods = &program.methods;
    let mut table = BTreeMap::new();
    let mut sites: Vec<&[Stmt]> = vec![program.body.as_slice()];
    sites.extend(methods.fns.iter().map(|def| def.body.as_slice()));
    for stmts in sites {
        walk_all_stmts(stmts, &mut |s| {
            if let Stmt::LetClosure { def } = s {
                table.insert(def.var, closure_info(methods, def));
            }
        });
    }
    CLOSURE_INFO.with(|cell| *cell.borrow_mut() = table);
}

fn info_of(var: u32) -> ClosureInfo {
    CLOSURE_INFO.with(|cell| cell.borrow().get(&var).cloned().unwrap_or_default())
}

/// Whether a call of closure `var` is order-sensitive in its arguments: two arguments, each an integer that the body
/// weights into the result, call `&mut self` on one root (the left argument must run first, or the result changes).
pub fn closure_args_ordered(methods: &Methods, var: u32, args: &[Expr]) -> bool {
    let info = info_of(var);
    let roots: Vec<Vec<Root>> = args
        .iter()
        .map(|arg| {
            let mut r = Vec::new();
            mutating_roots(methods, arg, &mut r);
            r
        })
        .collect();
    (0..args.len()).any(|a| {
        info.weighted.get(a) == Some(&true)
            && ((a + 1)..args.len()).any(|b| {
                info.weighted.get(b) == Some(&true) && roots[a].iter().any(|r| roots[b].contains(r))
            })
    })
}

/// The names a closure call counts under: itself, and the shape when two of its weighted arguments call `&mut self`
/// on one root (the left argument must run first).
pub fn closure_call_names(methods: &Methods, var: u32, args: &[Expr]) -> Vec<&'static str> {
    let mut names = vec![CLOSURE_CALL];
    if closure_args_ordered(methods, var, args) {
        names.push(SHAPE_CLOSURE_ARGS_ROOT);
    }
    names
}

/// `shape closure mutates its parameter, caller variable passed`: `f(v) ^ read`, the call of a closure whose body
/// writes its aggregate parameter, passed the caller's variable `v`, then (`^` does not short-circuit) a read of `v`
/// whose value goes into the result. The read shows whether the callee changed the caller's copy.
pub fn closure_observed_shape(op: BinOp, l: &Expr, r: &Expr) -> Option<&'static str> {
    if op != BinOp::BitXor {
        return None;
    }
    let Expr::ClosureCall { var, args, .. } = l else {
        return None;
    };
    let read = match r {
        Expr::Cast { e, .. } => e,
        other => other,
    };
    let Expr::Read { place, .. } = read else {
        return None;
    };
    let info = info_of(*var);
    args.iter()
        .enumerate()
        .any(|(k, arg)| {
            matches!(arg, Expr::Var(v, ty) if Root::Var(*v) == place.root && (ty.is_array() || ty.is_struct()))
                && info.writes.get(k) == Some(&true)
        })
        .then_some(SHAPE_CLOSURE_MUT_PARAM)
}

/// `shape closure call in an array literal beside a call on its root`: an element of the literal holds a closure call
/// whose arguments call `&mut self` on a root that another element calls `&mut self` on (the elements run left to right).
pub fn closure_array_lit_shape(methods: &Methods, elems: &[Expr]) -> Option<&'static str> {
    elems
        .iter()
        .enumerate()
        .any(|(k, element)| {
            closure_arg_roots(methods, element).iter().any(|root| {
                elems.iter().enumerate().any(|(l, other)| {
                    let mut others = Vec::new();
                    mutating_roots(methods, other, &mut others);
                    l != k && others.contains(root)
                })
            })
        })
        .then_some(SHAPE_CLOSURE_ARRAY_LIT)
}

/// `shape closure called twice with a side effect in an argument`: both operands of a binary call one closure with
/// arguments that call `&mut self` on one root (the left operand must run first).
pub fn closure_twice_shape(methods: &Methods, l: &Expr, r: &Expr) -> Option<&'static str> {
    let (left, right) = (
        closure_calls_with_roots(methods, l),
        closure_calls_with_roots(methods, r),
    );
    left.iter()
        .any(|(var, roots)| {
            right
                .iter()
                .any(|(v, rs)| v == var && roots.iter().any(|root| rs.contains(root)))
        })
        .then_some(SHAPE_CLOSURE_TWICE)
}

/// The closure shapes an assignment counts under: a compound set, or a set of an element, whose value calls a closure
/// with an argument that calls `&mut self` on the place's root.
pub fn closure_set_names(
    methods: &Methods,
    place: &Place,
    op: Option<BinOp>,
    value: &Expr,
) -> Vec<&'static str> {
    let mut names = Vec::new();
    if closure_arg_roots(methods, value).contains(&place.root) {
        if op.is_some() {
            names.push(SHAPE_CLOSURE_SET_OP);
        }
        if place.index.is_some() {
            names.push(SHAPE_CLOSURE_SET_INDEX);
        }
    }
    names
}

/// The closure shape of the final store: the result calls a closure whose weighted arguments call `&mut self` on one
/// root (an ordered call, [`closure_args_ordered`]).
pub fn closure_store_names(methods: &Methods, result: &Expr) -> Vec<&'static str> {
    let mut ordered = false;
    walk_expr(result, &mut |n| {
        if let Expr::ClosureCall { var, args, .. } = n {
            ordered |= closure_args_ordered(methods, *var, args);
        }
    });
    if ordered {
        vec![SHAPE_CLOSURE_STORE]
    } else {
        Vec::new()
    }
}

/// Whether a statement list writes the variable `var`: assigns to a place rooted at it, or calls `&mut self` on it.
pub fn writes_var(methods: &Methods, stmts: &[Stmt], var: u32) -> bool {
    let mut found = false;
    walk_all_stmts(stmts, &mut |s| match s {
        Stmt::Set { place, .. } => found |= place.root == Root::Var(var),
        Stmt::Assign { var: v, .. } | Stmt::AssignComp { var: v, .. } => found |= *v == var,
        Stmt::Call {
            f, recv: Some(p), ..
        } => {
            found |= methods.fns[*f].recv == Recv::Mut && p.root == Root::Var(var);
        }
        _ => {}
    });
    walk_stmts(stmts, &mut |e| {
        if let Expr::Call {
            f, recv: Some(p), ..
        } = e
        {
            found |= methods.fns[*f].recv == Recv::Mut && p.root == Root::Var(var);
        }
    });
    found
}

/// The closure-site constructs of a program, counted statically (the interpreter counts the same names when a `let`
/// runs and a call runs): where each closure is bound, what its signature holds, what its body calls, and the two
/// shapes that need the closure's definition.
pub(super) fn closure_site_counts(program: &Program, counts: &mut BTreeMap<String, usize>) {
    let methods = &program.methods;
    let mut defs: BTreeMap<u32, ClosureDef> = BTreeMap::new();
    let mut sites: Vec<(&[Stmt], Option<usize>)> = vec![(program.body.as_slice(), None)];
    for (k, def) in methods.fns.iter().enumerate() {
        sites.push((def.body.as_slice(), Some(k)));
    }
    for &(stmts, site) in &sites {
        walk_all_stmts(stmts, &mut |s| {
            if let Stmt::LetClosure { def } = s {
                defs.insert(def.var, (**def).clone());
                for name in closure_let_names(methods, def, site) {
                    *counts.entry(name.to_owned()).or_default() += 1;
                }
            }
        });
    }
    let mut call = |e: &Expr, in_method: bool| {
        if let Expr::ClosureCall { var, args, .. } = e
            && let Some(def) = defs.get(var)
        {
            for name in closure_mut_param_names(methods, def, args, in_method) {
                *counts.entry(name.to_owned()).or_default() += 1;
            }
        }
    };
    walk_stmts(&program.body, &mut |e| call(e, false));
    call(&program.result, false);
    for def in &methods.fns {
        let in_method = def.owner.is_some();
        walk_fn_exprs(def, &mut |e| call(e, in_method));
    }
}

/// The names a closure binding counts under, `site` the function that holds it (`None`: the kernel).
pub fn closure_let_names(
    methods: &Methods,
    def: &ClosureDef,
    site: Option<usize>,
) -> Vec<&'static str> {
    let mut names = vec![CLOSURE_LET];
    names.push(match site {
        None => CLOSURE_IN_KERNEL,
        Some(k) if methods.fns[k].owner.is_some() => CLOSURE_IN_METHOD,
        Some(_) => CLOSURE_IN_HELPER,
    });
    if def.params.iter().any(|p| p.1.is_array() && p.2) {
        names.push(CLOSURE_MUT_ARRAY);
    }
    if def.params.iter().any(|p| p.1.is_struct() && p.2) {
        names.push(CLOSURE_MUT_STRUCT);
    }
    if def.ret.is_array() {
        names.push(CLOSURE_ARRAY_RESULT);
    }
    let (mut helper, mut method) = (false, false);
    let mut see = |e: &Expr| {
        if let Expr::Call { f, .. } = e {
            if methods.fns[*f].recv == Recv::None && methods.fns[*f].owner.is_none() {
                helper = true;
            } else {
                method = true;
            }
        }
    };
    walk_stmts(&def.body, &mut see);
    walk_expr(&def.tail, &mut see);
    walk_all_stmts(&def.body, &mut |s| {
        if let Stmt::Call { f, .. } = s {
            if methods.fns[*f].recv == Recv::None && methods.fns[*f].owner.is_none() {
                helper = true;
            } else {
                method = true;
            }
        }
    });
    if helper {
        names.push(CLOSURE_BODY_HELPER);
    }
    if method {
        names.push(CLOSURE_BODY_METHOD);
    }
    names
}

/// The form that needs the definition, at a call (static form): a body that calls a method on a struct
/// parameter (not an order shape: only `&self` and by-value methods are called on a parameter).
pub fn closure_mut_param_names(
    methods: &Methods,
    def: &ClosureDef,
    _args: &[Expr],
    _in_method: bool,
) -> Vec<&'static str> {
    let mut names = Vec::new();
    if closure_calls_on_param(methods, def) {
        names.push(CLOSURE_PARAM_METHOD);
    }
    names
}

/// Whether a closure body calls a method on one of its struct parameters.
pub fn closure_calls_on_param(_methods: &Methods, def: &ClosureDef) -> bool {
    let is_param = |place: &Place| {
        def.params
            .iter()
            .any(|p| p.1.is_struct() && place.root == Root::Var(p.0))
    };
    let mut found = false;
    walk_stmts(&def.body, &mut |e| {
        if let Expr::Call { recv: Some(p), .. } = e {
            found |= is_param(p);
        }
    });
    walk_expr(&def.tail, &mut |e| {
        if let Expr::Call { recv: Some(p), .. } = e {
            found |= is_param(p);
        }
    });
    walk_all_stmts(&def.body, &mut |s| {
        if let Stmt::Call { recv: Some(p), .. } = s {
            found |= is_param(p);
        }
    });
    found
}

/// The names a destructuring counts under: itself, the annotation, and the shape of a `&mut self` call as its source.
pub fn destruct_names(methods: &Methods, annotate: bool, init: &Expr) -> Vec<&'static str> {
    let mut names = vec![TUPLE_DESTRUCT];
    if annotate {
        names.push(TUPLE_DESTRUCT_TYPED);
    }
    if mut_call_root(methods, init).is_some() {
        names.push(SHAPE_DESTRUCT_CALL);
    }
    names
}

/// The names a value `match` counts under (`tail`: it is the value of an arm of another one): itself, an integer
/// scrutinee, a grouped integer arm, an arm with statements, an arm that returns, and the scrutinee shape.
pub fn value_match_names(
    methods: &Methods,
    scrut: &Expr,
    arms: &[Arm],
    tail: bool,
) -> Vec<&'static str> {
    let mut names = vec![if tail {
        MATCH_VALUE_NESTED
    } else {
        MATCH_VALUE
    }];
    if !scrut.ty().is_enum() {
        names.push(MATCH_INT);
    }
    if arms
        .iter()
        .any(|arm| matches!(&arm.pat, Pat::Lits(lits) if lits.len() > 1))
    {
        names.push(MATCH_INT_GROUP);
    }
    if arms.iter().any(|arm| arm.body.len() > 1) {
        names.push(MATCH_VALUE_BLOCK);
    }
    if arms
        .iter()
        .any(|arm| matches!(arm.body.last(), Some(Stmt::Return { .. })))
    {
        names.push(MATCH_VALUE_RETURN);
    }
    if match_shape(methods, scrut, arms).is_some() {
        names.push(SHAPE_VMATCH_SCRUTINEE);
    }
    names
}

/// Whether a statement list refers to `root` anywhere: a variable, a field read, a place written, a receiver.
pub fn stmts_use_root(stmts: &[Stmt], root: Root) -> bool {
    let mut found = false;
    let mut look = |e: &Expr| {
        found |= match e {
            Expr::Var(v, _) => root == Root::Var(*v),
            Expr::Read { place, .. } => place.root == root,
            Expr::Call {
                recv: Some(place), ..
            } => place.root == root,
            _ => false,
        };
    };
    walk_stmts(stmts, &mut look);
    fn block(stmts: &[Stmt], root: Root) -> bool {
        stmts.iter().any(|stmt| match stmt {
            Stmt::Set { place, .. } => place.root == root,
            Stmt::Assign { var, .. } | Stmt::AssignComp { var, .. } => Root::Var(*var) == root,
            Stmt::Call {
                recv: Some(place), ..
            } => place.root == root,
            Stmt::If { then, els, .. } => block(then, root) || block(els, root),
            Stmt::For { body, .. } => block(body, root),
            Stmt::Match { arms, .. } => arms.iter().any(|arm| block(&arm.body, root)),
            _ => false,
        })
    }
    found || block(stmts, root)
}

/// The shape of a statement `match`: its scrutinee holds a `&mut self` call on a root that an arm uses.
pub fn match_shape(methods: &Methods, scrut: &Expr, arms: &[Arm]) -> Option<&'static str> {
    let mut roots = Vec::new();
    mutating_roots(methods, scrut, &mut roots);
    roots
        .iter()
        .any(|&root| arms.iter().any(|arm| stmts_use_root(&arm.body, root)))
        .then_some(SHAPE_MATCH_SCRUTINEE)
}

/// The shapes of a call with enum arguments: an argument that is a `&mut self` call (a pure helper call does not count:
/// nothing depends on its order), and two arguments that are calls (or hold calls) mutating one root.
pub fn enum_call_shapes(methods: &Methods, args: &[Expr]) -> Vec<&'static str> {
    let mut shapes = Vec::new();
    let built: Vec<&Expr> = args
        .iter()
        .filter(|arg| arg.ty().is_enum() && matches!(arg, Expr::Call { .. }))
        .collect();
    let mut per_arg: Vec<Vec<Root>> = Vec::new();
    for arg in &built {
        let mut roots = Vec::new();
        mutating_roots(methods, arg, &mut roots);
        roots.dedup();
        per_arg.push(roots);
    }
    if per_arg.iter().any(|roots| !roots.is_empty()) {
        shapes.push(SHAPE_ENUM_ARG);
    }
    let shared = per_arg.iter().enumerate().any(|(k, roots)| {
        roots
            .iter()
            .any(|root| per_arg[k + 1..].iter().any(|other| other.contains(root)))
    });
    if shared {
        shapes.push(SHAPE_ENUM_ARGS_ROOT);
    }
    shapes
}

/// The names a statement `match` counts under: itself, its kind of ending, a grouped arm, an arm that returns, calls or
/// assigns, and where it sits (`in_for`: inside a loop body of its function, `nested`: inside an arm of a `match`).
pub fn match_names(arms: &[Arm], in_for: bool, nested: bool) -> Vec<&'static str> {
    let mut names = vec![MATCH];
    names.push(if arms.iter().any(|arm| arm.pat == Pat::Wild) {
        MATCH_WILD
    } else {
        MATCH_DEFAULT
    });
    if arms
        .iter()
        .any(|arm| matches!(&arm.pat, Pat::Variants(v) if v.len() > 1))
    {
        names.push(MATCH_GROUPED);
    }
    if arms
        .iter()
        .any(|arm| matches!(arm.body.last(), Some(Stmt::Return { .. })))
    {
        names.push(MATCH_ARM_RETURN);
    }
    let calls = |arm: &Arm| {
        let mut found = arm.body.iter().any(|s| matches!(s, Stmt::Call { .. }));
        walk_stmts(&arm.body, &mut |e| found |= matches!(e, Expr::Call { .. }));
        found
    };
    if arms.iter().any(calls) {
        names.push(MATCH_ARM_CALL);
    }
    if arms.iter().any(|arm| {
        arm.body
            .iter()
            .any(|s| matches!(s, Stmt::Set { .. } | Stmt::Assign { .. }))
    }) {
        names.push(MATCH_ARM_SET);
    }
    if in_for {
        names.push(MATCH_IN_FOR);
    }
    if nested {
        names.push(MATCH_NESTED);
    }
    names
}
