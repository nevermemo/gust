//! The counted construct names and shape rules for calls, matches, traits, generics and tuples.

use super::*;

/// The counted name of a call (`call mut`, ...), the same string in `operator_counts` and in the interpreter's coverage.
pub fn call_name(recv: Recv, owner: Option<u8>) -> &'static str {
    match (recv, owner) {
        (Recv::Mut, _) => "call &mut self",
        (Recv::Shared, _) => "call &self",
        (Recv::Value, _) => "call self",
        (Recv::None, Some(_)) => "call assoc",
        (Recv::None, None) => "call helper",
    }
}

/// The counted name of a `Stmt::Set`: `set =`, `set +=`, ...
pub fn set_name(op: Option<BinOp>) -> String {
    match op {
        None => "set =".to_owned(),
        Some(op) => format!("set {}=", op.symbol()),
    }
}

/// Names of the order-sensitive shapes (round r1 of m9.d1a). The generator biases toward them, `operator_counts` counts
/// them statically and the interpreter counts them when it runs the node, under the same names.
pub const SHAPE_STMT_CALL: &str = "shape stmt &mut self call";
pub const SHAPE_EXPR_CALL: &str = "shape expr &mut self call";
pub const SHAPE_LOGIC_CALL: &str = "shape &mut self call right of &&/||";
pub const SHAPE_READ_AND_CALL: &str = "shape read and &mut self call on one root";
pub const SHAPE_VALUE_CALL_INNER: &str =
    "shape by-value call with &mut self call on its root in its arguments";

/// Whether `e` is a `&mut self` call, and on which root. An element of such a call's tuple result (`c.pair().1`) is one
/// too: the call runs before the element is read.
pub(super) fn mut_call_root(methods: &Methods, e: &Expr) -> Option<Root> {
    match e {
        Expr::PathCall { call, .. } => mut_call_root(methods, call),
        Expr::Call {
            f,
            recv: Some(place),
            ..
        } if methods.fns[*f].recv == Recv::Mut => Some(place.root),
        Expr::TupleField { e, .. } => mut_call_root(methods, e),
        _ => None,
    }
}

/// Whether `e` or a node below it is a `&mut self` call on `root`.
pub(super) fn contains_mut_call_on(methods: &Methods, e: &Expr, root: Root) -> bool {
    let mut found = false;
    walk_expr(e, &mut |n| found |= mut_call_root(methods, n) == Some(root));
    found
}

/// `shape same-root set <op>=`: an assignment to a place whose value contains a `&mut self` call on the place's root.
pub fn set_shape(
    methods: &Methods,
    place: &Place,
    op: Option<BinOp>,
    value: &Expr,
) -> Option<String> {
    contains_mut_call_on(methods, value, place.root)
        .then(|| format!("shape same-root {}", set_name(op)))
}

/// The shape of a binary: a `&mut self` call as the right operand of `&&`/`||`, or a read of a place and a `&mut self`
/// call on its root as the two operands.
pub fn bin_shape(methods: &Methods, op: BinOp, l: &Expr, r: &Expr) -> Option<&'static str> {
    if matches!(op, BinOp::And | BinOp::Or) && mut_call_root(methods, r).is_some() {
        return Some(SHAPE_LOGIC_CALL);
    }
    // `==` or `!=` of two enum values that are calls mutating one root: the left call must run first.
    if matches!(op, BinOp::Eq | BinOp::Ne) && l.ty().is_enum() {
        let (mut left, mut right) = (Vec::new(), Vec::new());
        mutating_roots(methods, l, &mut left);
        mutating_roots(methods, r, &mut right);
        return left
            .iter()
            .any(|root| right.contains(root))
            .then_some(SHAPE_ENUM_EQ);
    }
    let read_root = |e: &Expr| match e {
        Expr::Read { place, .. } => Some(place.root),
        _ => None,
    };
    let same = |a: Option<Root>, b: Option<Root>| a.is_some() && a == b;
    (same(read_root(l), mut_call_root(methods, r)) || same(mut_call_root(methods, l), read_root(r)))
        .then_some(SHAPE_READ_AND_CALL)
        .or_else(|| array_index_shape(methods, l, r))
        .or_else(|| closure_twice_shape(methods, l, r))
        .or_else(|| closure_observed_shape(op, l, r))
}

/// The shape of a call: a by-value call whose arguments contain a `&mut self` call on the receiver's root.
pub fn call_shape(
    methods: &Methods,
    f: usize,
    recv: &Option<Place>,
    args: &[Expr],
) -> Option<&'static str> {
    let place = recv.as_ref()?;
    (methods.fns[f].recv == Recv::Value
        && args
            .iter()
            .any(|arg| contains_mut_call_on(methods, arg, place.root)))
    .then_some(SHAPE_VALUE_CALL_INNER)
}

/// Names of the constructs of the enum space (format 9, slice m9.d1b), counted statically by `operator_counts` and when
/// executed by the interpreter, under the same names. The `shape` ones are the order-sensitive forms the generator is
/// biased toward.
pub const MATCH: &str = "match";
pub const MATCH_WILD: &str = "match with _";
pub const MATCH_DEFAULT: &str = "match without _ (the default rule)";
pub const MATCH_GROUPED: &str = "match grouped arm";
pub const MATCH_NESTED: &str = "match nested in an arm";
pub const MATCH_IN_FOR: &str = "match in a for body";
pub const MATCH_ARM_RETURN: &str = "match arm ends in return";
pub const MATCH_ARM_CALL: &str = "match arm calls";
pub const MATCH_ARM_SET: &str = "match arm assigns";
pub const SHAPE_MATCH_SCRUTINEE: &str =
    "shape match scrutinee &mut self call on a root an arm uses";
pub const SHAPE_ENUM_EQ: &str = "shape enum ==/!= of two calls on one root";
pub const SHAPE_ENUM_ARG: &str = "shape enum argument built from a &mut self call";
pub const SHAPE_ENUM_ARGS_ROOT: &str = "shape enum arguments built from calls on one root";

/// Names of the constructs of the trait space (format 13, slice m9.d2), counted statically by `operator_counts` and when
/// executed by the interpreter, under the same names. The `shape` ones are the order-sensitive forms the generator is
/// biased toward; each counts only where a `&mut self` call on a root is among the operands the order is observed on.
pub const TRAIT_CALL: &str = "trait call";
pub const TRAIT_CALL_ENUM: &str = "trait call on an enum";
pub const TRAIT_PATH_CALL: &str = "trait path call";
pub const TRAIT_PATH_QUALIFIED: &str = "trait path call, qualified";
pub const TRAIT_SELF_RESULT: &str = "trait call returning Self";
pub const THROUGH_PARAM: &str = "trait call through a type parameter";
pub const GENERIC_CALL: &str = "generic call";
pub const GENERIC_METHOD_CALL: &str = "generic method call";
pub const GENERIC_TWO_PARAMS: &str = "generic call with two type parameters";
pub const GENERIC_TWO_BOUNDS: &str = "generic call with a two-trait bound";
pub const GENERIC_WHERE: &str = "generic call with a where clause";
pub const GENERIC_RETURNS_T: &str = "generic call returning T";
pub const GENERIC_FROM_GENERIC: &str = "generic call in a generic body";
pub const SHAPE_TRAIT_ARGS_ROOT: &str = "shape trait call arguments built from calls on one root";
pub const SHAPE_GENERIC_ARGS_ROOT: &str =
    "shape generic call arguments built from calls on one root";
pub const SHAPE_GENERIC_VALUE_SOURCE: &str =
    "shape generic call, by-value T whose source a later argument mutates";
pub const SHAPE_TRAIT_VALUE_SELF: &str =
    "shape trait call, by-value self whose source an argument mutates";
pub const SHAPE_MUT_THROUGH_PARAM: &str = "shape &mut self through a type parameter";
pub const SHAPE_PARAM_ARGS_SELF_ROOT: &str =
    "shape call through a type parameter, arguments change the enclosing method's self";

/// The arguments of a call of `def` whose values reach its result in the order the call gives them: the integer ones
/// of a function with an integer result.
fn observed_args<'a>(def: &FnDef, args: &'a [Expr]) -> Vec<&'a Expr> {
    if !def.ret.is_some_and(Ty::is_int) {
        return Vec::new();
    }
    args.iter()
        .zip(&def.params)
        .filter(|(_, (_, ty))| ty.is_int())
        .map(|(arg, _)| arg)
        .collect()
}

/// Whether the body of the generic `def` reads the `T` of its type parameter `g` through an integer method (the last
/// parameter may be consumed by a by-value method, the others only with `Copy`): the generator folds that read into
/// every integer result.
pub fn generic_reads(methods: &Methods, def: &FnDef, g: usize) -> bool {
    let Some(gp) = def.gens.get(g) else {
        return false;
    };
    let by_value = gp.copy || g + 1 == def.gens.len();
    gp.bounds.iter().any(|&tr| {
        methods.traits[usize::from(tr)].methods.iter().any(|m| {
            matches!(m.ret, TRet::Ty(t) if t.is_int()) && (by_value || m.recv != Recv::Value)
        })
    })
}

/// The trait and generic names (constructs and shapes) of a call of function `f` with receiver `recv` and `args`.
pub fn trait_call_names(
    methods: &Methods,
    f: usize,
    recv: &Option<Place>,
    args: &[Expr],
) -> Vec<&'static str> {
    let def = &methods.fns[f];
    let mut names = Vec::new();
    // The arguments whose values reach the result: the integer ones of a function with an integer result (the impls
    // and generic functions weigh them, in the order given, into it).
    let observed = observed_args(def, args);
    let twice = || {
        let mut roots: Vec<Root> = Vec::new();
        for arg in observed.iter().copied() {
            let mut found = Vec::new();
            mutating_roots(methods, arg, &mut found);
            found.sort_by_key(|r| format!("{r:?}"));
            found.dedup();
            roots.extend(found);
        }
        roots
            .iter()
            .any(|r| roots.iter().filter(|o| *o == r).count() >= 2)
    };
    if def.imp.is_some() {
        names.push(TRAIT_CALL);
        if let Some(place) = recv {
            if place.root_ty.is_enum() {
                names.push(TRAIT_CALL_ENUM);
            }
            if matches!(place.root_ty, Ty::Param(_)) {
                names.push(THROUGH_PARAM);
                // A statement `c.m(..)` that changes the `T` the body reads afterwards (the generator changes a `T`
                // only when it reads or returns it).
                if def.recv == Recv::Mut && def.ret.is_none() {
                    names.push(SHAPE_MUT_THROUGH_PARAM);
                }
                if observed
                    .iter()
                    .any(|a| contains_mut_call_on(methods, a, Root::SelfV))
                {
                    names.push(SHAPE_PARAM_ARGS_SELF_ROOT);
                }
            }
            // A by-value struct impl reads its fields into its integer result: a copy taken after the argument ran
            // would show.
            if def.recv == Recv::Value
                && def.owner.is_some()
                && def.ret.is_some_and(Ty::is_int)
                && args
                    .iter()
                    .any(|a| contains_mut_call_on(methods, a, place.root))
            {
                names.push(SHAPE_TRAIT_VALUE_SELF);
            }
        }
        let self_ret = def.imp.is_some_and(|i| {
            methods.traits[usize::from(i.tr)].methods[usize::from(i.m)].ret == TRet::SelfTy
        });
        if self_ret {
            names.push(TRAIT_SELF_RESULT);
        }
        if twice() {
            names.push(SHAPE_TRAIT_ARGS_ROOT);
        }
    }
    if !def.gens.is_empty() {
        names.push(if def.recv == Recv::None {
            GENERIC_CALL
        } else {
            GENERIC_METHOD_CALL
        });
        if args.iter().any(|a| matches!(a.ty(), Ty::Param(_))) {
            names.push(GENERIC_FROM_GENERIC);
        }
        if def.gens.len() == 2 {
            names.push(GENERIC_TWO_PARAMS);
        }
        if def.gens.iter().any(|g| g.bounds.len() == 2) {
            names.push(GENERIC_TWO_BOUNDS);
        }
        if def.gens.iter().any(|g| g.where_clause) {
            names.push(GENERIC_WHERE);
        }
        if matches!(def.ret, Some(Ty::Param(_))) {
            names.push(GENERIC_RETURNS_T);
        }
        if twice() {
            names.push(SHAPE_GENERIC_ARGS_ROOT);
        }
        let source = args.iter().enumerate().any(|(i, arg)| {
            let root = match arg {
                Expr::Var(v, Ty::Struct(_)) => Some(Root::Var(*v)),
                Expr::Read {
                    place,
                    ty: Ty::Struct(_),
                } => Some(place.root),
                _ => None,
            };
            matches!(def.params[i].1, Ty::Param(g) if def.ret.is_some_and(Ty::is_int)
                    && generic_reads(methods, def, usize::from(g)))
                && root.is_some_and(|root| {
                    args[i + 1..]
                        .iter()
                        .any(|later| contains_mut_call_on(methods, later, root))
                })
        });
        if source {
            names.push(SHAPE_GENERIC_VALUE_SOURCE);
        }
    }
    names
}

/// The names the trait and generic constructs add to the known shape rows.
pub const TRAIT_SHAPE_NAMES: [&str; 19] = [
    TRAIT_CALL,
    TRAIT_CALL_ENUM,
    TRAIT_PATH_CALL,
    TRAIT_PATH_QUALIFIED,
    TRAIT_SELF_RESULT,
    THROUGH_PARAM,
    GENERIC_CALL,
    GENERIC_METHOD_CALL,
    GENERIC_TWO_PARAMS,
    GENERIC_TWO_BOUNDS,
    GENERIC_WHERE,
    GENERIC_RETURNS_T,
    GENERIC_FROM_GENERIC,
    SHAPE_TRAIT_ARGS_ROOT,
    SHAPE_GENERIC_ARGS_ROOT,
    SHAPE_GENERIC_VALUE_SOURCE,
    SHAPE_TRAIT_VALUE_SELF,
    SHAPE_MUT_THROUGH_PARAM,
    SHAPE_PARAM_ARGS_SELF_ROOT,
];

/// Every shape and `match` construct name that `check-difftest` prints a row for, so a name that is never written still
/// shows as a row of zeros (`coverage_json` adds these to the operator rows).
pub const KNOWN_SHAPE_NAMES: [&str; 62] = [
    SHAPE_STMT_CALL,
    SHAPE_EXPR_CALL,
    SHAPE_LOGIC_CALL,
    SHAPE_READ_AND_CALL,
    SHAPE_VALUE_CALL_INNER,
    SHAPE_MATCH_SCRUTINEE,
    SHAPE_ENUM_EQ,
    SHAPE_ENUM_ARG,
    SHAPE_ENUM_ARGS_ROOT,
    MATCH,
    MATCH_WILD,
    MATCH_DEFAULT,
    MATCH_GROUPED,
    MATCH_NESTED,
    MATCH_IN_FOR,
    MATCH_ARM_RETURN,
    MATCH_ARM_CALL,
    MATCH_ARM_SET,
    "return",
    "enum as u32",
    "enum as i32",
    MATCH_VALUE,
    MATCH_VALUE_NESTED,
    MATCH_VALUE_BLOCK,
    MATCH_VALUE_RETURN,
    MATCH_INT,
    MATCH_INT_GROUP,
    MATCH_ARM_RETURN_RUN,
    TUPLE_LIT,
    TUPLE_FIELD,
    TUPLE_READ,
    TUPLE_SET,
    TUPLE_SET_OP,
    TUPLE_DESTRUCT,
    TUPLE_DESTRUCT_TYPED,
    TUPLE_ARG,
    SHAPE_TUPLE_LIT_ROOT,
    SHAPE_TUPLE_SET_CALL,
    SHAPE_TUPLE_FIELD_CALL,
    SHAPE_DESTRUCT_CALL,
    SHAPE_LT_ELEMENT,
    SHAPE_LT_ARG,
    SHAPE_LT_METHOD_ARG,
    SHAPE_VMATCH_SCRUTINEE,
    CLOSURE_LET,
    CLOSURE_CALL,
    CLOSURE_IN_KERNEL,
    CLOSURE_IN_HELPER,
    CLOSURE_IN_METHOD,
    CLOSURE_BODY_HELPER,
    CLOSURE_BODY_METHOD,
    CLOSURE_MUT_ARRAY,
    CLOSURE_MUT_STRUCT,
    CLOSURE_ARRAY_RESULT,
    SHAPE_CLOSURE_ARGS_ROOT,
    SHAPE_CLOSURE_ARRAY_LIT,
    SHAPE_CLOSURE_SET_OP,
    SHAPE_CLOSURE_SET_INDEX,
    SHAPE_CLOSURE_STORE,
    SHAPE_CLOSURE_TWICE,
    SHAPE_CLOSURE_MUT_PARAM,
    CLOSURE_PARAM_METHOD,
];

/// Names of the constructs of format 10 (slice m9.d1c), counted statically by `operator_counts` and when executed by
/// the interpreter, under the same names. The `shape` ones are the order-sensitive forms the generator is biased toward.
pub const MATCH_VALUE: &str = "match value (typed let)";
pub const MATCH_VALUE_NESTED: &str = "match value arm is a nested value match";
pub const MATCH_VALUE_BLOCK: &str = "match value arm block with statements";
pub const MATCH_VALUE_RETURN: &str = "match value arm ends in return";
pub const MATCH_INT: &str = "match integer patterns";
pub const MATCH_INT_GROUP: &str = "match integer group arm";
/// Counted only when executed: a `return` ran inside an arm of a `match` (the nit of the m9.d1b re-check, R1).
pub const MATCH_ARM_RETURN_RUN: &str = "match arm return executed";
pub const TUPLE_LIT: &str = "tuple literal";
pub const TUPLE_FIELD: &str = "tuple field of a call";
pub const TUPLE_READ: &str = "tuple element read";
pub const TUPLE_SET: &str = "tuple element set";
pub const TUPLE_SET_OP: &str = "tuple element compound set";
pub const TUPLE_DESTRUCT: &str = "tuple destructuring";
pub const TUPLE_DESTRUCT_TYPED: &str = "tuple destructuring annotated";
pub const TUPLE_ARG: &str = "tuple argument";
pub const SHAPE_TUPLE_LIT_ROOT: &str =
    "shape tuple literal with a &mut self call and another element using its root";
pub const SHAPE_TUPLE_SET_CALL: &str =
    "shape compound set of a tuple element field with a &mut self call on that element";
pub const SHAPE_TUPLE_FIELD_CALL: &str = "shape f().k of a &mut self call";
pub const SHAPE_DESTRUCT_CALL: &str = "shape destructuring of a &mut self call";
pub const SHAPE_LT_ELEMENT: &str = "shape < before the last tuple element";
pub const SHAPE_LT_ARG: &str = "shape < before the last argument of a function";
pub const SHAPE_LT_METHOD_ARG: &str = "shape < before the last argument of a method";
pub const SHAPE_VMATCH_SCRUTINEE: &str =
    "shape value match scrutinee &mut self call on a root an arm uses";

/// The counted name of an enum value read as its repr type.
pub fn enum_as_name(to: Ty) -> String {
    format!("enum as {}", to.name())
}

/// Whether a place steps through a tuple (its root is a tuple local or parameter).
pub fn place_in_tuple(place: &Place) -> bool {
    place.root_ty.is_tuple()
}

/// Whether `e` reads or calls on `root` anywhere: a variable, a field read, a receiver.
fn uses_root(e: &Expr, root: Root) -> bool {
    let mut found = false;
    walk_expr(e, &mut |n| {
        found |= match n {
            Expr::Var(v, _) => root == Root::Var(*v),
            Expr::Read { place, .. } => place.root == root,
            Expr::Call {
                recv: Some(place), ..
            } => place.root == root,
            _ => false,
        };
    });
    found
}

/// `shape tuple literal with a &mut self call and another element using its root`: the elements run left to right, so
/// what an element sees of the root depends on the calls of the elements before it.
pub fn tuple_lit_shape(methods: &Methods, elems: &[Expr]) -> Option<&'static str> {
    let shared = elems.iter().enumerate().any(|(k, element)| {
        let mut roots = Vec::new();
        mutating_roots(methods, element, &mut roots);
        roots.iter().any(|&root| {
            elems
                .iter()
                .enumerate()
                .any(|(l, other)| l != k && uses_root(other, root))
        })
    });
    shared.then_some(SHAPE_TUPLE_LIT_ROOT)
}

/// `shape < before the last ...`: a `<` comparison that is not the last element of a tuple literal or the last argument
/// of a call (the shape that made the macro emit `__gust_keep`, fixed by m9.f1). The name is `name` (the three rows are
/// split so that dropping the method case, the one m9.f1 hit, cannot leave the row green).
pub fn lt_before_last(items: &[Expr], name: &'static str) -> Option<&'static str> {
    (items.len() > 1
        && items[..items.len() - 1]
            .iter()
            .any(|item| matches!(item, Expr::Bin { op: BinOp::Lt, .. })))
    .then_some(name)
}

/// The names a tuple literal counts under: itself, and the shapes it is.
pub fn tuple_lit_names(methods: &Methods, elems: &[Expr]) -> Vec<&'static str> {
    let mut names = vec![TUPLE_LIT];
    names.extend(tuple_lit_shape(methods, elems));
    names.extend(lt_before_last(elems, SHAPE_LT_ELEMENT));
    names
}

/// The names the arguments of a call count under: a tuple argument, and a `<` before the last argument (`method`: the
/// callee has a receiver).
pub fn call_arg_names(method: bool, args: &[Expr]) -> Vec<&'static str> {
    let mut names = Vec::new();
    if args.iter().any(|arg| arg.ty().is_tuple()) {
        names.push(TUPLE_ARG);
    }
    names.extend(lt_before_last(
        args,
        if method {
            SHAPE_LT_METHOD_ARG
        } else {
            SHAPE_LT_ARG
        },
    ));
    names
}

/// The names `f().k` counts under: itself, and the shape when `f` is a `&mut self` call.
pub fn tuple_field_names(methods: &Methods, e: &Expr) -> Vec<&'static str> {
    let mut names = vec![TUPLE_FIELD];
    if mut_call_root(methods, e).is_some() {
        names.push(SHAPE_TUPLE_FIELD_CALL);
    }
    names
}

/// The names an assignment counts under besides `set_name`: only for a place inside a tuple.
pub fn tuple_set_names(
    methods: &Methods,
    place: &Place,
    op: Option<BinOp>,
    value: &Expr,
) -> Vec<&'static str> {
    if !place.root_ty.is_tuple() {
        return Vec::new();
    }
    let mut names = vec![if op.is_some() {
        TUPLE_SET_OP
    } else {
        TUPLE_SET
    }];
    // The one assignment to a tuple place whose evaluation order shows: `t.k.f op= t.k.m()`, a compound set of a field
    // of a struct element whose value calls a `&mut self` method on that same element (the call changes the place).
    if op.is_some()
        && let Some((_, parent)) = place.path.split_last()
        && !parent.is_empty()
    {
        let mut same = false;
        walk_expr(value, &mut |n| {
            if let Expr::Call {
                f,
                recv: Some(recv),
                ..
            } = n
                && methods.fns[*f].recv == Recv::Mut
                && recv.root == place.root
                && recv.path == parent
            {
                same = true;
            }
        });
        if same {
            names.push(SHAPE_TUPLE_SET_CALL);
        }
    }
    names
}
