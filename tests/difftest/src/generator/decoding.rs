//! Decoding one program, and the attempt entry points.

use super::*;

/// Decodes one program from a byte stream, in the space that its attempt number selects (`space_of_attempt`). Pure:
/// the same bytes give the same program.
pub fn decode(bytes: &[u8], seed: u64, attempt: u64) -> Program {
    decode_in(space_of_attempt(attempt), bytes, seed, attempt)
}

/// Decodes one program of `space` from a byte stream.
pub fn decode_in(space: Space, bytes: &[u8], seed: u64, attempt: u64) -> Program {
    let mut g = Gen {
        b: Bytes {
            data: bytes,
            pos: 0,
        },
        floats: matches!(space, Space::Float | Space::Vector | Space::Math),
        vectors: space == Space::Vector,
        inputs: Vec::new(),
        data: Vec::new(),
        count: 1,
        next_var: 0,
        scope: Vec::new(),
        methods: matches!(
            space,
            Space::Method
                | Space::Enum
                | Space::Value
                | Space::Array
                | Space::Closure
                | Space::Trait
        ),
        table: Methods::default(),
        cur_fn: None,
        this: None,
        forbid: Vec::new(),
        enums: matches!(
            space,
            Space::Enum | Space::Value | Space::Array | Space::Closure | Space::Trait
        ),
        cur_ret: None,
        force_enum_ret: false,
        values: space == Space::Value,
        pre: Vec::new(),
        arm_return_fns: Vec::new(),
        arrays: matches!(space, Space::Array | Space::Closure),
        store: None,
        closures: space == Space::Closure,
        in_closure: false,
        probe_writers: Vec::new(),
        cls: Vec::new(),
        traits: space == Space::Trait,
        cur_gens: Vec::new(),
        inst_used: Vec::new(),
        math: space == Space::Math,
    };
    let count = match g.b.below(6) {
        k @ 0..=4 => COUNTS[k as usize],
        _ => 1 + g.b.below(4095),
    };
    g.count = count;
    let input_count = 1 + g.b.below(3);
    let mut inputs = Vec::new();
    for k in 0..input_count {
        let ty = if g.vectors {
            // Mostly `f32` (the vectors are mostly `f32`), and at least one `f32` buffer: a `Mat2` reads one.
            let ty = [Ty::U32, Ty::I32, Ty::F32][g.b.weighted(&[25, 25, 50])];
            if k + 1 == input_count && !g.inputs.contains(&Ty::F32) {
                Ty::F32
            } else {
                ty
            }
        } else if g.floats {
            [Ty::U32, Ty::I32, Ty::F32][g.b.weighted(&[35, 35, 30])]
        } else {
            g.int_ty()
        };
        let class = g.b.weighted(&[3, 4, 4, 3, 2, 1, 3]);
        let sub_seed = g.b.u64();
        let data = input_data(ty, class, count, sub_seed);
        g.inputs.push(ty);
        g.data.push(data.clone());
        inputs.push(InputBuf { ty, data });
    }
    // The math space draws its module constants before the kernel, which reads them.
    if g.math {
        g.gen_consts();
    }
    // The method space draws its structs and functions first; the kernel starts with one or two struct locals.
    let mut body = Vec::new();
    if g.methods {
        g.gen_tables();
        for _ in 0..1 + g.b.below(2) {
            let (pre_mark, at) = (g.pre.len(), body.len());
            let (j, init) = g.struct_init();
            let var = g.fresh();
            let mutable = g.b.chance(80);
            g.scope.push((var, Ty::Struct(j), mutable));
            body.push(Stmt::Let {
                var,
                ty: Ty::Struct(j),
                mutable,
                init,
            });
            g.splice_pre(pre_mark, at, &mut body);
        }
    }
    // The closure space binds one or two closures in the kernel before its ordinary block.
    if g.closures {
        for _ in 0..1 + g.b.below(2) {
            let (pre_mark, at) = (g.pre.len(), body.len());
            let stmt = g.closure_let(1);
            body.push(stmt);
            g.splice_pre(pre_mark, at, &mut body);
        }
        g.closure_probe_lets(&mut body);
    }
    // The trait space adds a local of every implementing type, so that the kernel can call the impls.
    if g.traits {
        g.trait_locals(&mut body);
    }
    body.extend(g.block(0, 2, 7));
    // The value space ends the kernel body, often, with `t.k.f op= t.k.m()` on a top-level tuple local (fix round 1),
    // and the result reads the field: the order-sensitive set then decides the output of the program.
    let mut forced_set = false;
    if g.values && g.b.chance(80) {
        if g.tuple_field_set_candidates().is_empty()
            && let Some(ty) = g.field_set_tuple_ty()
        {
            let (pre_mark, at) = (g.pre.len(), body.len());
            let init = g.expr(ty, 3);
            let var = g.fresh();
            g.scope.push((var, ty, true));
            body.push(Stmt::Let {
                var,
                ty,
                mutable: true,
                init,
            });
            g.splice_pre(pre_mark, at, &mut body);
        }
        let (pre_mark, at) = (g.pre.len(), body.len());
        if let Some(stmt) = g.tuple_field_set_stmt() {
            forced_set = true;
            body.push(stmt);
            g.splice_pre(pre_mark, at, &mut body);
        }
    }
    // The array space ends the kernel with its array statements, whose element reads join the result.
    let mut array_folds = Vec::new();
    if g.arrays {
        g.restore_scope(&body);
        g.restore_closures(&body);
        array_folds = g.array_section(&mut body);
    }
    // A boolean result would hide the field (`x != 0`): the forced set gets an integer result.
    let mut ty = g.scalar_ty();
    // A `bool` element reaches the output only through a `bool` result: often one.
    if array_folds.iter().any(|e| e.ty() == Ty::Bool) && g.b.chance(60) {
        ty = Ty::Bool;
    }
    if forced_set && ty == Ty::Bool {
        ty = Ty::U32;
    }
    // The closure space wants an integer result: the probe calls join it with a `^` that every path evaluates.
    if (g.closures && ty == Ty::Bool && g.b.chance(70)) || (g.traits && ty == Ty::Bool) {
        ty = Ty::U32;
    }
    if g.methods {
        // The result may call on the top-level locals (`out[i] = c.next()`).
        for stmt in &body {
            if let Stmt::Let {
                var, ty, mutable, ..
            }
            | Stmt::LetMatch {
                var, ty, mutable, ..
            } = stmt
                && !g.scope.iter().any(|(id, ..)| id == var)
            {
                g.scope.push((*var, *ty, *mutable));
            }
            if let Stmt::LetTuple {
                vars,
                ty: Ty::Tuple(t),
                ..
            } = stmt
            {
                for (&(var, mutable), elem) in vars.iter().zip(t.elems()) {
                    if !g.scope.iter().any(|(id, ..)| *id == var) {
                        g.scope.push((var, elem.ty(), mutable));
                    }
                }
            }
        }
    }
    let mut result = g.expr(ty, MAX_DEPTH);
    // The trait space joins calls of its impls and generic helpers to the result: every path evaluates them.
    if g.traits {
        for probe in g.trait_probes() {
            result = fold_in(result, probe);
        }
    }
    // The value space joins a call of a helper that returns from inside an arm to the result (nit R1 of the m9.d1b
    // re-check: such a `return` must run in many programs, not in a handful).
    let mut arm_call = None;
    if g.values && g.b.chance(75) {
        // A return type that has a scalar to fold: an integer, an enum (its discriminant), or an element of a tuple.
        let usable = |ret: Option<Ty>| match ret {
            Some(Ty::U32 | Ty::I32 | Ty::Enum(_)) => true,
            Some(Ty::Tuple(t)) => t.elems().iter().any(|e| !matches!(e, Elem::Struct(_))),
            _ => false,
        };
        let fns: Vec<usize> = g
            .arm_return_fns
            .iter()
            .copied()
            .filter(|&k| usable(g.table.fns[k].ret))
            .collect();
        if !fns.is_empty() {
            let f = fns[g.b.below(fns.len() as u32) as usize];
            let ret = g.table.fns[f].ret.expect("a return type");
            if let Some((recv, args)) = g.call_parts(f, 3) {
                let owner = if g.table.fns[f].recv == Recv::None {
                    g.table.fns[f].owner
                } else {
                    None
                };
                let call = Expr::Call {
                    f,
                    owner,
                    recv,
                    args,
                    ty: ret,
                };
                let as_repr = |g: &Gen, e: u8, value: Expr| Expr::EnumAs {
                    to: g.table.enums[usize::from(e)].repr,
                    e: Box::new(value),
                };
                arm_call = Some(match ret {
                    Ty::Enum(e) => as_repr(&g, e, call),
                    Ty::Tuple(t) => {
                        let ks: Vec<u8> = (0..t.n)
                            .filter(|&k| !matches!(t.elems()[usize::from(k)], Elem::Struct(_)))
                            .collect();
                        let index = ks[g.b.below(ks.len() as u32) as usize];
                        let field = Expr::TupleField {
                            e: Box::new(call),
                            index,
                        };
                        match t.elems()[usize::from(index)] {
                            Elem::Enum(e) => as_repr(&g, e, field),
                            _ => field,
                        }
                    }
                    _ => call,
                });
            }
        }
    }
    // Lets that the result hoisted come after the body and before the result.
    body.append(&mut g.pre);
    // Fold the final value of up to two top-level locals into the result, so that the computed values are live.
    let live: Vec<(u32, Ty)> = body
        .iter()
        .flat_map(|stmt| match stmt {
            Stmt::Let { var, ty, .. } if !ty.is_struct() && !ty.is_tuple() && !ty.is_array() => {
                vec![(*var, *ty)]
            }
            Stmt::LetMatch { var, ty, .. } => vec![(*var, *ty)],
            Stmt::LetTuple {
                vars,
                ty: Ty::Tuple(t),
                ..
            } => vars
                .iter()
                .zip(t.elems())
                .filter(|(_, elem)| !matches!(elem, Elem::Struct(_)))
                .map(|(&(var, _), elem)| (var, elem.ty()))
                .collect(),
            _ => Vec::new(),
        })
        .collect();
    // ... and one element of up to two tuple locals (the value space).
    let tuple_locals: Vec<(u32, TupleTy)> = body
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::Let {
                var,
                ty: Ty::Tuple(t),
                ..
            } => Some((*var, *t)),
            _ => None,
        })
        .collect();
    // ... and a scalar field of up to two struct locals (the method space), so that the state the calls left is live.
    if g.methods {
        let structs: Vec<(u32, u8)> = body
            .iter()
            .filter_map(|stmt| match stmt {
                Stmt::Let {
                    var,
                    ty: Ty::Struct(j),
                    ..
                } => Some((*var, *j)),
                _ => None,
            })
            .collect();
        let mut folded = Vec::new();
        for _ in 0..2 {
            if structs.is_empty() || !g.b.chance(85) {
                continue;
            }
            let (var, j) = structs[g.b.below(structs.len() as u32) as usize];
            let fields = g.table.structs[usize::from(j)].fields.clone();
            let k = g.b.below(fields.len() as u32) as u8;
            // A field is folded once: `x ^ x` is zero and would hide the state.
            if fields[usize::from(k)].is_struct() || folded.contains(&(var, k)) {
                continue;
            }
            folded.push((var, k));
            let place = Place::var(var, Ty::Struct(j)).field(k);
            let read = Expr::Read {
                place,
                ty: fields[usize::from(k)],
            };
            result = fold_in(result, read);
        }
    }
    if let Some(call) = arm_call {
        result = fold_in(result, call);
    }
    if g.values {
        for _ in 0..2 {
            if tuple_locals.is_empty() || !g.b.chance(85) {
                continue;
            }
            let (var, t) = tuple_locals[g.b.below(tuple_locals.len() as u32) as usize];
            let scalars: Vec<u8> = (0..t.n)
                .filter(|&k| !matches!(t.elems()[usize::from(k)], Elem::Struct(_)))
                .collect();
            if scalars.is_empty() {
                continue;
            }
            let k = scalars[g.b.below(scalars.len() as u32) as usize];
            let elem_ty = t.elems()[usize::from(k)].ty();
            let place = Place::var(var, Ty::Tuple(t)).field(k);
            let read = Expr::Read { place, ty: elem_ty };
            let read = match elem_ty {
                Ty::Enum(e) => Expr::EnumAs {
                    to: g.table.enums[usize::from(e)].repr,
                    e: Box::new(read),
                },
                _ => read,
            };
            result = fold_in(result, read);
        }
    }
    // The field that a top-level `t.k.f op= t.k.m()` set (fix round 1): read in the result, so that the order of the
    // place read and the call decides the output. A field is folded once.
    if g.values {
        let mut folded: Vec<Place> = Vec::new();
        for stmt in &body {
            if let Stmt::Set {
                place,
                ty,
                op: Some(op),
                value,
            } = stmt
                && crate::ast::tuple_set_names(&g.table, place, Some(*op), value)
                    .contains(&crate::ast::SHAPE_TUPLE_SET_CALL)
                && tuple_locals
                    .iter()
                    .any(|(var, _)| place.root == Root::Var(*var))
                && !folded.contains(place)
            {
                folded.push(place.clone());
                let read = Expr::Read {
                    place: place.clone(),
                    ty: *ty,
                };
                result = fold_in(result, read);
            }
        }
    }
    for read in array_folds {
        result = fold_in(result, read);
    }
    // ... and often a call on one struct root, so that the final store has a `&mut self` call in its value.
    if g.arrays
        && matches!(result.ty(), Ty::U32 | Ty::I32)
        && g.b.chance(50)
        && let Some(root) = g.bias_root()
        && let Some(call) = g.root_call(result.ty(), root, 3)
    {
        result = fold_in(result, call);
    }
    if g.closures && matches!(result.ty(), Ty::U32 | Ty::I32) {
        for probe in g.closure_probe_calls() {
            result = fold_in(result, probe);
        }
    }
    // ... and, in the closure space, a call of a closure whose arguments call it: the store's value then holds the call.
    if g.closures
        && matches!(result.ty(), Ty::U32 | Ty::I32)
        && g.b.chance(45)
        && let Some(root) = g.bias_root()
        && let Some(call) = g.closure_call(result.ty(), 3, Some(root))
    {
        result = fold_in(result, call);
    }
    g.store = g.array_store(&result);
    let (folds, fold_chance) = if g.vectors { (3, 85) } else { (2, 60) };
    for _ in 0..folds {
        if live.is_empty() || !g.b.chance(fold_chance) {
            continue;
        }
        let (var, var_ty) = live[g.b.below(live.len() as u32) as usize];
        let local = match var_ty {
            // A vector local joins the result through a component or (`f32`) its squared length; a matrix through a
            // component of its product with a fresh vector.
            Ty::Vec(lane, n) => {
                let vector = Expr::Var(var, var_ty);
                if lane == Lane::F32 && g.b.chance(35) {
                    Expr::dot(vector.clone(), vector)
                } else {
                    Expr::comp(vector, g.b.below(u32::from(n)) as u8)
                }
            }
            Ty::Mat(n) => {
                let column = g.build(Ty::Vec(Lane::F32, n), 1);
                let product = Expr::agg_bin(BinOp::Mul, Expr::Var(var, var_ty), column);
                Expr::comp(product, g.b.below(u32::from(n)) as u8)
            }
            // An enum local joins the result as its repr value.
            Ty::Enum(k) => Expr::EnumAs {
                to: g.table.enums[usize::from(k)].repr,
                e: Box::new(Expr::Var(var, var_ty)),
            },
            _ => Expr::Var(var, var_ty),
        };
        result = fold_in(result, local);
    }
    Program {
        seed,
        attempt,
        count,
        inputs,
        body,
        result,
        methods: g.table,
        store: g.store.take(),
    }
}

/// `result` combined with a local (`bit-exact` operators only: no new overflow, division or shift conditions).
pub(super) fn fold_in(result: Expr, local: Expr) -> Expr {
    let (ty, local_ty) = (result.ty(), local.ty());
    match (ty, local_ty) {
        // A float local joins a float result as one more term, if the chain stays within four written terms.
        (Ty::F32, Ty::F32) if chain_terms(&result) < 4 => Expr::bin(BinOp::Add, result, local),
        (Ty::F32, _) | (Ty::Bool, Ty::F32) => result,
        (Ty::Bool, Ty::Bool) => Expr::bin(BinOp::Or, result, local),
        (Ty::Bool, _) => Expr::bin(
            BinOp::And,
            result,
            Expr::bin(BinOp::Ne, local, Expr::Lit(local_ty, 0)),
        ),
        (_, Ty::Bool) => result,
        (_, _) if ty == local_ty => Expr::bin(BinOp::BitXor, result, local),
        _ => Expr::bin(BinOp::BitXor, result, Expr::cast(ty, local)),
    }
}

/// The program of `seed` at `attempt`, whether or not it is in the domain, in the space that the attempt number
/// selects (`space_of_attempt`).
pub fn generate_attempt(seed: u64, attempt: u64) -> Program {
    generate_attempt_in(space_of_attempt(attempt), seed, attempt)
}

/// The program of `seed` at `attempt` in `space`, whether or not it is in the domain.
pub fn generate_attempt_in(space: Space, seed: u64, attempt: u64) -> Program {
    decode_in(
        space,
        &byte_stream(seed, attempt, BYTES_PER_PROGRAM),
        seed,
        attempt,
    )
}
