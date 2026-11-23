//! Running a program: elements, coverage and mutants.

use super::*;

/// Every closure of the program by its variable, with the function that binds it (`None`: the kernel).
fn closure_table(program: &Program) -> BTreeMap<u32, (&ClosureDef, Option<usize>)> {
    crate::ast::register_closures(program);
    fn collect<'a>(
        stmts: &'a [Stmt],
        site: Option<usize>,
        table: &mut BTreeMap<u32, (&'a ClosureDef, Option<usize>)>,
    ) {
        for stmt in stmts {
            match stmt {
                Stmt::LetClosure { def } => {
                    table.insert(def.var, (&**def, site));
                }
                Stmt::If { then, els, .. } => {
                    collect(then, site, table);
                    collect(els, site, table);
                }
                Stmt::For { body, .. } => collect(body, site, table),
                Stmt::Match { arms, .. }
                | Stmt::LetMatch { arms, .. }
                | Stmt::TailMatch { arms, .. } => {
                    for arm in arms {
                        collect(&arm.body, site, table);
                    }
                }
                _ => {}
            }
        }
    }
    let mut table = BTreeMap::new();
    collect(&program.body, None, &mut table);
    for (k, def) in program.methods.fns.iter().enumerate() {
        collect(&def.body, Some(k), &mut table);
    }
    table
}

fn variable_slots(program: &Program) -> usize {
    let mut max = 0;
    fn block(stmts: &[Stmt], max: &mut usize) {
        for stmt in stmts {
            match stmt {
                Stmt::Let { var, .. }
                | Stmt::LetMatch { var, .. }
                | Stmt::Assign { var, .. }
                | Stmt::AssignComp { var, .. }
                | Stmt::For { var, .. } => {
                    *max = (*max).max(*var as usize + 1);
                }
                Stmt::LetTuple { vars, .. } => {
                    for &(var, _) in vars {
                        *max = (*max).max(var as usize + 1);
                    }
                }
                Stmt::LetClosure { def } => {
                    for &(var, ..) in &def.params {
                        *max = (*max).max(var as usize + 1);
                    }
                    block(&def.body, max);
                }
                Stmt::If { .. }
                | Stmt::Set { .. }
                | Stmt::Call { .. }
                | Stmt::Match { .. }
                | Stmt::TailMatch { .. }
                | Stmt::Yield { .. }
                | Stmt::Return { .. } => {}
            }
            match stmt {
                Stmt::If { then, els, .. } => {
                    block(then, max);
                    block(els, max);
                }
                Stmt::For { body, .. } => block(body, max),
                Stmt::Match { arms, .. }
                | Stmt::LetMatch { arms, .. }
                | Stmt::TailMatch { arms, .. } => arms.iter().for_each(|arm| block(&arm.body, max)),
                _ => {}
            }
        }
    }
    block(&program.body, &mut max);
    for def in &program.methods.fns {
        block(&def.body, &mut max);
        for (var, _) in &def.params {
            max = max.max(*var as usize + 1);
        }
    }
    crate::ast::visit_exprs(program, &mut |e| {
        if let Expr::Var(var, _) = e {
            max = max.max(*var as usize + 1);
        }
    });
    max
}

/// The outcome of a run that stayed inside the domain.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    /// The output buffer's bit patterns (`bool` as 0 or 1; an `f32` as the reference value's bits).
    pub values: Vec<u32>,
    /// For an `f32` output, the interval `[lo, hi]` of every element that a conforming implementation may hold;
    /// empty otherwise.
    pub intervals: Vec<(f64, f64)>,
}

/// Runs the kernel body for every element `0..count` and returns the output buffer and, for an `f32` output, the
/// interval of every element, or the first domain violation (lowest element, first violating evaluation in Rust's
/// order). An `f32` output element whose interval is wider than `2^-10` of its magnitude is `Unboundable`.
pub fn run_full(program: &Program) -> Result<Run, Failure> {
    run_machine(program, false, None).map(|(run, _)| run)
}

/// What a run executed, summed over all elements of the program: the evaluations of each operator (the names of
/// `ast::operator_counts`, so a syntactic count and an executed count compare), the `if` statements by the sides they
/// took, and the `&&`/`||` evaluations by whether the right operand ran. Dead code (a branch no element takes, the right
/// side of an `&&` whose left side is always false) shows as a syntactic operator with no evaluations.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Coverage {
    pub operators: BTreeMap<String, u64>,
    /// `if` statements in the program (each counted once; evaluated or not).
    pub if_statements: u64,
    /// `if` statements some element evaluated.
    pub if_evaluated: u64,
    /// ... that took the then branch for some element, and the else branch for some element.
    pub if_then_taken: u64,
    pub if_else_taken: u64,
    /// ... that took both sides over the elements of the program.
    pub if_both_sides: u64,
    pub rhs_evaluated: u64,
    pub rhs_skipped: u64,
    /// The (generic function, type bound to its first type parameter) pairs some call executed.
    pub instantiations: std::collections::BTreeSet<(usize, String)>,
}

impl Coverage {
    /// Whether some generic function ran with two different types bound to its first type parameter.
    pub fn two_instantiations(&self) -> bool {
        self.instantiations
            .iter()
            .any(|(f, t)| self.instantiations.iter().any(|(g, u)| g == f && u != t))
    }
}

/// `run_full` with a deliberately wrong `match` evaluation (the mutation checks of slice m9.d1b).
pub fn run_mutant(program: &Program, mutant: Mutant) -> Result<Run, Failure> {
    run_machine(program, false, Some(mutant)).map(|(run, _)| run)
}

/// `run_full` that also counts what was executed.
pub fn run_covered(program: &Program) -> Result<(Run, Coverage), Failure> {
    run_machine(program, true, None).map(|(run, coverage)| (run, coverage.unwrap_or_default()))
}

fn count_ifs(stmts: &[Stmt]) -> u64 {
    stmts
        .iter()
        .map(|stmt| match stmt {
            Stmt::If { then, els, .. } => 1 + count_ifs(then) + count_ifs(els),
            Stmt::For { body, .. } => count_ifs(body),
            Stmt::LetClosure { def } => count_ifs(&def.body),
            Stmt::Match { arms, .. }
            | Stmt::LetMatch { arms, .. }
            | Stmt::TailMatch { arms, .. } => arms.iter().map(|arm| count_ifs(&arm.body)).sum(),
            _ => 0,
        })
        .sum()
}

/// What one element gives: the output bits, the float interval if any, and the output slot of an array store.
type ElementRun = (u32, Option<(f64, f64)>, Option<usize>);

/// One element: the body, the result, and for an array program the slot of the final store. Rust's order for
/// `out[idx] op= result` on a primitive is the result first, then the index; the mutant `index-before-value` evaluates
/// the index first. The slot must be the element's own (the generator keeps it so; the GPU writes of different
/// invocations must not overlap), or the run is outside the checked domain.
fn run_element(machine: &mut Machine, program: &Program) -> Result<ElementRun, Violation> {
    machine.block(&program.body)?;
    let early = match &program.store {
        Some(store) if machine.mutant == Some(Mutant::IndexBeforeValue) => {
            Some(machine.expr(&store.index)?.int())
        }
        _ => None,
    };
    let value = machine.expr(&program.result)?;
    let (bits, interval) = match value {
        Val::I(bits) => (bits, None),
        Val::F(reference, fl) => {
            let (lo, hi) = fl.interval();
            if hi - lo > float::MAX_RELATIVE_WIDTH * lo.abs().max(hi.abs()) {
                return Err(Violation::Unboundable);
            }
            (reference.to_bits(), Some((lo, hi)))
        }
        Val::A(_) => unreachable!("Program::check rejects an aggregate result"),
    };
    let slot = match &program.store {
        Some(store) => {
            if machine.coverage.is_some() {
                for name in array_store_names(&program.methods, &program.result) {
                    machine.count_name(name);
                }
            }
            let idx = match early {
                Some(idx) => idx,
                None => machine.expr(&store.index)?.int(),
            };
            if idx != machine.element {
                return Err(Violation::Index);
            }
            Some(idx as usize)
        }
        None => None,
    };
    Ok((bits, interval, slot))
}

fn run_machine(
    program: &Program,
    covered: bool,
    mutant: Option<Mutant>,
) -> Result<(Run, Option<Coverage>), Failure> {
    let mut machine = Machine {
        program,
        element: 0,
        vars: vec![Val::I(0); variable_slots(program)],
        self_stack: Vec::new(),
        coverage: covered.then(Box::default),
        observed: None,
        returned: None,
        for_depth: 0,
        arm_depth: 0,
        mutant,
        closures: closure_table(program),
        closure_ctx: None,
        memo: HashMap::new(),
        binds: Vec::new(),
        first_bind: HashMap::new(),
    };
    // An array program's output starts as the store's `init`, and element `i` stores into `out[idx]`.
    let init = program.store.as_ref().map_or(0, |store| store.init);
    let mut values = if program.store.is_some() {
        vec![init; program.count as usize]
    } else {
        Vec::with_capacity(program.count as usize)
    };
    let mut intervals = Vec::new();
    for element in 0..program.count {
        machine.element = element;
        machine.self_stack.clear();
        machine.memo.clear();
        machine.first_bind.clear();
        let (bits, interval, slot) = run_element(&mut machine, program)
            .map_err(|violation| Failure { element, violation })?;
        match slot {
            Some(slot) => {
                let stored = match program.store.as_ref().and_then(|store| store.op) {
                    None => bits,
                    Some(op) => binary(op, program.result.ty(), values[slot], bits)
                        .map_err(|violation| Failure { element, violation })?,
                };
                values[slot] = stored;
            }
            None => values.push(bits),
        }
        intervals.extend(interval);
    }
    let coverage = machine.coverage.map(|counting| {
        let mut coverage = Coverage {
            if_statements: count_ifs(&program.body)
                + program
                    .methods
                    .fns
                    .iter()
                    .map(|def| count_ifs(&def.body))
                    .sum::<u64>(),
            rhs_evaluated: counting.rhs_evaluated,
            rhs_skipped: counting.rhs_skipped,
            instantiations: counting.instantiations,
            ..Coverage::default()
        };
        for ((kind, head, tail), n) in counting.operators {
            coverage
                .operators
                .insert(operator_name(kind, head, tail), n);
        }
        coverage.operators.extend(counting.named);
        for (then_taken, else_taken) in counting.branches.values() {
            coverage.if_evaluated += 1;
            coverage.if_then_taken += u64::from(*then_taken > 0);
            coverage.if_else_taken += u64::from(*else_taken > 0);
            coverage.if_both_sides += u64::from(*then_taken > 0 && *else_taken > 0);
        }
        coverage
    });
    Ok((Run { values, intervals }, coverage))
}

/// The first value (the bit pattern; an `f32` as its reference value) that each expression node of `program` takes at
/// `element`, by the node's address, for the reducer's "replace a subexpression by what it evaluated to" pass. A node
/// that element never evaluates is absent. Runs that one element only; a violation ends the walk early (the nodes seen
/// until then stay).
pub fn observed_values(program: &Program, element: u32) -> HashMap<usize, u32> {
    let mut machine = Machine {
        program,
        element,
        vars: vec![Val::I(0); variable_slots(program)],
        self_stack: Vec::new(),
        coverage: None,
        observed: Some(HashMap::new()),
        returned: None,
        for_depth: 0,
        arm_depth: 0,
        mutant: None,
        closures: closure_table(program),
        closure_ctx: None,
        memo: HashMap::new(),
        binds: Vec::new(),
        first_bind: HashMap::new(),
    };
    let _ = machine
        .block(&program.body)
        .and_then(|()| machine.expr(&program.result));
    machine.observed.unwrap_or_default()
}

/// `run_full` without the intervals: the output buffer's bit patterns, or the first domain violation.
pub fn run(program: &Program) -> Result<Vec<u32>, Failure> {
    run_full(program).map(|run| run.values)
}
