//! The machine state, the mutants, and the operator counters.

use super::*;

/// A place resolved: its index evaluated, so that reading and writing it again does not evaluate the index again.
#[derive(Clone, Copy, Debug)]
pub(super) struct Loc {
    pub(super) root: Root,
    /// The offset of the part in the root's flat components.
    pub(super) offset: usize,
    pub(super) ty: Ty,
    /// The place is the whole root.
    pub(super) whole: bool,
}

pub(super) struct Machine<'a> {
    pub(super) program: &'a Program,
    pub(super) element: u32,
    pub(super) vars: Vec<Val>,
    /// The receivers of the calls in progress (a struct as its flat components), innermost last: what `self` is.
    pub(super) self_stack: Vec<Val>,
    /// Executed-operator counts, kept only by `run_covered`.
    pub(super) coverage: Option<Box<Counting>>,
    /// The first value of every expression node (by address), kept only by `observed_values`.
    pub(super) observed: Option<HashMap<usize, u32>>,
    /// Set by `return` in a helper: the value the function ends with; every enclosing block stops while it is set.
    pub(super) returned: Option<Val>,
    /// `for` bodies and `match` arms the running function is inside (reset per call), for the counted names.
    pub(super) for_depth: u32,
    pub(super) arm_depth: u32,
    /// Every closure of the program by variable (ids are program-wide unique), with the function that binds it.
    pub(super) closures: BTreeMap<u32, (&'a ClosureDef, Option<usize>)>,
    /// The closure whose body is running, and whether it is bound in a method (for the counted names).
    pub(super) closure_ctx: Option<(&'a ClosureDef, bool)>,
    /// The result of the first call of each closure in this element, for the `closure-memo` mutant.
    pub(super) memo: HashMap<u32, Val>,
    /// The bindings of the type parameters of the generic functions in progress, innermost last (format 13).
    pub(super) binds: Vec<[Option<Ty>; 2]>,
    /// The first binding of each generic function in this element, for the `generic-specialisation-reuse` mutant.
    pub(super) first_bind: HashMap<usize, [Option<Ty>; 2]>,
    /// Test hook for the mutation checks (slice m9.d1b); `None` is the Rust semantics.
    pub(super) mutant: Option<Mutant>,
}

/// The mutant that the test run asks for through `DIFFTEST_MUTANT` (`default-first-arm` or `scrutinee-twice`). Only the
/// hand-vector tests apply it (`enum_tests::value_of`); `run` stays the reference, so an `xfail-` run fails exactly the
/// vector that catches the mutant. Not compiled outside the tests.
#[cfg(test)]
pub(crate) fn env_mutant() -> Option<Mutant> {
    match std::env::var("DIFFTEST_MUTANT").as_deref() {
        Ok("default-first-arm") => Some(Mutant::DefaultTakesFirstArm),
        Ok("scrutinee-twice") => Some(Mutant::ScrutineeTwice),
        Ok("tuple-right-to-left") => Some(Mutant::TupleRightToLeft),
        Ok("value-scrutinee-twice") => Some(Mutant::ValueScrutineeTwice),
        Ok("group-first-only") => Some(Mutant::GroupFirstOnly),
        Ok("compound-place-first") => Some(Mutant::CompoundPlaceFirst),
        Ok("array-lit-right-to-left") => Some(Mutant::ArrayLitRightToLeft),
        Ok("repeat-per-element") => Some(Mutant::RepeatPerElement),
        Ok("index-before-value") => Some(Mutant::IndexBeforeValue),
        Ok("array-arg-by-ref") => Some(Mutant::ArrayArgByRef),
        Ok("closure-args-right-to-left") => Some(Mutant::ClosureArgsRightToLeft),
        Ok("closure-param-by-ref") => Some(Mutant::ClosureParamByRef),
        Ok("closure-memo") => Some(Mutant::ClosureMemo),
        Ok("trait-wrong-impl") => Some(Mutant::TraitWrongImpl),
        Ok("generic-specialisation-reuse") => Some(Mutant::GenericReuse),
        Ok("trait-args-right-to-left") => Some(Mutant::TraitArgsRightToLeft),
        _ => None,
    }
}

/// A deliberately wrong `match` evaluation, used only by the `xfail-` mutation tests to show the vectors catch it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutant {
    /// In a `match` without `_`, the variants of the last arm (the Slang `default`) go to the first arm instead (a
    /// mis-applied `default` rule).
    DefaultTakesFirstArm,
    /// The scrutinee is evaluated again for every arm tried (read twice).
    ScrutineeTwice,
    /// The elements of a tuple literal run right to left (their values are still placed left to right). (m9.d1c)
    TupleRightToLeft,
    /// The scrutinee of a value `match` is evaluated again for every arm after the first. (m9.d1c)
    ValueScrutineeTwice,
    /// An integer pattern group (`1 | 2 | 3`) matches only its first literal. (m9.d1c)
    GroupFirstOnly,
    /// `place op= value` reads the place before it runs the value (the order of an operator-overloaded `AddAssign`, and
    /// of Slang for an indexed place), not after. (m9.d1c fix round 1)
    CompoundPlaceFirst,
    /// The elements of an array literal run right to left (their values are still placed left to right). (m9.d1d)
    ArrayLitRightToLeft,
    /// A repeat `[e; N]` evaluates `e` once per element instead of once. (m9.d1d)
    RepeatPerElement,
    /// `place = value` and `place op= value` locate the place (evaluate its index) before the value, and the output
    /// store evaluates its index before the result. (m9.d1d)
    IndexBeforeValue,
    /// An array argument passed by reference: what the callee leaves in the parameter is written back to the caller's
    /// variable. (m9.d1d)
    ArrayArgByRef,
    /// The arguments of a closure call run right to left (their values are still placed left to right). (m9.d1e)
    ClosureArgsRightToLeft,
    /// An array or struct argument of a closure call that is a variable is the closure's parameter itself: what the body
    /// leaves in it is written back to the caller's variable. (m9.d1e)
    ClosureParamByRef,
    /// A closure body is evaluated by the first call of the element only; later calls reuse the result. (m9.d1e)
    ClosureMemo,
    /// A call of a trait method runs the impl of the other implementing type. (m9.d2)
    TraitWrongImpl,
    /// A generic function runs every call with the type parameters of its first call. (m9.d2)
    GenericReuse,
    /// The arguments of a trait or generic call run right to left (their values are still placed left to right). (m9.d2)
    TraitArgsRightToLeft,
}

/// What `run_covered` counts while it runs.
#[derive(Default)]
pub(super) struct Counting {
    /// (kind, head, tail) -> evaluations; see `operator_name`.
    pub(super) operators: BTreeMap<(u8, &'static str, &'static str), u64>,
    /// Vector and matrix operators and component assignments by their counted name (`Expr::aggregate_name`).
    pub(super) named: BTreeMap<String, u64>,
    /// `if` statement (by address) -> (evaluations that took the then branch, evaluations that took the else branch).
    pub(super) branches: HashMap<usize, (u64, u64)>,
    /// `&&` and `||` evaluations whose right operand was evaluated, and whose right operand was skipped.
    pub(super) rhs_evaluated: u64,
    pub(super) rhs_skipped: u64,
    /// (generic function, type bound to its first type parameter) of every generic call that ran.
    pub(super) instantiations: std::collections::BTreeSet<(usize, String)>,
}

pub(super) const KIND_UNARY: u8 = 0;
pub(super) const KIND_BINARY: u8 = 1;
pub(super) const KIND_CAST: u8 = 2;
pub(super) const KIND_FOR: u8 = 3;

/// The name of a counted operator, the same spelling as `ast::operator_counts` (`+ u32`, `-f32`, `!bool`, `as f32`, `for`).
pub(super) fn operator_name(kind: u8, head: &str, tail: &str) -> String {
    match kind {
        KIND_UNARY if head == "!" => format!("!{tail}"),
        KIND_UNARY => head.to_owned(),
        KIND_BINARY => format!("{head} {tail}"),
        KIND_CAST => format!("as {tail}"),
        _ => "for".to_owned(),
    }
}

impl Machine<'_> {
    pub(super) fn count(&mut self, kind: u8, head: &'static str, tail: &'static str) {
        if let Some(counting) = &mut self.coverage {
            *counting.operators.entry((kind, head, tail)).or_default() += 1;
        }
    }

    /// Counts a vector or matrix operator by its name (only `run_covered` pays for the string).
    pub(super) fn count_aggregate(&mut self, expr: &Expr) {
        if let Some(counting) = &mut self.coverage
            && let Some(name) = expr.aggregate_name()
        {
            *counting.named.entry(name).or_default() += 1;
        }
    }
}
