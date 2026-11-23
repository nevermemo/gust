//! The interpreter: the independent host reference of the generator (D26). It evaluates the AST directly with
//! checked integer operations and never sees the emitted text.
//!
//! Besides the value it tracks the domain of D38 (docs/SEMANTICS.md section 3, items 1 to 3 and 6): a program run
//! that leaves the domain at any element, in any evaluated sub-expression, returns the first `Violation` and the
//! generator discards the program. Evaluation order and short-circuiting follow Rust, so a sub-expression Rust does
//! not evaluate (the right side of `&&` after `false`) cannot violate the domain.
//!
//! `f32` values (DIFFTEST_FORMAT 2) carry two things: the reference value, computed in `f32` in the written order
//! with round to nearest even (what the CPU twin computes), and an interval from `float.rs` that contains every value
//! a conforming WGSL implementation may hold (R1-4). A float construct whose result the interval cannot pin down is a
//! `Violation` too, and the program is discarded, never asserted.

use std::collections::{BTreeMap, HashMap};

use crate::agg::{self, Agg, Scalar};
use crate::ast::{
    ASSIGN_COMP, Arm, BinOp, CLOSURE_BODY_HELPER, CLOSURE_BODY_METHOD, CLOSURE_PARAM_METHOD,
    ClosureDef, Elem, Expr, MATCH_ARM_RETURN_RUN, MathFn, Methods, Pat, PathForm, Place, Program,
    Recv, Root, SHAPE_EXPR_CALL, SHAPE_STMT_CALL, STRUCT_LIT, Stmt, TRAIT_PATH_CALL,
    TRAIT_PATH_QUALIFIED, TUPLE_READ, Ty, UnOp, array_call_names, array_expr_names,
    array_set_names, array_store_names, bin_shape, call_arg_names, call_name, call_shape,
    closure_call_names, closure_let_names, closure_set_names, destruct_names, enum_as_name,
    enum_call_shapes, match_names, match_shape, set_name, set_shape, trait_call_names,
    tuple_field_names, tuple_lit_names, tuple_set_names, value_match_names,
};
use crate::float::{self, Fl};

mod calls;
mod eval;
mod fold;
mod machine;
mod ops;
mod run;
mod stmts;
#[cfg(test)]
mod tests;

pub use fold::*;
pub use machine::*;
pub use ops::*;
pub use run::*;

/// A way to leave the defined domain: the integer conditions of D38 and the float conditions of R1-4.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Violation {
    /// `+ - *` or unary `-` whose result does not fit the type.
    Overflow,
    /// `/` or `%` with a zero divisor.
    DivisionByZero,
    /// `i32::MIN / -1` or `i32::MIN % -1`.
    MinOverMinusOne,
    /// A shift amount of 32 or more.
    ShiftAmount,
    /// A buffer index not below `len()`.
    Index,
    /// A float interval outside `+-[2^-100, 2^100]` (or holding zero without being `[0, 0]`), a float divisor
    /// interval outside that range, or a float-to-integer cast whose interval leaves the target range.
    FloatDomain,
    /// A float comparison whose operand intervals are not disjoint.
    ComparisonOverlap,
    /// An output interval wider than `2^-10` relative, or a float-to-integer cast whose interval spans two integers.
    Unboundable,
    /// An integer of magnitude above `2^24` converted to `f32` (the two-neighbour set of 15.7.6 is v2 material).
    FloatConstruct,
}

impl Violation {
    pub const ALL: [Violation; 9] = [
        Violation::Overflow,
        Violation::DivisionByZero,
        Violation::MinOverMinusOne,
        Violation::ShiftAmount,
        Violation::Index,
        Violation::FloatDomain,
        Violation::ComparisonOverlap,
        Violation::Unboundable,
        Violation::FloatConstruct,
    ];

    /// The name used in discard counts and in `seeds.txt` (no spaces: a line is split on spaces).
    pub fn label(self) -> &'static str {
        match self {
            Violation::Overflow => "overflow",
            Violation::DivisionByZero => "division-by-zero",
            Violation::MinOverMinusOne => "min-over-minus-one",
            Violation::ShiftAmount => "shift-amount",
            Violation::Index => "index",
            Violation::FloatDomain => "float-domain",
            Violation::ComparisonOverlap => "comparison-overlap",
            Violation::Unboundable => "unboundable",
            Violation::FloatConstruct => "float-construct-not-in-v1",
        }
    }
}

/// A program run that left the domain: the first element that did, and how.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Failure {
    pub element: u32,
    pub violation: Violation,
}
